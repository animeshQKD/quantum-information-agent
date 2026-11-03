"""
app.py - the agent in your web browser (formatted like a chat app).
    python app.py        then open the address it prints (usually http://127.0.0.1:7860)
Answers render with real tables and bold text, figures appear inline, and the tool steps
are in a collapsible box above each answer. Every browser tab is its own session (session.py):
its own states, uploads and figures, deleted after SESSION_TTL_S seconds without a question.
The same file runs the public Hugging Face Space (see deploy_space.py); keys come from Space secrets there.
Limits per visitor: MAX_QUESTION_CHARS characters per question, QUESTIONS_PER_HOUR questions per hour.
"""
import logging
import os
import shutil
import time
import uuid

import gradio as gr
from langgraph.errors import GraphRecursionError

import agent_tools
import capabilities
import graph
import llm
import plots
import routing
import session
import workbench_ui
from agent_tools import TOOLS
from display import result_panels, summarize_steps, to_unicode
from graph import build_graph
from llm import AllProvidersFailed, get_llm

EXPECTED = "2026-11-02"
stale = [m.__name__ + ".py" for m in (graph, agent_tools, llm, plots, routing)
         if getattr(m, "FILES_VERSION", None) != EXPECTED]
if stale:
    raise SystemExit(f"Out-of-date file(s): {', '.join(stale)}. Replace them with the latest versions.")

SAMPLE_CSV = "data/state_tomo_data.csv"
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
QUESTIONS_PER_HOUR = int(os.getenv("QUESTIONS_PER_HOUR", "40" if session.ON_SPACE else "0"))   # 0 = no limit
NO_MODEL = ("**The language models are not available right now.** The free daily limits may be used up, "
            "or the providers are busy. Please try again later. The **Workbench** tab works without a "
            "language model, so every calculation is still available there.")
MAX_UPLOAD_BYTES = agent_tools.MAX_CSV_BYTES
WELCOME = ("**Welcome.** I compute properties of photonic polarization states — single photons and "
           "polarization-entangled photon pairs — with tested code: states and their density matrices, "
           "entanglement measures, local and global noise, time evolution, wave plates, tomography of your "
           "own counts, certification with error bars, protocols and plots.\n\n"
           "Type a question, pick an example from **What can this agent do?** below, or ask *\"What can "
           "you do?\"*. The **Workbench** tab runs the same tools from panels, without the language model.")


# Kets: the default font has no ⟨ ⟩, and the browser's fallback (e.g. Segoe UI Symbol on Windows) draws
# them like ( ), so |ψ⁻⟩⟨ψ⁻| looked like |ψ⁻)(ψ⁻|. Only these two characters come from a math font.
CSS = """
@font-face { font-family: "QIKets";
  src: local("Cambria Math"), local("Cambria"), local("STIX Two Math"), local("STIXGeneral"),
       local("DejaVu Sans"), local("Noto Sans Math");
  unicode-range: U+27E8-27E9; }
.gradio-container { --font: "QIKets", 'Source Sans Pro', ui-sans-serif, system-ui, sans-serif; }
"""


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def new_session_id() -> str:
    return uuid.uuid4().hex


def prepare_upload(file, sid):
    """Copy an uploaded CSV into THIS user's data folder and pre-fill the question box."""
    src = file if isinstance(file, str) else file.name
    if not src.lower().endswith(".csv"):
        return "Please upload a .csv file."
    if os.path.getsize(src) > MAX_UPLOAD_BYTES:
        return "That file is larger than 1 MB, too large for a two-qubit tomography."
    with session.use(sid) as s:
        os.makedirs(s.data_dir, exist_ok=True)
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(src))
        shutil.copy(src, f"{s.data_dir}/{name}")
    return (f"Reconstruct the density matrix from {session.UPLOADS}/{name} and report its entanglement "
            "with error bars, then plot the density matrix.")


def too_much(message: str, s) -> str | None:
    """A reason to refuse this question (too long, too many questions this hour), or None."""
    if len(message) > MAX_QUESTION_CHARS:
        return (f"**Question too long** ({len(message)} characters; the limit is {MAX_QUESTION_CHARS}). "
                "Please shorten it or ask one task at a time.")
    if QUESTIONS_PER_HOUR:
        now = time.time()
        s.asked[:] = [t for t in s.asked if now - t < 3600]
        if len(s.asked) >= QUESTIONS_PER_HOUR:
            wait = int(3600 - (now - s.asked[0])) // 60 + 1
            return (f"**Limit reached:** {QUESTIONS_PER_HOUR} questions per hour in this public demo. "
                    f"Please try again in about {wait} min. The **Workbench** tab has no limit.")
        s.asked.append(now)
    return None


def capabilities_panel(box):
    """One block per area: what it computes, and example questions that fill the question box."""
    with gr.Accordion("What can this agent do?", open=False):
        for a in capabilities.AREAS:
            text = f"**{a.title}.** {a.summary}\n\n" + "\n".join(f"- {c}" for c in a.computes)
            text += "".join(f"\n\n*{n}*" for n in a.notes)
            gr.Markdown(text)
            gr.Examples(list(a.examples), inputs=box, label=f"Try ({a.title.lower()})")
        gr.Markdown("**Concept questions.** " + capabilities.CONCEPTS + "\n\n**Limits**\n\n"
                    + "\n".join(f"- {x}" for x in capabilities.LIMITS))


def build_demo(model):
    agent = build_graph(model)

    def forget_chat(sid):                            # when a session expires, drop its chat memory too
        try:
            agent.checkpointer.delete_thread(sid)
        except Exception:  # noqa: BLE001 - older LangGraph versions: memory is freed with the process
            pass
    session.ON_DROP.append(forget_chat)

    def respond(message, history, sid):
        message = (message or "").strip()
        if not message:
            return "", history, sid
        history = history + [{"role": "user", "content": message}]
        config = {"configurable": {"thread_id": sid}, "recursion_limit": 60}
        with session.use(sid) as s:
            refusal = too_much(message, s)
            if refusal:
                history.append({"role": "assistant", "content": refusal})
                return "", history, sid
            s.last_provider = None
            before = agent.get_state(config).values or {}
            seen, old_figs = len(before.get("messages", [])), len(before.get("figures", []))
            t0 = time.time()
            try:
                out = agent.invoke({"messages": [("user", message)], "retries": 0}, config)
            except AllProvidersFailed as e:
                logging.getLogger(__name__).warning("no model available: %s", e)
                history.append({"role": "assistant", "content": NO_MODEL if session.ON_SPACE else
                                f"{NO_MODEL}\n\n```\n{e}\n```"})
                return "", history, sid
            except GraphRecursionError:
                history.append({"role": "assistant", "content":
                                "**Stopped:** the agent went round in circles. Please ask again, one task at a time."})
                return "", history, sid
            except Exception as e:  # noqa: BLE001 - a bug must not break the user's session
                logging.getLogger(__name__).exception("question failed")
                history.append({"role": "assistant", "content":
                                f"**Internal error** ({type(e).__name__}). Please try again or rephrase the question."})
                return "", history, sid
            who = ("no model needed · " if getattr(out["messages"][-1], "name", None) == graph.HELP_REPLY
                   else f"{s.last_provider} · " if s.last_provider else "")

        steps = summarize_steps(out["messages"][seen + 1:])
        if steps:
            lines = [("✓ " if kind == "ok" else "⚠ ") + to_unicode(x) for kind, x in steps]
            history.append({"role": "assistant", "content": "\n\n".join(lines),
                            "metadata": {"title": f"Steps ({len(steps)})", "status": "done"}})
        for _title, table in result_panels(out["messages"][seen + 1:]):
            history.append({"role": "assistant", "content": table})
        answer = to_unicode(_text(out["messages"][-1]))
        history.append({"role": "assistant", "content": answer + f"\n\n<sub>{who}{time.time() - t0:.0f} s</sub>"})
        for path in out.get("figures", [])[old_figs:]:
            history.append({"role": "assistant", "content": gr.Image(value=path, show_label=False)})
        return "", history, sid

    def new_chat(sid):
        session.drop(sid)                            # deletes this user's states, uploads and figures
        return [{"role": "assistant", "content": WELCOME}], new_session_id(), ""

    with gr.Blocks(title="Quantum Information Agent") as demo:
        gr.Markdown("## Quantum Information Agent\n"
                    "Single photons and photon pairs: states, density matrices, noise, entanglement measures, "
                    "tomography and plots. "
                    "Ask in plain English; every number comes from tested code.")
        sid = gr.State(new_session_id)
        with gr.Tabs() as tabs:
            with gr.Tab("Chat", id="chat"):
                chat = gr.Chatbot(value=[{"role": "assistant", "content": WELCOME}], height=620,
                                  show_label=False, buttons=["copy"])
                box = gr.Textbox(placeholder="Ask a question and press Enter…", show_label=False, autofocus=True)
                with gr.Row():
                    upload = gr.UploadButton("Upload coincidence counts (.csv) for tomography",
                                             file_types=[".csv"], size="sm")
                    gr.DownloadButton("Download a sample counts file", value=SAMPLE_CSV, size="sm")
                    clear = gr.Button("New chat", size="sm")
                capabilities_panel(box)
            workbench_ui.build(sid, box, tabs)
        box.submit(respond, [box, chat, sid], [box, chat, sid])
        upload.upload(prepare_upload, [upload, sid], box)
        clear.click(new_chat, sid, [chat, sid, box])
    return demo


class NoModel:
    """Stands in for the language model when none is configured: every chat question gets NO_MODEL."""

    def __init__(self, why: str):
        self.why = why

    def invoke(self, messages):
        raise AllProvidersFailed(self.why)


def remove_orphan_folders():
    """Sessions live in memory, so session folders left by a previous run belong to nobody."""
    for root in ("figures/sessions", "data/sessions"):
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    remove_orphan_folders()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        model = get_llm(TOOLS)
    except RuntimeError as e:                 # no key set (e.g. Space secrets missing): Workbench still works
        logging.getLogger(__name__).error("%s", e)
        model = NoModel(str(e))
    demo = build_demo(model)
    demo.queue(max_size=int(os.getenv("QUEUE_SIZE", "40")))     # extra visitors wait in line, then are told to retry
    demo.launch(css=CSS)
