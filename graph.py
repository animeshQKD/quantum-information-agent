"""
graph.py - first working LangGraph agent (Phase 2, simplified).

Graph:  START -> agent --(tool calls?)--> tools -> verifier -> agent ... -> END
                       `--(no tool calls)------------------------------> END

The LLM decides which tools to call; tools compute with tested code;
the verifier checks every result against physical limits before the LLM sees it again.
"""

from __future__ import annotations

FILES_VERSION = "2026-11-02"   # must match in graph.py, agent_tools.py, llm.py, run.py

import ast
import json
import logging
import math
import os
import re
from typing import Annotated, TypedDict

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import routing
from agent_tools import CURRENT_QUESTION, DESC, STORE, TOOLS


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # chat history
    stages: list[int]         # e.g. [3, 6]; empty = conceptual question (used later by a router)
    plan: list[dict]          # used later by a planner node
    objects: dict[str, str]   # handle -> short description
    results: list[dict]       # outputs of compute_measures
    figures: list[str]        # PNG paths
    checks: list[dict]        # verifier findings: {"ok": bool, "reason": str}
    retries: int


SYSTEM_PROMPT = routing.system_prompt()        # everything (the model gets only what a question needs)

MAX_RETRIES = 3
MAX_MODEL_CALLS = 10   # per question: stop looping (and spending free-tier quota) after this
BUDGET_STOP = "budget_stop"
HELP_REPLY = "capabilities"          # the fixed answer to "what can you do?" (no model call)
_HELP_ONLY = re.compile(r"^\s*((hi|hello|hey)[\s,!.]*)?(what\s+(else\s+)?can\s+(you|i|this\s+agent)\s+do(\s+for\s+me|\s+here)?"
                        r"|what\s+are\s+your\s+(capabilities|features)|(show|list|tell)(\s+me)?\s+(about\s+)?(your\s+|the\s+)?"
                        r"(capabilities|features|tools)|help|what\s+do\s+you\s+do|how\s+(can|do)\s+i\s+use\s+(you|this|this\s+agent|it))"
                        r"\s*[?.!]*\s*$", re.I)


def _parse(content):
    if isinstance(content, dict):
        return content
    for loader in (json.loads, ast.literal_eval):
        try:
            out = loader(content)
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    return None


def check_measures(m: dict) -> list[str]:
    """Physical-range and consistency checks for one compute_measures result."""
    tol = 1e-6
    problems = []
    if m.get("photons") == 1:                       # one photon: Stokes vector inside the Poincaré sphere
        if not (0.5 - tol <= m["purity"] <= 1 + tol):
            problems.append(f"purity {m['purity']} outside [1/2, 1]")
        if m["degree_of_polarization"] > 1 + tol:
            problems.append(f"degree of polarization {m['degree_of_polarization']} exceeds 1")
        return problems
    if not (0.25 - tol <= m["purity"] <= 1 + tol):
        problems.append(f"purity {m['purity']} outside [1/4, 1]")
    if not (-tol <= m["concurrence"] <= 1 + tol):
        problems.append(f"concurrence {m['concurrence']} outside [0, 1]")
    if not (-tol <= m["negativity"] <= 0.5 + tol):
        problems.append(f"negativity {m['negativity']} outside [0, 1/2]")
    if m["chsh_max"] > 2 * math.sqrt(2) + tol:
        problems.append(f"CHSH {m['chsh_max']} exceeds Tsirelson bound 2*sqrt(2)")
    if (m["concurrence"] > 1e-6) != (m["negativity"] > 1e-6):
        problems.append("concurrence and negativity disagree on whether the state is entangled")
    return problems


# ------------------------------------------------------------------ nodes
def _is_user_question(m) -> bool:
    return isinstance(m, HumanMessage) and not str(m.content).startswith("VERIFIER:")


def build_context(messages: list, groups: list | None = None) -> list:
    """What the LLM sees: a compact summary of earlier turns + the CURRENT turn only.

    Sending the whole chat history every call quickly exceeds free-tier token
    limits (Groq: tokens per minute). Earlier turns are replaced by
      - the list of states created so far (handles + descriptions), and
      - the previous final answer, shortened.
    """
    start = max((i for i, m in enumerate(messages) if _is_user_question(m)), default=0)
    prev_answer = next((m.content for m in reversed(messages[:start])
                        if isinstance(m, AIMessage) and not m.tool_calls and m.content), "")
    known = "\n".join(f"- {h}: {DESC.get(h, '')}" for h in STORE) or "- none yet"
    system = routing.system_prompt(groups) + f"\n\nStates created so far (most recent last):\n{known}"
    if prev_answer:
        system += f"\n\nYour previous answer (shortened):\n{str(prev_answer)[:600]}"

    turn = []
    for m in messages[start:]:
        if isinstance(m, AIMessage):  # drop provider-specific reasoning; keep text + tool calls
            m = AIMessage(content=m.content if isinstance(m.content, str) else "",
                          tool_calls=m.tool_calls, id=m.id)
        turn.append(m)
    return [SystemMessage(system)] + turn


def _turn_start(messages: list) -> int:
    return max((i for i, m in enumerate(messages) if _is_user_question(m)), default=0)


def _budget_answer(msgs: list, start: int, why: str = "budget") -> AIMessage:
    """Final answer when the model-call budget is used up or the model is looping on the same
    tool call (no LLM call: saves quota)."""
    turn = msgs[start:]
    draft = next((m for m in reversed(turn) if isinstance(m, AIMessage)
                  and not m.tool_calls and _text(m).strip()), None)
    calls = sum(isinstance(m, AIMessage) for m in turn)
    reason = (f"after {calls} model calls" if why == "budget" else
              f"after {calls} model calls because the model kept repeating tool calls it had already made")
    head = (f"[Stopped {reason}, without a fully checked answer. "
            "The steps above show what the tools computed.]")
    if draft is None:
        body = ("I could not finish this request. Try asking for one thing at a time "
                "(e.g. first create the state, then ask for the plot).")
        return AIMessage(content=body + "\n\n" + head, name=BUDGET_STOP)
    bad = ungrounded_numbers(_text(draft), msgs)
    warn = (f"\n\n[Warning: these numbers were not produced by any tool and may be wrong: "
            f"{', '.join(bad)}]") if bad else ""
    return AIMessage(content=_text(draft) + warn + "\n\n" + head, name=BUDGET_STOP)


REPEAT_NOTE = ("Already done in this question (same tool, same arguments) - the result is above. "
               "Do NOT call it again: use that result and write the final answer now.")


def make_tools_node():
    """Run the requested tools one after another (create_state then evolve_state in the same step
    must not race), and never repeat a call made earlier in the same question (a weak model can
    otherwise call the same tool ten times in a row)."""
    runner = ToolNode(TOOLS, handle_tool_errors=True)            # errors go back to the LLM as text

    def tools(state: AgentState):
        msgs = state["messages"]
        last = msgs[-1]
        start = _turn_start(msgs)
        ok_ids = {m.tool_call_id for m in msgs[start:-1] if isinstance(m, ToolMessage)
                  and getattr(m, "status", "") != "error" and not _text(m).startswith(("Error", REPEAT_NOTE[:20]))}
        seen = {(tc["name"], json.dumps(tc["args"], sort_keys=True))
                for m in msgs[start:-1] if isinstance(m, AIMessage)
                for tc in (m.tool_calls or []) if tc["id"] in ok_ids}
        fresh, repeated = [], []
        for tc in last.tool_calls:
            key = (tc["name"], json.dumps(tc["args"], sort_keys=True))
            if key in seen:
                repeated.append(ToolMessage(content=REPEAT_NOTE, tool_call_id=tc["id"], name=tc["name"]))
            else:
                seen.add(key)
                fresh.append(tc)
        fresh, merged = merge_noise_plots(fresh)
        out = []
        for tc in fresh:            # ONE AT A TIME, in order: a later call may use a handle that an
            one = AIMessage(content=last.content, tool_calls=[tc], id=last.id)   # earlier one creates
            out += runner.invoke({"messages": msgs[:-1] + [one]})["messages"]
        return {"messages": out + merged + repeated}
    return tools


MERGED_NOTE = "Merged with the previous call: all these states are drawn in ONE figure (see that result)."


def merge_noise_plots(calls: list) -> tuple[list, list]:
    """Several plot_measures_vs_noise calls in one step (one per Bell state, same channel and
    qubits) become ONE plot_sweep call, so the states are compared in one figure. Returns the
    calls to run and the ToolMessages answering the merged-away calls."""
    groups = {}
    for tc in calls:
        if tc["name"] == "plot_measures_vs_noise":
            a = tc["args"]
            groups.setdefault((a.get("channel"), a.get("qubits", "both")), []).append(tc)
    out, notes, done = [], [], set()
    for tc in calls:
        if tc["id"] in done:
            continue
        a = tc["args"]
        group = groups.get((a.get("channel"), a.get("qubits", "both")), []) if tc["name"] == "plot_measures_vs_noise" else []
        kinds = list(dict.fromkeys(str(g["args"].get("start_kind", "")) for g in group))
        if len(group) >= 2 and 2 <= len(kinds) <= 3:
            out.append({"name": "plot_sweep", "id": tc["id"], "type": "tool_call",
                        "args": {"sweep": "noise", "handles": kinds, "channel": a.get("channel"),
                                 "qubits": a.get("qubits", "both"),
                                 "quantities": ["concurrence", "negativity", "purity", "chsh_max"]}})
            for g in group[1:]:
                notes.append(ToolMessage(content=MERGED_NOTE, tool_call_id=g["id"], name="plot_measures_vs_noise"))
            done |= {g["id"] for g in group}
        else:
            out.append(tc)
    return _merge_sweeps(out, notes)


def _merge_sweeps(calls: list, notes: list) -> tuple[list, list]:
    """Several plot_sweep calls in one step that differ only in their states become ONE call with all
    the states (at most 3), so they are compared in one figure."""
    def key(a):
        return json.dumps({k: v for k, v in a.items() if k != "handles"}, sort_keys=True, default=str)
    groups = {}
    for tc in calls:
        if tc["name"] == "plot_sweep" and isinstance(tc["args"].get("handles"), list):
            groups.setdefault(key(tc["args"]), []).append(tc)
    out, done = [], set()
    for tc in calls:
        if tc["id"] in done:
            continue
        group = groups.get(key(tc["args"]), []) if tc["name"] == "plot_sweep" else []
        handles = list(dict.fromkeys(h for g in group for h in g["args"].get("handles") or []))
        if len(group) >= 2 and 2 <= len(handles) <= 3:
            out.append({**tc, "args": {**tc["args"], "handles": handles}})
            for g in group[1:]:
                notes.append(ToolMessage(content=MERGED_NOTE, tool_call_id=g["id"], name="plot_sweep"))
            done |= {g["id"] for g in group}
        else:
            out.append(tc)
    return out, notes


def only_repeats_responses(turn: list) -> int:
    """Model responses whose tool calls were ALL repeats of earlier ones (real looping). A response
    that repeats one call (e.g. create_state twice) but also makes new calls is progress."""
    repeated = {m.tool_call_id for m in turn
                if isinstance(m, ToolMessage) and _text(m).startswith(REPEAT_NOTE[:20])}
    return sum(1 for m in turn if isinstance(m, AIMessage) and m.tool_calls
               and all(tc["id"] in repeated for tc in m.tool_calls))


def make_agent_node(llm):
    def agent(state: AgentState):
        msgs = state["messages"]
        start = _turn_start(msgs)
        question = msgs[start] if msgs and _is_user_question(msgs[start]) else None
        CURRENT_QUESTION["text"] = _text(question) if question is not None else ""
        calls = sum(isinstance(m, AIMessage) for m in msgs[start:])
        if calls == 0 and question is not None and _HELP_ONLY.match(_text(question)):
            import capabilities                           # a fixed answer: free, instant, always right
            return {"messages": [AIMessage(content=capabilities.help_answer(), name=HELP_REPLY)]}
        if calls >= MAX_MODEL_CALLS:
            return {"messages": [_budget_answer(msgs, start, "budget")]}
        if only_repeats_responses(msgs[start:]) >= 2:     # looping on the same call: stop early
            return {"messages": [_budget_answer(msgs, start, "repeats")]}
        if not hasattr(llm, "for_tools") or os.getenv("ROUTE_TOOLS", "1") == "0":   # no routing
            return {"messages": [llm.invoke(build_context(msgs))]}
        names, groups = route(msgs, start)
        reply = llm.for_tools(names).invoke(build_context(msgs, groups))
        if (len(names) < len(TOOLS) and not getattr(reply, "tool_calls", None)
                and re.search(r"no tool|can.?t do that", _text(reply), re.I)):
            ROUTING["retried_with_all_tools"] += 1        # the routing missed a topic: try everything once
            reply = llm.invoke(build_context(msgs))
        return {"messages": [reply]}
    return agent


ROUTING = {"last_tools": [], "last_groups": [], "retried_with_all_tools": 0}


def route(msgs: list, start: int) -> tuple[list[str], list[str]]:
    """Tools and instruction groups for the current turn (see routing.py). Tools named by the user
    or by a VERIFIER note of this turn are always included."""
    question = _text(msgs[start]) if msgs and _is_user_question(msgs[start]) else ""
    prev_start = max((i for i, m in enumerate(msgs[:start]) if _is_user_question(m)), default=None)
    prev_q = _text(msgs[prev_start]) if prev_start is not None else ""
    prev_tools = [tc["name"] for m in msgs[prev_start or 0:start] if isinstance(m, AIMessage)
                  for tc in (m.tool_calls or [])] if prev_start is not None else []
    names, groups = routing.select(question, prev_tools, prev_q)
    this_turn = " ".join(_text(m) for m in msgs[start:] if _is_verifier(m)) + " " + question
    names += [t.name for t in TOOLS if re.search(rf"\b{t.name}\b", this_turn)]
    names += [tc["name"] for m in msgs[start:] if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    names = list(dict.fromkeys(n for n in names if any(t.name == n for t in TOOLS)))
    ROUTING["last_tools"], ROUTING["last_groups"] = names, groups
    return names, groups


def verifier(state: AgentState):
    """Look at the tool results produced since the last AI message (never crashes the program: on an
    internal error the results go on unchecked)."""
    try:
        return _verifier(state)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("verifier failed")
        return {}


def _verifier(state: AgentState):
    """Look at the tool results produced since the last AI message."""
    new_tool_msgs = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            new_tool_msgs.append(msg)
        else:
            break

    checks, results, figures = [], [], []
    for tm in reversed(new_tool_msgs):                 # in the order the tools ran
        if tm.name == "compute_measures":
            m = _parse(tm.content)
            if m is None:
                continue
            results.append(m)
            probs = check_measures(m)
            checks.append({"ok": not probs, "reason": "; ".join(probs) or "all checks passed"})
        elif isinstance(tm.content, str) and tm.content.endswith(".png"):
            figures.append(tm.content)
        else:
            d = _parse(tm.content)
            if d and isinstance(d.get("figure"), str):
                figures.append(d["figure"])
            if d and isinstance(d.get("elements_figure"), str):     # evolve_state(plot_elements=True)
                figures.append(d["elements_figure"])

    update = {
        "checks": checks,
        "results": state.get("results", []) + results,
        "figures": state.get("figures", []) + figures,
        "objects": {h: f"{STORE[h].shape[0]}x{STORE[h].shape[0]} density matrix" for h in STORE},
    }
    failed = [c for c in checks if not c["ok"]]
    if failed:
        retries = state.get("retries", 0) + 1
        update["retries"] = retries
        note = "VERIFIER: " + " | ".join(c["reason"] for c in failed)
        if retries >= MAX_RETRIES:
            note += " Max retries reached: report the results with this warning."
        update["messages"] = [HumanMessage(note)]
    return update


def route_after_agent(state: AgentState):
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "answer_check"


# ------------------------------------------------------------------ answer check
_NUM = re.compile(r"(?<![\w.])-?\d+\.\d+")          # decimals only, e.g. 0.245, 2.2272
# Constants an answer may legitimately quote (thresholds and bounds from the reference facts)
_CONSTANTS = [0.5, 0.25, 0.75, 1 / 3, 2 / 3, 1 / math.sqrt(2), math.sqrt(2), 2 * math.sqrt(2),
              1 - 1 / math.sqrt(3), 1 - 2 ** -0.25, math.log(2), 1.0, 2.0, 3.0, 0.0]


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def _grounded(token: str, sources: list[float]) -> bool:
    """True if the decimal `token` equals some source number at the precision written."""
    x = float(token)
    decimals = len(token.split(".")[1])
    tol = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(x - v) <= tol for v in sources)


def ungrounded_numbers(answer: str, messages: list) -> list[str]:
    """Decimal numbers in `answer` that appear in no tool result or user question in `messages`.

    `messages` is the whole conversation so far: a number from an earlier tool result may be
    quoted again. Signs are ignored (an answer may write '−0.113' with a typographic minus or
    '0.051 − 0.030i'), so only magnitudes are compared."""
    sources = [abs(c) for c in _CONSTANTS]
    for m in messages:
        if isinstance(m, ToolMessage) or _is_user_question(m):   # never trust VERIFIER notes
            for t in re.findall(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?", _text(m)):
                sources.append(abs(float(t)))
                if "e" in t:                      # 1.1e-05 may be written as 1.1 × 10⁻⁵
                    sources.append(float(t.split("e")[0]))
    bad = []
    for m in _NUM.finditer(answer):
        t = m.group(0).lstrip("-")
        if _grounded(t, sources):
            continue
        if re.match(r"\s?%", answer[m.end():]):     # '61.12%' of a tool fraction 0.61122 is fine
            dp = t.partition(".")[2]
            shifted = f"{float(t) / 100:.{len(dp) + 2}f}"
            if _grounded(shifted, sources):
                continue
        bad.append(t)
    return list(dict.fromkeys(bad))       # unique, keep order


def _is_verifier(m) -> bool:
    return isinstance(m, HumanMessage) and str(m.content).startswith("VERIFIER:")


_USES = ("compute_measures", "compare_states", "plot_density_matrix", "apply_local_noise",
         "mix_states", "get_tomography_report", "evolve_state", "apply_waveplate", "compensate_phase",
         "simulate_tomography", "teleportation_fidelity", "qkd_key_rate", "certify_correlations",
         "plot_sweep", "plot_bloch_and_correlations", "plot_fringes", "plot_tomography_fit",
         "show_density_matrix", "apply_global_noise", "apply_kraus_operators", "reduced_state", "combine_photons")


def unused_new_states(turn_messages: list) -> list[str]:
    """States made by create_state / create_pure_state in this turn that no later tool used,
    when at least one other such state WAS used. 'create phi+' alone or 'create phi+, add noise'
    is fine; 'create 3 states, compare rho_1 with 2 of them' is flagged."""
    created, used = [], set()
    for m in turn_messages:
        if isinstance(m, ToolMessage):
            hit = re.match(r"Created (rho_\d+)", _text(m))
            if hit and m.name in ("create_state", "create_pure_state", "create_single_photon_state"):
                created.append(hit.group(1))
        elif isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc["name"] in _USES:
                    used |= set(re.findall(r"rho_\d+", json.dumps(tc["args"])))
    if not created or not (set(created) & used):
        return []
    return [h for h in created if h not in used]


def suggest_missing_calls(turn_messages: list, unused: list[str]) -> str:
    """Repeat, for each unused new state, the tool the model used on the other new states,
    e.g. compare_states(handle_a='rho_1', handle_b='rho_4')."""
    pattern = None
    for m in turn_messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc["name"] in ("compare_states", "compute_measures", "plot_density_matrix"):
                    pattern = tc
    if pattern is None:
        return "the tools the user asked for on " + ", ".join(unused)
    calls = []
    for h in unused:
        args = dict(pattern["args"])
        key = "handle_b" if pattern["name"] == "compare_states" else "handle"
        args[key] = h
        calls.append(pattern["name"] + "(" + ", ".join(f"{k}='{v}'" for k, v in args.items()) + ")")
    return "; ".join(calls)


_BELL = {"phi+": r"phi\s*\+|φ\s*[+⁺]|Φ\s*[+⁺]", "phi-": r"phi\s*-|φ\s*[-⁻−]|Φ\s*[-⁻−]",
         "psi+": r"psi\s*\+|ψ\s*[+⁺]|Ψ\s*[+⁺]", "psi-": r"psi\s*-|ψ\s*[-⁻−]|Ψ\s*[-⁻−]"}


def bell_states_missing(question: str, turn_messages: list) -> list[str]:
    """Bell states named in the question that no tool call of this turn created or used."""
    calls = [tc for m in turn_messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    if not calls:
        return []                                  # conceptual answer: nothing to check
    named = [k for k, pat in _BELL.items() if re.search(pat, question, flags=re.I)]
    touched = set()
    for tc in calls:
        a = tc["args"]
        for key in ("kind", "bell", "start_kind", "reference", "compare_with", "handle", "handle_a", "handle_b"):
            if isinstance(a.get(key), str):
                touched.add(a[key].lower().replace(" ", ""))
        for h in a.get("handles") or [] if isinstance(a.get("handles"), list) else []:
            if isinstance(h, str):                     # plot_sweep(handles=['phi+', 'psi+'])
                touched.add(h.lower().replace(" ", ""))
        for h in re.findall(r"rho_\d+", json.dumps(a)):   # an existing state that IS that Bell state
            if h in STORE:
                touched.add(DESC.get(h, "").strip().lower())
    return [k for k in named if k not in touched]


_FALSE_CLAIMS = [
    # (pattern, short reason shown to the user)
    # 'global depolarizing keeps it a Werner state (rank preserved), while local mixes in more components'
    (r"depolariz[^.\n]{0,100}(rank[- ]structure|preserv\w*\s+(the\s+)?rank)"
     r"|global[^.\n]{0,80}\bWerner[^.\n]{0,80}\b(while|whereas|but|unlike)\b(?![^.\n]*Werner)[^.\n]{0,30}\blocal"
     r"|local\s+depolariz[^.\n]{0,80}(mix\w*\s+in\s+more|not\s+(a\s+)?Werner|no\s+longer\s+(a\s+)?Werner)",
     "local depolarizing on both photons of a Bell state also gives a Werner state (Bell weight (1-p)² instead "
     "of 1-p), and neither channel preserves the rank (the pure Bell state becomes full rank)"),
    (r"discord[^.\n]{0,120}beyond\s+entanglement|beyond\s+entanglement[^.\n]{0,120}discord",
     "discord is not a correlation 'beyond entanglement': it includes entanglement"),
    (r"near[- ]?maxim|close to (the )?maxim|almost maxim|nearly maxim",
     "'near-maximal' is not supported by the numbers"),
    (r"(?<!logarithmic )(?<!log-)(?<!log )\b(concurrence|negativity)\b[^.\n;]{0,30}\b(use|uses|using|in|with)\s+"
     r"(a\s+|the\s+)?log(arithm)?s?\s*(base\s*|_)?2\b",
     "concurrence and negativity have no log base (log2 is used for entropies and log-negativity)"),
    (r"(half\s+(of\s+)?the\s+concurrence|N\s*=\s*C\s*/\s*2|concurrence\s+is\s+(twice|double))[^.\n]{0,40}"
     r"\b(consistent with|because|due to|indicat\w*|reflect\w*|as expected for|typical (of|for)|shows?)\b"
     r"[^.\n]{0,40}\b(mixed|mixedness|noise|noisy)",
     "N = C/2 holds for every pure state and every Werner state; it says nothing about mixedness"),
    (r"\ball\s+(the\s+)?correlations\s+(are|is)\s+(purely\s+|entirely\s+)?quantum"
     r"|\bno\s+classical\s+(part|component|correlations?)\b|\bpurely\s+quantum\s+correlations",
     "mutual information = classical correlation + discord, and the classical part is > 0 for every "
     "correlated state (a Bell state has I = 2: classical correlation 1 plus discord 1)"),
    (r"\bmixed\b[^.\n]{0,60}\b(but|yet|while|although)\b[^.\n]{0,40}\b(significantly|largely|mostly|fairly|highly|quite|"
     r"still|rather|relatively)\s+pure\b",
     "a state is either pure (purity 1) or mixed (purity < 1); 'mixed but significantly pure' says nothing"),
    (r"bit[- ]flip[^.\n]{0,100}(locally\s+)?(equivalent|amounts|reduces)\s+to\s+(a\s+)?depolari",
     "bit flips on both photons turn phi+ into a mixture of phi+ and psi+ only (rank 2), not into a "
     "depolarized (white-noise) state"),
    (r"\bkey\s+(fraction|rate)\b[^.\n]{0,80}\b(above|beats?|exceeds?|over)\s+(the\s+)?classical\s+(limit|bound)"
     r"|classical\s+(limit|bound)[^.\n]{0,40}\bkey\s+(fraction|rate)",
     "a secret key fraction has no classical limit: any value > 0 means a key can be made"),
    (r"^(?!.*(F3|F₃|setting|CJWR|test|criterion|inequalit)).*\bsteerable\b[^.\n]{0,60}[>≥]\s*[~≈]?\s*(0\.57\d*|1/√3)",
     "F3 > 1 (three settings) is only a sufficient test: Werner states are steerable with projective "
     "measurements for p > 1/2"),
]
# a claim that the ± values SHOWN have been scaled (advice such as "should be scaled" is fine)
_SCALED = r"(uncertaint|error bar|±)[^.\n]{0,60}\b(are|were|have been|has been)\b[^.\n]{0,40}\bscaled"


def _realistic_errors(messages: list) -> list[float]:
    vals = []
    for m in messages:
        if isinstance(m, ToolMessage) and "uncertainty_realistic" in _text(m):
            d = _parse(m.content) or {}
            vals += list((d.get("uncertainty_realistic") or {}).values())
    return vals


_QUANTITY = {"purity": "purity", "concurrence": "concurrence", "negativity": "negativity",
             "chsh": "chsh_max"}


_SAMPLED_TOOLS = ("evolve_state", "plot_measures_vs_noise", "plot_sweep")


def _latest_samples(messages: list) -> list[tuple[str, dict]]:
    """[(x label, {quantity: value})] in x order from the most recent time evolution, noise plot or
    sweep (first state of a sweep)."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name in _SAMPLED_TOOLS:
            d = _parse(m.content) or {}
            if m.name == "plot_sweep":
                first = next(iter((d.get("results") or {}).values()), {})
                samples = first.get("samples") or {}
            else:
                samples = (d.get("features") or {}).get("samples") or {}
            return list(samples.items())
    return []


def _last_evolution_series(messages: list) -> dict:
    """{quantity: [values in x order]} from the most recent evolution, noise plot or sweep."""
    samples = _latest_samples(messages)
    return {q: [v[q] for _, v in samples] for q in _QUANTITY.values() if samples and q in samples[0][1]}


def _never_violates_rule(messages: list) -> list[tuple[str, str]]:
    """'CHSH never violates the bound' is wrong if a sampled CHSH value of the latest plot, sweep or
    evolution (not counting the end points) is above 2."""
    inner = _latest_samples(messages)[1:-1]
    above = [(k, v["chsh_max"]) for k, v in inner if v.get("chsh_max", 0) > 2.0005]
    if not above:
        return []
    k, v = above[0]
    return [(r"never\s+(violat|exceed|goes?\s+above|rises?\s+above)[^.\n]*?(CHSH|Bell|local bound)"
             r"|(CHSH|Bell)[^.\n]*?never\s+(violat|exceed|goes?\s+above|rises?\s+above)"
             r"|\bno\s+(CHSH\s+|Bell[- ]?(inequality\s+)?)?violation[^.\n]{0,50}\b(for|at)\s+(any|all)\b"
             r"|\b(does|do|can)\s*(not|n't)\s+violate[^.\n]{0,50}\b(for|at)\s+(any|all)\s+(values?\s+of\s+)?(p|t|γ|λ)\b",
             f"CHSH is above 2 at intermediate values too (e.g. {v:.4f} at {k})")]


def _poor_fit_rule(messages: list) -> list[tuple[str, str]]:
    """'The fit is reasonable / acceptable' after a tomography fit with chi2/dof well above 1."""
    chi = None
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name in ("reconstruct_state_from_counts", "get_tomography_report",
                                                     "plot_tomography_fit"):
            d = _parse(m.content) or {}
            chi = d.get("chi2_per_dof", (d.get("fit_quality") or {}).get("chi2_per_dof"))
            break
        if _is_user_question(m):
            break
    if chi is None or chi <= 2:
        return []
    return [(r"\bfit\b[^.\n]{0,60}\b(reasonable|acceptable|good|fine|adequate|satisfactory)\b"
             r"|\b(reasonable|acceptable|good|adequate|satisfactory)\s+fit\b"
             r"|\b(captures|describes|reproduces|matches|fits)\s+(the\s+)?(data|counts|measurements)\s+(well|closely)",
             f"χ²/dof = {chi}: the fit is NOT consistent with counting noise alone (systematic errors)")]


def _linear_rules(series: dict) -> list[tuple[str, str]]:
    """'Purity drops linearly' is wrong if the evenly spaced samples are not on a straight line."""
    rules = []
    for word, q in (("purity", "purity"), ("concurrence", "concurrence"), ("negativity", "negativity"),
                    ("chsh", "chsh_max")):
        ys = series.get(q) or []
        if len(ys) < 3:
            continue
        n = len(ys) - 1
        dev = max(abs(y - (ys[0] + (ys[-1] - ys[0]) * i / n)) for i, y in enumerate(ys))
        if dev > 0.02:
            rules.append((rf"\b{word}\w*\b[^.\n]{{0,60}}\blinear(ly)?\b",
                          f"the {q.replace('_max', '')} does not change linearly in this result "
                          f"(values: {', '.join(f'{y:.3f}' for y in ys)})"))
    return rules


def _shape_rules(messages: list) -> list[tuple[str, str]]:
    """'CHSH stays above 2 for all p' when a sample reaches 2; 'oscillates' when the curve has a single
    turning point."""
    samples = _latest_samples(messages)
    rules = []
    # the samples are rounded to 4 decimals: 2.0000 may be 2.0000137 (collective dephasing, CHSH = 2√(1+C²)).
    # The tool decided with full precision; if it says CHSH stays above 2, believe it.
    feats = next(((_parse(m.content) or {}).get("features") or {} for m in reversed(messages)
                  if isinstance(m, ToolMessage) and m.name in _SAMPLED_TOOLS), {})
    if str(feats.get("chsh_drops_to_2_at", "")).startswith("stays above 2"):
        samples = []
    low = [(k, v["chsh_max"]) for k, v in samples if isinstance(v.get("chsh_max"), (int, float))
           and v["chsh_max"] <= 2.00005]                 # 2.0003 IS above 2
    if low:
        k, v = low[0]
        rules.append((r"(CHSH|Bell)[^.\n]{0,60}\b(remains?|stays?|is|keeps?)\s+(strictly\s+)?above\s+(the\s+)?"
                      r"(local\s+|classical\s+)?(bound|limit|2)\b[^.\n]{0,30}\b(for\s+all|at\s+all|everywhere|throughout|always)"
                      r"|\b(for\s+all|every)\s+(values?\s+of\s+)?(p|t|γ|λ)\b[^.\n]{0,40}(CHSH|Bell)[^.\n]{0,30}above",
                      f"CHSH is {v:.4f} at {k}: it reaches the bound 2 there, so it is not above 2 everywhere"))
    return rules


_OSCILLATE = {"oscillate": "fall and then rise again", "oscillates": "falls and then rises again",
              "oscillating": "falling and then rising again", "oscillated": "fell and then rose again"}


def _oscillation_correction(messages: list) -> list:
    """'oscillates' when no plotted measure has more than one turning point: corrected in place."""
    series = _last_evolution_series(messages)
    if not series:
        return []
    def turns(ys):
        steps = [b - a for a, b in zip(ys, ys[1:]) if abs(b - a) > 1e-4]
        return sum(1 for a, b in zip(steps, steps[1:]) if a * b < 0)
    if any(len(ys) < 5 or turns(ys) > 1 for ys in series.values()):
        return []
    return [(re.compile(r"\b(oscillat(?:e|es|ing|ed))\b", re.I), lambda m: _OSCILLATE[m.group(1).lower()],
             "the curves do not oscillate: each has at most one turning point")]


def _certification_rules(answer: str, messages: list) -> list[tuple[str, str]]:
    """'All violations exceed 35σ' or 'the strongest is Bell nonlocality' against certify_correlations."""
    tests = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "certify_correlations":
            tests = (_parse(m.content) or {}).get("tests") or []
            break
    sig = [(t.get("property", ""), t.get("n_sigma_realistic", t.get("n_sigma"))) for t in tests]
    sig = [(p, v) for p, v in sig if isinstance(v, (int, float))]
    if len(sig) < 2:
        return []
    rules = []
    low = min(v for _, v in sig)
    for mm in re.finditer(r"\ball\b[^.\n]{0,80}?\b(exceed\w*|above|over|beyond|more than)\s+(\d+(?:\.\d+)?)\s*σ", answer, re.I):
        if float(mm.group(2)) > low + 0.05:
            rules.append((re.escape(mm.group(0)), f"not all tests exceed {mm.group(2)}σ: the lowest is {low}σ"))
    top = max(sig, key=lambda x: x[1])[0].lower()
    claim = r"(most\s+significant|highest\s+significance|strongest\s+(evidence|significance|signal|violation))"
    for name, key in (("bell nonlocality", "chsh"), ("chsh", "chsh"), ("steering", "steer"), ("entanglement", "entangl")):
        pat = rf"\b{claim}\b[^.\n]{{0,40}}\b{name}|\b{name}\b[^.\n]{{0,40}}\b(is|being)\s+the\s+{claim}"
        if re.search(pat, answer, re.I) and key not in top:
            rules.append((pat, f"the most significant test is '{max(sig, key=lambda x: x[1])[0]}' "
                               f"({max(v for _, v in sig)}σ)"))
            break
    return rules


def _only_one_rules(messages: list) -> list[tuple[str, str]]:
    """'Only p = 0.8 is entangled' when compute_measures found several entangled states this turn."""
    ps, conc, chsh = {}, {}, {}
    for m in messages:
        if isinstance(m, ToolMessage):
            for h, p in re.findall(r"Created (rho_\d+) \([^,)]*,\s*p=(\d+(?:\.\d+)?)", _text(m)):
                ps[h] = p
            d = _parse(m.content) if m.name == "compute_measures" else None
            if isinstance(d, dict) and d.get("handle"):
                conc[d["handle"]] = d.get("concurrence")
                chsh[d["handle"]] = d.get("chsh_max")
    rules = []
    for what, vals, test, word in (("entangled", conc, lambda v: v > 1e-9, r"entangled"),
                                   ("CHSH-violating", chsh, lambda v: v > 2 + 1e-9, r"violat\w*\s+(the\s+)?CHSH")):
        good = sorted({ps[h] for h, v in vals.items() if h in ps and isinstance(v, (int, float)) and test(v)}, key=float)
        if len(good) >= 2:
            rules.append((rf"\bonly\s+(the\s+)?(state\s+with\s+|werner\s+(state\s+)?(with\s+)?)?p\s*=\s*\d+(\.\d+)?"
                          rf"\s+(state\s+)?(is|was|are)\s+{word}",
                          f"the states with p = {', '.join(good)} are all {what}"))
    return rules


def _final_state_rules(messages: list) -> list[tuple[str, str]]:
    """'The final state is not entangled' when the evolution ends with concurrence > 0."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "evolve_state":
            f = (_parse(m.content) or {}).get("features") or {}
            c = (f.get("at_t_max") or {}).get("concurrence")
            if isinstance(c, (int, float)) and c > 1e-4:
                return [(r"\bfinal\s+state\b(?:[^.\n]|\.\d){0,160}\b(not\s+entangled|separable|unentangled|no\s+entanglement)",
                         f"the concurrence at t_max is {c} > 0, so the final state is still (weakly) entangled")]
            return []
    return []


def _low_error_rules(answer: str) -> list[tuple[str, str]]:
    """'a very low phase-error rate (≈ 39 %)': above the BB84 limit of about 11 % it is not low."""
    rules = []
    for m in re.finditer(r"\b(?:very\s+)?(?:low|small)\s+(?:\w+[- ])?(?:error|QBER)[^.\n(]{0,20}\(\s*[≈~]?\s*"
                         r"(\d+(?:\.\d+)?)\s*%", answer, flags=re.I):
        if float(m.group(1)) >= 11:
            rules.append((re.escape(m.group(0)), f"an error rate of {m.group(1)} % is high, not low (BB84 "
                                                 "gives no key above about 11 % in both bases)"))
    return rules


def _compensation_rules(messages: list) -> tuple[list, list]:
    """After compensate_phase: it is a three-plate phase shifter, and the result is not a perfect phi+."""
    rules, corrections = [], []
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "compensate_phase":
            d = _parse(m.content) or {}
            f1 = d.get("fidelity_with_target_after")
            corrections.append((re.compile(r"\b(?:by\s+)?a\s+single\s+half[- ]wave\s+plate", re.I),
                                "by a QWP(45°)–HWP–QWP(45°) phase shifter",
                                "the compensation needs a QWP(45°)–HWP–QWP(45°) phase shifter, not a single HWP"))
            if isinstance(f1, (int, float)) and f1 < 0.99:
                corrections.append((re.compile(r"\b(turn\w*|transform\w*|convert\w*|brought|bring\w*)\s+into\s+"
                                               r"an?\s+perfect\s+", re.I), r"\1 closer to ",
                                    f"after the compensation the fidelity with the target is {f1}, not 1"))
                rules.append((r"\bperfect(ly)?\s+(φ⁺|φ\+|phi\+|∣φ⁺⟩|Bell)",
                              f"after the compensation the fidelity with the target is {f1}, not 1"))
                rules.append((r"\b(compensated|corrected|resulting|new|final|rotated)\s+state\s+is\s+the\s+closest\s+"
                              r"maximally\s+entangled",
                              f"the compensated state is still mixed (fidelity {f1}); the closest maximally "
                              "entangled state is a pure state"))
            break
    return rules, corrections


def _maximally_mixed_rule(messages: list) -> list[tuple[str, str]]:
    """'Becomes maximally mixed' is wrong unless some state of the latest result reaches purity 1/4
    (two photons; one photon is maximally mixed at purity 1/2 = unpolarized)."""
    for m in reversed(messages):
        if _is_user_question(m):
            break
        if isinstance(m, ToolMessage):
            if re.search(r"['\"]photons['\"]\s*:\s*1\b|1 photon:|; one photon['\"]", _text(m)):
                return []                         # a one-photon result: I/2 (purity 1/2) IS maximally mixed
            vals = [float(v) for v in re.findall(r"['\"]purity['\"]\s*:\s*([\d.]+)", _text(m))]
            if vals:
                if min(vals) > 0.2505:
                    return [(r"^(?!(?:[^.\n]|\.\d)*\b(reduced|marginal|local|single[- ]qubit|one[- ](qubit|photon)|each\s+(qubit|photon)|"
                             r"subsystems?|partial\s+trace|ρ_?[AB]\b|rho_?[AB]\b|both\s+photons|photons?\s+[AB]\b|\bI\s*/\s*2\b|"
                             r"S\s*\(\s*[AB]\s*\)|S_?[AB]\b|individual|single[- ]photon|each\s+of)).*?(?<!not )(?<!never )(?<!isn't )"
                             r"\bmaximally\s+mixed",
                             f"only I/4 is maximally mixed (purity 1/4); the lowest purity here is {min(vals):.4f}")]
                return []
    return []


_COMPARE_Q = [(r"coherence", "coherence_l1"), (r"purit", "purity"), (r"concurrence|entangle", "concurrence"),
              (r"\bnegativity", "negativity"), (r"chsh|bell", "chsh_max"), (r"mutual information", "mutual_information"),
              (r"discord", "discord"), (r"entropy", "entropy"), (r"entanglement of formation|\beof\b",
                                                                   "entanglement_of_formation")]
_LESS = r"lower|smaller|less|reduced|weaker|below"
_MORE = r"higher|larger|bigger|greater|more|stronger|above|exceed"
_DESTROY = (r"\b(?:(?:destroy|destruct|reduc|degrad|damag|harm|detriment|los[estin]|kill|eras|suppress|attack|"
            r"deplet|deteriorat|wash|decay|decreas|diminish|hurt|severe|aggressive|drop|fall|declin|penal)\w*"
            r"|effective\s+at\s+remov\w*|effects?|impacts?|noisier|harsher|worse)\b")


def _wrong_comparison_rules(answer: str, messages: list) -> list[tuple[str, str]]:
    """'Coherence is lower than for phi+' is checked against the compute_measures results of the
    two states (only when exactly two states were measured)."""
    meas = {}
    for m in messages[_turn_start(messages):]:          # this question only (not earlier states)
        if isinstance(m, ToolMessage) and m.name == "compute_measures":
            d = _parse(m.content) or {}
            if d.get("handle"):
                meas[d["handle"]] = d
    if len(meas) != 2:
        return []
    (h1, m1), (h2, m2) = meas.items()

    def names(h):
        desc = DESC.get(h, "").strip().lower()
        out = [re.escape(h)]
        if desc in ("phi+", "phi-", "psi+", "psi-"):
            out += [_BELL[desc], r"ideal"]
        elif desc.startswith("reconstructed"):
            out += [r"reconstructed", r"\blab\b", r"measured"]
        elif re.match(r"(global|collective|correlated)_", desc):
            out += [r"\bglobal", r"\bcollective", r"\bcorrelated"]
        elif re.search(r"on (both qubits|qubit [AB] only) of", desc):
            out += [r"\blocal"]
        return "|".join(out)
    pats = {h1: names(h1), h2: names(h2)}
    rules = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", answer):
        # 'the CHSH maximum is closer to the classical bound 2 for local noise'
        close = re.search(r"CHSH[^.\n]*?\bcloser\s+to\s+(?:the\s+)?(?:classical\s+|local\s+)?bound(?:\s+(?:of\s+)?2)?"
                          r"(.{0,40})", sent, flags=re.I)
        if close and "chsh_max" in m1 and "chsh_max" in m2:
            named = [h for h in (h1, h2) if re.search(pats[h], close.group(1) + " " + sent[:close.start()], re.I)]
            if len(named) == 1:
                s_, o_ = named[0], (h2 if named[0] == h1 else h1)
                ds, do = abs(float(meas[s_]["chsh_max"]) - 2), abs(float(meas[o_]["chsh_max"]) - 2)
                if ds > do + 1e-4:
                    rules.append((re.escape(sent.strip()[:60]),
                                  f"CHSH max of {s_} is {float(meas[s_]['chsh_max']):.4f} and of {o_} is "
                                  f"{float(meas[o_]['chsh_max']):.4f}: {o_} is the one closer to 2"))
                    continue
        c = re.search(rf"\b({_LESS}|{_MORE})\b[^.\n]{{0,40}}?\bthan\b(.{{0,40}})", sent, flags=re.I)
        if not c:
            continue
        after = c.group(2)
        targets = [h for h in (h1, h2) if re.search(pats[h], after, flags=re.I)]
        if len(targets) != 1:
            continue
        t = targets[0]
        s = h2 if t == h1 else h1
        before = sent[:c.start()] + " " + sent[c.start():c.end() - len(after)]
        less = re.match(_LESS, c.group(1), flags=re.I) is not None
        # 'local noise DESTROYS more entanglement than global' = less entanglement is left: invert
        window = sent[max(0, c.start() - 40):c.start()] + " " + sent[c.end(1):c.end() - len(after)]
        if re.search(_DESTROY, window, flags=re.I):
            less = not less
        for pat, q in _COMPARE_Q:
            if not re.search(pat, before, flags=re.I) or q not in meas[s] or q not in meas[t]:
                continue
            vs, vt = float(meas[s][q]), float(meas[t][q])
            if (less and vs > vt + 1e-4) or (not less and vs < vt - 1e-4):
                rules.append((re.escape(sent.strip()[:60]),
                              f"{q.replace('_', ' ')} of {s} is {vs:.4f} and of {t} is {vt:.4f}, so it is "
                              f"{'not lower' if less else 'not higher'}"))
                break
    return rules


def _wrong_monotonic_rules(answer: str, series: dict) -> list[tuple[str, str]]:
    """'Purity increases monotonically' is checked against the evolution samples."""
    rules = []
    if not series:
        return rules
    for sent in re.split(r"(?<=[.!?])\s+|\n", answer):
        if not re.search(r"monoton|steadil|continuous|keeps|always|throughout", sent, flags=re.I):
            continue
        m = re.search(r"(purity|concurrence|negativity|chsh)\W+(?:[\w‑-]+\W+){0,4}?"
                      r"(increas|ris(?:e|es|ing)|grow|decreas|drop|fall|decay|declin)", sent, flags=re.I)
        if m is None:
            continue                               # direction = first verb after the quantity
        q = _QUANTITY[m.group(1).lower()]
        if q not in series:
            continue
        up = m.group(2).lower().startswith(("increas", "ris", "grow"))
        ys = series[q]
        steps = [b - a for a, b in zip(ys, ys[1:])]
        ok = all(s >= -1e-4 for s in steps) if up else all(s <= 1e-4 for s in steps)
        if not ok:
            word = "increase" if up else "decrease"
            rules.append((re.escape(sent.strip()[:60]),
                          f"the {q.replace('_max', '')} does not {word} monotonically in this result "
                          f"(values: {', '.join(f'{y:.3f}' for y in ys)})"))
    return rules


_BELL_LOCAL = (r"\bBell[- ]local\b|local[- ]hidden[- ]variable[- ]compatible|admits? a local (hidden[- ]variable )?model"
               r"|(compatible|consistent)\s+with\s+(a\s+|any\s+|some\s+)?local[- ](hidden[- ]variable|realis)"
               r"|(described|explained|reproduced|modell?ed)\s+by\s+(a\s+)?local[- ]hidden[- ]variable"
               r"|\b(state|correlations?|statistics|it|rho_\d+)\s+(is|becomes|are|become)\s+(now\s+)?(fully\s+|effectively\s+)?"
               r"local\b(?![- ](unitar|operation|transformation|channel|noise|gate|rotation))"
               r"|loss of (quantum )?non-?locality|no longer non-?local",
               "CHSH <= 2 does not prove that a local hidden-variable model exists")
# "separable, so no quantum correlations": separable states generally still have discord > 0
_NO_QUANTUM_CORR = (r"\bno(\s+longer)?\s+(exhibit\w*\s+|ha(s|ve)\s+|show\w*\s+|contain\w*\s+)?(any\s+)?quantum\s+correlations"
                    r"|\blos(e|es|ing|t)\s+(all\s+)?(of\s+)?(its\s+)?quantum\s+correlations"
                    r"|\bwithout\s+(any\s+)?quantum\s+correlations"
                    r"|quantum\s+correlations\s+beyond\s+(the\s+)?classical"
                    r"|\b(destroy\w*|eliminat\w*|remov\w*|kill\w*|eras\w*|wip\w*\s+out)\s+(all\s+)?(the\s+|its\s+)?quantum\s+correlations"
                    r"|\ball\s+(the\s+)?quantum\s+correlations\s+(disappear|vanish|are\s+(lost|gone|destroyed))"
                    r"|\b(becomes?|is|are)\s+(now\s+)?(purely\s+|fully\s+|effectively\s+)?classical\b"
                    r"(?!\s+(bound|limit|threshold|value|fidelity|teleport|channel|communication|strateg))",
                    "a separable state can still have quantum correlations (discord > 0); only zero discord "
                    "means purely classical correlations")


def _zero_discord_shown(messages: list) -> bool:
    """True if a tool result in the conversation reports discord = 0 (then 'no quantum
    correlations' may be right)."""
    for m in messages:
        if isinstance(m, ToolMessage):
            for v in re.findall(r"['\"]discord['\"]\s*:\s*([-\d.e]+)", _text(m)):
                try:
                    if abs(float(v)) < 1e-6:
                        return True
                except ValueError:
                    pass
    return False


_TIME = re.compile(r"\bt\s*[=≈~]\s*(\d+(?:\.\d+)?)")


def times_not_in_tools(question: str, turn: list, answer: str, history: list | None = None) -> list[str]:
    """Times written in the answer ('t = 2.0', or the first column of a 'Time' table) that are
    neither a sample time, a key time nor t_max of an evolution (this turn or earlier ones, so a
    comparison with a previous result is fine) nor a number in the question."""
    allowed = [float(x) for x in _TIME.findall(question)] + [0.0]     # only times the user wrote
    if not any(isinstance(m, ToolMessage) and m.name == "evolve_state" for m in turn):
        return []
    have = False
    for m in (history or turn):
        if isinstance(m, ToolMessage) and m.name == "evolve_state":
            d = _parse(m.content) or {}
            f = d.get("features") or {}
            for k in list(f.get("samples", {})) + list(f.get("at_key_times", {})):
                allowed += [float(x) for x in re.findall(r"t=(\d+(?:\.\d+)?)", k)]
            for v in (f.get("entanglement_vanishes_at"), f.get("concurrence_halves_at"), f.get("chsh_drops_to_2_at"),
                      f.get("degree_of_polarization_halves_at")):
                if isinstance(v, (int, float)):
                    allowed.append(float(v))
            have = True
    if not have:
        return []
    written = _TIME.findall(answer)
    lines = answer.split("\n")
    for i, line in enumerate(lines):                     # tables whose first header cell is a time
        if line.lstrip().startswith("|") and re.match(r"\s*\|\s*(time|t)\b", line, flags=re.I):
            for row in lines[i + 2:]:
                if not row.lstrip().startswith("|"):
                    break
                m = re.match(r"\s*\|\s*(?:t\s*[=≈]\s*)?(\d+(?:\.\d+)?)\s*\|", row)
                if m:
                    written.append(m.group(1))
    bad = []
    for tok in written:
        d = len(tok.split(".")[1]) if "." in tok else 0
        if not any(abs(float(tok) - a) <= 0.5 * 10 ** -d + 1e-9 for a in allowed):
            bad.append(tok)
    return list(dict.fromkeys(bad))


# statements corrected IN PLACE (deleting the bullet would also delete its correct part):
# (pattern, replacement, reason shown to the user)
_CORRECTIONS = [
    # 'H sits at the north pole of the Poincaré sphere': on the Poincaré sphere (S3 vertical) the poles are
    # R and L; linear states lie on the equator (our Bloch DRAWING puts H on top, which is a different picture)
    (re.compile(r"\b(?:sits|lies|is|located)\s+at\s+the\s+(?:north|south)\s+pole\s+of\s+the\s+Poincar[eé]\s+sphere"
                r"(\s*\((?:H|V|D|A)\))", re.I),
     r"lies on the equator of the Poincaré sphere\1",
     "on the Poincaré sphere the poles are R and L; linear polarizations lie on the equator"),
    # collective dephasing: |HH> and |VV> differ by TWO V's (factor (1-p)^4 / e^(-4γt)), not one
    (re.compile(r"(HH\s*[–—−-]\s*VV\b[^.\n]{0,80}?\b(?:differ(?:ing|s)?\s+by|with)\s+)one(\s+(?:more\s+)?V\b)", re.I),
     r"\1two\2", "|HH⟩ and |VV⟩ differ by two V's, not one"),
    # "CHSH <= 2, so it is not Bell nonlocal": the inference is wrong (other inequalities, POVMs)
    (re.compile(r",?\s*\b(so|hence|therefore|thus|which means|meaning)\s+(it|the state|this state)\s+"
                r"is\s+not\s+(Bell[- ])?non-?local\b", re.I),
     "; this alone does not prove that a local hidden-variable model exists",
     "no CHSH violation does not by itself prove that a state is Bell local"),
    # "one party can nonlocally affect the other's state": no signalling
    (re.compile(r"\b(nonlocally|remotely|instantaneously)\s+(affect|change|influence|alter|control)\w*\s+"
                r"(the\s+)?(other(\s+party)?(['’]s)?|its\s+partner(['’]s)?|Bob(['’]s)?)\s+(state|photon|qubit)", re.I),
     "steer (by the choice of measurement, without any signalling) the other's conditional state",
     "no measurement on one photon can change the other photon's state (no signalling)"),
    # the hierarchy written the wrong way round ('entangled ⊂ steerable ⊂ Bell-nonlocal')
    (re.compile(r"\bentangled\s*⊂\s*(?:EPR[- ])?steerable\s*⊂\s*(?:Bell[- ])?non-?local", re.I),
     "Bell-nonlocal ⊂ steerable ⊂ entangled",
     "the sets are nested the other way: every Bell-nonlocal state is steerable and every steerable state is entangled"),
    # "no Bell violation": only CHSH was computed (other Bell inequalities were not tested)
    (re.compile(r"\bno\s+(?:Bell|Bell[- ]inequality)\s+violation", re.I), "no CHSH violation",
     "only CHSH was tested: other Bell inequalities may still be violated"),
    # "CHSH max >= 2 means no violation": no violation is CHSH <= 2
    (re.compile(r"(?:≥|>=)\s*2(\s+(?:means|implies|indicates|shows)\s+no\s+(?:CHSH\s+|Bell\s+)?violation)", re.I),
     r"≤ 2\1", "no CHSH violation means CHSH ≤ 2, not ≥ 2"),
    # "between 1/3 and 1/√2 the states are entangled but local": that window is only CHSH-satisfying
    # (some Werner states just below 1/√2 violate other Bell inequalities). Only on a line about CHSH,
    # because e.g. Werner's local model for p ≤ 1/2 makes "entangled but local" correct elsewhere.
    (re.compile(r"(?m)^(?=[^\n]*(?:CHSH|0\.707|1/√2))([^\n]*?)(['\"“‘]?)entangled[,]?\s+but\s+(?:Bell[- ])?local\b"
                r"(['\"”’]?)", re.I),
     r"\1\2entangled but CHSH-satisfying\3",
     "no CHSH violation does not make a state local: other Bell inequalities can still be violated"),
]
# for a state made from DATA a CHSH value computed with the optimal settings indicates, but does not
# prove, Bell nonlocality (the settings were not measured directly, loopholes are open)
_PROVES_NONLOCAL = (re.compile(r"\b(?:prov(?:es|ing|e|en)|confirm(?:s|ing|ed)?|establish(?:es|ing|ed)?)\s+"
                               r"((?:genuine\s+)?(?:Bell\s+)?non-?local(?:ity|\s+correlations)?)"
                               r"|\brul(?:es|ing|e)\s+out\s+((?:all\s+|any\s+)?local[- ]hidden[- ]variable\s+(?:theories|models))",
                               re.I),
                    lambda m: f"indicating {m.group(1)}" if m.group(1) else
                    "indicating a violation of the bound for local hidden-variable models",
                    "a CHSH value computed from a reconstructed state indicates but does not prove Bell "
                    "nonlocality (that needs the CHSH settings measured directly, with the loopholes closed)")
_NOT_NONLOCAL = _CORRECTIONS[0]
# circular light: its polarization ellipse has no azimuth ('right circular, azimuth 45°' copied from the input)
_CIRCULAR_AZIMUTH = [
    (re.compile(r"(circular[^|\n.;]{0,40}?)\bazimuth\s*(?:of\s*|=\s*|:\s*)?[-+]?\d+(?:\.\d+)?\s*°", re.I),
     r"\1azimuth undefined", "circular polarization has no ellipse azimuth (it is undefined)"),
    (re.compile(r"\bazimuth\s*(?:of\s*|=\s*|:\s*)?[-+]?\d+(?:\.\d+)?\s*°(\s*(?:,|and)\s*ellipticity(?:\s+angle)?\s*"
                r"(?:of\s*|=\s*|:\s*)?[+]?45(?:\.0+)?\s*°)", re.I),
     r"azimuth undefined\1", "circular polarization has no ellipse azimuth (it is undefined)"),
]


def _about_one_photon(answer: str) -> bool:
    """The answer names only stored ONE-photon states (correlations between photons do not apply)."""
    hs = set(re.findall(r"\brho_\d+\b", answer))
    return bool(hs) and all(h in STORE and STORE[h].shape[0] == 2 for h in hs)


_CHSH_HALF_WHY = "values above 1.4142 are CHSH itself, not CHSH/2 (the figure plots CHSH max / 2)"
_CHSH_HALF = re.compile(r"CHSH(?:[ _]max(?:imum)?)?\s*/\s*2", re.I)


def fix_chsh_half_label(answer: str) -> tuple[str, bool]:
    """The figures plot 'CHSH max / 2' (to share the 0..1 axis); a value above sqrt(2) labelled 'CHSH/2' is
    the CHSH value itself (CHSH/2 <= sqrt(2)). Relabel such table headers and prose as 'CHSH max'."""
    changed = False
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        if not (line.lstrip().startswith("|") and _CHSH_HALF.search(line)):
            continue
        cells = line.strip().strip("|").split("|")
        cols = [j for j, c in enumerate(cells) if _CHSH_HALF.search(c)]
        vals = []
        for row in lines[i + 2:]:
            if not row.lstrip().startswith("|"):
                break
            rc = row.strip().strip("|").split("|")
            for j in cols:
                m = re.search(r"\d+\.\d+", rc[j]) if j < len(rc) else None
                if m:
                    vals.append(float(m.group(0)))
        if vals and max(vals) > 1.4143:
            lines[i] = _CHSH_HALF.sub("CHSH max", line)
            changed = True
    for i, line in enumerate(lines):          # a transposed table: 'CHSH/2' is the ROW label (first cell)
        cells = line.strip().strip("|").split("|") if line.lstrip().startswith("|") else []
        if cells and _CHSH_HALF.search(cells[0]):
            vals = [float(x) for c in cells[1:] for x in re.findall(r"\d+\.\d+", c)]
            if vals and max(vals) > 1.4143:
                lines[i] = line.replace(cells[0], _CHSH_HALF.sub("CHSH max", cells[0]), 1)
                changed = True
    text = "\n".join(lines)

    def prose(m):
        nonlocal changed
        if float(m.group(2)) > 1.4143:
            changed = True
            return "CHSH max" + m.group(1) + m.group(2)
        return m.group(0)
    text = re.sub(r"CHSH(?:[ _]max(?:imum)?)?\s*/\s*2(\s*(?:=|:|≈|of|is)?\s*)(\d+\.\d+)", prose, text, flags=re.I)
    return text, changed


def remove_false_claims(answer: str, messages: list) -> tuple[str, list[str]]:
    """Delete bullets/sentences that state known-false physics, WITHOUT asking the model again
    (a re-ask often returns a shortened answer). Returns (new answer, reasons)."""
    rules = list(_FALSE_CLAIMS) + [_BELL_LOCAL]
    if not _zero_discord_shown(messages) and not _about_one_photon(answer):
        rules.append(_NO_QUANTUM_CORR)
    real = _realistic_errors(messages)
    shown = [float(t) for t in re.findall(r"±\s*(\d+\.\d+)", answer)]     # the ± values written
    if shown and not any(abs(s - r) < 0.0006 for s in shown for r in real):  # 'scaled' but they aren't
        rules.append((_SCALED, "the ± values shown are statistical, not scaled"))
    series = _last_evolution_series(messages)
    rules += _wrong_monotonic_rules(answer, series)
    rules += _never_violates_rule(messages)
    rules += _maximally_mixed_rule(messages)
    rules += _wrong_comparison_rules(answer, messages)
    rules += _poor_fit_rule(messages)
    rules += _linear_rules(series)
    rules += _low_error_rules(answer)
    rules += _shape_rules(messages)
    rules += _certification_rules(answer, messages)
    rules += _only_one_rules(messages)
    rules += _final_state_rules(messages)
    comp_rules, comp_corrections = _compensation_rules(messages)
    rules += comp_rules
    reasons, out = [], []
    corrections = list(_CORRECTIONS) + comp_corrections + _oscillation_correction(messages)
    if any(isinstance(m, ToolMessage) and re.search(r"azimuth undefined|undefined \(circular", _text(m))
           for m in messages):
        corrections += _CIRCULAR_AZIMUTH
    if any(isinstance(m, ToolMessage) and m.name in ("reconstruct_state_from_counts", "simulate_tomography")
           for m in messages):
        corrections.append(_PROVES_NONLOCAL)
    answer, relabelled = fix_chsh_half_label(answer)
    if relabelled:
        reasons.append(_CHSH_HALF_WHY)
    for pat, repl, why in corrections:          # correct in place instead of deleting the line
        if pat.search(answer):
            answer = pat.sub(repl, answer)
            reasons.append(why)
    removed = []
    for line in answer.split("\n"):
        hit = next(((p, why) for p, why in rules if re.search(p, line, flags=re.I)), None)
        if hit is None:
            out.append(line)
            continue
        reasons.append(hit[1])
        if re.match(r"\s*([•*\-]|\d+\.)\s", line):
            removed.append((line.strip(), hit[1]))
            continue                                   # drop the whole bullet
        wrap = re.match(r"\s*(\*\*|\*|_)", line)
        marker = wrap.group(1) if wrap and line.rstrip().endswith(wrap.group(1)) else ""
        body = line.strip()[len(marker):len(line.strip()) - len(marker)] if marker else line
        sentences = re.split(r"(?<=[.!?])\s+", body)
        kept = [s for s in sentences if not re.search(hit[0], s, flags=re.I)]
        removed += [(s.strip(), hit[1]) for s in sentences if re.search(hit[0], s, flags=re.I)]
        if kept:
            out.append(f"{marker}{' '.join(kept).strip()}{marker}")   # keep a whole-line italic intact
    _log_removed(removed)
    return "\n".join(out), list(dict.fromkeys(reasons))


def _log_removed(removed: list[tuple[str, str]]) -> None:
    """Keep every deleted sentence (with its reason) in logs/answer_check.log, so a wrong deletion can be
    found later: the answer itself shows only the reason. Best effort, never raises; off during tests."""
    if not removed or "PYTEST_CURRENT_TEST" in os.environ:
        return
    try:
        import datetime
        folder = os.environ.get("QI_LOG_DIR", "logs")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "answer_check.log"), "a", encoding="utf-8") as f:
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            try:
                q = str(CURRENT_QUESTION["text"])[:200]
            except Exception:  # noqa: BLE001
                q = ""
            for text, why in removed:
                f.write(f"{stamp}\tQ: {q}\n  REMOVED: {text}\n  REASON: {why}\n")
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("could not write the answer-check log", exc_info=True)


def sudden_death_contradiction(turn: list, answer: str) -> str | None:
    """The tool says sudden death at t = X, the answer says there is none (or the reverse)."""
    deaths = []
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "evolve_state":
            v = ((_parse(m.content) or {}).get("features") or {}).get("entanglement_vanishes_at")
            deaths.append(v)
    if not deaths:
        return None
    denies = re.search(r"no (entanglement )?sudden death|not (a )?sudden death|only asymptotically|"
                       r"never (dies|vanishes|reaches zero)", answer, flags=re.I)
    claims = re.search(r"sudden[- ]death (at|occurs|happens)|dies (suddenly|at)", answer, flags=re.I)
    if denies and all(isinstance(v, (int, float)) for v in deaths):
        return (f"the tool reports SUDDEN DEATH at t = {deaths[-1]} (concurrence exactly 0 at a finite "
                f"time), but your answer says '{denies.group(0)}'")
    if claims and all(isinstance(v, str) for v in deaths):
        return "the tool reports NO sudden death in this time window, but your answer claims one"
    return None


_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def wrong_row_values(answer: str, history: list) -> list[str]:
    """Table rows that name a stored state (rho_k / ρₖ) but contain a decimal that is none of that
    state's measures (recomputed here), its parameters, its comparisons or report numbers."""
    import qi_tools as qt
    problems = []
    for line in answer.split("\n"):
        if not line.lstrip().startswith("|") or re.fullmatch(r"[\s|:\-]+", line):
            continue
        hs = set(re.findall(r"rho_(\d+)|ρ([₀-₉]+)", line))
        handles = {f"rho_{a or b.translate(_SUB)}" for a, b in hs}
        if len(handles) != 1:
            continue
        h = handles.pop()
        if h not in STORE:
            continue
        r = STORE[h]
        if r.shape[0] == 2:                       # one photon: Stokes, degree of polarization, ellipse, ...
            sv = qt.stokes_vector(r)
            ell = qt.polarization_ellipse(sv) or {}
            ok = [qt.purity(r), qt.von_neumann_entropy(r), float(np.linalg.norm(sv)), *map(float, sv),
                  *[v for v in ell.values() if isinstance(v, float)]]
            ok += [float(np.real(np.trace(r @ P))) for P in qt.pauli_projectors_1q().values()]
            ok += [abs(float(v)) for v in np.real(r).ravel()] + [abs(float(v)) for v in np.imag(r).ravel()]
        else:
            ok = [qt.purity(r), qt.concurrence(r), qt.negativity(r), qt.log_negativity(r), qt.chsh_max(r),
                  qt.coherence_l1(r), qt.mutual_information(r), qt.quantum_discord(r),
                  qt.von_neumann_entropy(r)]
        ok += [float(x) for x in re.findall(r"\d+(?:\.\d+)?", DESC.get(h, ""))]
        about_h = re.compile(rf"(?<![A-Za-z0-9]){h}(?!\d)")
        args_of = {tc["id"]: json.dumps(tc["args"]) for m in history if isinstance(m, AIMessage)
                   for tc in (m.tool_calls or [])}
        for m in history:                         # comparisons, plots and reports involving h
            if (isinstance(m, ToolMessage) and m.name != "evolve_state"     # (an evolution lists every time)
                    and (about_h.search(_text(m)) or about_h.search(args_of.get(m.tool_call_id, "")))):
                ok += [float(x) for x in re.findall(r"\d+\.\d+(?:e-?\d+)?", _text(m))]
        cells = line.split("|")[2:] if line.strip().startswith("|") else line.split("|")
        for tok in re.findall(r"(?<![\w.])\d+\.\d{2,}", " ".join(cells)):
            d = len(tok.split(".")[1])
            if not any(abs(float(tok) - v) <= 0.5 * 10 ** -d + 1e-9 for v in ok):
                problems.append(f"{h}: {tok}")
    return problems


_PARAM_HEADER = re.compile(r"^\s*(\*\*)?(noise\s+(level|parameter|probability|strength)?\s*)?"
                           r"(p|γ|gamma|λ|lambda|t|time|θ|theta|φ|phi|v|visibility|parameter|noise|angle)\b", re.I)
_CELL_PARAM = re.compile(r"\b(?:p|t|γ|λ|θ|φ|v)\s*[=≈]\s*(\d+(?:\.\d+)?)", re.I)
_APPROX_CELL = re.compile(r"^\s*([~≈<>≤≥]|approx|about\b|around\b)", re.I)


def _key_points(turn: list) -> tuple[str, list[tuple[float, dict]]]:
    """(tool name, [(x, {quantity: value})]) of the latest noise plot, sweep or time evolution this
    turn: its samples, key points / key times and the values at sweep crossings."""
    pts, last = [], ""
    for m in turn:                     # every sweep / evolution of this turn (several states may be compared)
        if not (isinstance(m, ToolMessage) and m.name in _SAMPLED_TOOLS):
            continue
        d = _parse(m.content) or {}
        last = m.name

        def add(key, vals):
            x = re.search(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", str(key))
            if x and isinstance(vals, dict):
                pts.append((float(x.group(0)), {k: v for k, v in vals.items() if isinstance(v, (int, float))}))
        if m.name == "plot_sweep":
            for res in (d.get("results") or {}).values():
                for k, v in (res.get("samples") or {}).items():
                    add(k, v)
                for info in (res.get("measures") or {}).values():
                    crossings = info.get("crossings")
                    for c in crossings if isinstance(crossings, list) else []:
                        add(c.get("at"), c.get("values_there"))
        else:
            f = d.get("features") or {}
            for part in ("samples", "at_key_points", "at_key_times"):
                for k, v in (f.get(part) or {}).items():
                    add(k, v)
    return last, pts


def wrong_parameter_rows(answer: str, turn: list) -> list[str]:
    """Rows (or, in a transposed table, columns) indexed by a noise parameter, Werner p or time whose
    cells are approximate ('~0.75', '>0') or are no tool value at that parameter (a guessed purity)."""
    name, pts = _key_points(turn)
    if not pts:
        return []
    def at(x_tok):
        dec = len(x_tok.split(".")[1]) if "." in x_tok else 0
        tol = 0.5 * 10 ** -dec + 1e-9
        return [vals for x, vals in pts if abs(x - float(x_tok)) <= tol]
    series = []                                        # (parameter as written, [cells at that parameter])
    lines = answer.split("\n")
    for i, line in enumerate(lines):
        if not (line.lstrip().startswith("|") and i + 1 < len(lines) and re.fullmatch(r"[\s|:\-]+", lines[i + 1])):
            continue                                       # not a header row
        header = [c.strip() for c in line.strip().strip("|").split("|")]
        rows = []
        for row in lines[i + 2:]:
            if not row.lstrip().startswith("|"):
                break
            rows.append([c.strip() for c in row.strip().strip("|").split("|")])
        head_params = [_CELL_PARAM.search(c) for c in header[1:]]
        if len(header) > 2 and all(head_params):          # transposed: | Measure | p = 0.0 | p = 0.5 | ...
            for k, mx in enumerate(head_params, start=1):
                series.append((mx.group(1), [r[k] for r in rows if k < len(r)]))
            continue
        col = next((j for j, c in enumerate(header) if _PARAM_HEADER.match(c) and not re.search(r"\d", c)), None)
        for cells in rows:
            if col is not None and col < len(cells):
                mx = re.search(r"\d+(?:\.\d+)?", cells[col])
                x_tok, pcol = (mx.group(0) if mx else None), col
            else:
                found = list(_CELL_PARAM.finditer(cells[0])) if cells else []
                x_tok, pcol = (found[-1].group(1) if found else None), 0   # '(γ=1) t=0.35': the row is t
            if x_tok:
                series.append((x_tok, [c for j, c in enumerate(cells) if j != pcol]))
    problems = []
    for x_tok, cells in series:
        there = at(x_tok)
        values = [v for vals in there for v in vals.values()]
        listing = (", ".join(f"{k} {v}" for k, v in there[0].items()) if there else "")
        for c in cells:
            if not c:
                continue
            if _APPROX_CELL.match(c) and re.search(r"\d", c):
                problems.append(f"at {x_tok}: '{c}' is not an exact value"
                                + (f" (the tool gives {listing})" if listing else ""))
                continue
            for tok in _NUM.findall(c):
                t = tok.lstrip("-")
                if not values:
                    problems.append(f"at {x_tok} the tool gave no values (only at its samples and key "
                                    f"points), so '{t}' cannot be quoted there")
                    break
                if not _grounded(t, [abs(v) for v in values]):
                    problems.append(f"at {x_tok}: {t} is not a tool value there (the tool gives {listing})")
    return list(dict.fromkeys(problems))


_QUOTED = re.compile(r"(?<![-\w])(?<!log )(purity|concurrence|negativity)\b\s*(?:=|≈|~|:|of|is|was|becomes)?\s*"
                     r"(?:about\s+|only\s+|≈\s*)?(\d+(?:\.\d+)?)(?![\d/.]\d|\s*/|\s*%|\s*[×x*·]\s*\w|\s+for\b)", re.I)
# a value of a reference or hypothetical state, not of the state computed ('a Bell state would have C = 1')
_REFERENCE_CONTEXT = re.compile(r"maxim|\bBell\b|\bpure\b|original|ideal|would|separable|initial|\be\.g\.|"
                                r"\bfrom\s+\d|\bφ⁺|\bphi\+|\bpsi", re.I)


def quoted_values_not_computed(answer: str, history: list) -> list[str]:
    """'purity 0.5' in prose when no tool in the conversation gave a purity of 0.5 (a bare 0.5 passes
    the number check because 0.5 is also a threshold)."""
    text = " ".join(_text(m) for m in history if isinstance(m, ToolMessage))
    if not text:
        return []                                      # a concept question: textbook values are fine
    bad = []
    for m in _QUOTED.finditer(answer):
        key, tok = m.group(1).lower(), m.group(2)
        line = answer[answer.rfind("\n", 0, m.start()) + 1:m.start()]
        if line.lstrip().startswith("|"):
            continue                                   # tables are checked elsewhere
        before = [x.end() for x in re.finditer(r"[.;](?=\s)|\n|\(", answer[:m.start()])]
        a = before[-1] if before else 0
        after = re.search(r"[.;](?=\s|$)|\n|\)", answer[m.end():])
        b = m.end() + after.start() if after else len(answer)
        if _REFERENCE_CONTEXT.search(answer[a:b]):
            continue
        vals = [abs(float(v)) for v in re.findall(rf"['\"]{key}['\"]\s*:\s*(-?\d+(?:\.\d+)?)", text)]
        vals += [abs(float(v)) for v in re.findall(rf"['\"]{key}['\"]\s*:\s*\[\s*(-?\d+(?:\.\d+)?)", text)]
        if not vals:
            bad.append(f"{key} {tok} (no tool in this conversation computed a {key})")
        elif not _grounded(tok if "." in tok else tok + ".0", vals):
            bad.append(f"{key} {tok} (the tools gave {key} " + ", ".join(f"{v:g}" for v in dict.fromkeys(vals)) + ")")
    return list(dict.fromkeys(bad))


def unresolved_failures(turn: list) -> list[str]:
    """Tools that failed in this turn and were never afterwards called successfully."""
    calls = {tc["id"]: tc for m in turn if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    failed, fixed = {}, set()
    for m in turn:
        if not isinstance(m, ToolMessage):
            continue
        name = calls.get(m.tool_call_id, {}).get("name", m.name)
        if _text(m).startswith("Error") or getattr(m, "status", "") == "error":
            failed.setdefault(name, re.sub(r"\s+", " ", _text(m))[:160])
        elif not _text(m).startswith(REPEAT_NOTE[:20]):
            fixed.add(name)
    return [f"{n} ({err})" for n, err in failed.items() if n not in fixed]


def misattributed_key_times(turn: list, answer: str, history: list) -> list[str]:
    """'CHSH is lost at t ≈ 0.276' when the tool says CHSH reaches 2 at 0.2877 (0.276 is the
    halving time), and the same for the entanglement-death and halving times."""
    if not any(isinstance(m, ToolMessage) and m.name == "evolve_state" for m in turn):
        return []
    keys = {"chsh": [], "death": [], "half": []}
    never = {"chsh": [], "death": [], "half": []}         # the tool's own words when the event does not happen
    for m in turn:                    # only THIS question's evolutions (an earlier state has other times)
        if isinstance(m, ToolMessage) and m.name == "evolve_state":
            f = (_parse(m.content) or {}).get("features") or {}
            for k, field in (("chsh", "chsh_drops_to_2_at"), ("death", "entanglement_vanishes_at"),
                             ("half", "concurrence_halves_at")):
                if isinstance(f.get(field), (int, float)):
                    keys[k].append(float(f[field]))
                elif isinstance(f.get(field), str):
                    never[k].append(f[field])
    bad = []
    for sent in re.split(r"(?<=[.!?;])\s+|\n", answer):
        times = _TIME.findall(sent) or re.findall(r"\(t\s*[≈=]\s*(\d+\.\d+)\)", sent)
        if len(times) != 1:
            continue
        tok = times[0]
        if re.search(r"half|halv", sent, flags=re.I):
            kind = "half"
        elif re.search(r"CHSH|Bell|non-?local", sent) and re.search(
                r"drop|lost|los|fall|reach|below|disappear|vanish|end", sent, flags=re.I):
            kind = "chsh"
        elif re.search(r"entangle|concurrence|negativity", sent, flags=re.I) and re.search(
                r"die|death|dead|vanish|zero|disappear", sent, flags=re.I):
            kind = "death"
        else:
            continue
        name = {"chsh": "CHSH reaches 2", "death": "entanglement vanishes", "half": "concurrence halves"}[kind]
        if not keys[kind]:
            if never[kind] and not re.search(r"\b(never|not|no)\b|stays? above", sent, flags=re.I):
                bad.append(f"t ≈ {tok} is given for '{name}', but the tool says: {never[kind][0]}")
            continue
        d = len(tok.split(".")[1]) if "." in tok else 0
        if not any(abs(float(tok) - v) <= 0.5 * 10 ** -d + 1e-9 for v in keys[kind]):
            bad.append(f"t ≈ {tok} is given for '{name}', but the tool says {name} at t = "
                       + " or ".join(f"{v:.4f}" for v in keys[kind]))
    return bad


_TOPIC_TOOLS = [  # (question pattern, topic, tools any of which answers it)
    (r"\bBB84\b|\bQKD\b|secret key|key rate|key fraction", "QKD / secret keys", ("qkd_key_rate", "plot_sweep")),
    (r"teleport", "teleportation", ("teleportation_fidelity", "plot_sweep")),
    (r"bit[- ]?flip code|repetition code", "the bit-flip code", ("bit_flip_code",)),
    (r"\bsteer", "EPR steering", ("certify_correlations", "plot_sweep", "plot_bloch_and_correlations")),
    (r"wave ?plate|\bHWP\b|\bQWP\b|retarder", "wave plates",
     ("apply_waveplate", "compensate_phase", "plot_sweep")),
    (r"\bplot|\bgraph\b|\bfigure\b|\bdraw\b|\bvisuali[sz]|\bshow me\b.{0,40}\b(curve|fringe|bloch|matrix)",
     "a plot", ("plot_sweep", "plot_density_matrix", "plot_measures_vs_noise", "plot_bloch_and_correlations",
                "plot_fringes", "plot_tomography_fit", "plot_data_file", "evolve_state")),
]


def required_tools_missing(question: str, turn: list, answer: str) -> list[tuple[str, str]]:
    """Topics of a computational request (at least one tool was used) that the dedicated tool never
    answered, e.g. 'can a secret key be made' answered from the fidelity alone."""
    called = {tc["name"] for m in turn if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    if not called or re.search(r"can.?t do that yet|no tool", answer, re.I):
        return []
    return [(topic, tools[0] if topic != "a plot" else "the matching plot tool")
            for pat, topic, tools in _TOPIC_TOOLS if re.search(pat, question, re.I) and not called & set(tools)]


_SIGNIFICANCE = re.compile(r"significan|standard deviation|\bsigma\b|σ|how sure|confiden", re.I)


def significance_not_certified(question: str, turn: list) -> list[str]:
    """States reconstructed from data in this turn that a significance question needs
    certify_correlations for. Simulated data carry their own CHSH nσ, so they need it only when
    the question is about entanglement or steering too."""
    if not _SIGNIFICANCE.search(question):
        return []
    made = []
    for m in turn:
        if isinstance(m, ToolMessage):
            d = _parse(m.content) or {}
            if m.name == "reconstruct_state_from_counts" and d.get("handle"):
                made.append(d["handle"])
            elif m.name == "simulate_tomography" and re.search(r"entangl|steer|witness", question, re.I):
                h = (d.get("reconstructed_raw") or {}).get("handle")
                if h:
                    made.append(h)
    ok = {m.tool_call_id for m in turn if isinstance(m, ToolMessage) and not _text(m).startswith("Error")}
    certified = {tc["args"].get("handle") for m in turn if isinstance(m, AIMessage)
                 for tc in (m.tool_calls or []) if tc["name"] == "certify_correlations" and tc["id"] in ok}
    return [h for h in dict.fromkeys(made) if h not in certified]


def completion_and_claim_notes(question: str, turn: list, answer: str,
                               history: list | None = None) -> list[tuple[str, str]]:
    """Problems an answer can have even when all its numbers are right: [(key, note)]."""
    notes = []
    history = history if history is not None else turn
    lost_calls = unresolved_failures(turn)
    admits = re.search(r"fail|could ?n.t|cannot|can.t|error|invalid|not possible|unable|must be|\?",
                       answer, flags=re.I)
    if lost_calls and not admits:          # fine if the answer tells the user / asks back
        notes.append(("failed tool", "these tool calls FAILED and were never redone: " + "; ".join(lost_calls)
                      + ". Your answer must not report results of a failed call. Call the tool again with "
                      "correct arguments (use the handles listed in 'States created so far'), or tell the "
                      "user plainly that this step failed."))
    sd = sudden_death_contradiction(turn, answer)
    if sd:
        notes.append(("sudden death", sd + ". Rewrite the COMPLETE answer so it agrees with the tool."))
    wrong_key = misattributed_key_times(turn, answer, history)
    if wrong_key:
        notes.append(("key time", "; ".join(wrong_key) + ". Use each time only for the event the tool "
                      "names in 'at_key_times'. Rewrite the COMPLETE answer."))
    rows = wrong_row_values(answer, history)
    if rows:
        notes.append(("row values", "these table values are not values of the state named in their row: "
                      + "; ".join(rows) + ". For a state at the end of an evolution use 'at_t_max'; for "
                      "any state you can call compute_measures(handle). Rewrite the COMPLETE answer."))
    quoted = quoted_values_not_computed(answer, history)
    if quoted:
        notes.append(("quoted value", "your answer quotes " + "; ".join(quoted) + ". Quote a measure only "
                      "with a value a tool returned for that state; if it was not computed, call "
                      "compute_measures(handle) first, or leave the value out. Rewrite the COMPLETE answer."))
    prow = wrong_parameter_rows(answer, turn)
    if prow:
        notes.append(("parameter rows", "these table cells are not the tool's values at the parameter of "
                      "their row: " + "; ".join(prow[:4]) + ". Every table cell must be an exact tool value at "
                      "a point listed in 'samples', 'at_key_points', 'at_key_times' or a crossing's "
                      "'values_there' (no '~', '>0' or estimates). Rewrite the COMPLETE answer."))
    missing = bell_states_missing(question, turn)
    if missing:
        notes.append(("missing", "the user's request names " + ", ".join(missing) + " but no tool "
                      "in this turn created or used " + ("it" if len(missing) == 1 else "them")
                      + ". Create " + ", ".join(f"create_state(kind='{k}')" for k in missing)
                      + " and apply the same tool you used for the other states, then give ONE "
                      "final answer with ALL results."))
    unused = unused_new_states(turn)
    if unused:
        notes.append(("never used", "the user's request is not finished: you created "
                      + ", ".join(unused) + " but never used " + ("it" if len(unused) == 1 else "them")
                      + ", while you used the other new states. Call now: "
                      + suggest_missing_calls(turn, unused)
                      + ". Then give ONE final answer that reports the results of ALL the calls."))
    wrong_t = times_not_in_tools(question, turn, answer, history)
    if wrong_t:
        notes.append(("time", "your answer shows values at t = " + ", ".join(wrong_t) + ", but the tool "
                      "gave values only at its 'samples' and 'at_key_times'. Relabelled times are wrong. "
                      "Rewrite the COMPLETE answer using only those times (the exact time-evolution table "
                      "is already shown to the user, so you need not repeat it)."))
    for topic, tool_name in required_tools_missing(question, turn, answer):
        notes.append((f"no {tool_name}", f"the user asks about {topic}, but no {tool_name} call was made in "
                      f"this turn, so your answer about it is not backed by a calculation. Call {tool_name} "
                      "now on the relevant state(s) and give ONE final answer that reports its numbers."))
    uncertified = significance_not_certified(question, turn)
    if uncertified:
        notes.append(("significance", "the user asks about statistical significance, but certify_correlations "
                      "was not called on the state(s) made from data in this turn: "
                      + ", ".join(uncertified) + ". Call now: "
                      + "; ".join(f"certify_correlations(handle='{h}')" for h in uncertified)
                      + ", then give ONE final answer quoting its n_sigma values (a value above the bound "
                      "alone does not make it significant)."))
    sim = simulation_settings_differ(question, turn)
    if sim:
        notes.append(("simulation settings", "your simulation did not use the settings the user asked for: "
                      + "; ".join(sim) + ". Call simulate_tomography again with exactly the user's values "
                      "and answer from that result only."))
    misbound_v = channel_parameters_misbound(question, turn, with_values=True)
    misbound = [w for _v, w in misbound_v]
    if misbound:
        notes.append(("binding", "; ".join(misbound) + ". The numbers you quote then describe a different "
                      "experiment from the one the user asked for. Redo those calculations with exactly the "
                      "user's values, each on the channel it was given for, and answer from those results only."))
    lost = [x for x in parameters_not_used(question, turn)            # a misbound value is told once, above
            if not any(abs(float(x) - v) < 1e-9 for v, _w in misbound_v)]
    if lost:
        notes.append(("parameter", "the user's request contains the value(s) " + ", ".join(lost)
                      + " but no tool call in this turn used " + ("it" if len(lost) == 1 else "them")
                      + " (e.g. a Werner p, a noise parameter or a time was skipped). If it is a "
                      "parameter of the request, call the tools for it now and give ONE final answer "
                      "with ALL results; if it is only a threshold to compare with, keep your answer."))
    tool_text = " ".join(_text(m) for m in turn if isinstance(m, ToolMessage)).lower()
    for word, key, tool_hint in (("fidelit", "fidelity", "compare_states"),
                                 ("trace distance", "trace_distance", "compare_states")):
        if word in answer.lower() and key not in tool_text and _NUM.search(answer):
            notes.append((f"no {key} tool", f"your answer quotes a {key.replace('_', ' ')}, but no tool in "
                          f"this turn computed one. Call {tool_hint}(handle_a, handle_b) (create the "
                          "reference state first if needed) and use its value; if you are quoting a "
                          "result from an earlier question, say so explicitly."))
    return notes


_PARAM = re.compile(r"(?<![\w.])(\d+\.\d+)(?!\w|\.\d)")      # '0.8.' at a sentence end counts


def parameters_not_used(question: str, turn: list) -> list[str]:
    """Decimal values written in the question that no tool call of this turn received
    (e.g. 'Werner states with p = 0.3, 0.5 and 0.8' but only 0.3 and 0.5 were created)."""
    ok_ids = {m.tool_call_id for m in turn if isinstance(m, ToolMessage)
              and getattr(m, "status", "") != "error" and not _text(m).startswith("Error")}
    calls = [tc for m in turn if isinstance(m, AIMessage) for tc in (m.tool_calls or []) if tc["id"] in ok_ids]
    if not calls:
        return []                                  # nothing succeeded (e.g. a refused guess): no nagging
    used = {float(x) for tc in calls for x in re.findall(r"\d+(?:\.\d+)?", json.dumps(tc["args"]))}
    asked = list(dict.fromkeys(_PARAM.findall(question)))
    return [x for x in asked if not any(abs(float(x) - u) < 1e-9 for u in used)]


# noise channels as people write them -> the channel names of apply_local_noise / apply_global_noise.
# Global and collective names are matched first and masked, so 'global depolarizing' is never read as local.
_GLOBAL_CHANNEL_PHRASES = [
    (r"(?:global|two[- ]qubit)\s+depolari[sz]\w*", "global_depolarizing"),
    (r"global\s+(?:dephasing|phase[- ]damping)", "global_dephasing"),
    (r"(?:collective|correlated)\s+dephasing", "collective_dephasing"),
    (r"(?:collective\s+(?:amplitude\s+)?(?:damping|decay)|global\s+amplitude\s+damping)", "collective_amplitude_damping"),
    (r"correlated\s+bit[- ]phase[- ]flip", "correlated_bit_phase_flip"),
    (r"correlated\s+bit[- ]flip", "correlated_bit_flip"),
    (r"correlated\s+phase[- ]flip", "correlated_phase_flip"),
]
_LOCAL_CHANNEL_PHRASES = [
    (r"(?:local\s+)?depolari[sz]\w*", "depolarizing"),
    (r"amplitude[- ]damping", "amplitude_damping"),
    (r"phase[- ]damping", "phase_damping"),
    (r"bit[- ]phase[- ]flip", "bit_phase_flip"),
    (r"bit[- ]flip", "bit_flip"),
    (r"phase[- ]flip", "phase_flip"),
]
_NOISE_TOOLS = ("apply_local_noise", "apply_global_noise", "get_kraus_operators")


def requested_channel_parameters(question: str) -> list[tuple[str, float]]:
    """(channel, value) pairs the question asks for: each decimal value is bound to the nearest noise channel
    named before it in the same sentence ('global depolarizing noise with p = 0.3 and local depolarizing
    with p = 0.1' -> [('global_depolarizing', 0.3), ('depolarizing', 0.1)])."""
    text = question.lower()
    mentions = []
    for phrases in (_GLOBAL_CHANNEL_PHRASES, _LOCAL_CHANNEL_PHRASES):
        for pat, name in phrases:
            for m in re.finditer(pat, text):
                mentions.append((m.start(), m.end(), name))
                text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]   # mask it
    pairs = []
    for m in _PARAM.finditer(question):
        before = [x for x in mentions if x[1] <= m.start() and m.start() - x[1] <= 80
                  and not re.search(r"[;!?]|\.(?!\d)", question[x[1]:m.start()])]
        if before:
            pairs.append((max(before)[2], float(m.group(1))))
    return list(dict.fromkeys(pairs))


def channel_parameters_misbound(question: str, turn: list, with_values: bool = False) -> list:
    """A channel the user gave a value for WAS applied, but never with that value (e.g. 'global p = 0.3,
    local p = 0.1' run as global 0.1 and local 0.3: every value is used somewhere, so parameters_not_used
    passes, yet each state answers the wrong experiment)."""
    import agent_tools as at
    ok_ids = {m.tool_call_id for m in turn if isinstance(m, ToolMessage)
              and getattr(m, "status", "") != "error" and not _text(m).startswith("Error")}
    used = {}
    for m in turn:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if tc["name"] in _NOISE_TOOLS and tc["id"] in ok_ids and "channel" in tc["args"]:
                    try:
                        used.setdefault(at._channel_name(str(tc["args"]["channel"])), set()).add(
                            float(tc["args"].get("param")))
                    except (TypeError, ValueError):
                        continue
    wrong = []
    for name, value in requested_channel_parameters(question):
        got = used.get(name)
        if got and not any(abs(g - value) < 1e-9 for g in got):
            words = name.replace("_", " ")
            wrong.append((value, f"{words} with parameter {value:g} was asked for, but the tool calls used "
                                 f"{words} with " + ", ".join(f"{g:g}" for g in sorted(got))))
    return [w for _v, w in wrong] if not with_values else wrong


_SIM_SETTINGS = [  # (pattern in the question, simulate_tomography arguments it sets)
    (r"pair\s+rate\s+(?:of\s+)?(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", ("pair_rate",)),
    (r"efficienc\w*\s+(?:of\s+)?(\d*\.\d+|\d+)", ("eta_a", "eta_b")),
    (r"(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+dark\s+counts", ("dark_a", "dark_b")),
    (r"(\d+(?:\.\d+)?)\s*ns\b", ("window_ns",)),
    (r"(\d+(?:\.\d+)?)\s*s\s+(?:per|for\s+each|each)\s+setting", ("time_per_setting_s",)),
]


def simulation_settings_differ(question: str, turn: list) -> list[str]:
    """The last successful simulate_tomography used other detector settings than the user asked for
    (e.g. 1 s per setting instead of 0.1 s)."""
    ok = {m.tool_call_id for m in turn if isinstance(m, ToolMessage) and m.name == "simulate_tomography"
          and not _text(m).startswith("Error")}
    calls = [tc for m in turn if isinstance(m, AIMessage) for tc in (m.tool_calls or [])
             if tc["name"] == "simulate_tomography" and tc["id"] in ok]
    if not calls:
        return []
    args = calls[-1]["args"]
    wrong = []
    for pat, names in _SIM_SETTINGS:
        mm = re.search(pat, question, re.I)
        if not mm:
            continue
        want = float(mm.group(1))
        for n in names:
            got = args.get(n)
            if isinstance(got, str):
                try:
                    got = float(got)
                except ValueError:
                    got = None
            if got is None or abs(float(got) - want) > 1e-9 * max(1.0, abs(want)):
                wrong.append(f"{n}={got if got is not None else '(default)'} instead of {mm.group(1)}")
    return wrong


def required_notes(turn: list, answer: str) -> list[str]:
    """Caveats a tool marked as essential but the answer left out; appended verbatim (no re-ask):
    the R/L-convention warning of compensate_phase (the other convention needs another angle) and
    the optimal-settings caveat of certify_correlations when the answer discusses CHSH."""
    notes = []
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "certify_correlations":
            d = _parse(m.content) or {}
            cav = d.get("caveats") or []
            chsh = [t for t in d.get("tests", []) if "CHSH" in t.get("property", "")]
            from_data_violating = bool(chsh and "n_sigma" in chsh[0] and chsh[0].get("holds"))
            if (cav and from_data_violating and re.search(r"CHSH", answer)
                    and not re.search(r"optimal|setting", answer, re.I)):
                notes.append("Note: " + cav[0] + ".")
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "plot_sweep" and re.search(r"steer", answer, re.I):
            sn = (_parse(m.content) or {}).get("steering_note")
            need = r"1/2|0\.5(?!\d)|½" if sn and "Werner" in sn else r"sufficient|not necessary|unsteerable"
            if sn and not re.search(need, answer, re.I):
                notes.append("Steering: " + sn + ".")
    for m in turn:
        if isinstance(m, ToolMessage) and m.name == "compensate_phase":
            warn = (_parse(m.content) or {}).get("convention_warning")
            alt = re.search(r"HWP angle is (\d+(?:\.\d+)?)°", warn or "")
            if warn and alt and alt.group(1) not in answer:
                notes.append("R/L convention: " + warn[0].upper() + warn[1:])
    return list(dict.fromkeys(notes))


_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u00a0": " ", "\u202f": " ",
                         "\u2009": " ", "\u2007": " "})


_META_START = re.compile(r"^\s*(you(['’]re| are)\s+(right|correct)|(i\s+)?apologi[sz]e|sorry|my\s+(mistake|apologies)"
                         r"|good\s+catch|thanks?\b|thank\s+you)", re.I)
_META_SENTENCE = re.compile(r"^\s*(let\s+me\s+(now\s+)?(correct|rewrite|redo|fix|report|give|recompute|provide|present|"
                            r"summari[sz]e|restate|write)\b|i(['’]ll|\s+will)\s+(now\s+)?(correct|rewrite|report|give)|here\s+(are|is)\s+the\s+"
                            r"(results?|answer|corrected)|(the\s+)?(state|states|data)\s+(has|have|was|were)\s+(now\s+)?(been\s+)?"
                            r"(reconstructed|created|computed|certified|plotted|simulated)[^.]*\.?\s*$"
                            r"|i(['’]ve|\s+have)\s+(now\s+)?(reconstructed|created|computed|certified|plotted|simulated)[^.]*\.?\s*$"
                            r"|#+\s*answer\s*$)", re.I)


_SELF_CHECK = re.compile(r"\b(?:wait|hmm+)\b\s*[,.!—–-]\s*(?:let\s+me\s+(?:re-?)?check(?:\s+(?:this|that|again))?\s*[:,.]?\s*)?"
                         r"|\blet\s+me\s+(?:re-?)?check(?:\s+(?:this|that|again))?\s*:\s*"
                         r"|\blet\s+me\s+(?:note|state|restate|recompute|re-?calculate|verify|be\s+precise|correct\s+(?:this|that))"
                         r"(?:\s+(?:the\s+)?(?:actual\s+|exact\s+|correct\s+)?(?:numbers|values|this|that))?(?:\s+again)?\s*:\s*", re.I)
_BULLET_LABEL = re.compile(r"^(\s*(?:[-*•]|\d+\.)\s+(?:\*\*[^*\n]+\*\*\s*[:—–-]\s*)?)")
_ASKED_AND_DENIED = re.compile(r"(\(|:\s+|^)(?:and\s+)?[^?!:\n()]{0,30}[<>≤≥][^?!:\n()]{0,15}\?\s*No\s*[—–-]+\s*")


def strip_self_talk(text: str) -> str:
    """Remove a model thinking aloud ('CHSH max = 2.263 > 2√2? No — … Wait, let me check: …'): keep what
    follows the last self-check of a line, and drop 'X? No —' fragments."""
    out = []
    for line in text.split("\n"):
        marks = list(_SELF_CHECK.finditer(line))
        if marks:
            lab = _BULLET_LABEL.match(line)
            prefix = lab.group(1) if lab else ""
            rest = line[marks[-1].end():].lstrip()
            line = prefix + (rest[:1].upper() + rest[1:] if not prefix.strip().endswith(":") else rest)
        line = _ASKED_AND_DENIED.sub(lambda m: m.group(1), line)
        out.append(line)
    return "\n".join(out)


_META_TO_CHECKER = re.compile(r"\b(the\s+user['’]?s?|the\s+verifier|the\s+answer\s+check)\b", re.I)
_META_ABOUT_SELF = re.compile(r"\b(i\s+keep\s+my\s+answer|my\s+(previous\s+|original\s+)?answer|"
                              r"(need|needed|have)\s+to\s+call\s+(a|the|any)\s+tool|call\s+a\s+tool\s+for|"
                              r"tool\s+results\s+i\s+(already\s+)?have)\b", re.I)


def strip_meta(text: str) -> str:
    """Remove a reply to the answer check that leaked into the answer ('You're right - I computed those
    sums myself. Let me rewrite...', 'The state has been reconstructed. Here are the results.'), and any
    paragraph that talks ABOUT the user to the checker ('... are the user's claimed values ..., not
    parameters I need to call a tool for. So I keep my answer.')."""
    parts = re.split(r"(\n\s*\n)", text)
    kept = [x for x in parts if x.lstrip().startswith("|") or
            not (_META_TO_CHECKER.search(x) and _META_ABOUT_SELF.search(x))]
    if len(kept) != len(parts) and "".join(kept).strip():
        text = re.sub(r"\n{3,}", "\n\n", "".join(kept)).lstrip("\n")
    paras = re.split(r"(\n\s*\n)", text.lstrip(), maxsplit=1)
    first, rest = paras[0], "".join(paras[1:])
    if "|" in first:
        return text
    if _META_START.match(first):
        return rest.lstrip() or text                   # the whole first paragraph speaks to the checker
    sentences = re.split(r"(?<=[.!?])\s+", first.strip())
    kept = [x for x in sentences if not _META_SENTENCE.match(x)]
    if len(kept) == len(sentences):
        return text
    return (" ".join(kept) + rest) if kept else (rest.lstrip() or text)


_COUNTS = re.compile(r"(?<![\w.,/^-])(\d{2,})(?![\w.]|,\d|/)")


def ungrounded_counts(answer: str, messages: list) -> list[str]:
    """Whole numbers >= 10 in table rows or in 'measured N, fit M' that no tool result or question contains
    (e.g. counts invented for settings the tool did not list)."""
    sources = []
    for m in messages:
        if isinstance(m, ToolMessage) or _is_user_question(m):
            sources += [abs(float(t)) for t in re.findall(r"\d+(?:\.\d+)?", _text(m).replace(",", ""))]
    pieces = [ln for ln in answer.split("\n") if ln.lstrip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", ln)]
    pieces += [mm.group(0) for mm in re.finditer(r"measured\s+\d+[^.\n|]{0,20}?fit\s+\d+", answer, re.I)]
    bad = []
    for piece in pieces:
        piece = re.sub(r"(?<=\d),(?=\d{3}\b)", "", piece)            # 16,549 -> 16549
        piece = re.sub(r"rho_\d+|ρ[₀-₉]+|\bT\d+\b|[×x·]\s*10(?=[⁰¹²³⁴⁵⁶⁷⁸⁹⁻])|\d+(?:\.\d+)?\s*[–-]\s*\d+(?:\.\d+)?\s*%"
                       r"|\d+(?:\.\d+)?\s*[°%σ]|\d+\.\d+|\d+(?=[⁰¹²³⁴⁵⁶⁷⁸⁹⁻])", " ", piece)
        for tok in _COUNTS.findall(piece):
            if not any(abs(float(tok) - v) <= 0.5 for v in sources):
                bad.append(tok)
    return list(dict.fromkeys(bad))


def _normalize(text: str) -> str:
    """Non-breaking hyphens and spaces -> ASCII (they look the same but defeat every regex check)."""
    return text.translate(_DASHES)


def answer_check(state: AgentState):
    """Run the answer checks. A check must never crash the program: if one raises an error (a bug in a
    check, e.g. a state type it does not know), the answer is shown with a short note instead."""
    try:
        return _answer_check(state)
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).exception("answer check failed")
        final = state["messages"][-1]
        note = (f"\n\n*(The automatic answer check could not run ({type(e).__name__}); this answer was "
                "not checked.)*")
        return {"messages": [AIMessage(content=_text(final) + note, id=final.id)]}


def _answer_check(state: AgentState):
    """Reject a final answer that contains numbers no tool produced (e.g. interpolated values)."""
    msgs = state["messages"]
    start = max((i for i, m in enumerate(msgs) if _is_user_question(m)), default=0)
    final = msgs[-1]
    if getattr(final, "name", None) in (BUDGET_STOP, HELP_REPLY):
        return {}
    text = strip_self_talk(strip_meta(_normalize(_text(final))))
    if text != _text(final):            # e.g. gpt-oss writes 'Bell‑local' with U+2011: checks need '-'
        final = AIMessage(content=text, id=final.id, tool_calls=[])
    earlier = " ".join(_text(m) for m in msgs[start:] if _is_verifier(m))
    if not text.strip() and "[empty]" not in earlier:   # some models end with an empty message
        return {"messages": [HumanMessage("VERIFIER: [empty] your last reply was empty. Give the final answer "
                                          "to the user's question now, from the tool results above.")]}
    bad = ungrounded_numbers(_text(final), msgs[:-1]) + ungrounded_counts(_text(final), msgs[:-1])
    if not bad:
        turn = msgs[start:-1]
        earlier_notes = " ".join(_text(m) for m in msgs[start:] if _is_verifier(m))
        notes = completion_and_claim_notes(_text(msgs[start]), turn, _text(final), msgs[:-1])
        for key, note in notes:
            if f"[{key}]" not in earlier_notes:   # each kind of note at most once per question
                return {"messages": [HumanMessage(f"VERIFIER: [{key}] {note}")]}
        unresolved = [unresolved_warning(key, _text(msgs[start]), turn) for key, _note in notes]
        fixed, reasons = remove_false_claims(_text(final), msgs[:-1])
        if reasons:
            fixes = ({c[2] for c in _CORRECTIONS + _CIRCULAR_AZIMUTH} | {_PROVES_NONLOCAL[2], _CHSH_HALF_WHY}
                     | {c[2] for c in _compensation_rules(msgs[:-1])[1] + _oscillation_correction(msgs[:-1])})
            verb = ("corrected" if set(reasons) <= fixes else
                    "removed or corrected" if set(reasons) & fixes else "removed")
            fixed += (f"\n\n*(Answer check {verb} an incorrect statement: " + "; ".join(reasons) + ".)*")
        extra = required_notes(turn, fixed)
        if extra:
            fixed += "\n\n" + "\n".join(f"*{x}*" for x in extra)
        if unresolved:                         # the model was told and did not fix it: tell the user
            fixed += "\n\n" + "\n".join(f"*(Answer check warning: {w}.)*" for w in dict.fromkeys(unresolved))
        if reasons or extra or unresolved or text != _text(msgs[-1]):
            return {"messages": [AIMessage(content=fixed, id=final.id)]}
        return {}
    retries = state.get("retries", 0) + 1
    if retries >= MAX_RETRIES:      # give up gracefully: show the answer with a warning
        warn = ("\n\n[Warning: these numbers were not produced by any tool and may be wrong: "
                + ", ".join(bad) + "]")
        return {"retries": retries,
                "messages": [AIMessage(content=_text(final) + warn, id=final.id)]}
    note = ("VERIFIER: your answer contains numbers that no tool produced: " + ", ".join(bad)
            + ". Rewrite the COMPLETE answer (opening sentence, table with every value the user "
            "asked for, bullets) using only numbers that appear in the tool results; delete only "
            "the others. Call a tool again only if the user explicitly asked for that value.")
    return {"retries": retries, "messages": [HumanMessage(note)]}


_UNRESOLVED_PLAIN = {
    "failed tool": "a calculation failed and was not redone",
    "missing": "a state named in the question was not computed",
    "never used": "a state was created but never used",
    "no certify_correlations": "the statistical significance was not computed",
    "significance": "the statistical significance was not computed",
    "simulation settings": "the simulation did not use the requested settings",
}


def unresolved_warning(key: str, question: str, turn: list) -> str:
    """What the user is told when the model was sent a note about this problem and did not fix it."""
    if key == "parameter":
        vals = parameters_not_used(question, turn)
        return ("the value(s) " + ", ".join(vals) + " from the question were not used in any calculation, so "
                "the numbers above may answer a different setting" if vals else "a value from the question was not used")
    if key == "binding":
        return "; ".join(channel_parameters_misbound(question, turn)) + ", so the numbers above answer a different experiment"
    if key.startswith("no "):
        return f"a calculation the question needs ({key[3:]}) was not run"
    return _UNRESOLVED_PLAIN.get(key, "some values above may not match the calculations")


def route_after_check(state: AgentState):
    last = state["messages"][-1]
    return "agent" if isinstance(last, HumanMessage) and str(last.content).startswith("VERIFIER:") else END


def build_graph(llm):
    g = StateGraph(AgentState)
    g.add_node("agent", make_agent_node(llm))
    g.add_node("tools", make_tools_node())
    g.add_node("verifier", verifier)
    g.add_edge(START, "agent")
    g.add_node("answer_check", answer_check)
    g.add_conditional_edges("agent", route_after_agent, ["tools", "answer_check"])
    g.add_conditional_edges("answer_check", route_after_check, ["agent", END])
    g.add_edge("tools", "verifier")
    g.add_edge("verifier", "agent")
    return g.compile(checkpointer=MemorySaver())
