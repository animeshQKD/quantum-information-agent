"""
run.py - chat with the agent in the terminal (nicely formatted with `rich`).
    python run.py              normal view: short step list + formatted answer
    python run.py --details    also show the raw tool calls and results
While chatting, type  details on / details off  to switch, and  quit  to exit.
Figures are saved in the figures/ folder.
"""
import sys
import time

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

import agent_tools
import graph
import llm
import plots
import routing
from agent_tools import TOOLS
from display import result_panels, summarize_steps, to_unicode
from graph import build_graph
from llm import USAGE, AllProvidersFailed, get_llm

EXPECTED = "2026-11-02"
stale = [m.__name__ + ".py" for m in (graph, agent_tools, llm, plots, routing)
         if getattr(m, "FILES_VERSION", None) != EXPECTED]
if stale:
    raise SystemExit(f"Out-of-date file(s): {', '.join(stale)}. Replace them with the latest versions.")

console = Console()


def text_of(msg) -> str:
    """Plain text of a reply (Gemini returns a list of text blocks, Groq a string)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


def show_raw(new_messages):
    for msg in new_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                console.print(f"    [dim]call   {tc['name']}({tc['args']})[/]", markup=True, highlight=False)
        elif isinstance(msg, ToolMessage):
            console.print(f"    [dim]result {str(msg.content)[:200]}[/]", highlight=False)


def main():
    details = "--details" in sys.argv
    app = build_graph(get_llm(TOOLS))
    config = {"configurable": {"thread_id": "session-1"}, "recursion_limit": 60}
    console.print(Panel.fit(
        "[bold]Quantum information agent[/]\n"
        "Ask in plain English, e.g. [italic]Create phi+, apply phase damping 0.7 to both qubits "
        "and compute its measures[/].\n"
        "Commands: [bold]details on[/] / [bold]details off[/] (show tool calls), [bold]quit[/].",
        border_style="cyan"))

    while True:
        q = console.input("\n[bold cyan]You ›[/] ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        if q.lower() in {"details on", "details off"}:
            details = q.lower().endswith("on")
            console.print(f"[dim]Details {'on' if details else 'off'}.[/]")
            continue

        before = app.get_state(config).values or {}
        seen, old_figs = len(before.get("messages", [])), len(before.get("figures", []))
        t0 = time.time()
        try:
            with console.status("[cyan]Thinking…[/]", spinner="dots"):
                out = app.invoke({"messages": [("user", q)], "retries": 0}, config)
        except AllProvidersFailed as e:
            console.print(Panel(str(e), title="No model available", border_style="red"))
            continue
        except GraphRecursionError:
            # safety net (graph.py normally stops itself after MAX_MODEL_CALLS): show what was done
            done = (app.get_state(config).values or {}).get("messages", [])[seen + 1:]
            for kind, sentence in summarize_steps(done):
                console.print(f"  {'[green]✓[/]' if kind == 'ok' else '[yellow]![/]'} {escape(to_unicode(sentence))}",
                              highlight=False)
            console.print(Panel("The agent went round in circles and was stopped. The steps above show "
                                "what the tools computed. Please ask again, one task at a time.",
                                title="Stopped", border_style="red"))
            continue
        except Exception as e:  # noqa: BLE001 - a bug must not end the session: report it and go on
            import traceback
            console.print(Panel(f"{type(e).__name__}: {escape(str(e))}\n\n[dim]{escape(traceback.format_exc()[-1500:])}[/]",
                                title="Internal error (please send this to the developer)", border_style="red"))
            continue

        new = out["messages"][seen + 1:]
        for kind, sentence in summarize_steps(new):
            mark = "[green]✓[/]" if kind == "ok" else "[yellow]![/]"
            console.print(f"  {mark} {escape(to_unicode(sentence))}", highlight=False)   # escape: keep "[from ...]"
        if details:
            show_raw(new)

        for title, table in result_panels(new):    # exact numbers, straight from the tools
            console.print(Panel(Markdown(table), title=f"[bold]{title}[/]", title_align="left",
                                border_style="green", padding=(0, 1)))

        answer = to_unicode(text_of(out["messages"][-1]))
        who = ("no model needed · " if getattr(out["messages"][-1], "name", None) == graph.HELP_REPLY else
               f"{USAGE['last_provider']} · " if USAGE["last_provider"] else "")
        footer = f"{who}{time.time() - t0:.0f} s"
        if USAGE["openai_calls"]:
            footer += f" · paid OpenAI calls: {USAGE['openai_calls']}"
        console.print(Panel(Markdown(answer), title="[bold]Answer[/]", title_align="left",
                            subtitle=f"[dim]{footer}[/]", subtitle_align="right",
                            border_style="blue", padding=(1, 2)))
        for path in out.get("figures", [])[old_figs:]:
            console.print(f"  [magenta]Figure saved:[/] {escape(path)}")


if __name__ == "__main__":
    main()
