"""
benchmark.py - score the REAL agent (real LLMs) on fixed tasks with known answers.

    python benchmark.py                                  all tasks, provider order from .env
    python benchmark.py --order groq,gemini,groq_backup,ollama
    python benchmark.py --only T01,T05,T11 --pause 20
    python benchmark.py --compare results/A.json results/B.json

Each task runs in a fresh session. A task passes when every check passes:
  numbers   each expected value appears in the answer (at the precision it is written)
  tools     the required tools were called
  text      required statements are present, forbidden ones absent (incl. global list)
  guards    e.g. it asked back instead of guessing, or created no state
Results go to results/benchmark_<date>_<time>.json and .md. The free Groq tier allows only a
few calls per minute, so --pause (seconds between tasks, default 15) keeps it from failing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime

TASKS_FILE = "benchmark_tasks.json"
_TOKEN = re.compile(r"([-−–]?)\s?(\d+(?:\.\d+)?)")


# ------------------------------------------------------------------ scoring (pure functions)
def _decimals(x) -> int:
    s = repr(float(x))
    return 0 if s.endswith(".0") else len(s.split(".")[1])


def number_found(text: str, value: float, min_decimals: int | None = None, sign: str | None = None) -> bool:
    """True if `value` appears in `text` at the precision it is written there.
    A token with d decimals matches if |token - value| <= 0.5*10^-d; tokens with fewer than
    `min_decimals` decimals do not count (default: min(2, decimals of the expected value))."""
    need = min(2, _decimals(value)) if min_decimals is None else min_decimals
    for m in _TOKEN.finditer(text.replace(",", "")):
        neg, tok = m.group(1), m.group(2)
        d = len(tok.split(".")[1]) if "." in tok else 0
        if d < need:
            continue
        if abs(float(tok) - abs(value)) <= 0.5 * 10 ** -d + 1e-9:
            if sign == "+" and neg:
                continue
            if sign == "-" and not neg:
                continue
            return True
    return False


_FOOTNOTE = re.compile(r"\s*\*\(Answer check (?:removed|corrected|removed or corrected) an incorrect statement:"
                       r"[\s\S]*?\.\)\*")
_ASKS = re.compile(r"\?|\bplease (provide|specify|give|write|tell|clarify|state)\b|\bcould you\b|\bcan you\b",
                   re.I)


def was_corrected(answer: str) -> bool:
    """The answer check had to remove or correct a statement (the final answer is fine, but the
    model made a mistake first): reported separately, not as a failure."""
    return _FOOTNOTE.search(answer) is not None


def score_task(task: dict, answer: str, tables: str, tool_names: list[str], states_created: int,
               global_forbidden: list[str]) -> list[dict]:
    """[{check, ok, detail}] for one task. The answer check's own footnote is not scored (it
    quotes the wrong statement it removed); was_corrected() reports it instead."""
    answer = _FOOTNOTE.sub("", answer)
    checks = []
    where = task.get("where", "answer")
    hay = answer if where == "answer" else answer + "\n" + tables
    for n in task.get("numbers", []):
        ok = number_found(hay, n["value"], n.get("min_decimals"), n.get("sign"))
        checks.append({"check": f"number {n['label']} = {n['value']}", "ok": ok,
                       "detail": "" if ok else "not found in the answer"})
    for t in task.get("tools", []):
        ok = t in tool_names
        checks.append({"check": f"tool {t}", "ok": ok, "detail": "" if ok else "never called"})
    for pat in task.get("text", []):
        ok = re.search(pat, answer, flags=re.I) is not None
        checks.append({"check": f"says /{pat}/", "ok": ok, "detail": "" if ok else "statement missing"})
    for pat in task.get("forbidden", []) + global_forbidden:
        hit = re.search(pat, answer, flags=re.I)
        checks.append({"check": f"avoids /{pat}/", "ok": hit is None,
                       "detail": "" if hit is None else f"found: {hit.group(0)!r}"})
    if task.get("expect_question_back"):
        ok = _ASKS.search(answer) is not None
        checks.append({"check": "asks the user instead of guessing", "ok": ok, "detail": "" if ok else "no question"})
    if "max_states_created" in task:
        ok = states_created <= task["max_states_created"]
        checks.append({"check": f"creates at most {task['max_states_created']} states", "ok": ok,
                       "detail": "" if ok else f"created {states_created}"})
    if "max_tool_calls" in task:
        ok = len(tool_names) <= task["max_tool_calls"]
        checks.append({"check": f"at most {task['max_tool_calls']} tool calls", "ok": ok,
                       "detail": "" if ok else f"{len(tool_names)} calls"})
    return checks


# ------------------------------------------------------------------ running
def run_task(app, task: dict) -> dict:
    from langchain_core.messages import AIMessage
    from langgraph.errors import GraphRecursionError

    import agent_tools
    import llm
    from display import result_panels, to_unicode

    agent_tools.STORE.clear(); agent_tools.DESC.clear(); agent_tools.TOMO.clear()
    import graph
    llm.USAGE["by_provider"] = {}
    tok0 = (llm.USAGE.get("input_tokens", 0), llm.USAGE.get("output_tokens", 0))
    retries0 = graph.ROUTING["retried_with_all_tools"]
    config = {"configurable": {"thread_id": f"bench-{uuid.uuid4()}"}, "recursion_limit": 60}
    t0 = time.time()
    error = None
    try:
        out = app.invoke({"messages": [("user", task["question"])], "retries": 0}, config)
        msgs = out["messages"]
    except (llm.AllProvidersFailed, GraphRecursionError, Exception) as e:   # noqa: BLE001 - record any failure
        error = f"{type(e).__name__}: {str(e)[:200]}"
        msgs = (app.get_state(config).values or {}).get("messages", [])
    secs = time.time() - t0
    new = msgs[1:]
    tool_names = [tc["name"] for m in new if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    final = new[-1] if new and isinstance(new[-1], AIMessage) and not new[-1].tool_calls else None
    answer = to_unicode(final.content if final is not None and isinstance(final.content, str) else
                        "".join(b.get("text", "") for b in (final.content if final is not None else [])
                                if isinstance(b, dict)))
    tables = "\n".join(t for _, t in result_panels(new))
    return {"answer": answer, "tables": tables, "tool_names": tool_names,
            "tools_offered": len(graph.ROUTING["last_tools"]) or None,
            "states_created": len(agent_tools.STORE), "seconds": round(secs, 1),
            "model_calls": sum(isinstance(m, AIMessage) for m in new),
            "providers": dict(llm.USAGE["by_provider"]), "error": error,
            "input_tokens": llm.USAGE.get("input_tokens", 0) - tok0[0],
            "output_tokens": llm.USAGE.get("output_tokens", 0) - tok0[1],
            "routing_retries": graph.ROUTING["retried_with_all_tools"] - retries0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", help="provider order, e.g. groq,gemini,groq_backup,ollama")
    ap.add_argument("--only", help="comma-separated task ids, e.g. T01,T11")
    ap.add_argument("--pause", type=float, default=15, help="seconds to wait between tasks (rate limits)")
    ap.add_argument("--hf-model", help="benchmark this Hugging Face router model (sets USE_HF=1 and HF_MODEL), "
                                       "e.g. openai/gpt-oss-120b or Qwen/Qwen3-32B:cheapest")
    ap.add_argument("--no-routing", action="store_true",
                    help="send every tool and the full prompt on every call (to measure what routing saves)")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two result files")
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if a.order:
        os.environ["PROVIDER_ORDER"] = a.order
    if a.no_routing:
        os.environ["ROUTE_TOOLS"] = "0"
    if a.hf_model:
        os.environ["USE_HF"] = "1"
        os.environ["HF_MODEL"] = a.hf_model
        os.environ["HF_FALLBACK_MODEL"] = ""
        os.environ["PROVIDER_ORDER"] = "hf"          # this model only: no fallback to Groq, Gemini or OpenAI
        os.environ["PROVIDERS_ONLY"] = "1"

    from rich.console import Console
    from rich.table import Table

    import llm
    from agent_tools import TOOLS
    from graph import build_graph

    console = Console()
    spec = json.load(open(TASKS_FILE, encoding="utf-8"))
    tasks = spec["tasks"]
    if a.only:
        wanted = {x.strip().upper() for x in a.only.split(",")}
        tasks = [t for t in tasks if t["id"] in wanted]
    app = build_graph(llm.get_llm(TOOLS))
    order = llm.USAGE.get("order") or os.getenv("PROVIDER_ORDER", llm.DEFAULT_ORDER)
    results = []
    for i, task in enumerate(tasks):
        if task.get("needs_file") and not os.path.exists(task["needs_file"]):
            console.print(f"[yellow]{task['id']} skipped: {task['needs_file']} not found[/]")
            continue
        if i and a.pause:
            time.sleep(a.pause)
        console.print(f"[cyan]{task['id']}[/] {task['question'][:90]}")
        run = run_task(app, task)
        checks = score_task(task, run["answer"], run["tables"], run["tool_names"],
                            run["states_created"], spec.get("global_forbidden", []))
        if run["error"]:
            checks.append({"check": "finished without error", "ok": False, "detail": run["error"]})
        passed = all(c["ok"] for c in checks)
        corrected = was_corrected(run["answer"])
        results.append({"id": task["id"], "stage": task.get("stage", ""), "question": task["question"],
                        "passed": passed, "corrected": corrected, "checks": checks, **run})
        mark = ("[red]FAIL[/]" if not passed else "[yellow]PASS (corrected by the answer check)[/]"
                if corrected else "[green]PASS[/]")
        failed = "; ".join(f"{c['check']} ({c['detail']})" for c in checks if not c["ok"])
        retry = f" (+{run['routing_retries']} retry with all tools)" if run.get("routing_retries") else ""
        tokens = f", {run['input_tokens']:,} tokens in / {run['output_tokens']:,} out" if run.get("input_tokens") else ""
        offered = "all" if os.getenv("ROUTE_TOOLS") == "0" else (run.get("tools_offered") or "all")
        console.print(f"  {mark} {run['seconds']} s, {run['model_calls']} model calls{retry}, "
                      f"{offered} tools offered{tokens}, {run['providers']}"
                      + (f"\n  [red]{failed}[/]" if failed else ""), highlight=False)

    n_pass = sum(r["passed"] for r in results)
    n_corr = sum(r["passed"] and r.get("corrected", False) for r in results)
    n_checks = sum(len(r["checks"]) for r in results)
    ok_checks = sum(c["ok"] for r in results for c in r["checks"])
    calls = {}
    for r in results:
        for k, v in r["providers"].items():
            calls[k] = calls.get(k, 0) + v
    summary = {"date": datetime.now().isoformat(timespec="seconds"), "provider_order": order,
               "tasks": len(results), "tasks_passed": n_pass, "passed_after_correction": n_corr, "checks": n_checks, "checks_passed": ok_checks,
               "total_seconds": round(sum(r["seconds"] for r in results), 1), "calls_by_provider": calls,
               "routing": os.getenv("ROUTE_TOOLS", "1") != "0",
               "input_tokens": sum(r.get("input_tokens", 0) for r in results),
               "output_tokens": sum(r.get("output_tokens", 0) for r in results)}
    os.makedirs("results", exist_ok=True)
    stem = f"results/benchmark_{datetime.now():%Y%m%d_%H%M}"
    json.dump({"summary": summary, "results": results}, open(stem + ".json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    open(stem + ".md", "w", encoding="utf-8").write(markdown_report(summary, results))

    table = Table(title=f"Benchmark — order: {order}")
    for col in ("task", "result", "checks", "seconds", "answered by"):
        table.add_column(col)
    for r in results:
        table.add_row(r["id"], ("PASS*" if r.get("corrected") else "PASS") if r["passed"] else "FAIL",
                      f"{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])}", str(r["seconds"]),
                      ", ".join(f"{k} ×{v}" for k, v in r["providers"].items()))
    console.print(table)
    import graph
    n_calls = sum(calls.values()) or 1
    if summary["routing"]:
        console.print(f"Tool routing: {graph.ROUTING['retried_with_all_tools']} answer(s) needed a retry with all tools.")
    else:
        console.print("Tool routing OFF (--no-routing): every call carried all tools and the full prompt.")
    if summary["input_tokens"]:
        console.print(f"Tokens: {summary['input_tokens']:,} in ({summary['input_tokens'] // n_calls:,} per model call), "
                      f"{summary['output_tokens']:,} out.")
        cost = hf_cost(a.hf_model, summary["input_tokens"], summary["output_tokens"]) if a.hf_model else None
        if cost is not None:
            console.print(f"Estimated Hugging Face cost of this run: ${cost:.3f} at the list price in "
                          "results/hf_models.json (if every call went to that model).")
    console.print(f"[bold]{n_pass}/{len(results)} tasks passed ({n_corr} of them only after the answer check "
                  f"corrected the model, marked PASS*), {ok_checks}/{n_checks} checks passed, "
                  f"{summary['total_seconds']} s.[/]  Saved {stem}.json and {stem}.md")


def hf_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    """USD at the list price saved by  python hf_models.py  (None if that file or model is missing)."""
    try:
        rows = json.load(open("results/hf_models.json", encoding="utf-8"))
    except (OSError, ValueError):
        return None
    base = model.split(":")[0].lower()
    row = next((r for r in rows if r["model"].lower() == base), None)
    if not row or row.get("price_in") is None or row.get("price_out") is None:
        return None
    return (row["price_in"] * tokens_in + row["price_out"] * tokens_out) / 1e6


def markdown_report(summary: dict, results: list[dict]) -> str:
    lines = [f"# Benchmark {summary['date']}", "",
             f"Provider order: `{summary['provider_order']}`  ",
             f"**{summary['tasks_passed']}/{summary['tasks']} tasks passed** "
             f"({summary.get('passed_after_correction', 0)} only after a correction, marked PASS*), "
             f"{summary['checks_passed']}/{summary['checks']} checks, {summary['total_seconds']} s  ",
             f"Calls by provider: {summary['calls_by_provider']}  ",
             f"Tool routing: {'on' if summary.get('routing', True) else 'OFF'}; tokens in "
             f"{summary.get('input_tokens', 0):,}, out {summary.get('output_tokens', 0):,}", "",
             "| Task | Stage | Result | Checks | s | Tokens in | Failed checks |", "|---|---|---|---|---|---|---|"]
    for r in results:
        failed = "; ".join(f"{c['check']} — {c['detail']}" for c in r["checks"] if not c["ok"])
        res = ("PASS*" if r.get("corrected") else "PASS") if r["passed"] else "FAIL"
        lines.append(f"| {r['id']} | {r['stage']} | {res} | "
                     f"{sum(c['ok'] for c in r['checks'])}/{len(r['checks'])} | {r['seconds']} | "
                     f"{r.get('input_tokens', 0):,} | {failed.replace('|', '/')} |")
    lines += ["", "## Answers", ""]
    for r in results:
        lines += [f"### {r['id']}: {r['question']}", "", r["answer"] or "_(no answer)_", ""]
    return "\n".join(lines)


def compare(path_a: str, path_b: str):
    A, B = (json.load(open(p, encoding="utf-8")) for p in (path_a, path_b))
    ra = {r["id"]: r for r in A["results"]}
    rb = {r["id"]: r for r in B["results"]}
    print(f"A: {path_a}  order {A['summary']['provider_order']}  "
          f"{A['summary']['tasks_passed']}/{A['summary']['tasks']} tasks")
    print(f"B: {path_b}  order {B['summary']['provider_order']}  "
          f"{B['summary']['tasks_passed']}/{B['summary']['tasks']} tasks")
    print(f"{'task':6} {'A':6} {'B':6}")
    for tid in sorted(set(ra) | set(rb)):
        f = lambda r: "-" if r is None else ("PASS" if r["passed"] else "FAIL")  # noqa: E731
        print(f"{tid:6} {f(ra.get(tid)):6} {f(rb.get(tid)):6}")


if __name__ == "__main__":
    sys.exit(main())
