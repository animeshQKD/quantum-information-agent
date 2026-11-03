"""
hf_models.py - which open models on the Hugging Face router can drive this agent, and what they cost.

    python hf_models.py                     list chat models whose provider supports TOOL CALLING,
                                            cheapest first, with the cost of one agent question
    python hf_models.py --search qwen       only models whose name contains 'qwen'
    python hf_models.py --test openai/gpt-oss-120b
    python hf_models.py --test openai/gpt-oss-120b:deepinfra Qwen/Qwen3-32B:nscale
                                            several models; ':provider' pins the provider (and its price)
                                            one tiny tool-calling request through the router
                                            (costs a fraction of a cent of your credits)

The agent needs a model that calls tools reliably (the physics is done by our tools), so only
providers with supports_tools are listed. Cost per question uses the measured benchmark average
(about 11,000 input and 430 output tokens per question); change it with --tokens-in/--tokens-out.
Needs HF_TOKEN in .env only for --test (the list itself is public).
"""
import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()
ROUTER = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
# models worth benchmarking first: strong tool callers of different families and sizes
SHORTLIST = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "Qwen/Qwen3-32B", "Qwen/Qwen3-235B-A22B",
             "meta-llama/Llama-3.3-70B-Instruct", "moonshotai/Kimi-K2", "deepseek-ai/DeepSeek-V3")


def fetch_models() -> list[dict]:
    import requests
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.getenv("HF_TOKEN") else {}
    r = requests.get(f"{ROUTER}/models", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def tool_capable(models: list[dict], tokens_in: int, tokens_out: int, search: str = "") -> list[dict]:
    """One row per model: its cheapest LIVE provider that supports tool calling (prices in USD per
    million tokens; cost of one agent question in USD; None when the provider gives no price)."""
    rows = []
    for m in models:
        mid = m.get("id", "")
        if search and search.lower() not in mid.lower():
            continue
        best = None
        for p in m.get("providers") or []:
            if not p.get("supports_tools") or p.get("status", "live") != "live":
                continue
            price = p.get("pricing") or {}
            pin, pout = price.get("input"), price.get("output")
            cost = (pin * tokens_in + pout * tokens_out) / 1e6 if pin is not None and pout is not None else None
            row = {"model": mid, "provider": p.get("provider", "?"), "context": p.get("context_length"),
                   "price_in": pin, "price_out": pout, "cost_per_question": cost,
                   "n_tool_providers": 0, "shortlist": any(mid.lower() == s.lower() for s in SHORTLIST)}
            if best is None or (cost is not None and (best["cost_per_question"] is None
                                                      or cost < best["cost_per_question"])):
                best = row
        if best:
            best["n_tool_providers"] = sum(1 for p in m.get("providers") or [] if p.get("supports_tools"))
            rows.append(best)
    rows.sort(key=lambda r: (r["cost_per_question"] is None, r["cost_per_question"] or 0, r["model"]))
    return rows


def print_table(rows: list[dict], limit: int, budget: float):
    from rich.console import Console
    from rich.table import Table
    t = Table(title=f"Hugging Face router: models with tool calling ({len(rows)} found; * = shortlist)")
    t.add_column("model", no_wrap=True)
    for c in ("cheapest tool provider", "context", "$ / M in", "$ / M out", "$ / question",
              f"questions per ${budget:g}"):
        t.add_column(c)
    for r in rows[:limit]:
        cost = r["cost_per_question"]
        t.add_row(("* " if r["shortlist"] else "") + r["model"], f"{r['provider']} (of {r['n_tool_providers']})",
                  f"{r['context']:,}" if r["context"] else "?",
                  "?" if r["price_in"] is None else f"{r['price_in']:.2f}",
                  "?" if r["price_out"] is None else f"{r['price_out']:.2f}",
                  "?" if cost is None else f"{cost:.4f}",
                  "?" if not cost else f"{budget / cost:,.0f}")
    Console().print(t)


def smoke_test(model: str) -> int:
    """One tool-calling request: does the model call the tool with the right argument?"""
    if not os.getenv("HF_TOKEN"):
        print("HF_TOKEN is empty in .env (create a token at https://huggingface.co/settings/tokens)")
        return 1
    from langchain_core.tools import tool

    import llm

    @tool
    def create_state(kind: str, p: float = 1.0) -> str:
        """Create a two-qubit state: kind = 'phi+', 'psi-', 'werner', ...; p = Werner parameter."""
        return "ok"

    t0 = time.time()
    try:
        reply = llm.hf_chat_model(model, max_tokens=300).bind_tools([create_state]).invoke(
            "Create a Werner state with p = 0.3. Use the tool.")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED ({llm._reason(e)}): {str(e)[:300]}")
        return 1
    calls = getattr(reply, "tool_calls", None) or []
    um = getattr(reply, "usage_metadata", None) or {}
    ok = bool(calls) and calls[0]["name"] == "create_state" and abs(float(calls[0]["args"].get("p", 0)) - 0.3) < 1e-9
    print(f"{'WORKS' if ok else 'NO CORRECT TOOL CALL'}: {model} in {time.time() - t0:.1f} s, "
          f"{um.get('input_tokens', '?')} tokens in / {um.get('output_tokens', '?')} out")
    print("  tool calls:", json.dumps(calls)[:300] if calls else f"none; reply: {str(reply.content)[:200]!r}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", default="", help="only models whose id contains this text")
    ap.add_argument("--limit", type=int, default=40, help="rows to show (default 40)")
    ap.add_argument("--tokens-in", type=int, default=11000, help="input tokens per question (default 11000)")
    ap.add_argument("--tokens-out", type=int, default=430, help="output tokens per question (default 430)")
    ap.add_argument("--budget", type=float, default=2.0, help="USD for the 'questions per $' column (default 2)")
    ap.add_argument("--test", metavar="MODEL", nargs="+",
                    help="send one tool-calling request to each model (pin a provider with model:provider)")
    a = ap.parse_args()
    if a.test:
        return max(smoke_test(m) for m in a.test)
    rows = tool_capable(fetch_models(), a.tokens_in, a.tokens_out, a.search)
    print_table(rows, a.limit, a.budget)
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/hf_models.json", "w", encoding="utf-8"), indent=1)
    print("Saved results/hf_models.json. Prices are the providers' list prices; the shortlist (*) is "
          "what I would benchmark first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
