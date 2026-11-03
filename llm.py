"""
llm.py - LLM chain that tries FREE providers first and uses paid OpenAI only as a
last resort, with a hard cap on paid calls.

Order (providers without a key are skipped):
  1. Groq      (free tier)        needs GROQ_API_KEY           model GROQ_MODEL
  1b. Groq     (free tier)        same key, second model       GROQ_FALLBACK_MODEL
     Each Groq model has its OWN per-minute token limit. When the main model is
     rate-limited we switch to the second one at once instead of waiting ~30 s.
  1c. Hugging Face (Pro credits)  needs HF_TOKEN and USE_HF=1  model HF_MODEL (+ HF_FALLBACK_MODEL)
     Open models through the Hugging Face router (OpenAI-compatible). Uses the monthly
     included credits of your account; capped per run by MAX_HF_CALLS (local default 300)
     and per day by DAILY_HF_CALLS (default 500 on a Space, no daily cap locally).
  2. Gemini    (free tier)        needs GOOGLE_API_KEY
  3. Ollama    (free, local)      needs Ollama installed and running
  4. OpenAI    (PAID, capped)     needs OPENAI_API_KEY *and* ALLOW_OPENAI=1 (never on a Space)

On a Hugging Face Space (SPACE_ID is set) paid OpenAI is switched off whatever the settings say,
and Ollama is off unless USE_OLLAMA=1 is set explicitly.

If a provider fails, the next one is tried. If all fail, you get ONE short message
listing why each failed (instead of a long traceback).
Run  python check_llms.py  to test each provider on its own.
"""
FILES_VERSION = "2026-11-02"   # must match in graph.py, agent_tools.py, llm.py, run.py

import logging
import os
import re
import threading
import time

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

load_dotenv()
# Hide a harmless notice printed when Gemini takes over from Groq mid-question.
for _name in ("langchain_google_genai", "langchain_google_genai.chat_models"):
    logging.getLogger(_name).setLevel(logging.ERROR)

HF_ROUTER = "https://router.huggingface.co/v1"
ON_SPACE = bool(os.getenv("SPACE_ID"))          # set by Hugging Face on a Space
USAGE = {"openai_calls": 0, "hf_calls": 0, "last_provider": None, "by_provider": {},   # successful calls
         "input_tokens": 0, "output_tokens": 0}                          # as reported by the provider


class AllProvidersFailed(RuntimeError):
    pass


def _short(err: Exception) -> str:
    text = f"{type(err).__name__}: {err}".replace("\n", " ")
    return text[:220] + ("..." if len(text) > 220 else "")


def _reason(err: Exception) -> str:
    """One or two words saying why a provider failed (shown in the terminal)."""
    t = f"{type(err).__name__} {err}".lower()
    # checked first: Groq reports a too-large request as 413 with code 'rate_limit_exceeded'
    if "413" in t or "too large" in t or "context length" in t or "context_length" in t:
        m = re.search(r"limit\D{0,3}(\d+)\D+requested\D{0,3}(\d+)", t)
        return f"request too large: limit {m.group(1)}, requested {m.group(2)} tokens" if m else \
            "request too large"
    if "402" in t or "payment required" in t or "included credits" in t or "insufficient credit" in t:
        return "Hugging Face credits used up"
    if "daily cap" in t:
        return "daily cap of this app reached"
    if "cap reached" in t:
        return "call cap of this run reached"
    if "per day" in t or "(tpd)" in t or "(rpd)" in t or "daily" in t:
        return "daily limit reached"
    if "429" in t or "rate limit" in t or "rate_limit" in t or "resource_exhausted" in t or "quota" in t:
        return "rate limit"
    if "tool_use_failed" in t or "failed to call a function" in t or \
            ("tool" in t and ("400" in t or "validation" in t)):
        return "bad tool call"
    if "503" in t or "overloaded" in t or "unavailable" in t:
        return "overloaded"
    if "timeout" in t or "timed out" in t:
        return "timeout"
    if "connect" in t or "refused" in t:
        return "not reachable"
    detail = re.sub(r"\s+", " ", str(err)).strip()[:70]       # unknown: show the start of the message
    return f"{type(err).__name__}: {detail}" if detail else type(err).__name__


_DAY_LOCK = threading.Lock()


def _usage_file() -> str:
    return os.path.join(os.getenv("QI_LOG_DIR", "logs"), "hf_usage.json")


def hf_calls_today(add: int = 0) -> int:
    """Hugging Face calls made today (UTC), kept in logs/hf_usage.json so a restart of the app does not
    reset the daily cap. add=1 counts one more call. Never raises (a broken file counts as 0)."""
    import datetime
    import json
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    with _DAY_LOCK:
        try:
            with open(_usage_file(), encoding="utf-8") as f:
                d = json.load(f)
            n = int(d.get("calls", 0)) if d.get("date") == today else 0
        except (OSError, ValueError, TypeError, AttributeError):
            n = 0
        if add:
            n += add
            try:
                os.makedirs(os.path.dirname(_usage_file()), exist_ok=True)
                with open(_usage_file(), "w", encoding="utf-8") as f:
                    json.dump({"date": today, "calls": n}, f)
            except OSError:
                pass
        return n


def _hf_with_cap(model):
    """Hugging Face credits are real money beyond the monthly included amount: hard caps per run
    (MAX_HF_CALLS, 0 = none) and per UTC day (DAILY_HF_CALLS, 0 = none). Over a cap the call fails,
    so the next (free) provider answers."""
    run_cap = int(os.getenv("MAX_HF_CALLS", "0" if ON_SPACE else "300"))
    day_cap = int(os.getenv("DAILY_HF_CALLS", "500" if ON_SPACE else "0"))

    def call(messages):
        if run_cap and USAGE["hf_calls"] >= run_cap:
            raise RuntimeError(f"Hugging Face cap reached ({run_cap} calls this run); raise MAX_HF_CALLS in .env")
        if day_cap and hf_calls_today() >= day_cap:
            raise RuntimeError(f"Hugging Face daily cap reached ({day_cap} calls today, UTC); "
                               "raise DAILY_HF_CALLS")
        USAGE["hf_calls"] += 1
        if day_cap:
            hf_calls_today(add=1)
        return model.invoke(messages)

    return RunnableLambda(call, name="hf_capped")


def hf_chat_model(model: str, max_tokens: int = 2048):
    """A chat model served by the Hugging Face router. `model` may name the provider or a policy,
    e.g. 'openai/gpt-oss-120b', 'Qwen/Qwen3-32B:groq' or 'meta-llama/Llama-3.3-70B-Instruct:cheapest'."""
    from langchain_openai import ChatOpenAI
    extra = {}
    effort = os.getenv("HF_REASONING_EFFORT", "low").strip().lower()
    if "gpt-oss" in model and effort not in ("", "none", "off"):
        extra["reasoning_effort"] = effort           # gpt-oss: think less -> faster, fewer tokens
    return ChatOpenAI(model=model, base_url=os.getenv("HF_BASE_URL", HF_ROUTER), api_key=os.environ["HF_TOKEN"],
                      temperature=0, max_tokens=max_tokens, max_retries=1, timeout=90, **extra)


def _openai_with_cap(model):
    cap = int(os.getenv("MAX_OPENAI_CALLS", "20"))

    def call(messages):
        if USAGE["openai_calls"] >= cap:
            raise RuntimeError(f"paid cap reached ({cap} calls this run); raise MAX_OPENAI_CALLS in .env")
        USAGE["openai_calls"] += 1
        return model.invoke(messages)

    return RunnableLambda(call, name="openai_capped")


def build_providers(tools):
    """Return [(name, runnable_with_tools)] in priority order (see PROVIDER_ORDER)."""
    providers = []

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        def groq(model, retries):
            # reasoning_effort="low": gpt-oss thinks less before answering -> faster, fewer tokens.
            extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
            # max_tokens caps the reply (our answers are short). If Groq counts the requested
            # size against the per-minute token limit, a smaller cap also leaves room for more calls.
            return ChatGroq(model=model, temperature=0, max_retries=retries, max_tokens=2048,
                            **extra).bind_tools(tools)

        main = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        backup = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
        if backup and backup != main:
            # main model: fail fast on a rate limit (no waiting) -> the backup answers at once
            providers.append(("groq", "Groq", groq(main, 0)))
            # backup model: may wait out its own rate limit (the client sleeps as Groq asks)
            providers.append(("groq_backup", f"Groq backup ({backup.split('/')[-1]})", groq(backup, 3)))
        else:
            providers.append(("groq", "Groq", groq(main, 4)))

    if os.getenv("HF_TOKEN") and os.getenv("USE_HF", "0") == "1":     # opt-in: credits are money
        main = os.getenv("HF_MODEL", "openai/gpt-oss-120b").strip()
        backup = os.getenv("HF_FALLBACK_MODEL", "").strip()
        short = lambda m: m.split("/")[-1]                                     # noqa: E731
        providers.append(("hf", f"HF {short(main)}", _hf_with_cap(hf_chat_model(main).bind_tools(tools))))
        if backup and backup != main:
            providers.append(("hf_backup", f"HF backup {short(backup)}",
                              _hf_with_cap(hf_chat_model(backup).bind_tools(tools))))

    if os.getenv("EXTRA_BASE_URL") and os.getenv("EXTRA_API_KEY") and os.getenv("EXTRA_MODEL"):
        # any OpenAI-compatible endpoint with a free tier (e.g. OpenRouter ':free' models, Cerebras,
        # the Hugging Face router); check that the model supports tool calling.
        from langchain_openai import ChatOpenAI
        m = ChatOpenAI(model=os.environ["EXTRA_MODEL"], base_url=os.environ["EXTRA_BASE_URL"],
                       api_key=os.environ["EXTRA_API_KEY"], temperature=0, max_tokens=2048,
                       max_retries=1, timeout=60)
        label = os.getenv("EXTRA_NAME", "Extra")
        providers.append(("extra", f"{label} ({os.environ['EXTRA_MODEL'].split('/')[-1]})", m.bind_tools(tools)))

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        # timeout: give up after 25 s (an overloaded Gemini can otherwise hang for minutes)
        m = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                                   temperature=0, max_retries=0, timeout=25)
        providers.append(("gemini", "Gemini", m.bind_tools(tools)))

    if os.getenv("USE_OLLAMA", "0" if ON_SPACE else "1") == "1":
        from langchain_ollama import ChatOllama
        # num_ctx: Ollama's default context (about 4096 tokens) is SMALLER than one agent request
        # (instructions + tool descriptions + tool results), so the start of the conversation was
        # silently cut off - the model then forgot what it had done and repeated calls.
        # reasoning=False: no long hidden "thinking" for models such as qwen3 (much faster).
        m = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), temperature=0,
                       num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")), num_predict=2048,
                       reasoning=os.getenv("OLLAMA_THINK", "0") == "1", keep_alive="30m")
        providers.append(("ollama", "Ollama (local)", m.bind_tools(tools)))

    if os.getenv("OPENAI_API_KEY") and os.getenv("ALLOW_OPENAI") == "1" and not ON_SPACE:   # paid: never public
        from langchain_openai import ChatOpenAI
        name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        m = ChatOpenAI(model=name, max_tokens=2048, max_retries=1, timeout=90, **_openai_options(name))
        providers.append(("openai", f"OpenAI {name} (paid, max {os.getenv('MAX_OPENAI_CALLS', '20')} calls)",
                          _openai_with_cap(m.bind_tools(tools))))
    return [(name, model) for _key, name, model in _ordered(providers)]


def _openai_options(model: str) -> dict:
    """Reasoning models (gpt-5*, o1/o3/o4*) reject temperature=0; give them low reasoning effort."""
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", "low")}
    return {"temperature": 0}


DEFAULT_ORDER = "groq,groq_backup,hf,hf_backup,extra,gemini,ollama"


def _ordered(providers):
    """Sort by PROVIDER_ORDER in .env (e.g. groq,gemini,groq_backup,ollama). Providers not
    listed keep their default place after the listed ones. Paid OpenAI is last, unless
    PAID_FIRST=1 is set explicitly (for development, together with ALLOW_OPENAI=1)."""
    wanted = [k.strip().lower() for k in os.getenv("PROVIDER_ORDER", DEFAULT_ORDER).split(",") if k.strip()]
    rank = {k: i for i, k in enumerate(wanted)}
    free = [p for p in providers if p[0] != "openai"]
    paid = [p for p in providers if p[0] == "openai"]
    free.sort(key=lambda p: rank.get(p[0], len(rank)))      # stable: unlisted keep default order
    if os.getenv("PROVIDERS_ONLY") == "1":                   # e.g. benchmark one model: no fallbacks
        free = [p for p in free if p[0] in rank]
        paid = [p for p in paid if "openai" in rank]
    return paid + free if os.getenv("PAID_FIRST") == "1" else free + paid


def chain_providers(providers):
    """Try each provider in order; report every failure if none works."""
    def call(messages):
        errors = []
        for name, model in providers:
            t0 = time.time()
            try:
                out = model.invoke(messages)
                secs = time.time() - t0
                if errors:      # tell the user a fallback was needed, and why
                    why = ", ".join(f"{n} ({r})" for n, _, r in errors)
                    print(f"  [llm] {why} failed -> answered by {name} ({secs:.0f} s)")
                    if name.startswith("Ollama") and not USAGE.get("warned_local"):
                        USAGE["warned_local"] = True
                        print("  [llm] NOTE: every cloud model is unavailable (daily free quota used up or "
                              "overloaded), so the small local Ollama model is answering. It is slow and "
                              "much less reliable. Best: stop and try again after the quotas reset "
                              "(Groq: within 24 h). To fail fast instead, set USE_OLLAMA=0 in .env.")
                elif secs > 15:
                    why = ("local model on this computer" if name.startswith("Ollama") else
                           "slow reply: a rate limit, a busy provider or long reasoning" if name.startswith("HF")
                           else "probably waiting out a rate limit")
                    print(f"  [llm] {name} took {secs:.0f} s ({why})")
                USAGE["last_provider"] = name
                import session
                session.current().last_provider = name      # per user in the web app
                USAGE["by_provider"][name] = USAGE["by_provider"].get(name, 0) + 1
                um = getattr(out, "usage_metadata", None) or {}
                USAGE["input_tokens"] += int(um.get("input_tokens") or 0)
                USAGE["output_tokens"] += int(um.get("output_tokens") or 0)
                return out
            except Exception as e:   # noqa: BLE001 - we want to try the next provider
                errors.append((name, _short(e), _reason(e)))
        lines = "\n".join(f"  - {n}: {msg}" for n, msg, _ in errors)
        raise AllProvidersFailed(f"Every LLM provider failed:\n{lines}\n"
                                 "Run  python check_llms.py  to diagnose.")
    return RunnableLambda(call, name="free_first_llm")


class RoutedLLM:
    """The provider chain, bound to a chosen subset of the tools on demand (cached per subset).
    invoke() uses all tools; for_tools(names) only the named ones (see routing.py)."""

    def __init__(self, tools):
        self.tools = list(tools)
        providers = build_providers(self.tools)
        if not providers:
            raise RuntimeError("No LLM configured. Add HF_TOKEN (with USE_HF=1), GROQ_API_KEY or GOOGLE_API_KEY "
                               "as Space secrets (Settings -> Variables and secrets)." if ON_SPACE else
                               "No LLM configured. Add GROQ_API_KEY or GOOGLE_API_KEY to .env, "
                               "install Ollama, or set ALLOW_OPENAI=1.")
        USAGE["order"] = " -> ".join(n for n, _ in providers)
        self._cache = {frozenset(t.name for t in self.tools): chain_providers(providers)}

    def for_tools(self, names):
        key = frozenset(n for n in names if any(t.name == n for t in self.tools))
        if key not in self._cache:
            self._cache[key] = chain_providers(build_providers([t for t in self.tools if t.name in key]))
        return self._cache[key]

    def invoke(self, messages):
        return self._cache[frozenset(t.name for t in self.tools)].invoke(messages)


def get_llm(tools):
    llm = RoutedLLM(tools)
    print("LLM order:", USAGE["order"])
    return llm
