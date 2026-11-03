"""
check_llms.py - test each LLM provider on its own and list the models your keys can use.
    python check_llms.py
Uses at most one tiny request per provider (OpenAI only if ALLOW_OPENAI=1).
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()
logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # hide an unrelated SDK notice


def try_chat(name, make_model):
    try:
        reply = make_model().invoke("Reply with the single word OK.")
        print(f"  {name}: WORKS  (reply: {str(reply.content).strip()[:40]!r})")
    except Exception as e:  # noqa: BLE001
        print(f"  {name}: FAILED -> {type(e).__name__}: {str(e)[:300]}")


print("\n1) Groq")
if not os.getenv("GROQ_API_KEY"):
    print("  skipped: GROQ_API_KEY is empty in .env")
else:
    try:
        from groq import Groq
        ids = sorted(m.id for m in Groq().models.list().data)
        print("  models your key can use:", ", ".join(ids))
    except Exception as e:  # noqa: BLE001
        print(f"  could not list models -> {type(e).__name__}: {str(e)[:300]}")
    from langchain_groq import ChatGroq
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    try_chat(f"Groq [{model}]", lambda: ChatGroq(model=model, temperature=0, max_retries=0))
    backup = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
    if backup:
        try_chat(f"Groq backup [{backup}]", lambda: ChatGroq(model=backup, temperature=0, max_retries=0))

print("\n2) Gemini")
if not os.getenv("GOOGLE_API_KEY"):
    print("  skipped: GOOGLE_API_KEY is empty in .env")
else:
    try:
        from google import genai
        client = genai.Client()          # keep a reference while listing
        names = sorted(m.name.replace("models/", "") for m in client.models.list()
                       if "generateContent" in (m.supported_actions or [])
                       and "gemini" in m.name and "image" not in m.name)
        print("  chat models your key can use:", ", ".join(names))
    except Exception as e:  # noqa: BLE001
        print(f"  could not list models -> {type(e).__name__}: {str(e)[:300]}")
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    try_chat(f"Gemini [{model}]",
             lambda: ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0))

print("\n3) Ollama (local)")
if os.getenv("USE_OLLAMA", "1") != "1":
    print("  skipped: USE_OLLAMA=0")
else:
    try:
        import httpx
        tags = httpx.get("http://localhost:11434/api/tags", timeout=3).json()
        names = [m["name"] for m in tags.get("models", [])]
        print("  Ollama is running; models pulled:", ", ".join(names) or "(none - run: ollama pull qwen2.5:7b)")
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        try_chat(f"Ollama [{model}]", lambda: ChatOllama(model=model, temperature=0))
    except Exception:  # noqa: BLE001
        print("  not running (install from https://ollama.com, or set USE_OLLAMA=0 in .env)")

print("\n4) Hugging Face router (uses your account's inference credits)")
if not os.getenv("HF_TOKEN"):
    print("  skipped: HF_TOKEN is empty in .env")
elif os.getenv("USE_HF", "0") != "1":
    print("  skipped: USE_HF is not 1 (set USE_HF=1 to use it; one test costs a fraction of a cent)")
else:
    import hf_models
    model = os.getenv("HF_MODEL", "openai/gpt-oss-120b")
    print("  ", end="")
    hf_models.smoke_test(model)
    backup = os.getenv("HF_FALLBACK_MODEL", "").strip()
    if backup:
        print("  ", end="")
        hf_models.smoke_test(backup)

print("\n5) OpenAI (paid)")
if os.getenv("ALLOW_OPENAI") != "1" or not os.getenv("OPENAI_API_KEY"):
    print("  skipped: not allowed (ALLOW_OPENAI is not 1) - no money spent")
else:
    try:
        from openai import OpenAI
        ids = sorted(m.id for m in OpenAI().models.list().data
                     if m.id.startswith(("gpt-", "o1", "o3", "o4")) and "audio" not in m.id
                     and "realtime" not in m.id and "image" not in m.id and "tts" not in m.id)
        print("  chat models your key can use:", ", ".join(ids))
    except Exception as e:  # noqa: BLE001
        print(f"  could not list models -> {type(e).__name__}: {str(e)[:300]}")
    from langchain_openai import ChatOpenAI
    from llm import _openai_options
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try_chat(f"OpenAI [{model}] (one tiny paid request)",
             lambda: ChatOpenAI(model=model, max_tokens=50, max_retries=0, **_openai_options(model)))
print()
