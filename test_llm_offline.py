"""
test_llm_offline.py - checks the free-first order and the paid-call cap without any network.
Run:  pytest -q test_llm_offline.py
"""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import llm


def _clear(monkeypatch):
    # Remove settings loaded from your .env so the tests behave the same on every machine
    for k in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ALLOW_OPENAI",
              "MAX_OPENAI_CALLS", "USE_OLLAMA", "GROQ_MODEL", "GROQ_FALLBACK_MODEL",
              "GEMINI_MODEL", "OLLAMA_MODEL", "PROVIDER_ORDER", "EXTRA_BASE_URL", "EXTRA_API_KEY",
              "EXTRA_MODEL", "EXTRA_NAME", "OLLAMA_NUM_CTX", "OLLAMA_THINK", "PAID_FIRST", "OPENAI_MODEL",
              "HF_TOKEN", "USE_HF", "HF_MODEL", "HF_FALLBACK_MODEL", "MAX_HF_CALLS", "HF_REASONING_EFFORT",
              "PROVIDERS_ONLY"]:
        monkeypatch.delenv(k, raising=False)


def test_openai_skipped_unless_allowed(monkeypatch, capsys):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("USE_OLLAMA", "1")
    llm.get_llm([])
    assert "OpenAI" not in capsys.readouterr().out


def test_order_is_free_first(monkeypatch, capsys):
    _clear(monkeypatch)
    for k, v in [("GROQ_API_KEY", "x"), ("GOOGLE_API_KEY", "x"),
                 ("OPENAI_API_KEY", "sk-test"), ("ALLOW_OPENAI", "1")]:
        monkeypatch.setenv(k, v)
    llm.get_llm([])
    out = capsys.readouterr().out
    assert out.index("Groq") < out.index("Gemini") < out.index("Ollama") < out.index("OpenAI")


def test_paid_cap_stops_calls(monkeypatch):
    monkeypatch.setenv("MAX_OPENAI_CALLS", "2")
    llm.USAGE["openai_calls"] = 0
    fake = GenericFakeChatModel(messages=iter([AIMessage("a"), AIMessage("b"), AIMessage("c")]))
    capped = llm._openai_with_cap(fake)
    capped.invoke("hi"); capped.invoke("hi")
    with pytest.raises(RuntimeError, match="cap reached"):
        capped.invoke("hi")
    assert llm.USAGE["openai_calls"] == 2


def test_nothing_configured_raises(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("USE_OLLAMA", "0")
    with pytest.raises(RuntimeError, match="No LLM configured"):
        llm.get_llm([])


class _Boom:
    def __init__(self, msg): self.msg = msg
    def invoke(self, _): raise ConnectionError(self.msg)


def test_fallback_uses_next_provider(capsys):
    ok = GenericFakeChatModel(messages=iter([AIMessage("hello")]))
    chain = llm.chain_providers([("Groq", _Boom("model_not_found")), ("Gemini", ok)])
    assert chain.invoke("hi").content == "hello"
    assert "answered by Gemini" in capsys.readouterr().out


def test_all_fail_gives_short_summary():
    chain = llm.chain_providers([("Groq", _Boom("model_not_found")), ("Ollama (local)", _Boom("refused"))])
    with pytest.raises(llm.AllProvidersFailed) as e:
        chain.invoke("hi")
    msg = str(e.value)
    assert "Groq: ConnectionError: model_not_found" in msg and "Ollama (local)" in msg


def test_groq_backup_model_comes_right_after_groq(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    names = [n for n, _ in llm.build_providers([])]
    assert names[:3] == ["Groq", "Groq backup (gpt-oss-20b)", "Gemini"]
    main = llm.build_providers([])[0][1]
    assert main.bound.max_retries == 0          # main model does not wait on a rate limit


def test_groq_backup_can_be_switched_off(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GROQ_FALLBACK_MODEL", "")
    providers = llm.build_providers([])
    assert [n for n, _ in providers][:1] == ["Groq"] and "backup" not in str([n for n, _ in providers])
    assert providers[0][1].bound.max_retries == 4


def test_provider_order_from_env(monkeypatch):
    _clear(monkeypatch)
    for k in ("GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("ALLOW_OPENAI", "1")
    monkeypatch.setenv("USE_OLLAMA", "1")
    monkeypatch.setenv("PROVIDER_ORDER", "groq, gemini, groq_backup, ollama")
    names = [n for n, _ in llm.build_providers([])]
    assert names[:4] == ["Groq", "Gemini", "Groq backup (gpt-oss-20b)", "Ollama (local)"]
    assert names[-1].startswith("OpenAI") and "(paid" in names[-1]                       # paid is always last
    monkeypatch.setenv("PROVIDER_ORDER", "openai,gemini")             # cannot move paid forward
    names = [n for n, _ in llm.build_providers([])]
    assert names[0] == "Gemini" and names[-1].startswith("OpenAI") and "(paid" in names[-1]


def test_extra_openai_compatible_provider(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("EXTRA_BASE_URL", "https://example.org/v1")
    monkeypatch.setenv("EXTRA_API_KEY", "k")
    monkeypatch.setenv("EXTRA_MODEL", "vendor/some-model:free")
    monkeypatch.setenv("EXTRA_NAME", "OpenRouter")
    names = [n for n, _ in llm.build_providers([])]
    assert names[:4] == ["Groq", "Groq backup (gpt-oss-20b)", "OpenRouter (some-model:free)", "Gemini"]


def test_ollama_gets_a_big_enough_context(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("USE_OLLAMA", "1")
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    (name, model), = [p for p in llm.build_providers([]) if p[0].startswith("Ollama")]
    assert model.bound.num_ctx == 16384 and model.bound.reasoning is False


def test_paid_first_only_when_asked(monkeypatch):
    _clear(monkeypatch)
    for k, v in [("GROQ_API_KEY", "x"), ("OPENAI_API_KEY", "sk-test"), ("ALLOW_OPENAI", "1"),
                 ("OPENAI_MODEL", "gpt-4.1-mini")]:
        monkeypatch.setenv(k, v)
    names = [n for n, _ in llm.build_providers([])]
    assert names[-1].startswith("OpenAI gpt-4.1-mini (paid")             # default: paid last
    monkeypatch.setenv("PAID_FIRST", "1")
    names = [n for n, _ in llm.build_providers([])]
    assert names[0].startswith("OpenAI gpt-4.1-mini (paid")
    monkeypatch.delenv("ALLOW_OPENAI")                                    # never without permission
    assert not any(n.startswith("OpenAI") for n, _ in llm.build_providers([]))


def test_reasoning_models_get_no_temperature():
    assert llm._openai_options("gpt-5-mini") == {"reasoning_effort": "low"}
    assert llm._openai_options("gpt-4.1-mini") == {"temperature": 0}


# ------------------------------------------------------------------ Hugging Face router (2026-10-29)
def _clear_hf(monkeypatch):
    _clear(monkeypatch)
    for k in ("HF_TOKEN", "USE_HF", "HF_MODEL", "HF_FALLBACK_MODEL", "MAX_HF_CALLS", "HF_REASONING_EFFORT",
              "HF_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("USE_OLLAMA", "0")


def test_hugging_face_is_opt_in_and_uses_the_router(monkeypatch):
    _clear_hf(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert llm.build_providers([]) == []                       # a token alone spends nothing
    monkeypatch.setenv("USE_HF", "1")
    monkeypatch.setenv("HF_FALLBACK_MODEL", "Qwen/Qwen3-32B")
    names = [n for n, _ in llm.build_providers([])]
    assert names == ["HF gpt-oss-120b", "HF backup Qwen3-32B"]
    m = llm.hf_chat_model("openai/gpt-oss-120b")
    assert m.openai_api_base == "https://router.huggingface.co/v1" and m.reasoning_effort == "low"
    assert llm.hf_chat_model("Qwen/Qwen3-32B").reasoning_effort is None
    monkeypatch.setenv("HF_REASONING_EFFORT", "none")
    assert llm.hf_chat_model("openai/gpt-oss-20b").reasoning_effort is None


def test_hugging_face_sits_after_groq_by_default(monkeypatch):
    _clear_hf(monkeypatch)
    for k, v in (("GROQ_API_KEY", "x"), ("GOOGLE_API_KEY", "x"), ("HF_TOKEN", "hf"), ("USE_HF", "1")):
        monkeypatch.setenv(k, v)
    names = [n for n, _ in llm.build_providers([])]
    assert names == ["Groq", "Groq backup (gpt-oss-20b)", "HF gpt-oss-120b", "Gemini"]
    monkeypatch.setenv("PROVIDER_ORDER", "hf,groq")
    assert [n for n, _ in llm.build_providers([])][:2] == ["HF gpt-oss-120b", "Groq"]
    monkeypatch.setenv("PROVIDER_ORDER", "hf")
    monkeypatch.setenv("PROVIDERS_ONLY", "1")                  # benchmark one model: no fallbacks at all
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    monkeypatch.setenv("ALLOW_OPENAI", "1")
    monkeypatch.setenv("PAID_FIRST", "1")
    assert [n for n, _ in llm.build_providers([])] == ["HF gpt-oss-120b"]
    monkeypatch.delenv("PROVIDERS_ONLY")


def test_hugging_face_calls_are_capped_and_credit_errors_named(monkeypatch):
    monkeypatch.setenv("MAX_HF_CALLS", "1")
    llm.USAGE["hf_calls"] = 0
    capped = llm._hf_with_cap(GenericFakeChatModel(messages=iter([AIMessage(content="a"), AIMessage(content="b")])))
    assert capped.invoke("hi").content == "a"
    with pytest.raises(RuntimeError, match="cap reached"):
        capped.invoke("again")
    llm.USAGE["hf_calls"] = 0
    assert llm._reason(RuntimeError("Error code: 402 - You have exceeded your monthly included credits")) == \
        "Hugging Face credits used up"


def test_model_list_keeps_tool_callers_and_prices_a_question():
    import hf_models
    data = [{"id": "openai/gpt-oss-120b", "providers": [
                {"provider": "a", "status": "live", "supports_tools": True, "context_length": 131072,
                 "pricing": {"input": 0.5, "output": 1.0}},
                {"provider": "b", "status": "live", "supports_tools": True, "pricing": {"input": 0.1, "output": 0.5}},
                {"provider": "c", "status": "live", "supports_tools": False, "pricing": {"input": 0.01, "output": 0.01}}]},
            {"id": "some/chat-only", "providers": [{"provider": "a", "status": "live", "supports_tools": False}]},
            {"id": "x/no-price", "providers": [{"provider": "d", "status": "live", "supports_tools": True}]},
            {"id": "x/staging", "providers": [{"provider": "e", "status": "staging", "supports_tools": True,
                                              "pricing": {"input": 0.0, "output": 0.0}}]}]
    rows = hf_models.tool_capable(data, 11000, 430)
    assert [r["model"] for r in rows] == ["openai/gpt-oss-120b", "x/no-price"]   # tool callers, live only
    best = rows[0]
    assert best["provider"] == "b" and best["n_tool_providers"] == 2 and best["shortlist"]
    assert best["cost_per_question"] == pytest.approx((0.1 * 11000 + 0.5 * 430) / 1e6)
    assert rows[1]["cost_per_question"] is None
    assert [r["model"] for r in hf_models.tool_capable(data, 1, 1, search="NO-PRICE")] == ["x/no-price"]


def test_benchmark_estimates_the_cost_of_a_run(tmp_path, monkeypatch):
    import json as _json

    import benchmark
    monkeypatch.chdir(tmp_path)
    assert benchmark.hf_cost("openai/gpt-oss-120b", 1_000_000, 0) is None          # no price list yet
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "hf_models.json").write_text(_json.dumps(
        [{"model": "openai/gpt-oss-120b", "price_in": 0.15, "price_out": 0.6}]))
    assert benchmark.hf_cost("openai/gpt-oss-120b:cheapest", 1_000_000, 100_000) == pytest.approx(0.21)
