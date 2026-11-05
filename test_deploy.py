"""
test_deploy.py - step 4 (public Hugging Face Space): the Space-only safety switches, the daily model
cap, the per-visitor limits, the compute lock, and the deploy script's whitelist and key scan.
No network: nothing is uploaded.  Run:  pytest -q test_deploy.py
"""
import os
import threading

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agent_tools as at
import deploy_space
import llm
import session

KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ALLOW_OPENAI", "USE_OLLAMA", "PROVIDER_ORDER",
        "PAID_FIRST", "HF_TOKEN", "USE_HF", "HF_MODEL", "HF_FALLBACK_MODEL", "MAX_HF_CALLS", "DAILY_HF_CALLS",
        "PROVIDERS_ONLY", "EXTRA_BASE_URL", "EXTRA_API_KEY", "EXTRA_MODEL"]


@pytest.fixture(autouse=True)
def clean_store():
    yield
    at.STORE.clear(); at.DESC.clear(); at.TOMO.clear()       # noqa: E702 - leave the default session empty


@pytest.fixture
def space(monkeypatch, tmp_path):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "ON_SPACE", True)
    monkeypatch.setenv("QI_LOG_DIR", str(tmp_path))
    llm.USAGE["hf_calls"] = 0
    yield monkeypatch
    llm.USAGE["hf_calls"] = 0


def test_space_never_uses_paid_openai_or_ollama(space):
    for k, v in (("OPENAI_API_KEY", "sk-test"), ("ALLOW_OPENAI", "1"), ("PAID_FIRST", "1"), ("GROQ_API_KEY", "x")):
        space.setenv(k, v)
    names = [n for n, _ in llm.build_providers([])]
    assert names == ["Groq", "Groq backup (gpt-oss-20b)"]            # no OpenAI, no Ollama
    space.setenv("USE_OLLAMA", "1")                                  # only if asked explicitly
    assert "Ollama (local)" in [n for n, _ in llm.build_providers([])]


def test_daily_cap_survives_a_restart_and_resets_next_day(space, tmp_path):
    space.setenv("DAILY_HF_CALLS", "2")
    replies = iter([AIMessage(content=str(i)) for i in range(5)])
    capped = llm._hf_with_cap(GenericFakeChatModel(messages=replies))
    assert capped.invoke("a").content == "0" and capped.invoke("b").content == "1"
    with pytest.raises(RuntimeError, match="daily cap"):
        capped.invoke("c")
    llm.USAGE["hf_calls"] = 0                                        # 'restart': the file still counts 2
    with pytest.raises(RuntimeError, match="daily cap"):
        llm._hf_with_cap(GenericFakeChatModel(messages=replies)).invoke("d")
    (tmp_path / "hf_usage.json").write_text('{"date": "2000-01-01", "calls": 999}')   # another day
    assert llm.hf_calls_today() == 0
    (tmp_path / "hf_usage.json").write_text("not json")
    assert llm.hf_calls_today() == 0                                 # a broken file never blocks the app
    assert llm._reason(RuntimeError("Hugging Face daily cap reached (2 calls today, UTC)")) == \
        "daily cap of this app reached"


def test_space_has_no_run_cap_but_local_runs_do(space):
    capped = llm._hf_with_cap(GenericFakeChatModel(messages=iter([AIMessage(content="x")] * 400)))
    llm.USAGE["hf_calls"] = 350                                       # above the local default of 300
    space.setenv("DAILY_HF_CALLS", "0")
    assert capped.invoke("q").content == "x"
    space.setattr(llm, "ON_SPACE", False)
    with pytest.raises(RuntimeError, match="this run"):
        llm._hf_with_cap(GenericFakeChatModel(messages=iter([AIMessage(content="x")]))).invoke("q")


def test_visitor_limits(monkeypatch):
    gr = pytest.importorskip("gradio")  # noqa: F841
    import app
    s = session.Session("visitor")
    monkeypatch.setattr(app, "QUESTIONS_PER_HOUR", 2)
    assert "too long" in app.too_much("x" * (app.MAX_QUESTION_CHARS + 1), s)
    assert app.too_much("one", s) is None and app.too_much("two", s) is None
    msg = app.too_much("three", s)
    assert "2 questions per hour" in msg and "Workbench" in msg
    s.asked[:] = [t - 3601 for t in s.asked]                          # an hour later
    assert app.too_much("four", s) is None
    monkeypatch.setattr(app, "QUESTIONS_PER_HOUR", 0)                 # 0 = no limit (local default)
    assert all(app.too_much("q", s) is None for _ in range(100))


def test_tools_hold_the_compute_lock():
    assert all(getattr(t.func, "_qi_locked", False) for t in at.TOOLS)
    session.COMPUTE_LOCK.acquire()                                    # another thread is plotting ...
    done = []
    th = threading.Thread(target=lambda: done.append(at.create_state.invoke({"kind": "phi+"})))
    th.start()
    th.join(0.3)
    assert not done                                                   # ... so this tool waits
    session.COMPUTE_LOCK.release()
    th.join(5)
    assert done and done[0].startswith("Created")
    assert at.create_state.invoke({"kind": "psi-"}).startswith("Created")   # re-entrant, no deadlock


def test_deploy_uploads_only_the_whitelist(tmp_path):
    files = deploy_space.stage(str(tmp_path), "Quantum Information Agent")
    assert ".env" not in files and not any(f.startswith(("logs", "figures", "data/sessions", "test_"))
                                           for f in files)
    assert {"README.md", "requirements.txt", "app.py", "data/state_tomo_data.csv"} <= set(files)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "sdk: gradio" in readme and "app_file: app.py" in readme and "sdk_version:" in readme
    req = (tmp_path / "requirements.txt").read_text()
    assert "langgraph==" in req and "gradio" not in req and "pytest" not in req and "ollama" not in req
    assert deploy_space.find_keys([str(tmp_path / f) for f in files]) == []


def test_deploy_stops_on_a_key_in_a_file(tmp_path):
    p = tmp_path / "leak.py"
    p.write_text('TOKEN = "hf_' + "a" * 34 + '"\n')
    q = tmp_path / "ok.py"
    q.write_text('os.environ["HF_TOKEN"]  # read from the environment\n')
    assert deploy_space.find_keys([str(p), str(q)]) == [str(p)]


def test_deploy_never_sends_the_openai_key():
    assert "OPENAI_API_KEY" not in deploy_space.SECRET_KEYS
    assert deploy_space.VARIABLES["PROVIDER_ORDER"].split(",")[0] == "hf"


def test_gitignore_keeps_secrets_and_run_files_out():
    text = open(".gitignore", encoding="utf-8").read()
    for line in (".env", "figures/", "logs/", "data/sessions/", "data/simulated_rho_*.csv"):
        assert line in text.splitlines()


def test_app_starts_without_any_model_key():
    """A Space whose secrets are missing must still start: the Workbench works, chat says why not."""
    pytest.importorskip("gradio")
    import app
    demo = app.build_demo(app.NoModel("No LLM configured."))
    respond = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "respond")
    _, hist, _ = respond("Create phi+ and compute its concurrence.", [], "nokey")
    assert "not available right now" in hist[-1]["content"] and "Workbench" in hist[-1]["content"]
    session.drop("nokey")


def test_space_card_helps_people_find_the_space(tmp_path):
    env = {"HF_MODEL": "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra", "HF_FALLBACK_MODEL": "openai/gpt-oss-120b:deepinfra"}
    deploy_space.stage(str(tmp_path), "Quantum Information Agent", env, "https://example.com/card.png")
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    head = card.split("\n---\n")[0]
    assert "tags:\n  - quantum\n" in head and "  - entanglement\n" in head
    assert "models:\n  - deepseek-ai/DeepSeek-V4-Flash-0731\n  - openai/gpt-oss-120b\n" in head
    assert "thumbnail: https://example.com/card.png" in head and "pinned: true" in head
    import yaml
    meta = yaml.safe_load(head.lstrip("-\n"))
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py" and len(meta["short_description"]) <= 60
    assert deploy_space.model_ids({}) == []


def test_pydantic_is_not_pinned():
    """Space build of 2026-11-01 failed: pydantic==2.13.5 (copied from this computer) conflicts with the
    Space's gradio[oauth,mcp]==6.27.0 (mcp extra: pydantic <= 2.12.5)."""
    req = deploy_space.requirements().splitlines()
    assert "pydantic" in req and not any(x.startswith("pydantic==") for x in req)


def test_pip_check_runs_the_spaces_install(monkeypatch, tmp_path):
    import subprocess
    import types
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=1, stdout="", stderr="ERROR: ResolutionImpossible\nconflict: pydantic")
    monkeypatch.setattr(subprocess, "run", fake_run)
    req = tmp_path / "requirements.txt"
    req.write_text(deploy_space.requirements())
    problem = deploy_space.check_install(str(req))
    assert "ResolutionImpossible" in problem
    cmd = " ".join(seen["cmd"])
    assert "--dry-run" in cmd and "gradio[oauth,mcp]==" in cmd and "spaces" in cmd and "--platform" not in cmd
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert deploy_space.check_install(str(req)) is None


def test_a_failing_check_uploads_nothing(monkeypatch):
    monkeypatch.setattr(deploy_space, "check_install", lambda path: "ERROR: conflict")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: (_ for _ in ()).throw(AssertionError("upload!")))
    with pytest.raises(SystemExit, match="nothing was uploaded"):
        deploy_space.main(["--repo", "me/x"])


def test_pip_check_shows_the_conflict_explanation(monkeypatch, tmp_path):
    """2026-11-01: the check said only 'Cannot install gradio' (its --quiet hid pip's explanation, which is
    printed on stdout). The whole 'The conflict is caused by' block must be shown."""
    import subprocess
    import types
    stdout = ("Collecting x\nERROR: Cannot install gradio and pydantic==2.13.5 because these package versions have "
              "conflicting dependencies.\n\nThe conflict is caused by:\n    The user requested pydantic==2.13.5\n"
              "    gradio[mcp,oauth] 6.27.0 depends on pydantic<=2.12.5\n")
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=1, stdout=stdout, stderr="ERROR: ResolutionImpossible"))
    req = tmp_path / "requirements.txt"
    req.write_text("pydantic==2.13.5\n")
    problem = deploy_space.check_install(str(req))
    assert "The conflict is caused by" in problem and "pydantic<=2.12.5" in problem and "Collecting" not in problem


def test_space_card_links_the_github_repository(tmp_path):
    deploy_space.stage(str(tmp_path), "Quantum Information Agent", {}, "", "https://github.com/me/quantum-information-agent")
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "**Source code, tests and design notes:** https://github.com/me/quantum-information-agent" in card
    deploy_space.stage(str(tmp_path / "b"), "Quantum Information Agent", {})
    assert "Source code" not in (tmp_path / "b" / "README.md").read_text(encoding="utf-8")


def test_upload_never_uses_the_inference_token_from_the_shell(monkeypatch):
    """2026-11-04: HF_TOKEN (the inference token) was set in the shell, so huggingface_hub used it instead
    of the fresh `hf auth login` token, and the upload failed with 403. Uploading must ignore HF_TOKEN."""
    import huggingface_hub
    seen = {}
    monkeypatch.setenv("HF_TOKEN", "inference-token")
    monkeypatch.delenv("HF_DEPLOY_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token",
                        lambda: seen.setdefault("env", os.environ.get("HF_TOKEN")) or "login-token")
    assert deploy_space.deploy_token() == ("login-token", "hf auth login")
    assert seen["env"] is None and os.environ["HF_TOKEN"] == "inference-token"     # restored afterwards
    monkeypatch.setenv("HF_DEPLOY_TOKEN", "deploy-token")
    assert deploy_space.deploy_token() == ("deploy-token", "HF_DEPLOY_TOKEN")


class FakeApi:
    def __init__(self, name="me", status=None):
        self.name, self.status, self.uploaded = name, status, False

    def __call__(self, token=None):
        return self

    def whoami(self):
        return {"name": self.name, "orgs": [], "auth": {"accessToken": {"role": "write"}}}

    def create_repo(self, *a, **k):
        if self.status:
            import httpx
            from huggingface_hub.errors import HfHubHTTPError
            r = httpx.Response(self.status, request=httpx.Request("POST", "https://huggingface.co"))
            raise HfHubHTTPError("403 Forbidden", response=r)

    def upload_folder(self, *a, **k):
        self.uploaded = True


def _deploy(monkeypatch, fake):
    import huggingface_hub
    monkeypatch.setattr(deploy_space, "check_install", lambda path: None)
    monkeypatch.setattr(deploy_space, "deploy_token", lambda: ("tok", "hf auth login"))
    monkeypatch.setattr(huggingface_hub, "HfApi", fake)
    return deploy_space.main(["--repo", "me/x"])


def test_a_403_is_explained_in_plain_words(monkeypatch):
    with pytest.raises(SystemExit, match="(?s)cannot write to me/x.*paste a token"):
        _deploy(monkeypatch, FakeApi(status=403))


def test_wrong_account_uploads_nothing(monkeypatch):
    fake = FakeApi(name="someone-else")
    with pytest.raises(SystemExit, match="logged in as someone-else"):
        _deploy(monkeypatch, fake)
    assert not fake.uploaded


def test_right_account_uploads(monkeypatch, capsys):
    fake = FakeApi()
    assert _deploy(monkeypatch, fake) == 0 and fake.uploaded
    assert "Uploading as me (token from hf auth login, type: write)" in capsys.readouterr().out
