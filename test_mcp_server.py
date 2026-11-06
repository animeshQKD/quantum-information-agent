"""
test_mcp_server.py - the MCP server (mcp_server.py): which tools are exposed and how they are described,
sessions (separate, unguessable, capped, expiring), the compute lock, limits, strict JSON results, figures
as PNG images, no internal paths, and one end-to-end run through a real MCP client over HTTP.
Run:  pytest -q test_mcp_server.py        (skipped if gradio is not installed)
"""
import asyncio
import base64
import json
import os
import socket
import threading
import typing

import numpy as np
import pytest

gr = pytest.importorskip("gradio")

import agent_tools as at  # noqa: E402
import app  # noqa: E402
import deploy_space  # noqa: E402
import mcp_server as ms  # noqa: E402
import qi_tools as qt  # noqa: E402
import session  # noqa: E402
from gradio import utils as gr_utils  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(ms, "CALLS_PER_MINUTE", 0)
    ms._calls.clear()
    yield
    for sid in session.active_ids():
        session.drop(sid)


def strict(text: str) -> dict:
    """The JSON a client receives: must parse without NaN or Infinity."""
    def no_constant(name):
        raise AssertionError(f"non-finite number {name} in the result")
    return json.loads(text, parse_constant=no_constant)


def new(kind="phi+", **kw) -> tuple[str, dict]:
    out = strict(ms.create_state(kind, **kw))
    return out["session_id"], out


# ------------------------------------------------------------------ what is exposed
def test_api_and_mcp_server_are_exactly_the_curated_tools():
    demo = app.build_demo(app.NoModel("none"))
    named = sorted(e.lstrip("/") for e in demo.get_api_info()["named_endpoints"])
    assert named == sorted(ms.NAMES) and len(named) == 32
    assert "respond" not in named and "new_session_id" not in named       # chat spends the model quota
    ui = [f for f in demo.fns.values() if f.api_name not in ms.NAMES]
    assert ui and all(f.api_visibility == "private" for f in ui)          # the browser UI still uses them
    tools = [f for f in demo.fns.values() if f.api_name in ms.NAMES]
    assert all(f.queue is False for f in tools)        # Gradio's fast MCP path (two outputs for figure tools)


def test_every_tool_and_parameter_has_a_description():
    for fn in ms.TOOLS:
        summary, params, _ = gr_utils.get_function_description(fn)
        assert len(summary) > 60, fn.__name__
        missing = [p for p, d in params.items() if not d]
        assert not missing, f"{fn.__name__}: {missing}"
        assert "you cannot see" not in summary and "shown to the user" not in summary   # chat-only notes


def test_tool_names_and_choices_match_the_physics_code():
    assert set(ms.NAMES) - {"list_states", "end_session"} <= {t.name for t in at.TOOLS}
    assert set(typing.get_args(ms.LocalChannel)) == set(at.CHANNELS)
    assert set(typing.get_args(ms.GlobalChannel)) == set(at.GLOBAL_CHANNELS)
    assert set(typing.get_args(ms.AnyChannel)) == set(at.CHANNELS) | set(at.GLOBAL_CHANNELS)
    assert set(typing.get_args(ms.Process)) == set(at.PROCESSES) | set(at.GLOBAL_PROCESSES)
    assert set(typing.get_args(ms.Sweep)) == set(at.SWEEPS)
    assert set(typing.get_args(ms.DMStyle)) == set(at._DM_STYLES)
    import workbench
    assert set(typing.get_args(ms.Quantity)) == set(workbench.QUANTITIES)


def test_figure_tools_declare_two_outputs():
    for fn in ms.TOOLS:
        n = len(gr_utils.get_return_types(fn))
        assert n == (2 if fn.__name__.startswith("plot_") or fn.__name__ == "evolve_state" else 1), fn.__name__


def test_deploy_uploads_the_mcp_server_and_tags_the_space():
    assert "mcp_server.py" in deploy_space.APP_FILES
    assert "mcp-server" in deploy_space.TAGS
    card = deploy_space.readme("Quantum Information Agent", repo="animesh1986/quantum-information-agent")
    assert "https://animesh1986-quantum-information-agent.hf.space/gradio_api/mcp/" in card
    assert deploy_space.space_host("Me/My_App.v2") == "me-my-app-v2.hf.space"


# ------------------------------------------------------------------ sessions
def test_first_call_opens_a_session_and_later_calls_keep_the_states():
    sid, out = new("phi+")
    assert ms._ID.match(sid) and out["handle"] == "rho_1" and "session_note" in out
    out = strict(ms.create_state("werner", 0.8, session_id=sid))
    assert out["handle"] == "rho_2" and "session_note" not in out
    m = strict(ms.compute_measures("rho_2", sid))
    assert m["concurrence"] == pytest.approx(0.7) and m["session_id"] == sid
    states = strict(ms.list_states(sid))["states"]
    assert [s["handle"] for s in states] == ["rho_1", "rho_2"] and states[1]["photons"] == 2


def test_clients_are_separate_and_cannot_reach_other_sessions():
    a, _ = new("phi+")
    b, _ = new("psi-")
    assert a != b
    assert strict(ms.list_states(b))["states"][0]["description"] == "psi-"
    with session.use("alice-browser-tab"):                     # a web visitor's session
        at.create_state.invoke({"kind": "phi+"})
    for bad in ("default", "alice-browser-tab", ms.STATELESS_ID, "mcp_" + "0" * 32, "mcp_../../x", a.upper()):
        with pytest.raises(gr.Error, match="Unknown or expired"):
            ms.list_states(bad)
    with pytest.raises(gr.Error, match="session_id is empty"):
        ms.compute_measures("rho_1", "")


def test_a_session_ends_and_expires(monkeypatch):
    sid, _ = new()
    strict(ms.end_session(sid))
    with pytest.raises(gr.Error, match="Unknown or expired"):
        ms.compute_measures("rho_1", sid)
    sid, _ = new()
    session.get(sid).last_used -= session.SESSION_TTL_S + 1               # idle for too long
    with pytest.raises(gr.Error, match="Unknown or expired"):
        ms.compute_measures("rho_1", sid)


def test_mcp_sessions_are_capped_and_never_push_out_web_visitors(monkeypatch):
    monkeypatch.setattr(ms, "MAX_MCP_SESSIONS", 3)
    session.get("webvisitor1")
    ids = [new()[0] for _ in range(6)]
    alive = [i for i in session.active_ids() if i.startswith(ms.PREFIX)]
    assert len(alive) == 3 and alive == ids[-3:]               # the longest-idle MCP sessions went first
    assert "webvisitor1" in session.active_ids()


# ------------------------------------------------------------------ results
def test_results_are_strict_json_without_nan():
    sid, _ = new("psi-")
    h = strict(ms.apply_global_noise("rho_1", "collective_dephasing", 0.5, sid))["handle"]
    m = strict(ms.compute_measures(h, sid))
    assert m["concurrence"] == pytest.approx(1) and m["entanglement_of_formation"] == pytest.approx(1)
    assert ms._finite({"x": [float("nan"), float("inf"), 1.0]}) == {"x": [None, None, 1.0]}


def test_concurrence_is_never_above_one_and_eof_never_nan():
    k = at._channel_kraus("collective_dephasing", 0.5, "both")
    r = qt.apply_kraus(qt.dm(qt.bell_state("psi-")), k)
    assert qt.concurrence(r) <= 1.0 and qt.entanglement_of_formation(r) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    for _ in range(200):
        psi = rng.normal(size=4) + 1j * rng.normal(size=4)
        r = qt.dm(psi / np.linalg.norm(psi))
        c, e = qt.concurrence(r), qt.entanglement_of_formation(r)
        assert 0 <= c <= 1 and np.isfinite(e) and 0 <= e <= 1 + 1e-12


def test_figures_come_back_as_png_and_no_internal_path_leaks():
    sid, _ = new("phi+")
    text, fig = ms.plot_density_matrix("rho_1", "heatmap", "", sid)
    out = strict(text)
    assert isinstance(fig, gr.FileData) and open(fig.path, "rb").read(4) == b"\x89PNG"
    assert "sessions" not in text and "figures/" not in text and out["figure"].startswith("returned as a PNG")
    text, fig = ms.evolve_state("rho_1", "dephasing", 1.0, 1.0, plot_elements=True, session_id=sid)
    out = strict(text)
    assert "2 figures stacked" in out["figure"] and open(fig.path, "rb").read(4) == b"\x89PNG"
    m = strict(ms.compute_measures(out["final_state"], sid))
    assert m["concurrence"] == pytest.approx(np.exp(-2), abs=1e-4)
    sim = strict(ms.simulate_tomography("rho_1", seed=2, session_id=sid))
    assert sim["csv_file"].startswith("uploads/") and "sessions" not in json.dumps(sim)


def test_tool_errors_reach_the_client_as_plain_messages():
    sid, _ = new()
    with pytest.raises(gr.Error) as e:
        ms.compute_measures("rho_7", sid)
    assert str(e.value).startswith("Unknown handle 'rho_7'")             # no quotes, no 'Error invoking tool'
    with pytest.raises(gr.Error, match="must be"):
        ms.mix_states(["rho_1"], [2.0], sid)


def test_stateless_tools_need_no_session():
    assert strict(ms.bit_flip_code(0.1))["p_logical"] == pytest.approx(0.028)
    q = strict(ms.qkd_key_rate(qber_z=0.05))
    assert q["secret_key_fraction"] == pytest.approx(0.42721, abs=1e-5) and "session_id" not in q
    assert strict(ms.get_kraus_operators("amplitude_damping", 0.3))["trace_preserving"]
    assert not [i for i in session.active_ids() if i.startswith(ms.PREFIX)]


# ------------------------------------------------------------------ tomography input
def test_counts_as_pasted_text_equal_the_file_on_the_server():
    with open("data/state_tomo_data.csv", encoding="utf-8") as fh:
        text = fh.read()
    a = strict(ms.reconstruct_state_from_counts(csv_text=text, error_bars=False))
    sid = a["session_id"]
    assert "session_note" in a and a["data"]["file"] == "uploads/counts_1.csv"
    b = strict(ms.reconstruct_state_from_counts(csv_path="data/state_tomo_data.csv", error_bars=False, session_id=sid))
    assert a["key_numbers"] == b["key_numbers"] and a["handle"] == "rho_1" and b["handle"] == "rho_2"


def test_counts_input_is_checked(monkeypatch):
    sid, _ = new()
    with pytest.raises(gr.Error, match="exactly one"):
        ms.reconstruct_state_from_counts(session_id=sid)
    with pytest.raises(gr.Error, match="exactly one"):
        ms.reconstruct_state_from_counts(csv_text="HH,1", csv_path="data/state_tomo_data.csv", session_id=sid)
    monkeypatch.setattr(at, "MAX_CSV_BYTES", 100)
    with pytest.raises(gr.Error, match="larger than 1 MB"):
        ms.reconstruct_state_from_counts(csv_text="HH,1\n" * 50, session_id=sid)
    for path in (".env", "app.py", "../data/x.csv", "data/sessions/other/x.csv", "/etc/passwd"):
        with pytest.raises(gr.Error):
            ms.reconstruct_state_from_counts(csv_path=path, session_id=sid)


# ------------------------------------------------------------------ lock and limits
def test_mcp_calls_wait_for_the_compute_lock():
    sid, _ = new()
    session.COMPUTE_LOCK.acquire()                       # e.g. a Workbench plot of another visitor
    done = []
    th = threading.Thread(target=lambda: done.append(ms.compute_measures("rho_1", sid)))
    th.start()
    th.join(0.3)
    assert not done
    session.COMPUTE_LOCK.release()
    th.join(10)
    assert done and strict(done[0])["concurrence"] == pytest.approx(1)


def test_parallel_clients_get_their_own_results():
    out = {}

    def client(p):
        sid, o = new("werner", p=p)
        out[p] = strict(ms.compute_measures(o["handle"], sid))["concurrence"]
    threads = [threading.Thread(target=client, args=(p,)) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert out == {p: pytest.approx((3 * p - 1) / 2) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}


def test_rate_limit_and_point_limits(monkeypatch):
    monkeypatch.setattr(ms, "CALLS_PER_MINUTE", 3)
    sid, _ = new()
    ms.compute_measures("rho_1", sid)
    ms.bit_flip_code(0.1)
    with pytest.raises(gr.Error, match="3 tool calls per minute"):
        ms.bit_flip_code(0.1)
    monkeypatch.setattr(ms, "CALLS_PER_MINUTE", 0)
    with pytest.raises(gr.Error, match="n_points"):
        ms.plot_measures_vs_noise("phi+", "depolarizing", n_points=100000, session_id=sid)


def test_question_checks_of_the_chat_do_not_block_mcp_calls():
    with session.use(session.DEFAULT_ID):
        at.CURRENT_QUESTION["text"] = "please make psi- for me"    # left over from a chat question
    sid, _ = new("phi+")                                         # the chat check would refuse phi+ here
    assert strict(ms.create_pure_state(amp_HH="1", amp_VV="i", session_id=sid))["handle"] == "rho_2"


# ------------------------------------------------------------------ end to end over HTTP
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_end_to_end_with_a_real_mcp_client(monkeypatch):
    pytest.importorskip("mcp")
    from mcp import ClientSession
    try:
        from mcp.client.streamable_http import streamable_http_client as http_client
    except ImportError:
        from mcp.client.streamable_http import streamablehttp_client as http_client
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    demo = app.build_demo(app.NoModel("none"))
    port = free_port()
    demo.launch(mcp_server=True, server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                quiet=True)
    url = f"http://127.0.0.1:{port}/gradio_api/mcp/"

    async def run():
        async with http_client(url) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = (await s.list_tools()).tools
                schema = next(t for t in tools if t.name == "apply_local_noise").inputSchema
                res = await s.call_tool("create_state", {"kind": "phi+"})
                sid = strict(res.content[0].text)["session_id"]
                fig = await s.call_tool("plot_fringes", {"handle": "rho_1", "session_id": sid})
                err = await s.call_tool("compute_measures", {"handle": "rho_5", "session_id": sid})
                return tools, schema, fig, err
    try:
        tools, schema, fig, err = asyncio.run(run())
    finally:
        demo.close()
    assert len(tools) == 32
    assert schema["properties"]["channel"]["enum"][:2] == ["phase_damping", "amplitude_damping"]
    assert set(schema["required"]) == {"handle", "channel", "param"}
    kinds = [c.type for c in fig.content]
    assert not fig.isError and kinds[:2] == ["text", "image"]
    assert base64.b64decode(fig.content[1].data)[:4] == b"\x89PNG"
    assert strict(fig.content[0].text)["fringes"][0]["visibility"] == pytest.approx(1)
    assert err.isError and err.content[0].text.startswith("Unknown handle 'rho_5'")
    assert os.path.isdir("data")


def test_the_app_shows_the_right_address_and_setup(monkeypatch):
    monkeypatch.delenv("SPACE_HOST", raising=False)
    assert app.base_url() == "http://127.0.0.1:7860"
    monkeypatch.setenv("SPACE_HOST", "animesh1986-quantum-information-agent.hf.space")
    assert ms.server_url(app.base_url()) == "https://animesh1986-quantum-information-agent.hf.space/gradio_api/mcp/"
    monkeypatch.setattr(session, "ON_SPACE", True)
    with gr.Blocks():
        app.mcp_panel()
        texts = [b.value for b in gr.context.get_blocks_context().blocks.values() if isinstance(b, gr.Markdown)]
    assert any("Add custom connector" in t and "https://animesh1986" in t for t in texts)
    assert not any("mcp-remote" in t for t in texts)          # the local bridge is only shown locally
