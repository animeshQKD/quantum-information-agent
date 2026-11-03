"""
test_graph_offline.py - tests the graph wiring WITHOUT any LLM or API key.
A scripted fake model plays the LLM's part, so this checks the graph, the tools
and the verifier only.   Run:  pytest -q test_graph_offline.py
"""
import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent_tools import STORE
from graph import build_graph, check_measures


class ScriptedLLM(GenericFakeChatModel):
    """Returns pre-written AI messages in order; bind_tools is a no-op."""
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def test_dephased_phi_plus_end_to_end():
    STORE.clear()
    script = iter([
        call("create_state", {"kind": "phi+"}, 1),
        call("apply_local_noise", {"handle": "rho_1", "channel": "phase_damping", "param": 0.7}, 2),
        call("compute_measures", {"handle": "rho_2"}, 3),
        call("plot_measures_vs_noise", {"start_kind": "phi+", "channel": "phase_damping"}, 4),
        AIMessage(content="Concurrence is 0.49."),
    ])
    app = build_graph(ScriptedLLM(messages=script))
    out = app.invoke({"messages": [("user", "dephase phi+ with lambda 0.7")], "retries": 0},
                     {"configurable": {"thread_id": "t1"}})
    m = out["results"][0]
    assert np.isclose(m["concurrence"], 0.49) and np.isclose(m["negativity"], 0.245)
    assert np.isclose(m["chsh_max"], 2 * np.sqrt(1 + 0.7 ** 4), atol=1e-5)
    assert all(c["ok"] for c in out["checks"])
    assert out["figures"] and out["figures"][0].endswith(".png")
    assert out["messages"][-1].content == "Concurrence is 0.49."


def test_tool_error_goes_back_to_llm():
    STORE.clear()
    script = iter([call("create_state", {"kind": "werner", "p": 1.5}, 1),
                   AIMessage(content="p must be in [0, 1].")])
    app = build_graph(ScriptedLLM(messages=script))
    out = app.invoke({"messages": [("user", "werner p=1.5")], "retries": 0},
                     {"configurable": {"thread_id": "t2"}})
    tool_msg = [x for x in out["messages"] if x.type == "tool"][0]
    assert "Error" in tool_msg.content and not STORE


def test_verifier_catches_bad_numbers():
    bad = {"purity": 1.0, "concurrence": 1.2, "negativity": 0.0, "chsh_max": 3.0}
    probs = check_measures(bad)
    assert len(probs) == 3   # concurrence range, Tsirelson bound, C/N disagreement


def test_measures_carry_correct_formulas():
    from agent_tools import compute_measures, create_state
    STORE.clear()
    create_state.invoke({"kind": "psi-"})
    m = compute_measures.invoke({"handle": "rho_1"})
    assert "log2(2N+1)" in m["definitions"]
    assert np.isclose(m["log_negativity"], np.log2(2 * m["negativity"] + 1))


class Recorder:
    """Fake LLM that records what it is sent and replies from a script."""
    def __init__(self, replies):
        self.replies, self.seen = iter(replies), []

    def invoke(self, msgs):
        self.seen.append(msgs)
        return next(self.replies)


def test_second_question_sends_only_current_turn():
    STORE.clear()
    rec = Recorder([
        call("create_state", {"kind": "phi+"}, 1),
        call("apply_local_noise", {"handle": "rho_1", "channel": "phase_damping", "param": 0.7}, 2),
        AIMessage(content="Done: rho_2 is dephased phi+."),
        call("compute_measures", {"handle": "rho_2"}, 3),
        AIMessage(content="C = 0.49."),
    ])
    app = build_graph(rec)
    cfg = {"configurable": {"thread_id": "t3"}}
    app.invoke({"messages": [("user", "make dephased phi+")], "retries": 0}, cfg)
    app.invoke({"messages": [("user", "measure that state")], "retries": 0}, cfg)

    first_call_of_turn2 = rec.seen[3]
    system, rest = first_call_of_turn2[0].content, first_call_of_turn2[1:]
    assert [m.type for m in rest] == ["human"]                    # no old tool chatter
    assert "rho_2: phase_damping(0.7) on both qubits of rho_1" in system
    assert "Done: rho_2 is dephased phi+." in system               # previous answer kept short


def test_parallel_tool_calls_get_distinct_handles():
    STORE.clear()
    three = AIMessage(content="", tool_calls=[
        {"name": "create_state", "args": {"kind": "werner", "p": p}, "id": f"w{i}"}
        for i, p in enumerate((0.3, 0.5, 0.8))])
    app = build_graph(ScriptedLLM(messages=iter([three, AIMessage(content="made three")])))
    app.invoke({"messages": [("user", "three werner states")], "retries": 0},
               {"configurable": {"thread_id": "t4"}})
    assert sorted(STORE) == ["rho_1", "rho_2", "rho_3"]


def test_all_files_same_version():
    import agent_tools, graph, llm
    assert agent_tools.FILES_VERSION == graph.FILES_VERSION == llm.FILES_VERSION


def test_prompt_forbids_fake_simulation():
    from graph import SYSTEM_PROMPT
    assert "there is no tool for it" in SYSTEM_PROMPT and "Never write code" in SYSTEM_PROMPT


def test_plot_features_match_analytic_thresholds():
    from agent_tools import plot_measures_vs_noise
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"})["features"]
    assert abs(f["entanglement_vanishes_at"] - (1 - 1 / np.sqrt(3))) < 2e-3   # 0.4226
    assert abs(f["chsh_drops_to_2_at"] - (1 - 2 ** -0.25)) < 2e-3            # 0.1591
    assert f["at_max_noise"]["purity"] == 0.25
    g = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "phase_damping"})["features"]
    assert "only at maximum noise" in str(g["entanglement_vanishes_at"])
