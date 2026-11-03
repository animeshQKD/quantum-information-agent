"""
test_answer_check.py - the automatic check that rejects numbers no tool produced.
Uses real answers from the agent's test runs.   Run:  pytest -q test_answer_check.py
"""
import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_tools import (STORE, apply_local_noise, compute_measures, create_state,
                         plot_measures_vs_noise)
from graph import build_graph, ungrounded_numbers


def tool_msg(result):
    return ToolMessage(json.dumps(result) if isinstance(result, dict) else result, tool_call_id="x")


def test_fabricated_plot_rows_are_caught():
    # Real Test F answer: rows for p = 0.25 and 0.75 were invented by the model.
    feats = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})
    answer = ("| 0.25 | 0.5000 | 0.2500 | 0.6250 | 2.4749 |\n"
              "| 0.50 | 0.0000 | 0.0000 | 0.5000 | 2.0000 |\n"
              "CHSH returns to 2.8284 at p = 1.")
    bad = ungrounded_numbers(answer, [HumanMessage("Plot phi+ under bit-flip"), tool_msg(feats)])
    assert "0.6250" in bad and "2.4749" in bad          # invented purity and CHSH
    assert "2.8284" not in bad and "2.0000" not in bad  # real values pass


def test_good_answer_passes():
    # Real Test B answer (all numbers come from the tools or the question)
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    msgs = [HumanMessage("depolarizing noise with p = 0.6 to qubit A only, then both")]
    for q in ("A", "both"):
        apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.6, "qubits": q})
    for h in ("rho_2", "rho_3"):
        msgs.append(tool_msg(compute_measures.invoke({"handle": h})))
    answer = ("| A only | 0.3700 | 0.10 | 0.05 | 0.1375 | 1.1314 | 0.40 | 0.2940 | 0.1753 |\n"
              "| both | 0.2692 | 0.00 | 0.00 | 0.00 | 0.4525 | 0.16 | 0.0508 | 0.0323 |\n"
              "CHSH 1.13 < 2; with noise p = 0.6 on both, CHSH max ≈ 0.45.")
    assert ungrounded_numbers(answer, msgs) == []


def test_rounding_mismatch_is_caught():
    feats = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"})
    msgs = [tool_msg(feats)]
    assert ungrounded_numbers("Entanglement ends near p ≈ 0.42.", msgs) == []
    assert ungrounded_numbers("Entanglement ends near p ≈ 0.44.", msgs) == ["0.44"]   # 0.423 -> 0.42


def test_known_constants_are_allowed():
    assert ungrounded_numbers("0 <= N <= 0.5; Tsirelson 2.8284; p > 0.707; 0.333", []) == []


def test_graph_retries_then_accepts_clean_answer():
    STORE.clear()
    call = AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "psi-"}, "id": "c1"}])
    measure = AIMessage(content="", tool_calls=[{"name": "compute_measures", "args": {"handle": "rho_1"}, "id": "c2"}])
    script = iter([call, measure,
                   AIMessage(content="C = 1.0 and the purity is 0.97."),   # 0.97 is invented
                   AIMessage(content="C = 1.0 and the purity is 1.0.")])

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    out = build_graph(Scripted(messages=script)).invoke(
        {"messages": [("user", "make psi- and measure it")], "retries": 0},
        {"configurable": {"thread_id": "ac1"}})
    notes = [m.content for m in out["messages"] if m.type == "human" and "VERIFIER" in str(m.content)]
    assert notes and "0.97" in notes[0]
    assert out["messages"][-1].content == "C = 1.0 and the purity is 1.0."


def test_graph_warns_after_max_retries():
    STORE.clear()
    script = iter([AIMessage(content="The value is 0.123.") for _ in range(5)])

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    out = build_graph(Scripted(messages=script)).invoke(
        {"messages": [("user", "tell me a number")], "retries": 0},
        {"configurable": {"thread_id": "ac2"}})
    assert "[Warning: these numbers were not produced by any tool" in out["messages"][-1].content
