"""
test_fixes.py - checks for the 2026-10-12b fixes:
  * amplitudes given as expressions ('cos(0.3)') and the cross-check against the user's text
    (catches swapped amplitudes such as 0.2955|HH> + 0.9553|VV> for cos(0.3)|HH> + sin(0.3)|VV>)
  * no "rescaled by 1/1.0000" note for already-normalized inputs
  * the per-question model-call budget (no GraphRecursionError crash)
Run:  pytest -q test_fixes.py
"""
import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agent_tools
from agent_tools import (CURRENT_QUESTION, STORE, compute_measures, create_pure_state,
                         eval_amplitude, plot_measures_vs_noise, state_from_question)


def fresh(question=""):
    STORE.clear()
    CURRENT_QUESTION["text"] = question


def pure(**amps):
    args = {"amp_HH": 0, "amp_HV": 0, "amp_VH": 0, "amp_VV": 0}
    args.update(amps)
    return create_pure_state.invoke(args)


# ------------------------------------------------------------------ amplitude expressions
@pytest.mark.parametrize("text, value", [
    ("0.6", 0.6), (0.6, 0.6), ("cos(0.3)", np.cos(0.3)), ("1/sqrt(3)", 1 / np.sqrt(3)),
    ("sqrt(2/3)", np.sqrt(2 / 3)), ("√2/2", np.sqrt(2) / 2), ("1/√3", 1 / np.sqrt(3)),
    ("i", 1j), ("i/sqrt(2)", 1j / np.sqrt(2)), ("2i", 2j), ("-sin(0.3)", -np.sin(0.3)),
    ("exp(i*pi/4)", np.exp(1j * np.pi / 4)), ("cos(pi/8)", np.cos(np.pi / 8)), ("2^0.5", np.sqrt(2)),
])
def test_eval_amplitude(text, value):
    assert abs(eval_amplitude(text) - value) < 1e-12


@pytest.mark.parametrize("bad", ["__import__('os')", "open('x')", "a", "cos(0.3", "9**9**9"])
def test_eval_amplitude_rejects_anything_else(bad):
    with pytest.raises(ValueError):
        eval_amplitude(bad)


# ------------------------------------------------------------------ reading the user's state
@pytest.mark.parametrize("q, expected", [
    ("Create the state cos(0.3)|HH> + sin(0.3)|VV> and compute", [np.cos(.3), 0, 0, np.sin(.3)]),
    ("Create the state 1/sqrt(3)|HH> + sqrt(2/3)|VV>.", [1 / np.sqrt(3), 0, 0, np.sqrt(2 / 3)]),
    ("Create the state (|HH> + i|VV>)/sqrt(2)", [1, 0, 0, 1j]),
    ("make 0.6|00> + 0.8|11>", [0.6, 0, 0, 0.8]),
    ("state 1/sqrt(2)(|HH> - |VV>)", [1, 0, 0, -1]),
    ("the state |HV> - |VH>", [0, 1, -1, 0]),
    ("1/√3|HH⟩ + √(2/3)|VV⟩", [1 / np.sqrt(3), 0, 0, np.sqrt(2 / 3)]),
])
def test_state_from_question(q, expected):
    v = np.asarray(expected, dtype=complex)
    v /= np.linalg.norm(v)
    _, got = state_from_question(q)
    assert abs(np.vdot(v, got)) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("q", ["a|HH> + b|VV>", "(1+i)/2|HH> + 1/sqrt(2)|VV>", "Create the state cos(0.3)",
                               "compare cos(0.3)|HH>+sin(0.3)|VV> with cos(0.6)|HH>+sin(0.6)|VV>"])
def test_unreadable_states_are_not_checked(q):
    assert state_from_question(q) is None


# ------------------------------------------------------------------ create_pure_state
def test_swapped_amplitudes_are_rejected():          # the Test C bug
    fresh("Create the state cos(0.3)|HH> + sin(0.3)|VV> and compute its measures.")
    with pytest.raises(ValueError, match="do not give the state"):
        pure(amp_HH=0.2955, amp_VV=0.9553)
    assert not STORE


def test_expression_amplitudes_give_the_right_state():
    fresh("Create the state cos(0.3)|HH> + sin(0.3)|VV> and compute its measures.")
    msg = pure(amp_HH="cos(0.3)", amp_VV="sin(0.3)")
    assert "0.9553|HH> + 0.2955|VV>" in msg and "[from HH: cos(0.3), VV: sin(0.3)]" in msg
    assert compute_measures.invoke({"handle": "rho_1"})["concurrence"] == pytest.approx(np.sin(0.6), abs=1e-6)


def test_correct_numbers_are_accepted_too():
    fresh("Create the state cos(0.3)|HH> + sin(0.3)|VV>")
    pure(amp_HH=0.9553, amp_VV=0.2955)
    assert "rho_1" in STORE


def test_wrong_relative_phase_is_rejected():
    fresh("Create the state (|HH> + i|VV>)/sqrt(2) and compute its measures.")
    with pytest.raises(ValueError, match="do not give the state"):
        pure(amp_HH=0.7071, amp_VV=0.7071, phase_VV_deg=-90)
    msg = pure(amp_HH=0.7071, amp_VV=0.7071, phase_VV_deg=90)
    assert "normalized" not in msg and "1.0000" not in msg      # no 'rescaled by 1/1.0000' (Test G)
    msg = pure(amp_HH="1", amp_VV="i")
    assert "e^(i90.0°)" in msg


def test_wrong_sign_is_rejected():
    fresh("cos(0.3)|HH> - sin(0.3)|VV>")
    with pytest.raises(ValueError):
        pure(amp_HH="cos(0.3)", amp_VV="sin(0.3)")
    pure(amp_HH="cos(0.3)", amp_VV="-sin(0.3)")


def test_unnormalized_input_still_reports_normalization():
    fresh()
    msg = pure(amp_HH=1, amp_VV=1)
    assert "normalized: amplitudes divided by 1.4142" in msg


def test_plot_gives_samples_every_0_1():
    fresh()
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["features"]
    assert list(f["samples"]) == [f"{x:.1f}" for x in np.linspace(0, 1, 11)]
    assert f["samples"]["0.3"]["concurrence"] == pytest.approx((1 - 0.6) ** 2, abs=1e-4)


# ------------------------------------------------------------------ model-call budget
class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def test_endless_tool_loop_stops_gracefully_without_recursion_error():
    from graph import MAX_MODEL_CALLS, build_graph
    fresh()
    n = {"calls": 0}

    def forever():
        while True:
            n["calls"] += 1
            yield AIMessage(content="", tool_calls=[{"name": "create_state", "id": f"c{n['calls']}",
                                                     "args": {"kind": "phi+"}}])

    out = build_graph(Scripted(messages=forever())).invoke(
        {"messages": [("user", "Create phi+ again and again")], "retries": 0},
        {"configurable": {"thread_id": "loop"}, "recursion_limit": 60})
    assert n["calls"] == 3              # the same call repeated twice is skipped, then it stops (2026-10-26)
    final = out["messages"][-1]
    assert final.name == "budget_stop" and "Stopped after" in final.content


def test_budget_answer_keeps_last_draft_with_warning():
    from graph import MAX_MODEL_CALLS, build_graph
    fresh()
    tool = lambda k: AIMessage(content="", tool_calls=[{"name": "create_state", "id": f"d{k}",  # noqa: E731
                                                        "args": {"kind": "psi-"}}])
    draft = AIMessage(content="Concurrence is 0.8123 here.")          # 0.8123: no tool made it
    script = [tool(0), draft] + [tool(k) for k in range(1, MAX_MODEL_CALLS)]
    out = build_graph(Scripted(messages=iter(script))).invoke(
        {"messages": [("user", "make psi- a few times")], "retries": 0},
        {"configurable": {"thread_id": "draft"}, "recursion_limit": 60})
    final = out["messages"][-1]
    assert final.name == "budget_stop"
    assert final.content.startswith("Concurrence is 0.8123") and "0.8123]" in final.content


def test_figure_paths_use_forward_slashes():      # 'figures\\phi+...' turned into 'figuresφ+...' on Windows
    fresh()
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})
    assert f["figure"] == "figures/phi+_bit_flip_both.png"
    from display import to_unicode
    assert to_unicode("Saved figure " + f["figure"]) == "Saved figure figures/phi+_bit_flip_both.png"


def test_phase_given_in_words_is_not_blocked():
    fresh("Create the state (|HH> + |VV>)/sqrt(2) with a relative phase of -108 degrees on VV")
    msg = pure(amp_HH=1, amp_VV=1, phase_VV_deg=-108)
    assert "e^(-i108.0°)" in msg
