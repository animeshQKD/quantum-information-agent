"""
test_agent_checks.py - the automatic checks on the agent's answers (offline, no API calls).
Added after the 10-question run of 2026-10-26:
  * Q7: Werner p = 0.8 was asked for but never created            -> parameters_not_used
  * Q8: a fidelity of 0.9000 was quoted although no tool computed it -> 'no fidelity tool' note
  * Q1/Q3: values from one time were shown under another time     -> exact values at key times
  * Q8: a table written on one line with non-breaking spaces       -> repaired
Run:  pytest -q test_agent_checks.py
"""
import json

import numpy as np
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_tools import STORE, create_state, evolve_state
from graph import build_graph, completion_and_claim_notes, parameters_not_used


def call(name, args, i="x"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i}])


def done(i="x", content="Created rho_1 (werner, p=0.3)."):
    return ToolMessage(content=content, tool_call_id=i, name="create_state")


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def test_skipped_parameter_is_found():
    q = "Create Werner states with p = 0.3, 0.5 and 0.8. Which are entangled?"
    turn = [call("create_state", {"kind": "werner", "p": 0.3}, "a"), done("a"),
            call("create_state", {"kind": "werner", "p": 0.5}, "b"), done("b")]
    assert parameters_not_used(q, turn) == ["0.8"]
    expr = [call("create_pure_state", {"amp_HH": "cos(0.3)", "amp_VV": "sin(0.3)"}, "c"), done("c")]
    assert parameters_not_used("Create cos(0.3)|HH> + sin(0.3)|VV>", expr) == []
    refused = [call("create_pure_state", {"amp_HH": 0.955}, "d"),
               ToolMessage(content="Error: ValueError('Do not guess')", tool_call_id="d", name="create_pure_state",
                           status="error")]
    assert parameters_not_used("Create the state cos(0.3)", refused) == []          # guard refused: no nagging


def test_q7_missing_werner_state_gets_created_end_to_end():
    STORE.clear()
    script = [AIMessage(content="", tool_calls=[
                  {"name": "create_state", "args": {"kind": "werner", "p": p}, "id": f"w{i}"} for i, p in enumerate((0.3, 0.5))]),
              AIMessage(content="", tool_calls=[{"name": "compute_measures", "args": {"handle": h}, "id": h}
                                                for h in ("rho_1", "rho_2")]),
              AIMessage(content="p = 0.5 is entangled."),
              call("create_state", {"kind": "werner", "p": 0.8}, "w2"),
              call("compute_measures", {"handle": "rho_3"}, "m3"),
              AIMessage(content="p = 0.5 and 0.8 are entangled; only 0.8 violates CHSH.")]
    out = build_graph(Scripted(messages=iter(script))).invoke(
        {"messages": [("user", "Create Werner states with p = 0.3, 0.5 and 0.8. Which violate CHSH?")], "retries": 0},
        {"configurable": {"thread_id": "q7"}})
    notes = [m.content for m in out["messages"] if isinstance(m, HumanMessage) and m.content.startswith("VERIFIER")]
    assert len(notes) == 1 and "[parameter]" in notes[0] and "0.8" in notes[0]
    assert sorted(STORE) == ["rho_1", "rho_2", "rho_3"]


def test_fidelity_without_a_tool_is_flagged():
    turn = [call("compute_measures", {"handle": "rho_3"}),
            ToolMessage(content=json.dumps({"concurrence": 0.6, "purity": 0.68}), tool_call_id="x",
                        name="compute_measures")]
    keys = [k for k, _ in completion_and_claim_notes("give its fidelity with phi+", turn,
                                                     "C = 0.6000, purity 0.6800, fidelity 0.9000")]
    assert "no fidelity tool" in keys
    turn_ok = turn + [ToolMessage(content=json.dumps({"fidelity": 0.8}), tool_call_id="y", name="compare_states")]
    keys = [k for k, _ in completion_and_claim_notes("q", turn_ok, "fidelity 0.8000")]
    assert "no fidelity tool" not in keys


def test_evolution_gives_exact_values_at_key_times():
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    f = evolve_state.invoke({"handle": "rho_1", "process": "depolarizing", "gamma": 1.0, "t_max": 1.0})["features"]
    key = f["at_key_times"]
    chsh = next(v for k, v in key.items() if "CHSH reaches 2" in k)
    v = np.exp(-2 * np.log(2) / 4)                                   # Werner visibility at t = ln2/4
    assert chsh["chsh_max"] == 2.0 and abs(chsh["concurrence"] - (3 * v - 1) / 2) < 1e-4
    death = next(v for k, v in key.items() if "vanishes" in k)
    assert death["concurrence"] == 0 and abs(death["purity"] - 1 / 3) < 1e-4


def test_time_evolution_table_is_shown():
    from display import result_panels
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    r = evolve_state.invoke({"handle": "rho_1", "process": "depolarizing", "gamma": 1.0, "t_max": 1.0})
    (title, table), = result_panels([ToolMessage(content=json.dumps(r), tool_call_id="e", name="evolve_state")])
    assert title == "Time evolution"
    assert "| t = 0.1733 (CHSH reaches 2) | 0.5607 | 0.2803 | 0.6250 | 2.0000 |" in table
    assert "sudden death at t = 0.5493" in table


def test_one_line_table_with_non_breaking_spaces():
    from display import to_unicode
    s = "| A | B | |---|---| | 1 | 2 |"
    assert to_unicode(s).splitlines() == ["| A | B |", "|---|---|", "| 1 | 2 |"]


def test_plot_title_for_reconstructed_state_is_not_cut():
    import os
    if not os.path.exists("data/state_tomo_data.csv"):
        import pytest
        pytest.skip("data/state_tomo_data.csv not present")
    from agent_tools import reconstruct_state_from_counts
    STORE.clear()
    h = reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv", "error_bars": False})["handle"]
    f = evolve_state.invoke({"handle": h, "process": "dephasing", "gamma": 1.0, "t_max": 2.0})["features"]
    assert abs(f["entanglement_vanishes_at"] - 1.9297) < 1e-4 and abs(f["concurrence_halves_at"] - 0.3376) < 1e-4


# ------------------------------------------------------------------ second run of 2026-10-26
def _q1_turn():
    from agent_tools import CURRENT_QUESTION, create_pure_state
    STORE.clear()
    CURRENT_QUESTION["text"] = "1/sqrt(3)|HH> + sqrt(2/3)|VV>"
    create_pure_state.invoke({"amp_HH": "1/sqrt(3)", "amp_VV": "sqrt(2/3)"})
    r = evolve_state.invoke({"handle": "rho_1", "process": "amplitude_decay", "gamma": 1.0, "t_max": 3.0})
    return [call("evolve_state", {"handle": "rho_1", "process": "amplitude_decay", "gamma": 1.0, "t_max": 3.0}, "e"),
            ToolMessage(content=json.dumps(r), tool_call_id="e", name="evolve_state")], r


def test_relabelled_time_is_caught():
    """Q1: a row 't = 2.0' showed the values of another sample time (samples are 0, 0.3, ..., 3.0)."""
    from graph import times_not_in_tools
    turn, r = _q1_turn()
    q = "Let it evolve under amplitude decay with rate gamma = 1 on both qubits up to t = 3."
    table = "| Time | Concurrence |\n|---|---|\n| 0.0 | 0.9428 |\n| 0.3 | 0.4424 |\n| 2.0 | 0.0000 |\n| 2.7 | 0.0000 |"
    assert times_not_in_tools(q, turn, table) == ["2.0"]
    assert times_not_in_tools(q, turn, "Entanglement dies at t ≈ 1.228; CHSH reaches 2 at t ≈ 0.288.") == []
    keys = [k for k, _ in completion_and_claim_notes(q, turn, table)]
    assert "time" in keys


def test_false_monotonic_claim_is_removed():
    """Q1: 'Purity increases monotonically' - it first falls to about 0.45, then rises."""
    from graph import remove_false_claims
    turn, _ = _q1_turn()
    ans = ("Entanglement dies at t ≈ 1.228.\n• CHSH violation disappears earlier.\n"
           "• Purity increases monotonically as the state becomes more mixed.")
    fixed, why = remove_false_claims(ans, turn)
    assert "monotonically" not in fixed and "CHSH violation disappears earlier." in fixed
    assert why and why[0].startswith("the purity does not increase monotonically")
    ok = "• Concurrence decreases monotonically until it dies."
    assert remove_false_claims(ok, turn) == (ok, [])


def test_local_unitary_text_says_constant():
    STORE.clear()
    create_state.invoke({"kind": "psi-"})
    f = evolve_state.invoke({"handle": "rho_1", "process": "none", "gamma": 0, "t_max": 3.0, "omega": 2.0})["features"]
    assert f["entanglement_vanishes_at"].startswith("never: concurrence is constant")


# ------------------------------------------------------------------ third run of 2026-10-26
def test_previous_result_time_is_allowed_in_a_comparison():
    """Q2 compared with the death time 1.2279 from Q1; that is not a relabelled time."""
    from graph import times_not_in_tools
    turn1, r1 = _q1_turn()
    r2 = evolve_state.invoke({"handle": "rho_1", "process": "amplitude_decay", "gamma": 1.0, "t_max": 3.0,
                              "qubits": "B"})
    turn2 = [call("evolve_state", {"handle": "rho_1", "process": "amplitude_decay", "gamma": 1.0,
                                   "t_max": 3.0, "qubits": "B"}, "f"),
             ToolMessage(content=json.dumps(r2), tool_call_id="f", name="evolve_state")]
    ans = "Here C stays > 0, while with both qubits damped it died at t ≈ 1.23."
    assert times_not_in_tools("same rate and time", turn2, ans, turn1 + turn2) == []


def test_steadily_increasing_purity_is_removed():
    from graph import remove_false_claims
    turn, _ = _q1_turn()
    fixed, why = remove_false_claims("• Purity steadily increases as the state becomes more mixed.\n• Done.", turn)
    assert "steadily" not in fixed and "• Done." in fixed and why


def test_row_value_from_another_time_is_caught():
    """Q2: 'rho_2 (both qubits) | 0.0000 | 0.0000 | 0.5912 | 1.1549' - purity/CHSH were from t = 1.5."""
    from graph import wrong_row_values
    turn, r = _q1_turn()
    row_bad = "| rho_2 (both qubits) | 0.0000 | 0.0000 | 0.5912 | 1.1549 |"
    row_ok = "| rho_2 (both qubits) | 0.0000 | 0.0000 | 0.8778 | 1.7502 |"
    assert wrong_row_values(row_bad, turn) == ["rho_2: 0.5912", "rho_2: 1.1549"]
    assert wrong_row_values(row_ok, turn) == []


def test_sudden_death_denial_is_caught():
    """Q5: the tool found sudden death at 1.9297 but the answer said 'there is no sudden death'."""
    from graph import sudden_death_contradiction
    turn, _ = _q1_turn()                                      # amplitude decay on both: dies at 1.2279
    assert sudden_death_contradiction(turn, "It vanishes only asymptotically - there is no sudden death.")
    assert sudden_death_contradiction(turn, "Sudden death occurs at t ≈ 1.228.") is None


def test_bell_local_overclaim_is_removed():
    from graph import remove_false_claims
    fixed, why = remove_false_claims("• After t ≈ 0.1733 it is Bell-local.\n• Entangled until 0.5493.", [])
    assert "Bell-local" not in fixed and "Entangled until 0.5493." in fixed and why


def test_wrong_bell_state_handle_is_refused():
    """Q4: the model evolved rho_4 (phi+) although the user asked about psi-."""
    import pytest
    from agent_tools import CURRENT_QUESTION
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    CURRENT_QUESTION["text"] = "Evolve psi- up to t = 3 under the local Hamiltonian"
    with pytest.raises(ValueError, match="create_state\\(kind='psi-'\\)"):
        evolve_state.invoke({"handle": "rho_1", "process": "none", "gamma": 0, "t_max": 3.0, "omega": 2.0})
    CURRENT_QUESTION["text"] = "Evolve rho_1 and compare it with psi-"          # handle named: allowed
    evolve_state.invoke({"handle": "rho_1", "process": "none", "gamma": 0, "t_max": 1.0, "omega": 2.0})
    CURRENT_QUESTION["text"] = ""


def test_unknown_kind_gets_a_clear_message():
    import pytest
    with pytest.raises(ValueError, match="kind must be"):
        create_state.invoke({"kind": "bell"})


def test_table_broken_mid_row_is_repaired():
    from display import to_unicode
    s = "| A | B | |---|---| | x |\n 1 | 2 |"
    lines = to_unicode(s).splitlines()
    assert lines[1] == "|---|---|---|" and lines[2] == "| x | 1 | 2 |"


# ------------------------------------------------------------------ fourth run of 2026-10-26
def test_q1_relabelled_time_and_trend_with_kets_in_question():
    """Q1 again: 't = 2.0' slipped through because '2' from sqrt(2/3) in the question counted as a
    time, and 'Purity increases monotonically as the population decays' was skipped because of 'decays'."""
    from graph import remove_false_claims, times_not_in_tools
    turn, _ = _q1_turn()
    q = ("Create the state 1/sqrt(3)|HH> + sqrt(2/3)|VV> and let it evolve under amplitude decay with "
         "rate gamma = 1 on both qubits up to t = 3.")
    ans = "| State | Purity |\n|---|---|\n| t = 0.3 | 0.5535 |\n| t = 2.0 | 0.7340 |"
    assert times_not_in_tools(q, turn, ans, turn) == ["2.0"]
    fixed, why = remove_false_claims("• Purity increases monotonically as the excited population decays.", turn)
    assert why and "monotonically" not in fixed


def test_repeated_identical_tool_calls_are_skipped():
    """Q2/Q3 with the local model: the same evolve_state call was made 8 times."""
    STORE.clear()
    same = {"name": "create_state", "args": {"kind": "phi+"}}
    script = [AIMessage(content="", tool_calls=[{**same, "id": "a"}]),
              AIMessage(content="", tool_calls=[{**same, "id": "b"}]),
              AIMessage(content="Created phi+ as rho_1.")]
    out = build_graph(Scripted(messages=iter(script))).invoke(
        {"messages": [("user", "create phi+")], "retries": 0}, {"configurable": {"thread_id": "rep"}})
    assert list(STORE) == ["rho_1"]                                   # executed only once
    tool_texts = [m.content for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_texts[1].startswith("Already done in this question")
    assert out["messages"][-1].content == "Created phi+ as rho_1."


# ------------------------------------------------------------------ run with qwen3:14b (Ollama)
def test_create_then_evolve_in_one_step_runs_in_order():
    """Q1/Q6: create + evolve (or evolve + compare) requested together; the second call used the
    handle made by the first and failed with 'Unknown handle' because the calls ran in parallel."""
    STORE.clear()
    both = AIMessage(content="", tool_calls=[
        {"name": "create_state", "args": {"kind": "phi+"}, "id": "a"},
        {"name": "evolve_state", "args": {"handle": "rho_1", "process": "dephasing", "gamma": 1.0, "t_max": 1.0}, "id": "b"},
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_2"}, "id": "c"}])
    out = build_graph(Scripted(messages=iter([both, AIMessage(content="Done: F = 0.6.")]))).invoke(
        {"messages": [("user", "make phi+, dephase it and compare")], "retries": 0},
        {"configurable": {"thread_id": "seq"}})
    texts = [m.content for m in out["messages"] if isinstance(m, ToolMessage)]
    assert not any(t.startswith("Error") for t in texts) and sorted(STORE) == ["rho_1", "rho_2"]


def test_answer_after_a_failed_tool_is_sent_back():
    """Q1: evolve_state failed, the answer still described an evolution (and invented numbers)."""
    turn = [call("evolve_state", {"handle": "rho_9"}, "e"),
            ToolMessage(content="Error: ValueError(\"Unknown handle 'rho_9'\")", tool_call_id="e",
                        name="evolve_state", status="error")]
    keys = [k for k, _ in completion_and_claim_notes("evolve it", turn, "It decays asymptotically.")]
    assert "failed tool" in keys
    keys = [k for k, _ in completion_and_claim_notes("evolve it", turn, "The evolution failed: unknown handle.")]
    assert "failed tool" not in keys


def test_halving_time_given_as_chsh_time_is_caught():
    """2026-10-27 Q1: 'Bell nonlocality is lost at t ≈ 0.276' (0.276 = halving time; CHSH = 2 at 0.2877)."""
    from graph import misattributed_key_times
    turn, _ = _q1_turn()
    bad = misattributed_key_times(turn, "• Bell-nonlocality (CHSH > 2) is lost earlier, at t ≈ 0.276.", turn)
    assert bad and "0.2877" in bad[0]
    ok = ("• CHSH drops to 2 at t ≈ 0.288. • Entanglement vanishes at t ≈ 1.228. "
          "• The concurrence halves at t ≈ 0.276.")
    assert misattributed_key_times(turn, ok, turn) == []


# ------------------------------------------------------------------ first run with OpenAI (gpt-4o-mini)
def test_named_bell_state_cannot_be_replaced_by_another_state():
    """Q3: 'Evolve phi+ under depolarizing noise' - the model evolved rho_1 = 1/√3|HH> + √(2/3)|VV>."""
    import pytest
    from agent_tools import CURRENT_QUESTION, create_pure_state
    STORE.clear()
    CURRENT_QUESTION["text"] = "1/sqrt(3)|HH> + sqrt(2/3)|VV>"
    create_pure_state.invoke({"amp_HH": "1/sqrt(3)", "amp_VV": "sqrt(2/3)"})
    CURRENT_QUESTION["text"] = "Evolve phi+ under depolarizing noise with rate gamma = 1 on both qubits up to t = 1."
    with pytest.raises(ValueError, match="create_state\\(kind='phi\\+'\\)"):
        evolve_state.invoke({"handle": "rho_1", "process": "depolarizing", "gamma": 1.0, "t_max": 1.0})
    CURRENT_QUESTION["text"] = "Now evolve the same state under dephasing and compare it with phi+."
    evolve_state.invoke({"handle": "rho_1", "process": "dephasing", "gamma": 1.0, "t_max": 1.0})   # refers back: ok
    CURRENT_QUESTION["text"] = ""


def test_markdown_images_become_a_figure_note():
    from display import to_unicode
    assert to_unicode("See ![Entanglement Evolution](figures/a.png) here") == "See (figure: figures/a.png) here"


# ------------------------------------------------------------------ 2026-10-28 run: local / quantum-correlation claims
def test_local_unitary_wording_is_not_a_bell_local_claim():
    from graph import remove_false_claims
    for ok in ("• The operation is local, so entanglement is unchanged.",
               "• The half-wave plate is a local unitary operation.",
               "• The CHSH maximum remains at 2.8284, indicating maximal Bell nonlocality."):
        assert remove_false_claims(ok, [])[1] == [], ok
    for bad in ("• After t = 0.1733 the state is local.", "• The correlations become effectively local."):
        assert remove_false_claims(bad, [])[1], bad


def test_separable_does_not_mean_no_quantum_correlations():
    from langchain_core.messages import ToolMessage
    from graph import remove_false_claims
    text = ("Entanglement vanishes at t = 0.5493.\n"
            "• After entanglement vanishes, the state is separable and no longer exhibits quantum correlations.\n"
            "• The transformation changes the basis but not the quantum correlations.")
    out, why = remove_false_claims(text, [])
    assert "no longer exhibits" not in out and "changes the basis" in out and why
    zero = [ToolMessage(content='{"discord": 0.0}', name="compute_measures", tool_call_id="x")]
    assert remove_false_claims(text, zero)[1] == []        # discord computed to be 0: claim is fine


def test_convention_warning_of_phase_compensation_is_appended_when_missing():
    from langchain_core.messages import ToolMessage
    from graph import required_notes
    warn = ("this phase comes from a reconstruction, so its SIGN depends on the R/L convention used there. "
            "With the opposite convention α = -108.28° and the HWP angle is 17.93°. Use the setting ...")
    turn = [ToolMessage(content=json.dumps({"convention_warning": warn}), name="compensate_phase",
                        tool_call_id="c")]
    assert required_notes(turn, "The HWP angle is 72.07°.")[0].startswith("R/L convention: This phase")
    assert required_notes(turn, "72.07° (or 17.93° in the other convention)") == []
    assert required_notes([], "anything") == []


def test_percent_of_a_tool_fraction_is_grounded():
    from langchain_core.messages import ToolMessage
    from graph import ungrounded_numbers
    msgs = [ToolMessage(content='{"secret_key_fraction": 0.61122, "qber": 0.01606}', name="qkd_key_rate",
                        tool_call_id="x")]
    assert ungrounded_numbers("Key fraction 61.12% (0.6112), QBER 1.61 %.", msgs) == []
    assert ungrounded_numbers("Key fraction 62.5% and 0.7123.", msgs) == ["62.5", "0.7123"]


# ------------------------------------------------------------------ 2026-10-28 run: significance questions
def _turn_with(reconstructed="rho_2", certified=None):
    from langchain_core.messages import AIMessage, ToolMessage
    turn = [AIMessage(content="", tool_calls=[{"name": "reconstruct_state_from_counts", "id": "r", "args": {}}]),
            ToolMessage(content=json.dumps({"handle": reconstructed}), name="reconstruct_state_from_counts",
                        tool_call_id="r")]
    if certified:
        turn += [AIMessage(content="", tool_calls=[{"name": "certify_correlations", "id": "c",
                                                    "args": {"handle": certified}}]),
                 ToolMessage(content=json.dumps({"tests": []}), name="certify_correlations", tool_call_id="c")]
    return turn


def test_significance_question_needs_certification_of_the_new_state():
    from graph import significance_not_certified
    q = "Reconstruct data/x.csv. Is the CHSH violation significant?"
    assert significance_not_certified(q, _turn_with()) == ["rho_2"]
    assert significance_not_certified(q, _turn_with(certified="rho_1")) == ["rho_2"]   # wrong (old) state
    assert significance_not_certified(q, _turn_with(certified="rho_2")) == []
    assert significance_not_certified("Reconstruct data/x.csv. How entangled is it?", _turn_with()) == []


def test_chsh_settings_caveat_is_appended_when_missing():
    from langchain_core.messages import ToolMessage
    from graph import required_notes
    cav = ["CHSH max and F3 use the measurement settings that are optimal for this state; ...", "x"]
    data = {"caveats": cav, "tests": [{"property": "Bell nonlocal (CHSH > 2)", "holds": True, "n_sigma": 29.7}]}
    turn = [ToolMessage(content=json.dumps(data), name="certify_correlations", tool_call_id="c")]
    assert required_notes(turn, "CHSH = 2.4572, 29.7σ above 2.")[0].startswith("Note: CHSH max and F3")
    assert required_notes(turn, "CHSH = 2.4572 with the optimal settings.") == []
    assert required_notes(turn, "The state is entangled.") == []


def test_chsh_not_violated_so_not_nonlocal_is_corrected_not_deleted():
    from graph import remove_false_claims
    text = "• It does not violate the CHSH inequality (CHSH ≤ 2), so it is not Bell nonlocal."
    out, why = remove_false_claims(text, [])
    assert out == ("• It does not violate the CHSH inequality (CHSH ≤ 2); this alone does not prove "
                   "that a local hidden-variable model exists.")
    assert why == ["no CHSH violation does not by itself prove that a state is Bell local"]
    assert remove_false_claims("• The state is Bell nonlocal (29.7σ).", [])[1] == []


def test_chsh_settings_note_only_for_measured_states_that_violate():
    from langchain_core.messages import ToolMessage
    from graph import required_notes
    cav = ["CHSH max and F3 use the measurement settings that are optimal for this state; ...", "x"]
    exact = {"caveats": cav, "tests": [{"property": "Bell nonlocal (CHSH > 2)", "holds": False}]}
    turn = [ToolMessage(content=json.dumps(exact), name="certify_correlations", tool_call_id="c")]
    assert required_notes(turn, "No CHSH violation (1.5556).") == []


# ------------------------------------------------------------------ benchmark 2026-10-27: early stop
def test_duplicate_call_next_to_new_calls_is_not_looping():
    from langchain_core.messages import AIMessage, ToolMessage
    from graph import REPEAT_NOTE, only_repeats_responses
    progress = [AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"}, "id": "a"}]),
                ToolMessage(content="Created rho_1 (phi+).", tool_call_id="a", name="create_state"),
                AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"}, "id": "b"},
                                                  {"name": "compute_measures", "args": {"handle": "rho_1"}, "id": "c"}]),
                ToolMessage(content="{}", tool_call_id="c", name="compute_measures"),
                ToolMessage(content=REPEAT_NOTE, tool_call_id="b", name="create_state")]
    assert only_repeats_responses(progress) == 0
    loop = progress + [AIMessage(content="", tool_calls=[{"name": "compute_measures", "args": {"handle": "rho_1"},
                                                          "id": "d"}]),
                       ToolMessage(content=REPEAT_NOTE, tool_call_id="d", name="compute_measures")]
    assert only_repeats_responses(loop) == 1


def test_budget_message_says_why_it_stopped():
    from langchain_core.messages import AIMessage, HumanMessage
    from graph import _budget_answer
    msgs = [HumanMessage("q"), AIMessage(content="draft answer")]
    assert "after 1 model calls because the model kept repeating" in _budget_answer(msgs, 0, "repeats").content
    assert "[Stopped after 1 model calls, without" in _budget_answer(msgs, 0, "budget").content


def test_nonlocal_influence_wording_is_corrected():
    from graph import remove_false_claims
    text = ("- It exhibits EPR steering, meaning one party can nonlocally affect the other's state in a way "
            "no local hidden state model can explain.")
    out, why = remove_false_claims(text, [])
    assert "nonlocally affect" not in out and out.startswith("- It exhibits EPR steering")
    assert "without any signalling" in out and why


# ------------------------------------------------------------------ 2026-10-27 20:01 benchmark answers
def test_never_violates_is_checked_against_the_plotted_samples():
    from langchain_core.messages import ToolMessage
    import agent_tools
    from graph import remove_false_claims
    r = agent_tools.plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})
    msgs = [ToolMessage(content=json.dumps(r), name="plot_measures_vs_noise", tool_call_id="p")]
    text = ("The CHSH maximum never drops below the local bound of 2.\n"
            "- The CHSH value never violates the local bound at intermediate noise but returns to maximal "
            "violation at full noise.\n- Purity recovers at p = 1.")
    out, why = remove_false_claims(text, msgs)
    assert "never violates" not in out and "never drops below" in out and "Purity recovers" in out
    assert why and "above 2 at intermediate values" in why[0]


def test_qkd_question_needs_the_qkd_tool():
    from langchain_core.messages import AIMessage
    from graph import required_tools_missing
    q = ("Reconstruct data/state_tomo_data.csv and use the state for entanglement-based BB84. Can a secret "
         "key be made, and what changes after compensating the phase?")
    turn = [AIMessage(content="", tool_calls=[{"name": "reconstruct_state_from_counts", "args": {}, "id": "a"},
                                              {"name": "compensate_phase", "args": {}, "id": "b"}])]
    assert required_tools_missing(q, turn, "It can produce a secret key.") == [("QKD / secret keys", "qkd_key_rate")]
    turn.append(AIMessage(content="", tool_calls=[{"name": "qkd_key_rate", "args": {"handle": "rho_1"}, "id": "c"}]))
    assert required_tools_missing(q, turn, "x") == []
    # a conceptual question (no tools at all) is not a computational request
    assert required_tools_missing("What is the difference between entanglement and EPR steering?", [], "x") == []


# ------------------------------------------------------------------ 2026-10-27 21:23 benchmark answers
def test_maximally_mixed_claim_needs_purity_one_quarter():
    from langchain_core.messages import ToolMessage
    import agent_tools
    from graph import remove_false_claims
    flip = agent_tools.plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})
    msgs = [ToolMessage(content=json.dumps(flip), name="plot_measures_vs_noise", tool_call_id="p")]
    text = ("- Purity follows a similar pattern, indicating the state becomes maximally mixed at p = 0.5.\n"
            "- Bit-flip noise at p = 1 acts as a local unitary.")
    out, why = remove_false_claims(text, msgs)
    assert "maximally mixed" not in out and "local unitary" in out and "purity 1/4" in why[0]
    dep = agent_tools.plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"})
    msgs = [ToolMessage(content=json.dumps(dep), name="plot_measures_vs_noise", tool_call_id="q")]
    ok = "- Purity decreases from 1 to 0.25 (maximally mixed) at p = 1."
    assert remove_false_claims(ok, msgs)[1] == []
    assert remove_false_claims("- The state is not maximally mixed.", [ToolMessage(
        content='{"purity": 0.5}', name="compute_measures", tool_call_id="r")])[1] == []
