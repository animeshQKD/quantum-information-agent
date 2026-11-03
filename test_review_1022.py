"""
test_review_1022.py - regression tests for the problems seen in the two runs of 2026-10-22.
Each test reproduces one observed failure offline (no API calls), so a later change cannot
bring it back silently.
First run:
  * false alarms of the number check (typographic minus signs, numbers from an earlier turn)
  * an unfinished comparison ("compare rho_1 with each of them") gets a concrete instruction
  * comparison table, fit verdict (p-value, sigma deviations), R/L convention note
Second run:
  * phi- named in the question but never created (Q4)
  * '1.1 × 10⁻⁵' flagged although the tool gave 1.1e-05 (Q6)
  * a Markdown table written on one line (Q6)
  * false claims: discord 'beyond entanglement' (Q3), errors 'have been scaled' (Q1)
Run:  pytest -q test_review_1022.py
"""
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_tools import STORE, compare_states, create_state
from graph import build_graph, suggest_missing_calls, ungrounded_numbers


def tool(content, name="t"):
    return ToolMessage(content=content if isinstance(content, str) else json.dumps(content),
                       tool_call_id="x", name=name)


def test_typographic_minus_is_not_a_false_alarm():          # Q2: '–0.113 + 0.343 i'
    src = [tool({"elements": ["ρ[HH,VV] = -0.113+0.343i", "ρ[HV,VV] = +0.051-0.030i"]})]
    assert ungrounded_numbers("ρ[HH,VV] = –0.113 + 0.343 i and 0.051 – 0.030 i", src) == []
    assert ungrounded_numbers("ρ = −0.114", src) == ["0.114"]


def test_numbers_from_an_earlier_turn_are_grounded():       # Q2 quoted concurrence 0.72 from Q1
    msgs = [HumanMessage("reconstruct"), tool({"concurrence": 0.7213, "negativity": 0.3599}),
            AIMessage("C = 0.7213"), HumanMessage("plot it"), tool({"figure": "f.png"})]
    assert ungrounded_numbers("Together with C ≈ 0.72 and N ≈ 0.36 ...", msgs) == []
    assert ungrounded_numbers("N ≈ 0.35", msgs) == ["0.35"]      # 0.3599 rounds to 0.36, not 0.35


def test_missing_comparison_gets_a_concrete_call():
    turn = [AIMessage(content="", tool_calls=[
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_3"}, "id": "a"},
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_2"}, "id": "b"}])]
    assert suggest_missing_calls(turn, ["rho_4"]) == "compare_states(handle_a='rho_1', handle_b='rho_4')"


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def test_q4_unfinished_comparison_is_completed():
    STORE.clear()
    create_state.invoke({"kind": "psi-"})                                  # stands in for rho_1
    make = AIMessage(content="", tool_calls=[
        {"name": "create_state", "args": {"kind": k}, "id": f"m{i}"} for i, k in enumerate(("phi+", "phi-"))])
    two = AIMessage(content="", tool_calls=[
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_2"}, "id": "c1"}])
    early = AIMessage(content="Done.")
    third = AIMessage(content="", tool_calls=[
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_3"}, "id": "c2"}])
    final = AIMessage(content="Both fidelities are 0.")
    out = build_graph(Scripted(messages=iter([make, two, early, third, final]))).invoke(
        {"messages": [("user", "create phi+ and phi- and compare rho_1 with each of them")], "retries": 0},
        {"configurable": {"thread_id": "q4"}})
    notes = [m.content for m in out["messages"] if isinstance(m, HumanMessage) and m.content.startswith("VERIFIER")]
    assert len(notes) == 1 and "compare_states(handle_a='rho_1', handle_b='rho_3')" in notes[0]
    assert [m.name for m in out["messages"] if isinstance(m, ToolMessage)].count("compare_states") == 2


def test_comparison_table_names_the_states():
    from display import result_panels
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    create_state.invoke({"kind": "phi-"})
    r = compare_states.invoke({"handle_a": "rho_1", "handle_b": "rho_2"})
    panels = result_panels([tool(r, "compare_states")])
    assert panels[0][0] == "Comparison"
    assert "rho_1 (phi+)" in panels[0][1] and "rho_2 (phi-)" in panels[0][1] and "0.0000" in panels[0][1]


def test_fit_verdict_and_convention_note(tmp_path):
    import numpy as np
    import qi_tools as qt
    from agent_tools import reconstruct_state_from_counts
    from test_tomography import write_csv
    STORE.clear()
    labels = [a + b for a in "HVDARL" for b in "HVDARL"]
    rng = np.random.default_rng(0)
    true = qt.werner_state(0.9, "phi+")
    P = qt.tomography_projectors(labels)
    counts = {k: int(rng.poisson(3000 * np.real(np.trace(true @ p)))) for k, p in zip(labels, P)}
    counts["DD"] = int(counts["DD"] * 1.15)                               # a "misaligned" setting
    path = tmp_path / "bad.csv"
    write_csv(path, counts)
    r = reconstruct_state_from_counts.invoke({"csv_path": str(path), "error_bars": False})
    fit = r["fit_quality"]
    assert fit["p_value"] < 0.01 and fit["verdict"].startswith("NOT consistent")
    assert "multiply the statistical error bars" in fit["error_bar_scale"]
    assert fit["largest_deviations"][0].startswith("DD:") and "σ)" in fit["largest_deviations"][0]
    note = r["convention_note"]
    assert "DIFFERENT states" in note and "describes the lab state" in note


def test_too_large_request_is_not_called_a_rate_limit():
    from llm import _reason
    msg = ("Error code: 413 - {'error': {'message': 'Request too large for model on tokens per minute',"
           " 'code': 'rate_limit_exceeded'}}")
    assert _reason(RuntimeError(msg)) == "request too large"


# ------------------------------------------------------------------ second run of 2026-10-22
def test_named_bell_state_that_was_never_created_is_caught():         # Q4: phi- forgotten
    from agent_tools import DESC
    from graph import bell_states_missing
    STORE.clear()
    DESC.clear()
    q = "Create the state ... and also create phi+ and phi-. Compare rho_1 with each of them."
    turn = [AIMessage(content="", tool_calls=[
        {"name": "create_pure_state", "args": {"amp_HH": "1", "amp_VV": "1"}, "id": "a"},
        {"name": "create_state", "args": {"kind": "phi+"}, "id": "b"},
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_3"}, "id": "c"}])]
    assert bell_states_missing(q, turn) == ["phi-"]
    assert bell_states_missing("What is phi-?", []) == []                    # conceptual: no tools
    plot = [AIMessage(content="", tool_calls=[{"name": "plot_measures_vs_noise",
                                               "args": {"start_kind": "phi+", "channel": "bit_flip"}, "id": "p"}])]
    assert bell_states_missing("Plot phi+ under bit-flip noise", plot) == []


def test_q4_forgotten_phi_minus_gets_created_end_to_end():
    STORE.clear()
    create_state.invoke({"kind": "psi-"})                                   # stands in for rho_1
    one = AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"}, "id": "a"}])
    cmp1 = AIMessage(content="", tool_calls=[
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_2"}, "id": "b"}])
    early = AIMessage(content="Fidelity with phi+ is 0.")
    fix = AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi-"}, "id": "c"}])
    cmp2 = AIMessage(content="", tool_calls=[
        {"name": "compare_states", "args": {"handle_a": "rho_1", "handle_b": "rho_3"}, "id": "d"}])
    final = AIMessage(content="Both fidelities are 0.")
    out = build_graph(Scripted(messages=iter([one, cmp1, early, fix, cmp2, final]))).invoke(
        {"messages": [("user", "create phi+ and phi-, compare rho_1 with each")], "retries": 0},
        {"configurable": {"thread_id": "q4b"}})
    notes = [m.content for m in out["messages"] if isinstance(m, HumanMessage) and m.content.startswith("VERIFIER")]
    assert len(notes) == 1 and "[missing]" in notes[0] and "create_state(kind='phi-')" in notes[0]
    assert out["messages"][-1].content == "Both fidelities are 0."


def test_scientific_notation_is_grounded():                             # Q6: 1.1 × 10⁻⁵
    assert ungrounded_numbers("p-value 1.1 × 10⁻⁵", [tool({"p_value": 1.1e-05})]) == []


def test_one_line_table_is_repaired():
    from display import to_unicode
    s = "| Largest deviations | |---|---|---| | AD | 605 | 534 | | VA | 428 | 375 |"
    lines = to_unicode(s).splitlines()
    assert lines[1] == "|---|---|---|" and lines[2] == "| AD | 605 | 534 |" and len(lines) == 4


def test_false_claims_are_removed_without_asking_the_model_again():
    """Q3: re-asking the model returned a shortened answer; now the bad bullet is just deleted."""
    from graph import remove_false_claims
    ans = ("**Purity 0.7643**\n\n| Measure | Value |\n|---|---|\n| Discord | 0.4294 |\n\n"
           "**What it means**\n• Entangled.\n• Discord reveals correlations beyond entanglement.\n")
    fixed, why = remove_false_claims(ans, [])
    assert "beyond entanglement" not in fixed and "| Discord | 0.4294 |" in fixed and "• Entangled." in fixed
    assert why == ["discord is not a correlation 'beyond entanglement': it includes entanglement"]
    fixed, why = remove_false_claims("C = 0.7213, close to the maximal value 1. Good.", [])
    assert fixed == "Good." and why


def test_scaled_claim_only_removed_when_false():
    from graph import remove_false_claims
    report = tool({"uncertainty_1sigma": {"concurrence": 0.012},
                   "uncertainty_realistic": {"concurrence": 0.021}})
    true_claim = "C = 0.7213 ± 0.021. Uncertainties are statistical errors scaled by 1.72."
    false_claim = "C = 0.7213 ± 0.012. Uncertainties have been scaled by 1.72."
    assert remove_false_claims(true_claim, [report])[1] == []
    assert remove_false_claims(false_claim, [report])[0] == "C = 0.7213 ± 0.012."


def test_answer_check_removes_claim_in_graph():
    STORE.clear()
    call = AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"}, "id": "a"}])
    ans = AIMessage(content="Made phi+.\n• Discord shows correlations beyond entanglement.\n• Pure state.")
    out = build_graph(Scripted(messages=iter([call, ans]))).invoke(
        {"messages": [("user", "make phi+")], "retries": 0}, {"configurable": {"thread_id": "cl"}})
    final = out["messages"][-1].content
    assert "shows correlations beyond" not in final and "• Pure state." in final and "Answer check removed" in final


def test_realistic_error_bars_for_a_poor_fit(tmp_path):
    import numpy as np
    import qi_tools as qt
    from agent_tools import reconstruct_state_from_counts
    from display import format_tomography_report
    from test_tomography import write_csv
    STORE.clear()
    labels = [a + b for a in "HVDARL" for b in "HVDARL"]
    true = qt.werner_state(0.9, "phi+")
    rng = np.random.default_rng(0)
    counts = {k: int(rng.poisson(3000 * np.real(np.trace(true @ p))))
              for k, p in zip(labels, qt.tomography_projectors(labels))}
    counts["DD"] = int(counts["DD"] * 1.15)
    write_csv(tmp_path / "bad.csv", counts)
    r = reconstruct_state_from_counts.invoke({"csv_path": str(tmp_path / "bad.csv")})
    stat, real = r["uncertainty_1sigma"]["concurrence"], r["uncertainty_realistic"]["concurrence"]
    assert real > stat
    assert "Realistic ± (scaled)" in format_tomography_report(r)


def test_daily_limit_is_named():
    from llm import _reason
    assert _reason(RuntimeError("429 Rate limit reached on tokens per day (TPD): Limit 200000")) == \
        "daily limit reached"


# ------------------------------------------------------------------ third run of 2026-10-22
def test_absolute_value_bars_do_not_break_tables():                    # Q2: '(|ρ| = 0.361)'
    from display import to_unicode
    s = "| Element | Value |\n|---|---|\n| ρ[HH,VV] | –0.113 + 0.343 i (|ρ| = 0.361) |\n| rest | |x| < 0.05 |"
    lines = to_unicode(s).splitlines()
    assert lines[1] == "|---|---|"
    assert lines[2] == "| ρ[HH,VV] | –0.113 + 0.343 i (∣ρ∣ = 0.361) |" and lines[3] == "| rest | ∣x∣ < 0.05 |"


def test_advice_to_scale_errors_is_not_removed():                       # Q6 false positive
    from graph import remove_false_claims
    report = tool({"uncertainty_realistic": {"concurrence": 0.021}})
    advice = "• Realistic uncertainties are about 1.7 times larger: the error bars should be scaled by 1.72."
    assert remove_false_claims(advice, [report]) == (advice, [])
    no_numbers = "• The error bars have been scaled by 1.72."                    # no ± shown: nothing to check
    assert remove_false_claims(no_numbers, [report])[1] == []


def test_unknown_errors_show_the_start_of_the_message():
    from llm import _reason
    assert _reason(ValueError("Invalid argument provided to Gemini: 400 schema")).startswith(
        "ValueError: Invalid argument provided to Gemini")
    assert _reason(RuntimeError("{'code': 'tool_use_failed', 'message': 'Failed to call a function'}")) == \
        "bad tool call"


# ------------------------------------------------------------------ run of 2026-10-23
def test_convention_note_names_each_convention_explicitly(tmp_path):
    """Q5 twice wrote the phase change backwards; the note now labels each phase with its convention."""
    import numpy as np
    import qi_tools as qt
    from agent_tools import reconstruct_state_from_counts
    from test_tomography import make
    STORE.clear()
    path = make(tmp_path, qt.dm(qt.pure_state_2q([1, 0, 0, 1], [0, 0, 0, -108])), n=4000)
    a = reconstruct_state_from_counts.invoke({"csv_path": path, "error_bars": False})
    phi = a["closest_maximally_entangled"]["phi_deg"]
    assert f"{phi}° with R=(H+iV)/√2 (this reconstruction), {-phi}° with R=(H-iV)/√2" in a["convention_note"]
    assert abs(phi + 108) < 2
