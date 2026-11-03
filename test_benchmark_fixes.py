"""
test_benchmark_fixes.py - errors found in the routed benchmark run of 2026-10-28 (offline):
  * T08: a table row at the CHSH crossing had '~0.75' (true purity 0.625) and '>0'  -> parameter rows
  * T29: "steerable (projective measurements) for p > 0.577" (true threshold p = 1/2) -> removed + note
  * T01: "N is half of C, consistent with the state being mixed" (pure states too)   -> removed
  * T07: "CHSH max >= 2 means no violation"                                         -> corrected to <=
  * T08/T13: "becomes classical" / "no quantum correlations beyond classical"       -> removed
  * T28: "proving Bell nonlocality" for a reconstructed state                         -> "indicating"
  * T30: low D/A visibility called "reduced coherence" - it is mostly the HH-VV phase -> tool reports it
Run:  pytest -q test_benchmark_fixes.py
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
from agent_tools import (DESC, STORE, TOMO, create_pure_state, create_state, evolve_state, plot_fringes,
                         plot_measures_vs_noise, plot_sweep, reconstruct_state_from_counts)


@pytest.fixture(autouse=True)
def clean():
    from agent_tools import CURRENT_QUESTION
    STORE.clear(); DESC.clear(); TOMO.clear()
    CURRENT_QUESTION["text"] = ""                 # a question left by another test names Bell states
    yield
    STORE.clear(); DESC.clear(); TOMO.clear()


def tm(name, result):
    return ToolMessage(content=json.dumps(result), tool_call_id=name, name=name)


T08 = """| Noise p | Concurrence | Negativity | Purity | CHSH max |
|---------|-------------|------------|--------|----------|
| 0.0     | 1.0000      | 0.5000     | 1.0000 | 2.8284   |
| 0.159   | >0          | >0         | ~0.75  | 2.0000   |
| 0.423   | 0.0000      | 0.0000     | ~0.35  | <2       |
| 1.0     | 0.0000      | 0.0000     | 0.2500 | 0.0000   |"""


def test_noise_plot_reports_all_measures_exactly_at_the_crossings():
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"})["features"]
    key = f["at_key_points"]
    assert key["p=0.159 (CHSH reaches 2)"] == {"concurrence": 0.5607, "negativity": 0.2803,
                                               "purity": 0.625, "chsh_max": 2.0}
    assert key["p=0.423 (entanglement vanishes)"]["purity"] == 0.3333        # Werner weight 1/3


def test_approximate_or_wrong_cells_in_a_parameter_table_are_flagged():
    turn = [tm("plot_measures_vs_noise", plot_measures_vs_noise.invoke({"start_kind": "phi+",
                                                                        "channel": "depolarizing"}))]
    bad = graph.wrong_parameter_rows(T08, turn)
    assert any("'~0.75'" in b and "purity 0.625" in b for b in bad)
    assert any("'>0'" in b for b in bad) and any("'<2'" in b for b in bad)
    assert not any("2.0000" in b for b in bad)                # CHSH = 2 at the crossing is right
    good = T08.replace(">0          | >0         | ~0.75 ", "0.5607      | 0.2803     | 0.6250").replace(
        "0.0000      | 0.0000     | ~0.35  | <2      ", "0.0000      | 0.0000     | 0.3333 | 0.9428  ")
    assert graph.wrong_parameter_rows(good, turn) == []
    notes = dict(graph.completion_and_claim_notes("Plot ... depolarizing", turn, T08))
    assert "parameter rows" in notes


def test_time_tables_with_key_times_are_accepted():
    create_state.invoke({"kind": "phi+"})
    turn = [tm("evolve_state", evolve_state.invoke({"handle": "rho_1", "process": "dephasing",
                                                    "gamma": 1, "t_max": 2}))]
    answer = """| State description | Time | Concurrence | Purity | CHSH max |
|---|---|---|---|---|
| phi+ after local dephasing (γ=1) on both | 0 | 1.0000 | 1.0000 | 2.8284 |
| phi+ after local dephasing (γ=1) on both | 0.3466 | 0.5000 | 0.6250 | 2.2361 |
| phi+ after local dephasing (γ=1) on both | 2 | 0.0183 | 0.5002 | 2.0003 |"""
    assert graph.wrong_parameter_rows(answer, turn) == []
    first_cell = """| State evolution | Concurrence |
|---|---|
| phi+ after local dephasing (γ=1) t=0 | 1.0000 |
| phi+ after local dephasing (γ=1) t=0.3466 | 0.5000 |"""
    assert graph.wrong_parameter_rows(first_cell, turn) == []     # the row is t, not γ = 1
    assert graph.wrong_parameter_rows(first_cell.replace("0.5000", "0.4500"), turn)


def test_sweep_crossings_carry_every_measure_and_the_steering_caveat():
    r = plot_sweep.invoke({"sweep": "werner_p", "quantities": ["concurrence", "chsh_max", "steering_f3"]})
    m = r["results"]["Werner on psi-"]["measures"]
    chsh = m["chsh_max"]["crossings"][0]
    assert chsh["at"] == 0.7071 and chsh["values_there"]["chsh_max"] == 2.0
    assert chsh["values_there"]["concurrence"] == 0.5607
    f3 = m["steering_f3"]["crossings"][0]
    assert f3["at"] == 0.5774 and f3["values_there"]["steering_f3"] == 1.0    # not 1.0001 (rounded p)
    assert all("_exact" not in c for q in m.values() for c in q["crossings"])
    assert "p > 1/2" in r["steering_note"] and "sufficient" in m["steering_f3"]["bound"]
    turn = [tm("plot_sweep", r)]
    answer = ("| Measure | Crossing p |\n|---|---|\n| Steering F3 | 0.5774 |\n\n**What it means:**\n"
              "- It becomes steerable (with projective measurements) for p > ~0.577.\n"
              "- It violates CHSH for p > 1/√2 ≈ 0.707.")
    fixed, reasons = graph.remove_false_claims(answer, turn)
    assert "projective measurements) for p > ~0.577" not in fixed and "| Steering F3 | 0.5774 |" in fixed
    assert any("p > 1/2" in x for x in reasons)
    notes = graph.required_notes(turn, fixed)
    assert notes and "Wiseman" in notes[0]
    assert graph.required_notes(turn, fixed + "\nWerner states are steerable for p > 1/2.") == []
    # a statement about the F3 test itself is right and is kept
    kept, why = graph.remove_false_claims("- F3 > 1 shows that it is steerable for p > 0.5774.", turn)
    assert why == [] and "0.5774" in kept


def test_wrong_inferences_are_removed_or_corrected():
    fixed, why = graph.remove_false_claims(
        "- Negativity is half the concurrence, consistent with the Werner state being mixed.\n- Fine.", [])
    assert "half the concurrence" not in fixed and "- Fine." in fixed and "mixedness" in why[0]
    fixed, _ = graph.remove_false_claims("- N = C/2 for pure states; for mixed states they can differ.", [])
    assert "N = C/2" in fixed                                  # a correct statement stays
    fixed, why = graph.remove_false_claims("*CHSH max ≥ 2 means no violation at that point.*", [])
    assert "≤ 2 means no violation" in fixed and why
    fixed, why = graph.remove_false_claims("- The state becomes classical and separable beyond p = 0.423.\n"
                                           "- It beats the classical limit 2/3.", [])
    assert "becomes classical" not in fixed and "classical limit" in fixed
    fixed, _ = graph.remove_false_claims("- After that it is no longer useful for quantum correlations "
                                         "beyond classical.", [])
    assert "beyond classical" not in fixed


def test_proving_nonlocality_is_softened_only_for_states_from_data():
    text = "- The CHSH inequality is violated, proving Bell nonlocality beyond classical explanations."
    fixed, why = graph.remove_false_claims(text, [])
    assert fixed == text and why == []                        # a theoretical state: CHSH > 2 proves it
    data = [tm("reconstruct_state_from_counts", {"handle": "rho_1"})]
    fixed, why = graph.remove_false_claims(text, data)
    assert "indicating Bell nonlocality" in fixed and "settings measured directly" in why[0]


def test_fringes_separate_the_phase_from_lost_coherence():
    reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})
    r = plot_fringes.invoke({"handle": "rho_1"})
    assert [f["visibility"] for f in r["fringes"]] == [0.9692, 0.2287, 0.9747, 0.3513]
    assert r["hh_vv_coherence"] == pytest.approx(0.7215, abs=1e-4)
    assert r["hh_vv_phase_magnitude_deg"] == pytest.approx(108.28, abs=0.01)
    assert r["visibility_with_phase_removed"]["D"] == pytest.approx(0.7183, abs=1e-4)
    assert set(r["visibility_with_phase_removed"]) == {"D", "A"}       # H/V are not affected by the phase
    assert "PHASE" in r["meaning"]
    create_state.invoke({"kind": "phi+"})
    r = plot_fringes.invoke({"handle": "rho_2"})
    assert "meaning" not in r and r["hh_vv_coherence"] == 1.0  # no phase: nothing to explain


def test_routing_can_be_switched_off(monkeypatch):
    from test_routing import FakeRoutedLLM
    llm = FakeRoutedLLM([], [AIMessage(content="Full answer.")])
    monkeypatch.setenv("ROUTE_TOOLS", "0")
    out = graph.make_agent_node(llm)({"messages": [HumanMessage("Create phi+ and give its concurrence.")]})
    assert llm.subsets == [] and out["messages"][0].content == "Full answer."


def test_provider_token_counts_are_recorded():
    import llm
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    reply = AIMessage(content="ok", usage_metadata={"input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290})
    before = (llm.USAGE["input_tokens"], llm.USAGE["output_tokens"])
    chain = llm.chain_providers([("Fake", GenericFakeChatModel(messages=iter([reply])))])
    chain.invoke([HumanMessage("hi")])
    assert llm.USAGE["input_tokens"] - before[0] == 1234 and llm.USAGE["output_tokens"] - before[1] == 56


def test_concurrence_negativity_fact_lists_all_three_relations():
    import routing
    core = routing.system_prompt([])
    assert "both vanish together" in core and "says nothing about mixedness" in core


# ------------------------------------------------------------------ free-model run of 2026-10-28 (Groq)
def test_non_breaking_hyphens_no_longer_hide_wrong_claims():
    create_state.invoke({"kind": "phi+"})
    tool = tm("evolve_state", evolve_state.invoke({"handle": "rho_1", "process": "depolarizing",
                                                   "gamma": 1, "t_max": 1}))
    answer = ("CHSH reaches 2 at t = 0.1733.\n\n**What it means**\n"
              "- Between these times the state remains entangled but is already Bell‑local.\n- Fine.")
    made = ToolMessage(content="Created rho_1 (phi+).", tool_call_id="c", name="create_state")
    state = {"messages": [HumanMessage("Evolve phi+ ..."),
                          AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"},
                                                             "id": "c"}]), made,
                          AIMessage(content="", tool_calls=[{"name": "evolve_state", "args": {"handle": "rho_1"},
                                                             "id": "evolve_state"}]), tool,
                          AIMessage(content=answer, id="a1")]}
    out = graph.answer_check(state)["messages"][0].content
    assert "Bell-local" not in out and "Bell‑local" not in out and "- Fine." in out
    assert "local hidden-variable model" in out                      # the footnote says why


def test_correlation_and_channel_claims():
    fixed, why = graph.remove_false_claims("- Mutual information = 2 bits, discord = 1 bit → all correlations "
                                           "are quantum; no classical part.\n- ok", [])
    assert "all correlations" not in fixed and "classical correlation 1" in why[0]
    fixed, why = graph.remove_false_claims("The bit-flip channel on both qubits turns phi+ into a mixture that "
                                           "is locally equivalent to a depolarising channel.", [])
    assert fixed.strip() == "" and "rank 2" in why[0]


def test_linear_claim_is_checked_against_the_samples():
    turn = [tm("plot_measures_vs_noise", plot_measures_vs_noise.invoke({"start_kind": "phi+",
                                                                        "channel": "depolarizing"}))]
    fixed, why = graph.remove_false_claims("- Purity drops linearly from 1 to 0.25.\n- ok", turn)
    assert "linearly" not in fixed and "0.742" in why[0]
    create_state.invoke({"kind": "phi+"})                  # phi+ under dephasing: purity is not linear in t
    turn = [tm("plot_measures_vs_noise", plot_measures_vs_noise.invoke({"start_kind": "phi+",
                                                                        "channel": "bit_flip"}))]
    fixed, why = graph.remove_false_claims("- The concurrence falls linearly.", turn)
    assert why                                                       # (1 - 2p)^2 is not linear


def test_low_error_rate_with_a_high_value_is_removed():
    fixed, why = graph.remove_false_claims("- The raw state has a very low phase-error rate (≈ 39 %) that "
                                           "prevents key extraction.\n- QBER is low (≈ 1.6 %).", [])
    assert "39" not in fixed and "1.6" in fixed and "11 %" in why[0]


def test_compensation_is_three_plates_and_not_a_perfect_phi_plus():
    from agent_tools import compensate_phase
    reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})
    turn = [tm("compensate_phase", compensate_phase.invoke({"handle": "rho_1"}))]
    answer = ("The reconstructed state rho_1 is turned into a perfect φ⁺ by a single half-wave plate on photon B.\n"
              "- The phase-compensated state is the closest maximally entangled state to the original.\n- ok")
    fixed, why = graph.remove_false_claims(answer, turn)
    assert "turned closer to φ⁺ by a QWP(45°)–HWP–QWP(45°) phase shifter on photon B" in fixed
    assert "closest maximally" not in fixed and "- ok" in fixed and any("0.8527" in w for w in why)


def test_quoted_purity_must_come_from_a_tool():
    from agent_tools import teleportation_fidelity, compute_measures
    create_state.invoke({"kind": "werner", "p": 0.5, "bell": "phi+"})
    turn = [tm("teleportation_fidelity", teleportation_fidelity.invoke({"handle": "rho_1"}))]
    answer = "- The state is still mixed (purity 0.5) but useful for teleportation."
    assert graph.quoted_values_not_computed(answer, turn) == ["purity 0.5 (no tool in this conversation "
                                                              "computed a purity)"]
    turn.append(tm("compute_measures", compute_measures.invoke({"handle": "rho_1"})))
    assert "0.4375" in graph.quoted_values_not_computed(answer, turn)[0]
    assert graph.quoted_values_not_computed("- It is mixed (purity 0.4375).", turn) == []
    assert graph.quoted_values_not_computed("- A Bell state has concurrence 1.", []) == []   # concept answer
    notes = dict(graph.completion_and_claim_notes("teleportation?", turn, answer))
    assert "quoted value" in notes


def test_display_repairs_escaped_kets_latex_identity_and_split_bold():
    from display import to_unicode
    assert to_unicode(r"cos(0.3)\|HH> + sin(0.3)\|VV>") == "cos(0.3)∣HH⟩ + sin(0.3)∣VV⟩"
    assert to_unicode(r"(1-p) mathbb I/4 or (1-p)\mathbb{1}/4") == "(1-p) I/4 or (1-p)I/4"
    assert to_unicode("**Concurrence = ** **0.6000**") == "**Concurrence = 0.6000**"


# ------------------------------------------------------------------ free-model rerun of 2026-10-29
def test_bit_flip_mechanism_is_reported_exactly():
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["features"]
    assert "turns phi+ into psi+" in f["mechanism"] and "(1-2p)²" in f["mechanism"]
    assert f["samples"]["0.2"]["concurrence"] == 0.36                  # (1 - 2·0.2)²
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "phase_flip", "qubits": "A"})["features"]
    assert "into phi-" in f["mechanism"] and "|1-2p|" in f["mechanism"]
    assert "mechanism" not in plot_measures_vs_noise.invoke({"start_kind": "phi+",
                                                             "channel": "depolarizing"})["features"]


def test_shape_and_locality_claims_after_a_bit_flip_plot():
    turn = [tm("plot_measures_vs_noise", plot_measures_vs_noise.invoke({"start_kind": "phi+",
                                                                        "channel": "bit_flip"}))]
    answer = ("The concurrence, negativity and CHSH value oscillate with the flip probability.\n"
              "- CHSH remains above the local bound for all p; the minimum value 2.0000 occurs at p = 0.5.\n"
              "- Purity is minimal at p = 0.5.")
    fixed, why = graph.remove_false_claims(answer, turn)
    assert "CHSH value fall and then rise again with the flip probability" in fixed        # kept, corrected
    assert "for all p" not in fixed and "Purity is minimal" in fixed
    assert any("turning point" in w for w in why) and any("not above 2 everywhere" in w for w in why)
    fixed, why = graph.remove_false_claims("- Beyond that the correlations are compatible with a local "
                                           "hidden-variable model.\n- ok", [])
    assert "compatible" not in fixed and "local hidden-variable model exists" in why[0]


def test_a_key_fraction_has_no_classical_limit():
    fixed, why = graph.remove_false_claims("- The key fraction is ≈ 30 %, well above the classical limit.\n"
                                           "- F = 0.75 beats the classical limit 2/3.", [])
    assert "30 %" not in fixed and "2/3" in fixed and "no classical limit" in why[0]



# ------------------------------------------------------------------ Hugging Face model comparison 2026-10-29
def test_empty_final_answer_is_asked_again_once():
    q = HumanMessage("Create the mixture 0.8|phi+><phi+| + 0.2|psi+><psi+| and give its concurrence.")
    out = graph.answer_check({"messages": [q, AIMessage(content="", id="a")]})["messages"][0]
    assert out.content.startswith("VERIFIER: [empty]")
    again = graph.answer_check({"messages": [q, AIMessage(content=""), out, AIMessage(content="", id="b")]})
    assert not again or not str(again["messages"][0].content).startswith("VERIFIER: [empty]")


def test_replies_to_the_answer_check_are_stripped():
    t = ("You're right — I computed those sums myself. Let me rewrite using only the tool's numbers.\n\n"
         "**A key can be made after compensation.**")
    assert graph.strip_meta(t) == "**A key can be made after compensation.**"
    t = "The state has been reconstructed and certified. Here are the results.\n\n**It is entangled.**"
    assert graph.strip_meta(t) == "**It is entangled.**"
    t = "The state has been reconstructed. Let me report the phase.\n\nThe phase is 108.3°."
    assert graph.strip_meta(t) == "The phase is 108.3°."
    keep = "The Werner state with p = 0.5 has concurrence 0.25.\n\n| a | b |"
    assert graph.strip_meta(keep) == keep


def test_more_wrong_claims_are_caught():
    fixed, why = graph.remove_false_claims("- Damping both qubits destroys all quantum correlations (C = 0, N = 0).\n"
                                           "- Above p = 0.423 the state is separable and all quantum correlations "
                                           "disappear.\n- ok", [])
    assert fixed.strip() == "- ok" and "discord" in why[0]
    turn = [tm("plot_measures_vs_noise", plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"}))]
    fixed, why = graph.remove_false_claims("- CHSH never falls below 2, meaning no violation of the Bell inequality "
                                           "occurs for any p.\n- ok", turn)
    assert "for any p" not in fixed and "2.3745" in why[0]
    data = [tm("reconstruct_state_from_counts", {"handle": "rho_1"})]
    fixed, _ = graph.remove_false_claims("- Bell nonlocality is violated at 17.3σ, ruling out local hidden-variable "
                                         "theories.", data)
    assert "indicating a violation of the bound for local hidden-variable models" in fixed


def test_chsh_just_above_two_is_above_two():
    create_state.invoke({"kind": "phi+"})
    turn = [tm("evolve_state", evolve_state.invoke({"handle": "rho_1", "process": "dephasing", "gamma": 1,
                                                    "t_max": 2}))]
    text = "- CHSH remains above 2 for all times up to t = 2."
    assert graph.remove_false_claims(text, turn)[0] == text           # 2.0003 at t = 2 is above 2


def test_certification_summaries_are_checked():
    from agent_tools import certify_correlations
    reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})
    turn = [tm("certify_correlations", certify_correlations.invoke({"handle": "rho_1"}))]
    fixed, why = graph.remove_false_claims(
        "The state is entangled, steerable and Bell nonlocal, with all violations exceeding 35σ.\n"
        "- All three are shown (≥ 5σ), with the most significant being Bell nonlocality.\n"
        "- All tests exceed 17σ.", turn)
    assert "35σ" not in fixed and "most significant" not in fixed and "exceed 17σ" in fixed
    assert any("17.2σ" in w for w in why) and any("entangled (concurrence > 0)" in w for w in why)
    _, why2 = graph.remove_false_claims("- It is Bell-nonlocal (the strongest property in the hierarchy).", turn)
    assert why2 == []                                      # strongest kind of correlation: that is right


def test_only_one_entangled_state_claim_is_checked():
    from agent_tools import compute_measures
    msgs = []
    for p in (0.3, 0.5, 0.8):
        msgs.append(tm("create_state", create_state.invoke({"kind": "werner", "p": p})))
    for h in ("rho_1", "rho_2", "rho_3"):
        msgs.append(tm("compute_measures", compute_measures.invoke({"handle": h})))
    fixed, why = graph.remove_false_claims("**Only p = 0.8 is entangled, and only p = 0.8 violates CHSH.**\n- ok", msgs)
    assert "Only p = 0.8" not in fixed and "0.5, 0.8" in why[0]
    fixed, why = graph.remove_false_claims("- Only p = 0.8 violates CHSH.", msgs)
    assert why == []                                                   # that one is right


def test_quoted_values_ignore_reference_states_and_ratios():
    from agent_tools import compute_measures
    reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})
    turn = [tm("compute_measures", compute_measures.invoke({"handle": "rho_1"}))]
    ok = ("- Concurrence = 1, negativity = 0.5 (its maximal possible value) and log-negativity = 1 bit.\n"
          "- For pure states, concurrence = 2×negativity.\n- Concurrence is 0 for separable states.\n"
          "- A maximally entangled state would have concurrence 1, negativity 0.5.")
    assert graph.quoted_values_not_computed(ok, turn) == []
    assert graph.quoted_values_not_computed("- It is mixed (purity 0.5).", turn)


def test_transposed_parameter_table_is_read_by_column():
    turn = [tm("plot_measures_vs_noise", plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"}))]
    table = ("| Measure | p = 0.0000 | p = 0.5000 | p = 1.0000 |\n|---|---|---|---|\n"
             "| Concurrence | 1.0000 | 0.0000 | 1.0000 |\n| CHSH max | 2.8284 | 2.0000 | 2.8284 |")
    assert graph.wrong_parameter_rows(table, turn) == []
    assert graph.wrong_parameter_rows(table.replace("| 2.0000 |", "| 2.1000 |"), turn)


def test_invented_counts_are_numbers_no_tool_produced():
    from agent_tools import plot_tomography_fit, get_tomography_report
    reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})
    turn = [tm("get_tomography_report", get_tomography_report.invoke({"handle": "rho_1"})),
            tm("plot_tomography_fit", plot_tomography_fit.invoke({"handle": "rho_1"}))]
    table = ("| Setting | Deviation | Measured | Fit |\n|---|---|---|---|\n| AD | +3.1 | 605 | 534 |\n"
             "| HA | -2.3 | 398 | 444 |\n| p-value | 1.1 × 10⁻⁵ | | |")
    assert graph.ungrounded_counts(table, turn) == ["398", "444"]


def test_bit_flip_mechanism_says_what_happens_at_p_equal_one():
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["features"]
    assert "At p = 1 both photons are flipped, which gives phi+ back exactly (q = 0), not psi+" in f["mechanism"]



# ------------------------------------------------------------------ DeepSeek rerun and gpt-oss T09/T24, 2026-10-29
def test_simulation_must_use_the_settings_the_user_gave():
    q = ("Simulate a tomography of phi+ with a pair rate of 1e6 per second, detection efficiency 0.2 on both "
         "sides, 500 dark counts per second, a 20 ns coincidence window and 0.1 s per setting.")
    good = {"handle": "rho_1", "pair_rate": 1e6, "eta_a": 0.2, "eta_b": 0.2, "dark_a": 500, "dark_b": 500,
            "window_ns": 20, "time_per_setting_s": 0.1}
    def turn(args):
        return [AIMessage(content="", tool_calls=[{"name": "simulate_tomography", "args": args, "id": "s"}]),
                ToolMessage(content="{}", tool_call_id="s", name="simulate_tomography")]
    assert graph.simulation_settings_differ(q, turn(good)) == []
    bad = graph.simulation_settings_differ(q, turn({**good, "time_per_setting_s": 1.0}))
    assert bad == ["time_per_setting_s=1.0 instead of 0.1"]
    assert "window_ns=(default) instead of 20" in graph.simulation_settings_differ(
        q, turn({k: v for k, v in good.items() if k != "window_ns"}))
    notes = dict(graph.completion_and_claim_notes(q, turn({**good, "time_per_setting_s": 1.0}), "answer"))
    assert "simulation settings" in notes


def test_final_state_with_concurrence_left_is_still_entangled():
    create_state.invoke({"kind": "phi+"})
    turn = [tm("evolve_state", evolve_state.invoke({"handle": "rho_1", "process": "dephasing", "gamma": 1,
                                                    "t_max": 2}))]
    fixed, why = graph.remove_false_claims("- The final state is fully dephased phi+ with purity 0.5002, which is "
                                           "classically correlated but not entangled.\n- ok", turn)
    assert "final state" not in fixed and "0.0183" in why[0]


def test_more_replies_to_the_checker_are_stripped():
    assert graph.strip_meta("I've reconstructed the state and certified its correlations.\n\nIt is entangled.") == \
        "It is entangled."
    assert graph.strip_meta("The state was reconstructed and its fringes plotted.\n\n**H/V high.**") == "**H/V high.**"
    assert graph.strip_meta("## Answer\n\n**Yes.**") == "**Yes.**"



def test_mixed_but_pure_is_removed():                    # app check of 2026-10-29, Werner p = 0.8
    fixed, why = graph.remove_false_claims("- The state is mixed (purity < 1) but still significantly pure.\n"
                                           "- It is entangled.", [])
    assert "significantly pure" not in fixed and "- It is entangled." in fixed and "purity 1" in why[0]
    text = "- The state is mixed (purity 0.73), but highly entangled."
    assert graph.remove_false_claims(text, [])[0] == text



def test_confirming_nonlocality_is_softened_for_data():           # app check 2026-10-29, uploaded counts
    data = [tm("reconstruct_state_from_counts", {"handle": "rho_5"})]
    fixed, why = graph.remove_false_claims("- It violates the CHSH inequality, confirming nonlocal correlations.", data)
    assert "indicating nonlocal correlations" in fixed and why
    text = "- It violates the CHSH inequality, confirming nonlocal correlations."
    assert graph.remove_false_claims(text, [])[0] == text          # a theoretical state: fine


# ------------------------------------------------------------------ app run with DeepSeek, 2026-10-30
def test_thinking_aloud_is_removed():
    t = ("- **Bell-nonlocal**: CHSH max = 2.263 > 2√2? No — 2√2 ≈ 2.828, so 2.263 < 2.828. Wait, let me check: "
         "CHSH max = 2.263 > 2 (local bound), so it **violates** the CHSH inequality.")
    assert graph.strip_self_talk(t) == ("- **Bell-nonlocal**: CHSH max = 2.263 > 2 (local bound), so it **violates** "
                                        "the CHSH inequality.")
    for keep in ("Is the fit good? No — the χ²/dof is 2.94.", "- Wait times are short.", "- CHSH max = 2.46 > 2."):
        assert graph.strip_self_talk(keep) == keep


def test_reversed_hierarchy_is_corrected():
    fixed, why = graph.remove_false_claims("- This ordering (entangled ⊂ steerable ⊂ Bell-nonlocal) is expected.", [])
    assert "(Bell-nonlocal ⊂ steerable ⊂ entangled)" in fixed and "other way" in why[0]


def test_reduced_states_may_be_maximally_mixed():
    from agent_tools import compute_measures
    create_state.invoke({"kind": "phi+"})
    turn = [tm("compute_measures", compute_measures.invoke({"handle": "rho_1"}))]
    text = "- Each photon on its own is maximally mixed (subsystem entropy 1 bit)."
    assert graph.remove_false_claims(text, turn)[0] == text
    assert graph.remove_false_claims("- The state is maximally mixed.", turn)[1]       # the whole state is not


def test_chsh_window_is_not_called_local():
    # DeepSeek, Werner sweep explanation (2026-10-30): the window 1/3 < p < 1/√2 only satisfies CHSH
    text = ('- **Hierarchy in action:** The two thresholds (0.3333 and 0.7071) are the known Werner-state values, '
            'and the gap between them is the region of "entangled but local" states.')
    fixed, why = graph.remove_false_claims(text, [])
    assert '"entangled but CHSH-satisfying" states' in fixed and "other Bell inequalities" in why[0]
    ok = "- Werner's model: for p ≤ 1/2 the states are entangled but local (projective measurements)."
    assert graph.remove_false_claims(ok, [])[0] == ok                  # true, and the line is not about CHSH
