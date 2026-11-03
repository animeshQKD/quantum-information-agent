"""
test_photons_global.py - one-photon states, density matrices as numbers, global (4x4 Kraus) noise, the
user's own Kraus operators and the new Lindblad processes (2026-10-31). Every expected value is analytic.
Run:  pytest -q test_photons_global.py
"""
import math

import numpy as np
import pytest

import agent_tools as at
import display
import qi_tools as qt
import session
import workbench as wb


@pytest.fixture(autouse=True)
def fresh():
    at.STORE.clear(); at.DESC.clear(); at.TOMO.clear()
    at.CURRENT_QUESTION["text"] = ""          # a question left by another test file must not apply here
    yield
    at.STORE.clear(); at.DESC.clear(); at.TOMO.clear()
    for sid in session.active_ids():
        session.drop(sid)


def handle_of(text: str) -> str:
    import re
    return re.search(r"rho_\d+", text).group(0)


# ------------------------------------------------------------------ one photon
@pytest.mark.parametrize("pol,stokes", [("H", (1, 0, 0)), ("V", (-1, 0, 0)), ("D", (0, 1, 0)), ("A", (0, -1, 0)),
                                        ("R", (0, 0, 1)), ("L", (0, 0, -1))])
def test_named_polarizations_have_the_right_stokes_vector(pol, stokes):
    h = handle_of(at.create_single_photon_state.invoke({"polarization": pol}))
    m = at.compute_measures.invoke({"handle": h})
    assert m["photons"] == 1
    assert [m["stokes"][k] for k in ("S1 (H-V)", "S2 (D-A)", "S3 (R-L)")] == pytest.approx(stokes, abs=1e-9)
    assert m["degree_of_polarization"] == pytest.approx(1) and m["purity"] == pytest.approx(1)
    assert m["entropy"] == pytest.approx(0, abs=1e-9)


def test_partly_polarized_linear_light_at_30_degrees():
    h = handle_of(at.create_single_photon_state.invoke({"azimuth_deg": 30, "dop": 0.8}))
    m = at.compute_measures.invoke({"handle": h})
    assert m["stokes"]["S1 (H-V)"] == pytest.approx(0.4, abs=1e-6)                 # 0.8 cos 60°
    assert m["stokes"]["S2 (D-A)"] == pytest.approx(0.8 * math.sqrt(3) / 2, abs=1e-6)
    assert m["purity"] == pytest.approx(0.82, abs=1e-6)                              # (1 + 0.64)/2
    assert m["entropy"] == pytest.approx(qt.binary_entropy(0.1), abs=1e-6)          # eigenvalues 0.9, 0.1
    assert m["ellipse"]["azimuth_deg"] == pytest.approx(30) and m["ellipse"]["shape"] == "linear"


def test_amplitudes_and_circular_light():
    h = handle_of(at.create_single_photon_state.invoke({"amp_H": "1", "amp_V": "i"}))
    m = at.compute_measures.invoke({"handle": h})
    assert m["stokes"]["S3 (R-L)"] == pytest.approx(1) and "(= R)" in m["polarized_part"]
    assert m["ellipse"]["azimuth_deg"] == "undefined (circular polarization)"
    assert m["ellipse"]["ellipticity_deg"] == pytest.approx(45) and m["ellipse"]["shape"] == "circular"


def test_exactly_one_way_to_give_the_photon():
    with pytest.raises(Exception, match="exactly one"):
        at.create_single_photon_state.invoke({"polarization": "H", "azimuth_deg": 10})
    with pytest.raises(Exception, match="exactly one"):
        at.create_single_photon_state.invoke({})
    with pytest.raises(Exception, match="not a physical"):
        at.create_single_photon_state.invoke({"s1": 0.9, "s2": 0.9})


def test_reduced_state_of_phi_plus_is_unpolarized_and_product_is_not_entangled():
    at.create_state.invoke({"kind": "phi+"})
    r = handle_of(at.reduced_state.invoke({"handle": "rho_1", "photon": "A"}))
    m = at.compute_measures.invoke({"handle": r})
    assert m["degree_of_polarization"] == pytest.approx(0, abs=1e-9) and m["entropy"] == pytest.approx(1)
    h = handle_of(at.create_single_photon_state.invoke({"polarization": "H"}))
    d = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    pair = handle_of(at.combine_photons.invoke({"handle_a": h, "handle_b": d}))
    mp = at.compute_measures.invoke({"handle": pair})
    assert mp["concurrence"] == 0 and mp["purity"] == pytest.approx(1)
    rows = at.show_density_matrix.invoke({"handle": pair})["matrix"]              # |H>|D> = (|HH> + |HV>)/√2
    assert rows[0][:2] == ["0.5000", "0.5000"] and rows[1][:2] == ["0.5000", "0.5000"] and rows[2] == ["0"] * 4


def test_one_photon_states_are_rejected_by_two_photon_tools_with_a_clear_message():
    h = handle_of(at.create_single_photon_state.invoke({"polarization": "H"}))
    for tool, args in ((at.certify_correlations, {"handle": h}), (at.teleportation_fidelity, {"handle": h}),
                       (at.apply_global_noise, {"handle": h, "channel": "global_depolarizing", "param": 0.1})):
        with pytest.raises(Exception, match="ONE-photon state"):
            tool.invoke(args)
    at.create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="different sizes"):
        at.compare_states.invoke({"handle_a": h, "handle_b": "rho_2"})


def test_wave_plate_noise_and_evolution_of_one_photon():
    d = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    r = at.apply_waveplate.invoke({"handle": d, "plate": "QWP", "angle_deg": 0, "qubit": "A"})
    assert r["after"]["S3 (R-L)"] == pytest.approx(1) and r["after"]["degree_of_polarization"] == pytest.approx(1)
    n = handle_of(at.apply_local_noise.invoke({"handle": d, "channel": "depolarizing", "param": 0.5}))
    assert at.compute_measures.invoke({"handle": n})["degree_of_polarization"] == pytest.approx(0.5)
    e = at.evolve_state.invoke({"handle": d, "process": "dephasing", "gamma": 1, "t_max": 2})
    assert e["features"]["at_t_max"]["degree_of_polarization"] == pytest.approx(math.exp(-2), abs=1e-4)
    assert e["features"]["at_t_max"]["purity"] == pytest.approx((1 + math.exp(-4)) / 2, abs=1e-4)
    assert e["features"]["degree_of_polarization_halves_at"] == pytest.approx(math.log(2), abs=1e-4)


def test_one_photon_tomography_from_counts():
    r = at.reconstruct_single_photon.invoke({"counts_H": 980, "counts_V": 20, "counts_D": 510, "counts_A": 490,
                                             "counts_R": 500, "counts_L": 500})
    assert r["stokes_mle"]["S1 (H-V)"].startswith("0.9600 ± ") and r["stokes_mle"]["S2 (D-A)"].startswith("0.0200")
    assert r["stokes_linear_inversion"] == {"S1 (H-V)": 0.96, "S2 (D-A)": 0.02, "S3 (R-L)": 0.0}
    assert r["degree_of_polarization"].startswith("0.9602")
    with pytest.raises(Exception, match="incomplete"):
        at.reconstruct_single_photon.invoke({"counts_H": 10, "counts_V": 10})


def test_one_photon_tomography_with_the_other_rl_convention():
    rho = qt.rho_from_stokes(0.2, 0.3, 0.5)
    P = qt.pauli_projectors_1q(-1)                                    # the lab's R is our L
    counts = {f"counts_{k}": float(1e6 * np.real(np.trace(rho @ P[k]))) for k in "HVDARL"}
    r = at.reconstruct_single_photon.invoke({**counts, "r_convention": "R=(H-iV)/sqrt2"})
    assert r["stokes_mle"]["S3 (R-L)"].startswith("0.5000")
    assert r["stokes_linear_inversion"]["S3 (R-L)"] == pytest.approx(0.5, abs=1e-4)


# ------------------------------------------------------------------ the density matrix as numbers
def test_show_density_matrix_and_the_change_of_every_element():
    at.create_state.invoke({"kind": "phi+"})
    a = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "amplitude_damping", "param": 0.3}))
    r = at.show_density_matrix.invoke({"handle": a, "compare_with": "rho_1"})
    m = r["matrix"]
    assert m[0][0] == "0.5450" and m[3][3] == "0.2450" and m[1][1] == m[2][2] == "0.1050"   # (1+γ²)/2 ...
    assert m[0][3] == "0.3500"                                                              # (1-γ)/2
    assert r["difference"][0][3] == "-0.1500" and r["difference"][3][3] == "-0.2550"
    assert "ρ[HH,VV] = -0.1500" in r["changed_elements"]
    text = display.format_density_matrix(r)
    assert text.count("| **HH** |") == 3 and "Change Δρ" in text


def test_complex_entries_are_written_a_plus_bi():
    assert at.cplx_text(0.425 - 0.1j) == "0.4250 - 0.1000i"
    assert at.cplx_text(0.3j) == "0.3000i" and at.cplx_text(1e-6) == "0" and at.cplx_text(-0.5) == "-0.5000"


def test_reconstructed_state_gets_error_bars_on_every_element():
    r = at.reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})
    m = at.show_density_matrix.invoke({"handle": r["handle"]})
    assert "element_errors_1sigma" in m and len(m["element_errors_1sigma"]["re"]) == 4
    assert "(±" in display.format_density_matrix(m)
    assert "<this handle>" not in display.format_density_matrix(m)


# ------------------------------------------------------------------ global channels
@pytest.mark.parametrize("name", list(at.GLOBAL_CHANNELS))
def test_global_channels_are_complete_and_4x4(name):
    for p in (0.0, 0.3, 1.0):
        K = at._channel_kraus(name, p, "both")
        assert all(k.shape == (4, 4) for k in K) and qt.completeness_error(K) < 1e-12


def test_global_versus_local_depolarizing_on_phi_plus():
    at.create_state.invoke({"kind": "phi+"})
    g = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}))
    loc = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.3}))
    assert at.compute_measures.invoke({"handle": g})["concurrence"] == pytest.approx(0.55)    # (3·0.7 - 1)/2
    assert at.compute_measures.invoke({"handle": loc})["concurrence"] == pytest.approx(0.235)  # (3·0.49 - 1)/2
    assert np.allclose(at.STORE[g], qt.werner_state(0.7, "phi+"))


def test_collective_dephasing_spares_psi_and_hits_phi_four_times_harder():
    at.create_state.invoke({"kind": "phi+"}); at.create_state.invoke({"kind": "psi-"})
    a = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "collective_dephasing", "param": 0.5}))
    b = handle_of(at.apply_global_noise.invoke({"handle": "rho_2", "channel": "collective_dephasing", "param": 0.5}))
    assert at.compute_measures.invoke({"handle": a})["concurrence"] == pytest.approx(0.5 ** 4)
    assert at.compute_measures.invoke({"handle": b})["concurrence"] == pytest.approx(1.0)


def test_collective_amplitude_damping_psi_minus_is_dark():
    at.create_state.invoke({"kind": "psi-"}); at.create_state.invoke({"kind": "psi+"})
    a = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "collective_amplitude_damping",
                                                "param": 0.9}))
    assert np.allclose(at.STORE[a], at.STORE["rho_1"])
    b = handle_of(at.apply_global_noise.invoke({"handle": "rho_2", "channel": "collective_amplitude_damping",
                                                "param": 0.9}))
    assert at.compute_measures.invoke({"handle": b})["concurrence"] == pytest.approx(0.01, abs=1e-6)  # (1-p)^2


def test_correlated_flips_leave_bell_states_unchanged_but_not_product_states():
    at.create_state.invoke({"kind": "phi-"})
    for ch in ("correlated_bit_flip", "correlated_phase_flip", "correlated_bit_phase_flip"):
        h = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": ch, "param": 0.4}))
        assert np.allclose(at.STORE[h], at.STORE["rho_1"])
    at.create_pure_state.invoke({"amp_HH": "1"})
    h = handle_of(at.apply_global_noise.invoke({"handle": "rho_5", "channel": "correlated_bit_flip", "param": 0.4}))
    assert at.STORE[h][3, 3].real == pytest.approx(0.4)


def test_local_and_global_tools_point_to_each_other():
    at.create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="apply_global_noise"):
        at.apply_local_noise.invoke({"handle": "rho_1", "channel": "collective_dephasing", "param": 0.2})
    with pytest.raises(Exception, match="apply_local_noise"):
        at.apply_global_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.2})


def test_kraus_operators_tool():
    r = at.get_kraus_operators.invoke({"channel": "collective_dephasing", "param": 0.3})
    assert r["basis"] == at.BASIS_2 and r["trace_preserving"] and r["number_of_operators"] == 3
    r = at.get_kraus_operators.invoke({"channel": "amplitude_damping", "param": 0.36, "qubits": "both"})
    assert r["kraus_operators"]["K0"] == [["1.0000", "0"], ["0", "0.8000"]] and "4 operators" in r["scope"]
    assert "Kraus operators" in display.format_kraus(r)


def test_users_own_kraus_operators():
    at.create_state.invoke({"kind": "phi+"})
    h = handle_of(at.apply_kraus_operators.invoke({"handle": "rho_1", "qubits": "B",
                                                   "kraus": "[[[1, 0], [0, sqrt(0.6)]], [[0, sqrt(0.4)], [0, 0]]]"}))
    assert at.compute_measures.invoke({"handle": h})["concurrence"] == pytest.approx(math.sqrt(0.6))  # AD on B
    ref = qt.apply_kraus(at.STORE["rho_1"], at._channel_kraus("amplitude_damping", 0.4, "B"))
    assert np.allclose(at.STORE[h], ref)
    with pytest.raises(Exception, match="Not a valid"):
        at.apply_kraus_operators.invoke({"handle": "rho_1", "kraus": [[[1, 0], [0, 1]], [[0, 1], [0, 0]]]})
    with pytest.raises(Exception, match="2x2 or 4x4"):
        at.apply_kraus_operators.invoke({"handle": "rho_1", "kraus": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]})


# ------------------------------------------------------------------ time evolution
@pytest.mark.parametrize("process", [p for p in at._TWIN])
def test_every_lindblad_process_equals_its_kraus_channel(process):
    at.create_pure_state.invoke({"amp_HH": "cos(0.4)", "amp_HV": "0.3", "amp_VV": "sin(0.4)"})
    final = at.evolve_state.invoke({"handle": "rho_1", "process": process, "gamma": 1.3, "t_max": 0.7})["final_state"]
    tool_name, channel, _, f = at._TWIN[process]
    tool = at.apply_local_noise if tool_name == "apply_local_noise" else at.apply_global_noise
    twin = handle_of(tool.invoke({"handle": "rho_1", "channel": channel, "param": float(f(0.91))}))
    assert np.abs(at.STORE[final] - at.STORE[twin]).max() < 1e-9


def test_collective_decay_psi_minus_survives_psi_plus_decays():
    at.create_state.invoke({"kind": "psi-"}); at.create_state.invoke({"kind": "psi+"})
    a = at.evolve_state.invoke({"handle": "rho_1", "process": "collective_decay", "gamma": 1, "t_max": 2})
    b = at.evolve_state.invoke({"handle": "rho_2", "process": "collective_decay", "gamma": 1, "t_max": 2})
    assert a["features"]["at_t_max"]["concurrence"] == 1.0
    assert b["features"]["at_t_max"]["concurrence"] == pytest.approx(math.exp(-4), abs=1e-4)


def test_asymptotic_decay_is_not_called_sudden_death():
    at.create_state.invoke({"kind": "phi+"})
    f = at.evolve_state.invoke({"handle": "rho_1", "process": "collective_dephasing", "gamma": 1,
                                "t_max": 10})["features"]
    assert "never" in f["entanglement_vanishes_at"] and "stays above 2" in f["chsh_drops_to_2_at"]
    s = at.plot_sweep.invoke({"sweep": "noise", "handles": ["rho_1"], "channel": "collective_dephasing",
                              "quantities": ["concurrence", "chsh_max", "fully_entangled_fraction"]})
    assert all(m["crossings"] == "none in this range" for m in s["results"]["rho_1"]["measures"].values())
    f = at.plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "collective_dephasing"})["features"]
    assert f["entanglement_vanishes_at"].startswith("only at maximum noise")


def test_real_sudden_death_and_touch_points_are_kept():
    at.create_pure_state.invoke({"amp_HH": "1", "amp_VV": "2"})
    f = at.evolve_state.invoke({"handle": "rho_1", "process": "amplitude_decay", "gamma": 1, "t_max": 3})["features"]
    assert f["entanglement_vanishes_at"] == pytest.approx(math.log(2), abs=1e-4)
    at.create_state.invoke({"kind": "phi+"})
    s = at.plot_sweep.invoke({"sweep": "noise", "handles": ["rho_3"], "channel": "bit_flip",
                              "quantities": ["concurrence", "chsh_max"]})["results"]["rho_3"]["measures"]
    for q in ("concurrence", "chsh_max"):
        assert [c["at"] for c in s[q]["crossings"]] == [0.5] and "touches" in s[q]["crossings"][0]["direction"]
    s = at.plot_sweep.invoke({"sweep": "noise", "handles": ["rho_3"], "channel": "depolarizing",
                              "quantities": ["concurrence"]})["results"]["rho_3"]["measures"]
    assert s["concurrence"]["crossings"][0]["at"] == pytest.approx(1 - 1 / math.sqrt(3), abs=1e-4)


def test_matrix_elements_plot():
    at.create_state.invoke({"kind": "phi+"})
    r = at.evolve_state.invoke({"handle": "rho_1", "process": "collective_dephasing", "gamma": 1, "t_max": 2,
                                "plot_elements": True})
    import os
    assert os.path.isfile(r["elements_figure"]) and os.path.isfile(r["figure"])


def test_conditional_entropy_of_werner():
    at.create_state.invoke({"kind": "werner", "p": 0.8})
    m = at.compute_measures.invoke({"handle": "rho_1"})
    S = -(0.85 * math.log2(0.85) + 3 * 0.05 * math.log2(0.05))
    assert m["entropy"] == pytest.approx(S, abs=1e-6) and m["entropy_A"] == pytest.approx(1)
    assert m["conditional_entropy_A_given_B"] == pytest.approx(S - 1, abs=1e-6)


# ------------------------------------------------------------------ the Workbench
SID = "wb_new"


def test_workbench_one_photon_noise_with_matrices_and_kraus():
    md, _, _ = wb.create(SID, "photon", photon_how="named", polarization="D")
    assert "| **H** | 0.5000 | 0.5000 |" in md and "one photon" in md
    wb.create(SID, "bell", kind="phi+")
    md, _, summ = wb.noise(SID, "rho_2", "collective_dephasing", 0.5, "both", True)
    assert "| Concurrence C | 1.0000 | 0.0625 |" in md and "Change Δρ" in md and "| K0 |" in md
    assert "on the pair (global)" in summ
    md, fig, _ = wb.noise_sweep(SID, "rho_2", "collective_dephasing", "both")
    assert fig and "none in this range" in md


def test_workbench_reduced_product_evolution_and_photon_tomography():
    wb.create(SID, "bell", kind="phi+")
    md, _, _ = wb.create(SID, "reduced", pair="rho_1", which="B")
    assert "| Degree of polarization | 0.0000 |" in md
    wb.create(SID, "photon", photon_how="ellipse", azimuth=0, ellipticity=45)
    md, _, _ = wb.create(SID, "product", photon_a="rho_2", photon_b="rho_3")
    assert "product rho_2 ⊗ rho_3" in md
    md, figs, _ = wb.evolve(SID, "rho_1", "collective_decay", 1, 2, "both", 0, True)
    assert isinstance(figs, list) and len(figs) == 2 and "same state as apply_global_noise" in md
    md, _, _ = wb.photon_tomography(SID, "980", "20", "510", "490", "500", "500")
    assert "0.9600 ± " in md and "| **H** |" in md
    with pytest.raises(wb.WorkbenchError, match="at least four"):
        wb.photon_tomography(SID, "1", "2", "", "", "", "")


def test_workbench_density_matrix_everywhere():
    wb.create(SID, "bell", kind="phi+")
    for md, _, _ in (wb.waveplate(SID, "rho_1", "HWP", 45, "B"), wb.analyze(SID, "rho_1", "measures"),
                     wb.analyze(SID, "rho_1", "matrix"), wb.show_matrix(SID, "rho_2", "rho_1"),
                     wb.density(SID, "rho_1", "heatmap")):
        assert "**Density matrix of" in md


def test_workbench_noise_curve_of_one_photon():
    wb.create(SID, "photon", photon_how="named", polarization="D")
    md, fig, _ = wb.noise_sweep(SID, "rho_1", "amplitude_damping", "both")
    assert fig and "| 0.5 | 0.8660 | 0.8750 |" in md and "-0.0000" not in md      # |S| = √(γ² + 1 - γ)
    with pytest.raises(wb.WorkbenchError, match="two-photon"):
        wb.noise_sweep(SID, "rho_1", "collective_dephasing", "both")


# ------------------------------------------------------------------ run.py findings of 2026-10-31 (OpenAI gpt-4.1-mini)
def _tm(name, result):
    import json
    from langchain_core.messages import ToolMessage
    return ToolMessage(content=json.dumps(result), tool_call_id=name, name=name)


def test_key_times_of_an_earlier_question_are_not_used():
    """Q16: the check took 'CHSH reaches 2 at 0.1733' from the psi+ evolution of the PREVIOUS question and
    pushed the model to write it for phi+ under collective dephasing, where CHSH never reaches 2."""
    import graph
    at.create_state.invoke({"kind": "psi+"}); at.create_state.invoke({"kind": "phi+"})
    old = _tm("evolve_state", at.evolve_state.invoke({"handle": "rho_1", "process": "collective_decay",
                                                       "gamma": 1, "t_max": 2}))
    new = _tm("evolve_state", at.evolve_state.invoke({"handle": "rho_2", "process": "collective_dephasing",
                                                       "gamma": 1, "t_max": 2}))
    assert "stays above 2" in __import__("json").loads(new.content)["features"]["chsh_drops_to_2_at"]
    bad = graph.misattributed_key_times([new], "The CHSH maximum drops to 2 at about t = 0.1733.", [old, new])
    assert bad and "stays above 2" in bad[0] and "0.1733" not in bad[0].split("tool says")[1]
    assert graph.misattributed_key_times([new], "CHSH never drops to 2 (t = 2 is the end).", [old, new]) == []
    assert graph.misattributed_key_times([new], "The concurrence halves at t = 0.1733.", [old, new]) == []


def test_bell_names_given_to_plot_sweep_count_as_used():
    import graph
    from langchain_core.messages import AIMessage
    turn = [AIMessage(content="", tool_calls=[{"name": "plot_sweep", "id": "a", "args": {
        "sweep": "noise", "handles": ["phi+", "psi+"], "channel": "collective_dephasing"}}])]
    assert graph.bell_states_missing("Plot the concurrence of phi+ and psi+ versus ...", turn) == []


def test_two_sweeps_are_merged_and_both_are_checked():
    import graph
    calls = [{"name": "plot_sweep", "id": i, "type": "tool_call",
              "args": {"sweep": "noise", "handles": [h], "channel": "collective_dephasing",
                       "quantities": ["concurrence"]}} for i, h in (("a", "phi+"), ("b", "psi+"))]
    out, notes = graph.merge_noise_plots(calls)
    assert len(out) == 1 and out[0]["args"]["handles"] == ["phi+", "psi+"] and len(notes) == 1
    r1 = at.plot_sweep.invoke({"sweep": "noise", "handles": ["phi+"], "channel": "collective_dephasing",
                               "quantities": ["concurrence"]})
    r2 = at.plot_sweep.invoke({"sweep": "noise", "handles": ["psi+"], "channel": "collective_dephasing",
                               "quantities": ["concurrence"]})
    table = ("| State | p | Concurrence |\n|---|---|---|\n| phi+ | 0.5 | 0.0625 |\n| psi+ | 0.5 | 1.0000 |")
    assert graph.wrong_parameter_rows(table, [_tm("plot_sweep", r1), _tm("plot_sweep", r2)]) == []


def test_asymptotic_minimum_is_not_a_flat_stretch():
    """Q15: 'Minimum 0.0000 at 0.97 to 1' made the model say phi+ becomes separable near p = 0.97."""
    r = at.plot_sweep.invoke({"sweep": "noise", "handles": ["phi+"], "channel": "collective_dephasing",
                              "quantities": ["concurrence"]})["results"]["phi+"]["measures"]["concurrence"]
    assert r["min"]["at"] == 1.0 and r["crossings"] == "none in this range"
    w = at.plot_sweep.invoke({"sweep": "werner_p", "quantities": ["concurrence"]})["results"]["Werner on psi-"]
    assert w["measures"]["concurrence"]["min"]["at"] == "0 to 0.33"            # a real flat zero stays a range


def test_table_cells_with_conditional_entropy_and_one_photon_wave_plate_panel():
    assert "S(A∣B)" in display.to_unicode("| Conditional entropy S(A|B) | -0.1524 |")
    d = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    text = display.format_waveplate(at.apply_waveplate.invoke({"handle": d, "plate": "QWP", "angle_deg": 0,
                                                                "qubit": "A"}))
    assert "nan" not in text and "undefined (circular)" in text


# ------------------------------------------------------------------ second run.py round (2026-10-31)
def test_circular_light_never_gets_an_azimuth():
    """Q2/Q3: 'circular polarization with azimuth 0°' and 'R (right circular, azimuth 45°, ellipticity 45°)'."""
    import graph
    text = at.create_single_photon_state.invoke({"amp_H": "1", "amp_V": "i"})
    assert "azimuth undefined (circular polarization)" in text and "ellipticity 45.00°" in text
    msgs = [_tm("create_single_photon_state", text)]
    cases = {"Its polarization ellipse corresponds to circular polarization with azimuth 0° and ellipticity +45°.":
             "circular polarization with azimuth undefined and ellipticity +45°.",
             "| Polarization ellipse | Azimuth 0°, Ellipticity +45° |":
             "| Polarization ellipse | azimuth undefined, Ellipticity +45° |",
             "| D (linear, azimuth 45°) | R (right circular, azimuth 45°, ellipticity 45°) |":
             "| D (linear, azimuth 45°) | R (right circular, azimuth undefined, ellipticity 45°) |"}
    for wrong, right in cases.items():
        assert graph.remove_false_claims(wrong, msgs)[0].endswith(right)      # D keeps its real azimuth 45°
    ok = "| D (linear, azimuth 45°) | R (right circular, azimuth undefined) |"
    assert graph.remove_false_claims(ok, msgs)[0] == ok                       # linear D keeps its azimuth
    d = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    r = at.apply_waveplate.invoke({"handle": d, "plate": "QWP", "angle_deg": 0, "qubit": "A"})
    assert r["after"]["ellipse_azimuth_deg"] == "undefined (circular)" and r["before"]["ellipse_azimuth_deg"] == 45


def test_reduced_state_accepts_the_usual_ways_to_name_the_photon():
    """Q6: the call was rejected and the model then asked the user which photon (the question said A)."""
    at.create_state.invoke({"kind": "phi+"})
    for args in ({"photon": "A"}, {"photon": "photon A"}, {"qubit": "A"}, {"photon": "a"}):
        assert "photon A of rho_1" in at.reduced_state.invoke({"handle": "rho_1", **args})
    with pytest.raises(Exception, match="photon='A' or photon='B'"):
        at.reduced_state.invoke({"handle": "rho_1"})


def test_helpful_messages_for_bell_names_and_invented_handles():
    at.create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="create_state\\(kind='phi\\+'\\)"):
        at.evolve_state.invoke({"handle": "phi+", "process": "dephasing", "gamma": 1, "t_max": 1})
    with pytest.raises(Exception, match="wait for that result"):
        at.show_density_matrix.invoke({"handle": "rho_1_after_amplitude_damping"})


def test_no_bell_violation_becomes_no_chsh_violation():
    import graph
    fixed, why = graph.remove_false_claims("- CHSH values are below 2 in both cases, so no Bell violation remains.", [])
    assert "no CHSH violation remains" in fixed and "other Bell inequalities" in why[0]


def test_a_constant_curve_has_one_flat_extremum():
    """Q15: psi+ under collective dephasing (C = 1 within 3e-8 of round-off) was listed as 30 separate minima."""
    r = at.plot_sweep.invoke({"sweep": "noise", "handles": ["psi+"], "channel": "collective_dephasing",
                              "quantities": ["concurrence"]})["results"]["psi+"]["measures"]["concurrence"]
    assert r["min"]["at"] == "0 to 1" and r["max"]["at"] == "0 to 1"


# ------------------------------------------------------------------ the answer checks with one-photon states
def _run_graph(script, question):
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    import graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self
    script = list(script) + [script[-1]] * 4         # a check may ask the model again: it answers the same
    app = graph.build_graph(Scripted(messages=iter(script)))
    out = app.invoke({"messages": [("user", question)], "retries": 0},
                     {"configurable": {"thread_id": "t"}, "recursion_limit": 60})
    return graph._text(out["messages"][-1])


def _call(name, args, i):
    from langchain_core.messages import AIMessage
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i}])


ONE_PHOTON_ANSWERS = [
    "| State | S1 | S2 | S3 | Degree of polarization | Purity |\n|---|---|---|---|---|---|\n"
    "| rho_1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |",
    "**rho_1 is R.** Its purity is 1.0000 and its concurrence is not defined for one photon.\n\n"
    "| Quantity | rho_1 |\n|---|---|\n| Ellipticity | 45.00 |\n| P(H) | 0.5000 |",
    "The degree of polarization halves at t = 0.6931 and reaches 0.1353 at t = 2.",
    "| Element | rho_1 |\n|---|---|\n| ρ[H,H] | 0.5000 |\n| ρ[H,V] | -0.5000i |",
]


@pytest.mark.parametrize("answer", ONE_PHOTON_ANSWERS)
def test_answer_checks_run_on_one_photon_states(answer):
    """run.py crashed in wrong_row_values (it computed the concurrence of a 2x2 state)."""
    from langchain_core.messages import AIMessage
    script = [_call("create_single_photon_state", {"amp_H": "1", "amp_V": "i"}, "a"),
              _call("compute_measures", {"handle": "rho_1"}, "b"),
              _call("show_density_matrix", {"handle": "rho_1"}, "c"),
              AIMessage(content=answer)]
    text = _run_graph(script, "Create a single photon in the state (|H> + i|V>)/sqrt(2) and describe it.")
    assert "could not run" not in text


def test_answer_checks_run_on_reduced_states_and_evolution():
    from langchain_core.messages import AIMessage
    script = [_call("create_state", {"kind": "phi+"}, "a"),
              _call("reduced_state", {"handle": "rho_1", "photon": "A"}, "b"),
              _call("evolve_state", {"handle": "rho_2", "process": "depolarizing", "gamma": 1, "t_max": 1}, "c"),
              AIMessage(content="| State | Degree of polarization | Entropy |\n|---|---|---|\n"
                                "| rho_2 (photon A of rho_1) | 0.0000 | 1.0000 |\n| rho_3 | 0.0000 | 1.0000 |")]
    text = _run_graph(script, "Create phi+ and give the reduced state of photon A, then let it depolarize.")
    assert "could not run" not in text and "rho_2" in text


def test_a_failing_check_never_crashes_the_program(monkeypatch):
    import graph
    from langchain_core.messages import AIMessage
    monkeypatch.setattr(graph, "completion_and_claim_notes", lambda *a: 1 / 0)
    text = _run_graph([_call("create_state", {"kind": "phi+"}, "a"), AIMessage(content="Created phi+.")],
                      "Create phi+.")
    assert text.startswith("Created phi+.") and "could not run (ZeroDivisionError)" in text


def test_two_photon_rules_do_not_correct_statements_about_one_photon():
    import graph
    h = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    u = handle_of(at.apply_local_noise.invoke({"handle": h, "channel": "depolarizing", "param": 1.0}))
    msgs = [_tm("compute_measures", at.compute_measures.invoke({"handle": u}))]
    text = f"{u} is unpolarized, i.e. maximally mixed (purity 0.5000)."
    assert graph.remove_false_claims(text, msgs)[0] == text                  # I/2 IS maximally mixed for one photon
    at.create_state.invoke({"kind": "werner", "p": 0.5})
    m2 = [_tm("compute_measures", at.compute_measures.invoke({"handle": "rho_3"}))]
    assert graph.remove_false_claims("rho_3 is maximally mixed.", m2)[1]     # a Werner state with p = 0.5 is not


# ------------------------------------------------------------------ fourth run.py round (2026-10-31)
def test_closer_to_the_classical_bound_is_checked():
    """Q10: 'The CHSH maximum is closer to the classical bound 2 for local noise' (local 1.3859, global 1.9799)."""
    import graph
    at.create_state.invoke({"kind": "phi+"})
    g = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}))
    loc = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.3}))
    msgs = [_tm("compute_measures", at.compute_measures.invoke({"handle": h})) for h in (g, loc)]
    wrong = ("- The CHSH maximum is closer to the classical bound 2 for local noise, indicating weaker quantum "
             "correlations.")
    fixed, why = graph.remove_false_claims(wrong, msgs)
    assert "closer" not in fixed and "is the one closer to 2" in why[0]
    right = "- The CHSH maximum is closer to the classical bound 2 for global noise."
    assert graph.remove_false_claims(right, msgs)[0] == right


def test_unclosed_italic_line_is_closed():
    assert display.to_unicode("*Conventions: CHSH max ≤ 2 means no CHSH violation.") == \
        "*Conventions: CHSH max ≤ 2 means no CHSH violation.*"
    assert display.to_unicode("* a bullet") == "* a bullet" and display.to_unicode("**bold**") == "**bold**"


# ---------------------------------------------------------------- run of 2026-10-31 (DeepSeek, all 16)
def test_destroys_more_entanglement_is_not_a_false_comparison():
    """Q10 (OpenAI): 'Local depolarizing noise destroys more entanglement than global' is TRUE (local leaves
    less); the comparison check read 'more ... than global' as 'more concurrence' and deleted the verdict."""
    import graph
    at.create_state.invoke({"kind": "phi+"})
    g = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}))
    loc = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.3}))
    msgs = [_tm("compute_measures", at.compute_measures.invoke({"handle": h})) for h in (g, loc)]
    for right in ("Local depolarizing noise destroys more entanglement than global depolarizing noise at the same p = 0.3.",
                  "Local noise reduces the concurrence more than global noise.",
                  "Global noise is less destructive to entanglement than local noise.",
                  "Global noise leaves more entanglement than local noise.",
                  "Local noise gives a lower concurrence than global noise."):
        assert graph.remove_false_claims(right, msgs) == (right, []), right
    for wrong in ("Global depolarizing noise destroys more entanglement than local depolarizing noise.",
                  "Local noise leaves more entanglement than global noise."):
        assert graph.remove_false_claims(wrong, msgs)[1], wrong


def test_comparison_uses_only_this_question():
    """Measures of states from EARLIER questions must not switch the check off (or on)."""
    import graph
    from langchain_core.messages import HumanMessage
    at.create_state.invoke({"kind": "phi+"})
    old = _tm("compute_measures", at.compute_measures.invoke({"handle": "rho_1"}))
    g = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}))
    loc = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.3}))
    msgs = [HumanMessage("first question"), old, HumanMessage("compare global and local")]
    msgs += [_tm("compute_measures", at.compute_measures.invoke({"handle": h})) for h in (g, loc)]
    assert graph.remove_false_claims("Local noise leaves more entanglement than global noise.", msgs)[1]


def test_one_photon_halving_time_is_a_key_time():
    """Q4: the tool computed the halving time 0.6931 but only as a separate number, so the model's row at
    t = 0.6931 was rejected. It is now in 'at_key_times' (exact values) and in the panel."""
    import graph
    h = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    r = at.evolve_state.invoke({"handle": h, "process": "dephasing", "gamma": 1, "t_max": 2})
    key = r["features"]["at_key_times"]
    (name, v), = key.items()
    assert name.startswith("t=0.6931") and v["degree_of_polarization"] == 0.5 and v["purity"] == 0.625
    assert "| t = 0.6931 (degree of polarization halves) | 0.0000 | 0.5000 |" in display.format_evolution(r)
    turn = [_tm("evolve_state", r)]
    assert graph.times_not_in_tools("Let it evolve up to t = 2", turn, "It halves at t = 0.6931.") == []


def test_one_photon_evolution_may_say_maximally_mixed():
    """Q4/Q7: 'approaches the maximally mixed state I/2' (one photon) and 'both photons are maximally mixed'
    (reduced states of a Werner state) are correct and were removed."""
    import graph
    h = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    r = at.evolve_state.invoke({"handle": h, "process": "dephasing", "gamma": 1, "t_max": 8})
    ok = "• The photon approaches the maximally mixed (unpolarized) state."
    assert graph.remove_false_claims(ok, [_tm("evolve_state", r)]) == (ok, [])
    w = handle_of(at.create_state.invoke({"kind": "werner", "p": 0.8}))
    msgs = [_tm("compute_measures", at.compute_measures.invoke({"handle": w}))]
    for ok in ("Both photons are maximally mixed (entropy 1 bit each).",
               "| S(A) photon A entropy | 1.0000 bit (maximally mixed) |",
               "Each reduced state is I/2, i.e. maximally mixed."):
        assert graph.remove_false_claims(ok, msgs) == (ok, []), ok
    assert graph.remove_false_claims("The Werner state is maximally mixed.", msgs)[1]


def test_chsh_half_label_is_corrected():
    """Q16: the model labelled a column 'CHSH/2' but filled it with CHSH values (2.8284 ...)."""
    import graph
    ans = ("| Time | Concurrence | CHSH/2 |\n|---|---|---|\n| t = 0 | 1.0000 | 2.8284 |\n| t = 2 | 0.0003 | 2.0000 |\n\n"
           "(Concurrence halves at t = 0.1733, where C = 0.5, CHSH/2 = 2.2361.)")
    fixed, why = graph.remove_false_claims(ans, [])
    assert "CHSH/2" not in fixed and "| CHSH max |" in fixed and "CHSH max = 2.2361" in fixed and why
    ok = "| t | CHSH max / 2 |\n|---|---|\n| 0 | 1.4142 |"
    assert graph.remove_false_claims(ok, []) == (ok, [])


def test_hh_vv_differ_by_two_v():
    import graph
    s = ("• Collective dephasing only attacks the HH–VV coherence (elements between states differing by one V), "
         "which phi+ has but psi+ does not.")
    fixed, why = graph.remove_false_claims(s, [])
    assert "differing by two V" in fixed and why


def test_table_separator_with_too_many_columns_is_repaired():
    """Q1: a header with '|S|' made DeepSeek write 8 '---' for 7 columns; the table did not render."""
    t = ("| State | S1 | S2 | S3 | |S| | Purity | Entropy (bits) |\n|---|---|---|---|---|---|---|---|\n"
         "| H (rho_1) | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |")
    out = display.to_unicode(t).split("\n")
    assert out[0].count("|") - 1 == out[1].count("|") - 1 == 7 and "∣S∣" in out[0]


def test_kraus_coherence_factors():
    """Q13: (1-p)^4 = 0.2401 was rejected as a number no tool produced; the tool now gives the factors."""
    r = at.get_kraus_operators.invoke({"channel": "collective_dephasing", "param": 0.3})
    f = r["coherence_factors"]
    assert f["HH-VV"] == 0.2401 and f["HH-HV"] == 0.7 and f["HV-VH"] == 1.0
    assert "HH–VV 0.2401" in display.format_kraus(r)
    assert "coherence_factors" not in at.get_kraus_operators.invoke({"channel": "amplitude_damping", "param": 0.3})


def test_one_photon_waveplate_shows_the_mapping():
    """Q3: the panel said 'H -> H; V -> V' for a QWP (the relative phase i was dropped)."""
    h = handle_of(at.create_single_photon_state.invoke({"polarization": "D"}))
    w = at.apply_waveplate.invoke({"handle": h, "plate": "QWP", "angle_deg": 0, "qubit": ""})
    text = display.format_waveplate(w)
    assert "polarization D → R" in text and "[[1, 0], [0, i]]" in text


def test_one_photon_tomography_note():
    r = at.reconstruct_single_photon.invoke({"counts_H": 980, "counts_V": 20, "counts_D": 510, "counts_A": 490,
                                             "counts_R": 500, "counts_L": 500})
    assert "400 Poisson" in r["note"] and r["convention"].count("(H+iV)") == 1
    s3 = float(r["stokes_mle"]["S3 (R-L)"].split("±")[1])
    assert abs(s3 - 0.0316) < 0.004                 # binomial σ = sqrt(1/1000)


# ---------------------------------------------------------------- re-run of 2026-10-31 (DeepSeek Q1, Q4, Q7, Q10, Q16)
def test_let_me_note_is_self_talk():
    import graph
    a = "• Here C = 0.55 and N = 0.275. Let me note the actual numbers: global gives C = 0.55, local gives C = 0.235."
    assert graph.strip_self_talk(a) == "• Global gives C = 0.55, local gives C = 0.235."
    ok = "• Let us compare the two states: global keeps more entanglement."
    assert graph.strip_self_talk(ok) == ok


def test_local_depolarizing_also_gives_werner():
    """Q10: 'Global depolarizing keeps the state a Werner state (rank structure preserved), while local
    depolarizing mixes in more components' is false: local depolarizing of a Bell state is Werner too."""
    import graph
    bad = ("• Global depolarizing keeps the state a Werner state (rank structure preserved), while local "
           "depolarizing mixes in more components, lowering purity (0.43 vs 0.62) and entanglement.")
    assert graph.remove_false_claims(bad, [])[1]
    for ok in ("• Global depolarizing gives a Werner state with weight 0.7, while local gives a Werner state with weight 0.49.",
               "• This matches the reference: local depolarizing p on both photons gives p_W = (1−p)² = 0.49, whereas "
               "global gives p_W = 1−p = 0.7."):
        assert graph.remove_false_claims(ok, []) == (ok, []), ok
    # and the physics behind the rule
    at.create_state.invoke({"kind": "phi+"})
    loc = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.3}))
    import qi_tools as qt
    assert np.allclose(at.STORE[loc], 0.49 * qt.dm(qt.bell_state("phi+")) + 0.51 * np.eye(4) / 4)


def test_tool_key_chsh_max_is_readable():
    assert display.to_unicode("violates CHSH (chsh_max = 2.2627 > 2)") == "violates CHSH (CHSH max = 2.2627 > 2)"


def test_more_comparison_phrasings_of_local_vs_global():
    """Q10 re-run: another sentence was removed with 'concurrence of rho_3 ... is not higher'. Phrasings where
    'larger/stronger/greater' describes the DAMAGE (not the entanglement) must not be flagged."""
    import graph
    at.create_state.invoke({"kind": "phi+"})
    g = handle_of(at.apply_global_noise.invoke({"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}))
    loc = handle_of(at.apply_local_noise.invoke({"handle": "rho_1", "channel": "depolarizing", "param": 0.3}))
    msgs = [_tm("compute_measures", at.compute_measures.invoke({"handle": h})) for h in (g, loc)]
    for right in ("Local noise has a much stronger effect on entanglement than global noise.",
                  "The drop in concurrence is larger for local noise than for global noise.",
                  "Local noise has a bigger impact on the entanglement than global noise.",
                  "Local depolarizing is far worse for entanglement than global depolarizing.",
                  "Global noise keeps a larger concurrence than local noise."):
        assert graph.remove_false_claims(right, msgs) == (right, []), right
    assert graph.remove_false_claims("Local noise keeps a larger concurrence than global noise.", msgs)[1]


def test_removed_sentences_are_logged(tmp_path, monkeypatch):
    import graph
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("QI_LOG_DIR", str(tmp_path))
    graph._log_removed([("• A wrong sentence.", "the reason")])
    text = (tmp_path / "answer_check.log").read_text(encoding="utf-8")
    assert "REMOVED: • A wrong sentence." in text and "REASON: the reason" in text


# ---------------------------------------------------------------- first Space run (2026-11-02)
def test_workbench_conditional_entropy_row_is_one_cell():
    """Workbench 'All measures': 'S(A|B)' split the last row into two cells and hid the value (-1 for phi+)."""
    import workbench as wb
    import session
    with session.use("wbtest"):
        at.create_state.invoke({"kind": "phi+"})
    md = wb.measures_table("wbtest", "rho_1")
    row = next(x for x in md.splitlines() if "Conditional entropy" in x)
    assert row.count("|") == 3 and "-1.0000" in row
    for _key, name in wb.MEASURE_NAMES:
        assert "|" not in name, name
    session.drop("wbtest")


def test_chsh_rounded_to_2_but_above_2_is_not_flagged():
    """Q16 on the Space: 'CHSH stays above 2 throughout' was removed because the sample at t = 1.4 shows 2.0000,
    but CHSH = 2√(1+C²) = 2.0000137 there; the tool (full precision) says it stays above 2."""
    import graph
    at.create_state.invoke({"kind": "phi+"})
    r = at.evolve_state.invoke({"handle": "rho_1", "process": "collective_dephasing", "gamma": 1, "t_max": 2})
    ok = "• CHSH stays above 2 throughout the evolution, approaching it only asymptotically."
    assert graph.remove_false_claims(ok, [_tm("evolve_state", r)]) == (ok, [])
    # a sample EXACTLY at 2 (bit flip p = 0.5) is still caught
    b = at.plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})
    assert graph.remove_false_claims("- CHSH remains above the local bound for all p.", [_tm("plot_measures_vs_noise", b)])[1]


def test_poincare_poles_are_circular():
    import graph
    s = "• rho_1 sits at the north pole of the Poincaré sphere (H), fully polarized and pure."
    fixed, why = graph.remove_false_claims(s, [])
    assert "lies on the equator of the Poincaré sphere (H)" in fixed and "poles are R and L" in why[0]


def test_meta_paragraph_to_the_checker_is_removed():
    """Space, 2026-11-02: '... are the user's claimed values ..., not parameters I need to call a tool for.
    ... So I keep my answer.' (a reply to the answer check) appeared at the top of the answer."""
    import graph
    a = ("The values 0.5, 0.625, and 2.2361 are the user's claimed values at t = 0.1733 — they are thresholds to "
         "compare against, not parameters I need to call a tool for. So I keep my answer.\n\n"
         "Your statement is incorrect on three points.\n\n| Quantity | Your claim |\n|---|---|\n| C | 0.5 |")
    out = graph.strip_meta(a)
    assert out.startswith("Your statement is incorrect") and "| C | 0.5 |" in out
    ok = "The user's photon is unpolarized.\n\nIts entropy is 1 bit."
    assert graph.strip_meta(ok) == ok


def test_tool_keys_become_words():
    assert display.to_unicode("The tool reports `concurrence_halves_at: 0.3466`, and C = 0.5.") == \
        "The tool reports concurrence halves at 0.3466, and C = 0.5."
    assert display.to_unicode("chsh_drops_to_2_at = 0.1733") == "CHSH drops to 2 at 0.1733"


def test_claim_without_a_process_rule_is_in_the_prompt():
    import routing
    assert "which one you assumed" in routing.CORE_RULES and "never about 'the user'" in routing.CORE_RULES


def test_chsh_half_as_a_row_label_is_corrected():
    """Space Q16 (2026-11-02): a transposed table with the ROW 'CHSH/2 | 2.8284 | 2.1926 | ...'."""
    import graph
    ans = ("| Quantity | t = 0 | t = 0.2 | t = 2.0 |\n|---|---|---|---|\n| Concurrence | 1.0000 | 0.4493 | 0.0003 |\n"
           "| CHSH/2 | 2.8284 | 2.1926 | 2.0000 |")
    fixed, why = graph.remove_false_claims(ans, [])
    assert "| CHSH max | 2.8284 | 2.1926 | 2.0000 |" in fixed and why
    ok = "| Quantity | t = 0 |\n|---|---|\n| CHSH max / 2 | 1.4142 |"
    assert graph.remove_false_claims(ok, []) == (ok, [])
