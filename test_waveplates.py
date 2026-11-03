"""
test_waveplates.py - wave plates, phase compensation and the detector model.
Run:  pytest -q test_waveplates.py
"""
import json
import os

import numpy as np
import pytest
from langchain_core.messages import AIMessage, ToolMessage

import qi_tools as qt
from agent_tools import (CURRENT_QUESTION, DESC, STORE, TOMO, apply_waveplate, compensate_phase,
                         create_pure_state, create_state, mix_states, reconstruct_state_from_counts,
                         simulate_tomography)

LAB_CSV = os.path.abspath("data/state_tomo_data.csv")


def fresh(question=""):
    STORE.clear(); DESC.clear(); TOMO.clear()
    CURRENT_QUESTION["text"] = question


def same_up_to_phase(u, v):
    return abs(abs(np.vdot(u, v)) - np.linalg.norm(u) * np.linalg.norm(v)) < 1e-9


H, V = np.array([1, 0], complex), np.array([0, 1], complex)
D, A = (H + V) / np.sqrt(2), (H - V) / np.sqrt(2)
R, L = (H + 1j * V) / np.sqrt(2), (H - 1j * V) / np.sqrt(2)


# ------------------------------------------------------------------ Jones matrices
def test_textbook_actions_of_hwp_and_qwp():
    assert same_up_to_phase(qt.hwp(22.5) @ H, D)
    assert same_up_to_phase(qt.hwp(22.5) @ V, A)
    assert same_up_to_phase(qt.hwp(45) @ H, V)
    assert same_up_to_phase(qt.hwp(0) @ D, A)                 # HWP at 0: V picks up a minus sign
    assert same_up_to_phase(qt.qwp(45) @ H, L)                # (H - iV)/√2
    assert same_up_to_phase(qt.qwp(45) @ V, R)
    assert same_up_to_phase(qt.qwp(0) @ D, R) or same_up_to_phase(qt.qwp(0) @ D, L)
    assert same_up_to_phase(qt.hwp(30) @ H, np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]))


@pytest.mark.parametrize("theta", [0, 13.7, 22.5, 45, 77])
@pytest.mark.parametrize("delta", [np.pi, np.pi / 2, 1.234])
def test_waveplate_is_unitary_and_periodic(theta, delta):
    U = qt.waveplate(theta, delta)
    assert np.allclose(U.conj().T @ U, np.eye(2))
    assert np.allclose(qt.waveplate(theta + 180, delta), U)


@pytest.mark.parametrize("alpha", [-170, -108.3, -45, 0, 30, 90, 108.3, 179])
def test_qwp_hwp_qwp_is_a_phase_shifter(alpha):
    theta = qt.phase_shifter_hwp_angle(alpha)
    assert 0 <= theta < 90
    U = qt.qwp(45) @ qt.hwp(theta) @ qt.qwp(45)
    assert abs(U[0, 1]) < 1e-12 and abs(U[1, 0]) < 1e-12
    assert np.isclose(U[1, 1] / U[0, 0], np.exp(1j * np.deg2rad(alpha)))


# ------------------------------------------------------------------ apply_waveplate tool
def test_waveplate_tool_keeps_entanglement_of_a_mixed_state():
    fresh()
    create_state.invoke({"kind": "werner", "p": 0.7, "bell": "phi+"})
    for plate, ang, q in (("HWP", 17.0, "A"), ("QWP", 33.0, "B"), ("retarder", 61.0, "A")):
        r = apply_waveplate.invoke({"handle": "rho_1", "plate": plate, "angle_deg": ang, "qubit": q,
                                    **({"retardance_deg": 40.0} if plate == "retarder" else {})})
        for k in ("concurrence", "negativity", "purity", "chsh_max"):
            assert r["after"][k] == pytest.approx(r["before"][k], abs=1e-4)


def test_hwp45_on_b_turns_phi_plus_into_psi_plus():
    fresh("Create phi+ and put a HWP at 45 degrees on photon B")
    create_state.invoke({"kind": "phi+"})
    r = apply_waveplate.invoke({"handle": "rho_1", "plate": "HWP", "angle_deg": 45, "qubit": "B"})
    assert r["after"]["fidelity_with_bell_states"]["psi+"] == pytest.approx(1, abs=1e-9)
    assert r["acts_on_H"] == "H -> 1.000|V> (= V)"
    r0 = apply_waveplate.invoke({"handle": "rho_1", "plate": "HWP", "angle_deg": 0, "qubit": "A"})
    assert r0["after"]["fidelity_with_bell_states"]["phi-"] == pytest.approx(1, abs=1e-9)


def test_waveplate_tool_rejects_bad_input():
    fresh(); create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="retardance"):
        apply_waveplate.invoke({"handle": "rho_1", "plate": "retarder", "angle_deg": 0, "qubit": "A"})
    with pytest.raises(Exception, match="qubit"):
        apply_waveplate.invoke({"handle": "rho_1", "plate": "HWP", "angle_deg": 0, "qubit": "both"})
    with pytest.raises(Exception, match="retardance 180"):
        apply_waveplate.invoke({"handle": "rho_1", "plate": "HWP", "angle_deg": 0, "qubit": "A",
                                "retardance_deg": 90})


def test_waveplate_on_the_wrong_bell_state_is_refused():
    fresh("Put a QWP at 45 degrees on photon A of psi-")
    create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="psi-"):
        apply_waveplate.invoke({"handle": "rho_1", "plate": "QWP", "angle_deg": 45, "qubit": "A"})


# ------------------------------------------------------------------ compensate_phase tool
@pytest.mark.parametrize("phi", [70.0, -108.3, 170.0])
@pytest.mark.parametrize("qubit", ["A", "B"])
def test_compensation_restores_phi_plus(phi, qubit):
    fresh()
    create_pure_state.invoke({"amp_HH": "1/sqrt(2)", "amp_VV": f"exp(i*{float(np.deg2rad(phi))!r})/sqrt(2)"})
    r = compensate_phase.invoke({"handle": "rho_1", "qubit": qubit})
    assert r["target"] == "phi+"
    assert r["fidelity_with_target_after"] == pytest.approx(1, abs=1e-6)
    assert r["phase_to_remove_deg"] == pytest.approx(phi, abs=0.01)


def test_compensation_of_hv_vh_state_gives_psi_plus():
    fresh()
    create_pure_state.invoke({"amp_HV": "1/sqrt(2)", "amp_VH": f"exp(i*{float(np.deg2rad(50))!r})/sqrt(2)"})
    r = compensate_phase.invoke({"handle": "rho_1", "qubit": "B"})
    assert r["target"] == "psi+" and r["fidelity_with_target_after"] == pytest.approx(1, abs=1e-6)


def test_compensation_cannot_remove_noise():
    fresh()
    psi = (np.kron(H, H) + np.exp(1j * 1.0) * np.kron(V, V)) / np.sqrt(2)
    from agent_tools import _save, _best_bell_like
    h = _save(0.8 * qt.dm(psi) + 0.2 * np.eye(4) / 4, "noisy phased phi+")
    r = compensate_phase.invoke({"handle": h})
    assert r["fidelity_with_target_after"] == pytest.approx(_best_bell_like(STORE[h])["fidelity"], abs=1e-4)
    assert r["fidelity_with_target_after"] == pytest.approx(0.85, abs=1e-4)      # 0.8 + 0.2/4


def test_compensation_of_the_lab_state():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    r = compensate_phase.invoke({"handle": rec["handle"], "qubit": "B"})
    assert r["phase_to_remove_deg"] == pytest.approx(-108.3, abs=0.05)
    assert r["wave_plates"]["hwp_angle_deg"] == pytest.approx(72.07, abs=0.02)
    assert r["fidelity_with_target_after"] == pytest.approx(0.8527, abs=1e-4)
    assert "convention_warning" in r and "17.93" in r["convention_warning"]
    rec2 = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False,
                                                 "r_convention": "R=(H-iV)/sqrt2"})
    r2 = compensate_phase.invoke({"handle": rec2["handle"], "qubit": "B"})
    assert r2["wave_plates"]["hwp_angle_deg"] == pytest.approx(17.93, abs=0.02)


# ------------------------------------------------------------------ detector model
def test_accidental_rate_formula():
    assert qt.accidental_rate(1e4, 2e4, 2e-9) == pytest.approx(0.4)


def test_accidentals_act_like_white_noise_on_phi_plus():
    rho = qt.dm(qt.bell_state("phi+"))
    labels = [a + b for a in "HVDARL" for b in "HVDARL"]
    Rp, eta, T, tau, dark = 1e6, 0.2, 0.1, 20e-9, 500.0
    sim = qt.simulate_experiment(rho, labels, Rp, T, eta, eta, dark, dark, tau, noiseless=True)
    S = Rp * eta * 0.5 + dark
    assert sim["accidental"]["HV"] == pytest.approx(S * S * tau * T)
    assert sim["true"]["HH"] == pytest.approx(Rp * eta * eta * 0.5 * T)
    f = 4 * S * S * tau * T / (Rp * eta * eta * T + 4 * S * S * tau * T)
    rho_m = qt.mle_tomography_counts(sim["counts"], rng=0)
    assert qt.concurrence(rho_m) == pytest.approx(1 - 1.5 * f, abs=2e-4)     # Werner-like: C = 1 - 3f/2


def test_simulated_tomography_tool(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "phi+"})
    r = simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 1e6, "window_ns": 20,
                                    "time_per_setting_s": 0.1})
    raw, sub, inf = (r["reconstructed_raw"], r["reconstructed_accidentals_subtracted"],
                     r["infinite_statistics_raw"])
    assert r["rates"]["accidental_fraction"] == pytest.approx(0.0198, abs=1e-4)
    assert inf["concurrence"] == pytest.approx(1 - 1.5 * 0.0198, abs=5e-4)
    assert raw["concurrence"] < 0.985 < sub["concurrence"]                 # subtraction restores most
    assert abs(raw["concurrence"] - inf["concurrence"]) < 4 * raw["concurrence_error_1sigma"] + 1e-3
    assert isinstance(r["counts_needed_for_target"], str) and "already reached" in r["counts_needed_for_target"]
    assert os.path.isfile(r["csv_file"])
    again = reconstruct_state_from_counts.invoke({"csv_path": r["csv_file"], "count_column": "counts_raw",
                                                  "error_bars": False})
    assert again["key_numbers"]["concurrence"] == pytest.approx(raw["concurrence"], abs=2e-4)
    assert {raw["handle"], sub["handle"]} <= set(STORE)


def test_counting_time_estimate_scales_as_one_over_sqrt_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "werner", "p": 0.8, "bell": "phi+"})
    r = simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 2e4, "time_per_setting_s": 0.05,
                                    "target_concurrence_error": 0.005})
    need = r["counts_needed_for_target"]
    sigma = r["reconstructed_raw"]["concurrence_error_1sigma"]
    assert isinstance(need, dict) and need["time_per_setting_s"] > 0.05
    assert need["time_per_setting_s"] == pytest.approx(0.05 * (sigma / 0.005) ** 2, rel=0.05)


def test_simulation_rejects_bad_parameters():
    fresh(); create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="eta_a"):
        simulate_tomography.invoke({"handle": "rho_1", "eta_a": 1.5})
    with pytest.raises(Exception, match="coincidences"):
        simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 10, "time_per_setting_s": 0.01})


# ------------------------------------------------------------------ display and answer checks
def test_new_result_panels(tmp_path, monkeypatch):
    from display import result_panels
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "phi+"})
    msgs = []
    for name, tool, args in (("apply_waveplate", apply_waveplate,
                              {"handle": "rho_1", "plate": "HWP", "angle_deg": 22.5, "qubit": "A"}),
                             ("compensate_phase", compensate_phase, {"handle": "rho_1"}),
                             ("simulate_tomography", simulate_tomography,
                              {"handle": "rho_1", "time_per_setting_s": 0.2})):
        msgs.append(ToolMessage(content=json.dumps(tool.invoke(args), ensure_ascii=False),
                                name=name, tool_call_id=name))
    panels = dict(result_panels(msgs))
    assert set(panels) == {"Wave plate", "Phase compensation", "Simulated experiment"}
    assert "(= D)" in panels["Wave plate"] and "| concurrence | 1.0000 | 1.0000 |" in panels["Wave plate"]
    assert "1.0000 → 1.0000" in panels["Phase compensation"]
    assert "Accidentals subtracted" in panels["Simulated experiment"]


def test_log_base_claim_is_removed():
    from graph import remove_false_claims
    text = ("CHSH drops to 2 at t = 0.1733.\n\n*Conventions: concurrence and negativity use log base 2; "
            "CHSH max is the maximum Bell parameter.*\nThe logarithmic negativity uses log base 2.")
    out, why = remove_false_claims(text, [])
    assert "concurrence and negativity use log" not in out
    assert "logarithmic negativity uses log base 2" in out and why


def test_waveplate_tools_count_as_using_a_state():
    from graph import unused_new_states
    turn = [ToolMessage(content="Created rho_1 (phi+).", name="create_state", tool_call_id="a"),
            ToolMessage(content="Created rho_2 (psi-).", name="create_state", tool_call_id="b"),
            AIMessage(content="", tool_calls=[{"name": "apply_waveplate", "id": "c",
                                               "args": {"handle": "rho_1", "plate": "HWP"}},
                                              {"name": "simulate_tomography", "id": "d",
                                               "args": {"handle": "rho_2"}}])]
    assert unused_new_states(turn) == []


# ------------------------------------------------------------------ benchmark answers T22-T24
def test_benchmark_values_for_waveplates_and_detectors(tmp_path, monkeypatch):
    spec = {t["id"]: t for t in json.load(open("benchmark_tasks.json", encoding="utf-8"))["tasks"]}
    val = lambda tid, i: spec[tid]["numbers"][i]["value"]      # noqa: E731
    fresh(); create_state.invoke({"kind": "phi+"})
    r = apply_waveplate.invoke({"handle": "rho_1", "plate": "HWP", "angle_deg": 45, "qubit": "B"})
    assert r["after"]["fidelity_with_bell_states"]["psi+"] == 1.0
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    c = compensate_phase.invoke({"handle": rec["handle"], "qubit": "B"})
    assert c["wave_plates"]["hwp_angle_deg"] == pytest.approx(val("T23", 0), abs=0.005)
    assert c["fidelity_with_target_after"] == pytest.approx(val("T23", 1), abs=5e-5)
    assert c["fidelity_with_target_before"] == pytest.approx(val("T23", 2), abs=5e-5)
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "phi+"})
    s = simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 1e6, "eta_a": 0.2, "eta_b": 0.2,
                                    "dark_a": 500, "dark_b": 500, "window_ns": 20,
                                    "time_per_setting_s": 0.1})
    assert s["reconstructed_raw"]["concurrence"] == pytest.approx(val("T24", 0), abs=5e-5)
    assert s["reconstructed_accidentals_subtracted"]["concurrence"] == pytest.approx(val("T24", 1), abs=5e-5)


def test_graph_runs_the_new_tools_end_to_end():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from benchmark import run_task, score_task
    from graph import build_graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    spec = json.load(open("benchmark_tasks.json", encoding="utf-8"))
    task = {t["id"]: t for t in spec["tasks"]}["T22"]
    script = [AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"}, "id": "a"}]),
              AIMessage(content="", tool_calls=[{"name": "apply_waveplate", "id": "b", "args": {
                  "handle": "rho_1", "plate": "HWP", "angle_deg": 45, "qubit": "B"}}]),
              AIMessage(content="You get **psi+** (fidelity 1.0000); the concurrence is unchanged (1.0000).")]
    run = run_task(build_graph(Scripted(messages=iter(script))), task)
    checks = score_task(task, run["answer"], run["tables"], run["tool_names"], run["states_created"],
                        spec["global_forbidden"])
    assert run["error"] is None and all(c["ok"] for c in checks), checks
    assert "Wave plate" in run["tables"]


def test_closest_state_phase_is_written_with_its_sign():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    r = compensate_phase.invoke({"handle": rec["handle"]})
    assert "e^(-i108.3°)" in r["closest_state"] and "i-" not in r["closest_state"]
