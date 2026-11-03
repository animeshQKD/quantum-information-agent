"""
test_evolution.py - the time-evolution (Lindblad) tool against analytic results.
Run:  pytest -q test_evolution.py
"""
import numpy as np
import pytest

import qi_tools as qt
from agent_tools import (CURRENT_QUESTION, STORE, apply_local_noise, compute_measures,
                         create_pure_state, create_state, evolve_state)


def fresh():
    STORE.clear()
    CURRENT_QUESTION["text"] = ""


def evolve(h, **kw):
    args = {"handle": h, "gamma": 1.0, "t_max": 2.0}
    args.update(kw)
    return evolve_state.invoke(args)


def test_trajectory_matches_exact_solution():
    rho0 = qt.dm(qt.bell_state("phi+"))
    H = 0.7 * np.kron(qt.X, qt.I2)
    Ls = [np.kron(qt.Z, qt.I2), np.kron(qt.I2, np.array([[0, 1], [0, 0]], complex))]
    rates = [0.3, 0.5]
    times = np.linspace(0, 2, 41)
    traj = qt.lindblad_trajectory(rho0, H, Ls, rates, times)
    for t, r in zip(times[::10], traj[::10]):
        assert np.allclose(r, qt.lindblad_evolve(rho0, H, Ls, rates, t), atol=1e-10)
    assert all(qt.is_valid_density_matrix(r, tol=1e-9) for r in traj)


def test_dephasing_phi_plus_decays_without_sudden_death():
    fresh(); create_state.invoke({"kind": "phi+"})
    f = evolve("rho_1", process="dephasing")["features"]
    assert f["at_t_max"]["concurrence"] == pytest.approx(np.exp(-4), abs=1e-4)      # C = e^(-2γt)
    assert f["concurrence_halves_at"] == pytest.approx(np.log(2) / 2, abs=1e-4)
    assert str(f["entanglement_vanishes_at"]).startswith("never")
    assert str(f["chsh_drops_to_2_at"]).startswith("stays above 2")                 # 2√(1+e^(-4γt)) > 2


def test_dephasing_equals_kraus_phase_damping():
    fresh(); create_state.invoke({"kind": "phi+"})
    r = evolve("rho_1", process="dephasing", gamma=0.8, t_max=1.5)
    apply_local_noise.invoke({"handle": "rho_1", "channel": "phase_damping", "param": float(np.exp(-0.8 * 1.5))})
    assert np.allclose(STORE[r["final_state"]], STORE[f"rho_{len(STORE)}"], atol=1e-10)
    assert "0.3012" in r["features"]["same_as_kraus_at_t_max"]


def test_amplitude_decay_sudden_death_time():
    fresh()
    CURRENT_QUESTION["text"] = "1/sqrt(3)|HH> + sqrt(2/3)|VV>"
    create_pure_state.invoke({"amp_HH": "1/sqrt(3)", "amp_VV": "sqrt(2/3)"})
    f = evolve("rho_1", process="amplitude_decay", t_max=3.0)["features"]
    t_sd = -np.log(1 - 1 / np.sqrt(2))                     # survival e^(-t) = 1 - a/b
    assert f["entanglement_vanishes_at"] == pytest.approx(t_sd, abs=1e-4)
    g = evolve("rho_1", process="amplitude_decay", t_max=3.0, qubits="B")["features"]
    assert str(g["entanglement_vanishes_at"]).startswith("never")                  # one qubit: no death
    assert g["at_t_max"]["concurrence"] == pytest.approx(2 * np.sqrt(2) / 3 * np.exp(-1.5), abs=1e-4)


def test_depolarizing_werner_thresholds():
    fresh(); create_state.invoke({"kind": "phi+"})
    f = evolve("rho_1", process="depolarizing", t_max=1.0)["features"]           # visibility e^(-2t)
    assert f["entanglement_vanishes_at"] == pytest.approx(np.log(3) / 2, abs=1e-4)
    assert f["chsh_drops_to_2_at"] == pytest.approx(np.log(2) / 4, abs=1e-4)


def test_local_hamiltonian_does_not_change_entanglement():
    fresh(); create_state.invoke({"kind": "psi-"})
    r = evolve("rho_1", process="none", gamma=0, t_max=3.0, omega=2.0, axis="y")
    f = r["features"]
    assert f["at_t0"] == f["at_t_max"] and "cannot change entanglement" in f["note"]
    m = compute_measures.invoke({"handle": r["final_state"]})
    assert m["concurrence"] == pytest.approx(1) and m["purity"] == pytest.approx(1)


def test_bad_inputs_are_refused():
    fresh(); create_state.invoke({"kind": "phi+"})
    for bad in ({"process": "heating"}, {"process": "dephasing", "gamma": -1},
                {"process": "dephasing", "t_max": 0}, {"process": "none", "axis": "w"}):
        with pytest.raises(ValueError):
            evolve("rho_1", **bad)


def test_step_line_and_figure():
    import json
    import os
    from langchain_core.messages import AIMessage, ToolMessage
    from display import summarize_steps
    fresh(); create_state.invoke({"kind": "phi+"})
    r = evolve("rho_1", process="dephasing")
    msgs = [AIMessage(content="", tool_calls=[{"name": "evolve_state", "id": "e", "args":
            {"handle": "rho_1", "process": "dephasing", "gamma": 1.0, "t_max": 2.0}}]),
            ToolMessage(content=json.dumps(r), tool_call_id="e", name="evolve_state")]
    kind, line = summarize_steps(msgs)[0]
    assert kind == "ok" and "Evolved rho_1 under dephasing" in line and os.path.basename(r["figure"]) in line
    assert r["figure"].endswith(".png")


def test_every_result_is_plain_json():
    """A sudden-death time came back as numpy float64; the model would have seen 'np.float64(1.9297)'."""
    import json
    fresh()
    CURRENT_QUESTION["text"] = "1/sqrt(3)|HH> + sqrt(2/3)|VV>"
    create_pure_state.invoke({"amp_HH": "1/sqrt(3)", "amp_VV": "sqrt(2/3)"})
    r = evolve("rho_1", process="amplitude_decay", t_max=3.0)
    text = json.dumps(r)                       # raises TypeError for numpy numbers
    assert "np.float64" not in text and type(r["features"]["entanglement_vanishes_at"]) is float
    from agent_tools import plot_measures_vs_noise
    json.dumps(plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"}))
