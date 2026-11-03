"""
test_certification.py - entanglement witnesses, the steering inequality and significance (nσ).
Run:  pytest -q test_certification.py
"""
import json
import os

import numpy as np
import pytest
from langchain_core.messages import ToolMessage

import qi_tools as qt
from agent_tools import (CURRENT_QUESTION, DESC, STORE, TOMO, _save, certify_correlations, create_state,
                         reconstruct_state_from_counts, simulate_tomography)

LAB_CSV = os.path.abspath("data/state_tomo_data.csv")


def fresh(question=""):
    STORE.clear(); DESC.clear(); TOMO.clear()
    CURRENT_QUESTION["text"] = question


def by_key(r):
    names = {"concurrence >": "concurrence", "fidelity witness": "witness_fidelity",
             "PPT witness": "witness_ppt", "F3 > 1": "steering", "exact criterion": "steering_exact",
             "CHSH": "chsh"}
    return {next(v for k, v in names.items() if k in t["property"]): t for t in r["tests"]}


# ------------------------------------------------------------------ physics
@pytest.mark.parametrize("p", [0.2, 0.5, 0.6, 0.8, 1.0])
def test_werner_witnesses_and_steering(p):
    rho = qt.werner_state(p)
    W, e = qt.fidelity_witness(rho)
    assert np.real(np.trace(W @ rho)) == pytest.approx(0.5 - (3 * p + 1) / 4, abs=1e-9)
    assert np.real(np.trace(qt.ppt_witness(rho) @ rho)) == pytest.approx((1 - 3 * p) / 4, abs=1e-9)
    assert qt.cjwr_steering(rho, 3) == pytest.approx(np.sqrt(3) * p, abs=1e-9)
    assert qt.cjwr_steering(rho, 2) == pytest.approx(np.sqrt(2) * p, abs=1e-9)


def test_witnesses_are_nonnegative_on_separable_states():
    rng = np.random.default_rng(5)
    ent = qt.werner_state(0.9, "phi+")
    W_f, _ = qt.fidelity_witness(ent)
    W_p = qt.ppt_witness(ent)

    def rand_dm(d):
        G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        r = G @ G.conj().T
        return r / np.trace(r)

    for _ in range(200):
        sep = sum(w * np.kron(rand_dm(2), rand_dm(2)) for w in rng.dirichlet(np.ones(3)))
        assert np.real(np.trace(W_f @ sep)) >= -1e-12
        assert np.real(np.trace(W_p @ sep)) >= -1e-12


def test_fidelity_witness_equals_half_minus_fully_entangled_fraction():
    rng = np.random.default_rng(2)
    G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    W, e = qt.fidelity_witness(rho)
    assert np.real(np.trace(W @ rho)) == pytest.approx(0.5 - qt.fully_entangled_fraction_2q(rho), abs=1e-12)
    assert qt.concurrence(qt.dm(e)) == pytest.approx(1, abs=1e-6)          # target is maximally entangled


# ------------------------------------------------------------------ the tool
@pytest.mark.parametrize("p,strongest", [(0.2, "not entangled"),
                                         (0.45, "entangled, not steerable with projective measurements"),
                                         (0.55, "steerable but no CHSH violation"),
                                         (0.65, "steerable but no CHSH violation"),
                                         (0.8, "Bell nonlocal")])
def test_hierarchy_for_exact_werner_states(p, strongest):
    fresh(); create_state.invoke({"kind": "werner", "p": p})
    r = certify_correlations.invoke({"handle": "rho_1"})
    assert r["strongest_property"] == strongest
    assert all("n_sigma" not in t for t in r["tests"])                    # exact state: no σ
    assert "no error bars" in r["error_method"]


def test_lab_state_significance_matches_tomography_error_bars():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV})
    assert rec["uncertainty_1sigma"] == {"purity": 0.0087, "concurrence": 0.012, "negativity": 0.006,
                                         "chsh_max": 0.015, "bell_like_fidelity": 0.0059}
    r = certify_correlations.invoke({"handle": rec["handle"]})
    t = by_key(r)
    assert t["concurrence"]["sigma"] == rec["uncertainty_1sigma"]["concurrence"]
    assert t["chsh"]["sigma"] == rec["uncertainty_1sigma"]["chsh_max"]
    assert t["chsh"]["value"] == pytest.approx(2.4572, abs=1e-4)
    assert t["chsh"]["n_sigma"] == pytest.approx(29.7, abs=0.15)
    assert t["chsh"]["n_sigma_realistic"] == pytest.approx(17.3, abs=0.15)
    assert t["steering"]["value"] == pytest.approx(1.4083, abs=1e-4)
    assert t["witness_fidelity"]["value"] == pytest.approx(0.5 - 0.8598, abs=1e-4)
    assert r["strongest_property"] == "Bell nonlocal"
    assert all(x["verdict"] == "shown (≥ 5σ)" for x in r["tests"])


def test_certification_without_error_bars_in_the_report_still_has_sigma():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    t = by_key(certify_correlations.invoke({"handle": rec["handle"]}))
    assert t["chsh"]["sigma"] == 0.015


def test_simulated_low_count_experiment_has_large_error_bars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "werner", "p": 0.75, "bell": "phi+"})
    s = simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 2e4, "time_per_setting_s": 0.05})
    r = certify_correlations.invoke({"handle": s["reconstructed_raw"]["handle"]})
    t = by_key(r)
    assert t["chsh"]["sigma"] > 0.05                                       # ~350 coincidences only
    assert t["chsh"]["n_sigma"] == pytest.approx((t["chsh"]["value"] - 2) / t["chsh"]["sigma"], abs=0.3)
    assert any("few counts" in c for c in r["caveats"])


def test_certification_panel():
    from display import result_panels
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    r = certify_correlations.invoke({"handle": rec["handle"]})
    panels = dict(result_panels([ToolMessage(content=json.dumps(r, ensure_ascii=False),
                                             name="certify_correlations", tool_call_id="c")]))
    text = panels["Certification"]
    assert "Bell nonlocal (CHSH > 2) | 2.4572 | 2 | 0.015 | 29.7σ | 17.3σ | shown (≥ 5σ) |" in text
    fresh(); create_state.invoke({"kind": "phi+"})
    r = certify_correlations.invoke({"handle": "rho_1"})
    text = dict(result_panels([ToolMessage(content=json.dumps(r), name="certify_correlations",
                                           tool_call_id="d")]))["Certification"]
    assert "σ" not in text.split("\n")[2]                                   # header without σ columns


def test_benchmark_value_for_certification():
    spec = {t["id"]: t for t in json.load(open("benchmark_tasks.json", encoding="utf-8"))["tasks"]}
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    t = by_key(certify_correlations.invoke({"handle": rec["handle"]}))
    assert t["chsh"]["n_sigma"] == spec["T28"]["numbers"][0]["value"]
    assert t["chsh"]["n_sigma_realistic"] == spec["T28"]["numbers"][1]["value"]


# ------------------------------------------------------------------ 2026-10-28 run: exact steering, wording
@pytest.mark.parametrize("p", [0.3, 0.5, 0.55, 0.8, 1.0])
def test_exact_t_state_steering_value_for_werner(p):
    assert qt.t_state_steering_value(qt.werner_state(p)) == pytest.approx(2 * p, abs=1e-6)


def test_exact_steering_row_only_for_unpolarized_photons():
    fresh(); create_state.invoke({"kind": "werner", "p": 0.55})
    r = certify_correlations.invoke({"handle": "rho_1"})
    t = by_key(r)
    assert t["steering_exact"]["holds"] and t["steering_exact"]["value"] == pytest.approx(1.1, abs=1e-4)
    assert "does not prove that the state is unsteerable" in t["steering"]["verdict"]
    assert "- 0.707|VH>" in r["fidelity_witness_target"] and "180" not in r["fidelity_witness_target"]
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    r = certify_correlations.invoke({"handle": rec["handle"]})
    assert "steering_exact" not in by_key(r) and "not available here" in r["caveats"][1]


def test_simulated_chsh_sigma_is_the_same_as_in_certification(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "werner", "p": 0.75, "bell": "phi+"})
    s = simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 2e4, "time_per_setting_s": 0.05})
    raw = s["reconstructed_raw"]
    t = by_key(certify_correlations.invoke({"handle": raw["handle"]}))
    assert raw["chsh_error_1sigma"] == t["chsh"]["sigma"]
    assert raw["chsh_violation_n_sigma"] == t["chsh"]["n_sigma"]
    assert raw["deviation_from_true_in_sigma"]["concurrence"] is not None


def test_marginal_significance_wording(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh(); create_state.invoke({"kind": "werner", "p": 0.75, "bell": "phi+"})
    s = simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 2e4, "time_per_setting_s": 0.05})
    v = s["reconstructed_raw"]["chsh_violation_verdict"]
    assert v.startswith("marginal (3-5σ)") and "few counts" in v
    t = by_key(certify_correlations.invoke({"handle": s["reconstructed_raw"]["handle"]}))
    assert t["chsh"]["verdict"].startswith("marginal")


def test_realistic_sigma_is_given_for_every_test():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    t = by_key(certify_correlations.invoke({"handle": rec["handle"]}))
    assert t["steering"]["sigma_realistic"] == 0.024 and t["chsh"]["sigma_realistic"] == 0.026
