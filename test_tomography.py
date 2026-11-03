"""
test_tomography.py - the experimental-data tomography tool.
Synthetic CSV files with KNOWN states are written in a temp folder, reconstructed, and checked.
Run:  pytest -q test_tomography.py
"""
import csv

import numpy as np
import pytest

import qi_tools as qt
from langchain_core.messages import AIMessage

from agent_tools import (STORE, compare_states, create_state, read_counts_csv,
                         reconstruct_state_from_counts)


def write_csv(path, counts, extra_zero_columns=True):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L2R2 ", "L1R2 ", "L2R1 ", "L1R1 ", "L1R + L2R2", ""])   # like the lab file
        for k, n in counts.items():
            w.writerow([0, 0, 0, n, n, k] if extra_zero_columns else [n, k])


def make(tmp_path, rho, n=4000, name="t.csv", r_sign=+1, settings=None, seed=0):
    rng = np.random.default_rng(seed)
    labels = settings or [a + b for a in "HVDARL" for b in "HVDARL"]
    P = qt.tomography_projectors(labels, r_sign)
    counts = {k: int(rng.poisson(n * np.real(np.trace(rho @ Pk)))) for k, Pk in zip(labels, P)}
    path = tmp_path / name
    write_csv(path, counts)
    return str(path)


def test_known_bell_state_is_recovered(tmp_path):
    STORE.clear()
    psi = (np.array([1, 0, 0, 0]) + np.exp(-1j * np.deg2rad(108)) * np.array([0, 0, 0, 1])) / np.sqrt(2)
    true = 0.9 * qt.dm(psi) + 0.1 * np.eye(4) / 4
    r = reconstruct_state_from_counts.invoke({"csv_path": make(tmp_path, true, n=20000)})
    assert r["data"]["count_column_used"] == "L1R1" and r["data"]["settings"] == 36
    assert qt.fidelity(STORE[r["handle"]], true) > 0.995
    best = r["closest_maximally_entangled"]
    assert best["state"].startswith("(|HH>") and abs(best["phi_deg"] + 108) < 2
    assert 0.5 < r["fit_quality"]["chi2_per_dof"] < 2.5         # pure counting noise -> about 1
    u = r["uncertainty_1sigma"]["concurrence"]
    assert 0 < u < 0.02


def test_r_convention_only_conjugates(tmp_path):
    STORE.clear()
    path = make(tmp_path, qt.dm(qt.pure_state_2q([1, 0, 0, 1], [0, 0, 0, 60])), n=5000)
    a = reconstruct_state_from_counts.invoke({"csv_path": path, "error_bars": False})
    b = reconstruct_state_from_counts.invoke({"csv_path": path, "error_bars": False,
                                              "r_convention": "R=(H-iV)/sqrt2"})
    assert np.allclose(STORE[a["handle"]].conj(), STORE[b["handle"]], atol=1e-3)
    assert a["key_numbers"]["concurrence"] == pytest.approx(b["key_numbers"]["concurrence"], abs=1e-3)
    assert a["closest_maximally_entangled"]["phi_deg"] == pytest.approx(
        -b["closest_maximally_entangled"]["phi_deg"], abs=0.5)


def test_standard_16_settings_work(tmp_path):
    STORE.clear()
    true = qt.dm(qt.bell_state("psi-"))
    path = make(tmp_path, true, n=20000, settings=[a + b for a in "HVDR" for b in "HVDR"])
    r = reconstruct_state_from_counts.invoke({"csv_path": path, "error_bars": False})
    assert r["data"]["settings"] == 16 and qt.fidelity(STORE[r["handle"]], true) > 0.99


def test_too_few_settings_are_refused(tmp_path):
    path = make(tmp_path, np.eye(4) / 4, settings=["HH", "HV", "VH", "VV", "DD", "AA"])
    with pytest.raises(ValueError, match="cannot determine"):
        reconstruct_state_from_counts.invoke({"csv_path": path})


def test_missing_file_lists_what_exists(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        reconstruct_state_from_counts.invoke({"csv_path": str(tmp_path / "nope.csv")})


def test_duplicate_settings_are_refused(tmp_path):
    p = tmp_path / "dup.csv"
    write_csv(p, {"HH": 5, "HV": 3})
    with open(p, "a", newline="") as fh:
        csv.writer(fh).writerow([0, 0, 0, 7, 7, "HH"])
    with pytest.raises(ValueError, match="more than once"):
        read_counts_csv(str(p))


def test_named_column_and_auto_column(tmp_path):
    p = tmp_path / "c.csv"
    write_csv(p, {"HH": 5, "VV": 9})
    counts, used, totals = read_counts_csv(str(p))
    assert used == "L1R1" and counts == {"HH": 5, "VV": 9} and totals["L2R2"] == 0
    counts, used, _ = read_counts_csv(str(p), "L2R2")
    assert used == "L2R2" and counts == {"HH": 0, "VV": 0}


def test_compare_states():
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    create_state.invoke({"kind": "phi-"})
    r = compare_states.invoke({"handle_a": "rho_1", "handle_b": "rho_2"})
    assert r["fidelity"] == pytest.approx(0) and r["trace_distance"] == pytest.approx(1)


def test_fast_mle_matches_36_setting_round_trip():
    true = qt.werner_state(0.9, "phi+")
    est = qt.mle_tomography_counts(qt.simulate_counts(true, 20000, rng=3), rng=4)
    assert qt.fidelity(true, est) > 0.995


def test_agent_end_to_end_with_data_file(tmp_path):
    """Scripted LLM: calls the tomography tool, then answers with numbers from its result."""
    import json
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from display import summarize_steps
    from graph import build_graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    STORE.clear()
    path = make(tmp_path, qt.werner_state(0.9, "phi+"), n=3000)
    call = AIMessage(content="", tool_calls=[{"name": "reconstruct_state_from_counts", "id": "t1",
                                              "args": {"csv_path": path}}])
    first = build_graph(Scripted(messages=iter([call, AIMessage(content="placeholder")])))
    out = first.invoke({"messages": [("user", f"Reconstruct the state from {path}")], "retries": 0},
                       {"configurable": {"thread_id": "tomo"}})
    result = json.loads([m for m in out["messages"] if m.type == "tool"][0].content)
    steps = summarize_steps(out["messages"][1:])
    assert steps[0][0] == "ok" and "Reconstructed rho_1" in steps[0][1] and "36 settings" in steps[0][1]
    c, dc = result["key_numbers"]["concurrence"], result["uncertainty_1sigma"]["concurrence"]
    from graph import ungrounded_numbers
    assert ungrounded_numbers(f"Concurrence {c:.4f} ± {dc:.4f}", out["messages"]) == []


# ------------------------------------------------------------------ 2026-10-14 additions
def test_mle_gradient_is_exact():
    import scipy.optimize as so
    captured = {}
    orig = qt.minimize

    def spy(fun, x0, **kw):
        captured["f"] = fun
        return orig(fun, x0, **kw)

    qt.minimize = spy
    try:
        qt.mle_tomography_counts(qt.simulate_counts(qt.werner_state(0.8, "phi+"), 500, rng=1),
                                 restarts=0, rng=0)
    finally:
        qt.minimize = orig
    f = captured["f"]
    x = np.random.default_rng(5).normal(size=16)
    num = so.approx_fprime(x, lambda t: f(t)[0], 1e-6)
    assert np.allclose(f(x)[1], num, rtol=1e-4, atol=1e-3)


def test_fit_is_fully_converged_from_any_start(tmp_path):
    counts = qt.simulate_counts(qt.werner_state(0.7, "psi-"), 800, rng=2)
    a = qt.mle_tomography_counts(counts, rng=0)
    b = qt.mle_tomography_counts(counts, rng=99, restarts=4)
    assert np.allclose(a, b, atol=1e-6)


def test_lab_file_if_present():
    """Your own data file: the same numbers on every machine (skipped if the file is absent)."""
    import os
    if not os.path.exists("data/state_tomo_data.csv"):
        pytest.skip("data/state_tomo_data.csv not present")
    STORE.clear()
    r = reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv", "error_bars": False})
    k = r["key_numbers"]
    assert (k["purity"], k["concurrence"], k["negativity"], k["chsh_max"]) == (0.7643, 0.7213, 0.3599, 2.4572)
    assert r["closest_maximally_entangled"]["phi_deg"] == pytest.approx(-108.3, abs=0.1)


def test_report_tool_convention_note_and_table(tmp_path):
    from agent_tools import get_tomography_report
    from display import format_tomography_report
    STORE.clear()
    r = reconstruct_state_from_counts.invoke({"csv_path": make(tmp_path, qt.werner_state(0.9, "phi+"))})
    assert get_tomography_report.invoke({"handle": r["handle"]}) == r
    assert "complex-conjugate" in r["convention_note"] and "DIFFERENT states" in r["convention_note"]
    table = format_tomography_report(r)
    for key in ("purity", "concurrence", "negativity", "chsh_max"):
        assert f"{r['key_numbers'][key]:.4f} ± {r['uncertainty_1sigma'][key]:.2g}" in table
    assert "χ²/dof" in table and "Assumptions" in table and "|HH>" not in table   # kets made table-safe
    with pytest.raises(ValueError, match="not reconstructed"):
        create_state.invoke({"kind": "phi+"})
        get_tomography_report.invoke({"handle": f"rho_{len(STORE)}"})


def test_uncertainties_have_two_significant_figures(tmp_path):
    STORE.clear()
    r = reconstruct_state_from_counts.invoke({"csv_path": make(tmp_path, qt.werner_state(0.9, "phi+"))})
    for v in r["uncertainty_1sigma"].values():
        assert v == float(f"{v:.2g}")


def test_plot_density_matrix_explains_itself():
    from agent_tools import plot_density_matrix
    STORE.clear()
    create_state.invoke({"kind": "phi+"})
    out = plot_density_matrix.invoke({"handle": "rho_1"})
    assert out["figure"] == "figures/rho_1_city.png"
    assert "red bars < 0" in out["how_to_read"] and "do not prove" in out["how_to_read"]
    assert "ρ[HH,VV] = +0.500+0.000i (magnitude 0.500, phase 0.0°)" in out["elements_with_magnitude_at_least_0.05"]


def test_unused_created_state_is_flagged_once():
    from langchain_core.messages import ToolMessage
    from graph import unused_new_states

    def created(h, name="create_state"):
        return ToolMessage(content=f"Created {h} (phi+).", tool_call_id=h, name=name)

    def uses(name, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "x"}])

    q4 = [created("rho_2", "create_pure_state"), created("rho_3"), created("rho_4"),
          uses("compare_states", handle_a="rho_1", handle_b="rho_3"),
          uses("compare_states", handle_a="rho_1", handle_b="rho_4")]
    assert unused_new_states(q4) == ["rho_2"]
    chain = [created("rho_1"), uses("apply_local_noise", handle="rho_1", channel="bit_flip", param=0.1),
             ToolMessage(content="Created rho_2 = bit_flip(0.1)", tool_call_id="n", name="apply_local_noise")]
    assert unused_new_states(chain) == []
    assert unused_new_states([created("rho_1")]) == []


def test_failure_reasons_are_short():
    from llm import _reason
    assert _reason(RuntimeError("Error code: 429 - rate_limit_exceeded")) == "rate limit"
    assert _reason(RuntimeError("Error code: 400 - tool call validation failed")) == "bad tool call"
    assert _reason(RuntimeError("503 UNAVAILABLE model overloaded")) == "overloaded"
    assert _reason(TimeoutError("timed out")) == "timeout"
