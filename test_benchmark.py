"""
test_benchmark.py - checks the benchmark itself (offline, no API calls):
  * every expected number in benchmark_tasks.json is re-derived from the tested physics code
  * the number matcher and the scoring rules behave as intended
  * one task runs end to end through the real graph with a scripted model
Run:  pytest -q test_benchmark.py
"""
import json

import numpy as np
import pytest

import qi_tools as qt
from benchmark import number_found, run_task, score_task

SPEC = json.load(open("benchmark_tasks.json", encoding="utf-8"))
TASKS = {t["id"]: t for t in SPEC["tasks"]}


def values(tid):
    return [n["value"] for n in TASKS[tid]["numbers"]]


def bell(k):
    return qt.dm(qt.bell_state(k))


def damp(rho, ka, kb):
    return qt.apply_kraus(rho, qt.local_channel(ka, kb))


def evo(rho0, process, t, qubits="both", omega=0.0, axis="x"):
    from agent_tools import two_qubit_model
    H, Ls, rates = two_qubit_model(process, 1.0, qubits, omega, axis)
    return qt.lindblad_evolve(rho0, H, Ls, rates, t)


def first_time(f, lo, hi):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return (lo + hi) / 2


def close(got, expected, tol=6e-5):
    assert got == pytest.approx(expected, abs=tol)


# ------------------------------------------------------------------ the expected answers are right
def test_expected_values_werner_and_dephasing():
    w5, w3, w8 = (qt.werner_state(p) for p in (0.5, 0.3, 0.8))
    assert np.allclose([qt.concurrence(w5), qt.negativity(w5)], values("T01"))
    close(qt.chsh_max(w3), values("T02")[0]); close(qt.concurrence(w5), values("T02")[1])
    close(qt.concurrence(w8), values("T02")[2]); close(qt.chsh_max(w8), values("T02")[3])
    r = damp(bell("phi+"), qt.kraus_phase_damping(0.7), qt.kraus_phase_damping(0.7))
    for got, exp in zip((qt.concurrence(r), qt.negativity(r), qt.purity(r), qt.chsh_max(r)), values("T03")):
        close(got, exp)
    r1 = damp(bell("phi+"), qt.kraus_phase_damping(0.5), [qt.I2])
    close(qt.concurrence(r1), values("T18")[0])


def test_expected_values_pure_states_mixtures_fidelity():
    p = qt.dm(qt.pure_state_2q([np.cos(0.3), 0, 0, np.sin(0.3)]))
    for got, exp in zip((qt.concurrence(p), qt.negativity(p), qt.chsh_max(p)), values("T04")):
        close(got, exp)
    q = qt.dm(qt.pure_state_2q([1, 0, 0, 1], [0, 0, 0, 90]))
    for got, exp in zip((qt.concurrence(q), qt.negativity(q), qt.chsh_max(q), qt.mutual_information(q)), values("T05")):
        close(got, exp)
    a = qt.dm(qt.pure_state_2q([1 / np.sqrt(3), 0, 0, np.sqrt(2 / 3)]))
    close(qt.concurrence(damp(a, [qt.I2], qt.kraus_amplitude_damping(0.8))), values("T06")[0])
    m = qt.mixture([bell("phi+"), bell("psi+")], [0.8, 0.2])
    close(qt.concurrence(m), values("T09")[0]); close(qt.purity(m), values("T09")[1])
    close(qt.fidelity(m, bell("phi+")), values("T09")[2])
    close(qt.fidelity(bell("phi+"), qt.werner_state(0.8, "phi+")), values("T10")[0])


def test_expected_values_noise_plots():
    from agent_tools import plot_measures_vs_noise
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["features"]
    assert f["entanglement_vanishes_at"] == values("T07")[0]
    g = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"})["features"]
    close(g["entanglement_vanishes_at"], values("T08")[0], 6e-4)
    close(g["chsh_drops_to_2_at"], values("T08")[1], 6e-4)


def test_expected_values_time_evolution():
    phi = bell("phi+")
    close(first_time(lambda t: qt.concurrence(evo(phi, "dephasing", t)) > 0.5, 0, 2), values("T11")[0])
    a = qt.dm(qt.pure_state_2q([1 / np.sqrt(3), 0, 0, np.sqrt(2 / 3)]))
    close(first_time(lambda t: qt.concurrence(evo(a, "amplitude_decay", t)) > 1e-12, 0, 3), values("T12")[0])
    close(first_time(lambda t: qt.chsh_max(evo(phi, "depolarizing", t)) > 2, 0, 1), values("T13")[0])
    close(first_time(lambda t: qt.concurrence(evo(phi, "depolarizing", t)) > 1e-12, 0, 1), values("T13")[1])
    psi = bell("psi-")
    close(qt.concurrence(evo(psi, "none", 3.0, omega=2.0)), 1.0)        # T14: unchanged


def test_expected_values_lab_file():
    import os
    if not os.path.exists("data/state_tomo_data.csv"):
        pytest.skip("data/state_tomo_data.csv not present")
    from agent_tools import reconstruct_state_from_counts
    r = reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv", "error_bars": False})
    close(r["key_numbers"]["concurrence"], values("T15")[0])
    close(r["fit_quality"]["chi2_per_dof"], values("T16")[0], 0.005)
    s = reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv", "error_bars": False,
                                              "r_convention": "R=(H-iV)/sqrt2"})
    close(s["closest_maximally_entangled"]["phi_deg"], values("T17")[0], 0.05)


# ------------------------------------------------------------------ the scoring rules
def test_number_matching_respects_written_precision_and_sign():
    assert number_found("C = 0.7213 ± 0.012", 0.7213)
    assert number_found("about 0.72", 0.7213) and not number_found("about 0.7", 0.7213)   # too coarse
    assert not number_found("C = 0.7313", 0.7213)
    assert number_found("p = 0.5", 0.5) and number_found("C = 1", 1.0, min_decimals=0)
    assert number_found("φ = 108.3°", 108.3, sign="+") and not number_found("φ = −108.3°", 108.3, sign="+")
    assert number_found("purity 0.6201", 0.62005) and number_found("purity 0.6200", 0.62005)


def test_scoring_reports_each_failure():
    task = {"numbers": [{"label": "C", "value": 0.25}], "tools": ["compute_measures"],
            "text": ["entangled"], "forbidden": ["maximal"], "max_states_created": 1}
    good = score_task(task, "The state is entangled: C = 0.25.", "", ["create_state", "compute_measures"], 1, [])
    assert all(c["ok"] for c in good)
    bad = score_task(task, "C = 0.30, maximal.", "", ["create_state"], 2, ["Warning: these numbers"])
    failed = [c["check"] for c in bad if not c["ok"]]
    assert failed == ["number C = 0.25", "tool compute_measures", "says /entangled/", "avoids /maximal/",
                      "creates at most 1 states"]


def test_tables_count_only_when_allowed():
    task = {"numbers": [{"label": "C", "value": 0.7213}]}
    assert not score_task(task, "entangled", "| C | 0.7213 |", [], 0, [])[0]["ok"]
    assert score_task({**task, "where": "answer_or_tables"}, "entangled", "| C | 0.7213 |", [], 0, [])[0]["ok"]


# ------------------------------------------------------------------ end to end with a scripted model
def test_run_task_end_to_end_with_scripted_model():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from graph import build_graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    script = [AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "werner", "p": 0.5}, "id": "a"}]),
              AIMessage(content="", tool_calls=[{"name": "compute_measures", "args": {"handle": "rho_1"}, "id": "b"}]),
              AIMessage(content="**Concurrence 0.2500**, negativity 0.1250.")]
    run = run_task(build_graph(Scripted(messages=iter(script))), TASKS["T01"])
    checks = score_task(TASKS["T01"], run["answer"], run["tables"], run["tool_names"], run["states_created"],
                        SPEC["global_forbidden"])
    assert run["error"] is None and all(c["ok"] for c in checks), checks


# ------------------------------------------------------------------ 2026-10-27 20:01 run: scoring fixes
def test_scoring_of_answers_from_the_second_baseline_run():
    t19 = ("Please provide the full two-qubit state explicitly, for example in the form "
           "cos(0.3)∣HH⟩ + sin(0.3)∣VV⟩.")
    assert all(c["ok"] for c in score_task(TASKS["T19"], t19, "", [], 0, SPEC["global_forbidden"]))
    t21 = ("For two qubits, concurrence C and negativity N are both entanglement measures with the relation "
           "2N ≤ C. They coincide (2N = C) for pure states but can rank mixed states differently. Both are "
           "zero if and only if the state is separable.")
    assert all(c["ok"] for c in score_task(TASKS["T21"], t21, "", [], 0, SPEC["global_forbidden"]))


def test_answer_check_footnote_is_reported_not_scored():
    from benchmark import was_corrected
    ans = ("The state is pure and maximally entangled: concurrence 1.0000, negativity 0.5000, CHSH 2.8284, "
           "mutual information 2.0000.\n\n*(Answer check removed an incorrect statement: discord is not a "
           "correlation 'beyond entanglement': it includes entanglement.)*")
    tools = ["create_pure_state", "compute_measures"]
    assert was_corrected(ans)
    assert all(c["ok"] for c in score_task(TASKS["T05"], ans, "", tools, 1, SPEC["global_forbidden"]))
    assert not was_corrected("plain answer")
