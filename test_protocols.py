"""
test_protocols.py - teleportation fidelity, QKD key rate, bit-flip code and the extra measures.
Run:  pytest -q test_protocols.py
"""
import os

import numpy as np
import pytest

import qi_tools as qt
from agent_tools import (CURRENT_QUESTION, DESC, STORE, TOMO, _save, bit_flip_code, compute_measures,
                         create_state, qkd_key_rate, reconstruct_state_from_counts, teleportation_fidelity)

LAB_CSV = os.path.abspath("data/state_tomo_data.csv")


def fresh(question=""):
    STORE.clear(); DESC.clear(); TOMO.clear()
    CURRENT_QUESTION["text"] = question


def h(q):
    return 0.0 if q in (0, 1) else -q * np.log2(q) - (1 - q) * np.log2(1 - q)


# ------------------------------------------------------------------ fully entangled fraction
def test_exact_fully_entangled_fraction_matches_numerical_optimum():
    rng = np.random.default_rng(7)
    for _ in range(10):
        G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho = G @ G.conj().T
        rho /= np.trace(rho)
        assert qt.fully_entangled_fraction_2q(rho) == pytest.approx(qt.fully_entangled_fraction_numeric(rho),
                                                                      abs=1e-6)


@pytest.mark.parametrize("p", [0.0, 1 / 3, 0.5, 0.8, 1.0])
def test_werner_fully_entangled_fraction_and_teleportation(p):
    fresh(); create_state.invoke({"kind": "werner", "p": p})
    r = teleportation_fidelity.invoke({"handle": "rho_1"})
    f = (3 * p + 1) / 4
    assert r["fully_entangled_fraction"] == pytest.approx(f, abs=1e-4)
    assert r["teleportation_fidelity"] == pytest.approx((2 * f + 1) / 3, abs=1e-4)
    assert r["beats_classical"] == (p > 1 / 3 + 1e-9)


def test_rotated_bell_states_teleport_perfectly():
    for b in ("phi+", "phi-", "psi+", "psi-"):
        fresh(); create_state.invoke({"kind": b})
        assert teleportation_fidelity.invoke({"handle": "rho_1"})["teleportation_fidelity"] == 1.0
    fresh()
    psi = (np.kron([1, 0], [1, 0]) + np.exp(1j * 1.9) * np.kron([0, 1], [0, 1])) / np.sqrt(2)
    hd = _save(qt.dm(psi), "phased phi+")
    assert teleportation_fidelity.invoke({"handle": hd})["fully_entangled_fraction"] == pytest.approx(1, abs=1e-9)


def test_entangled_state_that_does_not_beat_classical_teleportation():
    fresh()
    p = 0.3                                       # p|psi+><psi+| + (1-p)|HH><HH|: entangled for all p > 0
    rho = p * qt.dm(qt.bell_state("psi+")) + (1 - p) * qt.dm(qt.ket("00"))
    hd = _save(rho, "test state")
    r = teleportation_fidelity.invoke({"handle": hd})
    assert qt.concurrence(rho) == pytest.approx(p, abs=1e-9)
    assert r["fully_entangled_fraction"] <= 0.5 and not r["beats_classical"] and r["note"]


def test_lab_state_teleportation():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    r = teleportation_fidelity.invoke({"handle": rec["handle"]})
    assert r["fully_entangled_fraction"] >= rec["closest_maximally_entangled"]["fidelity"] - 1e-4
    assert r["teleportation_fidelity"] == pytest.approx(0.9065, abs=1e-4)


# ------------------------------------------------------------------ QKD
@pytest.mark.parametrize("p", [1.0, 0.9, 0.78, 0.5])
def test_werner_key_rate(p):
    fresh(); create_state.invoke({"kind": "werner", "p": p, "bell": "phi+"})
    r = qkd_key_rate.invoke({"handle": "rho_1"})
    q = (1 - p) / 2
    for b in ("Z", "X", "Y"):
        assert r["qber_by_basis"][b]["qber"] == pytest.approx(q, abs=1e-5)
    assert r["standard_BB84_HV_key_DA_test"]["secret_key_fraction"] == pytest.approx(max(0, 1 - 2 * h(q)), abs=1e-5)


def test_key_rate_from_given_qber_and_threshold():
    assert qkd_key_rate.invoke({"qber_z": 0.03})["secret_key_fraction"] == pytest.approx(1 - 2 * h(0.03), abs=1e-5)
    assert qkd_key_rate.invoke({"qber_z": 0.109})["key_possible"]
    assert not qkd_key_rate.invoke({"qber_z": 0.111})["key_possible"]
    r = qkd_key_rate.invoke({"qber_z": 0.01, "qber_x": 0.14})          # unequal errors: still a key
    assert r["key_possible"] and r["secret_key_fraction"] == pytest.approx(1 - h(0.01) - h(0.14), abs=1e-5)
    with pytest.raises(Exception, match="fraction"):
        qkd_key_rate.invoke({"qber_z": 3})
    with pytest.raises(Exception, match="handle or qber_z"):
        qkd_key_rate.invoke({})


def test_lab_state_needs_phase_compensation_for_a_key():
    fresh()
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    r = qkd_key_rate.invoke({"handle": rec["handle"]})
    assert not r["standard_BB84_HV_key_DA_test"]["key_possible"]
    assert r["qber_by_basis"]["Z"]["qber"] == pytest.approx(0.01606, abs=1e-5)
    after = r["after_phase_compensation"]
    assert after["key_possible"] and after["secret_key_fraction"] == pytest.approx(0.29971, abs=1e-4)


# ------------------------------------------------------------------ bit-flip code
@pytest.mark.parametrize("p,helps", [(0.0, False), (0.1, True), (0.3, True), (0.5, False), (0.7, False)])
def test_bit_flip_code(p, helps):
    r = bit_flip_code.invoke({"p": p})
    assert r["p_logical"] == pytest.approx(3 * p ** 2 - 2 * p ** 3, abs=1e-6)
    assert r["code_helps"] == helps


# ------------------------------------------------------------------ extra measures
def test_entanglement_of_formation_and_entropies():
    fresh(); create_state.invoke({"kind": "phi+"})
    m = compute_measures.invoke({"handle": "rho_1"})
    assert m["entanglement_of_formation"] == 1.0 and m["entropy"] == 0.0 and str(m["entropy"]) == "0.0"
    assert m["entropy_A"] == m["entropy_B"] == 1.0 and m["coherence_rel_entropy"] == 1.0
    fresh(); create_state.invoke({"kind": "werner", "p": 0.5})
    m = compute_measures.invoke({"handle": "rho_1"})
    C = 0.25
    assert m["entanglement_of_formation"] == pytest.approx(h((1 + np.sqrt(1 - C ** 2)) / 2), abs=1e-6)
    ev = [0.625, 0.125, 0.125, 0.125]
    assert m["entropy"] == pytest.approx(-sum(x * np.log2(x) for x in ev), abs=1e-6)


def test_protocol_steps_are_summarized_in_words():
    import json
    from langchain_core.messages import AIMessage, ToolMessage
    from display import summarize_steps
    msgs = [AIMessage(content="", tool_calls=[{"name": "bit_flip_code", "args": {"p": 0.1}, "id": "a"}]),
            ToolMessage(content=json.dumps(bit_flip_code.invoke({"p": 0.1})), name="bit_flip_code", tool_call_id="a")]
    assert summarize_steps(msgs) == [("ok", "Computed: Bit-flip code at p = 0.1")]


def test_benchmark_values_for_protocols():
    import json
    spec = {t["id"]: t for t in json.load(open("benchmark_tasks.json", encoding="utf-8"))["tasks"]}
    fresh(); create_state.invoke({"kind": "werner", "p": 0.5, "bell": "phi+"})
    assert teleportation_fidelity.invoke({"handle": "rho_1"})["teleportation_fidelity"] == \
        pytest.approx(spec["T25"]["numbers"][0]["value"], abs=5e-5)
    assert qkd_key_rate.invoke({"qber_z": 0.03})["secret_key_fraction"] == \
        pytest.approx(spec["T26"]["numbers"][0]["value"], abs=5e-5)
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    assert qkd_key_rate.invoke({"handle": rec["handle"]})["after_phase_compensation"]["secret_key_fraction"] == \
        pytest.approx(spec["T27"]["numbers"][0]["value"], abs=5e-5)


def test_unknown_handle_error_lists_descriptions():
    fresh(); create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match=r"rho_1 = phi\+") as e:
        teleportation_fidelity.invoke({"handle": "rho_1c"})
    assert "do not invent" in str(e.value)


def test_qkd_summary_shows_both_qbers():
    import json
    from langchain_core.messages import AIMessage, ToolMessage
    from display import summarize_steps
    args = {"qber_z": 0.01, "qber_x": 0.14}
    msgs = [AIMessage(content="", tool_calls=[{"name": "qkd_key_rate", "args": args, "id": "a"}]),
            ToolMessage(content=json.dumps(qkd_key_rate.invoke(args)), name="qkd_key_rate", tool_call_id="a")]
    assert summarize_steps(msgs)[0][1] == "Computed: QKD key rate for QBER 0.01 (H/V), 0.14 (D/A)"


def test_entangled_but_not_useful_for_teleportation_example():
    rho = 0.3 * qt.dm(qt.bell_state("psi+")) + 0.7 * qt.dm(qt.ket("00"))
    f = qt.fully_entangled_fraction_2q(rho)
    assert qt.concurrence(rho) == pytest.approx(0.3, abs=1e-9)
    assert f == pytest.approx(0.35, abs=1e-9) and (2 * f + 1) / 3 < 2 / 3
