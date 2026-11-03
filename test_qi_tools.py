"""Regression tests: every expected value is an analytic result from the literature.
Run:  pytest -q test_qi_tools.py
"""
import numpy as np
import pytest
from qi_tools import *

lams = [0.0, 0.3, 0.7, 1.0]


def dephased_phi_plus(lam):
    K = local_channel(kraus_phase_damping(lam), kraus_phase_damping(lam))
    return apply_kraus(dm(bell_state("phi+")), K)


# ---------------- Stage 0: representation
def test_bell_reduced_state_is_maximally_mixed():
    rA = partial_trace(dm(bell_state("phi+")), [0], [2, 2])
    assert np.allclose(rA, np.eye(2) / 2)


def test_partial_trace_product_state():
    a, b = dm(np.array([1, 1j]) / np.sqrt(2)), dm(np.array([0.6, 0.8]))
    rho = np.kron(a, b)
    assert np.allclose(partial_trace(rho, [0], [2, 2]), a)
    assert np.allclose(partial_trace(rho, [1], [2, 2]), b)


def test_schmidt_bell():
    assert np.allclose(np.sort(schmidt_coefficients(bell_state("psi-"), 2, 2)), [0.5, 0.5])


def test_werner_valid():
    for p in [0, 0.3, 1]:
        assert is_valid_density_matrix(werner_state(p))


def test_correlation_matrix_phi_plus():
    assert np.allclose(correlation_matrix(dm(bell_state("phi+"))), np.diag([1, -1, 1]))


# ---------------- Stage 1/3: dynamics
def test_rabi_oscillation():
    Om, t = 2.0, 0.9
    rho = unitary_evolve(dm(ket("0")), 0.5 * Om * X, t)
    assert np.isclose(np.real(rho[1, 1]), np.sin(Om * t / 2) ** 2)


def test_channels_cptp():
    for K in (kraus_phase_damping(0.4), kraus_amplitude_damping(0.3), kraus_depolarizing(0.2)):
        assert is_cptp(K)
        assert np.min(np.linalg.eigvalsh(choi_matrix(K))) > -1e-12


def test_choi_identity_is_unnormalized_phi_plus():
    assert np.allclose(choi_matrix([I2]), 2 * dm(bell_state("phi+")))


def test_depolarizing_full():
    assert np.allclose(apply_kraus(dm(ket("0")), kraus_depolarizing(1.0)), np.eye(2) / 2)


def test_lindblad_pure_dephasing_matches_kraus():
    g, t = 0.8, 1.3
    rho0 = dm(np.array([1, 1]) / np.sqrt(2))
    rt = lindblad_evolve(rho0, 0 * Z, [Z], [g / 2], t)
    assert np.isclose(rt[0, 1], 0.5 * np.exp(-g * t))
    assert np.allclose(rt, apply_kraus(rho0, kraus_phase_damping(np.exp(-g * t))))


def test_lindblad_T1_decay():
    g, t = 0.5, 2.0
    rt = lindblad_evolve(dm(ket("1")), 0 * Z, [np.array([[0, 1], [0, 0]], complex)], [g], t)
    assert np.isclose(np.real(rt[1, 1]), np.exp(-g * t))


# ---------------- Stage 6: quantification (running example)
@pytest.mark.parametrize("lam", lams)
def test_dephased_phi_plus_measures(lam):
    rho = dephased_phi_plus(lam)
    assert np.isclose(concurrence(rho), lam ** 2, atol=1e-9)
    assert np.isclose(negativity(rho), lam ** 2 / 2, atol=1e-9)
    assert np.isclose(purity(rho), (1 + lam ** 4) / 2)
    assert np.isclose(coherence_l1(rho), lam ** 2)
    assert np.isclose(chsh_max(rho), 2 * np.sqrt(1 + lam ** 4))


@pytest.mark.parametrize("p", [0.2, 1 / 3, 0.5, 0.8, 1.0])
def test_werner_thresholds(p):
    rho = werner_state(p)
    assert np.isclose(concurrence(rho), max(0, (3 * p - 1) / 2), atol=1e-9)
    assert np.isclose(negativity(rho), max(0, (3 * p - 1) / 4), atol=1e-9)
    assert np.isclose(chsh_max(rho), 2 * np.sqrt(2) * p)
    assert is_ppt(rho) == (p <= 1 / 3 + 1e-12)


def test_werner_chsh_threshold():
    assert chsh_max(werner_state(1 / np.sqrt(2) + 1e-3)) > 2
    assert chsh_max(werner_state(1 / np.sqrt(2) - 1e-3)) < 2


def test_bell_measures():
    rho = dm(bell_state("psi-"))
    assert np.isclose(concurrence(rho), 1)
    assert np.isclose(entanglement_of_formation(rho), 1)
    assert np.isclose(log_negativity(rho), 1)
    assert np.isclose(mutual_information(rho), 2)
    assert np.isclose(quantum_discord(rho), 1, atol=1e-4)


def test_discord_zero_for_classical_state():
    rho = 0.5 * (np.kron(dm(ket("0")), dm(ket("0"))) + np.kron(dm(ket("1")), dm(ket("1"))))
    assert quantum_discord(rho) < 1e-6


def test_fidelity_trace_distance():
    rho, sig = dm(ket("0")), dm(np.array([1, 1]) / np.sqrt(2))
    assert np.isclose(fidelity(rho, sig), 0.5)
    assert np.isclose(trace_distance(rho, sig), 1 / np.sqrt(2))


def test_coherence_plus_state():
    rho = dm(np.array([1, 1]) / np.sqrt(2))
    assert np.isclose(coherence_l1(rho), 1)
    assert np.isclose(coherence_rel_entropy(rho), 1)


def test_amplitude_damping_sudden_death():
    a, b = 1 / np.sqrt(3), np.sqrt(2 / 3)          # |b| > |a|, |1> excited
    psi = a * ket("00") + b * ket("11")
    p_death = 1 - a / b                             # survival prob. where C -> 0
    for p, alive in [(p_death + 0.02, True), (p_death - 0.02, False)]:
        K = local_channel(kraus_amplitude_damping(1 - p), kraus_amplitude_damping(1 - p))
        C = concurrence(apply_kraus(dm(psi), K))
        assert (C > 1e-9) == alive
        assert np.isclose(C, max(0, 2 * p * b * (a - b * (1 - p))), atol=1e-9)


def test_three_tangle():
    ghz = (ket("000") + ket("111")) / np.sqrt(2)
    w = (ket("001") + ket("010") + ket("100")) / np.sqrt(3)
    assert np.isclose(three_tangle(ghz), 1)
    assert np.isclose(three_tangle(w), 0)


# ---------------- Stage 5: tomography round trip
def test_mle_tomography_round_trip():
    true = werner_state(0.9, "phi+")
    est = mle_tomography_2q(simulate_counts(true, 20000, rng=3), rng=4)
    assert is_valid_density_matrix(est, tol=1e-6)
    assert fidelity(true, est) > 0.995


# ---------------- Stage 8/9/10: protection, tasks, scaling
def test_bitflip_code():
    assert np.isclose(bitflip_code_logical_error(0.1), 0.028)
    assert bitflip_code_logical_error(0.4) < 0.4 < bitflip_code_logical_error(0.6)


def test_bb84_threshold():
    assert bb84_key_rate(0.10) > 0 > bb84_key_rate(0.12)


@pytest.mark.parametrize("p", [0.2, 0.6, 1.0])
def test_teleportation_werner(p):
    assert np.isclose(teleportation_fidelity(werner_state(p)), (1 + p) / 2, atol=1e-6)


def test_qfi_heisenberg_scaling():
    for N in (1, 2, 4):
        ghz = (ket("0" * N) + ket("1" * N)) / np.sqrt(2)
        G = sum(np.kron(np.kron(np.eye(2 ** k), Z / 2), np.eye(2 ** (N - k - 1))) for k in range(N))
        assert np.isclose(qfi_pure(ghz, G), N ** 2)


def test_tmsv():
    assert np.isclose(tmsv_log_negativity(np.log(2) / 2), 1.0)
