"""
qi_tools.py - verified core tools for the Quantum Information Agent (starter set).

Pure NumPy/SciPy so the agent's numbers never depend on package conventions.
Conventions (enforced everywhere):
  * Big-endian qubit order: |q0 q1 ...>, q0 is the leftmost tensor factor.
  * Computational basis |0>=|H>, |1>=|V>.
  * Density matrices are complex numpy arrays, trace 1.
  * Choi matrix J = sum_ij |i><j| (x) E(|i><j|)  (trace = d, NOT normalized).
  * Logarithms base 2 unless stated.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm, expm, logm
from scipy.optimize import minimize

# ----------------------------------------------------------------- Stage 0
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [I2, X, Y, Z]


def ket(bits: str) -> np.ndarray:
    """Computational-basis ket, big-endian: ket('01') = |0>|1>."""
    v = np.zeros(2 ** len(bits), dtype=complex)
    v[int(bits, 2)] = 1
    return v


def dm(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def bell_state(name: str = "phi+") -> np.ndarray:
    s = 1 / np.sqrt(2)
    table = {
        "phi+": s * (ket("00") + ket("11")),
        "phi-": s * (ket("00") - ket("11")),
        "psi+": s * (ket("01") + ket("10")),
        "psi-": s * (ket("01") - ket("10")),
    }
    return table[name.lower()]


def werner_state(p: float, bell: str = "psi-") -> np.ndarray:
    """p |Bell><Bell| + (1-p) I/4."""
    return p * dm(bell_state(bell)) + (1 - p) * np.eye(4) / 4


def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-9) -> bool:
    herm = np.allclose(rho, rho.conj().T, atol=tol)
    tr1 = abs(np.trace(rho) - 1) < tol
    psd = np.min(np.linalg.eigvalsh((rho + rho.conj().T) / 2)) > -tol
    return bool(herm and tr1 and psd)


def partial_trace(rho: np.ndarray, keep: list[int], dims: list[int]) -> np.ndarray:
    """Trace out all subsystems not in `keep` (indices into dims, big-endian)."""
    n = len(dims)
    rho_t = rho.reshape(dims + dims)
    trace_out = [i for i in range(n) if i not in keep]
    # trace highest index first so axis numbers stay valid
    for count, i in enumerate(sorted(trace_out, reverse=True)):
        cur_n = n - count
        rho_t = np.trace(rho_t, axis1=i, axis2=i + cur_n)
    d = int(np.prod([dims[k] for k in keep]))
    return rho_t.reshape(d, d)


def partial_transpose(rho: np.ndarray, sys: int, dims: list[int]) -> np.ndarray:
    n = len(dims)
    rho_t = rho.reshape(dims + dims)
    axes = list(range(2 * n))
    axes[sys], axes[sys + n] = axes[sys + n], axes[sys]
    D = int(np.prod(dims))
    return rho_t.transpose(axes).reshape(D, D)


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    return np.real([np.trace(rho @ P) for P in (X, Y, Z)])


def correlation_matrix(rho: np.ndarray) -> np.ndarray:
    """T_ij = Tr[rho (sigma_i x sigma_j)], i,j in {x,y,z}."""
    S = (X, Y, Z)
    return np.real([[np.trace(rho @ np.kron(a, b)) for b in S] for a in S])


def schmidt_coefficients(psi: np.ndarray, dA: int, dB: int) -> np.ndarray:
    s = np.linalg.svd(np.asarray(psi).reshape(dA, dB), compute_uv=False)
    return s ** 2  # lambda_k


# ----------------------------------------------------------------- Stage 1/3
def unitary_evolve(rho: np.ndarray, H: np.ndarray, t: float) -> np.ndarray:
    U = expm(-1j * H * t)
    return U @ rho @ U.conj().T


def apply_kraus(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    return sum(K @ rho @ K.conj().T for K in kraus)


def is_cptp(kraus: list[np.ndarray], tol: float = 1e-9) -> bool:
    d = kraus[0].shape[1]
    return bool(np.allclose(sum(K.conj().T @ K for K in kraus), np.eye(d), atol=tol))


def kraus_phase_damping(lam: float) -> list[np.ndarray]:
    """Off-diagonals multiplied by lam."""
    return [np.sqrt((1 + lam) / 2) * I2, np.sqrt((1 - lam) / 2) * Z]


def kraus_amplitude_damping(gamma: float) -> list[np.ndarray]:
    """|1> is the excited state; decays to |0> with probability gamma."""
    return [np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)]


def kraus_depolarizing(p: float) -> list[np.ndarray]:
    """rho -> (1-p) rho + p I/2."""
    return [np.sqrt(1 - 3 * p / 4) * I2] + [np.sqrt(p / 4) * P for P in (X, Y, Z)]


def kraus_bit_flip(p: float) -> list[np.ndarray]:
    """With probability p apply X (H <-> V)."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]


def kraus_phase_flip(p: float) -> list[np.ndarray]:
    """With probability p apply Z. Same channel as phase damping with lambda = 1 - 2p."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]


def kraus_bit_phase_flip(p: float) -> list[np.ndarray]:
    """With probability p apply Y."""
    return [np.sqrt(1 - p) * I2, np.sqrt(p) * Y]


def local_channel(kraus_a, kraus_b) -> list[np.ndarray]:
    return [np.kron(a, b) for a in kraus_a for b in kraus_b]


def pure_state_2q(amps, phases_deg=(0, 0, 0, 0)) -> np.ndarray:
    """Normalized |psi> = sum_k a_k e^{i phi_k} |k>, k in (HH, HV, VH, VV) = (00, 01, 10, 11)."""
    a = np.asarray(amps, dtype=float) * np.exp(1j * np.deg2rad(np.asarray(phases_deg, dtype=float)))
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("All amplitudes are zero.")
    return a / n


def mixture(rhos, weights) -> np.ndarray:
    """Convex mixture sum_i w_i rho_i (weights must be >= 0 and sum to 1)."""
    w = np.asarray(weights, dtype=float)
    if len(w) != len(rhos) or np.any(w < -1e-12) or abs(w.sum() - 1) > 1e-9:
        raise ValueError("Weights must be non-negative, one per state, and sum to 1.")
    return sum(wi * r for wi, r in zip(w, rhos))


def choi_matrix(kraus: list[np.ndarray]) -> np.ndarray:
    d = kraus[0].shape[1]
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1
            J += np.kron(Eij, apply_kraus(Eij, kraus))
    return J


def lindblad_rhs(rho, H, Ls, rates):
    out = -1j * (H @ rho - rho @ H)
    for L, g in zip(Ls, rates):
        LdL = L.conj().T @ L
        out += g * (L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL))
    return out


def lindblad_evolve(rho0, H, Ls, rates, t: float) -> np.ndarray:
    """Exact solution via vectorized Liouvillian (row-stacking)."""
    d = rho0.shape[0]
    Id = np.eye(d)
    Lsup = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for L, g in zip(Ls, rates):
        LdL = L.conj().T @ L
        Lsup += g * (np.kron(L, L.conj()) - 0.5 * np.kron(LdL, Id) - 0.5 * np.kron(Id, LdL.T))
    vec = expm(Lsup * t) @ rho0.reshape(-1)
    return vec.reshape(d, d)


def liouvillian(H, Ls, rates) -> np.ndarray:
    """Superoperator of the Lindblad equation (row-stacking: vec(A rho B) = kron(A, B^T) vec(rho))."""
    d = H.shape[0]
    Id = np.eye(d)
    Lsup = -1j * (np.kron(H, Id) - np.kron(Id, H.T))
    for L, g in zip(Ls, rates):
        LdL = L.conj().T @ L
        Lsup = Lsup + g * (np.kron(L, L.conj()) - 0.5 * np.kron(LdL, Id) - 0.5 * np.kron(Id, LdL.T))
    return Lsup


def lindblad_trajectory(rho0, H, Ls, rates, times) -> list:
    """rho(t) for every t in `times` (uniformly spaced, starting at 0): one matrix exponential
    for the step, then repeated multiplication - exact for time-independent H and rates."""
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or abs(times[0]) > 1e-12 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at 0 and increase")
    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly spaced")
    d = rho0.shape[0]
    step = expm(liouvillian(H, Ls, rates) * dt)
    v = rho0.reshape(-1).astype(complex)
    out = [rho0.astype(complex)]
    for _ in times[1:]:
        v = step @ v
        r = v.reshape(d, d)
        out.append((r + r.conj().T) / 2)          # remove round-off anti-Hermitian part
    return out


# ----------------------------------------------------------------- Stage 5
def pauli_projectors_1q(r_sign: int = +1):
    """6 projectors: H,V,D,A,R,L with D=(H+V)/sqrt2, A=(H-V)/sqrt2,
    R=(H + r_sign*iV)/sqrt2, L=(H - r_sign*iV)/sqrt2  (default r_sign=+1: R=(H+iV)/sqrt2)."""
    s = 1 / np.sqrt(2)
    j = 1j * r_sign
    vecs = {"H": [1, 0], "V": [0, 1], "D": [s, s], "A": [s, -s], "R": [s, j * s], "L": [s, -j * s]}
    return {k: dm(np.array(v, dtype=complex)) for k, v in vecs.items()}


def simulate_counts(rho: np.ndarray, n_per_setting: int, rng=None) -> dict:
    """36 two-qubit projector counts, Poisson-sampled (rate = N * Tr[rho P])."""
    rng = np.random.default_rng(rng)
    P1 = pauli_projectors_1q()
    counts = {}
    for a, Pa in P1.items():
        for b, Pb in P1.items():
            mu = n_per_setting * np.real(np.trace(rho @ np.kron(Pa, Pb)))
            counts[a + b] = rng.poisson(max(mu, 0))
    return counts


def _cholesky_rho(t: np.ndarray) -> np.ndarray:
    """16 real parameters -> valid 4x4 density matrix  rho = T T^dagger / Tr(T T^dagger)."""
    T = np.zeros((4, 4), dtype=complex)
    rows, cols = np.tril_indices(4)
    T[rows, cols] = t[:10]
    off = rows != cols
    T[rows[off], cols[off]] += 1j * t[10:]
    R = T @ T.conj().T
    return R / np.real(np.trace(R))


def _rho_to_t(rho: np.ndarray) -> np.ndarray:
    """Inverse of _cholesky_rho (used to start the fit from a good guess)."""
    w, V = np.linalg.eigh((rho + rho.conj().T) / 2)
    rho = (V * np.clip(w, 1e-3, None)) @ V.conj().T          # make it strictly positive
    T = np.linalg.cholesky(rho / np.real(np.trace(rho)))
    rows, cols = np.tril_indices(4)
    off = rows != cols
    return np.concatenate([T[rows, cols].real, T[rows[off], cols[off]].imag])


def tomography_projectors(labels, r_sign: int = +1) -> np.ndarray:
    """Two-qubit projectors for setting labels like 'HV' (first letter = qubit A)."""
    P1 = pauli_projectors_1q(r_sign)
    return np.array([np.kron(P1[k[0]], P1[k[1]]) for k in labels])


def mle_tomography_counts(counts: dict, r_sign: int = +1, restarts: int = 2, rng=None) -> np.ndarray:
    """Poisson maximum-likelihood two-qubit tomography from projector counts.

    counts: {'HH': n, 'HV': n, ...}; any informationally complete subset of the 36
    H/V/D/A/R/L settings (e.g. the standard 16 or all 36). Assumes the same pair flux and
    counting time for every setting; the overall rate N is fitted (closed form), so the
    likelihood is  sum_i n_i log p_i - S log(sum_i p_i)  with p_i = Tr(rho P_i), S = sum n_i.
    Physicality is built in through rho = T T^dagger / Tr(T T^dagger)."""
    rng = np.random.default_rng(rng)
    labels = list(counts)
    P = tomography_projectors(labels, r_sign)
    Pt = P.transpose(0, 2, 1).reshape(len(labels), 16)        # Tr(rho P) = sum rho_ab P_ba
    n = np.array([counts[k] for k in labels], dtype=float)
    S = n.sum()
    if S <= 0:
        raise ValueError("All counts are zero.")

    rows, cols = np.tril_indices(4)
    off = rows != cols

    def nll_and_grad(t):
        """Negative log-likelihood and its exact gradient (fast and precise convergence)."""
        T = np.zeros((4, 4), dtype=complex)
        T[rows, cols] = t[:10]
        T[rows[off], cols[off]] += 1j * t[10:]
        A = T @ T.conj().T
        a = np.real(np.trace(A))
        rho = A / a
        p = np.clip(np.real(Pt @ rho.reshape(16)), 1e-12, None)
        f = -(n @ np.log(p)) + S * np.log(p.sum())
        w = -n / p + S / p.sum()                                # d f / d p_i
        M = np.tensordot(w, P, axes=1)                          # sum_i w_i P_i
        K = (M - np.real(np.trace(M @ rho)) * np.eye(4)) / a    # d f / d A
        X = T.conj().T @ K                                      # d f = 2 Re Tr(X dT)
        g_re = 2 * np.real(X.T[rows, cols])
        g_im = -2 * np.imag(X.T[rows[off], cols[off]])
        return f, np.concatenate([g_re, g_im])

    # start 1: linear inversion (least squares), forced positive
    A = Pt
    x, *_ = np.linalg.lstsq(A, n / S * (np.trace(P.sum(0)).real / 4), rcond=None)
    lin = x.reshape(4, 4)
    lin = (lin + lin.conj().T) / 2
    starts = [_rho_to_t(lin / max(np.real(np.trace(lin)), 1e-9) if np.real(np.trace(lin)) > 0
                        else np.eye(4) / 4)]
    starts += [rng.normal(size=16) for _ in range(restarts)]
    opts = {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000}     # tight: same 4 decimals on every machine
    best = min((minimize(nll_and_grad, t0, jac=True, method="L-BFGS-B", options=opts) for t0 in starts),
               key=lambda r: r.fun)
    return _cholesky_rho(best.x)


def predicted_counts(rho: np.ndarray, labels, total: float, r_sign: int = +1) -> np.ndarray:
    """Expected counts for each setting when the overall rate is fitted to `total`."""
    P = tomography_projectors(labels, r_sign)
    p = np.real(np.einsum("ab,kba->k", rho, P))
    return total * p / p.sum()


def mle_tomography_2q(counts: dict, restarts: int = 3, rng=None) -> np.ndarray:
    """Cholesky-parametrized Poisson MLE over the 36-projector set (R=(H+iV)/sqrt2)."""
    return mle_tomography_counts(counts, r_sign=+1, restarts=restarts, rng=rng)


# ----------------------------------------------------------------- photonic elements and detectors
def waveplate(theta_deg: float, retardance_rad: float) -> np.ndarray:
    """Jones matrix of a wave plate with fast axis at theta (from H) and retardance delta:
    U = cos(d/2) I - i sin(d/2) (cos 2t Z + sin 2t X).  HWP: d = pi, QWP: d = pi/2.
    Checks: HWP(22.5) H -> D, HWP(45) H -> V, QWP(45) H -> (H - iV)/sqrt2 (up to global phase)."""
    t = np.deg2rad(theta_deg)
    return (np.cos(retardance_rad / 2) * I2
            - 1j * np.sin(retardance_rad / 2) * (np.cos(2 * t) * Z + np.sin(2 * t) * X))


def hwp(theta_deg: float) -> np.ndarray:
    return waveplate(theta_deg, np.pi)


def qwp(theta_deg: float) -> np.ndarray:
    return waveplate(theta_deg, np.pi / 2)


def phase_shifter_hwp_angle(phi_deg: float) -> float:
    """QWP(45) - HWP(theta) - QWP(45) equals diag(1, e^{i phi}) up to a global phase, with
    phi = 180 + 4 theta (mod 360). Returns theta in [0, 90) for the wanted relative phase phi."""
    return float(((phi_deg - 180.0) % 360.0) / 4.0)


def on_qubit(U1: np.ndarray, which: str) -> np.ndarray:
    return np.kron(U1, I2) if which == "A" else np.kron(I2, U1)


def accidental_rate(singles_a: float, singles_b: float, window_s: float) -> float:
    """Accidental coincidence rate R_acc = S_A * S_B * tau for a coincidence window of width tau."""
    return singles_a * singles_b * window_s


def simulate_experiment(rho, labels, pair_rate, time_s, eta_a, eta_b, dark_a, dark_b,
                        window_s, r_sign=+1, rng=None, noiseless=False):
    """Coincidence counts for projective settings with one detector behind each analyser.
    true coincidences: R * eta_a * eta_b * Tr(rho Pa x Pb) * T
    singles rate:      S_A = R * eta_a * Tr(rho Pa x I) + dark_a        (same for B)
    accidentals:       S_A * S_B * tau * T
    Counts are Poisson (noiseless=True returns the expected values instead). The accidental
    estimate uses the MEASURED singles, as in a real experiment: N_acc = N_A N_B tau / T.
    Returns a dict of per-label dicts: counts, true, accidental (expected), accidental_estimate."""
    rng = np.random.default_rng(rng)
    P1 = pauli_projectors_1q(r_sign)
    out = {"counts": {}, "true": {}, "accidental": {}, "accidental_estimate": {}}
    for k in labels:
        Pa, Pb = P1[k[0]], P1[k[1]]
        p_ab = max(float(np.real(np.trace(rho @ np.kron(Pa, Pb)))), 0.0)
        pa = float(np.real(np.trace(rho @ np.kron(Pa, I2))))
        pb = float(np.real(np.trace(rho @ np.kron(I2, Pb))))
        sa, sb = pair_rate * eta_a * pa + dark_a, pair_rate * eta_b * pb + dark_b
        true = pair_rate * eta_a * eta_b * p_ab * time_s
        acc = accidental_rate(sa, sb, window_s) * time_s
        if noiseless:
            n, na, nb = true + acc, sa * time_s, sb * time_s
        else:
            n, na, nb = int(rng.poisson(true + acc)), rng.poisson(sa * time_s), rng.poisson(sb * time_s)
        out["counts"][k] = n
        out["true"][k] = true
        out["accidental"][k] = acc
        out["accidental_estimate"][k] = float(na) * float(nb) * window_s / time_s
    return out


# ----------------------------------------------------------------- Stage 6
def _psd_sqrt(A: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity (squared convention): F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    s = _psd_sqrt(rho)
    ev = np.linalg.eigvalsh(s @ sigma @ s)
    return float(np.sum(np.sqrt(np.clip(ev, 0, None))) ** 2)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(rho - sigma))))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray, base: float = 2) -> float:
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-12]
    return float(-np.sum(ev * np.log(ev)) / np.log(base))


def mutual_information(rho: np.ndarray) -> float:
    rA = partial_trace(rho, [0], [2, 2])
    rB = partial_trace(rho, [1], [2, 2])
    return von_neumann_entropy(rA) + von_neumann_entropy(rB) - von_neumann_entropy(rho)


def coherence_l1(rho: np.ndarray) -> float:
    return float(np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho))))


def coherence_rel_entropy(rho: np.ndarray) -> float:
    return von_neumann_entropy(np.diag(np.diag(rho))) - von_neumann_entropy(rho)


def concurrence(rho: np.ndarray) -> float:
    yy = np.kron(Y, Y)
    rt = yy @ rho.conj() @ yy
    ev = np.sqrt(np.clip(np.real(np.linalg.eigvals(rho @ rt)), 0, None))
    ev = np.sort(ev)[::-1]
    # C <= 1 exactly; rounding can give 1 + 4e-16 (e.g. psi- after collective dephasing), which made
    # sqrt(1 - C^2) in entanglement_of_formation NaN.
    return float(min(1.0, max(0.0, ev[0] - ev[1] - ev[2] - ev[3])))


def entanglement_of_formation(rho: np.ndarray) -> float:
    C = concurrence(rho)
    x = (1 + np.sqrt(max(0.0, 1 - C ** 2))) / 2
    if x in (0.0, 1.0):
        return 0.0
    return float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


def negativity(rho: np.ndarray, dims=(2, 2)) -> float:
    ev = np.linalg.eigvalsh(partial_transpose(rho, 1, list(dims)))
    return float(np.sum(np.abs(ev[ev < 0])))


def log_negativity(rho: np.ndarray, dims=(2, 2)) -> float:
    return float(np.log2(2 * negativity(rho, dims) + 1))


def is_ppt(rho: np.ndarray, dims=(2, 2), tol=1e-10) -> bool:
    return bool(np.min(np.linalg.eigvalsh(partial_transpose(rho, 1, list(dims)))) > -tol)


def chsh_max(rho: np.ndarray) -> float:
    """Horodecki criterion: 2*sqrt(t1^2 + t2^2), t = two largest singular values of T."""
    s = np.linalg.svd(correlation_matrix(rho), compute_uv=False)
    return float(2 * np.sqrt(s[0] ** 2 + s[1] ** 2))


def cjwr_steering(rho: np.ndarray, n_settings: int = 3) -> float:
    """Linear steering inequality (Cavalcanti-Jones-Wiseman-Reid) with optimal measurement
    directions: F_n = (1/sqrt n) * (sum of the n largest singular values of T).
    F_n > 1 proves EPR steering (sufficient, not necessary); n = 2 or 3."""
    s = np.linalg.svd(correlation_matrix(rho), compute_uv=False)
    return float(np.sum(s[:n_settings]) / np.sqrt(n_settings))


def t_state_steering_value(rho: np.ndarray, n_theta: int = 200, n_phi: int = 400) -> float:
    """S = (1/2π) ∫ sqrt(nᵀ T Tᵀ n) dΩ over the unit sphere. For T-states (both reduced states
    maximally mixed, i.e. unpolarized photons) the state is EPR steerable with projective
    measurements iff S > 1 (Jevtic et al., JOSA B 32, A40 (2015); proven by Nguyen & Vu, EPL
    115, 10003 (2016)). Werner: S = 2p, so steerable iff p > 1/2. Gauss-Legendre in cos(θ)."""
    T = correlation_matrix(rho)
    M = T @ T.T
    x, w = np.polynomial.legendre.leggauss(n_theta)            # x = cos θ
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    st = np.sqrt(1 - x ** 2)
    n = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)), np.outer(x, np.ones(n_phi))])
    q = np.einsum("i...,ij,j...->...", n, M, n)
    integral = np.sum(w[:, None] * np.sqrt(np.clip(q, 0, None))) * (2 * np.pi / n_phi)
    return float(integral / (2 * np.pi))


def is_t_state(rho: np.ndarray, tol: float = 1e-6) -> bool:
    """Both single-photon Bloch vectors vanish (unpolarized photons)."""
    a = np.real([np.trace(rho @ np.kron(P, I2)) for P in (X, Y, Z)])
    b = np.real([np.trace(rho @ np.kron(I2, P)) for P in (X, Y, Z)])
    return bool(np.linalg.norm(a) < tol and np.linalg.norm(b) < tol)


def fidelity_witness(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W = I/2 - |e><e| with |e> the maximally entangled state closest to rho
    (Tr W sigma >= 0 for every separable sigma). Returns (W, |e>)."""
    rho_m = _MAGIC.conj().T @ rho @ _MAGIC
    v = np.linalg.eigh(np.real(rho_m + rho_m.conj().T) / 2)[1][:, -1]
    e = _MAGIC @ v
    return np.eye(4, dtype=complex) / 2 - dm(e), e


def ppt_witness(rho: np.ndarray) -> np.ndarray:
    """Optimal (PPT) witness W = (|e><e|)^{T_B}, |e> = eigenvector of the smallest eigenvalue of
    rho^{T_B}; Tr W sigma = <e|sigma^{T_B}|e> >= 0 for every separable sigma."""
    e = np.linalg.eigh(partial_transpose(rho, 1, [2, 2]))[1][:, 0]
    return partial_transpose(dm(e), 1, [2, 2])


_MAGIC = np.array([[1, 0, 0, 1], [1j, 0, 0, -1j], [0, 1j, 1j, 0], [0, 1, -1, 0]], dtype=complex).T / np.sqrt(2)


def fully_entangled_fraction_2q(rho: np.ndarray) -> float:
    """Fully entangled fraction f = max over maximally entangled |e> of <e|rho|e>.
    Exact for two qubits: in the magic basis (Hill-Wootters) the maximally entangled states are
    exactly the real unit vectors (up to a global phase), so f = largest eigenvalue of Re(rho_M)."""
    rho_m = _MAGIC.conj().T @ rho @ _MAGIC
    return float(np.linalg.eigvalsh(np.real(rho_m + rho_m.conj().T) / 2)[-1])


def fully_entangled_fraction_numeric(rho: np.ndarray) -> float:
    """Numerical cross-check of fully_entangled_fraction_2q (max over U x I applied to phi+)."""
    best = 0.0
    for b in ("phi+", "phi-", "psi+", "psi-"):
        best = max(best, np.real(bell_state(b).conj() @ rho @ bell_state(b)))

    def neg_f(x):
        U = expm(-1j * (x[0] * X + x[1] * Y + x[2] * Z))
        r = np.kron(U, I2) @ rho @ np.kron(U, I2).conj().T
        return -np.real(bell_state("phi+").conj() @ r @ bell_state("phi+"))
    for x0 in np.random.default_rng(0).normal(size=(8, 3)):
        best = max(best, -minimize(neg_f, x0).fun)
    return float(best)


def basis_qber(rho: np.ndarray, basis: str) -> tuple[float, str]:
    """Error rate when both photons are measured in the same basis ('Z' = H/V, 'X' = D/A,
    'Y' = R/L). Bob may flip all his bits, so QBER = min(P(equal), P(different)).
    Returns (qber, 'correlated' or 'anti-correlated')."""
    P = pauli_projectors_1q()
    a, b = {"Z": ("H", "V"), "X": ("D", "A"), "Y": ("R", "L")}[basis]
    p = lambda x, y: float(np.real(np.trace(rho @ np.kron(P[x], P[y]))))   # noqa: E731
    same = p(a, a) + p(b, b)
    diff = p(a, b) + p(b, a)
    return (diff, "correlated") if diff <= same else (same, "anti-correlated")


def entanglement_key_rate(e_bit: float, e_phase: float) -> float:
    """Asymptotic key fraction of entanglement-based BB84 (BBM92) with one-way error correction
    at the Shannon limit: r = 1 - h(e_bit) - h(e_phase) (Shor-Preskill / Koashi), floored at 0."""
    return max(0.0, 1 - binary_entropy(e_bit) - binary_entropy(e_phase))


def teleportation_fidelity(rho: np.ndarray) -> float:
    f = fully_entangled_fraction_2q(rho)
    return (2 * f + 1) / 3


def quantum_discord(rho: np.ndarray, n_starts: int = 6) -> float:
    """Discord with projective measurement on B (qubit 1). Numerical Bloch-sphere optimization."""
    rA = partial_trace(rho, [0], [2, 2])
    rB = partial_trace(rho, [1], [2, 2])
    I_tot = von_neumann_entropy(rA) + von_neumann_entropy(rB) - von_neumann_entropy(rho)

    def cond_entropy(angles):
        th, ph = angles
        n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        S = 0.0
        for sgn in (+1, -1):
            Pi = 0.5 * (I2 + sgn * (n[0] * X + n[1] * Y + n[2] * Z))
            M = np.kron(I2, Pi)
            pk = np.real(np.trace(M @ rho))
            if pk > 1e-12:
                rA_k = partial_trace(M @ rho @ M, [0], [2, 2]) / pk
                S += pk * von_neumann_entropy(rA_k)
        return S

    rng = np.random.default_rng(1)
    best = min(minimize(cond_entropy, [rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)],
                        method="Nelder-Mead").fun for _ in range(n_starts))
    J = von_neumann_entropy(rA) - best
    return float(max(0.0, I_tot - J))


def three_tangle(psi: np.ndarray) -> float:
    """Coffman-Kundu-Wootters 3-tangle for a pure 3-qubit state (big-endian)."""
    a = np.asarray(psi).reshape(2, 2, 2)
    d1 = (a[0,0,0]**2*a[1,1,1]**2 + a[0,0,1]**2*a[1,1,0]**2 + a[0,1,0]**2*a[1,0,1]**2 + a[1,0,0]**2*a[0,1,1]**2)
    d2 = (a[0,0,0]*a[1,1,1]*a[0,1,1]*a[1,0,0] + a[0,0,0]*a[1,1,1]*a[1,0,1]*a[0,1,0]
          + a[0,0,0]*a[1,1,1]*a[1,1,0]*a[0,0,1] + a[0,1,1]*a[1,0,0]*a[1,0,1]*a[0,1,0]
          + a[0,1,1]*a[1,0,0]*a[1,1,0]*a[0,0,1] + a[1,0,1]*a[0,1,0]*a[1,1,0]*a[0,0,1])
    d3 = (a[0,0,0]*a[1,1,0]*a[1,0,1]*a[0,1,1] + a[1,1,1]*a[0,0,1]*a[0,1,0]*a[1,0,0])
    return float(4 * abs(d1 - 2 * d2 + 4 * d3))


# ----------------------------------------------------------------- Stage 8/9
def bitflip_code_logical_error(p: float) -> float:
    return 3 * p ** 2 - 2 * p ** 3


def binary_entropy(q: float) -> float:
    if q <= 0 or q >= 1:
        return 0.0
    return float(-q * np.log2(q) - (1 - q) * np.log2(1 - q))


def bb84_key_rate(qber: float) -> float:
    """Asymptotic Shor-Preskill rate r = 1 - 2 h(Q)."""
    return 1 - 2 * binary_entropy(qber)


def qfi_pure(psi: np.ndarray, G: np.ndarray) -> float:
    """QFI of |psi(theta)> = exp(-i theta G)|psi>: 4 Var(G)."""
    psi = psi / np.linalg.norm(psi)
    m1 = np.real(psi.conj() @ G @ psi)
    m2 = np.real(psi.conj() @ G @ G @ psi)
    return float(4 * (m2 - m1 ** 2))


def tmsv_log_negativity(r: float) -> float:
    """Two-mode squeezed vacuum: E_N = 2 r / ln 2 (bits)."""
    return 2 * r / np.log(2)


# ----------------------------------------------------------------- Stage 10: one photon
# A single polarization qubit, basis |0> = H, |1> = V. Stokes parameters (normalized, S0 = 1):
#   S1 = P(H) - P(V) = <σz>,  S2 = P(D) - P(A) = <σx>,  S3 = P(R) - P(L) = <σy>   (R = (H + iV)/√2)
# so rho = (I + S1 σz + S2 σx + S3 σy)/2; the degree of polarization is |S| (1 = pure, 0 = unpolarized).
def stokes_vector(rho: np.ndarray) -> np.ndarray:
    """(S1, S2, S3) of a one-photon density matrix."""
    return np.real([np.trace(rho @ Z), np.trace(rho @ X), np.trace(rho @ Y)])


def rho_from_stokes(s1: float, s2: float, s3: float) -> np.ndarray:
    s = np.array([s1, s2, s3], dtype=float)
    if np.linalg.norm(s) > 1 + 1e-9:
        raise ValueError(f"|S| = {np.linalg.norm(s):.4f} > 1: not a physical polarization state.")
    return (I2 + s[0] * Z + s[1] * X + s[2] * Y) / 2


def stokes_from_ellipse(azimuth_deg: float, ellipticity_deg: float, dop: float = 1.0) -> np.ndarray:
    """Stokes vector of light with polarization-ellipse azimuth ψ (major axis from H) and ellipticity
    angle χ (tan χ = minor/major axis; χ > 0 = right-handed, i.e. towards R), degree of polarization dop:
    S = dop (cos 2χ cos 2ψ, cos 2χ sin 2ψ, sin 2χ)."""
    if not 0 <= dop <= 1:
        raise ValueError("The degree of polarization must be in [0, 1].")
    if abs(ellipticity_deg) > 45 + 1e-9:
        raise ValueError("The ellipticity angle must be between -45° and +45°.")
    p, c = np.deg2rad(2 * azimuth_deg), np.deg2rad(2 * ellipticity_deg)
    return dop * np.array([np.cos(c) * np.cos(p), np.cos(c) * np.sin(p), np.sin(c)])


def polarization_ellipse(s: np.ndarray) -> dict | None:
    """Azimuth ψ in (-90°, 90°] and ellipticity angle χ in [-45°, 45°] of the polarized part
    (None for unpolarized light, where the ellipse is undefined; azimuth None for circular light)."""
    n = float(np.linalg.norm(s))
    if n < 1e-9:
        return None
    chi = 0.5 * np.degrees(np.arcsin(np.clip(s[2] / n, -1, 1)))
    if np.hypot(s[0], s[1]) < 1e-6 * n:                  # circular: the azimuth is undefined
        return {"azimuth_deg": None, "ellipticity_deg": float(chi)}
    psi = 0.5 * np.degrees(np.arctan2(s[1], s[0]))
    if psi <= -90 + 1e-9:
        psi += 180
    return {"azimuth_deg": float(psi), "ellipticity_deg": float(chi)}


def mle_tomography_1q(counts: dict, r_sign: int = +1) -> np.ndarray:
    """Poisson maximum-likelihood one-photon tomography from counts behind H/V/D/A/R/L analysers
    (any informationally complete subset, e.g. H, V, D, R or all six). Same assumptions as the
    two-photon fit: equal flux and counting time per setting, the overall rate is fitted.
    Physicality: rho = (I + r·σ)/2 with |r| = tanh(|u|) < 1."""
    P1 = pauli_projectors_1q(r_sign)
    labels = [k for k in counts]
    bad = [k for k in labels if k not in P1]
    if bad:
        raise ValueError(f"Unknown settings {bad}: use H, V, D, A, R, L.")
    n = np.array([float(counts[k]) for k in labels])
    if n.sum() <= 0:
        raise ValueError("All counts are zero.")
    P = np.array([P1[k] for k in labels])
    pick = lambda axis: any(k in labels for k in axis)          # noqa: E731
    if not (pick("HV") and pick("DA") and pick("RL")):
        raise ValueError("Informationally incomplete: need at least one of H/V, one of D/A and one of R/L "
                         "(plus a second setting of one pair to fix the overall rate), e.g. H, V, D, R.")
    S = n.sum()

    def rho_of(u):
        r = np.asarray(u, float)
        m = np.linalg.norm(r)
        r = r * (np.tanh(m) / m) if m > 1e-12 else r
        return (I2 + r[0] * Z + r[1] * X + r[2] * Y) / 2

    def nll(u):
        p = np.clip(np.real(np.einsum("ab,kba->k", rho_of(u), P)), 1e-12, None)
        return -(n @ np.log(p)) + S * np.log(p.sum())

    # start from the linear estimate (each available pair; one setting + the fitted total otherwise)
    n_of = {k: float(counts[k]) for k in labels}
    norm = n_of["H"] + n_of["V"] if "H" in n_of and "V" in n_of else S / len(labels) * 2
    r0 = []
    for a, b in ("HV", "DA", "RL"):
        if a in n_of and b in n_of and n_of[a] + n_of[b] > 0:
            r0.append((n_of[a] - n_of[b]) / (n_of[a] + n_of[b]))
        elif a in n_of:
            r0.append(2 * n_of[a] / norm - 1)
        else:
            r0.append(1 - 2 * n_of[b] / norm)
    r0 = np.asarray(r0, float)                     # (S1, S2, S3) = coefficients of (σz, σx, σy)
    m0 = np.linalg.norm(r0)
    if m0 > 0.95:
        r0 = r0 * 0.95 / m0
        m0 = 0.95
    u0 = r0 * (np.arctanh(m0) / m0) if m0 > 1e-9 else r0
    opts = {"ftol": 1e-15, "gtol": 1e-11, "maxiter": 2000}
    best = min((minimize(nll, u, method="L-BFGS-B", options=opts) for u in (u0, np.zeros(3) + 1e-3)),
               key=lambda r: r.fun)
    return rho_of(best.x)


# ----------------------------------------------------------------- Stage 10: global (two-photon) channels
# Kraus operators are 4x4 and act on the PAIR, not on each photon separately.
N_V = np.array([0, 1, 1, 2])                        # number of V photons in HH, HV, VH, VV
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)          # |H><V|: V decays to H
COLLECTIVE_LOWERING = np.kron(SIGMA_MINUS, I2) + np.kron(I2, SIGMA_MINUS)


def kraus_schur(M: np.ndarray) -> list[np.ndarray]:
    """Kraus operators of the 'Schur multiplier' channel rho_ij -> M_ij rho_ij (M positive
    semidefinite with unit diagonal): M = sum_k l_k v_k v_k^dagger gives K_k = sqrt(l_k) diag(v_k)."""
    w, V = np.linalg.eigh((M + M.conj().T) / 2)
    return [np.sqrt(l) * np.diag(V[:, k]) for k, l in enumerate(w) if l > 1e-14]


def kraus_global_depolarizing(p: float) -> list[np.ndarray]:
    """rho -> (1 - p) rho + p I/4 (the pair is replaced by white noise with probability p)."""
    ops = [np.kron(a, b) for a in PAULI for b in PAULI]
    return [np.sqrt(1 - 15 * p / 16) * ops[0]] + [np.sqrt(p / 16) * P for P in ops[1:]]


def kraus_global_dephasing(p: float) -> list[np.ndarray]:
    """Every off-diagonal element in the HH, HV, VH, VV basis is multiplied by (1 - p)."""
    return kraus_schur((1 - p) * np.ones((4, 4)) + p * np.eye(4))


def kraus_collective_dephasing(p: float) -> list[np.ndarray]:
    """Both photons get the SAME random phase diag(1, e^(iφ)) (Gaussian φ, e^(-<φ²>/2) = 1 - p):
    rho_ij -> rho_ij (1 - p)^((n_i - n_j)²) with n = number of V photons. HV <-> VH coherences
    (psi+ and psi-) are untouched: a decoherence-free subspace."""
    d = N_V[:, None] - N_V[None, :]
    return kraus_schur((1.0 - p) ** (d ** 2))


def kraus_correlated_pauli(P: np.ndarray, p: float) -> list[np.ndarray]:
    """With probability p BOTH photons get the same Pauli P (P ⊗ P); otherwise nothing."""
    return [np.sqrt(1 - p) * np.eye(4, dtype=complex), np.sqrt(p) * np.kron(P, P)]


def kraus_from_superoperator(S: np.ndarray, d: int) -> list[np.ndarray]:
    """Kraus operators of the channel whose superoperator (row-stacking, as liouvillian()) is S:
    Choi matrix J = sum_ij |i><j| (x) E(|i><j|), then J = sum_k |v_k><v_k|, K_k = v_k as d x d, transposed."""
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d), dtype=complex)
            E[i, j] = 1
            J += np.kron(E, (S @ E.reshape(-1)).reshape(d, d))
    w, V = np.linalg.eigh((J + J.conj().T) / 2)
    return [np.sqrt(l) * V[:, k].reshape(d, d).T for k, l in enumerate(w) if l > 1e-12]


def kraus_collective_amplitude_damping(p: float) -> list[np.ndarray]:
    """Both photons decay V -> H through ONE common channel (jump operator J = σ- ⊗ I + I ⊗ σ-,
    rate γ), stopped at γt = -ln(1 - p), so p = 1 - e^(-γt). psi- = (HV - VH)/√2 is dark
    (J psi- = 0) and never decays; psi+ decays at rate 2γ. p = 1 is the long-time limit."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1].")
    if p == 0:
        return [np.eye(4, dtype=complex)]
    gt = -np.log1p(-p) if p < 1 else 80.0             # e^(-80): the long-time limit to machine precision
    S = expm(liouvillian(np.zeros((4, 4), complex), [COLLECTIVE_LOWERING], [1.0]) * gt)
    return kraus_from_superoperator(S, 4)


def completeness_error(kraus: list[np.ndarray]) -> float:
    """max |sum K^dagger K - I| (0 for a trace-preserving channel)."""
    d = kraus[0].shape[1]
    return float(np.max(np.abs(sum(K.conj().T @ K for K in kraus) - np.eye(d))))
