"""
verify_features.py - check the features of 2026-10-31 WITHOUT a language model (free, about 20 s):
one-photon states, the density matrix as numbers, global (4x4 Kraus) noise, the user's own Kraus
operators, the new Lindblad processes and the entropies. Every line compares what the tools compute
with an analytic value and prints PASS or FAIL.
    python verify_features.py            (copy the whole output into the chat)
"""
import math
import re
import time

import numpy as np

import agent_tools as at
import display
import qi_tools as qt

RESULTS = []


def h(text: str) -> str:
    return re.search(r"rho_\d+", text).group(0)


def check(name: str, got, expected, tol: float = 1e-4):
    if isinstance(expected, str):
        ok = expected in str(got)
        shown_got, shown_exp = str(got), f"contains '{expected}'"
    else:
        g, e = np.atleast_1d(np.asarray(got, float)), np.atleast_1d(np.asarray(expected, float))
        ok = g.shape == e.shape and bool(np.all(np.abs(g - e) <= tol))
        fmt = lambda a: ", ".join(f"{v:.6g}" for v in a)        # noqa: E731
        shown_got, shown_exp = fmt(g), fmt(e)
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {len(RESULTS):02d} {name}\n         got: {shown_got}\n    expected: {shown_exp}")


def stokes(handle: str):
    m = at.compute_measures.invoke({"handle": handle})
    return [m["stokes"]["S1 (H-V)"], m["stokes"]["S2 (D-A)"], m["stokes"]["S3 (R-L)"]], m


def C(handle: str) -> float:
    return at.compute_measures.invoke({"handle": handle})["concurrence"]


def main():
    t0 = time.time()
    at.STORE.clear(); at.DESC.clear(); at.TOMO.clear()
    print("=== One photon ===")
    s, m = stokes(h(at.create_single_photon_state.invoke({"polarization": "H"})))
    check("H photon: Stokes (S1, S2, S3), degree of polarization, purity, entropy",
          s + [m["degree_of_polarization"], m["purity"], m["entropy"]], [1, 0, 0, 1, 1, 0])
    s, m = stokes(h(at.create_single_photon_state.invoke({"azimuth_deg": 30, "dop": 0.8})))
    check("linear 30°, dop 0.8: S = 0.8(cos60°, sin60°, 0); purity (1+0.64)/2; entropy h(0.1)",
          s + [m["purity"], m["entropy"]], [0.4, 0.8 * math.sqrt(3) / 2, 0, 0.82, qt.binary_entropy(0.1)])
    s, m = stokes(h(at.create_single_photon_state.invoke({"amp_H": "1", "amp_V": "i"})))
    check("(|H> + i|V>)/√2 is R: S3 = 1, ellipticity 45°", s + [m["ellipse"]["ellipticity_deg"]], [0, 0, 1, 45])
    check("   ... and its azimuth is undefined (circular)", m["ellipse"]["azimuth_deg"], "undefined")
    d = h(at.create_single_photon_state.invoke({"polarization": "D"}))
    r = at.apply_waveplate.invoke({"handle": d, "plate": "QWP", "angle_deg": 0, "qubit": "A"})
    check("D through a QWP with fast axis at 0° becomes R (S3 = 1), degree of polarization unchanged",
          [r["after"]["S3 (R-L)"], r["after"]["degree_of_polarization"]], [1, 1])
    ef = at.evolve_state.invoke({"handle": d, "process": "dephasing", "gamma": 1, "t_max": 2})["features"]
    e = ef["at_t_max"]
    check("D under dephasing γ = 1 to t = 2: degree of polarization e^-2, purity (1 + e^-4)/2, halves at ln 2",
          [e["degree_of_polarization"], e["purity"], ef["degree_of_polarization_halves_at"]],
          [math.exp(-2), (1 + math.exp(-4)) / 2, math.log(2)])
    r = at.reconstruct_single_photon.invoke({"counts_H": 980, "counts_V": 20, "counts_D": 510, "counts_A": 490,
                                             "counts_R": 500, "counts_L": 500})
    got = [float(r["stokes_mle"][k].split(" ±")[0]) for k in ("S1 (H-V)", "S2 (D-A)", "S3 (R-L)")]
    check("tomography from counts H 980, V 20, D 510, A 490, R 500, L 500: S = (0.96, 0.02, 0)", got,
          [0.96, 0.02, 0.0])
    check("   ... degree of polarization √(0.96² + 0.02²)", float(r["degree_of_polarization"].split(" ±")[0]),
          math.hypot(0.96, 0.02))

    print("\n=== Photon pairs and single photons together ===")
    phi = h(at.create_state.invoke({"kind": "phi+"}))
    a = h(at.reduced_state.invoke({"handle": phi, "photon": "A"}))
    s, m = stokes(a)
    check("photon A of phi+ is unpolarized: degree of polarization 0, entropy 1 bit",
          [m["degree_of_polarization"], m["entropy"]], [0, 1])
    hh = h(at.create_single_photon_state.invoke({"polarization": "H"}))
    pair = h(at.combine_photons.invoke({"handle_a": hh, "handle_b": d}))
    rows = at.show_density_matrix.invoke({"handle": pair})["matrix"]
    check("H ⊗ D = (|HH> + |HV>)/√2: rho[HH,HH], rho[HH,HV], rho[HV,HV], rho[VV,VV]",
          [float(rows[0][0]), float(rows[0][1]), float(rows[1][1]), float(rows[3][3] if rows[3][3] != "0" else 0)],
          [0.5, 0.5, 0.5, 0])
    check("   ... a product state has concurrence 0", C(pair), 0)
    w = h(at.create_state.invoke({"kind": "werner", "p": 0.8}))
    m = at.compute_measures.invoke({"handle": w})
    S = -(0.85 * math.log2(0.85) + 3 * 0.05 * math.log2(0.05))
    check("Werner p = 0.8: S(rho), S(A), S(B), S(A|B) = S(rho) - S(B)",
          [m["entropy"], m["entropy_A"], m["entropy_B"], m["conditional_entropy_A_given_B"]], [S, 1, 1, S - 1])

    print("\n=== The density matrix as numbers ===")
    ad = h(at.apply_local_noise.invoke({"handle": phi, "channel": "amplitude_damping", "param": 0.3}))
    r = at.show_density_matrix.invoke({"handle": ad, "compare_with": phi})
    m4 = [float(r["matrix"][i][j]) for i, j in ((0, 0), (1, 1), (3, 3), (0, 3))]
    check("phi+ after amplitude damping 0.3 on both: HH,HH (1+γ²)/2, HV,HV γ(1-γ)/2, VV,VV (1-γ)²/2, HH,VV (1-γ)/2",
          m4, [0.545, 0.105, 0.245, 0.35])
    check("   ... change Δρ of HH,VV and VV,VV", [float(r["difference"][0][3]), float(r["difference"][3][3])],
          [-0.15, -0.255])
    rec = at.reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})["handle"]
    r = at.show_density_matrix.invoke({"handle": rec})
    check("reconstructed sample data: every element has an error bar", "element_errors_1sigma" in r, 1)

    print("\n=== Global noise (4x4 Kraus operators on the pair) ===")
    errs = [qt.completeness_error(at._channel_kraus(c, 0.37, "both")) for c in at.GLOBAL_CHANNELS]
    check(f"all {len(errs)} global channels are complete: max |Σ K†K - I|", max(errs), 0, tol=1e-12)
    g = h(at.apply_global_noise.invoke({"handle": phi, "channel": "global_depolarizing", "param": 0.3}))
    loc = h(at.apply_local_noise.invoke({"handle": phi, "channel": "depolarizing", "param": 0.3}))
    check("phi+, p = 0.3: concurrence after GLOBAL depolarizing (Werner 0.7) and after LOCAL on both (Werner 0.49)",
          [C(g), C(loc)], [0.55, 0.235])
    psi = h(at.create_state.invoke({"kind": "psi-"}))
    a1 = h(at.apply_global_noise.invoke({"handle": phi, "channel": "collective_dephasing", "param": 0.5}))
    a2 = h(at.apply_global_noise.invoke({"handle": psi, "channel": "collective_dephasing", "param": 0.5}))
    check("collective dephasing p = 0.5: C(phi+) = 0.5^4, C(psi-) = 1 (decoherence-free)", [C(a1), C(a2)],
          [0.0625, 1])
    b1 = h(at.apply_global_noise.invoke({"handle": psi, "channel": "collective_amplitude_damping", "param": 0.9}))
    check("collective amplitude damping p = 0.9: psi- unchanged (dark)", np.abs(at.STORE[b1] - at.STORE[psi]).max(), 0,
          tol=1e-9)
    c1 = h(at.apply_global_noise.invoke({"handle": phi, "channel": "correlated_bit_flip", "param": 0.4}))
    check("correlated X⊗X flip p = 0.4 leaves phi+ unchanged", np.abs(at.STORE[c1] - at.STORE[phi]).max(), 0,
          tol=1e-12)
    k = h(at.apply_kraus_operators.invoke({"handle": phi, "qubits": "B",
                                           "kraus": "[[[1, 0], [0, sqrt(0.6)]], [[0, sqrt(0.4)], [0, 0]]]"}))
    check("your own Kraus operators (amplitude damping 0.4 on photon B): C = √0.6", C(k), math.sqrt(0.6))
    try:
        at.apply_kraus_operators.invoke({"handle": phi, "kraus": [[[1, 0], [0, 1]], [[0, 1], [0, 0]]]})
        check("incomplete Kraus operators are rejected", "accepted", "rejected")
    except Exception as ex:                                       # noqa: BLE001
        check("incomplete Kraus operators are rejected", str(ex)[:40], "Not a valid")
    r = at.get_kraus_operators.invoke({"channel": "collective_dephasing", "param": 0.3})
    check("Kraus operators of collective dephasing 0.3: 4x4, trace preserving",
          [len(r["kraus_operators"]["K0"]), r["trace_preserving"]], [4, 1])

    print("\n=== Time evolution: new processes ===")
    bad = []
    for proc, (tool_name, channel, _, f) in at._TWIN.items():
        fin = at.evolve_state.invoke({"handle": phi, "process": proc, "gamma": 1.3, "t_max": 0.7})["final_state"]
        tool = at.apply_local_noise if tool_name == "apply_local_noise" else at.apply_global_noise
        twin = h(tool.invoke({"handle": phi, "channel": channel, "param": float(f(0.91))}))
        bad.append(float(np.abs(at.STORE[fin] - at.STORE[twin]).max()))
    check(f"all {len(bad)} Lindblad processes equal their Kraus channel at t_max: largest difference", max(bad), 0,
          tol=1e-9)
    pp = h(at.create_state.invoke({"kind": "psi+"}))
    e1 = at.evolve_state.invoke({"handle": psi, "process": "collective_decay", "gamma": 1, "t_max": 2})
    e2 = at.evolve_state.invoke({"handle": pp, "process": "collective_decay", "gamma": 1, "t_max": 2})
    check("collective decay γ = 1, t = 2: C(psi-) stays 1, C(psi+) = e^-4",
          [e1["features"]["at_t_max"]["concurrence"], e2["features"]["at_t_max"]["concurrence"]], [1, math.exp(-4)])
    f = at.evolve_state.invoke({"handle": phi, "process": "collective_dephasing", "gamma": 1, "t_max": 2,
                                "plot_elements": True})
    check("collective dephasing γ = 1, t = 2: C(phi+) = e^-8 (local dephasing would give e^-4)",
          qt.concurrence(at.STORE[f["final_state"]]), math.exp(-8), tol=1e-8)
    check("   ... and the plot of every density-matrix element was saved", f.get("elements_figure", ""), "elements")
    f = at.evolve_state.invoke({"handle": phi, "process": "collective_dephasing", "gamma": 1, "t_max": 10})
    check("asymptotic decay is not called sudden death", f["features"]["entanglement_vanishes_at"], "never")

    print("\n=== Noise curve of a STORED state (Workbench 'Plot versus the parameter') ===")
    s = at.plot_sweep.invoke({"sweep": "noise", "handles": [phi], "channel": "depolarizing",
                              "quantities": ["concurrence", "chsh_max"]})["results"][phi]["measures"]
    check("local depolarizing on phi+: C = 0 at p = 1 - 1/√3, CHSH = 2 at p = 1 - 2^(-1/4)",
          [s["concurrence"]["crossings"][0]["at"], s["chsh_max"]["crossings"][0]["at"]],
          [1 - 1 / math.sqrt(3), 1 - 2 ** -0.25])
    s = at.plot_sweep.invoke({"sweep": "noise", "handles": [phi], "channel": "global_depolarizing",
                              "quantities": ["concurrence"]})["results"][phi]["measures"]
    check("global depolarizing on phi+: C = 0 at p = 2/3", s["concurrence"]["crossings"][0]["at"], 2 / 3)
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks passed ({time.time() - t0:.0f} s).")
    print(display.format_density_matrix(at.show_density_matrix.invoke({"handle": ad, "compare_with": phi})))


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0 if RESULTS and all(RESULTS) else 1)      # a failed check fails the GitHub test run
