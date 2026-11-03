"""
capabilities.py - what the agent can do, in ONE place.

The web app's capabilities page, the agent's answer to "what can you do?" and the list_capabilities
tool are all built from AREAS below. test_capabilities.py checks that every tool belongs to an area
and that every example question really reaches the area's tools, so this list cannot drift away
from the code.
    python capabilities.py        prints the page as Markdown
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Area:
    key: str
    title: str
    summary: str
    computes: tuple
    examples: tuple
    tools: tuple
    notes: tuple = field(default=())


AREAS = (
    Area("states", "Create two-photon states",
         "Bell states, Werner states, any pure state you write down, and mixtures of them.",
         ("phi+, phi-, psi+, psi-; Werner p|Bell⟩⟨Bell| + (1−p)I/4 on any Bell state",
          "pure states from amplitudes, e.g. cos(0.3)|HH⟩ + sin(0.3)|VV⟩ or (|HH⟩ + i|VV⟩)/√2",
          "mixtures such as 0.8|phi+⟩⟨phi+| + 0.2|psi+⟩⟨psi+|"),
         ("Create Werner states with p = 0.3, 0.5 and 0.8. Which of them are entangled and which violate the "
          "CHSH inequality?",
          "Create the state cos(0.3)|HH> + sin(0.3)|VV> and compute its concurrence, negativity and CHSH maximum.",
          "Create the mixture 0.8|phi+><phi+| + 0.2|psi+><psi+| and give its concurrence, purity and fidelity "
          "with phi+."),
         ("create_state", "create_pure_state", "mix_states")),
    Area("photons", "Single photons (one polarization qubit)",
         "One photon's polarization: create it, measure it, send it through wave plates and noise, or take one "
         "photon out of a pair.",
         ("H, V, D, A, R, L, any polarization ellipse (azimuth, ellipticity), Stokes parameters, partly polarized "
          "light (degree of polarization)",
          "Stokes parameters, degree of polarization, purity, entropy, the polarization ellipse; Poincaré sphere",
          "the reduced state of photon A or B of a pair; the product of two single photons; tomography from "
          "H/V/D/A/R/L counts with error bars"),
         ("Create a horizontally polarized single photon and a single photon with linear polarization at 30 "
          "degrees and degree of polarization 0.8. Give their Stokes parameters, degree of polarization, purity "
          "and entropy.",
          "Create phi+ and give the reduced state of photon A. Is photon A polarized?",
          "Reconstruct the polarization of a single photon from these counts: H = 980, V = 20, D = 510, A = 490, "
          "R = 500, L = 500. What are the Stokes parameters and the degree of polarization, with error bars?"),
         ("create_single_photon_state", "reduced_state", "combine_photons", "reconstruct_single_photon"),
         ("Wave plates, noise, time evolution, density matrices and the Bloch (Poincaré) sphere work for one "
          "photon too.",)),
    Area("measures", "Entanglement and correlation measures",
         "Every standard two-qubit measure of a state, and how close two states are.",
         ("purity, von Neumann entropy (of the pair and of each photon), conditional entropy S(A|B), concurrence, "
          "entanglement of formation",
          "negativity and log-negativity, CHSH maximum (Horodecki), l1 and relative-entropy coherence",
          "mutual information and quantum discord; fidelity and trace distance between two states",
          "the density matrix as numbers (4x4 or 2x2), with the change of every element between two states"),
         ("Create the state (|HH> + i|VV>)/sqrt(2) and compute all its measures.",
          "Create phi+ and a Werner state with p = 0.8 built on phi+. What is the fidelity between them?",
          "Create a Werner state with p = 0.8. Give its von Neumann entropy, the entropies of the two photons and "
          "the conditional entropy S(A|B).",
          "Create phi+, apply local amplitude damping with gamma = 0.3 to both photons, and show the density "
          "matrix before and after with the change of every element."),
         ("compute_measures", "compare_states", "show_density_matrix")),
    Area("noise", "Noise channels",
         "Local noise on one or both photons, global noise on the pair (4x4 Kraus operators), or your own "
         "Kraus operators, with the exact parameter where entanglement or the CHSH violation is lost.",
         ("local: phase damping, amplitude damping, depolarizing, bit flip, phase flip, bit-phase flip, on "
          "photon A, photon B or both",
          "global: two-photon depolarizing, dephasing, collective dephasing (common phase; psi± immune), "
          "collective amplitude damping (psi- dark), correlated X⊗X, Y⊗Y, Z⊗Z flips",
          "the Kraus operators of any channel with the completeness check; your own 2x2 or 4x4 Kraus operators",
          "where entanglement vanishes and where CHSH drops to 2, with a plot"),
         ("Create phi+, apply phase damping with lambda = 0.7 to both qubits, and give the concurrence, "
          "negativity, purity, l1 coherence and CHSH maximum.",
          "Plot how the measures of phi+ change under local depolarizing noise on both qubits. At which p does "
          "the entanglement vanish and where does CHSH drop to 2?",
          "Apply global depolarizing noise with p = 0.3 to phi+ and compare it with local depolarizing noise with "
          "p = 0.3 on both photons. Which destroys more entanglement?",
          "Show the Kraus operators of collective dephasing with p = 0.3 and check that they are complete."),
         ("apply_local_noise", "plot_measures_vs_noise", "apply_global_noise", "get_kraus_operators",
          "apply_kraus_operators")),
    Area("evolution", "Time evolution",
         "Continuous-time decoherence (Lindblad master equation) and local Hamiltonians.",
         ("local: dephasing (T2), amplitude decay (T1), depolarizing, bit flip, bit-phase flip, at a rate γ, on "
          "one or both photons",
          "global: two-photon depolarizing and dephasing, collective dephasing, collective decay "
          "(superradiance; psi- dark), correlated flips; every density-matrix element versus time",
          "local Hamiltonians such as (ω/2)σx; entanglement sudden death; half-life of the concurrence",
          "exact values at the key times (CHSH = 2, C halves, C = 0)"),
         ("Let phi+ evolve under local dephasing with rate gamma = 1 on both qubits up to t = 2. When has the "
          "concurrence dropped to half its initial value? Is there entanglement sudden death?",
          "Create 1/sqrt(3)|HH> + sqrt(2/3)|VV> and let it evolve under amplitude decay (T1, rate gamma = 1) on "
          "both qubits up to t = 3. At what time does the entanglement die?",
          "Create psi- and psi+ and let both evolve under collective decay with gamma = 1 up to t = 2. Which one "
          "keeps its entanglement?"),
         ("evolve_state",)),
    Area("waveplates", "Wave plates and phase compensation",
         "Half- and quarter-wave plates at any angle, and the plate settings that remove a phase error.",
         ("HWP and QWP on photon A or B (Jones matrices); which Bell state you get",
          "the QWP(45°)–HWP(θ)–QWP(45°) phase shifter that turns a state into phi+",
          "fidelity before and after; both R/L sign conventions"),
         ("Create phi+ and put a half-wave plate at 45 degrees on photon B. Which Bell state do you get, and does "
          "the concurrence change?",
          "Reconstruct data/state_tomo_data.csv and find the wave plates on photon B that turn it into phi+. "
          "What is the HWP angle, and what is the fidelity with phi+ before and after?"),
         ("apply_waveplate", "compensate_phase")),
    Area("tomography", "Tomography of your measured data",
         "Upload coincidence counts; get the density matrix by maximum likelihood, with error bars and a "
         "check of the fit.",
         ("maximum-likelihood reconstruction from 16 or 36 settings (H, V, D, A, R, L on each photon)",
          "bootstrap error bars, χ²/dof, p-value, and realistic (scaled) error bars when the fit is poor",
          "which settings deviate most; the closest maximally entangled state and its phase"),
         ("Reconstruct the density matrix from data/state_tomo_data.csv and tell me how entangled it is, with "
          "error bars.",
          "Reconstruct data/state_tomo_data.csv and plot how well the fit matches the measured counts. Which "
          "settings deviate most?"),
         ("reconstruct_state_from_counts", "get_tomography_report", "plot_tomography_fit"),
         ("CSV format: one row per setting, a column of labels such as HH, HV, …, RL (first letter = photon A) "
          "and one or more columns of counts. At least the 16 standard settings; all 36 are best. Max 1 MB.",)),
    Area("detectors", "Simulated experiments with real detectors",
         "A tomography experiment with finite efficiency, dark counts, a coincidence window and counting "
         "time, and what accidentals do to the result.",
         ("pair rate, detection efficiencies, dark counts, coincidence window, time per setting",
          "raw versus accidental-subtracted reconstruction; bias in σ",
          "the counting time needed for a target error bar"),
         ("Simulate a tomography of phi+ with a pair rate of 1e6 per second, detection efficiency 0.2 on both "
          "sides, 500 dark counts per second, a 20 ns coincidence window and 0.1 s per setting. How much do the "
          "accidentals lower the concurrence, and does subtracting them help?",),
         ("simulate_tomography",)),
    Area("certification", "Certification with statistical significance",
         "Is it really entangled, steerable, Bell nonlocal? With the number of standard deviations.",
         ("entanglement witnesses, the PPT criterion, concurrence > 0",
          "EPR steering (three-setting test F3 and the exact criterion where it applies), CHSH",
          "n σ statistical and realistic; verdicts < 3σ, 3–5σ, ≥ 5σ"),
         ("Reconstruct data/state_tomo_data.csv. Are the entanglement, EPR steering and the CHSH violation "
          "statistically significant? Give the number of standard deviations.",
          "Create the Werner state with p = 0.55. Is it steerable?"),
         ("certify_correlations",)),
    Area("protocols", "Protocols: teleportation, QKD, error correction",
         "How useful a state is for a task.",
         ("teleportation fidelity from the fully entangled fraction (classical limit 2/3)",
          "entanglement-based BB84: QBERs and the asymptotic secret-key fraction, before and after phase "
          "compensation",
          "the three-qubit bit-flip code: logical versus physical error"),
         ("Create a Werner state with p = 0.5 built on phi+. What is its teleportation fidelity, and does it "
          "beat the classical limit?",
          "Reconstruct data/state_tomo_data.csv and use the state for entanglement-based BB84. Can a secret key "
          "be made, and what changes after compensating the phase?",
          "For the three-qubit bit-flip code, compare the logical error with the physical error at p = 0.1 and "
          "p = 0.6."),
         ("teleportation_fidelity", "qkd_key_rate", "bit_flip_code")),
    Area("plots", "Plots",
         "Publication-style figures, each with the exact numbers behind it.",
         ("any measure versus Werner p, noise strength, a pure-state angle, a wave-plate angle or a phase "
          "(up to 3 states and 6 measures, with the exact crossing points)",
          "density matrices (heatmap, magnitude and phase, 3D bars), Bloch vectors and correlation matrices",
          "polarization-correlation fringes with visibilities; your own CSV columns with fits"),
         ("Plot the concurrence, CHSH maximum and steering F3 of Werner states versus p. Where does each of "
          "them cross its bound?",
          "Reconstruct data/state_tomo_data.csv and plot its polarization-correlation fringes. What are the "
          "visibilities in the H/V and D/A bases?",
          "Plot data/example_fringe_scan.csv: angle_deg against counts_HV and counts_DA with a fringe fit."),
         ("plot_sweep", "plot_density_matrix", "plot_bloch_and_correlations", "plot_fringes", "plot_data_file")),
)

LIMITS = (
    "One or two qubits (one photon's polarization, or a polarization-entangled photon pair). No three-qubit "
    "states, except inside the bit-flip code.",
    "Time evolution is Markovian (Lindblad). Non-Markovian dynamics are not modelled.",
    "QKD key rates are asymptotic (infinite key). Detector models are simple: one detector per photon, "
    "Poisson counts.",
    "Every number comes from tested code, never from the language model. If a task needs a tool the agent "
    "does not have, it says so instead of guessing.",
)

CONCEPTS = ("It also answers concept questions without calculation, e.g. \"For two qubits, how are concurrence "
            "and negativity related?\"")


def area(key: str) -> Area:
    return next(a for a in AREAS if a.key == key)


def markdown() -> str:
    """The capabilities page (web app and README)."""
    out = ["## What this agent can do", ""]
    for a in AREAS:
        out += [f"### {a.title}", a.summary, ""]
        out += [f"- {c}" for c in a.computes]
        out += [f"- *{n}*" for n in a.notes]
        out += ["", "Try: " + " · ".join(f"\"{e}\"" for e in a.examples[:1]), ""]
    out += ["### Concept questions", CONCEPTS, "", "### Limits"] + [f"- {x}" for x in LIMITS]
    return "\n".join(out)


def help_answer() -> str:
    """The agent's own reply to 'what can you do?' (no language model needed)."""
    lines = ["I compute properties of **single photons and photon pairs** (polarization states, including "
             "polarization-entangled pairs) with tested code. Ask in plain English; here is what I can do:", "",
             "| Area | What I compute |", "|---|---|"]
    for a in AREAS:
        lines.append(f"| {a.title} | {a.summary} |")
    lines += ["", "**Examples you can ask:**"]
    lines += [f"- {a.examples[0]}" for a in AREAS[:6]]
    lines += ["", "**Limits:** " + " ".join(LIMITS[:2]),
              "", "*You can upload coincidence counts (.csv) for tomography; the capabilities panel below the "
              "chat lists every area with more examples.*"]
    return "\n".join(lines)


def for_tool(key: str = "") -> dict:
    """What list_capabilities returns to the model."""
    areas = [a for a in AREAS if not key or key.lower() in (a.key, a.title.lower())] or list(AREAS)
    return {"areas": [{"area": a.title, "summary": a.summary, "computes": list(a.computes),
                       "example_questions": list(a.examples), **({"notes": list(a.notes)} if a.notes else {})}
                      for a in areas],
            "limits": list(LIMITS),
            "how_to_answer": ("say plainly whether the task is covered; if it is not listed, say there is no "
                              "tool for it and offer the closest covered task")}


if __name__ == "__main__":
    print(markdown())
