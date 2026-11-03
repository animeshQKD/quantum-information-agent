"""
workbench.py - the Workbench tab of app.py: run the tools DIRECTLY, without the language model.

Instant, exact and free (no tokens, no rate limits). Each function runs one tool in the user's session
(the same states as the chat) and returns (markdown, figure path or None, summary). The summary is what
"Explain in the chat" sends to the chat, so the model can interpret a result it did not compute.
These functions know nothing about Gradio; app.py builds the panels around them.
"""
import json

import agent_tools as at
import display
import session

BELL = ["phi+", "phi-", "psi+", "psi-"]
LOCAL_CHANNELS = list(at.CHANNELS)
GLOBAL_CHANNELS = list(at.GLOBAL_CHANNELS)
CHANNELS = LOCAL_CHANNELS + GLOBAL_CHANNELS
CHANNEL_CHOICES = ([(f"local · {c}", c) for c in LOCAL_CHANNELS] +
                   [(f"global (4x4 Kraus) · {c}", c) for c in GLOBAL_CHANNELS])
CHANNEL_HELP = dict(at.NOISE_HELP)
LOCAL_PROCESSES = list(at.PROCESSES)
GLOBAL_PROCESSES = list(at.GLOBAL_PROCESSES)
PROCESSES = LOCAL_PROCESSES + GLOBAL_PROCESSES
PROCESS_CHOICES = ([(f"local · {p}", p) for p in LOCAL_PROCESSES] +
                   [(f"global · {p}", p) for p in GLOBAL_PROCESSES])
PROCESS_HELP = {**at.PROCESSES, **at.GLOBAL_PROCESSES}
POLARIZATIONS = ["H", "V", "D", "A", "R", "L"]
QUBITS = {"both photons": "both", "photon A": "A", "photon B": "B"}
ON = {v: k for k, v in QUBITS.items()}                  # "both" -> "both photons", for summaries
SWEEPS = list(at.SWEEPS)
QUANTITIES = ["concurrence", "negativity", "log_negativity", "entanglement_of_formation", "purity", "entropy",
              "chsh_max", "mutual_information", "discord", "coherence_l1", "fidelity", "fully_entangled_fraction",
              "teleportation_fidelity", "steering_f3", "key_fraction"]
DM_STYLES = ["heatmap", "magnitude_phase", "bars3d"]
MEASURE_NAMES = [("purity", "Purity"), ("entropy", "von Neumann entropy (bits)"), ("concurrence", "Concurrence C"),
                 ("entanglement_of_formation", "Entanglement of formation"), ("negativity", "Negativity N"),
                 ("log_negativity", "Log-negativity"), ("chsh_max", "CHSH max (local bound 2)"),
                 ("coherence_l1", "l1 coherence (H/V)"), ("coherence_rel_entropy", "Relative-entropy coherence"),
                 ("mutual_information", "Mutual information (bits)"), ("discord", "Quantum discord (bits)"),
                 ("entropy_A", "Entropy of photon A (bits)"), ("entropy_B", "Entropy of photon B (bits)"),
                 ("conditional_entropy_A_given_B", "Conditional entropy S(A∣B) (bits)")]   # '∣' (U+2223): a plain '|' splits the table cell
ONE_PHOTON_NAMES = [("degree_of_polarization", "Degree of polarization"), ("purity", "Purity"),
                    ("entropy", "von Neumann entropy (bits)")]


class WorkbenchError(Exception):
    """A user-facing problem (bad input, unknown state) shown in the panel instead of a traceback."""


def _run(sid: str, tool, **args):
    """Run one tool in this user's session. Errors become WorkbenchError with the tool's own message."""
    with session.use(sid):
        at.CURRENT_QUESTION["text"] = ""          # no chat question applies to a Workbench action
        try:
            return tool.invoke(args)
        except Exception as e:  # noqa: BLE001 - every tool error is shown to the user
            msg = str(e).split("\n")[0].split(" Ask the user")[0]        # that hint is for the chat model
            raise WorkbenchError(msg.replace("Error invoking tool", "").strip(" :'")) from None


def _need(handle) -> str:
    if not handle:
        raise WorkbenchError("Choose a state first (create one in 'States', or in the chat).")
    return handle


def _new_handle(text: str) -> str:
    import re
    m = re.search(r"\b(rho_\d+)\b", text)
    return m.group(1) if m else ""


LABELS = {"e_bit": "bit error e_bit", "e_phase": "phase error e_phase", "h_e_bit": "h(e_bit)",
          "h_e_phase": "h(e_phase)"}


def _label(key: str) -> str:
    return LABELS.get(key, key.replace("_", " "))


def _short_desc(d: str) -> str:
    """A state description for people: no hint meant for the model, no session folder, no '<' tags."""
    d = session.display_path(d.split("; full report")[0])
    return display.to_unicode(d).replace("|", "∣").replace("<", "‹").replace(">", "›")


def _kv_table(d: dict, title: str = "") -> str:
    """A two-column table of a tool result (nested dicts flattened one level)."""
    rows = []
    for k, v in d.items():
        if k in ("figure", "formulas", "conventions", "note") or v in ("", None, [], {}):
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if not isinstance(v2, (dict, list)):
                    rows.append((f"{_label(k)} · {_label(k2)}", v2))
        elif not isinstance(v, list):
            rows.append((_label(k), v))
    body = "\n".join(f"| {k} | {_fmt(v)} |" for k, v in rows)
    return (f"**{title}**\n\n" if title else "") + "| Quantity | Value |\n|---|---|\n" + body


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return str(v).replace("|", "∣")


# ------------------------------------------------------------------ states
def state_choices(sid: str) -> list[tuple[str, str]]:
    """[(label, handle)] for the dropdowns, newest first."""
    with session.use(sid):
        items = [(h, at.DESC.get(h, "")) for h in at.STORE]
    return [(f"{h} — {_short_desc(d)[:60]}", h) for h, d in reversed(items)]


def states_table(sid: str) -> str:
    with session.use(sid):
        items = [(h, at.DESC.get(h, "")) for h in at.STORE]
    if not items:
        return "*No states yet. Create one below, or ask in the chat — both share the same states.*"
    rows = "\n".join(f"| {h} | {_short_desc(d)} |" for h, d in items)
    return "| State | Description |\n|---|---|\n" + rows


def measures_table(sid: str, handle: str) -> str:
    m = _run(sid, at.compute_measures, handle=_need(handle))
    desc = _short_desc(_desc(sid, handle))
    if m.get("photons") == 1:
        rows = [(k, v) for k, v in m["stokes"].items()]
        rows += [(name, m[key]) for key, name in ONE_PHOTON_NAMES]
        ell = m.get("ellipse")
        if ell:
            rows += [("Ellipse azimuth (degrees)", ell["azimuth_deg"]), ("Ellipticity angle (degrees)",
                                                                       ell["ellipticity_deg"]),
                     ("Shape", f"{ell['shape']}, handedness {ell['handedness']}")]
        rows += [("Polarized part", display.to_unicode(m["polarized_part"]))]
        rows += [(f"P({k})", v) for k, v in m["probabilities"].items()]
        body = "\n".join(f"| {a} | {_fmt(b) if not isinstance(b, str) else b.replace('|', '∣')} |" for a, b in rows)
        return (f"**All measures of {handle}** ({desc}; one photon)\n\n| Quantity | Value |\n|---|---|\n{body}"
                f"\n\n<sub>{m['convention']}</sub>")
    rows = "\n".join(f"| {name} | {m[key]:.4f} |" for key, name in MEASURE_NAMES if key in m)
    return f"**All measures of {handle}** ({desc})\n\n| Measure | Value |\n|---|---|\n{rows}"


def matrix_md(sid: str, handle: str, before: str = "") -> str:
    """The density matrix of `handle` as numbers; with `before`, also that matrix and Δρ = after - before."""
    r = _run(sid, at.show_density_matrix, handle=handle, compare_with=before or "")
    return display.format_density_matrix(r)


def show_matrix(sid: str, handle: str, before: str = ""):
    handle = _need(handle)
    before = (before or "").strip()
    return (matrix_md(sid, handle, before), None,
            f"the density matrix of {handle}" + (f" and its change from {before}" if before else ""))


def _with_matrix(md: str, sid: str, handle: str, before: str = "") -> str:
    return md + "\n\n---\n\n" + matrix_md(sid, handle, before)


def create(sid: str, mode: str, kind: str = "phi+", p: float = 1.0, bell: str = "psi-",
           amps=("1", "0", "0", "1"), mix_a: str = "", mix_b: str = "", weight: float = 0.5,
           photon_how: str = "named", polarization: str = "H", azimuth: float = 0.0, ellipticity: float = 0.0,
           dop: float = 1.0, pair: str = "", which: str = "A", photon_a: str = "", photon_b: str = ""):
    """Create a state: two photons (Bell/Werner, pure from amplitudes, a mixture of two stored states, the
    product of two one-photon states) or one photon (named, ellipse, or one photon of a stored pair)."""
    if mode == "photon":
        if photon_how == "ellipse":
            text = _run(sid, at.create_single_photon_state, azimuth_deg=float(azimuth),
                        ellipticity_deg=float(ellipticity), dop=float(dop))
        else:
            text = _run(sid, at.create_single_photon_state, polarization=polarization, dop=float(dop))
    elif mode == "reduced":
        if not pair:
            raise WorkbenchError("Choose the two-photon state whose photon you want.")
        text = _run(sid, at.reduced_state, handle=pair, photon=which)
    elif mode == "product":
        if not (photon_a and photon_b):
            raise WorkbenchError("Choose the two one-photon states (photon A and photon B).")
        text = _run(sid, at.combine_photons, handle_a=photon_a, handle_b=photon_b)
    elif mode == "pure":
        text = _run(sid, at.create_pure_state, amp_HH=amps[0] or "0", amp_HV=amps[1] or "0",
                    amp_VH=amps[2] or "0", amp_VV=amps[3] or "0")
    elif mode == "mixture":
        if not (mix_a and mix_b):
            raise WorkbenchError("Choose the two states to mix.")
        if not 0 <= float(weight) <= 1:
            raise WorkbenchError("The weight must be between 0 and 1.")
        text = _run(sid, at.mix_states, handles=[mix_a, mix_b], weights=[float(weight), 1 - float(weight)])
    else:
        args = {"kind": kind}
        if kind == "werner":
            args.update(p=float(p), bell=bell)
        text = _run(sid, at.create_state, **args)
    h = _new_handle(text)
    md = _with_matrix(f"✓ {display.to_unicode(text)}\n\n" + measures_table(sid, h), sid, h)
    return md, None, f"the state {h} ({session.display_path(_desc(sid, h).split('; full report')[0])})"


def _desc(sid: str, h: str) -> str:
    with session.use(sid):
        return at.DESC.get(h, "")


# ------------------------------------------------------------------ noise, evolution, wave plates
def _is_global(channel: str) -> bool:
    return channel in at.GLOBAL_CHANNELS


def _where(channel: str, qubits: str, photons: int) -> str:
    if photons == 1:
        return "the photon"
    return "the pair (global)" if _is_global(channel) else ON.get(qubits, qubits)


def kraus_md(sid: str, channel: str, param: float, qubits: str) -> str:
    return display.format_kraus(_run(sid, at.get_kraus_operators, channel=channel, param=float(param),
                                     qubits=qubits))


def noise(sid: str, handle: str, channel: str, param: float, qubits: str, show_kraus: bool = False):
    handle = _need(handle)
    if _is_global(channel):
        text = _run(sid, at.apply_global_noise, handle=handle, channel=channel, param=float(param))
    else:
        text = _run(sid, at.apply_local_noise, handle=handle, channel=channel, param=float(param), qubits=qubits)
    new = _new_handle(text)
    before = _run(sid, at.compute_measures, handle=handle)
    after = _run(sid, at.compute_measures, handle=new)
    photons = 1 if before.get("photons") == 1 else 2
    if photons == 1:
        keys = [("degree_of_polarization", "Degree of polarization"), ("purity", "Purity"),
                ("entropy", "von Neumann entropy (bits)")]
        rows = "\n".join(f"| {name} | {before[k]:.4f} | {after[k]:.4f} |" for k, name in keys)
        rows += "\n" + "\n".join(f"| {k} | {before['stokes'][k]:.4f} | {after['stokes'][k]:.4f} |"
                                  for k in before["stokes"])
    else:
        rows = "\n".join(f"| {name} | {before[k]:.4f} | {after[k]:.4f} |" for k, name in MEASURE_NAMES
                         if k in ("purity", "concurrence", "negativity", "chsh_max", "coherence_l1", "discord"))
    where = _where(channel, qubits, photons)
    md = (f"✓ {display.to_unicode(text)}\n\n| Measure | before ({handle}) | after ({new}) |\n|---|---|---|\n{rows}"
          f"\n\n<sub>{channel} on {where}: {CHANNEL_HELP[channel]}</sub>")
    md = _with_matrix(md, sid, new, handle)
    if show_kraus:
        md += "\n\n---\n\n" + kraus_md(sid, channel, param, qubits)
    return md, None, f"{new} = {channel} with parameter {param} on {where} of {handle}, compared with {handle}"


def noise_sweep(sid: str, handle: str, channel: str, qubits: str):
    """The chosen stored state under the chosen channel, parameter from 0 to 1 (plot + exact crossings)."""
    handle = _need(handle)
    with session.use(sid):
        one = handle in at.STORE and at.STORE[handle].shape[0] == 2
    if one:
        return _one_photon_noise_curve(sid, handle, channel)
    r = _run(sid, at.plot_sweep, sweep="noise", handles=[handle], channel=channel, qubits=qubits,
             quantities=["concurrence", "negativity", "purity", "chsh_max"])
    where = _where(channel, qubits, 2)
    return (display.format_sweep(r), r["figure"],
            f"the measures of {handle} under {channel} noise on {where} for the parameter from 0 to 1")


def _one_photon_noise_curve(sid: str, handle: str, channel: str):
    """Degree of polarization, purity and entropy of a ONE-photon state versus the channel parameter."""
    import numpy as np
    import plots as pl
    import qi_tools as qt
    if channel in at.GLOBAL_CHANNELS:
        raise WorkbenchError(f"'{channel}' is a two-photon (global) channel; a one-photon state takes the local "
                             "channels.")
    with session.use(sid):
        rho0 = at.STORE[handle]
        xs = np.linspace(0, 1, 101)
        rhos = [qt.apply_kraus(rho0, at.CHANNELS[channel][0](float(x))) for x in xs]
        dop = np.array([np.linalg.norm(qt.stokes_vector(r)) for r in rhos])
        pur = np.array([qt.purity(r) for r in rhos])
        ent = np.array([qt.von_neumann_entropy(r) for r in rhos]) + 0.0
        ent[np.abs(ent) < 5e-13] = 0.0
        panels = [{"title": "Degree of polarization |S|", "ys": [dop], "reference": None, "marks": []},
                  {"title": "Purity Tr ρ²", "ys": [pur], "reference": None, "marks": []},
                  {"title": "von Neumann entropy (bits)", "ys": [ent], "reference": None, "marks": []}]
        path = pl.figure_path(f"{handle}_{channel}_1photon_curve", [handle, channel])
        pl.sweep_figure(xs, panels, [handle], at.CHANNELS[channel][1], f"{handle} (one photon): {channel}", path)
    rows = "\n".join(f"| {x:.1f} | {dop[i]:.4f} | {pur[i]:.4f} | {ent[i]:.4f} |"
                      for i, x in enumerate(xs) if i % 10 == 0)
    md = (f"**{handle} (one photon) under {channel}**, x = {at.CHANNELS[channel][1]}\n\n"
          "| Parameter | Degree of polarization | Purity | Entropy (bits) |\n|---|---|---|---|\n" + rows)
    return md, path, f"the degree of polarization, purity and entropy of {handle} under {channel} from 0 to 1"


def evolve(sid: str, handle: str, process: str, gamma: float, t_max: float, qubits: str, omega: float = 0.0,
           elements: bool = False):
    handle = _need(handle)
    r = _run(sid, at.evolve_state, handle=handle, process=process, gamma=float(gamma), t_max=float(t_max),
             qubits=qubits, omega=float(omega), plot_elements=bool(elements))
    md = display.format_evolution(r)
    twin = r["features"].get("same_as_kraus_at_t_max")
    if twin:
        md += f"\n\n<sub>same state as {twin}</sub>"
    md = _with_matrix(md, sid, r["final_state"], handle)
    where = ("the pair (one common environment)" if process in at.GLOBAL_PROCESSES else ON.get(qubits, qubits))
    figs = [r["figure"], r["elements_figure"]] if r.get("elements_figure") else r["figure"]
    return (md, figs, f"the evolution of {handle} under {process} (rate {gamma}) on {where} up to t = {t_max}")


def waveplate(sid: str, handle: str, plate: str, angle: float, qubit: str):
    handle = _need(handle)
    r = _run(sid, at.apply_waveplate, handle=handle, plate=plate, angle_deg=float(angle), qubit=qubit)
    where = "the photon" if r.get("photons") == 1 else f"photon {qubit}"
    return (_with_matrix(display.format_waveplate(r), sid, r["new_handle"], handle), None,
            f"a {plate} at {angle}° on {where} applied to {handle}")


def compensate(sid: str, handle: str, qubit: str):
    handle = _need(handle)
    r = _run(sid, at.compensate_phase, handle=handle, qubit=qubit)
    return (_with_matrix(display.format_compensation(r), sid, r["new_handle"], handle), None,
            f"the phase compensation of {handle} on photon {qubit}")


# ------------------------------------------------------------------ analysis and protocols
def analyze(sid: str, handle: str, what: str):
    handle = _need(handle)
    if what == "measures":
        return _with_matrix(measures_table(sid, handle), sid, handle), None, f"all measures of {handle}"
    if what == "matrix":
        return matrix_md(sid, handle), None, f"the density matrix of {handle}"
    if what == "certify":
        r = _run(sid, at.certify_correlations, handle=handle)
        return display.format_certification(r), None, f"the certification (entanglement, steering, CHSH) of {handle}"
    if what == "teleportation":
        r = _run(sid, at.teleportation_fidelity, handle=handle)
        return _kv_table(r, f"Teleportation with {handle}"), None, f"the teleportation fidelity of {handle}"
    if what == "qkd":
        r = _run(sid, at.qkd_key_rate, handle=handle)
        return _kv_table(r, f"Entanglement-based BB84 with {handle}"), None, f"the BB84 key rate of {handle}"
    raise WorkbenchError(f"Unknown analysis '{what}'.")


def compare(sid: str, a: str, b: str):
    r = _run(sid, at.compare_states, handle_a=_need(a), handle_b=_need(b))
    return display.format_comparisons([r]), None, f"the comparison of {a} and {b}"


def key_from_qber(sid: str, qber_z: float, qber_x: float):
    r = _run(sid, at.qkd_key_rate, qber_z=float(qber_z), qber_x=float(qber_x))
    return (_kv_table(r, f"BB84 key with QBER {qber_z} (Z) and {qber_x} (X)"), None,
            f"the BB84 key fraction for QBERs {qber_z} and {qber_x}")


def repetition_code(sid: str, p: float):
    r = _run(sid, at.bit_flip_code, p=float(p))
    return _kv_table(r, f"Three-qubit bit-flip code at p = {p}"), None, f"the bit-flip code at p = {p}"


# ------------------------------------------------------------------ data and detectors
def tomography(sid: str, csv_path: str, count_column: str = "auto", convention: str = "R=(H+iV)/sqrt2"):
    if not csv_path:
        raise WorkbenchError("Upload a CSV file or choose the sample file first.")
    r = _run(sid, at.reconstruct_state_from_counts, csv_path=csv_path, count_column=count_column or "auto",
             r_convention=convention)
    h = r["handle"]
    report = display.format_tomography_report(_run(sid, at.get_tomography_report, handle=h))
    fit = _run(sid, at.plot_tomography_fit, handle=h)
    md = f"✓ Reconstructed {h}\n\n{report}\n\n{display.format_tomography_fit(fit)}"
    return _with_matrix(md, sid, h), fit["figure"], f"the tomography of {csv_path} (state {h})"


def photon_tomography(sid: str, n_h=None, n_v=None, n_d=None, n_a=None, n_r=None, n_l=None,
                      convention: str = "R=(H+iV)/sqrt2"):
    """One photon from counts behind H, V, D, A, R, L analysers (empty fields are left out)."""
    counts = {}
    for key, v in zip("HVDARL", (n_h, n_v, n_d, n_a, n_r, n_l)):
        if v not in (None, "") and str(v).strip():
            try:
                counts[f"counts_{key}"] = float(v)
            except ValueError:
                raise WorkbenchError(f"'{v}' is not a number of counts.") from None
    if len(counts) < 4:
        raise WorkbenchError("Give at least four counts, e.g. H, V, D and R (all six are best).")
    r = _run(sid, at.reconstruct_single_photon, r_convention=convention, **counts)
    h = r["handle"]
    return (_with_matrix(display.format_single_photon_tomography(r) + "\n\n" + measures_table(sid, h), sid, h),
            None, f"the one-photon tomography from the counts {', '.join(f'{k[-1]} = {v:g}' for k, v in counts.items())} "
                  f"(state {h})")


def simulate(sid: str, handle: str, pair_rate: float, eta_a: float, eta_b: float, dark: float, window_ns: float,
             time_s: float):
    r = _run(sid, at.simulate_tomography, handle=_need(handle), pair_rate=float(pair_rate), eta_a=float(eta_a),
             eta_b=float(eta_b), dark_a=float(dark), dark_b=float(dark), window_ns=float(window_ns),
             time_per_setting_s=float(time_s))
    md = _with_matrix(display.format_simulation(r), sid, r["reconstructed_raw"]["handle"], handle)
    return (md, None,
            f"a simulated tomography of {handle} (pair rate {pair_rate}/s, efficiencies {eta_a}/{eta_b}, "
            f"{dark} dark counts/s, {window_ns} ns window, {time_s} s per setting)")


# ------------------------------------------------------------------ plots
def sweep(sid: str, sweep_kind: str, quantities, handles=(), start=None, stop=None, bell: str = "psi-",
          channel: str = "depolarizing", qubits: str = "both", plate: str = "HWP", qubit: str = "B",
          reference: str = ""):
    if not quantities:
        raise WorkbenchError("Choose at least one quantity to plot.")
    args = {"sweep": sweep_kind, "quantities": list(quantities), "bell": bell, "channel": channel,
            "qubits": qubits, "plate": plate, "qubit": qubit, "reference": reference or ""}
    if at.SWEEPS[sweep_kind][3]:
        if not handles:
            raise WorkbenchError(f"The '{sweep_kind}' sweep needs 1 to 3 states.")
        args["handles"] = list(handles)
    for key, val in (("start", start), ("stop", stop)):
        if val not in (None, "") and str(val).strip():
            try:
                args[key] = float(val)
            except ValueError:
                raise WorkbenchError(f"'{val}' is not a number.") from None
    r = _run(sid, at.plot_sweep, **args)
    return display.format_sweep(r), r["figure"], f"the plot of {', '.join(quantities)} versus {sweep_kind}"


def density(sid: str, handle: str, style: str, compare_with: str = ""):
    r = _run(sid, at.plot_density_matrix, handle=_need(handle), style=style, compare_with=compare_with or "")
    md = matrix_md(sid, handle, compare_with or "")
    if "fidelity_between_plotted_states" in r:
        md += f"\n\n<sub>fidelity between the two plotted states: {r['fidelity_between_plotted_states']:.4f}</sub>"
    return md, r["figure"], f"the density matrix of {handle}"


def fringes(sid: str, handle: str):
    r = _run(sid, at.plot_fringes, handle=_need(handle))
    return display.format_fringes(r), r["figure"], f"the polarization-correlation fringes of {handle}"


def bloch(sid: str, handles):
    if not handles:
        raise WorkbenchError("Choose 1 to 3 states.")
    r = _run(sid, at.plot_bloch_and_correlations, handles=list(handles))
    return display.format_bloch(r), r["figure"], f"the Bloch vectors and correlations of {', '.join(handles)}"


def explain_prompt(summary: str) -> str:
    """The question 'Explain in the chat' puts in the chat box."""
    return (f"In the Workbench I computed {summary}. Explain what these results mean physically "
            "(recompute with the tools if you need the numbers).")


if __name__ == "__main__":
    md, fig, s = create("default", "bell", kind="werner", p=0.8)
    print(md, "\n", json.dumps(s))
