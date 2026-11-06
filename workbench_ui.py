"""
workbench_ui.py - the Workbench tab of the web app: panels that run the tools directly (workbench.py).

Six panels (States, Noise & time evolution, Wave plates, Analysis & protocols, Tomography & detectors,
Plots) and one shared result area; every result that makes or uses a state also shows its density matrix
(before, after and the change Δρ when a state is transformed). Every action uses the browser tab's own session, so the Workbench
and the chat see the same states. "Explain in the chat" sends the last result to the chat.
"""
import os
import shutil

import gradio as gr

import session
import workbench as wb

SAMPLE_CSV = "data/state_tomo_data.csv"
PRIVATE = "private"     # UI events are not API/MCP endpoints: the MCP tools are in mcp_server.py
SWEEP_HELP = {
    "werner_p": "Werner state p|B⟩⟨B| + (1−p)I/4 versus p (choose B below)",
    "pure_angle": "cos θ|HH⟩ + sin θ|VV⟩ versus θ (radians)",
    "visibility": "v ρ + (1−v) I/4 for the chosen states, versus v",
    "noise": "a noise channel on the chosen states, versus its strength",
    "waveplate_angle": "a wave plate on one photon of the chosen states, versus its angle (degrees)",
    "phase_shift": "a phase shifter diag(1, e^(iφ)) on one photon, versus φ (degrees)",
}


def build(sid, box, tabs):
    """Add the Workbench tab. sid: the session gr.State; box: the chat's question box; tabs: the main
    gr.Tabs (to switch to the chat)."""
    dropdowns = []                                     # every state picker, refreshed after each action

    def picker(label, multi=False):
        dd = gr.Dropdown(choices=[], label=label, multiselect=multi, max_choices=3 if multi else None,
                         allow_custom_value=False)
        dropdowns.append(dd)
        return dd

    with gr.Tab("Workbench", id="bench") as tab:
        gr.Markdown("Run the tools **directly**: instant, exact, no language model and no cost. The states here "
                    "are the same as in the chat. After a result, **Explain in the chat** asks the agent to "
                    "interpret it.")
        with gr.Accordion("States in this session", open=True):
            states_md = gr.Markdown()
            refresh = gr.Button("Refresh", size="sm")

        with gr.Tabs():
            with gr.Tab("States"):
                mode = gr.Radio([("Bell or Werner state", "bell"), ("Pure two-photon state", "pure"),
                                 ("Mixture of two stored states", "mixture"), ("One photon", "photon"),
                                 ("One photon of a stored pair", "reduced"),
                                 ("Pair from two one-photon states", "product")], value="bell", label="What to create")
                with gr.Group(visible=True) as g_bell:
                    with gr.Row():
                        kind = gr.Dropdown(wb.BELL + ["werner"], value="phi+", label="State")
                        p = gr.Slider(0, 1, value=0.8, step=0.01, label="Werner p (only for 'werner')")
                        base = gr.Dropdown(wb.BELL, value="psi-", label="Werner built on (only for 'werner')")
                with gr.Group(visible=False) as g_pure:
                    gr.Markdown("Amplitudes as numbers or expressions with sqrt, cos, sin, exp, pi and i "
                                "(normalized automatically), e.g. `cos(0.3)` and `sin(0.3)`.")
                    with gr.Row():
                        a_hh = gr.Textbox("1", label="|HH⟩")
                        a_hv = gr.Textbox("0", label="|HV⟩")
                        a_vh = gr.Textbox("0", label="|VH⟩")
                        a_vv = gr.Textbox("1", label="|VV⟩")
                with gr.Group(visible=False) as g_mix:
                    with gr.Row():
                        mix_a = picker("First state")
                        mix_b = picker("Second state")
                        weight = gr.Slider(0, 1, value=0.5, step=0.01, label="Weight of the first state")
                with gr.Group(visible=False) as g_photon:
                    gr.Markdown("One photon's polarization (a 2×2 density matrix). R = (H + iV)/√2. Degree of "
                                "polarization < 1 mixes in unpolarized light: ρ = dop·|ψ⟩⟨ψ| + (1 − dop)·I/2.")
                    with gr.Row():
                        ph_how = gr.Radio([("H, V, D, A, R or L", "named"), ("Polarization ellipse", "ellipse")],
                                          value="named", label="Given by")
                        ph_pol = gr.Dropdown(wb.POLARIZATIONS, value="H", label="Polarization")
                        ph_az = gr.Number(30.0, label="Ellipse azimuth from H (degrees)")
                        ph_el = gr.Number(0.0, label="Ellipticity angle (−45° = L … 0 = linear … +45° = R)")
                        ph_dop = gr.Slider(0, 1, value=1.0, step=0.01, label="Degree of polarization")
                with gr.Group(visible=False) as g_red:
                    with gr.Row():
                        red_pair = picker("Two-photon state")
                        red_which = gr.Radio(["A", "B"], value="A", label="Photon")
                with gr.Group(visible=False) as g_prod:
                    with gr.Row():
                        prod_a = picker("One-photon state for photon A")
                        prod_b = picker("One-photon state for photon B")
                create = gr.Button("Create", variant="primary")
                groups = [g_bell, g_pure, g_mix, g_photon, g_red, g_prod]
                kinds = ["bell", "pure", "mixture", "photon", "reduced", "product"]
                mode.change(lambda m: [gr.update(visible=m == k) for k in kinds], mode, groups, api_visibility=PRIVATE)
                gr.Markdown("**Density matrix of a stored state** (as numbers)")
                with gr.Row():
                    show_h = picker("State")
                    show_ref = gr.Dropdown(["", "phi+", "phi-", "psi+", "psi-"], value="", allow_custom_value=True,
                                           label="Change from (optional: a Bell state or a handle such as rho_1)")
                    show_go = gr.Button("Show the density matrix")

            with gr.Tab("Noise & time evolution"):
                gr.Markdown("**Noise channel on a stored state.** *Local* channels act on each photon with its own "
                            "environment (choose which photons under *On*); *global* channels have 4×4 Kraus "
                            "operators acting on the pair (*On* does not apply). A one-photon state takes the "
                            "local channels.")
                with gr.Row():
                    n_h = picker("State")
                    n_ch = gr.Dropdown(wb.CHANNEL_CHOICES, value="depolarizing", label="Channel")
                    n_p = gr.Slider(0, 1, value=0.3, step=0.01, label="Parameter")
                    n_q = gr.Radio(list(wb.QUBITS.items()), value="both", label="On (local channels)")
                n_info = gr.Markdown(wb.CHANNEL_HELP["depolarizing"])
                n_kraus = gr.Checkbox(False, label="Also show the Kraus operators")
                with gr.Row():
                    n_go = gr.Button("Apply noise (creates a new state)", variant="primary")
                    c_go = gr.Button("Plot versus the parameter, 0 → 1 (same state and channel)")
                n_ch.change(lambda c: wb.CHANNEL_HELP[c], n_ch, n_info, api_visibility=PRIVATE)
                gr.Markdown("**Time evolution** (Lindblad master equation; creates the state at t_max). Local "
                            "processes act on the photons chosen under *On*; global ones on the pair.")
                with gr.Row():
                    e_h = picker("State")
                    e_proc = gr.Dropdown(wb.PROCESS_CHOICES, value="dephasing", label="Process")
                    e_g = gr.Number(1.0, label="Rate γ")
                    e_t = gr.Number(2.0, label="t_max")
                    e_q = gr.Radio(list(wb.QUBITS.items()), value="both", label="On (local processes)")
                    e_w = gr.Number(0.0, label="ω of H = (ω/2)σx (0 = none)")
                e_info = gr.Markdown(wb.PROCESS_HELP["dephasing"])
                e_el = gr.Checkbox(False, label="Also plot every density-matrix element versus time")
                e_go = gr.Button("Evolve")
                e_proc.change(lambda p_: wb.PROCESS_HELP[p_], e_proc, e_info, api_visibility=PRIVATE)

            with gr.Tab("Wave plates"):
                with gr.Row():
                    w_h = picker("State")
                    w_plate = gr.Radio(["HWP", "QWP"], value="HWP", label="Plate")
                    w_ang = gr.Number(45.0, label="Fast-axis angle from H (degrees)")
                    w_q = gr.Radio(["A", "B"], value="B", label="Photon")
                w_go = gr.Button("Apply wave plate", variant="primary")
                gr.Markdown("**Phase compensation**: the QWP(45°)–HWP(θ)–QWP(45°) setting that turns the state into "
                            "phi+ (or psi+)")
                with gr.Row():
                    pc_h = picker("State")
                    pc_q = gr.Radio(["A", "B"], value="B", label="Photon")
                pc_go = gr.Button("Find the wave plates")

            with gr.Tab("Analysis & protocols"):
                with gr.Row():
                    an_h = picker("State")
                    an_what = gr.Radio([("All measures", "measures"), ("Density matrix (numbers)", "matrix"),
                                        ("Certification (σ)", "certify"),
                                        ("Teleportation", "teleportation"), ("BB84 key", "qkd")],
                                       value="measures", label="Analysis")
                an_go = gr.Button("Analyse", variant="primary")
                gr.Markdown("**Compare two states** (fidelity and trace distance)")
                with gr.Row():
                    cm_a = picker("State A")
                    cm_b = picker("State B")
                cm_go = gr.Button("Compare")
                gr.Markdown("**Without a state**")
                with gr.Row():
                    qz = gr.Number(0.03, label="QBER Z (H/V)")
                    qx = gr.Number(0.03, label="QBER X (D/A)")
                    q_go = gr.Button("BB84 key fraction")
                with gr.Row():
                    bf_p = gr.Number(0.1, label="Bit-flip probability p")
                    bf_go = gr.Button("Three-qubit bit-flip code")

            with gr.Tab("Tomography & detectors"):
                gr.Markdown("**Tomography of measured counts.** CSV: one row per setting, a column of labels "
                            "HH, HV, …, RL (first letter = photon A) and one or more count columns; at least 16 "
                            "settings, all 36 best; max 1 MB.")
                with gr.Row():
                    t_file = gr.File(label="Upload counts (.csv)", file_types=[".csv"], height=90)
                    with gr.Column():
                        t_path = gr.Textbox(label="File", interactive=False)
                        t_sample = gr.Button("Use the sample file", size="sm")
                with gr.Row():
                    t_col = gr.Textbox("auto", label="Count column ('auto' or a name, e.g. L1R1)")
                    t_conv = gr.Radio(["R=(H+iV)/sqrt2", "R=(H-iV)/sqrt2"], value="R=(H+iV)/sqrt2",
                                      label="R/L convention")
                t_go = gr.Button("Reconstruct", variant="primary")
                gr.Markdown("**One photon from counts** behind H, V, D, A, R, L analysers (at least H, V, D and R; "
                            "empty fields are left out)")
                with gr.Row():
                    pt = [gr.Textbox("", label=f"Counts {k}") for k in "HVDARL"]
                with gr.Row():
                    pt_conv = gr.Radio(["R=(H+iV)/sqrt2", "R=(H-iV)/sqrt2"], value="R=(H+iV)/sqrt2",
                                       label="R/L convention")
                    pt_go = gr.Button("Reconstruct the photon")
                gr.Markdown("**Simulated experiment with real detectors**")
                with gr.Row():
                    s_h = picker("True state")
                    s_rate = gr.Number(1e6, label="Pair rate (1/s)")
                    s_ea = gr.Number(0.2, label="Efficiency A")
                    s_eb = gr.Number(0.2, label="Efficiency B")
                with gr.Row():
                    s_dark = gr.Number(500, label="Dark counts (1/s, each)")
                    s_win = gr.Number(20, label="Coincidence window (ns)")
                    s_time = gr.Number(0.1, label="Time per setting (s)")
                s_go = gr.Button("Simulate")

            with gr.Tab("Plots"):
                gr.Markdown("**Any measure versus a parameter**")
                with gr.Row():
                    sw_kind = gr.Dropdown([(SWEEP_HELP[k], k) for k in wb.SWEEPS], value="werner_p", label="x axis")
                    sw_q = gr.Dropdown(wb.QUANTITIES, value=["concurrence", "chsh_max"], multiselect=True,
                                       max_choices=6, label="Measures (up to 6)")
                with gr.Row():
                    sw_h = picker("States (for the last four x axes; up to 3)", multi=True)
                    sw_start = gr.Textbox("", label="From (empty = default)")
                    sw_stop = gr.Textbox("", label="To (empty = default)")
                with gr.Accordion("Settings of the x axis", open=False):
                    with gr.Row():
                        sw_bell = gr.Dropdown(wb.BELL, value="psi-", label="Werner built on")
                        sw_ch = gr.Dropdown(wb.CHANNELS, value="depolarizing", label="Noise channel")
                        sw_qb = gr.Radio(list(wb.QUBITS.items()), value="both", label="Noise on")
                        sw_plate = gr.Radio(["HWP", "QWP"], value="HWP", label="Plate")
                        sw_photon = gr.Radio(["A", "B"], value="B", label="Plate / phase on photon")
                        sw_ref = gr.Dropdown(["", "phi+", "phi-", "psi+", "psi-"], value="",
                                             label="Reference for 'fidelity'")
                sw_go = gr.Button("Plot", variant="primary")
                gr.Markdown("**Density matrix, fringes, Bloch vectors**")
                with gr.Row():
                    dm_h = picker("State")
                    dm_style = gr.Radio(wb.DM_STYLES, value="heatmap", label="Style")
                    dm_ref = gr.Dropdown(["", "phi+", "phi-", "psi+", "psi-"], value="", label="Next to")
                    dm_go = gr.Button("Density matrix")
                with gr.Row():
                    fr_h = picker("State")
                    fr_go = gr.Button("Polarization fringes")
                with gr.Row():
                    bl_h = picker("States (up to 3)", multi=True)
                    bl_go = gr.Button("Bloch vectors and correlations")

        out_md = gr.Markdown("*Results appear here.*")
        out_img = gr.Image(visible=False, show_label=False, height=420)
        out_img2 = gr.Image(visible=False, show_label=False, height=420)
        summary = gr.State("")
        explain = gr.Button("Explain in the chat", size="sm")

    # ------------------------------------------------------------------ wiring
    def updates(sid_, *current, jump=False):
        """New choices for every state picker; a picker keeps its choice, an empty one gets the newest state.
        jump=True (after creating a state in 'States' or by tomography): every single picker moves to the new
        state, so the next action uses what was just created."""
        choices = wb.state_choices(sid_)
        valid = {h for _, h in choices}
        newest = choices[0][1] if choices else None
        ups = []
        for dd, cur in zip(dropdowns, list(current) + [None] * (len(dropdowns) - len(current))):
            if dd.multiselect:
                ups.append(gr.update(choices=choices, value=[h for h in (cur or []) if h in valid]))
            else:
                ups.append(gr.update(choices=choices, value=newest if jump or cur not in valid else cur))
        return [wb.states_table(sid_)] + ups

    def action(fn, n_inputs, jump=False):
        def run(sid_, *args):
            values, current = args[:n_inputs], args[n_inputs:]
            try:
                with session.COMPUTE_LOCK:                  # pyplot is not thread-safe (session.py)
                    md, fig, summ = fn(sid_, *values)
            except wb.WorkbenchError as e:
                md, fig, summ = f"⚠ {e}", None, ""
            except (TypeError, ValueError):                 # an empty or non-numeric field
                md, fig, summ = "⚠ Please fill in every field (a number where a number is expected).", None, ""
            except Exception as e:  # noqa: BLE001 - never a traceback in the panel
                md, fig, summ = f"⚠ Unexpected error: {e}", None, ""
            md = session.display_path(md)                  # 'uploads/x.csv', not data/sessions/<id>/x.csv
            fig1, fig2 = (fig[0], fig[1]) if isinstance(fig, (list, tuple)) else (fig, None)
            created = jump and not md.startswith("⚠")
            return ([md, gr.update(value=fig1, visible=bool(fig1)), gr.update(value=fig2, visible=bool(fig2)), summ]
                    + updates(sid_, *current, jump=created))
        return run

    outs = [out_md, out_img, out_img2, summary, states_md] + dropdowns

    def wire(button, fn, inputs, jump=False):
        button.click(action(fn, len(inputs), jump), [sid] + inputs + dropdowns, outs, api_visibility=PRIVATE)

    def make(s, m, k, p_, b, hh, hv, vh, vv, ma, mb, w, how, pol, az, el, dop, pair, which, pa, pb):
        return wb.create(s, m, k, p_, b, (hh, hv, vh, vv), ma, mb, w, how, pol, az, el, dop, pair, which, pa, pb)
    wire(create, make, [mode, kind, p, base, a_hh, a_hv, a_vh, a_vv, mix_a, mix_b, weight,
                        ph_how, ph_pol, ph_az, ph_el, ph_dop, red_pair, red_which, prod_a, prod_b], jump=True)
    wire(show_go, lambda s, h, ref: wb.show_matrix(s, h, ref), [show_h, show_ref])
    wire(n_go, wb.noise, [n_h, n_ch, n_p, n_q, n_kraus])
    wire(c_go, wb.noise_sweep, [n_h, n_ch, n_q])
    wire(e_go, wb.evolve, [e_h, e_proc, e_g, e_t, e_q, e_w, e_el])
    wire(w_go, wb.waveplate, [w_h, w_plate, w_ang, w_q])
    wire(pc_go, wb.compensate, [pc_h, pc_q])
    wire(an_go, wb.analyze, [an_h, an_what])
    wire(cm_go, wb.compare, [cm_a, cm_b])
    wire(q_go, wb.key_from_qber, [qz, qx])
    wire(bf_go, wb.repetition_code, [bf_p])
    wire(t_go, wb.tomography, [t_path, t_col, t_conv], jump=True)
    wire(pt_go, lambda s, *a: wb.photon_tomography(s, *a), pt + [pt_conv], jump=True)
    wire(s_go, wb.simulate, [s_h, s_rate, s_ea, s_eb, s_dark, s_win, s_time])
    wire(sw_go, lambda s, k, q, h, a, b, bell, ch, qb, pl, ph, ref: wb.sweep(s, k, q, h, a, b, bell, ch, qb, pl, ph, ref),
         [sw_kind, sw_q, sw_h, sw_start, sw_stop, sw_bell, sw_ch, sw_qb, sw_plate, sw_photon, sw_ref])
    wire(dm_go, wb.density, [dm_h, dm_style, dm_ref])
    wire(fr_go, wb.fringes, [fr_h])
    wire(bl_go, wb.bloch, [bl_h])

    t_file.upload(copy_upload, [t_file, sid], t_path, api_visibility=PRIVATE)
    t_sample.click(lambda: SAMPLE_CSV, None, t_path, api_visibility=PRIVATE)
    refresh.click(updates, [sid] + dropdowns, [states_md] + dropdowns, api_visibility=PRIVATE)
    tab.select(updates, [sid] + dropdowns, [states_md] + dropdowns, api_visibility=PRIVATE)
    explain.click(lambda s: (wb.explain_prompt(s), gr.Tabs(selected="chat")) if s else
                  (gr.update(), gr.update()), summary, [box, tabs], api_visibility=PRIVATE)
    return tab


def copy_upload(file, sid_):
    """Put an uploaded CSV into this user's data folder; return its path for the reconstruction."""
    if file is None:
        return ""
    src = file if isinstance(file, str) else file.name
    if not src.lower().endswith(".csv") or os.path.getsize(src) > 1_000_000:
        return ""
    with session.use(sid_) as s:
        os.makedirs(s.data_dir, exist_ok=True)
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in os.path.basename(src))
        shutil.copy(src, f"{s.data_dir}/{name}")
    return f"{session.UPLOADS}/{name}"                      # resolved to the user's own folder by the tool
