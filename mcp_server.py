"""
mcp_server.py - the physics tools as an MCP server (Model Context Protocol), served by app.py.

app.py calls register() inside its gr.Blocks and launches with mcp_server=True. Gradio then serves
    http://127.0.0.1:7860/gradio_api/mcp/                                    (locally)
    https://animesh1986-quantum-information-agent.hf.space/gradio_api/mcp/    (the Space)
(streamable HTTP), and any MCP client - Claude Desktop, Cursor, another agent - can call the tools below.

What is exposed: the TOOLS list at the end - the tested physics tools of agent_tools.py, each behind a
small typed wrapper with a description written for an outside agent (the agent_tools descriptions also
address this app's own chat model, e.g. "you cannot see the image", which is wrong over MCP: figures are
returned as PNG images). NOT exposed: the chat (it spends the language-model quota), the Workbench
buttons (app.py and workbench_ui.py mark every UI event api_visibility="private"), plot_data_file and
list_capabilities. No MCP tool ever calls a language model.

Sessions: states live on the server and are addressed by handles (rho_1, rho_2, ...), as in the chat.
The first call with session_id="" opens a session; every result returns its session_id, which the client
passes in later calls. Ids are random ("mcp_" + 32 hex digits) and cannot be chosen by a client, so one
client cannot reach another's states, a web visitor's session or the command-line default session.
Sessions expire like web sessions (SESSION_TTL_S idle), and MCP sessions are capped separately
(MAX_MCP_SESSIONS) so they can never push web visitors out.

Compute lock: every agent_tools tool already holds session.COMPUTE_LOCK while it runs (the _locked
wrapper there), so MCP calls, chat and Workbench never run matplotlib at the same time. The wrappers
also take the lock themselves (re-entrant) while they read and combine figures.

Limits: MCP_CALLS_PER_MINUTE for the whole server (default 120 on a Space, none locally); sweeps are
capped at their tool's maximum points; pasted CSV text at agent_tools.MAX_CSV_BYTES.
"""
# No "from __future__ import annotations": Gradio reads the return annotations as real types (a string
# "tuple[str, gr.FileData]" would make a figure tool's two outputs one).
import collections
import json
import math
import os
import re
import threading
import time
import uuid
from typing import Literal

import gradio as gr

import agent_tools as at
import session

ENABLED = os.getenv("MCP_SERVER", "1") != "0"
PREFIX = "mcp_"
MAX_MCP_SESSIONS = int(os.getenv("MAX_MCP_SESSIONS", "100"))
CALLS_PER_MINUTE = int(os.getenv("MCP_CALLS_PER_MINUTE", "120" if session.ON_SPACE else "0"))   # 0 = no limit
STATELESS_ID = "mcp-stateless"          # internal; does not match _ID, so no client can address it
_ID = re.compile(r"^mcp_[0-9a-f]{32}$")
_CREATED = re.compile(r"^Created (rho_\d+)")
FIGURE_KEYS = ("figure", "elements_figure")
_calls: collections.deque = collections.deque()
_calls_lock = threading.Lock()
_new_lock = threading.Lock()

Bell = Literal["phi+", "phi-", "psi+", "psi-"]
StateKind = Literal["phi+", "phi-", "psi+", "psi-", "werner"]
LocalChannel = Literal["phase_damping", "amplitude_damping", "depolarizing", "bit_flip", "phase_flip",
                       "bit_phase_flip"]
GlobalChannel = Literal["global_depolarizing", "global_dephasing", "collective_dephasing",
                        "collective_amplitude_damping", "correlated_bit_flip", "correlated_phase_flip",
                        "correlated_bit_phase_flip"]
AnyChannel = Literal["phase_damping", "amplitude_damping", "depolarizing", "bit_flip", "phase_flip",
                     "bit_phase_flip", "global_depolarizing", "global_dephasing", "collective_dephasing",
                     "collective_amplitude_damping", "correlated_bit_flip", "correlated_phase_flip",
                     "correlated_bit_phase_flip"]
Process = Literal["dephasing", "amplitude_decay", "depolarizing", "bit_flip", "bit_phase_flip", "none",
                  "global_depolarizing", "global_dephasing", "collective_dephasing", "collective_decay",
                  "correlated_bit_flip", "correlated_phase_flip"]
Qubits = Literal["both", "A", "B"]
Photon = Literal["A", "B"]
Plate = Literal["HWP", "QWP", "retarder"]
Sweep = Literal["werner_p", "pure_angle", "visibility", "noise", "waveplate_angle", "phase_shift"]
Quantity = Literal["concurrence", "negativity", "log_negativity", "entanglement_of_formation", "purity",
                   "entropy", "chsh_max", "mutual_information", "discord", "coherence_l1", "fidelity",
                   "fully_entangled_fraction", "teleportation_fidelity", "steering_f3", "key_fraction"]
DMStyle = Literal["bars3d", "heatmap", "magnitude_phase"]
RConvention = Literal["R=(H+iV)/sqrt2", "R=(H-iV)/sqrt2"]
Polarization = Literal["", "H", "V", "D", "A", "R", "L"]

SESSION = ("Session from an earlier result of this server. Leave empty on your first call: a new session "
           "is opened and its session_id is returned. Pass that session_id in every later call so the "
           "stored states (rho_1, rho_2, ...) are kept. Sessions are deleted after 2 hours without a call.")
HANDLE = "Handle of a stored state, e.g. 'rho_1', exactly as returned by an earlier call in this session."


class McpToolError(gr.Error):
    """A user-facing problem; the MCP client receives it as an error result with this message."""

    def __init__(self, message: str):
        super().__init__(message, print_exception=False)

    def __str__(self):                      # gr.Error gives repr(message): the client would see it in quotes
        return self.message


def describe(summary: str, **params: str):
    """Give a wrapper the docstring Gradio turns into the MCP tool description: the summary, then one
    line 'name: text' per parameter (Gradio reads only single-line parameter descriptions)."""
    def deco(fn):
        lines = [" ".join(summary.split()), "", "Args:"]
        lines += [f"    {k}: {' '.join(v.split())}" for k, v in params.items()]
        fn.__doc__ = "\n".join(lines)
        return fn
    return deco


# ------------------------------------------------------------------ sessions, limits, results
def _throttle():
    if not CALLS_PER_MINUTE:
        return
    now = time.time()
    with _calls_lock:
        while _calls and now - _calls[0] > 60:
            _calls.popleft()
        if len(_calls) >= CALLS_PER_MINUTE:
            wait = int(60 - (now - _calls[0])) + 1
            raise McpToolError(f"Rate limit of this public server: {CALLS_PER_MINUTE} tool calls per minute. "
                               f"Try again in about {wait} s.")
        _calls.append(now)


def _open(session_id: str, create: bool) -> tuple[str, bool]:
    """(session id to use, whether it was just created)."""
    sid = (session_id or "").strip()
    if not sid:
        if not create:
            raise McpToolError("session_id is empty, so there are no stored states yet. Create a state first "
                               "(e.g. create_state) and pass the session_id it returns.")
        with _new_lock:
            session.make_room(PREFIX, MAX_MCP_SESSIONS)
            sid = PREFIX + uuid.uuid4().hex
            session.get(sid)
        return sid, True
    if not _ID.match(sid) or not session.exists(sid):
        raise McpToolError(f"Unknown or expired session_id '{sid[:80]}'. Sessions are deleted after 2 hours "
                           "without a call, and when the server restarts; their states are gone. Call again "
                           "with session_id='' to start a new session, then re-create the states.")
    return sid, False


def _finite(obj):
    """NaN or infinity cannot be sent as JSON: they become null (none is expected; see the tests)."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _public(obj, sid: str):
    """Internal folders never reach the client: 'data/sessions/<id>/x.csv' -> 'uploads/x.csv'."""
    if isinstance(obj, str):
        return session.display_path(obj).replace(f"figures/{session.SESSIONS_DIR}/{sid}/", "")
    if isinstance(obj, dict):
        return {k: _public(v, sid) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_public(v, sid) for v in obj]
    return obj


def _one_image(paths: list[str], fig_dir: str) -> str:
    """One PNG for the client: the figure itself, or several figures stacked vertically."""
    paths = [p for p in paths if p and os.path.isfile(p)]
    if not paths:
        raise McpToolError("The tool did not produce a figure.")
    if len(paths) == 1:
        return os.path.abspath(paths[0])
    from PIL import Image
    images = [Image.open(p).convert("RGB") for p in paths]
    width = max(i.width for i in images)
    out = Image.new("RGB", (width, sum(i.height for i in images)), "white")
    y = 0
    for im in images:
        out.paste(im, ((width - im.width) // 2, y))
        y += im.height
    path = os.path.join(fig_dir, f"combined_{uuid.uuid4().hex[:8]}.png")
    out.save(path)
    return os.path.abspath(path)


def _message(e: Exception) -> str:
    msg = str(e).strip()
    msg = re.sub(r"^Error invoking tool '[^']*' with kwargs .*? with error:\s*", "", msg, flags=re.S)
    return msg.split("\n Please fix your mistakes.")[0].strip() or type(e).__name__


def run(tool, session_id: str = "", *, stateful: bool = True, create: bool = True,
        figure: bool = False, **args):
    """Run one agent_tools tool for an MCP client. Returns the JSON text, plus the figure (gr.FileData)
    for figure tools."""
    _throttle()
    if stateful:
        sid, created = _open(session_id, create)
    else:
        sid, created = STATELESS_ID, False
    with session.use(sid) as s, session.COMPUTE_LOCK:
        at.CURRENT_QUESTION["text"] = ""         # the chat's question checks do not apply to an MCP call
        try:
            out = tool.invoke(args)
        except Exception as e:  # noqa: BLE001 - every tool error goes back to the client as its message
            raise McpToolError(_message(e)) from None
        result = {"result": out} if isinstance(out, str) else dict(out)
        if isinstance(out, str) and (m := _CREATED.match(out)):
            result["handle"] = m.group(1)
        figs = [result.pop(k) for k in FIGURE_KEYS if isinstance(result.get(k), str)]
        image = _one_image(figs, s.fig_dir) if figure else None
        if figs:
            result["figure"] = ("returned as a PNG image with this result"
                                + (f" ({len(figs)} figures stacked vertically)" if len(figs) > 1 else ""))
    if stateful:
        result = {"session_id": sid, **result}
        if created:
            result["session_note"] = ("New session opened. Pass this session_id in every later call that "
                                      "uses these states.")
    text = json.dumps(_finite(_public(result, sid)), ensure_ascii=False, allow_nan=False)
    return (text, gr.FileData(path=image)) if figure else text


# ------------------------------------------------------------------ session tools
@describe("List the states stored in this session: handle, number of photons, description, and whether the "
          "state was reconstructed or simulated from counts (then get_tomography_report, certify_correlations "
          "and plot_tomography_fit give error bars).", session_id=SESSION)
def list_states(session_id: str) -> str:
    _throttle()
    sid, _ = _open(session_id, create=False)
    with session.use(sid):
        states = [{"handle": h, "photons": 1 if at.STORE[h].shape[0] == 2 else 2, "description": at.DESC.get(h, ""),
                   "from_counts": h in session.current().fit}
                  for h in sorted(at.STORE, key=lambda h: int(h.split("_")[1]))]
    return json.dumps({"session_id": sid, "states": states}, ensure_ascii=False)


@describe("Delete this session now, with all its states and files (otherwise it is deleted after 2 hours "
          "without a call).", session_id=SESSION)
def end_session(session_id: str) -> str:
    _throttle()
    sid, _ = _open(session_id, create=False)
    session.drop(sid)
    return json.dumps({"session_id": sid, "result": "Session deleted."})


# ------------------------------------------------------------------ states
@describe("Create a two-photon polarization state (4x4 density matrix, basis HH, HV, VH, VV, photon A first) "
          "and store it; returns its handle. Bell states phi± = (|HH> ± |VV>)/√2, psi± = (|HV> ± |VH>)/√2. "
          "'werner': rho = p|B><B| + (1-p) I/4 with B = bell (default psi-; bell='phi+' gives a noisy phi+ "
          "with visibility p). Other states: create_pure_state, create_single_photon_state, mix_states.",
          kind="'phi+', 'phi-', 'psi+', 'psi-' or 'werner'.",
          p="Werner weight p in [0, 1] (only for kind='werner').",
          bell="Bell state B inside the Werner state (only for kind='werner').",
          session_id=SESSION)
def create_state(kind: StateKind, p: float = 1.0, bell: Bell = "psi-", session_id: str = "") -> str:
    return run(at.create_state, session_id, kind=kind, p=p, bell=bell)


@describe("Create a pure two-photon state |psi> = a_HH|HH> + a_HV|HV> + a_VH|VH> + a_VV|VV> (normalized "
          "automatically) and store it. Amplitudes are text evaluated exactly: numbers, + - * / ^, sqrt, cos, "
          "sin, tan, exp, pi, i - e.g. amp_HH='cos(0.3)', amp_VV='sin(0.3)', or amp_HH='1', amp_VV='i' for "
          "(|HH> + i|VV>)/√2. Omitted amplitudes are 0. Optional extra phase per component in degrees.",
          amp_HH="Amplitude of |HH> as text, e.g. '1/sqrt(3)'.", amp_HV="Amplitude of |HV> as text.",
          amp_VH="Amplitude of |VH> as text.", amp_VV="Amplitude of |VV> as text, e.g. 'sqrt(2/3)'.",
          phase_HH_deg="Extra phase of the |HH> amplitude, degrees.",
          phase_HV_deg="Extra phase of the |HV> amplitude, degrees.",
          phase_VH_deg="Extra phase of the |VH> amplitude, degrees.",
          phase_VV_deg="Extra phase of the |VV> amplitude, degrees.",
          session_id=SESSION)
def create_pure_state(amp_HH: str = "0", amp_HV: str = "0", amp_VH: str = "0", amp_VV: str = "0",
                      phase_HH_deg: float = 0.0, phase_HV_deg: float = 0.0, phase_VH_deg: float = 0.0,
                      phase_VV_deg: float = 0.0, session_id: str = "") -> str:
    return run(at.create_pure_state, session_id, amp_HH=amp_HH, amp_HV=amp_HV, amp_VH=amp_VH, amp_VV=amp_VV,
               phase_HH_deg=phase_HH_deg, phase_HV_deg=phase_HV_deg, phase_VH_deg=phase_VH_deg,
               phase_VV_deg=phase_VV_deg)


@describe("Create the polarization state of ONE photon (2x2 density matrix, basis H, V) and store it. Give "
          "exactly ONE of: polarization; azimuth_deg (+ ellipticity_deg); s1, s2, s3; amp_H and amp_V. "
          "Convention R = (H + iV)/√2.",
          polarization="'H', 'V', 'D', 'A', 'R' or 'L' (or empty when another form is used).",
          azimuth_deg="Polarization-ellipse azimuth from H, degrees.",
          ellipticity_deg="Ellipticity angle χ in [-45, 45] degrees: 0 linear, +45 = R, -45 = L.",
          dop="Degree of polarization in [0, 1] for the polarization, azimuth and amplitude forms: rho = dop|ψ><ψ| + (1 - dop) I/2.",
          s1="Normalized Stokes S1 = H - V (give s1, s2 and s3 together; length <= 1).",
          s2="Normalized Stokes S2 = D - A.", s3="Normalized Stokes S3 = R - L.",
          amp_H="Amplitude of |H> as text, e.g. '1' or 'cos(0.3)'.",
          amp_V="Amplitude of |V> as text, e.g. 'i' or 'sin(0.3)'.",
          session_id=SESSION)
def create_single_photon_state(polarization: Polarization = "", azimuth_deg: float | None = None,
                               ellipticity_deg: float = 0.0, dop: float = 1.0, s1: float | None = None,
                               s2: float | None = None, s3: float | None = None, amp_H: str = "",
                               amp_V: str = "", session_id: str = "") -> str:
    return run(at.create_single_photon_state, session_id, polarization=polarization, azimuth_deg=azimuth_deg,
               ellipticity_deg=ellipticity_deg, dop=dop, s1=s1, s2=s2, s3=s3, amp_H=amp_H, amp_V=amp_V)


@describe("Store the mixture Σ w_i rho_i of stored states (all with the same number of photons). Example: "
          "0.8 phi+ + 0.2 |HH><HH| - create both states first, then mix them.",
          handles="Handles of the stored states to mix, e.g. ['rho_1', 'rho_2'].",
          weights="Weights, same length as handles, each >= 0, summing to 1.",
          session_id=SESSION)
def mix_states(handles: list[str], weights: list[float], session_id: str) -> str:
    return run(at.mix_states, session_id, create=False, handles=handles, weights=weights)


@describe("Reduced polarization state of ONE photon of a stored two-photon state (partial trace over the "
          "other photon), stored as a new one-photon state. For a maximally entangled state it is I/2: "
          "entanglement shows up as local mixedness.",
          handle=HANDLE, photon="Which photon to keep: 'A' or 'B'.", session_id=SESSION)
def reduced_state(handle: str, photon: Photon, session_id: str) -> str:
    return run(at.reduced_state, session_id, create=False, handle=handle, photon=photon)


@describe("Store the two-photon product state rho_A ⊗ rho_B of two stored ONE-photon states (photon A "
          "first). A product state is never entangled.",
          handle_a="Handle of the one-photon state of photon A.",
          handle_b="Handle of the one-photon state of photon B.", session_id=SESSION)
def combine_photons(handle_a: str, handle_b: str, session_id: str) -> str:
    return run(at.combine_photons, session_id, create=False, handle_a=handle_a, handle_b=handle_b)


# ------------------------------------------------------------------ analysis
@describe("All standard measures of a stored state. Two photons: purity, von Neumann entropy and the entropies "
          "of each photon, conditional entropy, concurrence, entanglement of formation, negativity, "
          "log-negativity, maximal CHSH value (local bound 2, Tsirelson 2√2), l1 and relative-entropy coherence "
          "(H/V basis), mutual information and discord (all logarithms base 2). One photon: Stokes "
          "parameters, degree of polarization, purity, entropy, ellipse and analyser probabilities.",
          handle=HANDLE, session_id=SESSION)
def compute_measures(handle: str, session_id: str) -> str:
    return run(at.compute_measures, session_id, create=False, handle=handle)


@describe("The density matrix of a stored state as numbers: 4x4 (HH, HV, VH, VV, photon A first) or 2x2 "
          "(H, V), each element 'a + bi' to 4 decimals, with eigenvalues, trace and the nonzero elements. "
          "With compare_with, also that matrix and the element-wise change Δρ = ρ(handle) - ρ(compare_with). "
          "States reconstructed from counts get ± (1σ) per element.",
          handle=HANDLE,
          compare_with="Optional second state of the same size: a handle, or 'phi+', 'phi-', 'psi+', 'psi-'.",
          session_id=SESSION)
def show_density_matrix(handle: str, compare_with: str = "", session_id: str = "") -> str:
    return run(at.show_density_matrix, session_id, create=False, handle=handle, compare_with=compare_with)


@describe("Fidelity F = (Tr √(√ρ σ √ρ))² (squared convention, 1 = identical) and trace distance "
          "D = ½‖ρ - σ‖₁ between two stored states.",
          handle_a="Handle of the first state.", handle_b="Handle of the second state.", session_id=SESSION)
def compare_states(handle_a: str, handle_b: str, session_id: str) -> str:
    return run(at.compare_states, session_id, create=False, handle_a=handle_a, handle_b=handle_b)


# ------------------------------------------------------------------ noise
@describe("Apply a single-photon (local) noise channel and store the result (new handle). Parameter meaning: "
          "phase_damping: λ = coherence factor (1 = NO noise, 0 = full dephasing); amplitude_damping: γ = decay "
          "probability V -> H (0 = no noise); depolarizing: p, rho -> (1-p) rho + p I/2; bit_flip, "
          "phase_flip, bit_phase_flip: probability p of X, Z or Y. For a one-photon state qubits is ignored. "
          "Channels on the pair as a whole: apply_global_noise.",
          handle=HANDLE, channel="Local channel name.",
          param="Channel parameter in [0, 1] (see the meaning per channel above).",
          qubits="'both' (the same channel on each photon), 'A' or 'B'.", session_id=SESSION)
def apply_local_noise(handle: str, channel: LocalChannel, param: float, qubits: Qubits = "both",
                      session_id: str = "") -> str:
    return run(at.apply_local_noise, session_id, create=False, handle=handle, channel=channel, param=param,
               qubits=qubits)


@describe("Apply a GLOBAL two-photon noise channel (4x4 Kraus operators acting on the pair) to a stored "
          "two-photon state and store the result. p in [0, 1], 0 = no noise. global_depolarizing: rho -> "
          "(1-p) rho + p I/4; global_dephasing: every off-diagonal element x (1-p); collective_dephasing: the "
          "same random phase on both photons (psi± immune: decoherence-free); collective_amplitude_damping: "
          "common decay V -> H, p = 1 - e^(-γt) (psi- is dark, psi+ decays twice as fast); correlated_bit_flip, "
          "correlated_phase_flip, correlated_bit_phase_flip: X⊗X, Z⊗Z or Y⊗Y with probability p.",
          handle=HANDLE, channel="Global channel name.", param="Noise probability p in [0, 1].",
          session_id=SESSION)
def apply_global_noise(handle: str, channel: GlobalChannel, param: float, session_id: str) -> str:
    return run(at.apply_global_noise, session_id, create=False, handle=handle, channel=channel, param=param)


@describe("The Kraus operators of a noise channel as matrices (H/V basis), with the completeness check "
          "Σ K†K = I. Local channels give the 2x2 operators and how they act on `qubits`; global channels give "
          "the 4x4 operators on the pair (HH, HV, VH, VV). Needs no session.",
          channel="Local or global channel name (as in apply_local_noise / apply_global_noise).",
          param="Channel parameter (phase_damping: coherence factor λ, 1 = no noise; all others 0 = no noise).",
          qubits="For local channels: 'both', 'A' or 'B'.")
def get_kraus_operators(channel: AnyChannel, param: float, qubits: Qubits = "both") -> str:
    return run(at.get_kraus_operators, stateful=False, channel=channel, param=param, qubits=qubits)


@describe("Apply your OWN Kraus operators to a stored state: rho -> Σ K ρ K†, stored as a new state. 2x2 "
          "operators act on a one-photon state, or on photon 'A', 'B' or 'both' (all products) of a pair; 4x4 "
          "operators act on the pair (basis HH, HV, VH, VV). Σ K†K = I is checked; if it fails nothing is "
          "stored and the sum is reported.",
          handle=HANDLE,
          kraus="List of matrices, each a list of rows; entries are numbers or text such as 'sqrt(0.3)' or 'i', e.g. [[[1,0],[0,'sqrt(0.7)']],[[0,'sqrt(0.3)'],[0,0]]].",
          qubits="For 2x2 operators on a pair: 'both', 'A' or 'B'.", session_id=SESSION)
def apply_kraus_operators(handle: str, kraus: list, qubits: Qubits = "both", session_id: str = "") -> str:
    return run(at.apply_kraus_operators, session_id, create=False, handle=handle, kraus=kraus, qubits=qubits)


# ------------------------------------------------------------------ dynamics and optics
@describe("Continuous-time evolution of a stored state with the Lindblad master equation (ħ = 1, exact "
          "matrix exponential); stores the state at t_max and returns a plot of concurrence, negativity, "
          "purity and CHSH/2 versus time (one photon: degree of polarization, S1, S2, S3, purity) as a PNG "
          "image, plus 'features': crossing times (e.g. entanglement sudden death, CHSH = 2), the measures at "
          "those times and at evenly spaced samples. LOCAL processes (each photon its own environment, on "
          "`qubits`): dephasing, amplitude_decay, depolarizing, bit_flip, bit_phase_flip, none. GLOBAL "
          "(one common environment): global_depolarizing, global_dephasing, collective_dephasing (psi± "
          "immune), collective_decay (psi- dark), correlated_bit_flip, correlated_phase_flip. Optional local "
          "Hamiltonian H = (omega/2) σ_axis, e.g. a birefringent element.",
          handle=HANDLE, process="Process name (local or global, see above).",
          gamma="Rate γ (1/time unit).", t_max="Final time (> 0), same time unit as 1/γ.",
          qubits="For local processes: 'both', 'A' or 'B'.",
          omega="Hamiltonian strength ω (0 = no Hamiltonian).", axis="Hamiltonian axis: 'x', 'y' or 'z'.",
          plot_elements="Also plot every density-matrix element (Re, Im) versus time; stacked below the first figure.",
          session_id=SESSION)
def evolve_state(handle: str, process: Process, gamma: float, t_max: float, qubits: Qubits = "both",
                 omega: float = 0.0, axis: Literal["x", "y", "z"] = "x", plot_elements: bool = False,
                 session_id: str = "") -> tuple[str, gr.FileData]:
    return run(at.evolve_state, session_id, create=False, figure=True, handle=handle, process=process,
               gamma=gamma, t_max=t_max, qubits=qubits, omega=omega, axis=axis, plot_elements=plot_elements)


@describe("Send ONE photon of a stored two-photon state (or a one-photon state) through a wave plate and store "
          "the result. Jones matrix U = cos(δ/2) I - i sin(δ/2) (cos2θ σz + sin2θ σx), θ = fast-axis angle "
          "from H: HWP at 22.5° turns H into D, HWP at 45° swaps H and V, QWP at 45° turns H into (H - iV)/√2. "
          "Returns the action on H and V and the measures before and after (a local unitary never changes "
          "entanglement, purity or CHSH; fidelities and phases can change).",
          handle=HANDLE, plate="'HWP', 'QWP' or 'retarder' (then give retardance_deg).",
          angle_deg="Fast-axis angle from H, degrees.", qubit="Photon that passes the plate: 'A' or 'B'.",
          retardance_deg="Retardance δ in degrees, only for plate='retarder'.", session_id=SESSION)
def apply_waveplate(handle: str, plate: Plate, angle_deg: float, qubit: Photon, retardance_deg: float | None = None,
                    session_id: str = "") -> str:
    return run(at.apply_waveplate, session_id, create=False, handle=handle, plate=plate, angle_deg=angle_deg,
               qubit=qubit, retardance_deg=retardance_deg)


@describe("Wave-plate setting that removes the relative phase of a (nearly) maximally entangled stored state: "
          "the closest (|HH> + e^(iφ)|VV>)/√2 becomes phi+ (or (|HV> + e^(iφ)|VH>)/√2 becomes psi+) with a "
          "phase shifter diag(1, e^(iα)) built as QWP(45°) - HWP(θ) - QWP(45°) on one photon. Returns φ, α, θ, "
          "the fidelity before and after and the compensated state (new handle). It removes the phase error, "
          "not noise or mixedness.",
          handle=HANDLE, qubit="Photon that gets the phase shifter: 'A' or 'B'.", session_id=SESSION)
def compensate_phase(handle: str, qubit: Photon = "B", session_id: str = "") -> str:
    return run(at.compensate_phase, session_id, create=False, handle=handle, qubit=qubit)


# ------------------------------------------------------------------ tomography and certification
@describe("Two-photon state tomography from coincidence counts: Poisson maximum likelihood (always physical), "
          "with 1σ error bars from Poisson resampling; stores the state and returns its handle and report "
          "(fit quality, eigenvalues, closest maximally entangled state, measures with ±). Give the counts "
          "EITHER as csv_text (one row per setting: a column with settings such as HH, HV, ..., DA, RL - first "
          "letter photon A, letters H V D A R L - and one or more numeric count columns; a header row is "
          "optional) OR as csv_path: 'data/state_tomo_data.csv' (example data) or 'uploads/<file>' written "
          "earlier in this session (e.g. by simulate_tomography). At least the 16 settings of H, V, D, R are "
          "needed; all 36 are best.",
          csv_text="The CSV content itself (at most 1 MB), e.g. 'setting,counts\\nHH,1005\\nHV,12\\n...'.",
          csv_path="Instead of csv_text: 'data/state_tomo_data.csv' or 'uploads/<file>' from this session.",
          count_column="'auto' (the numeric column with the largest total) or a column name, e.g. 'L1R1'.",
          r_convention="Which circular state the analyser labelled R projects on.",
          error_bars="Compute the 1σ error bars (about 2 s).", session_id=SESSION)
def reconstruct_state_from_counts(csv_text: str = "", csv_path: str = "", count_column: str = "auto",
                                  r_convention: RConvention = "R=(H+iV)/sqrt2", error_bars: bool = True,
                                  session_id: str = "") -> str:
    if bool(csv_text.strip()) == bool(csv_path.strip()):
        raise McpToolError("Give exactly one of csv_text (the counts as CSV text) or csv_path.")
    sid = session_id
    if csv_text.strip():
        if len(csv_text.encode("utf-8")) > at.MAX_CSV_BYTES:
            raise McpToolError("csv_text is larger than 1 MB, too large for a two-qubit tomography.")
        sid, _ = _open(session_id, create=True)
        with session.use(sid) as s:
            os.makedirs(s.data_dir, exist_ok=True)
            n = 1 + sum(f.startswith("counts_") for f in os.listdir(s.data_dir))
            name = f"counts_{n}.csv"
            with open(os.path.join(s.data_dir, name), "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text.strip() + "\n")
        csv_path = f"{session.UPLOADS}/{name}"
    text = run(at.reconstruct_state_from_counts, sid, csv_path=csv_path, count_column=count_column,
               r_convention=r_convention, error_bars=error_bars)
    if sid != session_id and csv_text.strip() and not session_id.strip():     # opened above, not by run()
        out = json.loads(text)
        out["session_note"] = "New session opened. Pass this session_id in every later call that uses these states."
        text = json.dumps(out, ensure_ascii=False)
    return text


@describe("The stored tomography report of a state made by reconstruct_state_from_counts (fit quality, largest "
          "deviations, eigenvalues, closest maximally entangled state, uncertainties, assumptions), without "
          "reconstructing again.", handle=HANDLE, session_id=SESSION)
def get_tomography_report(handle: str, session_id: str) -> str:
    return run(at.get_tomography_report, session_id, create=False, handle=handle)


@describe("Simulate a polarization-tomography experiment on a stored two-photon state with realistic "
          "detectors: source pair rate, detection efficiencies, dark counts, coincidence window, counting time "
          "per setting; one detector behind each analyser, all 36 settings. Singles S = R·η·p + dark, "
          "accidentals = S_A·S_B·τ·T, Poisson counts. Reconstructs by maximum likelihood from the raw counts "
          "and with accidentals subtracted (both stored as new states), compares both with the true state, "
          "writes the counts to 'uploads/<file>' (usable with reconstruct_state_from_counts) and estimates the "
          "counting time needed for a target concurrence error.",
          handle=HANDLE, pair_rate="Pairs per second at the source (> 0).",
          time_per_setting_s="Counting time per setting, seconds (> 0).",
          eta_a="Detection efficiency of photon A, in (0, 1].", eta_b="Detection efficiency of photon B, in (0, 1].",
          dark_a="Dark counts of detector A per second (>= 0).", dark_b="Dark counts of detector B per second (>= 0).",
          window_ns="Coincidence window τ in nanoseconds (> 0).",
          target_concurrence_error="Target 1σ error of the concurrence for the counting-time estimate.",
          seed="Random seed (same seed, same counts).", session_id=SESSION)
def simulate_tomography(handle: str, pair_rate: float = 1e5, time_per_setting_s: float = 1.0, eta_a: float = 0.2,
                        eta_b: float = 0.2, dark_a: float = 500.0, dark_b: float = 500.0, window_ns: float = 2.0,
                        target_concurrence_error: float = 0.01, seed: int = 0, session_id: str = "") -> str:
    return run(at.simulate_tomography, session_id, create=False, handle=handle, pair_rate=pair_rate,
               time_per_setting_s=time_per_setting_s, eta_a=eta_a, eta_b=eta_b, dark_a=dark_a, dark_b=dark_b,
               window_ns=window_ns, target_concurrence_error=target_concurrence_error, seed=seed)


@describe("One-photon polarization tomography from counts behind H, V, D, A, R, L analysers (same counting "
          "time each; at least H, V, D and R, all six best): Poisson maximum likelihood. Stores the state and "
          "returns Stokes parameters, degree of polarization and purity with ± (1σ, Poisson resampling) and "
          "the linear-inversion Stokes parameters.",
          counts_H="Counts behind the H analyser.", counts_V="Counts behind V.", counts_D="Counts behind D.",
          counts_A="Counts behind A.", counts_R="Counts behind R.", counts_L="Counts behind L.",
          r_convention="Which circular state the analyser labelled R projects on.", session_id=SESSION)
def reconstruct_single_photon(counts_H: float | None = None, counts_V: float | None = None,
                              counts_D: float | None = None, counts_A: float | None = None,
                              counts_R: float | None = None, counts_L: float | None = None,
                              r_convention: RConvention = "R=(H+iV)/sqrt2", session_id: str = "") -> str:
    return run(at.reconstruct_single_photon, session_id, counts_H=counts_H, counts_V=counts_V, counts_D=counts_D,
               counts_A=counts_A, counts_R=counts_R, counts_L=counts_L, r_convention=r_convention)


@describe("Diagnostics plot of a state reconstructed or simulated from counts (PNG image): measured vs fitted "
          "counts per setting, residuals in standard deviations, and the bootstrap distributions of "
          "concurrence and CHSH with their bounds.", handle=HANDLE, session_id=SESSION)
def plot_tomography_fit(handle: str, session_id: str) -> tuple[str, gr.FileData]:
    return run(at.plot_tomography_fit, session_id, create=False, figure=True, handle=handle)


@describe("Certify entanglement, EPR steering and Bell nonlocality of a stored two-photon state: fidelity "
          "witness and optimal PPT witness, the 3-setting linear steering inequality F3 = (t1 + t2 + t3)/√3 > 1 "
          "(plus the exact criterion for unpolarized photons, e.g. Werner states) and CHSH max > 2. For a state "
          "reconstructed or simulated from counts, each test also gets its significance in standard deviations.",
          handle=HANDLE, session_id=SESSION)
def certify_correlations(handle: str, session_id: str) -> str:
    return run(at.certify_correlations, session_id, create=False, handle=handle)


# ------------------------------------------------------------------ protocols
@describe("A stored two-photon state as the resource for teleporting one qubit (standard protocol, best local "
          "correction): fully entangled fraction f (exact), average fidelity F = (2f + 1)/3, the classical "
          "limit 2/3 and whether the state beats it (iff f > 1/2).", handle=HANDLE, session_id=SESSION)
def teleportation_fidelity(handle: str, session_id: str) -> str:
    return run(at.teleportation_fidelity, session_id, create=False, handle=handle)


@describe("Asymptotic secret-key fraction of entanglement-based BB84 / BBM92, r = 1 - h(e_bit) - h(e_phase) "
          "(h = binary entropy; ideal single pairs, infinite key, one-way error correction at the Shannon "
          "limit). Give EITHER a stored state (QBERs computed from it: key from H/V, phase errors from D/A, R/L "
          "also reported; needs session_id) OR qber_z (and optionally qber_x, default = qber_z). Equal QBERs: "
          "r = 1 - 2h(Q) > 0 only for Q < 11.0%.",
          handle="Handle of a stored state, or empty when QBERs are given.",
          qber_z="Bit error rate in the key basis, in [0, 0.5].",
          qber_x="Phase error rate (other basis); default = qber_z.",
          session_id="Needed only with handle. " + SESSION)
def qkd_key_rate(handle: str = "", qber_z: float | None = None, qber_x: float | None = None,
                 session_id: str = "") -> str:
    if handle.strip():
        return run(at.qkd_key_rate, session_id, create=False, handle=handle, qber_z=qber_z, qber_x=qber_x)
    return run(at.qkd_key_rate, stateful=False, handle="", qber_z=qber_z, qber_x=qber_x)


@describe("Three-qubit bit-flip repetition code with majority vote against independent bit flips of "
          "probability p per qubit: logical error p_L = 3p² - 2p³ versus p unencoded; the code helps iff "
          "p < 1/2 (it does not protect against phase flips). Needs no session.",
          p="Bit-flip probability per qubit, in [0, 1].")
def bit_flip_code(p: float) -> str:
    return run(at.bit_flip_code, stateful=False, p=p)


# ------------------------------------------------------------------ plots (each returns a PNG image)
@describe("Plot a stored density matrix (PNG image) and return how to read it and its largest elements. "
          "Styles: bars3d (3D bars of Re and Im), heatmap (Re and Im grids with every value written in), "
          "magnitude_phase (|ρ_ij| as bar height, phase as colour - best to see a relative phase). With "
          "compare_with, a second row shows that state and the fidelity between the two.",
          handle=HANDLE, style="'bars3d', 'heatmap' or 'magnitude_phase'.",
          compare_with="Optional second state: a handle, or 'phi+', 'phi-', 'psi+', 'psi-'.", session_id=SESSION)
def plot_density_matrix(handle: str, style: DMStyle = "bars3d", compare_with: str = "",
                        session_id: str = "") -> tuple[str, gr.FileData]:
    return run(at.plot_density_matrix, session_id, create=False, figure=True, handle=handle, style=style,
               compare_with=compare_with)


@describe("Plot concurrence, negativity, purity and CHSH/2 of one Bell state as the noise parameter of a "
          "channel goes from 0 to 1 (PNG image); returns 'features': exact parameter values where "
          "entanglement vanishes and where CHSH drops to 2, the measures there, at no and maximum noise and "
          "at 11 samples. To compare several or stored states, or choose the measures, use plot_sweep.",
          start_kind="Bell state: 'phi+', 'phi-', 'psi+' or 'psi-'.",
          channel="Local or global channel name (phase_damping: x axis is λ, 1 = no noise).",
          n_points="Grid points, 11 to 201.", qubits="For local channels: 'both', 'A' or 'B'.",
          session_id=SESSION)
def plot_measures_vs_noise(start_kind: Bell, channel: AnyChannel, n_points: int = 41, qubits: Qubits = "both",
                           session_id: str = "") -> tuple[str, gr.FileData]:
    if not 11 <= int(n_points) <= 201:
        raise McpToolError("n_points must be between 11 and 201.")
    return run(at.plot_measures_vs_noise, session_id, figure=True, start_kind=start_kind, channel=channel,
               n_points=int(n_points), qubits=qubits)


@describe("Plot measures against a parameter (one panel per measure, up to 3 states compared; PNG image). "
          "Returns per state and measure the values at both ends, min/max, the trend, the exact parameter "
          "values where it crosses its bound (e.g. CHSH = 2) and 11 samples. Sweeps: werner_p (Werner state "
          "p|B><B| + (1-p) I/4, B = bell), pure_angle (cos θ|HH> + sin θ|VV>, θ in radians), visibility "
          "(v ρ + (1-v) I/4 for `handles`), noise (`channel` with parameter p on `qubits`, for `handles`), "
          "waveplate_angle (`plate` at θ degrees on photon `qubit`, for `handles`), phase_shift (diag(1, e^(iφ)), "
          "φ in degrees, on photon `qubit`, for `handles`).",
          sweep="The x axis (see above).",
          handles="States for visibility, noise, waveplate_angle and phase_shift: 1 to 3 handles or Bell names such as 'phi+'.",
          quantities="Up to 6 measures; default concurrence, negativity, purity, chsh_max. 'fidelity' needs `reference`.",
          start="Start of the sweep range (a default exists for every sweep).",
          stop="End of the sweep range (a default exists for every sweep).",
          bell="Bell state B of the werner_p sweep.", channel="Channel name for sweep='noise' (local or global).",
          qubits="Photons for a local channel: 'both', 'A' or 'B'.",
          plate="'HWP' or 'QWP' for sweep='waveplate_angle'.",
          qubit="Photon for waveplate_angle and phase_shift: 'A' or 'B'.",
          reference="Reference state for 'fidelity': a handle or 'phi+', 'phi-', 'psi+', 'psi-'.",
          n_points="Grid points, 11 to 401 (discord: at most 41).", session_id=SESSION)
def plot_sweep(sweep: Sweep, handles: list[str] | None = None, quantities: list[Quantity] | None = None,
               start: float | None = None, stop: float | None = None, bell: Bell = "psi-", channel: str = "",
               qubits: Qubits = "both", plate: Literal["HWP", "QWP"] = "HWP", qubit: Photon = "B",
               reference: str = "", n_points: int = 101, session_id: str = "") -> tuple[str, gr.FileData]:
    return run(at.plot_sweep, session_id, figure=True, sweep=sweep, handles=handles or [],
               quantities=quantities or [], start=start, stop=stop, bell=bell, channel=channel, qubits=qubits,
               plate=plate, qubit=qubit, reference=reference, n_points=int(n_points))


@describe("Bloch vectors of each photon's reduced state (spheres for photon A and B) and the 3x3 correlation "
          "matrix T_ij = <σ_i ⊗ σ_j> (i, j = X (D/A), Y (R/L), Z (H/V)) of up to 3 stored two-photon states "
          "in one figure (PNG image); returns the vectors, T, its singular values, CHSH max = 2√(t1² + t2²) "
          "and steering F3 = (t1 + t2 + t3)/√3. One-photon states are drawn on one Poincaré sphere instead.",
          handles="1 to 3 handles, all two-photon or all one-photon.", session_id=SESSION)
def plot_bloch_and_correlations(handles: list[str], session_id: str) -> tuple[str, gr.FileData]:
    return run(at.plot_bloch_and_correlations, session_id, create=False, figure=True, handles=handles)


@describe("Polarization-correlation fringes of a stored two-photon state (PNG image): coincidence probability "
          "versus the angle of photon B's linear polarizer, for photon A's polarizer at each of alice_angles. "
          "Returns the exact visibility and the angle of the maximum of every fringe (phi+ gives visibility 1 "
          "in every basis).", handle=HANDLE,
          alice_angles="1 to 4 polarizer angles of photon A, degrees from H; default [0, 45, 90, 135] (H, D, V, A).",
          session_id=SESSION)
def plot_fringes(handle: str, alice_angles: list[float] | None = None, session_id: str = "") -> tuple[str, gr.FileData]:
    return run(at.plot_fringes, session_id, create=False, figure=True, handle=handle,
               alice_angles=alice_angles or [0.0, 45.0, 90.0, 135.0])


TOOLS = [list_states, end_session,
         create_state, create_pure_state, create_single_photon_state, mix_states, reduced_state, combine_photons,
         compute_measures, show_density_matrix, compare_states,
         apply_local_noise, apply_global_noise, get_kraus_operators, apply_kraus_operators,
         evolve_state, apply_waveplate, compensate_phase,
         reconstruct_state_from_counts, get_tomography_report, simulate_tomography, reconstruct_single_photon,
         plot_tomography_fit, certify_correlations,
         teleportation_fidelity, qkd_key_rate, bit_flip_code,
         plot_density_matrix, plot_measures_vs_noise, plot_sweep, plot_bloch_and_correlations, plot_fringes]
NAMES = [f.__name__ for f in TOOLS]


def register():
    """Add every MCP tool as an API endpoint of the current gr.Blocks. queue=False: Gradio's fast MCP path
    (no HTTP loopback) - the only one that returns a figure tool's two outputs (JSON text, PNG image) as
    two separate items; work is serialized by session.COMPUTE_LOCK instead of the queue."""
    for fn in TOOLS:
        gr.api(fn, api_name=fn.__name__, queue=False)


def restrict_api(demo: gr.Blocks) -> list[str]:
    """Default deny: every endpoint of the app that is not one of TOOLS becomes private (not in the API or
    the MCP server; the browser UI still uses it). Catches events Gradio creates by itself, e.g. the load
    event of gr.State(new_session_id). Returns the endpoints it made private."""
    hidden = []
    for fn in demo.fns.values():
        if fn.api_name and fn.api_name not in NAMES and fn.api_visibility != "private":
            fn.api_visibility = "private"
            hidden.append(fn.api_name)
    return hidden


def server_url(base: str) -> str:
    return base.rstrip("/") + "/gradio_api/mcp/"
