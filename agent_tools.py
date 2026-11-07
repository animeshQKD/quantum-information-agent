"""
agent_tools.py - the tools the LLM is allowed to call.

Each tool is a thin wrapper around a tested function in qi_tools.py.
Density matrices never pass through the LLM: they live in STORE and the LLM
only sees short handles such as "rho_1".
"""

from __future__ import annotations

FILES_VERSION = "2026-11-02"   # must match in graph.py, agent_tools.py, llm.py, run.py

import ast
import cmath
import math
import operator as op
import os
import re
import threading
from typing import Annotated

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no GUI window; save PNGs instead
import matplotlib.pyplot as plt
from langchain_core.tools import tool
from pydantic import BaseModel, BeforeValidator

import qi_tools as qt
import session
import plots as pl                         # one plotting style for every figure (sets rcParams)

# Per-user state (session.py): each web-app user has their own; the command line uses one default.
STORE = session.SessionDict("store")       # handle -> density matrix
DESC = session.SessionDict("desc")         # handle -> short description (shown to the LLM)
TOMO = session.SessionDict("tomo")         # handle -> full tomography report (reconstructed states)
_FIT = session.SessionDict("fit")          # handle -> fitted counts, for bootstrap error bars
_SAMPLES = session.SessionDict("samples")  # handle -> bootstrap states (same ones as the report's ±)
_LOCK = threading.Lock()                   # parallel tool calls must not reuse a handle
BELL_GREEK = {"phi+": "φ⁺", "phi-": "φ⁻", "psi+": "ψ⁺", "psi-": "ψ⁻"}   # for titles: |ψ⁻⟩ reads better than |psi-⟩
FIG_DIR = "figures"                        # of the default session; web sessions: figures/sessions/<id>
CURRENT_QUESTION = session.SessionDict("question")   # {'text': the user's latest question} (graph.py)


def _fig_dir() -> str:
    d = session.current().fig_dir
    os.makedirs(d, exist_ok=True)
    return d


def _data_dir() -> str:
    d = session.current().data_dir
    os.makedirs(d, exist_ok=True)
    return d
# an amplitude may arrive as a number (0.6) or as text ("cos(0.3)"); keep it as text
_Amp = Annotated[str, BeforeValidator(lambda v: str(v))]

_LABEL_PATTERNS = {"HH": r"(?<![a-z])hh(?![a-z])|\|00|00>", "HV": r"(?<![a-z])hv(?![a-z])|\|01|01>",
                   "VH": r"(?<![a-z])vh(?![a-z])|\|10|10>", "VV": r"(?<![a-z])vv(?![a-z])|\|11|11>"}


def _check_basis_in_question(labels: list[str]) -> None:
    """Refuse to build a state whose basis components the user never mentioned (no guessing)."""
    q = CURRENT_QUESTION["text"].lower()
    if not q:
        return
    missing = [k for k in labels if not re.search(_LABEL_PATTERNS[k], q)]
    if missing:
        raise ValueError(
            f"The user's message does not mention the basis state(s) {missing}. Do not guess: "
            "ask the user to write the full state, e.g. cos(0.3)|HH> + sin(0.3)|VV>.")


_BELL_NAMES = {"phi+": r"phi\s*\+|φ\s*[+⁺]", "phi-": r"phi\s*-|φ\s*[-⁻−]",
               "psi+": r"psi\s*\+|ψ\s*[+⁺]", "psi-": r"psi\s*-|ψ\s*[-⁻−]"}


_REFERS_BACK = re.compile(r"\bsame\b|previous|earlier|before|\bthis state\b|\bthat state\b|\bit\b|"
                          r"rho_\d+|reconstruct|result|above|final state|\blab\b|measured|\.csv|"
                          r"experiment|simulated", re.I)


def _check_named_bell_state(handle: str) -> None:
    """Refuse to act on a state the user did not ask for: the question names a Bell state (e.g.
    'Evolve phi+ ...') but the handle is a different state (rho_4 = psi-, or rho_1 = some other
    pure state). Allowed when the question names the handle or refers back to an earlier state."""
    desc = DESC.get(handle, "").strip().lower()
    q = CURRENT_QUESTION["text"]
    if not q or re.search(rf"\b{handle}\b", q):
        return                                   # the user named this handle explicitly: fine
    named = [k for k, pat in _BELL_NAMES.items() if re.search(pat, q, flags=re.I)]
    if not named or desc in named or any(n in desc for n in named):
        return                                   # e.g. 'werner, p=0.8, built on phi+' 
    if desc not in _BELL_NAMES and _REFERS_BACK.search(q):
        return                                   # "the same state", "the reconstructed state", ...
    raise ValueError(f"{handle} is '{desc}', but the user asked about {', '.join(named)}. "
                     f"Create it first with create_state(kind='{named[0]}') and use the new handle.")


def _save(rho: np.ndarray, desc: str = "") -> str:
    if not qt.is_valid_density_matrix(rho, tol=1e-8):
        raise ValueError("Result is not a valid density matrix (check the parameters).")
    if len(STORE) >= session.MAX_STATES:
        raise ValueError(f"This session already holds {len(STORE)} states (the limit). Start a new chat "
                         "or reload the page to begin a fresh session.")
    with _LOCK:
        handle = f"rho_{len(STORE) + 1}"
        STORE[handle] = rho
        DESC[handle] = desc
        TOMO.pop(handle, None)              # a reused handle must not keep an old report
        _FIT.pop(handle, None)
        _SAMPLES.pop(handle, None)
    return handle


def _clean(obj):
    """Make a tool result JSON-safe: numpy numbers -> Python numbers (else the model would see
    text like 'np.float64(1.9297)')."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _get(handle: str) -> np.ndarray:
    if handle not in STORE:
        known = "; ".join(f"{h} = {DESC.get(h, '')[:70]}" for h in STORE) or "none"
        key = str(handle).strip().lower().replace(" ", "")
        hint = (f" '{handle}' is a state name, not a handle: create it with create_state(kind='{key}') or use a "
                "stored handle that is that state." if key in ("phi+", "phi-", "psi+", "psi-") else
                " A new state's handle is in the result of the tool that creates it: wait for that result, then "
                "call again with it." if re.match(r"rho_\d+_", str(handle)) else "")
        raise ValueError(f"Unknown handle '{handle}'.{hint} Use one of these exactly (do not invent names): {known}")
    return STORE[handle]


ONE_PHOTON_TOOLS = ("compute_measures, show_density_matrix, apply_local_noise, apply_kraus_operators, evolve_state, "
                    "apply_waveplate, compare_states, mix_states, plot_density_matrix, plot_bloch_and_correlations")


def _photons(rho: np.ndarray) -> int:
    """1 for a one-photon (2x2) state, 2 for a photon pair (4x4)."""
    return 1 if rho.shape[0] == 2 else 2


def _get2(handle: str, what: str = "This tool") -> np.ndarray:
    """A stored TWO-photon state; a clear message for a one-photon state."""
    rho = _get(handle)
    if rho.shape[0] != 4:
        raise ValueError(f"{handle} is a ONE-photon state ({DESC.get(handle, '')}). {what} needs a two-photon "
                         f"state. For one photon use {ONE_PHOTON_TOOLS}; to make a pair use combine_photons.")
    return rho


def _same_size(handles: list[str]) -> list[np.ndarray]:
    rhos = [_get(h) for h in handles]
    if len({r.shape[0] for r in rhos}) > 1:
        kinds = ", ".join(f"{h}: {_photons(r)} photon{'s' if _photons(r) == 2 else ''}" for h, r in zip(handles, rhos))
        raise ValueError(f"These states have different sizes ({kinds}): a one-photon state can only be "
                         "compared or mixed with another one-photon state.")
    return rhos


# ------------------------------------------------------------------ channels
# name -> (Kraus maker, meaning of the parameter, value of the parameter that means "no noise")
CHANNELS = {
    "phase_damping":     (qt.kraus_phase_damping,     "coherence factor λ (1 = no noise, 0 = full dephasing)", 1.0),
    "amplitude_damping": (qt.kraus_amplitude_damping, "decay probability γ (0 = no noise)", 0.0),
    "depolarizing":      (qt.kraus_depolarizing,      "depolarizing probability p (0 = no noise)", 0.0),
    "bit_flip":          (qt.kraus_bit_flip,          "bit-flip probability p (0 = no noise)", 0.0),
    "phase_flip":        (qt.kraus_phase_flip,        "phase-flip probability p (0 = no noise)", 0.0),
    "bit_phase_flip":    (qt.kraus_bit_phase_flip,    "bit-phase-flip probability p (0 = no noise)", 0.0),
}
QUBITS = {"both": "both qubits", "A": "qubit A only", "B": "qubit B only"}

# GLOBAL channels: 4x4 Kraus operators acting on the PAIR (not photon by photon). Parameter p, 0 = no noise.
GLOBAL_CHANNELS = {
    "global_depolarizing": (qt.kraus_global_depolarizing,
                            "p: rho -> (1-p) rho + p I/4 (the pair is replaced by white noise with probability p)"),
    "global_dephasing": (qt.kraus_global_dephasing,
                         "p: every off-diagonal element in the HH, HV, VH, VV basis is multiplied by (1-p)"),
    "collective_dephasing": (qt.kraus_collective_dephasing,
                             "p: both photons get the SAME random phase; elements between states with one more V "
                             "x (1-p), with two more V x (1-p)^4; HV-VH coherence (psi+/psi-) unchanged"),
    "collective_amplitude_damping": (qt.kraus_collective_amplitude_damping,
                                     "p = 1 - e^(-γt): both photons decay V -> H through ONE common channel "
                                     "(jump operator σ-⊗I + I⊗σ-); psi- never decays, psi+ decays twice as fast"),
    "correlated_bit_flip": (lambda p: qt.kraus_correlated_pauli(qt.X, p),
                            "p: with probability p BOTH photons are flipped (X⊗X); Bell states are unchanged"),
    "correlated_phase_flip": (lambda p: qt.kraus_correlated_pauli(qt.Z, p),
                              "p: with probability p BOTH photons get Z (Z⊗Z); Bell states are unchanged"),
    "correlated_bit_phase_flip": (lambda p: qt.kraus_correlated_pauli(qt.Y, p),
                                  "p: with probability p BOTH photons get Y (Y⊗Y); Bell states are unchanged"),
}
GLOBAL_ALIASES = {"two_qubit_depolarizing": "global_depolarizing", "global_phase_damping": "global_dephasing",
                  "collective_amplitude_decay": "collective_amplitude_damping",
                  "collective_decay": "collective_amplitude_damping", "global_amplitude_damping":
                  "collective_amplitude_damping", "correlated_dephasing": "collective_dephasing"}
NOISE_HELP = {**{k: v[1] for k, v in CHANNELS.items()}, **{k: v[1] for k, v in GLOBAL_CHANNELS.items()}}


def _channel_name(channel: str) -> str:
    c = channel.strip().lower().replace(" ", "_").replace("-", "_")
    return GLOBAL_ALIASES.get(c, c)


def _channel_kraus(channel: str, param: float, qubits: str, photons: int = 2):
    """Kraus operators of a local channel (on 'both', 'A' or 'B'; on the photon itself for a one-photon
    state) or of a global channel (on the pair; `qubits` is ignored)."""
    channel = _channel_name(channel)
    if channel not in CHANNELS and channel not in GLOBAL_CHANNELS:
        raise ValueError(f"channel must be one of {list(CHANNELS)} (local) or {list(GLOBAL_CHANNELS)} (global)")
    if not 0 <= param <= 1:
        raise ValueError("Channel parameter must be in [0, 1].")
    if channel in GLOBAL_CHANNELS:
        if photons == 1:
            raise ValueError(f"'{channel}' is a two-photon (global) channel; a one-photon state takes the local "
                             f"channels {list(CHANNELS)}.")
        return GLOBAL_CHANNELS[channel][0](param)
    K = CHANNELS[channel][0](param)
    if photons == 1:
        return K
    if qubits not in QUBITS:
        raise ValueError("qubits must be 'both', 'A' or 'B'")
    I = [qt.I2]
    return {"both": qt.local_channel(K, K), "A": qt.local_channel(K, I),
            "B": qt.local_channel(I, K)}[qubits]


# ------------------------------------------------------------------ tools
@tool
def create_state(kind: str, p: float = 1.0, bell: str = "psi-") -> str:
    """Create a two-qubit state and return its handle.
    kind: 'phi+', 'phi-', 'psi+', 'psi-' (Bell states) or 'werner'.
    For 'werner': rho = p|B><B| + (1-p) I/4 with p in [0, 1] and B = `bell`
    (default 'psi-', the standard Werner state; e.g. bell='phi+' gives a noisy phi+ with
    visibility p). For other shapes use create_pure_state or mix_states."""
    kind, bell = kind.lower().replace(" ", ""), bell.lower().replace(" ", "")
    if kind not in ("phi+", "phi-", "psi+", "psi-", "werner"):
        raise ValueError("kind must be 'phi+', 'phi-', 'psi+', 'psi-' or 'werner' "
                         "(for other states use create_pure_state or mix_states).")
    if bell not in ("phi+", "phi-", "psi+", "psi-"):
        raise ValueError("bell must be 'phi+', 'phi-', 'psi+' or 'psi-'.")
    if kind == "werner":
        if not 0 <= p <= 1:
            raise ValueError("Werner parameter p must be in [0, 1].")
        rho = qt.werner_state(p, bell)
        desc = f"werner, p={p}" + (f", built on {bell}" if bell != "psi-" else "")
    else:
        rho = qt.dm(qt.bell_state(kind))
        desc = kind
    return f"Created {_save(rho, desc)} ({desc})."


# ------------------------------------------------------------------ amplitudes
_FUNCS = {"sqrt": cmath.sqrt, "cos": cmath.cos, "sin": cmath.sin, "tan": cmath.tan, "exp": cmath.exp}
_NAMES = {"i": 1j, "pi": math.pi}
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow}


def eval_amplitude(text) -> complex:
    """Safely evaluate an amplitude such as 0.6, 'cos(0.3)', '1/sqrt(3)', '√(2/3)', 'i/√2',
    'exp(i*pi/4)/sqrt(2)'. Only numbers, + - * / ^, sqrt/cos/sin/tan/exp, pi and i are allowed."""
    s = str(text).strip().lower().replace("**", "^").replace("−", "-").replace("·", "*")
    if not s:
        return 0j
    if len(s) > 80:
        raise ValueError(f"amplitude '{text}' is too long")
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[a-z]+)", r"sqrt(\1)", s).replace("√(", "sqrt(").replace("√", "sqrt")
    s = re.sub(r"(?<=[0-9)])\s*(?=[a-z(])", "*", s)      # implicit products: 2i, 2cos(x), )(
    s = re.sub(r"(?<![a-z])i(?=\s*[a-z(0-9])", "i*", s)   # i/sqrt(2) ok; i sqrt(2) -> i*sqrt(2)
    s = s.replace("^", "**")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return complex(n.value)
        if isinstance(n, ast.Name) and n.id in _NAMES:
            return complex(_NAMES[n.id])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 10:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS
                and len(n.args) == 1 and not n.keywords):
            return complex(_FUNCS[n.func.id](ev(n.args[0])))
        raise ValueError("unsupported")

    try:
        return ev(ast.parse(s, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, OverflowError):
        raise ValueError(f"Cannot read the amplitude '{text}'. Use a number or an expression "
                         "with sqrt, cos, sin, exp, pi and i, e.g. 'cos(0.3)' or '1/sqrt(2)'.") from None


_KET = re.compile(r"\|\s*(hh|hv|vh|vv|00|01|10|11)\s*(?:>|⟩)")
_LABEL = {"hh": "HH", "hv": "HV", "vh": "VH", "vv": "VV", "00": "HH", "01": "HV", "10": "VH", "11": "VV"}
_EXPR_TAIL = re.compile(r"(?:sqrt|cos|sin|tan|exp|pi|i|√|[0-9.()*/^\s])*$")


def state_from_question(text: str):
    """Best-effort reading of a two-qubit pure state written in the user's text, e.g.
    'cos(0.3)|HH> + sin(0.3)|VV>' or '(|HH> + i|VV>)/sqrt(2)'.
    Returns ({label: coefficient-text}, normalized vector) or None if it cannot be read safely."""
    t = text.lower()
    kets = list(_KET.finditer(t))
    if not kets:
        return None
    labels = [_LABEL[k.group(1)] for k in kets]
    if len(set(labels)) != len(labels):          # e.g. two different states in one message
        return None
    coeffs, vec, prev_end = {}, np.zeros(4, dtype=complex), 0
    for k, lab in zip(kets, labels):
        prefix = t[prev_end:k.start()]
        prev_end = k.end()
        m = _EXPR_TAIL.search(prefix)
        raw, before = m.group(0), prefix[:m.start()]
        depth, cut = 0, None                      # drop everything up to the last unmatched '('
        for ch in raw:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return None                  # e.g. '(1+i)/2|HH>' - too complex, skip
        if depth > 0:
            depth = 0
            for pos in range(len(raw) - 1, -1, -1):
                if raw[pos] == ")":
                    depth += 1
                elif raw[pos] == "(":
                    if depth == 0:
                        cut = pos
                        break
                    depth -= 1
            before, raw = before + raw[:cut + 1], raw[cut + 1:]
        prefix = before + raw
        expr = raw.strip()
        start = len(prefix) - len(raw.lstrip())  # where the coefficient starts in `prefix`
        glued = prefix[start - 1] if start > 0 else " "
        if glued.isalnum() or glued in "_.":
            return None                          # symbolic coefficient like a|HH> - cannot check
        sign = -1 if prefix[:start].rstrip().endswith("-") else 1
        try:
            c = sign * (eval_amplitude(expr) if expr else 1)
        except ValueError:
            return None
        coeffs[lab] = ("-" if sign < 0 else "") + (expr or "1")
        vec[("HH", "HV", "VH", "VV").index(lab)] = c
    n = np.linalg.norm(vec)
    return (coeffs, vec / n) if n > 1e-12 else None


def _check_against_question(psi: np.ndarray) -> None:
    """If the user's message spells out the state, the tool input must be that state."""
    if re.search(r"phase|degree|°|\bdeg\b", CURRENT_QUESTION["text"].lower()):
        return                                   # extra phases given in words: cannot check the text
    parsed = state_from_question(CURRENT_QUESTION["text"])
    if parsed is None:
        return
    coeffs, target = parsed
    used = {k for k, c in zip(("HH", "HV", "VH", "VV"), psi) if abs(c) > 1e-9}
    if used != set(coeffs):
        return                                   # different components: the other guard decides
    if abs(np.vdot(target, psi)) ** 2 < 0.999:
        wrote = ", ".join(f"{k}: {coeffs[k]}" for k in coeffs)
        raise ValueError(
            f"These amplitudes do not give the state in the user's message ({wrote}). "
            "Pass each amplitude exactly as the user wrote it, as text, e.g. "
            "amp_HH='cos(0.3)', amp_VV='sin(0.3)', amp_VV='i/sqrt(2)'.")


def _phase_text(deg: float, digits: int) -> str:
    """'' for no phase, else '·e^(i90.0°)' or '·e^(-i108°)'."""
    if abs(deg) < 0.5 * 10 ** -digits:
        return ""
    sign = "-" if deg < 0 else ""
    return f"·e^({sign}i{abs(deg):.{digits}f}°)"


class _PureStateArgs(BaseModel):
    amp_HH: _Amp = "0"
    amp_HV: _Amp = "0"
    amp_VH: _Amp = "0"
    amp_VV: _Amp = "0"
    phase_HH_deg: float = 0
    phase_HV_deg: float = 0
    phase_VH_deg: float = 0
    phase_VV_deg: float = 0


@tool(args_schema=_PureStateArgs)
def create_pure_state(amp_HH="0", amp_HV="0", amp_VH="0", amp_VV="0",
                      phase_HH_deg: float = 0, phase_HV_deg: float = 0,
                      phase_VH_deg: float = 0, phase_VV_deg: float = 0) -> str:
    """Create a general pure two-qubit state
    |psi> = a_HH|HH> + a_HV|HV> + a_VH|VH> + a_VV|VV>   (normalized automatically).
    Give each amplitude EXACTLY as the user wrote it, as text - do not evaluate it yourself:
      cos(0.3)|HH> + sin(0.3)|VV>    ->  amp_HH='cos(0.3)', amp_VV='sin(0.3)'
      1/sqrt(3)|HH> + sqrt(2/3)|VV>  ->  amp_HH='1/sqrt(3)', amp_VV='sqrt(2/3)'
      (|HH> + i|VV>)/sqrt(2)         ->  amp_HH='1', amp_VV='i'   (overall factors can be dropped)
    Allowed: numbers, + - * / ^, sqrt, cos, sin, exp, pi, i. Optional extra phases in degrees.
    Components not mentioned by the user are 0."""
    given = {"HH": amp_HH, "HV": amp_HV, "VH": amp_VH, "VV": amp_VV}
    phases = [phase_HH_deg, phase_HV_deg, phase_VH_deg, phase_VV_deg]
    c = np.array([eval_amplitude(given[k]) for k in ("HH", "HV", "VH", "VV")])
    c = c * np.exp(1j * np.deg2rad(phases))
    _check_basis_in_question([k for k, a in zip(("HH", "HV", "VH", "VV"), c) if abs(a) > 1e-12])
    norm = float(np.linalg.norm(c))
    psi = qt.pure_state_2q(np.abs(c), np.degrees(np.angle(c)))
    _check_against_question(psi)
    terms = [f"{abs(a):.4f}{_phase_text(np.degrees(np.angle(a)), 1)}|{k}>"
             for a, k in zip(psi, ("HH", "HV", "VH", "VV")) if abs(a) > 1e-12]
    desc = "pure " + " + ".join(terms)
    exprs = [f"{k}: {given[k]}" for k, a in zip(("HH", "HV", "VH", "VV"), c)
             if abs(a) > 1e-12 and not re.fullmatch(r"[-+]?\d*\.?\d+", str(given[k]).strip())]
    source = f" [from {', '.join(exprs)}]" if exprs else ""
    note = "" if abs(norm - 1) < 1e-4 else f" (normalized: amplitudes divided by {norm:.4f})"
    return f"Created {_save(qt.dm(psi), desc)} = {desc}{source}{note}."


@tool
def mix_states(handles: list[str], weights: list[float]) -> str:
    """Create the mixture sum_i w_i rho_i of stored states. Weights must be >= 0 and sum to 1.
    Example: 0.8*phi+ + 0.2*|HH><HH| -> create both states first, then mix them. All states must have
    the same number of photons (one or two)."""
    rho = qt.mixture(_same_size(handles), weights)
    desc = "mixture " + " + ".join(f"{w:g}·{h}" for w, h in zip(weights, handles))
    return f"Created {_save(rho, desc)} = {desc}."


@tool
def apply_local_noise(handle: str, channel: str, param: float, qubits: str = "both") -> str:
    """Apply a single-qubit noise channel to a stored state and return the new handle.
    channel and param:
      'phase_damping'     param = lambda, coherence factor (1 = no noise)
      'amplitude_damping' param = gamma, decay probability |V> -> |H> (0 = no noise)
      'depolarizing'      param = p, rho -> (1-p) rho + p I/2
      'bit_flip'          param = p, apply X (H <-> V) with probability p
      'phase_flip'        param = p, apply Z with probability p
      'bit_phase_flip'    param = p, apply Y with probability p
    qubits: 'both' (default, same channel on each qubit), 'A' (first qubit only) or 'B'.
    For a ONE-photon state the channel acts on that photon (qubits is ignored).
    Channels acting on the PAIR as a whole (4x4 Kraus operators): use apply_global_noise."""
    if _channel_name(channel) in GLOBAL_CHANNELS:
        raise ValueError(f"'{channel}' acts on the pair as a whole: use apply_global_noise.")
    rho0 = _get(handle)
    photons = _photons(rho0)
    K = _channel_kraus(channel, param, qubits, photons)
    _check_named_bell_state(handle)
    rho = qt.apply_kraus(rho0, K)
    desc = (f"{channel}({param}) on {QUBITS[qubits]} of {handle}" if photons == 2 else
            f"1 photon: {channel}({param}) on {handle}")
    return f"Created {_save(rho, desc)} = {desc}."


@tool
def compute_measures(handle: str) -> dict:
    """Compute purity, von Neumann entropy, concurrence, entanglement of formation, negativity,
    log-negativity, CHSH maximum, l1 and relative-entropy coherence (H/V basis), mutual
    information and discord for a stored state. log base 2 is used for the entropies, mutual
    information, discord, entanglement of formation and log-negativity (concurrence and
    negativity have no log base). For a ONE-photon state: Stokes parameters, degree of polarization,
    purity, entropy, polarization ellipse and the analyser probabilities instead."""
    rho = _get(handle)
    if rho.shape[0] == 2:
        return _one_photon_measures(handle, rho)
    rA, rB = qt.partial_trace(rho, [0], [2, 2]), qt.partial_trace(rho, [1], [2, 2])
    S_AB, S_B = qt.von_neumann_entropy(rho), qt.von_neumann_entropy(rB)
    return {
        "handle": handle,
        "purity": round(qt.purity(rho), 6),
        "entropy": round(qt.von_neumann_entropy(rho), 6) + 0.0,        # + 0.0: no '-0.0'
        "entropy_A": round(qt.von_neumann_entropy(rA), 6) + 0.0,
        "entropy_B": round(qt.von_neumann_entropy(rB), 6) + 0.0,
        "conditional_entropy_A_given_B": round(S_AB - S_B, 6) + 0.0,
        "concurrence": round(qt.concurrence(rho), 6),
        "entanglement_of_formation": round(qt.entanglement_of_formation(rho), 6),
        "negativity": round(qt.negativity(rho), 6),
        "log_negativity": round(qt.log_negativity(rho), 6),
        "chsh_max": round(qt.chsh_max(rho), 6),
        "coherence_l1": round(qt.coherence_l1(rho), 6),
        "coherence_rel_entropy": round(qt.coherence_rel_entropy(rho), 6),
        "mutual_information": round(qt.mutual_information(rho), 6),
        "discord": round(qt.quantum_discord(rho), 6),
        "convention": "big-endian qubits, basis {|0>=H,|1>=V}; log base 2 for entropy-type quantities",
        "definitions": ("N = sum |negative eigenvalues of rho^T_B| (max 1/2); "
                        "log_negativity = log2(2N+1); "
                        "EoF = h((1 + sqrt(1 - C^2))/2) with h the binary entropy (max 1); "
                        "entropy = -Tr rho log2 rho (0 for pure states); entropy_A, entropy_B of the "
                        "reduced states (= entanglement entropy when rho is pure); conditional_entropy_A_given_B "
                        "= S(AB) - S(B): negative proves entanglement, and -S(A|B) (coherent information) is a "
                        "lower bound on the distillable entanglement (hashing bound); "
                        "chsh_max = 2*sqrt(t1^2+t2^2) from correlation matrix T (local bound 2, quantum 2*sqrt2); "
                        "coherence_l1 = sum |off-diagonal elements| in H/V basis; coherence_rel_entropy = "
                        "S(diag rho) - S(rho); discord: projective measurement on qubit B"),
    }


# ------------------------------------------------------------------ one photon (a single polarization qubit)
_POLARIZATIONS = {"H": (0, 0), "V": (90, 0), "D": (45, 0), "A": (-45, 0), "R": (0, 45), "L": (0, -45)}


def _one_photon_measures(handle: str, rho: np.ndarray) -> dict:
    s = qt.stokes_vector(rho)
    dop = float(np.linalg.norm(s))
    ell = qt.polarization_ellipse(s)
    w, V = np.linalg.eigh(rho)
    P = qt.pauli_projectors_1q()
    out = {
        "handle": handle, "photons": 1,
        "stokes": {"S1 (H-V)": round(float(s[0]), 6) + 0.0, "S2 (D-A)": round(float(s[1]), 6) + 0.0,
                   "S3 (R-L)": round(float(s[2]), 6) + 0.0},
        "degree_of_polarization": round(dop, 6),
        "purity": round(qt.purity(rho), 6),
        "entropy": round(qt.von_neumann_entropy(rho), 6) + 0.0,
        "probabilities": {k: round(float(np.real(np.trace(rho @ P[k]))), 6) + 0.0 for k in "HVDARL"},
        "bloch_vector_xyz": [round(float(s[1]), 6) + 0.0, round(float(s[2]), 6) + 0.0, round(float(s[0]), 6) + 0.0],
        "polarized_part": _ket1_text(V[:, -1]) if dop > 1e-9 else "none (unpolarized light)",
    }
    if ell:
        chi = ell["ellipticity_deg"]
        out["ellipse"] = {"azimuth_deg": (round(ell["azimuth_deg"], 4) + 0.0 if ell["azimuth_deg"] is not None
                                          else "undefined (circular polarization)"),
                          "ellipticity_deg": round(chi, 4) + 0.0,
                          "shape": ("linear" if abs(chi) < 1e-4 else "circular" if abs(abs(chi) - 45) < 1e-4
                                    else "elliptical"),
                          "handedness": ("none (linear)" if abs(chi) < 1e-4 else
                                         "right (towards R)" if chi > 0 else "left (towards L)")}
    out["convention"] = ("one photon, basis H, V; S1 = P(H)-P(V), S2 = P(D)-P(A), S3 = P(R)-P(L) with "
                         "R = (H+iV)/√2; Bloch vector (x, y, z) = (S2, S3, S1), north pole = H")
    out["definitions"] = ("degree of polarization = |S| (1 = fully polarized = pure, 0 = unpolarized I/2); "
                          "purity = (1 + |S|²)/2; entropy = -Tr ρ log2 ρ (max 1 bit); ellipse azimuth "
                          "ψ = ½ atan2(S2, S1), ellipticity χ = ½ asin(S3/|S|)")
    return _clean(out)


@tool
def create_single_photon_state(polarization: str = "", azimuth_deg: float | None = None,
                               ellipticity_deg: float = 0.0, dop: float = 1.0, s1: float | None = None,
                               s2: float | None = None, s3: float | None = None, amp_H: str = "",
                               amp_V: str = "") -> str:
    """Create the polarization state of ONE photon (a 2x2 density matrix) and return its handle.
    Give exactly ONE of:
      polarization: 'H', 'V', 'D', 'A', 'R' or 'L'   (R = (H+iV)/√2)
      azimuth_deg (+ ellipticity_deg): polarization ellipse, azimuth from H; ellipticity angle χ in
        [-45°, 45°], 0 = linear, +45° = R, -45° = L
      s1, s2, s3: normalized Stokes parameters (S1 = H-V, S2 = D-A, S3 = R-L); |S| < 1 = partly polarized
      amp_H, amp_V: amplitudes as text exactly as the user wrote them ('1', 'i', 'cos(0.3)', ...)
    dop: degree of polarization for the first, second and fourth form (rho = dop|ψ><ψ| + (1-dop) I/2;
    dop = 0 is unpolarized light)."""
    words = {"horizontal": "H", "vertical": "V", "diagonal": "D", "antidiagonal": "A", "anti-diagonal": "A",
             "right": "R", "right-circular": "R", "left": "L", "left-circular": "L"}
    polarization = words.get(polarization.strip().lower(), polarization.strip())
    if polarization.lower() in ("linear", "elliptical", "ellipse", "custom") and azimuth_deg is not None:
        polarization = ""                         # the ellipse angles say it all
    modes = [bool(polarization.strip()), azimuth_deg is not None, any(v is not None for v in (s1, s2, s3)),
             bool(str(amp_H).strip() or str(amp_V).strip())]
    if sum(modes) != 1:
        raise ValueError("Give exactly one of: polarization ('H', 'V', 'D', 'A', 'R', 'L'), azimuth_deg "
                         "(+ ellipticity_deg), Stokes s1/s2/s3, or amplitudes amp_H/amp_V.")
    if not 0 <= dop <= 1:
        raise ValueError("dop (degree of polarization) must be in [0, 1].")
    tag = "" if dop == 1 else f", degree of polarization {dop:g}"
    if modes[0]:
        key = polarization.strip().upper()
        if key not in _POLARIZATIONS:
            raise ValueError("polarization must be 'H', 'V', 'D', 'A', 'R' or 'L'.")
        stokes = qt.stokes_from_ellipse(*_POLARIZATIONS[key], dop)
        desc = f"1 photon: {key}{tag}"
    elif modes[1]:
        stokes = qt.stokes_from_ellipse(float(azimuth_deg), float(ellipticity_deg), dop)
        shape = "linear" if abs(ellipticity_deg) < 1e-9 else "elliptical"
        desc = (f"1 photon: {shape}, azimuth {azimuth_deg:g}°"
                + (f", ellipticity {ellipticity_deg:g}°" if shape == "elliptical" else "") + tag)
    elif modes[2]:
        if dop != 1:
            raise ValueError("With Stokes parameters the degree of polarization is |S| itself: do not give dop.")
        stokes = np.array([s1 or 0.0, s2 or 0.0, s3 or 0.0], dtype=float)
        desc = f"1 photon: Stokes ({stokes[0]:g}, {stokes[1]:g}, {stokes[2]:g})"
    else:
        c = np.array([eval_amplitude(amp_H or "0"), eval_amplitude(amp_V or "0")])
        if np.linalg.norm(c) < 1e-12:
            raise ValueError("Both amplitudes are zero.")
        psi = c / np.linalg.norm(c)
        stokes = dop * qt.stokes_vector(qt.dm(psi))
        desc = f"1 photon: {_ket1_text(psi)}{tag}"
    rho = qt.rho_from_stokes(*stokes)
    s = qt.stokes_vector(rho)
    return (f"Created {_save(rho, desc)} ({desc}); Stokes (S1, S2, S3) = ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}), "
            f"degree of polarization {np.linalg.norm(s):.4f}; {_ellipse_text(s)}.")


def _ellipse_text(s: np.ndarray) -> str:
    ell = qt.polarization_ellipse(s)
    if ell is None:
        return "polarization ellipse undefined (unpolarized light)"
    chi = ell["ellipticity_deg"]
    shape = "linear" if abs(chi) < 1e-4 else "circular" if abs(abs(chi) - 45) < 1e-4 else "elliptical"
    az = (f"azimuth {ell['azimuth_deg']:.2f}°" if ell["azimuth_deg"] is not None
          else "azimuth undefined (circular polarization)")
    return f"polarization ellipse: {shape}, {az}, ellipticity {chi:.2f}°"


@tool
def reduced_state(handle: str, photon: str = "", qubit: str = "") -> str:
    """The polarization state of ONE photon of a stored two-photon state (partial trace over the other
    photon): photon='A' or 'B'. Stored as a new one-photon state. For a maximally entangled state it is
    I/2 (unpolarized): entanglement shows up as local mixedness."""
    photon = (photon or qubit).strip().upper().replace("PHOTON", "").replace("QUBIT", "").strip()
    if photon not in ("A", "B"):
        raise ValueError("Give photon='A' or photon='B' (the photon whose state you want).")
    rho = _get2(handle, "reduced_state")
    r = qt.partial_trace(rho, [0 if photon == "A" else 1], [2, 2])
    s = qt.stokes_vector(r)
    desc = f"1 photon: photon {photon} of {handle}"
    return (f"Created {_save(r, desc)} ({desc}); Stokes (S1, S2, S3) = ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}), "
            f"degree of polarization {np.linalg.norm(s):.4f}, entropy {qt.von_neumann_entropy(r):.4f} bits.")


@tool
def combine_photons(handle_a: str, handle_b: str) -> str:
    """Two stored ONE-photon states -> the two-photon product state rho_A ⊗ rho_B (photon A first).
    A product state is never entangled."""
    a, b = _get(handle_a), _get(handle_b)
    if a.shape[0] != 2 or b.shape[0] != 2:
        raise ValueError("combine_photons needs two ONE-photon states (create them with create_single_photon_state "
                         "or reduced_state).")
    desc = f"product {handle_a} ⊗ {handle_b}"
    return f"Created {_save(np.kron(a, b), desc)} ({desc})."


# ------------------------------------------------------------------ density matrix as numbers
BASIS_1 = ["H", "V"]
BASIS_2 = ["HH", "HV", "VH", "VV"]


def cplx_text(z: complex, digits: int = 4) -> str:
    """'0.5000', '-0.1000i', '0.4250 - 0.1000i'; '0' below half a unit of the last digit."""
    tol = 0.5 * 10 ** -digits
    re_, im = float(np.real(z)), float(np.imag(z))
    re_ = 0.0 if abs(re_) < tol else re_
    im = 0.0 if abs(im) < tol else im
    if re_ == 0 and im == 0:
        return "0"
    if im == 0:
        return f"{re_:.{digits}f}"
    if re_ == 0:
        return f"{im:.{digits}f}i"
    return f"{re_:.{digits}f} {'+' if im > 0 else '-'} {abs(im):.{digits}f}i"


def matrix_rows(M: np.ndarray, digits: int = 4) -> list[list[str]]:
    return [[cplx_text(z, digits) for z in row] for row in M]


def _basis(rho: np.ndarray) -> list[str]:
    return BASIS_1 if rho.shape[0] == 2 else BASIS_2


def _nonzero_elements(rho: np.ndarray, tol: float = 5e-5) -> list[str]:
    b = _basis(rho)
    return [f"ρ[{b[i]},{b[j]}] = {cplx_text(rho[i, j])}" for i in range(len(b)) for j in range(i, len(b))
            if abs(rho[i, j]) >= tol]


def _element_errors(handle: str) -> dict | None:
    """± (1σ) of Re and Im of every element, from the same resamplings as the tomography report."""
    states = _bootstrap_states(handle)
    if not states:
        return None
    arr = np.array(states)
    return {"re": [[f"{v:.4f}" for v in row] for row in arr.real.std(axis=0, ddof=1)],
            "im": [[f"{v:.4f}" for v in row] for row in arr.imag.std(axis=0, ddof=1)]}


@tool
def show_density_matrix(handle: str, compare_with: str = "") -> dict:
    """The density matrix of a stored state AS NUMBERS: 4x4 for two photons (rows and columns HH, HV,
    VH, VV, photon A first) or 2x2 for one photon (H, V), each element as 'a + bi' with 4 decimals,
    plus its eigenvalues. compare_with: an optional second state (a handle or 'phi+', 'phi-', 'psi+',
    'psi-') of the same size: its matrix and the element-wise change Δρ = ρ(handle) - ρ(compare_with)
    are added (e.g. after vs before noise). For a state reconstructed from counts every element also
    gets ± (1σ). The tables are shown to the user automatically: do not retype the matrix; describe
    it with 'nonzero_elements' and 'changed_elements'."""
    rho = _get(handle)
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    out = {"handle": handle, "state": DESC.get(handle, ""),
           "size": f"{rho.shape[0]}x{rho.shape[0]} ({_photons(rho)} photon{'s' if _photons(rho) == 2 else ''})",
           "basis": _basis(rho), "matrix": matrix_rows(rho),
           "eigenvalues": [round(float(v), 4) + 0.0 for v in w[::-1]],
           "trace": round(float(np.real(np.trace(rho))), 6),
           "nonzero_elements": _nonzero_elements(rho)}
    errs = _element_errors(handle) if rho.shape[0] == 4 else None
    if errs:
        out["element_errors_1sigma"] = errs
    if compare_with:
        label, ref = _reference_state(compare_with)
        if ref.shape != rho.shape:
            raise ValueError(f"{handle} and {compare_with} have different sizes (one and two photons).")
        D = rho - ref
        out["compared_with"] = {"handle": label, "state": DESC.get(label, label), "matrix": matrix_rows(ref)}
        out["difference"] = matrix_rows(D)
        out["changed_elements"] = _nonzero_elements(D) or ["none: the two matrices are equal"]
        out["difference_means"] = f"Δρ = ρ({handle}) - ρ({label})"
    out["how_to_read"] = ("ρ_ij with i = row, j = column; diagonal = populations (probabilities of the basis "
                          "states, sum 1), off-diagonal = coherences; ρ is Hermitian, so ρ_ji is the complex "
                          "conjugate of ρ_ij")
    return _clean(out)


# ------------------------------------------------------------------ global noise and Kraus operators
@tool
def apply_global_noise(handle: str, channel: str, param: float) -> str:
    """Apply a GLOBAL (two-photon) noise channel to a stored two-photon state: its Kraus operators are
    4x4 and act on the pair, not on each photon separately. param p in [0, 1], 0 = no noise:
      'global_depolarizing'          rho -> (1-p) rho + p I/4 (on a Bell state: a Werner state)
      'global_dephasing'             every off-diagonal element (HH, HV, VH, VV basis) x (1-p)
      'collective_dephasing'         both photons get the SAME random phase (e.g. one fibre); elements
                                     between states with one more V x (1-p), two more V x (1-p)^4;
                                     psi+ and psi- are immune (decoherence-free subspace)
      'collective_amplitude_damping' common decay V -> H (jump operator σ-⊗I + I⊗σ-), p = 1 - e^(-γt);
                                     psi- never decays (dark), psi+ decays twice as fast
      'correlated_bit_flip', 'correlated_phase_flip', 'correlated_bit_phase_flip'
                                     X⊗X, Z⊗Z or Y⊗Y on BOTH photons with probability p
    Local channels (each photon separately): apply_local_noise. Kraus operators: get_kraus_operators."""
    name = _channel_name(channel)
    if name not in GLOBAL_CHANNELS:
        if name in CHANNELS:
            raise ValueError(f"'{channel}' acts on each photon separately: use apply_local_noise.")
        raise ValueError(f"channel must be one of {list(GLOBAL_CHANNELS)}")
    rho = _get2(handle, "A global channel")
    _check_named_bell_state(handle)
    K = _channel_kraus(name, param, "both")
    desc = f"{name}({param}) on the pair {handle}"
    return f"Created {_save(qt.apply_kraus(rho, K), desc)} = {desc} ({len(K)} Kraus operators, 4x4)."


@tool
def get_kraus_operators(channel: str, param: float, qubits: str = "both") -> dict:
    """The Kraus operators of a noise channel at parameter `param`, as matrices (H/V basis), with the
    completeness check sum K†K = I. Local channels (apply_local_noise names) give the 2x2 one-photon
    operators and say how they act on `qubits` ('both' = all products K_i ⊗ K_j); global channels
    (apply_global_noise names) give the 4x4 operators on the pair (basis HH, HV, VH, VV). The
    operators are shown to the user automatically: do not retype them."""
    name = _channel_name(channel)
    if name in GLOBAL_CHANNELS:
        K, scope = _channel_kraus(name, param, "both"), "global: 4x4 operators acting on the pair"
        basis, meaning = BASIS_2, GLOBAL_CHANNELS[name][1]
    elif name in CHANNELS:
        _channel_kraus(name, param, qubits)                    # validates param and qubits
        K, basis, meaning = CHANNELS[name][0](param), BASIS_1, CHANNELS[name][1]
        n = len(K)
        scope = {"both": f"local: each photon gets these 2x2 operators independently; on the pair the "
                         f"{n * n} operators K_i ⊗ K_j",
                 "A": "local: photon A gets these 2x2 operators; on the pair K_i ⊗ I",
                 "B": "local: photon B gets these 2x2 operators; on the pair I ⊗ K_i"}[qubits]
    else:
        raise ValueError(f"channel must be one of {list(CHANNELS)} (local) or {list(GLOBAL_CHANNELS)} (global)")
    out = {"channel": name, "param": param, "parameter_meaning": meaning, "scope": scope,
           "basis": basis, "number_of_operators": len(K)}
    if all(np.allclose(k, np.diag(np.diag(k))) for k in K):   # diagonal operators: ρ_ij -> factor_ij · ρ_ij
        d = [np.diag(k) for k in K]
        out["coherence_factors"] = {
            f"{basis[i]}-{basis[j]}": round(float(np.real(sum(x[i] * np.conj(x[j]) for x in d))), 4) + 0.0
            for i in range(len(basis)) for j in range(i + 1, len(basis))}
        out["coherence_factors_meaning"] = "each off-diagonal element ρ_ij is multiplied by this number; populations unchanged"
    return _clean({**out,
                   "kraus_operators": {f"K{i}": matrix_rows(k) for i, k in enumerate(K)},
                   "completeness_error": float(f"{qt.completeness_error(K):.1e}"),
                   "trace_preserving": qt.completeness_error(K) < 1e-9,
                   "action": "ρ -> Σ_k K_k ρ K_k†"})


def _parse_matrices(value) -> list[np.ndarray]:
    """Kraus operators from the model: a list of matrices (lists of rows) whose entries are numbers
    or text expressions ('sqrt(0.3)', 'i/2'), or the same written as one string."""
    if isinstance(value, str):
        try:
            tree = ast.parse(value.strip(), mode="eval").body
        except SyntaxError:
            raise ValueError("Could not read the Kraus operators: write them as [[[a, b], [c, d]], ...].") from None

        def build(node):
            if isinstance(node, (ast.List, ast.Tuple)):
                return [build(e) for e in node.elts]
            return ast.get_source_segment(value.strip(), node)
        value = build(tree)
    mats = []
    for m in value:
        rows = [[eval_amplitude(x) for x in row] for row in m]
        M = np.array(rows, dtype=complex)
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] not in (2, 4):
            raise ValueError(f"Every Kraus operator must be a 2x2 or 4x4 matrix, got shape {M.shape}.")
        mats.append(M)
    if not mats or len({M.shape for M in mats}) > 1:
        raise ValueError("Give at least one Kraus operator; all must have the same size.")
    return mats


class _KrausArgs(BaseModel):
    handle: str
    kraus: list | str
    qubits: str = "both"


@tool(args_schema=_KrausArgs)
def apply_kraus_operators(handle: str, kraus: list | str, qubits: str = "both") -> str:
    """Apply the USER'S OWN Kraus operators to a stored state: rho -> Σ K ρ K†. kraus = list of matrices,
    each a list of rows, entries as numbers or text such as 'sqrt(0.3)' or 'i' (copy them exactly as
    written). 2x2 operators act on a one-photon state, or on photon 'A', 'B' or 'both' (all products)
    of a pair; 4x4 operators act on a pair (basis HH, HV, VH, VV). They must satisfy Σ K†K = I
    (checked; otherwise nothing is created and the sum is reported)."""
    K = _parse_matrices(kraus)
    rho = _get(handle)
    err = qt.completeness_error(K)
    if err > 1e-6:
        S = sum(k.conj().T @ k for k in K)
        raise ValueError(f"Not a valid (trace-preserving) channel: Σ K†K = {matrix_rows(S)} instead of the "
                         f"identity (largest deviation {err:.2e}). Check the operators with the user.")
    d = K[0].shape[0]
    if d == 4 and rho.shape[0] != 4:
        raise ValueError(f"4x4 Kraus operators need a two-photon state; {handle} has one photon.")
    if d == 2 and rho.shape[0] == 4:
        q = qubits.strip().upper() if qubits.strip().lower() != "both" else "both"
        if q not in ("both", "A", "B"):
            raise ValueError("qubits must be 'both', 'A' or 'B'.")
        K = {"both": qt.local_channel(K, K), "A": qt.local_channel(K, [qt.I2]),
             "B": qt.local_channel([qt.I2], K)}[q]
        where = {"both": "both photons", "A": "photon A", "B": "photon B"}[q]
    else:
        where = "the pair" if d == 4 else "the photon"
    desc = f"{handle} after the user's {len(_parse_matrices(kraus))} Kraus operators ({d}x{d}) on {where}"
    return f"Created {_save(qt.apply_kraus(rho, K), desc)} = {desc}."


# ------------------------------------------------------------------ one-photon tomography
@tool
def reconstruct_single_photon(counts_H: float | None = None, counts_V: float | None = None,
                              counts_D: float | None = None, counts_A: float | None = None,
                              counts_R: float | None = None, counts_L: float | None = None,
                              r_convention: str = "R=(H+iV)/sqrt2") -> dict:
    """One-photon polarization tomography from counts behind H, V, D, A, R, L analysers (give the
    numbers the user measured; at least H, V, D and R, all six best). Poisson maximum likelihood, same
    counting time per setting. Returns the new one-photon state, Stokes parameters, degree of
    polarization and purity with ± (1σ, 400 Poisson resamplings) and the linear-inversion Stokes."""
    given = {k: v for k, v in zip("HVDARL", (counts_H, counts_V, counts_D, counts_A, counts_R, counts_L))
             if v is not None}
    if any(v < 0 for v in given.values()):
        raise ValueError("Counts cannot be negative.")
    conv = r_convention.replace(" ", "")
    if conv not in ("R=(H+iV)/sqrt2", "R=(H-iV)/sqrt2"):
        raise ValueError("r_convention must be 'R=(H+iV)/sqrt2' or 'R=(H-iV)/sqrt2'.")
    sign = +1 if "+i" in conv else -1
    rho = qt.mle_tomography_1q(given, sign)
    P = qt.pauli_projectors_1q(sign)
    total = sum(given.values())
    p = np.array([np.real(np.trace(rho @ P[k])) for k in given])
    mu = total * p / p.sum()
    rng = np.random.default_rng(1)
    boot = [qt.mle_tomography_1q(dict(zip(given, rng.poisson(mu))), sign) for _ in range(N_RESAMPLES_1Q)]
    sv = np.array([qt.stokes_vector(b) for b in boot])
    rho_std = rho                 # the projectors used the user's R, so rho is already the physical state
    s = qt.stokes_vector(rho_std)
    dops = np.linalg.norm(sv, axis=1)
    lin = {}
    for name, (a, b) in (("S1 (H-V)", "HV"), ("S2 (D-A)", "DA"), ("S3 (R-L)", "RL")):
        if a in given and b in given and given[a] + given[b] > 0:
            lin[name] = round((given[a] - given[b]) / (given[a] + given[b]), 4) * (sign if a == "R" else 1)
    desc = f"1 photon: reconstructed from counts {', '.join(f'{k}={v:g}' for k, v in given.items())} (MLE, {conv})"
    h = _save(rho_std, desc)
    pm = lambda x, e: f"{round(float(x), 4) + 0.0:.4f} ± {_two_sig(e)}"        # noqa: E731
    return _clean({
        "handle": h, "settings": list(given), "total_counts": total,
        "stokes_mle": {"S1 (H-V)": pm(s[0], sv[:, 0].std(ddof=1)), "S2 (D-A)": pm(s[1], sv[:, 1].std(ddof=1)),
                       "S3 (R-L)": pm(s[2], sv[:, 2].std(ddof=1))},
        "degree_of_polarization": pm(np.linalg.norm(s), dops.std(ddof=1)),
        "purity": pm(qt.purity(rho), np.std([qt.purity(b) for b in boot], ddof=1)),
        "stokes_linear_inversion": lin or "needs both settings of a pair (H and V, D and A, R and L)",
        "convention": ("R = (H+iV)/√2, S3 = P(R) - P(L)" if sign > 0 else
                       "your counts used R = (H-iV)/√2; the stored state and S3 use R = (H+iV)/√2 (S3 = P(R) - P(L))"),
        "note": f"± = 1σ from {N_RESAMPLES_1Q} Poisson resamplings of the fitted counts (statistical only)"})


_DM_STYLES = {
    "bars3d": ("left Re(ρ), right Im(ρ); bar height = signed value of the element; blue bars >= 0, "
               "red bars < 0"),
    "heatmap": ("left Re(ρ), right Im(ρ) as coloured grids with the value written in every cell; "
                "blue = positive, red = negative, light grey = 0"),
    "magnitude_phase": ("one 3D bar chart: bar height = |ρ_ij|, bar colour = phase of ρ_ij (colour "
                        "bar in degrees; -180° and +180° have the same colour)"),
}


def _reference_state(name: str):
    """(label, matrix) for a stored handle or a Bell-state name used as a reference."""
    key = name.strip().lower().replace(" ", "")
    if key in ("phi+", "phi-", "psi+", "psi-"):
        return key, qt.dm(qt.bell_state(key))
    return name, _get(name)


@tool
def plot_density_matrix(handle: str, style: str = "bars3d", compare_with: str = "") -> dict:
    """Plot a stored density matrix. style: 'bars3d' (3D bars of Re and Im, default), 'heatmap'
    (Re and Im as grids with every value written in), or 'magnitude_phase' (|ρ_ij| as bar
    height, phase as colour - best to SEE a relative phase such as e^(-i108°)).
    compare_with: optional second state (a handle such as 'rho_3', or 'phi+', 'phi-', 'psi+',
    'psi-') drawn in a second row with the same style, plus the fidelity between the two.
    Returns the PNG path, how to read it and the largest elements. Describe the plot ONLY with
    these fields - you cannot see the image."""
    style = style.strip().lower().replace(" ", "_").replace("-", "_")
    style = {"3d": "bars3d", "bars": "bars3d", "city": "bars3d", "hinton": "heatmap",
             "phase": "magnitude_phase", "magnitude": "magnitude_phase"}.get(style, style)
    if style not in _DM_STYLES:
        raise ValueError(f"style must be one of {list(_DM_STYLES)}")
    rho = _get(handle)
    states = [(handle, rho)]
    out = {}
    if compare_with:
        label, ref = _reference_state(compare_with)
        if ref.shape != rho.shape:
            raise ValueError(f"{handle} and {compare_with} have different sizes (one and two photons).")
        states.append((label, ref))
        out["compared_with"] = label
        out["fidelity_between_plotted_states"] = round(qt.fidelity(rho, ref), 4)
    if style == "bars3d" and not compare_with:
        path = f"{_fig_dir()}/{handle}_city.png"          # "/" works on Windows too
    else:
        path = pl.figure_path(f"{handle}_{style}" + (f"_vs_{compare_with}" if compare_with else ""),
                              [style, compare_with, handle])
    pl.density_matrix_figure(states, style, path)
    labels = _basis(rho)
    d = len(labels)
    big = []
    for i in range(d):
        for j in range(i, d):
            if abs(rho[i, j]) >= 0.05:
                z = rho[i, j]
                val = (f"{z.real:.3f}" if i == j else
                       f"{z.real:+.3f}{z.imag:+.3f}i (magnitude {abs(z):.3f}, phase {np.degrees(np.angle(z)):.1f}°)")
                big.append(f"ρ[{labels[i]},{labels[j]}] = {val}")
    return _clean({"figure": path, "style": style,
                   "how_to_read": (_DM_STYLES[style] + f"; diagonal = populations of {', '.join(labels)}; "
                                   "off-diagonal = coherences (they alone do not prove entanglement: "
                                   "product states such as |DD> also have them)"
                                   + ("; second row: the comparison state" if compare_with else "")),
                   "elements_with_magnitude_at_least_0.05": big, **out})


def _first_crossing(xs, ys, level):
    """First x (moving away from no noise) where ys drops to <= level; None if never."""
    for i in range(1, len(xs)):
        if ys[i] <= level < ys[i - 1]:
            frac = (ys[i - 1] - level) / (ys[i - 1] - ys[i])
            return float(xs[i - 1] + frac * (xs[i] - xs[i - 1]))
    return None


@tool
def plot_measures_vs_noise(start_kind: str, channel: str, n_points: int = 41,
                           qubits: str = "both") -> dict:
    """Plot concurrence, negativity, purity and CHSH/2 of ONE Bell state ('phi+', 'psi-', ...)
    as the noise parameter of `channel` goes from 0 to 1, applied to `qubits`
    ('both', 'A' or 'B'). Channels as in apply_local_noise or apply_global_noise. To compare SEVERAL states, or to
    choose the measures, use plot_sweep(sweep='noise', handles=[...]) instead (one figure).
    Returns the PNG path AND 'features': the exact values where entanglement vanishes and
    where CHSH drops to 2, all measures there ('at_key_points'), at no noise, at maximum noise
    and at 11 'samples' (Pauli channels: also the exact 'mechanism'). Describe the figure ONLY with
    these features - you cannot see the image."""
    _channel_kraus(channel, 0.5, qubits)          # validates channel and qubits (local or global)
    channel = _channel_name(channel)
    n_points = int(np.clip(n_points, 11, 201))     # bounded work while the compute lock is held
    rho0 = qt.dm(qt.bell_state(start_kind))

    def measures(x):
        r = qt.apply_kraus(rho0, _channel_kraus(channel, x, qubits))
        return (qt.concurrence(r), qt.negativity(r), qt.purity(r), qt.chsh_max(r))

    xs = np.linspace(0, 1, n_points)
    rows = np.array([measures(x) for x in xs])

    # --- figure
    meaning = {**{name: info[1] for name, info in CHANNELS.items()},
               **{name: "noise parameter p (0 = no noise)" for name in GLOBAL_CHANNELS}}
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, name in enumerate(["concurrence", "negativity", "purity"]):
        ax.plot(xs, rows[:, col], label=name)
    ax.plot(xs, rows[:, 3] / 2, label="CHSH max / 2")
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(meaning[channel])
    ax.set_title(f"{start_kind} under {channel} on "
                 f"{'the pair (global)' if channel in GLOBAL_CHANNELS else QUBITS[qubits]}")
    ax.set_ylim(-0.03, 1.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=8)
    path = f"{_fig_dir()}/{start_kind}_{channel}_{qubits}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- numerical features on a fine grid, ordered from no noise to maximum noise
    fine = np.linspace(0, 1, 2001)
    if channel in CHANNELS and CHANNELS[channel][2] == 1.0:   # phase damping: lambda = 1 is the noiseless end
        fine = fine[::-1]
    f = np.array([measures(x) for x in fine])
    vanish = _first_crossing(fine, f[:, 0], 1e-9)
    if vanish is not None:                # the first grid point at or below the level must be separable
        probe = float(fine[int(np.argmax(f[:, 0] <= 1e-9))])
        if not _separable(qt.apply_kraus(rho0, _channel_kraus(channel, probe, qubits))):
            vanish = None                 # only decayed below 1e-9: asymptotic, not sudden death
    chsh_cross = _first_crossing(fine, f[:, 3], 2.0)
    if chsh_cross is not None and _only_approaches(f[int(np.argmax(f[:, 3] <= 2.0)):, 3], 2.0):
        chsh_cross = None                 # CHSH only approaches 2 (e.g. 2√(1 + c²) with c -> 0)
    names = ("concurrence", "negativity", "purity", "chsh_max")
    at = lambda row: {n: round(float(v), 4) for n, v in zip(names, row)}  # noqa: E731
    end = float(fine[-1])
    describe = lambda x: (f"only at maximum noise ({end:g})" if abs(x - end) < 1e-3  # noqa: E731
                          else round(x, 3))
    def refine(x, col, level):          # bisection around the grid estimate (values exactly at the crossing)
        if x is None:
            return None
        step = abs(fine[1] - fine[0])
        lo, hi = max(0.0, x - step), min(1.0, x + step)
        if (measures(lo)[col] > level + 1e-12) == (measures(hi)[col] > level + 1e-12):
            return x
        lo_above = measures(lo)[col] > level + 1e-12
        for _ in range(50):
            mid = (lo + hi) / 2
            if (measures(mid)[col] > level + 1e-12) == lo_above:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    raw = {"CHSH reaches 2": refine(chsh_cross, 3, 2.0), "entanglement vanishes": refine(vanish, 0, 1e-9)}
    vanish = describe(vanish) if vanish is not None else None
    chsh_cross = describe(chsh_cross) if chsh_cross is not None else None
    features = {
        "x_axis": meaning[channel],
        "entanglement_vanishes_at": vanish if vanish is not None else
            ((f"only at maximum noise ({end:g}): concurrence decays towards 0 and vanishes only there"
              if f[-1, 0] <= 1e-9 else
              f"never: concurrence stays > 0 (minimum {f[:, 0].min():.4f} at {fine[f[:, 0].argmin()]:.3f})")
             if f[0, 0] > 1e-9 else "not entangled even without noise"),
        "chsh_drops_to_2_at": chsh_cross if chsh_cross is not None else
            (f"never drops below 2 (minimum {f[:, 3].min():.4f} at {fine[f[:, 3].argmin()]:.3f})"
             if f[0, 3] > 2 else "does not violate CHSH even without noise"),
        "at_no_noise": at(f[0]),
        "at_max_noise": at(f[-1]),
        "samples": {f"{x:.1f}": at(measures(x)) for x in np.linspace(0, 1, 11)},
    }
    key = {}                       # all measures exactly at the crossings (for tables)
    for x, why in ((chsh_cross, "CHSH reaches 2"), (vanish, "entanglement vanishes")):
        if isinstance(x, float):
            key[f"p={x:g} ({why})"] = at(measures(raw[why]))
    if key:
        features["at_key_points"] = key
    features["quote_only"] = "values at the noise values listed in 'samples' and 'at_key_points'"
    pauli = {"bit_flip": ("bit flip X", np.array([[0, 1], [1, 0]])),
             "phase_flip": ("phase flip Z", np.diag([1, -1])),
             "bit_phase_flip": ("bit-phase flip Y", np.array([[0, -1j], [1j, 0]]))}
    if channel in pauli:                          # how a Pauli channel acts on a Bell state (exact)
        name, P = pauli[channel]
        flipped = np.kron(P, np.eye(2)) @ qt.bell_state(start_kind)
        other = next(b for b in ("phi+", "phi-", "psi+", "psi-")
                     if abs(abs(np.vdot(qt.bell_state(b), flipped)) - 1) < 1e-9)
        if qubits == "both":
            features["mechanism"] = (
                f"each photon independently gets a {name} with probability p. One flip (probability "
                f"2p(1-p)) turns {start_kind} into {other}; two flips (probability p²) undo each other. So the "
                f"state is (1-q)|{start_kind}><{start_kind}| + q|{other}><{other}| with q = 2p(1-p) (rank 2, not white "
                f"noise); concurrence = |1-2q| = (1-2p)². At p = 1/2 the two Bell states are mixed equally "
                f"(separable); the curves are symmetric about p = 1/2 (one minimum, no oscillation). At p = 1 "
                f"both photons are flipped, which gives {start_kind} back exactly (q = 0), not {other}.")
        else:
            features["mechanism"] = (
                f"photon {qubits} gets a {name} with probability p, turning {start_kind} into {other}: the state is "
                f"(1-p)|{start_kind}><{start_kind}| + p|{other}><{other}|; concurrence = |1-2p|, separable only "
                f"at p = 1/2.")
    if isinstance(vanish, float) and f[-1, 0] > 1e-9:
        features["entanglement_returns"] = (
            "yes: concurrence is zero at the crossing but positive again at maximum noise "
            f"(C = {f[-1, 0]:.4f})")
    return _clean({"figure": path, "features": features})


# ------------------------------------------------------------------ tomography (experimental data)
_SETTING = re.compile(r"^[HVDARL]{2}$")
MAX_CSV_BYTES = 1_000_000
N_RESAMPLES = 100
N_RESAMPLES_1Q = 400          # one photon: cheap fits, and 100 resamplings leave the ± itself uncertain by ~7%


def _two_sig(x: float) -> float:
    """Round an uncertainty to 2 significant figures (0.011347 -> 0.011)."""
    return 0.0 if x == 0 else float(f"{x:.2g}")


def _ket_text(v: np.ndarray) -> str:
    """'0.754|HH> + 0.649·e^(-i108°)|VV>' with the global phase removed (largest amplitude real)."""
    v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
    text = ""
    for c, k in zip(v, ("HH", "HV", "VH", "VV")):
        if abs(c) < 0.05:
            continue
        ph = np.degrees(np.angle(c))
        if abs(abs(ph) - 180) < 0.5:
            text += (" - " if text else "-") + f"{abs(c):.3f}|{k}>"
        else:
            text += (" + " if text else "") + f"{abs(c):.3f}{_phase_text(ph, 0)}|{k}>"
    return text


def _best_bell_like(rho: np.ndarray) -> dict:
    """Max fidelity with (|HH> + e^{i phi}|VV>)/sqrt2 and with (|HV> + e^{i phi}|VH>)/sqrt2."""
    out = []
    for (a, b), name in (((0, 3), "(|HH> + e^(iφ)|VV>)/√2"), ((1, 2), "(|HV> + e^(iφ)|VH>)/√2")):
        F = float(np.real(rho[a, a] + rho[b, b]) / 2 + abs(rho[a, b]))
        phi = float(np.degrees(-np.angle(rho[a, b])))      # maximizes Re(e^{i phi} rho_ab)
        out.append((F, name, phi))
    F, name, phi = max(out)
    return {"state": name, "phi_deg": round(phi, 1), "fidelity": round(F, 4)}


def _resolve_csv(path: str) -> str:
    p = os.path.normpath(session.real_path(path).strip().strip('"').strip("'"))   # 'uploads/x.csv' too
    if not p.lower().endswith(".csv"):
        raise ValueError("Please give a .csv file.")
    if not os.path.isfile(p) or not session.allowed_csv(p):    # a web user sees only their own uploads
        found = session.readable_csvs()
        raise ValueError(f"File '{path}' not found. CSV files available: {found or 'none'}. "
                         "Ask the user for the correct path - do not guess.")
    if os.path.getsize(p) > MAX_CSV_BYTES:
        raise ValueError("File too large for a two-qubit tomography (max 1 MB).")
    return p


def read_counts_csv(path: str, count_column: str = "auto"):
    """Read a coincidence-count CSV: one row per setting, a column with labels like 'HV'
    (first letter = qubit A) and one or more numeric count columns.
    Returns (counts, column_used, {numeric column: total})."""
    import csv
    with open(_resolve_csv(path), newline="", encoding="utf-8-sig") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh) if any(c.strip() for c in r)]
    header, body = rows[0], rows[1:]
    if all(_SETTING.match(c.upper()) or c.lstrip("-").isdigit() for c in header if c):
        header, body = [f"column {i + 1}" for i in range(len(rows[0]))], rows   # no header row
    width = max(len(r) for r in body)
    header = header + [""] * (width - len(header))
    cols = {i: [r[i] if i < len(r) else "" for r in body] for i in range(width)}
    label_cols = [i for i, v in cols.items() if v and all(_SETTING.match(x.upper()) for x in v)]
    if not label_cols:
        raise ValueError("No column with measurement settings like 'HH', 'HV', 'DA', 'RL' was found.")
    labels = [x.upper() for x in cols[label_cols[0]]]
    if len(set(labels)) != len(labels):
        dup = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(f"Settings appear more than once: {dup}. Merge or remove duplicate rows.")

    def as_number(x):
        try:
            return float(x)
        except ValueError:
            return None

    numeric = {}
    for i, v in cols.items():
        vals = [as_number(x) for x in v]
        if i not in label_cols and all(x is not None for x in vals):
            name = header[i] or f"column {i + 1}"
            numeric[name] = vals
    if not numeric:
        raise ValueError("No numeric count column was found.")
    totals = {k: int(round(sum(v))) for k, v in numeric.items()}
    if count_column == "auto":
        used = max(numeric, key=lambda k: totals[k])           # first column with the largest total
    else:
        match = [k for k in numeric if k.replace(" ", "").lower() == count_column.replace(" ", "").lower()]
        if not match:
            raise ValueError(f"Column '{count_column}' not found. Numeric columns: {list(numeric)}")
        used = match[0]
    vals = numeric[used]
    if any(x < 0 for x in vals):
        raise ValueError(f"Column '{used}' has negative counts (over-subtracted accidentals?).")
    return dict(zip(labels, vals)), used, totals


def _bootstrap_states(handle: str):
    """N_RESAMPLES states reconstructed from Poisson resamplings of the fitted counts of a state
    made from data (None for a state that was not measured). Computed once and cached, so every
    ± of one state comes from the same resamplings."""
    if handle in _SAMPLES:
        return _SAMPLES[handle]
    fit = _FIT.get(handle)
    if fit is None:
        return None
    rng = np.random.default_rng(1)
    states = [qt.mle_tomography_counts(dict(zip(fit["labels"], rng.poisson(fit["mu"]))),
                                       r_sign=fit["r_sign"], restarts=0) for _ in range(N_RESAMPLES)]
    _SAMPLES[handle] = states
    return states


@tool
def reconstruct_state_from_counts(csv_path: str, count_column: str = "auto",
                                  r_convention: str = "R=(H+iV)/sqrt2",
                                  error_bars: bool = True) -> dict:
    """Tomography of EXPERIMENTAL two-qubit coincidence counts in a CSV file (maximum likelihood,
    always physical, with 1-sigma error bars). Stores the state and returns its handle and report.
    csv_path: exactly as the user wrote it. count_column: 'auto' or a column name, e.g. 'L1R1'.
    r_convention: 'R=(H+iV)/sqrt2' (default) or 'R=(H-iV)/sqrt2'."""
    if r_convention.replace(" ", "") not in ("R=(H+iV)/sqrt2", "R=(H-iV)/sqrt2"):
        raise ValueError("r_convention must be 'R=(H+iV)/sqrt2' or 'R=(H-iV)/sqrt2'.")
    r_sign = +1 if "+i" in r_convention else -1
    other_convention = "R=(H-iV)/sqrt2" if r_sign > 0 else "R=(H+iV)/sqrt2"
    counts, used, totals = read_counts_csv(csv_path, count_column)
    labels = list(counts)
    P = qt.tomography_projectors(labels, r_sign)
    rank = np.linalg.matrix_rank(P.reshape(len(labels), 16), tol=1e-9)
    if rank < 16:
        raise ValueError(f"These {len(labels)} settings cannot determine a two-qubit state "
                         f"(rank {rank} < 16). At least the standard 16 settings are needed.")
    rho = qt.mle_tomography_counts(counts, r_sign=r_sign, rng=0)
    n = np.array([counts[k] for k in labels], dtype=float)
    mu = qt.predicted_counts(rho, labels, n.sum(), r_sign)
    dof = max(len(labels) - 16, 1)
    chi2 = float(np.sum((n - mu) ** 2 / np.clip(mu, 1, None)))
    worst = np.argsort(-np.abs(n - mu) / np.sqrt(np.clip(mu, 1, None)))[:3]

    def key_numbers(r):
        return {"purity": qt.purity(r), "concurrence": qt.concurrence(r),
                "negativity": qt.negativity(r), "chsh_max": qt.chsh_max(r),
                "bell_like_fidelity": _best_bell_like(r)["fidelity"]}

    from scipy.stats import chi2 as chi2_dist
    p_value = float(chi2_dist.sf(chi2, dof))
    verdict = ("consistent with counting noise" if p_value > 0.01 else
               "NOT consistent with counting noise alone: systematic errors such as "
               "wave-plate or alignment errors, or drift of the pair rate, are present")
    best = _best_bell_like(rho)
    w, V = np.linalg.eigh(rho)
    desc = (f"reconstructed from {csv_path} ({used}, MLE, {len(labels)} settings, "
            f"{r_convention}); full report: get_tomography_report('<this handle>')")
    handle = _save(rho, desc)
    result = {
        "handle": handle,
        "data": {"file": csv_path, "settings": len(labels), "total_counts": int(n.sum()),
                 "count_column_used": used, "all_numeric_columns_with_totals": totals},
        "method": "Poisson maximum likelihood, rho = T T^dagger / Tr (always physical)",
        "conventions": {"setting_label": "first letter = qubit A, second letter = qubit B",
                        "basis": "D=(H+V)/√2, A=(H-V)/√2, " + r_convention.replace("sqrt2", "√2"),
                        "log_base": 2},
        "assumptions": ["raw counts used as given (no accidental-coincidence subtraction)",
                        "same pair rate and counting time for every setting",
                        "ideal wave plates and polarizers"],
        "key_numbers": {k: round(float(v), 4) for k, v in key_numbers(rho).items()},
        "eigenvalues": [round(float(x), 4) for x in w[::-1]],
        "dominant_component": {"weight": round(float(w[-1]), 4), "ket": _ket_text(V[:, -1])},
        "closest_maximally_entangled": best,
        "convention_note": (
            f"Closest-state phase: {best['phi_deg']}° with {r_convention.replace('sqrt2', '√2')} (this "
            f"reconstruction), {-best['phi_deg']}° with {other_convention.replace('sqrt2', '√2')}. "
            "The two conventions give complex-conjugate matrices (every phase flips sign): they are "
            "DIFFERENT states, and only the one whose R/L definition matches how the R and L settings "
            "were really made describes the lab state. Purity, concurrence, negativity, CHSH and all "
            "entropies are the same for both."),
        "fit_quality": {
            "chi2_per_dof": round(chi2 / dof, 2),
            "dof": dof,
            "p_value": float(f"{p_value:.2g}"),
            "verdict": verdict,
            "error_bar_scale": (f"multiply the statistical error bars by about "
                                f"√(χ²/dof) = {np.sqrt(chi2 / dof):.2f} for a realistic uncertainty")
                               if chi2 / dof > 1.5 else "statistical error bars are realistic",
            "largest_deviations": [f"{labels[i]}: measured {int(n[i])}, fit {mu[i]:.0f} "
                                   f"({(n[i] - mu[i]) / np.sqrt(max(mu[i], 1)):+.1f}σ)" for i in worst]},
    }
    _FIT[handle] = {"labels": labels, "mu": mu, "n": n, "r_sign": r_sign, "chi2_per_dof": chi2 / dof,
                    "source": f"{csv_path} (column {used})"}
    if error_bars:
        states = _bootstrap_states(handle)
        samples = [key_numbers(r) for r in states]
        raw = {k: float(np.std([s[k] for s in samples], ddof=1)) for k in samples[0]}
        result["uncertainty_1sigma"] = {k: _two_sig(v) for k, v in raw.items()}
        result["uncertainty_method"] = (f"{N_RESAMPLES} Poisson resamplings of the fitted counts "
                                        "(statistical error only; quoted to 2 significant figures)")
        scale = float(np.sqrt(chi2 / dof))
        if scale > np.sqrt(1.5):                  # poor fit: give the scaled errors explicitly
            result["uncertainty_realistic"] = {k: _two_sig(v * scale) for k, v in raw.items()}
            result["uncertainty_realistic_method"] = (f"statistical errors × √(χ²/dof) = "
                                                      f"× {scale:.2f} (accounts for the misfit)")
    result = _clean(result)
    TOMO[handle] = result
    return result


@tool
def get_tomography_report(handle: str) -> dict:
    """Return the stored tomography report (fit quality, largest deviations, eigenvalues, closest
    maximally entangled state, uncertainties, assumptions) of a state made by
    reconstruct_state_from_counts. Use this instead of reconstructing again."""
    if handle not in TOMO:
        raise ValueError(f"{handle} was not reconstructed from data. Reconstructed states: {list(TOMO)}")
    return TOMO[handle]


@tool
def compare_states(handle_a: str, handle_b: str) -> dict:
    """Fidelity F = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2 (squared convention) and
    trace distance D = (1/2)||rho - sigma||_1 between two stored states."""
    a, b = _same_size([handle_a, handle_b])
    return {"handles": [handle_a, handle_b],
            "states": {handle_a: DESC.get(handle_a, ""), handle_b: DESC.get(handle_b, "")},
            "fidelity": round(qt.fidelity(a, b), 6),
            "trace_distance": round(qt.trace_distance(a, b), 6),
            "convention": "fidelity squared (1 = identical, 0 = orthogonal); 0 <= D <= 1"}


# ------------------------------------------------------------------ time evolution (Lindblad)
PROCESSES = {
    # local processes (each chosen photon has its own environment)
    "dephasing":       "L = σz at rate γ/2 on each chosen qubit: coherence factor λ(t) = e^(-γt)",
    "amplitude_decay": "L = σ- = |H><V| at rate γ: V population decays as e^(-γt) (T1 decay V -> H)",
    "depolarizing":    "L = σx, σy, σz at rate γ/4 each: Bloch vector shrinks as e^(-γt)",
    "bit_flip":        "L = σx at rate γ/2 on each chosen qubit: flip probability (1 - e^(-γt))/2",
    "bit_phase_flip":  "L = σy at rate γ/2 on each chosen qubit: flip probability (1 - e^(-γt))/2",
    "none":            "no dissipation (only the Hamiltonian acts)",
}
GLOBAL_PROCESSES = {
    # global processes (one common environment for the pair; `qubits` does not apply)
    "global_depolarizing": "L = all 15 σi⊗σj except I⊗I at rate γ/16 each: rho(t) = e^(-γt) rho + (1 - e^(-γt)) I/4",
    "global_dephasing":    "L = |k><k| for k = HH, HV, VH, VV at rate γ: every coherence decays as e^(-γt)",
    "collective_dephasing": ("ONE jump operator L = σz⊗I + I⊗σz at rate γ/2 (common random phase): coherences "
                             "with one more V decay as e^(-γt), two more V as e^(-4γt), HV-VH never"),
    "collective_decay":    ("ONE jump operator L = σ-⊗I + I⊗σ- at rate γ (common decay V -> H, superradiance): "
                            "psi- is dark (never decays), psi+ decays at 2γ"),
    "correlated_bit_flip":   "L = σx⊗σx at rate γ/2: both photons flip together, probability (1 - e^(-γt))/2",
    "correlated_phase_flip": "L = σz⊗σz at rate γ/2: both photons get Z together, probability (1 - e^(-γt))/2",
}
_SIGMA = {"x": qt.X, "y": qt.Y, "z": qt.Z}
_LOWER = np.array([[0, 1], [0, 0]], dtype=complex)          # |H><V| = |0><1|


def _on(op1: np.ndarray, which: str) -> np.ndarray:
    return np.kron(op1, qt.I2) if which == "A" else np.kron(qt.I2, op1)


def one_qubit_model(process: str, gamma: float, omega: float = 0.0, axis: str = "x"):
    """(H, jump operators, rates) for one photon."""
    if process not in PROCESSES:
        raise ValueError(f"For one photon the process must be one of {list(PROCESSES)} (the global processes "
                         "need two photons).")
    if axis not in _SIGMA:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    if gamma < 0:
        raise ValueError("gamma (a rate) must be >= 0")
    H = (omega / 2) * _SIGMA[axis] if omega else np.zeros((2, 2), complex)
    ops = {"dephasing": ([qt.Z], [gamma / 2]), "amplitude_decay": ([_LOWER], [gamma]),
           "depolarizing": ([qt.X, qt.Y, qt.Z], [gamma / 4] * 3), "bit_flip": ([qt.X], [gamma / 2]),
           "bit_phase_flip": ([qt.Y], [gamma / 2]), "none": ([], [])}[process]
    return H, list(ops[0]), list(ops[1])


def two_qubit_model(process: str, gamma: float, qubits: str, omega: float = 0.0, axis: str = "x"):
    """(H, jump operators, rates) for a local or global two-qubit Lindblad model."""
    if process not in PROCESSES and process not in GLOBAL_PROCESSES:
        raise ValueError(f"process must be one of {list(PROCESSES)} (local) or {list(GLOBAL_PROCESSES)} (global)")
    if qubits not in QUBITS:
        raise ValueError("qubits must be 'both', 'A' or 'B'")
    if axis not in _SIGMA:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    if gamma < 0:
        raise ValueError("gamma (a rate) must be >= 0")
    which = ["A", "B"] if qubits == "both" else [qubits]
    H = sum((omega / 2) * _on(_SIGMA[axis], w) for w in which) if omega else np.zeros((4, 4), complex)
    Ls, rates = [], []
    for w in which:
        if process == "dephasing":
            Ls.append(_on(qt.Z, w)); rates.append(gamma / 2)
        elif process == "amplitude_decay":
            Ls.append(_on(_LOWER, w)); rates.append(gamma)
        elif process == "depolarizing":
            for P in (qt.X, qt.Y, qt.Z):
                Ls.append(_on(P, w)); rates.append(gamma / 4)
        elif process == "bit_flip":
            Ls.append(_on(qt.X, w)); rates.append(gamma / 2)
        elif process == "bit_phase_flip":
            Ls.append(_on(qt.Y, w)); rates.append(gamma / 2)
    if process in GLOBAL_PROCESSES:
        Ls, rates = [], []
        if process == "global_depolarizing":
            for a in qt.PAULI:
                for b in qt.PAULI:
                    if a is qt.I2 and b is qt.I2:
                        continue
                    Ls.append(np.kron(a, b)); rates.append(gamma / 16)
        elif process == "global_dephasing":
            for k in range(4):
                P = np.zeros((4, 4), complex); P[k, k] = 1
                Ls.append(P); rates.append(gamma)
        elif process == "collective_dephasing":
            Ls, rates = [np.kron(qt.Z, qt.I2) + np.kron(qt.I2, qt.Z)], [gamma / 2]
        elif process == "collective_decay":
            Ls, rates = [qt.COLLECTIVE_LOWERING], [gamma]
        elif process == "correlated_bit_flip":
            Ls, rates = [np.kron(qt.X, qt.X)], [gamma / 2]
        elif process == "correlated_phase_flip":
            Ls, rates = [np.kron(qt.Z, qt.Z)], [gamma / 2]
    return H, Ls, rates


@tool
def evolve_state(handle: str, process: str, gamma: float, t_max: float, qubits: str = "both",
                 omega: float = 0.0, axis: str = "x", plot_elements: bool = False) -> dict:
    """Continuous-time evolution (Lindblad master equation, ħ = 1) of a stored state; gamma = rate (1/time).
    process, LOCAL (each chosen photon has its own environment; qubits 'both', 'A' or 'B'):
      'dephasing', 'amplitude_decay', 'depolarizing', 'bit_flip', 'bit_phase_flip', 'none'
    GLOBAL (one common environment for the pair; qubits does not apply):
      'global_depolarizing', 'global_dephasing', 'collective_dephasing' (common random phase: psi± immune),
      'collective_decay' (common V -> H decay: psi- dark), 'correlated_bit_flip', 'correlated_phase_flip'
    Optional local Hamiltonian H = (omega/2) σ_axis on the chosen qubits (axis 'x', 'y' or 'z'), e.g. a
    birefringent element; omega = 0 means none. A ONE-photon state takes the local processes.
    plot_elements=True also plots every density-matrix element (Re and Im) versus time.
    Stores the state at t_max (new handle) and saves a plot of concurrence, negativity, purity
    and CHSH/2 versus time (one photon: degree of polarization, S1, S2, S3, purity). Returns 'features':
    crossing times, the measures AT those times ('at_key_times') and at evenly spaced 'samples'. Quote a
    value only at a time listed there."""
    if t_max <= 0:
        raise ValueError("t_max must be > 0")
    rho0 = _get(handle)
    if rho0.shape[0] == 2:
        return _evolve_one_photon(handle, rho0, process, gamma, t_max, omega, axis, plot_elements)
    _check_named_bell_state(handle)
    H, Ls, rates = two_qubit_model(process, gamma, qubits, omega, axis)
    is_global = process in GLOBAL_PROCESSES
    n = 2001
    times = np.linspace(0, t_max, n)
    traj = qt.lindblad_trajectory(rho0, H, Ls, rates, times)
    names = ("concurrence", "negativity", "purity", "chsh_max")
    rows = np.array([(qt.concurrence(r), qt.negativity(r), qt.purity(r), qt.chsh_max(r)) for r in traj])

    where = "the pair (one common environment)" if is_global else QUBITS[qubits]
    ham = f" + H = ({omega:g}/2)σ{axis}" if omega else ""
    if is_global and omega:
        ham += f" on {QUBITS[qubits]}"
    desc = f"{handle} after {process} (γ={gamma:g}){ham} on {where} for t={t_max:g}"
    new = _save(traj[-1], desc)

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, name in enumerate(names[:3]):       # wider first: equal curves (C = purity = 1) both stay visible
        ax.plot(times, rows[:, col], label=name, lw=3.4 - 0.6 * col)
    ax.plot(times, rows[:, 3] / 2, label="CHSH max / 2", lw=1.5)
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(f"time t (same units as 1/γ; γ = {gamma:g}" + (f", ω = {omega:g})" if omega else ")"))
    ax.set_ylim(-0.03, 1.5)
    label = DESC.get(handle, handle)
    if label.startswith("reconstructed from"):
        label = f"{handle} ({label.split()[2]})"          # e.g. rho_9 (data/state_tomo_data.csv)
    elif len(label) > 40:
        label = handle
    ax.set_title(f"{label}: {process}{ham} on {where}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=8)
    path = pl.figure_path(f"{handle}_{process}_{qubits}_evolution", [gamma, t_max, omega, axis])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    at = lambda row: {k: round(float(v), 4) for k, v in zip(names, row)}  # noqa: E731
    Lsup = qt.liouvillian(H, Ls, rates)

    def state_at(t):
        from scipy.linalg import expm
        return (expm(Lsup * t) @ rho0.reshape(-1)).reshape(4, 4)

    def crossing(col, level, measure):
        """First time the measure falls to `level`: grid bracket, then bisection on the exact state."""
        rough = _first_crossing(times, rows[:, col], level)
        if rough is None:
            return None
        i = int(np.searchsorted(times, rough))
        lo, hi = times[max(i - 1, 0)], times[min(i, n - 1)]
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if measure(state_at(mid)) > level else (lo, mid)
        return (lo + hi) / 2

    vanish = crossing(0, 1e-9, qt.concurrence)
    if vanish is not None and not _separable(traj[int(np.argmax(rows[:, 0] <= 1e-9))]):
        vanish = None                     # only decayed below 1e-9: asymptotic, not sudden death
    chsh_cross = crossing(3, 2.0, qt.chsh_max)
    if chsh_cross is not None and _only_approaches(rows[int(np.argmax(rows[:, 3] <= 2.0)):, 3], 2.0):
        chsh_cross = None                 # CHSH only approaches 2 (asymptotically), it does not drop below
    half = crossing(0, rows[0, 0] / 2, qt.concurrence) if rows[0, 0] > 1e-9 else None
    revives = bool(vanish is not None and rows[-1, 0] > 1e-6)
    features = {
        "model": f"{ {**PROCESSES, **GLOBAL_PROCESSES}[process]}{'; ' + ham.strip(' +') if ham else ''}; acting on {where}",
        "time_units": "same units as 1/gamma (hbar = 1)",
        "entanglement_vanishes_at": (round(vanish, 4) if vanish is not None else
            ("not entangled even at t = 0" if rows[0, 0] <= 1e-9 else
             (f"never: concurrence is constant (C = {rows[0, 0]:.4f} at all times)"
              if np.ptp(rows[:, 0]) < 1e-6 else
              f"never before t_max: concurrence stays > 0 (C(t_max) = {rows[-1, 0]:.4f}); "
              "it decays asymptotically, no sudden death in this time window"))),
        "concurrence_halves_at": round(half, 4) if half is not None else "not within t_max",
        "chsh_drops_to_2_at": (round(chsh_cross, 4) if chsh_cross is not None else
            ("never above 2" if rows[0, 3] <= 2 else f"stays above 2 up to t_max (CHSH = {rows[-1, 3]:.4f})")),
        "at_t0": at(rows[0]),
        "at_t_max": at(rows[-1]),
        "samples": {f"t={t:.4g}": at(rows[i]) for t, i in
                    ((times[i], i) for i in np.linspace(0, n - 1, 11).astype(int))},
    }
    key = {}
    for t, why in ((half, "concurrence halves"), (chsh_cross, "CHSH reaches 2"),
                   (vanish, "entanglement vanishes")):
        if t is not None:
            r = state_at(t)
            key[f"t={t:.4f} ({why})"] = at((qt.concurrence(r), qt.negativity(r), qt.purity(r), qt.chsh_max(r)))
    features["at_key_times"] = key
    if revives:
        features["entanglement_revives"] = f"yes: C is zero at t = {vanish:.4f} but {rows[-1, 0]:.4f} at t_max"
    if process == "none":
        features["note"] = "a local Hamiltonian is a local unitary: it cannot change entanglement or purity"
    if process in _TWIN and not omega:
        features["same_as_kraus_at_t_max"] = _twin_text(process, gamma * t_max)
    out = {"final_state": new, "figure": path, "features": features}
    if plot_elements:
        out["elements_figure"] = _elements_figure(handle, process, qubits, times, traj,
                                                  f"{label}: {process}{ham} on {where}", [gamma, omega, axis])
    return _clean(out)


# the Kraus channel equal to a Lindblad process at time t: (tool, channel, parameter as a function of γt)
_TWIN = {"dephasing": ("apply_local_noise", "phase_damping", "λ = e^(-γt)", lambda g: np.exp(-g)),
         "amplitude_decay": ("apply_local_noise", "amplitude_damping", "γ_K = 1 - e^(-γt)", lambda g: 1 - np.exp(-g)),
         "depolarizing": ("apply_local_noise", "depolarizing", "p = 1 - e^(-γt)", lambda g: 1 - np.exp(-g)),
         "bit_flip": ("apply_local_noise", "bit_flip", "p = (1 - e^(-γt))/2", lambda g: (1 - np.exp(-g)) / 2),
         "bit_phase_flip": ("apply_local_noise", "bit_phase_flip", "p = (1 - e^(-γt))/2",
                            lambda g: (1 - np.exp(-g)) / 2),
         "global_depolarizing": ("apply_global_noise", "global_depolarizing", "p = 1 - e^(-γt)", lambda g: 1 - np.exp(-g)),
         "global_dephasing": ("apply_global_noise", "global_dephasing", "p = 1 - e^(-γt)", lambda g: 1 - np.exp(-g)),
         "collective_dephasing": ("apply_global_noise", "collective_dephasing", "p = 1 - e^(-γt)",
                                  lambda g: 1 - np.exp(-g)),
         "collective_decay": ("apply_global_noise", "collective_amplitude_damping", "p = 1 - e^(-γt)",
                              lambda g: 1 - np.exp(-g)),
         "correlated_bit_flip": ("apply_global_noise", "correlated_bit_flip", "p = (1 - e^(-γt))/2",
                                 lambda g: (1 - np.exp(-g)) / 2),
         "correlated_phase_flip": ("apply_global_noise", "correlated_phase_flip", "p = (1 - e^(-γt))/2",
                                   lambda g: (1 - np.exp(-g)) / 2)}


def _twin_text(process: str, gt: float) -> str:
    tool_name, channel, rule, f = _TWIN[process]
    return f"{tool_name}('{channel}', {rule} = {f(gt):.4f})"


def _elements_figure(handle, process, qubits, times, traj, title, settings=()) -> str:
    path = pl.figure_path(f"{handle}_{process}_{qubits}_elements",
                          [process, qubits, float(times[-1]), len(times), *settings])
    basis = BASIS_1 if traj[0].shape[0] == 2 else BASIS_2
    pl.matrix_elements_figure(times, np.array(traj), basis, title, path)
    return path


def _evolve_one_photon(handle, rho0, process, gamma, t_max, omega, axis, plot_elements) -> dict:
    H, Ls, rates = one_qubit_model(process, gamma, omega, axis)
    n = 801
    times = np.linspace(0, t_max, n)
    traj = qt.lindblad_trajectory(rho0, H, Ls, rates, times)
    S = np.array([qt.stokes_vector(r) for r in traj])
    dop, pur = np.linalg.norm(S, axis=1), np.array([qt.purity(r) for r in traj])
    ham = f" + H = ({omega:g}/2)σ{axis}" if omega else ""
    desc = f"1 photon: {handle} after {process} (γ={gamma:g}){ham} for t={t_max:g}"
    new = _save(traj[-1], desc)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, dop, label="degree of polarization", lw=4.5, color="#c3c2b7", zorder=2)   # wide grey band
    ax.plot(times, pur, label="purity", ls="--", color=pl.SERIES[4], zorder=3)
    for col, name in enumerate(("S1 (H-V)", "S2 (D-A)", "S3 (R-L)")):          # thin lines on top
        ax.plot(times, S[:, col], label=name, color=pl.SERIES[col], lw=1.6 - 0.3 * col, zorder=4 + col)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(f"time t (same units as 1/γ; γ = {gamma:g}" + (f", ω = {omega:g})" if omega else ")"))
    ax.set_title(f"{handle} (one photon): {process}{ham}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=8)
    path = pl.figure_path(f"{handle}_{process}_1photon_evolution", [gamma, t_max, omega, axis])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    at = lambda i: {"S1": round(float(S[i, 0]), 4) + 0.0, "S2": round(float(S[i, 1]), 4) + 0.0,  # noqa: E731
                    "S3": round(float(S[i, 2]), 4) + 0.0, "degree_of_polarization": round(float(dop[i]), 4),
                    "purity": round(float(pur[i]), 4)}
    half = None
    i_half = next((i for i in range(1, n) if dop[0] > 1e-9 and dop[i] <= dop[0] / 2), None)
    if i_half is not None:                       # bisection on the exact state between the two grid times
        from scipy.linalg import expm
        Lsup = qt.liouvillian(H, Ls, rates)
        dop_at = lambda t: np.linalg.norm(qt.stokes_vector((expm(Lsup * t) @ rho0.reshape(-1)).reshape(2, 2)))  # noqa: E731
        lo, hi = times[i_half - 1], times[i_half]
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if dop_at(mid) > dop[0] / 2 else (lo, mid)
        half = (lo + hi) / 2
    key = {}
    if half is not None:                          # exact values at the halving time (quotable by the model)
        from scipy.linalg import expm
        r_h = (expm(qt.liouvillian(H, Ls, rates) * half) @ rho0.reshape(-1)).reshape(2, 2)
        s_h = qt.stokes_vector(r_h)
        key[f"t={round(half, 4):.4g} (degree of polarization halves)"] = {
            "S1": round(float(s_h[0]), 4) + 0.0, "S2": round(float(s_h[1]), 4) + 0.0,
            "S3": round(float(s_h[2]), 4) + 0.0, "degree_of_polarization": round(float(np.linalg.norm(s_h)), 4),
            "purity": round(float(qt.purity(r_h)), 4)}
    features = {"model": f"{PROCESSES[process]}{'; ' + ham.strip(' +') if ham else ''}; one photon",
                "time_units": "same units as 1/gamma (hbar = 1)",
                "at_t0": at(0), "at_t_max": at(n - 1),
                "degree_of_polarization_halves_at": round(half, 4) if half is not None else "not within t_max",
                "at_key_times": key,
                "samples": {f"t={times[i]:.4g}": at(i) for i in np.linspace(0, n - 1, 11).astype(int)}}
    if process in _TWIN and not omega:
        features["same_as_kraus_at_t_max"] = _twin_text(process, gamma * t_max)
    out = {"final_state": new, "figure": path, "features": features}
    if plot_elements:
        out["elements_figure"] = _elements_figure(handle, process, "1photon", times, traj,
                                                  f"{handle} (one photon): {process}{ham}", [gamma, omega, axis])
    return _clean(out)


# ------------------------------------------------------------------ wave plates (local unitaries)
_RETARDANCE = {"HWP": 180.0, "QWP": 90.0}
_KNOWN_1Q = {"H": [1, 0], "V": [0, 1], "D": [1, 1], "A": [1, -1], "R": [1, 1j], "L": [1, -1j]}


def _ket1_text(v: np.ndarray) -> str:
    """'0.707|H> - 0.707|V> (= A)' for a one-photon polarization, global phase removed
    (the first of the largest amplitudes is made real and positive)."""
    big = np.abs(v).max()
    ref = next(c for c in v if abs(c) > big - 1e-9)
    v = v * np.exp(-1j * np.angle(ref))
    terms = []
    for c, k in zip(v, "HV"):
        if abs(c) < 5e-4:
            continue
        ph = np.degrees(np.angle(c))
        if abs(abs(ph) - 180) < 0.05:
            terms.append(("-", f"{abs(c):.3f}|{k}>"))
        else:
            terms.append(("+", f"{abs(c):.3f}{_phase_text(ph, 1)}|{k}>"))
    text = " ".join(f"{s} {t}" for s, t in terms).lstrip("+ ")
    text = text if not text.startswith("- ") else "-" + text[2:]
    name = [k for k, w in _KNOWN_1Q.items()
            if abs(np.vdot(np.array(w) / np.linalg.norm(w), v)) ** 2 > 1 - 1e-6]
    return text + (f" (= {name[0]})" if name else "")


def _local_numbers(rho: np.ndarray) -> dict:
    return {"concurrence": round(qt.concurrence(rho), 4), "negativity": round(qt.negativity(rho), 4),
            "purity": round(qt.purity(rho), 4), "chsh_max": round(qt.chsh_max(rho), 4),
            "fidelity_with_bell_states": {b: round(qt.fidelity(rho, qt.dm(qt.bell_state(b))), 4)
                                          for b in ("phi+", "phi-", "psi+", "psi-")}}


def _matrix_text(U: np.ndarray) -> list[list[str]]:
    U = U * np.exp(-1j * np.angle(U[np.unravel_index(np.argmax(np.abs(U)), U.shape)]))
    fmt = lambda z: f"{z.real:+.4f}{z.imag:+.4f}i"       # noqa: E731
    return [[fmt(z) for z in row] for row in U]


@tool
def apply_waveplate(handle: str, plate: str, angle_deg: float, qubit: str,
                    retardance_deg: float | None = None) -> dict:
    """Send ONE photon (qubit 'A' or 'B') of a stored two-qubit state through a wave plate and
    store the result (new handle). plate: 'HWP' (half-wave), 'QWP' (quarter-wave) or 'retarder'
    (then give retardance_deg). angle_deg: fast-axis angle measured from H, in degrees.
    Jones matrix U = cos(δ/2) I - i sin(δ/2) (cos2θ σz + sin2θ σx) (global phase irrelevant):
    HWP at 22.5° turns H into D, HWP at 45° swaps H and V, QWP at 45° turns H into (H - iV)/√2.
    Returns what the plate does to H and V, and the numbers before and after (a wave plate is a
    local unitary: entanglement, purity and CHSH never change; fidelities and phases can)."""
    plate = plate.strip().upper()
    if plate not in ("HWP", "QWP", "RETARDER"):
        raise ValueError("plate must be 'HWP', 'QWP' or 'retarder'.")
    if plate == "RETARDER":
        if retardance_deg is None:
            raise ValueError("A 'retarder' needs retardance_deg (e.g. 90 for a quarter-wave plate).")
        delta = float(retardance_deg)
    else:
        if retardance_deg is not None and abs(retardance_deg - _RETARDANCE[plate]) > 1e-9:
            raise ValueError(f"A {plate} has retardance {_RETARDANCE[plate]:g}°; use plate='retarder' "
                             "for another retardance.")
        delta = _RETARDANCE[plate]
    rho = _get(handle)
    if rho.shape[0] == 2:
        return _waveplate_one_photon(handle, rho, plate, delta, angle_deg)
    qubit = qubit.strip().upper()
    if qubit not in ("A", "B"):
        raise ValueError("qubit must be 'A' or 'B' (a wave plate acts on one photon).")
    _check_named_bell_state(handle)
    U1 = qt.waveplate(angle_deg, np.deg2rad(delta))
    U = qt.on_qubit(U1, qubit)
    out = U @ rho @ U.conj().T
    name = plate if plate != "RETARDER" else f"retarder ({delta:g}°)"
    desc = f"{handle} after {name} at {angle_deg:g}° on qubit {qubit}"
    new = _save(out, desc)
    return _clean({
        "new_handle": new,
        "element": f"{name}, fast axis at {angle_deg:g}° from H, on qubit {qubit}",
        "jones_matrix_HV_basis": _matrix_text(U1),
        "acts_on_H": "H -> " + _ket1_text(U1 @ np.array([1, 0], complex)),
        "acts_on_V": "V -> " + _ket1_text(U1 @ np.array([0, 1], complex)),
        "before": _local_numbers(rho),
        "after": _local_numbers(out),
        "note": "local unitary: concurrence, negativity, purity and CHSH are unchanged",
        "conventions": "R=(H+iV)/√2, L=(H-iV)/√2; global phases dropped",
    })


def _stokes_numbers(rho: np.ndarray) -> dict:
    s = qt.stokes_vector(rho)
    ell = qt.polarization_ellipse(s)
    out = {"S1 (H-V)": round(float(s[0]), 4) + 0.0, "S2 (D-A)": round(float(s[1]), 4) + 0.0,
           "S3 (R-L)": round(float(s[2]), 4) + 0.0, "degree_of_polarization": round(float(np.linalg.norm(s)), 4),
           "purity": round(qt.purity(rho), 4)}
    if ell:
        out["ellipse_azimuth_deg"] = (round(ell["azimuth_deg"], 2) + 0.0 if ell["azimuth_deg"] is not None
                                      else "undefined (circular)")
        out["ellipticity_deg"] = round(ell["ellipticity_deg"], 2) + 0.0
    return out


def _waveplate_one_photon(handle, rho, plate, delta, angle_deg) -> dict:
    U1 = qt.waveplate(angle_deg, np.deg2rad(delta))
    out = U1 @ rho @ U1.conj().T
    name = plate if plate != "RETARDER" else f"retarder ({delta:g}°)"
    new = _save(out, f"1 photon: {handle} after {name} at {angle_deg:g}°")
    w, V = np.linalg.eigh(out)
    return _clean({
        "new_handle": new, "photons": 1,
        "element": f"{name}, fast axis at {angle_deg:g}° from H",
        "jones_matrix_HV_basis": _matrix_text(U1),
        "acts_on_H": "H -> " + _ket1_text(U1 @ np.array([1, 0], complex)),
        "acts_on_V": "V -> " + _ket1_text(U1 @ np.array([0, 1], complex)),
        "before": _stokes_numbers(rho), "after": _stokes_numbers(out),
        "polarized_part_before": (_ket1_text(np.linalg.eigh(rho)[1][:, -1])
                                  if np.linalg.norm(qt.stokes_vector(rho)) > 1e-9 else "none"),
        "polarized_part_after": _ket1_text(V[:, -1]) if np.linalg.norm(qt.stokes_vector(out)) > 1e-9 else "none",
        "note": "a wave plate rotates the Stokes vector on the Poincaré sphere: degree of polarization and "
                "purity are unchanged",
        "conventions": "R=(H+iV)/√2, L=(H-iV)/√2; global phases dropped"})


def _phase_compensation(rho: np.ndarray, qubit: str = "B"):
    """(target name, target dm, phase φ of the closest state, phase-shifter angle α in degrees)."""
    best = _best_bell_like(rho)
    target_name = "phi+" if "HH" in best["state"] else "psi+"
    target = qt.dm(qt.bell_state(target_name))
    a, b = ((0, 3) if target_name == "phi+" else (1, 2))
    phi = float(np.degrees(-np.angle(rho[a, b])))

    def shifted(alpha_deg):
        U = qt.on_qubit(np.diag([1, np.exp(1j * np.deg2rad(alpha_deg))]), qubit)
        return U @ rho @ U.conj().T

    alpha = max((-phi, phi), key=lambda x: qt.fidelity(shifted(x), target))
    return target_name, target, phi, (alpha + 180) % 360 - 180


@tool
def compensate_phase(handle: str, qubit: str = "B") -> dict:
    """Wave-plate setting that removes the relative phase of a (nearly) maximally entangled
    state: the closest state (|HH> + e^(iφ)|VV>)/√2 becomes phi+ (or (|HV> + e^(iφ)|VH>)/√2
    becomes psi+) with a phase shifter diag(1, e^(iα)) on one photon, built as
    QWP(45°) - HWP(θ) - QWP(45°). Returns φ, α, θ, the fidelity with phi+/psi+ before and after,
    and the compensated state (new handle). The fidelity after equals the closest-state fidelity:
    a phase shifter removes the phase error, not noise or mixedness."""
    qubit = qubit.strip().upper()
    if qubit not in ("A", "B"):
        raise ValueError("qubit must be 'A' or 'B'.")
    rho = _get2(handle, "compensate_phase")
    best = _best_bell_like(rho)
    target_name, target, phi, alpha = _phase_compensation(rho, qubit)
    theta = qt.phase_shifter_hwp_angle(alpha)
    U1 = qt.qwp(45) @ qt.hwp(theta) @ qt.qwp(45)
    if abs(U1[0, 1]) > 1e-12 or abs(U1[1, 1] / U1[0, 0] - np.exp(1j * np.deg2rad(alpha))) > 1e-9:
        raise RuntimeError("internal check failed: the plate sequence is not the wanted phase shifter")
    U = qt.on_qubit(U1, qubit)
    out = U @ rho @ U.conj().T
    new = _save(out, f"{handle} after phase compensation (QWP45-HWP{theta:.2f}-QWP45 on qubit {qubit})")
    F0, F1 = qt.fidelity(rho, target), qt.fidelity(out, target)
    result = {
        "new_handle": new,
        "closest_state": best["state"].replace("e^(iφ)", f"e^({'-' if phi < 0 else ''}i{abs(phi):.1f}°)")
                         + f", fidelity {best['fidelity']}",
        "phase_to_remove_deg": round(phi, 2),
        "phase_shifter": f"diag(1, e^(iα)) with α = {alpha:.2f}° on qubit {qubit}",
        "wave_plates": {"sequence": f"QWP at 45°, HWP at {theta:.2f}°, QWP at 45° (on qubit {qubit})",
                        "hwp_angle_deg": round(theta, 2),
                        "rule": "QWP(45°)·HWP(θ)·QWP(45°) = diag(1, e^(iα)) up to a global phase, "
                                "α = 180° + 4θ (mod 360°)",
                        "alternative": f"any retarder with its axis at 0° and retardance {alpha:.2f}° "
                                       "(e.g. a tilted wave plate or a Soleil-Babinet compensator)"},
        "target": target_name,
        "fidelity_with_target_before": round(F0, 4),
        "fidelity_with_target_after": round(F1, 4),
        "fidelity_gain": round(F1 - F0, 4),
        "unchanged": {"concurrence": round(qt.concurrence(rho), 4), "purity": round(qt.purity(rho), 4)},
        "lab_check": (f"after compensation the DD and AA coincidences should be maximal and DA, AD minimal. "
                      f"If they get worse instead (the plate's fast and slow axes swapped relative to this "
                      f"Jones convention), the phase sign is reversed: use the HWP at "
                      f"{qt.phase_shifter_hwp_angle(-alpha):.2f}°."),
    }
    if abs((alpha + 180) % 360 - 180) < 0.05:         # nothing to compensate
        result["no_compensation_needed"] = (
            f"the relative phase is already 0°, so no phase shifter is needed: the QWP–HWP–QWP setting "
            f"below acts as the identity (fidelity with {target_name} stays {F0:.4f})")
        result["wave_plates"]["alternative"] = "none needed"
        result["lab_check"] = result["lab_check"].split(" If they get worse")[0]
    if DESC.get(handle, "").startswith("reconstructed from"):
        other = qt.phase_shifter_hwp_angle(-alpha)
        result["convention_warning"] = (
            "this phase comes from a reconstruction, so its SIGN depends on the R/L convention used "
            f"there. With the opposite convention α = {-alpha:.2f}° and the HWP angle is {other:.2f}°. "
            "Use the setting from the convention that matches how R and L were made in the lab.")
    return _clean(result)


# ------------------------------------------------------------------ detector model / simulated tomography
SIM_LABELS = [a + b for a in "HVDARL" for b in "HVDARL"]


def _sim_numbers(r: np.ndarray, true: np.ndarray) -> dict:
    return {"concurrence": round(qt.concurrence(r), 4), "negativity": round(qt.negativity(r), 4),
            "purity": round(qt.purity(r), 4), "chsh_max": round(qt.chsh_max(r), 4),
            "fidelity_with_true_state": round(qt.fidelity(r, true), 4)}


@tool
def simulate_tomography(handle: str, pair_rate: float = 1e5, time_per_setting_s: float = 1.0,
                        eta_a: float = 0.2, eta_b: float = 0.2, dark_a: float = 500.0,
                        dark_b: float = 500.0, window_ns: float = 2.0,
                        target_concurrence_error: float = 0.01, seed: int = 0) -> dict:
    """Simulate a polarization-tomography EXPERIMENT on a stored state with realistic detectors:
    pair_rate R (pairs/s at the source), detection efficiencies eta_a, eta_b (0-1), dark counts
    dark_a, dark_b (1/s), coincidence window window_ns (ns), counting time per setting (s);
    one detector behind each analyser, all 36 settings (H,V,D,A,R,L on each photon).
    Singles S = R·η·p + dark, accidentals = S_A·S_B·τ·T, Poisson counts. Reconstructs the state by
    maximum likelihood from the RAW counts and with accidentals SUBTRACTED, compares both with the
    true state, writes the counts to a CSV file (path in the result), and estimates the counting time
    needed for a target concurrence error (σ ∝ 1/√counts)."""
    for name, v in (("eta_a", eta_a), ("eta_b", eta_b)):
        if not 0 < v <= 1:
            raise ValueError(f"{name} must be in (0, 1].")
    if min(pair_rate, time_per_setting_s, window_ns, target_concurrence_error) <= 0 or min(dark_a, dark_b) < 0:
        raise ValueError("pair_rate, time, window and target error must be > 0; dark counts >= 0.")
    rho = _get2(handle, "simulate_tomography")
    _check_named_bell_state(handle)
    tau = window_ns * 1e-9
    args = (rho, SIM_LABELS, pair_rate, time_per_setting_s, eta_a, eta_b, dark_a, dark_b, tau)
    sim = qt.simulate_experiment(*args, rng=seed)
    ideal = qt.simulate_experiment(*args, noiseless=True)
    n = sim["counts"]
    sub = {k: max(n[k] - sim["accidental_estimate"][k], 0.0) for k in SIM_LABELS}
    if sum(n.values()) < 50:
        raise ValueError(f"Only {sum(n.values())} coincidences in total: increase the pair rate, "
                         "efficiency or counting time.")
    rho_raw = qt.mle_tomography_counts(n, rng=0)
    rho_sub = qt.mle_tomography_counts(sub, rng=0)
    rho_inf = qt.mle_tomography_counts(ideal["counts"], rng=0)       # infinite statistics, raw

    mu = qt.predicted_counts(rho_raw, SIM_LABELS, float(sum(n.values())), +1)

    tot_true, tot_acc = sum(sim["true"].values()), sum(sim["accidental"].values())
    S_a = [pair_rate * eta_a * np.real(np.trace(rho @ np.kron(qt.pauli_projectors_1q()[k], qt.I2))) + dark_a
           for k in "HVDARL"]
    path = f"{_data_dir()}/simulated_{handle}.csv"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("counts_raw,accidentals_estimated,counts_subtracted,label\n")
        for k in SIM_LABELS:
            fh.write(f"{n[k]},{sim['accidental_estimate'][k]:.2f},{sub[k]:.2f},{k}\n")
    h_raw = _save(rho_raw, f"simulated tomography of {handle} (raw counts, MLE)")
    _FIT[h_raw] = {"labels": SIM_LABELS, "mu": mu, "r_sign": +1, "source": f"simulated tomography of {handle}",
                   "n": np.array([n[k] for k in SIM_LABELS], dtype=float),
                   "chi2_per_dof": float(np.sum((np.array([n[k] for k in SIM_LABELS]) - mu) ** 2
                                                / np.clip(mu, 1, None)) / (len(SIM_LABELS) - 16))}
    h_sub = _save(rho_sub, f"simulated tomography of {handle} (accidentals subtracted, MLE)")
    boot = _bootstrap_states(h_raw)          # the same resamplings certify_correlations uses
    sigma_c = float(np.std([qt.concurrence(r) for r in boot], ddof=1))
    sigma_s = float(np.std([qt.chsh_max(r) for r in boot], ddof=1))

    c_raw = qt.concurrence(rho_raw)
    if c_raw <= 3 * sigma_c:
        needed = "not estimated: the measured concurrence is too close to 0"
    elif sigma_c <= target_concurrence_error:
        needed = (f"already reached: σ_C = {_two_sig(sigma_c):#.2g} <= target {target_concurrence_error} "
                  "with the current counting time (no extrapolation to shorter times)")
    else:
        factor = (sigma_c / target_concurrence_error) ** 2
        needed = {"target_concurrence_error": target_concurrence_error,
                  "time_per_setting_s": float(f"{time_per_setting_s * factor:.2g}"),
                  "total_coincidences": int(float(f"{sum(n.values()) * factor:.2g}")),
                  "accuracy": ("rough, about ±30%: σ_C comes from a short run with few counts"
                               if sum(n.values()) < 5000 else "about ±20% (bootstrap estimate)"),
                  "rule": "σ_C ∝ 1/√(counts): time needed = T × (σ_C / target)²"}
    result = {
        "true_state": {"handle": handle, "description": DESC.get(handle, ""),
                       **{k: v for k, v in _sim_numbers(rho, rho).items() if k != "fidelity_with_true_state"}},
        "detector_model": {
            "pair_rate_per_s": pair_rate, "eta_a": eta_a, "eta_b": eta_b, "dark_a_per_s": dark_a,
            "dark_b_per_s": dark_b, "window_ns": window_ns, "time_per_setting_s": time_per_setting_s,
            "settings": len(SIM_LABELS),
            "formulas": "coincidences = R·η_A·η_B·Tr(ρ P_a⊗P_b)·T; singles S = R·η·p + dark; "
                        "accidentals = S_A·S_B·τ·T; accidental estimate from measured singles N_A·N_B·τ/T"},
        "rates": {"singles_a_per_s_range": [round(min(S_a), 1), round(max(S_a), 1)],
                  "true_coincidences_per_setting_mean": round(tot_true / len(SIM_LABELS), 2),
                  "accidentals_per_setting_mean": round(tot_acc / len(SIM_LABELS), 3),
                  "accidental_fraction": round(tot_acc / (tot_true + tot_acc), 5),
                  "accidental_percent": round(100 * tot_acc / (tot_true + tot_acc), 3),
                  "total_coincidences_recorded": int(sum(n.values()))},
        "reconstructed_raw": {"handle": h_raw, **_sim_numbers(rho_raw, rho),
                              "concurrence_error_1sigma": _two_sig(sigma_c),
                              "chsh_error_1sigma": _two_sig(sigma_s),
                              "chsh_violation_n_sigma": round((qt.chsh_max(rho_raw) - 2) / sigma_s, 1)
                              if sigma_s > 0 else None,
                              "chsh_violation_verdict": (_significance_verdict((qt.chsh_max(rho_raw) - 2) / sigma_s)
                                                         if sigma_s > 0 else "no error bar")
                                                        + (f"; few counts ({sum(n.values())} coincidences): σ and "
                                                           "nσ are only rough" if sum(n.values()) < 5000 else ""),
                              "deviation_from_true_in_sigma": {
                                  "concurrence": round((qt.concurrence(rho_raw) - qt.concurrence(rho)) / sigma_c, 1)
                                  if sigma_c > 0 else None,
                                  "chsh_max": round((qt.chsh_max(rho_raw) - qt.chsh_max(rho)) / sigma_s, 1)
                                  if sigma_s > 0 else None,
                                  "meaning": "|value| < 2: the raw result agrees with the true state within "
                                             "counting noise; larger: a real bias"}},
        "reconstructed_accidentals_subtracted": {"handle": h_sub, **_sim_numbers(rho_sub, rho)},
        "concurrence_change_vs_true": {
            "raw": round(qt.concurrence(rho_raw) - qt.concurrence(rho), 4),
            "accidentals_subtracted": round(qt.concurrence(rho_sub) - qt.concurrence(rho), 4),
            "raw_without_counting_noise": round(qt.concurrence(rho_inf) - qt.concurrence(rho), 4)},
        "infinite_statistics_raw": {**_sim_numbers(rho_inf, rho),
                                    "meaning": "raw counts without counting noise: the bias caused by "
                                               "accidentals alone"},
        "counts_needed_for_target": needed,
        "csv_file": session.display_path(path),           # 'uploads/simulated_rho_1.csv' for a web user
        "csv_note": "columns counts_raw, accidentals_estimated, counts_subtracted, label; it can be "
                    "reconstructed with reconstruct_state_from_counts (count_column picks the column)",
        "conventions": "R=(H+iV)/√2; fidelity squared; one detector per photon",
    }
    return _clean(result)


# ------------------------------------------------------------------ protocols: teleportation, QKD, bit-flip code
@tool
def teleportation_fidelity(handle: str) -> dict:
    """How well a stored two-qubit state works as the resource for teleporting one qubit
    (standard protocol, best local unitary correction). Returns the fully entangled fraction f
    (exact), the average teleportation fidelity F = (2f + 1)/3, the classical limit 2/3 (measure
    and prepare) and whether the state beats it (iff f > 1/2)."""
    rho = _get2(handle, "teleportation_fidelity")
    _check_named_bell_state(handle)
    f = qt.fully_entangled_fraction_2q(rho)
    F = (2 * f + 1) / 3
    return _clean({
        "handle": handle, "state": DESC.get(handle, ""),
        "fully_entangled_fraction": round(f, 4),
        "teleportation_fidelity": round(F, 4),
        "classical_limit": round(2 / 3, 4),
        "beats_classical": bool(f > 0.5 + 1e-9),
        "margin_over_classical": round(F - 2 / 3, 4),
        "formulas": "f = max over maximally entangled |e> of <e|ρ|e> (exact: largest eigenvalue of "
                    "Re ρ in the magic basis); F = (2f + 1)/3 (Horodecki); classical limit 2/3",
        "note": ("entangled states with f <= 1/2 exist (they cannot beat 2/3 in this protocol "
                 "without local filtering)" if qt.concurrence(rho) > 1e-9 and f <= 0.5 else ""),
    })


def _qkd_numbers(e_bit: float, e_phase: float) -> dict:
    r = qt.entanglement_key_rate(e_bit, e_phase)
    return {"e_bit": round(e_bit, 5), "e_phase": round(e_phase, 5),
            "h_e_bit": round(qt.binary_entropy(e_bit), 5), "h_e_phase": round(qt.binary_entropy(e_phase), 5),
            "secret_key_fraction": round(r, 5), "key_possible": bool(r > 0)}


@tool
def qkd_key_rate(handle: str = "", qber_z: float | None = None, qber_x: float | None = None) -> dict:
    """Asymptotic secret-key fraction of (entanglement-based) BB84 / BBM92:
    r = 1 - h(e_bit) - h(e_phase), h = binary entropy; ideal single pairs, infinite key,
    one-way error correction at the Shannon limit. Either give a stored state `handle` (the
    QBERs are computed from it: key from the H/V basis, phase errors from the D/A basis; the
    R/L basis is also reported), or give qber_z (and optionally qber_x, default = qber_z) directly.
    Equal QBERs give r = 1 - 2h(Q), positive only for Q < 11.0%."""
    if handle:
        rho = _get2(handle, "qkd_key_rate")
        _check_named_bell_state(handle)
        q = {b: qt.basis_qber(rho, b) for b in ("Z", "X", "Y")}
        best_pair = min((("Z", "X"), ("Z", "Y"), ("X", "Z"), ("X", "Y"), ("Y", "Z"), ("Y", "X")),
                        key=lambda kb: -qt.entanglement_key_rate(q[kb[0]][0], q[kb[1]][0]))
        out = {"source": f"state {handle} ({DESC.get(handle, '')})",
               "qber_by_basis": {b: {"qber": round(v[0], 5), "outcomes": v[1]} for b, v in q.items()},
               "standard_BB84_HV_key_DA_test": _qkd_numbers(q["Z"][0], q["X"][0]),
               "best_basis_pair": {"key_basis": best_pair[0], "test_basis": best_pair[1],
                                   **_qkd_numbers(q[best_pair[0]][0], q[best_pair[1]][0])},
               }
        _, _, phi, alpha = _phase_compensation(rho, "B")
        if abs(alpha) > 0.5:
            U = qt.on_qubit(np.diag([1, np.exp(1j * np.deg2rad(alpha))]), "B")
            rc = U @ rho @ U.conj().T
            qz, qx = qt.basis_qber(rc, "Z")[0], qt.basis_qber(rc, "X")[0]
            out["after_phase_compensation"] = {
                "what": f"phase shifter α = {alpha:.2f}° on photon B first (see compensate_phase), then "
                        "key from H/V and test in D/A",
                **_qkd_numbers(qz, qx)}
    else:
        if qber_z is None:
            raise ValueError("Give either a state handle or qber_z (a fraction, e.g. 0.03 for 3%).")
        qx = qber_z if qber_x is None else qber_x
        for v in (qber_z, qx):
            if not 0 <= v <= 0.5:
                raise ValueError("QBER must be a fraction in [0, 0.5] (e.g. 0.03 for 3%).")
        out = {"source": "QBER given by the user", **_qkd_numbers(qber_z, qx)}
    out["threshold"] = ("key is possible iff h(e_bit) + h(e_phase) < 1. Only when the two errors are EQUAL "
                        "does this become QBER < 11.0%; with unequal errors one of them may exceed 11%")
    out["assumptions"] = ("asymptotic (infinite key), ideal entangled-pair or single-photon source, no "
                          "detector side channels, error correction at the Shannon limit")
    return _clean(out)


@tool
def bit_flip_code(p: float) -> dict:
    """Three-qubit bit-flip repetition code with majority vote against independent bit flips of
    probability p per qubit: logical error p_L = 3p² - 2p³ versus p unencoded. The code helps
    iff p < 1/2. (It does not protect against phase flips.)"""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1].")
    pl = qt.bitflip_code_logical_error(p)
    return _clean({
        "p_physical": p, "p_logical": round(pl, 6),
        "improvement_factor": (round(p / pl, 4) if pl > 0 else "infinite (no error)"),
        "code_helps": bool(pl < p - 1e-12),
        "break_even": "p = 1/2 (p_L = p there; for p > 1/2 the code makes things worse)",
        "formula": "p_L = 3p²(1 - p) + p³ = 3p² - 2p³ (two or three flips defeat the majority vote)",
        "limitation": "corrects any single bit flip; does nothing against phase (σz) errors",
    })


# ------------------------------------------------------------------ certification: witness, steering, Bell
_CERT_TESTS = [
    # (property, key, bound, direction): the property holds if value is beyond the bound
    ("entangled (concurrence > 0)", "concurrence", 0.0, +1),
    ("entangled (fidelity witness < 0)", "witness_fidelity", 0.0, -1),
    ("entangled (PPT witness < 0)", "witness_ppt", 0.0, -1),
    ("EPR steerable (3-setting steering inequality F3 > 1)", "steering_F3", 1.0, +1),
    ("Bell nonlocal (CHSH > 2)", "chsh_max", 2.0, +1),
]


def _significance_verdict(n: float) -> str:
    """Words for a distance n (in σ) beyond a bound."""
    return ("shown (≥ 5σ)" if n >= 5 else "marginal (3-5σ): evidence, not yet conclusive" if n >= 3
            else "not significant (< 3σ)")


_NOT_SHOWN = {
    "concurrence": "not entangled",
    "witness_fidelity": "not detected by this witness",
    "witness_ppt": "not entangled (PPT)",
    "steering_F3": "not shown by this test (F3 <= 1 does not prove that the state is unsteerable)",
    "chsh_max": "no CHSH violation (this does not prove a local model)",
}


@tool
def certify_correlations(handle: str) -> dict:
    """Certify entanglement, EPR steering and Bell nonlocality of a stored state, with the
    statistical significance (number of standard deviations beyond each bound) when the state
    was reconstructed or simulated from counts. Includes an entanglement witness (fidelity
    witness W = I/2 - |e><e| and the optimal PPT witness), the 3-setting linear steering
    inequality F3 = (t1 + t2 + t3)/√3 > 1 (plus the exact steering criterion when both photons
    are unpolarized, e.g. Werner states) and CHSH max > 2. Error bars come from the same
    Poisson resamplings as the tomography report."""
    rho = _get2(handle, "certify_correlations")
    _check_named_bell_state(handle)
    W_f, e = qt.fidelity_witness(rho)
    W_p = qt.ppt_witness(rho)

    def stats(r):
        return {"concurrence": qt.concurrence(r),
                "witness_fidelity": float(np.real(np.trace(W_f @ r))),
                "witness_ppt": float(np.real(np.trace(W_p @ r))),
                "steering_F3": qt.cjwr_steering(r, 3), "steering_F2": qt.cjwr_steering(r, 2),
                "chsh_max": qt.chsh_max(r)}

    val = stats(rho)
    states = _bootstrap_states(handle)
    sig = None
    if states:
        boot = [stats(r) for r in states]
        sig = {k: float(np.std([b[k] for b in boot], ddof=1)) for k in val}
    fit = _FIT.get(handle, {})
    scale = float(np.sqrt(fit["chi2_per_dof"])) if fit.get("chi2_per_dof", 0) > 1.5 else 1.0

    results = []
    for prop, key, bound, direction in _CERT_TESTS:
        v = val[key]
        holds = bool(direction * (v - bound) > 1e-9)
        row = {"property": prop, "value": round(v, 4), "bound": bound, "holds": holds}
        if sig is not None:
            s = sig[key]
            row["sigma"] = _two_sig(s)
            if s > 0:
                n = direction * (v - bound) / s
                row["n_sigma"] = round(n, 1)
                if scale > 1:
                    row["sigma_realistic"] = _two_sig(s * scale)
                    row["n_sigma_realistic"] = round(n / scale, 1)
                nn = n / scale
                row["verdict"] = _significance_verdict(nn) if nn > -3 else _NOT_SHOWN[key]
                if key == "concurrence" and v < 3 * s:
                    row["note"] = "concurrence is bounded below by 0, so its σ is not Gaussian near 0"
        else:
            row["verdict"] = "holds (exact state, no data)" if holds else _NOT_SHOWN[key] + " (exact state)"
        results.append(row)

    exact_steer = None
    if qt.is_t_state(rho):
        S = qt.t_state_steering_value(rho)
        exact_steer = {"property": "EPR steerable (exact criterion for unpolarized photons: S > 1)",
                       "value": round(S, 4), "bound": 1.0, "holds": bool(S > 1 + 1e-3),
                       "verdict": ("holds: steerable with projective measurements" if S > 1 + 1e-3 else
                                   "at the boundary (S = 1)" if S > 1 - 1e-3 else
                                   "does not hold: not steerable with projective measurements")}
        results.insert(4, exact_steer)
    bell = results[-1]["holds"]
    steer_f3 = results[3]["holds"]
    steer_exact = bool(exact_steer and exact_steer["holds"])
    ent = any(r["holds"] for r in results[:3])
    hierarchy = ("Bell nonlocal" if bell else
                 "steerable but no CHSH violation" if (steer_f3 or steer_exact) else
                 ("entangled, not steerable with projective measurements" if exact_steer is not None
                  else "entangled, steering not shown with 3 settings") if ent else "not entangled")
    caveats = [
        "CHSH max and F3 use the measurement settings that are optimal for this state; a direct "
        "measurement with fixed settings has an expected value at most this large",
        "F3 > 1 proves steering but is not necessary (Werner states: F3 > 1 only for p > 1/√3, "
        "steerable with projective measurements for p > 1/2)" + (
            "; the exact criterion S > 1 applies here because both photons are unpolarized"
            if exact_steer is not None else
            "; the exact criterion needs unpolarized photons, so it is not available here"),
        "the witnesses are chosen from the same data they are tested on (slight optimism)",
    ]
    total = float(np.sum(fit["mu"])) if fit else 0.0
    if fit and total < 5000:
        caveats.append(f"few counts ({total:.0f} coincidences): the σ and the nσ values are only rough")
    out = {
        "handle": handle, "state": DESC.get(handle, ""),
        "tests": results,
        "strongest_property": hierarchy,
        "fidelity_witness_target": _ket_text(e) + f" (fidelity {0.5 - val['witness_fidelity']:.4f})",
        "steering_F2": round(val["steering_F2"], 4),
        "error_method": (f"{N_RESAMPLES} Poisson resamplings of the fitted counts (statistical only); "
                         + (f"'n_sigma_realistic' divides by √(χ²/dof) = {scale:.2f} because the fit is "
                            "poor" if scale > 1 else "fit is consistent with counting noise"))
                        if sig is not None else "exact state (not from data): no error bars",
        "caveats": caveats,
        "hierarchy": "Bell nonlocal ⊂ EPR steerable ⊂ entangled",
    }
    return _clean(out)


# ================================================================== plot engine
def _entropy(r):
    return qt.von_neumann_entropy(r) + 0.0


def _key_fraction(r):
    return qt.entanglement_key_rate(qt.basis_qber(r, "Z")[0], qt.basis_qber(r, "X")[0])


def _sweep_quantities(reference: tuple[str, np.ndarray] | None) -> dict:
    """name -> (function of ρ, panel title, reference line (level, text) or None)."""
    q = {
        "concurrence": (qt.concurrence, "Concurrence C", (0.0, "C > 0: entangled")),
        "negativity": (qt.negativity, "Negativity N (max 1/2)", (0.0, "N > 0: entangled")),
        "log_negativity": (qt.log_negativity, "Log-negativity E_N (bits)", None),
        "entanglement_of_formation": (qt.entanglement_of_formation, "Entanglement of formation (ebits)", None),
        "purity": (qt.purity, "Purity Tr ρ²", None),
        "entropy": (_entropy, "von Neumann entropy (bits)", None),
        "chsh_max": (qt.chsh_max, "CHSH max", (2.0, "local bound 2")),
        "mutual_information": (qt.mutual_information, "Mutual information (bits)", None),
        "discord": (qt.quantum_discord, "Quantum discord (bits)", None),
        "coherence_l1": (qt.coherence_l1, "l1 coherence (H/V basis)", None),
        "fully_entangled_fraction": (qt.fully_entangled_fraction_2q, "Fully entangled fraction f",
                                     (0.5, "f > 1/2: beats classical teleportation")),
        "teleportation_fidelity": (lambda r: (2 * qt.fully_entangled_fraction_2q(r) + 1) / 3,
                                   "Teleportation fidelity", (2 / 3, "classical limit 2/3")),
        "steering_f3": (lambda r: qt.cjwr_steering(r, 3), "Steering F3 (3 settings)",
                        (1.0, "F3 > 1: steering shown with 3 settings (sufficient, not necessary)")),
        "key_fraction": (_key_fraction, "BB84 key fraction (H/V key, D/A test)", (0.0, "> 0: key possible")),
    }
    if reference is not None:
        name, ref = reference
        q["fidelity"] = (lambda r: qt.fidelity(r, ref), f"Fidelity with {name}", None)
    return q


_QUANTITY_ALIASES = {"chsh": "chsh_max", "chsh_value": "chsh_max", "bell": "chsh_max", "c": "concurrence",
                     "eof": "entanglement_of_formation", "f3": "steering_f3", "steering": "steering_f3",
                     "teleportation": "teleportation_fidelity", "fef": "fully_entangled_fraction",
                     "key": "key_fraction", "key_rate": "key_fraction", "secret_key_fraction": "key_fraction",
                     "mi": "mutual_information", "l1": "coherence_l1", "coherence": "coherence_l1",
                     "l1_coherence": "coherence_l1", "e_n": "log_negativity", "von_neumann_entropy": "entropy",
                     "n": "negativity", "fidelity_with_reference": "fidelity"}

SWEEPS = {
    # name: (description, x label, default range, needs stored states)
    "werner_p": ("Werner state p|B><B| + (1-p)I/4 (B = `bell`)", "Werner parameter p", (0.0, 1.0), False),
    "pure_angle": ("pure state cos θ|HH> + sin θ|VV>", "θ (radians) in cos θ|HH⟩ + sin θ|VV⟩", (0.0, np.pi / 2), False),
    "visibility": ("v ρ + (1 - v) I/4 (white noise) for each stored state", "visibility v", (0.0, 1.0), True),
    "noise": ("a local noise channel (channel, qubits) with parameter p applied to each stored state",
              "noise parameter", (0.0, 1.0), True),
    "waveplate_angle": ("a wave plate (plate) at angle θ on one photon (qubit) of each stored state",
                        "wave-plate angle θ (degrees from H)", (0.0, 180.0), True),
    "phase_shift": ("a phase shifter diag(1, e^(iφ)) on one photon (qubit) of each stored state",
                    "phase φ (degrees)", (0.0, 360.0), True),
}
_MAX_SWEEP_POINTS = 401


def _sweep_state_fn(sweep, rho0, bell, channel, qubits, plate, qubit):
    if sweep == "werner_p":
        return lambda p: qt.werner_state(p, bell)
    if sweep == "pure_angle":
        return lambda t: qt.dm(np.array([np.cos(t), 0, 0, np.sin(t)], dtype=complex))
    if sweep == "visibility":
        return lambda v: v * rho0 + (1 - v) * np.eye(4) / 4
    if sweep == "noise":
        return lambda p: qt.apply_kraus(rho0, _channel_kraus(channel, float(np.clip(p, 0, 1)), qubits))
    if sweep == "waveplate_angle":
        delta = {"HWP": np.pi, "QWP": np.pi / 2}[plate]

        def f(theta):
            U = qt.on_qubit(qt.waveplate(theta, delta), qubit)
            return U @ rho0 @ U.conj().T
        return f
    if sweep == "phase_shift":
        def g(phi):
            U = qt.on_qubit(np.diag([1, np.exp(1j * np.deg2rad(phi))]), qubit)
            return U @ rho0 @ U.conj().T
        return g
    raise ValueError(f"sweep must be one of {list(SWEEPS)}")


def _trend(x, y) -> str:
    span = float(np.ptp(y))
    if span < 1e-6:
        return f"constant ({y[0]:.4f})"
    d = np.diff(y)
    tol = 1e-9 + 1e-7 * span
    if np.all(d >= -tol):
        return "increases (monotonic)"
    if np.all(d <= tol):
        return "decreases (monotonic)"
    i_max, i_min = int(np.argmax(y)), int(np.argmin(y))
    return (f"not monotonic: maximum {y[i_max]:.4f} at {x[i_max]:.4g}, minimum {y[i_min]:.4f} at "
            f"{x[i_min]:.4g}")


def _extremum(x, y, kind: str) -> dict:
    """Value and position(s) of the min or max on the grid; separate equal extrema are all listed
    and a flat stretch is given as a range ('0 to 51.3')."""
    v = float(y.max() if kind == "max" else y.min())
    # an exact zero (a clipped measure) is matched only by zeros: (1-p)^4 = 1e-8 at p = 0.99 is NOT the minimum 0;
    # otherwise 1e-7 absorbs round-off (a concurrence of 1 comes out as 1 - 3e-8)
    tol = 1e-12 if abs(v) < 1e-12 else 1e-7 * max(1.0, abs(v))
    hit = np.abs(y - v) <= tol
    at, i = [], 0
    while i < len(y):
        if hit[i]:
            j = i
            while j + 1 < len(y) and hit[j + 1]:
                j += 1
            at.append(round(float(x[i]), 4) if j - i < 2 else f"{x[i]:.4g} to {x[j]:.4g}")
            i = j + 1
        else:
            i += 1
    return {"value": round(v, 4), "at": at[0] if len(at) == 1 else at}


def _separable(rho: np.ndarray) -> bool:
    """PPT with a margin: the smallest eigenvalue of rho^T_B is not negative. An entangled state whose
    concurrence has only decayed below 1e-9 (asymptotic decay) is still NPT, so it is not reported as a
    'crossing' where entanglement vanishes."""
    return bool(np.min(np.linalg.eigvalsh(qt.partial_transpose(rho, 1, [2, 2]))) >= -1e-15)


def _in_band(v, level: float) -> np.ndarray:
    v = np.asarray(v, float)
    return (v >= level - 1e-7) & (v <= level + 1e-9)


def _only_approaches(side: np.ndarray, level: float) -> bool:
    """True when, beyond a 'crossing', the curve stays within the numerical band around `level` up to the
    end of the range (side = the grid values from the crossing to that end) and is strictly above the
    level somewhere there, or reaches the level only at the end point itself: it only APPROACHES the bound
    (e.g. CHSH = 2√(1 + c²) with c -> 0), it does not cross it. A floor exactly at the level over several
    grid points (key fraction max(0, ...)) is a real crossing."""
    side = np.asarray(side, float)
    return bool(side.size and np.all(_in_band(side, level)) and (np.any(side > level) or side.size == 1))


def _crossings(x, y, level, exact, state_at=None) -> list[dict]:
    """Where the quantity crosses `level` (grid bracket + bisection on the exact function). state_at:
    for an entanglement measure with level 0, the state as a function of x, to drop 'crossings' that are
    only an asymptotic decay below the numerical threshold."""
    above = y > level + 1e-9
    out = []
    for i in np.nonzero(above[1:] != above[:-1])[0]:
        lo, hi = float(x[i]), float(x[i + 1])
        lo_above = bool(above[i])
        for _ in range(50):
            mid = (lo + hi) / 2
            if (exact(mid) > level + 1e-9) == lo_above:
                lo = mid
            else:
                hi = mid
        at = (lo + hi) / 2
        if min(at - x[0], x[-1] - at) < 1e-6 * (x[-1] - x[0]):
            continue                      # touches the bound only at an end point of the range
        if state_at is None and _only_approaches(y[i + 1:] if lo_above else y[:i + 1], level):
            continue                      # an asymptotic approach to the bound, not a crossing
        if state_at is not None:          # the separable side must really be separable
            # the grid point on the separable side (it may be the only separable point, as for a flip
            # channel at p = 1/2); at an end of the range, where an asymptotic decay also reaches 0, halfway
            grid = float(x[i + 1]) if lo_above else float(x[i])
            end = (i + 1 == len(x) - 1) if lo_above else (i == 0)
            probe = (at + grid) / 2 if end else grid
            if not _separable(state_at(probe)):
                continue
        out.append({"at": round(at, 4), "_exact": at, "_i": int(i),   # _exact, _i: removed below
                    "direction": "falls to the bound" if lo_above else "rises above the bound"})
    # a dip that only TOUCHES the bound (falls, then rises again, never clearly below: e.g. CHSH =
    # 2√(1 + (1-2p)^4) at p = 1/2) is one point, at the exact minimum
    merged, k = [], 0
    while k < len(out):
        c = out[k]
        if (k + 1 < len(out) and c["direction"].startswith("falls") and out[k + 1]["direction"].startswith("rises")
                and np.all(_in_band(y[c["_i"] + 1:out[k + 1]["_i"] + 1], level))):
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(exact, bounds=(c["_exact"], out[k + 1]["_exact"]), method="bounded",
                                  options={"xatol": 1e-10})
            merged.append({"at": round(float(res.x), 4), "_exact": float(res.x),
                           "direction": "touches the bound (only at this point)"})
            k += 2
            continue
        merged.append(c)
        k += 1
    for c in merged:
        c.pop("_i", None)
    return merged


class _SweepArgs(BaseModel):
    sweep: str
    handles: Annotated[list[str], BeforeValidator(lambda v: [v] if isinstance(v, str) else (v or []))] = []
    quantities: Annotated[list[str], BeforeValidator(
        lambda v: [x.strip() for x in v.split(",")] if isinstance(v, str) else (v or []))] = []
    start: float | None = None
    stop: float | None = None
    bell: str = "psi-"
    channel: str = ""
    qubits: str = "both"
    plate: str = "HWP"
    qubit: str = "B"
    reference: str = ""
    n_points: int = 101


@tool(args_schema=_SweepArgs)
def plot_sweep(sweep: str, handles: list[str] = [], quantities: list[str] = [], start: float | None = None,
               stop: float | None = None, bell: str = "psi-", channel: str = "", qubits: str = "both",
               plate: str = "HWP", qubit: str = "B", reference: str = "", n_points: int = 101) -> dict:
    """Plot any measure against a parameter (one panel per measure, up to 3 states compared).
    sweep (x axis):
      'werner_p'        Werner state p|B><B| + (1-p)I/4, B = bell ('psi-' default, 'phi+', ...)
      'pure_angle'      cos θ|HH> + sin θ|VV>, θ in radians (default 0 to π/2)
      'visibility'      v ρ + (1-v) I/4 for the stored states in `handles`
      'noise'           local `channel` ('phase_damping', 'amplitude_damping', 'depolarizing',
                        'bit_flip', 'phase_flip', 'bit_phase_flip') with parameter p on `qubits`, or a
                        global channel (apply_global_noise names, e.g. 'collective_dephasing')
      'waveplate_angle' `plate` ('HWP' or 'QWP') at angle θ (degrees) on photon `qubit` ('A'/'B')
      'phase_shift'     diag(1, e^(iφ)) with φ in degrees on photon `qubit`
    handles: states for the last four sweeps (1 to 3): stored handles or Bell-state names such as
      'phi+'. start/stop: range (defaults exist).
    quantities (<= 6): concurrence, negativity, log_negativity, entanglement_of_formation, purity,
      entropy, chsh_max, mutual_information, discord, coherence_l1, fidelity (with `reference`:
      a handle or 'phi+', 'phi-', 'psi+', 'psi-'), fully_entangled_fraction,
      teleportation_fidelity, steering_f3, key_fraction. Default: concurrence, negativity,
      purity, chsh_max.
    Returns the figure and, per state and measure: values at both ends, min/max, the trend and
    the exact parameter values where it crosses its bound (e.g. CHSH = 2), plus 11 samples."""
    if sweep not in SWEEPS:
        raise ValueError(f"sweep must be one of {list(SWEEPS)} (time evolution: use evolve_state).")
    desc, xlabel, (d0, d1), needs_states = SWEEPS[sweep]
    quantities = [_QUANTITY_ALIASES.get(q.strip().lower().replace(" ", "_").replace("-", "_"),
                                        q.strip().lower().replace(" ", "_").replace("-", "_")) for q in quantities]
    quantities = list(dict.fromkeys(quantities)) or ["concurrence", "negativity", "purity", "chsh_max"]
    if len(quantities) > 6:
        raise ValueError("At most 6 quantities per plot (one panel each); split the request.")
    ref = _reference_state(reference) if reference else None
    table = _sweep_quantities(ref)
    unknown = [q for q in quantities if q not in table]
    if unknown:
        hint = " ('fidelity' needs reference=, e.g. reference='phi+')" if "fidelity" in unknown else ""
        raise ValueError(f"Unknown quantities {unknown}{hint}. Choose from {sorted(table)}.")
    x0 = d0 if start is None else float(start)
    x1 = d1 if stop is None else float(stop)
    if x1 <= x0:
        raise ValueError("stop must be larger than start.")
    if sweep in ("werner_p", "visibility", "noise") and not (0 <= x0 and x1 <= 1):
        raise ValueError(f"For '{sweep}' the parameter must stay within [0, 1].")
    bell = bell.strip().lower().replace(" ", "")
    if bell not in ("phi+", "phi-", "psi+", "psi-"):
        raise ValueError("bell must be 'phi+', 'phi-', 'psi+' or 'psi-'.")
    plate, qubit = plate.strip().upper(), qubit.strip().upper()
    if sweep == "waveplate_angle" and plate not in ("HWP", "QWP"):
        raise ValueError("plate must be 'HWP' or 'QWP'.")
    if sweep in ("waveplate_angle", "phase_shift") and qubit not in ("A", "B"):
        raise ValueError("qubit must be 'A' or 'B'.")
    if sweep == "noise":
        _channel_kraus(channel, 0.5, qubits)                  # validates channel and qubits (local or global)
        channel = _channel_name(channel)
    n_points = int(np.clip(n_points, 11, _MAX_SWEEP_POINTS))
    if "discord" in quantities:
        n_points = min(n_points, 41)                          # discord needs an optimization per point

    if needs_states:
        if not handles:
            raise ValueError(f"The '{sweep}' sweep acts on stored states: give handles=['rho_1', ...].")
        if len(handles) > 3:
            raise ValueError("At most 3 states per plot.")
        states = []
        for h in handles:
            label, rho_h = _reference_state(h)            # a stored handle or 'phi+', 'psi-', ...
            if rho_h.shape[0] != 4:
                raise ValueError(f"{h} is a ONE-photon state: plot_sweep compares two-photon states (for one "
                                 "photon use evolve_state or plot_bloch_and_correlations).")
            if label == h and h in STORE:
                _check_named_bell_state(h)
            states.append((label, DESC.get(h, "") or label, rho_h))
    else:
        label = f"Werner on {bell}" if sweep == "werner_p" else "cos θ|HH⟩ + sin θ|VV⟩"
        states = [(label, label, None)]

    xs = np.linspace(x0, x1, n_points)
    sample_x = np.linspace(x0, x1, 11)
    results, panels_y = {}, {q: [] for q in quantities}
    marks = {q: [] for q in quantities}
    for i, (key, label, rho0) in enumerate(states):
        f = _sweep_state_fn(sweep, rho0, bell, channel, qubits, plate, qubit)
        rhos = [f(x) for x in xs]
        per_q = {}
        for q in quantities:
            fn, _, refline = table[q]
            y = np.array([fn(r) for r in rhos], dtype=float)
            panels_y[q].append(y)
            info = {"at_start": round(float(y[0]), 4), "at_end": round(float(y[-1]), 4),
                    "min": _extremum(xs, y, "min"), "max": _extremum(xs, y, "max"),
                    "trend": _trend(xs, y)}
            if refline is not None:
                cr = _crossings(xs, y, refline[0], lambda x, fn=fn, f=f: fn(f(x)),
                                state_at=f if q in ("concurrence", "negativity") and refline[0] == 0 else None)
                for c in cr:                  # every plotted measure exactly at the crossing (for tables)
                    x_c = c.pop("_exact")         # the unrounded crossing: F3 = 1 there, not 1.0001
                    c["values_there"] = {q2: round(float(table[q2][0](f(x_c))), 4) for q2 in quantities}
                info["bound"] = refline[1]
                info["crossings"] = cr or "none in this range"
                marks[q] += [(i, (c["at"], refline[0])) for c in cr]
            per_q[q] = info
        samples = {f"{x:.4g}": {q: round(float(table[q][0](f(x))), 4) for q in quantities} for x in sample_x}
        results[key] = {"state": label, "measures": per_q, "samples": samples}

    def short(key, label):
        if label.startswith("reconstructed from"):
            label = label.split()[2]
        label = label.split("; full report")[0]
        return key if label == key else f"{key}: {label[:34]}" + ("…" if len(label) > 34 else "")
    labels = [short(key, label) if needs_states else label for key, label, _ in states]
    same = [(labels[j], labels[i]) for i in range(len(states)) for j in range(i)
            if all(np.max(np.abs(panels_y[q][i] - panels_y[q][j])) < 1e-6 for q in quantities)]
    panels = [{"title": table[q][1], "ys": panels_y[q], "reference": table[q][2], "marks": marks[q]}
              for q in quantities]
    on = "the pair (global)" if channel in GLOBAL_CHANNELS else QUBITS.get(qubits, qubits)
    extra = {"werner_p": f" (Werner on {bell})", "noise": f" ({channel} on {on})" if sweep == "noise" else "",
             "waveplate_angle": f" ({plate} on photon {qubit})", "phase_shift": f" (on photon {qubit})"}.get(sweep, "")
    title = {"werner_p": f"Werner state p|{BELL_GREEK.get(bell, bell)}⟩⟨{BELL_GREEK.get(bell, bell)}| + (1 − p) I/4",
             "pure_angle": "cos θ|HH⟩ + sin θ|VV⟩",
             "visibility": "white noise: v ρ + (1 − v) I/4",
             "noise": f"{channel.replace('_', ' ')} on {on}",
             "waveplate_angle": f"{plate} at angle θ on photon {qubit}",
             "phase_shift": f"phase shifter diag(1, e^(iφ)) on photon {qubit}"}[sweep]
    path = pl.figure_path(f"sweep_{sweep}", [sweep, handles, quantities, x0, x1, bell, channel, qubits,
                                              plate, qubit, reference, n_points])
    if sweep == "noise":
        xlabel = (CHANNELS[channel][1] if channel in CHANNELS else
                  "noise parameter p (0 = no noise)")      # e.g. 'decay probability γ (0 = no noise)'
    pl.sweep_figure(xs, panels, labels, xlabel, title, path,
                    dashed=[i for i in range(len(states)) if any(b == labels[i] for _, b in same)])
    return _clean({"figure": path, "sweep": title, "x_axis": xlabel,
                   "range": [round(x0, 4), round(x1, 4)], "points": n_points,
                   "grid_step": round((x1 - x0) / (n_points - 1), 6),
                   "note": "min/max positions are grid points (accuracy ± grid_step/2); crossings are exact",
                   "results": results,
                   **({"steering_note": "F3 > 1 is a sufficient test with 3 measurement settings, not the "
                       "steering threshold" + (": Werner states are steerable with all projective measurements "
                       "for p > 1/2 (Wiseman, Jones & Doherty 2007), but F3 > 1 only for p > 1/√3 ≈ 0.5774"
                       if sweep == "werner_p" else "; F3 <= 1 does not prove that a state is unsteerable")}
                      if "steering_f3" in quantities else {}),
                   **({"identical_curves": [f"{b} gives exactly the same curves as {a} (drawn dashed on top)"
                                            for a, b in same]} if same else {}),
                   "how_to_read": "one panel per measure; one coloured line per state; dashed grey line = the "
                                  "bound of that measure; dots = exact crossing points"})


class _HandlesArgs(BaseModel):
    handles: Annotated[list[str], BeforeValidator(lambda v: [v] if isinstance(v, str) else v)]


@tool(args_schema=_HandlesArgs)
def plot_bloch_and_correlations(handles: list[str]) -> dict:
    """Plot the Bloch vector of each photon's reduced state (photon A and photon B spheres, up to 3
    states) and the 3x3 correlation matrix T_ij = <σ_i ⊗ σ_j> of each state (i, j = X (D/A),
    Y (R/L), Z (H/V)). Give ALL states in ONE call: they are drawn in one figure. Returns the
    vectors, T, its singular values and what follows from them (CHSH max = 2√(t1² + t2²),
    steering F3 = (t1 + t2 + t3)/√3). ONE-photon states (all of them) are drawn on one Bloch (Poincaré)
    sphere with their Stokes vectors instead."""
    if not 1 <= len(handles) <= 3:
        raise ValueError("Give 1 to 3 state handles.")
    sizes = {_get(h).shape[0] for h in handles}
    if sizes == {2}:
        return _bloch_one_photon(handles)
    if len(sizes) > 1:
        raise ValueError("Give only one-photon states or only two-photon states in one call (for a pair, the "
                         "spheres show each photon's reduced state).")
    items, out = [], {}
    P = (qt.X, qt.Y, qt.Z)
    for h in handles:
        rho = _get(h)
        _check_named_bell_state(h)
        a = np.real([np.trace(rho @ np.kron(s_, qt.I2)) for s_ in P])
        b = np.real([np.trace(rho @ np.kron(qt.I2, s_)) for s_ in P])
        T = qt.correlation_matrix(rho)
        sv = np.linalg.svd(T, compute_uv=False)
        items.append({"label": h, "a": a, "b": b, "T": T})
        out[h] = {"state": DESC.get(h, ""),
                  "bloch_A": [round(float(v), 4) for v in a], "length_A": round(float(np.linalg.norm(a)), 4),
                  "bloch_B": [round(float(v), 4) for v in b], "length_B": round(float(np.linalg.norm(b)), 4),
                  "T": [[round(float(v), 4) for v in row] for row in T],
                  "singular_values_of_T": [round(float(v), 4) for v in sv],
                  "chsh_max_from_T": round(float(2 * np.sqrt(sv[0] ** 2 + sv[1] ** 2)), 4),
                  "steering_F3_from_T": round(float(sv.sum() / np.sqrt(3)), 4)}
    path = pl.figure_path("bloch_correlations", handles)
    pl.bloch_correlation_figure(items, path)
    return _clean({"figure": path, "states": out,
                   "how_to_read": ("top: Bloch vectors (X = D/A, Y = R/L, Z = H/V; north pole = H) of the "
                                   "reduced state of each photon; length 1 = fully polarized, length 0 = "
                                   "unpolarized (every maximally entangled state has length 0 on both "
                                   "sides). Bottom: T_ij = <σ_i ⊗ σ_j> (row = photon A, column = photon B); "
                                   "phi+ has T = diag(+1, -1, +1); a phase φ in (|HH> + e^(iφ)|VV>)/√2 "
                                   "rotates the X-Y block")})


def _bloch_one_photon(handles: list[str]) -> dict:
    items, out = [], {}
    for h in handles:
        s = qt.stokes_vector(_get(h))
        vec = np.array([s[1], s[2], s[0]])                 # (x, y, z) = (S2, S3, S1): north pole = H
        items.append({"label": h, "vec": vec})
        out[h] = {"state": DESC.get(h, ""), **_stokes_numbers(_get(h))}
    path = pl.figure_path("bloch_one_photon", handles)
    pl.bloch_single_figure(items, path)
    return _clean({"figure": path, "states": out,
                   "how_to_read": ("drawn as a Bloch sphere: x = S2 (D/A), y = S3 (R/L), z = S1 (H/V), so H is at "
                                   "the top of THIS drawing; on the usual optics Poincaré sphere (S3 vertical) the "
                                   "poles are R and L and all linear states (H, V, D, A) lie on the equator; "
                                   "the length of the arrow is the degree of polarization (1 on the surface = "
                                   "fully polarized, 0 at the centre = unpolarized)")})


_ANALYZER_NAMES = {0: "H", 45: "D", 90: "V", 135: "A"}


class _FringeArgs(BaseModel):
    handle: str
    alice_angles: Annotated[list[float], BeforeValidator(
        lambda v: [v] if isinstance(v, (int, float)) else v)] = [0.0, 45.0, 90.0, 135.0]


@tool(args_schema=_FringeArgs)
def plot_fringes(handle: str, alice_angles: list[float] = [0.0, 45.0, 90.0, 135.0]) -> dict:
    """Polarization-correlation fringes of a stored state: coincidence probability versus the angle
    of photon B's linear polarizer, for photon A's polarizer fixed at each of alice_angles
    (degrees from H; default H, D, V, A). Returns the exact visibility of every fringe and the
    angle of its maximum (lab check of the source: phi+ gives visibility 1 in every basis)."""
    rho = _get2(handle, "plot_fringes")
    _check_named_bell_state(handle)
    if not 1 <= len(alice_angles) <= 4:
        raise ValueError("Give 1 to 4 angles for photon A.")

    def proj(theta_deg):
        t = np.deg2rad(theta_deg)
        return qt.dm(np.array([np.cos(t), np.sin(t)], dtype=complex))

    def prob(ta, tb):
        return float(np.real(np.trace(rho @ np.kron(proj(ta), proj(tb)))))

    tb = np.linspace(0, 180, 361)
    curves, fringes = [], []
    for ta in alice_angles:
        p0, p45, p90 = prob(ta, 0), prob(ta, 45), prob(ta, 90)
        c0, c1 = (p0 + p90) / 2, (p0 - p90) / 2
        c2 = p45 - c0
        amp = float(np.hypot(c1, c2))
        vis = amp / c0 if c0 > 1e-12 else 0.0
        tmax = float((np.degrees(np.arctan2(c2, c1)) / 2) % 180)
        name = _ANALYZER_NAMES.get(int(round(ta)) % 180 if abs(ta - round(ta)) < 1e-9 else -1, "")
        label = f"photon A at {ta:g}°" + (f" ({name})" if name else "") + f": V = {vis:.3f}"
        curves.append((label, np.array([prob(ta, x) for x in tb])))
        fringes.append({"photon_A_angle_deg": ta, "basis": name or "custom", "visibility": round(vis, 4),
                        "maximum_at_deg": round(tmax, 2), "max_probability": round(c0 + amp, 4),
                        "min_probability": round(c0 - amp, 4)})
    coh = 2 * abs(rho[0, 3])                      # |HH><VV| coherence, as in a Bell state (= 1)
    phase = float(np.degrees(np.angle(rho[0, 3])))
    phase_info = {"hh_vv_coherence": round(float(coh), 4), "hh_vv_phase_magnitude_deg": round(abs(phase), 2)}
    if coh > 0.05 and abs(phase) > 15:
        u = np.kron(np.eye(2), np.diag([1, np.exp(1j * np.deg2rad(phase))]))
        rho_c = u @ rho @ u.conj().T                   # the same state with the HH-VV phase removed
        vis_c = []
        for ta in alice_angles:
            pr = lambda tb_: float(np.real(np.trace(rho_c @ np.kron(proj(ta), proj(tb_)))))  # noqa: E731
            c0_, c1_, c2_ = (pr(0) + pr(90)) / 2, (pr(0) - pr(90)) / 2, pr(45) - (pr(0) + pr(90)) / 2
            vis_c.append(round(float(np.hypot(c1_, c2_) / c0_), 4) if c0_ > 1e-12 else 0.0)
        phase_info["visibility_with_phase_removed"] = {      # only the fringes the phase affects (not H/V)
            f["basis"]: v for f, v in zip(fringes, vis_c) if f["photon_A_angle_deg"] % 90 != 0}
        phase_info["meaning"] = (f"the HH-VV phase (|φ| ≈ {abs(phase):.0f}°) moves the coherence into circular "
                                 "(R/L) correlations that linear polarizers cannot see; a low D/A visibility "
                                 "here is mainly due to this PHASE (it rises to the values above when the "
                                 "phase is removed, e.g. with compensate_phase), not to lost coherence")
    path = pl.figure_path(f"{handle}_fringes", [handle, list(alice_angles)])
    pl.fringe_figure(tb, curves, f"Polarization-correlation fringes — {handle}", path)
    return _clean({"figure": path, "state": DESC.get(handle, ""), "fringes": fringes, **phase_info,
                   "definition": "V = (P_max - P_min)/(P_max + P_min) of P(θ_A, θ_B) = Tr[ρ Π(θ_A) ⊗ Π(θ_B)], "
                                 "Π(θ) = |θ><θ|, |θ> = cos θ|H> + sin θ|V> (linear polarizers only)",
                   "how_to_read": ("one curve per angle of photon A; the H/V fringes test the populations "
                                   "(correlation in H/V), the D/A fringes test the coherence between HH and "
                                   "VV: a relative phase or dephasing lowers the D/A visibility; "
                                   "compare with hh_vv_coherence to tell which")})


@tool
def plot_tomography_fit(handle: str) -> dict:
    """Diagnostics of a state reconstructed (or simulated) from counts: measured vs fitted counts for
    every setting, residuals in standard deviations (which settings the model cannot explain), and
    the bootstrap distributions of concurrence and CHSH with their bounds."""
    fit = _FIT.get(handle)
    if fit is None or "n" not in fit:
        raise ValueError(f"{handle} was not reconstructed from counts. Reconstructed states: "
                         f"{[h for h in _FIT if 'n' in _FIT[h]]}")
    labels, mu, n = fit["labels"], np.asarray(fit["mu"], float), np.asarray(fit["n"], float)
    r = (n - mu) / np.sqrt(np.clip(mu, 1, None))
    dof = max(len(labels) - 16, 1)
    chi2 = float(np.sum(r ** 2))
    rho = _get2(handle, "plot_tomography_fit")
    boot = _bootstrap_states(handle)
    C = np.array([qt.concurrence(x) for x in boot])
    S = np.array([qt.chsh_max(x) for x in boot])
    hists = [{"name": "concurrence", "samples": C, "value": qt.concurrence(rho), "bound": 0.0,
              "bound_text": "C = 0"},
             {"name": "CHSH max", "samples": S, "value": qt.chsh_max(rho), "bound": 2.0, "bound_text": "CHSH = 2"}]
    path = pl.figure_path(f"{handle}_tomography_fit", [handle, fit.get("source", "")])
    pl.tomography_fit_figure(labels, n, mu, r, hists, f"Tomography fit — {handle} ({fit.get('source', '')})", path)
    order = np.argsort(-np.abs(r))
    from scipy.stats import chi2 as chi2_dist
    p_value = float(chi2_dist.sf(chi2, dof))
    return _clean({
        "figure": path, "data": fit.get("source", ""),
        "chi2_per_dof": round(chi2 / dof, 2), "dof": dof, "p_value": float(f"{p_value:.2g}"),
        "fit_verdict": ("consistent with counting noise" if p_value > 0.01 else
                        "NOT consistent with counting noise alone: systematic errors are present (more "
                        "settings beyond 2σ than counting noise would give)"),
        "settings_beyond_2_sigma": [f"{labels[i]} ({r[i]:+.1f}σ)" for i in order if abs(r[i]) > 2],
        "settings_beyond_3_sigma": [f"{labels[i]} ({r[i]:+.1f}σ)" for i in order if abs(r[i]) > 3] or "none",
        "expected_beyond_2_sigma_if_only_counting_noise": round(0.0455 * len(labels), 1),
        "bootstrap": {
            "concurrence": {"value": round(qt.concurrence(rho), 4), "sigma": _two_sig(float(np.std(C, ddof=1))),
                            "fraction_of_resamplings_with_C_0": round(float(np.mean(C <= 1e-9)), 3)},
            "chsh_max": {"value": round(qt.chsh_max(rho), 4), "sigma": _two_sig(float(np.std(S, ddof=1))),
                         "fraction_of_resamplings_at_or_below_2": round(float(np.mean(S <= 2)), 3)},
            "resamplings": len(boot)},
        "how_to_read": ("top: measured counts (bars) and the maximum-likelihood fit (dots) per setting; "
                        "middle: residuals (measured - fit)/√fit with ±2σ and ±3σ lines, red = beyond 3σ; "
                        "bottom: bootstrap histograms with the measured value and the bound"),
    })


def _read_table(path: str):
    """Header + numeric columns of a CSV (non-numeric cells -> NaN)."""
    import csv
    with open(_resolve_csv(path), newline="", encoding="utf-8-sig") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError("The file needs a header row and at least one data row.")
    header = [h or f"column {i + 1}" for i, h in enumerate(rows[0])]
    width = len(header)

    def num(x):
        try:
            return float(x)
        except ValueError:
            return np.nan
    cols = {h: np.array([num(r[i]) if i < len(r) else np.nan for r in rows[1:]]) for i, h in enumerate(header[:width])}
    return cols


def _pick(cols: dict, name: str) -> str:
    key = name.replace(" ", "").lower()
    for c in cols:
        if c.replace(" ", "").lower() == key:
            return c
    numeric = [c for c, v in cols.items() if np.isfinite(v).any()]
    raise ValueError(f"Column '{name}' not found. Numeric columns: {numeric}")


def _sig(v: float, err: float | None = None) -> float:
    """Round a fit value sensibly: to 2 significant digits of its error, else 5 significant digits."""
    if err is not None and np.isfinite(err) and err > 0:
        digits = max(0, -int(np.floor(np.log10(err))) + 1)
        return round(float(v), digits)
    return float(f"{v:.5g}") if np.isfinite(v) else float("nan")


class _DataArgs(BaseModel):
    csv_path: str
    x_column: str
    y_columns: Annotated[list[str], BeforeValidator(
        lambda v: [x.strip() for x in v.split(",")] if isinstance(v, str) else v)]
    fit: str = "none"
    y_error: str = "poisson"
    period: float | None = None
    x_label: str = ""
    y_label: str = ""
    title: str = ""
    log_y: bool = False


@tool(args_schema=_DataArgs)
def plot_data_file(csv_path: str, x_column: str, y_columns: list[str], fit: str = "none",
                   y_error: str = "poisson", period: float | None = None, x_label: str = "",
                   y_label: str = "", title: str = "", log_y: bool = False) -> dict:
    """Plot columns of the user's own CSV file (header row needed) and optionally fit them.
    x_column: column for the x axis; y_columns: 1 to 4 columns plotted against it.
    fit: 'none', 'linear', 'exponential' (decay A·e^(-x/τ) + C), 'sine' (period fitted unless
    `period` is given), 'fringe' (C + A cos(2π(x - x0)/P), P = `period`, default 180 for a
    polarizer angle in degrees, 90 for a half-wave-plate angle; returns the visibility A/C) or
    'gaussian'. y_error: 'poisson' (√counts, for counts), 'none', or a column name holding the
    error bars. Returns fit parameters with 1σ errors and the fit quality."""
    fit = fit.strip().lower()
    if fit not in pl.FIT_MODELS:
        raise ValueError(f"fit must be one of {list(pl.FIT_MODELS)}")
    if not 1 <= len(y_columns) <= 4:
        raise ValueError("Give 1 to 4 y columns.")
    cols = _read_table(csv_path)
    xc = _pick(cols, x_column)
    series, fits, notes = [], {}, []
    for name in y_columns:
        yc = _pick(cols, name)
        x, y = cols[xc], cols[yc]
        err = None
        if y_error.strip().lower() == "poisson":
            err_all = np.sqrt(np.clip(y, 1, None))
        elif y_error.strip().lower() in ("none", ""):
            err_all = None
        else:
            err_all = cols[_pick(cols, y_error)]
        ok = np.isfinite(x) & np.isfinite(y) & (np.isfinite(err_all) if err_all is not None else True)
        dropped = int((~ok).sum())
        if dropped:
            notes.append(f"{yc}: {dropped} row(s) without numbers were skipped")
        x, y = x[ok], y[ok]
        if err_all is not None:
            err = err_all[ok]
            if y_error.strip().lower() == "poisson" and np.any(y < 0):
                raise ValueError(f"Column '{yc}' has negative values: Poisson errors need counts. "
                                 "Use y_error='none' or an error column.")
            if np.any(err <= 0):
                raise ValueError(f"Error bars for '{yc}' must be > 0.")
        if len(x) < 2:
            raise ValueError(f"Column '{yc}' has fewer than 2 numeric points.")
        order = np.argsort(x)
        x, y = x[order], y[order]
        err = err[order] if err is not None else None
        entry = {"label": yc, "x": x, "y": y, "yerr": err, "fit_x": None, "fit_y": None}
        if fit != "none":
            try:
                res = pl.fit_data(x, y, fit, sigma=err, period=period)
            except RuntimeError:
                fits[yc] = {"fit_verdict": f"failed: the {fit} fit did not converge (probably the wrong model "
                                           "for these data); the data are plotted without a fit"}
                series.append(entry)
                continue
            xf = np.linspace(x.min(), x.max(), 400)
            entry["fit_x"], entry["fit_y"] = xf, res.pop("curve")(xf)
            res["parameters"] = {k: {"value": _sig(v["value"], v["error"]), "error": _two_sig(v["error"])
                                     if np.isfinite(v["error"]) else None} for k, v in res["parameters"].items()}
            for key in ("visibility", "period", "fwhm"):
                if key in res:
                    v = res[key]
                    res[key] = {"value": _sig(v["value"], v["error"]),
                                "error": _two_sig(v["error"]) if np.isfinite(v["error"]) and v["error"] > 0 else None}
            if "rms_residual" in res:
                res["rms_residual"] = _sig(res["rms_residual"])
            scale_params = {"fringe": ("C", "A"), "sine": ("C", "A", "P"), "exponential": ("A", "tau"),
                            "gaussian": ("A", "s"), "linear": ()}[fit]
            rel = [abs(v["error"] / v["value"]) for k, v in res["parameters"].items()
                   if k.split(" ")[0] in scale_params and v["error"] is not None
                   and v["value"] not in (0, None) and np.isfinite(v["value"])]
            if any(r_ > 1 for r_ in rel):
                res["fit_verdict"] = ("NOT meaningful: an amplitude, width or time constant has an error "
                                      "larger than its value (probably the wrong model for these data)")
            elif res.get("chi2_per_dof", 1) > 3:
                res["fit_verdict"] = (f"poor: χ²/dof = {res['chi2_per_dof']} - the model does not describe the "
                                      "data within the error bars")
            else:
                res["fit_verdict"] = "acceptable" if "chi2_per_dof" in res else "no error bars: quality not tested"
            res["error_bars"] = ("Poisson √counts" if y_error.strip().lower() == "poisson" else
                                 "none given: parameter errors estimated from the scatter" if err is None
                                 else f"column {y_error}")
            fits[yc] = res
        series.append(entry)
    caption = "; ".join(
        f"{k}: " + ", ".join(f"{p.split(' ')[0]} = {v['value']:g}" + (f" ± {v['error']:g}" if v['error'] else "")
                             for p, v in r["parameters"].items())
        + (f", V = {r['visibility']['value']:g}" + (f" ± {r['visibility']['error']:g}" if r['visibility']['error'] else "")
           if "visibility" in r else "")
        + (f", χ²/dof = {r['chi2_per_dof']}" if "chi2_per_dof" in r else "")
        for k, r in fits.items() if "parameters" in r)
    path = pl.figure_path(f"data_{os.path.splitext(os.path.basename(csv_path))[0]}",
                          [csv_path, x_column, y_columns, fit, y_error, period, log_y])
    default_y = (series[0]["label"] if len(series) == 1 else
                 "counts" if y_error.strip().lower() == "poisson" else "value")
    angle_axis = fit == "fringe" or bool(re.search(r"angle|deg|θ|theta", xc, re.I))
    pl.data_figure(series, x_label or xc, y_label or default_y, title or os.path.basename(csv_path), log_y,
                   path, caption, angle_axis)
    out = {"figure": path, "file": csv_path, "x_column": xc,
           "data": {e["label"]: {"points": int(len(e["x"])), "x_range": [_sig(e["x"].min()), _sig(e["x"].max())],
                                 "y_range": [_sig(e["y"].min()), _sig(e["y"].max())]} for e in series}}
    if fits:
        out["fits"] = fits
        out["fit_model"] = pl.FIT_MODELS[fit]
    if notes:
        out["notes"] = notes
    return _clean(out)


@tool
def list_capabilities(area: str = "") -> dict:
    """What this agent can and cannot do: its areas (states, measures, noise, evolution, wave plates,
    tomography, detectors, certification, protocols, plots), example questions and limits. Use it for
    'can you ...?' and 'what can you do?' questions. area: optional, e.g. 'tomography'."""
    import capabilities
    return capabilities.for_tool(area)


TOOLS = [create_state, create_pure_state, mix_states, apply_local_noise, compute_measures,
         plot_density_matrix, plot_measures_vs_noise, reconstruct_state_from_counts,
         get_tomography_report, compare_states, evolve_state, apply_waveplate, compensate_phase,
         simulate_tomography, teleportation_fidelity, qkd_key_rate, bit_flip_code,
         certify_correlations, plot_sweep, plot_bloch_and_correlations, plot_fringes, plot_tomography_fit,
         plot_data_file, list_capabilities,
         show_density_matrix, create_single_photon_state, reduced_state, combine_photons,
         reconstruct_single_photon, apply_global_noise, get_kraus_operators, apply_kraus_operators]


def _locked(func):
    """Run a tool while holding session.COMPUTE_LOCK (pyplot is not thread-safe; see session.py)."""
    import functools

    @functools.wraps(func)
    def run(*args, **kwargs):
        with session.COMPUTE_LOCK:
            return func(*args, **kwargs)
    return run


for _t in TOOLS:
    if getattr(_t, "func", None) is not None and not getattr(_t.func, "_qi_locked", False):
        _t.func = _locked(_t.func)
        _t.func._qi_locked = True
