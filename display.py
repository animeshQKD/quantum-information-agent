"""
display.py - turn the agent's raw output into something a person can read.

Used by both run.py (terminal) and app.py (browser):
  * to_unicode(text)      LaTeX leftovers  ->  plain symbols  (\\(|\\psi\\rangle\\) -> |ψ⟩)
  * summarize_steps(msgs) tool calls/results -> short human sentences
"""
from __future__ import annotations

import json
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import session

# ------------------------------------------------------------------ LaTeX -> Unicode
_SYMBOLS = {
    r"\psi": "ψ", r"\Psi": "Ψ", r"\phi": "φ", r"\varphi": "φ", r"\Phi": "Φ", r"\rho": "ρ",
    r"\sigma": "σ", r"\lambda": "λ", r"\gamma": "γ", r"\theta": "θ", r"\alpha": "α",
    r"\beta": "β", r"\delta": "δ", r"\Delta": "Δ", r"\eta": "η", r"\mu": "μ", r"\pi": "π",
    r"\chi": "χ", r"\omega": "ω", r"\Omega": "Ω", r"\epsilon": "ε", r"\tau": "τ",
    r"\rangle": "⟩", r"\langle": "⟨", r"\otimes": "⊗", r"\oplus": "⊕", r"\dagger": "†",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\simeq": "≃", r"\sim": "~", r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\infty": "∞",
    r"\to": "→", r"\rightarrow": "→", r"\Rightarrow": "⇒", r"\Leftrightarrow": "⇔",
    r"\iff": "⇔", r"\ldots": "…", r"\dots": "…", r"\sum": "Σ", r"\in": "∈",
    r"\mathbb{I}": "I", r"\mathbb{1}": "I", r"\uparrow": "↑", r"\downarrow": "↓",
    r"\{": "{", r"\}": "}", r"\%": "%",
}
_SUP = str.maketrans("0123456789+-=()nTBA", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀᴮᴬ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")


def _wrap(s: str) -> str:
    s = s.strip()
    return s if re.fullmatch(r"[\w.√²³ψφρ]+", s) else f"({s})"


def _sup(m):  # ^{...} or ^x
    body = (m.group(1) or m.group(2)).replace("_", "")
    out = body.translate(_SUP)
    return out if all(ch in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀᴮᴬ" for ch in out) else "^" + _wrap(body)


def _subscript(m):  # symbol followed by _{...} or _digit
    symbol, body = m.group(1), m.group(2) or m.group(3)
    out = body.translate(_SUB)
    return symbol + (out if all(ch in "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎" for ch in out) else "_" + body)


_KET_BODY = r"[A-Za-z0-9ψφΨΦ⁺⁻±+\-]{1,6}"


def fix_kets(t: str) -> str:
    """Draw ket/bra bars with '∣' (U+2223) so they cannot break Markdown table columns,
    and turn ASCII kets |HH> into |HH⟩."""
    t = re.sub(r"\|(" + _KET_BODY + r")(?:⟩|>)", r"∣\1⟩", t)
    t = re.sub(r"⟨(" + _KET_BODY + r")\|", r"⟨\1∣", t)
    # ASCII bra '<psi+|' (as in 0.3|psi+><psi+|): its '|' would split a table cell
    t = re.sub(r"<([A-Za-zψφΨΦ][A-Za-z0-9ψφΨΦ⁺⁻±+\-]{0,5})\|", r"⟨\1∣", t)
    return t


def fix_bold(t: str) -> str:
    """Remove a stray, unclosed '**' that the model sometimes leaves at the start (or end) of a line."""
    out = []
    for line in t.split("\n"):
        if line.count("**") % 2:
            if line.lstrip().startswith("**"):
                line = line.replace("**", "", 1)
            else:
                k = line.rfind("**")
                line = line[:k] + line[k + 2:]
        out.append(line)
    return "\n".join(out)


def fix_abs_bars(t: str) -> str:
    """In table rows, |x| (absolute value, e.g. '(|ρ| = 0.361)') would split the cell: draw it
    with '∣' instead. Cell borders have spaces around them, absolute-value bars do not."""
    out = []
    for line in t.split("\n"):
        if line.lstrip().startswith("|") and not re.fullmatch(r"[\s|:\-]+", line):
            line = re.sub(r"(?<=[(=\s])\|([^\s|]{1,12})\|(?=[\s)=,;.]|$)",
                          lambda m: "∣" + m.group(1) + "∣" if not set(m.group(1)) <= set("-:") else m.group(0),
                          line)
        out.append(line)
    return "\n".join(out)


def fix_table_separators(t: str) -> str:
    """A separator row with a different number of columns than its header ('|S|' in a header made a model
    write one '---' too many) stops the whole table from rendering: give it the header's column count."""
    lines = t.split("\n")
    for i in range(1, len(lines)):
        sep, head = lines[i].strip(), lines[i - 1].strip()
        if not (head.startswith("|") and re.fullmatch(r"\|?(\s*:?-{3,}:?\s*\|)+\s*:?-*:?\s*\|?", sep)):
            continue
        n_head = len(head.strip("|").split("|"))
        cells = [c.strip() for c in sep.strip("|").split("|")]
        if len(cells) != n_head and n_head > 0:
            cells = (cells + ["---"] * n_head)[:n_head]
            lines[i] = "|" + "|".join(cells) + "|"
    return "\n".join(lines)


def fix_inline_tables(t: str) -> str:
    """'| a | b | |---|---| | 1 | 2 |' (a whole table on one line) -> a real multi-line table."""
    merged = []                                  # a one-line table may have been broken mid-row
    for line in t.split("\n"):
        prev = merged[-1].strip() if merged else ""
        if (merged and prev.startswith("|") and re.search(r"\|\s*:?-{3,}", prev)
                and line.strip() and not line.strip().startswith("|") and line.rstrip().endswith("|")):
            merged[-1] = merged[-1].rstrip() + " " + line.strip()
        else:
            merged.append(line)
    out = []
    for line in merged:
        s = line.strip()
        if not (s.startswith("|") and re.search(r"\|\s*:?-{3,}", s) and re.search(r"\|\s*\|", s)):
            out.append(line)
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        rows, row = [], []
        for c in cells:
            if c == "" and row:                  # '| |' separates rows
                rows.append(row)
                row = []
            elif c != "":
                row.append(c)
        if row:
            rows.append(row)
        rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]   # drop old separator
        if len(rows) < 2:
            out.append(line)
            continue
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        out.append("| " + " | ".join(rows[0]) + " |")
        out.append("|" + "---|" * n)
        out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(out)


_KEY_WORDS = {"concurrence_halves_at": "concurrence halves at", "entanglement_vanishes_at": "entanglement vanishes at",
              "chsh_drops_to_2_at": "CHSH drops to 2 at",
              "degree_of_polarization_halves_at": "degree of polarization halves at"}


def to_unicode(text: str) -> str:
    """Convert common LaTeX in an answer to readable Unicode, make kets table-safe,
    and repair unbalanced bold markers."""
    text = text.replace("\u00a0", " ").replace("\u202f", " ")   # non-breaking spaces (Gemini)
    text = session.display_path(text)                            # 'data/sessions/<id>/' -> 'uploads/'
    text = re.sub(r"\b(S|H|I)\(([A-Z])\|([A-Z])\)", r"\1(\2∣\3)", text)   # 'S(A|B)': a '|' breaks a table
    text = re.sub(r"(?m)^\*(?![*\s])([^*\n]+)$", r"*\1*", text)
    text = re.sub(r"\bchsh_max\b", "CHSH max", text)                 # a tool key copied into the prose
    keys = "|".join(_KEY_WORDS)
    text = re.sub(rf"`({keys})\s*[:=]?\s*([^`\n]*)`",                  # `concurrence_halves_at: 0.3466`
                  lambda m: (_KEY_WORDS[m.group(1)] + " " + m.group(2)).strip(), text)
    text = re.sub(rf"\b({keys})\b\s*[:=]?", lambda m: _KEY_WORDS[m.group(1)], text)   # '*Conventions: ...' lost its closing '*'
    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", r"(figure: \2)", text)   # images cannot show in text
    text = re.sub(r"\\(?=\|" + _KET_BODY + r"(?:⟩|>))", "", text)   # '\|HH>' (escaped bar) -> '|HH>'
    text = re.sub(r"\\?mathbb\s*\{?\s*([A-Z1])\s*\}?", lambda m: "I" if m.group(1) in "1I" else m.group(1), text)
    text = re.sub(r"\*\*([^*\n]+?)\s*\*\*[ \t]*\*\*([^*\n]+?)\*\*", r"**\1 \2**", text)  # '**C = ** **0.6**'
    return fix_table_separators(fix_inline_tables(fix_abs_bars(fix_bold(fix_kets(_latex_to_unicode(text))))))


def _latex_to_unicode(text: str) -> str:
    if "\\" not in text and "$" not in text:
        return text
    t = text
    t = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n\n    " + m.group(1).strip() + "\n\n", t, flags=re.S)
    t = t.replace(r"\(", "").replace(r"\)", "")
    t = re.sub(r"\$\$(.+?)\$\$", lambda m: "\n\n    " + m.group(1).strip() + "\n\n", t, flags=re.S)
    t = re.sub(r"(?<!\w)\$([^$\n]+?)\$", r"\1", t)
    t = re.sub(r"\\mathrm\{Tr\}|\\operatorname\{Tr\}|\\text\{Tr\}", "Tr ", t)
    t = re.sub(r"\\(cos|sin|tan|log|ln|exp|max|min|det)(?=\\|[A-Za-z])", r"\1 ", t)
    for cmd in ("text", "mathrm", "mathbf", "mathcal", "operatorname", "textbf", "mathit"):
        t = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", t)
    for _ in range(3):  # nested \frac / \sqrt
        t = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}",
                   lambda m: f"{_wrap(m.group(1))}/{_wrap(m.group(2))}", t)
        t = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: "√" + _wrap(m.group(1)), t)
    t = re.sub(r"\\sqrt(\d+)", r"√\1", t)
    for k in sorted(_SYMBOLS, key=len, reverse=True):
        t = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda m, v=_SYMBOLS[k]: v, t)
    t = re.sub(r"\\left|\\right|\\big|\\Big|\\bigl|\\bigr", "", t)
    t = re.sub(r"\^\{([^{}]*)\}|\^([0-9nTA-Z+-])", _sup, t)
    # subscripts only after a single symbol (l_1, E_{N}); never inside handles like rho_1
    t = re.sub(r"(?<![A-Za-z])([A-Za-zρψφλγθσ])_(?:\{([^{}]*)\}|([0-9]))", _subscript, t)
    t = re.sub(r"\\[,;:!]|\\quad|\\qquad", " " if "\\," in t else "", t)
    t = t.replace("\\,", " ").replace("\\!", "").replace("\\ ", " ")
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)           # any unknown command: drop the backslash
    t = re.sub(r"(?<![\w}])\{([^{}\n]*)\}", r"\1", t)  # stray grouping braces
    t = re.sub(r"Tr\s+ρ", "Tr ρ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t


# ------------------------------------------------------------------ tool steps -> sentences
def _short_error(text: str) -> str:
    if "ToolInvocationError" in text:
        return "it was called with wrong or missing arguments"
    m = re.search(r"ValueError\([\"'](.+?)[\"']\)", text, flags=re.S)
    msg = (m.group(1) if m else text.replace("Error:", "")).strip().split("\n")[0]
    return msg[:120] + ("…" if len(msg) > 120 else "")


def summarize_steps(new_messages: list) -> list[tuple[str, str]]:
    """Return [(kind, sentence)] for one turn. kind in {'ok', 'warn'}."""
    calls, steps = {}, []
    for m in new_messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                calls[tc["id"]] = tc
        elif isinstance(m, ToolMessage):
            tc = calls.get(m.tool_call_id, {})
            name, args = tc.get("name", m.name or "tool"), tc.get("args", {})
            content = str(m.content)
            if content.startswith("Merged with the previous call"):
                what = args.get("start_kind") or ", ".join(args.get("handles") or []) or "this state"
                steps.append(("ok", f"Drew {what} in the same figure"))
            elif content.startswith("Already done in this question"):
                steps.append(("warn", f"Skipped a repeated {name} call (same arguments as before)"))
            elif content.startswith("Error") or getattr(m, "status", "") == "error":
                steps.append(("warn", f"{name} was rejected: {_short_error(content)} (the agent retried)"))
            elif name == "compute_measures":
                steps.append(("ok", f"Computed all measures of {args.get('handle', 'the state')}"))
            elif name == "reconstruct_state_from_counts":
                try:
                    r = json.loads(content)
                    info = r["data"]
                    steps.append(("ok", f"Reconstructed {r['handle']} from {info['file']} by maximum likelihood "
                                        f"({info['settings']} settings, {info['total_counts']} counts, "
                                        f"column {info['count_column_used']})"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    steps.append(("ok", "Reconstructed a state from the data file"))
            elif name == "evolve_state":
                try:
                    r = json.loads(content)
                    proc = str(args.get("process", ""))
                    on = ("the pair" if proc.startswith(("global_", "collective_", "correlated_"))
                          else "the photon" if "one photon" in str(r.get("features", {}).get("model", ""))
                          else args.get("qubits", "both"))
                    steps.append(("ok", f"Evolved {args.get('handle')} under {proc} "
                                        f"(γ = {args.get('gamma')}, t up to {args.get('t_max')}, "
                                        f"{on}) -> {r['final_state']}; "
                                        f"saved figure {os.path.basename(str(r['figure']))}"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    steps.append(("ok", "Evolved a state in time"))
            elif name == "apply_waveplate":
                try:
                    r = json.loads(content)
                    steps.append(("ok", f"Sent {args.get('handle')} through a {r['element']} -> {r['new_handle']}"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    steps.append(("ok", "Applied a wave plate"))
            elif name == "compensate_phase":
                try:
                    r = json.loads(content)
                    steps.append(("ok", f"Phase compensation of {args.get('handle')}: "
                                        f"{r['wave_plates']['sequence']} -> {r['new_handle']}"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    steps.append(("ok", "Computed a phase compensation"))
            elif name == "simulate_tomography":
                try:
                    r = json.loads(content)
                    steps.append(("ok", f"Simulated a 36-setting tomography of {args.get('handle')} with "
                                        f"detector noise -> {r['reconstructed_raw']['handle']} (raw), "
                                        f"{r['reconstructed_accidentals_subtracted']['handle']} (accidentals "
                                        f"subtracted); counts saved to {r['csv_file']}"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    steps.append(("ok", "Simulated a tomography experiment"))
            elif name == "certify_correlations":
                steps.append(("ok", f"Certified entanglement, steering and Bell nonlocality of {args.get('handle')}"))
            elif name in ("teleportation_fidelity", "qkd_key_rate", "bit_flip_code"):
                what = {"teleportation_fidelity": f"Teleportation fidelity of {args.get('handle')}",
                        "qkd_key_rate": ("QKD key rate of " + args["handle"]) if args.get("handle")
                                        else f"QKD key rate for QBER {args.get('qber_z')} (H/V), "
                                             f"{args.get('qber_x') if args.get('qber_x') is not None else args.get('qber_z')} (D/A)",
                        "bit_flip_code": f"Bit-flip code at p = {args.get('p')}"}[name]
                steps.append(("ok", f"Computed: {what}"))
            elif name == "show_density_matrix":
                extra = f" and its change from {args['compare_with']}" if args.get("compare_with") else ""
                steps.append(("ok", f"Showed the density matrix of {args.get('handle')}{extra}"))
            elif name == "get_kraus_operators":
                steps.append(("ok", f"Listed the Kraus operators of {args.get('channel')} at {args.get('param')}"))
            elif name == "reconstruct_single_photon":
                try:
                    steps.append(("ok", f"Reconstructed the one-photon state {json.loads(content)['handle']} from the "
                                        "counts (maximum likelihood)"))
                except (json.JSONDecodeError, KeyError, TypeError):
                    steps.append(("ok", "Reconstructed a one-photon state"))
            elif name == "get_tomography_report":
                steps.append(("ok", f"Looked up the stored tomography report of {args.get('handle')}"))
            elif name == "compare_states":
                steps.append(("ok", f"Compared {args.get('handle_a')} and {args.get('handle_b')} "
                                    "(fidelity, trace distance)"))
            elif name.startswith("plot_"):
                try:
                    path = json.loads(content).get("figure", content)
                except (json.JSONDecodeError, AttributeError):
                    path = content
                steps.append(("ok", f"Saved figure {os.path.basename(str(path))}"))
            else:
                steps.append(("ok", content.rstrip(".")))
        elif isinstance(m, HumanMessage) and str(m.content).startswith("VERIFIER:"):
            body = re.sub(r"^\[[a-z ]+\]\s*", "", str(m.content)[10:])
            body = re.split(r"\. (?:Rewrite|Remove|Call now|Create )", body)[0]
            steps.append(("warn", "Answer check: " + body + " (the agent was asked to fix it)"))
    return steps


# ------------------------------------------------------------------ tomography report (no LLM involved)
def _pm(value, err) -> str:
    return f"{value:.4f} ± {err:#.2g}" if err is not None else f"{value:.4f}"   # 0.0060, not 0.006


def format_tomography_report(r: dict) -> str:
    """Markdown table built directly from the tool result, so every number is exact."""
    k, u = r["key_numbers"], r.get("uncertainty_1sigma", {})
    d, best, fit = r["data"], r["closest_maximally_entangled"], r["fit_quality"]
    phi = best["phi_deg"]
    phase = f"e^({'-' if phi < 0 else ''}i{abs(phi):.1f}°)"
    target = best["state"].replace("e^(iφ)", phase)
    chi = fit["chi2_per_dof"]
    verdict = fit.get("verdict") or ("consistent with counting noise" if chi < 1.5 else
                                     "larger than counting noise: systematic errors present")
    pval = f", dof {fit['dof']}, p = {fit['p_value']:.2g}" if "p_value" in fit else ""
    rows = [
        ("Data", f"{d['file']} · {d['settings']} settings · {d['total_counts']} counts · "
                 f"column {d['count_column_used']}"),
        ("Purity", _pm(k["purity"], u.get("purity"))),
        ("Concurrence", _pm(k["concurrence"], u.get("concurrence"))),
        ("Negativity", _pm(k["negativity"], u.get("negativity"))),
        ("CHSH max (local bound 2)", _pm(k["chsh_max"], u.get("chsh_max"))),
        ("Closest maximally entangled state", f"{target}, fidelity "
                                              f"{_pm(k['bell_like_fidelity'], u.get('bell_like_fidelity'))}"),
        ("Eigenvalues of ρ", ", ".join(f"{x:.4f}" for x in r["eigenvalues"])),
        ("Fit quality", f"χ²/dof = {chi}{pval}: {verdict}"),
        ("Largest deviations", "; ".join(fit["largest_deviations"])),
        ("Error bars", fit.get("error_bar_scale", "statistical only")),
        *([("Realistic ± (scaled)", ", ".join(f"{n} ± {r['uncertainty_realistic'][key]:#.2g}" for n, key in
                                               (("purity", "purity"), ("C", "concurrence"),
                                                ("N", "negativity"), ("CHSH", "chsh_max"))))]
          if "uncertainty_realistic" in r else []),
        ("Conventions", f"{r['conventions']['setting_label']}; {r['conventions']['basis']}"),
        ("Assumptions", "; ".join(r["assumptions"])),
    ]
    head = f"**Tomography report — {r['handle']}**\n\n| Quantity | Value |\n|---|---|\n"
    body = "\n".join(f"| {a} | {fix_kets(b)} |" for a, b in rows)
    note = f"\n\n<sub>± = 1σ, {r['uncertainty_method']}</sub>" if "uncertainty_method" in r else ""
    return head + body + note


def format_comparisons(results: list[dict]) -> str:
    """Markdown table of every compare_states result of one turn."""
    def name(r, h):
        desc = r.get("states", {}).get(h, "")
        desc = desc.split("; full report")[0]
        short = re.match(r"reconstructed from \S+", desc)
        desc = short.group(0) if short else desc
        return fix_kets(f"{h} ({desc})" if desc else h)

    rows = "\n".join(f"| {name(r, r['handles'][0])} | {name(r, r['handles'][1])} | "
                      f"{r['fidelity']:.4f} | {r['trace_distance']:.4f} |" for r in results)
    return ("**State comparisons**\n\n| State A | State B | Fidelity | Trace distance |\n"
            "|---|---|---|---|\n" + rows + "\n\n<sub>fidelity squared: 1 = identical, 0 = orthogonal; "
            "trace distance 0 = identical, 1 = perfectly distinguishable</sub>")


def result_panels(new_messages: list) -> list[tuple[str, str]]:
    """Tables built straight from tool results (no LLM in between): [(title, markdown)]."""
    panels = [("Tomography report", r) for r in tomography_reports(new_messages)]
    comps = []
    for m in new_messages:
        if isinstance(m, ToolMessage) and m.name == "compare_states":
            try:
                comps.append(json.loads(m.content))
            except (json.JSONDecodeError, TypeError):
                continue
    if comps:
        panels.append(("Comparison", format_comparisons(comps)))
    makers = {"evolve_state": ("Time evolution", lambda r: format_evolution(r)),
              "apply_waveplate": ("Wave plate", lambda r: format_waveplate(r)),
              "compensate_phase": ("Phase compensation", lambda r: format_compensation(r)),
              "simulate_tomography": ("Simulated experiment", lambda r: format_simulation(r)),
              "certify_correlations": ("Certification", lambda r: format_certification(r)),
              "plot_sweep": ("Sweep", lambda r: format_sweep(r)),
              "plot_fringes": ("Fringes", lambda r: format_fringes(r)),
              "plot_bloch_and_correlations": ("Bloch vectors and correlations", lambda r: format_bloch(r)),
              "plot_tomography_fit": ("Tomography fit", lambda r: format_tomography_fit(r)),
              "plot_data_file": ("Data fit", lambda r: format_data_fit(r)),
              "show_density_matrix": ("Density matrix", lambda r: format_density_matrix(r)),
              "get_kraus_operators": ("Kraus operators", lambda r: format_kraus(r)),
              "reconstruct_single_photon": ("One-photon tomography", lambda r: format_single_photon_tomography(r))}
    for m in new_messages:
        if isinstance(m, ToolMessage) and m.name in makers:
            title, make = makers[m.name]
            try:
                panels.append((title, make(json.loads(m.content))))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return panels


def _jones_text(rows) -> str:
    """[['+1.0000-0.0000i', ...], ...] -> '[[1, 0], [0, i]]' (entries rounded to 4 digits)."""
    def one(s):
        try:
            z = complex(str(s).replace("i", "j").replace(" ", ""))
        except ValueError:
            return str(s)
        re_, im = round(z.real, 4) + 0.0, round(z.imag, 4) + 0.0
        num = lambda x: f"{x:.4f}".rstrip("0").rstrip(".")            # noqa: E731
        if im == 0:
            return num(re_)
        imag = ("" if abs(im) == 1 else num(abs(im))) + "i"
        if re_ == 0:
            return ("-" if im < 0 else "") + imag
        return f"{num(re_)} {'-' if im < 0 else '+'} {imag}"
    try:
        return "[" + ", ".join("[" + ", ".join(one(x) for x in row) + "]" for row in rows) + "]"
    except TypeError:
        return str(rows)


def _pol_name(ket: str) -> str:
    m = re.search(r"\(= ([A-Z])\)", ket)
    return m.group(1) if m else fix_kets(ket)


def format_waveplate(r: dict) -> str:
    """Before/after numbers of one wave plate (a local unitary)."""
    b, a = r["before"], r["after"]
    if r.get("photons") == 1:
        keys = list(dict.fromkeys([k for k in b if isinstance(b[k], (int, float, str))] +
                                  [k for k in a if isinstance(a[k], (int, float, str))]))
        cell = lambda d, k: (f"{d[k]:.4f}" if isinstance(d.get(k), (int, float)) else    # noqa: E731
                             str(d.get(k, "undefined (circular)")))
        body = "\n".join(f"| {k.replace('_', ' ')} | {cell(b, k)} | {cell(a, k)} |" for k in keys)
        pb, pa = r.get("polarized_part_before", "none"), r.get("polarized_part_after", "none")
        moved = (f"polarization {_pol_name(pb)} → {_pol_name(pa)}" if pb != "none" else
                 "unpolarized light stays unpolarized")
        return (f"**Wave plate — {r['element']} → {r['new_handle']}** (one photon)\n\n"
                f"{moved}; Jones matrix (H, V basis, global phase dropped) {_jones_text(r['jones_matrix_HV_basis'])}\n\n"
                "| Quantity | Before | After |\n|---|---|---|\n" + body +
                f"\n\n<sub>{r['note']}; {r['conventions']}</sub>")
    rows = [(q, f"{b[q]:.4f}", f"{a[q]:.4f}") for q in ("concurrence", "negativity", "purity", "chsh_max")]
    rows += [(f"fidelity with {k}", f"{b['fidelity_with_bell_states'][k]:.4f}",
              f"{a['fidelity_with_bell_states'][k]:.4f}") for k in ("phi+", "phi-", "psi+", "psi-")]
    body = "\n".join(f"| {fix_kets(q)} | {x} | {y} |" for q, x, y in rows)
    return (f"**Wave plate — {r['element']} → {r['new_handle']}**\n\n"
            f"{fix_kets(r['acts_on_H'])}; {fix_kets(r['acts_on_V'])}\n\n"
            "| Quantity | Before | After |\n|---|---|---|\n" + body +
            f"\n\n<sub>{r['note']}; {r['conventions']}</sub>")


def format_compensation(r: dict) -> str:
    w = r["wave_plates"]
    rows = [("Needed?", "No — " + r["no_compensation_needed"])] if "no_compensation_needed" in r else []
    rows += [("Closest state", fix_kets(r["closest_state"])),
            ("Phase shifter", r["phase_shifter"]),
            ("Wave plates", w["sequence"]),
            ("Alternative", w["alternative"]),
            (f"Fidelity with {r['target']}", f"{r['fidelity_with_target_before']:.4f} → "
                                            f"{r['fidelity_with_target_after']:.4f}"),
            ("Unchanged", f"concurrence {r['unchanged']['concurrence']:.4f}, "
                          f"purity {r['unchanged']['purity']:.4f}"),
            ("Lab check", r["lab_check"])]
    if "convention_warning" in r:
        rows.append(("R/L convention", r["convention_warning"]))
    body = "\n".join(f"| {a} | {b} |" for a, b in rows)
    return (f"**Phase compensation → {r['new_handle']}**\n\n| Item | Value |\n|---|---|\n" + body +
            f"\n\n<sub>{w['rule']}</sub>")


def format_simulation(r: dict) -> str:
    t, raw, sub = r["true_state"], r["reconstructed_raw"], r["reconstructed_accidentals_subtracted"]
    inf = r["infinite_statistics_raw"]
    rows = []
    for q, label in (("concurrence", "Concurrence"), ("negativity", "Negativity"), ("purity", "Purity"),
                     ("chsh_max", "CHSH max")):
        err = f" ± {raw['concurrence_error_1sigma']:#.2g}" if q == "concurrence" else ""
        rows.append((label, f"{t[q]:.4f}", f"{raw[q]:.4f}{err}", f"{sub[q]:.4f}", f"{inf[q]:.4f}"))
    rows.append(("Fidelity with true state", "1.0000", f"{raw['fidelity_with_true_state']:.4f}",
                 f"{sub['fidelity_with_true_state']:.4f}", f"{inf['fidelity_with_true_state']:.4f}"))
    body = "\n".join("| " + " | ".join(x) + " |" for x in rows)
    d, rt = r["detector_model"], r["rates"]
    need = r["counts_needed_for_target"]
    need = (f"{need['time_per_setting_s']:g} s per setting ({need['total_coincidences']} coincidences) "
            f"for σ_C = {need['target_concurrence_error']}, {need.get('accuracy', '')}"
            if isinstance(need, dict) else need)
    return (f"**Simulated experiment — {t['handle']} ({fix_kets(t['description'])})**\n\n"
            f"R = {d['pair_rate_per_s']:,.0f} pairs/s, η = {d['eta_a']:g}/{d['eta_b']:g}, dark "
            f"{d['dark_a_per_s']:g}/{d['dark_b_per_s']:g} per s, τ = {d['window_ns']:g} ns, "
            f"T = {d['time_per_setting_s']:g} s per setting · accidental fraction "
            f"{rt['accidental_fraction']:.4g} · {rt['total_coincidences_recorded']} coincidences\n\n"
            f"| Quantity | True | Raw ({raw['handle']}) | Accidentals subtracted ({sub['handle']}) | "
            "Raw, no counting noise |\n|---|---|---|---|---|\n" + body +
            f"\n\n<sub>Counting time needed: {need}. Counts saved to {r['csv_file']}</sub>")


def format_evolution(r: dict) -> str:
    """Exact values at t = 0, at every key time (crossings) and at t_max."""
    f = r["features"]
    if "degree_of_polarization" in f["at_t0"]:                 # one photon
        rows = [(k.replace("t=", "t = "), v) for k, v in f["samples"].items()]
        rows += [(k.replace("t=", "t = "), v) for k, v in f.get("at_key_times", {}).items()]
        rows.sort(key=lambda kv: float(re.match(r"t = ([\d.]+)", kv[0]).group(1)))
        body = "\n".join(f"| {name} | {v['S1']:.4f} | {v['S2']:.4f} | {v['S3']:.4f} | "
                          f"{v['degree_of_polarization']:.4f} | {v['purity']:.4f} |" for name, v in rows)
        return (f"**Time evolution — {r['final_state']}** ({f['model']})\n\n"
                "| Time | S1 (H−V) | S2 (D−A) | S3 (R−L) | Degree of polarization | Purity |\n"
                "|---|---|---|---|---|---|\n" + body +
                f"\n\n<sub>figure {os.path.basename(str(r['figure']))}</sub>")
    rows = [("t = 0", f["at_t0"])]
    rows += [(k.replace("t=", "t = ", 1), v) for k, v in f.get("at_key_times", {}).items()]
    t_max = list(f["samples"])[-1].replace("t=", "t = ")
    rows.append((f"{t_max} (end)", f["at_t_max"]))
    rows.sort(key=lambda kv: float(re.match(r"t = ([\d.]+)", kv[0]).group(1)))
    body = "\n".join(f"| {name} | {v['concurrence']:.4f} | {v['negativity']:.4f} | {v['purity']:.4f} | "
                      f"{v['chsh_max']:.4f} |" for name, v in rows)
    death = f["entanglement_vanishes_at"]
    death = f"sudden death at t = {death:.4f}" if isinstance(death, (int, float)) else str(death)
    return (f"**Time evolution — {r['final_state']}** ({f['model']})\n\n"
            "| Time | Concurrence | Negativity | Purity | CHSH max |\n|---|---|---|---|---|\n" + body +
            f"\n\n<sub>{death}; figure {os.path.basename(str(r['figure']))}</sub>")


def _short_desc(d: str) -> str:
    """A state description for people: without the hint meant for the model ('<this handle>' would also be
    hidden by Markdown as an HTML tag)."""
    return d.split("; full report")[0]


def matrix_table(rows: list[list[str]], basis: list[str], errors: dict | None = None) -> str:
    """A density matrix as a Markdown table (rows and columns labelled with the basis states)."""
    head = "| ρ | " + " | ".join(basis) + " |\n|---|" + "---|" * len(basis) + "\n"
    lines = []
    for i, (b, row) in enumerate(zip(basis, rows)):
        cells = row
        if errors:
            cells = [f"{c} (±{errors['re'][i][j]}{', ±' + errors['im'][i][j] + 'i' if i != j else ''})"
                     for j, c in enumerate(row)]
        lines.append(f"| **{b}** | " + " | ".join(cells) + " |")
    return head + "\n".join(lines)


def format_density_matrix(r: dict) -> str:
    """show_density_matrix: the matrix (and the comparison and Δρ) as tables."""
    state = _short_desc(r.get("state", ""))
    title = (f"**Density matrix of {r['handle']}** ({fix_kets(state)}; {r['size']})" if state else
             f"**Density matrix of {r['handle']}** ({r['size']})")
    out = [title, matrix_table(r["matrix"], r["basis"], r.get("element_errors_1sigma"))]
    out.append(f"<sub>eigenvalues {', '.join(f'{v:.4f}' for v in r['eigenvalues'])}; rows and columns "
               f"{', '.join(r['basis'])}" + (" (first letter = photon A)" if len(r["basis"]) == 4 else "") +
               ("; ± = 1σ of Re and Im from the tomography resamplings" if r.get("element_errors_1sigma") else "")
               + "</sub>")
    if "compared_with" in r:
        c = r["compared_with"]
        out += [f"**Density matrix of {c['handle']}** ({fix_kets(_short_desc(c['state']))})",
                matrix_table(c["matrix"], r["basis"]),
                f"**Change Δρ = ρ({r['handle']}) − ρ({c['handle']})**", matrix_table(r["difference"], r["basis"])]
    return "\n\n".join(out)


def format_kraus(r: dict) -> str:
    """get_kraus_operators: every operator as a small table."""
    parts = [f"**Kraus operators — {r['channel']} at {r['param']:g}** ({r['scope']})",
             f"{r['parameter_meaning']}"]
    for name, rows in r["kraus_operators"].items():
        parts.append(matrix_table(rows, r["basis"]).replace("| ρ |", f"| {name} |"))
    if r.get("coherence_factors"):
        parts.append("Each coherence ρ_ij is multiplied by: " + ", ".join(
            f"{k.replace('-', '–')} {v:.4f}" for k, v in r["coherence_factors"].items()) + " (populations unchanged)")
    parts.append(f"<sub>ρ → Σ K ρ K†; completeness max|Σ K†K − I| = {r['completeness_error']:.1e} "
                 f"({'trace preserving' if r['trace_preserving'] else 'NOT trace preserving'})</sub>")
    return "\n\n".join(parts)


def format_single_photon_tomography(r: dict) -> str:
    rows = [("Settings", ", ".join(r["settings"]) + f" · {r['total_counts']:g} counts")]
    rows += [(k, v) for k, v in r["stokes_mle"].items()]
    rows += [("Degree of polarization", r["degree_of_polarization"]), ("Purity", r["purity"])]
    lin = r["stokes_linear_inversion"]
    rows.append(("Linear inversion", ", ".join(f"{k} = {v}" for k, v in lin.items()) if isinstance(lin, dict) else lin))
    body = "\n".join(f"| {a} | {b} |" for a, b in rows)
    return (f"**One-photon tomography → {r['handle']}**\n\n| Quantity | Value |\n|---|---|\n{body}\n\n"
            f"<sub>{r['note']}; {r['convention']}</sub>")


def tomography_reports(new_messages: list) -> list[str]:
    """Reports for every reconstruction (or stored report) produced in this turn."""
    out = []
    for m in new_messages:
        if isinstance(m, ToolMessage) and m.name in ("reconstruct_state_from_counts", "get_tomography_report"):
            try:
                r = json.loads(m.content)
                out.append(format_tomography_report(r))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return list(dict.fromkeys(out))


def format_certification(r: dict) -> str:
    """Entanglement / steering / Bell tests with their significance, straight from the tool."""
    has_sig = any("n_sigma" in t for t in r["tests"])
    real = any("n_sigma_realistic" in t for t in r["tests"])
    head = "| Property | Value | Bound |" + (" σ | Significance |" if has_sig else "") + \
           (" Realistic |" if real else "") + " Verdict |"
    cols = 4 + (2 if has_sig else 0) + (1 if real else 0)
    rows = []
    for t in r["tests"]:
        cells = [t["property"], f"{t['value']:.4f}", f"{t['bound']:g}"]
        if has_sig:
            cells += [f"{t['sigma']:#.2g}" if "sigma" in t else "",
                      f"{t['n_sigma']:.1f}σ" if "n_sigma" in t else ""]
        if real:
            cells.append(f"{t['n_sigma_realistic']:.1f}σ" if "n_sigma_realistic" in t else "")
        cells.append(t.get("verdict", ""))
        rows.append("| " + " | ".join(cells) + " |")
    return (f"**Certification — {r['handle']}** · strongest property: {r['strongest_property']}\n\n"
            + head + "\n|" + "---|" * cols + "\n" + "\n".join(rows)
            + f"\n\n<sub>{r['hierarchy']}. Fidelity witness target: {fix_kets(r['fidelity_witness_target'])}. "
            f"{r['error_method']}.</sub>")


# ------------------------------------------------------------------ plot results (no LLM involved)
_QNAME = {"concurrence": "Concurrence", "negativity": "Negativity", "log_negativity": "Log-negativity",
          "entanglement_of_formation": "EoF", "purity": "Purity", "entropy": "Entropy", "chsh_max": "CHSH max",
          "mutual_information": "Mutual information", "discord": "Discord", "coherence_l1": "l1 coherence",
          "fully_entangled_fraction": "Fully entangled fraction", "teleportation_fidelity": "Teleportation fidelity",
          "steering_f3": "Steering F3", "key_fraction": "Key fraction", "fidelity": "Fidelity"}


def format_sweep(r: dict) -> str:
    rows = []
    for key, res in r["results"].items():
        for q, m in res["measures"].items():
            cr = m.get("crossings", "")
            cr = ("; ".join(f"{c['at']:g} ({c['direction']})" for c in cr) if isinstance(cr, list) else
                  ("none in this range" if cr else "—"))
            at = lambda e: ", ".join(f"{a:g}" if isinstance(a, (int, float)) else str(a)   # noqa: E731
                                     for a in (e["at"] if isinstance(e["at"], list) else [e["at"]]))
            rows.append(f"| {fix_kets(key)} | {_QNAME.get(q, q)} | {m['at_start']:.4f} | {m['at_end']:.4f} | "
                        f"{m['min']['value']:.4f} at {at(m['min'])} | {m['max']['value']:.4f} at {at(m['max'])} | "
                        f"{cr} |")
    return (f"**Sweep — {fix_kets(r['sweep'])}**, x = {r['x_axis']} from {r['range'][0]:g} to {r['range'][1]:g}\n\n"
            "| State | Measure | Start | End | Minimum | Maximum | Crosses its bound at |\n|---|---|---|---|---|---|---|\n"
            + "\n".join(rows) + f"\n\n<sub>figure {os.path.basename(str(r['figure']))}; {r['note']}</sub>")


def format_fringes(r: dict) -> str:
    rows = "\n".join(f"| {f['photon_A_angle_deg']:g}° ({f['basis']}) | {f['visibility']:.4f} | {f['maximum_at_deg']:g}° | "
                     f"{f['max_probability']:.4f} | {f['min_probability']:.4f} |" for f in r["fringes"])
    return ("**Polarization-correlation fringes**\n\n| Photon A polarizer | Visibility | Maximum at (photon B) | "
            "P max | P min |\n|---|---|---|---|---|\n" + rows + f"\n\n<sub>figure {os.path.basename(str(r['figure']))}</sub>")


def format_bloch(r: dict) -> str:
    if r["states"] and "S1 (H-V)" in next(iter(r["states"].values())):        # one photon
        rows = "\n".join(f"| {h} | {s['S1 (H-V)']:+.4f} | {s['S2 (D-A)']:+.4f} | {s['S3 (R-L)']:+.4f} | "
                         f"{s['degree_of_polarization']:.4f} |" for h, s in r["states"].items())
        return ("**Poincaré sphere (one photon)**\n\n| State | S1 (H−V) | S2 (D−A) | S3 (R−L) | Degree of "
                "polarization |\n|---|---|---|---|---|\n" + rows
                + f"\n\n<sub>figure {os.path.basename(str(r['figure']))}</sub>")
    rows = "\n".join(f"| {h} | ({', '.join(f'{v:+.3f}' for v in s['bloch_A'])}) | {s['length_A']:.4f} | "
                     f"({', '.join(f'{v:+.3f}' for v in s['bloch_B'])}) | {s['length_B']:.4f} | "
                     f"{', '.join(f'{v:.4f}' for v in s['singular_values_of_T'])} |" for h, s in r["states"].items())
    return ("**Bloch vectors (X = D/A, Y = R/L, Z = H/V) and correlations**\n\n| State | Photon A | Length A | "
            "Photon B | Length B | Singular values of T |\n|---|---|---|---|---|---|\n" + rows
            + f"\n\n<sub>figure {os.path.basename(str(r['figure']))}</sub>")


def format_tomography_fit(r: dict) -> str:
    b = r["bootstrap"]
    beyond2 = ", ".join(r["settings_beyond_2_sigma"]) or "none"
    rows = [("Data", r["data"]), ("χ²/dof", f"{r['chi2_per_dof']} (dof {r['dof']}"
                                          + (f", p = {r['p_value']:.2g}" if "p_value" in r else "") + ")"
                                          + (f": {r['fit_verdict']}" if "fit_verdict" in r else "")),
            ("Settings beyond 2σ", beyond2 + f" (counting noise alone would give about "
                                             f"{r['expected_beyond_2_sigma_if_only_counting_noise']:g})"),
            ("Settings beyond 3σ", r["settings_beyond_3_sigma"] if isinstance(r["settings_beyond_3_sigma"], str)
             else ", ".join(r["settings_beyond_3_sigma"])),
            ("Concurrence (bootstrap)", f"{b['concurrence']['value']:.4f} ± {b['concurrence']['sigma']:#.2g}; "
                                        f"resamplings with C = 0: {b['concurrence']['fraction_of_resamplings_with_C_0']:.1%}"),
            ("CHSH max (bootstrap)", f"{b['chsh_max']['value']:.4f} ± {b['chsh_max']['sigma']:#.2g}; "
                                     f"resamplings with CHSH ≤ 2: {b['chsh_max']['fraction_of_resamplings_at_or_below_2']:.1%}")]
    return ("**Tomography fit diagnostics**\n\n| Item | Value |\n|---|---|\n"
            + "\n".join(f"| {a} | {v} |" for a, v in rows) + f"\n\n<sub>figure {os.path.basename(str(r['figure']))}</sub>")


def format_data_fit(r: dict) -> str:
    if "fits" not in r:
        rows = "\n".join(f"| {k} | {v['points']} | {v['x_range'][0]:g} – {v['x_range'][1]:g} | "
                         f"{v['y_range'][0]:g} – {v['y_range'][1]:g} |" for k, v in r["data"].items())
        return (f"**Data — {r['file']}** (x = {r['x_column']})\n\n| Column | Points | x range | y range |\n"
                "|---|---|---|---|\n" + rows + f"\n\n<sub>figure {os.path.basename(str(r['figure']))}</sub>")
    rows = []
    for col, f in r["fits"].items():
        if "parameters" not in f:
            rows.append(f"| {col} | fit | {f['fit_verdict']} |")
            continue
        for name, p in f["parameters"].items():
            rows.append(f"| {col} | {name} | {p['value']:g}" + (f" ± {p['error']:g}" if p.get("error") else "") + " |")
        extras = [k for k in ("visibility", "fwhm") if k in f]
        if "period" in f and "period_fixed" not in f:
            extras.append("period")
        for extra in extras:
            p = f[extra]
            if True:
                rows.append(f"| {col} | {extra} | {p['value']:g}" + (f" ± {p['error']:g}" if p.get("error") else "") + " |")
        quality = (f"χ²/dof = {f['chi2_per_dof']} (p = {f['p_value']:.2g})" if "chi2_per_dof" in f
                   else f"rms residual {f['rms_residual']:g}")
        rows.append(f"| {col} | fit quality | {quality}; {f.get('fit_verdict', '')} |")
    return (f"**Data fit — {r['file']}**: {r['fit_model']}\n\n| Column | Parameter | Value |\n|---|---|---|\n"
            + "\n".join(rows) + f"\n\n<sub>1σ errors; figure {os.path.basename(str(r['figure']))}</sub>")
