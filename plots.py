"""
plots.py - every figure the agent draws, in one consistent style, plus curve fitting for data.

Design rules (applied to every figure):
  * one validated categorical palette, assigned in fixed order (never cycled);
  * thin marks: 2 px lines, markers with a ring in the surface colour, hairline solid grid;
  * never two y-scales on one axis: quantities with different ranges get their own panel
    (small multiples), each with its own reference line (e.g. CHSH = 2);
  * text in ink colours, never in the series colour; a legend whenever there are >= 2 series.
Nothing here knows about the LLM: functions take numbers and return a saved PNG path.
"""
from __future__ import annotations

FILES_VERSION = "2026-11-02"   # must match in graph.py, agent_tools.py, llm.py, plots.py, run.py

import hashlib
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no GUI window; save PNGs instead
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from cycler import cycler

FIG_DIR = "figures"

# ------------------------------------------------------------------ palette (validated reference set)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"                         # status colour: only for "beyond 3σ" marks, always labelled
DIVERGING = LinearSegmentedColormap.from_list("qi_diverging", ["#e34948", "#f0efec", "#2a78d6"])
# cyclic phase map with the same meaning as the diverging map: 0° (positive real) = blue,
# ±180° (negative real) = red, +90° = aqua, -90° = yellow; -180° and +180° are the same colour
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "qi_phase", [(0.0, "#e34948"), (0.25, "#eda100"), (0.5, "#2a78d6"), (0.75, "#1baf7a"), (1.0, "#e34948")])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.prop_cycle": cycler(color=SERIES), "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "savefig.dpi": 130, "figure.dpi": 100,
    })


apply_style()


# ------------------------------------------------------------------ helpers
def figure_path(stem: str, params=None) -> str:
    """figures/<stem>[_<hash of params>].png - different settings never overwrite each other."""
    import session
    fig_dir = session.current().fig_dir             # figures/ (command line) or the web user's folder
    os.makedirs(fig_dir, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_+\-]+", "_", stem).strip("_")
    if params is not None:
        stem += "_" + hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{fig_dir}/{stem}.png"                      # "/" works on Windows too


def save(fig, path: str) -> str:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _dot(ax, x, y, color, **kw):
    """Marker with a 2 px ring in the surface colour (stays legible on top of lines)."""
    ax.plot(x, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5, **kw)


def _reference_line(ax, level: float, text: str):
    ax.axhline(level, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(text, xy=(1, level), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)


def _ink_on(value: float, vmax: float) -> str:
    """Text colour on a heat-map cell: white on strong fills, ink otherwise."""
    return "white" if vmax > 0 and abs(value) / vmax > 0.6 else INK


BASIS = ["HH", "HV", "VH", "VV"]


# ------------------------------------------------------------------ density matrices
def density_matrix_figure(states: list[tuple[str, np.ndarray]], style: str, path: str) -> str:
    """One row per state. style: 'bars3d' (Re, Im), 'heatmap' (Re, Im with values written in the
    cells) or 'magnitude_phase' (bar height |ρ_ij|, colour = phase of ρ_ij)."""
    rows = len(states)
    lim = max(0.05, max(float(np.abs(r).max()) for _, r in states))
    d = states[0][1].shape[0]
    basis = BASIS if d == 4 else ["H", "V"]
    if style == "magnitude_phase":
        fig = plt.figure(figsize=(5.2, 4.2 * rows))
        for k, (title, rho) in enumerate(states):
            ax = fig.add_subplot(rows, 1, k + 1, projection="3d")
            xs, ys = np.meshgrid(range(d), range(d), indexing="ij")
            mag, ph = np.abs(rho).ravel(), np.degrees(np.angle(rho)).ravel()
            ph = np.where(mag < 1e-6, 0.0, ph)
            colors = PHASE_CMAP(Normalize(-180, 180)(ph))
            ax.bar3d(xs.ravel() - 0.3, ys.ravel() - 0.3, 0, 0.6, 0.6, mag, color=colors, shade=True)
            ax.set_xticks(range(d), basis); ax.set_yticks(range(d), basis)
            ax.set_zlim(0, lim * 1.05)
            ax.set_title(f"|ρ| coloured by phase — {title}", fontsize=10)
            sm = plt.cm.ScalarMappable(cmap=PHASE_CMAP, norm=Normalize(-180, 180))
            cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.1, ticks=[-180, -90, 0, 90, 180])
            cb.set_label("phase of ρ_ij (degrees)")
        return save(fig, path)

    fig = plt.figure(figsize=(9.6, 4.1 * rows), layout="constrained" if style == "heatmap" else None)
    for k, (title, rho) in enumerate(states):
        for j, (part, name) in enumerate(((rho.real, "Re"), (rho.imag, "Im"))):
            if style == "heatmap":
                ax = fig.add_subplot(rows, 2, 2 * k + j + 1)
                im = ax.imshow(part, cmap=DIVERGING, vmin=-lim, vmax=lim)
                ax.set_xticks(range(d), basis); ax.set_yticks(range(d), basis)
                ax.grid(False)
                for a in range(d):
                    for b in range(d):
                        v = part[a, b]
                        ax.text(b, a, "0" if abs(v) < 0.005 else f"{v:+.2f}", ha="center", va="center",
                                fontsize=9, color=_ink_on(v, lim))
                if j == 1:
                    fig.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax = fig.add_subplot(rows, 2, 2 * k + j + 1, projection="3d")
                xs, ys = np.meshgrid(range(d), range(d), indexing="ij")
                vals = part.ravel()
                colors = [SERIES[0] if v >= 0 else SERIES[7] for v in vals]   # blue >= 0, red < 0
                ax.bar3d(xs.ravel() - 0.3, ys.ravel() - 0.3, 0, 0.6, 0.6, vals, color=colors, shade=True)
                ax.set_xticks(range(d), basis); ax.set_yticks(range(d), basis)
                ax.set_zlim(-lim, lim)
            ax.set_title(f"{name}(ρ) — {title}", fontsize=10)
    return save(fig, path)


# ------------------------------------------------------------------ Bloch vectors and correlations
_AXES_LABELS = {(0, 0, 1.25): "H", (0, 0, -1.25): "V", (1.25, 0, 0): "D", (-1.25, 0, 0): "A",
                (0, 1.25, 0): "R", (0, -1.25, 0): "L"}


def _bloch_sphere(ax, title: str):
    u, v = np.mgrid[0:2 * np.pi:25j, 0:np.pi:13j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color=GRID, lw=0.4)
    for d in np.eye(3):
        ax.plot(*np.stack([-d, d]).T, color=AXIS, lw=0.8)
    for (x, y, z), lab in _AXES_LABELS.items():
        ax.text(x, y, z, lab, color=INK2, fontsize=9, ha="center", va="center")
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def bloch_correlation_figure(items: list[dict], path: str) -> str:
    """items: [{'label', 'a' (3,), 'b' (3,), 'T' (3,3)}] (<= 3). Top: Bloch vectors of photon A
    and photon B for every state. Bottom: correlation matrix T of every state."""
    n = len(items)
    fig = plt.figure(figsize=(max(8.5, 3.9 * n), 7.6))
    gs = fig.add_gridspec(2, max(2, n), height_ratios=[1.15, 1], wspace=0.45, hspace=0.25)
    for side, key in ((0, "a"), (1, "b")):
        span = max(2, n) // 2
        ax = fig.add_subplot(gs[0, side * span:(side + 1) * span] if n > 2 else gs[0, side], projection="3d")
        _bloch_sphere(ax, f"photon {'A' if key == 'a' else 'B'} (reduced state)")
        for i, it in enumerate(items):
            vec = np.asarray(it[key], float)
            c = SERIES[i]
            if np.linalg.norm(vec) > 1e-3:
                ax.quiver(0, 0, 0, *vec, color=c, lw=2, arrow_length_ratio=0.12)
            ax.scatter(*vec, s=55, color=c, edgecolors=SURFACE, linewidths=2, depthshade=False,
                       label=f"{it['label']}: |r| = {np.linalg.norm(vec):.3f}")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    ticks = ["X (D/A)", "Y (R/L)", "Z (H/V)"]
    for i, it in enumerate(items):
        ax = fig.add_subplot(gs[1, i])
        T = np.asarray(it["T"], float)
        im = ax.imshow(T, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(3), ticks, fontsize=8); ax.set_yticks(range(3), ticks, fontsize=8)
        ax.set_xlabel("photon B"); ax.grid(False)
        if i == 0:
            ax.set_ylabel("photon A")
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{T[a, b]:+.2f}", ha="center", va="center", fontsize=9, color=_ink_on(T[a, b], 1))
        ax.set_title(f"T_ij = ⟨σ_i ⊗ σ_j⟩ — {it['label']}", fontsize=9)
    fig.colorbar(im, ax=fig.axes[-n:], shrink=0.8)
    return save(fig, path)


# ------------------------------------------------------------------ parameter sweeps (small multiples)
def sweep_figure(x: np.ndarray, panels: list[dict], series_labels: list[str], xlabel: str,
                 title: str, path: str, dashed: list[int] | None = None) -> str:
    """panels: [{'title', 'ys': [array per series], 'reference': (level, text) or None,
    'marks': [(x, y), ...]}]. One panel per quantity, one colour per state (fixed order)."""
    n = len(panels)
    ncols = 1 if n == 1 else 2 if n in (2, 4) else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.1 * nrows), squeeze=False, sharex=True)
    for k, p in enumerate(panels):
        ax = axes[k // ncols][k % ncols]
        for i, y in enumerate(p["ys"]):              # a series identical to an earlier one: dashed on top
            ax.plot(x, y, color=SERIES[i], label=series_labels[i],
                    ls=(0, (5, 4)) if dashed and i in dashed else "-", zorder=3 + i)
        if p.get("reference"):
            ax.axhline(p["reference"][0], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        for i, (mx, my) in p.get("marks", []):
            _dot(ax, mx, my, SERIES[i])
        sub = f"\ndashed line: {p['reference'][1]}" if p.get("reference") else ""
        ax.set_title(p["title"] + sub, fontsize=10, loc="left")
        lo = min(float(np.nanmin(y)) for y in p["ys"])
        hi = max(float(np.nanmax(y)) for y in p["ys"])
        if p.get("reference"):
            lo, hi = min(lo, p["reference"][0]), max(hi, p["reference"][0])
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + pad)
        if k // ncols == nrows - 1 or k + ncols >= n:
            ax.set_xlabel(xlabel)
            ax.tick_params(labelbottom=True)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    if len(series_labels) >= 2:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0),
                   fontsize=9)
        fig.suptitle(title, y=1.06, fontsize=11)
    else:                                            # one state: name it in the title (there is no legend)
        name = series_labels[0] if series_labels else ""
        fig.suptitle(f"{title} — {name}" if name and name not in title else title, fontsize=11)
    fig.tight_layout()
    return save(fig, path)


# ------------------------------------------------------------------ polarization-correlation fringes
def fringe_figure(theta_b: np.ndarray, curves: list[tuple[str, np.ndarray]], title: str, path: str) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, (label, p) in enumerate(curves):
        ax.plot(theta_b, p, color=SERIES[i], label=label)
    ax.set_xlabel("angle of photon B's linear polarizer (degrees from H)")
    ax.set_ylabel("coincidence probability")
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_xlim(0, 180)
    ax.set_ylim(0, max(0.55, max(float(p.max()) for _, p in curves) * 1.08))
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    return save(fig, path)


# ------------------------------------------------------------------ tomography diagnostics
def tomography_fit_figure(labels: list[str], n: np.ndarray, mu: np.ndarray, residuals: np.ndarray,
                          histograms: list[dict], title: str, path: str) -> str:
    """Measured vs fitted counts, residuals in σ, bootstrap histograms (with the bound and the
    measured value marked)."""
    fig = plt.figure(figsize=(11, 9.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])
    idx = np.arange(len(labels))
    ax = fig.add_subplot(gs[0, :])
    ax.bar(idx, n, width=0.62, color=SERIES[0], label="measured counts")
    ax.plot(idx, mu, "o", ms=6, color=SERIES[1], mec=SURFACE, mew=1.5, label="maximum-likelihood fit")
    ax.set_xticks(idx, labels, rotation=90, fontsize=7)
    ax.set_ylabel("coincidences")
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, :])
    colors = [CRITICAL if abs(r) > 3 else SERIES[0] for r in residuals]
    ax.bar(idx, residuals, width=0.62, color=colors)
    for lv in (-3, -2, 2, 3):
        ax.axhline(lv, color=MUTED, lw=1, ls=(0, (4, 3)))
    for i, r in enumerate(residuals):
        if abs(r) > 3:
            ax.annotate(f"{r:+.1f}σ", (i, r), xytext=(0, 3 if r > 0 else -11), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)
    ax.set_xticks(idx, labels, rotation=90, fontsize=7)
    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_ylabel("(measured − fit) / √fit")
    ax.set_title("residuals in standard deviations (dashed: ±2σ, ±3σ; red: beyond 3σ)", fontsize=9, loc="left")

    for k, h in enumerate(histograms[:2]):
        ax = fig.add_subplot(gs[2, k])
        ax.hist(h["samples"], bins=20, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
        ax.axvline(h["value"], color=INK, lw=1.5)
        ax.annotate("measured", xy=(h["value"], 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", fontsize=8, color=INK)
        if h.get("bound") is not None:
            lo, hi = ax.get_xlim()
            if lo - 0.5 * (hi - lo) <= h["bound"] <= hi + 0.5 * (hi - lo):
                ax.axvline(h["bound"], color=MUTED, lw=1, ls=(0, (4, 3)))
                ax.annotate(h["bound_text"], xy=(h["bound"], 0.92), xycoords=("data", "axes fraction"),
                            xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(f"{h['name']}: {len(h['samples'])} resamplings, σ = {np.std(h['samples'], ddof=1):.2g}",
                     fontsize=9, loc="left")
        ax.set_ylabel("resamplings")
        ax.set_xlabel(h["name"])
    fig.tight_layout()
    return save(fig, path)


# ------------------------------------------------------------------ data files: plotting + fits
FIT_MODELS = {
    "none": "no fit",
    "linear": "y = a·x + b",
    "exponential": "y = A·exp(−x/τ) + C",
    "sine": "y = C + A·cos(2π·x/P + φ) (period P fitted unless given)",
    "fringe": "y = C + A·cos(2π·(x − x0)/P) with fixed period P (default 180, polarizer angle in "
              "degrees; use 90 for a half-wave-plate angle); visibility V = |A|/C",
    "gaussian": "y = A·exp(−(x − μ)²/(2s²)) + C",
}


def _fit_result(names, popt, pcov, x, y, sigma, model, absolute_sigma):
    from scipy.stats import chi2 as chi2_dist
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if pcov is not None else np.full(len(popt), np.nan)
    yfit = model(x, *popt)
    dof = max(len(x) - len(popt), 1)
    chi2 = float(np.sum(((y - yfit) / sigma) ** 2))
    out = {"parameters": {k: {"value": float(v), "error": float(e)} for k, v, e in zip(names, popt, perr)},
           "points": int(len(x)), "dof": int(dof)}
    if absolute_sigma:
        out["chi2_per_dof"] = round(chi2 / dof, 3)
        out["p_value"] = float(f"{chi2_dist.sf(chi2, dof):.2g}")
    else:
        out["rms_residual"] = float(np.sqrt(np.mean((y - yfit) ** 2)))
    return out


def fit_data(x: np.ndarray, y: np.ndarray, kind: str, sigma: np.ndarray | None = None,
             period: float | None = None) -> dict:
    """Least-squares fit of one of FIT_MODELS. Returns parameters with 1σ errors, fit quality and
    a function for the fitted curve. sigma = y error bars (None: unweighted, errors from scatter)."""
    from scipy.optimize import curve_fit
    x, y = np.asarray(x, float), np.asarray(y, float)
    absolute = sigma is not None
    s = np.asarray(sigma, float) if absolute else np.ones_like(y)
    need = {"linear": 3, "exponential": 4, "sine": 5, "fringe": 4, "gaussian": 5}[kind]
    if len(x) < need:
        raise ValueError(f"A {kind} fit needs at least {need} data points (got {len(x)}).")

    if kind == "linear":
        model = lambda t, a, b: a * t + b                                   # noqa: E731
        popt, pcov = curve_fit(model, x, y, sigma=s, absolute_sigma=absolute)
        res = _fit_result(["a (slope)", "b (intercept)"], popt, pcov, x, y, s, model, absolute)
    elif kind == "exponential":
        model = lambda t, A, tau, C: A * np.exp(-t / tau) + C               # noqa: E731
        C0 = float(y[np.argmax(x)])
        A0 = float(y[np.argmin(x)] - C0) or 1.0
        target = C0 + A0 / np.e
        tau0 = float(x[np.argmin(np.abs(y - target))] - x.min()) or float(np.ptp(x) / 2) or 1.0
        popt, pcov = curve_fit(model, x, y, p0=[A0, tau0, C0], sigma=s, absolute_sigma=absolute, maxfev=20000)
        res = _fit_result(["A", "tau", "C"], popt, pcov, x, y, s, model, absolute)
    elif kind in ("fringe", "sine") and (kind == "fringe" or period):
        P = float(period or 180.0)
        # linear in (C, a, b): y = C + a cos(2πx/P) + b sin(2πx/P)  ->  exact weighted least squares
        M = np.stack([np.ones_like(x), np.cos(2 * np.pi * x / P), np.sin(2 * np.pi * x / P)], axis=1)
        W = 1 / s
        coef, *_ = np.linalg.lstsq(M * W[:, None], y * W, rcond=None)
        cov = np.linalg.inv((M * W[:, None]).T @ (M * W[:, None]))
        if not absolute:
            dof = max(len(x) - 3, 1)
            cov = cov * float(np.sum((y - M @ coef) ** 2) / dof)
        C, a, b = coef
        A = float(np.hypot(a, b))
        x0 = float((np.degrees(np.arctan2(b, a)) / 360.0 * P) % P)
        # error propagation for A, x0, V
        J_A = np.array([0, a / A, b / A]) if A > 0 else np.zeros(3)
        J_x0 = np.array([0, -b, a]) / (A ** 2) * P / (2 * np.pi) if A > 0 else np.zeros(3)
        V = A / C if C != 0 else np.nan
        J_V = np.array([-A / C ** 2, a / (A * C), b / (A * C)]) if A > 0 and C != 0 else np.zeros(3)
        err = lambda J: float(np.sqrt(max(J @ cov @ J, 0)))                 # noqa: E731
        model = lambda t, C=C, A=A, x0=x0: C + A * np.cos(2 * np.pi * (t - x0) / P)   # noqa: E731
        res = _fit_result(["C (offset)", "A (amplitude)", "x0 (position of maximum)"],
                          np.array([C, A, x0]), np.diag([cov[0, 0], err(J_A) ** 2, err(J_x0) ** 2]),
                          x, y, s, model, absolute)
        res["dof"] = int(max(len(x) - 3, 1))
        res["period_fixed"] = P
        res["visibility"] = {"value": float(V), "error": err(J_V)}
        if kind == "sine":
            res["period"] = {"value": P, "error": 0.0}
    elif kind == "sine":
        model = lambda t, C, A, P, phi: C + A * np.cos(2 * np.pi * t / P + phi)   # noqa: E731
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        grid = np.linspace(xs.min(), xs.max(), 512)
        yi = np.interp(grid, xs, ys) - ys.mean()
        freqs = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
        k = int(np.argmax(np.abs(np.fft.rfft(yi))[1:]) + 1)
        P0 = 1 / freqs[k]
        p0 = [float(ys.mean()), float(np.ptp(ys) / 2), P0, 0.0]
        best = None
        for phi0 in np.linspace(-np.pi, np.pi, 8, endpoint=False):
            try:
                p0[3] = phi0
                popt, pcov = curve_fit(model, x, y, p0=p0, sigma=s, absolute_sigma=absolute, maxfev=20000)
                r = float(np.sum(((y - model(x, *popt)) / s) ** 2))
                if best is None or r < best[0]:
                    best = (r, popt, pcov)
            except RuntimeError:
                continue
        if best is None:
            raise ValueError("The sine fit did not converge; give the period (period=...) if you know it.")
        _, popt, pcov = best
        if popt[1] < 0:                                   # keep the amplitude positive
            popt[1], popt[3] = -popt[1], popt[3] + np.pi
        popt[3] = (popt[3] + np.pi) % (2 * np.pi) - np.pi
        res = _fit_result(["C (offset)", "A (amplitude)", "P (period)", "phi (phase, rad)"], popt, pcov,
                          x, y, s, model, absolute)
        res["period"] = res["parameters"]["P (period)"]
        Cv, Av = popt[0], popt[1]
        if Cv:
            J = np.array([-Av / Cv ** 2, 1 / Cv, 0, 0])
            verr = float(np.sqrt(max(J @ pcov @ J, 0)))
        res["visibility"] = {"value": float(Av / Cv) if Cv else float("nan"),
                             "error": verr if Cv else float("nan")}
    elif kind == "gaussian":
        model = lambda t, A, mu, sd, C: A * np.exp(-(t - mu) ** 2 / (2 * sd ** 2)) + C   # noqa: E731
        C0 = float(np.min(y))
        i = int(np.argmax(y))
        p0 = [float(y[i] - C0), float(x[i]), float(np.ptp(x) / 6) or 1.0, C0]
        popt, pcov = curve_fit(model, x, y, p0=p0, sigma=s, absolute_sigma=absolute, maxfev=20000)
        popt[2] = abs(popt[2])
        res = _fit_result(["A (height)", "mu (centre)", "s (width σ)", "C (offset)"], popt, pcov,
                          x, y, s, model, absolute)
        res["fwhm"] = {"value": float(2 * np.sqrt(2 * np.log(2)) * popt[2]),
                       "error": float(2 * np.sqrt(2 * np.log(2)) * np.sqrt(max(pcov[2, 2], 0)))}
    else:
        raise ValueError(f"fit must be one of {list(FIT_MODELS)}")
    res["model"] = FIT_MODELS[kind]
    bound = locals().get("popt") if not (kind == "fringe" or (kind == "sine" and period)) else None
    res["curve"] = (lambda t, m=model, pp=tuple(bound): m(t, *pp)) if bound is not None else model
    return res


def data_figure(series: list[dict], xlabel: str, ylabel: str, title: str, log_y: bool, path: str,
                caption: str = "", angle_axis: bool = False) -> str:
    """series: [{'label', 'x', 'y', 'yerr' or None, 'fit_x', 'fit_y' or None}] (<= 4)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for i, s in enumerate(series):
        c = SERIES[i]
        if s.get("yerr") is not None:
            ax.errorbar(s["x"], s["y"], yerr=s["yerr"], fmt="o", ms=6, color=c, mec=SURFACE, mew=1.5,
                        elinewidth=1.2, capsize=0, label=s["label"], zorder=4)
        else:
            ax.plot(s["x"], s["y"], "o", ms=6, color=c, mec=SURFACE, mew=1.5, label=s["label"], zorder=4)
        if s.get("fit_y") is not None:
            ax.plot(s["fit_x"], s["fit_y"], color=c, lw=2, alpha=0.9, label=f"fit: {s['label']}", zorder=3)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if angle_axis:
        lo = min(float(np.min(s["x"])) for s in series)
        hi = max(float(np.max(s["x"])) for s in series)
        step = 45 if hi - lo > 120 else 15
        ax.set_xticks(np.arange(np.floor(lo / step) * step, hi + 0.01, step))
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=11, loc="left")
    if len(series) >= 2 or any(s.get("fit_y") is not None for s in series):
        ax.legend(fontsize=8, loc="best")
    if caption:
        fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=8, color=INK2, wrap=True)
    return save(fig, path)


# ------------------------------------------------------------------ density-matrix elements versus time
def matrix_elements_figure(times: np.ndarray, traj: np.ndarray, basis: list[str], title: str, path: str) -> str:
    """Re and Im of every element ρ_ij (i <= j) that is not zero at some time; the lower triangle is
    the complex conjugate, so it is not drawn. Left: populations and Re of coherences; right: Im."""
    d = len(basis)
    fig, (a_re, a_im) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    k = 0
    for i in range(d):
        for j in range(i, d):
            re, im = traj[:, i, j].real, traj[:, i, j].imag
            name = f"ρ[{basis[i]},{basis[j]}]"
            c = SERIES[k % len(SERIES)]
            lw = max(1.0, 3.6 - 0.45 * k)         # earlier curves wider: equal curves (ρ[HH,HH] = ρ[VV,VV]) both show
            if np.max(np.abs(re)) > 1e-3:
                a_re.plot(times, re, color=c, label=name, ls="-" if i == j else "--", lw=lw)
            if i != j and np.max(np.abs(im)) > 1e-3:
                a_im.plot(times, im, color=c, label=name, ls="--", lw=lw)
            k += 1
    a_re.set_title("Re ρ_ij (solid: populations, dashed: coherences)", fontsize=10)
    a_im.set_title("Im ρ_ij (coherences)", fontsize=10)
    for ax in (a_re, a_im):
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xlabel("time t")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="best")
    if not a_im.get_legend_handles_labels()[0]:
        a_im.text(0.5, 0.5, "all imaginary parts stay 0", ha="center", va="center", transform=a_im.transAxes,
                  color=INK2)
    fig.suptitle(title, fontsize=11)
    return save(fig, path)


def bloch_single_figure(items: list[dict], path: str) -> str:
    """One-photon states on one Bloch (Poincaré) sphere: items [{'label', 'vec' (x, y, z) = (S2, S3, S1)}]."""
    fig = plt.figure(figsize=(5.6, 5.4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _bloch_sphere(ax, "one photon: Bloch (Poincaré) sphere")
    for i, it in enumerate(items):
        vec = np.asarray(it["vec"], float)
        c = SERIES[i]
        if np.linalg.norm(vec) > 1e-3:
            ax.quiver(0, 0, 0, *vec, color=c, lw=2, arrow_length_ratio=0.12)
        ax.scatter(*vec, s=55, color=c, edgecolors=SURFACE, linewidths=2, depthshade=False,
                   label=f"{it['label']}: |S| = {np.linalg.norm(vec):.3f}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    return save(fig, path)
