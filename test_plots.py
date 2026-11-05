"""
test_plots.py - the plot engine: every number a plot tool returns is checked against an exact
result, and every figure is actually written.
Run:  pytest -q test_plots.py
"""
import json
import os

import numpy as np
import pytest
from langchain_core.messages import AIMessage, ToolMessage

import plots
import qi_tools as qt
from agent_tools import (CURRENT_QUESTION, DESC, STORE, TOMO, _save, create_pure_state, create_state,
                         plot_bloch_and_correlations, plot_data_file, plot_density_matrix, plot_fringes,
                         plot_measures_vs_noise, plot_sweep, plot_tomography_fit, reconstruct_state_from_counts)

LAB_CSV = os.path.abspath("data/state_tomo_data.csv")


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    STORE.clear(); DESC.clear(); TOMO.clear()
    CURRENT_QUESTION["text"] = ""
    yield


def sweep(**kw):
    out = plot_sweep.invoke(kw)
    assert os.path.isfile(out["figure"])
    return out


def measures(out, state=None):
    res = out["results"]
    return (res[state] if state else next(iter(res.values())))["measures"]


def crossing(m):
    return [c["at"] for c in m["crossings"]]


# ------------------------------------------------------------------ plot_sweep: state families
def test_werner_sweep_finds_every_threshold():
    m = measures(sweep(sweep="werner_p", quantities=["concurrence", "chsh_max", "steering_f3",
                                                      "teleportation_fidelity", "key_fraction", "purity"]))
    assert crossing(m["concurrence"]) == pytest.approx([1 / 3], abs=1e-4)
    assert crossing(m["chsh_max"]) == pytest.approx([1 / np.sqrt(2)], abs=1e-4)
    assert crossing(m["steering_f3"]) == pytest.approx([1 / np.sqrt(3)], abs=1e-4)
    assert crossing(m["teleportation_fidelity"]) == pytest.approx([1 / 3], abs=1e-4)
    q = 0.110028                                     # 1 - 2h(q) = 0
    assert crossing(m["key_fraction"]) == pytest.approx([1 - 2 * q], abs=1e-4)
    assert all(m[k]["trend"].startswith("increases") for k in m)
    assert m["purity"]["at_start"] == 0.25 and m["purity"]["at_end"] == 1.0


def test_pure_angle_sweep_concurrence_is_sin_2theta():
    out = sweep(sweep="pure_angle", quantities=["concurrence", "entropy"])
    m = measures(out)
    assert m["concurrence"]["max"] == {"value": 1.0, "at": round(np.pi / 4, 4)}
    assert m["concurrence"]["trend"].startswith("not monotonic")
    for x, v in next(iter(out["results"].values()))["samples"].items():
        assert v["concurrence"] == pytest.approx(abs(np.sin(2 * float(x))), abs=1e-3)


# ------------------------------------------------------------------ plot_sweep: operations on stored states
def test_noise_sweep_matches_the_noise_plot_tool():
    create_state.invoke({"kind": "phi+"})
    m = measures(sweep(sweep="noise", handles=["rho_1"], channel="depolarizing", quantities=["concurrence", "chsh"]))
    assert crossing(m["concurrence"]) == pytest.approx([0.4226], abs=1e-4)
    assert crossing(m["chsh_max"]) == pytest.approx([0.1591], abs=1e-4)
    ref = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing"})
    assert ref["features"]["entanglement_vanishes_at"] == pytest.approx(0.4226, abs=1e-3)


def test_phase_shift_sweep_of_phi_plus_and_of_the_lab_state():
    create_state.invoke({"kind": "phi+"})
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    out = sweep(sweep="phase_shift", handles=["rho_1", rec["handle"]], quantities=["fidelity", "concurrence",
                                                                                      "key_fraction"],
                reference="phi+", n_points=361)
    phi = measures(out, "rho_1")
    assert phi["fidelity"]["min"] == {"value": 0.0, "at": 180.0}            # cos²(φ/2)
    for x, v in out["results"]["rho_1"]["samples"].items():
        assert v["fidelity"] == pytest.approx(np.cos(np.deg2rad(float(x)) / 2) ** 2, abs=1e-4)
    lab = measures(out, rec["handle"])
    assert lab["fidelity"]["max"]["value"] == pytest.approx(0.8527, abs=2e-4)
    assert lab["fidelity"]["max"]["at"] == pytest.approx(108.28, abs=1.0)
    assert lab["concurrence"]["trend"] == "constant (0.7213)"              # a local unitary
    assert lab["key_fraction"]["max"]["at"] == [108.0, 288.0]              # Bob may flip all bits
    assert lab["key_fraction"]["min"]["at"][0].startswith("0 to")          # a flat stretch at 0


def test_waveplate_and_visibility_sweeps():
    create_state.invoke({"kind": "phi+"})
    m = measures(sweep(sweep="waveplate_angle", handles="rho_1", plate="QWP", quantities=["concurrence", "purity"]))
    assert m["concurrence"]["trend"] == "constant (1.0000)" and m["purity"]["trend"] == "constant (1.0000)"
    m = measures(sweep(sweep="visibility", handles=["rho_1"], quantities=["concurrence", "chsh_max"]))
    assert crossing(m["concurrence"]) == pytest.approx([1 / 3], abs=1e-4)   # = Werner on phi+
    assert crossing(m["chsh_max"]) == pytest.approx([1 / np.sqrt(2)], abs=1e-4)


def test_sweep_input_errors_are_explained():
    create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="reference="):
        plot_sweep.invoke({"sweep": "werner_p", "quantities": ["fidelity"]})
    with pytest.raises(Exception, match="handles="):
        plot_sweep.invoke({"sweep": "visibility"})
    with pytest.raises(Exception, match="At most 6"):
        plot_sweep.invoke({"sweep": "werner_p", "quantities": ["concurrence", "negativity", "purity", "entropy",
                                                               "chsh_max", "discord", "coherence_l1"]})
    with pytest.raises(Exception, match="within \\[0, 1\\]"):
        plot_sweep.invoke({"sweep": "werner_p", "start": 0, "stop": 2})
    with pytest.raises(Exception, match="evolve_state"):
        plot_sweep.invoke({"sweep": "time"})
    with pytest.raises(Exception, match="channel must be"):
        plot_sweep.invoke({"sweep": "noise", "handles": ["rho_1"], "channel": "dephasing_xyz"})
    out = sweep(sweep="werner_p", quantities=["discord"], n_points=401)
    assert out["points"] == 41                                               # discord is slow: capped


def test_sweeps_with_different_settings_get_different_files():
    a = sweep(sweep="werner_p", quantities=["concurrence"])
    b = sweep(sweep="werner_p", quantities=["concurrence"], bell="phi+")
    assert a["figure"] != b["figure"]


# ------------------------------------------------------------------ Bloch vectors and correlations
def test_bloch_vectors_and_correlation_matrix():
    create_state.invoke({"kind": "phi+"})
    CURRENT_QUESTION["text"] = "Plot the Bloch vectors of |HH>"
    create_pure_state.invoke({"amp_HH": "1"})
    CURRENT_QUESTION["text"] = ""
    out = plot_bloch_and_correlations.invoke({"handles": ["rho_1", "rho_2"]})
    assert os.path.isfile(out["figure"])
    phi, hh = out["states"]["rho_1"], out["states"]["rho_2"]
    assert phi["length_A"] == 0 and phi["length_B"] == 0
    assert phi["T"] == [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert phi["chsh_max_from_T"] == pytest.approx(2 * np.sqrt(2), abs=1e-4)
    assert hh["bloch_A"] == [0, 0, 1] and hh["bloch_B"] == [0, 0, 1] and hh["T"][2][2] == 1
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    lab = plot_bloch_and_correlations.invoke({"handles": rec["handle"]})["states"][rec["handle"]]
    assert lab["chsh_max_from_T"] == pytest.approx(2.4572, abs=1e-4)
    assert lab["steering_F3_from_T"] == pytest.approx(1.4083, abs=1e-4)


# ------------------------------------------------------------------ fringes
def test_fringe_visibilities():
    create_state.invoke({"kind": "phi+"})
    out = plot_fringes.invoke({"handle": "rho_1"})
    assert [f["visibility"] for f in out["fringes"]] == [1.0, 1.0, 1.0, 1.0]
    assert [f["maximum_at_deg"] for f in out["fringes"]] == [0.0, 45.0, 90.0, 135.0]   # phi+: θ_B = θ_A
    create_state.invoke({"kind": "werner", "p": 0.6})
    w = plot_fringes.invoke({"handle": "rho_2", "alice_angles": [0, 45]})
    assert [f["visibility"] for f in w["fringes"]] == [0.6, 0.6]
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    lab = plot_fringes.invoke({"handle": rec["handle"]})["fringes"]
    rho = STORE[rec["handle"]]

    def p(ta, tb):
        a, b = np.deg2rad(ta), np.deg2rad(tb)
        va, vb = np.array([np.cos(a), np.sin(a)]), np.array([np.cos(b), np.sin(b)])
        return float(np.real(np.kron(va, vb) @ rho @ np.kron(va, vb)))
    dense = np.array([p(45, t) for t in np.linspace(0, 180, 3601)])
    assert lab[1]["visibility"] == pytest.approx((dense.max() - dense.min()) / (dense.max() + dense.min()), abs=1e-4)
    assert lab[0]["visibility"] > 0.95 and lab[1]["visibility"] < 0.3        # the −108° phase spoils D/A


# ------------------------------------------------------------------ tomography diagnostics
def test_tomography_fit_diagnostics_of_the_lab_data():
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV})
    out = plot_tomography_fit.invoke({"handle": rec["handle"]})
    assert os.path.isfile(out["figure"])
    assert out["chi2_per_dof"] == 2.94 and out["settings_beyond_3_sigma"] == ["AD (+3.1σ)"]
    assert out["settings_beyond_2_sigma"][:3] == ["AD (+3.1σ)", "VA (+2.8σ)", "HD (-2.4σ)"]
    assert out["bootstrap"]["concurrence"]["sigma"] == rec["uncertainty_1sigma"]["concurrence"]
    create_state.invoke({"kind": "phi+"})
    with pytest.raises(Exception, match="not reconstructed from counts"):
        plot_tomography_fit.invoke({"handle": "rho_2"})


# ------------------------------------------------------------------ the user's own data files
def write_csv(name, header, columns):
    with open(f"data/{name}", "w") as fh:
        fh.write(",".join(header) + "\n")
        for row in zip(*columns):
            fh.write(",".join(str(v) for v in row) + "\n")
    return f"data/{name}"


def test_fringe_fit_recovers_the_visibility():
    rng = np.random.default_rng(3)
    th = np.arange(0, 181, 10.0)
    path = write_csv("scan.csv", ["angle_deg", "counts"],
                     [th, rng.poisson(500 * (1 + 0.85 * np.cos(2 * np.deg2rad(th - 20))))])
    f = plot_data_file.invoke({"csv_path": path, "x_column": "angle_deg", "y_columns": "counts",
                               "fit": "fringe"})["fits"]["counts"]
    V = f["visibility"]
    assert abs(V["value"] - 0.85) < 3 * V["error"]
    x0 = f["parameters"]["x0 (position of maximum)"]
    assert abs(x0["value"] - 20) < 3 * x0["error"]
    assert f["fit_verdict"] == "acceptable"
    # a half-wave-plate scan has period 90°
    path = write_csv("hwp.csv", ["hwp_angle", "counts"],
                     [th / 2, rng.poisson(400 * (1 + 0.9 * np.cos(4 * np.deg2rad(th / 2 - 10))))])
    g = plot_data_file.invoke({"csv_path": path, "x_column": "hwp_angle", "y_columns": ["counts"], "fit": "fringe",
                               "period": 90})["fits"]["counts"]
    assert abs(g["visibility"]["value"] - 0.9) < 3 * g["visibility"]["error"]


def test_exponential_gaussian_linear_and_sine_fits():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 10, 40)
    path = write_csv("decay.csv", ["time_us", "counts"], [t, rng.poisson(800 * np.exp(-t / 2.5) + 50)])
    tau = plot_data_file.invoke({"csv_path": path, "x_column": "time_us", "y_columns": ["counts"],
                                 "fit": "exponential"})["fits"]["counts"]["parameters"]["tau"]
    assert abs(tau["value"] - 2.5) < 3 * tau["error"]
    x = np.linspace(-5, 5, 41)
    path = write_csv("peak.csv", ["delay_ns", "coincidences"],
                     [x, rng.poisson(300 * np.exp(-(x - 0.7) ** 2 / (2 * 1.3 ** 2)) + 20)])
    g = plot_data_file.invoke({"csv_path": path, "x_column": "delay_ns", "y_columns": ["coincidences"],
                               "fit": "gaussian"})["fits"]["coincidences"]
    assert abs(g["parameters"]["mu (centre)"]["value"] - 0.7) < 3 * g["parameters"]["mu (centre)"]["error"]
    assert g["fwhm"]["value"] == pytest.approx(2.3548 * g["parameters"]["s (width σ)"]["value"], rel=1e-3)
    path = write_csv("line.csv", ["x", "y"], [np.arange(6.0), 2 * np.arange(6.0) + 1])
    lin = plot_data_file.invoke({"csv_path": path, "x_column": "x", "y_columns": ["y"], "fit": "linear",
                                 "y_error": "none"})["fits"]["y"]["parameters"]
    assert lin["a (slope)"]["value"] == pytest.approx(2) and lin["b (intercept)"]["value"] == pytest.approx(1)
    xs = np.linspace(0, 30, 61)
    path = write_csv("osc.csv", ["t", "signal"], [xs, rng.poisson(300 + 200 * np.cos(2 * np.pi * xs / 7.0 + 0.4))])
    s = plot_data_file.invoke({"csv_path": path, "x_column": "t", "y_columns": ["signal"], "fit": "sine"})["fits"]["signal"]
    assert abs(s["period"]["value"] - 7.0) < 3 * s["period"]["error"]


def test_data_file_guards():
    path = write_csv("bad.csv", ["x", "y", "label"], [[0, 1, 2, 3], [5, -1, 4, "n/a"], ["a", "b", "c", "d"]])
    with pytest.raises(Exception, match="negative values"):
        plot_data_file.invoke({"csv_path": path, "x_column": "x", "y_columns": ["y"]})
    out = plot_data_file.invoke({"csv_path": path, "x_column": "x", "y_columns": ["y"], "y_error": "none"})
    assert out["notes"] == ["y: 1 row(s) without numbers were skipped"]
    with pytest.raises(Exception, match="not found. Numeric columns"):
        plot_data_file.invoke({"csv_path": path, "x_column": "angle", "y_columns": ["y"]})
    rng = np.random.default_rng(1)
    t = np.linspace(0, 10, 30)
    path = write_csv("decay2.csv", ["t", "counts"], [t, rng.poisson(800 * np.exp(-t / 2.5) + 50)])
    f = plot_data_file.invoke({"csv_path": path, "x_column": "t", "y_columns": ["counts"], "fit": "gaussian"})
    assert f["fits"]["counts"]["fit_verdict"].startswith(("NOT meaningful", "poor", "failed"))   # wrong model is flagged


# ------------------------------------------------------------------ density matrices
def test_density_matrix_styles_and_comparison():
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    h = rec["handle"]
    for style in ("bars3d", "heatmap", "magnitude_phase"):
        out = plot_density_matrix.invoke({"handle": h, "style": style, "compare_with": "phi+"})
        assert os.path.isfile(out["figure"]) and out["fidelity_between_plotted_states"] == 0.3788
    assert plot_density_matrix.invoke({"handle": h})["figure"] == f"figures/{h}_city.png"   # unchanged default
    with pytest.raises(Exception, match="style must be"):
        plot_density_matrix.invoke({"handle": h, "style": "pie"})


# ------------------------------------------------------------------ fitting helper and panels
def test_fit_data_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 5"):
        plots.fit_data(np.arange(3.0), np.arange(3.0), "gaussian")


def test_every_plot_result_gets_an_exact_table():
    from display import result_panels, summarize_steps
    create_state.invoke({"kind": "phi+"})
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    path = write_csv("scan.csv", ["angle_deg", "counts"], [np.arange(0, 181, 20.0), [900, 820, 600, 350, 120, 60,
                                                                                         200, 450, 700, 880]])
    calls = [("plot_sweep", plot_sweep, {"sweep": "werner_p", "quantities": ["concurrence", "chsh_max"]}),
             ("plot_fringes", plot_fringes, {"handle": "rho_1"}),
             ("plot_bloch_and_correlations", plot_bloch_and_correlations, {"handles": ["rho_1"]}),
             ("plot_tomography_fit", plot_tomography_fit, {"handle": rec["handle"]}),
             ("plot_data_file", plot_data_file, {"csv_path": path, "x_column": "angle_deg", "y_columns": ["counts"],
                                                 "fit": "fringe"})]
    msgs = []
    for i, (name, tool, args) in enumerate(calls):
        msgs.append(AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": str(i)}]))
        msgs.append(ToolMessage(content=json.dumps(tool.invoke(args), ensure_ascii=False), name=name,
                                tool_call_id=str(i)))
    panels = dict(result_panels(msgs))
    assert set(panels) == {"Sweep", "Fringes", "Bloch vectors and correlations", "Tomography fit", "Data fit"}
    assert "0.3333 (rises above the bound)" in panels["Sweep"]
    assert "| 0° (H) | 1.0000 |" in panels["Fringes"]
    assert "AD (+3.1σ)" in panels["Tomography fit"]
    assert "visibility" in panels["Data fit"]
    assert all(s[1].startswith("Saved figure ") and s[1].endswith(".png") and "/" not in s[1]
               for s in summarize_steps(msgs))                 # the file name; run.py prints the full path


# ------------------------------------------------------------------ answer checks know about sweeps
def test_answer_checks_read_sweep_samples():
    from graph import _last_evolution_series, remove_false_claims, required_tools_missing
    create_state.invoke({"kind": "phi+"})
    out = plot_sweep.invoke({"sweep": "noise", "handles": ["rho_1"], "channel": "bit_flip",
                             "quantities": ["concurrence", "chsh_max", "purity"]})
    msgs = [ToolMessage(content=json.dumps(out), name="plot_sweep", tool_call_id="s")]
    assert len(_last_evolution_series(msgs)["purity"]) == 11
    text = "- The CHSH value never violates the local bound at intermediate noise.\n- Purity recovers at p = 1."
    fixed, why = remove_false_claims(text, msgs)
    assert "never violates" not in fixed and why
    turn = [AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "phi+"}, "id": "a"}])]
    assert required_tools_missing("Create phi+ and plot its density matrix", turn, "x") == \
        [("a plot", "the matching plot tool")]


def test_figures_are_listed_in_the_order_the_tools_ran():
    from graph import verifier
    msgs = [AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}, {"name": "b", "args": {}, "id": "2"}]),
            ToolMessage(content=json.dumps({"figure": "figures/first.png"}), name="plot_sweep", tool_call_id="1"),
            ToolMessage(content=json.dumps({"figure": "figures/second.png"}), name="plot_fringes", tool_call_id="2")]
    assert verifier({"messages": msgs})["figures"] == ["figures/first.png", "figures/second.png"]


def test_benchmark_values_for_plots():
    spec = {t["id"]: t for t in json.load(open(os.path.join(os.path.dirname(LAB_CSV), "..",
                                                            "benchmark_tasks.json"), encoding="utf-8"))["tasks"]}
    m = measures(sweep(sweep="werner_p", quantities=["concurrence", "chsh_max", "steering_f3"]))
    got = [crossing(m["concurrence"])[0], crossing(m["chsh_max"])[0], crossing(m["steering_f3"])[0]]
    assert got == pytest.approx([n["value"] for n in spec["T29"]["numbers"]], abs=5e-5)
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    f = plot_fringes.invoke({"handle": rec["handle"]})["fringes"]
    assert [f[0]["visibility"], f[1]["visibility"]] == [n["value"] for n in spec["T30"]["numbers"]]


def test_touching_the_bound_at_an_end_point_is_not_a_crossing():
    m = measures(sweep(sweep="pure_angle", quantities=["concurrence"]))
    assert m["concurrence"]["crossings"] == "none in this range"


def test_graph_end_to_end_with_a_sweep():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from benchmark import run_task, score_task
    from graph import build_graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    spec = json.load(open(os.path.join(os.path.dirname(LAB_CSV), "..", "benchmark_tasks.json"), encoding="utf-8"))
    task = {t["id"]: t for t in spec["tasks"]}["T29"]
    script = [AIMessage(content="", tool_calls=[{"name": "plot_sweep", "id": "a", "args": {
                  "sweep": "werner_p", "quantities": ["concurrence", "chsh_max", "steering_f3"]}}]),
              AIMessage(content="Entangled above **p = 0.3333**, steerable (F3 > 1) above p = 0.5774 and CHSH "
                                "is violated above p = 0.7071.")]
    run = run_task(build_graph(Scripted(messages=iter(script))), task)
    checks = score_task(task, run["answer"], run["tables"], run["tool_names"], run["states_created"],
                        spec["global_forbidden"])
    assert run["error"] is None and all(c["ok"] for c in checks), checks
    assert "Crosses its bound at" in run["tables"]


# ------------------------------------------------------------------ 2026-10-28 run (G1-G8)
def test_comparison_claims_are_checked_against_the_measures():
    from graph import remove_false_claims
    from agent_tools import compute_measures
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    create_state.invoke({"kind": "phi+"})
    msgs = [ToolMessage(content=json.dumps(compute_measures.invoke({"handle": h})), name="compute_measures",
                        tool_call_id=h) for h in (rec["handle"], "rho_2")]
    text = ("- The reconstructed state rho_1 is mixed (purity < 1) and less entangled than the ideal phi+.\n"
            "- Coherence and mutual information are lower than for phi+, reflecting noise and imperfections.")
    out, why = remove_false_claims(text, msgs)
    assert "less entangled" in out and "Coherence and mutual" not in out
    assert "coherence l1 of rho_1 is 1.0397 and of rho_2 (phi+) is 1.0000" in why[0]


def test_reasonable_fit_claim_after_a_poor_fit_is_removed():
    from graph import remove_false_claims
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    out = plot_tomography_fit.invoke({"handle": rec["handle"]})
    assert out["p_value"] == pytest.approx(1.1e-05, rel=0.1) and out["fit_verdict"].startswith("NOT consistent")
    msgs = [ToolMessage(content=json.dumps(out), name="plot_tomography_fit", tool_call_id="t")]
    text = ("The chi-squared per degree of freedom is 2.94, indicating the fit is not perfect but still "
            "reasonable.\n- AD deviates most.")
    fixed, why = remove_false_claims(text, msgs)
    assert "reasonable" not in fixed and "AD deviates most" in fixed and "NOT consistent" in why[0]


def test_location_parameters_near_zero_do_not_make_a_fit_meaningless():
    th = np.arange(0, 181, 10.0)
    rng = np.random.default_rng(2026)
    path = write_csv("hv.csv", ["angle_deg", "counts_HV"], [th, rng.poisson(600 * (1 + 0.92 * np.cos(2 * np.deg2rad(th))))])
    f = plot_data_file.invoke({"csv_path": path, "x_column": "angle_deg", "y_columns": ["counts_HV"],
                               "fit": "fringe"})["fits"]["counts_HV"]
    assert abs(f["parameters"]["x0 (position of maximum)"]["value"]) < 1.5 or f["parameters"]["x0 (position of maximum)"]["value"] > 178
    assert f["fit_verdict"] == "acceptable"


def test_bell_reference_in_a_sweep_counts_as_using_that_state_and_rows_link_to_plots():
    from graph import bell_states_missing, wrong_row_values
    turn = [AIMessage(content="", tool_calls=[{"name": "plot_sweep", "id": "s", "args": {
        "sweep": "phase_shift", "handles": ["rho_1"], "reference": "phi+"}}])]
    assert bell_states_missing("plot the fidelity with phi+", turn) == []
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    h = rec["handle"]
    dm = plot_density_matrix.invoke({"handle": h, "style": "heatmap", "compare_with": "phi+"})
    hist = [AIMessage(content="", tool_calls=[{"name": "plot_density_matrix", "id": "p",
                                               "args": {"handle": h, "compare_with": "phi+"}}]),
            ToolMessage(content=json.dumps(dm), name="plot_density_matrix", tool_call_id="p")]
    answer = f"| State | Fidelity with phi+ |\n|---|---|\n| {h} | 0.3788 |"
    assert wrong_row_values(answer, hist) == []


def test_soft_good_fit_claims_are_removed_after_a_poor_fit():
    from graph import remove_false_claims
    rec = reconstruct_state_from_counts.invoke({"csv_path": LAB_CSV, "error_bars": False})
    msgs = [ToolMessage(content=json.dumps(plot_tomography_fit.invoke({"handle": rec["handle"]})),
                        name="plot_tomography_fit", tool_call_id="t")]
    text = ("- The fit captures the data well overall but some settings show significant deviations.\n"
            "- AD setting shows the largest deviation.")
    fixed, why = remove_false_claims(text, msgs)
    assert "captures the data well" not in fixed and "AD setting" in fixed and why


def test_parallel_noise_plots_are_merged_into_one_figure():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from display import result_panels, summarize_steps
    from graph import build_graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    q = "Create phi+ and psi-, and plot their concurrence and purity versus amplitude damping on qubit B only."
    calls = [{"name": "plot_measures_vs_noise", "id": k, "args": {"start_kind": k, "channel": "amplitude_damping",
                                                                  "qubits": "B"}} for k in ("phi+", "psi-")]
    script = [AIMessage(content="", tool_calls=[{"name": "create_state", "id": "c1", "args": {"kind": "phi+"}},
                                                {"name": "create_state", "id": "c2", "args": {"kind": "psi-"}}]),
              AIMessage(content="", tool_calls=calls),
              AIMessage(content="Both lose entanglement only at γ = 1; concurrence 0.7071 at γ = 0.5.")]
    app = build_graph(Scripted(messages=iter(script)))
    out = app.invoke({"messages": [("user", q)], "retries": 0}, {"configurable": {"thread_id": "m"}})
    figs = out["figures"]
    assert len(figs) == 1 and os.path.isfile(figs[0]) and "sweep_noise" in figs[0]
    new = out["messages"][1:]
    assert [t for t, _ in result_panels(new)] == ["Sweep"]
    assert ("ok", "Drew psi- in the same figure") in summarize_steps(new)
    sweep_msg = next(m for m in new if isinstance(m, ToolMessage) and m.name == "plot_sweep")
    res = json.loads(sweep_msg.content)["results"]
    assert set(res) == {"phi+", "psi-"}
    assert res["phi+"]["samples"]["0.5"]["concurrence"] == pytest.approx(np.sqrt(0.5), abs=1e-4)


def test_identical_curves_are_reported_and_the_noise_axis_is_named():
    create_state.invoke({"kind": "phi+"}); create_state.invoke({"kind": "psi-"})
    out = sweep(sweep="noise", handles=["rho_1", "rho_2"], channel="amplitude_damping", qubits="B",
                quantities=["concurrence", "purity"])
    assert out["x_axis"] == "decay probability γ (0 = no noise)"
    assert out["identical_curves"] == ["rho_2: psi- gives exactly the same curves as rho_1: phi+ (drawn dashed on top)"]


def test_removing_a_sentence_keeps_a_whole_line_italic_intact():
    from graph import remove_false_claims
    msgs = [ToolMessage(content='{"purity": 0.5}', name="compute_measures", tool_call_id="x")]
    text = "_The state becomes maximally mixed at full damping. Here purity ends at 0.5 due to damping on one qubit._"
    out, why = remove_false_claims(text, msgs)
    assert out == "_Here purity ends at 0.5 due to damping on one qubit._" and why
