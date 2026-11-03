"""
test_workbench.py - the Workbench tab (workbench.py + workbench_ui.py) without a browser and without a
language model: every panel function returns correct numbers, errors become short messages, the
Workbench shares states with the chat of the same user only, and the Gradio wiring builds.
Run:  pytest -q test_workbench.py
"""
import math
import os
import shutil

import pytest

import agent_tools as at
import session
import workbench as wb

SID = "wbtest"


@pytest.fixture(autouse=True)
def clean():
    yield
    for sid in session.active_ids():
        session.drop(sid)


def cell(md: str, row: str, col: int = 1) -> float:
    """The number in column `col` of the markdown table row that starts with `row`."""
    for line in md.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if parts and parts[0] == row:
            return float(parts[col])
    raise AssertionError(f"row {row!r} not in:\n{md}")


# ------------------------------------------------------------------ states
def test_create_bell_state_shows_all_measures_with_description():
    md, fig, summ = wb.create(SID, "bell", kind="phi+")
    assert "Created rho_1" in md and fig is None
    assert "**All measures of rho_1** (phi+)" in md                     # description, not the handle twice
    assert cell(md, "Concurrence C") == 1.0 and cell(md, "Negativity N") == 0.5
    assert cell(md, "CHSH max (local bound 2)") == pytest.approx(2 * math.sqrt(2), abs=1e-4)
    assert summ == "the state rho_1 (phi+)"
    assert wb.state_choices(SID) == [("rho_1 — phi+", "rho_1")]


def test_create_werner_pure_and_mixture():
    md, _, _ = wb.create(SID, "bell", kind="werner", p=0.8, bell="psi-")
    assert cell(md, "Concurrence C") == pytest.approx(0.7, abs=1e-4)         # (3p - 1)/2
    md, _, _ = wb.create(SID, "pure", amps=("1", "0", "0", "1"))
    assert cell(md, "Concurrence C") == pytest.approx(1.0, abs=1e-4)        # normalized automatically
    md, _, _ = wb.create(SID, "mixture", mix_a="rho_2", mix_b="rho_1", weight=0.5)
    assert "rho_3" in md
    assert [h for _, h in wb.state_choices(SID)] == ["rho_3", "rho_2", "rho_1"]   # newest first
    table = wb.states_table(SID)
    assert table.count("| rho_") == 3


def test_create_errors_are_short_messages():
    with pytest.raises(wb.WorkbenchError, match="two states to mix"):
        wb.create(SID, "mixture", mix_a="", mix_b="")
    wb.create(SID, "bell", kind="phi+")
    with pytest.raises(wb.WorkbenchError, match="between 0 and 1"):
        wb.create(SID, "mixture", mix_a="rho_1", mix_b="rho_1", weight=1.5)
    with pytest.raises(wb.WorkbenchError) as e:
        wb.create(SID, "pure", amps=("0", "0", "0", "0"))
    assert "Ask the user" not in str(e.value) and "Traceback" not in str(e.value)
    with pytest.raises(wb.WorkbenchError, match="Choose a state first"):
        wb.noise(SID, "", "depolarizing", 0.3, "both")


# ------------------------------------------------------------------ noise, evolution, wave plates
def test_local_depolarizing_on_phi_plus_matches_werner_with_p_squared():
    wb.create(SID, "bell", kind="phi+")
    md, fig, summ = wb.noise(SID, "rho_1", "depolarizing", 0.3, "both")
    q = 0.7 ** 2                                                         # correlations shrink by (1 - p)^2
    assert cell(md, "Purity", 2) == pytest.approx((1 + 3 * q * q) / 4, abs=1e-4)
    assert cell(md, "Concurrence C", 2) == pytest.approx((3 * q - 1) / 2, abs=1e-4)
    assert cell(md, "Negativity N", 2) == pytest.approx((3 * q - 1) / 4, abs=1e-4)
    assert cell(md, "CHSH max (local bound 2)", 2) == pytest.approx(2 * math.sqrt(2) * q, abs=1e-4)
    assert cell(md, "Concurrence C", 1) == 1.0                           # before
    assert summ == "rho_2 = depolarizing with parameter 0.3 on both photons of rho_1, compared with rho_1"
    assert fig is None


def test_noise_curve_and_evolution_make_figures():
    wb.create(SID, "bell", kind="phi+")
    md, fig, summ = wb.noise_sweep(SID, "rho_1", "depolarizing", "A")      # the STORED state, 0 -> 1
    assert os.path.isfile(fig) and f"sessions/{SID}/" in fig and "photon A" in summ
    assert "rho_1" in md and "none in this range" not in md.split("Concurrence")[1].split("\n")[0]
    md, fig, summ = wb.evolve(SID, "rho_1", "dephasing", 1.0, 2.0, "both")
    assert os.path.isfile(fig) and "rho_1" in summ and "both photons" in summ
    assert "rho_2" in md                                                 # the state at t_max is stored


def test_waveplate_and_compensation():
    wb.create(SID, "bell", kind="phi+")
    md, _, summ = wb.waveplate(SID, "rho_1", "HWP", 45, "B")
    assert "rho_2" in md and "HWP at 45" in summ
    md, _, _ = wb.compensate(SID, "rho_1", "B")
    assert md


# ------------------------------------------------------------------ analysis and protocols
def test_analysis_panels():
    wb.create(SID, "bell", kind="werner", p=0.8)
    for what in ("measures", "certify", "teleportation", "qkd"):
        md, fig, summ = wb.analyze(SID, "rho_1", what)
        assert md and fig is None and "rho_1" in summ
    md, _, _ = wb.analyze(SID, "rho_1", "teleportation")
    assert "| teleportation fidelity | 0.9000 |" in md                  # (2F + 1)/3 with F = (1 + 3p)/4 = 0.85
    with pytest.raises(wb.WorkbenchError, match="Unknown analysis"):
        wb.analyze(SID, "rho_1", "nonsense")


def test_compare_key_and_repetition_code():
    wb.create(SID, "bell", kind="phi+")
    wb.create(SID, "bell", kind="psi-")
    md, _, summ = wb.compare(SID, "rho_1", "rho_2")
    assert md and summ == "the comparison of rho_1 and rho_2"
    md, _, _ = wb.key_from_qber(SID, 0.0, 0.0)
    assert "1.0000" in md
    md, _, _ = wb.repetition_code(SID, 0.1)
    assert "0.0280" in md                                                # 3p^2 - 2p^3


# ------------------------------------------------------------------ data and detectors
def test_tomography_of_the_sample_file_and_own_uploads_only():
    md, fig, summ = wb.tomography(SID, "data/state_tomo_data.csv")
    assert "Reconstructed rho_1" in md and os.path.isfile(fig)
    os.makedirs("data/sessions/other", exist_ok=True)
    with open("data/state_tomo_data.csv", encoding="utf-8") as f, \
            open("data/sessions/other/mine.csv", "w", encoding="utf-8") as g:
        g.write(f.read())
    try:
        with pytest.raises(wb.WorkbenchError, match="not found"):
            wb.tomography(SID, "data/sessions/other/mine.csv")         # another user's upload
    finally:
        shutil.rmtree("data/sessions/other", ignore_errors=True)
    with pytest.raises(wb.WorkbenchError, match="Upload a CSV"):
        wb.tomography(SID, "")


def test_simulated_tomography():
    wb.create(SID, "bell", kind="phi+")
    md, fig, summ = wb.simulate(SID, "rho_1", 1e5, 0.2, 0.2, 500, 1.0, 1.0)
    assert md and "rho_1" in summ


# ------------------------------------------------------------------ plots
def test_sweep_with_empty_and_given_ranges():
    md, fig, summ = wb.sweep(SID, "werner_p", ["concurrence", "chsh_max"], start="", stop="")
    assert os.path.isfile(fig) and "0.3333" in md and "0.7071" in md        # C > 0 above 1/3, CHSH > 2 above 1/sqrt2
    md, fig, _ = wb.sweep(SID, "werner_p", ["concurrence"], start="0.5", stop=" 1 ")
    assert os.path.isfile(fig) and "from 0.5 to 1" in md and "| 0.2500 |" in md   # C(0.5) = 0.25
    with pytest.raises(wb.WorkbenchError, match="not a number"):
        wb.sweep(SID, "werner_p", ["concurrence"], start="abc")
    with pytest.raises(wb.WorkbenchError, match="at least one quantity"):
        wb.sweep(SID, "werner_p", [])
    needs_states = next(k for k, v in at.SWEEPS.items() if v[3])
    with pytest.raises(wb.WorkbenchError, match="needs 1 to 3 states"):
        wb.sweep(SID, needs_states, ["concurrence"])


def test_density_fringes_bloch():
    wb.create(SID, "bell", kind="phi+")
    wb.create(SID, "bell", kind="werner", p=0.5)
    for md, fig, _ in (wb.density(SID, "rho_1", "heatmap"), wb.density(SID, "rho_1", "heatmap", "rho_2"),
                       wb.fringes(SID, "rho_1"), wb.bloch(SID, ["rho_1", "rho_2"])):
        assert md and os.path.isfile(fig) and f"sessions/{SID}/" in fig
    with pytest.raises(wb.WorkbenchError, match="1 to 3 states"):
        wb.bloch(SID, [])


# ------------------------------------------------------------------ sharing and the chat
def test_workbench_and_chat_share_states_per_user_only():
    wb.create(SID, "bell", kind="psi-")
    with session.use(SID):
        assert at.DESC["rho_1"] == "psi-"                               # the chat of this user sees it
    assert wb.state_choices("someone_else") == []                       # another user does not
    with session.use(SID):
        assert at.CURRENT_QUESTION.get("text", "") == ""                 # no chat question leaks into a tool


def test_explain_prompt():
    text = wb.explain_prompt("the state rho_1 (phi+)")
    assert text.startswith("In the Workbench I computed the state rho_1 (phi+).") and "Explain" in text


# ------------------------------------------------------------------ the Gradio wiring
gr = pytest.importorskip("gradio")


def _demo():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    class NoModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, *a, **kw):
            raise AssertionError("the Workbench must not call the model")
    import app
    return app.build_demo(NoModel(messages=iter([])))


def _fn(demo, name):
    return [f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == name]


def test_ui_builds_and_buttons_run_without_the_model():
    demo = _demo()
    runs = _fn(demo, "run")
    assert len(runs) >= 15                                               # one per Workbench button
    updates = _fn(demo, "updates")[0]
    out = updates(SID)
    assert "No states yet" in out[0]
    wb.create(SID, "bell", kind="phi+")
    wb.create(SID, "bell", kind="psi-")
    out = updates(SID)
    singles = [u for u in out[1:] if not isinstance(u.get("value"), list)]
    assert singles and all(u["value"] == "rho_2" for u in singles)      # an empty picker gets the newest state
    assert all(u["choices"] == wb.state_choices(SID) for u in out[1:])


def test_pickers_keep_their_choice_and_drop_deleted_states():
    demo = _demo()
    updates = _fn(demo, "updates")[0]
    wb.create(SID, "bell", kind="phi+")
    wb.create(SID, "bell", kind="psi-")
    n = len(updates(SID)) - 1
    out = updates(SID, *(["rho_1"] * n))
    kept = [u["value"] for u in out[1:]]
    assert "rho_1" in kept and "rho_2" not in [v for v in kept if not isinstance(v, list)]
    out = updates(SID, *(["rho_9"] * n))                                 # a state that no longer exists
    assert all(u["value"] in ("rho_2", []) for u in out[1:])


def test_button_errors_are_shown_in_the_panel():
    demo = _demo()
    wb.create(SID, "bell", kind="phi+")
    for run in _fn(demo, "run"):
        res = run(SID, *([None] * 30))                                  # every field empty
        md, img, img2, summ = res[0], res[1], res[2], res[3]
        assert md.startswith("⚠") and "Traceback" not in md, md
        assert img["visible"] is False and img2["visible"] is False and summ == ""   # nothing to explain


# ------------------------------------------------------------------ fixes after the first local test (2026-10-30)
def test_uploaded_file_shows_as_uploads_and_no_model_hint_in_the_states_table(tmp_path):
    import workbench_ui
    src = tmp_path / "my counts.csv"
    src.write_text(open("data/state_tomo_data.csv", encoding="utf-8").read(), encoding="utf-8")
    path = workbench_ui.copy_upload(str(src), SID)
    assert path == "uploads/my_counts.csv" and os.path.isfile(f"data/sessions/{SID}/my_counts.csv")
    md, fig, summ = wb.tomography(SID, path)
    assert "uploads/my_counts.csv" in md and "sessions" not in md.replace("figures/sessions", "")
    table = wb.states_table(SID)
    assert "reconstructed from uploads/my_counts.csv" in table
    assert "full report" not in table and "get_tomography_report" not in table and "<" not in table
    assert "full report" not in wb.state_choices(SID)[0][0]
    assert summ == "the tomography of uploads/my_counts.csv (state rho_1)"


def test_simulated_counts_file_is_shown_as_uploads():
    wb.create(SID, "bell", kind="phi+")
    md, _, _ = wb.simulate(SID, "rho_1", 1e6, 0.2, 0.2, 500, 20, 0.1)
    assert "Counts saved to uploads/simulated_rho_1.csv" in md
    assert "| Fidelity with true state | 1.0000 |" in md                 # same format as the other rows
    md, _, _ = wb.tomography(SID, "uploads/simulated_rho_1.csv")          # and it can be read back
    assert "Reconstructed rho_" in md


def test_bb84_rows_have_readable_labels():
    md, _, _ = wb.key_from_qber(SID, 0.03, 0.03)
    assert "| h(e_bit) | 0.1944 |" in md and "| h(e_phase) | 0.1944 |" in md
    assert "| secret key fraction | 0.6112 |" in md                       # 1 - 2 h(0.03)


def test_compensation_of_a_state_without_phase_says_none_is_needed():
    wb.create(SID, "bell", kind="phi+")
    md, _, _ = wb.compensate(SID, "rho_1", "B")
    assert "| Needed? | No — the relative phase is already 0°" in md
    assert "| Alternative | none needed |" in md and "If they get worse" not in md
    wb.create(SID, "pure", amps=("1", "0", "0", "i"))                     # phase 90°: compensation needed
    md, _, _ = wb.compensate(SID, "rho_3", "B")
    assert "Needed?" not in md and "HWP at 22.50°" in md                  # α = 180° + 4θ = -90°


def test_werner_sweep_title_uses_greek_kets():
    md, _, _ = wb.sweep(SID, "werner_p", ["concurrence"])
    assert "p∣ψ⁻⟩⟨ψ⁻∣ + (1 − p) I/4" in md and "psi-⟩" not in md
