"""
test_app.py - the web app (app.py) without a browser and without a language model: two users get
separate states and figures, uploads go to the user's own folder, and "What can you do?" works.
Run:  pytest -q test_app.py        (skipped if gradio is not installed)
"""
import os

import pytest

gr = pytest.importorskip("gradio")
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

import agent_tools as at  # noqa: E402
import app  # noqa: E402
import session  # noqa: E402


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i}])


@pytest.fixture(autouse=True)
def clean():
    yield
    for sid in session.active_ids():
        session.drop(sid)


def respond_fn(demo):
    return next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "respond")


def test_two_browser_tabs_are_separate_users():
    script = iter([call("create_state", {"kind": "werner", "p": 0.8}, "a"), AIMessage(content="Created."),
                   call("create_state", {"kind": "psi-"}, "b"), call("plot_density_matrix", {"handle": "rho_1"}, "c"),
                   AIMessage(content="Plotted.")])
    demo = app.build_demo(Scripted(messages=script))
    respond = respond_fn(demo)
    _, hist_a, a = respond("Create a Werner state with p = 0.8.", [], "tabA")
    _, hist_b, b = respond("Create psi- and plot its density matrix.", [], "tabB")
    with session.use("tabA"):
        assert [at.DESC[h] for h in at.STORE] == ["werner, p=0.8"]
    with session.use("tabB"):
        assert [at.DESC[h] for h in at.STORE] == ["psi-"]
    images = [m["content"] for m in hist_b if isinstance(m["content"], gr.Image)]
    assert len(images) == 1 and os.path.isfile("figures/sessions/tabB/rho_1_city.png")
    assert not os.path.exists("figures/sessions/tabA")             # tab A drew nothing
    assert len(at.STORE) == 0                                    # the command-line session is untouched


def test_what_can_you_do_in_the_app_needs_no_model():
    class NoModel(Scripted):
        def invoke(self, *a, **kw):
            raise AssertionError("no model call expected")
    respond = respond_fn(app.build_demo(NoModel(messages=iter([]))))
    _, hist, _ = respond("What can you do?", [], "tabC")
    assert "Tomography of your measured data" in hist[-1]["content"] and "no model needed" in hist[-1]["content"]


def test_upload_goes_to_the_users_folder_and_is_checked(tmp_path):
    src = tmp_path / "my counts.csv"
    src.write_text(open("data/state_tomo_data.csv", encoding="utf-8").read(), encoding="utf-8")
    text = app.prepare_upload(str(src), "tabD")
    assert "from uploads/my_counts.csv" in text and os.path.isfile("data/sessions/tabD/my_counts.csv")
    with session.use("tabD"):                                        # 'uploads/' = this user's own folder
        r = at.reconstruct_state_from_counts.invoke({"csv_path": "uploads/my_counts.csv"})
        assert r["handle"] and at.DESC[r["handle"]].startswith("reconstructed from uploads/my_counts.csv")
    for path in ("uploads/my_counts.csv", "data/sessions/tabD/my_counts.csv"):
        with session.use("tabE"), pytest.raises(Exception, match="not found") as e:
            at.reconstruct_state_from_counts.invoke({"csv_path": path})     # another user's upload
        assert "sessions" not in str(e.value).split("available:")[1]   # the list shows no other folder
    (tmp_path / "x.txt").write_text("a")
    assert "Please upload a .csv" in app.prepare_upload(str(tmp_path / "x.txt"), "tabD")


def test_new_chat_deletes_the_users_session():
    demo = app.build_demo(Scripted(messages=iter([call("create_state", {"kind": "phi+"}, "a"),
                                                  AIMessage(content="ok")])))
    respond = respond_fn(demo)
    respond("Create phi+.", [], "tabF")
    new_chat = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "new_chat")
    hist, new_id, _ = new_chat("tabF")
    assert "tabF" not in session.active_ids() and new_id != "tabF" and "Welcome" in hist[0]["content"]


def test_start_removes_folders_of_sessions_from_a_previous_run():
    os.makedirs("data/sessions/old_run", exist_ok=True)
    os.makedirs("figures/sessions/old_run", exist_ok=True)
    app.remove_orphan_folders()
    assert not os.path.exists("data/sessions/old_run") and not os.path.exists("figures/sessions/old_run")
    assert os.path.isfile("data/state_tomo_data.csv")                 # the shared sample file stays
