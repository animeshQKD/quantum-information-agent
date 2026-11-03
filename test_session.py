"""
test_session.py - several web-app users at the same time must never see each other's states, files or
figures (session.py). The command line keeps one default session that behaves as before.
Run:  pytest -q test_session.py
"""
import os
import threading

import numpy as np
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import agent_tools as at
import session
from graph import build_graph


@pytest.fixture(autouse=True)
def clean():
    at.STORE.clear(); at.DESC.clear(); at.TOMO.clear()
    at.CURRENT_QUESTION["text"] = ""
    yield
    for sid in session.active_ids():
        session.drop(sid)


def test_two_users_have_separate_states():
    with session.use("alice"):
        at.create_state.invoke({"kind": "phi+"})
        at.create_state.invoke({"kind": "werner", "p": 0.5})
    with session.use("bob"):
        assert len(at.STORE) == 0                        # Bob does not see Alice's rho_1 and rho_2
        at.create_state.invoke({"kind": "psi-"})
        assert list(at.STORE) == ["rho_1"] and "psi-" in at.DESC["rho_1"]
    with session.use("alice"):
        assert list(at.STORE) == ["rho_1", "rho_2"] and "phi+" in at.DESC["rho_1"]
        assert at.compute_measures.invoke({"handle": "rho_2"})["concurrence"] == 0.25
    assert len(at.STORE) == 0                            # the command-line session is untouched


def test_figures_and_simulated_data_go_to_the_users_folders():
    with session.use("alice"):
        at.create_state.invoke({"kind": "phi+"})
        fig = at.plot_density_matrix.invoke({"handle": "rho_1"})["figure"]
        sweep = at.plot_sweep.invoke({"sweep": "werner_p", "quantities": ["concurrence"]})["figure"]
        noise = at.plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["figure"]
        sim = at.simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 1e5, "time_per_setting_s": 0.1})
    for path in (fig, sweep, noise):
        assert path.startswith("figures/sessions/alice/") and os.path.isfile(path)
    assert sim["csv_file"] == "uploads/simulated_rho_1.csv"                 # shown short ...
    assert os.path.isfile("data/sessions/alice/simulated_rho_1.csv")        # ... stored in alice's folder
    at.create_state.invoke({"kind": "phi+"})
    assert at.plot_density_matrix.invoke({"handle": "rho_1"})["figure"] == "figures/rho_1_city.png"


def test_a_user_can_read_only_shared_examples_and_own_files():
    with session.use("alice"):
        at.create_state.invoke({"kind": "phi+"})
        own = at.simulate_tomography.invoke({"handle": "rho_1", "pair_rate": 1e5, "time_per_setting_s": 0.1})["csv_file"]
        assert at.reconstruct_state_from_counts.invoke({"csv_path": own})["handle"]
        assert at.reconstruct_state_from_counts.invoke({"csv_path": "data/state_tomo_data.csv"})["handle"]
    with session.use("bob"):
        with pytest.raises(Exception, match="not found"):          # Alice's upload is invisible to Bob
            at.reconstruct_state_from_counts.invoke({"csv_path": own})
        with pytest.raises(Exception, match="not found") as e:
            at.reconstruct_state_from_counts.invoke({"csv_path": os.path.abspath("benchmark_tasks.json")[:-5] + ".csv"})
        assert "alice" not in str(e.value)                         # the file list shows only allowed files
    assert session.allowed_csv("anything/anywhere.csv")            # the command line may read any file


def test_the_graph_runs_tools_in_the_users_session():
    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self
    script = iter([AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "werner", "p": 0.8},
                                                      "id": "c1"}]),
                   AIMessage(content="", tool_calls=[{"name": "compute_measures", "args": {"handle": "rho_1"},
                                                      "id": "c2"}]),
                   AIMessage(content="The concurrence is 0.7000.")])
    app = build_graph(Scripted(messages=script))
    with session.use("carol"):
        out = app.invoke({"messages": [("user", "Create a Werner state with p = 0.8.")], "retries": 0},
                         {"configurable": {"thread_id": "carol"}})
        assert list(at.STORE) == ["rho_1"] and "werner" in at.DESC["rho_1"]
    assert np.isclose(out["results"][0]["concurrence"], 0.7)
    assert len(at.STORE) == 0


def test_parallel_users_do_not_mix_up_handles():
    errors = []

    def work(name, kind, n):
        try:
            with session.use(name):
                for _ in range(n):
                    at.create_state.invoke({"kind": kind})
                assert len(at.STORE) == n and all(kind in d for d in at.DESC.values())
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    threads = [threading.Thread(target=work, args=(f"user{i}", k, 30))
               for i, k in enumerate(["phi+", "psi-", "phi-", "psi+"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_idle_sessions_are_deleted_with_their_files(monkeypatch):
    dropped = []
    monkeypatch.setattr(session, "ON_DROP", [dropped.append])
    with session.use("old"):
        at.create_state.invoke({"kind": "phi+"})
        fig = at.plot_density_matrix.invoke({"handle": "rho_1"})["figure"]
    assert os.path.isfile(fig)
    monkeypatch.setattr(session, "SESSION_TTL_S", 0)
    with session.use("new"):
        pass
    assert "old" in dropped and not os.path.exists("figures/sessions/old") and "old" not in session.active_ids()


def test_limits_and_bad_ids(monkeypatch):
    monkeypatch.setattr(session, "MAX_STATES", 2)
    with session.use("dave"):
        at.create_state.invoke({"kind": "phi+"})
        at.create_state.invoke({"kind": "psi-"})
        with pytest.raises(Exception, match="limit"):
            at.create_state.invoke({"kind": "phi-"})
    with pytest.raises(ValueError):
        session.get("../../etc")


def test_uploads_alias_and_short_paths_for_web_users():
    with session.use("aliasA") as s:
        assert session.real_path("uploads/x.csv") == f"{s.data_dir}/x.csv"
        assert session.real_path("uploads\\x.csv") == f"{s.data_dir}/x.csv"           # Windows spelling
        assert session.real_path("data/state_tomo_data.csv") == "data/state_tomo_data.csv"
        os.makedirs(s.data_dir, exist_ok=True)
        open(f"{s.data_dir}/mine.csv", "w").close()
        assert "uploads/mine.csv" in session.readable_csvs()
        assert not any("sessions" in p for p in session.readable_csvs())
    assert session.real_path("uploads/x.csv") == "uploads/x.csv"          # the command line: a normal path
    assert session.display_path("from data/sessions/ab12/x.csv and data\\sessions\\cd\\y.csv; data/z.csv") == \
        "from uploads/x.csv and uploads/y.csv; data/z.csv"
    session.drop("aliasA")
