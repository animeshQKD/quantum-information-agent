"""
test_new_tools.py - checks the new channels, one-qubit noise, pure states and mixtures
against analytic results.   Run:  pytest -q test_new_tools.py
"""
import numpy as np
import pytest

from agent_tools import (STORE, apply_local_noise, compute_measures, create_pure_state,
                         create_state, mix_states, plot_measures_vs_noise)


def fresh():
    from agent_tools import CURRENT_QUESTION
    STORE.clear()
    CURRENT_QUESTION["text"] = ""      # direct tool calls: no user question to check against


def measure(h):
    return compute_measures.invoke({"handle": h})


def noisy(h, channel, param, qubits="both"):
    apply_local_noise.invoke({"handle": h, "channel": channel, "param": param, "qubits": qubits})
    return f"rho_{len(STORE)}"


@pytest.mark.parametrize("channel", ["bit_flip", "phase_flip", "bit_phase_flip"])
@pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
def test_flip_channels_on_phi_plus(channel, p):
    fresh(); create_state.invoke({"kind": "phi+"})
    m = measure(noisy("rho_1", channel, p))
    assert np.isclose(m["concurrence"], (1 - 2 * p) ** 2, atol=1e-6)      # C = (1-2p)^2
    assert np.isclose(m["negativity"], (1 - 2 * p) ** 2 / 2, atol=1e-6)


def test_flip_with_p_one_is_unitary():
    fresh(); create_state.invoke({"kind": "phi+"})
    assert np.isclose(measure(noisy("rho_1", "bit_flip", 1.0))["concurrence"], 1.0)


def test_phase_flip_equals_phase_damping():
    fresh(); create_state.invoke({"kind": "phi+"})
    a, b = noisy("rho_1", "phase_flip", 0.15), noisy("rho_1", "phase_damping", 0.7)
    assert np.allclose(STORE[a], STORE[b])                                  # lambda = 1 - 2p


@pytest.mark.parametrize("p", [0.3, 0.6, 0.7])
def test_depolarizing_one_qubit_threshold_two_thirds(p):
    fresh(); create_state.invoke({"kind": "phi+"})
    m = measure(noisy("rho_1", "depolarizing", p, "A"))
    assert np.isclose(m["concurrence"], max(0, (3 * (1 - p) - 1) / 2), atol=1e-6)


def test_amplitude_damping_one_qubit_no_sudden_death():
    fresh()
    a, b = 1 / np.sqrt(3), np.sqrt(2 / 3)
    create_pure_state.invoke({"amp_HH": a, "amp_HV": 0, "amp_VH": 0, "amp_VV": b})
    for g in (0.5, 0.99):
        h = noisy("rho_1", "amplitude_damping", g, "B")
        assert np.isclose(measure(h)["concurrence"], np.sqrt(1 - g) * 2 * a * b, atol=1e-6)


def test_amplitude_damping_both_sudden_death():
    fresh()
    a, b = 1 / np.sqrt(3), np.sqrt(2 / 3)                     # |b| > |a|, |V> = |1> excited
    create_pure_state.invoke({"amp_HH": a, "amp_HV": 0, "amp_VH": 0, "amp_VV": b})
    g_death = a / b                                            # survival 1 - g = 1 - a/b
    assert measure(noisy("rho_1", "amplitude_damping", g_death - 0.02))["concurrence"] > 1e-6
    assert measure(noisy("rho_1", "amplitude_damping", g_death + 0.02))["concurrence"] < 1e-9


def test_pure_state_normalizes_and_has_sin2t_concurrence():
    fresh(); t = 0.3
    msg = create_pure_state.invoke({"amp_HH": 2 * np.cos(t), "amp_HV": 0, "amp_VH": 0,
                                    "amp_VV": 2 * np.sin(t)})
    assert "normalized: amplitudes divided by 2.0000" in msg
    assert np.isclose(measure("rho_1")["concurrence"], np.sin(2 * t), atol=1e-6)


def test_pure_state_phase_180_is_phi_minus():
    fresh()
    create_pure_state.invoke({"amp_HH": 1, "amp_HV": 0, "amp_VH": 0, "amp_VV": 1, "phase_VV_deg": 180})
    create_state.invoke({"kind": "phi-"})
    assert np.allclose(STORE["rho_1"], STORE["rho_2"])


def test_mixture_and_bad_weights():
    fresh()
    create_state.invoke({"kind": "phi+"}); create_state.invoke({"kind": "psi+"})
    mix_states.invoke({"handles": ["rho_1", "rho_2"], "weights": [0.5, 0.5]})
    assert measure("rho_3")["concurrence"] < 1e-9                         # 50/50 Bell mix is separable
    mix_states.invoke({"handles": ["rho_1", "rho_2"], "weights": [0.8, 0.2]})
    assert np.isclose(measure("rho_4")["concurrence"], 0.6, atol=1e-6)    # 2*0.8 - 1
    with pytest.raises(ValueError):
        mix_states.invoke({"handles": ["rho_1", "rho_2"], "weights": [0.7, 0.7]})


def test_werner_on_phi_plus_same_measures_as_standard():
    fresh()
    create_state.invoke({"kind": "werner", "p": 0.6})
    create_state.invoke({"kind": "werner", "p": 0.6, "bell": "phi+"})
    m1, m2 = measure("rho_1"), measure("rho_2")
    for k in ("concurrence", "negativity", "chsh_max", "purity"):
        assert np.isclose(m1[k], m2[k])


def test_bad_inputs_are_rejected():
    fresh(); create_state.invoke({"kind": "phi+"})
    for bad in [{"channel": "bit_flip", "param": 1.2, "qubits": "both"},
                {"channel": "teleport", "param": 0.1, "qubits": "both"},
                {"channel": "bit_flip", "param": 0.1, "qubits": "C"}]:
        with pytest.raises(ValueError):
            apply_local_noise.invoke({"handle": "rho_1", **bad})


def test_plot_flip_features_show_revival():
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["features"]
    assert abs(f["entanglement_vanishes_at"] - 0.5) < 2e-3
    assert "yes" in f["entanglement_returns"]


def test_plot_one_qubit_depolarizing_threshold():
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "depolarizing",
                                       "qubits": "A"})["features"]
    assert abs(f["entanglement_vanishes_at"] - 2 / 3) < 2e-3


def test_plot_chsh_minimum_reported_for_bit_flip():
    f = plot_measures_vs_noise.invoke({"start_kind": "phi+", "channel": "bit_flip"})["features"]
    assert "minimum 2.0000 at 0.500" in f["chsh_drops_to_2_at"]      # 2*sqrt(1+(1-2p)^4)


def test_no_guessing_of_basis_states():
    from agent_tools import CURRENT_QUESTION
    fresh()
    CURRENT_QUESTION["text"] = "Create the state cos(0.3)"                 # incomplete
    with pytest.raises(ValueError, match="Do not guess"):
        create_pure_state.invoke({"amp_HH": 0.955, "amp_HV": 0, "amp_VH": 0, "amp_VV": 0.296})
    for q, hh, vv, ph in (("Create the state cos(0.3)|HH> + sin(0.3)|VV>", "cos(0.3)", "sin(0.3)", 0),
                          ("Create the state (|HH> + i|VV>)/sqrt(2)", 1, 1, 90),
                          ("make 0.6|00> + 0.8|11>", 0.6, 0.8, 0)):
        CURRENT_QUESTION["text"] = q
        create_pure_state.invoke({"amp_HH": hh, "amp_HV": 0, "amp_VH": 0, "amp_VV": vv,
                                  "phase_VV_deg": ph})
    CURRENT_QUESTION["text"] = ""


def test_graph_guessing_guard_end_to_end():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from graph import build_graph

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    fresh()
    guess = AIMessage(content="", tool_calls=[{"name": "create_pure_state", "id": "g1", "args":
            {"amp_HH": 0.955, "amp_HV": 0, "amp_VH": 0, "amp_VV": 0.296}}])
    ask = AIMessage(content="Which basis states should cos(0.3) multiply? Please write the full state.")
    out = build_graph(Scripted(messages=iter([guess, ask]))).invoke(
        {"messages": [("user", "Create the state cos(0.3)")], "retries": 0},
        {"configurable": {"thread_id": "guard"}})
    tool_reply = [m for m in out["messages"] if m.type == "tool"][0]
    assert "Do not guess" in tool_reply.content and not STORE
