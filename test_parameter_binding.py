"""
test_parameter_binding.py - the agent must answer the experiment the user ASKED for, not just quote grounded
numbers. Raised by a reviewer on Hugging Face (2026-11-04): "ask for global depolarizing noise at p = 0.3,
have a scripted model call the tool with p = 0.03, then faithfully quote that result. All output numbers
could be grounded and within physical bounds, while answering the wrong experiment."

Every test runs the real graph (agent -> tools -> verifier -> answer check) with a scripted model, offline.
Run:  pytest -q test_parameter_binding.py
"""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import agent_tools as at
import graph


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kw):
        return self


def call(name, args, i):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": i}])


def run(question, replies):
    """Run one question; `replies` = the model's replies in order (a fresh message each, like a real model)."""
    at.STORE.clear(); at.DESC.clear()                                     # noqa: E702
    g = graph.build_graph(Scripted(messages=iter(replies)))
    out = g.invoke({"messages": [("user", question)], "retries": 0},
                   {"configurable": {"thread_id": "binding"}, "recursion_limit": 60})
    notes = [m.content for m in out["messages"] if isinstance(m, HumanMessage) and str(m.content).startswith("VERIFIER")]
    return notes, out["messages"][-1].content


def concurrence(channel, param, local=False):
    at.STORE.clear(); at.DESC.clear()                                     # noqa: E702
    at.create_state.invoke({"kind": "phi+"})
    tool = at.apply_local_noise if local else at.apply_global_noise
    h = tool.invoke({"handle": "rho_1", "channel": channel, "param": param}).split()[1]
    return at.compute_measures.invoke({"handle": h})["concurrence"]


Q = "Apply global depolarizing noise with p = 0.3 to phi+ and give its concurrence."
WRONG_CALLS = [call("create_state", {"kind": "phi+"}, "a"),
               call("apply_global_noise", {"handle": "rho_1", "channel": "global_depolarizing", "param": 0.03}, "b"),
               call("compute_measures", {"handle": "rho_2"}, "c")]


def test_wrong_value_is_sent_back_and_the_corrected_run_passes():
    """The reviewer's case: p = 0.3 asked, p = 0.03 run, the result quoted faithfully (grounded numbers)."""
    c_wrong, c_right = concurrence("global_depolarizing", 0.03), concurrence("global_depolarizing", 0.3)
    replies = WRONG_CALLS + [
        AIMessage(content=f"Global depolarizing noise at p = 0.3 leaves phi+ with concurrence {c_wrong:.4f}."),
        call("apply_global_noise", {"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}, "d"),
        call("compute_measures", {"handle": "rho_3"}, "e"),
        AIMessage(content=f"Global depolarizing noise at p = 0.3 leaves phi+ with concurrence {c_right:.4f}.")]
    notes, final = run(Q, replies)
    assert len(notes) == 1 and "[binding]" in notes[0] and "0.03" in notes[0]
    assert f"{c_right:.4f}" in final and "warning" not in final.lower()


def test_a_model_that_ignores_the_note_never_passes_silently():
    c_wrong = concurrence("global_depolarizing", 0.03)
    wrong = f"Global depolarizing noise at p = 0.3 leaves phi+ with concurrence {c_wrong:.4f}."
    notes, final = run(Q, WRONG_CALLS + [AIMessage(content=wrong) for _ in range(5)])
    assert len(notes) == 1                                  # one note (binding), not a second one for the same value
    assert "Answer check warning" in final and "global depolarizing with parameter 0.3" in final and "0.03" in final


def test_swapped_parameters_are_caught():
    """Global 0.3 and local 0.1 asked, run as global 0.1 and local 0.3: every value is used SOMEWHERE, so a
    value-only check passes, yet both states answer the wrong experiment."""
    q = ("Apply global depolarizing noise with p = 0.3 and local depolarizing noise with p = 0.1 on both "
         "photons of phi+, and compare their concurrence.")
    cg, cl = concurrence("global_depolarizing", 0.1), concurrence("depolarizing", 0.3, local=True)
    calls = [call("create_state", {"kind": "phi+"}, "a"),
             call("apply_global_noise", {"handle": "rho_1", "channel": "global_depolarizing", "param": 0.1}, "b"),
             call("apply_local_noise", {"handle": "rho_1", "channel": "depolarizing", "param": 0.3}, "c"),
             call("compute_measures", {"handle": "rho_2"}, "d"), call("compute_measures", {"handle": "rho_3"}, "e")]
    answer = f"Global depolarizing p = 0.3 gives concurrence {cg:.4f}; local p = 0.1 gives {cl:.4f}."
    notes, final = run(q, calls + [AIMessage(content=answer) for _ in range(5)])
    assert "[binding]" in notes[0] and "global depolarizing with parameter 0.3" in notes[0]
    assert "depolarizing with parameter 0.1" in notes[0]
    assert "Answer check warning" in final


def test_the_correct_run_gets_no_binding_note():
    """Q10 of the verification list, run correctly: no note at all."""
    q = ("Apply global depolarizing noise with p = 0.3 to phi+ and compare it with local depolarizing noise with "
         "p = 0.3 on both photons. Which destroys more entanglement?")
    cg, cl = concurrence("global_depolarizing", 0.3), concurrence("depolarizing", 0.3, local=True)
    calls = [call("create_state", {"kind": "phi+"}, "a"),
             call("apply_global_noise", {"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}, "b"),
             call("apply_local_noise", {"handle": "rho_1", "channel": "depolarizing", "param": 0.3}, "c"),
             call("compute_measures", {"handle": "rho_2"}, "d"), call("compute_measures", {"handle": "rho_3"}, "e")]
    answer = f"Local depolarizing destroys more entanglement: concurrence {cl:.4f} versus {cg:.4f} for global noise."
    notes, final = run(q, calls + [AIMessage(content=answer)])
    assert notes == [] and "warning" not in final.lower()


def test_values_are_bound_to_the_channel_named_before_them():
    pairs = graph.requested_channel_parameters
    assert pairs("Apply global depolarizing noise with p = 0.3 and local depolarizing noise with p = 0.1") == \
        [("global_depolarizing", 0.3), ("depolarizing", 0.1)]
    assert pairs("apply local amplitude damping with gamma = 0.3 to both photons") == [("amplitude_damping", 0.3)]
    assert pairs("apply collective dephasing with p = 0.5 to both") == [("collective_dephasing", 0.5)]
    assert pairs("Create a Werner state with p = 0.8 and apply phase damping 0.7 to qubit A.") == \
        [("phase_damping", 0.7)]                            # Werner p is not a channel value
    assert pairs("Create a Werner state with p = 0.8.") == []
    assert pairs("Apply phase damping. Then make a Werner state with p = 0.4.") == []   # another sentence


def test_no_note_when_the_channel_was_not_applied_with_a_value():
    """A sweep (no 'param' argument) or a time evolution never counts as the wrong value."""
    q = "Let phi+ evolve under collective dephasing with gamma = 1 up to t = 2.5 and plot every element."
    turn = [call("create_state", {"kind": "phi+"}, "a")]
    assert graph.channel_parameters_misbound(q, turn) == []


def test_ambiguous_binding_stays_silent():
    """'global and local depolarizing with p = 0.3 and 0.1 respectively': two channels, then two values.
    Binding each value to the nearest name would send a CORRECT run back, so nothing is bound."""
    pairs = graph.requested_channel_parameters
    assert pairs("Apply global and local depolarizing noise with p = 0.3 and 0.1 respectively to phi+.") == []
    assert pairs("Apply local depolarizing and global depolarizing noise with p = 0.3 and 0.1 to phi+.") == []
    q = "Apply global and local depolarizing noise with p = 0.3 and 0.1 respectively to phi+ and compare them."
    turn = [call("apply_global_noise", {"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}, "b"),
            call("apply_local_noise", {"handle": "rho_1", "channel": "depolarizing", "param": 0.1}, "c")]
    assert graph.channel_parameters_misbound(q, turn) == []
