"""
test_space_review.py - cases from the Space (DeepSeek V4 Flash, 2026-11-04), question Q10:
"Apply global depolarizing noise with p = 0.3 to phi+ and compare it with local depolarizing noise with
p = 0.3 on both photons. Which destroys more entanglement?"  (local destroys more: C 0.235 vs 0.55)

Run:  pytest -q test_space_review.py
"""
from langchain_core.messages import AIMessage

import graph
from test_parameter_binding import call, run

Q10 = ("Apply global depolarizing noise with p = 0.3 to phi+ and compare it with local depolarizing noise with "
       "p = 0.3 on both photons. Which destroys more entanglement?")
CALLS = [call("create_state", {"kind": "phi+"}, "a"),
         call("apply_global_noise", {"handle": "rho_1", "channel": "global_depolarizing", "param": 0.3}, "b"),
         call("apply_local_noise", {"handle": "rho_1", "channel": "depolarizing", "param": 0.3}, "c"),
         call("compute_measures", {"handle": "rho_2"}, "d"), call("compute_measures", {"handle": "rho_3"}, "e")]
HEAD = "Local depolarizing noise destroys more entanglement than global depolarizing noise at the same p = 0.3.\n\n"


def answer(text):
    return run(Q10, CALLS + [AIMessage(content=HEAD + text)])[1]


def test_true_sentences_with_unknown_damage_words_are_kept():
    """'takes a bigger toll on entanglement' is TRUE for local noise; it was read as 'bigger entanglement'."""
    for s in ["The local channel takes a bigger toll on entanglement than the global one.",
              "Local noise hits the entanglement harder than global noise.",
              "Local noise reduces the concurrence more than global noise does."]:
        final = answer(s)
        assert s in final and "removed" not in final, s


def test_false_comparisons_are_still_removed_with_readable_state_names():
    for s in ["Local noise leaves a higher concurrence than global noise.",
              "Global noise destroys more entanglement than local noise."]:
        final = answer(s)
        assert s not in final and "Answer check removed" in final, s
        assert "rho_3 (local depolarizing 0.3 on both photons)" in final
        assert "rho_2 (global depolarizing 0.3)" in final


def test_the_model_requoting_the_tool_numbers_is_removed():
    final = answer("- Let me report the tool's numbers: concurrence 0.55, negativity 0.275. These are the actual "
                   "computed values.\n- Both remain entangled (C > 0, N > 0).")
    assert "Let me" not in final and "actual computed values" not in final
    assert "- Both remain entangled (C > 0, N > 0)." in final


def test_restatement_stripping_keeps_ordinary_text():
    s = graph.strip_restatement
    assert s("Concurrence is 0.55. Let me report the actual values: C = 0.55. Done.") == "Concurrence is 0.55. Done."
    keep = "| let me report the values |\nThe values are 0.55 and 0.235.\n- These are close to the bound."
    assert s(keep) == keep
