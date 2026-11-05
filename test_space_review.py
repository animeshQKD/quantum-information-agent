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


# The second Q10 run on the Space (2026-11-05): correct table, two wrong sentences.
SPACE_2 = ("**Local depolarizing noise destroys more entanglement.** At p = 0.3, global depolarizing leaves "
           "concurrence 0.55, while local depolarizing on both photons leaves only 0.235.\n\n**What it means**\n"
           "- Global depolarizing p on a Bell state gives a Werner state with visibility p_W = 1 − p = 0.7, so "
           "concurrence = 0.7 (here 0.55 after the 0.3 mixing — the tool's value). Local depolarizing on both "
           "photons gives p_W = (1−p)² = 0.49, a much weaker state.\n"
           "- Local noise acts on each photon with its own environment, so it degrades the correlations twice as "
           "effectively as a single global channel.")


def test_a_weight_quoted_as_the_concurrence_is_sent_back():
    """'p_W = 0.7, so concurrence = 0.7': 0.7 is grounded (it is 1 - p) but it is not the concurrence (0.55).
    'Bell' earlier in the sentence made the check think it was a reference value."""
    good = SPACE_2.replace("concurrence = 0.7 (here 0.55 after the 0.3 mixing — the tool's value)",
                           "concurrence = (3·0.7 − 1)/2 = 0.55")
    notes, final = run(Q10, CALLS + [AIMessage(content=SPACE_2), AIMessage(content=good)])
    assert len(notes) == 1 and "[quoted value]" in notes[0] and "concurrence 0.7" in notes[0]
    assert "concurrence = 0.7" not in final and "concurrence = (3·0.7 − 1)/2 = 0.55" in final


def test_reference_values_of_a_bell_state_are_still_allowed():
    notes, final = run(Q10, CALLS + [AIMessage(content=HEAD + "A Bell state has concurrence 1; global noise "
                                                                "leaves 0.55 and local noise 0.235.")])
    assert notes == [] and "concurrence 1" in final


def test_twice_as_strongly_is_corrected():
    final = answer("- Local noise acts on each photon with its own environment, so it degrades the correlations "
                   "twice as effectively as a single global channel.")
    assert "twice" not in final.split("*(")[0]
    assert "degrades the correlations more strongly than a single global channel" in final
    assert "2p − p²" in final
    # 'twice' about something else is left alone
    kept = "The concurrence of the global state is more than twice that of the local one."
    assert kept in answer(kept)
