"""
test_self_correction.py - an answer in which the model takes back what it wrote ('Let me reconsider. ...')
and writes it again: the retracted part must not reach the user. Also the false formula 'C = p' for a
Werner state. The first test uses a real answer of the live Space (2026-11-06, DeepSeek V4 Flash).
Run:  pytest -q test_self_correction.py
"""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import graph
from agent_tools import STORE

REAL = """The concurrence of the Werner state with p = 0.8 is **0.7**.

| Quantity | Value |
|---|---|
| Concurrence C | 0.7000 |
| Negativity N | 0.3500 |
| Entanglement of formation | 0.5919 |
| Purity | 0.7300 |
| CHSH max | 2.2627 |

**What it means**

- For a Werner state p|ψ⁻⟩⟨ψ⁻| + (1−p)I/4, the concurrence is C = p (here 0.8), and indeed the tool reports **C = 0.7**...

Let me reconsider. The tool reports concurrence 0.7, not 0.8. Let me report exactly what the tool gave.

The tool reports **concurrence = 0.7000** for this Werner state with p = 0.8.

**What it means**

- The concurrence is **0.7**, well above zero, so the state is entangled (p = 0.8 > 1/3).
- Negativity N = 0.35 = C/2, consistent with the Werner-state relation N = C/2.
- CHSH max = 2.2627 > 2, so the state also violates the CHSH inequality (p = 0.8 > 1/√2 ≈ 0.707).

*Conventions: Werner state built on |ψ⁻⟩ (standard), p = 0.8 visibility.*"""


def test_the_retracted_section_of_a_real_answer_is_removed():
    text, changed = graph.drop_retracted_draft(REAL)
    assert changed
    assert "Let me reconsider" not in text and "C = p (here 0.8)" not in text
    assert text.count("**What it means**") == 1                       # the rewrite is kept, once
    assert "| Concurrence C | 0.7000 |" in text and text.startswith("The concurrence of the Werner")
    assert "CHSH max = 2.2627 > 2" in text and text.rstrip().endswith("p = 0.8 visibility.*")


def test_the_whole_chat_never_shows_the_retraction():
    STORE.clear()
    script = iter([
        AIMessage(content="", tool_calls=[{"name": "create_state", "args": {"kind": "werner", "p": 0.8}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "compute_measures", "args": {"handle": "rho_1"}, "id": "c2"}]),
        AIMessage(content=REAL)])

    class Scripted(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    out = graph.build_graph(Scripted(messages=script)).invoke(
        {"messages": [("user", "What is the concurrence of a Werner state with p = 0.8?")], "retries": 0},
        {"configurable": {"thread_id": "selfcorr"}})
    final = out["messages"][-1].content
    assert "Let me reconsider" not in final and "C = p" not in final
    assert "0.7000" in final and final.count("**What it means**") == 1


def test_a_retraction_without_a_heading_takes_the_paragraph_before():
    a = ("The state is entangled.\n\nIts negativity is 0.5, the same as its concurrence.\n\n"
         "Wait, that is wrong: the tool gives N = 0.35.\n\nIts negativity is 0.35, half the concurrence.")
    text, changed = graph.drop_retracted_draft(a)
    assert changed and text == "The state is entangled.\n\nIts negativity is 0.35, half the concurrence."


def test_what_must_never_be_removed():
    # the opening sentence, and tables
    a = "C = 0.8 for this state.\n\nHmm, let me check that.\n\nC = 0.7 for this state (tool result)."
    text, _ = graph.drop_retracted_draft(a)
    assert text.startswith("C = 0.8 for this state.") and "Hmm" not in text
    b = "Results:\n\n| C | 0.7 |\n|---|---|\n\nLet me reconsider.\n\nThe state is entangled."
    text, _ = graph.drop_retracted_draft(b)
    assert "| C | 0.7 |" in text and "reconsider" not in text
    # a retraction at the very end: removed alone
    c = "The state is entangled (C = 0.7).\n\nWait, let me double-check the CHSH value."
    assert graph.drop_retracted_draft(c) == ("The state is entangled (C = 0.7).", True)


def test_ordinary_answers_are_untouched():
    for a in ["Actually, this state is entangled: C = 0.7.",
              "**Result**\n\nThe concurrence is 0.7.\n\nThat is well above zero.",
              "| Wait time | 3 s |\n|---|---|",
              REAL.split("\n\nLet me reconsider")[0].replace("C = p (here 0.8)", "C = (3p − 1)/2")]:
        assert graph.drop_retracted_draft(a) == (a, False)


def test_werner_concurrence_written_as_p_is_corrected():
    h = [HumanMessage("q")]
    fixed, why = graph.remove_false_claims(
        "- For a Werner state p|ψ⁻⟩⟨ψ⁻| + (1−p)I/4, the concurrence is C = p (here 0.8), and indeed", h)
    assert "C = max(0, (3p − 1)/2)" in fixed and "here 0.8" not in fixed and why == [graph._WERNER_WHY]
    fixed, why = graph.remove_false_claims("For the Werner state the concurrence is simply p.", h)
    assert fixed == "For the Werner state the concurrence is max(0, (3p − 1)/2)." and why
    for ok in ["For a Werner state the concurrence is C = (3p − 1)/2 = 0.7.",
               "The Werner weight is p = 0.8 and the concurrence is 0.7.",
               "For the pure state cos θ|HH> + sin θ|VV> the concurrence is C = sin 2θ."]:
        assert graph.remove_false_claims(ok, h) == (ok, [])
