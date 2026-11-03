"""
test_display.py - the code that makes answers readable (LaTeX -> Unicode, step summaries).
Run:  pytest -q test_display.py
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from display import summarize_steps, to_unicode


def test_latex_from_real_answers_becomes_unicode():
    assert to_unicode(r"\(|\psi\rangle = \cos(0.3)\,|HH\rangle + \sin(0.3)\,|VV\rangle\)") == \
        "∣ψ⟩ = cos(0.3) ∣HH⟩ + sin(0.3) ∣VV⟩"
    assert to_unicode(r"the Tsirelson limit \(2\sqrt{2}\)") == "the Tsirelson limit 2√2"
    assert to_unicode(r"\(\frac{1}{\sqrt{3}}|HH\rangle+\sqrt{\frac{2}{3}}|VV\rangle\)") == "1/√3∣HH⟩+√(2/3)∣VV⟩"
    assert to_unicode(r"Purity \(\mathrm{Tr}\rho^{2}\), \(\rho^{T_B}\), \(\gamma \le 0.8\)") == \
        "Purity Tr ρ², ρᵀᴮ, γ ≤ 0.8"


def test_handles_and_plain_text_untouched():
    assert to_unicode("handle `rho_10` and rho_1") == "handle `rho_10` and rho_1"
    assert to_unicode("|a| < 1 and C | N") == "|a| < 1 and C | N"


def test_kets_cannot_break_markdown_tables():
    row = to_unicode("| 0.6|HH> + 0.8|VV> | 0.96 |")
    assert row.count("|") == 3               # only the 3 real column separators remain
    assert "∣HH⟩" in row and "∣VV⟩" in row
    assert to_unicode("0.8|φ⁺⟩⟨φ⁺|") == "0.8∣φ⁺⟩⟨φ⁺∣"


def test_steps_are_short_sentences():
    msgs = [AIMessage(content="", tool_calls=[
                {"name": "create_state", "args": {"kind": "phi+"}, "id": "1"},
                {"name": "compute_measures", "args": {"handle": "rho_1"}, "id": "2"},
                {"name": "apply_local_noise", "args": {}, "id": "3"}]),
            ToolMessage("Created rho_1 (phi+).", tool_call_id="1"),
            ToolMessage('{"concurrence": 1.0}', tool_call_id="2"),
            ToolMessage("Error: ValueError(\"Unknown handle 'rho_9'. Known handles: ['rho_1']\")\n Please fix your mistakes.",
                        tool_call_id="3", status="error"),
            HumanMessage("VERIFIER: your answer contains numbers that no tool produced: 0.97. Remove them, ...")]
    steps = summarize_steps(msgs)
    assert steps[0] == ("ok", "Created rho_1 (phi+)")
    assert steps[1] == ("ok", "Computed all measures of rho_1")
    assert steps[2][0] == "warn" and "Unknown handle 'rho_9'" in steps[2][1] and "Please fix" not in steps[2][1]
    assert steps[3][0] == "warn" and "0.97" in steps[3][1]


def test_system_prompt_has_no_hidden_control_characters():
    from graph import SYSTEM_PROMPT
    assert "\f" not in SYSTEM_PROMPT and "\r" not in SYSTEM_PROMPT   # \frac, \rangle must stay literal
    assert r"\frac" in SYSTEM_PROMPT and r"\rangle" in SYSTEM_PROMPT


def test_unclosed_bold_is_repaired():
    from display import to_unicode
    s = "**The state is entangled with concurrence **1.0**, negativity **0.5**."
    assert to_unicode(s) == "The state is entangled with concurrence **1.0**, negativity **0.5**."
    assert to_unicode("Result: **0.5** and **1.0") == "Result: **0.5** and 1.0"
    assert to_unicode("**ok** fine") == "**ok** fine"


def test_step_lines_keep_square_brackets_in_the_terminal():
    from io import StringIO
    from rich.console import Console
    from rich.markup import escape
    out = StringIO()
    Console(file=out, width=200).print("  ok " + escape("pure 0.9553|HH> [from HH: cos(0.3), VV: sin(0.3)]"))
    assert "[from HH: cos(0.3), VV: sin(0.3)]" in out.getvalue()


def test_ascii_bras_do_not_split_table_cells():
    from display import to_unicode
    t = ("| State description | Concurrence | Teleportation fidelity |\n|---|---|---|\n"
         "| mixture 0.3|psi+><psi+| + 0.7|HH><HH| | 0.3000 | 0.5667 |")
    row = to_unicode(t).split("\n")[2]
    assert row == "| mixture 0.3∣psi+⟩⟨psi+∣ + 0.7∣HH⟩⟨HH∣ | 0.3000 | 0.5667 |"
    assert row.count("|") == 4
    assert to_unicode("| C<2 | x |") == "| C<2 | x |"          # a less-than sign is left alone
