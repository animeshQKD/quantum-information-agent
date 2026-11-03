"""
test_capabilities.py - the capabilities list (capabilities.py) must match the code: every tool is in an
area, every example question reaches that area's tools, and "what can you do?" is answered without a
model call.   Run:  pytest -q test_capabilities.py
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

import capabilities
import graph
import routing
from agent_tools import TOOLS, list_capabilities


def test_every_tool_belongs_to_an_area():
    in_areas = {t for a in capabilities.AREAS for t in a.tools}
    names = {t.name for t in TOOLS}
    assert in_areas <= names, in_areas - names                       # no area names a missing tool
    assert names - in_areas == {"list_capabilities"}, names - in_areas


@pytest.mark.parametrize("a,question", [(a, q) for a in capabilities.AREAS for q in a.examples])
def test_every_example_reaches_its_area(a, question):
    names, _ = routing.select(question)
    assert set(a.tools) & set(names), (a.key, question)


def test_examples_are_benchmark_or_validated_questions():
    """The examples are questions we checked (benchmark or test_routing), not new untested ones."""
    import json

    from test_routing import ASKED
    known = {t["question"] for t in json.load(open("benchmark_tasks.json", encoding="utf-8"))["tasks"]}
    known |= {q for q, _ in ASKED}
    unknown = [q for a in capabilities.AREAS for q in a.examples if q not in known]
    assert not unknown, unknown


def test_page_help_answer_and_tool_output():
    page = capabilities.markdown()
    assert all(a.title in page for a in capabilities.AREAS) and "Limits" in page
    help_text = capabilities.help_answer()
    assert help_text.count("\n| ") == len(capabilities.AREAS) + 1     # header + one row per area
    out = list_capabilities.invoke({"area": "tomography"})
    assert [x["area"] for x in out["areas"]] == ["Tomography of your measured data"] and out["limits"]
    assert len(list_capabilities.invoke({})["areas"]) == len(capabilities.AREAS)


class NoModel:
    def invoke(self, messages):
        raise AssertionError("the model must not be called for 'what can you do?'")


@pytest.mark.parametrize("q", ["What can you do?", "help", "Hi, what can you do for me?",
                               "Show me your capabilities.", "How do I use this agent?"])
def test_what_can_you_do_needs_no_model(q):
    out = graph.make_agent_node(NoModel())({"messages": [HumanMessage(q)]})["messages"][0]
    assert out.name == graph.HELP_REPLY and "Tomography of your measured data" in out.content
    assert graph.answer_check({"messages": [HumanMessage(q), out]}) == {}   # its numbers are not 'ungrounded'


def test_specific_capability_questions_go_to_the_model_with_the_tool():
    names, groups = routing.select("Can you simulate three-qubit GHZ states under non-Markovian noise?")
    assert "list_capabilities" in names and "help" in groups
    called = []

    class Model:
        def invoke(self, messages):
            called.append(messages)
            return AIMessage(content="There is no tool for three-qubit states.")
    graph.make_agent_node(Model())({"messages": [HumanMessage("What can you do with Werner states?")]})
    assert called                                                     # not the fixed answer
