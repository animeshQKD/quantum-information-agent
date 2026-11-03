"""
test_routing.py - the model gets only the tools and instructions a question needs.
Every question used so far to validate the agent must still reach the tools it needs.
Run:  pytest -q test_routing.py
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import routing
from agent_tools import TOOLS

ASKED = [  # (question, tools it needs)
    # one photon, density matrices, global noise (2026-10-31); also the run.py verification questions
    ("Create a horizontally polarized single photon and a single photon with linear polarization at 30 degrees and "
     "degree of polarization 0.8. Give their Stokes parameters, degree of polarization, purity and entropy.",
     ["create_single_photon_state", "compute_measures"]),
    ("Create phi+ and give the reduced state of photon A. Is photon A polarized?", ["create_state", "reduced_state"]),
    ("Reconstruct the polarization of a single photon from these counts: H = 980, V = 20, D = 510, A = 490, R = 500, "
     "L = 500. What are the Stokes parameters and the degree of polarization, with error bars?",
     ["reconstruct_single_photon"]),
    ("Create a Werner state with p = 0.8. Give its von Neumann entropy, the entropies of the two photons and the "
     "conditional entropy S(A|B).", ["create_state", "compute_measures"]),
    ("Create phi+, apply local amplitude damping with gamma = 0.3 to both photons, and show the density matrix before "
     "and after with the change of every element.", ["apply_local_noise", "show_density_matrix"]),
    ("Apply global depolarizing noise with p = 0.3 to phi+ and compare it with local depolarizing noise with p = 0.3 "
     "on both photons. Which destroys more entanglement?", ["apply_global_noise", "apply_local_noise"]),
    ("Show the Kraus operators of collective dephasing with p = 0.3 and check that they are complete.",
     ["get_kraus_operators"]),
    ("Create psi- and psi+ and let both evolve under collective decay with gamma = 1 up to t = 2. Which one keeps its "
     "entanglement?", ["evolve_state"]),
    ("Create a single photon in the state (|H> + i|V>)/sqrt(2). Which polarization is it, and what is its "
     "polarization ellipse?", ["create_single_photon_state", "compute_measures"]),
    ("Create a D-polarized single photon and send it through a quarter-wave plate with its fast axis at 0 degrees. "
     "What is the output polarization?", ["create_single_photon_state", "apply_waveplate"]),
    ("Create a single D-polarized photon and let it evolve under dephasing with rate gamma = 1 up to t = 2. How does "
     "the degree of polarization change?", ["create_single_photon_state", "evolve_state"]),
    ("Create the product state of an H photon and a D photon and show its density matrix.",
     ["create_single_photon_state", "combine_photons", "show_density_matrix"]),
    ("Create phi+ and psi-, apply collective dephasing with p = 0.5 to both, and give their concurrence.",
     ["apply_global_noise"]),
    ("Create phi+ and apply my own Kraus operators to photon B: K0 = [[1, 0], [0, sqrt(0.6)]] and K1 = [[0, sqrt(0.4)], "
     "[0, 0]]. What is the concurrence after?", ["apply_kraus_operators"]),
    ("Plot the concurrence of phi+ and psi+ versus the parameter of collective dephasing.", ["plot_sweep"]),
    ("Let phi+ evolve under collective dephasing with gamma = 1 up to t = 2 and plot every density-matrix element "
     "versus time.", ["evolve_state"]),
    ("Create phi+, apply phase damping with lambda = 0.7 to both qubits, and compute all measures.", ["apply_local_noise"]),
    ("Evolve phi+ under depolarizing noise with rate gamma = 1 on both qubits up to t = 1. When does CHSH "
     "drop to 2 and when does the entanglement vanish?", ["evolve_state"]),
    ("Evolve psi- for t up to 3 under the local Hamiltonian H = (omega/2) sigma_x on both qubits with omega = 2 "
     "and no dissipation.", ["evolve_state"]),
    ("Create phi+ and put a half-wave plate at 22.5 degrees on photon A. What happens to the state?", ["apply_waveplate"]),
    ("Reconstruct data/state_tomo_data.csv and find the wave plates on photon B that turn it into phi+.",
     ["reconstruct_state_from_counts", "compensate_phase"]),
    ("Simulate a tomography of phi+ with a pair rate of 1e6 per second, detection efficiency 0.2 on both sides, "
     "500 dark counts per second, a 20 ns coincidence window and 0.1 s per setting.", ["simulate_tomography"]),
    ("Create a Werner state with p = 0.8 built on phi+ and simulate its tomography with a pair rate of 2e4 per "
     "second and 0.05 s per setting. How long must I count per setting to get the concurrence error down to 0.005?",
     ["create_state", "simulate_tomography"]),
    ("Create a Werner state with p = 0.5 built on phi+. What is its teleportation fidelity?", ["teleportation_fidelity"]),
    ("What is the key fraction if the QBER is 1% in the H/V basis and 14% in the D/A basis?", ["qkd_key_rate"]),
    ("For the three-qubit bit-flip code, compare the logical error with the physical error at p = 0.1 and p = 0.6.",
     ["bit_flip_code"]),
    ("Create the mixture 0.3|psi+><psi+| + 0.7|HH><HH|. Is it entangled, and does it beat the classical limit "
     "for teleportation?", ["mix_states", "create_pure_state", "teleportation_fidelity"]),
    ("Create the Werner state with p = 0.55. Is it steerable?", ["certify_correlations"]),
    ("Create a Werner state with p = 0.75 built on phi+, simulate its tomography with a pair rate of 2e4 per second "
     "and 0.05 s per setting, and tell me whether the CHSH violation of the reconstructed state is significant.",
     ["simulate_tomography", "certify_correlations"]),
    ("Reconstruct data/state_tomo_data.csv. Are the entanglement, EPR steering and the CHSH violation "
     "statistically significant? Give the number of standard deviations.",
     ["reconstruct_state_from_counts", "certify_correlations"]),
    ("Plot the polarization-correlation fringes of the reconstructed state. What are the visibilities?", ["plot_fringes"]),
    ("Plot the Bloch vectors and correlation matrices of the reconstructed state and of phi+.",
     ["plot_bloch_and_correlations"]),
    ("Plot how well the tomography fit matches the measured counts. Which settings deviate most?",
     ["plot_tomography_fit"]),
    ("Plot data/example_fringe_scan.csv: angle_deg against counts_HV and counts_DA with a fringe fit.",
     ["plot_data_file"]),
    ("Create phi+ and psi-, and plot their concurrence and purity versus amplitude damping on qubit B only.",
     ["plot_sweep", "create_state"]),
    ("Plot how the fidelity with phi+ and the BB84 key fraction of the reconstructed state change under a phase "
     "shifter on photon B from 0 to 360 degrees.", ["plot_sweep"]),
    ("Reconstruct data/state_tomo_data.csv and plot its density matrix as a heatmap next to phi+.",
     ["reconstruct_state_from_counts", "plot_density_matrix"]),
    ("Plot the measures of phi+ under bit-flip noise on both qubits and describe what happens.",
     ["plot_measures_vs_noise"]),
    ("Reconstruct data/state_tomo_data.csv using the convention R=(H-iV)/sqrt2. What is the phase of the closest "
     "maximally entangled state?", ["reconstruct_state_from_counts"]),
    ("Show me the tomography report of rho_3.", ["get_tomography_report"]),
    ("Create phi+ and a Werner state with p = 0.8 built on phi+. What is the fidelity between them?",
     ["create_state", "compare_states"]),
]


@pytest.mark.parametrize("question,needed", ASKED)
def test_every_validated_question_reaches_its_tools(question, needed):
    names, _ = routing.select(question)
    assert set(needed) <= set(names), set(needed) - set(names)


def test_benchmark_questions_reach_their_tools():
    spec = json.load(open("benchmark_tasks.json", encoding="utf-8"))
    for t in spec["tasks"]:
        names, _ = routing.select(t["question"])
        assert set(t.get("tools", [])) <= set(names), (t["id"], set(t.get("tools", [])) - set(names))


def test_routing_saves_most_of_the_prompt():
    from langchain_core.utils.function_calling import convert_to_openai_tool
    size = {t.name: len(json.dumps(convert_to_openai_tool(t))) for t in TOOLS}
    full = len(routing.system_prompt()) + sum(size.values())
    spec = json.load(open("benchmark_tasks.json", encoding="utf-8"))
    routed = []
    for t in spec["tasks"]:
        names, groups = routing.select(t["question"])
        routed.append(len(routing.system_prompt(groups)) + sum(size[n] for n in names))
    assert sum(routed) / len(routed) < 0.45 * full                    # less than half on average
    names, groups = routing.select("Create the Werner state with p = 0.3 and give its concurrence.")
    assert names == routing.CORE_TOOLS and groups == []


def test_follow_up_keeps_the_previous_topic():
    names, groups = routing.select("Now plot it.", previous_tools=["create_state", "apply_local_noise"],
                                   previous_question="Create phi+ and apply depolarizing noise p = 0.3 to both qubits.")
    assert "apply_local_noise" in names and "plot_sweep" in names and "plot_measures_vs_noise" in names
    assert {"noise", "plots"} <= set(groups)
    names, _ = routing.select("Create psi- and compute its discord, purity and coherence for comparison with "
                              "the Werner family from the textbook chapter on correlations.",
                              previous_tools=["evolve_state"], previous_question="Evolve phi+ ...")
    assert "evolve_state" not in names                                # a long new question starts fresh


def test_full_prompt_still_contains_every_section():
    full = routing.system_prompt()
    for rules, facts in routing.SECTIONS.values():
        assert all(part in full for part in (rules, ) if part)
    assert "there is no tool for it" in full and r"\frac" in full
    core = routing.system_prompt([])
    assert "evolve_state" not in core and "certify_correlations" not in core and "CHSH: local bound 2" in core


# ------------------------------------------------------------------ the agent node uses the routing
class FakeChain:
    def __init__(self, replies, log, label):
        self.replies, self.log, self.label = replies, log, label

    def invoke(self, messages):
        self.log.append((self.label, messages[0].content))
        return self.replies.pop(0)


class FakeRoutedLLM:
    """Records which tool subset each call used; for_tools() gives the routed chain."""
    def __init__(self, routed_replies, full_replies):
        self.log, self.subsets = [], []
        self._routed = FakeChain(routed_replies, self.log, "routed")
        self._full = FakeChain(full_replies, self.log, "full")

    def for_tools(self, names):
        self.subsets.append(list(names))
        return self._routed

    def invoke(self, messages):
        return self._full.invoke(messages)


def test_agent_node_routes_and_retries_with_all_tools_when_the_model_finds_no_tool():
    import graph
    llm = FakeRoutedLLM([AIMessage(content="I can't do that yet - there is no tool for it.")],
                        [AIMessage(content="Here is the answer.")])
    node = graph.make_agent_node(llm)
    out = node({"messages": [HumanMessage("Create the Werner state with p = 0.3 and give its concurrence.")]})
    assert llm.subsets[0] == routing.CORE_TOOLS
    assert [label for label, _ in llm.log] == ["routed", "full"]
    assert "evolve_state" not in llm.log[0][1] and "TIME EVOLUTION" in llm.log[1][1]
    assert out["messages"][0].content == "Here is the answer."


def test_verifier_named_tools_are_added():
    import graph
    msgs = [HumanMessage("Reconstruct data/x.csv. Is the CHSH violation large?"),
            AIMessage(content="", tool_calls=[{"name": "reconstruct_state_from_counts", "args": {}, "id": "r"}]),
            ToolMessage(content="{}", tool_call_id="r", name="reconstruct_state_from_counts"),
            AIMessage(content="It is large."),
            HumanMessage("VERIFIER: [significance] Call now: certify_correlations(handle='rho_1')")]
    names, _ = graph.route(msgs, 0)
    assert "certify_correlations" in names and "reconstruct_state_from_counts" in names


def test_routed_llm_binds_and_caches_subsets(monkeypatch):
    import llm
    bound = []
    monkeypatch.setattr(llm, "build_providers", lambda tools: (bound.append([t.name for t in tools])
                                                               or [("Fake", None)]))
    r = llm.RoutedLLM(TOOLS)
    a = r.for_tools(["create_state", "compute_measures", "not_a_tool"])
    b = r.for_tools(["compute_measures", "create_state"])
    assert a is b and bound[-1] == ["create_state", "compute_measures"] and len(bound) == 2
