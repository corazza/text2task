from typing import Tuple
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import IPython
from reward_machine import RewardMachine

from rm_compiler import CompileState, CompileStateDFA, RMNode


def nodes_and_edges_nda(x: CompileState) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    states = set()
    transitions = dict()
    to_visit = [x.initial]
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        states.add(visiting.id)
        for (transition, node) in visiting.transitions:
            if node not in states:
                to_visit.append(node)
            transitions[(visiting.id, node.id)] = transition
    return list(states), transitions


def nodes_and_edges_dfa(x: CompileStateDFA) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    states = set()
    transitions = dict()
    to_visit = [x.initial]
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        states.add(visiting.id)
        for (transition, node) in visiting.transitions.items():
            if node not in states:
                to_visit.append(node)
            transitions[(visiting.id, node.id)] = transition
    return list(states), transitions


def nodes_and_edges(x: CompileState | CompileStateDFA) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    if isinstance(x, CompileState):
        return nodes_and_edges_nda(x)
    else:
        return nodes_and_edges_dfa(x)


def node_colors(x: CompileState | CompileStateDFA, nodes: list[int]) -> list[str]:
    return ['green' if node == x.initial.id else 'red' if node == x.terminal.id else 'grey' for node in nodes]


def visualize_compilestate(x: CompileState | CompileStateDFA):
    nodes, edges = nodes_and_edges(x)
    colors = node_colors(x, nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.keys())

    plt.figure(figsize=(6*3, 4*3))

    # Draw the graph using Matplotlib
    pos = nx.spectral_layout(G)  # Compute the layout of the graph
    nx.draw(G, pos, node_color=colors,
            with_labels=True, node_size=1000, width=2)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edges, font_color='blue', font_size=16)

    plt.show()


def visualize_rm(x: RewardMachine):
    raise NotImplementedError()
