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


def transitions_to_label(transitions_and_rewards: list[Tuple[str, float]]) -> str:
    transitions = [a[0] for a in transitions_and_rewards]
    rewards = [a[1] for a in transitions_and_rewards]
    return '\n'.join(transitions)


def nodes_and_edges_rm(rm: RewardMachine) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    nodes = set()
    edges: dict[Tuple[int, int], list[Tuple[str, float]]] = dict()
    for (x, y, transition, reward) in rm.desc:
        nodes.add(x)
        nodes.add(y)
        if (x, y) not in edges:
            edges[(x, y)] = [(transition, reward)]
        else:
            edges[(x, y)].append((transition, reward))
    edges_transformed = {key: transitions_to_label(
        val) for key, val in edges.items()}
    return list(nodes), edges_transformed


def nodes_and_edges(x: CompileState | CompileStateDFA) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    if isinstance(x, CompileState):
        return nodes_and_edges_nda(x)
    else:
        return nodes_and_edges_dfa(x)


def node_colors(x: CompileState | CompileStateDFA, nodes: list[int]) -> list[str]:
    return ['green' if node == x.initial.id else 'red' if node == x.terminal.id else 'grey' for node in nodes]


def node_colors_rm(rm: RewardMachine, nodes: list[int]) -> list[str]:
    return ['green' if node == 0 else 'red' if node in rm.terminal_states else 'grey' for node in nodes]


def visualize_compilestate(x: CompileState | CompileStateDFA):
    nodes, edges = nodes_and_edges(x)
    colors = node_colors(x, nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.keys())

    plt.figure(figsize=(6*3, 4*3))

    pos = nx.spectral_layout(G)
    nx.draw(G, pos, node_color=colors,
            with_labels=True, node_size=1000, width=2)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edges, font_color='blue', font_size=16)

    plt.show()


def shift_label_pos(x):
    return [x[0], x[1]+0.1]


def visualize_rm(rm: RewardMachine):
    nodes, edges = nodes_and_edges_rm(rm)
    colors = node_colors_rm(rm, nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.keys())

    plt.figure(figsize=(6*3, 4*3))

    pos = nx.spectral_layout(G)
    nx.draw(G, pos, node_color=colors,
            with_labels=True, node_size=1000, width=2)

    pos = {key: shift_label_pos(value) for key, value in pos.items()}

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edges, font_color='blue', font_size=10)
    plt.show()
