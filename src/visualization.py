from typing import Tuple
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import IPython

from rm_compiler import CompileState, RMNode


def nodes_and_edges(x: CompileState) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    states = set()
    transitions = dict()
    to_visit = []
    to_visit.append(x.initial)
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        states.add(visiting.id)
        for (transition, node) in visiting.transitions:
            if node not in states:
                to_visit.append(node)
            transitions[(visiting.id, node.id)] = transition
    return list(states), transitions


def node_colors(x: CompileState, nodes: list[int]) -> list[str]:
    return ['green' if node == x.initial.id else 'red' if node == x.terminal.id else 'blue' for node in nodes]


def visualize_compilestate(x: CompileState):
    nodes, edges = nodes_and_edges(x)
    colors = node_colors(x, nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.keys())

    plt.figure(figsize=(6*3, 4*3))

    # Draw the graph using Matplotlib
    pos = nx.spring_layout(G)  # Compute the layout of the graph
    nx.draw(G, pos, node_color=colors,
            with_labels=True, node_size=1000, width=2)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edges, font_color='red', font_size=16)

    plt.show()
