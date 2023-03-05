from typing import Tuple
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib
import matplotlib.pyplot as plt
import IPython

from reward_machine import RewardMachine
import regex_ast
from regex_ast import RMExpr
from regex_compiler import CompileStateNFA, CompileStateDFA, NFANode, DFANode
from hierarchy_pos import hierarchy_pos


def input_symbol_to_str(input_symbol: frozenset[str]) -> str:
    vars = ', '.join(input_symbol)
    return '{' + vars + '}'


def nodes_and_edges_nda(x: CompileStateNFA) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    transitions: dict[Tuple[int, int], str] = dict()
    to_visit: set[NFANode] = set([x.initial])
    visited: set[NFANode] = set()
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        visited.add(visiting)
        for transition, nodes in visiting.transitions.items():
            t = input_symbol_to_str(transition)
            for node in nodes:
                if node not in visited:
                    to_visit.add(node)
                if (visiting.id, node.id) in transitions:
                    transitions[(visiting.id, node.id)] += f' | {t}'
                else:
                    transitions[(visiting.id, node.id)] = f'{t}'
    return [x.id for x in visited], transitions


def nodes_and_edges_dfa(x: CompileStateDFA) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    transitions: dict[Tuple[int, int], str] = dict()
    to_visit: set[DFANode] = set([x.initial])
    visited: set[DFANode] = set()
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        visited.add(visiting)
        for (transition, node) in visiting.transitions.items():
            t = input_symbol_to_str(transition)
            if node not in visited:
                to_visit.add(node)
            if (visiting.id, node.id) in transitions:
                transitions[(visiting.id, node.id)] += f' | {t}'
            else:
                transitions[(visiting.id, node.id)] = f'{t}'
    return [x.id for x in visited], transitions


def transitions_to_label(transitions_and_rewards: list[Tuple[frozenset[str], int]]) -> str:
    labels = [
        f'{input_symbol_to_str(a[0])}: {a[1]}' for a in transitions_and_rewards]
    return ' | '.join(labels)


def nodes_and_edges_rm(rm: RewardMachine) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    nodes: set[int] = set()
    edges: dict[Tuple[int, int], list[Tuple[frozenset[str], int]]] = dict()
    for from_state, transitions in rm.transitions.items():
        nodes.add(from_state)
        for transition, (to_state, reward) in transitions.items():
            nodes.add(to_state)
            if (from_state, to_state) not in edges:
                edges[(from_state, to_state)] = [(transition, reward)]
            else:
                edges[(from_state, to_state)].append((transition, reward))
    edges_transformed = {key: transitions_to_label(
        val) for key, val in edges.items()}
    return list(nodes), edges_transformed


def nodes_and_edges_and_labels_ast(ast: RMExpr) -> Tuple[list[int], dict[Tuple[int, int], str], dict[int, str]]:
    child_id: int = 0
    nodes: list[int] = []
    edges: dict[Tuple[int, int], str] = dict()
    node_labels: dict[int, str] = dict()
    to_visit: list[Tuple[int, RMExpr]] = [(0, ast)]
    while len(to_visit) > 0:
        id, visiting = to_visit.pop(0)
        child_id += 1
        if isinstance(visiting, regex_ast.Symbol):
            node_labels[id] = str(visiting)
            nodes.append(id)
        elif isinstance(visiting, regex_ast.Matcher):
            node_labels[id] = str(visiting)
            nodes.append(id)
        elif isinstance(visiting, regex_ast.RMExprSing):
            node_labels[id] = visiting.name
            edges[(id, child_id)] = ''
            nodes.append(id)
            to_visit.append((child_id, visiting.child))
        else:
            assert isinstance(visiting, regex_ast.RMExprMul), f'{visiting}'
            node_labels[id] = visiting.name
            nodes.append(id)
            for child in visiting.exprs:
                edges[(id, child_id)] = ''
                to_visit.append((child_id, child))
                child_id += 1
    return nodes, edges, node_labels


def nodes_and_edges(x: CompileStateNFA | CompileStateDFA) -> Tuple[list[int], dict[Tuple[int, int], str]]:
    if isinstance(x, CompileStateNFA):
        return nodes_and_edges_nda(x)
    else:
        return nodes_and_edges_dfa(x)


def node_color_cs(cs: CompileStateNFA | CompileStateDFA, node: int) -> str:
    terminal_ids = {x.id for x in cs.terminal_states}
    if node == cs.initial.id and node in terminal_ids:
        return 'yellow'
    elif node == cs.initial.id:
        return 'green'
    elif node in terminal_ids:
        return 'red'
    else:
        return 'grey'


def node_colors(cs: CompileStateNFA | CompileStateDFA, nodes: list[int]) -> list[str]:
    return [node_color_cs(cs, node) for node in nodes]


def node_color_rm(rm: RewardMachine, node: int) -> str:
    if node == 0 and node in rm.terminal_states:
        return 'yellow'
    elif node == 0:
        return 'green'
    elif node in rm.terminal_states:
        return 'red'
    else:
        return 'grey'


def node_colors_rm(rm: RewardMachine, nodes: list[int]) -> list[str]:
    return [node_color_rm(rm, node) for node in nodes]


def node_colors_ast(ast: RMExpr, nodes: list[int]) -> list[str]:
    return ['green' if node == 0 else 'gray' for node in nodes]


def visualize_ast(ast: RMExpr, title: str):
    nodes, edges, node_labels = nodes_and_edges_and_labels_ast(ast)
    colors = node_colors_ast(ast, nodes)
    draw_graph(title, nodes, node_labels, edges, colors)


def default_node_labels(nodes: list[int]) -> dict[int, str]:
    return {x: str(x) for x in nodes}


def visualize_compilestate(x: CompileStateNFA | CompileStateDFA, title: str):
    nodes, edges = nodes_and_edges(x)
    node_labels = default_node_labels(nodes)
    colors = node_colors(x, nodes)
    draw_graph(title, nodes, node_labels, edges, colors)


def visualize_rm(rm: RewardMachine, title: str):
    nodes, edges = nodes_and_edges_rm(rm)
    node_labels = default_node_labels(nodes)
    colors = node_colors_rm(rm, nodes)
    draw_graph(title, nodes, node_labels, edges, colors)


def draw_graph(title: str, nodes: list[int], node_labels: dict[int, str], edges: dict[Tuple[int, int], str], colors: list[str]):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.keys())
    pos = nx.nx_pydot.graphviz_layout(G, 'dot')
    plt.figure(title, figsize=(6*3, 4*3))
    nx.draw(G, pos, node_color=colors, alpha=0.5, labels=node_labels,
            with_labels=True, node_size=1000, width=2)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edges, bbox=dict(facecolor='none', edgecolor='none'), font_color='blue', font_size=16)
    plt.show()
