import itertools
from typing import Iterable, Tuple
import IPython
import copy

from reward_machine import RewardMachine
from rm_builder import Builder


class NFANode:
    def __init__(self, id: int):
        self.transitions: dict[frozenset[str], set['NFANode']] = dict()
        self.id: int = id

    def t(self, input_symbol: frozenset[str], node: 'NFANode'):
        if '*' in input_symbol:
            assert input_symbol == frozenset({'*'})
        if input_symbol not in self.transitions:
            self.transitions[input_symbol] = set()
        self.transitions[input_symbol].add(node)

    def along_symbol(self, input_symbol: frozenset[str]) -> frozenset['NFANode']:
        r: set['NFANode'] = set()
        for transition, node in self.transitions.items():
            if input_symbol == transition:
                r.update(node)
        return frozenset(r)


class DFANode:
    def __init__(self, id: int):
        self.transitions: dict[frozenset[str], 'DFANode'] = dict()
        self.id: int = id

    def t(self, input_symbol: frozenset[str], node: 'DFANode'):
        assert '*' not in input_symbol
        assert input_symbol not in self.transitions, f'failed: {input_symbol} not in {self.transitions}'
        self.transitions[input_symbol] = node

    def along_symbol(self, input_symbol: frozenset[str]) -> frozenset['DFANode']:
        if input_symbol in self.transitions:
            return frozenset({self.transitions[input_symbol]})
        else:
            return frozenset()


class NodeCreator:
    def __init__(self, appears: frozenset[str]):
        self.counter: int = 0
        self.appears: frozenset[str] = appears

    def convert_node_to_nfa(self, dfa_node: DFANode) -> NFANode:
        return NFANode(dfa_node.id)

    def new_nfa_sink(self):  # TODO abstract somehow
        node = self.new_nfa_node()
        for input_symbol in generate_inputs(self.appears):
            node.t(input_symbol, node)
        return node

    def new_dfa_sink(self):
        node = self.new_dfa_node()
        for input_symbol in generate_inputs(self.appears):
            node.t(input_symbol, node)
        return node

    def new_dfa_node(self) -> DFANode:
        counter = self.counter
        self.counter += 1
        return DFANode(counter)

    def new_nfa_node(self) -> NFANode:
        counter = self.counter
        self.counter += 1
        return NFANode(counter)


# TODO abstract a CompileState base class
class CompileStateNFA:
    """NDA representation of an RM"""

    def __init__(self, initial: NFANode, terminal_states: set[NFANode]):
        self.initial: NFANode = initial
        self.terminal_states: set[NFANode] = terminal_states

    def relabel_states(self) -> 'CompileStateNFA':
        counter = 0
        to_visit: list[NFANode] = [self.initial]
        visited: set[NFANode] = set([self.initial])
        while len(to_visit) > 0:
            visiting: NFANode = to_visit.pop(0)
            if visiting.id != -1:
                visiting.id = counter
            counter += 1
            visited.add(visiting)
            for _transition, nodes in visiting.transitions.items():
                for node in nodes:
                    if node not in visited and node not in to_visit:
                        to_visit.append(node)
        return self

    def has_terminals(self, ids: frozenset[int]) -> bool:
        for terminal in self.terminal_states:
            if terminal.id in ids:
                return True
        return False


class CompileStateDFA:
    """DFA representation of an RM"""

    def __init__(self, initial: DFANode, terminal_states: set[DFANode]):
        self.initial = initial
        self.terminal_states = terminal_states

    def relabel_states(self) -> 'CompileStateDFA':
        counter = 0
        to_visit: list[DFANode] = [self.initial]
        visited: set[DFANode] = set([self.initial])
        while len(to_visit) > 0:
            visiting: DFANode = to_visit.pop(0)
            if visiting.id != -1:
                visiting.id = counter
            counter += 1
            visited.add(visiting)
            for (_transition, node) in visiting.transitions.items():
                if node not in visited and node not in to_visit:
                    to_visit.append(node)
        return self


def epsilon_once(from_states: frozenset[NFANode]) -> frozenset[NFANode]:
    next_set = frozenset(
        [s.along_symbol(frozenset({'*'})) for s in from_states])
    new_states = frozenset(
        itertools.chain.from_iterable(next_set))
    return from_states.union(new_states)


def epsilon(from_states: frozenset[NFANode]) -> frozenset[NFANode]:
    old_states = frozenset()
    new_states = from_states
    while new_states != old_states:
        old_states = new_states
        new_states = epsilon_once(new_states)
    return new_states


def along_symbol(from_states: frozenset[NFANode], input_symbol: frozenset[str]) -> frozenset[NFANode]:
    next_set = frozenset([s.along_symbol(input_symbol) for s in from_states])
    return frozenset(itertools.chain.from_iterable(next_set))


def next_superstate(states: frozenset[NFANode], input_symbol: frozenset[str]) -> frozenset[NFANode]:
    states = along_symbol(states, input_symbol)
    states = epsilon(states)
    return frozenset(states)


def generate_inputs(appears: frozenset[str]) -> Iterable[frozenset[str]]:
    for i in range(len(appears)+1):
        for subset in itertools.combinations(appears, i):
            yield frozenset(subset)


def to_dfa(compiled_nfa: CompileStateNFA, node_creator: NodeCreator) -> CompileStateDFA:
    first_epsilon: frozenset[NFANode] = epsilon(
        frozenset({compiled_nfa.initial}))
    to_visit_nfa: set[frozenset[NFANode]] = set([first_epsilon])
    initial_dfa: DFANode = node_creator.new_dfa_node()
    terminal_states_dfa: set[DFANode] = set()
    state_dict: dict[frozenset[NFANode], DFANode] = {
        first_epsilon: initial_dfa}
    visited_nfa: set[frozenset[NFANode]] = set()
    while len(to_visit_nfa) > 0:
        visiting_nfa_superstate: frozenset[NFANode] = to_visit_nfa.pop()
        visited_nfa.add(visiting_nfa_superstate)
        current_dfa_state: DFANode = state_dict[visiting_nfa_superstate]
        for nfa_state in visiting_nfa_superstate:
            if nfa_state in compiled_nfa.terminal_states:
                terminal_states_dfa.add(current_dfa_state)
        for i in generate_inputs(node_creator.appears):
            next_nfa_superstate: frozenset[NFANode] = next_superstate(
                visiting_nfa_superstate, i)
            if next_nfa_superstate in state_dict:
                state_dfa: DFANode = state_dict[next_nfa_superstate]
            else:
                state_dfa: DFANode = node_creator.new_dfa_node()
                state_dict[next_nfa_superstate] = state_dfa
            if next_nfa_superstate not in visited_nfa:
                to_visit_nfa.add(next_nfa_superstate)
            current_dfa_state.t(i, state_dfa)
    return CompileStateDFA(initial_dfa, terminal_states_dfa)


def to_nfa(dfa: CompileStateDFA, node_creator: NodeCreator):
    """Reuses state ids, will not create new states"""
    terminal_states_nfa: set[NFANode] = set()
    to_visit_dfa: list[DFANode] = [dfa.initial]
    visited_dfa: set[DFANode] = set()
    dfa_id_to_nfa_node: dict[int, NFANode] = dict()
    while len(to_visit_dfa) > 0:
        visiting_dfa: DFANode = to_visit_dfa.pop()
        visited_dfa.add(visiting_dfa)
        if visiting_dfa.id not in dfa_id_to_nfa_node:
            nfa_visiting: NFANode = node_creator.convert_node_to_nfa(
                visiting_dfa)
            dfa_id_to_nfa_node[visiting_dfa.id] = nfa_visiting
        else:
            nfa_visiting: NFANode = dfa_id_to_nfa_node[visiting_dfa.id]
        if visiting_dfa.id == dfa.initial.id:
            initial = nfa_visiting
        if visiting_dfa in dfa.terminal_states:
            terminal_states_nfa.add(nfa_visiting)
        dfa_id_to_nfa_node[visiting_dfa.id] = nfa_visiting
        for input_symbol, dfa_child in visiting_dfa.transitions.items():
            if dfa_child.id not in dfa_id_to_nfa_node:
                nfa_child = node_creator.convert_node_to_nfa(dfa_child)
                dfa_id_to_nfa_node[dfa_child.id] = nfa_child
            else:
                nfa_child = dfa_id_to_nfa_node[dfa_child.id]
            nfa_visiting.t(input_symbol, nfa_child)
            if dfa_child not in visited_dfa:
                to_visit_dfa.append(dfa_child)
    return CompileStateNFA(initial, terminal_states_nfa)  # type: ignore


def nfa_union(nfas: list[CompileStateNFA], node_creator: NodeCreator) -> CompileStateNFA:
    initial = node_creator.new_nfa_node()
    terminal = node_creator.new_nfa_node()
    for c in nfas:
        initial.t(frozenset({'*'}), c.initial)
        for c_terminal in c.terminal_states:
            c_terminal.t(frozenset({'*'}), terminal)
    return CompileStateNFA(initial, {terminal})


def dfa_complement(dfa_original: CompileStateDFA) -> CompileStateDFA:
    dfa: CompileStateDFA = copy.copy(dfa_original)
    to_visit: set[DFANode] = set([dfa.initial])
    new_terminal: set[DFANode] = set()
    visited: set[DFANode] = set()
    while len(to_visit) > 0:
        visiting: DFANode = to_visit.pop()
        visited.add(visiting)
        if visiting not in dfa.terminal_states:
            new_terminal.add(visiting)
        for _transition, node in visiting.transitions.items():
            if node not in visited:
                to_visit.add(node)
    return CompileStateDFA(dfa.initial, new_terminal)


def nfa_complement(nfa: CompileStateNFA, node_creator: NodeCreator) -> CompileStateNFA:
    dfa = to_dfa(nfa, node_creator)
    return to_nfa(dfa_complement(dfa), node_creator)


def dfa_to_rm(dfa: CompileStateDFA, appears: frozenset[str]) -> RewardMachine:
    """Terminal in DFA: positive reward. Terminal in RM: end simulation."""
    builder: Builder = Builder(appears)
    to_visit: set[DFANode] = set([dfa.initial])
    visited: set[DFANode] = set()
    while len(to_visit) > 0:
        visiting: DFANode = to_visit.pop()
        visited.add(visiting)
        terminal = True
        for transition, dfa_child in visiting.transitions.items():
            if dfa_child != visiting:
                terminal = False  # nodes which only have themselves as children also terminate the task
            if dfa_child in dfa.terminal_states:
                r = 1
            else:
                r = 0
            builder = builder.t(visiting.id, dfa_child.id, transition, r)
            if dfa_child not in visited:
                to_visit.add(dfa_child)
        if terminal:
            builder = builder.terminal(visiting.id)
    return builder.build()
