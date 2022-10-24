import itertools
from typing import Iterable, Tuple
import IPython

from reward_machine import RewardMachine
from rm_builder import Builder
from transition_parser import parse


class RMNode:
    def __init__(self, id: int, transitions: set[Tuple[str, 'RMNode']]):
        self.transitions = transitions
        self.id = id

    def t(self, input_symbol: str, node: 'RMNode'):
        for (transition, _node) in self.transitions:
            if transition == input_symbol:
                assert transition == '*'
        self.transitions.add((input_symbol, node))

    def appears(self) -> frozenset[str]:
        r = set()
        for (transition, _node) in self.transitions:
            if transition != '*':
                r.add(transition)
        return frozenset(r)

    def __repr__(self):
        return str(self.id)

    def along_symbol(self, input_symbol: str) -> frozenset['RMNode']:
        r = set()
        for (transition, node) in self.transitions:
            if input_symbol == transition:
                r.add(node)
        return frozenset(r)


class RMNodeCreator:
    def __init__(self):
        self.counter = 0

    def new_node(self, transitions: set[Tuple[str, 'RMNode']]) -> RMNode:
        counter = self.counter
        self.counter += 1
        return RMNode(counter, transitions)


class CompileState:
    """NDA representation of an RM"""

    def __init__(self, initial: RMNode, terminal: RMNode):
        self.initial = initial
        self.terminal = terminal

    def state_ids(self, states: Iterable[RMNode]) -> list[int]:
        return list(map(lambda s: s.id, states))

    def accepts(self, input_symbols: str) -> bool:
        current_states = frozenset([self.initial])
        current_states = epsilon(current_states)
        for i in input_symbols:
            current_states = next_superstate(current_states, i)
        return self.terminal in current_states

    def node_by_id(self, id: int | frozenset[int]) -> RMNode:
        to_visit = set([self.initial])
        visited = set([self.initial])
        while len(to_visit) > 0:
            visiting = to_visit.pop()
            visited.add(visiting)
            if visiting.id == id:
                return visiting
            for (_transition, node) in visiting.transitions:
                if node not in visited:
                    to_visit.add(node)
        raise ValueError(f'no node with index {id}')

    def relabel_states(self) -> 'CompileState':
        counter = 0
        to_visit = list([self.initial])
        visited = set([self.initial])
        while len(to_visit) > 0:
            visiting = to_visit.pop(0)
            visiting.id = counter
            counter += 1
            visited.add(visiting)
            for (_transition, node) in visiting.transitions:
                if node not in visited:
                    to_visit.append(node)
        return self

    def __getitem__(self, id: int) -> RMNode:
        return self.node_by_id(id)


def epsilon_once(from_states: frozenset[RMNode]) -> frozenset[RMNode]:
    next_set = frozenset(
        map(lambda s: s.along_symbol('*'), from_states))
    new_states = frozenset(
        itertools.chain.from_iterable(next_set))
    return from_states.union(new_states)


def epsilon(from_states: frozenset[RMNode]) -> frozenset[RMNode]:
    old_states = frozenset()
    new_states = from_states
    while new_states != old_states:
        old_states = new_states
        new_states = epsilon_once(new_states)
    return new_states


def along_symbol(from_states: frozenset[RMNode], input_symbol: str) -> frozenset[RMNode]:
    next_set = frozenset(
        map(lambda s: s.along_symbol(input_symbol), from_states))
    return frozenset(itertools.chain.from_iterable(next_set))


def to_ids(states: frozenset[RMNode]) -> frozenset[int]:
    return frozenset(map(lambda s: s.id, states))


def appears(states: frozenset[RMNode]) -> frozenset[str]:
    r = set()
    for state in states:
        r.add(state.appears())
    return frozenset(itertools.chain.from_iterable(r))


def next_superstate(states: frozenset[RMNode], input_symbol: str) -> frozenset[RMNode]:
    states = along_symbol(states, input_symbol)
    states = epsilon(states)
    return frozenset(states)


class RMNodeDFA:
    def __init__(self, id: int, transitions: dict[str, 'RMNodeDFA']):
        self.transitions = transitions
        self.id = id

    def t(self, input_symbol: str, node: 'RMNodeDFA'):
        assert input_symbol not in self.transitions
        self.transitions[input_symbol] = node

    def __repr__(self):
        return str(self.id)

    def along_symbol(self, input_symbol: str) -> frozenset['RMNodeDFA']:
        if input_symbol in self.transitions:
            return frozenset({self.transitions[input_symbol]})
        else:
            return frozenset()


class CompileStateDFA:
    """DFA representation of an RM"""

    def __init__(self, initial: RMNodeDFA, terminal: RMNodeDFA):
        self.initial = initial
        self.terminal = terminal

    def appears(self) -> Tuple[frozenset[str], dict[int, frozenset[str]]]:
        to_visit = set([self.initial])
        visited = set()
        appears_dfa = set()
        appears_by_state = dict()
        while len(to_visit) > 0:
            visiting = to_visit.pop()
            visited.add(visiting)
            appears = set()
            for (transition, next) in visiting.transitions.items():
                appears.update(parse(transition).appears())
                if next not in visited:
                    to_visit.add(next)
            appears_by_state[visiting.id] = frozenset(appears)
            appears_dfa.update(appears)
        return frozenset(appears_dfa), appears_by_state

    def state_ids(self, states: Iterable[RMNode]) -> list[int]:
        return list(map(lambda s: s.id, states))

    def accepts(self, input_symbols: str) -> bool:
        current_state = self.initial
        for i in input_symbols:
            next_state = current_state.along_symbol(i)
            assert len(next_state) <= 1
            if len(next_state) == 0:
                return False
            current_state = list(next_state)[0]
        return current_state == self.terminal

    def node_by_id(self, id: int | frozenset[int]) -> RMNodeDFA:
        to_visit = set([self.initial])
        visited = set([self.initial])
        while len(to_visit) > 0:
            visiting = to_visit.pop()
            visited.add(visiting)
            if visiting.id == id:
                return visiting
            for (_transition, node) in visiting.transitions.items():
                if node not in visited:
                    to_visit.add(node)
        raise ValueError(f'no node with index {id}')

    def relabel_states(self) -> 'CompileStateDFA':
        counter = 0
        to_visit = list([self.initial])
        visited = set([self.initial])
        while len(to_visit) > 0:
            visiting = to_visit.pop(0)
            visiting.id = counter
            counter += 1
            visited.add(visiting)
            for (_transition, node) in visiting.transitions.items():
                if node not in visited:
                    to_visit.append(node)
        return self

    def __getitem__(self, id: int) -> RMNodeDFA:
        return self.node_by_id(id)


def to_dfa(compiled: CompileState) -> CompileStateDFA:
    first_epsilon = epsilon(frozenset({compiled.initial}))
    first_ids = to_ids(first_epsilon)
    to_visit = set([first_epsilon])
    id_counter = 0
    initial = RMNodeDFA(id_counter, dict())
    id_counter += 1
    state_dict = {to_ids(first_epsilon): initial}
    visited = set({first_ids})
    terminal = None
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        current_ids = to_ids(visiting)
        visited.add(current_ids)
        current_state = state_dict[current_ids]
        if compiled.terminal in visiting:
            assert len(visiting) == 1
            assert terminal == None
            terminal = current_state
        appears_in = appears(visiting)
        for i in appears_in:
            superstate = next_superstate(visiting, i)
            ids = to_ids(superstate)
            if ids in state_dict:
                state = state_dict[ids]
            else:
                state = RMNodeDFA(id_counter, dict())
                id_counter += 1
                state_dict[ids] = state
            if ids not in visited and superstate not in to_visit:
                to_visit.add(superstate)
            current_state.t(i, state)
    assert isinstance(terminal, RMNodeDFA), type(terminal)
    return CompileStateDFA(initial, terminal)


def negate_previous(previous: list[str]) -> str:
    assert len(previous) > 0
    r = f'!('
    for i in range(len(previous) - 1):
        r = f'{r}{previous[i]}|'
    r = f'{r}{previous[-1]})'
    return r


def negate_all(appears: list[str]) -> str:
    assert len(appears) > 0
    r = f'!('
    for i in range(len(appears) - 1):
        r = f'{r}{appears[i]}|'
    r = f'{r}{appears[-1]})'
    return r


def dfa_to_rm(dfa: CompileStateDFA) -> RewardMachine:
    terminal_id = dfa.terminal.id
    appears_dfa, appears_by_state = dfa.appears()
    builder = Builder(terminal_states={terminal_id})
    to_visit = set([dfa.initial])
    visited = set()
    empty_transition = negate_all(list(appears_dfa))
    while len(to_visit) > 0:
        visiting = to_visit.pop()
        visited.add(visiting)
        previous = list()
        self_loops = set()
        if visiting != dfa.terminal:
            for prop in appears_dfa - appears_by_state[visiting.id]:
                self_loops.add((prop, visiting))
            self_loops.add((empty_transition, visiting))
        for (transition, next) in itertools.chain(visiting.transitions.items(), self_loops):
            r = 1 if next == dfa.terminal else 0
            if len(previous) > 0:
                negated_previous = negate_previous(previous)
                new_transition = f'({negated_previous})&({transition})'
            else:
                new_transition = transition
            previous.append(transition)
            builder = builder.t(visiting.id, next.id, new_transition, r)
            if next not in visited:
                to_visit.add(next)
    return builder.build()
