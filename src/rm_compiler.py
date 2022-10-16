import itertools
from typing import Iterable, Tuple

from reward_machine import RewardMachine


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
    def __init__(self, id: frozenset[int] | int, transitions: dict[str, 'RMNodeDFA']):
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
    initial = RMNodeDFA(first_ids, dict())
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
                state = RMNodeDFA(ids, dict())
                state_dict[ids] = state
            if ids not in visited and superstate not in to_visit:
                to_visit.add(superstate)
            current_state.t(i, state)
    assert isinstance(terminal, RMNodeDFA), type(terminal)
    return CompileStateDFA(initial, terminal)


def dfa_to_rm(dfa: CompileStateDFA, appears: frozenset[str]) -> RewardMachine:
    raise NotImplementedError()
