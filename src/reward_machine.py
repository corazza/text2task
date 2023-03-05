from typing import Iterable, Sequence, Tuple
import IPython


class RewardMachine:
    """
    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function
    """

    def __init__(self, transitions: dict[int, dict[frozenset[str], Tuple[int, int]]], appears: frozenset[str], terminal_states: frozenset[int]):
        super().__init__()
        self.transitions = transitions
        self.appears = appears
        self.terminal_states = terminal_states
        self.desc = ''

    def transition(self, current_state: int, input_symbol: frozenset[str]) -> Tuple[int, int]:
        input_symbol = frozenset(self.appears.intersection(input_symbol))
        if current_state not in self.transitions:
            assert current_state in self.terminal_states
            return (current_state, 0)
        if input_symbol not in self.transitions[current_state]:
            return (current_state, 0)
        return self.transitions[current_state][input_symbol]

    def multiple_transitions(self, current_state: int, input_symbols: list[frozenset[str]]) -> list[int]:
        """Used for demos/testing"""
        rs = []
        for input_symbol in input_symbols:
            current_state, r = self.transition(current_state, input_symbol)
            rs.append(r)
        return rs

    def __call__(self, *input_symbols: Iterable[str]) -> list[int]:
        """Just a nicer interface for RewardMachine.multiple_transitions"""
        return self.multiple_transitions(0, [frozenset(x) for x in input_symbols])
