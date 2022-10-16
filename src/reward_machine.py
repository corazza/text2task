from typing import Tuple

from rm_util import get_one


class RewardMachine:
    """
    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function
    """

    def __init__(self, transitions: dict[int, dict[frozenset[str], Tuple[int, int]]], appears: frozenset[str], terminal_states: frozenset[str], desc: list[str]):
        super().__init__()
        self.transitions = transitions
        self.appears = appears
        self.terminal_states = terminal_states
        self.desc = desc

    def transition(self, current_state: int, input_symbol: frozenset[str]) -> Tuple[int, int]:
        if current_state in self.terminal_states:
            return (current_state, 0)
        input_symbol = frozenset(self.appears.intersection(input_symbol))
        if input_symbol not in self.transitions[current_state]:
            return (get_one(self.terminal_states), 0)
        return self.transitions[current_state][input_symbol]

    def __call__(self, current_state: int, inputs: str | list[str]) -> Tuple[int, int] | list[int]:
        if isinstance(inputs, str):
            input_symbol = frozenset(inputs)
            return self.transition(current_state, input_symbol)
        else:
            rs = list()
            for i in inputs:
                input_symbol = frozenset(i)
                (current_state, r) = self.transition(
                    current_state, input_symbol)
                rs.append(r)
            return rs
