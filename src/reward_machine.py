from typing import Tuple
import IPython

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

    def __call__(self, current_state: int, *inputs: str | frozenset[str]) -> Tuple[int, int] | list[int]:
        if len(inputs) == 1:
            first_input = inputs[0]
            if isinstance(first_input, str):
                input_symbol = frozenset({first_input})
            else:
                assert isinstance(first_input, frozenset)
                input_symbol = first_input
            return self.transition(current_state, input_symbol)
        else:
            rs = list()
            for input_symbol in inputs:
                if isinstance(input_symbol, str):
                    input_symbol = frozenset({input_symbol})
                (current_state, r) = self.transition(
                    current_state, input_symbol)
                rs.append(r)
            return rs
