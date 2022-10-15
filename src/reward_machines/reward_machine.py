from typing import Tuple

from util import get_one


class RewardMachine:
    """
    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function
    """

    def __init__(self, transitions, appears, terminal_states):
        super().__init__()
        self.transitions = transitions
        self.appears = appears
        self.terminal_states = terminal_states

    def transition(self, current_state: int, input_symbol: frozenset[str]) -> Tuple[int, float]:
        input_symbol = frozenset(self.appears.intersection(input_symbol))
        if input_symbol not in self.transitions[current_state]:
            return (get_one(self.terminal_states), 0)
        return self.transitions[current_state][input_symbol]

    def __call__(self, current_state: int, input_symbol_str: str) -> Tuple[int, float]:
        input_symbol = frozenset(input_symbol_str)
        return self.transition(current_state, input_symbol)
