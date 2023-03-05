from typing import Optional, Tuple
import IPython

from reward_machine import RewardMachine


class Builder:
    def __init__(self, appears: frozenset[str]):
        self.appears = appears
        self.transitions: dict[int,
                               dict[frozenset[str], Tuple[int, int]]] = dict()
        self.terminal_states: set[int] = set()

    def build(self) -> RewardMachine:
        return RewardMachine(self.transitions, frozenset(self.appears), frozenset(self.terminal_states))

    def terminal(self, state: int) -> 'Builder':
        self.terminal_states.add(state)
        return self

    def t(self, from_state: int, to_state: int, input_symbol: frozenset[str], output: int) -> 'Builder':
        if from_state in self.terminal_states:
            raise ValueError(
                f'terminal states can\'t have outgoing connections')
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        if input_symbol in self.transitions[from_state]:
            raise ValueError(
                f'{input_symbol} already in state {from_state} ({(from_state, to_state, input_symbol, output)})')
        self.transitions[from_state][input_symbol] = (to_state, output)
        return self
