from cgi import test
from typing import Tuple


class RewardMachine:
    """
    State set: subset of Nat_0, implicitly set by transition and output functions
    Initial state: 0
    Output alphabet: implicitly set by transition and output functions
    Input alphabet: implicitly set by transition function
    """

    def __init__(self):
        return

    def transition(self, current_state: int, input_symbol: frozenset[str]) -> Tuple[int, float]:
        raise NotImplementedError()


class RMUncompiled(RewardMachine):
    def __init__(self, transitions):
        super().__init__()
        self.transitions = transitions

    def transition(self, current_state: int, input_symbol: frozenset[str]) -> Tuple[int, float]:
        for expression, (state, output) in self.transitions[current_state]:
            if expression.test(input_symbol):
                return state, output
        raise ValueError('unrecognized input symbols')


class RMCompiled(RewardMachine):
    def __init__(self, transitions):
        super().__init__()
        self.transitions = transitions

    @staticmethod
    def compile(rm_uncompiled: RMUncompiled) -> 'RMCompiled':
        raise NotImplementedError()
        transitions = None
        return RMCompiled(transitions)
