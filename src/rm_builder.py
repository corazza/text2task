from typing import Tuple
import IPython

from reward_machine import RewardMachine
from transition_ast import TExp, compile
from transition_parser import parse


def compute_terminal_states(transitions: dict) -> frozenset[int]:
    terminal_states = set()
    for state in transitions:
        for intp in transitions[state]:
            reaching = transitions[state][intp][0]
            if reaching not in transitions:
                terminal_states.add(reaching)
    return frozenset(terminal_states)


def describe_command(c: Tuple[int, int, TExp, int, str]):
    return (c[0], c[1], c[4], c[3])


class Builder:
    def __init__(self, appears=None, terminal_states=set()):
        self.transitions = dict()
        if appears != None:
            self.appears = appears
            self.preset_appears = True
        else:
            self.appears = set()
            self.preset_appears = False
        self.terminal_states = frozenset(terminal_states)
        self.commands = list()

    def t(self, from_state: int, to_state: int, expr_src: str, output: int) -> 'Builder':
        expr = parse(expr_src)
        expr_appears = expr.appears()
        if not self.preset_appears:
            self.appears.update(expr_appears)
        else:
            assert expr_appears <= self.appears
        self.commands.append((from_state, to_state, expr, output, expr_src))
        return self

    def build(self) -> RewardMachine:
        for c in self.commands:
            self._t(c[0], c[1], c[2], c[3])
        # this no longer works in the presence of loops
        # assert self.terminal_states == compute_terminal_states(
        #     self.transitions)
        # for terminal_state in self.terminal_states:
        #     assert terminal_state not in self.transitions
        return RewardMachine(self.transitions, frozenset(self.appears), self.terminal_states, self.describe())

    def describe(self) -> list[Tuple[int, int, str, float]]:
        return list(map(describe_command, self.commands))

    def _t(self, from_state: int, to_state: int, expr: TExp, output: int) -> None:
        compiled = compile(expr, appears=self.appears)
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        for intp in compiled:
            if intp in self.transitions[from_state]:
                raise ValueError(
                    f'{intp} already in state {from_state} ({(from_state, to_state, expr, output)})')
            self.transitions[from_state][intp] = (to_state, output)
