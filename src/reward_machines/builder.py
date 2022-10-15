from expression import Expression, compile
from parser import parse
from reward_machines.reward_machine import RewardMachine
from util import powerset


def compute_terminal_states(transitions: dict) -> frozenset[int]:
    terminal_states = set()
    for state in transitions:
        for intp in transitions[state]:
            reaching = transitions[state][intp][0]
            if reaching not in transitions:
                terminal_states.add(reaching)
    return frozenset(terminal_states)


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

    def t(self, from_state: int, to_state: int, expr_src: str, output: float):
        expr = parse(expr_src)
        expr_appears = expr.appears()
        if not self.preset_appears:
            self.appears.update(expr_appears)
        else:
            assert expr_appears <= self.appears
        self.commands.append((from_state, to_state, expr, output))
        return self

    def _t(self, from_state: int, to_state: int, expr: Expression, output: float):
        compiled = compile(expr, appears=self.appears)
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        for intp in compiled:
            if intp in self.transitions[from_state]:
                raise ValueError(f'{intp} already in state {to_state}')
            self.transitions[from_state][intp] = (to_state, output)

    def build(self):
        for c in self.commands:
            self._t(c[0], c[1], c[2], c[3])
        for intp in powerset(frozenset(self.appears)):
            for state in self.transitions.keys():
                if intp not in self.transitions[state]:
                    raise ValueError(
                        f'intp {intp} missing from transitions out of {state}')
        assert self.terminal_states == compute_terminal_states(
            self.transitions)
        for terminal_state in self.terminal_states:
            assert terminal_state not in self.transitions
        return RewardMachine(self.transitions, self.appears, self.terminal_states)
