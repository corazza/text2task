import numpy as np
from typing import Iterable

from regex_compiler import CompileStateDFA, DFANode
from consts import *


def terminating(a: CompileStateDFA, state: DFANode) -> bool:
    return state in a.terminal_states


def equivalent_on(a: CompileStateDFA, b: CompileStateDFA, inputs: list[frozenset[str]]) -> bool:
    current_state_a: DFANode = a.initial
    current_state_b: DFANode = b.initial
    if terminating(a, current_state_a) != terminating(b, current_state_b):
        return False
    for input_symbol in inputs:
        current_state_a = current_state_a.transitions[input_symbol]
        current_state_b = current_state_b.transitions[input_symbol]
        if terminating(a, current_state_a) != terminating(b, current_state_b):
            return False
    return True


def generate_inputs(appears: frozenset[str], test_length: int, num_tests: int) -> Iterable[list[frozenset[str]]]:
    for i in range(num_tests):
        inputs: list[frozenset[str]] = []
        for j in range(test_length):
            if np.random.random() < REWRITE_VALIDATION_EMPTY_PROB:
                input_symbol: frozenset[str] = frozenset()
            else:
                n_elements = np.random.randint(0, len(appears)+1)
                input_symbol: frozenset[str] = frozenset(np.random.choice(
                    list(appears), n_elements))
            inputs.append(input_symbol)
        yield inputs


def equivalent(appears: frozenset[str], a: CompileStateDFA, b: CompileStateDFA, test_length: int, num_tests: int) -> list[list[frozenset[str]]]:
    not_equivalent: list[list[frozenset[str]]] = []
    for inputs in generate_inputs(appears, test_length, num_tests):
        if not equivalent_on(a, b, inputs):
            not_equivalent.append(inputs)
    return not_equivalent
