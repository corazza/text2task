from reward_machine import RewardMachine
from rm_ast import CompileStateDFA, RMExpr, RMNodeCreator, to_dfa
from rm_compiler import dfa_to_rm
import rm_parser


def parse(src: str) -> RMExpr:
    return rm_parser.parse(src)


def get_dfa(src: str) -> CompileStateDFA:
    ast = parse(src)
    node_creator = RMNodeCreator()
    compiled = ast.compile(node_creator).relabel_states()
    dfa = to_dfa(compiled).relabel_states()
    return dfa


def compile(src: str, appears: frozenset[str]) -> RewardMachine:
    dfa = get_dfa(src)
    return dfa_to_rm(dfa, appears)


def test(src: str, appears: frozenset[str]) -> RewardMachine:
    raise NotImplementedError()
