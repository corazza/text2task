from reward_machine import RewardMachine
from rm_ast import CompileStateDFA, RMExpr, RMNodeCreator, to_dfa
import rm_parser


def dfa_to_rm(dfa: CompileStateDFA, appears: frozenset[str]) -> RewardMachine:
    raise NotImplementedError()


def _compile(ast: RMExpr, appears: frozenset[str]) -> RewardMachine:
    node_creator = RMNodeCreator()
    compiled = ast.compile(node_creator).relabel_states()
    dfa = to_dfa(compiled).relabel_states()
    return dfa_to_rm(dfa, appears)


def compile(src: str, appears: frozenset[str]) -> RewardMachine:
    return _compile(rm_parser.parse(src), appears)


def test(src: str, appears: frozenset[str]) -> RewardMachine:
    raise NotImplementedError()
