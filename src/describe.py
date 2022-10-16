from reward_machine import RewardMachine
import IPython

from rm_ast import RMExpr


def load_semantic_map(path: str) -> dict[str, str]:
    sm = dict()
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            k, v = line.split(' ')
            sm[k] = v
    return sm


def describe(semantic_map: dict[str, str], expr: RMExpr) -> str:
    IPython.embed()
    raise NotImplementedError()
