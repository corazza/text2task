from reward_machines.reward_machine import RewardMachine
import IPython


def load_semantic_map(path: str) -> dict[str, str]:
    sm = dict()
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            k, v = line.split(' ')
            sm[k] = v
    return sm


def describe(semantic_map: dict[str, str], rm: RewardMachine) -> str:
    IPython.embed()
    raise NotImplementedError()
