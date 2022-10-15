from reward_machines.reward_machine import RewardMachine


def load_semantic_map(path: str) -> dict[str, str]:
    raise NotImplementedError()


def describe(semantic_map: dict[str, str], rm: RewardMachine) -> str:
    raise NotImplementedError()
