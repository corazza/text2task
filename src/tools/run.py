import IPython

import compiler_interface
import rl_agents.qrm
from consts import *
from maps import Map, MapEnv
from model_interface import answer_query_single, query_loop
from reward_machine import RewardMachine
from rl_agents.random_agent import RandomAgent
from rm_env import RMEnvWrapper
from tools.map_builder import MapBuilder
from util import set_all_seeds


class RunConfig():
    def __init__(self,
                 agent_name: str,
                 map_path: str,
                 step_limit: int):
        self.agent_name: str = agent_name
        self.map_path: str = map_path
        self.step_limit: int = step_limit


def run(config: RunConfig, env):
    if config.agent_name == 'random':
        raise NotImplementedError()
    elif config.agent_name == 'qrm':
        learn = rl_agents.qrm.learn
    else:
        raise ValueError(f"no known agent '{config.agent_name}'")

    return learn(env=env, total_timesteps=config.step_limit)


def demo(Q, env):
    actions = list(range(env.action_space.n))

    reward_total = 0
    step = 0
    num_episodes = 0
    actions = list(range(env.action_space.n))

    s = tuple(env.reset())
    env.render()
    while True:
        a = rl_agents.qrm.get_best_action(Q, s, actions, DEFAULT_Q_INIT)
        sn, r, done, info = env.step(a)
        sn = tuple(sn)
        env.render()
        reward_total += r
        step += 1
        if done:
            num_episodes += 1
            break
        s = sn


def main():
    set_all_seeds(42)
    reward_machine: RewardMachine
    desc: str
    src: str
    reward_machine, desc, src = query_loop(DEFAULT_USE_MODEL_PATH)
    config: RunConfig = RunConfig(
        agent_name=DEFAULT_AGENT,
        map_path=DEFAULT_MAP_PATH,
        step_limit=DEFAULT_STEPS,
    )
    map: Map = Map.from_path(config.map_path)
    builder: MapBuilder = MapBuilder.from_map(map)
    appears_rm: frozenset[str] = reward_machine.appears
    map = builder.fill_with_vars_that_dont_appear(appears_rm).build()
    env = RMEnvWrapper(MapEnv(map, src), reward_machine)

    Q = run(config, env)
    print('running demo...')
    demo(Q, env)


if __name__ == '__main__':
    main()
