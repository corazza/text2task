import random
from threading import Thread

import easygui
import IPython
import matplotlib.pyplot as plt

import compiler_interface
import model_interface
import rl_agents.qrm
from consts import *
from datasets_common import get_all_terms_from_tag, load_terms
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


def run(config: RunConfig, env, displayer):
    if config.agent_name == 'random':
        raise NotImplementedError()
    elif config.agent_name == 'qrm':
        learn = rl_agents.qrm.learn
    else:
        raise ValueError(f"no known agent '{config.agent_name}'")

    return learn(env=env, add_message=env.desc, total_timesteps=config.step_limit, displayer=displayer)


def demo(Q, env: RMEnvWrapper):
    env.start_render()
    actions = list(range(env.action_space.n))  # type: ignore

    reward_total = 0
    step = 0
    num_episodes = 0
    actions = list(range(env.action_space.n))  # type: ignore

    s = tuple(env.reset())
    env.render(mode='human')
    env.flash_agent()

    while num_episodes < N_DEMO_EPISODES and step < N_DEMO_MAX_STEPS:
        a = rl_agents.qrm.get_best_action(Q, s, actions, DEFAULT_Q_INIT)
        print(env.id_to_action[a])
        # a = random.choice(actions)
        sn, r, done, info = env.step(a)
        sn = tuple(sn)
        env.render(mode='human')

        reward_total += r
        step += 1
        if done:
            num_episodes += 1
            s = tuple(env.reset())
            env.render(mode='human')
            env.flash_agent()
        else:
            s = sn

    env.stop_render()


def main():
    set_all_seeds(42)
    terms = load_terms(DEFAULT_TERMS_PATH)

    config: RunConfig = RunConfig(
        agent_name=DEFAULT_AGENT,
        map_path=DEFAULT_MAP_PATH,
        step_limit=DEFAULT_STEPS,
    )
    map: Map = Map.from_path(config.map_path)
    builder: MapBuilder = MapBuilder.from_map(map)
    appears_rm: frozenset[str] = frozenset(
        get_all_terms_from_tag(terms, 'REQUIRED'))
    map = builder.fill_with_vars_that_dont_appear(appears_rm).build()
    env_initial = MapEnv(map, '', '')

    env_initial.display_once()
    desc: str = easygui.enterbox("Describe the task:")

    generator = model_interface.get_generator(DEFAULT_USE_MODEL_PATH)

    reward_machine: RewardMachine
    src: str
    reward_machine, src = model_interface.get_rm(
        generator, desc, do_cluster=False, displayer=env_initial.display_message)

    # easygui.msgbox(f'training for: {src}', title="Done")

    # if SHOW_COMPLETION:
    #     p = Thread(target=easygui.msgbox, args=(
    #         f'training for: {src}',), kwargs=dict())
    #     p.start()

    builder: MapBuilder = MapBuilder.from_map(map)
    appears_rm: frozenset[str] = reward_machine.appears
    map = builder.fill_with_vars_that_dont_appear(appears_rm).build()
    env = RMEnvWrapper(
        MapEnv(map, src, desc), reward_machine)

    Q = run(config, env, env_initial)

    env_initial.close_display()

    demo(Q, env)


if __name__ == '__main__':
    main()
