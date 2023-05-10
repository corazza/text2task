import compiler_interface
from consts import *
from maps import Map, MapEnv
from reward_machine import RewardMachine
from rl_agents.random_agent import RandomAgent


class RunConfig():
    def __init__(self,
                 agent_name: str,
                 map_path: str,
                 num_episodes: int,
                 step_limit: int):
        self.agent_name: str = agent_name
        self.map_path: str = map_path
        self.num_episodes: int = num_episodes
        self.step_limit: int = step_limit


def run(config: RunConfig, src: str):
    map: Map = Map.from_path(config.map_path)
    reward_machine: RewardMachine = compiler_interface.compile(src)
    env = MapEnv(map, reward_machine)

    if config.agent_name == 'random':
        agent = RandomAgent(env.action_space)
    else:
        raise ValueError(f"no known agent '{config.agent_name}'")

    for i_episode in range(config.num_episodes):
        observation = env.reset()
        for t in range(config.step_limit):
            env.render()
            action = agent.choose_action(observation)
            observation, reward, done, info = env.step(action)

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()


def main():
    config: RunConfig = RunConfig(
        agent_name=DEFAULT_AGENT,
        map_path=DEFAULT_MAP_PATH,
        num_episodes=DEFAULT_EPISODES,
        step_limit=DEFAULT_STEPS,
    )
    run(config, '(.)* > office > (.)* > mail')


if __name__ == '__main__':
    main()
