import gym
import numpy as np
from gym import spaces

from reward_machine import RewardMachine, RewardMachineRunner


class RMEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_machine: RewardMachine):
        super().__init__(env)

        self.reward_machine: RewardMachine = reward_machine
        self.runner: RewardMachineRunner = RewardMachineRunner(
            self.reward_machine)
        self.label_history: list[frozenset[str]] = []
        self.num_rm_states = len(self.reward_machine.get_nonterminal_states())

        self.observation_dict = spaces.Dict({'features': env.observation_space, 'rm-state': spaces.Box(
            low=0, high=1, shape=(self.num_rm_states,), dtype=np.uint8)})
        flatdim = spaces.flatdim(self.observation_dict)
        s_low = float(env.observation_space.low[0])
        s_high = float(env.observation_space.high[0])
        self.observation_space = spaces.Box(
            low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)

        self.rm_state_features: dict[int, np.ndarray] = dict()
        for u_id in self.reward_machine.get_nonterminal_states():
            u_features: np.ndarray = np.zeros(self.num_rm_states)
            u_features[len(self.rm_state_features)] = 1
            self.rm_state_features[u_id] = u_features

        self.rm_done_feat = np.zeros(self.num_rm_states)

    def reset(self):
        self.obs = self.env.reset()
        self.runner = RewardMachineRunner(self.reward_machine)
        # We use this set to compute RM states that are reachable by the last experience (None means that all of them are reachable!)
        self.valid_states = None
        return self.get_observation(self.obs, self.runner.current_state, False)

    def step(self, action):
        next_obs, original_reward, env_terminated, env_truncated, info = self.env.step(
            action)
        env_done: bool = env_terminated or env_truncated
        assert not env_done  # this is specific to my project

        true_props: frozenset[str] = self.env.get_events()  # type: ignore
        self.crm_params = self.obs, action, next_obs, env_done, true_props
        self.obs = next_obs

        rm_rew: int
        rm_done: bool
        rm_rew, rm_done = self.runner.transition(true_props)

        done = rm_done or env_done
        rm_obs = self.get_observation(
            next_obs, self.runner.current_state, done)

        crm_experience = self._get_crm_experience(*self.crm_params)
        info["crm-experience"] = crm_experience

        return rm_obs, rm_rew, done, info

    def get_observation(self, next_obs, u_id: int, done: bool):
        rm_feat = self.rm_done_feat if done else self.rm_state_features[u_id]
        rm_obs = {'features': next_obs, 'rm-state': rm_feat}
        return spaces.flatten(self.observation_dict, rm_obs)

    def _get_rm_experience(self, rm: RewardMachine, u_id: int, obs, action, next_obs, env_done: bool, true_props: frozenset[str]):
        rm_obs = self.get_observation(obs, u_id, False)
        next_u_id: int
        rm_rew: int
        rm_done: bool
        next_u_id, rm_rew, rm_done = rm.transition(u_id, true_props)
        done: bool = rm_done or env_done
        rm_next_obs = self.get_observation(next_obs, next_u_id, done)
        return (rm_obs, action, rm_rew, rm_next_obs, done), next_u_id

    def _get_crm_experience(self, obs, action, next_obs, env_done: bool, true_props: frozenset[str]):
        reachable_states: set[int] = set()
        experiences = []
        for u_id in self.reward_machine.get_nonterminal_states():
            next_u: int
            exp, next_u = self._get_rm_experience(
                self.reward_machine, u_id, obs, action, next_obs, env_done, true_props)
            reachable_states.add(next_u)
            if self.valid_states is None or u_id in self.valid_states:
                # We only add experience that are possible (i.e., it is possible to reach state u_id given the previous experience)
                experiences.append(exp)

        self.valid_states = reachable_states
        return experiences
