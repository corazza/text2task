import json
import random
from typing import Iterable

import gym
import IPython
import numpy as np
from gym import spaces
from transformers import set_seed

from consts import *
from datasets_common import (create_if_doesnt_exist, get_all_terms_from_tag,
                             load_terms)
from maps import *
from reward_machine import RewardMachine, RewardMachineRunner


class FrozensetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, frozenset):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class Map():
    def __init__(self,
                 content: list[list[frozenset[str]]],
                 size: int,
                 terms: dict[str, list[str]],
                 spawn_type: str,
                 spawn_location: tuple[int, int],
                 ):
        assert spawn_type in ['random', 'location']
        self.content: list[list[frozenset[str]]] = content
        self.terms: dict[str, list[str]] = terms
        self.size: int = size
        assert self.size == len(self.content)
        self.spawn_type: str = spawn_type
        self.spawn_location: tuple[int, int] = spawn_location

    def to_json(self):
        return json.dumps(self.__dict__, cls=FrozensetEncoder)

    def get_new_spawn(self) -> tuple[int, int]:
        if self.spawn_type == 'random':
            return (np.random.randint(self.size), np.random.randint(self.size))
        elif self.spawn_type == 'location':
            return self.spawn_location
        else:
            raise ValueError(f"unknown spawn type '{self.spawn_type}'")

    @staticmethod
    def from_json(code: str) -> 'Map':
        json_dict = json.loads(code)
        return Map(**json_dict)

    @staticmethod
    def from_path(path: str) -> 'Map':
        with open(path, 'r') as f:
            code = f.read()
            return Map.from_json(code)

    @staticmethod
    def pretty_row(row: list[frozenset[str]]) -> str:
        return ' '.join([''.join([var[0] for var in vars]) for vars in row])

    def pretty_rows(self) -> list[str]:
        result: list[str] = []
        for row in self.content:
            result.append(Map.pretty_row(row))
        return result


def action_map(terms: dict[str, list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    all_actions: list[str] = get_all_terms_from_tag(terms, 'ACTION')
    first: dict[str, int] = {action: i for i, action in enumerate(all_actions)}
    second: dict[int, str] = {i: action for action, i in first.items()}
    return first, second


class MapEnv(gym.Env):
    def __init__(self, map: Map, reward_machine: RewardMachine):
        super().__init__()
        self.map: Map = map
        self.reward_machine: RewardMachine = reward_machine
        self.runner: RewardMachineRunner = RewardMachineRunner(
            self.reward_machine)
        self.label_history: list[frozenset[str]] = []
        self.num_actions: int = len(
            get_all_terms_from_tag(self.map.terms, 'ACTION'))
        self.action_to_id: dict[str, int]
        self.id_to_action: dict[int, str]
        self.action_to_id, self.id_to_action = action_map(map.terms)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.MultiDiscrete(
            [self.map.size, self.map.size])

        self.reset()

    def step(self, action):
        action_id: int = action  # type: ignore
        action_name: str = self.id_to_action[action_id]

        horizontal_delta: int = 0
        vertical_delta: int = 0

        if action_name == 'left':
            horizontal_delta -= 1
        elif action_name == 'right':
            horizontal_delta += 1
        elif action_name == 'up':
            vertical_delta -= 1
        elif action_name == 'down':
            vertical_delta += 1

        new_i: int = self.state[0] + vertical_delta
        new_j: int = self.state[1] + horizontal_delta

        new_i = max(0, min(self.map.size-1, new_i))
        new_j = max(0, min(self.map.size-1, new_j))

        if 'wall' not in self.map.content[new_i][new_j]:
            self.state = [new_i, new_j]

        labels: frozenset[str] = frozenset([action_name]).union(
            frozenset(self.map.content[self.state[0]][self.state[1]]))

        self.label_history.append(labels)

        reward = float(self.runner.transition(labels))
        done = reward > 0.0

        return self.state, reward, done, {}

    def reset(self):
        self.state: list[int] = list(self.map.get_new_spawn())
        assert 'wall' not in self.map.content[self.state[0]][self.state[1]]
        self.runner = RewardMachineRunner(self.reward_machine)
        self.label_history: list[frozenset[str]] = []
        return self.state

    def render(self):
        print(self.state)
