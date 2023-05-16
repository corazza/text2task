import json
import random
from typing import Iterable

import gym
import IPython
import numpy as np
from gym import spaces

from consts import *
from datasets_common import get_all_terms_from_tag
from maps import *


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
            new_loc: tuple[int, int] = (np.random.randint(self.size),
                                        np.random.randint(self.size))
            while 'wall' in self.content[new_loc[0]][new_loc[1]]:
                new_loc = (np.random.randint(self.size),
                           np.random.randint(self.size))
            return new_loc

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


def action_map(terms: dict[str, list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    all_actions: list[str] = get_all_terms_from_tag(terms, 'ACTION')
    first: dict[str, int] = {action: i for i, action in enumerate(all_actions)}
    second: dict[int, str] = {i: action for action, i in first.items()}
    return first, second


class MapEnv(gym.Env):
    def __init__(self, map: Map, src: str):
        super().__init__()
        self.map: Map = map
        self.src: str = src
        self.num_actions: int = len(
            get_all_terms_from_tag(self.map.terms, 'ACTION'))
        self.action_to_id: dict[str, int]
        self.id_to_action: dict[int, str]
        self.action_to_id, self.id_to_action = action_map(map.terms)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=max(  # type: ignore
            [self.map.size, self.map.size]), shape=(2,), dtype=np.uint8)  # type: ignore

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

        reward: int = 0
        terminated: bool = False
        truncated: bool = False

        return self.state, reward, terminated, truncated, {}

    def get_events(self) -> frozenset[str]:
        return self.label_history[-1]

    def reset(self):
        self.state: list[int] = list(self.map.get_new_spawn())
        assert 'wall' not in self.map.content[self.state[0]][self.state[1]]
        self.label_history: list[frozenset[str]] = []
        return self.state

    def single_symbol(self, vars: frozenset[str]) -> str:
        if 'wall' in vars:
            return 'x'
        for var in vars:
            if var in self.src:
                return var[0].upper()
        return ' '

    def pretty_row(self, row: list[frozenset[str]]) -> list[str]:
        return ['.'] + [self.single_symbol(vars) for vars in row] + ['.']

    def render(self):
        pretty_rows: list[list[str]] = [
            self.pretty_row(row) for row in self.map.content]
        pretty_rows[self.state[0]][self.state[1]] = '@'
        print(' '.join(['.']*(self.map.size+2)))
        for row in pretty_rows:
            print(' '.join(row))
        print(' '.join(['.']*(self.map.size+2)))
