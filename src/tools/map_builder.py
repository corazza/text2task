import random
from pathlib import Path
from typing import Iterable

import IPython
import numpy as np

from consts import *
from datasets_common import (create_if_doesnt_exist, get_all_terms_from_tag,
                             load_terms)
from maps import *
from util import set_all_seeds


class MapBuilder():
    def __init__(self, size: int, terms: dict[str, list[str]]):
        self.size: int = size
        self.terms = terms
        self.content: list[list[set[str]]] = [
            [set() for j in range(size)] for i in range(size)]
        self.spawn_type: str = 'random'
        self.spawn_location: tuple[int, int] = (0, 0)

    @staticmethod
    def from_map(map: Map) -> 'MapBuilder':
        result: MapBuilder = MapBuilder(map.size, map.terms)
        result.content = [[set(vars) for vars in row] for row in map.content]
        result.spawn_type = map.spawn_type
        result.spawn_location = map.spawn_location
        return result

    def add(self, var: str, i: int, j: int):
        self.content[i][j].add(var)

    def remove(self, var: str, i: int, j: int):
        self.content[i][j].remove(var)

    def clear_nonarea(self, i: int, j: int):
        all_areas = get_all_terms_from_tag(self.terms, 'AREA')
        to_remove: set[str] = set()
        for var in self.content[i][j]:
            if var not in all_areas:
                to_remove.add(var)
        for var in to_remove:
            self.remove(var, i, j)

    def add_area(self, var: str, ij: tuple[int, int], hw: tuple[int, int]):
        i, j = ij
        for h in range(hw[0]):
            for w in range(hw[1]):
                self.add(var, i+h, j+w)

    def remove_area(self, var: str, ij: tuple[int, int], hw: tuple[int, int]):
        i, j = ij
        for h in range(hw[0]):
            for w in range(hw[1]):
                self.remove(var, i+h, j+w)

    def appears(self) -> frozenset[str]:
        result: set[str] = set()
        for row in self.content:
            for vars in row:
                result.update(vars)
        return frozenset(result)

    def fill_with_vars_that_dont_appear(self, vars: frozenset[str], times: int = 1):
        appears: frozenset[str] = self.appears()
        to_add: frozenset[str] = frozenset(
            [var for var in vars if var not in appears])
        for var in to_add:
            self.add_to_random_location(var)
        return self

    def free_to_add_at(self, i: int, j: int) -> bool:
        return 'wall' not in self.content[i][j]

    def get_random_free_coordinates(self) -> tuple[int, int]:
        i: int = np.random.randint(0, self.size)
        j: int = np.random.randint(0, self.size)
        while not self.free_to_add_at(i, j):
            i = np.random.randint(0, self.size)
            i = np.random.randint(0, self.size)
        return i, j

    def add_to_random_location(self, var: str):
        """Mutates the original map."""
        i: int
        j: int
        i, j = self.get_random_free_coordinates()
        self.content[i][j].add(var)

    def add_wall(self, start: tuple[int, int], delta: tuple[int, int]):
        length: int = abs(delta[0]) + abs(delta[1])
        assert length > 0
        assert delta[0]*delta[1] == 0

        horizontal: bool = delta[0] == 0
        forward: bool = delta[0] > 0 or delta[1] > 0

        for k in range(length):
            i, j = start
            if horizontal:
                if forward:
                    j += k
                else:
                    j -= k
            else:
                if forward:
                    i += k
                else:
                    i -= k
            self.clear_nonarea(i, j)
            self.add('wall', i, j)

    def build(self) -> Map:
        return Map([[frozenset(vars) for vars in row] for row in self.content],
                   self.size, self.terms, self.spawn_type, self.spawn_location)


class MapConfig():
    def __init__(self, size: int,
                 p_object: float,
                 p_color_object: float,
                 p_place: float,
                 p_color: float):
        self.size: int = size
        self.p_object: float = p_object
        self.p_color_object: float = p_color_object
        self.p_place: float = p_place
        self.p_color: float = p_color


class VarPicker():
    def __init__(self, pick_from: list[str]):
        self.pick_from = pick_from
        np.random.shuffle(self.pick_from)
        self.picker_i: int = 0

    def pick(self) -> str:
        chosen: str = self.pick_from[self.picker_i]
        self.increment()
        return chosen

    def increment(self):
        self.picker_i += 1
        if self.picker_i == len(self.pick_from):
            self.picker_i = 0
            np.random.shuffle(self.pick_from)

    def remove(self, var: str):
        self.pick_from.remove(var)

    def remove_all(self, vars: Iterable[str]):
        for var in vars:
            self.remove(var)


def example_map_1() -> Map:
    config: MapConfig = MapConfig(
        size=6,
        p_object=0.3,
        p_color_object=0.2,
        p_place=0.3,
        p_color=0.01,
    )
    terms: dict[str, list[str]] = load_terms(DEFAULT_TERMS_PATH)

    map: MapBuilder = MapBuilder(config.size, terms)

    map.add_area('forest', (0, 0), (int(config.size/2), int(config.size/2)))
    map.add_area('field', (0, int(config.size/2)),
                 (int(config.size/2), int(config.size/2)))
    map.add_area('town', (int(config.size/2), 0),
                 (int(config.size/2), int(config.size/2)))
    map.add_area('factory', (int(config.size/2), int(config.size/2)),
                 (int(config.size/2), int(config.size/2)))

    all_objects: list[str] = get_all_terms_from_tag(terms, 'OBJECT')
    all_areas: list[str] = get_all_terms_from_tag(terms, 'AREA')
    all_places: list[str] = get_all_terms_from_tag(terms, 'PLACE')
    all_colors: list[str] = get_all_terms_from_tag(terms, 'COLOR')

    object_picker: VarPicker = VarPicker(all_objects)
    area_picker: VarPicker = VarPicker(all_areas)
    place_picker: VarPicker = VarPicker(all_places)
    color_picker: VarPicker = VarPicker(all_colors)

    place_picker.remove_all(area_picker.pick_from)
    place_picker.remove('wall')

    for i in range(config.size):
        for j in range(config.size):
            if np.random.random() < config.p_object:
                to_add: str = object_picker.pick()
                map.add(to_add, i, j)
                if np.random.random() < config.p_color_object:
                    to_add: str = color_picker.pick()
                    map.add(to_add, i, j)
            if np.random.random() < config.p_place:
                to_add: str = place_picker.pick()
                map.add(to_add, i, j)
            if np.random.random() < config.p_color:
                to_add: str = color_picker.pick()
                map.add(to_add, i, j)

    map.add_wall((int(config.size/2), int(config.size/4)),
                 (0, int(config.size/2)+1))
    map.add_wall((int(config.size/4), int(config.size/2)),
                 (int(config.size/2)+1, 0))

    return map.build()


def save_map(path: str | Path, map: Map):
    with open(path, 'w') as f:
        f.write(map.to_json())


def map_builder(output_name: str):
    path = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.txt')
    print(path)
    map: Map = example_map_1()
    save_map(path, map)


def main():
    set_all_seeds(SEED)
    map_builder('map_test')


if __name__ == '__main__':
    main()
