from pathlib import Path
import more_itertools
from typing import Tuple

from rm_ast import *


def parse_all_props(lines: more_itertools.peekable) -> list[Tuple[str, list[str]]]:
    maps = [parse_prop_group(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_all_props(lines)
        maps.extend(others)
    return maps


def parse_prop_group(lines: more_itertools.peekable) -> Tuple[str, list[str]]:
    name = next(lines)
    props = parse_props(lines)
    return name, props


def parse_props(lines: more_itertools.peekable) -> list[str]:
    props = [next(lines).split()]
    while lines and not lines.peek() == '':
        props.append(next(lines).split())
    if lines and lines.peek() == '':
        next(lines)
    return list(itertools.chain.from_iterable(props))


def load_props(path: Path | str) -> dict[str, list[str]]:
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        maps = parse_all_props(more_itertools.peekable(iter(lines)))
        map = dict()
        for name, props in maps:
            assert name not in map
            map[name] = props
        return map


def generate(seed: int) -> RMExpr:
    raise NotImplementedError()
