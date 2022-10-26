from pathlib import Path
import more_itertools
from typing import Tuple
import IPython

from util import *


def count_vars(pattern: str) -> int:
    """Counts the number of variables in a pattern"""
    return sum(1 for char in pattern if char.isupper())


def categorize_patterns(patterns: list[str]) -> list[list[str]]:
    """Categorizes patterns by number of variables"""
    cats = list()
    for pattern in patterns:
        num_vars = count_vars(pattern)
        extend_until(cats, num_vars, lambda: list())
        cats[num_vars - 1].append(pattern)
    return cats


def parse_all_patterns(lines: more_itertools.peekable) -> list[Tuple[str, int, list[str]]]:
    """
        - returns [(name, level, patterns)]
    """
    groups = [parse_pattern_group(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_all_patterns(lines)
        groups.extend(others)
    return groups


def parse_pattern_group(lines: more_itertools.peekable) -> Tuple[str, int, list[str]]:
    name, level = parse_name_level(lines)
    patterns = parse_patterns(lines)
    return name, level, patterns


def parse_name_level(lines: more_itertools.peekable) -> Tuple[str, int]:
    name, level = next(lines).split()
    return name, int(level)


def parse_patterns(lines: more_itertools.peekable) -> list[str]:
    patterns = [next(lines)]
    while lines and not lines.peek() == '':
        patterns.append(next(lines))
    if lines and lines.peek() == '':
        next(lines)
    return patterns


def load_patterns(path: Path | str) -> dict[str, list[list[list[str]]]]:
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        groups = parse_all_patterns(
            more_itertools.peekable(iter(lines)))
        patterns = dict()
        for name, level, all_patterns in groups:
            if name not in patterns:
                patterns[name] = []
            extend_until(patterns[name], level, lambda: list())
            patterns[name][level-1] = categorize_patterns(all_patterns)
        return patterns
