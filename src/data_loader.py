from pathlib import Path
from curses.ascii import isspace
from typing import Iterator, Tuple
import more_itertools
import random
import copy
import IPython


class Entry:
    def __init__(self, expr_src: list[str], descriptions: list[str]):
        self.expr_sources = expr_src
        self.descriptions = descriptions


class DataLoader:
    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self._count = len(list(pairs(self.entries)))

    def get_prompts(self, count: int) -> list[str]:
        return list(self._get_prompts(count))

    def get_all_prompts(self) -> list[str]:
        return self.get_prompts(len(self))

    def count(self) -> int:
        return self._count

    def split(self, seed: int, r: float) -> Tuple[list[str], list[str]]:
        r = 1 - r
        prompts = self.get_all_prompts()
        random.Random(seed).shuffle(prompts)
        num_in_train = int(len(prompts)*r)
        train = prompts[:num_in_train]
        val = prompts[num_in_train:]
        return train, val

    def _get_prompts(self, count: int) -> Iterator[str]:
        p = pairs(self.entries)
        i = 0
        for (expr_src, desc) in p:
            yield f'{desc} => {expr_src}'
            i += 1
            if i >= count:
                break

    def __len__(self) -> int:
        return self.count()


def parse_entries(lines: more_itertools.peekable) -> list[Entry]:
    entries = [parse_entry(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_entries(lines)
        entries.extend(others)
    return entries


def parse_entry(lines: more_itertools.peekable) -> Entry:
    expr_src = parse_sources(lines)
    descriptions = parse_descriptions(lines)
    return Entry(expr_src, descriptions)


def parse_sources(lines: more_itertools.peekable) -> list[str]:
    sources = list()
    while lines and not lines.peek() == '=':
        sources.append(next(lines))
    sep = next(lines)
    assert sep == '='
    return sources


def parse_descriptions(lines: more_itertools.peekable) -> list[str]:
    descriptions = list()
    while lines and not lines.peek() == '':
        descriptions.append(next(lines))
    if lines and lines.peek() == '':
        next(lines)
    return descriptions


def load_file(path: str | Path) -> DataLoader:
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        return DataLoader(parse_entries(more_itertools.peekable(iter(lines))))


def load_lines(path: str | Path) -> list[str]:
    with open(path, 'r') as f:
        return f.read().splitlines()


def pairs(data: list[Entry]) -> Iterator[Tuple[str, str]]:
    for entry in data:
        for description in entry.descriptions:
            for src in entry.expr_sources:
                yield (src, description)
