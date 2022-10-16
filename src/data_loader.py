from curses.ascii import isspace
from typing import Iterator, Tuple

import more_itertools


class Entry:
    def __init__(self, appears: set[str], expr_src: str, descriptions: list[str]):
        self.appears = appears
        self.expr_src = expr_src
        self.descriptions = descriptions


class DataLoader:
    def __init__(self, entries: list[Entry]):
        self.entries = entries
        self._count = len(list(pairs(self.entries)))

    def count(self) -> int:
        return self._count

    def format_pairs(self, count: int):
        p = pairs(self.entries)
        i = 0
        for (expr_src, desc) in p:
            print(f'{desc} => {expr_src}')
            i += 1
            if i >= count:
                break


def parse_entries(lines: more_itertools.peekable) -> list[Entry]:
    entries = [parse_entry(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_entries(lines)
        entries.extend(others)
    return entries


def parse_entry(lines: more_itertools.peekable) -> Entry:
    appears = next(lines).split(' ')
    expr_src = next(lines)
    descriptions = list()
    while lines and not lines.peek() == '':
        descriptions.append(next(lines))
    if lines and lines.peek() == '':
        next(lines)
    return Entry(appears, expr_src, descriptions)


def load_file(path: str) -> DataLoader:
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        return DataLoader(parse_entries(more_itertools.peekable(iter(lines))))


def pairs(data: list[Entry]) -> Iterator[Tuple[str, str]]:
    for entry in data:
        for description in entry.descriptions:
            yield (entry.expr_src, description)
