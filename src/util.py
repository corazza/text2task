from pathlib import Path
from typing import Iterator
import more_itertools


def extend_until(xs, until, f):
    for i in range(until - len(xs)):
        xs.append(f())


# TODO substitute line_iter and parse_lines

def line_iter(path: Path | str) -> Iterator[str]:
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        return iter(filter(lambda l: '#' not in l, lines))


def parse_lines(lines: more_itertools.peekable) -> list[str]:
    r = list()  # TODO check if this enforces at least one description
    while lines and not lines.peek() == '':
        r.append(next(lines))
    if lines and lines.peek() == '':
        next(lines)
    return r
