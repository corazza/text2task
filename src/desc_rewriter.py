import numpy as np
from pathlib import Path
from curses.ascii import isspace, isupper
from typing import Iterator, Tuple
import more_itertools
import random
import copy
import IPython
import itertools
import re

import rm_generator
import util


def parse_entries(lines: more_itertools.peekable) -> list[Tuple[str, list[str]]]:
    entries = [parse_entry(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_entries(lines)
        entries.extend(others)
    return entries


def parse_entry(lines: more_itertools.peekable) -> Tuple[str, list[str]]:
    src = next(lines)
    dsts = util.parse_lines(lines)
    return src, dsts


def load_file(path: str | Path) -> dict[str, list[str]]:
    entries = parse_entries(more_itertools.peekable(util.line_iter(path)))
    return {e[0]: e[1] for e in entries}


def apply_rewrites(desc: str, rewrites: dict[str, list[str]]):
    for rule, target in rewrites.items():
        p = 1 / (len(target)+1)
        if np.random.random() < p:
            chosen = np.random.randint(len(target))
            desc = desc.replace(rule, target[chosen])
    return desc
