import copy
from functools import cmp_to_key
from typing import Iterator, Optional, Tuple

import IPython
import more_itertools

from compiler_interface import compile
from parser_util import *

# examples -> example
# examples -> example examples

# example -> pos_neg \n src_desc

# pos_neg -> positives \n negatives

# src_desc -> single | multi

# single -> desc => src
# multi -> descs \n = \n srcs

# descs -> desc
# descs -> desc descs

# srcs -> src
# srcs -> src srcs


def clean_desc(d: str) -> str:
    if d[-1] == '.':
        return d[:-1]
    return d


class Example:
    def __init__(self, synthetic: bool, example_rewrites: list[list[Tuple[str, str]]], runs: list[Tuple[int, list[frozenset[str]]]], descs: list[str], srcs: list[str], id: str):
        self.synthetic = synthetic
        self.descs = descs
        self.srcs = srcs
        self.runs = runs
        self.example_rewrites: list[list[Tuple[str, str]]] = example_rewrites
        self.id: str = id
        self.hard_cap_one: bool = False

        self._clean_descs()

    def _clean_descs(self):
        self.descs = [clean_desc(x) for x in self.descs]

    def average_desc_length(self):
        return float(sum([len(desc) for desc in self.descs])) / len(self.descs)

    def vars(self) -> list[str]:
        if len(self.example_rewrites) == 0:
            return []
        else:
            return list(reversed(sorted([x for (x, y) in self.example_rewrites[0]])))

    def desc_includes_first_second(self):
        for desc in self.descs:
            desc = desc.lower()
            if 'first' in desc:
                return True
            if 'second' in desc:
                return True
        return False

    def produce_examples(self) -> list[Tuple[str, str]]:
        result: list[Tuple[str, str]] = []
        for desc in self.descs:
            for src in self.srcs:
                result.append((desc, src))
        return result

    def representative(self) -> str:
        if self.id == '-1':
            return self.descs[0]
        else:
            return self.id


def parse_single_line_examples(lines: more_itertools.peekable) -> list[Example]:
    examples = [parse_single_line_example(lines)]
    while True:
        try:
            rest = parse_single_line_example(lines)
            examples.append(rest)
        except StopIteration:
            return examples


def parse_examples(lines: more_itertools.peekable) -> list[Example]:
    examples = [parse_example(lines)]
    while True:
        try:
            rest = parse_example(lines)
            examples.append(rest)
        except StopIteration:
            return examples


def parse_example(lines: more_itertools.peekable) -> Example:
    line: str = lines.peek()
    while line == '':
        next(lines)
        line = lines.peek()
    runs: list[Tuple[int, list[frozenset[str]]]] = parse_runs(lines)
    parse_the_separator(lines, '=')
    example_rewrites = parse_example_rewrites(lines)
    parse_the_separator(lines, '=')
    descs = parse_descs(lines)
    parse_the_separator(lines, '=')
    srcs = parse_srcs(lines)
    nextline = lines.peek()
    if '=' in nextline:
        parse_the_separator(lines, '=')
        id = parse_id(lines)
    else:
        id = '-1'
    parse_the_separator(lines, '')
    ex = Example(False, example_rewrites, runs, descs, srcs, id)
    ex.hard_cap_one = True
    return ex


def parse_single_line_example(lines: more_itertools.peekable) -> Example:
    line: str = next(lines)
    while line == '':
        line = next(lines)
    print(line)
    rew_str, desc, src = line.split('=>')
    example_rewrites: list[list[tuple[str, str]]] = [
        line_to_rewrite(rew_str.strip())]
    descs: list[str] = [desc.strip()]
    srcs: list[str] = [src.strip()]
    id: str = '-1'
    runs: list = []
    return Example(False, example_rewrites, runs, descs, srcs, id)


def parse_id(lines: more_itertools.peekable) -> str:
    return next(lines)


def parse_example_rewrites(lines: more_itertools.peekable) -> list[list[Tuple[str, str]]]:
    rewrite_lines = parse_until_separator(lines, {'='})
    return [line_to_rewrite(line) for line in rewrite_lines]


def compare_rewrite(a: Tuple[str, str], b: Tuple[str, str]):
    if len(a[0]) < len(b[0]):
        return -1
    elif len(a[0]) == len(b[0]):
        return 0
    else:
        return 1


def line_to_rewrite(line: str) -> list[Tuple[str, str]]:
    example_rewrite: list[Tuple[str, str]] = []
    line_apart: list[str] = line.split(',')
    for rewrite in line_apart:
        left, right = rewrite.split('%')
        left = left.strip()
        right = right.strip()
        for previous_left, previous_right in example_rewrite:
            assert previous_right != left, f'(1) circular rewrite {left}->{right}'
            assert left not in previous_right, f'(2) circular rewrite {left}->{right}'
            assert previous_left != left, f'redundant rewrite {left}->{right}'
        example_rewrite.append((left, right))
    return list(reversed(sorted(example_rewrite, key=cmp_to_key(compare_rewrite))))


def parse_runs(lines: more_itertools.peekable) -> list[Tuple[int, list[frozenset[str]]]]:
    run_lines = parse_until_separator(lines, {'='})
    return [line_to_run(line) for line in run_lines]


def line_to_run(line: str) -> Tuple[int, list[frozenset[str]]]:
    line_apart: list[str] = line.split('@')
    reward: int = int(line_apart[0].strip())
    input_symbols: list[set[str]] = eval('[' + line_apart[1].strip() + ']')
    return reward, [frozenset(x) for x in input_symbols]


def parse_descs(lines: more_itertools.peekable) -> list[str]:
    return parse_until_separator(lines, {'='})


def parse_srcs(lines: more_itertools.peekable) -> list[str]:
    return parse_until_separator(lines, {'', '='})


def parse_until_separator(lines: more_itertools.peekable, sep: set[str]) -> list[str]:
    parsed_lines: list[str] = []
    while lines.peek() not in sep:
        line = next(lines)
        parsed_lines.append(line.strip())
    return parsed_lines


def parse_the_separator(lines: more_itertools.peekable, sep: str):
    line = next(lines)
    assert line == sep, f'{line} === {sep}'
