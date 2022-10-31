import numpy as np
from pathlib import Path
from curses.ascii import isspace, isupper
from typing import Iterator, Tuple
import more_itertools
import random
import copy
import IPython
import itertools

import rm_generator


class Entry:
    def __init__(self, expr_src: list[str], descriptions: list[str]):
        self.expr_sources = expr_src
        self.descriptions = descriptions


def parse_entries(lines: more_itertools.peekable) -> list[Tuple[list[str], list[str]]]:
    entries = [parse_entry(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_entries(lines)
        entries.extend(others)
    return entries


def parse_entry(lines: more_itertools.peekable) -> Tuple[list[str], list[str]]:
    expr_src = parse_sources(lines)
    descriptions = parse_descriptions(lines)
    return expr_src, descriptions


def parse_sources(lines: more_itertools.peekable) -> list[str]:
    sources = list()
    while lines and not lines.peek() == '=':
        sources.append(next(lines))
    sep = next(lines)
    assert sep == '='
    return sources


def parse_descriptions(lines: more_itertools.peekable) -> list[str]:
    descriptions = list()  # TODO check if this enforces at least one description
    while lines and not lines.peek() == '':
        descriptions.append(next(lines))
    if lines and lines.peek() == '':
        next(lines)
    return descriptions


# HERE TODO take at most N, N is the inflation factor (shouldn't devote this much data space to learning a relatively simple transformation)


def apply_mapping(to: str,  mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        to = to.replace(k, v)
    return to


def generate_variants(desc: str, src: str, props: dict[str, list[str]], take: int) -> Iterator[Tuple[str, str]]:
    props_desc = extract_prop_types(desc, props)
    props_src = extract_prop_types(src, props)
    assert props_desc == props_src
    r = []
    for mapping in mappings(props_desc, props):
        r.append((apply_mapping(desc,  mapping), apply_mapping(src, mapping)))
    will_take = set()
    counter = 0
    while len(will_take) < take and counter < len(r):
        chosen = np.random.randint(0, len(r))
        if r[chosen] != (desc, src):
            will_take.add(chosen)
        counter += 1
    for to_take in will_take:
        yield r[to_take]


def reshape_entries(entries: list[Tuple[list[str], list[str]]]) -> Iterator[Tuple[str, str]]:
    for entry in entries:
        for desc, src in pairs(entry):
            yield (desc, src)


def load_file(path: str | Path, props_path: str | Path, inflation_factor: int) -> list[Tuple[str, str]]:
    assert inflation_factor >= 1
    props = rm_generator.load_props(props_path)
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        entries = parse_entries(more_itertools.peekable(iter(lines)))
        organic = list(reshape_entries(entries))
        variants = list(
            map(lambda desc_src: list(generate_variants(desc_src[0], desc_src[1], props, inflation_factor-1)), organic))
        flattened_variants = list(itertools.chain(*variants))
        return list(itertools.chain(organic, flattened_variants))


def pairs(entry: Tuple[list[str], list[str]]) -> Iterator[Tuple[str, str]]:
    for description in entry[0]:
        for src in entry[1]:
            yield (src, description)


def extract_prop_types(x: str,  props: dict[str, list[str]]) -> frozenset[str]:
    all_props = set(itertools.chain(*props.values()))
    uppercase_words = []
    current_word = ''
    for c in x:
        if isupper(c):
            current_word += c
        else:
            if len(current_word) > 1:
                uppercase_words.append(current_word)
                current_word = ''
    if len(current_word) > 1:
        uppercase_words.append(current_word)
        current_word = ''
    r = []
    for word in uppercase_words:
        if word in all_props:
            r.append(word)
    return frozenset(r)


def map_to_types(prop: str, props: dict[str, list[str]]) -> str:
    for key in props:
        if prop in props[key]:
            return key
    raise ValueError('unknown prop')


def mappings(props: frozenset[str], all_props: dict[str, list[str]]) -> Iterator[dict[str, str]]:
    types = {prop: map_to_types(prop, all_props) for prop in props}
    groups = {prop: all_props[t] for prop, t in types.items()}
    items = groups.items()
    props_ordered = list(map(lambda x: x[0], items))
    groups_ordered = list(map(lambda x: x[1], items))
    for comb in itertools.product(*groups_ordered):
        r = {}
        for i in range(len(comb)):
            r[props_ordered[i]] = comb[i]
        yield r
