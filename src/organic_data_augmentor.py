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
import compiler_interface
import expr_printer


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


def apply_mapping(to: str,  mapping: dict[str, str]) -> str:
    for k, v in mapping.items():
        to = to.replace(k, v)
    return to


def generate_variants(desc: str, src: str, props: dict[str, list[str]], take: int) -> Iterator[Tuple[str, str]]:
    props_desc = extract_prop_types(desc)
    props_src = extract_prop_types(src)
    assert props_desc == props_src
    r = []
    for mapping in mappings(props_desc, props):
        r.append((apply_mapping(desc, mapping), apply_mapping(src, mapping)))
    if len(r) == 0:  # there were not variables to map
        r.append((desc, src))
    will_take = set()
    while len(will_take) < take and len(will_take) < len(r):
        chosen = np.random.randint(0, len(r))
        will_take.add(chosen)
    for to_take in will_take:
        yield r[to_take]


def reshape_entries(entries: list[Tuple[list[str], list[str]]]) -> Iterator[Tuple[str, str]]:
    for entry in entries:
        for desc, src in pairs(entry):
            yield (desc, src)


def load_file(path: str | Path, props_path: str | Path, inflation_factor: int) -> list[Tuple[str, str]]:
    assert inflation_factor >= 1
    props = rm_generator.load_props(props_path)
    entries = parse_entries(more_itertools.peekable(util.line_iter(path)))
    organic = list(reshape_entries(entries))
    variants = list(
        map(lambda desc_src: list(generate_variants(desc_src[0], desc_src[1], props, inflation_factor)), organic))
    flattened_variants = list(itertools.chain(*variants))
    return list(map(lambda x: (x[0].lower(), x[1]), itertools.chain(flattened_variants)))


def pairs(entry: Tuple[list[str], list[str]]) -> Iterator[Tuple[str, str]]:
    for description in entry[0]:
        for src in entry[1]:
            yield (src, description)


def extract_prop_types(x: str) -> frozenset[str]:
    half_spec = frozenset(re.findall('[A-Z]+\%', x))
    full_spec = frozenset(re.findall('[A-Z]+\/[a-z]', x))
    for half in half_spec:
        for full in full_spec:
            assert half not in full
    return half_spec.union(full_spec)


def strip_prop(x: str) -> str:
    if '/' in x:
        return x[:x.find('/')]
    else:
        assert '%' in x
        return x[:x.find('%')]


def mappings(props: frozenset[str], all_props: dict[str, list[str]]) -> Iterator[dict[str, str]]:
    types = {prop: strip_prop(prop).lower() for prop in props}
    groups = {prop: all_props[t] for prop, t in types.items()}
    items = groups.items()
    props_ordered = list(map(lambda x: x[0], items))
    groups_ordered = list(map(lambda x: x[1], items))
    for comb in itertools.product(*groups_ordered):
        r = {}
        for i in range(len(comb)):
            r[props_ordered[i]] = comb[i]
        filtered = {k: v for k, v in r.items() if '%' not in k}
        suffixes = {v: list() for k, v in filtered.items()}
        for k, v in filtered.items():
            suffixes[v].append(k[-1])
        valid = True
        for k, v in suffixes.items():
            if len(set(v)) != 1:
                valid = False
                break
        if valid:
            yield r
