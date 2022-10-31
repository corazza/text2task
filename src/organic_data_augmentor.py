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


# def preprocess_string(x: str) -> str:
#     types = extract_prop_types_uppercase(x)
#     IPython.embed()
#     raise NotImplementedError()


# def prop_mapping(x: str, props: props: dict[str, list[str]]) -> dict[str, str]:
#     raise NotImplementedError()

def prop_to_type(x: str, props: dict[str, list[str]]) -> dict[str, str]:
    objects = extract_prop_types_uppercase(x)
    counters = {p: 0 for p in props}
    letters = {}
    types = {}
    for object in objects:
        for type in props:
            if object in props[type]:
                types[object] = type
                letters[object] = chr(counters[type] + ord('a'))
                counters[type] += 1
    r = {}
    for object in objects:
        r[object] = f'${types[object]}.{letters[object]}'
    return r


def preprocess_pair(x: Tuple[str, str], props: dict[str, list[str]]) -> Tuple[str, str]:
    desc = x[0]
    src = x[1]
    mapping = prop_to_type(desc, props)
    return apply_mapping(desc, mapping), apply_mapping(src, mapping)


def load_file(path: str | Path, props_path: str | Path, inflation_factor: int) -> list[Tuple[str, str]]:
    assert inflation_factor >= 1
    props = rm_generator.load_props(props_path)
    entries = parse_entries(more_itertools.peekable(util.line_iter(path)))
    organic = list(reshape_entries(entries))
    organic = [preprocess_pair(
        p, props) if '$' not in p[0] else p for p in organic]
    variants = list(
        map(lambda desc_src: list(generate_variants(desc_src[0], desc_src[1], props, inflation_factor)), organic))
    flattened_variants = list(itertools.chain(*variants))
    return list(map(lambda x: (x[0].lower(), x[1]), itertools.chain(flattened_variants)))


def pairs(entry: Tuple[list[str], list[str]]) -> Iterator[Tuple[str, str]]:
    for description in entry[0]:
        for src in entry[1]:
            yield (src, description)


def extract_prop_types(x: str,  props: dict[str, list[str]]) -> frozenset[str]:
    regex = '\$[a-z]+\.[a-zA-Z]'
    return frozenset(re.findall(regex, x))


def extract_prop_types_uppercase(x: str) -> frozenset[str]:
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
    return frozenset(uppercase_words)

# TODO rename to categories


def mappings(props: frozenset[str], all_props: dict[str, list[str]]) -> Iterator[dict[str, str]]:
    types = {prop: prop[1:len(prop)-2] for prop in props}
    groups = {prop: all_props[t] for prop, t in types.items()}
    items = groups.items()
    props_ordered = list(map(lambda x: x[0], items))
    groups_ordered = list(map(lambda x: x[1], items))
    for comb in itertools.product(*groups_ordered):
        r = {}
        taken = {prop: set() for prop in all_props.keys()}
        invalid = False
        for i in range(len(comb)):
            if comb[i] in taken[types[props_ordered[i]]]:
                invalid = True
                break
            r[props_ordered[i]] = comb[i]
            taken[types[props_ordered[i]]].add(comb[i])
        if invalid:
            continue
        yield r
