from functools import cmp_to_key
from typing import Iterator, Optional, Tuple

import IPython
import more_itertools

from compiler_interface import compile


def parse_terms(lines: more_itertools.peekable) -> dict[str, list[str]]:
    first, tags = parse_term(lines)
    terms = {first: tags}
    while True:
        try:
            rest, rest_tags = parse_term(lines)
            if rest in terms:
                print(f'duplicate: {rest}')
                continue
            terms[rest] = rest_tags
        except StopIteration:
            return terms


def parse_term(lines: more_itertools.peekable) -> Tuple[str, list[str]]:
    line: str = lines.peek()
    while line == '':
        next(lines)
        line = lines.peek()
    line = next(lines).strip()
    name_tags = line.split()
    tags = []
    name = name_tags[0].strip()
    for name_tag in name_tags[1:]:
        tags.append(name_tag.strip())
    return (name, tags)
