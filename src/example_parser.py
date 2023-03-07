import more_itertools
from typing import Iterator, Optional, Tuple
from compiler_interface import compile

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


class Example:
    def __init__(self, runs: list[Tuple[int, list[frozenset[str]]]], descs: list[str], srcs: list[str]):
        self.descs = descs
        self.srcs = srcs
        self.runs = runs

    def produce_examples(self) -> list[Tuple[str, str]]:
        result: list[Tuple[str, str]] = []
        for desc in self.descs:
            for src in self.srcs:
                result.append((desc, src))
        return result


def line_iter(path: str) -> Iterator[str]:
    with open(path, 'r') as f:
        for line in f:
            yield line.strip()
    yield ''


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
    runs = parse_runs(lines)
    descs, srcs = parse_descs_srcs(lines)
    return Example(runs, descs, srcs)


def parse_runs(lines: more_itertools.peekable) -> list[Tuple[int, list[frozenset[str]]]]:
    runs: list[Tuple[int, list[frozenset]]] = []
    while '@' in lines.peek():
        runs.append(parse_run(lines))
    return runs


def parse_run(lines: more_itertools.peekable) -> Tuple[int, list[frozenset[str]]]:
    line = next(lines)
    line_apart: list[str] = line.split('@')
    reward: int = int(line_apart[0].strip())
    input_symbols: list[set[str]] = eval(line_apart[1].strip())
    return reward, [frozenset(x) for x in input_symbols]


def parse_descs_srcs(lines: more_itertools.peekable) -> Tuple[list[str], list[str]]:
    line: str = lines.peek()
    if '=>' in line:
        return parse_single(lines)
    else:
        return parse_multi(lines)


def parse_single(lines: more_itertools.peekable) -> Tuple[list[str], list[str]]:
    line: str = next(lines)
    assert '=>' in line
    result = line.split('=>')
    desc = result[0].strip()
    src = result[1].strip()
    return ([desc], [src])


def parse_multi(lines: more_itertools.peekable) -> Tuple[list[str], list[str]]:
    descs = parse_until_separator(lines, '=')
    token = next(lines)
    assert token == '='
    srcs = parse_until_separator(lines, '')
    assert len(descs) > 0 and len(srcs) > 0
    return descs, srcs


def parse_until_separator(lines: more_itertools.peekable, sep: str) -> list[str]:
    parsed_lines: list[str] = []
    assert lines.peek() != sep
    while lines.peek() != sep:
        line = next(lines)
        parsed_lines.append(line.strip())
    return parsed_lines
