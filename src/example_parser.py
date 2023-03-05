import more_itertools
from typing import Iterator, Optional, Tuple

# examples -> example
# examples -> example examples

# example -> single | multi

# single -> desc => src
# multi -> descs \n = \n srcs

# descs -> desc
# descs -> desc descs

# srcs -> src
# srcs -> src srcs


def line_iter(path: str) -> Iterator[str]:
    with open(path, 'r') as f:
        for line in f:
            yield line.strip()
    yield ''


def parse_examples(lines: more_itertools.peekable) -> list[Tuple[str, str]]:
    example = parse_example(lines)
    try:
        rest = parse_examples(lines)
        return example + rest
    except StopIteration:
        return example


def parse_example(lines: more_itertools.peekable) -> list[Tuple[str, str]]:
    line: str = lines.peek()
    while line == '':
        next(lines)
        line = lines.peek()
    if '=>' in line:
        return [parse_single(lines)]
    else:
        return parse_multi(lines)


def parse_single(lines: more_itertools.peekable) -> Tuple[str, str]:
    line: str = next(lines)
    assert '=>' in line
    result = line.split('=>')
    desc = result[0].strip()
    src = result[1].strip()
    return (desc, src)


def parse_multi(lines: more_itertools.peekable) -> list[Tuple[str, str]]:
    descs = parse_until_separator(lines, '=')
    token = next(lines)
    assert token == '='
    srcs = parse_until_separator(lines, '')
    result: list[Tuple[str, str]] = []
    assert len(descs) > 0 and len(srcs) > 0
    for desc in descs:
        for src in srcs:
            result.append((desc, src))
    return result


def parse_until_separator(lines: more_itertools.peekable, sep: str) -> list[str]:
    parsed_lines: list[str] = []
    assert lines.peek() != sep
    while lines.peek() != sep:
        line = next(lines)
        parsed_lines.append(line.strip())
    return parsed_lines
