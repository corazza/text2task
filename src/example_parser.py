import more_itertools
from typing import Iterator, Optional, Tuple
from compiler_interface import compile

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


def parse_examples(lines: more_itertools.peekable, do_validate: bool) -> list[Tuple[str, str]]:
    example = parse_example(lines, do_validate)
    try:
        rest = parse_examples(lines, do_validate)
        return example + rest
    except StopIteration:
        return example


def parse_example(lines: more_itertools.peekable, do_validate: bool) -> list[Tuple[str, str]]:
    line: str = lines.peek()
    while line == '':
        next(lines)
        line = lines.peek()
    if '=>' in line:
        return [parse_single(lines, do_validate)]
    else:
        return parse_multi(lines, do_validate)


def validate(src: str):
    try:
        compile(src)
    except Exception as e:
        raise ValueError(f'failed to validate: {src}\n\n{e}')


def parse_single(lines: more_itertools.peekable, do_validate: bool) -> Tuple[str, str]:
    line: str = next(lines)
    assert '=>' in line
    result = line.split('=>')
    desc = result[0].strip()
    src = result[1].strip()
    if do_validate:
        validate(src)
    return (desc, src)


def parse_multi(lines: more_itertools.peekable, do_validate: bool) -> list[Tuple[str, str]]:
    descs = parse_until_separator(lines, '=')
    token = next(lines)
    assert token == '='
    srcs = parse_until_separator(lines, '')
    result: list[Tuple[str, str]] = []
    assert len(descs) > 0 and len(srcs) > 0
    for src in srcs:
        if do_validate:
            validate(src)
        for desc in descs:
            result.append((desc, src))
    return result


def parse_until_separator(lines: more_itertools.peekable, sep: str) -> list[str]:
    parsed_lines: list[str] = []
    assert lines.peek() != sep
    while lines.peek() != sep:
        line = next(lines)
        parsed_lines.append(line.strip())
    return parsed_lines
