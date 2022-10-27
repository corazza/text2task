import numpy as np
import copy
from pathlib import Path
from typing import Iterator, Tuple
import IPython
import random
import itertools
import more_itertools

import rm_ast
from rm_ast import RMExpr
import describe_patterns


class DescribeContext:
    def __init__(self, patterns: dict[str, list[list[list[str]]]], var_describe_map: dict[str, list[str]], max_level: int):
        self.patterns = patterns
        self.var_describe_map = var_describe_map
        self.max_level = max_level


def parse_maps(lines: more_itertools.peekable) -> list[Tuple[list[str], list[str]]]:
    """
        - returns [(vars, phrases)]
    """
    maps = [parse_map(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_maps(lines)
        maps.extend(others)
    return maps


def parse_map(lines: more_itertools.peekable) -> Tuple[list[str], list[str]]:
    vars = next(lines).split()
    phrases = parse_phrases(lines)
    return vars, phrases


def parse_phrases(lines: more_itertools.peekable) -> list[str]:
    phrases = [next(lines)]
    while lines and not lines.peek() == '':
        phrases.append(next(lines))
    if lines and lines.peek() == '':
        next(lines)
    return phrases


def load_var_describe_map(path: Path | str) -> dict[str, list[str]]:
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        maps = parse_maps(more_itertools.peekable(iter(lines)))
        map = dict()
        for vars, phrases in maps:
            for var in vars:
                assert var not in map
                map[var] = phrases
        return map


def compute_max_level(expr: RMExpr) -> int:
    if isinstance(expr, rm_ast.Then):
        levels = map(compute_max_level, expr.exprs)
        return 1 + max(levels)
    elif isinstance(expr, rm_ast.Or):
        levels = map(compute_max_level, expr.exprs)
        return 1 + max(levels)
    elif isinstance(expr, rm_ast.Repeat) or isinstance(expr, rm_ast.Plus):
        child = compute_max_level(expr.child)
        return 1 + child
    else:
        assert isinstance(expr, rm_ast.Vars)
        return 0


def _children_describe(context: DescribeContext, current_level: int, exprs: list[RMExpr]) -> list[list[str]]:
    return list(map(lambda c: _describe(context, current_level, c), exprs))


def _apply_pattern(pattern: str, using: list[str]) -> str:
    r = pattern
    char = 'A'
    for to_use in using:
        r = r.replace(f'${char}', to_use)
        char = chr(ord(char)+1)
    return r


def _combinations(children: list[list[str]]) -> Iterator[list[str]]:
    for c in itertools.product(*children):
        yield list(c)


def _apply_pattern_helper(patterns: list[str], children_descs: list[list[str]]) -> list[str]:
    r = []
    for pattern in patterns:
        for children in _combinations(children_descs):
            desc = _apply_pattern(pattern, children)
            r.append(desc)
    return r


def _pick_pattern(context: DescribeContext, current_level: int, patterns: list[list[list[str]]]) -> list[list[str]]:
    num_levels = len(patterns)
    if current_level >= num_levels:
        return patterns[num_levels-1]
    else:
        # chosen = np.random.randint(current_level, num_levels)
        # print(chosen, current_level, num_levels)
        return patterns[current_level]


def _describe_multiple(context: DescribeContext, current_level: int, patterns: list[list[list[str]]], exprs: list[RMExpr]) -> list[str]:
    """
        - patterns: levels, number of children, pattern
    """
    descs = _children_describe(context, current_level+1, exprs)
    num_children = len(descs)
    picked_patterns = _pick_pattern(context, current_level, patterns)
    return _apply_pattern_helper(picked_patterns[num_children - 1], descs)


def _describe_vars(context: DescribeContext, current_level: int, symbols: list[str]) -> list[str]:
    num_symbols = len(symbols)
    phrases = list(map(lambda s: _describe_var(
        context, current_level, s), symbols))
    desc = list()
    for c in _combinations(phrases):
        r = ''
        for i in range(num_symbols-1):
            r = f'{r}{c[i]} and '
        r = f'{r}{c[-1]}'
        desc.append(r)
    return desc


def _describe_var(context: DescribeContext, current_level: int, var: str) -> list[str]:
    r = set()
    if var[0] == '!':
        return [f'no {var[1:]}']
    for phrase in context.var_describe_map[var]:
        if phrase == '':
            r.add(var)
        else:
            r.add(f'{phrase} {var}')
    return list(r)


def _describe(context: DescribeContext, current_level: int, expr: RMExpr) -> list[str]:
    if isinstance(expr, rm_ast.Then):
        assert len(expr.exprs) >= 2
        return _describe_multiple(context, current_level, context.patterns['THEN'], expr.exprs)
    elif isinstance(expr, rm_ast.Or):
        assert len(expr.exprs) >= 2
        return _describe_multiple(context, current_level, context.patterns['OR'], expr.exprs)
    elif isinstance(expr, rm_ast.Repeat):
        return _describe_multiple(context, current_level, context.patterns['REPEAT'], [expr.child])
    elif isinstance(expr, rm_ast.Plus):
        return _describe_multiple(context, current_level, context.patterns['PLUS'], [expr.child])
    else:
        assert isinstance(expr, rm_ast.Vars)
        return _describe_vars(context, current_level, expr.symbols)


def describe(patterns: dict[str, list[list[list[str]]]], var_describe_map: dict[str, list[str]], expr: RMExpr) -> list[str]:
    max_level = compute_max_level(expr)
    context = DescribeContext(patterns, var_describe_map, max_level)
    return _describe(context, 0, expr)
