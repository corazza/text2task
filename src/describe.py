import numpy as np
from pathlib import Path
from typing import Iterator, Tuple
import IPython
import random
import itertools
import more_itertools

import rm_ast
from rm_ast import RMExpr
import util


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
    phrases = util.parse_lines(lines)
    return vars, phrases


def load_var_describe_map(path: Path | str) -> dict[str, list[str]]:
    lines = util.line_iter(path)
    maps = parse_maps(more_itertools.peekable(lines))
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


def _children_describe(context: DescribeContext, current_level: int, exprs: list[RMExpr]) -> list[str]:
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


def _apply_pattern_helper(patterns: list[str], children_descs: list[str]) -> str:
    r = []
    for pattern in patterns:
        desc = _apply_pattern(pattern, children_descs)
        r.append(desc)
    chosen = np.random.randint(0, len(r))
    return r[chosen]


def _pick_pattern(context: DescribeContext, current_level: int, patterns: list[list[list[str]]]) -> list[list[str]]:
    num_levels = len(patterns)
    if current_level >= num_levels:
        return patterns[num_levels-1]
    else:
        return patterns[current_level]


def _describe_multiple(context: DescribeContext, current_level: int, patterns: list[list[list[str]]], exprs: list[RMExpr]) -> str:
    """
        - patterns: levels, number of children, pattern
    """
    descs = _children_describe(context, current_level+1, exprs)
    num_children = len(descs)
    picked_patterns = _pick_pattern(context, current_level, patterns)
    return _apply_pattern_helper(picked_patterns[num_children - 1], descs)


# TODO describing multiple vars doesn't work like this
def _describe_vars(context: DescribeContext, current_level: int, symbols: list[str]) -> str:
    symbols = list(filter(lambda s: '!' not in s, symbols))
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
    chosen = np.random.randint(0, len(desc))
    return desc[chosen]


def _describe_var(context: DescribeContext, current_level: int, var: str) -> list[str]:
    r = set()
    for phrase in context.var_describe_map[var]:
        if phrase == '':
            r.add(var)
        else:
            if '!' in var:
                var = var[1:]
            r.add(phrase.replace('$', var))
    return list(r)


def _describe(context: DescribeContext, current_level: int, expr: RMExpr) -> str:
    if isinstance(expr, rm_ast.Then):
        assert len(expr.exprs) >= 2
        return _describe_multiple(context, current_level, context.patterns['THEN'], expr.exprs)
    elif isinstance(expr, rm_ast.Or):
        assert len(expr.exprs) >= 2
        return _describe_multiple(context, current_level, context.patterns['OR'], expr.exprs)
    elif isinstance(expr, rm_ast.Repeat):
        return _describe_multiple(context, current_level, context.patterns['REPEAT'], [expr.child])
    elif isinstance(expr, rm_ast.Plus):
        return _describe_multiple(context, current_level, context.patterns['REPEAT'], [expr.child])
    else:
        assert isinstance(expr, rm_ast.Vars)
        return _describe_vars(context, current_level, expr.symbols)


def apply_neg(context: DescribeContext, desc: str, appears_neg: str) -> list[str]:
    r = set()
    for phrase in context.var_describe_map[f'!{appears_neg}']:
        assert phrase != ''
        replaced = phrase.replace('$', appears_neg)
        r.add(f'{desc}. but {replaced}')
        r.add(f'while doing the task, {replaced}. {desc}')
    return list(r)


def describe(patterns: dict[str, list[list[list[str]]]], var_describe_map: dict[str, list[str]], expr: RMExpr) -> str:
    max_level = compute_max_level(expr)
    context = DescribeContext(patterns, var_describe_map, max_level)
    appears_neg = list(expr.appears_neg())
    assert len(appears_neg) <= 1
    desc = _describe(context, 0, expr)
    if len(appears_neg) == 1:
        negs = apply_neg(context, desc, appears_neg[0])
        chosen = np.random.randint(len(negs))
        return negs[chosen]
    else:
        return desc
