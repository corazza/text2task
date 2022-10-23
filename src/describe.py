from typing import Iterator
import IPython
import random
import itertools

import rm_ast
from rm_ast import RMExpr
import describe_patterns


def compute_max_level(expr: RMExpr) -> int:
    if isinstance(expr, rm_ast.Then):
        levels = map(compute_max_level, expr.exprs)
        return 1 + max(levels)
    elif isinstance(expr, rm_ast.Or):
        levels = map(compute_max_level, expr.exprs)
        return 1 + max(levels)
    elif isinstance(expr, rm_ast.Repeat):
        child = compute_max_level(expr.child)
        return 1 + child
    else:
        assert isinstance(expr, rm_ast.Vars)
        return 0


def _children_describe(current_level: int, max_level: int, exprs: list[RMExpr]) -> list[list[str]]:
    return list(map(lambda c: _describe(current_level+1, max_level, c), exprs))


def _apply_pattern(pattern: str, using: list[str]) -> str:
    r = pattern
    char = 'A'
    for to_use in using:
        r = r.replace(char, to_use)
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


def _describe_multiple(current_level: int, max_level: int, patterns: list[list[str]], exprs: list[RMExpr]) -> list[str]:
    descs = _children_describe(current_level, max_level, exprs)
    num_children = len(descs)
    return _apply_pattern_helper(patterns[num_children-2], descs)


def _describe_vars(current_level: int, max_level: int, symbols: list[str]) -> str:
    num_symbols = len(symbols)
    r = ''
    for i in range(num_symbols-1):
        r = f'{r}{symbols[i]} and '
    r = f'{r}{symbols[-1]}'
    return r


def _describe(current_level: int, max_level: int, expr: RMExpr) -> list[str]:
    if isinstance(expr, rm_ast.Then):
        assert len(expr.exprs) >= 2
        return _describe_multiple(current_level, max_level, describe_patterns.patterns_then, expr.exprs)
    elif isinstance(expr, rm_ast.Or):
        assert len(expr.exprs) >= 2
        return _describe_multiple(current_level, max_level, describe_patterns.patterns_then, expr.exprs)
    elif isinstance(expr, rm_ast.Repeat):
        return _describe_multiple(current_level, max_level, describe_patterns.patterns_repeat, [expr.child])
    else:
        assert isinstance(expr, rm_ast.Vars)
        return [_describe_vars(current_level, max_level, expr.symbols)]


def describe(expr: RMExpr) -> list[str]:
    max_level = compute_max_level(expr)
    return _describe(0, max_level, expr)
