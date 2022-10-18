import IPython

import rm_ast
from rm_ast import RMExpr


def load_semantic_map(path: str) -> dict[str, str]:
    sm = dict()
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            k, v = line.split(' ')
            sm[k] = v
    return sm


def compute_max_level(expr: RMExpr) -> int:
    if isinstance(expr, rm_ast.Then):
        left = compute_max_level(expr.left)
        right = compute_max_level(expr.right)
        return 1 + max(left, right)
    elif isinstance(expr, rm_ast.Or):
        left = compute_max_level(expr.left)
        right = compute_max_level(expr.right)
        return 1 + max(left, right)
    elif isinstance(expr, rm_ast.Repeat):
        child = compute_max_level(expr.child)
        return 1 + child
    else:
        assert isinstance(expr, rm_ast.Vars)
        return 0


def _describe(semantic_map: dict[str, str], current_level: int, max_level: int, expr: RMExpr) -> str:
    if isinstance(expr, rm_ast.Then):
        left = _describe(semantic_map, current_level+1, max_level, expr.left)
        right = _describe(semantic_map, current_level+1, max_level, expr.right)
        return f'first do {left}, then do {right}'
    elif isinstance(expr, rm_ast.Or):
        left = _describe(semantic_map, current_level+1, max_level, expr.left)
        right = _describe(semantic_map, current_level+1, max_level, expr.right)
        return f'either {left}, or {right}'
    elif isinstance(expr, rm_ast.Repeat):
        child = _describe(semantic_map, current_level+1, max_level, expr.child)
        return f'repeat {child}'
    else:
        assert isinstance(expr, rm_ast.Vars)
        return semantic_map[expr.symbol]


def describe(semantic_map: dict[str, str], expr: RMExpr) -> str:
    max_level = compute_max_level(expr)
    return _describe(semantic_map, 0, max_level, expr)
