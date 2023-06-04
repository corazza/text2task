from regex_ast import *


def operator_precedence(node: RENode) -> int:
    if isinstance(node, Or):
        return 0
    elif isinstance(node, Then):
        return 1
    elif isinstance(node, And):
        return 2
    elif isinstance(node, Matcher) or isinstance(node, RENodeSing):
        return 4
    else:
        raise ValueError(f'incomplete precedence rules')


def wrap(parent: RENode, child: RENode) -> bool:
    assert not isinstance(parent, Matcher)
    if isinstance(parent, RENodeSing):
        return True
    parent_precedence = operator_precedence(parent)
    child_precedence = operator_precedence(child)
    return child_precedence < parent_precedence


def children_to_str(con: str, parent: RENode, children: list[RENode]) -> str:
    r = ''
    for i in range(len(children)-1):
        if wrap(parent, children[i]):
            r = f'{r}({expr_to_str(children[i])}){con}'
        else:
            r = f'{r}{expr_to_str(children[i])}{con}'
    if wrap(parent, children[-1]):
        r = f'{r}({expr_to_str(children[-1])})'
    else:
        r = f'{r}{expr_to_str(children[-1])}'
    return r


def child_to_str(op: str, parent: RENode, child: RENode) -> str:
    r = expr_to_str(child)
    if wrap(parent, child):
        r = f'({r}){op}'
    else:
        r = f'{r}{op}'
    return r


def expr_to_str(expr: RENode) -> str:
    if isinstance(expr, RENodeMul):
        if len(expr.exprs) == 0:
            IPython.embed()
        sep = '' if isinstance(expr, And) else ' '
        return children_to_str(sep + expr.con + sep, expr, expr.exprs)
    elif isinstance(expr, RENodeSing):
        return child_to_str(expr.con, expr, expr.child)
    else:
        assert isinstance(expr, Matcher)
        return str(expr)
