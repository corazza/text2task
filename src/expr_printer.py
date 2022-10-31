from rm_ast import *


def wrap(parent: RMExpr, child: RMExpr) -> bool:
    if isinstance(parent, Repeat) or isinstance(parent, Plus):
        return True
    elif isinstance(child, Vars):
        return False
    else:
        return isinstance(parent, Then) and isinstance(child, Or)


def children_to_str(con: str, parent: RMExpr, children: list[RMExpr]) -> str:
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


def child_to_str(op: str, parent: RMExpr, child: RMExpr) -> str:
    r = expr_to_str(child)
    if wrap(parent, child):
        r = f'({r}){op}'
    else:
        r = f'{r}{op}'
    return r


def expr_to_str(expr: RMExpr) -> str:
    if isinstance(expr, Then):
        return children_to_str(' ', expr, expr.exprs)
    if isinstance(expr, Or):
        return children_to_str(' | ', expr, expr.exprs)
    elif isinstance(expr, Repeat):
        return child_to_str('*', expr, expr.child)
    elif isinstance(expr, Plus):
        return child_to_str('+', expr, expr.child)
    else:
        assert isinstance(expr, Vars)
        return expr.transition()
