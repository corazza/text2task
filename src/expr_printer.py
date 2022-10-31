from rm_ast import *


def wrap(parent: RMExpr, child: RMExpr) -> bool:
    if isinstance(parent, Repeat) or isinstance(parent, Plus):
        return True
    elif isinstance(child, Vars):
        return False
    else:
        return isinstance(parent, Then) and isinstance(child, Or)


def children_to_str(con: str, parent: RMExpr, children: list[RMExpr], randomize: bool) -> str:
    r = ''
    for i in range(len(children)-1):
        if wrap(parent, children[i]):
            r = f'{r}({expr_to_str(children[i], randomize)}){con}'
        else:
            r = f'{r}{expr_to_str(children[i], randomize)}{con}'
    if wrap(parent, children[-1]):
        r = f'{r}({expr_to_str(children[-1], randomize)})'
    else:
        r = f'{r}{expr_to_str(children[-1], randomize)}'
    return r


def child_to_str(op: str, parent: RMExpr, child: RMExpr, randomize: bool) -> str:
    r = expr_to_str(child, randomize)
    if wrap(parent, child):
        r = f'({r}){op}'
    else:
        r = f'{r}{op}'
    return r


def expr_to_str(expr: RMExpr, randomize: bool = False, connect_then: bool = False) -> str:
    if isinstance(expr, Then):
        connector = ' -> ' if connect_then else ' '
        return children_to_str(connector, expr, expr.exprs, randomize)
    if isinstance(expr, Or):
        return children_to_str(' | ', expr, expr.exprs, randomize)
    elif isinstance(expr, Repeat):
        return child_to_str('*', expr, expr.child, randomize)
    elif isinstance(expr, Plus):
        return child_to_str('+', expr, expr.child, randomize)
    else:
        assert isinstance(expr, Vars)
        return expr.transition(randomize)
