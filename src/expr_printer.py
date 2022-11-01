from rm_ast import *


def wrap(parent: RMExpr, child: RMExpr, connect_then: bool) -> bool:
    if isinstance(parent, Repeat) or isinstance(parent, Plus):
        return True
    elif isinstance(child, Vars):
        return False
    else:
        if isinstance(parent, Or) and isinstance(child, Then):
            return connect_then
        return isinstance(parent, Then) and isinstance(child, Or)


def children_to_str(con: str, parent: RMExpr, children: list[RMExpr], randomize: bool, connect_then: bool) -> str:
    r = ''
    for i in range(len(children)-1):
        if wrap(parent, children[i], connect_then):
            r = f'{r}({expr_to_str(children[i], randomize,connect_then)}){con}'
        else:
            r = f'{r}{expr_to_str(children[i], randomize,connect_then)}{con}'
    if wrap(parent, children[-1], connect_then):
        r = f'{r}({expr_to_str(children[-1], randomize,connect_then)})'
    else:
        r = f'{r}{expr_to_str(children[-1], randomize, connect_then)}'
    return r


def child_to_str(op: str, parent: RMExpr, child: RMExpr, randomize: bool, connect_then: bool) -> str:
    r = expr_to_str(child, randomize, connect_then)
    if wrap(parent, child, connect_then):
        r = f'({r}){op}'
    else:
        r = f'{r}{op}'
    return r


def expr_to_str(expr: RMExpr, randomize: bool, connect_then: bool) -> str:
    if isinstance(expr, Then):
        connector = ' -> ' if connect_then else ' '
        return children_to_str(connector, expr, expr.exprs, randomize, connect_then)
    if isinstance(expr, Or):
        return children_to_str(' | ', expr, expr.exprs, randomize, connect_then)
    elif isinstance(expr, Repeat):
        return child_to_str('*', expr, expr.child, randomize, connect_then)
    elif isinstance(expr, Plus):
        return child_to_str('+', expr, expr.child, randomize, connect_then)
    else:
        assert isinstance(expr, Vars)
        return expr.transition(randomize)
