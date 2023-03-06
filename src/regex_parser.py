from typing import Iterator
import itertools
import more_itertools
from torch import isin

from regex_ast import *
from regex_lexer import *

# expression -> term
# expression -> term | expression

# term -> step
# term -> step > term

# step -> factor
# step -> factor & step

# factor -> (expression)
# factor -> (expression)+
# factor -> (expression)*
# factor -> (expression)~
# factor -> matcher
# factor -> !matcher

# matcher -> symbol
# matcher -> _
# matcher -> .


def parse_expression(lex: more_itertools.peekable) -> RENode:
    left = parse_term(lex)
    token = lex.peek()
    if isinstance(token, OrT):
        next(lex)
        right = parse_expression(lex)
        or_terms = [left]
        if isinstance(right, Or):
            or_terms.extend(right.exprs)
        else:
            or_terms.append(right)
        return Or(or_terms)
    return left


def parse_term(lex: more_itertools.peekable) -> RENode:
    left = parse_step(lex)
    token = lex.peek()
    if isinstance(token, ThenT):
        next(lex)
        right = parse_term(lex)
        then_terms = [left]
        if isinstance(right, Then):
            then_terms.extend(right.exprs)
        else:
            then_terms.append(right)
        return Then(then_terms)
    return left


def parse_step(lex: more_itertools.peekable) -> RENode:
    left = parse_factor(lex)
    token = lex.peek()
    if isinstance(token, AndT):
        next(lex)
        right = parse_step(lex)
        and_terms = [left]
        if isinstance(right, And):
            and_terms.extend(right.exprs)
        else:
            and_terms.append(right)
        return And(and_terms)
    return left


def parse_factor(lex: more_itertools.peekable) -> RENode:
    token = lex.peek()
    if isinstance(token, OpenT):
        next(lex)
        expression = parse_expression(lex)
        token = next(lex)
        assert isinstance(token, CloseT), 'expected )'
        token = lex.peek()
        if isinstance(token, RepeatT):
            next(lex)
            return Repeat(expression)
        elif isinstance(token, PlusT):
            next(lex)
            return Plus(expression)
        elif isinstance(token, ComplementT):
            next(lex)
            return Complement(expression)
        else:
            return expression
    elif isinstance(token, NotT):
        next(lex)
        return parse_matcher(lex, True)
    else:
        return parse_matcher(lex, False)


def parse_matcher(lex: more_itertools.peekable, negated: bool) -> RENode:
    token = next(lex)
    if isinstance(token, SymbolT):
        return Symbol(token.symbol, negated)
    elif isinstance(token, NonemptyT):
        return Nonempty(negated)
    else:
        assert isinstance(token, AnyT), 'expected AnyT'
        return Any(negated)


def parse(src: str) -> RENode:
    lexed = more_itertools.peekable(lex(src))
    expr = parse_expression(lexed)
    token = next(lexed)
    if not isinstance(token, EndT):
        raise ValueError(
            f"root expression doesn't end with End, continues: {token}")
    return expr
