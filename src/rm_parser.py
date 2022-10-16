from curses.ascii import isalpha, islower, isspace
from typing import Iterator
import itertools
import more_itertools

from rm_ast import *


class Token:
    pass


class SymbolT(Token):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol


class RepeatT(Token):
    pass


class ThenT(Token):
    pass


class OrT(Token):
    pass


class OpenT(Token):
    pass


class CloseT(Token):
    pass


class EndT(Token):
    pass


def lex(src: str) -> Iterator[Token]:
    nospaces = list(filter(lambda c: not isspace(c), src))
    for c in nospaces:
        if isalpha(c):
            assert islower(c), 'propositional variables must be lower-case'
            yield SymbolT(c)
        elif c == '|':
            yield OrT()
        elif c == '*':
            yield RepeatT()
        elif c == '(':
            yield OpenT()
        elif c == ')':
            yield CloseT()
        else:
            raise ValueError(f'unrecognized character {c}')
    yield EndT()


# expression -> term
# expression -> term | expression

# term -> factor
# term -> factor  term

# factor -> factor*
# factor -> (expression)
# factor -> symbol


def parse_expression(lex: more_itertools.peekable) -> RMExp:
    left = parse_term(lex)
    token = lex.peek()
    if isinstance(token, OrT):
        next(lex)
        right = parse_expression(lex)
        return Or(left, right)
    return left


def parse_term(lex: more_itertools.peekable) -> RMExp:
    left = parse_factor(lex)
    token = lex.peek()
    if isinstance(token, SymbolT) or isinstance(token, OpenT):
        right = parse_term(lex)
        return Then(left, right)
    return left


def parse_factor(lex: more_itertools.peekable) -> RMExp:
    token = lex.peek()
    if isinstance(token, OpenT):
        next(lex)
        term = parse_expression(lex)
        token = next(lex)
        assert isinstance(token, CloseT), 'expected )'
    else:
        term = parse_symbol(lex)
    token = lex.peek()
    if isinstance(token, RepeatT):
        next(lex)
        return Repeat(term)
    else:
        return term


def parse_symbol(lex: Iterator[Token]) -> RMExp:
    token = next(lex)
    assert isinstance(token, SymbolT), f'expected symbol, got {token}'
    return Var(token.symbol)


def parse(src: str) -> RMExp:
    return parse_expression(more_itertools.peekable(lex(src)))
