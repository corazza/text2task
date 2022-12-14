from curses.ascii import isalpha, islower, isspace
from typing import Iterator
import itertools
import more_itertools

from transition_ast import *


class Token:
    pass


class SymbolT(Token):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol


class NotT(Token):
    pass


class AndT(Token):
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
    symbol_buffer = []
    for c in src:
        if isalpha(c):
            symbol_buffer.append(c)
        else:
            if len(symbol_buffer) > 0:
                yield SymbolT(''.join(symbol_buffer))
                symbol_buffer = []
            if c == ' ':
                continue
            elif c == '|':
                yield OrT()
            elif c == '&':
                yield AndT()
            elif c == '!':
                yield NotT()
            elif c == '(':
                yield OpenT()
            elif c == ')':
                yield CloseT()
            else:
                raise ValueError(f'unrecognized character {c}')
    if len(symbol_buffer) != 0:
        yield SymbolT(''.join(symbol_buffer))
    yield EndT()

# expression -> term
# expression -> term | expression

# term -> factor
# term -> factor & term

# factor -> !factor
# factor -> (expression)
# factor -> symbol


def parse_expression(lex: more_itertools.peekable) -> TExp:
    left = parse_term(lex)
    token = lex.peek()
    if isinstance(token, OrT):
        next(lex)
        right = parse_expression(lex)
        return Or(left, right)
    return left


def parse_term(lex: more_itertools.peekable) -> TExp:
    left = parse_factor(lex)
    token = lex.peek()
    if isinstance(token, AndT):
        next(lex)
        right = parse_term(lex)
        return And(left, right)
    return left


def parse_factor(lex: more_itertools.peekable) -> TExp:
    token = lex.peek()
    if isinstance(token, OpenT):
        next(lex)
        term = parse_expression(lex)
        token = next(lex)
        assert isinstance(token, CloseT), 'expected )'
        return term
    elif isinstance(token, NotT):
        next(lex)
        child = parse_factor(lex)
        return Not(child)
    else:
        return parse_symbol(lex)


def parse_symbol(lex: Iterator[Token]) -> TExp:
    token = next(lex)
    assert isinstance(token, SymbolT), f'expected symbol, got {token}'
    return Var(token.symbol)


def parse(src: str) -> TExp:
    return parse_expression(more_itertools.peekable(lex(src)))
