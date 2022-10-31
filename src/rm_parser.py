from curses.ascii import isalpha, islower, isspace
from typing import Iterator
import itertools
import more_itertools
from torch import isin

from rm_ast import *


class Token:
    def __repr__(self):
        raise NotImplementedError()

    def __str__(self):
        return repr(self)


class SymbolT(Token):
    def __init__(self, symbol):
        super().__init__()
        self.symbol = symbol

    def __repr__(self):
        return f'S<{self.symbol}>'


class RepeatT(Token):
    def __repr__(self):
        return '*'


class PlusT(Token):
    def __repr__(self):
        return '+'


class ThenT(Token):
    def __repr__(self):
        return '->'


class OrT(Token):
    def __repr__(self):
        return '|'


class NotT(Token):
    def __repr__(self):
        return '!'


class AndT(Token):
    def __repr__(self):
        return '&'


class OpenT(Token):
    def __repr__(self):
        return '('


class CloseT(Token):
    def __repr__(self):
        return ')'


class EndT(Token):
    def __repr__(self):
        return 'End'


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
            elif c == '*':
                yield RepeatT()
            elif c == '+':
                yield PlusT()
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
# term -> factor term

# factor -> (expression)
# factor -> (expression)*
# factor -> (expression)+
# factor -> vars

# vars -> symbol
# vars -> symbol & vars

# symbol -> var
# symbol -> !var


def parse_expression(lex: more_itertools.peekable) -> RMExpr:
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


def parse_term(lex: more_itertools.peekable) -> RMExpr:
    left = parse_factor(lex)
    token = lex.peek()
    if isinstance(token, NotT) or isinstance(token, SymbolT) or isinstance(token, OpenT):
        right = parse_term(lex)
        then_terms = [left]
        if isinstance(right, Then):
            then_terms.extend(right.exprs)
        else:
            then_terms.append(right)
        return Then(then_terms)
    return left


def parse_factor(lex: more_itertools.peekable) -> RMExpr:
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
        else:
            return expression
    else:
        return parse_vars(lex)


def parse_vars(lex: more_itertools.peekable) -> Vars:
    symbol = parse_symbol(lex)
    token = lex.peek()
    if isinstance(token, AndT):
        next(lex)
        vars = parse_vars(lex)
        vars.symbols.insert(0, symbol)
        return vars
    else:
        return Vars([symbol])


def parse_symbol(lex: Iterator[Token]) -> str:
    token = next(lex)
    if isinstance(token, NotT):
        symbol = next(lex)
        r = '!'
    else:
        r = ''
        symbol = token
    assert isinstance(symbol, SymbolT), f'expected symbol, got {token}'
    return f'{r}{symbol.symbol}'


def parse(src: str) -> RMExpr:
    return parse_expression(more_itertools.peekable(lex(src)))
