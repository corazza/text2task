from typing import Iterator
from curses.ascii import isalpha


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
        return '>'


class OrT(Token):
    def __repr__(self):
        return '|'


class NotT(Token):
    def __repr__(self):
        return '!'


class AnyT(Token):
    def __repr__(self):
        return '.'


class NonemptyT(Token):
    def __repr__(self):
        return '_'


class ComplementT(Token):
    def __repr__(self):
        return '~'


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
        if isalpha(c) or c == '$' or ((c == '_' or c.isnumeric()) and len(symbol_buffer) > 0):
            symbol_buffer.append(c)
        else:
            if len(symbol_buffer) > 0:
                yield SymbolT(''.join(symbol_buffer))
                symbol_buffer = []
            if c == ' ':
                continue
            elif c == '|':
                yield OrT()
            elif c == '>':
                yield ThenT()
            elif c == '&':
                yield AndT()
            elif c == '!':
                yield NotT()
            elif c == '.':
                yield AnyT()
            elif c == ':':
                yield NonemptyT()
            elif c == '*':
                yield RepeatT()
            elif c == '+':
                yield PlusT()
            elif c == '~':
                yield ComplementT()
            elif c == '(':
                yield OpenT()
            elif c == ')':
                yield CloseT()
            else:
                print(src)
                raise ValueError(f'unrecognized character `{c}`')
    if len(symbol_buffer) != 0:
        yield SymbolT(''.join(symbol_buffer))
    yield EndT()
