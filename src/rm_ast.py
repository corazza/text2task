import copy
import numpy as np

from rm_compiler import *


class RMExpr:
    def __init__(self):
        return

    def compile(self, _node_creator: RMNodeCreator) -> CompileState:
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        raise NotImplementedError()

    def _internal_repr(self, level: int) -> str:
        raise NotImplementedError()

    def __eq__(self, b) -> bool:
        raise NotImplementedError()


class Vars(RMExpr):
    def __init__(self, symbols: list[str]):
        super().__init__()
        self.symbols = symbols

    def appears(self) -> frozenset[str]:
        r = set()
        for symbol in self.symbols:
            if '!' not in symbol:
                r.add(symbol)
            else:
                r.add(symbol[1:])
        return frozenset(r)

    def transition(self, randomize: bool) -> str:
        symbols = copy.deepcopy(self.symbols)
        if randomize:
            np.random.shuffle(symbols)
        return '&'.join(symbols)

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        terminal = node_creator.new_node(set())
        initial = node_creator.new_node(
            set([(self.transition(False), terminal)]))
        return CompileState(initial, terminal)

    def __str__(self) -> str:
        return self.transition(False)

    def __eq__(self, b) -> bool:
        if not isinstance(b, Vars):
            return False
        return self.symbols == b.symbols

    def __repr__(self) -> str:
        return self._internal_repr(0)

    def _internal_repr(self, level: int) -> str:
        return self.transition(False)


class Or(RMExpr):
    def __init__(self, exprs: list[RMExpr]):
        super().__init__()
        self.exprs = exprs

    def appears(self) -> frozenset[str]:
        r = set()
        for expr in self.exprs:
            r.update(expr.appears())
        return frozenset(r)

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        compiled = list(map(lambda e: e.compile(node_creator), self.exprs))
        initial = node_creator.new_node(set())
        terminal = node_creator.new_node(set())
        for c in compiled:
            initial.t('*', c.initial)
            c.terminal.t('*', terminal)
        return CompileState(initial, terminal)

    def __eq__(self, b) -> bool:
        if not isinstance(b, Or):
            return False
        for e1, e2 in zip(self.exprs, b.exprs):
            if e1 != e2:
                return False
        return True

    def __str__(self):
        r = ''
        for i in range(len(self.exprs)-1):
            r = f'{r}{self.exprs[i]} | '
        return f'({r}{self.exprs[-1]})'

    def _internal_repr(self, level: int) -> str:
        es = list(map(lambda e: e._internal_repr(level+1), self.exprs))
        r = f'Or:'
        for e in es:
            r = f'{r}\n{"  " * level}{e}'
        return r

    def __repr__(self) -> str:
        return self._internal_repr(1)


class Then(RMExpr):
    def __init__(self, exprs: list[RMExpr]):
        super().__init__()
        self.exprs = exprs

    def appears(self) -> frozenset[str]:
        r = set()
        for expr in self.exprs:
            r.update(expr.appears())
        return frozenset(r)

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        compiled = list(map(lambda e: e.compile(node_creator), self.exprs))
        for i in range(len(compiled)-1):
            compiled[i].terminal.t('*', compiled[i+1].initial)
        return CompileState(compiled[0].initial, compiled[-1].terminal)

    def __eq__(self, b) -> bool:
        if not isinstance(b, Then):
            return False
        for e1, e2 in zip(self.exprs, b.exprs):
            if e1 != e2:
                return False
        return True

    def __str__(self):
        r = ''
        for i in range(len(self.exprs)-1):
            r = f'{r}{self.exprs[i]} -> '
        return f'({r}{self.exprs[-1]})'

    def _internal_repr(self, level: int) -> str:
        es = list(map(lambda e: e._internal_repr(level+1), self.exprs))
        r = f'Then:'
        for e in es:
            r = f'{r}\n{"  " * level}{e}'
        return r

    def __repr__(self) -> str:
        return self._internal_repr(1)


class Repeat(RMExpr):
    def __init__(self, child: RMExpr):
        super().__init__()
        self.child = child

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        child = self.child.compile(node_creator)
        child.terminal.t('*', child.initial)
        return CompileState(child.terminal, child.initial)

    def __eq__(self, b) -> bool:
        if not isinstance(b, Repeat):
            return False
        return self.child == b.child

    def __str__(self):
        return f'({self.child})*'

    def _internal_repr(self, level: int) -> str:
        return f'Repeat:\n{"  " * level}{self.child._internal_repr(level)}'

    def __repr__(self) -> str:
        return self._internal_repr(1)


class Plus(RMExpr):
    def __init__(self, child: RMExpr):
        super().__init__()
        self.child = child

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        child = self.child.compile(node_creator)
        child.terminal.t('*', child.initial)
        return CompileState(child.initial, child.terminal)

    def __eq__(self, b) -> bool:
        if not isinstance(b, Plus):
            return False
        return self.child == b.child

    def __str__(self):
        return f'({self.child})+'

    def _internal_repr(self, level: int) -> str:
        return f'Plus:\n{"  " * level}{self.child._internal_repr(level)}'

    def __repr__(self) -> str:
        return self._internal_repr(1)
