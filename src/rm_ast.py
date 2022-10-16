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


class Var(RMExpr):
    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    def appears(self) -> frozenset[str]:
        return frozenset(self.symbol)

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        terminal = node_creator.new_node(set())
        initial = node_creator.new_node(set([(self.symbol, terminal)]))
        return CompileState(initial, terminal)

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return self._internal_repr(0)

    def _internal_repr(self, level: int) -> str:
        return self.symbol


class Or(RMExpr):
    def __init__(self, left: RMExpr, right: RMExpr):
        super().__init__()
        self.left = left
        self.right = right

    def appears(self) -> frozenset[str]:
        return frozenset.union(self.left.appears(), self.right.appears())

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        left = self.left.compile(node_creator)
        right = self.right.compile(node_creator)
        initial = node_creator.new_node(set())
        terminal = node_creator.new_node(set())
        initial.t('*', left.initial)
        initial.t('*', right.initial)
        left.terminal.t('*', terminal)
        right.terminal.t('*', terminal)
        return CompileState(initial, terminal)

    def __str__(self):
        return f'({self.left} | {self.right})'

    def _internal_repr(self, level: int) -> str:
        return f'Or:\n{"  " * level}left: {self.left._internal_repr(level+1)}\n{"  " * level}right: {self.right._internal_repr(level+1)}'

    def __repr__(self) -> str:
        return self._internal_repr(1)


class Then(RMExpr):
    def __init__(self, left: RMExpr, right: RMExpr):
        super().__init__()
        self.left = left
        self.right = right

    def appears(self) -> frozenset[str]:
        return frozenset.union(self.left.appears(), self.right.appears())

    def compile(self, node_creator: RMNodeCreator) -> CompileState:
        left = self.left.compile(node_creator)
        right = self.right.compile(node_creator)
        left.terminal.t('*', right.initial)
        return CompileState(left.initial, right.terminal)

    def __str__(self):
        return f'({self.left} -> {self.right})'

    def _internal_repr(self, level: int) -> str:
        return f'Then:\n{"  " * level}left: {self.left._internal_repr(level+1)}\n{"  " * level}right: {self.right._internal_repr(level+1)}'

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

    def __str__(self):
        return f'({self.child})*'

    def _internal_repr(self, level: int) -> str:
        return f'Repeat:\n{"  " * level}{self.child._internal_repr(level)}'

    def __repr__(self) -> str:
        return self._internal_repr(1)
