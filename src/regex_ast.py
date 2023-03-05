import copy
import numpy as np

from regex_compiler import NodeCreator, CompileStateNFA, generate_inputs, nfa_union, nfa_complement
# from visualization import visualize_compilestate


class RMExpr:
    def __init__(self):
        return

    def compile(self, _node_creator: NodeCreator) -> CompileStateNFA:
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        """Variables that appear locally within the expression. Compilation should
        consider root.appears()"""
        raise NotImplementedError()

    def _internal_repr(self, level: int) -> str:
        raise NotImplementedError()

    def __eq__(self, b) -> bool:
        raise NotImplementedError()


class RMExprSing(RMExpr):
    """For expressions like Repeat and Plus, that have a single child"""

    def __init__(self, child: RMExpr, name: str, con: str):
        self.child = child
        self.name = name
        self.con = con

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def __eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        return self.child == b.child

    def __str__(self):
        return f'({self.child}){self.con}'

    def _internal_repr(self, level: int) -> str:
        return f'{self.name}:\n{"  " * level}{self.child._internal_repr(level)}'

    def __repr__(self) -> str:
        return self._internal_repr(1)


class RMExprMul(RMExpr):
    """For expressions like Then, And, and Or, that have multiple children"""

    def __init__(self, exprs: list[RMExpr], name: str, con: str):
        self.exprs: list[RMExpr] = exprs
        self.name: str = name
        self.con: str = con

    def appears(self) -> frozenset[str]:
        r = set()
        for expr in self.exprs:
            r.update(expr.appears())
        return frozenset(r)

    def __str__(self):
        r = ''
        for i in range(len(self.exprs)-1):
            r = f'{r}{self.exprs[i]} {self.con} '
        return f'({r}{self.exprs[-1]})'

    def _internal_repr(self, level: int) -> str:
        es = list(map(lambda e: e._internal_repr(level+1), self.exprs))
        r = f'{self.name}:'
        for e in es:
            r = f'{r}\n{"  " * level}{e}'
        return r

    def __eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        for e1, e2 in zip(self.exprs, b.exprs):
            if e1 != e2:
                return False
        return True

    def __repr__(self) -> str:
        return self._internal_repr(1)


class Or(RMExprMul):
    def __init__(self, exprs: list[RMExpr]):
        super().__init__(exprs, 'Or', '|')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        return nfa_union(compiled, node_creator)


class And(RMExprMul):
    def __init__(self, exprs: list[RMExpr]):
        super().__init__(exprs, 'And', '&')
        self.exprs: list[RMExpr] = exprs

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled: list[CompileStateNFA] = [
            e.compile(node_creator) for e in self.exprs]
        complements: list[CompileStateNFA] = [
            nfa_complement(c, node_creator) for c in compiled]
        union: CompileStateNFA = nfa_union(complements, node_creator)
        return nfa_complement(union, node_creator)


class Then(RMExprMul):
    def __init__(self, exprs: list[RMExpr]):
        super().__init__(exprs, 'Then', '->')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        for i in range(len(compiled)-1):
            for compiled_i_terminal in compiled[i].terminal_states:
                compiled_i_terminal.t(frozenset({'*'}), compiled[i+1].initial)
        return CompileStateNFA(compiled[0].initial, compiled[-1].terminal_states)


class Repeat(RMExprSing):
    def __init__(self, child: RMExpr):
        super().__init__(child, 'Repeat', '*')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        new_terminal = node_creator.new_nfa_node()
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({'*'}), new_terminal)
        new_terminal.t(frozenset({'*'}), child.initial)
        return CompileStateNFA(new_terminal, {child.initial})


class Plus(RMExprSing):
    def __init__(self, child: RMExpr):
        super().__init__(child, name='Plus', con='+')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({'*'}), child.initial)
        return CompileStateNFA(child.initial, child.terminal_states)


class Complement(RMExprSing):
    def __init__(self, child: RMExpr):
        super().__init__(child, 'Complement', '~')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        return nfa_complement(child, node_creator)


class Matcher(RMExpr):
    def __init__(self, negated: bool):
        super().__init__()
        self.negated: bool = negated

    def matches(self, input_symbol: frozenset[str]) -> bool:
        raise NotImplementedError()

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        terminal = node_creator.new_nfa_node()
        sink = node_creator.new_nfa_sink()
        initial = node_creator.new_nfa_node()
        for input_symbol in generate_inputs(node_creator.appears):
            terminal.t(input_symbol, sink)
            if self.matches(input_symbol) and not self.negated or not self.matches(input_symbol) and self.negated:
                initial.t(input_symbol, terminal)
            else:
                initial.t(input_symbol, sink)
        return CompileStateNFA(initial, {terminal})

    def __repr__(self) -> str:
        return self._internal_repr(0)

    def _internal_repr(self, level: int) -> str:
        return str(self)


class Symbol(Matcher):
    def __init__(self, symbol: str, negated: bool):
        super().__init__(negated)
        self.symbol = symbol

    def appears(self) -> frozenset[str]:
        return frozenset({self.symbol})

    def matches(self, input_symbol: frozenset[str]) -> bool:
        return self.symbol in input_symbol

    def __str__(self) -> str:
        return f'!{self.symbol}' if self.negated else self.symbol

    def __eq__(self, b) -> bool:
        if not isinstance(b, Symbol):
            return False
        return self.symbol == b.symbol and self.negated == b.negated


class Nonempty(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    def matches(self, input_symbol: frozenset[str]) -> bool:
        return len(input_symbol) > 0

    def __str__(self) -> str:
        return '_' if not self.negated else '!_'

    def __eq__(self, b) -> bool:
        return isinstance(b, Nonempty) and self.negated == b.negated


class Any(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    def matches(self, input_symbol: frozenset[str]) -> bool:
        return True

    def __str__(self) -> str:
        return '.' if not self.negated else '!.'

    def __eq__(self, b) -> bool:
        return isinstance(b, Any) and self.negated == b.negated
