import copy
from typing import Tuple
import numpy as np
import itertools

from regex_compiler import NodeCreator, CompileStateNFA, generate_inputs, nfa_union, nfa_complement


class RENode:
    def __init__(self):
        return

    def compile(self, _node_creator: NodeCreator) -> CompileStateNFA:
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        """Variables that appear locally within the expression. Compilation should
        consider root.appears()"""
        raise NotImplementedError()

    def rewrites(self, appears: frozenset[str], num: int) -> list['RENode']:
        """Must also return identity"""
        raise NotImplementedError()

    def __eq__(self, b) -> bool:
        raise NotImplementedError()


class RENodeSing(RENode):
    """For expressions like Repeat and Plus, that have a single child"""

    def __init__(self, child: RENode, name: str, con: str):
        self.child = child
        self.name = name
        self.con = con

    def appears(self) -> frozenset[str]:
        return self.child.appears()

    def rewrites(self, appears: frozenset[str], num: int) -> list[RENode]:
        child_rewrites: list[RENode] = self.child.rewrites(appears, num)
        rewrites: list[RENode] = []
        for i in range(len(child_rewrites)):
            rewrites_for_combination: list[RENode] = self.rewrites_with_rewritten_child(
                child_rewrites[i], appears, num)
            rewrites.extend(rewrites_for_combination)
        np.random.shuffle(rewrites)  # type: ignore
        return rewrites[:min(num, len(rewrites))]

    def rewrites_with_rewritten_child(self, child: RENode, appears: frozenset[str], num: int) -> list[RENode]:
        """Must return identity"""
        raise NotImplementedError()

    def __eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        return self.child == b.child


class RENodeMul(RENode):
    """For expressions like Then, And, and Or, that have multiple children"""

    def __init__(self, exprs: list[RENode], name: str, con: str):
        self.exprs: list[RENode] = exprs
        self.name: str = name
        self.con: str = con

    def appears(self) -> frozenset[str]:
        r = set()
        for expr in self.exprs:
            r.update(expr.appears())
        return frozenset(r)

    def rewrites(self, appears: frozenset[str], num: int) -> list[RENode]:
        children: list[list[RENode]] = [
            c.rewrites(appears, num) for c in self.exprs]
        combinations: list[Tuple[RENode]] = list(itertools.product(*children))
        np.random.shuffle(combinations)  # type: ignore
        rewrites: list[RENode] = []
        for i in range(len(combinations)):
            rewritten_children: list[RENode] = list(combinations[i])
            rewrites_for_combination: list[RENode] = self.rewrites_with_rewritten_children(
                rewritten_children, appears, num)
            rewrites.extend(rewrites_for_combination)
        np.random.shuffle(rewrites)  # type: ignore
        return rewrites[:min(num, len(rewrites))]

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        """Must return identity"""
        raise NotImplementedError()

    def __eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        for e1, e2 in zip(self.exprs, b.exprs):
            if e1 != e2:
                return False
        return True


class Or(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, 'Or', '|')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        return nfa_union(compiled, node_creator)

    def demorgan(self, exprs: list[RENode], clean: bool) -> RENode:
        complements: list[RENode] = [Complement(
            x).clean() if clean else Complement(x) for x in exprs]
        return Complement(And(complements))

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(self.demorgan(children, clean=True))
        # results.append(self.demorgan(children, clean=False))
        results.append(Or(children))
        return results


class And(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, 'And', '&')
        self.exprs: list[RENode] = exprs

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled: list[CompileStateNFA] = [
            e.compile(node_creator) for e in self.exprs]
        complements: list[CompileStateNFA] = [
            nfa_complement(c, node_creator) for c in compiled]
        union: CompileStateNFA = nfa_union(complements, node_creator)
        return nfa_complement(union, node_creator)

    def demorgan(self, exprs: list[RENode], clean: bool) -> RENode:
        complements: list[RENode] = [Complement(
            x).clean() if clean else Complement(x) for x in exprs]
        return Complement(Or(complements))

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(self.demorgan(children, clean=True))
        # results.append(self.demorgan(children, clean=False))
        results.append(And(children))
        return results


class Then(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, 'Then', '>')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        for i in range(len(compiled)-1):
            for compiled_i_terminal in compiled[i].terminal_states:
                compiled_i_terminal.t(frozenset({'*'}), compiled[i+1].initial)
        return CompileStateNFA(compiled[0].initial, compiled[-1].terminal_states)

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(Then(children))
        return results


class Repeat(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, 'Repeat', '*')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        new_terminal = node_creator.new_nfa_node()
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({'*'}), new_terminal)
        new_terminal.t(frozenset({'*'}), child.initial)
        return CompileStateNFA(new_terminal, {child.initial})

    def rewrites_with_rewritten_child(self, child: RENode, appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(Repeat(child))
        return results


class Plus(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, name='Plus', con='+')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({'*'}), child.initial)
        return CompileStateNFA(child.initial, child.terminal_states)

    def rewrites_with_rewritten_child(self, child: RENode, appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(Plus(child))
        return results


class Complement(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, 'Complement', '~')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        return nfa_complement(child, node_creator)

    def clean(self) -> RENode:
        if isinstance(self.child, Complement):
            return self.child.child
        else:
            return self

    def rewrites_with_rewritten_child(self, child: RENode, appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(Complement(child))
        if isinstance(child, Complement):
            results.append(child.child)
        return results


class Matcher(RENode):
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

    def rewrites(self, appears: frozenset[str], num: int) -> list[RENode]:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'{"!" if self.negated else ""}{self.content()}'

    def content(self) -> str:
        raise NotImplementedError()


class Symbol(Matcher):
    def __init__(self, symbol: str, negated: bool):
        super().__init__(negated)
        self.symbol = symbol

    def appears(self) -> frozenset[str]:
        return frozenset({self.symbol})

    def matches(self, input_symbol: frozenset[str]) -> bool:
        return self.symbol in input_symbol

    def rewrites(self, appears: frozenset[str], num: int) -> list[RENode]:
        return [self]

    def __eq__(self, b) -> bool:
        if not isinstance(b, Symbol):
            return False
        return self.symbol == b.symbol and self.negated == b.negated

    def content(self) -> str:
        return self.symbol


class Nonempty(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    def matches(self, input_symbol: frozenset[str]) -> bool:
        return len(input_symbol) > 0

    def rewrites(self, appears: frozenset[str], num: int) -> list[RENode]:
        return [self]

    def __eq__(self, b) -> bool:
        return isinstance(b, Nonempty) and self.negated == b.negated

    def content(self) -> str:
        return '_'


class Any(Matcher):
    def __init__(self, negated: bool):
        super().__init__(negated)

    def appears(self) -> frozenset[str]:
        return frozenset()

    def matches(self, input_symbol: frozenset[str]) -> bool:
        return True

    def rewrites(self, appears: frozenset[str], num: int) -> list[RENode]:
        return [self]

    def __eq__(self, b) -> bool:
        return isinstance(b, Any) and self.negated == b.negated

    def content(self) -> str:
        return '.'
