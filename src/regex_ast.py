import copy
import IPython
from typing import Tuple
import numpy as np
import itertools

from regex_compiler import NodeCreator, CompileStateNFA, generate_inputs, nfa_union, nfa_complement
from consts import *


def shuffle(xs: list):
    xs = xs.copy()
    np.random.shuffle(xs)
    return xs


def shuffle_pick_n(xs: list, n: int) -> list:
    xs = shuffle(xs)
    return xs[:min(n, len(xs))]


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
        return shuffle_pick_n(rewrites, num)

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
        return shuffle_pick_n(rewrites, num)

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        """Must return identity"""
        raise NotImplementedError()

    def clean(self) -> RENode:
        if len(self.exprs) == 1:
            return self.exprs[0]
        return self

    def _ordered__eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        for e1, e2 in zip(self.exprs, b.exprs):
            if e1 != e2:
                return False
        return True

    def _unordered__eq__(self, b) -> bool:
        if not isinstance(b, self.__class__):
            return False
        if len(self.exprs) != len(b.exprs):
            return False
        for child in self.exprs:
            if child not in b.exprs:
                return False
        return True


def reorder_children(children: list[RENode]) -> list[list[RENode]]:
    indices = list(range(len(children)))
    results: list[list[RENode]] = []
    for perm in itertools.permutations(indices):
        p_children: list[RENode] = [children[indices[i]] for i in perm]
        if p_children != children:  # skipping already-included identity permutation
            results.append(p_children)
    return results


class Or(RENodeMul):
    def __init__(self, exprs: list[RENode]):
        super().__init__(exprs, 'Or', '|')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        compiled = [e.compile(node_creator) for e in self.exprs]
        return nfa_union(compiled, node_creator)

    def _rewrite_demorgan(self, exprs: list[RENode], clean: bool) -> RENode:
        complements: list[RENode] = [Complement(
            x).remove_double_negation() if clean else Complement(x) for x in exprs]
        return Complement(And(complements).clean()).remove_double_negation()

    def _rewrite_distributive(self, children: list[RENode], num: int) -> list[RENode]:
        result: list[RENode] = []
        for subset_size in range(2, len(children)+1):
            for subset in itertools.combinations(children, subset_size):
                if any([not isinstance(child, And) for child in subset]):
                    continue
                all_subchildren: list[list[RENode]] = [
                    child.exprs for child in subset]  # type: ignore
                factor_out: list[RENode] = []
                for combination in itertools.product(*all_subchildren):
                    if any([combination[0] != x for x in combination]):
                        continue
                    factor_out.append(combination[0])
                for factor_subset_size in range(1, len(factor_out)+1):
                    for factor_subset in itertools.combinations(factor_out, factor_subset_size):
                        new_or_terms: list[RENode] = []
                        for child in subset:
                            assert isinstance(child, And)
                            new_and_terms: list[RENode] = []
                            for and_child in child.exprs:
                                if and_child not in factor_subset:
                                    new_and_terms.append(and_child)
                            new_or_terms.append(And(new_and_terms).clean())
                        factored_out = Or(new_or_terms).clean()
                        factorization = And(
                            [*factor_subset, factored_out]).clean()
                        result_or_terms: list[RENode] = []
                        result_or_terms.append(factorization)
                        for child in children:
                            if child not in subset:
                                result_or_terms.append(child)
                        result.append(Or(result_or_terms).clean())
        return shuffle_pick_n(result, num)

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        # because children are rewritten
        results: list[RENode] = [Or(children).clean()]
        results.extend(shuffle_pick_n([Or(children).clean()
                       for children in reorder_children(children)], num))
        if np.random.random() < REWRITE_EXPANSIVE_DEMORGAN_PROB:
            results.extend(shuffle_pick_n([self._rewrite_demorgan(children, clean=True)
                                           for children in reorder_children(children)], num))
        results.extend(shuffle_pick_n(
            self._rewrite_distributive(children, num), num))
        return results

    def __eq__(self, b) -> bool:
        return self._unordered__eq__(b)


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

    def _rewrite_demorgan(self, exprs: list[RENode], clean: bool) -> RENode:
        complements: list[RENode] = [Complement(
            x).remove_double_negation() if clean else Complement(x) for x in exprs]
        return Complement(Or(complements).clean()).remove_double_negation()

    def _rewrite_distributive(self, children: list[RENode], num: int) -> list[RENode]:
        result: list[RENode] = []
        for (i, or_child) in enumerate(children):
            if not isinstance(or_child, Or):
                continue
            for subset_size in range(1, len(children)):
                for subset in itertools.combinations(children, subset_size):
                    if or_child in subset:
                        continue
                    conjuncts: list[RENode] = []
                    for (k, child) in enumerate(children):
                        if k != i and child not in subset:
                            conjuncts.append(child)
                    or_terms: list[RENode] = [
                        And(shuffle([*subset, child])).clean() for child in or_child.exprs]
                    conjuncts.append(Or(shuffle(or_terms)).clean())
                    np.random.shuffle(conjuncts)  # type: ignore
                    if len(conjuncts) > 1:
                        result.append(And(conjuncts).clean())
                    else:
                        result.append(conjuncts[0])
        return result

    # TODO 10 random for every extend - so all have semi-equal probability of inclusion

    def rewrites_with_rewritten_children(self, children: list[RENode], appears: frozenset[str], num: int) -> list[RENode]:
        # because children are rewritten
        results: list[RENode] = [And(children).clean()]
        results.extend(shuffle_pick_n([And(children).clean()
                       for children in reorder_children(children)], num))
        if np.random.random() < REWRITE_EXPANSIVE_DEMORGAN_PROB:
            results.extend(shuffle_pick_n([self._rewrite_demorgan(children, clean=True)
                                           for children in reorder_children(children)], num))
        results.extend(shuffle_pick_n(
            self._rewrite_distributive(children, num), num))
        return results

    def __eq__(self, b) -> bool:
        return self._unordered__eq__(b)


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

    def __eq__(self, b):
        return self._ordered__eq__(b)


class Repeat(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, 'Repeat', '*')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        new_initial = node_creator.new_nfa_node()
        new_initial.t(frozenset({'*'}), child.initial)
        for child_terminal in child.terminal_states:
            child_terminal.t(frozenset({'*'}), new_initial)
        return CompileStateNFA(new_initial, {new_initial})

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
        results.append(Then([child, Repeat(child)]))
        return results


class Complement(RENodeSing):
    def __init__(self, child: RENode):
        super().__init__(child, 'Complement', '~')

    def compile(self, node_creator: NodeCreator) -> CompileStateNFA:
        child = self.child.compile(node_creator)
        return nfa_complement(child, node_creator)

    def remove_double_negation(self) -> RENode:
        if isinstance(self.child, Complement):
            return self.child.child
        else:
            return self

    def _demorgans(self, child: RENode) -> list[RENode]:
        if isinstance(child, Or):
            return [And([Complement(child_child).remove_double_negation() for child_child in child.exprs]).clean()]
        elif isinstance(child, And):
            return [Or([Complement(child_child).remove_double_negation() for child_child in child.exprs]).clean()]
        else:
            return []

    def rewrites_with_rewritten_child(self, child: RENode, appears: frozenset[str], num: int) -> list[RENode]:
        results: list[RENode] = []
        results.append(Complement(child).remove_double_negation())
        if isinstance(child, Complement):
            results.append(child.child)
        results.extend(shuffle_pick_n(self._demorgans(child), num))
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
        return ':'


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
