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

    def rewrites(self) -> list['RENode']:
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

    def rewrites(self) -> list[RENode]:
        child_rewrites: list[RENode] = self.child.rewrites()
        rewrites: list[RENode] = []
        for i in range(len(child_rewrites)):
            rewrites_for_combination: list[RENode] = self.rewrites_with_rewritten_child(
                child_rewrites[i])
            rewrites.extend(rewrites_for_combination)
        return shuffle_pick_n(rewrites, REWRITE_INFLATION_LIMIT)

    def rewrites_with_rewritten_child(self, child: RENode) -> list[RENode]:
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

    def rewrites(self) -> list[RENode]:
        children: list[list[RENode]] = [
            c.rewrites() for c in self.exprs]
        combinations: list[Tuple[RENode]] = list(itertools.product(*children))
        np.random.shuffle(combinations)  # type: ignore
        rewrites: list[RENode] = []
        for i in range(len(combinations)):
            rewritten_children: list[RENode] = list(combinations[i])
            rewrites_for_combination: list[RENode] = self.rewrites_with_rewritten_children(
                rewritten_children)
            rewrites.extend(rewrites_for_combination)
        return shuffle_pick_n(rewrites, REWRITE_INFLATION_LIMIT)

    def rewrites_with_rewritten_children(self, children: list[RENode]) -> list[RENode]:
        """Must return identity"""
        raise NotImplementedError()

    def _clean_single(self) -> RENode:
        if len(self.exprs) == 1:
            return self.exprs[0]
        return self

    def _clean_child_same(self) -> 'RENodeMul':
        new_children: list[RENode] = []
        for child in self.exprs:
            if isinstance(child, self.__class__):
                new_children.extend(child.exprs)
            else:
                new_children.append(child)
        if isinstance(self, Or):
            return Or(new_children)
        elif isinstance(self, And):
            return And(new_children)
        else:
            assert isinstance(self, Then)
            return Then(new_children)

    def clean(self) -> RENode:
        return self._clean_child_same()._clean_single()

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


def slice_from_back(a: list, i: int) -> list:
    return list(reversed(list(reversed(a))[0:i]))


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

    def _rewrite_distributive_and_inverse(self, children: list[RENode]) -> list[RENode]:
        """
        (A & D & B) | (A & D & C) -> A & D & (B | C)
        """
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
        return result

    def _rewrite_distributive_then_inverse(self, children: list[RENode]) -> list[RENode]:
        """
        B > A | C > A | D -> (B | C) > A | D
        """
        result: list[RENode] = []
        for subset_size in range(2, len(children)+1):
            for subset in itertools.combinations(children, subset_size):
                if any([not isinstance(child, Then) for child in subset]):
                    continue
                lengths_match: list[bool] = [len(subset_child.exprs) == len(  # type:ignore
                    subset[0].exprs) for subset_child in subset]  # type:ignore
                if not all(lengths_match):
                    continue
                length: int = len(subset[0].exprs)  # type: ignore
                for left_i in range(length):
                    for right_i in range(length):
                        if left_i + right_i >= length or left_i + right_i == 0:
                            continue
                        left_parts: list[list[RENode]] = []
                        middle_parts: list[list[RENode]] = []
                        right_parts: list[list[RENode]] = []
                        for then_expr in subset:
                            left_parts.append(
                                then_expr.exprs[0:left_i])  # type: ignore
                            middle_parts.append(
                                then_expr.exprs[left_i:(length-right_i)])  # type: ignore
                            right_parts.append(slice_from_back(
                                then_expr.exprs, right_i))  # type: ignore
                        left_parts_equal: bool = all(
                            [part == left_parts[0] for part in left_parts])
                        right_parts_equal: bool = all(
                            [part == right_parts[0] for part in right_parts])
                        if left_parts_equal and right_parts_equal:
                            or_term = Or([Then(part).clean()
                                         for part in middle_parts]).clean()
                            then_term = Then(
                                [*left_parts[0], or_term, *right_parts[0]]).clean()
                            final_or_terms: list[RENode] = [then_term]
                            for child in children:
                                if child not in subset:
                                    final_or_terms.append(child)
                            result.append(Or(final_or_terms).clean())
        return result

    def rewrites_with_rewritten_children(self, children: list[RENode]) -> list[RENode]:
        # because children are rewritten
        results: list[RENode] = [Or(children).clean()]
        results.extend(shuffle_pick_n([Or(children).clean()
                       for children in reorder_children(children)], REWRITE_INFLATION_LIMIT_REORDER))
        results.extend(shuffle_pick_n([self._rewrite_demorgan(children, clean=True)
                                       for children in reorder_children(children)], REWRITE_INFLATION_LIMIT_DEMORGAN))
        results.extend(shuffle_pick_n(
            self._rewrite_distributive_and_inverse(children), REWRITE_INFLATION_LIMIT_DISTRIBUTIVE))
        results.extend(shuffle_pick_n(
            self._rewrite_distributive_then_inverse(children), REWRITE_INFLATION_LIMIT_DISTRIBUTIVE))
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

    def _rewrite_distributive_and(self, children: list[RENode]) -> list[RENode]:
        """
        A & (B | C) -> A & B | A & C
        """
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

    def rewrites_with_rewritten_children(self, children: list[RENode]) -> list[RENode]:
        # because children are rewritten
        results: list[RENode] = [And(children).clean()]
        results.extend(shuffle_pick_n([And(children).clean()
                       for children in reorder_children(children)], REWRITE_INFLATION_LIMIT_REORDER))
        results.extend(shuffle_pick_n([self._rewrite_demorgan(children, clean=True)
                                       for children in reorder_children(children)], REWRITE_INFLATION_LIMIT_DEMORGAN))
        results.extend(shuffle_pick_n(
            self._rewrite_distributive_and(children), REWRITE_INFLATION_LIMIT_DISTRIBUTIVE))
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

    def _rewrite_distributive_then(self, children: list[RENode]) -> list[RENode]:
        """
        A > (B | C) > D -> (A > B | A > C) > D
        A > (B | C) > D -> A > (B > D | C > D)
        A > (B | C) > D -> A > B > D | A > C > D
        """
        result: list[RENode] = []
        for (i, or_child) in enumerate(children):
            if not isinstance(or_child, Or):
                continue
            for left_i in range(i+1):
                for right_i in range(len(children) - i):
                    if left_i + right_i == 0:
                        continue
                    middle_parts: list[RENode] = []
                    left_part: list[RENode] = slice_from_back(
                        children[0:i], left_i)
                    right_part: list[RENode] = children[i+1:i+1+right_i]
                    for or_child_child in or_child.exprs:
                        middle_parts.append(
                            Then(left_part + [or_child_child] + right_part).clean())
                    or_term = Or(middle_parts).clean()
                    untouched_left_part: list[RENode] = children[0:i-left_i]
                    untouched_right_part: list[RENode] = children[i +
                                                                  1+right_i:len(children)]
                    result.append(
                        Then([*untouched_left_part, or_term, *untouched_right_part]).clean())
        return result

    def rewrites_with_rewritten_children(self, children: list[RENode]) -> list[RENode]:
        results: list[RENode] = []
        results.append(Then(children).clean())
        results.extend(shuffle_pick_n(
            self._rewrite_distributive_then(children), REWRITE_INFLATION_LIMIT_DISTRIBUTIVE))
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

    def rewrites_with_rewritten_child(self, child: RENode) -> list[RENode]:
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

    def rewrites_with_rewritten_child(self, child: RENode) -> list[RENode]:
        results: list[RENode] = []
        results.append(Plus(child))
        results.append(Then([child, Repeat(child)]).clean())
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

    def rewrites_with_rewritten_child(self, child: RENode) -> list[RENode]:
        results: list[RENode] = []
        results.append(Complement(child).remove_double_negation())
        if isinstance(child, Complement):
            results.append(child.child)
        results.extend(shuffle_pick_n(self._demorgans(
            child), REWRITE_INFLATION_LIMIT_DEMORGAN))
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

    def rewrites(self) -> list[RENode]:
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

    def rewrites(self) -> list[RENode]:
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

    def rewrites(self) -> list[RENode]:
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

    def rewrites(self) -> list[RENode]:
        return [self]

    def __eq__(self, b) -> bool:
        return isinstance(b, Any) and self.negated == b.negated

    def content(self) -> str:
        return '.'
