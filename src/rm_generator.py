import copy
import numpy as np
from pathlib import Path
import more_itertools
import itertools
import IPython
from typing import Sequence, Tuple

import rm_ast
import util
import distributions


def parse_all_props(lines: more_itertools.peekable) -> list[Tuple[str, list[str]]]:
    maps = [parse_prop_group(lines)]
    while lines and lines.peek() == '':
        next(lines)
    if lines:
        others = parse_all_props(lines)
        maps.extend(others)
    return maps


def parse_prop_group(lines: more_itertools.peekable) -> Tuple[str, list[str]]:
    name = next(lines)
    props = parse_props(lines)
    return name, props


def parse_props(lines: more_itertools.peekable) -> list[str]:
    props = [next(lines).split()]
    while lines and not lines.peek() == '':
        props.append(next(lines).split())
    if lines and lines.peek() == '':
        next(lines)
    return list(itertools.chain.from_iterable(props))


def load_props(path: Path | str) -> dict[str, list[str]]:
    maps = parse_all_props(more_itertools.peekable(util.line_iter(path)))
    map = dict()
    for name, props in maps:
        assert name not in map
        map[name] = props
    return map


class GenerateNode:
    def __init__(self, level: int):
        self.level = level

    def clean(self) -> 'GenerateNode':
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        raise NotImplementedError()


class StemNode(GenerateNode):
    def __init__(self, level: int):
        super().__init__(level)

    def clean(self) -> 'StemNode':
        raise NotImplementedError()

    def appears(self) -> frozenset[str]:
        raise NotImplementedError()


class VarNode(GenerateNode):
    def __init__(self, level: int, vars: list[str]):
        super().__init__(level)
        self.vars = vars

    def clean(self) -> 'VarNode':
        return self

    def appears(self) -> frozenset[str]:
        return frozenset(self.vars)


class ThenNode(GenerateNode):
    def __init__(self, level: int, children: Sequence[GenerateNode]):
        super().__init__(level)
        self.children = list(children)

    def clean(self) -> 'ThenNode':
        self.children = list(map(lambda c: c.clean(), self.children))
        new_children = []
        for child in self.children:
            if isinstance(child, ThenNode):
                new_children.extend(child.children)
            else:
                new_children.append(child)
        self.children = new_children
        return self

    def appears(self) -> frozenset[str]:
        r = set()
        for child in self.children:
            r.update(child.appears())
        return frozenset(r)


class OrNode(GenerateNode):
    def __init__(self, level: int, children: Sequence[GenerateNode]):
        super().__init__(level)
        self.children = list(children)

    def clean(self) -> 'OrNode':
        self.children = list(map(lambda c: c.clean(), self.children))
        new_children = []
        for child in self.children:
            if isinstance(child, OrNode):
                new_children.extend(child.children)
            else:
                new_children.append(child)
        self.children = new_children
        return self

    def appears(self) -> frozenset[str]:
        r = set()
        for child in self.children:
            r.update(child.appears())
        return frozenset(r)


class RepeatNode(GenerateNode):
    def __init__(self, level: int, child: GenerateNode):
        super().__init__(level)
        self.child = child

    def clean(self) -> 'RepeatNode':
        self.child = self.child.clean()
        if isinstance(self.child, RepeatNode):
            return self.child
        else:
            return self

    def appears(self) -> frozenset[str]:
        return self.child.appears()


class PlusNode(GenerateNode):
    def __init__(self, level: int, child: GenerateNode):
        super().__init__(level)
        self.child = child

    def clean(self) -> 'PlusNode':  # TODO
        return self

    def appears(self) -> frozenset[str]:
        return self.child.appears()


def extract_props(props: dict[str, list[str]]) -> list[str]:
    r = set()
    for _cat, prop_list in props.items():
        r.update(prop_list)
    return list(r)


class NodeCreator:
    def __init__(self, max_level: int, dist_parameters: dict[str, distributions.DistBase], props: dict[str, list[str]]):
        self.complexity = dist_parameters['complexity'].sample_int()
        self.original_complexity = self.complexity
        self.max_level = max_level
        self.repeats = dist_parameters['repeats'].sample_int()
        self.dist_parameters = dist_parameters
        self.props = extract_props(props)

    def stems_left(self) -> int:
        return self.complexity

    def one_stem(self, level: int) -> StemNode:
        assert self.complexity >= 1
        self.complexity -= 1
        return StemNode(level)

    def two_or_more(self, level: int) -> list[StemNode]:
        assert self.complexity >= 2
        num = self.dist_parameters['children'].sample_int()
        num = min(num, self.complexity)
        self.complexity -= num
        return [StemNode(level) for i in range(num)]

    def generate_props(self) -> list[str]:
        num = self.dist_parameters['props'].sample_int()
        props = []
        for _i in range(num):
            negate = ''  # the semantics of negation map to avoidance, we do not want to negate variables on an individual basis
            chosen = np.random.randint(0, len(self.props))
            prop = f'{negate}{self.props[chosen]}'
            props.append(prop)
        return props

    def speciate_stem(self, level: int, banned: set[str]) -> GenerateNode:
        # banned = set()
        if self.complexity < 2:
            banned.add('THEN')
            banned.add('OR')

        if self.complexity < 1:
            banned.add('REPEAT')
            banned.add('PLUS')

        if self.complexity > 1:
            banned.add('VAR')

        type = self.dist_parameters['node'].sample_string_banned(banned, 'VAR')

        if level > self.max_level:
            type = 'VAR'

        if type == 'THEN':
            children = self.two_or_more(level+1)
            return ThenNode(level, children)
        elif type == 'OR':
            children = self.two_or_more(level+1)
            return OrNode(level, children)
        elif type == 'REPEAT':
            child = self.one_stem(level+1)
            self.repeats -= 1
            return RepeatNode(level, child)
        elif type == 'PLUS':
            child = self.one_stem(level+1)
            self.repeats -= 1
            return PlusNode(level, child)
        else:
            assert type == 'VAR'
            return VarNode(level, self.generate_props())


def to_rm_expr(node: GenerateNode) -> rm_ast.RMExpr:
    if isinstance(node, VarNode):
        return rm_ast.Vars(node.vars)
    elif isinstance(node, OrNode):
        children = map(to_rm_expr, node.children)
        return rm_ast.Or(list(children))
    elif isinstance(node, ThenNode):
        children = map(to_rm_expr, node.children)
        return rm_ast.Then(list(children))
    elif isinstance(node, PlusNode):
        child = to_rm_expr(node.child)
        return rm_ast.Plus(child)
    else:
        assert isinstance(node, RepeatNode)
        child = to_rm_expr(node.child)
        return rm_ast.Repeat(child)


def negate_all(node: GenerateNode, which_prop: str):
    if isinstance(node, VarNode):
        node.vars.append(f'!{which_prop}')
    elif isinstance(node, ThenNode) or isinstance(node, OrNode):
        for child in node.children:
            negate_all(child, which_prop)
    elif isinstance(node, RepeatNode) or isinstance(node, PlusNode):
        negate_all(node.child, which_prop)
    else:
        raise ValueError(f'unknown node type {type(node)} --- {node}')


def generate(dist_parameters: dict[str, distributions.DistBase], props: dict[str, list[str]]) -> rm_ast.RMExpr:
    max_level = 4
    node_creator = NodeCreator(max_level, dist_parameters, props)
    root = node_creator.speciate_stem(0, set())
    to_visit = [root]
    while len(to_visit) > 0:
        assert len(to_visit) > 0
        visiting = to_visit.pop(0)
        if isinstance(visiting, RepeatNode) or isinstance(visiting, PlusNode):
            visiting.child = node_creator.speciate_stem(
                visiting.child.level, banned=set(['REPEAT', 'PLUS']))
            to_visit.append(visiting.child)
        elif isinstance(visiting, VarNode):
            pass
        else:
            if isinstance(visiting, ThenNode):
                banned = 'THEN'
            else:
                assert isinstance(visiting, OrNode)
                banned = 'OR'
            visiting.children = [node_creator.speciate_stem(child.level, banned=set([banned]))
                                 for child in visiting.children]
            to_visit.extend(visiting.children)

    if dist_parameters['negate'].sample_bool():
        which_prop = 'MINE'
        # TODO extract these types of ops into Props class
        appear = root.appears()
        all_props = set()
        for _k, v in props.items():
            all_props.update(v)
        free = list(all_props - appear)
        chosen = np.random.randint(len(free))
        which_prop = free[chosen]
        negate_all(root, which_prop)

    return to_rm_expr(root.clean())


def generate_many(props_path: str | Path, dist_parameters: dict[str, distributions.DistBase], n: int) -> list[rm_ast.RMExpr]:
    props = load_props(props_path)
    exprs = []
    for _i in range(n):
        expr = generate(dist_parameters, props)
        exprs.append(expr)
    return exprs
