import numpy as np
from pathlib import Path
import more_itertools
import itertools
from typing import Sequence, Tuple

import rm_ast


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
    path = Path(path)
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        maps = parse_all_props(more_itertools.peekable(iter(lines)))
        map = dict()
        for name, props in maps:
            assert name not in map
            map[name] = props
        return map


class GenerateNode:
    def clean(self) -> 'GenerateNode':
        raise NotImplementedError()


class StemNode(GenerateNode):
    def clean(self) -> 'StemNode':
        raise NotImplementedError()


class VarNode(GenerateNode):
    def __init__(self, vars: list[str]):
        self.vars = vars

    def clean(self) -> 'VarNode':
        return self


class ThenNode(GenerateNode):
    def __init__(self, children: Sequence[GenerateNode]):
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


class OrNode(GenerateNode):
    def __init__(self, children: Sequence[GenerateNode]):
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


class RepeatNode(GenerateNode):
    def __init__(self, child: GenerateNode):
        self.child = child

    def clean(self) -> 'RepeatNode':
        self.child = self.child.clean()
        if isinstance(self.child, RepeatNode):
            return self.child
        else:
            return self


class PlusNode(GenerateNode):
    def __init__(self, child: GenerateNode):
        self.child = child

    def clean(self) -> 'PlusNode':  # TODO
        return self


def extract_props(props: dict[str, list[str]]) -> list[str]:
    r = set()
    for _cat, prop_list in props.items():
        r.update(prop_list)
    return list(r)


class NodeCreator:
    def __init__(self, dist_parameters: dict[str, float], props: dict[str, list[str]], complexity: int):
        self.complexity = complexity
        self.dist_parameters = dist_parameters
        self.props = extract_props(props)

    def stems_left(self) -> int:
        return self.complexity

    def one_stem(self) -> StemNode:
        assert self.complexity >= 1
        self.complexity -= 1
        return StemNode()

    def two_or_more(self) -> list[StemNode]:
        assert self.complexity >= 2
        num = int(
            2+np.random.exponential(self.dist_parameters['exp_children']))
        num = min(num, self.complexity)
        num = min(num, int(self.dist_parameters['clip_children']))
        self.complexity -= num
        return [StemNode() for i in range(num)]

    def generate_props(self) -> list[str]:
        num = int(1+np.random.exponential(self.dist_parameters['exp_props']))
        props = []
        for _i in range(num):
            if np.random.random() < self.dist_parameters['bin_negate']:
                negate = '!'
            else:
                negate = ''
            chosen = np.random.randint(0, len(self.props))
            prop = f'{negate}{self.props[chosen]}'
            props.append(prop)
        return props

    def speciate_stem(self, banned: list[str]) -> GenerateNode:
        types = ['THEN', 'OR', 'REPEAT', 'PLUS', 'VAR']
        for ban in banned:
            if ban in types:
                types.remove(ban)

        if self.complexity < 2:
            if 'THEN' in types:
                types.remove('THEN')
            if 'OR' in types:
                types.remove('OR')

        if self.complexity < 1:
            if 'REPEAT' in types:
                types.remove('REPEAT')
            if 'PLUS' in types:
                types.remove('PLUS')

        num_types = len(types)
        chosen = np.random.randint(num_types)

        if types[chosen] == 'THEN':
            children = self.two_or_more()
            return ThenNode(children)
        elif types[chosen] == 'OR':
            children = self.two_or_more()
            return OrNode(children)
        elif types[chosen] == 'REPEAT':
            child = self.one_stem()
            return RepeatNode(child)
        elif types[chosen] == 'PLUS':
            child = self.one_stem()
            return PlusNode(child)
        else:
            assert types[chosen] == 'VAR'
            return VarNode(self.generate_props())


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


def generate(dist_parameters: dict[str, float], props: dict[str, list[str]], complexity: int) -> rm_ast.RMExpr:
    node_creator = NodeCreator(dist_parameters, props, complexity)
    root = node_creator.speciate_stem([])
    to_visit = [root]
    while len(to_visit) > 0:
        assert len(to_visit) > 0
        visiting = to_visit.pop(0)
        if isinstance(visiting, RepeatNode) or isinstance(visiting, PlusNode):
            visiting.child = node_creator.speciate_stem(['REPEAT', 'PLUS'])
            to_visit.append(visiting.child)
        elif isinstance(visiting, VarNode):
            pass
        else:
            if isinstance(visiting, ThenNode):
                banned = 'THEN'
            else:
                assert isinstance(visiting, OrNode)
                banned = 'OR'
            visiting.children = [node_creator.speciate_stem([banned])
                                 for _child in visiting.children]
            to_visit.extend(visiting.children)
    return to_rm_expr(root.clean())
