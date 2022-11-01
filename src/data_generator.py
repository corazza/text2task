from pathlib import Path
from typing import Tuple
import numpy as np
import IPython

import describe
import describe_patterns
import rm_generator
import expr_printer
import compiler_interface
import rm_ast


def generate_synthetic(props_path: str | Path, var_path: str | Path, patterns_path: str | Path, dist_parameters: dict[str, rm_generator.DistBase], n: int) -> list[Tuple[str, str]]:
    var_describe_map = describe.load_var_describe_map(var_path)
    patterns = describe_patterns.load_patterns(patterns_path)

    exprs = rm_generator.generate_many(
        props_path, dist_parameters, n)

    prompts = []

    for i in range(n):
        desc = describe.describe(patterns, var_describe_map, exprs[i])
        prompts.append(
            (desc.lower(), expr_printer.expr_to_str(exprs[i], randomize=True, connect_then=False)))

    return prompts


def analyze_stats(expr: rm_ast.RMExpr) -> dict[str, int | list[int]]:
    max_level = describe.compute_max_level(expr)
    dist = {  # TODO write a class
        'nums_children': [],  # num. children per parent node
        'nums_props': [],  # num. props per leaf
        'num_negate': 0,  # total number of negated pvars
        'num_non_negate': 0,  # total number of non-negated pvars
        'max_level': max_level,  # max level per expr
        'num_repeats': 0,  # num. repeats per expr
        'complexity': 0
    }
    to_visit = [expr]
    while len(to_visit) != 0:
        visiting = to_visit.pop(0)
        dist['complexity'] += 1
        if isinstance(visiting, rm_ast.Vars):
            dist['complexity'] -= 1
            for symbol in visiting.symbols:
                if '!' in symbol:
                    dist['num_negate'] += 1
                else:
                    dist['num_non_negate'] += 1
            dist['nums_props'].append(len(visiting.symbols))
        elif isinstance(visiting, rm_ast.Or) or isinstance(visiting, rm_ast.Then):
            dist['nums_children'].append(len(visiting.exprs))
            to_visit.extend(visiting.exprs)
        elif isinstance(visiting, rm_ast.Repeat) or isinstance(visiting, rm_ast.Plus):
            dist['num_repeats'] += 1
            to_visit.append(visiting.child)
        else:
            raise ValueError('unknown RM Expr AST type')
    return dist


def analyze_dist(prompts: list[Tuple[str, str]]) -> dict[str, rm_generator.DistBase]:
    compiled = list(
        map(lambda x: compiler_interface.parse(x[1]), prompts))
    stats = list(map(analyze_stats, compiled))

    children = 0
    total_children = 0

    props = 0
    total_props = 0

    total_negate = 0
    total_non_negate = 0
    total_max_level = 0
    total_repeats = 0
    total_complexity = 0

    max_max_level = 0

    for stat in stats:
        children += sum(stat['nums_children'])  # type: ignore
        total_children += len(stat['nums_children'])  # type: ignore
        props += sum(stat['nums_props'])  # type: ignore
        total_props += len(stat['nums_props'])  # type: ignore
        total_negate += stat['num_negate']  # type: ignore
        total_non_negate += stat['num_non_negate']  # type: ignore
        total_max_level += stat['max_level']  # type: ignore
        total_repeats += stat['num_repeats']  # type: ignore
        total_complexity += stat['complexity']  # type: ignore

        if stat['max_level'] > max_max_level:  # type: ignore
            max_max_level = stat['max_level']

    assert isinstance(max_max_level, int)

    # TODO test invariance to generating from these stats
    num_samples = len(prompts)
    avg_children = children / total_children
    avg_props = props / total_props
    p_negate = total_negate / (total_negate + total_non_negate)
    avg_max_level = total_max_level / num_samples
    avg_repeats = total_repeats / num_samples
    avg_complexity = total_complexity / num_samples

    return {
        # defines distr. for # of children of nodes
        'children': rm_generator.ShiftedLimitedExpDist(2, avg_children, 4),
        # defines distr. for # of propvars in transitions
        'props': rm_generator.ShiftedLimitedExpDist(1, avg_props, 4),
        # probability to negate a propvar in transitions
        'negate': rm_generator.BinaryDist(p_negate),
        'max_level': rm_generator.ShiftedLimitedExpDist(0, avg_max_level, max_max_level),
        'repeats': rm_generator.ShiftedExpDist(0, avg_repeats),
        'complexity': rm_generator.ShiftedExpDist(0, 3*avg_complexity),
    }
