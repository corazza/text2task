from pathlib import Path
from typing import Tuple
import numpy as np

import describe
import describe_patterns
import rm_generator
import expr_printer


def generate_synthetic(props_path: str | Path, var_path: str | Path, patterns_path: str | Path, dist_parameters: dict[str, rm_generator.DistBase], n: int) -> list[Tuple[str, str]]:
    var_describe_map = describe.load_var_describe_map(var_path)
    patterns = describe_patterns.load_patterns(patterns_path)

    exprs = rm_generator.generate_many(
        props_path, dist_parameters, n)

    prompts = []

    for i in range(n):
        desc = describe.describe(patterns, var_describe_map, exprs[i])
        prompts.append(
            (desc.lower(), expr_printer.expr_to_str(exprs[i])))

    return prompts


def get_default_dist_params() -> dict[str, rm_generator.DistBase]:
    return {
        # defines distr. for # of children of nodes
        'children': rm_generator.ExpBasedDist(2, 2, 4),
        # defines distr. for # of propvars in transitions
        'props': rm_generator.ExpBasedDist(1, 0.5, 4),
        'complexity': rm_generator.ExpBasedDist(2, 7, 15),
        # probability to negate a propvar in transitions
        'negate': rm_generator.BinaryDist(0.05),
    }


def get_default(n: int) -> list[Tuple[str, str]]:
    props = '../datasets/text2task/prop.txt'
    var_describe_map = '../datasets/text2task/var_describe_map.txt'
    patterns = '../datasets/text2task/patterns.txt'
    dist_parameters = get_default_dist_params()
    return generate_synthetic(
        props, var_describe_map, patterns, dist_parameters, n)
