import sys  # noqa
sys.path.append('.')  # noqa
from pathlib import Path
from datetime import datetime
from typing import Tuple

import data_generator
from rm_ast import RMExpr
import rm_generator
import produce_datasets
import desc_rewriter
import expr_printer


def prompt_to_lines(p: Tuple[str, str]) -> str:
    return f'{p[1]}\n=\n{p[0]}\n'


def append_last(path: Path, prompt: Tuple[str, str]):
    with open(path, 'a') as f:
        f.write(prompt_to_lines(prompt) + '\n')


def collect_one(dist_parameters: dict[str, rm_generator.DistBase], props: dict[str, list[str]]) -> Tuple[str, str]:
    complexity = dist_parameters['complexity'].sample_int()
    rm = rm_generator.generate(dist_parameters, props, complexity)
    desc = input(expr_printer.expr_to_str(rm, connect_then=True) + '\n: ')
    if len(desc) == 0:
        return collect_one(dist_parameters, props)
    return (desc, expr_printer.expr_to_str(rm))


def main():
    name = datetime.today().strftime('%Y_%m_%d')
    path = produce_datasets.create_if_doesnt_exist(
        '../datasets/text2task/interactive', name, '.txt')
    props = rm_generator.load_props('../datasets/text2task/prop.txt')
    dist_parameters = data_generator.get_default_dist_params()
    rewrites = desc_rewriter.load_file('../datasets/text2task/rewrites.txt')
    while True:
        prompt = collect_one(dist_parameters, props)
        append_last(path, (desc_rewriter.apply_rewrites(
            prompt[0], rewrites), produce_datasets.randomize_conjuncts(prompt[1])))


if __name__ == '__main__':
    main()
