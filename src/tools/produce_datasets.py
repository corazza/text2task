import sys  # noqa
sys.path.append('.')  # noqa
import itertools
import copy
from pathlib import Path
from typing import Tuple
import numpy as np
import IPython
import os

import organic_data_augmentor
import data_generator
import desc_rewriter
import expr_printer
import compiler_interface


MAX_PROMPT_LENGTH = 550


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def ensure_max_length(lines: list[str]):
    for line in lines:
        if len(line) >= MAX_PROMPT_LENGTH:
            print('line too long')
            IPython.embed()
        assert len(line) < MAX_PROMPT_LENGTH


def prompt_to_line(p: Tuple[str, str]) -> str:
    return '{"a": "' + p[0] + '", "b":"' + p[1] + '"}\n'


def save_prompts(path: Path, prompts: list[Tuple[str, str]]):
    lines = [prompt_to_line(p) for p in prompts]
    ensure_max_length(lines)
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def randomize_conjuncts(x: str) -> str:
    parsed = compiler_interface.parse(x)
    return expr_printer.expr_to_str(parsed, randomize=True, connect_then=False)


def load_augmented(path) -> list[Tuple[str, str]]:
    return organic_data_augmentor.load_file(path, 'datasets/text2task/prop.txt', inflation_factor=None)


def get_interactive_prompts() -> list[Tuple[str, str]]:
    r = []
    dir = 'datasets/text2task/interactive/'
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            prompts = load_augmented(f)
            r.append(prompts)
    return list(itertools.chain(*r))


def save_both(use_synthetic=False):
    # TODO load from config/produce_datasets.json
    path = create_if_doesnt_exist(
        'preprocessed_datasets/text2task', 'train', '.json')
    path_synthetic = create_if_doesnt_exist(
        'preprocessed_datasets/text2task', 'synthetic', '.json')
    rewrites = desc_rewriter.load_file('datasets/text2task/rewrites.txt')
    props = 'datasets/text2task/prop.txt'
    var_describe_map = 'datasets/text2task/var_describe_map.txt'
    patterns = 'datasets/text2task/patterns.txt'

    organic_prompts = load_augmented('datasets/text2task/organic.txt')
    organic_dist = data_generator.analyze_dist(organic_prompts)
    interactive_prompts = get_interactive_prompts()

    # HERE never place !var alone

    # PLUS gets double-sampled
    options = copy.deepcopy(organic_dist['node'].options)  # type: ignore
    options['THEN'] = int(options['THEN'] * 20)
    options['OR'] = int(options['OR'] * 10)
    corrected_dist = copy.deepcopy(organic_dist)
    corrected_vals = corrected_dist['complexity'].values
    corrected_vals = list(filter(lambda x: x < 15, corrected_vals))
    corrected_dist['complexity'] = data_generator.HistogramDist(corrected_vals)
    corrected_dist['node'] = data_generator.ChoiceDist(options)

    if use_synthetic:
        synthetic_prompts = data_generator.generate_synthetic(
            props, var_describe_map, patterns, corrected_dist, len(organic_prompts) + len(interactive_prompts))
        synthetic_dist = data_generator.analyze_dist(synthetic_prompts)

    # prompts = organic_prompts + synthetic_prompts + interactive_prompts
    prompts = organic_prompts + interactive_prompts
    prompts = [(desc_rewriter.apply_rewrites(p[0], rewrites),
                randomize_conjuncts(p[1])) for p in prompts]
    np.random.shuffle(prompts)  # type: ignore

    save_prompts(path, prompts)
    if use_synthetic:
        save_prompts(path_synthetic, synthetic_prompts)

    print('distribution of organic data:')
    print(organic_dist)
    if use_synthetic:
        print('distribution of synthetic data:')
        print(synthetic_dist)

    print(f'num. organic = {len(organic_prompts)}')
    print(f'num. interactive = {len(interactive_prompts)}')
    if use_synthetic:
        print(f'num. synthetic = {len(synthetic_prompts)}')


def main():
    save_both(use_synthetic=True)


if __name__ == '__main__':
    main()
