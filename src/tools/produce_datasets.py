import sys  # noqa
sys.path.append('.')  # noqa
import itertools
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


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def save_prompts(path: Path, prompts: list[Tuple[str, str]]):
    lines = [prompt_to_line(p) for p in prompts]
    for line in lines:
        assert len(line) < 700
    lines = list(filter(lambda l: len(l) < 700, lines))
    lines = list(map(line_to_json, lines))
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def save_prompts_human(path: Path, prompts: list[Tuple[str, str]]):
    lines = [f'{prompt_to_line(p)}\n' for p in prompts]
    for line in lines:
        assert len(line) < 700
    lines = list(filter(lambda l: len(l) < 700, lines))
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def prompt_to_line(p: Tuple[str, str]) -> str:
    return f'{p[0]} => {p[1]}'


def line_to_json(x: str) -> str:
    return '{"text": "' + x + '"}\n'


def randomize_conjuncts(x: str) -> str:
    parsed = compiler_interface.parse(x)
    return expr_printer.expr_to_str(parsed, randomize=True)


def load_augmented(path) -> list[Tuple[str, str]]:
    return organic_data_augmentor.load_file(path, '../datasets/text2task/prop.txt', 4)


def get_interactive_prompts() -> list[Tuple[str, str]]:
    r = []
    dir = '../datasets/text2task/interactive/'
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        if os.path.isfile(f):
            prompts = load_augmented(f)
            r.append(prompts)
    return list(itertools.chain(*r))


def save_both():
    # TODO load from config/produce_datasets.json
    path = create_if_doesnt_exist(
        '../preprocessed_datasets/text2task', 'train', '.json')
    path_human = create_if_doesnt_exist(
        '../preprocessed_datasets/text2task', 'train', '.txt')
    rewrites = desc_rewriter.load_file('../datasets/text2task/rewrites.txt')
    organic_prompts = load_augmented('../datasets/text2task/organic.txt')
    synthetic_prompts = data_generator.get_default(len(organic_prompts))
    interactive_prompts = get_interactive_prompts()
    prompts = organic_prompts + synthetic_prompts + interactive_prompts
    prompts = [(desc_rewriter.apply_rewrites(p[0], rewrites),
                randomize_conjuncts(p[1])) for p in prompts]
    np.random.shuffle(prompts)  # type: ignore
    save_prompts(path, prompts)
    save_prompts_human(path_human, prompts)


def main():
    save_both()


if __name__ == '__main__':
    main()
