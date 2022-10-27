from genericpath import exists
import sys  # noqa
sys.path.append('.')  # noqa
import os
import IPython
from pathlib import Path
from typing import Tuple
import numpy as np

from data_loader import load_file
import data_generator


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def save_to_file(path: str | Path, lines: list[str]):
    path = Path(path)
    lines_special = lines
    with open(path, 'w') as f:
        f.write(('\n'.join(lines_special) + '\n'))


def save_split():
    f1 = load_file('../training_data/f1.txt')
    train, val = f1.split(42, 0.2)
    save_to_file('../training_data_tmp/train.txt', train)
    save_to_file('../training_data_tmp/val.txt', val)


def save_all(path: Path):
    f1 = load_file('../datasets/text2task/f1.txt')
    save_to_file(path, f1.get_all_prompts())


def save_synthetic(path: Path):
    props = '../datasets/text2task/prop.txt'
    var_describe_map = '../datasets/text2task/var_describe_map.txt'
    patterns = '../datasets/text2task/patterns.txt'
    dist_parameters = {
        'exp_children': 1.2,  # defines exponential distr. for # of children
        'clip_children': 3,
        'exp_props': 0.5,  # defines exponential distr. for # of propvars in transitions
        'bin_negate': 0.05,  # probability to negate a propvar in transitions
    }
    prompts = data_generator.generate_synthetic(
        props, var_describe_map, patterns, dist_parameters, 5, 10000)

    lines = ['{' + f'"text": "{desc} => {expr}"' +
             '}\n' for desc, expr in prompts]

    with open(path, 'w') as f:
        f.writelines(lines)


def main():
    # TODO load from config/produce_datasets.json
    path = create_if_doesnt_exist(
        '../preprocessed_datasets/text2task', 'train', '.json')
    save_synthetic(path)


if __name__ == '__main__':
    main()
