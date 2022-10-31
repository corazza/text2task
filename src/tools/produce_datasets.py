import sys  # noqa
sys.path.append('.')  # noqa
from pathlib import Path
from typing import Tuple

import organic_data_augmentor
import data_generator


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def save_prompts(path: Path, prompts: list[Tuple[str, str]]):
    lines = [
        f'{desc} => {expr}' for desc, expr in prompts]
    lines = list(filter(lambda l: len(l) < 700, lines))
    lines = list(map(lambda l: '{"text": "' + l + '"}\n', lines))
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def save_both():
    # TODO load from config/produce_datasets.json
    path = create_if_doesnt_exist(
        '../preprocessed_datasets/text2task', 'train', '.json')
    organic_prompts = organic_data_augmentor.load_file(
        '../datasets/text2task/f1.txt', '../datasets/text2task/prop.txt', 3)
    synthetic_prompts = data_generator.get_default(10000)
    prompts = organic_prompts + synthetic_prompts
    save_prompts(path, prompts)


def main():
    save_both()


if __name__ == '__main__':
    main()
