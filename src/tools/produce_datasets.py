import sys  # noqa
sys.path.append('.')  # noqa

import IPython
from data_loader import load_file
from pathlib import Path


def save_to_file(path: str | Path, lines: list['str']):
    path = Path(path)
    print(len(lines))
    lines_special = map(lambda l: f'<|endoftext|>{l}<|endoftext|>', lines)
    lines_special = lines
    with open(path, 'w') as f:
        f.write(('\n'.join(lines_special) + '\n'))


def save_split():
    f1 = load_file('../training_data/f1.txt')
    train, val = f1.split(42, 0.2)
    save_to_file('../training_data_tmp/train.txt', train)
    save_to_file('../training_data_tmp/val.txt', val)


def save_all():
    f1 = load_file('../datasets/f1.txt')
    save_to_file('../preprocessed_datasets/train.txt', f1.get_all_prompts())


def main():
    save_all()


if __name__ == '__main__':
    main()
