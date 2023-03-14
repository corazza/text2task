import IPython
import numpy as np

from consts import *
from datasets_common import ab_to_lines, ab_to_lines_human, ast_rewrites, augment_examples, create_if_doesnt_exist, ensure_max_length, examples_to_ab, load_examples, save_lines, validate_length, validate_runs
from training import get_args, get_tokenizer


def main():
    model_args, data_args, training_args = get_args()
    tokenizer = get_tokenizer(model_args)

    path = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'train', '.json')
    path_human = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'train', '.txt')
    examples = load_examples(
        'datasets/txt2task/organic.txt')

    assert not VALIDATE_AUGMENTED or VALIDATE_EXAMPLES

    if not VALIDATE_AUGMENTED and VALIDATE_EXAMPLES:
        validate_runs(examples)
    examples = augment_examples(examples)
    if VALIDATE_AUGMENTED and VALIDATE_EXAMPLES:
        validate_runs(examples)

    ab = examples_to_ab(examples)
    validate_length(ab, tokenizer)
    np.random.shuffle(ab)  # type: ignore
    lines = ab_to_lines(ab)
    lines_human = ab_to_lines_human(ab)
    save_lines(path, lines)
    save_lines(path_human, lines_human)


if __name__ == '__main__':
    main()
