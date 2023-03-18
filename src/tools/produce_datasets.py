import IPython
import numpy as np

from consts import *
from datasets_common import ab_to_lines, ab_to_lines_human, add_term_rewrites, ast_rewrites, augment_examples, create_if_doesnt_exist, examples_to_ab, filter_length, load_examples, load_terms, remove_residuals, sanity_checks, save_lines, text_rewrites, validate_length, validate_runs
from training import get_args, get_tokenizer


def produce_datasets(output_name: str, load_from: list[str], validate_all: bool):
    model_args, data_args, training_args = get_args()
    tokenizer = get_tokenizer(model_args)

    path = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.json')
    path_human = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.txt')

    terms = load_terms('datasets/txt2task/terms.txt')
    examples = load_examples(load_from[0])
    for load_path in load_from[1:]:
        examples.extend(load_examples(load_path))

    if VALIDATE_RAW or validate_all:
        validate_runs(examples)
    examples = augment_examples(examples)
    if VALIDATE_AUGMENTED or validate_all:
        validate_runs(examples)
    examples = add_term_rewrites(examples, terms, TERMS_INFLATION_LIMIT)
    examples = text_rewrites(examples)
    if VALIDATE_TEXT_REWRITES or validate_all:
        validate_runs(examples)

    ab = examples_to_ab(examples)
    ab = remove_residuals(ab)
    sanity_checks(ab)
    len_before = len(ab)
    ab = filter_length(ab, tokenizer)
    len_after = len(ab)
    diff = len_before - len_after
    print(f'removed {diff} examples based on length')

    np.random.shuffle(ab)  # type: ignore
    lines = ab_to_lines(ab)
    lines_human = ab_to_lines_human(ab)
    save_lines(path, lines)
    save_lines(path_human, lines_human)


def main():
    produce_datasets('train', ['datasets/txt2task/organic.txt',
                     'datasets/txt2task/organic2.txt'], validate_all=False)


if __name__ == '__main__':
    main()
