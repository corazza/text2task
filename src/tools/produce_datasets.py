from typing import Tuple
import IPython
import numpy as np

from consts import *
from datasets_common import ab_statistics, ab_to_lines, ab_to_lines_human, add_term_rewrites, apply_cap, ast_rewrites, augment_examples, create_if_doesnt_exist, examples_statistics, examples_to_ab, filter_length, load_examples, load_terms, make_unique, remove_residuals, sanity_checks, save_lines, text_rewrites, validate_length, validate_runs
from training import get_args, get_tokenizer


def report_statistics(statistics_before_augment, statistics_after_augment):
    sorted_before = sorted(statistics_before_augment,
                           key=lambda x: x['num_product'])
    sorted_after = sorted(statistics_after_augment,
                          key=lambda x: x['num_product'])

    repr_before = [(x['representative_desc'], x['num_product'])
                   for x in sorted_before]
    repr_after = [(x['representative_desc'], x['num_product'])
                  for x in sorted_after]

    print('before augmentation:')
    for repr, x in repr_after:
        print(x, repr)


def report_ab_statistics(statistics: dict[str, int]):
    sorted_statistics = sorted(statistics.items(), key=lambda x: x[1])
    for k, v in sorted_statistics:
        print(f'{v}: {k}')


def produce_datasets(output_name: str, load_from: list[str], validate_all: bool, inflation_limit: int):
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

    statistics_before_augment = examples_statistics(examples)

    if VALIDATE_RAW or validate_all:
        validate_runs(examples)
    examples = augment_examples(examples)
    if VALIDATE_AUGMENTED or validate_all:
        validate_runs(examples)
    examples = add_term_rewrites(examples, terms, inflation_limit)
    examples = text_rewrites(examples)
    if VALIDATE_TEXT_REWRITES or validate_all:
        validate_runs(examples)

    statistics_after_augment = examples_statistics(examples)
    # report_statistics(statistics_before_augment, statistics_after_augment)

    ab = examples_to_ab(examples)
    ab = remove_residuals(ab)
    sanity_checks(ab)

    len_before = len(ab)
    ab = filter_length(ab, tokenizer)
    len_after = len(ab)
    diff = len_before - len_after
    print(f'removed {diff} examples based on length')

    len_before = len(ab)
    ab = make_unique(ab)
    len_after = len(ab)
    diff = len_before - len_after
    print(f'removed {diff} examples based on uniqueness')

    np.random.shuffle(ab)  # type: ignore
    ab = apply_cap(ab)

    statistics = ab_statistics(ab)
    report_ab_statistics(statistics)

    lines = ab_to_lines(ab)
    lines_human = ab_to_lines_human(ab)
    save_lines(path, lines)
    save_lines(path_human, lines_human)

    # print()
    # print('after augmentation:')
    # for repr, x in repr_after[:10]:
    #     print(x, repr)


def main():
    load_from = [f'datasets/txt2task/use/{x}.txt' for x in SOURCES]
    produce_datasets('train', load_from, validate_all=False,
                     inflation_limit=TERMS_INFLATION_LIMIT)


if __name__ == '__main__':
    main()
