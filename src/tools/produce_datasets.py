import random
from transformers import pipeline, set_seed
from typing import Tuple
import IPython
import numpy as np

from consts import *
from datasets_common import *
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


def produce_datasets(output_name: str, load_from: list[str], validate_all: bool):
    model_args, data_args, training_args = get_args()
    tokenizer = get_tokenizer(model_args)

    path = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.json')
    path_human = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.txt')
    path_statistics = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'statistics', '.txt')

    terms = load_terms('datasets/txt2task/terms.txt')
    examples = load_examples(load_from[0])
    for load_path in load_from[1:]:
        examples.extend(load_examples(load_path))

    if VALIDATE_RAW and not (VALIDATE_AUGMENTED or validate_all):
        validate_runs(examples)
    # HERE TODO don't pick whole examples to add to: add to all of them, then pick random samples from them in 0.2 proportion
    print('augmenting examples...')
    examples = augment_examples(examples)
    if VALIDATE_AUGMENTED or validate_all:
        validate_runs(examples)

    print('converting to ab...')
    organic_ab = examples_to_ab(examples)
    to_take_organic = organic_ab_to_take(organic_ab)

    print('augmenting ab...')
    synthetic_ab = augmented_ab(examples, len(organic_ab))
    to_take_synthetic = synthetic_ab_to_take(synthetic_ab)

    print('applying rewrites...')
    ab = ab_rewrites(organic_ab, terms, to_take_organic) + \
        ab_rewrites(synthetic_ab, terms, to_take_synthetic)

    # ab = apply_cap(ab, SENTENCE_CAP)

    np.random.shuffle(ab)  # type: ignore

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

    print('running sanity checks...')
    sanity_checks(ab)

    statistics, num_synthetic = ab_statistics(ab)
    num_organic = len([e for e, a, b in ab if not e.synthetic])
    lines = ab_to_lines(ab)
    lines_human = ab_to_lines_human(ab)
    lines_statistics = statistics_to_lines(
        statistics) + [f'organic: {num_organic}\n', f'synthetic: {num_synthetic}\n']
    save_lines(path, lines)
    save_lines(path_human, lines_human)
    save_lines(path_statistics, lines_statistics)


def main():
    set_seed(42)
    np.random.seed(42)
    random.seed(42)
    load_from = [f'datasets/txt2task/use/{x}.txt' for x in SOURCES]
    produce_datasets('train', load_from, validate_all=False)


if __name__ == '__main__':
    main()
