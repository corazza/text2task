import random
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


def produce_datasets(output_name: str, load_from: list[str], load_from_single_line: list[str], validate_raw: bool):
    model_args, data_args, training_args = get_args()
    tokenizer = get_tokenizer(model_args)

    path = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.json')
    path_human = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', output_name, '.txt')
    path_statistics = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'statistics', '.txt')
    path_synthetic = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'synthetic', '.txt')

    terms = load_terms(DEFAULT_TERMS_PATH)
    examples: list[Example] = []
    for load_path in load_from:
        examples.extend(load_examples(load_path))
    for load_path in load_from_single_line:
        examples.extend(load_examples_single_line(load_path))

    print(f'original examples: {len(examples)}')

    if validate_raw and not VALIDATE_AUGMENTED:
        validate_runs(examples)
    # HERE TODO don't pick whole examples to add to: add to all of them, then pick random samples from them in 0.2 proportion
    print('augmenting examples...')
    examples = ast_rewrites(examples)
    if VALIDATE_AUGMENTED:
        validate_runs(examples)

    print('converting to ab...')
    organic_ab = examples_to_ab(examples)
    organic_ab = apply_cap(organic_ab, SENTENCE_CAP)
    print('applying organic rewrites...')
    organic_rewrites = ab_rewrites(organic_ab, terms, to_cap=True)
    print(f'num_organic={len(organic_rewrites)}')
    statistics = ab_statistics(organic_rewrites, only_example=True)

    print('augmenting ab...')
    patterns = load_patterns('datasets/txt2task/augment_patterns.txt')
    synthetic_ab = augmented_ab(patterns, organic_rewrites)

    # print('paraphrasing synthetic...')
    # synthetic_original, synthetic_paraphrased = paraphrase_split(
    #     synthetic_ab, PARAP_P)
    # synthetic_ab = synthetic_original + synthetic_paraphrased

    print('applying synthetic rewrites...')
    synthetic_rewrites = ab_rewrites(synthetic_ab, terms, to_cap=False)

    print(f'num_synthetic={len(synthetic_rewrites)}')

    ab = organic_rewrites + synthetic_rewrites
    ab = ab_rewrites_num(ab)

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

    print('applying manual fixes...')
    ab = manual_fixes(ab)
    # ab = text_tool_correction(ab)
    # ab = correct_ab(ab)
    print('running sanity checks...')
    sanity_checks(ab)

    lines = ab_to_lines(ab)
    lines_human = ab_to_lines_human(ab)
    lines_synthetic = ab_to_lines_synthetic(synthetic_rewrites)

    lines_statistics = statistics_to_lines(statistics)
    save_lines(path, lines)
    save_lines(path_human, lines_human)
    save_lines(path_synthetic, lines_synthetic)
    save_lines(path_statistics, lines_statistics)


def main():
    set_all_seeds(SEED)
    load_from = [f'datasets/txt2task/use/{x}.txt' for x in SOURCES]
    load_from_single_line = [
        f'datasets/txt2task/use_synth/{x}.txt' for x in SOURCES_SINGLE_LINE]
    produce_datasets('train', load_from,
                     load_from_single_line, validate_raw=False)


if __name__ == '__main__':
    main()
