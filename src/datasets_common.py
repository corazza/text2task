import math
import cProfile
import itertools
import numpy as np
import copy
from curses.ascii import isalpha
from pathlib import Path
from typing import Iterator, Optional, Tuple
import more_itertools
import IPython
import pstats
from functools import wraps


from consts import *
from regex_printer import expr_to_str
from regex_validation import equivalent
from example_parser import Example, parse_examples
from parser_util import line_iter
import compiler_interface
from compiler_interface import compile
from terms_parser import parse_terms


def profile(sort_by='cumulative'):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            result = profiler.runcall(func, *args, **kwargs)
            stats = pstats.Stats(profiler)
            stats.sort_stats(sort_by)
            stats.print_stats()
            return result
        return wrapper
    return inner


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def load_examples(path: str) -> list[Example]:
    lines = more_itertools.peekable(line_iter(path))
    return parse_examples(lines)


def load_terms(path: str) -> dict[str, list[str]]:
    lines = more_itertools.peekable(line_iter(path))
    return parse_terms(lines)


def augment_examples(examples: list[Example]) -> list[Example]:
    examples = ast_rewrites(examples)
    return examples


def augmented_ab(examples: list[Example], num_abs) -> list[Tuple[Example, str, str]]:
    new_abs: list[Tuple[Example, str, str]] = []
    np.random.shuffle(examples)  # type: ignore

    eligible_pairs: list[Tuple[int, int]] = []
    for i, example1 in enumerate(examples):
        for j, example2 in enumerate(examples):
            if example1 == example2 or i == j:
                continue
            if example1.average_desc_length() > DESC_LENGTH_LIMIT or example2.average_desc_length() > DESC_LENGTH_LIMIT:
                continue
            eligible_pairs.append((i, j))

    num_concat: int = round(ADD_CONCAT_P * ADD_P * num_abs)
    print(f'adding concat... ({num_concat})')
    np.random.shuffle(eligible_pairs)
    for counter, (i, j) in enumerate(eligible_pairs[:num_concat]):
        example1 = examples[i]
        example2 = examples[j]
        example = with_concat(example1, example2)  # type: ignore
        new_abs.append(get_single_ab(example))

    num_disjunct: int = round(ADD_DISJUNCT_P * ADD_P * num_abs)
    print(f'adding disjunct... ({num_disjunct})')
    np.random.shuffle(eligible_pairs)
    for i, j in eligible_pairs[:num_disjunct]:
        example1 = examples[i]
        example2 = examples[j]
        example = with_disjunct(example1, example2)
        new_abs.append(get_single_ab(example))

    num_conjunct: int = round(ADD_CONJUNCT_P * ADD_P * num_abs)
    print(f'adding conjunct... ({num_conjunct})')
    np.random.shuffle(eligible_pairs)
    for i, j in eligible_pairs[:num_conjunct]:
        example1 = examples[i]
        example2 = examples[j]
        example = with_conjunct(example1, example2)
        new_abs.append(get_single_ab(example))

    num_avoidance: int = round(ADD_AVOIDANCE_P * ADD_P * num_abs)
    print(f'adding avoidance... ({num_avoidance})')
    eligible_classes = ['NO_THE', 'OBJECT_A',
                        'AVOID', 'STEP_ON_A', 'FALL_IN_A', 'HIT_A']

    for i in range(num_avoidance):
        example = random_from(examples)
        pvar_class = random_from(eligible_classes)
        new_example = with_avoidance(pvar_class, example)
        new_abs.append(get_single_ab(new_example))

    if VALIDATE_AB_AUGMENTS:
        validate_runs([e for e, a, b in new_abs])

    return new_abs


def get_single_ab(example: Example):
    np.random.shuffle(example.descs)
    np.random.shuffle(example.srcs)
    return example, example.descs[0], example.srcs[0]


def apply_text_rewrite_with_concat(x: str, rewrite: list[Tuple[str, str]], sep: str) -> str:
    new_string: str = x
    for left, right in rewrite:
        if ' ' in right:
            right = sep.join(right.split(' '))
            right = right.replace('the_', '')
        new_string = new_string.replace(left, right)
    return new_string


def get_all_terms_from_tag(terms: dict[str, list[str]], term_tag: str) -> list[str]:
    applicable: list[str] = []
    for term, tags in terms.items():
        if term_tag in tags:
            applicable.append(term)
    assert len(applicable) > 0
    return applicable


def organic_ab_to_take(abs: list[Tuple[Example, str, str]]) -> dict[str, int]:
    statistics, num_synthetic = ab_statistics(abs)
    assert num_synthetic == 0
    to_take: dict[str, int] = dict()
    for example, desc, src in abs:
        r = example.representative()
        assert not example.synthetic
        # to_take[r] = max(1, round(SENTENCE_CAP / statistics[r]))
        to_take[r] = 2
    return to_take


def synthetic_ab_to_take(abs: list[Tuple[Example, str, str]]) -> dict[str, int]:
    statistics, num_synthetic = ab_statistics(abs)
    assert num_synthetic == len(abs)
    to_take: dict[str, int] = dict()
    for example, desc, src in abs:
        r = example.representative()
        assert example.synthetic
        to_take[r] = 1
    return to_take


def ab_rewrites(abs: list[Tuple[Example, str, str]], terms, to_take: dict[str, int]) -> list[Tuple[Example, str, str]]:
    np.random.shuffle(abs)  # type: ignore
    new_abs: list[Tuple[Example, str, str]] = []
    for i, (example, desc, src) in enumerate(abs):
        if i % 250 == 0:
            print(f'{i}/{len(abs)}')
        r = example.representative()
        new_rewrites = get_new_rewrites(example, terms, to_take[r])
        for rewrite in new_rewrites:
            r_desc = apply_text_rewrite_with_concat(desc, rewrite, ' ')
            r_src = apply_text_rewrite_with_concat(src, rewrite, '_')
            new_abs.append((example, r_desc, r_src))
    return new_abs


def get_new_rewrites(example: Example, terms, num_new: int) -> list[list[Tuple[str, str]]]:
    new_rewrites: list[list[Tuple[str, str]]] = []
    for rewrite in example.example_rewrites:
        new_versions: list[list[Tuple[str, str]]] = []
        all_replacements: dict[str, list[str]] = dict()
        for left, right in rewrite:
            if right.isupper():
                all_replacements[left] = get_all_terms_from_tag(
                    terms, right)
            else:
                all_replacements[left] = [right]
        value_combinations = itertools.product(*all_replacements.values())
        for combination in value_combinations:
            if len(combination) != len(set(combination)):
                continue
            new_dict = {k: v for k, v in zip(
                all_replacements.keys(), combination)}
            new_versions.append(list(new_dict.items()))
        if len(new_versions) > 0:
            new_rewrites.extend(new_versions)
        else:
            new_rewrites.append(rewrite)
    np.random.shuffle(new_rewrites)
    seen = set()
    for rewrite in new_rewrites:
        seen.add(tuple(rewrite))
    assert len(seen) == len(new_rewrites)
    num_new_rewrites: int = len(new_rewrites)
    to_take: int = min(num_new_rewrites, num_new)
    return new_rewrites[:to_take]


def clean_desc(desc: str) -> str:
    desc = desc.replace('..', '.')
    desc = desc.replace('. .', '.')
    desc = desc.replace('.,', ',')
    desc = desc.replace('. ,', ',')
    return desc


def with_avoidance(pvar_class: str, example: Example) -> Example:
    variable = '$Z'
    desc_before: dict[str, list[str]] = {
        'NO_THE': [],
        'OBJECT_A': [],
        'AVOID': [],
        'STEP_ON_A': [],
        'FALL_IN_A': [],
        'HIT_A': [],
    }
    desc_after: dict[str, list[str]] = {
        'NO_THE': [],
        'OBJECT_A': [],
        'AVOID': [],
        'STEP_ON_A': [],
        'FALL_IN_A': [],
        'HIT_A': [],
    }

    desc_before['NO_THE'].append(f'Avoid {variable}. ')
    desc_before['NO_THE'].append(
        f'Do the following task, but avoid {variable}: ')
    desc_after['NO_THE'].append(f'. Avoid {variable}.')
    desc_after['NO_THE'].append(f', and avoid {variable}.')
    desc_after['NO_THE'].append(f'. Don\'t get near {variable}.')

    desc_before['OBJECT_A'].append(f'Avoid {variable}s. ')
    desc_before['OBJECT_A'].append(
        f'Do the following task, but avoid {variable}s: ')
    desc_after['OBJECT_A'].append(f'. Avoid {variable}s.')
    desc_after['OBJECT_A'].append(f', and avoid {variable}s')
    desc_after['OBJECT_A'].append(f'. Don\'t get near {variable}s')

    desc_before['AVOID'].append(f'Avoid {variable}s. ')
    desc_before['AVOID'].append(
        f'Do the following task, but avoid {variable}s: ')
    desc_after['AVOID'].append(f'. Avoid {variable}s')
    desc_after['AVOID'].append(f', and avoid {variable}s.')
    desc_after['AVOID'].append(f'. Don\'t get near {variable}s')

    desc_before['STEP_ON_A'].append(f'Don\'t step on {variable}s. ')
    desc_before['STEP_ON_A'].append(
        f'Do the following task, but don\'t step on {variable}s: ')
    desc_after['STEP_ON_A'].append(f', and never step on {variable}s')
    desc_after['STEP_ON_A'].append(f'. Don\'t step on {variable}s.')

    desc_before['FALL_IN_A'].append(f'Don\'t fall in a {variable}. ')
    desc_before['FALL_IN_A'].append(
        f'Do the following task, but don\'t fall in a {variable}: ')
    desc_after['FALL_IN_A'].append(f', and never fall in a {variable}.')
    desc_after['FALL_IN_A'].append(f'. Don\'t fall in a {variable}s')

    desc_before['HIT_A'].append(f'Don\'t hit a {variable}. ')
    desc_before['HIT_A'].append(
        f'Do the following task, but don\'t hit a {variable}: ')
    desc_after['HIT_A'].append(f', and never hit a {variable}')
    desc_after['HIT_A'].append(f'. Don\'t hit a {variable}s.')

    src_after: list[str] = [
        f'(!{variable})*',  # (!$B)*
        f'((.)* > {variable} > (.)*)~',  # ((.)* > $B > (.)*)~
    ]

    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = [rewrite_list + [(variable, pvar_class)] for rewrite_list in example.example_rewrites]
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example.representative()}_ADDED_AVOID'

    for (reward, pvars) in example.runs:
        new_runs.append((reward, pvars))
        new_runs.append((0, [frozenset({variable})] + pvars))

    for desc in example.descs:
        if np.random.random() < AUGMENT_PREFER_BEFORE:
            before: str = random_from(desc_before[pvar_class])
            new_desc = f'{before}{desc}'
        else:
            after: str = random_from(desc_after[pvar_class])
            new_desc = f'{desc}{after}'
        new_desc = clean_desc(new_desc)
        new_descs.append(new_desc)

    for src in example.srcs:
        for after in src_after:
            new_src = f'({src})&{after}'
            new_srcs.append(new_src)

    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def map_rewrite(letter: str, example_rewrites: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
    return [(f'{variable}{letter}' if '$' in variable else variable, to_class) for (variable, to_class) in example_rewrites]


def map_trace(letter: str, trace: list[frozenset[str]]) -> list[frozenset[str]]:
    return [frozenset([f'{v}{letter}' if '$' in v else v for v in vars]) for vars in trace]


def map_desc(rewrite_list: list[Tuple[str, str]], letter: str, desc: str) -> str:
    vars = list(reversed(sorted([x for (x, y) in rewrite_list])))
    for v in vars:
        desc = desc.replace(v, f'{v}{letter}' if '$' in v else v)
    return desc


def merge_rewrites(example1: Example, example2: Example) -> list[list[Tuple[str, str]]]:
    new_example_rewrites: list[list[Tuple[str, str]]] = [map_rewrite('Q', rewrite_list1) + map_rewrite(
        'P', rewrite_list2) for rewrite_list1 in example1.example_rewrites for rewrite_list2 in example2.example_rewrites]
    return new_example_rewrites


def with_concat(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = merge_rewrites(example1, example2)
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.representative()}_CONCAT_{example2.representative()}'

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            if len(example1.example_rewrites) > 0:
                src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            else:
                src1 = src1_or
            if len(example2.example_rewrites) > 0:
                src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            new_src = f'({src1}) > ({src2})'
            new_srcs.append(new_src)

    if VALIDATE_AB_AUGMENTS:
        concat_rm = compiler_interface.compile(new_srcs[0])
        example1_rm = compiler_interface.compile(example1.srcs[0])
        example2_rm = compiler_interface.compile(example2.srcs[0])
        for (_reward1, _trace1) in example1.runs:
            reward1 = example1_rm(*_trace1)
            first_positive = next(
                (i for i, x in enumerate(reward1) if x > 0), None)
            for (_reward2, trace2) in example2.runs:
                reward2 = example2_rm(*trace2)
                if first_positive == None:
                    trace1 = _trace1
                    reward = 0
                else:
                    trace1 = _trace1[:first_positive+1]
                    reward = sum(reward2)
                trace = map_trace('Q', trace1) + map_trace('P', trace2)
                reward = sum(concat_rm(*trace))  # TODO this is wrong
                new_runs.append((reward, trace))

    for desc1_or in example1.descs:
        for desc2 in example2.descs:
            desc1 = desc1_or
            if len(example1.example_rewrites) > 0:
                desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            if len(example2.example_rewrites) > 0:
                desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
            if '$' not in desc1[:2]:
                desc1 = desc1[:2].lower() + desc1[2:]
            if '$' not in desc2[:2]:
                desc2 = desc2[:2].lower() + desc2[2:]
            if 'first' not in desc1.lower() and 'second' not in desc2.lower():
                new_descs.append(f'First: {desc1}. Second: {desc2}')
                new_descs.append(
                    f'There are two tasks. First {desc1}. Second {desc2}.')
            if 'first' not in desc1.lower():
                new_descs.append(f'First: {desc1}. Then: {desc2}')
                new_descs.append(
                    f'There are two tasks. First {desc1}. Second {desc2}.')
            new_descs.append(f'{desc1}. Then {desc2}')
    new_descs = [clean_desc(desc) for desc in new_descs]
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def with_disjunct(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = merge_rewrites(example1, example2)
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.representative()}_DISJUNCT_{example2.representative()}'

    if VALIDATE_AB_AUGMENTS:
        example1_rm = compiler_interface.compile(example1.srcs[0])
        example2_rm = compiler_interface.compile(example2.srcs[0])

        for (reward, trace) in example1.runs:
            reward1 = example1_rm(*trace)
            reward2 = example2_rm(*trace)
            new_runs.append(
                (sum(reward1) + sum(reward2), map_trace('Q', trace)))

        for (reward, trace) in example2.runs:
            reward1 = example1_rm(*trace)
            reward2 = example2_rm(*trace)
            new_runs.append(
                (sum(reward1) + sum(reward2), map_trace('P', trace)))

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            if len(example1.example_rewrites) > 0:
                src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            else:
                src1 = src1_or
            if len(example2.example_rewrites) > 0:
                src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            new_src = f'({src1}) | ({src2})'
            new_srcs.append(new_src)

    for desc1_or in example1.descs:
        for desc2 in example2.descs:
            desc1 = desc1_or
            if len(example1.example_rewrites) > 0:
                desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            if len(example2.example_rewrites) > 0:
                desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
            if '$' not in desc1[:2]:
                desc1 = desc1[:2].lower() + desc1[2:]
            if '$' not in desc2[:2]:
                desc2 = desc2[:2].lower() + desc2[2:]
            new_descs.append(
                f'There are two tasks. Either {desc1}, or {desc2}.')
            new_descs.append(
                f'Either {desc1}. Or, {desc2}.')
    new_descs = [clean_desc(desc) for desc in new_descs]
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def merge_traces(trace1: list[frozenset[str]], trace2: list[frozenset[str]]) -> list[frozenset[str]]:
    trace: list[frozenset[str]] = []
    for vars1, vars2 in itertools.zip_longest(trace1, trace2, fillvalue=frozenset()):
        trace.append(vars1.union(vars2))
    return trace


def merge_rewards(rewards1: list[int], rewards2: list[int]):
    rewards: list[int] = []
    for r1, r2 in itertools.zip_longest(rewards1, rewards2, fillvalue=0):
        rewards.append(r1*r2)
    return rewards


def with_conjunct(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]
                               ] = merge_rewrites(example1, example2)
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.representative()}_CONJUNCT_{example2.representative()}'

    if VALIDATE_AB_AUGMENTS:
        example1_rm = compiler_interface.compile(example1.srcs[0])
        example2_rm = compiler_interface.compile(example2.srcs[0])

        for (_reward1, trace1) in example1.runs:
            reward1 = example1_rm(*trace1)
            for (_reward2, trace2) in example2.runs:
                reward2 = example2_rm(*trace2)
                rewards = merge_rewards(reward1, reward2)
                reward = sum(rewards)
                trace_merged = merge_traces(trace1, trace2)
                new_runs.append((reward, trace_merged))

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            if len(example1.example_rewrites) > 0:
                src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            else:
                src1 = src1_or
            if len(example2.example_rewrites) > 0:
                src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            new_src = f'(({src1}) > (.)*) & (({src2}) > (.)*)'
            new_srcs.append(new_src)

    for desc1_or in example1.descs:
        for desc2 in example2.descs:
            desc1 = desc1_or
            if len(example1.example_rewrites) > 0:
                desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            if len(example2.example_rewrites) > 0:
                desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
            if '$' not in desc1[:2]:
                desc1 = desc1[:2].lower() + desc1[2:]
            if '$' not in desc2[:2]:
                desc2 = desc2[:2].lower() + desc2[2:]
            new_descs.append(
                f'There are two tasks and you must do both at the same time. {desc1} (first task), and {desc2} (second task).')
            new_descs.append(
                f'{desc1}. And at the same time, {desc2}.')
            new_descs.append(
                f'{desc1}. And, {desc2}.')
    new_descs = [clean_desc(desc) for desc in new_descs]
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def random_from(xs: list):
    num = len(xs)
    i = np.random.randint(num)
    return xs[i]


def ast_rewrites(examples: list[Example]) -> list[Example]:
    for example in examples:
        new_srcs: list[str] = []
        for src in example.srcs:
            ast = compiler_interface.parse(src)
            rewrites_nodes = ast.rewrites()
            rewrites: list[str] = [expr_to_str(r) for r in rewrites_nodes]
            new_srcs.extend(rewrites)
        example.srcs.extend(new_srcs)
    return examples


def desc_src_to_line(a: str, b: str) -> str:
    return '{"a": "' + a + '", "b":"' + b + '"}\n'


def desc_src_to_line_human(a: str, b: str) -> str:
    return a + ' => ' + b + '\n'


def validate_runs(examples: list[Example]):
    for (i, example) in enumerate(examples):
        print(f'{i}/{len(examples)}')
        if len(example.runs) == 0:
            print(f'no runs to validate: {example.srcs}')
            continue
        rewards_sets: dict[Tuple[frozenset[str]], set[Tuple[int]]] = dict()
        for (i_src, src) in enumerate(example.srcs):
            print(f'    src={i_src}/{len(example.srcs)}')
            rm = compiler_interface.compile(src)
            for (i_rr, (reward, run)) in enumerate(example.runs):
                rewards = rm.multiple_transitions(0, run)
                if tuple(run) not in rewards_sets:
                    rewards_sets[tuple(run)] = set()
                rewards_sets[tuple(run)].add(tuple(rewards))
                assert reward == sum(
                    rewards), f'{reward} @ {run} failed for {src} -> {rewards}'
        for run, rewards_set in rewards_sets.items():
            assert len(rewards_set) == 1, f'failed on run {run}'


def validate_equiv(examples: list[Example]):
    # if validate:
    #     dfa, _node_creator = compiler_interface.get_dfa(src)
    #     for rewrite in rewrites:
    #         dfa_b, _ = compiler_interface.get_dfa(rewrite)
    #         ineq_evidence = equivalent(
    #             appears, dfa, dfa_b, test_length, num_tests)
    #         if len(ineq_evidence) != 0:
    #             print(src)
    #             print(rewrite)
    #             IPython.embed()  # type: ignore
    #         assert len(ineq_evidence) == 0
    raise NotImplementedError()


def example_length(ab: Tuple[Example, str, str], tokenizer) -> int:
    encoded = tokenizer('<|bos|>' + ab[1] + '<|sep|>' + ab[2] + '<|eos|>')
    return len(encoded.input_ids)


def validate_length(abs: list[Tuple[Example, str, str]], tokenizer):
    for ab in abs:
        ab_len = example_length(ab, tokenizer)
        assert ab_len <= PAD_SIZE, f'{ab[0]} => {ab[1]} over {PAD_SIZE}: {ab_len}'


def filter_length(abs: list[Tuple[Example, str, str]], tokenizer):
    result = list(
        filter(lambda x: example_length(x, tokenizer) <= PAD_SIZE, abs))
    return result


def sanity_check(ab: Tuple[Example, str, str]):
    e, desc, src = ab
    for i, c in enumerate(src):
        if i == 0 or i == len(src) - 1:
            continue
        if c == ' ':
            assert not isalpha(src[i-1]) or not isalpha(src[i+1])


def sanity_checks(abs: list[Tuple[Example, str, str]]):
    for ab in abs:
        sanity_check(ab)


def remove_residuals(abs: list[Tuple[Example, str, str]]) -> list[Tuple[Example, str, str]]:
    result: list[Tuple[Example, str, str]] = []
    for e, a, b in abs:
        if '$' in a or '$' in b:
            continue
        result.append((e, a, b))
    return result


def make_unique(abs: list[Tuple[Example, str, str]]) -> list[Tuple[Example, str, str]]:
    return list(set(abs))


def save_lines(path: Path, lines: list[str]):
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def examples_to_ab(examples: list[Example]) -> list[Tuple[Example, str, str]]:
    lines: list[Tuple[Example, str, str]] = []
    for (i, example) in enumerate(examples):
        for desc in example.descs:
            for src in example.srcs:
                lines.append((example, desc, src))
    return lines


def example_statistics(example: Example) -> dict[str, int | str]:
    return {
        'representative_desc': example.descs[0],
        'num_descs': len(example.descs),
        'num_srcs': len(example.srcs),
        'num_product': len(example.descs) * len(example.srcs),
    }


def examples_statistics(examples: list[Example]) -> list[dict[str, int | str]]:
    return [example_statistics(example) for example in examples]


def ab_statistics(abs: list[Tuple[Example, str, str]]) -> Tuple[dict[str, int], int]:
    result = dict()
    synthetic = 0
    for e, desc, src in abs:
        if e.synthetic:
            synthetic += 1
        representative = e.representative()
        if representative not in result:
            result[representative] = 0
        result[representative] += 1
    return result, synthetic


def statistics_to_lines(statistics: dict[str, int]) -> list[str]:
    result: list[str] = []
    sorted_statistics = sorted(statistics.items(), key=lambda x: x[1])
    for k, v in sorted_statistics:
        result.append(f'{v}: {k}\n')
    return result


def apply_cap(abs: list[Tuple[Example, str, str]], cap: int) -> list[Tuple[Example, str, str]]:
    taken: dict[str, int] = dict()
    result: list[Tuple[Example, str, str]] = list()
    for e, desc, src in abs:
        representative = e.representative()
        if representative not in taken:
            taken[representative] = 0
        if taken[representative] < cap:
            result.append((e, desc, src))
            taken[representative] += 1
    return result


def ab_to_lines(abs: list[Tuple[Example, str, str]]) -> list[str]:
    result: list[str] = []
    for e, desc, src in abs:
        result.append(desc_src_to_line(desc, src))
    return result


def ab_to_lines_human(ab: list[Tuple[Example, str, str]]) -> list[str]:
    result: list[str] = []
    for e, desc, src in ab:
        result.append(desc_src_to_line_human(desc, src))
    return result
