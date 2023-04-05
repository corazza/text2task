import itertools
import numpy as np
import copy
from curses.ascii import isalpha
from pathlib import Path
from typing import Iterator, Optional, Tuple
import more_itertools
import IPython


from consts import *
from regex_printer import expr_to_str
from regex_validation import equivalent
from example_parser import Example, parse_examples
from parser_util import line_iter
import compiler_interface
from compiler_interface import compile
from terms_parser import parse_terms


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
    # HERE TODO not as probability, but as proportion. pick x and (x, y) combinations beforehand
    # pick out from (e, a, b)
    examples = ast_rewrites(examples)
    examples = add_concat(examples)
    examples = add_avoidance(examples)
    return examples


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


def add_term_rewrites(examples: list[Example], terms: dict[str, list[str]], num_new: int) -> list[Example]:
    for example in examples:
        num_old_rewrites = len(example.example_rewrites)
        new_rewrites: list[list[Tuple[str, str]]] = []
        for rewrite in example.example_rewrites:
            new_versions: list[list[Tuple[str, str]]] = []
            all_replacements: dict[str, list[str]] = dict()
            gotten_terms: set[str] = set()
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
        to_take: int = min(num_new_rewrites, num_new*num_old_rewrites)
        example.example_rewrites = new_rewrites[:to_take]
    return examples


def text_rewrites(examples: list[Example]) -> list[Example]:
    new_examples: list[Example] = []
    for example in examples:
        for rewrite in example.example_rewrites:
            new_runs: list[Tuple[int, list[frozenset[str]]]] = []
            for (reward, runs) in example.runs:
                new_traces: list[frozenset[str]] = [
                    frozenset({apply_text_rewrite_with_concat(x, rewrite, '_') for x in run}) for run in runs]
                new_runs.append((reward, new_traces))
            new_descs: list[str] = [apply_text_rewrite_with_concat(
                desc, rewrite, ' ') for desc in example.descs]
            new_srcs: list[str] = [apply_text_rewrite_with_concat(
                src, rewrite, '_') for src in example.srcs]
            new_example = Example([], new_runs, new_descs,
                                  new_srcs, example.id)
            new_example.parent = example
            new_examples.append(new_example)
    return examples + new_examples


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
    new_id: str = '-1' if example.id == '-1' else f'{example.id}_ADDED_AVOID'

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

    return Example(new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def map_rewrite(letter: str, example_rewrites: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
    return [(f'{variable}{letter}', to_class) for (variable, to_class) in example_rewrites]


def map_trace(letter: str, trace: list[frozenset[str]]) -> list[frozenset[str]]:
    return [frozenset([f'{v}{letter}' for v in vars]) for vars in trace]


def map_desc(rewrite_list: list[Tuple[str, str]], letter: str, desc: str) -> str:
    vars = list(reversed(sorted([x for (x, y) in rewrite_list])))
    for v in vars:
        desc = desc.replace(v, f'{v}{letter}')
    return desc


def with_concat(example1: Example, example2: Example) -> Example:
    new_example_rewrites: list[list[Tuple[str, str]]] = [map_rewrite('Q', rewrite_list1) + map_rewrite(
        'P', rewrite_list2) for rewrite_list1 in example1.example_rewrites for rewrite_list2 in example2.example_rewrites]
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'{example1.id}_CONCAT_{example2.id}'

    for (reward1, trace1) in example1.runs:
        for (reward2, trace2) in example2.runs:
            reward = reward2 if reward1 > 0 and reward2 > 0 else 0
            trace = map_trace('Q', trace1) + map_trace('P', trace2)
            new_runs.append((reward, trace))

    for src1_or in example1.srcs:
        for src2 in example2.srcs:
            src1 = map_desc(example1.example_rewrites[0], 'Q', src1_or)
            src2 = map_desc(example2.example_rewrites[0], 'P', src2)
            new_src = f'({src1}) > ({src2})'
            new_srcs.append(new_src)

    for desc1_or in example1.descs:
        for desc2 in example2.descs:
            desc1 = desc1_or
            desc1 = map_desc(example1.example_rewrites[0], 'Q', desc1)
            desc2 = map_desc(example2.example_rewrites[0], 'P', desc2)
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

    return Example(new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def random_from(xs: list):
    num = len(xs)
    i = np.random.randint(num)
    return xs[i]


def add_avoidance(examples: list[Example]) -> list[Example]:
    new_examples: list[Example] = []
    for example in examples:
        eligible_classes = ['NO_THE', 'OBJECT_A',
                            'AVOID', 'STEP_ON_A', 'FALL_IN_A', 'HIT_A']
        pvar_class = random_from(eligible_classes)
        if np.random.random() < ADD_AVOIDANCE_P/2.0:
            new_examples.append(with_avoidance(pvar_class, example))
    print(f'avoidance added {len(new_examples)} new examples')
    return examples + new_examples


def add_concat(examples: list[Example]) -> list[Example]:
    num_eligible = len(
        [x for x in examples if x.average_desc_length() <= DESC_LENGTH_LIMIT])
    add_p = ADD_CONCAT_P / num_eligible
    print(f'eligible={num_eligible}/{len(examples)}')
    new_examples: list[Example] = []
    for i, example1 in enumerate(examples):
        for j, example2 in enumerate(examples):
            if i == j:
                continue
            if example1.average_desc_length() > DESC_LENGTH_LIMIT or example2.average_desc_length() > DESC_LENGTH_LIMIT:
                continue
            if len(example1.example_rewrites) == 0 or len(example2.example_rewrites) == 0:
                continue  # TODO fix in with_concat
            if np.random.random() < add_p:
                new_examples.append(with_concat(example1, example2))
    print(f'concat added {len(new_examples)} new examples')
    return examples + new_examples


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


def ab_statistics(abs: list[Tuple[Example, str, str]]) -> dict[str, int]:
    result = dict()
    for e, desc, src in abs:
        representative = e.representative()
        if representative not in result:
            result[representative] = 0
        result[representative] += 1
    return result


def apply_cap(abs: list[Tuple[Example, str, str]]) -> list[Tuple[Example, str, str]]:
    taken: dict[str, int] = dict()
    result: list[Tuple[Example, str, str]] = list()
    for e, desc, src in abs:
        representative = e.representative()
        if representative not in taken:
            taken[representative] = 0
        if taken[representative] < SENTENCE_CAP:
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
