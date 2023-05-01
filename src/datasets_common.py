import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
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
from regex_ast import RENode
from regex_printer import expr_to_str
from regex_validation import equivalent
from example_parser import Example, parse_examples
from parser_util import line_iter
import compiler_interface
from compiler_interface import compile
from terms_parser import parse_terms
from util import *


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


def has_without_dot(example: Example) -> bool:
    for desc in example.descs:
        if '.' not in desc:
            return True
    return False


def get_eligible_pairs(examples: list[Example]) -> list[Tuple[int, int]]:
    eligible_pairs: list[Tuple[int, int]] = []
    for i, example1 in enumerate(examples):
        for j, example2 in enumerate(examples):
            if i >= j:
                continue
            if example1.average_desc_length() > DESC_LENGTH_LIMIT or example2.average_desc_length() > DESC_LENGTH_LIMIT:
                continue
            if not has_without_dot(example1) or not has_without_dot(example2):
                continue
            eligible_pairs.append((i, j))
    return eligible_pairs


def load_patterns(path: str) -> dict[str, list[Example]]:
    as_examples: list[Example] = load_examples(path)
    result: dict[str, list[Example]] = dict()
    for example in as_examples:
        assert example.id != '-1'
        if example.id not in result:
            result[example.id] = [example]
        else:
            result[example.id].append(example)
    return result


def get_adds(pattern_id: str, example: Example) -> str:
    if pattern_id == 'conjunct':
        rep_src = example.srcs[0]
        ast = compiler_interface.parse(rep_src)
        return '' if ast.repetative() else ' > (.)*'
    else:
        return ''


def apply_replacements(original: str, which: list[Tuple[str, str]]) -> str:
    for left, right in which:
        original = original.replace(left, right)
    return original


def eligible_combination_in_pair(pattern_desc: str, example_descs: list[str]) -> bool:
    cant_appear_both = ['first', 'second', 'finally']
    for example_desc in example_descs:
        for disallowed in cant_appear_both:
            if disallowed in pattern_desc.lower() and example_desc.lower():
                return False
    return True


def eligible_desc_in_pair(example_desc: str) -> bool:
    return '.' not in example_desc


def apply_pattern(pattern: Example, examples: list[Example], eligible_desc_filter, eligible_combination_filter) -> Example:
    representatives: list[str] = [x.representative() for x in examples]
    new_example_rewrites: list[list[Tuple[str, str]]] = []
    new_runs: list[Tuple[int, list[frozenset[str]]]] = []
    new_descs: list[str] = []
    new_srcs: list[str] = []
    new_id: str = f'_ADDED_{pattern.id}_' + '|'.join(representatives)

    merged_rewrites: list[list[Tuple[str, str]]] = merge_rewrites(examples)
    if len(pattern.example_rewrites) > 0:
        for r in merged_rewrites:
            for pattern_r in pattern.example_rewrites:
                new_example_rewrites.append(r + pattern_r)
    else:
        new_example_rewrites = merged_rewrites

    src_candidates: list[list[str]] = [x.srcs for x in examples]
    desc_candidates: list[list[str]] = [
        list(filter(eligible_desc_filter, x.descs)) for x in examples]

    desc_combinations: list[list[str]] = list(
        itertools.product(*desc_candidates))
    src_combinations: list[list[str]] = list(
        itertools.product(*src_candidates))

    assert len(desc_combinations) > 0

    adds: list[str] = [get_adds(pattern.id, x) for x in examples]

    for chosen_descs in desc_combinations:
        for pattern_desc in pattern.descs:
            if not eligible_combination_filter(pattern_desc, chosen_descs):
                continue
            replacements: list[Tuple[str, str]] = []
            for i in range(len(examples)):
                desc_i: str = map_desc(
                    examples[i].vars(), AUGMENT_CHAR_LIST[i], chosen_descs[i])
                desc_i = desc_i[0].lower() + desc_i[1:]
                replacements.append((f'DESC{i}', desc_i))
                replacements.append((f'ADD{i}', adds[i]))
            new_descs.append(clean_desc(
                apply_replacements(pattern_desc, replacements)))
    for chosen_srcs in src_combinations:
        for pattern_src in pattern.srcs:
            replacements: list[Tuple[str, str]] = []
            for i in range(len(examples)):
                src_i: str = map_desc(
                    examples[i].vars(), AUGMENT_CHAR_LIST[i], chosen_srcs[i])
                replacements.append((f'SRC{i}', src_i))
                replacements.append((f'ADD{i}', adds[i]))
            new_srcs.append(apply_replacements(pattern_src, replacements))
    return Example(True, new_example_rewrites, new_runs, new_descs, new_srcs, new_id)


def augmented_ab(patterns: dict[str, list[Example]], examples: list[Example], num_abs) -> list[Tuple[Example, str, str]]:
    new_abs: list[Tuple[Example, str, str]] = []
    np.random.shuffle(examples)  # type: ignore
    eligible_pairs: list[Tuple[int, int]] = get_eligible_pairs(examples)
    eligible_single: list[int] = list(range(len(examples)))

    if len(eligible_pairs) < 1 or len(eligible_single) < 1:
        return []

    to_add_pairs: dict[str, int] = {
        'concat': round(ADD_CONCAT_P * ADD_P * num_abs),
        'disjunct': round(ADD_DISJUNCT_P * ADD_P * num_abs),
        'conjunct': round(ADD_CONJUNCT_P * ADD_P * num_abs),
    }

    for pattern_id, pattern_num in to_add_pairs.items():
        print(f'adding {pattern_id}... ({pattern_num})')
        np.random.shuffle(eligible_pairs)
        for counter in range(pattern_num):
            pattern = random_from(patterns[pattern_id])
            ij_pair = list(eligible_pairs[counter % len(eligible_pairs)])
            np.random.shuffle(ij_pair)
            i, j = ij_pair
            example1 = examples[i]
            example2 = examples[j]
            example = apply_pattern(
                pattern, [example1, example2], eligible_desc_in_pair, eligible_combination_in_pair)
            new_abs.append(get_single_ab(example))

    to_add_single: dict[str, int] = {
        'avoid': round(ADD_AVOIDANCE_P * ADD_P * num_abs)
    }

    for pattern_id, pattern_num in to_add_single.items():
        print(f'adding {pattern_id}... ({pattern_num})')
        np.random.shuffle(eligible_single)
        for counter in range(pattern_num):
            pattern = random_from(patterns[pattern_id])
            i = eligible_single[counter % len(eligible_single)]
            example = examples[i]
            new_example = apply_pattern(
                pattern, [example], lambda x: True, lambda x, y: True)
            new_abs.append(get_single_ab(new_example))

    return new_abs


def get_single_ab(example: Example):
    np.random.shuffle(example.descs)
    np.random.shuffle(example.srcs)
    ast = compiler_interface.parse(example.srcs[0])
    return example, example.descs[0], expr_to_str(ast)


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


def ab_rewrites(abs: list[Tuple[Example, str, str]], terms, to_cap: bool) -> list[Tuple[Example, str, str]]:
    np.random.shuffle(abs)  # type: ignore
    statistics = ab_statistics(abs)
    new_abs: list[Tuple[Example, str, str]] = []
    for i, (example, desc, src) in enumerate(abs):
        # if i % 250 == 0:
        #     print(f'{i}/{len(abs)}')
        r = example.representative()
        if not to_cap:
            to_take = 1
        else:
            id = example.representative() + '|' + desc
            to_take = max(1, round(SENTENCE_CAP / statistics[id]))
        new_rewrites = get_new_rewrites(example, terms, to_take)[:to_take]
        for rewrite in new_rewrites:
            r_desc = apply_text_rewrite_with_concat(desc, rewrite, ' ')
            r_src = apply_text_rewrite_with_concat(src, rewrite, '_')
            new_abs.append((example, r_desc, r_src))
        if '$' not in src and '$' not in desc:
            new_abs.append((example, desc, src))
    return new_abs


def get_new_rewrites(example: Example, terms, to_take: int) -> list[list[Tuple[str, str]]]:
    new_rewrites: list[list[Tuple[str, str]]] = []
    for rewrite in example.example_rewrites:
        counter: int = to_take
        new_versions: list[list[Tuple[str, str]]] = []
        all_replacements: dict[str, list[str]] = dict()
        for left, right in rewrite:
            if right.isupper():
                all_replacements[left] = get_all_terms_from_tag(
                    terms, right)
            else:
                all_replacements[left] = [right]
        for k in all_replacements:
            np.random.shuffle(all_replacements[k])
        for combination in itertools.product(*all_replacements.values()):
            if counter == 0:
                break
            if len(combination) != len(set(combination)):
                continue
            new_dict = {k: v for k, v in zip(
                all_replacements.keys(), combination)}
            new_versions.append(list(new_dict.items()))
            counter -= 1
        if len(new_versions) > 0:
            new_rewrites.extend(new_versions)
        else:
            has_vars = False
            for left, right in rewrite:
                if right.isupper():
                    has_vars = True
            if not has_vars:
                new_rewrites.append(rewrite)

    np.random.shuffle(new_rewrites)
    seen = set()
    for rewrite in new_rewrites:
        seen.add(tuple(rewrite))
    assert len(seen) == len(new_rewrites)
    return new_rewrites


def clean_desc(desc: str) -> str:
    desc = desc.replace('..', '.')
    desc = desc.replace('. .', '.')
    desc = desc.replace('.,', ',')
    desc = desc.replace('. ,', ',')
    if np.random.random() < 0.5:
        desc = desc[0].upper() + desc[1:]
    return desc


def map_rewrite(letter: str, example_rewrites: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
    return [(f'{variable}{letter}' if '$' in variable else variable, to_class) for (variable, to_class) in example_rewrites]


def map_trace(letter: str, trace: list[frozenset[str]]) -> list[frozenset[str]]:
    return [frozenset([f'{v}{letter}' if '$' in v else v for v in vars]) for vars in trace]


def map_desc(vars: list[str], letter: str, desc: str) -> str:
    for v in vars:
        desc = desc.replace(v, f'{v}{letter}' if '$' in v else v)
    return desc


def merge_rewrites(examples: list[Example]) -> list[list[Tuple[str, str]]]:
    old_rewrites: list[list[list[Tuple[str, str]]]] = [
        x.example_rewrites if len(x.example_rewrites) > 0 else [[]] for x in examples]
    new_example_rewrites: list[list[Tuple[str, str]]] = []
    for combination in itertools.product(*old_rewrites):
        new_rewrite_list: list[Tuple[str, str]] = []
        for i, rewrite_list in enumerate(combination):
            mapped = map_rewrite(f'{AUGMENT_CHAR_LIST[i]}', rewrite_list)
            new_rewrite_list.extend(mapped)
        new_example_rewrites.append(new_rewrite_list)
    return new_example_rewrites


def filter_rewrites(srcs, rewrites_nodes: list[RENode]) -> list[RENode]:
    filtered = []
    gotten_srcs: set[str] = set()
    for node in rewrites_nodes:
        rewrite_src = expr_to_str(node)
        found = False
        for src in srcs:
            if rewrite_src == src:
                found = True
        if not found and rewrite_src not in gotten_srcs:
            filtered.append(node)
            gotten_srcs.add(rewrite_src)
    return filtered


def ast_rewrites(examples: list[Example]) -> list[Example]:
    statistics2: dict[str, int] = {'has_demorgan': 0, 'no_demorgan': 0}
    for i, example in enumerate(examples):
        to_take: int = max(1, SENTENCE_CAP - len(example.srcs))
        new_srcs: list[str] = []
        asts = [compiler_interface.parse(src) for src in example.srcs]
        for src, ast in zip(example.srcs, asts):
            rewrites_nodes = ast.rewrites()
            np.random.shuffle(rewrites_nodes)  # type: ignore
            rewrites_nodes = rewrites_nodes[:to_take]
            filtered = filter_rewrites(example.srcs, rewrites_nodes)
            rewrites = [expr_to_str(f) for f in filtered]
            for ast2 in filtered:
                rewrite_statistics = ast2.get_statistics()
                if rewrite_statistics.num_demorgan > 0:
                    statistics2['has_demorgan'] += 1
                else:
                    statistics2['no_demorgan'] += 1
            new_srcs.extend(rewrites)
        np.random.shuffle(new_srcs)
        new_srcs = new_srcs[:to_take]
        example.srcs.extend(new_srcs)
    return examples


def desc_src_to_line(a: str, b: str) -> str:
    return '{"a": "' + a.lower() + '", "b":"' + b.lower() + '"}'


def desc_src_to_line_human(a: str, b: str) -> str:
    return a.lower() + ' => ' + b.lower()


# @profile(sort_by='tottime')
def validate_runs(examples: list[Example]):
    for (i, example) in enumerate(examples):
        print(f'{i}/{len(examples)}')
        if len(example.runs) == 0:
            print(f'no runs to validate: {example.srcs}')
            continue
        rewards_sets: dict[Tuple[frozenset[str]], set[Tuple[int]]] = dict()
        for (i_src, src) in enumerate(example.srcs):
            print(f'    src={i_src}/{len(example.srcs)}')
            # ast = compiler_interface.parse(src)
            # src2 = expr_to_str(ast)
            rm = compiler_interface.compile(src)
            # rm2 = compiler_interface.compile(src2)
            for (i_rr, (reward, run)) in enumerate(example.runs):
                rewards = rm.multiple_transitions(0, run)
                # rewards2 = rm2.multiple_transitions(0, run)
                # assert rewards == rewards2, 'printer failed'
                if tuple(run) not in rewards_sets:
                    rewards_sets[tuple(run)] = set()
                rewards_sets[tuple(run)].add(tuple(rewards))
                if POSNEG_VALIDATION:
                    assert (reward > 0) == (sum(
                        rewards) > 0), f'{reward} @ {run} failed for {src} -> {rewards}'
                else:
                    assert reward == sum(
                        rewards), f'{reward} @ {run} failed for {src} -> {rewards}'
        for run, rewards_set in rewards_sets.items():
            assert len(
                rewards_set) == 1, f'failed on run {run}, rewards {rewards_set}'


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
    assert '$' not in desc


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
    synthetic = 0
    for e, desc, src in abs:
        if e.synthetic:
            synthetic += 1
        representative = e.representative() + '|' + desc
        if representative not in result:
            result[representative] = 0
        result[representative] += 1
    return result


def statistics_to_lines(statistics: dict[str, int]) -> list[str]:
    result: list[str] = []
    sorted_statistics = sorted(statistics.items(), key=lambda x: x[1])
    for k, v in sorted_statistics:
        result.append(f'{v}: {k}\n')
    return result


def apply_cap(abs: list[Tuple[Example, str, str]], cap: int) -> list[Tuple[Example, str, str]]:
    np.random.shuffle(abs)  # type: ignore
    taken: dict[str, int] = dict()
    result: list[Tuple[Example, str, str]] = list()
    for e, desc, src in abs:
        representative = e.representative() + '|' + desc
        if representative not in taken:
            taken[representative] = 0
        if taken[representative] < cap:
            result.append((e, desc, src))
            taken[representative] += 1
    return result


def paraphrase_ab(abs: list[Tuple[Example, str, str]]) -> list[Tuple[Example, str, str]]:
    new_abs: list[Tuple[Example, str, str]] = []
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(  # type: ignore
        model_name).to(torch_device)  # type: ignore

    def get_response(input_text, num_return_sequences, num_beams) -> list[str]:
        batch = tokenizer([input_text], truncation=True, padding='longest',
                          max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch, max_length=2*len(input_text), num_beams=num_beams,
                                    num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def has_vars(e: Example, a: str) -> bool:
        for v in e.vars():
            if v not in a:
                return False
        return True

    num_beams = 10
    num_return_sequences = 10
    fail_counter: int = 0
    for e, a, b in abs:
        try:
            p_as: list[str] = get_response(a, num_return_sequences, num_beams)
            p_as = [p_a for p_a in p_as if has_vars(e, p_a)]
            p_a: str = max(p_as, key=len)
            new_abs.append((e, p_a, b))
        except:
            fail_counter += 1
            new_abs.append((e, a, b))
    print(f'total={len(new_abs)}, failed={fail_counter}')
    return new_abs


def paraphrase_split(abs: list[Tuple[Example, str, str]], prop: float) -> Tuple[list[Tuple[Example, str, str]], list[Tuple[Example, str, str]]]:
    np.random.shuffle(abs)  # type: ignore
    num_original = int(len(abs) * prop)
    original: list[Tuple[Example, str, str]] = abs[:num_original]
    to_paraphrase: list[Tuple[Example, str, str]] = abs[num_original:]
    paraphrased: list[Tuple[Example, str, str]] = paraphrase_ab(to_paraphrase)
    assert len(original) + len(paraphrased) == len(abs)
    IPython.embed()  # type: ignore
    return original, paraphrased


def ab_to_lines(abs: list[Tuple[Example, str, str]]) -> list[str]:
    result: list[str] = []
    for e, desc, src in abs:
        result.append(desc_src_to_line(desc, src) + '\n')
    return result


def ab_to_lines_human(ab: list[Tuple[Example, str, str]]) -> list[str]:
    result: list[str] = []
    for e, desc, src in ab:
        result.append(desc_src_to_line_human(desc, src) + '\n')
    return result


def ab_to_lines_synthetic(ab: list[Tuple[Example, str, str]]) -> list[str]:
    result: list[str] = []
    for e, desc, src in ab:
        # result.append(
        #     f'{desc_src_to_line_human(desc, src)}         ~ {e.id}\n')
        skip = len('_ADDED_')
        prefix = e.id[skip:skip+len('disjunct')]
        result.append(f'{desc_src_to_line_human(desc, src)} ~ {prefix}\n')
    return result
