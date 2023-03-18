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
    examples = ast_rewrites(examples)
    return examples


def apply_text_rewrite_with_concat(x: str, rewrite: list[Tuple[str, str]], sep: str) -> str:
    new_string: str = x
    for left, right in rewrite:
        if ' ' in right:
            right = sep.join(right.split(' '))
            right = right.replace('the_', '')
        new_string = new_string.replace(left, right)
    return new_string


def get_random_term_example_from_tag(terms: dict[str, list[str]], term_tag: str) -> str:
    applicable: list[str] = []
    for term, tags in terms.items():
        if term_tag in tags:
            applicable.append(term)
    return np.random.choice(np.array(applicable))


def add_term_rewrites(examples: list[Example], terms: dict[str, list[str]], num_new: int) -> list[Example]:
    for example in examples:
        new_rewrites: list[list[Tuple[str, str]]] = []
        for rewrite in example.example_rewrites:
            new_versions: list[list[Tuple[str, str]]] = []
            for i in range(num_new):
                new_rewrite: list[Tuple[str, str]] = []
                found_new_version: bool = False
                for left, right in rewrite:
                    if right.isupper():
                        new_rewrite.append(
                            (left, get_random_term_example_from_tag(terms, right)))
                        found_new_version = True
                    else:
                        new_rewrite.append((left, right))
                if found_new_version:
                    new_versions.append(new_rewrite)
            if len(new_versions) > 0:
                new_rewrites.extend(new_versions)
            else:
                new_rewrites.append(rewrite)
        example.example_rewrites = new_rewrites
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
            new_examples.append(
                Example([], new_runs, new_descs, new_srcs))
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


def example_length(ab: Tuple[str, str], tokenizer) -> int:
    encoded = tokenizer('<|bos|>' + ab[0] + '<|sep|>' + ab[1] + '<|eos|>')
    return len(encoded.input_ids)


def validate_length(abs: list[Tuple[str, str]], tokenizer):
    for ab in abs:
        ab_len = example_length(ab, tokenizer)
        assert ab_len <= PAD_SIZE, f'{ab[0]} => {ab[1]} over {PAD_SIZE}: {ab_len}'


def filter_length(abs: list[Tuple[str, str]], tokenizer):
    result = list(
        filter(lambda x: example_length(x, tokenizer) <= PAD_SIZE, abs))
    return result


def sanity_check(ab: Tuple[str, str]):
    desc, src = ab
    for i, c in enumerate(src):
        if i == 0 or i == len(src) - 1:
            continue
        if c == ' ':
            assert not isalpha(src[i-1]) or not isalpha(src[i+1])


def sanity_checks(abs: list[Tuple[str, str]]):
    for ab in abs:
        sanity_check(ab)


def remove_residuals(abs: list[Tuple[str, str]]) -> list[Tuple[str, str]]:
    result: list[Tuple[str, str]] = []
    for a, b in abs:
        if '$' in a or '$' in b:
            continue
        result.append((a, b))
    return result


def save_lines(path: Path, lines: list[str]):
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def examples_to_ab(examples: list[Example]) -> list[Tuple[str, str]]:
    lines: list[Tuple[str, str]] = []
    for example in examples:
        for desc in example.descs:
            for src in example.srcs:
                lines.append((desc, src))
    return lines


def ab_to_lines(ab: list[Tuple[str, str]]) -> list[str]:
    result: list[str] = []
    for desc, src in ab:
        result.append(desc_src_to_line(desc, src))
    return result


def ab_to_lines_human(ab: list[Tuple[str, str]]) -> list[str]:
    result: list[str] = []
    for desc, src in ab:
        result.append(desc_src_to_line_human(desc, src))
    return result
