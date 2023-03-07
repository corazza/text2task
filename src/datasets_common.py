import copy
from pathlib import Path
from typing import Iterator, Optional, Tuple
import more_itertools
import IPython


from const import *
from regex_printer import expr_to_str
from regex_validation import equivalent
from example_parser import Example, parse_examples, line_iter
import compiler_interface
from compiler_interface import compile


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def load_examples(path: str) -> list[Example]:
    lines = more_itertools.peekable(line_iter(path))
    return parse_examples(lines)


def augment_examples(examples: list[Example], num_rewrites: int) -> list[Example]:
    examples = ast_rewrites(examples, num_rewrites)
    return examples


def text_rewrites(examples: list[Example]) -> list[Example]:
    raise NotImplementedError()


def ast_rewrites(examples: list[Example], num_rewrites: int) -> list[Example]:
    for example in examples:
        new_srcs: list[str] = []
        for src in example.srcs:
            ast = compiler_interface.parse(src)
            appears = ast.appears()
            rewrites_nodes = ast.rewrites(appears, num_rewrites)
            rewrites: list[str] = [expr_to_str(r) for r in rewrites_nodes]
            new_srcs.extend(rewrites)
        example.srcs.extend(new_srcs)
    return examples


def desc_src_to_line(a: str, b: str) -> str:
    return '{"a": "' + a + '", "b":"' + b + '"}\n'


def desc_src_to_line_human(a: str, b: str) -> str:
    return a + ' => ' + b + '\n'


def validate_runs(examples: list[Example]):
    for example in examples:
        if len(example.runs) == 0:
            print(f'no runs to validate: {example.srcs}')
            continue
        for src in example.srcs:
            rm = compiler_interface.compile(src)
            for reward, run in example.runs:
                rewards = rm.multiple_transitions(0, run)
                assert reward == sum(
                    rewards), f'{reward} @ {run} failed for {src} -> {rewards}'


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


def ensure_max_length(a: str, b: str, tokenizer):
    encoded = tokenizer('<|bos|>' + a + '<|sep|>' + b + '<|eos|>')
    if len(encoded.input_ids) >= PAD_SIZE:
        print('line too long')
        IPython.embed()  # type: ignore


def validate_length(ab: list[Tuple[str, str]], tokenizer):
    for desc, src in ab:
        ensure_max_length(desc, src, tokenizer)


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
