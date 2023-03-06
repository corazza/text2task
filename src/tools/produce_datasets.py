from pathlib import Path
from typing import Iterator, Optional, Tuple
import numpy as np
import more_itertools
import IPython
import sys
import os
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

from const import *
from regex_printer import expr_to_str
from regex_validation import equivalent
from tools.train import ModelArguments, DataTrainingArguments
from example_parser import parse_examples, line_iter
import compiler_interface
from compiler_interface import compile


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def ensure_max_length(a: str, b: str, tokenizer):
    encoded = tokenizer('<|bos|>' + a + '<|sep|>' + b + '<|eos|>')
    if len(encoded.input_ids) >= PAD_SIZE:
        print('line too long')
        IPython.embed()  # type: ignore


def example_to_line(p: Tuple[str, str]) -> str:
    return '{"a": "' + p[0] + '", "b":"' + p[1] + '"}\n'


def example_to_line_human(p: Tuple[str, str]) -> str:
    return p[0] + ' => ' + p[1] + '\n'


def save_lines(path: Path, lines: list[str]):
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def load_examples(path: str, validate: bool) -> list[Tuple[str, str]]:
    lines = more_itertools.peekable(line_iter(path))
    return parse_examples(lines, validate)


def augment_examples(examples: list[Tuple[str, str]], num_rewrites: int, test_length: int, num_tests: int, validate: bool) -> list[Tuple[str, str]]:
    new_examples: list[Tuple[str, str]] = []
    for desc, src in examples:
        ast = compiler_interface.parse(src)
        appears = ast.appears()
        rewrites_nodes = ast.rewrites(appears, num_rewrites)
        rewrites: list[str] = [expr_to_str(r) for r in rewrites_nodes]
        if validate:
            dfa, _node_creator = compiler_interface.get_dfa(src)
            for rewrite in rewrites:
                dfa_b, _ = compiler_interface.get_dfa(rewrite)
                ineq_evidence = equivalent(
                    appears, dfa, dfa_b, test_length, num_tests)
                if len(ineq_evidence) != 0:
                    print(src)
                    print(rewrite)
                    IPython.embed()  # type: ignore
                assert len(ineq_evidence) == 0
        new_examples.append((desc, src))
        new_examples.extend([(desc, rewrite) for rewrite in rewrites])
    return new_examples


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer.add_special_tokens({'bos_token': '<|bos|>',
                                  'eos_token': '<|eos|>',
                                  'sep_token': '<|sep|>',
                                  'pad_token': '<|pad|>'})

    path = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'train', '.json')
    path_human = create_if_doesnt_exist(
        'preprocessed_datasets/txt2task', 'train', '.txt')
    examples = load_examples(
        'datasets/txt2task/organic.txt', VALIDATE_EXAMPLES)
    examples = augment_examples(
        examples, REWRITE_INFLATION_FACTOR, REWRITE_VALIDATIN_TEST_LENGTH, REWRITE_VALIDATION_NUM_TESTS, validate=VALIDATE_EXAMPLES)
    for a, b in examples:
        ensure_max_length(a, b, tokenizer)
    np.random.shuffle(examples)  # type: ignore
    lines = [example_to_line(p) for p in examples]
    lines_human = [example_to_line_human(p) for p in examples]
    save_lines(path, lines)
    save_lines(path_human, lines_human)


if __name__ == '__main__':
    main()
