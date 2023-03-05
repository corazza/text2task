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

from const import PAD_SIZE
from tools.train import ModelArguments, DataTrainingArguments

from example_parser import parse_examples, line_iter
from compiler_interface import compile


def create_if_doesnt_exist(in_dir: str, filename: str, suffix: str) -> Path:
    path = Path(in_dir)
    path.mkdir(parents=True, exist_ok=True)
    return Path(path, filename).with_suffix(suffix)


def ensure_max_length(examples: list[Tuple[str, str]], tokenizer):
    for a, b in examples:
        encoded = tokenizer('<|bos|>' + a + '<|sep|>' + b + '<|eos|>')
        if len(encoded.input_ids) >= PAD_SIZE:
            print('line too long')
            IPython.embed()  # type: ignore


def example_to_line(p: Tuple[str, str]) -> str:
    return '{"a": "' + p[0] + '", "b":"' + p[1] + '"}\n'


def save_examples(path: Path, examples: list[Tuple[str, str]], tokenizer):
    ensure_max_length(examples, tokenizer)
    lines = [example_to_line(p) for p in examples]
    with open(path, 'w') as f:
        f.writelines(lines)
        print(f'wrote {len(lines)} lines to {path}')


def load_examples(path: str) -> list[Tuple[str, str]]:
    lines = more_itertools.peekable(line_iter(path))
    return parse_examples(lines)


def validate_examples(examples: list[Tuple[str, str]]):
    for desc, src in examples:
        compile(src)


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
    examples = load_examples('datasets/txt2task/organic.txt')
    validate_examples(examples)
    np.random.shuffle(examples)  # type: ignore
    save_examples(path, examples, tokenizer)


if __name__ == '__main__':
    main()
