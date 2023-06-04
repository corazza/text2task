import random
import time
from typing import Tuple

import IPython
import numpy as np
import pyperclip

from consts import *
from datasets_common import *
from training import get_args, get_tokenizer

MAX_EXAMPLE_DESC_LENGTH: int = 100
NUM_GIVE_EXAMPLES: int = 100
NUM_ASK_EXAMPLES: int = 10


def prompt_example_filter(example: Example) -> bool:
    if len(example.example_rewrites) == 0:
        return False
    for desc in example.descs:
        if len(desc) > MAX_EXAMPLE_DESC_LENGTH:
            return False
    return True


def rewrite_string(r: list[tuple[str, str]]) -> str:
    result: str = ""
    for left, right in r:
        result = f'{result}, {left} % {right}'
    return result[2:]


def single_line(example: Example) -> str:
    desc: str = random_from(example.descs)
    src: str = random_from(example.srcs)
    rewrite: list[tuple[str, str]] = random_from(example.example_rewrites)
    rewrite_printed: str = rewrite_string(rewrite)
    return f'{rewrite_printed} => {desc} => {src}'


def terms_to_string(terms: dict[str, list[str]]) -> str:
    rows: list[str] = []
    for var, tags in terms.items():
        rows.append(f'{var}: {tags}')
    return '\n'.join(rows)


def generate_prompt(load_from: list[str]):
    examples = load_examples(load_from[0])
    terms = load_terms(DEFAULT_TERMS_PATH)

    for load_path in load_from[1:]:
        examples.extend(load_examples(load_path))

    examples = [x for x in examples if prompt_example_filter(x)]

    np.random.shuffle(examples)  # type: ignore
    single_lines: list[str] = [single_line(
        x) for x in examples[:NUM_GIVE_EXAMPLES]]
    example_string: str = '\n'.join(single_lines)

    terms_string: str = terms_to_string(terms)

    prompt: str = f"""Please generate {NUM_ASK_EXAMPLES} new examples for my dataset. Don't repeat examples I show you. Be creative but follow the rules. And DON'T repeat the same examples.

Each example should be a single line in the form "X => Y => Z", where X, Y, and Z are strings such that:

1. X is a list of variable declarations like "$A % OBJECT, $B % PLACE" (OBJECT and PLACE are tags that take specific when I process these examples),
2. Y is an English description of a simple task like "Go to $A", and
3. Z is a regex-like pattern that matches action sequences which satisfy the task described by Y.

Here are {NUM_GIVE_EXAMPLES} examples from my current dataset:

{example_string}

And here's a list of values and tags that variable declarations range over:

{terms_string}

Here are several additional instructions:

1. Pay close attention to how substituting variables like $A % OBJECT will affect grammar. Consult the list of tags. For example if $B ranges over OBJECT, it's better to write "without touching _the_ $B", and not "without touching $B". If unsure, prefer writing "the" before variables where it makes grammatical sense.
2. You don't need to wrap expressions in top-level parentheses, so just "<expression>" instead of "(<expression>)" is fine."""

    pyperclip.copy(prompt)


def main():
    set_all_seeds(int(time.time()*1000) % 2**32)
    load_from = [f'datasets/txt2task/use/{x}.txt' for x in SOURCES]
    generate_prompt(load_from)


if __name__ == '__main__':
    main()
