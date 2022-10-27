from pathlib import Path
from typing import Tuple
import numpy as np

import data_loader
import compiler_interface
import example_rms
import describe
import rm_compiler
import describe_patterns
import rm_generator


def generate_synthetic(props_path: str | Path, var_path: str | Path, patterns_path: str | Path, dist_parameters: dict[str, float], complexity: int, n: int) -> list[Tuple[str, str]]:
    # happy_tt = HappyTextToText("T5", "prithivida/grammar_error_correcter_v1")
    # tool = language_tool_python.LanguageTool('en-US')
    props = rm_generator.load_props(props_path)
    var_describe_map = describe.load_var_describe_map(var_path)
    patterns = describe_patterns.load_patterns(patterns_path)

    prompts = []

    for i in range(n):
        expr = rm_generator.generate(dist_parameters, props, complexity)
        desc = describe.describe(patterns, var_describe_map, expr)
        chosen = np.random.randint(0, len(desc))
        chosen_desc = desc[chosen]
        # num_tokens = len(happy_tt.tokenizer(chosen_desc)
        #                  ['input_ids'])  # type: ignore
        # settings = TTSettings(do_sample=True, top_k=50,
        #                       temperature=0.9,  min_length=min(0, num_tokens-5), max_length=num_tokens+5)
        # improved_desc = improve_desc(happy_tt, settings, chosen_desc)
        # improved_desc = improve_desc_lt(
        #     tool, list(var_describe_map.keys()), chosen_desc)
        improved_desc = chosen_desc
        prompts.append((improved_desc, expr))

    return prompts
