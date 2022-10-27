from happytransformer import HappyTextToText
from happytransformer import TTSettings
import language_tool_python
from transformers import pipeline, set_seed, GPT2Tokenizer
import IPython
import numpy as np

import data_loader
import compiler_interface
import example_rms
import describe
import rm_compiler
import describe_patterns
import rm_generator
import data_generator


def models_test():
    generator = pipeline('text-generation', model='distilgpt2')
    set_seed(42)
    demonstrations = data_loader.load_lines('../training_data_tmp/train.txt')
    query = 'get COFFEE and MAIL then go to SPOTA'
    demonstrations = demonstrations[0:10]
    lines = demonstrations + [f'<|endoftext|>{query} =>']
    lines = demonstrations + [f'{query} =>']
    prompt = "\n".join(lines)
    # output = generator(prompt, return_full_text=False,
    #                    max_new_tokens=100, num_return_sequences=1)
    # generated_text = list(output)[
    #     0]['generated_text']
    # final_output = generated_text.splitlines()[0].strip()
    # print(prompt)
    # print(f'{final_output}')

    outputs = generator.tokenizer(  # type:ignore
        prompt,
        truncation=True,
        max_length=100,
        return_overflowing_tokens=True,
        return_length=True,
    )

    print(f"Input IDs length: {len(outputs['input_ids'])}")  # type:ignore
    print(f"Input chunk lengths: {(outputs['length'])}")
    print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    for o in outputs['input_ids']:  # type:ignore
        print(generator.tokenizer.decode(o))  # type:ignore
        print()
        print()
    IPython.embed()


def compiler_test():
    # HERE
    # probably don't want to learn to specify these semantics
    # implicit self-loops are good
    # figure out the regex->rm semantics map
    src = '(COFFEE&!DECORATION MAIL&!DECORATION | MAIL&!DECORATION COFFEE&!DECORATION) OFFICE&!DECORATION'
    rm = compiler_interface.compile(src)
    IPython.embed()


def improve_desc(happy_tt: HappyTextToText, settings: TTSettings, desc: str) -> str:
    result = happy_tt.generate_text(desc, args=settings)
    return result.text


def improve_desc_lt(tool: language_tool_python.LanguageTool, props: list[str], desc: str) -> str:
    ignore = [
        'UPPERCASE_SENTENCE_START'
    ]

    def _filter_ignore(m):
        return m.ruleId not in ignore

    def _filter_props(m):
        return desc[m.offset:m.offset+m.errorLength] not in props

    matches = tool.check(desc)
    matches = list(filter(_filter_ignore, matches))
    matches = list(filter(_filter_props, matches))

    if len(matches) == 0:
        return desc

    IPython.embed()
    raise NotImplementedError()


def generator_test():
    dist_parameters = {
        'exp_children': 1.2,  # defines exponential distr. for # of children
        'clip_children': 3,
        'exp_props': 0.5,  # defines exponential distr. for # of propvars in transitions
        'bin_negate': 0.05,  # probability to negate a propvar in transitions
    }
    prompts = data_generator.generate_synthetic(dist_parameters, 100)
    IPython.embed()


def main():
    generator_test()


if __name__ == "__main__":
    main()
