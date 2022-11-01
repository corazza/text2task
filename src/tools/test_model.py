import sys  # noqa
sys.path.append('.')  # noqa
from transformers import pipeline, set_seed, GPT2Tokenizer
import IPython

import compiler_interface
import expr_printer


def synthesize(generator, desc: str) -> str:
    prompt = f'{desc} =>'
    output = generator(prompt,
                       return_full_text=False,
                       pad_token_id=-100,
                       eos_token_id=-100,
                       max_new_tokens=100,
                       num_return_sequences=1)
    generated_text = list(output)[
        0]['generated_text']
    final_output = generated_text.splitlines()[0].strip()
    return final_output


# def postprocess(output: str) -> str:  # TODO fix this in the model
#     for i in range(len(output)):
#         substr = output[0:i+1]
#         try:
#             parsed = compiler_interface.parse(substr)
#         except:
#             continue
#         return expr_printer.expr_to_str(parsed)
#     raise ValueError('couldn\'t find a valid substring')


def answer_query(generator):
    desc = input(': ')
    output = synthesize(generator, desc)
    print(output)
    # print(postprocess(output))


def test_model():
    generator = pipeline(
        'text-generation', model='/mnt/e/work_dirs/text2task_distilgpt2/')
    set_seed(42)
    while True:
        answer_query(generator)


def main():
    test_model()


if __name__ == '__main__':
    main()
