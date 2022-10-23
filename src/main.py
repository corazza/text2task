from transformers import pipeline, set_seed, GPT2Tokenizer
import IPython

import data_loader
import compiler_interface
import example_rms
import describe
import rm_compiler


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


def describe_test():
    data = data_loader.load_file('../datasets/f1.txt')
    # src = data.entries[0].expr_sources[0]
    src = '(COFFEE MAIL | MAIL COFFEE)* OFFICE'
    parsed = compiler_interface.parse(src)
    IPython.embed()


def compiler_test():
    # HERE
    # probably don't want to learn to specify these semantics
    # implicit self-loops are good
    # figure out the regex->rm semantics map
    src = '(!DECORATION)* (COFFEE&!DECORATION (!DECORATION)* MAIL&!DECORATION | MAIL&!DECORATION (!DECORATION)* COFFEE&!DECORATION) (!DECORATION)* OFFICE'
    rm = compiler_interface.compile(src)
    IPython.embed()


def rm_test():
    pass


def main():
    compiler_test()


if __name__ == "__main__":
    main()
