from transformers import pipeline, set_seed, GPT2Tokenizer


def synthesize(generator, desc: str) -> str:
    prompt = f'<|bos|>{desc}<|sep|>'
    output = generator(prompt,
                       max_new_tokens=100,
                       bos_token_id=generator.tokenizer.sep_token_id,
                       eos_token_id=generator.tokenizer.eos_token_id,
                       pad_token_id=generator.tokenizer.pad_token_id,
                       num_return_sequences=1)
    generated_text = list(output)[
        0]['generated_text']
    final_output = generated_text.splitlines()[0].strip()
    return final_output.split(generator.tokenizer.sep_token)[-1]


def answer_query(generator):
    desc = input(': ')
    output = synthesize(generator, desc)
    print(output)


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
