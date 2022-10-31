from transformers import pipeline, set_seed, GPT2Tokenizer
import IPython


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


def test_model():
    generator = pipeline(
        'text-generation', model='/mnt/e/work_dirs/text2task_distilgpt2/')
    set_seed(42)
    desc = "repeat TRAP"
    output = synthesize(generator, desc)
    print(f'{desc} => {output}')
    IPython.embed()


def main():
    test_model()


if __name__ == '__main__':
    main()
