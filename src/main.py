import IPython

import data_generator
import compiler_interface
import organic_data_augmentor
from tools import produce_datasets
import desc_rewriter


def test_generator():
    prompts = data_generator.get_default(100)
    IPython.embed()


def test_compiler():
    rm = compiler_interface.compile('(A B A&!B)+ C')
    IPython.embed()


def test_augmentor():
    organic_prompts = organic_data_augmentor.load_file(
        '../datasets/text2task/f_test.txt', '../datasets/text2task/prop.txt', 10)
    path_human = produce_datasets.create_if_doesnt_exist(
        '../preprocessed_datasets/text2task', 'train_test', '.txt')
    produce_datasets.save_prompts_human(path_human, organic_prompts)
    IPython.embed()


def main():
    test_augmentor()


if __name__ == "__main__":
    main()
