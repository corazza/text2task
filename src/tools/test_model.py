import IPython

import model_interface
from consts import *
from reward_machine import RewardMachine
from util import set_all_seeds


def test_model():
    generator = model_interface.get_generator(DEFAULT_USE_MODEL_PATH)
    while True:
        rm: RewardMachine
        desc: str
        src: str
        rm, desc, src = model_interface.answer_query(generator, True, print)
        IPython.embed()


def main():
    set_all_seeds(SEED)
    test_model()


if __name__ == '__main__':
    main()
