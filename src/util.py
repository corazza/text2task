import random

import numpy as np
from transformers import pipeline, set_seed

from consts import *


def random_from(xs: list):
    num = len(xs)
    i = np.random.randint(num)
    return xs[i]


def set_all_seeds(seed: int):
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
