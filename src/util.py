import numpy as np


def random_from(xs: list):
    num = len(xs)
    i = np.random.randint(num)
    return xs[i]
