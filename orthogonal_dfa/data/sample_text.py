import numpy as np

from .exon import text


def sample_text(seed, count):
    length = len(text)
    random = np.random.RandomState(seed).choice(4, size=(count, length - 410))
    arr = np.concatenate(
        [
            np.repeat(np.array(text[:202])[None], count, axis=0),
            random,
            np.repeat(np.array(text[-202:])[None], count, axis=0),
        ],
        axis=1,
    )
    return random, arr
