import numpy as np

from .exon import RawExon


def sample_text(exon: RawExon, seed, count):
    trim_zone = exon.cl // 2 + 2
    length = len(exon.text)
    assert length > trim_zone * 2
    random = np.random.default_rng(seed).choice(4, size=(count, length - trim_zone * 2))
    assert random.shape[1] == exon.random_text_length
    arr = np.concatenate(
        [
            np.repeat(np.array(exon.text[:trim_zone])[None], count, axis=0),
            random,
            np.repeat(np.array(exon.text[-trim_zone:])[None], count, axis=0),
        ],
        axis=1,
    )
    return random, arr
