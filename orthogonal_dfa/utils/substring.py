import numpy as np


class SubstringDataProvider:
    def __init__(self, x, y, *, size_each, alphabet_size, meta_seed):
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y
        self.alphabet_size = alphabet_size
        self.size_each = size_each
        self.meta_seed = meta_seed
        assert x.shape[0] >= size_each

    def sample(self, seq_len, *, seed):
        rng = np.random.default_rng((self.meta_seed, seed))
        idxs = rng.choice(len(self.x), size=(self.size_each,), replace=False)
        core = self.x[idxs]
        labels = self.y[idxs]
        data = extend_randomly_to_length(
            core, seq_len, alphabet_size=self.alphabet_size, rng=rng
        )
        return data, labels


def extend_randomly_to_length(core, seq_len, *, alphabet_size, rng):
    assert seq_len >= core.shape[1]
    data = rng.choice(alphabet_size, size=(core.shape[0], seq_len))
    idxs = rng.choice(1 + seq_len - core.shape[1], size=(core.shape[0]))
    idxs = idxs[:, None] + np.arange(core.shape[1])
    data[np.arange(data.shape[0])[:, None], idxs] = core
    return data
