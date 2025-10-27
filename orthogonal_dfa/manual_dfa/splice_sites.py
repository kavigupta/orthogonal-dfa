import numpy as np


def construct_kmers(count):
    return construct_kmers_unreshaped(count).reshape(count, -1).T


def construct_kmers_unreshaped(count):
    return np.array(np.meshgrid(*((np.arange(4),) * count), indexing="ij"))


def count_kmers_in_subset(sequence_length, sequence_indices, k):
    """
    Given sequences of length `sequence_length` indexed by `sequence_indices`,
    (i.e., the sequences are construct_kmers(sequence_length)[sequence_indices]),
    count the occurrences of each k-mer in the sequences.
    """
    counts = np.zeros((4**k,), dtype=int)
    np.add.at(
        counts,
        (
            (sequence_indices[:, None] // 4 ** np.arange(sequence_length - k + 1))
            % 4**k
        ).flatten(),
        1,
    )
    return counts
