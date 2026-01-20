import numpy as np
import torch

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.experiments.train_from_scratch import oracle
from orthogonal_dfa.oracle.run_model import create_dataset


def compute_sparse_results(sm, *, count=10_000, seed, batch_size=1000):
    assert not sm.training
    device = next(sm.parameters()).device
    data, _ = create_dataset(default_exon, oracle(), count=count, seed=seed)
    results = []
    for i in range(0, len(data), batch_size):
        xbatch = torch.tensor(data[i : i + batch_size], device=device)
        x = torch.eye(4, device=device)[xbatch]
        with torch.no_grad():
            results.append(sm.psams_forward(x))
    return data, torch.cat(results, dim=0).cpu().numpy()


def get_sequences(x, m, channel):
    length = x.shape[1] - m.shape[1] + 1
    assert length >= 1
    batch, seq = np.where(m[..., channel])
    batch, seq = batch[:, None], seq[:, None]
    seq = seq + np.arange(length)
    return x[batch, seq]


def get_ngrams(sequences, k):
    length = sequences.shape[1]
    return np.concatenate([sequences[:, i : i + k] for i in range(length - k + 1)])


def get_all_logos(x, m):
    psams = []
    for channel in range(m.shape[2]):
        seqs = get_sequences(x, m, channel)
        psams.append(np.eye(4)[seqs].mean(0))
    return np.array(psams)


def get_ngram_enrichments_for_channel(x, m, channel, k):
    sequences = get_ngrams(get_sequences(x, m, channel), k)
    packed = to_packed(sequences)
    return np.bincount(packed, minlength=4**k) / packed.shape[0] * 4**k


def get_all_ngram_enrichments(x, m, k):
    enrichments = []
    for channel in range(m.shape[2]):
        enrichments.append(get_ngram_enrichments_for_channel(x, m, channel, k))
    return np.array(enrichments)


def to_packed(sequences):
    """
    sequences: (..., k); where elements are integers from 0 to 3

    output: (...), where elements are integers from 0 to 4 ** k - 1
    """
    return sequences @ (4 ** np.arange(sequences.shape[-1]))


def to_unpacked(packed, k):
    """
    packed: (...), where elements are integers from 0 to 4 ** k - 1

    output: (..., k); where elements are integers from 0 to 3
    """
    return packed[:, None] // (4 ** np.arange(k)) % 4
