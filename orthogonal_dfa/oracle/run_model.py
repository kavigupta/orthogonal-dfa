import numpy as np
import torch
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.spliceai.exon_score import (
    assert_output_width,
    device_of,
    forward_batch,
    full_lengths,
    spliceai_exon_scores,
)


def batched_run(model, arr, batch_size=1024):
    device = device_of(model)
    return torch.cat(
        [
            forward_batch(model, arr[i : i + batch_size], device=device)
            for i in range(0, len(arr), batch_size)
        ],
        dim=0,
    )


def compute_exon_scores(model, arr, *, cl):
    """Exon scores for the full-length windows ``arr``, cut for a cl-``cl`` exon."""
    logits = batched_run(model, arr)
    # spliceai_exon_scores reads the lengths back off the output width, so its own
    # width check cannot fail here; the input width is what pins down the cl.
    assert_output_width(logits.shape[1], arr.shape[1] - cl)
    return spliceai_exon_scores(logits, full_lengths(logits))


def run_model(exon, model, arr):
    calibration = calibrate(exon, model, count=100_000)
    with torch.no_grad():
        yp = compute_exon_scores(model, arr, cl=exon.cl)
        normalized_target = (yp - calibration["mean"]) / calibration["std"]
        hard_target = normalized_target > 0
    return normalized_target, hard_target


@permacache(
    "orthogonal_dfa/oracle/run_model/create_dataset_just_output_2",
    # using legacy hashing here because this is just for the spliceai models
    key_function=dict(model=lambda model: stable_hash(model, version=1)),
    multiprocess_safe=True,
)
def create_dataset_just_output(exon, model, *, count, seed):
    _, arr = sample_text(exon, seed, count)
    _, hard_targets = run_model(exon, model, arr)
    arr = hard_targets.cpu().numpy()
    arr = np.packbits(arr)
    return arr


def create_dataset(exon, model, *, count, seed):
    random, arr = sample_text(exon, seed, count)
    arr = create_dataset_just_output(exon, model, count=count, seed=seed)
    hard_targets = torch.tensor(np.unpackbits(arr).astype(bool), device=random.device)
    return random, hard_targets


@permacache(
    "orthogonal_dfa/oracle/evaluate/calibrate",
    key_function=dict(exon=stable_hash, model=lambda m: stable_hash(m, version=2)),
)
def calibrate(exon: RawExon, model, count=100_000):
    """
    Returns the fraction of positive predictions on random data.

    :param exon: exon to sample from
    :param model: model to evaluate
    :param seed: random seed
    :param count: number of samples
    """
    _, arr = sample_text(exon, int(stable_hash("calibration"), 16), count)
    with torch.no_grad():
        yp = compute_exon_scores(model, arr, cl=exon.cl)
        median = yp.median()
        mean = yp.mean()
        std = yp.std()
    return dict(median=median.item(), mean=mean.item(), std=std.item())
