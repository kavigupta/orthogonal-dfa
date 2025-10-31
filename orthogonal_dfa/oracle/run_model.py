import torch
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text


def batched_run(model, arr, batch_size=1024):
    ys = []
    for i in range(0, len(arr), batch_size):
        xpack = torch.tensor(arr[i : i + batch_size]).cuda()
        x = torch.eye(4).cuda()[xpack]

        with torch.no_grad():
            ys.append(model(x))
    return torch.cat(ys, dim=0)


def compute_exon_scores(model, arr):
    yp = batched_run(model, arr).log_softmax(-1)
    yp = yp[:, [0, -1], [1, 2]].mean(-1)
    return yp


def run_model(exon, model, arr):
    calibration = calibrate(exon, model, count=100_000)
    with torch.no_grad():
        yp = compute_exon_scores(model, arr)
        normalized_target = (yp - calibration["mean"]) / calibration["std"]
        hard_target = normalized_target > 0
    return normalized_target, hard_target


@permacache(
    "orthogonal_dfa/oracle/run_model/create_dataset_2",
    key_function=dict(model=stable_hash),
)
def create_dataset(exon, model, *, count, seed):
    random, arr = sample_text(exon, seed, count)
    _, hard_targets = run_model(exon, model, arr)
    return random, hard_targets


@permacache("orthogonal_dfa/oracle/evaluate/calibrate")
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
        yp = compute_exon_scores(model, arr)
        median = yp.median()
        mean = yp.mean()
        std = yp.std()
    return dict(median=median.item(), mean=mean.item(), std=std.item())
