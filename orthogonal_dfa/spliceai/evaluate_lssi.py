import h5py
import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import permacache, stable_hash


@permacache(
    "orthogonal_dfa/spliceai/evaluate_lssi/evaluate_lssi",
    key_function=dict(model=stable_hash),
)
def evaluate_lssi(model, padding, which_channel):
    batch_size = 128
    assert not model.training
    assert which_channel in (1, 2)
    with h5py.File(
        "/mnt/md0/ExpeditionsCommon/spliceai/Canonical/dataset_test_0.h5"
    ) as f:
        yps, targets = [], []
        for i in tqdm.trange(len(f) // 2):
            x, [y] = f[f"X{i}"][:], f[f"Y{i}"][:]
            x[x.sum(-1) == 0] = 1  # make the PSAMs negative
            for i in range(0, x.shape[0], batch_size):
                yp, target = compute_yp_and_target(
                    model,
                    padding,
                    which_channel,
                    x[i : i + batch_size],
                    y[i : i + batch_size],
                )
                yps.append(yp)
                targets.append(target)
    yps, targets = np.concatenate(yps), np.concatenate(targets)
    thresh = np.quantile(yps, 1 - targets.mean())
    return ((yps > thresh) & targets).sum() / targets.sum()


def compute_yp_and_target(model, padding, which_channel, x, y):
    remove = (x.shape[1] - y.shape[1]) // 2
    assert remove > 0
    with torch.no_grad():
        x = torch.tensor(x).float()
        if next(model.parameters()).is_cuda:
            x = x.cuda()
        yp = model(x)
    yp = (
        torch.nn.functional.pad(yp, padding, value=-np.inf)[:, remove:-remove]
        .cpu()
        .numpy()
    )
    target = y[:, :, which_channel] == 1
    return yp, target


def evaluate_5prime_psams(psams):
    return evaluate_lssi(psams, padding=(2, 6), which_channel=2)


def evaluate_3prime_psams(psams):
    return evaluate_lssi(psams, padding=(20, 2), which_channel=1)
