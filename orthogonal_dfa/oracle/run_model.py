import torch

from orthogonal_dfa.math import corr


def batched_run(model, arr, batch_size=1024):
    ys = []
    for i in range(0, len(arr), batch_size):
        xpack = torch.tensor(arr[i : i + batch_size]).cuda()
        x = torch.eye(4).cuda()[xpack]

        with torch.no_grad():
            ys.append(model(x))
    return torch.cat(ys, dim=0)


def run_model(model, arr):
    with torch.no_grad():
        yp = batched_run(model, arr).log_softmax(-1)
        yp = yp[:, [0, -1], [1, 2]].mean(-1)
        normalized_target = corr.normalize(yp)
        hard_target = yp > yp.median()
    return normalized_target, hard_target
