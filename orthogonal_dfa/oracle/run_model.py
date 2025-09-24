import torch

from orthogonal_dfa.math import corr


def run_model(model, arr):
    xpack = torch.tensor(arr).cuda()
    x = torch.eye(4).cuda()[xpack]
    with torch.no_grad():
        yp = model(x).log_softmax(-1)
        yp = yp[:, [0, -1], [1, 2]].mean(-1)
        normalized_target = corr.normalize(yp)
        hard_target = yp > yp.median()
    return normalized_target, hard_target
