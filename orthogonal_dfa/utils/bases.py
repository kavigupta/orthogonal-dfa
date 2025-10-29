import torch

from orthogonal_dfa.utils.probability import ZeroProbability


def parse_nucleotides(seq: str) -> list[int]:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    return [mapping[nuc] for nuc in seq]


def parse_nucleotides_as_one_hot(seq: str) -> torch.Tensor:
    indices = parse_nucleotides(seq)
    lookup = torch.cat(
        [
            torch.eye(4),
            torch.zeros(1, 4),
        ],
        dim=0,
    )  # (5, 4)
    return lookup[indices]


def parse_nucleotides_as_one_hot_logit(
    seq: str, zero_prob: ZeroProbability
) -> torch.Tensor:
    one_hot = parse_nucleotides_as_one_hot(seq)
    one_hot[one_hot == 0] = zero_prob.logit_probability
    one_hot[one_hot == 1] = -zero_prob.logit_probability
    return one_hot
