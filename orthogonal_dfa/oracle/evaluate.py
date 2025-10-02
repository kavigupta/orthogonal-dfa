import numpy as np
import torch
from permacache import stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.oracle.run_model import run_model
from orthogonal_dfa.utils.dfa import TorchDFA

TEST_SEED = int(stable_hash("testing"), 16)
DEMO_SEED = int(stable_hash("demo"), 16)


def multidimensional_confusion(
    exon: RawExon,
    dfas_to_test: TorchDFA,
    dfas_prev: TorchDFA,
    model,
    count=100_000,
    *,
    seed,
):
    """
    Returns confusion matrix indexed as
        confusion[dfa_to_test_idx][dfa_to_test_val][dfas_prev_id][actual]
    where predicted and actual are in {0, 1)

    :param exon: exon to sample from
    :param dfa: DFA to evaluate
    :param model: model to evaluate
    :param seed: random seed
    :param count: number of samples
    """
    assert len(dfas_prev) <= 32, "Too many previous DFAs to meaningfully track"
    model_res, dfa_res = run_model_and_dfas(
        exon, TorchDFA.concat(dfas_to_test, dfas_prev), model, count, seed
    )
    dfa_to_test_res = dfa_res[: len(dfas_to_test)]
    dfa_prev_res = dfa_res[len(dfas_to_test) :]
    dfa_prev_res = pack_as_uint32(dfa_prev_res)
    confusion = np.zeros((len(dfas_to_test), 2, 2 ** len(dfas_prev), 2), dtype=np.int64)
    np.add.at(
        confusion,
        (
            np.arange(len(dfas_to_test))[:, None],
            dfa_to_test_res.astype(np.uint8),
            dfa_prev_res[None, :],
            model_res.astype(np.uint8)[None, :],
        ),
        1,
    )
    return confusion


def pack_as_uint32(values):
    values = np.packbits(values, axis=0, bitorder="little").astype(np.uint32)
    # return values
    values = (2 ** (8 * np.arange(len(values), dtype=np.uint32))) @ values
    return values


def run_model_and_dfas(exon, dfas, model, count, seed):
    random, arr = sample_text(exon, seed, count)
    _, hard_target = run_model(exon, model, arr)
    hard_target = hard_target.cpu().numpy()
    hard_pred = dfas.cuda()(torch.tensor(random).cuda()).cpu().numpy()
    return hard_target, hard_pred


def conditional_mutual_information(confusions):
    assert (
        confusions.ndim == 4
    ), f"Expected confusions to have 4 dimensions, got {confusions.shape}"
    # confusions is indexed as [batch][phi_i_val][control_val][actual_val]
    # we are computing I(phi_i; actual | control)
    # this is computed as sum_control p(control) * I(phi_i; actual | control=control_val)
    confusions = confusions.transpose(
        0, 2, 1, 3
    )  # [batch][control_val][phi_i_val][actual_val]
    count_per_control = confusions.sum(axis=(2, 3))  # [batch, control_val]
    informations = mutual_information(confusions)  # [batch * control_val]
    probabilities = count_per_control / count_per_control.sum(
        axis=1, keepdims=True
    )  # [batch][control_val]
    return (probabilities * informations).sum(axis=1)  # [batch]


def mutual_information(confusion):
    """
    :param confusion: np.ndarray of shape (..., 2, 2) with counts
    :return: np.ndarray of shape (...) with mutual information in bits
    """
    total = confusion.sum(axis=(-2, -1), keepdims=True)
    p_xy = confusion / total
    p_x = p_xy.sum(axis=-1, keepdims=True)
    p_y = p_xy.sum(axis=-2, keepdims=True)
    log_pxy = np.log2(np.clip(p_xy, 1e-100, None))
    mi = (p_xy * (log_pxy - np.log2(p_x) - np.log2(p_y))).sum(axis=(-2, -1))
    return mi


def actual_pct_difference_by_prediction(confusion):
    # confusion[batch][phi_i_val][control_val][actual_val]
    # we are computing E[|P(actual=1 | phi_i=1, control) - P(actual=1 | phi_i=0, control)|]
    p_control = confusion.sum(axis=(-2, -1))  # [batch][control_val]
    p_control = p_control / p_control.sum(axis=1, keepdims=True)  # [batch][control_val]
    confusion = confusion.transpose(
        0, 2, 1, 3
    )  # [batch][control_val][phi_i_val][actual_val]
    p_actual_given_phi_and_control = confusion[..., 1] / confusion.sum(
        axis=-1
    )  # [batch][control_val][phi_i_val]
    pct_diffs = (
        p_actual_given_phi_and_control[..., 1] - p_actual_given_phi_and_control[..., 0]
    )  # [batch][control_val]
    return (pct_diffs * p_control).sum(axis=1)  # [batch]


def simulate_bootstrap_confusion(fn, confusion, *, interval_pct=95, n=10000):
    total = confusion.sum()
    p = confusion / total
    sims = np.random.default_rng(0).multinomial(total, p.flatten(), size=n)
    sims = sims.reshape((n, *confusion.shape))
    tail = (100 - interval_pct) / 2
    return np.percentile(fn(sims), [tail, 100 - tail])


def print_with_uncertainty(confusion, fn, name, fmt):
    [mi] = fn(confusion[None, ...])
    lb, ub = simulate_bootstrap_confusion(fn, confusion)
    print(f"{name}:\n    {fmt(mi)} ({fmt(lb)}, {fmt(ub)})")


def print_metrics(confusion):
    # these are not f-stringable because there's no arguments.
    # pylint: disable=consider-using-f-string
    print_with_uncertainty(
        confusion,
        conditional_mutual_information,
        "Mutual Information",
        "{:.4f}b".format,
    )
    print_with_uncertainty(
        confusion,
        actual_pct_difference_by_prediction,
        "Actual % Difference by Prediction",
        "{:.2%}".format,
    )
