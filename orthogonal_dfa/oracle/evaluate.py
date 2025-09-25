import numpy as np
import pythomata
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.oracle.run_model import run_model
from orthogonal_dfa.utils.dfa import hash_dfa

TEST_SEED = int(stable_hash("testing"), 16)


@permacache(
    "orthogonal_dfa/oracle/evaluate/evaluate_hard_dfa", key_function=dict(dfa=hash_dfa)
)
def evaluate_hard_dfa(
    exon: RawExon, dfa: pythomata.SimpleDFA, model, count=100_000, seed=TEST_SEED
):
    """
    Returns confusion matrix indexed as
        confusion[predicted][actual]
    where predicted and actual are in {0, 1)

    :param exon: exon to sample from
    :param dfa: DFA to evaluate
    :param model: model to evaluate
    :param seed: random seed
    :param count: number of samples
    """
    assert set(dfa.alphabet) == {
        0,
        1,
        2,
        3,
    }, f"DFA alphabet is {set(dfa.alphabet)}; expected {{0,1,2,3}}"
    random, arr = sample_text(exon, seed, count)
    _, hard_target = run_model(model, arr)
    hard_target = hard_target.cpu().numpy()
    hard_pred = np.array([dfa.accepts(list(x)) for x in random])
    return np.array(
        [
            [((hard_target == i) & (hard_pred == j)).sum() for i in (False, True)]
            for j in (False, True)
        ]
    )


def mutual_information(confusion):
    confusion = confusion / confusion.sum((-1, -2), keepdims=True)
    p_pred = confusion.sum(-1)
    p_actual = confusion.sum(-2)
    p_joint = confusion
    mi = np.zeros(p_pred.shape[:-1])
    for i in (0, 1):
        for j in (0, 1):
            p_joint_ij = p_joint[..., i, j]
            p_pred_i = p_pred[..., i]
            p_actual_j = p_actual[..., j]
            mask = p_joint_ij > 0
            mi[mask] += p_joint_ij[mask] * np.log(
                p_joint_ij[mask] / (p_pred_i[mask] * p_actual_j[mask])
            )
    return mi / np.log(2)


def actual_pct_difference_by_prediction(confusion):
    perc_actual_by_predicted = confusion[..., :, 1] / (
        confusion[..., :, 0] + confusion[..., :, 1]
    )
    return perc_actual_by_predicted[..., 1] - perc_actual_by_predicted[..., 0]


def simulate_bootstrap_confusion(fn, confusion, *, interval_pct=95, n=10000):
    total = confusion.sum()
    p = confusion / total
    sims = np.random.default_rng(0).multinomial(total, p.flatten(), size=n)
    sims = sims.reshape((n, 2, 2))
    tail = (100 - interval_pct) / 2
    return np.percentile(fn(sims), [tail, 100 - tail])


def print_with_uncertainty(confusion, fn, name, fmt):
    mi = fn(confusion)
    lb, ub = simulate_bootstrap_confusion(fn, confusion)
    print(f"{name}:\n    {fmt(mi)} ({fmt(lb)}, {fmt(ub)})")


def print_metrics(confusion):
    # these are not f-stringable because there's no arguments.
    # pylint: disable=consider-using-f-string
    print_with_uncertainty(
        confusion, mutual_information, "Mutual Information", "{:.4f}b".format
    )
    print_with_uncertainty(
        confusion,
        actual_pct_difference_by_prediction,
        "Actual % Difference by Prediction",
        "{:.2%}".format,
    )
