from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import pythomata
import torch
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import RawExon
from orthogonal_dfa.data.sample_text import sample_text
from orthogonal_dfa.oracle.run_model import create_dataset, run_model
from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA
from orthogonal_dfa.psams.psams import flip_log_probs
from orthogonal_dfa.utils.dfa import TorchDFA, hash_dfa

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
    return multidimensional_confusion_from_results(
        model_res, dfa_to_test_res, dfa_prev_res
    )


def multidimensional_confusion_from_results(model_res, dfa_to_test_res, dfa_prev_res):
    """
    :param model_res: np.ndarray of shape (num_samples,) with model results
    :param dfa_to_test_res: np.ndarray of shape (num_dfas_to_test, num_samples) with DFA results
    :param dfa_prev_res: np.ndarray of shape (num_dfas_prev, num_samples) with DFA results
    :return: np.ndarray of shape (num_dfas_to_test, 2, 2**num_dfas_prev, 2) with confusion matrix
    """
    num_dfas_test = dfa_to_test_res.shape[0]
    num_dfas_prev = dfa_prev_res.shape[0]
    dfa_prev_res = pack_as_uint32(dfa_prev_res)
    confusion = np.zeros((num_dfas_test, 2, 2**num_dfas_prev, 2), dtype=np.int64)
    np.add.at(
        confusion,
        (
            np.arange(num_dfas_test)[:, None],
            dfa_to_test_res.astype(np.uint8),
            dfa_prev_res[None, :],
            model_res.astype(np.uint8)[None, :],
        ),
        1,
    )
    return confusion


def multidimensional_confusion_from_proabilistic_results(
    model_res, dfa_to_test_res, dfa_prev_res
):
    """
    Like multidimensional_confusion_from_results but where `model_res`, `dfa_to_test_res`, and `dfa_prev_res`
        all contain log probabilities of acceptance rather than hard 0/1 values. Also, operates over torch
        tensors rather than numpy arrays.

    The output confusion matrix contains log probabilities rather than counts.

    Unfortunately, this is somewhat memory inefficient, as it constructs the entire sub-confusion log-matrix for
    each sample, so might need to be batched in the future.

    :param model_res: (N,) tensor of log-probabilities of acceptance from the model
    :param dfa_to_test_res: (num_dfas_to_test, N) tensor of log-probabilities of acceptance from the DFAs to test
    :param dfa_prev_res: list of (N,) tensors of log-probabilities of acceptance from the previous DFAs
    :return: (num_dfas_to_test, 2, *[2, 2, ..., 2], 2) tensor of log-probabilities in the confusion matrix
    """
    num_samples = model_res.shape[0]

    # At the end, we should have
    # confusion_each[sample, dfa_to_test_idx, dfa_to_test_val, *dfa_prev_val, actual_val]
    confusion_each = torch.stack(
        [flip_log_probs(dfa_to_test_res).T, dfa_to_test_res.T], dim=2
    )

    def add_confusion(data):
        nonlocal confusion_each
        assert data.shape == (num_samples,)
        data = torch.stack([flip_log_probs(data), data], dim=1)
        for _ in range(len(confusion_each.shape) - 1):
            data = data[:, None, :]
        confusion_each = confusion_each[..., None]
        assert len(data.shape) == len(confusion_each.shape)
        confusion_each = confusion_each + data

    for res in dfa_prev_res:
        add_confusion(res)
    add_confusion(model_res)
    return torch.logsumexp(confusion_each, dim=0) - np.log(num_samples)


def mutual_information_from_log_confusion(log_confusion):
    """
    Like mutual_information but where `log_confusion` contains log-probabilities rather than counts.

    :param log_confusion: torch.tensor of shape (..., 2, 2) with log-probabilities
    :return: torch.tensor of shape (...) with mutual information in bits
    """
    total = torch.logsumexp(log_confusion, dim=(-2, -1), keepdim=True)
    log_p_xy = log_confusion - total
    log_p_x = torch.logsumexp(log_p_xy, dim=-1, keepdim=True)
    log_p_y = torch.logsumexp(log_p_xy, dim=-2, keepdim=True)
    mi = torch.exp(log_p_xy) * (log_p_xy - log_p_x - log_p_y)
    mi = mi.sum(dim=(-2, -1)) / np.log(2)
    return mi


def conditional_mutual_information_from_log_confusion(log_confusion):
    """
    :param log_confusion: torch.tensor of shape (batch, control_val, ...phi_i_val, actual_val) with log-probabilities
    :return: torch.tensor of shape (batch,) with conditional mutual information in bits
    """
    is_numpy = isinstance(log_confusion, np.ndarray)
    if is_numpy:
        log_confusion = torch.tensor(log_confusion)
    # flatten all phi_i_val dimensions into one
    batch_size, control_size = log_confusion.shape[:2]
    phi_dims = log_confusion.shape[2:-1]
    log_confusion = log_confusion.reshape(
        batch_size, control_size, np.prod(phi_dims, dtype=int), 2
    )
    # move control dimension to the end for easier processing
    log_confusion = log_confusion.permute(
        0, 2, 1, 3
    )  # [batch, phi_i_val, control_val, actual_val]
    mi = mutual_information_from_log_confusion(log_confusion)
    probs_each = torch.logsumexp(log_confusion, dim=(-2, -1))  # [batch, phi_i_val]
    result = (mi * torch.exp(probs_each)).sum(-1)  # [batch]
    if is_numpy:
        result = result.numpy()
    return result


@permacache(
    "orthogonal_dfa/oracle/evaluate/evaluate_pdfas_5",
    key_function=dict(
        dfas_to_test=lambda x: stable_hash(x, version=2),
        dfas_control=lambda x: stable_hash(x, version=2),
        model=lambda x: stable_hash(x, version=2),
    ),
)
def evaluate_pdfas(
    exon: RawExon,
    dfas_to_test: PSAMPDFA,
    dfas_control: List[PSAMPDFA],
    model,
    count=100_000,
    *,
    seed,
):
    random, hard_target = create_dataset(exon, model, count=count, seed=seed)
    random = torch.eye(4)[random].cuda()
    with torch.no_grad():
        predictions_to_test = batch_run(dfas_to_test, random)
    with torch.no_grad():
        predictions_control = [batch_run(dfa, random) for dfa in dfas_control]
        assert all(len(x) == 1 for x in predictions_control)
        predictions_control = [x[0] for x in predictions_control]
    result = multidimensional_confusion_from_proabilistic_results(
        torch.clamp(hard_target.float(), min=1e-7).log().cuda(),
        predictions_to_test,
        predictions_control,
    )
    return result.cpu().numpy()


def batch_run(m, x, batch_size=1_000):
    return torch.cat(
        [m(x[i : i + batch_size]) for i in range(0, x.shape[0], batch_size)], dim=1
    )


@permacache(
    "orthogonal_dfa/oracle/evaluate/evaluate_dfas",
    key_function=dict(
        dfas_to_test=hash_dfa, dfas_control=lambda x: tuple(hash_dfa(d) for d in x)
    ),
    parallel=["dfas_to_test"],
)
def evaluate_dfas(
    exon: RawExon,
    dfas_to_test: List[pythomata.SimpleDFA],
    dfas_control: List[pythomata.SimpleDFA],
    model,
    count=100_000,
    *,
    seed,
):
    dfas_to_test = TorchDFA.concat(
        *[TorchDFA.from_pythomata(d) for d in dfas_to_test],
    )
    dfas_control = TorchDFA.concat(
        *[TorchDFA.from_pythomata(d) for d in dfas_control],
        num_symbols=dfas_to_test.alphabet_size,
    )
    results = multidimensional_confusion(
        exon,
        dfas_to_test,
        dfas_control,
        model,
        count=count,
        seed=seed,
    )
    return list(results)


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
    log_px = np.log2(np.clip(p_x, 1e-100, None))
    log_py = np.log2(np.clip(p_y, 1e-100, None))
    mi = (p_xy * (log_pxy - log_px - log_py)).sum(axis=(-2, -1))
    return mi


def actual_pct_difference_by_prediction(confusion):
    # confusion[batch][phi_i_val][control_val][actual_val]
    # we are computing E[|P(actual=1 | phi_i=1, control) - P(actual=1 | phi_i=0, control)|]
    p_control = confusion.sum(axis=(-2, -1))  # [batch][control_val]
    p_control = p_control / (
        1e-100 + p_control.sum(axis=1, keepdims=True)
    )  # [batch][control_val]
    confusion = confusion.transpose(
        0, 2, 1, 3
    )  # [batch][control_val][phi_i_val][actual_val]
    p_actual_given_phi_and_control = confusion[..., 1] / (
        1e-100 + confusion.sum(axis=-1)
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


class Metric(ABC):
    @abstractmethod
    def __call__(self, confusion: np.ndarray) -> np.ndarray: ...


@dataclass
class ConditionalMutualInformation(Metric):
    def __call__(self, confusion: np.ndarray) -> np.ndarray:
        return conditional_mutual_information(confusion)

    @property
    def name(self):
        return "Conditional Mutual Information [b]"
