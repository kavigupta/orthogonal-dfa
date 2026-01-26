import numpy as np
import torch
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.experiments.train_from_scratch import oracle
from orthogonal_dfa.experiments.train_gate import evaluate, train
from orthogonal_dfa.experiments.training_curves import models_by_sparsity
from orthogonal_dfa.manual_dfa.stop_codon_dfa import stop_codon_psamdfa
from orthogonal_dfa.module.residual_gate import InputMonotonicModelingGate
from orthogonal_dfa.utils.probability import ZeroProbability

no_orf = stop_codon_psamdfa("TAG", "TAA", "TGA", zero_prob=ZeroProbability(1e-7)).cuda()
no_orf_ta = stop_codon_psamdfa("TAG", "TAA", zero_prob=ZeroProbability(1e-7)).cuda()


def train_pdfa_wrt(test_dfa, control_gates):
    torch.manual_seed(0)
    gate, _ = train(
        InputMonotonicModelingGate(test_dfa, 5, 100).cuda(),
        1e-2,
        default_exon,
        oracle(),
        control_gates,
        epochs=10,
        batch_size=1000,
        train_count=100_000,
        seed=0,
        do_not_train_phi=True,
    )
    end, start = evaluate(
        default_exon, oracle(), control_gates, gate, count=100_000, seed=0
    )
    return start - end


@permacache(
    "orthogonal_dfa/experiments/pdfa_residual/train_pdfa_wrt_all",
    key_function=dict(
        test_dfa=lambda x: stable_hash(x, version=2),
        control_gates=lambda x: stable_hash(x, version=2),
    ),
)
def train_pdfa_wrt_all(test_dfa, control_gates):
    return np.array(
        [
            train_pdfa_wrt(test_dfa, control_gates[:i])
            for i in range(0, 1 + len(control_gates))
        ]
    )


def just_plot_last(dfa, all_gates):
    ys = [1000 * train_pdfa_wrt_all(dfa, gates)[-1] for gates in all_gates]
    return [len(gates) for gates in all_gates], ys


def plot_residuals(gates_mll, gates_psams, gates_psams_orig, results, names):
    """
    This is kind of a mess, very specialized to the specific inputs. We should rewrite this
    if we ever touch this again.
    """
    markers = [".", "*", "+"]

    assert len(results) == len(names) <= len(markers)

    plt.figure(dpi=200, figsize=(10, 10))
    for i, (dfa, label) in enumerate(zip((no_orf, no_orf_ta), ("no-ORF", "no-ORF-TA"))):
        plt.plot(
            1000 * train_pdfa_wrt_all(dfa, gates_mll),
            label=f"{label} | Linear",
            color=f"C{i}",
        )
        plt.plot(
            1000 * train_pdfa_wrt_all(dfa, gates_psams),
            label=f"{label} | PSAMs",
            color=f"C{i}",
            linestyle="--",
        )
        plt.plot(
            1000 * train_pdfa_wrt_all(dfa, gates_psams_orig),
            label=f"{label} | PSAMs [orig]",
            color=f"C{i}",
            linestyle="-.",
        )
        for r, n, m in zip(results, names, markers):
            plt.scatter(
                *just_plot_last(dfa, r["gates"]),
                label=f"{label} | {n}",
                color=f"C{i}",
                marker=m,
            )

    # plt.axhline(1000 * )
    plt.ylim(0, plt.ylim()[1])
    plt.legend()
    plt.xlabel("Number controls")
    plt.ylabel("Relative marginal entropy [mb]")
    plt.grid()
    plt.show()


def do_plot_last(dfa, r, n):
    plt.scatter(
        just_plot_last(dfa, r["gates"])[-1],
        1000 * np.array(r["eva"]),
        label=f"no-ORF | {n}",
    )
    plt.xlabel("Remaining signal after controlling for learned PDFA [mb]")
    plt.ylabel("Learned PDFA signal [mb]")
    plt.legend()


def compute_original_and_residual_by_sparsity(dfa, model):
    model, _ = train(
        model,
        lr=1e-5,
        exon=default_exon,
        oracle=oracle(),
        prev_gates=[],
        epochs=5,
        train_count=10_000,
        batch_size=1000,
        new_data_every_epoch=None,
        seed=int(stable_hash(("finetune", stable_hash(model, version=2))), 16)
        % (2**32 - 1),
        do_not_train_phi=True,
        notify_epoch_loss=False,
    )
    original_signal = evaluate(default_exon, oracle(), [], model, count=100_000, seed=0)
    original_signal = original_signal[1] - original_signal[0]
    no_orf_signal = train_pdfa_wrt_all(dfa, [model.eval()])[-1]
    return original_signal, no_orf_signal


def compute_all_original_and_residual_by_sparsity(results):
    by_sparsity = models_by_sparsity(results)
    return {
        k: np.array([compute_original_and_residual_by_sparsity(no_orf, v) for v in vs])
        for k, vs in tqdm.tqdm(by_sparsity.items())
    }


def plot_residual_results_by_sparsity(original_and_residual_by_sparsity):
    plt.figure(dpi=200, figsize=(10, 10))
    means = []
    for k, vs in original_and_residual_by_sparsity.items():
        vs = 1000 * vs
        y, x = vs.mean(0)
        means.append((x, y))
        pts = plt.scatter(vs[:, 1], vs[:, 0], marker=".")
        plt.text(x=x, y=y, s=f"{1-k:.2%}", color=pts._facecolors[0])
    means = np.array(means)
    plt.plot(means[:, 0], means[:, 1], color="black")
    plt.xlabel("Remaining signal in no-ORF | RNN")
    plt.ylabel("Signal in RNN")
    plt.grid()
