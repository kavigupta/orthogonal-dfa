import numpy as np
import torch
from matplotlib import pyplot as plt
from permacache import permacache, stable_hash

from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.experiments.train_from_scratch import oracle
from orthogonal_dfa.experiments.train_gate import evaluate, train
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
