from matplotlib import pyplot as plt

from orthogonal_dfa.psams.data import SpliceDataLoader
from orthogonal_dfa.psams.train import train_psams
from orthogonal_dfa.spliceai.evaluate_lssi import (
    evaluate_3prime_psams,
    evaluate_5prime_psams,
)


def train_and_evaluate_lssi_psams(which, *, num_batches, **kwargs):
    accuracies = {}
    losses = {}
    for num_psams in 1, 2, 4, 8, 16, 32, 64, 128:
        accuracies[num_psams] = []
        losses[num_psams] = []
        for seed in range(10):
            evaluate = {
                "donor": evaluate_5prime_psams,
                "acceptor": evaluate_3prime_psams,
            }[which]
            psams, loss = train_psams_for_splice_site(
                which, num_psams, num_batches=num_batches, seed=seed, **kwargs
            )
            accuracies[num_psams].append(100 * evaluate(psams.eval()))
            losses[num_psams].append(loss)
    return dict(accuracies=accuracies, losses=losses)


def train_psams_for_splice_site(which, num_psams, *, num_batches, seed, **kwargs):
    two_r = {"donor": 8, "acceptor": 22}[which]
    psams, loss = train_psams(
        two_r=two_r,
        num_psams=num_psams,
        seed=seed,
        data_loader=SpliceDataLoader(which, 1, 10_000, two_r + 1),
        num_batches=num_batches,
        **kwargs,
    )

    return psams, loss


def plot_accuracies(result, lssi_perf, maxent_perf, label, *, ax=None):
    accuracies = result["accuracies"]
    if ax is None:
        ax = plt.gca()
    ax.axhline(lssi_perf, label="LSSI", color="black")
    ax.axhline(maxent_perf, label="MaxEntScan", color="green")
    for it, (x, ys) in enumerate(accuracies.items()):
        ax.scatter(
            [x] * len(ys),
            ys,
            color="blue",
            alpha=0.5,
            label="PSAMs" if it == 0 else None,
        )
    ax.set_xlabel("Num PSAMs")
    ax.set_xscale("log")
    ax.set_xticks(list(accuracies))
    ax.set_xticklabels(list(accuracies))
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.legend()
    ax.set_title(label)


def accuracy_vs_loss(result):
    accuracies = result["accuracies"]
    losses = result["losses"]
    for k in accuracies:
        accs = accuracies[k]
        loss_vals = [min(loss) for loss in losses[k]]
        plt.scatter(loss_vals, accs, label=f"{k} PSAMs")
    plt.xlabel("Final Training Loss")
    plt.ylabel("LSSI Accuracy (%)")
    plt.legend()
