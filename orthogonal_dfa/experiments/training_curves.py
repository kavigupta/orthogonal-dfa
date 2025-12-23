import numpy as np
from matplotlib import pyplot as plt

from orthogonal_dfa.experiments.gate_experiments import train_psamdfa


def process_results(results):
    gates, train, results = zip(*results)
    return dict(
        gates=gates,
        last_gates=[g[-1] for g in gates],
        curves=-np.array(train)[:, 0, :, :].mean(-1),
        eva=[r[-1] for r in results],
    )


def pdfa_results(num_seeds, baselines, **kwargs):
    return process_results(
        train_psamdfa(baselines, seed=seed, **kwargs) for seed in range(num_seeds)
    )


def plot_training_curves(results, labels):
    size = 5
    ncols = 3

    assert len(results) == len(labels)

    _, axs = plt.subplots(
        (len(results) + ncols - 1) // ncols,
        ncols,
        figsize=(ncols * size, size * len(results) / ncols),
        dpi=200,
    )
    for ax, r, label in zip(
        axs.flatten(),
        results,
        labels,
    ):
        for seed, (x, result) in enumerate(zip(r["curves"], r["eva"])):
            ax.plot(
                x * 1000,
                label=f"Seed={seed}: eval={1000 * result:.0f}mb",
            )
        ax.set_ylim(0, ax.set_ylim()[1])
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Information [mb]")
        ax.set_title(label)
