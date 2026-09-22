from functools import lru_cache

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.base import defaultdict

from orthogonal_dfa.baseline import MonolithicLinearLayer, PSAMsFollowedByLinear
from orthogonal_dfa.data.exon import default_exon
from orthogonal_dfa.psams.psam_pdfa import PSAMPDFA, PSAMPDFAWithTemperature
from orthogonal_dfa.psams.train import (
    identify_first_best_by_validation,
    train_psam_pdfa_full_learning_curve,
)
from orthogonal_dfa.spliceai.load_model import load_spliceai


@lru_cache(None)
def oracle():
    return load_spliceai(400, 0).cuda()


def train_model_from_spec(
    seed, lr, construct_model, *, epochs=100, baselines=(), batch_size=1000
):
    # print(seed, lr, construct_model)
    torch.random.manual_seed(seed)
    m = construct_model()
    val_every = 10
    ms, meta = train_psam_pdfa_full_learning_curve(
        default_exon,
        oracle(),
        m,
        baselines,
        seed=1,
        epochs=epochs,
        val_every=val_every,
        lr=lr,
        train_dataset_size=100_000,
        batch_size=batch_size,
    )

    ms_selected = ms[val_every - 1 :: val_every]
    if len(ms_selected) != len(meta["val_loss"]):
        assert len(ms_selected) + 1 == len(meta["val_loss"])
        ms_selected.append(ms[-1])

    return ms_selected, meta


def train_model(seed, num_psams, lr, with_temperature=None, **kwargs):
    def construct_model():
        m = PSAMPDFA.create(
            num_input_channels=4, num_psams=num_psams, two_r=8, num_states=4
        )
        if with_temperature is not None:
            m = PSAMPDFAWithTemperature(m, with_temperature)
        return m

    return train_model_from_spec(seed, lr, construct_model, **kwargs)


def plot_curve_generic(meta, label, window=100, models=None):
    # use a new color if and only if it's not reflected in preceding plots
    preceding_labels = [
        line for line in plt.gca().get_lines() if line.get_label() == label
    ]
    assert len(preceding_labels) <= 1
    if preceding_labels:
        label = None
        color = preceding_labels[0].get_color()
    else:
        color = None
    tl = meta["train_loss"]
    plt.plot(
        np.arange(window - 1, len(tl)),
        np.convolve(tl, np.ones(window) / window, mode="valid"),
        label=label,
        color=color,
    )
    plt.scatter(meta["val_loss_epochs"], meta["val_loss"], color=color)

    best_idx = identify_first_best_by_validation(meta["val_loss"], tolerance=0.1e-2)
    assert len(models) == len(
        meta["val_loss"]
    ), f"Expected {len(meta['val_loss'])} models, got {len(models)}"
    return models[best_idx] if models is not None else None, meta["val_loss"][best_idx]


def plot_curve(seed, num_psams, lr, with_temperature=None, **kwargs):
    models, meta = train_model(
        seed,
        num_psams,
        lr=lr,
        with_temperature=with_temperature,
        **kwargs,
    )
    flags = []
    if with_temperature is not None:
        flags.append("temp")
    flags = "" if not flags else f" [{','.join(flags)}]"
    label = f"{num_psams=}; {lr=:.0e}{flags}"
    return plot_curve_generic(meta, label, models=models)


def plot_curve_baseline(seed, lr, constructor, typ, epochs=120):
    models, meta = train_model_from_spec(
        seed,
        lr,
        lambda: constructor(4, default_exon.random_text_length),
        epochs=epochs,
    )
    # print(sum(meta["train_loss"]))
    return plot_curve_generic(meta, f"{typ} [{lr=:.0e}]", models=models)


def plots_for_pdfa():
    results = defaultdict(list)
    results["pdfa_4_1e-2"] = [plot_curve(0, 4, 1e-2), plot_curve(1, 4, 1e-2)]
    results["pdfa_4_1e-1_wt"] = [plot_curve(0, 4, 1e-1, with_temperature=1)]
    for i in range(3):
        results["pdfa_4_5e-3_wt"].append(
            plot_curve(i, 4, 5e-3, with_temperature=1, epochs=200)
        )
        results["pdfa_4_1e-2_wt"].append(
            plot_curve(i, 4, 1e-2, with_temperature=1, epochs=200)
        )
    return dict(results.items())


def plots_for_baseline():
    psams = lambda num_psams, width: lambda c, il: PSAMsFollowedByLinear(
        c, num_psams, width - 1, il
    )

    results = defaultdict(list)

    for seed in range(5):
        results["4x9"].append(
            plot_curve_baseline(seed, 1e-3, psams(4, 9), "psams [4x9]", epochs=500)
        )
        results["10x9"].append(
            plot_curve_baseline(seed, 1e-3, psams(10, 9), "psams [10x9]", epochs=500)
        )
        results["100x9"].append(
            plot_curve_baseline(seed, 1e-3, psams(100, 9), "psams [100x9]", epochs=500)
        )
    for seed in range(3):

        results["linear_1e-2"].append(
            plot_curve_baseline(seed, 1e-2, MonolithicLinearLayer, "linear")
        )
        results["linear_1e-3"].append(
            plot_curve_baseline(seed, 1e-3, MonolithicLinearLayer, "linear")
        )

    return dict(results.items())


def plot_best_validations_by_model(results):
    """
    Plot for each result the best validation losses by seed, as a scatterplot, where x is model type
    and y is best validation loss, for each seed.
    """
    labels = list(results.keys())
    all_vals = [results[label] for label in labels]
    x = []
    y = []
    for i, label in enumerate(labels):
        for _, val in all_vals[i]:
            x.append(i)
            y.append(val)
    plt.scatter(x, y)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha="right")
    plt.ylabel("Best validation loss")


if __name__ == "__main__":
    plots_for_pdfa()
    plots_for_baseline()
