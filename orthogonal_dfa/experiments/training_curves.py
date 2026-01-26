from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from orthogonal_dfa.experiments.gate_experiments import train_psamdfa


def process_results(results):
    gates, train, results = zip(*results)
    train, metadata = unpack_train(train)
    return dict(
        gates=gates,
        last_gates=[g[-1] for g in gates],
        curves=-np.array(train)[:, 0, :, :].mean(-1),
        eva=[r[-1] for r in results],
        metadata=[x[0] for x in metadata],
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


def extract_train_metadata(train_tuple):
    if isinstance(train_tuple, dict):
        train, meta = train_tuple["loss"], train_tuple["sparse_info"]
        return train, dict(
            sparsity=meta["original_sparsity"],
            gate_before_update=train_tuple.get("gate_before_update", None),
        )
    if isinstance(train_tuple[0], float):
        return train_tuple, None
    train, meta = train_tuple
    assert isinstance(meta, float)
    meta = dict(sparsity=meta)
    return train, meta


def fill_in(metadata_list):
    last_non_none = None
    results = []
    for meta in metadata_list:
        if meta is not None:
            last_non_none = meta
            results.append(meta)
        else:
            results.append(last_non_none)
    if last_non_none is None:
        return None
    return results


def unpack_train(ts_by_seed):
    train, metadata = [], []
    for ts in ts_by_seed:
        train_for_seed, metadata_for_seed = [], []
        for t in ts:
            train_el, meta_el = zip(*(extract_train_metadata(a) for a in t))
            meta_el = fill_in(meta_el)
            train_for_seed.append(train_el)
            metadata_for_seed.append(meta_el)
        train.append(np.array(train_for_seed))
        metadata.append(metadata_for_seed)
    return train, metadata


def sparsity_training_curve(train, metadata):
    assert len(train) == len(metadata), f"{len(train)} != {len(metadata)}"
    sparsities = np.array([m["sparsity"] for m in metadata])
    train = np.array(train)
    [idxs] = np.where(sparsities[1:] != sparsities[:-1])
    sparsities, train = sparsities[idxs], train[idxs]
    densities = 1.0 - sparsities
    return densities, train


def plot_sparsity_curves(models):
    size = 5
    models = [m for m in models if any(md is not None for md in m.results["metadata"])]
    ncols = min(len(models), 3)

    _, axs = plt.subplots(
        (len(models) + ncols - 1) // ncols,
        ncols,
        figsize=(ncols * size, size * len(models) / ncols),
        dpi=200,
    )
    if len(models) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for ax, model in zip(
        axs,
        models,
    ):
        for seed, (train, metadata) in enumerate(
            zip(model.results["curves"], model.results["metadata"])
        ):
            if metadata is None:
                continue
            densities, train = sparsity_training_curve(train, metadata)
            ax.plot(
                train * 1000,
                np.log(100 * densities),
                label=f"Seed={seed}",
            )
        ax.set_xlabel("Information [mb]")
        ax.set_ylabel("Density")
        setup_yticks_as_log(ax)
        ax.set_title(model.name)
        ax.legend()


def models_by_sparsity(results):
    by_sparsity = defaultdict(list)

    for points in results["metadata"]:
        for point in points:
            if point["gate_before_update"] is None:
                continue
            by_sparsity[point["sparsity"]].append(point["gate_before_update"])
    return by_sparsity


def setup_yticks_as_log(ax):
    """
    Plot y-axis ticks in logarithmic scale, i.e., map log(2) -> 2, log(5) -> 5, etc.

    Put the 1, 2, 5 for every order of magnitude relevant
    """
    lo, hi = ax.get_ylim()
    lo = lo - 0.2
    lo_actual, hi_actual = np.exp(lo), np.exp(hi)
    min_oom, max_oom = int(np.floor(np.log10(lo_actual))), int(
        np.ceil(np.log10(hi_actual))
    )
    ticks = []
    for oom in range(min_oom, max_oom + 1):
        for m in [1, 2, 5]:
            val = m * 10**oom
            if lo_actual <= val <= hi_actual:
                ticks.append(val)
    ax.set_yticks(np.log(ticks))
    ax.set_yticklabels(ticks)
