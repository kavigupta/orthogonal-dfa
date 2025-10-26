from typing import Dict, Tuple, Union

import frame_alignment_checks as fac
import numpy as np
import pythomata
from matplotlib import pyplot as plt
from permacache import permacache, drop_if_equal

from orthogonal_dfa.mutation.mutation import (
    Mutation,
    RandomSingleMutation,
    RepeatedMutations,
)
from orthogonal_dfa.oracle.evaluate import (
    Metric,
    evaluate_dfas,
    multidimensional_confusion,
    simulate_bootstrap_confusion,
)
from orthogonal_dfa.utils.dfa import TorchDFA, hash_dfa
from orthogonal_dfa.utils.plotting import plot_vertical_histogram


@permacache(
    "orthogonal_dfa/experiments/sensitivity/sensitivity_analysis",
    key_function=dict(
        dfa=hash_dfa,
        control_dfas=lambda x: tuple(hash_dfa(d) for d in x),
        count=drop_if_equal(100_000),
    ),
)
def sensitivity_analysis(
    dfa: pythomata.SimpleDFA,
    control_dfas: list[pythomata.SimpleDFA],
    model,
    exon,
    mutation: Mutation,
    *,
    num_samples: Union[int, "all"],
    seed: int,
    count: int = 100_000,
):
    print(
        f"Running sensitivity analysis on {hash_dfa(dfa)} "
        f"(controlling for {[hash_dfa(x) for x in  control_dfas]}) using {mutation}"
    )
    dfa = TorchDFA.from_pythomata(dfa)
    control_dfas = TorchDFA.concat(
        *[TorchDFA.from_pythomata(d) for d in control_dfas],
        num_symbols=dfa.alphabet_size,
    )
    rng = np.random.default_rng(seed)
    mut_desc = (
        mutation.all_mutations(dfa)
        if num_samples == "all"
        else mutation.sample_mutations(dfa, num_samples, rng)
    )
    dfas = mutation.apply_mutations(dfa, mut_desc)
    confs = multidimensional_confusion(
        exon,
        dfas,
        control_dfas,
        model,
        count=count,
        seed=rng.integers(1 << 32),
    )
    return confs, mut_desc


def sensitivities_to_plot(settings, model, exon, num_samples, seed, *, count):
    results = {}
    for name, (d, *controls) in settings.items():
        results[name, 0] = np.array(
            evaluate_dfas(exon, [d], controls, model, count=count, seed=seed)
        )
        for num_sample in (1, 2, 3):
            results[name, num_sample] = sensitivity_analysis(
                d,
                controls,
                model,
                exon,
                RepeatedMutations(RandomSingleMutation(), num_sample),
                num_samples=num_samples,
                seed=seed,
                count=count,
            )[0]

    return results


def plot_sensitivity(
    settings: Dict[str, Tuple[pythomata.SimpleDFA, ...]],
    model,
    exon,
    num_samples: int,
    metric: Metric,
    *,
    seed: int = 0,
    ax=None,
    count: int = 100_000,
):
    if ax is None:
        ax = plt.gca()
    results = sensitivities_to_plot(
        settings, model, exon, num_samples, seed, count=count
    )
    xticks = [f"[{count} mut]" if count > 0 else name for name, count in results]
    gaps = np.ones((len(settings), len(results) // len(settings)), dtype=int)
    gaps[:, 0] = 2
    xlocs = np.cumsum(gaps.flatten())
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticks, rotation=90, ha="right")
    ax.set_ylabel(metric.name)
    name_to_color = {
        name: fac.plotting.line_color(i % 6) for i, name in enumerate(settings)
    }
    for x, ((name, _), orig) in zip(xlocs, results.items()):
        if orig.shape[0] == 1:
            (lo, hi) = simulate_bootstrap_confusion(metric, orig[0])
            plt.errorbar(
                [x],
                [(lo + hi) / 2],
                yerr=[(hi - lo) / 2],
                fmt="o",
                color=name_to_color[name],
                capsize=5,
            )
        orig = metric(orig)
        many = len(orig) > 10
        plot_vertical_histogram(
            x,
            orig,
            ax=ax,
            color=name_to_color[name],
            alpha=0.3 if many else 1,
            marker="." if many else None,
        )
