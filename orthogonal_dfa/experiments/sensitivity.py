from typing import Dict, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
import pythomata
from permacache import permacache
import frame_alignment_checks as fac

from orthogonal_dfa.mutation.mutation import Mutation, RandomSingleMutation
from orthogonal_dfa.oracle.evaluate import (
    Metric,
    evaluate_dfas,
    multidimensional_confusion,
)
from orthogonal_dfa.utils.dfa import TorchDFA, hash_dfa
from orthogonal_dfa.utils.plotting import plot_vertical_histogram


@permacache(
    "orthogonal_dfa/experiments/sensitivity/sensitivity_analysis",
    key_function=dict(
        dfa=hash_dfa, control_dfas=lambda x: tuple(hash_dfa(d) for d in x)
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
):
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
        count=100_000,
        seed=rng.integers(1 << 32),
    )
    return confs, mut_desc


def sensitivities_to_plot(settings, model, exon, num_samples, seed):
    results = {}
    for name, (d, *controls) in settings.items():
        results[name, 0] = np.array(
            evaluate_dfas(exon, [d], controls, model, count=100_000, seed=seed)
        )
        results[name, 1] = sensitivity_analysis(
            d,
            controls,
            model,
            exon,
            RandomSingleMutation(),
            num_samples=num_samples,
            seed=seed,
        )[0]

    return results


def plot_sensitivity(
    settings: Dict[str, Tuple[pythomata.SimpleDFA, ...]],
    model,
    exon,
    num_samples: int,
    seed: int,
    metric: Metric,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    results = sensitivities_to_plot(settings, model, exon, num_samples, seed)
    results = {k: metric(v) for k, v in results.items()}
    xticks = [f"[{count} mut]" if count > 0 else name for name, count in results.keys()]
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=90, ha="right")
    name_to_color = {
        name: fac.plotting.line_color(i) for i, name in enumerate(settings)
    }
    for x, ((name, _), orig) in enumerate(results.items()):
        many = len(orig) > 10
        plot_vertical_histogram(x, orig, ax=ax, color=name_to_color[name], alpha=0.3 if many else 1, marker="." if many else None)
