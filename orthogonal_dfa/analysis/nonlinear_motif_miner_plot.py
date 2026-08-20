"""Plotting for the nonlinear motif miner (see :mod:`nonlinear_motif_miner`)."""

from matplotlib import pyplot as plt


def plot_top_motifs(
    stats, *, top=15, value="marginal", xlabel="marginal benefit", ax=None
):
    """Horizontal bar chart of the top motifs by the ``value`` attribute (e.g. ``marginal``
    or ``magnitude``) of the ``MotifRecord``\\ s in ``stats``."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    top_stats = sorted(stats, key=lambda s: -getattr(s, value))[:top]
    ax.barh(
        range(len(top_stats)), [getattr(s, value) for s in top_stats], color="#4c72b0"
    )
    ax.set_yticks(range(len(top_stats)))
    ax.set_yticklabels([s.motif for s in top_stats])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    return ax
