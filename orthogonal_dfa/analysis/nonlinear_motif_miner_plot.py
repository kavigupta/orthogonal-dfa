"""Plotting for the nonlinear motif miner."""

from matplotlib import pyplot as plt


def plot_top_motifs(stats, *, top=15, ax=None):
    """Horizontal bar chart of the top motifs in stats by marginal benefit.

    Draws on the current axes when ax is not given, so the caller can set the figure up
    (size, subplots) first.
    """
    if ax is None:
        ax = plt.gca()
    top_stats = sorted(stats, key=lambda s: -s.marginal)[:top]
    ax.barh(range(len(top_stats)), [s.marginal for s in top_stats], color="#4c72b0")
    ax.set_yticks(range(len(top_stats)))
    ax.set_yticklabels([s.motif for s in top_stats])
    ax.invert_yaxis()
    ax.set_xlabel("marginal benefit")
    return ax
