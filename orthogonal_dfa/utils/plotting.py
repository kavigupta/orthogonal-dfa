import numpy as np
from matplotlib import pyplot as plt


def plot_vertical_histogram(x, yvals, width=0.25, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    yvals = np.sort(np.array(yvals))

    x = np.zeros(len(yvals)) + x

    if len(yvals) > 50:
        jenks = np.linspace(yvals.min(), yvals.max(), 11)
        starts, ends = jenks[:-1], jenks[1:]
        counts_each = ((yvals >= starts[:, None]) & (yvals < ends[:, None])).sum(axis=1)
        counts_each = counts_each / counts_each.max() * width
        noise_level = np.zeros(len(yvals))
        for start, end, count in zip(starts, ends, counts_each):
            if count > 0:
                noise_level[(yvals >= start) & (yvals < end)] = count
        x += (np.random.random(len(yvals)) - 0.5) * 2 * noise_level
    ax.scatter(x, yvals, **kwargs)
