"""Progress bars for the phases of a synthesis round.

Transient and delayed: the round prints its own summary lines, and a phase that
finishes quickly should not flash a bar at all.
"""

import tqdm.auto as tqdm

#: Seconds a phase must run before its bar appears.
DELAY = 5


def counter(desc, total):
    return tqdm.tqdm(total=total, desc=desc, leave=False, delay=DELAY)


def track(iterable, desc):
    return tqdm.tqdm(iterable, desc=desc, leave=False, delay=DELAY)


def write(line):
    """Print without tearing whatever bar is on screen."""
    tqdm.tqdm.write(line)
