"""Transient, delayed progress bars for the phases of a synthesis round."""

import tqdm.auto as tqdm

#: Seconds a phase must run before its bar appears.
DELAY = 5


def counter(total, desc, *, delay=DELAY):
    return tqdm.tqdm(total=total, desc=desc, leave=False, delay=delay)


def track(iterable, desc):
    return tqdm.tqdm(iterable, desc=desc, leave=False, delay=DELAY)


def write(line):
    """Print without tearing whatever bar is on screen.

    Only safe under a bar built with ``delay=0``: this redraws every live bar
    whatever its delay, and one redrawn before its delay elapses is one
    ``close`` then leaves on screen, still believing it never displayed."""
    tqdm.tqdm.write(line)
