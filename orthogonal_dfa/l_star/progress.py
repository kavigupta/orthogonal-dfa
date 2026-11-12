"""Transient, delayed progress bars for the phases of a synthesis round."""

import tqdm.auto as tqdm

#: Seconds a phase must run before its bar appears.
DELAY = 5


def counter(total, desc):
    return tqdm.tqdm(total=total, desc=desc, leave=False, delay=DELAY)


def track(iterable, desc):
    return tqdm.tqdm(iterable, desc=desc, leave=False, delay=DELAY)
