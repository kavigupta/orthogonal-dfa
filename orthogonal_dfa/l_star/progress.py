"""Transient, delayed progress bars for the phases of a synthesis round."""

import tqdm.auto as tqdm

#: Seconds a phase must run before its bar appears.
DELAY = 5


def counter(total, desc):
    return tqdm.tqdm(total=total, desc=desc, leave=False, delay=DELAY)


def track(iterable, desc):
    return tqdm.tqdm(iterable, desc=desc, leave=False, delay=DELAY)


def write(line):
    """Print without tearing whatever bar is on screen.

    A bar this forces up before its delay elapses is one ``close`` would leave
    there, still believing it never displayed, so mark them all displayed."""
    tqdm.tqdm.write(line)
    for bar in list(tqdm.tqdm._instances):  # pylint: disable=protected-access
        bar.last_print_t = max(bar.last_print_t, bar.start_t + bar.delay)
