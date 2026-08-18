"""Recovering a regular signal buried under *deterministic* non-regular noise.

The noise is lex-compare -- ``accept iff prefix > suffix`` -- a fixed function of
the string, not a coin.  It is the honest adversary because it is
*non-averageable*: the shatter lives in the accept RATE itself (a prefix's accept
rate over random suffixes is its rank), so no amount of averaging over a suffix
family removes it.  (A whole-string hash would be averageable -- per suffix it is
an independent coin -- i.e. just random noise, which proves nothing.)

Two distinguishing rules are compared on the same fixed-length observation matrix
(as ``shattering_repro`` does -- lex needs a fixed midpoint, so it cannot be fed
to the streaming learner):

* exact rows (what E-L* distinguishes on) -- shatters to ~one state per prefix and
  carries no held-out signal;
* accept-rate within a tolerance (what direct-L* distinguishes on) -- stays
  bounded and recovers the signal held-out.

The signal g = "at least 2 of 3 reading frames closed" is entangled: signal
columns give ``g(prefix + suffix)``, so g is only recoverable by averaging over
the family, never readable from a single cell.
"""

import hashlib
import unittest

import numpy as np

LENGTH = 48
N_SUFFIXES = 4000
N_TRAIN = 300
N_TEST = 150
SIGNAL_WEIGHT = 0.6  # fraction of distinguishers that reveal g; the rest are lex noise
TAU = 0.05  # accept-rate gap a direct-L* split must clear
MIN_LEAF = 6
#: TAG, TGA, TAA over the ACGT -> 0123 encoding.
STOPS = {(3, 0, 2), (3, 2, 0), (3, 0, 0)}


def _frames_ge2(strings):
    """Signal g: at least 2 of the 3 reading frames contain a stop codon.
    ``strings`` is an (n, m) array of equal-length rows."""
    count = np.zeros(strings.shape[0], int)
    width = strings.shape[1]
    for phase in range(3):
        closed = np.zeros(strings.shape[0], bool)
        k = 0
        while phase + 3 * k + 2 < width:
            i = phase + 3 * k
            a, b, c = strings[:, i], strings[:, i + 1], strings[:, i + 2]
            closed |= (
                ((a == 3) & (b == 0) & (c == 2))
                | ((a == 3) & (b == 2) & (c == 0))
                | ((a == 3) & (b == 0) & (c == 0))
            )
            k += 1
        count += closed
    return count >= 2


def _lex_gt(prefixes, suffix):
    """Non-averageable shatter: ``prefix > suffix`` lexicographically, per prefix."""
    diff = prefixes != suffix
    first = diff.argmax(1)
    greater = prefixes[np.arange(prefixes.shape[0]), first] > suffix[first]
    return np.where(diff.any(1), greater, False)


def _observation_matrix(prefixes, suffixes, signal_weight):
    """M[i, j] = oracle(prefix_i + suffix_j): a signal column gives g of the whole
    string, a noise column gives the lex shatter."""
    reveals_g = np.array(
        [
            int.from_bytes(hashlib.blake2b(bytes(s), digest_size=8).digest(), "big")
            / 2**64
            < signal_weight
            for s in suffixes.tolist()
        ],
        bool,
    )
    matrix = np.empty((prefixes.shape[0], suffixes.shape[0]), bool)
    for j in range(suffixes.shape[0]):
        if reveals_g[j]:
            whole = np.hstack([prefixes, np.tile(suffixes[j], (prefixes.shape[0], 1))])
            matrix[:, j] = _frames_ge2(whole)
        else:
            matrix[:, j] = _lex_gt(prefixes, suffixes[j])
    return matrix


# -- the two distinguishing rules, each scored held-out ---------------------


def _exact_row_heldout(matrix, train, test, target, columns):
    """E-L*: a state is an exact membership row. A held-out prefix is placed only
    if its row was seen in training, else it falls to the prior -- so a shattered
    table cannot generalise."""
    prior = bool(target[train].mean() >= 0.5)
    by_row = {}
    for i in train:
        by_row.setdefault(matrix[i, columns].tobytes(), []).append(target[i])
    distinct = len(by_row)
    predictions = np.array(
        [
            (
                bool(np.mean(by_row[matrix[j, columns].tobytes()]) >= 0.5)
                if matrix[j, columns].tobytes() in by_row
                else prior
            )
            for j in test
        ],
        bool,
    )
    accuracy = float((predictions == target[test]).mean())
    return distinct, accuracy


def _accept_rate_tree(matrix, rows, assign, test, tau):
    """direct-L*: split a group only where a distinguisher separates it by a
    held-out accept-rate gap over ``tau``; ASSIGN columns propose, TEST columns
    score. Returns a routable tree of (column) nodes and leaves."""
    if len(rows) < 2 * MIN_LEAF:
        return ("leaf", rows)
    best_gap, best_column, best_side = 0.0, -1, None
    scored = matrix[np.ix_(rows, test)]
    for c in assign:
        side = matrix[rows, c]
        n_accept = int(side.sum())
        if n_accept < MIN_LEAF or len(rows) - n_accept < MIN_LEAF:
            continue
        gap = abs(scored[side].mean() - scored[~side].mean())
        if gap > best_gap:
            best_gap, best_column, best_side = gap, c, side
    if best_column < 0 or best_gap <= tau:
        return ("leaf", rows)
    return (
        "node",
        best_column,
        _accept_rate_tree(matrix, rows[~best_side], assign, test, tau),
        _accept_rate_tree(matrix, rows[best_side], assign, test, tau),
    )


def _count_leaves(tree):
    if tree[0] == "leaf":
        return 1
    return _count_leaves(tree[2]) + _count_leaves(tree[3])


def _label_leaves(tree, target):
    if tree[0] == "leaf":
        rows = tree[1]
        return ("leaf", bool(target[rows].mean() >= 0.5) if len(rows) else False)
    return (
        "node",
        tree[1],
        _label_leaves(tree[2], target),
        _label_leaves(tree[3], target),
    )


def _route(tree, row):
    while tree[0] == "node":
        tree = tree[3] if row[tree[1]] else tree[2]
    return tree[1]


class TestRecoverUnderShatteringNoise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        prefixes = rng.integers(0, 4, (N_TRAIN + N_TEST, LENGTH))
        suffixes = rng.integers(0, 4, (N_SUFFIXES, LENGTH))
        cls.matrix = _observation_matrix(prefixes, suffixes, SIGNAL_WEIGHT)
        cls.target = _frames_ge2(prefixes)  # recover g on each prefix
        cls.train = np.arange(N_TRAIN)
        cls.test = np.arange(N_TRAIN, N_TRAIN + N_TEST)
        cls.assign = np.arange(0, N_SUFFIXES, 2)
        cls.scoring = np.arange(1, N_SUFFIXES, 2)
        cls.baseline = max(cls.target[cls.test].mean(), 1 - cls.target[cls.test].mean())

    def test_exact_distinguishing_shatters_and_loses_the_signal(self):
        distinct, accuracy = _exact_row_heldout(
            self.matrix, self.train, self.test, self.target, self.assign
        )
        self.assertGreater(
            distinct,
            0.8 * N_TRAIN,
            f"exact distinguishing should shatter (~1 state/prefix); "
            f"got {distinct}/{N_TRAIN}",
        )
        self.assertLess(
            accuracy,
            self.baseline + 0.1,
            f"a shattered table should not generalise; held-out accuracy "
            f"{accuracy:.3f} vs baseline {self.baseline:.3f}",
        )

    def test_accept_rate_stays_bounded_and_recovers_the_signal(self):
        tree = _accept_rate_tree(
            self.matrix, self.train, self.assign, self.scoring, TAU
        )
        states = _count_leaves(tree)
        labelled = _label_leaves(tree, self.target)
        predictions = np.array(
            [_route(labelled, self.matrix[j]) for j in self.test], bool
        )
        accuracy = float((predictions == self.target[self.test]).mean())
        self.assertLess(
            states,
            N_TRAIN // 5,
            f"accept-rate distinguishing should stay bounded; " f"got {states} states",
        )
        self.assertGreater(
            accuracy,
            0.85,
            f"accept-rate distinguishing should recover the signal; held-out "
            f"accuracy {accuracy:.3f} (baseline {self.baseline:.3f})",
        )


if __name__ == "__main__":
    unittest.main()
