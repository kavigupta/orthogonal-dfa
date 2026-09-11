"""Reproduce E-L*'s suffix-family "shattering" with tiny, deterministic grammars -- no
SpliceAI, no GPU, no learned weights.

Hypothesis (from the gate-deconfounded run): E-L* shatters -- suffix family and state
count grow without bound, ~0 held-out signal -- when the oracle is **deterministic but
not compactly regular**, so its observation-table rows never collapse into finitely many
states.  We show this with grammars you can read:

- ``RandomDFAOracle`` -- a random k-state DFA: genuinely regular, saturates at k states.
- ``AllFramesClosedOracle`` (imported) -- the no-ORF language: regular, ~20 states.
- ``HalfCountCompareOracle`` -- "first half has more A's than the second half".  A
  comparison, but of a *statistic*, so it collapses to ~(length/2) states.  The control
  that shows shattering is not just "many states".
- ``HalfLexCompareOracle`` -- "first half > second half, lexicographically".  Same idea,
  but comparing the *whole* prefix: two prefixes are E-L*-equivalent only if they are
  identical, so every prefix is its own state.  This shatters.

Measured directly and cheaply: over a fixed prefix set, count the distinct
observation-table rows (= states E-L* would create by exact deterministic distinguishing)
as the random suffix set grows.  Regular/statistic targets saturate; the lexicographic
grammar runs away to one-state-per-prefix.  ``run_round0`` runs the real (slow) synthesis
if you want to confirm end to end.
"""

import collections

import numpy as np

from orthogonal_dfa.l_star.examples.bernoulli_parity import AllFramesClosedOracle
from orthogonal_dfa.l_star.lstar import counterexample_driven_synthesis
from orthogonal_dfa.l_star.preconditions import _endpoint
from orthogonal_dfa.l_star.prefix_suffix_tracker import PrefixSuffixTracker, SearchConfig
from orthogonal_dfa.l_star.sampler import UniformSampler
from orthogonal_dfa.l_star.statistics import (
    compute_suffix_size_counterexample_gen,
    population_size_and_evidence_margin,
)
from orthogonal_dfa.l_star.structures import NoiseModel, Oracle


class NoNoise(NoiseModel):
    def apply_noise(self, correct_value, string, seed):
        return correct_value


class RandomDFAOracle(Oracle):
    """A random k-state DFA over the 4-letter alphabet -- a genuinely regular target that
    E-L* should learn (saturates at k states)."""

    def __init__(self, *, n_states=6, accept_frac=0.5, seed=1):
        rng = np.random.default_rng(seed)
        self.trans = rng.integers(0, n_states, (n_states, 4))
        acc = rng.random(n_states) < accept_frac
        acc[0] = acc[0] or not acc.any()
        self.accept = acc
        self.n_states = n_states

    @property
    def alphabet_size(self):
        return 4

    def membership_queries(self, strings):
        strings = list(strings)
        n = len(strings)
        lens = np.fromiter((len(s) for s in strings), int, n)
        ML = int(lens.max()) if n else 0
        arr = np.zeros((n, ML), dtype=np.int64)
        for i, s in enumerate(strings):
            arr[i, : len(s)] = s
        st = np.zeros(n, dtype=np.int64)
        for pos in range(ML):
            st = np.where(pos < lens, self.trans[st, arr[:, pos]], st)
        return self.accept[st]

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])


class HalfLexCompareOracle(Oracle):
    """accept iff the first half of the string is lexicographically greater than the
    second half.

    Non-regular / shattering: for prefixes p1 != p2 there is always a suffix s with
    [p1 > s] != [p2 > s] (any s lying between them), so E-L* can never merge two distinct
    prefixes -- every prefix becomes its own state.  Balanced (~0.5)."""

    @property
    def alphabet_size(self):
        return 4

    def membership_queries(self, strings):
        strings = [list(s) for s in strings]
        n = len(strings)
        if n and len({len(s) for s in strings}) == 1:
            a = np.asarray(strings, dtype=np.int64)
            h = a.shape[1] // 2
            A, B = a[:, :h], a[:, h : 2 * h]
            diff = A != B
            first = diff.argmax(1)
            gt = A[np.arange(n), first] > B[np.arange(n), first]
            return np.where(diff.any(1), gt, False)
        out = np.empty(n, bool)
        for i, s in enumerate(strings):
            h = len(s) // 2
            out[i] = s[:h] > s[h : 2 * h]
        return out

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])


class HalfCountCompareOracle(Oracle):
    """accept iff the first half has more A's than the second half.  A comparison, but of
    a *count*, so E-L* only needs to remember the running A-count -- it collapses to about
    length/2 states rather than shattering.  The control that isolates *why* lex compare
    shatters (whole prefix) while this does not (a statistic)."""

    @property
    def alphabet_size(self):
        return 4

    def membership_queries(self, strings):
        strings = [list(s) for s in strings]
        n = len(strings)
        if n and len({len(s) for s in strings}) == 1:
            a = np.asarray(strings, dtype=np.int64)
            h = a.shape[1] // 2
            return (a[:, :h] == 0).sum(1) > (a[:, h : 2 * h] == 0).sum(1)
        out = np.empty(n, bool)
        for i, s in enumerate(strings):
            h = len(s) // 2
            out[i] = s[:h].count(0) > s[h : 2 * h].count(0)
        return out

    def membership_query(self, string):
        return bool(self.membership_queries([string])[0])


def shattering_curve(oracle, *, length, n_prefixes=300, n_suffixes=4000, seed=0):
    """Distinct observation-table rows (= states, by E-L*'s exact deterministic
    distinguishing) as the suffix set grows.  Saturates for regular/statistic targets;
    climbs toward n_prefixes for a shattering one."""
    rng = np.random.default_rng(seed)
    prefixes = rng.integers(0, 4, (n_prefixes, length))
    suffixes = rng.integers(0, 4, (n_suffixes, length))
    M = np.empty((n_prefixes, n_suffixes), dtype=bool)
    for i in range(n_prefixes):
        p = prefixes[i].tolist()
        M[i] = oracle.membership_queries([p + suffixes[j].tolist() for j in range(n_suffixes)])
    ks = [k for k in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000) if k <= n_suffixes]
    counts = [int(len(np.unique(M[:, :k], axis=0))) for k in ks]
    return dict(n_prefixes=n_prefixes, ks=ks, distinct_rows=counts,
                final_frac=counts[-1] / n_prefixes)


def _phi(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.std() == 0 or truth.std() == 0:
        return 0.0
    return float(np.corrcoef(pred, truth)[0, 1])


def _mi_bits(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    mi = 0.0
    for a in (False, True):
        pa = np.mean(pred == a)
        if pa == 0:
            continue
        for b in (False, True):
            pb = np.mean(truth == b)
            pab = np.mean((pred == a) & (truth == b))
            if pab > 0:
                mi += pab * np.log2(pab / (pa * pb))
    return float(mi)


def _elstar_pred(dfa, strings, start):
    return np.array([q in dfa.final_states for q in (_endpoint(dfa, s, start) for s in strings)], bool)


def _measure(dfa, oracle, *, length, n, seed):
    """Held-out phi/MI of the DFA vs the oracle, choosing the best start state on a
    validation half and scoring on a disjoint test half (self-contained)."""
    rng = np.random.default_rng(seed)
    strings = rng.integers(0, 4, (n, length)).tolist()
    truth = np.asarray(oracle.membership_queries(strings), bool)
    h = n // 2
    val, test, vt, tt = strings[:h], strings[h:], truth[:h], truth[h:]
    rr = max(dfa.states, key=lambda q: abs(_phi(_elstar_pred(dfa, val, q), vt)))
    pred = _elstar_pred(dfa, test, rr)
    return dict(num_states=len(dfa.states), accept_rate=float(truth.mean()),
                phi_rerooted=_phi(pred, tt), mi_rerooted=_mi_bits(pred, tt))


def _search_config(mss, addtl, fnr_limit):
    """The E-L* search config for a given minimum signal strength (same recipe the
    learnability driver uses)."""
    n, eps = population_size_and_evidence_margin(
        signal_strength=mss, acceptable_fpr=0.01, acceptable_fnr=0.01
    )
    return SearchConfig(
        suffix_family_size=n,
        evidence_margin=eps,
        decision_rule_fpr=0.01,
        suffix_size_counterexample_gen=compute_suffix_size_counterexample_gen(0.01, 0.5 + mss),
        min_signal_strength=mss,
        num_addtl_prefixes=addtl,
        fnr_limit=fnr_limit,
    )


def run_round0(oracle, *, length, mss=0.06, num_prefixes=200, addtl=100, seed=0, eval_n=8000):
    """Run the real (slow) round-0 synthesis and report suffix-family size, state count,
    and held-out phi/MI vs the oracle."""
    config = _search_config(mss, addtl, fnr_limit=0.05)
    pst = PrefixSuffixTracker.create(
        UniformSampler(length), np.random.default_rng(seed), oracle, config,
        num_prefixes=num_prefixes,
    )
    gen = counterexample_driven_synthesis(pst, additional_counterexamples=addtl, acc_threshold=0.98)
    dfa, dt, _ = next(gen)
    n_suffixes = len(pst.table._suffixes)  # pylint: disable=protected-access
    m = _measure(dfa, oracle, length=length, n=eval_n, seed=999)
    return dict(n_suffixes=int(n_suffixes), **m)


def scenarios():
    return {
        "randomDFA (6 states, regular)": RandomDFAOracle(n_states=6, seed=1),
        "all-frames-closed (regular)": AllFramesClosedOracle(NoNoise(), seed=0),
        "half count-compare (statistic)": HalfCountCompareOracle(),
        "half lex-compare (whole prefix)": HalfLexCompareOracle(),
    }


REPORT_KS = (10, 100, 1000, 4000)


def to_markdown(results, length, n_prefixes):
    cols = " | ".join(f"{k} suf" for k in REPORT_KS)
    lines = [
        "# Reproducing E-L* suffix shattering with a simple grammar",
        "",
        f"No SpliceAI, no GPU, no learned weights.  Over {n_prefixes} random length-{length} "
        "prefixes we count the **distinct observation-table rows** (= states E-L* would "
        "create by exact deterministic distinguishing) as the random suffix set grows.  A "
        "regular target saturates at its state count; a non-regular one climbs toward one "
        "state per prefix -- that runaway is the shattering, the 397k-suffix SpliceAI "
        "explosion in miniature.",
        "",
        f"| oracle (grammar) | {cols} | final frac |",
        "|---|" + "---:|" * (len(REPORT_KS) + 1),
    ]
    for name, r in results.items():
        at = dict(zip(r["ks"], r["distinct_rows"]))
        cells = " | ".join(str(at.get(k, "")) for k in REPORT_KS)
        lines.append(f"| {name} | {cells} | {r['final_frac']:.2f} |")
    lines += [
        "",
        "**Reading.** Both comparison grammars ask \"is the first half bigger than the "
        "second half?\" -- the only difference is *bigger how*:",
        "",
        "- **count-compare** (more A's) compares a *statistic*, so E-L* only needs the "
        "running A-count -> it collapses to ~length/2 states and saturates.",
        "- **lex-compare** compares the *whole* prefix, so two prefixes are E-L*-equivalent "
        "only if identical -> every prefix becomes its own state, and the table shatters "
        "(final frac -> 1.0 as suffixes grow; still climbing where the others are flat).",
        "",
        "The random DFA (~5) and all-frames-closed (~20) also saturate -- so shattering is "
        "not \"many states\", it is **unbounded, non-regular** state growth.  And "
        "all-frames-closed is regular and learnable in isolation; it is only lost when "
        "embedded in a shattering function (as in SpliceAI's residual), which explodes "
        "E-L* before it can isolate the frame automaton.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    import sys

    length = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results/shattering_repro.md"
    n_prefixes = 300
    results = {}
    for name, oracle in scenarios().items():
        results[name] = shattering_curve(oracle, length=length, n_prefixes=n_prefixes)
        r = results[name]
        print(f"{name}: {dict(zip(r['ks'], r['distinct_rows']))} "
              f"final_frac={r['final_frac']:.2f}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(to_markdown(results, length, n_prefixes))
    print("\n" + to_markdown(results, length, n_prefixes))
