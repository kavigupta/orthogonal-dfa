#!/usr/bin/env python3
"""Validate the learnability preconditions on random DFAs.

Samples uniformly random DFAs (``sample_random_dfa``), keeps the ones that pass
``preconditions.satisfies_preconditions``, and runs E-L* on each to confirm it
actually learns them -- i.e. the criterion has no *false positives* (a DFA it
admits that E-L* cannot learn). Also reports how closely the covered-accuracy
ceiling tracks E-L*'s actual accuracy, and (optionally) whether any *excluded*
DFA would have learned (a false negative / over-exclusion).

Usage:
    python scripts/validate_preconditions.py [num_dfas] --eta 0.05
"""
from __future__ import annotations

import argparse
import signal

import numpy as np

from orthogonal_dfa.l_star import preconditions as P
from orthogonal_dfa.l_star.examples.benchmark_generator import (
    DFAOracle,
    sample_random_dfa,
)
from tests.test_lstar import compute_dfa_accuracy, compute_dfa_for_oracle

LEARNED = 0.95  # E-L* accuracy bar for "learned it"


def elstar_accuracy(aut, eta: float, timeout: int):
    """Noiseless accuracy of the DFA E-L* learns for ``aut``, or None if it
    gave up / errored / timed out."""
    oracle_creator = lambda nm, s, _a=aut: DFAOracle(nm, s, _a)  # noqa: E731

    def _timeout(*_):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(timeout)
    try:
        _, dfa, _ = compute_dfa_for_oracle(
            oracle_creator, min_signal_strength=0.5 - eta, seed=0
        )
        if dfa is None:
            return None
        acc, _, _ = compute_dfa_accuracy(dfa, oracle_creator)
        return acc
    except Exception:  # pylint: disable=broad-exception-caught
        # gave up, timed out, or errored -- all count as "did not learn"
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("num_dfas", type=int, nargs="?", default=100)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--min-states", type=int, default=2)
    ap.add_argument("--max-states", type=int, default=5)
    ap.add_argument(
        "--check-excluded",
        action="store_true",
        help="Also run E-L* on excluded DFAs to look for over-exclusions.",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    passed, false_pos, over_excl = [], [], []
    n_pass = n_excl = 0
    for i in range(args.num_dfas):
        n_states = int(rng.integers(args.min_states, args.max_states + 1))
        aut = sample_random_dfa(rng, num_states=n_states, alphabet_size=2)
        ok = P.satisfies_preconditions(aut, length=args.length)
        ceiling = P.covered_accuracy_ceiling(aut, length=args.length)
        if ok:
            n_pass += 1
            acc = elstar_accuracy(aut, args.eta, args.timeout)
            passed.append((n_states, ceiling, acc))
            learned = acc is not None and acc >= LEARNED
            print(
                f"  [{i}] PASS {n_states}st ceiling={ceiling:.3f} "
                f"E-L*={acc}" + ("" if learned else "   <-- did NOT learn"),
                flush=True,
            )
            if not learned:
                false_pos.append((n_states, ceiling, acc))
        elif args.check_excluded:
            n_excl += 1
            acc = elstar_accuracy(aut, args.eta, args.timeout)
            if acc is not None and acc >= LEARNED:
                over_excl.append((n_states, ceiling, acc))
                print(
                    f"  [{i}] EXCL {n_states}st ceiling={ceiling:.3f} "
                    f"but E-L*={acc:.3f}   <-- over-excluded",
                    flush=True,
                )

    print(f"\n{n_pass}/{args.num_dfas} random DFAs passed the criterion.")
    print(
        f"FALSE POSITIVES (passed but E-L* did not learn to {LEARNED}): "
        f"{len(false_pos)}"
    )
    for n_states, ceiling, acc in false_pos:
        print(f"   {n_states}st ceiling={ceiling:.3f} acc={acc}")
    if args.check_excluded:
        print(
            f"OVER-EXCLUSIONS (excluded but E-L* learned) out of {n_excl} "
            f"excluded: {len(over_excl)}"
        )


if __name__ == "__main__":
    main()
