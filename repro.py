"""Minimal reproducer: the per-round accept-preserving invariant fails on the
repo's own `subseq` language under the repo's own asymmetric noise, at every
shipped default.  Run from the repo root."""
import sys

from orthogonal_dfa.l_star.examples.bernoulli_parity import BernoulliRegex
from orthogonal_dfa.l_star.structures import AsymmetricBernoulli, NoisyOracle
from tests.lstar_common import evaluate_accuracy, learn_dfa_verified


def oracle(noise_model, seed):
    return NoisyOracle(BernoulliRegex(regex=r".*1010101.*"), noise_model, seed)


def main(seeds):
    for seed in seeds:
        try:
            dfa = learn_dfa_verified(
                oracle, min_signal_strength=0.2, seed=seed,
                noise_model=AsymmetricBernoulli(p_0=0.15, p_1=0.7))
        except AssertionError as e:
            print(f"seed {seed}: ROUND CHECK FAILED -- {e}", file=sys.stderr)
            continue
        acc = evaluate_accuracy(dfa, oracle)
        print(f"seed {seed}: ok, {len(dfa.states)} states, accuracy {acc:.4f}",
              file=sys.stderr)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or list(range(20)))
