"""Adapter that runs the official CAPAL implementation against this repo's
oracles.

The upstream code lives at github.com/lkwargs/CAPAL (cloned locally to
/tmp/CAPAL). We import it via sys.path insertion -- it is a single-file
module with no installable package layout. This adapter:

- Constructs the noiseless target DFA for each of our oracle creators (their
  CAPALLearner needs a target DFA so PerfectEQ can do a BFS product
  counterexample search).
- Runs the official `CAPALLearner.fit()` with the user's eta and seed.
- Returns a callable that evaluates the learned DFA against a noiseless
  ground-truth sampler.
"""

from .adapter import (
    OFFICIAL_AVAILABLE,
    build_all_frames_closed_dfa,
    build_modulo_dfa,
    build_regex_dfa,
    evaluate_official_dfa,
    run_official_capal,
)
