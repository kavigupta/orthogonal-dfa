"""Adapter that runs the official CAPAL implementation against this repo's
oracles.

The upstream code lives at github.com/lkwargs/CAPAL, expected as a sibling
checkout at `../capal` (override with $ORTHO_CAPAL_DIR). We import it via
sys.path insertion -- it is a single-file module with no installable package
layout -- after checking it sits at the pinned commit with a clean tree, so
the numbers in data/capal_findings.md stay reproducible. This adapter:

- Constructs the noiseless target DFA for each of our oracle creators (their
  CAPALLearner needs a target DFA so PerfectEQ can do a BFS product
  counterexample search).
- Builds the official `CAPALLearner` with the user's eta and seed, and runs
  `.fit()` with the iteration cap treated as a non-convergent result rather
  than an error.

Scoring the learned DFA is the caller's job -- see
`orthogonal_dfa.experiments.capal_comparison`, which scores every learner on
one shared word list.
"""

from .adapter import (
    DEFAULT_CAPAL_DIR,
    PINNED_COMMIT,
    build_modulo_dfa,
    build_regex_dfa,
    fit_with_fallback,
    import_capal,
    make_learner,
    resolve_capal_dir,
    verify_pinned,
)
