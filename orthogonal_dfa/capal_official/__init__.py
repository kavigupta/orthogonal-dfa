"""
Run the baseline CAPAL learner (github.com/lkwargs/CAPAL) against this repo's oracles.

Upstream is a pinned sibling checkout at `../capal`.
Scoring the learned DFA is the caller's job, see `orthogonal_dfa.experiments.capal_comparison`.
"""

from .adapter import (
    DEFAULT_CAPAL_DIR,
    PINNED_COMMIT,
    fit_with_fallback,
    import_capal,
    make_learner,
    resolve_capal_dir,
    verify_pinned,
)
from .porters import build_modulo_dfa, build_regex_dfa, to_automata_dfa
