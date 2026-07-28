"""One target, presentable to *both* learners.

A benchmark has to be readable two ways: as an upstream `capal.DFA` (what CAPAL
learns) and as one of this repo's `Oracle`s (what E-L* learns). The two views
must denote the same language under the same symbol ordering, or the
head-to-head is meaningless -- so both are derived from a single source of
truth per family, in `our_targets` and `capal_targets`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

from orthogonal_dfa.capal_official import to_automata_dfa

from . import regime

FAMILY_OURS = "ours"
FAMILY_CAPAL = "capal_dataset"


@dataclass
class Benchmark:
    """One target, in both learners' terms.

    `alphabet` is the index -> symbol mapping shared by both views: E-L* works
    in symbol indices, CAPAL in characters, and this is what ties them.
    """

    name: str
    family: str
    target: Any  # upstream capal.DFA
    oracle_creator: Callable[[Any, int], Any]
    alphabet: Sequence[str]
    symbols: int
    target_states: int

    def truth(self) -> Callable[[List[int]], bool]:
        """Noiseless ground truth, taken from the upstream DFA so that both
        learners are scored against one definition of the language."""
        dfa, alpha = self.target, self.alphabet
        return lambda w: bool(dfa.run("".join(alpha[i] for i in w)))

    def regime_report(self) -> Dict[str, Any]:
        """Is this target inside E-L*'s designed regime, and if not, why not?"""
        return regime.report(to_automata_dfa(self.target))
