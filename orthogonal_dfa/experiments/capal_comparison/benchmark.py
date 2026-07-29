"""One target, presentable to *both* learners.

A benchmark has to be readable two ways: as an upstream `capal.DFA` (what CAPAL
learns) and as one of this repo's `Oracle`s (what E-L* learns).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence

from orthogonal_dfa.capal_official import to_automata_dfa
from orthogonal_dfa.l_star import preconditions
from orthogonal_dfa.l_star.learn import DEFAULT_SAMPLE_LENGTH

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
        return lambda w: dfa.run("".join(alpha[i] for i in w))

    def regime_report(self) -> preconditions.PreconditionReport:
        """Is this target inside E-L*'s designed regime, and if not, why not?

        Measured at the length E-L* draws its own words at, so the verdict is
        about the distribution the learner actually sees. The thresholds are
        `satisfies_preconditions`' own defaults, which are the values this
        repo's benchmark generator applies. Every reason is collected rather
        than just the first, so a recorded exclusion says everything that
        disqualified the target.
        """
        return preconditions.satisfies_preconditions(
            to_automata_dfa(self.target),
            length=DEFAULT_SAMPLE_LENGTH,
            short_circuit=False,
        )
