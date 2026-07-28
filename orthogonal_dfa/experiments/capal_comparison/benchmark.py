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

from . import regime

FAMILY_OURS = "ours"
FAMILY_CAPAL = "capal_dataset"


def taf_to_automata_dfa(target: Any) -> Any:
    """Upstream `capal.DFA` -> automata-lib DFA over integer symbols.

    `DFAOracle` (and hence E-L*) works in symbol *indices*; upstream works in
    characters. Index i always means `target.alphabet[i]`, which is the same
    mapping `Benchmark.alphabet` hands to the CAPAL side.
    """
    from automata.fa.dfa import DFA as AutDFA

    alphabet = list(target.alphabet)
    idx = {c: i for i, c in enumerate(alphabet)}
    transitions = {
        q: {idx[c]: target.step(q, c) for c in alphabet}
        for q in range(target.num_states)
    }
    return AutDFA(
        states=set(range(target.num_states)),
        input_symbols=set(idx.values()),
        transitions=transitions,
        initial_state=target.start,
        final_states=set(target.accept),
        allow_partial=False,
    )


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
        return regime.report(taf_to_automata_dfa(self.target))
