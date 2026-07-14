"""Observation table for CAPAL.

Holds:
- S: prefix-closed set of access strings (as tuples of ints).
- E_core: ordered list of must-use discriminators (suffixes).
- E_pool: large pool of short suffixes, used to give SAMESTATE more votes.
- y_cache: persistent membership labels (the noisy oracle is queried at most
  once per word; subsequent reads of the same word return the cached value).
- gold: trusted labels for CE words coming back from the perfect EQ oracle.
- same_state_neg: negative cache for the SAMESTATE test (we only cache
  "DIFFERENT" outcomes; see App. A.3.2).

SAMESTATE(u, v; p_0, alpha): noise-aware row-equality test using the empirical
disagreement rate D(u, v) over E_core u sub(E_pool) against threshold
p_0 + tau, with tau = min(tau_max, sqrt(ln(2/alpha) / (2m))).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from orthogonal_dfa.l_star.structures import Oracle

Word = Tuple[int, ...]


def _gen_short_suffixes(
    alphabet_size: int,
    max_len: int,
    long_len: int = 0,
    num_long: int = 0,
    rng_seed: int = 0,
) -> List[Word]:
    """All words up to length max_len plus optionally num_long randomly sampled
    words of length long_len -- the long samples are what discriminate states
    far apart in the residue graph (e.g., parities 2 and 5 in mod-9 only
    diverge on length-4+ suffixes with carefully chosen sums)."""
    out: List[Word] = [()]
    cur: List[Word] = [()]
    for _ in range(max_len):
        nxt: List[Word] = []
        for w in cur:
            for a in range(alphabet_size):
                nxt.append(w + (a,))
        out.extend(nxt)
        cur = nxt
    if num_long > 0 and long_len > max_len:
        rng = np.random.default_rng(rng_seed)
        seen = set(out)
        attempts = 0
        while (
            len([w for w in out if len(w) == long_len]) < num_long
            and attempts < num_long * 20
        ):
            w = tuple(int(x) for x in rng.integers(0, alphabet_size, size=long_len))
            if w not in seen:
                out.append(w)
                seen.add(w)
            attempts += 1
    return out


@dataclass
class ObservationTable:
    oracle: Oracle
    alphabet_size: int
    eta: float
    alpha: float = 0.01
    tau_max: float = 0.1
    pool_max_len: int = 6
    pool_long_len: int = 10
    pool_num_long: int = 40
    pool_cap: int = 256  # cap suffix budget per SAMESTATE call

    S: Set[Word] = field(default_factory=lambda: {()})
    E_core: List[Word] = field(default_factory=list)
    E_pool: List[Word] = field(init=False)
    y_cache: Dict[Word, bool] = field(default_factory=dict)
    gold: Dict[Word, bool] = field(default_factory=dict)
    same_state_neg: Dict[Tuple[Word, Word, int], bool] = field(default_factory=dict)
    mq_count: int = 0

    def __post_init__(self) -> None:
        self.E_pool = _gen_short_suffixes(
            self.alphabet_size,
            self.pool_max_len,
            long_len=self.pool_long_len,
            num_long=self.pool_num_long,
        )
        # p_0 = 2 eta (1 - eta) per the paper's noise floor.
        self.p0 = 2.0 * self.eta * (1.0 - self.eta)

    # -- membership labels ---------------------------------------------------

    def y(self, word: Word) -> bool:
        """Persistent noisy label, with CE gold overrides taking precedence."""
        if word in self.gold:
            return self.gold[word]
        if word not in self.y_cache:
            self.y_cache[word] = bool(self.oracle.membership_query(list(word)))
            self.mq_count += 1
        return self.y_cache[word]

    def set_gold(self, word: Word, label: bool) -> None:
        """Mark word's true label as `label` (from a CE returned by EQ).
        Subsequent y() calls for `word` return `label`, ignoring the noisy
        oracle's vote."""
        self.gold[word] = label

    # -- pool / E_core management -------------------------------------------

    def add_to_E_core(self, suffix: Word) -> None:
        if suffix not in self.E_core:
            self.E_core.append(suffix)

    def add_to_S(self, word: Word) -> None:
        """Insert word and all its prefixes into S."""
        for i in range(len(word) + 1):
            self.S.add(word[:i])

    # -- same-state test ----------------------------------------------------

    def _pool_set(self) -> List[Word]:
        """Capped sample of E_pool, excluding entries already in E_core."""
        used = set(self.E_core)
        pool = [e for e in self.E_pool if e not in used]
        budget = max(0, self.pool_cap - len(self.E_core))
        return pool[:budget]

    def _tau(self, m: int) -> float:
        if m <= 0:
            return self.tau_max
        return min(self.tau_max, math.sqrt(math.log(2.0 / self.alpha) / (2.0 * m)))

    def _core_disagreement(self, u: Word, v: Word) -> int:
        return sum(1 for e in self.E_core if self.y(u + e) != self.y(v + e))

    def same_state(self, u: Word, v: Word) -> bool:
        """SAMESTATE(u, v; p_0, alpha).

        The CAPAL paper computes one statistical disagreement rate over
        E_core u E_pool. In practice E_core's CE-derived entries are the
        actual minimal Myhill-Nerode discriminators while E_pool entries
        are random short strings that mostly *agree* even between truly
        different states, so naively pooling them dilutes the signal of a
        single rare-but-correct discriminator below tau.

        We split the test:
        - E_core disagreements: under exact (noiseless) MQs, each one is a
          witness that u !~ v. We threshold the per-entry disagreement rate
          against p_0 + tau(|E_core|). One disagreement in a single-entry
          E_core (p_0 = 0) already exceeds tau_max -- so noiseless cases work.
        - If E_core is "indecisive" (disagreement consistent with noise) we
          fall back to the Hoeffding test on the pool.
        Negative results are cached (keyed on the current E_core size).
        """
        if u == v:
            return True
        key_pair = (u, v) if u <= v else (v, u)
        key = (key_pair[0], key_pair[1], len(self.E_core))
        if key in self.same_state_neg:
            return False
        # E_core check.
        if self.E_core:
            disagree_core = self._core_disagreement(u, v)
            m_core = len(self.E_core)
            D_core = disagree_core / m_core
            threshold_core = self.p0 + self._tau(m_core)
            if D_core > threshold_core:
                self.same_state_neg[key] = True
                return False
        # Pool check.
        pool = self._pool_set()
        m_pool = len(pool)
        if m_pool == 0:
            return True
        disagree_pool = sum(1 for e in pool if self.y(u + e) != self.y(v + e))
        D_pool = disagree_pool / m_pool
        threshold_pool = self.p0 + self._tau(m_pool)
        if D_pool > threshold_pool:
            self.same_state_neg[key] = True
            return False
        return True
