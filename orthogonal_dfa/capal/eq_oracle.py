"""Random-word equivalence oracle.

CAPAL assumes a perfect EQ oracle. In the synthetic test setup we have access
to the ground-truth oracle (the noisy oracle re-instantiated with
SymmetricBernoulli(p_correct=1.0)), so we just sample random words and report
the first one where the hypothesis disagrees with the ground truth.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from automata.fa.dfa import DFA

from orthogonal_dfa.l_star.structures import Oracle


@dataclass
class RandomWordEqOracle:
    truth: Oracle
    alphabet_size: int
    num_walks: int = 5000
    min_walk_len: int = 1
    max_walk_len: int = 30
    seed: int = 0

    def find_counterexample(self, hypothesis: DFA) -> Optional[List[int]]:
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_walks):
            length = int(rng.integers(self.min_walk_len, self.max_walk_len + 1))
            word = rng.integers(0, self.alphabet_size, size=length).tolist()
            try:
                h_label = hypothesis.accepts_input(word)
            except Exception:
                # Partial DFA / missing transition: treat as disagreement.
                return word
            truth_label = self.truth.membership_query(list(word))
            if bool(h_label) != bool(truth_label):
                return word
        # Also test the empty word
        try:
            h_label = hypothesis.accepts_input([])
        except Exception:
            return []
        if bool(h_label) != bool(self.truth.membership_query([])):
            return []
        return None
