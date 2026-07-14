"""CAPAL: Class-query Active, Persistent-noise-Aware Learning.

Reimplementation of Algorithm 1 from Chen, Trivedi, Velasquez (ICLR 2026),
"Towards Persistent Noise-Tolerant Active Learning of Regular Languages with
Class Query". The authors' GitHub repo (github.com/lkwargs/CAPAL) was empty as
of 2026-06-17, so this module follows the paper directly.

Scope vs. the paper:

- We implement Algorithm 1, the SAMESTATE test of App. A.3.2, the
  Rivest-Schapire decomposition with label-only-CE gating of App. A.3.3, and
  the LCA discriminator / closedness / consistency loop.

- We omit the LLM-specific prompt strategies of Sections 5.1 / A.5 and the
  bootstrap of eta-hat: this module is meant for the synthetic-noise oracles
  in orthogonal_dfa/l_star/examples/, where the true noise rate is known.

- The equivalence oracle is approximated by random sampling against a perfect
  ground-truth oracle (the same oracle_creator instantiated with
  SymmetricBernoulli(p_correct=1.0)).
"""

from .capal import CAPAL, run_capal
from .eq_oracle import RandomWordEqOracle
