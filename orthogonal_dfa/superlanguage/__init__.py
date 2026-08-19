"""Superlanguages: learn a DFA over an alphabet of frequent kmers plus a
wildcard, lifted from a base-alphabet oracle.

The pieces compose:

* :class:`KmerVocabulary` -- the ``K`` kmers + ``X`` super alphabet, its uniform
  null emission distribution, and the nondeterministic compilation back to the
  base alphabet.
* :class:`SuperSampler` -- draws super-strings for the learner.
* :class:`LiftedOracle` -- answers super-string membership by compiling to the
  base alphabet and majority-voting a base oracle over several compilations.
* :func:`learn_superlanguage` / :func:`learn_superlanguage_from_corpus` -- wire
  those into E-L* and return ``(dfa, vocabulary)``.
"""

from .learn import learn_superlanguage, learn_superlanguage_from_corpus
from .oracle import LiftedOracle
from .sampler import SuperSampler
from .vocabulary import KmerVocabulary

__all__ = [
    "KmerVocabulary",
    "SuperSampler",
    "LiftedOracle",
    "learn_superlanguage",
    "learn_superlanguage_from_corpus",
]
