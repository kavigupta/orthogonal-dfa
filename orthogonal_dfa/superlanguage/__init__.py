"""Superlanguages: learn a DFA over an alphabet of frequent kmers plus a
wildcard, lifted from a base-alphabet oracle.

The pieces compose:

* :class:`KmerVocabulary` -- the ``K`` prefix-free kmers + ``X`` super alphabet,
  with an invertible, distribution-preserving compilation to/from the base
  alphabet (``parse``/``compile``).
* :class:`SuperSampler` -- draws super-strings for the learner (parse of a
  uniform base stream).
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
