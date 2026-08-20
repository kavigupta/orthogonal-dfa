"""Superlanguages: alphabets built on top of a base alphabet.

:class:`KmerVocabulary` is the alphabet itself -- ``K`` frequent kmers plus
interchangeable wildcards -- together with the invertible, uniformity-preserving
translation to and from the base alphabet.
"""

from .vocabulary import KmerVocabulary

__all__ = ["KmerVocabulary"]
