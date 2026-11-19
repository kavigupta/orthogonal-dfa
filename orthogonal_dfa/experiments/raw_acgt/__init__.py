"""The flat length oracle learned directly over length-40 ACGT strings (0=A 1=C 2=G 3=T),
with no superlanguage, then residual stages on top.  From the repo root:

    python -u -m orthogonal_dfa.experiments.raw_acgt.run --seed S         # S in 0..3
    python -u -m orthogonal_dfa.experiments.raw_acgt.residual --seed S
    python -m orthogonal_dfa.experiments.raw_acgt.collect

``run`` and ``residual`` need a GPU; their results are checked in under
data/neural/raw_acgt.  ``analyze``, ``features``, ``states`` and ``sanity`` print the
numbers reported about them."""
