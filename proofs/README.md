# Machine-checked correctness of the E-L\* clustering algorithm

- `TrainFamilyClaim` (`Claims/TrainFamily.lean`): on an oracle whose two classes' accept rates are
  at least `s₀` apart and a pool holding at least `β` of each class, but for chance `δ`, the
  clustering stops and returns a suffix family that cuts all but `εcov + ε_s` of the pool as the
  noiseless oracle does,
  every state's band reading its rarer side at most `κ` times as often as undecided, with its band
  width, suffix pool, prefix count and atom tolerances each within a stated big-O shape.
- `CounterexSynthClaim` (`Claims/CounterexSynth.lean`): but for chance `δ`, one L\* round whose
  reads are wrong at most `κ` times as often as undecided ends well within a stated big-O probe
  budget, harvest slope and slack, given stated bounds on how spread the walk starts and how small
  `κ` must be.
- `RLStarClaim` (`Claims/RLStar.lean`): on an oracle whose two classes' accept rates are at
  least `s₀` apart, over an input distribution `D` that puts at most a stated big-O `p₀Tol` on any
  first `k` letters, `k ≤ L` the trie depth and walk start, and whose strings of length `L` hold
  at least `β` of each class, but for chance `δ` the loop's DFA is right on
  all but `ξ` of `D`'s strings of length `L`, every round's tree has at most `|Q| + 2` leaves, and
  the whole run's prefixes, suffixes and probes are each within a stated big-O shape. But for
  chance `δ`, the run's output also depends on at most `totalReads` of the oracle's answers,
  `totalReads` within a stated big-O shape.
- `RLStarClaimUniform` (`Claims/RLStar.lean`): `RLStarClaim` over uniform strings, once `k` is past
  a stated big-O depth `k₀`, logarithmic in `L` and the parameters. The shapes of both claims take
  `ξc = min(ξ, β)` where they took `ξ`.
- `RLStarClaimUniformSimple` (`Claims/RLStar.lean`): `RLStarClaimUniform` with the prefix and
  suffix totals each a single product.
