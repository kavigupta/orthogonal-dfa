# Machine-checked correctness of the E-L\* clustering algorithm

- `TrainFamilyClaim` (`Claims/TrainFamily.lean`): but for chance `δ`, the clustering stops and
  returns a suffix family that cuts all but `εcov + ε_s` of the pool as the noiseless oracle does,
  every state's band reading its rarer side at most `κ` times as often as undecided, with its band
  width, suffix pool, prefix count and atom tolerances each within a stated big-O shape.
- `CounterexSynthClaim` (`Claims/CounterexSynth.lean`): but for chance `δ`, one L\* round whose
  reads are wrong at most `κ` times as often as undecided ends well within a stated big-O probe
  budget, harvest slope and slack, given stated bounds on how spread the walk starts and how small
  `κ` must be.
- `RLStarClaim` (`Claims/RLStar.lean`): over an input distribution `D` that puts at most a
  stated big-O `p₀Tol` on any first `⌈L/2⌉` letters, but for chance `δ` the loop's DFA is right on
  all but `ξ` of `D`'s strings of length `L`, every round's tree has at most `|Q| + 2` leaves, and
  the whole run's prefixes, suffixes and probes are each within a stated big-O shape. But for
  chance `δ`, the run's output also depends on at most `totalReads` of the oracle's answers,
  `totalReads` within a stated big-O shape.
- `RLStarClaimUniform` (`Claims/RLStar.lean`): `RLStarClaim` over uniform strings, once `L` is past
  a stated big-O length `L₀`.
- `RLStarClaimUniformSimple` (`Claims/RLStar.lean`): `RLStarClaimUniform` with the prefix and
  suffix totals each a single product.
