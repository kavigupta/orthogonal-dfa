# Machine-checked correctness of the E-L\* clustering algorithm

- `ClusteringClaim` (`Claims/Clustering.lean`): but for chance `δ`, the clustering stops and
  returns a suffix family that cuts all but `εcov + ε_s` of the pool as the noiseless oracle does,
  every state's band reading its rarer side at most `κ` times as often as undecided, with its band
  width, suffix pool, prefix count and atom tolerances each within a stated big-O shape.
- `ClusteringClaimSimple` (`Claims/Clustering.lean`): `ClusteringClaim` with each shape a single
  product.
- `Level2Claim` (`Claims/Levels.lean`): but for chance `δ`, one level-2 L\* round ends well within
  a probe budget of `O(|Q||Σ|/ε · log(|Q||Σ|/δ))`.
- `Level3Claim`, `Level4Claim` and `Level5Claim` (`Claims/Levels.lean`): but for chance `δ`, one L\*
  round at that level ends well within a stated big-O probe budget, harvest slope and slack, levels
  4 and 5 also stating how spread the walk starts and how rarely wrong the reads must be.
- `LoopClaim` (`Claims/Loop.lean`): over an input distribution `D` and a target admitted by the
  relative rule at visit ratio `R`, but for chance `δ` the loop's DFA is right on all but `γ + ξ`
  of `D`'s strings of length `L`, every round's tree has at most `|Q| + 2` leaves, and the whole
  run's prefixes, suffixes and probes are each within a stated big-O shape.
- `LoopClaimUniform` (`Claims/Loop.lean`): `LoopClaim` over uniform strings, once `L` is past a
  stated big-O length `L₀`.
- `LoopClaimSimple` (`Claims/Loop.lean`): `LoopClaimUniform` with the prefix and suffix totals each
  a single product.
