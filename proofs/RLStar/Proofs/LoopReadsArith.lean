import RLStar.Proofs.LoopDFAArith
import RLStar.Proofs.LoopQuery

/-!
# The strings the learner asks, as a real bound

`lazyReads` is `K` rounds of `roundCallBound` and the last round's table (`reads_le_lazyReads`).
`learnerReads` is main's count, split by source, which the asymptotic files still bound.
`loopSC` and `loopHB` bound `LearnerLoopDFASimple`'s scan cap and the probes a round's harvested draws walk at
`ℓ = L₁ + 1`, the prefix count `RB` at `loopCount`, and `loopSH` the draws a harvested
population makes in a round.
-/

noncomputable section

namespace OrthoDFA

/-- The clusterings' reads: `K` rounds of `a` prefixes against `b` suffixes and the empty one. -/
def readsCluster (K a b : ℝ) : ℝ := K * (a * (b + 1))

/-- Level 3's probes: `K T` probes, each `lg` sifts down a tree of at most `q + 2` leaves, a
family read of `f` strings at each. -/
def readsSearch (K f T q lg : ℝ) : ℝ := K * T * (f * (lg * (q + 1)))

/-- Level 3's re-sifts: in each of `K` rounds at most `q` splits, each re-sifting at most
`q + 2` leaves' edges over `A` letters down at most `q + 2` midfixes, a family read of `f` strings
at each. -/
def readsResift (K f q A : ℝ) : ℝ := K * (f * (q * ((q + 2) * A * (q + 2))))

/-- The samplers' reads: `K` rounds of at most `B` probes walked by the harvested draws, each
`lg` sifts down a tree of at most `q + 2` leaves, a family read of `f` strings at each. -/
def readsHarvest (K f q B lg : ℝ) : ℝ := K * (f * (B * (lg * (q + 1))))

/-- The start leaf's reads: `n` held-out strings, a family read each. -/
def readsStart (n f : ℝ) : ℝ := n * f

def learnerReads (K a b f T q B lg A n : ℝ) : ℝ :=
  readsCluster K a b + readsSearch K f T q lg + readsResift K f q A + readsHarvest K f q B lg
    + readsStart n f

lemma learnerReads_mono {K a b f T q B lg A n K' a' b' f' T' q' B' lg' A' n' : ℝ}
    (hK0 : 0 ≤ K) (ha0 : 0 ≤ a) (hb0 : 0 ≤ b) (hf0 : 0 ≤ f) (hT0 : 0 ≤ T) (hq0 : 0 ≤ q)
    (hB0 : 0 ≤ B) (hlg0 : 0 ≤ lg) (hA0 : 0 ≤ A) (hn0 : 0 ≤ n)
    (hK : K ≤ K') (ha : a ≤ a') (hb : b ≤ b') (hf : f ≤ f') (hT : T ≤ T') (hq : q ≤ q')
    (hB : B ≤ B') (hlg : lg ≤ lg') (hA : A ≤ A') (hn : n ≤ n') :
    learnerReads K a b f T q B lg A n ≤ learnerReads K' a' b' f' T' q' B' lg' A' n' := by
  have : 0 ≤ K' := hK0.trans hK
  have : 0 ≤ f' := hf0.trans hf
  have : 0 ≤ T' := hT0.trans hT
  have : 0 ≤ q' := hq0.trans hq
  have : 0 ≤ A' := hA0.trans hA
  have : 0 ≤ lg' := hlg0.trans hlg
  have : 0 ≤ B' := hB0.trans hB
  unfold learnerReads readsCluster readsSearch readsResift readsHarvest readsStart
  gcongr
  all_goals first | positivity | linarith

/-- The lazy learner's reads: `K` rounds of `a` prefixes against `b` suffixes and the empty one
and of a lazy round's `Nr` strings, each a family read of `f`, and the table's sifts, at most
`q + 2` questions for each of the `A nX + 1` frontier strings of a trie of `nX` strings. -/
def lazyReads (K a b f Nr A nX q : ℝ) : ℝ :=
  K * (a * (b + 1) + f * Nr) + f * ((A * nX + 1) * (q + 2))

lemma reads_le_lazyReads (K a b f Nr A nX q : ℕ) :
    ((K * roundCallBound a b f Nr + f * ((A * nX + 1) * (q + 2)) : ℕ) : ℝ)
      ≤ lazyReads K a b f Nr A nX q :=
  le_of_eq (by unfold roundCallBound lazyReads; push_cast; ring)

/-- `LearnerLoopDFASimple`'s scan cap, its prefix count at `loopCount ℓ`. -/
def loopSC (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  8 * (Q + 4) * loopCount ℓ Q A Ae δ c ξ pAP η₀
    + 16 * Real.log (16 * (Q + 2) / (δ / (8 * (Q + 1)))) + 2

/-- The draws a harvested population of `LearnerLoopDFASimple` makes in a round, its prefix count
at `loopCount ℓ`. -/
def loopSH (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  24 * loopCount ℓ Q A Ae δ c ξ pAP η₀ + 16 * Real.log (16 * (Q + 2) / (δ / (8 * (Q + 1)))) + 2

/-- The probes a round of `LearnerLoopDFASimple`'s harvested draws walk, its window's `N` lengths
at most `ℓ` and each harvested population's draws at most `loopSH ℓ`. -/
def loopHB (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  384 * (4 * (Ae + 1) * ℓ / c) * ((Q + 1) * loopSH ℓ Q A Ae δ c ξ pAP η₀ * Real.log 2
    + Real.log (2 / (δ / (8 * (Q + 1))))) + 1

end OrthoDFA

end
