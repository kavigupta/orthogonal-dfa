import OrthoDFA.Proofs.LoopDFAArith
import OrthoDFA.Proofs.LoopQuery

/-!
# The strings the learner asks, as a real bound

`learnerReads` is `K` rounds of `roundCallBound` and the start leaf's `n f` reads, split by
source, `3 + ⌈log₂ L⌉` standing for the sifts of a probe (`reads_le_learnerReads`). `loopSC` and
`loopHN` bound `LearnerLoopDFASimple`'s scan and harvest caps at `ℓ = L₁ + 1`, the prefix count
`RB` at `loopCount`.
-/

noncomputable section

namespace OrthoDFA

/-- The clusterings' reads: `K` rounds of `a` prefixes against `b` suffixes and the empty one. -/
def readsCluster (K a b : ℝ) : ℝ := K * (a * (b + 1))

/-- Level 3's reads: `K T` probes, each `lg` sifts down a tree of at most `q + 2` leaves and a
retarget over `A` letters, a family read of `f` strings at each. -/
def readsSearch (K f T q lg A : ℝ) : ℝ := K * T * (f * (lg * (q + 1) + (q + 2) * A * (q + 2)))

/-- The samplers' reads: `K` rounds of `K` populations' `M` draws, each walking at most `N`
probes down a tree of at most `q + 2` leaves. -/
def readsHarvest (K f q M N lg : ℝ) : ℝ := K * (K * (f * (M * (N * (lg * (q + 1))))))

/-- The start leaf's reads: `n` held-out strings, a family read each. -/
def readsStart (n f : ℝ) : ℝ := n * f

def learnerReads (K a b f T q M N lg A n : ℝ) : ℝ :=
  readsCluster K a b + readsSearch K f T q lg A + readsHarvest K f q M N lg + readsStart n f

lemma learnerReads_mono {K a b f T q M N lg A n K' a' b' f' T' q' M' N' lg' A' n' : ℝ}
    (hK0 : 0 ≤ K) (ha0 : 0 ≤ a) (hb0 : 0 ≤ b) (hf0 : 0 ≤ f) (hT0 : 0 ≤ T) (hq0 : 0 ≤ q)
    (hM0 : 0 ≤ M) (hN0 : 0 ≤ N) (hlg0 : 0 ≤ lg) (hA0 : 0 ≤ A) (hn0 : 0 ≤ n)
    (hK : K ≤ K') (ha : a ≤ a') (hb : b ≤ b') (hf : f ≤ f') (hT : T ≤ T') (hq : q ≤ q')
    (hM : M ≤ M') (hN : N ≤ N') (hlg : lg ≤ lg') (hA : A ≤ A') (hn : n ≤ n') :
    learnerReads K a b f T q M N lg A n ≤ learnerReads K' a' b' f' T' q' M' N' lg' A' n' := by
  have : 0 ≤ K' := hK0.trans hK
  have : 0 ≤ f' := hf0.trans hf
  have : 0 ≤ T' := hT0.trans hT
  have : 0 ≤ q' := hq0.trans hq
  have : 0 ≤ A' := hA0.trans hA
  have : 0 ≤ lg' := hlg0.trans hlg
  have : 0 ≤ M' := hM0.trans hM
  have : 0 ≤ N' := hN0.trans hN
  unfold learnerReads readsCluster readsSearch readsHarvest readsStart
  gcongr
  all_goals first | positivity | linarith

lemma reads_le_learnerReads (K a b f T M N L A n q : ℕ) :
    ((K * roundCallBound K a b f T M N L A q + n * f : ℕ) : ℝ)
      ≤ learnerReads K a b f T q M N ((3 + Nat.clog 2 L : ℕ) : ℝ) A n := by
  have hp : Random.Kappa.probeSifts L ≤ 3 + Nat.clog 2 L := min_le_right _ _
  have hp' : (Random.Kappa.probeSifts L : ℝ) ≤ ((3 + Nat.clog 2 L : ℕ) : ℝ) := by exact_mod_cast hp
  unfold roundCallBound learnerReads readsCluster readsSearch readsHarvest readsStart
  push_cast at hp' ⊢
  have h : (K : ℝ) * ((f : ℝ) * (T * (Random.Kappa.probeSifts L * (q + 1) + (q + 2) * A * (q + 2))))
      ≤ K * T * (f * ((3 + Nat.clog 2 L) * (q + 1) + (q + 2) * A * (q + 2))) := by
    rw [show (K : ℝ) * T * (f * ((3 + Nat.clog 2 L) * (q + 1) + (q + 2) * A * (q + 2)))
      = K * (f * (T * ((3 + Nat.clog 2 L) * (q + 1) + (q + 2) * A * (q + 2)))) by ring]
    gcongr
  linarith [h]

/-- `LearnerLoopDFASimple`'s scan cap, its prefix count at `loopCount ℓ`. -/
def loopSC (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  8 * (Q + 4) * loopCount ℓ Q A Ae δ c ξ pAP η₀
    + 16 * Real.log (16 * (Q + 2) / (δ / (8 * (Q + 1)))) + 2

/-- `LearnerLoopDFASimple`'s harvest cap, its window's `N` lengths at most `ℓ` and its scan cap at
`loopSC ℓ`. -/
def loopHN (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  192 * (4 * (Ae + 1) * ℓ / c)
      * Real.log (4 * (Q + 1) * loopSC ℓ Q A Ae δ c ξ pAP η₀ / (δ / (8 * (Q + 1)))) + 1

end OrthoDFA

end
