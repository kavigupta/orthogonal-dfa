import OrthoDFA.Proofs.LoopDFAArith
import OrthoDFA.Proofs.LoopQuery

/-!
# The strings the loop asks, as a real bound

`loopReads K a b T M N L A` is `roundCallBound` over `K` rounds with `3 + ⌈log₂ L⌉` and
`probeSifts L` relaxed to `L + 3` and `L + 1` (`reads_le_loopReads`). `loopSC` and `loopHN` bound
`LearnerLoopDFASimple`'s scan and harvest caps at `ℓ = L₁ + 1`, the prefix count `RB` at
`loopCount`.
-/

noncomputable section

namespace OrthoDFA

/-- The strings `K` rounds ask, each round's clustering reading at most `a` prefixes and `b` suffixes,
at probe budget `T`, scan cap `M`, harvest cap `N`, probes at most `L` long and `A` letters. -/
def loopReads (K a b T M N L A : ℝ) : ℝ :=
  K * (K * (b * (M * (N * ((L + 3) * (T + 1))))) + a * (b + 1)
    + b * (T * ((L + 1) * (T + 1) + (T + 2) * A * (T + 2))))

lemma loopReads_mono {K a b T M N L A K' a' b' T' M' N' L' A' : ℝ} (hK0 : 0 ≤ K) (ha0 : 0 ≤ a)
    (hb0 : 0 ≤ b) (hT0 : 0 ≤ T) (hM0 : 0 ≤ M) (hN0 : 0 ≤ N) (hL0 : 0 ≤ L) (hA0 : 0 ≤ A)
    (hK : K ≤ K') (ha : a ≤ a') (hb : b ≤ b') (hT : T ≤ T') (hM : M ≤ M') (hN : N ≤ N')
    (hL : L ≤ L') (hA : A ≤ A') :
    loopReads K a b T M N L A ≤ loopReads K' a' b' T' M' N' L' A' := by
  have : 0 ≤ T' := hT0.trans hT
  have : 0 ≤ A' := hA0.trans hA
  unfold loopReads
  gcongr
  all_goals linarith

lemma reads_le_loopReads (K a b T M N L A : ℕ) :
    ((K * roundCallBound K a b T M N L A : ℕ) : ℝ) ≤ loopReads K a b T M N L A := by
  have hc : 3 + Nat.clog 2 L ≤ L + 3 := by
    have := Nat.clog_le_of_le_pow (b := 2) (x := L) (y := L) Nat.lt_two_pow_self.le
    omega
  have hp : Random.Kappa.probeSifts L ≤ L + 1 := min_le_left _ _
  have hc' : ((3 + Nat.clog 2 L : ℕ) : ℝ) ≤ (L : ℝ) + 3 := by exact_mod_cast hc
  have hp' : (Random.Kappa.probeSifts L : ℝ) ≤ (L : ℝ) + 1 := by exact_mod_cast hp
  unfold roundCallBound loopReads
  push_cast at hc' hp' ⊢
  gcongr

/-- `LearnerLoopDFASimple`'s scan cap, its prefix count at `loopCount ℓ`. -/
def loopSC (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  8 * (Q + 4) * loopCount ℓ Q A Ae δ c ξ pAP η₀
    + 16 * Real.log (16 * (Q + 2) / (δ / (8 * (Q + 1)))) + 2

/-- `LearnerLoopDFASimple`'s harvest cap, its window's `N` lengths at most `ℓ`, its probe budget
at `loopBudget ℓ` and its scan cap at `loopSC ℓ`. -/
def loopHN (ℓ Q A Ae δ c ξ pAP η₀ : ℝ) : ℝ :=
  ℓ + loopBudget ℓ Q A Ae δ c + 3
    + 192 * (4 * (Ae + 1) * ℓ / c)
      * Real.log (4 * (Q + 1) * loopSC ℓ Q A Ae δ c ξ pAP η₀ / (δ / (8 * (Q + 1))))

end OrthoDFA

end
