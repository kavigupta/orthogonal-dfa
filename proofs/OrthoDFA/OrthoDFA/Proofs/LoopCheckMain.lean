import OrthoDFA.LoopCheckClaim
import OrthoDFA.Proofs.LoopCheckLaws

/-!
# The checked loop's first round: the proof at explicit settings

The schedule is `clustering_schedule_uni`'s at gate level and share `δc` and slack `ι`, with the
suffixes' class-preserving share `pAP/2`, which the suffixes off what the earlier rounds read keep;
its counts are at most `PB` suffixes and `RB` prefixes per state (`counts_le`). The probes' atoms
and prefix masses are at most `|Σ|^(−k)` and the suffixes' `|Σ|^(−Ls)`, the uniform pool gives each
covered state `c/(2N)`, and the strings of length `L` are half the pool, so `check_first_core`
applies at `u = 14 ι N / c`.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Every population's law off `T` holds no string at least `Ls` long. -/
lemma popOff_short {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (pops : List (Measure (FreeMonoid α))) (T : Finset (FreeMonoid α)) {A : Set (FreeMonoid α)}
    (hD : D A = 0) (hpops : ∀ ν ∈ pops, IsProbabilityMeasure ν ∧ ν A = 0)
    (hT : D.real ↑T ≤ 1 / 2) (j : Option (Fin K)) (hj : j ∈ loopPopulations K pops.length) :
    popOff D pops j T A = 0 := by
  have hprob : IsProbabilityMeasure (loopDistributions (K := K) D pops j) :=
    isProbabilityMeasure_loopDistributions D pops (fun ν hν => (hpops ν hν).1) j
  unfold popOff
  split_ifs with hdeg
  · exact offLaw_null (compl_ne_zero hT) hD
  · refine offLaw_null hdeg ?_
    cases j with
    | none => exact hD
    | some i =>
      have hi : i.val < pops.length := lt_of_some_mem_loopPops hj
      show (pops.getD i (Measure.dirac 1)) A = 0
      rw [List.getD_eq_getElem (hn := hi)]
      exact (hpops _ (List.getElem_mem _)).2

theorem learnerLoopCheckSimple_holds : LearnerLoopCheckSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L T₀ η₀ c δ pAP εcov ε Q A k N ℓ a Ls s S δ₃ δc T n Ms h ι τ J W
    PB RB Kt NK hL hη hη₀ hc0 hc1 hδ0 hδ1 hpAP hεc0 hεc1 hε0 hε1 hA2 hpAPu hpre hkQ hQL hhL h1 h2
    h3 h4 h5 h6 h7 h8
  classical
  have hQd : Q = Fintype.card σ := rfl
  have hAd : A = Fintype.card α := rfl
  have hkd : k = (L + 1) / 2 := rfl
  have hNd : N = L + 1 - k := rfl
  have hℓd : ℓ = L + 1 := rfl
  have had : a = k - Fintype.card σ - 1 := rfl
  have hLsd : Ls = L + 1 := rfl
  have hsd : s = 1 / 2 - η₀ := rfl
  have hSd : S = Q * ((Q + 2) * A + 1) + 2 * A + 1 := rfl
  have hδ₃d : δ₃ = δ / 4 := rfl
  have hδcd : δc = δ / 16 := rfl
  have hTd : T = ⌈36 * S / ε * Real.log (16 * S * (S + 1) / δ₃) + S⌉₊ := rfl
  have hnd : n = ⌈16 / (3 * ε) * Real.log (86 * S * (S + 1) * (S + 2) / δ ^ 2)⌉₊ := rfl
  have hMsd : Ms = (Q + 1) * A ^ Fintype.card σ := rfl
  have hhd : h = Nat.clog (Fintype.card α) ⌈16 * Real.log (64 / δ) / ε⌉₊ := rfl
  have hιd : ι = min (c * δ / (896 * (h + 1) * A ^ h * Ms * ℓ))
      (min (c / (28 * (Q + 1) * (288 / ε + 16 * ℓ) * ℓ)) (c * ε / (896 * Ms * ℓ ^ 2))) := rfl
  have hτd : τ = min (εcov * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι := rfl
  have hJd : J = Q + 2 := rfl
  have hWd : W = T * ((L + 1) * Q + n * ((L + 1) * Q) + (n + 1) * ((Q + 1) ^ 2 * A)) := rfl
  have hPBd : PB = (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ₃)) / s ^ 2 + 20 / s + 28)
      / pAP + 64 * Real.log ((J + 2) / (δc / 2)) / pAP ^ 2 := rfl
  have hRBd : RB = 64000 * J ^ 2 * Real.log ((J + 2) * PB / (δc / 2 * δc * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 2) := rfl
  have hKtd : Kt = (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) := rfl
  have hNKd : NK = 6 * J * RB := rfl
  clear_value Q A k N ℓ a Ls s S δ₃ δc T n Ms h ι τ J W PB RB Kt NK
  -- the basic quantities
  have hA : (2 : ℝ) ≤ A := by rw [hAd]; exact_mod_cast hA2
  have hA1 : (1 : ℝ) ≤ A := by linarith only [hA]
  have hA0 : (0 : ℝ) < A := by linarith only [hA]
  have hQ0 : (0 : ℝ) ≤ Q := by rw [hQd]; exact Nat.cast_nonneg _
  have hkL : k ≤ L := by rw [hkd]; omega
  have hk1 : 1 ≤ k := by omega
  have haQ : a + Fintype.card σ + 1 = k := by rw [had]; omega
  have ha1 : 1 ≤ a := by omega
  have haL : a ≤ L := by omega
  have hN1 : (1 : ℝ) ≤ N := by
    rw [hNd]
    have : (k : ℝ) ≤ L := by exact_mod_cast hkL
    linarith only [this]
  have hN0 : (0 : ℝ) < N := by linarith only [hN1]
  have hNℓ : N ≤ ℓ := by
    rw [hNd, hℓd]
    have : (0 : ℝ) ≤ k := Nat.cast_nonneg _
    linarith only [this]
  have hℓ1 : (1 : ℝ) ≤ ℓ := by rw [hℓd]; linarith only [(Nat.cast_nonneg L : (0 : ℝ) ≤ L)]
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hs0 : 0 < s := by rw [hsd]; linarith only [hη₀]
  have hpAP1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hδ₃0 : 0 < δ₃ := by rw [hδ₃d]; positivity
  have hδc0 : 0 < δc := by rw [hδcd]; positivity
  have hδc1 : δc ≤ 1 := by rw [hδcd]; linarith only [hδ1]
  have hδch : δc < 1 / 2 := by rw [hδcd]; linarith only [hδ1]
  have hMs1 : (1 : ℝ) ≤ Ms := by
    rw [hMsd]
    have : (1 : ℝ) ≤ A ^ Fintype.card σ := one_le_pow₀ hA1
    exact one_le_mul_of_one_le_of_one_le (by linarith only [hQ0]) this
  have hAh1 : (1 : ℝ) ≤ A ^ h := one_le_pow₀ hA1
  -- the indecision limit
  have hι0 : 0 < ι := by
    rw [hιd]
    refine lt_min (by positivity) (lt_min (by positivity) (by positivity))
  have hιa : ι ≤ c * δ / (896 * (h + 1) * A ^ h * Ms * ℓ) := hιd ▸ min_le_left _ _
  have hιb : ι ≤ c / (28 * (Q + 1) * (288 / ε + 16 * ℓ) * ℓ) :=
    hιd ▸ (min_le_right _ _).trans (min_le_left _ _)
  have hιc : ι ≤ c * ε / (896 * Ms * ℓ ^ 2) := hιd ▸ (min_le_right _ _).trans (min_le_right _ _)
  have hι1 : ι ≤ 1 / 2 := by
    refine hιc.trans ?_
    rw [div_le_iff₀ (by positivity)]
    have : (1 : ℝ) ≤ ℓ ^ 2 := one_le_pow₀ hℓ1
    have hm : 1 ≤ Ms * ℓ ^ 2 := one_le_mul_of_one_le_of_one_le hMs1 this
    have hce : c * ε ≤ 1 := by nlinarith only [hc1, hε1, hc0, hε0]
    linarith only [hm, hce]
  have hτ : rateScale η₀ ι εcov = τ := by rw [hτd]; rfl
  have hτb : budgetScale η₀ ι εcov = τ := by
    show min (min εcov (1 / 2) * gateA η₀ ^ 2) ι = τ
    rw [min_eq_left hεc1, hτd]
    rfl
  have hτ0 : 0 < τ := by
    have ha' : 0 < (1 - 2 * η₀) / (1 - η₀) := gateA_pos hη0 hη₀
    rw [hτd]
    exact lt_min (mul_pos hεc0 (pow_pos ha' 2)) hι0
  have hτι : τ ≤ ι := hτd ▸ min_le_right _ _
  -- level 3's budget, the check's size and the head's length
  have hS1 : (1 : ℝ) ≤ S := by
    rw [hSd]
    have hq' : 0 ≤ Q * ((Q + 2) * A + 1) := mul_nonneg hQ0 (by nlinarith only [hQ0, hA1])
    linarith only [hq', hA1]
  have hSe : ((Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
      + 2 * Fintype.card α + 1 : ℕ) : ℝ) = S := by rw [hSd, hQd, hAd]; push_cast; ring
  have hT : 36 * S / ε * Real.log (16 * S * (S + 1) / δ₃) + S ≤ T := hTd ▸ Nat.le_ceil _
  have hlogT : 0 ≤ Real.log (16 * S * (S + 1) / δ₃) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ₃0, hδ₃d]; nlinarith only [hS1, hδ1])
  have hT1 : (1 : ℝ) ≤ T := by
    have : 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ₃) :=
      mul_nonneg (div_nonneg (by linarith only [hS1]) hε0.le) hlogT
    linarith only [this, hT, hS1]
  have hn : 16 / (3 * ε) * Real.log (86 * S * (S + 1) * (S + 2) / δ ^ 2) ≤ n := hnd ▸ Nat.le_ceil _
  have hAh : 16 * Real.log (64 / δ) / ε ≤ A ^ h := by
    have h1' := Nat.le_ceil (16 * Real.log (64 / δ) / ε)
    have h2' := Nat.le_pow_clog (show 1 < Fintype.card α by omega)
      ⌈16 * Real.log (64 / δ) / ε⌉₊
    have : (⌈16 * Real.log (64 / δ) / ε⌉₊ : ℝ) ≤ A ^ h := by
      rw [hAd, hhd]
      exact_mod_cast h2'
    linarith only [this, h1']
  -- the wrong reads' rate
  have hW0 : 0 < W := by
    rw [hWd]
    have : 0 < (n + 1 : ℝ) * ((Q + 1) ^ 2 * A) := by positivity
    have : (0 : ℝ) ≤ (L + 1) * Q + n * ((L + 1) * Q) := by positivity
    exact mul_pos (by linarith only [hT1]) (by linarith only [this, ‹0 < (n + 1 : ℝ) * _›])
  set crossLimit := δ₃ / (8 * W) with hcl
  have hcross : 0 < crossLimit := by positivity
  sorry

end OrthoDFA
