import OrthoDFA.Proofs.FamilyRead
import Mathlib.Probability.Moments.SubGaussian

/-!
# A wide band passes its check

The vote is Poisson-binomial over `N` bits. The mean is at least `(hi + 1 − lo)/2` from one of the
band's sides, and Hoeffding's inequality holds that side's read to
`exp (-(hi + 1 − lo)² / (2N))`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

/-- The Poisson-binomial law of independent bits at rates `ps`. -/
def pbLaw (ps : List ℝ) : ℕ → ℝ := ps.foldr pbStep pbBase

lemma pbLaw_cons (p : ℝ) (ps : List ℝ) : pbLaw (p :: ps) = pbStep p (pbLaw ps) := rfl

lemma pbLaw_nonneg : ∀ (ps : List ℝ), (∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) → ∀ j, 0 ≤ pbLaw ps j
  | [], _, j => by
    simp only [pbLaw, List.foldr_nil, pbBase]
    split_ifs <;> norm_num
  | p :: ps, h, j => by
    have hp := h p List.mem_cons_self
    have ih := pbLaw_nonneg ps (fun q hq => h q (List.mem_cons_of_mem _ hq))
    rw [pbLaw_cons]
    cases j with
    | zero => simp only [pbStep]; exact mul_nonneg (by linarith) (ih 0)
    | succ j =>
      simp only [pbStep]
      exact add_nonneg (mul_nonneg (by linarith) (ih _)) (mul_nonneg hp.1 (ih j))

lemma pbLaw_zero_of_lt : ∀ (ps : List ℝ) (j : ℕ), ps.length < j → pbLaw ps j = 0
  | [], j, h => by
    have hj : j ≠ 0 := by simp only [List.length_nil] at h; omega
    simp [pbLaw, pbBase, hj]
  | p :: ps, j, h => by
    rw [pbLaw_cons]
    simp only [List.length_cons] at h
    cases j with
    | zero => omega
    | succ j =>
      simp only [pbStep]
      rw [pbLaw_zero_of_lt ps (j + 1) (by omega), pbLaw_zero_of_lt ps j (by omega)]
      ring

/-- `E[g(X)]` for `X` of law `pbLaw ps`. -/
def pbMean (ps : List ℝ) (g : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (ps.length + 1), g j * pbLaw ps j

lemma pbMean_cons (p : ℝ) (ps : List ℝ) (g : ℕ → ℝ) :
    pbMean (p :: ps) g = (1 - p) * pbMean ps g + p * pbMean ps (fun j => g (j + 1)) := by
  unfold pbMean
  rw [List.length_cons, Finset.sum_range_succ', pbLaw_cons]
  simp only [pbStep]
  have h1 : ∑ j ∈ Finset.range (ps.length + 1),
        g (j + 1) * ((1 - p) * pbLaw ps (j + 1) + p * pbLaw ps j)
      = (1 - p) * ∑ j ∈ Finset.range (ps.length + 1), g (j + 1) * pbLaw ps (j + 1)
        + p * ∑ j ∈ Finset.range (ps.length + 1), g (j + 1) * pbLaw ps j := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [h1]
  have h2 := Finset.sum_range_succ' (fun j => g j * pbLaw ps j) (ps.length + 1)
  have h3 := Finset.sum_range_succ (fun j => g j * pbLaw ps j) (ps.length + 1)
  rw [pbLaw_zero_of_lt ps (ps.length + 1) (by omega), mul_zero, add_zero] at h3
  linear_combination (1 - p) * (h3.symm.trans h2).symm

/-- Hoeffding's lemma for one bit at rate `p`. -/
lemma bit_mgf_le {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (t : ℝ) :
    1 - p + p * Real.exp t ≤ Real.exp (p * t + t ^ 2 / 8) := by
  set ν : Measure ℝ := ENNReal.ofReal (1 - p) • Measure.dirac 0
    + ENNReal.ofReal p • Measure.dirac 1 with hν
  have : IsProbabilityMeasure ν := by
    constructor
    simp only [hν, Measure.add_apply, Measure.smul_apply, measure_univ, smul_eq_mul, mul_one]
    rw [← ENNReal.ofReal_add (by linarith) hp0]
    simp
  have hint : ∀ g : ℝ → ℝ, ∫ x, g x ∂ν = (1 - p) * g 0 + p * g 1 := by
    intro g
    rw [hν, integral_add_measure, integral_smul_measure, integral_smul_measure, integral_dirac,
      integral_dirac, ENNReal.toReal_ofReal (by linarith), ENNReal.toReal_ofReal hp0]
    · simp
    · exact (integrable_dirac (by simp)).smul_measure (by simp)
    · exact (integrable_dirac (by simp)).smul_measure (by simp)
  have hb : ∀ᵐ x ∂ν, id x ∈ Set.Icc (0 : ℝ) 1 := by
    rw [ae_iff, hν, Measure.add_apply, Measure.smul_apply, Measure.smul_apply]
    simp
  have h := (hasSubgaussianMGF_of_mem_Icc (μ := ν) (X := id) measurable_id.aemeasurable hb).mgf_le t
  have hm : ν[id] = p := by rw [hint]; simp
  rw [hm] at h
  unfold mgf at h
  rw [hint] at h
  simp only [id] at h
  have hc : (((‖(1 : ℝ) - 0‖₊ / 2) ^ 2 : NNReal) : ℝ) = 1 / 4 := by
    simp; norm_num
  rw [hc] at h
  have e : (1 - p) * Real.exp (t * (0 - p)) + p * Real.exp (t * (1 - p))
      = Real.exp (-(p * t)) * (1 - p + p * Real.exp t) := by
    rw [show t * (1 - p) = t + -(p * t) by ring, Real.exp_add]; ring_nf
  rw [e] at h
  have h2 : 1 - p + p * Real.exp t ≤ Real.exp (p * t) * Real.exp (1 / 4 * t ^ 2 / 2) := by
    calc 1 - p + p * Real.exp t
        = Real.exp (p * t) * (Real.exp (-(p * t)) * (1 - p + p * Real.exp t)) := by
          rw [← mul_assoc, ← Real.exp_add]; simp
      _ ≤ _ := mul_le_mul_of_nonneg_left h (Real.exp_pos _).le
  rw [← Real.exp_add] at h2
  convert h2 using 2; ring

lemma pbMean_exp_le : ∀ (ps : List ℝ), (∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) → ∀ t : ℝ,
    pbMean ps (fun j => Real.exp (t * j)) ≤ Real.exp (t * ps.sum + ps.length * t ^ 2 / 8)
  | [], _, t => by simp [pbMean, pbLaw, pbBase]
  | p :: ps, h, t => by
    have hp := h p List.mem_cons_self
    have hps : ∀ q ∈ ps, 0 ≤ q ∧ q ≤ 1 := fun q hq => h q (List.mem_cons_of_mem _ hq)
    rw [pbMean_cons]
    have e : pbMean ps (fun j => Real.exp (t * ((j + 1 : ℕ) : ℝ)))
        = Real.exp t * pbMean ps (fun j => Real.exp (t * j)) := by
      unfold pbMean
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun j _ => ?_
      push_cast
      rw [mul_add, mul_one, Real.exp_add]; ring
    rw [e]
    have ih := pbMean_exp_le ps hps t
    have h0 : 0 ≤ pbMean ps (fun j => Real.exp (t * j)) :=
      Finset.sum_nonneg fun j _ => mul_nonneg (Real.exp_pos _).le (pbLaw_nonneg ps hps j)
    calc (1 - p) * pbMean ps (fun j => Real.exp (t * j))
          + p * (Real.exp t * pbMean ps (fun j => Real.exp (t * j)))
        = (1 - p + p * Real.exp t) * pbMean ps (fun j => Real.exp (t * j)) := by ring
      _ ≤ Real.exp (p * t + t ^ 2 / 8) * Real.exp (t * ps.sum + ps.length * t ^ 2 / 8) :=
          mul_le_mul (bit_mgf_le hp.1 hp.2 t) ih h0 (Real.exp_pos _).le
      _ = _ := by
          rw [← Real.exp_add]; congr 1; simp only [List.sum_cons, List.length_cons]; push_cast; ring

/-- Hoeffding's inequality for the vote: a side at least `d` past the mean has mass at most
`exp (-2d²/N)`. -/
lemma pbTail_le (ps : List ℝ) (hps : ∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) (hN : 0 < ps.length) {d s : ℝ}
    (hs : s = 1 ∨ s = -1) (T : ℕ → Prop) [DecidablePred T]
    (hT : ∀ j, T j → d ≤ s * ((j : ℝ) - ps.sum)) (hd : 0 ≤ d) :
    ∑ j ∈ Finset.range (ps.length + 1), (if T j then pbLaw ps j else 0)
      ≤ Real.exp (-2 * d ^ 2 / ps.length) := by
  set N : ℝ := (ps.length : ℝ)
  have hN0 : 0 < N := Nat.cast_pos.2 hN
  set m := ps.sum
  set t : ℝ := 4 * d * s / N with ht
  have hs2 : s * s = 1 := by rcases hs with rfl | rfl <;> norm_num
  have hpt : ∀ j : ℕ, T j → 4 * d ^ 2 / N ≤ t * (j - m) := by
    intro j hj
    have h1 := hT j hj
    have e : t * (j - m) = 4 * d / N * (s * (j - m)) := by rw [ht]; ring
    rw [e, show 4 * d ^ 2 / N = 4 * d / N * d by ring]
    exact mul_le_mul_of_nonneg_left h1 (by positivity)
  calc ∑ j ∈ Finset.range (ps.length + 1), (if T j then pbLaw ps j else 0)
      ≤ ∑ j ∈ Finset.range (ps.length + 1),
          Real.exp (t * j) * Real.exp (-(t * m) - 4 * d ^ 2 / N) * pbLaw ps j :=
        Finset.sum_le_sum fun j _ => by
          have hf := pbLaw_nonneg ps hps j
          split_ifs with hj
          · have : 1 ≤ Real.exp (t * j) * Real.exp (-(t * m) - 4 * d ^ 2 / N) := by
              rw [← Real.exp_add]
              exact Real.one_le_exp (by linarith [hpt j hj])
            nlinarith
          · positivity
    _ = Real.exp (-(t * m) - 4 * d ^ 2 / N) * pbMean ps (fun j => Real.exp (t * j)) := by
        unfold pbMean; rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun j _ => by ring
    _ ≤ Real.exp (-(t * m) - 4 * d ^ 2 / N) * Real.exp (t * m + N * t ^ 2 / 8) :=
        mul_le_mul_of_nonneg_left (pbMean_exp_le ps hps t) (Real.exp_pos _).le
    _ = Real.exp (-2 * d ^ 2 / N) := by
        rw [← Real.exp_add]; congr 1; rw [ht]; field_simp; linear_combination (16 * d ^ 2) * hs2
/-- A band whose width squared is at least `2N log (1/ε)` passes its check at `ε`. -/
theorem bandPasses_of_wide {N lo hi : ℕ} {p r ε ε₂ κ : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hlohi : lo ≤ hi) (hε : 0 < ε)
    (hw : 2 * N * Real.log (1 / ε) ≤ ((hi : ℝ) + 1 - lo) ^ 2) :
    BandPasses N lo (hi + 1) p r ε ε₂ κ := by
  intro a ha
  set ps := List.replicate a p ++ List.replicate (N - a) r with hpsd
  have hlen : ps.length = N := by simp only [ps, List.length_append, List.length_replicate]; omega
  have hps : ∀ q ∈ ps, 0 ≤ q ∧ q ≤ 1 := by
    intro q hq
    simp only [ps, List.mem_append, List.mem_replicate] at hq
    rcases hq with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact ⟨hp0, hp1⟩
    · exact ⟨hr0, hr1⟩
  have hdist : ∀ rd, (∑ j ∈ Finset.range (N + 1), if readOf lo (hi + 1) j = rd then
        ∑ x ∈ Finset.antidiagonal j, (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
          * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2)) else 0)
      = ∑ j ∈ Finset.range (N + 1), if readOf lo (hi + 1) j = rd then pbLaw ps j else 0 :=
    fun rd => Finset.sum_congr rfl fun j _ => by
      rw [show pbLaw ps j = voteLaw N a p r j from (voteLaw_eq_foldr N a p r j).symm]
      rfl
  set m := ps.sum
  set d : ℝ := ((hi : ℝ) + 1 - lo) / 2 with hd
  have hlh : (lo : ℝ) ≤ hi := by exact_mod_cast hlohi
  have hd0 : 0 < d := by rw [hd]; linarith
  set m := ps.sum
  set d : ℝ := ((hi : ℝ) + 1 - lo) / 2 with hd
  have hlh : (lo : ℝ) ≤ hi := by exact_mod_cast hlohi
  have hd0 : 0 < d := by rw [hd]; linarith
  have htail : ∀ (s : ℝ), (s = 1 ∨ s = -1) → ∀ (T : ℕ → Prop) [DecidablePred T],
      (∀ j, T j → d ≤ s * ((j : ℝ) - m)) →
      ∑ j ∈ Finset.range (N + 1), (if T j then pbLaw ps j else 0) ≤ ε := by
    intro s hs T _ hT
    rcases Nat.eq_zero_or_pos N with hN | hN
    · have hp : ps = [] := List.eq_nil_of_length_eq_zero (hlen.trans hN)
      subst hN
      rw [Finset.sum_range_one, if_neg]
      · exact hε.le
      intro h0
      have := hT 0 h0
      simp only [m, hp, List.sum_nil, Nat.cast_zero, sub_zero, mul_zero] at this
      linarith
    · have h := pbTail_le ps hps (by omega) hs T hT hd0.le
      rw [hlen] at h
      refine h.trans ?_
      have hN0 : (0 : ℝ) < N := by exact_mod_cast hN
      calc Real.exp (-2 * d ^ 2 / N) ≤ Real.exp (-Real.log (1 / ε)) := by
            refine Real.exp_le_exp.2 ?_
            have h1 : Real.log (1 / ε) ≤ 2 * d ^ 2 / N := by
              rw [le_div_iff₀ hN0, hd]; nlinarith [hw]
            have e : -2 * d ^ 2 / N = -(2 * d ^ 2 / N) := by ring
            rw [e]; linarith
        _ = ε := by rw [Real.exp_neg, Real.exp_log (by positivity)]; simp
  dsimp only
  rw [hdist .accept, hdist .reject]
  have hacc : m ≤ lo + d → ∑ j ∈ Finset.range (N + 1),
      (if readOf lo (hi + 1) j = .accept then pbLaw ps j else 0) ≤ ε := by
    intro hm
    refine htail 1 (.inl rfl) _ fun j hj => ?_
    have hj' : hi + 1 ≤ j := by
      unfold readOf at hj
      split_ifs at hj with h1
      exact h1
    have : ((hi : ℝ) + 1) ≤ j := by exact_mod_cast hj'
    rw [hd] at hm ⊢
    linarith
  have hrej : lo + d < m → ∑ j ∈ Finset.range (N + 1),
      (if readOf lo (hi + 1) j = .reject then pbLaw ps j else 0) ≤ ε := by
    intro hm
    refine htail (-1) (.inr rfl) _ fun j hj => ?_
    have hj' : j ≤ lo := by
      unfold readOf at hj
      split_ifs at hj with h1 h2
      exact h2
    have : (j : ℝ) ≤ lo := by exact_mod_cast hj'
    linarith
  rcases le_or_gt m (lo + d) with hm | hm
  · exact ⟨Or.inl (hacc hm), (min_le_left _ _).trans ((hacc hm).trans (le_max_left _ _))⟩
  · exact ⟨Or.inr (Or.inl (hrej hm)), (min_le_right _ _).trans ((hrej hm).trans (le_max_left _ _))⟩

end OrthoDFA
