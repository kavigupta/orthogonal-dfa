import OrthoDFA.LowLevelClaims.SeqTest
import Mathlib.Probability.ProductMeasure
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The sequential test is sound

Each draw multiplies the likelihood ratio by `p₁ / p₀` on a hit and by `(1 − p₁) / (1 − p₀)`
otherwise. On the far side of `p₀` from `p₁` that factor averages at most 1, so `a` times the
ratio is a nonnegative supermartingale from `a`, and it reaches 1, where the test fires, with chance
at most `a` (Ville's inequality, here by induction on the draws).

Where `p₀ < p₁ < p₀'`, one of the tests of `p₀` and of `p₀'` against `p₁` fires within
`stretchBound` draws, whatever the count.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

section Draws

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

omit [Countable X] [MeasurableSingletonClass X] in
theorem piFinSuccAbove_cons_zero {n : ℕ} (d : Fin (n + 1) → X) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) 0 d = (d 0, Fin.tail d) := by
  ext i
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove_apply, Fin.tail]

theorem pi_cons_apply (D : Measure X) [IsProbabilityMeasure D] (T : ℕ)
    (E : Set (Fin (T + 1) → X)) :
    (Measure.pi fun _ : Fin (T + 1) => D) E
      = ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈ E} ∂D := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (T + 1) => D) 0
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (T + 1) => X) 0
  have hE : E = e ⁻¹' {p | Fin.cons p.1 p.2 ∈ E} := by
    ext d
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, e, piFinSuccAbove_cons_zero, Fin.cons_self_tail]
  rw [hE, hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
    Measure.prod_apply (Set.to_countable _).measurableSet]
  rfl

end Draws

section Ratio

variable {p₀ p₁ : ℝ}

theorem likelihoodRatio_zero : likelihoodRatio p₀ p₁ 0 0 = 1 := by simp [likelihoodRatio]

theorem likelihoodRatio_nonneg (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁1 : p₁ ≤ 1)
    (n h : ℕ) : 0 ≤ likelihoodRatio p₀ p₁ n h := by
  unfold likelihoodRatio
  have : 0 ≤ 1 - p₁ := by linarith
  have : 0 < 1 - p₀ := by linarith
  positivity

/-- A draw's factor: `p₁ / p₀` on a hit, `(1 − p₁) / (1 − p₀)` otherwise. -/
noncomputable def ratioFactor (p₀ p₁ : ℝ) (b : Bool) : ℝ :=
  if b then p₁ / p₀ else (1 - p₁) / (1 - p₀)

theorem likelihoodRatio_succ {n h : ℕ} (hh : h ≤ n) (b : Bool) :
    likelihoodRatio p₀ p₁ (n + 1) (h + if b then 1 else 0)
      = ratioFactor p₀ p₁ b * likelihoodRatio p₀ p₁ n h := by
  unfold likelihoodRatio ratioFactor
  cases b
  · simp only [Bool.false_eq_true, if_false, add_zero]
    rw [show n + 1 - h = n - h + 1 by omega, pow_succ]
    ring
  · simp only [if_true]
    rw [show n + 1 - (h + 1) = n - h by omega, pow_succ]
    ring

/-- More hits raise the ratio where `p₀ ≤ p₁`. -/
theorem likelihoodRatio_mono (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : p₀ ≤ p₁) (hp₁1 : p₁ ≤ 1)
    {n h h' : ℕ} (hh : h ≤ h') (hh' : h' ≤ n) :
    likelihoodRatio p₀ p₁ n h ≤ likelihoodRatio p₀ p₁ n h' := by
  induction hh with
  | refl => exact le_rfl
  | @step k hk ih =>
    refine (ih (by omega)).trans ?_
    unfold likelihoodRatio
    rw [show n - k = n - (k + 1) + 1 by omega, pow_succ, pow_succ]
    have hA : 1 ≤ p₁ / p₀ := by rw [le_div_iff₀ hp₀]; linarith
    have hB0 : 0 ≤ (1 - p₁) / (1 - p₀) := div_nonneg (by linarith) (by linarith)
    have hB : (1 - p₁) / (1 - p₀) ≤ 1 := by rw [div_le_one (by linarith)]; linarith
    have hX : 0 ≤ (p₁ / p₀) ^ k * ((1 - p₁) / (1 - p₀)) ^ (n - (k + 1)) := by
      have : 0 ≤ p₁ / p₀ := by positivity
      positivity
    nlinarith

end Ratio

section Sound

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) {p₀ p₁ a : ℝ}

/-- On the far side of `p₀` from `p₁`, a draw's factor averages at most 1. -/
theorem lintegral_ratioFactor_le [DecidablePred P] (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁)
    (hp₁1 : p₁ ≤ 1) (hq : (D.real {x | P x} - p₀) * (p₁ - p₀) ≤ 0) :
    ∫⁻ x, ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x))) ∂D ≤ 1 := by
  set A := p₁ / p₀
  set B := (1 - p₁) / (1 - p₀)
  have hA0 : 0 ≤ A := by positivity
  have hB0 : 0 ≤ B := div_nonneg (by linarith) (by linarith)
  set S := {x | P x}
  have hS : MeasurableSet S := (Set.to_countable _).measurableSet
  set q := D.real S
  have hq0 : 0 ≤ q := measureReal_nonneg
  have hq1 : q ≤ 1 := measureReal_le_one
  have hpt : ∀ x, ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x)))
      = S.indicator (fun _ => ENNReal.ofReal A) x + Sᶜ.indicator (fun _ => ENNReal.ofReal B) x := by
    intro x
    by_cases hx : P x
    · simp [ratioFactor, hx, S, A]
    · simp [ratioFactor, hx, S, B]
  have hDS : D S = ENNReal.ofReal q := (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
  have hDSc : D Sᶜ = ENNReal.ofReal (1 - q) := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top D Sᶜ), ← measureReal_def, measureReal_compl hS,
      probReal_univ]
  simp_rw [hpt]
  rw [lintegral_add_left (measurable_const.indicator hS), lintegral_indicator_const hS,
    lintegral_indicator_const hS.compl, hDS, hDSc, ← ENNReal.ofReal_mul hA0,
    ← ENNReal.ofReal_mul hB0, ← ENNReal.ofReal_add (by positivity) (by nlinarith),
    ← ENNReal.ofReal_one]
  refine ENNReal.ofReal_le_ofReal ?_
  have h1 : p₀ * A = p₁ := by simp only [A]; field_simp
  have h2 : (1 - p₀) * B = 1 - p₁ := by
    simp only [B]; field_simp [show (1 - p₀) ≠ 0 by linarith]
  have key : (A * q + B * (1 - q) - 1) * (p₀ * (1 - p₀)) = (q - p₀) * (p₁ - p₀) := by
    have : (A * q + B * (1 - q) - 1) * (p₀ * (1 - p₀))
        = q * (1 - p₀) * (p₀ * A) + (1 - q) * p₀ * ((1 - p₀) * B) - p₀ * (1 - p₀) := by ring
    rw [this, h1, h2]
    ring
  have hpos : 0 < p₀ * (1 - p₀) := mul_pos hp₀ (by linarith)
  by_contra hc
  push Not at hc
  have : 0 < (A * q + B * (1 - q) - 1) * (p₀ * (1 - p₀)) := mul_pos (by linarith) hpos
  linarith

open scoped Classical in
/-- Ville's inequality for the ratio started at `ℓ`: it reaches `1 / a` at some look with chance
at most `a ℓ`. -/
theorem ville (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁1 : p₁ ≤ 1) (ha : 0 < a)
    (hq : (D.real {x | P x} - p₀) * (p₁ - p₀) ≤ 0) :
    ∀ (N : ℕ) (ℓ : ℝ), 0 ≤ ℓ → (Measure.pi fun _ : Fin N => D)
      {xs | ∃ n ≤ N, 1 / a ≤ ℓ * likelihoodRatio p₀ p₁ n
        (((List.ofFn xs).take n).countP fun x => decide (P x))} ≤ ENNReal.ofReal (a * ℓ) := by
  intro N
  induction N with
  | zero =>
    intro ℓ hℓ
    by_cases h0 : 1 / a ≤ ℓ
    · refine prob_le_one.trans ?_
      rw [← ENNReal.ofReal_one]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [div_le_iff₀ ha] at h0
      linarith
    · have : {xs : Fin 0 → X | ∃ n ≤ 0, 1 / a ≤ ℓ * likelihoodRatio p₀ p₁ n
          (((List.ofFn xs).take n).countP fun x => decide (P x))} = ∅ := by
        ext xs
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
        intro n hn
        obtain rfl : n = 0 := by omega
        simpa [likelihoodRatio_zero] using h0
      rw [this, measure_empty]
      exact bot_le
  | succ N ih =>
    intro ℓ hℓ
    by_cases h0 : 1 / a ≤ ℓ
    · refine prob_le_one.trans ?_
      rw [← ENNReal.ofReal_one]
      refine ENNReal.ofReal_le_ofReal ?_
      rw [div_le_iff₀ ha] at h0
      linarith
    rw [pi_cons_apply]
    have hr0 : ∀ x, 0 ≤ ratioFactor p₀ p₁ (decide (P x)) := by
      intro x
      unfold ratioFactor
      split_ifs
      · positivity
      · exact div_nonneg (by linarith) (by linarith)
    calc ∫⁻ x, (Measure.pi fun _ : Fin N => D) {ys | Fin.cons x ys ∈
          {xs : Fin (N + 1) → X | ∃ n ≤ N + 1, 1 / a ≤ ℓ * likelihoodRatio p₀ p₁ n
            (((List.ofFn xs).take n).countP fun x => decide (P x))}} ∂D
        ≤ ∫⁻ x, ENNReal.ofReal (a * (ℓ * ratioFactor p₀ p₁ (decide (P x)))) ∂D := by
          refine lintegral_mono fun x => le_trans (measure_mono fun ys hys => ?_)
            (ih _ (mul_nonneg hℓ (hr0 x)))
          obtain ⟨n, hn, hle⟩ := hys
          rcases n with _ | k
          · exact absurd (by simpa [likelihoodRatio_zero] using hle) h0
          · refine ⟨k, by omega, ?_⟩
            simp only [List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, List.take_succ_cons,
              List.countP_cons] at hle
            have hk : ((List.ofFn ys).take k).countP (fun x => decide (P x)) ≤ k :=
              List.countP_le_length.trans (by simp)
            rw [show (if decide (P x) = true then 1 else 0) = if decide (P x) then 1 else 0
              from rfl,
              likelihoodRatio_succ hk] at hle
            linarith
      _ = ENNReal.ofReal (a * ℓ) * ∫⁻ x, ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x))) ∂D := by
          rw [← lintegral_const_mul _ (measurable_of_countable _)]
          refine lintegral_congr fun x => ?_
          rw [← ENNReal.ofReal_mul (by positivity)]
          ring_nf
      _ ≤ ENNReal.ofReal (a * ℓ) * 1 :=
          by gcongr; exact lintegral_ratioFactor_le D P hp₀ hp₀1 hp₁ hp₁1 hq
      _ = _ := mul_one _

theorem seqRateSound : SeqRateSound := by
  intro X _ _ _ D _ P p₀ p₁ a N hp₀ hp₀1 hp₁ hp₁1 ha hq
  classical
  have := ville D P hp₀ hp₀1 hp₁ hp₁1 ha hq N 1 zero_le_one
  simp only [one_mul, mul_one] at this
  refine ENNReal.toReal_le_of_le_ofReal ha.le (le_trans (le_of_eq ?_) this)
  rfl

/-- `a` times the ratio after one more draw averages at most `a` times the ratio now, on the far
side of `p₀` from `p₁`. -/
theorem lintegral_ratio_succ [DecidablePred P]
    (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 ≤ p₁) (hp₁1 : p₁ ≤ 1) (ha : 0 ≤ a)
    (hq : (D.real {x | P x} - p₀) * (p₁ - p₀) ≤ 0) {n h : ℕ} (hh : h ≤ n) :
    ∫⁻ x, ENNReal.ofReal (a * likelihoodRatio p₀ p₁ (n + 1)
      (h + if decide (P x) then 1 else 0)) ∂D ≤ ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h) := by
  have hl := likelihoodRatio_nonneg hp₀ hp₀1 hp₁ hp₁1 n h
  have hr : ∀ x, 0 ≤ ratioFactor p₀ p₁ (decide (P x)) := by
    intro x
    unfold ratioFactor
    split_ifs
    · positivity
    · exact div_nonneg (by linarith) (by linarith)
  calc _ = ∫⁻ x, ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h)
        * ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x))) ∂D := by
        refine lintegral_congr fun x => ?_
        rw [likelihoodRatio_succ hh, ← ENNReal.ofReal_mul (by positivity)]
        ring_nf
    _ = ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h)
        * ∫⁻ x, ENNReal.ofReal (ratioFactor p₀ p₁ (decide (P x))) ∂D :=
        lintegral_const_mul _ (measurable_of_countable _)
    _ ≤ ENNReal.ofReal (a * likelihoodRatio p₀ p₁ n h) * 1 := by
        gcongr; exact lintegral_ratioFactor_le D P hp₀ hp₀1 hp₁ hp₁1 hq
    _ = _ := mul_one _

end Sound

section Decide

variable {p₀ p₁ p₀' a : ℝ}

theorem log_likelihoodRatio (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁) (hp₁1 : p₁ < 1)
    {n d : ℕ} (hd : d ≤ n) :
    Real.log (likelihoodRatio p₀ p₁ n d)
      = d * Real.log (p₁ / p₀) + (n - d : ℝ) * Real.log ((1 - p₁) / (1 - p₀)) := by
  unfold likelihoodRatio
  have h1 : 0 < p₁ / p₀ := div_pos hp₁ hp₀
  have h2 : 0 < (1 - p₁) / (1 - p₀) := div_pos (by linarith) (by linarith)
  rw [Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow,
    Nat.cast_sub hd]

/-- Where `p₀ < p₁ < p₀'`, the draws within which one of the tests of `p₀` and of `p₀'` against
`p₁` at level `a` fires, whatever the count: `⌈(A₁ + B₁ + A₂ + B₂) log(1/a) / (A₁ A₂ − B₁ B₂)⌉`,
the `A`s and `B`s the logs of the ratios' factors. -/
noncomputable def stretchBound (p₀ p₁ p₀' a : ℝ) : ℕ :=
  ⌈(Real.log (p₁ / p₀) + Real.log ((1 - p₀) / (1 - p₁)) + Real.log ((1 - p₁) / (1 - p₀'))
      + Real.log (p₀' / p₁)) * Real.log (1 / a)
    / (Real.log (p₁ / p₀) * Real.log ((1 - p₁) / (1 - p₀'))
      - Real.log ((1 - p₀) / (1 - p₁)) * Real.log (p₀' / p₁))⌉₊

/-- Where `p₀ < p₁ < p₀'`, at any look from `stretchBound` on one of the tests of `p₀` and of `p₀'`
against `p₁` fires, whatever the count. -/
theorem seqRate_decides (hp₀ : 0 < p₀) (h01 : p₀ < p₁) (h12 : p₁ < p₀') (hp₀' : p₀' < 1)
    (ha : 0 < a) (ha1 : a < 1) {n : ℕ} (hn : stretchBound p₀ p₁ p₀' a ≤ n) :
    ∀ d ≤ n, seqRate p₀ p₁ a n d ∨ seqRate p₀' p₁ a n d := by
  intro d hd
  have hp₁ : 0 < p₁ := by linarith
  have hp₁1 : p₁ < 1 := by linarith
  set A₁ := Real.log (p₁ / p₀)
  set B₁ := Real.log ((1 - p₀) / (1 - p₁))
  set A₂ := Real.log ((1 - p₁) / (1 - p₀'))
  set B₂ := Real.log (p₀' / p₁)
  have hA₁ : 0 < A₁ := Real.log_pos (by rw [one_lt_div hp₀]; linarith)
  have hB₁ : 0 < B₁ := Real.log_pos (by rw [one_lt_div (by linarith)]; linarith)
  have hA₂ : 0 < A₂ := Real.log_pos (by rw [one_lt_div (by linarith)]; linarith)
  have hB₂ : 0 < B₂ := Real.log_pos (by rw [one_lt_div hp₁]; linarith)
  -- the thresholds part: `(1 − p₁) B₁ < p₁ A₁` and `p₁ B₂ < (1 − p₁) A₂`
  have hneg : ∀ {x y : ℝ}, 0 < x → 0 < y → Real.log (x / y) = - Real.log (y / x) := by
    intro x y hx hy
    rw [Real.log_div hx.ne' hy.ne', Real.log_div hy.ne' hx.ne']; ring
  have k1 : (1 - p₁) * B₁ < p₁ * A₁ := by
    have e1 := Real.log_lt_sub_one_of_pos (x := (1 - p₀) / (1 - p₁)) (by
      exact div_pos (by linarith) (by linarith)) (by
      intro h; rw [div_eq_one_iff_eq (by linarith)] at h; linarith)
    have e2 := Real.log_lt_sub_one_of_pos (x := p₀ / p₁) (div_pos hp₀ hp₁) (by
      intro h; rw [div_eq_one_iff_eq hp₁.ne'] at h; linarith)
    have hA : A₁ = - Real.log (p₀ / p₁) := hneg hp₁ hp₀
    have t1 : (1 - p₁) * ((1 - p₀) / (1 - p₁) - 1) = p₁ - p₀ := by
      field_simp [show (1 - p₁) ≠ 0 by linarith]; ring
    have t2 : p₁ * (p₀ / p₁ - 1) = p₀ - p₁ := by field_simp
    have := mul_lt_mul_of_pos_left e1 (show 0 < 1 - p₁ by linarith)
    have := mul_lt_mul_of_pos_left e2 hp₁
    rw [hA]
    nlinarith
  have k2 : p₁ * B₂ < (1 - p₁) * A₂ := by
    have e1 := Real.log_lt_sub_one_of_pos (x := p₀' / p₁) (div_pos (by linarith) hp₁) (by
      intro h; rw [div_eq_one_iff_eq hp₁.ne'] at h; linarith)
    have e2 := Real.log_lt_sub_one_of_pos (x := (1 - p₀') / (1 - p₁))
      (div_pos (by linarith) (by linarith)) (by
      intro h; rw [div_eq_one_iff_eq (by linarith)] at h; linarith)
    have hA : A₂ = - Real.log ((1 - p₀') / (1 - p₁)) := hneg (by linarith) (by linarith)
    have t1 : p₁ * (p₀' / p₁ - 1) = p₀' - p₁ := by field_simp
    have t2 : (1 - p₁) * ((1 - p₀') / (1 - p₁) - 1) = p₁ - p₀' := by
      field_simp [show (1 - p₁) ≠ 0 by linarith]; ring
    have := mul_lt_mul_of_pos_left e1 hp₁
    have := mul_lt_mul_of_pos_left e2 (show 0 < 1 - p₁ by linarith)
    rw [hA]
    nlinarith
  have hK : 0 < A₁ * A₂ - B₁ * B₂ := by
    have hq : 0 < p₁ * (1 - p₁) := mul_pos hp₁ (by linarith)
    have hprod := mul_lt_mul k1 k2.le (by positivity) (by positivity)
    have : p₁ * (1 - p₁) * (B₁ * B₂) < p₁ * (1 - p₁) * (A₁ * A₂) := by nlinarith
    have := lt_of_mul_lt_mul_left this hq.le
    linarith
  set Λ := Real.log (1 / a)
  have hΛ : 0 < Λ := Real.log_pos (by rw [one_div]; exact one_lt_inv_iff₀.2 ⟨ha, ha1⟩)
  have hn' : (A₁ + B₁ + A₂ + B₂) * Λ / (A₁ * A₂ - B₁ * B₂) ≤ n :=
    (Nat.le_ceil _).trans (by exact_mod_cast hn)
  rw [div_le_iff₀ hK] at hn'
  by_contra hno
  push Not at hno
  obtain ⟨h1, h2⟩ := hno
  have hlr1 : 0 < likelihoodRatio p₀ p₁ n d := by
    unfold likelihoodRatio
    have : 0 < (1 - p₁) / (1 - p₀) := div_pos (by linarith) (by linarith)
    have : 0 < p₁ / p₀ := div_pos hp₁ hp₀
    positivity
  have hlr2 : 0 < likelihoodRatio p₀' p₁ n d := by
    unfold likelihoodRatio
    have : 0 < (1 - p₁) / (1 - p₀') := div_pos (by linarith) (by linarith)
    have : 0 < p₁ / p₀' := div_pos hp₁ (by linarith)
    positivity
  have l1 : Real.log (likelihoodRatio p₀ p₁ n d) < Λ :=
    (Real.log_lt_log_iff hlr1 (by positivity)).2 (not_le.1 h1)
  have l2 : Real.log (likelihoodRatio p₀' p₁ n d) < Λ :=
    (Real.log_lt_log_iff hlr2 (by positivity)).2 (not_le.1 h2)
  rw [log_likelihoodRatio hp₀ (by linarith) hp₁ hp₁1 hd] at l1
  rw [log_likelihoodRatio (by linarith) hp₀' hp₁ hp₁1 hd] at l2
  have r1 : Real.log ((1 - p₁) / (1 - p₀)) = -B₁ := hneg (by linarith) (by linarith)
  have r2 : Real.log (p₁ / p₀') = -B₂ := hneg hp₁ (by linarith)
  rw [r1] at l1
  rw [r2] at l2
  have hdn : (d : ℝ) ≤ n := by exact_mod_cast hd
  have hd0 : (0 : ℝ) ≤ d := Nat.cast_nonneg d
  -- `(A₂ + B₂) L₁ + (A₁ + B₁) L₂ = n (A₁ A₂ − B₁ B₂)`
  have hsum : (A₂ + B₂) * (d * A₁ + (n - d) * -B₁) + (A₁ + B₁) * (d * -B₂ + (n - d) * A₂)
      = n * (A₁ * A₂ - B₁ * B₂) := by ring
  have := mul_lt_mul_of_pos_left l1 (show 0 < A₂ + B₂ by linarith)
  have := mul_lt_mul_of_pos_left l2 (show 0 < A₁ + B₁ by linarith)
  nlinarith

end Decide

end OrthoDFA
