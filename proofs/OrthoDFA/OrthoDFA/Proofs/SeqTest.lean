import OrthoDFA.SeqTest
import Mathlib.Probability.ProductMeasure

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
  · simp [MeasurableEquiv.piFinSuccAbove_apply, Fin.removeNth, Fin.tail]

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
  push_neg at hc
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
  push_neg at hno
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

/-! ## At the midpoint -/

/-- The algebra behind `drift_ge`. -/
theorem drift_alg {s t W Y p q : ℝ} (hs0 : 0 < s) (hs1 : s ≤ 1) (ht0 : 0 < t) (ht1 : t ≤ 1)
    (hW1 : W ≤ s) (hW2 : s / 2 ≤ W) (hY1 : Y ≤ t) (hY2 : t / 2 ≤ Y) (hp : s ^ 2 ≤ p)
    (hq : t ^ 2 ≤ q) :
    s * t / 8 * (W + p + Y + (Y + q) + W) ≤ (W + p) * (Y + q) - Y * W := by
  have hp0 : 0 ≤ p := le_trans (sq_nonneg s) hp
  have hq0 : 0 ≤ q := le_trans (sq_nonneg t) hq
  have hst : 0 < s * t := mul_pos hs0 ht0
  have hκs : s * t / 8 ≤ s / 4 := by nlinarith
  have hκt : s * t / 8 ≤ t / 4 := by nlinarith
  have a1 : (s / 2) * q ≤ W * q := mul_le_mul_of_nonneg_right hW2 hq0
  have a2 : (t / 2) * p ≤ Y * p := mul_le_mul_of_nonneg_right hY2 hp0
  have a3 : (s / 2 - s * t / 8) * t ^ 2 ≤ (s / 2 - s * t / 8) * q :=
    mul_le_mul_of_nonneg_left hq (by linarith)
  have a4 : (t / 2 - s * t / 8) * s ^ 2 ≤ (t / 2 - s * t / 8) * p :=
    mul_le_mul_of_nonneg_left hp (by linarith)
  have a5 : s ^ 2 * t ^ 2 ≤ p * q := mul_le_mul hp hq (sq_nonneg t) hp0
  have a6 : s * t / 8 * W ≤ s * t / 8 * s := mul_le_mul_of_nonneg_left hW1 (by positivity)
  have a7 : s * t / 8 * Y ≤ s * t / 8 * t := mul_le_mul_of_nonneg_left hY1 (by positivity)
  have a8 : s * t * (s ^ 2 + t ^ 2) ≤ s * t * (s + t) := by
    apply mul_le_mul_of_nonneg_left _ hst.le; nlinarith
  have a9 : 0 ≤ s * t * (s + t) := by positivity
  have a10 : 0 ≤ s ^ 2 * t ^ 2 := by positivity
  linear_combination a1 + a2 + a3 + a4 + a5 + 2 * a6 + 2 * a7 + a8 / 8 + a9 / 8 + a10

/-- Testing `y` and `ε` against their midpoint, the drift `A₁A₂ − B₁B₂` is at least
`(ε − y)² / (32 ε)` times `A₁ + B₁ + A₂ + B₂`. -/
theorem drift_ge {y ε : ℝ} (hy : 0 < y) (hyε : y < ε) (hε : ε < 1) :
    (ε - y) ^ 2 / (32 * ε)
        * (Real.log ((y + ε) / 2 / y) + Real.log ((1 - y) / (1 - (y + ε) / 2))
          + Real.log ((1 - (y + ε) / 2) / (1 - ε)) + Real.log (ε / ((y + ε) / 2)))
      ≤ Real.log ((y + ε) / 2 / y) * Real.log ((1 - (y + ε) / 2) / (1 - ε))
        - Real.log ((1 - y) / (1 - (y + ε) / 2)) * Real.log (ε / ((y + ε) / 2)) := by
  set m := (y + ε) / 2 with hm
  set h := (ε - y) / 2 with hh
  have hm0 : 0 < m := by linarith
  have hu0 : 0 < 1 - m := by linarith
  have hh0 : 0 < h := by linarith
  have hhm : h < m := by linarith
  have hhu : h < 1 - m := by linarith
  have hε' : ε = m + h := by linarith
  have hy' : y = m - h := by linarith
  set s := h / m with hs
  set t := h / (1 - m) with ht
  have hs0 : 0 < s := div_pos hh0 hm0
  have ht0 : 0 < t := div_pos hh0 hu0
  have hs1 : s ≤ 1 := by rw [hs, div_le_one hm0]; linarith
  have ht1 : t ≤ 1 := by rw [ht, div_le_one hu0]; linarith
  set X := Real.log (m / y)
  set Y := Real.log ((1 - y) / (1 - m))
  set Z := Real.log ((1 - m) / (1 - ε))
  set W := Real.log (ε / m)
  have hεpos : 0 < ε := by linarith
  have h1y : 0 < 1 - y := by linarith
  have h1ε : 0 < 1 - ε := by linarith
  -- upper bounds
  have hW1 : W ≤ s := by
    have := Real.log_le_sub_one_of_pos (div_pos hεpos hm0)
    have e : ε / m - 1 = s := by rw [hs, hε']; field_simp; ring
    linarith
  have hY1 : Y ≤ t := by
    have := Real.log_le_sub_one_of_pos (div_pos h1y hu0)
    have e : (1 - y) / (1 - m) - 1 = t := by rw [ht, hy']; field_simp; ring
    linarith
  -- lower bounds
  have hW2 : s / 2 ≤ W := by
    have := Real.one_sub_inv_le_log_of_pos (div_pos hεpos hm0)
    have e : 1 - (ε / m)⁻¹ = h / (m + h) := by
      rw [inv_div, one_sub_div hεpos.ne']; congr 1 <;> linarith
    have e2 : s / 2 ≤ h / (m + h) := by
      rw [hs, div_div, div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith
    linarith
  have hY2 : t / 2 ≤ Y := by
    have := Real.one_sub_inv_le_log_of_pos (div_pos h1y hu0)
    have e : 1 - ((1 - y) / (1 - m))⁻¹ = h / (1 - m + h) := by
      rw [inv_div, one_sub_div h1y.ne']; congr 1 <;> linarith
    have e2 : t / 2 ≤ h / (1 - m + h) := by
      rw [ht, div_div, div_le_div_iff₀ (by positivity) (by positivity)]; nlinarith
    linarith
  -- the gaps
  have hp : s ^ 2 ≤ X - W := by
    have hpos : 0 < m / y / (ε / m) := by positivity
    have e1 : X - W = Real.log (m / y / (ε / m)) := by
      rw [Real.log_div (by positivity) (by positivity)]
    have := Real.one_sub_inv_le_log_of_pos hpos
    have e2 : 1 - (m / y / (ε / m))⁻¹ = s ^ 2 := by
      rw [hs, hy', hε']; field_simp; ring
    linarith
  have hq : t ^ 2 ≤ Z - Y := by
    have hpos : 0 < (1 - m) / (1 - ε) / ((1 - y) / (1 - m)) := by positivity
    have e1 : Z - Y = Real.log ((1 - m) / (1 - ε) / ((1 - y) / (1 - m))) := by
      rw [Real.log_div (by positivity) (by positivity)]
    have := Real.one_sub_inv_le_log_of_pos hpos
    have e2 : 1 - ((1 - m) / (1 - ε) / ((1 - y) / (1 - m)))⁻¹ = t ^ 2 := by
      rw [ht, hy', hε']; field_simp; ring
    linarith
  have key : s * t / 8 * (X + Y + Z + W) ≤ X * Z - Y * W := by
    have := drift_alg hs0 hs1 ht0 ht1 hW1 hW2 hY1 hY2 hp hq
    have e1 : W + (X - W) = X := by ring
    have e2 : Y + (Z - Y) = Z := by ring
    rw [e1, e2] at this
    linarith
  have hsum : 0 ≤ X + Y + Z + W := by nlinarith
  have hκ : (ε - y) ^ 2 / (32 * ε) ≤ s * t / 8 := by
    have e : (ε - y) ^ 2 / (32 * ε) = h ^ 2 / (8 * ε) := by rw [hh]; ring
    have e2 : s * t / 8 = h ^ 2 / (8 * (m * (1 - m))) := by rw [hs, ht]; field_simp
    rw [e, e2]
    apply div_le_div_of_nonneg_left (by positivity) (by positivity)
    nlinarith
  calc (ε - y) ^ 2 / (32 * ε) * (X + Y + Z + W) ≤ s * t / 8 * (X + Y + Z + W) :=
        mul_le_mul_of_nonneg_right hκ hsum
    _ ≤ _ := key

/-- Testing `y` and `ε` against their midpoint, one of the tests at level `a` fires within
`32 ε / (ε − y)² · log(1/a) + 1` draws. -/
theorem stretchBound_mid_le {y ε a : ℝ} (hy : 0 < y) (hyε : y < ε) (hε : ε < 1) (ha0 : 0 < a)
    (ha1 : a ≤ 1) :
    (stretchBound y ((y + ε) / 2) ε a : ℝ) ≤ 32 * ε / (ε - y) ^ 2 * Real.log (1 / a) + 1 := by
  have hd := drift_ge hy hyε hε
  set m := (y + ε) / 2 with hm'
  set X := Real.log (m / y)
  set Y := Real.log ((1 - y) / (1 - m))
  set Z := Real.log ((1 - m) / (1 - ε))
  set W := Real.log (ε / m)
  have hX : 0 < X := Real.log_pos (by rw [one_lt_div hy]; linarith [hm'])
  have hY : 0 < Y := Real.log_pos (by rw [one_lt_div (by linarith [hm'])]; linarith [hm'])
  have hZ : 0 < Z := Real.log_pos (by rw [one_lt_div (by linarith [hm'])]; linarith [hm'])
  have hW : 0 < W := Real.log_pos (by rw [one_lt_div (by linarith [hm'])]; linarith [hm'])
  have hκ : 0 < (ε - y) ^ 2 / (32 * ε) := by
    have : 0 < ε - y := by linarith
    have : 0 < ε := by linarith
    positivity
  have hsum : 0 < X + Y + Z + W := by linarith
  have hN : 0 < X * Z - Y * W := lt_of_lt_of_le (mul_pos hκ hsum) hd
  have hL : 0 ≤ Real.log (1 / a) := Real.log_nonneg (by rw [le_div_iff₀ ha0]; linarith)
  have hx : 0 ≤ (X + Y + Z + W) * Real.log (1 / a) / (X * Z - Y * W) := by positivity
  unfold stretchBound
  refine (Nat.ceil_lt_add_one hx).le.trans ?_
  have : (X + Y + Z + W) * Real.log (1 / a) / (X * Z - Y * W)
      ≤ 32 * ε / (ε - y) ^ 2 * Real.log (1 / a) := by
    rw [div_le_iff₀ hN]
    have e : 32 * ε / (ε - y) ^ 2 = ((ε - y) ^ 2 / (32 * ε))⁻¹ := by rw [inv_div]
    rw [e]
    calc (X + Y + Z + W) * Real.log (1 / a)
        = ((ε - y) ^ 2 / (32 * ε))⁻¹ * ((ε - y) ^ 2 / (32 * ε) * (X + Y + Z + W))
          * Real.log (1 / a) := by rw [← mul_assoc, inv_mul_cancel₀ hκ.ne', one_mul]
      _ ≤ ((ε - y) ^ 2 / (32 * ε))⁻¹ * (X * Z - Y * W) * Real.log (1 / a) := by
          gcongr
      _ = _ := by ring
  linarith

end Decide

end OrthoDFA
