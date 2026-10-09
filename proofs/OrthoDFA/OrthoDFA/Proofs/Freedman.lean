import Mathlib.Probability.Martingale.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Basic

/-! # Bennett's tail for adapted increments in `[0, b]` -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Finset Real

theorem exp_mul_le_chord {l b x : ℝ} (hb : 0 < b) (hx0 : 0 ≤ x) (hxb : x ≤ b) :
    exp (l * x) ≤ 1 + x * ((exp (l * b) - 1) / b) := by
  have h := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ (l * b))
    (show (0 : ℝ) ≤ 1 - x / b by rw [sub_nonneg, div_le_one hb]; exact hxb)
    (div_nonneg hx0 hb.le) (show 1 - x / b + x / b = 1 by ring)
  simp only [smul_eq_mul, mul_zero, zero_add, exp_zero, mul_one] at h
  have hx : x / b * (l * b) = l * x := by field_simp
  rw [hx] at h
  calc exp (l * x) ≤ 1 - x / b + x / b * exp (l * b) := h
    _ = _ := by field_simp; ring

theorem le_xlogx (A s : ℝ) (hA : 0 < A) (hs : A ≤ s) : 0 ≤ s * log (s / A) - s + A := by
  have hsA : 0 < s / A := div_pos (hA.trans_le hs) hA
  have h := one_sub_inv_le_log_of_pos hsA
  rw [inv_div] at h
  have hs0 : 0 < s := hA.trans_le hs
  have : s * (1 - A / s) = s - A := by field_simp
  nlinarith [mul_le_mul_of_nonneg_left h (hA.le.trans hs)]

variable {Ω : Type*} {m0 : MeasurableSpace Ω} {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Increments `X i` are revealed at time `i + 1` with conditional mean at most `m i`. -/
theorem bennett_tail (ℱ : Filtration ℕ m0) (X m : ℕ → Ω → ℝ) {b A s : ℝ} (n : ℕ)
    (hX : ∀ i, StronglyMeasurable[ℱ (i + 1)] (X i)) (hm : ∀ i, StronglyMeasurable[ℱ i] (m i))
    (hX0 : ∀ i ω, 0 ≤ X i ω) (hXb : ∀ i ω, X i ω ≤ b) (hmean : ∀ i, μ[X i | ℱ i] ≤ᵐ[μ] m i)
    (hA : 0 < A) (hs : A ≤ s) :
    μ.real {ω | s ≤ ∑ i ∈ range n, X i ω ∧ ∑ i ∈ range n, m i ω ≤ A}
      ≤ exp (-(s * log (s / A) - s + A) / b) := by
  have hc := le_xlogx A s hA hs
  rcases le_or_gt b 0 with hb | hb
  · refine measureReal_le_one.trans (one_le_exp ?_)
    exact div_nonneg_of_nonpos (neg_nonpos.2 hc) hb
  set l := log (s / A) / b with hl
  set k := (s / A - 1) / b with hk
  have hsA : 1 ≤ s / A := (one_le_div hA).2 hs
  have hl0 : 0 ≤ l := div_nonneg (log_nonneg hsA) hb.le
  have hk0 : 0 ≤ k := div_nonneg (by linarith) hb.le
  have hlb : exp (l * b) = s / A := by
    rw [hl, div_mul_cancel₀ _ hb.ne', exp_log (by linarith)]
  set m' : ℕ → Ω → ℝ := fun i ω => max (min (m i ω) b) 0 with hm'
  set Z : ℕ → Ω → ℝ := fun j ω =>
    exp (l * ∑ i ∈ range j, X i ω - k * ∑ i ∈ range j, m' i ω) with hZ
  have hXm : ∀ i j, i < j → Measurable[ℱ j] (X i) := fun i j h =>
    ((hX i).mono (ℱ.mono (by omega))).measurable
  have hm'm : ∀ i, Measurable[ℱ i] (m' i) := fun i =>
    (((hm i).measurable.min measurable_const).max measurable_const)
  have hm'0 : ∀ i ω, 0 ≤ m' i ω := fun i ω => le_max_right _ _
  have hm'b : ∀ i ω, m' i ω ≤ b := fun i ω => max_le (min_le_right _ _) hb.le
  have hZm : ∀ j, Measurable[ℱ j] (Z j) := by
    intro j
    refine measurable_exp.comp ((Finset.measurable_sum _ fun i hi => ?_).const_mul _ |>.sub
      ((Finset.measurable_sum _ fun i hi => ?_).const_mul _))
    · exact hXm i j (mem_range.1 hi)
    · exact (hm'm i).mono (ℱ.mono (mem_range.1 hi).le) le_rfl
  have hZb : ∀ j ω, Z j ω ≤ exp (l * (j * b)) := by
    intro j ω
    refine exp_le_exp.2 ?_
    have h1 : ∑ i ∈ range j, X i ω ≤ j * b := by
      simpa using Finset.sum_le_sum fun i (_ : i ∈ range j) => hXb i ω
    have h2 : 0 ≤ ∑ i ∈ range j, m' i ω := Finset.sum_nonneg fun i _ => hm'0 i ω
    nlinarith [mul_le_mul_of_nonneg_left h1 hl0, mul_nonneg hk0 h2]
  have hint : ∀ {j} {f : Ω → ℝ} (C : ℝ), Measurable[ℱ j] f → (∀ ω, |f ω| ≤ C) →
      Integrable f μ := fun C hf hC =>
    Integrable.of_bound ((hf.mono (ℱ.le _) le_rfl).aestronglyMeasurable) C
      (ae_of_all _ fun ω => by simpa [Real.norm_eq_abs] using hC ω)
  have hXint : ∀ i, Integrable (X i) μ := fun i =>
    hint (j := i + 1) b (hXm i (i + 1) (by omega)) fun ω => by
      rw [abs_of_nonneg (hX0 i ω)]; exact hXb i ω
  have hce : ∀ i, μ[X i | ℱ i] ≤ᵐ[μ] m' i := by
    intro i
    have hb' : μ[X i | ℱ i] ≤ᵐ[μ] fun _ => b := by
      have := condExp_mono (m := ℱ i) (hXint i) (integrable_const b) (ae_of_all _ (hXb i))
      rwa [condExp_const (ℱ.le i)] at this
    filter_upwards [hmean i, hb'] with ω h1 h2
    exact le_max_of_le_left (le_min h1 h2)
  have hstep : ∀ j, ∫ ω, Z (j + 1) ω ∂μ ≤ ∫ ω, Z j ω ∂μ := by
    intro j
    set W : Ω → ℝ := fun ω => Z j ω * exp (-k * m' j ω) with hW
    have hWm : Measurable[ℱ j] W := (hZm j).mul (measurable_exp.comp ((hm'm j).const_mul _))
    have hW0 : ∀ ω, 0 ≤ W ω := fun ω => mul_nonneg (exp_pos _).le (exp_pos _).le
    have hWb : ∀ ω, |W ω| ≤ exp (l * (j * b)) := by
      intro ω
      rw [abs_of_nonneg (hW0 ω)]
      calc W ω ≤ Z j ω * 1 := mul_le_mul_of_nonneg_left
            (exp_le_one_iff.2 (by nlinarith [hm'0 j ω])) (exp_pos _).le
        _ ≤ _ := by rw [mul_one]; exact hZb j ω
    have hWi : Integrable W μ := hint _ hWm hWb
    have hWX : Integrable (W * X j) μ :=
      (hXint j).bdd_mul (hWm.mono (ℱ.le j) le_rfl).aestronglyMeasurable
        (ae_of_all _ fun ω => by simpa [Real.norm_eq_abs] using hWb ω)
    have hWm' : Integrable (fun ω => W ω * (1 + k * m' j ω)) μ :=
      hint (j := j) (exp (l * (j * b)) * (1 + k * b))
        (hWm.mul (((hm'm j).const_mul _).const_add _)) fun ω => by
          rw [abs_mul, abs_of_nonneg (by nlinarith [hm'0 j ω] : 0 ≤ 1 + k * m' j ω)]
          exact mul_le_mul (hWb ω) (by nlinarith [hm'b j ω]) (by nlinarith [hm'0 j ω])
            (exp_pos _).le
    have hZ1 : ∀ ω, Z (j + 1) ω ≤ W ω + k * (W ω * X j ω) := by
      intro ω
      have hsplit : Z (j + 1) ω = W ω * exp (l * X j ω) := by
        simp only [hZ, hW, sum_range_succ, ← exp_add]
        ring_nf
      rw [hsplit]
      have := exp_mul_le_chord (l := l) hb (hX0 j ω) (hXb j ω)
      rw [hlb, ← hk] at this
      nlinarith [mul_le_mul_of_nonneg_left this (hW0 ω)]
    have hpull : ∫ ω, W ω * X j ω ∂μ ≤ ∫ ω, W ω * m' j ω ∂μ := by
      have h1 : ∫ ω, W ω * X j ω ∂μ = ∫ ω, W ω * μ[X j | ℱ j] ω ∂μ :=
        calc ∫ ω, W ω * X j ω ∂μ = ∫ ω, (W * X j) ω ∂μ := rfl
          _ = ∫ ω, μ[W * X j | ℱ j] ω ∂μ := (integral_condExp (ℱ.le j)).symm
          _ = ∫ ω, (W * μ[X j | ℱ j]) ω ∂μ := integral_congr_ae
            (condExp_mul_of_stronglyMeasurable_left hWm.stronglyMeasurable hWX (hXint j))
          _ = _ := rfl
      rw [h1]
      refine integral_mono_ae ?_ ?_ ?_
      · exact integrable_condExp.bdd_mul (hWm.mono (ℱ.le j) le_rfl).aestronglyMeasurable
          (ae_of_all _ fun ω => by simpa [Real.norm_eq_abs] using hWb ω)
      · exact hint (j := j) (exp (l * (j * b)) * b) (hWm.mul (hm'm j)) fun ω => by
          rw [abs_mul, abs_of_nonneg (hm'0 j ω)]
          exact mul_le_mul (hWb ω) (hm'b j ω) (hm'0 j ω) (exp_pos _).le
      · filter_upwards [hce j] with ω h
        exact mul_le_mul_of_nonneg_left h (hW0 ω)
    calc ∫ ω, Z (j + 1) ω ∂μ ≤ ∫ ω, W ω + k * (W ω * X j ω) ∂μ :=
          integral_mono (hint _ (hZm (j + 1)) fun ω => by
              rw [abs_of_nonneg (exp_pos _).le]; exact hZb (j + 1) ω)
            (hWi.add (hWX.const_mul k)) hZ1
      _ = ∫ ω, W ω ∂μ + k * ∫ ω, W ω * X j ω ∂μ := by
          have e := integral_add hWi (hWX.const_mul k)
          rw [integral_const_mul] at e
          exact e
      _ ≤ ∫ ω, W ω ∂μ + k * ∫ ω, W ω * m' j ω ∂μ := by gcongr
      _ = ∫ ω, W ω * (1 + k * m' j ω) ∂μ := by
          have hWM : Integrable (fun ω => W ω * m' j ω) μ :=
            hint (j := j) (exp (l * (j * b)) * b) (hWm.mul (hm'm j)) fun ω => by
              rw [abs_mul, abs_of_nonneg (hm'0 j ω)]
              exact mul_le_mul (hWb ω) (hm'b j ω) (hm'0 j ω) (exp_pos _).le
          have e := integral_add hWi (hWM.const_mul k)
          rw [integral_const_mul] at e
          rw [← e]
          congr 1; ext ω; ring
      _ ≤ ∫ ω, Z j ω ∂μ := by
          refine integral_mono hWm' (hint _ (hZm j) fun ω => by
              rw [abs_of_nonneg (exp_pos _).le]; exact hZb j ω) fun ω => ?_
          have h1 : 1 + k * m' j ω ≤ exp (k * m' j ω) := by
            linarith [add_one_le_exp (k * m' j ω)]
          have h2 : W ω * exp (k * m' j ω) = Z j ω := by
            simp only [hW, mul_assoc, ← exp_add]; ring_nf; simp
          calc W ω * (1 + k * m' j ω) ≤ W ω * exp (k * m' j ω) :=
                mul_le_mul_of_nonneg_left h1 (hW0 ω)
            _ = Z j ω := h2
  have hZ1 : ∫ ω, Z n ω ∂μ ≤ 1 := by
    induction n with
    | zero => simp [hZ]
    | succ j ih => exact (hstep j).trans ih
  have hpos : ∀ᵐ ω ∂μ, ∀ i, 0 ≤ m i ω := by
    refine ae_all_iff.2 fun i => ?_
    filter_upwards [condExp_nonneg (m := ℱ i) (ae_of_all _ (hX0 i)), hmean i] with ω h1 h2
    exact h1.trans h2
  have hsub : {ω | s ≤ ∑ i ∈ range n, X i ω ∧ ∑ i ∈ range n, m i ω ≤ A}
      ≤ᵐ[μ] {ω | exp (l * s - k * A) ≤ Z n ω} := by
    filter_upwards [hpos] with ω hω hE
    obtain ⟨h1, h2⟩ := hE
    have h3 : ∑ i ∈ range n, m' i ω ≤ A := (Finset.sum_le_sum fun i _ =>
      max_le ((min_le_left _ _)) (hω i)).trans h2
    show exp (l * s - k * A) ≤ exp _
    exact exp_le_exp.2 (by nlinarith [mul_le_mul_of_nonneg_left h1 hl0,
      mul_le_mul_of_nonneg_left h3 hk0])
  have hmarkov := mul_meas_ge_le_integral_of_nonneg (μ := μ) (ae_of_all _ fun ω =>
    (exp_pos _).le) (hint _ (hZm n) fun ω => by
      rw [abs_of_nonneg (exp_pos _).le]; exact hZb n ω) (exp (l * s - k * A))
  have hmono : μ.real {ω | s ≤ ∑ i ∈ range n, X i ω ∧ ∑ i ∈ range n, m i ω ≤ A}
      ≤ μ.real {ω | exp (l * s - k * A) ≤ Z n ω} :=
    ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae hsub)
  have hexp : -(s * log (s / A) - s + A) / b = -(l * s - k * A) := by
    rw [hl, hk]; field_simp; ring
  rw [hexp, exp_neg]
  set F := {ω | exp (l * s - k * A) ≤ Z n ω}
  calc _ ≤ μ.real F := hmono
    _ = (exp (l * s - k * A))⁻¹ * (exp (l * s - k * A) * μ.real F) := by field_simp
    _ ≤ (exp (l * s - k * A))⁻¹ * 1 := by gcongr; exact hmarkov.trans hZ1
    _ = _ := mul_one _

end OrthoDFA
