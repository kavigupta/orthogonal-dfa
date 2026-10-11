import OrthoDFA.Proofs.RoundEnd
import Mathlib.Probability.Moments.SubGaussian

/-!
# The anchor's read concentrates

Over reads independent across strings, the share of draws and anchors whose anchor prefix reads
undecided is at least its mean less `t`, but for chance `exp(−2t²n/q)`, where `n` is the number
of anchors and `q` the most a draw puts on starting with any one string past `⌈L/2⌉`: it is a
weighted sum of independent reads with weights at most `q / n`, summing to at most one.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {α : Type*} [Fintype α]

/-- The strings of length `i`. -/
noncomputable def strsOfLen (i : ℕ) : Finset (FreeMonoid α) :=
  (Finset.univ : Finset (Fin i → α)).image fun f => FreeMonoid.ofList (List.ofFn f)

theorem mem_strsOfLen {i : ℕ} {u : FreeMonoid α} : u ∈ strsOfLen i ↔ u.toList.length = i := by
  constructor
  · intro hu
    obtain ⟨f, -, rfl⟩ := Finset.mem_image.1 hu
    simp
  · intro h
    subst h
    exact Finset.mem_image.2 ⟨u.toList.get, Finset.mem_univ _, by simp⟩

/-- The strings an anchor prefix can be. -/
noncomputable def anchorStrs (L : ℕ) : Finset (FreeMonoid α) :=
  (Finset.Icc (startLength L) L).biUnion strsOfLen

/-- The anchor prefix is `u` on this share of draws and anchors. -/
noncomputable def anchorWeight (D : Measure (FreeMonoid α)) (L : ℕ) (u : FreeMonoid α) : ℝ :=
  D.real {x | u.toList <+: x.toList} / (L + 1 - startLength L : ℕ)

omit [Fintype α] in
theorem prefix_iff_take {x u : FreeMonoid α} :
    u.toList <+: x.toList ↔
      u.toList.length ≤ x.toList.length ∧
        FreeMonoid.ofList (x.toList.take u.toList.length) = u := by
  rw [List.prefix_iff_eq_take]
  constructor
  · intro h
    refine ⟨?_, ?_⟩
    · rw [h]; simp
    · rw [← h]; simp
  · rintro ⟨-, h⟩
    have := congrArg FreeMonoid.toList h
    simp only [FreeMonoid.toList_ofList] at this
    exact this.symm

/-- The share of draws and anchors whose anchor prefix satisfies `Q`, as a sum over the anchor
strings. -/
theorem anchor_sum (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (L : ℕ)
    (Q : FreeMonoid α → Prop) [DecidablePred Q] :
    (D.prod (anchorLaw L)).real
        {r | r.2 ≤ r.1.toList.length ∧ Q (FreeMonoid.ofList (r.1.toList.take r.2))}
      = ∑ u ∈ anchorStrs L, if Q u then anchorWeight D L u else 0 := by
  classical
  set S := {r : FreeMonoid α × ℕ |
    r.2 ≤ r.1.toList.length ∧ Q (FreeMonoid.ofList (r.1.toList.take r.2))}
  set c : ℝ≥0∞ := ((L + 1 - startLength L : ℕ) : ℝ≥0∞)⁻¹
  set E : ℕ → FreeMonoid α → Set (FreeMonoid α) := fun k u =>
    {x | k ≤ x.toList.length ∧ FreeMonoid.ofList (x.toList.take k) = u}
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hsec : ∀ x, anchorLaw L (Prod.mk x ⁻¹' S)
      = ∑ k ∈ Finset.Icc (startLength L) L,
          Set.indicator
            {y : FreeMonoid α | k ≤ y.toList.length ∧ Q (FreeMonoid.ofList (y.toList.take k))}
            (fun _ => c) x := by
    intro x
    rw [anchorLaw_apply, Finset.card_filter, Nat.cast_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun k _ => ?_
    simp only [Set.indicator, Set.mem_preimage, S, Set.mem_ofPred_eq]
    split_ifs <;> simp [c]
  have hk : ∀ k,
      D {y : FreeMonoid α | k ≤ y.toList.length ∧ Q (FreeMonoid.ofList (y.toList.take k))}
      = ∑ u ∈ strsOfLen k, if Q u then D (E k u) else 0 := by
    intro k
    rw [← Finset.sum_filter]
    have hset : {y : FreeMonoid α | k ≤ y.toList.length ∧ Q (FreeMonoid.ofList (y.toList.take k))}
        = ⋃ u ∈ (strsOfLen k).filter Q, E k u := by
      ext y
      simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Finset.mem_filter, mem_strsOfLen, E]
      constructor
      · rintro ⟨h1, h2⟩
        exact ⟨_, ⟨by simp [h1], h2⟩, h1, rfl⟩
      · rintro ⟨u, ⟨-, hu⟩, h1, rfl⟩
        exact ⟨h1, hu⟩
    rw [hset, measure_biUnion_finset]
    · intro u _ v _ huv
      refine Set.disjoint_left.2 fun y hu hv => huv ?_
      exact hu.2.symm.trans hv.2
    · intro u _; exact (Set.to_countable _).measurableSet
  have htot : (D.prod (anchorLaw L)) S
      = ∑ u ∈ anchorStrs L, if Q u then c * D {x | u.toList <+: x.toList} else 0 := by
    rw [Measure.prod_apply hS]
    simp_rw [hsec]
    rw [lintegral_finsetSum _ fun k _ => (measurable_const.indicator
      (Set.to_countable _).measurableSet)]
    simp_rw [lintegral_indicator_const (Set.to_countable _).measurableSet, hk, Finset.mul_sum]
    rw [anchorStrs, Finset.sum_biUnion]
    · refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun u hu => ?_
      have hl := mem_strsOfLen.1 hu
      split_ifs
      · congr 2
        ext x
        simp only [E, Set.mem_ofPred_eq, prefix_iff_take, hl]
      · simp
    · intro k _ k' _ hkk'
      refine Finset.disjoint_left.2 fun u hu hu' => hkk' ?_
      rw [← mem_strsOfLen.1 hu, ← mem_strsOfLen.1 hu']
  rw [measureReal_def, htot, ENNReal.toReal_sum fun u _ => by
    split_ifs
    · exact ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 (Nat.cast_ne_zero.2 (anchorCount_pos L).ne'))
        (measure_ne_top _ _)
    · simp]
  refine Finset.sum_congr rfl fun u _ => ?_
  split_ifs
  · simp only [c]
    rw [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast, anchorWeight,
      measureReal_def]
    ring
  · simp

theorem anchorWeight_nonneg (D : Measure (FreeMonoid α)) (L : ℕ) (u : FreeMonoid α) :
    0 ≤ anchorWeight D L u :=
  div_nonneg measureReal_nonneg (Nat.cast_nonneg _)

theorem sum_anchorWeight_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ) :
    ∑ u ∈ anchorStrs L, anchorWeight D L u ≤ 1 := by
  have h := anchor_sum D L (fun _ => True)
  simp only [if_true] at h
  rw [← h]
  exact measureReal_le_one

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Over reads independent across strings, each string satisfying `P` reading undecided with
chance at least `c`, and no string past `⌈L/2⌉` starting more than `q` of the draws, the share of
draws and anchors whose anchor prefix reads undecided falls `t` below `c` times the share whose
anchor prefix satisfies `P` with chance at most `exp(−2t²n/q)`. -/
theorem anchor_reads (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : iIndepFun read μ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ)
    (P : FreeMonoid α → Prop) (c : ℝ)
    (hc : ∀ u, P u → c ≤ μ.real {ω | read u ω = .undecided}) (t q : ℝ) (ht : 0 ≤ t)
    (hq : 0 < q)
    (hpre : ∀ u : FreeMonoid α, startLength L ≤ u.toList.length → u.toList.length ≤ L →
      D.real {x | u.toList <+: x.toList} ≤ q) :
    μ.real {ω | (D.prod (anchorLaw L)).real
        {r | r.2 ≤ r.1.toList.length ∧
          read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided}
      < c * (D.prod (anchorLaw L)).real
        {r | r.2 ≤ r.1.toList.length ∧ P (FreeMonoid.ofList (r.1.toList.take r.2))} - t}
      ≤ Real.exp (-2 * t ^ 2 * (L + 1 - startLength L : ℕ) / q) := by
  classical
  have hn : (0 : ℝ) < (L + 1 - startLength L : ℕ) := by exact_mod_cast anchorCount_pos L
  have hw0 : ∀ u, 0 ≤ anchorWeight D L u := anchorWeight_nonneg D L
  let g : FreeMonoid α → ARU → ℝ := fun u a => anchorWeight D L u * if a = .undecided then 1 else 0
  let Z : FreeMonoid α → Ω → ℝ := fun u => g u ∘ read u
  have hZm : ∀ u, Measurable (Z u) := fun u =>
    Measurable.comp (fun _ _ => trivial : Measurable (g u)) (hmeas u)
  have hZI : ∀ u, ∀ᵐ ω ∂μ, Z u ω ∈ Set.Icc 0 (anchorWeight D L u) := fun u =>
      ae_of_all _ fun ω => by
    have := hw0 u
    simp only [Z, g, Function.comp_apply, Set.mem_Icc]
    split_ifs <;> constructor <;> linarith
  have hEZ : ∀ u, μ[Z u] = anchorWeight D L u * μ.real {ω | read u ω = .undecided} := by
    intro u
    have hs : MeasurableSet {ω | read u ω = .undecided} :=
      @hmeas u {ARU.undecided} trivial
    have : Z u = fun ω => anchorWeight D L u
        * Set.indicator {ω | read u ω = .undecided} (fun _ => (1 : ℝ)) ω := by
      ext ω; simp only [Z, g, Function.comp_apply, Set.indicator, Set.mem_ofPred_eq]
    rw [this, integral_const_mul, integral_indicator_const (1 : ℝ) hs, smul_eq_mul, mul_one]
  let V : FreeMonoid α → Ω → ℝ := fun u => (fun a => -(g u a - μ[Z u])) ∘ read u
  have hVind : iIndepFun V μ := hind.comp _ fun u => fun _ _ => trivial
  let cw : FreeMonoid α → ℝ≥0 := fun u => (‖anchorWeight D L u - 0‖₊ / 2) ^ 2
  have hVsub : ∀ u ∈ anchorStrs L, HasSubgaussianMGF (V u) (cw u) μ := fun u _ =>
    (hasSubgaussianMGF_of_mem_Icc (hZm u).aemeasurable (hZI u)).neg
  have hY : ∀ ω, (D.prod (anchorLaw L)).real
      {r | r.2 ≤ r.1.toList.length ∧
        read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided}
      = ∑ u ∈ anchorStrs L, Z u ω := by
    intro ω
    rw [anchor_sum D L (fun u => read u ω = .undecided)]
    refine Finset.sum_congr rfl fun u _ => ?_
    simp only [Z, g, Function.comp_apply]
    split_ifs <;> simp
  have hmean : c * (D.prod (anchorLaw L)).real
      {r | r.2 ≤ r.1.toList.length ∧ P (FreeMonoid.ofList (r.1.toList.take r.2))}
      ≤ ∑ u ∈ anchorStrs L, μ[Z u] := by
    rw [anchor_sum D L P, Finset.mul_sum]
    refine Finset.sum_le_sum fun u _ => ?_
    rw [hEZ]
    split_ifs with hP
    · rw [mul_comm]; exact mul_le_mul_of_nonneg_left (hc u hP) (hw0 u)
    · simp only [mul_zero]; exact mul_nonneg (hw0 u) measureReal_nonneg
  have hVsum : ∀ ω, ∑ u ∈ anchorStrs L, V u ω
      = ∑ u ∈ anchorStrs L, μ[Z u] - ∑ u ∈ anchorStrs L, Z u ω := by
    intro ω
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun u _ => ?_
    simp only [V, Z, Function.comp_apply]
    ring
  have hsub : {ω | (D.prod (anchorLaw L)).real
        {r | r.2 ≤ r.1.toList.length ∧
          read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided}
      < c * (D.prod (anchorLaw L)).real
        {r | r.2 ≤ r.1.toList.length ∧ P (FreeMonoid.ofList (r.1.toList.take r.2))} - t}
      ⊆ {ω | t ≤ ∑ u ∈ anchorStrs L, V u ω} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq, hY] at hω ⊢
    rw [hVsum]
    linarith
  have hw : ∀ u ∈ anchorStrs L, anchorWeight D L u ≤ q / (L + 1 - startLength L : ℕ) := by
    intro u hu
    obtain ⟨k, hk, hu'⟩ := Finset.mem_biUnion.1 hu
    have hl := mem_strsOfLen.1 hu'
    obtain ⟨h1, h2⟩ := Finset.mem_Icc.1 hk
    exact div_le_div_of_nonneg_right (hpre u (hl ▸ h1) (hl ▸ h2)) hn.le
  have hcw : ∀ u, (cw u : ℝ) = anchorWeight D L u ^ 2 / 4 := by
    intro u
    simp only [cw, NNReal.coe_pow, NNReal.coe_div, coe_nnnorm, sub_zero, Real.norm_eq_abs,
      abs_of_nonneg (hw0 u)]
    norm_num
    ring
  have hcsum : (∑ u ∈ anchorStrs L, (cw u : ℝ)) ≤ q / (4 * (L + 1 - startLength L : ℕ)) := by
    have : ∀ u ∈ anchorStrs L, (cw u : ℝ)
        ≤ q / (4 * (L + 1 - startLength L : ℕ)) * anchorWeight D L u := by
      intro u hu
      rw [hcw]
      have h1 := hw u hu
      have h2 := hw0 u
      rw [show q / (4 * ((L + 1 - startLength L : ℕ) : ℝ)) * anchorWeight D L u
        = anchorWeight D L u * (q / (L + 1 - startLength L : ℕ)) / 4 by ring]
      nlinarith
    calc (∑ u ∈ anchorStrs L, (cw u : ℝ))
        ≤ ∑ u ∈ anchorStrs L, q / (4 * (L + 1 - startLength L : ℕ)) * anchorWeight D L u :=
          Finset.sum_le_sum this
      _ = q / (4 * (L + 1 - startLength L : ℕ)) * ∑ u ∈ anchorStrs L, anchorWeight D L u := by
          rw [Finset.mul_sum]
      _ ≤ q / (4 * (L + 1 - startLength L : ℕ)) * 1 :=
          mul_le_mul_of_nonneg_left (sum_anchorWeight_le D L) (by positivity)
      _ = _ := mul_one _
  refine (measureReal_mono hsub).trans ?_
  rcases (Finset.sum_nonneg fun u _ => (cw u).2 : 0 ≤ ∑ u ∈ anchorStrs L, (cw u : ℝ)).lt_or_eq
    with hpos | hzero
  · refine (HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hVind hVsub ht).trans ?_
    refine Real.exp_le_exp.2 ?_
    push_cast
    rw [div_le_div_iff₀ (by positivity) hq]
    have h4 : 2 * t ^ 2 * (2 * ∑ u ∈ anchorStrs L, (cw u : ℝ)) * ((L + 1 - startLength L : ℕ) : ℝ)
        ≤ 2 * t ^ 2 * (2 * (q / (4 * (L + 1 - startLength L : ℕ))))
          * (L + 1 - startLength L : ℕ) := by
      gcongr
    have h5 : 2 * t ^ 2 * (2 * (q / (4 * ((L + 1 - startLength L : ℕ) : ℝ))))
        * (L + 1 - startLength L : ℕ) = t ^ 2 * q := by
      field_simp; ring
    nlinarith
  · have hw0' : ∀ u ∈ anchorStrs L, anchorWeight D L u = 0 := by
      intro u hu
      have h := (Finset.sum_eq_zero_iff_of_nonneg fun u _ => (cw u).2).1 hzero.symm u hu
      have h' := (hcw u).symm.trans h
      have : anchorWeight D L u ^ 2 = 0 := by linarith
      exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 this
    have hV0 : ∀ ω, ∑ u ∈ anchorStrs L, V u ω = 0 := by
      intro ω
      refine Finset.sum_eq_zero fun u hu => ?_
      simp only [V, Function.comp_apply, hEZ, g, hw0' u hu, zero_mul, sub_zero, neg_zero]
    rcases ht.lt_or_eq with htp | rfl
    · have : {ω | t ≤ ∑ u ∈ anchorStrs L, V u ω} = ∅ := by
        ext ω
        simp only [hV0, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_le]
        exact htp
      rw [this, measureReal_empty]; positivity
    · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero,
        zero_mul, zero_div, Real.exp_zero]
      exact measureReal_le_one

end OrthoDFA
