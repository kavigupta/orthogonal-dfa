import OrthoDFA.Proofs.RoundEnd
import Mathlib.Probability.Moments.Basic

/-!
# The anchor's read concentrates

Over reads independent across strings, the share of draws and anchors whose anchor prefix reads
undecided is a weighted sum of independent reads with weights at most `q / n`, where `n` is the
number of anchors and `q` the most a draw puts on starting with any one string at least `s` long,
and Chernoff's bound holds its lower tail.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

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
noncomputable def anchorStrs (s L : ℕ) : Finset (FreeMonoid α) :=
  (Finset.Icc (s) L).biUnion strsOfLen

theorem mem_anchorStrs {s L : ℕ} {u : FreeMonoid α} :
    u ∈ anchorStrs s L ↔ s ≤ u.toList.length ∧ u.toList.length ≤ L := by
  simp only [anchorStrs, Finset.mem_biUnion, Finset.mem_Icc, mem_strsOfLen]
  exact ⟨fun ⟨_, h, rfl⟩ => h, fun h => ⟨_, h, rfl⟩⟩

/-- The anchor prefix is `u` on this share of draws and anchors. -/
noncomputable def anchorWeight (D : Measure (FreeMonoid α)) (s L : ℕ) (u : FreeMonoid α) : ℝ :=
  D.real {x | u.toList <+: x.toList} / (L + 1 - s : ℕ)

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
theorem anchor_sum (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] {s L : ℕ} (hs : s ≤ L)
    (Q : FreeMonoid α → Prop) [DecidablePred Q] :
    (D.prod (uniformAnchor s L)).real
        {r | r.2 ≤ r.1.toList.length ∧ Q (FreeMonoid.ofList (r.1.toList.take r.2))}
      = ∑ u ∈ anchorStrs s L, if Q u then anchorWeight D s L u else 0 := by
  classical
  set S := {r : FreeMonoid α × ℕ |
    r.2 ≤ r.1.toList.length ∧ Q (FreeMonoid.ofList (r.1.toList.take r.2))}
  set c : ℝ≥0∞ := ((L + 1 - s : ℕ) : ℝ≥0∞)⁻¹
  set E : ℕ → FreeMonoid α → Set (FreeMonoid α) := fun k u =>
    {x | k ≤ x.toList.length ∧ FreeMonoid.ofList (x.toList.take k) = u}
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hsec : ∀ x, uniformAnchor s L (Prod.mk x ⁻¹' S)
      = ∑ k ∈ Finset.Icc (s) L,
          Set.indicator
            {y : FreeMonoid α | k ≤ y.toList.length ∧ Q (FreeMonoid.ofList (y.toList.take k))}
            (fun _ => c) x := by
    intro x
    rw [uniformAnchor_apply, Finset.card_filter, Nat.cast_sum, Finset.mul_sum]
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
  have htot : (D.prod (uniformAnchor s L)) S
      = ∑ u ∈ anchorStrs s L, if Q u then c * D {x | u.toList <+: x.toList} else 0 := by
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
    · exact ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 (Nat.cast_ne_zero.2 (anchorCount_pos hs).ne'))
        (measure_ne_top _ _)
    · simp]
  refine Finset.sum_congr rfl fun u _ => ?_
  split_ifs
  · simp only [c]
    rw [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast, anchorWeight,
      measureReal_def]
    ring
  · simp

theorem anchorWeight_nonneg (D : Measure (FreeMonoid α)) (s L : ℕ) (u : FreeMonoid α) :
    0 ≤ anchorWeight D s L u :=
  div_nonneg measureReal_nonneg (Nat.cast_nonneg _)

theorem sum_anchorWeight_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {s L : ℕ}
    (hs : s ≤ L) :
    ∑ u ∈ anchorStrs s L, anchorWeight D s L u ≤ 1 := by
  have h := anchor_sum D hs (fun _ => True)
  simp only [if_true] at h
  rw [← h]
  have := uniformAnchor_isProb hs
  exact measureReal_le_one

theorem one_sub_exp_neg_ge {x : ℝ} (hx : 0 ≤ x) : x * (1 - x) ≤ 1 - Real.exp (-x) := by
  have h1 : Real.exp (-x) ≤ 1 / (1 + x) := by
    rw [Real.exp_neg, ← one_div]
    exact one_div_le_one_div_of_le (by positivity) (by linarith [Real.add_one_le_exp x])
  have h2 : x * (1 - x) ≤ 1 - 1 / (1 + x) := by
    have hx1 : (1 + x) ≠ 0 := by positivity
    rw [show 1 - 1 / (1 + x) = x / (1 + x) by field_simp; ring, le_div_iff₀ (by positivity)]
    nlinarith [pow_nonneg hx 3]
  linarith

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Over reads independent across strings, no string at least `s` long starting more than `q` of
the draws, the share of draws and anchors whose anchor prefix reads undecided is at most `t` with
chance at most `exp(λt − λ(1 − λq/n) m)` at any `λ ≥ 0`, `m` being its mean. -/
theorem anchor_reads (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : iIndepFun read μ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {s L : ℕ}
    (hs : s ≤ L) (q : ℝ)
    (hpre : ∀ u : FreeMonoid α, s ≤ u.toList.length → u.toList.length ≤ L →
      D.real {x | u.toList <+: x.toList} ≤ q) (t lam : ℝ) (hlam : 0 ≤ lam) :
    μ.real {ω | (D.prod (uniformAnchor s L)).real
        {r | r.2 ≤ r.1.toList.length ∧
          read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided} ≤ t}
      ≤ Real.exp (lam * t - lam * (1 - lam * (q / (L + 1 - s : ℕ)))
        * ∑ u ∈ anchorStrs s L, anchorWeight D s L u * μ.real {ω | read u ω = .undecided}) := by
  classical
  set n : ℝ := ((L + 1 - s : ℕ) : ℝ)
  have hn : (0 : ℝ) < n := by simp only [n]; exact_mod_cast anchorCount_pos hs
  set W : ℝ := q / n
  have hw0 : ∀ u, 0 ≤ anchorWeight D s L u := anchorWeight_nonneg D s L
  set p : FreeMonoid α → ℝ := fun u => μ.real {ω | read u ω = .undecided}
  let g : FreeMonoid α → ARU → ℝ := fun u a =>
    anchorWeight D s L u * if a = .undecided then 1 else 0
  let Z : FreeMonoid α → Ω → ℝ := fun u => g u ∘ read u
  have hZm : ∀ u, Measurable (Z u) := fun u =>
    Measurable.comp (fun _ _ => trivial : Measurable (g u)) (hmeas u)
  have hZind : iIndepFun Z μ := hind.comp _ fun u => fun _ _ => trivial
  have hY : ∀ ω, (D.prod (uniformAnchor s L)).real
      {r | r.2 ≤ r.1.toList.length ∧
        read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided}
      = ∑ u ∈ anchorStrs s L, Z u ω := by
    intro ω
    rw [anchor_sum D hs (fun u => read u ω = .undecided)]
    refine Finset.sum_congr rfl fun u _ => ?_
    simp only [Z, g, Function.comp_apply]
    split_ifs <;> simp
  have hw : ∀ u ∈ anchorStrs s L, anchorWeight D s L u ≤ W := by
    intro u hu
    obtain ⟨k, hk, hu'⟩ := Finset.mem_biUnion.1 hu
    have hl := mem_strsOfLen.1 hu'
    obtain ⟨h1, h2⟩ := Finset.mem_Icc.1 hk
    exact div_le_div_of_nonneg_right (hpre u (hl ▸ h1) (hl ▸ h2)) hn.le
  have hmgf : ∀ u ∈ anchorStrs s L, mgf (Z u) μ (-lam)
      ≤ Real.exp (-(lam * (1 - lam * W)) * (anchorWeight D s L u * p u)) := by
    intro u hu
    have hs : MeasurableSet {ω | read u ω = .undecided} := @hmeas u {ARU.undecided} trivial
    have hfun : (fun ω => Real.exp (-lam * Z u ω)) = fun ω =>
        1 + Set.indicator {ω | read u ω = .undecided}
          (fun _ => Real.exp (-lam * anchorWeight D s L u) - 1) ω := by
      ext ω
      simp only [Z, g, Function.comp_apply, Set.indicator, Set.mem_ofPred_eq]
      split_ifs <;> simp
    have hm : mgf (Z u) μ (-lam) = 1 + p u * (Real.exp (-lam * anchorWeight D s L u) - 1) := by
      unfold mgf
      rw [hfun, integral_add (integrable_const _) ((integrable_const _).indicator hs),
        integral_const, integral_indicator_const _ hs, smul_eq_mul, smul_eq_mul]
      simp [p]
    rw [hm]
    have hx := one_sub_exp_neg_ge (mul_nonneg hlam (hw0 u))
    have hp0 : 0 ≤ p u := measureReal_nonneg
    have hwu := hw u hu
    have h1 : lam * anchorWeight D s L u * (1 - lam * W)
        ≤ lam * anchorWeight D s L u * (1 - lam * anchorWeight D s L u) :=
      mul_le_mul_of_nonneg_left (by nlinarith) (mul_nonneg hlam (hw0 u))
    have h2 : Real.exp (-lam * anchorWeight D s L u) - 1
        ≤ -(lam * anchorWeight D s L u * (1 - lam * W)) := by
      rw [show -lam * anchorWeight D s L u = -(lam * anchorWeight D s L u) by ring]
      linarith
    calc 1 + p u * (Real.exp (-lam * anchorWeight D s L u) - 1)
        ≤ Real.exp (p u * (Real.exp (-lam * anchorWeight D s L u) - 1)) := by
          linarith [Real.add_one_le_exp (p u * (Real.exp (-lam * anchorWeight D s L u) - 1))]
      _ ≤ _ := by
          refine Real.exp_le_exp.2 ?_
          have := mul_le_mul_of_nonneg_left h2 hp0
          linarith
  have hint : Integrable (fun ω => Real.exp (-lam * ∑ u ∈ anchorStrs s L, Z u ω)) μ := by
    refine Integrable.of_bound (C := 1) ?_ (ae_of_all _ fun ω => ?_)
    · exact (measurable_const.mul (Finset.measurable_sum _ fun u _ => hZm u)).exp
        |>.aestronglyMeasurable
    · have : 0 ≤ ∑ u ∈ anchorStrs s L, Z u ω := Finset.sum_nonneg fun u _ => by
        simp only [Z, g, Function.comp_apply]; split_ifs <;> simp [hw0 u]
      rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
      exact Real.exp_le_one_iff.2 (by nlinarith)
  have hsub : {ω | (D.prod (uniformAnchor s L)).real
        {r | r.2 ≤ r.1.toList.length ∧
          read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided} ≤ t}
      ⊆ {ω | (∑ u ∈ anchorStrs s L, Z u) ω ≤ t} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq, hY, Finset.sum_apply] at hω ⊢
    exact hω
  refine (measureReal_mono hsub).trans ((measure_le_le_exp_mul_mgf t
    (neg_nonpos.2 hlam) (by simpa [Finset.sum_apply] using hint)).trans ?_)
  rw [hZind.mgf_sum hZm]
  calc Real.exp (- -lam * t) * ∏ u ∈ anchorStrs s L, mgf (Z u) μ (-lam)
      ≤ Real.exp (lam * t) * ∏ u ∈ anchorStrs s L,
          Real.exp (-(lam * (1 - lam * W)) * (anchorWeight D s L u * p u)) := by
        rw [neg_neg]
        exact mul_le_mul_of_nonneg_left (Finset.prod_le_prod (fun u _ => mgf_nonneg) hmgf)
          (Real.exp_pos _).le
    _ = _ := by
        have hs : ∑ u ∈ anchorStrs s L, -(lam * (1 - lam * W)) * (anchorWeight D s L u * p u)
            = -(lam * (1 - lam * W)) * ∑ u ∈ anchorStrs s L, anchorWeight D s L u * p u := by
          rw [Finset.mul_sum]
        rw [← Real.exp_sum, ← Real.exp_add, hs]
        congr 1
        ring

end OrthoDFA
