import OrthoDFA.ReturnAccuracy
import OrthoDFA.Proofs.ClusteringQuality

/-!
# What a returned hypothesis is worth: the proof

A state's reads share one persistent noise draw, so they are independent only at distinct
strings.  On distinct draws the reads have the law of the fresh-noise model, where every read
draws its own string and its own noise (`coupling_le`), and there they are i.i.d. with mean
at least `½ + γ` on the majority side.  Repeats are paid for by the atom bound.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-! ## Counting the reads that came back `1` -/

/-- How many of `n` reads came back `1`. -/
noncomputable def hits {n : ℕ} (y : Fin n → ℝ) : ℝ := ∑ i, if y i = 1 then 1 else 0

lemma measurable_hits (n : ℕ) : Measurable (hits (n := n)) :=
  Finset.measurable_sum _ fun i _ =>
    Measurable.ite (measurableSet_eq_fun (measurable_pi_apply i) measurable_const)
      measurable_const measurable_const

omit [IsProbabilityMeasure μ] in
lemma denoisedLabel_iff (O : Oracle μ S) (n : ℕ) (d : ℕ → S) (ω : Ω) :
    denoisedLabel O n d ω ↔ (n : ℝ) < 2 * hits (fun i : Fin n => O.mq (d i) ω) := by
  classical
  have hcard : hits (fun i : Fin n => O.mq (d i) ω)
      = (((Finset.range n).filter fun i => O.mq (d i) ω = 1).card : ℝ) := by
    rw [hits, Fin.sum_univ_eq_sum_range (fun i => if O.mq (d i) ω = 1 then (1 : ℝ) else 0) n,
      Finset.sum_boole]
  rw [hcard, denoisedLabel]
  exact_mod_cast Iff.rfl

/-! ## One noise draw against a fresh one per read -/

/-- The reads of `n` strings, all under one noise draw. -/
noncomputable def reads {n : ℕ} (O : Oracle μ S) (p : Ω × (Fin n → S)) : Fin n → ℝ :=
  fun i => O.mq (p.2 i) p.1

/-- The reads of `n` strings, each under its own noise draw. -/
noncomputable def freshReads {n : ℕ} (O : Oracle μ S) (q : (Fin n → S) × (Fin n → Ω)) : Fin n → ℝ :=
  fun i => O.mq (q.1 i) (q.2 i)

lemma measurable_reads {n : ℕ} (O : Oracle μ S) : Measurable (reads (n := n) O) :=
  measurable_from_prod_countable_left fun s =>
    measurable_pi_lambda _ fun i => mq_meas O (s i)

lemma measurable_freshReads {n : ℕ} (O : Oracle μ S) : Measurable (freshReads (n := n) O) :=
  measurable_from_prod_countable_right fun s =>
    measurable_pi_lambda _ fun i => (mq_meas O (s i)).comp (measurable_pi_apply i)

/-- At distinct strings the persistent noise bits are independent, so one draw reads them
exactly as fresh draws would. -/
lemma noise_law_eq (O : Oracle μ S) {n : ℕ} {s : Fin n → S} (hs : Function.Injective s)
    {T : Set (Fin n → ℝ)} (hT : MeasurableSet T) :
    μ {ω | (fun i => O.mq (s i) ω) ∈ T}
      = (Measure.pi fun _ : Fin n => μ) {ω' | (fun i => O.mq (s i) (ω' i)) ∈ T} := by
  have : ∀ i, IsProbabilityMeasure (μ.map (O.mq (s i))) := fun i =>
    Measure.isProbabilityMeasure_map (mq_meas O (s i)).aemeasurable
  have hind : iIndepFun (fun i => O.mq (s i)) μ := (mq_indep O).precomp hs
  have h1 := (iIndepFun_iff_map_fun_eq_pi_map (fun i => (mq_meas O (s i)).aemeasurable)).1 hind
  have h2 : (Measure.pi fun _ : Fin n => μ).map (fun ω' i => O.mq (s i) (ω' i))
      = Measure.pi (fun i => μ.map (O.mq (s i))) :=
    Measure.pi_map_pi (fun i => (mq_meas O (s i)).aemeasurable)
  have hm1 : Measurable (fun ω i => O.mq (s i) ω) :=
    measurable_pi_lambda _ fun i => mq_meas O (s i)
  have hm2 : Measurable (fun (ω' : Fin n → Ω) i => O.mq (s i) (ω' i)) :=
    measurable_pi_lambda _ fun i => (mq_meas O (s i)).comp (measurable_pi_apply i)
  change μ ((fun ω i => O.mq (s i) ω) ⁻¹' T)
    = (Measure.pi fun _ : Fin n => μ) ((fun (ω' : Fin n → Ω) i => O.mq (s i) (ω' i)) ⁻¹' T)
  rw [← Measure.map_apply hm1 hT, ← Measure.map_apply hm2 hT, h1, h2]

/-- The coupling: the reads under one noise draw land in `T` no more often than under fresh
draws, once the draws repeating is paid for. -/
lemma coupling_le (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] (n : ℕ)
    {T : Set (Fin n → ℝ)} (hT : MeasurableSet T) :
    (μ.prod (Measure.pi fun _ : Fin n => ρ)) (reads O ⁻¹' T)
      ≤ (Measure.pi fun _ : Fin n => ρ) {s | ¬ Function.Injective s}
        + ((Measure.pi fun _ : Fin n => ρ).prod (Measure.pi fun _ : Fin n => μ))
          (freshReads O ⁻¹' T) := by
  set π := Measure.pi fun _ : Fin n => ρ
  have hB : MeasurableSet {s : Fin n → S | ¬ Function.Injective s} :=
    (Set.to_countable _).measurableSet
  rw [Measure.prod_apply_symm (measurable_reads O hT),
    Measure.prod_apply (measurable_freshReads O hT), ← lintegral_indicator_one hB,
    ← lintegral_add_left (measurable_one.indicator hB)]
  refine lintegral_mono fun s => ?_
  by_cases hs : Function.Injective s
  · have hs' : s ∉ {s : Fin n → S | ¬ Function.Injective s} := fun h => h hs
    rw [Set.indicator_of_notMem hs', zero_add]
    exact (noise_law_eq O hs hT).le
  · have hs' : s ∈ {s : Fin n → S | ¬ Function.Injective s} := hs
    rw [Set.indicator_of_mem hs', Pi.one_apply]
    exact le_add_right prob_le_one

/-! ## Repeats -/

lemma tsum_measureReal_singleton (ρ : Measure S) [IsProbabilityMeasure ρ] :
    ∑' a : S, ρ.real {a} = 1 := by
  have h := tsum_singleton_eq ρ Set.univ
  rw [tsum_univ (f := fun a : S => ρ {a}), measure_univ] at h
  simp only [measureReal_def]
  rw [← ENNReal.tsum_toReal_eq (fun a => measure_ne_top ρ {a}), h, ENNReal.toReal_one]

lemma collisionMass_le (ρ : Measure S) [IsProbabilityMeasure ρ] {c : ℝ}
    (hc : ∀ a, ρ.real {a} ≤ c) : collisionMass ρ ≤ c := by
  have hle : ∀ a : S, ρ.real {a} ^ 2 ≤ c * ρ.real {a} := fun a => by
    rw [sq]; exact mul_le_mul_of_nonneg_right (hc a) measureReal_nonneg
  calc collisionMass ρ = ∑' a : S, ρ.real {a} ^ 2 := rfl
    _ ≤ ∑' a : S, c * ρ.real {a} :=
      (summable_singleton_sq ρ).tsum_le_tsum hle ((summable_singleton_real ρ).mul_left c)
    _ = c := by rw [tsum_mul_left, tsum_measureReal_singleton, mul_one]

/-- Unordered pairs: `n` draws repeat with probability at most `(n²/2)·max_a ρ{a}`. -/
lemma pi_not_injective_le_half (ρ : Measure S) [IsProbabilityMeasure ρ] (n : ℕ) {c : ℝ}
    (hc : ∀ a, ρ.real {a} ≤ c) :
    (Measure.pi fun _ : Fin n => ρ).real {s | ¬ Function.Injective s} ≤ (n : ℝ) ^ 2 / 2 * c := by
  classical
  have hc0 : 0 ≤ c := le_trans measureReal_nonneg (hc 1)
  set P := (Finset.univ : Finset (Fin n × Fin n)).filter (fun q => q.1 < q.2)
  set P' := (Finset.univ : Finset (Fin n × Fin n)).filter (fun q => q.2 < q.1)
  have hsub : {s : Fin n → S | ¬ Function.Injective s} ⊆ ⋃ q ∈ P, {s | s q.1 = s q.2} := by
    intro s hs
    simp only [Set.mem_ofPred_eq, Function.Injective, not_forall] at hs
    obtain ⟨a, b, hab, hne⟩ := hs
    rcases lt_or_gt_of_ne hne with h | h
    · exact Set.mem_biUnion (show (a, b) ∈ P by simp [P, h]) hab
    · exact Set.mem_biUnion (show (b, a) ∈ P by simp [P, h]) hab.symm
  have hPP' : P.card = P'.card :=
    Finset.card_nbij' Prod.swap Prod.swap (by intro q hq; simpa [P, P'] using hq)
      (by intro q hq; simpa [P, P'] using hq) (fun q _ => rfl) (fun q _ => rfl)
  have hdisj : Disjoint P P' := by
    rw [Finset.disjoint_left]
    intro q hq hq'
    simp only [P, P', Finset.mem_filter] at hq hq'
    exact lt_asymm hq.2 hq'.2
  have hcard : 2 * (P.card : ℝ) ≤ (n : ℝ) ^ 2 := by
    have h1 : (P ∪ P').card ≤ n * n := by
      have := Finset.card_le_univ (P ∪ P')
      simpa using this
    rw [Finset.card_union_of_disjoint hdisj, ← hPP'] at h1
    have h2 : ((2 * P.card : ℕ) : ℝ) ≤ ((n * n : ℕ) : ℝ) := by exact_mod_cast (by omega)
    push_cast at h2
    nlinarith
  calc (Measure.pi fun _ : Fin n => ρ).real {s | ¬ Function.Injective s}
      ≤ (Measure.pi fun _ : Fin n => ρ).real (⋃ q ∈ P, {s : Fin n → S | s q.1 = s q.2}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ q ∈ P, (Measure.pi fun _ : Fin n => ρ).real {s : Fin n → S | s q.1 = s q.2} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _q ∈ P, c := Finset.sum_le_sum fun q hq => by
        rw [pi_coord_eq ρ (Finset.mem_filter.mp hq).2.ne]
        exact collisionMass_le ρ hc
    _ = (P.card : ℝ) * c := by rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (n : ℝ) ^ 2 / 2 * c := mul_le_mul_of_nonneg_right (by linarith) hc0


/-! ## The fresh-noise model: i.i.d. reads -/

/-- `1` when the read of `p.1` under noise `p.2` comes back `1`. -/
noncomputable def hitBit (O : Oracle μ S) (p : S × Ω) : ℝ := if O.mq p.1 p.2 = 1 then 1 else 0

lemma measurable_hitBit (O : Oracle μ S) : Measurable (hitBit O) :=
  measurable_from_prod_countable_right fun v =>
    Measurable.ite (measurableSet_eq_fun (mq_meas O v) measurable_const)
      measurable_const measurable_const

omit [IsProbabilityMeasure μ] in
lemma hitBit_mem_Icc (O : Oracle μ S) (p : S × Ω) : hitBit O p ∈ Set.Icc (0 : ℝ) 1 := by
  unfold hitBit; split_ifs <;> simp

/-- The chance a read of a `ρ`-drawn string comes back `1`. -/
noncomputable def hitRate (O : Oracle μ S) (ρ : Measure S) : ℝ :=
  ∫ v, μ.real {ω | O.mq v ω = 1} ∂ρ

lemma integral_hitBit (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] :
    ∫ p, hitBit O p ∂(ρ.prod μ) = hitRate O ρ := by
  have hint : Integrable (hitBit O) (ρ.prod μ) :=
    Integrable.of_mem_Icc 0 1 (measurable_hitBit O).aemeasurable
      (ae_of_all _ (hitBit_mem_Icc O))
  rw [integral_prod _ hint, hitRate]
  refine integral_congr_ae (ae_of_all _ fun v => ?_)
  have hset : MeasurableSet {ω | O.mq v ω = 1} :=
    measurableSet_eq_fun (mq_meas O v) measurable_const
  have hfun : (fun ω => hitBit O (v, ω)) = {ω | O.mq v ω = 1}.indicator 1 := by
    funext ω; simp [hitBit, Set.indicator_apply]
  change ∫ ω, (fun ω => hitBit O (v, ω)) ω ∂μ = _
  rw [hfun, integral_indicator_one hset]

lemma integral_hitBit_eval (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] {n : ℕ}
    (i : Fin n) :
    ∫ x, hitBit O (x i) ∂(Measure.pi fun _ : Fin n => ρ.prod μ) = hitRate O ρ := by
  have hev := measurePreserving_eval (fun _ : Fin n => ρ.prod μ) i
  calc ∫ x, hitBit O (x i) ∂(Measure.pi fun _ : Fin n => ρ.prod μ)
      = ∫ p, hitBit O p ∂((Measure.pi fun _ : Fin n => ρ.prod μ).map (Function.eval i)) :=
        (integral_map hev.measurable.aemeasurable
          (measurable_hitBit O).aestronglyMeasurable).symm
    _ = hitRate O ρ := by rw [hev.map_eq, integral_hitBit]

lemma iIndepFun_hitBit (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] (n : ℕ) :
    iIndepFun (fun (i : Fin n) (x : Fin n → S × Ω) => hitBit O (x i))
      (Measure.pi fun _ : Fin n => ρ.prod μ) :=
  iIndepFun_pi (X := fun _ => hitBit O) (fun _ => (measurable_hitBit O).aemeasurable)

omit [IsProbabilityMeasure μ] in
lemma hits_freshReads (O : Oracle μ S) {n : ℕ} (x : Fin n → S × Ω) :
    hits (freshReads O (MeasurableEquiv.arrowProdEquivProdArrow S Ω (Fin n) x))
      = ∑ i, hitBit O (x i) := rfl

/-- Reads leaning towards `1` by `γ` come back `1` at most half the time with probability at
most `exp(−2nγ²)`. -/
theorem fresh_lower_tail (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] (n : ℕ)
    {γ : ℝ} (hγ : 0 ≤ γ) (hmean : 1 / 2 + γ ≤ hitRate O ρ) :
    ((Measure.pi fun _ : Fin n => ρ).prod (Measure.pi fun _ : Fin n => μ)).real
        (freshReads O ⁻¹' {y | 2 * hits y ≤ n}) ≤ Real.exp (-2 * n * γ ^ 2) := by
  have hmp := measurePreserving_arrowProdEquivProdArrow S Ω (Fin n) (fun _ => ρ) (fun _ => μ)
  have hT : MeasurableSet (freshReads O ⁻¹' {y : Fin n → ℝ | 2 * hits y ≤ n}) :=
    measurable_freshReads O
      (measurableSet_le (measurable_const.mul (measurable_hits n)) measurable_const)
  rw [← hmp.measureReal_preimage hT.nullMeasurableSet]
  have h := sumLower_le (fun (i : Fin n) (x : Fin n → S × Ω) => hitBit O (x i)) Finset.univ
    (1 / 2 + γ) γ (fun i => ((measurable_hitBit O).comp (measurable_pi_apply i)).aemeasurable)
    (iIndepFun_hitBit O ρ n) (fun i => ae_of_all _ fun x => hitBit_mem_Icc O (x i))
    (by
      simp only [integral_hitBit_eval, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul]
      exact mul_le_mul_of_nonneg_left hmean (Nat.cast_nonneg n))
    hγ
  simp only [Finset.card_univ, Fintype.card_fin] at h
  refine le_trans (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _)) h
  simp only [Set.mem_preimage, Set.mem_ofPred_eq, hits_freshReads] at hx ⊢
  linarith

/-- Reads leaning towards `0` by `γ` come back `1` more than half the time with probability at
most `exp(−2nγ²)`. -/
theorem fresh_upper_tail (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] (n : ℕ)
    {γ : ℝ} (hγ : 0 ≤ γ) (hmean : hitRate O ρ ≤ 1 / 2 - γ) :
    ((Measure.pi fun _ : Fin n => ρ).prod (Measure.pi fun _ : Fin n => μ)).real
        (freshReads O ⁻¹' {y | (n : ℝ) < 2 * hits y}) ≤ Real.exp (-2 * n * γ ^ 2) := by
  have hmp := measurePreserving_arrowProdEquivProdArrow S Ω (Fin n) (fun _ => ρ) (fun _ => μ)
  have hT : MeasurableSet (freshReads O ⁻¹' {y : Fin n → ℝ | (n : ℝ) < 2 * hits y}) :=
    measurable_freshReads O
      (measurableSet_lt measurable_const (measurable_const.mul (measurable_hits n)))
  rw [← hmp.measureReal_preimage hT.nullMeasurableSet]
  have h := sumUpper_le (fun (i : Fin n) (x : Fin n → S × Ω) => hitBit O (x i)) Finset.univ
    (1 / 2 - γ) γ (fun i => ((measurable_hitBit O).comp (measurable_pi_apply i)).aemeasurable)
    (iIndepFun_hitBit O ρ n) (fun i => ae_of_all _ fun x => hitBit_mem_Icc O (x i))
    (by
      simp only [integral_hitBit_eval, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul]
      exact mul_le_mul_of_nonneg_left hmean (Nat.cast_nonneg n))
    hγ
  simp only [Finset.card_univ, Fintype.card_fin] at h
  refine le_trans (measureReal_mono (fun x hx => ?_) (measure_ne_top _ _)) h
  simp only [Set.mem_preimage, Set.mem_ofPred_eq, hits_freshReads] at hx ⊢
  linarith

/-- A string in the language reads `1` at rate at least `1 − η₀`, one outside at most `η₀`. -/
lemma hitRate_bounds (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] {η₀ : ℝ}
    (hη : O.η ≤ η₀) :
    (1 - η₀) * ρ.real O.L ≤ hitRate O ρ ∧ hitRate O ρ ≤ η₀ + (1 - η₀) * ρ.real O.L := by
  have hpt : ∀ v, (1 - η₀) * O.L.indicator 1 v ≤ μ.real {ω | O.mq v ω = 1}
      ∧ μ.real {ω | O.mq v ω = 1} ≤ η₀ + (1 - η₀) * O.L.indicator 1 v := by
    intro v
    rw [measureReal_mq_eq_one, mq_mean]
    have h0 := O.rate_nonneg v
    have h1 := (O.rate_le_eta v).trans hη
    unfold Oracle.label
    by_cases hv : v ∈ O.L
    · simp only [Set.indicator_of_mem hv, Pi.one_apply]
      constructor <;> linarith
    · simp only [Set.indicator_of_notMem hv, mul_zero, add_zero]
      constructor <;> linarith
  have hq : Integrable (fun v => μ.real {ω | O.mq v ω = 1}) ρ :=
    Integrable.of_mem_Icc 0 1 (measurable_of_countable _).aemeasurable
      (ae_of_all _ fun v => ⟨measureReal_nonneg, measureReal_le_one⟩)
  have hind : Integrable (O.L.indicator (1 : S → ℝ)) ρ :=
    (integrable_const (1 : ℝ)).indicator O.L_meas
  have hI : ∫ v, (1 - η₀) * O.L.indicator 1 v ∂ρ = (1 - η₀) * ρ.real O.L := by
    rw [integral_const_mul, integral_indicator_one O.L_meas]
  constructor
  · rw [← hI]; exact integral_mono (hind.const_mul _) hq fun v => (hpt v).1
  · calc hitRate O ρ ≤ ∫ v, (η₀ + (1 - η₀) * O.L.indicator 1 v) ∂ρ :=
          integral_mono hq ((integrable_const η₀).add (hind.const_mul _)) fun v => (hpt v).2
      _ = η₀ + (1 - η₀) * ρ.real O.L := by
          rw [integral_add (integrable_const η₀) (hind.const_mul _), hI]
          simp

/-! ## A state's own sampler -/

section States
variable {Q R : Type*} [Fintype Q] [Fintype R]

omit [Fintype R] in
lemma isProbabilityMeasure_reaching (H : DFA S R) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] (h : R) : IsProbabilityMeasure (reaching H Dsamp h) := by
  unfold reaching
  split_ifs with h0
  · infer_instance
  · exact cond_isProbabilityMeasure h0

omit [Fintype R] in
lemma reaching_real_apply (H : DFA S R) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    {h : R} (h0 : Dsamp {v | H.state v = h} ≠ 0) (t : Set S) :
    Dsamp.real ({v | H.state v = h} ∩ t)
      = Dsamp.real {v | H.state v = h} * (reaching H Dsamp h).real t := by
  have ha : (Dsamp {v | H.state v = h}).toReal ≠ 0 :=
    ENNReal.toReal_ne_zero.2 ⟨h0, measure_ne_top _ _⟩
  rw [reaching, if_neg h0, measureReal_def, measureReal_def, measureReal_def,
    cond_apply (measurableSet_of_countable _), ENNReal.toReal_mul, ENNReal.toReal_inv,
    ← mul_assoc, mul_inv_cancel₀ ha, one_mul]

omit [Fintype Q] [Fintype R] in
/-- Labelled with its majority, a state whose minority share is below `w` is right on all but
`w` of its mass. -/
lemma contrib_ge (A : DFA S Q) (H : DFA S R) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    {h : R} {w : ℝ} (h0 : Dsamp {v | H.state v = h} ≠ 0) (hshare : minorityShare A H Dsamp h < w)
    (lab : Prop) (hlab : lab ↔ (reaching H Dsamp h).real {v | A.state v ∉ A.accept} < w) :
    (1 - w) * Dsamp.real {v | H.state v = h}
      ≤ Dsamp.real ({v | H.state v = h} ∩ {v | lab ↔ A.state v ∈ A.accept}) := by
  have := isProbabilityMeasure_reaching H Dsamp h
  set ρ := reaching H Dsamp h
  set m := Dsamp.real {v | H.state v = h}
  have hm : 0 ≤ m := measureReal_nonneg
  have hsum : ρ.real {v | A.state v ∈ A.accept} + ρ.real {v | A.state v ∉ A.accept} = 1 :=
    by
      have := measureReal_add_measureReal_compl (μ := ρ)
        (measurableSet_of_countable {v | A.state v ∈ A.accept})
      rw [probReal_univ] at this
      exact this
  by_cases hl : lab
  · have hset : {v | lab ↔ A.state v ∈ A.accept} = {v | A.state v ∈ A.accept} := by
      ext v; simp [hl]
    rw [hset, reaching_real_apply H Dsamp h0]
    have := hlab.1 hl
    nlinarith
  · have hset : {v | lab ↔ A.state v ∈ A.accept} = {v | A.state v ∉ A.accept} := by
      ext v; simp [hl]
    rw [hset, reaching_real_apply H Dsamp h0]
    have hnot : ¬ ρ.real {v | A.state v ∉ A.accept} < w := fun h' => hl (hlab.2 h')
    have hacc : ρ.real {v | A.state v ∈ A.accept} < w := by
      rcases min_lt_iff.1 hshare with h' | h'
      · exact h'
      · exact absurd h' hnot
    nlinarith

omit [Fintype Q] in
/-- States lighter than `ε/|R|` carry at most `ε` between them, and every heavier state loses
at most `w` of its mass. -/
lemma accuracy_ge (A : DFA S Q) (H : DFA S R) (label : R → Prop) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] {w ε : ℝ} (hw0 : 0 ≤ w) (hε : 0 ≤ ε)
    (hgood : ∀ h, ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} →
      (1 - w) * Dsamp.real {v | H.state v = h}
        ≤ Dsamp.real ({v | H.state v = h} ∩ {v | label h ↔ A.state v ∈ A.accept})) :
    1 - w - ε ≤ accuracy A H label Dsamp := by
  classical
  have hne : Nonempty R := ⟨H.state 1⟩
  have hN : (0 : ℝ) < Fintype.card R := by exact_mod_cast Fintype.card_pos
  set m : R → ℝ := fun h => Dsamp.real {v | H.state v = h}
  set c : R → ℝ := fun h =>
    Dsamp.real ({v | H.state v = h} ∩ {v | label h ↔ A.state v ∈ A.accept})
  have hm1 : ∑ h, m h = 1 := by
    have := sum_measureReal_preimage_singleton (μ := Dsamp) (Finset.univ : Finset R)
      (f := H.state) (fun y _ => measurableSet_of_countable _)
    simp only [Finset.coe_univ, Set.preimage_univ, probReal_univ] at this
    exact this
  have hacc : accuracy A H label Dsamp = ∑ h, c h := by
    have hU : {v | label (H.state v) ↔ A.state v ∈ A.accept}
        = ⋃ h ∈ (Finset.univ : Finset R),
            ({v | H.state v = h} ∩ {v | label h ↔ A.state v ∈ A.accept}) := by
      ext v
      simp only [Set.mem_ofPred_eq, Finset.mem_univ, Set.iUnion_true, Set.mem_iUnion,
        Set.mem_inter_iff]
      exact ⟨fun hv => ⟨H.state v, rfl, hv⟩, fun ⟨h, hh, hv⟩ => hh ▸ hv⟩
    rw [accuracy, hU, measureReal_biUnion_finset _ (fun _ _ => measurableSet_of_countable _)]
    intro a _ b _ hab
    exact Set.disjoint_left.2 fun v ha hb => hab (ha.1.symm.trans hb.1)
  set light : R → ℝ := fun h => if m h < ε / Fintype.card R then m h else 0
  have hlight : ∑ h, light h ≤ ε := by
    calc ∑ h, light h ≤ ∑ _h : R, ε / Fintype.card R :=
          Finset.sum_le_sum fun h _ => by
            simp only [light]; split_ifs with hh
            · exact hh.le
            · exact div_nonneg hε hN.le
      _ = ε := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; field_simp
  have hlight0 : 0 ≤ ∑ h, light h :=
    Finset.sum_nonneg fun h _ => by
      simp only [light]; split_ifs
      · exact measureReal_nonneg
      · exact le_rfl
  have hpt : ∀ h, (1 - w) * m h - (1 - w) * light h ≤ c h := by
    intro h
    simp only [light]
    split_ifs with hh
    · simp only [sub_self]; exact measureReal_nonneg
    · rw [mul_zero, sub_zero]; exact hgood h (not_lt.1 hh)
  have hsum := Finset.sum_le_sum fun h (_ : h ∈ Finset.univ) => hpt h
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hm1] at hsum
  rw [hacc]
  nlinarith

end States

/-! ## One state's stream -/

/-- The reads that make denoising disagree with `M`. -/
def wrongReads (n : ℕ) (M : Prop) : Set (Fin n → ℝ) := {y | ¬ ((n : ℝ) < 2 * hits y ↔ M)}

lemma wrongReads_eq (n : ℕ) (M : Prop) [Decidable M] :
    wrongReads n M = if M then {y | 2 * hits y ≤ n} else {y | (n : ℝ) < 2 * hits y} := by
  ext y
  by_cases hm : M <;> simp [wrongReads, hm]

lemma measurableSet_wrongReads (n : ℕ) (M : Prop) : MeasurableSet (wrongReads n M) := by
  classical
  rw [wrongReads_eq]
  split_ifs
  · exact measurableSet_le (measurable_const.mul (measurable_hits n)) measurable_const
  · exact measurableSet_lt measurable_const (measurable_const.mul (measurable_hits n))

omit [IsProbabilityMeasure μ] in
lemma wrong_eq_preimage (O : Oracle μ S) (n : ℕ) (M : Prop) :
    {p : Ω × (ℕ → S) | ¬ (denoisedLabel O n p.2 p.1 ↔ M)}
      = Prod.map id (fun (p : ℕ → S) (i : Fin n) => p i.val) ⁻¹' (reads O ⁻¹' wrongReads n M) := by
  ext p
  simp only [Set.mem_ofPred_eq, Set.mem_preimage, wrongReads, denoisedLabel_iff]
  rfl

lemma measurableSet_wrong (O : Oracle μ S) (n : ℕ) (M : Prop) :
    MeasurableSet {p : Ω × (ℕ → S) | ¬ (denoisedLabel O n p.2 p.1 ↔ M)} := by
  rw [wrong_eq_preimage]
  exact (measurable_id.prodMap (by fun_prop)) (measurable_reads O (measurableSet_wrongReads n M))

/-- A stream of `n` draws from `ρ` is denoised against `M` with probability at most the repeats
plus one Hoeffding tail, when the reads lean towards `M` by `γ`. -/
theorem stream_wrong_le (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] (n : ℕ)
    {c γ : ℝ} (M : Prop) (hc : ∀ a, ρ.real {a} ≤ c) (hγ : 0 ≤ γ)
    (hM : M → 1 / 2 + γ ≤ hitRate O ρ) (hnM : ¬ M → hitRate O ρ ≤ 1 / 2 - γ) :
    (μ.prod (Measure.infinitePi fun _ : ℕ => ρ)).real {p | ¬ (denoisedLabel O n p.2 p.1 ↔ M)}
      ≤ (n : ℝ) ^ 2 / 2 * c + Real.exp (-2 * n * γ ^ 2) := by
  classical
  have hT := measurableSet_wrongReads n M
  have hfin := (MeasurePreserving.id μ).prod (measurePreserving_finRestrict ρ n)
  rw [wrong_eq_preimage, hfin.measureReal_preimage (measurable_reads O hT).nullMeasurableSet]
  have hreal : (μ.prod (Measure.pi fun _ : Fin n => ρ)).real (reads O ⁻¹' wrongReads n M)
      ≤ (Measure.pi fun _ : Fin n => ρ).real {s | ¬ Function.Injective s}
        + ((Measure.pi fun _ : Fin n => ρ).prod (Measure.pi fun _ : Fin n => μ)).real
          (freshReads O ⁻¹' wrongReads n M) := by
    simp only [measureReal_def]
    rw [← ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _)]
    exact ENNReal.toReal_mono (by finiteness) (coupling_le O ρ n hT)
  refine hreal.trans (add_le_add (pi_not_injective_le_half ρ n hc) ?_)
  by_cases hm : M
  · rw [wrongReads_eq, if_pos hm]; exact fresh_lower_tail O ρ n hγ (hM hm)
  · rw [wrongReads_eq, if_neg hm]; exact fresh_upper_tail O ρ n hγ (hnM hm)

/-! ## The theorem -/

theorem return_accuracy_holds : ReturnAccuracy := by
  intro Ω _ μ _ S _ Q R _ _ X _ A H O Dsamp ν passes η₀ w ε δ κ β n hD hν hL hη hη₀ hw0 hγ hε hδ
    hκ hn hrep hβ hpow
  classical
  have : ∀ h, IsProbabilityMeasure (reaching H Dsamp h) := isProbabilityMeasure_reaching H Dsamp
  have hne : Nonempty R := ⟨H.state 1⟩
  have hN : (0 : ℝ) < Fintype.card R := by exact_mod_cast Fintype.card_pos
  set N : ℝ := (Fintype.card R : ℝ) with hNdef
  set γ := denoiseMargin η₀ w with hγdef
  have hη0 : 0 ≤ η₀ := (eta_nonneg O).trans hη
  have hw12 : w < 1 / 2 := by
    have : w * (1 - η₀) < 1 / 2 * (1 - η₀) := by rw [hγdef, denoiseMargin] at hγ; nlinarith
    exact lt_of_mul_lt_mul_right this (by linarith)
  set m : R → ℝ := fun h => Dsamp.real {v | H.state v = h}
  set Mh : R → Prop := fun h => (reaching H Dsamp h).real {v | A.state v ∉ A.accept} < w
  set Z := returnMeasure (μ := μ) H Dsamp ν with hZ
  have : IsProbabilityMeasure Z := by rw [hZ, returnMeasure]; infer_instance
  set Pass := Finset.univ.filter (fun h => w ≤ minorityShare A H Dsamp h)
  set Heavy := Finset.univ.filter (fun h => ε / N ≤ m h ∧ minorityShare A H Dsamp h < w)
  set Wrong : R → Set (Ω × (R → ℕ → S) × X) :=
    fun h => {z | ¬ (denoisedLabel O n (z.2.1 h) z.1 ↔ Mh h)}
  have hpos : ∀ h, ε / N ≤ m h → Dsamp {v | H.state v = h} ≠ 0 := by
    intro h hh h0
    have : m h = 0 := by simp [m, measureReal_def, h0]
    have : 0 < ε / N := div_pos hε hN
    linarith
  -- Every heavy state is denoised wrongly with probability at most `δ/|R|`.
  have hwrong : ∀ h ∈ Heavy, Z.real (Wrong h) ≤ δ / N := by
    intro h hh
    obtain ⟨hheavy, hshare⟩ := (Finset.mem_filter.1 hh).2
    have h0 := hpos h hheavy
    have hmpos : 0 < m h := lt_of_lt_of_le (div_pos hε hN) hheavy
    set ρ := reaching H Dsamp h
    have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg (hκ 1)
    have hatom : ∀ a, ρ.real {a} ≤ κ * N / ε := by
      intro a
      have h1 : Dsamp.real ({v | H.state v = h} ∩ {a}) = m h * ρ.real {a} :=
        reaching_real_apply H Dsamp h0 {a}
      have h2 : Dsamp.real ({v | H.state v = h} ∩ {a}) ≤ κ :=
        (measureReal_mono Set.inter_subset_right).trans (hκ a)
      have h3 : ρ.real {a} ≤ κ / m h := by
        rw [le_div_iff₀ hmpos]; linarith
      calc ρ.real {a} ≤ κ / m h := h3
        _ ≤ κ / (ε / N) := div_le_div_of_nonneg_left hκ0 (div_pos hε hN) hheavy
        _ = κ * N / ε := by field_simp
    have hsum : ρ.real {v | A.state v ∈ A.accept} + ρ.real {v | A.state v ∉ A.accept} = 1 := by
      have := measureReal_add_measureReal_compl (μ := ρ)
        (measurableSet_of_countable {v | A.state v ∈ A.accept})
      rw [probReal_univ] at this
      exact this
    have hb := hitRate_bounds O ρ hη
    rw [hL] at hb
    have hγ' : 1 / 2 + γ = (1 - η₀) * (1 - w) := by rw [hγdef, denoiseMargin]; ring
    have hM : Mh h → 1 / 2 + γ ≤ hitRate O ρ := by
      intro hm
      change ρ.real {v | A.state v ∉ A.accept} < w at hm
      rw [hγ']
      have : (1 - η₀) * (1 - w) ≤ (1 - η₀) * ρ.real {v | A.state v ∈ A.accept} :=
        mul_le_mul_of_nonneg_left (by linarith) (by linarith)
      exact this.trans hb.1
    have hnM : ¬ Mh h → hitRate O ρ ≤ 1 / 2 - γ := by
      intro hm
      change ¬ ρ.real {v | A.state v ∉ A.accept} < w at hm
      have hacc : ρ.real {v | A.state v ∈ A.accept} < w := by
        rcases min_lt_iff.1 hshare with h' | h'
        · exact h'
        · exact absurd h' hm
      have : 1 / 2 - γ = η₀ + (1 - η₀) * w := by rw [hγdef, denoiseMargin]; ring
      rw [this]
      have : (1 - η₀) * ρ.real {v | A.state v ∈ A.accept} ≤ (1 - η₀) * w :=
        mul_le_mul_of_nonneg_left hacc.le (by linarith)
      linarith [hb.2]
    have hstream := stream_wrong_le O ρ n (Mh h) hatom hγ.le hM hnM
    have hmp : MeasurePreserving (Prod.map id (fun y : (R → ℕ → S) × X => y.1 h)) Z
        (μ.prod (Measure.infinitePi fun _ : ℕ => ρ)) :=
      (MeasurePreserving.id μ).prod
        ((measurePreserving_eval (fun h => Measure.infinitePi fun _ : ℕ => reaching H Dsamp h)
          h).comp measurePreserving_fst)
    have hW : Wrong h = Prod.map id (fun y : (R → ℕ → S) × X => y.1 h) ⁻¹'
        {p : Ω × (ℕ → S) | ¬ (denoisedLabel O n p.2 p.1 ↔ Mh h)} := rfl
    rw [hW, hmp.measureReal_preimage (measurableSet_wrong O n (Mh h)).nullMeasurableSet]
    refine hstream.trans ?_
    have hrep' : (n : ℝ) ^ 2 / 2 * (κ * N / ε) ≤ δ / (2 * N) := by
      rw [show (n : ℝ) ^ 2 / 2 * (κ * N / ε) = (κ * N ^ 2 * n ^ 2) / (2 * N * ε) by
          field_simp,
        show δ / (2 * N) = (δ * ε) / (2 * N * ε) by field_simp]
      exact div_le_div_of_nonneg_right hrep (by positivity)
    have hexp : Real.exp (-2 * n * γ ^ 2) ≤ δ / (2 * N) := by
      have hγ2 : 0 < 2 * γ ^ 2 := by positivity
      have hlog : Real.log (1 / (δ / (2 * N))) ≤ 2 * n * γ ^ 2 := by
        rw [one_div_div]
        have := (div_le_iff₀ hγ2).1 hn
        linarith
      have := exp_neg_le_of_log_le (by positivity) hlog
      rwa [show -(2 * (n : ℝ) * γ ^ 2) = -2 * n * γ ^ 2 by ring] at this
    calc (n : ℝ) ^ 2 / 2 * (κ * N / ε) + Real.exp (-2 * n * γ ^ 2)
        ≤ δ / (2 * N) + δ / (2 * N) := add_le_add hrep' hexp
      _ = δ / N := by field_simp; ring
  -- The bad event needs a failed merge check or a wrongly denoised heavy state.
  have hsub : {z : Ω × (R → ℕ → S) × X | (∀ h, z ∈ passes h)
        ∧ accuracy A H (fun h => denoisedLabel O n (z.2.1 h) z.1) Dsamp < 1 - w - ε}
      ⊆ (⋃ h ∈ Pass, passes h) ∪ ⋃ h ∈ Heavy, Wrong h := by
    rintro z ⟨hall, hacc⟩
    by_contra hz
    rw [Set.mem_union, not_or] at hz
    obtain ⟨hz1, hz2⟩ := hz
    have hshare : ∀ h, minorityShare A H Dsamp h < w := by
      intro h
      by_contra hc
      exact hz1 (Set.mem_biUnion (by simp [Pass, not_lt.1 hc]) (hall h))
    have hright : ∀ h, ε / N ≤ m h → (denoisedLabel O n (z.2.1 h) z.1 ↔ Mh h) := by
      intro h hh
      by_contra hc
      exact hz2 (Set.mem_biUnion (show h ∈ Heavy by simp [Heavy, hh, hshare h]) hc)
    have := accuracy_ge A H (fun h => denoisedLabel O n (z.2.1 h) z.1) Dsamp hw0 hε.le
      (fun h hh => contrib_ge A H Dsamp (hpos h hh) (hshare h) _ (hright h hh))
    linarith
  calc Z.real {z | (∀ h, z ∈ passes h)
        ∧ accuracy A H (fun h => denoisedLabel O n (z.2.1 h) z.1) Dsamp < 1 - w - ε}
      ≤ Z.real (⋃ h ∈ Pass, passes h) + Z.real (⋃ h ∈ Heavy, Wrong h) :=
        (measureReal_mono hsub (measure_ne_top _ _)).trans (measureReal_union_le _ _)
    _ ≤ ∑ h ∈ Pass, Z.real (passes h) + ∑ h ∈ Heavy, Z.real (Wrong h) :=
        add_le_add (measureReal_biUnion_finset_le _ _) (measureReal_biUnion_finset_le _ _)
    _ ≤ ∑ _h ∈ Pass, β + ∑ _h ∈ Heavy, δ / N :=
        add_le_add (Finset.sum_le_sum fun h hh => hpow h (Finset.mem_filter.1 hh).2)
          (Finset.sum_le_sum hwrong)
    _ ≤ N * β + N * (δ / N) := by
        simp only [Finset.sum_const, nsmul_eq_mul]
        have h1 : (Pass.card : ℝ) ≤ N := by rw [hNdef]; exact_mod_cast Finset.card_le_univ Pass
        have h2 : (Heavy.card : ℝ) ≤ N := by rw [hNdef]; exact_mod_cast Finset.card_le_univ Heavy
        exact add_le_add (mul_le_mul_of_nonneg_right h1 hβ)
          (mul_le_mul_of_nonneg_right h2 (div_nonneg hδ.le hN.le))
    _ = δ + N * β := by field_simp; ring

end OrthoDFA
