import OrthoDFA.Proofs.ClusteringQuality
import OrthoDFA.Proofs.Hits

/-!
# What the learner returns is worth: the proof

Each round is bounded one cell at a time: fixing the earlier rounds' check draws and the noise
at the strings the stage read fixes the hypothesis, and leaves the rest of the noise independent
of the cell (`pinned_inter_reads`).

A state's reads share one persistent noise draw, so they are independent only at distinct
strings off the cell's.  There the reads have the law of the fresh-noise model, where every read
draws its own string and its own noise (`coupling_le`), and are i.i.d. with mean at least
`½ + γ` on the majority side.  Repeats, and draws the stage already read, are paid for by the
atom bound.
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

/-! ## Pinned noise -/

omit [IsProbabilityMeasure μ] in
lemma measurableSet_pinned (O : Oracle μ S) (U : Finset S) (b : S → ℝ) :
    MeasurableSet (pinned O U b) := by
  have : pinned O U b = ⋂ s ∈ U, {ω | O.noise s ω = b s} := by
    ext ω; simp [pinned]
  rw [this]
  exact Finset.measurableSet_biInter U fun s _ =>
    measurableSet_eq_fun (O.noise_meas s) measurable_const

/-- Reads at distinct strings off `U` are independent of the noise at `U`, and read as fresh
draws would. -/
lemma pinned_inter_reads (O : Oracle μ S) (U : Finset S) (b : S → ℝ) {n : ℕ} {s : Fin n → S}
    (hs : Function.Injective s) (hU : ∀ i, s i ∉ U) {T : Set (Fin n → ℝ)}
    (hT : MeasurableSet T) :
    μ (pinned O U b ∩ {ω | (fun i => O.mq (s i) ω) ∈ T})
      = μ (pinned O U b)
        * (Measure.pi fun _ : Fin n => μ) {ω' | (fun i => O.mq (s i) (ω' i)) ∈ T} := by
  classical
  set V := Finset.univ.image s
  have hmem : ∀ i, s i ∈ V := fun i => Finset.mem_image_of_mem s (Finset.mem_univ i)
  have hdisj : Disjoint U V := by
    rw [Finset.disjoint_left]
    intro a ha haV
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.1 haV
    exact hU i ha
  have hind := O.noise_indep.indepFun_finset U V hdisj O.noise_meas
  have hA : MeasurableSet {g : U → ℝ | ∀ v : U, g v = b v} := by
    rw [Set.ofPred_forall]
    exact .iInter fun v => measurableSet_eq_fun (measurable_pi_apply v) measurable_const
  set F : (V → ℝ) → (Fin n → ℝ) :=
    fun g i => O.label (s i) + (1 - 2 * O.label (s i)) * g ⟨s i, hmem i⟩
  have hF : Measurable F := measurable_pi_lambda _ fun i =>
    measurable_const.add (measurable_const.mul (measurable_pi_apply _))
  have h1 : pinned O U b = (fun ω (v : U) => O.noise v ω) ⁻¹' {g | ∀ v : U, g v = b v} := by
    ext ω; simp [pinned]
  have h2 : {ω | (fun i => O.mq (s i) ω) ∈ T} = (fun ω (v : V) => O.noise v ω) ⁻¹' (F ⁻¹' T) :=
    rfl
  rw [h1, h2, hind.measure_inter_preimage_eq_mul _ _ hA (hF hT), ← h2, noise_law_eq O hs hT]

/-- The coupling on a pinned cell: its reads land in `T` no more often than fresh draws would,
once draws that repeat or land in `U` are paid for. -/
lemma coupling_le (O : Oracle μ S) (U : Finset S) (b : S → ℝ) (ρ : Measure S)
    [IsProbabilityMeasure ρ] (n : ℕ) {T : Set (Fin n → ℝ)} (hT : MeasurableSet T) :
    ((μ.restrict (pinned O U b)).prod (Measure.pi fun _ : Fin n => ρ)) (reads O ⁻¹' T)
      ≤ μ (pinned O U b)
        * ((Measure.pi fun _ : Fin n => ρ) {s | ¬ Function.Injective s ∨ ∃ i, s i ∈ U}
          + ((Measure.pi fun _ : Fin n => ρ).prod (Measure.pi fun _ : Fin n => μ))
            (freshReads O ⁻¹' T)) := by
  set P := pinned O U b
  have hP := measurableSet_pinned O U b
  set B := {s : Fin n → S | ¬ Function.Injective s ∨ ∃ i, s i ∈ U}
  have hB : MeasurableSet B := (Set.to_countable _).measurableSet
  rw [Measure.prod_apply_symm (measurable_reads O hT),
    Measure.prod_apply (measurable_freshReads O hT), ← lintegral_indicator_one hB,
    ← lintegral_add_left (measurable_one.indicator hB),
    ← lintegral_const_mul' _ _ (measure_ne_top μ P)]
  refine lintegral_mono fun s => ?_
  by_cases hs : Function.Injective s ∧ ∀ i, s i ∉ U
  · have hs' : s ∉ B := by
      rintro (h | ⟨i, hi⟩)
      · exact h hs.1
      · exact hs.2 i hi
    rw [Set.indicator_of_notMem hs', zero_add, Measure.restrict_apply' hP, Set.inter_comm]
    exact (pinned_inter_reads O U b hs.1 hs.2 hT).le
  · have hs' : s ∈ B := by
      rcases not_and_or.1 hs with h | h
      · exact Or.inl h
      · exact Or.inr ((not_forall.1 h).imp fun i hi => not_not.1 hi)
    rw [Set.indicator_of_mem hs', Pi.one_apply]
    calc (μ.restrict P) ((fun ω => (ω, s)) ⁻¹' (reads O ⁻¹' T)) ≤ μ.restrict P Set.univ :=
          measure_mono (Set.subset_univ _)
      _ = μ P * 1 := by rw [Measure.restrict_apply_univ, mul_one]
      _ ≤ _ := by gcongr; exact le_self_add

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

variable {R : Type*} [Fintype R]

omit [Fintype R] in
/-- On a pinned cell, the first `n` draws reaching `h` are denoised against `M` with
probability at most the repeats, the draws landing in `U`, and one Hoeffding tail, when their
reads lean towards `M` by `γ`. -/
theorem pinned_wrong_le (O : Oracle μ S) (U : Finset S) (b : S → ℝ) (H : DFA S R)
    (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] {h : R}
    (hp : Dsamp {v | H.state v = h} ≠ 0) (n : ℕ) {c γ : ℝ} (M : Prop)
    (hc : ∀ a, (reaching H Dsamp h).real {a} ≤ c) (hγ : 0 ≤ γ)
    (hM : M → 1 / 2 + γ ≤ hitRate O (reaching H Dsamp h))
    (hnM : ¬ M → hitRate O (reaching H Dsamp h) ≤ 1 / 2 - γ) :
    ((μ.restrict (pinned O U b)).prod (Measure.infinitePi fun _ : ℕ => Dsamp))
        {p | ¬ (denoisedLabel O n (hitsOf H h p.2) p.1 ↔ M)}
      ≤ μ (pinned O U b) * ENNReal.ofReal
          ((n : ℝ) ^ 2 / 2 * c + n * U.card * c + Real.exp (-2 * n * γ ^ 2)) := by
  classical
  have := isProbabilityMeasure_reaching H Dsamp h
  set ρ := reaching H Dsamp h
  set π := Measure.pi fun _ : Fin n => ρ
  have hc0 : 0 ≤ c := le_trans measureReal_nonneg (hc 1)
  have hT := measurableSet_wrongReads n M
  have hmp := (MeasurePreserving.id (μ.restrict (pinned O U b))).prod
    (measurePreserving_hitsOf H h hp n)
  have hset : {p : Ω × (ℕ → S) | ¬ (denoisedLabel O n (hitsOf H h p.2) p.1 ↔ M)}
      = Prod.map id (fun (u : ℕ → S) (i : Fin n) => hitsOf H h u i) ⁻¹'
        (reads O ⁻¹' wrongReads n M) := by
    ext p
    simp only [Set.mem_ofPred_eq, Set.mem_preimage, wrongReads, denoisedLabel_iff]
    rfl
  rw [hset, hmp.measure_preimage (measurable_reads O hT).nullMeasurableSet]
  refine (coupling_le O U b ρ n hT).trans ?_
  gcongr
  have hrep : π {s | ¬ Function.Injective s ∨ ∃ i, s i ∈ U}
      ≤ ENNReal.ofReal ((n : ℝ) ^ 2 / 2 * c + n * U.card * c) := by
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal ?_
    have hU : ∀ i : Fin n, π.real {s : Fin n → S | s i ∈ U} ≤ U.card * c := by
      intro i
      have hev := measurePreserving_eval (fun _ : Fin n => ρ) i
      rw [show {s : Fin n → S | s i ∈ U} = Function.eval i ⁻¹' (U : Set S) from rfl,
        hev.measureReal_preimage (measurableSet_of_countable _).nullMeasurableSet]
      calc ρ.real (U : Set S) ≤ ρ.real (⋃ v ∈ U, {v}) :=
            measureReal_mono (fun v hv => by simpa using hv) (measure_ne_top _ _)
        _ ≤ ∑ v ∈ U, ρ.real {v} := measureReal_biUnion_finset_le _ _
        _ ≤ ∑ _v ∈ U, c := Finset.sum_le_sum fun v _ => hc v
        _ = U.card * c := by rw [Finset.sum_const, nsmul_eq_mul]
    calc π.real {s | ¬ Function.Injective s ∨ ∃ i, s i ∈ U}
        ≤ π.real {s : Fin n → S | ¬ Function.Injective s}
          + π.real (⋃ i ∈ (Finset.univ : Finset (Fin n)), {s : Fin n → S | s i ∈ U}) := by
          refine (measureReal_mono ?_).trans (measureReal_union_le _ _)
          rintro s (hs | ⟨i, hi⟩)
          · exact Or.inl hs
          · exact Or.inr (Set.mem_biUnion (Finset.mem_univ i) hi)
      _ ≤ (n : ℝ) ^ 2 / 2 * c + ∑ i : Fin n, π.real {s : Fin n → S | s i ∈ U} :=
          add_le_add (pi_not_injective_le_half ρ n hc) (measureReal_biUnion_finset_le _ _)
      _ ≤ (n : ℝ) ^ 2 / 2 * c + ∑ _i : Fin n, (U.card * c : ℝ) :=
          add_le_add le_rfl (Finset.sum_le_sum fun i _ => hU i)
      _ = _ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hfresh : (π.prod (Measure.pi fun _ : Fin n => μ)) (freshReads O ⁻¹' wrongReads n M)
      ≤ ENNReal.ofReal (Real.exp (-2 * n * γ ^ 2)) := by
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal ?_
    by_cases hm : M
    · rw [wrongReads_eq, if_pos hm]; exact fresh_lower_tail O ρ n hγ (hM hm)
    · rw [wrongReads_eq, if_neg hm]; exact fresh_upper_tail O ρ n hγ (hnM hm)
  calc _ ≤ ENNReal.ofReal ((n : ℝ) ^ 2 / 2 * c + n * U.card * c)
        + ENNReal.ofReal (Real.exp (-2 * n * γ ^ 2)) := add_le_add hrep hfresh
    _ = _ := (ENNReal.ofReal_add (by positivity) (Real.exp_pos _).le).symm

omit [IsProbabilityMeasure μ] in
lemma prod_restrict_apply [SFinite μ] {Y : Type*} [MeasurableSpace Y] (ξ : Measure Y) [SFinite ξ]
    {P : Set Ω} (hP : MeasurableSet P) (E : Set (Ω × Y)) :
    (μ.prod ξ) (E ∩ P ×ˢ Set.univ) = ((μ.restrict P).prod ξ) E := by
  rw [← Measure.restrict_apply' (hP.prod MeasurableSet.univ), ← Measure.prod_restrict,
    Measure.restrict_univ]

/-! ## One round on one pinned cell -/

/-- With the hypothesis and the check fixed and the noise pinned at `U`, the check passes a
state it should not, or denoising gets a heavy state wrong, on at most `δ + |R|β` of the cell. -/
theorem cell_le {Q X : Type*} [MeasurableSpace X] (A : DFA S Q) (H : DFA S R)
    (O : Oracle μ S) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (ν : Measure X)
    [IsProbabilityMeasure ν] (pass : R → Set (Ω × X)) (U : Finset S) (b : S → ℝ)
    {η₀ w ε δ κ β : ℝ} {n T : ℕ}
    (hL : O.L = {v | A.state v ∈ A.accept}) (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (hw0 : 0 ≤ w)
    (hγ : 0 < denoiseMargin η₀ w) (hε : 0 < ε) (hδ : 0 < δ) (hκ : ∀ a, Dsamp.real {a} ≤ κ)
    (hn : Real.log (2 * Fintype.card R / δ) / (2 * denoiseMargin η₀ w ^ 2) ≤ n)
    (hrep : κ * (Fintype.card R : ℝ) ^ 2 * n * (n + 2 * T) ≤ δ * ε) (hβ : 0 ≤ β)
    (hU : U.card ≤ T)
    (hpow : ∀ h, ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} →
      w ≤ minorityShare A H Dsamp h → ((μ[|pinned O U b]).prod ν).real (pass h) ≤ β) :
    ((μ.prod ν).prod (Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp))
        ({q | (∀ h, q.1 ∈ pass h)
          ∧ accuracy A H (fun h => denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1) Dsamp
            < 1 - w - ε} ∩ {q | q.1.1 ∈ pinned O U b})
      ≤ μ (pinned O U b) * ENNReal.ofReal (δ + Fintype.card R * β) := by
  classical
  have : ∀ h, IsProbabilityMeasure (reaching H Dsamp h) := isProbabilityMeasure_reaching H Dsamp
  have hne : Nonempty R := ⟨H.state 1⟩
  have hN : (0 : ℝ) < Fintype.card R := by exact_mod_cast Fintype.card_pos
  set N : ℝ := (Fintype.card R : ℝ) with hNdef
  set γ := denoiseMargin η₀ w with hγdef
  set P := pinned O U b
  have hP := measurableSet_pinned O U b
  have hη0 : 0 ≤ η₀ := (eta_nonneg O).trans hη
  set m : R → ℝ := fun h => Dsamp.real {v | H.state v = h}
  set Mh : R → Prop := fun h => (reaching H Dsamp h).real {v | A.state v ∉ A.accept} < w
  set ξ := Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp
  set Z := (μ.prod ν).prod ξ
  set Pass := Finset.univ.filter (fun h => ε / N ≤ m h ∧ w ≤ minorityShare A H Dsamp h)
  set Heavy := Finset.univ.filter (fun h => ε / N ≤ m h ∧ minorityShare A H Dsamp h < w)
  set PassE : R → Set ((Ω × X) × (R → ℕ → S)) := fun h => {q | q.1 ∈ pass h ∧ q.1.1 ∈ P}
  set Wrong : R → Set ((Ω × X) × (R → ℕ → S)) :=
    fun h => {q | ¬ (denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1 ↔ Mh h) ∧ q.1.1 ∈ P}
  have hpos : ∀ h, ε / N ≤ m h → Dsamp {v | H.state v = h} ≠ 0 := by
    intro h hh h0
    have : m h = 0 := by simp [m, measureReal_def, h0]
    have : 0 < ε / N := div_pos hε hN
    linarith
  have hreal : ∀ {E : Set ((Ω × X) × (R → ℕ → S))} {x : ℝ}, 0 ≤ x →
      Z E ≤ μ P * ENNReal.ofReal x → Z.real E ≤ μ.real P * x := by
    intro E x hx hle
    rw [measureReal_def, measureReal_def, ← ENNReal.toReal_ofReal hx, ← ENNReal.toReal_mul]
    exact ENNReal.toReal_mono (by finiteness) hle
  -- A heavy state the check should reject passes on at most `β` of the cell.
  have hpass : ∀ h ∈ Pass, Z.real (PassE h) ≤ μ.real P * β := by
    intro h hh
    obtain ⟨hheavy, hshare⟩ := (Finset.mem_filter.1 hh).2
    refine hreal hβ ?_
    have hPE : PassE h = (pass h ∩ P ×ˢ Set.univ) ×ˢ Set.univ := by
      ext q
      simp only [PassE, Set.mem_ofPred_eq, Set.mem_prod, Set.mem_inter_iff, Set.mem_univ,
        and_true]
    rw [hPE, Measure.prod_prod, measure_univ, mul_one, prod_restrict_apply ν hP]
    by_cases hP0 : μ P = 0
    · rw [Measure.restrict_eq_zero.2 hP0, Measure.zero_prod, Measure.coe_zero, Pi.zero_apply]
      exact zero_le
    · have hsm : μ.restrict P = μ P • μ[|P] := by
        rw [ProbabilityTheory.cond, smul_smul, ENNReal.mul_inv_cancel hP0 (measure_ne_top _ _),
          one_smul]
      have := cond_isProbabilityMeasure (μ := μ) hP0
      rw [hsm, Measure.prod_smul_left, Measure.smul_apply, smul_eq_mul]
      gcongr
      rw [ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _) hβ]
      exact hpow h hheavy hshare
  -- A heavy state is denoised wrongly on at most `δ/|R|` of the cell.
  have hwrong : ∀ h ∈ Heavy, Z.real (Wrong h) ≤ μ.real P * (δ / N) := by
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
    have hb := hitRate_bounds O ρ hη
    rw [hL] at hb
    have hγ' : 1 / 2 + γ = (1 - η₀) * (1 - w) := by rw [hγdef, denoiseMargin]; ring
    have hM : Mh h → 1 / 2 + γ ≤ hitRate O ρ := by
      intro hm
      change ρ.real {v | A.state v ∉ A.accept} < w at hm
      have hsum : ρ.real {v | A.state v ∈ A.accept} + ρ.real {v | A.state v ∉ A.accept} = 1 := by
        have := measureReal_add_measureReal_compl (μ := ρ)
          (measurableSet_of_countable {v | A.state v ∈ A.accept})
        rw [probReal_univ] at this
        exact this
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
    have hstream := pinned_wrong_le O U b H Dsamp h0 n (Mh h) hatom hγ.le hM hnM
    have hmp : MeasurePreserving (Prod.map Prod.fst (fun v : R → ℕ → S => v h)) Z
        (μ.prod (Measure.infinitePi fun _ : ℕ => Dsamp)) :=
      measurePreserving_fst.prod
        (measurePreserving_eval (fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp) h)
    have hW : Wrong h = Prod.map Prod.fst (fun v : R → ℕ → S => v h) ⁻¹'
        ({p : Ω × (ℕ → S) | ¬ (denoisedLabel O n (hitsOf H h p.2) p.1 ↔ Mh h)}
          ∩ P ×ˢ Set.univ) := by
      ext q
      simp only [Wrong, Set.mem_ofPred_eq, Set.mem_preimage, Set.mem_inter_iff, Set.mem_prod,
        Set.mem_univ, and_true]
      exact Iff.rfl
    have hnum : (n : ℝ) ^ 2 / 2 * (κ * N / ε) + n * U.card * (κ * N / ε)
        + Real.exp (-2 * n * γ ^ 2) ≤ δ / N := by
      have hc0 : 0 ≤ κ * N / ε := by positivity
      have hUT : (U.card : ℝ) ≤ T := by exact_mod_cast hU
      have hrep' : (n : ℝ) ^ 2 / 2 * (κ * N / ε) + n * U.card * (κ * N / ε) ≤ δ / (2 * N) := by
        calc (n : ℝ) ^ 2 / 2 * (κ * N / ε) + n * U.card * (κ * N / ε)
            ≤ (n : ℝ) ^ 2 / 2 * (κ * N / ε) + n * T * (κ * N / ε) := by gcongr
          _ = (κ * N ^ 2 * n * (n + 2 * T)) / (2 * N * ε) := by field_simp
          _ ≤ (δ * ε) / (2 * N * ε) := div_le_div_of_nonneg_right hrep (by positivity)
          _ = δ / (2 * N) := by field_simp
      have hexp : Real.exp (-2 * n * γ ^ 2) ≤ δ / (2 * N) := by
        have hγ2 : 0 < 2 * γ ^ 2 := by positivity
        have hlog : Real.log (1 / (δ / (2 * N))) ≤ 2 * n * γ ^ 2 := by
          rw [one_div_div]
          have := (div_le_iff₀ hγ2).1 hn
          linarith
        have := exp_neg_le_of_log_le (by positivity) hlog
        rwa [show -(2 * (n : ℝ) * γ ^ 2) = -2 * n * γ ^ 2 by ring] at this
      calc _ ≤ δ / (2 * N) + δ / (2 * N) := add_le_add hrep' hexp
        _ = δ / N := by field_simp; ring
    refine hreal (by positivity) ?_
    rw [hW]
    refine (Measure.le_map_apply hmp.measurable.aemeasurable _).trans ?_
    rw [hmp.map_eq, prod_restrict_apply _ hP]
    refine hstream.trans ?_
    gcongr
  -- The bad event needs a check passing a state it should reject, or a heavy state denoised
  -- wrongly.
  have hsub : {q : (Ω × X) × (R → ℕ → S) | (∀ h, q.1 ∈ pass h)
        ∧ accuracy A H (fun h => denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1) Dsamp
          < 1 - w - ε} ∩ {q | q.1.1 ∈ P}
      ⊆ (⋃ h ∈ Pass, PassE h) ∪ ⋃ h ∈ Heavy, Wrong h := by
    rintro q ⟨⟨hall, hacc⟩, hq⟩
    by_contra hz
    rw [Set.mem_union, not_or] at hz
    obtain ⟨hz1, hz2⟩ := hz
    have hshare : ∀ h, ε / N ≤ m h → minorityShare A H Dsamp h < w := by
      intro h hh
      by_contra hc
      exact hz1 (Set.mem_biUnion (by simp [Pass, hh, not_lt.1 hc]) ⟨hall h, hq⟩)
    have hright : ∀ h, ε / N ≤ m h →
        (denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1 ↔ Mh h) := by
      intro h hh
      by_contra hc
      exact hz2 (Set.mem_biUnion (show h ∈ Heavy by simp [Heavy, hh, hshare h hh]) ⟨hc, hq⟩)
    have := accuracy_ge A H (fun h => denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1) Dsamp hw0
      hε.le (fun h hh => contrib_ge A H Dsamp (hpos h hh) (hshare h hh) _ (hright h hh))
    linarith
  have htot : Z.real ({q : (Ω × X) × (R → ℕ → S) | (∀ h, q.1 ∈ pass h)
        ∧ accuracy A H (fun h => denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1) Dsamp
          < 1 - w - ε} ∩ {q | q.1.1 ∈ P})
      ≤ μ.real P * (δ + N * β) := by
    have hPr : 0 ≤ μ.real P := measureReal_nonneg
    calc _ ≤ Z.real (⋃ h ∈ Pass, PassE h) + Z.real (⋃ h ∈ Heavy, Wrong h) :=
          (measureReal_mono hsub (measure_ne_top _ _)).trans (measureReal_union_le _ _)
      _ ≤ ∑ h ∈ Pass, Z.real (PassE h) + ∑ h ∈ Heavy, Z.real (Wrong h) :=
          add_le_add (measureReal_biUnion_finset_le _ _) (measureReal_biUnion_finset_le _ _)
      _ ≤ ∑ _h ∈ Pass, μ.real P * β + ∑ _h ∈ Heavy, μ.real P * (δ / N) :=
          add_le_add (Finset.sum_le_sum hpass) (Finset.sum_le_sum hwrong)
      _ ≤ N * (μ.real P * β) + N * (μ.real P * (δ / N)) := by
          simp only [Finset.sum_const, nsmul_eq_mul]
          have h1 : (Pass.card : ℝ) ≤ N := by
            rw [hNdef]; exact_mod_cast Finset.card_le_univ Pass
          have h2 : (Heavy.card : ℝ) ≤ N := by
            rw [hNdef]; exact_mod_cast Finset.card_le_univ Heavy
          exact add_le_add (mul_le_mul_of_nonneg_right h1 (by positivity))
            (mul_le_mul_of_nonneg_right h2 (by positivity))
      _ = μ.real P * (δ + N * β) := by field_simp; ring
  calc _ = ENNReal.ofReal (Z.real ({q : (Ω × X) × (R → ℕ → S) | (∀ h, q.1 ∈ pass h)
        ∧ accuracy A H (fun h => denoisedLabel O n (hitsOf H h (q.2 h)) q.1.1) Dsamp
          < 1 - w - ε} ∩ {q | q.1.1 ∈ P})) := (ofReal_measureReal).symm
    _ ≤ ENNReal.ofReal (μ.real P * (δ + N * β)) := ENNReal.ofReal_le_ofReal htot
    _ = μ P * ENNReal.ofReal (δ + N * β) := by
        rw [ENNReal.ofReal_mul measureReal_nonneg, ofReal_measureReal]

/-! ## The loop -/

/-- Round `r` of the loop, given what one round does on one pinned cell.  It conditions on the
earlier rounds' check draws and on the noise at the strings the stage read. -/
theorem loop_round_le {X Y C : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] [MeasurableSpace Y] (O : Oracle μ S) (ν : Measure X)
    [IsProbabilityMeasure ν] (ξ : Measure Y) [IsProbabilityMeasure ξ] {K T : ℕ} (r : Fin K)
    (stage : Ω → (Fin K → X) → C) (queried : Ω → (Fin K → X) → Finset S)
    (hdep : ∀ ω x x', (∀ i < r, x i = x' i) → stage ω x = stage ω x' ∧ queried ω x = queried ω x')
    (hreads : ∀ x, ReadsOnly O (fun ω => queried ω x) (fun ω => stage ω x))
    (hT : ∀ ω x, (queried ω x).card ≤ T) (F : C → Set ((Ω × X) × Y)) (B : ℝ≥0∞)
    (hF : ∀ c U b, U.card ≤ T →
      ((μ.prod ν).prod ξ) (F c ∩ {q | q.1.1 ∈ pinned O U b}) ≤ μ (pinned O U b) * B) :
    (μ.prod ((Measure.pi fun _ : Fin K => ν).prod (Measure.pi fun _ : Fin K => ξ)))
      {z | ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage z.1 z.2.1)} ≤ B := by
  classical
  obtain ⟨x0⟩ := nonempty_of_isProbabilityMeasure ν
  set I := {i : Fin K // i < r}
  set π := Measure.pi fun _ : I => ν
  set L := μ.prod ((Measure.pi fun _ : Fin K => ν).prod (Measure.pi fun _ : Fin K => ξ))
  set ext : (I → X) → Fin K → X := fun a i => if hi : i < r then a ⟨i, hi⟩ else x0
  set bits : Finset S → S → ℝ := fun t s => if s ∈ t then 1 else 0
  set cell : (I → X) → Finset S → Finset S → Set Ω := fun a U t =>
    {ω | queried ω (ext a) = U ∧ (∀ s ∈ U, O.noise s ω = 0 ∨ O.noise s ω = 1)
      ∧ U.filter (fun s => O.noise s ω = 1) = t}
  -- A nonempty cell is a pinned event, on which the stage is constant.
  have hcell : ∀ a U t ω0, ω0 ∈ cell a U t →
      cell a U t = pinned O U (bits t)
        ∧ ∀ ω ∈ cell a U t, stage ω (ext a) = stage ω0 (ext a) := by
    intro a U t ω0 h0
    have hbits : ∀ ω ∈ cell a U t, ∀ s ∈ U, O.noise s ω = bits t s := by
      rintro ω ⟨_, hb, ht⟩ s hs
      simp only [bits]
      rcases hb s hs with h | h
      · rw [if_neg, h]
        intro hst
        rw [← ht] at hst
        have := (Finset.mem_filter.1 hst).2
        rw [h] at this
        norm_num at this
      · rw [if_pos, h]
        rw [← ht]
        exact Finset.mem_filter.2 ⟨hs, h⟩
    have hsame : ∀ ω, (∀ s ∈ U, O.noise s ω = bits t s) →
        queried ω (ext a) = U ∧ stage ω (ext a) = stage ω0 (ext a) := by
      intro ω hω
      have := hreads (ext a) ω0 ω (fun s hs => by
        have hs' : s ∈ U := by rw [← h0.1]; exact hs
        rw [hbits ω0 h0 s hs', hω s hs'])
      exact ⟨this.1.trans h0.1, this.2⟩
    have htU : t ⊆ U := by rw [← h0.2.2]; exact Finset.filter_subset _ _
    refine ⟨Set.ext fun ω => ⟨fun hω => hbits ω hω, fun hω => ?_⟩,
      fun ω hω => (hsame ω (hbits ω hω)).2⟩
    refine ⟨(hsame ω hω).1, fun s hs => ?_, ?_⟩
    · rw [hω s hs]
      simp only [bits]
      split_ifs <;> simp
    · ext s
      simp only [Finset.mem_filter]
      constructor
      · rintro ⟨hs, h1⟩
        rw [hω s hs] at h1
        simp only [bits] at h1
        split_ifs at h1 with hst
        · exact hst
        · norm_num at h1
      · intro hst
        exact ⟨htU hst, by rw [hω s (htU hst)]; simp [bits, hst]⟩
  have hcellm : ∀ a U t, MeasurableSet (cell a U t) := by
    intro a U t
    by_cases hne : (cell a U t).Nonempty
    · obtain ⟨ω0, h0⟩ := hne
      rw [(hcell a U t ω0 h0).1]
      exact measurableSet_pinned O U _
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]
      exact MeasurableSet.empty
  have hcelld : ∀ a, Pairwise (Function.onFun Disjoint
      fun Ut : Finset S × Finset S => cell a Ut.1 Ut.2) := by
    rintro a ⟨U, t⟩ ⟨U', t'⟩ hne
    rw [Function.onFun, Set.disjoint_left]
    rintro ω ⟨hU, _, ht⟩ ⟨hU', _, ht'⟩
    have hUU : U = U' := hU.symm.trans hU'
    subst hUU
    exact hne (Prod.ext rfl (ht.symm.trans ht'))
  -- The check draws of the earlier rounds, then the noise, the round's check draw and streams.
  set Φ : Ω × (Fin K → X) × (Fin K → Y) → (I → X) × ((Ω × X) × Y) :=
    fun z => (fun i : I => z.2.1 i, ((z.1, z.2.1 r), z.2.2 r))
  have hΦ : MeasurePreserving Φ L (π.prod ((μ.prod ν).prod ξ)) := by
    have hsplit : MeasurePreserving (fun x : Fin K → X => ((fun i : I => x i), x r))
        (Measure.pi fun _ : Fin K => ν) (π.prod ν) :=
      ((MeasurePreserving.id π).prod
        (measurePreserving_eval (fun _ : {i : Fin K // ¬ i < r} => ν) ⟨r, lt_irrefl r⟩)).comp
        (measurePreserving_piEquivPiSubtypeProd (fun _ : Fin K => ν) (fun i : Fin K => i < r))
    have hstep := (MeasurePreserving.id μ).prod
      (hsplit.prod (measurePreserving_eval (fun _ : Fin K => ξ) r))
    have e1 := (MeasurePreserving.id μ).prod (measurePreserving_prodAssoc π ν ξ)
    have e2 := MeasurePreserving.symm _ (measurePreserving_prodAssoc μ π (ν.prod ξ))
    have e3 := (Measure.measurePreserving_swap (μ := μ) (ν := π)).prod
      (MeasurePreserving.id (ν.prod ξ))
    have e4 := measurePreserving_prodAssoc π μ (ν.prod ξ)
    have e5 := (MeasurePreserving.id π).prod
      (MeasurePreserving.symm _ (measurePreserving_prodAssoc μ ν ξ))
    exact e5.comp (e4.comp (e3.comp (e2.comp (e1.comp hstep))))
  set piece : (I → X) → Finset S × Finset S → Set (Ω × (Fin K → X) × (Fin K → Y)) :=
    fun a Ut => {z | (fun i : I => z.2.1 i) = a ∧ z.1 ∈ cell a Ut.1 Ut.2
      ∧ ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage z.1 z.2.1)}
  have hpiece : ∀ a Ut, L (piece a Ut) ≤ π {a} * (μ (cell a Ut.1 Ut.2) * B) := by
    rintro a ⟨U, t⟩
    by_cases hne : (cell a U t).Nonempty
    · obtain ⟨ω0, h0⟩ := hne
      obtain ⟨hceq, hst⟩ := hcell a U t ω0 h0
      have hU : U.card ≤ T := by rw [← h0.1]; exact hT ω0 (ext a)
      have hsub : piece a (U, t)
          ⊆ Φ ⁻¹' ({a} ×ˢ (F (stage ω0 (ext a)) ∩ {q | q.1.1 ∈ pinned O U (bits t)})) := by
        rintro z ⟨ha, hz, hFz⟩
        have hx : ∀ i < r, z.2.1 i = ext a i := fun i hi => by
          simp only [ext, dif_pos hi, ← ha]
        have hst' : stage z.1 z.2.1 = stage ω0 (ext a) :=
          (hdep z.1 z.2.1 (ext a) hx).1.trans (hst z.1 hz)
        refine ⟨ha, ?_, ?_⟩
        · rw [← hst']; exact hFz
        · change z.1 ∈ pinned O U (bits t)
          rw [← hceq]; exact hz
      calc L (piece a (U, t))
          ≤ L (Φ ⁻¹' ({a} ×ˢ (F (stage ω0 (ext a)) ∩ {q | q.1.1 ∈ pinned O U (bits t)}))) :=
            measure_mono hsub
        _ ≤ (L.map Φ) ({a} ×ˢ (F (stage ω0 (ext a)) ∩ {q | q.1.1 ∈ pinned O U (bits t)})) :=
            Measure.le_map_apply hΦ.measurable.aemeasurable _
        _ = π {a} * ((μ.prod ν).prod ξ)
              (F (stage ω0 (ext a)) ∩ {q | q.1.1 ∈ pinned O U (bits t)}) := by
            rw [hΦ.map_eq, Measure.prod_prod]
        _ ≤ π {a} * (μ (pinned O U (bits t)) * B) := by
            gcongr
            exact hF _ U (bits t) hU
        _ = π {a} * (μ (cell a U t) * B) := by rw [← hceq]
    · have : piece a (U, t) = ∅ := by
        ext z
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨_, hz, _⟩
        exact hne ⟨z.1, hz⟩
      rw [this, measure_empty]
      exact zero_le
  -- Almost every noise draw is `0/1` everywhere, so the cells cover.
  set N0 := {ω : Ω | ∃ s, ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)}
  have hN0 : μ N0 = 0 := by
    have : N0 = ⋃ s, {ω | ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)} := by
      ext ω; simp only [N0, Set.mem_ofPred_eq, Set.mem_iUnion]
    rw [this]
    exact measure_iUnion_null fun s => ae_iff.1 (O.noise_bit s)
  have hcover : {z : Ω × (Fin K → X) × (Fin K → Y) |
        ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage z.1 z.2.1)}
      ⊆ N0 ×ˢ Set.univ ∪ ⋃ a, ⋃ Ut, piece a Ut := by
    intro z hz
    by_cases h0 : z.1 ∈ N0
    · exact Or.inl ⟨h0, trivial⟩
    · right
      have h0' : ∀ s, O.noise s z.1 = 0 ∨ O.noise s z.1 = 1 := fun s => by
        by_contra hc; exact h0 ⟨s, hc⟩
      set a := fun i : I => z.2.1 i
      refine Set.mem_iUnion.2 ⟨a, Set.mem_iUnion.2 ⟨(queried z.1 (ext a),
        (queried z.1 (ext a)).filter fun s => O.noise s z.1 = 1), rfl,
        ⟨rfl, fun s _ => h0' s, rfl⟩, hz⟩⟩
  calc L {z | ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage z.1 z.2.1)}
      ≤ L (N0 ×ˢ Set.univ ∪ ⋃ a, ⋃ Ut, piece a Ut) := measure_mono hcover
    _ ≤ L (N0 ×ˢ Set.univ) + L (⋃ a, ⋃ Ut, piece a Ut) := measure_union_le _ _
    _ ≤ 0 + ∑' a, ∑' Ut, L (piece a Ut) := by
        gcongr
        · rw [Measure.prod_prod, hN0, zero_mul]
        · exact (measure_iUnion_le _).trans (ENNReal.tsum_le_tsum fun a => measure_iUnion_le _)
    _ ≤ ∑' a, ∑' Ut : Finset S × Finset S, π {a} * (μ (cell a Ut.1 Ut.2) * B) := by
        rw [zero_add]
        exact ENNReal.tsum_le_tsum fun a => ENNReal.tsum_le_tsum fun Ut => hpiece a Ut
    _ = ∑' a, π {a} * ((∑' Ut : Finset S × Finset S, μ (cell a Ut.1 Ut.2)) * B) := by
        refine tsum_congr fun a => ?_
        rw [ENNReal.tsum_mul_left, ENNReal.tsum_mul_right]
    _ ≤ ∑' a, π {a} * (1 * B) := by
        gcongr with a
        rw [← measure_iUnion (hcelld a) fun Ut => hcellm a Ut.1 Ut.2]
        exact prob_le_one
    _ = (∑' a, π {a}) * B := by rw [← ENNReal.tsum_mul_right]; simp only [one_mul]
    _ ≤ 1 * B := by
        gcongr
        rw [← measure_iUnion (fun a a' hne => Set.disjoint_singleton.2 hne)
          fun a => measurableSet_singleton a]
        exact prob_le_one
    _ = B := one_mul B

/-! ## The theorem -/

theorem return_accuracy_holds : ReturnAccuracy := by
  intro Ω _ μ _ S _ Q R _ _ X _ _ _ C A O Dsamp ν K T stage hyp queried passes η₀ w ε δ κ β n
    hD hν hL hη hη₀ hw0 hγ hε hδ hκ hn hrep hβ hdep hreads hT hpow
  classical
  set ξ := Measure.pi fun _ : R => Measure.infinitePi fun _ : ℕ => Dsamp
  set F : C → Set ((Ω × X) × (R → ℕ → S)) := fun c =>
    {q | (∀ h, q.1 ∈ passes c h)
      ∧ accuracy A (hyp c) (fun h => denoisedLabel O n (hitsOf (hyp c) h (q.2 h)) q.1.1) Dsamp
        < 1 - w - ε}
  have hround : ∀ r : Fin K, (loopMeasure (μ := μ) (R := R) K Dsamp ν)
      {z | ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage r z.1 z.2.1)}
      ≤ ENNReal.ofReal (δ + Fintype.card R * β) := fun r =>
    loop_round_le O ν ξ r (stage r) (queried r) (hdep r) (hreads r) (hT r) F _
      fun c U b hU => cell_le A (hyp c) O Dsamp ν (passes c) U b hL hη hη₀ hw0 hγ hε hδ hκ hn
        hrep hβ hU fun h => hpow c h U b hU
  have hset : {z : Ω × (Fin K → X) × (Fin K → R → ℕ → S) | ∃ r,
        (∀ h, (z.1, z.2.1 r) ∈ passes (stage r z.1 z.2.1) h)
        ∧ accuracy A (hyp (stage r z.1 z.2.1))
            (fun h => denoisedLabel O n (hitsOf (hyp (stage r z.1 z.2.1)) h (z.2.2 r h)) z.1)
            Dsamp
          < 1 - w - ε}
      = ⋃ r, {z | ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage r z.1 z.2.1)} := by
    ext z
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion]
    rfl
  rw [hset]
  calc _ ≤ ∑ r, (loopMeasure (μ := μ) (R := R) K Dsamp ν).real
        {z | ((z.1, z.2.1 r), z.2.2 r) ∈ F (stage r z.1 z.2.1)} :=
        measureReal_iUnion_fintype_le _
    _ ≤ ∑ _r : Fin K, (δ + Fintype.card R * β) := Finset.sum_le_sum fun r _ => by
        rw [measureReal_def]
        exact ENNReal.toReal_le_of_le_ofReal (by positivity) (hround r)
    _ = K * (δ + Fintype.card R * β) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end OrthoDFA
