import OrthoDFA.Proofs.LearnerHits

/-!
# One round's clustering, on a pinned cell

Given the earlier rounds, the round clusters over fixed populations, so
`clustering_quality_bad` applies, but to i.i.d. streams under fresh noise.  The round instead
reads the hits among `M` draws, under noise pinned where earlier rounds read.  Three things
separate the two, each paid for once:

* the clustering reads only the first `P` prefixes of each stream, and when a population has
  that many hits they are no likelier than i.i.d. draws to take any values (`pi_hitsIn_le`);
* a population has fewer hits with probability at most a Hoeffding tail;
* off the pinned strings the noise is fresh, and the clustering's strings miss them unless a
  prefix lands on one of `|U|` strings, which the atom bound makes unlikely.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

namespace LearnerProof

/-! ## What the clustering's verdict depends on -/

section Generic

variable {J : Type*} {Q : Type*}

omit [IsProbabilityMeasure μ] in
lemma cluster_eq_of_finsets (O : Oracle μ S) (pops : Finset J) (il α : ℝ) (B : State)
    {x x' : Run Ω S J} (hω : x.1 = x'.1)
    (hP : prefixesAt pops B.npref x = prefixesAt pops B.npref x')
    (hV : poolAt B.nsuff x = poolAt B.nsuff x')
    (hC : ∀ j ∈ pops, certOf j B.npref x = certOf j B.npref x') :
    clusterAt O.mq pops x B = clusterAt O.mq pops x' B
      ∧ (x ∈ ret O.mq pops il α B ↔ x' ∈ ret O.mq pops il α B) := by
  rcases x with ⟨ω, d⟩
  rcases x' with ⟨ω', d'⟩
  obtain rfl : ω = ω' := hω
  have hc : clusterAt O.mq pops ((ω, d) : Run Ω S J) B = clusterAt O.mq pops (ω, d') B := by
    unfold clusterAt screenedAt
    rw [hP, hV]
    rfl
  refine ⟨hc, ?_⟩
  change _ ∧ _ ∧ _ ↔ _ ∧ _ ∧ _
  rw [hc]
  refine and_congr Iff.rfl (and_congr (forall₂_congr fun j hj => ?_)
    (forall₂_congr fun j hj => ?_))
  · rw [hC j hj]; rfl
  · rw [hC j hj]; rfl

omit [IsProbabilityMeasure μ] in
/-- `QualityGood` reads the draws only through the finsets its states cluster and gate on. -/
lemma qualityGood_congr_draws (A : DFA S Q) (O : Oracle μ S) (pops : Finset J)
    (D : J → Measure S) (states : Finset State) (il α tolerance εcov : ℝ) {x x' : Run Ω S J}
    (hω : x.1 = x'.1)
    (h : ∀ B ∈ states, prefixesAt pops B.npref x = prefixesAt pops B.npref x'
      ∧ poolAt B.nsuff x = poolAt B.nsuff x'
      ∧ ∀ j ∈ pops, certOf j B.npref x = certOf j B.npref x') :
    QualityGood A O pops D states il α tolerance εcov x
      ↔ QualityGood A O pops D states il α tolerance εcov x' := by
  have hB : ∀ B ∈ states, clusterAt O.mq pops x B = clusterAt O.mq pops x' B
      ∧ (x ∈ ret O.mq pops il α B ↔ x' ∈ ret O.mq pops il α B) := fun B hB =>
    cluster_eq_of_finsets O pops il α B hω (h B hB).1 (h B hB).2.1 (h B hB).2.2
  unfold QualityGood
  refine and_congr ⟨fun ⟨B, hBs, hr⟩ => ⟨B, hBs, (hB B hBs).2.1 hr⟩,
    fun ⟨B, hBs, hr⟩ => ⟨B, hBs, (hB B hBs).2.2 hr⟩⟩ (forall₂_congr fun B hBs => ?_)
  rw [(hB B hBs).2, (hB B hBs).1]

set_option linter.unusedFintypeInType false in
/-- `QualityGood` reads the noise only at the strings its states cluster and gate on. -/
lemma qualityGood_congr_noise [Fintype J] (A : DFA S Q) (O : Oracle μ S) (pops : Finset J)
    (D : J → Measure S) (states : Finset State) (il α tolerance εcov : ℝ)
    (d : ((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)) {ω ω' : Ω}
    (h : ∀ B ∈ states, ∀ w ∈ readSet (prefixesAt pops B.npref ((ω, d) : Run Ω S J)
        ∪ pops.biUnion (fun j => certOf j B.npref ((ω, d) : Run Ω S J)))
        (poolAt B.nsuff ((ω, d) : Run Ω S J)), O.noise w ω = O.noise w ω') :
    QualityGood A O pops D states il α tolerance εcov ((ω, d) : Run Ω S J)
      ↔ QualityGood A O pops D states il α tolerance εcov ((ω', d) : Run Ω S J) := by
  have hB : ∀ B ∈ states, clusterAt O.mq pops ((ω, d) : Run Ω S J) B
        = clusterAt O.mq pops ((ω', d) : Run Ω S J) B
      ∧ (((ω, d) : Run Ω S J) ∈ ret O.mq pops il α B
        ↔ ((ω', d) : Run Ω S J) ∈ ret O.mq pops il α B) := fun B hBs =>
    ⟨clusterAt_congr O pops B d fun w hw =>
      h B hBs w (readSet_mono Finset.subset_union_left subset_rfl hw),
      ret_congr O pops il α B d (h B hBs)⟩
  unfold QualityGood
  refine and_congr ⟨fun ⟨B, hBs, hr⟩ => ⟨B, hBs, (hB B hBs).2.1 hr⟩,
    fun ⟨B, hBs, hr⟩ => ⟨B, hBs, (hB B hBs).2.2 hr⟩⟩ (forall₂_congr fun B hBs => ?_)
  rw [(hB B hBs).2, (hB B hBs).1]

end Generic

/-! ## Noise at disjoint strings -/

/-- An event the noise at `Q` decides is, on the runs whose bits there are `0` or `1`, an
event of those bits alone. -/
lemma measurableSet_inter_noiseClean (O : Oracle μ S) (Q : Finset S) (E : Set Ω)
    (hE : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → (ω ∈ E ↔ ω' ∈ E)) :
    MeasurableSet[noiseAlg O ↑Q] (E ∩ noiseClean O Q) := by
  classical
  have hset : E ∩ noiseClean O Q = ⋃ t ∈ Q.powerset.filter
      (fun t => ∃ ω ∈ E ∩ noiseClean O Q, noisePattern O Q ω = t),
      ({ω | noisePattern O Q ω = t} ∩ noiseClean O Q) := by
    ext ω
    simp only [Set.mem_iUnion, Finset.mem_filter, Set.mem_inter_iff, Set.mem_ofPred_eq,
      exists_prop]
    constructor
    · rintro ⟨hE', hc⟩
      exact ⟨noisePattern O Q ω, ⟨noisePattern_mem O Q ω, ω, ⟨hE', hc⟩, rfl⟩, rfl, hc⟩
    · rintro ⟨t, ⟨-, ω0, ⟨hE0, hc0⟩, ht0⟩, ht, hc⟩
      exact ⟨(hE ω0 ω (noise_eq_of_pattern O hc0 hc (ht0.trans ht.symm))).1 hE0, hc⟩
  rw [hset]
  exact Finset.measurableSet_biUnion _ fun t _ =>
    (measurableSet_noisePattern O Q t).inter (measurableSet_noiseClean O Q)

lemma measurableSet_pinned_noiseAlg (O : Oracle μ S) (U : Finset S) (b : S → ℝ) :
    MeasurableSet[noiseAlg O ↑U] (pinned O U b) := by
  have : pinned O U b = ⋂ s ∈ U, O.noise s ⁻¹' {b s} := by
    ext ω; simp [pinned]
  rw [this]
  exact Finset.measurableSet_biInter U fun s hs =>
    measurableSet_noise_preimage O (Finset.mem_coe.2 hs) (measurableSet_singleton _)

lemma measure_pinned_inter (O : Oracle μ S) (U Q : Finset S) (b : S → ℝ)
    (hdisj : Disjoint U Q) {G : Set Ω} (hG : MeasurableSet[noiseAlg O ↑Q] G) :
    μ (pinned O U b ∩ G) = μ (pinned O U b) * μ G :=
  (Indep_iff _ _ μ).1 (indep_noiseAlg O (Finset.disjoint_coe.2 hdisj)) _ _
    (measurableSet_pinned_noiseAlg O U b) hG

/-! ## Comparing laws on countable spaces -/

/-- Where `En` holds, `f` is no likelier under `ν` than under `lam` to take any value. -/
lemma prod_preimage_le {X V : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] [MeasurableSpace V] [Countable V] [MeasurableSingletonClass V]
    (μ' : Measure Ω) [SFinite μ'] (ν : Measure X) [SFinite ν] (lam : Measure V) [SFinite lam]
    (f : X → V) (En : Set X) (hle : ∀ s, ν (En ∩ f ⁻¹' {s}) ≤ lam {s}) {G : Set (Ω × V)}
    (hG : MeasurableSet G) :
    (μ'.prod ν) (Prod.map id f ⁻¹' G) ≤ (μ'.prod lam) G + μ' Set.univ * ν Enᶜ := by
  have hf : Measurable f := measurable_of_countable f
  set g : V → ℝ≥0∞ := fun v => μ' ((fun ω => (ω, v)) ⁻¹' G)
  have hg : Measurable g := measurable_of_countable g
  have hEn : MeasurableSet En := (Set.to_countable En).measurableSet
  have hmap : (ν.restrict En).map f ≤ lam := by
    refine Measure.le_iff.2 fun s hs => ?_
    rw [Measure.map_apply hf hs, ← tsum_measure_preimage_singleton (Set.to_countable s)
      fun y _ => hf (measurableSet_singleton y)]
    calc ∑' b : s, ν.restrict En (f ⁻¹' {(b : V)})
        ≤ ∑' b : s, lam {(b : V)} := ENNReal.tsum_le_tsum fun b => by
          rw [Measure.restrict_apply (hf (measurableSet_singleton _)), Set.inter_comm]
          exact hle b
      _ = lam s := by
          have := tsum_measure_preimage_singleton (μ := lam) (Set.to_countable s)
            (f := id) fun y _ => measurableSet_singleton y
          simpa using this
  rw [Measure.prod_apply_symm (hG.preimage (measurable_id.prodMap hf)),
    Measure.prod_apply_symm hG]
  calc ∫⁻ x, μ' ((fun ω => (ω, x)) ⁻¹' (Prod.map id f ⁻¹' G)) ∂ν
      = ∫⁻ x, g (f x) ∂ν := rfl
    _ ≤ ∫⁻ x, (En.indicator (fun x => g (f x)) x + Enᶜ.indicator (fun _ => μ' Set.univ) x) ∂ν := by
        refine lintegral_mono fun x => ?_
        by_cases hx : x ∈ En
        · rw [Set.indicator_of_mem hx]; exact le_self_add
        · rw [Set.indicator_of_mem (Set.mem_compl hx)]
          exact le_add_left (measure_mono (Set.subset_univ _))
    _ = ∫⁻ x, g (f x) ∂(ν.restrict En) + μ' Set.univ * ν Enᶜ := by
        rw [lintegral_add_right _ ((measurable_const.indicator hEn.compl)),
          lintegral_indicator hEn, lintegral_indicator_const hEn.compl]
    _ = ∫⁻ v, g v ∂((ν.restrict En).map f) + μ' Set.univ * ν Enᶜ := by
        rw [lintegral_map hg hf]
    _ ≤ ∫⁻ v, g v ∂lam + μ' Set.univ * ν Enᶜ := by
        gcongr

/-- On a pinned cell, an event of the noise at strings `RV v` off the pinned ones is as likely as
under fresh noise. -/
lemma pinned_prod_le {V : Type*} [MeasurableSpace V] [Countable V] [MeasurableSingletonClass V]
    (O : Oracle μ S) (U : Finset S) (b : S → ℝ) (lam : Measure V) [SFinite lam]
    (RV : V → Finset S) {G : Set (Ω × V)} (hG : MeasurableSet G)
    (hGv : ∀ v, MeasurableSet[noiseAlg O ↑(RV v)] ((fun ω => (ω, v)) ⁻¹' G)) :
    ((μ.restrict (pinned O U b)).prod lam) G
      ≤ μ (pinned O U b) * ((μ.prod lam) G + lam {v | ¬ Disjoint U (RV v)}) := by
  classical
  set P := pinned O U b
  have hP := measurableSet_pinned O U b
  set Hit := {v | ¬ Disjoint U (RV v)}
  have hHit : MeasurableSet Hit := (Set.to_countable Hit).measurableSet
  have hpt : ∀ v, μ.restrict P ((fun ω => (ω, v)) ⁻¹' G)
      ≤ μ P * (μ ((fun ω => (ω, v)) ⁻¹' G) + Hit.indicator 1 v) := by
    intro v
    rw [Measure.restrict_apply (noiseAlg_le O _ _ (hGv v)), Set.inter_comm]
    by_cases hv : v ∈ Hit
    · rw [Set.indicator_of_mem hv, Pi.one_apply]
      calc μ (P ∩ _) ≤ μ P := measure_mono Set.inter_subset_left
        _ = μ P * 1 := (mul_one _).symm
        _ ≤ _ := by gcongr; exact le_add_self
    · rw [Set.indicator_of_notMem hv, add_zero]
      exact (measure_pinned_inter O U (RV v) b (not_not.1 hv) (hGv v)).le
  rw [Measure.prod_apply_symm hG, Measure.prod_apply_symm hG]
  calc ∫⁻ v, μ.restrict P ((fun ω => (ω, v)) ⁻¹' G) ∂lam
      ≤ ∫⁻ v, μ P * (μ ((fun ω => (ω, v)) ⁻¹' G) + Hit.indicator 1 v) ∂lam :=
        lintegral_mono hpt
    _ = μ P * (∫⁻ v, μ ((fun ω => (ω, v)) ⁻¹' G) ∂lam + lam Hit) := by
        rw [lintegral_const_mul _ (measurable_of_countable _),
          lintegral_add_right _ (measurable_one.indicator hHit), lintegral_indicator_one hHit]

/-! ## One round's clustering -/

section Round

variable {K : ℕ} {R : Type*} {Q : Type*}

/-- The populations' laws after the rounds `past`: the sampler, and each failed state's own. -/
noncomputable def popMeasure (Dsamp : Measure S) (past : List (DFA S R × Finset R)) :
    Option (Fin K × R) → Measure S
  | none => Dsamp
  | some (i, h) => match past[i.val]? with
    | some o => reaching o.1 Dsamp h
    | none => Dsamp

lemma isProbabilityMeasure_popMeasure (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    (past : List (DFA S R × Finset R)) (j : Option (Fin K × R)) :
    IsProbabilityMeasure (popMeasure Dsamp past j) := by
  rcases j with _ | ⟨i, h⟩
  · simp only [popMeasure]; infer_instance
  · simp only [popMeasure]
    cases past[i.val]? with
    | none => simp only; infer_instance
    | some o => exact isProbabilityMeasure_reaching o.1 Dsamp h

variable [Fintype R]

/-- The first `M` suffixes and the first `P` prefixes and certification prefixes of each
population. -/
abbrev Trunc (S : Type*) (K : ℕ) (R : Type*) (M P : ℕ) :=
  ((Fin M → S) × (Option (Fin K × R) → Fin P → S)) × (Option (Fin K × R) → Fin P → S)

open scoped Classical in
/-- Population `j`'s first `P` draws, if it is a population. -/
noncomputable def popTrunc {M : ℕ} (P : ℕ) (past : List (DFA S R × Finset R))
    (j : Option (Fin K × R)) (a : Fin M → S) : Fin P → S :=
  if j ∈ populationsAfter past then fun i => populationDraws past j a i else fun _ => 1

noncomputable def truncL {M : ℕ} (P : ℕ) (past : List (DFA S R × Finset R))
    (e : ClusterPart S K R M) : Trunc S K R M P :=
  ((e.1, fun j => popTrunc P past j (e.2.1 j)), fun j => popTrunc P past j (e.2.2 j))

open scoped Classical in
/-- A stream's first `P` draws, if it is a population's. -/
noncomputable def streamTrunc (P : ℕ) (pops : Finset (Option (Fin K × R)))
    (j : Option (Fin K × R)) (u : ℕ → S) : Fin P → S :=
  if j ∈ pops then fun i => u i else fun _ => 1

noncomputable def truncR (M P : ℕ) (pops : Finset (Option (Fin K × R))) :
    ((ℕ → S) × (Option (Fin K × R) → ℕ → S)) × (Option (Fin K × R) → ℕ → S) →
      Trunc S K R M P :=
  Prod.map (Prod.map (fun (u : ℕ → S) (i : Fin M) => u i)
    (fun a j => streamTrunc P pops j (a j))) (fun a j => streamTrunc P pops j (a j))

noncomputable def untrunc {M P : ℕ} (v : Trunc S K R M P) :
    ((ℕ → S) × (Option (Fin K × R) → ℕ → S)) × (Option (Fin K × R) → ℕ → S) :=
  ((padded v.1.1, fun j => padded (v.1.2 j)), fun j => padded (v.2 j))

section Finsets

variable {J : Type*}

omit [MeasurableSpace Ω] in
lemma prefixesAt_congr (pops : Finset J) (n : ℕ) {x x' : Run Ω S J}
    (h : ∀ j ∈ pops, ∀ i < n, prefixDraw j i x = prefixDraw j i x') :
    prefixesAt pops n x = prefixesAt pops n x' := by
  classical
  exact Finset.biUnion_congr rfl fun j hj =>
    Finset.image_congr fun i hi => h j hj i (Finset.mem_range.1 hi)

omit [MeasurableSpace Ω] in
lemma certOf_congr (j : J) (n : ℕ) {x x' : Run Ω S J}
    (h : ∀ i < n, certPrefix j i x = certPrefix j i x') : certOf j n x = certOf j n x' := by
  classical
  exact Finset.image_congr fun i hi => h i (Finset.mem_range.1 hi)

omit [MeasurableSpace Ω] in
lemma poolAt_congr (n : ℕ) {x x' : Run Ω S J} (h : ∀ i < n, suffixDraw i x = suffixDraw i x') :
    poolAt n x = poolAt n x' := by
  classical
  unfold poolAt
  rw [Finset.image_congr fun i hi => h i (Finset.mem_range.1 hi)]

end Finsets

omit [IsProbabilityMeasure μ] in
lemma good_truncL (A : DFA S Q) (O : Oracle μ S) (D : Option (Fin K × R) → Measure S)
    (states : Finset State) (il α tolerance εcov : ℝ) {M P : ℕ}
    (hst : ∀ B ∈ states, B.npref ≤ P ∧ B.nsuff ≤ M) (past : List (DFA S R × Finset R))
    (ω : Ω) (e : ClusterPart S K R M) :
    QualityGood A O (populationsAfter past) D states il α tolerance εcov
        ((ω, clusterDraws past e) : Run Ω S (Option (Fin K × R)))
      ↔ QualityGood A O (populationsAfter past) D states il α tolerance εcov
        ((ω, untrunc (truncL P past e)) : Run Ω S (Option (Fin K × R))) := by
  classical
  refine qualityGood_congr_draws A O _ D states il α tolerance εcov rfl fun B hB => ?_
  obtain ⟨hp, hs⟩ := hst B hB
  refine ⟨prefixesAt_congr _ _ fun j hj i hi => ?_, poolAt_congr _ fun i hi => ?_,
    fun j hj => certOf_congr _ _ fun i hi => ?_⟩
  · change populationDraws past j (e.2.1 j) i = padded _ i
    simp [padded, show i < P by omega, hj, truncL, popTrunc]
  · rfl
  · change populationDraws past j (e.2.2 j) i = padded _ i
    simp [padded, show i < P by omega, hj, truncL, popTrunc]

omit [Fintype R] [IsProbabilityMeasure μ] in
lemma good_truncR (A : DFA S Q) (O : Oracle μ S) (D : Option (Fin K × R) → Measure S)
    (pops : Finset (Option (Fin K × R))) (states : Finset State) (il α tolerance εcov : ℝ)
    {M P : ℕ} (hst : ∀ B ∈ states, B.npref ≤ P ∧ B.nsuff ≤ M) (ω : Ω)
    (xd : ((ℕ → S) × (Option (Fin K × R) → ℕ → S)) × (Option (Fin K × R) → ℕ → S)) :
    QualityGood A O pops D states il α tolerance εcov ((ω, xd) : Run Ω S (Option (Fin K × R)))
      ↔ QualityGood A O pops D states il α tolerance εcov
        ((ω, untrunc (truncR M P pops xd)) : Run Ω S (Option (Fin K × R))) := by
  classical
  refine qualityGood_congr_draws A O _ D states il α tolerance εcov rfl fun B hB => ?_
  obtain ⟨hp, hs⟩ := hst B hB
  refine ⟨prefixesAt_congr _ _ fun j hj i hi => ?_, poolAt_congr _ fun i hi => ?_,
    fun j hj => certOf_congr _ _ fun i hi => ?_⟩
  · change xd.1.2 j i = padded _ i
    simp [padded, show i < P by omega, hj, truncR, streamTrunc]
  · change xd.1.1 i = padded _ i
    simp [padded, show i < M by omega, truncR]
  · change xd.2 j i = padded _ i
    simp [padded, show i < P by omega, hj, truncR, streamTrunc]

/-- The strings the clustering reads at the states of `states`, from the truncated draws. -/
noncomputable def truncReads {M P : ℕ} (pops : Finset (Option (Fin K × R)))
    (v : Trunc S K R M P) : Finset S :=
  readSet (pops.biUnion fun j => Finset.univ.image (v.1.2 j) ∪ Finset.univ.image (v.2 j))
    (insert 1 (Finset.univ.image v.1.1))

set_option linter.unusedFintypeInType false in
lemma good_noise (A : DFA S Q) (O : Oracle μ S) (D : Option (Fin K × R) → Measure S)
    (pops : Finset (Option (Fin K × R))) (states : Finset State) (il α tolerance εcov : ℝ)
    {M P : ℕ} (hst : ∀ B ∈ states, B.npref ≤ P ∧ B.nsuff ≤ M) (v : Trunc S K R M P)
    {ω ω' : Ω} (h : ∀ w ∈ truncReads pops v, O.noise w ω = O.noise w ω') :
    QualityGood A O pops D states il α tolerance εcov
        ((ω, untrunc v) : Run Ω S (Option (Fin K × R)))
      ↔ QualityGood A O pops D states il α tolerance εcov
        ((ω', untrunc v) : Run Ω S (Option (Fin K × R))) := by
  classical
  refine qualityGood_congr_noise A O pops D states il α tolerance εcov _ fun B hB w hw => h w ?_
  obtain ⟨hp, hs⟩ := hst B hB
  refine readSet_mono ?_ ?_ hw
  · refine Finset.union_subset ?_ ?_
    · intro p hp'
      obtain ⟨j, hj, hp'⟩ := Finset.mem_biUnion.1 hp'
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp'
      have hi : i < P := by have := Finset.mem_range.1 hi; omega
      refine Finset.mem_biUnion.2 ⟨j, hj, Finset.mem_union_left _ (Finset.mem_image.2
        ⟨⟨i, hi⟩, Finset.mem_univ _, ?_⟩)⟩
      simp [prefixDraw, untrunc, padded, hi]
    · intro p hp'
      obtain ⟨j, hj, hp'⟩ := Finset.mem_biUnion.1 hp'
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hp'
      have hi : i < P := by have := Finset.mem_range.1 hi; omega
      refine Finset.mem_biUnion.2 ⟨j, hj, Finset.mem_union_right _ (Finset.mem_image.2
        ⟨⟨i, hi⟩, Finset.mem_univ _, ?_⟩)⟩
      simp [certPrefix, untrunc, padded, hi]
  · intro w hw'
    rcases Finset.mem_insert.1 hw' with rfl | hw'
    · exact Finset.mem_insert_self _ _
    obtain ⟨i, hi, rfl⟩ := Finset.mem_image.1 hw'
    have hi : i < M := by have := Finset.mem_range.1 hi; omega
    refine Finset.mem_insert_of_mem (Finset.mem_image.2 ⟨⟨i, hi⟩, Finset.mem_univ _, ?_⟩)
    simp [suffixDraw, untrunc, padded, hi]

/-! ### The laws of the truncated draws -/

open scoped Classical in
noncomputable def popLaw (Dsamp : Measure S) (P : ℕ) (past : List (DFA S R × Finset R))
    (j : Option (Fin K × R)) : Measure (Fin P → S) :=
  if j ∈ populationsAfter past then Measure.pi fun _ : Fin P => popMeasure Dsamp past j
  else Measure.dirac fun _ => 1

lemma isProbabilityMeasure_popLaw (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (P : ℕ)
    (past : List (DFA S R × Finset R)) (j : Option (Fin K × R)) :
    IsProbabilityMeasure (popLaw Dsamp P past j) := by
  have := isProbabilityMeasure_popMeasure Dsamp past j
  unfold popLaw
  split_ifs <;> infer_instance

/-- The law the clustering's truncated draws have under i.i.d. streams. -/
noncomputable def truncLaw (Dsamp Dsf : Measure S) (M P : ℕ) (past : List (DFA S R × Finset R)) :
    Measure (Trunc S K R M P) :=
  ((Measure.pi fun _ : Fin M => Dsf).prod (Measure.pi (popLaw Dsamp P past))).prod
    (Measure.pi (popLaw Dsamp P past))

/-- The law of a round's clustering draws. -/
noncomputable def clusterLaw (Dsamp Dsf : Measure S) (M : ℕ) : Measure (ClusterPart S K R M) :=
  (Measure.pi fun _ : Fin M => Dsf).prod
    ((Measure.pi fun _ : Option (Fin K × R) => Measure.pi fun _ : Fin M => Dsamp).prod
      (Measure.pi fun _ : Option (Fin K × R) => Measure.pi fun _ : Fin M => Dsamp))

instance isProbabilityMeasure_clusterLaw (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] (M : ℕ) :
    IsProbabilityMeasure (clusterLaw (K := K) (R := R) Dsamp Dsf M) := by
  unfold clusterLaw; infer_instance

lemma isProbabilityMeasure_truncLaw (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] (M P : ℕ) (past : List (DFA S R × Finset R)) :
    IsProbabilityMeasure (truncLaw (K := K) Dsamp Dsf M P past) := by
  have := isProbabilityMeasure_popLaw (K := K) Dsamp P past
  unfold truncLaw; infer_instance

/-- `runMeasure` without the noise. -/
noncomputable def runDraws {J : Type*} [Fintype J] (D : J → Measure S) (Dsf : Measure S) :
    Measure (((ℕ → S) × (J → ℕ → S)) × (J → ℕ → S)) :=
  ((Measure.infinitePi fun _ : ℕ => Dsf).prod
    (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => D j)).prod
    (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => D j)

lemma measurePreserving_truncR (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] (M P : ℕ) (past : List (DFA S R × Finset R)) :
    MeasurePreserving (truncR M P (populationsAfter past))
      (runDraws (popMeasure Dsamp past) Dsf) (truncLaw (K := K) Dsamp Dsf M P past) := by
  classical
  have := isProbabilityMeasure_popMeasure (K := K) Dsamp past
  have := isProbabilityMeasure_popLaw (K := K) Dsamp P past
  have hj : ∀ j : Option (Fin K × R), MeasurePreserving (streamTrunc P (populationsAfter past) j)
      (Measure.infinitePi fun _ : ℕ => popMeasure Dsamp past j) (popLaw Dsamp P past j) := by
    intro j
    unfold streamTrunc popLaw
    by_cases hmem : j ∈ populationsAfter past
    · simp only [hmem, if_true]
      exact measurePreserving_finRestrict _ P
    · simp only [hmem, if_false]
      refine ⟨measurable_const, ?_⟩
      rw [Measure.map_const, measure_univ, one_smul]
  exact ((measurePreserving_finRestrict Dsf M).prod (measurePreserving_pi _ _ hj)).prod
    (measurePreserving_pi _ _ hj)

/-- Population `j` has at least `P` hits among the draws `a`. -/
def enoughAt (P : ℕ) (past : List (DFA S R × Finset R)) {M : ℕ} :
    Option (Fin K × R) → (Fin M → S) → Prop
  | none, _ => True
  | some (i, h), a => ∀ o ∈ past[i.val]?, P ≤ hitCount o.1 h a

/-- Every population has at least `P` hits among its prefix draws and its certification draws. -/
def enoughSet {M : ℕ} (P : ℕ) (past : List (DFA S R × Finset R)) : Set (ClusterPart S K R M) :=
  {e | ∀ j ∈ populationsAfter past, enoughAt P past j (e.2.1 j) ∧ enoughAt P past j (e.2.2 j)}

lemma pi_padded_eq (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] {M P : ℕ} (hPM : P ≤ M)
    (t : Fin P → S) :
    (Measure.pi fun _ : Fin M => Dsamp) {a | (fun i : Fin P => padded a i) = t}
      = Measure.pi (fun _ : Fin P => Dsamp) {t} := by
  have hM := measurePreserving_finRestrict Dsamp M
  have hP := measurePreserving_finRestrict Dsamp P
  rw [← hM.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
    ← hP.measure_preimage (measurableSet_singleton t).nullMeasurableSet]
  congr 1
  ext u
  simp only [Set.mem_preimage, Set.mem_ofPred_eq, Set.mem_singleton_iff, funext_iff]
  refine forall_congr' fun i => ?_
  simp [padded, show (i : ℕ) < M by omega]

omit [Fintype R] in
lemma measure_fiber_le {M P : ℕ} (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (hPM : P ≤ M)
    [Fintype R] (past : List (DFA S R × Finset R))
    (hpos : ∀ (i : Fin K) (h : R) o, past[i.val]? = some o → h ∈ o.2 →
      Dsamp {v | o.1.state v = h} ≠ 0)
    (j : Option (Fin K × R)) (t : Fin P → S) :
    (Measure.pi fun _ : Fin M => Dsamp)
        {a | (j ∈ populationsAfter past → enoughAt P past j a) ∧ popTrunc P past j a = t}
      ≤ popLaw Dsamp P past j {t} := by
  classical
  unfold popLaw
  by_cases hj : j ∈ populationsAfter past
  · rw [if_pos hj]
    rcases j with _ | ⟨i, h⟩
    · refine (measure_mono fun a ha => ?_).trans (pi_padded_eq Dsamp hPM t).le
      have := ha.2
      simp only [popTrunc, if_pos hj] at this
      exact this
    · obtain ⟨o, ho, hho⟩ := (Finset.mem_filter.1 hj).2
      have ho' : past[i.val]? = some o := by simpa using ho
      have hpm : popMeasure Dsamp past (some (i, h)) = reaching o.1 Dsamp h := by
        simp [popMeasure, ho']
      have := isProbabilityMeasure_reaching o.1 Dsamp h
      rw [hpm, Measure.pi_singleton]
      refine (measure_mono fun a ha => ?_).trans (pi_hitsIn_le Dsamp o.1 h
        (hpos i h o ho' hho) M P t)
      obtain ⟨hen, ht⟩ := ha
      refine ⟨hen hj o (by simp [ho']), ?_⟩
      simpa [popTrunc, hj, populationDraws, ho'] using ht
  · rw [if_neg hj]
    by_cases ht : (fun _ : Fin P => (1 : S)) = t
    · rw [Measure.dirac_apply_of_mem (by simp [ht])]
      exact prob_le_one
    · refine le_of_eq_of_le (measure_mono_null (fun a ha => ?_) measure_empty) zero_le
      have := ha.2
      simp only [popTrunc, if_neg hj] at this
      exact ht this

/-- With enough hits, the truncated draws are no likelier than under i.i.d. streams to take
any value. -/
lemma clusterLaw_fiber_le (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] {M P : ℕ} (hPM : P ≤ M) (past : List (DFA S R × Finset R))
    (hpos : ∀ (i : Fin K) (h : R) o, past[i.val]? = some o → h ∈ o.2 →
      Dsamp {v | o.1.state v = h} ≠ 0) (s : Trunc S K R M P) :
    clusterLaw Dsamp Dsf M (enoughSet P past ∩ truncL P past ⁻¹' {s})
      ≤ truncLaw Dsamp Dsf M P past {s} := by
  classical
  have := isProbabilityMeasure_popLaw (K := K) Dsamp P past
  set Aj : Option (Fin K × R) → (Fin P → S) → Set (Fin M → S) := fun j t =>
    {a | (j ∈ populationsAfter past → enoughAt P past j a) ∧ popTrunc P past j a = t}
  have hsub : enoughSet P past ∩ truncL P past ⁻¹' {s}
      ⊆ {s.1.1} ×ˢ (Set.univ.pi (fun j => Aj j (s.1.2 j))
        ×ˢ Set.univ.pi (fun j => Aj j (s.2 j))) := by
    rintro e ⟨hen, hs⟩
    simp only [Set.mem_preimage, Set.mem_singleton_iff, truncL] at hs
    rw [← hs]
    refine ⟨rfl, fun j _ => ⟨fun hj => (hen j hj).1, rfl⟩, fun j _ => ⟨fun hj => (hen j hj).2, rfl⟩⟩
  have hs : ({s} : Set (Trunc S K R M P))
      = ({s.1.1} ×ˢ Set.univ.pi (fun j => {s.1.2 j})) ×ˢ Set.univ.pi (fun j => {s.2 j}) := by
    ext x
    simp only [Set.mem_singleton_iff, Set.mem_prod, Set.mem_pi, Set.mem_univ, forall_const]
    constructor
    · rintro rfl; exact ⟨⟨rfl, fun _ => rfl⟩, fun _ => rfl⟩
    · rintro ⟨⟨h1, h2⟩, h3⟩
      exact Prod.ext (Prod.ext h1 (funext h2)) (funext h3)
  refine (measure_mono hsub).trans ?_
  rw [hs, clusterLaw, truncLaw, Measure.prod_prod, Measure.prod_prod, Measure.prod_prod,
    Measure.prod_prod, Measure.pi_pi, Measure.pi_pi, Measure.pi_pi, Measure.pi_pi, mul_assoc]
  gcongr with j _ j _
  · exact measure_fiber_le Dsamp hPM past hpos j (s.1.2 j)
  · exact measure_fiber_le Dsamp hPM past hpos j (s.2 j)

/-- A population short of hits: a Hoeffding tail per array. -/
lemma clusterLaw_short_le (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] {M P : ℕ} (hM : 1 ≤ M) (past : List (DFA S R × Finset R))
    {ε : ℝ}
    (hheavy : ∀ (i : Fin K) (h : R) o, past[i.val]? = some o → h ∈ o.2 →
      ε / Fintype.card R ≤ Dsamp.real {v | o.1.state v = h})
    (hP : (P : ℝ) ≤ M * ε / Fintype.card R) :
    clusterLaw Dsamp Dsf M (enoughSet (K := K) P past)ᶜ
      ≤ ENNReal.ofReal (2 * Fintype.card (Option (Fin K × R))
        * Real.exp (-2 * (M * ε / Fintype.card R - P) ^ 2 / M)) := by
  classical
  set tail := Real.exp (-2 * (M * ε / Fintype.card R - P) ^ 2 / M)
  have hM0 : (0 : ℝ) < M := by exact_mod_cast hM
  have harr : ∀ j ∈ populationsAfter (K := K) past,
      (Measure.pi fun _ : Fin M => Dsamp).real {a | ¬ enoughAt P past j a} ≤ tail := by
    intro j hj
    rcases j with _ | ⟨i, h⟩
    · rw [show {a : Fin M → S | ¬ enoughAt P past (none : Option (Fin K × R)) a} = ∅ from
        Set.eq_empty_of_forall_notMem fun a ha => ha trivial, measureReal_empty]
      exact (Real.exp_pos _).le
    obtain ⟨o, ho, hho⟩ := (Finset.mem_filter.1 hj).2
    have ho' : past[i.val]? = some o := by simpa using ho
    set W := {v : S | o.1.state v = h}
    set q := Dsamp.real W
    have hq : ε / Fintype.card R ≤ q := hheavy i h o ho' hho
    have hPq : (P : ℝ) / M ≤ q := by
      rw [div_le_iff₀ hM0]
      calc (P : ℝ) ≤ M * ε / Fintype.card R := hP
        _ = ε / Fintype.card R * M := by ring
        _ ≤ q * M := by gcongr
    have hsub : {a : Fin M → S | ¬ enoughAt P past (some (i, h)) a}
        ⊆ {r | (((Finset.univ : Finset (Fin M)).filter (fun k => r k ∈ W)).card : ℝ)
          ≤ (M : ℝ) * (q - (q - P / M))} := by
      intro a ha
      simp only [enoughAt, ho', Option.mem_def, Option.some.injEq, forall_eq', not_le,
        Set.mem_ofPred_eq] at ha
      rw [hitCount_eq_card] at ha
      simp only [Set.mem_ofPred_eq]
      rw [sub_sub_cancel, mul_div_cancel₀ _ hM0.ne']
      exact_mod_cast ha.le
    refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
    refine (pi_hits_lower Dsamp M W q (q - P / M) measureReal_nonneg (by linarith) le_rfl).trans ?_
    refine Real.exp_le_exp.2 ?_
    have h1 : 0 ≤ M * ε / Fintype.card R - P := by linarith
    have h2 : M * ε / Fintype.card R - P ≤ M * q - P := by
      have : M * ε / Fintype.card R = M * (ε / Fintype.card R) := by ring
      rw [this]
      nlinarith
    have h3 : (M * ε / Fintype.card R - P) ^ 2 ≤ (M * q - P) ^ 2 := by nlinarith
    have h4 : -2 * (M : ℝ) * (q - P / M) ^ 2 = -2 * (M * q - P) ^ 2 / M := by
      field_simp
    rw [h4]
    exact div_le_div_of_nonneg_right (by nlinarith) hM0.le
  have hproj1 : ∀ j, MeasurePreserving (fun e : ClusterPart S K R M => e.2.1 j)
      (clusterLaw Dsamp Dsf M) (Measure.pi fun _ : Fin M => Dsamp) := fun j =>
    (measurePreserving_eval _ j).comp (measurePreserving_fst.comp measurePreserving_snd)
  have hproj2 : ∀ j, MeasurePreserving (fun e : ClusterPart S K R M => e.2.2 j)
      (clusterLaw Dsamp Dsf M) (Measure.pi fun _ : Fin M => Dsamp) := fun j =>
    (measurePreserving_eval _ j).comp (measurePreserving_snd.comp measurePreserving_snd)
  have hsub : (enoughSet (M := M) P past)ᶜ ⊆ ⋃ j ∈ populationsAfter (K := K) past,
      ((fun e : ClusterPart S K R M => e.2.1 j) ⁻¹' {a | ¬ enoughAt P past j a}
        ∪ (fun e : ClusterPart S K R M => e.2.2 j) ⁻¹' {a | ¬ enoughAt P past j a}) := by
    intro e he
    simp only [enoughSet, Set.mem_compl_iff, Set.mem_ofPred_eq, not_forall, not_and_or] at he
    obtain ⟨j, hj, h⟩ := he
    exact Set.mem_biUnion hj h
  rw [← ofReal_measureReal (μ := clusterLaw Dsamp Dsf M)]
  refine ENNReal.ofReal_le_ofReal ?_
  calc (clusterLaw Dsamp Dsf M).real (enoughSet (K := K) P past)ᶜ
      ≤ ∑ j ∈ populationsAfter (K := K) past, (clusterLaw Dsamp Dsf M).real
          ((fun e : ClusterPart S K R M => e.2.1 j) ⁻¹' {a | ¬ enoughAt P past j a}
            ∪ (fun e : ClusterPart S K R M => e.2.2 j) ⁻¹' {a | ¬ enoughAt P past j a}) :=
        (measureReal_mono hsub (measure_ne_top _ _)).trans (measureReal_biUnion_finset_le _ _)
    _ ≤ ∑ _j ∈ populationsAfter (K := K) past, 2 * tail := by
        refine Finset.sum_le_sum fun j hj => (measureReal_union_le _ _).trans ?_
        rw [(hproj1 j).measureReal_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
          (hproj2 j).measureReal_preimage (Set.to_countable _).measurableSet.nullMeasurableSet]
        linarith [harr j hj]
    _ ≤ Fintype.card (Option (Fin K × R)) * (2 * tail) := by
        rw [Finset.sum_const, nsmul_eq_mul]
        exact mul_le_mul_of_nonneg_right (by exact_mod_cast Finset.card_le_univ _)
          (by positivity)
    _ = _ := by ring

lemma measure_mul_mem_le (ρ : Measure S) [IsFiniteMeasure ρ] {c : ℝ}
    (hc : ∀ a, ρ.real {a} ≤ c) (U : Finset S) (x : S) :
    ρ {p | p * x ∈ U} ≤ ENNReal.ofReal (U.card * c) := by
  classical
  have hone : ∀ u : S, ρ {p | p * x = u} ≤ ENNReal.ofReal c := by
    intro u
    by_cases hex : ∃ p0, p0 * x = u
    · obtain ⟨p0, hp0⟩ := hex
      have : {p | p * x = u} = {p0} := by
        ext p
        simp only [Set.mem_ofPred_eq, Set.mem_singleton_iff]
        exact ⟨fun hp => mul_right_cancel (hp.trans hp0.symm), fun hp => hp ▸ hp0⟩
      rw [this, ← ofReal_measureReal]
      exact ENNReal.ofReal_le_ofReal (hc p0)
    · simp only [not_exists] at hex
      rw [show {p | p * x = u} = ∅ from Set.eq_empty_of_forall_notMem fun p hp => hex p hp,
        measure_empty]
      exact zero_le
  calc ρ {p | p * x ∈ U} ≤ ρ (⋃ u ∈ U, {p | p * x = u}) := measure_mono fun p hp => by
        simpa using hp
    _ ≤ ∑ u ∈ U, ρ {p | p * x = u} := measure_biUnion_finset_le _ _
    _ ≤ ∑ _u ∈ U, ENNReal.ofReal c := Finset.sum_le_sum fun u _ => hone u
    _ = ENNReal.ofReal (U.card * c) := by
        rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]

lemma prod_mul_mem_le (Dsf ρ : Measure S) [IsProbabilityMeasure Dsf] [IsFiniteMeasure ρ]
    {c : ℝ} (hc0 : 0 ≤ c) (hc : ∀ a, ρ.real {a} ≤ c) (U : Finset S) :
    (Dsf.prod ρ).real {q | q.2 * q.1 ∈ U} ≤ U.card * c := by
  have h : (Dsf.prod ρ) {q | q.2 * q.1 ∈ U} ≤ ENNReal.ofReal (U.card * c) := by
    rw [Measure.prod_apply (Set.to_countable _).measurableSet]
    calc ∫⁻ x, ρ (Prod.mk x ⁻¹' {q : S × S | q.2 * q.1 ∈ U}) ∂Dsf
        ≤ ∫⁻ _x, ENNReal.ofReal (U.card * c) ∂Dsf :=
          lintegral_mono fun x => measure_mul_mem_le ρ hc U x
      _ = _ := by rw [lintegral_const, measure_univ, mul_one]
  rw [measureReal_def]
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) h

/-- Some string the clustering reads is pinned: a prefix, or a prefix extended by a suffix,
lands on one of `|U|` strings. -/
lemma truncLaw_hit_le (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] {M P : ℕ} (past : List (DFA S R × Finset R)) (U : Finset S)
    {c : ℝ} (hc0 : 0 ≤ c)
    (hc : ∀ j ∈ populationsAfter (K := K) past, ∀ a, (popMeasure Dsamp past j).real {a} ≤ c) :
    (truncLaw (K := K) Dsamp Dsf M P past).real
        {v | ¬ Disjoint U (truncReads (populationsAfter past) v)}
      ≤ 2 * Fintype.card (Option (Fin K × R)) * P * (M + 1) * U.card * c := by
  classical
  have := isProbabilityMeasure_popLaw (K := K) Dsamp P past
  have := isProbabilityMeasure_popMeasure (K := K) Dsamp past
  have := isProbabilityMeasure_truncLaw (K := K) Dsamp Dsf M P past
  set L := truncLaw (K := K) Dsamp Dsf M P past
  set pops := populationsAfter (K := K) past
  set E : Set (S × S) := {q | q.2 * q.1 ∈ U}
  have hE : MeasurableSet E := (Set.to_countable E).measurableSet
  have hUm : MeasurableSet (U : Set S) := (Set.to_countable _).measurableSet
  have hB : ∀ j ∈ pops, ∀ i : Fin P, MeasurePreserving
      (fun w : Option (Fin K × R) → Fin P → S => w j i) (Measure.pi (popLaw Dsamp P past))
      (popMeasure Dsamp past j) := fun j hj i => by
    have h1 := measurePreserving_eval (popLaw Dsamp P past) j
    rw [show popLaw Dsamp P past j = Measure.pi fun _ : Fin P => popMeasure Dsamp past j
      from if_pos hj] at h1
    exact (measurePreserving_eval _ i).comp h1
  have hU1 : ∀ j ∈ pops, (popMeasure Dsamp past j).real U ≤ U.card * c := fun j hj =>
    calc (popMeasure Dsamp past j).real U ≤ (popMeasure Dsamp past j).real (⋃ v ∈ U, {v}) :=
          measureReal_mono (fun v hv => by simpa using hv) (measure_ne_top _ _)
      _ ≤ ∑ v ∈ U, (popMeasure Dsamp past j).real {v} := measureReal_biUnion_finset_le _ _
      _ ≤ ∑ _v ∈ U, c := Finset.sum_le_sum fun v _ => hc j hj v
      _ = U.card * c := by rw [Finset.sum_const, nsmul_eq_mul]
  have hU2 : ∀ j ∈ pops, (Dsf.prod (popMeasure Dsamp past j)).real E ≤ U.card * c :=
    fun j hj => prod_mul_mem_le Dsf _ hc0 (hc j hj) U
  set A1 : Option (Fin K × R) → Fin P → Set (Trunc S K R M P) := fun j i =>
    (fun v => v.1.2 j i) ⁻¹' U
  set A2 : Option (Fin K × R) → Fin P → Set (Trunc S K R M P) := fun j i =>
    (fun v => v.2 j i) ⁻¹' U
  set A3 : Option (Fin K × R) → Fin P → Fin M → Set (Trunc S K R M P) := fun j i l =>
    (fun v => (v.1.1 l, v.1.2 j i)) ⁻¹' E
  set A4 : Option (Fin K × R) → Fin P → Fin M → Set (Trunc S K R M P) := fun j i l =>
    (fun v => (v.1.1 l, v.2 j i)) ⁻¹' E
  have h1 : ∀ j ∈ pops, ∀ i, L.real (A1 j i) ≤ U.card * c := fun j hj i =>
    (((hB j hj i).comp (measurePreserving_snd.comp measurePreserving_fst)).measureReal_preimage
      hUm.nullMeasurableSet).le.trans (hU1 j hj)
  have h2 : ∀ j ∈ pops, ∀ i, L.real (A2 j i) ≤ U.card * c := fun j hj i =>
    (((hB j hj i).comp measurePreserving_snd).measureReal_preimage
      hUm.nullMeasurableSet).le.trans (hU1 j hj)
  have h3 : ∀ j ∈ pops, ∀ i l, L.real (A3 j i l) ≤ U.card * c := fun j hj i l =>
    ((((measurePreserving_eval _ l).prod (hB j hj i)).comp
      measurePreserving_fst).measureReal_preimage hE.nullMeasurableSet).le.trans (hU2 j hj)
  have h4 : ∀ j ∈ pops, ∀ i l, L.real (A4 j i l) ≤ U.card * c := fun j hj i l =>
    ((((measurePreserving_eval _ l).comp measurePreserving_fst).prod
      (hB j hj i)).measureReal_preimage hE.nullMeasurableSet).le.trans (hU2 j hj)
  have hsub : {v | ¬ Disjoint U (truncReads pops v)}
      ⊆ ⋃ j ∈ pops, ⋃ i : Fin P, (A1 j i ∪ A2 j i ∪ ⋃ l : Fin M, (A3 j i l ∪ A4 j i l)) := by
    intro v hv
    obtain ⟨w, hwU, hwR⟩ := Finset.not_disjoint_iff.1 hv
    obtain ⟨⟨p, x⟩, hpx, rfl⟩ := Finset.mem_image.1 hwR
    obtain ⟨hp, hx⟩ := Finset.mem_product.1 hpx
    obtain ⟨j, hj, hp⟩ := Finset.mem_biUnion.1 hp
    refine Set.mem_biUnion hj ?_
    simp only [Set.mem_iUnion, Set.mem_union]
    rcases Finset.mem_union.1 hp with hp | hp <;>
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hp <;> refine ⟨i, ?_⟩ <;>
      rcases Finset.mem_insert.1 hx with rfl | hx
    · left; left; simpa [A1] using hwU
    · obtain ⟨l, -, rfl⟩ := Finset.mem_image.1 hx
      right; exact ⟨l, Or.inl hwU⟩
    · left; right; simpa [A2] using hwU
    · obtain ⟨l, -, rfl⟩ := Finset.mem_image.1 hx
      right; exact ⟨l, Or.inr hwU⟩
  have hpops : (pops.card : ℝ) ≤ Fintype.card (Option (Fin K × R)) := by
    exact_mod_cast Finset.card_le_univ _
  have hUc : 0 ≤ (U.card : ℝ) * c := by positivity
  calc L.real {v | ¬ Disjoint U (truncReads pops v)}
      ≤ ∑ j ∈ pops, L.real (⋃ i : Fin P, (A1 j i ∪ A2 j i ∪ ⋃ l : Fin M, (A3 j i l ∪ A4 j i l))) :=
        (measureReal_mono hsub (measure_ne_top _ _)).trans (measureReal_biUnion_finset_le _ _)
    _ ≤ ∑ _j ∈ pops, (P * ((2 + 2 * M) * (U.card * c)) : ℝ) := by
        refine Finset.sum_le_sum fun j hj => (measureReal_iUnion_fintype_le _).trans ?_
        calc ∑ i : Fin P, L.real (A1 j i ∪ A2 j i ∪ ⋃ l : Fin M, (A3 j i l ∪ A4 j i l))
            ≤ ∑ _i : Fin P, ((2 + 2 * M) * (U.card * c) : ℝ) := by
              refine Finset.sum_le_sum fun i _ => ?_
              refine (measureReal_union_le _ _).trans ?_
              have hl : L.real (⋃ l : Fin M, (A3 j i l ∪ A4 j i l)) ≤ M * (2 * (U.card * c)) :=
                calc L.real (⋃ l : Fin M, (A3 j i l ∪ A4 j i l))
                    ≤ ∑ l : Fin M, L.real (A3 j i l ∪ A4 j i l) := measureReal_iUnion_fintype_le _
                  _ ≤ ∑ _l : Fin M, (2 * (U.card * c) : ℝ) := Finset.sum_le_sum fun l _ =>
                      (measureReal_union_le _ _).trans (by linarith [h3 j hj i l, h4 j hj i l])
                  _ = M * (2 * (U.card * c)) := by simp
              linarith [measureReal_union_le (μ := L) (A1 j i) (A2 j i), h1 j hj i, h2 j hj i]
          _ = P * ((2 + 2 * M) * (U.card * c)) := by simp
    _ = pops.card * (P * ((2 + 2 * M) * (U.card * c))) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ Fintype.card (Option (Fin K × R)) * (P * ((2 + 2 * M) * (U.card * c))) :=
        mul_le_mul_of_nonneg_right hpops (by positivity)
    _ = _ := by ring

/-- One round's clustering, on a pinned cell: it fails `QualityGood` no more often than i.i.d.
streams under fresh noise do, beyond a population short of hits and a pinned string read. -/
theorem cluster_cell_le (A : DFA S Q) (O : Oracle μ S) (Dsamp Dsf : Measure S)
    [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] (past : List (DFA S R × Finset R))
    (states : Finset State) (il α tolerance εcov : ℝ) {M P : ℕ} (hM : 1 ≤ M) (hPM : P ≤ M)
    (hst : ∀ B ∈ states, B.npref ≤ P ∧ B.nsuff ≤ M) {ε κ δc : ℝ} (hε : 0 < ε) (hκ : 0 ≤ κ)
    (hheavy : ∀ (i : Fin K) (h : R) o, past[i.val]? = some o → h ∈ o.2 →
      ε / Fintype.card R ≤ Dsamp.real {v | o.1.state v = h})
    (hP : (P : ℝ) ≤ M * ε / Fintype.card R)
    (hatom : ∀ j ∈ populationsAfter (K := K) past, ∀ a,
      (popMeasure Dsamp past j).real {a} ≤ κ * Fintype.card R / ε)
    (hbad : (runMeasure μ (popMeasure (K := K) Dsamp past) Dsf).real
      {x | ¬ QualityGood A O (populationsAfter past) (popMeasure Dsamp past) states il α
        tolerance εcov x} ≤ δc)
    (U : Finset S) (b : S → ℝ) :
    ((μ.restrict (pinned O U b)).prod (clusterLaw (K := K) Dsamp Dsf M))
        {q | ¬ QualityGood A O (populationsAfter past) (popMeasure Dsamp past) states il α
          tolerance εcov ((q.1, clusterDraws past q.2) : Run Ω S (Option (Fin K × R)))}
      ≤ μ (pinned O U b) * ENNReal.ofReal (δc
        + 2 * Fintype.card (Option (Fin K × R))
          * Real.exp (-2 * (M * ε / Fintype.card R - P) ^ 2 / M)
        + 2 * Fintype.card (Option (Fin K × R)) * M * (M + 1) * U.card
          * (κ * Fintype.card R / ε)) := by
  classical
  have := isProbabilityMeasure_popMeasure (K := K) Dsamp past
  have := isProbabilityMeasure_truncLaw (K := K) Dsamp Dsf M P past
  set pops := populationsAfter (K := K) past
  set D := popMeasure (K := K) Dsamp past
  set Good : Run Ω S (Option (Fin K × R)) → Prop :=
    QualityGood A O pops D states il α tolerance εcov
  set RV : Trunc S K R M P → Finset S := truncReads pops
  set G : Set (Ω × Trunc S K R M P) :=
    {q | q.1 ∈ noiseClean O (RV q.2) ∧ ¬ Good (q.1, untrunc q.2)}
  set c := κ * Fintype.card R / ε
  have hc0 : 0 ≤ c := by positivity
  have hδ0 : 0 ≤ δc := le_trans measureReal_nonneg hbad
  have hpos : ∀ (i : Fin K) (h : R) o, past[i.val]? = some o → h ∈ o.2 →
      Dsamp {v | o.1.state v = h} ≠ 0 := by
    intro i h o ho hh h0
    have hR : (0 : ℝ) < Fintype.card R := by exact_mod_cast Fintype.card_pos_iff.2 ⟨h⟩
    have := hheavy i h o ho hh
    rw [measureReal_def, h0, ENNReal.toReal_zero] at this
    exact absurd this (not_le.2 (div_pos hε hR))
  have hGv : ∀ v, MeasurableSet[noiseAlg O ↑(RV v)] ((fun ω => (ω, v)) ⁻¹' G) := fun v => by
    have := measurableSet_inter_noiseClean O (RV v) {ω | ¬ Good (ω, untrunc v)}
      fun ω ω' h => not_congr (good_noise A O D pops states il α tolerance εcov hst v h)
    convert this using 1
    ext ω
    simp only [G, Set.mem_preimage, Set.mem_ofPred_eq, Set.mem_inter_iff]
    exact and_comm
  have hG : MeasurableSet G := by
    have : G = ⋃ v, ((fun ω => (ω, v)) ⁻¹' G) ×ˢ {v} := by
      ext ⟨ω, v⟩
      simp only [Set.mem_iUnion, Set.mem_prod, Set.mem_preimage, Set.mem_singleton_iff]
      exact ⟨fun h => ⟨v, h, rfl⟩, fun ⟨v', h, hv⟩ => hv ▸ h⟩
    rw [this]
    exact MeasurableSet.iUnion fun v =>
      (noiseAlg_le O _ _ (hGv v)).prod (measurableSet_singleton v)
  set N0 := {ω : Ω | ∃ s, ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)}
  have hN0 : μ N0 = 0 := by
    have : N0 = ⋃ s, {ω | ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)} := by
      ext ω; simp only [N0, Set.mem_ofPred_eq, Set.mem_iUnion]
    rw [this]
    exact measure_iUnion_null fun s => ae_iff.1 (O.noise_bit s)
  have hsub : {q : Ω × ClusterPart S K R M | ¬ Good (q.1, clusterDraws past q.2)}
      ⊆ Prod.map id (truncL P past) ⁻¹' G ∪ N0 ×ˢ Set.univ := by
    rintro ⟨ω, e⟩ hq
    by_cases h0 : ω ∈ N0
    · exact Or.inr ⟨h0, trivial⟩
    · left
      refine ⟨fun s _ => ?_, ?_⟩
      · by_contra hc; exact h0 ⟨s, hc⟩
      · exact (not_congr (good_truncL A O D states il α tolerance εcov hst past ω e)).1 hq
  have hcouple := prod_preimage_le (μ.restrict (pinned O U b)) (clusterLaw (K := K) Dsamp Dsf M)
    (truncLaw Dsamp Dsf M P past) (truncL P past) (enoughSet P past)
    (fun s => clusterLaw_fiber_le Dsamp Dsf hPM past hpos s) hG
  have hpin := pinned_prod_le O U b (truncLaw (K := K) Dsamp Dsf M P past) RV hG hGv
  have hrun : (μ.prod (truncLaw (K := K) Dsamp Dsf M P past)) G ≤ ENNReal.ofReal δc := by
    have : IsProbabilityMeasure (runDraws D Dsf) := by unfold runDraws; infer_instance
    have hmp := (MeasurePreserving.id μ).prod (measurePreserving_truncR (K := K) Dsamp Dsf M P past)
    rw [← hmp.measure_preimage hG.nullMeasurableSet]
    have hb' : runMeasure μ D Dsf {x | ¬ Good x} ≤ ENNReal.ofReal δc :=
      (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _) hδ0).2 hbad
    refine le_trans (measure_mono ?_) hb'
    rintro ⟨ω, xd⟩ ⟨-, hq⟩
    exact (not_congr (good_truncR A O D pops states il α tolerance εcov hst ω xd)).2 hq
  have hhit : truncLaw (K := K) Dsamp Dsf M P past {v | ¬ Disjoint U (RV v)}
      ≤ ENNReal.ofReal (2 * Fintype.card (Option (Fin K × R)) * M * (M + 1) * U.card * c) := by
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal ((truncLaw_hit_le Dsamp Dsf past U hc0 hatom).trans ?_)
    have : (P : ℝ) ≤ M := by exact_mod_cast hPM
    gcongr
  have hshort := clusterLaw_short_le (K := K) Dsamp Dsf hM past hheavy hP
  set tail := Real.exp (-2 * (M * ε / Fintype.card R - P) ^ 2 / M)
  calc ((μ.restrict (pinned O U b)).prod (clusterLaw (K := K) Dsamp Dsf M))
        {q | ¬ Good (q.1, clusterDraws past q.2)}
      ≤ ((μ.restrict (pinned O U b)).prod (clusterLaw (K := K) Dsamp Dsf M))
          (Prod.map id (truncL P past) ⁻¹' G)
        + ((μ.restrict (pinned O U b)).prod (clusterLaw (K := K) Dsamp Dsf M))
          (N0 ×ˢ Set.univ) := (measure_mono hsub).trans (measure_union_le _ _)
    _ ≤ (((μ.restrict (pinned O U b)).prod (truncLaw Dsamp Dsf M P past)) G
          + μ (pinned O U b) * clusterLaw (K := K) Dsamp Dsf M (enoughSet P past)ᶜ) + 0 := by
        gcongr
        · refine hcouple.trans (le_of_eq ?_)
          rw [Measure.restrict_apply_univ]
        · rw [Measure.prod_prod]
          exact le_of_eq (by rw [Measure.restrict_apply' (measurableSet_pinned O U b),
            measure_mono_null Set.inter_subset_left hN0, zero_mul])
    _ ≤ μ (pinned O U b) * (ENNReal.ofReal δc
          + ENNReal.ofReal (2 * Fintype.card (Option (Fin K × R)) * M * (M + 1) * U.card * c))
        + μ (pinned O U b) * ENNReal.ofReal (2 * Fintype.card (Option (Fin K × R)) * tail) := by
        rw [add_zero]
        gcongr
        exact hpin.trans (by gcongr)
    _ = _ := by
        rw [← mul_add, ← ENNReal.ofReal_add hδ0 (by positivity), ← ENNReal.ofReal_add
          (by positivity) (by positivity)]
        congr 2
        ring

end Round

end LearnerProof

end OrthoDFA
