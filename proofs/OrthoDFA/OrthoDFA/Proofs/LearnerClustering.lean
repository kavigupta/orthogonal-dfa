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

/-! ## A harvest's law -/

section Harvest

/-- `hv` reads the noise only at `rd`, choosing it as it goes. -/
def HarvReads (hv : Harvester S) (rd : S → (S → ℝ) → Finset S) : Prop :=
  ∀ p (f f' : S → ℝ), (∀ t ∈ rd p f, f t = f' t) → rd p f' = rd p f ∧ hv p f' = hv p f

/-- The runs whose every bit is `0` or `1`. -/
def allClean (O : Oracle μ S) : Set Ω := {ω | ∀ s, O.noise s ω = 0 ∨ O.noise s ω = 1}

lemma measurableSet_allClean (O : Oracle μ S) : MeasurableSet (allClean O) := by
  have : allClean O = ⋂ s, O.noise s ⁻¹' ({0, 1} : Set ℝ) := by
    ext ω; simp [allClean]
  rw [this]
  exact MeasurableSet.iInter fun s => O.noise_meas s (by measurability)

lemma measure_allClean_compl (O : Oracle μ S) : μ (allClean O)ᶜ = 0 := by
  have : (allClean O)ᶜ = ⋃ s, {ω | ¬ (O.noise s ω = 0 ∨ O.noise s ω = 1)} := by
    ext ω; simp [allClean]
  rw [this]
  exact measure_iUnion_null fun s => ae_iff.1 (O.noise_bit s)

/-- The clean runs on which `hv`'s verdict on `p` satisfies `Z`, as an event of its reads. -/
def harvSet (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S) (p : S)
    (Z : Option S → Prop) : Set Ω :=
  ⋃ R : Finset S, ({ω | rd p (fun t => O.mq t ω) = R ∧ Z (hv p fun t => O.mq t ω)}
    ∩ noiseClean O R)

lemma measurableSet_harvSet (O : Oracle μ S) {hv : Harvester S} {rd : S → (S → ℝ) → Finset S}
    (hrd : HarvReads hv rd) (p : S) (Z : Option S → Prop) :
    MeasurableSet (harvSet O hv rd p Z) := by
  refine MeasurableSet.iUnion fun R => noiseAlg_le O (↑R) _ ?_
  refine measurableSet_inter_noiseClean O R _ fun ω ω' h => ?_
  have hsym : ∀ ω ω' : Ω, (∀ w ∈ R, O.noise w ω = O.noise w ω') →
      rd p (fun t => O.mq t ω) = R ∧ Z (hv p fun t => O.mq t ω) →
      rd p (fun t => O.mq t ω') = R ∧ Z (hv p fun t => O.mq t ω') := by
    rintro ω ω' h ⟨hR, hZ⟩
    have := hrd p (fun t => O.mq t ω) (fun t => O.mq t ω') fun t ht =>
      mq_congr O (h t (hR ▸ ht))
    exact ⟨this.1.trans hR, this.2 ▸ hZ⟩
  exact ⟨hsym ω ω' h, hsym ω' ω fun w hw => (h w hw).symm⟩

lemma harvSet_subset (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (p : S) (Z : Option S → Prop) :
    harvSet O hv rd p Z ⊆ {ω | Z (hv p fun t => O.mq t ω)} := by
  intro ω hω
  obtain ⟨R, hR, -⟩ := Set.mem_iUnion.1 hω
  exact hR.2

lemma harvSet_clean (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (p : S) (Z : Option S → Prop) :
    {ω | Z (hv p fun t => O.mq t ω)} ∩ allClean O ⊆ harvSet O hv rd p Z := by
  rintro ω ⟨hZ, hc⟩
  exact Set.mem_iUnion.2 ⟨rd p (fun t => O.mq t ω), ⟨rfl, hZ⟩, fun w _ => hc w⟩

/-- The pairs `(p, ω)` whose verdict satisfies `Z`, on clean runs. -/
def harvProd (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (Z : Option S → Prop) : Set (S × Ω) :=
  ⋃ p, {p} ×ˢ harvSet O hv rd p Z

lemma measurableSet_harvProd (O : Oracle μ S) {hv : Harvester S}
    {rd : S → (S → ℝ) → Finset S} (hrd : HarvReads hv rd) (Z : Option S → Prop) :
    MeasurableSet (harvProd O hv rd Z) :=
  MeasurableSet.iUnion fun p => (measurableSet_singleton p).prod (measurableSet_harvSet O hrd p Z)

lemma harvProd_ae_eq (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (Z : Option S → Prop) (ν : Measure S) [SFinite ν] :
    harvProd O hv rd Z =ᵐ[ν.prod μ] {q : S × Ω | Z (hv q.1 fun t => O.mq t q.2)} := by
  refine (ae_eq_set.2 ⟨?_, ?_⟩)
  · refine measure_mono_null (fun q hq => ?_) (measure_empty (μ := ν.prod μ))
    obtain ⟨p, hp⟩ := Set.mem_iUnion.1 hq.1
    obtain ⟨hp1, hp2⟩ := hp
    rw [Set.mem_singleton_iff] at hp1
    rw [← hp1] at hp2
    exact hq.2 (harvSet_subset O hv rd q.1 Z hp2)
  · refine measure_mono_null (t := Set.univ ×ˢ (allClean O)ᶜ) (fun q hq => ?_) ?_
    · refine ⟨trivial, fun hc => hq.2 ?_⟩
      exact Set.mem_iUnion.2 ⟨q.1, rfl, harvSet_clean O hv rd q.1 Z ⟨hq.1, hc⟩⟩
    · rw [Measure.prod_prod, measure_allClean_compl, mul_zero]

lemma harvest_prod_eq (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (Z : Option S → Prop) (ν : Measure S) [SFinite ν] :
    (ν.prod μ) {q : S × Ω | Z (hv q.1 fun t => O.mq t q.2)} = (ν.prod μ) (harvProd O hv rd Z) :=
  (measure_congr (harvProd_ae_eq O hv rd Z ν)).symm

lemma harvestLaw_apply_singleton (O : Oracle μ S) (hv : Harvester S) (Dsamp : Measure S) (a : S) :
    harvestLaw O hv Dsamp {a} = ENNReal.ofReal
      ((Dsamp.prod μ).real {q | hv q.1 (fun t => O.mq t q.2) = some a}
        / harvestYield O hv Dsamp) := by
  classical
  rw [harvestLaw, Measure.sum_apply _ (measurableSet_singleton a)]
  rw [tsum_eq_single a fun b hb => by simp [hb]]
  simp

lemma isProbabilityMeasure_harvestLaw (O : Oracle μ S) (hv : Harvester S)
    {rd : S → (S → ℝ) → Finset S} (hrd : HarvReads hv rd) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] (hy : 0 < harvestYield O hv Dsamp) :
    IsProbabilityMeasure (harvestLaw O hv Dsamp) := by
  classical
  set ν := Dsamp.prod μ
  set E : S → Set (S × Ω) := fun a => {q | hv q.1 (fun t => O.mq t q.2) = some a}
  have hnull : ∀ a, NullMeasurableSet (E a) ν := fun a =>
    (measurableSet_harvProd O hrd (fun o => o = some a)).nullMeasurableSet.congr
      (harvProd_ae_eq O hv rd (fun o => o = some a) Dsamp)
  have hdisj : Pairwise (Function.onFun (AEDisjoint ν) E) := fun a b hab =>
    Disjoint.aedisjoint (Set.disjoint_left.2 fun q ha hb => hab (Option.some.inj (ha.symm.trans hb)))
  have hunion : (⋃ a, E a) = {q : S × Ω | (hv q.1 fun t => O.mq t q.2).isSome} := by
    ext q
    simp only [Set.mem_iUnion, Set.mem_ofPred_eq, E, Option.isSome_iff_exists]
  have hsum : ∑' a, ν (E a) = ENNReal.ofReal (harvestYield O hv Dsamp) := by
    rw [← measure_iUnion₀ hdisj hnull, hunion, harvestYield, ofReal_measureReal]
  constructor
  rw [harvestLaw, Measure.sum_apply _ MeasurableSet.univ]
  simp only [Measure.smul_apply, Measure.dirac_apply_of_mem (Set.mem_univ _), smul_eq_mul,
    mul_one]
  have hterm : ∀ a, ENNReal.ofReal ((ν.real (E a)) / harvestYield O hv Dsamp)
      = ν (E a) / ENNReal.ofReal (harvestYield O hv Dsamp) := fun a => by
    rw [ENNReal.ofReal_div_of_pos hy, ofReal_measureReal]
  simp only [ν, E] at hterm hsum ⊢
  simp_rw [hterm]
  simp_rw [div_eq_mul_inv]
  rw [ENNReal.tsum_mul_right, hsum, ENNReal.mul_inv_cancel (ENNReal.ofReal_pos.2 hy).ne'
    ENNReal.ofReal_ne_top]

omit [IsProbabilityMeasure μ] in
lemma harvestLaw_compl (O : Oracle μ S) (hv : Harvester S) (Dsamp : Measure S) {Pre : Set S}
    (hPre : ∀ p f a, hv p f = some a → a ∈ Pre) : harvestLaw O hv Dsamp Preᶜ = 0 := by
  classical
  rw [harvestLaw, Measure.sum_apply _ (Set.to_countable _).measurableSet]
  refine ENNReal.tsum_eq_zero.2 fun a => ?_
  by_cases ha : a ∈ Pre
  · simp [ha]
  · have : {q : S × Ω | hv q.1 (fun t => O.mq t q.2) = some a} = ∅ :=
      Set.eq_empty_of_forall_notMem fun q hq => ha (hPre _ _ _ hq)
    simp [this]

lemma harvestLaw_atom_le (O : Oracle μ S) (hv : Harvester S)
    {rd : S → (S → ℝ) → Finset S} (hrd : HarvReads hv rd) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] {κa : ℝ} (hκa : ∀ f a, Dsamp.real {p | hv p f = some a} ≤ κa)
    (hy : 0 < harvestYield O hv Dsamp) (a : S) :
    (harvestLaw O hv Dsamp).real {a} ≤ κa / harvestYield O hv Dsamp := by
  have hκa0 : 0 ≤ κa := le_trans measureReal_nonneg (hκa (fun _ => 0) a)
  have hp : (Dsamp.prod μ) {q : S × Ω | hv q.1 (fun t => O.mq t q.2) = some a}
      ≤ ENNReal.ofReal κa := by
    rw [harvest_prod_eq O hv rd (fun o => o = some a) Dsamp,
      Measure.prod_apply_symm (measurableSet_harvProd O hrd _)]
    calc ∫⁻ ω, Dsamp ((fun p => (p, ω)) ⁻¹' harvProd O hv rd (fun o => o = some a)) ∂μ
        ≤ ∫⁻ _ω, ENNReal.ofReal κa ∂μ := lintegral_mono fun ω => by
          refine (measure_mono fun p hp => ?_).trans
            (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _) hκa0 |>.2
              (hκa (fun t => O.mq t ω) a))
          obtain ⟨p', hp'⟩ := Set.mem_iUnion.1 hp
          obtain ⟨h1, h2⟩ := hp'
          rw [Set.mem_singleton_iff] at h1
          subst h1
          exact harvSet_subset O hv rd _ _ h2
      _ = ENNReal.ofReal κa := by rw [lintegral_const, measure_univ, mul_one]
  rw [measureReal_def, harvestLaw_apply_singleton, ENNReal.toReal_ofReal (by positivity)]
  gcongr
  exact ENNReal.toReal_le_of_le_ofReal hκa0 hp

end Harvest

/-! ## One round's clustering -/

section Round

variable {K : ℕ} {R : Type*} {Q : Type*}

/-- The populations' laws after the rounds `past`: the sampler; the strings reaching a state; and
what a harvest keeps, when it keeps anything. -/
noncomputable def popMeasure (O : Oracle μ S) (Dsamp : Measure S) (past : List (Outcome S R)) :
    Pop K R → Measure S
  | none => Dsamp
  | some (i, some h) => match past[i.val]? with
    | some o => reaching o.1 Dsamp h
    | none => Dsamp
  | some (i, none) => match past[i.val]? with
    | some (_, _, _, some hv) =>
      if 0 < harvestYield O hv Dsamp then harvestLaw O hv Dsamp else Dsamp
    | _ => Dsamp

/-- Every harvest in `past` reads the noise only at some `rd`. -/
def PastReads (past : List (Outcome S R)) : Prop :=
  ∀ o ∈ past, ∀ hv, o.2.2.2 = some hv → ∃ rd, HarvReads hv rd

lemma isProbabilityMeasure_popMeasure (O : Oracle μ S) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] (past : List (Outcome S R)) (hpast : PastReads past)
    (j : Pop K R) : IsProbabilityMeasure (popMeasure O Dsamp past j) := by
  rcases j with _ | ⟨i, _ | h⟩
  · simp only [popMeasure]; infer_instance
  · simp only [popMeasure]
    rcases hget : past[i.val]? with _ | ⟨H, g, fl, _ | hv⟩
    · simp only; infer_instance
    · simp only; infer_instance
    · simp only
      split_ifs with hy
      · obtain ⟨rd, hrd⟩ := hpast _ (List.mem_of_getElem? hget) hv rfl
        exact isProbabilityMeasure_harvestLaw O hv hrd Dsamp hy
      · infer_instance
  · simp only [popMeasure]
    cases past[i.val]? with
    | none => simp only; infer_instance
    | some o => exact isProbabilityMeasure_reaching o.1 Dsamp h

variable [Fintype R]

/-- The law of a round's clustering draws. -/
noncomputable def clusterLaw (Dsamp Dsf : Measure S) (M : ℕ) : Measure (ClusterPart S K R M) :=
  (Measure.pi fun _ : Fin M => Dsf).prod
    ((Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin M => Dsamp).prod
      (Measure.pi fun _ : Pop K R => Measure.pi fun _ : Fin M => Dsamp))

instance isProbabilityMeasure_clusterLaw (Dsamp Dsf : Measure S) [IsProbabilityMeasure Dsamp]
    [IsProbabilityMeasure Dsf] (M : ℕ) :
    IsProbabilityMeasure (clusterLaw (K := K) (R := R) Dsamp Dsf M) := by
  unfold clusterLaw; infer_instance

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

/-- One round's clustering, on a pinned cell, over populations whose harvests keep at least
`π₀` of the sampler: it fails `QualityGood` no more often than i.i.d. streams under fresh noise
do, beyond a population short of draws and a read landing where another read did. -/
theorem cluster_cell_le (A : DFA S Q) (O : Oracle μ S) (Dsamp Dsf : Measure S)
    [IsProbabilityMeasure Dsamp] [IsProbabilityMeasure Dsf] (past : List (Outcome S R))
    (rd : Harvester S → S → (S → ℝ) → Finset S)
    (states : Finset State) (il α tolerance εcov : ℝ) {M P Th : ℕ} (hM : 1 ≤ M) (hPM : P ≤ M)
    (hst : ∀ B ∈ states, B.npref ≤ P ∧ B.nsuff ≤ M) {ε κ κh κf π₀ δc : ℝ} (hε : 0 < ε)
    (hκ : 0 ≤ κ) (hP : (P : ℝ) ≤ M * ε / Fintype.card R) (hPπ : (P : ℝ) ≤ M * π₀)
    (hπ₀ : 0 < π₀)
    (hharv : ∀ o ∈ past, ∀ hv, o.2.2.2 = some hv → HarvReads hv (rd hv)
      ∧ π₀ ≤ harvestYield O hv Dsamp
      ∧ (∀ p f, (rd hv p f).card ≤ Th)
      ∧ (∀ f t, Dsamp.real {p | t ∈ rd hv p f} ≤ κh)
      ∧ ∀ p f a, hv p f = some a → a ∉ rd hv p f)
    (hκf : ∀ a, Dsf.real {a} ≤ κf)
    (hatom : ∀ j ∈ populationsAfter (K := K) Dsamp ε past, ∀ a,
      (popMeasure O Dsamp past j).real {a} ≤ κ * Fintype.card R / ε)
    (hbad : (runMeasure μ (popMeasure (K := K) O Dsamp past) Dsf).real
      {x | ¬ QualityGood A O (populationsAfter Dsamp ε past) (popMeasure O Dsamp past) states il α
        tolerance εcov x} ≤ δc)
    (U : Finset S) (b : S → ℝ) :
    ((μ.restrict (pinned O U b)).prod (clusterLaw (K := K) Dsamp Dsf M))
        {q | ¬ QualityGood A O (populationsAfter Dsamp ε past) (popMeasure O Dsamp past) states
          il α tolerance εcov ((q.1, clusterDraws O q.1 past q.2) : Run Ω S (Pop K R))}
      ≤ μ (pinned O U b) * ENNReal.ofReal (δc
        + 2 * Fintype.card (Pop K R) * Real.exp (-2 * (M * ε / Fintype.card R - P) ^ 2 / M)
        + 2 * Fintype.card (Pop K R) * Real.exp (-2 * (M * π₀ - P) ^ 2 / M)
        + 2 * Fintype.card (Pop K R) * M * (M + 1) * U.card * (κ * Fintype.card R / ε)
        + 2 * Fintype.card (Pop K R) * M
          * (U.card + 2 * Fintype.card (Pop K R) * M * (M + 1)
            + 2 * Fintype.card (Pop K R) * M * Th) * κh
        + 2 * Fintype.card (Pop K R) * M * (M + 1) * Th * κf) := by
  sorry

end Round

end LearnerProof

end OrthoDFA
