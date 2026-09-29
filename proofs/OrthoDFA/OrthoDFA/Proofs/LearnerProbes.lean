import OrthoDFA.Proofs.LearnerClustering

/-!
# Probes read under shared noise

A harvest reads the noise at a few strings of each probe, choosing them as it goes.  Probes whose
reads avoid one another, and a block of strings `W`, behave as if each were read with fresh
noise, independently of any event of the noise at `W` (`probe_couple`); they fail to avoid one
another rarely when a harvest's reads are spread over the sampler (`probe_overlap`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

namespace LearnerProof

/-- Every set of outcomes is an event. -/
def optionMeasurableSpace : MeasurableSpace (Option S) := ⊤

attribute [local instance] optionMeasurableSpace

instance : DiscreteMeasurableSpace (Option S) := ⟨fun _ => MeasurableSpace.measurableSet_top⟩

/-! ## Events of adaptively read noise -/

lemma noiseAlg_mono (O : Oracle μ S) {T T' : Set S} (h : T ⊆ T') :
    noiseAlg O T ≤ noiseAlg O T' :=
  iSup₂_le fun w hw => le_iSup₂ (f := fun w (_ : w ∈ T') =>
    MeasurableSpace.comap (O.noise w) inferInstance) w (h hw)

/-- An event read at `ρ ω` is, on clean runs, an event. -/
lemma measurableSet_of_reads (O : Oracle μ S) (ρ : Ω → Finset S) (Φ : Ω → Prop)
    (h : ∀ ω ω', (∀ s ∈ ρ ω, O.noise s ω = O.noise s ω') → ρ ω' = ρ ω ∧ (Φ ω' ↔ Φ ω)) :
    MeasurableSet ({ω | Φ ω} ∩ allClean O) := by
  have hset : {ω | Φ ω} ∩ allClean O
      = (⋃ R : Finset S, ({ω | ρ ω = R ∧ Φ ω} ∩ noiseClean O R)) ∩ allClean O := by
    ext ω
    simp only [Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_iUnion]
    constructor
    · rintro ⟨hΦ, hc⟩
      exact ⟨⟨ρ ω, ⟨rfl, hΦ⟩, fun w _ => hc w⟩, hc⟩
    · rintro ⟨⟨R, ⟨_, hΦ⟩, _⟩, hc⟩
      exact ⟨hΦ, hc⟩
  rw [hset]
  refine (MeasurableSet.iUnion fun R : Finset S => noiseAlg_le O (R : Set S) _ ?_).inter
    (measurableSet_allClean O)
  refine measurableSet_inter_noiseClean O R _ fun ω ω' hω => ?_
  have hsym : ∀ ω ω' : Ω, (∀ w ∈ R, O.noise w ω = O.noise w ω') →
      ρ ω = R ∧ Φ ω → ρ ω' = R ∧ Φ ω' := by
    rintro ω ω' h' ⟨hR, hΦ⟩
    have := h ω ω' fun s hs => h' s (hR ▸ hs)
    exact ⟨this.1.trans hR, this.2.2 hΦ⟩
  exact ⟨hsym ω ω' hω, hsym ω' ω fun w hw => (hω w hw).symm⟩

lemma measure_inter_allClean (O : Oracle μ S) (A : Set Ω) : μ (A ∩ allClean O) = μ A := by
  refine le_antisymm (measure_mono Set.inter_subset_left) ?_
  calc μ A ≤ μ (A ∩ allClean O ∪ (allClean O)ᶜ) := measure_mono fun ω hω => by
        by_cases h : ω ∈ allClean O
        exacts [Or.inl ⟨hω, h⟩, Or.inr h]
    _ ≤ μ (A ∩ allClean O) + μ (allClean O)ᶜ := measure_union_le _ _
    _ = μ (A ∩ allClean O) := by rw [measure_allClean_compl, add_zero]

/-- A product with a countable space, bounded by its sections. -/
lemma prod_le_tsum (μ' : Measure Ω) [SFinite μ'] {X : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] (ν : Measure X) [SFinite ν] (E : Set (Ω × X)) :
    (μ'.prod ν) E ≤ ∑' x, ν {x} * μ' {ω | (ω, x) ∈ E} := by
  calc (μ'.prod ν) E ≤ (μ'.prod ν) (⋃ x, {ω | (ω, x) ∈ E} ×ˢ {x}) :=
        measure_mono fun q hq => Set.mem_iUnion.2 ⟨q.2, hq, rfl⟩
    _ ≤ ∑' x, (μ'.prod ν) ({ω | (ω, x) ∈ E} ×ˢ {x}) := measure_iUnion_le _
    _ = ∑' x, ν {x} * μ' {ω | (ω, x) ∈ E} := by
        congr 1 with x
        rw [Measure.prod_prod, mul_comm]

/-! ## What one probe keeps -/

/-- The chance a probe, read with fresh noise, gives `o`. -/
noncomputable def probeMass (O : Oracle μ S) (Dsamp : Measure S) (hv : Harvester S)
    (o : Option S) : ℝ≥0∞ :=
  (Dsamp.prod μ) {q | hv q.1 (fun t => O.mq t q.2) = o}

/-- What a probe gives, read with fresh noise. -/
noncomputable def probeLaw (O : Oracle μ S) (Dsamp : Measure S) (hv : Harvester S) :
    Measure (Option S) :=
  Measure.sum fun o => probeMass O Dsamp hv o • Measure.dirac o

lemma probeLaw_singleton (O : Oracle μ S) (Dsamp : Measure S) (hv : Harvester S)
    (o : Option S) : probeLaw O Dsamp hv {o} = probeMass O Dsamp hv o := by
  classical
  rw [probeLaw, Measure.sum_apply _ (measurableSet_singleton o)]
  rw [tsum_eq_single o fun b hb => by simp [hb]]
  simp

lemma measure_harvSet (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (p : S) (Z : Option S → Prop) :
    μ (harvSet O hv rd p Z) = μ {ω | Z (hv p fun t => O.mq t ω)} := by
  refine le_antisymm (measure_mono (harvSet_subset O hv rd p Z)) ?_
  rw [← measure_inter_allClean O]
  exact measure_mono (harvSet_clean O hv rd p Z)

omit [IsProbabilityMeasure μ] in
lemma harvProd_section (O : Oracle μ S) (hv : Harvester S) (rd : S → (S → ℝ) → Finset S)
    (Z : Option S → Prop) (a : S) : Prod.mk a ⁻¹' harvProd O hv rd Z = harvSet O hv rd a Z := by
  ext ω
  constructor
  · intro h
    obtain ⟨p, hp⟩ := Set.mem_iUnion.1 h
    obtain ⟨h1, h2⟩ := hp
    rw [Set.mem_singleton_iff] at h1
    rw [← h1] at h2
    exact h2
  · intro h
    exact Set.mem_iUnion.2 ⟨a, rfl, h⟩

lemma tsum_probe (O : Oracle μ S) {hv : Harvester S} {rd : S → (S → ℝ) → Finset S}
    (hrd : HarvReads hv rd) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (o : Option S) :
    ∑' a, Dsamp {a} * μ {ω | hv a (fun t => O.mq t ω) = o} = probeMass O Dsamp hv o := by
  rw [probeMass, harvest_prod_eq O hv rd (fun x => x = o) Dsamp,
    Measure.prod_apply (measurableSet_harvProd O hrd _), lintegral_countable']
  congr 1 with a
  rw [mul_comm]
  congr 1
  rw [harvProd_section, measure_harvSet O hv rd a (fun x => x = o)]

lemma tsum_measure_hv (O : Oracle μ S) {hv : Harvester S} {rd : S → (S → ℝ) → Finset S}
    (hrd : HarvReads hv rd) (a : S) :
    ∑' o : Option S, μ {ω | hv a (fun t => O.mq t ω) = o} = 1 := by
  have h : ∀ o, μ {ω | hv a (fun t => O.mq t ω) = o} = μ (harvSet O hv rd a (fun x => x = o)) :=
    fun o => (measure_harvSet O hv rd a (fun x => x = o)).symm
  simp_rw [h]
  rw [← measure_iUnion ?_ fun o => measurableSet_harvSet O hrd a _]
  · refine le_antisymm prob_le_one ?_
    calc (1 : ℝ≥0∞) = μ Set.univ := measure_univ.symm
      _ = μ (Set.univ ∩ allClean O) := (measure_inter_allClean O _).symm
      _ ≤ _ := measure_mono fun ω hω => Set.mem_iUnion.2 ⟨hv a fun t => O.mq t ω,
          harvSet_clean O hv rd a (fun x => x = hv a fun t => O.mq t ω) ⟨rfl, hω.2⟩⟩
  · intro o o' hoo'
    refine Set.disjoint_left.2 fun ω h1 h2 => hoo' ?_
    exact (harvSet_subset O hv rd a _ h1).symm.trans (harvSet_subset O hv rd a _ h2)

lemma isProbabilityMeasure_probeLaw (O : Oracle μ S) {hv : Harvester S}
    {rd : S → (S → ℝ) → Finset S} (hrd : HarvReads hv rd) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] : IsProbabilityMeasure (probeLaw O Dsamp hv) := by
  constructor
  rw [probeLaw, Measure.sum_apply _ MeasurableSet.univ]
  simp only [Measure.smul_apply, Measure.dirac_apply_of_mem (Set.mem_univ _), smul_eq_mul,
    mul_one]
  simp_rw [← tsum_probe O hrd Dsamp]
  rw [ENNReal.tsum_comm]
  simp_rw [ENNReal.tsum_mul_left, tsum_measure_hv O hrd, mul_one]
  have := Measure.tsum_indicator_apply_singleton Dsamp Set.univ MeasurableSet.univ
  simpa using this


/-! ## Probes -/

/-- Each probe's reads avoid `W` and the reads of the probes before it. -/
def Fresh (O : Oracle μ S) : {n : ℕ} → (Fin n → S → (S → ℝ) → Finset S) → Finset S →
    (Fin n → S) → Ω → Prop
  | 0, _, _, _, _ => True
  | _ + 1, rds, W, g, ω => Disjoint (rds 0 (g 0) fun t => O.mq t ω) W
      ∧ Fresh O (Fin.tail rds) (W ∪ rds 0 (g 0) fun t => O.mq t ω) (Fin.tail g) ω

omit [IsProbabilityMeasure μ] in
lemma out_cons {n : ℕ} (hvs : Fin (n + 1) → Harvester S) (a : S) (g' : Fin n → S) (f : S → ℝ) :
    (fun i => hvs i ((Fin.cons a g' : Fin (n + 1) → S) i) f)
      = Fin.cons (hvs 0 a f) (fun j => Fin.tail hvs j (g' j) f) := by
  funext i
  refine Fin.cases ?_ (fun j => ?_) i
  · simp
  · simp [Fin.tail]

omit [IsProbabilityMeasure μ] in
lemma pi_singleton_cons (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] {n : ℕ} (a : S)
    (g' : Fin n → S) :
    (Measure.pi fun _ : Fin (n + 1) => Dsamp) {(Fin.cons a g' : Fin (n + 1) → S)}
      = Dsamp {a} * (Measure.pi fun _ : Fin n => Dsamp) {g'} := by
  rw [Measure.pi_singleton, Measure.pi_singleton, Fin.prod_univ_succ]
  simp

lemma tsum_fin_succ {n : ℕ} {β : Type*} (f : (Fin (n + 1) → β) → ℝ≥0∞) :
    ∑' g, f g = ∑' a, ∑' g' : Fin n → β, f (Fin.cons a g') := by
  rw [← ENNReal.tsum_prod, ← (Fin.consEquiv fun _ : Fin (n + 1) => β).tsum_eq]
  rfl

lemma pi_apply_cons {n : ℕ} (Φ : Fin (n + 1) → Measure (Option S))
    [∀ i, IsProbabilityMeasure (Φ i)] (T : Set (Fin (n + 1) → Option S)) :
    (Measure.pi Φ) T
      = ∑' o, Φ 0 {o} * (Measure.pi fun j => Φ j.succ) {t' | Fin.cons o t' ∈ T} := by
  classical
  rw [← Measure.tsum_indicator_apply_singleton _ T (Set.to_countable _).measurableSet,
    tsum_fin_succ]
  congr 1 with o
  rw [← Measure.tsum_indicator_apply_singleton (Measure.pi fun j => Φ j.succ)
    {t' | Fin.cons o t' ∈ T} (Set.to_countable _).measurableSet, ← ENNReal.tsum_mul_left]
  congr 1 with t'
  simp only [Set.indicator_apply, Set.mem_ofPred_eq, mul_ite, mul_zero]
  split_ifs
  · rw [Measure.pi_singleton, Measure.pi_singleton, Fin.prod_univ_succ]
    simp
  · rfl

/-- Probes reading fresh noise off `W` give their outcomes as if read with independent fresh
noise, independently of any event of the noise at `W`. -/
theorem probe_couple (O : Oracle μ S) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] :
    ∀ (n : ℕ) (hvs : Fin n → Harvester S) (rds : Fin n → S → (S → ℝ) → Finset S),
      (∀ i, HarvReads (hvs i) (rds i)) → ∀ (W : Finset S) (H : Set Ω),
      MeasurableSet[noiseAlg O ↑W] H → ∀ T : Set (Fin n → Option S),
      ∑' g : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g}
          * μ (H ∩ allClean O ∩ {ω | Fresh O rds W g ω
            ∧ (fun i => hvs i (g i) fun t => O.mq t ω) ∈ T})
        ≤ μ H * (Measure.pi fun i => probeLaw O Dsamp (hvs i)) T := by
  classical
  intro n
  induction n with
  | zero =>
    intro hvs rds hrds W H _ T
    have : ∀ i, IsProbabilityMeasure (probeLaw O Dsamp (hvs i)) := fun i => i.elim0
    rw [tsum_eq_single (default : Fin 0 → S) fun g hg => (hg (Subsingleton.elim _ _)).elim,
      Measure.pi_singleton]
    simp only [Finset.univ_eq_empty, Finset.prod_empty, one_mul]
    by_cases hT : (default : Fin 0 → Option S) ∈ T
    · have h1 : (Measure.pi fun i => probeLaw O Dsamp (hvs i)) T = 1 := by
        refine le_antisymm prob_le_one ?_
        calc (1 : ℝ≥0∞) = (Measure.pi fun i => probeLaw O Dsamp (hvs i)) {default} := by
              rw [Measure.pi_singleton]; simp
          _ ≤ _ := measure_mono (Set.singleton_subset_iff.2 hT)
      rw [h1, mul_one]
      exact measure_mono fun ω hω => hω.1.1
    · refine le_of_eq_of_le ?_ zero_le
      refine measure_mono_null (fun ω hω => ?_) measure_empty
      exact hT (Subsingleton.elim (fun i => hvs i _ _) (default : Fin 0 → Option S) ▸ hω.2.2)
  | succ n ih =>
    intro hvs rds hrds W H hH T
    have hΦ : ∀ i, IsProbabilityMeasure (probeLaw O Dsamp (hvs i)) := fun i =>
      isProbabilityMeasure_probeLaw O (hrds i) Dsamp
    set Tc : Option S → Set (Fin n → Option S) := fun o => {t' | Fin.cons o t' ∈ T}
    set B : S → Finset S → Option S → Set Ω := fun a R o =>
      {ω | rds 0 a (fun t => O.mq t ω) = R ∧ hvs 0 a (fun t => O.mq t ω) = o} ∩ noiseClean O R
    have hB : ∀ a (R : Finset S) o, MeasurableSet[noiseAlg O ↑R] (B a R o) := by
      intro a R o
      refine measurableSet_inter_noiseClean O R _ fun ω ω' h => ?_
      have hsym : ∀ ω ω' : Ω, (∀ w ∈ R, O.noise w ω = O.noise w ω') →
          rds 0 a (fun t => O.mq t ω) = R ∧ hvs 0 a (fun t => O.mq t ω) = o →
          rds 0 a (fun t => O.mq t ω') = R ∧ hvs 0 a (fun t => O.mq t ω') = o := by
        rintro ω ω' h' ⟨hR, ho⟩
        have := hrds 0 a (fun t => O.mq t ω) (fun t => O.mq t ω') fun t ht =>
          mq_congr O (h' t (hR ▸ ht))
        exact ⟨this.1.trans hR, this.2.trans ho⟩
      exact ⟨hsym ω ω' h, hsym ω' ω fun w hw => (h w hw).symm⟩
    -- One probe first, then the rest with its reads added to `W`.
    have hstep : ∀ a, ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
          * μ (H ∩ allClean O ∩ {ω | Fresh O rds W (Fin.cons a g') ω
            ∧ (fun i => hvs i ((Fin.cons a g' : Fin (n + 1) → S) i) fun t => O.mq t ω) ∈ T})
        ≤ μ H * ∑' o, (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o)
          * μ {ω | hvs 0 a (fun t => O.mq t ω) = o} := by
      intro a
      have hcover : ∀ g' : Fin n → S, H ∩ allClean O ∩ {ω | Fresh O rds W (Fin.cons a g') ω
            ∧ (fun i => hvs i ((Fin.cons a g' : Fin (n + 1) → S) i) fun t => O.mq t ω) ∈ T}
          ⊆ ⋃ p : {R : Finset S // Disjoint R W} × Option S,
            (H ∩ B a p.1.1 p.2) ∩ allClean O ∩ {ω | Fresh O (Fin.tail rds) (W ∪ p.1.1) g' ω
              ∧ (fun j => Fin.tail hvs j (g' j) fun t => O.mq t ω) ∈ Tc p.2} := by
        intro g' ω ⟨⟨hωH, hωc⟩, hfr, hT⟩
        simp only [Fresh, Fin.cons_zero, Fin.tail_cons] at hfr
        rw [out_cons] at hT
        refine Set.mem_iUnion.2 ⟨(⟨rds 0 a fun t => O.mq t ω, hfr.1⟩,
          hvs 0 a fun t => O.mq t ω), ⟨⟨hωH, ⟨rfl, rfl⟩, fun w _ => hωc w⟩, hωc⟩, hfr.2, hT⟩
      calc ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
            * μ (H ∩ allClean O ∩ {ω | Fresh O rds W (Fin.cons a g') ω
              ∧ (fun i => hvs i ((Fin.cons a g' : Fin (n + 1) → S) i) fun t => O.mq t ω) ∈ T})
          ≤ ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
            * ∑' p : {R : Finset S // Disjoint R W} × Option S,
              μ ((H ∩ B a p.1.1 p.2) ∩ allClean O ∩ {ω | Fresh O (Fin.tail rds) (W ∪ p.1.1) g' ω
                ∧ (fun j => Fin.tail hvs j (g' j) fun t => O.mq t ω) ∈ Tc p.2}) :=
            ENNReal.tsum_le_tsum fun g' => by
              gcongr
              exact (measure_mono (hcover g')).trans (measure_iUnion_le _)
        _ = ∑' p : {R : Finset S // Disjoint R W} × Option S, ∑' g' : Fin n → S,
              (Measure.pi fun _ : Fin n => Dsamp) {g'}
              * μ ((H ∩ B a p.1.1 p.2) ∩ allClean O ∩ {ω | Fresh O (Fin.tail rds) (W ∪ p.1.1) g' ω
                ∧ (fun j => Fin.tail hvs j (g' j) fun t => O.mq t ω) ∈ Tc p.2}) := by
            simp_rw [← ENNReal.tsum_mul_left]
            exact ENNReal.tsum_comm
        _ ≤ ∑' p : {R : Finset S // Disjoint R W} × Option S,
              μ (H ∩ B a p.1.1 p.2) * (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc p.2) :=
            ENNReal.tsum_le_tsum fun p => by
              refine ih (Fin.tail hvs) (Fin.tail rds) (fun j => hrds j.succ) (W ∪ p.1.1) _ ?_ _
              rw [Finset.coe_union]
              exact (noiseAlg_mono O Set.subset_union_left _ hH).inter
                (noiseAlg_mono O Set.subset_union_right _ (hB a p.1.1 p.2))
        _ = ∑' p : {R : Finset S // Disjoint R W} × Option S,
              μ H * μ (B a p.1.1 p.2) * (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc p.2) := by
            congr 1 with p
            rw [(Indep_iff _ _ μ).1 (indep_noiseAlg O (Finset.disjoint_coe.2 p.1.2.symm)) _ _ hH
              (hB a p.1.1 p.2)]
        _ = μ H * ∑' o, (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o)
              * ∑' R : {R : Finset S // Disjoint R W}, μ (B a R.1 o) := by
            rw [ENNReal.tsum_prod', ENNReal.tsum_comm, ← ENNReal.tsum_mul_left]
            congr 1 with o
            rw [← ENNReal.tsum_mul_left, ← ENNReal.tsum_mul_left]
            congr 1 with R
            ring
        _ ≤ μ H * ∑' o, (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o)
              * μ {ω | hvs 0 a (fun t => O.mq t ω) = o} := by
            gcongr with o
            rw [← measure_iUnion (fun R R' hRR' => Set.disjoint_left.2 fun ω h1 h2 =>
              hRR' (Subtype.ext (h1.1.1.symm.trans h2.1.1)))
              fun R => noiseAlg_le O _ _ (hB a R.1 o)]
            exact measure_mono (Set.iUnion_subset fun R ω hω => hω.1.2)
    rw [tsum_fin_succ, pi_apply_cons]
    calc ∑' a, ∑' g' : Fin n → S, (Measure.pi fun _ : Fin (n + 1) => Dsamp) {Fin.cons a g'}
          * μ (H ∩ allClean O ∩ {ω | Fresh O rds W (Fin.cons a g') ω
            ∧ (fun i => hvs i ((Fin.cons a g' : Fin (n + 1) → S) i) fun t => O.mq t ω) ∈ T})
        = ∑' a, Dsamp {a} * ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
          * μ (H ∩ allClean O ∩ {ω | Fresh O rds W (Fin.cons a g') ω
            ∧ (fun i => hvs i ((Fin.cons a g' : Fin (n + 1) → S) i) fun t => O.mq t ω) ∈ T}) := by
          congr 1 with a
          rw [← ENNReal.tsum_mul_left]
          congr 1 with g'
          rw [pi_singleton_cons, mul_assoc]
      _ ≤ ∑' a, Dsamp {a} * (μ H * ∑' o, (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ))
            (Tc o) * μ {ω | hvs 0 a (fun t => O.mq t ω) = o}) :=
          ENNReal.tsum_le_tsum fun a => by gcongr; exact hstep a
      _ = μ H * ∑' o, probeLaw O Dsamp (hvs 0) {o}
            * (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o) := by
          have e1 : ∀ a, Dsamp {a} * (μ H * ∑' o, (Measure.pi fun j =>
                probeLaw O Dsamp (hvs j.succ)) (Tc o) * μ {ω | hvs 0 a (fun t => O.mq t ω) = o})
              = ∑' o, μ H * ((Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o)
                * (Dsamp {a} * μ {ω | hvs 0 a (fun t => O.mq t ω) = o})) := fun a => by
            rw [← ENNReal.tsum_mul_left, ← ENNReal.tsum_mul_left]
            exact tsum_congr fun o => by ring
          have e2 : ∀ o, μ H * (probeLaw O Dsamp (hvs 0) {o}
                * (Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o))
              = ∑' a, μ H * ((Measure.pi fun j => probeLaw O Dsamp (hvs j.succ)) (Tc o)
                * (Dsamp {a} * μ {ω | hvs 0 a (fun t => O.mq t ω) = o})) := fun o => by
            rw [probeLaw_singleton, ← tsum_probe O (hrds 0) Dsamp, ← ENNReal.tsum_mul_right,
              ← ENNReal.tsum_mul_left]
            exact tsum_congr fun a => by ring
          rw [tsum_congr e1, ENNReal.tsum_comm, ← ENNReal.tsum_mul_left]
          exact (tsum_congr e2).symm


omit [IsProbabilityMeasure μ] in
/-- A probe's reads meet `W` with chance at most `|W|·κh`, whatever the noise. -/
lemma hits_le (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] (rd : S → (S → ℝ) → Finset S)
    {κh : ℝ} (hκh0 : 0 ≤ κh) (hκh : ∀ f t, Dsamp.real {p | t ∈ rd p f} ≤ κh) (W : Finset S)
    (f : S → ℝ) : Dsamp {p | ¬ Disjoint (rd p f) W} ≤ ENNReal.ofReal (W.card * κh) := by
  classical
  calc Dsamp {p | ¬ Disjoint (rd p f) W} ≤ Dsamp (⋃ t ∈ W, {p | t ∈ rd p f}) :=
        measure_mono fun p hp => by
          obtain ⟨t, h1, h2⟩ := Finset.not_disjoint_iff.1 hp
          exact Set.mem_biUnion h2 h1
    _ ≤ ∑ t ∈ W, Dsamp {p | t ∈ rd p f} := measure_biUnion_finset_le _ _
    _ ≤ ∑ _t ∈ W, ENNReal.ofReal κh := Finset.sum_le_sum fun t _ =>
        (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _) hκh0).2 (hκh f t)
    _ = ENNReal.ofReal (W.card * κh) := by
        rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
          ENNReal.ofReal_natCast]

/-- Over the first probe: the chance, on `H`, that its reads meet `W`. -/
lemma tsum_hits_le (O : Oracle μ S) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    {hv : Harvester S} {rd : S → (S → ℝ) → Finset S} (hrd : HarvReads hv rd)
    {κh : ℝ} (hκh0 : 0 ≤ κh) (hκh : ∀ f t, Dsamp.real {p | t ∈ rd p f} ≤ κh) (W : Finset S)
    (H : Set Ω) (hH : MeasurableSet H) :
    ∑' a, Dsamp {a} * μ (H ∩ ({ω | ¬ Disjoint (rd a fun t => O.mq t ω) W} ∩ allClean O))
      ≤ μ H * ENNReal.ofReal (W.card * κh) := by
  classical
  have hM : ∀ a, MeasurableSet ({ω | ¬ Disjoint (rd a fun t => O.mq t ω) W} ∩ allClean O) :=
    fun a => measurableSet_of_reads O (fun ω => rd a fun t => O.mq t ω)
      (fun ω => ¬ Disjoint (rd a fun t => O.mq t ω) W) fun ω ω' h => by
        have := hrd a (fun t => O.mq t ω) (fun t => O.mq t ω') fun t ht => mq_congr O (h t ht)
        exact ⟨this.1, by rw [this.1]⟩
  calc ∑' a, Dsamp {a} * μ (H ∩ ({ω | ¬ Disjoint (rd a fun t => O.mq t ω) W} ∩ allClean O))
      = ∑' a, ∫⁻ ω, Dsamp {a} * (H ∩ ({ω | ¬ Disjoint (rd a fun t => O.mq t ω) W}
          ∩ allClean O)).indicator 1 ω ∂μ := by
        congr 1 with a
        rw [lintegral_const_mul _ (measurable_one.indicator (hH.inter (hM a))),
          lintegral_indicator_one (hH.inter (hM a))]
    _ = ∫⁻ ω, ∑' a, Dsamp {a} * (H ∩ ({ω | ¬ Disjoint (rd a fun t => O.mq t ω) W}
          ∩ allClean O)).indicator 1 ω ∂μ :=
        (lintegral_tsum fun a => ((measurable_one.indicator (hH.inter (hM a))).const_mul
          _).aemeasurable).symm
    _ ≤ ∫⁻ ω, H.indicator (fun _ => ENNReal.ofReal (W.card * κh)) ω ∂μ := by
        refine lintegral_mono fun ω => ?_
        by_cases hω : ω ∈ H
        · rw [Set.indicator_of_mem hω]
          refine le_trans ?_ (hits_le Dsamp rd hκh0 hκh W fun t => O.mq t ω)
          rw [← Measure.tsum_indicator_apply_singleton _ _ (Set.to_countable _).measurableSet]
          refine ENNReal.tsum_le_tsum fun a => ?_
          by_cases ha : ¬ Disjoint (rd a fun t => O.mq t ω) W
          · rw [Set.indicator_of_mem (show a ∈ {p | ¬ Disjoint (rd p fun t => O.mq t ω) W}
              from ha)]
            exact mul_le_of_le_one_right' (Set.indicator_le_self' (fun _ _ => zero_le) ω)
          · rw [Set.indicator_of_notMem (show a ∉ {p | ¬ Disjoint (rd p fun t => O.mq t ω) W}
              from ha), Set.indicator_of_notMem (fun h => ha h.2.1), mul_zero]
        · rw [Set.indicator_of_notMem hω]
          refine le_of_eq (ENNReal.tsum_eq_zero.2 fun a => ?_)
          rw [Set.indicator_of_notMem (fun h => hω h.1), mul_zero]
    _ = μ H * ENNReal.ofReal (W.card * κh) := by
        rw [lintegral_indicator_const hH, mul_comm]

/-- Probes whose reads are spread over the sampler rarely read what an earlier probe did, or
a string of `W`. -/
theorem probe_overlap (O : Oracle μ S) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    {Th : ℕ} {κh : ℝ} (hκh0 : 0 ≤ κh) :
    ∀ (n : ℕ) (hvs : Fin n → Harvester S) (rds : Fin n → S → (S → ℝ) → Finset S),
      (∀ i, HarvReads (hvs i) (rds i)) → (∀ i p f, (rds i p f).card ≤ Th) →
      (∀ i f t, Dsamp.real {p | t ∈ rds i p f} ≤ κh) → ∀ (W : Finset S) (H : Set Ω),
      MeasurableSet H →
      ∑' g : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g}
          * μ (H ∩ allClean O ∩ {ω | ¬ Fresh O rds W g ω})
        ≤ μ H * ENNReal.ofReal (n * (W.card + n * Th) * κh) := by
  classical
  intro n
  induction n with
  | zero =>
    intro hvs rds _ _ _ W H _
    refine le_of_eq_of_le (ENNReal.tsum_eq_zero.2 fun g => ?_) zero_le
    rw [show H ∩ allClean O ∩ {ω | ¬ Fresh O rds W g ω} = ∅ from
      Set.eq_empty_of_forall_notMem fun ω hω => hω.2 trivial, measure_empty, mul_zero]
  | succ n ih =>
    intro hvs rds hrds hTh hκh W H hH
    set c := ENNReal.ofReal (n * ((W.card + Th) + n * Th) * κh)
    set B : S → Finset S → Set Ω := fun a R =>
      {ω | rds 0 a (fun t => O.mq t ω) = R} ∩ noiseClean O R
    have hB : ∀ a (R : Finset S), MeasurableSet (B a R) := by
      intro a R
      refine noiseAlg_le O (R : Set S) _ (measurableSet_inter_noiseClean O R _ fun ω ω' h => ?_)
      have hsym : ∀ ω ω' : Ω, (∀ w ∈ R, O.noise w ω = O.noise w ω') →
          rds 0 a (fun t => O.mq t ω) = R → rds 0 a (fun t => O.mq t ω') = R := by
        rintro ω ω' h' hR
        exact (hrds 0 a (fun t => O.mq t ω) (fun t => O.mq t ω') fun t ht =>
          mq_congr O (h' t (hR ▸ ht))).1.trans hR
      exact ⟨hsym ω ω' h, hsym ω' ω fun w hw => (h w hw).symm⟩
    have hstep : ∀ a, ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
          * μ (H ∩ allClean O ∩ {ω | ¬ Fresh O rds W (Fin.cons a g') ω})
        ≤ μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W} ∩ allClean O)) + μ H * c := by
      intro a
      have hcover : ∀ g' : Fin n → S, H ∩ allClean O ∩ {ω | ¬ Fresh O rds W (Fin.cons a g') ω}
          ⊆ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W} ∩ allClean O))
            ∪ ⋃ R : Finset S, (H ∩ B a R) ∩ allClean O
              ∩ {ω | ¬ Fresh O (Fin.tail rds) (W ∪ R) g' ω} := by
        intro g' ω ⟨⟨hωH, hωc⟩, hfr⟩
        simp only [Set.mem_ofPred_eq, Fresh, Fin.cons_zero, Fin.tail_cons, not_and_or] at hfr
        rcases hfr with h | h
        · exact Or.inl ⟨hωH, h, hωc⟩
        · exact Or.inr (Set.mem_iUnion.2 ⟨_, ⟨⟨hωH, rfl, fun w _ => hωc w⟩, hωc⟩, h⟩)
      have hR : ∀ R : Finset S, ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
            * μ ((H ∩ B a R) ∩ allClean O ∩ {ω | ¬ Fresh O (Fin.tail rds) (W ∪ R) g' ω})
          ≤ μ (H ∩ B a R) * c := by
        intro R
        by_cases hRc : R.card ≤ Th
        · refine (ih (Fin.tail hvs) (Fin.tail rds) (fun j => hrds j.succ) (fun j => hTh j.succ)
            (fun j => hκh j.succ) (W ∪ R) _ (hH.inter (hB a R))).trans ?_
          gcongr
          have : ((W ∪ R).card : ℝ) ≤ W.card + Th := by
            have h1 := Finset.card_union_le W R
            have h2 : ((W ∪ R).card : ℝ) ≤ W.card + R.card := by exact_mod_cast h1
            have h3 : (R.card : ℝ) ≤ Th := by exact_mod_cast hRc
            linarith
          have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg _
          refine ENNReal.ofReal_le_ofReal ?_
          have h4 := mul_le_mul_of_nonneg_left this hn
          have h5 := mul_le_mul_of_nonneg_right (add_le_add_right h4 ((n : ℝ) * (n * Th))) hκh0
          nlinarith [h5]
        · have : B a R = ∅ := Set.eq_empty_of_forall_notMem fun ω hω =>
            hRc (hω.1 ▸ hTh 0 a _)
          simp [this]
      calc ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
            * μ (H ∩ allClean O ∩ {ω | ¬ Fresh O rds W (Fin.cons a g') ω})
          ≤ ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
            * (μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W} ∩ allClean O))
              + ∑' R : Finset S, μ ((H ∩ B a R) ∩ allClean O
                ∩ {ω | ¬ Fresh O (Fin.tail rds) (W ∪ R) g' ω})) :=
            ENNReal.tsum_le_tsum fun g' => by
              gcongr
              exact (measure_mono (hcover g')).trans ((measure_union_le _ _).trans
                (add_le_add le_rfl (measure_iUnion_le _)))
        _ = μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W} ∩ allClean O))
              + ∑' R : Finset S, ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
                * μ ((H ∩ B a R) ∩ allClean O ∩ {ω | ¬ Fresh O (Fin.tail rds) (W ∪ R) g' ω}) := by
            simp_rw [mul_add, ENNReal.tsum_add, ← ENNReal.tsum_mul_left]
            rw [ENNReal.tsum_mul_right, ENNReal.tsum_comm]
            congr 1
            have := Measure.tsum_indicator_apply_singleton
              (Measure.pi fun _ : Fin n => Dsamp) Set.univ MeasurableSet.univ
            simp only [Set.indicator_univ, measure_univ] at this
            rw [this, one_mul]
        _ ≤ μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W} ∩ allClean O))
              + ∑' R : Finset S, μ (H ∩ B a R) * c := by
            gcongr with R
            exact hR R
        _ ≤ μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W} ∩ allClean O)) + μ H * c := by
            gcongr
            rw [ENNReal.tsum_mul_right, ← measure_iUnion (fun R R' hRR' =>
              Set.disjoint_left.2 fun ω h1 h2 => hRR' (h1.2.1.symm.trans h2.2.1))
              fun R => hH.inter (hB a R)]
            gcongr
            exact Set.iUnion_subset fun R => Set.inter_subset_left
    rw [tsum_fin_succ]
    calc ∑' a, ∑' g' : Fin n → S, (Measure.pi fun _ : Fin (n + 1) => Dsamp) {Fin.cons a g'}
          * μ (H ∩ allClean O ∩ {ω | ¬ Fresh O rds W (Fin.cons a g') ω})
        = ∑' a, Dsamp {a} * ∑' g' : Fin n → S, (Measure.pi fun _ : Fin n => Dsamp) {g'}
          * μ (H ∩ allClean O ∩ {ω | ¬ Fresh O rds W (Fin.cons a g') ω}) := by
          congr 1 with a
          rw [← ENNReal.tsum_mul_left]
          congr 1 with g'
          rw [pi_singleton_cons, mul_assoc]
      _ ≤ ∑' a, Dsamp {a} * (μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W}
            ∩ allClean O)) + μ H * c) := ENNReal.tsum_le_tsum fun a => by gcongr; exact hstep a
      _ = ∑' a, Dsamp {a} * μ (H ∩ ({ω | ¬ Disjoint (rds 0 a fun t => O.mq t ω) W}
            ∩ allClean O)) + μ H * c := by
          simp_rw [mul_add]
          rw [ENNReal.tsum_add, ENNReal.tsum_mul_right]
          congr 1
          have := Measure.tsum_indicator_apply_singleton Dsamp Set.univ MeasurableSet.univ
          simp only [Set.indicator_univ, measure_univ] at this
          rw [this, one_mul]
      _ ≤ μ H * ENNReal.ofReal (W.card * κh) + μ H * c := by
          gcongr
          exact tsum_hits_le O Dsamp (hrds 0) hκh0 (hκh 0) W H hH
      _ ≤ μ H * ENNReal.ofReal ((n + 1 : ℕ) * (W.card + (n + 1 : ℕ) * Th) * κh) := by
          rw [← mul_add, ← ENNReal.ofReal_add (by positivity) (by positivity)]
          gcongr
          push_cast
          have hW : (0 : ℝ) ≤ W.card := Nat.cast_nonneg _
          have hT : (0 : ℝ) ≤ Th := Nat.cast_nonneg _
          have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg _
          nlinarith [mul_nonneg (mul_nonneg hn hT) hκh0, mul_nonneg hT hκh0,
            mul_nonneg (mul_nonneg hn hn) (mul_nonneg hT hκh0), mul_nonneg hW hκh0]


/-! ## The yield test -/

/-- Hoeffding's lower tail for how many of `M` i.i.d. draws land in `W`. -/
lemma pi_count_lower {X : Type*} [MeasurableSpace X] [MeasurableSingletonClass X] [Countable X]
    (Φ : Measure X) [IsProbabilityMeasure Φ] (M : ℕ) (W : Set X) [DecidablePred (· ∈ W)]
    (q t : ℝ) (hq : 0 ≤ q) (ht : 0 ≤ t) (hW : q ≤ Φ.real W) :
    (Measure.pi fun _ : Fin M => Φ).real
        {r : Fin M → X | (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)
          ≤ (M : ℝ) * (q - t)}
      ≤ Real.exp (-2 * (M : ℝ) * t ^ 2) := by
  classical
  set ν : Measure (Fin M → X) := Measure.pi fun _ : Fin M => Φ with hνdef
  have hWm : MeasurableSet W := (Set.to_countable W).measurableSet
  set ind : X → ℝ := W.indicator 1 with hinddef
  have hindm : Measurable ind := measurable_const.indicator hWm
  set Xf : Fin M → (Fin M → X) → ℝ := fun i r => ind (r i) with hXdef
  have hindep : iIndepFun Xf ν := iIndepFun_pi (fun _ => hindm.aemeasurable)
  have hmeas : ∀ i, AEMeasurable (Xf i) ν := fun i =>
    (hindm.comp (measurable_pi_apply _)).aemeasurable
  have hicc : ∀ i, ∀ᵐ r ∂ν, Xf i r ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with r
    by_cases h : r i ∈ W
    · simp [hXdef, hinddef, Set.indicator_of_mem h]
    · simp [hXdef, hinddef, Set.indicator_of_notMem h]
  have hmean : ∀ i, ν[Xf i] = Φ.real W := by
    intro i
    have hmp : MeasurePreserving (fun r : Fin M → X => r i) ν Φ :=
      measurePreserving_eval (fun _ : Fin M => Φ) i
    calc ν[Xf i] = ∫ s, ind s ∂Φ := by
          rw [← hmp.map_eq,
            integral_map hmp.measurable.aemeasurable hindm.aestronglyMeasurable]
      _ = Φ.real W := by rw [hinddef, integral_indicator_one hWm]
  have hsum : ((Finset.univ : Finset (Fin M)).card : ℝ) * q ≤ ∑ i, ν[Xf i] := by
    rw [Finset.sum_congr rfl (fun i _ => hmean i), Finset.sum_const, nsmul_eq_mul]
    have hc : (0 : ℝ) ≤ ((Finset.univ : Finset (Fin M)).card : ℝ) := Nat.cast_nonneg _
    nlinarith [hW]
  have hmain := sumLower_le Xf (Finset.univ : Finset (Fin M)) q t hmeas hindep hicc hsum ht
  have hcount : ∀ r : Fin M → X, ∑ i, Xf i r
      = (((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ) := by
    intro r
    rw [← Finset.sum_filter_add_sum_filter_not (Finset.univ : Finset (Fin M))
      (fun i => r i ∈ W)]
    have h1 : ∑ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W), Xf i r
        = ((((Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W)).card : ℝ)) := by
      have hone : ∀ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => r i ∈ W),
          Xf i r = (1 : ℝ) := by
        intro i hi
        simp [hXdef, hinddef, Set.indicator_of_mem (Finset.mem_filter.1 hi).2]
      rw [Finset.sum_congr rfl hone, Finset.sum_const, nsmul_eq_mul, mul_one]
    have h0 : ∑ i ∈ (Finset.univ : Finset (Fin M)).filter (fun i => ¬ (r i ∈ W)), Xf i r = 0 :=
      Finset.sum_eq_zero (fun i hi => by
        simp [hXdef, hinddef, Set.indicator_of_notMem (Finset.mem_filter.1 hi).2])
    rw [h1, h0, add_zero]
  have hcard : ((Finset.univ : Finset (Fin M)).card : ℝ) = (M : ℝ) := by simp
  refine le_trans (le_trans (measureReal_mono ?_ (measure_ne_top _ _)) hmain) ?_
  · intro r hr
    simp only [Set.mem_ofPred_eq] at hr ⊢
    rw [hcount r, hcard]
    linarith [hr]
  · rw [hcard]

lemma probeLaw_isSome (O : Oracle μ S) {hv : Harvester S} {rd : S → (S → ℝ) → Finset S}
    (hrd : HarvReads hv rd) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp] :
    (probeLaw O Dsamp hv).real {o | o.isSome} = harvestYield O hv Dsamp := by
  classical
  set ν := Dsamp.prod μ
  set E : S → Set (S × Ω) := fun a => {q | hv q.1 (fun t => O.mq t q.2) = some a}
  have hnull : ∀ a, NullMeasurableSet (E a) ν := fun a =>
    (measurableSet_harvProd O hrd (fun o => o = some a)).nullMeasurableSet.congr
      (harvProd_ae_eq O hv rd (fun o => o = some a) Dsamp)
  have hdisj : Pairwise (Function.onFun (AEDisjoint ν) E) := fun a b hab =>
    Disjoint.aedisjoint (Set.disjoint_left.2 fun q ha hb =>
      hab (Option.some.inj (ha.symm.trans hb)))
  have hunion : (⋃ a, E a) = {q : S × Ω | (hv q.1 fun t => O.mq t q.2).isSome} := by
    ext q
    simp only [Set.mem_iUnion, Set.mem_ofPred_eq, E, Option.isSome_iff_exists]
  have hsome : {o : Option S | o.isSome} = ⋃ a : S, {some a} := by
    ext o
    rcases o with _ | a
    · simp
    · exact ⟨fun _ => Set.mem_iUnion.2 ⟨a, rfl⟩, fun _ => rfl⟩
  rw [harvestYield, measureReal_def, measureReal_def, hsome,
    measure_iUnion (fun a b hab => Set.disjoint_singleton.2 fun h => hab (Option.some.inj h))
      fun a => measurableSet_singleton _]
  simp_rw [probeLaw_singleton]
  rw [← hunion, measure_iUnion₀ hdisj hnull]
  rfl

theorem yield_cell_le (O : Oracle μ S) (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]
    (hv : Harvester S) {rd : S → (S → ℝ) → Finset S} (hrd : HarvReads hv rd) {Th : ℕ}
    (hTh : ∀ p f, (rd p f).card ≤ Th) {κh : ℝ} (hκh : ∀ f t, Dsamp.real {p | t ∈ rd p f} ≤ κh)
    (Y yth : ℕ) {π₀ π₁ : ℝ} (h0 : π₀ * Y ≤ yth) (h1 : (yth : ℝ) ≤ π₁ * Y)
    (U : Finset S) (b : S → ℝ) :
    (μ.prod (Measure.pi fun _ : Fin Y => Dsamp))
        ({q | (yth ≤ keptCount O hv q.2 q.1 ∧ harvestYield O hv Dsamp < π₀)
            ∨ (¬ yth ≤ keptCount O hv q.2 q.1 ∧ π₁ ≤ harvestYield O hv Dsamp)}
          ∩ {q | q.1 ∈ pinned O U b})
      ≤ μ (pinned O U b) * ENNReal.ofReal (Real.exp (-2 * Y * (yth / Y - π₀) ^ 2)
        + Real.exp (-2 * Y * (π₁ - yth / Y) ^ 2) + Y * (U.card + Y * Th) * κh) := by
  classical
  have hκh0 : 0 ≤ κh := le_trans measureReal_nonneg (hκh (fun _ => 0) 1)
  set P := pinned O U b
  have hP := measurableSet_pinned O U b
  set y := harvestYield O hv Dsamp
  set eL := Real.exp (-2 * Y * (yth / Y - π₀) ^ 2)
  set eH := Real.exp (-2 * Y * (π₁ - yth / Y) ^ 2)
  set ov := (Y : ℝ) * (U.card + Y * Th) * κh
  have heL : 0 ≤ eL := (Real.exp_pos _).le
  have heH : 0 ≤ eH := (Real.exp_pos _).le
  have hov : 0 ≤ ov := by positivity
  set E := {q : Ω × (Fin Y → S) | (yth ≤ keptCount O hv q.2 q.1 ∧ y < π₀)
      ∨ (¬ yth ≤ keptCount O hv q.2 q.1 ∧ π₁ ≤ y)} ∩ {q | q.1 ∈ P}
  rcases Nat.eq_zero_or_pos Y with hY | hY
  · subst hY
    have : eL = 1 := by simp [eL]
    calc (μ.prod (Measure.pi fun _ : Fin 0 => Dsamp)) E
        ≤ (μ.prod (Measure.pi fun _ : Fin 0 => Dsamp)) (P ×ˢ Set.univ) :=
          measure_mono fun q hq => ⟨hq.2, trivial⟩
      _ = μ P := by rw [Measure.prod_prod, measure_univ, mul_one]
      _ ≤ μ P * ENNReal.ofReal (eL + eH + ov) := by
          refine le_mul_of_one_le_right zero_le ?_
          rw [← ENNReal.ofReal_one]
          exact ENNReal.ofReal_le_ofReal (by linarith)
  have hY' : (0 : ℝ) < Y := by exact_mod_cast hY
  set Φ := probeLaw O Dsamp hv
  have hΦ : IsProbabilityMeasure Φ := isProbabilityMeasure_probeLaw O hrd Dsamp
  set T : Set (Fin Y → Option S) := {t | (yth ≤ (Finset.univ.filter fun i => (t i).isSome).card
      ∧ y < π₀) ∨ (¬ yth ≤ (Finset.univ.filter fun i => (t i).isSome).card ∧ π₁ ≤ y)}
  -- The fresh product's tail.
  have hTail : (Measure.pi fun _ : Fin Y => Φ) T ≤ ENNReal.ofReal (eL + eH) := by
    have hy : Φ.real {o | o.isSome} = y := probeLaw_isSome O hrd Dsamp
    have hyn : Φ.real {o | o.isSome}ᶜ = 1 - y := by
      rw [measureReal_compl (Set.to_countable _).measurableSet, probReal_univ, hy]
    have hy0 : 0 ≤ y := hy ▸ measureReal_nonneg
    have hy1 : y ≤ 1 := hy ▸ measureReal_le_one
    have hcnt : ∀ t : Fin Y → Option S, ((Finset.univ.filter fun i => (t i).isSome).card : ℝ)
        + ((Finset.univ.filter fun i => t i ∈ {o : Option S | o.isSome}ᶜ).card : ℝ) = Y := by
      intro t
      have := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin Y)))
        (fun i => (t i).isSome)
      rw [Finset.card_univ, Fintype.card_fin] at this
      have e : (Finset.univ.filter fun i => t i ∈ {o : Option S | o.isSome}ᶜ)
          = Finset.univ.filter fun i => ¬ (t i).isSome := by
        ext i; simp
      rw [e]
      exact_mod_cast this
    have hlow : y < π₀ → (Measure.pi fun _ : Fin Y => Φ).real
        {t : Fin Y → Option S | yth ≤ (Finset.univ.filter fun i => (t i).isSome).card} ≤ eL := by
      intro hyπ
      have hπY : π₀ ≤ yth / Y := by rw [le_div_iff₀ hY']; linarith
      refine (measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)).trans
        ((pi_count_lower Φ Y {o | o.isSome}ᶜ (1 - y) (yth / Y - y) (by linarith) (by linarith)
          hyn.ge).trans ?_)
      · simp only [Set.mem_ofPred_eq] at ht ⊢
        have h1 := hcnt t
        have h2 : (yth : ℝ) ≤ (Finset.univ.filter fun i => (t i).isSome).card := by
          exact_mod_cast ht
        have : (Y : ℝ) * (1 - y - (yth / Y - y)) = Y - yth := by field_simp; ring
        rw [this]
        linarith
      · refine Real.exp_le_exp.2 ?_
        have ha : 0 ≤ yth / Y - π₀ := by linarith
        have hb : (yth / Y - π₀) ^ 2 ≤ (yth / Y - y) ^ 2 := by nlinarith
        nlinarith
    have hhigh : π₁ ≤ y → (Measure.pi fun _ : Fin Y => Φ).real
        {t : Fin Y → Option S | ¬ yth ≤ (Finset.univ.filter fun i => (t i).isSome).card}
          ≤ eH := by
      intro hyπ
      have hπY : (yth : ℝ) / Y ≤ π₁ := by rw [div_le_iff₀ hY']; linarith
      refine (measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)).trans
        ((pi_count_lower Φ Y {o | o.isSome} y (y - yth / Y) hy0 (by linarith) hy.ge).trans ?_)
      · simp only [Set.mem_ofPred_eq, not_le] at ht ⊢
        have h2 : ((Finset.univ.filter fun i => (t i).isSome).card : ℝ) ≤ yth := by
          exact_mod_cast ht.le
        have : (Y : ℝ) * (y - (y - yth / Y)) = yth := by field_simp; ring
        rw [this]
        exact_mod_cast h2
      · refine Real.exp_le_exp.2 ?_
        have ha : 0 ≤ π₁ - yth / Y := by linarith
        have hb : (π₁ - yth / Y) ^ 2 ≤ (y - yth / Y) ^ 2 := by nlinarith
        nlinarith
    rw [← ofReal_measureReal]
    refine ENNReal.ofReal_le_ofReal ?_
    by_cases hlo : y < π₀
    · by_cases hhi : π₁ ≤ y
      · refine (measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)).trans
          ((measureReal_union_le _ _).trans (add_le_add (hlow hlo) (hhigh hhi)))
        rcases ht with ⟨h, -⟩ | ⟨h, -⟩
        exacts [Or.inl h, Or.inr h]
      · refine (measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)).trans
          ((hlow hlo).trans (by linarith))
        rcases ht with ⟨h, -⟩ | ⟨-, h⟩
        exacts [h, absurd h hhi]
    · by_cases hhi : π₁ ≤ y
      · refine (measureReal_mono (fun t ht => ?_) (measure_ne_top _ _)).trans
          ((hhigh hhi).trans (by linarith))
        rcases ht with ⟨-, h⟩ | ⟨h, -⟩
        exacts [absurd h hlo, h]
      · have : T = ∅ := Set.eq_empty_of_forall_notMem fun t ht => by
          rcases ht with ⟨-, h⟩ | ⟨-, h⟩
          exacts [hlo h, hhi h]
        rw [this, measureReal_empty]
        linarith
  -- The decomposition over the probes.
  have hcpl := probe_couple O Dsamp Y (fun _ => hv) (fun _ => rd) (fun _ => hrd) U P
    (measurableSet_pinned_noiseAlg O U b) T
  have hovl := probe_overlap O Dsamp hκh0 Y (fun _ => hv) (fun _ => rd) (fun _ => hrd)
    (fun _ => hTh) (fun _ => hκh) U P hP
  calc (μ.prod (Measure.pi fun _ : Fin Y => Dsamp)) E
      ≤ ∑' g : Fin Y → S, (Measure.pi fun _ : Fin Y => Dsamp) {g} * μ {ω | (ω, g) ∈ E} :=
        prod_le_tsum μ _ E
    _ ≤ ∑' g : Fin Y → S, (Measure.pi fun _ : Fin Y => Dsamp) {g}
          * (μ (P ∩ allClean O ∩ {ω | ¬ Fresh O (fun _ => rd) U g ω})
            + μ (P ∩ allClean O ∩ {ω | Fresh O (fun _ => rd) U g ω
              ∧ (fun i => hv (g i) fun t => O.mq t ω) ∈ T})) := by
        refine ENNReal.tsum_le_tsum fun g => ?_
        gcongr
        rw [← measure_inter_allClean O]
        refine (measure_mono fun ω hω => ?_).trans (measure_union_le _ _)
        obtain ⟨⟨hE, hωP⟩, hc⟩ := hω
        by_cases hf : Fresh O (fun _ => rd) U g ω
        · exact Or.inr ⟨⟨hωP, hc⟩, hf, hE⟩
        · exact Or.inl ⟨⟨hωP, hc⟩, hf⟩
    _ = ∑' g : Fin Y → S, (Measure.pi fun _ : Fin Y => Dsamp) {g}
          * μ (P ∩ allClean O ∩ {ω | ¬ Fresh O (fun _ => rd) U g ω})
        + ∑' g : Fin Y → S, (Measure.pi fun _ : Fin Y => Dsamp) {g}
          * μ (P ∩ allClean O ∩ {ω | Fresh O (fun _ => rd) U g ω
            ∧ (fun i => hv (g i) fun t => O.mq t ω) ∈ T}) := by
        simp_rw [mul_add]
        exact ENNReal.tsum_add
    _ ≤ μ P * ENNReal.ofReal (Y * (U.card + Y * Th) * κh)
        + μ P * (Measure.pi fun _ : Fin Y => Φ) T := add_le_add hovl hcpl
    _ ≤ μ P * ENNReal.ofReal ov + μ P * ENNReal.ofReal (eL + eH) := by gcongr
    _ = μ P * ENNReal.ofReal (eL + eH + ov) := by
        rw [← mul_add, ← ENNReal.ofReal_add hov (by positivity), add_comm ov]

end LearnerProof

end OrthoDFA
