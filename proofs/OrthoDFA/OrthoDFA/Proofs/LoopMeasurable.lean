import OrthoDFA.Proofs.LoopCodes
import OrthoDFA.Proofs.Lloyd

/-!
# A round's outcomes are measurable

An event the oracle's answers at finitely many strings fix is measurable
(`measurableSet_of_reads`). Over given probes, level 5 reads only strings of bounded length, so its
result is fixed by finitely many answers; where it fails its claim also asks how much of the
probes' distribution the round's reads make search or harvest, a countable sum of such events
(`measurableSet_level5Fail`). A round's harvest data are fixed by finitely many answers too, so
the earlier rounds' data are a measurable function of their draws (`measurableSet_codesOf`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The strings a family's reads of `Z` ask the oracle at. -/
def readStrings (Z F : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (Z ×ˢ F).image fun p => p.1 * p.2

lemma measurableSet_prod_countable_right {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    [Countable Y] [MeasurableSingletonClass Y] (E : Set (X × Y))
    (h : ∀ y, MeasurableSet {x | (x, y) ∈ E}) : MeasurableSet E := by
  have hset : E = ⋃ y, {x | (x, y) ∈ E} ×ˢ {y} := by
    ext ⟨x, y⟩
    simp
  rw [hset]
  exact MeasurableSet.iUnion fun y => (h y).prod (measurableSet_singleton y)

section Reads

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- An event the oracle's answers at the strings of `A` fix is measurable. -/
theorem measurableSet_of_reads (O : Oracle μ (FreeMonoid α)) (A : Finset (FreeMonoid α))
    (E : Set Ω) (hE : ∀ ω ω', (∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) → ω ∈ E → ω' ∈ E) :
    MeasurableSet E := by
  classical
  have hset : E = {ω | (fun U => ∃ ω' ∈ E, A.filter (fun p => O.mq p ω' = 1) = U)
      (A.filter (fun p => O.mq p ω = 1))} := by
    ext ω
    constructor
    · intro h; exact ⟨ω, h, rfl⟩
    · rintro ⟨ω', hω', hU⟩
      refine hE ω' ω (fun p hp => ?_) hω'
      have := Finset.ext_iff.1 hU p
      simp only [Finset.mem_filter, hp, true_and] at this
      exact this
  rw [hset]
  refine noiseAlg_le O Set.univ _ ?_
  convert measurableSet_filter_pred O (T := Set.univ) (A := A) (by simp)
    (fun U => ∃ ω' ∈ E, A.filter (fun p => O.mq p ω' = 1) = U)

omit [IsProbabilityMeasure μ] in
theorem familyRead_congr (O : Oracle μ (FreeMonoid α)) {Z F : Finset (FreeMonoid α)} {kl kh : ℕ}
    {ω ω' : Ω} (h : ∀ p ∈ readStrings Z F, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) {z : FreeMonoid α}
    (hz : z ∈ Z) : familyRead O.mq F kl kh z ω = familyRead O.mq F kl kh z ω' := by
  unfold familyRead voteCount
  congr 2
  exact Finset.filter_congr fun v hv =>
    h _ (Finset.mem_image.2 ⟨(z, v), Finset.mem_product.2 ⟨hz, hv⟩, rfl⟩)

/-- The probes' longest. -/
def probesLen {T : ℕ} (xs : Fin T → FreeMonoid α) : ℕ :=
  Finset.univ.sup fun i => (xs i).toList.length

omit [Fintype α] [DecidableEq α] in
theorem le_probesLen {T : ℕ} (xs : Fin T → FreeMonoid α) (i : Fin T) :
    (xs i).toList.length ≤ probesLen xs :=
  Finset.le_sup (f := fun i => (xs i).toList.length) (Finset.mem_univ i)

omit [IsProbabilityMeasure μ] in
/-- Level 5's result, over given probes, as a function of the oracle's answers, is fixed by its
answers at finitely many strings. -/
theorem level5_result_congr (O : Oracle μ (FreeMonoid α)) (C : Level5Settings)
    (hC : 2 ≤ C.maxLeaves) (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ p ∈ readStrings (shortStrings (probesLen xs + 1 + C.maxLeaves)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω) xs
      = indecisiveLStarLevel5 C (fun z => familyRead O.mq F kl kh z ω') xs :=
  level5_congr_up C hC xs (le_probesLen xs)
    (fun _ hz => familyRead_congr O h (mem_shortStrings.2 hz))

/-- A measure's mass on a set the oracle's answers move, where each point's membership is
measurable. -/
theorem measurable_measure_of_mem {X : Type*} [MeasurableSpace X] [Countable X]
    [MeasurableSingletonClass X] (ν : Measure X) (S : Ω → Set X)
    (hS : ∀ x, MeasurableSet {ω | x ∈ S ω}) : Measurable fun ω => ν (S ω) := by
  have hfun : (fun ω => ν (S ω))
      = fun ω => ∑' x, {ω | x ∈ S ω}.indicator (fun _ => ν {x}) ω := by
    funext ω
    rw [← Measure.tsum_indicator_apply_singleton ν (S ω) ((Set.to_countable _).measurableSet)]
    exact tsum_congr fun x => by simp only [Set.indicator]; rfl
  rw [hfun]
  exact Measurable.ennreal_tsum fun x => measurable_const.indicator (hS x)

/-- The midfixes' longest. -/
def midLen (T : DTree α) : ℕ := T.midfixes.sup fun m => m.toList.length

theorem midLe_midLen (T : DTree α) : MidLe (midLen T) T := fun _ hm =>
  Finset.le_sup (f := fun m : FreeMonoid α => m.toList.length) hm

theorem measurableSet_mem_searchingProbes (O : Oracle μ (FreeMonoid α))
    (F : Finset (FreeMonoid α)) (kl kh k : ℕ) (H : HypothesisDFA α) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ searchingProbes (fun z => familyRead O.mq F kl kh z ω) k H} := by
  refine measurableSet_of_reads O
    (readStrings (shortStrings (x.toList.length + midLen H.tree)) F) _ fun ω ω' h hω => ?_
  have hagree : AgreeUpTo (x.toList.length + midLen H.tree)
      (fun z => (familyRead O.mq F kl kh z ω).cut) (fun z => (familyRead O.mq F kl kh z ω').cut) :=
    fun z hz => by dsimp only; rw [familyRead_congr O h (mem_shortStrings.2 hz)]
  simp only [Set.mem_setOf_eq, searchingProbes] at hω ⊢
  rwa [← probeOrAgree_congr_up_rd hagree (midLe_midLen H.tree) le_rfl]

theorem measurableSet_mem_siftedProbes (O : Oracle μ (FreeMonoid α))
    (F : Finset (FreeMonoid α)) (kl kh k : ℕ) (sift : Bool) (H : HypothesisDFA α)
    (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ siftedProbes sift (fun z => familyRead O.mq F kl kh z ω) k H} := by
  refine measurableSet_of_reads O
    (readStrings (shortStrings (x.toList.length + midLen H.tree)) F) _ fun ω ω' h hω => ?_
  have hagree : AgreeUpTo (x.toList.length + midLen H.tree)
      (fun z => (familyRead O.mq F kl kh z ω).cut) (fun z => (familyRead O.mq F kl kh z ω').cut) :=
    fun z hz => by dsimp only; rw [familyRead_congr O h (mem_shortStrings.2 hz)]
  simp only [Set.mem_setOf_eq, siftedProbes, undSiftSet] at hω ⊢
  rwa [← probeOrAgree_congr_up hagree (midLe_midLen H.tree) le_rfl]

lemma length_le_of_mem_shortWords {n : ℕ} {m : FreeMonoid α} (h : m ∈ shortWords n) :
    m.toList.length ≤ n := by
  obtain ⟨⟨⟨i, hi⟩, v⟩, -, rfl⟩ := Finset.mem_image.1 h
  simp only [FreeMonoid.toList_ofList, List.length_ofFn]
  omega

theorem measurableSet_mem_checkedProbes {σ : Type*} (O : Oracle μ (FreeMonoid α))
    (F : Finset (FreeMonoid α)) (kl kh k : ℕ) (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (Dm : Measure (FreeMonoid α)) (n J : ℕ) (ū : ℝ) (H : HypothesisDFA α) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ checkedProbes M dist Dm (fun z => familyRead O.mq F kl kh z ω) k n J ū
      H} := by
  refine measurableSet_of_reads O
    (readStrings (shortStrings (x.toList.length + midLen H.tree + n)) F) _ fun ω ω' h hω => ?_
  have hr : ∀ z : FreeMonoid α, z.toList.length ≤ x.toList.length + midLen H.tree + n →
      familyRead O.mq F kl kh z ω = familyRead O.mq F kl kh z ω' := fun z hz =>
    familyRead_congr O h (mem_shortStrings.2 hz)
  have hagree : AgreeUpTo (x.toList.length + midLen H.tree)
      (fun z => (familyRead O.mq F kl kh z ω).cut) (fun z => (familyRead O.mq F kl kh z ω').cut) :=
    fun z hz => by dsimp only; rw [hr z (by omega)]
  have hpre : ∀ j ≤ x.toList.length, ∀ m ∈ shortWords n,
      familyRead O.mq F kl kh (prefixOf x j * m) ω
        = familyRead O.mq F kl kh (prefixOf x j * m) ω' := fun j hj m hm => by
    refine hr _ ?_
    have := length_le_of_mem_shortWords hm
    simp only [FreeMonoid.toList_mul, List.length_append, prefixOf, FreeMonoid.toList_ofList,
      List.length_take]
    omega
  have hpreM : ∀ j ≤ x.toList.length, ∀ m ∈ H.tree.midfixes,
      familyRead O.mq F kl kh (prefixOf x j * m) ω
        = familyRead O.mq F kl kh (prefixOf x j * m) ω' := fun j hj m hm => by
    refine hr _ ?_
    have := midLe_midLen H.tree m hm
    simp only [FreeMonoid.toList_mul, List.length_append, prefixOf, FreeMonoid.toList_ofList,
      List.length_take]
    omega
  rcases hω with (h1 | ⟨j, hkj, hjx, m, hm, hc⟩) | ⟨hu, hJ, j, hkj, hjx, m, hm, hc⟩
  · refine .inl (.inl ?_)
    simp only [Set.mem_setOf_eq, searchingProbes] at h1 ⊢
    rwa [← probeOrAgree_congr_up_rd hagree (midLe_midLen H.tree) le_rfl]
  · exact .inl (.inr ⟨j, hkj, hjx, m, hm, by beta_reduce at hc ⊢; rw [← hpreM j hjx m hm]; exact hc⟩)
  · exact .inr ⟨hu, hJ, j, hkj, hjx, m, hm, by beta_reduce at hc ⊢; rw [← hpre j hjx m hm]; exact hc⟩

omit [IsProbabilityMeasure μ] in
theorem siteHarvest_congr_reads (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)}
    {kl kh : ℕ} (k : ℕ) (site : Level5Site α) (v : DTree α × Edges α) (x : FreeMonoid α)
    {ω ω' : Ω}
    (h : ∀ p ∈ readStrings (shortStrings (x.toList.length + midLen v.1)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    siteHarvest (fun z => (familyRead O.mq F kl kh z ω).cut) k site v x
      = siteHarvest (fun z => (familyRead O.mq F kl kh z ω').cut) k site v x := by
  have hagree : AgreeUpTo (x.toList.length + midLen v.1)
      (fun z => (familyRead O.mq F kl kh z ω).cut) (fun z => (familyRead O.mq F kl kh z ω').cut) :=
    fun z hz => by dsimp only; rw [familyRead_congr O h (mem_shortStrings.2 hz)]
  cases site with
  | start => exact startHarvestBy_congr_up hagree (midLe_midLen v.1) le_rfl k
  | edge e =>
    change edgeHarvestBy (fun z => (familyRead O.mq F kl kh z ω).cut) v.1 v.2 k x e
      = edgeHarvestBy (fun z => (familyRead O.mq F kl kh z ω').cut) v.1 v.2 k x e
    rw [edgeHarvestBy_congr_pref (prefStrings_agree_up hagree (midLe_midLen v.1) le_rfl)]
  | middles => exact middleHarvestBy_congrK (prefAgreeK_up hagree (midLe_midLen v.1) le_rfl k)

theorem measurable_siteCount (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (site : Level5Site α) (v : DTree α × Edges α) (x : FreeMonoid α)
    (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    Measurable fun ω =>
      ((siteHarvest (fun z => (familyRead O.mq F kl kh z ω).cut) k site v x).filter
        (· ∈ A)).length :=
  measurable_to_countable' fun ω₀ => measurableSet_of_reads O _ _ fun ω ω' h hω => by
    simp only [Set.mem_preimage, Set.mem_singleton_iff] at hω ⊢
    rw [← siteHarvest_congr_reads O k site v x h]
    exact hω

theorem measurable_harvestLaw (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (D : Measure (FreeMonoid α)) (H : HypothesisDFA α) (site : Level5Site α)
    (A : Set (FreeMonoid α)) :
    Measurable fun ω => harvestLaw D (fun z => (familyRead O.mq F kl kh z ω).cut) k H site A := by
  classical
  simp_rw [harvestLaw_apply]
  simp_rw [lintegral_countable']
  exact Measurable.ennreal_tsum fun x =>
    (measurable_from_nat.comp (measurable_siteCount O F kl kh k site _ x A)).mul_const _

end Reads

section Round

variable {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

open scoped Classical in
/-- The state a round's clustering returns at, after the history `h`. -/
def roundState (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) (c : ClusterDraws Ω α K) : State :=
  Classical.epsilon fun B => B ∈ settings.schedule (loopPopulations K h.pops.length)
    ∧ clusterRun D Dsf h c ∈ ret O.mq (loopPopulations K h.pops.length) none
      settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B

/-- A round at the state `B` and family `F`, its clustering having read the prefixes `Pr` and the
suffixes `Sf`, on level 5's noise and probes `d`. -/
def roundAt (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (x₀ : Run Ω (FreeMonoid α) (Option (Fin K))) (B : State) (F Pr Sf : Finset (FreeMonoid α))
    (d : Ω × (Fin settings.level5.probeBudget → FreeMonoid α)) : LoopRound Ω α K :=
  ⟨x₀, B, F, fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1,
    indecisiveLStarLevel5 settings.level5 (fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1) d.2,
    Pr, Sf⟩

/-- Level 5's probes, drawn by uniforms from `D` avoiding `K₀`. -/
def probesOf (settings : LoopSettings K) (D : Measure (FreeMonoid α)) (K₀ : Finset (FreeMonoid α))
    (u : Fin settings.level5.probeBudget → unitInterval) :
    Fin settings.level5.probeBudget → FreeMonoid α :=
  fun i => drawFrom (avoidLaw D settings.level5.walkStart K₀) (u i)

/-- A round at the state `B` and family `F`, its probes drawn from `D` avoiding `K₀`, on level 5's
noise and uniforms `d`. -/
def mkRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α))
    (x₀ : Run Ω (FreeMonoid α) (Option (Fin K))) (B : State) (F K₀ Pr Sf : Finset (FreeMonoid α))
    (d : Ω × (Fin settings.level5.probeBudget → unitInterval)) : LoopRound Ω α K :=
  roundAt settings O x₀ B F Pr Sf (d.1, probesOf settings D K₀ d.2)

/-- The first `k` letters of the prefixes a round's clustering reads. -/
def roundKeys (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (c : ClusterDraws Ω α K) : Finset (FreeMonoid α) :=
  clusterKeys (settings.schedule (loopPopulations K h.pops.length))
    (loopPopulations K h.pops.length) settings.level5.walkStart (clusterRun D Dsf h c)

/-- The prefixes and suffixes a round's clustering reads. -/
def roundSets (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (c : ClusterDraws Ω α K) : Finset (FreeMonoid α) × Finset (FreeMonoid α) :=
  (roundPrefs (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) (clusterRun D Dsf h c),
    roundSuffs (settings.schedule (loopPopulations K h.pops.length)) (clusterRun D Dsf h c))

omit [IsProbabilityMeasure μ] in
lemma loopRound_eq_mkRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level5.probeBudget) :
    loopRound settings O D Dsf h d = mkRound settings O D (clusterRun D Dsf h (d.1, d.2.1))
      (roundState settings O D Dsf h (d.1, d.2.1))
      (clusterAt O.mq (loopPopulations K h.pops.length) (clusterRun D Dsf h (d.1, d.2.1))
        (roundState settings O D Dsf h (d.1, d.2.1)))
      (roundKeys settings D Dsf h (d.1, d.2.1)) (roundSets settings D Dsf h (d.1, d.2.1)).1
      (roundSets settings D Dsf h (d.1, d.2.1)).2 (d.1, d.2.2) := rfl


omit [Fintype σ] in
/-- Where a round ending at hypothesis `H` and ending `e` fails its claim is measurable in the
oracle's answers. -/
theorem measurableSet_not_endsWell (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (M : _root_.DFA α σ) (bad : σ → Prop) (dist : σ → ARU → ℝ)
    (D Dm : Measure (FreeMonoid α)) (n J : ℕ) (ū : ℝ) (sift : Bool) (ε y : ℝ)
    (H : HypothesisDFA α) (e : RoundEnd (Level5Site α)) :
    MeasurableSet {ω | ¬ EndsWell M bad D ε
      (checkedProbes M dist Dm (fun z => familyRead O.mq F kl kh z ω) k n J ū H
        ∪ siftedProbes sift (fun z => familyRead O.mq F kl kh z ω) k H) k
      (harvestLaw D (fun z => (familyRead O.mq F kl kh z ω).cut) k H) y e} := by
  classical
  cases e with
  | consistent =>
    have hf : Measurable fun ω =>
        D.real (checkedProbes M dist Dm (fun z => familyRead O.mq F kl kh z ω) k n J ū H
          ∪ siftedProbes sift (fun z => familyRead O.mq F kl kh z ω) k H) :=
      ENNReal.measurable_toReal.comp (measurable_measure_of_mem D _
        fun x => (measurableSet_mem_checkedProbes O F kl kh k M dist Dm n J ū H x).union
          (measurableSet_mem_siftedProbes O F kl kh k sift H x))
    exact (measurableSet_le hf measurable_const).compl
  | harvest h =>
    have hm : ∀ A : Set (FreeMonoid α), Measurable fun ω =>
        (harvestLaw D (fun z => (familyRead O.mq F kl kh z ω).cut) k H h).real A := fun A =>
      ENNReal.measurable_toReal.comp (measurable_harvestLaw O F kl kh k D H h A)
    simp only [EndsWell, Spread, Yields]
    exact measurableSet_setOf.2 (Measurable.not
      ((measurableSet_setOf.1 (measurableSet_lt (hm _) ((hm _).const_mul 2))).and
        ((Measurable.forall fun t =>
            measurableSet_setOf.1 (measurableSet_le (hm _) measurable_const)).and
          (measurableSet_setOf.1 (measurableSet_le measurable_const (hm _))))))
  | failed =>
    simp only [EndsWell, not_false_eq_true, Set.setOf_true]
    exact MeasurableSet.univ

end Round

/-! ## The clustering's outcomes -/

section Cluster

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

lemma measurableSet_of_data {X Y : Type*} [MeasurableSpace X] [Countable Y] (d : X → Y)
    (hd : ∀ y, MeasurableSet {x | d x = y}) (R : Y → Set X) (hR : ∀ y, MeasurableSet (R y)) :
    MeasurableSet {x | x ∈ R (d x)} := by
  have hset : {x | x ∈ R (d x)} = ⋃ y, {x | d x = y} ∩ R y := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨y, hy, h⟩ => hy ▸ h⟩
  rw [hset]
  exact MeasurableSet.iUnion fun y => (hd y).inter (hR y)

omit [IsProbabilityMeasure μ] in
lemma voteCount_alike (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)}
    {p : FreeMonoid α} {ω ω' : Ω} (h : ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    voteCount O.mq F p ω = voteCount O.mq F p ω' := by
  unfold voteCount
  rw [Finset.filter_congr fun v hv => h v hv]

omit [IsProbabilityMeasure μ] in
lemma miscutBound_alike (O : Oracle μ (FreeMonoid α)) {a η₀ : ℝ} {lo hi : ℕ}
    {F R P : Finset (FreeMonoid α)} {ω ω' : Ω}
    (hF : ∀ p ∈ P, ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hR : ∀ p ∈ P, ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hs : ∀ p ∈ P, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    miscutBound O.mq a lo hi F R P η₀ ω = miscutBound O.mq a lo hi F R P η₀ ω' := by
  classical
  have hv : ∀ p ∈ P, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_alike O (hF p hp)
  have hacc : acceptSide O.mq hi F P ω = acceptSide O.mq hi F P ω' := by
    unfold acceptSide
    exact Finset.filter_congr fun p hp => by rw [hv p hp]
  have hrej : rejectSide O.mq lo F P ω = rejectSide O.mq lo F P ω' := by
    unfold rejectSide
    exact Finset.filter_congr fun p hp => by rw [hv p hp]
  have hside : ∀ (accept : Bool) (A : Finset (FreeMonoid α)), A ⊆ P →
      sideCap O.mq accept A R a η₀ ω = sideCap O.mq accept A R a η₀ ω' := by
    intro accept A hA
    unfold sideCap
    have h1 : againstOn O.mq accept A ω = againstOn O.mq accept A ω' := by
      unfold againstOn
      congr 1
      exact Finset.filter_congr fun p hp => by rw [hs p (hA hp)]
    have h2 : refApart O.mq A R ω = refApart O.mq A R ω' := by
      unfold refApart
      split_ifs with h
      · refine Finset.inf'_congr h rfl fun v hv => ?_
        congr 1
        exact Finset.filter_congr fun p hp => by rw [hR p (hA hp) v hv, hs p (hA hp)]
      · rfl
    rw [h1, h2]
  unfold miscutBound
  have hA : acceptSide O.mq hi F P ω' ⊆ P := by unfold acceptSide; exact Finset.filter_subset _ _
  have hR' : rejectSide O.mq lo F P ω' ⊆ P := by unfold rejectSide; exact Finset.filter_subset _ _
  rw [hacc, hrej, hside true _ hA, hside false _ hR']

omit [IsProbabilityMeasure μ] in
lemma certifiesAt_alike (O : Oracle μ (FreeMonoid α)) {η₀ lim a : ℝ} {lo hi N : ℕ}
    {F R P : Finset (FreeMonoid α)} {ω ω' : Ω}
    (hF : ∀ p ∈ P, ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hR : ∀ p ∈ P, ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hs : ∀ p ∈ P, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    certifiesAt O.mq η₀ lim a lo hi F R P N ω ↔ certifiesAt O.mq η₀ lim a lo hi F R P N ω' := by
  unfold certifiesAt
  rw [miscutBound_alike O hF hR hs]

omit [IsProbabilityMeasure μ] in
lemma measurableSet_drawsTaken {J : Type*} [Fintype J] (P : Finset (FreeMonoid α)) (j : J) (m N : ℕ) :
    MeasurableSet {x : Run Ω (FreeMonoid α) J |
      drawsTaken P (fun i => certPrefix j i x) m = N} := by
  classical
  have hg : ∀ n, MeasurableSet {x : Run Ω (FreeMonoid α) J |
      m ≤ (((Finset.range n).image (fun i => certPrefix j i x)) \ P).card} := by
    intro n
    have hm : Measurable (fun x : Run Ω (FreeMonoid α) J => (fun i : Fin n => certPrefix j i x)) :=
      measurable_pi_lambda _ (fun i => measurable_cert j i)
    have : {x : Run Ω (FreeMonoid α) J |
        m ≤ (((Finset.range n).image (fun i => certPrefix j i x)) \ P).card}
        = (fun x : Run Ω (FreeMonoid α) J => (fun i : Fin n => certPrefix j i x)) ⁻¹'
          {t : Fin n → FreeMonoid α | m ≤ ((Finset.univ.image t) \ P).card} := by
      ext x
      have : (Finset.range n).image (fun i => certPrefix j i x)
          = Finset.univ.image (fun i : Fin n => certPrefix j i x) := by
        ext s
        simp only [Finset.mem_image, Finset.mem_range, Finset.mem_univ, true_and]
        exact ⟨fun ⟨a, ha, h⟩ => ⟨⟨a, ha⟩, h⟩, fun ⟨a, h⟩ => ⟨a, a.2, h⟩⟩
      simp only [Set.mem_setOf_eq, Set.mem_preimage, this]
    rw [this]
    exact hm (Set.to_countable _).measurableSet
  have key : {x : Run Ω (FreeMonoid α) J | drawsTaken P (fun i => certPrefix j i x) m = N}
      = ({x | m ≤ (((Finset.range N).image (fun i => certPrefix j i x)) \ P).card}
          ∩ ⋂ k ∈ Finset.range N, {x | ¬ m ≤ (((Finset.range k).image
            (fun i => certPrefix j i x)) \ P).card})
        ∪ ({x | N = 0} ∩ ⋂ n : ℕ, {x | ¬ m ≤ (((Finset.range n).image
            (fun i => certPrefix j i x)) \ P).card}) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_union, Set.mem_inter_iff, Set.mem_iInter,
      Finset.mem_range]
    unfold drawsTaken
    split_ifs with h
    · rw [Nat.find_eq_iff]
      constructor
      · rintro ⟨a, b⟩
        exact Or.inl ⟨a, fun k hk => b k hk⟩
      · rintro (⟨a, b⟩ | ⟨-, h'⟩)
        · exact ⟨a, fun k hk => b k hk⟩
        · exact absurd h (not_exists.2 h')
    · push Not at h
      constructor
      · rintro rfl
        exact Or.inr ⟨rfl, fun n => not_le.2 (h n)⟩
      · rintro (⟨a, -⟩ | ⟨h0, -⟩)
        · exact absurd a (not_le.2 (h N))
        · exact h0.symm
  rw [key]
  refine MeasurableSet.union (MeasurableSet.inter (hg N)
    (MeasurableSet.biInter (Finset.range N).countable_toSet fun k _ => (hg k).compl)) ?_
  by_cases hN : N = 0
  · simp only [hN, Set.setOf_true, Set.univ_inter]
    exact MeasurableSet.iInter fun n => (hg n).compl
  · simp only [hN, Set.setOf_false, Set.empty_inter]
    exact MeasurableSet.empty

/-- Where a run returns at the state `B` is measurable. -/
theorem measurableSet_ret (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (uni : Option (Fin K)) (η₀ ι εcov a : ℝ) (B : State) :
    MeasurableSet (ret O.mq pops uni η₀ ι εcov a B : Set (Run Ω (FreeMonoid α) (Option (Fin K)))) := by
  classical
  set cl : Finset (FreeMonoid α) → Finset (FreeMonoid α) → Ω → Finset (FreeMonoid α) :=
    fun P C ω => clusterOf lloydClusterer O B.sc B.scd P C B.k ω
  have hret : (ret O.mq pops uni η₀ ι εcov a B : Set (Run Ω (FreeMonoid α) (Option (Fin K))))
      = {x | B.k ≤ (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x) (oracleNoise x)).card + 1
          ∧ ∀ j ∈ pops, (((certOf (prefixesAt pops B.npref x) j B.npref x).filter fun p => ¬ decided O.mq B.lo (B.hi + 1)
            (insert 1 (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x) (oracleNoise x))) p
              (oracleNoise x)).card : ℝ) ≤ ι * (certOf (prefixesAt pops B.npref x) j B.npref x).card}
        ∩ ⋂ j ∈ pops, ⋃ e : ℕ, ⋃ _ : e ≤ B.npref, {x | certifiesAt O.mq η₀ (gateLimit uni εcov j)
          (gateLevel a pops B) B.lo B.hi
          (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x) (oracleNoise x))
          (poolAt B.nsuff x \ insert 1 (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x)
            (oracleNoise x)))
          (certOf (prefixesAt pops B.npref x) j (B.npref + e) x)
          (drawsTaken (prefixesAt pops B.npref x) (fun i => certPrefix j i x) (B.npref + e))
          (oracleNoise x)} := by
    ext x
    simp only [ret, familyAt, refsAt, clusterAt_eq, clusterAt_eq_clusterOf, cl, Set.mem_setOf_eq,
      Set.mem_inter_iff, Set.mem_iUnion, Set.mem_iInter, exists_prop]
    tauto
  rw [hret]
  refine MeasurableSet.inter ?_ (MeasurableSet.biInter pops.countable_toSet fun j hj =>
    MeasurableSet.iUnion fun e => MeasurableSet.iUnion fun _ => ?_)
  · -- the size and the FNR test, at the table, the pool and the certificates
    refine measurableSet_of_data
      (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
        (prefixesAt pops B.npref x, poolAt B.nsuff x, fun j => certOf (prefixesAt pops B.npref x) j B.npref x))
      (fun y => ?_)
      (fun y => if (1 : FreeMonoid α) ∈ y.2.1 then Prod.fst ⁻¹' {ω | B.k ≤ (cl y.1 y.2.1 ω).card + 1
          ∧ ∀ j ∈ pops, (((y.2.2 j).filter fun p => ¬ decided O.mq B.lo (B.hi + 1)
            (insert 1 (cl y.1 y.2.1 ω)) p ω).card : ℝ) ≤ ι * (y.2.2 j).card} else ∅)
      (fun y => ?_) |> fun h => by
        convert h using 1
        ext x
        simp only [Set.mem_setOf_eq, one_mem_poolAt, if_true, Set.mem_preimage]
        rfl
    · obtain ⟨P, C, cs⟩ := y
      have : {x : Run Ω (FreeMonoid α) (Option (Fin K)) | (prefixesAt pops B.npref x,
          poolAt B.nsuff x, fun j => certOf (prefixesAt pops B.npref x) j B.npref x) = (P, C, cs)}
          = {x | prefixesAt pops B.npref x = P} ∩ ({x | poolAt B.nsuff x = C}
            ∩ ⋂ j, {x | certOf (prefixesAt pops B.npref x) j B.npref x = cs j}) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iInter, Prod.mk.injEq,
          funext_iff]
      rw [this]
      exact (measurableSet_prefixesAt pops B.npref P).inter ((measurableSet_poolAt B.nsuff C).inter
        (MeasurableSet.iInter fun j => measurableSet_certOf pops B.npref j B.npref (cs j)))
    · obtain ⟨P, C, cs⟩ := y
      dsimp only
      split_ifs with hone
      · refine measurable_fst (measurableSet_of_reads O
          (readSet P C ∪ readSet (pops.biUnion cs) (insert 1 C)) _ fun ω ω' h hω => ?_)
        have hcl : cl P C ω = cl P C ω' := clusterOf_congr_mq O B.sc B.scd P C B.k hone
          fun w hw => h w (Finset.mem_union_left _ hw)
        simp only [Set.mem_setOf_eq] at hω ⊢
        rw [← hcl]
        refine ⟨hω.1, fun j hj => ?_⟩
        have hins : ∀ v ∈ insert 1 (cl P C ω), v ∈ insert 1 C := fun v hv => by
          rcases Finset.mem_insert.1 hv with rfl | hv
          · exact Finset.mem_insert_self _ _
          · exact Finset.mem_insert_of_mem (clusterOf_subset O B.sc B.scd P C B.k ω hone hv)
        have hf : (cs j).filter (fun p => ¬ decided O.mq B.lo (B.hi + 1) (insert 1 (cl P C ω)) p ω')
            = (cs j).filter (fun p =>
              ¬ decided O.mq B.lo (B.hi + 1) (insert 1 (cl P C ω)) p ω) :=
          Finset.filter_congr fun p hp => by
            rw [decided, decided, voteCount_alike O (ω := ω') (ω' := ω) fun v hv =>
              (h _ (Finset.mem_union_right _
                (mem_readSet (Finset.mem_biUnion.2 ⟨j, hj, hp⟩) (hins v hv)))).symm]
        rw [hf]
        exact hω.2 j hj
      · exact MeasurableSet.empty
  · -- the gate, at the table, the pool and population `j`'s certificates
    refine measurableSet_of_data
      (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
        (prefixesAt pops B.npref x, poolAt B.nsuff x,
          certOf (prefixesAt pops B.npref x) j (B.npref + e) x,
          drawsTaken (prefixesAt pops B.npref x) (fun i => certPrefix j i x) (B.npref + e)))
      (fun y => ?_)
      (fun y => if (1 : FreeMonoid α) ∈ y.2.1 then Prod.fst ⁻¹'
          {ω | certifiesAt O.mq η₀ (gateLimit uni εcov j) (gateLevel a pops B) B.lo B.hi
            (cl y.1 y.2.1 ω) (y.2.1 \ insert 1 (cl y.1 y.2.1 ω)) y.2.2.1 y.2.2.2 ω} else ∅)
      (fun y => ?_) |> fun h => by
        convert h using 1
        ext x
        simp only [Set.mem_setOf_eq, one_mem_poolAt, if_true, Set.mem_preimage]
        rfl
    · obtain ⟨P, C, Q, N⟩ := y
      have : {x : Run Ω (FreeMonoid α) (Option (Fin K)) | (prefixesAt pops B.npref x,
          poolAt B.nsuff x, certOf (prefixesAt pops B.npref x) j (B.npref + e) x,
          drawsTaken (prefixesAt pops B.npref x) (fun i => certPrefix j i x) (B.npref + e))
            = (P, C, Q, N)}
          = {x | prefixesAt pops B.npref x = P} ∩ ({x | poolAt B.nsuff x = C}
            ∩ ({x | certOf (prefixesAt pops B.npref x) j (B.npref + e) x = Q}
              ∩ {x | drawsTaken P (fun i => certPrefix j i x) (B.npref + e) = N})) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Prod.mk.injEq]
        constructor
        · rintro ⟨h1, h2, h3, h4⟩
          exact ⟨h1, h2, h3, by rw [← h1]; exact h4⟩
        · rintro ⟨h1, h2, h3, h4⟩
          exact ⟨h1, h2, h3, by rw [h1]; exact h4⟩
      rw [this]
      exact (measurableSet_prefixesAt pops B.npref P).inter ((measurableSet_poolAt B.nsuff C).inter
        ((measurableSet_certOf pops B.npref j (B.npref + e) Q).inter
          (measurableSet_drawsTaken P j (B.npref + e) N)))
    · obtain ⟨P, C, Q, N⟩ := y
      dsimp only
      split_ifs with hone
      · refine measurable_fst (measurableSet_of_reads O
          (readSet P C ∪ readSet Q (insert 1 C)) _ fun ω ω' h hω => ?_)
        have hcl : cl P C ω = cl P C ω' := clusterOf_congr_mq O B.sc B.scd P C B.k hone
          fun w hw => h w (Finset.mem_union_left _ hw)
        simp only [Set.mem_setOf_eq] at hω ⊢
        rw [← hcl]
        have hins : ∀ v ∈ insert 1 (cl P C ω), v ∈ insert 1 C := fun v hv => by
          rcases Finset.mem_insert.1 hv with rfl | hv
          · exact Finset.mem_insert_self _ _
          · exact Finset.mem_insert_of_mem (clusterOf_subset O B.sc B.scd P C B.k ω hone hv)
        have hrd : ∀ p ∈ Q, ∀ v ∈ insert 1 C, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1) :=
          fun p hp v hv => h _ (Finset.mem_union_right _ (mem_readSet hp hv))
        exact (certifiesAt_alike O (fun p hp v hv => hrd p hp v (Finset.mem_insert_of_mem
            (clusterOf_subset O B.sc B.scd P C B.k ω hone hv)))
          (fun p hp v hv => hrd p hp v (Finset.mem_insert_of_mem (Finset.mem_sdiff.1 hv).1))
          (fun p hp => by simpa using hrd p hp 1 (Finset.mem_insert_self 1 C))).1 hω
      · exact MeasurableSet.empty

/-- A run's family at the state `B` is a measurable function of the run. -/
theorem measurableSet_clusterAt (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (B : State) (F : Finset (FreeMonoid α)) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) | clusterAt O.mq pops x B = F} := by
  classical
  refine measurableSet_of_data
    (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) => (prefixesAt pops B.npref x, poolAt B.nsuff x))
    (fun y => (measurableSet_prefixesAt pops B.npref y.1).inter (measurableSet_poolAt B.nsuff y.2)
      |> fun h => by convert h using 1; ext x; simp [Prod.ext_iff])
    (fun y => if (1 : FreeMonoid α) ∈ y.2 then Prod.fst ⁻¹'
        {ω | clusterOf lloydClusterer O B.sc B.scd y.1 y.2 B.k ω = F} else ∅)
    (fun y => by
      split_ifs with hone
      · exact measurable_fst (measurableSet_clusterOf (rule := lloydClusterer) O B.sc B.scd y.1 y.2
          B.k hone F)
      · exact MeasurableSet.empty) |> fun h => by
    convert h using 1
    ext x
    simp only [Set.mem_setOf_eq, one_mem_poolAt, if_true, Set.mem_preimage, clusterAt_eq]
    rfl

end Cluster

/-! ## The rounds' outcomes -/

section Rounds

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

/-- After the history `h`, the state a round's clustering returns at and its family are a
measurable function of the round's clustering draws. -/
theorem measurableSet_roundPick (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) (B : State) (F : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K | roundState settings O D Dsf h c = B
      ∧ clusterAt O.mq (loopPopulations K h.pops.length) (clusterRun D Dsf h c)
        (roundState settings O D Dsf h c) = F} := by
  classical
  have hrun := measurable_clusterRun (Ω := Ω) (K := K) D Dsf h
  set pops := loopPopulations K h.pops.length
  set sched := settings.schedule pops
  set Sc : ClusterDraws Ω α K → Finset State := fun c => sched.filter fun B' =>
    clusterRun D Dsf h c ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B'
  have hstate : ∀ c, roundState settings O D Dsf h c = Classical.epsilon (· ∈ Sc c) := by
    intro c
    unfold roundState
    congr 1
    funext B'
    simp only [Sc, Finset.mem_filter]
    rfl
  have hset : {c : ClusterDraws Ω α K | roundState settings O D Dsf h c = B
      ∧ clusterAt O.mq pops (clusterRun D Dsf h c) (roundState settings O D Dsf h c) = F}
      = ⋃ T ∈ sched.powerset, ({c | Sc c = T} ∩ {c | Classical.epsilon (· ∈ T) = B
        ∧ clusterAt O.mq pops (clusterRun D Dsf h c) (Classical.epsilon (· ∈ T)) = F}) := by
    ext c
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_powerset,
      exists_prop]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨Sc c, Finset.filter_subset _ _, rfl, ?_⟩
      rw [← hstate]
      exact ⟨h1, h2⟩
    · rintro ⟨T, -, hT, h'⟩
      rw [hstate, hT]
      exact h'
  rw [hset]
  refine Finset.measurableSet_biUnion _ fun T hT => MeasurableSet.inter ?_ ?_
  · have hT' : T ⊆ sched := Finset.mem_powerset.1 hT
    have : {c : ClusterDraws Ω α K | Sc c = T} = ⋂ B' ∈ sched, {c |
        (clusterRun D Dsf h c ∈ ret O.mq pops none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B'
          ↔ B' ∈ T)} := by
      ext c
      simp only [Set.mem_setOf_eq, Set.mem_iInter, Finset.ext_iff, Sc, Finset.mem_filter]
      constructor
      · intro h' B' hB'
        have := h' B'
        tauto
      · intro h' B'
        by_cases hB' : B' ∈ sched
        · have := h' B' hB'
          tauto
        · exact ⟨fun h'' => absurd h''.1 hB', fun h'' => absurd (hT' h'') hB'⟩
    rw [this]
    refine Finset.measurableSet_biInter _ fun B' _ => ?_
    by_cases hB' : B' ∈ T
    · simp only [hB', iff_true]
      exact hrun (measurableSet_ret O pops none _ _ _ _ B')
    · simp only [hB', iff_false]
      exact (hrun (measurableSet_ret O pops none _ _ _ _ B')).compl
  · by_cases hB : Classical.epsilon (· ∈ T) = B
    · simp only [hB, true_and]
      exact hrun (measurableSet_clusterAt O pops B F)
    · simp only [hB, false_and, Set.setOf_false]
      exact MeasurableSet.empty

omit [IsProbabilityMeasure μ] in
/-- The prefixes a run's clustering reads over a schedule are a measurable function of it. -/
lemma measurableSet_roundPrefs (sched : Finset State) (pops : Finset (Option (Fin K)))
    (P : Finset (FreeMonoid α)) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) | roundPrefs sched pops x = P} := by
  classical
  set dat : Run Ω (FreeMonoid α) (Option (Fin K)) →
      ({B // B ∈ sched} → Finset (FreeMonoid α))
        × ((B : {B // B ∈ sched}) → pops → Fin (B.1.npref + 1) → Finset (FreeMonoid α)) :=
    fun x => (fun B => prefixesAt pops B.1.npref x,
      fun B j e => certOf (prefixesAt pops B.1.npref x) j.1 (B.1.npref + e.1) x)
  set G : (({B // B ∈ sched} → Finset (FreeMonoid α))
      × ((B : {B // B ∈ sched}) → pops → Fin (B.1.npref + 1) → Finset (FreeMonoid α)))
      → Finset (FreeMonoid α) :=
    fun y => (Finset.univ : Finset {B // B ∈ sched}).biUnion fun B =>
      y.1 B ∪ (Finset.univ : Finset pops).biUnion fun j =>
        (Finset.univ : Finset (Fin (B.1.npref + 1))).biUnion fun e => y.2 B j e
  have hG : ∀ x, roundPrefs sched pops x = G (dat x) := by
    intro x
    simp only [roundPrefs, G, dat]
    ext p
    simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_union, Subtype.exists,
      Finset.mem_range]
    constructor
    · rintro ⟨B, hB, h⟩
      refine ⟨B, hB, ?_⟩
      rcases h with h | ⟨j, hj, e, he, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨j, hj, ⟨e, he⟩, h⟩
    · rintro ⟨B, hB, h⟩
      refine ⟨B, hB, ?_⟩
      rcases h with h | ⟨j, hj, e, h⟩
      · exact Or.inl h
      · exact Or.inr ⟨j, hj, e.1, e.2, h⟩
  have hdat : ∀ y, MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) | dat x = y} := by
    intro y
    have : {x : Run Ω (FreeMonoid α) (Option (Fin K)) | dat x = y}
        = (⋂ B : {B // B ∈ sched}, {x | prefixesAt pops B.1.npref x = y.1 B})
          ∩ ⋂ B : {B // B ∈ sched}, ⋂ j : pops, ⋂ e : Fin (B.1.npref + 1),
            {x | certOf (prefixesAt pops B.1.npref x) j.1 (B.1.npref + e.1) x = y.2 B j e} := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iInter, dat, Prod.ext_iff,
        funext_iff]
    rw [this]
    exact (MeasurableSet.iInter fun B => measurableSet_prefixesAt pops _ _).inter
      (MeasurableSet.iInter fun B => MeasurableSet.iInter fun j => MeasurableSet.iInter fun e =>
        measurableSet_certOf pops _ _ _ _)
  have := measurableSet_of_data dat hdat (fun y => if G y = P then Set.univ else ∅)
    (fun y => by split_ifs <;> simp)
  convert this using 1
  ext x
  simp only [Set.mem_setOf_eq, hG]
  split_ifs with h
  · exact iff_of_true h (Set.mem_univ x)
  · exact iff_of_false h (Set.notMem_empty x)

omit [IsProbabilityMeasure μ] in
/-- The suffixes a run's clustering reads over a schedule are a measurable function of it. -/
lemma measurableSet_roundSuffs (sched : Finset State) (S : Finset (FreeMonoid α)) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) | roundSuffs sched x = S} := by
  classical
  set dat : Run Ω (FreeMonoid α) (Option (Fin K)) → ({B // B ∈ sched} → Finset (FreeMonoid α)) :=
    fun x B => poolAt B.1.nsuff x
  have hdat : ∀ y, MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) | dat x = y} := by
    intro y
    have : {x : Run Ω (FreeMonoid α) (Option (Fin K)) | dat x = y}
        = ⋂ B : {B // B ∈ sched}, {x | poolAt B.1.nsuff x = y B} := by
      ext x
      simp only [Set.mem_setOf_eq, Set.mem_iInter, dat, funext_iff]
    rw [this]
    exact MeasurableSet.iInter fun B => measurableSet_poolAt _ _
  have := measurableSet_of_data dat hdat
    (fun y => if (Finset.univ : Finset {B // B ∈ sched}).biUnion y = S then Set.univ else ∅)
    (fun y => by split_ifs <;> simp)
  convert this using 1
  ext x
  have e : roundSuffs sched x = (Finset.univ : Finset {B // B ∈ sched}).biUnion (dat x) := by
    simp only [roundSuffs, dat]
    ext v
    simp
  simp only [Set.mem_setOf_eq, e]
  split_ifs with h
  · exact iff_of_true h (Set.mem_univ x)
  · exact iff_of_false h (Set.notMem_empty x)

/-- After the history `h`, a round's keys are a measurable function of its clustering draws. -/
theorem measurableSet_roundKeys (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (h : LoopHist α) (K₀ : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K | roundKeys settings D Dsf h c = K₀} := by
  classical
  have hrun := measurable_clusterRun (Ω := Ω) (K := K) D Dsf h
  have : {c : ClusterDraws Ω α K | roundKeys settings D Dsf h c = K₀}
      = ⋃ P : Finset (FreeMonoid α), clusterRun D Dsf h ⁻¹'
        ({x | roundPrefs (settings.schedule (loopPopulations K h.pops.length))
          (loopPopulations K h.pops.length) x = P}
          ∩ {_x | P.image (fun p => prefixOf p settings.level5.walkStart) = K₀}) := by
    ext c
    simp only [roundKeys, clusterKeys, Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_preimage,
      Set.mem_inter_iff]
    exact ⟨fun h' => ⟨_, rfl, h'⟩, fun ⟨P, hP, h'⟩ => hP ▸ h'⟩
  rw [this]
  exact MeasurableSet.iUnion fun P => hrun ((measurableSet_roundPrefs _ _ P).inter
    (MeasurableSet.const _))

/-- After the history `h`, what a round's clustering reads is a measurable function of its
clustering draws. -/
theorem measurableSet_roundSets (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (h : LoopHist α) (PS : Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K | roundSets settings D Dsf h c = PS} := by
  have hrun := measurable_clusterRun (Ω := Ω) (K := K) D Dsf h
  have : {c : ClusterDraws Ω α K | roundSets settings D Dsf h c = PS}
      = clusterRun D Dsf h ⁻¹' ({x | roundPrefs (settings.schedule (loopPopulations K
          h.pops.length)) (loopPopulations K h.pops.length) x = PS.1}
        ∩ {x | roundSuffs (settings.schedule (loopPopulations K h.pops.length)) x = PS.2}) := by
    ext c
    simp only [roundSets, Set.mem_setOf_eq, Set.mem_preimage, Set.mem_inter_iff, Prod.ext_iff]
  rw [this]
  exact hrun ((measurableSet_roundPrefs _ _ _).inter (measurableSet_roundSuffs _ _))

instance (D : Measure (FreeMonoid α)) (k : ℕ) (K₀ : Finset (FreeMonoid α)) :
    IsProbabilityMeasure (avoidLaw D k K₀) := by
  unfold avoidLaw; infer_instance

theorem measurable_probesOf (settings : LoopSettings K) (D : Measure (FreeMonoid α))
    (K₀ : Finset (FreeMonoid α)) : Measurable (probesOf settings D K₀) :=
  measurable_pi_lambda _ fun i => (measurable_drawFrom _).comp (measurable_pi_apply i)

/-- After the history `h`, a round's harvest data are a measurable function of its draws. -/
theorem measurableSet_roundCode (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (hC : 2 ≤ settings.level5.maxLeaves) (N : ℕ) (D Dsf : Measure (FreeMonoid α))
    (h : LoopHist α) (oc : Option (HarvestCode α)) :
    MeasurableSet {d : LoopRoundDraws Ω α K settings.level5.probeBudget |
      popCode N (loopRound settings O D Dsf h d) = oc} := by
  classical
  set pops := loopPopulations K h.pops.length
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set x₀ : Run Ω (FreeMonoid α) (Option (Fin K)) := (ω₀, ((fun _ => 1, fun _ _ => 1), fun _ _ => 1))
  have hcl : Measurable fun d : LoopRoundDraws Ω α K settings.level5.probeBudget =>
      ((d.1, d.2.1) : ClusterDraws Ω α K) := measurable_fst.prodMk (measurable_fst.comp measurable_snd)
  have hlv : Measurable fun d : LoopRoundDraws Ω α K settings.level5.probeBudget => (d.1, d.2.2) :=
    measurable_fst.prodMk (measurable_snd.comp measurable_snd)
  have hset : {d : LoopRoundDraws Ω α K settings.level5.probeBudget |
      popCode N (loopRound settings O D Dsf h d) = oc}
      = ⋃ BFK : State × Finset (FreeMonoid α) × Finset (FreeMonoid α)
          × (Finset (FreeMonoid α) × Finset (FreeMonoid α)),
        ((fun d : LoopRoundDraws Ω α K settings.level5.probeBudget =>
            ((d.1, d.2.1) : ClusterDraws Ω α K)) ⁻¹'
          ({c | roundState settings O D Dsf h c = BFK.1
            ∧ clusterAt O.mq pops (clusterRun D Dsf h c) (roundState settings O D Dsf h c)
              = BFK.2.1}
            ∩ {c | roundKeys settings D Dsf h c = BFK.2.2.1}
            ∩ {c | roundSets settings D Dsf h c = BFK.2.2.2}))
        ∩ ((fun d : LoopRoundDraws Ω α K settings.level5.probeBudget => (d.1, d.2.2)) ⁻¹'
          ((fun e : Ω × (Fin settings.level5.probeBudget → unitInterval) =>
            (e.1, probesOf settings D BFK.2.2.1 e.2)) ⁻¹'
            {d₂ : Ω × (Fin settings.level5.probeBudget → FreeMonoid α) |
              popCode N (roundAt settings O x₀ BFK.1 BFK.2.1 BFK.2.2.2.1 BFK.2.2.2.2 d₂)
                = oc})) := by
    ext d
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
    constructor
    · intro hd
      refine ⟨(roundState settings O D Dsf h (d.1, d.2.1),
        clusterAt O.mq pops (clusterRun D Dsf h (d.1, d.2.1)) (roundState settings O D Dsf h
          (d.1, d.2.1)), roundKeys settings D Dsf h (d.1, d.2.1),
          roundSets settings D Dsf h (d.1, d.2.1)), ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, ?_⟩
      rw [loopRound_eq_mkRound] at hd
      exact hd
    · rintro ⟨⟨B, F, K₀, PS⟩, ⟨⟨⟨hB, hF⟩, hK⟩, hS⟩, hd⟩
      rw [loopRound_eq_mkRound]
      dsimp only at hB hF hK hS hd
      rw [hF, hB, hK, hS]
      exact hd
  rw [hset]
  refine MeasurableSet.iUnion fun BFK => MeasurableSet.inter
    (hcl (((measurableSet_roundPick settings O D Dsf h BFK.1 BFK.2.1).inter
      (measurableSet_roundKeys settings D Dsf h BFK.2.2.1)).inter
      (measurableSet_roundSets settings D Dsf h BFK.2.2.2)))
    (hlv ((measurable_fst.prodMk ((measurable_probesOf settings D BFK.2.2.1).comp measurable_snd))
      (measurableSet_prod_countable_right _ fun xs => ?_)))
  set A := readStrings (shortStrings (probesLen xs + 1 + settings.level5.maxLeaves)) BFK.2.1
    ∪ readStrings (shortStrings N) BFK.2.1
  refine measurableSet_of_reads O A _ fun ω ω' h' hω => ?_
  simp only [Set.mem_setOf_eq] at hω ⊢
  rw [← hω]
  have h1 : ∀ p ∈ readStrings (shortStrings (probesLen xs + 1 + settings.level5.maxLeaves))
      BFK.2.1, (O.mq p ω' = 1 ↔ O.mq p ω = 1) := fun p hp =>
    (h' p (Finset.mem_union_left _ hp)).symm
  have h2 : ∀ p ∈ readStrings (shortStrings N) BFK.2.1, (O.mq p ω' = 1 ↔ O.mq p ω = 1) :=
    fun p hp => (h' p (Finset.mem_union_right _ hp)).symm
  have hres := level5_result_congr O settings.level5 hC BFK.2.1 BFK.1.lo (BFK.1.hi + 1) xs h1
  have hgraph : readGraph N (fun z => familyRead O.mq BFK.2.1 BFK.1.lo (BFK.1.hi + 1) z ω')
      = readGraph N (fun z => familyRead O.mq BFK.2.1 BFK.1.lo (BFK.1.hi + 1) z ω) := by
    unfold readGraph
    refine Finset.image_congr fun z hz => ?_
    simp only [Prod.mk.injEq, true_and]
    exact familyRead_congr O h2 hz
  unfold popCode roundAt
  dsimp only
  rw [hres, hgraph]

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma codesOf_append_single (N : ℕ) (rounds : List (LoopRound Ω α K)) (o : LoopRound Ω α K) :
    codesOf N (rounds ++ [o]) = codesOf N rounds ++ (popCode N o).toList := by
  unfold codesOf
  rw [List.filterMap_append]
  cases h : popCode N o <;> simp [h]

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma length_codesOf (N : ℕ) (D : Measure (FreeMonoid α)) (k : ℕ)
    (rounds : List (LoopRound Ω α K)) :
    (codesOf N rounds).length = (harvestsOf D k rounds).length := by
  induction rounds with
  | nil => rfl
  | cons o rounds ih =>
    unfold codesOf harvestsOf at ih ⊢
    rw [List.filterMap_cons, List.filterMap_cons]
    rcases h : o.result.ending.harvested with _ | hv
    · have h1 : popCode N o = none := by simp [popCode, h]
      have h2 : o.population D k = none := by simp [LoopRound.population, h]
      rw [h1, h2]; exact ih
    · have h1 : popCode N o = some (hv, (o.result.hypothesis.tree,
          edgeGraph o.result.hypothesis.edges),
          readGraph N o.read, o.prefs, o.suffs) := by simp [popCode, h]
      have h2 : o.population D k
          = some (popLaw (harvestLaw D (fun z => (o.read z).cut) k o.result.hypothesis hv)) := by
        simp [LoopRound.population, h]
      rw [h1, h2]
      simp [ih]

omit [IsProbabilityMeasure μ] in
lemma loopRounds_result (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget) (r : ℕ) :
    ∀ o ∈ loopRounds settings O D Dsf draws r, ∃ T, ∃ xs : Fin T → FreeMonoid α,
      o.result = indecisiveLStarLevel5 settings.level5 o.read xs := by
  intro o ho
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
  obtain ⟨hK, e⟩ := getElem_loopRounds settings O D Dsf draws r i hi
  rw [e]
  exact ⟨_, _, rfl⟩

omit [IsProbabilityMeasure μ] in
lemma codesOf_loopRounds_succ (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) {L N : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level5.maxLeaves) (hN : L + settings.level5.maxLeaves ≤ N)
    (draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget) (r : ℕ) (hrK : r < K) :
    codesOf N (loopRounds settings O D Dsf draws (r + 1))
      = if (codesOf N (loopRounds settings O D Dsf draws r)).length = r then
          codesOf N (loopRounds settings O D Dsf draws r)
            ++ (popCode N (loopRound settings O D Dsf
              (decodeHist D settings.level5.walkStart
                (codesOf N (loopRounds settings O D Dsf draws r))) (draws ⟨r, hrK⟩))).toList
        else codesOf N (loopRounds settings O D Dsf draws r) := by
  have hlenc := length_codesOf N D settings.level5.walkStart (loopRounds settings O D Dsf draws r)
  rw [loopRounds_succ]
  split_ifs with h1 h2 h2
  · rw [codesOf_append_single, decodeHist_codesOf settings.level5 hlen hC hN
      settings.level5.walkStart _ (loopRounds_result settings O D Dsf draws r) h1.2]
  · exact absurd (by rw [hlenc, length_harvestsOf h1.2,
      length_eq_of_allHarvested settings O D Dsf draws hrK.le h1.2]) h2
  · exfalso
    apply h1
    refine ⟨hrK, fun o ho => ?_⟩
    have hle := length_loopRounds_le settings O D Dsf draws r
    have hlen' : (harvestsOf D settings.level5.walkStart
        (loopRounds settings O D Dsf draws r)).length
        = (loopRounds settings O D Dsf draws r).length := by
      have := List.length_filterMap_le (fun o : LoopRound Ω α K =>
        o.population D settings.level5.walkStart) (loopRounds settings O D Dsf draws r)
      unfold harvestsOf at hlenc this ⊢
      omega
    unfold harvestsOf at hlen'
    have := (List.filterMap_length_eq_length.1 hlen') o ho
    rw [Option.isSome_iff_ne_none] at this
    intro hn
    apply this
    simp [LoopRound.population, hn]
  · rfl

/-- The earlier rounds' harvest data are a measurable function of the rounds' draws. -/
theorem measurableSet_codesOf (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) {L N : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level5.maxLeaves) (hN : L + settings.level5.maxLeaves ≤ N) (r : ℕ)
    (c : List (HarvestCode α)) :
    MeasurableSet {draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget |
      codesOf N (loopRounds settings O D Dsf draws r) = c} := by
  classical
  induction r generalizing c with
  | zero =>
    simp only [loopRounds, codesOf, List.filterMap_nil]
    exact MeasurableSet.const _
  | succ r ih =>
    by_cases hrK : r < K
    · have hset : {draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget |
          codesOf N (loopRounds settings O D Dsf draws (r + 1)) = c}
          = ⋃ c', {draws | codesOf N (loopRounds settings O D Dsf draws r) = c'}
            ∩ (fun draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget =>
              draws ⟨r, hrK⟩) ⁻¹' {d | (if c'.length = r then
                c' ++ (popCode N (loopRound settings O D Dsf
                  (decodeHist D settings.level5.walkStart c') d)).toList else c') = c} := by
        ext draws
        simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage,
          codesOf_loopRounds_succ settings O D Dsf hlen hC hN draws r hrK]
        exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨c', hc', h⟩ => hc' ▸ h⟩
      rw [hset]
      refine MeasurableSet.iUnion fun c' => (ih c').inter (measurable_pi_apply _ ?_)
      by_cases hc : c'.length = r
      · simp only [hc, if_true]
        have : {d : LoopRoundDraws Ω α K settings.level5.probeBudget |
            c' ++ (popCode N (loopRound settings O D Dsf
              (decodeHist D settings.level5.walkStart c') d)).toList = c}
            = ⋃ oc : Option (HarvestCode α),
              if c' ++ oc.toList = c then
                {d | popCode N (loopRound settings O D Dsf
                  (decodeHist D settings.level5.walkStart c') d) = oc} else ∅ := by
          ext d
          simp only [Set.mem_setOf_eq, Set.mem_iUnion]
          constructor
          · intro h
            exact ⟨_, by rw [if_pos h]; rfl⟩
          · rintro ⟨oc, ho⟩
            split_ifs at ho with hh
            · rw [Set.mem_setOf_eq.mp ho]; exact hh
            · exact absurd ho (Set.notMem_empty _)
        rw [this]
        refine MeasurableSet.iUnion fun oc => ?_
        split_ifs
        · exact measurableSet_roundCode settings O hC N D Dsf _ oc
        · exact MeasurableSet.empty
      · simp only [hc, if_false]
        exact MeasurableSet.const _
    · have hst : ∀ draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget,
          loopRounds settings O D Dsf draws (r + 1) = loopRounds settings O D Dsf draws r :=
        fun draws => by rw [loopRounds_succ, dif_neg (fun h => hrK h.1)]
      simp only [hst]
      exact ih c

end Rounds

end OrthoDFA
