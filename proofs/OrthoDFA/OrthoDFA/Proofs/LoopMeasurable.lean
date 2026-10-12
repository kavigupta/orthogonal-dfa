import OrthoDFA.Proofs.Level4Reads
import OrthoDFA.Proofs.LoopLaw
import OrthoDFA.Proofs.LoopTermination
import OrthoDFA.Proofs.Lloyd

/-!
# A round's outcomes are measurable

An event the oracle's answers at finitely many strings fix is measurable
(`measurableSet_of_reads`). Over given probes, level 4 reads only strings of bounded length, so its
result is fixed by finitely many answers; where it fails its claim also asks how much of the
probes' and the replay's distributions a read's probes cover, a countable sum of such events
(`measurableSet_level4Fail`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The strings at most `N` long. -/
def shortStrings (N : ℕ) : Finset (FreeMonoid α) :=
  (Finset.range (N + 1)).biUnion fun n =>
    (Finset.univ : Finset (Fin n → α)).image fun f => FreeMonoid.ofList (List.ofFn f)

theorem mem_shortStrings {N : ℕ} {z : FreeMonoid α} :
    z ∈ shortStrings N ↔ z.toList.length ≤ N := by
  constructor
  · intro h
    obtain ⟨n, hn, hz⟩ := Finset.mem_biUnion.1 h
    obtain ⟨f, -, rfl⟩ := Finset.mem_image.1 hz
    simp only [Finset.mem_range] at hn
    simp only [FreeMonoid.toList_ofList, List.length_ofFn]
    omega
  · intro h
    refine Finset.mem_biUnion.2 ⟨z.toList.length, Finset.mem_range.2 (by omega),
      Finset.mem_image.2 ⟨fun i => z.toList.get i, Finset.mem_univ _, ?_⟩⟩
    apply FreeMonoid.toList.injective
    simp

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
/-- Level 4's result, over given probes, as a function of the oracle's answers, is fixed by its
answers at finitely many strings. -/
theorem level4_result_congr (O : Oracle μ (FreeMonoid α)) (C : Level4Settings)
    (hC : 2 ≤ C.maxLeaves) (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ}
    (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ p ∈ readStrings (shortStrings (probesLen xs + 1 + C.maxLeaves)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    indecisiveLStarLevel4 C (fun z => familyRead O.mq F kl kh z ω) xs
      = indecisiveLStarLevel4 C (fun z => familyRead O.mq F kl kh z ω') xs :=
  level4_congr_up C hC xs (le_probesLen xs)
    (fun _ hz => familyRead_congr O h (mem_shortStrings.2 hz))

theorem measurableSet_level4_result (O : Oracle μ (FreeMonoid α)) (C : Level4Settings)
    (hC : 2 ≤ C.maxLeaves) (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ}
    (xs : Fin T → FreeMonoid α) (Y : Set (Level4Result α)) :
    MeasurableSet {ω | indecisiveLStarLevel4 C (fun z => familyRead O.mq F kl kh z ω) xs ∈ Y} :=
  measurableSet_of_reads O _ _ fun ω ω' h hω => by
    simp only [Set.mem_setOf_eq] at hω ⊢
    rwa [← level4_result_congr O C hC F kl kh xs h]

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
    (F : Finset (FreeMonoid α)) (kl kh k : ℕ) (H : Hypothesis α) (x : FreeMonoid α) :
    MeasurableSet {ω | x ∈ searchingProbes (fun z => familyRead O.mq F kl kh z ω) k H} := by
  refine measurableSet_of_reads O
    (readStrings (shortStrings (x.toList.length + midLen H.tree)) F) _ fun ω ω' h hω => ?_
  have hagree : AgreeUpTo (x.toList.length + midLen H.tree)
      (fun z => (familyRead O.mq F kl kh z ω).cut) (fun z => (familyRead O.mq F kl kh z ω').cut) :=
    fun z hz => by dsimp only; rw [familyRead_congr O h (mem_shortStrings.2 hz)]
  simp only [Set.mem_setOf_eq, searchingProbes] at hω ⊢
  rwa [← probeBy_congr_up hagree (midLe_midLen H.tree) le_rfl]

theorem measurableSet_mem_level4Replay (O : Oracle μ (FreeMonoid α))
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (H : Hypothesis α) (t w : FreeMonoid α) (e : ℕ) :
    MeasurableSet {ω | t ∈ level4Replay (fun z => familyRead O.mq F kl kh z ω) H w e} := by
  refine measurableSet_of_reads O
    (readStrings (shortStrings (w.toList.length + midLen H.tree)) F) _ fun ω ω' h hω => ?_
  have hagree : AgreeUpTo (w.toList.length + midLen H.tree)
      (fun z => (familyRead O.mq F kl kh z ω).cut) (fun z => (familyRead O.mq F kl kh z ω').cut) :=
    fun z hz => by dsimp only; rw [familyRead_congr O h (mem_shortStrings.2 hz)]
  have hpref := prefStrings_agree_up hagree (midLe_midLen H.tree) le_rfl
  simp only [Set.mem_setOf_eq, level4Replay] at hω ⊢
  rw [← firstsBy_congr_pref hpref]
  have heq : (firstsBy (fun z => (familyRead O.mq F kl kh z ω).cut) H.tree
        (Function.curry H.edges) e w).filter
        (fun q => decide ((familyRead O.mq F kl kh q.2 ω').cut = none))
      = (firstsBy (fun z => (familyRead O.mq F kl kh z ω).cut) H.tree
        (Function.curry H.edges) e w).filter
        (fun q => decide ((familyRead O.mq F kl kh q.2 ω).cut = none)) :=
    List.filter_congr fun q hq => by
      rw [hpref q.2 (route_mem_prefStrings (firstsBy_pos _ _ _ _ _ hq).1)]
  rw [heq]
  exact hω

end Reads

section Round

variable {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

open scoped Classical in
/-- The state a round's clustering returns at, after `harvests`. -/
def roundState (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (harvests : List (List (FreeMonoid α))) (c : ClusterDraws Ω α K) : State :=
  Classical.epsilon fun B => B ∈ settings.schedule (loopPopulations K harvests.length)
    ∧ clusterRun harvests c ∈ ret O.mq (loopPopulations K harvests.length) none
      settings.indecisionLimit settings.gateLevel B

/-- A round at the state `B` and family `F`, on level 4's draws `d`. -/
def mkRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (x₀ : Run Ω (FreeMonoid α) (Option (Fin K))) (B : State) (F : Finset (FreeMonoid α))
    (d : Ω × (Fin settings.level4.probeBudget → FreeMonoid α)) : LoopRound Ω α K :=
  ⟨x₀, B, F, fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1,
    indecisiveLStarLevel4 settings.level4 (fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1) d.2⟩

omit [IsProbabilityMeasure μ] in
lemma loopRound_eq_mkRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (harvests : List (List (FreeMonoid α)))
    (d : LoopRoundDraws Ω α K settings.level4.probeBudget) :
    loopRound settings O harvests d = mkRound settings O (clusterRun harvests d.1)
      (roundState settings O harvests d.1)
      (clusterAt O.mq (loopPopulations K harvests.length) (clusterRun harvests d.1)
        (roundState settings O harvests d.1)) d.2 := rfl


omit [Fintype σ] in
/-- Where a round ending at hypothesis `H` and ending `e` fails its claim is measurable in the
oracle's answers. -/
theorem measurableSet_not_endsWell (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α))
    (kl kh k : ℕ) (M : _root_.DFA α σ) (bad : σ → Prop) (D : Measure (FreeMonoid α))
    (ε : ℝ) (s L : ℕ) (y : ℝ) (H : Hypothesis α) (e : RoundEnd α) :
    MeasurableSet {ω | ¬ EndsWell M bad D ε
      (searchingProbes (fun z => familyRead O.mq F kl kh z ω) k H) s L
      (level4Replay (fun z => familyRead O.mq F kl kh z ω) H) y e} := by
  classical
  cases e with
  | consistent =>
    have hf : Measurable fun ω =>
        D.real (searchingProbes (fun z => familyRead O.mq F kl kh z ω) k H) :=
      ENNReal.measurable_toReal.comp (measurable_measure_of_mem D _
        fun x => measurableSet_mem_searchingProbes O F kl kh k H x)
    exact (measurableSet_le hf measurable_const).compl
  | harvest zs =>
    have hrep : ∀ we : FreeMonoid α × ℕ, MeasurableSet {ω |
        level4Replay (fun z => familyRead O.mq F kl kh z ω) H we.1 we.2 ≠ []} := by
      intro we
      have hset : {ω | level4Replay (fun z => familyRead O.mq F kl kh z ω) H we.1 we.2 ≠ []}
          = ⋃ t, {ω | t ∈ level4Replay (fun z => familyRead O.mq F kl kh z ω) H we.1 we.2} := by
        ext ω
        simp only [Set.mem_setOf_eq, Set.mem_iUnion]
        exact ⟨List.exists_mem_of_ne_nil _, fun ⟨i, hi⟩ h => by simp [h] at hi⟩
      rw [hset]
      exact MeasurableSet.iUnion fun t => measurableSet_mem_level4Replay O F kl kh H t we.1 we.2
    have hS : ∀ t, Measurable fun ω => (D.prod (uniformAnchor s L)).real
        {we | t ∈ level4Replay (fun z => familyRead O.mq F kl kh z ω) H we.1 we.2} := fun t =>
      ENNReal.measurable_toReal.comp (measurable_measure_of_mem _ _
        fun we => measurableSet_mem_level4Replay O F kl kh H t we.1 we.2)
    have hY : Measurable fun ω => (D.prod (uniformAnchor s L)).real
        {we | level4Replay (fun z => familyRead O.mq F kl kh z ω) H we.1 we.2 ≠ []} :=
      ENNReal.measurable_toReal.comp (measurable_measure_of_mem _ _ hrep)
    simp only [EndsWell, Spread, Yields]
    exact measurableSet_setOf.2 (Measurable.not (measurable_const.and
      ((Measurable.forall fun t =>
          measurableSet_setOf.1 (measurableSet_le (hS t) measurable_const)).and
        (measurableSet_setOf.1 (measurableSet_le measurable_const hY)))))
  | failed =>
    simp only [EndsWell, not_false_eq_true, Set.setOf_true]
    exact MeasurableSet.univ

omit [Fintype σ] in
/-- At a given state and family, where level 4 fails its claim is a measurable set of its noise
and probes. -/
theorem measurableSet_level4Fail (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (P : PrefixDistribution α) (θ δ εd : ℝ) (k : ℕ) (settings : LoopSettings K)
    (hC : 2 ≤ settings.level4.maxLeaves) (x₀ : Run Ω (FreeMonoid α) (Option (Fin K)))
    (B : State) (F : Finset (FreeMonoid α)) :
    MeasurableSet {d : Ω × (Fin settings.level4.probeBudget → FreeMonoid α) |
      ¬ Level4Spec M O P θ δ εd k (mkRound settings O x₀ B F d)} := by
  classical
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  refine measurableSet_prod_countable_right _ fun xs => ?_
  set A := readStrings (shortStrings (probesLen xs + 1 + settings.level4.maxLeaves)) F
  set res : Ω → Level4Result α := fun ω =>
    indecisiveLStarLevel4 settings.level4 (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω) xs
  have hres : ∀ ω ω', (∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) → res ω = res ω' :=
    fun ω ω' h => level4_result_congr O _ hC F _ _ xs h
  set rep : Finset (FreeMonoid α) → Level4Result α := fun U =>
    if h : ∃ ω, A.filter (fun p => O.mq p ω = 1) = U then res h.choose else res ω₀
  have hrepω : ∀ ω, res ω = rep (A.filter (fun p => O.mq p ω = 1)) := by
    intro ω
    have h : ∃ ω', A.filter (fun p => O.mq p ω' = 1) = A.filter (fun p => O.mq p ω = 1) :=
      ⟨ω, rfl⟩
    simp only [rep, dif_pos h]
    refine hres _ _ fun p hp => ?_
    have := Finset.ext_iff.1 h.choose_spec p
    simp only [Finset.mem_filter, hp, true_and] at this
    exact this.symm
  set dist := familyReadDist O M F B.lo (B.hi + 1)
  set V := undecidedCount P M dist
  have hset : {ω | (ω, xs) ∈ {d : Ω × (Fin settings.level4.probeBudget → FreeMonoid α) |
        ¬ Level4Spec M O P θ δ εd k (mkRound settings O x₀ B F d)}}
      = ⋃ U ∈ A.powerset, ({ω | A.filter (fun p => O.mq p ω = 1) = U}
        ∩ {ω | ¬ EndsWell M (fun q => ¬ dist q .undecided ≤ 3 / 2 * θ) P.D εd
          (searchingProbes (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω) k (rep U).hypothesis)
          P.firstAnchor P.L
          (level4Replay (fun z => familyRead O.mq F B.lo (B.hi + 1) z ω) (rep U).hypothesis)
          ((V - 2 * Real.sqrt (P.prefixBound * V * Real.log (1 / δ)))
            / (P.L + 1 - P.firstAnchor : ℕ)) (rep U).ending}) := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_powerset,
      exists_prop]
    constructor
    · intro h
      refine ⟨_, Finset.filter_subset _ _, rfl, ?_⟩
      rw [← hrepω ω]
      exact h
    · rintro ⟨U, -, hU, h⟩
      rw [← hU, ← hrepω ω] at h
      exact h
  rw [hset]
  refine Finset.measurableSet_biUnion _ fun U _ => MeasurableSet.inter ?_ ?_
  · exact measurableSet_of_reads O A _ fun ω ω' h (hω : A.filter _ = U) => by
      show A.filter _ = U
      rw [← hω]
      exact Finset.filter_congr fun p hp => (h p hp).symm
  · exact measurableSet_not_endsWell O F _ _ k M _ P.D εd _ _ _ _ _

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
lemma measurableSet_certOf (j : Option (Fin K)) (m : ℕ) (Q : Finset (FreeMonoid α)) :
    MeasurableSet {x : Run Ω (FreeMonoid α) (Option (Fin K)) | certOf j m x = Q} := by
  classical
  have hrw : ∀ x : Run Ω (FreeMonoid α) (Option (Fin K)), certOf j m x
      = (Finset.univ : Finset (Fin m)).image (fun i => (fun z : Fin m => certPrefix j z.val x) i) :=
    fun x => image_range_eq_image_univ m (fun i => certPrefix j i x)
  simp only [hrw]
  exact measurableSet_finData (fun z : Fin m => certPrefix j z.val)
    (fun z => measurable_cert j z.val)
    (fun t => (Finset.univ : Finset (Fin m)).image (fun i => t i)) Q

omit [IsProbabilityMeasure μ] in
lemma voteCount_alike (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)}
    {p : FreeMonoid α} {ω ω' : Ω} (h : ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    voteCount O.mq F p ω = voteCount O.mq F p ω' := by
  unfold voteCount
  rw [Finset.filter_congr fun v hv => h v hv]

omit [IsProbabilityMeasure μ] in
lemma agreeCount_alike (O : Oracle μ (FreeMonoid α)) {lo hi : ℕ} {F P : Finset (FreeMonoid α)}
    {ω ω' : Ω} (h1 : ∀ p ∈ P, ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (h2 : ∀ p ∈ P, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    agreeCount O.mq lo hi F P ω = agreeCount O.mq lo hi F P ω' := by
  have hv : ∀ p ∈ P, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_alike O (h1 p hp)
  have hc : cutSides O.mq lo hi F P ω = cutSides O.mq lo hi F P ω' := by
    unfold cutSides
    congr 1 <;> exact Finset.filter_congr fun p hp => by rw [hv p hp]
  unfold agreeCount
  rw [hc, Finset.filter_congr fun p hp => h2 p hp]

lemma readSet_mono {P P' C C' : Finset (FreeMonoid α)} (hP : P ⊆ P') (hC : C ⊆ C') :
    readSet P C ⊆ readSet P' C' := fun w hw => by
  obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
  exact mem_readSet (hP hp) (hC hv)

/-- Where a run returns at the state `B` is measurable. -/
theorem measurableSet_ret (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (uni : Option (Fin K)) (ι a : ℝ) (B : State) :
    MeasurableSet (ret O.mq pops uni ι a B : Set (Run Ω (FreeMonoid α) (Option (Fin K)))) := by
  classical
  set cl : Finset (FreeMonoid α) → Finset (FreeMonoid α) → Ω → Finset (FreeMonoid α) :=
    fun P C ω => clusterOf lloydClusterer O B.sc B.scd P C B.k ω
  have hret : (ret O.mq pops uni ι a B : Set (Run Ω (FreeMonoid α) (Option (Fin K))))
      = {x | B.k ≤ (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x) (oracleNoise x)).card + 1
          ∧ ∀ j ∈ pops, (((certOf j B.npref x).filter fun p => ¬ decided O.mq B.lo (B.hi + 1)
            (insert 1 (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x) (oracleNoise x))) p
              (oracleNoise x)).card : ℝ) ≤ ι * (certOf j B.npref x).card}
        ∩ ⋃ e : ℕ, {x | admitted O.mq B.lo B.hi a
          (cl (prefixesAt pops B.npref x) (poolAt B.nsuff x) (oracleNoise x))
          (certOf uni (B.npref + e) x) (oracleNoise x)} := by
    ext x
    simp only [ret, familyAt, clusterAt_eq, clusterAt_eq_clusterOf, cl, Set.mem_setOf_eq,
      Set.mem_inter_iff, Set.mem_iUnion]
    tauto
  rw [hret]
  refine MeasurableSet.inter ?_ (MeasurableSet.iUnion fun e => ?_)
  · -- the size and the FNR test, at the table, the pool and the certificates
    refine measurableSet_of_data
      (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
        (prefixesAt pops B.npref x, poolAt B.nsuff x, fun j => certOf j B.npref x))
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
          poolAt B.nsuff x, fun j => certOf j B.npref x) = (P, C, cs)}
          = {x | prefixesAt pops B.npref x = P} ∩ ({x | poolAt B.nsuff x = C}
            ∩ ⋂ j, {x | certOf j B.npref x = cs j}) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Set.mem_iInter, Prod.mk.injEq,
          funext_iff]
      rw [this]
      exact (measurableSet_prefixesAt pops B.npref P).inter ((measurableSet_poolAt B.nsuff C).inter
        (MeasurableSet.iInter fun j => measurableSet_certOf j B.npref (cs j)))
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
  · -- the gate, at the table, the pool and the uniform pool's certificates
    refine measurableSet_of_data
      (fun x : Run Ω (FreeMonoid α) (Option (Fin K)) =>
        (prefixesAt pops B.npref x, poolAt B.nsuff x, certOf uni (B.npref + e) x))
      (fun y => ?_)
      (fun y => if (1 : FreeMonoid α) ∈ y.2.1 then Prod.fst ⁻¹'
          {ω | admitted O.mq B.lo B.hi a (cl y.1 y.2.1 ω) y.2.2 ω} else ∅)
      (fun y => ?_) |> fun h => by
        convert h using 1
        ext x
        simp only [Set.mem_setOf_eq, one_mem_poolAt, if_true, Set.mem_preimage]
        rfl
    · obtain ⟨P, C, Q⟩ := y
      have : {x : Run Ω (FreeMonoid α) (Option (Fin K)) | (prefixesAt pops B.npref x,
          poolAt B.nsuff x, certOf uni (B.npref + e) x) = (P, C, Q)}
          = {x | prefixesAt pops B.npref x = P} ∩ ({x | poolAt B.nsuff x = C}
            ∩ {x | certOf uni (B.npref + e) x = Q}) := by
        ext x
        simp only [Set.mem_setOf_eq, Set.mem_inter_iff, Prod.mk.injEq]
      rw [this]
      exact (measurableSet_prefixesAt pops B.npref P).inter ((measurableSet_poolAt B.nsuff C).inter
        (measurableSet_certOf uni (B.npref + e) Q))
    · obtain ⟨P, C, Q⟩ := y
      dsimp only
      split_ifs with hone
      · refine measurable_fst (measurableSet_of_reads O
          (readSet P C ∪ readSet Q (insert 1 C)) _ fun ω ω' h hω => ?_)
        have hcl : cl P C ω = cl P C ω' := clusterOf_congr_mq O B.sc B.scd P C B.k hone
          fun w hw => h w (Finset.mem_union_left _ hw)
        simp only [Set.mem_setOf_eq] at hω ⊢
        rw [← hcl]
        unfold admitted at hω ⊢
        rw [← agreeCount_alike O (lo := B.lo) (hi := B.hi) (F := cl P C ω) (P := Q)
          (fun p hp v hv => h _ (Finset.mem_union_right _ (mem_readSet hp
            (Finset.mem_insert_of_mem (clusterOf_subset O B.sc B.scd P C B.k ω hone hv)))))
          (fun p hp => by
            have := h p (Finset.mem_union_right _ (by
              simpa using mem_readSet (cands := insert 1 C) hp (Finset.mem_insert_self 1 C)))
            exact this)]
        exact hω
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

/-- After given harvests, the state a round's clustering returns at and its family are a
measurable function of the round's clustering draws. -/
theorem measurableSet_roundPick (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (harvests : List (List (FreeMonoid α))) (B : State) (F : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K | roundState settings O harvests c = B
      ∧ clusterAt O.mq (loopPopulations K harvests.length) (clusterRun harvests c)
        (roundState settings O harvests c) = F} := by
  classical
  have hrun := (measurePreserving_clusterRun (μ := μ) (K := K) (Measure.dirac (1 : FreeMonoid α))
    (Measure.dirac 1) harvests).measurable
  set pops := loopPopulations K harvests.length
  set sched := settings.schedule pops
  set Sc : ClusterDraws Ω α K → Finset State := fun c => sched.filter fun B' =>
    clusterRun harvests c ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B'
  have hstate : ∀ c, roundState settings O harvests c = Classical.epsilon (· ∈ Sc c) := by
    intro c
    unfold roundState
    congr 1
    funext B'
    simp only [Sc, Finset.mem_filter]
    rfl
  have hset : {c : ClusterDraws Ω α K | roundState settings O harvests c = B
      ∧ clusterAt O.mq pops (clusterRun harvests c) (roundState settings O harvests c) = F}
      = ⋃ T ∈ sched.powerset, ({c | Sc c = T} ∩ {c | Classical.epsilon (· ∈ T) = B
        ∧ clusterAt O.mq pops (clusterRun harvests c) (Classical.epsilon (· ∈ T)) = F}) := by
    ext c
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_powerset,
      exists_prop]
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨Sc c, Finset.filter_subset _ _, rfl, ?_⟩
      rw [← hstate]
      exact ⟨h1, h2⟩
    · rintro ⟨T, -, hT, h⟩
      rw [hstate, hT]
      exact h
  rw [hset]
  refine Finset.measurableSet_biUnion _ fun T hT => MeasurableSet.inter ?_ ?_
  · have hT' : T ⊆ sched := Finset.mem_powerset.1 hT
    have : {c : ClusterDraws Ω α K | Sc c = T} = ⋂ B' ∈ sched, {c |
        (clusterRun harvests c ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B'
          ↔ B' ∈ T)} := by
      ext c
      simp only [Set.mem_setOf_eq, Set.mem_iInter, Finset.ext_iff, Sc, Finset.mem_filter]
      constructor
      · intro h B' hB'
        have := h B'
        tauto
      · intro h B'
        by_cases hB' : B' ∈ sched
        · have := h B' hB'
          tauto
        · exact ⟨fun h' => absurd h'.1 hB', fun h' => absurd (hT' h') hB'⟩
    rw [this]
    refine Finset.measurableSet_biInter _ fun B' _ => ?_
    by_cases hB' : B' ∈ T
    · simp only [hB', iff_true]
      exact hrun (measurableSet_ret O pops none _ _ B')
    · simp only [hB', iff_false]
      exact (hrun (measurableSet_ret O pops none _ _ B')).compl
  · by_cases hB : Classical.epsilon (· ∈ T) = B
    · simp only [hB, true_and]
      exact hrun (measurableSet_clusterAt O pops B F)
    · simp only [hB, false_and, Set.setOf_false]
      exact MeasurableSet.empty

/-- After given harvests, what a round harvests is a measurable function of its draws. -/
theorem measurableSet_roundHarvested (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (hC : 2 ≤ settings.level4.maxLeaves) (harvests : List (List (FreeMonoid α)))
    (h : Option (List (FreeMonoid α))) :
    MeasurableSet {d : LoopRoundDraws Ω α K settings.level4.probeBudget |
      (loopRound settings O harvests d).result.ending.harvested = h} := by
  classical
  set pops := loopPopulations K harvests.length
  have hset : {d : LoopRoundDraws Ω α K settings.level4.probeBudget |
      (loopRound settings O harvests d).result.ending.harvested = h}
      = ⋃ BF : State × Finset (FreeMonoid α),
        (Prod.fst ⁻¹' {c | roundState settings O harvests c = BF.1
          ∧ clusterAt O.mq pops (clusterRun harvests c) (roundState settings O harvests c) = BF.2})
        ∩ (Prod.snd ⁻¹' {d₂ : Ω × (Fin settings.level4.probeBudget → FreeMonoid α) |
          (indecisiveLStarLevel4 settings.level4
            (fun z => familyRead O.mq BF.2 BF.1.lo (BF.1.hi + 1) z d₂.1) d₂.2).ending.harvested
              = h}) := by
    ext d
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
    constructor
    · intro hd
      exact ⟨(roundState settings O harvests d.1,
        clusterAt O.mq pops (clusterRun harvests d.1) (roundState settings O harvests d.1)),
        ⟨rfl, rfl⟩, hd⟩
    · rintro ⟨⟨B, F⟩, ⟨hB, hF⟩, hd⟩
      rw [loopRound_eq_mkRound]
      show (indecisiveLStarLevel4 settings.level4 (fun z => familyRead O.mq
        (clusterAt O.mq pops (clusterRun harvests d.1) (roundState settings O harvests d.1))
        (roundState settings O harvests d.1).lo ((roundState settings O harvests d.1).hi + 1) z
          d.2.1) d.2.2).ending.harvested = h
      rw [hF, hB]
      exact hd
  rw [hset]
  refine MeasurableSet.iUnion fun BF => MeasurableSet.inter
    (measurable_fst (measurableSet_roundPick settings O harvests BF.1 BF.2))
    (measurable_snd (measurableSet_prod_countable_right _ fun xs => ?_))
  exact measurableSet_level4_result O settings.level4 hC BF.2 _ _ xs
    {r | r.ending.harvested = h}

omit [Fintype α] [DecidableEq α] [MeasurableSpace Ω] in
lemma harvestsOf_append_single (rounds : List (LoopRound Ω α K)) (o : LoopRound Ω α K) :
    harvestsOf (rounds ++ [o]) = harvestsOf rounds ++ o.result.ending.harvested.toList := by
  unfold harvestsOf
  rw [List.filterMap_append]
  cases h : o.result.ending.harvested <;> simp [h]

omit [IsProbabilityMeasure μ] in
lemma harvestsOf_loopRounds_succ (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget) (r : ℕ) (hrK : r < K) :
    harvestsOf (loopRounds settings O draws (r + 1))
      = if (harvestsOf (loopRounds settings O draws r)).length = r then
          harvestsOf (loopRounds settings O draws r)
            ++ (loopRound settings O (harvestsOf (loopRounds settings O draws r))
              (draws ⟨r, hrK⟩)).result.ending.harvested.toList
        else harvestsOf (loopRounds settings O draws r) := by
  rw [loopRounds_succ]
  split_ifs with h1 h2 h2
  · rw [harvestsOf_append_single]
  · exact absurd (by rw [length_harvestsOf h1.2,
      length_eq_of_allHarvested settings O draws hrK.le h1.2]) h2
  · exfalso
    apply h1
    refine ⟨hrK, fun o ho => ?_⟩
    have hle := length_loopRounds_le settings O draws r
    have hlen : (harvestsOf (loopRounds settings O draws r)).length
        = (loopRounds settings O draws r).length := by
      have := List.length_filterMap_le (fun o : LoopRound Ω α K => o.result.ending.harvested)
        (loopRounds settings O draws r)
      unfold harvestsOf at h2 this ⊢
      omega
    unfold harvestsOf at hlen
    have := (List.filterMap_length_eq_length.1 hlen) o ho
    rw [Option.isSome_iff_ne_none] at this
    exact this
  · rfl

/-- The harvests of the first `r` rounds are a measurable function of the rounds' draws. -/
theorem measurableSet_harvestsOf (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (hC : 2 ≤ settings.level4.maxLeaves) (r : ℕ) (H : List (List (FreeMonoid α))) :
    MeasurableSet {draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget |
      harvestsOf (loopRounds settings O draws r) = H} := by
  classical
  induction r generalizing H with
  | zero =>
    simp only [loopRounds, harvestsOf, List.filterMap_nil]
    exact MeasurableSet.const _
  | succ r ih =>
    by_cases hrK : r < K
    · have hset : {draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget |
          harvestsOf (loopRounds settings O draws (r + 1)) = H}
          = ⋃ H', {draws | harvestsOf (loopRounds settings O draws r) = H'}
            ∩ (fun draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget =>
              draws ⟨r, hrK⟩) ⁻¹' {d | (if H'.length = r then
                H' ++ (loopRound settings O H' d).result.ending.harvested.toList else H') = H} := by
        ext draws
        simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage,
          harvestsOf_loopRounds_succ settings O draws r hrK]
        exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨H', hH', h⟩ => hH' ▸ h⟩
      rw [hset]
      refine MeasurableSet.iUnion fun H' => (ih H').inter (measurable_pi_apply _ ?_)
      by_cases hc : H'.length = r
      · simp only [hc, if_true]
        have : {d : LoopRoundDraws Ω α K settings.level4.probeBudget |
            H' ++ (loopRound settings O H' d).result.ending.harvested.toList = H}
            = ⋃ o : Option (List (FreeMonoid α)),
              if H' ++ o.toList = H then
                {d | (loopRound settings O H' d).result.ending.harvested = o} else ∅ := by
          ext d
          simp only [Set.mem_setOf_eq, Set.mem_iUnion]
          constructor
          · intro h
            exact ⟨(loopRound settings O H' d).result.ending.harvested, by rw [if_pos h]; rfl⟩
          · rintro ⟨o, ho⟩
            split_ifs at ho with hh
            · rw [Set.mem_setOf_eq.mp ho]; exact hh
            · exact absurd ho (Set.notMem_empty _)
        rw [this]
        refine MeasurableSet.iUnion fun o => ?_
        split_ifs
        · exact measurableSet_roundHarvested settings O hC H' o
        · exact MeasurableSet.empty
      · simp only [hc, if_false]
        exact MeasurableSet.const _
    · have hst : ∀ draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget,
          loopRounds settings O draws (r + 1) = loopRounds settings O draws r := fun draws => by
        rw [loopRounds_succ, dif_neg (fun h => hrK h.1)]
      simp only [hst]
      exact ih H

end Rounds

end OrthoDFA
