import RLStar.Proofs.LoopTermination
import RLStar.Proofs.Nearest
import RLStar.Proofs.RawMeas

/-!
# A round's outcomes are measurable

An event the oracle's answers at finitely many strings fix is measurable
(`measurableSet_of_reads`). Over given probes, level 3 and the lazy round read only strings of
bounded length, so their results are fixed by finitely many answers (`lazyRound_result_congr`).
A round's harvest data are fixed by finitely many answers too, so the earlier rounds' data are a
measurable function of their draws (`measurableSet_codesOf`).
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
  (simp only [familyRead_eq]; unfold voteCount)
  congr 2
  exact Finset.filter_congr fun v hv =>
    h _ (Finset.mem_image.2 ⟨(z, v), Finset.mem_product.2 ⟨hz, hv⟩, rfl⟩)

/-- The probes' longest. -/
def probesLen {T : ℕ} (xs : Fin T → FreeMonoid α) : ℕ :=
  Finset.univ.sup fun i => (xs i).toList.length

/-- The longest of a lazy round's probes and block draws. -/
def lazyProbesLen {E P J b : ℕ} (d : Fin E → Random.EpochDraws α P J b) : ℕ :=
  Finset.univ.sup fun e =>
    max (probesLen (d e).1) (Finset.univ.sup fun j => probesLen ((d e).2 j))

omit [Fintype α] [DecidableEq α] in
theorem le_probesLen {T : ℕ} (xs : Fin T → FreeMonoid α) (i : Fin T) :
    (xs i).toList.length ≤ probesLen xs :=
  Finset.le_sup (f := fun i => (xs i).toList.length) (Finset.mem_univ i)

omit [IsProbabilityMeasure μ] in
/-- Level 3's result, over given probes, as a function of the oracle's answers, is fixed by its
answers at finitely many strings. -/
theorem level3_result_congr (O : Oracle μ (FreeMonoid α)) (C : Random.Level3Settings)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ p ∈ readStrings (shortStrings (probesLen xs + 1 + T + 2)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs
      = Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω') xs :=
  Random.level3_congr_zone C xs fun i j _ hj m hm => familyRead_congr O h (mem_shortStrings.2 (by
    have h1 := le_probesLen xs i
    have h2 : (prefixOf (xs i) j).toList.length ≤ (xs i).toList.length := by simp [prefixOf]
    simp only [FreeMonoid.toList_mul, List.length_append]
    omega))

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
def midLen (T : DTree α) : ℕ := (Random.mids T).sup fun m => m.toList.length

theorem midsLe_midLen (T : DTree α) : Random.MidsLe (midLen T) T := fun _ hm =>
  Finset.le_sup (f := fun m : FreeMonoid α => m.toList.length) hm

omit [IsProbabilityMeasure μ] in
theorem probeOrAgree_congr_reads (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)}
    {kl kh : ℕ} (k : ℕ) (H : HypothesisDFA α) (x : FreeMonoid α) {ω ω' : Ω}
    (h : ∀ p ∈ readStrings (shortStrings (x.toList.length + midLen H.tree)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    probeOrAgree (fun z => familyRead O.mq F kl kh z ω) H.tree H.edges k x
      = probeOrAgree (fun z => familyRead O.mq F kl kh z ω') H.tree H.edges k x :=
  Random.probeOrAgree_congr_zone H.edges k fun j _ hj m hm =>
    familyRead_congr O h (mem_shortStrings.2 (by
      have h1 := midsLe_midLen H.tree m hm
      have h2 : (prefixOf x j).toList.length ≤ x.toList.length := by simp [prefixOf]
      simp only [FreeMonoid.toList_mul, List.length_append]
      omega))

end Reads

/-! ## The lazy round's reads -/

section LazyCongr

open Random Query

omit [Fintype α] [DecidableEq α] in
private lemma length_unpad_prefixOf_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (j : ℕ) : (unpad (prefixOf (padIn K X x) j)).toList.length ≤ x.toList.length := by
  have hp : ((padIn K X x).toList.take j).reduceOption <+: (padIn K X x).toList.reduceOption :=
    (List.take_prefix j _).filterMap id
  have hx : (padIn K X x).toList.reduceOption = x.toList := by
    simp [padIn, List.reduceOption, List.filterMap_append]
  simpa [unpad, prefixOf, hx] using hp.length_le

omit [DecidableEq α] in
private lemma length_unpad_mul_le (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) (j : ℕ)
    (m : FreeMonoid (Option α)) :
    (unpad (prefixOf (padIn K X x) j * m)).toList.length
      ≤ x.toList.length + m.toList.length := by
  have h1 := length_unpad_prefixOf_padIn K X x j
  have h2 : (unpad m).toList.length ≤ m.toList.length := by
    simpa [unpad, List.reduceOption] using List.length_filterMap_le id m.toList
  have h3 : unpad (prefixOf (padIn K X x) j * m) = unpad (prefixOf (padIn K X x) j) * unpad m := by
    simp [unpad, List.reduceOption, List.filterMap_append]
  rw [h3, FreeMonoid.toList_mul, List.length_append]
  omega

private lemma epochQ_eval_congr {r r' : FreeMonoid α → ARU} (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P L : ℕ} (xs : Fin P → FreeMonoid α)
    (hx : ∀ i, (xs i).toList.length ≤ L)
    (h : ∀ z : FreeMonoid α, z.toList.length ≤ L + P → r z = r' z) :
    (epochQ C K X xs).eval r = (epochQ C K X xs).eval r' := by
  simp only [epochQ, eval_subst, unpadQ, eval_query]
  refine level3_congr_zone { C with walkStart := K } (fun i => padIn K X (xs i))
    fun i j _ _ m hm => h _ ((length_unpad_mul_le K X (xs i) j m).trans ?_)
  have := hx i
  omega

private lemma midsLe_epochQ (r : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (xs : Fin P → FreeMonoid α) :
    MidsLe P ((epochQ C K X xs).eval r).hypothesis.tree := by
  simp only [epochQ, eval_subst, unpadQ, eval_query]
  have := (runProbes_tree (fun z => r (unpad z)) { C with walkStart := K }
    (List.ofFn fun i => padIn K X (xs i))).2.2
  rw [List.length_ofFn] at this
  exact (indecisiveLStarLevel3_tree (fun z => r (unpad z)) { C with walkStart := K }
    (fun i => padIn K X (xs i))) ▸ this

private lemma blockQ_eval_congr {r r' : FreeMonoid α → ARU} (K : ℕ) (H : HypothesisDFA (Option α))
    (s : Level3Site) (X : Finset (FreeMonoid α)) {L T : ℕ} (hH : MidsLe T H.tree)
    (h : ∀ z : FreeMonoid α, z.toList.length ≤ L + T → r z = r' z) :
    ∀ xs : List (FreeMonoid α), (∀ x ∈ xs, x.toList.length ≤ L) →
      (blockQ K H s X xs).eval r = (blockQ K H s X xs).eval r'
  | [], _ => rfl
  | x :: xs, hx => by
    have hpr : (probeOrAgreeQ H.tree H.edges K (padIn K X x)).eval (fun z => r (unpad z))
        = (probeOrAgreeQ H.tree H.edges K (padIn K X x)).eval (fun z => r' (unpad z)) :=
      probeOrAgree_congr_zone H.edges K fun j _ _ m hm =>
        h _ ((length_unpad_mul_le K X x j m).trans (by
          have := hx x List.mem_cons_self
          have := hH m hm
          omega))
    have hrest := blockQ_eval_congr K H s X hH h xs fun y hy => hx y (List.mem_cons_of_mem _ hy)
    simp only [blockQ, eval_bind, eval_map, eval_subst, unpadQ, eval_query]
    rw [hrest]
    cases s <;> simp only [siteHarvestQ, eval_map, hpr]

private lemma topUpQ_eval_congr {r r' : FreeMonoid α → ARU} (K nh : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) {L T b : ℕ} (hH : MidsLe T H.tree)
    (h : ∀ z : FreeMonoid α, z.toList.length ≤ L + T → r z = r' z) :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)),
      (∀ B ∈ Bs, ∀ t, (B t).toList.length ≤ L) →
      (topUpQ K nh H s Bs X ℓ).eval r = (topUpQ K nh H s Bs X ℓ).eval r'
  | [], _, _, _ => rfl
  | B :: Bs, X, ℓ, hB => by
    simp only [topUpQ, eval_bind]
    rw [blockQ_eval_congr K H s X hH h (List.ofFn B) fun x hx => by
      obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
      exact hB B List.mem_cons_self t]
    generalize (blockQ K H s X (List.ofFn B)).eval r' = ws
    split_ifs
    · rfl
    · rfl
    · exact topUpQ_eval_congr K nh H s hH h Bs _ _ fun B' hB' => hB B' (List.mem_cons_of_mem _ hB')

private lemma lazyEpochsQ_eval_congr {r r' : FreeMonoid α → ARU} (C : Level3Settings) (K nh : ℕ)
    {J b L : ℕ} (h : ∀ z : FreeMonoid α, z.toList.length ≤ L + C.probeBudget → r z = r' z) :
    ∀ (e pk : ℕ) (ds : List (EpochDraws α C.probeBudget J b)) (X : Finset (FreeMonoid α))
      (ls : LazyLists α), (∀ d ∈ ds, (∀ i, (d.1 i).toList.length ≤ L)
        ∧ ∀ j t, (d.2 j t).toList.length ≤ L) →
      (lazyEpochsQ C K nh e pk ds X ls).eval r = (lazyEpochsQ C K nh e pk ds X ls).eval r'
  | _, _, [], _, _, _ => rfl
  | e, pk, d :: ds, X, ls, hd => by
    have hd0 := hd d List.mem_cons_self
    simp only [lazyEpochsQ, eval_bind]
    rw [epochQ_eval_congr (epochSettings C e) K X d.1 hd0.1 h]
    have hmid := midsLe_epochQ r' (epochSettings C e) K X d.1
    generalize (epochQ (epochSettings C e) K X d.1).eval r' = res at hmid ⊢
    rcases res with ⟨hyp, _ | s | _⟩
    · rfl
    · simp only [eval_bind]
      rw [topUpQ_eval_congr K nh hyp s hmid h (List.ofFn d.2) X (ls s) fun B hB t => by
        obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hB
        exact hd0.2 j t]
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval r' = o
      rcases o with _ | ⟨_ | _, rest⟩
      · rfl
      · rfl
      · exact lazyEpochsQ_eval_congr C K nh h _ _ ds _ _ fun d' hd' =>
          hd d' (List.mem_cons_of_mem _ hd')
    · rfl

end LazyCongr

/-- Reads agreeing on every string at most `probes + T` long give the same lazy round. -/
theorem lazyRound_eval_congr (C : Random.Level3Settings) (K nh : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (d : Fin E → Random.EpochDraws α C.probeBudget J b)
    {r r' : FreeMonoid α → ARU}
    (h : ∀ z : FreeMonoid α, z.toList.length ≤ lazyProbesLen d + C.probeBudget → r z = r' z) :
    (Random.lazyRoundQ C K nh X₀ d).eval r = (Random.lazyRoundQ C K nh X₀ d).eval r' := by
  refine lazyEpochsQ_eval_congr C K nh h _ _ _ _ _ fun dd hdd => ?_
  obtain ⟨e, rfl⟩ := List.mem_ofFn.1 hdd
  have he : max (probesLen (d e).1) (Finset.univ.sup fun j => probesLen ((d e).2 j))
      ≤ lazyProbesLen d :=
    Finset.le_sup (f := fun e => max (probesLen (d e).1)
      (Finset.univ.sup fun j => probesLen ((d e).2 j))) (Finset.mem_univ e)
  refine ⟨fun i => (le_probesLen _ i).trans ((le_max_left _ _).trans he), fun j t => ?_⟩
  have hj : probesLen ((d e).2 j) ≤ Finset.univ.sup fun j => probesLen ((d e).2 j) :=
    Finset.le_sup (f := fun j => probesLen ((d e).2 j)) (Finset.mem_univ j)
  exact (le_probesLen _ t).trans (hj.trans ((le_max_right _ _).trans he))

section LazyResult

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The lazy round's result, over given draws, as a function of the oracle's answers, is fixed by
its answers at finitely many strings. -/
theorem lazyRound_result_congr (O : Oracle μ (FreeMonoid α)) (C : Random.Level3Settings)
    (K nh : ℕ) {E J b : ℕ} (X₀ F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (d : Fin E → Random.EpochDraws α C.probeBudget J b) {ω ω' : Ω}
    (h : ∀ p ∈ readStrings (shortStrings (lazyProbesLen d + 1 + C.probeBudget + 2)) F,
      (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    (Random.lazyRoundQ C K nh X₀ d).eval (fun z => familyRead O.mq F kl kh z ω)
      = (Random.lazyRoundQ C K nh X₀ d).eval (fun z => familyRead O.mq F kl kh z ω') :=
  lazyRound_eval_congr C K nh X₀ d fun _ hz => familyRead_congr O h (mem_shortStrings.2 (by omega))

end LazyResult

section Round

variable {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

open scoped Classical in
/-- The state a round's clustering returns at, after the history `h`. -/
def roundState (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α) (c : ClusterDraws Ω α K
        settings.level3.harvestReadCap) : State :=
  Classical.epsilon fun B => B ∈ settings.schedule (loopPopulations K h.pops.length)
    ∧ idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h c ∈ ret O.mq
        (loopPopulations K h.pops.length) none
      settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B

/-- A round at the state `B` and family `F`, its clustering having read the prefixes `CP` and the
suffixes `Sf`, its lazy round from the trie `X₀` on the noise and probes `d`. -/
def roundAt (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (B : State) (F X₀ CP Sf : Finset (FreeMonoid α))
    (d : Ω × (Fin settings.epochs →
      Random.EpochDraws α settings.level3.probeBudget settings.blocks settings.blockSize)) :
    SpecRound α :=
  ⟨B, F, fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1,
    (Random.lazyRoundQ settings.level3 settings.level3.walkStart settings.listSize X₀ d.2).eval
      fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1,
    CP, Sf⟩

/-- The round's probes, drawn by the lazy uniforms from `D` avoiding `K₀`. -/
def probesOf (settings : LoopSettings K) (D : Measure (FreeMonoid α)) (K₀ : Finset (FreeMonoid α))
    (u : settings.Uniforms) :
    Fin settings.epochs →
      Random.EpochDraws α settings.level3.probeBudget settings.blocks settings.blockSize :=
  lazyProbes (avoidLaw D settings.level3.walkStart K₀) u.2

/-- A round at the state `B` and family `F`, its trie that of `CP` and the heavy prefixes of the
draws from `D`, its probes drawn from `D` avoiding `K₀`, on the noise and uniforms `d`. -/
def mkRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α))
    (B : State) (F K₀ CP Sf : Finset (FreeMonoid α))
    (d : Ω × (settings.Uniforms)) : SpecRound α :=
  roundAt settings O B F (Random.prefixTrie settings.level3.walkStart
    (CP ∪ heavySet D settings.heavyCount settings.level3.walkStart d.2.1)) CP Sf
    (d.1, probesOf settings D K₀ d.2)

/-- The first `k` letters of the prefixes a round's clustering reads. -/
def roundKeys (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (c : ClusterDraws Ω α K settings.level3.harvestReadCap) : Finset (FreeMonoid α) :=
  clusterKeys (settings.schedule (loopPopulations K h.pops.length))
    (loopPopulations K h.pops.length) settings.level3.walkStart (idealRun D Dsf
        settings.level3.walkStart settings.level3.harvestReadCap h c)

/-- The prefixes and suffixes a round's clustering reads. -/
def roundSets (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (c : ClusterDraws Ω α K settings.level3.harvestReadCap) : Finset (FreeMonoid α) × Finset
        (FreeMonoid α) :=
  (roundPrefs (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) (idealRun D Dsf settings.level3.walkStart
          settings.level3.harvestReadCap h c),
    roundSuffs (settings.schedule (loopPopulations K h.pops.length)) (idealRun D Dsf
        settings.level3.walkStart settings.level3.harvestReadCap h c))

omit [IsProbabilityMeasure μ] in
lemma idealRound_eq_mkRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α settings) :
    idealRound settings O D Dsf h d = mkRound settings O D
      (roundState settings O D Dsf h (d.1, d.2.1))
      (clusterAt O.mq (loopPopulations K h.pops.length) (idealRun D Dsf settings.level3.walkStart
          settings.level3.harvestReadCap h (d.1, d.2.1))
        (roundState settings O D Dsf h (d.1, d.2.1)))
      (roundKeys settings D Dsf h (d.1, d.2.1)) (roundSets settings D Dsf h (d.1, d.2.1)).1
      (roundSets settings D Dsf h (d.1, d.2.1)).2 (d.1, d.2.2) := rfl


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
lemma miscutBound_alike (O : Oracle μ (FreeMonoid α)) {a c yAcc yRej : ℝ} {lo hi : ℕ}
    {F R P : Finset (FreeMonoid α)} {ω ω' : Ω}
    (hF : ∀ p ∈ P, ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hR : ∀ p ∈ P, ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hs : ∀ p ∈ P, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    miscutBound O.mq a lo hi F R P c yAcc yRej ω
      = miscutBound O.mq a lo hi F R P c yAcc yRej ω' := by
  classical
  have hv : ∀ p ∈ P, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_alike O (hF p hp)
  have hacc : acceptSide O.mq hi F P ω = acceptSide O.mq hi F P ω' := by
    unfold acceptSide
    exact Finset.filter_congr fun p hp => by rw [hv p hp]
  have hrej : rejectSide O.mq lo F P ω = rejectSide O.mq lo F P ω' := by
    unfold rejectSide
    exact Finset.filter_congr fun p hp => by rw [hv p hp]
  have hside : ∀ (accept : Bool) (A : Finset (FreeMonoid α)), A ⊆ P → ∀ y : ℝ,
      sideCap O.mq accept c A R a y ω = sideCap O.mq accept c A R a y ω' := by
    intro accept A hA y
    unfold sideCap
    have h1 : againstOn O.mq accept A ω = againstOn O.mq accept A ω' := by
      unfold againstOn
      congr 1
      exact Finset.filter_congr fun p hp => by rw [hs p (hA hp)]
    have h2 : refApart O.mq c A R ω = refApart O.mq c A R ω' := by
      unfold refApart
      split_ifs with h
      · refine Finset.inf'_congr h rfl fun v hv => ?_
        exact Finset.sum_congr rfl fun p hp => apartCost_congr c (hs p (hA hp)) (hR p (hA hp) v hv)
      · rfl
    rw [h1, h2]
  unfold miscutBound
  have hA : acceptSide O.mq hi F P ω' ⊆ P := by unfold acceptSide; exact Finset.filter_subset _ _
  have hR' : rejectSide O.mq lo F P ω' ⊆ P := by unfold rejectSide; exact Finset.filter_subset _ _
  rw [hacc, hrej, hside true _ hA, hside false _ hR']

omit [IsProbabilityMeasure μ] in
lemma certifiesAt_alike (O : Oracle μ (FreeMonoid α)) {c yAcc yRej lim a : ℝ} {lo hi : ℕ}
    {F R P : Finset (FreeMonoid α)} {ω ω' : Ω}
    (hF : ∀ p ∈ P, ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hR : ∀ p ∈ P, ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hs : ∀ p ∈ P, (O.mq p ω = 1 ↔ O.mq p ω' = 1)) :
    certifiesAt O.mq c yAcc yRej lim a lo hi F R P ω
      ↔ certifiesAt O.mq c yAcc yRej lim a lo hi F R P ω' := by
  unfold certifiesAt
  rw [miscutBound_alike O hF hR hs]

/-- Where a run returns at the state `B` is measurable. -/
theorem measurableSet_ret (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (uni : Option (Fin K)) (η₀ ι εcov a : ℝ) (B : State) :
    MeasurableSet (ret O.mq pops uni η₀ ι εcov a B : Set (Run Ω (FreeMonoid α) (Option (Fin K)))) :=
        by
  rw [ret_eq]
  exact measurableSet_retBy O pops uni (readη η₀) ι εcov a B

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
        {ω | clusterOf (nearestClusterer B.centre) O B.sc B.scd y.1 y.2 B.k ω = F} else ∅)
    (fun y => by
      split_ifs with hone
      · exact measurable_fst (measurableSet_clusterOf (rule := nearestClusterer B.centre) O B.sc
          B.scd y.1
        y.2
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
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (h : SpecHist α) (B : State) (F :
        Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K settings.level3.harvestReadCap | roundState settings O D
        Dsf h c = B
      ∧ clusterAt O.mq (loopPopulations K h.pops.length) (idealRun D Dsf settings.level3.walkStart
          settings.level3.harvestReadCap h c)
        (roundState settings O D Dsf h c) = F} := by
  classical
  have hrun := measurable_idealRun (Ω := Ω) (K := K) D Dsf settings.level3.walkStart
    settings.level3.harvestReadCap h
  set pops := loopPopulations K h.pops.length
  set sched := settings.schedule pops
  set Sc : ClusterDraws Ω α K settings.level3.harvestReadCap → Finset State := fun c => sched.filter
      fun B' =>
    idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h c ∈ ret O.mq pops none
        settings.η₀ settings.indecisionLimit settings.εcov
      settings.gateLevel B'
  have hstate : ∀ c, roundState settings O D Dsf h c = Classical.epsilon (· ∈ Sc c) := by
    intro c
    unfold roundState
    congr 1
    funext B'
    simp only [Sc, Finset.mem_filter]
    rfl
  have hset : {c : ClusterDraws Ω α K settings.level3.harvestReadCap | roundState settings O D Dsf h
      c = B
      ∧ clusterAt O.mq pops (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap
          h c) (roundState settings O D Dsf h c) = F}
      = ⋃ T ∈ sched.powerset, ({c | Sc c = T} ∩ {c | Classical.epsilon (· ∈ T) = B
        ∧ clusterAt O.mq pops (idealRun D Dsf settings.level3.walkStart
            settings.level3.harvestReadCap h c) (Classical.epsilon (· ∈ T)) = F}) := by
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
    have : {c : ClusterDraws Ω α K settings.level3.harvestReadCap | Sc c = T} = ⋂ B' ∈ sched, {c |
        (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h c ∈ ret O.mq pops
            none settings.η₀ settings.indecisionLimit settings.εcov
          settings.gateLevel B'
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
    [IsProbabilityMeasure D]
    (h : SpecHist α) (K₀ : Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K settings.level3.harvestReadCap | roundKeys settings D Dsf
        h c = K₀} := by
  classical
  have hrun := measurable_idealRun (Ω := Ω) (K := K) D Dsf settings.level3.walkStart
    settings.level3.harvestReadCap h
  have : {c : ClusterDraws Ω α K settings.level3.harvestReadCap | roundKeys settings D Dsf h c = K₀}
      = ⋃ P : Finset (FreeMonoid α), idealRun D Dsf settings.level3.walkStart
          settings.level3.harvestReadCap h ⁻¹'
        ({x | roundPrefs (settings.schedule (loopPopulations K h.pops.length))
          (loopPopulations K h.pops.length) x = P}
          ∩ {_x | P.image (fun p => prefixOf p settings.level3.walkStart) = K₀}) := by
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
    [IsProbabilityMeasure D]
    (h : SpecHist α) (PS : Finset (FreeMonoid α) × Finset (FreeMonoid α)) :
    MeasurableSet {c : ClusterDraws Ω α K settings.level3.harvestReadCap | roundSets settings D Dsf
        h c = PS} := by
  have hrun := measurable_idealRun (Ω := Ω) (K := K) D Dsf settings.level3.walkStart
    settings.level3.harvestReadCap h
  have : {c : ClusterDraws Ω α K settings.level3.harvestReadCap | roundSets settings D Dsf h c = PS}
      = idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h ⁻¹' ({x |
          roundPrefs (settings.schedule (loopPopulations K
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
    (K₀ : Finset (FreeMonoid α)) : Measurable (probesOf settings D K₀) := by
  unfold probesOf lazyProbes
  refine measurable_pi_lambda _ fun e => Measurable.prodMk ?_ ?_
  · exact measurable_pi_lambda _ fun i => (measurable_drawFrom _).comp
      ((measurable_pi_apply i).comp (measurable_fst.comp ((measurable_pi_apply e).comp
        measurable_snd)))
  · exact measurable_pi_lambda _ fun j => measurable_pi_lambda _ fun t =>
      (measurable_drawFrom _).comp ((measurable_pi_apply t).comp
        ((measurable_pi_apply j).comp (measurable_snd.comp ((measurable_pi_apply e).comp
          measurable_snd))))

/-- The heavy draws. -/
def heavyDrawsOf (settings : LoopSettings K) (D : Measure (FreeMonoid α)) (u : settings.Uniforms) :
    Fin settings.heavyDraws → FreeMonoid α :=
  fun i => drawFrom D (u.1 i)

omit [DecidableEq α] in
theorem measurable_heavyDrawsOf (settings : LoopSettings K) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] :
    Measurable (heavyDrawsOf settings D) :=
  measurable_pi_lambda _ fun i => (measurable_drawFrom _).comp
    ((measurable_pi_apply i).comp measurable_fst)

/-- After the history `h`, a round's harvest data are a measurable function of its draws. -/
theorem measurableSet_roundCode (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (N : ℕ)
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (h : SpecHist α)
    (oc : Option (HarvestCode α)) :
    MeasurableSet {d : LoopRoundDraws Ω α settings |
      popCode N (idealRound settings O D Dsf h d) = oc} := by
  classical
  set pops := loopPopulations K h.pops.length
  set Kw := settings.level3.walkStart
  have hcl : Measurable fun d : LoopRoundDraws Ω α settings =>
      ((d.1, d.2.1) : ClusterDraws Ω α K settings.level3.harvestReadCap) :=
    measurable_fst.prodMk (measurable_fst.comp measurable_snd)
  have hlv : Measurable fun d : LoopRoundDraws Ω α settings => (d.1, d.2.2) :=
    measurable_fst.prodMk (measurable_snd.comp measurable_snd)
  have hset : {d : LoopRoundDraws Ω α settings |
      popCode N (idealRound settings O D Dsf h d) = oc}
      = ⋃ BFK : State × Finset (FreeMonoid α) × Finset (FreeMonoid α)
          × (Finset (FreeMonoid α) × Finset (FreeMonoid α)),
        ((fun d : LoopRoundDraws Ω α settings =>
            ((d.1, d.2.1) : ClusterDraws Ω α K settings.level3.harvestReadCap)) ⁻¹'
          ({c | roundState settings O D Dsf h c = BFK.1
            ∧ clusterAt O.mq pops (idealRun D Dsf Kw settings.level3.harvestReadCap h c)
              (roundState settings O D Dsf h c) = BFK.2.1}
            ∩ {c | roundKeys settings D Dsf h c = BFK.2.2.1}
            ∩ {c | roundSets settings D Dsf h c = BFK.2.2.2}))
        ∩ ((fun d : LoopRoundDraws Ω α settings => (d.1, d.2.2)) ⁻¹'
          ((fun e : Ω × settings.Uniforms =>
              (e.1, (heavyDrawsOf settings D e.2, probesOf settings D BFK.2.2.1 e.2))) ⁻¹'
            {d₂ | popCode N (roundAt settings O BFK.1 BFK.2.1
                (Random.prefixTrie Kw (BFK.2.2.2.1 ∪ heavyOf settings.heavyCount Kw d₂.2.1))
                BFK.2.2.2.1 BFK.2.2.2.2 (d₂.1, d₂.2.2)) = oc})) := by
    ext d
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
    constructor
    · intro hd
      refine ⟨(roundState settings O D Dsf h (d.1, d.2.1),
        clusterAt O.mq pops (idealRun D Dsf Kw settings.level3.harvestReadCap h (d.1, d.2.1))
          (roundState settings O D Dsf h (d.1, d.2.1)), roundKeys settings D Dsf h (d.1, d.2.1),
          roundSets settings D Dsf h (d.1, d.2.1)), ⟨⟨⟨rfl, rfl⟩, rfl⟩, rfl⟩, ?_⟩
      rw [idealRound_eq_mkRound] at hd
      exact hd
    · rintro ⟨⟨B, F, K₀, PS⟩, ⟨⟨⟨hB, hF⟩, hK⟩, hS⟩, hd⟩
      rw [idealRound_eq_mkRound]
      dsimp only at hB hF hK hS hd
      rw [hF, hB, hK, hS]
      exact hd
  rw [hset]
  refine MeasurableSet.iUnion fun BFK => MeasurableSet.inter
    (hcl (((measurableSet_roundPick settings O D Dsf h BFK.1 BFK.2.1).inter
      (measurableSet_roundKeys settings D Dsf h BFK.2.2.1)).inter
      (measurableSet_roundSets settings D Dsf h BFK.2.2.2)))
    (hlv ((measurable_fst.prodMk (((measurable_heavyDrawsOf settings D).comp measurable_snd).prodMk
      ((measurable_probesOf settings D BFK.2.2.1).comp measurable_snd)))
      (measurableSet_prod_countable_right _ fun xs => ?_)))
  set A := readStrings (shortStrings (lazyProbesLen xs.2 + 1 + settings.level3.probeBudget + 2))
      BFK.2.1 ∪ readStrings (shortStrings N) BFK.2.1
  refine measurableSet_of_reads O A _ fun ω ω' h' hω => ?_
  simp only [Set.mem_setOf_eq] at hω ⊢
  rw [← hω]
  have h1 : ∀ p ∈ readStrings (shortStrings
      (lazyProbesLen xs.2 + 1 + settings.level3.probeBudget + 2)) BFK.2.1,
      (O.mq p ω' = 1 ↔ O.mq p ω = 1) := fun p hp => (h' p (Finset.mem_union_left _ hp)).symm
  have h2 : ∀ p ∈ readStrings (shortStrings N) BFK.2.1, (O.mq p ω' = 1 ↔ O.mq p ω = 1) :=
    fun p hp => (h' p (Finset.mem_union_right _ hp)).symm
  have hres := lazyRound_result_congr O settings.level3 Kw settings.listSize
    (Random.prefixTrie Kw (BFK.2.2.2.1 ∪ heavyOf settings.heavyCount Kw xs.1)) BFK.2.1 BFK.1.lo
    (BFK.1.hi + 1) xs.2 h1
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
lemma codesOf_append_single (N : ℕ) (rounds : List (SpecRound α)) (o : SpecRound α) :
    codesOf N (rounds ++ [o]) = codesOf N rounds ++ (popCode N o).toList := by
  unfold codesOf
  rw [List.filterMap_append]
  cases h : popCode N o <;> simp [h]

omit [MeasurableSpace Ω] [IsProbabilityMeasure μ] in
lemma length_codesOf (N : ℕ) (rounds : List (SpecRound α)) :
    (codesOf N rounds).length = (specHistOf rounds).pops.length := by
  induction rounds with
  | nil => rfl
  | cons o rounds ih =>
    unfold codesOf specHistOf at ih ⊢
    rw [List.filterMap_cons, List.filterMap_cons]
    rcases h : o.result.ending with _ | ⟨s, ℓ⟩ | _ <;>
      simp [popCode, SpecRound.sampler, h, ih]

omit [IsProbabilityMeasure μ] in
lemma idealRounds_hyp (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (draws : Fin K → LoopRoundDraws Ω α settings) (r : ℕ) :
    ∀ o ∈ idealRounds settings O D Dsf draws r, EdgesFin o.result.hypothesis.edges
      ∧ Random.MidsLe settings.level3.probeBudget o.result.hypothesis.tree := by
  intro o ho
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
  obtain ⟨hK, e⟩ := getElem_idealRounds settings O D Dsf draws r i hi
  rw [e]
  exact lazyRound_hyp _ _ _ _ _ _

omit [IsProbabilityMeasure μ] in
lemma codesOf_idealRounds_succ (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {N : ℕ} (hN : L + settings.level3.probeBudget ≤ N)
    (draws : Fin K → LoopRoundDraws Ω α settings) (r : ℕ) (hrK : r < K) :
    codesOf N (idealRounds settings O D Dsf draws (r + 1))
      = if (codesOf N (idealRounds settings O D Dsf draws r)).length = r then
          codesOf N (idealRounds settings O D Dsf draws r)
            ++ (popCode N (idealRound settings O D Dsf
              (decodeHist (codesOf N (idealRounds settings O D Dsf draws r)))
              (draws ⟨r, hrK⟩))).toList
        else codesOf N (idealRounds settings O D Dsf draws r) := by
  have hlenc := length_codesOf N (idealRounds settings O D Dsf draws r)
  rw [idealRounds_succ]
  split_ifs with h1 h2 h2
  · rw [codesOf_append_single, idealRound_alike settings O D Dsf hlen
      (decodeHist_alike hN _ (idealRounds_hyp settings O D Dsf draws r) h1.2)]
  · exact absurd (by rw [hlenc, length_specHistOf_pops h1.2,
      length_eq_of_allHarvested settings O D Dsf draws hrK.le h1.2]) h2
  · exfalso
    apply h1
    refine ⟨hrK, fun o ho => ?_⟩
    have hle := length_idealRounds_le settings O D Dsf draws r
    have hlen' : ((idealRounds settings O D Dsf draws r).filterMap fun o => o.sampler).length
        = (idealRounds settings O D Dsf draws r).length := by
      have := List.length_filterMap_le (fun o : SpecRound α => o.sampler)
        (idealRounds settings O D Dsf draws r)
      have h3 : ((idealRounds settings O D Dsf draws r).filterMap fun o => o.sampler).length
          = r := (hlenc.symm.trans h2 :)
      omega
    have := (List.filterMap_length_eq_length.1 hlen') o ho
    rcases he : o.result.ending with _ | ⟨s, ℓ⟩ | _
    · simp [SpecRound.sampler, he] at this
    · trivial
    · simp [SpecRound.sampler, he] at this
  · rfl

/-- The earlier rounds' harvest data are a measurable function of the rounds' draws. -/
theorem measurableSet_codesOf (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) {N : ℕ} (hN : L + settings.level3.probeBudget ≤ N)
    (r : ℕ) (c : List (HarvestCode α)) :
    MeasurableSet {draws : Fin K → LoopRoundDraws Ω α settings |
      codesOf N (idealRounds settings O D Dsf draws r) = c} := by
  classical
  induction r generalizing c with
  | zero =>
    simp only [idealRounds, codesOf, List.filterMap_nil]
    exact MeasurableSet.const _
  | succ r ih =>
    by_cases hrK : r < K
    · have hset : {draws : Fin K → LoopRoundDraws Ω α settings |
          codesOf N (idealRounds settings O D Dsf draws (r + 1)) = c}
          = ⋃ c', {draws | codesOf N (idealRounds settings O D Dsf draws r) = c'}
            ∩ (fun draws : Fin K → LoopRoundDraws Ω α settings =>
              draws ⟨r, hrK⟩) ⁻¹' {d | (if c'.length = r then
                c' ++ (popCode N (idealRound settings O D Dsf
                  (decodeHist c') d)).toList else c') = c} := by
        ext draws
        simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage,
          codesOf_idealRounds_succ settings O D Dsf hlen hN draws r hrK]
        exact ⟨fun h => ⟨_, rfl, h⟩, fun ⟨c', hc', h⟩ => hc' ▸ h⟩
      rw [hset]
      refine MeasurableSet.iUnion fun c' => (ih c').inter (measurable_pi_apply _ ?_)
      by_cases hc : c'.length = r
      · simp only [hc, if_true]
        have : {d : LoopRoundDraws Ω α settings |
            c' ++ (popCode N (idealRound settings O D Dsf
              (decodeHist c') d)).toList = c}
            = ⋃ oc : Option (HarvestCode α),
              if c' ++ oc.toList = c then
                {d | popCode N (idealRound settings O D Dsf
                  (decodeHist c') d) = oc} else ∅ := by
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
        · exact measurableSet_roundCode settings O N D Dsf _ oc
        · exact MeasurableSet.empty
      · simp only [hc, if_false]
        exact MeasurableSet.const _
    · have hst : ∀ draws : Fin K → LoopRoundDraws Ω α settings,
          idealRounds settings O D Dsf draws (r + 1) = idealRounds settings O D Dsf draws r :=
        fun draws => by rw [idealRounds_succ, dif_neg (fun h => hrK h.1)]
      simp only [hst]
      exact ih c

end Rounds

end OrthoDFA
