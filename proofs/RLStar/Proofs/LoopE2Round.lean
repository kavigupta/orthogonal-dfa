import RLStar.Proofs.LoopE2

/-!
# Round `r`'s lazy round, given the rounds before it

What reaches round `r`'s lazy round is the earlier rounds' data and its clustering's state,
family and prefixes (`stage4`), fixed by the answers the earlier rounds and the clustering read.
Off a collision the lazy round reads none of those strings, so it fails its claim with chance at
most `lazyRoundBound` (`loop_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Stage

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))

/-- Some state of the schedule returns. -/
def Returns (h : SpecHist α) (c : ClusterDraws Ω α K) : Prop :=
  ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
    idealRun D Dsf h c ∈ ret O.mq
        (loopPopulations K h.pops.length) none
      settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B

open Classical in
/-- What reaches round `r`'s lazy round: the earlier rounds' data and the round's state, family and
prefixes, where every earlier clustering and this one returned. -/
def stage4 (r : ℕ) (hr : r < K) (ω : Ω) (g : Fin K → settings.RoundDraws α) :
    Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α)) :=
  if HistOK settings O D Dsf ω g r ∧ (histCodes settings O D Dsf ω g r).length = r
      ∧ Returns settings O D Dsf
        (decodeHist (histCodes settings O D Dsf ω g r))
        (ω, (g ⟨r, hr⟩).1) then
    some (histCodes settings O D Dsf ω g r,
      roundState settings O D Dsf
        (decodeHist (histCodes settings O D Dsf ω g r))
        (ω, (g ⟨r, hr⟩).1),
      clusterAt O.mq (loopPopulations K
          (decodeHist (histCodes settings O D Dsf ω g r)).pops.length)
        (idealRun D Dsf
          (decodeHist (histCodes settings O D Dsf ω g r))
          (ω, (g ⟨r, hr⟩).1))
        (roundState settings O D Dsf
          (decodeHist (histCodes settings O D Dsf ω g r))
          (ω, (g ⟨r, hr⟩).1)),
      (roundSets settings D Dsf
        (decodeHist (histCodes settings O D Dsf ω g r))
        (ω, (g ⟨r, hr⟩).1)).1)
  else none

omit [IsProbabilityMeasure μ] in
lemma stage4_eq_some (r : ℕ) (hr : r < K) (ω : Ω)
    (g : Fin K → settings.RoundDraws α) (c :
        List (HarvestCode α)) (B : State)
    (F K₀ : Finset (FreeMonoid α)) :
    stage4 settings O D Dsf r hr ω g = some (c, B, F, K₀) ↔
      HistOK settings O D Dsf ω g r ∧ histCodes settings O D Dsf ω g r = c ∧ c.length = r
        ∧ Returns settings O D Dsf (decodeHist c) (ω, (g ⟨r, hr⟩).1)
        ∧ roundState settings O D Dsf (decodeHist c)
            (ω, (g ⟨r, hr⟩).1) = B
        ∧ clusterAt O.mq (loopPopulations K (decodeHist c).pops.length)
            (idealRun D Dsf (decodeHist c)
                (ω, (g ⟨r, hr⟩).1)) B = F
        ∧ (roundSets settings D Dsf (decodeHist c)
            (ω, (g ⟨r, hr⟩).1)).1 = K₀ := by
  unfold stage4
  split_ifs with h
  · simp only [Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨rfl, rfl, rfl, rfl⟩
      exact ⟨h.1, rfl, h.2.1, h.2.2, rfl, rfl, rfl⟩
    · rintro ⟨-, rfl, -, -, rfl, rfl, rfl⟩
      exact ⟨rfl, rfl, rfl, rfl⟩
  · simp only [reduceCtorEq, false_iff]
    rintro ⟨h1, rfl, h3, h4, -⟩
    exact h ⟨h1, h3, h4⟩

omit [IsProbabilityMeasure μ] in
lemma stage4_eq_none (r : ℕ) (hr : r < K) (ω : Ω)
    (g : Fin K → settings.RoundDraws α) :
    stage4 settings O D Dsf r hr ω g = none ↔
      ¬ (HistOK settings O D Dsf ω g r ∧ (histCodes settings O D Dsf ω g r).length = r
        ∧ Returns settings O D Dsf (decodeHist
          (histCodes settings O D Dsf ω g r)) (ω, (g ⟨r, hr⟩).1)) := by
  unfold stage4
  split_ifs with h
  · simp only [reduceCtorEq, false_iff, not_not]; exact h
  · simp only [true_iff]; exact h

omit [IsProbabilityMeasure μ] in
lemma stage4_congr_draws (r : ℕ) (hr : r < K) (ω : Ω)
    (g g' : Fin K → settings.RoundDraws α)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) (h1 : (g ⟨r, hr⟩).1 = (g' ⟨r, hr⟩).1) :
    stage4 settings O D Dsf r hr ω g = stage4 settings O D Dsf r hr ω g' := by
  unfold stage4
  rw [histCodes_congr_draws settings O D Dsf r ω g g' h, h1]
  have hok := histOK_congr_draws settings O D Dsf ω g g' r h
  split_ifs with ha hb hb
  · rfl
  · exact absurd ⟨hok.1 ha.1, ha.2⟩ hb
  · exact absurd ⟨hok.2 hb.1, hb.2⟩ ha
  · rfl

lemma measurableSet_returns [IsProbabilityMeasure D] (r : ℕ) (hr : r < K) (h : SpecHist α) :
    MeasurableSet {p : Ω × (Fin K → settings.RoundDraws α) |
      Returns settings O D Dsf h (p.1, (p.2 ⟨r, hr⟩).1)} := by
  classical
  have hset : {p : Ω × (Fin K → settings.RoundDraws α) |
      Returns settings O D Dsf h (p.1, (p.2 ⟨r, hr⟩).1)}
      = ⋃ B ∈ settings.schedule (loopPopulations K h.pops.length),
        (fun p : Ω × (Fin K → settings.RoundDraws α) =>
          idealRun D Dsf h (p.1, (p.2 ⟨r,
              hr⟩).1)) ⁻¹'
        ret O.mq (loopPopulations K h.pops.length) none
          settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B := by
    ext p
    simp [Returns]
  rw [hset]
  exact Finset.measurableSet_biUnion _ fun B _ =>
    measurable_idealRun_at settings D Dsf h ⟨r, hr⟩ (measurableSet_ret O _ none _ _ _ _ B)

lemma measurableSet_stage4 [IsProbabilityMeasure D]
    (r : ℕ) (hr : r < K)
    (y : Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))) :
    MeasurableSet {p : Ω × (Fin K → settings.RoundDraws α) |
      stage4 settings O D Dsf r hr p.1 p.2 = y} := by
  classical
  have hhc := measurableSet_histCodes settings O D Dsf r
  have hok := measurableSet_histOK settings O D Dsf r
  set dec := decodeHist (α := α)
  rcases y with _ | ⟨c, B, F, K₀⟩
  · have hset : {p : Ω × (Fin K → settings.RoundDraws α) |
        stage4 settings O D Dsf r hr p.1 p.2 = none}
        = ({p | HistOK settings O D Dsf p.1 p.2 r} ∩ ⋃ c : List (HarvestCode α),
          ({p | histCodes settings O D Dsf p.1 p.2 r = c} ∩ {_p | c.length = r}
            ∩ {p | Returns settings O D Dsf (dec c) (p.1, (p.2 ⟨r, hr⟩).1)}))ᶜ := by
      ext p
      simp only [Set.mem_ofPred_eq, stage4_eq_none, Set.mem_compl_iff, Set.mem_inter_iff,
        Set.mem_iUnion]
      constructor
      · rintro h ⟨h1, c, ⟨rfl, h2⟩, h3⟩
        exact h ⟨h1, h2, h3⟩
      · rintro h ⟨h1, h2, h3⟩
        exact h ⟨h1, _, ⟨rfl, h2⟩, h3⟩
    rw [hset]
    exact (hok.inter (MeasurableSet.iUnion fun c => ((hhc c).inter (MeasurableSet.const _)).inter
      (measurableSet_returns settings O D Dsf r hr (dec c)))).compl
  · set H := dec c
    have hφ : Measurable fun p : Ω × (Fin K → settings.RoundDraws α) =>
        ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K) :=
      measurable_fst.prodMk (measurable_fst.comp ((measurable_pi_apply _).comp measurable_snd))
    have hset : {p : Ω × (Fin K → settings.RoundDraws α) |
        stage4 settings O D Dsf r hr p.1 p.2 = some (c, B, F, K₀)}
        = {p | HistOK settings O D Dsf p.1 p.2 r}
          ∩ {p | histCodes settings O D Dsf p.1 p.2 r = c}
          ∩ {_p | c.length = r}
          ∩ {p | Returns settings O D Dsf H (p.1, (p.2 ⟨r, hr⟩).1)}
          ∩ (fun p : Ω × (Fin K → settings.RoundDraws α) =>
              ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K)) ⁻¹'
            ({cd | roundState settings O D Dsf H cd = B
              ∧ clusterAt O.mq (loopPopulations K H.pops.length) (idealRun D Dsf H cd)
                (roundState settings O D Dsf H cd) = F}
            ∩ {cd | (roundSets settings D Dsf H cd).1 = K₀}) := by
      ext p
      simp only [Set.mem_ofPred_eq, stage4_eq_some, Set.mem_inter_iff, Set.mem_preimage]
      constructor
      · rintro ⟨h1, h2, h3, h4, h5, h6, h7⟩
        exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, ⟨h5, by rw [h5]; exact h6⟩, h7⟩
      · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, ⟨h5, h6⟩, h7⟩
        exact ⟨h1, h2, h3, h4, h5, by rw [← h5]; exact h6, h7⟩
    have hCP : MeasurableSet
        {cd : ClusterDraws Ω α K | (roundSets settings D Dsf H cd).1 = K₀} :=
      measurable_idealRun (Ω := Ω) (K := K) D Dsf H (measurableSet_roundPrefs
        (settings.schedule (loopPopulations K H.pops.length)) (loopPopulations K H.pops.length) K₀)
    rw [hset]
    exact ((((hok.inter (hhc c)).inter (MeasurableSet.const _)).inter
      (measurableSet_returns settings O D Dsf r hr H)).inter
      (hφ ((measurableSet_roundPick settings O D Dsf H B F).inter hCP)))

omit [IsProbabilityMeasure μ] in
lemma roundKeys_eq_prefsOf (h : SpecHist α) (ω : Ω) (c : ClusterRest α K) :
    roundKeys settings D Dsf h (ω, c)
      = (prefsOf settings D Dsf h c).image fun p => prefixOf p settings.level3.walkStart :=
  rfl

/-- Where the clustering's state test returned, the answers at the strings the earlier rounds and
it read fix what reaches level 3. -/
lemma stage4_stop [IsProbabilityMeasure D]
    (r : ℕ) (hr : r < K) (c : List (HarvestCode α)) (B : State) (F K₀ : Finset (FreeMonoid α))
    (g : Fin K → settings.RoundDraws α) (ω ω' ω₀
        : Ω)
    (hs : stage4 settings O D Dsf r hr ω g = some (c, B, F, K₀))
    (h : ∀ w ∈ histReads settings D Dsf c g r ω₀
      ∪ schedReads (settings.schedule (loopPopulations K
          (decodeHist c).pops.length))
        (loopPopulations K (decodeHist c).pops.length)
        ((ω₀, runOf settings D Dsf (decodeHist c) (g ⟨r, hr⟩).1) :
          Run Ω (FreeMonoid α) (Option (Fin K))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    stage4 settings O D Dsf r hr ω' g = some (c, B, F, K₀) := by
  classical
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ :=
    (stage4_eq_some settings O D Dsf r hr ω g c B F K₀).1 hs
  have hst : stageH settings O D Dsf r ω g = some c := by
    simp only [stageH, if_pos h1, h2]
  have hst' := stageH_stop settings O D Dsf r c g ω ω' ω₀ hst
    fun w hw => h w (Finset.mem_union_left _ hw)
  have hok' : HistOK settings O D Dsf ω' g r := by
    by_contra hn
    simp only [stageH, if_neg hn, reduceCtorEq] at hst'
  have hc' : histCodes settings O D Dsf ω' g r = c := by
    simp only [stageH, if_pos hok', Option.some.injEq] at hst'
    exact hst'
  set H := decodeHist c
  set sched := settings.schedule (loopPopulations K H.pops.length)
  have hsr : ∀ w ∈ schedReads sched (loopPopulations K H.pops.length)
      (idealRun D Dsf H (ω, (g ⟨r,
          hr⟩).1)), (O.mq w ω = 1 ↔ O.mq w ω' = 1) :=
    fun w hw => h w (Finset.mem_union_right _ hw)
  have hret' : Returns settings O D Dsf H (ω', (g ⟨r, hr⟩).1) := by
    obtain ⟨B', hB', hret⟩ := h4
    exact ⟨B', hB', (ret_congr O sched (loopPopulations K H.pops.length) none
      (Finset.mem_insert_self _ _) _ _ _ _ hB' _ hsr).1 hret⟩
  have hrs : roundState settings O D Dsf H (ω', (g ⟨r, hr⟩).1) = B := by
    rw [← h5]
    exact (roundState_congr settings O D Dsf H _ hsr).symm
  have hBs : B ∈ sched := by
    rw [← h5]
    exact (Classical.epsilon_spec h4).1
  refine (stage4_eq_some settings O D Dsf r hr ω' g c B F K₀).2
    ⟨hok', hc', h3, hret', hrs, ?_, by rw [← h7]; rfl⟩
  rw [← h6]
  exact (clusterAt_congr_sched O sched (loopPopulations K H.pops.length) hBs _ hsr).symm

end Stage

lemma clusterAt_subset_suffsOf (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α) (ω : Ω) (c : ClusterRest α K) {B : State}
    (hB : B ∈ settings.schedule (loopPopulations K h.pops.length)) :
    clusterAt O.mq (loopPopulations K h.pops.length) (idealRun D Dsf h (ω, c)) B
      ⊆ suffsOf settings D Dsf h c := by
  intro v hv
  have hv1 : v ≠ 1 := (Finset.mem_erase.1 hv).1
  rw [clusterAt_eq, clusterAt_eq_clusterOf] at hv
  have := poolAt_sub_roundSuffs hB _ (clusterOf_subset O B.sc B.scd _ _ B.k ω
    (Finset.mem_insert_self _ _) hv)
  rcases Finset.mem_insert.1 this with h1 | h1
  · exact absurd h1 hv1
  · exact Finset.mem_erase.2 ⟨hv1, h1⟩

section Round

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf] (settings : LoopSettings K)

include hL in
/-- Round `r`'s lazy round, where every earlier clustering and this one returned and nothing
collides, fails its claim with chance at most `lazyRoundBound`. -/
theorem loop_round_le_persist (Ls NP : ℕ) (m crossLimit δH tw : ℝ) (Nr : ℕ)
    (hNP : ∀ pops, ∑ n ∈ (settings.schedule pops).image State.npref, n ≤ 2 * NP)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
        RarerBounded N B.lo (B.hi + 1) p r crossLimit)
    (hk : 1 ≤ settings.level3.walkStart)
    (hKw : ∀ᵐ x ∂P.D, settings.level3.walkStart ≤ x.toList.length)
    (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : LazyPremises σ α settings.level3 settings.listSize settings.epochs settings.blocks
      (2 * prefixMax P.D settings.level3.walkStart))
    (hNr : lazyReadCap settings.level3 settings.epochs settings.blocks settings.blockSize P.L
      (Fintype.card σ) (Fintype.card α) ≤ Nr)
    (hδH : 0 < δH) (htw : 0 < tw)
    (hm : 6 * (K + 1) * NP * prefixMax P.D settings.level3.walkStart ≤ m)
    (hm1 : m ≤ 1 / 2)
    (r : ℕ) (hr : r < K) :
    (settings.measure μ α).real
      {p | HistOK settings O P.D Dsf p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (decodeHist
            (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ RoundSpec M O P (lazySlope σ Nr settings.listSize) (lazySlack σ settings.listSize δH)
            (settings.level3.disagreementRate + tw + m) settings.level3.walkStart
            (idealRound settings O P.D Dsf (decodeHist
              (histCodes settings O P.D Dsf p.1 p.2 r)) (p.1, p.2 ⟨r, hr⟩))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf p.1 p.2 r) p.2 r hr}
      ≤ lazyRoundBound σ settings.level3 settings.epochs settings.blocks
          settings.blockSize crossLimit Nr P.L δH tw := by
  set δb := lazyRoundBound σ settings.level3 settings.epochs settings.blocks settings.blockSize
    crossLimit Nr P.L δH tw with hδb
  have hδb0 : 0 ≤ δb := by
    obtain ⟨-, -, hy1, hδ0, -⟩ := hprem
    have h1 : 0 ≤ 1 - settings.level3.harvestRate / 4 := by linarith
    have h2 : (0 : ℝ) ≤ settings.epochs * settings.blocks
        * (1 - settings.level3.harvestRate / 4) ^ settings.blockSize := by positivity
    have h3 : (0 : ℝ) ≤ crossLimit * Nr := by positivity
    have h4 : (0 : ℝ) ≤ (P.L + 1) * (Fintype.card σ + 2) * crossLimit / tw := by positivity
    simp only [hδb, lazyRoundBound]
    linarith
  classical
  set T := settings.level3.probeBudget
  set k := settings.level3.walkStart
  set dec := decodeHist (α := α) with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set rK : Fin K := ⟨r, hr⟩
  set ν₂ := lazyLaw settings.epochs T settings.blocks settings.blockSize
  set ν := settings.roundLaw α
  have hσ : MeasurePreserving (id : settings.RoundDraws α → settings.RoundDraws α) ν
      ((restLaw α K).prod ν₂) :=
    MeasurePreserving.id _
  set Y := Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set Good : List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α) → Prop :=
    fun t => t.1.length = r ∧ t.2.1 ∈ settings.schedule (loopPopulations K (dec t.1).pops.length)
      ∧ SuffixFree t.2.2.1 ∧ t.2.2.1.card + 1 ≤ t.2.1.k ∧ t.2.2.2.card ≤ 6 * (K + 1) * NP
    with hGood
  have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
  have hex : ∀ t, Good t → ∃ Bd : Set (Ω × settings.Uniforms), MeasurableSet Bd
      ∧ (μ.prod ν₂) Bd ≤ ENNReal.ofReal δb
      ∧ (∀ v ω ω', (∀ w ∈ lazyZone (max (P.L + T + 2)
          (lazyProbesLen (probesOf settings P.D (t.2.2.2.image fun p => prefixOf p k) v)
            + 1 + T + 2)) k t.2.2.2 t.2.2.1, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, v) ∈ Bd → (ω', v) ∈ Bd)
      ∧ ∀ Sf ω v, (ω, v) ∉ Bd →
        RoundSpec M O P (lazySlope σ Nr settings.listSize) (lazySlack σ settings.listSize δH)
          (settings.level3.disagreementRate + tw + m) k
          (mkRound settings O P.D t.2.1 t.2.2.1 (t.2.2.2.image fun p => prefixOf p k)
            t.2.2.2 Sf (ω, v)) :=
    fun t ht => lazy_round_bad_exists M O hL P settings m crossLimit δH tw Nr hk hKw hη1 hcross
      hprem hNr hδH htw hm1 t.2.1 t.2.2.1 t.2.2.2 ht.2.2.1 ht.2.2.2.1
      ((hband _ _ ht.2.1).2.2) (by
        have h1 : ((t.2.2.2.image fun p => prefixOf p k).card : ℝ) ≤ 6 * (K + 1) * NP := by
          exact_mod_cast Finset.card_image_le.trans ht.2.2.2.2
        exact (mul_le_mul_of_nonneg_right h1 hpm0).trans hm)
  set Bad : Y → Set (Ω × settings.Uniforms) := fun y => match y with
    | some t => if h : Good t then Classical.choose (hex t h) else ∅
    | none => ∅ with hBad
  set W : Y → (Fin K → settings.RoundDraws α) → Finset (FreeMonoid α) :=
      fun y x => match y with
    | some t => histReads settings P.D Dsf t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (dec t.1).pops.length))
          (loopPopulations K (dec t.1).pops.length)
          ((ω₀, runOf settings P.D Dsf (dec t.1) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K)))
    | none => ∅
  set G : Y → Prop := fun y => ∃ t, y = some t ∧ t.1.length = r
  set Z : Y → settings.Uniforms → Finset (FreeMonoid α) := fun y v => match y with
    | some t => lazyZone (max (P.L + T + 2)
          (lazyProbesLen (probesOf settings P.D (t.2.2.2.image fun p => prefixOf p k) v)
            + 1 + T + 2)) k t.2.2.2 t.2.2.1
    | none => ∅
  set Dj : Y → Set (Fin K → settings.RoundDraws α) := fun y => match y with
    | some t => {x | NoColl settings P.D Dsf Ls t.1 x r hr
        ∧ t.2.2.1 ⊆ suffsOf settings P.D Dsf (histAt t.1 r) (x rK).1
        ∧ t.2.2.2 = prefsOf settings P.D Dsf (histAt t.1 r) (x rK).1}
    | none => Set.univ
  have hmain := persist_stage_le O ν (restLaw α K) ν₂ id hσ id measurable_id
    (fun w => rfl) rK (stage4 settings O P.D Dsf r hr)
    (fun ω x x' hxx' h1 => stage4_congr_draws settings O P.D Dsf r hr ω x x'
      (fun i hi => hxx' i hi) h1)
    (measurableSet_stage4 settings O P.D Dsf r hr) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, -⟩ := hG
      exact stage4_stop settings O P.D Dsf r hr c B F K₀ x ω ω' ω₀ hs h)
    Bad (fun y hy => by
      rcases y with _ | t
      · rfl
      · show (if h : Good t then Classical.choose (hex t h) else ∅) = ∅
        rw [dif_neg fun h => hy ⟨t, rfl, h.1⟩])
    (fun y => by
      rcases y with _ | t
      · exact MeasurableSet.empty
      · show MeasurableSet (if h : Good t then Classical.choose (hex t h) else ∅)
        split_ifs with h
        · exact (Classical.choose_spec (hex t h)).1
        · exact MeasurableSet.empty)
    Z (fun y v ω ω' h hq => by
      rcases y with _ | t
      · exact hq
      · change (ω, v) ∈ (if h : Good t then Classical.choose (hex t h) else ∅) at hq
        show (ω', v) ∈ (if h : Good t then Classical.choose (hex t h) else ∅)
        split_ifs at hq ⊢ with hg
        · exact (Classical.choose_spec (hex t hg)).2.2.1 v ω ω' h hq
        · exact hq)
    Dj (fun y => by
      rcases y with _ | ⟨c, B, F, CP⟩
      · exact MeasurableSet.univ
      · set h := histAt c r
        have hx : Measurable fun x : Fin K → settings.RoundDraws α => (x rK).1 :=
          measurable_fst.comp (measurable_pi_apply rK)
        have h2 : MeasurableSet {x : Fin K → settings.RoundDraws α |
            F ⊆ suffsOf settings P.D Dsf h (x rK).1} := by
          have := measurableSet_of_data (fun x : Fin K → settings.RoundDraws α =>
              suffsOf settings P.D Dsf h (x rK).1)
            (fun Q => hx (measurableSet_suffsOf settings P.D Dsf h Q))
            (fun Q => if F ⊆ Q then Set.univ else ∅) (fun Q => by split_ifs <;> simp)
          convert this using 1
          ext x
          simp only [Set.mem_setOf_eq]
          split_ifs with h'
          · exact iff_of_true h' (Set.mem_univ x)
          · exact iff_of_false h' (Set.notMem_empty x)
        have h3 : MeasurableSet {x : Fin K → settings.RoundDraws α |
            CP = prefsOf settings P.D Dsf h (x rK).1} :=
          hx (measurableSet_prefsOf settings P.D Dsf h CP) |> fun hm => by
            convert hm using 1
            ext x
            exact eq_comm
        exact (measurableSet_noColl settings P.D Dsf Ls c r hr).inter (h2.inter h3))
    (fun y x hG hx => by
      obtain ⟨⟨c, B, F, CP⟩, rfl, hcr⟩ := hG
      obtain ⟨hnc, hF, hCP⟩ := hx
      have hpop : histAt c r = dec c := by
        simp only [histAt, List.take_of_length_le (le_of_eq hcr), hdec] <;> rfl
      set N' := max (P.L + T + 2)
        (lazyProbesLen (probesOf settings P.D (CP.image fun p => prefixOf p k) (x rK).2)
          + 1 + T + 2)
      have hdis := disjoint_hist_family settings P.D Dsf Ls N' c x r hr ω₀ hnc F hF
      rw [hpop] at hdis
      refine hdis.mono_right fun w hw => ?_
      obtain ⟨hrs, z, hz, -, v, hv, rfl⟩ := Finset.mem_filter.1 hw
      refine Finset.mem_sdiff.2 ⟨hrs, fun hw' => ?_⟩
      obtain ⟨⟨p, v'⟩, hpv, he⟩ := Finset.mem_image.1 hw'
      obtain ⟨hp, hv'⟩ := Finset.mem_product.1 hpv
      have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu =>
        hnc.1 rK le_rfl u (hF hu)
      obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he.symm ((hlenF v hv).trans (hlenF v' hv').symm)
      rw [hpop] at hCP
      rw [← hCP] at hp
      subst hzp
      obtain ⟨i, -, hiz, hne⟩ := hz
      exact hne _ hp hiz rfl)
    (ENNReal.ofReal δb) (fun y => by
      rcases y with _ | t
      · show (μ.prod _) ∅ ≤ _
        simp
      · show (μ.prod _) (if h : Good t then Classical.choose (hex t h) else ∅) ≤ _
        split_ifs with h
        · exact (Classical.choose_spec (hex t h)).2.1
        · simp)
  refine ENNReal.toReal_le_of_le_ofReal hδb0 ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hret, hnot, hnc⟩
  set c := histCodes settings O P.D Dsf p.1 p.2 r
  set H := dec c
  set cd : ClusterDraws Ω α K := (p.1, (p.2 rK).1)
  set B := roundState settings O P.D Dsf H cd
  set F := clusterAt O.mq (loopPopulations K H.pops.length) (idealRun P.D Dsf H cd) B
  set CP := (roundSets settings P.D Dsf H cd).1
  have hst : stage4 settings O P.D Dsf r hr p.1 p.2 = some (c, B, F, CP) :=
    (stage4_eq_some settings O P.D Dsf r hr p.1 p.2 c B F CP).2
      ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
  have hBs : B ∈ settings.schedule (loopPopulations K H.pops.length) :=
    (Classical.epsilon_spec hret).1
  have hpop : histAt c r = H := by
    simp only [histAt, List.take_of_length_le (le_of_eq hlen), H, hdec] <;> rfl
  have hFs : F ⊆ suffsOf settings P.D Dsf (histAt c r) (p.2 rK).1 := by
    rw [hpop]
    exact clusterAt_subset_suffsOf settings O P.D Dsf H p.1 (p.2 rK).1 hBs
  have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu => hnc.1 rK le_rfl u (hFs hu)
  have hCP : CP = prefsOf settings P.D Dsf H (p.2 rK).1 := rfl
  have hgood : Good (c, B, F, CP) := by
    refine ⟨hlen, hBs, ?_, ?_, ?_⟩
    · intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    · obtain ⟨-, hk2, -⟩ := hband _ B hBs
      have h1 := clusterAt_card_add_one (rule := nearestClusterer B.centre) O
        (loopPopulations K H.pops.length) B (idealRun P.D Dsf H cd)
      have h2 := (nearestClusterer B.centre).card_le (fun w => O.mq w (oracleNoise (idealRun P.D Dsf
          H cd)) = 1)
        (prefixesAt (loopPopulations K H.pops.length) B.npref (idealRun P.D Dsf H cd))
        (screenedAt O.mq (loopPopulations K H.pops.length) B (idealRun P.D Dsf H cd)) B.k
        (by omega)
      show F.card + 1 ≤ B.k
      simp only [F]
      rw [clusterAt_eq]
      unfold clusterFull at h1
      omega
    · show CP.card ≤ 6 * (K + 1) * NP
      refine (card_roundPrefs_le _ _ _).trans ?_
      have hpops : (loopPopulations K H.pops.length).card ≤ K + 1 := by
        have := Finset.card_le_univ (loopPopulations K H.pops.length)
        simpa using this
      have := hNP (loopPopulations K H.pops.length)
      calc 3 * (loopPopulations K H.pops.length).card
            * ∑ n ∈ (settings.schedule (loopPopulations K H.pops.length)).image State.npref, n
          ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) this
        _ = 6 * (K + 1) * NP := by ring
  show (p.1, (id (p.2 rK)).2) ∈ Bad (stage4 settings O P.D Dsf r hr p.1 p.2)
    ∧ p.2 ∈ Dj (stage4 settings O P.D Dsf r hr p.1 p.2)
  rw [hst]
  refine ⟨?_, hnc, hFs, by rw [hpop]; exact hCP⟩
  show (p.1, (p.2 rK).2) ∈ (if h : Good (c, B, F, CP) then Classical.choose (hex _ h) else ∅)
  rw [dif_pos hgood]
  by_contra hnb
  exact hnot ((Classical.choose_spec (hex _ hgood)).2.2.2
    (roundSets settings P.D Dsf H cd).2 p.1 (p.2 rK).2 hnb)

end Round

end OrthoDFA
