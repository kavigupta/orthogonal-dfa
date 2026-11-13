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
    have hCP : MeasurableSet {cd : ClusterDraws Ω α K | (roundSets settings D Dsf H cd).1 = K₀} := by
      sorry -- closes by: measurableSet_roundPrefs through measurable_idealRun
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
    (hprem : LazyPremises σ α settings (2 * prefixMax P.D settings.level3.walkStart))
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
      ≤ lazyRoundBound σ settings crossLimit Nr P.L δH tw := by
  sorry -- closes by: main's proof, hex from lazy_round_bad_exists at stage4's (B, F, CP)

end Round

end OrthoDFA
