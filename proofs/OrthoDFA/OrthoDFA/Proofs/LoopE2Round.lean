import OrthoDFA.Proofs.LoopE2

/-!
# Round `r`'s level 3, given the rounds before it

What reaches round `r`'s level 3 is the earlier rounds' data and its clustering's state, family
and keys (`stage4`), fixed by the answers the earlier rounds and the clustering read. Off a
collision level 3 reads none of those strings, so it fails its claim with chance at most `δ`
(`loop_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Stage

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))
  (N : ℕ)

/-- Some state of the schedule returns. -/
def Returns (h : LoopHist α) (c : ClusterDraws Ω α K) : Prop :=
  ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
    clusterRun D Dsf h c ∈ ret O.mq (loopPopulations K h.pops.length) none
      settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B

open Classical in
/-- What reaches round `r`'s level 3: the earlier rounds' data and the round's state, family and
keys, where every earlier clustering and this one returned. -/
def stage4 (r : ℕ) (hr : r < K) (ω : Ω) (g : Fin K → RoundDraws α K settings.level3.probeBudget) :
    Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α)) :=
  if HistOK settings O D Dsf N ω g r ∧ (histCodes settings O D Dsf N ω g r).length = r
      ∧ Returns settings O D Dsf (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N ω g r)) (ω, (g ⟨r, hr⟩).1) then
    some (histCodes settings O D Dsf N ω g r,
      roundState settings O D Dsf (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N ω g r)) (ω, (g ⟨r, hr⟩).1),
      clusterAt O.mq (loopPopulations K (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N ω g r)).pops.length) (clusterRun D Dsf (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N ω g r)) (ω, (g ⟨r, hr⟩).1))
        (roundState settings O D Dsf (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N ω g r)) (ω, (g ⟨r, hr⟩).1)),
      roundKeys settings D Dsf (decodeHist D settings.level3.walkStart (histCodes settings O D Dsf N ω g r)) (ω, (g ⟨r, hr⟩).1))
  else none

omit [IsProbabilityMeasure μ] in
lemma stage4_eq_some (r : ℕ) (hr : r < K) (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level3.probeBudget) (c : List (HarvestCode α)) (B : State)
    (F K₀ : Finset (FreeMonoid α)) :
    stage4 settings O D Dsf N r hr ω g = some (c, B, F, K₀) ↔
      HistOK settings O D Dsf N ω g r ∧ histCodes settings O D Dsf N ω g r = c ∧ c.length = r
        ∧ Returns settings O D Dsf (decodeHist D settings.level3.walkStart c) (ω, (g ⟨r, hr⟩).1)
        ∧ roundState settings O D Dsf (decodeHist D settings.level3.walkStart c)
            (ω, (g ⟨r, hr⟩).1) = B
        ∧ clusterAt O.mq (loopPopulations K (decodeHist D settings.level3.walkStart c).pops.length)
            (clusterRun D Dsf (decodeHist D settings.level3.walkStart c) (ω, (g ⟨r, hr⟩).1)) B = F
        ∧ roundKeys settings D Dsf (decodeHist D settings.level3.walkStart c)
            (ω, (g ⟨r, hr⟩).1) = K₀ := by
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
    (g : Fin K → RoundDraws α K settings.level3.probeBudget) :
    stage4 settings O D Dsf N r hr ω g = none ↔
      ¬ (HistOK settings O D Dsf N ω g r ∧ (histCodes settings O D Dsf N ω g r).length = r
        ∧ Returns settings O D Dsf (decodeHist D settings.level3.walkStart
          (histCodes settings O D Dsf N ω g r)) (ω, (g ⟨r, hr⟩).1)) := by
  unfold stage4
  split_ifs with h
  · simp only [reduceCtorEq, false_iff, not_not]; exact h
  · simp only [true_iff]; exact h

omit [IsProbabilityMeasure μ] in
lemma stage4_congr_draws (r : ℕ) (hr : r < K) (ω : Ω)
    (g g' : Fin K → RoundDraws α K settings.level3.probeBudget)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) (h1 : (g ⟨r, hr⟩).1 = (g' ⟨r, hr⟩).1) :
    stage4 settings O D Dsf N r hr ω g = stage4 settings O D Dsf N r hr ω g' := by
  unfold stage4
  rw [histCodes_congr_draws settings O D Dsf N r ω g g' h, h1]
  have hok := histOK_congr_draws settings O D Dsf N ω g g' r h
  split_ifs with ha hb hb
  · rfl
  · exact absurd ⟨hok.1 ha.1, ha.2⟩ hb
  · exact absurd ⟨hok.2 hb.1, hb.2⟩ ha
  · rfl

lemma measurableSet_returns (r : ℕ) (hr : r < K) (h : LoopHist α) :
    MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) |
      Returns settings O D Dsf h (p.1, (p.2 ⟨r, hr⟩).1)} := by
  classical
  have hset : {p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) |
      Returns settings O D Dsf h (p.1, (p.2 ⟨r, hr⟩).1)}
      = ⋃ B ∈ settings.schedule (loopPopulations K h.pops.length),
        (fun p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) =>
          clusterRun D Dsf h (p.1, (p.2 ⟨r, hr⟩).1)) ⁻¹'
        ret O.mq (loopPopulations K h.pops.length) none
          settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B := by
    ext p
    simp [Returns]
  rw [hset]
  exact Finset.measurableSet_biUnion _ fun B _ =>
    measurable_clusterRun_at settings D Dsf h ⟨r, hr⟩ (measurableSet_ret O _ none _ _ _ _ B)

lemma measurableSet_stage4 {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : L + settings.level3.probeBudget + 2 ≤ N) (r : ℕ)
    (hr : r < K)
    (y : Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))) :
    MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) |
      stage4 settings O D Dsf N r hr p.1 p.2 = y} := by
  classical
  have hhc := measurableSet_histCodes settings O D Dsf N hlen hN r
  have hok := measurableSet_histOK settings O D Dsf N hlen hN r
  set dec := decodeHist D settings.level3.walkStart
  rcases y with _ | ⟨c, B, F, K₀⟩
  · have hset : {p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) |
        stage4 settings O D Dsf N r hr p.1 p.2 = none}
        = ({p | HistOK settings O D Dsf N p.1 p.2 r} ∩ ⋃ c : List (HarvestCode α),
          ({p | histCodes settings O D Dsf N p.1 p.2 r = c} ∩ {_p | c.length = r}
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
    have hφ : Measurable fun p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) =>
        ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K) :=
      measurable_fst.prodMk (measurable_fst.comp ((measurable_pi_apply _).comp measurable_snd))
    have hset : {p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) |
        stage4 settings O D Dsf N r hr p.1 p.2 = some (c, B, F, K₀)}
        = {p | HistOK settings O D Dsf N p.1 p.2 r}
          ∩ {p | histCodes settings O D Dsf N p.1 p.2 r = c}
          ∩ {_p | c.length = r}
          ∩ {p | Returns settings O D Dsf H (p.1, (p.2 ⟨r, hr⟩).1)}
          ∩ (fun p : Ω × (Fin K → RoundDraws α K settings.level3.probeBudget) =>
              ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K)) ⁻¹'
            ({cd | roundState settings O D Dsf H cd = B
              ∧ clusterAt O.mq (loopPopulations K H.pops.length) (clusterRun D Dsf H cd)
                (roundState settings O D Dsf H cd) = F}
            ∩ {cd | roundKeys settings D Dsf H cd = K₀}) := by
      ext p
      simp only [Set.mem_ofPred_eq, stage4_eq_some, Set.mem_inter_iff, Set.mem_preimage]
      constructor
      · rintro ⟨h1, h2, h3, h4, h5, h6, h7⟩
        exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, ⟨h5, by rw [h5]; exact h6⟩, h7⟩
      · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, ⟨h5, h6⟩, h7⟩
        exact ⟨h1, h2, h3, h4, h5, by rw [← h5]; exact h6, h7⟩
    rw [hset]
    exact ((((hok.inter (hhc c)).inter (MeasurableSet.const _)).inter
      (measurableSet_returns settings O D Dsf r hr H)).inter
      (hφ ((measurableSet_roundPick settings O D Dsf H B F).inter
        (measurableSet_roundKeys settings D Dsf H K₀))))

omit [IsProbabilityMeasure μ] in
lemma roundKeys_eq_prefsOf (h : LoopHist α) (ω : Ω) (c : ClusterRest α K) :
    roundKeys settings D Dsf h (ω, c)
      = (prefsOf settings D Dsf h c).image fun p => prefixOf p settings.level3.walkStart :=
  rfl

/-- Where the clustering's state test returned, the answers at the strings the earlier rounds and
it read fix what reaches level 3. -/
lemma stage4_stop {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hN : L + settings.level3.probeBudget + 2 ≤ N) (r : ℕ)
    (hr : r < K) (c : List (HarvestCode α)) (B : State) (F K₀ : Finset (FreeMonoid α))
    (g : Fin K → RoundDraws α K settings.level3.probeBudget) (ω ω' ω₀ : Ω)
    (hs : stage4 settings O D Dsf N r hr ω g = some (c, B, F, K₀))
    (h : ∀ w ∈ histReads settings D Dsf N c g r ω₀
      ∪ schedReads (settings.schedule (loopPopulations K
          (decodeHist D settings.level3.walkStart c).pops.length))
        (loopPopulations K (decodeHist D settings.level3.walkStart c).pops.length)
        ((ω₀, runOf D Dsf (decodeHist D settings.level3.walkStart c) (g ⟨r, hr⟩).1) :
          Run Ω (FreeMonoid α) (Option (Fin K))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    stage4 settings O D Dsf N r hr ω' g = some (c, B, F, K₀) := by
  classical
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ :=
    (stage4_eq_some settings O D Dsf N r hr ω g c B F K₀).1 hs
  have hst : stageH settings O D Dsf N r ω g = some c := by
    simp only [stageH, if_pos h1, h2]
  have hst' := stageH_stop settings O D Dsf N hlen hN r c g ω ω' ω₀ hst
    fun w hw => h w (Finset.mem_union_left _ hw)
  have hok' : HistOK settings O D Dsf N ω' g r := by
    by_contra hn
    simp only [stageH, if_neg hn, reduceCtorEq] at hst'
  have hc' : histCodes settings O D Dsf N ω' g r = c := by
    simp only [stageH, if_pos hok', Option.some.injEq] at hst'
    exact hst'
  set H := decodeHist D settings.level3.walkStart c
  set sched := settings.schedule (loopPopulations K H.pops.length)
  have hsr : ∀ w ∈ schedReads sched (loopPopulations K H.pops.length)
      (clusterRun D Dsf H (ω, (g ⟨r, hr⟩).1)), (O.mq w ω = 1 ↔ O.mq w ω' = 1) :=
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
  refine (stage4_eq_some settings O D Dsf N r hr ω' g c B F K₀).2
    ⟨hok', hc', h3, hret', hrs, ?_, by rw [← h7]; rfl⟩
  rw [← h6]
  exact (clusterAt_congr_sched O sched (loopPopulations K H.pops.length) hBs _ hsr).symm

end Stage

lemma clusterAt_subset_suffsOf (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α) (ω : Ω) (c : ClusterRest α K) {B : State}
    (hB : B ∈ settings.schedule (loopPopulations K h.pops.length)) :
    clusterAt O.mq (loopPopulations K h.pops.length) (clusterRun D Dsf h (ω, c)) B
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
/-- Round `r`'s level 3, where every earlier clustering and this one returned and nothing
collides, fails its claim with chance at most
`δ + |Q||Σ| misledTail T (2 prefixMax) (2(|Q| + 1)ζ) u`.
-/
theorem loop_round_le_persist (E : σ → Set (FreeMonoid α)) (Ls NP N : ℕ) (m u crossLimit : ℝ)
    (hu : 0 ≤ u)
    (hN : P.L + settings.level3.probeBudget + 2 ≤ N)
    (hNP : ∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * NP)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 →
        RarerBounded N B.lo (B.hi + 1) p r crossLimit)
    (hk : 1 ≤ settings.level3.walkStart) (hη1 : O.η ≤ 1) (hcross : 0 < crossLimit)
    (hprem : RoundPremises σ α settings.level3 crossLimit
      (2 * prefixMax P.D settings.level3.walkStart))
    (hm : 6 * (K + 1) * NP * prefixMax P.D settings.level3.walkStart ≤ m)
    (hm1 : m ≤ 1 / 2)
    (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.probeBudget).real
      {p | HistOK settings O P.D Dsf N p.1 p.2 r
        ∧ (histCodes settings O P.D Dsf N p.1 p.2 r).length = r
        ∧ Returns settings O P.D Dsf (decodeHist P.D settings.level3.walkStart
            (histCodes settings O P.D Dsf N p.1 p.2 r)) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ RoundSpec M O P (roundSlope σ P.L settings.level3)
            (roundSlack σ α settings.level3 (2 * prefixMax P.D settings.level3.walkStart)) m
            (settings.level3.disagreementRate + m)
            (settings.level3.harvestRate / 2) settings.level3.walkStart E (u + m)
            (loopRound settings O P.D Dsf (decodeHist P.D settings.level3.walkStart
              (histCodes settings O P.D Dsf N p.1 p.2 r)) (p.1, p.2 ⟨r, hr⟩))
        ∧ NoColl settings P.D Dsf Ls (histCodes settings O P.D Dsf N p.1 p.2 r) p.2 r hr}
      ≤ settings.level3.δ + Fintype.card σ * Fintype.card α
        * misledTail settings.level3.probeBudget (2 * prefixMax P.D settings.level3.walkStart)
            (2 * (Fintype.card σ + 1) * crossLimit) u := by
  set δb := settings.level3.δ + Fintype.card σ * Fintype.card α
    * misledTail settings.level3.probeBudget (2 * prefixMax P.D settings.level3.walkStart)
      (2 * (Fintype.card σ + 1) * crossLimit) u with hδb
  have hδb0 : 0 ≤ δb := by
    obtain ⟨hδ0, -⟩ := hprem
    have := (prefixMax_pos P.D settings.level3.walkStart).le
    have := misledTail_nonneg (T := (settings.level3.probeBudget : ℝ)) (u := u)
      (by positivity : 0 ≤ 2 * prefixMax P.D settings.level3.walkStart)
      (by positivity : (0 : ℝ) ≤ 2 * (Fintype.card σ + 1) * crossLimit)
    have : (0 : ℝ) ≤ Fintype.card σ * Fintype.card α := by positivity
    nlinarith
  classical
  set T := settings.level3.probeBudget
  set k := settings.level3.walkStart
  set dec := decodeHist P.D k with hdec
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set rK : Fin K := ⟨r, hr⟩
  set ν := roundLaw α K T
  have hσ : MeasurePreserving (id : RoundDraws α K T → RoundDraws α K T) ν
      ((restLaw α K).prod (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))) :=
    MeasurePreserving.id _
  set Y := Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set Good : List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α) → Prop :=
    fun t => t.1.length = r ∧ t.2.1 ∈ settings.schedule (loopPopulations K (dec t.1).pops.length)
      ∧ SuffixFree t.2.2.1 ∧ t.2.2.1.card + 1 ≤ t.2.1.k ∧ t.2.2.2.card ≤ 6 * (K + 1) * NP
    with hGood
  have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
  have hex : ∀ t, Good t → ∃ Bd : Set (Ω × (Fin T → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd ≤ ENNReal.ofReal δb
      ∧ (∀ u ω ω', (∀ w ∈ roundZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D t.2.2.2 u) + 1 + settings.level3.probeBudget + 2))
          k t.2.2.2 t.2.2.1, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, u) ∈ Bd → (ω', u) ∈ Bd)
      ∧ ∀ x₀ Pr Sf ω v, (ω, v) ∉ Bd →
        RoundSpec M O P (roundSlope σ P.L settings.level3)
          (roundSlack σ α settings.level3 (2 * prefixMax P.D k)) m
          (settings.level3.disagreementRate + m) (settings.level3.harvestRate / 2) k E (u + m)
          (mkRound settings O P.D x₀ t.2.1 t.2.2.1 t.2.2.2 Pr Sf (ω, v)) :=
    fun t ht => round_bad_exists M O hL P settings E m u crossLimit hu hk hη1 hcross hprem
      hm1 t.2.1 t.2.2.1 t.2.2.2 ht.2.2.1 ht.2.2.2.1
      ((hband _ _ ht.2.1).2.2) (by
        have h1 : (t.2.2.2.card : ℝ) ≤ 6 * (K + 1) * NP := by exact_mod_cast ht.2.2.2.2
        exact (mul_le_mul_of_nonneg_right h1 hpm0).trans hm)
  set Bad : Y → Set (Ω × (Fin T → unitInterval)) := fun y => match y with
    | some t => if h : Good t then Classical.choose (hex t h) else ∅
    | none => ∅ with hBad
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some t => histReads settings P.D Dsf N t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (dec t.1).pops.length))
          (loopPopulations K (dec t.1).pops.length)
          ((ω₀, runOf P.D Dsf (dec t.1) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K)))
    | none => ∅
  set G : Y → Prop := fun y => ∃ t, y = some t ∧ t.1.length = r
  set Z : Y → (Fin T → unitInterval) → Finset (FreeMonoid α) := fun y u => match y with
    | some t => roundZone (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D t.2.2.2 u) + 1 + settings.level3.probeBudget + 2))
          k t.2.2.2 t.2.2.1
    | none => ∅
  set Dj : Y → Set (Fin K → RoundDraws α K T) := fun y => match y with
    | some t => {x | NoColl settings P.D Dsf Ls t.1 x r hr
        ∧ t.2.2.1 ⊆ suffsOf settings P.D Dsf (histAt settings P.D t.1 r) (x rK).1
        ∧ t.2.2.2 = (prefsOf settings P.D Dsf (histAt settings P.D t.1 r) (x rK).1).image
          fun p => prefixOf p k}
    | none => Set.univ
  have hmain := persist_stage_le O ν (restLaw α K)
    (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) id hσ id measurable_id
    (fun w => rfl) rK (stage4 settings O P.D Dsf N r hr)
    (fun ω x x' hxx' h1 => stage4_congr_draws settings O P.D Dsf N r hr ω x x'
      (fun i hi => hxx' i hi) h1)
    (measurableSet_stage4 settings O P.D Dsf N P.length_le hN r hr) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, -⟩ := hG
      exact stage4_stop settings O P.D Dsf N P.length_le hN r hr c B F K₀ x ω ω' ω₀ hs h)
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
    Z (fun y u ω ω' h hq => by
      rcases y with _ | t
      · exact hq
      · change (ω, u) ∈ (if h : Good t then Classical.choose (hex t h) else ∅) at hq
        show (ω', u) ∈ (if h : Good t then Classical.choose (hex t h) else ∅)
        split_ifs at hq ⊢ with hg
        · exact (Classical.choose_spec (hex t hg)).2.2.1 u ω ω' h hq
        · exact hq)
    Dj (fun y => by
      rcases y with _ | ⟨c, B, F, K₀⟩
      · exact MeasurableSet.univ
      · set h := histAt settings P.D c r
        have hx : Measurable fun x : Fin K → RoundDraws α K T => (x rK).1 :=
          measurable_fst.comp (measurable_pi_apply rK)
        have h2 : MeasurableSet {x : Fin K → RoundDraws α K T |
            F ⊆ suffsOf settings P.D Dsf h (x rK).1} := by
          have := measurableSet_of_data (fun x : Fin K → RoundDraws α K T =>
              suffsOf settings P.D Dsf h (x rK).1)
            (fun Q => hx (measurableSet_suffsOf settings P.D Dsf h Q))
            (fun Q => if F ⊆ Q then Set.univ else ∅) (fun Q => by split_ifs <;> simp)
          convert this using 1
          ext x
          simp only [Set.mem_setOf_eq]
          split_ifs with h'
          · exact iff_of_true h' (Set.mem_univ x)
          · exact iff_of_false h' (Set.notMem_empty x)
        have h3 : MeasurableSet {x : Fin K → RoundDraws α K T |
            K₀ = (prefsOf settings P.D Dsf h (x rK).1).image fun p => prefixOf p k} := by
          have := measurableSet_of_data (fun x : Fin K → RoundDraws α K T =>
              prefsOf settings P.D Dsf h (x rK).1)
            (fun Q => hx (measurableSet_prefsOf settings P.D Dsf h Q))
            (fun Q => if K₀ = Q.image fun p => prefixOf p k then Set.univ else ∅)
            (fun Q => by split_ifs <;> simp)
          convert this using 1
          ext x
          simp only [Set.mem_setOf_eq]
          split_ifs with h'
          · exact iff_of_true h' (Set.mem_univ x)
          · exact iff_of_false h' (Set.notMem_empty x)
        exact (measurableSet_noColl settings P.D Dsf Ls c r hr).inter (h2.inter h3))
    (fun y x hG hx => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, hcr⟩ := hG
      obtain ⟨hnc, hF, hK⟩ := hx
      have hpop : histAt settings P.D c r = dec c := by
        simp only [histAt, List.take_of_length_le (le_of_eq hcr), hdec] <;> rfl
      have hdis := disjoint_hist_family settings P.D Dsf N Ls
        (max (P.L + settings.level3.probeBudget + 2)
          (probesLen (probesOf settings P.D K₀ (x rK).2) + 1 + settings.level3.probeBudget + 2))
        c x r hr ω₀ hnc F hF
      rw [hpop] at hdis
      refine hdis.mono_right fun w hw => ?_
      obtain ⟨hrs, z, hz, v, hv, rfl⟩ := Finset.mem_filter.1 hw
      refine Finset.mem_sdiff.2 ⟨hrs, fun hw' => ?_⟩
      obtain ⟨⟨p, v'⟩, hpv, he⟩ := Finset.mem_image.1 hw'
      obtain ⟨hp, hv'⟩ := Finset.mem_product.1 hpv
      have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu =>
        hnc.1 rK le_rfl u (hF hu)
      obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he.symm ((hlenF v hv).trans (hlenF v' hv').symm)
      apply hz.2
      rw [hK]
      dsimp only
      rw [hpop, hzp]
      exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨p, hp, rfl⟩))
    (ENNReal.ofReal δb) (fun y => by
      rcases y with _ | t
      · show (μ.prod _) ∅ ≤ _
        simp
      · show (μ.prod _) (if h : Good t then Classical.choose (hex t h) else ∅) ≤ _
        split_ifs with h
        · exact (Classical.choose_spec (hex t h)).2.1
        · simp)
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hret, hnot, hnc⟩
  set c := histCodes settings O P.D Dsf N p.1 p.2 r
  set H := dec c
  set cd : ClusterDraws Ω α K := (p.1, (p.2 rK).1)
  set B := roundState settings O P.D Dsf H cd
  set F := clusterAt O.mq (loopPopulations K H.pops.length) (clusterRun P.D Dsf H cd) B
  set K₀ := roundKeys settings P.D Dsf H cd
  have hst : stage4 settings O P.D Dsf N r hr p.1 p.2 = some (c, B, F, K₀) :=
    (stage4_eq_some settings O P.D Dsf N r hr p.1 p.2 c B F K₀).2
      ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
  have hBs : B ∈ settings.schedule (loopPopulations K H.pops.length) :=
    (Classical.epsilon_spec hret).1
  have hpop : histAt settings P.D c r = H := by
    simp only [histAt, List.take_of_length_le (le_of_eq hlen), H, hdec] <;> rfl
  have hFs : F ⊆ suffsOf settings P.D Dsf (histAt settings P.D c r) (p.2 rK).1 := by
    rw [hpop]
    exact clusterAt_subset_suffsOf settings O P.D Dsf H p.1 (p.2 rK).1 hBs
  have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu => hnc.1 rK le_rfl u (hFs hu)
  have hK₀ : K₀ = (prefsOf settings P.D Dsf H (p.2 rK).1).image fun q => prefixOf q k :=
    roundKeys_eq_prefsOf settings P.D Dsf H p.1 (p.2 rK).1
  have hgood : Good (c, B, F, K₀) := by
    refine ⟨hlen, hBs, ?_, ?_, ?_⟩
    · intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    · obtain ⟨-, hk2, -⟩ := hband _ B hBs
      have h1 := clusterAt_card_add_one (rule := nearestClusterer) O
        (loopPopulations K H.pops.length) B (clusterRun P.D Dsf H cd)
      have h2 := nearestClusterer.card_le (fun w => O.mq w (oracleNoise (clusterRun P.D Dsf H cd))
        = 1)
        (prefixesAt (loopPopulations K H.pops.length) B.npref (clusterRun P.D Dsf H cd))
        (screenedAt O.mq (loopPopulations K H.pops.length) B (clusterRun P.D Dsf H cd)) B.k
        (by omega)
      show F.card + 1 ≤ B.k
      simp only [F]
      rw [clusterAt_eq]
      unfold clusterFull at h1
      omega
    · show K₀.card ≤ 6 * (K + 1) * NP
      rw [hK₀]
      refine Finset.card_image_le.trans ((card_roundPrefs_le _ _ _).trans ?_)
      have hpops : (loopPopulations K H.pops.length).card ≤ K + 1 := by
        have := Finset.card_le_univ (loopPopulations K H.pops.length)
        simpa using this
      have := hNP (loopPopulations K H.pops.length)
      calc 3 * (loopPopulations K H.pops.length).card
            * ∑ B ∈ settings.schedule (loopPopulations K H.pops.length), B.npref
          ≤ 3 * (K + 1) * (2 * NP) := Nat.mul_le_mul (Nat.mul_le_mul_left 3 hpops) this
        _ = 6 * (K + 1) * NP := by ring
  show (p.1, (id (p.2 rK)).2) ∈ Bad (stage4 settings O P.D Dsf N r hr p.1 p.2)
    ∧ p.2 ∈ Dj (stage4 settings O P.D Dsf N r hr p.1 p.2)
  rw [hst]
  refine ⟨?_, hnc, hFs, by rw [hpop]; exact hK₀⟩
  show (p.1, (p.2 rK).2) ∈ (if h : Good (c, B, F, K₀) then Classical.choose (hex _ h) else ∅)
  rw [dif_pos hgood]
  by_contra hnb
  exact hnot ((Classical.choose_spec (hex _ hgood)).2.2.2 (clusterRun P.D Dsf H cd)
    (roundSets settings P.D Dsf H cd).1 (roundSets settings P.D Dsf H cd).2 p.1 (p.2 rK).2 hnb)

end Round

end OrthoDFA
