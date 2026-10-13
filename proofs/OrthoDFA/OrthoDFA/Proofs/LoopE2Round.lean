import OrthoDFA.Proofs.LoopE2

/-!
# Round `r`'s level 5, given the rounds before it

What reaches round `r`'s level 5 is the earlier rounds' data and its clustering's state, family
and keys (`stage4`), fixed by the answers the earlier rounds and the clustering read. Off a
collision level 5 reads none of those strings, so it fails its claim with chance at most `δ`
(`loop_level5_round_le_persist`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

section Stage

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α))
  (N : ℕ)

/-- Some state of the schedule returns. -/
def Returns (harvests : List (Measure (FreeMonoid α))) (c : ClusterDraws Ω α K) : Prop :=
  ∃ B ∈ settings.schedule (loopPopulations K harvests.length),
    clusterRun harvests c ∈ ret O.mq (loopPopulations K harvests.length) none
      settings.indecisionLimit settings.gateLevel B

open Classical in
/-- What reaches round `r`'s level 5: the earlier rounds' data and the round's state, family and
keys, where every earlier clustering and this one returned. -/
def stage4 (r : ℕ) (hr : r < K) (ω : Ω) (g : Fin K → RoundDraws α K settings.level5.probeBudget) :
    Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α)) :=
  if HistOK settings O D N ω g r ∧ (histCodes settings O D N ω g r).length = r
      ∧ Returns settings O ((histCodes settings O D N ω g r).map
        (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1) then
    some (histCodes settings O D N ω g r,
      roundState settings O ((histCodes settings O D N ω g r).map
        (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1),
      clusterAt O.mq (loopPopulations K (histCodes settings O D N ω g r).length)
        (clusterRun ((histCodes settings O D N ω g r).map (decodePop D settings.level5.walkStart))
          (ω, (g ⟨r, hr⟩).1))
        (roundState settings O ((histCodes settings O D N ω g r).map
          (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1)),
      roundKeys settings ((histCodes settings O D N ω g r).map
        (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1))
  else none

omit [IsProbabilityMeasure μ] in
lemma stage4_eq_some (r : ℕ) (hr : r < K) (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level5.probeBudget) (c : List (HarvestCode α)) (B : State)
    (F K₀ : Finset (FreeMonoid α)) :
    stage4 settings O D N r hr ω g = some (c, B, F, K₀) ↔
      HistOK settings O D N ω g r ∧ histCodes settings O D N ω g r = c ∧ c.length = r
        ∧ Returns settings O (c.map (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1)
        ∧ roundState settings O (c.map (decodePop D settings.level5.walkStart))
            (ω, (g ⟨r, hr⟩).1) = B
        ∧ clusterAt O.mq (loopPopulations K (c.map (decodePop D settings.level5.walkStart)).length)
            (clusterRun (c.map (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1)) B = F
        ∧ roundKeys settings (c.map (decodePop D settings.level5.walkStart))
            (ω, (g ⟨r, hr⟩).1) = K₀ := by
  unfold stage4
  split_ifs with h
  · simp only [Option.some.injEq, Prod.mk.injEq, List.length_map]
    constructor
    · rintro ⟨rfl, rfl, rfl, rfl⟩
      exact ⟨h.1, rfl, h.2.1, h.2.2, rfl, rfl, rfl⟩
    · rintro ⟨-, rfl, -, -, rfl, rfl, rfl⟩
      exact ⟨rfl, rfl, rfl, rfl⟩
  · simp only [false_iff]
    rintro ⟨h1, rfl, h3, h4, -⟩
    exact h ⟨h1, h3, h4⟩

omit [IsProbabilityMeasure μ] in
lemma stage4_eq_none (r : ℕ) (hr : r < K) (ω : Ω)
    (g : Fin K → RoundDraws α K settings.level5.probeBudget) :
    stage4 settings O D N r hr ω g = none ↔
      ¬ (HistOK settings O D N ω g r ∧ (histCodes settings O D N ω g r).length = r
        ∧ Returns settings O ((histCodes settings O D N ω g r).map
          (decodePop D settings.level5.walkStart)) (ω, (g ⟨r, hr⟩).1)) := by
  unfold stage4
  split_ifs with h
  · simp only [false_iff, not_not]; exact h
  · simp only [true_iff]; exact h

omit [IsProbabilityMeasure μ] in
lemma stage4_congr_draws (r : ℕ) (hr : r < K) (ω : Ω)
    (g g' : Fin K → RoundDraws α K settings.level5.probeBudget)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) (h1 : (g ⟨r, hr⟩).1 = (g' ⟨r, hr⟩).1) :
    stage4 settings O D N r hr ω g = stage4 settings O D N r hr ω g' := by
  unfold stage4
  rw [histCodes_congr_draws settings O D N r ω g g' h, h1]
  have hok := histOK_congr_draws settings O D N ω g g' r h
  split_ifs with ha hb hb
  · rfl
  · exact absurd ⟨hok.1 ha.1, ha.2⟩ hb
  · exact absurd ⟨hok.2 hb.1, hb.2⟩ ha
  · rfl

lemma measurableSet_returns (r : ℕ) (hr : r < K)
    (c : List (HarvestCode α)) :
    MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) |
      Returns settings O (c.map (decodePop D settings.level5.walkStart)) (p.1, (p.2 ⟨r, hr⟩).1)} := by
  classical
  have hrun := (measurePreserving_clusterRun (μ := μ) (K := K) (Measure.dirac (1 : FreeMonoid α))
    (Measure.dirac 1) _ (decodePop_prob settings D c)).measurable
  have hset : {p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) |
      Returns settings O (c.map (decodePop D settings.level5.walkStart)) (p.1, (p.2 ⟨r, hr⟩).1)}
      = ⋃ B ∈ settings.schedule (loopPopulations K (c.map (decodePop D
          settings.level5.walkStart)).length),
        (fun p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) =>
          clusterRun (c.map (decodePop D settings.level5.walkStart)) (p.1, (p.2 ⟨r, hr⟩).1)) ⁻¹'
        ret O.mq (loopPopulations K (c.map (decodePop D settings.level5.walkStart)).length) none
          settings.indecisionLimit settings.gateLevel B := by
    ext p
    simp [Returns]
  rw [hset]
  exact Finset.measurableSet_biUnion _ fun B _ =>
    measurable_clusterRun_at settings hrun ⟨r, hr⟩ (measurableSet_ret O _ none _ _ B)

lemma measurableSet_stage4 {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level5.maxLeaves) (hN : L + settings.level5.maxLeaves ≤ N) (r : ℕ)
    (hr : r < K)
    (y : Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))) :
    MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) |
      stage4 settings O D N r hr p.1 p.2 = y} := by
  classical
  have hhc := measurableSet_histCodes settings O D N hlen hC hN r
  have hok := measurableSet_histOK settings O D N hlen hC hN r
  rcases y with _ | ⟨c, B, F, K₀⟩
  · have hset : {p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) |
        stage4 settings O D N r hr p.1 p.2 = none}
        = ({p | HistOK settings O D N p.1 p.2 r} ∩ ⋃ c : List (HarvestCode α),
          ({p | histCodes settings O D N p.1 p.2 r = c} ∩ {_p | c.length = r}
            ∩ {p | Returns settings O (c.map (decodePop D settings.level5.walkStart))
              (p.1, (p.2 ⟨r, hr⟩).1)}))ᶜ := by
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
      (measurableSet_returns settings O D r hr c))).compl
  · set H := c.map (decodePop D settings.level5.walkStart)
    have hrun := (measurePreserving_clusterRun (μ := μ) (K := K) (Measure.dirac (1 : FreeMonoid α))
      (Measure.dirac 1) _ (decodePop_prob settings D c)).measurable
    have hφ : Measurable fun p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) =>
        ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K) :=
      measurable_fst.prodMk (measurable_fst.comp ((measurable_pi_apply _).comp measurable_snd))
    have hset : {p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) |
        stage4 settings O D N r hr p.1 p.2 = some (c, B, F, K₀)}
        = {p | HistOK settings O D N p.1 p.2 r} ∩ {p | histCodes settings O D N p.1 p.2 r = c}
          ∩ {_p | c.length = r}
          ∩ {p | Returns settings O H (p.1, (p.2 ⟨r, hr⟩).1)}
          ∩ (fun p : Ω × (Fin K → RoundDraws α K settings.level5.probeBudget) =>
              ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K)) ⁻¹'
            ({cd | roundState settings O H cd = B
              ∧ clusterAt O.mq (loopPopulations K H.length) (clusterRun H cd)
                (roundState settings O H cd) = F}
            ∩ {cd | roundKeys settings H cd = K₀}) := by
      ext p
      simp only [Set.mem_ofPred_eq, stage4_eq_some, Set.mem_inter_iff, Set.mem_preimage]
      constructor
      · rintro ⟨h1, h2, h3, h4, h5, h6, h7⟩
        exact ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, ⟨h5, by rw [h5]; exact h6⟩, h7⟩
      · rintro ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, ⟨h5, h6⟩, h7⟩
        exact ⟨h1, h2, h3, h4, h5, by rw [← h5]; exact h6, h7⟩
    rw [hset]
    exact ((((hok.inter (hhc c)).inter (MeasurableSet.const _)).inter
      (measurableSet_returns settings O D r hr c)).inter
      (hφ ((measurableSet_roundPick settings O H (decodePop_prob settings D c) B F).inter
        (measurableSet_roundKeys settings H hrun K₀))))

lemma roundKeys_eq_prefsOf (harvests : List (Measure (FreeMonoid α))) (ω : Ω)
    (c : ClusterRest α K) :
    roundKeys settings harvests (ω, c)
      = (prefsOf settings harvests c).image fun p => prefixOf p settings.level5.walkStart := by
  unfold roundKeys
  rw [clusterKeys_eq]
  rfl

/-- Where the clustering's state test returned, the answers at the strings the earlier rounds and
it read fix what reaches level 5. -/
lemma stage4_stop {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level5.maxLeaves) (hN : L + settings.level5.maxLeaves ≤ N) (r : ℕ)
    (hr : r < K) (c : List (HarvestCode α)) (B : State) (F K₀ : Finset (FreeMonoid α))
    (g : Fin K → RoundDraws α K settings.level5.probeBudget) (ω ω' ω₀ : Ω)
    (hs : stage4 settings O D N r hr ω g = some (c, B, F, K₀))
    (h : ∀ w ∈ histReads settings D N c g r ω₀
      ∪ schedReads (settings.schedule (loopPopulations K
          (c.map (decodePop D settings.level5.walkStart)).length))
        (loopPopulations K (c.map (decodePop D settings.level5.walkStart)).length)
        ((ω₀, runOf (c.map (decodePop D settings.level5.walkStart)) (g ⟨r, hr⟩).1) :
          Run Ω (FreeMonoid α) (Option (Fin K))),
      (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    stage4 settings O D N r hr ω' g = some (c, B, F, K₀) := by
  classical
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := (stage4_eq_some settings O D N r hr ω g c B F K₀).1 hs
  have hst : stageH settings O D N r ω g = some c := by
    simp only [stageH, if_pos h1, h2]
  have hst' := stageH_stop settings O D N hlen hC hN r c g ω ω' ω₀ hst
    fun w hw => h w (Finset.mem_union_left _ hw)
  have hok' : HistOK settings O D N ω' g r := by
    by_contra hn
    simp only [stageH, if_neg hn, reduceCtorEq] at hst'
  have hc' : histCodes settings O D N ω' g r = c := by
    simp only [stageH, if_pos hok', Option.some.injEq] at hst'
    exact hst'
  set H := c.map (decodePop D settings.level5.walkStart)
  set sched := settings.schedule (loopPopulations K H.length)
  have hsr : ∀ w ∈ schedReads sched (loopPopulations K H.length)
      (clusterRun H (ω, (g ⟨r, hr⟩).1)), (O.mq w ω = 1 ↔ O.mq w ω' = 1) :=
    fun w hw => h w (Finset.mem_union_right _ hw)
  have hret' : Returns settings O H (ω', (g ⟨r, hr⟩).1) := by
    obtain ⟨B', hB', hret⟩ := h4
    exact ⟨B', hB', (ret_congr O sched (loopPopulations K H.length) none
      (Finset.mem_insert_self _ _) _ _ hB' _ hsr).1 hret⟩
  have hrs : roundState settings O H (ω', (g ⟨r, hr⟩).1) = B := by
    rw [← h5]
    exact (roundState_congr settings O H _ hsr).symm
  have hBs : B ∈ sched := by
    rw [← h5]
    exact (Classical.epsilon_spec h4).1
  refine (stage4_eq_some settings O D N r hr ω' g c B F K₀).2
    ⟨hok', hc', h3, hret', hrs, ?_, by rw [← h7]; rfl⟩
  rw [← h6]
  exact (clusterAt_congr_sched O sched (loopPopulations K H.length) hBs _ hsr).symm

end Stage

lemma clusterAt_subset_suffsOf (O : Oracle μ (FreeMonoid α)) (pops : Finset (Option (Fin K)))
    (harvests : List (Measure (FreeMonoid α))) (ω : Ω) (c : ClusterRest α K) (B : State) {NS : ℕ}
    (hB : B.nsuff ≤ NS) : clusterAt O.mq pops (clusterRun harvests (ω, c)) B ⊆ suffsOf NS c := by
  intro w hw
  unfold clusterAt at hw
  obtain ⟨hw1, hw⟩ := Finset.mem_erase.1 hw
  have h1 := clusterAround_subset _ 1 2 _ _ _ _ (one_mem_screenedAt O pops B _) hw
  have h2 := screenedAt_subset O pops B _ h1
  simp only [poolAt, Finset.mem_insert, Finset.mem_image, Finset.mem_range] at h2
  rcases h2 with h | ⟨i, hi, rfl⟩
  · exact absurd h hw1
  · exact Finset.mem_image.2 ⟨i, Finset.mem_range.2 (by omega), rfl⟩

section Round

variable [Nonempty α] {σ : Type*} [Fintype σ] (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
  (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf] (settings : LoopSettings K)

include hL in
/-- Round `r`'s level 5, where every earlier clustering and this one returned and nothing
collides, fails its claim with chance at most `δ`. -/
theorem loop_level5_round_le_persist (S Ls NS NP N : ℕ) (εd δ crossLimit θ y m : ℝ)
    (hmax : settings.level5.maxLeaves = Fintype.card σ + S + 3)
    (hN : P.L + settings.level5.maxLeaves ≤ N)
    (hNS : ∀ pops, schedNS (settings.schedule pops) ≤ NS)
    (hNP : ∀ pops, schedNP (settings.schedule pops) ≤ NP)
    (hband : ∀ pops, ∀ B ∈ settings.schedule pops, B.lo ≤ B.hi ∧ 2 ≤ B.k ∧ ∀ N, N + 1 ≤ B.k →
      ∀ p r ε₂ κ : ℝ, 0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → 0 ≤ κ →
        BandPasses N B.lo (B.hi + 1) p r crossLimit ε₂ κ)
    (hk : 1 ≤ settings.level5.walkStart) (hrec : 1 ≤ settings.level5.recordsToMove)
    (hdis : settings.level5.disagreementRate = εd / 2) (hη1 : O.η ≤ 1)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hy : 0 < y) (hcross : 0 < crossLimit) (hθ0 : 0 ≤ θ)
    (hθ : 384 * (P.L + 1) * (Fintype.card σ + S + 1) * θ ≤ 1)
    (hε : (P.L + 1) * (Fintype.card σ + S + 1) * crossLimit
      + 3 * unionLog (α := α) σ S δ * (2 * prefixMax P.D settings.level5.walkStart) ≤ y)
    (hm : 3 * (K + 1) * NP * prefixMax P.D settings.level5.walkStart ≤ m)
    (hm1 : m ≤ 1 / 2) (hmε : m ≤ εd / 2) (hmc : (P.L + 1) * m ≤ y / 2)
    (hlevel5 : ∀ (P' : PrefixDistribution α), P'.L = P.L → ∀ (O' : TriOracle (μ.prod μ) M) (θ : ℝ),
      let n : ℝ := Fintype.card σ + S
      let A : ℝ := Fintype.card α
      let bad := fun q => ¬ O'.Good θ q
      0 ≤ θ → 384 * (P'.L + 1) * (n + 1) * θ ≤ 1 →
      0 < O'.κ → 4 * O'.κ * (8 * (n + 3) * (P'.L + 1) * Real.sqrt (A * y / 3) + 1) ≤ y →
      0 ≤ O'.ε → (P'.L + 1) * (n + 1) * O'.ε
        + 3 * unionLog (α := α) σ S δ * prefixMax P'.D settings.level5.walkStart ≤ y →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.level5.probeBudget => P'.D)
          {draws | let result := indecisiveLStarLevel5 settings.level5 (O'.read · ω) draws
            ¬ EndsWell M bad P'.D settings.level5.disagreementRate
              (searchingProbes (O'.read · ω) settings.level5.walkStart result.hypothesis)
              settings.level5.walkStart
              (harvestLaw P'.D (fun z => (O'.read z ω).cut) settings.level5.walkStart)
              y result.ending} ∂(μ.prod μ)
        ≤ ENNReal.ofReal δ)
    (r : ℕ) (hr : r < K) :
    (loopMeasure μ P.D Dsf K settings.level5.probeBudget).real
      {p | HistOK settings O P.D N p.1 p.2 r
        ∧ (histCodes settings O P.D N p.1 p.2 r).length = r
        ∧ Returns settings O ((histCodes settings O P.D N p.1 p.2 r).map
            (decodePop P.D settings.level5.walkStart)) (p.1, (p.2 ⟨r, hr⟩).1)
        ∧ ¬ Level5Spec M O P θ εd (y / 2) settings.level5.walkStart
            (loopRound settings O P.D ((histCodes settings O P.D N p.1 p.2 r).map
              (decodePop P.D settings.level5.walkStart)) (p.1, p.2 ⟨r, hr⟩))
        ∧ NoCollD settings P.D Ls NS NP (histCodes settings O P.D N p.1 p.2 r) p.2 r hr} ≤ δ := by
  classical
  set T := settings.level5.probeBudget
  set k := settings.level5.walkStart
  set dec := decodePop P.D k with hdec
  have hC : 2 ≤ settings.level5.maxLeaves := by rw [hmax]; omega
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set rK : Fin K := ⟨r, hr⟩
  set ν := roundLaw P.D Dsf K T
  have hσ : MeasurePreserving (id : RoundDraws α K T → RoundDraws α K T) ν
      ((restLaw P.D Dsf K).prod (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))) :=
    MeasurePreserving.id _
  set Y := Option (List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α))
  set Good : List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α) → Prop :=
    fun t => t.1.length = r ∧ t.2.1 ∈ settings.schedule (loopPopulations K (t.1.map dec).length)
      ∧ SuffixFree t.2.2.1 ∧ t.2.2.1.card + 1 ≤ t.2.1.k ∧ t.2.2.2.card ≤ 3 * (K + 1) * NP
    with hGood
  have hpm0 : 0 ≤ prefixMax P.D k := (measureReal_nonneg).trans (le_prefixMax P.D k 1)
  have hex : ∀ t, Good t → ∃ Bd : Set (Ω × (Fin T → unitInterval)), MeasurableSet Bd
      ∧ (μ.prod (Measure.pi fun _ => volume)) Bd ≤ ENNReal.ofReal δ
      ∧ (∀ u ω ω', (∀ w ∈ l4Zone (max (P.L + settings.level5.maxLeaves)
          (probesLen (probesOf settings P.D t.2.2.2 u) + 1 + settings.level5.maxLeaves))
          k t.2.2.2 t.2.2.1, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) →
        (ω, u) ∈ Bd → (ω', u) ∈ Bd)
      ∧ ∀ x₀ ω u, (ω, u) ∉ Bd →
        Level5Spec M O P θ εd (y / 2) k (mkRound settings O P.D x₀ t.2.1 t.2.2.1 t.2.2.2 (ω, u)) :=
    fun t ht => level5_bad_exists M O hL P settings S εd δ crossLimit θ y m hmax hk hrec hdis hη1
      hδ hδ1 hy hcross hθ0 hθ hε hm1 hmε hmc hlevel5 t.2.1 t.2.2.1 t.2.2.2 ht.2.2.1 ht.2.2.2.1
      ((hband _ _ ht.2.1).2.2) (by
        have h1 : (t.2.2.2.card : ℝ) ≤ 3 * (K + 1) * NP := by exact_mod_cast ht.2.2.2.2
        exact (mul_le_mul_of_nonneg_right h1 hpm0).trans hm)
  set Bad : Y → Set (Ω × (Fin T → unitInterval)) := fun y => match y with
    | some t => if h : Good t then Classical.choose (hex t h) else ∅
    | none => ∅ with hBad
  set W : Y → (Fin K → RoundDraws α K T) → Finset (FreeMonoid α) := fun y x => match y with
    | some t => histReads settings P.D N t.1 x r ω₀
        ∪ schedReads (settings.schedule (loopPopulations K (t.1.map dec).length))
          (loopPopulations K (t.1.map dec).length)
          ((ω₀, runOf (t.1.map dec) (x rK).1) : Run Ω (FreeMonoid α) (Option (Fin K)))
    | none => ∅
  set G : Y → Prop := fun y => ∃ t, y = some t ∧ t.1.length = r
  set Z : Y → (Fin T → unitInterval) → Finset (FreeMonoid α) := fun y u => match y with
    | some t => l4Zone (max (P.L + settings.level5.maxLeaves)
          (probesLen (probesOf settings P.D t.2.2.2 u) + 1 + settings.level5.maxLeaves))
          k t.2.2.2 t.2.2.1
    | none => ∅
  set Dj : Y → Set (Fin K → RoundDraws α K T) := fun y => match y with
    | some t => {x | NoCollD settings P.D Ls NS NP t.1 x r hr ∧ t.2.2.1 ⊆ suffsOf NS (x rK).1
        ∧ t.2.2.2 = (prefsOf settings (popsAt settings P.D t.1 r) (x rK).1).image
          fun p => prefixOf p k}
    | none => Set.univ
  have hmain := persist_stage_le O ν (restLaw P.D Dsf K)
    (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) id hσ id measurable_id
    (fun w => rfl) rK (stage4 settings O P.D N r hr)
    (fun ω x x' hxx' h1 => stage4_congr_draws settings O P.D N r hr ω x x' (fun i hi => hxx' i hi) h1)
    (measurableSet_stage4 settings O P.D N P.length_le hC hN r hr) W G
    (fun y x ω ω' hG hs h => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, -⟩ := hG
      exact stage4_stop settings O P.D N P.length_le hC hN r hr c B F K₀ x ω ω' ω₀ hs h)
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
      · have hsf : ∀ a, Measurable fun x : Fin K → RoundDraws α K T => (x rK).1.1 a := fun a =>
          (measurable_pi_apply a).comp (measurable_fst.comp (measurable_fst.comp
            (measurable_pi_apply rK)))
        have h2 : {x : Fin K → RoundDraws α K T | F ⊆ suffsOf NS (x rK).1}
            = ⋂ v ∈ F, ⋃ a ∈ Finset.range NS, {x | (x rK).1.1 a = v} := by
          ext x
          simp [suffsOf, Finset.subset_iff]
        have hrunU := (measurePreserving_clusterRun (μ := Measure.dirac ()) (K := K)
          (Measure.dirac (1 : FreeMonoid α)) (Measure.dirac 1) (popsAt settings P.D c r)
          (decodePop_prob settings P.D (c.take r))).measurable
        have h3 : {x : Fin K → RoundDraws α K T | K₀ = (prefsOf settings (popsAt settings P.D c r)
            (x rK).1).image fun p => prefixOf p k}
            = (fun x : Fin K → RoundDraws α K T => ((((), (x rK).1) : ClusterDraws Unit α K))) ⁻¹'
              {cd | roundKeys settings (popsAt settings P.D c r) cd = K₀} := by
          ext x
          simp only [Set.mem_ofPred_eq, Set.mem_preimage]
          rw [roundKeys_eq_prefsOf, eq_comm]
        show MeasurableSet ({x | NoCollD settings P.D Ls NS NP c x r hr}
          ∩ ({x | F ⊆ suffsOf NS (x rK).1} ∩ {x | K₀ = (prefsOf settings (popsAt settings P.D c r)
            (x rK).1).image fun p => prefixOf p k}))
        rw [h2, h3]
        exact (measurableSet_noCollD settings P.D Ls NS NP c r hr).inter
          ((Finset.measurableSet_biInter _ fun v _ => Finset.measurableSet_biUnion _ fun a _ =>
            hsf a (measurableSet_singleton v)).inter
          ((measurable_const.prodMk (measurable_fst.comp (measurable_pi_apply rK)))
            (measurableSet_roundKeys settings _ hrunU K₀))))
    (fun y x hG hx => by
      obtain ⟨⟨c, B, F, K₀⟩, rfl, hcr⟩ := hG
      obtain ⟨hnc, hF, hK⟩ := hx
      have hpop : popsAt settings P.D c r = c.map dec := by
        simp only [popsAt, List.take_of_length_le (le_of_eq hcr)]
        rfl
      have hnc' := noColl_of_noCollD settings P.D (fun i => hNP _) hnc
      have hdis := disjoint_hist_level5 settings P.D N Ls NS
        (max (P.L + settings.level5.maxLeaves)
          (probesLen (probesOf settings P.D K₀ (x rK).2) + 1 + settings.level5.maxLeaves))
        c x r hr ω₀ (fun i => hNS _) hnc' F hF
      rw [hpop] at hdis
      refine hdis.mono_right fun w hw => ?_
      obtain ⟨hrs, z, hz, v, hv, rfl⟩ := Finset.mem_filter.1 hw
      refine Finset.mem_sdiff.2 ⟨hrs, fun hw' => ?_⟩
      obtain ⟨⟨p, v'⟩, hpv, he⟩ := Finset.mem_image.1 hw'
      obtain ⟨hp, hv'⟩ := Finset.mem_product.1 hpv
      have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu =>
        hnc'.1 rK le_rfl u (hF hu)
      obtain ⟨-, hzp⟩ := suffix_eq_of_mul_eq he.symm ((hlenF v hv).trans (hlenF v' hv').symm)
      apply hz.2
      rw [hK]
      dsimp only
      rw [hpop, hzp]
      exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨p, hp, rfl⟩))
    (ENNReal.ofReal δ) (fun y => by
      rcases y with _ | t
      · show (μ.prod _) ∅ ≤ _
        simp
      · show (μ.prod _) (if h : Good t then Classical.choose (hex t h) else ∅) ≤ _
        split_ifs with h
        · exact (Classical.choose_spec (hex t h)).2.1
        · simp)
  refine ENNReal.toReal_le_of_le_ofReal hδ.le ((measure_mono ?_).trans hmain)
  rintro p ⟨hok, hlen, hret, hnot, hnc⟩
  set c := histCodes settings O P.D N p.1 p.2 r
  set H := c.map dec
  set cd : ClusterDraws Ω α K := (p.1, (p.2 rK).1)
  set B := roundState settings O H cd
  set F := clusterAt O.mq (loopPopulations K H.length) (clusterRun H cd) B
  set K₀ := roundKeys settings H cd
  have hst : stage4 settings O P.D N r hr p.1 p.2 = some (c, B, F, K₀) :=
    (stage4_eq_some settings O P.D N r hr p.1 p.2 c B F K₀).2
      ⟨hok, rfl, hlen, hret, rfl, rfl, rfl⟩
  have hBs : B ∈ settings.schedule (loopPopulations K H.length) := (Classical.epsilon_spec hret).1
  have hnc' := noColl_of_noCollD settings P.D (fun i => hNP _) hnc
  have hFs : F ⊆ suffsOf NS (p.2 rK).1 := clusterAt_subset_suffsOf O _ H p.1 (p.2 rK).1 B
    ((Finset.le_sup (f := State.nsuff) hBs).trans (hNS _))
  have hlenF : ∀ u ∈ F, u.toList.length = Ls := fun u hu => hnc'.1 rK le_rfl u (hFs hu)
  have hpop : popsAt settings P.D c r = H := by
    simp only [popsAt, List.take_of_length_le (le_of_eq hlen)]
    rfl
  have hK₀ : K₀ = (prefsOf settings H (p.2 rK).1).image fun q => prefixOf q k :=
    roundKeys_eq_prefsOf settings H p.1 (p.2 rK).1
  have hgood : Good (c, B, F, K₀) := by
    refine ⟨hlen, hBs, ?_, ?_, ?_⟩
    · intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    · obtain ⟨-, hk2, -⟩ := hband _ B hBs
      have h1 := clusterAt_card_add_one (rule := lloydClusterer) O (loopPopulations K H.length) B
        (clusterRun H cd)
      have h2 := lloydClusterer.card_le (fun w => O.mq w (oracleNoise (clusterRun H cd)) = 1)
        (prefixesAt (loopPopulations K H.length) B.npref (clusterRun H cd))
        (screenedAt O.mq (loopPopulations K H.length) B (clusterRun H cd)) B.k (by omega)
      show F.card + 1 ≤ B.k
      simp only [F]
      rw [clusterAt_eq]
      unfold clusterFull at h1
      omega
    · show K₀.card ≤ 3 * (K + 1) * NP
      rw [hK₀]
      exact Finset.card_image_le.trans (card_prefsOf_le settings H (p.2 rK).1 (hNP _))
  show (p.1, (id (p.2 rK)).2) ∈ Bad (stage4 settings O P.D N r hr p.1 p.2)
    ∧ p.2 ∈ Dj (stage4 settings O P.D N r hr p.1 p.2)
  rw [hst]
  refine ⟨?_, hnc, hFs, by rw [hpop]; exact hK₀⟩
  show (p.1, (p.2 rK).2) ∈ (if h : Good (c, B, F, K₀) then Classical.choose (hex _ h) else ∅)
  rw [dif_pos hgood]
  by_contra hnb
  exact hnot ((Classical.choose_spec (hex _ hgood)).2.2.2 (clusterRun H cd) p.1 (p.2 rK).2 hnb)

end Round

end OrthoDFA
