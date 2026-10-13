import OrthoDFA.Proofs.LoopCollide

/-!
# The earlier rounds, summed up, on one noise

`stageH` is the earlier rounds' harvest data where every earlier clustering returned. It depends
on the draws only through the earlier rounds', is measurable, and the answers at the strings the
earlier rounds read fix it (`hist_congr`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

lemma lt_of_some_mem_loopPops {n : ℕ} {i : Fin K} (h : some i ∈ loopPopulations K n) :
    i.val < n := by
  simp only [loopPopulations, Finset.mem_insert, reduceCtorEq, false_or, Finset.mem_image,
    Finset.mem_filter, Finset.mem_univ, true_and, Option.some.injEq] at h
  obtain ⟨a, ha, rfl⟩ := h
  exact ha

section Stage

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α))
  {harvests : List (Measure (FreeMonoid α))}
  (N : ℕ)

open scoped Classical in
/-- The earlier rounds' harvest data, where every earlier clustering returned. -/
def stageH (r : ℕ) (ω : Ω) (g : Fin K → RoundDraws α K settings.level4.probeBudget) :
    Option (List (HarvestCode α)) :=
  if HistOK settings O D N ω g r then some (histCodes settings O D N ω g r) else none

omit [IsProbabilityMeasure μ] in
lemma histCodes_congr_draws (r : ℕ) (ω : Ω) (g g' : Fin K → RoundDraws α K settings.level4.probeBudget)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) :
    histCodes settings O D N ω g r = histCodes settings O D N ω g' r := by
  unfold histCodes
  rw [loopRounds_congr settings O D (fun i => (ω, g i)) (fun i => (ω, g' i)) r
    fun i hi => by rw [h i hi]]

omit [IsProbabilityMeasure μ] in
lemma histOK_congr_draws (ω : Ω) (g g' : Fin K → RoundDraws α K settings.level4.probeBudget) :
    ∀ r : ℕ, (∀ i : Fin K, i.1 < r → g i = g' i) →
      (HistOK settings O D N ω g r ↔ HistOK settings O D N ω g' r)
  | 0, _ => Iff.rfl
  | r + 1, h => by
    have ih := histOK_congr_draws ω g g' r fun i hi => h i (by omega)
    have hc := histCodes_congr_draws settings O D N r ω g g' fun i hi => h i (by omega)
    simp only [HistOK]
    rw [ih, hc]
    refine and_congr_right fun _ => forall_congr' fun hr => ?_
    rw [h ⟨r, hr⟩ (Nat.lt_succ_self r)]

omit [IsProbabilityMeasure μ] in
lemma stageH_congr_draws (r : ℕ) (ω : Ω) (g g' : Fin K → RoundDraws α K settings.level4.probeBudget)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) :
    stageH settings O D N r ω g = stageH settings O D N r ω g' := by
  classical
  unfold stageH
  rw [histCodes_congr_draws settings O D N r ω g g' h]
  simp only [histOK_congr_draws settings O D N ω g g' r h]

omit [IsProbabilityMeasure μ] in
lemma measurable_persist :
    Measurable (fun p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) =>
      fun i => ((p.1, p.2 i) : LoopRoundDraws Ω α K settings.level4.probeBudget)) :=
  measurable_pi_lambda _ fun i => measurable_fst.prodMk ((measurable_pi_apply i).comp measurable_snd)

lemma measurableSet_histCodes {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level4.maxLeaves) (hN : L + settings.level4.maxLeaves ≤ N) (r : ℕ)
    (c : List (HarvestCode α)) :
    MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
      histCodes settings O D N p.1 p.2 r = c} :=
  measurable_persist settings (measurableSet_codesOf settings O D hlen hC hN r c)

omit [IsProbabilityMeasure μ] in
lemma measurable_clusterRun_at (hrun : Measurable (clusterRun (Ω := Ω) (K := K) harvests))
    (i : Fin K) :
    Measurable (fun p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) =>
      clusterRun harvests ((p.1, (p.2 i).1) : ClusterDraws Ω α K)) :=
  hrun.comp (measurable_fst.prodMk (measurable_fst.comp ((measurable_pi_apply i).comp
    measurable_snd)))

lemma decodePop_prob (c : List (HarvestCode α)) :
    ∀ ν ∈ c.map (decodePop D settings.level4.walkStart), IsProbabilityMeasure ν := by
  intro ν h
  obtain ⟨c', -, rfl⟩ := List.mem_map.1 h
  unfold decodePop
  infer_instance

lemma measurableSet_histOK {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level4.maxLeaves) (hN : L + settings.level4.maxLeaves ≤ N) :
    ∀ r : ℕ, MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
      HistOK settings O D N p.1 p.2 r}
  | 0 => by simp only [HistOK, Set.setOf_true]; exact MeasurableSet.univ
  | r + 1 => by
    classical
    simp only [HistOK, Set.setOf_and]
    refine (measurableSet_histOK hlen hC hN r).inter ?_
    by_cases hr : r < K
    · have hset : {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
          ∀ h : r < K, (histCodes settings O D N p.1 p.2 r).length = r →
            ∃ B ∈ settings.schedule (loopPopulations K ((histCodes settings O D N p.1 p.2 r).map
                (decodePop D settings.level4.walkStart)).length),
              clusterRun ((histCodes settings O D N p.1 p.2 r).map
                (decodePop D settings.level4.walkStart)) (p.1, (p.2 ⟨r, h⟩).1)
                ∈ ret O.mq (loopPopulations K ((histCodes settings O D N p.1 p.2 r).map
                  (decodePop D settings.level4.walkStart)).length) none settings.indecisionLimit
                  settings.gateLevel B}
          = ⋃ c : List (HarvestCode α), {p | histCodes settings O D N p.1 p.2 r = c}
            ∩ (if c.length = r then ⋃ B ∈ settings.schedule (loopPopulations K
                (c.map (decodePop D settings.level4.walkStart)).length),
              (fun p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) =>
                clusterRun (c.map (decodePop D settings.level4.walkStart))
                  ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K)) ⁻¹'
                (ret O.mq (loopPopulations K (c.map (decodePop D settings.level4.walkStart)).length)
                  none settings.indecisionLimit settings.gateLevel B : Set _)
              else Set.univ) := by
        ext p
        simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
        constructor
        · intro h
          refine ⟨_, rfl, ?_⟩
          split_ifs with hl
          · obtain ⟨B, hB, hrB⟩ := h hr hl
            exact Set.mem_iUnion₂.2 ⟨B, hB, hrB⟩
          · trivial
        · rintro ⟨c, hc, h⟩ h' hl
          subst hc
          rw [if_pos hl] at h
          obtain ⟨B, hB, hrB⟩ := Set.mem_iUnion₂.1 h
          exact ⟨B, hB, hrB⟩
      rw [hset]
      refine MeasurableSet.iUnion fun c => (measurableSet_histCodes settings O D N hlen hC hN r c).inter
        ?_
      split_ifs
      · exact Finset.measurableSet_biUnion _ fun B _ =>
          measurable_clusterRun_at settings ((measurePreserving_clusterRun (μ := μ) (K := K)
            (Measure.dirac (1 : FreeMonoid α)) (Measure.dirac 1) _
            (decodePop_prob settings D c)).measurable) ⟨r, hr⟩ (measurableSet_ret O _ none _ _ B)
      · exact MeasurableSet.univ
    · have : {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
          ∀ h : r < K, (histCodes settings O D N p.1 p.2 r).length = r →
            ∃ B ∈ settings.schedule (loopPopulations K ((histCodes settings O D N p.1 p.2 r).map
                (decodePop D settings.level4.walkStart)).length),
              clusterRun ((histCodes settings O D N p.1 p.2 r).map
                (decodePop D settings.level4.walkStart)) (p.1, (p.2 ⟨r, h⟩).1)
                ∈ ret O.mq (loopPopulations K ((histCodes settings O D N p.1 p.2 r).map
                  (decodePop D settings.level4.walkStart)).length) none settings.indecisionLimit
                  settings.gateLevel B} = Set.univ := by
        ext p
        simp only [Set.mem_setOf_eq, Set.mem_univ, iff_true]
        intro h; exact absurd h hr
      rw [this]
      exact MeasurableSet.univ

lemma measurableSet_stageH {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level4.maxLeaves) (hN : L + settings.level4.maxLeaves ≤ N) (r : ℕ)
    (y : Option (List (HarvestCode α))) :
    MeasurableSet {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
      stageH settings O D N r p.1 p.2 = y} := by
  classical
  rcases y with _ | c
  · have : {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
        stageH settings O D N r p.1 p.2 = none} = {p | HistOK settings O D N p.1 p.2 r}ᶜ := by
      ext p
      simp only [stageH, Set.mem_setOf_eq, Set.mem_compl_iff]
      split_ifs with h <;> simp [h]
    rw [this]
    exact (measurableSet_histOK settings O D N hlen hC hN r).compl
  · have : {p : Ω × (Fin K → RoundDraws α K settings.level4.probeBudget) |
        stageH settings O D N r p.1 p.2 = some c}
        = {p | HistOK settings O D N p.1 p.2 r} ∩ {p | histCodes settings O D N p.1 p.2 r = c} := by
      ext p
      simp only [stageH, Set.mem_setOf_eq, Set.mem_inter_iff]
      split_ifs with h <;> simp [h]
    rw [this]
    exact (measurableSet_histOK settings O D N hlen hC hN r).inter
      (measurableSet_histCodes settings O D N hlen hC hN r c)

omit [IsProbabilityMeasure μ] in
lemma histReads_noise (c : List (HarvestCode α))
    (g : Fin K → RoundDraws α K settings.level4.probeBudget) (r : ℕ) (ω ω' : Ω) :
    histReads settings D N c g r ω = histReads settings D N c g r ω' := rfl

/-- Where every earlier clustering returned, the answers at the strings the earlier rounds read
fix the stage. -/
lemma stageH_stop {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (hC : 2 ≤ settings.level4.maxLeaves) (hN : L + settings.level4.maxLeaves ≤ N) (r : ℕ)
    (c : List (HarvestCode α)) (g : Fin K → RoundDraws α K settings.level4.probeBudget)
    (ω ω' ω₀ : Ω) (hs : stageH settings O D N r ω g = some c)
    (h : ∀ w ∈ histReads settings D N c g r ω₀, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    stageH settings O D N r ω' g = some c := by
  classical
  unfold stageH at hs ⊢
  split_ifs at hs with hok
  obtain rfl := Option.some.inj hs
  obtain ⟨hok', hc⟩ := hist_congr settings O D N hlen hC hN g r ω ω' hok h
  rw [if_pos hok', hc]

end Stage

end OrthoDFA
