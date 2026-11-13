import RLStar.Proofs.LoopCollide

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

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (D Dsf : Measure (FreeMonoid α))

open scoped Classical in
/-- The earlier rounds' harvest data, where every earlier clustering returned. -/
def stageH (r : ℕ) (ω : Ω) (g : Fin K → settings.RoundDraws α) :
    Option (List (HarvestCode α)) :=
  if HistOK settings O D Dsf ω g r then some (histCodes settings O D Dsf ω g r) else none

omit [IsProbabilityMeasure μ] in
lemma histCodes_congr_draws (r : ℕ) (ω : Ω) (g g' : Fin K → settings.RoundDraws α)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) :
    histCodes settings O D Dsf ω g r = histCodes settings O D Dsf ω g' r := by
  unfold histCodes
  rw [idealRounds_congr settings O D Dsf (fun i => (ω, g i)) (fun i => (ω, g' i)) r
    fun i hi => by rw [h i hi]]

omit [IsProbabilityMeasure μ] in
lemma histOK_congr_draws (ω : Ω) (g g' : Fin K → settings.RoundDraws α) :
    ∀ r : ℕ, (∀ i : Fin K, i.1 < r → g i = g' i) →
      (HistOK settings O D Dsf ω g r ↔ HistOK settings O D Dsf ω g' r)
  | 0, _ => Iff.rfl
  | r + 1, h => by
    have ih := histOK_congr_draws ω g g' r fun i hi => h i (by omega)
    have hc := histCodes_congr_draws settings O D Dsf r ω g g' fun i hi => h i (by omega)
    simp only [HistOK]
    rw [ih, hc]
    refine and_congr_right fun _ => forall_congr' fun hr => ?_
    rw [h ⟨r, hr⟩ (Nat.lt_succ_self r)]

omit [IsProbabilityMeasure μ] in
lemma stageH_congr_draws (r : ℕ) (ω : Ω) (g g' : Fin K → settings.RoundDraws α)
    (h : ∀ i : Fin K, i.1 < r → g i = g' i) :
    stageH settings O D Dsf r ω g = stageH settings O D Dsf r ω g' := by
  classical
  unfold stageH
  rw [histCodes_congr_draws settings O D Dsf r ω g g' h]
  simp only [histOK_congr_draws settings O D Dsf ω g g' r h]

omit [IsProbabilityMeasure μ] in
lemma measurable_persist :
    Measurable (fun p : Ω × (Fin K → settings.RoundDraws α) =>
      fun i => ((p.1, p.2 i) : LoopRoundDraws Ω α settings)) :=
  measurable_pi_lambda _ fun i => measurable_fst.prodMk ((measurable_pi_apply i).comp measurable_snd)

lemma measurableSet_histCodes [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (r : ℕ)
    (c : List (HarvestCode α)) :
    MeasurableSet {p : Ω × (Fin K → settings.RoundDraws α) |
      histCodes settings O D Dsf p.1 p.2 r = c} :=
  measurable_persist settings (measurableSet_codesOf settings O D Dsf (ae_lenBound hlen) (N := codeLen settings D) le_rfl r c)

omit [IsProbabilityMeasure μ] in
lemma measurable_idealRun_at [IsProbabilityMeasure D] (h : SpecHist α) (i : Fin K) :
    Measurable (fun p : Ω × (Fin K → settings.RoundDraws α) =>
      idealRun D Dsf h ((p.1, (p.2 i).1) : ClusterDraws Ω α K)) :=
  (measurable_idealRun D Dsf h).comp (measurable_fst.prodMk (measurable_fst.comp
    ((measurable_pi_apply i).comp measurable_snd)))

lemma measurableSet_histOK [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    :
    ∀ r : ℕ, MeasurableSet {p : Ω × (Fin K → settings.RoundDraws α) |
      HistOK settings O D Dsf p.1 p.2 r}
  | 0 => by simp only [HistOK, Set.setOf_true]; exact MeasurableSet.univ
  | r + 1 => by
    classical
    simp only [HistOK, Set.setOf_and]
    refine (measurableSet_histOK hlen r).inter ?_
    by_cases hr : r < K
    · have hset : {p : Ω × (Fin K → settings.RoundDraws α) |
          ∀ h : r < K, (histCodes settings O D Dsf p.1 p.2 r).length = r →
            ∃ B ∈ settings.schedule (loopPopulations K (decodeHist
                (histCodes settings O D Dsf p.1 p.2 r)).pops.length),
              idealRun D Dsf (decodeHist
                (histCodes settings O D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, h⟩).1)
                ∈ ret O.mq (loopPopulations K (decodeHist
                  (histCodes settings O D Dsf p.1 p.2 r)).pops.length) none
                  settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B}
          = ⋃ c : List (HarvestCode α), {p | histCodes settings O D Dsf p.1 p.2 r = c}
            ∩ (if c.length = r then ⋃ B ∈ settings.schedule (loopPopulations K
                (decodeHist c).pops.length),
              (fun p : Ω × (Fin K → settings.RoundDraws α) =>
                idealRun D Dsf (decodeHist c)
                  ((p.1, (p.2 ⟨r, hr⟩).1) : ClusterDraws Ω α K)) ⁻¹'
                (ret O.mq (loopPopulations K (decodeHist c).pops.length)
                  none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B : Set _)
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
      refine MeasurableSet.iUnion fun c =>
        (measurableSet_histCodes settings O D Dsf hlen r c).inter ?_
      split_ifs
      · exact Finset.measurableSet_biUnion _ fun B _ =>
          measurable_idealRun_at settings D Dsf _ ⟨r, hr⟩ (measurableSet_ret O _ none _ _ _ _ B)
      · exact MeasurableSet.univ
    · have : {p : Ω × (Fin K → settings.RoundDraws α) |
          ∀ h : r < K, (histCodes settings O D Dsf p.1 p.2 r).length = r →
            ∃ B ∈ settings.schedule (loopPopulations K (decodeHist
                (histCodes settings O D Dsf p.1 p.2 r)).pops.length),
              idealRun D Dsf (decodeHist
                (histCodes settings O D Dsf p.1 p.2 r)) (p.1, (p.2 ⟨r, h⟩).1)
                ∈ ret O.mq (loopPopulations K (decodeHist
                  (histCodes settings O D Dsf p.1 p.2 r)).pops.length) none
                  settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B} = Set.univ := by
        ext p
        simp only [Set.mem_setOf_eq, Set.mem_univ, iff_true]
        intro h; exact absurd h hr
      rw [this]
      exact MeasurableSet.univ

lemma measurableSet_stageH [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (r : ℕ)
    (y : Option (List (HarvestCode α))) :
    MeasurableSet {p : Ω × (Fin K → settings.RoundDraws α) |
      stageH settings O D Dsf r p.1 p.2 = y} := by
  classical
  rcases y with _ | c
  · have : {p : Ω × (Fin K → settings.RoundDraws α) |
        stageH settings O D Dsf r p.1 p.2 = none} = {p | HistOK settings O D Dsf p.1 p.2 r}ᶜ := by
      ext p
      simp only [stageH, Set.mem_setOf_eq, Set.mem_compl_iff]
      split_ifs with h <;> simp [h]
    rw [this]
    exact (measurableSet_histOK settings O D Dsf hlen r).compl
  · have : {p : Ω × (Fin K → settings.RoundDraws α) |
        stageH settings O D Dsf r p.1 p.2 = some c}
        = {p | HistOK settings O D Dsf p.1 p.2 r}
          ∩ {p | histCodes settings O D Dsf p.1 p.2 r = c} := by
      ext p
      simp only [stageH, Set.mem_setOf_eq, Set.mem_inter_iff]
      split_ifs with h <;> simp [h]
    rw [this]
    exact (measurableSet_histOK settings O D Dsf hlen r).inter
      (measurableSet_histCodes settings O D Dsf hlen r c)

/-- Where every earlier clustering returned, the answers at the strings the earlier rounds read
fix the stage. -/
lemma stageH_stop [IsProbabilityMeasure D] {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (r : ℕ)
    (c : List (HarvestCode α)) (g : Fin K → settings.RoundDraws α)
    (ω ω' ω₀ : Ω) (hs : stageH settings O D Dsf r ω g = some c)
    (h : ∀ w ∈ histReads settings D Dsf c g r ω₀, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    stageH settings O D Dsf r ω' g = some c := by
  classical
  unfold stageH at hs ⊢
  split_ifs at hs with hok
  obtain rfl := Option.some.inj hs
  obtain ⟨hok', hc⟩ := hist_congr settings O D Dsf hlen g r ω ω' hok h
  rw [if_pos hok', hc]

end Stage

end OrthoDFA
