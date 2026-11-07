import OrthoDFA.Proofs.LoopCodes

/-!
# The loop on filled draws

`idealRounds` runs the loop's rounds on the filled draws (`idealDraws`), never failing.  Where
every round it runs hits, scans within `scanCap`, returns at a scheduled state and keeps level 3
within `leafCap` leaves (`RoundGood`), the loop is these rounds (`loopRounds_eq_some`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}
  {μ : Measure Ω}

/-- The round after the history `h` on the filled draws. -/
def idealRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    LoopRound α :=
  roundOf settings O D h
    (idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)) d.2.2

/-- The first `r` rounds on the filled draws, cut short at a round that does not end in a
harvest. -/
def idealRounds (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
      settings.level3.probeBudget) :
    ℕ → List (LoopRound α)
  | 0 => []
  | r + 1 =>
    if h : r < K ∧ ∀ o ∈ idealRounds settings O D Dsf draws r, o.result.ending.harvested ≠ none
    then idealRounds settings O D Dsf draws r
      ++ [idealRound settings O D Dsf (histOf (idealRounds settings O D Dsf draws r))
        (draws ⟨r, h.1⟩)]
    else idealRounds settings O D Dsf draws r

/-- Alike histories make the same round. -/
theorem idealRound_alike {L : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    {h h' : LoopHist α} (hh : HistAlike L h h')
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    idealRound settings O D Dsf h d = idealRound settings O D Dsf h' d := by
  unfold idealRound roundOf
  rw [idealRun_alike hh D Dsf hD, hh.length]

/-- The round's draws hit, its filled draws scan within `scanCap`, its clustering on them
returns at a scheduled state, and its level 3 keeps within `leafCap` leaves. -/
def RoundGood (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) : Prop :=
  (clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap h
      (d.1, d.2.1)).isSome
    ∧ ScanWithin (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) settings.scanCap
      (idealDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1))
    ∧ (∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf settings.level3.walkStart settings.level3.harvestReadCap h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B)
    ∧ (idealRound settings O D Dsf h d).result.hypothesis.tree.leaves.length
      ≤ settings.level3.leafCap

/-- Where every round run on the filled draws is good, the loop runs those rounds. -/
theorem loopRounds_eq_some (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
      settings.level3.probeBudget)
    (hg : ∀ r (hr : r < K), (∀ o ∈ idealRounds settings O D Dsf draws r,
        o.result.ending.harvested ≠ none) →
      RoundGood settings O D Dsf (histOf (idealRounds settings O D Dsf draws r)) (draws ⟨r, hr⟩))
    (n : ℕ) :
    loopRounds settings O D Dsf draws n = some (idealRounds settings O D Dsf draws n) := by
  induction n with
  | zero => rfl
  | succ r ih =>
    rw [loopRounds, ih, Option.bind_some, idealRounds]
    split_ifs with h
    · obtain ⟨h1, h2, h3, h4⟩ := hg r h.1 h.2
      rw [loopRound_eq_some settings O D Dsf _ _ h1 h2 h3 h4]
      rfl
    · rfl

/-! ## The samplers' populations -/

/-- The populations' distributions, given populations: the probes' `D`, then each in turn. -/
def popDists (D : Measure (FreeMonoid α)) (pops : List (Measure (FreeMonoid α))) :
    Option (Fin K) → Measure (FreeMonoid α)
  | none => D
  | some i => pops.getD i (Measure.dirac 1)

/-- Each sampler's population over probes from `D` walked from `k`. -/
def samplerLaws (D : Measure (FreeMonoid α)) (k : ℕ) (pops : List (HarvestSampler α)) :
    List (Measure (FreeMonoid α)) :=
  pops.map fun s => popLaw (Random.harvestLaw s.read D k s.hyp s.site)

omit [MeasurableSpace Ω] in
lemma loopDistributions_eq_popDists (D : Measure (FreeMonoid α)) (k : ℕ)
    (pops : List (HarvestSampler α)) (j : Option (Fin K)) :
    loopDistributions D k pops j = popDists D (samplerLaws D k pops) j := by
  rcases j with _ | i
  · rfl
  simp only [loopDistributions, popDists, samplerLaws]
  rcases h : pops[i.val]? with _ | s
  · rw [List.getD_eq_default _ _ (by simpa using List.getElem?_eq_none_iff.1 h)]
  · obtain ⟨hi, hs⟩ := List.getElem?_eq_some_iff.1 h
    rw [List.getD_eq_getElem _ _ (by simpa using hi)]
    simp [hs]

omit [MeasurableSpace Ω] in
lemma samplerLaws_alike {L : ℕ} {pops pops' : List (HarvestSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) :
    samplerLaws D k pops = samplerLaws D k pops' := by
  induction hp with
  | nil => rfl
  | cons h _ ih =>
    simp only [samplerLaws, List.map_cons] at ih ⊢
    rw [ih, harvestLaw_alike h D hD k]

omit [MeasurableSpace Ω] in
lemma samplerLaws_length (D : Measure (FreeMonoid α)) (k : ℕ) (pops : List (HarvestSampler α)) :
    (samplerLaws D k pops).length = pops.length := List.length_map _

end OrthoDFA

end
