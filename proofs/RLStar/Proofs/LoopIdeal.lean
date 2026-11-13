import RLStar.Proofs.LoopCodes

/-!
# The loop on filled draws

`idealRounds` runs the loop's rounds on the streams (`idealDraws`), never failing.  Where every
round it runs scans within `scanCap` and returns at a scheduled state (`RoundGood`), the loop is
these rounds (`loopRounds_eq_some`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}
  {μ : Measure Ω}

/-- The round after the history `h` on the filled draws. -/
def idealRound (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α settings) :
    SpecRound α :=
  roundOf settings O D h
    (idealRun D Dsf h (d.1, d.2.1)) d.2.2

/-- The first `r` rounds on the filled draws, cut short at a round that does not end in a
harvest. -/
def idealRounds (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α settings) :
    ℕ → List (SpecRound α)
  | 0 => []
  | r + 1 =>
    if h : r < K ∧ ∀ o ∈ idealRounds settings O D Dsf draws r, o.result.ending.IsHarvest
    then idealRounds settings O D Dsf draws r
      ++ [idealRound settings O D Dsf (specHistOf (idealRounds settings O D Dsf draws r))
        (draws ⟨r, h.1⟩)]
    else idealRounds settings O D Dsf draws r

/-- The round's draws scan within `scanCap`, and its clustering on them returns at a scheduled
state. -/
def RoundGood (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α settings) : Prop :=
  ScanWithin (settings.schedule (loopPopulations K h.pops.length))
      (loopPopulations K h.pops.length) (fun _ => settings.scanCap)
      (idealDraws D Dsf h (d.1, d.2.1))
    ∧ ∃ B ∈ settings.schedule (loopPopulations K h.pops.length),
      idealRun D Dsf h (d.1, d.2.1)
        ∈ ret O.mq (loopPopulations K h.pops.length) none settings.η₀ settings.indecisionLimit
          settings.εcov settings.gateLevel B

/-- Where every round run on the filled draws is good, the loop runs those rounds. -/
theorem loopRounds_eq_some (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α settings)
    (hg : ∀ r (hr : r < K), (∀ o ∈ idealRounds settings O D Dsf draws r,
        o.result.ending.IsHarvest) →
      RoundGood settings O D Dsf (specHistOf (idealRounds settings O D Dsf draws r))
        (draws ⟨r, hr⟩))
    (n : ℕ) :
    loopRounds settings O D Dsf draws n = some (idealRounds settings O D Dsf draws n) := by
  induction n with
  | zero => rfl
  | succ r ih =>
    rw [loopRounds, ih, Option.bind_some, idealRounds]
    split_ifs with h
    · obtain ⟨h2, h3⟩ := hg r h.1 h.2
      rw [loopRound_eq_some settings O D Dsf _ _ h2 h3]
      rfl
    · rfl


end OrthoDFA

end
