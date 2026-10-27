import OrthoDFA.LowLevelClaims.Level3Claim

/-!
# `Level3AllowanceCorrect`: level 3 ends well while its tree stays within an allowance

Level 3 (`indecisiveLStarLevel3`) runs on reads independent across strings, each undecided with
its state's chance. Nothing is asked of how often a read is on its state's wrong side: level 3's
stretches need reads to be right only to keep the tree small. So the round fails to end well or
grows the tree past `|Q| + f + 2` leaves, with level 3's premises taken at `|Q| + f` states, with
chance at most `7δ/8` plus the chance that some step grows it past them.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*} [DecidableEq α]

/-- The step changes the hypothesis to a tree of more than `n + 2` leaves. -/
def Grows (read : FreeMonoid α → ARU) (C : Level3Settings) (n : ℕ) (s : Level3State α)
    (x : FreeMonoid α) : Prop :=
  ∃ s', applyProbe read C s x = .inl s' ∧ n + 2 < s'.tree.leaves.length

/-- Some step of the run from `s` over the probes grows the tree past `n + 2` leaves. -/
def GrowsRun (read : FreeMonoid α → ARU) (C : Level3Settings) (n : ℕ) :
    Level3State α → List (FreeMonoid α) → Prop
  | _, [] => False
  | s, x :: xs =>
    Grows read C n s x ∨ ∃ s', applyProbe read C s x = .inl s' ∧ GrowsRun read C n s' xs

end Random

open MeasureTheory Random

/-- Level 3 fails to end well, or grows its tree past `|Q| + f + 2` leaves, with chance at most
`7δ/8` plus the chance that a step grows it past them. -/
def Level3AllowanceCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (U : σ → ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (read : FreeMonoid α → Ω → ARU), (∀ z, Measurable (read z)) →
    ProbabilityTheory.iIndepFun read μ →
    (∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) →
    ∀ (P : PrefixDistribution α) (C : Level3Settings) (p₀ : ℝ) (f : ℕ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ + f
    let nα := Fintype.card α
    -- Level 3's bound on the hypotheses, at `|Q| + f` states.
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    -- Level 3's count of trees and edge maps, at `|Q| + f` states.
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (2 + Nat.clog 2 P.L))
    -- As at level 3, at chance `δ/16` at each site.
    let η := p₀ * Real.log (16 * (nQ + 1) * K / δ)
    -- No length-`walkStart` prefix holds more than `p₀` of the probes.
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < δ → δ < 1 → 0 < y →
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (read · ω) draws
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := η / y,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (read · ω) C.walkStart result.hypothesis
                ∪ endsSet (read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }
          ∨ GrowsRun (read · ω) C nQ initialState (List.ofFn draws)} ∂μ
      ≤ ENNReal.ofReal (7 * δ / 8)
        + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {d | GrowsRun (read · ω) C nQ initialState (List.ofFn d)} ∂μ

end OrthoDFA
