import OrthoDFA.Level2

/-! `IdealRoundCorrect`: with ideal reads, level 2 ends well but for a small chance. -/

namespace OrthoDFA

namespace Ideal

open MeasureTheory

variable {α : Type*} (read : FreeMonoid α → ARU)

/-- Every string in the state reads undecided. -/
def AllUndecided {σ : Type*} (M : DFA α σ) (q : σ) : Prop :=
  ∀ w : FreeMonoid α, M.eval w.toList = q → read w = .undecided

/-- Every target state's strings all read undecided, or all read its side. -/
def IdealReads {σ : Type*} (M : DFA α σ) (side : σ → Bool) : Prop :=
  ∀ q, AllUndecided read M q ∨
    ∀ w : FreeMonoid α, M.eval w.toList = q → read w = if side q then .accept else .reject

/-- Enough probes for the round to end. -/
def probeBudget (C : Level2Settings) (nα : ℕ) : ℕ :=
  (C.maxLeaves * (2 * C.maxLeaves * nα + 1) + 1)
    * ((C.recordsToMove - 1) * C.maxLeaves ^ 2 * nα
      + (C.harvestSize - 1) * (C.maxLeaves * nα + 2) + 1) * C.cleanProbesToEnd

/-- The hypothesis `H` replayed on a draw `w` from the anchor `e`: a probe of `w` walked from
`e`, harvesting the undecided strings it turns up, or nothing where `w` is shorter than `e`. -/
def replay (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ) : List (FreeMonoid α) :=
  if w.toList.length < e then [] else (probe read H.tree H.edges e w).undecidedStrings

/-- With ideal reads and enough probes, level 2 fails to end well (`OrthoDFA.RoundEnd`), a state
being bad where all its strings read undecided, with chance at most `(P + 1)(1 − ε)^n`, `n` the
clean probes that end it. Its harvest is replayed from an anchor drawn uniformly from `s, …, L`,
and yields the share of draws visiting a bad state at a position from `s` to `L`, over
`L + 1 − s`. -/
def IdealRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Level2Settings) (P : ℕ),
    IdealReads read M side → 1 ≤ C.recordsToMove → 1 ≤ C.harvestSize → 1 ≤ C.cleanProbesToEnd →
    Fintype.card σ + 2 ≤ C.maxLeaves → probeBudget C (Fintype.card α) ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (ε : ℝ) (s L : ℕ), ε ≤ 1 →
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell M (AllUndecided read M) D ε
              {x | Disagrees read result.hypothesis C.walkStart x}
              s L (replay read result.hypothesis)
              (coverage D s L M (AllUndecided read M) / (L + 1 - s : ℕ))
              result.ending}
        ≤ (P + 1) * (1 - ε) ^ C.cleanProbesToEnd

end Ideal

end OrthoDFA
