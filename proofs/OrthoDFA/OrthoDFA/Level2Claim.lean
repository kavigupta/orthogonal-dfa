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

variable [DecidableEq α]

/-- The strings a probe of `x` walked from `k` under `H` reads undecided at the site, each once. -/
noncomputable def siteHarvest [Countable α] (H : Hypothesis α) (k : ℕ) (site : UndecidedSite α)
    (x : FreeMonoid α) : List (FreeMonoid α) :=
  match probeOrAgree read H.tree H.edges k x with
  | .undecided site' zs _ => if site' = site then zs.dedup else []
  | _ => []

/-- The law of what the harvest's sampler finds, each string as often as a probe drawn from `D`
and walked under `H` finds it at the site. -/
noncomputable def harvestLaw [Countable α] (D : Measure (FreeMonoid α)) (k : ℕ) (H : Hypothesis α)
    (site : UndecidedSite α) : Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest read H k site x).map Measure.dirac).sum

/-- With ideal reads and enough probes, level 2 fails to end well (`OrthoDFA.RoundEnd`), a state
being bad where all its strings read undecided, with chance at most `(P + 1)(1 − ε)^n`, `n` the
clean probes that end it, plus `P (2 + (|Q| + 2)|Σ|) y`. A harvest is a site of the final
hypothesis, and its sampler yields `y`: each of the round's probes is charged at one of at most
`2 + (|Q| + 2)|Σ|` sites, and is charged a string at a site whose sampler yields at most `y` with
chance at most `y`. -/
def IdealRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Level2Settings) (P : ℕ),
    IdealReads read M side → 1 ≤ C.recordsToMove → 1 ≤ C.harvestSize → 1 ≤ C.cleanProbesToEnd →
    Fintype.card σ + 2 ≤ C.maxLeaves → probeBudget C (Fintype.card α) ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (ε y : ℝ), ε ≤ 1 → 0 ≤ y →
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell M (AllUndecided read M) D ε
              {x | Disagrees read result.hypothesis C.walkStart x}
              C.walkStart (harvestLaw read D C.walkStart result.hypothesis) y
              result.ending}
        ≤ (P + 1) * (1 - ε) ^ C.cleanProbesToEnd
          + P * (2 + (Fintype.card σ + 2) * Fintype.card α) * y

end Ideal

end OrthoDFA
