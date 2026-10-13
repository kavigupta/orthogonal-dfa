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

/-- At most this many hypotheses: each split adds a leaf, at most `|Q| + 2` of them, and between
splits each of a leaf's edges is learned once. -/
def stretches (nσ nα : ℕ) : ℕ := nσ * ((nσ + 2) * nα + 1) + 2 * nα + 1

variable [DecidableEq α]

/-- The undecided string a probe of `x` walked from `k` under `H` stops at, if any. -/
def stuckString (H : HypothesisDFA α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (probeOrAgree false read H.tree H.edges k x).undecidedStrings

/-- The law of what the harvest's sampler finds: the undecided string a probe drawn from `D` and
walked under `H` stops at. -/
noncomputable def harvestLaw [Countable α] (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((stuckString read H k x).map Measure.dirac).sum

/-- The fewest probes after which, whatever count `d` of disagreements and at least as many
undecided reads they hold, the rate tests at level `a` decide: the undecided reads above `y`, or
the disagreements below `ε`. -/
noncomputable def stretchLength (y ε a : ℝ) : ℕ :=
  sInf {n | ∀ d ≤ n, binomSfGe n y d < a ∨ rateSide ε a 0 n d = some false}

/-- With ideal reads, `harvestRate < disagreementRate` and `stretches · n ≤ P` probes,
`n = stretchLength harvestRate disagreementRate testLevel` the probes by which a stretch ends,
level 2 fails to end well (`OrthoDFA.RoundEnd`), a state being bad where all its strings read
undecided, with chance at most `2 S n a`, `S` the stretches and `a = testLevel`: the rate tests'
error at each look. A consistent ending has its hypothesis and tree parting on at most
`disagreementRate` of the probes, and a harvest's sampler yields `harvestRate`. -/
def IdealRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Level2Settings) (P : ℕ),
    IdealReads read M side → 0 < C.harvestRate → C.harvestRate < C.disagreementRate →
    C.disagreementRate ≤ 1 → 0 < C.testLevel → C.testLevel < 1 →
    stretches (Fintype.card σ) (Fintype.card α)
        * stretchLength C.harvestRate C.disagreementRate C.testLevel ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell M (AllUndecided read M) D C.disagreementRate
              {x | Disagrees read result.hypothesis C.walkStart x}
              C.walkStart (fun _ => harvestLaw read D C.walkStart result.hypothesis)
              C.harvestRate result.ending}
        ≤ 2 * stretches (Fintype.card σ) (Fintype.card α)
          * stretchLength C.harvestRate C.disagreementRate C.testLevel * C.testLevel

end Ideal

end OrthoDFA
