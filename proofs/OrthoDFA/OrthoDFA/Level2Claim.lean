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
def stuckString (H : Hypothesis α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (probeOrAgree false read H.tree H.edges k x).undecidedStrings

/-- The law of what the harvest's sampler finds: the undecided string a probe drawn from `D` and
walked under `H` stops at. -/
noncomputable def harvestLaw [Countable α] (D : Measure (FreeMonoid α)) (k : ℕ) (H : Hypothesis α) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((stuckString read H k x).map Measure.dirac).sum

/-- The fewest undecided reads out of `n` that the rate test calls above `y` at level `a`. -/
noncomputable def fireCount (y a : ℝ) (n : ℕ) : ℕ := sInf {k | binomSfGe n y k < a}

/-- The fewest probes after which the rate test at level `a` calls no disagreement below `ε`:
the least `n` with `(1 − ε)^n < a`. -/
noncomputable def consistentAfter (ε a : ℝ) : ℕ := sInf {n | (1 - ε) ^ n < a}

/-- With ideal reads and `stretches · n ≤ P` probes, `n = consistentAfter disagreementRate
testLevel` the probes after which a stretch with no disagreement ends consistent, level 2 fails to
end well (`OrthoDFA.RoundEnd`), a state being bad where all its strings read undecided, with chance
at most `S (a + P(Bin(n, r) < K)) + P a`: `S` the stretches, `y` and `a` the rate tests' rate for
the undecided reads and their level, and `K` the fewest undecided reads out of `n` the test calls
above `y`. A consistent ending has its hypothesis and tree parting on at most
`disagreementRate + r` of the probes, and a harvest's sampler yields `y`. -/
def IdealRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Level2Settings) (P : ℕ),
    IdealReads read M side → 0 < C.harvestRate → C.harvestRate ≤ 1 →
    0 < C.disagreementRate → C.disagreementRate ≤ 1 → 0 < C.testLevel → C.testLevel ≤ 1 →
    stretches (Fintype.card σ) (Fintype.card α)
        * consistentAfter C.disagreementRate C.testLevel ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (r : ℝ), 0 ≤ r → r ≤ 1 →
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell M (AllUndecided read M) D (C.disagreementRate + r)
              {x | Disagrees read result.hypothesis C.walkStart x}
              C.walkStart (fun _ => harvestLaw read D C.walkStart result.hypothesis)
              C.harvestRate result.ending}
        ≤ stretches (Fintype.card σ) (Fintype.card α) * (C.testLevel
            + (1 - binomSfGe (consistentAfter C.disagreementRate C.testLevel) r
              (fireCount C.harvestRate C.testLevel
                (consistentAfter C.disagreementRate C.testLevel))))
          + P * C.testLevel

end Ideal

end OrthoDFA
