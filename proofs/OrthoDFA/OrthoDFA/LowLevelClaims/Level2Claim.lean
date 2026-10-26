import OrthoDFA.Code.Level2
import OrthoDFA.LowLevelClaims.RoundEnd

/-! `Level2RoundCorrect`: with reads fixed by the target state, level 2 ends well but for a small
chance. -/

namespace OrthoDFA

namespace Level2

open MeasureTheory

variable {α : Type*} (read : FreeMonoid α → ARU)

/-- Every string in the state reads undecided. -/
def AllUndecided {σ : Type*} (M : DFA α σ) (q : σ) : Prop :=
  ∀ w : FreeMonoid α, M.eval w.toList = q → read w = .undecided

/-- Every target state's strings all read undecided, or all read its side. -/
def ReadsByState {σ : Type*} (M : DFA α σ) (side : σ → Bool) : Prop :=
  ∀ q, AllUndecided read M q ∨
    ∀ w : FreeMonoid α, M.eval w.toList = q → read w = if side q then .accept else .reject

open scoped Classical in
/-- `1` where every string in the state reads undecided, `0` elsewhere. -/
noncomputable def undecidedOf {σ : Type*} (M : DFA α σ) (q : σ) : ℝ :=
  if AllUndecided read M q then 1 else 0

variable [DecidableEq α]

/-- The undecided string a probe of `x` walked from `k` under `H` stops at, if any. -/
def stuckString (H : HypothesisDFA α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (probeOrAgree read H.tree H.edges k x).undecidedStrings

/-- The law of what the harvest's sampler finds: the undecided string a probe drawn from `D` and
walked under `H` stops at. -/
noncomputable def harvestLaw [Countable α] (D : Measure (FreeMonoid α)) (k : ℕ)
    (H : HypothesisDFA α) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((stuckString read H k x).map Measure.dirac).sum

end Level2

open MeasureTheory Level2

/-- With reads fixed by the target state, `0 < y < ε < 1` (`harvestRate`, `disagreementRate`),
`0 < δ < 1`, and at least `12 S ε / (ε − y)² · log(2 S (S + 1) / δ) + S` probes, level 2 fails to
end well (`OrthoDFA.LowLevelClaims.RoundEnd`), a state reading undecided where all its strings do,
with chance at most `δ`. `S` bounds the hypotheses: each split adds a leaf, at most `|Q| + 2` of
them, and between splits each of a leaf's edges is learned once. A consistent ending has its
hypothesis and tree parting on at most `ε` of the probes, and a harvest's sampler yields `y`. -/
def Level2RoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Settings) (P : ℕ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    ReadsByState read M side → 0 < y → y < ε → ε < 1 → 0 < δ → δ < 1 →
    12 * (S : ℝ) * ε / (ε - y) ^ 2 * Real.log (2 * S * (S + 1) / δ) + S ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell
              { target := M, undecided := undecidedOf read M, harvestSlope := 1, harvestSlack := 0,
                probes := D, disagreeRate := ε,
                disagree := {x | Disagrees read result.hypothesis C.walkStart x},
                walkStart := C.walkStart,
                harvestLaw := fun _ => harvestLaw read D C.walkStart result.hypothesis,
                yield := y, ending := result.ending }}
        ≤ δ

end OrthoDFA
