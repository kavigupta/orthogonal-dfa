import OrthoDFA.Level2

/-! `Level2.Correct`: with reads fixed by the target state, level 2 ends well but for a small
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

/-- With reads fixed by the target state, `0 < y < m < ε < 1` (`harvestRate`, `alternativeRate`,
`disagreementRate`), a level `0 < a < 1`, and at least `S n` probes, level 2 fails to end well
(`OrthoDFA.RoundEnd`), a state being bad where all its strings read undecided, with chance at most
`2 S a`. `S` bounds the hypotheses: each split adds a leaf, at most `|Q| + 2` of them, and between
splits each of a leaf's edges is learned once. Under one hypothesis one of the two sequential tests
fires within `n` probes, and each errs with chance at most `a` over that stretch. A consistent
ending has its hypothesis and tree parting on at most `ε` of the probes, and a harvest's sampler
yields `y`. -/
def Correct : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Settings) (P : ℕ),
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let y := C.harvestRate
    let m := C.alternativeRate
    let ε := C.disagreementRate
    let a := C.testLevel
    let A₁ := Real.log (m / y)
    let B₁ := Real.log ((1 - y) / (1 - m))
    let A₂ := Real.log ((1 - m) / (1 - ε))
    let B₂ := Real.log (ε / m)
    let S := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let n := ⌈(A₁ + B₁ + A₂ + B₂) * Real.log (1 / a) / (A₁ * A₂ - B₁ * B₂)⌉₊
    ReadsByState read M side → 0 < y → y < m → m < ε → ε < 1 → 0 < a → a < 1 → S * n ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell M (AllUndecided read M) D ε
              {x | Disagrees read result.hypothesis C.walkStart x}
              C.walkStart (fun _ => harvestLaw read D C.walkStart result.hypothesis) y
              result.ending}
        ≤ 2 * S * a

end Level2

end OrthoDFA
