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

/-- With reads fixed by the target state, `0 < y < ε < 1` (`harvestRate`, `disagreementRate`),
`0 < δ < 1`, and at least `64 K ε / (ε − y)² · log(4 K² / δ)` probes, `K = (|Q| + 2)² (|Σ| + 1)`,
level 2 fails to end well (`OrthoDFA.RoundEnd`), a state being bad where all its strings read
undecided, with chance at most `δ`. A consistent ending has its hypothesis and tree parting on at
most `ε` of the probes, and a harvest's sampler yields `y`. -/
def Correct : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (read : FreeMonoid α → ARU) (side : σ → Bool) (C : Settings) (P : ℕ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let K : ℕ := (Fintype.card σ + 2) ^ 2 * (Fintype.card α + 1)
    ReadsByState read M side → 0 < y → y < ε → ε < 1 → 0 < δ → δ < 1 →
    64 * (K : ℝ) * ε / (ε - y) ^ 2 * Real.log (4 * (K : ℝ) ^ 2 / δ) ≤ P →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
      (Measure.pi fun _ : Fin P => D).real
          {draws | let result := indecisiveLStarLevel2 C read draws
            ¬ EndsWell M (AllUndecided read M) D ε
              {x | Disagrees read result.hypothesis C.walkStart x}
              C.walkStart (fun _ => harvestLaw read D C.walkStart result.hypothesis) y
              result.ending}
        ≤ δ

end Level2

end OrthoDFA
