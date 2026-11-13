import RLStar.LowLevelClaims.Level3Claim

/-! # `Level3NoWrongCorrect`: level 3 under fixed reads, none of them wrong -/

namespace OrthoDFA

open MeasureTheory

/-- No string reads on its state's wrong side. -/
def ReadsNoWrong {α σ : Type*} (M : DFA α σ) (side : σ → Bool) (read : FreeMonoid α → ARU) :
    Prop :=
  ∀ z, read z ≠ if side (M.eval z.toList) then .reject else .accept

/-- With fixed reads none of which is wrong, level 3 fails to end well, or ends past `|Q| + 2`
leaves, with chance at most `δ`: a consistent ending has its hypothesis and tree parting on at most
`ε` of the probes, and a harvest's sampler yields `y`. -/
def Level3NoWrongCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (read : FreeMonoid α → ARU) (C : Random.Level3Settings)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D],
    let y := C.harvestRate
    let ε := C.disagreementRate
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    ReadsNoWrong M side read → 0 < C.δ → C.δ < 1 → 0 < ε → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget →
    (Measure.pi fun _ : Fin C.probeBudget => D).real
        {draws | let r := Random.indecisiveLStarLevel3 C read draws
          nQ + 2 < r.hypothesis.tree.leaves.length ∨
          match r.ending with
          | .consistent => ε < D.real (Random.searchSet read C.walkStart r.hypothesis
              ∪ Random.endsSet read C.walkStart r.hypothesis)
          | .harvest s => (Random.harvestLaw read D C.walkStart r.hypothesis s).real Set.univ < y
          | .failed => True}
      ≤ C.δ

end OrthoDFA
