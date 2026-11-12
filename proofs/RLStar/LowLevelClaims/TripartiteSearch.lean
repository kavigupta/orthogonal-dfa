import RLStar.Code.TripartiteSearch
import Mathlib.Data.Finset.Card

/-! What the tripartite search promises. -/

namespace OrthoDFA

/-- Each result says what `agrees` is where it points; the search asks only positions from `lo`
to `hi`, at most `2 + 3⌈log₂(hi − lo)⌉` distinct ones, and without recovery at most
`2 + ⌈log₂(hi − lo)⌉`; and where `agrees` turns from agreeing to disagreeing at `t` with nothing
undecided, it finds `t`. -/
def TripartiteSearchCorrect : Prop :=
  ∀ (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ), lo ≤ hi →
    let r := tripartiteSearch recover agrees lo hi
    let asked := (tripartiteQ recover lo hi).reads agrees
    (r = .undecidedLo → agrees lo = none)
    ∧ (r = .disagreeLo → agrees lo = some false)
    ∧ (r = .undecidedHi → agrees lo = some true ∧ agrees hi = none)
    ∧ (r = .allAgree → agrees lo = some true ∧ agrees hi = some true)
    ∧ (∀ j, r = .edge j →
        lo < j ∧ j ≤ hi ∧ agrees (j - 1) = some true ∧ agrees j = some false)
    ∧ (∀ j, r = .pair j → lo ≤ j ∧ j + 1 ≤ hi ∧ agrees j = none ∧ agrees (j + 1) = none)
    ∧ (∀ j, r = .triple j → lo < j ∧ j < hi
        ∧ agrees (j - 1) = some true ∧ agrees j = none ∧ agrees (j + 1) = some false)
    ∧ (∀ j, r = .undecidedMid j → recover = false ∧ lo < j ∧ j < hi ∧ agrees j = none)
    ∧ (∀ p ∈ asked, lo ≤ p ∧ p ≤ hi)
    ∧ asked.card ≤ 2 + 3 * Nat.clog 2 (hi - lo)
    ∧ (recover = false → asked.card ≤ 2 + Nat.clog 2 (hi - lo))
    ∧ (∀ t, lo < t → t ≤ hi → (∀ i, lo ≤ i → i ≤ hi → agrees i = some (decide (i < t))) →
        r = .edge t)

end OrthoDFA
