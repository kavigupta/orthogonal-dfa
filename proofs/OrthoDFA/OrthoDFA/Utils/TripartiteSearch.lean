import OrthoDFA.TripartiteSearch

/-! What the tripartite search promises. -/

namespace OrthoDFA

/-- Each result says what `agrees` is where it points; the search queries only positions from
`lo` to `hi`, at most `2 + 3⌈log₂(hi − lo)⌉` of them, and without recovery at most
`2 + ⌈log₂(hi − lo)⌉`; and where `agrees` turns from agreeing to disagreeing at `t` with nothing
undecided, it finds `t`. -/
def TripartiteSearchCorrect : Prop :=
  ∀ (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ), lo ≤ hi →
    let r := tripartiteSearch recover agrees lo hi
    (r.1 = .undecidedLo → agrees lo = none)
    ∧ (r.1 = .disagreeLo → agrees lo = some false)
    ∧ (r.1 = .undecidedHi → agrees lo = some true ∧ agrees hi = none)
    ∧ (r.1 = .allAgree → agrees lo = some true ∧ agrees hi = some true)
    ∧ (∀ j, r.1 = .edge j →
        lo < j ∧ j ≤ hi ∧ agrees (j - 1) = some true ∧ agrees j = some false)
    ∧ (∀ j, r.1 = .pair j → lo ≤ j ∧ j + 1 ≤ hi ∧ agrees j = none ∧ agrees (j + 1) = none)
    ∧ (∀ j, r.1 = .triple j → lo < j ∧ j < hi
        ∧ agrees (j - 1) = some true ∧ agrees j = none ∧ agrees (j + 1) = some false)
    ∧ (∀ j, r.1 = .undecidedMid j → recover = false ∧ lo < j ∧ j < hi ∧ agrees j = none)
    ∧ (∀ p ∈ r.2, lo ≤ p ∧ p ≤ hi)
    ∧ r.2.length ≤ 2 + 3 * Nat.clog 2 (hi - lo)
    ∧ (recover = false → r.2.length ≤ 2 + Nat.clog 2 (hi - lo))
    ∧ (∀ t, lo < t → t ≤ hi → (∀ i, lo ≤ i → i ≤ hi → agrees i = some (decide (i < t))) →
        r.1 = .edge t)

end OrthoDFA
