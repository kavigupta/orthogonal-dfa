import Mathlib.Data.Nat.Log

/-! A search for where `agrees` turns from `some true` to `some false`, past `none`s. -/

namespace OrthoDFA

/-- `edge j`: `j - 1` agrees and `j` disagrees. `pair j`: `j` and `j + 1` are undecided.
`triple j`: `j` is undecided between an agreement and a disagreement. `undecidedMid j`: `j` is
undecided, and the search was not to recover. -/
inductive SearchResult
  | undecidedLo
  | disagreeLo
  | undecidedHi
  | allAgree
  | edge (j : ℕ)
  | pair (j : ℕ)
  | triple (j : ℕ)
  | undecidedMid (j : ℕ)
  deriving DecidableEq

/-- Bisection between `lo`, which agrees, and `hi`, which disagrees, and the positions it
queried, in order. Where the midpoint is undecided and `recover`, its neighbours are queried. -/
def bisectBoundary (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ) :
    SearchResult × List ℕ :=
  if _h : hi ≤ lo + 1 then (.edge hi, [])
  else
    match agrees ((lo + hi) / 2) with
    | some true =>
      let r := bisectBoundary recover agrees ((lo + hi) / 2) hi
      (r.1, (lo + hi) / 2 :: r.2)
    | some false =>
      let r := bisectBoundary recover agrees lo ((lo + hi) / 2)
      (r.1, (lo + hi) / 2 :: r.2)
    | none =>
      if recover then
        match agrees ((lo + hi) / 2 - 1) with
        | none => (.pair ((lo + hi) / 2 - 1), [(lo + hi) / 2, (lo + hi) / 2 - 1])
        | some left =>
          let q := [(lo + hi) / 2, (lo + hi) / 2 - 1, (lo + hi) / 2 + 1]
          match agrees ((lo + hi) / 2 + 1), left with
          | none, _ => (.pair ((lo + hi) / 2), q)
          | some false, true => (.triple ((lo + hi) / 2), q)
          | some _, false =>
            let r := bisectBoundary recover agrees lo ((lo + hi) / 2 - 1)
            (r.1, q ++ r.2)
          | some true, true =>
            let r := bisectBoundary recover agrees ((lo + hi) / 2 + 1) hi
            (r.1, q ++ r.2)
      else (.undecidedMid ((lo + hi) / 2), [(lo + hi) / 2])
termination_by hi - lo
decreasing_by all_goals omega

/-- The search's result, and the positions it queried, in order. -/
def tripartiteSearch (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ) :
    SearchResult × List ℕ :=
  match agrees lo with
  | none => (.undecidedLo, [lo])
  | some false => (.disagreeLo, [lo])
  | some true =>
    match agrees hi with
    | none => (.undecidedHi, [lo, hi])
    | some true => (.allAgree, [lo, hi])
    | some false =>
      let r := bisectBoundary recover agrees lo hi
      (r.1, lo :: hi :: r.2)

/-- Each result says what `agrees` is where it points; the search queries only positions from
`lo` to `hi`, at most `2 + 3⌈log₂(hi − lo)⌉` of them; and where `agrees` turns from agreeing to
disagreeing at `t` with nothing undecided, it finds `t`. -/
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
    ∧ (∀ t, lo < t → t ≤ hi → (∀ i, lo ≤ i → i ≤ hi → agrees i = some (decide (i < t))) →
        r.1 = .edge t)

end OrthoDFA
