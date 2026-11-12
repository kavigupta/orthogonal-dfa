import Mathlib.Data.Nat.Log
import RLStar.Code.Query

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

/-- Bisection between `lo`, which agrees, and `hi`, which disagrees, asking whether each position
it reaches agrees. Where the midpoint is undecided and `recover`, its neighbours are asked. -/
def bisectQ (recover : Bool) (lo hi : ℕ) : Query ℕ (Option Bool) SearchResult :=
  if _h : hi ≤ lo + 1 then .pure (.edge hi)
  else .ask ((lo + hi) / 2) fun
    | some true => bisectQ recover ((lo + hi) / 2) hi
    | some false => bisectQ recover lo ((lo + hi) / 2)
    | none =>
      if recover then
        .ask ((lo + hi) / 2 - 1) fun
          | none => .pure (.pair ((lo + hi) / 2 - 1))
          | some left => .ask ((lo + hi) / 2 + 1) fun right =>
            match right, left with
            | none, _ => .pure (.pair ((lo + hi) / 2))
            | some false, true => .pure (.triple ((lo + hi) / 2))
            | some _, false => bisectQ recover lo ((lo + hi) / 2 - 1)
            | some true, true => bisectQ recover ((lo + hi) / 2 + 1) hi
      else .pure (.undecidedMid ((lo + hi) / 2))
termination_by hi - lo
decreasing_by all_goals omega

def tripartiteQ (recover : Bool) (lo hi : ℕ) : Query ℕ (Option Bool) SearchResult :=
  .ask lo fun
    | none => .pure .undecidedLo
    | some false => .pure .disagreeLo
    | some true => .ask hi fun
      | none => .pure .undecidedHi
      | some true => .pure .allAgree
      | some false => bisectQ recover lo hi

def tripartiteSearch (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ) : SearchResult :=
  (tripartiteQ recover lo hi).eval agrees

end OrthoDFA
