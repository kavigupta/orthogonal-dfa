import Mathlib.Data.Nat.Log
import OrthoDFA.Code.Query

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

section Query

variable {Q R : Type*}

def bisectQ (recover : Bool) (agrees : ℕ → Query Q R (Option Bool)) (lo hi : ℕ) :
    Query Q R SearchResult :=
  if _h : hi ≤ lo + 1 then .pure (.edge hi)
  else (agrees ((lo + hi) / 2)).bind fun
    | some true => bisectQ recover agrees ((lo + hi) / 2) hi
    | some false => bisectQ recover agrees lo ((lo + hi) / 2)
    | none =>
      if recover then
        (agrees ((lo + hi) / 2 - 1)).bind fun
          | none => .pure (.pair ((lo + hi) / 2 - 1))
          | some left => (agrees ((lo + hi) / 2 + 1)).bind fun right =>
            match right, left with
            | none, _ => .pure (.pair ((lo + hi) / 2))
            | some false, true => .pure (.triple ((lo + hi) / 2))
            | some _, false => bisectQ recover agrees lo ((lo + hi) / 2 - 1)
            | some true, true => bisectQ recover agrees ((lo + hi) / 2 + 1) hi
      else .pure (.undecidedMid ((lo + hi) / 2))
termination_by hi - lo
decreasing_by all_goals omega

def tripartiteQ (recover : Bool) (agrees : ℕ → Query Q R (Option Bool)) (lo hi : ℕ) :
    Query Q R SearchResult :=
  (agrees lo).bind fun
    | none => .pure .undecidedLo
    | some false => .pure .disagreeLo
    | some true => (agrees hi).bind fun
      | none => .pure .undecidedHi
      | some true => .pure .allAgree
      | some false => bisectQ recover agrees lo hi

end Query

end OrthoDFA
