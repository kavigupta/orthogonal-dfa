import Mathlib.Algebra.FreeMonoid.Basic

/-! A tree that sifts strings to its leaves, reading each string followed by a node's midfix. -/

namespace OrthoDFA

variable {α : Type*}

/-- A tree whose nodes read the string followed by their midfix. -/
inductive DTree (α : Type*)
  | leaf
  | node (m : FreeMonoid α) (rej acc : DTree α)

namespace DTree

def leaves : DTree α → List (List Bool)
  | leaf => [[]]
  | node _ r a => r.leaves.map (false :: ·) ++ a.leaves.map (true :: ·)

def midfixAt : DTree α → List Bool → FreeMonoid α
  | leaf, _ => 1
  | node m _ _, [] => m
  | node _ r _, false :: p => r.midfixAt p
  | node _ _ a, true :: p => a.midfixAt p

def depth : DTree α → ℕ
  | leaf => 0
  | node _ r a => max r.depth a.depth + 1

/-- The strings sifting `z` reads, in order, and the path to its leaf, `none` where the last read
is undecided. -/
def siftWithPath (cut : FreeMonoid α → Option Bool) :
    DTree α → FreeMonoid α → List (FreeMonoid α) × Option (List Bool)
  | leaf, _ => ([], some [])
  | node m r a, z =>
    match cut (z * m) with
    | some true => (z * m :: (a.siftWithPath cut z).1, (a.siftWithPath cut z).2.map (true :: ·))
    | some false => (z * m :: (r.siftWithPath cut z).1, (r.siftWithPath cut z).2.map (false :: ·))
    | none => ([z * m], none)

end DTree

def SiftWithPathCorrect : Prop :=
  ∀ {α : Type*} (cut : FreeMonoid α → Option Bool) (T : DTree α) (z : FreeMonoid α),
    let reads := (T.siftWithPath cut z).1
    let path := (T.siftWithPath cut z).2
    -- The `i`th read is `z` followed by the midfix of the node the first `i` answers lead to.
    (∀ i (h : i < reads.length), reads[i] = z * T.midfixAt ((reads.filterMap cut).take i))
    -- One read per level at most.
    ∧ reads.length ≤ T.depth
    -- A decided sift ends at a leaf, along the path its answers spell, with every read decided.
    ∧ (∀ p, path = some p → p ∈ T.leaves ∧ p = reads.filterMap cut ∧ ∀ w ∈ reads, cut w ≠ none)
    -- An undecided sift stops at its first undecided read.
    ∧ (path = none →
        ∃ w, reads.getLast? = some w ∧ cut w = none ∧ ∀ w' ∈ reads.dropLast, cut w' ≠ none)

end OrthoDFA
