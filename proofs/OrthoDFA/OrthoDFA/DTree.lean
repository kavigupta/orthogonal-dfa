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

def midAt : DTree α → List Bool → FreeMonoid α
  | leaf, _ => 1
  | node m _ _, [] => m
  | node _ r _, false :: p => r.midAt p
  | node _ _ a, true :: p => a.midAt p

def depth : DTree α → ℕ
  | leaf => 0
  | node _ r a => max r.depth a.depth + 1

/-- The strings sifting `z` reads, in order, and the path to its leaf, `none` where the last read
is undecided. -/
def route (cut : FreeMonoid α → Option Bool) :
    DTree α → FreeMonoid α → List (FreeMonoid α) × Option (List Bool)
  | leaf, _ => ([], some [])
  | node m r a, z =>
    match cut (z * m) with
    | some true => (z * m :: (a.route cut z).1, (a.route cut z).2.map (true :: ·))
    | some false => (z * m :: (r.route cut z).1, (r.route cut z).2.map (false :: ·))
    | none => ([z * m], none)

end DTree

/-- Routing reads `z` followed by the midfix of each node on its way, the `i`th at the node the
first `i` decided reads lead to, and no more of them than the tree is deep. A decided route ends at
a leaf, along the path its reads decide; an undecided one ends at its only undecided read. -/
def RouteCorrect : Prop :=
  ∀ {α : Type*} (cut : FreeMonoid α → Option Bool) (T : DTree α) (z : FreeMonoid α),
    let r := T.route cut z
    (∀ i (h : i < r.1.length), r.1[i] = z * T.midAt ((r.1.filterMap cut).take i))
    ∧ r.1.length ≤ T.depth
    ∧ (∀ p, r.2 = some p → p ∈ T.leaves ∧ p = r.1.filterMap cut ∧ ∀ w ∈ r.1, cut w ≠ none)
    ∧ (r.2 = none →
        ∃ w, r.1.getLast? = some w ∧ cut w = none ∧ ∀ w' ∈ r.1.dropLast, cut w' ≠ none)

end OrthoDFA
