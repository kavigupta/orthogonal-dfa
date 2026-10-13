import OrthoDFA.DTree
import OrthoDFA.FamilyRead

/-! A hypothesis: a tree that sifts strings to its leaves, and edges between the leaves. -/

namespace OrthoDFA

variable {α : Type*}

namespace DTree

/-- The path to the leaf `z` sifts to, or the string whose read was undecided. -/
def sift (read : FreeMonoid α → ARU) : DTree α → FreeMonoid α → List Bool ⊕ FreeMonoid α
  | leaf, _ => .inl []
  | node m r a, z =>
    match read (z * m) with
    | .accept => (a.sift read z).map (true :: ·) id
    | .reject => (r.sift read z).map (false :: ·) id
    | .undecided => .inr (z * m)

/-- The leaf at `p` replaced by a node reading at `d`. -/
def splitAt (d : FreeMonoid α) : DTree α → List Bool → DTree α
  | leaf, [] => node d leaf leaf
  | node m r a, false :: p => node m (r.splitAt d p) a
  | node m r a, true :: p => node m r (a.splitAt d p)
  | t, _ => t

end DTree

def commonPrefix : List Bool → List Bool → List Bool
  | a :: as, b :: bs => if a = b then a :: commonPrefix as bs else []
  | _, _ => []

/-- Each learned edge's target, with a string at its leaf whose successor sifts there. -/
abbrev Edges (α : Type*) := List Bool × α → Option (List Bool × FreeMonoid α)

/-- The leaves the edges visit from `s` along `cs`, `s` first; it stops at an unlearned edge. -/
def walk (E : Edges α) : List Bool → List α → List (List Bool)
  | s, [] => [s]
  | s, c :: cs => s :: match E (s, c) with
    | some (t, _) => walk E t cs
    | none => []

def walkEnd (E : Edges α) : List Bool → List α → Option (List Bool)
  | s, [] => some s
  | s, c :: cs => (E (s, c)).bind fun e => walkEnd E e.1 cs

def prefixOf (x : FreeMonoid α) (i : ℕ) : FreeMonoid α := FreeMonoid.ofList (x.toList.take i)

structure Hypothesis (α : Type*) where
  tree : DTree α
  edges : Edges α

/-- The edges, walked from the leaf `x`'s first `k` letters sift to, end at a leaf other than the
one `x` sifts to. -/
def Disagrees (read : FreeMonoid α → ARU) (H : Hypothesis α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ a q e, H.tree.sift read (prefixOf x k) = .inl a ∧
    walkEnd H.edges a (x.toList.drop k) = some q ∧ H.tree.sift read x = .inl e ∧ q ≠ e

end OrthoDFA
