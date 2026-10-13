import OrthoDFA.DTree

/-! A hypothesis DFA: a tree's leaves as states, edges between them, and the walk a string takes. -/

namespace OrthoDFA

variable {α : Type*}

/-- Each learned edge's target, with a string at its leaf whose successor sifts there. -/
abbrev Edges (α : Type*) := List Bool × α → Option (List Bool × FreeMonoid α)

structure HypothesisDFA (α : Type*) where
  tree : DTree α
  edges : Edges α

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

/-- The edges, walked from the leaf `x`'s first `k` letters sift to, end at a leaf other than the
one `x` sifts to. -/
def Disagrees (read : FreeMonoid α → ARU) (H : HypothesisDFA α) (k : ℕ) (x : FreeMonoid α) :
    Prop :=
  ∃ a q e, H.tree.sift read (prefixOf x k) = .inl a ∧
    walkEnd H.edges a (x.toList.drop k) = some q ∧ H.tree.sift read x = .inl e ∧ q ≠ e

def WalkCorrect : Prop :=
  ∀ {α : Type*},
    (∀ (H : HypothesisDFA α) (s : List Bool) (cs : List α),
      let ss := walk H.edges s cs
      -- The walk starts at `s` and visits at most one leaf per letter after it.
      ss.head? = some s ∧ ss.length ≤ cs.length + 1
      -- Each step follows the learned edge on the next letter,
      ∧ (∀ i, i + 1 < ss.length →
          ∃ c w, cs[i]? = some c ∧ H.edges (ss.getD i [], c) = some (ss.getD (i + 1) [], w))
      -- and it stops short only at an unlearned edge.
      ∧ (ss.length ≤ cs.length →
          ∃ c, cs[ss.length - 1]? = some c ∧ H.edges (ss.getD (ss.length - 1) [], c) = none)
      -- `walkEnd` is where a walk that follows every letter ends.
      ∧ (∀ q, walkEnd H.edges s cs = some q ↔ ss.length = cs.length + 1 ∧ ss.getLast? = some q)
      -- Where every learned edge out of a leaf leads to a leaf, a walk from a leaf stays on them.
      ∧ ((∀ q ∈ H.tree.leaves, ∀ c t w, H.edges (q, c) = some (t, w) → t ∈ H.tree.leaves) →
          s ∈ H.tree.leaves → ∀ q ∈ ss, q ∈ H.tree.leaves))
    ∧ (∀ (x : FreeMonoid α) (i : ℕ),
      -- A prefix grows by the next letter, and stops at the whole string.
      (∀ c, x.toList[i]? = some c → prefixOf x (i + 1) = prefixOf x i * FreeMonoid.of c)
      ∧ (x.toList.length ≤ i → prefixOf x i = x))

end OrthoDFA
