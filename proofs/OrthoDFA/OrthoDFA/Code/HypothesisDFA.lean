import OrthoDFA.Code.DTree

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

section Split

/-- The edges after the leaf `p` splits into `T'`: those into `p` sift their witness's successor
again, those out of it are dropped. -/
def retarget (read : FreeMonoid α → ARU) (T' : DTree α) (p : List Bool) (E : Edges α) :
    Edges α := fun e =>
  if p.isPrefixOf e.1 then none
  else
    match E e with
    | some (t, w) =>
      if t = p then
        (T'.sift read (w * FreeMonoid.of e.2)).elim (fun t' => some (t', w)) fun _ => none
      else some (t, w)
    | none => none

/-- `retarget` at the edge `e`, as a program. -/
def retargetAtQ (T' : DTree α) (p : List Bool) (E : Edges α) (e : List Bool × α) :
    Query (FreeMonoid α) ARU (Option (List Bool × FreeMonoid α)) :=
  if p.isPrefixOf e.1 then .pure none
  else
    match E e with
    | some (t, w) =>
      if t = p then
        (T'.siftQ (w * FreeMonoid.of e.2)).map fun r =>
          r.elim (fun t' => some (t', w)) fun _ => none
      else .pure (some (t, w))
    | none => .pure none

/-- `retarget` as a program over the edges `es`, every other edge unlearned. -/
def retargetQ [DecidableEq α] (T' : DTree α) (p : List Bool) (E : Edges α) :
    List (List Bool × α) → Query (FreeMonoid α) ARU (Edges α)
  | [] => .pure fun _ => none
  | e :: es => (retargetAtQ T' p E e).bind fun v =>
    (retargetQ T' p E es).map fun E' => Function.update E' e v

/-- The leaf `p` split into a node reading at `d`, and the edges retargeted. -/
def HypothesisDFA.split (read : FreeMonoid α → ARU) (H : HypothesisDFA α) (d : FreeMonoid α)
    (p : List Bool) : HypothesisDFA α :=
  ⟨H.tree.splitAt d p, retarget read (H.tree.splitAt d p) p H.edges⟩

/-- Every learned edge out of a leaf is witnessed: a string at the leaf whose successor sifts to
the edge's target. -/
def EdgesOK (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) : Prop :=
  ∀ q ∈ T.leaves, ∀ c t w, E (q, c) = some (t, w) →
    T.sift read w = .inl q ∧ T.sift read (w * FreeMonoid.of c) = .inl t

end Split

end OrthoDFA
