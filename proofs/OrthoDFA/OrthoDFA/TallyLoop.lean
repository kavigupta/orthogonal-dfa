import OrthoDFA.Loop

/-!
# The tally loop

Probes are fresh draws, each walked from `k` along the learned edges and searched for its first
disagreement (`probeOutcome`, whose bisection steps past an undecided midpoint whose neighbours
read alike). A clean disagreement records, at the edge it crosses, the probe's prefix there and the
leaf the next prefix sifts to; undecided outcomes are counted for the exits; a probe at an
unlearned edge learns it.

After every record the edges settle: where an edge has `m` records at a target it does not point
at, its leaf splits on the letter and the midfix where that target and the current one part if
the current one also has `m`, and otherwise it is redirected there. A split re-reads each record
of the leaf at the new node to place it at a child, re-reads each record into the leaf at the
new node to refine its target, drops the two triggering targets' records, and retargets the
edges into the leaf; the edges then settle again. Every change of the tree or the edges counts
in `version`.

How the round ends (harvests, success, its caps) is left as any rule `ends`. Undecided outcomes go
to one counter, not to the edge each is charged to, and an undecided re-read at a split is dropped
rather than charged to a child; the exits will need both per edge.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- What the loop carries: the hypothesis, its version, each edge's records (the probe's prefix at
the edge and the target leaf), and the undecided outcomes. -/
structure TState (α : Type*) where
  tree : DTree α
  edges : Edges α
  version : ℕ
  recs : List Bool → α → List (FreeMonoid α × List Bool)
  undecided : ℕ

/-- The records of the edge `(p, c)` at the target `t`. -/
def TState.tally (s : TState α) (p : List Bool) (c : α) (t : List Bool) : ℕ :=
  ((s.recs p c).filter (·.2 = t)).length

/-- The edge `(p, c)` has `m` records at a target `t` it does not point at. -/
def Violates (m : ℕ) (s : TState α) (p : List Bool) (c : α) (t : List Bool) : Prop :=
  (s.edges p c).map Prod.fst ≠ some t ∧ m ≤ s.tally p c t

/-- No edge has `m` records at a target it does not point at. -/
def Settled (m : ℕ) (s : TState α) : Prop := ∀ p c t, ¬ Violates m s p c t

/-- `s` with the edge `(p, c)` set to `e`, as a new version. -/
def TState.setEdge (s : TState α) (p : List Bool) (c : α) (e : List Bool × FreeMonoid α) :
    TState α :=
  let row := Function.update (s.edges p) c (some e)
  { s with edges := Function.update s.edges p row, version := s.version + 1 }

/-- `s` with the record `r` added at the edge `(p, c)`. -/
def TState.addRec (s : TState α) (p : List Bool) (c : α) (r : FreeMonoid α × List Bool) :
    TState α :=
  let row := Function.update (s.recs p) c (s.recs p c ++ [r])
  { s with recs := Function.update s.recs p row }

/-- The first state: the root reads at `ε`, no edge learned. -/
def tallyStart : TState α := ⟨.node 1 .leaf .leaf, fun _ _ => none, 0, fun _ _ => [], 0⟩

/-- The record a probe of `x` makes against the hypothesis `(t, e)`: at a clean disagreement, the
edge it crosses, the leaf its next prefix sifts to, and its prefix at the edge. -/
noncomputable def recordOf (R : CutReads α) (k : ℕ) (h : DTree α × Edges α) (x : FreeMonoid α) :
    Option ((List Bool × α × List Bool) × FreeMonoid α) :=
  match probeOutcome R h.1 h.2 k x with
  | .edge ps fd =>
    match x.toList[fd - 1]?, h.1.sift R.cut (prefixOf x fd) with
    | some c, .inl t => some ((ps.getD (fd - 1 - k) [], c, t), prefixOf x (fd - 1))
    | _, _ => none
  | _ => none

open scoped Classical in
/-- `s`'s records after its leaf `p` splits on `d` into `t'`: those at `p` re-read at the new node
to place them at a child, those into `p` re-read to refine their target, the triggering targets
`t₁` and `t₂` of the edge `(p, c₀)` dropped. An undecided re-read is dropped. -/
noncomputable def repartition (R : CutReads α) (p : List Bool) (c₀ : α) (t₁ t₂ : List Bool)
    (d : FreeMonoid α) (s : TState α) : List Bool → α → List (FreeMonoid α × List Bool) :=
  let reTarget : α → FreeMonoid α × List Bool → Option (FreeMonoid α × List Bool) := fun c r =>
    if r.2 = p then (R.cut (r.1 * FreeMonoid.of c * d)).map fun b => (r.1, p ++ [b]) else some r
  fun q c =>
    if h : ∃ b, q = p ++ [b] then
      (s.recs p c).filterMap fun r =>
        if c = c₀ ∧ (r.2 = t₁ ∨ r.2 = t₂) then none
        else if R.cut (r.1 * d) = some h.choose then reTarget c r else none
    else if q = p then []
    else (s.recs q c).filterMap (reTarget c)

open scoped Classical in
/-- Fix the edge `(p, c)` at the target `t` it has `m` records at: split where the current target
also has `m`, else redirect, with a record's prefix as the witness. -/
noncomputable def fixEdge (R : CutReads α) (m : ℕ) (s : TState α) (p : List Bool) (c : α)
    (t : List Bool) : TState α :=
  let w := (((s.recs p c).find? (·.2 = t)).map Prod.fst).getD 1
  let redirect : TState α := s.setEdge p c (t, w)
  match s.edges p c with
  | some (t₀, _) =>
    if m ≤ s.tally p c t₀ then
      let d := FreeMonoid.of c * s.tree.midAt (lcp t t₀)
      let t' := s.tree.splitAt d p
      { s with tree := t', edges := retarget R t' p s.edges, version := s.version + 1,
               recs := repartition R p c t t₀ d s }
    else redirect
  | none => redirect

open scoped Classical in
/-- The edges settle, fixing one edge at a time, within `fuel` fixes and `Lmax` leaves; `none`
where either runs out. -/
noncomputable def settleEdges (R : CutReads α) (m Lmax : ℕ) : ℕ → TState α → Option (TState α)
  | 0, s => if Settled m s then some s else none
  | fuel + 1, s =>
    if h : ∃ p c t, Violates m s p c t then
      if Lmax < (fixEdge R m s h.choose h.choose_spec.choose
          h.choose_spec.choose_spec.choose).tree.paths.length then none
      else settleEdges R m Lmax fuel
        (fixEdge R m s h.choose h.choose_spec.choose h.choose_spec.choose_spec.choose)
    else some s

/-- The loop's settings for its records and fixes. -/
structure TallyCfg where
  k : ℕ
  m : ℕ
  Lmax : ℕ
  fuel : ℕ

/-- A probe's outcome learned, recorded or counted. -/
noncomputable def tallyPre (C : TallyCfg) (R : CutReads α) (s : TState α) (x : FreeMonoid α) :
    TState α :=
  match probeOutcome R s.tree s.edges C.k x with
  | .member u =>
    match x.toList[u.toList.length]?, s.tree.sift R.cut u with
    | some c, .inl p =>
      match s.tree.sift R.cut (u * FreeMonoid.of c) with
      | .inl t => s.setEdge p c (t, u)
      | .inr _ => { s with undecided := s.undecided + 1 }
    | _, _ => { s with undecided := s.undecided + 1 }
  | .edge _ _ =>
    match recordOf R C.k (s.tree, s.edges) x with
    | some ((p, c, t), sp) => s.addRec p c (sp, t)
    | none => s
  | .agree => s
  | _ => { s with undecided := s.undecided + 1 }

open scoped Classical in
/-- One probe: the round ends where `ends` says so; otherwise its outcome is learned, recorded or
counted, and the edges settle. -/
noncomputable def tallyStep (C : TallyCfg) (R : CutReads α)
    (ends : TState α → FreeMonoid α → Prop) (s : TState α) (x : FreeMonoid α) :
    Option (TState α) :=
  if ends s x then none else settleEdges R C.m C.Lmax C.fuel (tallyPre C R s x)

end OrthoDFA
