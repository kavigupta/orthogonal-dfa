import OrthoDFA.Learner

/-!
# The L\* stage

A discrimination tree over the family's cut, in the manner of Kearns and Vazirani.  Every node
reads a string at its midfix, through the family's vote, and sends it to the side the cut
decides; every leaf is a state of the hypothesis, named by its access string.  A probe on which
the hypothesis and the tree disagree splits a leaf; a probe some node cannot place is set aside,
and what the tree cannot place among long prefixes is the harvest.

Known modelling gap.  `TransitionResolver` takes an edge's target from a majority over the
leaf's members; here it is where the leaf's access string, extended by the letter, sifts.

Known modelling gap.  `SplitEvidence` splits a leaf only on a Bayes factor over its members;
here the two reads that disagree split it.

Known modelling gap.  `anchored_walk` starts the walk at the shortest prefix the tree places;
here it starts at `ε`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : MeasurableSpace (FreeMonoid α) := ⊤

instance : Stringlike (FreeMonoid α) where
  measurableSet_singleton _ := trivial
  measurable_const_mul _ _ _ := trivial
  measurable_mul_const _ _ _ := trivial
  exists_injective_nat' := Countable.exists_injective_nat (List α)
  decEq := fun a b => (inferInstance : DecidableEq (List α)) a b

/-- A discrimination tree.  A node's accepting side is `acc`, its rejecting side `rej`. -/
inductive DTree (α : Type*)
  | leaf (access : FreeMonoid α)
  | node (midfix : FreeMonoid α) (rej acc : DTree α)

open scoped Classical in
/-- The cut of `F` at `B` on `x`, read through `f`: `some true` above `hi`, `some false` at or
below `lo`, `none` between. -/
noncomputable def cutOn (B : State) (F : Finset (FreeMonoid α)) (f : FreeMonoid α → ℝ)
    (x : FreeMonoid α) : Option Bool :=
  let c := (F.filter fun v => f (x * v) = 1).card
  if B.hi < c then some true else if c ≤ B.lo then some false else none

namespace DTree

/-- The path to the leaf `x` sifts to, `true` for an accepting side, or the string a node could
not place. -/
def sift (cut : FreeMonoid α → Option Bool) : DTree α → FreeMonoid α → List Bool ⊕ FreeMonoid α
  | .leaf _, _ => .inl []
  | .node m r a, x =>
    match cut (x * m) with
    | none => .inr (x * m)
    | some true => (a.sift cut x).map (true :: ·) id
    | some false => (r.sift cut x).map (false :: ·) id

/-- Every leaf, with its path, rejecting sides first. -/
def leaves : DTree α → List (List Bool × FreeMonoid α)
  | .leaf a => [([], a)]
  | .node _ r a =>
    (r.leaves.map fun l => (false :: l.1, l.2)) ++ (a.leaves.map fun l => (true :: l.1, l.2))

/-- The midfix of the node at the end of `path`. -/
def midfixAt : DTree α → List Bool → Option (FreeMonoid α)
  | .node m _ _, [] => some m
  | .node _ r _, false :: p => r.midfixAt p
  | .node _ _ a, true :: p => a.midfixAt p
  | .leaf _, _ => none

/-- `t` with the leaf at the end of `path` replaced by `s`. -/
def replaceAt (s : DTree α) : DTree α → List Bool → DTree α
  | .leaf _, [] => s
  | .node m r a, false :: p => .node m (r.replaceAt s p) a
  | .node m r a, true :: p => .node m r (a.replaceAt s p)
  | t, _ => t

/-- The state a path names: its place among the leaves, `0` past the last of `N + 1`. -/
def stateOf (t : DTree α) (N : ℕ) (path : List Bool) : Fin (N + 1) :=
  let i := t.leaves.findIdx (·.1 = path)
  if h : i < N + 1 then ⟨i, h⟩ else 0

/-- One letter's step: where the state's access string, extended by `c`, sifts.  A state with
no leaf, or a step the cut cannot place, stays put. -/
def letterStep (t : DTree α) (cut : FreeMonoid α → Option Bool) (N : ℕ) (q : Fin (N + 1))
    (c : α) : Fin (N + 1) :=
  match t.leaves[q.val]? with
  | none => q
  | some l => match t.sift cut (l.2 * FreeMonoid.of c) with
    | .inl path => t.stateOf N path
    | .inr _ => q

/-- The hypothesis the tree names, over `N + 1` states: the leaves in order, then states no
string reaches.  It starts where `ε` sifts and accepts on the root's accepting side. -/
def hyp (t : DTree α) (cut : FreeMonoid α → Option Bool) (N : ℕ) :
    DFA (FreeMonoid α) (Fin (N + 1)) where
  step q w := w.toList.foldl (t.letterStep cut N) q
  step_one _ := rfl
  step_mul q a b := by
    simp only [FreeMonoid.toList_mul, List.foldl_append]
  start := match t.sift cut 1 with
    | .inl path => t.stateOf N path
    | .inr _ => 0
  accept := {q | ∃ l ∈ t.leaves[q.val]?, l.1.head? = some true}

end DTree

/-- The first `i` letters of `w`. -/
def prefixOf (w : FreeMonoid α) (i : ℕ) : FreeMonoid α := FreeMonoid.ofList (w.toList.take i)

/-- What a probe does to the tree: a new tree if it splits a leaf, nothing if the tree cannot
place one of the strings it reads, or if it agrees with the hypothesis. -/
def DTree.process (t : DTree α) (cut : FreeMonoid α → Option Bool) (N : ℕ) (w : FreeMonoid α) :
    Option (DTree α) :=
  let H := t.hyp cut N
  let n := w.toList.length
  -- The first prefix the walk and the sift disagree on.
  match (List.range (n + 1)).find? (fun i =>
      match t.sift cut (prefixOf w i) with
      | .inl path => decide (t.stateOf N path ≠ H.state (prefixOf w i))
      | .inr _ => true) with
  | none => none
  | some 0 => none
  | some (i + 1) =>
    match t.sift cut (prefixOf w (i + 1)), w.toList[i]?,
        t.leaves[(H.state (prefixOf w i)).val]? with
    | .inl p₁, some c, some (pj, a) =>
      match t.sift cut (a * FreeMonoid.of c) with
      | .inl p₂ =>
        -- The node where `prefixOf w i · c` and `a · c` part.
        let lca := (p₁.zip p₂).takeWhile (fun x => x.1 = x.2) |>.map Prod.fst
        match t.midfixAt lca with
        | some m =>
          let d := FreeMonoid.of c * m
          let u := prefixOf w i
          match cut (a * d), cut (u * d) with
          | some true, some false => some (t.replaceAt (.node d (.leaf u) (.leaf a)) pj)
          | some false, some true => some (t.replaceAt (.node d (.leaf a) (.leaf u)) pj)
          | _, _ => none
        | none => none
      | .inr _ => none
    | _, _, _ => none

/-- The pass: probes in order, each splitting a leaf or not, stopping once `patience` in a row
have not. -/
def DTree.pass (cut : FreeMonoid α → Option Bool) (N patience : ℕ) (t : DTree α)
    (probes : List (FreeMonoid α)) : DTree α :=
  (probes.foldl (fun (s : DTree α × ℕ) w =>
    if patience ≤ s.2 then s else
    match s.1.process cut N w with
    | some t' => (t', 0)
    | none => (s.1, s.2 + 1)) (t, 0)).1

/-- The first tree: the root reads a string at `ε`, and a first decided draw of each side names
its leaf. -/
def DTree.init (cut : FreeMonoid α → Option Bool) (draws : List (FreeMonoid α)) : DTree α :=
  .node 1 (.leaf ((draws.find? (cut · = some false)).getD 1))
    (.leaf ((draws.find? (cut · = some true)).getD 1))

/-- The harvest: the first string the tree cannot place while sifting a probe's prefixes of
length at least `ℓ₀`. -/
def DTree.harvestOf (t : DTree α) (cut : FreeMonoid α → Option Bool) (ℓ₀ : ℕ)
    (w : FreeMonoid α) : Option (FreeMonoid α) :=
  ((List.range (w.toList.length + 1)).filter (ℓ₀ ≤ ·)).findSome? fun i =>
    match t.sift cut (prefixOf w i) with
    | .inr b => some b
    | .inl _ => none

end OrthoDFA
