import OrthoDFA.Stage
import OrthoDFA.Automaton

/-!
# The counterexample pass

`TransitionResolver.counterexample_pass` over a round's cut: probes walked through the hypothesis
and re-sifted, leaves split where the evidence confirms a disagreement, and what the cut cannot
place harvested.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : Stringlike (FreeMonoid α) where
  measurableSet_singleton _ := trivial
  measurable_const_mul _ _ _ := trivial
  measurable_mul_const _ _ _ := trivial
  exists_injective_nat' := Countable.exists_injective_nat (List α)
  decEq := fun a b => (inferInstance : DecidableEq (List α)) a b

namespace DTree

/-- `t` with the leaf at the end of `path` split on `m`. -/
def splitAt (m : FreeMonoid α) : DTree α → List Bool → DTree α
  | .leaf, [] => .node m .leaf .leaf
  | .node n r a, false :: p => .node n (r.splitAt m p) a
  | .node n r a, true :: p => .node n r (a.splitAt m p)
  | t, _ => t

/-- `first_disagreement`: walking down the branch on which `x · pre` and `y · pre` agree, the
first node where they part, as `pre` followed by its midfix; `none` if a read there cannot be
placed or they reach a leaf together. -/
def firstDisagreement (cut : FreeMonoid α → Option Bool) (x y pre : FreeMonoid α) :
    DTree α → Option (FreeMonoid α)
  | .leaf => none
  | .node m r a =>
    match cut (x * (pre * m)), cut (y * (pre * m)) with
    | some true, some true => a.firstDisagreement cut x y pre
    | some false, some false => r.firstDisagreement cut x y pre
    | some _, some _ => some (pre * m)
    | _, _ => none

end DTree

/-- `SplitEvidence`'s verdicts. -/
inductive Verdict
  | split
  | noSplit
  | undecided

/-- The pass's knobs.  `train F` is the family's training half; the rest is its test half. -/
structure StageKnobs (α : Type*) where
  train : Finset (FreeMonoid α) → Finset (FreeMonoid α)
  memberLimit : ℕ
  patience : ℕ
  splitFpr : ℝ
  minSplit : ℝ
  missRate : ℝ
  /-- `BoundarySource` keeps only boundaries of prefixes at least this long. -/
  longEnough : ℕ

/-- What the pass carries from probe to probe: the tree, the population in the order its strings
arrived, each edge's target and witness, the probes since the last split, how many of those a
read the cut could not place kept from being checked, the strings the cut could not place, and
the shorter string of each disagreement the evidence said was no split. -/
structure PassState (α : Type*) where
  tree : DTree α
  pool : List (FreeMonoid α)
  edges : List Bool → α → Option (List Bool × FreeMonoid α)
  streak : ℕ
  unchecked : ℕ
  boundary : List (FreeMonoid α)
  disagreements : List (FreeMonoid α)

variable (K : StageKnobs α) (R : CutReads α)

/-- A leaf's members: the first `memberLimit` of the population the tree sends to it. -/
noncomputable def members (t : DTree α) (pool : List (FreeMonoid α)) (path : List Bool) :
    List (FreeMonoid α) :=
  (pool.filter fun s => decide (t.sift R.cut s = .inl path)).take K.memberLimit

/-- `decisive_target`: where most of the leaf's members, extended by `c`, sift, with the first
member voting for it as witness; ties keep `current`.  `none` when no member places. -/
noncomputable def decisiveTarget (t : DTree α) (pool : List (FreeMonoid α)) (path : List Bool)
    (c : α) (current : Option (List Bool)) : Option (List Bool × FreeMonoid α) :=
  let votes := (members K R t pool path).filterMap fun m =>
    match t.sift R.cut (m * FreeMonoid.of c) with
    | .inl p => some (p, m)
    | .inr _ => none
  let count := fun p => (votes.filter fun v => v.1 = p).length
  votes.foldl (fun best v =>
    match best with
    | none => some v
    | some b =>
      if count b.1 < count v.1 ∨ (count b.1 = count v.1 ∧ some v.1 = current ∧ some b.1 ≠ current)
      then some v else some b) none

/-- What `decisive_target` harvests on the way: the members, extended by `c`, that the tree
cannot place, up to the first it can. -/
noncomputable def edgeMisses (t : DTree α) (c : α) :
    List (FreeMonoid α) → List (FreeMonoid α)
  | [] => []
  | m :: ms =>
    match t.sift R.cut (m * FreeMonoid.of c) with
    | .inl _ => []
    | .inr b => b :: edgeMisses t c ms

/-- `close`: every edge re-voted; an edge no member can place keeps its target. -/
noncomputable def closeEdges (t : DTree α) (pool : List (FreeMonoid α))
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) :
    List Bool → α → Option (List Bool × FreeMonoid α) :=
  fun path c => (decisiveTarget K R t pool path c ((edges path c).map Prod.fst)).orElse
    fun _ => edges path c

/-- The totalised step on a leaf's path: the edge's target when it is a leaf, else stay put. -/
def stepPath (t : DTree α) (edges : List Bool → α → Option (List Bool × FreeMonoid α))
    (path : List Bool) (c : α) : List Bool :=
  match edges path c with
  | some (p, _) => if p ∈ t.paths then p else path
  | none => path

/-- `anchored_walk`: the first prefix of `w` the tree places, and the paths the walk from it
visits, one per letter after it. -/
noncomputable def anchoredWalk (t : DTree α)
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) (w : FreeMonoid α) :
    Option (ℕ × List (List Bool)) :=
  match (List.range (w.toList.length + 1)).findSome? fun i =>
      match t.sift R.cut (prefixOf w i) with
      | .inl p => some (i, p)
      | .inr _ => none with
  | none => none
  | some (i, p) => some (i, (w.toList.drop i).scanl (stepPath t edges) p)

/-- `first_disagreeing_edge`: between an index where the sift agrees with the walk and one where
it does not, the first where it does not, or what a sift on the way could not place.  `walk j`
is the walk's path after `j` letters. -/
noncomputable def firstDisagreeingEdge (t : DTree α) (w : FreeMonoid α)
    (walk : ℕ → List Bool) : ℕ → ℕ → ℕ → FreeMonoid α ⊕ ℕ
  | 0, _, hi => .inr hi
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      match t.sift R.cut (prefixOf w ((lo + hi) / 2)) with
      | .inr b => .inl b
      | .inl p =>
        if p = walk ((lo + hi) / 2) then firstDisagreeingEdge t w walk fuel ((lo + hi) / 2) hi
        else firstDisagreeingEdge t w walk fuel lo ((lo + hi) / 2)
    else .inr hi

/-- What `anchored_walk` harvests: the prefixes of `w` it sifts before the first the tree
places. -/
noncomputable def anchorMisses (t : DTree α) (w : FreeMonoid α) : List (FreeMonoid α) :=
  ((List.range (w.toList.length + 1)).map fun i => t.sift R.cut (prefixOf w i)).takeWhile
    (·.isRight) |>.filterMap Sum.getRight?

/-- The two-group test's tally: over the members the training half places, the test half's
accepts and the member count on each side, accepting side first. -/
noncomputable def tally (t : DTree α) (pool : List (FreeMonoid α)) (path : List Bool)
    (d : FreeMonoid α) : (ℕ × ℕ) × (ℕ × ℕ) :=
  (members K R t pool path).foldl (fun acc m =>
    let s := acceptsOn (K.train R.F) R.f (m * d)
    let e := acceptsOn (R.F \ K.train R.F) R.f (m * d)
    if R.B.hi * (K.train R.F).card < R.F.card * s then ((acc.1.1 + e, acc.1.2 + 1), acc.2)
    else if R.F.card * s ≤ R.B.lo * (K.train R.F).card then (acc.1, (acc.2.1 + e, acc.2.2 + 1))
    else acc) ((0, 0), (0, 0))

/-- `SplitEvidence.verdict`: split when the two sides' held-out votes differ in rate by more than
Hoeffding allows at `splitFpr` over `tests` tests; no split when one side is too small for a
`minSplit` share to be missed more than `missRate` of the time; otherwise undecided. -/
noncomputable def verdict (t : DTree α) (pool : List (FreeMonoid α)) (path : List Bool)
    (d : FreeMonoid α) (tests : ℕ) : Verdict :=
  let x := tally K R t pool path d
  let E : ℝ := (R.F \ K.train R.F).card
  if 0 < x.1.2 ∧ 0 < x.2.2 ∧ 0 < E ∧
      Real.log (2 * tests / K.splitFpr)
        ≤ 2 * E * ((x.1.2 * x.2.2 : ℕ) : ℝ) / (x.1.2 + x.2.2)
          * ((x.1.1 : ℝ) / (x.1.2 * E) - (x.2.1 : ℝ) / (x.2.2 * E)) ^ 2 then .split
  else if 0 < x.1.2 + x.2.2
      ∧ 1 - binomSfGe (x.1.2 + x.2.2) K.minSplit (min x.1.2 x.2.2 + 1) ≤ K.missRate
  then .noSplit
  else .undecided

/-- Re-vote every edge, harvesting what that cannot place, and carry the streak. -/
noncomputable def settle (t : DTree α) (pool : List (FreeMonoid α))
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) (streak unchecked : ℕ)
    (boundary disagreements : List (FreeMonoid α)) : PassState α :=
  { tree := t, pool := pool, edges := closeEdges K R t pool edges, streak := streak,
    unchecked := unchecked,
    boundary := boundary ++ t.paths.flatMap fun p =>
      (Finset.univ : Finset α).toList.flatMap fun c => edgeMisses R t c (members K R t pool p),
    disagreements := disagreements }

/-- After a probe's walk and sift part at `fd`: the disagreement on the edge into `fd`.  The edge
the walk took splits its leaf when the evidence confirms it, and the two strings that exhibited it
join the population; otherwise the probe's own string at the leaf joins it ahead of the rest, so
it is a member however many the leaf holds, and is harvested when the evidence says no split. -/
noncomputable def onEdge (s : PassState α) (w : FreeMonoid α) (walkAt : ℕ → List Bool)
    (pool boundary : List (FreeMonoid α)) (fd : ℕ) : PassState α :=
  let t := s.tree
  let clean := settle K R t pool s.edges (s.streak + 1) s.unchecked boundary s.disagreements
  match w.toList[fd - 1]? with
  | none => clean
  | some c =>
    let s1 := walkAt (fd - 1)
    let sprime := prefixOf w (fd - 1)
    match s.edges s1 c with
    | none => clean
    | some (s2, x) =>
      if s2 ≠ walkAt fd ∨ t.sift R.cut x ≠ .inl s1 ∨ t.sift R.cut sprime ≠ .inl s1 then clean
      else
      match t.firstDisagreement R.cut x sprime (FreeMonoid.of c) with
      | none => clean
      | some d =>
        let kept := sprime :: pool.filter (· ≠ sprime)
        match verdict K R t pool s1 d (t.paths.length * Fintype.card α) with
        | .split =>
          let cleared := fun p c' => match s.edges p c' with
            | some (q, y) => if p = s1 ∨ q = s1 then none else some (q, y)
            | none => none
          settle K R (t.splitAt d s1) (pool ++ ([x, sprime].filter (· ∉ pool))) cleared 0 0
            boundary s.disagreements
        | .noSplit =>
          settle K R t kept s.edges (s.streak + 1) s.unchecked boundary
            (s.disagreements ++ [sprime])
        | .undecided => settle K R t kept s.edges 0 0 boundary s.disagreements

/-- What one probe does: `_process` and `_act_on_disagreement`.  The anchor joins the
population.  A probe with no prefix the tree places, whose own sift cannot be placed, or whose
search for the disagreeing edge meets a read it cannot place, goes unchecked, and what could not
be placed is harvested; one whose walk ends where it sifts is clean.  Then every edge is
re-voted. -/
noncomputable def probeStep (s : PassState α) (w : FreeMonoid α) : PassState α :=
  let t := s.tree
  let boundary := s.boundary ++ anchorMisses R t w
  match anchoredWalk R t s.edges w with
  | none =>
    settle K R t s.pool s.edges (s.streak + 1) (s.unchecked + 1) boundary s.disagreements
  | some (start, walk) =>
    let pool := if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]
    let n := w.toList.length
    let walkAt := fun j => walk.getD (j - start) []
    let unchecked := fun b =>
      settle K R t pool s.edges (s.streak + 1) (s.unchecked + 1) (boundary ++ [b])
        s.disagreements
    match t.sift R.cut w with
    | .inr b => unchecked b
    | .inl actual =>
      if actual = walkAt n then
        settle K R t pool s.edges (s.streak + 1) s.unchecked boundary s.disagreements
      else
      match firstDisagreeingEdge R t w walkAt n start n with
      | .inl b => unchecked b
      | .inr fd => onEdge K R s w walkAt pool boundary fd

/-- The pass: probes in order until `patience` in a row have neither split a leaf nor left the
evidence undecided. -/
noncomputable def runPass (s : PassState α) (probes : List (FreeMonoid α)) : PassState α :=
  probes.foldl (fun s w => if K.patience ≤ s.streak then s else probeStep K R s w) s

/-- The first state: the root reads at `ε`, the population is the table's prefixes, and the
edges are closed once. -/
noncomputable def initialState (seed : List (FreeMonoid α)) : PassState α :=
  settle K R (.node 1 .leaf .leaf) seed (fun _ _ => none) 0 0 [] []

/-- The pass ran out of patience, and most of the probes since its last split went unchecked:
the round halves the limits every later family is held to. -/
def PassState.halves (s : PassState α) : Prop := K.patience ≤ s.streak ∧ s.streak < 2 * s.unchecked


/-- The hypothesis a pass state names: its tree and the targets of its edges. -/
def PassState.hyp (s : PassState α) : Hypothesis α :=
  ⟨s.tree, fun p c => (s.edges p c).map Prod.fst⟩

end OrthoDFA
