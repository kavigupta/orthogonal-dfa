import OrthoDFA.Stage
import OrthoDFA.Automaton

/-!
# The counterexample pass

`TransitionResolver.counterexample_pass` over a round's cut: probes walked through the hypothesis
from where the gate places `ε` and re-sifted, leaves split where the evidence confirms a
disagreement, and what the cut cannot place harvested: into the bisection population when the
probe disagrees, else into the boundary population.
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

/-- `x`'s way down the tree as the gate reads it, the middle of the band `mid` deciding wherever
`cut` cannot: its leaf, and the strings read in the band on the way. -/
def halfway (cut : FreeMonoid α → Option Bool) (mid : FreeMonoid α → Bool) :
    DTree α → FreeMonoid α → List Bool × List (FreeMonoid α)
  | .leaf, _ => ([], [])
  | .node m r a, x =>
    let band := if (cut (x * m)).isNone then [x * m] else []
    match (cut (x * m)).getD (mid (x * m)) with
    | true => (true :: (a.halfway cut mid x).1, band ++ (a.halfway cut mid x).2)
    | false => (false :: (r.halfway cut mid x).1, band ++ (r.halfway cut mid x).2)

end DTree

/-- The gate's read of `y`: past the middle of the band. -/
noncomputable def CutReads.mid (R : CutReads α) (y : FreeMonoid α) : Bool :=
  decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f y)

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
arrived, each edge's target and witness, the probes since the last split or undecided evidence,
how many of those a read the cut could not place kept from being checked, the strings the cut
could not place, and the bisection population. -/
structure PassState (α : Type*) where
  tree : DTree α
  pool : List (FreeMonoid α)
  edges : List Bool → α → Option (List Bool × FreeMonoid α)
  streak : ℕ
  unchecked : ℕ
  /-- The node reads the probes since the last split made. -/
  reads : ℕ
  boundary : List (FreeMonoid α)
  /-- What disagreeing probes could not place: the first read the search for the disagreeing
  edge meets undecided, the prefix before an edge the hypothesis does not hold that it lands on,
  or every read in the band on the gate's reading of a probe the cut cannot place that it sends
  off the walk; and every read in the band on the gate's reading of `ε`. -/
  bisected : List (FreeMonoid α)

variable (K : StageKnobs α) (R : CutReads α)

/-- Where the gate's reading sends `x`: down the tree by the middle of the band. -/
noncomputable def midLeaf (t : DTree α) (x : FreeMonoid α) : List Bool :=
  (t.sift (fun y => some (R.mid y)) x).elim id fun _ => []

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
    (boundary bisected : List (FreeMonoid α)) : PassState α :=
  { tree := t, pool := pool, edges := closeEdges K R t pool edges, streak := streak,
    unchecked := unchecked, reads := 0,
    boundary := boundary ++ t.paths.flatMap fun p =>
      (Finset.univ : Finset α).toList.flatMap fun c => edgeMisses R t c (members K R t pool p),
    bisected := bisected }

/-- After a probe's walk and sift part at `fd`: the disagreement on the edge into `fd`.  One the
walk took on an edge the hypothesis does not hold holds its prefix in the bisection population.
The edge the walk took splits its leaf when the evidence confirms it, and the two strings that
exhibited it join the population; otherwise the probe's own string at the leaf joins it ahead of
the rest, so it is a member however many the leaf holds, and the pass keeps probing whether the
evidence says no split or is undecided (#412). -/
noncomputable def onEdge (s : PassState α) (w : FreeMonoid α) (walkAt : ℕ → List Bool)
    (pool boundary held : List (FreeMonoid α)) (fd : ℕ) : PassState α :=
  let t := s.tree
  let clean := settle K R t pool s.edges (s.streak + 1) s.unchecked boundary held
  match w.toList[fd - 1]? with
  | none => clean
  | some c =>
    let s1 := walkAt (fd - 1)
    let sprime := prefixOf w (fd - 1)
    let placeholder := settle K R t pool s.edges (s.streak + 1) s.unchecked boundary
      (held ++ [sprime])
    match s.edges s1 c with
    | none => placeholder
    | some (s2, x) =>
      if s2 ≠ walkAt fd then placeholder
      else if t.sift R.cut x ≠ .inl s1 ∨ t.sift R.cut sprime ≠ .inl s1 then clean
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
            boundary held
        | _ => settle K R t kept s.edges 0 0 boundary held

/-- The walk a probe is checked against, and the index the sift is known to agree with it at:
from where the gate places `ε`, unless that start already parts from the leaf the cut places the
shortest prefix it can at, in which case from that prefix. -/
noncomputable def probeWalk (s : PassState α) (w : FreeMonoid α) : ℕ × (ℕ → List Bool) :=
  let fromStart := fun j =>
    (w.toList.scanl (stepPath s.tree s.edges) (midLeaf R s.tree 1)).getD j []
  match anchoredWalk R s.tree s.edges w with
  | none => (0, fromStart)
  | some (start, walk) =>
    if walk.head? = some (fromStart start) then (start, fromStart)
    else (start, fun j => walk.getD (j - start) [])

/-- The population with the probe's anchor, if the tree places any prefix of it. -/
noncomputable def probePool (s : PassState α) (w : FreeMonoid α) : List (FreeMonoid α) :=
  match anchoredWalk R s.tree s.edges w with
  | none => s.pool
  | some (start, _) => if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]

/-- What one probe does: `_process` and `_act_on_disagreement`.  The anchor joins the
population, and the gate's reading of `ε` holds its reads in the band.  A probe whose own sift
cannot be placed, or whose search for the disagreeing edge meets a read it cannot place, goes
unchecked: the read joins the bisection population if the probe disagrees, the gate's reading of
a probe the cut cannot place holding every read it meets in the band when it leaves the walk, and
the boundary population otherwise.  One whose walk ends where it sifts is clean.  Then every edge
is re-voted. -/
noncomputable def probeStep (s : PassState α) (w : FreeMonoid α) : PassState α :=
  let t := s.tree
  let boundary := s.boundary ++ anchorMisses R t w
  let held := s.bisected ++ (t.halfway R.cut R.mid 1).2
  let pool := probePool R s w
  let n := w.toList.length
  let lo := (probeWalk R s w).1
  let walkAt := (probeWalk R s w).2
  let unchecked := fun bd hd =>
    settle K R t pool s.edges (s.streak + 1) (s.unchecked + 1) bd hd
  match t.sift R.cut w with
  | .inr b =>
    if (t.halfway R.cut R.mid w).1 ≠ walkAt n then
      unchecked boundary (held ++ (t.halfway R.cut R.mid w).2)
    else unchecked (boundary ++ [b]) held
  | .inl actual =>
    if actual = walkAt n then
      settle K R t pool s.edges (s.streak + 1) s.unchecked boundary held
    else
    match firstDisagreeingEdge R t w walkAt n lo n with
    | .inl b => unchecked boundary (held ++ [b])
    | .inr fd => onEdge K R s w walkAt pool boundary held fd

/-- How many node reads `x`'s sift makes, the one that fails included. -/
noncomputable def siftReads (t : DTree α) (x : FreeMonoid α) : ℕ := (t.route R.cut x).1.length

/-- The node reads a probe's sifts make, as `Sifter.reads` counts them around `_process`: the
anchor search's, the probe's own, the bisection's, and the witness's and `sprime`'s re-sifts.
The gate's readings of `ε` and of the probe are not the pass's sifts. -/
noncomputable def probeReads (s : PassState α) (w : FreeMonoid α) : ℕ :=
  let t := s.tree
  let n := w.toList.length
  let anchors := ((List.range (n + 1)).map fun i => t.sift R.cut (prefixOf w i)).findIdx (·.isLeft)
  let anchorReads := ((List.range (min (anchors + 1) (n + 1))).map
    fun i => siftReads R t (prefixOf w i)).sum
  let walkAt := (probeWalk R s w).2
  let own := anchorReads + siftReads R t w
  match t.sift R.cut w with
  | .inr _ => own
  | .inl actual =>
    if actual = walkAt n then own else
    let found := bisect R t w walkAt n (probeWalk R s w).1 n
    let bisected := own + (found.1.map fun i => siftReads R t (prefixOf w i)).sum
    match found.2 with
    | .inl _ => bisected
    | .inr fd =>
      match w.toList[fd - 1]? with
      | none => bisected
      | some c =>
        match s.edges (walkAt (fd - 1)) c with
        | none => bisected
        | some (s2, x) =>
          if s2 ≠ walkAt fd then bisected
          else if t.sift R.cut x ≠ .inl (walkAt (fd - 1)) then bisected + siftReads R t x
          else bisected + siftReads R t x + siftReads R t (prefixOf w (fd - 1))

/-- The pass: probes in order until `patience` in a row have neither split a leaf nor weighed
evidence on one, counting the node reads of the probes since the last split. -/
noncomputable def runPass (s : PassState α) (probes : List (FreeMonoid α)) : PassState α :=
  probes.foldl (fun s w =>
    if K.patience ≤ s.streak then s
    else
      let s' := probeStep K R s w
      { s' with reads := if s'.streak = 0 then 0 else s.reads + probeReads R s w }) s

/-- The first state: the root reads at `ε`, the population is the table's prefixes, and the
edges are closed once. -/
noncomputable def initialState (seed : List (FreeMonoid α)) : PassState α :=
  settle K R (.node 1 .leaf .leaf) seed (fun _ _ => none) 0 0 [] []

/-- `_blocked_at_limit`: the probes since the pass's last split went unchecked more than half as
often as a family undecided at `τ` per read could leave them, so the round halves the limits
every later family is held to. -/
def PassState.halves (τ : ℝ) (s : PassState α) : Prop := τ * s.reads < 2 * s.unchecked


/-- The hypothesis a pass state names: its tree and the targets of its edges. -/
def PassState.hyp (s : PassState α) : Hypothesis α :=
  ⟨s.tree, fun p c => (s.edges p c).map Prod.fst⟩

end OrthoDFA
