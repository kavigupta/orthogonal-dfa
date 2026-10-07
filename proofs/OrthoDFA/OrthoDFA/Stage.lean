import OrthoDFA.ClusteringQuality
import OrthoDFA.Automaton

/-!
# The L\* stage

`TransitionResolver`: a discrimination tree over the family's cut.  Every node reads a string at
its midfix through the family's vote and sends it to the side the cut decides; every leaf is a
state, and its members are the population's strings that sift to it.  An edge points where most
members, extended by the letter, sift.  A probe on which the walk and the tree disagree proposes
a distinguisher for a leaf, and `SplitEvidence` weighs it over the leaf's members before the
leaf splits.  The pass harvests every string the cut cannot place on its way, and the shorter
string of every disagreement the evidence says is no split.  `BoundarySource` walks fresh probes
through the finished tree and keeps what it cannot place.

Known modelling gap.  `EdgeResolver` lets a member after the first vote only when its sift is
already memoized; here every member the tree places votes.

Known modelling gap.  `SplitEvidence` splits on a Bayes factor between one pooled Beta-Bernoulli
rate and two; here it splits when the two groups' held-out votes differ in rate by more than a
Hoeffding bound over those votes allows at `splitFpr` over the tests it makes.

Known modelling gap.  `SuffixFamily` holds the family in order and trains on its even places;
here the training half is a given subset of the family.

Known modelling gap.  The hypothesis starts where the tree sifts `ε`, where `to_dfa_and_tree`
classifies `ε` with `oracle_decider` at `decision_boundary`.

Known modelling gap.  Once the rounds stall, `SuffixFamily` reads a string the family leaves
undecided again over up to three families' worth of further suffixes from the same cluster, against
the same thresholds; here every read is over the family alone.

Known modelling gap.  The population holds the round's table prefixes and what the pass adds;
the resolver's own boundary strings are not a population here, only `BoundarySource`'s.

Known modelling gap.  `LeafPopulation` harvests the population's strings the tree cannot place
when it re-sifts them; here they only drop out of the members.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : MeasurableSpace (FreeMonoid α) := ⊤

instance : Stringlike (FreeMonoid α) where
  measurableSet_singleton _ := trivial
  measurable_const_mul _ _ _ := trivial
  measurable_mul_const _ _ _ := trivial
  exists_injective_nat' := Countable.exists_injective_nat (List α)
  decEq := fun a b => (inferInstance : DecidableEq (List α)) a b

/-- The first `i` letters of `w`. -/
def prefixOf (w : FreeMonoid α) (i : ℕ) : FreeMonoid α := FreeMonoid.ofList (w.toList.take i)

open scoped Classical in
/-- How many of `V` accept `x` extended by them, read through `f`. -/
noncomputable def acceptsOn (V : Finset (FreeMonoid α)) (f : FreeMonoid α → ℝ)
    (x : FreeMonoid α) : ℕ :=
  (V.filter fun v => f (x * v) = 1).card

/-- The cut of `F` at `B` on `x`: `some true` above `hi`, `some false` at or below `lo`, `none`
between. -/
noncomputable def cutOn (B : State) (F : Finset (FreeMonoid α)) (f : FreeMonoid α → ℝ)
    (x : FreeMonoid α) : Option Bool :=
  if B.hi < acceptsOn F f x then some true
  else if acceptsOn F f x ≤ B.lo then some false else none

/-- A discrimination tree.  A node's accepting side is `acc`, its rejecting side `rej`. -/
inductive DTree (α : Type*)
  | leaf
  | node (midfix : FreeMonoid α) (rej acc : DTree α)

namespace DTree

/-- The path to the leaf `x` sifts to, `true` for an accepting side, or the string a node could
not place. -/
def sift (cut : FreeMonoid α → Option Bool) : DTree α → FreeMonoid α → List Bool ⊕ FreeMonoid α
  | .leaf, _ => .inl []
  | .node m r a, x =>
    match cut (x * m) with
    | none => .inr (x * m)
    | some true => (a.sift cut x).map (true :: ·) id
    | some false => (r.sift cut x).map (false :: ·) id

/-- The leaves' paths, rejecting sides first. -/
def paths : DTree α → List (List Bool)
  | .leaf => [[]]
  | .node _ r a => r.paths.map (false :: ·) ++ a.paths.map (true :: ·)

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

/-- The round's inputs to the pass: the family's cut, and the reads. -/
structure CutReads (α : Type*) where
  B : State
  F : Finset (FreeMonoid α)
  f : FreeMonoid α → ℝ

/-- The cut, read through the reads. -/
noncomputable def CutReads.cut (R : CutReads α) : FreeMonoid α → Option Bool :=
  cutOn R.B R.F R.f

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

/-- The state a path names: its place among the leaves, `0` past the last of `N + 1`. -/
def stateOfPath (t : DTree α) (N : ℕ) (path : List Bool) : Fin (N + 1) :=
  if h : t.paths.findIdx (· = path) < N + 1 then ⟨_, h⟩ else 0

/-- The hypothesis a finished pass names, over `N + 1` states: the leaves in order, then states
no string reaches.  It starts where `ε` sifts and accepts on the root's accepting side. -/
noncomputable def hypOf (s : PassState α) (N : ℕ) : DFA (FreeMonoid α) (Fin (N + 1)) where
  step q w := w.toList.foldl (fun q c =>
    match s.tree.paths[q.val]? with
    | some p => stateOfPath s.tree N (stepPath s.tree s.edges p c)
    | none => q) q
  step_one _ := rfl
  step_mul q a b := by simp only [FreeMonoid.toList_mul, List.foldl_append]
  start := match s.tree.sift R.cut 1 with
    | .inl p => stateOfPath s.tree N p
    | .inr _ => 0
  accept := {q | ∃ p ∈ s.tree.paths[q.val]?, p.head? = some true}

/-- `BoundarySource`'s search on one probe: the binary search's sifts, in the order it makes
them, keeping the first it cannot place at a prefix at least `longEnough` long. -/
noncomputable def boundarySearch (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ℕ → ℕ → ℕ → Option (FreeMonoid α)
  | 0, _, _ => none
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      match t.sift R.cut (prefixOf w ((lo + hi) / 2)) with
      | .inr b => if K.longEnough ≤ (lo + hi) / 2 then some b else none
      | .inl p =>
        if p = walk ((lo + hi) / 2) then boundarySearch t w walk fuel ((lo + hi) / 2) hi
        else boundarySearch t w walk fuel lo ((lo + hi) / 2)
    else none

/-- `BoundarySource`: walk a probe through the finished tree as a pass would, and keep the first
string it cannot place while sifting a prefix at least `longEnough` long. -/
noncomputable def boundaryOf (s : PassState α) (w : FreeMonoid α) : Option (FreeMonoid α) :=
  let t := s.tree
  let n := w.toList.length
  let keep := fun (i : ℕ) => match t.sift R.cut (prefixOf w i) with
    | .inr b => if K.longEnough ≤ i then some b else none
    | .inl _ => none
  -- The anchor's search sifts every prefix up to the first it places.
  let anchorSifts := (List.range (n + 1)).takeWhile fun i =>
    match t.sift R.cut (prefixOf w i) with | .inr _ => true | .inl _ => false
  match anchorSifts.findSome? keep with
  | some b => some b
  | none =>
    match anchoredWalk R t s.edges w with
    | none => none
    | some (start, walk) =>
      let walkAt := fun j => walk.getD (j - start) []
      match t.sift R.cut w with
      | .inr b => if K.longEnough ≤ n then some b else none
      | .inl actual =>
        if actual = walkAt n then none else boundarySearch K R t w walkAt n start n

/-- What a probe that neither split a leaf nor left the evidence undecided did, read against
the state `s` it was walked in:

* its walk ends at the leaf the tree sifts it to;
* it met a string the cut cannot place, now in the boundary harvest;
* its walk crossed an edge left open because the cut places none of its leaf's members extended
  by the letter, all of which are in the boundary harvest; or
* its walk and sift parted at an edge the evidence says is no split, and the probe up to that
  edge is in the disagreement harvest. -/
def Accounted (s : PassState α) (w : FreeMonoid α) : Prop :=
  let n := w.toList.length
  (∃ start walk, anchoredWalk R s.tree s.edges w = some (start, walk)
      ∧ s.tree.sift R.cut w = .inl (walk.getD (n - start) []))
    ∨ (∃ i ≤ n, ∃ b, s.tree.sift R.cut (prefixOf w i) = .inr b
      ∧ b ∈ (probeStep K R s w).boundary)
    ∨ (∃ start walk i c, anchoredWalk R s.tree s.edges w = some (start, walk) ∧ start ≤ i
      ∧ w.toList[i]? = some c ∧ s.edges (walk.getD (i - start) []) c = none
      ∧ members K R s.tree s.pool (walk.getD (i - start) []) ≠ []
      ∧ ∀ m ∈ members K R s.tree s.pool (walk.getD (i - start) []), ∃ b,
        s.tree.sift R.cut (m * FreeMonoid.of c) = .inr b ∧ b ∈ (probeStep K R s w).boundary)
    ∨ ∃ i < n, prefixOf w i ∈ (probeStep K R s w).disagreements

/-- `ws`, walked in order from `s`, are each quiet and accounted for, and leave the pass at
`s'`. -/
inductive QuietRun : PassState α → List (FreeMonoid α) → PassState α → Prop
  | nil (s : PassState α) : QuietRun s [] s
  | cons (s : PassState α) (w : FreeMonoid α) (ws : List (FreeMonoid α)) (s' : PassState α) :
      (probeStep K R s w).streak ≠ 0 → Accounted K R s w → QuietRun (probeStep K R s w) ws s' →
      QuietRun s (w :: ws) s'

/-- `PassDichotomy`: when a pass started from a population with strings on both sides of the
root runs out of patience, its last `patience` probes ran in a row, each neither splitting a
leaf nor leaving the evidence undecided, and each is accounted for. -/
def PassDichotomy : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (K : StageKnobs α) (R : CutReads α)
    (seed probes : List (FreeMonoid α)),
    0 < K.memberLimit → (∀ b : Bool, ∃ x ∈ seed, R.cut x = some b) →
    K.patience ≤ (runPass K R (initialState K R seed) probes).streak →
    ∃ s ws, ws <:+: probes ∧ ws.length = K.patience
      ∧ QuietRun K R s ws (runPass K R (initialState K R seed) probes)

/-! ## The harvest's replay

`Walked` replays, as #398 has them, anchor a probe no earlier than a point drawn uniformly below
its length.  `Disagreed` replays (#403) keep the probe up to the first edge where walk and sift
part, when every read on the way is decided.

Known modelling gap.  Until #398 and #403 merge, `Walked` on main anchors every replay at the
start, and there is no `Disagreed`.

Known modelling gap.  `Disagreed` walks from the start; here it shares the replay's anchor.

Known modelling gap.  The two harvests are drawn by separate sources; here one replay of a probe
does both, and harvests what either would.

Known modelling gap.  `HarvestSource` counts an attempt only when it turns up a string it has
not served, and retires a source whose yield falls below `(1 - acc_threshold) / 2`.
`ReplayYield` guarantees a refused round's replay only `(1 - acc_threshold) / L`, so a source
between the two can be retired that the argument relies on.
-/

namespace DTree

/-- The strings a sift of `x` asks the cut about, root first: `x` followed by the midfix of each
node on its way.  The cut reads each extended by every suffix of the family. -/
def siftQueries (cut : FreeMonoid α → Option Bool) : DTree α → FreeMonoid α → List (FreeMonoid α)
  | .leaf, _ => []
  | .node m r a, x =>
    x * m :: match cut (x * m) with
      | none => []
      | some true => a.siftQueries cut x
      | some false => r.siftQueries cut x

/-- The path to the leaf `x` reaches when every node decides by `g`. -/
def classify (g : FreeMonoid α → Bool) : DTree α → FreeMonoid α → List Bool
  | .leaf, _ => []
  | .node m r a, x => if g (x * m) then true :: a.classify g x else false :: r.classify g x

end DTree

/-- `anchored_walk` with an earliest anchor `e`: the first prefix of `w` at least `e` long the
tree places, and the paths the walk from it visits. -/
noncomputable def anchoredWalkFrom (t : DTree α)
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) (w : FreeMonoid α) (e : ℕ) :
    Option (ℕ × List (List Bool)) :=
  match ((List.range (w.toList.length + 1)).filter (e ≤ ·)).findSome? fun i =>
      match t.sift R.cut (prefixOf w i) with
      | .inl p => some (i, p)
      | .inr _ => none with
  | none => none
  | some (i, p) => some (i, (w.toList.drop i).scanl (stepPath t edges) p)

/-- The prefix lengths `first_disagreeing_edge` sifts, in order. -/
noncomputable def bisectionSifts (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ℕ → ℕ → ℕ → List ℕ
  | 0, _, _ => []
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      (lo + hi) / 2 :: match t.sift R.cut (prefixOf w ((lo + hi) / 2)) with
        | .inr _ => []
        | .inl p =>
          if p = walk ((lo + hi) / 2) then bisectionSifts t w walk fuel ((lo + hi) / 2) hi
          else bisectionSifts t w walk fuel lo ((lo + hi) / 2)
    else []

/-- The prefix lengths a `Walked` replay of `w` anchored no earlier than `e` sifts: the anchor's
search, the probe's own sift, and the search for the disagreeing edge. -/
noncomputable def replaySifts (s : PassState α) (w : FreeMonoid α) (e : ℕ) : List ℕ :=
  let t := s.tree
  let n := w.toList.length
  let cands := (List.range (n + 1)).filter (e ≤ ·)
  let unplaced := cands.takeWhile fun i => (t.sift R.cut (prefixOf w i)).isRight
  unplaced ++ (cands.drop unplaced.length).take 1 ++
    match anchoredWalkFrom R t s.edges w e with
    | none => []
    | some (start, walk) =>
      let walkAt := fun j => walk.getD (j - start) []
      n :: match t.sift R.cut w with
        | .inr _ => []
        | .inl actual => if actual = walkAt n then [] else bisectionSifts R t w walkAt n start n

/-- The strings whose noise a `Walked` replay of `w` anchored no earlier than `e` reads. -/
def replayReads (s : PassState α) (w : FreeMonoid α) (e : ℕ) : Set (FreeMonoid α) :=
  {x | ∃ i ∈ replaySifts R s w e, ∃ q ∈ s.tree.siftQueries R.cut (prefixOf w i), ∃ v ∈ R.F,
    x = q * v}

/-- What a `Walked` replay of `w` anchored no earlier than `e` harvests: the strings the cut
cannot place on its way. -/
noncomputable def walkedHarvest (s : PassState α) (w : FreeMonoid α) (e : ℕ) :
    List (FreeMonoid α) :=
  let t := s.tree
  let n := w.toList.length
  (((List.range (n + 1)).filter (e ≤ ·)).map fun i => t.sift R.cut (prefixOf w i)).takeWhile
      (·.isRight) |>.filterMap Sum.getRight? |>.append <|
    match anchoredWalkFrom R t s.edges w e with
    | none => []
    | some (start, walk) =>
      let walkAt := fun j => walk.getD (j - start) []
      match t.sift R.cut w with
      | .inr b => [b]
      | .inl actual =>
        if actual = walkAt n then [] else
        match firstDisagreeingEdge R t w walkAt n start n with
        | .inl b => [b]
        | .inr _ => []

/-- What a `Disagreed` replay of `w` anchored no earlier than `e` harvests: with every read
decided, the probe up to the first edge where walk and sift part. -/
noncomputable def disagreedHarvest (s : PassState α) (w : FreeMonoid α) (e : ℕ) :
    List (FreeMonoid α) :=
  let t := s.tree
  let n := w.toList.length
  match anchoredWalkFrom R t s.edges w e with
  | none => []
  | some (start, walk) =>
    let walkAt := fun j => walk.getD (j - start) []
    match t.sift R.cut w with
    | .inr _ => []
    | .inl actual =>
      if actual = walkAt n then [] else
      match firstDisagreeingEdge R t w walkAt n start n with
      | .inl _ => []
      | .inr fd => [prefixOf w (fd - 1)]

/-- What a replay of `w` with earliest anchor `e` harvests, into either harvest. -/
noncomputable def replayHarvest (s : PassState α) (w : FreeMonoid α) (e : ℕ) :
    List (FreeMonoid α) :=
  walkedHarvest R s w e ++ disagreedHarvest R s w e

/-- The earliest anchor's law: uniform on `{0, …, L - 1}`. -/
noncomputable def anchorLaw (L : ℕ) : Measure ℕ :=
  (L : ℝ≥0∞)⁻¹ • ∑ k ∈ Finset.range L, Measure.dirac k

/-- The uniform sampler of length-`L` strings. -/
noncomputable def uniformStrings (α : Type*) [Fintype α] (L : ℕ) : Measure (FreeMonoid α) :=
  ((Fintype.card α : ℝ≥0∞) ^ L)⁻¹ • ∑ f : Fin L → α, Measure.dirac (FreeMonoid.ofList (List.ofFn f))

/-- `ReplaySpread`: a `Walked` replay of a probe drawn from `D` reads any one string `t` with
chance at most

    ∑_{i ≤ |t|} P(e ≤ i) · D(the probe's first i letters are t's),   P(e ≤ i) = min(i + 1, L) / L,

since every read extends a prefix of the probe at least `e` long. -/
def ReplaySpread : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (s : PassState α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ) (t : FreeMonoid α),
    (D.prod (anchorLaw L)).real {q | t ∈ replayReads R s q.1 q.2}
      ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
          ((min (i + 1) L : ℕ) : ℝ) / L * D.real {p | p.toList.take i = t.toList.take i}

/-- `ReplaySpreadUniform`: under the uniform sampler of length-`L` strings over `α`, a `Walked`
replay reads any one string `t` with chance at most

    (1 / L) · ∑_{i ≤ |t|} (i + 1) / |α|^i,

the `κh` the round model asks of the harvest's reads. -/
def ReplaySpreadUniform : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] (R : CutReads α) (s : PassState α)
    (L : ℕ) (t : FreeMonoid α),
    ((uniformStrings α L).prod (anchorLaw L)).real {q | t ∈ replayReads R s q.1 q.2}
      ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
          ((i + 1 : ℕ) : ℝ) / L * ((Fintype.card α : ℝ)⁻¹) ^ i

/-! ## The round's outcome -/

/-- The cut read at the middle of its band, as `estimate_agreement_rate` reads it at
`decision_boundary`: every string is decided. -/
noncomputable def midCut (R : CutReads α) (x : FreeMonoid α) : Bool :=
  decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f x)

/-- Where the hypothesis a finished pass names leaves `x`: walked from the leaf the tree sends
`ε` to, read at the middle of the band, as `to_dfa_and_tree` starts it. -/
noncomputable def gateEnd (s : PassState α) (x : FreeMonoid α) : List Bool :=
  x.toList.foldl (stepPath s.tree s.edges) (s.tree.classify (midCut R) 1)

/-- `estimate_agreement_rate` counts `x` against the hypothesis: its walk ends somewhere other
than where the tree, read at the middle of the band, sends it. -/
def GateDisagrees (s : PassState α) (x : FreeMonoid α) : Prop :=
  gateEnd R s x ≠ s.tree.classify (midCut R) x

/-- `ReplayYield`: whatever the pass left, a replay of a draw from `D` harvests something with
chance at least `1/L` of the share of `D` on which the DFA/DT agreement gate counts the hypothesis
wrong.  So a round either passes the gate or leaves a harvest whose sampler yields at least `d/L`,
`d` the rate it fails by. -/
def ReplayYield : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (s : PassState α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ),
    R.B.lo ≤ R.B.hi →
    D.real {x | GateDisagrees R s x} / L
      ≤ (D.prod (anchorLaw L)).real {q | replayHarvest R s q.1 q.2 ≠ []}

end OrthoDFA
