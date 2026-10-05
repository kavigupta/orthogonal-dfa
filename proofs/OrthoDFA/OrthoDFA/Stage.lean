import OrthoDFA.ClusteringQuality
import OrthoDFA.Automaton

/-!
# The L\* stage

`TransitionResolver`: a discrimination tree over the family's cut.  Every node reads a string at
its midfix through the family's vote and sends it to the side the cut decides; every leaf is a
state, and its members are the population's strings that sift to it.  An edge points where most
members, extended by the letter, sift.  A probe on which the walk and the tree disagree proposes
a distinguisher for a leaf, and `SplitEvidence` weighs it over the leaf's members before the
leaf splits.  `BoundarySource` walks fresh probes through the finished tree and keeps what it
cannot place.

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
arrived, each edge's target and witness, the probes since the last split, and how many of those
a read the cut could not place kept from being checked. -/
structure PassState (α : Type*) where
  tree : DTree α
  pool : List (FreeMonoid α)
  edges : List Bool → α → Option (List Bool × FreeMonoid α)
  streak : ℕ
  unchecked : ℕ

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
it does not, the first where it does not; `none` if a sift on the way cannot be placed.  `walk j`
is the walk's path after `j` letters. -/
noncomputable def firstDisagreeingEdge (t : DTree α) (w : FreeMonoid α)
    (walk : ℕ → List Bool) : ℕ → ℕ → ℕ → Option ℕ
  | 0, _, hi => some hi
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      match t.sift R.cut (prefixOf w ((lo + hi) / 2)) with
      | .inr _ => none
      | .inl p =>
        if p = walk ((lo + hi) / 2) then firstDisagreeingEdge t w walk fuel ((lo + hi) / 2) hi
        else firstDisagreeingEdge t w walk fuel lo ((lo + hi) / 2)
    else some hi

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

/-- Re-vote every edge and carry the streak. -/
noncomputable def settle (t : DTree α) (pool : List (FreeMonoid α))
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) (streak unchecked : ℕ) :
    PassState α :=
  { tree := t, pool := pool, edges := closeEdges K R t pool edges, streak := streak,
    unchecked := unchecked }

/-- What one probe does: `_process` and `_act_on_disagreement`.  The anchor joins the
population; a disagreement the evidence confirms splits its leaf, whose edges are cleared, and
the two strings that exhibited it join the population.  Otherwise the probe's own string at the
leaf joins it ahead of the rest, so it is a member however many the leaf holds.  A probe with no
prefix the tree places, whose own sift cannot be placed, or whose search for the disagreeing
edge meets a read it cannot place, goes unchecked.  Then every edge is re-voted. -/
noncomputable def probeStep (s : PassState α) (w : FreeMonoid α) : PassState α :=
  let t := s.tree
  match anchoredWalk R t s.edges w with
  | none => settle K R t s.pool s.edges (s.streak + 1) (s.unchecked + 1)
  | some (start, walk) =>
    let pool := if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]
    let n := w.toList.length
    let walkAt := fun j => walk.getD (j - start) []
    let clean := settle K R t pool s.edges (s.streak + 1) s.unchecked
    let unchecked := settle K R t pool s.edges (s.streak + 1) (s.unchecked + 1)
    match t.sift R.cut w with
    | .inr _ => unchecked
    | .inl actual =>
      if actual = walkAt n then clean else
      match firstDisagreeingEdge R t w walkAt n start n with
      | none => unchecked
      | some fd =>
        match w.toList[fd - 1]? with
        | none => clean
        | some c =>
          let s1 := walkAt (fd - 1)
          let sprime := prefixOf w (fd - 1)
          match s.edges s1 c with
          | none => clean
          | some (s2, x) =>
            if s2 ≠ walkAt fd ∨ t.sift R.cut x ≠ .inl s1 ∨ t.sift R.cut sprime ≠ .inl s1 then
              clean
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
              | .noSplit => settle K R t kept s.edges (s.streak + 1) s.unchecked
              | .undecided => settle K R t kept s.edges 0 0

/-- The pass: probes in order until `patience` in a row have neither split a leaf nor left the
evidence undecided. -/
noncomputable def runPass (s : PassState α) (probes : List (FreeMonoid α)) : PassState α :=
  probes.foldl (fun s w => if K.patience ≤ s.streak then s else probeStep K R s w) s

/-- The first state: the root reads at `ε`, the population is the table's prefixes, and the
edges are closed once. -/
noncomputable def initialState (seed : List (FreeMonoid α)) : PassState α :=
  settle K R (.node 1 .leaf .leaf) seed (fun _ _ => none) 0 0

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

end OrthoDFA
