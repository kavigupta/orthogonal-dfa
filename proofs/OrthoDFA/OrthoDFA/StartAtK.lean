import OrthoDFA.Pass

/-!
# The round, walked from position `k`

A probe is walked from where the cut places its first `k` letters, along learned edges only, and
checked against where the cut places the whole probe.

The counterexample pass runs, and the gate then reads fresh draws against the frozen hypothesis:
- its blocked rate tripping adds the check source and halves the limit;
- its agreement rate decides between passing and refusing, a refusal's decided disagreements
  becoming the next pass's first probes.

A round can both add a source and pass. Each rate is `SequentialRate`'s exact binomial test,
settling early or read out at the batch's end.

`RoundAtK`: given the round's reads, each of these readings is right about the hypothesis, but
for the test's failure chance at each look and Hoeffding's tail at the last. Every probe a refusal
seeds splits a leaf, adds a member, or stops at a string the cut cannot place.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Learned edges: each leaf's edge by a letter, its target and the member that voted for it. -/
abbrev Edges (α : Type*) := List Bool → α → Option (List Bool × FreeMonoid α)

/-- Following only learned edges from `p` along `cs`: the leaves visited, `p` first, or the leaf,
the letter and the step at which an edge is not learned. -/
def follow (edges : Edges α) : List Bool → List α → List (List Bool) ⊕ (List Bool × α × ℕ)
  | p, [] => .inl [p]
  | p, c :: cs =>
    match edges p c with
    | some (q, _) =>
      match follow edges q cs with
      | .inl ps => .inl (p :: ps)
      | .inr (s, c', i) => .inr (s, c', i + 1)
    | none => .inr (p, c, 0)

/-- How a walk from position `k` ends: the cut cannot place the first `k` letters, an edge out of
leaf `s` by the letter at position `j` is not learned, or the walk reaches the end, the leaves it
visits listed from position `k`. -/
inductive KWalk (α : Type*)
  | anchor
  | edge (s : List Bool) (c : α) (j : ℕ)
  | reached (ps : List (List Bool))

/-- What one probe of the counterexample check finds: agreement, a block with what the boundary
source outputs for it, or a decided disagreement with the walk's leaves. -/
inductive KCheck (α : Type*)
  | agree
  | blocked (out : Option (FreeMonoid α))
  | disagree (ps : List (List Bool))

/-- What the pass does with a disagreeing probe: split a leaf, add `sprime` as a member, stop at
a string the cut cannot place, or drop it. -/
inductive SeedResult (α : Type*)
  | split (d : FreeMonoid α) (s1 : List Bool) (y sprime : FreeMonoid α)
  | member (s1 : List Bool) (sprime : FreeMonoid α)
  | stopped (b : FreeMonoid α)
  | dropped

namespace DTree

/-- `first_disagreement`, saying why it finds none: the midfix where `x·pre` and `y·pre` part,
the first string on the way the cut cannot place, or `none` where they reach a leaf together. -/
def parting (cut : FreeMonoid α → Option Bool) (x y pre : FreeMonoid α) :
    DTree α → Option (FreeMonoid α ⊕ FreeMonoid α)
  | .leaf => none
  | .node m r a =>
    match cut (x * (pre * m)), cut (y * (pre * m)) with
    | some true, some true => a.parting cut x y pre
    | some false, some false => r.parting cut x y pre
    | some _, some _ => some (.inl (pre * m))
    | none, _ => some (.inr (x * (pre * m)))
    | _, none => some (.inr (y * (pre * m)))

end DTree

/-- `first_disagreeing_edge` between an index where `place` agrees with `walk` and one where it
does not. -/
def bisectAt (place walk : ℕ → List Bool) : ℕ → ℕ → ℕ → ℕ
  | 0, _, hi => hi
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      if place ((lo + hi) / 2) = walk ((lo + hi) / 2) then
        bisectAt place walk fuel ((lo + hi) / 2) hi
      else bisectAt place walk fuel lo ((lo + hi) / 2)
    else hi

variable (K : StageKnobs α) (R : CutReads α)

/-- The walk of `x` from position `k`, along `edges`. -/
noncomputable def kWalk (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : KWalk α :=
  match t.sift R.cut (prefixOf x k) with
  | .inr _ => .anchor
  | .inl p =>
    match follow edges p (x.toList.drop k) with
    | .inl ps => .reached ps
    | .inr (s, c, i) => .edge s c (k + i)

/-- What the walk source outputs for `x`: the first `k` letters where the cut cannot place them;
at an unlearned edge out of `s` at position `j`, the prefix through it where the cut cannot place
that, else the prefix before it where the cut places it at `s` or cannot place it; else nothing. -/
noncomputable def walkOutput (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Option (FreeMonoid α) :=
  match kWalk R t edges k x with
  | .anchor => some (prefixOf x k)
  | .edge s _ j =>
    match t.sift R.cut (prefixOf x (j + 1)) with
    | .inr _ => some (prefixOf x (j + 1))
    | .inl _ =>
      match t.sift R.cut (prefixOf x j) with
      | .inl p => if p = s then some (prefixOf x j) else none
      | .inr _ => some (prefixOf x j)
  | .reached _ => none

/-- One probe of the counterexample check: a walk that does not reach the end is blocked, with
`walkOutput`'s string; so is one whose whole probe the cut cannot place, with the string the cut
cannot place. -/
noncomputable def kCheck (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : KCheck α :=
  match kWalk R t edges k x with
  | .reached ps =>
    match t.sift R.cut x with
    | .inr b => .blocked (some b)
    | .inl a => if some a = ps.getLast? then .agree else .disagree ps
  | _ => .blocked (walkOutput R t edges k x)

/-- Where the bisection places `x`'s first `i` letters: where the cut does, else where the gate's
reading does. -/
noncomputable def place (t : DTree α) (x : FreeMonoid α) (i : ℕ) : List Bool :=
  (t.sift R.cut (prefixOf x i)).elim id fun _ => (t.halfway R.cut R.mid (prefixOf x i)).1

/-- `_act_on_disagreement` on a probe whose walk from `k` visits `ps` and whose sift disagrees:
bisect to the edge where they part, placing what the cut cannot at the middle of the band (#411),
then the guards and the split test, a no-split answered as undecided is (#412). -/
noncomputable def seedStep (t : DTree α) (pool : List (FreeMonoid α)) (edges : Edges α) (k : ℕ)
    (x : FreeMonoid α) (ps : List (List Bool)) : SeedResult α :=
  let walkAt := fun j => ps.getD (j - k) []
  let n := x.toList.length
  let fd := bisectAt (place R t x) walkAt (n - k) k n
  match x.toList[fd - 1]? with
  | none => .dropped
  | some c =>
    let s1 := walkAt (fd - 1)
    let sprime := prefixOf x (fd - 1)
    match edges s1 c with
    | none => .dropped
    | some (s2, y) =>
      if s2 ≠ walkAt fd then .dropped
      else
      match t.sift R.cut sprime with
      | .inr b => .stopped b
      | .inl p =>
        if p ≠ s1 ∨ t.sift R.cut y ≠ .inl s1 then .dropped
        else
        match t.parting R.cut y sprime (FreeMonoid.of c) with
        | none => .dropped
        | some (.inr b) => .stopped b
        | some (.inl d) =>
          match verdict K R t pool s1 d (t.paths.length * Fintype.card α) with
          | .split => .split d s1 y sprime
          | _ => .member s1 sprime

/-- What the pass carries: the tree, the population, the learned edges, and the probes since the
last split or evidence weighed. -/
structure KState (α : Type*) where
  tree : DTree α
  pool : List (FreeMonoid α)
  edges : Edges α
  streak : ℕ

/-- Every edge re-voted, as after every probe. -/
noncomputable def closeK (t : DTree α) (pool : List (FreeMonoid α)) (edges : Edges α)
    (streak : ℕ) : KState α :=
  ⟨t, pool, closeEdges K R t pool edges, streak⟩

/-- One probe of the counterexample pass. -/
noncomputable def probeStepK (k : ℕ) (s : KState α) (x : FreeMonoid α) : KState α :=
  let quiet := closeK K R s.tree s.pool s.edges (s.streak + 1)
  match kCheck R s.tree s.edges k x with
  | .disagree ps =>
    match seedStep K R s.tree s.pool s.edges k x ps with
    | .split d s1 y sprime =>
      let cleared : Edges α := fun p c' =>
        match s.edges p c' with
        | some (q, w) => if p = s1 ∨ q = s1 then none else some (q, w)
        | none => none
      closeK K R (s.tree.splitAt d s1) (s.pool ++ ([y, sprime].filter (· ∉ s.pool))) cleared 0
    | .member _ sprime => closeK K R s.tree (sprime :: s.pool.filter (· ≠ sprime)) s.edges 0
    | _ => quiet
  | _ => quiet

/-- The pass: probes in order until `patience` in a row are quiet. -/
noncomputable def runPassK (k : ℕ) (s : KState α) (probes : List (FreeMonoid α)) : KState α :=
  probes.foldl (fun s x => if K.patience ≤ s.streak then s else probeStepK K R k s x) s

/-- The first state: the root reads at `ε`, the population is the table's prefixes, and the
edges are voted once. -/
noncomputable def initialK (seed : List (FreeMonoid α)) : KState α :=
  closeK K R (.node 1 .leaf .leaf) seed (fun _ _ => none) 0

/-- The probe is blocked. -/
def KCheck.isBlocked : KCheck α → Prop
  | .blocked _ => True
  | _ => False

/-- The walk reached an unlearned edge through a wrong earlier one: the prefix through the edge
is placed, and the prefix before it is placed at some other leaf. -/
def wrongEarlier (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  ∃ s c j p, kWalk R t edges k x = .edge s c j ∧ (t.sift R.cut (prefixOf x (j + 1))).isLeft
    ∧ t.sift R.cut (prefixOf x j) = .inl p ∧ p ≠ s

open scoped Classical in
/-- How many of a batch's first `n` draws satisfy `P`. -/
noncomputable def hitsIn {N : ℕ} (b : Fin N → FreeMonoid α) (P : FreeMonoid α → Prop) (n : ℕ) :
    ℕ :=
  (Finset.univ.filter fun i : Fin N => (i : ℕ) < n ∧ P (b i)).card

/-- `binomial_side_of_boundary` once `n₀` draws are in: above `θ` when `Bin(n, θ)` reaches `h`
hits with chance below `a`, below when it stays at or under `h` with chance below `a`. -/
noncomputable def rateSide (θ a : ℝ) (n₀ n h : ℕ) : Option Bool :=
  if n₀ ≤ n then
    if binomSfGe n θ h < a then some true
    else if 1 - binomSfGe n θ (h + 1) < a then some false
    else none
  else none

/-- `SequentialRate.above` over a batch of `N`: the side the test first settles on, read one draw
at a time, or, where it never settles, whether the batch's rate exceeds `θ`. -/
noncomputable def seqAbove (θ a : ℝ) (n₀ : ℕ) {N : ℕ} (b : Fin N → FreeMonoid α)
    (P : FreeMonoid α → Prop) : Prop :=
  match (List.range' 1 N).findSome? fun n => rateSide θ a n₀ n (hitsIn b P n) with
  | some s => s = true
  | none => θ * N < hitsIn b P N

/-- The gate's reading of a draw disagrees with the walk: from where the gate places its first `k`
letters, the walk along learned edges meets an unlearned edge, or ends somewhere other than where
the gate places the whole draw. -/
def gateDisagrees (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  match follow edges (place R t x k) (x.toList.drop k) with
  | .inl ps => ps.getLast? ≠ some (place R t x x.toList.length)
  | .inr _ => True

/-- What the check source outputs for a draw: what a blocked probe of the check leaves. -/
noncomputable def checkOutput (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Option (FreeMonoid α) :=
  match kCheck R t edges k x with
  | .blocked out => out
  | _ => none

/-- A draw a refusal can carry: its check disagrees, decided, or its walk reached an unlearned
edge through a wrong earlier one, so that its prefix before that edge disagrees, decided. -/
def Carried (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Prop :=
  (∃ ps, kCheck R t edges k x = .disagree ps) ∨ wrongEarlier R t edges k x

open scoped Classical in
/-- What the gate's readings of batch `bg` against the frozen hypothesis `s` claim:
* a tripped blocked check: the check is blocked on at least `θc − δ` of draws;
* a passing gate: the gate's reading disagrees on at most `1 − acc + δ` of draws;
* a refusing gate: on at least `1 − acc − δ` of draws; every probe of the batch whose check
  disagrees, run again, splits a leaf, adds a member, or stops at a string the cut cannot place;
  and where none of the batch's first `n₀` draws can be carried, the check source, which the
  round then adds as it halves the limit, outputs a string on at least the gate's disagreement
  less `δ`. -/
def RoundAtKHolds (s : KState α) (D : Measure (FreeMonoid α)) (k : ℕ) (θc acc a δ : ℝ) (n₀ : ℕ)
    {ng : ℕ} (bg : Fin ng → FreeMonoid α) : Prop :=
  (seqAbove θc a n₀ bg (fun x => (kCheck R s.tree s.edges k x).isBlocked) →
      θc - δ ≤ D.real {x | (kCheck R s.tree s.edges k x).isBlocked})
    ∧ (seqAbove acc a n₀ bg (fun x => ¬ gateDisagrees R s.tree s.edges k x) →
      D.real {x | gateDisagrees R s.tree s.edges k x} ≤ 1 - acc + δ)
    ∧ (¬ seqAbove acc a n₀ bg (fun x => ¬ gateDisagrees R s.tree s.edges k x) →
      1 - acc - δ ≤ D.real {x | gateDisagrees R s.tree s.edges k x}
        ∧ (∀ i ps, kCheck R s.tree s.edges k (bg i) = .disagree ps →
          seedStep K R s.tree s.pool s.edges k (bg i) ps ≠ .dropped)
        ∧ ((∀ i : Fin ng, (i : ℕ) < n₀ → ¬ Carried R s.tree s.edges k (bg i)) →
          D.real {x | gateDisagrees R s.tree s.edges k x} - δ
            ≤ D.real {x | (checkOutput R s.tree s.edges k x).isSome}))

/-- `RoundAtK`: for any reads of the round's family and whatever probes the pass draws, the gate's
readings of the hypothesis the pass ends with are right but for twice `ng·a + exp(−2·ng·δ²)` and
`exp(−min(n₀, ng)·δ)`. -/
def RoundAtK : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (K : StageKnobs α) (R : CutReads α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k ng n₀ : ℕ)
    (seed probes : List (FreeMonoid α)) (θc acc a δ : ℝ),
    θc ≤ 1 → 0 ≤ acc → acc ≤ 1 → 0 ≤ a → 0 ≤ δ →
    let s := runPassK K R k (initialK K R seed) probes
    (Measure.pi fun _ : Fin ng => D).real {bg | ¬ RoundAtKHolds K R s D k θc acc a δ n₀ bg}
      ≤ 2 * (ng * a + Real.exp (-2 * ng * δ ^ 2)) + Real.exp (-(min n₀ ng : ℕ) * δ)

/-- `CheckYield`: the check source outputs a string on exactly the blocked draws that did not
reach an unlearned edge through a wrong earlier one. -/
def CheckYield : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (D : Measure (FreeMonoid α))
    [IsFiniteMeasure D] (t : DTree α) (edges : Edges α) (k : ℕ),
    D.real {x | (checkOutput R t edges k x).isSome}
      = D.real {x | (kCheck R t edges k x).isBlocked} - D.real {x | wrongEarlier R t edges k x}

/-- `SourceSpread`: every draw the check source outputs `u` on begins with `u`'s first `k`
letters, so no string takes more of the source than the draws with one prefix of length `k`. -/
def SourceSpread : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (D : Measure (FreeMonoid α))
    [IsFiniteMeasure D] (t : DTree α) (edges : Edges α) (k L : ℕ) (u : FreeMonoid α),
    k ≤ L → (∀ᵐ x ∂D, x.toList.length = L) →
    D.real {x | checkOutput R t edges k x = some u} ≤ D.real {x | u.toList.take k <+: x.toList}

end OrthoDFA
