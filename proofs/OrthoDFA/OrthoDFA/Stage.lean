import OrthoDFA.Clustering

/-!
# The round's outcome

A round of the L\* stage ends with a hypothesis: `TransitionResolver`'s discrimination tree over
the family's cut, and the edges between its leaves.  The DFA/DT agreement gate
(`estimate_agreement_rate`) reads the tree at the middle of the cut's band and walks the
hypothesis from where that reading sends `ε`.  The round's harvest is replayed as `Walked` does:
from the first prefix at least as long as an earliest anchor drawn uniformly below `L` that the
tree places, walk the hypothesis, sift the draw, and bisect to the first edge where walk and sift
part, keeping every string the cut cannot place and, with every read decided, the draw up to that
edge.

`RoundOutcome`: for any tree and DFA a round ends with, if the DFA/DT agreement check fails on a
share `d` of sampler strings, the harvest sampler finds at least one string on at least a `d/L`
share of attempts, harvests no single string `t` on more than a `B(t)` share of them, and reads no
single string `t` on more than a `B(t)` share of them, where

    B(t) = ∑_{i ≤ |t|} min(i + 1, L) / L · D(the draw's first i letters are t's).

Known modelling gap.  This is `Walked` as #398 (the uniform anchor) and #403 (one harvest, which
keeps the disagreement's prefix) have it; on main it walks from the start and keeps no prefix.

Known modelling gap.  `HarvestSource` counts an attempt only when it turns up a string it has not
served, and retires a source whose yield falls below `(1 - acc_threshold) / 2`, while
`RoundOutcome` guarantees a refused round's replay only `(1 - acc_threshold) / L`, so a source
between the two can be retired that the argument relies on.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : MeasurableSpace (FreeMonoid α) := ⊤

/-- The first `i` letters of `w`. -/
def prefixOf (w : FreeMonoid α) (i : ℕ) : FreeMonoid α := FreeMonoid.ofList (w.toList.take i)

open scoped Classical in
/-- How many of `V` accept `x` extended by them, read through `f`. -/
noncomputable def acceptsOn (V : Finset (FreeMonoid α)) (f : FreeMonoid α → ℝ)
    (x : FreeMonoid α) : ℕ :=
  (V.filter fun v => f (x * v) = 1).card

/-- The family `F`, read through `f` and cut at `B`: `x` accepts above `B.hi`, rejects at or
below `B.lo`, and is undecided between. -/
structure CutReads (α : Type*) where
  B : State
  F : Finset (FreeMonoid α)
  f : FreeMonoid α → ℝ

noncomputable def CutReads.cut (R : CutReads α) (x : FreeMonoid α) : Option Bool :=
  if R.B.hi < acceptsOn R.F R.f x then some true
  else if acceptsOn R.F R.f x ≤ R.B.lo then some false else none

/-- A discrimination tree.  A node's accepting side is `acc`, its rejecting side `rej`. -/
inductive DTree (α : Type*)
  | leaf
  | node (midfix : FreeMonoid α) (rej acc : DTree α)

namespace DTree

/-- `x`'s way down the tree by `cut`: the strings it asks the cut about, root first (`x` followed
by each node's midfix), and the path to its leaf (`true` for an accepting side) or the string a
node could not place. -/
def route (cut : FreeMonoid α → Option Bool) :
    DTree α → FreeMonoid α → List (FreeMonoid α) × (List Bool ⊕ FreeMonoid α)
  | .leaf, _ => ([], .inl [])
  | .node m r a, x =>
    match cut (x * m) with
    | none => ([x * m], .inr (x * m))
    | some true => (x * m :: (a.route cut x).1, (a.route cut x).2.map (true :: ·) id)
    | some false => (x * m :: (r.route cut x).1, (r.route cut x).2.map (false :: ·) id)

def sift (cut : FreeMonoid α → Option Bool) (t : DTree α) (x : FreeMonoid α) :
    List Bool ⊕ FreeMonoid α :=
  (t.route cut x).2

/-- The leaves' paths. -/
def paths : DTree α → List (List Bool)
  | .leaf => [[]]
  | .node _ r a => r.paths.map (false :: ·) ++ a.paths.map (true :: ·)

end DTree

/-- The hypothesis: the tree, and each leaf's edges by letter, open where `none`. -/
structure Hypothesis (α : Type*) where
  tree : DTree α
  edges : List Bool → α → Option (List Bool)

/-- The totalised step: the edge's target when it is a leaf, else stay put. -/
def Hypothesis.step (H : Hypothesis α) (path : List Bool) (c : α) : List Bool :=
  match H.edges path c with
  | some p => if p ∈ H.tree.paths then p else path
  | none => path

variable (R : CutReads α)

/-- `anchored_walk`'s search over the prefix lengths `is`: the ones whose prefix of `w` the tree
cannot place, up to the first it places, and that one with its leaf. -/
noncomputable def anchorSearch (t : DTree α) (w : FreeMonoid α) :
    List ℕ → List ℕ × Option (ℕ × List Bool)
  | [] => ([], none)
  | i :: is =>
    match t.sift R.cut (prefixOf w i) with
    | .inl p => ([], some (i, p))
    | .inr _ => let rest := anchorSearch t w is; (i :: rest.1, rest.2)

/-- `first_disagreeing_edge` between an index where the sift agrees with the walk and one where
it does not: the prefix lengths it sifts, and the first index where they do not agree, or the
string a sift on the way could not place. -/
noncomputable def bisect (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ℕ → ℕ → ℕ → List ℕ × (FreeMonoid α ⊕ ℕ)
  | 0, _, hi => ([], .inr hi)
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      match t.sift R.cut (prefixOf w ((lo + hi) / 2)) with
      | .inr b => ([(lo + hi) / 2], .inl b)
      | .inl p =>
        let rest := if p = walk ((lo + hi) / 2) then bisect t w walk fuel ((lo + hi) / 2) hi
          else bisect t w walk fuel lo ((lo + hi) / 2)
        ((lo + hi) / 2 :: rest.1, rest.2)
    else ([], .inr hi)

/-- `Walked`'s replay of `w` from the earliest anchor `e`: the prefix lengths it sifts, and what
it harvests. -/
noncomputable def replay (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ) :
    List ℕ × List (FreeMonoid α) :=
  let t := H.tree
  let n := w.toList.length
  let search := anchorSearch R t w ((List.range (n + 1)).filter (e ≤ ·))
  let missed := search.1.filterMap fun i => (t.sift R.cut (prefixOf w i)).getRight?
  match search.2 with
  | none => (search.1, missed)
  | some (start, p) =>
    let walk := (w.toList.drop start).scanl H.step p
    let walkAt := fun j => walk.getD (j - start) []
    match t.sift R.cut w with
    | .inr b => (search.1 ++ [start, n], missed ++ [b])
    | .inl actual =>
      if actual = walkAt n then (search.1 ++ [start, n], missed) else
      let found := bisect R t w walkAt n start n
      (search.1 ++ [start, n] ++ found.1, missed ++ match found.2 with
        | .inl b => [b]
        | .inr fd => [prefixOf w (fd - 1)])

/-- The strings whose noise the replay reads: each sifted prefix's queries, extended by every
suffix of the family. -/
def replayReads (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ) : Set (FreeMonoid α) :=
  {x | ∃ i ∈ (replay R H w e).1, ∃ q ∈ (H.tree.route R.cut (prefixOf w i)).1, ∃ v ∈ R.F,
    x = q * v}

/-- The earliest anchor's law: uniform on `{0, …, L - 1}`. -/
noncomputable def anchorLaw (L : ℕ) : Measure ℕ :=
  (L : ℝ≥0∞)⁻¹ • ∑ k ∈ Finset.range L, Measure.dirac k

/-- The tree read at the middle of the band, as `estimate_agreement_rate` reads it at
`decision_boundary`: every string is decided. -/
noncomputable def midPath (H : Hypothesis α) (x : FreeMonoid α) : List Bool :=
  (H.tree.sift (fun y => some (decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f y))) x).elim id
    fun _ => []

/-- `estimate_agreement_rate` counts `x` against the hypothesis: walked from where the tree read
at the middle of the band sends `ε`, it ends somewhere other than where that reading sends `x`. -/
def DFAandDTDisagree (H : Hypothesis α) (x : FreeMonoid α) : Prop :=
  x.toList.foldl H.step (midPath R H 1) ≠ midPath R H x

/-- `RoundOutcome`: for any tree and DFA a round ends with, if the DFA/DT agreement check fails
on a share `d` of `D`, the harvest's replay, from an earliest anchor drawn uniformly below `L`,

* finds at least one string on at least a `d/L` share of attempts;
* harvests no single string `t` on more than a `B(t)` share of them; and
* reads no single string `t` on more than a `B(t)` share of them,

where `B(t) = ∑_{i ≤ |t|} min(i + 1, L) / L · D(the draw's first i letters are t's)`. -/
def RoundOutcome : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ),
    R.B.lo ≤ R.B.hi →
    D.real {x | DFAandDTDisagree R H x} / L
        ≤ (D.prod (anchorLaw L)).real {q | (replay R H q.1 q.2).2 ≠ []}
      ∧ (∀ t : FreeMonoid α, (D.prod (anchorLaw L)).real {q | t ∈ (replay R H q.1 q.2).2}
          ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
              ((min (i + 1) L : ℕ) : ℝ) / L * D.real {p | p.toList.take i = t.toList.take i})
      ∧ ∀ t : FreeMonoid α, (D.prod (anchorLaw L)).real {q | t ∈ replayReads R H q.1 q.2}
          ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
              ((min (i + 1) L : ℕ) : ℝ) / L * D.real {p | p.toList.take i = t.toList.take i}

end OrthoDFA
