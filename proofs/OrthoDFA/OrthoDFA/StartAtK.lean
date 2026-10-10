import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead

/-!
# Walks and searches from position `k`

A probe is walked from where the cut places its first `k` letters along learned edges (`follow`,
`KWalk`); a disagreement is searched by bisection (`bracketAt`) to what the probe comes to
(`Outcome`). `rateSide` is a test's verdict on a count against a rate.
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

/-- What one probe comes to: the walk ends where the cut places the probe; the cut cannot place
the first `k` letters, `w`; it cannot place the probe, or, at an unlearned edge at `j`, its first
`j + 1` or `j` letters, `w`; the search of a decided disagreement ends at undecided reads at `j`
and `j + 1`, at the edge into `fd` of the walk `ps`, or at an undecided read at `j` between an
agreeing and a disagreeing one; or an unlearned edge's source is where the cut places `u`. -/
inductive Outcome (α : Type*)
  | agree
  | startUndecided (w : FreeMonoid α)
  | endUndecided (w : FreeMonoid α)
  | pair (j : ℕ)
  | edge (ps : List (List Bool)) (fd : ℕ)
  | triple (j : ℕ)
  | member (u : FreeMonoid α)

namespace DTree

open scoped Classical in
noncomputable def midfixes : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (r.midfixes ∪ a.midfixes)

end DTree

/-- `bracket` from `lo`, agreeing, to `hi`, disagreeing, over the walk `ps`, `agrees i` being
`none` where the read of the first `i` letters is undecided: a decided middle narrows, an
undecided one reads its neighbours. -/
def bracketAt (agrees : ℕ → Option Bool) (ps : List (List Bool)) : ℕ → ℕ → ℕ → Outcome α
  | 0, _, hi => .edge ps hi
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      let ag := fun p => if p = lo then some true else if p = hi then some false else agrees p
      let mid := (lo + hi) / 2
      match ag mid with
      | some true => bracketAt agrees ps fuel mid hi
      | some false => bracketAt agrees ps fuel lo mid
      | none =>
        match ag (mid - 1), ag (mid + 1) with
        | none, _ => .pair (mid - 1)
        | some _, none => .pair mid
        | some true, some false => .triple mid
        | some true, some true => bracketAt agrees ps fuel (mid + 1) hi
        | some false, some _ => bracketAt agrees ps fuel lo (mid - 1)
    else .edge ps hi

variable (R : CutReads α)

/-- `binomial_side_of_boundary` once `n₀` draws are in: above `θ` when `Bin(n, θ)` reaches `h`
hits with chance below `a`, below when it stays at or under `h` with chance below `a`. -/
noncomputable def rateSide (θ a : ℝ) (n₀ n h : ℕ) : Option Bool :=
  if n₀ ≤ n then
    if binomSfGe n θ h < a then some true
    else if 1 - binomSfGe n θ (h + 1) < a then some false
    else none
  else none

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

end OrthoDFA
