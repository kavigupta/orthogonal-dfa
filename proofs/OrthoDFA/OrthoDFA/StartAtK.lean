import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import OrthoDFA.FamilyRead
import OrthoDFA.RateTest
import OrthoDFA.TripartiteSearch

/-!
# Walks and searches from position `k`

A probe is walked from where the cut places its first `k` letters along learned edges (`follow`,
`KWalk`); a disagreement is searched by main's bisection without recovery
(`bisectBoundary false`) to what the probe comes to (`Outcome`). `bracketAt` is the same search
by fuel, which the proofs induct on.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `Edges`, curried: each leaf's edge by a letter, its target and the member that voted for it. -/
abbrev EdgeTable (α : Type*) := List Bool → α → Option (List Bool × FreeMonoid α)

/-- Following only learned edges from `p` along `cs`: the leaves visited, `p` first, or the leaf,
the letter and the step at which an edge is not learned. -/
def follow (edges : EdgeTable α) : List Bool → List α → List (List Bool) ⊕ (List Bool × α × ℕ)
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
`j + 1` or `j` letters, `w`; the search of a decided disagreement stops at its first undecided
midpoint `j`, or ends at the edge into `fd` of the walk `ps`; or an unlearned edge's source is
where the cut places `u`. -/
inductive Outcome (α : Type*)
  | agree
  | startUndecided (w : FreeMonoid α)
  | endUndecided (w : FreeMonoid α)
  | undecidedMid (j : ℕ)
  | edge (ps : List (List Bool)) (fd : ℕ)
  | member (u : FreeMonoid α)

namespace DTree

open scoped Classical in
noncomputable def midfixes : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (r.midfixes ∪ a.midfixes)

end DTree

/-- Main's search result as level 4's outcome, over the walk `ps`. -/
def searchOutcome (ps : List (List Bool)) : SearchResult → Outcome α
  | .undecidedMid j => .undecidedMid j
  | .edge j => .edge ps j
  | _ => .agree

/-- `bisectBoundary false` by fuel, over the walk `ps`. -/
def bracketAt (agrees : ℕ → Option Bool) (ps : List (List Bool)) : ℕ → ℕ → ℕ → Outcome α
  | 0, _, hi => .edge ps hi
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      match agrees ((lo + hi) / 2) with
      | some true => bracketAt agrees ps fuel ((lo + hi) / 2) hi
      | some false => bracketAt agrees ps fuel lo ((lo + hi) / 2)
      | none => .undecidedMid ((lo + hi) / 2)
    else .edge ps hi

variable (R : CutReads α)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}

end OrthoDFA
