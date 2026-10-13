import OrthoDFA.Hypothesis
import OrthoDFA.TripartiteSearch

/-! Level 2's probe: a string walked along the edges and sifted, and where the two part. -/

namespace OrthoDFA

namespace Ideal

variable {α : Type*}

/-- What undecided reads are charged to: the start, the stop, or an edge. -/
inductive UndecidedSite (α : Type*)
  | start
  | stop
  | edge (s : List Bool) (c : α)
  deriving DecidableEq

/-- How a probe ends. `wrongEdge s c u t`: `u` sifts to `s` and `u·c` to `t`, which the edge
`(s, c)` does not point at; `unlearnedEdge` is the same at an unlearned edge. `undecided k zs`:
undecided strings charged to `k`, by a probe that disagreed unless `clean`. -/
inductive Outcome (α : Type*)
  | agree
  | wrongEdge (s : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool)
  | unlearnedEdge (s : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool)
  | undecided (k : UndecidedSite α) (zs : List (FreeMonoid α)) (clean : Bool)

def Outcome.undecidedStrings : Outcome α → List (FreeMonoid α)
  | .undecided _ zs _ => zs
  | _ => []

section Probe

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

def siftAgrees (p : ℕ) : Option Bool :=
  (T.sift read (prefixOf x p)).elim (fun l => some (decide (l = st p))) fun _ => none

def undecidedAt (p : ℕ) : List (FreeMonoid α) :=
  (T.sift read (prefixOf x p)).elim (fun _ => []) fun z => [z]

def siteAt (p : ℕ) : UndecidedSite α :=
  match x.toList[p]? with
  | some c => .edge (st p) c
  | none => .stop

/-- The search's result as an outcome, against the walk's leaves `st`, with `hi` the search's
upper end. `unlearned` is the letter and the successor's leaf of the unlearned edge the walk
stopped at, if it stopped at one. -/
def searchOutcome (unlearned : Option (α × List Bool)) (hi : ℕ) : SearchResult → Outcome α
  | .undecidedHi => .undecided .stop (undecidedAt read T x hi) true
  | .allAgree =>
    match unlearned with
    | some (c, t) => .unlearnedEdge (st hi) c (prefixOf x hi) t
    | none => .agree
  | .edge p =>
    match x.toList[p - 1]?, T.sift read (prefixOf x p) with
    | some c, .inl t => .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t
    | _, _ => .undecided .stop [] false
  | .pair p | .triple p | .undecidedMid p =>
    .undecided (siteAt x st p) (undecidedAt read T x p) false
  | .undecidedLo | .disagreeLo => .undecided .start [] true

end Probe

/-- A probe of `x`: its first `k` letters sifted, then walked along the edges to `hi`, where the
walk stops; the unlearned edge it stops at, if any, has its successor sifted; and the search,
recovering past an undecided midpoint where `recover`, finds where the sift and the walk part
between `k` and `hi`. -/
def probe (recover : Bool) (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    (x : FreeMonoid α) : Outcome α :=
  match T.sift read (prefixOf x k) with
  | .inr z => .undecided .start [z] true
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let hi := k + ss.length - 1
    let search := (tripartiteSearch recover (siftAgrees read T x st) k hi).1
    match x.toList[hi]? with
    | some c =>
      match T.sift read (prefixOf x (hi + 1)) with
      | .inr z => .undecided .stop [z] true
      | .inl t => searchOutcome read T x st (some (c, t)) hi search
    | none => searchOutcome read T x st none hi search

/-- A probe, which counts as agreeing where it is shorter than `k`. -/
def probeOrAgree (recover : Bool) (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α)
    (k : ℕ) (x : FreeMonoid α) : Outcome α :=
  if x.toList.length < k then .agree else probe recover read T E k x

end Ideal

end OrthoDFA
