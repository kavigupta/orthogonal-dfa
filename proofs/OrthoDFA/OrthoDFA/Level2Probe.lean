import OrthoDFA.Hypothesis

/-! Level 2's probe: a string walked along the edges and sifted, and where the two part. -/

namespace OrthoDFA

namespace Ideal

variable {α : Type*}

inductive SearchResult
  | edge (p : ℕ)
  | triple (p : ℕ)
  | pair (p : ℕ)

/-- Bisection between `lo`, where the sift agrees with the walk, and `hi`, where it disagrees,
on the decided reads of `ag`: an edge `p` agrees at `p - 1` and disagrees at `p`, a triple is
undecided at `p` between the two, a pair undecided at `p` and `p + 1`. -/
def bisect (ag : ℕ → Option Bool) (lo hi : ℕ) : SearchResult :=
  if _h : hi ≤ lo + 1 then .edge hi
  else
    match ag ((lo + hi) / 2) with
    | some true => bisect ag ((lo + hi) / 2) hi
    | some false => bisect ag lo ((lo + hi) / 2)
    | none =>
      match ag ((lo + hi) / 2 - 1), ag ((lo + hi) / 2 + 1) with
      | none, _ => .pair ((lo + hi) / 2 - 1)
      | _, none => .pair ((lo + hi) / 2)
      | some true, some false => .triple ((lo + hi) / 2)
      | some false, _ => bisect ag lo ((lo + hi) / 2 - 1)
      | some true, some true => bisect ag ((lo + hi) / 2 + 1) hi
termination_by hi - lo
decreasing_by all_goals omega

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

/-- The bisection between `lo` and `hi`, against the walk's leaves `st`. -/
def locate (lo hi : ℕ) : Outcome α :=
  match bisect (siftAgrees read T x st) lo hi with
  | .edge p =>
    match x.toList[p - 1]?, T.sift read (prefixOf x p) with
    | some c, .inl t => .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t
    | _, _ => .undecided .stop [] false
  | .triple p => .undecided (siteAt x st p) (undecidedAt read T x p) false
  | .pair p =>
    .undecided (siteAt x st p) (undecidedAt read T x p ++ undecidedAt read T x (p + 1)) false

end Probe

/-- A probe of `x`: its first `k` letters sifted, then walked along the edges and sifted whole. -/
def probe (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α :=
  match T.sift read (prefixOf x k) with
  | .inr z => .undecided .start [z] true
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let j := k + ss.length - 1
    match x.toList[j]? with
    | some c =>
      match T.sift read (prefixOf x (j + 1)), T.sift read (prefixOf x j) with
      | .inr z, _ => .undecided .stop [z] true
      | _, .inr z => .undecided .stop [z] true
      | .inl t, .inl s =>
        if s = st j then .unlearnedEdge s c (prefixOf x j) t else locate read T x st k j
    | none =>
      match T.sift read x with
      | .inr z => .undecided .stop [z] true
      | .inl e => if e = st j then .agree else locate read T x st k j

end Ideal

end OrthoDFA
