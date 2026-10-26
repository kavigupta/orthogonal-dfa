import OrthoDFA.Code.HypothesisDFA
import OrthoDFA.Code.TripartiteSearch

/-! A probe of the hypothesis: a string walked along the edges and sifted, and where the two
part. -/

namespace OrthoDFA

variable {α : Type*}

/-- How a probe ends. `wrongEdge s c u t`: `u` sifts to `s` and `u·c` to `t`, which the edge
`(s, c)` does not point at; `unlearnedEdge` is the same at an unlearned edge. `undecidedStart z`:
the probe's first `k` letters' sift read `z` undecided. `undecided zs`: the undecided strings the
probe stopped at past its start, by a probe that disagreed unless `clean`. -/
inductive Outcome (α : Type*)
  | agree
  | wrongEdge (s : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool)
  | unlearnedEdge (s : List Bool) (c : α) (u : FreeMonoid α) (t : List Bool)
  | undecidedStart (z : FreeMonoid α)
  | undecided (zs : List (FreeMonoid α)) (clean : Bool)

def Outcome.undecidedStrings : Outcome α → List (FreeMonoid α)
  | .undecidedStart z => [z]
  | .undecided zs _ => zs
  | _ => []

/-- The outcomes that do not disagree. -/
def Outcome.clean : Outcome α → Bool
  | .agree => true
  | .undecidedStart _ => true
  | .undecided _ cl => cl
  | _ => false

section Search

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

def siftAgrees (p : ℕ) : Option Bool :=
  (T.sift read (prefixOf x p)).elim (fun l => some (decide (l = st p))) fun _ => none

def undecidedAt (p : ℕ) : List (FreeMonoid α) :=
  (T.sift read (prefixOf x p)).elim (fun _ => []) fun z => [z]

/-- The search's result as an outcome, against the walk's leaves `st`, with `hi` the search's
upper end. `unlearned` is the letter and the successor's leaf of the unlearned edge the walk
stopped at, if it stopped at one. The search does not recover, so it reports no pair or
triple. -/
def searchOutcome (unlearned : Option (α × List Bool)) (hi : ℕ) : SearchResult → Outcome α
  | .undecidedHi => .undecided (undecidedAt read T x hi) true
  | .allAgree =>
    match unlearned with
    | some (c, t) => .unlearnedEdge (st hi) c (prefixOf x hi) t
    | none => .agree
  | .edge p =>
    match x.toList[p - 1]?, T.sift read (prefixOf x p) with
    | some c, .inl t => .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t
    | _, _ => .undecided [] false
  | .undecidedMid p => .undecided (undecidedAt read T x p) false
  | _ => .undecided [] true

end Search

/-- A probe of `x`: its first `k` letters sifted, then walked along the edges to `hi`, where the
walk stops; the unlearned edge it stops at, if any, has its successor sifted; and the search finds
where the sift and the walk part between `k` and `hi`, stopping at the first undecided read. -/
def probe (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α :=
  match T.sift read (prefixOf x k) with
  | .inr z => .undecidedStart z
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let hi := k + ss.length - 1
    let search := (tripartiteSearch false (siftAgrees read T x st) k hi).1
    match x.toList[hi]? with
    | some c =>
      match T.sift read (prefixOf x (hi + 1)) with
      | .inr z => .undecided [z] true
      | .inl t => searchOutcome read T x st (some (c, t)) hi search
    | none => searchOutcome read T x st none hi search

/-- A probe, which counts as agreeing where it is shorter than `k`. -/
def probeOrAgree (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    (x : FreeMonoid α) : Outcome α :=
  if x.toList.length < k then .agree else probe read T E k x

end OrthoDFA
