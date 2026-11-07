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

def siftAgreesQ (p : ℕ) : Query (FreeMonoid α) ARU (Option Bool) :=
  (T.siftQ (prefixOf x p)).map fun r => r.elim (fun l => some (decide (l = st p))) fun _ => none

def undecidedAtQ (p : ℕ) : Query (FreeMonoid α) ARU (List (FreeMonoid α)) :=
  (T.siftQ (prefixOf x p)).map fun r => r.elim (fun _ => []) fun z => [z]

/-- The search's result as an outcome, against the walk's leaves `st`, with `hi` the search's
upper end. `unlearned` is the letter and the successor's leaf of the unlearned edge the walk
stopped at, if it stopped at one. The search does not recover, so it reports no pair or
triple. -/
def searchOutcomeQ (unlearned : Option (α × List Bool)) (hi : ℕ) :
    SearchResult → Query (FreeMonoid α) ARU (Outcome α)
  | .undecidedHi => (undecidedAtQ T x hi).map fun zs => .undecided zs true
  | .allAgree =>
    .pure (match unlearned with
      | some (c, t) => .unlearnedEdge (st hi) c (prefixOf x hi) t
      | none => .agree)
  | .edge p =>
    match x.toList[p - 1]? with
    | some c => (T.siftQ (prefixOf x p)).map fun r =>
      match r with
      | .inl t => .wrongEdge (st (p - 1)) c (prefixOf x (p - 1)) t
      | .inr _ => .undecided [] false
    | none => .pure (.undecided [] false)
  | .undecidedMid p => (undecidedAtQ T x p).map fun zs => .undecided zs false
  | _ => .pure (.undecided [] true)

def siftAgrees (p : ℕ) : Option Bool := (siftAgreesQ T x st p).eval read

def undecidedAt (p : ℕ) : List (FreeMonoid α) := (undecidedAtQ T x p).eval read

def searchOutcome (unlearned : Option (α × List Bool)) (hi : ℕ) (r : SearchResult) : Outcome α :=
  (searchOutcomeQ T x st unlearned hi r).eval read

end Search

/-- A probe of `x`: its first `k` letters sifted, then walked along the edges to `hi`, where the
walk stops; the unlearned edge it stops at, if any, has its successor sifted; and the search finds
where the sift and the walk part between `k` and `hi`, stopping at the first undecided read. -/
def probeQ (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Query (FreeMonoid α) ARU (Outcome α) :=
  (T.siftQ (prefixOf x k)).bind fun
  | .inr z => .pure (.undecidedStart z)
  | .inl a =>
    let ss := walk E a (x.toList.drop k)
    let st := fun p => ss.getD (p - k) []
    let hi := k + ss.length - 1
    let search := (tripartiteQ false k hi).subst (siftAgreesQ T x st)
    match x.toList[hi]? with
    | some c =>
      (T.siftQ (prefixOf x (hi + 1))).bind fun
      | .inr z => .pure (.undecided [z] true)
      | .inl t => search.bind (searchOutcomeQ T x st (some (c, t)) hi)
    | none => search.bind (searchOutcomeQ T x st none hi)

def probe (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α :=
  (probeQ T E k x).eval read

/-- A probe, which counts as agreeing where it is shorter than `k`. -/
def probeOrAgreeQ (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Query (FreeMonoid α) ARU (Outcome α) :=
  if x.toList.length < k then .pure .agree else probeQ T E k x

def probeOrAgree (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)
    (x : FreeMonoid α) : Outcome α :=
  (probeOrAgreeQ T E k x).eval read

end OrthoDFA
