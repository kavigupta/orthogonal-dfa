import OrthoDFA.RoundStrong

/-!
# The L\* loop

Probes are fresh draws, each walked from `k` along the learned edges and searched for its first
disagreement (`probeOutcome`). Counts run over a stretch of probes, which starts afresh whenever
the tree or the edges change, and are tested after every probe:
* a start, end, triple or pair rate above its threshold ends the loop with a harvest of those
  strings;
* a disagreement rate (edges, pairs and triples) settling below `1 − acc` ends it consistent;
* at `nmax`, with nothing settled and no change, the loop ends flagged unsettled.

A probe reaching an unlearned edge learns it from its own prefix. A probe ending at an edge
records its prefix and the leaf its next prefix sifts to. An edge with `m` of a stretch's records
at a target it does not point at is redirected there, once; with `m` at two targets, or after a
redirect at another, its leaf splits on the letter and the midfix where the targets part. A tree
past `Lmax` leaves ends the loop flagged, and so does running out of the round's probes.

`LoopSucceeds`: with the thresholds below `τ₀`, the loop ends consistent with the draws
disagreeing at most `1 − acc` of the time, or with a harvest of a class above its threshold, but
for chance at most its tests' levels over the round's probes, a decided read off its side, more
than `j` in-band reads off their side, a misread string drawn `m / j` times within a stretch, and
an unsettled stretch.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The undecided outcomes the loop's exits watch. -/
inductive Cls
  | start | stop | triple | pair | edge
  deriving DecidableEq

instance : Fintype Cls :=
  ⟨{.start, .stop, .triple, .pair, .edge}, fun c => by cases c <;> simp⟩

/-- The class an outcome counts toward. -/
def Outcome.cls : Outcome α → Option Cls
  | .startUndecided _ => some .start
  | .endUndecided _ => some .stop
  | .triple _ => some .triple
  | .pair _ => some .pair
  | .edge _ _ => some .edge
  | _ => none

/-- The outcomes that disagree: an edge, a pair or a triple. -/
def Outcome.Disagrees (o : Outcome α) : Prop :=
  o.cls = some .edge ∨ o.cls = some .pair ∨ o.cls = some .triple

/-- The loop's settings: each class's threshold by tree depth. -/
structure LoopCfg where
  k : ℕ
  L : ℕ
  m : ℕ
  nmax : ℕ
  Lmax : ℕ
  acc : ℝ
  a : ℝ
  θ : Cls → ℕ → ℝ

/-- What the loop carries: the hypothesis, each edge's records this stretch, whether it has been
redirected since it was learned, the strings read so far, and the stretch's counts. -/
structure LoopState (α : Type*) where
  tree : DTree α
  edges : Edges α
  recs : List Bool → α → List (FreeMonoid α × List Bool)
  moved : List Bool → α → Bool
  log : Set (FreeMonoid α)
  n : ℕ
  hits : Cls → ℕ

/-- How the loop ends: (a) consistent, (b) a harvest, (c) no progress: the tree too big or a
stretch unsettled. -/
inductive LoopEnd
  | agree
  | harvest (c : Cls)
  | tooBig
  | unsettled

/-- `s` with a fresh stretch. -/
def LoopState.fresh (s : LoopState α) : LoopState α :=
  { s with recs := fun _ _ => [], n := 0, hits := fun _ => 0 }

/-- The stretch's disagreeing probes. -/
def LoopState.dis (s : LoopState α) : ℕ := s.hits .edge + s.hits .pair + s.hits .triple

/-- The longest common prefix. -/
def lcp : List Bool → List Bool → List Bool
  | a :: as, b :: bs => if a = b then a :: lcp as bs else []
  | _, _ => []

/-- The midfix at the node `p` leads to, `ε` past a leaf. -/
def DTree.midAt : DTree α → List Bool → FreeMonoid α
  | .leaf, _ => 1
  | .node m _ _, [] => m
  | .node _ r _, false :: p => midAt r p
  | .node _ _ a, true :: p => midAt a p

/-- After the leaf `p` splits into `t'`, an edge into `p` sifts its witness's successor again. -/
noncomputable def retarget (R : CutReads α) (t' : DTree α) (p : List Bool) (edges : Edges α) :
    Edges α := fun q c =>
  match edges q c with
  | some (s, w) =>
    if s = p then (t'.sift R.cut (w * FreeMonoid.of c)).elim (fun s' => some (s', w)) fun _ => none
    else some (s, w)
  | none => none

/-- A change of the hypothesis an edge's records call for. -/
inductive Change (α : Type*)
  | split (d : FreeMonoid α)
  | redirect (t : List Bool) (w : FreeMonoid α)

/-- `s` with the edge `(p, c)` set to `e`. -/
def LoopState.setEdge (s : LoopState α) (p : List Bool) (c : α)
    (e : Option (List Bool × FreeMonoid α)) (moved : Bool) : LoopState α :=
  { s with edges := Function.update s.edges p (Function.update (s.edges p) c e),
           moved := Function.update s.moved p (Function.update (s.moved p) c moved) }

/-- `s` with a record of `(sp, t)` at the edge `(p, c)`. -/
def LoopState.record (s : LoopState α) (p : List Bool) (c : α) (sp : FreeMonoid α)
    (t : List Bool) : LoopState α :=
  let r := Function.update (s.recs p) c (s.recs p c ++ [(sp, t)])
  { s with recs := Function.update s.recs p r }

end OrthoDFA
