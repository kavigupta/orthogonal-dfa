import OrthoDFA.Loop
import OrthoDFA.FamilyRead

/-!
# The tally loop

The family is the oracle: a read of a string is accept, reject or undecided, and a node of the
tree reads `w·m` as one read. `rd` holds every string's read; its cut is the read's side, or
`none` where undecided.

Probes are fresh draws, each walked from `k` along the learned edges and searched for its first
disagreement as `sifting.read` does (`probeBy`, whose bisection steps past an undecided midpoint
whose neighbours read alike). A clean disagreement records, at the edge it crosses, the probe's
prefix there and the leaf the next prefix sifts to; a probe at an unlearned edge learns it.

Counts run over a stretch of probes, which starts afresh whenever the tree or the edges change:
start-undecided outcomes; undecided outcomes, each charged to an edge (a pair's or a triple's
first undecided position's outgoing edge on the walk, an end left undecided at the walk's last
edge); and decided disagreements (edges, pairs and triples). After every probe:
* the start or any edge's rate above its threshold ends the round with a harvest there;
* the disagreement rate settling below `εd` ends it in success;
* otherwise the edges settle. Where an edge has `m` records at a target it does not point at, its
  leaf splits on the letter and the midfix where that target and the current one part if the
  current one also has `m`, and otherwise it is redirected there. A split drops the records at
  the leaf and into it, clears the edges out of it and re-sifts the witnesses of those into it.

A tree past `Lmax` leaves, or edges unsettled after `fuel` fixes, end the round failed, and so
does running out of probes.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A read's side, or `none` where undecided. -/
def ARU.cut : ARU → Option Bool
  | .accept => some true
  | .reject => some false
  | .undecided => none

section Probe

variable (cut : FreeMonoid α → Option Bool)

/-- `kWalk` through `cut`. -/
def kWalkBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : KWalk α :=
  match t.sift cut (prefixOf x k) with
  | .inr _ => .anchor
  | .inl p =>
    match follow edges p (x.toList.drop k) with
    | .inl ps => .reached ps
    | .inr (s, c, i) => .edge s c (k + i)

/-- `walkTo` through `cut`. -/
def walkToBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (j : ℕ) :
    List (List Bool) :=
  match t.sift cut (prefixOf x k) with
  | .inr _ => []
  | .inl p => (follow edges p ((x.toList.drop k).take (j - k))).elim id fun _ => []

/-- `walkCheck` through `cut`. -/
def walkCheckBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Outcome α ⊕ (List (List Bool) × ℕ) :=
  match kWalkBy cut t edges k x with
  | .anchor => .inl (.startUndecided (prefixOf x k))
  | .edge s _ j =>
    match t.sift cut (prefixOf x (j + 1)) with
    | .inr _ => .inl (.endUndecided (prefixOf x (j + 1)))
    | .inl _ =>
      match t.sift cut (prefixOf x j) with
      | .inr _ => .inl (.endUndecided (prefixOf x j))
      | .inl p => if p = s then .inl (.member (prefixOf x j))
        else .inr (walkToBy cut t edges k x j, j)
  | .reached ps =>
    match t.sift cut x with
    | .inr _ => .inl (.endUndecided x)
    | .inl a => if some a = ps.getLast? then .inl .agree else .inr (ps, x.toList.length)

/-- `agreesAt` through `cut`. -/
def agreesAtBy (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (i : ℕ) :
    Option Bool :=
  (t.sift cut (prefixOf x i)).elim (fun p => some (decide (p = walkAt i))) fun _ => none

/-- What one probe comes to, read through `cut`. -/
def probeBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : Outcome α :=
  (walkCheckBy cut t edges k x).elim id fun d =>
    bracketAt (agreesAtBy cut t x fun j => d.1.getD (j - k) []) d.1 (d.2 - k) k d.2

/-- The record a probe of `x` makes against the hypothesis `h`: at a clean disagreement, the edge
it crosses, the leaf its next prefix sifts to, and its prefix at the edge. -/
def recordBy (k : ℕ) (h : DTree α × Edges α) (x : FreeMonoid α) :
    Option ((List Bool × α × List Bool) × FreeMonoid α) :=
  match probeBy cut h.1 h.2 k x with
  | .edge ps fd =>
    match x.toList[fd - 1]?, h.1.sift cut (prefixOf x fd) with
    | some c, .inl t => some ((ps.getD (fd - 1 - k) [], c, t), prefixOf x (fd - 1))
    | _, _ => none
  | _ => none

/-- The walk's edge out of position `i`: its leaf there and the probe's next letter. -/
def edgeAtBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (i : ℕ) :
    Option (List Bool × α) :=
  match (walkToBy cut t edges k x i).getLast?, x.toList[i]? with
  | some p, some c => some (p, c)
  | _, _ => none

/-- The edge an undecided outcome is charged to, with the prefix whose sift was left undecided. -/
def chargeBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Option ((List Bool × α) × FreeMonoid α) :=
  match probeBy cut t edges k x with
  | .endUndecided w =>
    (edgeAtBy cut t edges k x (w.toList.length - 1)).map (·, w)
  | .pair j | .triple j => (edgeAtBy cut t edges k x j).map (·, prefixOf x j)
  | _ => none

/-- `retarget` through `cut`. -/
def retargetBy (t' : DTree α) (p : List Bool) (edges : Edges α) : Edges α := fun q c =>
  match edges q c with
  | some (s, w) =>
    if s = p then (t'.sift cut (w * FreeMonoid.of c)).elim (fun s' => some (s', w)) fun _ => none
    else some (s, w)
  | none => none

end Probe

/-- What the loop carries: the hypothesis, its version, each edge's records (the probe's prefix at
the edge and the target leaf), and the stretch's counts: its probes, start-undecided outcomes,
undecided outcomes charged to each edge, and decided disagreements. -/
structure TState (α : Type*) where
  tree : DTree α
  edges : Edges α
  version : ℕ
  recs : List Bool → α → List (FreeMonoid α × List Bool)
  n : ℕ
  starts : ℕ
  und : List Bool → α → ℕ
  dis : ℕ

/-- `s` with a fresh stretch. -/
def TState.fresh (s : TState α) : TState α :=
  { s with n := 0, starts := 0, und := fun _ _ => 0, dis := 0 }

/-- The records of the edge `(p, c)` at the target `t`. -/
def TState.tally (s : TState α) (p : List Bool) (c : α) (t : List Bool) : ℕ :=
  ((s.recs p c).filter (·.2 = t)).length

/-- The edge `(p, c)` out of a leaf has `m` records at a target it does not point at. -/
def Violates (m : ℕ) (s : TState α) (p : List Bool) (c : α) (t : List Bool) : Prop :=
  p ∈ s.tree.paths ∧ (s.edges p c).map Prod.fst ≠ some t ∧ m ≤ s.tally p c t

/-- No edge has `m` records at a target it does not point at. -/
def Settled (m : ℕ) (s : TState α) : Prop := ∀ p c t, ¬ Violates m s p c t

/-- `s` with the edge `(p, c)` set to `e`, as a new version. -/
def TState.setEdge (s : TState α) (p : List Bool) (c : α) (e : List Bool × FreeMonoid α) :
    TState α :=
  let row := Function.update (s.edges p) c (some e)
  { s.fresh with edges := Function.update s.edges p row, version := s.version + 1 }

/-- `s` with the record `r` added at the edge `(p, c)`. -/
def TState.addRec (s : TState α) (p : List Bool) (c : α) (r : FreeMonoid α × List Bool) :
    TState α :=
  let row := Function.update (s.recs p) c (s.recs p c ++ [r])
  { s with recs := Function.update s.recs p row }

/-- `s` with one more undecided outcome charged to the edge `(p, c)`. -/
def TState.charge (s : TState α) (e : Option ((List Bool × α) × FreeMonoid α)) : TState α :=
  match e with
  | some ((p, c), _) =>
    { s with und := Function.update s.und p (Function.update (s.und p) c (s.und p c + 1)) }
  | none => s

/-- The first state: the root reads at `ε`, no edge learned. -/
def tallyStart : TState α :=
  ⟨.node 1 .leaf .leaf, fun _ _ => none, 0, fun _ _ => [], 0, 0, fun _ _ => 0, 0⟩

open scoped Classical in
/-- Fix the edge `(p, c)` at the target `t` it has `m` records at: split where the current target
also has `m`, else redirect, with a record's prefix as the witness. -/
noncomputable def fixEdge (cut : FreeMonoid α → Option Bool) (m : ℕ) (s : TState α)
    (p : List Bool) (c : α) (t : List Bool) : TState α :=
  let w := (((s.recs p c).find? (·.2 = t)).map Prod.fst).getD 1
  let redirect : TState α := s.setEdge p c (t, w)
  match s.edges p c with
  | some (t₀, _) =>
    if m ≤ s.tally p c t₀ then
      let d := FreeMonoid.of c * s.tree.midAt (lcp t t₀)
      let t' := s.tree.splitAt d p
      { s.fresh with
        tree := t'
        version := s.version + 1
        edges := fun q e => if p <+: q then none else retargetBy cut t' p s.edges q e
        recs := fun q e => if p <+: q then [] else (s.recs q e).filter (·.2 ≠ p) }
    else redirect
  | none => redirect

/-- How the round ends: in success, a harvest at the start or at an edge, the tree past `Lmax`
leaves, or the edges unsettled within `fuel` fixes. -/
inductive TEnd (α : Type*)
  | success
  | harvestStart
  | harvest (e : List Bool × α)
  | tooBig
  | stuck

open scoped Classical in
/-- The edges settle, fixing one edge at a time, within `fuel` fixes and `Lmax` leaves. -/
noncomputable def settleEdges (cut : FreeMonoid α → Option Bool) (m Lmax : ℕ) :
    ℕ → TState α → TState α ⊕ (TEnd α × TState α)
  | 0, s => if Settled m s then .inl s else .inr (.stuck, s)
  | fuel + 1, s =>
    if h : ∃ p c t, Violates m s p c t then
      if Lmax < (fixEdge cut m s h.choose h.choose_spec.choose
          h.choose_spec.choose_spec.choose).tree.paths.length then
        .inr (.tooBig, fixEdge cut m s h.choose h.choose_spec.choose
          h.choose_spec.choose_spec.choose)
      else settleEdges cut m Lmax fuel
        (fixEdge cut m s h.choose h.choose_spec.choose h.choose_spec.choose_spec.choose)
    else .inl s

/-- The loop's settings: the start, the records a fix takes, the caps, and the tests' thresholds,
level and first look. -/
structure TallyCfg where
  k : ℕ
  m : ℕ
  Lmax : ℕ
  fuel : ℕ
  θs : ℝ
  θe : ℝ
  εd : ℝ
  a : ℝ
  n₀ : ℕ

/-- A probe's outcome learned, recorded or counted. -/
def tallyPre (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (s : TState α)
    (x : FreeMonoid α) : TState α :=
  let s₁ := { s with n := s.n + 1 }
  match probeBy cut s.tree s.edges C.k x with
  | .member u =>
    match x.toList[u.toList.length]?, s.tree.sift cut u with
    | some c, .inl p =>
      match s.tree.sift cut (u * FreeMonoid.of c), s.edges p c with
      | .inl t, none => s.setEdge p c (t, u)
      | _, _ => s₁
    | _, _ => s₁
  | .edge _ _ =>
    let s₂ := { s₁ with dis := s₁.dis + 1 }
    match recordBy cut C.k (s.tree, s.edges) x with
    | some ((p, c, t), sp) => s₂.addRec p c (sp, t)
    | none => s₂
  | .pair _ | .triple _ =>
    ({ s₁ with dis := s₁.dis + 1 }).charge (chargeBy cut s.tree s.edges C.k x)
  | .endUndecided _ => s₁.charge (chargeBy cut s.tree s.edges C.k x)
  | .startUndecided _ => { s₁ with starts := s₁.starts + 1 }
  | .agree => s₁

open scoped Classical in
/-- The stretch's tests: the start's or an edge's undecided rate above its threshold, or the
disagreement rate settling below `εd`. -/
noncomputable def tallyLook (C : TallyCfg) (s : TState α) : Option (TEnd α) :=
  if rateSide C.θs C.a C.n₀ s.n s.starts = some true then some .harvestStart
  else if h : ∃ e : List Bool × α,
      e.1 ∈ s.tree.paths ∧ rateSide C.θe C.a C.n₀ s.n (s.und e.1 e.2) = some true then
    some (.harvest h.choose)
  else if rateSide C.εd C.a C.n₀ s.n s.dis = some false then some .success
  else none

/-- One probe: its outcome is learned, recorded or counted, the stretch's tests may end the round,
and otherwise the edges settle. -/
noncomputable def tallyStep (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (s : TState α)
    (x : FreeMonoid α) : TState α ⊕ (TEnd α × TState α) :=
  let s₁ := tallyPre C cut s x
  match tallyLook C s₁ with
  | some e => .inr (e, s₁)
  | none => settleEdges cut C.m C.Lmax C.fuel s₁

end OrthoDFA
