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

Each place keeps its count over the whole round: the start, its undecided reads' strings, and
each edge, the reads charged to it and the strings of those that were undecided. Every position a
probe sifts past its start, the bisection's stepped-past middles included, charges its reads to an
edge: the walk's edge out of that position, or, where there is none, the walk's last edge. The
disagreement rate, and the undecided middles searches stop at, are counted over a stretch, which
starts afresh whenever the tree or the edges change. After every probe:
* the start-undecided rate above `θs`, an edge's undecided reads exceeding `θe` of its reads by
  `exc` of them, or the stretch's rate of searches stopping at an undecided middle (a pair or a
  triple) above `θpt`, ends the round with a harvest there;
* the disagreement rate settling below `εd` ends it in success;
* otherwise one edge is fixed. Where an edge has `m` records at a target it does not point at,
  its leaf splits on the letter and the midfix where that target and the current one part if the
  current one also has `m`, and otherwise it is redirected there. A split drops every record,
  clears the edges out of its leaf and re-sifts the witnesses of those into it.

A tree past `Lmax` leaves ends the round failed, and so does running out of probes.
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

/-- The positions `bracketAt` sifts: each middle, and its neighbours where it is undecided (the
right one only where the left one is decided). -/
def bracketSifts (agrees : ℕ → Option Bool) : ℕ → ℕ → ℕ → List ℕ
  | 0, _, _ => []
  | fuel + 1, lo, hi =>
    if lo + 1 < hi then
      let ag := fun p => if p = lo then some true else if p = hi then some false else agrees p
      let mid := (lo + hi) / 2
      match ag mid with
      | some true => mid :: bracketSifts agrees fuel mid hi
      | some false => mid :: bracketSifts agrees fuel lo mid
      | none =>
        match ag (mid - 1), ag (mid + 1) with
        | none, _ => [mid, mid - 1]
        | some true, some true =>
          mid :: (mid - 1) :: (mid + 1) :: bracketSifts agrees fuel (mid + 1) hi
        | some false, some _ =>
          mid :: (mid - 1) :: (mid + 1) :: bracketSifts agrees fuel lo (mid - 1)
        | some _, _ => [mid, mid - 1, mid + 1]
    else []

/-- The positions a probe sifts past its start, as `probeBy` reads them. -/
def siftsBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : List ℕ :=
  let found := fun (ps : List (List Bool)) (hi : ℕ) =>
    hi :: bracketSifts (agreesAtBy cut t x fun j => ps.getD (j - k) []) (hi - k) k hi
  let all := match kWalkBy cut t edges k x with
    | .anchor => []
    | .edge s _ j =>
      match t.sift cut (prefixOf x (j + 1)) with
      | .inr _ => [j + 1]
      | .inl _ =>
        match t.sift cut (prefixOf x j) with
        | .inr _ => [j + 1, j]
        | .inl p => if p = s then [j + 1, j] else (j + 1) :: found (walkToBy cut t edges k x j) j
    | .reached ps =>
      match t.sift cut x with
      | .inl a => if some a = ps.getLast? then [x.toList.length] else found ps x.toList.length
      | .inr _ => [x.toList.length]
  (all.filter (k < ·)).dedup

/-- The edge a position's reads are charged to: the walk's edge out of it, else the walk's last
edge. -/
def posEdgeBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (i : ℕ) :
    Option (List Bool × α) :=
  (edgeAtBy cut t edges k x i).orElse fun _ => edgeAtBy cut t edges k x (i - 1)

/-- How many reads a probe charges to the edge `e`. -/
def edgeReadsBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)
    (e : List Bool × α) : ℕ :=
  (((siftsBy cut t edges k x).filter fun i => posEdgeBy cut t edges k x i = some e).map
    fun i => (t.route cut (prefixOf x i)).1.length).sum

/-- The strings it read undecided there. -/
def edgeHarvBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)
    (e : List Bool × α) : List (FreeMonoid α) :=
  (siftsBy cut t edges k x).filterMap fun i =>
    if posEdgeBy cut t edges k x i = some e then
      (t.sift cut (prefixOf x i)).elim (fun _ => none) some
    else none

/-- The undecided middle its search stopped at, as a pair or a triple, if any. -/
def ptHarvBy (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  match probeBy cut t edges k x with
  | .pair j | .triple j => (t.sift cut (prefixOf x j)).elim (fun _ => []) fun b => [b]
  | _ => []

/-- The string it read undecided at its start, if any. -/
def startHarvBy (t : DTree α) (k : ℕ) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (t.sift cut (prefixOf x k)).elim (fun _ => []) fun b => [b]

/-- `retarget` through `cut`. -/
def retargetBy (t' : DTree α) (p : List Bool) (edges : Edges α) : Edges α := fun q c =>
  match edges q c with
  | some (s, w) =>
    if s = p then (t'.sift cut (w * FreeMonoid.of c)).elim (fun s' => some (s', w)) fun _ => none
    else some (s, w)
  | none => none

end Probe

/-- What the loop carries: the hypothesis, its version, each edge's records (the probe's prefix at
the edge and the target leaf), the stretch's probes and decided disagreements, and over the round
its probes, the strings read undecided at the start, and for each edge the reads charged to it and
the strings of those that were undecided. -/
structure TState (α : Type*) where
  tree : DTree α
  edges : Edges α
  version : ℕ
  recs : List Bool → α → List (FreeMonoid α × List Bool)
  n : ℕ
  dis : ℕ
  pt : List (FreeMonoid α)
  probes : ℕ
  startH : List (FreeMonoid α)
  reads : List Bool → α → ℕ
  harv : List Bool → α → List (FreeMonoid α)

/-- `s` with a fresh stretch. -/
def TState.fresh (s : TState α) : TState α := { s with n := 0, dis := 0, pt := [] }

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

/-- `s` with a probe counted and its reads charged to their places. -/
def TState.charge (cut : FreeMonoid α → Option Bool) (k : ℕ) (s : TState α)
    (x : FreeMonoid α) : TState α :=
  { s with
    n := s.n + 1
    pt := s.pt ++ ptHarvBy cut s.tree s.edges k x
    probes := s.probes + 1
    startH := s.startH ++ startHarvBy cut s.tree k x
    reads := fun p c => s.reads p c + edgeReadsBy cut s.tree s.edges k x (p, c)
    harv := fun p c => s.harv p c ++ edgeHarvBy cut s.tree s.edges k x (p, c) }

/-- The first state: the root reads at `ε`, no edge learned. -/
def tallyStart : TState α :=
  ⟨.node 1 .leaf .leaf, fun _ _ => none, 0, fun _ _ => [], 0, 0, [], 0, [], fun _ _ => 0,
    fun _ _ => []⟩

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
        recs := fun _ _ => [] }
    else redirect
  | none => redirect

/-- How the round ends: in success, a harvest at the start, at an edge or of the undecided
middles, or the tree past `Lmax` leaves. -/
inductive TEnd (α : Type*)
  | success
  | harvestStart
  | harvest (e : List Bool × α)
  | harvestPT
  | tooBig

open scoped Classical in
/-- One edge with `m` records at a target it does not point at is fixed, within `Lmax` leaves. -/
noncomputable def settleOne (cut : FreeMonoid α → Option Bool) (m Lmax : ℕ) (s : TState α) :
    TState α ⊕ (TEnd α × TState α) :=
  if h : ∃ p c t, Violates m s p c t then
    if Lmax < (fixEdge cut m s h.choose h.choose_spec.choose
        h.choose_spec.choose_spec.choose).tree.paths.length then
      .inr (.tooBig, fixEdge cut m s h.choose h.choose_spec.choose
        h.choose_spec.choose_spec.choose)
    else .inl (fixEdge cut m s h.choose h.choose_spec.choose h.choose_spec.choose_spec.choose)
  else .inl s

/-- The loop's settings: the start, the records a fix takes, the cap, the tests' thresholds, level
and first look, and the excess `exc r` an edge's undecided reads need after `r` reads. -/
structure TallyCfg where
  k : ℕ
  m : ℕ
  Lmax : ℕ
  θs : ℝ
  θe : ℝ
  θpt : ℝ
  εd : ℝ
  a : ℝ
  n₀ : ℕ
  exc : ℕ → ℝ

open scoped Classical in
/-- A probe's reads charged, and its outcome learned, recorded or counted. -/
noncomputable def tallyPre (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (s : TState α)
    (x : FreeMonoid α) : TState α :=
  let s₁ := s.charge cut C.k x
  match probeBy cut s.tree s.edges C.k x with
  | .member u =>
    match x.toList[u.toList.length]?, s.tree.sift cut u with
    | some c, .inl p =>
      match s.tree.sift cut (u * FreeMonoid.of c), s.edges p c with
      | .inl t, none => s₁.setEdge p c (t, u)
      | _, _ => s₁
    | _, _ => s₁
  | .edge _ _ =>
    let s₂ := { s₁ with dis := s₁.dis + 1 }
    match recordBy cut C.k (s.tree, s.edges) x with
    | some ((p, c, t), sp) => s₂.addRec p c (sp, t)
    | none => s₂
  | .pair _ | .triple _ => { s₁ with dis := s₁.dis + 1 }
  | _ => s₁

open scoped Classical in
/-- The tests: the start's undecided rate above `θs` over the round, an edge's undecided reads
exceeding `θe` of its reads by `exc` of its reads, the stretch's rate of searches stopping at an
undecided middle above `θpt`, or its disagreement rate settling below `εd`. -/
noncomputable def tallyLook (C : TallyCfg) (s : TState α) : Option (TEnd α) :=
  if rateSide C.θs C.a C.n₀ s.probes s.startH.length = some true then some .harvestStart
  else if h : ∃ e : List Bool × α, e.1 ∈ s.tree.paths
      ∧ C.exc (s.reads e.1 e.2) ≤ ((s.harv e.1 e.2).length : ℝ) - C.θe * s.reads e.1 e.2 then
    some (.harvest h.choose)
  else if rateSide C.θpt C.a C.n₀ s.n s.pt.length = some true then some .harvestPT
  else if rateSide C.εd C.a C.n₀ s.n s.dis = some false then some .success
  else none

/-- One probe: its reads are charged and its outcome learned, recorded or counted, the tests may
end the round, and otherwise one edge is fixed. -/
noncomputable def tallyStep (C : TallyCfg) (cut : FreeMonoid α → Option Bool) (s : TState α)
    (x : FreeMonoid α) : TState α ⊕ (TEnd α × TState α) :=
  let s₁ := tallyPre C cut s x
  match tallyLook C s₁ with
  | some e => .inr (e, s₁)
  | none => settleOne cut C.m C.Lmax s₁

end OrthoDFA
