import OrthoDFA.Exhausted

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

/-- The first state: the root reads at `ε`, no edge learned. -/
def loopStart : LoopState α :=
  ⟨.node 1 .leaf .leaf, fun _ _ => none, fun _ _ => [], fun _ _ => false, ∅, 0, fun _ => 0⟩

/-- `s` with a fresh stretch. -/
def LoopState.fresh (s : LoopState α) : LoopState α :=
  { s with recs := fun _ _ => [], n := 0, hits := fun _ => 0 }

/-- The stretch's disagreeing probes. -/
def LoopState.dis (s : LoopState α) : ℕ := s.hits .edge + s.hits .pair + s.hits .triple

/-- What sifting `z` in `t` may read: `z` followed by a midfix. -/
def readsOf (t : DTree α) (z : FreeMonoid α) : Set (FreeMonoid α) :=
  {y | ∃ m ∈ t.midfixes, y = z * m}

/-- What a probe of `x` may read against `t`: each prefix of `x`, sifted. -/
def probeLog (t : DTree α) (x : FreeMonoid α) : Set (FreeMonoid α) :=
  {y | ∃ i ≤ x.toList.length, y ∈ readsOf t (prefixOf x i)}

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

/-- What the retargeted edges into `p` read. -/
def retargetLog (t' : DTree α) (p : List Bool) (edges : Edges α) : Set (FreeMonoid α) :=
  {y | ∃ q c w, edges q c = some (p, w) ∧ y ∈ readsOf t' (w * FreeMonoid.of c)}

/-- A change of the hypothesis an edge's records call for. -/
inductive Change (α : Type*)
  | split (d : FreeMonoid α)
  | redirect (t : List Bool) (w : FreeMonoid α)

open scoped Classical in
/-- What the records of the edge `(p, c)` call for: a split on the letter and the midfix where two
targets with `m` records part, or after a redirect where a second one gets `m`; a redirect to
one target with `m` records the edge does not point at. -/
noncomputable def edgeChange (m : ℕ) (s : LoopState α) (p : List Bool) (c : α) :
    Option (Change α) :=
  let cur := (s.edges p c).map Prod.fst
  let strong := (((s.recs p c).map Prod.snd).dedup).filter fun t =>
    m ≤ ((s.recs p c).filter (·.2 = t)).length
  match strong, cur with
  | t₁ :: t₂ :: _, _ => some (.split (FreeMonoid.of c * s.tree.midAt (lcp t₁ t₂)))
  | [t], some t₀ =>
    if s.moved p c then some (.split (FreeMonoid.of c * s.tree.midAt (lcp t t₀)))
    else ((s.recs p c).find? (·.2 = t)).map fun r => .redirect t r.1
  | _, _ => none

open scoped Classical in
/-- The stretch's tests, after every probe. -/
noncomputable def look (C : LoopCfg) (s : LoopState α) : LoopState α ⊕ (LoopEnd × LoopState α) :=
  let d := s.tree.depth
  let fires := fun c => rateSide (C.θ c d) C.a 1 s.n (s.hits c) = some true
  if fires .start then .inr (.harvest .start, s)
  else if fires .stop then .inr (.harvest .stop, s)
  else if fires .triple then .inr (.harvest .triple, s)
  else if fires .pair then .inr (.harvest .pair, s)
  else if rateSide (1 - C.acc) C.a 1 s.n s.dis = some false then .inr (.agree, s)
  else if C.nmax ≤ s.n then .inr (.unsettled, s)
  else .inl s

/-- `s` with the edge `(p, c)` set to `e`. -/
def LoopState.setEdge (s : LoopState α) (p : List Bool) (c : α)
    (e : Option (List Bool × FreeMonoid α)) (moved : Bool) : LoopState α :=
  { s with edges := Function.update s.edges p (Function.update (s.edges p) c e),
           moved := Function.update s.moved p (Function.update (s.moved p) c moved) }

open scoped Classical in
/-- Apply a change to the edge `(p, c)`, ending the loop if the tree grows past `Lmax`. -/
noncomputable def applyChange (C : LoopCfg) (R : CutReads α) (s : LoopState α) (p : List Bool)
    (c : α) : Change α → LoopState α ⊕ (LoopEnd × LoopState α)
  | .split d =>
    let t' := s.tree.splitAt d p
    let lg := s.log ∪ retargetLog t' p s.edges
    let s' := LoopState.fresh { s with tree := t', edges := retarget R t' p s.edges, log := lg }
    let s'' := { s' with moved := fun _ _ => false }
    if C.Lmax < t'.paths.length then .inr (.tooBig, s'') else .inl s''
  | .redirect t w => .inl (LoopState.fresh (s.setEdge p c (some (t, w)) true))

/-- A count of `c` more. -/
def bump (h : Cls → ℕ) (c : Cls) : Cls → ℕ := Function.update h c (h c + 1)

/-- `s` with a record of `(sp, t)` at the edge `(p, c)`. -/
def LoopState.record (s : LoopState α) (p : List Bool) (c : α) (sp : FreeMonoid α)
    (t : List Bool) : LoopState α :=
  let r := Function.update (s.recs p) c (s.recs p c ++ [(sp, t)])
  { s with recs := Function.update s.recs p r }

open scoped Classical in
/-- One probe: its reads join the log, then its outcome is counted or acted on. -/
noncomputable def loopStep (C : LoopCfg) (R : CutReads α) (s : LoopState α) (x : FreeMonoid α) :
    LoopState α ⊕ (LoopEnd × LoopState α) :=
  let s₁ := { s with n := s.n + 1, log := s.log ∪ probeLog s.tree x }
  match probeOutcome R s.tree s.edges C.k x with
  | .member u =>
    match x.toList[u.toList.length]?, s.tree.sift R.cut u with
    | some c, .inl p =>
      match s.tree.sift R.cut (u * FreeMonoid.of c) with
      | .inl t => .inl (LoopState.fresh (s₁.setEdge p c (some (t, u)) false))
      | .inr _ => look C { s₁ with hits := bump s.hits .stop }
    | _, _ => look C { s₁ with hits := bump s.hits .stop }
  | .edge ps fd =>
    let p := ps.getD (fd - 1 - C.k) []
    let sp := prefixOf x (fd - 1)
    let s₂ := { s₁ with hits := bump s.hits .edge }
    match x.toList[fd - 1]?, s.tree.sift R.cut (prefixOf x fd) with
    | some c, .inl t =>
      let s₃ := s₂.record p c sp t
      match edgeChange C.m s₃ p c with
      | some ch => applyChange C R s₃ p c ch
      | none => look C s₃
    | _, _ => look C s₂
  | o =>
    match o.cls with
    | some cl => look C { s₁ with hits := bump s.hits cl }
    | none => look C s₁

/-- The loop over the probes `xs`; `none` where they run out first. -/
noncomputable def loopRun (C : LoopCfg) (R : CutReads α) :
    LoopState α → List (FreeMonoid α) → LoopState α × Option LoopEnd
  | s, [] => (s, none)
  | s, x :: xs =>
    match loopStep C R s x with
    | .inl s' => loopRun C R s' xs
    | .inr (e, s') => (s', some e)

/-- The share of draws the hypothesis `s` brings to an outcome satisfying `P`. -/
noncomputable def outRate (C : LoopCfg) (R : CutReads α) (D : Measure (FreeMonoid α))
    (s : LoopState α) (P : Outcome α → Prop) : ℝ :=
  D.real {x | P (probeOutcome R s.tree s.edges C.k x)}

/-- What each ending claims: consistent, the draws disagreeing at most `1 − acc` of the time; a
harvest, its class above its threshold; no progress or the probes run out, nothing. -/
def LoopGenuine (C : LoopCfg) (R : CutReads α) (D : Measure (FreeMonoid α)) (s : LoopState α) :
    Option LoopEnd → Prop
  | some .agree => outRate C R D s Outcome.Disagrees ≤ 1 - C.acc
  | some (.harvest c) => C.θ c s.tree.depth < outRate C R D s (·.cls = some c)
  | _ => False

section Band

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `z`'s mean count lies in the band. -/
def InBand (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (z : FreeMonoid α) : Prop :=
  (B.lo : ℝ) < meanVote O F z ∧ meanVote O F z ≤ B.hi

/-- The side a read of `z` belongs on: accepting where its mean count is past the band's centre. -/
noncomputable def refSide (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (z : FreeMonoid α) : Bool :=
  decide (((B.lo + B.hi : ℕ) : ℝ) < 2 * meanVote O F z)

/-- The chance the cut decides `z` off its side. -/
noncomputable def wrongProb (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (z : FreeMonoid α) : ℝ :=
  μ.real {ω | (readsAt O B F ω).cut z = some (!refSide O B F z)}

/-- The chance the cut leaves `z` undecided. -/
noncomputable def undecProb (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (z : FreeMonoid α) : ℝ :=
  μ.real {ω | (readsAt O B F ω).cut z = none}

/-- The most a read outside the band is decided off its side. -/
noncomputable def crossB (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) :
    ℝ :=
  ⨆ z : {z // ¬ InBand O B F z}, wrongProb O B F z

/-- The most a read inside the band is decided off its side, per chance it is undecided. -/
noncomputable def kappa (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) :
    ℝ :=
  ⨆ z : {z // InBand O B F z}, wrongProb O B F z / undecProb O B F z

end Band

section Budget

/-- `P(Bin(n, p) ≤ h)`. -/
noncomputable def binomCdfLe (n : ℕ) (p : ℝ) (h : ℕ) : ℝ := 1 - binomSfGe n p (h + 1)

/-- The most hits at which a rate test at `θ` and level `a` settles below over `n` draws. -/
noncomputable def lowCount (θ a : ℝ) (n : ℕ) : ℕ := sSup {h | binomCdfLe n θ h < a}

/-- The stretches a loop with no misread runs: each but the last ends at a change, a split (at
most `|Q|`), a redirect (one per learned edge) or a learning (the root's edges, and per split
the new leaves' edges and those into the split leaf). -/
def stretchGood (nα nQ : ℕ) : ℕ := 1 + nQ + 2 * (2 * nα + nQ * nα * (nQ + 4))

/-- The most strings the round reads over `P` probes: each prefix of a probe sifted, and each
split's retargeted witnesses. -/
def readMax (C : LoopCfg) (nα P : ℕ) : ℕ :=
  P * (C.L + 2) * C.Lmax + C.Lmax * nα * C.Lmax ^ 2

/-- An unsettled stretch's chance with no misread, where edges run at `1 − acc − η` or less:
the disagreement short of settling below `1 − acc`, or edges past it short of `m` records at one
of the `|Σ|·|Q|` edges and targets correct reads reach. -/
noncomputable def unsettledAt (C : LoopCfg) (nα nQ : ℕ) (η : ℝ) : ℝ :=
  binomSfGe C.nmax (1 - C.acc - η) (lowCount (1 - C.acc) C.a C.nmax + 1)
    + binomCdfLe C.nmax (1 - C.acc - η) ((C.m - 1) * (nα * nQ) + C.m)

/-- A misread string drawn `r` times among `nmax + 1` draws, for `j` strings in each window of
`P` probes: `(e·(nmax + 1)·prefixMax/(r − 1))^(r − 1)` each. -/
noncomputable def hitBound (C : LoopCfg) (D : Measure (FreeMonoid α)) (P j r : ℕ) : ℝ :=
  P * j * (Real.exp 1 * (C.nmax + 1) * prefixMax D C.k / (r - 1 : ℕ)) ^ (r - 1)

end Budget

/-- The harvest tests fire on a stretch's first hit up to `nmax + 1` probes: the thresholds are
below `τ₀`. -/
def BelowTau (C : LoopCfg) : Prop :=
  ∀ c ∈ ({.start, .stop, .triple, .pair} : Finset Cls), ∀ d ≤ C.Lmax,
    binomSfGe (C.nmax + 1) (C.θ c d) 1 < C.a

/-- `LoopSucceeds`: below `τ₀`, from the root, over `P` fresh draws, enough for the stretches a
loop with no misread runs, the loop ends genuinely consistent or with a genuine harvest, but for
chance at most its tests' levels over the `P` probes, a decided read off its side among the
reads of `P` probes, more than `j` in-band reads off their side, a misread string drawn `m / j`
times within a stretch, and an unsettled stretch with none of these. -/
def LoopSucceeds : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} [Fintype Q] (C : LoopCfg) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (P j : ℕ) (η : ℝ),
    O.L = {w | A.state w ∈ A.accept} → SuffixFree F → (∀ᵐ x ∂D, x.toList.length = C.L) →
    C.k ≤ C.L → 1 ≤ j → 2 ≤ C.m / j → Fintype.card Q + 3 ≤ C.Lmax → BelowTau C →
    stretchGood (Fintype.card α) (Fintype.card Q) * (C.nmax + 1) ≤ P →
    1 - (5 * P * C.a + readMax C (Fintype.card α) P * crossB O B F + kappa O B F ^ (j + 1)
        + hitBound C D P j (C.m / j)
        + stretchGood (Fintype.card α) (Fintype.card Q)
          * unsettledAt C (Fintype.card α) (Fintype.card Q) η)
      ≤ (μ.prod (Measure.pi fun _ : Fin P => D)).real
          {p | LoopGenuine C (readsAt O B F p.1) D
            (loopRun C (readsAt O B F p.1) loopStart (List.ofFn p.2)).1
            (loopRun C (readsAt O B F p.1) loopStart (List.ofFn p.2)).2}

end OrthoDFA
