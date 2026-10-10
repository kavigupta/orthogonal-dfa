import OrthoDFA.RoundLevel

/-!
# The round as Python runs it

The round of `RoundLevel` with what Python carries across its readings: each pass starts its quiet
streak afresh, a split is recorded with what it split, and the probes spent are counted against a
backstop budget set by the leaves. Every edge a refusal draw's search ends at is live.

A gate is a pass only where its test settles above `acc`; one that does not settle is refused,
and the certificate's failure chance is spent per call. On a refusal, a class that fires ends the
round holding it; only when none fires do the live edges rerun.

The round ends exhausted only when, reading after reading, the refusal sample finds some live edge
and no class firing, until the probes spent reach the budget.

`RoundStrongReadings`: every reading but the last spends at least `patience` probes, so the
readings times `patience` are at most the budget and one more `patience`.
`RoundStrongLeaves`: the leaves are at most `|Q| + 2` and the splits that one of their at most
`2·depth + 2` reads, landing off a reference placement's side, let through.
`RoundStrongSameState`: a split between two strings of one state is one of those.
`RoundStrongLeafPaths`: the learned edges join leaves.
`RoundStrongNoStop`: at the hypothesis the round ends with, every attempt on an edge a draw's
search ends at splits or adds a member.
`RoundStrongTrichotomy` and `RoundStrongQuality`: `RoundTrichotomyLevel` and
`RoundQualityLevel` for this round.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A round's settings: as `RoundCfg`, with the probes the round may spend by its leaf count, and
the certificate by how many calls came before, reading the bits at its draws. -/
structure StrongCfg (α : Type*) where
  K : StageKnobs α
  k : ℕ
  L : ℕ
  np : ℕ
  ng : ℕ
  nr : ℕ
  nc : ℕ
  f : ℝ
  c : ℝ
  θM : ℝ
  acc : ℝ
  a : ℝ
  budget : ℕ → ℕ
  cert : ℕ → DTree α → Edges α → (Fin nc → FreeMonoid α) → (Fin nc → ℝ) → Bool

/-- One reading's draws: probes, gate batch, refusal sample, certificate sample. -/
abbrev StrongCfg.Draws (C : StrongCfg α) : Type _ :=
  (Fin C.np → FreeMonoid α) × (Fin C.ng → FreeMonoid α) × (Fin C.nr → FreeMonoid α)
    × (Fin C.nc → FreeMonoid α)

/-- One reading's draws, each from `D`. -/
noncomputable def StrongCfg.drawMeasure (C : StrongCfg α) (D : Measure (FreeMonoid α)) :
    Measure C.Draws :=
  (Measure.pi fun _ : Fin C.np => D).prod ((Measure.pi fun _ : Fin C.ng => D).prod
    ((Measure.pi fun _ : Fin C.nr => D).prod (Measure.pi fun _ : Fin C.nc => D)))

/-- A split: the tree it split, the leaf, the distinguisher, the edge's witness and the probe's
prefix. -/
structure SplitRec (α : Type*) where
  tree : DTree α
  leaf : List Bool
  d : FreeMonoid α
  y : FreeMonoid α
  sprime : FreeMonoid α

/-- What the round carries: the pass's state, its splits, the probes spent, the certificate's
calls, and what the readings have read outside the passes. -/
structure RoundAcc (α : Type*) where
  s : KState α
  splits : List (SplitRec α)
  used : ℕ
  certs : ℕ
  reads : Finset (FreeMonoid α)

open scoped Classical in
/-- What a reading's gate, refusal sample and certificate may read against `t`: a gate draw, or a
refusal draw's prefix from `k` on and a midfix, then a suffix of the family or its training half;
or a certificate draw. -/
noncomputable def readingReads (C : StrongCfg α) (F : Finset (FreeMonoid α)) (t : DTree α)
    (y : C.Draws) : Finset (FreeMonoid α) :=
  (Finset.univ.biUnion fun i => (F ∪ C.K.train F).image (y.2.1 i * ·))
    ∪ (Finset.univ.biUnion fun i =>
      ((((List.range ((y.2.2.1 i).toList.length + 1)).map fun j =>
        prefixOf (y.2.2.1 i) (max C.k j)).toFinset ×ˢ t.midfixes) ×ˢ (F ∪ C.K.train F)).image
          fun z => z.1.1 * z.1.2 * z.2)
    ∪ Finset.univ.image y.2.2.2

/-- One probe: `probeStepK`, a split recorded with the tree it split. -/
noncomputable def strongStep (C : StrongCfg α) (R : CutReads α) (A : RoundAcc α)
    (x : FreeMonoid α) : RoundAcc α :=
  let s := A.s
  let A₁ := { A with s := probeStepK C.K R C.k s x, used := A.used + 1 }
  match probeOutcome R s.tree s.edges C.k x with
  | .edge ps fd =>
    match seedStep C.K R s.tree s.pool s.edges (stepSkip C.K R C.k s x) C.K.forced C.k x ps fd with
    | .split d s1 y sp => { A₁ with splits := A.splits ++ [⟨s.tree, s1, d, y, sp⟩] }
    | _ => A₁
  | _ => A₁

/-- `counterexample_pass`: from a fresh quiet streak, with the readings' reads logged, probes in
order until `patience` in a row are quiet or the budget is spent. -/
noncomputable def strongPass (C : StrongCfg α) (R : CutReads α) (A : RoundAcc α)
    (probes : List (FreeMonoid α)) : RoundAcc α :=
  probes.foldl (fun A x =>
      if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then A
      else strongStep C R A x)
    { A with s := { A.s with streak := 0, log := A.s.log ∪ A.reads } }

/-- How a round ends: consistent at a start, holding the classes that fired and halving where
pairs were over half, halving, with the budget spent, or out of readings. -/
inductive StrongEnd
  | consistent (q : List Bool)
  | harvest (halve : Bool)
  | halve (gateRefused : Bool)
  | exhausted
  | cap

open scoped Classical in
/-- Reading `j`: the pass with `first` ahead of the reading's probes; the gate at failure chance
`a·2⁻ʲ`, and on a settled pass the certificate; otherwise a class firing on the refusal sample is
held, halving where pairs were over half, else its live-edge draws rerun while the budget lasts,
else the limit halves. -/
noncomputable def strongReading (C : StrongCfg α) (R : CutReads α) (j : ℕ) (A : RoundAcc α)
    (first : List (FreeMonoid α)) (y : C.Draws) : RoundAcc α × (StrongEnd ⊕ List (FreeMonoid α)) :=
  let A' := strongPass C R A (first ++ List.ofFn y.1)
  let s := A'.s
  let aj := C.a / 2 ^ j
  let side := gateSide R s.tree s.edges y.2.1 C.acc aj (gateStop R s.tree s.edges y.2.1 C.acc aj)
  let A₀ := if side = some true then { A' with certs := A'.certs + 1 } else A'
  let A'' := { A₀ with reads := A₀.reads ∪ readingReads C R.F s.tree y }
  if side = some true ∧ C.cert A'.certs s.tree s.edges y.2.2.2 (fun i => R.f (y.2.2.2 i)) = true
  then
    (A'', .inl (.consistent (gateStart R s.tree s.edges y.2.1 C.acc aj)))
  else
    let tests := harvestTests R s.tree s.edges C.k C.L C.f C.c C.θM
    let live := LiveEdge R s.tree s.edges C.k fun _ => False
    let br := y.2.2.1
    let Tr := refusalStop br C.a tests live
    let lv := ((List.finRange C.nr).filter fun i : Fin C.nr =>
      decide ((i : ℕ) < Tr ∧ live (br i))).map br
    if ∃ T ∈ tests, T.fires br C.a Tr then
      (A'', .inl (.harvest (decide ((pairTrip R s.tree s.edges C.k).fires br C.a Tr))))
    else if lv ≠ [] then
      if C.budget s.tree.paths.length ≤ A'.used then (A'', .inl .exhausted) else (A'', .inr lv)
    else (A'', .inl (.halve (decide (side = some false))))

/-- The round from reading `j` with `n` readings left: its end, what it carries, and at each
reading's gate the leaf count and the probes spent. -/
noncomputable def strongRound (C : StrongCfg α) (R : CutReads α) :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws)
      → StrongEnd × RoundAcc α × List (ℕ × ℕ)
  | 0, _, A, _, _ => (.cap, A, [])
  | n + 1, j, A, first, d =>
    match strongReading C R j A first (d 0) with
    | (A', .inl e) => (e, A', [(A'.s.tree.paths.length, A'.used)])
    | (A', .inr lv) =>
      let r := strongRound C R n (j + 1) A' lv (Fin.tail d)
      (r.1, r.2.1, (A'.s.tree.paths.length, A'.used) :: r.2.2)

/-- The probes the round's passes are given, in order. -/
noncomputable def strongProbes (C : StrongCfg α) (R : CutReads α) :
    (n : ℕ) → ℕ → RoundAcc α → List (FreeMonoid α) → (Fin n → C.Draws) → List (FreeMonoid α)
  | 0, _, _, _, _ => []
  | n + 1, j, A, first, d =>
    match strongReading C R j A first (d 0) with
    | (_, .inl _) => first ++ List.ofFn (d 0).1
    | (A', .inr lv) => first ++ List.ofFn (d 0).1 ++ strongProbes C R n (j + 1) A' lv (Fin.tail d)

/-- The leaf `x` reaches when every node reads it on the side `side` gives. -/
def sidePath (side : FreeMonoid α → Bool) : DTree α → FreeMonoid α → List Bool
  | .leaf, _ => []
  | .node m r a, x => if side (x * m) then true :: sidePath side a x else false :: sidePath side r x

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- Every read of `x`'s sift lands on the side `side` gives `z`, down `z`'s path. -/
def FaithfulRoute (R : CutReads α) (side : FreeMonoid α → Bool) (z : FreeMonoid α) :
    DTree α → FreeMonoid α → Prop
  | .leaf, _ => True
  | .node m r a, x => R.cut (x * m) = some (side (z * m))
      ∧ if side (z * m) then FaithfulRoute R side z a x else FaithfulRoute R side z r x

variable {Q : Type*}

end OrthoDFA
