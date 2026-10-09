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
the certificate by how many calls came before. -/
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
  cert : ℕ → CutReads α → KState α → (Fin nc → FreeMonoid α) → Bool

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

/-- What the round carries: the pass's state, its splits, the probes spent and the certificate's
calls. -/
structure RoundAcc (α : Type*) where
  s : KState α
  splits : List (SplitRec α)
  used : ℕ
  certs : ℕ

/-- The round's start. -/
noncomputable def startAcc (C : StrongCfg α) (R : CutReads α) (seed : List (FreeMonoid α)) :
    RoundAcc α :=
  ⟨initialK C.K R seed, [], 0, 0⟩

/-- One probe: `probeStepK`, a split recorded with the tree it split. -/
noncomputable def strongStep (C : StrongCfg α) (R : CutReads α) (A : RoundAcc α)
    (x : FreeMonoid α) : RoundAcc α :=
  let s := A.s
  let A₁ := { A with s := probeStepK C.K R C.k s x, used := A.used + 1 }
  match probeOutcome R s.tree s.edges C.k x with
  | .edge ps fd =>
    match seedStep C.K R s.tree s.pool s.edges C.k x ps fd with
    | .split d s1 y sp => { A₁ with splits := A.splits ++ [⟨s.tree, s1, d, y, sp⟩] }
    | _ => A₁
  | _ => A₁

/-- `counterexample_pass`: from a fresh quiet streak, probes in order until `patience` in a row
are quiet or the budget is spent. -/
noncomputable def strongPass (C : StrongCfg α) (R : CutReads α) (A : RoundAcc α)
    (probes : List (FreeMonoid α)) : RoundAcc α :=
  probes.foldl (fun A x =>
      if C.K.patience ≤ A.s.streak ∨ C.budget A.s.tree.paths.length ≤ A.used then A
      else strongStep C R A x)
    { A with s := { A.s with streak := 0 } }

/-- How a round ends: consistent at a start, holding a class that fired, halving, with the
budget spent, or out of readings. -/
inductive StrongEnd
  | consistent (q : List Bool)
  | harvest
  | halve (gateRefused : Bool)
  | exhausted
  | cap

open scoped Classical in
/-- Reading `j`: the pass with `first` ahead of the reading's probes; the gate at failure chance
`a·2⁻ʲ`, and on a settled pass the certificate; otherwise a class firing on the refusal sample is
held, else its live-edge draws rerun while the budget lasts, else the limit halves. -/
noncomputable def strongReading (C : StrongCfg α) (R : CutReads α) (j : ℕ) (A : RoundAcc α)
    (first : List (FreeMonoid α)) (y : C.Draws) : RoundAcc α × (StrongEnd ⊕ List (FreeMonoid α)) :=
  let A' := strongPass C R A (first ++ List.ofFn y.1)
  let s := A'.s
  let aj := C.a / 2 ^ j
  let side := gateSide R s.tree s.edges y.2.1 C.acc aj (gateStop R s.tree s.edges y.2.1 C.acc aj)
  let A'' := if side = some true then { A' with certs := A'.certs + 1 } else A'
  if side = some true ∧ C.cert A'.certs R s y.2.2.2 = true then
    (A'', .inl (.consistent (gateStart R s.tree s.edges y.2.1 C.acc aj)))
  else
    let tests := harvestTests R s.tree s.edges C.k C.L C.f C.c C.θM
    let live := LiveEdge R s.tree s.edges C.k fun _ => False
    let br := y.2.2.1
    let Tr := refusalStop br C.a tests live
    let lv := ((List.finRange C.nr).filter fun i : Fin C.nr =>
      decide ((i : ℕ) < Tr ∧ live (br i))).map br
    if ∃ T ∈ tests, T.fires br C.a Tr then (A'', .inl .harvest)
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

/-- The round over `Rmax` readings' draws, from `seed`. -/
noncomputable def strongRun (C : StrongCfg α) (R : CutReads α) (seed : List (FreeMonoid α))
    (Rmax : ℕ) (d : Fin Rmax → C.Draws) : StrongEnd × RoundAcc α × List (ℕ × ℕ) :=
  strongRound C R Rmax 0 (startAcc C R seed) [] d

/-- The leaf `x` reaches when every node reads it on the side `side` gives. -/
def sidePath (side : FreeMonoid α → Bool) : DTree α → FreeMonoid α → List Bool
  | .leaf, _ => []
  | .node m r a, x => if side (x * m) then true :: sidePath side a x else false :: sidePath side r x

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The side the cut more often decides `w` on. -/
noncomputable def majSide (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (w : FreeMonoid α) : Bool :=
  decide (μ.real {ω | (readsAt O B F ω).cut w = some false}
    ≤ μ.real {ω | (readsAt O B F ω).cut w = some true})

/-- A split separates the reference's states at its leaf: two of them go to opposite sides of
the distinguisher. -/
def SplitRec.Separates {Q : Type*} (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (r : SplitRec α) : Prop :=
  ∃ q q', sidePath side r.tree (rep q) = r.leaf ∧ sidePath side r.tree (rep q') = r.leaf
    ∧ side (rep q * r.d) ≠ side (rep q' * r.d)

/-- The read of `x·m` lands on the side `side` gives its state's representative. -/
def Faithful {Q : Type*} (R : CutReads α) (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool)
    (rep : Q → FreeMonoid α) (x m : FreeMonoid α) : Prop :=
  R.cut (x * m) = some (side (rep (A.state x) * m))

/-- Every read of `x`'s sift lands on the side `side` gives `z`, down `z`'s path. -/
def FaithfulRoute (R : CutReads α) (side : FreeMonoid α → Bool) (z : FreeMonoid α) :
    DTree α → FreeMonoid α → Prop
  | .leaf, _ => True
  | .node m r a, x => R.cut (x * m) = some (side (z * m))
      ∧ if side (z * m) then FaithfulRoute R side z a x else FaithfulRoute R side z r x

/-- One of the split's reads lands off its state's side: sifting its witness or its probe's
prefix to the leaf, or parting them. -/
def SplitRec.Noisy {Q : Type*} (R : CutReads α) (A : DFA (FreeMonoid α) Q)
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (r : SplitRec α) : Prop :=
  ¬ FaithfulRoute R side (rep (A.state r.y)) r.tree r.y
    ∨ ¬ FaithfulRoute R side (rep (A.state r.sprime)) r.tree r.sprime
    ∨ ¬ Faithful R A side rep r.y r.d ∨ ¬ Faithful R A side rep r.sprime r.d

open scoped Classical in
/-- The noisy splits. -/
noncomputable def noisySplits {Q : Type*} (R : CutReads α) (A : DFA (FreeMonoid α) Q)
    (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (rs : List (SplitRec α)) : ℕ :=
  (rs.filter fun r => r.Noisy R A side rep).length

/-- Every learned edge joins two leaves. -/
def EdgesOnLeaves (s : KState α) : Prop :=
  ∀ p c q w, s.edges p c = some (q, w) → p ∈ s.tree.paths ∧ q ∈ s.tree.paths

/-- `RoundStrongReadings`: where each pass is given at least `patience` probes, the readings times
`patience` are at most the budget at the round's last leaf count and one more `patience`. -/
def RoundStrongReadings : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    C.K.patience ≤ C.np → Monotone C.budget →
    (strongRun C R seed Rmax d).2.2.length * C.K.patience
      ≤ C.budget (strongRun C R seed Rmax d).2.1.s.tree.paths.length + C.K.patience

/-- `RoundStrongLeaves`: placing each state `q` where every node reads `rep q` on the side
`side` gives, the round ends with at most `|Q| + 2` leaves and its noisy splits. -/
def RoundStrongLeaves : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} [Fintype Q] (C : StrongCfg α)
    (R : CutReads α) (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool)
    (rep : Q → FreeMonoid α) (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    (strongRun C R seed Rmax d).2.1.s.tree.paths.length
      ≤ Fintype.card Q + 2 + noisySplits R A side rep (strongRun C R seed Rmax d).2.1.splits

/-- `RoundStrongSameState`: a split between two strings of one state has one of its two parting
reads off their state's side. -/
def RoundStrongSameState : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} (C : StrongCfg α) (R : CutReads α)
    (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    ∀ r ∈ (strongRun C R seed Rmax d).2.1.splits, A.state r.y = A.state r.sprime →
      ¬ Faithful R A side rep r.y r.d ∨ ¬ Faithful R A side rep r.sprime r.d

/-- `RoundStrongLeafPaths`: the round's learned edges join leaves. -/
def RoundStrongLeafPaths : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    EdgesOnLeaves (strongRun C R seed Rmax d).2.1.s

/-- `RoundStrongNoStop`: at the hypothesis the round ends with, an attempt on the edge any draw's
search ends at splits or adds a member; it never stops at a read it cannot place. -/
def RoundStrongNoStop : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws) (x : FreeMonoid α)
    (ps : List (List Bool)) (fd : ℕ),
    let s := (strongRun C R seed Rmax d).2.1.s
    probeOutcome R s.tree s.edges C.k x = .edge ps fd →
      (∃ dd s1 y sp, seedStep C.K R s.tree s.pool s.edges C.k x ps fd = .split dd s1 y sp)
        ∨ ∃ s1 sp, seedStep C.K R s.tree s.pool s.edges C.k x ps fd = .member s1 sp

variable {Q : Type*}

open scoped Classical in
/-- What each end claims: consistent, the start agreeing on at least `acc` and certified; a
halving, as `RoundEndHolds` with no edge given up, every covering start's residue claimed only
where the gate settled below; exhausted, nothing; out of readings, `patience` times the readings
within the budget. -/
def StrongEndHolds (C : StrongCfg α) (R : CutReads α) (A : DFA (FreeMonoid α) Q)
    (D : Measure (FreeMonoid α)) (CertGood : KState α → Prop) (η minCov ν : ℝ) (Rmax : ℕ)
    (Ac : RoundAcc α) : StrongEnd → Prop
  | .consistent q => D.real {x | StartDis R Ac.s.edges q x} ≤ 1 - C.acc ∧ CertGood Ac.s
  | .harvest => True
  | .halve gr =>
    let s := Ac.s
    let gu : List Bool × α → Prop := fun _ => False
    tauZero s.tree C.k C.L C.nr C.c C.a ≤ C.f ∨ 1 - C.a ≤ C.θM * C.nr
      ∨ (D.real {x | NAOff R s.tree s.edges C.k gu x} ≤ ν
        ∧ (gr = true → ∀ (q₀ : Q) (h : Q → List Bool), q₀ ∈ Covered A D C.k C.L minCov →
          h q₀ ∈ s.tree.paths → 1 - η ≤ D.real (CoverGood A (Covered A D C.k C.L minCov) q₀) →
          (∀ q ∈ Covered A D C.k C.L minCov, leafAccepts (h q) = decide (q ∈ A.accept)) →
          need R A D (Covered A D C.k C.L minCov) h s.tree s.edges C.k q₀ gu C.acc η ≤ ν))
  | .exhausted => True
  | .cap => Rmax * C.K.patience ≤ C.budget Ac.s.tree.paths.length

/-- `RoundStrongTrichotomy`: for any reads, over `Rmax` readings' draws, the round's end breaks
its claim with chance at most the gate's spent failure chances, the refusal sample's miss per
reading made in expectation, and the certificate's failure chances over its calls in
expectation. Ending exhausted claims nothing. -/
def RoundStrongTrichotomy : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} (C : StrongCfg α)
    (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (seed : List (FreeMonoid α)) (CertGood : KState α → Prop) (αs : ℕ → ℝ) (η minCov ν : ℝ)
    (Rmax : ℕ),
    0 ≤ C.acc → C.acc ≤ 1 → 0 ≤ C.f → 0 ≤ C.a → ν ≤ 1 →
    C.K.patience ≤ C.np → Monotone C.budget →
    (∀ i R s, (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert i R s cs = true ∧ ¬ CertGood s} ≤ αs i) →
    ∀ R : CutReads α,
      (Measure.pi fun _ : Fin Rmax => C.drawMeasure D).real
          {d | ¬ StrongEndHolds C R A D CertGood η minCov ν Rmax (strongRun C R seed Rmax d).2.1
            (strongRun C R seed Rmax d).1}
        ≤ 4 * (Nat.log 2 (C.ng / 30) + 2) * C.a
          + (∫ d, ((strongRun C R seed Rmax d).2.2.length : ℝ)
              ∂(Measure.pi fun _ : Fin Rmax => C.drawMeasure D)) * (1 - ν) ^ C.nr
          + ∫ d, ∑ i ∈ Finset.range (strongRun C R seed Rmax d).2.1.certs, αs i
              ∂(Measure.pi fun _ : Fin Rmax => C.drawMeasure D)

/-- The fluctuation allowed a round that made `r` readings, at total failure chance `δ`. -/
noncomputable def strongEps (C : StrongCfg α) (D : Measure (FreeMonoid α)) (δ : ℝ) (r : ℕ) : ℝ :=
  Real.sqrt (prefixMax D C.k * ((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2
    + Real.log (5 / δ)) / 2)

/-- `RoundStrongQuality`: `RoundQualityLevel` for this round. -/
def RoundStrongQuality : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (C : StrongCfg α) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (seed : List (FreeMonoid α)) (δ : ℝ)
    (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    0 ≤ C.f → 0 ≤ C.c → 0 < δ → δ ≤ 1 → C.k ≤ C.L → (∀ᵐ x ∂D, x.toList.length = C.L) →
    SuffixFree (F ∪ C.K.train F) →
    ∃ E : Set Ω, μ.real E ≤ δ ∧ ∀ ω ∉ E,
      let R := readsAt O B F ω
      let r := strongRun C R seed Rmax d
      QualityHolds R A O B F r.2.1.s D C.k C.L seed
        (strongProbes C R Rmax 0 (startAcc C R seed) [] d) C.f C.c (strongEps C D δ r.2.2.length)

end OrthoDFA
