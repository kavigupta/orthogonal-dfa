import OrthoDFA.RoundLevel

/-!
# The round, with its edges' attempts counted

The round of `RoundLevel` with what Python carries across its readings: every end of the split
test but a split counts against its edge, an edge with `mmax` such ends is given up, the
strings that stop the test's guards are held, each pass starts its quiet streak afresh, and the
probes spent are counted against a budget set by the leaves.

A gate is a pass only where its test settles above `acc`; one that does not settle is refused,
and the certificate's failure chance is spent per call.

`RoundStrongBudget`: the budget is never reached, so the round never ends exhausted.
`RoundStrongReadings`: a round makes at most `readStar` of its last leaf count readings.
`RoundStrongLeaves`: for any reference placement of the target's states, the leaves are at most
`|Q| + 2` and the splits that do not separate the reference's states.
`RoundStrongLeafPaths`: the learned edges join leaves.
`RoundStrongTrichotomy` and `RoundStrongQuality`: `RoundTrichotomyLevel` and
`RoundQualityLevel` for this round.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A round's settings: as `RoundCfg`, with the ends of the split test after which an edge is
given up by the leaf count, the fewest probes the budget allows, and the certificate by how many
calls came before. -/
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
  mmax : ℕ → ℕ
  minProbes : ℕ
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

/-- The readings a round with `n` leaves can have made. -/
def readStar (C : StrongCfg α) (n : ℕ) : ℕ :=
  n - 1 + C.mmax n * Fintype.card α * (2 * (n - 1) + 1)

/-- `probe_budget`. -/
def budgetOf (C : StrongCfg α) (n : ℕ) : ℕ :=
  max C.minProbes (readStar C n * (C.K.patience + C.nr) + C.K.patience * (n - 2))

/-- A split: the tree it split, the leaf, the distinguisher, the edge's witness and the probe's
prefix. -/
structure SplitRec (α : Type*) where
  tree : DTree α
  leaf : List Bool
  d : FreeMonoid α
  y : FreeMonoid α
  sprime : FreeMonoid α

/-- What the round carries: the pass's state, each edge's ends of the split test other than a
split, the strings that stopped its guards, its splits, the probes spent and the certificate's
calls. -/
structure RoundAcc (α : Type*) where
  s : KState α
  att : List Bool × α → ℕ
  stopped : List (FreeMonoid α)
  splits : List (SplitRec α)
  used : ℕ
  certs : ℕ

/-- The round's start. -/
noncomputable def startAcc (C : StrongCfg α) (R : CutReads α) (seed : List (FreeMonoid α)) :
    RoundAcc α :=
  ⟨initialK C.K R seed, fun _ => 0, [], [], 0, 0⟩

/-- `given_up`. -/
def givenUp (C : StrongCfg α) (A : RoundAcc α) (e : List Bool × α) : Prop :=
  C.mmax A.s.tree.paths.length ≤ A.att e

instance (C : StrongCfg α) (A : RoundAcc α) (e : List Bool × α) : Decidable (givenUp C A e) :=
  inferInstanceAs (Decidable (_ ≤ _))

/-- The string that stopped the split test's guards, if any. -/
def SeedResult.stoppedAt : SeedResult α → List (FreeMonoid α)
  | .stopped b => [b]
  | _ => []

/-- One probe: `probeStepK`, with each end of the split test but a split counted against its
edge; a member added on a given-up edge leaves the quiet streak running. -/
noncomputable def strongStep (C : StrongCfg α) (R : CutReads α) (A : RoundAcc α)
    (x : FreeMonoid α) : RoundAcc α :=
  let s := A.s
  let s' := probeStepK C.K R C.k s x
  let A₁ := { A with s := s', used := A.used + 1 }
  match probeOutcome R s.tree s.edges C.k x with
  | .edge ps fd =>
    match seedStep C.K R s.tree s.pool s.edges C.k x ps fd, edgeAt R s.tree s.edges C.k x with
    | .split d s1 y sp, _ => { A₁ with splits := A.splits ++ [⟨s.tree, s1, d, y, sp⟩] }
    | r, some e =>
      let A₂ : RoundAcc α :=
        { A₁ with
          att := Function.update A.att e (A.att e + 1)
          stopped := A.stopped ++ r.stoppedAt }
      if givenUp C A e then { A₂ with s := { s' with streak := s.streak + 1 } } else A₂
    | _, none => A₁
  | _ => A₁

/-- `counterexample_pass`: from a fresh quiet streak, probes in order until `patience` in a row
are quiet or the budget is spent. -/
noncomputable def strongPass (C : StrongCfg α) (R : CutReads α) (A : RoundAcc α)
    (probes : List (FreeMonoid α)) : RoundAcc α :=
  probes.foldl (fun A x =>
      if C.K.patience ≤ A.s.streak ∨ budgetOf C A.s.tree.paths.length ≤ A.used then A
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
`a·2⁻ʲ`, and on a settled pass the certificate; otherwise the refusal sample's live-edge draws
rerun while the budget lasts, else a fired class is held, else the limit halves. -/
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
    let live := LiveEdge R s.tree s.edges C.k (givenUp C A')
    let br := y.2.2.1
    let Tr := refusalStop br C.a tests live
    let lv := ((List.finRange C.nr).filter fun i : Fin C.nr =>
      decide ((i : ℕ) < Tr ∧ live (br i))).map br
    if lv ≠ [] then
      if budgetOf C s.tree.paths.length ≤ A'.used then (A'', .inl .exhausted) else (A'', .inr lv)
    else if ∃ T ∈ tests, T.fires br C.a Tr then (A'', .inl .harvest)
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
def majPath (side : FreeMonoid α → Bool) : DTree α → FreeMonoid α → List Bool
  | .leaf, _ => []
  | .node m r a, x => if side (x * m) then true :: majPath side a x else false :: majPath side r x

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
  ∃ q q', majPath side r.tree (rep q) = r.leaf ∧ majPath side r.tree (rep q') = r.leaf
    ∧ side (rep q * r.d) ≠ side (rep q' * r.d)

open scoped Classical in
/-- The splits that do not separate the reference's states. -/
noncomputable def badSplits {Q : Type*} (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (rs : List (SplitRec α)) : ℕ :=
  (rs.filter fun r => ¬ r.Separates side rep).length

/-- Every learned edge joins two leaves. -/
def EdgesOnLeaves (s : KState α) : Prop :=
  ∀ p c q w, s.edges p c = some (q, w) → p ∈ s.tree.paths ∧ q ∈ s.tree.paths

/-- `RoundStrongBudget`: at every reading's gate the probes spent are below the budget, so no
pass meets it and the round never ends exhausted. -/
def RoundStrongBudget : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    1 ≤ C.K.patience → C.K.patience ≤ C.nr → Monotone C.mmax →
    (strongRun C R seed Rmax d).1 ≠ .exhausted
      ∧ ∀ p ∈ (strongRun C R seed Rmax d).2.2, p.2 < budgetOf C p.1

/-- `RoundStrongReadings`: a round makes at most `readStar` of its last leaf count readings. -/
def RoundStrongReadings : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    1 ≤ C.K.patience → C.K.patience ≤ C.nr → Monotone C.mmax →
    (strongRun C R seed Rmax d).2.2.length
      ≤ readStar C (strongRun C R seed Rmax d).2.1.s.tree.paths.length

/-- `RoundStrongLeaves`: placing each state `q` where every node reads `rep q` on the side
`side` gives, the round ends with at most `|Q| + 2` leaves and its splits that separate no two
states. -/
def RoundStrongLeaves : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} [Fintype Q] (C : StrongCfg α)
    (R : CutReads α) (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    (strongRun C R seed Rmax d).2.1.s.tree.paths.length
      ≤ Fintype.card Q + 2 + badSplits side rep (strongRun C R seed Rmax d).2.1.splits

/-- `RoundStrongLeafPaths`: the round's learned edges join leaves. -/
def RoundStrongLeafPaths : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (C : StrongCfg α) (R : CutReads α)
    (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    EdgesOnLeaves (strongRun C R seed Rmax d).2.1.s

variable {Q : Type*}

open scoped Classical in
/-- What each end claims: consistent, the start agreeing on at least `acc` and certified; a
halving, as `RoundEndHolds` with the edges given up, every covering start's residue claimed only
where the gate settled below; never exhausted; out of readings only past `readStar`. -/
def StrongEndHolds (C : StrongCfg α) (R : CutReads α) (A : DFA (FreeMonoid α) Q)
    (D : Measure (FreeMonoid α)) (CertGood : KState α → Prop) (η minCov ν : ℝ) (Rmax : ℕ)
    (Ac : RoundAcc α) : StrongEnd → Prop
  | .consistent q => D.real {x | StartDis R Ac.s.edges q x} ≤ 1 - C.acc ∧ CertGood Ac.s
  | .harvest => True
  | .halve gr =>
    let s := Ac.s
    let gu := givenUp C Ac
    tauZero s.tree C.k C.L C.nr C.c C.a ≤ C.f ∨ 1 - C.a ≤ C.θM * C.nr
      ∨ (D.real {x | NAOff R s.tree s.edges C.k gu x} ≤ ν
        ∧ (gr = true → ∀ (q₀ : Q) (h : Q → List Bool), q₀ ∈ Covered A D C.k C.L minCov →
          h q₀ ∈ s.tree.paths → 1 - η ≤ D.real (CoverGood A (Covered A D C.k C.L minCov) q₀) →
          (∀ q ∈ Covered A D C.k C.L minCov, leafAccepts (h q) = decide (q ∈ A.accept)) →
          need R A D (Covered A D C.k C.L minCov) h s.tree s.edges C.k q₀ gu C.acc η ≤ ν))
  | .exhausted => False
  | .cap => Rmax ≤ readStar C Ac.s.tree.paths.length

/-- `RoundStrongTrichotomy`: for any reads, over `Rmax` readings' draws, the round's end breaks
its claim with chance at most the gate's spent failure chances, the refusal sample's miss per
reading made in expectation, and the certificate's failure chances over its calls in
expectation. -/
def RoundStrongTrichotomy : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} (C : StrongCfg α)
    (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (seed : List (FreeMonoid α)) (CertGood : KState α → Prop) (αs : ℕ → ℝ) (η minCov ν : ℝ)
    (Rmax : ℕ),
    0 ≤ C.acc → C.acc ≤ 1 → 0 ≤ C.f → 0 ≤ C.a → ν ≤ 1 →
    1 ≤ C.K.patience → C.K.patience ≤ C.nr → Monotone C.mmax →
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
