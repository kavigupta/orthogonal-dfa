import OrthoDFA.Trichotomy

/-!
# The round's trichotomy

A round reads until a reading ends it: each reading runs the pass on the previous hypothesis with
the previous reading's live-edge draws first, then the gate and, where the gate's test settles
above, the certificate; anything else reads a refusal sample, which holds a class that fired,
or else reruns live edges, or else halves.

`RoundTrichotomyLevel`: for any reads, the round ends consistent with the certificate's guarantee,
holding a class that fired, halving only above `τ₀` or with the walk's non-agreeing mass at most
`ν`, or with its budget exhausted, but for the gate's failure chances spent `a·2⁻ʲ` over the
readings and, per reading made, the refusal sample's miss and the certificate's failure.

The budget exit is a problematic component: it stays because give-ups on genuinely wrong edges and
attempts stopped at reads the cut cannot place are not bounded.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A round's settings: the pass's knobs, the start `k`, the draws' length, the sizes of each
reading's probes, gate batch, refusal sample and certificate sample, the limit, the classes'
constants, the gate's accuracy and failure chance, the most leaves, the certificate, and which
edges the round has given up given the readings so far. -/
structure RoundCfg (α : Type*) where
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
  Pmax : ℕ
  cert : CutReads α → KState α → (Fin nc → FreeMonoid α) → Bool
  gu : CutReads α → List ((Fin np → FreeMonoid α) × (Fin ng → FreeMonoid α)
    × (Fin nr → FreeMonoid α) × (Fin nc → FreeMonoid α)) → List Bool × α → Prop

/-- One reading's draws: probes, gate batch, refusal sample, certificate sample. -/
abbrev RoundCfg.Draws (C : RoundCfg α) : Type _ :=
  (Fin C.np → FreeMonoid α) × (Fin C.ng → FreeMonoid α) × (Fin C.nr → FreeMonoid α)
    × (Fin C.nc → FreeMonoid α)

/-- One reading's draws, each from `D`. -/
noncomputable def RoundCfg.drawMeasure (C : RoundCfg α) (D : Measure (FreeMonoid α)) :
    Measure C.Draws :=
  (Measure.pi fun _ : Fin C.np => D).prod ((Measure.pi fun _ : Fin C.ng => D).prod
    ((Measure.pi fun _ : Fin C.nr => D).prod (Measure.pi fun _ : Fin C.nc => D)))

/-- How a round ends. -/
inductive LevelEnd (α : Type*)
  | consistent (s : KState α) (q : List Bool)
  | harvest (s : KState α) (halve : Bool)
  | halve (s : KState α) (gu : List Bool × α → Prop) (gateRefused : Bool)
  | exhausted (s : KState α)

/-- The hypothesis a round ends with. -/
def LevelEnd.state : LevelEnd α → KState α
  | .consistent s _ => s
  | .harvest s _ => s
  | .halve s _ _ => s
  | .exhausted s => s

/-- How a reading ends: the round, or a rerun with these first probes. -/
inductive ReadStep (α : Type*)
  | done (e : LevelEnd α)
  | rerun (s : KState α) (first : List (FreeMonoid α))

open scoped Classical in
/-- Reading `j`: the pass from `s₀` with `first` ahead of the reading's probes; past `Pmax`
leaves the budget is out; the gate at failure chance `a·2⁻ʲ`, and on a settled pass the
certificate; otherwise a class firing on the refusal sample is held, halving where pairs were
over half, else its live-edge draws rerun, else the limit halves. -/
noncomputable def readingStep (C : RoundCfg α) (R : CutReads α) (hist : List C.Draws) (j : ℕ)
    (s₀ : KState α) (first : List (FreeMonoid α)) (y : C.Draws) : ReadStep α :=
  let s := runPassK C.K R C.k s₀ (first ++ List.ofFn y.1)
  let aj := C.a / 2 ^ j
  if C.Pmax < s.tree.paths.length then .done (.exhausted s)
  else if gateSide R s.tree s.edges y.2.1 C.acc aj (gateStop R s.tree s.edges y.2.1 C.acc aj)
      = some true ∧ C.cert R s y.2.2.2 = true then
    .done (.consistent s (gateStart R s.tree s.edges y.2.1 C.acc aj))
  else
    let gu := C.gu R hist
    let tests := harvestTests R s.tree s.edges C.k C.L C.f C.c C.θM
    let live := LiveEdge R s.tree s.edges C.k gu
    let br := y.2.2.1
    let Tr := refusalStop br C.a tests live
    let lv := ((List.finRange C.nr).filter fun i : Fin C.nr =>
      decide ((i : ℕ) < Tr ∧ live (br i))).map br
    if ∃ T ∈ tests, T.fires br C.a Tr then
      .done (.harvest s (decide ((pairTrip R s.tree s.edges C.k).fires br C.a Tr)))
    else if lv ≠ [] then .rerun s lv
    else .done (.halve s gu (decide (gateSide R s.tree s.edges y.2.1 C.acc aj
      (gateStop R s.tree s.edges y.2.1 C.acc aj) = some false)))

/-- The round from reading `j` with `n` readings left, and how many it makes. -/
noncomputable def roundAux (C : RoundCfg α) (R : CutReads α) :
    (n : ℕ) → ℕ → List C.Draws → KState α → List (FreeMonoid α) → (Fin n → C.Draws)
      → LevelEnd α × ℕ
  | 0, _, _, s, _, _ => (.exhausted s, 0)
  | n + 1, j, hist, s, first, d =>
    match readingStep C R hist j s first (d 0) with
    | .done e => (e, 1)
    | .rerun s' first' =>
      ((roundAux C R n (j + 1) (hist ++ [d 0]) s' first' (Fin.tail d)).1,
        (roundAux C R n (j + 1) (hist ++ [d 0]) s' first' (Fin.tail d)).2 + 1)

/-- The probes the round's passes take, in order. -/
noncomputable def roundProbes (C : RoundCfg α) (R : CutReads α) :
    (n : ℕ) → ℕ → List C.Draws → KState α → List (FreeMonoid α) → (Fin n → C.Draws)
      → List (FreeMonoid α)
  | 0, _, _, _, _, _ => []
  | n + 1, j, hist, s, first, d =>
    match readingStep C R hist j s first (d 0) with
    | .done _ => first ++ List.ofFn (d 0).1
    | .rerun s' first' =>
      first ++ List.ofFn (d 0).1 ++ roundProbes C R n (j + 1) (hist ++ [d 0]) s' first' (Fin.tail d)

variable {Q : Type*}

open scoped Classical in
/-- What each end claims: consistent, its start agreeing on at least `acc` and certified; a
halving, above `τ₀`, or with members not firing on a first hit, or with the walk's non-agreeing
mass at most `ν` and, where the gate settled below, every covering start leaving at most `ν`. -/
def RoundEndHolds (C : RoundCfg α) (R : CutReads α) (A : DFA (FreeMonoid α) Q)
    (D : Measure (FreeMonoid α)) (CertGood : KState α → Prop) (η minCov ν : ℝ) :
    LevelEnd α → Prop
  | .consistent s q => D.real {x | StartDis R s.edges q x} ≤ 1 - C.acc ∧ CertGood s
  | .harvest _ _ => True
  | .halve s gu gr =>
    tauZero s.tree C.k C.L C.nr C.c C.a ≤ C.f ∨ 1 - C.a ≤ C.θM * C.nr
      ∨ (D.real {x | NAOff R s.tree s.edges C.k gu x} ≤ ν
        ∧ (gr = true → ∀ (q₀ : Q) (h : Q → List Bool), q₀ ∈ Covered A D C.k C.L minCov →
          h q₀ ∈ s.tree.paths → 1 - η ≤ D.real (CoverGood A (Covered A D C.k C.L minCov) q₀) →
          (∀ q ∈ Covered A D C.k C.L minCov, leafAccepts (h q) = decide (q ∈ A.accept)) →
          need R A D (Covered A D C.k C.L minCov) h s.tree s.edges C.k q₀ gu C.acc η ≤ ν))
  | .exhausted _ => True

/-- `RoundTrichotomyLevel`: for any reads, over `Rmax` readings' draws, the round's end breaks
its claim with chance at most the gate's spent failure chances, plus, per reading made in
expectation, the refusal sample's miss and the certificate's failure. -/
def RoundTrichotomyLevel : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*} (C : RoundCfg α)
    (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (seed : List (FreeMonoid α)) (CertGood : KState α → Prop) (αc η minCov ν : ℝ)
    (Rmax : ℕ),
    0 ≤ C.acc → C.acc ≤ 1 → 0 ≤ C.f → 0 ≤ C.a → ν ≤ 1 →
    (∀ R s, (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert R s cs = true ∧ ¬ CertGood s} ≤ αc) →
    ∀ R : CutReads α,
      (Measure.pi fun _ : Fin Rmax => C.drawMeasure D).real
          {d | ¬ RoundEndHolds C R A D CertGood η minCov ν
            (roundAux C R Rmax 0 [] (initialK C.K R seed) [] d).1}
        ≤ 4 * (Nat.log 2 (C.ng / 30) + 2) * C.a
          + (∫ d, ((roundAux C R Rmax 0 [] (initialK C.K R seed) [] d).2 : ℝ)
              ∂(Measure.pi fun _ : Fin Rmax => C.drawMeasure D))
            * ((1 - ν) ^ C.nr + αc)

/-- The fluctuation allowed a round that made `r` readings, at total failure chance `δ`. -/
noncomputable def roundEps (C : RoundCfg α) (D : Measure (FreeMonoid α)) (δ : ℝ) (r : ℕ) : ℝ :=
  Real.sqrt (prefixMax D C.k * ((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2
    + Real.log (5 / δ)) / 2)

/-- `RoundQualityLevel`: for any draws, outside a set of the oracle's noise of measure at most
`δ`, the hypothesis the round ends with meets `QualityHolds` against everything its passes probed,
at the fluctuation `roundEps` of the number of readings it made. The set covers every choice of
live-edge draws the readings could rerun. -/
def RoundQualityLevel : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} (C : RoundCfg α) (A : DFA (FreeMonoid α) Q)
    (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (seed : List (FreeMonoid α)) (δ : ℝ)
    (Rmax : ℕ) (d : Fin Rmax → C.Draws),
    0 ≤ C.f → 0 ≤ C.c → 0 < δ → δ ≤ 1 → C.k ≤ C.L → (∀ᵐ x ∂D, x.toList.length = C.L) →
    SuffixFree (C.K.suffixes F) →
    ∃ E : Set Ω, μ.real E ≤ δ ∧ ∀ ω ∉ E,
      let R := readsAt O B F ω
      let r := roundAux C R Rmax 0 [] (initialK C.K R seed) [] d
      QualityHolds R A O B F r.1.state D C.k C.L seed
        (roundProbes C R Rmax 0 [] (initialK C.K R seed) [] d) C.f C.c (roundEps C D δ r.2)

end OrthoDFA
