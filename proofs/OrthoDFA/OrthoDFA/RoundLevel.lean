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

end OrthoDFA
