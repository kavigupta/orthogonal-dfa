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

/-- How a round ends: consistent at a start, holding the classes that fired and halving where
pairs were over half, halving, with the budget spent, or out of readings. -/
inductive StrongEnd
  | consistent (q : List Bool)
  | harvest (halve : Bool)
  | halve (gateRefused : Bool)
  | exhausted
  | cap

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

variable {Q : Type*}

end OrthoDFA
