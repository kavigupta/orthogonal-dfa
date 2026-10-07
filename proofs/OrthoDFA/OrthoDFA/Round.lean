import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# The round's trichotomy

A round reads its family through the oracle, runs the counterexample pass on fresh sampler draws,
and either the DFA and tree it ends with agree, or its harvest is spread and heavy on the states
the family reads badly, or the pass halves the indecision limit.

`ν(y)` is how often a uniformly random position of a draw from `D` has the prefix `y`, and `V(q)`
how often it reaches the target state `q`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {Q : Type*}

/-- The family `F` cut at `B`, read through the oracle under noise `ω`. -/
noncomputable def readsAt (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (ω : Ω) : CutReads α :=
  ⟨B, F, fun w => O.mq w ω⟩

/-- `ν(y) = D(y is a prefix of the draw) / L`. -/
noncomputable def prefixWeight (D : Measure (FreeMonoid α)) (L : ℕ) (y : FreeMonoid α) : ℝ :=
  D.real {x | y.toList <+: x.toList} / L

open scoped Classical in
/-- `V(q) = E[the draw's prefixes reaching q] / L`, which is `ν` summed over the strings
reaching `q`. -/
noncomputable def stateWeight (A : DFA (FreeMonoid α) Q) (D : Measure (FreeMonoid α)) (L : ℕ)
    (q : Q) : ℝ :=
  (∫ x, (((Finset.range (x.toList.length + 1)).filter fun i =>
    A.state (prefixOf x i) = q).card : ℝ) ∂D) / L

/-- `u(q)`: the chance the cut leaves a read of a string reaching `q` undecided.  The family's vote
on `s` depends on `s` only through its state, its noise on `s` itself. -/
noncomputable def stateIndecision (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (B : State) (F : Finset (FreeMonoid α)) (q : Q) : ℝ :=
  sSup ((fun s => undecidedProb O B.lo B.hi F s) '' {s | A.state s = q})

/-- Read at the middle of the band, some string reaching `q` is taken by `c` somewhere other than
the hypothesis's edge out of where it sits. -/
def edgeWrong (R : CutReads α) (H : Hypothesis α) (A : DFA (FreeMonoid α) Q) (q : Q) : Prop :=
  ∃ y, A.state y = q ∧ ∃ c : α, midPath R H (y * FreeMonoid.of c) ≠ H.step (midPath R H y) c

open scoped Classical in
/-- How badly the round reads the state of `t`: `1` if an edge out of it is wrong, else `u`. -/
noncomputable def badness (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (R : CutReads α) (H : Hypothesis α) (t : FreeMonoid α) : ℝ :=
  if edgeWrong R H A (A.state t) then 1 else stateIndecision A O R.B R.F (A.state t)

/-- `∑` of `g` over what an attempt harvests, averaged over attempts. -/
noncomputable def harvestMass (R : CutReads α) (H : Hypothesis α) (D : Measure (FreeMonoid α))
    (L : ℕ) (g : FreeMonoid α → ℝ) : ℝ :=
  ∫ q, ((replay R H q.1 q.2).2.map g).sum ∂(D.prod (anchorLaw L))

/-- The harvest is not concentrated: given that an attempt harvests, it takes `t` with chance at
most `(L / d) · κ · ∑_{i ≤ |t|} min(i + 1, L) · V(state of t's first i letters)`, `d` the DFA/DT
disagreement. -/
def HarvestSpread (A : DFA (FreeMonoid α) Q) (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) (L : ℕ) (κ : ℝ) : Prop :=
  ∀ t : FreeMonoid α,
    D.real {x | DFAandDTDisagree R H x} * (D.prod (anchorLaw L)).real {q | t ∈ (replay R H q.1 q.2).2}
      ≤ L * κ * (∑ i ∈ Finset.range (t.toList.length + 1),
          ((min (i + 1) L : ℕ) : ℝ) * stateWeight A D L (A.state (prefixOf t i)))
        * (D.prod (anchorLaw L)).real {q | (replay R H q.1 q.2).2 ≠ []}

/-- The harvest is heavy on badly read states: its average `badness` exceeds `2τ`. -/
def HarvestBad (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (R : CutReads α)
    (H : Hypothesis α) (D : Measure (FreeMonoid α)) (L : ℕ) (τ : ℝ) : Prop :=
  2 * τ * harvestMass R H D L (fun _ => 1) < harvestMass R H D L (badness A O R H)

/-- What a round on noise `ω` and probes `p` ends with. -/
noncomputable def roundEnd (K : StageKnobs α) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (seed : List (FreeMonoid α)) {N : ℕ}
    (θ : Ω × (Fin N → FreeMonoid α)) : PassState α :=
  runPass K (readsAt O B F θ.1) (initialState K (readsAt O B F θ.1) seed) (List.ofFn θ.2)

/-- The round's setting: the target, an oracle for it, the pass's knobs, a family cut at `B`
whose indecision on the table's prefixes averages at most `τ`, and every target state spread
under `ν` with constant `κ`. -/
structure RoundSetting (α : Type*) [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) (Q : Type*) where
  A : DFA (FreeMonoid α) Q
  O : Oracle μ (FreeMonoid α)
  K : StageKnobs α
  B : State
  F : Finset (FreeMonoid α)
  seed : List (FreeMonoid α)
  D : Measure (FreeMonoid α)
  L : ℕ
  N : ℕ
  τ : ℝ
  κ : ℝ

/-- `RoundSetting`'s premises. -/
def RoundSetting.Valid (S : RoundSetting α μ Q) : Prop :=
  S.O.L = {w | S.A.state w ∈ S.A.accept}
    ∧ IsProbabilityMeasure S.D
    ∧ S.B.lo ≤ S.B.hi
    ∧ (S.seed.map fun p => undecidedProb S.O S.B.lo S.B.hi S.F p).sum ≤ S.τ * S.seed.length
    ∧ ∀ y, prefixWeight S.D S.L y ≤ S.κ * stateWeight S.A S.D S.L (S.A.state y)

/-- `RoundTrichotomy`: but for `(1 - ε)^patience`, a round ends with (1) the DFA and tree
disagreeing on at most `ε` of `D`, (2) a harvest spread and heavy on badly read states, or (3) a
pass that halves the indecision limit. -/
def RoundTrichotomy : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} [Fintype Q] (S : RoundSetting α μ Q) (ε : ℝ),
    S.Valid →
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real {θ |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε
          ∧ ¬ (HarvestSpread S.A R s.hyp S.D S.L S.κ ∧ HarvestBad S.A S.O R s.hyp S.D S.L S.τ)
          ∧ ¬ s.halves S.K}
      ≤ (1 - ε) ^ S.K.patience

/-- `HalvingRare`: once `τ` is below `c / (|Q| · (log₂ L + 2))`, a round halves with chance at
most `exp(-patience / 8)`. -/
def HalvingRare (c : ℝ) : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} [Fintype Q] (S : RoundSetting α μ Q),
    S.Valid → S.τ < c / (Fintype.card Q * (Real.logb 2 S.L + 2)) →
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real
        {θ | (roundEnd S.K S.O S.B S.F S.seed θ).halves S.K}
      ≤ Real.exp (-(S.K.patience : ℝ) / 8)

end OrthoDFA
