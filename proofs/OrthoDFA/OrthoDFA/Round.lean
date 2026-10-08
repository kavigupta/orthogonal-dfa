import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality

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

/-- The chance, over fresh noise, that read at the middle of the band a string reaching `q` and
its extension by some letter disagree with the hypothesis's edge between them, at worst over the
strings and letters.  A probability, not an `∃` over strings: under string-keyed noise some string
of nearly every state is misread. -/
noncomputable def edgeError (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (H : Hypothesis α) (q : Q) : ℝ :=
  sSup (⋃ c : α, (fun y => μ.real {ω | midPath (readsAt O B F ω) H (y * FreeMonoid.of c)
      ≠ H.step (midPath (readsAt O B F ω) H y) c}) '' {y | A.state y = q})

/-- How badly the round reads the state of `t`: its indecision, or its edge error if larger. -/
noncomputable def badness (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (R : CutReads α) (H : Hypothesis α) (t : FreeMonoid α) : ℝ :=
  max (stateIndecision A O R.B R.F (A.state t)) (edgeError A O R.B R.F H (A.state t))

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
          ∧ ¬ s.halves S.τ}
      ≤ (1 - ε) ^ S.K.patience

/-- Walked from where the middle-of-band reading puts `x`'s first `e` letters, the hypothesis
ends somewhere other than where that reading puts `x`.  At `e = 0` this is `DFAandDTDisagree`. -/
def SuffixDisagree (R : CutReads α) (H : Hypothesis α) (x : FreeMonoid α) (e : ℕ) : Prop :=
  (x.toList.drop e).foldl H.step (midPath R H (prefixOf x e)) ≠ midPath R H x

/-- `AnchoredYield`: the harvest finds a string on at least the share of attempts whose draw, walked
from its anchor, disagrees with the tree.  Every anchor before the first place a draw's walk from
`ε` and its reading part qualifies, so this is at least `E[1{x disagrees} · min(j*(x), L)] / L`. -/
def AnchoredYield : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ),
    R.B.lo ≤ R.B.hi →
    (∑ e ∈ Finset.range L, D.real {x | SuffixDisagree R H x e}) / L
      ≤ (D.prod (anchorLaw L)).real {q | (replay R H q.1 q.2).2 ≠ []}

/-- The hypothesis's walk from where the middle-of-band reading puts `ε` is, after `x`'s first `e`
letters, where that reading puts them. -/
def InSync (R : CutReads α) (H : Hypothesis α) (x : FreeMonoid α) (e : ℕ) : Prop :=
  (x.toList.take e).foldl H.step (midPath R H 1) = midPath R H (prefixOf x e)

/-- `InSyncYield`: the harvest finds a string on at least `1/L` of the expected number of anchors
below `L` at which a draw the DFA/DT check counts against the hypothesis is still in sync. -/
def InSyncYield : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (R : CutReads α) (H : Hypothesis α)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ),
    R.B.lo ≤ R.B.hi →
    (∑ e ∈ Finset.range L, D.real {x | InSync R H x e ∧ DFAandDTDisagree R H x}) / L
      ≤ (D.prod (anchorLaw L)).real {q | (replay R H q.1 q.2).2 ≠ []}

/-- Every target state is read cleanly, undecided at most `uLo`, or badly, at least `uHi`. -/
def GapPremise (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (uLo uHi : ℝ) : Prop :=
  ∀ q, stateIndecision A O B F q ≤ uLo ∨ uHi ≤ stateIndecision A O B F q

/-- How many strings an attempt asks the cut about. -/
noncomputable def queryCount (R : CutReads α) (H : Hypothesis α) (x : FreeMonoid α) (e : ℕ) : ℕ :=
  ((replay R H x e).1.map fun i => (H.tree.route R.cut (prefixOf x i)).1.length).sum

/-- `R̄`: the queries of an attempt on the round's harvest, averaged over the round and the
attempt. -/
noncomputable def meanQueries (S : RoundSetting α μ Q) : ℝ :=
  ∫ θ, (∫ q, (queryCount (readsAt S.O S.B S.F θ.1) (roundEnd S.K S.O S.B S.F S.seed θ).hyp
      q.1 q.2 : ℝ) ∂(S.D.prod (anchorLaw S.L))) ∂(μ.prod (Measure.pi fun _ : Fin S.N => S.D))

/-- `RoundOrHarvest`: a round ends in (1) or (2) but for chance `16 τ L R̄ / ε`, so with
`τ ≤ δ ε / (16 L R̄)` but for `δ`. -/
def RoundOrHarvest : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} [Fintype Q] (S : RoundSetting α μ Q) (ε : ℝ),
    S.Valid → 0 < ε →
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real {θ |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε
          ∧ ¬ (HarvestSpread S.A R s.hyp S.D S.L S.κ ∧ HarvestBad S.A S.O R s.hyp S.D S.L S.τ)}
      ≤ 16 * S.τ * S.L * meanQueries S / ε

end OrthoDFA
