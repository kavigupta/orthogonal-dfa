import OrthoDFA.Pass
import OrthoDFA.ClusteringQuality

/-!
# The round's tetrachotomy

A round reads its family through the oracle, runs the counterexample pass on fresh sampler draws,
and either the DFA and tree it ends with agree, or it leaves a population the next gate must act
on, or the pass halves the indecision limit, or some chain's odds of reading a state badly or an
edge wrongly grow by a fixed factor.

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

/-- What a per-state population of leaf `s` draws: a draw from `D` that the hypothesis walks to
`s` and the tree places at `s`, as `StateSource` keeps an aim only where it rests. -/
def settlesAt (R : CutReads α) (H : Hypothesis α) (s : List Bool) : Set (FreeMonoid α) :=
  {x | x.toList.foldl H.step (midPath R H 1) = s ∧ H.tree.sift R.cut x = .inl s}

/-- (2a) Some population the round makes reads, on average over its strings, more than `2τ`
undecided under the current family, so a family held to `τ` on it reads its states differently:
the harvest (`Walked` and `Sifted` provenances alike, mixed as they found strings), or a per-state
population. -/
def PopulationIndecisive (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (R : CutReads α) (H : Hypothesis α) (D : Measure (FreeMonoid α)) (L : ℕ) (τ : ℝ) : Prop :=
  2 * τ * harvestMass R H D L (fun _ => 1)
      < harvestMass R H D L (fun t => stateIndecision A O R.B R.F (A.state t))
    ∨ ∃ s ∈ H.tree.paths, 2 * τ
      < ∫ x, stateIndecision A O R.B R.F (A.state x) ∂(D[|settlesAt R H s])

open scoped Classical in
/-- (2b) At some leaf, the harvest's disagreement prefixes whose every successor state is read
cleanly make up at least `σ` of what the round's populations place there, `nH` harvest strings
and `nS` per-state ones: enough that the split evidence cannot call that leaf one state.  A
harvested string the tree places is a disagreement prefix; one whose successor is read badly is
left out, since splitting its leaf does not mend its successor. -/
def WrongEdgeHarvest (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (R : CutReads α)
    (H : Hypothesis α) (D : Measure (FreeMonoid α)) (L nH nS : ℕ) (σ uLo : ℝ) : Prop :=
  ∃ s ∈ H.tree.paths,
    let atS := harvestMass R H D L (fun t => if H.tree.sift R.cut t = .inl s then 1 else 0)
    let wrongAtS := harvestMass R H D L (fun t =>
      if H.tree.sift R.cut t = .inl s
          ∧ ∀ c : α, stateIndecision A O R.B R.F (A.state (t * FreeMonoid.of c)) ≤ uLo
      then 1 else 0)
    σ * (nS * harvestMass R H D L (fun _ => 1) + nH * atS) ≤ nH * wrongAtS

/-- An edge population of #408: draws of `X` extended by `c` that the cut leaves undecided.  Over
fresh noise a string joins it with chance `u` of its state, so its average indecision is
`∫ u² / ∫ u`; it is indecisive past `2τ` when that exceeds `2τ`. -/
def EdgePopulationIndecisive (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (R : CutReads α) (X : Measure (FreeMonoid α)) (c : α) (τ : ℝ) : Prop :=
  2 * τ * ∫ x, stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c)) ∂X
    < ∫ x, stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c)) ^ 2 ∂X

/-- #408's rule, with the pass's baseline replaced by the clean states' bound `a`: the edge of
per-state population `s` by `c` is held apart when its draws extended by `c` come out undecided
more than `f` times as often as a string read cleanly at every node, `r` nodes deep, would. -/
def EdgeSelected (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (R : CutReads α)
    (H : Hypothesis α) (D : Measure (FreeMonoid α)) (s : List Bool) (c : α) (f a : ℝ) (r : ℕ) :
    Prop :=
  f * a * r < ∫ x, stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c)) ∂(D[|settlesAt R H s])

/-- The `ν` root: a draw from `D` cut at a position drawn uniformly below `L`, so a string `y`
comes out with chance `ν(y)` (up to the cut `L`).  Every round tests its edges alongside the
per-state populations'. -/
noncomputable def nuRoot (D : Measure (FreeMonoid α)) (L : ℕ) : Measure (FreeMonoid α) :=
  (D.prod (anchorLaw L)).map fun q => prefixOf q.1 q.2

/-- The chance, over fresh noise, that the middle reading of `x·c` lands off the hypothesis's edge
out of where the middle reading puts `x`. -/
noncomputable def edgeDisagreeProb (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (H : Hypothesis α) (x : FreeMonoid α) (c : α) : ℝ :=
  μ.real {ω | midPath (readsAt O B F ω) H (x * FreeMonoid.of c)
    ≠ H.step (midPath (readsAt O B F ω) H x) c}

/-- A chain on source `X`, filtered by a link that keeps `x` with chance `g x`, has its odds of
`g ≥ hi` multiplied by at least `β`: `X` holds strings with `g ≥ hi`, and the link multiplies their
mass by at least `hi` and the rest's by at most `hi / β`. -/
def ChainAdvancesBy (X : Measure (FreeMonoid α)) (g : FreeMonoid α → ℝ) (β hi : ℝ) : Prop :=
  0 < X.real {x | hi ≤ g x}
    ∧ hi * X.real {x | hi ≤ g x}
        ≤ (X.withDensity fun x => ENNReal.ofReal (g x)).real {x | hi ≤ g x}
    ∧ (X.withDensity fun x => ENNReal.ofReal (g x)).real {x | hi ≤ g x}ᶜ
        ≤ hi / β * X.real {x | hi ≤ g x}ᶜ

/-- (4) Some chain's odds multiply by at least `β`: on a source `X` (a live chain's law, or a
per-state population's) and letter `c`, filtered by `x·c`'s indecision with `hi` a rung
`uHi / β^i` of the ladder, or by `x·c`'s disagreement with the hypothesis's edge with `hi = wHi`. -/
def ChainAdvances [Fintype Q] (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α))
    (R : CutReads α) (H : Hypothesis α) (sources : List (Measure (FreeMonoid α)))
    (β uHi wHi : ℝ) : Prop :=
  ∃ X ∈ sources, ∃ c : α,
    (∃ i ≤ Fintype.card Q, ChainAdvancesBy X
        (fun x => stateIndecision A O R.B R.F (A.state (x * FreeMonoid.of c))) β (uHi / β ^ i))
      ∨ ChainAdvancesBy X (fun x => edgeDisagreeProb O R.B R.F H x c) β wHi

/-- A node read of `z` at the middle of the band: the family's vote on `z` past the middle. -/
def midRead (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (ω : Ω)
    (z : FreeMonoid α) : Prop :=
  B.lo + B.hi < 2 * acceptsOn F (fun w => O.mq w ω) z

/-- A node read of a string whose state is not badly read falls on one side of the middle but
for chance `φ`: the vote's law is the state's, a full margin from the middle. -/
def MidFlipPremise (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (uHi φ : ℝ) : Prop :=
  ∀ z, stateIndecision A O B F (A.state z) < uHi →
    μ.real {ω | midRead O B F ω z} ≤ φ ∨ 1 - φ ≤ μ.real {ω | midRead O B F ω z}

/-- No suffix in `V` ends another: reads of different strings through `V` ask the oracle about
different strings. -/
def SuffixFree (V : Finset (FreeMonoid α)) : Prop :=
  ∀ v ∈ V, ∀ v' ∈ V, ∀ u : FreeMonoid α, v = u * v' → u = 1

/-- Every badly read state is reached by a letter from a string read before position `L`. -/
def BadVisited (A : DFA (FreeMonoid α) Q) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (D : Measure (FreeMonoid α)) (L : ℕ) (uHi : ℝ) : Prop :=
  ∀ q, uHi ≤ stateIndecision A O B F q →
    ∃ y c, y.toList.length < L ∧ 0 < D.real {p | y.toList <+: p.toList}
      ∧ A.state (y * FreeMonoid.of c) = q

/-- A bound on the node reads a pass makes: every string it sifts is the seed or a probe's prefix,
extended by at most a letter, and every midfix it reads at is the final tree's, of which there are
at most `N + 2`, preceded by at most a letter. -/
def passReadBound (S : RoundSetting α μ Q) : ℕ :=
  (S.seed.length + S.N * (S.L + 1)) * (1 + Fintype.card α) * ((S.N + 2) * (1 + Fintype.card α))

/-- `RoundTetrachotomy`: but for the chance the gate's node reads flip, a round ends with (1)
agreement within `ε`, (2) a population the next gate must act on, (3) halving, or (4) a chain's
odds multiplied by at least `β`.  `live` are the chains carried into the round; the round also
starts one from the `ν` root and from every per-state population. -/
def RoundTetrachotomy : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Q : Type*} [Fintype Q] (S : RoundSetting α μ Q) (ε : ℝ)
    (nH nS : ℕ) (σ f a : ℝ) (r : ℕ) (live : List (Measure (FreeMonoid α))) (β uHi φ : ℝ),
    S.Valid → 0 < ε → 1 < β → 0 ≤ φ → 2 * (β + 1) * (S.N + 1) * φ ≤ 1 →
    (∀ᵐ x ∂S.D, x.toList.length = S.L) → SuffixFree (S.F ∪ S.K.train S.F) →
    MidFlipPremise S.A S.O S.B S.F uHi φ → BadVisited S.A S.O S.B S.F S.D S.L uHi →
    (μ.prod (Measure.pi fun _ : Fin S.N => S.D)).real {θ |
        let R := readsAt S.O S.B S.F θ.1
        let s := roundEnd S.K S.O S.B S.F S.seed θ
        ¬ S.D.real {x | DFAandDTDisagree R s.hyp x} ≤ ε
          ∧ ¬ (HarvestSpread S.A R s.hyp S.D S.L S.κ
            ∧ (PopulationIndecisive S.A S.O R s.hyp S.D S.L S.τ
              ∨ WrongEdgeHarvest S.A S.O R s.hyp S.D S.L nH nS σ a
              ∨ ∃ l ∈ s.hyp.tree.paths, ∃ c : α, EdgeSelected S.A S.O R s.hyp S.D l c f a r
                  ∧ EdgePopulationIndecisive S.A S.O R (S.D[|settlesAt R s.hyp l]) c S.τ))
          ∧ ¬ s.halves S.τ
          ∧ ¬ ChainAdvances S.A S.O R s.hyp
              (nuRoot S.D S.L :: live ++ s.hyp.tree.paths.map fun l => S.D[|settlesAt R s.hyp l])
              β uHi (1 - 2 * (S.N + 1) * φ)}
      ≤ (S.L + 1) * (S.N + 1) * φ / ε + passReadBound S * φ

end OrthoDFA
