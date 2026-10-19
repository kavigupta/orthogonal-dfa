import OrthoDFA.Level5OldClaim

/-!
# Level 5's detailed bound

Internal: `Level5Correct` is derived from what is here, and none of it needs to be read to
trust that claim.

`Level5RoundRisk`: the round fails to end well with chance at most `roundRisk`, the least, over the
proof's free parameters (`Knobs`) meeting `Knobs.Admissible`, of `roundBound`, a sum of named
terms. `Level5RiskBound`: `roundRisk ≤ δ` under explicit conditions on the
laws' `κ` and `ε`, the goodness level `θ` and the settings.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A crude bound on `log (8 K / δ)`, `K` the most trees, edge maps and edges a tail is united
over. -/
def unionLog (σ : Type*) [Fintype σ] (δ : ℝ) : ℝ :=
  4 * (Fintype.card σ + 2) * Fintype.card α * Real.log ((Fintype.card σ + 3) * Fintype.card α / δ)

/-- The probes that search, at a tree and edges. -/
def searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    Set (FreeMonoid α) :=
  {x | match probeOrAgree rd T edges k x with
    | .wrongEdge _ _ _ _ | .undecided _ false => True
    | _ => False}

theorem searchAt_eq (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    searchAt rd k T edges = {x | match probeOrAgree rd T edges k x with
      | .wrongEdge _ _ _ _ | .undecided _ false => True
      | _ => False} := rfl

/-- The run from `s` over the draws ends with an ending satisfying `Q`. -/
def RunEnds {S X E : Type*} (step : S → X → S ⊕ (E × S)) (Q : E → S → Prop) :
    S → List X → Prop
  | _, [] => False
  | s, x :: xs => match step s x with
    | .inl s' => RunEnds step Q s' xs
    | .inr (e, s') => Q e s'

/-- The sequential tests' settings: the level in `(0, 1)`, every test's rate in `(0, 1)` with its
alternative on its far side, above it for the harvests' tests and below it for success's. -/
structure Level5Settings.SeqOK (C : Level5Settings) : Prop where
  a_pos : 0 < C.testLevel
  a_lt : C.testLevel < 1
  θs_pos : 0 < C.startUndecidedRate
  θs_alt : C.startUndecidedRate < C.startAltRate
  θs_alt_lt : C.startAltRate < 1
  θpt_pos : 0 < C.middleUndecidedRate
  θpt_alt : C.middleUndecidedRate < C.middleAltRate
  θpt_alt_lt : C.middleAltRate < 1
  εd_pos : 0 < C.disagreementRate
  εd_lt : C.disagreementRate < 1
  εd_alt_pos : 0 < C.disagreementAltRate
  εd_alt : C.disagreementAltRate < C.disagreementRate
  φ_pos : 0 < C.searchShare
  φ_lt : C.searchShare < 1

/-- What a harvest's sampler must yield: the least of what the start's, an edge's and the middles'
tests certify. -/
noncomputable def Level5Settings.harvestYield (C : Level5Settings) : ℝ :=
  min C.startUndecidedRate
    (min (C.edgeTraffic * C.edgeUndecidedRate / 2) (C.searchShare * C.middleUndecidedRate / 2))

/-- The round ends well at the ending `e` and the state `s` it ends in, the read distributions
`dist` saying which side is rarer. -/
def Level5EndsWell {σ : Type*} (M : _root_.DFA α σ) (bad : σ → Prop) (dist : σ → ARU → ℝ)
    (D : Measure (FreeMonoid α)) (C : Level5Settings) (J : ℕ) (ū : ℝ) (rd : FreeMonoid α → ARU)
    (e : Level5Ending α) (s : Level5State α) : Prop :=
  EndsWell M bad D (3 / 2 * C.disagreementRate)
    (checkedProbes M dist D rd C.walkStart C.maxLeaves J ū ⟨s.tree, s.edges⟩)
    C.walkStart
    (harvestLaw D (fun z => (rd z).cut) C.walkStart ⟨s.tree, s.edges⟩)
    C.harvestYield (toRoundEnd e)

/-- The most hypotheses a round goes through: each split adds a leaf, and between splits each
edge out of a leaf is learned at most once. -/
def versionBound (C : Level5Settings) (nα : ℕ) : ℕ := C.maxLeaves * (C.maxLeaves * nα + 1)

/-- The proof's free parameters. -/
structure Knobs where
  /-- The probes a hypothesis gets before success must have fired. -/
  nEnd : ℕ
  /-- The success test's count at `nEnd` probes. -/
  hS : ℕ
  /-- The good read-states' undecided first reads, as a share of an edge's first reads. -/
  θg : ℝ
  /-- The good read-states' undecided start reads, as a share of the probes. -/
  θgs : ℝ
  /-- The searches stopping at an undecided middle at a good read-state, as a share of searches. -/
  θgpt : ℝ
  /-- The start field's exponent. -/
  l : ℝ

section Bound

/-- At least the trees of the class with at most `S` splits that are not genuine. -/
def classCount (σ : Type*) [Fintype σ] (S : ℕ) : ℝ :=
  (1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α) ^ (Fintype.card σ + S)

/-- At least the edge maps into such a tree's leaves. -/
def edgeMapCount (σ : Type*) [Fintype σ] (S : ℕ) : ℝ :=
  (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)

/-- The relations the knobs must meet, at `|Q| = card σ` states, the laws' `κ` and `ε`, the
goodness level `θ`, the probes' law `D` with lengths at most `L` and the settings `C`. -/
structure Knobs.Admissible (σ : Type*) [Fintype σ] (κ ε θ : ℝ) (D : Measure (FreeMonoid α)) (L : ℕ)
    (C : Level5Settings) (kn : Knobs) : Prop where
  κ_nonneg : 0 ≤ κ
  ε_nonneg : 0 ≤ ε
  θ_nonneg : 0 ≤ θ
  l_nonneg : 0 ≤ kn.l
  θg_ge : 2 * (3 / 2 * θ) ≤ kn.θg
  θgpt_ge : 2 * ((L + 1) * (Fintype.card σ + 1) * (3 / 2 * θ)) ≤ kn.θgpt
  θgpt_le : 4 * kn.θgpt ≤ C.middleUndecidedRate
  θe_ge : 8 * kn.θg ≤ C.edgeUndecidedRate
  θgs_nonneg : 0 ≤ kn.θgs
  θgs_le : 2 * kn.θgs ≤ C.startUndecidedRate
  φe_nonneg : 0 ≤ C.edgeTraffic
  Lmax_ge : Fintype.card σ + 2 ≤ C.maxLeaves
  L_pos : 1 ≤ L
  seq : C.SeqOK
  n₀_le_hS : C.firstLook ≤ kn.hS + 1
  hS_le : kn.hS ≤ kn.nEnd
  pt_level : seqRate C.middleUndecidedRate C.middleAltRate C.testLevel (kn.hS + 1) (kn.hS + 1)
  success_level : seqRate C.disagreementRate C.disagreementAltRate C.testLevel kn.nEnd kn.hS
  φpt_nEnd : C.searchShare * kn.nEnd ≤ kn.hS + 1
  exc_gt : ∀ j, 2 * (L + 1) * (1 + C.edgeUndecidedRate * C.maxLeaves)
      * Real.log (1 / C.testLevel) < C.edgeExcess j
  gate_test : 6 * Real.log (1 / C.testLevel) ≤ C.firstLook
  budget_ge : versionBound C (Fintype.card α) * kn.nEnd ≤ C.probeBudget
  alt_eq : C.disagreementAltRate = 7 / 8 * C.disagreementRate

variable (σ : Type*) [Fintype σ] (κ ε θ : ℝ) (D : Measure (FreeMonoid α)) (L : ℕ)
    (C : Level5Settings)
  (kn : Knobs)

/-- Good read-states' undecided first reads at an edge exceeding `θg` of its first reads past the
edge test's slack, at some tree, edge map and edge. -/
def edgeTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ 0
    * (edgeMapCount (α := α) σ 0 * ((Fintype.card σ + 2) * Fintype.card α))
    * Real.exp (-(C.edgeTraffic * (C.edgeUndecidedRate
        - 8 * kn.θg) / 4 / (2 * prefixMax D C.walkStart * (L + 1)
      * (1 + 4 * (3 / 2 * θ) * (Fintype.card σ + 1))))))

/-- Good read-states' undecided start reads exceeding `θgs` of the probes, at some tree. -/
def startTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ 0
    * Real.exp ((Fintype.card σ + 1) * (3 / 2 * θ)
      * (Real.exp (kn.l * prefixMax D C.walkStart) - 1) / prefixMax D C.walkStart - kn.l * kn.θgs))

/-- Searches stopping at an undecided middle at a good read-state exceeding `θgpt` of the
searches past the middles test's slack, at some tree and edge map. -/
def middlesTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ 0 * edgeMapCount (α := α) σ 0
    * Real.exp (-(kn.θgpt * C.searchShare / 2 / (2 * prefixMax D C.walkStart
      * (1 + 4 * ((L + 1) * (Fintype.card σ + 1) * (3 / 2 * θ)))))))

/-- Some step reading a string on its state's rarer side: a step reads at most
`(L + 1) maxLeaves` of its prefixes followed by midfixes and re-sifts at most `maxLeaves² |Σ|`,
each on its rarer side with chance at most `ε + κ` when first read. -/
def wrongTerm : ENNReal :=
  ENNReal.ofReal ((ε + κ) * (C.probeBudget
    * ((L + 1) * C.maxLeaves + C.maxLeaves * Fintype.card α * C.maxLeaves)))

/-- The harvests' and success's tests firing wrongly in some hypothesis: the start's, the
middles', their gate's and success's tests, two potentials at each edge out of a leaf, and the
success test passing with the rarer probes heavy, each at most once a hypothesis. -/
def testLevels : ENNReal :=
  ENNReal.ofReal (versionBound C (Fintype.card α) * (2 * C.maxLeaves * Fintype.card α + 5)
    * C.testLevel)

/-- The round's failure chance at the knobs `kn`: the three noise fields' tails, a read on the
rarer side splitting wrongly, and the tests' levels. -/
def roundBound : ENNReal :=
  edgeTail σ θ D L C kn + startTail σ θ D C kn + middlesTail σ θ D L C kn
    + wrongTerm (α := α) κ ε L C + testLevels (α := α) C

/-- The least of `roundBound` over the admissible knobs. -/
def roundRisk : ENNReal :=
  ⨅ (kn : Knobs) (_ : kn.Admissible σ κ ε θ D L C), roundBound σ κ ε θ D L C kn

end Bound

/-- `Level5RiskBound`: `roundRisk ≤ δ` once the settings are wide enough for the laws' `κ` and
`ε` and the goodness level `θ`, with `Λ = unionLog σ δ`, `p₀ = prefixMax D C.walkStart` and
`lg = log (1 / C.testLevel)`. -/
def Level5RiskBound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] (σ : Type*) [Fintype σ]
    (κ ε θ : ℝ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ)
    (C : Level5Settings) (δ : ℝ),
    let n : ℝ := Fintype.card σ
    let Λ := unionLog (α := α) σ δ
    let p₀ := prefixMax D C.walkStart
    let lg := Real.log (1 / C.testLevel)
    let A : ℝ := Fintype.card α
    0 < δ → δ ≤ 1 → 1 ≤ L → Fintype.card σ + 2 ≤ C.maxLeaves →
    0 ≤ κ → 0 ≤ ε → 0 ≤ θ → 48 * θ ≤ C.edgeUndecidedRate →
    96 * ((L + 1) * (n + 1)) * θ ≤ C.middleUndecidedRate →
    6 * (n + 1) * θ + 2 * p₀ * Λ ≤ C.startUndecidedRate →
    32 * (L + 1) * p₀ * Λ ≤ C.edgeTraffic * C.edgeUndecidedRate
        → 256 * p₀ * Λ ≤ C.middleUndecidedRate * C.searchShare →
    C.disagreementRate < 1 → C.startUndecidedRate < C.startAltRate → C.startAltRate < 1 →
    C.middleUndecidedRate = 1 / 2 → C.middleAltRate = 3 / 4 →
    C.disagreementAltRate = 7 / 8 * C.disagreementRate →
    C.searchShare = 3 / 4 * C.disagreementRate → 6 * lg ≤ C.firstLook → C.firstLook ≤ 7 * lg →
    (∀ j, 2 * (L + 1) * (1 + C.edgeUndecidedRate * C.maxLeaves) * lg < C.edgeExcess j) →
    0 < C.testLevel →
    4 * (versionBound C (Fintype.card α) * (2 * C.maxLeaves * A + 4)) * C.testLevel ≤ δ →
    (versionBound C (Fintype.card α) : ℝ) * (112 * lg / C.disagreementRate + 1) ≤ C.probeBudget →
    4 * (ε + κ) * C.probeBudget * ((L + 1) * C.maxLeaves + C.maxLeaves * A * C.maxLeaves) ≤ δ →
    roundRisk σ κ ε θ D L C ≤ ENNReal.ofReal δ

/-- `Level5RoundRisk`: with an oracle reading every string with the read model's laws, the round
over `C.probeBudget` probes drawn from `P.D` fails to end well with chance at most `roundRisk`,
and `32ū/εd` more for the undecided reads it answers for. -/
def Level5RoundRisk : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ : ℝ) (P : PrefixDistribution α) (C : Level5Settings) (J : ℕ) (ū : ℝ),
    0 ≤ ū → 32 ≤ C.disagreementRate * C.probeBudget →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (O.read z ω).cut)
          (Level5EndsWell M (fun q => ¬ O.Good θ q) O.readDist P.D C J ū (O.read · ω))
          initialState (List.ofFn xs)} ∂μ
      ≤ roundRisk σ O.κ O.ε θ P.D P.L C + ENNReal.ofReal (32 * ū / C.disagreementRate)

end OrthoDFA

end
