import OrthoDFA.Level4Claim

/-!
# Level 4's detailed bound

Internal: `Level4Correct` is derived from what is here, and none of it needs to be read to
trust that claim.

`Level4RoundRisk`: the round fails to end well with chance at most `roundRisk`, the least, over the
proof's free parameters (`Knobs`) meeting `Knobs.Admissible`, of `roundBound`, a sum of named
terms, plus the yield's term. `Level4RiskBound`: `roundRisk ≤ δ` under explicit conditions on the
laws' `κ` and `ε`, the goodness level `θ` and the settings.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `searchingProbes` at a tree and curried edges. -/
def searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α) :
    Set (FreeMonoid α) :=
  searchingProbes rd k ⟨T, Function.uncurry edges⟩

theorem searchAt_eq (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : EdgeTable α) :
    searchAt rd k T edges = {x | match probeBy (fun z => (rd z).cut) T edges k x with
      | .pair _ | .edge _ _ | .triple _ => True
      | _ => False} := rfl

/-- `level4Replay` at a tree and curried edges. -/
def replayAt (rd : FreeMonoid α → ARU) (T : DTree α) (edges : EdgeTable α) (w : FreeMonoid α)
    (e : ℕ) : List (FreeMonoid α) :=
  level4Replay rd ⟨T, Function.uncurry edges⟩ w e

theorem replayAt_eq (rd : FreeMonoid α → ARU) (T : DTree α) (edges : EdgeTable α) (w : FreeMonoid α)
    (e : ℕ) : replayAt rd T edges w e =
      ((firstsBy (fun z => (rd z).cut) T edges e w).filter fun q => (rd q.2).cut = none).map
        Prod.snd := rfl

/-- The run from `s` over the draws ends with an ending satisfying `Q`. -/
def RunEnds {S X E : Type*} (step : S → X → S ⊕ (E × S)) (Q : E → S → Prop) :
    S → List X → Prop
  | _, [] => False
  | s, x :: xs => match step s x with
    | .inl s' => RunEnds step Q s' xs
    | .inr (e, s') => Q e s'

/-- The round ends well at the ending `e` and the state `s` it ends in. -/
def Level4EndsWell {σ : Type*} (M : _root_.DFA α σ) (bad : σ → Prop) (D : Measure (FreeMonoid α))
    (C : Level4Settings) (a L : ℕ) (y : ℝ) (rd : FreeMonoid α → ARU) (e : Level4Ending α)
    (s : Level4State α) : Prop :=
  EndsWell M bad D C.disagreementRate (searchAt rd C.walkStart s.tree s.edges) a L
    (replayAt rd s.tree s.edges) y (toRoundEnd e s)

/-- A hypothesis recording less than `θr` of the time keeps over `nEnd` probes with chance at most
this: `θpt'` of its searches stopping at undecided middles but fewer than `hP` of the first
`hS + 1`, or the disagreement rate at most `εd'` with more than `hS` of `nEnd`, too many for
success. -/
noncomputable def termLevel (nEnd hP hS : ℕ) (θpt' εd' : ℝ) : ℝ :=
  (1 - binomSfGe (hS + 1) θpt' hP) + binomSfGe nEnd εd' (hS + 1)

/-- The most versions within a sub-round: each edge learned, and redirected, at most once. -/
def subVersions (C : Level4Settings) (nα : ℕ) : ℕ := 2 * C.maxLeaves * nα

/-- A sub-round's probes: `nEnd` for each version, and `nRec` at hypotheses that record. -/
def subT (C : Level4Settings) (nα nEnd nRec : ℕ) : ℕ := (subVersions C nα + 1) * nEnd + nRec

/-- A sub-round's chance of keeping its tree over its `subT` probes with fewer than `B` records
that are not true: some version keeping over `nEnd` probes, or `nRec` probes, each recording with
chance `θr`, making no more than the `|Q| |Σ| m` records true edges and targets hold and the
`B - 1` that are not true. -/
noncomputable def subOpen (C : Level4Settings) (nα nσ nRec B : ℕ) (θr βt : ℝ) : ℝ :=
  (subVersions C nα + 1) * βt + (1 - binomSfGe nRec θr (nσ * nα * C.recordsToMove + B))

/-- The chance that the round's records that are not true reach `(S + 1) m` within `W` probes:
a probe's are at most `κ` times its undecided strings plus `ρ`, which the start's and the edges'
tests hold to `θs'` a probe past `Xs`, and at each of at most `2 Lmax |Σ|` edges to `θe` of its
reads past `Xe` once charged `φe` of the probes, with at most `L + 1` undecided strings and
`(L + 1) Lmax` first reads a probe. -/
noncomputable def fakeRun (C : Level4Settings) (nα S L W : ℕ) (ρ Xe θs' Xs η ν : ℝ) : ℝ :=
  Real.exp (-η * ((S + 1) * C.recordsToMove) + ν * (2 * C.maxLeaves * nα * Xe + Xs)
    + W * (ν * (C.edgeUndecidedRate * ((L + 1) * C.maxLeaves) + θs'
        + 2 * C.maxLeaves * nα * C.edgeTraffic)
      + (Real.exp η - 1) * ρ))

/-- The proof's free parameters. -/
structure Knobs where
  /-- The splits that are not genuine a round's trees may hold. -/
  S : ℕ
  /-- The probes each hypothesis gets before the termination tests read it. -/
  nEnd : ℕ
  /-- The probes at hypotheses that record. -/
  nRec : ℕ
  /-- The middles test's count at `hS + 1` searches. -/
  hP : ℕ
  /-- The success test's count at `nEnd` probes. -/
  hS : ℕ
  /-- The records field's slack. -/
  ρ : ℝ
  /-- The good read-states' undecided first reads, as a share of an edge's first reads. -/
  θg : ℝ
  /-- The good read-states' undecided start reads, as a share of the probes. -/
  θgs : ℝ
  /-- The searches stopping at an undecided middle at a good read-state, as a share of searches. -/
  θgpt : ℝ
  /-- The middles rate termination tests against. -/
  θpt' : ℝ
  /-- The rate below which a hypothesis counts as not recording. -/
  θr : ℝ
  /-- The disagreement rate termination tests against. -/
  εd' : ℝ
  /-- The most any edge test's excess is. -/
  Xe : ℝ
  /-- The start test's allowance: a rate. -/
  θs' : ℝ
  /-- The start test's allowance: an offset. -/
  Xs : ℝ
  /-- The fake race's exponent on records that are not true. -/
  η : ℝ
  /-- The fake race's exponent on the tests' excess. -/
  ν : ℝ
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
goodness level `θ`, the probes' law `D` with lengths at most `L`, the loop's settings `C` and the
records' coefficient `κs`. -/
structure Knobs.Admissible (σ : Type*) [Fintype σ] (κ ε θ : ℝ) (D : Measure (FreeMonoid α)) (L : ℕ)
    (C : Level4Settings) (κs : ℝ) (kn : Knobs) : Prop where
  κ_pos : 0 < κ
  ε_nonneg : 0 ≤ ε
  θ_nonneg : 0 ≤ θ
  records_slack : 2 * (ε * ((L + 1) * (Fintype.card σ + kn.S + 1))) ≤ kn.ρ
  l_nonneg : 0 ≤ kn.l
  θg_ge : 2 * (3 / 2 * θ) ≤ kn.θg
  θgpt_ge : 2 * ((L + 1) * (Fintype.card σ + kn.S + 1) * (3 / 2 * θ)) ≤ kn.θgpt
  θgpt_le : 2 * kn.θgpt ≤ 1
  θe_ge : 4 * kn.θg ≤ C.edgeUndecidedRate
  m_pos : 0 < C.recordsToMove
  Lmax_ge : Fintype.card σ + kn.S + 3 ≤ C.maxLeaves
  L_pos : 1 ≤ L
  a_pos : 0 < C.testLevel
  a_le : C.testLevel ≤ 1
  θpt_nonneg : 0 ≤ C.middleUndecidedRate
  θpt_le : C.middleUndecidedRate ≤ 1
  εd_nonneg : 0 ≤ C.disagreementRate
  εd_le : C.disagreementRate ≤ 1
  θpt'_nonneg : 0 ≤ kn.θpt'
  θpt'_le : kn.θpt' ≤ 1
  θr_nonneg : 0 ≤ kn.θr
  θr_le_one : kn.θr ≤ 1
  εd'_nonneg : 0 ≤ kn.εd'
  εd'_le : kn.εd' ≤ 1
  θr_le : kn.θr ≤ (1 - kn.θpt') * kn.εd'
  θgs_nonneg : 0 ≤ kn.θgs
  θgs_le : kn.θgs ≤ 1
  φe_nonneg : 0 ≤ C.edgeTraffic
  φpt_nonneg : 0 ≤ C.searchShare
  φpt_le : C.searchShare ≤ 1
  n₀_le_nEnd : C.firstLook ≤ kn.nEnd
  n₀_le_hS : C.firstLook ≤ kn.hS + 1
  pt_level : binomSfGe (kn.hS + 1) C.middleUndecidedRate kn.hP < C.testLevel
  success_level : 1 - binomSfGe kn.nEnd C.disagreementRate (kn.hS + 1) < C.testLevel
  success_reach : C.testLevel ≤ binomSfGe kn.nEnd C.disagreementRate kn.hS
  φpt_nEnd : C.searchShare * kn.nEnd ≤ kn.hS + 1
  exc_gt : ∀ j, 4 * (L + 1) * (1 + 4 * kn.θg * C.maxLeaves) * Real.log (1 / C.testLevel)
      < C.edgeExcess j
  start_test : ∀ t h, C.firstLook ≤ t → binomSfGe t C.startUndecidedRate h < C.testLevel →
    binomSfGe t kn.θgs ((h + 1) / 2) ≤ C.testLevel
  pt_test : ∀ t h, C.firstLook ≤ t → binomSfGe t C.middleUndecidedRate h < C.testLevel →
    binomSfGe t (2 * kn.θgpt) ((h + 1) / 2) ≤ C.testLevel
  φpt_test : ∀ n, C.firstLook ≤ n → binomSfGe n (C.searchShare / 2) ⌈C.searchShare * n⌉₊
      ≤ C.testLevel
  Xe_nonneg : 0 ≤ kn.Xe
  exc_le : ∀ j, C.edgeExcess j ≤ kn.Xe
  θs'_nonneg : 0 ≤ kn.θs'
  n₀_le_Xs : C.firstLook ≤ kn.Xs
  start_lin : ∀ n h : ℕ, C.firstLook ≤ n → kn.θs' * n + kn.Xs ≤ h
      → binomSfGe n C.startUndecidedRate h < C.testLevel
  η_nonneg : 0 ≤ kn.η
  ν_nonneg : 0 ≤ kn.ν
  race_side : (Real.exp kn.η - 1) * κs * (L + 1) ≤ 1 - Real.exp (-(kn.ν * (L + 1)))
  budget_ge : (kn.S + Fintype.card σ + 1) * subT C (Fintype.card α) kn.nEnd kn.nRec ≤ C.probeBudget

variable (σ : Type*) [Fintype σ] (κ ε θ : ℝ) (D : Measure (FreeMonoid α)) (L : ℕ)
    (C : Level4Settings)
  (kn : Knobs)

/-- Records that are not true exceeding `κs` times the undecided strings plus `ρ`, at some tree of
the class and edge map. -/
def recordsTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S * edgeMapCount (α := α) σ kn.S
    * Real.exp (-((kn.ρ - 2 * (ε * ((L + 1) * (Fintype.card σ + kn.S + 1))))
      / (2 * prefixMax D C.walkStart
        * (1 + 4 * (κ * (L + 1) + ε * ((L + 1) * (Fintype.card σ + kn.S + 1))))))))

/-- Good read-states' undecided first reads at an edge exceeding `θg` of its first reads past the
edge test's slack, at some tree, edge map and edge. -/
def edgeTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S
    * (edgeMapCount (α := α) σ kn.S * ((Fintype.card σ + kn.S + 2) * Fintype.card α))
    * Real.exp (-(C.edgeTraffic * (C.edgeUndecidedRate / 2
        - 2 * kn.θg) / 2 / (2 * prefixMax D C.walkStart * (L + 1)
      * (1 + 4 * (3 / 2 * θ) * (Fintype.card σ + kn.S + 1))))))

/-- Good read-states' undecided start reads exceeding `θgs` of the probes, at some tree. -/
def startTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S
    * Real.exp ((Fintype.card σ + kn.S + 1) * (3 / 2 * θ)
      * (Real.exp (kn.l * prefixMax D C.walkStart) - 1) / prefixMax D C.walkStart - kn.l * kn.θgs))

/-- Searches stopping at an undecided middle at a good read-state exceeding `θgpt` of the
searches past the middles test's slack, at some tree and edge map. -/
def middlesTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S * edgeMapCount (α := α) σ kn.S
    * Real.exp (-(kn.θgpt * C.searchShare / 2 / (2 * prefixMax D C.walkStart
      * (1 + 4 * ((L + 1) * (Fintype.card σ + kn.S + 1) * (3 / 2 * θ)))))))

/-- The harvests' and success's tests firing wrongly at some probe of the round. -/
def testLevels : ENNReal :=
  ENNReal.ofReal ((2 ^ (C.maxLeaves + 1) * Fintype.card α + C.probeBudget
      + 3 * (C.probeBudget * C.probeBudget)) * C.testLevel)

/-- The round's failure chance at the knobs `kn`: the four noise fields' tails, the records that
are not true racing the tests (`fakeRun`), a sub-round keeping its tree too long (`subOpen`), and
the tests' levels. -/
def roundBound : ENNReal :=
  recordsTail σ κ ε D L C kn + edgeTail σ θ D L C kn + startTail σ θ D C kn
    + middlesTail σ θ D L C kn
    + ENNReal.ofReal (fakeRun C (Fintype.card α) kn.S L
      ((kn.S + Fintype.card σ + 1) * subT C (Fintype.card α) kn.nEnd kn.nRec)
      kn.ρ kn.Xe kn.θs' kn.Xs kn.η kn.ν)
    + ENNReal.ofReal ((kn.S + Fintype.card σ + 1) * subOpen C (Fintype.card α) (Fintype.card σ)
      kn.nRec ((kn.S + 1) * C.recordsToMove) kn.θr (termLevel kn.nEnd kn.hP kn.hS kn.θpt' kn.εd'))
    + testLevels (α := α) C

/-- The least of `roundBound` over the admissible knobs. -/
def roundRisk (κs : ℝ) : ENNReal :=
  ⨅ (kn : Knobs) (_ : kn.Admissible σ κ ε θ D L C κs), roundBound σ κ ε θ D L C kn

end Bound

/-- `Level4RiskBound`: `roundRisk ≤ δ` once the loop's settings are wide enough for the laws' `κ`
and `ε` and the goodness level `θ`,
at `S` splits that are not genuine, with `Λ = unionLog σ S δ`, `p₀ = prefixMax D C.walkStart` and
`lg = log (1 / C.testLevel)`. -/
def Level4RiskBound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] (σ : Type*) [Fintype σ]
    (κ ε θ : ℝ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ)
    (C : Level4Settings) (κs δ : ℝ) (S : ℕ),
    let n : ℝ := Fintype.card σ + S
    let Λ := unionLog (α := α) σ S δ
    let p₀ := prefixMax D C.walkStart
    let lg := Real.log (1 / C.testLevel)
    let A : ℝ := Fintype.card α
    0 < δ → δ ≤ 1 → 1 ≤ L → Fintype.card σ + S + 3 ≤ C.maxLeaves →
    0 < κ → 2 * κ ≤ κs → 4 * κs * (L + 1) ≤ 1 →
    0 ≤ ε → 4 * ε * ((L + 1) * (n + 1)) ≤ 1 →
    0 ≤ θ → 24 * θ ≤ C.edgeUndecidedRate → 96 * ((L + 1) * (n + 1)) * θ ≤ C.middleUndecidedRate →
    24 * (n + 1) * θ + 8 * p₀ * Λ ≤ C.startUndecidedRate → C.startUndecidedRate ≤ 1 →
    32 * (L + 1) * p₀ * Λ ≤ C.edgeTraffic * C.edgeUndecidedRate
        → 256 * p₀ * Λ ≤ C.middleUndecidedRate * C.searchShare →
    C.disagreementRate ≤ 1 → C.middleUndecidedRate ≤ 1 / 4
        → C.searchShare ≤ 3 / 4 * C.disagreementRate →
    32 * lg ≤ C.firstLook * C.startUndecidedRate ^ 2
        → 32 * lg ≤ C.firstLook * C.middleUndecidedRate ^ 2
        → 2 * lg ≤ C.firstLook * C.searchShare ^ 2 →
    (∀ j, 4 * (L + 1) * (1 + 12 * θ * C.maxLeaves) * lg < C.edgeExcess j) →
    (∀ j, C.edgeExcess j ≤ 8 * (L + 1) * (1 + 12 * θ * C.maxLeaves) * lg) →
    0 < C.testLevel → 4 * (2 ^ (C.maxLeaves + 1) * A + 7 * C.probeBudget ^ 2) * C.testLevel ≤ δ →
    (S + Fintype.card σ + 1) * ((2 * C.maxLeaves * A + 1)
        * (8 * lg / (C.disagreementRate ^ 2 * C.middleUndecidedRate ^ 2)
            + 4 * C.firstLook / (3 * C.disagreementRate) + 2)
        + 8 * (Fintype.card σ * A + S + 1) * C.recordsToMove / C.disagreementRate
            + 32 * lg / C.disagreementRate ^ 2 + 1) ≤ C.probeBudget →
    4 * Real.log (8 / δ) ≤ (S + 1) * C.recordsToMove →
    16 * κs * (2 * C.maxLeaves * A * (8 * (L + 1) * (1 + 12 * θ * C.maxLeaves) * lg) + C.firstLook)
      ≤ (S + 1) * C.recordsToMove →
    16 * κs * C.probeBudget * (C.edgeUndecidedRate * ((L + 1) * C.maxLeaves)
        + 2 * C.startUndecidedRate + 2 * C.maxLeaves * A * C.edgeTraffic)
      ≤ (S + 1) * C.recordsToMove →
    16 * C.probeBudget * (ε * ((L + 1) * (n + 1)) + 3 * p₀ * Λ) ≤ (S + 1) * C.recordsToMove →
    roundRisk σ κ ε θ D L C κs ≤ ENNReal.ofReal δ


/-- `Level4RoundRisk`: with an oracle reading every string with the read model's laws, the round
over `C.probeBudget` probes drawn from `P.D` fails to end well with chance at most `roundRisk` plus
`δ`, at any records coefficient `κs ≥ 2κ`. The harvest's replay yields `(V − 2√(q V log(1/δ))) / n`,
where `V` is the mean count of positions from `start` to `L` at which a draw's prefix reads
undecided and `n = L + 1 − start` the anchors. -/
def Level4RoundRisk : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ : ℝ) (P : PrefixDistribution α) (C : Level4Settings) (κs δ : ℝ),
    2 * O.κ ≤ κs → 0 < δ → δ ≤ 1 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (O.read z ω).cut)
          (Level4EndsWell M (fun q => ¬ O.Good θ q) P.D C P.firstAnchor P.L
            ((undecidedCount P M O.readDist
              - 2 * Real.sqrt (P.prefixBound * undecidedCount P M O.readDist * Real.log (1 / δ)))
              / (P.L + 1 - P.firstAnchor : ℕ)) (O.read · ω))
          initialState (List.ofFn xs)} ∂μ
      ≤ roundRisk σ O.κ O.ε θ P.D P.L C κs + ENNReal.ofReal δ

end OrthoDFA

end
