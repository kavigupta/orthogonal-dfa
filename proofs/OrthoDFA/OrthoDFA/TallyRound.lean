import OrthoDFA.TallyLoop
import OrthoDFA.RoundEnd

/-!
# One round of the tally loop

The reads are the oracle: every string's read is accept, reject or undecided, independently across
strings, with a law that depends only on the string's state in the target DFA (`ReadModel`). A
read-state is good where it reads undecided at most `1.5θ` of the time.

`TallyRound`: over `C.budget` probes drawn from `D`, of length at most `L`, the round ends well
(`TallyEndsWell`: consistent with the hypothesis disagreeing on at most `εd` of the probes, or a
harvest more than half of whose strings are at read-states that are not good) but for chance at
most `roundRisk`: the least, over the proof's free parameters (`Knobs`) meeting
`Knobs.Admissible`, of `roundBound`, a sum of named terms.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The target, each of its states' read law, and what makes a read-state good. Every law is
nonnegative and reads its rarer decided side at most `κ` times as often as it is undecided, or at
most `ε`. -/
structure ReadModel (α σ : Type*) where
  M : _root_.DFA α σ
  dist : σ → ARU → ℝ
  θ : ℝ
  κ : ℝ
  ε : ℝ
  nonneg : ∀ r v, 0 ≤ dist r v
  rare : ∀ r, min (dist r .accept) (dist r .reject) ≤ max ε (κ * dist r .undecided)

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

/-- A read-state is good where it reads undecided at most `1.5θ` of the time. -/
def Good (r : σ) : Prop := G.dist r .undecided ≤ 3 / 2 * G.θ

/-- The probes that search: their walk and the cut disagree at a decided read. -/
def searchAt (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    Set (FreeMonoid α) :=
  {x | match probeBy (fun z => (rd z).cut) T edges k x with
    | .pair _ | .edge _ _ | .triple _ => True
    | _ => False}

end ReadModel

/-- The run from `s` over the draws ends with an ending satisfying `Q`. -/
def RunEnds {S X E : Type*} (step : S → X → S ⊕ (E × S)) (Q : E → S → Prop) :
    S → List X → Prop
  | _, [] => False
  | s, x :: xs => match step s x with
    | .inl s' => RunEnds step Q s' xs
    | .inr (e, s') => Q e s'

/-- The round's ending, as every level of the loop states it: success is consistent, a harvest
is its strings. -/
def tallyEnd : TEnd α → TState α → RoundEnd α
  | .success, _ => .consistent
  | .harvestStart, s => .harvest s.startH
  | .harvest e, s => .harvest (s.harv e.1 e.2)
  | .harvestPT, s => .harvest s.pt
  | .tooBig, _ => .failed

/-- The round ends well: consistent with the hypothesis disagreeing on at most `εd` of the probes,
or a harvest more than half of whose strings are at read-states that are not good. -/
def TallyEndsWell {σ : Type*} (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (C : TallyCfg)
    (rd : FreeMonoid α → ARU) (e : TEnd α) (s : TState α) : Prop :=
  EndsWell G.M (fun q => ¬ G.Good q) D C.εd (ReadModel.searchAt rd C.k s.tree s.edges)
    (tallyEnd e s)

/-- The largest mass the probes' law puts on one length-`k` prefix. -/
def prefixMax (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ := ⨆ u, D.real {x | prefixOf x k = u}

/-- A hypothesis recording less than `θr` of the time keeps over `nEnd` probes with chance at most
this: `θpt'` of its searches stopping at undecided middles but fewer than `hP` of the first
`hS + 1`, or the disagreement rate at most `εd'` with more than `hS` of `nEnd`, too many for
success. -/
noncomputable def termLevel (nEnd hP hS : ℕ) (θpt' εd' : ℝ) : ℝ :=
  (1 - binomSfGe (hS + 1) θpt' hP) + binomSfGe nEnd εd' (hS + 1)

/-- The most versions within a sub-round: each edge learned, and redirected, at most once. -/
def subVersions (C : TallyCfg) (nα : ℕ) : ℕ := 2 * C.Lmax * nα

/-- A sub-round's probes: `nEnd` for each version, and `nRec` at hypotheses that record. -/
def subT (C : TallyCfg) (nα nEnd nRec : ℕ) : ℕ := (subVersions C nα + 1) * nEnd + nRec

/-- A sub-round's chance of keeping its tree over its `subT` probes with fewer than `B` records
that are not true: some version keeping over `nEnd` probes, or `nRec` probes, each recording with
chance `θr`, making no more than the `|Q| |Σ| m` records true edges and targets hold and the
`B - 1` that are not true. -/
noncomputable def subOpen (C : TallyCfg) (nα nσ nRec B : ℕ) (θr βt : ℝ) : ℝ :=
  (subVersions C nα + 1) * βt + (1 - binomSfGe nRec θr (nσ * nα * C.m + B))

/-- The chance that the round's records that are not true reach `(S + 1) m` within `W` probes:
a probe's are at most `κ` times its undecided strings plus `ρ`, which the start's and the edges'
tests hold to `θs'` a probe past `Xs`, and at each of at most `2 Lmax |Σ|` edges to `θe` of its
reads past `Xe` once charged `φe` of the probes, with at most `L + 1` undecided strings and
`(L + 1) Lmax` first reads a probe. -/
noncomputable def fakeRun (C : TallyCfg) (nα S L W : ℕ) (ρ Xe θs' Xs η ν : ℝ) : ℝ :=
  Real.exp (-η * ((S + 1) * C.m) + ν * (2 * C.Lmax * nα * Xe + Xs)
    + W * (ν * (C.θe * ((L + 1) * C.Lmax) + θs' + 2 * C.Lmax * nα * C.φe)
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

variable {σ : Type*} [Fintype σ]

/-- At least the trees of the class with at most `S` splits that are not genuine. -/
def classCount (σ : Type*) [Fintype σ] (S : ℕ) : ℝ :=
  (1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α) ^ (Fintype.card σ + S)

/-- At least the edge maps into such a tree's leaves. -/
def edgeMapCount (σ : Type*) [Fintype σ] (S : ℕ) : ℝ :=
  (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)

/-- The relations the knobs must meet, at the read model `G`, the probes' law `D` with lengths at
most `L`, the loop's settings `C` and the records' coefficient `κs`. -/
structure Knobs.Admissible (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (L : ℕ)
    (C : TallyCfg) (κs : ℝ) (kn : Knobs) : Prop where
  κ_pos : 0 < G.κ
  ε_nonneg : 0 ≤ G.ε
  θ_nonneg : 0 ≤ G.θ
  records_slack : 2 * (G.ε * ((L + 1) * (Fintype.card σ + kn.S + 1))) ≤ kn.ρ
  l_nonneg : 0 ≤ kn.l
  θg_ge : 2 * (3 / 2 * G.θ) ≤ kn.θg
  θgpt_ge : 2 * ((L + 1) * (Fintype.card σ + kn.S + 1) * (3 / 2 * G.θ)) ≤ kn.θgpt
  θgpt_le : 2 * kn.θgpt ≤ 1
  θe_ge : 4 * kn.θg ≤ C.θe
  m_pos : 0 < C.m
  Lmax_ge : Fintype.card σ + kn.S + 3 ≤ C.Lmax
  L_pos : 1 ≤ L
  a_pos : 0 < C.a
  a_le : C.a ≤ 1
  θpt_nonneg : 0 ≤ C.θpt
  θpt_le : C.θpt ≤ 1
  εd_nonneg : 0 ≤ C.εd
  εd_le : C.εd ≤ 1
  θpt'_nonneg : 0 ≤ kn.θpt'
  θpt'_le : kn.θpt' ≤ 1
  θr_nonneg : 0 ≤ kn.θr
  θr_le_one : kn.θr ≤ 1
  εd'_nonneg : 0 ≤ kn.εd'
  εd'_le : kn.εd' ≤ 1
  θr_le : kn.θr ≤ (1 - kn.θpt') * kn.εd'
  θgs_nonneg : 0 ≤ kn.θgs
  θgs_le : kn.θgs ≤ 1
  φe_nonneg : 0 ≤ C.φe
  φpt_nonneg : 0 ≤ C.φpt
  φpt_le : C.φpt ≤ 1
  n₀_le_nEnd : C.n₀ ≤ kn.nEnd
  n₀_le_hS : C.n₀ ≤ kn.hS + 1
  pt_level : binomSfGe (kn.hS + 1) C.θpt kn.hP < C.a
  success_level : 1 - binomSfGe kn.nEnd C.εd (kn.hS + 1) < C.a
  success_reach : C.a ≤ binomSfGe kn.nEnd C.εd kn.hS
  φpt_nEnd : C.φpt * kn.nEnd ≤ kn.hS + 1
  exc_gt : ∀ j, 4 * (L + 1) * (1 + 4 * kn.θg * C.Lmax) * Real.log (1 / C.a) < C.exc j
  start_test : ∀ t h, C.n₀ ≤ t → binomSfGe t C.θs h < C.a →
    binomSfGe t kn.θgs ((h + 1) / 2) ≤ C.a
  pt_test : ∀ t h, C.n₀ ≤ t → binomSfGe t C.θpt h < C.a →
    binomSfGe t (2 * kn.θgpt) ((h + 1) / 2) ≤ C.a
  φpt_test : ∀ n, C.n₀ ≤ n → binomSfGe n (C.φpt / 2) ⌈C.φpt * n⌉₊ ≤ C.a
  Xe_nonneg : 0 ≤ kn.Xe
  exc_le : ∀ j, C.exc j ≤ kn.Xe
  θs'_nonneg : 0 ≤ kn.θs'
  n₀_le_Xs : C.n₀ ≤ kn.Xs
  start_lin : ∀ n h : ℕ, C.n₀ ≤ n → kn.θs' * n + kn.Xs ≤ h → binomSfGe n C.θs h < C.a
  η_nonneg : 0 ≤ kn.η
  ν_nonneg : 0 ≤ kn.ν
  race_side : (Real.exp kn.η - 1) * κs * (L + 1) ≤ 1 - Real.exp (-(kn.ν * (L + 1)))
  budget_ge : (kn.S + Fintype.card σ + 1) * subT C (Fintype.card α) kn.nEnd kn.nRec ≤ C.budget

variable (G : ReadModel α σ) (D : Measure (FreeMonoid α)) (L : ℕ) (C : TallyCfg) (kn : Knobs)

/-- Records that are not true exceeding `κs` times the undecided strings plus `ρ`, at some tree of
the class and edge map. -/
def recordsTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S * edgeMapCount (α := α) σ kn.S
    * Real.exp (-((kn.ρ - 2 * (G.ε * ((L + 1) * (Fintype.card σ + kn.S + 1))))
      / (2 * prefixMax D C.k
        * (1 + 4 * (G.κ * (L + 1) + G.ε * ((L + 1) * (Fintype.card σ + kn.S + 1))))))))

/-- Good read-states' undecided first reads at an edge exceeding `θg` of its first reads past the
edge test's slack, at some tree, edge map and edge. -/
def edgeTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S
    * (edgeMapCount (α := α) σ kn.S * ((Fintype.card σ + kn.S + 2) * Fintype.card α))
    * Real.exp (-(C.φe * (C.θe / 2 - 2 * kn.θg) / 2 / (2 * prefixMax D C.k * (L + 1)
      * (1 + 4 * (3 / 2 * G.θ) * (Fintype.card σ + kn.S + 1))))))

/-- Good read-states' undecided start reads exceeding `θgs` of the probes, at some tree. -/
def startTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S
    * Real.exp ((Fintype.card σ + kn.S + 1) * (3 / 2 * G.θ)
      * (Real.exp (kn.l * prefixMax D C.k) - 1) / prefixMax D C.k - kn.l * kn.θgs))

/-- Searches stopping at an undecided middle at a good read-state exceeding `θgpt` of the
searches past the middles test's slack, at some tree and edge map. -/
def middlesTail : ENNReal :=
  ENNReal.ofReal (classCount (α := α) σ kn.S * edgeMapCount (α := α) σ kn.S
    * Real.exp (-(kn.θgpt * C.φpt / 2 / (2 * prefixMax D C.k
      * (1 + 4 * ((L + 1) * (Fintype.card σ + kn.S + 1) * (3 / 2 * G.θ)))))))

/-- The harvests' and success's tests firing wrongly at some probe of the round. -/
def testLevels : ENNReal :=
  ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + C.budget + 3 * (C.budget * C.budget)) * C.a)

/-- The round's failure chance at the knobs `kn`: the four noise fields' tails, the records that
are not true racing the tests (`fakeRun`), a sub-round keeping its tree too long (`subOpen`), and
the tests' levels. -/
def roundBound : ENNReal :=
  recordsTail G D L C kn + edgeTail G D L C kn + startTail G D C kn + middlesTail G D L C kn
    + ENNReal.ofReal (fakeRun C (Fintype.card α) kn.S L
      ((kn.S + Fintype.card σ + 1) * subT C (Fintype.card α) kn.nEnd kn.nRec)
      kn.ρ kn.Xe kn.θs' kn.Xs kn.η kn.ν)
    + ENNReal.ofReal ((kn.S + Fintype.card σ + 1) * subOpen C (Fintype.card α) (Fintype.card σ)
      kn.nRec ((kn.S + 1) * C.m) kn.θr (termLevel kn.nEnd kn.hP kn.hS kn.θpt' kn.εd'))
    + testLevels (α := α) C

/-- The least of `roundBound` over the admissible knobs. -/
def roundRisk (κs : ℝ) : ENNReal :=
  ⨅ (kn : Knobs) (_ : kn.Admissible G D L C κs), roundBound G D L C kn

end Bound

/-- `TallyRound`: with independent reads with the read model's laws, the round over `C.budget`
probes drawn from `D`, of length at most `L`, fails to end well with chance at most `roundRisk`,
at any records coefficient `κs ≥ 2κ`. -/
def TallyRound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (G : ReadModel α σ)
    (read : FreeMonoid α → Ω → ARU), (∀ z, Measurable (read z)) → iIndepFun read μ →
    (∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) →
    ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (L : ℕ),
    (∀ᵐ x ∂D, x.toList.length ≤ L) → ∀ (C : TallyCfg) (κs : ℝ), 2 * G.κ ≤ κs →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C (read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ roundRisk G D L C κs

end OrthoDFA

end
