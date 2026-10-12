import OrthoDFA.Level4

/-!
# Level 4's claim

The oracle (`TriOracle`): every string's read is accept, reject or undecided, independently across
strings, with a distribution that depends only on the string's state in the target DFA. A
read-state is good where it reads undecided at most `1.5θ` of the time.

`Level4Correct`: over `settings.probeBudget` probes drawn from a `PrefixDistribution`,
`indecisiveLStarLevel4` ends well (`EndsWell`: consistent with the hypothesis searching on at most
`εd` of the probes, or a harvest whose sampler finds more than half of what it finds at
read-states that are not good, is spread and yields) but for chance at most `δ`, at settings the
proof chooses, once the reads' `κ` and `ε`, the goodness level and the probes' prefix mass are
small enough. It is derived from a detailed bound (`Proofs/Level4RiskDefs.lean`) that need not be
read to trust it.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The oracle: a read of every string, accept, reject or undecided, independent across strings,
each distributed as `readDist` at its state in the target `M`. Every state reads its rarer decided
side at most `κ` times as often as it is undecided, or at most `ε`. -/
structure TriOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : _root_.DFA α σ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : iIndepFun read μ
  readDist : σ → ARU → ℝ
  reads_by_state : ∀ z r, μ.real {ω | read z ω = r} = readDist (M.eval z.toList) r
  κ : ℝ
  ε : ℝ
  readDist_nonneg : ∀ q v, 0 ≤ readDist q v
  rarer_side_le : ∀ q, min (readDist q .accept) (readDist q .reject)
    ≤ max ε (κ * readDist q .undecided)

/-- A read-state is good where it reads undecided at most `1.5θ` of the time. -/
def TriOracle.Good {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {σ : Type*}
    {M : _root_.DFA α σ} (O : TriOracle μ M) (θ : ℝ) (q : σ) : Prop :=
  O.readDist q .undecided ≤ 3 / 2 * θ

/-- The probes that search: their walk from `k` along `H`'s edges and the sift disagree at a
decided read. -/
def searchingProbes (read : FreeMonoid α → ARU) (k : ℕ) (H : Hypothesis α) :
    Set (FreeMonoid α) :=
  {x | match probeBy (fun z => (read z).cut) H.tree (Function.curry H.edges) k x with
    | .pair _ | .edge _ _ | .triple _ => True
    | _ => False}

/-- The largest mass the probes' distribution puts on one length-`k` prefix. -/
def prefixMax (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ := ⨆ u, D.real {x | prefixOf x k = u}

/-- A crude bound on `log (8 K / δ)`, `K` the most trees, edge maps and edges a tail is united
over. -/
def unionLog (σ : Type*) [Fintype σ] (S : ℕ) (δ : ℝ) : ℝ :=
  4 * (Fintype.card σ + S + 2) * Fintype.card α
    * Real.log ((Fintype.card σ + S + 3) * Fintype.card α / δ)

/-- `Level4Guarantee M O θ εd P δ S k`: some settings, walking from `k`, with disagreement rate
`εd` and a budget of at most `K · G` probes, make level 4 fail to end well, a harvest's sampler
yielding `ρ`, with chance at most `δ`, once the goodness level `θ`, the reads' `κ` and `ε` and the
probes' prefix mass at `k` are small enough for `n = |Q| + S` (at most `S` splits that are not
genuine), `A = |Σ|` and the draws' length `L`. `16ρ` is about the share of the probes at which the
round affords a record that is not true. -/
def Level4Guarantee [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : TriOracle μ M)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ) : Prop :=
  let n : ℝ := Fintype.card σ + S
  let A : ℝ := Fintype.card α
  let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
  let K : ℝ := (n + 1) * (2 * (n + 3) * A + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
    + 8 * (n + 3) * A * (P.L + 1)
  let X : ℝ := 4 * (2 ^ (Fintype.card σ + S + 5) * A + 7 * K ^ 2) / δ
  let G : ℝ := Real.log X + 2 * Real.log (2 * Real.log X)
  let bad := fun q => ¬ O.Good θ q
  0 < δ → δ ≤ 1 → 0 < εd → εd ≤ 1 → 1 ≤ P.L → 0 ≤ θ →
  384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
  0 < O.κ → 4 * O.κ * (8 * (n + 3) * (P.L + 1) * Real.sqrt (A * ρ / 3) + 1) ≤ ρ →
  0 ≤ O.ε → (P.L + 1) * (n + 1) * O.ε + 3 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ →
  ∃ settings : Level4Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧
    ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel4 settings (O.read · ω) draws
          ¬ EndsWell M bad P.D settings.disagreementRate
            (searchingProbes (O.read · ω) settings.walkStart result.hypothesis)
            settings.walkStart (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart) ρ
            result.ending} ∂μ
      ≤ ENNReal.ofReal δ

/-- `Level4Guarantee` at every target, oracle, goodness level, disagreement rate, probe
distribution, confidence, count of splits that are not genuine, and start `k`. -/
def Level4Correct : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ),
    Level4Guarantee M O θ εd P δ S k

end OrthoDFA

end
