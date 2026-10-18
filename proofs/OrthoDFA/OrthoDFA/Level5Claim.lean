import OrthoDFA.Level5
import OrthoDFA.ProbeDistribution

/-!
# Level 5's claim

The oracle (`TriOracle`): every string's read is accept, reject or undecided, independently across
strings, with a distribution that depends only on the string's state in the target DFA. A
read-state is good where it reads undecided at most `1.5θ` of the time.

`Level5Correct`: over `settings.probeBudget` probes drawn from a `PrefixDistribution`,
`indecisiveLStarLevel5` ends well (`EndsWell`: consistent with the hypothesis searching on at most
`εd` of the probes, or a harvest whose sampler finds more than half of what it finds at
read-states that are not good, is spread and yields) but for chance at most `δ`, at settings the
proof chooses, once the reads' `κ` and `ε`, the goodness level and the probes' prefix mass are
small enough. It is derived from a detailed bound (`Proofs/Level5RiskDefs.lean`) that need not be
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
def searchingProbes (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Set (FreeMonoid α) :=
  {x | match probeBy (fun z => (read z).cut) H.tree (Function.curry H.edges) k x with
    | .undecidedMid _ | .edge _ _ => True
    | _ => False}

/-- The largest mass the probes' distribution puts on one length-`k` prefix. -/
def prefixMax (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ := ⨆ u, D.real {x | prefixOf x k = u}

/-- `Level5Guarantee M O θ εd P δ k`: some settings, walking from `k`, with disagreement rate `εd`
and a budget of at most `K · G` probes, make level 5 fail to end well, a harvest's sampler
yielding `3εd/16`, with chance at most `δ`, once the goodness level `θ`, the reads' `κ` and `ε`
and the probes' prefix mass at `k` are small enough for `n = |Q|`, `A = |Σ|` and the draws'
length `L`. `V` bounds the hypotheses a round goes through. A read on its state's rarer side
among the strings a step reads splits wrongly, and `W` bounds those strings: a probe's at most
`(L + 1)(n + 2)` prefixes followed by midfixes, and a split's at most `(n + 2)² A` re-sifts. `U`
bounds the log of the trees, edge maps and edges a noise field is united over, over `δ`. -/
def Level5Guarantee [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : TriOracle μ M)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k : ℕ) : Prop :=
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let K : ℝ := V * (112 / εd + 1)
  let W : ℝ := (n + 2) * (P.L + 1 + (n + 2) * A)
  let U : ℝ := 4 * (n + 2) * A * Real.log ((n + 3) * A / δ)
  let bad := fun q => ¬ O.Good θ q
  0 < δ → δ ≤ 1 → 0 < εd → εd < 1 → 1 ≤ P.L → 0 ≤ θ →
  384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
  0 ≤ O.κ → 0 ≤ O.ε → 4 * (O.ε + O.κ) * K * G * W ≤ δ →
  1024 * (P.L + 1) * U * prefixMax P.D k ≤ εd →
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧
    ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
          ¬ EndsWell M bad P.D settings.disagreementRate
            (searchingProbes (O.read · ω) settings.walkStart result.hypothesis)
            settings.walkStart
            (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
            (3 * εd / 16) result.ending} ∂μ
      ≤ ENNReal.ofReal δ

/-- `Level5Guarantee` at every target, oracle, goodness level, disagreement rate, probe
distribution, confidence and start `k`. -/
def Level5Correct : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k : ℕ),
    Level5Guarantee M O θ εd P δ k

end OrthoDFA

end
