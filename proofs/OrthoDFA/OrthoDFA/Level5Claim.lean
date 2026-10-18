import OrthoDFA.Level5
import OrthoDFA.ProbeDistribution

/-!
# Level 5's claim

The oracle (`TriOracle`): every string's read is accept, reject or undecided, independently across
strings, with a distribution that depends only on the string's state in the target DFA. A
read-state is good where it reads undecided at most `1.5θ` of the time.

`Level5Correct`: over `settings.probeBudget` probes drawn from a `PrefixDistribution`,
`indecisiveLStarLevel5` ends well (`EndsWell`: consistent with at most `3εd/2` of the probes
checked, or a harvest whose sampler finds more than half of what it finds at read-states that are
not good, is spread and yields) but for chance at most `δ + 32ū/εd`, at settings the proof
chooses, once the reads' `κ` and `ε`, the goodness level and the probes' prefix mass are small
enough. A checked probe searches or learns, reads a prefix followed by a midfix of its tree on its
state's rarer side or, where the states' undecided rates make `undecidedMass` at most `ū`, reads
one undecided. It is derived from a detailed bound (`Proofs/Level5RiskDefs.lean`) that need not be
read to trust it.

`Level5CorrectAt` lets the walk start before the depth the noise concentrates from, as the Python's
walk from `0` needs: the strings read before that depth are each checked instead, at a cost
linear in `θ` added to the failure chance.
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

/-- The probes that search or learn: their walk from `k` along `H`'s edges and the sift disagree
at a decided read, or the walk stops at an unlearned edge it could learn. -/
def searchingProbes (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Set (FreeMonoid α) :=
  {x | match probeBy (fun z => (read z).cut) H.tree (Function.curry H.edges) k x with
    | .undecidedMid _ | .edge _ _ | .member _ => True
    | _ => False}

/-- The words of at most `n` letters. -/
noncomputable def shortWords (n : ℕ) : Finset (FreeMonoid α) :=
  Finset.univ.image fun v : (Σ i : Fin (n + 1), Fin i → α) => FreeMonoid.ofList (List.ofFn v.2)

/-- The side a read distribution reads on more often. -/
def sideOf (d : ARU → ℝ) : Bool := decide (d .reject < d .accept)

/-- The probes a prefix of which from `k` on, followed by a midfix of `H`'s tree, reads on its
state's rarer side under `dist`. -/
def rarerProbes {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) : Set (FreeMonoid α) :=
  {x | ∃ j, k ≤ j ∧ j ≤ x.toList.length ∧ ∃ m ∈ H.tree.midfixes,
    (read (prefixOf x j * m)).cut = some (!sideOf (dist (M.eval (prefixOf x j * m).toList)))}

/-- The probes at most `J` long a prefix of which from `k` on, followed by at most `n` letters,
reads undecided. -/
def undecidedProbes (read : FreeMonoid α → ARU) (k n J : ℕ) : Set (FreeMonoid α) :=
  {x | x.toList.length ≤ J ∧ ∃ j, k ≤ j ∧ j ≤ x.toList.length ∧ ∃ m ∈ shortWords n,
    read (prefixOf x j * m) = .undecided}

/-- The reads `undecidedProbes` looks at that a probe makes undecided, on average, at the states'
undecided rates. -/
def undecidedMass {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (D : MeasureTheory.Measure (FreeMonoid α)) (k n J : ℕ) : ℝ :=
  ∫ x, ∑ j ∈ Finset.Icc k (min x.toList.length J), ∑ m ∈ shortWords n,
    dist (M.eval (prefixOf x j * m).toList) .undecided ∂D

/-- The probes a consistent round answers for: those that search or learn, those that read on a
rarer side and, once `undecidedMass` is at most `ū`, those that read undecided. -/
def checkedProbes {σ : Type*} (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (D : MeasureTheory.Measure (FreeMonoid α)) (read : FreeMonoid α → ARU) (k n J : ℕ) (ū : ℝ)
    (H : HypothesisDFA α) : Set (FreeMonoid α) :=
  searchingProbes read k H ∪ rarerProbes M dist read k H
    ∪ {x | undecidedMass M dist D k n J ≤ ū ∧ x ∈ undecidedProbes read k n J}

/-- The largest mass the probes' distribution puts on one length-`k` prefix. -/
def prefixMax (D : Measure (FreeMonoid α)) (k : ℕ) : ℝ := ⨆ u, D.real {x | prefixOf x k = u}

/-- `Level5Guarantee M O θ εd P δ k J ū`: some settings, walking from `k`, with disagreement rate
`εd` and a budget of at most `K · G` probes, make level 5 fail to end well, a harvest's sampler
yielding `3εd/16` and a consistent round checking at most `3εd/2` of the probes, with chance at
most `δ + 32ū/εd`, once the goodness level `θ`, the reads' `κ` and `ε`
and the probes' prefix mass at `k` are small enough for `n = |Q|`, `A = |Σ|` and the draws'
length `L`. `V` bounds the hypotheses a round goes through. A read on its state's rarer side
among the strings a step reads splits wrongly, and `W` bounds those strings: a probe's at most
`(L + 1)(n + 2)` prefixes followed by midfixes, and a split's at most `(n + 2)² A` re-sifts. `U`
bounds the log of the trees, edge maps and edges a noise field is united over, over `δ`. -/
def Level5Guarantee [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : TriOracle μ M)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k J : ℕ) (ū : ℝ) : Prop :=
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
  1024 * (P.L + 1) * U * prefixMax P.D k ≤ εd → 0 ≤ ū →
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧
    ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
          ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
            (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
              (Fintype.card σ + 2) J ū result.hypothesis)
            settings.walkStart
            (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
            (3 * εd / 16) result.ending} ∂μ
      ≤ ENNReal.ofReal (δ + 32 * ū / εd)

/-- `Level5Guarantee` with the noise fields concentrating over the probes' first `kd` letters
rather than their first `k`: the prefix mass is read at `kd`, and the round may also fail where a
string it reads at a position before `kd`, at a good read-state, reads undecided. The steps read at
most `K · G · W` strings, each undecided at a good read-state with chance at most `1.5θ`, and the
final tree reads before `kd` at most `H (n + 2)`, each also on its rarer side with chance at most
`ε + κ`, `H` the strings of length `k` to `kd − 1`. At `kd = k` it is `Level5Guarantee`. -/
def Level5GuaranteeAt [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : TriOracle μ M)
    (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k kd J : ℕ) (ū : ℝ) : Prop :=
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let K : ℝ := V * (112 / εd + 1)
  let W : ℝ := (n + 2) * (P.L + 1 + (n + 2) * A)
  let U : ℝ := 4 * (n + 2) * A * Real.log ((n + 3) * A / δ)
  let H : ℝ := ∑ i ∈ Finset.Ico k kd, A ^ i
  let bad := fun q => ¬ O.Good θ q
  0 < δ → δ ≤ 1 → 0 < εd → εd < 1 → 1 ≤ P.L → 0 ≤ θ →
  384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
  0 ≤ O.κ → 0 ≤ O.ε → 4 * (O.ε + O.κ) * K * G * W ≤ δ →
  1024 * (P.L + 1) * U * prefixMax P.D kd ≤ εd → 0 ≤ ū →
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧
    ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
          ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
            (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
              (Fintype.card σ + 2) J ū result.hypothesis)
            settings.walkStart
            (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
            (3 * εd / 16) result.ending} ∂μ
      ≤ ENNReal.ofReal (δ + 32 * ū / εd + if k < kd then
          3 / 2 * θ * (K * G * W) + (O.ε + O.κ + 3 / 2 * θ) * (H * (n + 2)) else 0)

/-- `Level5Guarantee` at every target, oracle, goodness level, disagreement rate, probe
distribution, confidence, start `k`, length `J` and undecided mass `ū`. -/
def Level5Correct : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k J : ℕ) (ū : ℝ),
    Level5Guarantee M O θ εd P δ k J ū

/-- `Level5GuaranteeAt` at every target, oracle, goodness level, disagreement rate, probe
distribution, confidence, start `k`, concentration depth `kd`, length `J` and undecided mass `ū`. -/
def Level5CorrectAt : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ εd : ℝ) (P : PrefixDistribution α) (δ : ℝ) (k kd J : ℕ) (ū : ℝ),
    Level5GuaranteeAt M O θ εd P δ k kd J ū

end OrthoDFA

end
