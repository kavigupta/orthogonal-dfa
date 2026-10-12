import OrthoDFA.Level4Family
import Mathlib.MeasureTheory.Constructions.UnitInterval

/-!
# The learner's loop

Each round of `indecisiveLStarLoop` clusters a suffix family over the populations so far, runs
level 4 of indecisive L* reading that family, and stops at the first round that does not end in a
harvest. A harvest's strings become a population of every later round. The first population is
the probes' own distribution.

The clustering is the one `ClusteringQualityGuarantee` is about: the family is `clusterAt` at a
state of the schedule at which `ret` returns. Level 4 reads the family without its seed, at the
gate's cut: a string reads accept at `hi + 1` or more of the family's votes and reject at `lo` or
fewer. Without its seed the family is suffix draws of one length, so it is suffix-free.

Known modelling gap. Every round, and within a round the clustering and level 4, read fresh noise,
where the Python's noise is fixed per string. The strings two phases both read are a harvested
string, which later rounds read again on its own (the seed's column) and after new suffixes, and
a probe's prefix that is also one of the round's clustered prefixes.

Known modelling gap. A harvest population draws its strings uniformly from the harvested list,
repeats included. `_accumulate_harvest` keeps up to `target` distinct new strings, and
`HarvestSource` draws more by replaying the round's hypothesis.

Known modelling gap. The Python reads the family with its seed, at `pst.accept_thresh` and
`pst.reject_thresh`.

Known modelling gap. `counterexample_driven_synthesis` also runs the certificate, halves the FNR
limit, stops on a stall or on certificate patience, adds per-state populations, keeps one uniform
pool across rounds and returns the best round. Here the uniform pool is drawn afresh each round
and none of the rest is modelled.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : Nonempty State := ⟨⟨0, 0, 0, 0, 0, 0, 0⟩⟩

/-- A harvest's strings, each weighted by how often it was harvested. -/
def harvestMeasure (zs : List (FreeMonoid α)) : Measure (FreeMonoid α) :=
  if zs = [] then Measure.dirac 1
  else ((zs.length : ℝ≥0∞)⁻¹) • (zs.map Measure.dirac).sum

/-- A draw from `harvestMeasure zs`, made by a uniform `u`. -/
def pickFrom (zs : List (FreeMonoid α)) (u : unitInterval) : FreeMonoid α :=
  zs.getD ⌊(u : ℝ) * zs.length⌋₊ 1

/-- Round `r`'s populations: the probes' distribution, `none`, and each earlier round's harvest. -/
def loopPopulations (K r : ℕ) : Finset (Option (Fin K)) :=
  insert none ((Finset.univ.filter fun i : Fin K => i.val < r).image some)

/-- The populations' distributions, given the harvests so far. -/
def loopDistributions {K : ℕ} (D : Measure (FreeMonoid α))
    (harvests : List (List (FreeMonoid α))) : Option (Fin K) → Measure (FreeMonoid α)
  | none => D
  | some i => harvestMeasure (harvests.getD i [])

/-- The clustering's draws: its noise, the suffixes, the uniform pool's prefixes and certification
prefixes, and the uniform picks by which each harvest population draws its prefixes and
certification prefixes. -/
abbrev ClusterDraws (Ω α : Type*) (K : ℕ) :=
  Ω × ((ℕ → FreeMonoid α) × (((ℕ → FreeMonoid α) × (ℕ → FreeMonoid α))
    × ((Fin K → ℕ → unitInterval) × (Fin K → ℕ → unitInterval))))

/-- A round's draws: the clustering's, then level 4's noise and probes. -/
abbrev LoopRoundDraws (Ω α : Type*) (K P : ℕ) :=
  ClusterDraws Ω α K × (Ω × (Fin P → FreeMonoid α))

variable {Ω : Type*} [MeasurableSpace Ω]

def clusterLaw (μ : Measure Ω) (D Dsf : Measure (FreeMonoid α)) (K : ℕ) :
    Measure (ClusterDraws Ω α K) :=
  μ.prod ((Measure.infinitePi fun _ : ℕ => Dsf).prod
    (((Measure.infinitePi fun _ : ℕ => D).prod (Measure.infinitePi fun _ : ℕ => D)).prod
      (Measure.prod
        (Measure.pi fun _ : Fin K =>
          Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval))
        (Measure.pi fun _ : Fin K =>
          Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)))))

/-- `K` rounds' draws, independent, the probes and the uniform pool drawn from `D`. -/
def loopMeasure (μ : Measure Ω) (D Dsf : Measure (FreeMonoid α)) (K P : ℕ) :
    Measure (Fin K → LoopRoundDraws Ω α K P) :=
  Measure.pi fun _ : Fin K => (clusterLaw μ D Dsf K).prod (μ.prod (Measure.pi fun _ : Fin P => D))

/-- The clustering's run after `harvests`, each harvest population's draws picked from its
strings. -/
def clusterRun {K : ℕ} (harvests : List (List (FreeMonoid α))) (c : ClusterDraws Ω α K) :
    Run Ω (FreeMonoid α) (Option (Fin K)) :=
  (c.1, ((c.2.1, fun j => match j with
      | none => c.2.2.1.1
      | some i => fun n => pickFrom (harvests.getD i []) (c.2.2.2.1 i n)),
    fun j => match j with
      | none => c.2.2.1.2
      | some i => fun n => pickFrom (harvests.getD i []) (c.2.2.2.2 i n)))

/-- The clustering's schedule, FNR limit and gate level, and level 4's settings. -/
structure LoopSettings (K : ℕ) where
  schedule : Finset (Option (Fin K)) → Finset State
  indecisionLimit : ℝ
  gateLevel : ℝ
  level4 : Level4Settings

/-- A round: the clustering's run, the state it returned at and its family, level 4's reads, and
what level 4 returned. -/
structure LoopRound (Ω α : Type*) (K : ℕ) where
  run : Run Ω (FreeMonoid α) (Option (Fin K))
  state : State
  family : Finset (FreeMonoid α)
  read : FreeMonoid α → ARU
  result : Level4Result α

variable {μ : Measure Ω}

/-- The round after `harvests`. -/
def loopRound {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (harvests : List (List (FreeMonoid α)))
    (d : LoopRoundDraws Ω α K settings.level4.probeBudget) : LoopRound Ω α K :=
  let pops := loopPopulations K harvests.length
  let x := clusterRun harvests d.1
  let B := Classical.epsilon fun B => B ∈ settings.schedule pops
    ∧ x ∈ ret O.mq pops none settings.indecisionLimit settings.gateLevel B
  let F := clusterAt O.mq pops x B
  let read := fun z => familyRead O.mq F B.lo (B.hi + 1) z d.2.1
  ⟨x, B, F, read, indecisiveLStarLevel4 settings.level4 read d.2.2⟩

def RoundEnd.harvested : RoundEnd α → Option (List (FreeMonoid α))
  | .harvest zs => some zs
  | _ => none

/-- The first `r` rounds, cut short at a round that does not end in a harvest. -/
def loopRounds {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget) :
    ℕ → List (LoopRound Ω α K)
  | 0 => []
  | r + 1 =>
    if h : r < K ∧ ∀ o ∈ loopRounds settings O draws r, o.result.ending.harvested ≠ none then
      loopRounds settings O draws r
        ++ [loopRound settings O ((loopRounds settings O draws r).filterMap
          fun o => o.result.ending.harvested) (draws ⟨r, h.1⟩)]
    else loopRounds settings O draws r

/-- The rounds the loop runs, at most `K`. -/
def indecisiveLStarLoop {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level4.probeBudget) : List (LoopRound Ω α K) :=
  loopRounds settings O draws K

open scoped Classical in
/-- The chance `F`'s read of `p`, cut at `kl` and `kh`, is decided against `p`'s membership. -/
def misreadProb (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (p : FreeMonoid α) : ℝ :=
  μ.real {ω | familyRead O.mq F kl kh p ω = if p ∈ O.L then .reject else .accept}

/-- The round ended consistent; its read decides a draw from `D` against the target at most `εr`
of the time and leaves it undecided at most `εu`, and its hypothesis's walk from `k` and its tree
part on at most `εd` of `D`. -/
def LoopRound.Agrees {K : ℕ} (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α)) (k : ℕ)
    (εr εu εd : ℝ) (o : LoopRound Ω α K) : Prop :=
  o.result.ending = .consistent
    ∧ ∫ p, misreadProb O o.family o.state.lo (o.state.hi + 1) p ∂D ≤ εr
    ∧ ∫ p, readProb O o.family o.state.lo (o.state.hi + 1) p .undecided ∂D ≤ εu
    ∧ D.real (searchingProbes o.read k o.result.hypothesis) ≤ εd

/-- The learner's loop terminates in agreement. Within `|Q| + 1` rounds, the loop stops at a
round whose hypothesis's walk and tree part on at most `εd` of the probes, and whose family's
read of a probe is decided against the target at most `εcov + 2·indecisionLimit + 2·slack` of the
time and undecided at most `2·indecisionLimit + slack`; or some round harvests a list too
concentrated for the clustering, its collision mass above `cap`. It fails to do either with
probability at most `(|Q| + 1)(δc + 2δ)`: per round, the clustering's `δc` and level 4's `2δ`.

The clustering's parameters are `ClusteringQualityGuarantee`'s, the uniform pool drawn from the
probes' distribution, and the suffixes of one length `Ls`, longer than any string level 4 reads.
Level 4's are `Level4Guarantee`'s: its `θ` is set so that a harvested state some later family
leaves undecided more than `1.5θ` of the time holds more of the harvest than the FNR gate lets
through, and its `ε` is the clustering's `crossLimit`. -/
def LearnerLoopCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (K S k Ls : ℕ) (η₀ indecisionLimit gateLevel crossLimit εcov δc slack pAP εd δ : ℝ),
  let n : ℝ := Fintype.card σ + S
  let A : ℝ := Fintype.card α
  let ρ : ℝ := (S + 1) * εd / (256 * (n + 1) * (Fintype.card σ * A + S + 1))
  let Kp : ℝ := (n + 1) * (2 * (n + 3) * A + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
    + 8 * (n + 3) * A * (P.L + 1)
  let X : ℝ := 4 * (2 ^ (Fintype.card σ + S + 4) * A + 7 * Kp ^ 2) / δ
  let G : ℝ := Real.log X + 2 * Real.log (2 * Real.log X)
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < indecisionLimit → indecisionLimit ≤ 1 / 2 →
  0 < gateLevel → gateLevel < 1 / 2 →
  0 < crossLimit →
  0 < εcov → εcov ≤ 1 →
  0 < δc → δc ≤ 1 →
  0 < slack → 0 < pAP →
  0 < εd → εd ≤ 1 → 0 < δ → δ ≤ 1 → 1 ≤ P.L →
  Fintype.card σ + 1 ≤ K →
  P.L + Fintype.card σ + S + 3 ≤ Ls →
  512 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * indecisionLimit + slack) ≤ 1 →
  (P.L + 1) * (n + 1) * crossLimit + 3 * unionLog (α := α) σ S δ * prefixMax P.D k ≤ ρ →
  ∃ cap : ℝ, 0 < cap ∧ ∃ settings : LoopSettings K,
    settings.indecisionLimit = indecisionLimit ∧ settings.gateLevel = gateLevel ∧
    settings.level4.walkStart = k ∧ settings.level4.disagreementRate = εd ∧
    (settings.level4.probeBudget : ℝ) ≤ Kp * G ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ Ls} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      collisionMass P.D ≤ cap → collisionMass Dsf ≤ cap →
      (loopMeasure μ P.D Dsf K settings.level4.probeBudget).real
          {draws | let rounds := indecisiveLStarLoop settings O draws
            ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                  o.Agrees O P.D k (εcov + 2 * indecisionLimit + 2 * slack)
                    (2 * indecisionLimit + slack) εd)
              ∨ ∃ o ∈ rounds, ∃ zs, o.result.ending = .harvest zs
                  ∧ cap < collisionMass (harvestMeasure zs))}
        ≤ (Fintype.card σ + 1) * (δc + 2 * δ)

end OrthoDFA

end
