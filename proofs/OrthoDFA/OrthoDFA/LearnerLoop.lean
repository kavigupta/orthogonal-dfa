import OrthoDFA.Level5Family
import Mathlib.MeasureTheory.Constructions.UnitInterval

/-!
# The learner's loop

Each round of `indecisiveLStarLoop` clusters a suffix family over the populations so far, runs
level 5 of indecisive L* reading that family, and stops at the first round that does not end in a
harvest. A harvest's sampler, scaled to a probability, becomes a population of every later round:
it draws a fresh probe from the probes' distribution `D`, walks it under the round's last
hypothesis and takes its undecided strings at the harvest's site, as the round reads them. The first population is `D` itself. A population's
draws are made from uniforms (`drawFrom`).

The clustering is the one `ClusteringQualityGuarantee` is about: the family is `clusterAt` at a
state of the schedule at which `ret` returns. Level 5 reads the family without its seed, at the
gate's cut: a string reads accept at `hi + 1` or more of the family's votes and reject at `lo` or
fewer. Without its seed the family is suffix draws of one length, so it is suffix-free.

The whole run reads one noise, fixed per string as the Python's is.

Known modelling gap. Level 5 draws its probes from `D` conditioned on their first `walkStart`
letters avoiding those of every prefix the round's clustering read, so that it never reads a string
the clustering read. The Python draws them from `D`.

Known modelling gap. The uniform pool is drawn afresh each round, where the Python keeps one pool
across rounds; on one noise a kept pool's prefixes would be read again by every round.

Known modelling gap. The gate draws its certification pool to at most twice its size. The
Python caps each verdict's extra draws at `certification_budget`, but `sample_suffix_family`'s
outer resampling loop is unbounded.

Known modelling gap. The Python reads the family with its seed, at `pst.accept_thresh` and
`pst.reject_thresh`.

Known modelling gap. `counterexample_driven_synthesis` also runs the certificate, halves the FNR
limit, stops on a stall or on certificate patience, adds per-state populations and returns the
best round. None of these is modelled.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : Nonempty State := ⟨⟨0, 0, 0, 0, 0, 0, 0⟩⟩

/-- Uniform over the strings of length `n`. -/
def uniformStrings (n : ℕ) : Measure (FreeMonoid α) :=
  ((Fintype.card α : ℝ≥0∞) ^ n)⁻¹
    • ∑ v : List.Vector α n, Measure.dirac (FreeMonoid.ofList v.toList)

/-- `ν` scaled to a probability, or a point mass where it has no mass or infinite mass. -/
def popLaw (ν : Measure (FreeMonoid α)) : Measure (FreeMonoid α) :=
  if ν Set.univ = 0 ∨ ν Set.univ = ∞ then Measure.dirac 1 else (ν Set.univ)⁻¹ • ν

/-- A numbering of the strings, one number each. -/
def strCode : FreeMonoid α → ℕ := Classical.choose (exists_injective_nat (FreeMonoid α))

open scoped Classical in
/-- The string a number numbers, if any. -/
def strAt (n : ℕ) : Option (FreeMonoid α) :=
  if h : ∃ z, strCode z = n then some h.choose else none

/-- The mass `ν` puts on the strings numbered below `n`. -/
def massBelow (ν : Measure (FreeMonoid α)) (n : ℕ) : ℝ :=
  ∑ i ∈ Finset.range n, ν.real {z | strCode z = i}

open scoped Classical in
/-- A draw from `ν` made by a uniform `u`: the first string at which `ν`'s running mass passes
`u`. -/
def drawFrom (ν : Measure (FreeMonoid α)) (u : unitInterval) : FreeMonoid α :=
  if h : ∃ n, (u : ℝ) < massBelow ν (n + 1) then (strAt (Nat.find h)).getD 1 else 1

/-- Round `r`'s populations: the probes' distribution, `none`, and each earlier round's harvest. -/
def loopPopulations (K r : ℕ) : Finset (Option (Fin K)) :=
  insert none ((Finset.univ.filter fun i : Fin K => i.val < r).image some)

/-- The populations' distributions, given the earlier harvests' populations. -/
def loopDistributions {K : ℕ} (D : Measure (FreeMonoid α))
    (pops : List (Measure (FreeMonoid α))) : Option (Fin K) → Measure (FreeMonoid α)
  | none => D
  | some i => pops.getD i (Measure.dirac 1)

/-- `ν` conditioned off `T`. -/
def offLaw (ν : Measure (FreeMonoid α)) (T : Finset (FreeMonoid α)) : Measure (FreeMonoid α) :=
  popLaw (ν.restrict (↑T)ᶜ)

/-- What the earlier rounds hand on: their harvests' populations, and every prefix and suffix
their clusterings read. -/
structure LoopHist (α : Type*) where
  pops : List (Measure (FreeMonoid α))
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- The uniforms by which the clustering draws its suffixes, and each population its prefixes and
certification prefixes. -/
abbrev ClusterRest (α : Type*) (K : ℕ) :=
  (ℕ → unitInterval) × ((Option (Fin K) → ℕ → unitInterval) × (Option (Fin K) → ℕ → unitInterval))

/-- The clustering's draws: the noise and the rest. -/
abbrev ClusterDraws (Ω α : Type*) (K : ℕ) := Ω × ClusterRest α K

/-- A round's draws apart from the noise: the clustering's, and the uniforms by which level 5
draws its probes. -/
abbrev RoundDraws (α : Type*) (K P : ℕ) := ClusterRest α K × (Fin P → unitInterval)

/-- A round's draws with the noise it reads. -/
abbrev LoopRoundDraws (Ω α : Type*) (K P : ℕ) := Ω × RoundDraws α K P

def restLaw (α : Type*) (K : ℕ) : Measure (ClusterRest α K) :=
  (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
    ((Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
      (Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)))

def roundLaw (α : Type*) (K P : ℕ) : Measure (RoundDraws α K P) :=
  (restLaw α K).prod (Measure.pi fun _ : Fin P => (volume : Measure unitInterval))

variable {Ω : Type*} [MeasurableSpace Ω]

def clusterLaw (μ : Measure Ω) (α : Type*) (K : ℕ) : Measure (ClusterDraws Ω α K) :=
  μ.prod (restLaw α K)

/-- One noise for the whole run, which every string's read keeps, and `K` rounds' uniforms,
independent. -/
def loopMeasure (μ : Measure Ω) (α : Type*) (K P : ℕ) :
    Measure (Ω × (Fin K → RoundDraws α K P)) :=
  μ.prod (Measure.pi fun _ : Fin K => roundLaw α K P)

/-- The rounds' draws, each read through the run's one noise. -/
def persistDraws {K P : ℕ} (p : Ω × (Fin K → RoundDraws α K P)) :
    Fin K → LoopRoundDraws Ω α K P :=
  fun i => (p.1, p.2 i)

/-- The clustering's run after the history `h`: suffixes from `Dsf` and each population's
prefixes, all off what the earlier rounds read. -/
def clusterRun {K : ℕ} (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (c : ClusterDraws Ω α K) : Run Ω (FreeMonoid α) (Option (Fin K)) :=
  (c.1, ((fun n => drawFrom (offLaw Dsf h.suffs) (c.2.1 n),
      fun j n => drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.1 j n)),
    fun j n => drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.2 j n)))

/-- The clustering's schedule, FNR limit and gate level, and level 5's settings. -/
structure LoopSettings (K : ℕ) where
  schedule : Finset (Option (Fin K)) → Finset State
  indecisionLimit : ℝ
  gateLevel : ℝ
  level5 : Level5Settings

/-- A round: the clustering's run, the state it returned at and its family, level 5's reads, what
level 5 returned, and the prefixes and suffixes its clustering read. -/
structure LoopRound (Ω α : Type*) (K : ℕ) where
  run : Run Ω (FreeMonoid α) (Option (Fin K))
  state : State
  family : Finset (FreeMonoid α)
  read : FreeMonoid α → ARU
  result : Level5Result α
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

variable {μ : Measure Ω}

/-- Every prefix the clustering reads at a state of `sched`: each population's draws, and its
certification draws off them at every count the state's tests read. -/
def roundPrefs {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Finset (FreeMonoid α) :=
  sched.biUnion fun B => prefixesAt pops B.npref x
    ∪ pops.biUnion fun j => (Finset.range (B.npref + 1)).biUnion fun e =>
      certOf (prefixesAt pops B.npref x) j (B.npref + e) x

/-- Every suffix the clustering reads at a state of `sched`. -/
def roundSuffs (sched : Finset State) {J : Type*} (x : Run Ω (FreeMonoid α) J) :
    Finset (FreeMonoid α) :=
  sched.biUnion fun B => poolAt B.nsuff x

/-- The first `k` letters of every prefix the clustering reads at a state of `sched`. -/
def clusterKeys {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K))) (k : ℕ)
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Finset (FreeMonoid α) :=
  (roundPrefs sched pops x).image fun p => prefixOf p k

/-- `D` conditioned on the probes whose first `k` letters avoid `K₀`. -/
def avoidLaw (D : Measure (FreeMonoid α)) (k : ℕ) (K₀ : Finset (FreeMonoid α)) :
    Measure (FreeMonoid α) :=
  popLaw (D.restrict {x | k ≤ x.toList.length → prefixOf x k ∉ K₀})

/-- The round after the history `h`. Its clustering draws off what the earlier rounds read, and
level 5 draws its probes from `D` avoiding the first `k` letters of the prefixes the clustering
read. -/
def loopRound {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level5.probeBudget) : LoopRound Ω α K :=
  let ps := loopPopulations K h.pops.length
  let x := clusterRun D Dsf h (d.1, d.2.1)
  let B := Classical.epsilon fun B => B ∈ settings.schedule ps
    ∧ x ∈ ret O.mq ps none settings.indecisionLimit settings.gateLevel B
  let F := clusterAt O.mq ps x B
  let read := fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1
  let K₀ := clusterKeys (settings.schedule ps) ps settings.level5.walkStart x
  ⟨x, B, F, read, indecisiveLStarLevel5 settings.level5 read
    (fun i => drawFrom (avoidLaw D settings.level5.walkStart K₀) (d.2.2 i)),
    roundPrefs (settings.schedule ps) ps x, roundSuffs (settings.schedule ps) x⟩

def RoundEnd.harvested {H : Type*} : RoundEnd H → Option H
  | .harvest h => some h
  | _ => none

/-- The population a round's harvest hands on: its sampler's law over probes from `D`, walked
from `k`, scaled to a probability. -/
def LoopRound.population {K : ℕ} (D : Measure (FreeMonoid α)) (k : ℕ) (o : LoopRound Ω α K) :
    Option (Measure (FreeMonoid α)) :=
  o.result.ending.harvested.map fun site =>
    popLaw (harvestLaw D (fun z => (o.read z).cut) k o.result.hypothesis site)

/-- What the rounds hand on: their harvests' populations, and what their clusterings read. -/
def histOf {K : ℕ} (D : Measure (FreeMonoid α)) (k : ℕ) (rounds : List (LoopRound Ω α K)) :
    LoopHist α :=
  ⟨rounds.filterMap fun o => o.population D k,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

/-- The first `r` rounds, cut short at a round that does not end in a harvest. -/
def loopRounds {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget) :
    ℕ → List (LoopRound Ω α K)
  | 0 => []
  | r + 1 =>
    if h : r < K ∧ ∀ o ∈ loopRounds settings O D Dsf draws r, o.result.ending.harvested ≠ none then
      loopRounds settings O D Dsf draws r
        ++ [loopRound settings O D Dsf
          (histOf D settings.level5.walkStart (loopRounds settings O D Dsf draws r))
          (draws ⟨r, h.1⟩)]
    else loopRounds settings O D Dsf draws r

/-- The rounds the loop runs, at most `K`. -/
def indecisiveLStarLoop {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level5.probeBudget) : List (LoopRound Ω α K) :=
  loopRounds settings O D Dsf draws K

open scoped Classical in
/-- The chance `F`'s read of `p`, cut at `kl` and `kh`, is decided against `p`'s membership. -/
def misreadProb (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (p : FreeMonoid α) : ℝ :=
  μ.real {ω | familyRead O.mq F kl kh p ω = if p ∈ O.L then .reject else .accept}

/-- The round ended consistent; its read decides a draw from `D` against the target at most `εr`
of the time and leaves it undecided at most `εu`, and its hypothesis's walk from `k` and its tree
part, or the walk stops at an edge it could learn, on at most `εd` of `D`. -/
def LoopRound.Agrees {K : ℕ} (O : Oracle μ (FreeMonoid α)) (D : Measure (FreeMonoid α)) (k : ℕ)
    (εr εu εd : ℝ) (o : LoopRound Ω α K) : Prop :=
  o.result.ending = .consistent
    ∧ ∫ p, misreadProb O o.family o.state.lo (o.state.hi + 1) p ∂D ≤ εr
    ∧ ∫ p, readProb O o.family o.state.lo (o.state.hi + 1) p .undecided ∂D ≤ εu
    ∧ D.real (searchingProbes o.read k o.result.hypothesis) ≤ εd

/-- The learner's loop terminates in agreement. Within `|Q| + 1` rounds, the loop stops at a
round whose hypothesis's walk and tree part on at most `εd` of the probes, and whose family's
read of a probe is decided against the target at most `εcov + 2·indecisionLimit + 2·slack` of the
time and undecided at most `2·indecisionLimit + slack`. It fails to with probability at most
`(|Q| + 1)(δc + δ)`: per round, the clustering's `δc` and level 5's `δ`.

The whole run reads one noise. The clustering's parameters are `ClusteringQualityGuarantee`'s,
the uniform pool drawn from the probes' distribution, and the suffixes of one length `Ls`, longer
than any string level 5 reads. Each round's clustering draws its prefixes and suffixes off every
one the earlier rounds' clusterings read, so no two rounds read the same string. Level 5 runs at
disagreement rate `εd / 2` and draws its probes from `D` avoiding the first `k ≥ 1` letters of
every prefix the round's clustering read. `cap` bounds the atoms of the populations and the
suffixes: against it the clustering's collision conditions hold, and what the earlier rounds read
holds little of any population or of the suffixes. A harvest's sampler yields at least `ρ / 4`, and finds a
string at most as often as the probes share its length-`k` prefix, so its population puts at most
`4 prefixMax / ρ` on any one string. Level 5's `θ` is set so that a harvested state some later
family leaves undecided more than `1.5θ` of the time holds more of the population than the FNR
gate lets through, and the band's far side so that level 5 rarely reads a state on its rarer
side. -/
def LearnerLoopCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (K Ls : ℕ) (η₀ indecisionLimit gateLevel εcov δc slack pAP εd δ : ℝ),
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let ρ : ℝ := 3 * εd / 16
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let Kp : ℝ := V * (224 / εd + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let U : ℝ := 4 * (n + 2) * A * Real.log ((n + 3) * A / δ)
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < indecisionLimit → indecisionLimit ≤ 1 / 2 →
  0 < gateLevel → gateLevel < 1 / 2 →
  0 < εcov → εcov ≤ 1 →
  0 < δc → δc ≤ 1 →
  0 < slack → 0 < pAP →
  0 < εd → εd ≤ 1 → 0 < δ → δ ≤ 1 → 1 ≤ P.L →
  Fintype.card σ + 1 ≤ K →
  P.L + Fintype.card σ + 2 ≤ Ls →
  1024 * (P.L + 1) * (n + 1) * Fintype.card σ * (2 * indecisionLimit + slack) ≤ 1 →
  ∃ cap : ℝ, 0 < cap ∧ ∀ k : ℕ, 1 ≤ k → prefixMax P.D k ≤ cap * ρ / 4 →
    4096 * (P.L + 1) * U * prefixMax P.D k ≤ εd →
  ∃ settings : LoopSettings K,
    settings.indecisionLimit = indecisionLimit ∧ settings.gateLevel = gateLevel ∧
    settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ Ls} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ v, Dsf.real {v} ≤ cap) →
      (loopMeasure μ α K settings.level5.probeBudget).real
          {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
            ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                o.Agrees O P.D k (εcov + 2 * indecisionLimit + 2 * slack)
                  (2 * indecisionLimit + slack) εd)}
        ≤ (Fintype.card σ + 1) * (δc + δ)

/-- The learner's loop at settings the theorem chooses, every hypothesis an explicit inequality.
With probability at least `1 − δ`, within `|Q| + 1` rounds the loop stops at a round whose
hypothesis's walk and tree part on at most `εd` of the probes, and whose family's read of a probe
is decided against the target at most `εcov` of the time and undecided at most `εcov` of the
time. Level 5 draws at most `Kp · G` probes a round.

The theorem chooses `indecisionLimit = slack = ι`, the FNR gate at `1/4`, `|Q| + 1` rounds and
suffixes of length `L + |Q| + 3`, and splits `δ` evenly over the rounds' clustering and level 5.
The clustering runs at half of `εcov`, `ι` and `pAP`, the other halves paying for what the earlier
rounds read, and its band's far side at `δ' / (4 Kp G (L + 1)(|Q| + 3)|Σ|^(|Q|+2))`, so that level
5 rarely reads a state on its rarer side. `PB` and `RB` bound its suffix pool and prefix count;
`cap` is linear in them, and holds the clustering's collision conditions and what the
earlier rounds read of any population or of the suffixes. `ρ / 4` is what a harvest's sampler
yields, so a population puts at most `4 prefixMax / ρ` on a string. The last hypothesis holds
level 5's noise fields, which concentrate over the probes' length-`k` prefixes. -/
def LearnerLoopSimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov εd δ pAP : ℝ) (k : ℕ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let L : ℝ := P.L
  let δ' : ℝ := δ / (2 * (Q + 1))
  let ι : ℝ := min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))
  let ρ : ℝ := 3 * εd / 16
  let s : ℝ := 1 / 2 - η₀
  let τ : ℝ := min (εcov / 2) (min s ι)
  let J : ℝ := Q + 2
  let V : ℝ := (Q + 2) * ((Q + 2) * A + 1)
  let Kp : ℝ := V * (224 / εd + 1)
  let G : ℝ := Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
  let PB : ℝ := 256 * (Real.log (2 / τ)
      + 8 * Kp * G * (L + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ') / (s ^ 2 * pAP)
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
  let RB : ℝ := 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 3)
  let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
      (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * (L + 1)))))
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < εcov → εcov ≤ 1 → 0 < εd → εd ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 1 ≤ P.L → 1 ≤ k →
  prefixMax P.D k ≤ cap * ρ / 4 →
  16384 * (L + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * prefixMax P.D k ≤ εd →
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
      Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ v, Dsf.real {v} ≤ cap) →
      (loopMeasure μ α (Fintype.card σ + 1) settings.level5.probeBudget).real
          {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
            ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
                o.Agrees O P.D k εcov εcov εd)}
        ≤ δ

/-- The learner's loop on uniform probes and suffixes, once the probes are at least `L₀` long.
With probes uniform over the strings of length `L` and suffixes uniform over those of length
`L + |Q| + 3`, of which at least `pAP` keep every string's membership, `L ≥ L₀` makes the loop,
walking from `L − 1`, stop within `|Q| + 1` rounds at a round whose hypothesis's walk and tree
part on at most `εd` of the probes and whose read is wrong at most `εcov` and undecided at most
`εcov` of the time, but for chance `δ`. `L₀` is logarithmic in `1/δ`, `1/εd`, `1/εcov`,
`1/(1/2 − η₀)`, `|Q|` and `1/pAP`, over `log |Σ|`. -/
def LearnerLoopUniform : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov εd δ pAP : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let L : ℝ := P.L
  let s : ℝ := 1 / 2 - η₀
  let δ' : ℝ := δ / (2 * (Q + 1))
  let ρ : ℝ := 3 * εd / 16
  let V : ℝ := (Q + 2) * ((Q + 2) * A + 1)
  let Kp : ℝ := V * (224 / εd + 1)
  let G : ℝ := Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
  let Θ : ℝ := 1 / s + 8 / εcov + 3072 * Q * (Q + 1)
  let P₀ : ℝ := 256 * (2 * Θ + 8 * Kp * G * (Q + 3) * A ^ (Fintype.card σ + 2) / δ')
    / (s ^ 2 * pAP)
    + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
  let R₀ : ℝ := 6000 * (Q + 2) ^ 2 * Θ ^ 3
    * Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2))) / s ^ 4
  let E : ℝ := 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 1) / pAP + A ^ 2) / (δ' * ρ)
  let L₀ : ℝ := 2 + 28 * (Real.log E + 24) / (25 * Real.log A)
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 →
  0 < εcov → εcov ≤ 1 → 0 < εd → εd ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 2 ≤ Fintype.card α →
  P.D = uniformStrings P.L →
  pAP ≤ (uniformStrings (P.L + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  L₀ ≤ L →
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level5.walkStart = P.L - 1 ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    (loopMeasure μ α (Fintype.card σ + 1) settings.level5.probeBudget).real
      {p | let rounds := indecisiveLStarLoop settings O P.D
            (uniformStrings (P.L + Fintype.card σ + 3)) (persistDraws p)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.Agrees O P.D (P.L - 1) εcov εcov εd)}
      ≤ δ

end OrthoDFA

end
