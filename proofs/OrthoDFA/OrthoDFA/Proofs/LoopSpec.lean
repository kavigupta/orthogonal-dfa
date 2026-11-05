import OrthoDFA.Code.LearnerLoop

/-!
# The loop as functions of the oracle

The loop of `OrthoDFA.Code.LearnerLoop` written as functions of the oracle and its noise rather
than as a program: a sampler and a round carry their family's read, and each round may read its
own noise. `loopQ`'s run against one noise's answers is `loopRounds` at that noise in every round
(`OrthoDFA.Proofs.LoopQuery`), so what is proved of these is proved of the loop.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- What a harvested population hands on to draw from it: its round's family read, its last
hypothesis, and the level-3 site of its harvest. -/
structure SpecSampler (α : Type*) [Countable α] where
  read : FreeMonoid α → ARU
  hyp : HypothesisDFA α
  site : Random.Level3Site

/-- The populations' distributions, given the earlier harvests' samplers: the probes' `D`, and
each harvested population's sampler law over probes from `D` walked from `k`, as a probability. -/
def loopDistributions {K : ℕ} (D : Measure (FreeMonoid α)) (k : ℕ)
    (pops : List (SpecSampler α)) : Option (Fin K) → Measure (FreeMonoid α)
  | none => D
  | some i =>
    match pops[i.val]? with
    | some s => popLaw (Random.harvestLaw s.read D k s.hyp s.site)
    | none => Measure.dirac 1

/-- What the earlier rounds hand on: their harvests' samplers, and every prefix and suffix
their clusterings read. -/
structure SpecHist (α : Type*) [Countable α] where
  pops : List (SpecSampler α)
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- The clustering's draws: the noise and the rest. -/
abbrev ClusterDraws (Ω α : Type*) (K N : ℕ) := Ω × ClusterRest α K N

/-- A round's draws with the noise it reads. -/
abbrev LoopRoundDraws (Ω α : Type*) (K N P : ℕ) := Ω × RoundDraws α K N P

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The rounds' draws, each read through the run's one noise. -/
def persistDraws {K N P : ℕ} (p : Ω × (Fin K → RoundDraws α K N P)) :
    Fin K → LoopRoundDraws Ω α K N P :=
  fun i => (p.1, p.2 i)

/-- Population `j`'s first `M` draws off `T`, draw `n` by `u n`: for the probes' population from
`D` conditioned off `T` by the uniform, for a harvested one by its sampler at cap `N` over the
probes the block draws from `D`.  `none` if a harvested draw misses. -/
def popDraws {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) : Option (Fin M → FreeMonoid α) :=
  match j with
  | none => some fun n => drawFrom (offLaw D T) (u n).1
  | some i =>
    match pops[i.val]? with
    | some s => Random.drawManyFirst s.read k s.hyp s.site ↑T
        fun (n : Fin M) (t : Fin N) => drawFrom D ((u n).2 t)
    | none => some fun _ => 1

/-- The clustering's draws after the history `h`: suffixes from `Dsf` off what the earlier rounds
read, each population's first `M` draws (`popDraws`) as its stream, the probes' population a
second `M` off the earlier rounds' prefixes as its certification stream, and the hash's sides.
`none` if a draw misses. -/
def clusterDraws {K : ℕ} (D Dsf : Measure (FreeMonoid α)) (k N M : ℕ) (h : SpecHist α)
    (c : ClusterDraws Ω α K N) : Option (Draws Ω (FreeMonoid α) (Option (Fin K))) :=
  (allSome fun j => popDraws D k N M h.pops h.prefs j (c.2.2.1 j)).map fun tab =>
    ((c.1, ((fun n => drawFrom (offLaw Dsf h.suffs) (c.2.1 n), fun j => padDraws (tab j)),
      fun j => match j with
        | none => padDraws fun n : Fin M => drawFrom (offLaw D h.prefs) (c.2.2.2.1 none n)
        | some _ => fun _ => 1)),
      c.2.2.2.2)

/-- A round: the state its clustering returned at and its family, the family's reads, what level 3
returned, and the prefixes and suffixes its clustering read. -/
structure SpecRound (α : Type*) where
  state : State
  family : Finset (FreeMonoid α)
  read : FreeMonoid α → ARU
  result : Random.Level3Result α
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

variable {μ : Measure Ω}

/-- The round after the history `h` whose clustering ran `x`: the state it returned at and its
family, and level 3's run over probes drawn by `u` from `D` avoiding the first `k` letters of the
prefixes the clustering read. -/
def roundOf {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : SpecHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : Fin settings.level3.probeBudget → unitInterval) : SpecRound α :=
  let ps := loopPopulations K h.pops.length
  let B := Classical.epsilon fun B => B ∈ settings.schedule ps
    ∧ x ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  let F := clusterAt O.mq ps x B
  let read := fun z => familyRead O.mq F B.lo (B.hi + 1) z x.1
  let K₀ := clusterKeys (settings.schedule ps) ps settings.level3.walkStart x
  ⟨B, F, read, Random.indecisiveLStarLevel3 settings.level3 read
    (fun i => drawFrom (avoidLaw D settings.level3.walkStart K₀) (u i)),
    roundPrefs (settings.schedule ps) ps x, roundSuffs (settings.schedule ps) x⟩

open scoped Classical in
/-- The round after the history `h`: the clustering's draws off what the earlier rounds read, then
the clustering on them and level 3.  `none` if a harvested draw misses, the clustering would read
past the `scanCap` draws, or no state of the schedule returns. -/
def loopRound {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    Option (SpecRound α) :=
  let ps := loopPopulations K h.pops.length
  (clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap h
      (d.1, d.2.1)).bind fun dr =>
    if ScanWithin (settings.schedule ps) ps settings.scanCap dr
        ∧ ∃ B ∈ settings.schedule ps, streams (fun _ => ∅) none dr
          ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
    then some (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2)
    else none

/-- The population a round's harvest hands on: its sampler's law over probes from `D`, walked
from `k`, scaled to a probability. -/
def SpecRound.population (D : Measure (FreeMonoid α)) (k : ℕ) (o : SpecRound α) :
    Option (Measure (FreeMonoid α)) :=
  o.result.ending.harvested.map fun site =>
    popLaw (Random.harvestLaw o.read D k o.result.hypothesis site)

/-- The sampler a round's harvest hands on: its family read, hypothesis and level-3 site. -/
def SpecRound.sampler (o : SpecRound α) : Option (SpecSampler α) :=
  o.result.ending.harvested.map fun site => ⟨o.read, o.result.hypothesis, site⟩

/-- What the rounds hand on: their harvests' samplers, and what their clusterings read. -/
def specHistOf (rounds : List (SpecRound α)) : SpecHist α :=
  ⟨rounds.filterMap fun o => o.sampler,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

/-- The first `r` rounds, cut short at a round that does not end in a harvest; `none` if a round
fails. -/
def loopRounds {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
      settings.level3.probeBudget) :
    ℕ → Option (List (SpecRound α))
  | 0 => some []
  | r + 1 => (loopRounds settings O D Dsf draws r).bind fun rs =>
    if h : r < K ∧ ∀ o ∈ rs, o.result.ending.harvested ≠ none then
      (loopRound settings O D Dsf (specHistOf rs) (draws ⟨r, h.1⟩)).map fun o => rs ++ [o]
    else some rs

end OrthoDFA

end
