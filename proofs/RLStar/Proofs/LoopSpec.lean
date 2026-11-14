import RLStar.Proofs.ProgramEqns
import RLStar.Code.LearnerLoop
import RLStar.Code.Level3HarvestCode
import RLStar.LowLevelClaims.LazyRound
import RLStar.Proofs.Query

/-!
# The loop as functions of the oracle

The loop of `RLStar.Code.LearnerLoop` written as functions of the oracle and its noise rather
than as a program: a sampler and a round carry their family's read, and each round may read its
own noise. `loopQ`'s run against one noise's answers is `loopRounds` at that noise in every round
(`RLStar.Proofs.LoopQuery`), so what is proved of these is proved of the loop.

A harvested population's draw is its sampler's hit where the sampler hits, and a draw from the
population's law by the draw's uniform where it misses (`fillDraw`); the round fails where one of
its first `scanCap` misses (`PopsHit`), so the round's draws are those the loop makes.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace Random

/-- What a probe from `D`, padded from `X` and walked under `H`, harvests at the site, its pads
dropped. -/
def lazyHarvestLaw (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (K : ℕ)
    (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (s : Level3Site) :
    Measure (FreeMonoid α) :=
  (harvestLaw (padRead read) (D.map (padIn K X)) K H s).map unpad

/-- The string the first of `xs`, padded from `X` and walked under `H`, harvests at the site off
`T`, its pads dropped. -/
def lazyHarvestFirst (read : FreeMonoid α → ARU) (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) (s : Level3Site) (T : Finset (FreeMonoid α))
    (xs : List (FreeMonoid α)) : Option (FreeMonoid α) :=
  (harvestFirst (padRead read) K H s (unpad ⁻¹' ↑T) (xs.map (padIn K X))).map unpad

end Random

/-- What a harvested population hands on to draw from it: its round's family read, its last
hypothesis and trie, and the site of its harvest. -/
structure SpecSampler (α : Type*) where
  read : FreeMonoid α → ARU
  hyp : HypothesisDFA (Option α)
  trie : Finset (FreeMonoid α)
  site : Random.Level3Site

/-- The populations' distributions, given the earlier harvests' samplers: the probes' `D`, and
each harvested population's harvest law over probes from `D`, as a probability. -/
def loopDistributions {K : ℕ} (D : Measure (FreeMonoid α)) (Kw : ℕ)
    (pops : List (SpecSampler α)) : Option (Fin K) → Measure (FreeMonoid α)
  | none => D
  | some i =>
    match pops[i.val]? with
    | some s => popLaw (Random.lazyHarvestLaw s.read D Kw s.trie s.hyp s.site)
    | none => Measure.dirac 1

/-- What the earlier rounds hand on: their harvests' samplers, and every prefix and suffix their
clusterings read. -/
structure SpecHist (α : Type*) where
  pops : List (SpecSampler α)
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- The clustering's draws: the noise and the rest. -/
abbrev ClusterDraws (Ω α : Type*) (K N : ℕ) := Ω × ClusterRest α K N

/-- A round's draws with the noise it reads. -/
abbrev LoopRoundDraws {K : ℕ} (Ω α : Type*) (s : LoopSettings K) := Ω × s.RoundDraws α

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The rounds' draws, each read through the run's one noise. -/
def persistDraws {K : ℕ} {s : LoopSettings K} (p : s.Draws Ω α) : Fin K → LoopRoundDraws Ω α s :=
  fun i => (p.1, p.2 i)

/-- A harvested population's sampler's hit off `T` over the `N` probes the block `u.2` draws from
`D`; `none` at the probes' population. -/
def popHit {K : ℕ} (D : Measure (FreeMonoid α)) (Kw N : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K)) (u : unitInterval × (Fin N → unitInterval)) :
    Option (FreeMonoid α) :=
  match j with
  | none => none
  | some i =>
    match pops[i.val]? with
    | some s => Random.lazyHarvestFirst s.read Kw s.trie s.hyp s.site T
        (List.ofFn fun t => drawFrom D (u.2 t))
    | none => none

/-- Population `j`'s draw off `T` by `u`: its sampler's hit, where it hits, else a draw from its
law off `T` by the uniform `u.1`. -/
def fillDraw {K : ℕ} (D : Measure (FreeMonoid α)) (Kw N : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K)) (u : unitInterval × (Fin N → unitInterval)) :
    FreeMonoid α :=
  (popHit D Kw N pops T j u).getD (drawFrom (offLaw (loopDistributions D Kw pops j) T) u.1)

/-- Population `j`'s first `M` draws off `T`, draw `n` by `u n`; none for a population not yet
harvested. -/
def popTable {K : ℕ} (D : Measure (FreeMonoid α)) (Kw N M : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) : List (FreeMonoid α) :=
  match j with
  | none => uniDraws D M T u
  | some i =>
    match pops[i.val]? with
    | some _ => List.ofFn fun n : Fin M => fillDraw D Kw N pops T (some i) (u n)
    | none => []

/-- Every harvested population's first `M` draws hit. -/
def PopsHit {K : ℕ} (D : Measure (FreeMonoid α)) (Kw N M : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (c : ClusterRest α K N) : Prop :=
  ∀ i : Fin K, i.val < pops.length → ∀ n < M,
    (popHit D Kw N pops T (some i) (c.2.1 (some i) n)).isSome

/-- A round: the state its clustering returned at and its family, the family's reads, what the lazy
round returned, and the prefixes and suffixes its clustering read. -/
structure SpecRound (α : Type*) where
  state : State
  family : Finset (FreeMonoid α)
  read : FreeMonoid α → ARU
  result : Random.LazyResult α
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

variable {μ : Measure Ω}

/-- The round after the history `h` whose clustering ran `x`: the state it returned at and its
family, and the lazy round from the trie of the clustering's prefixes and the trie draws, its
probes drawn by `u` from `D` avoiding the first `walkStart` letters of those prefixes. -/
def roundOf {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D : Measure (FreeMonoid α)) (h : SpecHist α) (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : settings.Uniforms) : SpecRound α :=
  let ps := loopPopulations K h.pops.length
  let B := Classical.epsilon fun B => B ∈ settings.schedule ps
    ∧ x ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  let F := clusterAt O.mq ps x B
  let read := fun z => familyRead O.mq F B.lo (B.hi + 1) z x.1
  let Kw := settings.level3.walkStart
  let CP := roundPrefs (settings.schedule ps) ps x
  ⟨B, F, read, (Random.lazyRoundQ settings.level3 Kw settings.listSize
      (Random.prefixTrie Kw (CP ∪ heavySet D u.1))
      (lazyProbes (avoidLaw D Kw (clusterKeys (settings.schedule ps) ps Kw x)) u.2)).eval read,
    CP, roundSuffs (settings.schedule ps) x⟩

/-- The round's tables after the history `h`: each population's first `scanCap` draws off what
the earlier rounds read. -/
def specTables {K : ℕ} (settings : LoopSettings K) (D : Measure (FreeMonoid α)) (h : SpecHist α)
    (c : ClusterRest α K settings.level3.harvestReadCap) : Option (Fin K) → List (FreeMonoid α) :=
  fun j => popTable D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
    h.pops h.prefs j (c.2.1 j)

open scoped Classical in
/-- The round after the history `h`: each population's draws off what the earlier rounds read
(`specTables`), then the clustering on them (`drawsOf`) and the lazy round.  `none` if a harvested
draw misses, the clustering would read past the draws or no state of the schedule returns. -/
def loopRound {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α) (d : LoopRoundDraws Ω α settings) :
    Option (SpecRound α) :=
  let ps := loopPopulations K h.pops.length
  let dr := drawsOf d.1 D Dsf settings.scanCap h.prefs h.suffs d.2.1 (specTables settings D h d.2.1)
  if PopsHit D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap h.pops
        h.prefs d.2.1
      ∧ ScanWithin (settings.schedule ps) ps (fun _ => settings.scanCap) dr
      ∧ ∃ B ∈ settings.schedule ps, streams (fun _ => ∅) none dr
        ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  then some (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2)
  else none

/-- The sampler a round's harvest hands on: its family read, hypothesis, trie and site. -/
def SpecRound.sampler (o : SpecRound α) : Option (SpecSampler α) :=
  match o.result.ending with
  | .harvest s _ => some ⟨o.read, o.result.hypothesis, o.result.expanded, s⟩
  | _ => none

/-- What the rounds hand on: their harvests' samplers, and what their clusterings read. -/
def specHistOf (rounds : List (SpecRound α)) : SpecHist α :=
  ⟨rounds.filterMap fun o => o.sampler,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

open scoped Classical in
/-- The first `r` rounds, cut short at a round that does not end in a harvest; `none` if a round
fails. -/
def loopRounds {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (draws : Fin K → LoopRoundDraws Ω α settings) :
    ℕ → Option (List (SpecRound α))
  | 0 => some []
  | r + 1 => (loopRounds settings O D Dsf draws r).bind fun rs =>
    if h : r < K ∧ ∀ o ∈ rs, o.result.ending.IsHarvest then
      (loopRound settings O D Dsf (specHistOf rs) (draws ⟨r, h.1⟩)).map fun o => rs ++ [o]
    else some rs

end OrthoDFA

end
