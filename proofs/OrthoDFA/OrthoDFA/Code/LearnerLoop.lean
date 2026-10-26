import OrthoDFA.Code.Level3Harvest
import Mathlib.MeasureTheory.Constructions.UnitInterval

/-!
# The learner's loop

Each round of `indecisiveLStarLoop` clusters a suffix family over the populations so far, runs
level 3's algorithm (`indecisiveLStarLevel3`) reading that family, and stops at the first round
that does not end in a harvest. A harvest's sampler, scaled to a probability, becomes a population
of every later round: it draws a fresh probe from the probes' distribution `D`, walks it under the
round's last hypothesis and takes its undecided strings at the harvest's site, as the round reads
them. The first population is `D` itself. Draws are made from uniforms (`drawFrom`).

The clustering is the one `ClusteringQualityGuarantee` is about, every population tame and none
harvested into a table: the family is `clusterAt` at a state of the schedule at which `ret`
returns, on the sequences `streams` reads off the round's draws. The round reads the family
without its seed, at the gate's cut: a string reads accept at `hi + 1` or more of the family's
votes and reject at `lo` or fewer. Without its seed the family is suffix draws of one length, so
it is suffix-free.

The whole run reads one noise, fixed per string as the Python's is.

Known modelling gap. The round draws its probes from `D` conditioned on their first `walkStart`
letters avoiding those of every prefix the round's clustering read, so that it never reads a string
the clustering read. The Python draws them from `D`.

Known modelling gap. Each round draws its hash's sides afresh, where the Python's key
(`PrefixPopulations.key`) is drawn once per run. With one key, a later round could draw a string
whose side an earlier round's hash read without tabling or certifying it, and that side would not
be fresh.

Known modelling gap. A harvest's table here is drawn from its population's law, where the Python
first puts the strings the round harvested into it (`take`).

Known modelling gap. The uniform pool is drawn afresh each round, where the Python keeps one pool
across rounds; on one noise a kept pool's prefixes would be read again by every round.

Known modelling gap. The gate draws its certification pool to at most twice its size. The
Python caps each verdict's extra draws at `certification_budget`, but `sample_suffix_family`'s
outer resampling loop is unbounded.

Known modelling gap. The Python reads the family with its seed, at `pst.accept_thresh` and
`pst.reject_thresh`.

Known modelling gap. `counterexample_driven_synthesis` also runs the certificate, halves the FNR
limit, stops on a stall or on certificate patience and returns the
best round. None of these is modelled.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

instance : Nonempty State := ⟨⟨0, 0, 0, 0, 0, 0, 0⟩⟩

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
structure LoopHist (α : Type*) [Countable α] where
  pops : List (Measure (FreeMonoid α))
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- The uniforms by which the clustering draws its suffixes, and each population its two streams,
and the side of the hash each string falls on. -/
abbrev ClusterRest (α : Type*) (K : ℕ) :=
  (ℕ → unitInterval) × ((Option (Fin K) → ℕ → unitInterval)
    × ((Option (Fin K) → ℕ → unitInterval) × (FreeMonoid α → Bool)))

/-- The clustering's draws: the noise and the rest. -/
abbrev ClusterDraws (Ω α : Type*) (K : ℕ) := Ω × ClusterRest α K

/-- A round's draws apart from the noise: the clustering's, and the uniforms by which the round
draws its probes. -/
abbrev RoundDraws (α : Type*) (K P : ℕ) := ClusterRest α K × (Fin P → unitInterval)

/-- A round's draws with the noise it reads. -/
abbrev LoopRoundDraws (Ω α : Type*) (K P : ℕ) := Ω × RoundDraws α K P

def restLaw (α : Type*) (K : ℕ) : Measure (ClusterRest α K) :=
  (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
    ((Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
      ((Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
        (Measure.infinitePi fun _ : FreeMonoid α => fairCoin)))

def roundLaw (α : Type*) (K P : ℕ) : Measure (RoundDraws α K P) :=
  (restLaw α K).prod (Measure.pi fun _ : Fin P => (volume : Measure unitInterval))

variable {Ω : Type*} [MeasurableSpace Ω]

/-- One noise for the whole run, which every string's read keeps, and `K` rounds' uniforms,
independent. -/
def loopMeasure (μ : Measure Ω) (α : Type*) (K P : ℕ) :
    Measure (Ω × (Fin K → RoundDraws α K P)) :=
  μ.prod (Measure.pi fun _ : Fin K => roundLaw α K P)

/-- The rounds' draws, each read through the run's one noise. -/
def persistDraws {K P : ℕ} (p : Ω × (Fin K → RoundDraws α K P)) :
    Fin K → LoopRoundDraws Ω α K P :=
  fun i => (p.1, p.2 i)

/-- The clustering's draws after the history `h`: suffixes from `Dsf` and each population's two
streams, all off what the earlier rounds read, and the hash's sides. -/
def clusterDraws {K : ℕ} (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (c : ClusterDraws Ω α K) : Draws Ω (FreeMonoid α) (Option (Fin K)) :=
  ((c.1, ((fun n => drawFrom (offLaw Dsf h.suffs) (c.2.1 n),
      fun j n => drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.1 j n)),
    fun j n => drawFrom (offLaw (loopDistributions D h.pops j) h.prefs) (c.2.2.2.1 j n))),
    c.2.2.2.2)

/-- The clustering's run: no population harvests strings into its table. -/
def clusterRun {K : ℕ} (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (c : ClusterDraws Ω α K) : Run Ω (FreeMonoid α) (Option (Fin K)) :=
  streams (fun _ => ∅) none (clusterDraws D Dsf h c)

/-- The clustering's schedule, FNR limit and gate level, and level 3's settings. -/
structure LoopSettings (K : ℕ) where
  schedule : Finset (Option (Fin K)) → Finset State
  η₀ : ℝ
  indecisionLimit : ℝ
  εcov : ℝ
  gateLevel : ℝ
  level3 : Random.Level3Settings

/-- A round: the clustering's run, the state it returned at and its family, the family's reads,
what level 3 returned, and the prefixes and suffixes its clustering read. -/
structure LoopRound (Ω α : Type*) (K : ℕ) where
  run : Run Ω (FreeMonoid α) (Option (Fin K))
  state : State
  family : Finset (FreeMonoid α)
  read : FreeMonoid α → ARU
  result : Random.Level3Result α
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
level 3 draws its probes from `D` avoiding the first `k` letters of the prefixes the clustering
read. -/
def loopRound {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.probeBudget) : LoopRound Ω α K :=
  let ps := loopPopulations K h.pops.length
  let x := clusterRun D Dsf h (d.1, d.2.1)
  let B := Classical.epsilon fun B => B ∈ settings.schedule ps
    ∧ x ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  let F := clusterAt O.mq ps x B
  let read := fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1
  let K₀ := clusterKeys (settings.schedule ps) ps settings.level3.walkStart x
  ⟨x, B, F, read, Random.indecisiveLStarLevel3 settings.level3 read
    (fun i => drawFrom (avoidLaw D settings.level3.walkStart K₀) (d.2.2 i)),
    roundPrefs (settings.schedule ps) ps x, roundSuffs (settings.schedule ps) x⟩

def RoundEnd.harvested {H : Type*} : RoundEnd H → Option H
  | .harvest h => some h
  | _ => none

/-- The population a round's harvest hands on: its sampler's law over probes from `D`, walked
from `k`, scaled to a probability. -/
def LoopRound.population {K : ℕ} (D : Measure (FreeMonoid α)) (k : ℕ) (o : LoopRound Ω α K) :
    Option (Measure (FreeMonoid α)) :=
  o.result.ending.harvested.map fun site =>
    popLaw (Random.harvestLaw o.read D k o.result.hypothesis site)

/-- What the rounds hand on: their harvests' populations, and what their clusterings read. -/
def histOf {K : ℕ} (D : Measure (FreeMonoid α)) (k : ℕ) (rounds : List (LoopRound Ω α K)) :
    LoopHist α :=
  ⟨rounds.filterMap fun o => o.population D k,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

/-- The first `r` rounds, cut short at a round that does not end in a harvest. -/
def loopRounds {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.probeBudget) :
    ℕ → List (LoopRound Ω α K)
  | 0 => []
  | r + 1 =>
    if h : r < K ∧ ∀ o ∈ loopRounds settings O D Dsf draws r, o.result.ending.harvested ≠ none then
      loopRounds settings O D Dsf draws r
        ++ [loopRound settings O D Dsf
          (histOf D settings.level3.walkStart (loopRounds settings O D Dsf draws r))
          (draws ⟨r, h.1⟩)]
    else loopRounds settings O D Dsf draws r

/-- The rounds the loop runs, at most `K`. -/
def indecisiveLStarLoop {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.probeBudget) : List (LoopRound Ω α K) :=
  loopRounds settings O D Dsf draws K

end OrthoDFA

end
