import OrthoDFA.Code.Level3Harvest
import OrthoDFA.Code.Level3HarvestCode
import OrthoDFA.Code.Level3DrawMany
import Mathlib.MeasureTheory.Constructions.UnitInterval

/-!
# The learner's loop

Each round of `indecisiveLStarLoop` clusters a suffix family over the populations so far, runs
level 3's algorithm (`indecisiveLStarLevel3`) reading that family, and stops at the first round
that does not end in a harvest. A harvest's sampler becomes a population of every later round:
a draw from it walks fresh probes from the probes' distribution `D` under the round's last
hypothesis, at most level 3's `harvestReadCap` of them, and takes the first string undecided
at the harvest's site, as the round reads them, that no earlier clustering read. The first
population is `D` itself. Each population's streams are its first `scanCap` draws; the round fails
if a harvested draw finds no string, the clustering would read past them, or no state of the
schedule returns, and the loop fails with it. Draws are made from uniforms (`drawFrom`).

The clustering is the one `ClusteringQualityGuarantee` is about, every population tame and none
harvested into a table: the family is `clusterAt` at a state of the schedule at which `ret`
returns, on the sequences `streams` reads off the round's draws. The round reads the family
without its seed, at the gate's cut: a string reads accept at `hi + 1` or more of the family's
votes and reject at `lo` or fewer. Without its seed the family is suffix draws of one length, so
it is suffix-free.

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

/-- `ν` conditioned off `T`. -/
def offLaw (ν : Measure (FreeMonoid α)) (T : Finset (FreeMonoid α)) : Measure (FreeMonoid α) :=
  popLaw (ν.restrict (↑T)ᶜ)

/-- The uniforms by which the clustering draws its suffixes; each population's draws, a uniform
and a block of `N` uniforms each; the probes' population's certification draws; and the side of
the hash each string falls on.  A draw from the probes' population reads its uniform, a harvested
population's draw its block. -/
abbrev ClusterRest (α : Type*) (K N : ℕ) :=
  (ℕ → unitInterval) × ((Option (Fin K) → ℕ → unitInterval × (Fin N → unitInterval))
    × ((Option (Fin K) → ℕ → unitInterval) × (FreeMonoid α → Bool)))

/-- A round's draws apart from the noise: the clustering's, and the uniforms by which the round
draws its probes. -/
abbrev RoundDraws (α : Type*) (K N P : ℕ) := ClusterRest α K N × (Fin P → unitInterval)

/-- A draw's uniform and block, independent. -/
def drawLaw (N : ℕ) : Measure (unitInterval × (Fin N → unitInterval)) :=
  (volume : Measure unitInterval).prod (Measure.pi fun _ : Fin N => (volume : Measure unitInterval))

def restLaw (α : Type*) (K N : ℕ) : Measure (ClusterRest α K N) :=
  (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
    ((Measure.pi fun _ : Option (Fin K) => Measure.infinitePi fun _ : ℕ => drawLaw N).prod
      ((Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
        (Measure.infinitePi fun _ : FreeMonoid α => fairCoin)))

def roundLaw (α : Type*) (K N P : ℕ) : Measure (RoundDraws α K N P) :=
  (restLaw α K N).prod (Measure.pi fun _ : Fin P => (volume : Measure unitInterval))

variable {Ω : Type*} [MeasurableSpace Ω]

/-- A run's randomness: the oracle's noise, which every round reads, and each round's draws. -/
abbrev LoopDraws (Ω α : Type*) (K N P : ℕ) := Ω × (Fin K → RoundDraws α K N P)

/-- One noise for the whole run, which every string's read keeps, and `K` rounds' uniforms,
independent. -/
def loopMeasure (μ : Measure Ω) (α : Type*) (K N P : ℕ) : Measure (LoopDraws Ω α K N P) :=
  μ.prod (Measure.pi fun _ : Fin K => roundLaw α K N P)

open scoped Classical in
/-- `f`'s values, where every one is `some`. -/
def allSome {ι β : Type*} (f : ι → Option β) : Option (ι → β) :=
  if h : ∀ i, (f i).isSome then some fun i => (f i).get (h i) else none

/-- Finite draws as a stream, `1` past them. -/
def padDraws {M : ℕ} (f : Fin M → FreeMonoid α) (n : ℕ) : FreeMonoid α :=
  if h : n < M then f ⟨n, h⟩ else 1

/-- What the first `M` draws of population `j`'s streams fix of its table and certification
sequences, in order: the probes' population reads its two streams as they are, a harvested one
the hash's two sides of its one stream. -/
def seqsOf {K : ℕ} (M : ℕ) (dr : Draws Ω (FreeMonoid α) (Option (Fin K))) :
    Option (Fin K) → List (FreeMonoid α) × List (FreeMonoid α)
  | none => ((List.range M).map (dr.1.2.1.2 none), (List.range M).map (dr.1.2.2 none))
  | some i =>
    ((List.range M).map (dr.1.2.1.2 (some i)) |>.filter fun s => dr.2 s,
      (List.range M).map (dr.1.2.1.2 (some i)) |>.filter fun s => !dr.2 s)

/-- At every state of `sched`, the clustering finds each sample it takes among what the first `M`
draws of each population's streams fix. -/
def ScanWithin {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K))) (M : ℕ)
    (dr : Draws Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  ∀ B ∈ sched, ∀ j ∈ pops, B.npref ≤ (seqsOf M dr j).1.toFinset.card
    ∧ ∀ e ≤ B.npref, B.npref + e
      ≤ ((seqsOf M dr j).2.toFinset \ prefixesAt pops B.npref (streams (fun _ => ∅) none dr)).card

/-- The clustering's schedule, FNR limit and gate level, and level 3's settings. -/
structure LoopSettings (K : ℕ) where
  schedule : Finset (Option (Fin K)) → Finset State
  η₀ : ℝ
  indecisionLimit : ℝ
  εcov : ℝ
  gateLevel : ℝ
  level3 : Random.Level3Settings
  /-- The draws of each population's streams the clustering may read. -/
  scanCap : ℕ

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

def RoundEnd.harvested {H : Type*} : RoundEnd H → Option H
  | .harvest h => some h
  | _ => none

/-- What a harvested population hands on to draw from it: its round's family and cut, read as
`familyRead` reads it, its last hypothesis, and the level-3 site of its harvest. -/
structure HarvestSampler (α : Type*) where
  family : Finset (FreeMonoid α)
  kl : ℕ
  kh : ℕ
  hyp : HypothesisDFA α
  site : Random.Level3Site

/-- What the earlier rounds hand on: their harvests' samplers, and every prefix and suffix
their clusterings read. -/
structure LoopHist (α : Type*) where
  pops : List (HarvestSampler α)
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- A round: the state its clustering returned at and its family, what level 3 returned, and the
prefixes and suffixes its clustering read. -/
structure LoopRound (α : Type*) where
  state : State
  family : Finset (FreeMonoid α)
  result : Random.Level3Result α
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- The round's family read at the noise `ω`, at the gate's cut. -/
def LoopRound.read (O : Oracle μ (FreeMonoid α)) (ω : Ω) (o : LoopRound α) :
    FreeMonoid α → ARU :=
  fun z => familyRead O.mq o.family o.state.lo (o.state.hi + 1) z ω

/-- The sampler a round's harvest hands on. -/
def LoopRound.sampler (o : LoopRound α) : Option (HarvestSampler α) :=
  o.result.ending.harvested.map fun site =>
    ⟨o.family, o.state.lo, o.state.hi + 1, o.result.hypothesis, site⟩

/-- What the rounds hand on: their harvests' samplers, and what their clusterings read. -/
def histOf (rounds : List (LoopRound α)) : LoopHist α :=
  ⟨rounds.filterMap fun o => o.sampler,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

def clusterReads {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Finset (FreeMonoid α) :=
  (roundPrefs sched pops x ×ˢ insert 1 (roundSuffs sched x)).image fun z => z.1 * z.2

/-- Population `j`'s first `M` draws off `T`; `none` if a harvested draw misses. -/
def popDrawsQ {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (HarvestSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    Query (FreeMonoid α) ℝ (Option (Fin M → FreeMonoid α)) :=
  match j with
  | none => .pure (some fun n => drawFrom (offLaw D T) (u n).1)
  | some i =>
    match pops[i.val]? with
    | some s => (Random.drawManyFirstQ k s.hyp s.site ↑T
        fun (n : Fin M) (t : Fin N) => drawFrom D ((u n).2 t)).subst
          (familyReadQ s.family s.kl s.kh)
    | none => .pure (some fun _ => 1)

/-- The clustering's draws after the history `h`; `none` if a draw misses. -/
def clusterDrawsQ {K : ℕ} (D Dsf : Measure (FreeMonoid α)) (k N M : ℕ) (h : LoopHist α)
    (c : ClusterRest α K N) :
    Query (FreeMonoid α) ℝ (Option (Draws Unit (FreeMonoid α) (Option (Fin K)))) :=
  (Query.pi fun j => popDrawsQ D k N M h.pops h.prefs j (c.2.1 j)).map fun tabs =>
    (allSome tabs).map fun tab =>
      (((), ((fun n => drawFrom (offLaw Dsf h.suffs) (c.1 n), fun j => padDraws (tab j)),
        fun j => match j with
          | none => padDraws fun n : Fin M => drawFrom (offLaw D h.prefs) (c.2.2.1 none n)
          | some _ => fun _ => 1)),
        c.2.2.2)

open scoped Classical in
/-- The round after the history `h`; `none` if it fails. -/
def roundQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    Query (FreeMonoid α) ℝ (Option (LoopRound α)) :=
  let ps := loopPopulations K h.pops.length
  let sched := settings.schedule ps
  (clusterDrawsQ D Dsf settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap h
      d.1).bind fun
  | none => .pure none
  | some dr =>
    let x := streams (fun _ => ∅) none dr
    if ScanWithin sched ps settings.scanCap dr then
      (Query.table (clusterReads sched ps x).toList).bind fun t =>
        let mq : FreeMonoid α → Unit → ℝ := fun w _ => t w
        if ∃ B ∈ sched, x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
            settings.gateLevel B then
          let B := Classical.epsilon fun B => B ∈ sched
            ∧ x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
              settings.gateLevel B
          let F := clusterAt mq ps x B
          ((Random.level3Q settings.level3 fun i =>
              drawFrom (avoidLaw D settings.level3.walkStart
                (clusterKeys sched ps settings.level3.walkStart x)) (d.2 i)).subst
            (familyReadQ F B.lo (B.hi + 1))).map fun res =>
            some ⟨B, F, res, roundPrefs sched ps x, roundSuffs sched x⟩
        else .pure none
    else .pure none

/-- The first `r` rounds, cut short at a round that does not end in a harvest; `none` if a round
fails. -/
def loopRoundsQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    ℕ → Query (FreeMonoid α) ℝ (Option (List (LoopRound α)))
  | 0 => .pure (some [])
  | r + 1 => (loopRoundsQ settings D Dsf draws r).bind fun
    | none => .pure none
    | some rs =>
      if h : r < K ∧ ∀ o ∈ rs, o.result.ending.harvested ≠ none then
        (roundQ settings D Dsf (histOf rs) (draws ⟨r, h.1⟩)).map fun oo => oo.map fun o => rs ++ [o]
      else .pure (some rs)

/-- The loop: at most `K` rounds; `none` if a round fails. -/
def loopQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    Query (FreeMonoid α) ℝ (Option (List (LoopRound α))) :=
  loopRoundsQ settings D Dsf draws K

/-- The rounds the loop runs at the noise `p.1` and the draws `p.2`: `loopQ` asking `O`'s answers
at that noise. -/
def indecisiveLStarLoop {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (p : LoopDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    Option (List (LoopRound α)) :=
  (loopQ settings D Dsf p.2).eval fun w => O.mq w p.1

end OrthoDFA

end
