import RLStar.Code.LazyRound
import Mathlib.MeasureTheory.Constructions.UnitInterval

/-!
# The learner's loop

Each round of `indecisiveLStarLoop` clusters a suffix family over the populations so far, runs
the lazy round (`Random.lazyRoundQ`) reading that family, and stops at the first round that does
not end in a harvest. A harvest's population is uniform over the list that filled, and it is a
population of every later round. The first population is `D`
itself. Every population's table is its first `scanCap` draws off the prefixes the earlier rounds
read. The round fails if the clustering would read past the draws or no state of the schedule
returns, and the loop fails with it. Draws are made from uniforms (`drawFrom`). After the loop,
`learnerQ` builds the last round's walk-start table (`Random.lazyTableQ`).

The clustering is the one `ClusteringQualityGuarantee` is about, every population tame and none
harvested into a table: the family is `clusterAt` at a state of the schedule at which `ret`
returns, on the sequences `streams` reads off the round's draws. The round reads the family
without its seed, at the gate's cut: a string reads accept at `hi + 1` or more of the family's
votes and reject at `lo` or fewer. Without its seed the family is suffix draws of one length, so
it is suffix-free.

Known modelling gap. The round starts its trie from every prefix shorter than `walkStart` of
every prefix its clustering read (`Random.prefixTrie`). The Python starts from the empty trie.

Known modelling gap. The round draws its probes from `D` conditioned on their first `walkStart`
letters avoiding those of every prefix the round's clustering read, so that it never reads a string
the clustering read. The Python draws them from `D`.

Known modelling gap. Each round draws its hash's sides afresh, where the Python's key
(`PrefixPopulations.key`) is drawn once per run. With one key, a later round could draw a string
whose side an earlier round's hash read without tabling or certifying it, and that side would not
be fresh.

Known modelling gap. A harvest's table here is drawn from its population's law, where the Python
first puts the strings the round harvested into it (`take`).

Known modelling gap. Every population draws `scanCap` strings. The Python's `split_draws` stops
at a count of table-side draws that counts duplicates and grows with the state the clustering
reaches, and `prefixes_for_split` excludes `tabled()`, the strings of every table and of `seen`,
from its reserve-side draws.

Known modelling gap. The uniform pool is drawn afresh each round, where the Python keeps one pool
across rounds; on one noise a kept pool's prefixes would be read again by every round.

Known modelling gap. The gate draws its certification pool to at most twice its size. The
Python caps each verdict's extra draws at `certification_budget`, but `sample_suffix_family`'s
outer resampling loop is unbounded.

Known modelling gap. The Python reads the family with its seed, at `pst.accept_thresh` and
`pst.reject_thresh`.

Known modelling gap. The DFA reads each walk start's sift off a table the last round builds over
its trie's frontier. The Python sifts at the decision boundary with the base family.

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

/-- Uniform over the distinct strings of `l`. -/
def listLaw (l : List (FreeMonoid α)) : Measure (FreeMonoid α) :=
  popLaw (∑ z ∈ l.toFinset, Measure.dirac z)

/-- The populations' distributions after the harvests `pops`: the probes' `D`, and each harvest's
list law. -/
def loopDistributions {K : ℕ} (D : Measure (FreeMonoid α)) (pops : List (List (FreeMonoid α))) :
    Option (Fin K) → Measure (FreeMonoid α)
  | none => D
  | some i =>
    match pops[i.val]? with
    | some l => listLaw l
    | none => Measure.dirac 1

/-- The uniforms by which the clustering draws its suffixes; each population's table draws; the
probes' population's certification draws; and the side of the hash each string falls on. -/
abbrev ClusterRest (α : Type*) (K : ℕ) :=
  (ℕ → unitInterval) × ((Option (Fin K) → ℕ → unitInterval)
    × ((Option (Fin K) → ℕ → unitInterval) × (FreeMonoid α → Bool)))

/-- Each epoch's probe uniforms and its blocks' uniforms. -/
abbrev LazyUniforms (E P J b : ℕ) :=
  Fin E → (Fin P → unitInterval) × (Fin J → Fin b → unitInterval)

def restLaw (α : Type*) (K : ℕ) : Measure (ClusterRest α K) :=
  (Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
    ((Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
      ((Measure.pi fun _ : Option (Fin K) =>
        Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)).prod
        (Measure.infinitePi fun _ : FreeMonoid α => fairCoin)))

def lazyLaw (E P J b : ℕ) : Measure (LazyUniforms E P J b) :=
  Measure.pi fun _ : Fin E => (Measure.pi fun _ : Fin P => (volume : Measure unitInterval)).prod
    (Measure.pi fun _ : Fin J => Measure.pi fun _ : Fin b => (volume : Measure unitInterval))

/-- Finite draws as a stream, `1` past them. -/
def padDraws {M : ℕ} (f : Fin M → FreeMonoid α) (n : ℕ) : FreeMonoid α :=
  if h : n < M then f ⟨n, h⟩ else 1

variable {Ω : Type*} [MeasurableSpace Ω]

/-- What the first `n j` draws of population `j`'s streams fix of its table and certification
sequences, in order: the probes' population reads its two streams as they are, a harvested one
the hash's two sides of its one stream. -/
def seqsOf {K : ℕ} (n : Option (Fin K) → ℕ) (dr : Draws Ω (FreeMonoid α) (Option (Fin K))) :
    Option (Fin K) → List (FreeMonoid α) × List (FreeMonoid α)
  | none => ((List.range (n none)).map (dr.1.2.1.2 none), (List.range (n none)).map (dr.1.2.2 none))
  | some i =>
    ((List.range (n (some i))).map (dr.1.2.1.2 (some i)) |>.filter fun s => dr.2 s,
      (List.range (n (some i))).map (dr.1.2.1.2 (some i)) |>.filter fun s => !dr.2 s)

/-- At every state of `sched`, the clustering finds each sample it takes among what the first `n j`
draws of each population `j`'s streams fix. -/
def ScanWithin {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K)))
    (n : Option (Fin K) → ℕ) (dr : Draws Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  ∀ B ∈ sched, ∀ j ∈ pops, B.npref ≤ (seqsOf n dr j).1.toFinset.card
    ∧ ∀ e ≤ B.npref, B.npref + e
      ≤ ((seqsOf n dr j).2.toFinset \ prefixesAt pops B.npref (streams (fun _ => ∅) none dr)).card

/-- The clustering's schedule, FNR limit and gate level, level 3's settings, and the lazy round's
list size `nh`, epochs `E`, blocks per top-up `J` and block size `b`. -/
structure LoopSettings (K : ℕ) where
  schedule : Finset (Option (Fin K)) → Finset State
  η₀ : ℝ
  indecisionLimit : ℝ
  εcov : ℝ
  gateLevel : ℝ
  level3 : Random.Level3Settings
  /-- The draws each population makes. -/
  scanCap : ℕ
  listSize : ℕ
  epochs : ℕ
  blocks : ℕ
  blockSize : ℕ

abbrev LoopSettings.Uniforms {K : ℕ} (s : LoopSettings K) :=
  LazyUniforms s.epochs s.level3.probeBudget s.blocks s.blockSize

/-- A round's draws apart from the noise: the clustering's, and the lazy round's uniforms. -/
abbrev LoopSettings.RoundDraws {K : ℕ} (s : LoopSettings K) (α : Type*) :=
  ClusterRest α K × s.Uniforms

/-- A run's randomness: the oracle's noise, which every round reads, and each round's draws. -/
abbrev LoopSettings.Draws {K : ℕ} (s : LoopSettings K) (Ω α : Type*) :=
  Ω × (Fin K → s.RoundDraws α)

def LoopSettings.roundLaw {K : ℕ} (s : LoopSettings K) (α : Type*) : Measure (s.RoundDraws α) :=
  (restLaw α K).prod (lazyLaw s.epochs s.level3.probeBudget s.blocks s.blockSize)

/-- One noise for the whole run, which every string's read keeps, and `K` rounds' uniforms,
independent. -/
def LoopSettings.measure {K : ℕ} (s : LoopSettings K) (μ : Measure Ω) (α : Type*) :
    Measure (s.Draws Ω α) :=
  μ.prod (Measure.pi fun _ : Fin K => s.roundLaw α)

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

/-- What the earlier rounds hand on: their harvests' lists, and every prefix and suffix their
clusterings read. -/
structure LoopHist (α : Type*) where
  pops : List (List (FreeMonoid α))
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- A round: the state its clustering returned at and its family, what the lazy round returned,
and the prefixes and suffixes its clustering read. -/
structure LoopRound (α : Type*) where
  state : State
  family : Finset (FreeMonoid α)
  result : Random.LazyResult α
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

/-- The round's family read at the noise `ω`, at the gate's cut. -/
def LoopRound.read (O : Oracle μ (FreeMonoid α)) (ω : Ω) (o : LoopRound α) :
    FreeMonoid α → ARU :=
  fun z => familyRead O.mq o.family o.state.lo (o.state.hi + 1) z ω

/-- A harvest's list. -/
def LoopRound.population (o : LoopRound α) : Option (List (FreeMonoid α)) :=
  match o.result.ending with
  | .harvest _ ℓ => some ℓ
  | _ => none

/-- What the rounds hand on: their harvests' populations, and what their clusterings read. -/
def histOf (rounds : List (LoopRound α)) : LoopHist α :=
  ⟨rounds.filterMap fun o => o.population,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

def clusterReads {K : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K)))
    (x : Run Ω (FreeMonoid α) (Option (Fin K))) : Finset (FreeMonoid α) :=
  (roundPrefs sched pops x ×ˢ insert 1 (roundSuffs sched x)).image fun z => z.1 * z.2

/-- Finite draws as a stream, `1` past them. -/
def padList (l : List (FreeMonoid α)) (n : ℕ) : FreeMonoid α := l.getD n 1

/-- `ν`'s first `M` draws off `T`, draw `n` by `u n`. -/
def uniDraws (ν : Measure (FreeMonoid α)) (M : ℕ) (T : Finset (FreeMonoid α))
    (u : ℕ → unitInterval) : List (FreeMonoid α) :=
  List.ofFn fun n : Fin M => drawFrom (offLaw ν T) (u n)

/-- Each population's first `M` draws after the history `h`, off what the earlier rounds read. -/
def popTables {K : ℕ} (D : Measure (FreeMonoid α)) (M : ℕ) (h : LoopHist α)
    (c : ClusterRest α K) : Option (Fin K) → List (FreeMonoid α) :=
  fun j => uniDraws (loopDistributions D h.pops j) M h.prefs (c.2.1 j)

/-- The clustering's draws at the noise `ω`, each population's stream its draws `tabs`: suffixes
from `Dsf` off `suffs`, the probes' population a second `M` off `prefs` as its certification
stream, and the hash's sides. -/
def drawsOf {K : ℕ} (ω : Ω) (D Dsf : Measure (FreeMonoid α)) (M : ℕ)
    (prefs suffs : Finset (FreeMonoid α)) (c : ClusterRest α K)
    (tabs : Option (Fin K) → List (FreeMonoid α)) : Draws Ω (FreeMonoid α) (Option (Fin K)) :=
  ((ω, ((fun n => drawFrom (offLaw Dsf suffs) (c.1 n), fun j => padList (tabs j)),
    fun j => match j with
      | none => padDraws fun n : Fin M => drawFrom (offLaw D prefs) (c.2.2.1 none n)
      | some _ => fun _ => 1)),
    c.2.2.2)

/-- The lazy round's probes and blocks, drawn from `ν` by the uniforms `u`. -/
def lazyProbes {E P J b : ℕ} (ν : Measure (FreeMonoid α)) (u : LazyUniforms E P J b) :
    Fin E → Random.EpochDraws α P J b :=
  fun e => (fun i => drawFrom ν ((u e).1 i), fun j t => drawFrom ν ((u e).2 j t))

open scoped Classical in
/-- The round after the history `h`; `none` if it fails. -/
def roundQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : settings.RoundDraws α) : Query (FreeMonoid α) ℝ (Option (LoopRound α)) :=
  let ps := loopPopulations K h.pops.length
  let sched := settings.schedule ps
  let Kw := settings.level3.walkStart
  let tabs := popTables D settings.scanCap h d.1
  let dr := drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs
  let x := streams (fun _ => ∅) none dr
  if ScanWithin sched ps (fun _ => settings.scanCap) dr then
    (Query.table (clusterReads sched ps x).toList).bind fun t =>
      let mq : FreeMonoid α → Unit → ℝ := fun w _ => t w
      if ∃ B ∈ sched, x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
          settings.gateLevel B then
        let B := Classical.epsilon fun B => B ∈ sched
          ∧ x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
            settings.gateLevel B
        let F := clusterAt mq ps x B
        let CP := roundPrefs sched ps x
        ((Random.lazyRoundQ settings.level3 Kw settings.listSize (Random.prefixTrie Kw CP)
            (lazyProbes (avoidLaw D Kw (clusterKeys sched ps Kw x)) d.2)).subst
          (familyReadQ F B.lo (B.hi + 1))).map fun res =>
          some ⟨B, F, res, CP, roundSuffs sched x⟩
      else .pure none
  else .pure none

open scoped Classical in
/-- The first `r` rounds, cut short at a round that does not end in a harvest; `none` if a round
fails. -/
def loopRoundsQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → settings.RoundDraws α) :
    ℕ → Query (FreeMonoid α) ℝ (Option (List (LoopRound α)))
  | 0 => .pure (some [])
  | r + 1 => (loopRoundsQ settings D Dsf draws r).bind fun
    | none => .pure none
    | some rs =>
      if h : r < K ∧ ∀ o ∈ rs, o.result.ending.IsHarvest then
        (roundQ settings D Dsf (histOf rs) (draws ⟨r, h.1⟩)).map fun oo =>
          oo.map fun o => rs ++ [o]
      else .pure (some rs)

/-- The loop: at most `K` rounds; `none` if a round fails. -/
def loopQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → settings.RoundDraws α) :
    Query (FreeMonoid α) ℝ (Option (List (LoopRound α))) :=
  loopRoundsQ settings D Dsf draws K

/-- The rounds the loop runs at the noise `p.1` and the draws `p.2`: `loopQ` asking `O`'s answers
at that noise. -/
def indecisiveLStarLoop {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (p : settings.Draws Ω α) : Option (List (LoopRound α)) :=
  (loopQ settings D Dsf p.2).eval fun w => O.mq w p.1

/-- The learner: the loop, then the last round's walk-start table read through its family. -/
def learnerQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → settings.RoundDraws α) :
    Query (FreeMonoid α) ℝ
      (Option (List (LoopRound α) × (FreeMonoid α → List Bool ⊕ FreeMonoid (Option α)))) :=
  (loopQ settings D Dsf draws).bind fun
  | none => .pure none
  | some rs =>
    match rs.getLast? with
    | none => .pure none
    | some o =>
      ((Random.lazyTableQ settings.level3.walkStart o.result.hypothesis.tree
          o.result.expanded).subst (familyReadQ o.family o.state.lo (o.state.hi + 1))).map
        fun tab => some (rs, tab)

end OrthoDFA

end
