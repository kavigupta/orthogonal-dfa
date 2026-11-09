import RLStar.Code.LearnerLoop
import RLStar.Code.TrieDFA

/-!
# The learner's loop with local walk starts

`indecisiveLStarLoop` with level 3 run with local walk starts (`indecisiveLStarLevel3Trie`): each
round clusters a suffix family over the populations so far, as there, and runs the round's probes
from `D` off the clustering's keys at `walkStart` letters, each walked past the prefixes its round
has expanded. A harvest's sampler walks its fresh probes past its round's expanded prefixes. The
loop stops at the first round that does not end in a harvest, and its DFA is that round's
hypothesis with its expanded prefixes spelled out (`HypothesisDFA.TrieClassifies`).

The modelling gaps of `RLStar.Code.LearnerLoop` stand, but the held-out start pick, which this loop
does not make.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- What a harvested population hands on to draw from it: its round's family and cut, its last
hypothesis and expanded prefixes, and the site of its harvest. -/
structure TrieHarvestSampler (α : Type*) where
  family : Finset (FreeMonoid α)
  kl : ℕ
  kh : ℕ
  hyp : HypothesisDFA α
  expanded : Finset (FreeMonoid α)
  site : Random.TrieSite

structure TrieLoopHist (α : Type*) where
  pops : List (TrieHarvestSampler α)
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

structure TrieLoopRound (α : Type*) where
  state : State
  family : Finset (FreeMonoid α)
  result : Random.TrieResult α
  prefs : Finset (FreeMonoid α)
  suffs : Finset (FreeMonoid α)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The round's family read at the noise `ω`, at the gate's cut. -/
def TrieLoopRound.read (O : Oracle μ (FreeMonoid α)) (ω : Ω) (o : TrieLoopRound α) :
    FreeMonoid α → ARU :=
  fun z => familyRead O.mq o.family o.state.lo (o.state.hi + 1) z ω

def TrieLoopRound.sampler (o : TrieLoopRound α) : Option (TrieHarvestSampler α) :=
  o.result.ending.harvested.map fun site =>
    ⟨o.family, o.state.lo, o.state.hi + 1, o.result.hypothesis, o.result.expanded, site⟩

def trieHistOf (rounds : List (TrieLoopRound α)) : TrieLoopHist α :=
  ⟨rounds.filterMap fun o => o.sampler,
    rounds.foldr (fun o T => o.prefs ∪ T) ∅, rounds.foldr (fun o T => o.suffs ∪ T) ∅⟩

open scoped Classical in
/-- `popDrawsQ` with the harvested populations walked past their expanded prefixes. -/
def triePopDrawsQ {K : ℕ} (D : Measure (FreeMonoid α)) (N M : ℕ)
    (pops : List (TrieHarvestSampler α)) (T : Finset (FreeMonoid α))
    (stop : List (FreeMonoid α) → Prop) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    Query (FreeMonoid α) ℝ (Option (List (FreeMonoid α))) :=
  match j with
  | none => .pure (some (uniDraws D M T u))
  | some i =>
    match pops[i.val]? with
    | some s => (Query.extendUntil (fun n => Random.trieHarvestFirstQ s.expanded s.hyp s.site ↑T
        (List.ofFn fun t : Fin N => drawFrom D ((u n).2 t))) stop M []).subst
          (familyReadQ s.family s.kl s.kh)
    | none => .pure (some [])

def trieDrawListsQ {K : ℕ} (D : Measure (FreeMonoid α)) (N M : ℕ) (sched : Finset State)
    (h : TrieLoopHist α) (c : ClusterRest α K N) :
    Query (FreeMonoid α) ℝ (Option (Option (Fin K) → List (FreeMonoid α))) :=
  (Query.pi fun j => triePopDrawsQ D N M h.pops h.prefs
      (DrawsHold sched c.2.2.2 (uniTable D M h.prefs (c.2.1 none))) j (c.2.1 j)).map allSome

open scoped Classical in
/-- `roundQ` with level 3 run with local walk starts. -/
def trieRoundQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (h : TrieLoopHist α)
    (d : RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    Query (FreeMonoid α) ℝ (Option (TrieLoopRound α)) :=
  let ps := loopPopulations K h.pops.length
  let sched := settings.schedule ps
  (trieDrawListsQ D settings.level3.harvestReadCap settings.scanCap sched h d.1).bind fun
  | none => .pure none
  | some tabs =>
    let dr := drawsOf () D Dsf settings.scanCap h.prefs h.suffs d.1 tabs
    let x := streams (fun _ => ∅) none dr
    if ScanWithin sched ps (fun j => (tabs j).length) dr then
      (Query.table (clusterReads sched ps x).toList).bind fun t =>
        let mq : FreeMonoid α → Unit → ℝ := fun w _ => t w
        if ∃ B ∈ sched, x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
            settings.gateLevel B then
          let B := Classical.epsilon fun B => B ∈ sched
            ∧ x ∈ ret mq ps none settings.η₀ settings.indecisionLimit settings.εcov
              settings.gateLevel B
          let F := clusterAt mq ps x B
          ((Random.level3TrieQ settings.level3 fun i =>
              drawFrom (avoidLaw D settings.level3.walkStart
                (clusterKeys sched ps settings.level3.walkStart x)) (d.2 i)).subst
            (familyReadQ F B.lo (B.hi + 1))).map fun res =>
            some ⟨B, F, res, roundPrefs sched ps x, roundSuffs sched x⟩
        else .pure none
    else .pure none

def trieLoopRoundsQ {K : ℕ} (settings : LoopSettings K) (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → RoundDraws α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    ℕ → Query (FreeMonoid α) ℝ (Option (List (TrieLoopRound α)))
  | 0 => .pure (some [])
  | r + 1 => (trieLoopRoundsQ settings D Dsf draws r).bind fun
    | none => .pure none
    | some rs =>
      if h : r < K ∧ ∀ o ∈ rs, o.result.ending.harvested ≠ none then
        (trieRoundQ settings D Dsf (trieHistOf rs) (draws ⟨r, h.1⟩)).map fun oo =>
          oo.map fun o => rs ++ [o]
      else .pure (some rs)

/-- The rounds the loop runs at the noise `p.1` and the draws `p.2`; `none` if a round fails. -/
def indecisiveLStarTrieLoop {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (p : LoopDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) :
    Option (List (TrieLoopRound α)) :=
  (trieLoopRoundsQ settings D Dsf p.2 K).eval fun w => O.mq w p.1

end OrthoDFA

end
