import OrthoDFA.Code.LearnerLoop
import OrthoDFA.Code.Reads

/-!
# The oracle calls the loop makes

A round calls the oracle in three places. A harvested population's draw walks probes under its
sampler, each read a `familyRead` of its round's family, `|F|` calls; the probes it walks are those
up to the first that harvests (`harvestCount`). The clustering reads each prefix followed by each
pool suffix or by nothing, one call each. Level 3 reads through the round's family, `|F|` calls a
read, at each step its probe's and its split's strings (`runReadCount`). Each piece counts the
strings it reads; pieces, steps and rounds add up, and a round that fails counts what it read
before it failed (`indecisiveLStarLoopCalls`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*}

namespace Random

open scoped Classical in
/-- The strings `harvestFirst` reads over `xs`: each probe's, up to the first that harvests
outside `T`. -/
def harvestCount (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) : List (FreeMonoid α) → ℕ
  | [] => 0
  | x :: xs => (Kappa.probeReads read H.tree H.edges k x).card
      + if ((siteHarvest read k H site x).head?.filter fun z => z ∉ T).isSome then 0
        else harvestCount read k H site T xs

/-- `harvestCount` over each of `drawManyFirst`'s blocks. -/
def drawManyCount (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) {m N' : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α)) : ℕ :=
  ∑ i, harvestCount read k H site T (List.ofFn (yss i))

end Random

variable [Fintype α] [DecidableEq α]

/-- Each harvested population's calls per read, in the order of the populations: the size of its
round's family. -/
def samplerCosts (rounds : List (LoopRound α)) : List ℕ :=
  rounds.filterMap fun o => o.sampler.map fun _ => o.family.card

/-- The oracle calls of `popDraws D k N M pops T j u`, a read of population `j`'s sampler costing
`costs[j]` calls. -/
def popDrawCalls {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ) (pops : List (HarvestSampler α))
    (costs : List ℕ) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) : ℕ :=
  match j with
  | none => 0
  | some i =>
    match pops[i.val]? with
    | some s => costs.getD i.val 0 * Random.drawManyCount s.read k s.hyp s.site ↑T
        fun (n : Fin M) (t : Fin N) => drawFrom D ((u n).2 t)
    | none => 0

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- The probes `roundOf` runs level 3 on. -/
def roundProbes {K : ℕ} (settings : LoopSettings K) (D : Measure (FreeMonoid α)) (h : LoopHist α)
    (x : Run Ω (FreeMonoid α) (Option (Fin K)))
    (u : Fin settings.level3.probeBudget → unitInterval) :
    Fin settings.level3.probeBudget → FreeMonoid α :=
  let ps := loopPopulations K h.pops.length
  fun i => drawFrom (avoidLaw D settings.level3.walkStart
    (clusterKeys (settings.schedule ps) ps settings.level3.walkStart x)) (u i)

open scoped Classical in
/-- The oracle calls of `loopRound settings O D Dsf h d`, a read of population `j`'s sampler
costing `costs[j]` calls: its populations' draws, its clustering's reads where they draw, and
level 3's where the round runs. -/
def roundCalls {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (costs : List ℕ) (h : LoopHist α)
    (d : LoopRoundDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget) : ℕ :=
  let ps := loopPopulations K h.pops.length
  (∑ j, popDrawCalls D settings.level3.walkStart settings.level3.harvestReadCap settings.scanCap
      h.pops costs h.prefs j (d.2.1.2.1 j))
    + match clusterDraws D Dsf settings.level3.walkStart settings.level3.harvestReadCap
        settings.scanCap h (d.1, d.2.1) with
      | none => 0
      | some dr =>
        (roundPrefs (settings.schedule ps) ps (streams (fun _ => ∅) none dr)).card
            * (insert 1 (roundSuffs (settings.schedule ps) (streams (fun _ => ∅) none dr))).card
          + if ScanWithin (settings.schedule ps) ps settings.scanCap dr
              ∧ ∃ B ∈ settings.schedule ps, streams (fun _ => ∅) none dr
                ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov
                  settings.gateLevel B
            then (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2).family.card
              * Random.runReadCount
                (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2).read
                settings.level3 Random.initialState
                (List.ofFn (roundProbes settings D h (streams (fun _ => ∅) none dr) d.2.2))
            else 0

open scoped Classical in
/-- The oracle calls of the first `r` rounds `loopRounds` runs, the round that fails included. -/
def loopCalls {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
      settings.level3.probeBudget) :
    ℕ → ℕ
  | 0 => 0
  | r + 1 => loopCalls settings O D Dsf draws r
    + match loopRounds settings O D Dsf draws r with
      | some rs =>
        if h : r < K ∧ ∀ o ∈ rs, o.result.ending.harvested ≠ none then
          roundCalls settings O D Dsf (samplerCosts rs) (histOf rs) (draws ⟨r, h.1⟩)
        else 0
      | none => 0

/-- The oracle calls the loop makes. -/
def indecisiveLStarLoopCalls {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α))
    (draws : Fin K → LoopRoundDraws Ω α K settings.level3.harvestReadCap
      settings.level3.probeBudget) : ℕ :=
  loopCalls settings O D Dsf draws K

end OrthoDFA

end
