import RLStar.Proofs.ProgramEqns
import RLStar.Code.LearnerLoop
import RLStar.Proofs.Query

/-!
# The loop as functions of the oracle

The loop of `RLStar.Code.LearnerLoop` written as functions of the oracle and its noise rather
than as a program: a round carries its family's read, and each round may read its own noise.
`loopQ`'s run against one noise's answers is `loopRounds` at that noise in every round
(`RLStar.Proofs.LoopQuery`), so what is proved of these is proved of the loop.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- What the earlier rounds hand on: their harvests' lists, and every prefix and suffix their
clusterings read. -/
abbrev SpecHist (α : Type*) := LoopHist α

/-- The clustering's draws: the noise and the rest. -/
abbrev ClusterDraws (Ω α : Type*) (K : ℕ) := Ω × ClusterRest α K

/-- A round's draws with the noise it reads. -/
abbrev LoopRoundDraws {K : ℕ} (Ω α : Type*) (s : LoopSettings K) := Ω × s.RoundDraws α

variable {Ω : Type*} [MeasurableSpace Ω]

/-- The rounds' draws, each read through the run's one noise. -/
def persistDraws {K : ℕ} {s : LoopSettings K} (p : s.Draws Ω α) : Fin K → LoopRoundDraws Ω α s :=
  fun i => (p.1, p.2 i)

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
family, and the lazy round from the trie of the clustering's prefixes, its probes drawn by `u`
from `D` avoiding the first `walkStart` letters of those prefixes. -/
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
  ⟨B, F, read, (Random.lazyRoundQ settings.level3 Kw settings.listSize (Random.prefixTrie Kw CP)
      (lazyProbes (avoidLaw D Kw (clusterKeys (settings.schedule ps) ps Kw x)) u)).eval read,
    CP, roundSuffs (settings.schedule ps) x⟩

open scoped Classical in
/-- The round after the history `h`: each population's draws off what the earlier rounds read
(`popTables`), then the clustering on them (`drawsOf`) and the lazy round.  `none` if the
clustering would read past the draws or no state of the schedule returns. -/
def loopRound {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Dsf : Measure (FreeMonoid α)) (h : SpecHist α) (d : LoopRoundDraws Ω α settings) :
    Option (SpecRound α) :=
  let ps := loopPopulations K h.pops.length
  let dr := drawsOf d.1 D Dsf settings.scanCap h.prefs h.suffs d.2.1
    (popTables D settings.scanCap h d.2.1)
  if ScanWithin (settings.schedule ps) ps (fun _ => settings.scanCap) dr
      ∧ ∃ B ∈ settings.schedule ps, streams (fun _ => ∅) none dr
        ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  then some (roundOf settings O D h (streams (fun _ => ∅) none dr) d.2.2)
  else none

/-- A harvest's list. -/
def SpecRound.population (o : SpecRound α) : Option (List (FreeMonoid α)) :=
  match o.result.ending with
  | .harvest _ ℓ => some ℓ
  | _ => none

/-- What the rounds hand on: their harvests' lists, and what their clusterings read. -/
def specHistOf (rounds : List (SpecRound α)) : SpecHist α :=
  ⟨rounds.filterMap fun o => o.population,
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
