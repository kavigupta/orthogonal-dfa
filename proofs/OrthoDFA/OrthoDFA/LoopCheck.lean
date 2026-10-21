import OrthoDFA.LearnerLoop
import OrthoDFA.Level3Check

/-!
# The learner's loop with the start check

`indecisiveLStarCheckLoop` is `indecisiveLStarLoop` with each round's level 3 replaced by the
checked round (`indecisiveLStarCheck`), started from what earlier rounds handed on (`h₀`). The
clustering's uniform pool is drawn from `Du`. Each of the round's draws is a probe, drawn from `D` with its first `avoidDepth` letters avoiding those of
the prefixes the round's clustering read, and a block of check strings, each drawn from `Dc` with
its first `checkDepth` letters avoiding them. The rounds go on while each ends in a harvest.

Known modelling gap. The Python raises `k` from `0` by one at each round whose harvest repeats an
earlier one, up to `⌈L/2⌉`; the model covers the run from its first round at `⌈L/2⌉`. What the
earlier rounds built is `h₀`: their deep populations, whose strings are at least `⌈L/2⌉` long, as
the Python keeps at the switch, and what their clusterings read.

Known modelling gap. The check strings avoid the first `checkDepth` letters of the prefixes the
round's clustering read, so that no check string's prefix followed by a midfix is one of them. The
Python draws them freely.

Known modelling gap. The Python draws its uniform pool at length `L`, as it does its probes; here
the pool is drawn from `Du`, half at length `L` and half over the window of lengths `⌈L/2⌉` to `L`
(`poolMix`), so that every covered state has pool mass.

Known modelling gap. Every population is drawn off the strings the earlier rounds' clusterings
read (`popOff`), and off them the inherited ones too; the Python reads its tables' strings again.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]

/-- The loop's settings and the check's: its rate, level and size, and how deep its strings
avoid the clustering's prefixes. -/
structure CheckLoopSettings (K : ℕ) extends LoopSettings K where
  check : Random.CheckSettings
  checkDepth : ℕ

/-- A round's draws apart from the noise: the clustering's, and for each of level 3's draws the
uniforms of its probe and of its check strings. -/
abbrev CheckRoundDraws (α : Type*) (K P n : ℕ) :=
  ClusterRest α K × (Fin P → unitInterval × (Fin n → unitInterval))

def checkRoundLaw (α : Type*) (K P n : ℕ) : Measure (CheckRoundDraws α K P n) :=
  (restLaw α K).prod (Measure.pi fun _ : Fin P =>
    (volume : Measure unitInterval).prod (Measure.pi fun _ : Fin n => volume))

/-- One noise for the whole run and `K` rounds' uniforms, independent. -/
def checkLoopMeasure (μ : Measure Ω) (α : Type*) (K P n : ℕ) :
    Measure (Ω × (Fin K → CheckRoundDraws α K P n)) :=
  μ.prod (Measure.pi fun _ : Fin K => checkRoundLaw α K P n)

/-- `h₀`'s populations and reads before `h`'s. -/
def LoopHist.append (h₀ h : LoopHist α) : LoopHist α :=
  ⟨h₀.pops ++ h.pops, h₀.prefs ∪ h.prefs, h₀.suffs ∪ h.suffs⟩

/-- Uniform over the lengths `lo` to `hi`, and over the strings of each length. -/
def windowStrings (lo hi : ℕ) : Measure (FreeMonoid α) :=
  ((hi + 1 - lo : ℕ) : ENNReal)⁻¹ • ∑ n ∈ Finset.Icc lo hi, uniformStrings n

/-- Half the strings of length `L`, half the window of lengths `k` to `L`. -/
def poolMix (k L : ℕ) : Measure (FreeMonoid α) :=
  (2 : ENNReal)⁻¹ • uniformStrings L + (2 : ENNReal)⁻¹ • windowStrings k L

/-- The states at least `c` of the strings of length `L` visit at some position from `⌈L/2⌉` to
`L`. -/
def coveredStates {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (c : ℝ) : Set σ :=
  {q | c ≤ (uniformStrings L).real
    {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}}

/-- The start `q` re-roots the target on every string of each length from `⌈L/2⌉` to `2L`, its
run staying in the covered states. -/
def CoveredAccurate {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (c : ℝ) (q : σ) : Prop :=
  ∀ ℓ, (L + 1) / 2 ≤ ℓ → ℓ ≤ 2 * L → ∀ x : FreeMonoid α, x.toList.length = ℓ →
    (∀ i ≤ ℓ, M.evalFrom q (prefixOf x i).toList ∈ coveredStates M L c)
      ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)

variable {μ : Measure Ω}

/-- The round after the history `h`, with the checked round in level 3's place. -/
def checkRound {K : ℕ} (settings : CheckLoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Du Dsf Dc : Measure (FreeMonoid α)) (h : LoopHist α)
    (d : Ω × CheckRoundDraws α K settings.level3.probeBudget settings.check.size) :
    LoopRound Ω α K :=
  let ps := loopPopulations K h.pops.length
  let x := clusterRun Du Dsf h (d.1, d.2.1)
  let B := Classical.epsilon fun B => B ∈ settings.schedule ps
    ∧ x ∈ ret O.mq ps none settings.η₀ settings.indecisionLimit settings.εcov settings.gateLevel B
  let F := clusterAt O.mq ps x B
  let read := fun z => familyRead O.mq F B.lo (B.hi + 1) z d.1
  let K₀ := clusterKeys (settings.schedule ps) ps settings.avoidDepth x
  let Kc := clusterKeys (settings.schedule ps) ps settings.checkDepth x
  ⟨x, B, F, read, Random.indecisiveLStarCheck settings.level3 settings.check read
    (fun i => (drawFrom (avoidLaw D settings.avoidDepth K₀) (d.2.2 i).1,
      fun j => drawFrom (avoidLaw Dc settings.checkDepth Kc) ((d.2.2 i).2 j))),
    roundPrefs (settings.schedule ps) ps x, roundSuffs (settings.schedule ps) x⟩

/-- The first `r` rounds after `h₀`, cut short at a round that does not end in a harvest. -/
def checkRounds {K : ℕ} (settings : CheckLoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Du Dsf Dc : Measure (FreeMonoid α)) (h₀ : LoopHist α)
    (p : Ω × (Fin K → CheckRoundDraws α K settings.level3.probeBudget settings.check.size)) :
    ℕ → List (LoopRound Ω α K)
  | 0 => []
  | r + 1 =>
    if h : r < K ∧ ∀ o ∈ checkRounds settings O D Du Dsf Dc h₀ p r,
        o.result.ending.harvested ≠ none
    then checkRounds settings O D Du Dsf Dc h₀ p r
        ++ [checkRound settings O D Du Dsf Dc
          (h₀.append (histOf D settings.level3.walkStart (checkRounds settings O D Du Dsf Dc h₀ p r)))
          (p.1, p.2 ⟨r, h.1⟩)]
    else checkRounds settings O D Du Dsf Dc h₀ p r

/-- The rounds the loop runs after `h₀`, at most `K`. -/
def indecisiveLStarCheckLoop {K : ℕ} (settings : CheckLoopSettings K) (O : Oracle μ (FreeMonoid α))
    (D Du Dsf Dc : Measure (FreeMonoid α)) (h₀ : LoopHist α)
    (p : Ω × (Fin K → CheckRoundDraws α K settings.level3.probeBudget settings.check.size)) :
    List (LoopRound Ω α K) :=
  checkRounds settings O D Du Dsf Dc h₀ p K

end OrthoDFA

end
