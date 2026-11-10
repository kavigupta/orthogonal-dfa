import RLStar.Claims.RLStar
import RLStar.Code.LearnerLoopTrie

/-!
# The learner with local walk starts, asymptotically

Open: stated, not proved. The loop with local walk starts, its probes and its error over the
strings of length `L`, outputs the DFA its last round spells out, with no start premise: no zone,
no visit ratio, no re-rooting accuracy and no held-out pick. Walk starts are expanded below
⌈L/2⌉ letters.
-/

namespace OrthoDFA

open MeasureTheory Asymptotics Filter

/-- `Q` states, `A` letters, `ξ` the accuracy slack, `pAP` the share of suffixes that preserve
membership, and the signal `s₀ = 1/2 − η₀`. -/
structure RLStarTrieInputs where
  Q : ℕ
  A : ℕ
  δ : ℝ
  ξ : ℝ
  pAP : ℝ
  s₀ : ℝ

def rlStarTrieRegion : Set RLStarTrieInputs :=
  {x | 1 ≤ x.Q ∧ 2 ≤ x.A ∧ 0 < x.δ ∧ x.δ ≤ 1 / 2 ∧ 0 < x.ξ ∧ x.ξ ≤ 1 ∧
    0 < x.pAP ∧ x.pAP ≤ 1 ∧ 0 < x.s₀ ∧ x.s₀ ≤ 1 / 2}

structure RLStarTrieRun extends RLStarTrieInputs where
  L : ℕ

def rlStarTrieRunRegion : Set RLStarTrieRun := {r | r.toRLStarTrieInputs ∈ rlStarTrieRegion}

/-- Within `|Q| + 1` rounds the loop ends consistent and its DFA misjudges at most `ξ` of `T`'s
strings, reading at most `totalReads` strings. -/
def RLStarTrieSucceeds {A : Type*} [Fintype A] [DecidableEq A] {Q : Type*} [Fintype Q] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K) (y : ℝ) (cap : ℕ)
    (O : Oracle μ (FreeMonoid A)) (D Dsf T : Measure (FreeMonoid A)) (M : _root_.DFA A Q) (ξ : ℝ)
    (totalReads : ℝ)
    (p : LoopDraws Ω A K settings.level3.harvestReadCap settings.level3.probeBudget) : Prop :=
  (∃ rounds, indecisiveLStarTrieLoop settings y cap O D Dsf p = some rounds ∧
    rounds.length ≤ Fintype.card Q + 1 ∧ ∃ o ∈ rounds.getLast?,
      o.result.ending = .consistent ∧ T.real {w | ¬ o.result.hypothesis.TrieClassifies
        (o.read O p.1) o.result.expanded w (M.eval w.toList ∈ M.accept)} ≤ ξ) ∧
  ∃ S : Finset (FreeMonoid A), (S.card : ℝ) ≤ totalReads ∧
    DependsOnlyOn (fun o => (trieLoopRoundsQ settings y cap D Dsf p.2 K).eval o)
      (fun w => O.mq w p.1) S

def RLStarTrieWithinUniform.{u, v, w} (L₀ : RLStarTrieInputs → ℝ)
    (totalProbes totalReads : RLStarTrieRun → ℝ) : Prop :=
  ∀ {A : Type u} [Fintype A] [DecidableEq A] [Nonempty A] {Q : Type v} [Fintype Q] {Ω : Type w}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (x : RLStarTrieInputs) (M : _root_.DFA A Q) (O : Oracle μ (FreeMonoid A)) (L : ℕ),
  let Ls : ℕ := L + 2 * Fintype.card Q + 6
  x ∈ rlStarTrieRegion → Fintype.card Q = x.Q → Fintype.card A = x.A →
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ 1 / 2 - x.s₀ →
  x.pAP ≤ (uniformStrings Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  L₀ x ≤ L →
  ∃ (settings : LoopSettings (Fintype.card Q + 1)) (y : ℝ),
    (∀ p : LoopDraws Ω A (Fintype.card Q + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget,
      ∀ rounds, indecisiveLStarTrieLoop settings y ((L + 1) / 2) O (uniformStrings L)
          (uniformStrings Ls) p = some rounds →
        (rounds.length * settings.level3.probeBudget : ℝ) ≤ totalProbes { x with L }) ∧
    (loopMeasure μ A (Fintype.card Q + 1) settings.level3.harvestReadCap
        settings.level3.probeBudget).real
      {p | ¬ RLStarTrieSucceeds settings y ((L + 1) / 2) O (uniformStrings L) (uniformStrings Ls)
        (uniformStrings L) M x.ξ (totalReads { x with L }) p}
      ≤ x.δ

open Real in
/-- Polynomially many probes and reads, at a length logarithmic in the inputs. -/
def RLStarTrieClaim.{u, v, w} : Prop :=
  ∃ (L₀ : RLStarTrieInputs → ℝ) (totalProbes totalReads : RLStarTrieRun → ℝ) (c : ℕ),
    L₀ =O[𝓟 rlStarTrieRegion] (fun x =>
      1 + log (x.Q / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A) ∧
    totalProbes =O[𝓟 rlStarTrieRunRegion] (fun r =>
      (r.Q * r.A * (r.L + 1) / (r.δ * r.ξ * r.pAP * r.s₀)) ^ c) ∧
    totalReads =O[𝓟 rlStarTrieRunRegion] (fun r =>
      (r.Q * r.A * (r.L + 1) / (r.δ * r.ξ * r.pAP * r.s₀)) ^ c) ∧
    RLStarTrieWithinUniform.{u, v, w} L₀ totalProbes totalReads

end OrthoDFA
