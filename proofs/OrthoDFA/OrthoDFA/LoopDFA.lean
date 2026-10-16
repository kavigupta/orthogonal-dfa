import OrthoDFA.LearnerLoop

/-!
# The learner's DFA

The Python admits a target when re-rooting it at its best covered state classifies almost every
string: `covered_states` are the states at least `min_coverage` of the strings of length `L` visit
at some position from `⌈L/2⌉` to `L`, and `covered_accuracy_ceiling` is the best share, over
covered starts, of the strings whose re-rooted run stays in the covered states and is classified
as the target classifies them. `coveredStates` and `rerootAccuracy` are the same quantities, as
probabilities rather than shares of a sample.

`LearnerLoopDFA`: where some covered start re-roots with accuracy at least `1 − η`, the loop, its
probes uniform over the lengths `⌈L/2⌉` to `L + |Q| + 3` and walked from `⌈L/2⌉`, stops within
`|Q| + 1` rounds at a consistent round whose hypothesis, run from one of its leaves, classifies all
but `η` of the strings of length `L` as the target does, but for chance `δ`. A run that reaches an
unlearned edge counts as wrong. The disagreement rate the loop runs level 5 at is `c/(2(|Σ|+1)N)`,
`N` the probes' lengths: a covered state is visited `c` of the time across the window, so a wrong
edge out of it is crossed `c/((|Σ|+1)N)` of the time at some length.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Uniform over the lengths `lo` to `hi`, and over the strings of each length. -/
def windowStrings (lo hi : ℕ) : Measure (FreeMonoid α) :=
  ((hi + 1 - lo : ℕ) : ℝ≥0∞)⁻¹ • ∑ n ∈ Finset.Icc lo hi, uniformStrings n

/-- The states at least `c` of the strings of length `L` visit at some position from `⌈L/2⌉` to
`L`. -/
def coveredStates {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (c : ℝ) : Set σ :=
  {q | c ≤ (uniformStrings L).real
    {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}}

/-- The share of the strings of length `L` on which the target re-rooted at `q` stays in `S` and
classifies as the target does. -/
def rerootAccuracy {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (S : Set σ) (q : σ) : ℝ :=
  (uniformStrings L).real {x : FreeMonoid α | (∀ i ≤ L, M.evalFrom q (prefixOf x i).toList ∈ S)
    ∧ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}

/-- From the leaf `s`, `H` follows learned edges along all of `x` to a leaf, which is on the root's
accept side exactly when `b`. -/
def HypothesisDFA.Classifies (H : HypothesisDFA α) (s : List Bool) (x : FreeMonoid α) (b : Prop) :
    Prop :=
  ∃ p, walkEnd H.edges s x.toList = some p ∧ (p.head? = some true ↔ b)

/-- The loop's DFA at settings the theorem chooses, every hypothesis an explicit inequality. With
`L` the Python's length, `k = ⌈L/2⌉` the walk's start and `L₁ = L + |Q| + 3` the longest probe,
the probes uniform over the `N = L₁ + 1 − k` lengths from `k` to `L₁`: if some state that at least
`c` of the strings of length `L` visit from `k` on re-roots the target with accuracy at least
`1 − η` without leaving such states, then with probability at least `1 − δ` the loop stops within
`|Q| + 1` rounds at a consistent round with a leaf from which its hypothesis classifies all but `η`
of the strings of length `L` as the target does. Level 5 draws at most `Kp · G` probes a round.

The theorem runs `LearnerLoopSimple`'s settings at disagreement rate `εd = c/(2(|Σ|+1)N)`, the
clustering's accuracy at `εcov` and a quarter of `δ` a round, the rest paying for reads left
undecided. -/
def LearnerLoopDFASimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c η δ pAP : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  let N : ℝ := L₁ + 1 - k
  let εd : ℝ := c / (2 * (A + 1) * N)
  let εcov : ℝ := δ * c / (512 * (Q + 1) * (Q + 3) * (A + 1) * A ^ (Fintype.card σ + 2) * N ^ 3)
  let δ' : ℝ := δ / (4 * (Q + 1))
  let ι : ℝ := min (εcov / 8) (1 / (3072 * (L₁ + 1) * (Q + 1) * Q))
  let ρ : ℝ := 3 * εd / 16
  let s : ℝ := 1 / 2 - η₀
  let τ : ℝ := min (εcov / 2) (min s ι)
  let J : ℝ := Q + 2
  let V : ℝ := (Q + 2) * ((Q + 2) * A + 1)
  let Kp : ℝ := V * (224 / εd + 1)
  let G : ℝ := Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
  let PB : ℝ := 256 * (Real.log (2 / τ)
      + 8 * Kp * G * (L₁ + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ') / (s ^ 2 * pAP)
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
  let RB : ℝ := 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 3)
  let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
      (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * (L₁ + 1)))))
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 2 ≤ Fintype.card α →
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  (∃ q ∈ coveredStates M L c, 1 - η ≤ rerootAccuracy M L (coveredStates M L c) q) →
  (A ^ k)⁻¹ ≤ cap * ρ / 4 →
  16384 * (L₁ + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * (A ^ k)⁻¹ ≤ εd →
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    (loopMeasure μ α (Fintype.card σ + 1) settings.level5.probeBudget).real
      {p | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws p)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ ∃ s ∈ o.result.hypothesis.tree.leaves,
              (uniformStrings L).real {x | ¬ o.result.hypothesis.Classifies s x
                (M.eval x.toList ∈ M.accept)} ≤ η)}
      ≤ δ

end OrthoDFA

end
