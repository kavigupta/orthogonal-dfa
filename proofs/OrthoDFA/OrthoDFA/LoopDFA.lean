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
`|Q| + 1` rounds at a consistent round, and the leaf it then picks classifies all but `η + ξ` of the
strings of length `L` as the target does, but for chance `δ`. The leaf is picked on `n` held-out
strings of length `L`, drawn after the loop: each leaf is scored by how many of them its run
classifies as the last round's family reads them, the undecided ones counting for none. A run that
reaches an unlearned edge counts as wrong. A covered state is visited `c` of the time across the
window of `N` lengths, so a wrong edge out of it is crossed `c/((|Σ|+1)N)` of the time at some
length; the probes that disagree in a consistent round, searching, stopping undecided at their
start or end sift, or reading a string on its state's rarer side, must be fewer. Level 3's own
success test bounds the first two at its disagreement rate.

Known modelling gap. The Python draws its probes at one length; here they are uniform over the
window of lengths, so that every position a run can use an edge at is some probe's last step.

Known modelling gap. The Python estimates `covered_states` and `covered_accuracy_ceiling` on 2000
sampled strings; here they are exact probabilities.

Known modelling gap. The Python starts its DFA at the leaf the empty string sifts to; here the
start is picked on held-out strings.
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

open scoped Classical in
/-- How many of `xs` the hypothesis, run from the leaf `s`, classifies as `read` decides them. -/
def leafScore (H : HypothesisDFA α) (read : FreeMonoid α → ARU) {n : ℕ}
    (xs : Fin n → FreeMonoid α) (s : List Bool) : ℕ :=
  (Finset.univ.filter fun i => read (xs i) ≠ .undecided
    ∧ H.Classifies s (xs i) (read (xs i) = .accept)).card

/-- The leaf of highest `leafScore`, the first of those tied. -/
def pickLeaf (H : HypothesisDFA α) (read : FreeMonoid α → ARU) {n : ℕ}
    (xs : Fin n → FreeMonoid α) : List Bool :=
  (H.tree.leaves.argmax (leafScore H read xs)).getD []

/-- The loop's DFA, run from a leaf picked on held-out strings, classifies all but `η + ξ` of the
strings of length `L`, but for chance `δ`; every premise an explicit inequality. -/
def LearnerLoopDFASimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c η δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- Each round's disagreement rate, low enough to catch a wrong edge out of any covered state.
  let ε : ℝ := c / (4 * (A + 1) * N)
  -- The clustering's accuracy.
  let εcov : ℝ := min ε (ξ / (8 * N))
  -- Each round's share of `δ`.
  let δ' : ℝ := δ / (8 * (Q + 1))
  -- The clustering's indecision limit: good states' undecided reads, over every round and state,
  -- stay under `ε`.
  let ι : ℝ := min (εcov / 8) (ε / (13824 * Q * (Q + 1)))
  let s : ℝ := 1 / 2 - η₀
  -- The clustering gate's rate scale.
  let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  -- The clustering's populations.
  let J : ℝ := Q + 2
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The strings a round can read wrong; the clustering's band must keep them right.
  let W : ℝ := B₃ * (L₁ + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
  -- The clustering's suffix pool.
  let PB : ℝ := 256 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
  -- The clustering's prefix count.
  let RB : ℝ := 60000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  -- No string may hold more than this of a population or of the suffixes.
  let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
      (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  -- The held-out strings the start leaf is picked on.
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's precondition: some covered state re-roots the target with error at most `η`.
  (∃ q ∈ coveredStates M L c, 1 - η ≤ rerootAccuracy M L (coveredStates M L c) q) →
  -- The walk start's prefixes are spread thinly enough for the clustering's collisions,
  (A ^ k)⁻¹ ≤ cap * ε / 96 →
  -- for each round's good-state noise to concentrate,
  768 * Real.log (16 * Kc / δ') * (A ^ k)⁻¹ ≤ ε →
  -- and for the held-out strings to read only fresh strings.
  (Q + 1) * n * (6 * J * RB + B₃ + n) * (A ^ k)⁻¹ ≤ δ / 4 →
  -- Within `|Q| + 1` rounds the loop ends consistent, and the picked leaf misclassifies at most
  -- `η + ξ`, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
                (M.eval x.toList ∈ M.accept)} ≤ η + ξ)}
      ≤ δ

/-- `LearnerLoopDFASimple` once the Python's length `L` is at least `L₀`. -/
def LearnerLoopDFA : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c η δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- Each round's disagreement rate, low enough to catch a wrong edge out of any covered state.
  let ε : ℝ := c / (4 * (A + 1) * N)
  -- Each round's share of `δ`.
  let δ' : ℝ := δ / (8 * (Q + 1))
  let s : ℝ := 1 / 2 - η₀
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The held-out strings the start leaf is picked on.
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The rest bound `LearnerLoopDFASimple`'s premises with the length `ℓ = L₁ + 1` absorbed:
  -- `1/cap ≤ ℓ⁵ E₁`, `1/ε ≤ ℓ aε`, the prefix count `≤ ℓ³ R₀`, the budget `≤ ℓ K₁`.
  let aε : ℝ := 4 * (A + 1) / c
  let Ce : ℝ := aε + 8 / ξ
  let Θ : ℝ := Ce / (2 * s ^ 2) + 8 * Ce + 13824 * Q * (Q + 1) * aε
  let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
  let W₀ : ℝ := K₁ * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ * aε
  let P₀ : ℝ := 256 * (Real.log (2 * Θ) + 4 * Real.log (8 * W₀ / δ') + 9) / (s ^ 2 * pAP)
    + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
  let R₀ : ℝ := 60000 * (Q + 2) ^ 2 * Θ ^ 2
    * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
  let E₁ : ℝ := 2 ^ 23 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP) / δ'
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  let E : ℝ := 96 * aε * E₁ + 768 * Real.log (16 * Kc / δ') * aε
  let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
  let L₀ : ℝ := Q + 4 + 4 * (Real.log (E + E₂) + 30) / Real.log A
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's precondition: some covered state re-roots the target with error at most `η`.
  (∃ q ∈ coveredStates M L c, 1 - η ≤ rerootAccuracy M L (coveredStates M L c) q) →
  -- The Python's length is long enough.
  L₀ ≤ L →
  -- Within `|Q| + 1` rounds the loop ends consistent, and the picked leaf misclassifies at most
  -- `η + ξ`, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
                (M.eval x.toList ∈ M.accept)} ≤ η + ξ)}
      ≤ δ

end OrthoDFA

end
