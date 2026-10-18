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
length; the probes a consistent round answers for, those whose start or end sift stops undecided
among them, must be fewer. Level 5's start and edge tests certify the undecided sifts, at a rate
`(L₁ + 1)(|Q| + 2)(|Σ| + 1)` times its disagreement rate.

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

/-- The loop's DFA at settings the theorem chooses, every hypothesis an explicit inequality. With
`L` the Python's length, `k = ⌈L/2⌉` the walk's start and `L₁ = L + |Q| + 3` the longest probe,
the probes uniform over the `N = L₁ + 1 − k` lengths from `k` to `L₁`: if some state that at least
`c` of the strings of length `L` visit from `k` on re-roots the target with accuracy at least
`1 − η` without leaving such states, then with probability at least `1 − δ` the loop stops within
`|Q| + 1` rounds at a consistent round, and the leaf `pickLeaf` picks on `n` held-out strings
uniform of length `L` classifies all but `η + ξ` of the strings of length `L` as the target does.
Level 5 draws at most `Kp · G` probes a round.

The theorem runs `LearnerLoopSimple`'s settings at disagreement rate `εd/2`, with level 5's start
test at `εd` against `2εd` and its edges' at rate `εd` and traffic `εd/((|Q|+2)|Σ|)`, so that a
harvest's sampler yields `ρ/2`. The clustering runs at accuracy `εcov`, small enough that level 5's
good states are undecided rarely against those gates. `δ` splits into half for the loop, a quarter
for the scores' deviation and a quarter for a held-out string whose reads some round read: one
whose first `k` letters are those of a prefix the last round's clustering read or of one of its
probes, or one drawn twice. -/
def LearnerLoopDFASimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c η δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  let N : ℝ := L₁ + 1 - k
  let εd : ℝ := c / (34 * (A + 1) ^ 2 * (Q + 2) * N * (L₁ + 1))
  let εcov : ℝ := min (εd / (48 * (Q + 1) ^ 2)) (ξ / (8 * N))
  let δ' : ℝ := δ / (8 * (Q + 1))
  let ι : ℝ := min (εcov / 8) (1 / (3072 * (L₁ + 1) * (Q + 1) * Q))
  let ρ : ℝ := εd ^ 2 / ((Q + 2) * A)
  let s : ℝ := 1 / 2 - η₀
  let τ : ℝ := min (εcov / 2) (min s ι)
  let J : ℝ := Q + 2
  let V : ℝ := (Q + 2) * ((Q + 2) * A + 1)
  let Kp : ℝ := V * (224 / εd + 1)
  let G : ℝ := Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
  let PB : ℝ := 256 * (Real.log (2 / τ)
      + 4 * Real.log (4 * Kp * G * (L₁ + 1) * (Q + 3) * A ^ (Fintype.card σ + 2) / δ'))
      / (s ^ 2 * pAP)
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
  let RB : ℝ := 6000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (1 / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 3)
  let cap : ℝ := min (min (δ' * τ ^ 2 / (2 ^ 22 * (J + 1) ^ 2 * (RB + 1)))
      (δ' * ι ^ 2 / (2 ^ 11 * (J + 1) * (RB + 1))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1) * (L₁ + 1)))))
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  (∃ q ∈ coveredStates M L c, 1 - η ≤ rerootAccuracy M L (coveredStates M L c) q) →
  (A ^ k)⁻¹ ≤ cap * ρ / 4 →
  16384 * (L₁ + 1) * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * (A ^ k)⁻¹ ≤ ρ →
  (Q + 1) * n * (6 * J * RB + Kp * G + n) * (A ^ k)⁻¹ ≤ δ / 4 →
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level5.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
                (M.eval x.toList ∈ M.accept)} ≤ η + ξ)}
      ≤ δ

/-- The learner's DFA once the Python's length `L` is at least `L₀`. With the walk starting at
`k = ⌈L/2⌉`, probes uniform over the lengths `k` to `L₁ = L + |Q| + 3` and suffixes uniform at
length `L₁ + |Q| + 3`, of which at least `pAP` keep every string's membership: if some state that
at least `c` of the strings of length `L` visit from `k` on re-roots the target with accuracy at
least `1 − η` without leaving such states, then with probability at least `1 − δ` the loop stops
within `|Q| + 1` rounds at a consistent round, and the leaf `pickLeaf` picks on `n` held-out
strings uniform of length `L` classifies all but `η + ξ` of the strings of length `L` as the target
does. Level 5 runs at `εd/2`, `εd = c/(34(|Σ|+1)²(|Q|+2)N(L₁+1))`, and draws at most `Kp · G`
probes a round, `N = L₁ + 1 − k` the probes' lengths.

`L₀` is `|Q| + 4` plus `4 (log (E + E₂) + 100)/log |Σ|`, `E` and `E₂` free of `L`: the claim's
`cap` is at least `1/((L₁ + 1)¹⁶ E₁)`, `ρ` at least `1/((L₁ + 1)⁴ (|Q|+2)|Σ| c₂²)` with
`c₂ = 34(|Σ|+1)²(|Q|+2)/c`, the prefix count at most `(L₁ + 1)¹⁰ R₀`, and
`20 log (L₁ + 1) ≤ 100 + (L₁ + 1) log |Σ| / 4` absorbs the length. -/
def LearnerLoopDFA : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c η δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  let N : ℝ := L₁ + 1 - k
  let εd : ℝ := c / (34 * (A + 1) ^ 2 * (Q + 2) * N * (L₁ + 1))
  let δ' : ℝ := δ / (8 * (Q + 1))
  let s : ℝ := 1 / 2 - η₀
  let V : ℝ := (Q + 2) * ((Q + 2) * A + 1)
  let Kp : ℝ := V * (224 / εd + 1)
  let G : ℝ := Real.log (4 * V * (2 * (Q + 2) * A + 4) / δ')
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  let Ce : ℝ := 1632 * (Q + 1) ^ 2 * (A + 1) ^ 2 * (Q + 2) / c + 8 / ξ
  let Θ : ℝ := 1 / s + 8 * Ce + 3072 * Q * (Q + 1)
  let K₁ : ℝ := V * (7616 * (A + 1) ^ 2 * (Q + 2) / c + 1)
  let B₁ : ℝ := 4 * (3 + Real.log (4 * K₁ * G * (Q + 3) * A ^ (Fintype.card σ + 2) / δ'))
  let P₀ : ℝ := 256 * (2 * Θ + B₁) / (s ^ 2 * pAP) + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
  let R₀ : ℝ := 6000 * (Q + 2) ^ 2 * Θ ^ 3
    * (Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * (pAP / 2))) + 6) / s ^ 4
  let E₁ : ℝ := 2 ^ 24 * (Q + 3) ^ 2 * (Θ ^ 2 * (R₀ + 1) + (P₀ + 2) / pAP) / δ'
  let E : ℝ := 4 * (Q + 2) * A * (34 * (A + 1) ^ 2 * (Q + 2) / c) ^ 2 * E₁
    + 16384 * (Q + 2) ^ 2 * A ^ 2 * (34 * (A + 1) ^ 2 * (Q + 2) / c) ^ 2
      * Real.log ((Q + 3) * A / δ')
  let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ * G + n) / δ
  let L₀ : ℝ := Q + 4 + 4 * (Real.log (E + E₂) + 100) / Real.log A
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  (∃ q ∈ coveredStates M L c, 1 - η ≤ rerootAccuracy M L (coveredStates M L c) q) →
  L₀ ≤ L →
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level5.walkStart = k ∧ settings.level5.disagreementRate = εd / 2 ∧
    (settings.level5.probeBudget : ℝ) ≤ Kp * G ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level5.probeBudget).prod
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
