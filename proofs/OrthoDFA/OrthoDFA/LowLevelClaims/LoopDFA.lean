import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Code.LearnerLoop
import OrthoDFA.Code.LoopDFA
import OrthoDFA.LowLevelClaims.CoveredStart

/-!
# The learner's DFA

The Python admits a target when re-rooting it at its best covered state classifies almost every
string: `covered_states` are the states at least `min_coverage` of the strings of length `L` visit
at some position from `⌈L/2⌉` to `L`, and `covered_accuracy_ceiling` is the best share, over
covered starts, of the strings whose re-rooted run stays in the covered states and is classified
as the target classifies them. `coveredStates` and `rerootAccuracy` are the same quantities, as
probabilities rather than shares of a sample.

`LearnerLoopDFA`: where some covered start re-roots with accuracy at least `1 − γ`, the loop, its
probes uniform over the lengths `⌈L/2⌉` to `L + |Q| + 3` and walked from `⌈L/2⌉`, stops within
`|Q| + 1` rounds at a consistent round, and its re-rooted DFA (`rerootedDFA`) misclassifies at most
`γ + ξ` of the strings of length `L`, but for chance `δ`. The final hypothesis is started at the
leaf that scores best on `n` held-out strings of length `L`, drawn after the loop: each leaf is
scored by how many of them its run classifies as the last round's family reads them, the undecided
ones counting for none. A run that reaches an unlearned edge counts as wrong. A covered state is visited `c` of the time across the
window of `N` lengths, so a wrong edge out of it is crossed `c/((|Σ|+1)N)` of the time at some
length by probes that disagree in a consistent round, searching or stopping undecided at their
start or end sift, unless the probe or its extension reads a string on its state's rarer side.
Level 3's own success test bounds the disagreeing probes at its disagreement rate. Which probes
cross the edge is fixed by the target before any read, and a misread is rare and independent of
it, so but for chance at most half of them, and a little more, are misled.

`LearnerLoopDFARelative` admits the target by a relative rule instead: a start may re-root it
through the states whose per-position visit rate from it is at most `R` times that of a string from
`⌈L/2⌉` on (`nonStartZoneStates`), itself among them. With `C = R(L+1)/(L+1−⌈L/2⌉)`, it is
`LearnerLoopDFA` at `c = ξ/(2C|Q|)` and `ξ/2`: the start is in its own zone so covered at `1/C`, and
the non-start-zone states that fewer than `c` of the strings visit from `⌈L/2⌉` on cost the other
`ξ/2`.

`LearnerLoopDFADist` is `LearnerLoopDFASimple` over strings drawn from `D`: the probes are `D`'s
strings cut at a length uniform over the window (`windowLaw`), the suffixes their first
`L + 2|Q| + 6` letters, and the coverage, error and held-out strings are over their first `L`
(`prefixLaw`, `coveredStatesOn`, `rerootAccuracyOn`). Where `D` extends each prefix by each letter
at least `β` of the time, a wrong edge is crossed `c/((1/β+1)N)` of the time. Each collision the
proof rules out, among suffixes, prefixes and held-out strings, lies within one cylinder of first
`k` letters, so `D`'s largest such mass `p₀` takes `|Σ|^-k`'s place. The uniform claims are it at
`β = 1/|Σ|`.

Known modelling gap. The Python draws its probes at one length; here they are uniform over the
window of lengths, so that every position a run can use an edge at is some probe's last step.

Known modelling gap. The Python estimates `non_start_zone_states` and
`non_start_zone_accuracy_ceiling` on 2000 sampled strings; here they are exact probabilities.

Known modelling gap. The Python's relative rule may start at any state its 2000 samples visit in the
window; `LearnerLoopDFARelative`'s start must be covered at `c = ξ/(2C|Q|)`.

Known modelling gap. The Python starts its DFA at the leaf the empty string sifts to; here the
start is picked on held-out strings.

Known modelling gap. The Python draws its probes, prefixes and suffixes from one sampler of length
`L`; `LearnerLoopDFADist` draws its suffixes as `D`'s first `L + 2|Q| + 6` letters, so `D`'s strings
must be at least that long.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Uniform over the lengths `lo` to `hi`, and over the strings of each length. -/
def windowStrings (lo hi : ℕ) : Measure (FreeMonoid α) :=
  ((hi + 1 - lo : ℕ) : ℝ≥0∞)⁻¹ • ∑ n ∈ Finset.Icc lo hi, uniformStrings n

/-- `D`'s strings cut at a length uniform over `lo` to `hi`. -/
def windowLaw (D : Measure (FreeMonoid α)) (lo hi : ℕ) : Measure (FreeMonoid α) :=
  ((hi + 1 - lo : ℕ) : ℝ≥0∞)⁻¹ • ∑ n ∈ Finset.Icc lo hi, prefixLaw D n

/-- The share of `T`'s strings that visit `q` at some position from `⌈L/2⌉` to `L`. -/
def windowVisitRateOn {σ : Type*} (T : Measure (FreeMonoid α)) (M : _root_.DFA α σ) (L : ℕ)
    (q : σ) : ℝ :=
  T.real {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}

/-- The states at least `c` of `T`'s strings visit at some position from `⌈L/2⌉` to `L`. -/
def coveredStatesOn {σ : Type*} (T : Measure (FreeMonoid α)) (M : _root_.DFA α σ) (L : ℕ)
    (c : ℝ) : Set σ :=
  {q | c ≤ windowVisitRateOn T M L q}

/-- The share of `T`'s strings whose run from `q` visits `q'` at some position up to `L`. -/
def rerootVisitRateOn {σ : Type*} (T : Measure (FreeMonoid α)) (M : _root_.DFA α σ) (L : ℕ)
    (q q' : σ) : ℝ :=
  T.real {x : FreeMonoid α | ∃ i ≤ L, M.evalFrom q (prefixOf x i).toList = q'}

/-- The states whose visit rate from `q` is at most `R` per position times that from `⌈L/2⌉` on:
the factor is the run's `L + 1` positions over the window's. -/
def nonStartZoneStatesOn {σ : Type*} (T : Measure (FreeMonoid α)) (M : _root_.DFA α σ) (L : ℕ)
    (R : ℝ) (q : σ) : Set σ :=
  {q' | rerootVisitRateOn T M L q q' ≤
    R * (L + 1) / ((L + 1 - (L + 1) / 2 : ℕ) : ℝ) * windowVisitRateOn T M L q'}

/-- `coveredStatesOn` the strings of length `L`. -/
def coveredStates {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (c : ℝ) : Set σ :=
  coveredStatesOn (uniformStrings L) M L c

/-- `windowVisitRateOn` the strings of length `L`. -/
def windowVisitRate {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (q : σ) : ℝ :=
  windowVisitRateOn (uniformStrings L) M L q

/-- `rerootVisitRateOn` the strings of length `L`. -/
def rerootVisitRate {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (q q' : σ) : ℝ :=
  rerootVisitRateOn (uniformStrings L) M L q q'

/-- `nonStartZoneStatesOn` the strings of length `L`. -/
def nonStartZoneStates {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (R : ℝ) (q : σ) : Set σ :=
  nonStartZoneStatesOn (uniformStrings L) M L R q

/-- The loop's re-rooted DFA misclassifies at most `γ + ξ` of the strings of length `L`, but for
chance `δ`; every premise an explicit inequality. -/
def LearnerLoopDFASimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c γ δ pAP ξ : ℝ),
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
  -- The clustering's indecision limit: a harvest puts more than `3ι/T` of its population on a
  -- state read undecided more than `T` of the time, `T` a fixed threshold, so each harvest has a
  -- new such state.
  let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
  let s : ℝ := 1 / 2 - η₀
  -- The clustering gate's rate scale.
  let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  -- The clustering's populations.
  let J : ℝ := Q + 2
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The strings a round reads undecided; the clustering's band must keep their wrong reads within
  -- level 5's race tail.
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1
  -- The clustering's suffix pool.
  let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP
  -- The clustering's prefix count.
  let RB : ℝ := 161000 * J * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  -- No string may hold more than this of a population or of the suffixes.
  let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
        (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
      (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
        (ι / 2 / (8 * J * (RB + 1)))))
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
  -- The Python's precondition: some covered state re-roots the target with error at most `γ`.
  (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) →
  -- The walk start's prefixes are spread thinly enough for the clustering's collisions,
  (A ^ k)⁻¹ ≤ cap * ε / 1536 →
  -- for each round's harvests to concentrate at every state's rate,
  1536 * Real.log (16 * (Q + 1) * Kc / δ') * (A ^ k)⁻¹ ≤ ε →
  -- for the held-out strings to read only fresh strings,
  (Q + 1) * n * (6 * J * RB + B₃ + n) * (A ^ k)⁻¹ ≤ δ / 4 →
  -- and for no two of a round's probes to share their walk start.
  16 * B₃ ^ 2 * (A ^ k)⁻¹ ≤ δ' →
  -- Within `|Q| + 1` rounds the loop ends consistent, and its re-rooted DFA (the final hypothesis
  -- started at the leaf that scores best on `n` held-out strings) misclassifies at most `γ + ξ` of
  -- the strings of length `L`, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 x
                (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ≤ δ

/-- `LearnerLoopDFASimple` once the Python's length `L` is at least `L₀`. -/
def LearnerLoopDFA : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c γ δ pAP ξ : ℝ),
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
  -- `1/cap ≤ ℓ³ (1 + log ℓ) E₁`, `1/ε ≤ ℓ aε`, the prefix count `≤ ℓ² (1 + log ℓ) R₀`, the budget
  -- `≤ ℓ K₁`, `W ≤ ℓ² W₀`.
  let aε : ℝ := 4 * (A + 1) / c
  let Ce : ℝ := aε + 8 / ξ
  let Θ : ℝ := Ce / (2 * s ^ 2) + 8 * Ce + 4608 * Q * (Q + 1) * aε
  let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W₀ : ℝ := Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
  let P₀ : ℝ := (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP
  let R₀ : ℝ := 161000 * (Q + 2) * Θ ^ 2
    * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
  let E₁ : ℝ := (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
    + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  let E : ℝ := 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε
  let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
  let E₃ : ℝ := 16 * K₁ ^ 2 / δ'
  -- `1 + log ℓ` is bounded by `ℓ`, and `log ℓ` by its tangent at `T`.
  let T : ℝ := 2 * (Real.log (E + E₂ + E₃) + 10) / Real.log A + Q + 4
  let L₀ : ℝ := 2 * (Real.log (E + E₂ + E₃) + 5 * Real.log T) / Real.log A
    + 100 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 10))
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → 2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's precondition: some covered state re-roots the target with error at most `γ`.
  (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (coveredStates M L c) q) →
  -- The Python's length is long enough.
  L₀ ≤ L →
  -- Within `|Q| + 1` rounds the loop ends consistent, and its re-rooted DFA (the final hypothesis
  -- started at the leaf that scores best on `n` held-out strings) misclassifies at most `γ + ξ` of
  -- the strings of length `L`, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 x
                (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ≤ δ

/-- `LearnerLoopDFASimple` with the target admitted by the relative rule: if some state `q` in
`nonStartZoneStates M L R q` re-roots the target with accuracy at least `1 − γ` without leaving it,
the conclusion is `LearnerLoopDFASimple`'s. The settings are its at `c = ξ/(2C|Q|)`,
`C = R(L+1)/(L+1−k)`, with `ξ/2` in place of its `ξ`. -/
def LearnerLoopDFARelativeSimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ R γ δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let C : ℝ := R * (L + 1) / ((L + 1 - k : ℕ) : ℝ)
  -- The coverage the learner runs at, low enough that the non-start-zone states below it cost
  -- `ξ/2`.
  let c : ℝ := ξ / (2 * C * Q)
  let L₁ : ℕ := L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- Each round's disagreement rate, low enough to catch a wrong edge out of any covered state.
  let ε : ℝ := c / (4 * (A + 1) * N)
  -- The clustering's accuracy.
  let εcov : ℝ := min ε (ξ / (16 * N))
  -- Each round's share of `δ`.
  let δ' : ℝ := δ / (8 * (Q + 1))
  -- The clustering's indecision limit: a harvest puts more than `3ι/T` of its population on a
  -- state read undecided more than `T` of the time, `T` a fixed threshold, so each harvest has a
  -- new such state.
  let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
  let s : ℝ := 1 / 2 - η₀
  -- The clustering gate's rate scale.
  let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  -- The clustering's populations.
  let J : ℝ := Q + 2
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The strings a round reads undecided; the clustering's band must keep their wrong reads within
  -- level 5's race tail.
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1
  -- The clustering's suffix pool.
  let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP
  -- The clustering's prefix count.
  let RB : ℝ := 161000 * J * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  -- No string may hold more than this of a population or of the suffixes.
  let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
        (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
      (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
        (ι / 2 / (8 * J * (RB + 1)))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  -- The held-out strings the start leaf is picked on.
  let n : ℕ := ⌈128 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < R → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → ξ ≤ 1 →
  2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's relative precondition: some state in its own non-start zone re-roots the target
  -- with error at most `γ` through it.
  (∃ q, q ∈ nonStartZoneStates M L R q ∧
    1 - γ ≤ rerootAccuracy M L (nonStartZoneStates M L R q) q) →
  -- The walk start's prefixes are spread thinly enough for the clustering's collisions,
  (A ^ k)⁻¹ ≤ cap * ε / 1536 →
  -- for each round's harvests to concentrate at every state's rate,
  1536 * Real.log (16 * (Q + 1) * Kc / δ') * (A ^ k)⁻¹ ≤ ε →
  -- for the held-out strings to read only fresh strings,
  (Q + 1) * n * (6 * J * RB + B₃ + n) * (A ^ k)⁻¹ ≤ δ / 4 →
  -- and for no two of a round's probes to share their walk start.
  16 * B₃ ^ 2 * (A ^ k)⁻¹ ≤ δ' →
  -- Within `|Q| + 1` rounds the loop ends consistent, and its re-rooted DFA (the final hypothesis
  -- started at the leaf that scores best on `n` held-out strings) misclassifies at most `γ + ξ` of
  -- the strings of length `L`, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 x
                (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ≤ δ

/-- `LearnerLoopDFA` with the target admitted by the relative rule, as in
`LearnerLoopDFARelativeSimple`: its settings and `L₀` are `LearnerLoopDFA`'s at `c = ξ/(2C|Q|)`
for `C = R(L+1)/(L+1−k)`, with `ξ/2` in place of its `ξ`. -/
def LearnerLoopDFARelative : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ R γ δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let C : ℝ := R * (L + 1) / ((L + 1 - k : ℕ) : ℝ)
  -- The coverage the learner runs at, low enough that the non-start-zone states below it cost
  -- `ξ/2`.
  let c : ℝ := ξ / (2 * C * Q)
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
  let n : ℕ := ⌈128 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The rest bound `LearnerLoopDFASimple`'s premises with the length `ℓ = L₁ + 1` absorbed:
  -- `1/cap ≤ ℓ³ (1 + log ℓ) E₁`, `1/ε ≤ ℓ aε`, the prefix count `≤ ℓ² (1 + log ℓ) R₀`, the budget
  -- `≤ ℓ K₁`, `W ≤ ℓ² W₀`.
  let aε : ℝ := 4 * (A + 1) / c
  let Ce : ℝ := aε + 16 / ξ
  let Θ : ℝ := Ce / (2 * s ^ 2) + 8 * Ce + 4608 * Q * (Q + 1) * aε
  let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W₀ : ℝ := Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
  let P₀ : ℝ := (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP
  let R₀ : ℝ := 161000 * (Q + 2) * Θ ^ 2
    * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
  let E₁ : ℝ := (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
    + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  let E : ℝ := 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε
  let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
  let E₃ : ℝ := 16 * K₁ ^ 2 / δ'
  -- `1 + log ℓ` is bounded by `ℓ`, and `log ℓ` by its tangent at `T`.
  let T : ℝ := 2 * (Real.log (E + E₂ + E₃) + 10) / Real.log A + Q + 4
  let L₀ : ℝ := 2 * (Real.log (E + E₂ + E₃) + 5 * Real.log T) / Real.log A
    + 100 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 10))
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < R → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → ξ ≤ 1 →
  2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's relative precondition: some state in its own non-start zone re-roots the target
  -- with error at most `γ` through it.
  (∃ q, q ∈ nonStartZoneStates M L R q ∧
    1 - γ ≤ rerootAccuracy M L (nonStartZoneStates M L R q) q) →
  -- The Python's length is long enough.
  L₀ ≤ L →
  -- Within `|Q| + 1` rounds the loop ends consistent, and its re-rooted DFA (the final hypothesis
  -- started at the leaf that scores best on `n` held-out strings) misclassifies at most `γ + ξ` of
  -- the strings of length `L`, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => uniformStrings L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
            (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 x
                (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ≤ δ

/-- `LearnerLoopDFASimple` over strings drawn from `D`: the probes are `D`'s strings cut at a length
uniform over `k` to `L₁`, the suffixes their first `Ls` letters, and the coverage, error and
held-out strings are over their first `L` letters. `β` takes `1/|Σ|`'s place in `ε`, and `p₀`
takes `|Σ|^-k`'s in the conditions on the walk start. -/
def LearnerLoopDFADist : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (η₀ c γ δ pAP ξ β p₀ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  let Ls : ℕ := L₁ + Fintype.card σ + 3
  -- Probes are cut at the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- Each round's disagreement rate, low enough to catch a wrong edge out of any covered state.
  let ε : ℝ := c / (4 * (1 / β + 1) * N)
  -- The clustering's accuracy.
  let εcov : ℝ := min ε (ξ / (8 * N))
  -- Each round's share of `δ`.
  let δ' : ℝ := δ / (8 * (Q + 1))
  -- The clustering's indecision limit: a harvest puts more than `3ι/T` of its population on a
  -- state read undecided more than `T` of the time, `T` a fixed threshold, so each harvest has a
  -- new such state.
  let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
  let s : ℝ := 1 / 2 - η₀
  -- The clustering gate's rate scale.
  let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  -- The clustering's populations.
  let J : ℝ := Q + 2
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The strings a round reads undecided; the clustering's band must keep their wrong reads within
  -- level 5's race tail.
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1
  -- The clustering's suffix pool.
  let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP
  -- The clustering's prefix count.
  let RB : ℝ := 161000 * J * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  -- No string may hold more than this of a population or of the suffixes.
  let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
        (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
      (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
        (ι / 2 / (8 * J * (RB + 1)))))
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
  0 < β → β * A ≤ 1 →
  -- `D`'s strings are long enough for the suffixes,
  (∀ᵐ w ∂D, Ls ≤ w.toList.length) →
  -- extend each prefix from `k` to `L` letters by each letter at least `β` of the time,
  (∀ i, k ≤ i → i ≤ L → ∀ y a,
    β * D.real {w | prefixOf w i = y} ≤ D.real {w | prefixOf w (i + 1) = y * FreeMonoid.of a}) →
  -- and hold at most `p₀` on any first `k` letters.
  (∀ u, D.real {w | prefixOf w k = u} ≤ p₀) →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's precondition: some covered state re-roots the target with error at most `γ`.
  (∃ q ∈ coveredStatesOn (prefixLaw D L) M L c,
    1 - γ ≤ rerootAccuracyOn (prefixLaw D L) M L (coveredStatesOn (prefixLaw D L) M L c) q) →
  -- The walk start's prefixes are spread thinly enough for the clustering's collisions,
  p₀ ≤ cap * ε / 1536 →
  -- for each round's harvests to concentrate at every state's rate,
  1536 * Real.log (16 * (Q + 1) * Kc / δ') * p₀ ≤ ε →
  -- for the held-out strings to read only fresh strings,
  (Q + 1) * n * (6 * J * RB + B₃ + n) * p₀ ≤ δ / 4 →
  -- and for no two of a round's probes to share their walk start.
  16 * B₃ ^ 2 * p₀ ≤ δ' →
  -- Within `|Q| + 1` rounds the loop ends consistent, and its re-rooted DFA misclassifies at most
  -- `γ + ξ` of `D`'s first `L` letters, but for chance `δ`.
  ∃ settings : LoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
    (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
    ((loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).prod
        (Measure.pi fun _ : Fin n => prefixLaw D L)).real
      {q | let rounds := indecisiveLStarLoop settings O (windowLaw D k L₁) (prefixLaw D Ls)
            (persistDraws q.1)
        ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
            o.result.ending = .consistent ∧ (prefixLaw D L).real {x | ¬
              o.result.hypothesis.rerootedDFA o.read q.2 x
                (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ≤ δ

end OrthoDFA

end
