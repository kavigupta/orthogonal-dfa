import OrthoDFA.LoopDFA

/-!
# The learner's DFA under the relative rule, by summed occupancy

`LearnerLoopDFASummed` admits the target by `LearnerLoopDFARelative`'s rule: some state `q` covered
at `c` re-roots it with accuracy at least `1 − γ` without leaving `{q} ∪ dominatedStates M L R q`.
It is proved directly, not through `LearnerLoopDFA`. A target state is wrong when an edge out of
its true leaf is wrong or unlearned, or it is read on its wrong side. The wrong states together
occupy the window less than `m = min c (3ξ/(8R))`, so the start is not wrong, and the runs from it
through a wrong dominated state cost at most `C m ≤ 3ξ/4`, `C = R(L+1)/(L+1−⌈L/2⌉) ≤ 2R`. No
state but the start need be covered. The pick of the start leaf gets the other `ξ/4`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

/-- The loop's re-rooted DFA under the relative rule misclassifies at most `γ + ξ` of the strings
of length `L`, but for chance `δ`; every premise an explicit inequality. -/
def LearnerLoopDFASummedSimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c R γ δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- The start's coverage, capped where the wrong states it dominates cost `3ξ/4`.
  let m : ℝ := min c (3 * ξ / (8 * R))
  -- Each round's disagreement rate, low enough that the wrong states together occupy the window
  -- less than `m`.
  let ε : ℝ := m / (3 * (A + Q + 1) * N)
  -- The clustering's accuracy.
  let εcov : ℝ := min ε (ξ / (32 * N))
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
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
  -- The clustering's prefix count.
  let RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
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
  let n : ℕ := ⌈512 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < R → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ →
    2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's relative precondition: some state re-roots the target with error at most `γ`
  -- through the states it dominates.
  (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (insert q (dominatedStates M L R q)) q) →
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

/-- `LearnerLoopDFASummedSimple` once the Python's length `L` is at least `L₀`. -/
def LearnerLoopDFASummed : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ c R γ δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- The start's coverage, capped where the wrong states it dominates cost `3ξ/4`.
  let m : ℝ := min c (3 * ξ / (8 * R))
  -- Each round's disagreement rate, low enough that the wrong states together occupy the window
  -- less than `m`.
  let ε : ℝ := m / (3 * (A + Q + 1) * N)
  -- Each round's share of `δ`.
  let δ' : ℝ := δ / (8 * (Q + 1))
  let s : ℝ := 1 / 2 - η₀
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The held-out strings the start leaf is picked on.
  let n : ℕ := ⌈512 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The rest bound `LearnerLoopDFASummedSimple`'s premises with the length `ℓ = L₁ + 1` absorbed:
  -- `1/cap ≤ ℓ⁵ E₁`, `1/ε ≤ ℓ aε`, the prefix count `≤ ℓ³ R₀`, the budget `≤ ℓ K₁`, `W ≤ ℓ² W₀`.
  let aε : ℝ := 3 * (A + Q + 1) / m
  let Ce : ℝ := aε + 32 / ξ
  let Θ : ℝ := Ce / (2 * s ^ 2) + 8 * Ce + 4608 * Q * (Q + 1) * aε
  let K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
  let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  let W₀ : ℝ := Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
  let P₀ : ℝ := (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
  let R₀ : ℝ := 161000 * (Q + 2) ^ 2 * Θ ^ 2
    * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
  let E₁ : ℝ := (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
    + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  let E : ℝ := 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε
  let E₂ : ℝ := 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ
  let E₃ : ℝ := 16 * K₁ ^ 2 / δ'
  -- `log ℓ` is bounded by its tangent at `T`.
  let T : ℝ := 2 * (Real.log (E + E₂ + E₃) + 12) / Real.log A + Q + 4
  let L₀ : ℝ := 2 * (Real.log (E + E₂ + E₃) + 6 * Real.log T) / Real.log A
    + 144 * (Real.log T - 2) / (Real.log A * (T * Real.log A - 12))
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → 0 < R → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ →
    2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's relative precondition: some state re-roots the target with error at most `γ`
  -- through the states it dominates.
  (∃ q ∈ coveredStates M L c, 1 - γ ≤ rerootAccuracy M L (insert q (dominatedStates M L R q)) q) →
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

end OrthoDFA

end
