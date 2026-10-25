import OrthoDFA.LoopDFA

/-!
# The learner's DFA under typical-start replication, probes to twice the length

A target is admitted at `(m, γ)` when some state `q` the window visits `m` of the time
(`windowVisitRate`) re-roots it with error at most `γ` (`rerootError`); no other state need be
covered. The probes reach `L₁ = 2L + |Q| + 3`, so the window holds every position from `⌈L/2⌉` to
`2L`. A target state is wrong when an edge out of its true leaf is wrong or unlearned, or it is
read on its wrong side; the wrong states together occupy the lengths `⌈L/2⌉` to `2L` at most

    X = N(2((|Σ|+1)εB + |Q|εm) + 2εr + εu),

`εB`, `εm` the disagreeing and misled slack and `εr`, `εu` the wrong and undecided reads. The first
visit to `q` from `⌈L/2⌉` on is followed by `L` letters uniform and independent of it, so the run
re-rooted at `q` meets a wrong state on at most `X/m` of the strings. At
`ε = ξm/(5(|Σ|+|Q|+3)N)` that and the pick's misreads cost at most `ξ/2`, the pick's scores the
other `ξ/2`.

Known modelling gap. The probes reach `2L + |Q| + 3`, past `LearnerLoopDFA`'s `L + |Q| + 3`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The share of the strings of length `L` that visit `q` at some position from `⌈L/2⌉` to `L`. -/
def windowVisitRate {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (q : σ) : ℝ :=
  (uniformStrings L).real
    {x : FreeMonoid α | ∃ j, (L + 1) / 2 ≤ j ∧ j ≤ L ∧ M.eval (prefixOf x j).toList = q}

/-- The share of the strings of length `L` that `M` re-rooted at `q` labels otherwise than `M`. -/
def rerootError {σ : Type*} (M : _root_.DFA α σ) (L : ℕ) (q : σ) : ℝ :=
  (uniformStrings L).real
    {x : FreeMonoid α | ¬ (M.evalFrom q x.toList ∈ M.accept ↔ M.eval x.toList ∈ M.accept)}

/-- The loop, its probes reaching `2L + |Q| + 3`, outputs a re-rooted DFA that misclassifies at
most `γ + ξ` of the strings of length `L`, but for chance `δ`, where some state visited `m` of
the time re-roots the target with error at most `γ`; every premise an explicit inequality. -/
def LearnerLoopDFATSRBSimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ m γ δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := 2 * L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- Each round's disagreement rate, low enough that the wrong states cost the start's runs `ξ/2`.
  let ε : ℝ := ξ * m / (5 * (A + Q + 3) * N)
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
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < m → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → ξ ≤ 1 →
  2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- Typical-start replication: some state the window visits `m` of the time re-roots the target
  -- with error at most `γ`.
  (∃ q, m ≤ windowVisitRate M L q ∧ rerootError M L q ≤ γ) →
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

/-- `LearnerLoopDFATSRBSimple` once the Python's length `L` is at least `L₀`. -/
def LearnerLoopDFATSRB : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L : ℕ) (η₀ m γ δ pAP ξ : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length.
  let k : ℕ := (L + 1) / 2
  let L₁ : ℕ := 2 * L + Fintype.card σ + 3
  -- Probes are uniform over the `N` lengths `k` to `L₁`.
  let N : ℝ := L₁ + 1 - k
  -- Each round's disagreement rate, low enough that the wrong states cost the start's runs `ξ/2`.
  let ε : ℝ := ξ * m / (5 * (A + Q + 3) * N)
  -- Each round's share of `δ`.
  let δ' : ℝ := δ / (8 * (Q + 1))
  let s : ℝ := 1 / 2 - η₀
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Each round's level-3 probe budget.
  let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
  -- The held-out strings the start leaf is picked on.
  let n : ℕ := ⌈32 * Real.log (16 * (Q + 1) * (Q + 2) / δ) / ξ ^ 2⌉₊
  -- The rest bound `LearnerLoopDFATSRBSimple`'s premises with the length `ℓ = L₁ + 1` absorbed:
  -- `1/cap ≤ ℓ⁵ E₁`, `1/ε ≤ ℓ aε`, the prefix count `≤ ℓ³ R₀`, the budget `≤ ℓ K₁`, `W ≤ ℓ² W₀`.
  let aε : ℝ := 5 * (A + Q + 3) / (ξ * m)
  let Ce : ℝ := aε + 8 / ξ
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
  let T : ℝ := 4 * (Real.log (E + E₂ + E₃) + 12) / Real.log A + Q + 4
  let L₀ : ℝ := (2 * T * (Real.log (E + E₂ + E₃) + 6 * Real.log T - 6) + 12 * (Q + 4))
    / (T * Real.log A - 24)
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < m → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ → ξ ≤ 1 →
  2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings (L₁ + Fintype.card σ + 3)).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- Typical-start replication: some state the window visits `m` of the time re-roots the target
  -- with error at most `γ`.
  (∃ q, m ≤ windowVisitRate M L q ∧ rerootError M L q ≤ γ) →
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
