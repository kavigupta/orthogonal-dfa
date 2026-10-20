import OrthoDFA.LearnerLoop

/-!
# The learner's rate test certifies its DFA

The Python argues its DFA's accuracy from its last round's success test: the hypothesis, started
where `ε` sifts, disagrees with the family's reads on at most `1 − acc_threshold` of the probes,
which are drawn at its length `L` and walked from `0`. `LearnerLoopSoundSimple` states that
argument of the loop: its probes uniform over the strings of length `L` and walked from `0`, every
round that ends consistent, at rate `ε`, has its DFA started where `ε` sifts misjudge at most

    ε + ε/96 + (513/512) εcov + 2ι

of the strings of length `L`, but for chance `δ`, whatever level 3's yield and budget. `εcov` and
`ι` are the clustering's miscut and indecision limits on the uniform pool, which it holds at the
run's own noise; nothing is asked of the populations the harvests hand on, nor of the target
beyond its oracle and its suffixes. That the loop reaches a consistent round is not claimed here.

`LearnerLoopSound` states the same once `L ≥ L₀`.

Known modelling gap. The Python's success test counts the searches and the probes stuck at an
unlearned edge; level 3 also counts the probes stopping undecided at their start or end sift.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

/-- Every round of the loop walked from `0` that ends consistent has its DFA from where `ε` sifts
misjudge at most `ε + ε/96 + (513/512) εcov + 2ι` of the strings of length `L`, but for chance
`δ`; every premise an explicit inequality. -/
def LearnerLoopSoundSimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L R : ℕ) (η₀ ε εcov ι δ pAP : ℝ),
  let A : ℝ := Fintype.card α
  let Ls : ℕ := L + Fintype.card σ + 3
  -- The probes avoid the clustering's prefixes' first `a` letters.
  let a : ℕ := (L + 1) / 2
  -- The populations: the uniform pool and one per round.
  let J : ℝ := R + 1
  -- Each round's share of `δ` for its clustering's draws, and for its gate.
  let δc : ℝ := δ / (4 * R)
  let s : ℝ := 1 / 2 - η₀
  let τ : ℝ := min (εcov * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  -- The clustering's suffix pool and prefix count.
  let PB : ℝ := (50 * Real.log (20 / τ) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((J + 2) / (δc / 2)) / pAP ^ 2
  let RB : ℝ := 64000 * J ^ 2 * Real.log ((J + 2) * PB / (δc / 2 * δc * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  -- The prefixes a round reads.
  let NP : ℝ := 6 * J * RB
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < ε → ε ≤ 1 / 2 → 0 < εcov → εcov ≤ 1 / 2 → 0 < ι → ι ≤ 1 / 2 →
  0 < δ → δ ≤ 1 → 0 < pAP → 1 ≤ R →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- No string of length `L` is heavy against the clustering's prefixes,
  2 ^ 22 * (J + 1) ^ 2 * RB / A ^ L ≤ δc * τ ^ 2 →
  -- nor a suffix against its pool,
  2 ^ 15 * (J + 1) * PB / A ^ Ls ≤ δc * pAP →
  2 * R * (PB + 1) / A ^ Ls ≤ pAP →
  -- and the strings the rounds' clusterings read, and those whose first `a` letters they read,
  -- hold at most `ε/96` of the probes.
  96 * NP * (R + 1) / A ^ a ≤ ε →
  ∀ (y : ℝ) (budget : ℕ), ∃ settings : LoopSettings R,
    settings.level3 = ⟨0, 4 * δ / R, y, ε, budget⟩ ∧ settings.avoidDepth = a ∧
    settings.indecisionLimit = ι ∧ settings.εcov = εcov ∧
    (loopMeasure μ α R settings.level3.probeBudget).real
      {p | ∃ o ∈ indecisiveLStarLoop settings O (uniformStrings L) (uniformStrings Ls)
          (persistDraws p),
        o.result.ending = .consistent
          ∧ ¬ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
            ∧ (uniformStrings L).real
                {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
              ≤ ε + ε / 96 + 513 / 512 * εcov + 2 * ι}
      ≤ δ

/-- `LearnerLoopSoundSimple` once the Python's length `L` is at least `L₀`. -/
def LearnerLoopSound : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L R : ℕ) (η₀ ε εcov ι δ pAP : ℝ),
  let A : ℝ := Fintype.card α
  let Ls : ℕ := L + Fintype.card σ + 3
  let a : ℕ := (L + 1) / 2
  let J : ℝ := R + 1
  let δc : ℝ := δ / (4 * R)
  let s : ℝ := 1 / 2 - η₀
  let τ : ℝ := min (εcov * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  let PB : ℝ := (50 * Real.log (20 / τ) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((J + 2) / (δc / 2)) / pAP ^ 2
  let RB : ℝ := 64000 * J ^ 2 * Real.log ((J + 2) * PB / (δc / 2 * δc * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  let NP : ℝ := 6 * J * RB
  -- What `|Σ|^a` must clear: each of `LearnerLoopSoundSimple`'s inequalities on lengths.
  let E : ℝ := max 1 (max (max (2 ^ 22 * (J + 1) ^ 2 * RB / (δc * τ ^ 2))
      (2 ^ 15 * (J + 1) * PB / (δc * pAP)))
    (max (2 * R * (PB + 1) / pAP) (96 * NP * (R + 1) / ε)))
  let L₀ : ℝ := 2 * Real.log E / Real.log A
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < ε → ε ≤ 1 / 2 → 0 < εcov → εcov ≤ 1 / 2 → 0 < ι → ι ≤ 1 / 2 →
  0 < δ → δ ≤ 1 → 0 < pAP → 1 ≤ R → 2 ≤ Fintype.card α →
  pAP ≤ (uniformStrings Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's length is long enough.
  L₀ ≤ L →
  ∀ (y : ℝ) (budget : ℕ), ∃ settings : LoopSettings R,
    settings.level3 = ⟨0, 4 * δ / R, y, ε, budget⟩ ∧ settings.avoidDepth = a ∧
    settings.indecisionLimit = ι ∧ settings.εcov = εcov ∧
    (loopMeasure μ α R settings.level3.probeBudget).real
      {p | ∃ o ∈ indecisiveLStarLoop settings O (uniformStrings L) (uniformStrings Ls)
          (persistDraws p),
        o.result.ending = .consistent
          ∧ ¬ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
            ∧ (uniformStrings L).real
                {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
              ≤ ε + ε / 96 + 513 / 512 * εcov + 2 * ι}
      ≤ δ

end OrthoDFA

end
