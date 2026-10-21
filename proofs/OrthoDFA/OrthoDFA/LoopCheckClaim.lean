import OrthoDFA.LoopCheck

/-!
# The learner's checked loop stops at its first round with an accurate DFA

`LearnerLoopCheckSimple`: where the target re-rooted at some state `q*` classifies every string of
each length from `⌈L/2⌉` to `2L` with its run in the covered states, the checked loop, its probes
half of length `L` and half uniform over the lengths `⌈L/2⌉` to `L` (`probeMix`), and walked
from `⌈L/2⌉`, started from any inherited state
whose populations hold only strings `⌈L/2⌉` to `L` long and which read at most `T₀` prefixes and
`T₀` suffixes, ends at its first round, consistent, and its DFA started where `ε` sifts
misclassifies at most

    ε + 2 ((513/512) εcov + 2 ι + T₀ |Σ|^(−k)) + N_K |Σ|^(−a)

of the strings of length `L`, but for chance `δ`: the check's rate, the family's miscut and
indecision on the uniform pool, the pool's mass the earlier rounds read, and the check strings'
mass on the clustering's prefixes. `ε` is the Python's `1 − acc_threshold`, the rate both level
3's success test and the start check test, and `εcov` the clustering's accuracy. The strings of
length `L` are half the pool, whence the factor `2`.

The family's indecision limit `ι` is small enough that every state reads undecided so rarely that
no harvest can put its law where `#478`'s formulation asks; the theorem's first round therefore
never harvests, and it exercises neither harvests nor the progress they make.

`LearnerLoopCheck` states the same once `L ≥ L₀`.

Known modelling gap. The Python admits a target whose covered start re-roots it with error up to
`γ = 1 − 0.99`; the model asks `γ = 0`, on every length from `⌈L/2⌉` to `2L`. The Python tolerates
the error as a kind of noise the proof does not model.

Known modelling gap. The Python draws its probes at length `L`; here half of them are uniform over
the window of lengths `⌈L/2⌉` to `L`, so that the uniform pool holds every covered state.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

/-- The checked loop stops at its first round, consistent, with its DFA from where `ε` sifts
misclassifying at most its check's rate and its reads' errors on the strings of length `L`, but for chance `δ`; every premise an
explicit inequality. -/
def LearnerLoopCheckSimple : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L T₀ : ℕ) (η₀ c δ pAP εcov ε : ℝ),
  let Q : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  -- Probes are walked from half the Python's length; half of them are uniform over the `N`
  -- lengths `k` to `L`.
  let k : ℕ := (L + 1) / 2
  let N : ℝ := L + 1 - k
  let ℓ : ℝ := L + 1
  -- The check strings avoid the first `a` letters of the clustering's prefixes.
  let a : ℕ := k - Fintype.card σ - 1
  -- The clustering's suffixes' length.
  let Ls : ℕ := L + 1
  let s : ℝ := 1 / 2 - η₀
  -- Bounds the hypotheses a round goes through.
  let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
  -- Level 3's and the clustering's shares of `δ`.
  let δ₃ : ℝ := δ / 4
  let δc : ℝ := δ / 16
  -- Level 3's probe budget and the check's size.
  let T : ℕ := ⌈36 * S / ε * Real.log (16 * S * (S + 1) / δ₃) + S⌉₊
  let n : ℕ := ⌈16 / (3 * ε) * Real.log (86 * S * (S + 1) * (S + 2) / δ ^ 2)⌉₊
  -- The strings of at most `|Q|` letters, and the length of the check strings' prefixes every
  -- check shares.
  let Ms : ℝ := (Q + 1) * A ^ Fintype.card σ
  let h : ℕ := Nat.clog (Fintype.card α) ⌈16 * Real.log (64 / δ) / ε⌉₊
  -- The clustering's indecision limit: every state reads undecided at most `14 ι N / c` of the
  -- time, too rarely for a harvest or for the check strings to waste a check.
  let ι : ℝ := min (c * δ / (896 * (h + 1) * A ^ h * Ms * ℓ))
    (min (c / (28 * (Q + 1) * (288 / ε + 16 * ℓ) * ℓ)) (c * ε / (896 * Ms * ℓ ^ 2)))
  -- The clustering gate's rate scale.
  let τ : ℝ := min (εcov * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
  -- The clustering's populations.
  let J : ℝ := Q + 2
  -- The strings a round can read wrong; the clustering's band must keep them right.
  let W : ℝ := T * ((L + 1) * Q + n * ((L + 1) * Q) + (n + 1) * ((Q + 1) ^ 2 * A))
  -- The clustering's suffix pool and prefix count.
  let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ₃)) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((J + 2) / (δc / 2)) / pAP ^ 2
  let RB : ℝ := 64000 * J ^ 2 * Real.log ((J + 2) * PB / (δc / 2 * δc * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2)
  -- Counts the trees a round can grow, with an edge map into each one's leaves.
  let Kt : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
  -- The first letters the probes and the check strings avoid.
  let NK : ℝ := 6 * J * RB
  -- The oracle's language is the target's.
  (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
  O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < εcov → εcov ≤ 1 / 2 →
  0 < ε → ε ≤ 1 / 2 → 2 ≤ Fintype.card α →
  -- At least `pAP` of the suffixes keep every string's membership.
  pAP ≤ (uniformStrings Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
  -- The Python's precondition, at `γ = 0`.
  (∃ q, CoveredAccurate M L c q) →
  -- The Python's length is long enough for the walk, the check and the re-rooting.
  Fintype.card σ + 2 ≤ k → Fintype.card σ ^ 2 ≤ L → h ≤ L →
  -- No string is heavy against the clustering's prefixes,
  2 ^ 22 * (Q + 3) ^ 2 * RB / A ^ k ≤ δc * τ ^ 2 →
  2 ^ 9 * (Q + 3) * RB / A ^ k ≤ δc * ι ^ 2 →
  -- nor a suffix against its pool;
  2 ^ 15 * (Q + 3) * PB / A ^ Ls ≤ δc * pAP →
  -- the probes' first `k` letters are spread thinly enough that a harvest cannot concentrate,
  384 * Real.log (16 * (Q + 1) * Kt / δ₃) / (ε * A ^ k) ≤ 1 →
  -- and the clustering's prefixes hold few probes and check strings;
  NK / A ^ k ≤ 1 / 2 → NK / A ^ a ≤ 1 / 2 →
  -- what the earlier rounds read holds few suffixes and probes.
  4 * T₀ / A ^ Ls ≤ pAP → T₀ / A ^ k ≤ min (c / (4 * N)) (1 / 2) →
  ∃ settings : CheckLoopSettings (Fintype.card σ + 1),
    settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε
      ∧ settings.check.rate = ε ∧ settings.level3.probeBudget = T ∧ settings.check.size = n ∧
    ∀ h₀ : LoopHist α,
      (∀ ν ∈ h₀.pops, IsProbabilityMeasure ν
        ∧ ν {q | ¬ (k ≤ q.toList.length ∧ q.toList.length ≤ L)} = 0) →
      h₀.prefs.card ≤ T₀ → h₀.suffs.card ≤ T₀ →
      (checkLoopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget
          settings.check.size).real
        {p | ¬ ∃ o, indecisiveLStarCheckLoop settings O (probeMix k L) (uniformStrings Ls)
              (uniformStrings L) h₀ p = [o]
            ∧ o.result.ending = .consistent
            ∧ ∃ s, o.result.hypothesis.tree.sift o.read 1 = .inl s
              ∧ (uniformStrings L).real
                  {x | ¬ o.result.hypothesis.Classifies s x (M.eval x.toList ∈ M.accept)}
                ≤ ε + 2 * (513 / 512 * εcov + 2 * ι + T₀ / A ^ k) + NK / A ^ a}
        ≤ δ

end OrthoDFA

end
