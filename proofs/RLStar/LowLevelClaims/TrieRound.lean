import RLStar.LowLevelClaims.Trie

/-!
# The trie round ends well but for a small chance

Reads are drawn as in `Level3RoundCorrect` (`ReadOracle`). A harvest's sampler
(`trieHarvestLaw`) walks a fresh probe past the round's final expanded prefixes, under its final
hypothesis.

`Level3TrieYields` (open): a harvest's sampler yields `κ y`, `κ` depending only on `|Σ|`, while
the cap binds with chance at most `δ / 8`. The harvest tests count across expansions and drop the
probes whose strings repeat, so their counts are not those of one law; a bound on the probes
they keep is the missing step.

`Level3TrieRoundCorrect`: but for chance `δ`, a round that does not run out of probes ends well.
- A consistent ending has at most `ε` of the probes disagreeing.
- A harvest's sampler yields `y₁ = κ y`.
- Its law puts at most `A·T + η/y₁` of itself on the states read undecided at most `T` of the
  time, with
      A = 8(L + 1)(|Q| + 1)/y₁ + 1/T₀,
      η = p_w log(8(|Q| + 1) K / δ),
      T₀ = δ / (8 (N + 1)).
  Here `p_w` bounds the probes' length-`k_w` prefix masses, `K` counts the trees, and `N` bounds
  the strings the round reads plus those `v m` with `|v| < k_w` and `m` a midfix.
- It is spread from `0`: no string has more than all of it.

Strings read at a prefix of at least `k_w` letters are light: their sum concentrates at scale
`p_w` for every expanded set at once. Shorter ones are few, and none of them, nor any string the
round reads, is undecided at a state read undecided at most `T₀` of the time, but for `δ / 8`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

/-- A harvest's sampler yields `κ y` but for chance `δ / 4`. -/
def Level3TrieYields : Prop :=
  ∃ κ : ℕ → ℝ, (∀ n, 0 < κ n) ∧
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (y : ℝ) (cap : ℕ) (pc : ℝ),
    (∀ u, P.D.real {x | prefixOf x cap = u} ≤ pc) →
    (C.probeBudget : ℝ) ^ 3 * (cap + 1) * pc ≤ C.δ / 8 →
    0 < C.δ → 0 < y → y < 1 / 4 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let r := indecisiveLStarLevel3Trie C y cap (R.read · ω) draws
          ∃ site, r.ending = .harvest site
            ∧ (trieHarvestLaw (R.read · ω) P.D r.expanded r.hypothesis site).real Set.univ
              < κ (Fintype.card α) * y} ∂μ
      ≤ ENNReal.ofReal (C.δ / 4)

/-- A trie round that does not run out of probes fails to end well with chance at most `δ`. -/
def Level3TrieRoundCorrect : Prop :=
  ∃ κ : ℕ → ℝ, (∀ n, 0 < κ n) ∧
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : ReadOracle μ M side U) (P : PrefixDistribution α)
    (C : Level3Settings) (y : ℝ) (cap kw : ℕ) (pc pw : ℝ),
    let δ := C.δ
    let ε := C.disagreementRate
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let y₁ := κ nα * y
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ
    let η := pw * Real.log (8 * (nQ + 1) * K / δ)
    -- A probe reads at most `3 + ⌈log₂ L⌉` sifts of at most `|Q| + 1` strings, a split at most
    -- `(|Q| + 2)|Σ|(|Q| + 2)`; the heavy strings are `v m` with `|v| < k_w`.
    let N : ℕ := C.probeBudget * ((3 + Nat.clog 2 P.L) * (nQ + 1))
      + nQ * ((nQ + 2) * nα * (nQ + 2)) + (∑ j ∈ Finset.range kw, nα ^ j) * (nQ + 1)
    let T₀ := δ / (8 * (N + 1))
    let A := 8 * (P.L + 1) * (nQ + 1) / y₁ + 1 / T₀
    (∀ u, P.D.real {x | prefixOf x kw = u} ≤ pw) →
    (∀ u, P.D.real {x | prefixOf x cap = u} ≤ pc) →
    (C.probeBudget : ℝ) ^ 3 * (cap + 1) * pc ≤ δ / 8 →
    0 < δ → 0 < ε → ε < 1 → 0 < y → y < 1 / 4 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let r := indecisiveLStarLevel3Trie C y cap (R.read · ω) draws
          r.ending ≠ .failed ∧
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := η / y₁,
              probes := P.D,
              disagreeRate := ε,
              disagree := trieDisagreeSet (R.read · ω) r.expanded r.hypothesis,
              walkStart := 0,
              harvestLaw := trieHarvestLaw (R.read · ω) P.D r.expanded r.hypothesis,
              yield := y₁,
              ending := r.ending }} ∂μ
      ≤ ENNReal.ofReal δ

end OrthoDFA

end
