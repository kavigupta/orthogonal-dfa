import OrthoDFA.Level3Check
import OrthoDFA.Level4Claim

/-!
# `Level4CheckRoundCorrect`: the checked round ends well but for a small chance

The checked round (`indecisiveLStarCheck`) runs with the reads of `Level4RoundCorrect`, each draw
a probe from `P` and a block of check strings from `Dc`. It ends well as level 3's round does,
and fails to only where level 4's round would, where some step reads a string wrong, where the
check strings read undecided more often than a share `q` of the time, or where a check silent at
that share fails without a fix:

    P(¬ ends well) ≤ δ + Σ_{j < S} b(q)^n / a_j + |heavy| u + exp(−η_c / p_h),

`n` the check's size, `a_j` the `j`-th check's level and `b(q)` the test's silent base
(`CheckSettings.silentBase`). Each check string's prefix followed by at most `|Q|` letters reads
undecided with chance at most `u`; those with prefixes shorter than `h` are the `|heavy|` strings
every check shares, and past them a check string's reads share its first `h` letters, no
length-`h` prefix holding more than `p_h` of the check strings.
-/

namespace OrthoDFA

namespace Random

/-- The check's test, silent after `n` strings at a failing rate `q`, has chance at most this to
the `n`th power over its level. -/
noncomputable def CheckSettings.silentBase (K : CheckSettings) (q : ℝ) : ℝ :=
  q * (K.rate / (K.rate / 2)) + (1 - q) * ((1 - K.rate) / (1 - K.rate / 2))

end Random

open MeasureTheory Random Random.Eps

/-- The checked round fails to end well with chance at most level 4's, the check's silent tails
and the chance its strings read undecided too often. -/
def Level4CheckRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ]
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw)
    (P : PrefixDistribution α) (Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure Dc] (Lc h : ℕ) (C : Level3Settings)
    (K : CheckSettings) (p₀ u ph ηc : ℝ),
    let y := C.harvestRate
    let ε := C.disagreementRate
    let δ := C.δ
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    -- Bounds the hypotheses, as at level 3.
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    -- Counts the trees level 3 can grow, as at level 3.
    let Kt : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let A : ℝ := (nQ + 1) * (6 / y + 16 * (P.L + 1))
    let η := p₀ * Real.log (16 * (nQ + 1) * Kt / δ)
    -- The strings of at most `|Q|` letters bound the midfixes.
    let Ms : ℝ := (nQ + 1) * nα ^ nQ
    let θ₀ : ℝ := (Lc + 1) * Ms * u
    -- The share of check strings reading undecided the check is held to.
    let q : ℝ := 2 * θ₀ + ηc
    -- Level 4's premises.
    (∀ v, P.D.real {x | prefixOf x C.walkStart = v} ≤ p₀) → 0 < δ → 0 < y →
    y ≤ ε / 48 → ε < 1 →
    36 * S / ε * Real.log (16 * S * (S + 1) / δ) + S ≤ C.probeBudget →
    -- The check strings, at most `L_c` long, their prefixes followed by at most `|Q|` letters
    -- reading undecided with chance at most `u`.
    (∀ᵐ x ∂Dc, x.toList.length ≤ Lc) →
    (∀ z : FreeMonoid α, z.toList.length ≤ Lc + nQ → U (M.eval z.toList) ≤ u) → 0 ≤ u →
    (∀ v, Dc.real {x | prefixOf x h = v} ≤ ph) → 0 < ph → h ≤ Lc → 0 ≤ ηc →
    8 * θ₀ ≤ 1 → q ≤ 1 → 0 < K.rate → K.rate < 1 → 0 < K.δ →
    -- A probe reads at most `(L + 1)|Q|` strings, a check `n (L_c + 1)|Q|`, and the splits
    -- re-sift at most `(n + 1)(|Q| + 1)²|Σ|`, so the round reads some string wrong with chance
    -- at most `δ/8`.
    εw * C.probeBudget * ((P.L + 1) * nQ + K.size * ((Lc + 1) * nQ)
      + (K.size + 1) * ((nQ + 1) ^ 2 * nα)) ≤ δ / 8 →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget =>
        P.D.prod (Measure.pi fun _ : Fin K.size => Dc))
        {draws | let result := indecisiveLStarCheck C K (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := η / y,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (δ + ∑ j ∈ Finset.range S, K.silentBase q ^ K.size / K.level j
        + (h + 1) * nα ^ h * Ms * u + Real.exp (-(ηc / ph)))

end OrthoDFA
