import RLStar.Code.Clustering
import RLStar.LowLevelClaims.FamilyRead

/-! The guarantee of the clustering model in `Code/Clustering.lean`: `ClusteringGuarantee`. -/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-! ### Vote -/

/-- What `voteCount` averages to over the oracle's noise. -/
noncomputable def meanVote (O : Oracle μ S) (F : Finset S) (p : S) : ℝ :=
  ∑ v ∈ F, μ.real {ω | O.mq (p * v) ω = 1}

/-! ### The cut -/

/-- Where the family decides `p`, it decides the way the noiseless label does.  Indecisive
prefixes hold vacuously.  This is the family's cut, not accept preservation by each member. -/
def cutCorrect (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) (ω : Ω) : Prop :=
  (hi < voteCount O.mq F p ω → O.label p = 1) ∧ (voteCount O.mq F p ω ≤ lo → O.label p = 0)

/-! ## What the input distributions must satisfy -/

/-- Two prefixes never extend, by drawn suffixes or by `ε`, to the same query string.

`sample_more_suffixes` draws every suffix at the sampler's length `L`, so `p * v = p' * v'`
with both suffixes drawn forces `p = p'` on length alone.  Against the seed `ε` it would need
`p` to be longer than `p'` by `L`, which prefixes whose lengths lie within a window narrower than
`L` never are.  Stated this way because `Monoid` on its own does not know that strings factor. -/
def Flat (Pre Suf : Set S) : Prop :=
  ∀ p ∈ Pre, ∀ p' ∈ Pre, ∀ v ∈ insert 1 Suf, ∀ v' ∈ insert 1 Suf, p * v = p' * v' → p = p'

/-- The rates sit at least `½ − η₀` either side of `c`: the language reads accept at least that
far above it, the rest at most that far below. -/
def Oracle.Straddles (O : Oracle μ S) (c η₀ : ℝ) : Prop :=
  max O.ηIn 0 ≤ η₀ + 1 / 2 - c ∧ max O.ηOut 0 ≤ η₀ - 1 / 2 + c

/-! ## The theorem -/

/-- The rate the family's cut budget resolves:

    min(εcov·a², indecisionLimit),   a = 1 − 2η₀

The gate charges each wrongly cut prefix at most `7/(3a²)` of a read, so the share a family may
cut wrongly is held at a multiple of `εcov·a²`. -/
noncomputable def rateScale (η₀ indecisionLimit εcov : ℝ) : ℝ :=
  min (εcov * (1 - 2 * η₀) ^ 2) indecisionLimit

/-- The E-L\* clustering algorithm is correct at a polynomial cost.  With probability
`≥ 1 − δ − α` the loop stops at one of `states`, and whenever it stops, the family it returns
there cuts `≥ 1 − εcov − εs` of the uniform pool the way the noiseless oracle does, reads
no tame population backwards (cuts at least `1/2 − εs` of each
that way), and leaves at most `2·indecisionLimit` of each tame population undecided; no state
draws more prefixes than the first count below, nor asks for a family or a suffix pool larger
than the sizes below.  Some state draws no more than the second count, what validity costs
and what makes a state's samplers rarely reject many of its draws, so a stop is covered from
there; the first count is what the loop may need before a round passes.  Every state's band is
wide enough that a vote over a family no larger than its own, whose mean lies above the band,
lands at or below `lo`, or one whose mean lies at or below `lo` lands above `hi`, at most
`crossLimit` of the time; and whatever its two rates, such a vote reads its rarer decided side at
most `κ` times as often as undecided, for any `κ` with `crossLimit ≤ κ⁴`.

The algorithm is told only `η₀`, whose signal `½ − η₀` is at most a quarter of the gap
`1 − ηIn − ηOut`; it reads where the two rates sit off the data.  `uni` is the uniform pool,
the one population every prefix of the learner's distribution is drawn from, and it holds at
least `2·εcov` of the language and of its complement.  The second count's log reads the
family's size: the band is tried at each of its centres.  `εs` is the slack
the samplers' redraws and the hash's split may cost the cut.  `pAP` lower-bounds
the share of suffixes that preserve membership for every prefix.  `α` is what the gate spends:
the chance it passes a family wrong on more than its limit, over every state.  The states depend
on none of the distributions.  A tame population need only have no string heavy against the
prefixes a state draws at the signal times `min(εcov·signal, indecisionLimit)`, and none against
`|J| + 3` times them at `min(εs, indecisionLimit)`: a draw then lands on what the state already
holds rarely enough that its rejections are a binomial tail; and no string heavy against
`min(εs, indecisionLimit)²/log(|J|·k/δ)`, so the hash splits it evenly at every centre.  The suffix
measure needs no string heavy against the suffixes; the other populations are arbitrary.  A
population is read only while `keptAt` keeps it.  `H j` are the strings population `j` harvested;
the first count grows with how many there are. -/
def ClusteringGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations tame : Finset J) (uni : J) (H : J → Finset S)
    (Pre Suf : Set S) (η₀ indecisionLimit εcov εs α δ pAP crossLimit εg : ℝ),
  4 * (1 / 2 - η₀) ≤ O.gap →
  η₀ < 1 / 2 →
  uni ∈ tame →
  tame ⊆ populations →
  Flat Pre Suf →
  (∀ j, ∀ p ∈ H j, p ∈ Pre) →
  0 < pAP →
  0 < indecisionLimit →
  indecisionLimit ≤ 1 / 2 →
  0 < α →
  α < 1 / 2 →
  0 < εcov →
  εcov ≤ 1 / 2 →
  0 < εs →
  εs ≤ εcov →
  εcov ≤ εg →
  εg ≤ 1 / 2 →
  0 < δ →
  δ ≤ 1 →
  0 < crossLimit →
  crossLimit ≤ 1 →
  ∃ states : Finset State,
    (∀ B ∈ states, (B.npref : ℝ) ≤
      2 ^ 22
      * (populations.card : ℝ)
      * Real.log (64 * ((B.nsuff : ℝ) + 1) / δ)
      / ((1 / 2 - η₀) ^ 4 * min εcov indecisionLimit ^ 2)
      + 3 * 2 ^ 25
      * Real.log (
        ((populations.card : ℝ) + 2) * B.nsuff
        / (δ * α * pAP * rateScale η₀ indecisionLimit εcov)
      )
      / rateScale η₀ indecisionLimit εcov ^ 2
      + 6 * Real.log (256 * (populations.card : ℝ) / δ) / εs
      + 2 ^ 11 * ((allHarvest H).card : ℝ) / rateScale η₀ indecisionLimit εcov
    ) ∧
    (∃ B ∈ states, (B.npref : ℝ) ≤ max
      (2 ^ 13 * Real.log (32 * (populations.card : ℝ) * B.k / δ)
        / ((1 / 2 - η₀) * min (εcov * (1 / 2 - η₀)) indecisionLimit))
      (12 * Real.log (32 * (populations.card : ℝ) / δ) / εs)
    ) ∧
    (∀ B ∈ states,
      (B.k : ℝ) ≤ 25 * (max (4 * Real.log (20 / rateScale η₀ indecisionLimit εcov))
          (Real.log (1 / crossLimit)) + 12) / (8 * (1 / 2 - η₀) ^ 2) + 5 / (1 / 2 - η₀) + 7
      ∧ (B.nsuff : ℝ) ≤ 2 * (25 * (max (4 * Real.log (20 / rateScale η₀ indecisionLimit εcov))
          (Real.log (1 / crossLimit)) + 12) / (8 * (1 / 2 - η₀) ^ 2) + 5 / (1 / 2 - η₀) + 7) / pAP
        + 60 * Real.log (3 / δ) / pAP) ∧
    (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit)) ∧
    (∀ B ∈ states, ∀ N, N + 1 ≤ B.k → ∀ κ p r : ℝ, 0 ≤ κ → crossLimit ≤ κ ^ 4 →
      0 ≤ p → p ≤ 1 → 0 ≤ r → r ≤ 1 → RarerBounded N B.lo (B.hi + 1) p r κ) ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      2 * εcov ≤ (D uni).real {p | p ∈ O.L} →
      2 * εcov ≤ (D uni).real {p | p ∉ O.L} →
      Dsf Sufᶜ = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 19 * (B.npref : ℝ) * (D j).real {t}
          ≤ (1 / 2 - η₀) * min (εcov * (1 / 2 - η₀)) indecisionLimit) →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 14 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
          ≤ min εs indecisionLimit) →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        512 * Real.log (64 * (populations.card : ℝ) * B.k * (3 + 2 * (Nat.log 2 B.npref : ℝ))
          / δ) * (D j).real {t} ≤ min εs indecisionLimit ^ 2) →
      (∀ B ∈ states, ∀ t : S, 2 ^ 16 * (B.nsuff : ℝ) * Dsf.real {t} ≤ pAP) →
      1 - δ - α ≤ (runMeasure μ D Dsf).real
        {x | (∃ B : {B : State // B ∈ states},
              streams H uni x ∈ ret O.mq populations uni η₀ indecisionLimit εg α B.val)
          ∧ ∀ B : {B : State // B ∈ states},
            streams H uni x ∈ ret O.mq populations uni η₀ indecisionLimit εg α B.val →
            1 - εg - εs
              ≤ (D uni).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
            ∧ ∀ j ∈ tame, 1 / 2 - εs
              ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
              ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations (streams H uni x) B.val) p (oracleNoise x.1)}
                ≤ 2 * indecisionLimit}

end OrthoDFA
