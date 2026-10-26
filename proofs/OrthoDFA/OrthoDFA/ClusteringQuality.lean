import OrthoDFA.Clustering

/-!
# How well a family cuts a population

The clustering's gate reads each population on a sample.  What it certifies, averaged over the
population, is the chance a prefix is decided the wrong way and the chance it is left
undecided, over the oracle's noise.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

/-- The chance, over the oracle's noise, that `F` decides `p` on the wrong side. -/
noncomputable def miscutProb (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) : ℝ :=
  μ.real {ω | ¬ cutCorrect O lo hi F p ω}

/-- The chance that `F` leaves `p` undecided. -/
noncomputable def undecidedProb (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) : ℝ :=
  μ.real {ω | ¬ decided O.mq lo hi F p ω}

/-- With probability `≥ 1 − δ − α` the loop stops at one of `states`, and whenever it stops, the
family it returns there, averaged over a tame population, decides a prefix the wrong way at
most `εcov + εs + slack` of the time on the uniform pool and at most
`1/2 + εs + slack` on any tame population, and leaves it undecided at most
`2·indecisionLimit + slack` of the time on each.  Every state's band is as wide as
`ClusteringGuarantee`'s.  The tame populations' atoms are held as there, at `δ/2`, and also
against `slack`: the table's own prefixes are read with the noise the family was chosen on. -/
def ClusteringQualityGuarantee : Prop :=
  ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [Stringlike S] {J : Type*} [Fintype J]
    (O : Oracle μ S) (populations tame : Finset J) (uni : J) (H : J → Finset S)
    (Pre Suf : Set S) (η₀ indecisionLimit εcov εs α δ pAP crossLimit slack : ℝ),
  O.η ≤ η₀ →
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
  0 < δ →
  δ ≤ 1 →
  0 < crossLimit →
  0 < slack →
  ∃ states : Finset State,
    (∀ B ∈ states, ∀ F : Finset S, F.card + 1 ≤ B.k → ∀ p,
      (B.hi < meanVote O F p → μ.real {ω | voteCount O.mq F p ω ≤ B.lo} ≤ crossLimit)
      ∧ (meanVote O F p ≤ B.lo → μ.real {ω | B.hi < voteCount O.mq F p ω} ≤ crossLimit)) ∧
    ∀ (D : J → Measure S) (Dsf : Measure S),
      (∀ j, IsProbabilityMeasure (D j)) → IsProbabilityMeasure Dsf →
      (∀ j ∈ populations, D j Preᶜ = 0) →
      Dsf Sufᶜ = 0 →
      pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 18 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t}
          ≤ rateScale η₀ indecisionLimit εcov * (1 / 2 - η₀)) →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 8 * ((populations.card : ℝ) + 3) * (B.npref : ℝ) * (D j).real {t} ≤ εs) →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 16 * (populations.card : ℝ) * (3 + 2 * (Nat.log 2 B.npref : ℝ)) * (D j).real {t}
          ≤ δ * min εs indecisionLimit ^ 2) →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        2 ^ 4 * ((populations.card : ℝ) + 1) * (3 + 2 * (Nat.log 2 B.npref : ℝ)) * (D j).real {t}
          ≤ δ * slack ^ 2) →
      (∀ B ∈ states, ∀ j ∈ tame, ∀ t : S,
        4 * (populations.card : ℝ) * (B.npref : ℝ) * (D j).real {t} ≤ slack) →
      (∀ B ∈ states,
        2 ^ 13 * ((populations.card : ℝ) + 1) * (B.nsuff : ℝ) * collisionMass Dsf ≤ δ * pAP) →
        1 - δ - α ≤ (runMeasure μ D Dsf).real
          {x | (∃ B : {B : State // B ∈ states},
                streams H uni x ∈ ret O.mq populations uni η₀ indecisionLimit εcov α B.val)
            ∧ ∀ B : {B : State // B ∈ states},
              streams H uni x ∈ ret O.mq populations uni η₀ indecisionLimit εcov α B.val →
              ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                  (familyAt O.mq populations (streams H uni x) B.val) p ∂(D uni)
                ≤ εcov + εs + slack
              ∧ ∀ j ∈ tame,
                ∫ p, miscutProb O B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations (streams H uni x) B.val) p ∂(D j)
                  ≤ 1 / 2 + εs + slack
                ∧ ∫ p, undecidedProb O B.val.lo (B.val.hi + 1)
                    (familyAt O.mq populations (streams H uni x) B.val) p ∂(D j)
                  ≤ 2 * indecisionLimit + slack}

end OrthoDFA
