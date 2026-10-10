import OrthoDFA.TallyRound

/-!
# One round of the tally loop, read by the suffix family

`TallyRound` with its reads the family's (`FamilyReadBound`) and its noise event `TallyE`
replaced by its fields' tails.

Over the class of at most `(1 + (n + 1)³ |Σ|)ⁿ` trees, `n = |Q| + S`, at most
`(n + 3)^((n + 2) |Σ|)` edge maps each, the tails are, with `θ₁ = (n + 1) · 1.5θ`:
* records, at `κs ≥ 2κ` of a probe's undecided strings, `ρ` less `2ε (L + 1)(n + 1)` against
  `exp(−slack / (2 p₀ (1 + 4 (κ (L + 1) + ε (L + 1)(n + 1)))))`;
* edges, at each of `(n + 2) |Σ|` edges, its traffic gate's slack `φe (θe/2 − 2θg)/2` against
  `exp(−slack / (2 L p₀ (1 + 8θ₁)))`;
* the start, `exp(θ₁ (e^{l p₀} − 1)/p₀ − l θgs)` for any `l ≥ 0`;
* the middles, the search-rate gate's slack `θgpt φpt/2` against
  `exp(−slack / (2 p₀ (1 + 4 (L + 1) θ₁)))`;
where `p₀` bounds the mass of the probes' length-`k` prefixes, and the fields' rates are
`θg ≥ 2θ₁` of an edge's positions and `θgpt ≥ 2 (L + 1) θ₁` of the searches.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

/-- `TallyRoundFamily`: with the language a DFA's, the family suffix-free and the band passing
`BandPasses` at `ε, ε₂, κ`, there is a read model whose laws are the family's, at `θ` and `κs`,
under which the round over the family's reads ends well but for the noise fields' tails and
`TallyRound`'s other terms. -/
def TallyRoundFamily : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (O : Oracle μ (FreeMonoid α)) (M : _root_.DFA α σ) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (ε ε₂ κ κs θ : ℝ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : TallyCfg) (S nEnd nRec hP hS L T : ℕ)
    (ρ θg θgs θgpt θpt' θr εd' Xe θs' Xs η ν p₀ l : ℝ),
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) → SuffixFree F →
    BandPasses F.card kl kh (1 - O.ηIn) O.ηOut ε ε₂ κ →
    0 < κ → 2 * κ ≤ κs → 0 ≤ ε → 2 * (ε * ((L + 1) * (Fintype.card σ + S + 1))) ≤ ρ →
    0 ≤ θ → (∀ᵐ x ∂D, C.k ≤ x.toList.length) → 0 < p₀ →
    (∀ u, D.real {x | prefixOf x C.k = u} ≤ p₀) → 0 ≤ l →
    2 * ((Fintype.card σ + S + 1) * (3 / 2 * θ)) ≤ θg →
    2 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * θ)) ≤ θgpt →
    (∀ᵐ x ∂D, x.toList.length ≤ L) →
    0 < C.m → Fintype.card σ + S + 3 ≤ C.Lmax → C.n₀ ≤ nEnd →
    C.n₀ ≤ hS + 1 → 0 ≤ C.θpt → C.θpt ≤ 1 → 0 ≤ C.εd → C.εd ≤ 1 → 0 ≤ θpt' → θpt' ≤ 1 →
    0 ≤ θr → θr ≤ 1 → 0 ≤ εd' → εd' ≤ 1 → θr ≤ (1 - θpt') * εd' →
    binomSfGe (hS + 1) C.θpt hP < C.a → 1 - binomSfGe nEnd C.εd (hS + 1) < C.a →
    C.a ≤ binomSfGe nEnd C.εd hS → C.φpt * nEnd ≤ hS + 1 →
    1 ≤ L → 0 < C.a → C.a ≤ 1 → 4 * θg ≤ C.θe → 0 ≤ C.φe →
    (∀ j, 4 * L * (1 + 4 * θg * C.Lmax) * Real.log (1 / C.a) < C.exc j) → 0 ≤ θgs → θgs ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θs h < C.a → binomSfGe t θgs ((h + 1) / 2) ≤ C.a) →
    2 * θgpt ≤ 1 →
    (∀ t h, C.n₀ ≤ t → binomSfGe t C.θpt h < C.a → binomSfGe t (2 * θgpt) ((h + 1) / 2) ≤ C.a) →
    0 ≤ C.φpt → C.φpt ≤ 1 →
    (∀ n, C.n₀ ≤ n → binomSfGe n (C.φpt / 2) ⌈C.φpt * n⌉₊ ≤ C.a) →
    0 ≤ Xe → (∀ j, C.exc j ≤ Xe) → 0 ≤ θs' → C.n₀ ≤ Xs →
    (∀ n h : ℕ, C.n₀ ≤ n → θs' * n + Xs ≤ h → binomSfGe n C.θs h < C.a) →
    0 ≤ η → 0 ≤ ν → (Real.exp η - 1) * κs * (L + 1) ≤ 1 - Real.exp (-(ν * (L + 1))) →
    (S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec ≤ T →
    ∃ G : ReadModel α σ, G.M = M ∧ G.θ = θ ∧ G.κ = κs ∧ G.ε = ε
      ∧ (∀ w, readProb O F kl kh w = G.dist (M.eval w.toList))
      ∧ ∫⁻ ω, (Measure.pi fun _ : Fin T => D)
          {xs | ¬ RunEnds (tallyStep C fun z => (familyRead O.mq F kl kh z ω).cut)
            (TallyEndsWell G D C (familyRead O.mq F kl kh · ω)) tallyStart (List.ofFn xs)} ∂μ
        ≤ ENNReal.ofReal ((1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α)
              ^ (Fintype.card σ + S)
            * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
            * Real.exp (-((ρ - 2 * (ε * ((L + 1) * (Fintype.card σ + S + 1))))
              / (2 * p₀ * (1 + 4 * (κ * (L + 1) + ε * ((L + 1) * (Fintype.card σ + S + 1))))))))
          + ENNReal.ofReal ((1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α)
              ^ (Fintype.card σ + S)
            * ((Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
              * ((Fintype.card σ + S + 2) * Fintype.card α))
            * Real.exp (-(C.φe * (C.θe / 2 - 2 * θg) / 2
              / (2 * L * p₀ * (1 + 8 * ((Fintype.card σ + S + 1) * (3 / 2 * θ)))))))
          + ENNReal.ofReal ((1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α)
              ^ (Fintype.card σ + S)
            * Real.exp ((Fintype.card σ + S + 1) * (3 / 2 * θ) * (Real.exp (l * p₀) - 1) / p₀
              - l * θgs))
          + ENNReal.ofReal ((1 + (Fintype.card σ + S + 1) ^ 3 * Fintype.card α)
              ^ (Fintype.card σ + S)
            * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
            * Real.exp (-(θgpt * C.φpt / 2
              / (2 * p₀ * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * θ)))))))
          + ENNReal.ofReal (fakeRun C (Fintype.card α) S L
            ((S + Fintype.card σ + 1) * subT C (Fintype.card α) nEnd nRec) ρ Xe θs' Xs η ν)
          + ENNReal.ofReal ((S + Fintype.card σ + 1) * subOpen C (Fintype.card α)
            (Fintype.card σ) nRec ((S + 1) * C.m) θr (termLevel nEnd hP hS θpt' εd'))
          + ENNReal.ofReal ((2 ^ (C.Lmax + 1) * Fintype.card α + T + 3 * (T * T)) * C.a)

end OrthoDFA

end
