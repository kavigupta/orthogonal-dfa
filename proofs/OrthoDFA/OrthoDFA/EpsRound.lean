import OrthoDFA.RandomRound

/-!
# The random round with rare wrong reads

The round of `OrthoDFA.RandomRound`, unchanged, with every read on its state's wrong side with
chance at most `εw`. A wrong read can make a record untrue, and `m` untrue records at one edge
and a target that no state at the edge's leaf reaches under its letter redirect the edge there,
or split its leaf where no two of its states part.

`EpsRoundCorrect`: the chance of not ending well is bounded as in `RandomRoundCorrect`, plus
`noiseRisk` for the probes that can read a string wrong reaching `Gw` of the probes, and, for
each stretch, `fakeRisk`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (pre Disagrees)

/-- What a stretch risks where reads can be wrong: `m` records, over its first `Ns` probes, at
one of the at most `Lmax² |Σ|` edges and targets, each from a probe that can read a string wrong,
at most `Gw` of the probes. -/
noncomputable def fakeRisk (C : Cfg) (nα Ns : ℕ) (Gw : ℝ) : ℝ :=
  (C.Lmax ^ 2 * nα : ℕ) * binomSfGe Ns Gw C.m

/-- `RandomRoundCorrect` with each read wrong with chance at most `εw`. -/
def EpsRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (C : Cfg) (L P Ns N₁ : ℕ) (p₀ ε G Gw θr εd' θpt' : ℝ),
    (∀ z, Measurable (read z)) → ProbabilityTheory.iIndepFun read μ →
    (∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) →
    (∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} ≤ εw) →
    0 ≤ εw → (∀ᵐ x ∂D, x.toList.length ≤ L) → 0 < p₀ →
    (∀ u : FreeMonoid α, u.toList.length = C.k → D.real {x | pre x C.k = u} ≤ p₀) →
    Fintype.card σ + 2 ≤ C.Lmax → 1 ≤ C.m → 1 ≤ C.n₀ → C.n₀ ≤ N₁ → N₁ ≤ Ns →
    0 ≤ θ → 0 ≤ G → G ≤ 1 → 0 ≤ Gw → Gw ≤ 1 → 0 ≤ C.a → 0 ≤ C.θs → C.θs ≤ 1 → 0 ≤ C.θe →
    0 ≤ C.θpt → C.θpt ≤ 1 → 0 ≤ C.εd → C.εd ≤ 1 → C.εd ≤ ε → 0 ≤ θr → θr ≤ 1 → 0 ≤ εd' →
    εd' ≤ 1 → 0 ≤ θpt' → θpt' ≤ 1 →
    (C.Lmax : ℝ) ^ 2 * Fintype.card α * θr ≤ (1 - θpt') * εd' →
    stretches C (Fintype.card α) * Ns ≤ P →
    ∫⁻ ω, (Measure.pi fun _ : Fin P => D)
        {xs | let r := round (read · ω) C start (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) r.1.tree r.1.edges C.k x}
            (toRoundEnd r.2)} ∂μ
      ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G
        + noiseRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw
        + stretches C (Fintype.card α)
          * (stretchRisk C L Ns N₁ G θr εd' θpt' + fakeRisk C (Fintype.card α) Ns Gw))

end Random

end OrthoDFA
