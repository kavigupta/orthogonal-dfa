import OrthoDFA.Level3Claim

/-!
# The random round with rare wrong reads

Level 3 (`OrthoDFA.Level3`), unchanged, with every read on its state's wrong side with
chance at most `εw`. A wrong read can make a record untrue, and `m` untrue records at one edge
and a target that no state at the edge's leaf reaches under its letter redirect the edge there,
or split its leaf where no two of its states part.

`EpsRoundCorrect`: the chance of not ending well is at most `epsRoundRisk`, the least over the
admissible knobs of `epsRoundBound`: `roundBound`, plus `noiseRisk` for the probes that can read a
string wrong reaching `Gw` of the probes, and, for each stretch, `fakeRisk`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (replay)

/-- What a stretch risks where reads can be wrong: `m` records, over its first `Ns` probes, at
one of the at most `Lmax² |Σ|` edges and targets, each from a probe that can read a string wrong,
at most `Gw` of the probes. -/
noncomputable def fakeRisk (C : Level3Settings) (nα Ns : ℕ) (Gw : ℝ) : ℝ :=
  (C.maxLeaves ^ 2 * nα : ℕ) * binomSfGe Ns Gw C.recordsToMove

/-- The proof's free parameters where reads can be wrong: `Knobs`, and the probes that can read a
string wrong, as a share of the probes. -/
structure EpsKnobs extends Knobs where
  Gw : ℝ

/-- `Knobs.Admissible`, and `Gw` a share. -/
structure EpsKnobs.Admissible {σ : Type*} [Fintype σ] [Fintype α] (θ : ℝ) (C : Level3Settings)
    (ε : ℝ) (kn : EpsKnobs) : Prop where
  base : kn.toKnobs.Admissible (σ := σ) (α := α) θ C ε
  Gw_nonneg : 0 ≤ kn.Gw
  Gw_le : kn.Gw ≤ 1

/-- `roundBound`, plus `noiseRisk` for the probes that can read a string wrong reaching `Gw` of the
probes, and, for each stretch, `fakeRisk`. -/
noncomputable def epsRoundBound (σ : Type*) [Fintype σ] [Fintype α] (θ εw : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (kn : EpsKnobs) : ℝ :=
  roundBound (α := α) σ θ D L C kn.toKnobs
    + noiseRisk (Fintype.card σ) (Fintype.card α) L εw (prefixMax D C.walkStart) kn.Gw
    + stretches C (Fintype.card α) * fakeRisk C (Fintype.card α) kn.Ns kn.Gw

/-- The least of `epsRoundBound` over the admissible knobs. -/
noncomputable def epsRoundRisk (σ : Type*) [Fintype σ] [Fintype α] (θ εw : ℝ)
    (D : Measure (FreeMonoid α)) (L : ℕ) (C : Level3Settings) (ε : ℝ) : ENNReal :=
  ⨅ (kn : EpsKnobs) (_ : kn.Admissible (σ := σ) (α := α) θ C ε),
    ENNReal.ofReal (epsRoundBound (α := α) σ θ εw D L C kn)

/-- `ReadOracle` with each read on its state's wrong side with chance at most `εw`. -/
structure EpsReadOracle {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) {σ : Type*}
    (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (εw : ℝ) where
  read : FreeMonoid α → Ω → ARU
  measurable : ∀ z, Measurable (read z)
  indep : ProbabilityTheory.iIndepFun read μ
  undecided : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)
  wrong : ∀ z, μ.real {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} ≤ εw

/-- `RandomRoundCorrect`, its yield included, with each read wrong with chance at most `εw`. -/
def EpsRoundCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
    (side : σ → Bool) (U : σ → ℝ) (θ εw : ℝ) {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α)
    (C : Level3Settings) (ε : ℝ),
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | let r := runProbes (R.read · ω) C initialState (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) P.D ε
            {x | Disagrees (R.read · ω) ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
            P.firstAnchor P.L (replay (R.read · ω) ⟨r.1.tree, r.1.edges⟩)
            (3 / 4 * θ * coverage P.D P.firstAnchor P.L M (BadAt U θ)
              / (P.L + 1 - P.firstAnchor : ℕ))
            r.2} ∂μ
      ≤ epsRoundRisk (α := α) σ θ εw P.D P.L C ε
        + ENNReal.ofReal (Real.exp (-(3 / 2 * θ * coverage P.D P.firstAnchor P.L M (BadAt U θ))
          / (16 * P.prefixBound)))

end Random

end OrthoDFA
