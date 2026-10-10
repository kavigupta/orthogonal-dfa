import OrthoDFA.Proofs.RandomTree

/-!
# The read field

The reads are independent across strings, each undecided with its state's chance. A probe of
`x` reads only strings with `x`'s length-`k` prefix, so the probes of distinct length-`k`
prefixes read disjoint strings, and for a fixed tree the mass of probes that can read a good
read-state undecided is a sum of independent terms, each at most `p₀`, of mean at most
`1.5θ (L + 1)(|Q| + 1)` times their prefix's mass. Its exponential moment bounds its upper tail;
`noise_le` takes the union over `classSet |Q|`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory
open OrthoDFA.Ideal (DTree pre)

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem noise_le {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) {Ω : Type*}
    [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L : ℕ) (p₀ G : ℝ)
    (hmeas : ∀ z, Measurable (read z)) (hind : ProbabilityTheory.iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList))
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, u.toList.length = k → D.real {x | pre x k = u} ≤ p₀)
    (hθ : 0 ≤ θ) :
    μ {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
        D.real (PotGood M U θ k (read · ω) T) ≤ G}
      ≤ ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L θ p₀ G) := by
  sorry

end Random

end OrthoDFA
