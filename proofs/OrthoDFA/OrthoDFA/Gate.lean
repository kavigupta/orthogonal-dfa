import OrthoDFA.Estimate

/-!
# The gate model: what the gate accepts and rejects

The gate makes two checks per population, both sum-vs-threshold on reads:

* **certification** — admit a side when its read-sum clears the admit threshold
  `m((β+τ)+margin)`.  A *drifted* side (mean ≤ β+τ) clears it w.p. ≤ α
  (`certErr_bound`); a *clean* side (mean ≥ β+s) fails to clear it only w.p.
  ≤ `exp(-2m((s-τ)-margin)²)` (`cleanAdmit_le`, below).
* **FNR** — accept when the indecisive count stays below `m·εfnr`.  An
  accept-preserving family, whose per-prefix indecision rate is `fnrpp ≤ εfnr`,
  exceeds it only w.p. ≤ `exp(-2m(εfnr-fnrpp)²)` (`apLowFNR_le`, below).

So an accept-preserving family clears both checks except with exponentially small
probability, while a drifted one is admitted only with probability ≤ α.  These are
the two facts `hgood`/`hbad` of `algorithm_correct` rest on; both are instances of
the discharged one-sided Hoeffding lemmas.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- **Gate accepts a clean side.**  A side whose read-mean is at least `β+s`
(genuinely its own class) fails the admit test — its sum falling below the
threshold `m((β+τ)+margin)` — with probability at most `exp(-2m((s-τ)-margin)²)`,
provided the admit margin is within `s-τ`. -/
theorem cleanAdmit_le (X : ℕ → Ω → ℝ) (m : ℕ) (β s τ margin : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : (m : ℝ) * (β + s) ≤ ∑ i ∈ Finset.range m, μ[X i])
    (hmargin : margin ≤ s - τ) :
    μ.real {ω | ∑ i ∈ Finset.range m, X i ω ≤ (m : ℝ) * ((β + τ) + margin)}
      ≤ Real.exp (-2 * (m : ℝ) * ((s - τ) - margin) ^ 2) := by
  have h := sumLower_le X (Finset.range m) (β + s) ((s - τ) - margin) hmeas h_indep hIcc
    (by simpa [Finset.card_range] using hmean) (by linarith)
  rw [Finset.card_range] at h
  simpa only [show (β + s) - ((s - τ) - margin) = (β + τ) + margin by ring] using h

/-- **Gate accepts an accept-preserving family (FNR check).**  If the per-prefix
indecision rate is at most `fnrpp ≤ εfnr`, the empirical indecisive count reaching
`m·εfnr` has probability at most `exp(-2m(εfnr-fnrpp)²)`. -/
theorem apLowFNR_le (I : ℕ → Ω → ℝ) (m : ℕ) (fnrpp εfnr : ℝ)
    (hmeas : ∀ i, AEMeasurable (I i) μ) (h_indep : iIndepFun I μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, I i ω ∈ Set.Icc (0 : ℝ) 1)
    (hmean : ∑ i ∈ Finset.range m, μ[I i] ≤ (m : ℝ) * fnrpp)
    (hle : fnrpp ≤ εfnr) :
    μ.real {ω | (m : ℝ) * εfnr ≤ ∑ i ∈ Finset.range m, I i ω}
      ≤ Real.exp (-2 * (m : ℝ) * (εfnr - fnrpp) ^ 2) := by
  have h := sumUpper_le I (Finset.range m) fnrpp (εfnr - fnrpp) hmeas h_indep hIcc
    (by simpa [Finset.card_range] using hmean) (by linarith)
  rw [Finset.card_range] at h
  simpa only [show fnrpp + (εfnr - fnrpp) = εfnr by ring] using h

#print axioms cleanAdmit_le
#print axioms apLowFNR_le

end OrthoDFA
