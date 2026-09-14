import Mathlib.MeasureTheory.Measure.Prod
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic

/-!
# Fusing findability with the gate-accept factor

The remaining wiring behind `hgood`.  A round's randomness is a proposal together
with the oracle reads.  The persistent oracle pre-draws every string's read, so
the reads are independent of which family is proposed: the round's space is the
product `μ_prop × μ_reads`.

`fuse_findability` then fuses the two facts behind a good pass with no bespoke
kernel — just Fubini on the product:

* findability `hfind` — the proposal is accept-preserving with probability ≥ `pp`;
* the gate-accept factor `haccept` — for every accept-preserving proposal `V`, the
  gate clears it on the reads with probability ≥ `1 - reject` (this is what
  `cleanAdmit_le` / `apLowFNR_le` give).

Conclusion: a good pass — an accept-preserving proposal that the gate clears — has
probability at least `(1 - reject)·pp`.  That is the `p` of `algorithm_correct`,
so `hgood` is no longer a bare input: it is `p_preserve` times a proved factor.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {P R : Type*} [MeasurableSpace P] [MeasurableSpace R]

/-- **Findability fusion.**  On the product of the proposal measure and the reads
measure, the good-pass event `{(V,ω) | V accept-preserving ∧ gate clears V on ω}`
has probability at least `(1 - reject)·pp`. -/
theorem fuse_findability
    (μprop : Measure P) [IsProbabilityMeasure μprop]
    (μreads : Measure R) [IsProbabilityMeasure μreads]
    (isAP : Set P) (hAP : MeasurableSet isAP)
    (clears : P → Set R) (hclears : MeasurableSet {x : P × R | x.2 ∈ clears x.1})
    (pp reject : ℝ) (hpp0 : 0 ≤ pp) (hrej1 : reject ≤ 1)
    (hfind : pp ≤ μprop.real isAP)
    (haccept : ∀ V ∈ isAP, 1 - reject ≤ μreads.real (clears V)) :
    (1 - reject) * pp
      ≤ (μprop.prod μreads).real {x | x.1 ∈ isAP ∧ x.2 ∈ clears x.1} := by
  have hrej0 : 0 ≤ 1 - reject := by linarith
  set gp : Set (P × R) := {x | x.1 ∈ isAP ∧ x.2 ∈ clears x.1} with hgp
  have hgpmeas : MeasurableSet gp := (hAP.preimage measurable_fst).inter hclears
  have hlow : ∀ V, isAP.indicator (fun _ => ENNReal.ofReal (1 - reject)) V
      ≤ μreads (Prod.mk V ⁻¹' gp) := by
    intro V
    by_cases h : V ∈ isAP
    · have hpre : Prod.mk V ⁻¹' gp = clears V := by ext ω; simp [hgp, h]
      rw [Set.indicator_of_mem h, hpre]
      calc ENNReal.ofReal (1 - reject)
          ≤ ENNReal.ofReal (μreads.real (clears V)) := ENNReal.ofReal_le_ofReal (haccept V h)
        _ = μreads (clears V) := by
            rw [measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)]
    · have hz : isAP.indicator (fun _ => ENNReal.ofReal (1 - reject)) V = 0 := by simp [h]
      rw [hz]; positivity
  have key : ENNReal.ofReal ((1 - reject) * pp) ≤ (μprop.prod μreads) gp := by
    rw [Measure.prod_apply hgpmeas, ENNReal.ofReal_mul hrej0]
    calc ENNReal.ofReal (1 - reject) * ENNReal.ofReal pp
        ≤ ENNReal.ofReal (1 - reject) * μprop isAP := by
          gcongr
          calc ENNReal.ofReal pp
              ≤ ENNReal.ofReal (μprop.real isAP) := ENNReal.ofReal_le_ofReal hfind
            _ = μprop isAP := by
                rw [measureReal_def, ENNReal.ofReal_toReal (measure_ne_top _ _)]
      _ = ∫⁻ V, isAP.indicator (fun _ => ENNReal.ofReal (1 - reject)) V ∂μprop := by
          rw [lintegral_indicator hAP, setLIntegral_const, mul_comm]
      _ ≤ ∫⁻ V, μreads (Prod.mk V ⁻¹' gp) ∂μprop := lintegral_mono hlow
  rw [measureReal_def]
  calc (1 - reject) * pp
      = (ENNReal.ofReal ((1 - reject) * pp)).toReal :=
        (ENNReal.toReal_ofReal (mul_nonneg hrej0 hpp0)).symm
    _ ≤ ((μprop.prod μreads) gp).toReal := ENNReal.toReal_mono (measure_ne_top _ _) key

#print axioms fuse_findability

end OrthoDFA
