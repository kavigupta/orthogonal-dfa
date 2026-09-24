import OrthoDFA.Proofs.Algorithm
import OrthoDFA.Proofs.Fuse

/-!
# The finished capstone

Wires `fuse_findability` and `certErr_bound` into `algorithm_correct`, so neither
per-round input is a bare hypothesis:

* `hgood` is discharged by `fuse_findability` — a good pass has probability
  `≥ (1-reject)·pp`, findability `pp` times the gate-accept factor `1-reject`
  (the factor is `cleanAdmit_le`/`apLowFNR_le`, supplied as `haccept`);
* `hbad` is discharged by `certErr_bound` on the reads marginal.

Each round's space is the product of a proposal space and a reads space (the
persistent oracle pre-draws every read, so reads are independent of the proposal).
The only inputs left are the oracle model, the findability rate `pp`, and the
gate-accept factor `1-reject`.  Conclusion:

    P[failure] ≤ (1 - (1-reject)·pp)^N + N·a.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

/-- Second marginal of a product: a reads-only event has the product-probability
of its reads-probability. -/
theorem prod_snd_real {P R : Type*} [MeasurableSpace P] [MeasurableSpace R]
    (μprop : Measure P) [IsProbabilityMeasure μprop]
    (μreads : Measure R) [SFinite μreads]
    (E : Set R) :
    (μprop.prod μreads).real {y : P × R | y.2 ∈ E} = μreads.real E := by
  have hset : {y : P × R | y.2 ∈ E} = Set.univ ×ˢ E := by ext y; simp
  rw [measureReal_def, hset, Measure.prod_prod, measure_univ, one_mul, ← measureReal_def]

variable {N : ℕ} {Pr Rr : Fin N → Type*}
  [∀ r, MeasurableSpace (Pr r)] [∀ r, MeasurableSpace (Rr r)]

/-- Finished capstone.  Inputs: the oracle model (per-round certification reads
`Xc`), findability `pp`, and the gate-accept factor `1-reject`.  Both per-round
bounds of `algorithm_correct` are discharged. -/
theorem clustering_algorithm_correct_fused
    (μprop : ∀ r, Measure (Pr r)) [∀ r, IsProbabilityMeasure (μprop r)]
    (μreads : ∀ r, Measure (Rr r)) [∀ r, IsProbabilityMeasure (μreads r)]
    (isAP : ∀ r, Set (Pr r)) (hAP : ∀ r, MeasurableSet (isAP r))
    (clears : ∀ r, Pr r → Set (Rr r))
    (hclears : ∀ r, MeasurableSet {x : Pr r × Rr r | x.2 ∈ clears r x.1})
    (pp reject : ℝ) (hpp0 : 0 ≤ pp) (hpp1 : pp ≤ 1) (hrej0 : 0 ≤ reject) (hrej1 : reject ≤ 1)
    (hfind : ∀ r, pp ≤ (μprop r).real (isAP r))
    (haccept : ∀ r, ∀ V ∈ isAP r, 1 - reject ≤ (μreads r).real (clears r V))
    (β : ℝ) (n : ℕ) (τ a : ℝ) (hn : 0 < n) (ha0 : 0 < a) (ha1 : a ≤ 1)
    (Xc : ∀ r, ℕ → Rr r → ℝ)
    (hcmeas : ∀ r i, Measurable (Xc r i))
    (hcindep : ∀ r, iIndepFun (Xc r) (μreads r))
    (hcIcc : ∀ r i, ∀ᵐ ω ∂(μreads r), Xc r i ω ∈ Set.Icc (0 : ℝ) 1)
    (hcmean : ∀ r, ∑ i ∈ Finset.range n, (μreads r)[Xc r i] ≤ (n : ℝ) * (β + τ)) :
    (Measure.pi (fun r => (μprop r).prod (μreads r))).real
        ({x | ∀ r, x r ∉ {y : Pr r × Rr r | y.1 ∈ isAP r ∧ y.2 ∈ clears r y.1}}
          ∪ {x | ∃ r, x r ∈ {y : Pr r × Rr r |
              (n : ℝ) * ((β + τ) + certMargin n a)
                ≤ ∑ i ∈ Finset.range n, Xc r i y.2}})
      ≤ (1 - (1 - reject) * pp) ^ N + (N : ℝ) * a := by
  -- reads-only certification event and its measurability
  have hEmeas : ∀ r, MeasurableSet {ω : Rr r |
      (n : ℝ) * ((β + τ) + certMargin n a) ≤ ∑ i ∈ Finset.range n, Xc r i ω} := fun r =>
    measurableSet_le measurable_const (Finset.measurable_sum _ (fun i _ => hcmeas r i))
  refine algorithm_correct (fun r => (μprop r).prod (μreads r))
    (fun r => {y : Pr r × Rr r | y.1 ∈ isAP r ∧ y.2 ∈ clears r y.1})
    (fun r => {y : Pr r × Rr r |
      (n : ℝ) * ((β + τ) + certMargin n a) ≤ ∑ i ∈ Finset.range n, Xc r i y.2})
    (fun r => ((hAP r).preimage measurable_fst).inter (hclears r))
    (fun r => measurableSet_le measurable_const
      (Finset.measurable_sum _ (fun i _ => (hcmeas r i).comp measurable_snd)))
    ((1 - reject) * pp) a (mul_le_one₀ (by linarith) hpp0 hpp1) ?_ ?_
  · exact fun r => fuse_findability (μprop r) (μreads r) (isAP r) (hAP r) (clears r)
      (hclears r) pp reject hpp0 hrej1 (hfind r) (haccept r)
  · intro r
    have heq : ((μprop r).prod (μreads r)).real
          {y : Pr r × Rr r | (n : ℝ) * ((β + τ) + certMargin n a)
            ≤ ∑ i ∈ Finset.range n, Xc r i y.2}
        = (μreads r).real {ω : Rr r | (n : ℝ) * ((β + τ) + certMargin n a)
            ≤ ∑ i ∈ Finset.range n, Xc r i ω} :=
      prod_snd_real (μprop r) (μreads r)
        {ω : Rr r | (n : ℝ) * ((β + τ) + certMargin n a)
          ≤ ∑ i ∈ Finset.range n, Xc r i ω}
    rw [heq]
    exact certErr_bound (Xc r) n β τ a (fun i => (hcmeas r i).aemeasurable) (hcindep r)
      (hcIcc r) (hcmean r) hn ha0 ha1

#print axioms clustering_algorithm_correct_fused

/-- Finished capstone, accumulating pool.  The independent-rounds idealization
removed: the good-pass event `goodErr r` may depend on the whole accumulated
history (the representative pool that grows as fresh strings are added each round);
only the findability `trigger r` — the fresh material drawn in round `r` — is
block-local and independent across rounds, and it forces a good round
(`himplies`).  Certification uses each round's fresh reads `Xc r` (block-local, as
#257 draws fresh prefixes to certify), so `hbad` is discharged from `certErr_bound`
via the coordinate marginal.  Conclusion, with only the oracle model and
findability as inputs:

    P[failure] ≤ (1 - p)^N + N·a. -/
theorem clustering_algorithm_correct_general
    {N : ℕ} {α : Fin N → Type*} [∀ r, MeasurableSpace (α r)]
    (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    (trigger : ∀ r, Set (α r)) (htrig : ∀ r, MeasurableSet (trigger r))
    (goodErr : Fin N → Set (∀ r, α r))
    (himplies : ∀ r, ∀ x : ∀ r, α r, x r ∈ trigger r → x ∉ goodErr r)
    (p : ℝ) (hp1 : p ≤ 1) (hp : ∀ r, p ≤ (ν r).real (trigger r))
    (β : ℝ) (n : ℕ) (τ a : ℝ) (hn : 0 < n) (ha0 : 0 < a) (ha1 : a ≤ 1)
    (Xc : ∀ r, ℕ → α r → ℝ)
    (hcmeas : ∀ r i, Measurable (Xc r i))
    (hcindep : ∀ r, iIndepFun (Xc r) (ν r))
    (hcIcc : ∀ r i, ∀ᵐ ω ∂(ν r), Xc r i ω ∈ Set.Icc (0 : ℝ) 1)
    (hcmean : ∀ r, ∑ i ∈ Finset.range n, (ν r)[Xc r i] ≤ (n : ℝ) * (β + τ)) :
    (Measure.pi ν).real
        ({x | ∀ r, x ∈ goodErr r}
          ∪ (⋃ r, {x | x r ∈ {ω | (n : ℝ) * ((β + τ) + certMargin n a)
                ≤ ∑ i ∈ Finset.range n, Xc r i ω}}))
      ≤ (1 - p) ^ N + (N : ℝ) * a := by
  refine algorithm_correct_general ν trigger htrig goodErr
    (fun r => {x | x r ∈ {ω | (n : ℝ) * ((β + τ) + certMargin n a)
      ≤ ∑ i ∈ Finset.range n, Xc r i ω}})
    himplies p a hp1 hp ?_
  intro r
  rw [pi_coord_real ν r _
    (measurableSet_le measurable_const (Finset.measurable_sum _ (fun i _ => hcmeas r i)))]
  exact certErr_bound (Xc r) n β τ a (fun i => (hcmeas r i).aemeasurable) (hcindep r)
    (hcIcc r) (hcmean r) hn ha0 ha1

#print axioms clustering_algorithm_correct_general

end OrthoDFA
