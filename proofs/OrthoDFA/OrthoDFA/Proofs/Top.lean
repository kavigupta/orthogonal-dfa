import OrthoDFA.Proofs.Discharge

/-!
# Clustering correctness (top theorem, axiom-free, DFA-free)

The clustering algorithm's own guarantee, stated in its own terms: prefixes,
reads, and membership bits — no states, no automaton.  A run's oracle noise is the
probability space `μ`.  Two kinds of check can fail:

* placement — for a prefix with a clear membership (member or non-member under
  the returned accept-preserving family), the vote fails to decide its correct
  side.  By `misplacedMember_le` / `misplacedNonmember_le` each such event has
  probability at most `exp(-2k(s-τ)²)`.
* certification — for a prefix population, the returned family's cut is
  admitted while drifted past tolerance.  By `certErr_bound` each such event has
  probability at most `α`.

If the overall failure event lies in the union of these, its probability is at
most `#placement · exp(-2k(s-τ)²) + #populations · α`.  This is PR #257's promise
— the family properly places all but a bounded fraction of *each* population, and
each population is certified accept-preserving on its own — with the rate held per
population, never pooled.

`#print axioms` shows only Lean's core: the whole guarantee is proved from Mathlib.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Real-valued finite subadditivity, from the `ℝ≥0∞` version. -/
theorem measureReal_biUnion_le {ι : Type*} (s : Finset ι) (f : ι → Set Ω) :
    μ.real (⋃ i ∈ s, f i) ≤ ∑ i ∈ s, μ.real (f i) := by
  have hle : μ (⋃ i ∈ s, f i) ≤ ∑ i ∈ s, μ (f i) := measure_biUnion_finset_le s f
  have hfin : (∑ i ∈ s, μ (f i)) ≠ ⊤ :=
    (ENNReal.sum_lt_top.mpr (fun i _ => measure_lt_top μ (f i))).ne
  calc μ.real (⋃ i ∈ s, f i) = (μ (⋃ i ∈ s, f i)).toReal := rfl
    _ ≤ (∑ i ∈ s, μ (f i)).toReal := ENNReal.toReal_mono hfin hle
    _ = ∑ i ∈ s, (μ (f i)).toReal :=
          ENNReal.toReal_sum (fun i _ => measure_ne_top μ (f i))
    _ = ∑ i ∈ s, μ.real (f i) := rfl

/-- Clustering correctness (composition).  The failure probability of a run is
at most `#placement · fnrBound + #populations · α`, where `fnrBound` bounds each
placement failure (supply `exp(-2k(s-τ)²)` via `misplacedMember_le` /
`misplacedNonmember_le`) and `α` bounds each population's false-admit (via
`certErr_bound`).  The union bound is Mathlib's; the per-check bounds are the
discharged Hoeffding/certification lemmas. -/
theorem clustering_union_bound
    {ιp ιq : Type*} (fnrBound α : ℝ)
    (placement : Finset ιp) (placeErr : ιp → Set Ω)
    (hplace : ∀ p ∈ placement, μ.real (placeErr p) ≤ fnrBound)
    (populations : Finset ιq) (certErr : ιq → Set Ω)
    (hcert : ∀ q ∈ populations, μ.real (certErr q) ≤ α)
    (fail : Set Ω)
    (hfail : fail ⊆ (⋃ p ∈ placement, placeErr p) ∪ (⋃ q ∈ populations, certErr q)) :
    μ.real fail ≤ placement.card • fnrBound + populations.card • α := by
  calc μ.real fail
      ≤ μ.real ((⋃ p ∈ placement, placeErr p) ∪ (⋃ q ∈ populations, certErr q)) :=
        measureReal_mono hfail
    _ ≤ μ.real (⋃ p ∈ placement, placeErr p) + μ.real (⋃ q ∈ populations, certErr q) :=
        measureReal_union_le _ _
    _ ≤ (∑ p ∈ placement, μ.real (placeErr p)) + (∑ q ∈ populations, μ.real (certErr q)) :=
        add_le_add (measureReal_biUnion_le placement placeErr)
          (measureReal_biUnion_le populations certErr)
    _ ≤ (∑ _p ∈ placement, fnrBound) + (∑ _q ∈ populations, α) :=
        add_le_add (Finset.sum_le_sum hplace) (Finset.sum_le_sum hcert)
    _ = placement.card • fnrBound + populations.card • α := by
        rw [Finset.sum_const, Finset.sum_const]

#print axioms clustering_union_bound

end OrthoDFA
