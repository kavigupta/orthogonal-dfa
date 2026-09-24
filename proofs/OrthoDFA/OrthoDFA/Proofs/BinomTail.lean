import OrthoDFA.Proofs.Estimate
import Mathlib.Probability.Distributions.Binomial
import Mathlib.Probability.Independence.InfinitePi
import Mathlib.Probability.ProductMeasure

/-!
# The binomial tails are Hoeffding's

The gate's verdict is a binomial p-value, and what the proof needs of it is a tail bound.
Mathlib's `Bin(n, p)` is the size of a random subset of `Iio n` whose elements are taken
independently, so its coordinates are exactly the independent bits `wrongDecisive_le`
consumes — no "sum of i.i.d. Bernoulli is binomial" law has to be proved separately.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Measure Set
open scoped ENNReal NNReal unitInterval

/-- The coordinate laws behind `Bin(n, p)`: bit `i` is `p`-biased inside `Iio n` and dead
outside it. -/
noncomputable def binBits (n : ℕ) (p : I) (i : ℕ) : Measure Prop :=
  unitInterval.toNNReal p • dirac (i ∈ Iio n) + unitInterval.toNNReal (σ p) • dirac False

instance (n : ℕ) (p : I) (i : ℕ) : IsProbabilityMeasure (binBits n p i) := by
  unfold binBits
  constructor
  simp only [Measure.add_apply, Measure.smul_apply, measure_univ, smul_eq_mul, mul_one,
    ← ENNReal.coe_add]

lemma binomial_eq_map_bits (n : ℕ) (p : I) :
    ProbabilityTheory.binomial n p
      = Measure.map (fun q : ℕ → Prop => {i | q i}.ncard) (Measure.infinitePi (binBits n p)) := by
  rw [ProbabilityTheory.binomial, setBernoulli_eq_map, Measure.map_map measurable_ncard
    (by fun_prop)]
  rfl

/-- Outside `Iio n` the bits are dead. -/
lemma binBits_tail (n : ℕ) (p : I) {i : ℕ} (hi : n ≤ i) : binBits n p i = dirac False := by
  have hmem : (i ∈ Iio n) = False := by
    simp [Set.mem_Iio, Nat.not_lt.2 hi]
  rw [binBits, hmem, ← add_smul]
  have : unitInterval.toNNReal p + unitInterval.toNNReal (σ p) = 1 := by
    refine NNReal.coe_injective ?_
    rw [NNReal.coe_add, NNReal.coe_one]
    show (p : ℝ) + (σ p : ℝ) = 1
    rw [unitInterval.coe_symm_eq]
    ring
  rw [this, one_smul]

/-- The `i`-th bit has law `binBits n p i`. -/
lemma map_binBit (n : ℕ) (p : I) (i : ℕ) :
    Measure.map (fun q : ℕ → Prop => q i) (Measure.infinitePi (binBits n p)) = binBits n p i :=
  (measurePreserving_eval_infinitePi (binBits n p) i).map_eq

/-- Almost every subset of the bits lies in `Iio n`. -/
lemma ae_bits_lt (n : ℕ) (p : I) :
    ∀ᵐ q ∂(Measure.infinitePi (binBits n p)), ∀ i, n ≤ i → ¬ q i := by
  refine ae_all_iff.2 (fun i => ?_)
  by_cases hi : n ≤ i
  · have hz : Measure.infinitePi (binBits n p) {q : ℕ → Prop | ¬ (n ≤ i → ¬ q i)} = 0 := by
      have hset : {q : ℕ → Prop | ¬ (n ≤ i → ¬ q i)}
          = (fun q : ℕ → Prop => q i) ⁻¹' {P : Prop | P} := by
        ext q; simp [hi]
      rw [hset, ← Measure.map_apply (measurable_pi_apply i) (by trivial), map_binBit,
        binBits_tail n p hi, dirac_apply' _ (by trivial)]
      simp
    exact ae_iff.2 hz
  · filter_upwards with q hc
    exact absurd hc hi

open scoped Classical in
/-- On those subsets the count is the sum of the first `n` bits. -/
lemma ncard_eq_sum_bits {n : ℕ} {q : ℕ → Prop} (h : ∀ i, n ≤ i → ¬ q i) :
    (({i | q i}.ncard : ℕ) : ℝ) = ∑ i ∈ Finset.range n, (if q i then (1 : ℝ) else 0) := by
  classical
  have hset : {i | q i} = ↑((Finset.range n).filter (fun i => q i)) := by
    ext i
    simp only [Set.mem_setOf_eq, Finset.coe_filter, Finset.mem_range, Set.mem_setOf_eq]
    exact ⟨fun hq => ⟨by by_contra hc; exact h i (Nat.not_lt.1 hc) hq, hq⟩, fun hq => hq.2⟩
  rw [hset, Set.ncard_coe_finset, Finset.card_filter, Nat.cast_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  by_cases hq : q i <;> simp [hq]

open scoped Classical in
/-- Each bit of `Iio n` is a `[0,1]` read of mean `p`. -/
lemma integral_binBit (n : ℕ) (p : I) {i : ℕ} (hi : i < n) :
    (Measure.infinitePi (binBits n p))[fun q : ℕ → Prop => if q i then (1 : ℝ) else 0]
      = (p : ℝ) := by
  classical
  have hf : Measurable (fun P : Prop => if P then (1 : ℝ) else 0) := by
    exact measurable_from_top
  calc (Measure.infinitePi (binBits n p))[fun q : ℕ → Prop => if q i then (1 : ℝ) else 0]
      = ∫ P, (if P then (1 : ℝ) else 0) ∂(binBits n p i) := by
        rw [← map_binBit n p i,
          integral_map (measurable_pi_apply i).aemeasurable hf.aestronglyMeasurable]
    _ = (p : ℝ) := by
        have hmem : (i ∈ Iio n) = True := by simp [Set.mem_Iio, hi]
        rw [binBits, hmem, integral_add_measure
          ((integrable_dirac (by simp)).smul_measure (by simp))
          ((integrable_dirac (by simp)).smul_measure (by simp)),
          ]
        rw [integral_smul_nnreal_measure, integral_smul_nnreal_measure, integral_dirac,
          integral_dirac]
        simp only [NNReal.smul_def, smul_eq_mul, if_true, if_false, mul_one, mul_zero,
          add_zero]
        rfl

open scoped Classical in
/-- The bits are independent `[0,1]` reads of mean `p` on `Iio n`, which is what turns the
binomial tail into Hoeffding's. -/
lemma binBits_indep (n : ℕ) (p : I) :
    iIndepFun (fun (i : ℕ) (q : ℕ → Prop) => if q i then (1 : ℝ) else 0)
      (Measure.infinitePi (binBits n p)) :=
  (iIndepFun_infinitePi (fun _ => measurable_id)).comp
    (fun _ => fun P : Prop => if P then (1 : ℝ) else 0) (fun _ => measurable_from_top)

open scoped Classical in
/-- Hoeffding's upper tail for the binomial. -/
theorem binomial_real_ge_le (n : ℕ) (p : I) (τ : ℝ) (hτ : 0 ≤ τ) :
    (ProbabilityTheory.binomial n p).real {i : ℕ | (n : ℝ) * ((p : ℝ) + τ) ≤ (i : ℝ)}
      ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  classical
  set ν : Measure (ℕ → Prop) := Measure.infinitePi (binBits n p) with hν
  set X : ℕ → (ℕ → Prop) → ℝ := fun i q => if q i then (1 : ℝ) else 0 with hX
  have hmeasX : ∀ i, AEMeasurable (X i) ν := fun i =>
    ((measurable_from_top (f := fun P : Prop => if P then (1 : ℝ) else 0)).comp
      (measurable_pi_apply i)).aemeasurable
  have hIcc : ∀ i, ∀ᵐ q ∂ν, X i q ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with q
    by_cases h : q i <;> simp [hX, h]
  have hmean : ∑ i ∈ Finset.range n, ν[X i] ≤ ((Finset.range n).card : ℝ) * (p : ℝ) := by
    rw [Finset.sum_congr rfl (fun i hi => integral_binBit n p (Finset.mem_range.1 hi)),
      Finset.sum_const, nsmul_eq_mul]
  have hmain := wrongDecisive_le X (Finset.range n) (p : ℝ) τ hmeasX (binBits_indep n p) hIcc
    hmean hτ
  rw [Finset.card_range] at hmain
  have hmapmeas : Measurable (fun q : ℕ → Prop => ({i | q i}).ncard) :=
    measurable_ncard.comp (by fun_prop)
  have hAmeas : MeasurableSet {i : ℕ | (n : ℝ) * ((p : ℝ) + τ) ≤ (i : ℝ)} :=
    (Set.to_countable _).measurableSet
  have hpre : (ProbabilityTheory.binomial n p).real
        {i : ℕ | (n : ℝ) * ((p : ℝ) + τ) ≤ (i : ℝ)}
      = ν.real {q : ℕ → Prop | (n : ℝ) * ((p : ℝ) + τ) ≤ (({i | q i}).ncard : ℝ)} := by
    rw [binomial_eq_map_bits, measureReal_def, Measure.map_apply hmapmeas hAmeas,
      ← measureReal_def]
    rfl
  have hae : ν.real {q : ℕ → Prop | (n : ℝ) * ((p : ℝ) + τ) ≤ (({i | q i}).ncard : ℝ)}
      = ν.real {q : ℕ → Prop | (n : ℝ) * ((p : ℝ) + τ) ≤ ∑ i ∈ Finset.range n, X i q} := by
    refine measureReal_congr ?_
    filter_upwards [ae_bits_lt n p] with q hq
    show ((n : ℝ) * ((p : ℝ) + τ) ≤ (({i | q i}).ncard : ℝ))
      = ((n : ℝ) * ((p : ℝ) + τ) ≤ ∑ i ∈ Finset.range n, X i q)
    rw [ncard_eq_sum_bits hq]
  rw [hpre, hae]
  exact hmain

open scoped Classical in
/-- Hoeffding's lower tail for the binomial. -/
theorem binomial_real_le_le (n : ℕ) (p : I) (τ : ℝ) (hτ : 0 ≤ τ) :
    (ProbabilityTheory.binomial n p).real {i : ℕ | (i : ℝ) ≤ (n : ℝ) * ((p : ℝ) - τ)}
      ≤ Real.exp (-2 * (n : ℝ) * τ ^ 2) := by
  classical
  set ν : Measure (ℕ → Prop) := Measure.infinitePi (binBits n p) with hν
  set X : ℕ → (ℕ → Prop) → ℝ := fun i q => if q i then (1 : ℝ) else 0 with hX
  have hmeasX : ∀ i, AEMeasurable (X i) ν := fun i =>
    ((measurable_from_top (f := fun P : Prop => if P then (1 : ℝ) else 0)).comp
      (measurable_pi_apply i)).aemeasurable
  have hIcc : ∀ i, ∀ᵐ q ∂ν, X i q ∈ Set.Icc (0 : ℝ) 1 := by
    intro i
    filter_upwards with q
    by_cases h : q i <;> simp [hX, h]
  have hmean : ((Finset.range n).card : ℝ) * (p : ℝ) ≤ ∑ i ∈ Finset.range n, ν[X i] := by
    rw [Finset.sum_congr rfl (fun i hi => integral_binBit n p (Finset.mem_range.1 hi)),
      Finset.sum_const, nsmul_eq_mul]
  have hmain := sumLower_le X (Finset.range n) (p : ℝ) τ hmeasX (binBits_indep n p) hIcc
    hmean hτ
  rw [Finset.card_range] at hmain
  have hmapmeas : Measurable (fun q : ℕ → Prop => ({i | q i}).ncard) :=
    measurable_ncard.comp (by fun_prop)
  have hAmeas : MeasurableSet {i : ℕ | (i : ℝ) ≤ (n : ℝ) * ((p : ℝ) - τ)} :=
    (Set.to_countable _).measurableSet
  have hpre : (ProbabilityTheory.binomial n p).real
        {i : ℕ | (i : ℝ) ≤ (n : ℝ) * ((p : ℝ) - τ)}
      = ν.real {q : ℕ → Prop | ((({i | q i}).ncard : ℝ)) ≤ (n : ℝ) * ((p : ℝ) - τ)} := by
    rw [binomial_eq_map_bits, measureReal_def, Measure.map_apply hmapmeas hAmeas,
      ← measureReal_def]
    rfl
  have hae : ν.real {q : ℕ → Prop | ((({i | q i}).ncard : ℝ)) ≤ (n : ℝ) * ((p : ℝ) - τ)}
      = ν.real {q : ℕ → Prop | ∑ i ∈ Finset.range n, X i q ≤ (n : ℝ) * ((p : ℝ) - τ)} := by
    refine measureReal_congr ?_
    filter_upwards [ae_bits_lt n p] with q hq
    show ((({i | q i}).ncard : ℝ) ≤ (n : ℝ) * ((p : ℝ) - τ))
      = (∑ i ∈ Finset.range n, X i q ≤ (n : ℝ) * ((p : ℝ) - τ))
    rw [ncard_eq_sum_bits hq]
  rw [hpre, hae]
  exact hmain

open scoped Classical in
/-- The binomial mass of a finite set of counts is the sum of its terms. -/
lemma binomial_real_finset (N : ℕ) (p : I) (s : Finset ℕ) :
    (ProbabilityTheory.binomial N p).real ↑s
      = ∑ i ∈ s, ((N.choose i : ℝ) * (p : ℝ) ^ i * (1 - (p : ℝ)) ^ (N - i)) := by
  classical
  have hcov : (↑s : Set ℕ) = ⋃ i ∈ s, ({i} : Set ℕ) := by
    ext i; simp
  rw [hcov, measureReal_biUnion_finset
    (fun i _ j _ hij => by simpa using hij) (fun i _ => measurableSet_singleton i)]
  exact Finset.sum_congr rfl (fun i _ => ProbabilityTheory.binomial_real_singleton N i p)

end OrthoDFA
