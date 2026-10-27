import OrthoDFA.LowLevelClaims.CoveredStart
import OrthoDFA.Proofs.RandomRisk

/-! # Uniform strings of one length: their mass, and their prefixes' -/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

section Uniform

variable {α : Type*} [Fintype α]

omit [Fintype α] in
lemma ofList_toList_injective (n : ℕ) :
    Function.Injective fun v : List.Vector α n => FreeMonoid.ofList v.toList :=
  FreeMonoid.ofList.injective.comp List.Vector.toList_injective

lemma uniformStrings_apply (n : ℕ) (S : Set (FreeMonoid α)) :
    uniformStrings n S = ((Fintype.card α : ℝ≥0∞) ^ n)⁻¹
      * (Nat.card {v : List.Vector α n // FreeMonoid.ofList v.toList ∈ S} : ℝ≥0∞) := by
  classical
  unfold uniformStrings
  rw [Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply, smul_eq_mul]
  congr 1
  simp only [Measure.dirac_apply, Set.indicator_apply, Pi.one_apply]
  rw [Finset.sum_boole, Nat.card_eq_fintype_card, Fintype.card_subtype]

lemma uniformStrings_real (n : ℕ) (S : Set (FreeMonoid α)) :
    (uniformStrings n).real S = (Nat.card {v : List.Vector α n // FreeMonoid.ofList v.toList ∈ S}
      : ℝ) / (Fintype.card α : ℝ) ^ n := by
  rw [measureReal_def, uniformStrings_apply, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_pow, ENNReal.toReal_natCast, ENNReal.toReal_natCast, div_eq_inv_mul]

instance [Nonempty α] (n : ℕ) : IsProbabilityMeasure (uniformStrings (α := α) n) := by
  constructor
  rw [uniformStrings_apply]
  have h : Nat.card {v : List.Vector α n // FreeMonoid.ofList v.toList ∈ (Set.univ : Set _)}
      = Fintype.card α ^ n := by
    rw [Nat.card_congr (Equiv.subtypeUnivEquiv fun _ => Set.mem_univ _), Nat.card_eq_fintype_card,
      card_vector]
  rw [h, Nat.cast_pow]
  exact ENNReal.inv_mul_cancel (pow_ne_zero _ (by simp)) (by simp)

lemma uniformStrings_length (n : ℕ) :
    uniformStrings (α := α) n {v | v.toList.length ≠ n} = 0 := by
  rw [uniformStrings_apply]
  have : IsEmpty {v : List.Vector α n // FreeMonoid.ofList v.toList ∈
      {v : FreeMonoid α | v.toList.length ≠ n}} :=
    ⟨fun ⟨v, hv⟩ => hv (by simp)⟩
  rw [Nat.card_of_isEmpty, Nat.cast_zero, mul_zero]

lemma uniformStrings_singleton_le (n : ℕ) (t : FreeMonoid α) :
    (uniformStrings n).real {t} ≤ 1 / (Fintype.card α : ℝ) ^ n := by
  rw [uniformStrings_real]
  have : Nat.card {v : List.Vector α n // FreeMonoid.ofList v.toList ∈ ({t} : Set _)} ≤ 1 := by
    refine Finite.card_le_one_iff_subsingleton.mpr ⟨fun ⟨v, hv⟩ ⟨w, hw⟩ => ?_⟩
    exact Subtype.ext (ofList_toList_injective n (hv.trans hw.symm))
  exact div_le_div_of_nonneg_right (by exact_mod_cast this) (by positivity)

end Uniform

end OrthoDFA

end
