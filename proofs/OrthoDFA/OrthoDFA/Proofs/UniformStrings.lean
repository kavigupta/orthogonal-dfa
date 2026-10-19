import OrthoDFA.CoveredStart
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

lemma prefixMax_uniformStrings_le [Nonempty α] {n k : ℕ} (hk : k ≤ n) :
    Random.prefixMax (uniformStrings (α := α) n) k ≤ 1 / (Fintype.card α : ℝ) ^ k := by
  have hA : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  unfold Random.prefixMax
  refine ciSup_le fun u => ?_
  rw [uniformStrings_real]
  have hinj : Function.Injective fun v : {v : List.Vector α n //
      FreeMonoid.ofList v.toList ∈ {x : FreeMonoid α | prefixOf x k = u}} =>
      (v.1.drop k : List.Vector α (n - k)) := by
    rintro ⟨v, hv⟩ ⟨w, hw⟩ h
    have hv' : v.toList.take k = u.toList := by
      have := congrArg FreeMonoid.toList hv
      simpa [prefixOf] using this
    have hw' : w.toList.take k = u.toList := by
      have := congrArg FreeMonoid.toList hw
      simpa [prefixOf] using this
    have hd : v.toList.drop k = w.toList.drop k := by
      have := congrArg List.Vector.toList h
      simpa using this
    refine Subtype.ext (List.Vector.toList_injective ?_)
    rw [← List.take_append_drop k v.toList, ← List.take_append_drop k w.toList, hv', hw', hd]
  have hc := Nat.card_le_card_of_injective _ hinj
  rw [Nat.card_eq_fintype_card (α := List.Vector α (n - k)), card_vector] at hc
  have hc' : (Nat.card {v : List.Vector α n // FreeMonoid.ofList v.toList ∈
      {x : FreeMonoid α | prefixOf x k = u}} : ℝ) ≤ (Fintype.card α : ℝ) ^ (n - k) := by
    exact_mod_cast hc
  have hpow : (Fintype.card α : ℝ) ^ n
      = (Fintype.card α : ℝ) ^ k * (Fintype.card α : ℝ) ^ (n - k) := by
    rw [← pow_add, Nat.add_sub_cancel' hk]
  rw [div_le_div_iff₀ (by positivity) (by positivity), hpow, one_mul]
  calc _ * (Fintype.card α : ℝ) ^ k ≤ (Fintype.card α : ℝ) ^ (n - k) * (Fintype.card α : ℝ) ^ k :=
        mul_le_mul_of_nonneg_right hc' (by positivity)
    _ = _ := mul_comm _ _

end Uniform

end OrthoDFA

end
