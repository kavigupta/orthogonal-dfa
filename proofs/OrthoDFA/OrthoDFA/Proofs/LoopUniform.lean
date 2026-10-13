import OrthoDFA.Proofs.LoopSimple

/-!
# The loop on uniform probes and suffixes

`cap_bound` turns `L ≥ L₀` into the probes' prefix mass at `L − 1` being under the simple claim's
`cap · ρ / 4`: the claim's cap is its first term, and `4/(cap ρ) ≤ (L + 1)⁷ e^Λ` with `Λ` free of
`L`, so `L ≥ L₀` gives `(L − 1) log |Σ| ≥ 7 log (L + 1) + Λ`.
-/

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

lemma collisionMass_uniformStrings_le [Nonempty α] (n : ℕ) :
    collisionMass (uniformStrings (α := α) n) ≤ 1 / (Fintype.card α : ℝ) ^ n := by
  have hq : (0 : ℝ) ≤ 1 / (Fintype.card α : ℝ) ^ n := by positivity
  unfold collisionMass
  refine tsum_le_of_sum_le' hq fun s => ?_
  calc ∑ t ∈ s, (uniformStrings n).real {t} ^ 2
        ≤ ∑ t ∈ s, 1 / (Fintype.card α : ℝ) ^ n * (uniformStrings n).real {t} :=
        Finset.sum_le_sum fun t _ => by
          rw [sq]
          exact mul_le_mul_of_nonneg_right (uniformStrings_singleton_le n t) measureReal_nonneg
    _ = 1 / (Fintype.card α : ℝ) ^ n * (uniformStrings n).real (s : Set (FreeMonoid α)) := by
        rw [← Finset.mul_sum, sum_measureReal_singleton]
    _ ≤ 1 / (Fintype.card α : ℝ) ^ n * 1 := mul_le_mul_of_nonneg_left measureReal_le_one hq
    _ = 1 / (Fintype.card α : ℝ) ^ n := mul_one _

lemma prefixMax_uniformStrings_le [Nonempty α] {n k : ℕ} (hk : k ≤ n) :
    prefixMax (uniformStrings (α := α) n) k ≤ 1 / (Fintype.card α : ℝ) ^ k := by
  have hA : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  unfold prefixMax
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

section Cap

lemma seven_log_le {A L : ℝ} (hA : 2 ≤ A) (hL : 0 < L) :
    7 * Real.log L ≤ 28 + L * Real.log A / 8 := by
  have hl2 := Real.log_two_gt_d9
  have hlA : Real.log 2 ≤ Real.log A := Real.log_le_log (by norm_num) hA
  have hlA0 : 0 < Real.log A := by linarith
  have hx : 0 < L * Real.log A / 56 := by positivity
  have h1 := Real.log_le_sub_one_of_pos hx
  rw [Real.log_div (by positivity) (by norm_num), Real.log_mul hL.ne' hlA0.ne'] at h1
  have h2 : Real.log (1 / 2) ≤ Real.log (Real.log A) := Real.log_le_log (by norm_num) (by linarith)
  rw [one_div, Real.log_inv] at h2
  have h3 : Real.log 56 + Real.log 2 ≤ 5 := by
    rw [← Real.log_mul (by norm_num) (by norm_num), Real.log_le_iff_le_exp (by norm_num)]
    have he := Real.exp_one_gt_d9
    have : Real.exp 5 = Real.exp 1 ^ 5 := by rw [← Real.exp_nat_mul]; norm_num
    rw [this]
    have h27 : (2.7 : ℝ) ≤ Real.exp 1 := by linarith
    nlinarith [pow_le_pow_left₀ (by norm_num : (0 : ℝ) ≤ 2.7) h27 5]
  nlinarith

set_option maxHeartbeats 1000000 in -- many small nonlinear steps
/-- The simple claim's cap is its first term, and is large once `L ≥ L₀`. -/
lemma cap_bound {Q A L η₀ εcov εd δ pAP δ' ι ρ s Θ P₀ R₀ : ℝ} {n : ℕ}
    (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hLn : L = n + 1) (hη0 : 0 ≤ η₀) (hη : η₀ < 1 / 2)
    (hε : 0 < εcov) (hε1 : εcov ≤ 1) (hεd : 0 < εd) (hεd1 : εd ≤ 1) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hp : 0 < pAP) (hp1 : pAP ≤ 1)
    (hs : s = 1 / 2 - η₀) (hδ' : δ' = δ / (3 * (Q + 1)))
    (hι : ι = min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
    (hρ : ρ = εd / (256 * (Q + 1) * (Q * A + 1)))
    (hΘ : Θ = 1 / s + 8 / εcov + 3072 * Q * (Q + 1))
    (hP₀ : P₀ = 128 * (2 * Θ + 8 * (Q + 1) / ρ) / (s ^ 2 * pAP)
      + 16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2)
    (hR₀ : R₀ = 6000 * (Q + 2) ^ 2 * Θ ^ 3
      * Real.log ((Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * pAP)) / s ^ 4)
    (hL₀ : 2 + 8 * (Real.log (512 * (Q + 5) ^ 3 * (R₀ ^ 2 + P₀ ^ 2 + 1) / (δ' * ρ)) + 28)
      / (7 * Real.log A) ≤ L) :
    let cLog := 8 * (L + 1) * (Q + 1) / ρ
    let PB := capPB η₀ ι (εcov / 2) δ' pAP cLog (Q + 2)
    let RB := capRB η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog (Q + 2)
    let cap := min (capFloor η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog ι (Q + 2))
      (min (1 / (3 * (Q + 2) * (RB + 1) * (L + 1)))
        (δ' / ((Q + 2) * (PB ^ 2 + 16 * (Q + 2) ^ 2 * RB ^ 2 + 1))))
    1 / A ^ n ≤ cap * ρ / 4 ∧ 24 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * cap ≤ 1 := by
  intro cLog PBc RBc cap
  -- the problem's sizes
  have hs0 : 0 < s := by rw [hs]; linarith
  have hs1 : s ≤ 1 / 2 := by rw [hs]; linarith
  have hA0 : 0 < A := by linarith
  have hδ'0 : 0 < δ' := by rw [hδ']; positivity
  have hδ'1 : δ' ≤ 1 := by rw [hδ', div_le_one (by positivity)]; linarith
  have hρ0 : 0 < ρ := by rw [hρ]; positivity
  have hQA : A ≤ (Q + 1) * (Q * A + 1) := by
    have h1 : A ≤ Q * A := le_mul_of_one_le_left hA0.le hQ
    have h2 : Q * A + 1 ≤ (Q + 1) * (Q * A + 1) :=
      le_mul_of_one_le_left (by positivity) (by linarith)
    linarith
  have hρ1 : ρ ≤ 1 := by rw [hρ, div_le_one (by positivity)]; linarith
  have hAρ : A ≤ 1 / ρ := by
    rw [hρ, one_div_div, le_div_iff₀ hεd]
    have : A * εd ≤ A := mul_le_of_le_one_right hA0.le hεd1
    linarith
  have hL1 : 1 ≤ L := by rw [hLn]; linarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]
  have hℓ0 : 0 < L + 1 := by linarith
  have h8ε : 0 < 8 / εcov := by positivity
  have h1s : 0 < 1 / s := by positivity
  have hQQ : 3072 ≤ 3072 * Q * (Q + 1) := by nlinarith only [hQ]
  have hΘs : 1 / s ≤ Θ := by rw [hΘ]; linarith
  have hΘε : 8 / εcov ≤ Θ := by rw [hΘ]; linarith
  have hΘQ : 3072 * Q * (Q + 1) ≤ Θ := by rw [hΘ]; linarith
  have hΘ1 : 3072 ≤ Θ := by linarith
  have hΘ0 : 0 < Θ := by linarith
  -- the band's far side, `τ`, is at least `1/(Θ (L + 1))`
  obtain ⟨τ, hτ⟩ : ∃ τ, τ = min (εcov / 2) (min s ι) := ⟨_, rfl⟩
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hι1 : ι ≤ 1 := by rw [hι]; exact (min_le_left _ _).trans (by linarith)
  have hτι : τ ≤ ι := by rw [hτ]; exact (min_le_right _ _).trans (min_le_right _ _)
  have hτ0 : 0 < τ := by rw [hτ]; exact lt_min (by positivity) (lt_min hs0 hι0)
  have hτ1 : τ ≤ 1 := hτι.trans hι1
  have hΘℓ : 0 < Θ * (L + 1) := by positivity
  have hτΘ : 1 / τ ≤ Θ * (L + 1) := by
    rw [div_le_iff₀ hτ0, ← div_le_iff₀' hΘℓ]
    have h1 : 1 / (Θ * (L + 1)) ≤ 1 / Θ :=
      one_div_le_one_div_of_le hΘ0 (le_mul_of_one_le_right hΘ0.le (by linarith))
    have h2 : 1 / Θ ≤ εcov / 8 := by
      rw [div_le_div_iff₀ hΘ0 (by norm_num)]
      rw [div_le_iff₀ hε] at hΘε; linarith
    have h3 : 1 / Θ ≤ s := by
      rw [div_le_iff₀ hΘ0]; rw [div_le_iff₀ hs0] at hΘs; linarith
    have h4 : 1 / (Θ * (L + 1)) ≤ 1 / (3072 * (L + 1) * (Q + 1) * Q) :=
      one_div_le_one_div_of_le (by positivity) (by nlinarith only [hΘQ, hℓ0])
    rw [hτ, hι]
    exact le_min (by linarith) (le_min (by linarith) (le_min (by linarith) h4))
  -- the suffix pool's bound `PB`
  obtain ⟨PB, hPB⟩ : ∃ PB, PB = 128 * (Real.log (2 / τ) + 8 * (L + 1) * (Q + 1) / ρ) / (s ^ 2 * pAP)
      + 16 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2 := ⟨_, rfl⟩
  have hJ2 : Q + 2 + 2 = Q + 4 := by ring
  rw [hJ2] at hPB
  have hlτ0 : 0 ≤ Real.log (2 / τ) := Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith)
  have hlτ : Real.log (2 / τ) ≤ 2 * (Θ * (L + 1)) := by
    have h1 := Real.log_le_sub_one_of_pos (by positivity : 0 < 2 / τ)
    have h2 : 2 / τ = 2 * (1 / τ) := by ring
    rw [h2] at h1 ⊢
    linarith
  have hlJ0 : 0 ≤ Real.log ((Q + 4) / (δ' / 2)) :=
    Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; linarith)
  have hsp : s ^ 2 * pAP ≤ 1 / 4 := by
    have : s ^ 2 ≤ 1 / 4 := by nlinarith only [hs0, hs1]
    nlinarith only [this, hp1, sq_nonneg s]
  have hsp0 : 0 < s ^ 2 * pAP := by positivity
  have hc : 1 / ρ ≤ 8 * (L + 1) * (Q + 1) / ρ := by
    rw [div_le_div_iff_of_pos_right hρ0]; nlinarith only [hL1, hQ]
  have hPBle : PB ≤ (L + 1) * P₀ := by
    have e : (L + 1) * P₀ = 128 * (2 * (Θ * (L + 1)) + 8 * (L + 1) * (Q + 1) / ρ) / (s ^ 2 * pAP)
        + (L + 1) * (16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) := by rw [hP₀]; ring
    rw [e, hPB]
    have h1 : 128 * (Real.log (2 / τ) + 8 * (L + 1) * (Q + 1) / ρ) / (s ^ 2 * pAP)
        ≤ 128 * (2 * (Θ * (L + 1)) + 8 * (L + 1) * (Q + 1) / ρ) / (s ^ 2 * pAP) :=
      div_le_div_of_nonneg_right (by linarith) hsp0.le
    have h2 : 16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2
        ≤ (L + 1) * (16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2) :=
      le_mul_of_one_le_left (by positivity) (by linarith)
    linarith
  have hPBc : 8 * (L + 1) * (Q + 1) / ρ ≤ PB := by
    have h1 : 8 * (L + 1) * (Q + 1) / ρ ≤ 128 * (Real.log (2 / τ) + 8 * (L + 1) * (Q + 1) / ρ)
        / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]
      have : 0 ≤ 8 * (L + 1) * (Q + 1) / ρ := by positivity
      nlinarith only [this, hsp, hlτ0]
    have h2 : 0 ≤ 16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
    rw [hPB]; linarith
  have hPBL : L + 1 ≤ PB := by
    refine le_trans ?_ hPBc
    rw [le_div_iff₀ hρ0]
    nlinarith only [hρ1, hρ0, hℓ0, hQ]
  have hPBA : A ≤ PB := by
    have h1 : 8 * (L + 1) * (Q + 1) / ρ ≤ 128 * (Real.log (2 / τ) + 8 * (L + 1) * (Q + 1) / ρ)
        / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]
      have : 0 ≤ 8 * (L + 1) * (Q + 1) / ρ := by positivity
      nlinarith only [this, hsp, hlτ0]
    have h2 : 0 ≤ 16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
    rw [hPB]; linarith
  have hPB1 : 1 ≤ PB := by linarith
  have hP₀1 : 1 ≤ P₀ := by
    have hx : 0 ≤ 2 * Θ + 8 * (Q + 1) / ρ := by positivity
    have h1 : 2 * Θ + 8 * (Q + 1) / ρ ≤ 128 * (2 * Θ + 8 * (Q + 1) / ρ) / (s ^ 2 * pAP) := by
      rw [le_div_iff₀ hsp0]; nlinarith only [hx, hsp]
    have h2 : 0 ≤ 16 * Real.log ((Q + 4) / (δ' / 2)) / pAP ^ 2 := by positivity
    have h3 : 0 ≤ 8 * (Q + 1) / ρ := by positivity
    rw [hP₀]; linarith
  -- the prefix count's bound `RB`
  obtain ⟨RB, hRB⟩ : ∃ RB, RB = 6000 * (Q + 2) ^ 2
      * Real.log ((Q + 2 + 2) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) / (s ^ 4 * τ ^ 3) := ⟨_, rfl⟩
  obtain ⟨Y₀, hY₀⟩ : ∃ Y₀, Y₀ = (Q + 4) * P₀ * Θ / (δ' / 2 * (1 / 4) * pAP) := ⟨_, rfl⟩
  have hD0 : 0 < δ' / 2 * (1 / 4) * pAP := by positivity
  have hD1 : δ' / 2 * (1 / 4) * pAP ≤ 1 := by nlinarith only [hδ'0, hδ'1, hp, hp1]
  have hY5 : 5 ≤ (Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ) := by
    rw [le_div_iff₀ (by positivity)]
    have : δ' / 2 * (1 / 4) * pAP * τ ≤ 1 := mul_le_one₀ hD1 hτ0.le hτ1
    nlinarith only [this, hQ, hPB1]
  have hYle : (Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ) ≤ (L + 1) ^ 2 * Y₀ := by
    have e : (Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)
        = (Q + 4) * PB * (1 / τ) / (δ' / 2 * (1 / 4) * pAP) := by field_simp
    have e2 : (L + 1) ^ 2 * Y₀ = (Q + 4) * ((L + 1) * P₀) * (Θ * (L + 1))
        / (δ' / 2 * (1 / 4) * pAP) := by rw [hY₀]; ring
    rw [e, e2]
    refine div_le_div_of_nonneg_right ?_ hD0.le
    have h1 : 0 ≤ (Q + 4) * PB := by positivity
    calc (Q + 4) * PB * (1 / τ) ≤ (Q + 4) * PB * (Θ * (L + 1)) :=
          mul_le_mul_of_nonneg_left hτΘ h1
      _ ≤ (Q + 4) * ((L + 1) * P₀) * (Θ * (L + 1)) := by
          gcongr
  have hY₀4 : 4 ≤ Real.log Y₀ := by
    rw [Real.le_log_iff_exp_le (by rw [hY₀]; positivity)]
    have he := Real.exp_one_lt_d9
    have : Real.exp 4 = Real.exp 1 ^ 4 := by rw [← Real.exp_nat_mul]; norm_num
    rw [this]
    have h1 : Real.exp 1 ^ 4 ≤ 3 ^ 4 := pow_le_pow_left₀ (Real.exp_pos 1).le (by linarith) 4
    have h2 : (Q + 4) * P₀ * Θ ≤ Y₀ := by
      rw [hY₀, le_div_iff₀ hD0]
      have : 0 ≤ (Q + 4) * P₀ * Θ := by positivity
      nlinarith only [this, hD1]
    have h3 : 5 * 1 * 3072 ≤ (Q + 4) * P₀ * Θ := by
      have := mul_le_mul (mul_le_mul (by linarith : (5 : ℝ) ≤ Q + 4) hP₀1 (by norm_num)
        (by linarith)) hΘ1 (by norm_num) (by positivity)
      linarith
    linarith
  have hlY1 : 1 ≤ Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) := by
    rw [Real.le_log_iff_exp_le (by linarith)]
    have := Real.exp_one_lt_d9
    linarith
  have hkey : Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) ^ 2
      ≤ (L + 1) * Real.log Y₀ ^ 2 := by
    set u := Real.log (L + 1) with hu
    set w := Real.log Y₀ with hw
    have hu0 : 0 ≤ u := Real.log_nonneg (by linarith)
    have h1 : Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) ≤ w + 2 * u := by
      have := Real.log_le_log (by linarith) hYle
      rw [Real.log_mul (by positivity) (by rw [hY₀]; positivity), Real.log_pow, Nat.cast_ofNat,
        ← hu, ← hw] at this
      linarith
    have h2 : w + 2 * u ≤ w * (1 + u / 2) := by nlinarith only [hu0, hY₀4]
    have h3 : 1 + u / 2 ≤ Real.exp (u / 2) := by linarith [Real.add_one_le_exp (u / 2)]
    have h4 : Real.exp (u / 2) ^ 2 = L + 1 := by
      have e : ((2 : ℕ) : ℝ) * (u / 2) = Real.log (L + 1) := by rw [hu]; push_cast; ring
      rw [← Real.exp_nat_mul, e, Real.exp_log hℓ0]
    calc Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) ^ 2 ≤ (w * (1 + u / 2)) ^ 2 :=
          pow_le_pow_left₀ (by linarith) (h1.trans h2) 2
      _ ≤ (w * Real.exp (u / 2)) ^ 2 :=
          pow_le_pow_left₀ (by nlinarith only [hu0, hY₀4])
            (mul_le_mul_of_nonneg_left h3 (by linarith)) 2
      _ = (L + 1) * w ^ 2 := by rw [mul_pow, h4]; ring
  rw [hJ2] at hRB
  have hRBle : RB ^ 2 ≤ (L + 1) ^ 7 * R₀ ^ 2 := by
    have hK : 0 ≤ 6000 * (Q + 2) ^ 2 / s ^ 4 := by positivity
    have e1 : RB = 6000 * (Q + 2) ^ 2 / s ^ 4 * (1 / τ) ^ 3
        * Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) := by rw [hRB]; field_simp
    have e2 : (L + 1) ^ 7 * R₀ ^ 2 = (6000 * (Q + 2) ^ 2 / s ^ 4 * (Θ * (L + 1)) ^ 3) ^ 2
        * ((L + 1) * Real.log Y₀ ^ 2) := by rw [hR₀, ← hY₀]; ring
    rw [e1, e2, mul_pow]
    have h1 : (1 / τ) ^ 3 ≤ (Θ * (L + 1)) ^ 3 := pow_le_pow_left₀ (by positivity) hτΘ 3
    refine mul_le_mul (pow_le_pow_left₀ (by positivity) ?_ 2) hkey (by positivity) (by positivity)
    exact mul_le_mul_of_nonneg_left h1 hK
  have hPBsq : PB ^ 2 ≤ (L + 1) ^ 7 * P₀ ^ 2 := by
    have h1 : PB ^ 2 ≤ ((L + 1) * P₀) ^ 2 := pow_le_pow_left₀ (by linarith) hPBle 2
    have h2 : (L + 1) ^ 2 ≤ (L + 1) ^ 7 := pow_le_pow_right₀ (by linarith) (by norm_num)
    rw [mul_pow] at h1
    nlinarith only [h1, h2, sq_nonneg P₀]
  have hℓ7 : 1 ≤ (L + 1) ^ 7 := one_le_pow₀ (by linarith)
  have hsum : RB ^ 2 + PB ^ 2 + 1 ≤ (L + 1) ^ 7 * (R₀ ^ 2 + P₀ ^ 2 + 1) := by
    nlinarith only [hRBle, hPBsq, hℓ7]
  -- `RB ≥ 1/ι²`, so the first term is the least
  have hN : 1 ≤ 6000 * (Q + 2) ^ 2 * Real.log ((Q + 4) * PB / (δ' / 2 * (1 / 4) * pAP * τ)) := by
    have : (1 : ℝ) ≤ (Q + 2) ^ 2 := by nlinarith only [hQ]
    nlinarith only [this, hlY1]
  have hιRB : 1 ≤ ι ^ 2 * RB := by
    rw [hRB, ← mul_div_assoc, le_div_iff₀ (by positivity), one_mul]
    have h1 : s ^ 4 ≤ 1 := pow_le_one₀ hs0.le (by linarith)
    have h2 : τ ^ 3 ≤ ι ^ 2 := by
      have : τ ^ 3 ≤ τ ^ 2 := pow_le_pow_of_le_one hτ0.le hτ1 (by norm_num)
      nlinarith only [this, pow_le_pow_left₀ hτ0.le hτι 2]
    have h3 : 0 ≤ τ ^ 3 := by positivity
    calc s ^ 4 * τ ^ 3 ≤ 1 * τ ^ 3 := mul_le_mul_of_nonneg_right h1 h3
      _ ≤ ι ^ 2 * 1 := by linarith
      _ ≤ _ := mul_le_mul_of_nonneg_left hN (by positivity)
  have hRB1 : 1 ≤ RB := by
    have : ι ^ 2 ≤ 1 := pow_le_one₀ hι0.le hι1
    have h0 : 0 < ι ^ 2 := by positivity
    by_contra h
    nlinarith only [this, h0, not_le.mp h, hιRB]
  have hW : 0 < RB ^ 2 + PB ^ 2 + 1 := by positivity
  have hcap : capFloor η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog ι (Q + 2)
      = δ' / 2 / (64 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) := by
    unfold capFloor
    dsimp only
    rw [← hs, ← hτ, hJ2, ← hPB, ← hRB]
    have hRB0 : 0 ≤ RB := by linarith
    have key : ∀ C : ℝ, 0 ≤ C → C * RB ≤ ι ^ 2 * (C * (RB ^ 2 + PB ^ 2 + 1)) := by
      intro C hC
      have h1 : C * RB ≤ C * RB * (ι ^ 2 * RB) := le_mul_of_one_le_right (mul_nonneg hC hRB0) hιRB
      have h2 : C * RB * (ι ^ 2 * RB) = ι ^ 2 * (C * RB ^ 2) := by ring
      have h3 : ι ^ 2 * (C * RB ^ 2) ≤ ι ^ 2 * (C * (RB ^ 2 + PB ^ 2 + 1)) := by
        gcongr; nlinarith only [sq_nonneg PB]
      linarith
    have hq : 64 * (Q + 3) ≤ 128 * (Q + 2 + 3) ^ 3 := by nlinarith only [hQ, sq_nonneg (Q + 5)]
    have hq' := mul_le_mul_of_nonneg_right hq hRB0
    refine min_eq_left (le_min ?_ ?_)
    · rw [div_div, div_le_div_iff₀ (by positivity) (by positivity)]
      calc δ' * (64 * ((Q + 2) * RB + 1)) ≤ 1 * (64 * ((Q + 2) * RB + 1)) :=
            mul_le_mul_of_nonneg_right hδ'1 (by positivity)
        _ ≤ 128 * (Q + 2 + 3) ^ 3 * RB := by nlinarith only [hq', hRB1, hQ]
        _ ≤ ι ^ 2 * (128 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) := key _ (by positivity)
        _ = ι ^ 2 * (2 * (64 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1))) := by ring
    · rw [div_div, div_le_div_iff₀ (by positivity) (by positivity)]
      have h1 : 16 * (RB + 2) * (Q + 2 + 1) ≤ 128 * (Q + 2 + 3) ^ 3 * RB := by
        nlinarith only [hq', hRB1, hQ]
      calc δ' * (16 * (RB + 2) * (Q + 2 + 1)) ≤ δ' * (128 * (Q + 2 + 3) ^ 3 * RB) :=
            mul_le_mul_of_nonneg_left h1 hδ'0.le
        _ ≤ δ' * (ι ^ 2 * (128 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1))) :=
            mul_le_mul_of_nonneg_left (key _ (by positivity)) hδ'0.le
        _ = δ' * ι ^ 2 * (2 * (64 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1))) := by ring
  have hJ3 : Q + 2 + 3 = Q + 5 := by ring
  -- the cap's last two terms are larger than its first
  have hcPB : PBc = PB := by
    show capPB η₀ ι (εcov / 2) δ' pAP cLog (Q + 2) = PB
    unfold capPB
    simp only [cLog]
    rw [← hs, ← hτ, hJ2, hPB]
  have hcRB : RBc = RB := by
    show capRB η₀ ι (εcov / 2) (1 / 4) δ' pAP cLog (Q + 2) = RB
    unfold capRB
    rw [show capPB η₀ ι (εcov / 2) δ' pAP cLog (Q + 2) = PB from hcPB, ← hs, ← hτ, hJ2, hRB]
  have hT4 : δ' / 2 / (64 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1))
      ≤ 1 / (3 * (Q + 2) * (RB + 1) * (L + 1)) := by
    rw [div_div, div_le_div_iff₀ (by positivity) (by positivity), one_mul]
    have hRB0 : 0 ≤ RB := by linarith
    have h1 : (RB + 1) * (L + 1) ≤ RB ^ 2 + PB ^ 2 + 1 := by
      have : (RB + 1) * (L + 1) ≤ (RB + 1) * PB := mul_le_mul_of_nonneg_left hPBL (by linarith)
      nlinarith only [this, sq_nonneg (RB + 1 - PB), sq_nonneg RB, hRB0]
    have h2 : 3 * (Q + 2) * δ' ≤ 128 * (Q + 2 + 3) ^ 3 := by
      nlinarith only [hQ, hδ'1, hδ'0, sq_nonneg (Q + 5)]
    have h3 := mul_le_mul h2 h1 (by positivity) (by positivity)
    nlinarith only [h3]
  have hT5 : δ' / 2 / (64 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1))
      ≤ δ' / ((Q + 2) * (PB ^ 2 + 16 * (Q + 2) ^ 2 * RB ^ 2 + 1)) := by
    rw [div_div]
    refine div_le_div_of_nonneg_left hδ'0.le (by positivity) ?_
    have h1 : (Q + 2) * (PB ^ 2 + 16 * (Q + 2) ^ 2 * RB ^ 2 + 1)
        ≤ 16 * (Q + 2) ^ 3 * (RB ^ 2 + PB ^ 2 + 1) := by
      have hq : (1 : ℝ) ≤ (Q + 2) ^ 2 := by nlinarith only [hQ]
      have hq3 : Q + 2 ≤ 16 * (Q + 2) ^ 3 := by nlinarith only [hq, hQ]
      have e : (Q + 2) * (PB ^ 2 + 16 * (Q + 2) ^ 2 * RB ^ 2 + 1)
          = (Q + 2) * PB ^ 2 + 16 * (Q + 2) ^ 3 * RB ^ 2 + (Q + 2) := by ring
      have e' : 16 * (Q + 2) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)
          = 16 * (Q + 2) ^ 3 * PB ^ 2 + 16 * (Q + 2) ^ 3 * RB ^ 2 + 16 * (Q + 2) ^ 3 := by ring
      rw [e, e']
      have := mul_le_mul_of_nonneg_right hq3 (sq_nonneg PB)
      linarith
    have h2 : 16 * (Q + 2) ^ 3 ≤ 64 * (Q + 2 + 3) ^ 3 * 2 := by
      nlinarith only [hQ, sq_nonneg (Q + 5), sq_nonneg (Q + 2)]
    nlinarith only [h1, h2, hW]
  have hcapE : cap = δ' / 2 / (64 * (Q + 2 + 3) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) := by
    show min _ (min (1 / (3 * (Q + 2) * (RBc + 1) * (L + 1)))
      (δ' / ((Q + 2) * (PBc ^ 2 + 16 * (Q + 2) ^ 2 * RBc ^ 2 + 1)))) = _
    rw [hcap, hcPB, hcRB]
    exact min_eq_left (le_min hT4 hT5)
  rw [hcapE, hJ3]
  have hcap0 : 0 < δ' / 2 / (64 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) := by positivity
  refine ⟨?_, ?_⟩
  · -- `4/(cap ρ) ≤ (L + 1)⁷ e^Λ ≤ A^n`
    obtain ⟨E, hE⟩ : ∃ E, E = 512 * (Q + 5) ^ 3 * (R₀ ^ 2 + P₀ ^ 2 + 1) / (δ' * ρ) := ⟨_, rfl⟩
    have hE0 : 0 < E := by rw [hE]; positivity
    have h1 : 1 / (δ' / 2 / (64 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) * ρ / 4)
        ≤ (L + 1) ^ 7 * E := by
      have e : 1 / (δ' / 2 / (64 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) * ρ / 4)
          = 512 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1) / (δ' * ρ) := by field_simp; ring
      have e2 : (L + 1) ^ 7 * E = 512 * (Q + 5) ^ 3 * ((L + 1) ^ 7 * (R₀ ^ 2 + P₀ ^ 2 + 1))
          / (δ' * ρ) := by rw [hE]; ring
      rw [e, e2]
      exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hsum (by positivity))
        (by positivity)
    have hlA0 : 0 < Real.log A := Real.log_pos (by linarith)
    have h2 : 7 * Real.log (L + 1) ≤ 28 + (L + 1) * Real.log A / 8 := seven_log_le hA hℓ0
    have h3 : Real.log E + 28 ≤ (L - 2) * (7 * Real.log A) / 8 := by
      rw [← hE] at hL₀
      have := (le_sub_iff_add_le'.mpr hL₀)
      rw [div_le_iff₀ (by positivity)] at this
      linarith
    have h4 : (L + 1) ^ 7 * E ≤ A ^ n := by
      rw [← Real.log_le_log_iff (by positivity) (by positivity), Real.log_mul (by positivity)
        hE0.ne', Real.log_pow, Real.log_pow]
      have : (n : ℝ) = L - 1 := by rw [hLn]; ring
      rw [this]
      push_cast
      nlinarith only [h2, h3, hlA0]
    rw [one_div_le (by positivity) (by positivity)]
    exact h1.trans h4
  · -- the noise fields' condition
    have hlg0 : 0 < (Q + 3) * A / δ' := by positivity
    have hlg : Real.log ((Q + 3) * A / δ') ≤ (Q + 3) * A / δ' := by
      linarith [Real.log_le_sub_one_of_pos hlg0]
    have hlg1 : 0 ≤ Real.log ((Q + 3) * A / δ') :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ, hA])
    calc 24 * (Q + 2) * A * Real.log ((Q + 3) * A / δ')
          * (δ' / 2 / (64 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)))
        ≤ 24 * (Q + 2) * A * ((Q + 3) * A / δ')
          * (δ' / 2 / (64 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1))) := by gcongr
      _ = 24 * (Q + 2) * (Q + 3) * A ^ 2 / (128 * (Q + 5) ^ 3 * (RB ^ 2 + PB ^ 2 + 1)) := by
          field_simp; ring
      _ ≤ 1 := by
          rw [div_le_one (by positivity)]
          have h1 : A ^ 2 ≤ RB ^ 2 + PB ^ 2 + 1 := by
            nlinarith only [pow_le_pow_left₀ hA0.le hPBA 2, sq_nonneg RB]
          have h2 : 24 * (Q + 2) * (Q + 3) ≤ 128 * (Q + 5) ^ 3 := by
            nlinarith only [hQ, sq_nonneg (Q + 5)]
          have h4 := mul_le_mul_of_nonneg_left h1
            (by positivity : (0 : ℝ) ≤ 24 * (Q + 2) * (Q + 3))
          have h5 := mul_le_mul_of_nonneg_right h2 hW.le
          linarith

end Cap

set_option maxHeartbeats 1000000 in -- unfolding the simple claim's cap
theorem learner_loop_uniform_holds : LearnerLoopUniform := by
  intro α _ _ _ σ _ Ω _ μ _ M O P η₀ εcov εd δ pAP Q A L s δ' ρ Θ P₀ R₀ L₀ Kp X G
    hL hη hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP hA2 hD hpAPD hL₀
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPD.trans measureReal_le_one
  have hδ'0 : 0 < δ' := by show 0 < δ / (3 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (3 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]; linarith
  have hρ0 : 0 < ρ := by show 0 < εd / (256 * (Q + 1) * (Q * A + 1)); positivity
  have hρ1 : ρ ≤ 1 := by
    show εd / (256 * (Q + 1) * (Q * A + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    have : 1 ≤ (Q + 1) * (Q * A + 1) := one_le_mul_of_one_le_of_one_le (by linarith)
      (by nlinarith only [hQ1, hA])
    linarith
  -- the probes are at least 2 long
  have hL2 : 2 ≤ P.L := by
    have hE : 1 ≤ 512 * (Q + 5) ^ 3 * (R₀ ^ 2 + P₀ ^ 2 + 1) / (δ' * ρ) := by
      rw [le_div_iff₀ (by positivity), one_mul]
      have h1 : δ' * ρ ≤ 1 := mul_le_one₀ hδ'1 hρ0.le hρ1
      have h2 : 1 ≤ 512 * (Q + 5) ^ 3 * (R₀ ^ 2 + P₀ ^ 2 + 1) := by
        have : (1 : ℝ) ≤ (Q + 5) ^ 3 := one_le_pow₀ (by linarith)
        nlinarith only [this, sq_nonneg R₀, sq_nonneg P₀]
      linarith
    have hlA : 0 < Real.log A := Real.log_pos (by linarith)
    have : 0 ≤ 8 * (Real.log (512 * (Q + 5) ^ 3 * (R₀ ^ 2 + P₀ ^ 2 + 1) / (δ' * ρ)) + 28)
        / (7 * Real.log A) := by
      have := Real.log_nonneg hE
      positivity
    have h : (2 : ℝ) ≤ P.L := le_trans (by linarith) hL₀
    exact_mod_cast h
  have hLn : L = ((P.L - 1 : ℕ) : ℝ) + 1 := by
    show (P.L : ℝ) = _
    rw [Nat.cast_sub (by omega)]; ring
  obtain ⟨hcap, hnoise⟩ := cap_bound (ι := min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
    hQ1 hA hLn hη0 hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP hp1 rfl rfl rfl rfl rfl rfl rfl hL₀
  set cap := min (capFloor η₀ (min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))) (εcov / 2)
      (1 / 4) δ' pAP (8 * (L + 1) * (Q + 1) / ρ)
      (min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q))) (Q + 2))
    (min (1 / (3 * (Q + 2) * (capRB η₀ (min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
        (εcov / 2) (1 / 4) δ' pAP (8 * (L + 1) * (Q + 1) / ρ) (Q + 2) + 1) * (L + 1)))
      (δ' / ((Q + 2) * (capPB η₀ (min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
        (εcov / 2) δ' pAP (8 * (L + 1) * (Q + 1) / ρ) (Q + 2) ^ 2
        + 16 * (Q + 2) ^ 2 * capRB η₀ (min (εcov / 8) (1 / (3072 * (L + 1) * (Q + 1) * Q)))
          (εcov / 2) (1 / 4) δ' pAP (8 * (L + 1) * (Q + 1) / ρ) (Q + 2) ^ 2 + 1)))) with hcapd
  have hpow : 1 / A ^ P.L ≤ 1 / A ^ (P.L - 1) :=
    one_div_pow_le_one_div_pow_of_le (by linarith) (Nat.sub_le _ _)
  have hcap0 : 0 ≤ cap := by
    have h := (by positivity : (0 : ℝ) < 1 / A ^ (P.L - 1)).trans_le hcap
    have : 0 < cap * ρ := by linarith
    exact (pos_of_mul_pos_left this hρ0.le).le
  have hcapc : 1 / A ^ (P.L - 1) ≤ cap := by
    refine hcap.trans ?_
    have : ρ / 4 ≤ 1 := by linarith
    calc cap * ρ / 4 = cap * (ρ / 4) := by ring
      _ ≤ cap * 1 := mul_le_mul_of_nonneg_left this hcap0
      _ = cap := mul_one _
  have hpre : prefixMax P.D (P.L - 1) ≤ 1 / A ^ (P.L - 1) := by
    rw [hD]; exact prefixMax_uniformStrings_le (Nat.sub_le _ _)
  have hc0 : 0 ≤ 24 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') := by
    have : 0 ≤ Real.log ((Q + 3) * A / δ') :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ1, hA])
    have : 0 ≤ Q + 2 := by linarith
    have : 0 ≤ A := by linarith
    positivity
  obtain ⟨settings, hw, hd, hb, hmain⟩ := learner_loop_simple_holds M O P η₀ εcov εd δ pAP
    (P.L - 1) hL hη hη₀ hε hε1 hεd hεd1 hδ hδ1 hpAP (by omega) (by omega)
    (hpre.trans hcap)
    (by
      calc 96 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * prefixMax P.D (P.L - 1)
          ≤ 96 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * (cap * ρ / 4) :=
            mul_le_mul_of_nonneg_left (hpre.trans hcap) (by linarith)
        _ = (24 * (Q + 2) * A * Real.log ((Q + 3) * A / δ') * cap) * ρ := by ring
        _ ≤ 1 * ρ := mul_le_mul_of_nonneg_right hnoise hρ0.le
        _ = ρ := one_mul _)
  refine ⟨settings, hw, hd, hb, ?_⟩
  refine hmain _ inferInstance (uniformStrings_length _) hpAPD fun v => ?_
  refine (uniformStrings_singleton_le _ v).trans (le_trans ?_ hcapc)
  exact one_div_pow_le_one_div_pow_of_le (le_trans (by norm_num) hA) (by omega)

end OrthoDFA
