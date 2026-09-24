import OrthoDFA.Clustering

/-! # Facts about the oracle, the greedy's argmin, and the retry loop -/

namespace OrthoDFA

section leastLoss
variable {S : Type*} [DecidableEq S] (ℓ : S → ℝ) (cands : Finset S) (k : ℕ)

lemma leastLossSubset_mem (hk : k ≤ cands.card) :
    leastLossSubset ℓ cands k ∈ cands.powersetCard k := by
  have h : (cands.powersetCard k).Nonempty := Finset.powersetCard_nonempty.mpr hk
  rw [leastLossSubset, dif_pos h]
  exact (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose_spec.1

lemma leastLossSubset_subset (hk : k ≤ cands.card) : leastLossSubset ℓ cands k ⊆ cands :=
  (Finset.mem_powersetCard.mp (leastLossSubset_mem ℓ cands k hk)).1

lemma leastLossSubset_card (hk : k ≤ cands.card) : (leastLossSubset ℓ cands k).card = k :=
  (Finset.mem_powersetCard.mp (leastLossSubset_mem ℓ cands k hk)).2

/-- The defining property: every chosen element has loss ≤ every unchosen candidate.
Proved from minimality of the argmin by the swap `v ↦ w`. -/
lemma leastLossSubset_least (hk : k ≤ cands.card) :
    ∀ v ∈ leastLossSubset ℓ cands k, ∀ w ∈ cands, w ∉ leastLossSubset ℓ cands k →
      ℓ v ≤ ℓ w := by
  intro v hv w hw hwnot
  by_contra hlt
  push_neg at hlt
  have h : (cands.powersetCard k).Nonempty := Finset.powersetCard_nonempty.mpr hk
  have hspec :=
    (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose_spec
  have hTeq : leastLossSubset ℓ cands k
      = (Finset.exists_min_image (cands.powersetCard k) (fun T => ∑ x ∈ T, ℓ x) h).choose := by
    rw [leastLossSubset, dif_pos h]
  have hwnoterase : w ∉ (leastLossSubset ℓ cands k).erase v :=
    fun hh => hwnot (Finset.mem_of_mem_erase hh)
  set T' := insert w ((leastLossSubset ℓ cands k).erase v) with hT'
  have hT'mem : T' ∈ cands.powersetCard k := by
    rw [Finset.mem_powersetCard]
    refine ⟨?_, ?_⟩
    · rw [hT', Finset.insert_subset_iff]
      exact ⟨hw, (Finset.erase_subset v _).trans (leastLossSubset_subset ℓ cands k hk)⟩
    · have hkpos : 0 < k := by
        have hp := Finset.card_pos.mpr ⟨v, hv⟩
        rwa [leastLossSubset_card ℓ cands k hk] at hp
      rw [hT', Finset.card_insert_of_notMem hwnoterase, Finset.card_erase_of_mem hv,
        leastLossSubset_card ℓ cands k hk]
      omega
  have hmin := hspec.2 T' hT'mem
  rw [← hTeq] at hmin
  have hsum : ∑ x ∈ T', ℓ x = (∑ x ∈ leastLossSubset ℓ cands k, ℓ x) - ℓ v + ℓ w := by
    rw [hT', Finset.sum_insert hwnoterase, Finset.sum_erase_eq_sub hv]; ring
  rw [hsum] at hmin
  linarith
end leastLoss

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] (O : Oracle μ S)

/-- Bit-valuedness, by construction: an indicator is `0` or `1`. -/
lemma label_bit (w : S) : O.label w = 0 ∨ O.label w = 1 := by
  by_cases h : w ∈ O.L <;> simp [Oracle.label, Set.indicator_apply, h]

/-- Two strings get the same bit exactly when they agree on membership. -/
lemma label_eq_iff (x y : S) : O.label x = O.label y ↔ (x ∈ O.L ↔ y ∈ O.L) := by
  by_cases hx : x ∈ O.L <;> by_cases hy : y ∈ O.L <;>
    simp [Oracle.label, Set.indicator_apply, hx, hy]

/-- Accept-preservation, as membership rather than as an equality of bits. -/
lemma apSet_eq : {v : S | ∀ p, p * v ∈ O.L ↔ p ∈ O.L}
    = {v : S | ∀ p, O.label (p * v) = O.label p} :=
  Set.ext fun v => forall_congr' fun p => (O.label_eq_iff (p * v) p).symm

lemma label_meas : Measurable O.label :=
  (measurable_one.indicator O.L_meas)

/-- Whether `v` flips `p`'s acceptance:

    flip v p = ℓ(p·v) ⊕ ℓ(p) = ℓ(p·v) + ℓ(p) − 2·ℓ(p·v)·ℓ(p)

so it is `0` exactly when `v` is accept-preserving at `p`. -/
noncomputable def flip (v p : S) : ℝ :=
  O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p

/-- Derived bit-valuedness of `flip`: an XOR of two bits is a bit. -/
lemma flip_bit (v p : S) : O.flip v p = 0 ∨ O.flip v p = 1 := by
  rcases O.label_bit (p * v) with h1 | h1 <;> rcases O.label_bit p with h2 | h2 <;>
    · rw [Oracle.flip, h1, h2]; norm_num

/-- Derived boundedness: a `{0,1}` bit lies in `[0,1]` (what the Hoeffding
bounds consume). -/
lemma noise_icc (w) : ∀ᵐ ω ∂μ, O.noise w ω ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_bit w] with ω hω
  rcases hω with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num


lemma noise_int (w) : Integrable (O.noise w) μ :=
  MeasureTheory.Integrable.of_mem_Icc 0 1 (O.noise_meas w).aemeasurable (O.noise_icc w)

open scoped Classical in
/-- The noise rate at one string. -/
noncomputable def rate (w : S) : ℝ := if w ∈ O.L then O.ηIn else O.ηOut

lemma noise_mean (w : S) : μ[O.noise w] = O.rate w := by
  unfold rate
  split_ifs with h
  · exact O.noise_mean_in w h
  · exact O.noise_mean_out w h

lemma rate_nonneg (w : S) : 0 ≤ O.rate w := by
  rw [← O.noise_mean w]
  refine integral_nonneg_of_ae ?_
  filter_upwards [O.noise_icc w] with ω hω
  exact hω.1

lemma rate_le_eta (w : S) : O.rate w ≤ O.η := by
  unfold rate Oracle.η
  split_ifs
  · exact le_max_left _ _
  · exact le_max_right _ _

/-- Two strings in the same class share a rate. -/
lemma rate_eq_of_label_eq {x y : S} (h : O.label x = O.label y) : O.rate x = O.rate y := by
  have hxy := (O.label_eq_iff x y).1 h
  unfold rate
  by_cases hx : x ∈ O.L
  · rw [if_pos hx, if_pos (hxy.1 hx)]
  · rw [if_neg hx, if_neg (fun hy => hx (hxy.2 hy))]

end Oracle

section Flip
variable {S : Type*} [MeasurableSpace S] [Monoid S] [MeasurableMul S]

/-- The `Dj`-flip-mass of a suffix: the probability that `v` flips a `Dj`-drawn prefix.
This is the *distributional* quantity `good`/`bad` are defined by. -/
noncomputable def flipMass (O : Oracle μ S) (Dj : Measure S) (v : S) : ℝ :=
  ∫ p, O.flip v p ∂Dj

lemma flip_meas (O : Oracle μ S) (v : S) : Measurable (fun p => O.flip v p) := by
  have h1 : Measurable (fun p : S => O.label (p * v)) :=
    O.label_meas.comp (measurable_mul_const v)
  show Measurable (fun p => O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
  exact (h1.add O.label_meas).sub ((measurable_const.mul h1).mul O.label_meas)

lemma flip_icc (O : Oracle μ S) (v p : S) : O.flip v p ∈ Set.Icc (0 : ℝ) 1 := by
  rcases O.flip_bit v p with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num

end Flip

open scoped ENNReal in
/-- A failure bound gives the complementary success bound, with no measurability
needed: outer measure is subadditive and `s ∪ sᶜ = univ`. -/
theorem one_sub_le_compl_real {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] (s : Set α) (d : ℝ) (h : ν.real s ≤ d) : 1 - d ≤ ν.real sᶜ := by
  have hsub : (1 : ℝ≥0∞) ≤ ν s + ν sᶜ := by
    calc (1 : ℝ≥0∞) = ν Set.univ := measure_univ.symm
      _ = ν (s ∪ sᶜ) := by rw [Set.union_compl_self]
      _ ≤ ν s + ν sᶜ := measure_union_le _ _
  have hreal : (1 : ℝ) ≤ ν.real s + ν.real sᶜ := by
    have := ENNReal.toReal_mono (by finiteness) hsub
    rwa [ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _), ENNReal.toReal_one] at this
  linarith

/-- Soundness + termination ⇒ correctness.  The two halves of a retry loop compose by
a union bound: if whatever is returned is valid except w.p. `δ/2` (uniformly over *when*
it is returned), and the loop returns at all except w.p. `δ/2`, then with probability
`≥ 1 − δ` the loop returns something *and* what it returns is valid. -/
theorem sound_and_terminating {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] {T : Type*} [Countable T]
    (Fail Ret : T → Set α) (δ : ℝ)
    (hvalid : ν.real (⋃ t, Fail t) ≤ δ / 2)
    (hterm : ν.real {y | ∀ t, y ∉ Ret t} ≤ δ / 2) :
    1 - δ ≤ ν.real {y | (∃ t, y ∈ Ret t) ∧ ∀ t, y ∉ Fail t} := by
  have hcompl : {y : α | (∃ t, y ∈ Ret t) ∧ ∀ t, y ∉ Fail t}
      = ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t})ᶜ := by
    ext y
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_union, Set.mem_iUnion, not_or,
      not_exists, not_forall, not_not]
    constructor
    · rintro ⟨⟨t, ht⟩, hF⟩
      exact ⟨hF, ⟨t, ht⟩⟩
    · rintro ⟨hF, ⟨t, ht⟩⟩
      exact ⟨⟨t, ht⟩, hF⟩
  rw [hcompl]
  refine one_sub_le_compl_real ν _ δ ?_
  calc ν.real ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t})
      ≤ ν.real (⋃ t, Fail t) + ν.real {y | ∀ t, y ∉ Ret t} := measureReal_union_le _ _
    _ ≤ δ / 2 + δ / 2 := add_le_add hvalid hterm
    _ = δ := by ring

end OrthoDFA
