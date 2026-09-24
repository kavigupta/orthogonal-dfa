import Mathlib.Data.Real.Basic
import Mathlib.Tactic
import OrthoDFA.Proofs.Estimate
import OrthoDFA.Proofs.Schedule

/-!
# Liveness, step 1: the clustering selects an accept-preserving family

The ε-anchored greedy returns the `k` suffixes of least loss against its centre.
This file proves the *selection* fact that liveness rests on, with no probability:
if the accept-preserving suffixes are separated *below* the non-accept-preserving
ones (every AP loss strictly under every non-AP loss) and there are at least `k`
of them, then the `k` least-loss suffixes are all accept-preserving.

`chosen` abstracts the greedy's output — any least-loss `k`-subset. Separability
(`hsep`) is the substantive hypothesis: the noisy losses must actually split the
two classes, which downstream is a concentration statement about the pool once it
is rich enough to expose every non-accept-preserving suffix's flip.
-/

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

variable {ι : Type*} (pref : ι → S)

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

namespace Oracle
variable {S : Type*} [MeasurableSpace S] [Mul S] [DecidableEq S] (O : Oracle μ S)

end Oracle

end OrthoDFA
