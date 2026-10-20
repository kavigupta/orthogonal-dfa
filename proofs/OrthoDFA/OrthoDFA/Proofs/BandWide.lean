import OrthoDFA.Proofs.FamilyRead

/-!
# A wide band passes its check

The vote is Poisson-binomial over `N ≤ κ` bits.  A band centred at `κ/2` has the mean on one side
of its centre, and `BitSumTails`' comparison holds the far side's read to `Bin(κ, 1/2)`'s tail at
`lo`.
-/

namespace OrthoDFA

open Polynomial

/-- The Poisson-binomial law of independent bits at rates `ps`. -/
def pbLaw (ps : List ℝ) : ℕ → ℝ := ps.foldr pbStep pbBase

lemma pbLaw_eq_coeff : ∀ (ps : List ℝ) (j : ℕ), pbLaw ps j = (bitPoly (ps : Multiset ℝ)).coeff j
  | [], j => by
    simp [pbLaw, pbBase, bitPoly, coeff_one]
  | p :: ps, j => by
    rw [← Multiset.cons_coe, bitPoly_cons]
    change pbStep p (pbLaw ps) j = _
    cases j with
    | zero => rw [pbStep, coeff_bitLin_mul_zero, pbLaw_eq_coeff ps]
    | succ j => rw [pbStep, coeff_bitLin_mul_succ, pbLaw_eq_coeff ps, pbLaw_eq_coeff ps]

/-- A band at `lo` and `hi + 1`, centred at `κ/2` and at least three counts from it either side,
passes its check over at most `κ` reads at `ε`, once `Bin(κ, 1/2)` lands at or below `lo` at most
`ε` of the time. -/
theorem bandPasses_of_halfTail {N κ lo hi : ℕ} {p r ε ε₂ κ' : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hN : N ≤ κ) (hsum : lo + hi + 1 = κ) (hlo : lo + 3 ≤ hi)
    (hε : 1 - binomSfGe κ (1 / 2) (lo + 1) ≤ ε) :
    BandPasses N lo (hi + 1) p r ε ε₂ κ' := by
  classical
  intro a ha
  set ps := List.replicate a p ++ List.replicate (N - a) r with hpsd
  set M : Multiset ℝ := (ps : Multiset ℝ) with hM
  have hlen : M.card = N := by
    simp only [hM, Multiset.coe_card, ps, List.length_append, List.length_replicate]; omega
  have hM01 : ∀ q ∈ M, 0 ≤ q ∧ q ≤ 1 := by
    intro q hq
    simp only [hM, Multiset.mem_coe, ps, List.mem_append, List.mem_replicate] at hq
    rcases hq with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact ⟨hp0, hp1⟩
    · exact ⟨hr0, hr1⟩
  have hdist : ∀ rd, (∑ j ∈ Finset.range (N + 1), if readOf lo (hi + 1) j = rd then
        ∑ x ∈ Finset.antidiagonal j, (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
          * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2)) else 0)
      = ∑ j ∈ Finset.range (N + 1), if readOf lo (hi + 1) j = rd then (bitPoly M).coeff j
          else 0 :=
    fun rd => Finset.sum_congr rfl fun j _ => by
      rw [hM, ← pbLaw_eq_coeff, show pbLaw ps j = voteLaw N a p r j from
        (voteLaw_eq_foldr N a p r j).symm]
      rfl
  have hκR : ((κ : ℕ) : ℝ) = lo + hi + 1 := by rw [← hsum]; push_cast; ring
  have hloR : (lo : ℝ) + 3 ≤ hi := by exact_mod_cast hlo
  have hκ : (κ : ℝ) * (1 / 2) = ((lo : ℝ) + hi + 1) / 2 := by rw [hκR]; ring
  have hb1 : binomSfGe κ (1 / 2) (lo + 1) ≤ 1 := binomSfGe_le_one (by norm_num) (by norm_num) _
  dsimp only
  rw [hdist .accept, hdist .reject]
  have hacc : M.sum ≤ (κ : ℝ) * (1 / 2) → ∑ j ∈ Finset.range (N + 1),
      (if readOf lo (hi + 1) j = .accept then (bitPoly M).coeff j else 0) ≤ ε := by
    intro hm
    have h := highMass_le_binomSfGe M hM01 κ (hi + 1) (1 / 2) (by omega) (by norm_num)
      (by norm_num) hm (by push_cast; linarith)
    rw [binomSfGe_symm _ κ (hi + 1) (by omega), show κ + 1 - (hi + 1) = lo + 1 by omega,
      show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num, hlen] at h
    refine le_trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_)) (h.trans hε)
    unfold readOf
    by_cases hj : hi + 1 ≤ j
    · simp [hj]
    · by_cases hj' : j ≤ lo <;> simp [hj, hj']
  have hrej : (κ : ℝ) * (1 / 2) ≤ M.sum → ∑ j ∈ Finset.range (N + 1),
      (if readOf lo (hi + 1) j = .reject then (bitPoly M).coeff j else 0) ≤ ε := by
    intro hm
    have h := lowMass_le_binomSfGe M hM01 κ lo (1 / 2) (by omega) (by norm_num) hm
      (by rw [hκ]; linarith)
    refine le_trans ?_ (h.trans hε)
    rw [← Finset.sum_filter, lowMass]
    refine Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => ?_)
      (fun j _ _ => bitPoly_coeff_nonneg M hM01 j)
    rw [Finset.mem_filter] at hj
    obtain ⟨-, hj⟩ := hj
    unfold readOf at hj
    rw [Finset.mem_range]
    split_ifs at hj with h1 h2
    exact Nat.lt_succ_of_le h2
  rcases le_or_gt M.sum ((κ : ℝ) * (1 / 2)) with hm | hm
  · exact ⟨Or.inl (hacc hm), (min_le_left _ _).trans ((hacc hm).trans (le_max_left _ _))⟩
  · exact ⟨Or.inr (Or.inl (hrej hm.le)),
      (min_le_right _ _).trans ((hrej hm.le).trans (le_max_left _ _))⟩

end OrthoDFA
