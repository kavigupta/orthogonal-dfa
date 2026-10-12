import OrthoDFA.Proofs.FamilyRead

/-!
# A wide band passes its check

The vote is Poisson-binomial over `N` bits, so its variance is at most `N/4`. The mean is at least
`(hi + 1 − lo)/2` from one of the band's sides, and Chebyshev's inequality holds that side's read
to `N / (hi + 1 − lo)²`.
-/

namespace OrthoDFA

/-- The Poisson-binomial law of independent bits at rates `ps`. -/
def pbLaw (ps : List ℝ) : ℕ → ℝ := ps.foldr pbStep pbBase

lemma pbLaw_cons (p : ℝ) (ps : List ℝ) : pbLaw (p :: ps) = pbStep p (pbLaw ps) := rfl

lemma pbLaw_nonneg : ∀ (ps : List ℝ), (∀ p ∈ ps, 0 ≤ p ∧ p ≤ 1) → ∀ j, 0 ≤ pbLaw ps j
  | [], _, j => by
    simp only [pbLaw, List.foldr_nil, pbBase]
    split_ifs <;> norm_num
  | p :: ps, h, j => by
    have hp := h p List.mem_cons_self
    have ih := pbLaw_nonneg ps (fun q hq => h q (List.mem_cons_of_mem _ hq))
    rw [pbLaw_cons]
    cases j with
    | zero => simp only [pbStep]; exact mul_nonneg (by linarith) (ih 0)
    | succ j =>
      simp only [pbStep]
      exact add_nonneg (mul_nonneg (by linarith) (ih _)) (mul_nonneg hp.1 (ih j))

lemma pbLaw_zero_of_lt : ∀ (ps : List ℝ) (j : ℕ), ps.length < j → pbLaw ps j = 0
  | [], j, h => by
    have hj : j ≠ 0 := by simp only [List.length_nil] at h; omega
    simp [pbLaw, pbBase, hj]
  | p :: ps, j, h => by
    rw [pbLaw_cons]
    simp only [List.length_cons] at h
    cases j with
    | zero => omega
    | succ j =>
      simp only [pbStep]
      rw [pbLaw_zero_of_lt ps (j + 1) (by omega), pbLaw_zero_of_lt ps j (by omega)]
      ring

/-- `E[g(X)]` for `X` of law `pbLaw ps`. -/
def pbMean (ps : List ℝ) (g : ℕ → ℝ) : ℝ :=
  ∑ j ∈ Finset.range (ps.length + 1), g j * pbLaw ps j

lemma pbMean_cons (p : ℝ) (ps : List ℝ) (g : ℕ → ℝ) :
    pbMean (p :: ps) g = (1 - p) * pbMean ps g + p * pbMean ps (fun j => g (j + 1)) := by
  unfold pbMean
  rw [List.length_cons, Finset.sum_range_succ', pbLaw_cons]
  simp only [pbStep]
  have h1 : ∑ j ∈ Finset.range (ps.length + 1),
        g (j + 1) * ((1 - p) * pbLaw ps (j + 1) + p * pbLaw ps j)
      = (1 - p) * ∑ j ∈ Finset.range (ps.length + 1), g (j + 1) * pbLaw ps (j + 1)
        + p * ∑ j ∈ Finset.range (ps.length + 1), g (j + 1) * pbLaw ps j := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [h1]
  have h2 := Finset.sum_range_succ' (fun j => g j * pbLaw ps j) (ps.length + 1)
  have h3 := Finset.sum_range_succ (fun j => g j * pbLaw ps j) (ps.length + 1)
  rw [pbLaw_zero_of_lt ps (ps.length + 1) (by omega), mul_zero, add_zero] at h3
  linear_combination (1 - p) * (h3.symm.trans h2).symm

lemma pbMean_sq (ps : List ℝ) : ∀ c : ℝ, pbMean ps (fun j => ((j : ℝ) - c) ^ 2)
    = (ps.map fun p => p * (1 - p)).sum + (ps.sum - c) ^ 2 := by
  induction ps with
  | nil => intro c; simp [pbMean, pbLaw, pbBase]
  | cons p ps ih =>
    intro c
    rw [pbMean_cons]
    have e : (fun j : ℕ => (((j + 1 : ℕ) : ℝ) - c) ^ 2) = fun j : ℕ => ((j : ℝ) - (c - 1)) ^ 2 := by
      funext j; push_cast; ring
    rw [e, ih c, ih (c - 1)]
    simp only [List.map_cons, List.sum_cons]
    ring

/-- A band whose width squared is at least `N / ε` passes its check at `ε`. -/
theorem bandPasses_of_wide {N lo hi : ℕ} {p r ε ε₂ κ : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hlohi : lo ≤ hi)
    (hw : (N : ℝ) ≤ ε * ((hi : ℝ) + 1 - lo) ^ 2) :
    BandPasses N lo (hi + 1) p r ε ε₂ κ := by
  intro a ha
  set ps := List.replicate a p ++ List.replicate (N - a) r with hpsd
  have hlen : ps.length = N := by simp only [ps, List.length_append, List.length_replicate]; omega
  have hps : ∀ q ∈ ps, 0 ≤ q ∧ q ≤ 1 := by
    intro q hq
    simp only [ps, List.mem_append, List.mem_replicate] at hq
    rcases hq with ⟨-, rfl⟩ | ⟨-, rfl⟩
    · exact ⟨hp0, hp1⟩
    · exact ⟨hr0, hr1⟩
  have hdist : ∀ rd, (∑ j ∈ Finset.range (N + 1), if readOf lo (hi + 1) j = rd then
        ∑ x ∈ Finset.antidiagonal j, (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
          * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2)) else 0)
      = ∑ j ∈ Finset.range (N + 1), if readOf lo (hi + 1) j = rd then pbLaw ps j else 0 :=
    fun rd => Finset.sum_congr rfl fun j _ => by
      rw [show pbLaw ps j = voteLaw N a p r j from (voteLaw_eq_foldr N a p r j).symm]
      rfl
  set m := ps.sum
  set d : ℝ := ((hi : ℝ) + 1 - lo) / 2 with hd
  have hlh : (lo : ℝ) ≤ hi := by exact_mod_cast hlohi
  have hd0 : 0 < d := by rw [hd]; linarith
  have hvar : (ps.map fun q => q * (1 - q)).sum ≤ (N : ℝ) / 4 := by
    have hq : ∀ x ∈ ps.map (fun q => q * (1 - q)), x ≤ 1 / 4 := by
      intro x hx
      obtain ⟨q, -, rfl⟩ := List.mem_map.1 hx
      nlinarith [sq_nonneg (q - 1 / 2)]
    have := List.sum_le_card_nsmul _ (1 / 4) hq
    rw [List.length_map, hlen, nsmul_eq_mul] at this
    linarith
  have hcheb : pbMean ps (fun j => ((j : ℝ) - m) ^ 2) ≤ (N : ℝ) / 4 := by
    rw [pbMean_sq]
    simp only [m, sub_self]
    linarith
  have htail : ∀ (T : ℕ → Prop) [DecidablePred T], (∀ j, T j → d ≤ |(j : ℝ) - m|) →
      ∑ j ∈ Finset.range (N + 1), (if T j then pbLaw ps j else 0) ≤ ε := by
    intro T _ hT
    calc ∑ j ∈ Finset.range (N + 1), (if T j then pbLaw ps j else 0)
        ≤ ∑ j ∈ Finset.range (N + 1), (((j : ℝ) - m) ^ 2 / d ^ 2) * pbLaw ps j :=
          Finset.sum_le_sum fun j _ => by
            have hf := pbLaw_nonneg ps hps j
            split_ifs with hj
            · have h1 := hT j hj
              have h2 : d ^ 2 ≤ ((j : ℝ) - m) ^ 2 := by
                rw [← sq_abs ((j : ℝ) - m)]
                exact pow_le_pow_left₀ hd0.le h1 2
              have : 1 ≤ ((j : ℝ) - m) ^ 2 / d ^ 2 := by
                rw [le_div_iff₀ (by positivity)]; linarith
              nlinarith
            · exact mul_nonneg (by positivity) hf
      _ = pbMean ps (fun j => ((j : ℝ) - m) ^ 2) / d ^ 2 := by
          unfold pbMean
          rw [hlen, Finset.sum_div]
          exact Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ((N : ℝ) / 4) / d ^ 2 := div_le_div_of_nonneg_right hcheb (by positivity)
      _ ≤ ε := by
          rw [div_le_iff₀ (by positivity), hd]
          nlinarith [hw]
  dsimp only
  rw [hdist .accept, hdist .reject]
  have hacc : m ≤ lo + d → ∑ j ∈ Finset.range (N + 1),
      (if readOf lo (hi + 1) j = .accept then pbLaw ps j else 0) ≤ ε := by
    intro hm
    refine htail _ fun j hj => ?_
    have hj' : hi + 1 ≤ j := by
      unfold readOf at hj
      split_ifs at hj with h1
      exact h1
    have : ((hi : ℝ) + 1) ≤ j := by exact_mod_cast hj'
    rw [abs_of_nonneg (by rw [hd] at hm; linarith)]
    rw [hd] at hm ⊢
    linarith
  have hrej : lo + d < m → ∑ j ∈ Finset.range (N + 1),
      (if readOf lo (hi + 1) j = .reject then pbLaw ps j else 0) ≤ ε := by
    intro hm
    refine htail _ fun j hj => ?_
    have hj' : j ≤ lo := by
      unfold readOf at hj
      split_ifs at hj with h1 h2
      exact h2
    have : (j : ℝ) ≤ lo := by exact_mod_cast hj'
    rw [abs_of_nonpos (by linarith)]
    linarith
  rcases le_or_gt m (lo + d) with hm | hm
  · exact ⟨Or.inl (hacc hm), (min_le_left _ _).trans ((hacc hm).trans (le_max_left _ _))⟩
  · exact ⟨Or.inr (Or.inl (hrej hm)), (min_le_right _ _).trans ((hrej hm).trans (le_max_left _ _))⟩

end OrthoDFA
