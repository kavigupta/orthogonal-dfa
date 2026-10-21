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

lemma bitPoly_sum_coeff (M : Multiset ℝ) :
    ∑ j ∈ Finset.range (M.card + 1), (bitPoly M).coeff j = 1 := by
  have h := eval_eq_sum_range' (n := M.card + 1)
    (Nat.lt_succ_of_le (natDegree_bitPoly_le M)) (1 : ℝ)
  simp only [one_pow, mul_one] at h
  rw [← h, bitPoly, eval_multiset_prod, Multiset.map_map]
  have : ∀ a ∈ M, (eval 1 ∘ bitLin) a = 1 := fun a _ => by simp [bitLin]
  rw [Multiset.map_congr rfl this, Multiset.map_const', Multiset.prod_replicate, one_pow]

/-- `BandPasses` is monotone in its three rates. -/
theorem BandPasses.mono {N kl kh : ℕ} {p r ε ε₂ κ ε' ε₂' κ' : ℝ}
    (h : BandPasses N kl kh p r ε ε₂ κ) (hε : ε ≤ ε') (hε₂ : ε₂ ≤ ε₂') (hκ : κ ≤ κ')
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hr0 : 0 ≤ r) (hr1 : r ≤ 1) :
    BandPasses N kl kh p r ε' ε₂' κ' := by
  intro a ha
  obtain ⟨h1, h2⟩ := h a ha
  dsimp only at h1 h2 ⊢
  have hU : 0 ≤ ∑ j ∈ Finset.range (N + 1),
      if readOf kl kh j = .undecided then
        ∑ x ∈ Finset.antidiagonal j,
          (a.choose x.1 * p ^ x.1 * (1 - p) ^ (a - x.1))
            * ((N - a).choose x.2 * r ^ x.2 * (1 - r) ^ (N - a - x.2))
      else 0 := by
    refine Finset.sum_nonneg fun j _ => ?_
    split_ifs
    · refine Finset.sum_nonneg fun x _ => mul_nonneg ?_ ?_ <;>
        exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg (by assumption) _))
          (pow_nonneg (by linarith) _)
    · exact le_rfl
  refine ⟨?_, h2.trans (max_le_max hε (mul_le_mul_of_nonneg_right hκ hU))⟩
  rcases h1 with h1 | h1 | ⟨h1 | h1, h3⟩
  · exact Or.inl (h1.trans hε)
  · exact Or.inr (Or.inl (h1.trans hε))
  · exact Or.inr (Or.inr ⟨Or.inl (h1.trans hε₂), h3⟩)
  · exact Or.inr (Or.inr ⟨Or.inr (h1.trans hε₂), h3⟩)

/-- A band at `lo` and `hi + 1`, centred over `n` reads and split `g` counts inside its lower edge
and mirrored at its upper, passes its check over at most `n` reads. A vote whose mean lies outside
the split reads past the far edge at most `t`, the tail of `Bin(n, (lo + g)/n)` at `hi + 1`. A
vote whose mean lies inside it reads past the near edge at most `h`, the tail of `Bin(n, 1/2)` at
`lo`, and is undecided at least

    u = 1 − h − P(Bin(n, (lo + g)/n) ≤ lo),

so at least a third of the time once `u ≥ 1/3`, and it reads wrong at most `h/u` times as often
as undecided. -/
theorem bandPasses_of_split {N n lo hi g : ℕ} {p r : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hr0 : 0 ≤ r) (hr1 : r ≤ 1) (hN : N ≤ n) (hsum : lo + hi + 1 = n) (hlo : lo + 3 ≤ hi)
    (hg2 : 2 ≤ g) (hg : lo + g + 1 ≤ hi)
    (hu : 1 / 3 ≤ 1 - (1 - binomSfGe n (1 / 2) (lo + 1))
      - (1 - binomSfGe n (((lo + g : ℕ) : ℝ) / n) (lo + 1))) :
    BandPasses N lo (hi + 1) p r (binomSfGe n (((lo + g : ℕ) : ℝ) / n) (hi + 1))
      (1 - binomSfGe n (1 / 2) (lo + 1))
      ((1 - binomSfGe n (1 / 2) (lo + 1)) / (1 - (1 - binomSfGe n (1 / 2) (lo + 1))
        - (1 - binomSfGe n (((lo + g : ℕ) : ℝ) / n) (lo + 1)))) := by
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
  dsimp only
  rw [hdist .accept, hdist .reject, hdist .undecided]
  set A := ∑ j ∈ Finset.range (N + 1),
    if readOf lo (hi + 1) j = .accept then (bitPoly M).coeff j else 0 with hA
  set R := ∑ j ∈ Finset.range (N + 1),
    if readOf lo (hi + 1) j = .reject then (bitPoly M).coeff j else 0 with hR
  set U := ∑ j ∈ Finset.range (N + 1),
    if readOf lo (hi + 1) j = .undecided then (bitPoly M).coeff j else 0 with hU
  have htot : A + R + U = 1 := by
    rw [hA, hR, hU, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib, ← hlen,
      ← bitPoly_sum_coeff M]
    refine Finset.sum_congr rfl fun j _ => ?_
    unfold readOf
    split_ifs <;> simp_all
  have hnR : (n : ℝ) = lo + hi + 1 := by rw [← hsum]; push_cast; ring
  have hloR : (lo : ℝ) + 3 ≤ hi := by exact_mod_cast hlo
  have hgR : (lo : ℝ) + g + 1 ≤ hi := by exact_mod_cast hg
  have hg2R : (2 : ℝ) ≤ g := by exact_mod_cast hg2
  have hn0 : (0 : ℝ) < n := by rw [hnR]; positivity
  set q₁ : ℝ := (((lo + g : ℕ) : ℝ)) / n with hq₁
  have hnq₁ : (n : ℝ) * q₁ = lo + g := by rw [hq₁]; push_cast; field_simp
  have hq₁0 : 0 ≤ q₁ := by positivity
  have hq₁1 : q₁ ≤ 1 := by rw [hq₁, div_le_one hn0]; push_cast; linarith
  set h := 1 - binomSfGe n (1 / 2) (lo + 1) with hh
  set e := 1 - binomSfGe n q₁ (lo + 1) with he
  set t := binomSfGe n q₁ (hi + 1) with ht
  have hb1 : binomSfGe n (1 / 2) (lo + 1) ≤ 1 := binomSfGe_le_one (by norm_num) (by norm_num) _
  have hh0 : 0 ≤ h := by rw [hh]; linarith
  have hupos : 0 < 1 - h - e := by linarith
  -- the masses past an edge, from Hoeffding's comparison
  have hacc : ∀ q : ℝ, 0 ≤ q → q ≤ 1 → M.sum ≤ n * q → n * q + 2 ≤ hi + 1 →
      A ≤ binomSfGe n q (hi + 1) := by
    intro q hq0 hq1 hm hc
    have h := highMass_le_binomSfGe M hM01 n (hi + 1) q (by omega) hq0 hq1 hm
      (by push_cast; linarith)
    rw [hlen] at h
    refine le_trans (le_of_eq (Finset.sum_congr rfl fun j _ => ?_)) h
    unfold readOf
    by_cases hj : hi + 1 ≤ j
    · simp [hj]
    · by_cases hj' : j ≤ lo <;> simp [hj, hj']
  have hrej : ∀ q : ℝ, 0 ≤ q → n * q ≤ M.sum → (lo : ℝ) + 2 ≤ n * q →
      R ≤ 1 - binomSfGe n q (lo + 1) := by
    intro q hq0 hm hb
    have h := lowMass_le_binomSfGe M hM01 n lo q (by omega) hq0 hm hb
    refine le_trans ?_ h
    rw [hR, ← Finset.sum_filter, lowMass, ← hlen]
    refine Finset.sum_le_sum_of_subset_of_nonneg (fun j hj => ?_)
      (fun j _ _ => bitPoly_coeff_nonneg M hM01 j)
    rw [Finset.mem_filter] at hj
    obtain ⟨-, hj⟩ := hj
    unfold readOf at hj
    rw [Finset.mem_range]
    split_ifs at hj with h1 h2
    exact Nat.lt_succ_of_le h2
  -- the mirrored rate, and the tails at it
  have hsymm1 : binomSfGe n (1 / 2) (hi + 1) = h := by
    rw [binomSfGe_symm _ n (hi + 1) (by omega), show n + 1 - (hi + 1) = lo + 1 by omega,
      show (1 : ℝ) - 1 / 2 = 1 / 2 by norm_num]
  have hq₂ : (1 : ℝ) - q₁ = ((hi + 1 : ℝ) - g) / n := by
    rw [hq₁, eq_div_iff hn0.ne', sub_mul, div_mul_cancel₀ _ hn0.ne', hnR]; push_cast; ring
  have hsymm2 : binomSfGe n (((hi + 1 : ℝ) - g) / n) (hi + 1) = e := by
    rw [binomSfGe_symm _ n (hi + 1) (by omega), show n + 1 - (hi + 1) = lo + 1 by omega, ← hq₂,
      sub_sub_cancel]
  have hsymm3 : 1 - binomSfGe n (((hi + 1 : ℝ) - g) / n) (lo + 1) = t := by
    rw [ht, binomSfGe_symm _ n (hi + 1) (by omega), show n + 1 - (hi + 1) = lo + 1 by omega, hq₂]
  have hq₂0 : 0 ≤ ((hi + 1 : ℝ) - g) / n := by
    rw [← hq₂]; linarith
  have hq₂1 : ((hi + 1 : ℝ) - g) / n ≤ 1 := by rw [← hq₂]; linarith
  have hnq₂ : (n : ℝ) * (((hi + 1 : ℝ) - g) / n) = hi + 1 - g := by field_simp
  have hA0 : 0 ≤ A := Finset.sum_nonneg fun j _ => by
    split_ifs
    · exact bitPoly_coeff_nonneg M hM01 j
    · exact le_rfl
  have hR0 : 0 ≤ R := Finset.sum_nonneg fun j _ => by
    split_ifs
    · exact bitPoly_coeff_nonneg M hM01 j
    · exact le_rfl
  -- inside the split, the near edge is read at most `h` and the far one at most `e`
  have hin : ∀ X Y : ℝ, X ≤ h → Y ≤ e → X + Y + U = 1 → 0 ≤ X →
      (X ≤ h ∧ 1 / 3 ≤ U) ∧ X ≤ h / (1 - h - e) * U := by
    intro X Y hX hY hXYU hX0
    have hU : 1 - h - e ≤ U := by linarith
    refine ⟨⟨hX, by linarith⟩, ?_⟩
    calc X ≤ h := hX
      _ = h / (1 - h - e) * (1 - h - e) := by field_simp
      _ ≤ h / (1 - h - e) * U := mul_le_mul_of_nonneg_left hU (div_nonneg hh0 hupos.le)
  rcases le_or_gt M.sum (lo + g) with hm | hm
  · have hAt : A ≤ t := hacc q₁ hq₁0 hq₁1 (by rw [hnq₁]; exact hm) (by rw [hnq₁]; linarith)
    exact ⟨Or.inl hAt, (min_le_left _ _).trans (hAt.trans (le_max_left _ _))⟩
  rcases le_or_gt M.sum ((n : ℝ) * (1 / 2)) with hm2 | hm2
  · have hAh : A ≤ h := hsymm1 ▸ hacc (1 / 2) (by norm_num) (by norm_num) hm2
      (by rw [hnR]; linarith)
    have hRe : R ≤ e := hrej q₁ hq₁0 (by rw [hnq₁]; exact hm.le) (by rw [hnq₁]; linarith)
    obtain ⟨h1, h2⟩ := hin A R hAh hRe (by linarith) hA0
    exact ⟨Or.inr (Or.inr ⟨Or.inl h1.1, h1.2⟩),
      (min_le_left _ _).trans (h2.trans (le_max_right _ _))⟩
  rcases lt_or_ge M.sum ((hi + 1 : ℝ) - g) with hm3 | hm3
  · have hRh : R ≤ h := hrej (1 / 2) (by norm_num) hm2.le (by rw [hnR]; linarith)
    have hAe : A ≤ e := hsymm2 ▸ hacc _ hq₂0 hq₂1 (by rw [hnq₂]; exact hm3.le)
      (by rw [hnq₂]; linarith)
    obtain ⟨h1, h2⟩ := hin R A hRh hAe (by linarith) hR0
    exact ⟨Or.inr (Or.inr ⟨Or.inr h1.1, h1.2⟩),
      (min_le_right _ _).trans (h2.trans (le_max_right _ _))⟩
  · have hRt : R ≤ t := hsymm3 ▸ hrej _ hq₂0 (by rw [hnq₂]; exact hm3) (by rw [hnq₂]; linarith)
    exact ⟨Or.inr (Or.inl hRt), (min_le_right _ _).trans (hRt.trans (le_max_left _ _))⟩

end OrthoDFA
