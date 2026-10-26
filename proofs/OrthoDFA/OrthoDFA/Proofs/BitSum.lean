import OrthoDFA.LowLevelClaims.BitSum
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.Hoeffding
import Mathlib.Algebra.Polynomial.Derivative
import Mathlib.Probability.Distributions.Bernoulli

/-!
# A sum of independent bits against a binomial

Hoeffding's (1956) comparison, for counts at least two below the mean.  The sum's law is the
coefficients of `∏ (1 − pᵢ + pᵢ X)`.  Replacing two rates `a < p̄ < c` by `p̄` and `a + c − p̄`
keeps the sum and raises `ac`, which moves the mass at or below `b` by `Δ(ac)·(R_b − R_{b−1})`,
`R` the law of the other bits; that is nonnegative while `R`'s mass still rises into `b`, which it
does for `b` up to its mean.  Each step sets one more rate to the mean, ending at the binomial.
-/

namespace OrthoDFA

open Polynomial MeasureTheory ProbabilityTheory

/-- One bit at rate `a`, as a generating polynomial. -/
noncomputable def bitLin (a : ℝ) : ℝ[X] := C (1 - a) + C a * X

/-- The generating polynomial of independent bits at the rates `M`. -/
noncomputable def bitPoly (M : Multiset ℝ) : ℝ[X] := (M.map bitLin).prod

/-- The mass a generating polynomial puts below `t`. -/
noncomputable def lowMass (P : ℝ[X]) (t : ℕ) : ℝ := ∑ k ∈ Finset.range t, P.coeff k

lemma bitPoly_cons (a : ℝ) (M : Multiset ℝ) : bitPoly (a ::ₘ M) = bitLin a * bitPoly M := by
  simp [bitPoly]

lemma bitPoly_add (M N : Multiset ℝ) : bitPoly (M + N) = bitPoly M * bitPoly N := by
  simp [bitPoly]

lemma coeff_bitLin_mul_zero (a : ℝ) (R : ℝ[X]) : (bitLin a * R).coeff 0 = (1 - a) * R.coeff 0 := by
  rw [bitLin, add_mul, coeff_add, coeff_C_mul, mul_assoc, coeff_C_mul, coeff_X_mul_zero, mul_zero,
    add_zero]

lemma coeff_bitLin_mul_succ (a : ℝ) (R : ℝ[X]) (k : ℕ) :
    (bitLin a * R).coeff (k + 1) = (1 - a) * R.coeff (k + 1) + a * R.coeff k := by
  rw [bitLin, add_mul, coeff_add, coeff_C_mul, mul_assoc, coeff_C_mul, coeff_X_mul]

lemma lowMass_zero (P : ℝ[X]) : lowMass P 0 = 0 := by simp [lowMass]

lemma lowMass_succ (P : ℝ[X]) (t : ℕ) : lowMass P (t + 1) = lowMass P t + P.coeff t := by
  simp [lowMass, Finset.sum_range_succ]

lemma lowMass_bitLin_mul (a : ℝ) (R : ℝ[X]) (t : ℕ) :
    lowMass (bitLin a * R) (t + 1) = (1 - a) * lowMass R (t + 1) + a * lowMass R t := by
  induction t with
  | zero =>
    rw [zero_add, lowMass, lowMass, lowMass, Finset.sum_range_one, Finset.sum_range_one,
      Finset.sum_range_zero, coeff_bitLin_mul_zero]
    ring
  | succ t ih =>
    rw [lowMass_succ, ih, coeff_bitLin_mul_succ, lowMass_succ R (t + 1), lowMass_succ R t]
    ring

lemma bitPoly_coeff_nonneg (M : Multiset ℝ) (hM : ∀ a ∈ M, 0 ≤ a ∧ a ≤ 1) (k : ℕ) :
    0 ≤ (bitPoly M).coeff k := by
  induction M using Multiset.induction_on generalizing k with
  | empty =>
    simp only [bitPoly, Multiset.map_zero, Multiset.prod_zero, coeff_one]
    split_ifs <;> norm_num
  | cons a M ih =>
    have ha := hM a (Multiset.mem_cons_self a M)
    have ih' := fun k => ih (fun b hb => hM b (Multiset.mem_cons_of_mem hb)) k
    rw [bitPoly_cons]
    cases k with
    | zero => rw [coeff_bitLin_mul_zero]; exact mul_nonneg (by linarith) (ih' 0)
    | succ k =>
      rw [coeff_bitLin_mul_succ]
      exact add_nonneg (mul_nonneg (by linarith) (ih' _)) (mul_nonneg ha.1 (ih' _))

/-- The derivative's coefficients: `(k+1)·c_{k+1}` sums each rate times the coefficient without
it. -/
lemma bitPoly_coeff_succ_mul (M : Multiset ℝ) (k : ℕ) :
    (bitPoly M).coeff (k + 1) * ((k : ℝ) + 1)
      = (M.map fun a => (bitPoly (M.erase a)).coeff k * a).sum := by
  classical
  have hd : derivative (bitPoly M) = (M.map fun a => bitPoly (M.erase a) * C a).sum := by
    rw [bitPoly, derivative_prod]
    refine congrArg Multiset.sum (Multiset.map_congr rfl fun a _ => ?_)
    rw [bitLin, derivative_add, derivative_C, zero_add, derivative_C_mul_X, bitPoly]
  have h := congrArg (fun P => P.coeff k) hd
  simp only [coeff_derivative] at h
  rw [h, ← lcoeff_apply, map_multiset_sum, Multiset.map_map]
  refine congrArg Multiset.sum (Multiset.map_congr rfl fun a _ => ?_)
  simp [coeff_mul_C]

/-- A sum of bits puts no less mass at `k + 1` than at `k` while `k + 1` is at most its mean. -/
theorem bitPoly_coeff_mono : ∀ (n : ℕ) (M : Multiset ℝ), M.card = n →
    (∀ a ∈ M, 0 ≤ a ∧ a ≤ 1) → ∀ k : ℕ, (k : ℝ) + 1 ≤ M.sum →
    (bitPoly M).coeff k ≤ (bitPoly M).coeff (k + 1) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro M hcard hM k hk
  -- dropping any one bit only raises the mass at `k`
  have hdrop : ∀ a ∈ M, (bitPoly M).coeff k ≤ (bitPoly (M.erase a)).coeff k := by
    intro a ha
    have hE : bitPoly M = bitLin a * bitPoly (M.erase a) := by
      rw [← bitPoly_cons, Multiset.cons_erase ha]
    have hEM : ∀ b ∈ M.erase a, 0 ≤ b ∧ b ≤ 1 := fun b hb => hM b (Multiset.mem_of_mem_erase hb)
    have ha01 := hM a ha
    rw [hE]
    cases k with
    | zero =>
      rw [coeff_bitLin_mul_zero]
      have := bitPoly_coeff_nonneg _ hEM 0
      nlinarith
    | succ k =>
      rw [coeff_bitLin_mul_succ]
      have hlt : (M.erase a).card < n := by
        have := Multiset.card_pos_iff_exists_mem.2 ⟨a, ha⟩
        rw [Multiset.card_erase_of_mem ha, Nat.pred_eq_sub_one]
        omega
      have hsum : (M.erase a).sum = M.sum - a := by
        have := Multiset.sum_erase ha
        linarith
      have hmono := ih _ hlt (M.erase a) rfl hEM k (by push_cast at hk ⊢; linarith [ha01.2])
      nlinarith
  have hc0 := bitPoly_coeff_nonneg M hM k
  have hkey := bitPoly_coeff_succ_mul M k
  have hsum : (M.map fun a => (bitPoly M).coeff k * a).sum
      ≤ (M.map fun a => (bitPoly (M.erase a)).coeff k * a).sum :=
    Multiset.sum_map_le_sum_map _ _ fun a ha =>
      mul_le_mul_of_nonneg_right (hdrop a ha) (hM a ha).1
  rw [Multiset.sum_map_mul_left, Multiset.map_id'] at hsum
  have hk0 : (0 : ℝ) < (k : ℝ) + 1 := by positivity
  nlinarith


/-- Moving two rates to `a'` and `a + c − a'`, whose product is no smaller, raises the mass at or
below `b` while the rest's mass is still rising into `b`. -/
lemma lowMass_pair (a c a' : ℝ) (R : ℝ[X]) (b : ℕ) (hπ : a * c ≤ a' * (a + c - a'))
    (hR0 : 0 ≤ R.coeff 0) (hR : ∀ b', b = b' + 1 → R.coeff b' ≤ R.coeff (b' + 1)) :
    lowMass (bitLin a * (bitLin c * R)) (b + 1)
      ≤ lowMass (bitLin a' * (bitLin (a + c - a') * R)) (b + 1) := by
  cases b with
  | zero =>
    simp only [lowMass_bitLin_mul, lowMass_zero, zero_add]
    rw [lowMass_succ, lowMass_zero]
    nlinarith [mul_nonneg (sub_nonneg.2 hπ) hR0]
  | succ b =>
    simp only [lowMass_bitLin_mul]
    rw [lowMass_succ R (b + 1), lowMass_succ R b]
    nlinarith [mul_nonneg (sub_nonneg.2 hπ) (sub_nonneg.2 (hR b rfl))]

lemma bitPoly_replicate (n : ℕ) (q : ℝ) : bitPoly (Multiset.replicate n q) = bitLin q ^ n := by
  simp [bitPoly]

open scoped Classical in
/-- Averaging the rates two at a time, each step setting one more to the mean, only raises the
mass at or below `b`, and ends at the binomial. -/
theorem lowMass_le_avg (n : ℕ) (m : ℝ) (b : ℕ) (hb : (b : ℝ) + 2 ≤ m) :
    ∀ (N : ℕ) (M : Multiset ℝ), M.card = n → M.sum = m → (∀ a ∈ M, 0 ≤ a ∧ a ≤ 1) →
      (M.filter (· ≠ m / n)).card = N →
      lowMass (bitPoly M) (b + 1) ≤ lowMass (bitLin (m / n) ^ n) (b + 1) := by
  intro N
  induction N using Nat.strong_induction_on with
  | _ N ih =>
  intro M hcard hsum hM hN
  by_cases hall : ∀ x ∈ M, x = m / n
  · rw [Multiset.eq_replicate.2 ⟨hcard, hall⟩, bitPoly_replicate]
  push Not at hall
  obtain ⟨x, hx, hxne⟩ := hall
  have hconst : (M.map fun _ => m / n).sum = m := by
    have hn : (n : ℝ) ≠ 0 := by
      have := Multiset.card_pos_iff_exists_mem.2 ⟨x, hx⟩
      rw [hcard] at this
      exact_mod_cast this.ne'
    rw [Multiset.map_const', Multiset.sum_replicate, hcard, nsmul_eq_mul]
    field_simp
  have hid : (M.map fun y => y).sum = m := by rw [Multiset.map_id']; exact hsum
  obtain ⟨a, ha, hlt⟩ : ∃ a ∈ M, a < m / n := by
    by_contra h
    push Not at h
    have := Multiset.sum_lt_sum (s := M) (f := fun _ => m / n) (g := fun y => y) h
      ⟨x, hx, lt_of_le_of_ne (h x hx) (Ne.symm hxne)⟩
    linarith
  obtain ⟨c, hc, hgt⟩ : ∃ c ∈ M, m / n < c := by
    by_contra h
    push Not at h
    have := Multiset.sum_lt_sum (s := M) (f := fun y => y) (g := fun _ => m / n) h
      ⟨x, hx, lt_of_le_of_ne (h x hx) hxne⟩
    linarith
  have hca : c ≠ a := by intro h; rw [h] at hgt; linarith
  have hc' : c ∈ M.erase a := Multiset.mem_erase_of_ne hca |>.2 hc
  set R := (M.erase a).erase c with hRdef
  have hMR : M = a ::ₘ c ::ₘ R := by rw [hRdef, Multiset.cons_erase hc', Multiset.cons_erase ha]
  have ha01 := hM a ha
  have hc01 := hM c hc
  have hR01 : ∀ y ∈ R, 0 ≤ y ∧ y ≤ 1 := fun y hy => hM y (by rw [hMR]; simp [hy])
  have hRsum : R.sum = m - a - c := by
    rw [hMR, Multiset.sum_cons, Multiset.sum_cons] at hsum
    linarith
  set μ₀ := m / n with hμ₀
  set M' := μ₀ ::ₘ (a + c - μ₀) ::ₘ R with hM'def
  have hM' : ∀ y ∈ M', 0 ≤ y ∧ y ≤ 1 := by
    intro y hy
    rw [hM'def, Multiset.mem_cons, Multiset.mem_cons] at hy
    rcases hy with rfl | rfl | hy
    · constructor <;> linarith
    · constructor <;> linarith
    · exact hR01 y hy
  have hcard' : M'.card = n := by
    rw [hM'def, Multiset.card_cons, Multiset.card_cons, ← hcard, hMR, Multiset.card_cons,
      Multiset.card_cons]
  have hsum' : M'.sum = m := by
    rw [hM'def, Multiset.sum_cons, Multiset.sum_cons, hRsum]
    ring
  have hN' : (M'.filter (· ≠ μ₀)).card < N := by
    rw [← hN, hMR, Multiset.filter_cons_of_pos (p := fun x => x ≠ μ₀) _ (ne_of_lt hlt),
      Multiset.filter_cons_of_pos (p := fun x => x ≠ μ₀) _ (ne_of_gt hgt), hM'def,
      Multiset.filter_cons_of_neg (p := fun x => x ≠ μ₀) _ (by simp)]
    simp only [Multiset.card_cons]
    rw [Multiset.filter_cons]
    split_ifs <;> simp
  refine le_trans ?_ (ih _ hN' M' hcard' hsum' hM' rfl)
  rw [hMR, bitPoly_cons, bitPoly_cons, hM'def, bitPoly_cons, bitPoly_cons]
  refine lowMass_pair a c μ₀ (bitPoly R) b (by nlinarith) (bitPoly_coeff_nonneg R hR01 0) ?_
  intro b' hb'
  refine bitPoly_coeff_mono _ R rfl hR01 b' ?_
  rw [hRsum]
  have : (b : ℝ) = b' + 1 := by rw [hb']; push_cast; ring
  linarith [ha01.2, hc01.2]


lemma bitLin_zero : bitLin 0 = 1 := by simp [bitLin]

lemma lowMass_one (t : ℕ) : lowMass 1 (t + 1) = 1 := by
  induction t with
  | zero => rw [zero_add, lowMass_succ, lowMass_zero, coeff_one_zero, zero_add]
  | succ t ih => rw [lowMass_succ, ih, coeff_one, if_neg (by omega), add_zero]

lemma lowMass_bitLin_pow (q : ℝ) : ∀ n t, lowMass (bitLin q ^ n) t = 1 - binomSfGe n q t
  | _, 0 => by rw [lowMass_zero, binomSfGe_zero_right, sub_self]
  | 0, t + 1 => by rw [pow_zero, lowMass_one, binomSfGe_zero_left, sub_zero]
  | n + 1, t + 1 => by
    rw [pow_succ', lowMass_bitLin_mul, lowMass_bitLin_pow q n (t + 1), lowMass_bitLin_pow q n t,
      binomSfGe_succ]
    ring

lemma binomSfGe_eq_zero (n : ℕ) (p : ℝ) {j : ℕ} (h : n < j) : binomSfGe n p j = 0 := by
  unfold binomSfGe
  rw [Finset.Icc_eq_empty (by omega), Finset.sum_empty]

/-- `P[Bin(n, q) ≥ c] = P[Bin(n, 1 − q) ≤ n − c]`. -/
lemma binomSfGe_symm (q : ℝ) : ∀ n c, c ≤ n + 1 →
    binomSfGe n q c = 1 - binomSfGe n (1 - q) (n + 1 - c)
  | n, 0, _ => by
    rw [binomSfGe_zero_right, binomSfGe_eq_zero _ _ (by omega), sub_zero]
  | 0, c + 1, h => by
    rw [binomSfGe_zero_left, show 0 + 1 - (c + 1) = 0 by omega, binomSfGe_zero_right, sub_self]
  | n + 1, c + 1, h => by
    rcases Nat.lt_or_ge n c with hnc | hcn
    · rw [binomSfGe_eq_zero _ _ (by omega), show n + 1 + 1 - (c + 1) = 0 by omega,
        binomSfGe_zero_right, sub_self]
    · rw [binomSfGe_succ, binomSfGe_symm q n c (by omega), binomSfGe_symm q n (c + 1) (by omega),
        show n + 1 + 1 - (c + 1) = (n - c) + 1 by omega, binomSfGe_succ,
        show n + 1 - (c + 1) = n - c by omega, show n - c + 1 = n + 1 - c by omega]
      ring

lemma bitPoly_pad_zero (d : ℕ) (M : Multiset ℝ) :
    bitPoly (Multiset.replicate d 0 + M) = bitPoly M := by
  rw [bitPoly_add, bitPoly_replicate, bitLin_zero, one_pow, one_mul]

lemma lowMass_pad_one (M : Multiset ℝ) (t : ℕ) : ∀ d : ℕ,
    lowMass (bitPoly (Multiset.replicate d 1 + M)) (t + d) = lowMass (bitPoly M) t
  | 0 => by rw [Multiset.replicate_zero, zero_add, add_zero]
  | d + 1 => by
    rw [Multiset.replicate_succ, Multiset.cons_add, bitPoly_cons, ← add_assoc,
      lowMass_bitLin_mul, lowMass_pad_one M t d]
    ring

/-- The combinatorial core: rates in `[0, 1]` summing to at least `nq ≥ b + 2`, over at most `n`
bits, put no more mass at or below `b` than `Bin(n, q)`. -/
theorem lowMass_le_binomSfGe (M : Multiset ℝ) (hM : ∀ a ∈ M, 0 ≤ a ∧ a ≤ 1) (n b : ℕ) (q : ℝ)
    (hn : M.card ≤ n) (hq0 : 0 ≤ q) (hq : n * q ≤ M.sum) (hb : (b : ℝ) + 2 ≤ n * q) :
    lowMass (bitPoly M) (b + 1) ≤ 1 - binomSfGe n q (b + 1) := by
  classical
  set M₀ := Multiset.replicate (n - M.card) 0 + M with hM₀
  have hM₀01 : ∀ a ∈ M₀, 0 ≤ a ∧ a ≤ 1 := by
    intro a ha
    rcases Multiset.mem_add.1 ha with h | h
    · rw [Multiset.eq_of_mem_replicate h]; norm_num
    · exact hM a h
  have hcard : M₀.card = n := by
    rw [hM₀, Multiset.card_add, Multiset.card_replicate]; omega
  have hsum : M₀.sum = M.sum := by
    rw [hM₀, Multiset.sum_add, Multiset.sum_replicate, smul_zero, zero_add]
  have hnpos : (0 : ℝ) < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · rw [h] at hb; push_cast at hb; linarith [Nat.cast_nonneg (α := ℝ) b]
    · exact_mod_cast h
  have hm1 : M.sum ≤ n := by
    have h := Multiset.sum_le_card_nsmul M 1 (fun a ha => (hM a ha).2)
    rw [nsmul_eq_mul, mul_one] at h
    exact le_trans h (by exact_mod_cast hn)
  have h := lowMass_le_avg n M.sum b (by linarith) _ M₀ hcard hsum hM₀01 rfl
  rw [hM₀, bitPoly_pad_zero, lowMass_bitLin_pow] at h
  refine le_trans h (sub_le_sub_left (binomSfGe_mono hq0 ?_ ?_ n (b + 1)) 1)
  · rw [div_le_one hnpos]; exact hm1
  · rw [le_div_iff₀ hnpos, mul_comm]; exact hq

lemma natDegree_bitLin_le (a : ℝ) : (bitLin a).natDegree ≤ 1 := by
  unfold bitLin
  refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
  exact (natDegree_C_mul_le _ _).trans (by simp)

lemma natDegree_bitPoly_le (M : Multiset ℝ) : (bitPoly M).natDegree ≤ M.card := by
  induction M using Multiset.induction_on with
  | empty => simp [bitPoly]
  | cons a M ih =>
    rw [bitPoly_cons, Multiset.card_cons]
    exact natDegree_mul_le.trans (by linarith [natDegree_bitLin_le a])

lemma bitLin_one_sub (a : ℝ) : bitLin (1 - a) = reflect 1 (bitLin a) := by
  ext i
  rw [coeff_reflect]
  rcases i with _ | _ | i
  · rw [revAt_le (by norm_num)]
    simp [bitLin, coeff_one]
  · rw [revAt_le (by norm_num)]
    simp [bitLin]
  · rw [revAt_eq_self_of_lt (by omega)]
    simp [bitLin, coeff_one]

/-- Complementing every bit reflects the generating polynomial. -/
lemma bitPoly_map_one_sub (M : Multiset ℝ) :
    bitPoly (M.map fun a => 1 - a) = reflect M.card (bitPoly M) := by
  induction M using Multiset.induction_on with
  | empty => simp [bitPoly]
  | cons a M ih =>
    rw [Multiset.map_cons, bitPoly_cons, bitPoly_cons, ih, bitLin_one_sub, Multiset.card_cons,
      add_comm M.card 1, reflect_mul _ _ (natDegree_bitLin_le a) (natDegree_bitPoly_le M)]

/-- The mass at or above `c` is the complementary bits' mass at or below `|M| − c`. -/
lemma highMass_eq (M : Multiset ℝ) {c : ℕ} (hc : c ≤ M.card) :
    ∑ j ∈ Finset.range (M.card + 1), (if c ≤ j then (bitPoly M).coeff j else 0)
      = lowMass (bitPoly (M.map fun a => 1 - a)) (M.card + 1 - c) := by
  classical
  rw [← Finset.sum_range_reflect]
  have h : ∀ j ∈ Finset.range (M.card + 1),
      (if c ≤ M.card + 1 - 1 - j then (bitPoly M).coeff (M.card + 1 - 1 - j) else 0)
        = if j ≤ M.card - c then (bitPoly (M.map fun a => 1 - a)).coeff j else 0 := by
    intro j hj
    rw [Finset.mem_range] at hj
    rw [bitPoly_map_one_sub, coeff_reflect, revAt_le (by omega),
      show M.card + 1 - 1 - j = M.card - j by omega]
    by_cases h : j ≤ M.card - c
    · rw [if_pos (by omega), if_pos h]
    · rw [if_neg (by omega), if_neg h]
  rw [Finset.sum_congr rfl h, ← Finset.sum_filter, lowMass]
  congr 1
  ext j
  simp only [Finset.mem_filter, Finset.mem_range]
  omega

/-- The upper tail: rates summing to at most `nq ≤ c − 2`, over at most `n` bits, put no more
mass at or above `c` than `Bin(n, q)`. -/
theorem highMass_le_binomSfGe (M : Multiset ℝ) (hM : ∀ a ∈ M, 0 ≤ a ∧ a ≤ 1) (n c : ℕ) (q : ℝ)
    (hn : M.card ≤ n) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hq : M.sum ≤ n * q) (hc : n * q + 2 ≤ c) :
    ∑ j ∈ Finset.range (M.card + 1), (if c ≤ j then (bitPoly M).coeff j else 0)
      ≤ binomSfGe n q c := by
  classical
  by_cases hcM : c ≤ M.card
  · rw [highMass_eq M hcM, ← lowMass_pad_one _ (M.card + 1 - c) (n - M.card),
      show M.card + 1 - c + (n - M.card) = (n - c) + 1 by omega,
      binomSfGe_symm q n c (by omega), show n + 1 - c = n - c + 1 by omega]
    have hnc : ((n - c : ℕ) : ℝ) = n - c := Nat.cast_sub (le_trans hcM hn)
    have hnM : ((n - M.card : ℕ) : ℝ) = n - M.card := Nat.cast_sub hn
    have hsum : (M.map fun a => 1 - a).sum = M.card - M.sum := by
      rw [Multiset.sum_map_sub, Multiset.map_const', Multiset.sum_replicate, nsmul_eq_mul, mul_one,
        Multiset.map_id']
    refine lowMass_le_binomSfGe _ ?_ n (n - c) (1 - q) ?_ (by linarith) ?_ ?_
    · intro a ha
      rcases Multiset.mem_add.1 ha with h | h
      · rw [Multiset.eq_of_mem_replicate h]; norm_num
      · obtain ⟨b, hb, rfl⟩ := Multiset.mem_map.1 h
        have := hM b hb
        constructor <;> linarith
    · rw [Multiset.card_add, Multiset.card_replicate, Multiset.card_map]; omega
    · rw [Multiset.sum_add, Multiset.sum_replicate, nsmul_eq_mul, mul_one, hsum, hnM]
      linarith
    · rw [hnc]; linarith
  · refine le_trans (le_of_eq (Finset.sum_eq_zero fun j hj => ?_)) (binomSfGe_nonneg hq0 hq1 c)
    rw [Finset.mem_range] at hj
    rw [if_neg (by omega)]

section Law

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {ι : Type*}

lemma integrable_of_bit [IsProbabilityMeasure μ] {Y : Ω → ℝ} (hY : Measurable Y)
    (hbit : ∀ᵐ ω ∂μ, Y ω = 0 ∨ Y ω = 1) :
    Integrable Y μ :=
  Integrable.of_mem_Icc 0 1 hY.aemeasurable (by
    filter_upwards [hbit] with ω h
    rcases h with h | h <;> rw [h] <;> norm_num)

lemma measureReal_bit_one {Y : Ω → ℝ} (hY : Measurable Y)
    (hbit : ∀ᵐ ω ∂μ, Y ω = 0 ∨ Y ω = 1) :
    μ.real {ω | Y ω = 1} = μ[Y] := by
  have hmeas : MeasurableSet {ω | Y ω = 1} := measurableSet_eq_fun hY measurable_const
  have hae : Y =ᵐ[μ] ({ω | Y ω = 1} : Set Ω).indicator 1 := by
    filter_upwards [hbit] with ω hω
    rcases hω with h | h
    · simp [h]
    · simp [h]
  rw [integral_congr_ae hae, integral_indicator_one hmeas]

lemma measureReal_bit_zero [IsProbabilityMeasure μ] {Y : Ω → ℝ} (hY : Measurable Y)
    (hbit : ∀ᵐ ω ∂μ, Y ω = 0 ∨ Y ω = 1) :
    μ.real {ω | Y ω = 0} = 1 - μ[Y] := by
  rw [← measureReal_bit_one hY hbit]
  have hmeas : MeasurableSet {ω | Y ω = 1} := measurableSet_eq_fun hY measurable_const
  have hae : ({ω | Y ω = 0} : Set Ω) =ᵐ[μ] ({ω | Y ω = 1}ᶜ : Set Ω) := by
    refine Filter.eventuallyEq_set.2 ?_
    filter_upwards [hbit] with ω hω
    rcases hω with h | h <;> simp [h]
  rw [measureReal_congr hae, measureReal_compl hmeas, probReal_univ]

lemma bit_mean_mem [IsProbabilityMeasure μ] {Y : Ω → ℝ} (hY : Measurable Y)
    (hbit : ∀ᵐ ω ∂μ, Y ω = 0 ∨ Y ω = 1) :
    0 ≤ μ[Y] ∧ μ[Y] ≤ 1 := by
  rw [← measureReal_bit_one hY hbit]
  exact ⟨measureReal_nonneg, measureReal_le_one⟩

lemma measureReal_null_of_ae {A : Set Ω} (h : ∀ᵐ ω ∂μ, ω ∉ A) : μ.real A = 0 := by
  rw [measureReal_def, measure_eq_zero_iff_ae_notMem.2 h, ENNReal.toReal_zero]

/-- The law of a sum of independent bits: below `t` with the generating polynomial's mass there. -/
theorem measureReal_bitSum [IsProbabilityMeasure μ] (X : ι → Ω → ℝ)
    (hmeas : ∀ i, Measurable (X i)) (hind : iIndepFun X μ)
    (hbit : ∀ i, ∀ᵐ ω ∂μ, X i ω = 0 ∨ X i ω = 1) (s : Finset ι) :
    ∀ t : ℕ, μ.real {ω | ∑ i ∈ s, X i ω + 1 ≤ t}
      = lowMass (bitPoly (s.val.map fun i => μ[X i])) t := by
  classical
  induction s using Finset.induction_on with
  | empty =>
    intro t
    simp only [Finset.sum_empty, Finset.empty_val, Multiset.map_zero, zero_add]
    rcases t with _ | t
    · rw [lowMass_zero]
      have : {ω : Ω | (1 : ℝ) ≤ ((0 : ℕ) : ℝ)} = ∅ := by ext ω; simp
      rw [this, measureReal_empty]
    · rw [show bitPoly 0 = 1 by simp [bitPoly], lowMass_one]
      have : {ω : Ω | (1 : ℝ) ≤ ((t + 1 : ℕ) : ℝ)} = Set.univ := by
        ext ω; simp
      rw [this, probReal_univ]
  | insert a s ha ih =>
    intro t
    rw [Finset.insert_val_of_notMem ha, Multiset.map_cons, bitPoly_cons]
    have hsplit : ∀ ω, ∑ i ∈ insert a s, X i ω = X a ω + ∑ i ∈ s, X i ω :=
      fun ω => Finset.sum_insert ha
    simp only [hsplit]
    have hnn : ∀ᵐ ω ∂μ, 0 ≤ X a ω ∧ 0 ≤ ∑ i ∈ s, X i ω := by
      filter_upwards [hbit a, (ae_ball_iff (insert a s).countable_toSet).2
        (fun i _ => hbit i)] with ω h0 hall
      refine ⟨by rcases h0 with h | h <;> simp [h], Finset.sum_nonneg fun i hi => ?_⟩
      rcases hall i (Finset.mem_insert_of_mem hi) with h | h <;> simp [h]
    rcases t with _ | t
    · rw [lowMass_zero]
      refine measureReal_null_of_ae ?_
      filter_upwards [hnn] with ω h hle
      simp only [Nat.cast_zero] at hle
      linarith [h.1, h.2]
    rw [lowMass_bitLin_mul, ← ih (t + 1), ← ih t]
    set T : Ω → ℝ := fun ω => ∑ i ∈ s, X i ω with hT
    have hTmeas : Measurable T := Finset.measurable_sum s fun i _ => hmeas i
    have hindep : IndepFun (X a) T μ := by
      have h := (hind.indepFun_finsetSum_of_notMem hmeas ha).symm
      have e : (∑ j ∈ s, X j) = T := by ext ω; simp [hT, Finset.sum_apply]
      rwa [e] at h
    have hprod : ∀ (A B : Set ℝ), MeasurableSet A → MeasurableSet B →
        μ.real ({ω | X a ω ∈ A} ∩ {ω | T ω ∈ B})
          = μ.real {ω | X a ω ∈ A} * μ.real {ω | T ω ∈ B} := by
      intro A B hA hB
      rw [measureReal_def, measureReal_def, measureReal_def, ← ENNReal.toReal_mul]
      exact congrArg ENNReal.toReal (hindep.measure_inter_preimage_eq_mul A B hA hB)
    have hsets : ({ω | X a ω + T ω + 1 ≤ ((t + 1 : ℕ) : ℝ)} : Set Ω)
        =ᵐ[μ] (({ω | X a ω ∈ ({0} : Set ℝ)} ∩ {ω | T ω ∈ {x : ℝ | x + 1 ≤ ((t + 1 : ℕ) : ℝ)}})
          ∪ ({ω | X a ω ∈ ({1} : Set ℝ)} ∩ {ω | T ω ∈ {x : ℝ | x + 1 ≤ (t : ℝ)}}) : Set Ω) := by
      refine Filter.eventuallyEq_set.2 ?_
      filter_upwards [hbit a] with ω h
      push_cast
      rcases h with h | h
      · simp only [h, Set.mem_union, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_singleton_iff]
        norm_num
      · simp only [h, Set.mem_union, Set.mem_inter_iff, Set.mem_ofPred_eq, Set.mem_singleton_iff]
        norm_num
        constructor <;> intro <;> linarith
    have hdisj : Disjoint
        ({ω | X a ω ∈ ({0} : Set ℝ)} ∩ {ω | T ω ∈ {x : ℝ | x + 1 ≤ ((t + 1 : ℕ) : ℝ)}})
        ({ω | X a ω ∈ ({1} : Set ℝ)} ∩ {ω | T ω ∈ {x : ℝ | x + 1 ≤ (t : ℝ)}}) := by
      rw [Set.disjoint_left]
      rintro ω ⟨h0, -⟩ ⟨h1, -⟩
      simp only [Set.mem_ofPred_eq, Set.mem_singleton_iff] at h0 h1
      rw [h0] at h1; norm_num at h1
    have hB : ∀ c : ℝ, MeasurableSet {x : ℝ | x + 1 ≤ c} := fun c =>
      measurableSet_le (measurable_id.add_const 1) measurable_const
    rw [measureReal_congr hsets, measureReal_union hdisj
      (((hmeas a) (measurableSet_singleton 1)).inter (hTmeas (hB _))),
      hprod _ _ (measurableSet_singleton 0) (hB _), hprod _ _ (measurableSet_singleton 1) (hB _)]
    have e0 : μ.real {ω | X a ω ∈ ({0} : Set ℝ)} = 1 - μ[X a] :=
      measureReal_bit_zero (hmeas a) (hbit a)
    have e1 : μ.real {ω | X a ω ∈ ({1} : Set ℝ)} = μ[X a] :=
      measureReal_bit_one (hmeas a) (hbit a)
    rw [e0, e1]
    rfl

end Law

/-- Hoeffding (1956), for counts at least two from the mean. -/
theorem bitSumTails : BitSumTails := by
  intro Ω _ μ _ ι X s n q hmeas hind hbit hn hq0 hq1
  classical
  set M := s.val.map fun i => μ[X i] with hM
  have hM01 : ∀ a ∈ M, 0 ≤ a ∧ a ≤ 1 := by
    intro a ha
    obtain ⟨i, -, rfl⟩ := Multiset.mem_map.1 ha
    exact bit_mean_mem (hmeas i) (hbit i)
  have hMcard : M.card = s.card := by rw [hM, Multiset.card_map]; rfl
  have hMsum : M.sum = ∑ i ∈ s, μ[X i] := by rw [hM, Finset.sum_eq_multiset_sum]
  refine ⟨fun b hmean hb => ?_, fun c hmean hc => ?_⟩
  · have hset : {ω | ∑ i ∈ s, X i ω ≤ b} = {ω | ∑ i ∈ s, X i ω + 1 ≤ ((b + 1 : ℕ) : ℝ)} := by
      ext ω
      simp only [Set.mem_ofPred_eq]
      push_cast
      constructor <;> intro h <;> linarith
    rw [hset, measureReal_bitSum X hmeas hind hbit s (b + 1)]
    exact lowMass_le_binomSfGe M hM01 n b q (by rw [hMcard]; exact hn) hq0
      (by rw [hMsum]; exact hmean) hb
  set Y : ι → Ω → ℝ := fun i ω => 1 - X i ω with hY
  have hYmeas : ∀ i, Measurable (Y i) := fun i => measurable_const.sub (hmeas i)
  have hYind : iIndepFun Y μ :=
    hind.comp (fun _ x => 1 - x) (fun _ => measurable_const.sub measurable_id)
  have hYbit : ∀ i, ∀ᵐ ω ∂μ, Y i ω = 0 ∨ Y i ω = 1 := fun i => by
    filter_upwards [hbit i] with ω h
    rcases h with h | h <;> simp [hY, h]
  have hYmean : ∀ i, μ[Y i] = 1 - μ[X i] := fun i => by
    simp only [hY]
    rw [integral_sub (integrable_const 1) (integrable_of_bit (hmeas i) (hbit i)), integral_const,
      probReal_univ, one_smul]
  by_cases hcs : c ≤ s.card
  · have hset : {ω | (c : ℝ) ≤ ∑ i ∈ s, X i ω}
        = {ω | ∑ i ∈ s, Y i ω + 1 ≤ ((s.card - c + 1 : ℕ) : ℝ)} := by
      ext ω
      simp only [Set.mem_ofPred_eq, hY, Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
        mul_one]
      rw [Nat.cast_add, Nat.cast_sub hcs]
      push_cast
      constructor <;> intro h <;> linarith
    rw [hset, measureReal_bitSum Y hYmeas hYind hYbit s]
    have hcn : c ≤ n := le_trans hcs hn
    rw [← lowMass_pad_one _ (s.card - c + 1) (n - s.card),
      show s.card - c + 1 + (n - s.card) = (n - c) + 1 by omega,
      binomSfGe_symm q n c (by omega), show n + 1 - c = n - c + 1 by omega]
    have hMY : (s.val.map fun i => μ[Y i] : Multiset ℝ).sum
        = (s.card : ℝ) - ∑ i ∈ s, μ[X i] := by
      rw [← Finset.sum_eq_multiset_sum, Finset.sum_congr rfl fun i _ => hYmean i,
        Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, mul_one]
    have hns : ((n - s.card : ℕ) : ℝ) = n - s.card := Nat.cast_sub hn
    have hnc : ((n - c : ℕ) : ℝ) = n - c := Nat.cast_sub hcn
    refine lowMass_le_binomSfGe _ ?_ n (n - c) (1 - q) ?_ (by linarith) ?_ ?_
    · intro a ha
      rcases Multiset.mem_add.1 ha with h | h
      · rw [Multiset.eq_of_mem_replicate h]; norm_num
      · obtain ⟨i, -, rfl⟩ := Multiset.mem_map.1 h
        exact bit_mean_mem (hYmeas i) (hYbit i)
    · rw [Multiset.card_add, Multiset.card_replicate, Multiset.card_map]
      change n - s.card + s.card ≤ n
      omega
    · rw [Multiset.sum_add, Multiset.sum_replicate, nsmul_eq_mul, mul_one, hMY, hns]
      linarith
    · rw [hnc]
      linarith
  · refine le_trans (le_of_eq (measureReal_null_of_ae ?_)) (binomSfGe_nonneg hq0 hq1 c)
    filter_upwards [(ae_ball_iff s.countable_toSet).2 (fun i _ => hbit i)] with ω hall hge
    have hle : ∑ i ∈ s, X i ω ≤ ∑ _i ∈ s, (1 : ℝ) := Finset.sum_le_sum fun i hi => by
      rcases hall i hi with h | h <;> simp [h]
    rw [Finset.sum_const, nsmul_eq_mul, mul_one] at hle
    have : (s.card : ℝ) < c := by exact_mod_cast not_le.1 hcs
    have hge' : (c : ℝ) ≤ ∑ i ∈ s, X i ω := hge
    linarith

/-- Hoeffding's bound on a binomial's lower tail. -/
theorem one_sub_binomSfGe_le_exp (n b : ℕ) {q γ : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hγ : 0 ≤ γ)
    (hb : (b : ℝ) ≤ n * (q - γ)) : 1 - binomSfGe n q (b + 1) ≤ Real.exp (-2 * n * γ ^ 2) := by
  classical
  set D : Measure Bool := bernoulliMeasure true false ⟨q, hq0, hq1⟩ with hD
  set ν := Measure.pi fun _ : Fin n => D with hν
  have hall : ∀ s : Set (Fin n → Bool), MeasurableSet s := fun s =>
    (Set.to_countable s).measurableSet
  have hDq : D.real {x | x = true} = q :=
    bernoulliMeasure_real_apply_of_mem_of_notMem _ (Set.to_countable _).measurableSet rfl
      (by simp)
  have hcount := pi_count_ge D (fun x => x = true) (Finset.univ : Finset (Fin n)) (b + 1)
  rw [Finset.card_univ, Fintype.card_fin, hDq] at hcount
  have hcount' : ν.real {ω : Fin n → Bool | b + 1 ≤ (Finset.univ.filter fun i => ω i = true).card}
      = binomSfGe n q (b + 1) := by convert hcount
  set Xi : Fin n → (Fin n → Bool) → ℝ := fun i ω => if ω i = true then 1 else 0 with hXi
  have hmarg : ∀ (i : Fin n) (A : Set Bool), ν.real ((fun ω => ω i) ⁻¹' A) = D.real A := by
    intro i A
    rw [measureReal_def, measureReal_def, ← Measure.map_apply (measurable_pi_apply i)
      (Set.to_countable A).measurableSet, (measurePreserving_eval _ i).map_eq]
  have hmean : ∀ i, ν[Xi i] = q := by
    intro i
    have e : Xi i = ({ω | ω i = true} : Set (Fin n → Bool)).indicator 1 := by
      ext ω
      by_cases h : ω i = true <;> simp [hXi, h]
    rw [e, integral_indicator_one (hall _), ← hDq]
    exact hmarg i {x | x = true}
  have hind : iIndepFun Xi ν :=
    (iIndepFun_pi (X := fun _ (x : Bool) => x) fun _ => measurable_id.aemeasurable).comp
      (fun _ (x : Bool) => if x = true then (1 : ℝ) else 0) (fun _ => measurable_of_countable _)
  have hsub : {ω : Fin n → Bool | b + 1 ≤ (Finset.univ.filter fun i => ω i = true).card}ᶜ
      ⊆ {ω | ∑ i ∈ Finset.univ, Xi i ω ≤ ((Finset.univ : Finset (Fin n)).card : ℝ) * (q - γ)} := by
    intro ω hω
    simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_le] at hω
    simp only [Set.mem_ofPred_eq, Finset.card_univ, Fintype.card_fin]
    have hc : ∑ i, Xi i ω = ((Finset.univ.filter fun i => ω i = true).card : ℝ) := by
      simp only [hXi]
      rw [Finset.sum_boole]
    rw [hc]
    have hle : (Finset.univ.filter fun i => ω i = true).card ≤ b := Nat.lt_succ_iff.1 hω
    have : ((Finset.univ.filter fun i => ω i = true).card : ℝ) ≤ b := by exact_mod_cast hle
    linarith
  have h := sumLower_le Xi Finset.univ q γ (fun i => (measurable_of_countable _).aemeasurable)
    hind (fun i => Filter.Eventually.of_forall fun ω => by
      by_cases h : ω i = true <;> simp [hXi, h])
    (by simp [hmean]) hγ
  rw [Finset.card_univ, Fintype.card_fin] at h
  calc 1 - binomSfGe n q (b + 1)
      = ν.real {ω : Fin n → Bool | b + 1 ≤ (Finset.univ.filter fun i => ω i = true).card}ᶜ := by
        rw [measureReal_compl (hall _), probReal_univ, hcount']
    _ ≤ _ := measureReal_mono (by simpa using hsub)
    _ ≤ _ := h

end OrthoDFA
