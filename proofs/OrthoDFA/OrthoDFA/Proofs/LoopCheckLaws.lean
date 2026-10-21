import OrthoDFA.Proofs.LoopCheckCore
import OrthoDFA.Proofs.LoopSimple
import OrthoDFA.Proofs.UniformStrings

/-!
# The laws of the checked loop at explicit settings

The window's strings are at least `k` long, so its atoms and its prefix mass at `k` are at most
`|Σ|^(−k)`; it gives each covered state `c/N` and each set at least `1/N` of what the strings of
length `L` give it. Conditioned off a set holding at most half of it, a law at most doubles its
atoms. The check's silent base at a share of undecided strings well under its rate is at most
`1 − 3ε/16`, and its levels' inverses sum to `8 S (S + 1) (S + 2) / (3 δ)`.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α]

lemma windowStrings_singleton_le {k L : ℕ} (hkL : k ≤ L) (t : FreeMonoid α) :
    (windowStrings k L).real {t} ≤ 1 / (Fintype.card α : ℝ) ^ k := by
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [windowStrings_real hkL, div_le_iff₀ (by
    have : (k : ℝ) ≤ L := by exact_mod_cast hkL
    linarith)]
  calc ∑ n ∈ Finset.Icc k L, (uniformStrings n).real {t}
      ≤ ∑ _n ∈ Finset.Icc k L, 1 / (Fintype.card α : ℝ) ^ k :=
        Finset.sum_le_sum fun n hn => (uniformStrings_singleton_le n t).trans
          (one_div_le_one_div_of_le (by positivity) (pow_le_pow_right₀ hA (Finset.mem_Icc.1 hn).1))
    _ = 1 / (Fintype.card α : ℝ) ^ k * ((L : ℝ) + 1 - k) := by
        rw [Finset.sum_const, Nat.card_Icc, nsmul_eq_mul, mul_comm]
        congr 1
        rw [Nat.cast_sub (by omega)]
        push_cast
        ring

lemma uniformStrings_le_window {k L : ℕ} (hkL : k ≤ L) (S : Set (FreeMonoid α)) :
    (uniformStrings L).real S ≤ ((L : ℝ) + 1 - k) * (windowStrings k L).real S := by
  have hN : (0 : ℝ) < (L : ℝ) + 1 - k := by
    have : (k : ℝ) ≤ L := by exact_mod_cast hkL
    linarith
  rw [windowStrings_real hkL, mul_div_cancel₀ _ hN.ne']
  exact Finset.single_le_sum (f := fun n => (uniformStrings n).real S)
    (fun _ _ => measureReal_nonneg) (Finset.mem_Icc.2 ⟨hkL, le_rfl⟩)

lemma window_covered {σ : Type*} (M : _root_.DFA α σ) {L : ℕ} {c : ℝ} (hk : (L + 1) / 2 ≤ L)
    {q : σ} (hq : q ∈ coveredStates M L c) :
    c / ((L : ℝ) + 1 - ((L + 1) / 2 : ℕ)) ≤ (windowStrings ((L + 1) / 2) L).real
      {z | M.eval z.toList = q} := by
  have hN : (0 : ℝ) < (L : ℝ) + 1 - ((L + 1) / 2 : ℕ) := by
    have : (((L + 1) / 2 : ℕ) : ℝ) ≤ L := by exact_mod_cast hk
    linarith
  rw [windowStrings_real hk]
  exact div_le_div_of_nonneg_right (occupancy_of_covered M hq) hN.le

omit [Nonempty α] in
lemma offLaw_singleton_le {ν : Measure (FreeMonoid α)} [IsProbabilityMeasure ν]
    {T : Finset (FreeMonoid α)} (hT : ν.real ↑T ≤ 1 / 2) (t : FreeMonoid α) :
    (offLaw ν T).real {t} ≤ 2 * ν.real {t} := by
  rw [offLaw_real (compl_ne_zero hT), div_le_iff₀ (by linarith [half_le_compl hT])]
  have h1 : ν.real ({t} ∩ (↑T)ᶜ) ≤ ν.real {t} :=
    measureReal_mono Set.inter_subset_left (measure_ne_top _ _)
  have h2 : 0 ≤ ν.real {t} := measureReal_nonneg
  nlinarith [half_le_compl hT]

/-- At a share of undecided strings at most `5/32` of the check's rate, its silent base is at most
`1 − 3 rate/16`. -/
lemma silentBase_le (K : CheckSettings) {q : ℝ} (hr0 : 0 < K.rate) (hr1 : K.rate ≤ 1 / 2)
    (hq0 : 0 ≤ q) (hq : q ≤ 5 / 32 * K.rate) :
    0 ≤ K.silentBase q ∧ K.silentBase q ≤ 1 - 3 / 16 * K.rate := by
  unfold CheckSettings.silentBase
  set r := K.rate
  have e : r / (r / 2) = 2 := by field_simp
  have hb : (1 - r) / (1 - r / 2) ≤ 1 - r / 2 := by
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have hb0 : 0 ≤ (1 - r) / (1 - r / 2) := div_nonneg (by linarith) (by linarith)
  rw [e]
  have h1 : 0 ≤ 1 - q := by linarith
  refine ⟨by positivity, ?_⟩
  have : (1 - q) * ((1 - r) / (1 - r / 2)) ≤ 1 - r / 2 := by
    calc (1 - q) * ((1 - r) / (1 - r / 2)) ≤ 1 * ((1 - r) / (1 - r / 2)) :=
          mul_le_mul_of_nonneg_right (by linarith) hb0
      _ ≤ 1 - r / 2 := by linarith
  linarith

lemma sum_inv_level (K : CheckSettings) (hδ : 0 < K.δ) :
    ∀ S : ℕ, ∑ j ∈ Finset.range S, 1 / K.level j = 8 * S * (S + 1) * (S + 2) / (3 * K.δ)
  | 0 => by simp
  | S + 1 => by
    rw [Finset.sum_range_succ, sum_inv_level K hδ S]
    unfold CheckSettings.level
    push_cast
    field_simp
    ring

/-- At the check's size, its silent tails sum to at most `δ/32`. -/
lemma check_tail_le (K : CheckSettings) {q : ℝ} (hr0 : 0 < K.rate) (hr1 : K.rate ≤ 1 / 2)
    (hq0 : 0 ≤ q) (hq : q ≤ 5 / 32 * K.rate) (hδ : 0 < K.δ) (S : ℕ) (hS : 1 ≤ S)
    (hn : 16 / (3 * K.rate) * Real.log (86 * S * (S + 1) * (S + 2) / K.δ ^ 2) ≤ K.size) :
    ∑ j ∈ Finset.range S, K.silentBase q ^ K.size / K.level j ≤ K.δ / 32 := by
  obtain ⟨hb0, hb⟩ := silentBase_le K hr0 hr1 hq0 hq
  set r := K.rate
  set X : ℝ := 86 * S * (S + 1) * (S + 2) / K.δ ^ 2
  have hS1 : (1 : ℝ) ≤ S := by exact_mod_cast hS
  have hX0 : 0 < X := by positivity
  have hpow : K.silentBase q ^ K.size ≤ 1 / X := by
    calc K.silentBase q ^ K.size ≤ (1 - 3 / 16 * r) ^ K.size := pow_le_pow_left₀ hb0 hb _
      _ ≤ Real.exp (-(3 / 16 * r)) ^ K.size := by
          refine pow_le_pow_left₀ (by linarith) ?_ _
          have := Real.add_one_le_exp (-(3 / 16 * r))
          linarith
      _ = Real.exp (-(3 / 16 * r * K.size)) := by
          rw [← Real.exp_nat_mul]; ring_nf
      _ ≤ Real.exp (-Real.log X) := by
          refine Real.exp_le_exp.2 ?_
          have : Real.log X ≤ 3 / 16 * r * K.size := by
            have h := mul_le_mul_of_nonneg_left hn (show 0 ≤ 3 / 16 * r by positivity)
            have e : 3 / 16 * r * (16 / (3 * r) * Real.log X) = Real.log X := by
              field_simp
            linarith
          linarith
      _ = 1 / X := by rw [Real.exp_neg, Real.exp_log hX0, one_div]
  calc ∑ j ∈ Finset.range S, K.silentBase q ^ K.size / K.level j
      = K.silentBase q ^ K.size * ∑ j ∈ Finset.range S, 1 / K.level j := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun j _ => by ring
    _ ≤ 1 / X * (8 * S * (S + 1) * (S + 2) / (3 * K.δ)) := by
        rw [sum_inv_level K hδ S]
        exact mul_le_mul_of_nonneg_right hpow (by positivity)
    _ ≤ K.δ / 32 := by
        simp only [X]
        rw [div_div_eq_mul_div, one_mul]
        field_simp
        nlinarith [sq_nonneg K.δ, mul_pos hδ hδ]

end OrthoDFA
