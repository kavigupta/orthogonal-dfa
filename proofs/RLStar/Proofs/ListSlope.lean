import RLStar.LowLevelClaims.ListSlope
import RLStar.Proofs.QueryBadTail

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

private def aruFin3 : ARU → Fin 3
  | .accept => 0
  | .reject => 1
  | .undecided => 2

private theorem aruFin3_injective : Function.Injective aruFin3 := by
  intro a b h
  cases a <;> cases b <;> simp_all [aruFin3]

private theorem le_one_sub_exp {u : ℝ} (h0 : 0 ≤ u) (h1 : u ≤ 1 / 2) :
    1 * u ≤ 1 - Real.exp (-(2 * u)) := by
  have he : (1 + u) ^ 2 ≤ Real.exp (2 * u) := by
    rw [show 2 * u = u + u by ring, Real.exp_add, sq]
    have := Real.add_one_le_exp u
    exact mul_le_mul (by linarith) (by linarith) (by linarith) (Real.exp_pos u).le
  have hm : Real.exp (-(2 * u)) * Real.exp (2 * u) = 1 := by
    rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
  have h3 : 1 ≤ (1 - u) * Real.exp (2 * u) := by
    nlinarith [mul_le_mul_of_nonneg_left he (by linarith : (0 : ℝ) ≤ 1 - u),
      mul_nonneg h0 (show 0 ≤ 1 - u - u ^ 2 by nlinarith)]
  nlinarith [Real.exp_pos (-(2 * u)), Real.exp_pos (2 * u)]

theorem listSlope_holds : ListSlope := by
  intro Z Ω σ β _ _ _ _ μ _ read st U p ℓ N δ hmeas hind hU hN hℓ hδ
  classical
  have : Countable ARU := aruFin3_injective.countable
  have : MeasurableSingletonClass ARU := ⟨fun _ => trivial⟩
  rcases le_or_gt 1 δ with hδ1 | hδ1
  · exact measureReal_le_one.trans hδ1
  rcases isEmpty_or_nonempty σ with hσ | hσ
  · have : IsEmpty Z := ⟨fun z => hσ.false (st z)⟩
    have hnil : ∀ l : List Z, l = [] := fun
      | [] => rfl
      | w :: _ => isEmptyElim w
    refine (measureReal_mono (s₂ := ∅) fun ω hω => ?_).trans (by simp [hδ.le])
    obtain ⟨T, -, hT⟩ := hω
    rw [hnil (List.filter _ _)] at hT
    simp [Fintype.card_eq_zero] at hT
  set n := Fintype.card σ with hn
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast Fintype.card_pos
  have hl2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hK : 0 < Real.log (n / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ]; linarith)
  set K := Real.log (n / δ)
  let u : σ → ℝ := fun q => max (U q) 0
  let x : σ → ℝ := fun q => (2 * u q * N + K) / Real.log 2
  let B : σ → Z → Set ARU := fun q z => if st z = q then {ARU.undecided} else ∅
  let E : σ → Set Ω := fun q => {ω | U q ≤ 1 / 2 ∧
    x q < ((p.reads fun z => read z ω).filter fun z => read z ω ∈ B q z).card}
  have hE : ∀ q, μ.real (E q) ≤ δ / n := by
    intro q
    by_cases hq : U q ≤ 1 / 2
    swap
    · have : E q = ∅ := Set.eq_empty_of_forall_notMem fun ω hω => hq hω.1
      rw [this, measureReal_empty]
      positivity
    have hu0 : 0 ≤ u q := le_max_right _ _
    have hu1 : u q ≤ 1 / 2 := max_le hq (by norm_num)
    have hB : ∀ z, μ.real {ω | read z ω ∈ B q z} ≤ u q := by
      intro z
      by_cases hz : st z = q
      · simp only [B, hz, if_true, Set.mem_singleton_iff]
        exact (hU z).trans (by rw [hz]; exact le_max_left _ _)
      · simp [B, hz, hu0]
    have h := queryBadTail_holds μ read (B q) (u q) p N ⌈x q⌉₊ 1 2 hmeas hind hu0 hB hN
      zero_le_one zero_le_two (le_one_sub_exp hu0 hu1)
    have hx : 2 * (u q * N) + K ≤ ⌈x q⌉₊ * Real.log 2 := by
      have := Nat.le_ceil (x q)
      rw [div_le_iff₀ hl2] at this
      linarith
    have h2f : Real.exp (2 * (u q * N)) * (n / δ) ≤ (1 + 1 : ℝ) ^ ⌈x q⌉₊ := by
      calc Real.exp (2 * (u q * N)) * (n / δ) = Real.exp (2 * (u q * N) + K) := by
            rw [Real.exp_add, Real.exp_log (by positivity)]
        _ ≤ Real.exp (⌈x q⌉₊ * Real.log 2) := Real.exp_le_exp.mpr hx
        _ = (1 + 1) ^ ⌈x q⌉₊ := by rw [Real.exp_nat_mul, Real.exp_log two_pos]; norm_num
    calc μ.real (E q) ≤ μ.real {ω | ⌈x q⌉₊ ≤
          ((p.reads fun z => read z ω).filter fun z => read z ω ∈ B q z).card} :=
          measureReal_mono fun ω hω => Nat.ceil_le.mpr hω.2.le
      _ ≤ Real.exp (2 * (u q * N)) / (1 + 1) ^ ⌈x q⌉₊ :=
          ENNReal.toReal_le_of_le_ofReal (by positivity) h
      _ ≤ δ / n := by
          rw [div_le_div_iff₀ (by positivity) (by positivity)]
          calc Real.exp (2 * (u q * N)) * n = Real.exp (2 * (u q * N)) * (n / δ) * δ := by
                field_simp
            _ ≤ (1 + 1) ^ ⌈x q⌉₊ * δ := by gcongr
            _ = δ * (1 + 1) ^ ⌈x q⌉₊ := mul_comm _ _
  have hsub : {ω | ∃ T, 0 ≤ T ∧ (2 * n * N * T + n * K) / Real.log 2 <
      ((ℓ (p.eval fun z => read z ω)).filter fun w => U (st w) ≤ T).length} ⊆ ⋃ q, E q := by
    rintro ω ⟨T, hT0, hT⟩
    by_contra hne
    simp only [Set.mem_iUnion, not_exists, E, Set.mem_ofPred_eq, not_and, not_lt] at hne
    obtain ⟨hnd, hmem⟩ := hℓ fun z => read z ω
    refine absurd hT (not_lt.mpr ?_)
    rcases lt_or_ge T (1 / 2) with hT2 | hT2
    · set s := ((ℓ (p.eval fun z => read z ω)).filter fun w => U (st w) ≤ T).toFinset
      rw [← List.toFinset_card_of_nodup (hnd.filter _),
        Finset.card_eq_sum_card_fiberwise (f := st) (t := Finset.univ)
          fun _ _ => Finset.mem_coe.mpr (Finset.mem_univ _)]
      push_cast
      have hy : ∀ q, ((s.filter fun w => st w = q).card : ℝ) ≤ (2 * T * N + K) / Real.log 2 := by
        intro q
        by_cases hq : U q ≤ T
        · calc ((s.filter fun w => st w = q).card : ℝ)
              ≤ ((p.reads fun z => read z ω).filter fun z => read z ω ∈ B q z).card := by
                exact_mod_cast Finset.card_le_card fun w hw => by
                  simp only [s, Finset.mem_filter, List.mem_toFinset, List.mem_filter,
                    decide_eq_true_eq] at hw
                  obtain ⟨⟨hwl, -⟩, hwq⟩ := hw
                  obtain ⟨hr, hund⟩ := hmem w hwl
                  simp [B, hwq, hr, hund]
            _ ≤ x q := hne q (hq.trans hT2.le)
            _ ≤ _ := by
                simp only [x, u]
                gcongr
                exact max_le hq hT0
        · have : s.filter (fun w => st w = q) = ∅ := by
            ext w
            simp only [s, Finset.mem_filter, List.mem_toFinset, List.mem_filter,
              decide_eq_true_eq, Finset.notMem_empty, iff_false, not_and]
            rintro ⟨-, hwT⟩ rfl
            exact hq hwT
          rw [this, Finset.card_empty, Nat.cast_zero]
          positivity
      calc ∑ q, ((s.filter fun w => st w = q).card : ℝ)
          ≤ ∑ _q : σ, (2 * T * N + K) / Real.log 2 := Finset.sum_le_sum fun q _ => hy q
        _ = _ := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring
    · have hlen : ((ℓ (p.eval fun z => read z ω)).filter fun w => U (st w) ≤ T).length ≤ N := by
        calc _ ≤ (ℓ (p.eval fun z => read z ω)).length := List.length_filter_le _ _
          _ = (ℓ (p.eval fun z => read z ω)).toFinset.card :=
              (List.toFinset_card_of_nodup hnd).symm
          _ ≤ (p.reads fun z => read z ω).card :=
              Finset.card_le_card fun w hw => (hmem w (List.mem_toFinset.mp hw)).1
          _ ≤ N := hN _
      have hl1 : Real.log 2 ≤ 1 := by
        have := Real.log_le_sub_one_of_pos two_pos
        linarith
      rw [le_div_iff₀ hl2]
      calc (((ℓ (p.eval fun z => read z ω)).filter fun w => U (st w) ≤ T).length : ℝ)
            * Real.log 2 ≤ N * 1 :=
            mul_le_mul (by exact_mod_cast hlen) hl1 hl2.le (Nat.cast_nonneg _)
        _ ≤ 2 * n * N * T + n * K := by
            nlinarith [mul_nonneg (Nat.cast_nonneg N) (show (0 : ℝ) ≤ 2 * n * T - 1 by nlinarith),
              mul_nonneg (by linarith : (0 : ℝ) ≤ n) hK.le]
  calc _ ≤ μ.real (⋃ q, E q) := measureReal_mono hsub
    _ ≤ ∑ q, μ.real (E q) := measureReal_iUnion_fintype_le _
    _ ≤ ∑ _q : σ, δ / n := Finset.sum_le_sum fun q _ => hE q
    _ = δ := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← hn]
        field_simp

end OrthoDFA
