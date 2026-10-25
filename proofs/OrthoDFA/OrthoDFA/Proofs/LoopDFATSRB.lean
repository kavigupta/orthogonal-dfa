import OrthoDFA.Proofs.TSRBridge

/-!
# The learner's DFA under typical-start replication, probes to twice the length: the proof

`loop_simple_checks` at the window `⌈L/2⌉` to `2L + |Q| + 3` leaves a last round as in
`learner_loop_dfa_simple_holds`, with the misled probes counted among each state's strings shorter
than `2L + 1`. `dfa_of_checks_tsrb` gives a leaf misjudging at most
`γ + N(2(|Σ| + 1 + |Q|)(ε + ε/96) + (1025/512 + 3/8)εcov)/m`, at most `γ + ξ/2` less the pick's
misreads (`tsrb_harm`), and the pick gets `ξ/2` as in `learner_loop_dfa_simple_holds`. The closed
form `L₀` bounds `log ℓ` for `ℓ = 2L + |Q| + 4` (`dfa_pow_twice`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

/-- The claim's rates, from the start's visit rate `m`. -/
lemma tsrb_rates {Q A N ℓ m ξ ε εcov ι : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hN : 1 ≤ N)
    (hℓ : ℓ ≤ 2 * N) (hm : 0 < m) (hm1 : m ≤ 1) (hξ : 0 < ξ) (hξ1 : ξ ≤ 1)
    (he : ε = ξ * m / (5 * (A + Q + 3) * N)) (hεc : εcov = min ε (ξ / (8 * N)))
    (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1)))) :
    0 < ε ∧ ε ≤ 1 / 2 ∧ 0 < εcov ∧ εcov ≤ ε ∧ 0 < ι ∧ ι ≤ εcov / 8 ∧ ℓ * ε < 3 / 8 := by
  have hP : 0 < 5 * (A + Q + 3) := by positivity
  have he0 : 0 < ε := by rw [he]; positivity
  have hεN : ε * N ≤ 1 / 30 := by
    have e : ε * N = ξ * m / (5 * (A + Q + 3)) := by rw [he]; field_simp
    rw [e, div_le_iff₀ hP]
    have : ξ * m ≤ 1 := by nlinarith
    nlinarith
  have hε30 : ε ≤ 1 / 30 := by nlinarith
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min he0 (by positivity)
  have hεcε : εcov ≤ ε := by rw [hεc]; exact min_le_left _ _
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := by rw [hι]; exact min_le_left _ _
  refine ⟨he0, by linarith, hεc0, hεcε, hι0, hι8, ?_⟩
  have : ℓ * ε ≤ 2 * N * ε := mul_le_mul_of_nonneg_right hℓ he0.le
  nlinarith

/-- The wrong states' share of the start's runs, and the pick's misreads, cost at most `ξ/2`. -/
lemma tsrb_harm {A Q N m ξ ε εcov : ℝ} (hA : 0 ≤ A) (hQ : 0 ≤ Q) (hN : 0 < N) (hm : 0 < m)
    (hm1 : m ≤ 1) (he : ε = ξ * m / (5 * (A + Q + 3) * N)) (hεc0 : 0 ≤ εcov)
    (hεcε : εcov ≤ ε) :
    N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
        + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)) / m
      + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) ≤ ξ / 2 := by
  have hP : 0 < 5 * (A + Q + 3) := by positivity
  have hu : N * ε / m = ξ / (5 * (A + Q + 3)) := by rw [he]; field_simp
  have hu0 : 0 ≤ N * ε := mul_nonneg hN.le (hεc0.trans hεcε)
  have hv : N * εcov ≤ N * ε := mul_le_mul_of_nonneg_left hεcε hN.le
  have hv0 : 0 ≤ N * εcov := mul_nonneg hN.le hεc0
  have huu : N * ε ≤ N * ε / m := le_div_self hu0 hm hm1
  set c : ℝ := 2 * (1025 / 1024) + 3 / 8 with hcd
  have e1 : N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
        + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov))
      = 97 / 48 * (A + Q + 1) * (N * ε) + c * (N * εcov) := by rw [hcd]; ring
  have e2 : 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) = c * (N * εcov) := by
    rw [hcd]; ring
  rw [e1, add_assoc, e2]
  have h1 : (97 / 48 * (A + Q + 1) * (N * ε) + c * (N * εcov)) / m
      ≤ (97 / 48 * (A + Q + 1) + c) * (N * ε / m) := by
    rw [mul_div_assoc', div_le_div_iff_of_pos_right hm]
    nlinarith
  have h2 : c * (N * εcov) ≤ c * (N * ε / m) :=
    mul_le_mul_of_nonneg_left (hv.trans huu) (by rw [hcd]; norm_num)
  have h3 : (97 / 48 * (A + Q + 1) + c) * (N * ε / m) + c * (N * ε / m)
      = (97 / 48 * (A + Q + 1) + 2 * c) * (ξ / (5 * (A + Q + 3))) := by rw [hu]; ring
  have hξ0 : 0 ≤ ξ / (5 * (A + Q + 3)) := by
    rw [← hu]; exact div_nonneg hu0 hm.le
  have h4 : (97 / 48 * (A + Q + 1) + 2 * c) * (ξ / (5 * (A + Q + 3)))
      ≤ 5 / 2 * (A + Q + 3) * (ξ / (5 * (A + Q + 3))) :=
    mul_le_mul_of_nonneg_right (by rw [hcd]; nlinarith) hξ0
  have h5 : 5 / 2 * (A + Q + 3) * (ξ / (5 * (A + Q + 3))) = ξ / 2 := by field_simp
  linarith

/-- `6 log ℓ` lies under its tangent at `T`, so `L ≥ L₀` gives `6 log ℓ + log E ≤ (L/2) log A`
for `ℓ = 2L + Q + 4`. -/
lemma dfa_pow_twice {A ℓ E L Q T : ℝ} {k : ℕ} (hA : 2 ≤ A) (hℓ : ℓ = 2 * L + Q + 4)
    (hQ : 0 ≤ Q) (hL : 0 ≤ L) (hE : 1 ≤ E) (hk : L ≤ 2 * k)
    (hT : T = 4 * (Real.log E + 12) / Real.log A + Q + 4)
    (hL₀ : (2 * T * (Real.log E + 6 * Real.log T - 6) + 12 * (Q + 4))
      / (T * Real.log A - 24) ≤ L) :
    ℓ ^ 6 * E ≤ A ^ k := by
  have ha : 0 < Real.log A := Real.log_pos (by linarith)
  have hx : 0 ≤ Real.log E := Real.log_nonneg hE
  have hTa : T * Real.log A = 4 * (Real.log E + 12) + (Q + 4) * Real.log A := by
    rw [hT]; field_simp; ring
  have hT0 : 0 < T := by rw [hT]; positivity
  have hd : 0 < T * Real.log A - 24 := by nlinarith
  have hℓ0 : 0 < ℓ := by rw [hℓ]; linarith
  have htan : Real.log ℓ ≤ Real.log T + ℓ / T - 1 := by
    have := Real.log_le_sub_one_of_pos (div_pos hℓ0 hT0)
    rw [Real.log_div hℓ0.ne' hT0.ne'] at this
    linarith
  have h1 : 2 * T * (Real.log E + 6 * Real.log T - 6) + 12 * (Q + 4)
      ≤ L * (T * Real.log A - 24) := by
    rwa [div_le_iff₀ hd] at hL₀
  have h4 : T * Real.log ℓ ≤ T * Real.log T + ℓ - T := by
    have h := mul_le_mul_of_nonneg_left htan hT0.le
    have e : T * (Real.log T + ℓ / T - 1) = T * Real.log T + ℓ - T := by field_simp
    linarith
  have h5 : T * (2 * (6 * Real.log ℓ + Real.log E)) ≤ T * (L * Real.log A) := by
    linarith
  have h3 : 2 * (6 * Real.log ℓ + Real.log E) ≤ L * Real.log A := le_of_mul_le_mul_left h5 hT0
  have h6 : Real.log (ℓ ^ 6 * E) ≤ Real.log (A ^ k) := by
    rw [Real.log_mul (pow_pos hℓ0 6).ne' (by linarith), Real.log_pow, Real.log_pow]
    have : L * Real.log A ≤ 2 * k * Real.log A := mul_le_mul_of_nonneg_right hk ha.le
    push_cast
    linarith
  exact (Real.log_le_log_iff (by positivity) (by positivity)).1 h6

/-- `dfa_tail` with the probes to `2L + Q + 3`. -/
lemma tsrb_tail {A ℓ ε cap aε E₁ E E₂ E₃ lg Q n J RB R₀ B₃ K₁ δ δ' c s pAP Θ P₀ L T : ℝ}
    {k : ℕ} (hA : 2 ≤ A) (hQ : 0 ≤ Q) (hL : 0 ≤ L) (hc : 0 < c) (hc1 : c ≤ 1) (hs0 : 0 < s)
    (hs1 : s ≤ 1 / 2) (hpAP : 0 < pAP) (hδ'0 : 0 < δ') (hδ : 0 < δ) (hΘ1 : 1 ≤ Θ)
    (hR₀ : 0 ≤ R₀) (hP₀ : 0 ≤ P₀) (hlR₀ : 0 ≤ Real.logb 2 (R₀ + 1)) (hK₁0 : 0 ≤ K₁)
    (hn : 0 ≤ n) (hℓd : ℓ = 2 * L + Q + 4) (hℓ1 : 1 ≤ ℓ) (hε0 : 0 < ε)
    (hinvε : 1 / ε ≤ aε * ℓ) (hcap : 1 / (E₁ * ℓ ^ 5) ≤ cap) (hlg : 0 ≤ lg)
    (haεd : aε = 4 * (A + 1) / c)
    (hE₁d : E₁ = (Q + 3) ^ 2 * (2 ^ 20 * Θ * (R₀ + 1) / s
      + 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' + 2 ^ 16 * (P₀ + 2) / (pAP * δ')))
    (hEd : E = 1536 * aε * E₁ + 1536 * lg * aε)
    (hE₂d : E₂ = 4 * (Q + 1) * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ)
    (hE₃d : E₃ = 16 * K₁ ^ 2 / δ') (hJ : J = Q + 2) (hRB0 : 0 ≤ RB) (hRB : RB ≤ R₀ * ℓ ^ 3)
    (hB₃0 : 0 ≤ B₃) (hB₃ : B₃ ≤ K₁ * ℓ) (hk : L ≤ 2 * k)
    (hT : T = 4 * (Real.log (E + E₂ + E₃) + 12) / Real.log A + Q + 4)
    (hL₀ : (2 * T * (Real.log (E + E₂ + E₃) + 6 * Real.log T - 6) + 12 * (Q + 4))
      / (T * Real.log A - 24) ≤ L) :
    (A ^ k)⁻¹ ≤ cap * ε / 1536 ∧ 1536 * lg * (A ^ k)⁻¹ ≤ ε
      ∧ (Q + 1) * n * (6 * J * RB + B₃ + n) * (A ^ k)⁻¹ ≤ δ / 4
      ∧ 16 * B₃ ^ 2 * (A ^ k)⁻¹ ≤ δ' := by
  have haε : 1 ≤ aε := by
    rw [haεd, le_div_iff₀ hc, one_mul]
    linarith only [hA, hc1]
  have haε0 : 0 < aε := by linarith only [haε]
  have hE₁ : 1 ≤ E₁ := by
    rw [hE₁d]
    have h1 : 1 ≤ 2 ^ 20 * Θ * (R₀ + 1) / s := by
      rw [le_div_iff₀ hs0, one_mul]
      have : 1 ≤ Θ * (R₀ + 1) := one_le_mul_of_one_le_of_one_le hΘ1 (by linarith only [hR₀])
      linarith only [this, hs1]
    have h2 : 0 ≤ 2 ^ 36 * Θ ^ 2 * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ' := by positivity
    have h3 : 0 ≤ 2 ^ 16 * (P₀ + 2) / (pAP * δ') := by positivity
    have h4 : (1 : ℝ) ≤ (Q + 3) ^ 2 := one_le_pow₀ (by linarith only [hQ])
    exact one_le_mul_of_one_le_of_one_le h4 (by linarith only [h1, h2, h3])
  have hE₂0 : 0 ≤ E₂ := by rw [hE₂d]; positivity
  have hE₃0 : 0 ≤ E₃ := by rw [hE₃d]; positivity
  have hEE : 1 ≤ E + E₂ := by
    have h1 : 0 ≤ 1536 * lg * aε := by positivity
    have h2 : 1 ≤ 1536 * aε * E₁ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by norm_num) haε) hE₁
    rw [hEd]
    linarith only [h1, h2, hE₂0]
  have hpow3 : ℓ ^ 6 * (E + E₂ + E₃) ≤ A ^ k :=
    dfa_pow_twice hA hℓd hQ hL (by linarith only [hEE, hE₃0]) hk hT hL₀
  have hℓ6 : 0 ≤ ℓ ^ 6 := pow_nonneg (by linarith only [hℓ1]) 6
  have hpow : ℓ ^ 6 * (E + E₂) ≤ A ^ k :=
    (mul_le_mul_of_nonneg_left (by linarith only [hE₃0]) hℓ6).trans hpow3
  have hAk : 0 < A ^ k := pow_pos (by linarith only [hA]) k
  obtain ⟨h1, h2, h3⟩ := dfa_final (Ak := A ^ k) hAk hℓ1 hε0 hinvε haε0 hcap
    (by linarith only [hE₁]) hlg hEd hE₂d hQ hn hδ hJ hRB0 hRB hB₃0 hB₃ hpow
  exact ⟨h1, h2, h3, dfa_noise_tail hℓ1 hB₃0 hB₃ hE₃d hδ'0 hEE hpow3 hAk⟩

set_option maxHeartbeats 8000000 in -- the claim's settings meet `loop_simple_checks`'s conditions
theorem learner_loop_dfa_tsrb_simple_holds : LearnerLoopDFATSRBSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ m γ δ pAP ξ Q A k L₁ N ε εcov δ' ι s τ J S B₃ Wm W PB RB
    cap Kc n hL hη hη₀ hm hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hstart hk1 hk2 hk3 hk4
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ 2 * L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - k := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  obtain ⟨q₀, hq₀w, hq₀e⟩ := hstart
  have hm1 : m ≤ 1 := hq₀w.trans measureReal_le_one
  have hwN : 2 + (Nat.clog 2 L₁ : ℝ) ≤ 2 * N := by
    have h1 : 1 ≤ L₁ := by change 1 ≤ 2 * L + Fintype.card σ + 3; omega
    have h2 : Nat.clog 2 L₁ ≤ L₁ - 1 :=
      Nat.clog_le_of_le_pow (by have := (L₁ - 1).lt_two_pow_self; omega)
    have : 2 + Nat.clog 2 L₁ ≤ L₁ + 1 := by omega
    have : 2 + (Nat.clog 2 L₁ : ℝ) ≤ L₁ + 1 := by exact_mod_cast this
    have h3 : (2 * k : ℕ) ≤ L₁ + 1 := by
      show 2 * ((L + 1) / 2) ≤ 2 * L + Fintype.card σ + 3 + 1; omega
    have : (2 : ℝ) * k ≤ L₁ + 1 := by exact_mod_cast h3
    rw [hNe]; linarith
  obtain ⟨hε0, hε12, hεc0, hεcε, hι0, hι8, hιL⟩ :=
    tsrb_rates (ℓ := 2 + (Nat.clog 2 L₁ : ℝ)) hQ1 hA2' hN1 hwN hm hm1 hξ hξ1 rfl rfl rfl
  have hεcov1 : εcov ≤ 1 := by linarith
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1' : δ' < 1 := by
    show δ / (8 * (Q + 1)) < 1
    rw [div_lt_one (by positivity)]
    linarith
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith)
  have hlogK : 2 ≤ Real.log (16 * (Q + 1) * Kc / δ') := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    have he := Real.exp_one_lt_d9
    have : Real.exp 2 ≤ 16 := by
      calc Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
        _ ≤ 3 * 3 := by gcongr <;> linarith
        _ ≤ 16 := by norm_num
    refine this.trans ?_
    rw [le_div_iff₀ hδ'0]
    nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)]
  -- the walk starts after at least one letter
  have hk0 : 1 ≤ k := by
    by_contra h
    have hk : k = 0 := by omega
    rw [hk, pow_zero, inv_one, mul_one] at hk2
    linarith
  have hkL : k ≤ L := by
    have : (L + 1) / 2 ≤ L ∨ L = 0 := by omega
    rcases this with h | h
    · exact h
    · exfalso; have : k = 0 := by show (L + 1) / 2 = 0; omega
      omega
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  have hpm : prefixMax P.D k ≤ (A ^ k)⁻¹ := by
    have := prefixMax_windowStrings_le (α := α) (le_refl k) hkL₁
    rwa [one_div] at this
  have : IsProbabilityMeasure P.D := windowStrings_isProbability hkL₁
  have hpm0 : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hmain⟩ :=
    loop_simple_checks M O P (shortAt M (2 * L + 1)) η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1' hpAP
    (by show 1 ≤ 2 * L + Fintype.card σ + 3; omega)
    hk0 hιL (hpm.trans hk1)
    ((mul_le_mul_of_nonneg_left hpm (mul_nonneg (by norm_num) (by linarith [hlogK]))).trans hk2)
    ((mul_le_mul_of_nonneg_left hpm (by positivity)).trans hk4)
  refine ⟨settings, hwalk, hdis, hbud, ?_⟩
  have hcap0 : 0 < cap := by
    have h1 : 0 < (A ^ k)⁻¹ := by positivity
    by_contra hc0
    push Not at hc0
    have : cap * ε / 1536 ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hc0 hε0.le
      linarith
    linarith
  have hatom : ∀ v : FreeMonoid α,
      (uniformStrings (L₁ + Fintype.card σ + 3)).real {v} ≤ cap := fun v => by
    refine (uniformStrings_singleton_le _ v).trans ?_
    have h1 : 1 / A ^ (L₁ + Fintype.card σ + 3) ≤ (A ^ k)⁻¹ := by
      rw [one_div]
      exact inv_anti₀ (by positivity) (pow_le_pow_right₀ (by linarith) (by omega))
    refine h1.trans (hk1.trans ?_)
    have : cap * ε ≤ cap * 1 := mul_le_mul_of_nonneg_left (by linarith) hcap0.le
    linarith
  set Dsf : Measure (FreeMonoid α) := uniformStrings (L₁ + Fintype.card σ + 3) with hDsfd
  have hD := hmain Dsf inferInstance (uniformStrings_length _) hpAPu hatom
  set T := settings.level3.probeBudget
  set μL := loopMeasure μ α (Fintype.card σ + 1) T
  set β := Measure.pi fun _ : Fin n => uniformStrings (α := α) L
  set Nn := P.L + T + 2
  set Ls := P.L + Fintype.card σ + 3
  set t : ℝ := ξ / 4 with htd
  set Bloop : Set (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T)) :=
    {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96)
            (shortAt M (2 * L + 1)) (ε + ε / 96))
        ∧ RoundsFresh settings O P.D Dsf Nn Ls p)} with hBloopd
  set Sel : ℕ → Set ((Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
      × (Fin n → FreeMonoid α)) := fun r =>
    if hr : r < Fintype.card σ + 1 then
      selFail settings O P.D Dsf Nn Ls L n (Fintype.card σ + 2) t r hr else ∅
    with hSeld
  -- the loop's last round, consistent and fresh, picks well unless its scores stray or a
  -- held-out string collides
  have hsub : {q : (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
        × (Fin n → FreeMonoid α) |
      let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
            o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
              (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ⊆ (Prod.fst ⁻¹' Bloop) ∪ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Sel r := by
    intro q hq
    by_cases hlp : q.1 ∈ Bloop
    · exact Or.inl hlp
    right
    have hlp' := not_not.1 hlp
    obtain ⟨⟨hlen, o, ho, hch⟩, hfr⟩ := hlp'
    have hcons : o.result.ending = .consistent := hch.1.1
    obtain ⟨r, hr, -, tt, hst, hG, hOk, hB, hF, hread, hhyp⟩ :=
      stageSel_last settings O P Dsf Ls q.1 hfr o ho hcons
    refine Set.mem_iUnion₂.2 ⟨r, Finset.mem_range.2 hr, ?_⟩
    simp only [hSeld, dif_pos hr]
    by_contra hnot
    apply hq
    refine ⟨hlen, o, ho, hcons, ?_⟩
    have hnot' : (q.1.1, q.2) ∉ SelBad O L n (Fintype.card σ + 2) t (some (tt, resCode o.result))
        ∧ (q.1.2, q.2) ∈ SelDk settings P.D L n r hr (some (tt, resCode o.result)) := by
      constructor
      · intro h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inl h⟩
      · by_contra h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inr h⟩
    obtain ⟨hnb, hdk⟩ := hnot'
    obtain ⟨hinj, -⟩ := hdk
    have hsf : SuffixFree o.family := by
      have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
        hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
      intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    have hleaves : o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 :=
      (Random.classSet_props _ _ hch.2.2.1).2
    have hgood : (q.1.1, q.2) ∉ scoreBad O o.result.hypothesis o.family o.state.lo
        (o.state.hi + 1) L n t := by
      intro h
      apply hnb
      show (q.1.1, q.2) ∈ (if SuffixFree tt.2.2.1
          ∧ (resCode o.result).hyp.tree.leaves.length ≤ Fintype.card σ + 2
        then scoreBad O (resCode o.result).hyp tt.2.2.1 tt.2.1.lo (tt.2.1.hi + 1) L n t else ∅)
      rw [hF, hB, hhyp, if_pos ⟨hsf, hleaves⟩]
      exact h
    -- the start's true leaf, and the family's reads on the strings of length `L`
    have hs := dfa_of_checks_tsrb M O hL o L hch (by linarith) hm ⟨q₀, hq₀w, hq₀e⟩
    have hn : 0 < n := by
      refine Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos ?_)) (by positivity))
      rw [one_lt_div hδ]
      nlinarith only [hQ0, hδ1, mul_nonneg hQ0 hQ0]
    have hu : ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
        ∂(uniformStrings L) ≤ 3 / 8 * (N * εcov) := by
      refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
        fun x => measureReal_nonneg).trans ?_
      exact (mul_le_mul_of_nonneg_left hch.1.2.2 (zero_le_one.trans hN1)).trans
        (le_of_eq (by ring))
    have hmr : ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(uniformStrings L)
        ≤ 1025 / 1024 * (N * εcov) := by
      refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
        fun x => measureReal_nonneg).trans ?_
      exact (mul_le_mul_of_nonneg_left hch.1.2.1 (zero_le_one.trans hN1)).trans
        (le_of_eq (by ring))
    have h := pickLeaf_le M O o.result.hypothesis o.family o.state.lo (o.state.hi + 1) hL L n hn
      t (3 / 8 * (N * εcov)) (1025 / 1024 * (N * εcov)) _ q.1.1 q.2 hinj hgood hu hmr hs
    rw [hread]
    have hharm := tsrb_harm (A := A) (Q := Q) (N := N) (by linarith) hQ0 (by linarith) hm hm1 rfl
      hεc0.le hεcε
    have h' : (uniformStrings L).real {x | ¬ o.result.hypothesis.Classifies
        (pickLeaf o.result.hypothesis (fun z => familyRead O.mq o.family o.state.lo
          (o.state.hi + 1) z q.1.1) q.2) x (M.eval x.toList ∈ M.accept)}
        ≤ γ + N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
          + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)) / m
          + 2 * t + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) := h
    refine h'.trans ?_
    rw [htd]
    linarith
  -- the loop's failure, and each round's selection
  have hBl : (μL.prod β).real (Prod.fst ⁻¹' Bloop) ≤ δ / 2 := by
    have e : (μL.prod β).real (Prod.fst ⁻¹' Bloop) = μL.real Bloop := by
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    refine hD.trans ?_
    show (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) ≤ δ / 2
    have e : (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) = 5 * δ / 16 := by field_simp; ring
    rw [e]; linarith
  have hsel : ∀ r ∈ Finset.range (Fintype.card σ + 1), (μL.prod β).real (Sel r)
      ≤ δ / (2 * (Q + 1)) := by
    intro r hr
    have hr' := Finset.mem_range.1 hr
    simp only [hSeld, dif_pos hr']
    have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
    have hs1 : s ≤ 1 / 2 := by
      show 1 / 2 - η₀ ≤ 1 / 2
      linarith
    have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
    have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
      (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
    have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
    have hW1 : 1 ≤ W := le_max_right _ _
    have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hτ0 hτ1
      (Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith))
    refine (loop_select_round_le settings O P Dsf Ls L n (Fintype.card σ + 2) ⌊RB⌋₊ t
      (by positivity) (hwalk ▸ hkL) hnp r hr').trans ?_
    rw [hwalk]
    have e1 : ((Fintype.card σ + 2 : ℕ) : ℝ) = Q + 2 := by push_cast; rfl
    have e2 : ((Fintype.card σ + 1 : ℕ) : ℝ) + 1 = Q + 2 := by push_cast; ring
    rw [e1, e2, htd]
    exact dfa_select_arith hQ0 hA2' hδ hξ hkL (Nat.le_ceil _) hbud (Nat.floor_le hRB0) hk3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  refine (measureReal_union_le _ _).trans ?_
  refine (add_le_add hBl ((measureReal_biUnion_finset_le _ _).trans
    (Finset.sum_le_sum hsel))).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  show δ / 2 + ((Fintype.card σ + 1 : ℕ) : ℝ) * (δ / (2 * (Q + 1))) = δ
  push_cast
  field_simp
  ring

set_option maxRecDepth 4000 in
set_option maxHeartbeats 8000000 in -- unfolding the simple claim's settings
theorem learner_loop_dfa_tsrb_holds : LearnerLoopDFATSRB := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ m γ δ pAP ξ Q A k L₁ N ε δ' s S B₃ n aε Ce Θ K₁ Wm W₀
    P₀ R₀ E₁ Kc E E₂ E₃ T L₀ hL hη hη₀ hm hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hstart hL₀
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ 2 * L + Fintype.card σ + 3; omega
  have hkr : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hN1 : 1 ≤ N := by show (1 : ℝ) ≤ (L₁ : ℝ) + 1 - k; linarith
  have hNℓ : N ≤ (L₁ : ℝ) + 1 := by show (L₁ : ℝ) + 1 - k ≤ _; linarith
  have hℓ1 : (1 : ℝ) ≤ (L₁ : ℝ) + 1 := by have : (0 : ℝ) ≤ L₁ := Nat.cast_nonneg _; linarith
  have hm1 : m ≤ 1 := by
    obtain ⟨q₀, hq₀, -⟩ := hstart
    exact hq₀.trans measureReal_le_one
  -- `ε` and `aε` are `LearnerLoopDFA`'s at the coverage `c`
  set c : ℝ := 4 * (A + 1) * (ξ * m) / (5 * (A + Q + 3)) with hcd
  have hc : 0 < c := by rw [hcd]; positivity
  have hc1 : c ≤ 1 := by
    rw [hcd, div_le_one (by positivity)]
    have : ξ * m ≤ 1 := by nlinarith
    nlinarith
  have he : ε = c / (4 * (A + 1) * N) := by
    show ξ * m / (5 * (A + Q + 3) * N) = _
    rw [hcd]; field_simp
  have haε : aε = 4 * (A + 1) / c := by
    show 5 * (A + Q + 3) / (ξ * m) = _
    rw [hcd]; field_simp
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by show 1 / 2 - η₀ ≤ 1 / 2; linarith
  set ℓ : ℝ := (L₁ : ℝ) + 1 with hℓd
  set εcov : ℝ := min ε (ξ / (8 * N)) with hεcd
  set ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1))) with hιd
  set τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι with hτd
  set J : ℝ := Q + 2 with hJd
  set W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1 with hWd
  set PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2 with hPBd
  set RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
    / (s ^ 4 * τ ^ 2) with hRBd
  set cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
        (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
      (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
        (ι / 2 / (8 * J * (RB + 1)))))
    (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1))))) with hcapd
  obtain ⟨hε0, hinvε, hι0, hinvι, hτ0, hinvτ, hΘ1, haε0⟩ :=
    dfa_rate_bounds (ℓ := ℓ) hQ1 hA hℓ1 hN1 hNℓ hc hc1 hξ hη0 hη₀ he rfl rfl rfl rfl haε rfl rfl
  have hτ1 : τ ≤ 1 := by
    have h1 : ι ≤ εcov / 8 := min_le_left _ _
    have h2 : εcov ≤ ε := min_le_left _ _
    have h3 : ε ≤ 1 := he ▸ rate_le_one hc1 hA hN1
    exact (min_le_right _ _).trans (by linarith)
  have hS1 : 1 ≤ S := by
    show 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    linarith
  have hlS : 0 ≤ Real.log (16 * S * (S + 1) / δ') := log_budget_nonneg hS1 hδ'0 hδ'1
  have hB₃ : B₃ ≤ K₁ * ℓ := budget_le (by linarith) hε0 hinvε hℓ1 hlS
  have hB₃0 : 0 ≤ B₃ := by
    show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hK₁0 : 0 ≤ K₁ := by
    show 0 ≤ 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hWW : W ≤ W₀ * ℓ ^ 2 := by
    have hWm0 : 0 ≤ Wm := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
      show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
      positivity
    exact w_le hℓ1 hWm0
  have hW1 : 1 ≤ W := le_max_right _ _
  have hW₀1 : 1 ≤ W₀ := by
    have : 0 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
      have : 0 ≤ Wm := by
        show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
        positivity
      positivity
    show 1 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
    linarith
  obtain ⟨hPB1, hPB⟩ := pb_le (Q := Q) (W := W) (W₀ := W₀) hQ0 hτ0 hτ1 hinvτ hΘ1 hℓ1 hW1 hW₀1
    hWW hδ'0 hδ'1 hs0 hs1 hpAP hp1
  have hP₀1 : 1 ≤ P₀ := pool_ge_one hQ0 hs0 hs1 hpAP hp1 hΘ1 hW₀1 hδ'0 hδ'1
  have hPBJ : PB = (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((Q + 2 + 2) / (δ' / 2)) / pAP ^ 2 := rfl
  obtain ⟨hRB0, hRB⟩ := rb_le (Q := Q) (PB := PB) (P₀ := P₀) hQ0 hτ0 hτ1 hinvτ hΘ1 hℓ1
    (by rw [hPBJ]; exact hPB1) (by rw [hPBJ]; exact hPB) hP₀1 hδ'0 hδ'1 hs0 hpAP hp1
  have hR₀ : 0 ≤ R₀ := by
    have := log_count_nonneg hQ0 hP₀1 hΘ1 hδ'0 hδ'1 hpAP hp1
    show 0 ≤ 161000 * (Q + 2) ^ 2 * Θ ^ 2
      * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) / s ^ 4
    positivity
  have hτm : τ ≤ min (εcov / 2) ι := by
    refine le_min ?_ (min_le_right _ _)
    have ha := gateA_le_one hη0 hη₀
    have ha0 := (gateA_pos hη0 hη₀).le
    have hsq : ((1 - 2 * η₀) / (1 - η₀)) ^ 2 ≤ 1 := by
      have : (1 - 2 * η₀) / (1 - η₀) = gateA η₀ := rfl
      rw [this]; exact pow_le_one₀ ha0 ha
    have hεc0 : 0 ≤ εcov / 2 := by
      have : 0 < εcov := lt_min hε0 (by positivity)
      positivity
    exact (min_le_left _ _).trans (mul_le_of_le_one_right hεc0 hsq)
  have hlR₀ : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  have hcap := cap_ge (Q := Q) (RB := RB) (R₀ := R₀) (PB := PB) (P₀ := P₀) (E₁ := E₁) hQ0 hτ0
    hinvτ hι0 hinvι hΘ1 hℓ1 hRB0 (by rw [hRBd]; exact hRB) (by linarith) (by rw [hPBJ]; exact hPB)
    (by linarith) hR₀ hδ'0 hδ'1 hpAP hp1 hs0 hs1 hτm rfl
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith)
  have hlg : 0 ≤ Real.log (16 * (Q + 1) * Kc / δ') :=
    Real.log_nonneg (by
      rw [le_div_iff₀ hδ'0]
      nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)])
  obtain ⟨h1, h2, h3, h4⟩ := tsrb_tail (k := k) hA hQ0 (Nat.cast_nonneg L) hc hc1 hs0 hs1 hpAP
    hδ'0 hδ hΘ1 hR₀ (by linarith only [hP₀1]) hlR₀ hK₁0 (Nat.cast_nonneg n)
    (by show ((2 * L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 = _; push_cast; ring) hℓ1 hε0 hinvε hcap
    hlg haε rfl rfl rfl rfl rfl hRB0 hRB hB₃0 hB₃
    (by exact_mod_cast (show L ≤ 2 * ((L + 1) / 2) by omega)) rfl hL₀
  exact learner_loop_dfa_tsrb_simple_holds M O L η₀ m γ δ pAP ξ hL hη hη₀ hm hδ hδ1 hpAP hξ hξ1
    hA2 hpAPu hstart h1 h2 h3 h4

end OrthoDFA

end
