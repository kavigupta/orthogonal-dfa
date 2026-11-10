import RLStar.Proofs.TrieCore
import RLStar.Proofs.TrieTolArith

/-!
# The loop with local walk starts at a tolerance

`trie_loop_core`'s conditions, from a walk start of `k` letters with `2^30 T^9 ≤ |Σ|^k`, `T` main's
`loopTolInv`: `T` bounds by itself the prefix mass main's loop needs (`dfa_mass`), and `T` is at
least `1/δ`, `(|Q| + 1)²/δ`, the scan cap, the prefix count, the family size, `|Σ|` and the
inverse rates, so the batches' collisions and the family's misreads fit too.
-/

noncomputable section

set_option maxHeartbeats 400000

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ]
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

lemma trieSucceeds_mono {K : ℕ} {settings : LoopSettings K} {cap : ℕ}
    {O : Oracle μ (FreeMonoid α)} {D Dsf T : Measure (FreeMonoid α)} {M : _root_.DFA α σ} {ξ R R' : ℝ}
    (hR : R ≤ R')
    {p : LoopDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget}
    (h : RLStarTrieSucceeds settings cap O D Dsf T M ξ R p) :
    RLStarTrieSucceeds settings cap O D Dsf T M ξ R' p := by
  obtain ⟨h1, S, hS, hdep⟩ := h
  exact ⟨h1, S, hS.trans hR, hdep⟩

/-- The loop with local walk starts of `kw` letters, once `2^30 T^9 ≤ |Σ|^kw`. -/
theorem trie_loop_tol (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L kw tcap : ℕ)
    (η₀ c δ pAP ξ ξT Ae : ℝ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let Ls : ℕ := L₁ + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - k
    let ε : ℝ := c / (4 * (Ae + 1) * N)
    let εcov : ℝ := min (c / (10 * N)) (ξ / (8 * N))
    let δ' : ℝ := δ / (8 * (Q + 1))
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let n : ℕ := ⌈32 * Real.log (16 * (Q + 2) / δ) / ξ ^ 2⌉₊
    let SC : ℝ := 8 * (Q + 4) * (((L₁ : ℝ) + 1) ^ 2 * (1 + Real.log ((L₁ : ℝ) + 1))
        * loopR₀ Q A Ae δ c ξ pAP η₀) + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let HN : ℝ := 192 / ε * Real.log (4 * (Q + 1) * SC / δ') + 1
    let SH : ℝ := 24 * (((L₁ : ℝ) + 1) ^ 2 * (1 + Real.log ((L₁ : ℝ) + 1))
        * loopR₀ Q A Ae δ c ξ pAP η₀) + 16 * Real.log (16 * (Q + 2) / δ') + 2
    let HB : ℝ := 384 / ε * ((Q + 1) * SH * Real.log 2 + Real.log (2 / δ')) + 1
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 → 0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ →
    2 ≤ Fintype.card α → A ≤ Ae → kw ≤ L →
    pAP ≤ (uniformStrings Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    2 ^ 30 * loopTolInv (L + Q + 4) Q A Ae (Fintype.card σ) (Fintype.card α) δ c ξ pAP η₀ ^ 9
      ≤ A ^ kw →
    c / 12 * (97 / 96) + (1025 / 1024 + 3 / 8) * (ξ / 8) + c ≤ ξT →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = kw ∧
      (∀ (p : LoopDraws Ω α (Fintype.card σ + 1) settings.level3.harvestReadCap
            settings.level3.probeBudget) rounds,
          indecisiveLStarTrieLoop settings tcap O (uniformStrings L) (uniformStrings Ls) p
            = some rounds →
        ((rounds.map fun o => o.prefs.card).sum : ℝ)
            ≤ (Q + 1) * (6 * (Q + 2) * loopCount (L + Q + 4) Q A Ae δ c ξ pAP η₀)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ)
            ≤ (Q + 1) * (loopPool (L + Q + 4) Q A Ae δ c ξ pAP η₀ + 1)
          ∧ (rounds.length * settings.level3.probeBudget : ℝ)
            ≤ (Q + 1) * loopBudget (L + Q + 4) Q A Ae δ c) ∧
      (loopMeasure μ α (Fintype.card σ + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget).real
        {p | ¬ RLStarTrieSucceeds settings tcap O (uniformStrings L) (uniformStrings Ls)
          (uniformStrings L) M ξT
          (learnerReads (Q + 1) (6 * (Q + 2) * loopCount (L + Q + 4) Q A Ae δ c ξ pAP η₀)
            (loopPool (L + Q + 4) Q A Ae δ c ξ pAP η₀ + 1) (loopFam (L + Q + 4) Q A Ae δ c ξ η₀)
            (loopBudget (L + Q + 4) Q A Ae δ c) Q
            ((⌈loopHB (L + Q + 4) Q A Ae δ c ξ pAP η₀⌉₊ : ℕ) : ℝ)
            ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0) p}
        ≤ δ := by
  intro Q A k L₁ Ls N ε εcov δ' S B₃ n SC HN SH HB hL hη hη₀ hc hc1 hδ hδ1 hpAP hξ hA2 hAe hkw
    hpAPu hZ hξT
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith only [hQ1]
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hAe2 : (2 : ℝ) ≤ Ae := hA.trans hAe
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hkr : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hN1 : 1 ≤ N := by show (1 : ℝ) ≤ (L₁ : ℝ) + 1 - k; linarith only [hkr]
  have hNℓ : N ≤ (L₁ : ℝ) + 1 := by show (L₁ : ℝ) + 1 - k ≤ _; linarith only [hk0]
  have hℓ1 : (1 : ℝ) ≤ (L₁ : ℝ) + 1 := by have : (0 : ℝ) ≤ L₁ := Nat.cast_nonneg _; linarith only [this]
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith only [hδ1, hQ0]
  set s : ℝ := 1 / 2 - η₀ with hsd
  have hs0 : 0 < s := by rw [hsd]; linarith only [hη₀]
  have hs1 : s ≤ 1 / 2 := by rw [hsd]; linarith only [hη0]
  set aε : ℝ := 4 * (Ae + 1) / c with haεd
  set Θι : ℝ := loopΘι Q Ae c ξ with hΘιd
  set Θ : ℝ := loopΘ Q Ae c ξ η₀ with hΘd
  set Θa : ℝ := loopΘa Q Ae c ξ η₀ with hΘad
  set K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1 with hK₁d
  set W₀ : ℝ := loopW₀ Q A δ with hW₀d
  set P₀ : ℝ :=
    (50 * (max (Real.log (20 * Θ)) (Real.log (8 * W₀ / δ')) + 2) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP with hP₀d
  set R₀ : ℝ := 2 ^ 20 * (Q + 2) * Θι ^ 2 * (Real.log (256 * P₀ / δ') + 1) / s ^ 4
    + 2 ^ 27 * Θ ^ 2 * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2) with hR₀d
  set E₁ : ℝ := 2 ^ 20 * Θa * (R₀ + 1) / s + (Q + 3) ^ 2 * (2 ^ 24 * Θι * (R₀ + 1)
    + 2 ^ 28 * Θι ^ 2 * Real.log (128 * (Q + 2) * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ')
    + 2 ^ 16 * (P₀ + 2) / (pAP * δ'))
    with hE₁d
  set Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) with hKcd
  set E : ℝ := 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε with hEd
  set E₂ : ℝ := 4 * n * (6 * (Q + 2) * R₀ + K₁ + n) / δ with hE₂d
  set E₃ : ℝ := 16 * K₁ ^ 2 / δ' with hE₃d
  set ℓ : ℝ := (L₁ : ℝ) + 1 with hℓd
  set Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  set ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1))) with hιd
  set τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι with hτd
  set J : ℝ := Q + 2 with hJd
  set W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) 1 with hWd
  set PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP with hPBd
  set RB : ℝ := 2 ^ 20 * J * Real.log (128 * (PB + 1) / δ') / (s ^ 4 * ι ^ 2)
    + 2 ^ 27 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / τ ^ 2 with hRBd
  set cap : ℝ := min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
        (ι ^ 2 / (2 ^ 28 * Real.log (128 * J * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))))
      (ι / 2 / (8 * J * (RB + 1))))
    (min (min (δ' * pAP / (2 ^ 16 * (PB + 1))) (ι / (2 ^ 24 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1))))) with hcapd
  obtain ⟨hε0, hinvε, hι0, hinvι, hτ0, hinvτ, hinvA, hΘι1, hΘιΘ, hΘιΘa, haε0⟩ :=
    dfa_rate_bounds (ℓ := ℓ) (Ce := 10 / c + 8 / ξ) (Θι := Θι) (Θ := Θ) (Θa := Θa) hQ1 hAe2 hℓ1
      hN1 hNℓ hc hc1 hξ hη0 hη₀ rfl rfl rfl rfl rfl rfl rfl rfl rfl rfl
  have hΘ1 : 1 ≤ Θ := hΘι1.trans hΘιΘ
  have hΘa1 : 1 ≤ Θa := hΘι1.trans hΘιΘa
  have hτ1 : τ ≤ 1 := by
    have h1 : ι ≤ εcov / 8 := min_le_left _ _
    have h2 : εcov ≤ c / (10 * N) := min_le_left _ _
    have h3 : c / (10 * N) ≤ 1 := by rw [div_le_one (by positivity)]; linarith only [hc1, hN1]
    exact (min_le_right _ _).trans (by linarith only [h1, h2, h3])
  have hS1 : 1 ≤ S := by
    show 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    linarith only [this, hA]
  have hlS : 0 ≤ Real.log (16 * S * (S + 1) / δ') := log_budget_nonneg hS1 hδ'0 hδ'1
  have hB₃ : B₃ ≤ K₁ * ℓ := budget_le (by linarith only [hS1]) hε0 hinvε hℓ1 hlS
  have hB₃0 : 0 ≤ B₃ := by
    show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hK₁0 : 0 ≤ K₁ := by
    show 0 ≤ 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hWW : W ≤ W₀ * ℓ ^ 2 := by
    have hWm0 : 0 ≤ Wm := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hS1])
      show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
      positivity
    exact w_le hℓ1 hWm0
  have hW1 : 1 ≤ W := le_max_right _ _
  have hW₀1 : 1 ≤ W₀ := by
    have : 0 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 := by
      have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
        Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hS1])
      have : 0 ≤ Wm := by
        show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
        positivity
      positivity
    show 1 ≤ Real.exp 1 * Real.sqrt δ' * Wm / 2 + 1
    linarith only [this]
  obtain ⟨hPB1, hPB⟩ := pb_le (W := W) (W₀ := W₀) hτ0 hτ1 hinvτ hΘ1 hℓ1 hW1 hWW
    hδ'0 hδ'1 hs0 hs1 hpAP hp1
  have hP₀1 : 1 ≤ P₀ := pool_ge_one hs0 hs1 hpAP hp1 hΘ1 hW₀1 hδ'0 hδ'1
  have hPBJ : PB = (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 120 * Real.log (3 / (δ' / 2)) / pAP := rfl
  obtain ⟨hRB0, hRB⟩ := rb_le (Q := Q) (PB := PB) (P₀ := P₀) hQ0 hι0 hinvι hτ0 hτ1 hinvτ hΘ1
    hℓ1 (by rw [hPBJ]; exact hPB1) (by rw [hPBJ]; exact hPB) hP₀1 hδ'0 hδ'1 hs0 hpAP hp1
  have hR₀ : 0 ≤ R₀ := by
    have := log_count_nonneg hQ0 hP₀1 hΘ1 hδ'0 hδ'1 hpAP hp1
    have : 0 ≤ Real.log (256 * P₀ / δ') := Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith only [hP₀1, hδ'1])
    show 0 ≤ 2 ^ 20 * (Q + 2) * Θι ^ 2 * (Real.log (256 * P₀ / δ') + 1) / s ^ 4
      + 2 ^ 27 * Θ ^ 2 * (Real.log (16 * (Q + 4) * P₀ * Θ / (δ' ^ 2 * pAP)) + 2)
    positivity
  have hmA0 : 0 < min (εcov / 2 * s) ι :=
    lt_min (mul_pos (div_pos (lt_min (by positivity) (by positivity)) (by norm_num)) hs0) hι0
  have hlR₀ : 0 ≤ Real.logb 2 (R₀ + 1) := Real.logb_nonneg (by norm_num) (by linarith only [hR₀])
  have hcap := cap_ge (Q := Q) (RB := RB) (R₀ := R₀) (PB := PB) (P₀ := P₀) (E₁ := E₁) hQ0 hι0
    hinvι hmA0 hinvA hΘι1 hΘιΘa hℓ1 hRB0 (by rw [hRBd]; exact hRB) (by rw [hPBJ]; linarith only [hPB1])
    (by rw [hPBJ]; exact hPB) (by linarith only [hP₀1]) hR₀ hδ'0 hδ'1 hpAP hp1 hs0 hs1 rfl
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith only [hA])
  have hlg : 0 ≤ Real.log (16 * (Q + 1) * Kc / δ') :=
    Real.log_nonneg (by
      rw [le_div_iff₀ hδ'0]
      nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)])
  -- the prefix mass, from the tolerance
  obtain ⟨haε1, hE₁1, hE1, hE₂0, hE₃0⟩ := dfa_mass_one (aε := aε) (E₁ := E₁) (E := E) (E₂ := E₂)
    (E₃ := E₃) hAe2 hQ0 hc hc1 hs0 hs1 hpAP hδ'0 hδ'1 hδ hΘa1 (by linarith only [hΘι1]) hR₀
    (by linarith only [hP₀1]) hlR₀ hK₁0 (Nat.cast_nonneg n) hlg rfl rfl rfl rfl rfl
  have hℓe : (L : ℝ) + Q + 4 = ℓ := by
    show (L : ℝ) + (Fintype.card σ : ℝ) + 4 = ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1
    push_cast; ring
  have hMI : loopTolInv ℓ Q A Ae (Fintype.card σ) (Fintype.card α) δ c ξ pAP η₀
      = ℓ ^ 4 * (1 + Real.log ℓ) * E + ℓ ^ 2 * (1 + Real.log ℓ) * E₂ + ℓ ^ 2 * E₃ :=
    rfl
  have hΛ1 : 1 ≤ 1 + Real.log ℓ := by have := Real.log_nonneg hℓ1; linarith only [this]
  -- the tolerance
  obtain ⟨T, hTd⟩ : ∃ T : ℝ,
      T = ℓ ^ 4 * (1 + Real.log ℓ) * E + ℓ ^ 2 * (1 + Real.log ℓ) * E₂ + ℓ ^ 2 * E₃ := ⟨_, rfl⟩
  have hTe : loopTolInv ((L : ℝ) + Q + 4) Q A Ae (Fintype.card σ) (Fintype.card α) δ c ξ pAP η₀
      = T := by rw [hℓe, hTd]; exact hMI
  rw [hTe] at hZ
  obtain ⟨hT1, hTlE, hTE₁, hlℓ, hTℓ, hTaℓ, hTaε⟩ :=
    tol_T_le hℓ1 hΛ1 haε1 hE₁1 hE₂0 hE₃0 hlg hEd hTd
  have hT0 : 0 ≤ T := by linarith only [hT1]
  have hℓ0 : 0 ≤ ℓ := by linarith only [hℓ1]
  have hl2 : 1 ≤ ℓ ^ 2 * (1 + Real.log ℓ) :=
    one_le_mul_of_one_le_of_one_le (one_le_pow₀ hℓ1) hΛ1
  have hA0 : (0 : ℝ) < A := by linarith only [hA]
  have hA1 : (1 : ℝ) ≤ A := by linarith only [hA]
  have hAk : 0 < A ^ kw := pow_pos hA0 kw
  have hp₀0 : 0 < (A ^ kw)⁻¹ := inv_pos.2 hAk
  have hT9 : T ≤ 2 ^ 30 * T ^ 9 := by
    have : T ≤ T ^ 9 := le_self_pow₀ hT1 (by norm_num)
    linarith only [this, hT1]
  have hT9p : 2 ^ 30 * T ^ 9 * (A ^ kw)⁻¹ ≤ 1 := by
    rw [← div_eq_mul_inv, div_le_one hAk]; exact hZ
  have hpow3 : ℓ ^ 4 * (1 + Real.log ℓ) * E + ℓ ^ 2 * (1 + Real.log ℓ) * E₂ + ℓ ^ 2 * E₃
      ≤ A ^ kw := by
    rw [← hTd]; linarith only [hT9, hZ]
  obtain ⟨h1, h2, h3, h4⟩ := dfa_mass (aε := aε) (E₁ := E₁) (E := E) (E₂ := E₂) (E₃ := E₃)
    hAe2 hQ0 hc hc1 hs0 hs1 hpAP hδ'0 hδ'1 hδ hΘa1 (by linarith only [hΘι1]) hR₀
    (by linarith only [hP₀1]) hlR₀ hK₁0 (Nat.cast_nonneg n) hℓ1 hΛ1 hε0 hinvε hcap hlg rfl rfl rfl
    rfl rfl rfl hRB0 hRB hB₃0 hB₃ hAk hpow3
  have hkw1 : 1 ≤ kw := by
    by_contra h0
    have : kw = 0 := by omega
    rw [this, pow_zero] at hZ
    have : (1 : ℝ) ≤ T ^ 9 := one_le_pow₀ hT1
    linarith only [this, hZ]
  have hn1 : (1 : ℝ) ≤ n := by
    have hn : 0 < n := Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos (by
      rw [one_lt_div hδ]; linarith only [hδ1, hQ0]))) (by positivity))
    exact_mod_cast hn
  have hJ0 : 0 ≤ J := by rw [hJd]; linarith only [hQ0]
  have hk3' : 6 * J * RB * (A ^ kw)⁻¹ ≤ 1 / 2 := by
    have ha0 : 0 ≤ 6 * J * RB := mul_nonneg (mul_nonneg (by norm_num) hJ0) hRB0
    have hle : 6 * J * RB ≤ n * (6 * J * RB + B₃ + n) := by
      nlinarith only [hn1, ha0, hB₃0, mul_nonneg (by linarith only [hn1] : (0 : ℝ) ≤ n) hB₃0]
    have := mul_le_mul_of_nonneg_right hle hp₀0.le
    have h3' : (n : ℝ) * (6 * J * RB + B₃ + n) * (A ^ kw)⁻¹ ≤ δ / 4 := h3
    linarith only [this, h3', hδ1]
  -- `E₁` is at least what the scan cap, the prefix count and `1/δ` need
  have hlgd : 0 ≤ Real.log (16 * (Q + 2) / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ0])
  have hl : Real.log (16 * (Q + 2) / δ')
      ≤ Real.log (128 * (Q + 2) * (13 + 2 * Real.logb 2 (R₀ + 1)) / δ') := by
    refine Real.log_le_log (by positivity) (div_le_div_of_nonneg_right ?_ hδ'0.le)
    nlinarith only [hlR₀, hQ0, mul_nonneg (by linarith only [hQ0] : (0 : ℝ) ≤ Q + 2) hlR₀]
  have hE₁ge := tol_E₁_ge (Q := Q) hR₀ hΘι1 hΘa1 hP₀1 hδ'0 hpAP hp1 hs0 hlgd hl hE₁d
  have hPd0 : 0 ≤ (P₀ + 2) / δ' := div_nonneg (by linarith only [hP₀1]) hδ'0.le
  obtain ⟨hQδ, hδT, hP₀E⟩ := tol_Qδ hQ0 hP₀1 hδ rfl hδ'1
    (add_nonneg (mul_nonneg (by norm_num) (by linarith only [hR₀]))
      (mul_nonneg (by norm_num) hlgd)) hE₁ge hTE₁
  have hk5' := tol_k5 hQ0 hRB0 hRB hR₀ hl2 hlgd hPd0 hE₁ge hTlE hT1 hQδ hδ rfl hδ'1 hε0
    (hinvε.trans hTaℓ) hp₀0 hT9p rfl rfl rfl
  -- the family's misreads
  have hm0 : 0 ≤ max (Real.log (20 / τ)) (Real.log (8 * W / δ')) :=
    le_max_of_le_left (Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith only [hτ1]))
  have hX0 : 0 ≤ 50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28 := by
    positivity
  have hFVc0 : 0 ≤ (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28)
      / 4 := by positivity
  have hPB' : PB ≤ P₀ * ℓ := by rw [hPBJ]; exact hPB
  have hlp : 0 ≤ 120 * Real.log (3 / (δ' / 2)) / pAP :=
    div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ (by positivity)]; linarith only [hδ'1]))) hpAP.le
  have hFVcT := tol_FV hX0 hpAP hp1 hlp hPBd hPB' hP₀E hℓ0 (by linarith only [hE₁1]) hlℓ hTlE
  have hFT : ((⌊(50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28)
      / 4⌋₊ : ℕ) : ℝ) ≤ T := (Nat.floor_le hFVc0).trans hFVcT
  have hAT : A ≤ T := by
    have : Ae ≤ aε := by
      rw [haεd, le_div_iff₀ hc]
      nlinarith only [mul_nonneg (sub_nonneg.2 hc1) (by linarith only [hAe2] : (0 : ℝ) ≤ Ae), hAe2]
    linarith only [this, hTaε, hAe]
  have hLsT : (Ls : ℝ) + 1 ≤ 2 * T := by
    have : (Ls : ℝ) + 1 ≤ 2 * ℓ := by
      show ((L + Fintype.card σ + 3 + Fintype.card σ + 3 : ℕ) : ℝ) + 1
        ≤ 2 * (((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1)
      push_cast; linarith only [(Nat.cast_nonneg L : (0 : ℝ) ≤ L)]
    linarith only [this, hTℓ]
  have hcT : 1 ≤ c * T := by
    have h1' : 1 ≤ c * aε := by rw [haεd, mul_div_cancel₀ _ hc.ne']; linarith only [hAe2]
    have h2' := mul_le_mul_of_nonneg_left hTaε hc.le
    linarith only [h1', h2']
  have hk6' := tol_k6 hA hT1 hδ hFT hAT hLsT hcT hδT (hZ.trans (pow_le_pow_right₀ hA1 hkw))
  have hξT' : ε + ε / 96 + (1025 / 1024 + 3 / 8) * εcov + c ≤ ξT := by
    have he : ε ≤ c / 12 := by
      have k1 : (0 : ℝ) ≤ Ae + 1 - 3 := by linarith only [hAe2]
      have k2 : (0 : ℝ) ≤ N - 1 := by linarith only [hN1]
      have k3 : 12 ≤ 4 * (Ae + 1) * N := by nlinarith only [mul_nonneg k1 k2, hAe2, hN1]
      show c / (4 * (Ae + 1) * N) ≤ c / 12
      exact div_le_div_of_nonneg_left hc.le (by norm_num) k3
    have hec : εcov ≤ ξ / 8 := (min_le_right _ _).trans
      (div_le_div_of_nonneg_left hξ.le (by norm_num) (by linarith only [hN1]))
    linarith only [he, hec, hξT]
  obtain ⟨settings, hwalk, hbud, -, -, hlenT, hprob⟩ :=
    trie_loop_core M O L kw tcap η₀ c δ pAP ξ ξT c Ae
      hL hη hη₀ hc hc1 hδ hδ1 hpAP hξ hA2 hAe hkw1 hkw hc.le hpAPu h1 h2 hk3' h4 hk5' hk6' hξT'
  -- the caps' bounds, the prefix count at `ℓ² (1 + log ℓ) R₀`
  have hlg16 : 0 ≤ Real.log (16 * (Q + 2) / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ0])
  have hSCle : 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2 ≤ SC := by
    have hR : RB ≤ ℓ ^ 2 * (1 + Real.log ℓ) * loopR₀ Q A Ae δ c ξ pAP η₀ := by
      rw [hRBd]
      refine hRB.trans (le_of_eq ?_)
      change R₀ * (ℓ ^ 2 * (1 + Real.log ℓ)) = ℓ ^ 2 * (1 + Real.log ℓ) * R₀
      ring
    have := mul_le_mul_of_nonneg_left hR (by linarith only [hQ0] : (0 : ℝ) ≤ 8 * (Q + 4))
    show _ ≤ 8 * (Q + 4) * (ℓ ^ 2 * (1 + Real.log ℓ) * loopR₀ Q A Ae δ c ξ pAP η₀)
      + 16 * Real.log (16 * (Q + 2) / δ') + 2
    linarith only [this]
  have hSC0 : 0 < 8 * (Q + 4) * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2 := by
    have : 0 ≤ 8 * (Q + 4) * RB := mul_nonneg (by linarith only [hQ0]) hRB0
    linarith only [this, hlg16]
  have hPBf : PB ≤ loopPool ℓ Q A Ae δ c ξ pAP η₀ :=
    pool_le hτ0 hinvτ hΘ1 hℓ1 hW1 hWW hδ'0 hs0 hpAP
  have hcount : RB ≤ loopCount ℓ Q A Ae δ c ξ pAP η₀ :=
    count_le hQ0 hι0 hinvι hτ0 hτ1 hinvτ hPB1 hPBf hδ'0 hδ'1 hs0 hpAP hp1
  have hQ1' : (0 : ℝ) ≤ Q + 1 := by linarith only [hQ0]
  have hSHle : 24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2 ≤ SH := by
    have hR : RB ≤ ℓ ^ 2 * (1 + Real.log ℓ) * loopR₀ Q A Ae δ c ξ pAP η₀ := by
      rw [hRBd]
      refine hRB.trans (le_of_eq ?_)
      change R₀ * (ℓ ^ 2 * (1 + Real.log ℓ)) = ℓ ^ 2 * (1 + Real.log ℓ) * R₀
      ring
    have := mul_le_mul_of_nonneg_left hR (by norm_num : (0 : ℝ) ≤ 24)
    show _ ≤ 24 * (ℓ ^ 2 * (1 + Real.log ℓ) * loopR₀ Q A Ae δ c ξ pAP η₀)
      + 16 * Real.log (16 * (Q + 2) / δ') + 2
    linarith only [this]
  have hSH0 : 0 < 24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2 := by
    have : 0 ≤ 24 * RB := mul_nonneg (by norm_num) hRB0
    linarith only [this, hlg16]
  have hl1 : 0 ≤ Real.log (2 / δ') := Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
  have hl2 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have hHB0 : 0 ≤ (Q + 1) * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2)
      * Real.log 2 + Real.log (2 / δ') :=
    add_nonneg (mul_nonneg (mul_nonneg hQ1' hSH0.le) hl2) hl1
  have hMle : 24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
      ≤ loopSH ℓ Q A Ae δ c ξ pAP η₀ := by
    have := mul_le_mul_of_nonneg_left hcount (by norm_num : (0 : ℝ) ≤ 24)
    show _ ≤ 24 * loopCount ℓ Q A Ae δ c ξ pAP η₀ + 16 * Real.log (16 * (Q + 2) / δ') + 2
    linarith only [this]
  have hBle : 384 / ε * ((Q + 1) * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2)
      * Real.log 2 + Real.log (2 / δ')) + 1 ≤ loopHB ℓ Q A Ae δ c ξ pAP η₀ := by
    have hε : 384 / ε ≤ 384 * (4 * (Ae + 1) * ℓ / c) := by
      have : 4 * (Ae + 1) * ℓ / c = aε * ℓ := by rw [haεd]; ring
      rw [this, div_eq_mul_one_div]
      exact mul_le_mul_of_nonneg_left hinvε (by norm_num)
    have hin : (Q + 1) * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2)
        * Real.log 2 + Real.log (2 / δ')
        ≤ (Q + 1) * loopSH ℓ Q A Ae δ c ξ pAP η₀ * Real.log 2 + Real.log (2 / δ') := by
      linarith only [mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hMle hQ1') hl2]
    have := mul_le_mul hε hin hHB0 (mul_nonneg (by norm_num)
      (div_nonneg (mul_nonneg (by linarith only [hAe2]) (by linarith only [hℓ1])) hc.le))
    show _ ≤ 384 * (4 * (Ae + 1) * ℓ / c)
      * ((Q + 1) * loopSH ℓ Q A Ae δ c ξ pAP η₀ * Real.log 2 + Real.log (2 / δ')) + 1
    linarith only [this]
  have hFV : ((50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / 4) ≤ loopFam ℓ Q A Ae δ c ξ η₀ := fam_le hτ0 hinvτ hΘ1 hℓ1 hW1 hWW hδ'0 hs0
  have hPB0' : (0 : ℝ) ≤ PB + 1 := by rw [hPBJ]; linarith only [hPB1]
  have hReads : learnerReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1)
        ((50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / 4) B₃ Q (⌈384 / ε * ((Q + 1)
        * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2) * Real.log 2
        + Real.log (2 / δ')) + 1⌉₊ : ℝ) ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0
      ≤ learnerReads (Q + 1) (6 * (Q + 2) * loopCount (L + Q + 4) Q A Ae δ c ξ pAP η₀)
            (loopPool (L + Q + 4) Q A Ae δ c ξ pAP η₀ + 1) (loopFam (L + Q + 4) Q A Ae δ c ξ η₀)
            (loopBudget (L + Q + 4) Q A Ae δ c) Q
            ((⌈loopHB (L + Q + 4) Q A Ae δ c ξ pAP η₀⌉₊ : ℕ) : ℝ)
            ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0 := by
    rw [hℓe]
    exact learnerReads_mono hQ1' (mul_nonneg (by linarith only [hQ0]) hRB0) hPB0' hFVc0 hB₃0
      hQ0 (Nat.cast_nonneg _) (Nat.cast_nonneg _) hA0.le le_rfl le_rfl
      (mul_le_mul_of_nonneg_left hcount (by linarith only [hQ0])) (by linarith only [hPBf]) hFV
      hB₃ le_rfl (Nat.cast_le.2 (Nat.ceil_mono hBle)) le_rfl le_rfl le_rfl
  refine ⟨settings, hwalk, fun p rounds hr => ?_, ?_⟩
  · obtain ⟨hlr, hpr, hsu⟩ := hlenT p rounds hr
    have hlen' : (rounds.length : ℝ) ≤ Q + 1 := by
      show _ ≤ (Fintype.card σ : ℝ) + 1
      exact_mod_cast hlr
    rw [hℓe]
    have hPBf1 : PB + 1 ≤ loopPool ℓ Q A Ae δ c ξ pAP η₀ + 1 := by linarith only [hPBf]
    refine ⟨hpr.trans (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hcount
        (by linarith only [hQ0])) hQ1'), hsu.trans (mul_le_mul_of_nonneg_left hPBf1 hQ1'), ?_⟩
    calc (rounds.length * settings.level3.probeBudget : ℝ)
        ≤ (Q + 1) * (K₁ * ℓ) := mul_le_mul hlen' (hbud.trans hB₃) (Nat.cast_nonneg _) hQ1'
      _ = (Q + 1) * loopBudget ℓ Q A Ae δ c := rfl
  · refine (measureReal_mono ?_ (measure_ne_top _ _)).trans hprob
    intro p hp hs
    exact hp (trieSucceeds_mono hReads hs)

end OrthoDFA

end
