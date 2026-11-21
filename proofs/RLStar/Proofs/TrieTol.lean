import RLStar.Proofs.TrieCore
import RLStar.Proofs.TrieTolArith

/-!
# The loop with local walk starts at a tolerance

`trie_loop_core`'s conditions, from a walk start of `k` letters on whose every first `k` letters `D`
holds at most `1/Ak`, with `2^4 T^5 ≤ Ak`, `T` the `trieTolInv`: `T` bounds by itself the prefix
mass the loop needs (`dfa_final` without the held-out pick), `T` is at least the walk starts' budget
(`tol_B₃`), `log(1/δ)`, the family size, `|Σ|` and the inverse rates, so the walk starts' and the
family's misreads fit too.
-/

noncomputable section

set_option maxHeartbeats 1000000

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*} [Fintype σ]
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

lemma trieSucceeds_mono {K : ℕ} {settings : LoopSettings K} {cap : ℕ}
    {O : Oracle μ (FreeMonoid α)} {D Dsf T : Measure (FreeMonoid α)} {M : _root_.DFA α σ} {ξ R R' :
        ℝ}
    (hR : R ≤ R')
    {p : LoopDraws Ω α K settings.level3.harvestReadCap settings.level3.probeBudget}
    (h : RLStarSucceeds settings cap O D Dsf T M ξ R p) :
    RLStarSucceeds settings cap O D Dsf T M ξ R' p := by
  obtain ⟨h1, S, hS, hdep⟩ := h
  exact ⟨h1, S, hS.trans hR, hdep⟩

/-- The loop with local walk starts of `kw` letters, once `D` holds at most `1/Ak` on any first `kw`
letters and `2^4 T^5 ≤ Ak`. -/
theorem trie_loop_tol (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (L kw tcap : ℕ)
    (η₀ c δ pAP ξ ξT Ak : ℝ) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let k : ℕ := (L + 1) / 2
    let L₁ : ℕ := L + Fintype.card σ + 3
    let Ls : ℕ := L₁ + Fintype.card σ + 3
    let N : ℝ := L₁ + 1 - k
    let ε : ℝ := c / (8 + 4 * c * (2 + (Nat.clog 2 L₁ : ℝ)))
    let εcov : ℝ := min (c / (10 * N)) (ξ / (8 * N))
    let δ' : ℝ := δ / (8 * (Q + 1))
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    4 * (1 / 2 - η₀) ≤ O.gap → η₀ < 1 / 2 →
    ξ ≤ (L + Q) * (prefixLaw D L).real {p | p ∈ O.L} →
    ξ ≤ (L + Q) * (prefixLaw D L).real {p | p ∉ O.L} →
    0 < c → c ≤ 1 → 0 < δ → δ ≤ 1 → 0 < pAP → 0 < ξ →
    2 ≤ Fintype.card α → kw ≤ L → tcap ≤ kw →
    (∀ᵐ w ∂D, Ls ≤ w.toList.length) → (∀ u, D.real {w | prefixOf w kw = u} ≤ Ak⁻¹) →
    pAP ≤ (prefixLaw D Ls).real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
    2 ^ 4 * trieTolInv (L + Q + 4) Q A (Fintype.card σ) (Fintype.card α) δ c ξ pAP η₀ ^ 5
      ≤ Ak →
    c / 8 * (97 / 96) + (1025 / 1024 + 3 / 8) * (ξ / 8) + c ≤ ξT →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = kw ∧
      (∀ (p : LoopDraws Ω α (Fintype.card σ + 1) settings.level3.harvestReadCap
            settings.level3.probeBudget) rounds,
          indecisiveLStarTrieLoop settings tcap O (prefixLaw D L) (prefixLaw D Ls) p
            = some rounds →
        ((rounds.map fun o => o.prefs.card).sum : ℝ)
            ≤ (Q + 1) * (6 * (Q + 2) * trieCount (L + Q + 4) Q A δ c ξ pAP η₀)
          ∧ ((rounds.map fun o => o.suffs.card).sum : ℝ)
            ≤ (Q + 1) * (triePool (L + Q + 4) Q A δ c ξ pAP η₀ + 1)
          ∧ (rounds.length * settings.level3.probeBudget : ℝ)
            ≤ (Q + 1) * trieBudget (L + Q + 4) Q A δ c) ∧
      (loopMeasure μ α (Fintype.card σ + 1) settings.level3.harvestReadCap
          settings.level3.probeBudget).real
        {p | ¬ RLStarSucceeds settings tcap O (prefixLaw D L) (prefixLaw D Ls)
          (prefixLaw D L) M ξT
          (learnerReads (Q + 1) (6 * (Q + 2) * trieCount (L + Q + 4) Q A δ c ξ pAP η₀)
            (triePool (L + Q + 4) Q A δ c ξ pAP η₀ + 1) (trieFam (L + Q + 4) Q A δ c ξ η₀)
            (trieBudget (L + Q + 4) Q A δ c) Q
            ((⌈trieHB (L + Q + 4) Q A δ c ξ pAP η₀⌉₊ : ℕ) : ℝ)
            ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0) p}
        ≤ δ := by
  intro Q A k L₁ Ls N ε εcov δ' S B₃ hL hgap hη₀ hbL hbN hc hc1 hδ hδ1 hpAP hξ hA2
    hkw hcapw hlong hpre hpAPu hZ hξT
  classical
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith only [hQ1]
  have hA : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have hη0 : 0 ≤ η₀ := by have := gap_le_one O; linarith only [this, hgap]
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hkr : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hN1 : 1 ≤ N := by show (1 : ℝ) ≤ (L₁ : ℝ) + 1 - k; linarith only [hkr]
  have hNℓ : N ≤ (L₁ : ℝ) + 1 := by show (L₁ : ℝ) + 1 - k ≤ _; linarith only [hk0]
  have hℓ1 : (1 : ℝ) ≤ (L₁ : ℝ) + 1 :=
      by have : (0 : ℝ) ≤ L₁ := Nat.cast_nonneg _; linarith only [this]
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1 : δ' ≤ 1 := by
    show δ / (8 * (Q + 1)) ≤ 1
    rw [div_le_one (by positivity)]
    linarith only [hδ1, hQ0]
  set s : ℝ := 1 / 2 - η₀ with hsd
  have hs0 : 0 < s := by rw [hsd]; linarith only [hη₀]
  have hs1 : s ≤ 1 / 2 := by rw [hsd]; linarith only [hη0]
  set ℓ : ℝ := (L₁ : ℝ) + 1 with hℓd
  set aε : ℝ := trieIε ℓ c with haεd
  set Θι : ℝ := trieIι ℓ Q c ξ with hΘιd
  set Θ : ℝ := trieIτ ℓ Q c ξ η₀ with hΘd
  set Θa : ℝ := trieIa ℓ Q c ξ η₀ with hΘad
  set K₁ : ℝ := 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1 with hK₁d
  set W₀ : ℝ := loopW₀ Q A δ with hW₀d
  set Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
  set ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1))) with hιd
  set τ : ℝ := min ((εcov / 2) * (1 - 2 * η₀) ^ 2) ι with hτd
  set J : ℝ := Q + 2 with hJd
  set W : ℝ := max (2 * Real.exp 2 * Wm) 1 with hWd
  set PB : ℝ := (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28) /
      pAP
    + 120 * Real.log (3 / (δ' / 2)) / pAP with hPBd
  set RB : ℝ := 2 ^ 22 * J * Real.log (128 * (PB + 1) / δ') / (s ^ 4 * ι ^ 2)
    + 2 ^ 27 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ)) / τ ^ 2 with hRBd
  set cap : ℝ := min (min (min (s * min (εcov / 2 * s) ι / (2 ^ 20 * (RB + 1)))
        (ι ^ 2 / (2 ^ 28 * Real.log (128 * J * (PB + 1) * (3 + 2 * Real.logb 2 (RB + 1)) / δ'))))
      (ι / 2 / (8 * J * (RB + 1))))
    (min (min (pAP / (2 ^ 18 * (PB + 1))) (ι / (2 ^ 24 * J ^ 2 * (RB + 1))))
      (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1))))) with hcapd
  set Pf : ℝ := triePool ℓ Q A δ c ξ pAP η₀ with hPfd
  set C : ℝ := trieCount ℓ Q A δ c ξ pAP η₀ with hCd
  have hL1 : 1 < L₁ := by show 1 < L + Fintype.card σ + 3; omega
  have hlam := clog_le_log hL1
  have hlam' : 2 + (Nat.clog 2 L₁ : ℝ) ≤ 3 * (1 + Real.log ℓ) := by
    rw [hℓd]; exact_mod_cast hlam
  obtain ⟨hε0, hinvε, hι0, hinvι, hτ0, hinvτ, hinvA, haε1, hΘι1, hΘιΘ, hΘιΘa, hℓΘa⟩ :=
    trie_rate_bounds (ℓ := ℓ) (lam := 2 + (Nat.clog 2 L₁ : ℝ)) hQ1 hℓ1 hN1 hNℓ (by positivity)
      hlam' hc hc1 hξ hη0 hη₀ rfl rfl rfl rfl rfl
  have hinvε1 : 1 / ε ≤ aε * 1 := by rw [mul_one]; exact hinvε
  have hinvι1 : 1 / ι ≤ Θι * 1 := by rw [mul_one]; exact hinvι
  have hinvτ1 : 1 / τ ≤ Θ * 1 := by rw [mul_one]; exact hinvτ
  have hinvA1 : 1 / min (εcov / 2 * s) ι ≤ Θa * 1 := by rw [mul_one]; exact hinvA
  have hΘ1 : 1 ≤ Θ := hΘι1.trans hΘιΘ
  have hΘa1 : 1 ≤ Θa := hΘι1.trans hΘιΘa
  have hε8 : ε ≤ c / 8 := by
    show c / (8 + 4 * c * (2 + (Nat.clog 2 L₁ : ℝ))) ≤ c / 8
    exact div_le_div_of_nonneg_left hc.le (by norm_num) (by
      have : 0 ≤ 4 * c * (2 + (Nat.clog 2 L₁ : ℝ)) := by positivity
      linarith)
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
  have hB₃ : B₃ ≤ K₁ := by
    have := budget_le (S := S) (by linarith only [hS1]) hε0 hinvε1 le_rfl hlS
    rwa [mul_one] at this
  have hB₃0 : 0 ≤ B₃ := by
    show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    positivity
  have hK₁0 : 0 ≤ K₁ := by
    show 0 ≤ 36 * S * aε * Real.log (16 * S * (S + 1) / δ') + S + 1
    have : 0 ≤ aε := by linarith only [haε1]
    positivity
  have hWm0 : 0 ≤ Wm := by
    have : 0 ≤ Real.log (8 * S * (S + 1) / δ') :=
      Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hS1])
    show 0 ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
    positivity
  have hWW : W ≤ W₀ * 1 ^ 2 := w_le le_rfl hWm0
  have hW1 : 1 ≤ W := le_max_right _ _
  have hPB1 : 1 ≤ PB := (pb_le (W := W) (W₀ := W₀) hτ0 hτ1 hinvτ1 hΘ1 le_rfl hW1 hWW
    hδ'0 hδ'1 hs0 hs1 hpAP hp1).1
  have hPBf : PB ≤ Pf := by
    have h := pool_le hτ0 hinvτ1 hΘ1 le_rfl hW1 hWW hδ'0 hs0 hpAP
    rw [Real.log_one, mul_zero, zero_add] at h
    exact h
  have hPf1 : 1 ≤ Pf := hPB1.trans hPBf
  have hPBf1 : PB ≤ Pf * 1 := by rw [mul_one]; exact hPBf
  have hRB0 : 0 ≤ RB := (rb_le (Q := Q) (PB := PB) (P₀ := Pf) hQ0 hι0 hinvι1 hτ0 hτ1 hinvτ1 hΘ1
    le_rfl hPB1 hPBf1 hPf1 hδ'0 hδ'1 hs0 hpAP hp1).1
  have hRBC : RB ≤ C := by
    have h := count_le (Q := Q) (PB := PB) (PBf := Pf) hQ0 hι0 hinvι1 hτ0 hτ1 hinvτ1 hPB1 hPBf
      hδ'0 hδ'1 hs0 hpAP hp1
    rw [one_pow, mul_one, mul_one, mul_one] at h
    exact h
  have hC0 : 0 ≤ C := hRB0.trans hRBC
  have hmA0 : 0 < min (εcov / 2 * s) ι :=
    lt_min (mul_pos (div_pos (lt_min (by positivity) (by positivity)) (by norm_num)) hs0) hι0
  have hlC : 0 ≤ Real.logb 2 (C + 1) := Real.logb_nonneg (by norm_num) (by linarith only [hC0])
  set E₁ : ℝ := 2 ^ 20 * Θa * (C + 1) / s + (Q + 3) ^ 2 * (2 ^ 24 * Θι * (C + 1)
    + 2 ^ 28 * Θι ^ 2 * Real.log (128 * (Q + 2) * (Pf + 2) * (13 + 2 * Real.logb 2 (C + 1)) / δ')
    + 2 ^ 18 * (Pf + 2) / pAP) with hE₁d
  have hRB1 : RB ≤ C * (1 ^ 2 * (1 + Real.log 1)) := by
    rw [Real.log_one, add_zero, one_pow, mul_one, mul_one]; exact hRBC
  have hcap := cap_ge (Q := Q) (RB := RB) (R₀ := C) (PB := PB) (P₀ := Pf) (E₁ := E₁) hQ0 hι0
    hinvι1 hmA0 hinvA1 hΘι1 hΘιΘa le_rfl hRB0 hRB1 (by linarith only [hPB1]) hPBf1
    (by linarith only [hPf1]) hC0 hδ'0 hδ'1 hpAP hp1 hs0 hs1 rfl
  rw [Real.log_one, add_zero, one_pow, mul_one, mul_one] at hcap
  set Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
    * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) with hKcd
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith only [hA])
  have hlg : 0 ≤ Real.log (16 * (Q + 1) * Kc / δ') :=
    Real.log_nonneg (by
      rw [le_div_iff₀ hδ'0]
      nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)])
  -- `E₁` is at least what the scan cap, the prefix count and `1/δ` need
  have hlgd : 0 ≤ Real.log (16 * (Q + 2) / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ0])
  have hl : Real.log (16 * (Q + 2) / δ')
      ≤ Real.log (128 * (Q + 2) * (Pf + 2) * (13 + 2 * Real.logb 2 (C + 1)) / δ') := by
    refine Real.log_le_log (by positivity) (div_le_div_of_nonneg_right ?_ hδ'0.le)
    have h1 : 1 ≤ (Pf + 2) * (13 + 2 * Real.logb 2 (C + 1)) :=
      one_le_mul_of_one_le_of_one_le (by linarith only [hPf1]) (by linarith only [hlC])
    nlinarith only [hQ0, mul_le_mul_of_nonneg_left h1 (by linarith only [hQ0] : (0 : ℝ) ≤ Q + 2)]
  obtain ⟨hE₁ge, hΘaE⟩ := tol_E₁_ge (Q := Q) hC0 hΘι1 hΘa1 hPf1 hpAP hp1 hs0 (by linarith only [hs1]) hlgd hl hE₁d
  have hE₁1 : 1 ≤ E₁ := hΘa1.trans hΘaE
  have hPE : Pf ≤ E₁ := by
    have h0 : 0 ≤ 2 ^ 24 * (C + 1) + 2 ^ 28 * Real.log (16 * (Q + 2) / δ') := by positivity
    have hQ9 : 1 ≤ (Q + 3) ^ 2 := one_le_pow₀ (by linarith only [hQ0])
    nlinarith only [hE₁ge, h0, hQ9, hPf1]
  -- the tolerance
  have hℓe : (L : ℝ) + Q + 4 = ℓ := by
    show (L : ℝ) + (Fintype.card σ : ℝ) + 4 = ((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1
    push_cast; ring
  obtain ⟨T, hTd⟩ : ∃ T : ℝ,
      T = 1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε + K₁ := ⟨_, rfl⟩
  have hTe : trieTolInv ((L : ℝ) + Q + 4) Q A (Fintype.card σ) (Fintype.card α) δ c ξ pAP η₀
      = T := by rw [hℓe, hTd]; rfl
  rw [hTe] at hZ
  obtain ⟨hT1, hE₁T, haεT, hlgT, hK₁T, hET⟩ := tol_T_le haε1 hE₁1 hlg hK₁0 hTd
  have hT0 : 0 ≤ T := by linarith only [hT1]
  have hℓ0 : 0 ≤ ℓ := by linarith only [hℓ1]
  have hA0 : (0 : ℝ) < A := by linarith only [hA]
  have hAk : 0 < Ak := lt_of_lt_of_le (by positivity) hZ
  have hp₀0 : 0 < Ak⁻¹ := inv_pos.2 hAk
  have hpow : (1 : ℝ) ^ 4 * 1 * (1536 * aε * E₁ + 1536 * Real.log (16 * (Q + 1) * Kc / δ') * aε)
      + 1 ^ 2 * 1 * 0 ≤ Ak := by
    have : T ≤ T ^ 5 := le_self_pow₀ hT1 (by norm_num)
    have : T ^ 5 ≤ 2 ^ 4 * T ^ 5 := le_mul_of_one_le_left (by positivity) (by norm_num)
    linarith only [hET, hZ, this, ‹T ≤ T ^ 5›]
  have hRB1' : RB ≤ C * (1 ^ 2 * 1) := by rw [one_pow, mul_one, mul_one]; exact hRBC
  have hB₃1 : B₃ ≤ K₁ * 1 := by rw [mul_one]; exact hB₃
  have hcap' : 1 / (E₁ * (1 ^ 3 * 1)) ≤ cap := by rw [one_pow, mul_one, mul_one]; exact hcap
  obtain ⟨h1, h2, -⟩ := dfa_final (aε := aε) (E₁ := E₁) (E₂ := 0) (n := 0) (K₁ := K₁)
    (B₃ := B₃) (ℓ := 1) (Λ := 1) hAk le_rfl le_rfl hε0 hinvε1 (by linarith only [haε1]) hcap'
    (by linarith only [hE₁1]) hlg rfl (by simp) hQ0 le_rfl hδ rfl hRB0 hRB1' hB₃0 hB₃1 hpow
  have hkw1 : 1 ≤ kw := by
    by_contra h0
    have : kw = 0 := by omega
    have h1 := hpre 1
    rw [this] at h1
    have e : {w : FreeMonoid α | prefixOf w 0 = 1} = Set.univ := by
      ext w; simp [prefixOf]
    rw [e, probReal_univ] at h1
    have h9 : (1 : ℝ) ≤ T ^ 5 := one_le_pow₀ hT1
    have hAk1 : 1 < Ak := by linarith only [h9, hZ]
    have := inv_lt_one_of_one_lt₀ hAk1
    linarith only [h1, this]
  have hJ0 : 0 ≤ J := by rw [hJd]; linarith only [hQ0]
  -- the probes' avoided mass, from the cap's last term
  have hk3' : 6 * J * RB * (Ak)⁻¹ ≤ 1 / 2 := by
    have hc6 : cap ≤ 1 / (6 * J * (RB + 1)) :=
      (min_le_right _ _).trans ((min_le_right _ _).trans (min_le_right _ _))
    have hcap0 : 0 ≤ cap := (by positivity : (0 : ℝ) ≤ 1 / E₁).trans hcap
    have hε1 : ε ≤ 1 := hε8.trans (by linarith only [hc1])
    have hJR : 6 * J * RB * cap ≤ 1 := by
      have h := mul_le_mul_of_nonneg_left hc6 (by positivity : (0 : ℝ) ≤ 6 * J * (RB + 1))
      rw [mul_one_div_cancel (by positivity)] at h
      nlinarith only [h, hcap0, hJ0, hRB0]
    have hA' : 6 * J * RB * (Ak)⁻¹ ≤ 6 * J * RB * (cap * ε / 1536) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have : 6 * J * RB * (cap * ε / 1536) ≤ 1 / 1536 := by
      have := mul_le_mul hJR hε1 hε0.le zero_le_one
      rw [show 6 * J * RB * (cap * ε / 1536) = 6 * J * RB * cap * ε / 1536 by ring]
      linarith only [this]
    linarith only [hA', this]
  have h4 := tol_B₃ hB₃0 (hB₃.trans hK₁T) hT1 hZ
  -- the family's misreads
  have hm0 : 0 ≤ max (Real.log (20 / τ)) (Real.log (8 * W / δ')) :=
    le_max_of_le_left (Real.log_nonneg (by rw [le_div_iff₀ hτ0]; linarith only [hτ1]))
  have hX0 : 0 ≤ 50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s +
      28 := by
    positivity
  have hFVc0 : 0 ≤ (50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s +
      28)
      / 4 := by positivity
  have hlp : 0 ≤ 120 * Real.log (3 / (δ' / 2)) / pAP :=
    div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg (by
      rw [le_div_iff₀ (by positivity)]; linarith only [hδ'1]))) hpAP.le
  have hFVcT := tol_FV hX0 hpAP hp1 hlp hPBd hPBf hPE hE₁T
  have hFT : ((⌊(50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28)
      / 4⌋₊ : ℕ) : ℝ) ≤ T := (Nat.floor_le hFVc0).trans hFVcT
  have hKcl : Real.log Kc ≤ Real.log (16 * (Q + 1) * Kc / δ') := by
    refine Real.log_le_log (by linarith only [hKc1]) ?_
    rw [le_div_iff₀ hδ'0]
    nlinarith only [hδ'1, hQ0, hKc1, hδ'0, mul_nonneg hQ0 (by linarith only [hKc1] : (0 : ℝ) ≤ Kc),
      mul_le_mul_of_nonneg_left hδ'1 (by linarith only [hKc1] : (0 : ℝ) ≤ Kc)]
  have hAT : A ≤ T := ((A_le_log_kc hQ0).trans hKcl).trans hlgT
  have hLsT : (Ls : ℝ) + 1 ≤ 2 * T := by
    have : (Ls : ℝ) + 1 ≤ 2 * ℓ := by
      show ((L + Fintype.card σ + 3 + Fintype.card σ + 3 : ℕ) : ℝ) + 1
        ≤ 2 * (((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1)
      push_cast; linarith only [(Nat.cast_nonneg L : (0 : ℝ) ≤ L)]
    linarith only [this, hℓΘa, hΘaE, hE₁T]
  have hcT : 1 ≤ c * T := by
    have h1' : 8 ≤ c * aε := by
      show 8 ≤ c * (8 / c + 12 * (1 + Real.log ℓ))
      have : 0 ≤ c * (12 * (1 + Real.log ℓ)) := by
        have := Real.log_nonneg hℓ1
        positivity
      rw [mul_add, mul_div_cancel₀ _ hc.ne']; linarith only [this]
    have h2' := mul_le_mul_of_nonneg_left haεT hc.le
    linarith only [h1', h2']
  have hδT : Real.log (16 / δ) ≤ 16 * T := by
    have hKc0 : (0 : ℝ) < Kc := by linarith only [hKc1]
    have hl16 : Real.log (16 / δ) ≤ Real.log (16 * (Q + 1) * Kc / δ') := by
      refine Real.log_le_log (by positivity) ?_
      show 16 / δ ≤ 16 * (Q + 1) * Kc / (δ / (8 * (Q + 1)))
      rw [div_div_eq_mul_div, div_le_div_iff_of_pos_right hδ]
      have hq : 1 ≤ Q + 1 := by linarith only [hQ0]
      have : 1 ≤ (Q + 1) * Kc * (8 * (Q + 1)) :=
        one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hq hKc1)
          (by linarith only [hQ0])
      nlinarith only [this]
    linarith only [hl16, hlgT, hT1]
  have hk6' := tol_k6 hA hT1 hδ hFT hAT hLsT hcT hδT hZ
  rw [show -2 * Ak * c ^ 2 = -2 * c ^ 2 / Ak⁻¹ by rw [div_inv_eq_mul]; ring] at hk6'
  have hξT' : ε + ε / 96 + (1025 / 1024 + 3 / 8) * εcov + c ≤ ξT := by
    have hec : εcov ≤ ξ / 8 := (min_le_right _ _).trans
      (div_le_div_of_nonneg_left hξ.le (by norm_num) (by linarith only [hN1]))
    linarith only [hε8, hec, hξT]
  -- the clustering's balance: `2 εcov ≤ ξ/(4N)`, and `L + |Q| ≤ 4N`
  have hLQN : (L : ℝ) + Q ≤ 4 * N := by
    have h2k : 2 * k ≤ L + 1 := by show 2 * ((L + 1) / 2) ≤ L + 1; omega
    have h2k' : 2 * (k : ℝ) ≤ L + 1 := by exact_mod_cast h2k
    have hL0 : (0 : ℝ) ≤ L := Nat.cast_nonneg _
    have hσ0 : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
    show (L : ℝ) + (Fintype.card σ : ℝ) ≤ 4 * (((L + Fintype.card σ + 3 : ℕ) : ℝ) + 1 - k)
    push_cast
    linarith only [h2k', hL0, hσ0]
  have hN0 : 0 < N := by linarith only [hN1]
  have hbal : ∀ m : ℝ, 0 ≤ m → ξ ≤ (L + Q) * m → 2 * εcov ≤ m := fun m hm h => by
    have h1 : ξ ≤ 4 * N * m := h.trans (mul_le_mul_of_nonneg_right hLQN hm)
    have h2 : ξ / (4 * N) ≤ m := by rw [div_le_iff₀ (by positivity)]; linarith only [h1]
    have h3 : εcov ≤ ξ / (8 * N) := min_le_right _ _
    have e : ξ / (8 * N) = ξ / (4 * N) / 2 := by ring
    linarith only [h2, h3, e]
  have hbalL : 2 * εcov ≤ (prefixLaw D L).real {p | p ∈ O.L} := hbal _ measureReal_nonneg hbL
  have hbalN : 2 * εcov ≤ (prefixLaw D L).real {p | p ∉ O.L} := hbal _ measureReal_nonneg hbN
  obtain ⟨settings, hwalk, hbud, -, -, hlenT, hprob⟩ :=
    trie_loop_core M O L kw tcap η₀ c δ pAP ξ ξT c Ak⁻¹ D
      hL hgap hη₀ hbalL hbalN hc hc1 hδ hδ1 hpAP hξ hkw1 hkw hcapw hc.le hlong hp₀0 hpre
      hpAPu h1 h2 hk3' h4 hk6' hξT'
  -- the caps' bounds
  have hlg16 : 0 ≤ Real.log (16 * (Q + 2) / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith only [hδ'1, hQ0])
  have hQ1' : (0 : ℝ) ≤ Q + 1 := by linarith only [hQ0]
  have hSH0 : 0 < 24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2 := by
    have : 0 ≤ 24 * RB := mul_nonneg (by norm_num) hRB0
    linarith only [this, hlg16]
  have hl1 : 0 ≤ Real.log (2 / δ') := Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; linarith)
  have hl2 : 0 ≤ Real.log 2 := Real.log_nonneg (by norm_num)
  have hHB0 : 0 ≤ (Q + 1) * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2)
      * Real.log 2 + Real.log (2 / δ') :=
    add_nonneg (mul_nonneg (mul_nonneg hQ1' hSH0.le) hl2) hl1
  have hMle : 24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2
      ≤ trieSH ℓ Q A δ c ξ pAP η₀ := by
    have := mul_le_mul_of_nonneg_left hRBC (by norm_num : (0 : ℝ) ≤ 24)
    show _ ≤ 24 * C + 16 * Real.log (16 * (Q + 2) / δ') + 2
    linarith only [this]
  have hBle : 384 / ε * ((Q + 1) * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2)
      * Real.log 2 + Real.log (2 / δ')) + 1 ≤ trieHB ℓ Q A δ c ξ pAP η₀ := by
    have hε : 384 / ε ≤ 384 * aε := by
      rw [div_eq_mul_one_div]
      exact mul_le_mul_of_nonneg_left hinvε (by norm_num)
    have hin : (Q + 1) * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2)
        * Real.log 2 + Real.log (2 / δ')
        ≤ (Q + 1) * trieSH ℓ Q A δ c ξ pAP η₀ * Real.log 2 + Real.log (2 / δ') := by
      linarith only [mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hMle hQ1') hl2]
    have := mul_le_mul hε hin hHB0 (by linarith only [haε1])
    show _ ≤ 384 * aε * ((Q + 1) * trieSH ℓ Q A δ c ξ pAP η₀ * Real.log 2 + Real.log (2 / δ'))
      + 1
    linarith only [this]
  have hFV : ((50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28) /
      4) ≤ trieFam ℓ Q A δ c ξ η₀ := by
    have h := fam_le hτ0 hinvτ1 hΘ1 le_rfl hW1 hWW hδ'0 hs0
    rw [Real.log_one, mul_zero, zero_add] at h
    exact h
  have hPB0' : (0 : ℝ) ≤ PB + 1 := by linarith only [hPB1]
  have hReads : learnerReads (Q + 1) (6 * (Q + 2) * RB) (PB + 1)
        ((50 * (max (Real.log (20 / τ)) (Real.log (8 * W / δ')) + 3) / s ^ 2 + 20 / s + 28) / 4) B₃
            Q (⌈384 / ε * ((Q + 1)
        * (24 * RB + 16 * Real.log (16 * (Q + 2) / δ') + 2) * Real.log 2
        + Real.log (2 / δ')) + 1⌉₊ : ℝ) ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0
      ≤ learnerReads (Q + 1) (6 * (Q + 2) * trieCount (L + Q + 4) Q A δ c ξ pAP η₀)
            (triePool (L + Q + 4) Q A δ c ξ pAP η₀ + 1) (trieFam (L + Q + 4) Q A δ c ξ η₀)
            (trieBudget (L + Q + 4) Q A δ c) Q
            ((⌈trieHB (L + Q + 4) Q A δ c ξ pAP η₀⌉₊ : ℕ) : ℝ)
            ((3 + Nat.clog 2 L₁ : ℕ) : ℝ) A 0 := by
    rw [hℓe]
    exact learnerReads_mono hQ1' (mul_nonneg (by linarith only [hQ0]) hRB0) hPB0' hFVc0 hB₃0
      hQ0 (Nat.cast_nonneg _) (Nat.cast_nonneg _) hA0.le le_rfl le_rfl
      (mul_le_mul_of_nonneg_left hRBC (by linarith only [hQ0])) (by linarith only [hPBf]) hFV
      hB₃ le_rfl (Nat.cast_le.2 (Nat.ceil_mono hBle)) le_rfl le_rfl le_rfl
  refine ⟨settings, hwalk, fun p rounds hr => ?_, ?_⟩
  · obtain ⟨hlr, hpr, hsu⟩ := hlenT p rounds hr
    have hlen' : (rounds.length : ℝ) ≤ Q + 1 := by
      show _ ≤ (Fintype.card σ : ℝ) + 1
      exact_mod_cast hlr
    rw [hℓe]
    have hPBf1' : PB + 1 ≤ Pf + 1 := by linarith only [hPBf]
    refine ⟨hpr.trans (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hRBC
        (by linarith only [hQ0])) hQ1'), hsu.trans (mul_le_mul_of_nonneg_left hPBf1' hQ1'), ?_⟩
    calc (rounds.length * settings.level3.probeBudget : ℝ)
        ≤ (Q + 1) * K₁ := mul_le_mul hlen' (hbud.trans hB₃) (Nat.cast_nonneg _) hQ1'
      _ = (Q + 1) * trieBudget ℓ Q A δ c := rfl
  · refine (measureReal_mono ?_ (measure_ne_top _ _)).trans hprob
    intro p hp hs
    exact hp (trieSucceeds_mono hReads hs)

end OrthoDFA

end
