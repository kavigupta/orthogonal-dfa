import RLStar.Proofs.ClaimRLStarArith

/-!
# The knobs' totals against the claim's shapes

`knΛ ≤ 2⁹ logZ` and `knιInv = 2²⁰ ιInv`, so each total of `ClaimRLStarArith` is at most a
constant times its shape in `RLStarClaim`.
-/

noncomputable section

namespace OrthoDFA

open Real

section shapes

variable {r : RLStarRun}

lemma shape_ιInv (r : RLStarRun) :
    r.ιInv = (r.Q ^ 2 + knX r) * (1 / r.ξc) + r.Q ^ 2 * knLg r := by
  unfold RLStarRun.ιInv knLg knX; ring

lemma shape_nCap (r : RLStarRun) :
    r.nCap = r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
      + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 + r.Q * r.A * r.logZ := by
  unfold RLStarRun.nCap; ring

lemma shape_nPref (r : RLStarRun) :
    r.nPref = r.Q ^ 3 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4 := by
  unfold RLStarRun.nPref; ring

lemma shape_nSuff (r : RLStarRun) : r.nSuff = r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) := by
  unfold RLStarRun.nSuff; ring

lemma shape_nHeavy (r : RLStarRun) : r.nHeavy = (1 / r.ξc) * r.logZ * r.nCap := by
  unfold RLStarRun.nHeavy; ring

lemma shape_nProbe (r : RLStarRun) :
    r.nProbe = r.Q * r.nHeavy + r.Q ^ 3 * r.A * knX r * (1 / r.ξc) ^ 2 * r.logZ := by
  unfold RLStarRun.nProbe knX; ring

lemma shape_nReads (r : RLStarRun) :
    r.nReads = r.nPref * r.nSuff / r.Q
      + (r.k + r.logZ) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2) := by
  unfold RLStarRun.nReads; ring

lemma kn_shape_base (hr : r ∈ rlStarRunRegion) :
    2 ≤ r.logZ ∧ knΛ r ≤ 2 ^ 9 * r.logZ ∧ log 2 ≤ knLg r ∧ knLg r ≤ knX r := by
  obtain ⟨hZ8, -⟩ := knZ_ge hr
  have e : r.logZ = log (knZ r) := rfl
  have h8 : log (8 : ℝ) = 3 * log 2 := by
    rw [show (8 : ℝ) = 2 ^ 3 by norm_num, Real.log_pow]; norm_num
  have h8' : log 8 ≤ log (knZ r) := Real.log_le_log (by norm_num) hZ8
  have hlt := Real.log_two_gt_d9
  have hl2 : 2 ≤ r.logZ := by rw [e]; linarith
  refine ⟨hl2, ?_, knLg_ge_log_two hr, (knLg_bounds hr).2⟩
  unfold knΛ
  rw [← e]
  linarith

lemma shape_ιInv_ge (hr : r ∈ rlStarRunRegion) :
    1 ≤ r.ιInv ∧ knX r * (1 / r.ξc) ≤ r.ιInv ∧ r.Q * knLg r ≤ r.ιInv := by
  obtain ⟨hQ, -, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have hlg := (knLg_bounds hr).1
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have h1 : (0 : ℝ) ≤ r.Q ^ 2 * (1 / r.ξc) := by positivity
  have h2 : (r.Q : ℝ) * knLg r ≤ r.Q ^ 2 * knLg r :=
    mul_le_mul_of_nonneg_right (le_self_pow₀ hQ (by norm_num)) (by linarith)
  have h3 : (0 : ℝ) ≤ r.Q * knLg r := mul_nonneg (by positivity) (by linarith)
  have h4 : (1 : ℝ) ≤ knX r * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le hX iξ
  rw [shape_ιInv]
  refine ⟨by nlinarith, by nlinarith, by nlinarith⟩

lemma shape_lower (hr : r ∈ rlStarRunRegion) :
    r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 ≤ r.nCap
      ∧ r.Q ^ 3 * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 ≤ r.nCap
      ∧ r.Q * r.A * r.logZ ≤ r.nCap := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -, hp, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  have hι : 0 ≤ r.ιInv := zero_le_one.trans (shape_ιInv_ge hr).1
  rw [shape_nCap]
  have a0 : (0 : ℝ) ≤ r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 := by positivity
  have a3 : (0 : ℝ) ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
  have a4 : (0 : ℝ) ≤ r.Q * r.A * r.logZ := by positivity
  have b1 := mul_le_mul_of_nonneg_left (show (1 / r.s₀) ^ 2 ≤ (1 / r.s₀) ^ 2 + r.Q ^ 2 by
    nlinarith) a0
  have b2 := mul_le_mul_of_nonneg_left (show (r.Q : ℝ) ^ 2 ≤ (1 / r.s₀) ^ 2 + r.Q ^ 2 by
    nlinarith) a0
  have b0 : (0 : ℝ) ≤ r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2) := by
    positivity
  have e1 : r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * (1 / r.s₀) ^ 2
      = r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 := by ring
  have e2 : r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * (r.Q : ℝ) ^ 2
      = r.Q ^ 3 * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 := by ring
  refine ⟨by linarith, by linarith, by linarith⟩

lemma shape_pos (hr : r ∈ rlStarRunRegion) :
    0 ≤ r.ιInv ∧ 0 ≤ r.nCap ∧ 0 ≤ r.nPref ∧ 0 ≤ r.nSuff ∧ 0 ≤ r.nHeavy ∧ 0 ≤ r.nProbe
      ∧ 0 ≤ r.nReads := by
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  have hι : 0 ≤ r.ιInv := zero_le_one.trans (shape_ιInv_ge hr).1
  have hL : 0 ≤ r.nCap := by rw [shape_nCap]; positivity
  have hP : 0 ≤ r.nPref := by rw [shape_nPref]; positivity
  have hS : 0 ≤ r.nSuff := by rw [shape_nSuff]; positivity
  have hH : 0 ≤ r.nHeavy := by rw [shape_nHeavy]; positivity
  have hPr : 0 ≤ r.nProbe := by rw [shape_nProbe]; positivity
  refine ⟨hι, hL, hP, hS, hH, hPr, ?_⟩
  rw [shape_nReads]
  positivity

/-- `nCap` above `ιInv³`, and the heavy draws above `X² ξi² logZ`. -/
lemma nHeavy_ge (hr : r ∈ rlStarRunRegion) :
    r.ιInv ^ 3 ≤ r.nCap ∧ 1 ≤ r.nHeavy
      ∧ knX r ^ 2 * (1 / r.ξc) ^ 2 * r.logZ ≤ r.nHeavy := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨hι1, hιX, -⟩ := shape_ιInv_ge hr
  obtain ⟨hC1, -⟩ := shape_lower hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 1 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hι0 : 0 ≤ r.ιInv ^ 3 := by positivity
  have h1 : r.ιInv ^ 3 ≤ r.nCap := by
    have a1 : (1 : ℝ) ≤ r.Q * r.logZ * (1 / r.s₀) ^ 6 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ (by linarith))
        (one_le_pow₀ is)
    have := le_mul_of_one_le_left hι0 a1
    have e : r.Q * r.logZ * (1 / r.s₀) ^ 6 * r.ιInv ^ 3
        = r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 := by ring
    linarith
  have hξl : (1 : ℝ) ≤ (1 / r.ξc) * r.logZ := one_le_mul_of_one_le_of_one_le iξ (by linarith)
  have h3 : 1 ≤ r.ιInv ^ 3 := one_le_pow₀ hι1
  have hCH : r.nCap ≤ r.nHeavy := by
    rw [shape_nHeavy]; exact le_mul_of_one_le_left (by linarith) hξl
  refine ⟨h1, by linarith, ?_⟩
  -- `X² ξi ≤ (X ξi)³ ≤ ιInv³ ≤ nCap`
  have hXξ : (1 : ℝ) ≤ knX r * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le hX iξ
  have b1 : knX r ^ 2 * (1 / r.ξc) ≤ (knX r * (1 / r.ξc)) ^ 3 := by
    have : knX r ^ 2 * (1 / r.ξc) ≤ knX r ^ 2 * (1 / r.ξc) * (knX r * (1 / r.ξc) ^ 2) :=
      le_mul_of_one_le_right (by positivity) (one_le_mul_of_one_le_of_one_le hX
        (one_le_pow₀ iξ))
    nlinarith
  have b2 : (knX r * (1 / r.ξc)) ^ 3 ≤ r.ιInv ^ 3 := pow_le_pow_left₀ (by linarith) hιX 3
  have b3 := mul_le_mul_of_nonneg_left (b1.trans (b2.trans h1))
    (show (0 : ℝ) ≤ (1 / r.ξc) * r.logZ by positivity)
  rw [shape_nHeavy]
  have e : (1 / r.ξc) * r.logZ * (knX r ^ 2 * (1 / r.ξc))
      = knX r ^ 2 * (1 / r.ξc) ^ 2 * r.logZ := by ring
  linarith

end shapes

section knobs

variable {r : RLStarRun}

lemma knιInv_eq (r : RLStarRun) : knιInv r = 2 ^ 20 * r.ιInv := by
  rw [shape_ιInv]; unfold knιInv; ring

lemma knRBb_le (hr : r ∈ rlStarRunRegion) :
    knRBb r ≤ 2 ^ 79 * (r.Q * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  have hι := knιInv_eq r
  have hιs : 0 ≤ r.ιInv := (shape_pos hr).1
  unfold knRBb
  rw [hι]
  calc 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * (2 ^ 20 * r.ιInv) ^ 2
      ≤ 2 ^ 30 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 4 * (2 ^ 20 * r.ιInv) ^ 2 := by gcongr
    _ = 2 ^ 79 * (r.Q * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by ring

lemma knCapInv_le (hr : r ∈ rlStarRunRegion) : knCapInv r ≤ 2 ^ 129 * r.nCap := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, hA, -, -, -, -, -, -, -, hp, -, hs, -⟩ := kn_base hr
  have hι := knιInv_eq r
  have hιs : 0 ≤ r.ιInv := (shape_pos hr).1
  rw [shape_nCap]
  unfold knCapInv
  rw [hι]
  have h0 : (0 : ℝ) ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
  have h1 : (0 : ℝ) ≤ r.Q * r.A * r.logZ := by positivity
  calc 2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * (2 ^ 20 * r.ιInv) ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 + 2 ^ 6 * r.Q * r.A * knΛ r
      ≤ 2 ^ 60 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 4 * (2 ^ 20 * r.ιInv) ^ 3
          * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + 2 ^ 30 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2
        + 2 ^ 6 * r.Q * r.A * (2 ^ 9 * r.logZ) := by gcongr
    _ = 2 ^ 129 * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2))
        + 2 ^ 39 * (r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2)
        + 2 ^ 15 * (r.Q * r.A * r.logZ) := by ring
    _ ≤ 2 ^ 129 * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 + r.Q * r.A * r.logZ) := by linarith

/-- The prefix count's bound against the cap's. -/
lemma knRBb_cap (hr : r ∈ rlStarRunRegion) : 36 * r.Q * knRBb r ≤ knCapInv r := by
  obtain ⟨hQ, hA, -, -, -, -, -, -, -, hp, -, hs, -⟩ := kn_base hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hι1 := knιInv_one hr
  have hm : (0 : ℝ) ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 := by positivity
  have hιι : knιInv r ^ 2 ≤ knιInv r ^ 3 := pow_le_pow_right₀ hι1 (by norm_num)
  have hQQ : (r.Q : ℝ) ≤ (1 / r.s₀) ^ 2 + r.Q ^ 2 := by
    have : (r.Q : ℝ) ≤ r.Q ^ 2 := le_self_pow₀ hQ (by norm_num)
    have : (0 : ℝ) ≤ (1 / r.s₀) ^ 2 := by positivity
    linarith
  have h1 := mul_le_mul (mul_le_mul_of_nonneg_left hιι hm) hQQ (by linarith)
    (mul_nonneg hm (by positivity))
  have h2 : (0 : ℝ) ≤ 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
  have h3 : (0 : ℝ) ≤ 2 ^ 6 * r.Q * r.A * knΛ r := by positivity
  have h0 : (0 : ℝ) ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2 * r.Q := by positivity
  unfold knCapInv knRBb
  linarith

lemma knNh_le (hr : r ∈ rlStarRunRegion) : (knNh r : ℝ) ≤ 2 ^ 150 * r.nHeavy := by
  obtain ⟨hnh, -⟩ := knNh_bounds hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  have hc := knCapInv_le hr
  have hc1 := (knCapInv_bounds hr).1
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  obtain ⟨-, hC0, -⟩ := shape_pos hr
  refine hnh.trans ?_
  rw [shape_nHeavy]
  calc 2 ^ 12 * knCapInv r * (1 / r.ξc) * knΛ r
      ≤ 2 ^ 12 * (2 ^ 129 * r.nCap) * (1 / r.ξc) * (2 ^ 9 * r.logZ) := by gcongr
    _ = 2 ^ 150 * ((1 / r.ξc) * r.logZ * r.nCap) := by ring

end knobs

section totals

variable {r : RLStarRun}

lemma knPref_le (hr : r ∈ rlStarRunRegion) : knPref r ≤ 2 ^ 85 * r.nPref := by
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hRB0, hRB, -⟩ := knRB_bounds hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  have hιs : 0 ≤ r.ιInv := (shape_pos hr).1
  have hb := knRBb_le hr
  rw [shape_nPref]
  unfold knPref
  have h1 : ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r) ≤ (2 * r.Q) * (6 * (3 * r.Q) * knRBb r) := by
    gcongr <;> linarith
  have h2 := mul_le_mul_of_nonneg_left hb (show (0 : ℝ) ≤ 36 * r.Q ^ 2 by positivity)
  have h3 : (0 : ℝ) ≤ r.Q ^ 3 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4 := by positivity
  linarith

lemma knSuff_le (hr : r ∈ rlStarRunRegion) : knSuff r ≤ 2 ^ 21 * r.nSuff := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hPB28, hPB, -⟩ := knPB_bounds hr
  have hΛ30 := (knΛ_bounds hr).1
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have ip : 1 ≤ 1 / r.pAP := by rw [le_div_iff₀ hp]; linarith
  rw [shape_nSuff]
  unfold knSuff
  have h1 : (1 : ℝ) ≤ 2 ^ 10 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) := by
    have a : (1 : ℝ) ≤ (1 / r.s₀) ^ 2 := one_le_pow₀ (by linarith)
    have b : (1 : ℝ) ≤ knΛ r * (1 / r.s₀) ^ 2 := one_le_mul_of_one_le_of_one_le (by linarith) a
    have c := one_le_mul_of_one_le_of_one_le b ip
    linarith
  have h2 : knΛ r * ((1 / r.s₀) ^ 2 * (1 / r.pAP))
      ≤ 2 ^ 9 * r.logZ * ((1 / r.s₀) ^ 2 * (1 / r.pAP)) :=
    mul_le_mul_of_nonneg_right hΛ (by positivity)
  have hQ1 : ((r.Q : ℝ) + 1) * (knPB r + 1)
      ≤ 2 * r.Q * (2 ^ 11 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP)) :=
    mul_le_mul (by linarith) (by linarith) (by linarith) (by positivity)
  have h3 := mul_le_mul_of_nonneg_left h2 (show (0 : ℝ) ≤ 2 ^ 12 * r.Q by positivity)
  linarith

lemma knTrie_le (hr : r ∈ rlStarRunRegion) : knTrie r ≤ 2 ^ 151 * r.nHeavy := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hRB0, hRB, -⟩ := knRB_bounds hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨-, hC0, -, -, hH0, -⟩ := shape_pos hr
  have hRBc := knRBb_cap hr
  have hc := knCapInv_le hr
  have hnh := knNh_le hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hCH : r.nCap ≤ r.nHeavy := by
    rw [shape_nHeavy]
    exact le_mul_of_one_le_left hC0 (one_le_mul_of_one_le_of_one_le iξ (by linarith))
  have h1 := mul_le_mul_of_nonneg_left hRB (show (0 : ℝ) ≤ 6 * (r.Q + 2) by positivity)
  have h2 : 6 * ((r.Q : ℝ) + 2) * knRBb r ≤ 18 * r.Q * knRBb r :=
    mul_le_mul_of_nonneg_right (by linarith) (hRB0.trans hRB)
  unfold knTrie
  linarith

lemma knEB_le (hr : r ∈ rlStarRunRegion) :
    (knE r : ℝ) * (knB₃ r + knJ r * knB r)
      ≤ 2 ^ 22 * r.Q ^ 2 * r.A * knX r * (1 / r.ξc) ^ 2 * knΛ r
        + 2 ^ 23 * knX r ^ 2 * (1 / r.ξc) ^ 2 * knΛ r := by
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hb1, hb, -⟩ := knB_bounds hr
  obtain ⟨hB₃0, hB₃⟩ := knB₃_bounds hr
  obtain ⟨hJ1, hJX, -⟩ := knJ_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have h1 := mul_le_mul hE hB₃ hB₃0 (by positivity)
  have h2 := mul_le_mul hJX hb (by linarith) (by positivity)
  have h3 := mul_le_mul hE h2 (by positivity) (by positivity)
  have h4 : (0 : ℝ) ≤ knX r ^ 2 * (1 / r.ξc) ^ 2 * knΛ r := by positivity
  have e : (knE r : ℝ) * (knB₃ r + knJ r * knB r)
      = knE r * knB₃ r + knE r * (knJ r * knB r) := by ring
  rw [e]
  linarith

lemma knProbes_le (hr : r ∈ rlStarRunRegion) : knProbes r ≤ 2 ^ 152 * r.nProbe := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨-, -, hXX⟩ := nHeavy_ge hr
  obtain ⟨-, -, -, -, hH0, -⟩ := shape_pos hr
  have hnh := knNh_le hr
  have hEB := knEB_le hr
  have h1 := mul_le_mul_of_nonneg_left hΛ
    (show (0 : ℝ) ≤ 2 ^ 22 * r.Q ^ 2 * r.A * knX r * (1 / r.ξc) ^ 2 by positivity)
  have h2 := mul_le_mul_of_nonneg_left hΛ
    (show (0 : ℝ) ≤ 2 ^ 23 * knX r ^ 2 * (1 / r.ξc) ^ 2 by positivity)
  set P : ℝ := r.Q ^ 2 * r.A * knX r * (1 / r.ξc) ^ 2 * r.logZ with hPd
  have hP0 : 0 ≤ P := by rw [hPd]; positivity
  have hsum : (knNh r : ℝ) + knE r * (knB₃ r + knJ r * knB r) ≤ 2 ^ 151 * (r.nHeavy + P) := by
    have e1 : 2 ^ 22 * r.Q ^ 2 * r.A * knX r * (1 / r.ξc) ^ 2 * (2 ^ 9 * r.logZ) = 2 ^ 31 * P := by
      rw [hPd]; ring
    have e2 : 2 ^ 23 * knX r ^ 2 * (1 / r.ξc) ^ 2 * (2 ^ 9 * r.logZ)
        = 2 ^ 32 * (knX r ^ 2 * (1 / r.ξc) ^ 2 * r.logZ) := by ring
    linarith
  unfold knProbes
  rw [shape_nProbe]
  have h3 := mul_le_mul (show (r.Q : ℝ) + 1 ≤ 2 * r.Q by linarith) hsum
    (by have := (knNh r).cast_nonneg (α := ℝ); obtain ⟨hB0, -⟩ := knB₃_bounds hr; positivity)
    (by positivity)
  have e3 : r.Q ^ 3 * r.A * knX r * (1 / r.ξc) ^ 2 * r.logZ = r.Q * P := by rw [hPd]; ring
  rw [e3]
  linarith

/-- The table's shape per letter of the walk start, `Q A nHeavy logZ / s₀²`, bounds the samplers'
and the search's reads through `nCap`'s `Q²` term. -/
lemma reads_t2_ge (hr : r ∈ rlStarRunRegion) :
    r.Q ^ 4 * r.A * r.logZ ^ 3 * (1 / r.ξc) * r.ιInv ^ 3 * (1 / r.s₀) ^ 6
      ≤ r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2 := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨-, hC2, -⟩ := shape_lower hr
  have := mul_le_mul_of_nonneg_left hC2
    (show (0 : ℝ) ≤ r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 * ((1 / r.ξc) * r.logZ) by positivity)
  have e1 : r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 * ((1 / r.ξc) * r.logZ)
      * (r.Q ^ 3 * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4)
      = r.Q ^ 4 * r.A * r.logZ ^ 3 * (1 / r.ξc) * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 := by ring
  have e2 : r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 * ((1 / r.ξc) * r.logZ) * r.nCap
      = r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2 := by rw [shape_nHeavy]; ring
  linarith

lemma knFV_le (hr : r ∈ rlStarRunRegion) : knFV r ≤ 2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2 := by
  obtain ⟨-, -, -, hFV⟩ := knPB_bounds hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  have := mul_le_mul_of_nonneg_right hΛ (show (0 : ℝ) ≤ 2 ^ 8 * (1 / r.s₀) ^ 2 by positivity)
  linarith

/-- The clusterings' reads against `nPref nSuff / Q`. -/
lemma reads_clust_le (hr : r ∈ rlStarRunRegion) :
    ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r * (knPB r + 1 + 1))
      ≤ 2 ^ 107 * (r.nPref * r.nSuff / r.Q) := by
  obtain ⟨hQ, -⟩ := kn_base hr
  obtain ⟨hPB28, -⟩ := knPB_bounds hr
  obtain ⟨hRB0, -⟩ := knRB_bounds hr
  obtain ⟨-, -, hP0, -⟩ := shape_pos hr
  have hpref := knPref_le hr
  have hsuff := knSuff_le hr
  have hQ0 : (0 : ℝ) < r.Q := by linarith
  have a0 : (0 : ℝ) ≤ ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r) := by positivity
  have a1 : (r.Q : ℝ) * (knPB r + 1 + 1) ≤ 2 * (((r.Q : ℝ) + 1) * (knPB r + 1)) := by
    nlinarith [mul_nonneg hQ0.le (show (0 : ℝ) ≤ knPB r by linarith)]
  have a2 := mul_le_mul_of_nonneg_left a1 a0
  have a3 : knPref r * knSuff r ≤ (2 ^ 85 * r.nPref) * (2 ^ 21 * r.nSuff) :=
    mul_le_mul hpref hsuff (by unfold knSuff; positivity) (mul_nonneg (by norm_num) hP0)
  have e1 : knPref r * knSuff r = ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r)
      * (((r.Q : ℝ) + 1) * (knPB r + 1)) := by unfold knPref knSuff; ring
  rw [e1] at a3
  rw [mul_div_assoc', le_div_iff₀ hQ0]
  linarith

/-- The search's reads against `(k + 1) Q A nHeavy logZ / s₀²`. -/
lemma reads_search_le (hr : r ∈ rlStarRunRegion) :
    ((r.Q : ℝ) + 1) * (knFV r * knNr r)
      ≤ 2 ^ 58 * (((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, hlg2, hlgX⟩ := kn_shape_base hr
  obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
  obtain ⟨hNr, -⟩ := knNr_bounds hr
  obtain ⟨-, -, hKX⟩ := knK_bounds hr
  obtain ⟨hι1, hιX, hιQ⟩ := shape_ιInv_ge hr
  have ht2 := reads_t2_ge hr
  have hFV' := knFV_le hr
  have hlt := Real.log_two_gt_d9
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hκ0 : (0 : ℝ) ≤ (knK r : ℝ) + 1 := by positivity
  have hlg0 : 0 ≤ knLg r := by linarith
  set κ : ℝ := (knK r : ℝ) + 1 with hκd
  set ξi : ℝ := 1 / r.ξc with hξid
  -- `κ ξi lg ≤ ιInv³`
  have b1 : κ ≤ r.ιInv := by
    have : knX r ≤ knX r * ξi := le_mul_of_one_le_right (by linarith) iξ
    linarith
  have b2 : ξi ≤ r.ιInv := by
    have : ξi ≤ knX r * ξi := le_mul_of_one_le_left (by linarith) hX
    linarith
  have b3 : knLg r ≤ r.ιInv := by
    have : knLg r ≤ r.Q * knLg r := le_mul_of_one_le_left hlg0 hQ
    linarith
  have b4 : κ * ξi * knLg r ≤ r.ιInv ^ 3 := by
    have := mul_le_mul (mul_le_mul b1 b2 (by linarith) (by linarith)) b3 hlg0 (by positivity)
    calc κ * ξi * knLg r ≤ r.ιInv * r.ιInv * r.ιInv := this
      _ = r.ιInv ^ 3 := by ring
  have b5 : r.ιInv ^ 3 ≤ r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 := by
    have : (1 : ℝ) ≤ r.logZ * (1 / r.s₀) ^ 4 :=
      one_le_mul_of_one_le_of_one_le (by linarith) (one_le_pow₀ (by linarith))
    nlinarith [pow_nonneg (show (0 : ℝ) ≤ r.ιInv by linarith) 3]
  have a1 : (knNr r : ℝ) ≤ 2 ^ 31 * r.Q ^ 3 * r.A * κ ^ 2 * ξi ^ 2 * (2 ^ 9 * r.logZ) * knLg r := by
    refine hNr.trans ?_
    gcongr
  have a2 : ((r.Q : ℝ) + 1) * (knFV r * knNr r)
      ≤ (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2)
        * (2 ^ 31 * r.Q ^ 3 * r.A * κ ^ 2 * ξi ^ 2 * (2 ^ 9 * r.logZ) * knLg r)) :=
    mul_le_mul (by linarith) (mul_le_mul hFV' a1 (Nat.cast_nonneg _) (by positivity))
      (mul_nonneg hFV0 (Nat.cast_nonneg _)) (by positivity)
  have c0 : (0 : ℝ) ≤ κ * (r.Q ^ 4 * r.A * ξi * r.logZ ^ 2 * (1 / r.s₀) ^ 2) := by positivity
  have c1 := mul_le_mul_of_nonneg_left (b4.trans b5) c0
  have c2 := mul_le_mul_of_nonneg_left ht2 hκ0
  have e1 : (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2)
        * (2 ^ 31 * r.Q ^ 3 * r.A * κ ^ 2 * ξi ^ 2 * (2 ^ 9 * r.logZ) * knLg r))
      = 2 ^ 58 * (κ * (r.Q ^ 4 * r.A * ξi * r.logZ ^ 2 * (1 / r.s₀) ^ 2) * (κ * ξi * knLg r)) := by
    ring
  have e2 : κ * (r.Q ^ 4 * r.A * ξi * r.logZ ^ 2 * (1 / r.s₀) ^ 2)
        * (r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4)
      = κ * (r.Q ^ 4 * r.A * r.logZ ^ 3 * ξi * r.ιInv ^ 3 * (1 / r.s₀) ^ 6) := by ring
  linarith

lemma sampler_le {Q F S H U F' S' H' U' : ℝ} (hQ : 1 ≤ Q) (hF0 : 0 ≤ F) (hF : F ≤ F')
    (hS0 : 0 ≤ S) (hS : S ≤ S') (hH0 : 0 ≤ H) (hH : H ≤ H') (hU0 : 0 ≤ U) (hU : U ≤ U') :
    (Q + 1) * (F * ((Q + 1) * S * H * U)) ≤ (2 * Q) * (F' * ((2 * Q) * S' * H' * U')) := by
  have s1 := mul_le_mul (show Q + 1 ≤ 2 * Q by linarith) hS hS0 (by linarith)
  have s2 := mul_le_mul s1 hH hH0 (mul_nonneg (by linarith) (hS0.trans hS))
  have s3 := mul_le_mul s2 hU hU0
    (mul_nonneg (mul_nonneg (by linarith) (hS0.trans hS)) (hH0.trans hH))
  have hS' := hS0.trans hS
  have hH' := hH0.trans hH
  have hU' := hU0.trans hU
  have hF' := hF0.trans hF
  have s4 := mul_le_mul hF s3 (by positivity) hF'
  exact mul_le_mul (by linarith) s4 (by positivity) (by linarith)

/-- The samplers' reads against `logZ Q A nHeavy logZ / s₀²`. -/
lemma reads_sampler_le (hr : r ∈ rlStarRunRegion) :
    ((r.Q : ℝ) + 1) * (knFV r * ((r.Q + 1) * knSC r * knHR r * (knRp r * (r.Q + 1))))
      ≤ 2 ^ 131 * (r.logZ * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, hlg2, hlgX⟩ := kn_shape_base hr
  obtain ⟨hι0, -⟩ := shape_pos hr
  obtain ⟨-, -, hιQ⟩ := shape_ιInv_ge hr
  obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
  obtain ⟨hSC0, hSC⟩ := knSC_bounds hr
  obtain ⟨hHR1, hHR⟩ := knHR_bounds hr
  obtain ⟨hrp1, hrp⟩ := knRp_bounds hr
  have ht2 := reads_t2_ge hr
  have hlt := Real.log_two_gt_d9
  have hRBl := knRBb_le hr
  have hFV' := knFV_le hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hu : (knRp r : ℝ) * ((r.Q : ℝ) + 1) ≤ 64 * r.Q * knLg r := by
    have := mul_le_mul hrp (show (r.Q : ℝ) + 1 ≤ 4 * r.Q by linarith) (by linarith) (by linarith)
    linarith
  have hHR' : knHR r ≤ 2 ^ 21 * (1 / r.ξc) * r.logZ := by
    have := mul_le_mul_of_nonneg_left hΛ (show (0 : ℝ) ≤ 2 ^ 12 * (1 / r.ξc) by positivity)
    linarith
  have hSC' : knSC r ≤ 2 ^ 85 * (r.Q ^ 2 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by
    have := mul_le_mul_of_nonneg_left hRBl (show (0 : ℝ) ≤ 2 ^ 6 * r.Q by positivity)
    linarith
  have s5 := sampler_le hQ hFV0 hFV' hSC0 hSC' (by linarith) hHR'
    (mul_nonneg (Nat.cast_nonneg _) (by linarith)) hu
  set W' : ℝ := r.Q ^ 4 * r.logZ ^ 3 * (1 / r.ξc) * r.ιInv ^ 2 * (1 / r.s₀) ^ 6 with hW'd
  have hW'0 : 0 ≤ W' := by rw [hW'd]; positivity
  have a2 := mul_le_mul_of_nonneg_left hιQ hW'0
  have hAl : (1 : ℝ) ≤ r.A * r.logZ := one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
  have a4 := le_mul_of_one_le_left (show (0 : ℝ) ≤ W' * r.ιInv by positivity) hAl
  have a3 := mul_le_mul_of_nonneg_left ht2 (show (0 : ℝ) ≤ r.logZ by linarith)
  have e1 : (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2)
        * ((2 * r.Q) * (2 ^ 85 * (r.Q ^ 2 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4))
          * (2 ^ 21 * (1 / r.ξc) * r.logZ) * (64 * r.Q * knLg r)))
      = 2 ^ 131 * (W' * (r.Q * knLg r)) := by rw [hW'd]; ring
  have e2 : r.A * r.logZ * (W' * r.ιInv)
      = r.logZ * (r.Q ^ 4 * r.A * r.logZ ^ 3 * (1 / r.ξc) * r.ιInv ^ 3 * (1 / r.s₀) ^ 6) := by
    rw [hW'd]; ring
  linarith

/-- The walk-start table's reads against `(k + 1) Q A nHeavy logZ / s₀²`. -/
lemma reads_table_le (hr : r ∈ rlStarRunRegion) :
    knFV r * ((r.A * (knK r * knTrie r) + 1) * (r.Q + 2))
      ≤ 2 ^ 171 * (((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
  obtain ⟨hQ, hA, hX, -, hLX, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨-, -, -, -, hH0, -⟩ := shape_pos hr
  obtain ⟨-, b1, -⟩ := nHeavy_ge hr
  obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
  obtain ⟨hRB0, -⟩ := knRB_bounds hr
  have htrie := knTrie_le hr
  have hFV' := knFV_le hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hK0 : (0 : ℝ) ≤ knK r := Nat.cast_nonneg _
  have hT0 : 0 ≤ knTrie r := by unfold knTrie; positivity
  have a1 : (knK r : ℝ) * knTrie r ≤ knK r * (2 ^ 151 * r.nHeavy) :=
    mul_le_mul_of_nonneg_left htrie hK0
  have a2 : (1 : ℝ) ≤ r.A * r.nHeavy := one_le_mul_of_one_le_of_one_le (by linarith) b1
  have a3 : (r.A : ℝ) * (knK r * knTrie r) + 1
      ≤ 2 ^ 152 * (((knK r : ℝ) + 1) * (r.A * r.nHeavy)) := by
    have := mul_le_mul_of_nonneg_left a1 (show (0 : ℝ) ≤ r.A by positivity)
    have e : (r.A : ℝ) * (knK r * (2 ^ 151 * r.nHeavy)) = 2 ^ 151 * (knK r * (r.A * r.nHeavy)) := by
      ring
    have e2 : 2 ^ 152 * (((knK r : ℝ) + 1) * (r.A * r.nHeavy))
        = 2 ^ 152 * (knK r * (r.A * r.nHeavy)) + 2 ^ 152 * (r.A * r.nHeavy) := by ring
    have : (0 : ℝ) ≤ knK r * (r.A * r.nHeavy) := by positivity
    linarith
  have a4 := mul_le_mul hFV' (mul_le_mul a3 (show (r.Q : ℝ) + 2 ≤ 3 * r.Q by linarith)
    (by positivity) (by positivity)) (by positivity) (by positivity)
  have e : 2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2 * (2 ^ 152 * (((knK r : ℝ) + 1) * (r.A * r.nHeavy))
      * (3 * r.Q))
      = 3 * 2 ^ 169 * (((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
    ring
  have : (0 : ℝ) ≤ ((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2) := by
    positivity
  linarith

lemma knReads_le (hr : r ∈ rlStarRunRegion) : knReads r ≤ 2 ^ 172 * r.nReads := by
  have h1 := reads_clust_le hr
  have h2 := reads_search_le hr
  have h3 := reads_sampler_le hr
  have h4 := reads_table_le hr
  obtain ⟨-, -, hP0, hS0, hH0, -⟩ := shape_pos hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨T, hTd⟩ : ∃ T, T = r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2 := ⟨_, rfl⟩
  rw [← hTd] at h2 h3 h4
  have hT0 : 0 ≤ T := by rw [hTd]; positivity
  have hc0 : 0 ≤ r.nPref * r.nSuff / r.Q := by positivity
  have hKk : (knK r : ℝ) ≤ r.k := by exact_mod_cast min_le_left r.k r.L
  have hκ : (knK r : ℝ) + 1 ≤ r.k + r.logZ := by linarith
  have b1 := mul_le_mul_of_nonneg_right hκ hT0
  have b2 := mul_le_mul_of_nonneg_right (show r.logZ ≤ r.k + r.logZ by
    have := (r.k).cast_nonneg (α := ℝ); linarith) hT0
  have e : knReads r = ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r * (knPB r + 1 + 1))
      + ((r.Q : ℝ) + 1) * (knFV r * knNr r)
      + ((r.Q : ℝ) + 1) * (knFV r * ((r.Q + 1) * knSC r * knHR r * (knRp r * (r.Q + 1))))
      + knFV r * ((r.A * (knK r * knTrie r) + 1) * (r.Q + 2)) := by
    unfold knReads lazyReads; ring
  rw [e, shape_nReads, ← hTd]
  have : (0 : ℝ) ≤ ((knK r : ℝ) + 1) * T := by positivity
  have : (0 : ℝ) ≤ r.logZ * T := by positivity
  linarith

lemma mul_log_le {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : b * log a ≤ a + b ^ 2 := by
  have h1 := Real.log_le_sub_one_of_pos (div_pos ha hb)
  rw [Real.log_div ha.ne' hb.ne'] at h1
  have h2 := Real.log_le_sub_one_of_pos hb
  have h3 := mul_le_mul_of_nonneg_left h1 hb.le
  have e : b * (a / b - 1) = a - b := by field_simp
  rw [e] at h3
  have h4 := mul_le_mul_of_nonneg_left h2 hb.le
  nlinarith

set_option maxHeartbeats 800000 in -- `linarith` over the two inverse terms
lemma knP₀_inv_le (hr : r ∈ rlStarRunRegion) : 1 / knP₀ r ≤ 2 ^ 146 * (r.nCap / r.ξc) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, hp, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hc1, -⟩ := knCapInv_bounds hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨-, hC0, -⟩ := shape_pos hr
  obtain ⟨hι1, hιX, -⟩ := shape_ιInv_ge hr
  obtain ⟨hC1, -, hC3⟩ := shape_lower hr
  have hc := knCapInv_le hr
  have hMc := knMcD_pos hr
  have hcp := knCapP_pos hr
  have hcg := knCapP_ge hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hpa : 0 < knCapP r * knε r / 12288 := by positivity
  have h1 : 1 / knP₀ r ≤ 1 / (knCapP r * knε r / 12288) + 1 / knMcD r := by
    unfold knP₀
    exact one_div_min_le hpa hMc
  have hic : 1 / knCapP r ≤ knCapInv r := (one_div_le hcp (by linarith)).2 hcg
  have ea : 1 / (knCapP r * knε r / 12288) = 12288 * (1 / knCapP r) * (1 / knε r) := by
    field_simp
  rw [ea] at h1
  have b1 : 12288 * (1 / knCapP r) * (1 / knε r) ≤ 12288 * knCapInv r * (8 * (1 / r.ξc)) := by
    gcongr
  -- the misread term: `X ξi s₀⁻² log |Σ| ≤ |Σ| + (X ξi s₀⁻²)²`
  set B : ℝ := knX r * (1 / r.ξc) * (1 / r.s₀) ^ 2 with hBd
  have hB1 : 1 ≤ B :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hX iξ)
      (one_le_pow₀ (by linarith))
  have hBl := mul_log_le (show (0 : ℝ) < r.A by linarith) (show (0 : ℝ) < B by linarith)
  have hB2 : B ^ 2 ≤ r.ιInv ^ 3 * (1 / r.s₀) ^ 6 := by
    have a1 : (knX r * (1 / r.ξc)) ^ 2 ≤ r.ιInv ^ 3 := by
      have := pow_le_pow_left₀ (by positivity) hιX 2
      have : r.ιInv ^ 2 ≤ r.ιInv ^ 3 := pow_le_pow_right₀ hι1 (by norm_num)
      linarith
    have a2 : ((1 / r.s₀) ^ 2) ^ 2 ≤ (1 / r.s₀) ^ 6 := by
      rw [← pow_mul]; exact pow_le_pow_right₀ (by linarith) (by norm_num)
    have e : B ^ 2 = (knX r * (1 / r.ξc)) ^ 2 * ((1 / r.s₀) ^ 2) ^ 2 := by rw [hBd]; ring
    rw [e]
    exact mul_le_mul a1 a2 (by positivity) (by positivity)
  have b4 : 1 / knMcD r ≤ 2 ^ 26 * (1 / r.ξc) * r.nCap := by
    have e : 1 / knMcD r = 2 ^ 16 * knΛ r * (1 / r.ξc) * (B * log r.A) := by
      unfold knMcD; rw [one_div_div, hBd]; ring
    rw [e]
    have c1 : 2 ^ 16 * knΛ r * (1 / r.ξc) * (B * log r.A)
        ≤ 2 ^ 16 * (2 ^ 9 * r.logZ) * (1 / r.ξc) * (r.A + r.ιInv ^ 3 * (1 / r.s₀) ^ 6) := by
      have : 0 ≤ B * log (r.A : ℝ) :=
        mul_nonneg (by linarith) (Real.log_nonneg (by linarith))
      gcongr
      linarith
    have c2 : (r.A : ℝ) * r.logZ ≤ r.nCap := by
      have : (r.A : ℝ) * r.logZ ≤ r.Q * r.A * r.logZ := by
        have h0 : (0 : ℝ) ≤ r.A * r.logZ := by positivity
        nlinarith
      linarith
    have c3 : r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 ≤ r.nCap := by
      have h0 : (0 : ℝ) ≤ r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 := by positivity
      have : r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 ≤ r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 := by
        nlinarith
      linarith
    have e2 : 2 ^ 16 * (2 ^ 9 * r.logZ) * (1 / r.ξc) * (r.A + r.ιInv ^ 3 * (1 / r.s₀) ^ 6)
        = 2 ^ 25 * (1 / r.ξc) * (r.A * r.logZ + r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6) := by ring
    have c4 := mul_le_mul_of_nonneg_left (add_le_add c2 c3)
      (show (0 : ℝ) ≤ 2 ^ 25 * (1 / r.ξc) by positivity)
    linarith
  have e : r.nCap / r.ξc = (1 / r.ξc) * r.nCap := by ring
  rw [e]
  have h5 := mul_le_mul_of_nonneg_left hc (show (0 : ℝ) ≤ 12288 * (8 * (1 / r.ξc)) by positivity)
  have h6 : (0 : ℝ) ≤ (1 / r.ξc) * r.nCap := by positivity
  nlinarith

end totals

end OrthoDFA

end
