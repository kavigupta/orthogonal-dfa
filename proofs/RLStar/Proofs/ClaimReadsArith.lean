import RLStar.Proofs.ClaimRLStarArith

/-!
# The knobs' totals against the claim's shapes

`knΛ ≤ 2⁹ logZ` and `knιInv = 2²² ιInv`, so each total of `ClaimRLStarArith` is at most a
constant times its shape in `RLStarClaim`.
-/

noncomputable section

namespace OrthoDFA

open Real

section shapes

variable {r : RLStarRun}

lemma shape_ιInv (r : RLStarRun) : r.ιInv = r.Q ^ 2 * r.A * knX r * (1 / r.ξc) := by
  unfold RLStarRun.ιInv knX; ring

lemma shape_nCap (r : RLStarRun) :
    r.nCap = r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
      + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by
  unfold RLStarRun.nCap; ring

lemma shape_nPref (r : RLStarRun) :
    r.nPref = r.Q ^ 3 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4 := by
  unfold RLStarRun.nPref; ring

lemma shape_nSuff (r : RLStarRun) : r.nSuff = r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) := by
  unfold RLStarRun.nSuff; ring

lemma shape_nHeavy (r : RLStarRun) :
    r.nHeavy = r.A * knX r * (1 / r.ξc) * r.logZ * r.nCap := by
  unfold RLStarRun.nHeavy knX; ring

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

lemma knLg_logZ (hr : r ∈ rlStarRunRegion) : knLg r ≤ r.logZ := by
  obtain ⟨hQ, hA, hX, -, -, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := kn_base hr
  have hd0 : 0 < r.δ * r.ξc * r.pAP * r.s₀ := by positivity
  have hd1 : r.δ * r.ξc * r.pAP * r.s₀ ≤ 1 :=
    mul_le_one₀ (mul_le_one₀ (mul_le_one₀ (by linarith) hξ.le hξ1) hp.le hp1) hs.le (by linarith)
  have hXZ : knX r * r.A ≤ knZ r := by
    unfold knZ
    rw [le_div_iff₀ hd0]
    exact mul_le_of_le_one_right (by positivity) hd1
  have e : r.logZ = log (knZ r) := rfl
  unfold knLg
  rw [e]
  exact Real.log_le_log (by linarith) (by nlinarith)

/-- `nCap`'s first term at `Q² ≤ (1/s₀)² + Q²`, and the heavy draws above it. -/
lemma shape_lower (hr : r ∈ rlStarRunRegion) :
    r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 ≤ r.nCap
      ∧ r.Q ^ 7 * r.A ^ 4 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 2 * (1 / r.s₀) ^ 6
        ≤ r.nHeavy := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -, hp, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  have hι : 0 ≤ r.ιInv := by rw [shape_ιInv]; positivity
  have h1 : r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6 ≤ r.nCap := by
    rw [shape_nCap]
    have a1 : (1 / r.s₀) ^ 6 ≤ (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2) := by
      have : (0 : ℝ) ≤ (1 / r.s₀) ^ 4 * r.Q ^ 2 := by positivity
      nlinarith
    have a2 := mul_le_mul_of_nonneg_left a1 (show (0 : ℝ) ≤ r.Q * r.logZ * r.ιInv ^ 3 by
      positivity)
    have a3 : (0 : ℝ) ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
    nlinarith
  refine ⟨h1, ?_⟩
  rw [shape_nHeavy]
  have h2 := mul_le_mul_of_nonneg_left h1
    (show (0 : ℝ) ≤ r.A * knX r * (1 / r.ξc) * r.logZ by positivity)
  have e : r.A * knX r * (1 / r.ξc) * r.logZ * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6)
      = r.Q ^ 7 * r.A ^ 4 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 2 * (1 / r.s₀) ^ 6 := by
    rw [shape_ιInv]; ring
  linarith

lemma shape_pos (hr : r ∈ rlStarRunRegion) :
    0 ≤ r.ιInv ∧ 0 ≤ r.nCap ∧ 0 ≤ r.nPref ∧ 0 ≤ r.nSuff ∧ 0 ≤ r.nHeavy ∧ 0 ≤ r.nProbe
      ∧ 0 ≤ r.nReads := by
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  have hι : 0 ≤ r.ιInv := by rw [shape_ιInv]; positivity
  have hL : 0 ≤ r.nCap := by rw [shape_nCap]; positivity
  have hP : 0 ≤ r.nPref := by rw [shape_nPref]; positivity
  have hS : 0 ≤ r.nSuff := by rw [shape_nSuff]; positivity
  have hH : 0 ≤ r.nHeavy := by rw [shape_nHeavy]; positivity
  have hPr : 0 ≤ r.nProbe := by unfold RLStarRun.nProbe; positivity
  refine ⟨hι, hL, hP, hS, hH, hPr, ?_⟩
  rw [shape_nReads]
  positivity

end shapes

section knobs

variable {r : RLStarRun}

lemma knιInv_eq (r : RLStarRun) : knιInv r = 2 ^ 22 * r.ιInv := by
  rw [shape_ιInv]; unfold knιInv; ring

lemma knRBb_le (hr : r ∈ rlStarRunRegion) :
    knRBb r ≤ 2 ^ 83 * (r.Q * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  have hι := knιInv_eq r
  have hιs : 0 ≤ r.ιInv := (shape_pos hr).1
  unfold knRBb
  rw [hι]
  calc 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * (2 ^ 22 * r.ιInv) ^ 2
      ≤ 2 ^ 30 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 4 * (2 ^ 22 * r.ιInv) ^ 2 := by gcongr
    _ = 2 ^ 83 * (r.Q * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by ring

lemma knCapInv_le (hr : r ∈ rlStarRunRegion) : knCapInv r ≤ 2 ^ 135 * r.nCap := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, hp, -, hs, -⟩ := kn_base hr
  have hι := knιInv_eq r
  have hιs : 0 ≤ r.ιInv := (shape_pos hr).1
  rw [shape_nCap]
  unfold knCapInv
  rw [hι]
  have h0 : (0 : ℝ) ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
  calc 2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * (2 ^ 22 * r.ιInv) ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2
      ≤ 2 ^ 60 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 4 * (2 ^ 22 * r.ιInv) ^ 3
          * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + 2 ^ 30 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by gcongr
    _ = 2 ^ 135 * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2))
        + 2 ^ 39 * (r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2) := by ring
    _ ≤ 2 ^ 135 * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2) := by linarith

/-- The prefix count's bound against the cap's. -/
lemma knRBb_cap (hr : r ∈ rlStarRunRegion) : 36 * r.Q * knRBb r ≤ knCapInv r := by
  obtain ⟨hQ, -, -, -, -, -, -, -, -, hp, -, hs, -⟩ := kn_base hr
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
  have h0 : (0 : ℝ) ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2 * r.Q := by positivity
  unfold knCapInv knRBb
  linarith

lemma knNh_le (hr : r ∈ rlStarRunRegion) : (knNh r : ℝ) ≤ 2 ^ 158 * r.nHeavy := by
  obtain ⟨hnh, -⟩ := knNh_bounds hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  have hc := knCapInv_le hr
  have hc1 := (knCapInv_bounds hr).1
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  obtain ⟨-, hC0, -⟩ := shape_pos hr
  refine hnh.trans ?_
  rw [shape_nHeavy]
  calc 2 ^ 14 * knCapInv r * r.A * knX r * (1 / r.ξc) * knΛ r
      ≤ 2 ^ 14 * (2 ^ 135 * r.nCap) * r.A * knX r * (1 / r.ξc) * (2 ^ 9 * r.logZ) := by gcongr
    _ = 2 ^ 158 * (r.A * knX r * (1 / r.ξc) * r.logZ * r.nCap) := by ring

end knobs

section totals

variable {r : RLStarRun}

lemma knPref_le (hr : r ∈ rlStarRunRegion) : knPref r ≤ 2 ^ 89 * r.nPref := by
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

lemma knTrie_le (hr : r ∈ rlStarRunRegion) : knTrie r ≤ 2 ^ 159 * r.nHeavy := by
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
    have a1 : (1 : ℝ) ≤ r.A * knX r * (1 / r.ξc) * r.logZ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (by linarith) hX) iξ) (by linarith)
    exact le_mul_of_one_le_left hC0 a1
  have h1 := mul_le_mul_of_nonneg_left hRB (show (0 : ℝ) ≤ 6 * (r.Q + 2) by positivity)
  have h2 : 6 * ((r.Q : ℝ) + 2) * knRBb r ≤ 18 * r.Q * knRBb r :=
    mul_le_mul_of_nonneg_right (by linarith) (hRB0.trans hRB)
  unfold knTrie
  linarith

/-- `E (B₃ + J b) ≤ 2²⁸ Q² A³ X⁴ ξi² Λ` from each factor's bound. -/
lemma probes_body_le {E B₃ J b Q A X ξi Λ : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X)
    (hξi : 1 ≤ ξi) (hΛ : 30 ≤ Λ) (hE : E ≤ 2 ^ 12 * A * X ^ 2 * ξi)
    (hB0 : 0 ≤ B₃) (hB : B₃ ≤ 2 ^ 14 * Q ^ 2 * A ^ 2 * X * ξi * Λ)
    (hJ1 : 1 ≤ J) (hJ : J ≤ 3 * X) (hb1 : 1 ≤ b) (hb : b ≤ 2 ^ 13 * A * X * ξi * Λ) :
    E * (B₃ + J * b) ≤ 2 ^ 28 * Q ^ 2 * A ^ 3 * X ^ 4 * ξi ^ 2 * Λ := by
  obtain ⟨m, hm⟩ : ∃ m : ℝ, m = Q ^ 2 * A ^ 2 * X ^ 2 * ξi * Λ := ⟨_, rfl⟩
  have hQ2 : 1 ≤ Q ^ 2 := one_le_pow₀ hQ
  have hA2 : A ≤ A ^ 2 := le_self_pow₀ (by linarith) (by norm_num)
  have hX2 : X ≤ X ^ 2 := le_self_pow₀ hX (by norm_num)
  have hm0 : 0 ≤ m := by rw [hm]; positivity
  have hBm : B₃ ≤ 2 ^ 14 * m := by
    have := mul_le_mul_of_nonneg_left hX2 (show 0 ≤ Q ^ 2 * A ^ 2 by positivity)
    have := mul_le_mul_of_nonneg_right this (show 0 ≤ ξi * Λ by positivity)
    rw [hm]; nlinarith
  have hJbm : J * b ≤ 3 * 2 ^ 13 * m := by
    have h1 := mul_le_mul hJ hb (by linarith) (by linarith)
    have h3 : A ≤ Q ^ 2 * A ^ 2 := hA2.trans (le_mul_of_one_le_left (by positivity) hQ2)
    have := mul_le_mul_of_nonneg_right h3 (show 0 ≤ X ^ 2 * ξi * Λ by positivity)
    rw [hm]; nlinarith
  have h1 : B₃ + J * b ≤ 2 ^ 16 * m := by linarith
  have h2 := mul_le_mul hE h1 (by positivity) (by positivity)
  have e : 2 ^ 12 * A * X ^ 2 * ξi * (2 ^ 16 * m)
      = 2 ^ 28 * Q ^ 2 * A ^ 3 * X ^ 4 * ξi ^ 2 * Λ := by
    rw [hm]; ring
  linarith

lemma knEB_le (hr : r ∈ rlStarRunRegion) :
    (knE r : ℝ) * (knB₃ r + knJ r * knB r)
      ≤ 2 ^ 28 * r.Q ^ 2 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r := by
  obtain ⟨-, hE, -⟩ := knE_bounds hr
  obtain ⟨hb1, hb, -⟩ := knB_bounds hr
  obtain ⟨hB₃0, hB₃⟩ := knB₃_bounds hr
  obtain ⟨hJ1, hJX, -⟩ := knJ_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  exact probes_body_le hQ hA hX iξ hΛ hE hB₃0 hB₃ hJ1 hJX hb1 hb

lemma knProbes_le (hr : r ∈ rlStarRunRegion) : knProbes r ≤ 2 ^ 160 * r.nProbe := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨-, hHv⟩ := shape_lower hr
  obtain ⟨-, -, -, -, hH0, -⟩ := shape_pos hr
  have hnh := knNh_le hr
  have hEB := knEB_le hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 1 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  -- `Q² A³ X⁴ ξi² logZ ≤ nHeavy`
  have hP : r.Q ^ 2 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ ≤ r.nHeavy := by
    refine le_trans ?_ hHv
    have a1 : (1 : ℝ) ≤ r.Q ^ 5 * r.A * (1 / r.ξc) ^ 2 * r.logZ * (1 / r.s₀) ^ 6 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ)
          (by linarith)) (one_le_pow₀ iξ)) (by linarith)) (one_le_pow₀ is)
    have a0 : (0 : ℝ) ≤ r.Q ^ 2 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ := by positivity
    have e : r.Q ^ 2 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ
        * (r.Q ^ 5 * r.A * (1 / r.ξc) ^ 2 * r.logZ * (1 / r.s₀) ^ 6)
        = r.Q ^ 7 * r.A ^ 4 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 2 * (1 / r.s₀) ^ 6 := by ring
    have := le_mul_of_one_le_right a0 a1
    linarith
  have hEB' : (knE r : ℝ) * (knB₃ r + knJ r * knB r) ≤ 2 ^ 37 * r.nHeavy := by
    have := mul_le_mul_of_nonneg_left hΛ
      (show (0 : ℝ) ≤ 2 ^ 28 * r.Q ^ 2 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2 by positivity)
    linarith
  unfold knProbes RLStarRun.nProbe
  have h1 : (knNh r : ℝ) + knE r * (knB₃ r + knJ r * knB r) ≤ 2 ^ 159 * r.nHeavy := by linarith
  have h2 := mul_le_mul (show (r.Q : ℝ) + 1 ≤ 2 * r.Q by linarith) h1
    (by have := (knNh r).cast_nonneg (α := ℝ); obtain ⟨hB0, -⟩ := knB₃_bounds hr; positivity)
    (by positivity)
  linarith

/-- The table's shape per letter of the walk start, `Q A nHeavy logZ / s₀²`, by its factors. -/
lemma reads_t2_ge (hr : r ∈ rlStarRunRegion) :
    r.Q ^ 8 * r.A ^ 5 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 3 * (1 / r.s₀) ^ 8
        ≤ r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2
      ∧ r.Q ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc) * r.logZ ^ 3 * (1 / r.s₀) ^ 8 * r.ιInv ^ 3
        ≤ r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2 := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨hCv, hHv⟩ := shape_lower hr
  constructor
  · have := mul_le_mul_of_nonneg_left hHv
      (show (0 : ℝ) ≤ r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 by positivity)
    have e1 : r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2
        * (r.Q ^ 7 * r.A ^ 4 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 2 * (1 / r.s₀) ^ 6)
        = r.Q ^ 8 * r.A ^ 5 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 3 * (1 / r.s₀) ^ 8 := by ring
    have e2 : r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 * r.nHeavy
        = r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2 := by ring
    linarith
  · have := mul_le_mul_of_nonneg_left hCv
      (show (0 : ℝ) ≤ r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2
        * (r.A * knX r * (1 / r.ξc) * r.logZ) by positivity)
    have e1 : r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 * (r.A * knX r * (1 / r.ξc) * r.logZ)
        * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 6)
        = r.Q ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc) * r.logZ ^ 3 * (1 / r.s₀) ^ 8
          * r.ιInv ^ 3 := by ring
    have e2 : r.Q * r.A * r.logZ * (1 / r.s₀) ^ 2 * (r.A * knX r * (1 / r.ξc) * r.logZ)
        * r.nCap = r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2 := by
      rw [shape_nHeavy]; ring
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
      ≤ 2 ^ 111 * (r.nPref * r.nSuff / r.Q) := by
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
  have a3 : knPref r * knSuff r ≤ (2 ^ 89 * r.nPref) * (2 ^ 21 * r.nSuff) :=
    mul_le_mul hpref hsuff (by unfold knSuff; positivity) (mul_nonneg (by norm_num) hP0)
  have e1 : knPref r * knSuff r = ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r)
      * (((r.Q : ℝ) + 1) * (knPB r + 1)) := by unfold knPref knSuff; ring
  rw [e1] at a3
  rw [mul_div_assoc', le_div_iff₀ hQ0]
  linarith

/-- The search's reads against `(k + 1) Q A nHeavy logZ / s₀²`. -/
lemma reads_search_le (hr : r ∈ rlStarRunRegion) :
    ((r.Q : ℝ) + 1) * (knFV r * knNr r)
      ≤ 2 ^ 63 * (((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, hlg2, hlgX⟩ := kn_shape_base hr
  obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
  obtain ⟨hNr, -⟩ := knNr_bounds hr
  obtain ⟨-, -, hKX⟩ := knK_bounds hr
  obtain ⟨ht2l, -⟩ := reads_t2_ge hr
  have hFV' := knFV_le hr
  have hlt := Real.log_two_gt_d9
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hκ0 : (0 : ℝ) ≤ (knK r : ℝ) + 1 := by positivity
  have hκ2 : ((knK r : ℝ) + 1) ^ 2 ≤ ((knK r : ℝ) + 1) * knX r := by
    rw [sq]; exact mul_le_mul_of_nonneg_left hKX hκ0
  have a1 : (knNr r : ℝ) ≤ 2 ^ 36 * r.Q ^ 3 * r.A ^ 3 * (((knK r : ℝ) + 1) * knX r) * knX r ^ 2
      * (1 / r.ξc) ^ 2 * (2 ^ 9 * r.logZ) * knX r := by
    refine hNr.trans ?_
    have : 0 ≤ knLg r := by linarith
    have : 0 ≤ knΛ r := by linarith [(knΛ_bounds hr).1]
    gcongr
  have a2 : ((r.Q : ℝ) + 1) * (knFV r * knNr r)
      ≤ (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2) * (2 ^ 36 * r.Q ^ 3 * r.A ^ 3
        * (((knK r : ℝ) + 1) * knX r) * knX r ^ 2 * (1 / r.ξc) ^ 2 * (2 ^ 9 * r.logZ)
        * knX r)) :=
    mul_le_mul (by linarith) (mul_le_mul hFV' a1 (Nat.cast_nonneg _) (by positivity))
      (mul_nonneg hFV0 (Nat.cast_nonneg _)) (by positivity)
  have a3 : (1 : ℝ) ≤ r.Q ^ 4 * r.A ^ 2 * (1 / r.ξc) ^ 2 * r.logZ * (1 / r.s₀) ^ 6 :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ)
        (one_le_pow₀ (by linarith))) (one_le_pow₀ iξ)) (by linarith))
      (one_le_pow₀ (by linarith))
  have a4 : (0 : ℝ) ≤ ((knK r : ℝ) + 1) * (r.Q ^ 4 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2
      * r.logZ ^ 2 * (1 / r.s₀) ^ 2) := by positivity
  have a5 := le_mul_of_one_le_right a4 a3
  have a6 := mul_le_mul_of_nonneg_left ht2l hκ0
  have e1 : (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2) * (2 ^ 36 * r.Q ^ 3 * r.A ^ 3
        * (((knK r : ℝ) + 1) * knX r) * knX r ^ 2 * (1 / r.ξc) ^ 2 * (2 ^ 9 * r.logZ)
        * knX r))
      = 2 ^ 63 * (((knK r : ℝ) + 1) * (r.Q ^ 4 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2
        * r.logZ ^ 2 * (1 / r.s₀) ^ 2)) := by ring
  have e2 : ((knK r : ℝ) + 1) * (r.Q ^ 4 * r.A ^ 3 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ ^ 2
      * (1 / r.s₀) ^ 2) * (r.Q ^ 4 * r.A ^ 2 * (1 / r.ξc) ^ 2 * r.logZ * (1 / r.s₀) ^ 6)
      = ((knK r : ℝ) + 1) * (r.Q ^ 8 * r.A ^ 5 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 3
        * (1 / r.s₀) ^ 8) := by ring
  linarith

/-- `Q³ ≤ A ιInv / s₀²`. -/
lemma q_cube_le (hr : r ∈ rlStarRunRegion) :
    (r.Q : ℝ) ^ 3 ≤ r.A * (1 / r.s₀) ^ 2 * r.ιInv := by
  obtain ⟨hQ, hA, hX, hQX, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  rw [shape_ιInv]
  have c1 : (1 : ℝ) ≤ r.A * (1 / r.s₀) ^ 2 * (r.A * (1 / r.ξc)) :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith)
      (one_le_pow₀ (by linarith))) (one_le_mul_of_one_le_of_one_le (by linarith) iξ)
  have c2 := le_mul_of_one_le_right (show (0 : ℝ) ≤ knX r by linarith) c1
  have b1 : (r.Q : ℝ) ≤ r.A * (1 / r.s₀) ^ 2 * (r.A * knX r * (1 / r.ξc)) := by
    have e : knX r * (r.A * (1 / r.s₀) ^ 2 * (r.A * (1 / r.ξc)))
        = r.A * (1 / r.s₀) ^ 2 * (r.A * knX r * (1 / r.ξc)) := by ring
    linarith
  have b2 := mul_le_mul_of_nonneg_left b1 (show (0 : ℝ) ≤ r.Q ^ 2 by positivity)
  have e : (r.Q : ℝ) ^ 2 * (r.A * (1 / r.s₀) ^ 2 * (r.A * knX r * (1 / r.ξc)))
      = r.A * (1 / r.s₀) ^ 2 * (r.Q ^ 2 * r.A * knX r * (1 / r.ξc)) := by ring
  have e3 : (r.Q : ℝ) ^ 2 * r.Q = r.Q ^ 3 := by ring
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
      ≤ 2 ^ 137 * (r.logZ * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, hlg2, hlgX⟩ := kn_shape_base hr
  obtain ⟨hι0, -⟩ := shape_pos hr
  obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
  obtain ⟨hSC0, hSC⟩ := knSC_bounds hr
  obtain ⟨hHR1, hHR⟩ := knHR_bounds hr
  obtain ⟨hrp1, hrp⟩ := knRp_bounds hr
  obtain ⟨-, ht2l⟩ := reads_t2_ge hr
  have hp : (0 : ℝ) ≤ r.Q ^ 2 * r.A * knX r * (1 / r.ξc) * r.logZ ^ 4 * (1 / r.s₀) ^ 6
      * r.ιInv ^ 2 := by positivity
  have hlgZ := knLg_logZ hr
  have hRBl := knRBb_le hr
  have hFV' := knFV_le hr
  have hQι := q_cube_le hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hu : (knRp r : ℝ) * ((r.Q : ℝ) + 1) ≤ 64 * r.Q * r.logZ := by
    have := mul_le_mul (hrp.trans (show 16 * knLg r ≤ 16 * r.logZ by linarith))
      (show (r.Q : ℝ) + 1 ≤ 4 * r.Q by linarith) (by linarith) (by linarith)
    linarith
  have hHR' : knHR r ≤ 2 ^ 23 * r.A * knX r * (1 / r.ξc) * r.logZ := by
    have := mul_le_mul_of_nonneg_left hΛ
      (show (0 : ℝ) ≤ 2 ^ 14 * r.A * knX r * (1 / r.ξc) by positivity)
    linarith
  have hSC' : knSC r ≤ 2 ^ 89 * (r.Q ^ 2 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by
    have := mul_le_mul_of_nonneg_left hRBl (show (0 : ℝ) ≤ 2 ^ 6 * r.Q by positivity)
    linarith
  have s5 := sampler_le hQ hFV0 hFV' hSC0 hSC' (by linarith) hHR'
    (mul_nonneg (Nat.cast_nonneg _) (by linarith)) hu
  have a2 := mul_le_mul_of_nonneg_left hQι hp
  have a3 := mul_le_mul_of_nonneg_left ht2l (show (0 : ℝ) ≤ r.logZ by linarith)
  have e1 : (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2)
        * ((2 * r.Q) * (2 ^ 89 * (r.Q ^ 2 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4))
          * (2 ^ 23 * r.A * knX r * (1 / r.ξc) * r.logZ) * (64 * r.Q * r.logZ)))
      = 2 ^ 137 * (r.Q ^ 2 * r.A * knX r * (1 / r.ξc) * r.logZ ^ 4 * (1 / r.s₀) ^ 6
        * r.ιInv ^ 2 * r.Q ^ 3) := by ring
  have e2 : r.Q ^ 2 * r.A * knX r * (1 / r.ξc) * r.logZ ^ 4 * (1 / r.s₀) ^ 6 * r.ιInv ^ 2
      * (r.A * (1 / r.s₀) ^ 2 * r.ιInv)
      = r.logZ * (r.Q ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc) * r.logZ ^ 3 * (1 / r.s₀) ^ 8
        * r.ιInv ^ 3) := by ring
  linarith

/-- The walk-start table's reads against `(k + 1) Q A nHeavy logZ / s₀²`. -/
lemma reads_table_le (hr : r ∈ rlStarRunRegion) :
    knFV r * ((r.A * (knK r * knTrie r) + 1) * (r.Q + 2))
      ≤ 2 ^ 179 * (((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
  obtain ⟨hQ, hA, hX, -, hLX, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨-, -, -, -, hH0, -⟩ := shape_pos hr
  obtain ⟨-, hHv⟩ := shape_lower hr
  obtain ⟨-, -, hFV0, -⟩ := knPB_bounds hr
  obtain ⟨hRB0, -⟩ := knRB_bounds hr
  have htrie := knTrie_le hr
  have hFV' := knFV_le hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hK0 : (0 : ℝ) ≤ knK r := Nat.cast_nonneg _
  have hT0 : 0 ≤ knTrie r := by unfold knTrie; positivity
  have a1 : (knK r : ℝ) * knTrie r ≤ knK r * (2 ^ 159 * r.nHeavy) :=
    mul_le_mul_of_nonneg_left htrie hK0
  have b1 : (1 : ℝ) ≤ r.nHeavy := by
    have : (1 : ℝ) ≤ r.Q ^ 7 * r.A ^ 4 * knX r ^ 4 * (1 / r.ξc) ^ 4 * r.logZ ^ 2
        * (1 / r.s₀) ^ 6 :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
          (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) (one_le_pow₀ (by linarith)))
          (one_le_pow₀ hX)) (one_le_pow₀ iξ)) (one_le_pow₀ (by linarith)))
        (one_le_pow₀ (by linarith))
    linarith
  have a2 : (1 : ℝ) ≤ r.A * r.nHeavy := one_le_mul_of_one_le_of_one_le (by linarith) b1
  have a3 : (r.A : ℝ) * (knK r * knTrie r) + 1
      ≤ 2 ^ 160 * (((knK r : ℝ) + 1) * (r.A * r.nHeavy)) := by
    have := mul_le_mul_of_nonneg_left a1 (show (0 : ℝ) ≤ r.A by positivity)
    have e : (r.A : ℝ) * (knK r * (2 ^ 159 * r.nHeavy)) = 2 ^ 159 * (knK r * (r.A * r.nHeavy)) := by
      ring
    have e2 : 2 ^ 160 * (((knK r : ℝ) + 1) * (r.A * r.nHeavy))
        = 2 ^ 160 * (knK r * (r.A * r.nHeavy)) + 2 ^ 160 * (r.A * r.nHeavy) := by ring
    have : (0 : ℝ) ≤ knK r * (r.A * r.nHeavy) := by positivity
    linarith
  have a4 := mul_le_mul hFV' (mul_le_mul a3 (show (r.Q : ℝ) + 2 ≤ 3 * r.Q by linarith)
    (by positivity) (by positivity)) (by positivity) (by positivity)
  have e : 2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2 * (2 ^ 160 * (((knK r : ℝ) + 1) * (r.A * r.nHeavy))
      * (3 * r.Q))
      = 3 * 2 ^ 177 * (((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2)) := by
    ring
  have : (0 : ℝ) ≤ ((knK r : ℝ) + 1) * (r.Q * r.A * r.nHeavy * r.logZ * (1 / r.s₀) ^ 2) := by
    positivity
  linarith

lemma knReads_le (hr : r ∈ rlStarRunRegion) : knReads r ≤ 2 ^ 180 * r.nReads := by
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

set_option maxHeartbeats 800000 in -- `linarith` over the two inverse terms
lemma knP₀_inv_le (hr : r ∈ rlStarRunRegion) :
    1 / knP₀ r ≤ 2 ^ 155 * (r.A * (r.L + r.Q) * r.nCap / r.ξc) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, hp, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hc1, -⟩ := knCapInv_bounds hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨-, hC0, -⟩ := shape_pos hr
  have hc := knCapInv_le hr
  have hMc := knMcD_pos hr
  have hcp := knCapP_pos hr
  have hcg := knCapP_ge hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hι1 := knιInv_one hr
  have hιX := knιInv_ge hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  set M := r.A * knX r * (1 / r.ξc) with hMd
  have hM1 : 1 ≤ M := one_le_mul_of_one_le_of_one_le
    (one_le_mul_of_one_le_of_one_le (by linarith) hX) iξ
  have hpa : 0 < knCapP r * knε r / 12288 := by positivity
  have h1 : 1 / knP₀ r ≤ 1 / (knCapP r * knε r / 12288) + 1 / knMcD r := by
    unfold knP₀
    exact one_div_min_le hpa hMc
  have hic : 1 / knCapP r ≤ knCapInv r := (one_div_le hcp (by linarith)).2 hcg
  have ea : 1 / (knCapP r * knε r / 12288) = 12288 * (1 / knCapP r) * (1 / knε r) := by
    field_simp
  rw [ea] at h1
  have hiε' : 1 / knε r ≤ 30 * M := by rw [hMd]; linarith
  have b1 : 12288 * (1 / knCapP r) * (1 / knε r) ≤ 12288 * knCapInv r * (30 * M) := by gcongr
  -- the misread term
  have b4 : 1 / knMcD r ≤ knCapInv r * M := by
    have hlA : log (r.A : ℝ) ≤ r.A := by
      have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < r.A by linarith); linarith
    have e : 1 / knMcD r = 2 ^ 16 * knΛ r * knX r * log r.A * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 := by
      unfold knMcD; rw [one_div_div]; ring
    rw [e]
    have c1 : 2 ^ 16 * knΛ r * knX r * log r.A * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2
        ≤ 2 ^ 16 * knΛ r * knX r * r.A * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 := by
      have : 0 ≤ log (r.A : ℝ) := Real.log_nonneg (by linarith)
      gcongr
    have c2 : 2 ^ 60 * knΛ r * (1 / r.s₀) ^ 2 * knιInv r ≤ knCapInv r := by
      have d1 : (1 / r.s₀) ^ 2 ≤ (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2) := by
        have : (1 : ℝ) ≤ (1 / r.s₀) ^ 4 := one_le_pow₀ (by linarith)
        have : (1 / r.s₀) ^ 2 ≤ (1 / r.s₀) ^ 2 + r.Q ^ 2 := by
          have := sq_nonneg (r.Q : ℝ); linarith
        exact this.trans (le_mul_of_one_le_left (by positivity) ‹(1 : ℝ) ≤ (1 / r.s₀) ^ 4›)
      have d2 : knιInv r ≤ r.Q * knιInv r ^ 3 := by
        have : knιInv r ≤ knιInv r ^ 3 := le_self_pow₀ hι1 (by norm_num)
        have : knιInv r ^ 3 ≤ r.Q * knιInv r ^ 3 :=
          le_mul_of_one_le_left (pow_nonneg (by linarith) 3) hQ
        linarith
      have d3 := mul_le_mul d1 d2 (by linarith) (by positivity)
      have d4 := mul_le_mul_of_nonneg_left d3 (show 0 ≤ 2 ^ 60 * knΛ r by positivity)
      have d5 : (0 : ℝ) ≤ 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
      unfold knCapInv
      linarith
    have c3 : 1 / r.ξc ≤ knιInv r := by
      have : 1 / r.ξc ≤ 2 ^ 23 * knX r * (1 / r.ξc) := by
        have : (1 : ℝ) ≤ 2 ^ 23 * knX r := by linarith
        exact le_mul_of_one_le_left (by positivity) this
      linarith
    have c4 := mul_le_mul_of_nonneg_left c3
      (show 0 ≤ 2 ^ 60 * knΛ r * (1 / r.s₀) ^ 2 by positivity)
    have c5 : 2 ^ 60 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.ξc) ≤ knCapInv r := c4.trans c2
    have c6 := mul_le_mul_of_nonneg_right c5 (show 0 ≤ M by linarith)
    have e2 : 2 ^ 60 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.ξc) * M
        = 2 ^ 60 * knΛ r * knX r * r.A * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 := by rw [hMd]; ring
    rw [e2] at c6
    have c8 : 2 ^ 16 * knΛ r * knX r * r.A * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2
        ≤ 2 ^ 60 * knΛ r * knX r * r.A * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 := by
      gcongr <;> norm_num
    exact c1.trans (c8.trans c6)
  have hsum : 1 / knP₀ r ≤ 2 ^ 20 * (knCapInv r * M) := by
    have : M ≤ knCapInv r * M := le_mul_of_one_le_left (by linarith) hc1
    have e : 12288 * knCapInv r * (30 * M) = 368640 * (knCapInv r * M) := by ring
    have : (0 : ℝ) ≤ knCapInv r * M := by positivity
    linarith
  have e : r.A * (r.L + r.Q) * r.nCap / r.ξc = r.nCap * M := by rw [hMd]; unfold knX; ring
  rw [e]
  have := mul_le_mul_of_nonneg_right hc (show 0 ≤ M by linarith)
  linarith

end totals

end OrthoDFA

end
