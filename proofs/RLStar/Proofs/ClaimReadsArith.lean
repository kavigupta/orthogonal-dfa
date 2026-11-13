import RLStar.Proofs.ClaimRLStarArith

/-!
# The knobs' totals against the claim's shapes

`knΛ ≤ 2⁹ logZ` and `knιInv ≤ 2⁵⁰ ιInv`, so each total of `ClaimRLStarArith` is at most a
constant times its shape in `RLStarClaim`.
-/

set_option exponentiation.threshold 300

noncomputable section

namespace OrthoDFA

open Real

section shapes

variable {r : RLStarRun}

lemma shape_ιInv (r : RLStarRun) :
    r.ιInv = r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ * knLg r := by
  unfold RLStarRun.ιInv knLg knX; ring

lemma shape_nList (r : RLStarRun) :
    r.nList = r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
      + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by
  unfold RLStarRun.nList; ring

lemma shape_nPref (r : RLStarRun) :
    r.nPref = r.Q ^ 3 * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4 := by
  unfold RLStarRun.nPref; ring

lemma shape_nSuff (r : RLStarRun) : r.nSuff = r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) := by
  unfold RLStarRun.nSuff; ring

lemma shape_nProbe (r : RLStarRun) :
    r.nProbe = r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ * r.nList := by
  unfold RLStarRun.nProbe knX; ring

lemma shape_nReads (r : RLStarRun) :
    r.nReads = r.nPref * r.nSuff / r.Q + r.nProbe * r.logZ * (1 / r.s₀) ^ 2 * r.Q * knLg r := by
  unfold RLStarRun.nReads knLg knX; ring

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

lemma shape_pos (hr : r ∈ rlStarRunRegion) :
    0 ≤ r.ιInv ∧ 0 ≤ r.nList ∧ 0 ≤ r.nPref ∧ 0 ≤ r.nSuff ∧ 0 ≤ r.nProbe ∧ 0 ≤ r.nReads := by
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, -, hs, -⟩ := kn_base hr
  obtain ⟨hl2, -, hlg, -⟩ := kn_shape_base hr
  have hlg0 : 0 < knLg r := by linarith [Real.log_two_gt_d9]
  have hι : 0 ≤ r.ιInv := by rw [shape_ιInv]; positivity
  have hL : 0 ≤ r.nList := by rw [shape_nList]; positivity
  have hP : 0 ≤ r.nPref := by rw [shape_nPref]; positivity
  have hS : 0 ≤ r.nSuff := by rw [shape_nSuff]; positivity
  have hPr : 0 ≤ r.nProbe := by rw [shape_nProbe]; positivity
  refine ⟨hι, hL, hP, hS, hPr, ?_⟩
  rw [shape_nReads]
  positivity

end shapes

section knobs

variable {r : RLStarRun}

lemma knCapInv_one (hr : r ∈ rlStarRunRegion) : 1 ≤ knCapInv r := by
  obtain ⟨hQ, -⟩ := kn_base hr
  obtain ⟨hcap, -⟩ := knCapInv_bounds hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hι1 := knιInv_one hr
  have h1 : (1 : ℝ) ≤ r.Q ^ 3 := one_le_pow₀ hQ
  have h2 : (1 : ℝ) ≤ r.Q ^ 3 * knΛ r := one_le_mul_of_one_le_of_one_le h1 (by linarith)
  have h3 : (1 : ℝ) ≤ r.Q ^ 3 * knΛ r * knιInv r := one_le_mul_of_one_le_of_one_le h2 hι1
  linarith

lemma knιInv_le (hr : r ∈ rlStarRunRegion) : knιInv r ≤ 2 ^ 50 * r.ιInv := by
  obtain ⟨hl2, hΛ, hlg2, -⟩ := kn_shape_base hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  have hlg0 : 0 < knLg r := by linarith [Real.log_two_gt_d9]
  rw [shape_ιInv]
  have h0 : 0 ≤ (r.Q : ℝ) ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knLg r := by positivity
  have h1 := mul_le_mul_of_nonneg_left hΛ h0
  have h2 : 0 ≤ (r.Q : ℝ) ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knLg r * r.logZ :=
    mul_nonneg h0 (by linarith)
  unfold knιInv
  linarith

lemma knRBb_le (hr : r ∈ rlStarRunRegion) :
    knRBb r ≤ 2 ^ 139 * (r.Q * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hι := knιInv_le hr
  have hι1 := knιInv_one hr
  have hι2 : knιInv r ^ 2 ≤ (2 ^ 50 * r.ιInv) ^ 2 := pow_le_pow_left₀ (by linarith) hι 2
  unfold knRBb
  calc 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2
      ≤ 2 ^ 30 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 4 * (2 ^ 50 * r.ιInv) ^ 2 := by gcongr
    _ = 2 ^ 139 * (r.Q * r.logZ * r.ιInv ^ 2 * (1 / r.s₀) ^ 4) := by ring

lemma knCapInv_le (hr : r ∈ rlStarRunRegion) : knCapInv r ≤ 2 ^ 219 * r.nList := by
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, hp, -, hs, -⟩ := kn_base hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hι := knιInv_le hr
  have hι1 := knιInv_one hr
  have hι3 : knιInv r ^ 3 ≤ (2 ^ 50 * r.ιInv) ^ 3 := pow_le_pow_left₀ (by linarith) hι 3
  have hιs : 0 ≤ r.ιInv := (shape_pos hr).1
  rw [shape_nList]
  unfold knCapInv
  have h0 : (0 : ℝ) ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
  calc 2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2
      ≤ 2 ^ 60 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 4 * (2 ^ 50 * r.ιInv) ^ 3
          * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + 2 ^ 30 * r.Q * (2 ^ 9 * r.logZ) * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by gcongr
    _ = 2 ^ 219 * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2))
        + 2 ^ 39 * (r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2) := by ring
    _ ≤ 2 ^ 219 * (r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
        + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2) := by linarith

lemma knNh_le (hr : r ∈ rlStarRunRegion) : (knNh r : ℝ) ≤ 2 ^ 221 * r.nList := by
  obtain ⟨-, hnh, -⟩ := knNh_bounds hr
  have hc := knCapInv_le hr
  have hc1 := knCapInv_one hr
  linarith

/-- The prefix count's bound against the list size's. -/
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

lemma knB_nh (hr : r ∈ rlStarRunRegion) : (knB r : ℝ) ≤ knNh r := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨-, hb, -⟩ := knB_bounds hr
  obtain ⟨hcap, -⟩ := knCapInv_bounds hr
  obtain ⟨hnh, -⟩ := knNh_bounds hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hι := knιInv_ge' hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  -- `A X (1/ξc) ≤ Q³ A² X⁴ (1/ξc)²`
  have h1 : (r.A : ℝ) * knX r * (1 / r.ξc) ≤ r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 := by
    have a1 : (1 : ℝ) ≤ r.Q ^ 3 := one_le_pow₀ hQ
    have a2 : (r.A : ℝ) ≤ r.A ^ 2 := le_self_pow₀ (by linarith) (by norm_num)
    have a3 : knX r ≤ knX r ^ 4 := le_self_pow₀ hX (by norm_num)
    have a4 : 1 / r.ξc ≤ (1 / r.ξc) ^ 2 := le_self_pow₀ iξ (by norm_num)
    have b1 : (r.A : ℝ) ≤ r.Q ^ 3 * r.A ^ 2 := a2.trans (le_mul_of_one_le_left (by positivity) a1)
    have b2 : (r.A : ℝ) * knX r ≤ r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 :=
      mul_le_mul b1 a3 (by linarith) (by positivity)
    exact mul_le_mul b2 a4 (by positivity) (by positivity)
  have h2 := mul_le_mul_of_nonneg_left h1 (show (0 : ℝ) ≤ 2 ^ 13 * knΛ r by positivity)
  have h3 := mul_le_mul_of_nonneg_left hι (show (0 : ℝ) ≤ 2 ^ 64 * r.Q ^ 3 * knΛ r by positivity)
  have h4 : (0 : ℝ) ≤ r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r := by positivity
  have h5 : (1 : ℝ) ≤ r.Q ^ 3 := one_le_pow₀ hQ
  have h6 := mul_le_mul_of_nonneg_left h5 h4
  linarith

end knobs

section totals

variable {r : RLStarRun}

lemma knPref_le (hr : r ∈ rlStarRunRegion) : knPref r ≤ 2 ^ 145 * r.nPref := by
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

lemma knProbes_le (hr : r ∈ rlStarRunRegion) : knProbes r ≤ 2 ^ 259 * r.nProbe := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hb1, hb, -⟩ := knB_bounds hr
  obtain ⟨hB₃0, -⟩ := knB₃_bounds hr
  obtain ⟨-, -, hnh1, hnhB, -⟩ := knNh_bounds hr
  obtain ⟨-, hJX, -⟩ := knJ_bounds hr
  obtain ⟨hl2, hΛ, -⟩ := kn_shape_base hr
  obtain ⟨-, hL, -⟩ := shape_pos hr
  have hnh := knNh_le hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hV0 : (0 : ℝ) ≤ (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2)) := by positivity
  have h1 : knB₃ r + knJ r * knB r ≤ 3 * knX r * knNh r * knB r := by
    have a2 := le_mul_of_one_le_right (show (0 : ℝ) ≤ knNh r by linarith)
      (one_le_mul_of_one_le_of_one_le hX hb1)
    have a3 : (knJ r : ℝ) * knB r ≤ 2 * knX r * knNh r * knB r :=
      mul_le_mul_of_nonneg_right hJX (by linarith)
    linarith
  have h2 : 3 * knX r * knNh r * knB r
      ≤ 3 * knX r * (2 ^ 221 * r.nList)
        * (2 ^ 13 * r.A * knX r * (1 / r.ξc) * (2 ^ 9 * r.logZ)) := by
    gcongr
    exact hb.trans (by gcongr)
  rw [shape_nProbe]
  unfold knProbes
  calc ((r.Q : ℝ) + 1) * (knE r * (knB₃ r + knJ r * knB r))
      ≤ (2 * r.Q) * ((2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)) * (3 * knX r * (2 ^ 221 * r.nList)
          * (2 ^ 13 * r.A * knX r * (1 / r.ξc) * (2 ^ 9 * r.logZ)))) := by
        gcongr
        · linarith
        · exact h1.trans h2
    _ = 3 * 2 ^ 256 * (r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ * r.nList) := by ring
    _ ≤ 2 ^ 259 * (r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * r.logZ * r.nList) := by
        gcongr; norm_num

set_option maxHeartbeats 800000 in -- `linarith` over the three terms' products
lemma knReads_le (hr : r ∈ rlStarRunRegion) : knReads r ≤ 2 ^ 283 * r.nReads := by
  obtain ⟨hQ, hA, hX, -, hLX, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, hΛ, hlg2, hlgX⟩ := kn_shape_base hr
  obtain ⟨hι0, hL0, hP0, hS0, hPr0, -⟩ := shape_pos hr
  obtain ⟨hPB28, -, hFV0, hFV⟩ := knPB_bounds hr
  obtain ⟨hRB0, hRB, -⟩ := knRB_bounds hr
  obtain ⟨-, hNr, hNr1⟩ := knNr_bounds hr
  obtain ⟨hnhc, -, hnh1, -⟩ := knNh_bounds hr
  have hnh := knNh_le hr
  have hRBc := knRBb_cap hr
  have hQR : knRBb r ≤ r.Q * knRBb r := le_mul_of_one_le_left (by linarith [hRB0.trans hRB]) hQ
  have hbn := knB_nh hr
  have hpref := knPref_le hr
  have hsuff := knSuff_le hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hlt := Real.log_two_gt_d9
  have hlg0 : 0 < knLg r := by linarith
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hQ0 : (0 : ℝ) < r.Q := by linarith
  have hL0' : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  set t2 := r.nProbe * r.logZ * (1 / r.s₀) ^ 2 * r.Q * knLg r with ht2
  have ht2' : t2 = r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 * knLg r
      * r.logZ ^ 2 * r.nList := by rw [ht2, shape_nProbe]; ring
  have ht20 : 0 ≤ t2 := by rw [ht2']; positivity
  set c1 := r.nPref * r.nSuff / r.Q with hc1
  have hc10 : 0 ≤ c1 := by positivity
  rw [shape_nReads, ← hc1, ← ht2]
  unfold knReads lazyReads
  -- the clusterings
  have hi : ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r * (knPB r + 1 + 1)) ≤ 2 ^ 167 * c1 := by
    have a0 : (0 : ℝ) ≤ ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r) := by positivity
    have a1 : (r.Q : ℝ) * (knPB r + 1 + 1) ≤ 2 * (((r.Q : ℝ) + 1) * (knPB r + 1)) := by
      nlinarith [mul_nonneg hQ0.le (show (0 : ℝ) ≤ knPB r by linarith)]
    have a2 := mul_le_mul_of_nonneg_left a1 a0
    have a3 : knPref r * knSuff r ≤ (2 ^ 145 * r.nPref) * (2 ^ 21 * r.nSuff) :=
      mul_le_mul hpref hsuff (by unfold knSuff; positivity) (mul_nonneg (by norm_num) hP0)
    have a4 : (r.Q : ℝ) * (((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r * (knPB r + 1 + 1)))
        ≤ 2 ^ 167 * (r.nPref * r.nSuff) := by
      have e1 : knPref r * knSuff r = ((r.Q : ℝ) + 1) * (6 * (r.Q + 2) * knRB r)
          * (((r.Q : ℝ) + 1) * (knPB r + 1)) := by unfold knPref knSuff; ring
      rw [e1] at a3
      linarith
    rw [hc1, mul_div_assoc', le_div_iff₀ hQ0]
    linarith
  -- the search
  have hii : ((r.Q : ℝ) + 1) * (knFV r * knNr r) ≤ 2 ^ 282 * t2 := by
    rw [ht2']
    have a1 : knFV r ≤ 2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2 := by
      have := mul_le_mul_of_nonneg_right hΛ (show (0 : ℝ) ≤ 2 ^ 8 * (1 / r.s₀) ^ 2 by positivity)
      linarith
    have a2 : (knNr r : ℝ) ≤ 2 ^ 34 * r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2
        * (2 ^ 9 * r.logZ) * knLg r * (2 ^ 221 * r.nList) := by
      refine hNr.trans ?_
      gcongr
    calc ((r.Q : ℝ) + 1) * (knFV r * knNr r)
        ≤ (2 * r.Q) * ((2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2) * (2 ^ 34 * r.Q * r.A ^ 2 * knX r ^ 4
          * (1 / r.ξc) ^ 2 * (2 ^ 9 * r.logZ) * knLg r * (2 ^ 221 * r.nList))) := by
          gcongr
          linarith
      _ = 2 ^ 282 * (r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 * knLg r
          * r.logZ ^ 2 * r.nList) := by ring
  -- the table
  have hiii : knFV r * ((r.A * (r.L * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) + 1)
      * (r.Q + 2)) ≤ 2 ^ 243 * t2 := by
    have a1 : 6 * ((r.Q : ℝ) + 2) * knRB r ≤ knNh r := by
      have := mul_le_mul_of_nonneg_left hRB (show (0 : ℝ) ≤ 6 * (r.Q + 2) by positivity)
      linarith
    have a2 : (r.L : ℝ) * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r)) ≤ knX r * (5 * knNh r) :=
      mul_le_mul hLX (by linarith) (by positivity) (by linarith)
    have a3 : (1 : ℝ) ≤ r.A * knX r * knNh r :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hX) hnh1
    have a4 : (r.A : ℝ) * (r.L * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) + 1
        ≤ 6 * (r.A * knX r * knNh r) := by
      have := mul_le_mul_of_nonneg_left a2 (show (0 : ℝ) ≤ r.A by positivity)
      linarith
    have a5 : knFV r * ((r.A * (r.L * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) + 1)
        * (r.Q + 2)) ≤ (2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2) * ((6 * (r.A * knX r
          * (2 ^ 221 * r.nList))) * (3 * r.Q)) := by
      have b1 : knFV r ≤ 2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2 := by
        have := mul_le_mul_of_nonneg_right hΛ (show (0 : ℝ) ≤ 2 ^ 8 * (1 / r.s₀) ^ 2 by positivity)
        linarith
      have b2 : r.A * knX r * (knNh r : ℝ) ≤ r.A * knX r * (2 ^ 221 * r.nList) :=
        mul_le_mul_of_nonneg_left hnh (by positivity)
      have b3 : (r.A : ℝ) * (r.L * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) + 1
          ≤ 6 * (r.A * knX r * (2 ^ 221 * r.nList)) := by linarith
      have b4 : (r.Q : ℝ) + 2 ≤ 3 * r.Q := by linarith
      have b5 : (0 : ℝ) ≤ (r.A : ℝ) * (r.L * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) + 1 :=
        by positivity
      have b6 : (0 : ℝ) ≤ 6 * (r.A * knX r * (2 ^ 221 * r.nList)) :=
        mul_nonneg (by norm_num) (mul_nonneg (mul_nonneg (by linarith) (by linarith))
          (mul_nonneg (pow_nonneg (by norm_num) _) hL0))
      have b7 : (0 : ℝ) ≤ 2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2 :=
        mul_nonneg (mul_nonneg (pow_nonneg (by norm_num) _) (by linarith))
          (pow_nonneg (by linarith) _)
      exact mul_le_mul b1 (mul_le_mul b3 b4 (by linarith) b6)
        (mul_nonneg b5 (by linarith)) b7
    -- `Q A X logZ (1/s₀)² nList ≤ t2`
    have a6 : (1 : ℝ) ≤ r.Q * r.A * knX r ^ 3 * (1 / r.ξc) ^ 2 * knLg r * r.logZ := by
      have b1 : (1 : ℝ) ≤ r.Q * r.A := one_le_mul_of_one_le_of_one_le hQ (by linarith)
      have b2 : (1 : ℝ) ≤ r.Q * r.A * knX r ^ 3 :=
        one_le_mul_of_one_le_of_one_le b1 (one_le_pow₀ hX)
      have b3 : (1 : ℝ) ≤ r.Q * r.A * knX r ^ 3 * (1 / r.ξc) ^ 2 :=
        one_le_mul_of_one_le_of_one_le b2 (one_le_pow₀ iξ)
      have b4 : (1 : ℝ) ≤ knLg r * r.logZ := by
        have := mul_le_mul (show (1 / 2 : ℝ) ≤ knLg r by linarith) hl2 (by norm_num) (by linarith)
        linarith
      have := one_le_mul_of_one_le_of_one_le b3 b4
      linarith
    have a7 : (0 : ℝ) ≤ r.Q * r.A * knX r * r.logZ * (1 / r.s₀) ^ 2 * r.nList := by positivity
    have a8 := le_mul_of_one_le_right a7 a6
    rw [ht2']
    have e : (2 ^ 17 * r.logZ * (1 / r.s₀) ^ 2) * ((6 * (r.A * knX r * (2 ^ 221 * r.nList)))
        * (3 * r.Q)) = 18 * 2 ^ 238 * (r.Q * r.A * knX r * r.logZ * (1 / r.s₀) ^ 2 * r.nList) := by
      ring
    have e2 : r.Q * r.A * knX r * r.logZ * (1 / r.s₀) ^ 2 * r.nList
        * (r.Q * r.A * knX r ^ 3 * (1 / r.ξc) ^ 2 * knLg r * r.logZ)
        = r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 * knLg r
          * r.logZ ^ 2 * r.nList := by ring
    calc _ ≤ _ := a5
      _ = 18 * 2 ^ 238 * (r.Q * r.A * knX r * r.logZ * (1 / r.s₀) ^ 2 * r.nList) := e
      _ ≤ 18 * 2 ^ 238 * (r.Q * r.A * knX r * r.logZ * (1 / r.s₀) ^ 2 * r.nList
          * (r.Q * r.A * knX r ^ 3 * (1 / r.ξc) ^ 2 * knLg r * r.logZ)) :=
          mul_le_mul_of_nonneg_left a8 (by norm_num)
      _ = 18 * 2 ^ 238 * (r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 * knLg r
          * r.logZ ^ 2 * r.nList) := by rw [e2]
      _ ≤ 2 ^ 243 * (r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * (1 / r.s₀) ^ 2 * knLg r
          * r.logZ ^ 2 * r.nList) := mul_le_mul_of_nonneg_right (by norm_num) (by positivity)
  linarith

set_option maxHeartbeats 800000 in -- `linarith` over the four inverse terms
lemma knP₀_inv_le (hr : r ∈ rlStarRunRegion) :
    1 / knP₀ r ≤ 2 ^ 236 * (r.A * (r.L + r.Q) * r.nList / r.ξc) := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, hp, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hcap, -⟩ := knCapInv_bounds hr
  obtain ⟨hRB0, hRB, hRBb1⟩ := knRB_bounds hr
  obtain ⟨hl2, -⟩ := kn_shape_base hr
  obtain ⟨-, hL0, -⟩ := shape_pos hr
  have hc := knCapInv_le hr
  have hc1 := knCapInv_one hr
  have hRBc := knRBb_cap hr
  have hMc := knMcD_pos hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hι1 := knιInv_one hr
  have hιX := knιInv_ge hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  set M := r.A * knX r * (1 / r.ξc) with hMd
  have hM1 : 1 ≤ M := one_le_mul_of_one_le_of_one_le
    (one_le_mul_of_one_le_of_one_le (by linarith) hX) iξ
  have hpa : 0 < knε r / (1536 * knCapInv r) := by positivity
  have hpb : 0 < 1 / (12 * (r.Q + 2) * knRBb r) := by positivity
  have hpc : 0 < knε r / (768 * (2 * knCapInv r + 1)) := by positivity
  have h1 : 1 / knP₀ r ≤ 1 / (knε r / (1536 * knCapInv r)) + 1 / (1 / (12 * (r.Q + 2) * knRBb r))
      + (1 / (knε r / (768 * (2 * knCapInv r + 1))) + 1 / knMcD r) := by
    unfold knP₀
    refine (one_div_min_le (lt_min hpa hpb) (lt_min hpc hMc)).trans ?_
    exact add_le_add (one_div_min_le hpa hpb) (one_div_min_le hpc hMc)
  have ea : 1 / (knε r / (1536 * knCapInv r)) = 1536 * knCapInv r * (1 / knε r) := by
    rw [one_div_div]; ring
  have ec : 1 / (knε r / (768 * (2 * knCapInv r + 1)))
      = 768 * (2 * knCapInv r + 1) * (1 / knε r) := by
    rw [one_div_div]; ring
  rw [ea, ec, one_div_one_div] at h1
  have hiε' : 1 / knε r ≤ 30 * M := by rw [hMd]; linarith
  have b1 := mul_le_mul_of_nonneg_left hiε' (show 0 ≤ 1536 * knCapInv r by positivity)
  have b2 := mul_le_mul_of_nonneg_left hiε' (show 0 ≤ 768 * (2 * knCapInv r + 1) by positivity)
  have b3 : 12 * ((r.Q : ℝ) + 2) * knRBb r ≤ knCapInv r * M := by
    have := le_mul_of_one_le_right (show 0 ≤ knCapInv r by linarith) hM1
    have : knRBb r ≤ r.Q * knRBb r := le_mul_of_one_le_left (by linarith) hQ
    linarith
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
    -- `capInv ≥ 2⁶⁰ Λ (1/s₀)² ιInv` and `ιInv ≥ 1/ξc`
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
      have : 1 / r.ξc ≤ 2 ^ 46 * knX r * (1 / r.ξc) := by
        have : (1 : ℝ) ≤ 2 ^ 46 * knX r := by linarith
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
  have hsum : 1 / knP₀ r ≤ 2 ^ 17 * (knCapInv r * M) := by
    have : M ≤ knCapInv r * M := le_mul_of_one_le_left (by linarith) hc1
    linarith
  have e : r.A * (r.L + r.Q) * r.nList / r.ξc = r.nList * M := by rw [hMd]; unfold knX; ring
  rw [e]
  have := mul_le_mul_of_nonneg_right hc (show 0 ≤ M by linarith)
  linarith

end totals

end OrthoDFA

end
