import RLStar.Proofs.ClaimReadsArith
import RLStar.Proofs.DFABridge

/-!
# The uniform claims from the general one

Uniform strings hold `|Σ|^-k` on any first `k` letters, and `1/p₀Tol ≤ C S` for the shape `S` of
`RLStarClaim`, so `k ≥ k₀ = log (C₀ S)/log |Σ|` gives `1/p₀Tol ≤ |Σ|^k`. Each factor of `S` is at
most `U = 8 |Σ| W`, `W = (L + |Q|) log(1/δ)/(ξc pAP s₀)`, and `S ≤ 3 U²⁶`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

/-- The shape of `1/p₀Tol`. -/
def tolShape (r : RLStarRun) : ℝ := r.A * (r.L + r.Q) * r.nCap / r.ξc

/-- `W = (L + |Q|) log(1/δ)/(ξc pAP s₀)`. -/
def uW (r : RLStarRun) : ℝ := (r.L + r.Q) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀)

section uniform

variable {r : RLStarRun}

set_option maxHeartbeats 800000 in -- `linarith` over the factors' bounds
lemma tolShape_le (hr : r ∈ rlStarRunRegion) :
    1 ≤ uW r ∧ 16 ≤ tolShape r ∧ tolShape r ≤ 3 * (8 * r.A * uW r) ^ 26 := by
  obtain ⟨hQ, hA, hX, hQX, -, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hl2, -, hlg2, hlgX⟩ := kn_shape_base hr
  obtain ⟨hι0, hL0, -⟩ := shape_pos hr
  have hlt := Real.log_two_gt_d9
  have eW : uW r = knX r * log (1 / r.δ) * (1 / (r.ξc * r.pAP * r.s₀)) := by
    unfold uW knX; ring
  have e1 : (r.L + r.Q : ℝ) * r.A / (r.δ * r.ξc * r.pAP * r.s₀)
      = knX r * (1 / (r.ξc * r.pAP * r.s₀)) * r.A * (1 / r.δ) := by
    unfold knX; ring
  have hDξ : r.ξc * r.pAP * r.s₀ ≤ r.ξc := by
    have : r.pAP * r.s₀ ≤ 1 := mul_le_one₀ hp1 hs.le (by linarith)
    calc r.ξc * r.pAP * r.s₀ = r.ξc * (r.pAP * r.s₀) := by ring
      _ ≤ r.ξc * 1 := by gcongr
      _ = r.ξc := mul_one _
  have hDs : r.ξc * r.pAP * r.s₀ ≤ r.s₀ := by
    have : r.ξc * r.pAP ≤ 1 := mul_le_one₀ hξ1 hp.le hp1
    calc r.ξc * r.pAP * r.s₀ ≤ 1 * r.s₀ := by gcongr
      _ = r.s₀ := one_mul _
  have hDp : r.ξc * r.pAP * r.s₀ ≤ r.pAP := by
    have : r.ξc * r.s₀ ≤ 1 := mul_le_one₀ hξ1 hs.le (by linarith)
    calc r.ξc * r.pAP * r.s₀ = r.pAP * (r.ξc * r.s₀) := by ring
      _ ≤ r.pAP * 1 := by gcongr
      _ = r.pAP := mul_one _
  have hD1 : r.ξc * r.pAP * r.s₀ ≤ 1 / 2 := hDs.trans hs1
  obtain ⟨D, hDd⟩ : ∃ D, D = r.ξc * r.pAP * r.s₀ := ⟨_, rfl⟩
  rw [← hDd] at eW e1 hDξ hDs hDp hD1
  have hD0 : 0 < D := by rw [hDd]; positivity
  have iD : 2 ≤ 1 / D := by rw [le_div_iff₀ hD0]; linarith
  obtain ⟨lδ, hlδd⟩ : ∃ l, l = log (1 / r.δ) := ⟨_, rfl⟩
  rw [← hlδd] at eW
  have hlδ : 1 / 2 ≤ lδ := by
    rw [hlδd]; exact half_le_log ((le_div_iff₀ hδ).2 (by linarith))
  obtain ⟨W, hWd⟩ : ∃ W, W = uW r := ⟨_, rfl⟩
  rw [← hWd] at eW
  rw [← hWd]
  -- `X ≤ W`, `2 lδ ≤ W`, `1/D ≤ 2W`
  have hXW : knX r ≤ W := by
    rw [eW]
    have := mul_le_mul hlδ iD (by norm_num) (by linarith)
    have h1 : (1 : ℝ) ≤ lδ * (1 / D) := by linarith
    have := le_mul_of_one_le_right (by linarith : 0 ≤ knX r) h1
    linarith
  have hlδW : 2 * lδ ≤ W := by
    rw [eW]
    have h1 := mul_le_mul_of_nonneg_left iD (show 0 ≤ knX r * lδ by positivity)
    have h2 : lδ ≤ knX r * lδ := le_mul_of_one_le_left (by linarith) hX
    linarith
  have hDW : 1 / D ≤ 2 * W := by
    rw [eW]
    have h1 : 1 * (1 / 2) ≤ knX r * lδ := mul_le_mul hX hlδ (by norm_num) (by linarith)
    have := mul_le_mul_of_nonneg_right h1 (show 0 ≤ 1 / D by positivity)
    linarith
  have hW1 : 1 ≤ W := by linarith
  obtain ⟨U, hUd⟩ : ∃ U, U = 8 * r.A * W := ⟨_, rfl⟩
  rw [← hUd]
  have hWU : 8 * W ≤ U := by
    rw [hUd]
    have := mul_le_mul_of_nonneg_right hA (show 0 ≤ 8 * W by linarith)
    linarith
  have hAU : (r.A : ℝ) ≤ U := by
    rw [hUd]
    have : (r.A : ℝ) ≤ r.A * (8 * W) := le_mul_of_one_le_right (by linarith) (by linarith)
    linarith
  have hU1 : 1 ≤ U := by linarith
  -- each factor at most `U`
  have fQ : (r.Q : ℝ) ≤ U := by linarith
  have fX : knX r ≤ U := by linarith
  have fξ : 1 / r.ξc ≤ U := by
    have := one_div_le_one_div_of_le hD0 hDξ
    linarith
  have fs : 1 / r.s₀ ≤ U := by
    have := one_div_le_one_div_of_le hD0 hDs
    linarith
  have fp : 1 / r.pAP ≤ U := by
    have := one_div_le_one_div_of_le hD0 hDp
    linarith
  have fg : knLg r ≤ U := by linarith
  have fz : r.logZ ≤ U := by
    have e : r.logZ = log (knX r * (1 / D)) + log (r.A : ℝ) + log (1 / r.δ) := by
      unfold RLStarRun.logZ
      rw [e1, Real.log_mul (x := knX r * (1 / D) * r.A) (y := 1 / r.δ) (by positivity)
        (by positivity), Real.log_mul (x := knX r * (1 / D)) (y := r.A) (by positivity)
        (by positivity)]
    rw [← hlδd] at e
    have l1 : log (knX r * (1 / D)) ≤ knX r * (1 / D) := by
      have := Real.log_le_sub_one_of_pos (show 0 < knX r * (1 / D) by positivity); linarith
    have l2 : log (r.A : ℝ) ≤ r.A := by
      have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < r.A by linarith); linarith
    have l3 : knX r * (1 / D) ≤ 2 * W := by
      rw [eW]
      have := mul_le_mul_of_nonneg_right (show 1 ≤ 2 * lδ by linarith)
        (show 0 ≤ knX r * (1 / D) by positivity)
      linarith
    have l4 : 3 * W + r.A ≤ U := by
      rw [hUd]
      have := mul_le_mul_of_nonneg_right hA (show 0 ≤ 4 * W by linarith)
      have : (r.A : ℝ) ≤ r.A * (4 * W) := le_mul_of_one_le_right (by linarith) (by linarith)
      linarith
    linarith
  have hS : tolShape r = r.A * knX r * (1 / r.ξc) * r.nCap := by unfold tolShape knX; ring
  -- the factors' products
  have hιU : r.ιInv ≤ U ^ 5 := by
    rw [shape_ιInv]
    calc (r.Q : ℝ) ^ 2 * r.A * knX r * (1 / r.ξc) ≤ U ^ 2 * U * U * U := by gcongr
      _ = U ^ 5 := by ring
  have hLU : r.nCap ≤ 3 * U ^ 23 := by
    rw [shape_nCap]
    calc (r.Q : ℝ) * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
          + r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2
        ≤ U * U * (U ^ 5) ^ 3 * U ^ 4 * (U ^ 2 + U ^ 2) + U * U * U ^ 2 * U ^ 2 := by
          gcongr
      _ = 2 * U ^ 23 + U ^ 6 := by ring
      _ ≤ 3 * U ^ 23 := by
          have : U ^ 6 ≤ U ^ 23 := pow_le_pow_right₀ hU1 (by norm_num)
          linarith
  have hS16 : 16 ≤ tolShape r := by
    rw [hS]
    have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
    have ip : 1 ≤ 1 / r.pAP := by rw [le_div_iff₀ hp]; linarith
    have h1 : (8 : ℝ) ≤ r.nCap := by
      rw [shape_nCap]
      have a1 : (1 : ℝ) * 2 ≤ r.Q * r.logZ := mul_le_mul hQ hl2 (by norm_num) (by linarith)
      have is2 : (4 : ℝ) ≤ (1 / r.s₀) ^ 2 := by
        calc (4 : ℝ) = 2 * 2 := by norm_num
          _ ≤ (1 / r.s₀) * (1 / r.s₀) := mul_le_mul is is (by norm_num) (by linarith)
          _ = (1 / r.s₀) ^ 2 := by ring
      have a2 : (2 : ℝ) * 4 ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 :=
        mul_le_mul (by linarith) is2 (by norm_num) (by positivity)
      have a3 : (8 : ℝ) * 1 ≤ r.Q * r.logZ * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 :=
        mul_le_mul (by linarith) (one_le_pow₀ ip) (by norm_num) (by positivity)
      have a4 : (0 : ℝ) ≤ r.Q * r.logZ * r.ιInv ^ 3 * (1 / r.s₀) ^ 4
          * ((1 / r.s₀) ^ 2 + r.Q ^ 2) := by positivity
      linarith
    have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
    have b1 : (2 : ℝ) * 1 ≤ r.A * knX r := mul_le_mul hA hX (by norm_num) (by linarith)
    have b2 : (2 : ℝ) * 1 ≤ r.A * knX r * (1 / r.ξc) :=
      mul_le_mul (by linarith) iξ (by norm_num) (by positivity)
    have b3 : (2 : ℝ) * 8 ≤ r.A * knX r * (1 / r.ξc) * r.nCap :=
      mul_le_mul (by linarith) h1 (by norm_num) (by positivity)
    linarith
  refine ⟨hW1, hS16, ?_⟩
  rw [hS]
  calc (r.A : ℝ) * knX r * (1 / r.ξc) * r.nCap ≤ U * U * U * (3 * U ^ 23) := by gcongr
    _ = 3 * U ^ 26 := by ring

end uniform

theorem rlStarClaimUniform_of_rlStarClaim (h : RLStarClaim.{u, v, w}) :
    RLStarClaimUniform.{u, v, w} := by
  obtain ⟨p₀Tol, tp, ts, tpr, tt, tr, hp₀, hpref, hsuff, hprobes, htrie, hreads, hpos, hw⟩ := h
  obtain ⟨C, hC⟩ := isBigO_principal.1 hp₀
  set C₀ := max 1 C with hC₀d
  have hC₀ : 1 ≤ C₀ := le_max_left 1 C
  set c₁ := (log (3 * C₀) + 26 * log 8) / log 2 with hc₁d
  have hlt := Real.log_two_gt_d9
  have hc₁ : 0 ≤ c₁ := by
    have : 0 ≤ log (3 * C₀) := Real.log_nonneg (by linarith)
    have : 0 ≤ log (8 : ℝ) := Real.log_nonneg (by norm_num)
    positivity
  refine ⟨fun r => log (C₀ * tolShape r) / log r.A, tp, ts, tpr, tt, tr, ?_, hpref, hsuff, hprobes,
    htrie, hreads, ?_⟩
  · refine isBigO_on (c₁ + 26) fun r hr => ?_
    obtain ⟨hQ, hA, -⟩ := kn_base hr
    obtain ⟨hW1, hS16, hSU⟩ := tolShape_le hr
    have hWe : (r.L + r.Q : ℝ) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀) = uW r := rfl
    obtain ⟨W, hWd⟩ : ∃ W, W = uW r := ⟨_, rfl⟩
    rw [← hWd] at hW1 hSU hWe
    have hlA : log 2 ≤ log (r.A : ℝ) := Real.log_le_log (by norm_num) hA
    have hlA0 : 0 < log (r.A : ℝ) := by linarith
    have hV : 0 ≤ log W := Real.log_nonneg hW1
    have hCS : 1 ≤ C₀ * tolShape r := one_le_mul_of_one_le_of_one_le hC₀ (by linarith)
    have hk0 : 0 ≤ log (C₀ * tolShape r) := Real.log_nonneg hCS
    -- `log (C₀ S) ≤ log (3 C₀) + 26 (log 8 + log |Σ| + log W)`
    have hlog : log (C₀ * tolShape r) ≤ log (3 * C₀) + 26 * log 8 + 26 * log r.A + 26 * log W := by
      have h1 : C₀ * tolShape r ≤ 3 * C₀ * (8 * r.A * W) ^ 26 := by
        have := mul_le_mul_of_nonneg_left hSU (show 0 ≤ C₀ by linarith)
        linarith
      have h2 := Real.log_le_log (by linarith) h1
      have hR : log (3 * C₀ * (8 * r.A * W) ^ 26)
          = log (3 * C₀) + 26 * (log 8 + log r.A + log W) := by
        rw [Real.log_mul (x := 3 * C₀) (y := (8 * r.A * W) ^ 26) (by positivity) (by positivity),
          Real.log_pow, Real.log_mul (x := 8 * (r.A : ℝ)) (y := W) (by positivity) (by positivity),
          Real.log_mul (x := 8) (y := (r.A : ℝ)) (by norm_num) (by positivity)]
        push_cast; ring
      linarith
    have hdiv : (log (3 * C₀) + 26 * log 8) / log r.A ≤ c₁ := by
      rw [hc₁d]
      refine div_le_div_of_nonneg_left ?_ (by linarith) hlA
      have : 0 ≤ log (3 * C₀) := Real.log_nonneg (by linarith)
      have : 0 ≤ log (8 : ℝ) := Real.log_nonneg (by norm_num)
      positivity
    have hVA : 0 ≤ log W / log r.A := div_nonneg hV hlA0.le
    refine ⟨div_nonneg hk0 hlA0.le, ?_, ?_⟩
    · show 0 ≤ 1 + log ((r.L + r.Q : ℝ) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀)) / log r.A
      rw [hWe]
      linarith
    have e : (log (3 * C₀) + 26 * log 8 + 26 * log r.A + 26 * log W) / log r.A
        = (log (3 * C₀) + 26 * log 8) / log r.A + 26 * (log r.A / log r.A)
          + 26 * (log W / log r.A) := by ring
    rw [div_self hlA0.ne', mul_one] at e
    have h3 : log (C₀ * tolShape r) / log r.A
        ≤ (log (3 * C₀) + 26 * log 8 + 26 * log r.A + 26 * log W) / log r.A :=
      div_le_div_of_nonneg_right hlog hlA0.le
    rw [e] at h3
    show log (C₀ * tolShape r) / log r.A
      ≤ (c₁ + 26) * (1 + log ((r.L + r.Q : ℝ) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀)) / log r.A)
    rw [hWe]
    linarith [mul_nonneg hc₁ hVA]
  -- the run, at `D` uniform
  intro A _ _ _ Q _ Ω _ μ _ x M O L k L₁ hx hQ hA hL hgap hbL hbN hpAPu hk₀ hkL
  have hr : ({ x with L, k } : RLStarRun) ∈ rlStarRunRegion := hx
  have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hx.2.1
  have hAc : (Fintype.card A : ℝ) = x.A := by exact_mod_cast hA
  obtain ⟨-, hS16, -⟩ := tolShape_le hr
  -- `D`'s first `k` letters weigh at most `p₀Tol`
  have hp₀k : ((x.A : ℝ) ^ k)⁻¹ ≤ p₀Tol { x with L, k } := by
    have hpos' := hpos { x with L, k }
    have hc : ‖1 / p₀Tol { x with L, k }‖ ≤ C * ‖tolShape { x with L, k }‖ := hC _ hr
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (one_div_pos.2 hpos'),
      abs_of_nonneg (show (0 : ℝ) ≤ tolShape { x with L, k } by linarith)] at hc
    have hc' : 1 / p₀Tol { x with L, k } ≤ C₀ * tolShape { x with L, k } :=
      hc.trans (mul_le_mul_of_nonneg_right (le_max_right 1 C) (by linarith))
    have hlA : 0 < log (x.A : ℝ) := Real.log_pos (by linarith)
    have hpow : C₀ * tolShape { x with L, k } ≤ (x.A : ℝ) ^ k := by
      have hk' : log (C₀ * tolShape { x with L, k }) / log (x.A : ℝ) ≤ k := hk₀
      have h9 := (div_le_iff₀ hlA).1 hk'
      rw [← Real.log_le_log_iff (by positivity) (pow_pos (by linarith) k), Real.log_pow]
      linarith
    have h1 : 1 / p₀Tol { x with L, k } ≤ (x.A : ℝ) ^ k := hc'.trans hpow
    rw [one_div] at h1
    exact (inv_le_comm₀ (pow_pos (by linarith : (0 : ℝ) < x.A) k) hpos').2 h1
  set m := L + Fintype.card Q + 3 + Fintype.card Q + 3 with hmd
  have hpL : prefixLaw (uniformStrings (α := A) m) L = uniformStrings L :=
    prefixLaw_uniformStrings (by omega)
  have hpS : prefixLaw (uniformStrings (α := A) m) m = uniformStrings m :=
    prefixLaw_uniformStrings le_rfl
  obtain ⟨settings, hwk, hcnt, hprob⟩ := hw x M O L k (uniformStrings m) hx hQ hA hL hgap
    (by rw [hpL]; exact hbL) (by rw [hpL]; exact hbN) (uniformStrings_long m) hkL
    (fun u => (uniformStrings_prefix_le (by omega) u).trans (by rw [hAc]; exact hp₀k))
    (by rw [hpS]; exact hpAPu)
  rw [hpL, hpS] at hcnt
  rw [hpL, hpS] at hprob
  exact ⟨settings, hwk, hcnt, hprob⟩

section simple

variable {r : RLStarRun}

lemma simple_pref (r : RLStarRun) :
    r.nPref = r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.logZ / (r.ξc ^ 2 * r.s₀ ^ 4) := by
  unfold RLStarRun.nPref RLStarRun.ιInv; ring

lemma simple_trie (r : RLStarRun) :
    r.nHeavy = r.Q ^ 7 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 6)
      + r.Q ^ 9 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 4)
      + r.Q * r.A * (r.L + r.Q) * r.logZ ^ 2 / (r.ξc * r.pAP ^ 2 * r.s₀ ^ 2) := by
  unfold RLStarRun.nHeavy RLStarRun.nCap RLStarRun.ιInv; ring

lemma simple_probe (r : RLStarRun) :
    r.nProbe = r.Q ^ 8 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 6)
      + r.Q ^ 10 * r.A ^ 4 * (r.L + r.Q) ^ 4 * r.logZ ^ 2 / (r.ξc ^ 4 * r.s₀ ^ 4)
      + r.Q ^ 2 * r.A * (r.L + r.Q) * r.logZ ^ 2 / (r.ξc * r.pAP ^ 2 * r.s₀ ^ 2) := by
  unfold RLStarRun.nProbe; rw [simple_trie]; ring

lemma simple_reads (hr : r ∈ rlStarRunRegion) :
    r.nReads = r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.logZ ^ 2 / (r.ξc ^ 2 * r.pAP * r.s₀ ^ 6)
      + (r.k + r.logZ) * (r.Q ^ 8 * r.A ^ 5 * (r.L + r.Q) ^ 4 * r.logZ ^ 3 / (r.ξc ^ 4 * r.s₀ ^ 8)
        + r.Q ^ 10 * r.A ^ 5 * (r.L + r.Q) ^ 4 * r.logZ ^ 3 / (r.ξc ^ 4 * r.s₀ ^ 6)
        + r.Q ^ 2 * r.A ^ 2 * (r.L + r.Q) * r.logZ ^ 3 / (r.ξc * r.pAP ^ 2 * r.s₀ ^ 4)) := by
  obtain ⟨hQ, -⟩ := kn_base hr
  have hQ0 : (r.Q : ℝ) ≠ 0 := by positivity
  have e : r.nPref * r.nSuff / r.Q
      = r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.logZ ^ 2 / (r.ξc ^ 2 * r.pAP * r.s₀ ^ 6) := by
    rw [simple_pref]
    unfold RLStarRun.nSuff
    field_simp
  unfold RLStarRun.nReads
  rw [e, simple_trie]
  ring

theorem rlStarClaimUniformSimple_of_claim (h : RLStarClaimUniform.{u, v, w}) :
    RLStarClaimUniformSimple.{u, v, w} := by
  obtain ⟨k₀, tp, ts, tpr, tt, tr, hk, hp, hs, hpr, ht, hr, hw⟩ := h
  refine ⟨k₀, tp, ts, tpr, tt, tr, hk, hp.trans ?_, hs, hpr.trans ?_, ht.trans ?_, hr.trans ?_,
    hw⟩
  · refine isBigO_on 1 fun r hr => ?_
    obtain ⟨-, -, hP0, -⟩ := shape_pos hr
    rw [← simple_pref r]
    exact ⟨hP0, hP0, by linarith⟩
  · refine isBigO_on 1 fun r hr => ?_
    obtain ⟨-, -, -, -, -, hPr0, -⟩ := shape_pos hr
    rw [← simple_probe r]
    exact ⟨hPr0, hPr0, by linarith⟩
  · refine isBigO_on 1 fun r hr => ?_
    obtain ⟨-, -, -, -, hH0, -⟩ := shape_pos hr
    rw [← simple_trie r]
    exact ⟨hH0, hH0, by linarith⟩
  · refine isBigO_on 1 fun r hr => ?_
    obtain ⟨-, -, -, -, -, -, hR0⟩ := shape_pos hr
    rw [← simple_reads hr]
    exact ⟨hR0, hR0, by linarith⟩

end simple

end OrthoDFA

end
