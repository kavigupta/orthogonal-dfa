import RLStar.Proofs.ClaimRLStarArith
import RLStar.Proofs.DFABridge

/-!
# The uniform claims from the general one

`RLStarClaimUniform` is `RLStarClaim` at `D` uniform over the strings of length `L + 2|Q| + 6`:
uniform strings hold `|Σ|^-k` on any first `k` letters, and `1/p₀Tol` is at most
`((L + |Q|)⁶ C₀ G)⁵` for a constant `C₀`, `G` free of `L`, so
`k ≥ k₀ = 5 log ((L + |Q|)⁶ C₀ G)/log |Σ|` gives `((L + |Q|)⁶ C₀ G)⁵ ≤ |Σ|^k`.
`RLStarClaimUniformSimple` follows from
`RLStarClaimUniform`, the prefix and suffix totals' shapes bounded by their products, and the
search's reads by the harvest's.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

/-- A uniform instance as a general one, at `β = 1/|Σ|`. -/
def toDist (x : RatioInputs) : RLStarDistInputs := { toRatioInputs := x, β := 1 / x.A }

def toDistRun (r : RatioRun) : RLStarDistRun :=
  { toRLStarDistInputs := toDist r.toRatioInputs, L := r.L }

lemma toDist_mem {x : RatioInputs} (hx : x ∈ ratioRegion) : toDist x ∈ rlStarDistRegion := by
  have hA : (2 : ℝ) ≤ x.A := by exact_mod_cast hx.2.1
  refine ⟨hx, ?_, ?_⟩
  · show 0 < 1 / (x.A : ℝ); positivity
  · show 1 / (x.A : ℝ) * x.A ≤ 1
    rw [one_div_mul_cancel (by positivity)]

/-- `1/p₀Tol`'s shape at `β = 1/|Σ|` is at most `2 (L + |Q|)⁵ loopG`. -/
def loopG (x : RatioInputs) : ℝ :=
  x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4
    * log (x.Q * x.R * x.A * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀)) ^ 3
    * (1 / x.ξ) ^ 4 * (1 / x.pAP) ^ 3 * (1 / x.s₀) ^ 6

lemma shape_toDist_le {x : RatioInputs} (hx : x ∈ ratioRegion) (L : ℕ) :
    0 ≤ ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (1 / (toDist x).β) ^ 4 * x.R ^ 4
        * log (((L : ℝ) + x.Q) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀))
        * log (x.Q * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
        / (x.ξ ^ 4 * x.pAP ^ 3 * x.s₀ ^ 6)
      ∧ ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (1 / (toDist x).β) ^ 4 * x.R ^ 4
        * log (((L : ℝ) + x.Q) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀))
        * log (x.Q * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
        / (x.ξ ^ 4 * x.pAP ^ 3 * x.s₀ ^ 6)
      ≤ 2 * (((L : ℝ) + x.Q) ^ 6 * loopG x) := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hA
  have hLQ : (1 : ℝ) ≤ (L : ℝ) + x.Q := by have : (0 : ℝ) ≤ L := Nat.cast_nonneg L; linarith
  set Y : ℝ := x.Q * x.R * x.A * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀) with hYd
  have hY : 3 ≤ Y := by
    have h1 : 2 ≤ (x.Q : ℝ) * x.R * x.A := by
      have : 1 ≤ (x.Q : ℝ) * x.R := one_le_mul_of_one_le_of_one_le hQ' hR
      nlinarith
    have h4 : 2 ≤ 1 / x.s₀ := (le_div_iff₀ hs).2 (by linarith)
    have h5 : 1 ≤ (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        ((le_div_iff₀ hδ).2 (by linarith)) ((le_div_iff₀ hξ).2 (by linarith)))
        ((le_div_iff₀ hp).2 (by linarith))
    have e : Y = (x.Q * x.R * x.A) * ((1 / x.δ) * (1 / x.ξ) * (1 / x.pAP)) * (1 / x.s₀) := by
      rw [hYd]; ring
    rw [e]
    nlinarith [mul_le_mul h1 h5 (by norm_num) (by linarith)]
  have hlY : 1 ≤ log Y := one_le_log_of_three_le hY
  have hlLY : log (((L : ℝ) + x.Q) * Y) ≤ 2 * ((L : ℝ) + x.Q) * log Y := by
    rw [Real.log_mul (by linarith) (by linarith)]
    have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < (L : ℝ) + x.Q by linarith)
    nlinarith
  have hQ0 : (0 : ℝ) < x.Q := by linarith
  have hLY0 : 0 < ((L : ℝ) + x.Q) * Y := mul_pos (by linarith) (by linarith)
  have hlLQ : log (((L : ℝ) + x.Q) * Y / x.Q) ≤ 2 * ((L : ℝ) + x.Q) * log Y :=
    (Real.log_le_log (div_pos hLY0 hQ0) (div_le_self hLY0.le hQ')).trans hlLY
  have hlLY0 : 0 ≤ log (((L : ℝ) + x.Q) * Y / x.Q) := Real.log_nonneg (by
    rw [le_div_iff₀ hQ0]
    nlinarith [mul_nonneg (show (0 : ℝ) ≤ (L : ℝ) + x.Q by linarith)
      (show (0 : ℝ) ≤ Y - 3 by linarith), Nat.cast_nonneg (α := ℝ) L])
  have e1 : ((L : ℝ) + x.Q) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)
      = ((L : ℝ) + x.Q) * Y / x.Q := by
    show ((L : ℝ) + x.Q) * x.R / (1 / (x.A : ℝ) * x.δ * x.ξ * x.pAP * x.s₀) = _
    rw [hYd]; field_simp
  have e2 : (x.Q : ℝ) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀) = Y := by
    show (x.Q : ℝ) * x.R / (1 / (x.A : ℝ) * x.δ * x.ξ * x.pAP * x.s₀) = _
    rw [hYd]; field_simp
  have e4 : 1 / (toDist x).β = x.A := by
    show 1 / (1 / (x.A : ℝ)) = _
    rw [one_div_one_div]
  set K : ℝ := (1 / x.ξ) ^ 4 * (1 / x.pAP) ^ 3 * (1 / x.s₀) ^ 6 with hKd
  have e3 : ∀ z : ℝ, z / (x.ξ ^ 4 * x.pAP ^ 3 * x.s₀ ^ 6) = z * K := by
    intro z
    rw [hKd]; field_simp
  have hG : loopG x = x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4 * log Y ^ 3 * K := by
    rw [loopG, hKd]; ring
  have hK0 : 0 ≤ K := by positivity
  rw [e1, e2, e3, e4, hG]
  refine ⟨by positivity, ?_⟩
  have h5 : ((L : ℝ) + x.Q) ^ 5 ≤ ((L : ℝ) + x.Q) ^ 6 := pow_le_pow_right₀ hLQ (by norm_num)
  have hb : 0 ≤ (x.Q : ℝ) ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4 * log Y ^ 3 * K := by
    have : 0 ≤ log Y := by linarith
    positivity
  calc ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (x.A : ℝ) ^ 4 * x.R ^ 4
        * log (((L : ℝ) + x.Q) * Y / x.Q) * log Y ^ 2 * K
      ≤ ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4
        * log (((L : ℝ) + x.Q) * Y / x.Q) * log Y ^ 2 * K := by gcongr; linarith
    _ ≤ ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4
        * (2 * ((L : ℝ) + x.Q) * log Y) * log Y ^ 2 * K := by gcongr
    _ = 2 * (((L : ℝ) + x.Q) ^ 5 * (x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4 * log Y ^ 3 * K)) := by
      ring
    _ ≤ 2 * (((L : ℝ) + x.Q) ^ 6 * (x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4 * log Y ^ 3 * K)) := by
      gcongr

lemma loopG_ge_one {x : RatioInputs} (hx : x ∈ ratioRegion) : 1 ≤ loopG x := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  exact (loopL0_le (iδ := 1 / x.δ) (iξ := 1 / x.ξ) (ip := 1 / x.pAP) (is := 1 / x.s₀) (C₀ := 1)
    (by exact_mod_cast hQ) (by exact_mod_cast hA) hR ((le_div_iff₀ hδ).2 (by linarith))
    ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
    ((le_div_iff₀ hs).2 (by linarith)) le_rfl).1

lemma uniform_rewrite {X R A δ ξ pAP s₀ : ℝ} (hA : 0 < A) (hδ : 0 < δ) (hξ : 0 < ξ)
    (hp : 0 < pAP) (hs : 0 < s₀) :
    X * R / (1 / A * δ * ξ * pAP * s₀) = X * A * R / (δ * ξ * pAP * s₀)
      ∧ X * R / (1 / A * δ * ξ * s₀) = X * A * R / (δ * ξ * s₀) :=
  ⟨by field_simp, by field_simp⟩

/-- A claims' instance as one of the rates', at `R = 1`. -/
def ratioOne (x : RLStarInputs) : RatioInputs := ⟨x.Q, x.A, x.δ, 1, x.ξ, x.pAP, x.s₀⟩

def ratioOneRun (r : RLStarRun) : RatioRun :=
  { toRatioInputs := ratioOne r.toRLStarInputs, L := r.L }

lemma ratioOne_mem {x : RLStarInputs} (hx : x ∈ rlStarRegion) : ratioOne x ∈ ratioRegion := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := hx
  exact ⟨h1, h2, h3, h4, le_rfl, h5, h6, h7, h8, h9, h10⟩

lemma p₀Shape_one {r : RLStarRun} (hr : r ∈ rlStarRunRegion) :
    p₀Shape (toDistRun (ratioOneRun r)) =
      (r.L + r.Q) ^ 4 * log ((r.L + r.Q) * r.A / (r.δ * r.ξ * r.pAP * r.s₀))
          * log (r.Q * r.A / (r.δ * r.ξ * r.pAP * r.s₀)) * r.Q ^ 3 * r.A / r.ξ
          * (r.Q ^ 2 * r.A ^ 2 / r.ξ ^ 3
              * (1 / r.s₀ ^ 6 + r.Q ^ 2 * r.A / r.s₀ ^ 5 + r.Q ^ 4 * r.A / r.s₀ ^ 4)
            + 1 / (r.pAP ^ 2 * r.s₀ ^ 2)) := by
  obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
  have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
  simp only [p₀Shape, toDistRun, ratioOneRun, ratioOne, toDist, one_div_one_div]
  rw [(uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs).1,
    (uniform_rewrite (X := (r.Q : ℝ)) (R := 1) hA' hδ hξ hp hs).1]
  simp only [mul_one, one_pow]
  ring

theorem rlStarClaimUniform_of_rlStarClaim (h : RLStarClaim.{u, v, w}) :
    RLStarClaimUniform.{u, v, w} := by
  obtain ⟨p₀Tol, tp, ts, tpr, tr, hp₀, hpref, hsuff, hprobes, hreads, hpos, hw⟩ := h
  obtain ⟨C, hC⟩ := isBigO_principal.1 hp₀
  set C₀ := 16 * max 1 C with hC₀d
  have hC₀ : 1 ≤ C₀ := by rw [hC₀d]; linarith [le_max_left 1 C]
  refine ⟨fun r => 5 * log (((r.L : ℝ) + r.Q) ^ 6 * (C₀ * loopG (ratioOne r.toRLStarInputs)))
      / log r.A, tp, ts, tpr, tr, ?_, hpref, hsuff, hprobes, hreads, ?_⟩
  · refine isBigO_on (30 + 10 * (275 + 6 * log C₀)) fun r hr => ?_
    obtain ⟨hQ, hA, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := hr
    have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
    have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
    obtain ⟨hG, hle⟩ := loopL0_le (R := 1) (iδ := 1 / r.δ) (iξ := 1 / r.ξ) (ip := 1 / r.pAP)
      (is := 1 / r.s₀) hQ' hA' le_rfl ((le_div_iff₀ hδ).2 (by linarith))
      ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
      ((le_div_iff₀ hs).2 (by linarith)) hC₀
    set G := loopG (ratioOne r.toRLStarInputs) with hGd
    change 1 ≤ G at hG
    set ℓ := log (r.A : ℝ) with hℓd
    set c₁ := 275 + 6 * log C₀ with hc₁d
    set X := log (C₀ * G) with hXd
    set D := r.ξ * r.pAP * r.s₀ with hDd
    set lδ := log (1 / r.δ) with hlδd
    set V := log (((r.L : ℝ) + r.Q) * lδ / D) with hVd
    change 1 + 4 * (X + 6 * log ((r.Q : ℝ) + 4) + 24) / ℓ
        ≤ c₁ * (1 + log ((r.Q : ℝ) * 1 * (2 * lδ) * (1 / r.ξ) * (1 / r.pAP) * (1 / r.s₀)) / ℓ)
      at hle
    have hlA : 0 < ℓ := Real.log_pos (by linarith)
    have hLQ : (1 : ℝ) ≤ r.L + r.Q := by have : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _; linarith
    have hD0 : 0 < D := by positivity
    have hlδ : 1 / 2 ≤ lδ := by
      have := Real.log_two_gt_d9
      have := Real.log_le_log (by norm_num) ((le_div_iff₀ hδ).2 (by linarith) : (2 : ℝ) ≤ 1 / r.δ)
      linarith
    have hD2 : lδ / D ≥ 1 := by
      rw [ge_iff_le, le_div_iff₀ hD0]
      have : D ≤ 1 / 2 := by
        have : r.ξ * r.pAP ≤ 1 := by nlinarith
        rw [hDd]; nlinarith
      linarith
    have e : (r.Q : ℝ) * 1 * (2 * lδ) * (1 / r.ξ) * (1 / r.pAP) * (1 / r.s₀)
        = 2 * (r.Q * lδ / D) := by
      rw [hDd]; field_simp
    rw [e] at hle
    have hLV : log ((r.L : ℝ) + r.Q) ≤ V := by
      refine Real.log_le_log (by linarith) ?_
      rw [mul_div_assoc]
      exact le_mul_of_one_le_right (by linarith) hD2
    have hQl : 0 < (r.Q : ℝ) * lδ / D := by
      have : 0 < lδ := by linarith
      positivity
    have hWV : log (2 * ((r.Q : ℝ) * lδ / D)) ≤ ℓ + V := by
      rw [Real.log_mul (by norm_num) hQl.ne']
      have h2 : log 2 ≤ ℓ := log_two_le_log hA'
      have h3 : log ((r.Q : ℝ) * lδ / D) ≤ V := by
        refine Real.log_le_log hQl ?_
        have : (0 : ℝ) ≤ lδ / D := div_nonneg (by linarith) hD0.le
        rw [mul_div_assoc, mul_div_assoc]
        exact mul_le_mul_of_nonneg_right
          (by have : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _; linarith) this
      linarith
    have hL0 : 0 ≤ log ((r.L : ℝ) + r.Q) := Real.log_nonneg hLQ
    have hX0 : 0 ≤ X := Real.log_nonneg (one_le_mul_of_one_le_of_one_le hC₀ hG)
    have hY0 : 0 ≤ log ((r.Q : ℝ) + 4) := Real.log_nonneg (by linarith)
    have hc₁ : 0 ≤ c₁ := by have := Real.log_nonneg hC₀; rw [hc₁d]; linarith
    have h1 := div_le_div_of_nonneg_right hLV hlA.le
    have hWV' : log (2 * ((r.Q : ℝ) * lδ / D)) / ℓ ≤ 1 + V / ℓ := by
      rw [div_le_iff₀ hlA, add_mul, one_mul, div_mul_cancel₀ _ hlA.ne']; exact hWV
    have h2 : c₁ * (log (2 * ((r.Q : ℝ) * lδ / D)) / ℓ) ≤ c₁ + c₁ * (V / ℓ) :=
      (mul_le_mul_of_nonneg_left hWV' hc₁).trans_eq (by ring)
    have hV : 0 ≤ V / ℓ := div_nonneg (hL0.trans hLV) hlA.le
    have hcV : 0 ≤ c₁ * (V / ℓ) := mul_nonneg hc₁ hV
    have hXl : 0 ≤ X / ℓ := div_nonneg hX0 hlA.le
    have hYl : 0 ≤ log ((r.Q : ℝ) + 4) / ℓ := div_nonneg hY0 hlA.le
    have hl : 0 ≤ 1 / ℓ := div_nonneg zero_le_one hlA.le
    have e1 : 4 * (X + 6 * log ((r.Q : ℝ) + 4) + 24) / ℓ
        = 4 * (X / ℓ) + 24 * (log ((r.Q : ℝ) + 4) / ℓ) + 96 * (1 / ℓ) := by ring
    have e2 : c₁ * (1 + log (2 * ((r.Q : ℝ) * lδ / D)) / ℓ)
        = c₁ + c₁ * (log (2 * ((r.Q : ℝ) * lδ / D)) / ℓ) := by
      ring
    rw [e1, e2] at hle
    have ef : 5 * log (((r.L : ℝ) + r.Q) ^ 6 * (C₀ * G)) / ℓ
        = 30 * (log ((r.L : ℝ) + r.Q) / ℓ) + 5 * (X / ℓ) := by
      rw [Real.log_mul (by positivity) (by linarith [one_le_mul_of_one_le_of_one_le hC₀ hG]),
        Real.log_pow]
      push_cast; ring
    have eg : (30 + 10 * c₁) * (1 + V / ℓ)
        = 30 + 10 * c₁ + 30 * (V / ℓ) + 10 * (c₁ * (V / ℓ)) := by
      ring
    change 0 ≤ 5 * log (((r.L : ℝ) + r.Q) ^ 6 * (C₀ * G)) / ℓ ∧ 0 ≤ 1 + V / ℓ
      ∧ 5 * log (((r.L : ℝ) + r.Q) ^ 6 * (C₀ * G)) / ℓ ≤ (30 + 10 * c₁) * (1 + V / ℓ)
    rw [ef, eg]
    refine ⟨by positivity, by linarith, by linarith⟩
  -- the run, at `D` uniform
  intro A _ _ _ Q _ Ω _ μ _ x M O L k L₁ hx hQ hA hL hη hpAPu hk₀ hkL
  have hr : ({ x with L } : RLStarRun) ∈ rlStarRunRegion := hx
  have hxr := ratioOne_mem hx
  have hrd : toDistRun (ratioOneRun { x with L }) ∈ rlStarDistRunRegion := toDist_mem hxr
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hx.1
  have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hx.2.1
  have hAc : (Fintype.card A : ℝ) = x.A := by exact_mod_cast hA
  -- `D`'s first `k` letters weigh at most `p₀Tol`
  have hp₀k : ((x.A : ℝ) ^ k)⁻¹ ≤ p₀Tol { x with L } := by
    have hpos' := hpos { x with L }
    have hc : ‖1 / p₀Tol { x with L }‖
        ≤ C * ‖p₀Shape (toDistRun (ratioOneRun { x with L })) ^ 5‖ := by
      rw [p₀Shape_one hr]; exact hC _ hr
    have hG1 : 1 ≤ loopG (ratioOne x) := loopG_ge_one hxr
    obtain ⟨hs0, hsle⟩ := p₀Shape_le hrd
    obtain ⟨-, hsh⟩ := shape_toDist_le hxr L
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (one_div_pos.2 hpos'),
      abs_of_nonneg (pow_nonneg hs0 5)] at hc
    change _ ≤ 2 * (((L : ℝ) + x.Q) ^ 6 * loopG (ratioOne x)) at hsh
    have h8 : p₀Shape (toDistRun (ratioOneRun { x with L }))
        ≤ 16 * (((L : ℝ) + x.Q) ^ 6 * loopG (ratioOne x)) :=
      (hsle.trans (mul_le_mul_of_nonneg_left hsh (by norm_num))).trans_eq (by ring)
    obtain ⟨y, hyd⟩ : ∃ y, y = ((L : ℝ) + x.Q) ^ 6 * loopG (ratioOne x) := ⟨_, rfl⟩
    rw [← hyd] at h8
    have hy : 0 ≤ y := hyd ▸ mul_nonneg (by positivity) (by linarith)
    have hm1 : 1 ≤ max 1 C := le_max_left 1 C
    have hc' : 1 / p₀Tol { x with L } ≤ (((L : ℝ) + x.Q) ^ 6 * (C₀ * loopG (ratioOne x))) ^ 5 :=
      calc 1 / p₀Tol { x with L }
          ≤ max 1 C * (16 * y) ^ 5 :=
            hc.trans (mul_le_mul (le_max_right 1 C) (pow_le_pow_left₀ hs0 h8 5)
              (pow_nonneg hs0 5) (by linarith))
        _ ≤ max 1 C ^ 5 * (16 * y) ^ 5 :=
            mul_le_mul_of_nonneg_right (le_self_pow₀ hm1 (by norm_num)) (by positivity)
        _ = _ := by rw [hC₀d, hyd]; ring
    have hpow : (((L : ℝ) + x.Q) ^ 6 * (C₀ * loopG (ratioOne x))) ^ 5 ≤ (x.A : ℝ) ^ k := by
      have hlA : 0 < log (x.A : ℝ) := Real.log_pos (by linarith)
      have hZ : 0 < ((L : ℝ) + x.Q) ^ 6 * (C₀ * loopG (ratioOne x)) :=
        mul_pos (pow_pos (by have : (0 : ℝ) ≤ L := Nat.cast_nonneg _; linarith) 6)
          (by linarith [one_le_mul_of_one_le_of_one_le hC₀ hG1])
      change 5 * log (((L : ℝ) + x.Q) ^ 6 * (C₀ * loopG (ratioOne x))) / log (x.A : ℝ) ≤ k
        at hk₀
      have h9 := (div_le_iff₀ hlA).1 hk₀
      rw [← Real.log_le_log_iff (pow_pos hZ 5) (pow_pos (by linarith) k), Real.log_pow,
        Real.log_pow]
      push_cast
      linarith
    have h1 : 1 / p₀Tol { x with L } ≤ (x.A : ℝ) ^ k := hc'.trans hpow
    rw [one_div] at h1
    exact (inv_le_comm₀ (pow_pos (by linarith : (0 : ℝ) < x.A) k) hpos').2 h1
  set m := L + Fintype.card Q + 3 + Fintype.card Q + 3 with hmd
  have hpL : prefixLaw (uniformStrings (α := A) m) L = uniformStrings L :=
    prefixLaw_uniformStrings (by omega)
  have hpS : prefixLaw (uniformStrings (α := A) m) m = uniformStrings m :=
    prefixLaw_uniformStrings le_rfl
  obtain ⟨settings, hwk, hcnt, hprob⟩ := hw x M O L k (uniformStrings m) hx hQ hA hL hη
    (uniformStrings_long m) hkL
    (fun u => (uniformStrings_prefix_le (by omega) u).trans (by rw [hAc]; exact hp₀k))
    (by rw [hpS]; exact hpAPu)
  rw [hpL, hpS] at hcnt
  rw [hpL, hpS] at hprob
  exact ⟨settings, hwk, hcnt, hprob⟩

/-- The reads' factors other than the prefix and suffix totals, at uniform strings. -/
lemma uniform_factors {r : RatioRun} (hr : r ∈ ratioRunRegion) :
    0 < (r.Q : ℝ)
      ∧ 0 ≤ log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / r.s₀ ^ 2
      ∧ 0 ≤ r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
      ∧ 0 ≤ log (r.L + r.Q + 1)
      ∧ 0 ≤ r.R * r.A * (r.L + r.Q) / r.ξ
      ∧ 0 ≤ log (r.Q / r.δ) / r.ξ ^ 2 := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hN : 1 ≤ ((r.L : ℝ) + r.Q) * r.A * r.R :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)) hR
  have d1 : r.δ * r.ξ * r.s₀ ≤ 1 :=
    mul_le_one₀ (mul_le_one₀ (by linarith) hξ.le hξ1) hs.le (by linarith)
  have l1 : 0 ≤ log (((r.L : ℝ) + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) :=
    Real.log_nonneg ((one_le_div (mul_pos (mul_pos hδ hξ) hs)).2 (d1.trans hN))
  have hQA : (1 : ℝ) ≤ r.Q * r.A := one_le_mul_of_one_le_of_one_le hQ' (by linarith)
  have l3 : 0 ≤ log ((r.Q : ℝ) * r.A / r.δ) :=
    Real.log_nonneg ((one_le_div hδ).2 (by linarith))
  have l4 : 0 ≤ log ((r.Q : ℝ) / r.δ) := Real.log_nonneg ((one_le_div hδ).2 (by linarith))
  refine ⟨by linarith, div_nonneg l1 (sq_nonneg _),
    mul_nonneg (div_nonneg (mul_nonneg (by positivity) (by linarith)) hξ.le) l3,
    Real.log_nonneg (by linarith),
    div_nonneg (mul_nonneg (mul_nonneg (by linarith) (by positivity)) (by positivity)) hξ.le,
    div_nonneg l4 (sq_nonneg _)⟩

/-- The reads' shape is monotone in its prefix and suffix totals. -/
lemma reads_dom {Q P P' S S' F Pr lg ρ Hd : ℝ} (hQ : 0 < Q) (hP : 0 ≤ P) (hPP : P ≤ 4 * P')
    (hS : 0 ≤ S) (hSS : S ≤ 2 * S') (hF : 0 ≤ F) (hPr : 0 ≤ Pr) (hlg : 0 ≤ lg)
    (hρ : 0 ≤ ρ) (hHd : 0 ≤ Hd) :
    0 ≤ P * S / Q + Pr * F * Q * lg + Q * P * F * lg * ρ + Hd * F
      ∧ 0 ≤ P' * S' / Q + Pr * F * Q * lg + Q * P' * F * lg * ρ + Hd * F
      ∧ P * S / Q + Pr * F * Q * lg + Q * P * F * lg * ρ + Hd * F
        ≤ 8 * (P' * S' / Q + Pr * F * Q * lg + Q * P' * F * lg * ρ
          + Hd * F) := by
  have hP' : 0 ≤ P' := by linarith
  have hS' : 0 ≤ S' := by linarith
  set G := F * lg * ρ with hG
  have hG0 : 0 ≤ G := mul_nonneg (mul_nonneg hF hlg) hρ
  have e : ∀ P, Q * P * F * lg * ρ = Q * P * G := fun P => by
    rw [hG]; ring
  rw [e, e]
  have hT2 : 0 ≤ Pr * F * Q * lg := mul_nonneg (mul_nonneg (mul_nonneg hPr hF) hQ.le) hlg
  have hT4 : 0 ≤ Hd * F := mul_nonneg hHd hF
  have hPS : P * S ≤ 8 * (P' * S') := by
    have := mul_le_mul hPP hSS hS (by linarith)
    linarith
  have hPS' : P * S / Q ≤ 8 * (P' * S' / Q) := by
    rw [← mul_div_assoc]
    exact div_le_div_of_nonneg_right hPS hQ.le
  have hQPG : Q * P * G ≤ 4 * (Q * P' * G) := by
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hPP hQ.le) hG0
    linarith
  have n1 : 0 ≤ P * S / Q := div_nonneg (mul_nonneg hP hS) hQ.le
  have n2 : 0 ≤ P' * S' / Q := div_nonneg (mul_nonneg hP' hS') hQ.le
  have n3 : 0 ≤ Q * P * G := mul_nonneg (mul_nonneg hQ.le hP) hG0
  have n4 : 0 ≤ Q * P' * G := mul_nonneg (mul_nonneg hQ.le hP') hG0
  refine ⟨by linarith, by linarith, by linarith⟩

/-- The reads' search and start terms within the harvest's: from `Pr ≤ P ρ`, `Hd ≤ Q P ρ` and
`lg ≥ 1/2`. -/
lemma reads_two_term {Q P S F Pr lg ρ Hd : ℝ} (hQ : 0 < Q) (hP : 0 ≤ P) (hS : 0 ≤ S) (hF : 0 ≤ F)
    (hPr0 : 0 ≤ Pr) (hlg : 1 / 2 ≤ lg) (hρ : 0 ≤ ρ) (hHd0 : 0 ≤ Hd) (hPr : Pr ≤ P * ρ)
    (hHd : Hd ≤ Q * P * ρ) :
    0 ≤ P * S / Q + Pr * F * Q * lg + Q * P * F * lg * ρ + Hd * F
      ∧ 0 ≤ P * (S / Q + Q * F * lg * ρ)
      ∧ P * S / Q + Pr * F * Q * lg + Q * P * F * lg * ρ + Hd * F
        ≤ 4 * (P * (S / Q + Q * F * lg * ρ)) := by
  have hlg0 : 0 ≤ lg := by linarith
  have hFQlg : 0 ≤ F * Q * lg := mul_nonneg (mul_nonneg hF hQ.le) hlg0
  have h1 : Pr * F * Q * lg ≤ Q * P * F * lg * ρ :=
    calc Pr * F * Q * lg = Pr * (F * Q * lg) := by ring
      _ ≤ P * ρ * (F * Q * lg) := mul_le_mul_of_nonneg_right hPr hFQlg
      _ = Q * P * F * lg * ρ := by ring
  have hm : 0 ≤ Q * P * ρ * F := mul_nonneg (mul_nonneg (mul_nonneg hQ.le hP) hρ) hF
  have h2 : Hd * F ≤ 2 * (Q * P * F * lg * ρ) :=
    calc Hd * F ≤ Q * P * ρ * F := mul_le_mul_of_nonneg_right hHd hF
      _ ≤ Q * P * ρ * F * (2 * lg) := le_mul_of_one_le_right hm (by linarith)
      _ = 2 * (Q * P * F * lg * ρ) := by ring
  have n1 : 0 ≤ P * S / Q := div_nonneg (mul_nonneg hP hS) hQ.le
  have n2 : 0 ≤ Q * P * F * lg * ρ :=
    mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg hQ.le hP) hF) hlg0) hρ
  have n3 : 0 ≤ Pr * F * Q * lg := mul_nonneg (mul_nonneg (mul_nonneg hPr0 hF) hQ.le) hlg0
  have n4 : 0 ≤ Hd * F := mul_nonneg hHd0 hF
  have e : P * (S / Q + Q * F * lg * ρ) = P * S / Q + Q * P * F * lg * ρ := by ring
  rw [e]
  exact ⟨by linarith, by linarith, by linarith⟩

/-- At uniform strings, `N_probe ≤ N_pref / ρ`, `n_held ≤ |Q| N_pref / ρ` and
`log(L + |Q| + 1) ≥ 1/2`. -/
lemma uniform_search_start_le {r : RatioRun} (hr : r ∈ ratioRunRegion) :
    r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
        ≤ r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2
            * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 4)
          * (r.R * r.A * (r.L + r.Q) / r.ξ)
      ∧ log (r.Q / r.δ) / r.ξ ^ 2
        ≤ r.Q * (r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2
            * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / (r.ξ ^ 2 * r.s₀ ^ 4))
          * (r.R * r.A * (r.L + r.Q) / r.ξ)
      ∧ 1 / 2 ≤ log (r.L + r.Q + 1) := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hLQ : (1 : ℝ) ≤ r.L + r.Q := by linarith
  have hAQ : (1 : ℝ) ≤ r.A := by linarith
  have hR0 : 0 < r.R := by linarith
  have hiξ : 1 ≤ 1 / r.ξ := (le_div_iff₀ hξ).2 (by linarith)
  have his : 1 ≤ 1 / r.s₀ := (le_div_iff₀ hs).2 (by linarith)
  set Y := (r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀) with hYd
  have hd1 : r.ξ * r.pAP * r.s₀ ≤ 1 := mul_le_one₀ (mul_le_one₀ hξ1 hp.le hp1) hs.le (by linarith)
  have hQA : (r.Q : ℝ) * r.A ≤ (r.L + r.Q) * r.A * r.R :=
    (mul_le_mul_of_nonneg_right (by linarith : (r.Q : ℝ) ≤ r.L + r.Q) (by linarith)).trans
      (le_mul_of_one_le_right (by positivity) hR)
  have hQAY : (r.Q : ℝ) * r.A / r.δ ≤ Y :=
    calc (r.Q : ℝ) * r.A / r.δ ≤ (r.L + r.Q) * r.A * r.R / r.δ :=
          div_le_div_of_nonneg_right hQA hδ.le
      _ ≤ (r.L + r.Q) * r.A * r.R / (r.δ * (r.ξ * r.pAP * r.s₀)) :=
          div_le_div_of_nonneg_left (by positivity) (by positivity)
            (mul_le_of_le_one_right hδ.le hd1)
      _ = Y := by rw [hYd]; ring
  have hQQA : (r.Q : ℝ) / r.δ ≤ r.Q * r.A / r.δ :=
    div_le_div_of_nonneg_right (le_mul_of_one_le_right (by linarith) (by linarith)) hδ.le
  have hl1 : 0 ≤ log ((r.Q : ℝ) / r.δ) := Real.log_nonneg ((one_le_div hδ).2 (by linarith))
  have hl2 : log ((r.Q : ℝ) / r.δ) ≤ log (r.Q * r.A / r.δ) :=
    Real.log_le_log (div_pos (by linarith) hδ) hQQA
  have hl3 : log ((r.Q : ℝ) * r.A / r.δ) ≤ log Y :=
    Real.log_le_log (div_pos (by positivity) hδ) hQAY
  set b : ℝ := r.Q ^ 7 * r.A ^ 3 * (r.L + r.Q) ^ 3 * r.R ^ 3
    * ((1 / r.ξ) ^ 3 * (1 / r.s₀) ^ 4) with hbd
  have hb0 : 0 ≤ b := by rw [hbd]; positivity
  have ha : (r.Q : ℝ) ^ 3 * (r.A * r.A) * (r.L + r.Q) * r.R * (1 / r.ξ) ≤ b := by
    have f1 : (r.Q : ℝ) ^ 3 ≤ r.Q ^ 7 := pow_le_pow_right₀ hQ' (by norm_num)
    have f2 : (r.A : ℝ) * r.A ≤ r.A ^ 3 :=
      calc (r.A : ℝ) * r.A ≤ r.A * r.A :=
            mul_le_mul_of_nonneg_right (by linarith) (by linarith)
        _ = r.A ^ 2 := (sq _).symm
        _ ≤ r.A ^ 3 := pow_le_pow_right₀ hAQ (by norm_num)
    have f3 : (r.L : ℝ) + r.Q ≤ (r.L + r.Q) ^ 3 := le_self_pow₀ hLQ (by norm_num)
    have f4 : r.R ≤ r.R ^ 3 := le_self_pow₀ hR (by norm_num)
    have f5 : 1 / r.ξ ≤ (1 / r.ξ) ^ 3 * (1 / r.s₀) ^ 4 :=
      (le_self_pow₀ hiξ (by norm_num)).trans
        (le_mul_of_one_le_right (by positivity) (one_le_pow₀ his))
    rw [hbd]
    gcongr ?_ * ?_ * ?_ * ?_ * ?_
  have hQb : (1 / r.ξ) ^ 2 ≤ r.Q * b := by
    have e : r.Q * b = r.Q ^ 8 * r.A ^ 3 * (r.L + r.Q) ^ 3 * r.R ^ 3 * (1 / r.ξ)
        * (1 / r.s₀) ^ 4 * (1 / r.ξ) ^ 2 := by rw [hbd]; ring
    rw [e]
    refine le_mul_of_one_le_left (by positivity) ?_
    exact one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ') (one_le_pow₀ hAQ)) (one_le_pow₀ hLQ))
        (one_le_pow₀ hR)) hiξ) (one_le_pow₀ his)
  have e1 : r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
      = r.Q ^ 3 * (r.A * r.A) * (r.L + r.Q) * r.R * (1 / r.ξ)
        * log (r.Q * r.A / r.δ) := by ring
  have e2 : r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 * log Y / (r.ξ ^ 2 * r.s₀ ^ 4)
      * (r.R * r.A * (r.L + r.Q) / r.ξ) = b * log Y := by rw [hbd]; ring
  have e3 : r.Q * (r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 * log Y
      / (r.ξ ^ 2 * r.s₀ ^ 4)) * (r.R * r.A * (r.L + r.Q) / r.ξ) = r.Q * b * log Y := by
    rw [hbd]; ring
  have e4 : log (r.Q / r.δ) / r.ξ ^ 2 = (1 / r.ξ) ^ 2 * log (r.Q / r.δ) := by ring
  refine ⟨?_, ?_, half_le_log (by linarith)⟩
  · rw [e1, e2]
    exact mul_le_mul ha hl3 (hl1.trans hl2) hb0
  · rw [e4, e3]
    exact mul_le_mul hQb ((hl2.trans hl3)) hl1 (mul_nonneg (by linarith) hb0)

theorem rlStarClaimUniformSimple_of_claim (h : RLStarClaimUniform.{u, v, w}) :
    RLStarClaimUniformSimple.{u, v, w} := by
  obtain ⟨k₀, pref, suff, probes, reads, hk₀, hp, hs, hpr, hrd, hw⟩ := h
  refine ⟨k₀, pref, suff, probes, reads, hk₀,
    hp.trans (isBigO_on 4 fun r hr => ?_), hs.trans (isBigO_on 2 fun r hr => ?_), hpr,
    hrd.trans (isBigO_on 32 fun r hr => ?_), hw⟩
  · have h := loop_pref_dom (r := ratioOneRun r) (ratioOne_mem hr)
    simp only [ratioOneRun, ratioOne, mul_one, one_pow] at h
    exact h
  · have h := loop_suff_dom (r := ratioOneRun r) (ratioOne_mem hr)
    simp only [ratioOneRun, ratioOne, mul_one] at h
    exact h
  have hr' : ratioOneRun r ∈ ratioRunRegion := ratioOne_mem hr
  obtain ⟨hp0, hp0', hp1⟩ := loop_pref_dom hr'
  obtain ⟨hs0, hs0', hs1⟩ := loop_suff_dom hr'
  obtain ⟨hQ, hF, hPr, hlg, hρ, -⟩ := uniform_factors hr'
  obtain ⟨hPrle, -, hlg2⟩ := uniform_search_start_le hr'
  have h4 := reads_dom (Hd := 0) hQ hp0 hp1 hs0 hs1 hF hPr hlg hρ le_rfl
  have h2 := reads_two_term (Hd := 0) hQ hp0' hs0' hF hPr hlg2 hρ le_rfl hPrle
    (mul_nonneg (mul_nonneg hQ.le hp0') hρ)
  simp only [ratioOneRun, ratioOne, mul_one, one_mul, one_pow, zero_mul, add_zero] at h4 h2
  dsimp only
  refine ⟨h4.1, h2.2.1, h4.2.2.trans ?_⟩
  linarith [h2.2.2]

end OrthoDFA

end
