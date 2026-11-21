import RLStar.Claims.RLStar
import RLStar.Proofs.ClaimReadsArith
import RLStar.Proofs.DFABridge

/-!
# The uniform claims from the general one

`RLStarClaimUniform` is `RLStarClaim` at `D` uniform over the strings of length `L + 2|Q| + 6`:
uniform strings hold `|Σ|^-k` on any first `k` letters, and `1/p₀Tol` is at most `C₀ (|Σ| V)⁹⁰`
(`tc_tolPow`), `V = (L + |Q|) log(1/δ)/(ξc pAP s₀)`, so `k ≥ k₀ = log (C₀ (|Σ| V)⁹⁰)/log |Σ|`
gives `C₀ (|Σ| V)⁹⁰ ≤ |Σ|^k`. `RLStarClaimUniformSimple` follows from `RLStarClaimUniform` and
`tc_simple`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

lemma ξc_mem {r : RLStarRun} (hr : r ∈ rlStarRunRegion) :
    0 < r.ξc ∧ r.ξc ≤ 1 ∧ r.ξc ≤ r.ξ ∧ r.ξc ≤ (r.L + r.Q) * r.β := by
  obtain ⟨hQ, -, -, -, hξ, hξ1, -, -, -, -, hβ⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  exact ⟨lt_min hξ (mul_pos (by linarith) hβ), (min_le_left _ _).trans hξ1, min_le_left _ _,
    min_le_right _ _⟩

lemma run_tc {r : RLStarRun} (hr : r ∈ rlStarRunRegion) :
    TC r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀ := by
  obtain ⟨hξc, hξc1, -, -⟩ := ξc_mem hr
  obtain ⟨hQ, hA, hδ, hδ2, -, -, hp, hp1, hs, hs1, -⟩ := hr
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, by linarith, hδ, hδ2, hξc, hξc1, hp, hp1, hs,
    hs1⟩

/-- `V`, the uniform claims' `k₀` against `log |Σ|`. -/
def uniformV (r : RLStarRun) : ℝ := (r.L + r.Q) * log (1 / r.δ) / (r.ξc * r.pAP * r.s₀)

theorem rlStarClaimUniform_of_rlStarClaim (h : RLStarClaim.{u, v, w}) :
    RLStarClaimUniform.{u, v, w} := by
  obtain ⟨p₀Tol, tp, ts, tpr, tr, hp₀, hpref, hsuff, hprobes, hreads, hpos, hw⟩ := h
  obtain ⟨C, hC⟩ := isBigO_principal.1 hp₀
  set C₀ := 2 ^ 45 * max 1 C with hC₀d
  have hC₀ : 1 ≤ C₀ := by
    rw [hC₀d]; have := le_max_left 1 C
    have : (1 : ℝ) ≤ 2 ^ 45 := by norm_num
    nlinarith
  clear_value C₀
  refine ⟨fun r => log (C₀ * ((r.A : ℝ) * uniformV r) ^ 90) / log r.A, tp, ts, tpr, tr, ?_,
    hpref, hsuff, hprobes, hreads, ?_⟩
  · refine isBigO_on (log C₀ / log 2 + 90) fun r hr => ?_
    obtain ⟨hV1, -, -⟩ := tc_tolPow (run_tc hr)
    change 1 ≤ uniformV r at hV1
    have hA : (2 : ℝ) ≤ r.A := (run_tc hr).hA
    beta_reduce
    set V := uniformV r
    have hl2 : 0 < log (2 : ℝ) := Real.log_pos (by norm_num)
    have hℓ : log 2 ≤ log (r.A : ℝ) := Real.log_le_log (by norm_num) hA
    have hℓ0 : 0 < log (r.A : ℝ) := by linarith
    have hlV : 0 ≤ log V := Real.log_nonneg hV1
    have hlC : 0 ≤ log C₀ := Real.log_nonneg hC₀
    have e0 : ∀ a b c : ℝ, b ≠ 0 → (a + 90 * (b + c)) / b = a / b + 90 + 90 * (c / b) := by
      intro a b c hb; field_simp; ring
    have e : log (C₀ * ((r.A : ℝ) * V) ^ 90) / log r.A
        = log C₀ / log r.A + 90 + 90 * (log V / log r.A) := by
      rw [Real.log_mul (by positivity) (by positivity), Real.log_pow,
        Real.log_mul (by positivity) (by positivity)]
      push_cast
      exact e0 _ _ _ hℓ0.ne'
    have k1 : log C₀ / log r.A ≤ log C₀ / log 2 := div_le_div_of_nonneg_left hlC hl2 hℓ
    have hq : 0 ≤ log V / log r.A := div_nonneg hlV hℓ0.le
    have hc : 0 ≤ log C₀ / log 2 := div_nonneg hlC hl2.le
    have hc' : 0 ≤ log C₀ / log r.A := div_nonneg hlC hℓ0.le
    show 0 ≤ log (C₀ * ((r.A : ℝ) * V) ^ 90) / log r.A ∧ 0 ≤ 1 + log V / log r.A
      ∧ log (C₀ * ((r.A : ℝ) * V) ^ 90) / log r.A ≤ (log C₀ / log 2 + 90) * (1 + log V / log r.A)
    rw [e]
    refine ⟨by positivity, by linarith, by nlinarith⟩
  -- the run, at `D` uniform
  intro A _ _ _ Q _ Ω _ μ _ x M O L k L₁ hx hQ hA hL hgap hbL hbN hpAPu hk₀ hkL
  have hr : ({ x with L } : RLStarRun) ∈ rlStarRunRegion := hx
  have ht := run_tc hr
  have hA' : (2 : ℝ) ≤ x.A := ht.hA
  have hAc : (Fintype.card A : ℝ) = x.A := by exact_mod_cast hA
  -- `D`'s first `k` letters weigh at most `p₀Tol`
  have hp₀k : ((x.A : ℝ) ^ k)⁻¹ ≤ p₀Tol { x with L } := by
    have hpos' := hpos { x with L }
    obtain ⟨hV1, hs0, hsle⟩ := tc_tolPow ht
    have hc := hC _ hr
    change ‖1 / p₀Tol { x with L }‖ ≤ C * ‖shTol (x.Q : ℝ) x.A ((L : ℝ) + x.Q) x.δ
      (RLStarRun.ξc { x with L }) x.pAP x.s₀ ^ 5‖ at hc
    rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (one_div_pos.2 hpos'),
      abs_of_nonneg (pow_nonneg hs0 5)] at hc
    change 1 ≤ uniformV { x with L } at hV1
    change _ ≤ 2 ^ 9 * ((x.A : ℝ) * uniformV { x with L }) ^ 18 at hsle
    set U := (x.A : ℝ) * uniformV { x with L } with hUd
    have hU : 1 ≤ U := by rw [hUd]; nlinarith
    have hm1 : 1 ≤ max 1 C := le_max_left 1 C
    have hc' : 1 / p₀Tol { x with L } ≤ C₀ * U ^ 90 :=
      calc 1 / p₀Tol { x with L }
          ≤ max 1 C * (2 ^ 9 * U ^ 18) ^ 5 :=
            hc.trans (mul_le_mul (le_max_right 1 C) (pow_le_pow_left₀ hs0 hsle 5)
              (pow_nonneg hs0 5) (by linarith))
        _ = C₀ * U ^ 90 := by rw [hC₀d]; ring
    have hpow : C₀ * U ^ 90 ≤ (x.A : ℝ) ^ k := by
      have hlA : 0 < log (x.A : ℝ) := Real.log_pos (by linarith)
      change log (C₀ * U ^ 90) / log (x.A : ℝ) ≤ k at hk₀
      have h9 := (div_le_iff₀ hlA).1 hk₀
      rw [← Real.log_le_log_iff (by positivity) (pow_pos (by linarith) k), Real.log_pow]
      linarith
    have h1 : 1 / p₀Tol { x with L } ≤ (x.A : ℝ) ^ k := hc'.trans hpow
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

theorem rlStarClaimUniformSimple_of_claim (h : RLStarClaimUniform.{u, v, w}) :
    RLStarClaimUniformSimple.{u, v, w} := by
  obtain ⟨k₀, pref, suff, probes, reads, hk₀, hp, hs, hpr, hrd, hw⟩ := h
  exact ⟨k₀, pref, suff, probes, reads, hk₀,
    hp.trans (isBigO_on 4 fun r hr => (tc_simple (run_tc hr)).1),
    hs.trans (isBigO_on 2 fun r hr => (tc_simple (run_tc hr)).2.1), hpr,
    hrd.trans (isBigO_on 8 fun r hr => (tc_simple (run_tc hr)).2.2), hw⟩

end OrthoDFA

end
