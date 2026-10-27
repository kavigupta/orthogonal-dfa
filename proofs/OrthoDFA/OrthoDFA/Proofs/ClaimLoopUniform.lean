import OrthoDFA.Proofs.ClaimLoopArith
import OrthoDFA.Proofs.DFABridge

/-!
# The uniform claims from the general one

`LoopClaim` is `LoopDistClaim` at `D` uniform over the strings of length `L + 2|Q| + 6` and
`β = 1/|Σ|`: uniform strings extend a prefix by each letter `1/|Σ|` of the time
(`uniformStrings_floor`) and hold `|Σ|^-k` on any first `k` letters, and `1/p₀Tol` is at most
`C₀ (L + |Q|)⁶ G` for a constant `C₀`, `G` free of `L`, so `L ≥ L₀` gives
`(L + |Q|)⁶ C₀ G ≤ |Σ|^k` (`pow_of_tangent`). `LoopClaimSimple` follows from `LoopClaim`, the
prefix and suffix totals' shapes bounded by their products.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

/-- A uniform instance as a general one, at `β = 1/|Σ|`. -/
def toDist (x : LoopInputs) : LoopDistInputs := { toLoopInputs := x, β := 1 / x.A }

def toDistRun (r : LoopRun) : LoopDistRun := { toLoopDistInputs := toDist r.toLoopInputs, L := r.L }

lemma toDist_mem {x : LoopInputs} (hx : x ∈ loopRegion) : toDist x ∈ loopDistRegion := by
  have hA : (2 : ℝ) ≤ x.A := by exact_mod_cast hx.2.1
  refine ⟨hx, ?_, ?_⟩
  · show 0 < 1 / (x.A : ℝ); positivity
  · show 1 / (x.A : ℝ) * x.A ≤ 1
    rw [one_div_mul_cancel (by positivity)]

/-- `1/p₀Tol` at `β = 1/|Σ|`, with `(L + |Q|)⁶` taken out. -/
def loopG (x : LoopInputs) : ℝ :=
  x.Q ^ 14 * x.R ^ 4 * log (x.Q * x.R * x.A * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀)) ^ 2
    * x.A ^ 4 * (1 / x.δ) * (1 / x.ξ) ^ 4 * (1 / x.pAP) ^ 3 * (1 / x.s₀) ^ 11

lemma shape_toDist {x : LoopInputs} (hx : x ∈ loopRegion) (L : ℕ) :
    ((L : ℝ) + x.Q) ^ 6 * x.Q ^ 14 * x.R ^ 4
        * log (x.Q * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
        / ((toDist x).β ^ 4 * x.δ * x.ξ ^ 4 * x.pAP ^ 3 * x.s₀ ^ 11)
      = ((L : ℝ) + x.Q) ^ 6 * loopG x := by
  obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hx
  have hA' : (0 : ℝ) < x.A := by exact_mod_cast (show 0 < x.A by omega)
  have e : (x.Q : ℝ) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)
      = x.Q * x.R * x.A * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀) := by
    show (x.Q : ℝ) * x.R / (1 / x.A * x.δ * x.ξ * x.pAP * x.s₀) = _
    field_simp
  rw [e, loopG]
  show _ / ((1 / (x.A : ℝ)) ^ 4 * x.δ * x.ξ ^ 4 * x.pAP ^ 3 * x.s₀ ^ 11) = _
  field_simp

theorem loopClaim_of_dist (h : LoopDistClaim.{u, v, w}) : LoopClaim.{u, v, w} := by
  obtain ⟨p₀Tol, tp, ts, tpr, ho, hp₀, hpref, hsuff, hprobes, hho, hpos, hw⟩ := h
  obtain ⟨C, hC⟩ := isBigO_principal.1 hp₀
  set C₀ := max 1 C with hC₀d
  have hC₀ : 1 ≤ C₀ := le_max_left _ _
  have hCC₀ : C ≤ C₀ := le_max_right _ _
  have hmapsR : Set.MapsTo toDistRun loopRunRegion loopDistRunRegion := fun r hr => toDist_mem hr
  have hmaps : Set.MapsTo toDist loopRegion loopDistRegion := fun x hx => toDist_mem hx
  refine ⟨fun x => 1 + 4 * (log (C₀ * loopG x) + 6 * log (x.Q + 4) + 24) / log x.A,
    fun r => tp (toDistRun r), fun r => ts (toDistRun r), fun r => tpr (toDistRun r),
    fun x => ho (toDist x), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine isBigO_on (241 + 6 * log C₀) fun x hx => ?_
    obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
    have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ
    have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hA
    obtain ⟨hG, hle⟩ := loopL0_le (iδ := 1 / x.δ) (iξ := 1 / x.ξ) (ip := 1 / x.pAP) (is := 1 / x.s₀)
      hQ' hA' hR ((le_div_iff₀ hδ).2 (by linarith))
      ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
      ((le_div_iff₀ hs).2 (by linarith)) hC₀
    have hlA : 0 < log (x.A : ℝ) := Real.log_pos (by linarith)
    have hnum : 0 ≤ 4 * (log (C₀ * loopG x) + 6 * log (x.Q + 4) + 24) := by
      have : 0 ≤ log (C₀ * loopG x) := Real.log_nonneg (one_le_mul_of_one_le_of_one_le hC₀ hG)
      have : 0 ≤ log ((x.Q : ℝ) + 4) := Real.log_nonneg (by linarith)
      linarith
    have hZ : 1 ≤ (x.Q : ℝ) * x.R / (x.δ * x.ξ * x.pAP * x.s₀) := by
      rw [le_div_iff₀ (by positivity), one_mul]
      have : x.δ * x.ξ * x.pAP * x.s₀ ≤ 1 := by
        have : x.δ * x.ξ ≤ 1 := by nlinarith
        have : x.δ * x.ξ * x.pAP ≤ 1 := by nlinarith
        nlinarith
      nlinarith
    have e : (x.Q : ℝ) * x.R * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀)
        = x.Q * x.R / (x.δ * x.ξ * x.pAP * x.s₀) := by ring
    rw [e] at hle
    refine ⟨by positivity, ?_, hle⟩
    exact add_nonneg zero_le_one (div_nonneg (Real.log_nonneg hZ) hlA.le)
  · refine (isBigO_comp_on hpref hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    show _ = r.Q ^ 6 * (r.L + r.Q) ^ 2 * r.A ^ 2 * r.R ^ 2
      * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
      * (1 / r.s₀ ^ 8 + r.Q ^ 4 / r.s₀ ^ 4)
    simp only [Function.comp, toDistRun, toDist]
    have e : ((r.L : ℝ) + r.Q) * r.R / (1 / r.A * r.δ * r.ξ * r.pAP * r.s₀)
        = (r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀) := by field_simp
    rw [e]
    field_simp
  · refine (isBigO_comp_on hsuff hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    show _ = r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP ^ 2)
    simp only [Function.comp, toDistRun, toDist]
    have e : ((r.L : ℝ) + r.Q) * r.R / (1 / r.A * r.δ * r.ξ * r.s₀)
        = (r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀) := by field_simp
    rw [e]
  · refine (isBigO_comp_on hprobes hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    show _ = r.Q ^ 4 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
    simp only [Function.comp, toDistRun, toDist]
    field_simp
  · exact isBigO_comp_on hho hmaps
  -- the run, at `D` uniform
  intro A _ _ _ Q _ Ω _ μ _ x M O L γ k L₁ hx hQ hA hL hη hpAPu hdom hL₀
  have hxd := toDist_mem hx
  have hrd : ({ toDist x with L } : LoopDistRun) ∈ loopDistRunRegion := hxd
  obtain ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ1
  have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hA2
  have hAc : (Fintype.card A : ℝ) = x.A := by exact_mod_cast hA
  -- `D`'s first `k` letters weigh at most `p₀Tol`
  have hp₀ : ((x.A : ℝ) ^ k)⁻¹ ≤ p₀Tol { toDist x with L } := by
    have hc := hC _ hrd
    have hpos' := hpos { toDist x with L }
    have hsh : 0 ≤ ((L : ℝ) + x.Q) ^ 6 * loopG x := by
      have := (loopL0_le (iδ := 1 / x.δ) (iξ := 1 / x.ξ) (ip := 1 / x.pAP) (is := 1 / x.s₀)
      hQ' hA' hR ((le_div_iff₀ hδ).2 (by linarith))
        ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
        ((le_div_iff₀ hs).2 (by linarith)) hC₀).1
      have : 1 ≤ loopG x := this
      positivity
    have hG1 : 1 ≤ loopG x := (loopL0_le (iδ := 1 / x.δ) (iξ := 1 / x.ξ) (ip := 1 / x.pAP) (is := 1 / x.s₀)
      hQ' hA' hR ((le_div_iff₀ hδ).2 (by linarith))
      ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
      ((le_div_iff₀ hs).2 (by linarith)) hC₀).1
    have e := shape_toDist ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ L
    have hc' : 1 / p₀Tol { toDist x with L } ≤ C₀ * (((L : ℝ) + x.Q) ^ 6 * loopG x) := by
      rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (one_div_pos.2 hpos')] at hc
      change _ ≤ C * |((L : ℝ) + x.Q) ^ 6 * x.Q ^ 14 * x.R ^ 4
        * log (x.Q * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
        / ((toDist x).β ^ 4 * x.δ * x.ξ ^ 4 * x.pAP ^ 3 * x.s₀ ^ 11)| at hc
      rw [e, abs_of_nonneg hsh] at hc
      exact hc.trans (mul_le_mul_of_nonneg_right hCC₀ hsh)
    have hpow := pow_of_tangent (E := C₀ * loopG x) hA' hQ'
      (one_le_mul_of_one_le_of_one_le hC₀ hG1) hL₀
    have h1 : 1 / p₀Tol { toDist x with L } ≤ (x.A : ℝ) ^ k := by
      refine hc'.trans (le_of_eq_of_le ?_ hpow)
      ring
    rw [one_div] at h1
    exact (inv_le_comm₀ (pow_pos (by linarith : (0 : ℝ) < x.A) k) hpos').2 h1
  set m := L + Fintype.card Q + 3 + Fintype.card Q + 3 with hmd
  have hpL : prefixLaw (uniformStrings (α := A) m) L = uniformStrings L :=
    prefixLaw_uniformStrings (by omega)
  have hpS : prefixLaw (uniformStrings (α := A) m) m = uniformStrings m :=
    prefixLaw_uniformStrings le_rfl
  have hwS : windowLaw (uniformStrings (α := A) m) k L₁ = windowStrings k L₁ :=
    windowLaw_uniformStrings (by omega)
  obtain ⟨settings, hwk, hcnt, hprob⟩ := hw (toDist x) M O L γ (uniformStrings m) hxd hQ hA hL hη
    (uniformStrings_long m)
    (fun u => (uniformStrings_prefix_le (by omega) u).trans (by rw [hAc]; exact hp₀))
    (fun i _ hi y a => by
      have h := uniformStrings_floor (α := A) (show i + 1 ≤ m by omega) y a
      show 1 / (x.A : ℝ) * _ ≤ _
      rw [hAc] at h
      rw [one_div_mul_eq_div, div_le_iff₀ (by positivity), mul_comm]
      exact h)
    (by rw [hpS]; exact hpAPu) (by rw [hpL]; exact hdom)
  rw [hwS, hpS] at hcnt
  rw [hwS, hpS, hpL] at hprob
  exact ⟨settings, hwk, hcnt, hprob⟩

theorem loopClaimSimple_of_claim (h : LoopClaim.{u, v, w}) : LoopClaimSimple.{u, v, w} := by
  obtain ⟨L₀, pref, suff, probes, heldOut, hL₀, hp, hs, hpr, hh, hw⟩ := h
  exact ⟨L₀, pref, suff, probes, heldOut, hL₀,
    hp.trans (isBigO_on 2 fun r hr => loop_pref_dom hr),
    hs.trans (isBigO_on 2 fun r hr => loop_suff_dom hr), hpr, hh, hw⟩

end OrthoDFA

end
