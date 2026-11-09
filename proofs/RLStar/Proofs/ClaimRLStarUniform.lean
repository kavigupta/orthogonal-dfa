import RLStar.Proofs.ClaimRLStarArith
import RLStar.Proofs.DFABridge

/-!
# The uniform claims from the general one

`RLStarClaimUniform` is `RLStarClaim` at `D` uniform over the strings of length `L + 2|Q| + 6` and
`β = 1/|Σ|`: uniform strings extend a prefix by each letter `1/|Σ|` of the time
(`uniformStrings_floor`) and hold `|Σ|^-k` on any first `k` letters, and `1/p₀Tol` is at most
`C₀ (L + |Q|)⁶ G` for a constant `C₀`, `G` free of `L`, so `L ≥ L₀` gives `(L + |Q|)⁶ C₀ G ≤ |Σ|^k`
(`pow_of_tangent`). `RLStarClaimUniformSimple` follows from `RLStarClaimUniform`, the prefix and
suffix totals' shapes bounded by their products, and the search's and start leaf's reads by the
harvest's.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

/-- A uniform instance as a general one, at `β = 1/|Σ|`. -/
def toDist (x : RLStarInputs) : RLStarDistInputs := { toRLStarInputs := x, β := 1 / x.A }

def toDistRun (r : RLStarRun) : RLStarDistRun :=
  { toRLStarDistInputs := toDist r.toRLStarInputs, L := r.L }

lemma toDist_mem {x : RLStarInputs} (hx : x ∈ rlStarRegion) : toDist x ∈ rlStarDistRegion := by
  have hA : (2 : ℝ) ≤ x.A := by exact_mod_cast hx.2.1
  refine ⟨hx, ?_, ?_⟩
  · show 0 < 1 / (x.A : ℝ); positivity
  · show 1 / (x.A : ℝ) * x.A ≤ 1
    rw [one_div_mul_cancel (by positivity)]

/-- `1/p₀Tol`'s shape at `β = 1/|Σ|` is at most `2 (L + |Q|)⁵ loopG`. -/
def loopG (x : RLStarInputs) : ℝ :=
  x.Q ^ 9 * (x.A + x.Q) ^ 4 * x.R ^ 4
    * log (x.Q * x.R * x.A * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀)) ^ 3
    * (1 / x.δ) * (1 / x.ξ) ^ 4 * (1 / x.pAP) ^ 2 * (1 / x.s₀) ^ 6

lemma shape_toDist_le {x : RLStarInputs} (hx : x ∈ rlStarRegion) (L : ℕ) :
    0 ≤ ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (1 / (toDist x).β) ^ 4 * x.R ^ 4
        * log (((L : ℝ) + x.Q) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀))
        * log (x.Q * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
        / (x.δ * x.ξ ^ 4 * x.pAP ^ 2 * x.s₀ ^ 6)
      ∧ ((L : ℝ) + x.Q) ^ 4 * x.Q ^ 9 * (1 / (toDist x).β) ^ 4 * x.R ^ 4
        * log (((L : ℝ) + x.Q) * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀))
        * log (x.Q * x.R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
        / (x.δ * x.ξ ^ 4 * x.pAP ^ 2 * x.s₀ ^ 6)
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
  set K : ℝ := (1 / x.δ) * (1 / x.ξ) ^ 4 * (1 / x.pAP) ^ 2 * (1 / x.s₀) ^ 6 with hKd
  have e3 : ∀ z : ℝ, z / (x.δ * x.ξ ^ 4 * x.pAP ^ 2 * x.s₀ ^ 6) = z * K := by
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

lemma loopG_ge_one {x : RLStarInputs} (hx : x ∈ rlStarRegion) : 1 ≤ loopG x := by
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

theorem rlStarClaimUniform_of_rlStarClaim (h : RLStarClaim.{u, v, w}) :
    RLStarClaimUniform.{u, v, w} := by
  obtain ⟨p₀Tol, tp, ts, tpr, tr, ho, hp₀, hpref, hsuff, hprobes, hreads, hho, hpos, hw⟩ := h
  obtain ⟨C, hC⟩ := isBigO_principal.1 hp₀
  set C₀ := 16 * max 1 C with hC₀d
  have hC₀ : 1 ≤ C₀ := by rw [hC₀d]; linarith [le_max_left 1 C]
  have hCC₀ : C ≤ max 1 C := le_max_right _ _
  have hmapsR : Set.MapsTo toDistRun rlStarRunRegion rlStarDistRunRegion :=
    fun r hr => toDist_mem hr
  have hmaps : Set.MapsTo toDist rlStarRegion rlStarDistRegion := fun x hx => toDist_mem hx
  refine ⟨fun x => 1 + 4 * (log (C₀ * loopG x) + 6 * log (x.Q + 4) + 24) / log x.A,
    fun r => tp (toDistRun r), fun r => ts (toDistRun r), fun r => tpr (toDistRun r),
    fun r => tr (toDistRun r), fun x => ho (toDist x), ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine isBigO_on (263 + 6 * log C₀) fun x hx => ?_
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
    show _ = r.Q ^ 7 * r.A ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / (r.ξ ^ 2 * r.s₀ ^ 4)
          * log (r.Q * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.δ * r.pAP * r.s₀))
        + r.Q ^ 2 * (r.L + r.Q) ^ 2 * r.R ^ 2 / r.ξ ^ 2
          * (1 / r.s₀ ^ 4 + r.Q ^ 4 * r.A ^ 2)
          * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀))
    simp only [Function.comp, toDistRun, toDist, one_div_one_div]
    obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := r.R) hA' hδ hξ hp hs
    rw [e, e']
  · refine (isBigO_comp_on hsuff hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    show _ = r.Q * (log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.s₀)) / (r.pAP * r.s₀ ^ 2)
        + log (r.Q / r.δ) / r.pAP)
    simp only [Function.comp, toDistRun, toDist]
    rw [(uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := r.R) hA' hδ hξ hp hs).2]
  · refine (isBigO_comp_on hprobes hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    show _ = r.Q ^ 3 * r.A ^ 2 * (r.L + r.Q) * r.R / r.ξ * log (r.Q * r.A / r.δ)
    simp only [Function.comp, toDistRun, toDist, one_div_one_div]
    ring
  · refine (isBigO_comp_on hreads hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, toDistRun, toDist, one_div_one_div]
    obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := r.R) hA' hδ hξ hp hs
    rw [e, e']
    ring
  · exact isBigO_comp_on hho hmaps
  -- the run, at `D` uniform
  intro A _ _ _ Q _ Ω _ μ _ x M O L γ k L₁ hx hQ hA hL hη hpAPu hdom hL₀
  have hxd := toDist_mem hx
  have hrd : ({ toDist x with L } : RLStarDistRun) ∈ rlStarDistRunRegion := hxd
  obtain ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ1
  have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hA2
  have hAc : (Fintype.card A : ℝ) = x.A := by exact_mod_cast hA
  -- `D`'s first `k` letters weigh at most `p₀Tol`
  have hp₀ : ((x.A : ℝ) ^ k)⁻¹ ≤ p₀Tol { toDist x with L } := by
    have hc := hC _ hrd
    have hpos' := hpos { toDist x with L }
    have hG1 : 1 ≤ loopG x := loopG_ge_one ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩
    obtain ⟨-, hsh⟩ := shape_toDist_le ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ L
    have hc' : 1 / p₀Tol { toDist x with L } ≤ C₀ * (((L : ℝ) + x.Q) ^ 6 * loopG x) := by
      rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (one_div_pos.2 hpos')] at hc
      obtain ⟨hs0, hsle⟩ := p₀Shape_le hrd
      change _ ≤ C * |p₀Shape { toDist x with L }| at hc
      rw [abs_of_nonneg hs0] at hc
      have h8 : p₀Shape { toDist x with L } ≤ 8 * (2 * (((L : ℝ) + x.Q) ^ 6 * loopG x)) :=
        hsle.trans (mul_le_mul_of_nonneg_left hsh (by norm_num))
      refine hc.trans ((mul_le_mul_of_nonneg_right hCC₀ hs0).trans ?_)
      refine (mul_le_mul_of_nonneg_left h8 ?_).trans (le_of_eq ?_)
      · linarith [le_max_left 1 C]
      · rw [hC₀d]; ring
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
  obtain ⟨settings, hwk, hcnt, hprob⟩ := hw (toDist x) M O L γ (uniformStrings m) hxd hQ
    hA hL hη
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

/-- The reads' factors other than the prefix and suffix totals, at uniform strings. -/
lemma uniform_factors {r : RLStarRun} (hr : r ∈ rlStarRunRegion) :
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
lemma uniform_search_start_le {r : RLStarRun} (hr : r ∈ rlStarRunRegion) :
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
  obtain ⟨L₀, pref, suff, probes, reads, heldOut, hL₀, hp, hs, hpr, hrd, hh, hw⟩ := h
  refine ⟨L₀, pref, suff, probes, reads, heldOut, hL₀,
    hp.trans (isBigO_on 4 fun r hr => loop_pref_dom hr),
    hs.trans (isBigO_on 2 fun r hr => loop_suff_dom hr), hpr,
    hrd.trans (isBigO_on 32 fun r hr => ?_), hh, hw⟩
  obtain ⟨hp0, hp0', hp1⟩ := loop_pref_dom hr
  obtain ⟨hs0, hs0', hs1⟩ := loop_suff_dom hr
  obtain ⟨hQ, hF, hPr, hlg, hρ, hHd⟩ := uniform_factors hr
  obtain ⟨hPrle, hHdle, hlg2⟩ := uniform_search_start_le hr
  have h4 := reads_dom hQ hp0 hp1 hs0 hs1 hF hPr hlg hρ hHd
  have h2 := reads_two_term hQ hp0' hs0' hF hPr hlg2 hρ hHd hPrle hHdle
  dsimp only
  refine ⟨h4.1, h2.2.1, h4.2.2.trans ?_⟩
  linarith [h2.2.2]

end OrthoDFA

end
