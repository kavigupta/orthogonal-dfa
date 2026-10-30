import OrthoDFA.Proofs.ClaimLoopArith
import OrthoDFA.Proofs.DFABridge

/-!
# The uniform claims from the general one

`LoopClaimUniform` is `LoopClaim` at `D` uniform over the strings of length `L + 2|Q| + 6` and
`β = 1/|Σ|`: uniform strings extend a prefix by each letter `1/|Σ|` of the time
(`uniformStrings_floor`) and hold `|Σ|^-k` on any first `k` letters, and `1/p₀Tol` is at most
`C₀ (L + |Q|)⁶ G` for a constant `C₀`, `G = |Σ|⁷ Z³⁸` free of `L`, so `L ≥ L₀` gives `(L + |Q|)⁶ C₀ G ≤ |Σ|^k`
(`pow_of_tangent`). `LoopClaimSimple` follows from `LoopClaimUniform`, the prefix and suffix
totals' shapes bounded by their products.
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

/-- `|Σ|⁷ Z³⁸`, `Z = |Q| R/(δ ξ pAP s₀)`: `1/p₀Tol`'s shape at `β = 1/|Σ|` is at most
`16 (L + |Q|)⁶ loopG`. -/
def loopG (x : LoopInputs) : ℝ :=
  x.A ^ 7 * (x.Q * x.R * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀)) ^ 38

set_option maxHeartbeats 1000000 in -- three terms, each against `|Σ|⁷ Z³⁸`
lemma shape_toDist_le {x : LoopInputs} (hx : x ∈ loopRegion) (L : ℕ) :
    0 ≤ p₀Shape { toDist x with L } ∧ p₀Shape { toDist x with L } ≤ 16 * (((L : ℝ) + x.Q) ^ 6 * loopG x) := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hA
  have hLQ : (1 : ℝ) ≤ (L : ℝ) + x.Q := by have : (0 : ℝ) ≤ L := Nat.cast_nonneg L; linarith
  have hiδ : 1 ≤ 1 / x.δ := (le_div_iff₀ hδ).2 (by linarith)
  have hiξ : 1 ≤ 1 / x.ξ := (le_div_iff₀ hξ).2 (by linarith)
  have hip : 1 ≤ 1 / x.pAP := (le_div_iff₀ hp).2 (by linarith)
  have his : 2 ≤ 1 / x.s₀ := (le_div_iff₀ hs).2 (by linarith)
  set iδ := 1 / x.δ with hiδd
  set iξ := 1 / x.ξ with hiξd
  set ip := 1 / x.pAP with hipd
  set is := 1 / x.s₀ with hisd
  set Q : ℝ := (x.Q : ℝ)
  set A : ℝ := (x.A : ℝ)
  set R := x.R
  set LQ : ℝ := (L : ℝ) + Q with hLQd
  set Z := Q * R * iδ * iξ * ip * is with hZd
  have hs1' : 1 ≤ is := by linarith
  have hA1 : 1 ≤ A := by linarith
  have hZ1 : 1 ≤ Z := one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
    (one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le hQ' hR) hiδ) hiξ) hip) hs1'
  have hl : ∀ y ∈ [Q, R, iδ, iξ, ip, is], 1 ≤ y := by simp [hQ', hR, hiδ, hiξ, hip, hs1']
  have hsub (l : List ℝ) (hl' : l.Sublist [Q, R, iδ, iξ, ip, is]) : l.prod ≤ Z := by
    simpa [mul_assoc, hZd] using prod_le_prod_of_sublist hl' hl
  have hQZ : Q ≤ Z := by simpa using hsub [Q] (by simp)
  have hRZ : R ≤ Z := by simpa using hsub [R] (by simp)
  have hδZ : iδ ≤ Z := by simpa using hsub [iδ] (by simp)
  have hξZ : iξ ≤ Z := by simpa using hsub [iξ] (by simp)
  have hpZ : ip ≤ Z := by simpa using hsub [ip] (by simp)
  have hsZ : is ≤ Z := by simpa using hsub [is] (by simp)
  have hZ4 : 4 ≤ Z := by
    have h := hsub [iδ, is] (by simp)
    simp only [List.prod_cons, List.prod_nil, mul_one] at h
    have : 4 ≤ iδ * is := by
      have : 2 ≤ iδ := (le_div_iff₀ hδ).2 (by linarith)
      nlinarith
    linarith
  set Y := A * Z with hYd
  have hY : 3 ≤ Y := by rw [hYd]; nlinarith
  have hlY : 1 ≤ log Y := one_le_log_of_three_le hY
  have hlYY : log Y ≤ A * Z := Real.log_le_self (by linarith)
  have hlLY : log (LQ * Y) ≤ 2 * LQ * log Y := by
    rw [Real.log_mul (by linarith) (by linarith)]
    have := Real.log_le_sub_one_of_pos (show (0 : ℝ) < LQ by linarith)
    nlinarith
  have hlLY0 : 0 ≤ log (LQ * Y) := Real.log_nonneg (by nlinarith)
  have e1 : LQ * Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀) = LQ * Y := by
    show LQ * Q * R / (1 / A * x.δ * x.ξ * x.pAP * x.s₀) = _
    rw [hYd, hZd, hiδd, hiξd, hipd, hisd]; field_simp
  have e2 : Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀) = Y := by
    show Q * R / (1 / A * x.δ * x.ξ * x.pAP * x.s₀) = _
    rw [hYd, hZd, hiδd, hiξd, hipd, hisd]; field_simp
  set F₁ := A * R * Q ^ 4 * iξ * (R ^ 3 * Q ^ 3 * iξ ^ 3 * (is ^ 11 + Q ^ 6 * A ^ 3 * is ^ 5)
    + R ^ 2 * Q ^ 2 * iξ ^ 2 * iδ * (is ^ 4 + Q ^ 4 * A ^ 2) + iδ * ip ^ 2 * is ^ 2) with hF₁d
  set F₂ := Q ^ 5 * R ^ 2 * iξ ^ 4 * iδ * (is ^ 8 + Q ^ 4 * A ^ 2 * is ^ 4) with hF₂d
  set F₃ := Q ^ 7 * A ^ 4 * R ^ 2 * iξ ^ 2 * iδ with hF₃d
  have hsh : p₀Shape { toDist x with L }
      = LQ ^ 4 * log (LQ * Y) * log Y * F₁ + LQ ^ 2 * log (LQ * Y) * log Y ^ 2 * F₂
        + LQ ^ 2 * log Y ^ 2 * F₃ := by
    show LQ ^ 4 * log (LQ * Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀))
          * log (Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) * Q ^ 4 * R
          / ((toDist x).β * x.ξ)
          * (Q ^ 3 * R ^ 3 / x.ξ ^ 3 * (1 / x.s₀ ^ 11 + Q ^ 6 / ((toDist x).β ^ 3 * x.s₀ ^ 5))
            + Q ^ 2 * R ^ 2 / (x.ξ ^ 2 * x.δ) * (1 / x.s₀ ^ 4 + Q ^ 4 / (toDist x).β ^ 2)
            + 1 / (x.δ * x.pAP ^ 2 * x.s₀ ^ 2))
        + LQ ^ 2 * log (LQ * Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀))
          * log (Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2 * Q ^ 5 * R ^ 2
          / (x.ξ ^ 4 * x.δ) * (1 / x.s₀ ^ 8 + Q ^ 4 / ((toDist x).β ^ 2 * x.s₀ ^ 4))
        + LQ ^ 2 * log (Q * R / ((toDist x).β * x.δ * x.ξ * x.pAP * x.s₀)) ^ 2
          * Q ^ 7 * A ^ 2 * R ^ 2 / ((toDist x).β ^ 2 * x.ξ ^ 2 * x.δ) = _
    rw [e1, e2]
    rw [show (toDist x).β = 1 / A from rfl]
    rw [hF₁d, hF₂d, hF₃d, hiδd, hiξd, hipd, hisd]
    field_simp
  have hmono : ∀ a b : ℕ, a ≤ 4 → b ≤ 26 → A ^ a * Z ^ b ≤ A ^ 4 * Z ^ 26 := fun a b ha hb =>
    mul_le_mul (pow_le_pow_right₀ hA1 ha) (pow_le_pow_right₀ hZ1 hb) (by positivity) (by positivity)
  have hF₁ : F₁ ≤ 5 * (A ^ 4 * Z ^ 26) := by
    have h : F₁ ≤ A * Z * Z ^ 4 * Z * (Z ^ 3 * Z ^ 3 * Z ^ 3 * (Z ^ 11 + Z ^ 6 * A ^ 3 * Z ^ 5)
        + Z ^ 2 * Z ^ 2 * Z ^ 2 * Z * (Z ^ 4 + Z ^ 4 * A ^ 2) + Z * Z ^ 2 * Z ^ 2) := by
      rw [hF₁d]; gcongr
    have e : A * Z * Z ^ 4 * Z * (Z ^ 3 * Z ^ 3 * Z ^ 3 * (Z ^ 11 + Z ^ 6 * A ^ 3 * Z ^ 5)
        + Z ^ 2 * Z ^ 2 * Z ^ 2 * Z * (Z ^ 4 + Z ^ 4 * A ^ 2) + Z * Z ^ 2 * Z ^ 2)
        = A ^ 1 * Z ^ 26 + A ^ 4 * Z ^ 26 + A ^ 1 * Z ^ 17 + A ^ 3 * Z ^ 17 + A ^ 1 * Z ^ 11 := by
      ring
    linarith [hmono 1 26 (by norm_num) (by norm_num), hmono 1 17 (by norm_num) (by norm_num),
      hmono 3 17 (by norm_num) (by norm_num), hmono 1 11 (by norm_num) (by norm_num)]
  have hF₂ : F₂ ≤ 2 * (A ^ 4 * Z ^ 26) := by
    have h : F₂ ≤ Z ^ 5 * Z ^ 2 * Z ^ 4 * Z * (Z ^ 8 + Z ^ 4 * A ^ 2 * Z ^ 4) := by
      rw [hF₂d]; gcongr
    have e : Z ^ 5 * Z ^ 2 * Z ^ 4 * Z * (Z ^ 8 + Z ^ 4 * A ^ 2 * Z ^ 4)
        = A ^ 0 * Z ^ 20 + A ^ 2 * Z ^ 20 := by ring
    linarith [hmono 0 20 (by norm_num) (by norm_num), hmono 2 20 (by norm_num) (by norm_num)]
  have hF₃ : F₃ ≤ A ^ 4 * Z ^ 26 := by
    have h : F₃ ≤ Z ^ 7 * A ^ 4 * Z ^ 2 * Z ^ 2 * Z := by rw [hF₃d]; gcongr
    have e : Z ^ 7 * A ^ 4 * Z ^ 2 * Z ^ 2 * Z = A ^ 4 * Z ^ 12 := by ring
    linarith [hmono 4 12 (by norm_num) (by norm_num)]
  have hF₁0 : 0 ≤ F₁ := by rw [hF₁d]; positivity
  have hF₂0 : 0 ≤ F₂ := by rw [hF₂d]; positivity
  have hF₃0 : 0 ≤ F₃ := by rw [hF₃d]; positivity
  have hlY0 : 0 ≤ log Y := by linarith
  rw [hsh]
  refine ⟨by positivity, ?_⟩
  have hLQ6 : ∀ k : ℕ, k ≤ 6 → LQ ^ k ≤ LQ ^ 6 := fun k hk => pow_le_pow_right₀ hLQ hk
  have hlY3 : log Y ^ 3 ≤ (A * Z) ^ 3 := pow_le_pow_left₀ hlY0 hlYY 3
  have hlY2 : log Y ^ 2 ≤ log Y ^ 3 := by nlinarith
  have hAZ : (A * Z) ^ 3 * (A ^ 4 * Z ^ 26) ≤ loopG x := by
    show (A * Z) ^ 3 * (A ^ 4 * Z ^ 26) ≤ A ^ 7 * Z ^ 38
    have : Z ^ 29 ≤ Z ^ 38 := pow_le_pow_right₀ hZ1 (by norm_num)
    nlinarith [show (A * Z) ^ 3 * (A ^ 4 * Z ^ 26) = A ^ 7 * Z ^ 29 by ring,
      pow_pos (show (0 : ℝ) < A by linarith) 7]
  set W := A ^ 4 * Z ^ 26
  have hW0 : 0 ≤ W := by positivity
  have t1 : LQ ^ 4 * log (LQ * Y) * log Y * F₁ ≤ 10 * (LQ ^ 6 * (log Y ^ 3 * W)) := by
    calc LQ ^ 4 * log (LQ * Y) * log Y * F₁ ≤ LQ ^ 4 * (2 * LQ * log Y) * log Y * (5 * W) := by
          gcongr
      _ = 10 * (LQ ^ 5 * (log Y ^ 2 * W)) := by ring
      _ ≤ 10 * (LQ ^ 6 * (log Y ^ 3 * W)) := by
          have := mul_le_mul (hLQ6 5 (by norm_num)) (mul_le_mul_of_nonneg_right hlY2 hW0)
            (by positivity) (by positivity)
          linarith
  have t2 : LQ ^ 2 * log (LQ * Y) * log Y ^ 2 * F₂ ≤ 4 * (LQ ^ 6 * (log Y ^ 3 * W)) := by
    calc LQ ^ 2 * log (LQ * Y) * log Y ^ 2 * F₂ ≤ LQ ^ 2 * (2 * LQ * log Y) * log Y ^ 2 * (2 * W) :=
          by gcongr
      _ = 4 * (LQ ^ 3 * (log Y ^ 3 * W)) := by ring
      _ ≤ 4 * (LQ ^ 6 * (log Y ^ 3 * W)) := by
          have := mul_le_mul_of_nonneg_right (hLQ6 3 (by norm_num))
            (by positivity : 0 ≤ log Y ^ 3 * W)
          linarith
  have t3 : LQ ^ 2 * log Y ^ 2 * F₃ ≤ LQ ^ 6 * (log Y ^ 3 * W) := by
    have := mul_le_mul (mul_le_mul (hLQ6 2 (by norm_num)) hlY2 (by positivity) (by positivity))
      hF₃ hF₃0 (by positivity)
    linarith [show LQ ^ 6 * log Y ^ 3 * W = LQ ^ 6 * (log Y ^ 3 * W) by ring]
  have hfin : LQ ^ 6 * (log Y ^ 3 * W) ≤ LQ ^ 6 * loopG x := by
    gcongr
    calc log Y ^ 3 * W ≤ (A * Z) ^ 3 * W := by gcongr
      _ ≤ loopG x := hAZ
  have : 0 ≤ LQ ^ 6 * (log Y ^ 3 * W) := by positivity
  linarith

lemma loopG_ge_one {x : LoopInputs} (hx : x ∈ loopRegion) : 1 ≤ loopG x := by
  obtain ⟨hQ, hA, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  exact (loopL0_le (iδ := 1 / x.δ) (iξ := 1 / x.ξ) (ip := 1 / x.pAP) (is := 1 / x.s₀) (C₀ := 1)
    (by exact_mod_cast hQ) (by exact_mod_cast hA) hR ((le_div_iff₀ hδ).2 (by linarith))
    ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
    ((le_div_iff₀ hs).2 (by linarith)) le_rfl).1

theorem loopClaimUniform_of_loopClaim (h : LoopClaim.{u, v, w}) : LoopClaimUniform.{u, v, w} := by
  obtain ⟨p₀Tol, tp, ts, tpr, ho, hp₀, hpref, hsuff, hprobes, hho, hpos, hw⟩ := h
  obtain ⟨C, hC⟩ := isBigO_principal.1 hp₀
  set C₀ := 16 * max 1 C with hC₀d
  have hC₀ : 1 ≤ C₀ := by rw [hC₀d]; linarith [le_max_left 1 C]
  have hCC₀ : C ≤ max 1 C := le_max_right _ _
  have hmapsR : Set.MapsTo toDistRun loopRunRegion loopDistRunRegion := fun r hr => toDist_mem hr
  have hmaps : Set.MapsTo toDist loopRegion loopDistRegion := fun x hx => toDist_mem hx
  refine ⟨fun x => 1 + 4 * (log (C₀ * loopG x) + 6 * log (x.Q + 4) + 24) / log x.A,
    fun r => tp (toDistRun r), fun r => ts (toDistRun r), fun r => tpr (toDistRun r),
    fun x => ho (toDist x), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine isBigO_on (245 + 6 * log C₀) fun x hx => ?_
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
    refine ⟨by positivity, ?_, hle.trans_eq (by rw [e])⟩
    exact add_nonneg zero_le_one (div_nonneg (Real.log_nonneg hZ) hlA.le)
  · refine (isBigO_comp_on hpref hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    show _ = r.Q ^ 5 * (r.L + r.Q) ^ 2 * r.R ^ 2
      * log ((r.L + r.Q) * r.A * r.R / (r.δ * r.ξ * r.pAP * r.s₀)) / r.ξ ^ 2
      * (1 / r.s₀ ^ 8 + r.A ^ 2 * r.Q ^ 4 / r.s₀ ^ 4)
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
        + log (r.Q / r.δ) / r.pAP)
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
    have hG1 : 1 ≤ loopG x := loopG_ge_one ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩
    obtain ⟨hsh0, hsh⟩ := shape_toDist_le ⟨hQ1, hA2, hδ, hδ2, hR, hξ, hξ1, hp, hp1, hs, hs1⟩ L
    have hc' : 1 / p₀Tol { toDist x with L } ≤ C₀ * (((L : ℝ) + x.Q) ^ 6 * loopG x) := by
      rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (one_div_pos.2 hpos')] at hc
      change _ ≤ C * |p₀Shape { toDist x with L }| at hc
      rw [abs_of_nonneg hsh0] at hc
      refine hc.trans ((mul_le_mul_of_nonneg_right hCC₀ hsh0).trans ?_)
      refine (mul_le_mul_of_nonneg_left hsh ?_).trans (le_of_eq ?_)
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

theorem loopClaimSimple_of_claim (h : LoopClaimUniform.{u, v, w}) : LoopClaimSimple.{u, v, w} := by
  obtain ⟨L₀, pref, suff, probes, heldOut, hL₀, hp, hs, hpr, hh, hw⟩ := h
  exact ⟨L₀, pref, suff, probes, heldOut, hL₀,
    hp.trans (isBigO_on 2 fun r hr => loop_pref_dom hr),
    hs.trans (isBigO_on 2 fun r hr => loop_suff_dom hr), hpr, hh, hw⟩

end OrthoDFA

end
