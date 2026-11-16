import RLStar.Proofs.TrieTolArith
import RLStar.Proofs.LoopReadsArith
import RLStar.Proofs.BigO

/-!
# The trie loop's costs against the claim's shapes

`trie_loop_tol` at `X = L + |Q|`, `ℓ = X + 4`, `c = ξ/2`, `ξ/4` and `η₀ = 1/2 − s/2`: each of its
bounds within a constant of its shape on `TC`. The logs are bounded through the polynomials under
them: `Y = X|Σ|/(δξs)`, `Z = Y/p`, `X|Σ|/δ` and `|Q| log Y/(δps)`.
-/

noncomputable section

namespace OrthoDFA

open Real

/-- The claims' region at `X = L + |Q|`. -/
structure TC (Q A X δ ξ p s : ℝ) : Prop where
  hQ : 1 ≤ Q
  hA : 2 ≤ A
  hQX : Q ≤ X
  hδ : 0 < δ
  hδ2 : δ ≤ 1 / 2
  hξ : 0 < ξ
  hξ1 : ξ ≤ 1
  hp : 0 < p
  hp1 : p ≤ 1
  hs : 0 < s
  hs2 : s ≤ 1 / 2

def shρ (X ξ : ℝ) : ℝ := 1 / ξ + log (X + 1)

def shι (Q X ξ : ℝ) : ℝ := (X + Q ^ 2) / ξ + Q ^ 2 * log (X + 1)

def shA (A X δ ξ s : ℝ) : ℝ := log (X * A / (δ * ξ * s))

def shZ (A X δ ξ p s : ℝ) : ℝ := log (X * A / (δ * ξ * p * s))

def shP (Q A X δ ξ p s : ℝ) : ℝ := log (Q * shA A X δ ξ s / (δ * p * s))

def shPref (Q A X δ ξ p s : ℝ) : ℝ :=
  Q ^ 3 * shι Q X ξ ^ 2 / s ^ 4 * shP Q A X δ ξ p s
    + Q ^ 2 * (X ^ 2 / (ξ ^ 2 * s ^ 4) + shι Q X ξ ^ 2) * shZ A X δ ξ p s

def shSuff (Q A X δ ξ p s : ℝ) : ℝ := Q * (shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p)

def shProbe (Q A X δ ξ : ℝ) : ℝ := Q ^ 3 * A * shρ X ξ * log (Q * A / δ)

def shFam (A X δ ξ s : ℝ) : ℝ := shA A X δ ξ s / s ^ 2

def shReads (Q A X δ ξ p s : ℝ) : ℝ :=
  shPref Q A X δ ξ p s * shSuff Q A X δ ξ p s / Q
    + shProbe Q A X δ ξ * shFam A X δ ξ s * Q * log (X + 1)
    + Q * shPref Q A X δ ξ p s * shFam A X δ ξ s * log (X + 1) * shρ X ξ

def shTol (Q A X δ ξ p s : ℝ) : ℝ :=
  shρ X ξ * (Q * shι Q X ξ ^ 3 * shZ A X δ ξ p s * (1 / s ^ 6 + Q ^ 2 / s ^ 4)
    + Q ^ 2 * shZ A X δ ξ p s / (p ^ 2 * s ^ 2) + Q ^ 2 * A * log (Q * A / δ))

lemma log_le_of_le_pow {x Y : ℝ} (n : ℕ) (hx : 0 < x) (h : x ≤ Y ^ n) :
    log x ≤ n * log Y :=
  log_le_mul_log_of_le_pow n hx h

namespace TC

variable {Q A X δ ξ p s : ℝ} (h : TC Q A X δ ξ p s)
include h

lemma X1 : 1 ≤ X := h.hQ.trans h.hQX

lemma iξ : 1 ≤ 1 / ξ := (le_div_iff₀ h.hξ).2 (by linarith [h.hξ1])

lemma is2 : 2 ≤ 1 / s := (le_div_iff₀ h.hs).2 (by linarith [h.hs2])

lemma ip : 1 ≤ 1 / p := (le_div_iff₀ h.hp).2 (by linarith [h.hp1])

lemma iδ : 2 ≤ 1 / δ := (le_div_iff₀ h.hδ).2 (by linarith [h.hδ2])

lemma lX0 : 0 ≤ log (X + 1) := Real.log_nonneg (by linarith [h.X1])

lemma lX_le : log (X + 1) ≤ X := by
  have := Real.log_le_sub_one_of_pos (show 0 < X + 1 by linarith [h.X1]); linarith

lemma ρ1 : 1 ≤ shρ X ξ := by unfold shρ; linarith [h.iξ, h.lX0]

lemma ι_eq : shι Q X ξ = X / ξ + Q ^ 2 * shρ X ξ := by unfold shι shρ; ring

lemma ρι : Q ^ 2 * shρ X ξ ≤ shι Q X ξ := by
  rw [h.ι_eq]; have : 0 ≤ X / ξ := div_nonneg (by linarith [h.X1]) h.hξ.le; linarith

lemma Xι : X / ξ ≤ shι Q X ξ := by
  rw [h.ι_eq]; have : 0 ≤ Q ^ 2 * shρ X ξ := by have := h.ρ1; positivity
  linarith

lemma ρle : shρ X ξ ≤ Q ^ 2 * shρ X ξ :=
  le_mul_of_one_le_left (by linarith [h.ρ1]) (one_le_pow₀ h.hQ)

lemma ι1 : 1 ≤ shι Q X ξ := h.ρ1.trans (h.ρle.trans h.ρι)

/-- `Y = X|Σ|/(δξs)`. -/
lemma Y_eq : X * A / (δ * ξ * s) = X * A * (1 / δ) * (1 / ξ) * (1 / s) := by
  have := h.hδ.ne'; have := h.hξ.ne'; have := h.hs.ne'; field_simp

lemma Y_ge : 8 * X ≤ X * A / (δ * ξ * s) ∧ 2 * A ≤ X * A / (δ * ξ * s)
    ∧ X ≤ X * A / (δ * ξ * s) ∧ A ≤ X * A / (δ * ξ * s) ∧ 1 / δ ≤ X * A / (δ * ξ * s)
    ∧ 1 / ξ ≤ X * A / (δ * ξ * s) ∧ 1 / s ≤ X * A / (δ * ξ * s)
    ∧ 8 ≤ X * A / (δ * ξ * s) ∧ Q * A / δ ≤ X * A / (δ * ξ * s) := by
  rw [h.Y_eq]
  have hX := h.X1; have hA := h.hA; have hδ := h.iδ; have hξ := h.iξ; have hs := h.is2
  set a := 1 / δ; set b := 1 / ξ; set c := 1 / s
  have hab : 2 ≤ a * b := by nlinarith [mul_nonneg (sub_nonneg.2 hδ) (sub_nonneg.2 hξ)]
  have k4 : 4 ≤ a * b * c := by nlinarith [mul_nonneg (sub_nonneg.2 hab) (sub_nonneg.2 hs)]
  have ka : a ≤ a * b * c := by
    have : a ≤ a * b := le_mul_of_one_le_right (by linarith) hξ
    have : a * b ≤ a * b * c := le_mul_of_one_le_right (by linarith) (by linarith)
    linarith
  have kb : b ≤ a * b * c := by
    have : b ≤ a * b := le_mul_of_one_le_left (by linarith) (by linarith)
    have : a * b ≤ a * b * c := le_mul_of_one_le_right (by linarith) (by linarith)
    linarith
  have kc : c ≤ a * b * c := le_mul_of_one_le_left (by linarith) (by linarith)
  have hXA : 2 * X ≤ X * A := by nlinarith
  have hA1 : A ≤ X * A := le_mul_of_one_le_left (by linarith) hX
  have e : X * A * a * b * c = (X * A) * (a * b * c) := by ring
  rw [e]
  have hXA1 : 1 ≤ X * A := by linarith
  have hm : ∀ u, 0 ≤ u → u ≤ X * A → 4 * u ≤ X * A * (a * b * c) := fun u hu hux =>
    (mul_le_mul_of_nonneg_left k4 hu).trans_eq' (by ring) |>.trans
      (mul_le_mul_of_nonneg_right hux (by linarith))
  have hm' : ∀ u, 0 ≤ u → u ≤ a * b * c → u ≤ X * A * (a * b * c) := fun u hu hux =>
    hux.trans (le_mul_of_one_le_left (by linarith) hXA1)
  have hQA : Q * A * a ≤ X * A * (a * b * c) := by
    have h1 : Q * A ≤ X * A := mul_le_mul_of_nonneg_right h.hQX (by linarith)
    exact mul_le_mul h1 ka (by linarith) (by linarith)
  have k8X : 8 * X ≤ X * A * (a * b * c) := by
    have := mul_le_mul hXA k4 (by norm_num) (by linarith); linarith
  have k2A : 2 * A ≤ X * A * (a * b * c) := by
    have := mul_le_mul hA1 k4 (by norm_num) (by linarith); linarith
  have k8 : 8 ≤ X * A * (a * b * c) := by
    have := mul_le_mul (show 2 ≤ X * A by linarith) k4 (by norm_num) (by linarith); linarith
  refine ⟨k8X, k2A, by linarith, by linarith, hm' a (by linarith) ka, hm' b (by linarith) kb,
    hm' c (by linarith) kc, k8, ?_⟩
  rw [show Q * A / δ = Q * A * a by ring]; exact hQA

lemma Z_eq : X * A / (δ * ξ * p * s) = X * A / (δ * ξ * s) * (1 / p) := by
  have := h.hδ.ne'; have := h.hξ.ne'; have := h.hs.ne'; have := h.hp.ne'; field_simp

lemma Z_ge : X * A / (δ * ξ * s) ≤ X * A / (δ * ξ * p * s) ∧ 1 / p ≤ X * A / (δ * ξ * p * s) := by
  rw [h.Z_eq]
  have := h.ip; have := h.Y_ge.2.2.2.2.2.2.2.1
  exact ⟨le_mul_of_one_le_right (by linarith) h.ip, by nlinarith⟩

lemma Yq_ge : 4 ≤ Q * A / δ ∧ 4 * Q ≤ Q * A / δ ∧ Q ≤ Q * A / δ ∧ A ≤ Q * A / δ
    ∧ 1 / δ ≤ Q * A / δ := by
  have e : Q * A / δ = Q * (A * (1 / δ)) := by ring
  rw [e]
  have hQ := h.hQ; have hA := h.hA; have hδ := h.iδ
  have k : 4 ≤ A * (1 / δ) := by nlinarith
  refine ⟨by nlinarith, by nlinarith, by nlinarith, by nlinarith, by nlinarith⟩

lemma S_bounds :
    1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1 ∧ Q * ((Q + 2) * A + 1) + 2 * A + 1 ≤ 6 * (Q ^ 2 * A)
    ∧ 1 ≤ Q ^ 2 * A ∧ Q ^ 2 * A ≤ (Q * A / δ) ^ 2 ∧ (Q + 2) * A * (Q + 1) ≤ 6 * (Q ^ 2 * A) := by
  have hQ := h.hQ; have hA := h.hA
  have hq2 : Q ≤ Q ^ 2 := by nlinarith
  have h1 : Q * A ≤ Q ^ 2 * A := mul_le_mul_of_nonneg_right hq2 (by linarith)
  have h2 : A ≤ Q ^ 2 * A := le_mul_of_one_le_left (by linarith) (one_le_pow₀ hQ)
  have hqa : 2 * Q ≤ Q * A := by nlinarith
  have h0 : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
  have hQA : Q * A ≤ Q * A / δ := le_div_self (by positivity) h.hδ (by linarith [h.hδ2])
  have hsq : (Q * A) ^ 2 ≤ (Q * A / δ) ^ 2 := pow_le_pow_left₀ (by positivity) hQA 2
  have hA2 : Q ^ 2 * A ≤ (Q * A) ^ 2 := by
    rw [mul_pow]; exact mul_le_mul_of_nonneg_left (by nlinarith) (by positivity)
  refine ⟨by linarith, by linarith, by linarith, hA2.trans hsq, by linarith⟩

lemma δ'1 : δ / (8 * (Q + 1)) ≤ 1 := by
  rw [div_le_one (by linarith [h.hQ])]; linarith [h.hQ, h.hδ2]

lemma log_ge_of {x : ℝ} (hx : 1 ≤ x) : 0 ≤ log (x / (δ / (8 * (Q + 1)))) :=
  Real.log_nonneg (by
    rw [le_div_iff₀ (by have := h.hδ; have := h.hQ; positivity)]
    linarith [h.δ'1])

lemma lQ0 : 0 ≤ log (Q / δ) := Real.log_nonneg (by
  rw [le_div_iff₀ h.hδ]; linarith [h.hQ, h.hδ2])

lemma Q_div : Q / δ ≤ Q * A / δ :=
  div_le_div_of_nonneg_right (le_mul_of_one_le_right (by linarith [h.hQ]) (by linarith [h.hA]))
    h.hδ.le

lemma lQA1 : 1 ≤ log (Q * A / δ) := one_le_log_of_three_le (by linarith [h.Yq_ge.1])

lemma lQ_le : log (Q / δ) ≤ log (Q * A / δ) := Real.log_le_log
  (div_pos (by linarith [h.hQ]) h.hδ)
  (div_le_div_of_nonneg_right (le_mul_of_one_le_right (by linarith [h.hQ]) (by linarith [h.hA]))
    h.hδ.le)

lemma lQA_le : log (Q * A / δ) ≤ shA A X δ ξ s :=
  Real.log_le_log (by linarith [h.Yq_ge.1]) h.Y_ge.2.2.2.2.2.2.2.2

lemma lA1 : 1 ≤ shA A X δ ξ s := h.lQA1.trans h.lQA_le

lemma lA_le : shA A X δ ξ s ≤ shZ A X δ ξ p s :=
  Real.log_le_log (by linarith [h.Y_ge.2.2.2.2.2.2.2.1]) h.Z_ge.1

lemma lZ1 : 1 ≤ shZ A X δ ξ p s := h.lA1.trans h.lA_le

lemma ps4 : 4 ≤ 1 / (p * s ^ 2) := by
  rw [le_div_iff₀ (by have := h.hp; have := h.hs; positivity)]
  have : s ^ 2 ≤ 1 / 4 := by nlinarith [h.hs2, h.hs]
  nlinarith [h.hp1, h.hp]

lemma lAps : 1 ≤ shA A X δ ξ s / (p * s ^ 2) := by
  rw [div_eq_mul_one_div]; nlinarith [h.lA1, h.ps4]

lemma lQp : log (Q / δ) / p ≤ shA A X δ ξ s / (p * s ^ 2) := by
  have hp := h.hp; have hs := h.hs
  rw [div_le_div_iff₀ hp (by positivity)]
  have hs1 : s ^ 2 ≤ 1 := by nlinarith [h.hs2]
  have k1 : log (Q / δ) * (p * s ^ 2) ≤ log (Q / δ) * p :=
    mul_le_mul_of_nonneg_left (mul_le_of_le_one_right hp.le hs1) h.lQ0
  have k2 : log (Q / δ) * p ≤ shA A X δ ξ s * p :=
    mul_le_mul_of_nonneg_right (h.lQ_le.trans h.lQA_le) hp.le
  linarith

lemma lA_Y : shA A X δ ξ s ≤ X * A / (δ * ξ * s) := by
  have := Real.log_le_sub_one_of_pos (show 0 < X * A / (δ * ξ * s) by
    linarith [h.Y_ge.2.2.2.2.2.2.2.1])
  unfold shA; linarith

lemma lZ_Z : shZ A X δ ξ p s ≤ X * A / (δ * ξ * p * s) := by
  have := Real.log_le_sub_one_of_pos (show 0 < X * A / (δ * ξ * p * s) by
    linarith [h.Y_ge.2.2.2.2.2.2.2.1, h.Z_ge.1])
  unfold shZ; linarith

lemma lQ_lZ : log (Q / δ) ≤ shZ A X δ ξ p s := (h.lQ_le.trans h.lQA_le).trans h.lA_le

lemma Yp_ge : 4 ≤ Q * shA A X δ ξ s / (δ * p * s) ∧ 1 / s ≤ Q * shA A X δ ξ s / (δ * p * s) := by
  have e : Q * shA A X δ ξ s / (δ * p * s) = (Q * shA A X δ ξ s) * ((1 / δ) * (1 / p)) * (1 / s) := by
    have := h.hδ.ne'; have := h.hs.ne'; have := h.hp.ne'; field_simp
  rw [e]
  have h1 : 1 ≤ Q * shA A X δ ξ s := one_le_mul_of_one_le_of_one_le h.hQ h.lA1
  have h2 : 2 ≤ (1 / δ) * (1 / p) := by nlinarith [h.iδ, h.ip]
  have h3 : 2 ≤ Q * shA A X δ ξ s * ((1 / δ) * (1 / p)) := by nlinarith
  have := h.is2
  constructor <;> nlinarith

lemma lP1 : 1 ≤ shP Q A X δ ξ p s := one_le_log_of_three_le (by linarith [h.Yp_ge.1])

lemma lP_le : shP Q A X δ ξ p s ≤ 2 * shZ A X δ ξ p s := by
  set Z := X * A / (δ * ξ * p * s)
  have hZ8 : 8 ≤ Z := h.Y_ge.2.2.2.2.2.2.2.1.trans h.Z_ge.1
  have hb : Q * shA A X δ ξ s / (δ * p * s) ≤ Z ^ 2 := by
    have e : Q * shA A X δ ξ s / (δ * p * s) = shA A X δ ξ s * (Q / (δ * p * s)) := by ring
    have hq : Q / (δ * p * s) ≤ Z := by
      have e2 : Z = (X * A / ξ) / (δ * p * s) := by
        show X * A / (δ * ξ * p * s) = _; have := h.hξ.ne'; field_simp
      rw [e2]
      have hd : 0 ≤ δ * p * s := by have := h.hδ; have := h.hp; have := h.hs; positivity
      refine div_le_div_of_nonneg_right ?_ hd
      have : X ≤ X * A / ξ := by
        rw [le_div_iff₀ h.hξ]; nlinarith [h.X1, h.hA, h.hξ1, h.hξ]
      linarith [h.hQX]
    have hq0 : 0 ≤ Q / (δ * p * s) := by
      have := h.hQ; have := h.hδ; have := h.hp; have := h.hs; positivity
    rw [e, sq]
    exact mul_le_mul (h.lA_Y.trans h.Z_ge.1) hq hq0 (by linarith)
  have := log_le_of_le_pow 2 (by linarith [h.Yp_ge.1]) hb
  unfold shP shZ; push_cast at this; linarith

/-- `shι ≤ 3 Y³`. -/
lemma ι_Y : shι Q X ξ ≤ 3 * (X * A / (δ * ξ * s)) ^ 3 := by
  obtain ⟨-, -, hXY, -, -, hξY, -, hY8, -⟩ := h.Y_ge
  set Y := X * A / (δ * ξ * s)
  have hQY : Q ≤ Y := h.hQX.trans hXY
  have hQ := h.hQ; have hX := h.X1
  have e : shι Q X ξ = (X + Q ^ 2) * (1 / ξ) + Q ^ 2 * log (X + 1) := by unfold shι; ring
  rw [e]
  have hξ0 : 0 ≤ 1 / ξ := by have := h.hξ; positivity
  have h1 : (X + Q ^ 2) * (1 / ξ) ≤ (Y + Y ^ 2) * Y :=
    mul_le_mul (add_le_add hXY (pow_le_pow_left₀ (by linarith) hQY 2)) hξY hξ0 (by positivity)
  have h2 : Q ^ 2 * log (X + 1) ≤ Y ^ 2 * Y :=
    mul_le_mul (pow_le_pow_left₀ (by linarith) hQY 2) (h.lX_le.trans hXY) h.lX0 (by positivity)
  have h3 : Y ≤ Y ^ 2 := by nlinarith
  have h4 : (Y + Y ^ 2) * Y + Y ^ 2 * Y ≤ 3 * Y ^ 3 := by nlinarith
  linarith

end TC

/-! ## The rates -/

section Rates

variable {Q A X δ ξ p s : ℝ} (h : TC Q A X δ ξ p s)
include h

lemma tc_Iε : 1 ≤ trieIε (X + 4) (ξ / 2) ∧ trieIε (X + 4) (ξ / 2) ≤ 45 * shρ X ξ := by
  have hξ := h.hξ; have := h.iξ; have hX := h.X1; have := hξ.ne'
  have e : trieIε (X + 4) (ξ / 2) = 16 * (1 / ξ) + 12 * (1 + log (X + 4)) := by
    unfold trieIε; field_simp; try ring
  have hl : log (X + 4) ≤ log 4 + log (X + 1) := by
    rw [← Real.log_mul (by norm_num) (by linarith)]
    exact Real.log_le_log (by linarith) (by linarith)
  have hl4 : log 4 < 1.3863 := by
    have := Real.log_two_lt_d9
    have e4 : log 4 = 2 * log 2 := by
      rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.log_pow]; norm_num
    linarith
  have hl0 : 0 ≤ log (X + 4) := Real.log_nonneg (by linarith)
  rw [e]
  unfold shρ
  constructor <;> nlinarith [h.lX0]

lemma tc_Iι : 1 ≤ trieIι (X + 4) Q (ξ / 2) (ξ / 4)
    ∧ trieIι (X + 4) Q (ξ / 2) (ξ / 4) ≤ 2 ^ 19 * shι Q X ξ := by
  obtain ⟨hε1, hε⟩ := tc_Iε h
  have hξ := h.hξ; have hX := h.X1; have hQ := h.hQ; have := hξ.ne'
  have e : trieIι (X + 4) Q (ξ / 2) (ξ / 4)
      = 416 * (X + 4) * (1 / ξ) + 4608 * Q * (Q + 1) * trieIε (X + 4) (ξ / 2) := by
    unfold trieIι; field_simp; try ring
  rw [e, h.ι_eq]
  have h1 : 416 * (X + 4) * (1 / ξ) ≤ 2080 * (X / ξ) := by
    rw [show X / ξ = X * (1 / ξ) by ring]
    have := h.iξ
    nlinarith
  have hQQ : 4608 * Q * (Q + 1) ≤ 9216 * Q ^ 2 := by nlinarith
  have h2 : 4608 * Q * (Q + 1) * trieIε (X + 4) (ξ / 2) ≤ 9216 * Q ^ 2 * (45 * shρ X ξ) :=
    mul_le_mul hQQ hε (by linarith) (by positivity)
  have h3 : 0 ≤ X / ξ := by positivity
  have h4 : 0 ≤ Q ^ 2 * shρ X ξ := by have := h.ρ1; positivity
  have h5 : 0 ≤ 416 * (X + 4) * (1 / ξ) := by positivity
  have h6 : 1 ≤ 4608 * Q * (Q + 1) * trieIε (X + 4) (ξ / 2) := by
    have : 1 ≤ 4608 * Q * (Q + 1) := by nlinarith
    nlinarith
  constructor <;> nlinarith

lemma tc_Iτ : trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
    ≤ 2 ^ 19 * (X / (ξ * s ^ 2) + shι Q X ξ) := by
  obtain ⟨-, hι⟩ := tc_Iι h
  have hξ := h.hξ; have hs := h.hs; have hX := h.X1
  have := hξ.ne'; have := hs.ne'
  have e : trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
      = 104 * (X + 4) / (ξ * s ^ 2) + trieIι (X + 4) Q (ξ / 2) (ξ / 4) := by
    have e1 : (10 / (ξ / 2) + 8 / (ξ / 4)) * (X + 4) / (2 * (1 / 2 - (1 / 2 - s / 2)) ^ 2)
        = 104 * (X + 4) / (ξ * s ^ 2) := by
      rw [show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]; field_simp; try ring
    unfold trieIτ; rw [e1]
  rw [e]
  have h1 : 104 * (X + 4) / (ξ * s ^ 2) ≤ 520 * (X / (ξ * s ^ 2)) := by
    rw [mul_div_assoc']
    exact div_le_div_of_nonneg_right (by linarith) (by positivity)
  have : 0 ≤ X / (ξ * s ^ 2) := by positivity
  nlinarith

lemma tc_Iτ1 : 1 ≤ trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2) := by
  have := (tc_Iι h).1
  unfold trieIτ
  have : 0 ≤ (10 / (ξ / 2) + 8 / (ξ / 4)) * (X + 4) / (2 * (1 / 2 - (1 / 2 - s / 2)) ^ 2) := by
    have := h.hξ; have := h.X1; have := h.hs
    have e : 1 / 2 - (1 / 2 - s / 2) = s / 2 := by ring
    rw [e]; positivity
  linarith

lemma tc_Iτ' : X / (ξ * s ^ 2) + shι Q X ξ ≤ 2 * shι Q X ξ / s ^ 2 := by
  have hs := h.hs
  have h1 : X / (ξ * s ^ 2) ≤ shι Q X ξ / s ^ 2 := by
    rw [show X / (ξ * s ^ 2) = X / ξ / s ^ 2 by rw [div_div]]
    exact div_le_div_of_nonneg_right h.Xι (by positivity)
  have h2 : shι Q X ξ ≤ shι Q X ξ / s ^ 2 := by
    rw [le_div_iff₀ (by positivity)]
    have : s ^ 2 ≤ 1 := by nlinarith [h.hs2]
    nlinarith [h.ι1]
  rw [show 2 * shι Q X ξ / s ^ 2 = shι Q X ξ / s ^ 2 + shι Q X ξ / s ^ 2 by ring]
  linarith

lemma tc_Ia : trieIa (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2) ≤ 2 ^ 20 * shι Q X ξ / s := by
  obtain ⟨-, hι⟩ := tc_Iι h
  have hξ := h.hξ; have hs := h.hs; have hX := h.X1
  have := hξ.ne'; have := hs.ne'
  have e : trieIa (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
      = 208 * (X + 4) / (ξ * s) + trieIι (X + 4) Q (ξ / 2) (ξ / 4) := by
    have e1 : 2 * (10 / (ξ / 2) + 8 / (ξ / 4)) * (X + 4) / (1 / 2 - (1 / 2 - s / 2))
        = 208 * (X + 4) / (ξ * s) := by
      rw [show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]; field_simp; try ring
    unfold trieIa; rw [e1]
  rw [e]
  have h1 : 208 * (X + 4) / (ξ * s) ≤ 1040 * (shι Q X ξ / s) := by
    calc 208 * (X + 4) / (ξ * s) ≤ 1040 * (X / ξ / s) := by
          rw [div_div, mul_div_assoc']
          exact div_le_div_of_nonneg_right (by linarith) (by positivity)
      _ ≤ 1040 * (shι Q X ξ / s) := by gcongr; exact h.Xι
  have h2 : shι Q X ξ ≤ shι Q X ξ / s / 2 := by
    rw [le_div_iff₀ (by norm_num), le_div_iff₀ hs]; nlinarith [h.ι1, h.hs2]
  rw [show 2 ^ 20 * shι Q X ξ / s = 2 ^ 20 * (shι Q X ξ / s) by ring]
  have : 0 ≤ shι Q X ξ / s := by have := h.ι1; positivity
  linarith

end Rates

/-! ## The pool, family and count -/

section Counts

variable {Q A X δ ξ p s : ℝ} (h : TC Q A X δ ξ p s)
include h

/-- The pool's and family's log term. -/
lemma tc_poolLog :
    0 ≤ max (log (20 * trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)))
        (log (8 * loopW₀ Q A δ / (δ / (8 * (Q + 1)))))
      ∧ max (log (20 * trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)))
        (log (8 * loopW₀ Q A δ / (δ / (8 * (Q + 1))))) ≤ 25 * shA A X δ ξ s := by
  obtain ⟨-, -, hXY, hAY, hδY, hξY, hsY, hY8, hqY⟩ := h.Y_ge
  set Y := X * A / (δ * ξ * s) with hYd
  have hQ := h.hQ; have hA := h.hA; have hX := h.X1; have hδ := h.hδ
  have hτ1 := tc_Iτ1 h
  refine ⟨le_max_of_le_left (Real.log_nonneg (by linarith)), max_le ?_ ?_⟩
  · have hb : 20 * trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2) ≤ Y ^ 13 := by
      have h1 := tc_Iτ h
      have h2 : X / (ξ * s ^ 2) ≤ Y ^ 4 := by
        rw [show X / (ξ * s ^ 2) = X * (1 / ξ) * (1 / s) ^ 2 by field_simp]
        calc X * (1 / ξ) * (1 / s) ^ 2 ≤ Y * Y * Y ^ 2 := by
              gcongr
              · have := h.iξ; linarith
              · have := h.is2; linarith
          _ = Y ^ 4 := by ring
      have h3 : shι Q X ξ ≤ 3 * Y ^ 4 := by
        have := h.ι_Y; have : Y ^ 3 ≤ Y ^ 4 := pow_le_pow_right₀ (by linarith) (by norm_num)
        linarith
      have h9 : (2 : ℝ) ^ 26 ≤ Y ^ 9 :=
        (show (2 : ℝ) ^ 26 ≤ 8 ^ 9 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hY8 9)
      have hY4 : 0 ≤ Y ^ 4 := by positivity
      calc 20 * trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
          ≤ 2 ^ 26 * Y ^ 4 := by nlinarith
        _ ≤ Y ^ 9 * Y ^ 4 := mul_le_mul_of_nonneg_right h9 hY4
        _ = Y ^ 13 := by ring
    have := log_le_of_le_pow 13 (by linarith) hb
    have hYl : shA A X δ ξ s = log Y := rfl
    push_cast at this; linarith [h.lA1]
  · obtain ⟨hq4, h4q, hQq, hAq, hδq⟩ := h.Yq_ge
    obtain ⟨hS1, hS6, hQA1, hQAq, hQ6⟩ := h.S_bounds
    set Yq := Q * A / δ with hYqd
    set S := Q * ((Q + 2) * A + 1) + 2 * A + 1 with hSd
    have hSq : S ≤ 6 * Yq ^ 2 := by linarith
    have hS1q : S + 1 ≤ 7 * Yq ^ 2 := by linarith
    have hδ' : 8 * (Q + 1) / δ ≤ 16 * Yq := by
      calc 8 * (Q + 1) / δ ≤ 16 * Q / δ := div_le_div_of_nonneg_right (by linarith) hδ.le
        _ = 16 * (Q / δ) := by ring
        _ ≤ 16 * Yq := by have := h.Q_div; rw [hYqd]; linarith
    have hS0 : 0 < 8 * S * (S + 1) / (δ / (8 * (Q + 1))) := by
      have : 0 < S := by linarith
      positivity
    have hl0 : 0 ≤ log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) :=
      h.log_ge_of (by nlinarith)
    have hlog : log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) ≤ 2 ^ 13 * Yq ^ 6 := by
      have e : 8 * S * (S + 1) / (δ / (8 * (Q + 1))) = 8 * S * (S + 1) * (8 * (Q + 1) / δ) := by
        field_simp
      have hb : 8 * S * (S + 1) * (8 * (Q + 1) / δ) ≤ 5376 * Yq ^ 5 := by
        calc 8 * S * (S + 1) * (8 * (Q + 1) / δ)
            ≤ 8 * (6 * Yq ^ 2) * (7 * Yq ^ 2) * (16 * Yq) := by gcongr
          _ = 5376 * Yq ^ 5 := by ring
      have hl := Real.log_le_sub_one_of_pos hS0
      rw [e] at hl ⊢
      have h56 : Yq ^ 5 ≤ Yq ^ 6 := pow_le_pow_right₀ (by linarith) (by norm_num)
      have : 0 ≤ Yq ^ 5 := by positivity
      linarith
    have hW : loopW₀ Q A δ ≤ 2 ^ 24 * Yq ^ 8 := by
      have he : Real.exp 2 ≤ 8 := by
        have := Real.exp_one_lt_d9
        have e : Real.exp 2 = Real.exp 1 ^ 2 := by rw [← Real.exp_nat_mul]; norm_num
        rw [e]; nlinarith only [this, Real.exp_pos 1]
      have h1 : S * log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) ≤ (6 * Yq ^ 2) * (2 ^ 13 * Yq ^ 6) :=
        mul_le_mul hSq hlog hl0 (by positivity)
      have hY8 : 1 ≤ Yq ^ 2 := by nlinarith only [hq4]
      have hY82 : Yq ^ 2 ≤ Yq ^ 8 := pow_le_pow_right₀ (by linarith) (by norm_num)
      have hY81 : 1 ≤ Yq ^ 8 := by linarith
      have hin : 16 / 5 * S * log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) + 3 + (Q + 2) * A * (Q + 1)
          ≤ 2 ^ 19 * Yq ^ 8 := by linarith only [h1, hQ6, hQAq, hY82, hY81]
      have hin0 : 0 ≤ 16 / 5 * S * log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) + 3
          + (Q + 2) * A * (Q + 1) := by
        have : 0 ≤ S := by linarith
        positivity
      have := mul_le_mul he hin hin0 (by norm_num)
      show 2 * Real.exp 2 * (16 / 5 * S * log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) + 3
        + (Q + 2) * A * (Q + 1)) + 1 ≤ 2 ^ 24 * Yq ^ 8
      linarith only [this, hY81]
    have hb : 8 * loopW₀ Q A δ / (δ / (8 * (Q + 1))) ≤ Yq ^ 25 := by
      have e : 8 * loopW₀ Q A δ / (δ / (8 * (Q + 1))) = 8 * loopW₀ Q A δ * (8 * (Q + 1) / δ) := by
        field_simp
      rw [e]
      have h16 : (2 : ℝ) ^ 31 ≤ Yq ^ 16 :=
        (show (2 : ℝ) ^ 31 ≤ 4 ^ 16 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hq4 16)
      calc 8 * loopW₀ Q A δ * (8 * (Q + 1) / δ) ≤ 8 * (2 ^ 24 * Yq ^ 8) * (16 * Yq) := by
            gcongr
        _ = 2 ^ 31 * Yq ^ 9 := by ring
        _ ≤ Yq ^ 16 * Yq ^ 9 := mul_le_mul_of_nonneg_right h16 (by positivity)
        _ = Yq ^ 25 := by ring
    have hpos : 0 < 8 * loopW₀ Q A δ / (δ / (8 * (Q + 1))) := by
      have : 1 ≤ loopW₀ Q A δ := by
        show 1 ≤ 2 * Real.exp 2 * (16 / 5 * S * log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) + 3
          + (Q + 2) * A * (Q + 1)) + 1
        have : 0 ≤ 16 / 5 * S * log (8 * S * (S + 1) / (δ / (8 * (Q + 1)))) + 3
            + (Q + 2) * A * (Q + 1) := by
          have : 0 ≤ S := by linarith
          positivity
        have := mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) (Real.exp_pos 2).le) this
        linarith
      positivity
    have := log_le_of_le_pow 25 hpos hb
    have hl := h.lQA_le
    push_cast at this
    linarith

lemma tc_pool : 1 ≤ triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
    ∧ triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      ≤ 2 ^ 13 * (shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p) := by
  obtain ⟨hm0, hm⟩ := tc_poolLog h
  have hs := h.hs; have hp := h.hp; have hδ := h.hδ; have hQ := h.hQ
  have hlA := h.lA1
  set m := max (log (20 * trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)))
    (log (8 * loopW₀ Q A δ / (δ / (8 * (Q + 1))))) with hmd
  have := hs.ne'; have := hp.ne'; have := hδ.ne'
  set lq := log (48 * (Q + 1) / δ) with hlqd
  have e : triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      = (200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) + 28) * (1 / p) + 120 * lq * (1 / p) := by
    have e3 : (3 : ℝ) / (δ / (8 * (Q + 1)) / 2) = 48 * (Q + 1) / δ := by field_simp; try ring
    unfold triePool
    simp only []
    rw [← hmd, e3, ← hlqd, show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]
    field_simp
    try ring
  rw [e]
  have hlq : log (48 * (Q + 1) / δ) ≤ 8 * log (Q / δ) := by
    have hq2 : 2 ≤ Q / δ := by rw [le_div_iff₀ hδ]; linarith [h.hδ2]
    have hb : 48 * (Q + 1) / δ ≤ (Q / δ) ^ 8 := by
      have e2 : 48 * (Q + 1) / δ ≤ 96 * (Q / δ) := by
        rw [show 96 * (Q / δ) = 96 * Q / δ by ring]
        exact div_le_div_of_nonneg_right (by linarith) hδ.le
      have : (96 : ℝ) ≤ (Q / δ) ^ 7 :=
        (show (96 : ℝ) ≤ 2 ^ 7 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hq2 7)
      calc 48 * (Q + 1) / δ ≤ 96 * (Q / δ) := e2
        _ ≤ (Q / δ) ^ 7 * (Q / δ) := mul_le_mul_of_nonneg_right this (by linarith)
        _ = (Q / δ) ^ 8 := by ring
    have := log_le_of_le_pow 8 (by positivity) hb
    push_cast at this; linarith
  have his := h.is2; have hip := h.ip
  have hs2 : 4 ≤ 1 / s ^ 2 := by rw [le_div_iff₀ (by positivity)]; nlinarith [h.hs2]
  have hlq0 : 0 ≤ log (48 * (Q + 1) / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith [h.hδ2])
  have e3 : shA A X δ ξ s / (p * s ^ 2) = shA A X δ ξ s * (1 / s ^ 2) * (1 / p) := by
    field_simp
  have e4 : log (Q / δ) / p = log (Q / δ) * (1 / p) := by ring
  rw [e3, e4]
  have hmA : 200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) + 28 ≤ 5627 * (shA A X δ ξ s * (1 / s ^ 2)) := by
    have h1 : 40 * (1 / s) ≤ 20 * (1 / s ^ 2) := by
      rw [show 1 / s ^ 2 = (1 / s) * (1 / s) by ring]; nlinarith
    have h2 : 200 * (m + 3) ≤ 5600 * shA A X δ ξ s := by linarith
    have h3 : 200 * (m + 3) * (1 / s ^ 2) ≤ 5600 * shA A X δ ξ s * (1 / s ^ 2) :=
      mul_le_mul_of_nonneg_right h2 (by positivity)
    have h4 : (1 / s ^ 2) ≤ shA A X δ ξ s * (1 / s ^ 2) :=
      le_mul_of_one_le_left (by positivity) hlA
    nlinarith
  have hpos : 0 ≤ 200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) + 28 := by positivity
  constructor
  · have : 28 ≤ 200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) + 28 := by
      have : 0 ≤ 200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) := by positivity
      linarith
    have h5 : 28 * 1 ≤ (200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) + 28) * (1 / p) :=
      mul_le_mul this hip (by norm_num) hpos
    have : 0 ≤ 120 * log (48 * (Q + 1) / δ) * (1 / p) := by positivity
    linarith
  · have h5 := mul_le_mul_of_nonneg_right hmA (show 0 ≤ 1 / p by positivity)
    have h6 : 120 * log (48 * (Q + 1) / δ) * (1 / p) ≤ 960 * (log (Q / δ) * (1 / p)) := by
      have := mul_le_mul_of_nonneg_right hlq (show 0 ≤ 1 / p by positivity)
      nlinarith
    have h7 : 0 ≤ shA A X δ ξ s * (1 / s ^ 2) * (1 / p) := by positivity
    have h8 : 0 ≤ log (Q / δ) * (1 / p) := mul_nonneg h.lQ0 (by positivity)
    nlinarith

lemma tc_pool_le : triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 2
      ≤ 2 ^ 14 * (shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p)
    ∧ 1 ≤ shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p
    ∧ shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p ≤ 2 * (shZ A X δ ξ p s / (p * s ^ 2)) := by
  obtain ⟨-, hP⟩ := tc_pool h
  have hs := h.hs; have hp := h.hp
  have hps : 4 ≤ 1 / (p * s ^ 2) := by
    rw [le_div_iff₀ (by positivity)]
    have : s ^ 2 ≤ 1 / 4 := by nlinarith [h.hs2]
    nlinarith [h.hp1]
  have h1 : 1 ≤ shA A X δ ξ s / (p * s ^ 2) := by
    rw [div_eq_mul_one_div]; nlinarith [h.lA1]
  have h2 : 0 ≤ log (Q / δ) / p := div_nonneg h.lQ0 hp.le
  have h3 : log (Q / δ) / p ≤ shZ A X δ ξ p s / (p * s ^ 2) := by
    rw [div_le_div_iff₀ hp (by positivity)]
    have hs1 : s ^ 2 ≤ 1 := by nlinarith [h.hs2]
    have k1 : log (Q / δ) * (p * s ^ 2) ≤ log (Q / δ) * p :=
      mul_le_mul_of_nonneg_left (mul_le_of_le_one_right hp.le hs1) h.lQ0
    have k2 : log (Q / δ) * p ≤ shZ A X δ ξ p s * p := mul_le_mul_of_nonneg_right h.lQ_lZ hp.le
    linarith
  have h4 : shA A X δ ξ s / (p * s ^ 2) ≤ shZ A X δ ξ p s / (p * s ^ 2) :=
    div_le_div_of_nonneg_right h.lA_le (by positivity)
  refine ⟨by nlinarith, by linarith, by linarith⟩

lemma tc_fam : 0 ≤ trieFam (X + 4) Q A δ (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
    ∧ trieFam (X + 4) Q A δ (ξ / 2) (ξ / 4) (1 / 2 - s / 2) ≤ 2 ^ 11 * shFam A X δ ξ s := by
  obtain ⟨hm0, hm⟩ := tc_poolLog h
  have hs := h.hs
  set m := max (log (20 * trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2)))
    (log (8 * loopW₀ Q A δ / (δ / (8 * (Q + 1))))) with hmd
  have := hs.ne'
  have e : trieFam (X + 4) Q A δ (ξ / 2) (ξ / 4) (1 / 2 - s / 2)
      = (200 * (m + 3) * (1 / s ^ 2) + 40 * (1 / s) + 28) / 4 := by
    unfold trieFam
    simp only []
    rw [← hmd, show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]
    field_simp
    try ring
  rw [e]
  have his := h.is2
  have hs2 : 4 ≤ 1 / s ^ 2 := by rw [le_div_iff₀ (by positivity)]; nlinarith [h.hs2]
  have hlA := h.lA1
  have h1 : 40 * (1 / s) ≤ 20 * (1 / s ^ 2) := by
    rw [show 1 / s ^ 2 = (1 / s) * (1 / s) by ring]; nlinarith
  have h3 : 200 * (m + 3) * (1 / s ^ 2) ≤ 5600 * shA A X δ ξ s * (1 / s ^ 2) :=
    mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  have h4 : (1 / s ^ 2) ≤ shA A X δ ξ s * (1 / s ^ 2) :=
    le_mul_of_one_le_left (by positivity) hlA
  have e2 : shFam A X δ ξ s = shA A X δ ξ s * (1 / s ^ 2) := by unfold shFam; ring
  rw [e2]
  constructor
  · positivity
  · nlinarith

/-- `C ≤ 2⁷⁰ (|Q| ι² log Yp/s⁴ + τ² log Z)`. -/
lemma tc_count : 0 ≤ trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
    ∧ trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      ≤ 2 ^ 70 * (Q * shι Q X ξ ^ 2 * shP Q A X δ ξ p s / s ^ 4
        + (X / (ξ * s ^ 2) + shι Q X ξ) ^ 2 * shZ A X δ ξ p s) := by
  obtain ⟨hP1, hP⟩ := tc_pool h
  obtain ⟨hP2, hPs1, hPsZ⟩ := tc_pool_le h
  obtain ⟨hι1, hι⟩ := tc_Iι h
  have hτ := tc_Iτ h
  have hτ1 := tc_Iτ1 h
  have hs := h.hs; have hp := h.hp; have hδ := h.hδ; have hQ := h.hQ; have hξ := h.hξ
  set P := triePool (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) with hPd
  set Iι := trieIι (X + 4) Q (ξ / 2) (ξ / 4) with hIιd
  set Iτ := trieIτ (X + 4) Q (ξ / 2) (ξ / 4) (1 / 2 - s / 2) with hIτd
  set ι := shι Q X ξ
  set τ := X / (ξ * s ^ 2) + ι with hτd
  have hι0 : 1 ≤ ι := h.ι1
  have hτ0 : 0 ≤ τ := by have : 0 ≤ X / (ξ * s ^ 2) := by have := h.X1; positivity
                         linarith
  have := hs.ne'; have := hp.ne'; have := hδ.ne'
  have e : trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2)
      = 2 ^ 26 * (Q + 2) * Iι ^ 2 * log (1024 * (Q + 1) * (P + 1) / δ) * (1 / s ^ 4)
        + 2 ^ 27 * Iτ ^ 2 * log (1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p)) := by
    have e1 : 128 * (P + 1) / (δ / (8 * (Q + 1))) = 1024 * (Q + 1) * (P + 1) / δ := by
      field_simp; try ring
    have e2 : 16 * (Q + 4) * P * Iτ / ((δ / (8 * (Q + 1))) ^ 2 * p)
        = 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p) := by field_simp; try ring
    unfold trieCount
    simp only []
    rw [← hPd, ← hIιd, ← hIτd, e1, e2, show (1 : ℝ) / 2 - (1 / 2 - s / 2) = s / 2 by ring]
    generalize log (1024 * (Q + 1) * (P + 1) / δ) = l1
    generalize log (1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p)) = l2
    field_simp
    try ring
  -- the first log, against `Yp = |Q| log Y/(δps)`
  set Yp := Q * shA A X δ ξ s / (δ * p * s) with hYpd
  obtain ⟨hYp4, hsYp⟩ := h.Yp_ge
  have hl1 : log (1024 * (Q + 1) * (P + 1) / δ) ≤ 15 * shP Q A X δ ξ p s := by
    have hb : 1024 * (Q + 1) * (P + 1) / δ ≤ Yp ^ 15 := by
      have hP1' : P + 1 ≤ 2 ^ 15 * (shA A X δ ξ s / (p * s ^ 2)) := by
        have := h.lQp; have := h.lAps
        linarith
      have e2 : 1024 * (Q + 1) * (P + 1) / δ ≤ 2 ^ 26 * (Yp * (1 / s)) := by
        have h1 : 1024 * (Q + 1) * (P + 1) / δ
            ≤ 1024 * (2 * Q) * (2 ^ 15 * (shA A X δ ξ s / (p * s ^ 2))) / δ := by
          gcongr
          · linarith
        refine h1.trans (le_of_eq ?_)
        rw [hYpd]; field_simp; try ring
      have h25 : (2 : ℝ) ^ 26 ≤ Yp ^ 13 :=
        (show (2 : ℝ) ^ 26 ≤ 4 ^ 13 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hYp4 13)
      calc 1024 * (Q + 1) * (P + 1) / δ ≤ 2 ^ 26 * (Yp * (1 / s)) := e2
        _ ≤ Yp ^ 13 * (Yp * Yp) := by gcongr
        _ = Yp ^ 15 := by ring
    have := log_le_of_le_pow 15 (by positivity) hb
    unfold shP; push_cast at this; exact this
  -- the second, against `Z`
  obtain ⟨-, -, hXY, hAY, hδY, hξY, hsY, hY8, -⟩ := h.Y_ge
  obtain ⟨hYZ, hpZ⟩ := h.Z_ge
  set Z := X * A / (δ * ξ * p * s) with hZd
  have hZ16 : 8 ≤ Z := hY8.trans hYZ
  have hZ1 : 1 ≤ Z := by linarith
  have hl2 : log (1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p)) ≤ 31 * shZ A X δ ξ p s := by
    have hQZ : Q ≤ Z := (h.hQX.trans hXY).trans hYZ
    have hδZ : 1 / δ ≤ Z := hδY.trans hYZ
    have hsZ : 1 / s ≤ Z := hsY.trans hYZ
    have hPZ : P ≤ 2 ^ 14 * Z ^ 4 := by
      have : shA A X δ ξ s / (p * s ^ 2) + log (Q / δ) / p ≤ 2 * Z ^ 4 := by
        have e3 : shZ A X δ ξ p s / (p * s ^ 2) = shZ A X δ ξ p s * (1 / p) * (1 / s) ^ 2 := by
          field_simp
        have : shZ A X δ ξ p s * (1 / p) * (1 / s) ^ 2 ≤ Z * Z * Z ^ 2 := by
          have hs0 : 0 ≤ 1 / s := by positivity
          have hp0 : 0 ≤ 1 / p := by positivity
          have k := mul_le_mul h.lZ_Z hpZ hp0 (by linarith)
          exact mul_le_mul k (pow_le_pow_left₀ hs0 hsZ 2) (by positivity) (by positivity)
        linarith only [hPsZ, e3, this]
      linarith only [hP, this]
    have hIτZ : Iτ ≤ 2 ^ 21 * Z ^ 4 := by
      have h2 : X / (ξ * s ^ 2) ≤ Z ^ 4 := by
        rw [show X / (ξ * s ^ 2) = X * (1 / ξ) * (1 / s) ^ 2 by field_simp]
        have hXZ : X ≤ Z := hXY.trans hYZ
        have hξZ : 1 / ξ ≤ Z := hξY.trans hYZ
        have k := mul_le_mul hXZ hξZ (by have := h.hξ; positivity) (by linarith)
        calc X * (1 / ξ) * (1 / s) ^ 2 ≤ Z * Z * Z ^ 2 :=
              mul_le_mul k (pow_le_pow_left₀ (by positivity) hsZ 2) (by positivity) (by positivity)
          _ = Z ^ 4 := by ring
      have h3 : ι ≤ 3 * Z ^ 4 := by
        have := h.ι_Y
        have : (X * A / (δ * ξ * s)) ^ 3 ≤ Z ^ 3 := pow_le_pow_left₀ (by linarith) hYZ 3
        have : Z ^ 3 ≤ Z ^ 4 := pow_le_pow_right₀ hZ1 (by norm_num)
        linarith
      linarith only [hτ, h2, h3]
    have hb : 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p) ≤ Z ^ 31 := by
      have e3 : 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p)
          = 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ * ((1 / δ) ^ 2 * (1 / p)) := by
        field_simp
      rw [e3]
      have hP0 : 0 ≤ P := by linarith
      have hIτ0 : 0 ≤ Iτ := by linarith
      have hQ4Z : Q + 4 ≤ 5 * Z := by linarith
      have hQ1Z : Q + 1 ≤ 2 * Z := by linarith
      calc 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ * ((1 / δ) ^ 2 * (1 / p))
          ≤ 1024 * (5 * Z) * (2 * Z) ^ 2 * (2 ^ 14 * Z ^ 4) * (2 ^ 21 * Z ^ 4) * (Z ^ 2 * Z) := by
            gcongr
        _ = 5 * 2 ^ 47 * Z ^ 14 := by ring
        _ ≤ 8 ^ 17 * Z ^ 14 := by gcongr; norm_num
        _ ≤ Z ^ 17 * Z ^ 14 := by gcongr
        _ ≤ Z ^ 31 := by rw [← pow_add]
    have hpos : 0 < 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p) := by
      have : 0 < P := by linarith
      have : 0 < Iτ := by linarith
      positivity
    have := log_le_of_le_pow 31 hpos hb
    unfold shZ; push_cast at this; exact this
  have hl1' : 0 ≤ log (1024 * (Q + 1) * (P + 1) / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ]; nlinarith only [h.hδ2, hP1, hQ])
  have hl2' : 0 ≤ log (1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p)) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ (by positivity)]
    have h1 : δ ^ 2 * p ≤ 1 := by
      have : δ ^ 2 ≤ 1 := pow_le_one₀ hδ.le (by linarith [h.hδ2])
      nlinarith only [this, h.hp1, hp, sq_nonneg δ]
    have h2 : 1 ≤ 1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ := by
      have h3 : 1 ≤ (Q + 4) * (Q + 1) ^ 2 := by nlinarith only [hQ]
      have h4 : 1 ≤ P * Iτ := one_le_mul_of_one_le_of_one_le hP1 hτ1
      have h5 := one_le_mul_of_one_le_of_one_le h3 h4
      linarith only [h5]
    linarith only [h1, h2]
  rw [e]
  constructor
  · positivity
  · have k1 : 2 ^ 26 * (Q + 2) * Iι ^ 2 * log (1024 * (Q + 1) * (P + 1) / δ) * (1 / s ^ 4)
        ≤ 2 ^ 26 * (3 * Q) * (2 ^ 19 * ι) ^ 2 * (15 * shP Q A X δ ξ p s) * (1 / s ^ 4) := by
      gcongr
      linarith
    have k2 : 2 ^ 27 * Iτ ^ 2 * log (1024 * (Q + 4) * (Q + 1) ^ 2 * P * Iτ / (δ ^ 2 * p))
        ≤ 2 ^ 27 * (2 ^ 19 * τ) ^ 2 * (31 * shZ A X δ ξ p s) := by
      gcongr
    have e1 : 2 ^ 26 * (3 * Q) * (2 ^ 19 * ι) ^ 2 * (15 * shP Q A X δ ξ p s) * (1 / s ^ 4)
        = 45 * 2 ^ 64 * (Q * ι ^ 2 * shP Q A X δ ξ p s / s ^ 4) := by field_simp; try ring
    have e2 : 2 ^ 27 * (2 ^ 19 * τ) ^ 2 * (31 * shZ A X δ ξ p s)
        = 31 * 2 ^ 65 * (τ ^ 2 * shZ A X δ ξ p s) := by ring
    have p1 : 0 ≤ Q * ι ^ 2 * shP Q A X δ ξ p s / s ^ 4 := by have := h.lP1; positivity
    have p2 : 0 ≤ τ ^ 2 * shZ A X δ ξ p s := by have := h.lZ1; positivity
    rw [e1] at k1; rw [e2] at k2
    linarith only [k1, k2, p1, p2]

lemma tc_count_le : trieCount (X + 4) Q A δ (ξ / 2) (ξ / 4) p (1 / 2 - s / 2) + 1
    ≤ 2 ^ 73 * (Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4) := by
  obtain ⟨-, hC⟩ := tc_count h
  have hs := h.hs; have hQ := h.hQ
  have hτ := tc_Iτ' h
  have hτ0 : 0 ≤ X / (ξ * s ^ 2) + shι Q X ξ := by
    have : 0 ≤ X / (ξ * s ^ 2) := by have := h.X1; have := h.hξ; positivity
    linarith [h.ι1]
  have hτ2 : (X / (ξ * s ^ 2) + shι Q X ξ) ^ 2 ≤ 4 * (Q * shι Q X ξ ^ 2 / s ^ 4) := by
    have := pow_le_pow_left₀ hτ0 hτ 2
    have e : (2 * shι Q X ξ / s ^ 2) ^ 2 = 4 * (shι Q X ξ ^ 2 / s ^ 4) := by ring
    rw [e] at this
    have : shι Q X ξ ^ 2 / s ^ 4 ≤ Q * shι Q X ξ ^ 2 / s ^ 4 := by
      rw [mul_div_assoc]; exact le_mul_of_one_le_left (by positivity) hQ
    linarith
  have hZ := h.lZ1
  have hPZ := h.lP_le
  have hb : 1 ≤ Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4 := by
    rw [le_div_iff₀ (by positivity)]
    have : s ^ 4 ≤ 1 := pow_le_one₀ hs.le (by linarith [h.hs2])
    have : 1 ≤ Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s := by
      have := one_le_pow₀ (n := 2) h.ι1
      have := one_le_mul_of_one_le_of_one_le hQ this
      exact one_le_mul_of_one_le_of_one_le this hZ
    linarith
  have k1 : Q * shι Q X ξ ^ 2 * shP Q A X δ ξ p s / s ^ 4
      ≤ 2 * (Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4) := by
    rw [show 2 * (Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4)
      = Q * shι Q X ξ ^ 2 * (2 * shZ A X δ ξ p s) / s ^ 4 by ring]
    gcongr
  have k2 : (X / (ξ * s ^ 2) + shι Q X ξ) ^ 2 * shZ A X δ ξ p s
      ≤ 4 * (Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4) := by
    have := mul_le_mul_of_nonneg_right hτ2 (by linarith : 0 ≤ shZ A X δ ξ p s)
    rw [show 4 * (Q * shι Q X ξ ^ 2 * shZ A X δ ξ p s / s ^ 4)
      = 4 * (Q * shι Q X ξ ^ 2 / s ^ 4) * shZ A X δ ξ p s by ring]
    exact this
  linarith only [hC, k1, k2, hb]

lemma tc_budget : 0 ≤ trieBudget (X + 4) Q A δ (ξ / 2)
    ∧ trieBudget (X + 4) Q A δ (ξ / 2) ≤ 2 ^ 17 * (Q ^ 2 * A * shρ X ξ * log (Q * A / δ)) := by
  obtain ⟨hε1, hε⟩ := tc_Iε h
  have hQ := h.hQ; have hA := h.hA; have hδ := h.hδ
  obtain ⟨hq4, -, hQq, hAq, hδq⟩ := h.Yq_ge
  obtain ⟨hS1, hS6, h1, hQA, -⟩ := h.S_bounds
  set Yq := Q * A / δ with hYqd
  set S := Q * ((Q + 2) * A + 1) + 2 * A + 1 with hSd
  have hS7 : S + 1 ≤ 7 * (Q ^ 2 * A) := by linarith
  have hl : log (16 * S * (S + 1) / (δ / (8 * (Q + 1)))) ≤ 13 * log Yq := by
    have hb : 16 * S * (S + 1) / (δ / (8 * (Q + 1))) ≤ Yq ^ 13 := by
      have e : 16 * S * (S + 1) / (δ / (8 * (Q + 1))) = 128 * (Q + 1) * S * (S + 1) * (1 / δ) := by
        field_simp; try ring
      rw [e]
      have hQ2 : Q + 1 ≤ 2 * Yq := by linarith
      have hS6' : S ≤ 6 * Yq ^ 2 := by linarith
      have hS7' : S + 1 ≤ 7 * Yq ^ 2 := by linarith
      have h7 : (2 : ℝ) ^ 14 ≤ Yq ^ 7 :=
        (show (2 : ℝ) ^ 14 ≤ 4 ^ 7 by norm_num).trans (pow_le_pow_left₀ (by norm_num) hq4 7)
      calc 128 * (Q + 1) * S * (S + 1) * (1 / δ)
          ≤ 128 * (2 * Yq) * (6 * Yq ^ 2) * (7 * Yq ^ 2) * Yq := by
            gcongr
        _ = 10752 * Yq ^ 6 := by ring
        _ ≤ 2 ^ 14 * Yq ^ 6 := by gcongr; norm_num
        _ ≤ Yq ^ 7 * Yq ^ 6 := by gcongr
        _ = Yq ^ 13 := by ring
    have := log_le_of_le_pow 13 (by positivity) hb
    push_cast at this; exact this
  have hl0 : 0 ≤ log (16 * S * (S + 1) / (δ / (8 * (Q + 1)))) :=
    h.log_ge_of (by nlinarith only [hS1])
  have e : trieBudget (X + 4) Q A δ (ξ / 2)
      = 36 * S * trieIε (X + 4) (ξ / 2) * log (16 * S * (S + 1) / (δ / (8 * (Q + 1)))) + S + 1 :=
    rfl
  rw [e]
  have hlQA := h.lQA1
  have hρ := h.ρ1
  constructor
  · positivity
  · have k : 36 * S * trieIε (X + 4) (ξ / 2) * log (16 * S * (S + 1) / (δ / (8 * (Q + 1))))
        ≤ 36 * (6 * (Q ^ 2 * A)) * (45 * shρ X ξ) * (13 * log Yq) := by
      gcongr
    have h2 := one_le_mul_of_one_le_of_one_le hρ hlQA
    have : Q ^ 2 * A ≤ Q ^ 2 * A * shρ X ξ * log Yq := by
      rw [mul_assoc]; exact le_mul_of_one_le_right (by linarith) h2
    have e3 : 36 * (6 * (Q ^ 2 * A)) * (45 * shρ X ξ) * (13 * log Yq)
        = 126360 * (Q ^ 2 * A * shρ X ξ * log Yq) := by ring
    linarith only [k, e3, this, hS7, h1]

end Counts

end OrthoDFA

end
