import OrthoDFA.Claims.Levels
import OrthoDFA.Proofs.BigO
import OrthoDFA.Proofs.Level2Round
import OrthoDFA.Proofs.RandomCorrect
import OrthoDFA.Proofs.Level4
import OrthoDFA.Proofs.Level5

/-!
# `Level2Claim` to `Level5Claim`: the proofs

The witnesses are the level theorems' premises read off `toVec x`, the counts and the rates'
reciprocals, and are bounded there before being carried back to the records. Off the instances
they take any value, the tolerances `1` out of range. Levels 4 and 5 take the allowance
`f = ⌈log(16/δ)⌉` and split their `δ/8` into `δ/16` for two probes colliding and `δ/16` for the
race tail, at `t = e² − 1` and `t' = 2t`: when `λ ≤ log(16/δ)/(2t)` and `c ≤ 1/(2t)` it is at most
`e^{log(16/δ) − 2f − 4} ≤ δ/16`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real Random Random.Eps Random.Kappa

/-- `(|Q|, |Σ|, 1/ε, 1/δ, L)`. -/
private abbrev LevelVec := ℝ × ℝ × ℝ × ℝ × ℝ

/-- `(|Q|, |Σ|, 1/ε, 1/δ)`. -/
private abbrev Level2Vec := ℝ × ℝ × ℝ × ℝ

private def vecRegion : Set LevelVec := Set.Ici (1, 1, 1, 2, 2)

private def vec2Region : Set Level2Vec := Set.Ici (1, 1, 1, 2)

private def lvSN (nQ nα : ℕ) : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1

private def lvKN (nQ nα : ℕ) : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)

private def lvWN (nQ L : ℕ) : ℕ := min (L + 1) (3 + Nat.clog 2 L) * (nQ + 1)

private def lvMN (nQ nα : ℕ) (δ : ℝ) : ℝ :=
  16 / 5 * lvSN nQ nα * Real.log (8 * lvSN nQ nα * (lvSN nQ nα + 1) / δ) + 1
    + ((nQ : ℝ) + 2) * nα * (nQ + 1)

private def lvS (Q A : ℝ) : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1

private def lvProbes : LevelVec → ℕ
  | (Q, A, iε, iδ, _) =>
    ⌈36 * lvS Q A / (1 / iε) * Real.log (16 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + lvS Q A⌉₊

private def lvProbes₂ : Level2Vec → ℕ
  | (Q, A, iε, iδ) =>
    ⌈48 * lvS Q A / (1 / iε) * Real.log (2 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + lvS Q A⌉₊

private def lvSlope : LevelVec → ℝ
  | (Q, _, iε, _, L) => (Q + 1) * (288 * iε + 16 * (2 + Nat.clog 2 ⌊L⌋₊))

private def lvLogK (Q A : ℝ) : ℝ :=
  Q * Real.log (1 + (Q + 1) ^ 3 * A) + (Q + 2) * A * Real.log (Q + 3)

private def lvSlack : LevelVec → ℝ
  | (Q, A, iε, iδ, _) =>
    (Real.log 16 + Real.log (Q + 1) + lvLogK Q A + Real.log iδ) * (48 * iε)

private def lvP₀ : LevelVec → ℝ
  | (Q, A, iε, iδ, L) =>
    if 0 < iδ then 1 / (8 * ((lvProbes (Q, A, iε, iδ, L) : ℝ) ^ 2 + 1) * iδ) else 1

private def lvM (Q A iδ : ℝ) : ℝ :=
  16 / 5 * lvS Q A * Real.log (8 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + 1 + (Q + 2) * A * (Q + 1)

private def lvT : ℝ := exp 2 - 1

private def lvκ : LevelVec → ℝ
  | (Q, A, _, iδ, _) =>
    if 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ then
      Real.log (16 * iδ) / (2 * lvT * (lvM Q A iδ + Real.log (16 * iδ)))
    else 1

private def lvW (Q L : ℝ) : ℝ := min ((⌊L⌋₊ : ℝ) + 1) (3 + (Nat.clog 2 ⌊L⌋₊ : ℝ)) * (Q + 1)

private def lvB : LevelVec → ℝ
  | (Q, A, iε, iδ, L) =>
    lvProbes (Q, A, iε, iδ, L) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1)

private def lvεw : LevelVec → ℝ
  | (Q, A, iε, iδ, L) =>
    if 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ then
      Real.log (16 * iδ)
        / (2 * lvT * (lvB (Q, A, iε, iδ, L) + Real.log (16 * iδ) * max (lvW Q L) (Q + 2)))
    else 1

private def lvF (iδ : ℝ) : ℕ := ⌈Real.log (16 * iδ)⌉₊

private def lvShift : LevelVec → LevelVec
  | (Q, A, iε, iδ, L) => (Q + lvF iδ, A, iε, iδ, L)

private lemma lvS_cast (nQ nα : ℕ) : lvS nQ nα = lvSN nQ nα := by
  simp only [lvS, lvSN]; push_cast; ring

private lemma lvS_ge {Q A : ℝ} (hQ : 0 ≤ Q) (hA : 0 ≤ A) : 1 ≤ lvS Q A := by
  have := mul_nonneg hQ (show 0 ≤ (Q + 2) * A + 1 by positivity)
  unfold lvS; linarith

private lemma lvT_pos : 0 < lvT := by
  unfold lvT; linarith [Real.add_one_le_exp (2 : ℝ)]

private lemma lvW_nonneg (L : ℝ) {Q : ℝ} (hQ : 0 ≤ Q) : 0 ≤ lvW Q L := by
  unfold lvW; positivity

private lemma lvM_nonneg {Q A iδ : ℝ} (hQ : 0 ≤ Q) (hA : 0 ≤ A) (hδ : 1 < iδ) : 0 ≤ lvM Q A iδ := by
  have hS := lvS_ge hQ hA
  have : 0 ≤ Real.log (8 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) := by
    rw [div_one_div']
    have h1 : (1 : ℝ) ≤ 8 * lvS Q A * (lvS Q A + 1) := by nlinarith
    exact Real.log_nonneg (one_le_mul_of_one_le_of_one_le h1 hδ.le)
  unfold lvM; positivity

/-! ### At an instance -/

private lemma endsWell_mono {α σ H : Type*} [Countable α] {M : DFA α σ} {U : σ → ℝ} {a a' b b' : ℝ}
    {D : Measure (FreeMonoid α)} {ρ : ℝ} {dis : Set (FreeMonoid α)} {k : ℕ}
    {law : H → Measure (FreeMonoid α)} {y : ℝ} {e : RoundEnd H} (ha : a ≤ a') (hb : b ≤ b')
    (h : EndsWell ⟨M, U, a, b, D, ρ, dis, k, law, y, e⟩) :
    EndsWell ⟨M, U, a', b', D, ρ, dis, k, law, y, e⟩ := by
  cases e with
  | consistent => exact h
  | harvest hh =>
    refine ⟨fun T hT => (h.1 T hT).trans ?_, h.2⟩
    exact mul_le_mul_of_nonneg_right (by nlinarith) measureReal_nonneg
  | failed => exact h.elim

private lemma budget_le (nQ nα : ℕ) {ε δ L : ℝ} {B : ℕ}
    (hB : B = lvProbes (nQ, nα, 1 / ε, 1 / δ, L)) :
    36 * (lvSN nQ nα : ℝ) / ε * Real.log (16 * (lvSN nQ nα : ℝ) * (lvSN nQ nα + 1) / δ)
      + lvSN nQ nα ≤ B := by
  rw [hB, lvProbes, lvS_cast, one_div_one_div, one_div_one_div]
  exact Nat.le_ceil _

private lemma slope_le (nQ L : ℕ) {y ε A iδ : ℝ} (hy : y = ε / 48) :
    ((nQ : ℝ) + 1) * (6 / y + 16 * (2 + (Nat.clog 2 L : ℝ)))
      ≤ lvSlope (nQ, A, 1 / ε, iδ, L) :=
  le_of_eq (by simp only [lvSlope, Nat.floor_natCast, hy]; ring)

private lemma slack_le (nQ nα : ℕ) {p₀ y ε δ c L : ℝ} (hp₀ : 0 ≤ p₀) (hy : 0 < y)
    (hyε : y = ε / 48) (hδ : 0 < δ) (hc : 0 < c) (hc16 : c ≤ 16) :
    p₀ * Real.log (c * ((nQ : ℝ) + 1) * (lvKN nQ nα : ℝ) / δ) / y
      ≤ p₀ * lvSlack (nQ, nα, 1 / ε, 1 / δ, L) := by
  have hK : (0 : ℝ) < lvKN nQ nα := by unfold lvKN; positivity
  have e : Real.log (16 * ((nQ : ℝ) + 1) * (lvKN nQ nα : ℝ) / δ)
      = Real.log 16 + Real.log (nQ + 1) + lvLogK nQ nα + Real.log (1 / δ) := by
    rw [Real.log_div (by positivity) hδ.ne', Real.log_mul (by positivity) hK.ne',
      Real.log_mul (by norm_num) (by positivity), one_div, Real.log_inv]
    simp only [lvKN, lvLogK]
    push_cast
    rw [Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow]
    push_cast; ring
  simp only [lvSlack]
  rw [← e]
  calc p₀ * Real.log (c * ((nQ : ℝ) + 1) * (lvKN nQ nα : ℝ) / δ) / y
      ≤ p₀ * Real.log (16 * ((nQ : ℝ) + 1) * (lvKN nQ nα : ℝ) / δ) / y := by
        gcongr
    _ = _ := by rw [hyε]; ring

private lemma collision_le {B : ℕ} {p₀ δ Q A iε L : ℝ} (hδ : 0 < δ) (hp₀0 : 0 ≤ p₀)
    (hB : B = lvProbes (Q, A, iε, 1 / δ, L)) (hp₀ : p₀ ≤ lvP₀ (Q, A, iε, 1 / δ, L)) :
    (B : ℝ) * ((B : ℝ) - 1) / 2 * p₀ ≤ δ / 16 := by
  simp only [lvP₀, if_pos (one_div_pos.2 hδ), ← hB] at hp₀
  have hn : (0 : ℝ) ≤ B := Nat.cast_nonneg B
  rw [show 1 / (8 * ((B : ℝ) ^ 2 + 1) * (1 / δ)) = δ / (8 * ((B : ℝ) ^ 2 + 1)) by
    field_simp, le_div_iff₀ (by positivity)] at hp₀
  have h1 : (B : ℝ) * ((B : ℝ) - 1) ≤ (B : ℝ) ^ 2 + 1 := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right h1 hp₀0]

private lemma tol_split {x b X ℓ : ℝ} (hx : 0 ≤ x) (hb : 0 ≤ b) (hX : 0 ≤ X) (hℓ : 0 < ℓ)
    (hD : 0 < b + ℓ * X) (h : x ≤ ℓ / (2 * lvT * (b + ℓ * X))) :
    2 * lvT * (x * b) ≤ ℓ ∧ 2 * lvT * (x * X) ≤ 1 := by
  have hT := lvT_pos
  have h1 := (le_div_iff₀ (mul_pos (mul_pos two_pos hT) hD)).1 h
  have p1 : 0 ≤ 2 * lvT * (x * b) := mul_nonneg (mul_nonneg two_pos.le hT.le) (mul_nonneg hx hb)
  have p2 : 0 ≤ 2 * lvT * (x * X) := mul_nonneg (mul_nonneg two_pos.le hT.le) (mul_nonneg hx hX)
  have e : x * (2 * lvT * (b + ℓ * X)) = 2 * lvT * (x * b) + 2 * lvT * (x * X) * ℓ := by ring
  have p3 : 0 ≤ 2 * lvT * (x * X) * ℓ := mul_nonneg p2 hℓ.le
  refine ⟨by linarith, le_of_mul_le_mul_right ?_ hℓ⟩
  linarith

private lemma εw_race (nQ nα L B : ℕ) {εw δ iε : ℝ} (hδ : 0 < δ) (hδ1 : δ < 1) (hεw0 : 0 ≤ εw)
    (hB : B = lvProbes (nQ, nα, iε, 1 / δ, L))
    (hεw : εw ≤ lvεw (nQ, nα, iε, 1 / δ, L)) :
    2 * lvT * (εw * ((B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1)))
        ≤ Real.log (16 * (1 / δ))
      ∧ 2 * lvT * (εw * ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ)) ≤ 1 := by
  have hW : lvW nQ L = lvWN nQ L := by
    simp only [lvW, lvWN, Nat.floor_natCast]; push_cast; ring
  have hD : lvB (nQ, nα, iε, 1 / δ, L) + Real.log (16 * (1 / δ)) * max (lvW nQ L) (nQ + 2)
      = (B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1)
        + Real.log (16 * (1 / δ)) * ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ) := by
    simp only [lvB, hW, ← hB]; push_cast; ring
  simp only [lvεw, if_pos (⟨Nat.cast_nonneg _, Nat.cast_nonneg _, one_lt_one_div hδ hδ1⟩ :
    (0 : ℝ) ≤ nQ ∧ (0 : ℝ) ≤ nα ∧ 1 < 1 / δ), hD] at hεw
  have hℓ : 0 < Real.log (16 * (1 / δ)) :=
    Real.log_pos (by linarith [one_lt_one_div hδ hδ1])
  have hX : (2 : ℝ) ≤ ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ) := by
    exact_mod_cast le_max_of_le_right (by omega)
  have hb : 0 ≤ (B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1) := by
    positivity
  exact tol_split hεw0 hb (by linarith) hℓ
    (add_pos_of_nonneg_of_pos hb (mul_pos hℓ (by linarith))) hεw

private lemma κ_race (nQ nα : ℕ) {κ δ iε L : ℝ} (hδ : 0 < δ) (hδ1 : δ < 1) (hκ0 : 0 ≤ κ)
    (hκ : κ ≤ lvκ (nQ, nα, iε, 1 / δ, L)) :
    0 ≤ κ * lvMN nQ nα δ ∧ 2 * lvT * (κ * lvMN nQ nα δ) ≤ Real.log (16 * (1 / δ))
      ∧ 2 * lvT * κ ≤ 1 := by
  have hm : lvM nQ nα (1 / δ) = lvMN nQ nα δ := by
    simp only [lvM, lvMN, lvS_cast, one_div_one_div]
  have hm0 : 0 ≤ lvMN nQ nα δ :=
    hm ▸ lvM_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _) (one_lt_one_div hδ hδ1)
  simp only [lvκ, if_pos (⟨Nat.cast_nonneg _, Nat.cast_nonneg _, one_lt_one_div hδ hδ1⟩ :
    (0 : ℝ) ≤ nQ ∧ (0 : ℝ) ≤ nα ∧ 1 < 1 / δ), hm] at hκ
  have hℓ : 0 < Real.log (16 * (1 / δ)) :=
    Real.log_pos (by linarith [one_lt_one_div hδ hδ1])
  obtain ⟨h1, h2⟩ := tol_split hκ0 hm0 zero_le_one hℓ (by linarith) (by rwa [mul_one])
  exact ⟨mul_nonneg hκ0 hm0, h1, by rwa [mul_one] at h2⟩

private lemma race_tail_apply {lam c δ X : ℝ} {G : Prop} (hδ : 0 < δ) (hl : 0 ≤ lam) (hc : 0 ≤ c)
    (hlam : 2 * lvT * lam ≤ Real.log (16 * (1 / δ))) (hc2 : 2 * lvT * c ≤ 1) (hX : X ≤ δ / 16)
    (k : ∀ t t' : ℝ, 0 ≤ t → 0 ≤ t' → t * c ≤ 1 - Real.exp (-(t' * c)) →
      X + Real.exp (t' * lam) / (1 + t) ^ (lvF (1 / δ) + 2) ≤ δ / 8 → G) : G := by
  have hT := lvT_pos
  refine k lvT (2 * lvT) hT.le (by linarith) ?_ ?_
  · have hx0 : 0 ≤ lvT * c := mul_nonneg hT.le hc
    have h1 : 1 + 2 * (lvT * c) ≤ Real.exp (2 * (lvT * c)) := by
      linarith [Real.add_one_le_exp (2 * (lvT * c))]
    have h2 : Real.exp (-(2 * (lvT * c))) ≤ 1 / (1 + 2 * (lvT * c)) := by
      rw [Real.exp_neg, ← one_div]
      exact one_div_le_one_div_of_le (by linarith) h1
    have h3 : 1 / (1 + 2 * (lvT * c)) ≤ 1 - lvT * c := by
      rw [div_le_iff₀ (by linarith)]; nlinarith
    rw [show 2 * lvT * c = 2 * (lvT * c) by ring]
    linarith
  · have hF : Real.log (16 * (1 / δ)) ≤ lvF (1 / δ) := Nat.le_ceil _
    have hℓ0 : 0 ≤ Real.log (16 * (1 / δ)) :=
      le_trans (mul_nonneg (mul_nonneg two_pos.le hT.le) hl) hlam
    have e : 1 + lvT = Real.exp 2 := by unfold lvT; ring
    rw [e, ← Real.exp_nat_mul, ← Real.exp_sub]
    have : Real.exp (2 * lvT * lam - ((lvF (1 / δ) + 2 : ℕ) : ℝ) * 2) ≤ δ / 16 :=
      calc _ ≤ Real.exp (-Real.log (16 * (1 / δ))) := Real.exp_le_exp.2 (by push_cast; linarith)
        _ = δ / 16 := by
          rw [Real.exp_neg, Real.exp_log (by positivity)]; field_simp
    linarith

/-! ### Big-O over the regions -/

private lemma clog_le {L : ℝ} (hL : 2 ≤ L) : (Nat.clog 2 ⌊L⌋₊ : ℝ) ≤ 2 * Real.log L + 1 := by
  have hn : 2 ≤ ⌊L⌋₊ := Nat.le_floor (by exact_mod_cast hL)
  have hnL : (⌊L⌋₊ : ℝ) ≤ L := Nat.floor_le (by linarith)
  have h := Nat.pow_pred_clog_lt_self (b := 2) one_lt_two (x := ⌊L⌋₊) (by omega)
  generalize Nat.clog 2 ⌊L⌋₊ = k at h ⊢
  have hk : k ≤ k.pred + 1 := by rw [Nat.pred_eq_sub_one]; omega
  have h' : (2 : ℝ) ^ k.pred < L := by
    calc (2 : ℝ) ^ k.pred < ⌊L⌋₊ := by exact_mod_cast h
      _ ≤ L := hnL
  have hlog : (k.pred : ℝ) * Real.log 2 < Real.log L := by
    rw [← Real.log_pow]; exact Real.log_lt_log (by positivity) h'
  have h2 := Real.log_two_gt_d9
  have hp : (0 : ℝ) ≤ k.pred := Nat.cast_nonneg _
  have : (k.pred : ℝ) ≤ 2 * Real.log L := by nlinarith
  have : (k : ℝ) ≤ k.pred + 1 := by exact_mod_cast hk
  linarith

private lemma lvS_le {Q A : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) : lvS Q A ≤ 7 * (Q ^ 2 * A) := by
  have h1 : 1 ≤ Q * A := one_le_mul_of_one_le_of_one_le hQ hA
  have h2 : Q * A ≤ Q ^ 2 * A := by nlinarith
  have h3 : Q ≤ Q * A := le_mul_of_one_le_right (by linarith) hA
  have h4 : A ≤ Q * A := le_mul_of_one_le_left (by linarith) hQ
  unfold lvS; nlinarith

private lemma log_lvS_le {Q A iδ c P : ℝ} (k : ℕ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 1 ≤ iδ)
    (hc1 : 1 ≤ c) (hc : c ≤ 16) (h : 896 * ((Q ^ 2 * A) ^ 2 * iδ) ≤ P ^ k) :
    0 ≤ Real.log (c * lvS Q A * (lvS Q A + 1) * iδ)
      ∧ Real.log (c * lvS Q A * (lvS Q A + 1) * iδ) ≤ k * Real.log P := by
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hX : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have h1 : lvS Q A * (lvS Q A + 1) ≤ 7 * (Q ^ 2 * A) * (8 * (Q ^ 2 * A)) :=
    mul_le_mul hS7 (by linarith) (by linarith) (by positivity)
  have h0 : 1 ≤ c * lvS Q A * (lvS Q A + 1) * iδ :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
      (one_le_mul_of_one_le_of_one_le hc1 hS1) (by linarith)) hδ
  refine ⟨Real.log_nonneg h0, log_le_mul_log_of_le_pow k (by linarith) ?_⟩
  calc c * lvS Q A * (lvS Q A + 1) * iδ = c * (lvS Q A * (lvS Q A + 1)) * iδ := by ring
    _ ≤ 16 * (7 * (Q ^ 2 * A) * (8 * (Q ^ 2 * A))) * iδ :=
        mul_le_mul_of_nonneg_right (mul_le_mul hc h1 (by nlinarith) (by norm_num)) (by linarith)
    _ = 896 * ((Q ^ 2 * A) ^ 2 * iδ) := by ring
    _ ≤ P ^ k := h

private lemma two_le_qa_iδ {Q A iδ : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ) :
    2 ≤ Q * A * iδ := by
  have := one_le_mul_of_one_le_of_one_le hQ hA
  nlinarith

private lemma log_lvS_le_fourteen {Q A iδ c : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ)
    (hc1 : 1 ≤ c) (hc : c ≤ 16) :
    0 ≤ Real.log (c * lvS Q A * (lvS Q A + 1) * iδ)
      ∧ Real.log (c * lvS Q A * (lvS Q A + 1) * iδ) ≤ 14 * Real.log (Q * A * iδ) := by
  have hz := two_le_qa_iδ hQ hA hδ
  have hu : 1 ≤ A ^ 2 * iδ ^ 3 :=
    one_le_mul_of_one_le_of_one_le (one_le_pow₀ hA) (one_le_pow₀ (by linarith))
  have h2 : (Q ^ 2 * A) ^ 2 * iδ ≤ (Q * A * iδ) ^ 4 := by
    calc (Q ^ 2 * A) ^ 2 * iδ = (Q ^ 4 * A ^ 2 * iδ) * 1 := by ring
      _ ≤ (Q ^ 4 * A ^ 2 * iδ) * (A ^ 2 * iδ ^ 3) := by gcongr
      _ = (Q * A * iδ) ^ 4 := by ring
  have h10 : (1024 : ℝ) ≤ (Q * A * iδ) ^ 10 :=
    (by norm_num : (1024 : ℝ) = 2 ^ 10).trans_le (pow_le_pow_left₀ (by norm_num) hz 10)
  have h := log_lvS_le (Q := Q) (A := A) (iδ := iδ) (c := c) (P := Q * A * iδ) 14 hQ hA
    (by linarith) hc1 hc
    (by
      calc 896 * ((Q ^ 2 * A) ^ 2 * iδ) ≤ (Q * A * iδ) ^ 10 * (Q * A * iδ) ^ 4 :=
          mul_le_mul (by linarith) h2 (by positivity) (by positivity)
        _ = (Q * A * iδ) ^ 14 := by ring)
  exact ⟨h.1, h.2.trans_eq (by norm_num)⟩

private lemma probes_pt {Q A iε iδ : ℝ} (L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    (lvProbes (Q, A, iε, iδ, L) : ℝ) ≤ 3544 * (Q ^ 2 * A * iε * Real.log (Q * A * iδ))
      ∧ 1 ≤ 2 * (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) := by
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hX : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) := by
    have : 1 ≤ 2 * (iε * Real.log (Q * A * iδ)) := by nlinarith
    nlinarith
  refine ⟨?_, by linarith⟩
  obtain ⟨hlog0, hlog⟩ := log_lvS_le_fourteen (c := 16) hQ hA hδ (by norm_num) le_rfl
  simp only [lvProbes, div_one_div']
  have h36 : 36 * lvS Q A * iε * Real.log (16 * lvS Q A * (lvS Q A + 1) * iδ)
      ≤ 36 * (7 * (Q ^ 2 * A)) * iε * (14 * Real.log (Q * A * iδ)) := by
    gcongr
  have h0 : 0 ≤ 36 * lvS Q A * iε * Real.log (16 * lvS Q A * (lvS Q A + 1) * iδ) := by
    positivity
  have hc := Nat.ceil_lt_add_one (show 0 ≤ 36 * lvS Q A * iε
    * Real.log (16 * lvS Q A * (lvS Q A + 1) * iδ) + lvS Q A by linarith)
  nlinarith

private lemma probes₂_pt {Q A iε iδ : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    (lvProbes₂ (Q, A, iε, iδ) : ℝ) ≤ 4720 * (Q ^ 2 * A * iε * Real.log (Q * A * iδ))
      ∧ 0 ≤ Q ^ 2 * A * iε * Real.log (Q * A * iδ) := by
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hX : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) := by
    have : 1 ≤ 2 * (iε * Real.log (Q * A * iδ)) := by nlinarith
    nlinarith
  refine ⟨?_, by positivity⟩
  obtain ⟨hlog0, hlog⟩ := log_lvS_le_fourteen (c := 2) hQ hA hδ (by norm_num) (by norm_num)
  simp only [lvProbes₂, div_one_div']
  have h48 : 48 * lvS Q A * iε * Real.log (2 * lvS Q A * (lvS Q A + 1) * iδ)
      ≤ 48 * (7 * (Q ^ 2 * A)) * iε * (14 * Real.log (Q * A * iδ)) := by
    gcongr
  have h0 : 0 ≤ 48 * lvS Q A * iε * Real.log (2 * lvS Q A * (lvS Q A + 1) * iδ) := by
    positivity
  have hc := Nat.ceil_lt_add_one (show 0 ≤ 48 * lvS Q A * iε
    * Real.log (2 * lvS Q A * (lvS Q A + 1) * iδ) + lvS Q A by linarith)
  nlinarith

private lemma slope_pt {Q A iε iδ L : ℝ} (hQ : 1 ≤ Q) (hε : 1 ≤ iε) (hL : 2 ≤ L) :
    0 ≤ lvSlope (Q, A, iε, iδ, L) ∧ 0 ≤ Q * (iε + Real.log L)
      ∧ lvSlope (Q, A, iε, iδ, L) ≤ 672 * (Q * (iε + Real.log L)) := by
  simp only [lvSlope]
  have hk := clog_le hL
  have hlL : 0 ≤ Real.log L := Real.log_nonneg (by linarith)
  have hk0 : (0 : ℝ) ≤ Nat.clog 2 ⌊L⌋₊ := Nat.cast_nonneg _
  have h1 : 288 * iε + 16 * (2 + (Nat.clog 2 ⌊L⌋₊ : ℝ)) ≤ 336 * (iε + Real.log L) := by linarith
  refine ⟨mul_nonneg (by linarith) (by linarith), mul_nonneg (by linarith) (by linarith), ?_⟩
  calc (Q + 1) * (288 * iε + 16 * (2 + (Nat.clog 2 ⌊L⌋₊ : ℝ)))
      ≤ (2 * Q) * (336 * (iε + Real.log L)) :=
        mul_le_mul (by linarith) h1 (by linarith) (by linarith)
    _ = 672 * (Q * (iε + Real.log L)) := by ring

private lemma slack_pt {Q A iε iδ L : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ)
    (hε : 1 ≤ iε) :
    0 ≤ lvSlack (Q, A, iε, iδ, L) ∧ 0 ≤ (Q * A * Real.log (Q * A) + Real.log iδ) * iε
      ∧ lvSlack (Q, A, iε, iδ, L)
        ≤ 3360 * ((Q * A * Real.log (Q * A) + Real.log iδ) * iε) := by
  simp only [lvSlack, lvLogK]
  have hu1 : 1 ≤ Q * A := one_le_mul_of_one_le_of_one_le hQ hA
  have hQu : Q ≤ Q * A := le_mul_of_one_le_right (by linarith) hA
  set u := Q * A with hu
  have hlu : 0 ≤ Real.log u := Real.log_nonneg hu1
  have hulu : u - 1 ≤ u * Real.log u := by
    have h := mul_le_mul_of_nonneg_left (Real.one_sub_inv_le_log_of_pos (by linarith : 0 < u))
      (by linarith : (0 : ℝ) ≤ u)
    have hu0 : u ≠ 0 := by positivity
    have e : u * (1 - u⁻¹) = u - 1 := by field_simp
    linarith
  have hδl : 1 ≤ 2 * Real.log iδ := by linarith [half_le_log hδ]
  have t1 : Real.log 16 ≤ 15 := by
    linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 16)]
  have t2 : Real.log (Q + 1) ≤ u := by
    linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < Q + 1)]
  have t3 : Q * Real.log (1 + (Q + 1) ^ 3 * A) ≤ 8 * u + 3 * (u * Real.log u) := by
    have hq : (Q + 1) ^ 3 ≤ 8 * Q ^ 3 := by
      have := pow_le_pow_left₀ (by linarith) (by linarith : Q + 1 ≤ 2 * Q) 3
      linarith [show (2 * Q) ^ 3 = 8 * Q ^ 3 by ring]
    have hA3 : A ≤ A ^ 3 := by
      have : 1 ≤ A ^ 2 := one_le_pow₀ hA
      calc A = A * 1 := by ring
        _ ≤ A * A ^ 2 := by gcongr
        _ = A ^ 3 := by ring
    have hQ3 : Q ^ 3 * A ≤ u ^ 3 := by
      calc Q ^ 3 * A ≤ Q ^ 3 * A ^ 3 := by gcongr
        _ = u ^ 3 := by rw [hu]; ring
    have h9 : 1 + (Q + 1) ^ 3 * A ≤ 9 * u ^ 3 := by
      have : (Q + 1) ^ 3 * A ≤ 8 * Q ^ 3 * A := by gcongr
      have : 1 ≤ u ^ 3 := one_le_pow₀ hu1
      linarith
    have hl : Real.log (1 + (Q + 1) ^ 3 * A) ≤ Real.log 9 + 3 * Real.log u := by
      have e : Real.log (9 * u ^ 3) = Real.log 9 + 3 * Real.log u := by
        rw [Real.log_mul (by norm_num) (by positivity), Real.log_pow]; push_cast; ring
      rw [← e]
      exact Real.log_le_log (by positivity) h9
    have hl9 : Real.log 9 ≤ 8 := by
      linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 9)]
    have := mul_le_mul_of_nonneg_left hl (by linarith : 0 ≤ Q)
    nlinarith [mul_le_mul_of_nonneg_right hQu hlu]
  have t4 : (Q + 2) * A * Real.log (Q + 3) ≤ 9 * u + 3 * (u * Real.log u) := by
    have h1 : (Q + 2) * A ≤ 3 * u := by
      nlinarith [mul_le_mul_of_nonneg_right hQ (by linarith : (0 : ℝ) ≤ A)]
    have h2 : Real.log (Q + 3) ≤ Real.log 4 + Real.log u := by
      rw [← Real.log_mul (by norm_num) (by positivity)]
      exact Real.log_le_log (by linarith) (by linarith)
    have h4 : Real.log 4 ≤ 3 := by
      linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 4)]
    have h3 : 0 ≤ Real.log (Q + 3) := Real.log_nonneg (by linarith)
    calc (Q + 2) * A * Real.log (Q + 3) ≤ 3 * u * (Real.log 4 + Real.log u) :=
          mul_le_mul h1 h2 h3 (by positivity)
      _ ≤ 9 * u + 3 * (u * Real.log u) := by nlinarith [mul_le_mul_of_nonneg_left h4 hlu]
  have h16 : 0 ≤ Real.log 16 := Real.log_nonneg (by norm_num)
  have hQ1 : 0 ≤ Real.log (Q + 1) := Real.log_nonneg (by linarith)
  have hc0 : 0 ≤ Q * Real.log (1 + (Q + 1) ^ 3 * A) :=
    mul_nonneg (by linarith) (Real.log_nonneg (le_add_of_nonneg_right (by positivity)))
  have hd0 : 0 ≤ (Q + 2) * A * Real.log (Q + 3) :=
    mul_nonneg (by positivity) (Real.log_nonneg (by linarith))
  have hδ0 : 0 ≤ Real.log iδ := Real.log_nonneg (by linarith)
  have hsum : Real.log 16 + Real.log (Q + 1)
        + (Q * Real.log (1 + (Q + 1) ^ 3 * A) + (Q + 2) * A * Real.log (Q + 3)) + Real.log iδ
      ≤ 70 * (u * Real.log u + Real.log iδ) := by
    have : 0 ≤ u * Real.log u := by positivity
    linarith
  refine ⟨mul_nonneg (by linarith) (by linarith),
    mul_nonneg (by positivity) (by linarith), ?_⟩
  calc _ ≤ 70 * (u * Real.log u + Real.log iδ) * (48 * iε) :=
        mul_le_mul_of_nonneg_right hsum (by linarith)
    _ = _ := by ring

private lemma p₀_pt {Q A iε iδ : ℝ} (L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    0 ≤ 1 / lvP₀ (Q, A, iε, iδ, L)
      ∧ 0 ≤ (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2 * iδ
      ∧ 1 / lvP₀ (Q, A, iε, iδ, L)
        ≤ 8 * (3544 ^ 2 + 4) * ((Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2 * iδ) := by
  obtain ⟨hn, hX1⟩ := probes_pt L hQ hA hε hδ
  simp only [lvP₀, if_pos (by linarith : (0 : ℝ) < iδ), one_div_one_div]
  have hn0 : (0 : ℝ) ≤ lvProbes (Q, A, iε, iδ, L) := Nat.cast_nonneg _
  refine ⟨by positivity, by positivity, ?_⟩
  set X := Q ^ 2 * A * iε * Real.log (Q * A * iδ)
  have h1 : ((lvProbes (Q, A, iε, iδ, L) : ℕ) : ℝ) ^ 2 ≤ 3544 ^ 2 * X ^ 2 := by
    rw [← mul_pow]; exact pow_le_pow_left₀ hn0 hn 2
  have h2 : 1 ≤ 4 * X ^ 2 := by nlinarith
  calc 8 * (((lvProbes (Q, A, iε, iδ, L) : ℕ) : ℝ) ^ 2 + 1) * iδ
      ≤ 8 * (3544 ^ 2 * X ^ 2 + 4 * X ^ 2) * iδ := by gcongr
    _ = _ := by ring

private lemma tol_inv_le {b X ℓ l Y C C' : ℝ} (hb : 0 ≤ b) (hX : 0 ≤ X) (hl : 0 < l)
    (hlℓ : l ≤ ℓ) (hbY : b ≤ C * Y) (hXY : X * l ≤ C' * Y) :
    0 ≤ 2 * lvT * (b + ℓ * X) / ℓ ∧ 2 * lvT * (b + ℓ * X) / ℓ ≤ 2 * lvT * (C + C') * (Y / l) := by
  have hT := lvT_pos
  have hℓ : 0 < ℓ := hl.trans_le hlℓ
  have hD : 0 ≤ b + ℓ * X := add_nonneg hb (mul_nonneg hℓ.le hX)
  refine ⟨div_nonneg (mul_nonneg (mul_nonneg two_pos.le hT.le) hD) hℓ.le, ?_⟩
  rw [mul_div_assoc', div_le_div_iff₀ hℓ hl]
  have a1 : b * l ≤ b * ℓ := mul_le_mul_of_nonneg_left hlℓ hb
  have a2 : b * ℓ ≤ C * Y * ℓ := mul_le_mul_of_nonneg_right hbY hℓ.le
  have a3 : ℓ * (X * l) ≤ ℓ * (C' * Y) := mul_le_mul_of_nonneg_left hXY hℓ.le
  have : (b + ℓ * X) * l ≤ (C + C') * Y * ℓ := by linarith
  calc 2 * lvT * (b + ℓ * X) * l = 2 * lvT * ((b + ℓ * X) * l) := by ring
    _ ≤ 2 * lvT * ((C + C') * Y * ℓ) :=
        mul_le_mul_of_nonneg_left this (mul_nonneg two_pos.le hT.le)
    _ = 2 * lvT * (C + C') * Y * ℓ := by ring

private lemma κ_pt {Q A iδ : ℝ} (iε L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ) :
    0 ≤ 1 / lvκ (Q, A, iε, iδ, L) ∧ 0 ≤ Q ^ 2 * A * Real.log (Q * A * iδ) / Real.log iδ
      ∧ 1 / lvκ (Q, A, iε, iδ, L)
        ≤ 2 * lvT * (340 + 1) * (Q ^ 2 * A * Real.log (Q * A * iδ) / Real.log iδ) := by
  simp only [lvκ, if_pos (⟨by linarith, by linarith, by linarith⟩ : 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ),
    one_div_div]
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hm0 := lvM_nonneg (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
    (by linarith : 1 < iδ)
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hz := two_le_qa_iδ hQ hA hδ
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log hz
  obtain ⟨hlog0, hlog⟩ := log_lvS_le_fourteen (c := 8) hQ hA hδ (by norm_num) (by norm_num)
  have hm : lvM Q A iδ ≤ 340 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by
    have h1 : 16 / 5 * lvS Q A * Real.log (8 * lvS Q A * (lvS Q A + 1) * iδ)
        ≤ 16 / 5 * (7 * (Q ^ 2 * A)) * (14 * Real.log (Q * A * iδ)) := by gcongr
    have h2 : (Q + 2) * A * (Q + 1) ≤ 6 * (Q ^ 2 * A) := by
      have : (Q + 2) * (Q + 1) ≤ 6 * Q ^ 2 := by nlinarith
      nlinarith
    have h3 : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by nlinarith
    unfold lvM; rw [div_one_div']; linarith
  have hl : 1 / 2 ≤ Real.log iδ := half_le_log hδ
  have hl16 : Real.log iδ ≤ Real.log (16 * iδ) := Real.log_le_log (by linarith) (by linarith)
  have hX : 1 * Real.log iδ ≤ 1 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by
    have h1 : Real.log iδ ≤ Real.log (Q * A * iδ) := Real.log_le_log (by linarith)
      (le_mul_of_one_le_left (by linarith) (one_le_mul_of_one_le_of_one_le hQ hA))
    have h2 : Real.log (Q * A * iδ) ≤ Q ^ 2 * A * Real.log (Q * A * iδ) :=
      le_mul_of_one_le_left (by linarith) hQA
    linarith
  obtain ⟨h0, hle⟩ := tol_inv_le (ℓ := Real.log (16 * iδ)) hm0 zero_le_one (by linarith) hl16 hm hX
  rw [mul_one] at h0 hle
  exact ⟨h0, div_nonneg (by positivity) (by linarith), hle⟩

private lemma εw_pt {Q A iε iδ L : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) (hL : 2 ≤ L) :
    0 ≤ 1 / lvεw (Q, A, iε, iδ, L)
      ∧ 0 ≤ Q ^ 3 * A * iε * Real.log (Q * A * iδ) * Real.log L / Real.log iδ
      ∧ 1 / lvεw (Q, A, iε, iδ, L)
        ≤ 2 * lvT * (71100 + 26)
          * (Q ^ 3 * A * iε * Real.log (Q * A * iδ) * Real.log L / Real.log iδ) := by
  obtain ⟨hn, hX1⟩ := probes_pt L hQ hA hε hδ
  simp only [lvεw, if_pos (⟨by linarith, by linarith, by linarith⟩ : 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ),
    lvB, one_div_div]
  have hn0 : (0 : ℝ) ≤ lvProbes (Q, A, iε, iδ, L) := Nat.cast_nonneg _
  have hW0 := lvW_nonneg L (by linarith : (0 : ℝ) ≤ Q)
  have hk := clog_le hL
  have hlL : 1 / 2 ≤ Real.log L := half_le_log hL
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hW : lvW Q L ≤ 20 * (Q * Real.log L) := by
    unfold lvW
    calc min ((⌊L⌋₊ : ℝ) + 1) (3 + (Nat.clog 2 ⌊L⌋₊ : ℝ)) * (Q + 1)
        ≤ (10 * Real.log L) * (2 * Q) :=
          mul_le_mul (by linarith [min_le_right ((⌊L⌋₊ : ℝ) + 1) (3 + (Nat.clog 2 ⌊L⌋₊ : ℝ))])
            (by linarith) (by linarith) (by linarith)
      _ = 20 * (Q * Real.log L) := by ring
  have hmax : max (lvW Q L) (Q + 2) ≤ lvW Q L + Q + 2 := max_le (by linarith) (by linarith)
  have hmx0 : 0 ≤ max (lvW Q L) (Q + 2) := le_trans (by linarith) (le_max_right _ _)
  have hb0 : 0 ≤ (lvProbes (Q, A, iε, iδ, L) : ℝ) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1) := by
    have : 0 ≤ (lvProbes (Q, A, iε, iδ, L) : ℝ) * lvW Q L := mul_nonneg hn0 hW0
    have : 0 ≤ (Q + 2) * A * (Q + 2) * (Q + 1) := by positivity
    linarith
  have hl : 1 / 2 ≤ Real.log iδ := half_le_log hδ
  have hl16 : Real.log iδ ≤ Real.log (16 * iδ) := Real.log_le_log (by linarith) (by linarith)
  have hlq : Real.log iδ ≤ Real.log (Q * A * iδ) := Real.log_le_log (by linarith)
    (le_mul_of_one_le_left (by linarith) (one_le_mul_of_one_le_of_one_le hQ hA))
  set X := Q ^ 2 * A * iε * Real.log (Q * A * iδ) with hXd
  have hY : Q ^ 3 * A * iε * Real.log (Q * A * iδ) * Real.log L = Q * X * Real.log L := by
    rw [hXd]; ring
  have hXq : Real.log (Q * A * iδ) ≤ X :=
    le_mul_of_one_le_left (by linarith) (one_le_mul_of_one_le_of_one_le hQA hε)
  have h1 : (lvProbes (Q, A, iε, iδ, L) : ℝ) * lvW Q L
      ≤ 3544 * X * (20 * (Q * Real.log L)) := mul_le_mul hn hW hW0 (by positivity)
  have h2 : (Q + 2) * A * (Q + 2) * (Q + 1) ≤ 72 * (Q * X * Real.log L) := by
    have a : (Q + 2) * A * (Q + 2) * (Q + 1) ≤ 18 * (Q ^ 3 * A) := by
      have e1 : (Q + 2) * (Q + 2) ≤ 9 * Q ^ 2 := by nlinarith
      have e2 := mul_le_mul e1 (by linarith : Q + 1 ≤ 2 * Q) (by linarith) (by positivity)
      nlinarith [mul_le_mul_of_nonneg_right e2 (by linarith : (0 : ℝ) ≤ A)]
    have b : Q ^ 3 * A ≤ 4 * (Q * X * Real.log L) := by
      have p1 : 1 / 2 ≤ iε * Real.log (Q * A * iδ) := by nlinarith
      have p2 : 1 / 4 ≤ iε * Real.log (Q * A * iδ) * Real.log L := by nlinarith
      have := mul_le_mul_of_nonneg_left p2 (by positivity : 0 ≤ Q ^ 3 * A)
      rw [hXd]; linarith
    linarith
  have hsum : (lvProbes (Q, A, iε, iδ, L) : ℝ) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1)
      ≤ 71100 * (Q * X * Real.log L) := by
    have : 3544 * X * (20 * (Q * Real.log L)) = 70880 * (Q * X * Real.log L) := by ring
    linarith
  have hmx : max (lvW Q L) (Q + 2) * Real.log iδ ≤ 26 * (Q * X * Real.log L) := by
    have e2 : max (lvW Q L) (Q + 2) * Real.log iδ
        ≤ (20 * (Q * Real.log L) + 3 * Q) * Real.log (Q * A * iδ) :=
      mul_le_mul (by linarith) hlq (by linarith) (by positivity)
    have e3 : Q * Real.log L * Real.log (Q * A * iδ) ≤ Q * Real.log L * X :=
      mul_le_mul_of_nonneg_left hXq (by positivity)
    have e4 := mul_le_mul_of_nonneg_left hlL (by positivity : 0 ≤ Q * Real.log (Q * A * iδ))
    linarith
  obtain ⟨h0, hle⟩ := tol_inv_le (ℓ := Real.log (16 * iδ)) hb0 hmx0 (by linarith) hl16 hsum hmx
  exact ⟨h0, div_nonneg (by positivity) (by linarith), by rw [hY]; exact hle⟩

private lemma tol_pos (v : LevelVec) : 0 < lvP₀ v ∧ 0 < lvκ v ∧ 0 < lvεw v := by
  obtain ⟨Q, A, iε, iδ, L⟩ := v
  simp only [lvP₀, lvκ, lvεw]
  refine ⟨?_, ?_, ?_⟩
  · split_ifs with h
    · positivity
    · norm_num
  · split_ifs with h
    · obtain ⟨hQ, hA, hδ⟩ := h
      have := lvM_nonneg hQ hA hδ
      have hℓ : 0 < Real.log (16 * iδ) := Real.log_pos (by linarith)
      exact div_pos hℓ (mul_pos (mul_pos two_pos lvT_pos) (by linarith))
    · norm_num
  · split_ifs with h
    · obtain ⟨hQ, hA, hδ⟩ := h
      have hW := lvW_nonneg L hQ
      have hB : 0 ≤ lvB (Q, A, iε, iδ, L) := by simp only [lvB]; positivity
      have := le_max_right (lvW Q L) (Q + 2)
      have hℓ : 0 < Real.log (16 * iδ) := Real.log_pos (by linarith)
      exact div_pos hℓ (mul_pos (mul_pos two_pos lvT_pos)
        (add_pos_of_nonneg_of_pos hB (mul_pos hℓ (by linarith))))
    · norm_num

private lemma level_region {Q A iε iδ L : ℝ} (hb : (Q, A, iε, iδ, L) ∈ vecRegion) :
    1 ≤ Q ∧ 1 ≤ A ∧ 1 ≤ iε ∧ 2 ≤ iδ ∧ 2 ≤ L := by
  simpa only [vecRegion, Set.mem_Ici, Prod.mk_le_mk] using hb

private lemma lvProbes_isBigO : (fun v => (lvProbes v : ℝ)) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => Q ^ 2 * A * iε * Real.log (Q * A * iδ) : LevelVec → ℝ) := by
  refine isBigO_on 3544 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  obtain ⟨h, h1⟩ := probes_pt L hQ hA hε hδ
  dsimp only
  exact ⟨Nat.cast_nonneg _, by linarith, h⟩

private lemma lvProbes₂_isBigO : (fun v => (lvProbes₂ v : ℝ)) =O[𝓟 vec2Region]
    (fun (Q, A, iε, iδ) => Q ^ 2 * A * iε * Real.log (Q * A * iδ) : Level2Vec → ℝ) := by
  refine isBigO_on 4720 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ⟩ := b
  obtain ⟨hQ, hA, hε, hδ⟩ : 1 ≤ Q ∧ 1 ≤ A ∧ 1 ≤ iε ∧ 2 ≤ iδ := by
    simpa only [vec2Region, Set.mem_Ici, Prod.mk_le_mk] using hb
  obtain ⟨h, h0⟩ := probes₂_pt hQ hA hε hδ
  dsimp only
  exact ⟨Nat.cast_nonneg _, h0, h⟩

private lemma lvSlope_isBigO : lvSlope =O[𝓟 vecRegion]
    (fun (Q, _, iε, _, L) => Q * (iε + Real.log L) : LevelVec → ℝ) := by
  refine isBigO_on 672 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, -, hε, -, hL⟩ := level_region hb
  exact slope_pt (A := A) (iδ := iδ) hQ hε hL

private lemma lvSlack_isBigO : lvSlack =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => (Q * A * Real.log (Q * A) + Real.log iδ) * iε
      : LevelVec → ℝ) := by
  refine isBigO_on 3360 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  exact slack_pt (L := L) hQ hA hδ hε

private lemma lvP₀_isBigO : (fun v => 1 / lvP₀ v) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2 * iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (8 * (3544 ^ 2 + 4)) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  exact p₀_pt L hQ hA hε hδ

private lemma lvκ_isBigO : (fun v => 1 / lvκ v) =O[𝓟 vecRegion]
    (fun (Q, A, _, iδ, _) => Q ^ 2 * A * Real.log (Q * A * iδ) / Real.log iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (2 * lvT * (340 + 1)) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, -, hδ, -⟩ := level_region hb
  exact κ_pt iε L hQ hA hδ

private lemma lvεw_isBigO : (fun v => 1 / lvεw v) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, L) => Q ^ 3 * A * iε * Real.log (Q * A * iδ) * Real.log L / Real.log iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (2 * lvT * (71100 + 26)) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, hL⟩ := level_region hb
  exact εw_pt hQ hA hε hδ hL

/-! ### At `|Q| + f` states -/

private lemma shift_pt {Q A iδ : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ) :
    (lvF iδ : ℝ) ≤ 33 * Real.log iδ ∧ 1 ≤ Q + lvF iδ ∧ Q + lvF iδ ≤ 17 * (Q + Real.log iδ)
      ∧ 0 ≤ Real.log ((Q + lvF iδ) * A * iδ)
      ∧ Real.log ((Q + lvF iδ) * A * iδ) ≤ 8 * Real.log (Q * A * iδ) := by
  have hl := half_le_log hδ
  have h16 : Real.log (16 * iδ) = Real.log 16 + Real.log iδ :=
    Real.log_mul (by norm_num) (by linarith)
  have h15 : Real.log 16 ≤ 15 := by
    linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 16)]
  have h0 : 0 ≤ Real.log 16 := Real.log_nonneg (by norm_num)
  have hF : (lvF iδ : ℝ) ≤ 16 + Real.log iδ := by
    have := Nat.ceil_lt_add_one (show 0 ≤ Real.log (16 * iδ) by linarith)
    unfold lvF; linarith
  have hF0 : (0 : ℝ) ≤ lvF iδ := Nat.cast_nonneg _
  have hli : Real.log iδ ≤ iδ := by
    linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < iδ)]
  have hP := two_le_qa_iδ hQ hA hδ
  have hQA : 1 ≤ Q * A := one_le_mul_of_one_le_of_one_le hQ hA
  have h1 : Q + lvF iδ ≤ 34 * (Q * iδ) := by
    have : iδ ≤ Q * iδ := le_mul_of_one_le_left (by linarith) hQ
    have : 2 * Q ≤ Q * iδ := by nlinarith
    linarith
  have hX1 : 1 ≤ (Q + lvF iδ) * A * iδ :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) hA)
      (by linarith)
  refine ⟨by linarith, by linarith, by linarith, Real.log_nonneg hX1,
    log_le_mul_log_of_le_pow 8 (by linarith) ?_⟩
  have h2 : (34 : ℝ) ≤ (Q * A * iδ) ^ 6 :=
    (by norm_num : (34 : ℝ) ≤ 2 ^ 6).trans (pow_le_pow_left₀ (by norm_num) hP 6)
  have h3 : Q * A * iδ * iδ ≤ (Q * A * iδ) ^ 2 := by
    calc Q * A * iδ * iδ = Q * A * iδ * iδ * 1 := by ring
      _ ≤ Q * A * iδ * iδ * (Q * A) := mul_le_mul_of_nonneg_left hQA (by positivity)
      _ = (Q * A * iδ) ^ 2 := by ring
  calc (Q + lvF iδ) * A * iδ ≤ 34 * (Q * iδ) * A * iδ := by gcongr
    _ = 34 * (Q * A * iδ * iδ) := by ring
    _ ≤ (Q * A * iδ) ^ 6 * (Q * A * iδ) ^ 2 := mul_le_mul h2 h3 (by positivity) (by positivity)
    _ = (Q * A * iδ) ^ 8 := by ring

private lemma shift_mapsTo : Set.MapsTo lvShift vecRegion vecRegion := by
  rintro ⟨Q, A, iε, iδ, L⟩ hb
  obtain ⟨hQ, hA, hε, hδ, hL⟩ := level_region hb
  have : (0 : ℝ) ≤ lvF iδ := Nat.cast_nonneg _
  simp only [lvShift, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by linarith, hA, hε, hδ, hL⟩

private lemma shift_isBigO {f g g' : LevelVec → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : (g ∘ lvShift) =O[𝓟 vecRegion] g') : (f ∘ lvShift) =O[𝓟 vecRegion] g' :=
  (isBigO_comp_on h shift_mapsTo).trans hg

private lemma lvF_isBigO : (fun (_, _, _, iδ, _) => (lvF iδ : ℝ) : LevelVec → ℝ)
    =O[𝓟 vecRegion] (fun (_, _, _, iδ, _) => Real.log iδ : LevelVec → ℝ) := by
  refine isBigO_on 33 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, -, hδ, -⟩ := level_region hb
  exact ⟨Nat.cast_nonneg _, by linarith [half_le_log hδ], (shift_pt hQ hA hδ).1⟩

private lemma probes_shift :
    ((fun (Q, A, iε, iδ, _) => Q ^ 2 * A * iε * Real.log (Q * A * iδ) : LevelVec → ℝ)
      ∘ lvShift) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => (Q + Real.log iδ) ^ 2 * A * iε * Real.log (Q * A * iδ)
      : LevelVec → ℝ) := by
  refine isBigO_on (289 * 8) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  obtain ⟨-, hQ', hQR, hl0, hl⟩ := shift_pt hQ hA hδ
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hR : 0 ≤ Real.log iδ := by linarith [half_le_log hδ]
  simp only [Function.comp, lvShift]
  refine ⟨by positivity, by positivity, ?_⟩
  calc (Q + lvF iδ) ^ 2 * A * iε * Real.log ((Q + lvF iδ) * A * iδ)
      ≤ (17 * (Q + Real.log iδ)) ^ 2 * A * iε * (8 * Real.log (Q * A * iδ)) := by gcongr
    _ = _ := by ring

private lemma slope_shift :
    ((fun (Q, _, iε, _, L) => Q * (iε + Real.log L) : LevelVec → ℝ) ∘ lvShift)
      =O[𝓟 vecRegion]
    (fun (Q, _, iε, iδ, L) => (Q + Real.log iδ) * (iε + Real.log L) : LevelVec → ℝ) := by
  refine isBigO_on 17 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, hL⟩ := level_region hb
  obtain ⟨-, hQ', hQR, -, -⟩ := shift_pt hQ hA hδ
  have hR : 0 ≤ Real.log iδ := by linarith [half_le_log hδ]
  have hlL : 0 ≤ Real.log L := by linarith [half_le_log hL]
  simp only [Function.comp, lvShift]
  refine ⟨by positivity, by positivity, ?_⟩
  calc (Q + lvF iδ) * (iε + Real.log L) ≤ (17 * (Q + Real.log iδ)) * (iε + Real.log L) := by
        gcongr
    _ = _ := by ring

private lemma slack_shift :
    ((fun (Q, A, iε, iδ, _) => (Q * A * Real.log (Q * A) + Real.log iδ) * iε
      : LevelVec → ℝ) ∘ lvShift) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => (Q + Real.log iδ) * A * Real.log ((Q + Real.log iδ) * A) * iε
      : LevelVec → ℝ) := by
  refine isBigO_on 700 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  obtain ⟨-, hQ', hQR, -, -⟩ := shift_pt hQ hA hδ
  have hl2 := Real.log_le_log (by norm_num) hδ
  have hR : 0 ≤ Real.log iδ := by linarith [half_le_log hδ]
  set R := Q + Real.log iδ
  have hRA : 5 / 3 ≤ R * A := by
    have : 5 / 3 ≤ R := by linarith [Real.log_two_gt_d9]
    nlinarith
  have hlRA : 2 / 5 ≤ Real.log (R * A) := by
    have h := Real.one_sub_inv_le_log_of_pos (by linarith : 0 < R * A)
    have : (R * A)⁻¹ ≤ (5 / 3)⁻¹ := inv_anti₀ (by norm_num) hRA
    rw [show ((5 : ℝ) / 3)⁻¹ = 3 / 5 by norm_num] at this
    linarith
  have hQ'A : 1 ≤ (Q + lvF iδ) * A := one_le_mul_of_one_le_of_one_le hQ' hA
  have h17 : Real.log (17 * (R * A)) ≤ 41 * Real.log (R * A) := by
    rw [Real.log_mul (by norm_num) (by linarith)]
    linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 17)]
  have h1 : (Q + lvF iδ) * A * Real.log ((Q + lvF iδ) * A)
      ≤ 17 * (R * A) * (41 * Real.log (R * A)) := by
    have hle : (Q + lvF iδ) * A ≤ 17 * (R * A) := by nlinarith
    calc (Q + lvF iδ) * A * Real.log ((Q + lvF iδ) * A)
        ≤ 17 * (R * A) * Real.log (17 * (R * A)) :=
          mul_le_mul hle (Real.log_le_log (by linarith) hle) (Real.log_nonneg hQ'A)
            (by linarith)
      _ ≤ _ := mul_le_mul_of_nonneg_left h17 (by linarith)
  have h2 : Real.log iδ ≤ 5 / 2 * (R * A * Real.log (R * A)) := by
    have : R ≤ R * A := le_mul_of_one_le_right (by linarith) hA
    nlinarith
  simp only [Function.comp, lvShift]
  have hlQ' : 0 ≤ (Q + lvF iδ) * A * Real.log ((Q + lvF iδ) * A) :=
    mul_nonneg (by linarith) (Real.log_nonneg hQ'A)
  refine ⟨mul_nonneg (by linarith) (by linarith),
    mul_nonneg (mul_nonneg (by linarith) (by linarith)) (by linarith), ?_⟩
  calc ((Q + lvF iδ) * A * Real.log ((Q + lvF iδ) * A) + Real.log iδ) * iε
      ≤ (700 * (R * A * Real.log (R * A))) * iε :=
        mul_le_mul_of_nonneg_right (by linarith) (by linarith)
    _ = _ := by ring

private lemma p₀_shift :
    ((fun (Q, A, iε, iδ, _) => (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2 * iδ
      : LevelVec → ℝ) ∘ lvShift) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => ((Q + Real.log iδ) ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2 * iδ
      : LevelVec → ℝ) := by
  refine isBigO_on ((289 * 8) ^ 2) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  obtain ⟨-, hQ', hQR, hl0, hl⟩ := shift_pt hQ hA hδ
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hR : 0 ≤ Real.log iδ := by linarith [half_le_log hδ]
  simp only [Function.comp, lvShift]
  refine ⟨by positivity, by positivity, ?_⟩
  have h : (Q + lvF iδ) ^ 2 * A * iε * Real.log ((Q + lvF iδ) * A * iδ)
      ≤ 289 * 8 * ((Q + Real.log iδ) ^ 2 * A * iε * Real.log (Q * A * iδ)) :=
    calc _ ≤ (17 * (Q + Real.log iδ)) ^ 2 * A * iε * (8 * Real.log (Q * A * iδ)) := by gcongr
      _ = _ := by ring
  calc ((Q + lvF iδ) ^ 2 * A * iε * Real.log ((Q + lvF iδ) * A * iδ)) ^ 2 * iδ
      ≤ (289 * 8 * ((Q + Real.log iδ) ^ 2 * A * iε * Real.log (Q * A * iδ))) ^ 2 * iδ := by
        gcongr
    _ = _ := by ring

private lemma κ_shift :
    ((fun (Q, A, _, iδ, _) => Q ^ 2 * A * Real.log (Q * A * iδ) / Real.log iδ
      : LevelVec → ℝ) ∘ lvShift) =O[𝓟 vecRegion]
    (fun (Q, A, _, iδ, _) => (Q + Real.log iδ) ^ 2 * A * Real.log (Q * A * iδ) / Real.log iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (289 * 8) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, -, hδ, -⟩ := level_region hb
  obtain ⟨-, hQ', hQR, hl0, hl⟩ := shift_pt hQ hA hδ
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hR : 0 ≤ Real.log iδ := by linarith [half_le_log hδ]
  simp only [Function.comp, lvShift]
  refine ⟨div_nonneg (by positivity) hR, div_nonneg (by positivity) hR, ?_⟩
  calc (Q + lvF iδ) ^ 2 * A * Real.log ((Q + lvF iδ) * A * iδ) / Real.log iδ
      ≤ (17 * (Q + Real.log iδ)) ^ 2 * A * (8 * Real.log (Q * A * iδ)) / Real.log iδ :=
        div_le_div_of_nonneg_right (by gcongr) hR
    _ = _ := by ring

private lemma εw_shift :
    ((fun (Q, A, iε, iδ, L) => Q ^ 3 * A * iε * Real.log (Q * A * iδ) * Real.log L / Real.log iδ
      : LevelVec → ℝ) ∘ lvShift) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, L) => (Q + Real.log iδ) ^ 3 * A * iε * Real.log (Q * A * iδ)
      * Real.log L / Real.log iδ : LevelVec → ℝ) := by
  refine isBigO_on (4913 * 8) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, hL⟩ := level_region hb
  obtain ⟨-, hQ', hQR, hl0, hl⟩ := shift_pt hQ hA hδ
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log (two_le_qa_iδ hQ hA hδ)
  have hR : 0 ≤ Real.log iδ := by linarith [half_le_log hδ]
  have hlL : 0 ≤ Real.log L := by linarith [half_le_log hL]
  simp only [Function.comp, lvShift]
  refine ⟨div_nonneg (by positivity) hR, div_nonneg (by positivity) hR, ?_⟩
  calc (Q + lvF iδ) ^ 3 * A * iε * Real.log ((Q + lvF iδ) * A * iδ) * Real.log L / Real.log iδ
      ≤ (17 * (Q + Real.log iδ)) ^ 3 * A * iε * (8 * Real.log (Q * A * iδ)) * Real.log L
          / Real.log iδ :=
        div_le_div_of_nonneg_right (by gcongr) hR
    _ = _ := by ring

private def toVec (x : LevelInputs) : LevelVec := (x.Q, x.A, 1 / x.ε, 1 / x.δ, x.L)

private def toVec₂ (x : Level2Inputs) : Level2Vec := (x.Q, x.A, 1 / x.ε, 1 / x.δ)

private lemma transfer {f g : LevelVec → ℝ} {g' : LevelInputs → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : ∀ x, g (toVec x) = g' x) : (fun x => f (toVec x)) =O[𝓟 levelRegion] g' := by
  refine (isBigO_comp_on h fun x hx => ?_).congr_right hg
  obtain ⟨hQ, hA, hδ, hδ2, hε0, hε, hL⟩ := hx
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, (le_div_iff₀ hε0).2 (by linarith),
    (le_div_iff₀ hδ).2 (by linarith), by exact_mod_cast hL⟩

private lemma transfer₂ {f g : Level2Vec → ℝ} {g' : Level2Inputs → ℝ} (h : f =O[𝓟 vec2Region] g)
    (hg : ∀ x, g (toVec₂ x) = g' x) : (fun x => f (toVec₂ x)) =O[𝓟 level2Region] g' := by
  refine (isBigO_comp_on h fun x hx => ?_).congr_right hg
  obtain ⟨hQ, hA, hδ, hδ2, hε0, hε⟩ := hx
  simp only [toVec₂, vec2Region, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, (le_div_iff₀ hε0).2 (by linarith),
    (le_div_iff₀ hδ).2 (by linarith)⟩

theorem level2Claim_holds : Level2Claim := by
  refine ⟨fun x => lvProbes₂ (toVec₂ x),
    transfer₂ lvProbes₂_isBigO fun x => by simp only [toVec₂]; ring_nf, ?_⟩
  intro α _ _ σ _ M read side C
  dsimp only
  intro hI hx D _
  have hy : C.harvestRate = C.disagreementRate / 2 := rfl
  simp only [level2Region, Set.mem_ofPred_eq] at hx
  obtain ⟨-, -, hδ, hδ2, hε0, hε1⟩ := hx
  refine Level2.level2RoundCorrect_holds M read side C _ hI (by rw [hy]; positivity)
    (by rw [hy]; linarith) hε1 hδ (by linarith) ?_ D
  simp only [toVec₂, lvProbes₂]
  refine le_trans (le_of_eq ?_) (Nat.le_ceil _)
  have hε' : C.disagreementRate ≠ 0 := hε0.ne'
  simp only [lvS_cast, lvSN, one_div_one_div, hy, sub_half]
  field_simp
  ring

theorem level3Claim_holds : Level3Claim := by
  refine ⟨fun x => lvProbes (toVec x), fun x => lvSlope (toVec x), fun x => lvSlack (toVec x),
    transfer lvProbes_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer lvSlope_isBigO fun x => by simp only [toVec],
    transfer lvSlack_isBigO fun x => by simp only [toVec]; ring_nf, ?_⟩
  intro α _ _ σ _ M side U Ω _ μ _ R P C p₀
  dsimp only
  intro hx hP hpre
  simp only [levelRegion, Set.mem_ofPred_eq] at hx
  obtain ⟨-, -, hδ, hδ2, hε0, hε, hL⟩ := hx
  have hyε : C.harvestRate = C.disagreementRate / 48 := rfl
  have hy : 0 < C.harvestRate := by rw [hyε]; positivity
  simp only [toVec] at hP ⊢
  have hp₀ : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  refine le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
    (level3RoundCorrect_holds M side U μ R P C p₀ hpre hδ hy hyε.le hε
      (budget_le (Fintype.card σ) (Fintype.card α) hP))
  exact fun hw => hd (endsWell_mono (slope_le _ _ hyε) (slack_le _ _ hp₀ hy hyε hδ (by norm_num)
    (by norm_num)) hw)

theorem level4Claim_holds : Level4Claim := by
  refine ⟨fun x => lvProbes (lvShift (toVec x)), fun x => lvF (1 / x.δ),
    fun x => lvSlope (lvShift (toVec x)), fun x => lvSlack (lvShift (toVec x)),
    fun x => lvP₀ (lvShift (toVec x)), fun x => lvεw (lvShift (toVec x)),
    transfer (shift_isBigO lvProbes_isBigO probes_shift) fun x => by simp only [toVec]; ring_nf,
    transfer (shift_isBigO lvSlope_isBigO slope_shift) fun x => by simp only [toVec],
    transfer (shift_isBigO lvSlack_isBigO slack_shift) fun x => by simp only [toVec]; ring_nf,
    transfer (shift_isBigO lvP₀_isBigO p₀_shift) fun x => by simp only [toVec]; ring_nf,
    transfer (shift_isBigO lvεw_isBigO εw_shift) fun x => by simp only [toVec]; ring_nf,
    transfer lvF_isBigO fun x => by simp only [toVec],
    fun x => ⟨(tol_pos _).1, (tol_pos _).2.2⟩, ?_⟩
  intro α _ _ σ _ M side U εw Ω _ μ _ R P C p₀
  dsimp only
  intro hx hP hpre hp₀ hεw0 hεw
  simp only [levelRegion, Set.mem_ofPred_eq] at hx
  obtain ⟨-, -, hδ, hδ2, hε0, hε, hL⟩ := hx
  have hyε : C.harvestRate = C.disagreementRate / 48 := rfl
  have hy : 0 < C.harvestRate := by rw [hyε]; positivity
  simp only [toVec, lvShift, ← Nat.cast_add] at hP hp₀ hεw ⊢
  have hδ1 : C.δ < 1 := by linarith
  have hp₀0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  set f := lvF (1 / C.δ)
  obtain ⟨hl, hc⟩ :=
    εw_race (Fintype.card σ + f) (Fintype.card α) P.L C.probeBudget hδ hδ1 hεw0 hP hεw
  refine race_tail_apply hδ (by positivity) (by positivity) hl hc
    (collision_le hδ hp₀0 hP hp₀) fun t t' ht ht' htc hfin =>
      le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
        (level4RoundCorrect_holds M side U εw μ R P C p₀ f t t' hpre hδ hδ1 hy hyε.le hε
          (budget_le (Fintype.card σ + f) (Fintype.card α) hP) ht ht' htc hfin)
  exact hd.imp (fun h hw => h (endsWell_mono (slope_le _ _ hyε)
    (slack_le _ _ hp₀0 hy hyε hδ (by norm_num) le_rfl) hw)) id

theorem level5Claim_holds : Level5Claim := by
  refine ⟨fun x => lvProbes (lvShift (toVec x)), fun x => lvF (1 / x.δ),
    fun x => lvSlope (lvShift (toVec x)), fun x => lvSlack (lvShift (toVec x)),
    fun x => lvP₀ (lvShift (toVec x)), fun x => lvκ (lvShift (toVec x)),
    transfer (shift_isBigO lvProbes_isBigO probes_shift) fun x => by simp only [toVec]; ring_nf,
    transfer (shift_isBigO lvSlope_isBigO slope_shift) fun x => by simp only [toVec],
    transfer (shift_isBigO lvSlack_isBigO slack_shift) fun x => by simp only [toVec]; ring_nf,
    transfer (shift_isBigO lvP₀_isBigO p₀_shift) fun x => by simp only [toVec]; ring_nf,
    transfer (shift_isBigO lvκ_isBigO κ_shift) fun x => by simp only [toVec]; ring_nf,
    transfer lvF_isBigO fun x => by simp only [toVec],
    fun x => ⟨(tol_pos _).1, (tol_pos _).2.1⟩, ?_⟩
  intro α _ _ σ _ M side U κ Ω _ μ _ R P C p₀
  dsimp only
  intro hx hP hpre hp₀ hκ0 hκ
  simp only [levelRegion, Set.mem_ofPred_eq] at hx
  obtain ⟨-, -, hδ, hδ2, hε0, hε, hL⟩ := hx
  have hyε : C.harvestRate = C.disagreementRate / 48 := rfl
  have hy : 0 < C.harvestRate := by rw [hyε]; positivity
  simp only [toVec, lvShift, ← Nat.cast_add] at hP hp₀ hκ ⊢
  have hδ1 : C.δ < 1 := by linarith
  have hp₀0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  set f := lvF (1 / C.δ)
  obtain ⟨hl0, hl, hc⟩ := κ_race (Fintype.card σ + f) (Fintype.card α)
    (iε := 1 / C.disagreementRate) (L := P.L) hδ hδ1 hκ0 hκ
  refine race_tail_apply hδ hl0 hκ0 hl hc (collision_le hδ hp₀0 hP hp₀)
    fun t t' ht ht' htc hfin =>
      le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
        (level5RoundCorrect_holds M side U κ μ R P C p₀ f t t' hpre hδ hδ1 hy hyε.le hε
          (budget_le (Fintype.card σ + f) (Fintype.card α) hP) hκ0 ht ht' htc hfin)
  exact hd.imp (fun h hw => h (endsWell_mono (slope_le _ _ hyε)
    (slack_le _ _ hp₀0 hy hyε hδ (by norm_num) le_rfl) hw)) id

end OrthoDFA
