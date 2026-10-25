import OrthoDFA.Claims.Levels
import OrthoDFA.Proofs.BigO
import OrthoDFA.Proofs.Level2Round
import OrthoDFA.Proofs.RandomCorrect
import OrthoDFA.Proofs.Level4
import OrthoDFA.Proofs.Level5

/-!
# `LevelsClaim`: the proof

The witnesses are the level theorems' premises read off `toVec x`, the counts and the rates'
reciprocals, and are bounded there before being carried back to the records. Off the instances
they take any value: level 2's budget is `0` where `ig < iε`, which no instance has and without
which its shape is no bound, and the tolerances are `1` out of range. Levels 4 and 5 split their `δ/8` into
`δ/16` for two probes colliding and `δ/16` for the race tail, at `t = 1/(λ + 2c)` and `t' = 2t`,
or at `t = 4/√δ` and `t' = 0` when `λ = c = 0`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real Random Random.Eps Random.Kappa

/-- `(|Q|, |Σ|, 1/ε, 1/δ, 1/y, L)`. -/
private abbrev LevelVec := ℝ × ℝ × ℝ × ℝ × ℝ × ℝ

/-- `(|Q|, |Σ|, 1/ε, 1/δ, 1/(ε − y))`. -/
private abbrev Level2Vec := ℝ × ℝ × ℝ × ℝ × ℝ

private def vecRegion : Set LevelVec := Set.Ici (1, 1, 1, 2, 1, 2)

private def vec2Region : Set Level2Vec := Set.Ici (1, 1, 1, 2, 1)

private def lvSN (nQ nα : ℕ) : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1

private def lvKN (nQ nα : ℕ) : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)

private def lvWN (nQ L : ℕ) : ℕ := min (L + 1) (3 + Nat.clog 2 L) * (nQ + 1)

private def lvMN (nQ nα : ℕ) (δ : ℝ) : ℝ :=
  16 / 5 * lvSN nQ nα * Real.log (8 * lvSN nQ nα * (lvSN nQ nα + 1) / δ) + 1
    + ((nQ : ℝ) + 2) * nα * (nQ + 1)

private def lvS (Q A : ℝ) : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1

private def lvProbes : LevelVec → ℕ
  | (Q, A, iε, iδ, _, _) =>
    ⌈36 * lvS Q A / (1 / iε) * Real.log (16 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + lvS Q A⌉₊

private def lvProbes₂ : Level2Vec → ℕ
  | (Q, A, iε, iδ, ig) =>
    if iε ≤ ig then
      ⌈12 * lvS Q A * (1 / iε) / (1 / ig) ^ 2
          * Real.log (2 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + lvS Q A⌉₊
    else 0

private def lvSlope : LevelVec → ℝ
  | (Q, _, _, _, iy, L) => (Q + 1) * (6 * iy + 16 * (2 + Nat.clog 2 ⌊L⌋₊))

private def lvLogK (Q A : ℝ) : ℝ :=
  Q * Real.log (1 + (Q + 1) ^ 3 * A) + (Q + 2) * A * Real.log (Q + 3)

private def lvSlack : LevelVec → ℝ
  | (Q, A, _, iδ, iy, _) =>
    (Real.log 16 + Real.log (Q + 1) + lvLogK Q A + Real.log iδ) * iy

private def lvP₀ : LevelVec → ℝ
  | (Q, A, iε, iδ, iy, L) =>
    if 0 < iδ then 1 / (8 * ((lvProbes (Q, A, iε, iδ, iy, L) : ℝ) ^ 2 + 1) * iδ) else 1

private def lvM (Q A iδ : ℝ) : ℝ :=
  16 / 5 * lvS Q A * Real.log (8 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + 1 + (Q + 2) * A * (Q + 1)

private def lvκ : LevelVec → ℝ
  | (Q, A, _, iδ, _, _) =>
    if 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ then Real.sqrt (1 / iδ) / (4 * exp 1 * (lvM Q A iδ + 2)) else 1

private def lvW (Q L : ℝ) : ℝ := min ((⌊L⌋₊ : ℝ) + 1) (3 + (Nat.clog 2 ⌊L⌋₊ : ℝ)) * (Q + 1)

private def lvB : LevelVec → ℝ
  | (Q, A, iε, iδ, iy, L) =>
    lvProbes (Q, A, iε, iδ, iy, L) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1)

private def lvεw : LevelVec → ℝ
  | (Q, A, iε, iδ, iy, L) =>
    if 0 ≤ Q ∧ 0 ≤ A ∧ 0 < iδ then
      Real.sqrt (1 / iδ) / (4 * exp 1 * (lvB (Q, A, iε, iδ, iy, L) + 2 * max (lvW Q L) (Q + 2)))
    else 1

private lemma lvS_cast (nQ nα : ℕ) : lvS nQ nα = lvSN nQ nα := by
  simp only [lvS, lvSN]; push_cast; ring

private lemma lvS_ge {Q A : ℝ} (hQ : 0 ≤ Q) (hA : 0 ≤ A) : 1 ≤ lvS Q A := by
  have := mul_nonneg hQ (show 0 ≤ (Q + 2) * A + 1 by positivity)
  unfold lvS; linarith

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

private lemma budget_le (nQ nα : ℕ) {ε δ iy L : ℝ} {B : ℕ}
    (hB : B = lvProbes (nQ, nα, 1 / ε, 1 / δ, iy, L)) :
    36 * (lvSN nQ nα : ℝ) / ε * Real.log (16 * (lvSN nQ nα : ℝ) * (lvSN nQ nα + 1) / δ)
      + lvSN nQ nα ≤ B := by
  rw [hB, lvProbes, lvS_cast, one_div_one_div, one_div_one_div]
  exact Nat.le_ceil _

private lemma slope_le (nQ L : ℕ) {y A iε iδ : ℝ} :
    ((nQ : ℝ) + 1) * (6 / y + 16 * (2 + (Nat.clog 2 L : ℝ)))
      ≤ lvSlope (nQ, A, iε, iδ, 1 / y, L) :=
  le_of_eq (by simp only [lvSlope, Nat.floor_natCast, mul_one_div])

private lemma slack_le (nQ nα : ℕ) {p₀ y δ c iε L : ℝ} (hp₀ : 0 ≤ p₀) (hy : 0 < y) (hδ : 0 < δ)
    (hc : 0 < c) (hc16 : c ≤ 16) :
    p₀ * Real.log (c * ((nQ : ℝ) + 1) * (lvKN nQ nα : ℝ) / δ) / y
      ≤ p₀ * lvSlack (nQ, nα, iε, 1 / δ, 1 / y, L) := by
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
    _ = _ := by ring

private lemma collision_le {B : ℕ} {p₀ δ Q A iε iy L : ℝ} (hδ : 0 < δ) (hp₀0 : 0 ≤ p₀)
    (hB : B = lvProbes (Q, A, iε, 1 / δ, iy, L)) (hp₀ : p₀ ≤ lvP₀ (Q, A, iε, 1 / δ, iy, L)) :
    (B : ℝ) * ((B : ℝ) - 1) / 2 * p₀ ≤ δ / 16 := by
  simp only [lvP₀, if_pos (one_div_pos.2 hδ), ← hB] at hp₀
  have hn : (0 : ℝ) ≤ B := Nat.cast_nonneg B
  rw [show 1 / (8 * ((B : ℝ) ^ 2 + 1) * (1 / δ)) = δ / (8 * ((B : ℝ) ^ 2 + 1)) by
    field_simp, le_div_iff₀ (by positivity)] at hp₀
  have h1 : (B : ℝ) * ((B : ℝ) - 1) ≤ (B : ℝ) ^ 2 + 1 := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right h1 hp₀0]

private lemma εw_race (nQ nα L B : ℕ) {εw δ iε iy : ℝ} (hδ : 0 < δ)
    (hB : B = lvProbes (nQ, nα, iε, 1 / δ, iy, L))
    (hεw : εw ≤ lvεw (nQ, nα, iε, 1 / δ, iy, L)) :
    εw * ((B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1))
      + 2 * (εw * ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ)) ≤ Real.sqrt δ / (4 * exp 1) := by
  have hW : lvW nQ L = lvWN nQ L := by
    simp only [lvW, lvWN, Nat.floor_natCast]; push_cast; ring
  have hX : lvB (nQ, nα, iε, 1 / δ, iy, L) + 2 * max (lvW nQ L) (nQ + 2)
      = (B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1)
        + 2 * ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ) := by
    simp only [lvB, hW, ← hB]; push_cast; ring
  simp only [lvεw, if_pos (⟨Nat.cast_nonneg _, Nat.cast_nonneg _, one_div_pos.2 hδ⟩ :
    (0 : ℝ) ≤ nQ ∧ (0 : ℝ) ≤ nα ∧ 0 < 1 / δ), one_div_one_div, hX] at hεw
  have hX0 : 0 < (B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1)
      + 2 * ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ) := by
    have : (2 : ℝ) ≤ ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ) := by
      exact_mod_cast le_max_of_le_right (by omega)
    have : 0 ≤ (B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1) := by
      positivity
    linarith
  calc _ = εw * ((B : ℝ) * (lvWN nQ L : ℝ) + ((nQ : ℝ) + 2) * nα * (nQ + 2) * (nQ + 1)
        + 2 * ((max (lvWN nQ L) (nQ + 2) : ℕ) : ℝ)) := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_right hεw hX0.le
    _ = Real.sqrt δ / (4 * exp 1) := by field_simp

private lemma κ_race (nQ nα : ℕ) {κ δ iε iy L : ℝ} (b c : ℝ) (hδ : 0 < δ) (hδ1 : δ < 1)
    (hκ0 : 0 ≤ κ)
    (hκ : κ ≤ lvκ (nQ, nα, iε, 1 / δ, iy, L)) :
    0 ≤ 0 * b + κ * lvMN nQ nα δ ∧ 0 ≤ 0 * c + κ
      ∧ 0 * b + κ * lvMN nQ nα δ + 2 * (0 * c + κ) ≤ Real.sqrt δ / (4 * exp 1) := by
  have hm : lvM nQ nα (1 / δ) = lvMN nQ nα δ := by
    simp only [lvM, lvMN, lvS_cast, one_div_one_div]
  have hm0 : 0 ≤ lvMN nQ nα δ :=
    hm ▸ lvM_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _) (one_lt_one_div hδ hδ1)
  simp only [lvκ, if_pos (⟨Nat.cast_nonneg _, Nat.cast_nonneg _, one_lt_one_div hδ hδ1⟩ :
    (0 : ℝ) ≤ nQ ∧ (0 : ℝ) ≤ nα ∧ 1 < 1 / δ), one_div_one_div, hm] at hκ
  refine ⟨by simpa using mul_nonneg hκ0 hm0, by simpa using hκ0, ?_⟩
  calc _ = κ * (lvMN nQ nα δ + 2) := by ring
    _ ≤ _ := mul_le_mul_of_nonneg_right hκ (by positivity)
    _ = Real.sqrt δ / (4 * exp 1) := by field_simp

private lemma race_tail_apply {lam c δ X : ℝ} {G : Prop} (hδ : 0 < δ) (hl : 0 ≤ lam) (hc : 0 ≤ c)
    (h : lam + 2 * c ≤ Real.sqrt δ / (4 * exp 1)) (hX : X ≤ δ / 16)
    (k : ∀ t t' : ℝ, 0 ≤ t → 0 ≤ t' → t * c ≤ 1 - Real.exp (-(t' * c)) →
      X + Real.exp (t' * lam) / (1 + t) ^ (0 + 2) ≤ δ / 8 → G) : G := by
  have hs : 0 < Real.sqrt δ := Real.sqrt_pos.2 hδ
  rw [zero_add] at k
  by_cases hpos : 0 < lam + 2 * c
  · obtain ⟨ht, ht', htc, htail⟩ := race_tail_choice hl hc hpos
    refine k _ _ ht ht' htc ?_
    have hsq : Real.exp 2 * (lam + 2 * c) ^ 2 ≤ δ / 16 := by
      have e : Real.exp 2 * (Real.sqrt δ / (4 * Real.exp 1)) ^ 2 = δ / 16 := by
        rw [show (2 : ℝ) = 1 + 1 by norm_num, Real.exp_add, div_pow, Real.sq_sqrt hδ.le]
        field_simp; ring
      calc Real.exp 2 * (lam + 2 * c) ^ 2
          ≤ Real.exp 2 * (Real.sqrt δ / (4 * Real.exp 1)) ^ 2 := by gcongr
        _ = δ / 16 := e
    linarith [htail.trans hsq]
  · have hl0 : lam = 0 := by linarith
    have hc0 : c = 0 := by linarith
    subst hl0 hc0
    refine k (4 / Real.sqrt δ) 0 (by positivity) le_rfl (by simp) ?_
    have ht : (4 / Real.sqrt δ) ^ 2 ≤ (1 + 4 / Real.sqrt δ) ^ 2 :=
      pow_le_pow_left₀ (by positivity) (by linarith) 2
    have e : 1 / (4 / Real.sqrt δ) ^ 2 = δ / 16 := by
      rw [div_pow, Real.sq_sqrt hδ.le]; field_simp; norm_num
    have : 1 / (1 + 4 / Real.sqrt δ) ^ 2 ≤ δ / 16 :=
      e ▸ one_div_le_one_div_of_le (by positivity) ht
    simp only [zero_mul, Real.exp_zero]
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

private lemma log_lvS_le_twelve {Q A iδ c : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ)
    (hc1 : 1 ≤ c) (hc : c ≤ 16) :
    0 ≤ Real.log (c * lvS Q A * (lvS Q A + 1) * iδ)
      ∧ Real.log (c * lvS Q A * (lvS Q A + 1) * iδ) ≤ 12 * Real.log (Q ^ 2 * A * iδ) := by
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hx : 2 ≤ Q ^ 2 * A * iδ := by nlinarith
  have h2 : (Q ^ 2 * A) ^ 2 * iδ ≤ (Q ^ 2 * A * iδ) ^ 2 := by
    have : iδ ≤ iδ ^ 2 := by nlinarith
    calc (Q ^ 2 * A) ^ 2 * iδ ≤ (Q ^ 2 * A) ^ 2 * iδ ^ 2 := by gcongr
      _ = (Q ^ 2 * A * iδ) ^ 2 := by ring
  have h10 : (1024 : ℝ) ≤ (Q ^ 2 * A * iδ) ^ 10 :=
    (by norm_num : (1024 : ℝ) = 2 ^ 10).trans_le (pow_le_pow_left₀ (by norm_num) hx 10)
  have h := log_lvS_le (Q := Q) (A := A) (iδ := iδ) (P := Q ^ 2 * A * iδ) 12 hQ hA (by linarith)
    hc1 hc
    (by
      calc 896 * ((Q ^ 2 * A) ^ 2 * iδ) ≤ (Q ^ 2 * A * iδ) ^ 10 * (Q ^ 2 * A * iδ) ^ 2 :=
          mul_le_mul (by linarith) h2 (by positivity) (by positivity)
        _ = (Q ^ 2 * A * iδ) ^ 12 := by ring)
  exact ⟨h.1, h.2.trans_eq (by norm_num)⟩

private lemma probes_pt {Q A iε iδ : ℝ} (iy L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    (lvProbes (Q, A, iε, iδ, iy, L) : ℝ) ≤ 3040 * (Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ))
      ∧ 1 ≤ 2 * (Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ)) := by
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hℓ : 1 / 2 ≤ Real.log (Q ^ 2 * A * iδ) := half_le_log (by nlinarith)
  have hX : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ)) := by
    have : 1 ≤ 2 * (iε * Real.log (Q ^ 2 * A * iδ)) := by nlinarith
    nlinarith
  refine ⟨?_, by linarith⟩
  obtain ⟨hlog0, hlog⟩ := log_lvS_le_twelve (c := 16) hQ hA hδ (by norm_num) le_rfl
  simp only [lvProbes, div_one_div']
  have h36 : 36 * lvS Q A * iε * Real.log (16 * lvS Q A * (lvS Q A + 1) * iδ)
      ≤ 36 * (7 * (Q ^ 2 * A)) * iε * (12 * Real.log (Q ^ 2 * A * iδ)) := by
    gcongr
  have h0 : 0 ≤ 36 * lvS Q A * iε * Real.log (16 * lvS Q A * (lvS Q A + 1) * iδ) := by
    positivity
  have hc := Nat.ceil_lt_add_one (show 0 ≤ 36 * lvS Q A * iε
    * Real.log (16 * lvS Q A * (lvS Q A + 1) * iδ) + lvS Q A by linarith)
  nlinarith

private lemma probes₂_pt {Q A iε iδ ig : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    (lvProbes₂ (Q, A, iε, iδ, ig) : ℝ)
        ≤ 1024 * (Q ^ 2 * A * ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ))
      ∧ 0 ≤ Q ^ 2 * A * ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ) := by
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hℓ : 1 / 2 ≤ Real.log (Q ^ 2 * A * iδ) := half_le_log (by nlinarith)
  have hG0 : 0 ≤ Q ^ 2 * A * ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ) := by positivity
  refine ⟨?_, hG0⟩
  simp only [lvProbes₂]
  split_ifs with h
  · have hr : 1 ≤ ig ^ 2 / iε := by rw [le_div_iff₀ (by linarith)]; nlinarith
    have e1 : 12 * lvS Q A * (1 / iε) / (1 / ig) ^ 2 = 12 * lvS Q A * (ig ^ 2 / iε) := by
      field_simp
    rw [e1, div_one_div']
    obtain ⟨hlog0, hlog⟩ := log_lvS_le_twelve (c := 2) hQ hA hδ (by norm_num) (by norm_num)
    have h12 : 12 * lvS Q A * (ig ^ 2 / iε) * Real.log (2 * lvS Q A * (lvS Q A + 1) * iδ)
        ≤ 12 * (7 * (Q ^ 2 * A)) * (ig ^ 2 / iε) * (12 * Real.log (Q ^ 2 * A * iδ)) := by
      gcongr
    have h0 : 0 ≤ 12 * lvS Q A * (ig ^ 2 / iε) * Real.log (2 * lvS Q A * (lvS Q A + 1) * iδ) := by
      positivity
    have hc := Nat.ceil_lt_add_one (show 0 ≤ 12 * lvS Q A * (ig ^ 2 / iε)
      * Real.log (2 * lvS Q A * (lvS Q A + 1) * iδ) + lvS Q A by linarith)
    have hX : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ)) := by
      have : 1 ≤ 2 * (ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ)) := by nlinarith
      calc Q ^ 2 * A = Q ^ 2 * A * 1 := by ring
        _ ≤ Q ^ 2 * A * (2 * (ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ))) := by gcongr
        _ = _ := by ring
    have e3 : 12 * (7 * (Q ^ 2 * A)) * (ig ^ 2 / iε) * (12 * Real.log (Q ^ 2 * A * iδ))
        = 1008 * (Q ^ 2 * A * ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ)) := by ring
    linarith
  · simp only [Nat.cast_zero]; positivity

private lemma slope_pt {Q A iε iδ iy L : ℝ} (hQ : 1 ≤ Q) (hy : 1 ≤ iy) (hL : 2 ≤ L) :
    0 ≤ lvSlope (Q, A, iε, iδ, iy, L) ∧ 0 ≤ Q * (iy + Real.log L)
      ∧ lvSlope (Q, A, iε, iδ, iy, L) ≤ 160 * (Q * (iy + Real.log L)) := by
  simp only [lvSlope]
  have hk := clog_le hL
  have hlL : 0 ≤ Real.log L := Real.log_nonneg (by linarith)
  have hk0 : (0 : ℝ) ≤ Nat.clog 2 ⌊L⌋₊ := Nat.cast_nonneg _
  have h1 : 6 * iy + 16 * (2 + (Nat.clog 2 ⌊L⌋₊ : ℝ)) ≤ 80 * (iy + Real.log L) := by linarith
  refine ⟨mul_nonneg (by linarith) (by linarith), mul_nonneg (by linarith) (by linarith), ?_⟩
  calc (Q + 1) * (6 * iy + 16 * (2 + (Nat.clog 2 ⌊L⌋₊ : ℝ)))
      ≤ (2 * Q) * (80 * (iy + Real.log L)) :=
        mul_le_mul (by linarith) h1 (by linarith) (by linarith)
    _ = 160 * (Q * (iy + Real.log L)) := by ring

private lemma slack_pt {Q A iε iδ iy L : ℝ} (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ)
    (hy : 1 ≤ iy) :
    0 ≤ lvSlack (Q, A, iε, iδ, iy, L) ∧ 0 ≤ (Q * A * Real.log (Q * A) + Real.log iδ) * iy
      ∧ lvSlack (Q, A, iε, iδ, iy, L) ≤ 70 * ((Q * A * Real.log (Q * A) + Real.log iδ) * iy) := by
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
  calc _ ≤ 70 * (u * Real.log u + Real.log iδ) * iy :=
        mul_le_mul_of_nonneg_right hsum (by linarith)
    _ = _ := by ring

private lemma p₀_pt {Q A iε iδ : ℝ} (iy L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    0 ≤ 1 / lvP₀ (Q, A, iε, iδ, iy, L)
      ∧ 0 ≤ (Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ)) ^ 2 * iδ
      ∧ 1 / lvP₀ (Q, A, iε, iδ, iy, L)
        ≤ 8 * (3040 ^ 2 + 4) * ((Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ)) ^ 2 * iδ) := by
  obtain ⟨hn, hX1⟩ := probes_pt iy L hQ hA hε hδ
  simp only [lvP₀, if_pos (by linarith : (0 : ℝ) < iδ), one_div_one_div]
  have hn0 : (0 : ℝ) ≤ lvProbes (Q, A, iε, iδ, iy, L) := Nat.cast_nonneg _
  refine ⟨by positivity, by positivity, ?_⟩
  set X := Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ)
  have h1 : ((lvProbes (Q, A, iε, iδ, iy, L) : ℕ) : ℝ) ^ 2 ≤ 3040 ^ 2 * X ^ 2 := by
    rw [← mul_pow]; exact pow_le_pow_left₀ hn0 hn 2
  have h2 : 1 ≤ 4 * X ^ 2 := by nlinarith
  calc 8 * (((lvProbes (Q, A, iε, iδ, iy, L) : ℕ) : ℝ) ^ 2 + 1) * iδ
      ≤ 8 * (3040 ^ 2 * X ^ 2 + 4 * X ^ 2) * iδ := by gcongr
    _ = _ := by ring

private lemma κ_pt {Q A iδ : ℝ} (iε iy L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ) :
    0 ≤ 1 / lvκ (Q, A, iε, iδ, iy, L) ∧ 0 ≤ Q ^ 2 * A * Real.log (Q * A * iδ) * Real.sqrt iδ
      ∧ 1 / lvκ (Q, A, iε, iδ, iy, L)
        ≤ 4 * exp 1 * 340 * (Q ^ 2 * A * Real.log (Q * A * iδ) * Real.sqrt iδ) := by
  simp only [lvκ, if_pos (⟨by linarith, by linarith, by linarith⟩ : 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ),
    one_div_div, one_div iδ, Real.sqrt_inv, div_inv_eq_mul]
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hm0 := lvM_nonneg (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
    (by linarith : 1 < iδ)
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hz : 2 ≤ Q * A * iδ := by nlinarith
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log hz
  have hs : 0 ≤ Real.sqrt iδ := Real.sqrt_nonneg _
  refine ⟨by positivity, by positivity, ?_⟩
  have hu : 1 ≤ A ^ 2 * iδ ^ 3 :=
    one_le_mul_of_one_le_of_one_le (one_le_pow₀ hA) (one_le_pow₀ (by linarith))
  have h2 : (Q ^ 2 * A) ^ 2 * iδ ≤ (Q * A * iδ) ^ 4 := by
    calc (Q ^ 2 * A) ^ 2 * iδ = (Q ^ 4 * A ^ 2 * iδ) * 1 := by ring
      _ ≤ (Q ^ 4 * A ^ 2 * iδ) * (A ^ 2 * iδ ^ 3) := by gcongr
      _ = (Q * A * iδ) ^ 4 := by ring
  have h10 : (1024 : ℝ) ≤ (Q * A * iδ) ^ 10 :=
    (by norm_num : (1024 : ℝ) = 2 ^ 10).trans_le (pow_le_pow_left₀ (by norm_num) hz 10)
  obtain ⟨hlog0, hlog⟩ := log_lvS_le (Q := Q) (A := A) (iδ := iδ) (c := 8) (P := Q * A * iδ) 14
    hQ hA (by linarith) (by norm_num) (by norm_num)
    (by
      calc 896 * ((Q ^ 2 * A) ^ 2 * iδ) ≤ (Q * A * iδ) ^ 10 * (Q * A * iδ) ^ 4 :=
          mul_le_mul (by linarith) h2 (by positivity) (by positivity)
        _ = (Q * A * iδ) ^ 14 := by ring)
  simp only [Nat.cast_ofNat] at hlog
  have hm : lvM Q A iδ + 2 ≤ 340 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by
    have h1 : 16 / 5 * lvS Q A * Real.log (8 * lvS Q A * (lvS Q A + 1) * iδ)
        ≤ 16 / 5 * (7 * (Q ^ 2 * A)) * (14 * Real.log (Q * A * iδ)) := by gcongr
    have h2 : (Q + 2) * A * (Q + 1) ≤ 6 * (Q ^ 2 * A) := by
      have : (Q + 2) * (Q + 1) ≤ 6 * Q ^ 2 := by nlinarith
      nlinarith
    have h3 : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by nlinarith
    unfold lvM; rw [div_one_div']; linarith
  calc 4 * exp 1 * (lvM Q A iδ + 2) * Real.sqrt iδ
      ≤ 4 * exp 1 * (340 * (Q ^ 2 * A * Real.log (Q * A * iδ))) * Real.sqrt iδ := by gcongr
    _ = _ := by ring

private lemma εw_pt {Q A iε iδ L : ℝ} (iy : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) (hL : 2 ≤ L) :
    0 ≤ 1 / lvεw (Q, A, iε, iδ, iy, L)
      ∧ 0 ≤ Q ^ 3 * A * iε * Real.log (Q ^ 2 * A * iδ) * Real.log L * Real.sqrt iδ
      ∧ 1 / lvεw (Q, A, iε, iδ, iy, L) ≤ 4 * exp 1 * 61000
        * (Q ^ 3 * A * iε * Real.log (Q ^ 2 * A * iδ) * Real.log L * Real.sqrt iδ) := by
  obtain ⟨hn, hX1⟩ := probes_pt iy L hQ hA hε hδ
  simp only [lvεw, if_pos (⟨by linarith, by linarith, by linarith⟩ : 0 ≤ Q ∧ 0 ≤ A ∧ 0 < iδ),
    lvB, one_div_div, one_div iδ, Real.sqrt_inv, div_inv_eq_mul]
  have hn0 : (0 : ℝ) ≤ lvProbes (Q, A, iε, iδ, iy, L) := Nat.cast_nonneg _
  have hW0 := lvW_nonneg L (by linarith : (0 : ℝ) ≤ Q)
  have hk := clog_le hL
  have hlL : 1 / 2 ≤ Real.log L := half_le_log hL
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hℓ : 1 / 2 ≤ Real.log (Q ^ 2 * A * iδ) := half_le_log (by nlinarith)
  have hs : 0 ≤ Real.sqrt iδ := Real.sqrt_nonneg _
  have hW : lvW Q L ≤ 20 * (Q * Real.log L) := by
    unfold lvW
    calc min ((⌊L⌋₊ : ℝ) + 1) (3 + (Nat.clog 2 ⌊L⌋₊ : ℝ)) * (Q + 1)
        ≤ (10 * Real.log L) * (2 * Q) :=
          mul_le_mul (by linarith [min_le_right ((⌊L⌋₊ : ℝ) + 1) (3 + (Nat.clog 2 ⌊L⌋₊ : ℝ))])
            (by linarith) (by linarith) (by linarith)
      _ = 20 * (Q * Real.log L) := by ring
  have hmax : max (lvW Q L) (Q + 2) ≤ lvW Q L + Q + 2 := max_le (by linarith) (by linarith)
  have hpos : 0 ≤ lvProbes (Q, A, iε, iδ, iy, L) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1)
      + 2 * max (lvW Q L) (Q + 2) := by
    have := le_max_right (lvW Q L) (Q + 2)
    have : 0 ≤ (lvProbes (Q, A, iε, iδ, iy, L) : ℝ) * lvW Q L := mul_nonneg hn0 hW0
    have : 0 ≤ (Q + 2) * A * (Q + 2) * (Q + 1) := by positivity
    linarith
  refine ⟨mul_nonneg (mul_nonneg (by positivity) hpos) hs, by positivity, ?_⟩
  set X := Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ) with hXd
  have hY : Q ^ 3 * A * iε * Real.log (Q ^ 2 * A * iδ) * Real.log L = Q * X * Real.log L := by
    rw [hXd]; ring
  have hXL : 1 ≤ 4 * (X * Real.log L) := by nlinarith
  have hQXL : Q ≤ 4 * (Q * X * Real.log L) := by nlinarith
  have hQL : Q * Real.log L ≤ 2 * (Q * X * Real.log L) := by
    have := mul_le_mul_of_nonneg_left hX1 (by positivity : 0 ≤ Q * Real.log L)
    nlinarith
  have h1 : (lvProbes (Q, A, iε, iδ, iy, L) : ℝ) * lvW Q L
      ≤ 3040 * X * (20 * (Q * Real.log L)) := mul_le_mul hn hW hW0 (by positivity)
  have h2 : (Q + 2) * A * (Q + 2) * (Q + 1) ≤ 72 * (Q * X * Real.log L) := by
    have a : (Q + 2) * A * (Q + 2) * (Q + 1) ≤ 18 * (Q ^ 3 * A) := by
      have e1 : (Q + 2) * (Q + 2) ≤ 9 * Q ^ 2 := by nlinarith
      have e2 := mul_le_mul e1 (by linarith : Q + 1 ≤ 2 * Q) (by linarith) (by positivity)
      nlinarith [mul_le_mul_of_nonneg_right e2 (by linarith : (0 : ℝ) ≤ A)]
    have b : Q ^ 3 * A ≤ 4 * (Q * X * Real.log L) := by
      have p1 : 1 / 2 ≤ iε * Real.log (Q ^ 2 * A * iδ) := by nlinarith
      have p2 : 1 / 4 ≤ iε * Real.log (Q ^ 2 * A * iδ) * Real.log L := by nlinarith
      have := mul_le_mul_of_nonneg_left p2 (by positivity : 0 ≤ Q ^ 3 * A)
      rw [hXd]; linarith
    linarith
  have hsum : lvProbes (Q, A, iε, iδ, iy, L) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1)
      + 2 * max (lvW Q L) (Q + 2) ≤ 61000 * (Q * X * Real.log L) := by
    have : 3040 * X * (20 * (Q * Real.log L)) = 60800 * (Q * X * Real.log L) := by ring
    linarith
  rw [hY]
  calc 4 * exp 1 * (lvProbes (Q, A, iε, iδ, iy, L) * lvW Q L + (Q + 2) * A * (Q + 2) * (Q + 1)
        + 2 * max (lvW Q L) (Q + 2)) * Real.sqrt iδ
      ≤ 4 * exp 1 * (61000 * (Q * X * Real.log L)) * Real.sqrt iδ := by gcongr
    _ = _ := by ring

private lemma tol_pos (v : LevelVec) : 0 < lvP₀ v ∧ 0 < lvκ v ∧ 0 < lvεw v := by
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := v
  simp only [lvP₀, lvκ, lvεw]
  refine ⟨?_, ?_, ?_⟩
  · split_ifs with h
    · positivity
    · norm_num
  · split_ifs with h
    · obtain ⟨hQ, hA, hδ⟩ := h
      have := lvM_nonneg hQ hA hδ
      exact div_pos (Real.sqrt_pos.2 (by positivity)) (by positivity)
    · norm_num
  · split_ifs with h
    · obtain ⟨hQ, hA, hδ⟩ := h
      have hW := lvW_nonneg L hQ
      have hB : 0 ≤ lvB (Q, A, iε, iδ, iy, L) := by simp only [lvB]; positivity
      have := le_max_right (lvW Q L) (Q + 2)
      exact div_pos (Real.sqrt_pos.2 (by positivity)) (mul_pos (by positivity) (by linarith))
    · norm_num

private lemma level_region {Q A iε iδ iy L : ℝ} (hb : (Q, A, iε, iδ, iy, L) ∈ vecRegion) :
    1 ≤ Q ∧ 1 ≤ A ∧ 1 ≤ iε ∧ 2 ≤ iδ ∧ 1 ≤ iy ∧ 2 ≤ L := by
  simpa only [vecRegion, Set.mem_Ici, Prod.mk_le_mk] using hb

private lemma lvProbes_isBigO : (fun v => (lvProbes v : ℝ)) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _, _) => Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ) : LevelVec → ℝ) := by
  refine isBigO_on 3040 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -, -⟩ := level_region hb
  obtain ⟨h, h1⟩ := probes_pt iy L hQ hA hε hδ
  dsimp only
  exact ⟨Nat.cast_nonneg _, by linarith, h⟩

private lemma lvProbes₂_isBigO : (fun v => (lvProbes₂ v : ℝ)) =O[𝓟 vec2Region]
    (fun (Q, A, iε, iδ, ig) => Q ^ 2 * A * ig ^ 2 / iε * Real.log (Q ^ 2 * A * iδ)
      : Level2Vec → ℝ) := by
  refine isBigO_on 1024 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, ig⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ : 1 ≤ Q ∧ 1 ≤ A ∧ 1 ≤ iε ∧ 2 ≤ iδ ∧ 1 ≤ ig := by
    simpa only [vec2Region, Set.mem_Ici, Prod.mk_le_mk] using hb
  obtain ⟨h, h0⟩ := probes₂_pt (ig := ig) hQ hA hε hδ
  dsimp only
  exact ⟨Nat.cast_nonneg _, h0, h⟩

private lemma lvSlope_isBigO : lvSlope =O[𝓟 vecRegion]
    (fun (Q, _, _, _, iy, L) => Q * (iy + Real.log L) : LevelVec → ℝ) := by
  refine isBigO_on 160 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := b
  obtain ⟨hQ, -, -, -, hy, hL⟩ := level_region hb
  exact slope_pt (A := A) (iε := iε) (iδ := iδ) hQ hy hL

private lemma lvSlack_isBigO : lvSlack =O[𝓟 vecRegion]
    (fun (Q, A, _, iδ, iy, _) => (Q * A * Real.log (Q * A) + Real.log iδ) * iy
      : LevelVec → ℝ) := by
  refine isBigO_on 70 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := b
  obtain ⟨hQ, hA, -, hδ, hy, -⟩ := level_region hb
  exact slack_pt (iε := iε) (L := L) hQ hA hδ hy

private lemma lvP₀_isBigO : (fun v => 1 / lvP₀ v) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _, _) => (Q ^ 2 * A * iε * Real.log (Q ^ 2 * A * iδ)) ^ 2 * iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (8 * (3040 ^ 2 + 4)) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -, -⟩ := level_region hb
  exact p₀_pt iy L hQ hA hε hδ

private lemma lvκ_isBigO : (fun v => 1 / lvκ v) =O[𝓟 vecRegion]
    (fun (Q, A, _, iδ, _, _) => Q ^ 2 * A * Real.log (Q * A * iδ) * Real.sqrt iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (4 * exp 1 * 340) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := b
  obtain ⟨hQ, hA, -, hδ, -, -⟩ := level_region hb
  exact κ_pt iε iy L hQ hA hδ

private lemma lvεw_isBigO : (fun v => 1 / lvεw v) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _, L) => Q ^ 3 * A * iε * Real.log (Q ^ 2 * A * iδ) * Real.log L
      * Real.sqrt iδ : LevelVec → ℝ) := by
  refine isBigO_on (4 * exp 1 * 61000) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, iy, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -, hL⟩ := level_region hb
  exact εw_pt iy hQ hA hε hδ hL

private def toVec (x : LevelInputs) : LevelVec := (x.Q, x.A, 1 / x.ε, 1 / x.δ, 1 / x.y, x.L)

private def toVec₂ (x : Level2Inputs) : Level2Vec :=
  (x.Q, x.A, 1 / x.ε, 1 / x.δ, 1 / (x.ε - x.y))

private lemma transfer {f g : LevelVec → ℝ} {g' : LevelInputs → ℝ} (h : f =O[𝓟 vecRegion] g)
    (hg : ∀ x, g (toVec x) = g' x) : (fun x => f (toVec x)) =O[𝓟 levelRegion] g' := by
  refine (isBigO_comp_on h fun x hx => ?_).congr_right hg
  obtain ⟨hQ, hA, hδ, hδ2, hy, hyε, hε, hL⟩ := hx
  have hε0 : 0 < x.ε := by linarith
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, (le_div_iff₀ hε0).2 (by linarith),
    (le_div_iff₀ hδ).2 (by linarith), (le_div_iff₀ hy).2 (by linarith), by exact_mod_cast hL⟩

private lemma transfer₂ {f g : Level2Vec → ℝ} {g' : Level2Inputs → ℝ} (h : f =O[𝓟 vec2Region] g)
    (hg : ∀ x, g (toVec₂ x) = g' x) : (fun x => f (toVec₂ x)) =O[𝓟 level2Region] g' := by
  refine (isBigO_comp_on h fun x hx => ?_).congr_right hg
  obtain ⟨hQ, hA, hδ, hδ2, hy, hyε, hε⟩ := hx
  simp only [toVec₂, vec2Region, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, (le_div_iff₀ (by linarith)).2 (by linarith),
    (le_div_iff₀ hδ).2 (by linarith), (le_div_iff₀ (by linarith)).2 (by linarith)⟩

theorem levelsClaim_holds : LevelsClaim := by
  refine ⟨fun x => lvProbes (toVec x), fun x => lvProbes₂ (toVec₂ x), fun x => lvSlope (toVec x),
    fun x => lvSlack (toVec x), fun x => lvP₀ (toVec x), fun x => lvκ (toVec x),
    fun x => lvεw (toVec x),
    transfer lvProbes_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer₂ lvProbes₂_isBigO fun x => by
      simp only [toVec₂]; rw [one_div_pow, div_one_div']; ring_nf,
    transfer lvSlope_isBigO fun x => by simp only [toVec],
    transfer lvSlack_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer lvP₀_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer lvκ_isBigO fun x => by simp only [toVec, one_div, Real.sqrt_inv]; ring_nf,
    transfer lvεw_isBigO fun x => by simp only [toVec, one_div, Real.sqrt_inv]; ring_nf,
    fun x => tol_pos _, ?_, ?_, ?_, ?_⟩
  · intro α _ _ σ _ M read side C
    dsimp only
    intro hI hx D _
    simp only [level2Region, Set.mem_ofPred_eq] at hx
    obtain ⟨-, -, hδ, hδ2, hy, hyε, hε1⟩ := hx
    refine Level2.level2RoundCorrect_holds M read side C _ hI hy hyε hε1 hδ (by linarith) ?_ D
    have hg : 0 < C.disagreementRate - C.harvestRate := by linarith
    simp only [toVec₂]
    rw [lvProbes₂, if_pos (one_div_le_one_div_of_le hg (by linarith))]
    refine le_trans (le_of_eq ?_) (Nat.le_ceil _)
    simp only [lvS_cast, one_div_one_div]
    rfl
  · intro α _ _ σ _ M side U Ω _ μ _ R P C p₀
    dsimp only
    intro hx hP hpre
    simp only [levelRegion, Set.mem_ofPred_eq] at hx
    obtain ⟨-, -, hδ, hδ2, hy, hyε, hε, hL⟩ := hx
    simp only [toVec] at hP ⊢
    have hp₀ : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
    refine le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
      (level3RoundCorrect_holds M side U μ R P C p₀ hpre hδ hy hyε hε
        (budget_le (Fintype.card σ) (Fintype.card α) hP))
    exact fun hw => hd (endsWell_mono (slope_le _ _) (slack_le _ _ hp₀ hy hδ (by norm_num)
      (by norm_num)) hw)
  · intro α _ _ σ _ M side U εw Ω _ μ _ R P C p₀
    dsimp only
    intro hx hP hpre hp₀ hεw0 hεw
    simp only [levelRegion, Set.mem_ofPred_eq] at hx
    obtain ⟨-, -, hδ, hδ2, hy, hyε, hε, hL⟩ := hx
    simp only [toVec] at hP hp₀ hεw ⊢
    have hδ1 : C.δ < 1 := by linarith
    have hp₀0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
    refine race_tail_apply hδ (by positivity) (by positivity)
      (εw_race (Fintype.card σ) (Fintype.card α) P.L C.probeBudget hδ hP hεw)
      (collision_le hδ hp₀0 hP hp₀) fun t t' ht ht' htc hfin =>
        le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
          (level4RoundCorrect_holds M side U εw μ R P C p₀ 0 t t' hpre hδ hδ1 hy hyε hε
            (budget_le (Fintype.card σ) (Fintype.card α) hP) ht ht' htc hfin)
    exact hd.imp (fun h hw => h (endsWell_mono (slope_le _ _)
      (slack_le _ _ hp₀0 hy hδ (by norm_num) le_rfl) hw)) id
  · intro α _ _ σ _ M side U κ Ω _ μ _ R P C p₀
    dsimp only
    intro hx hP hpre hp₀ hκ0 hκ
    simp only [levelRegion, Set.mem_ofPred_eq] at hx
    obtain ⟨-, -, hδ, hδ2, hy, hyε, hε, hL⟩ := hx
    simp only [toVec] at hP hp₀ hκ ⊢
    have hδ1 : C.δ < 1 := by linarith
    have hp₀0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
    have hr := fun b c => κ_race (Fintype.card σ) (Fintype.card α) (iε := 1 / C.disagreementRate)
      (iy := 1 / C.harvestRate) (L := P.L) b c hδ hδ1 hκ0 hκ
    refine race_tail_apply hδ (hr _ 0).1 (hr 0 _).2.1 (hr _ _).2.2 (collision_le hδ hp₀0 hP hp₀)
      fun t t' ht ht' htc hfin =>
        le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
          (level5RoundCorrect_holds M side U 0 κ μ R P C p₀ 0 t t' hpre hδ hδ1 hy hyε hε
            (budget_le (Fintype.card σ) (Fintype.card α) hP) le_rfl hκ0 ht ht' htc hfin)
    exact hd.imp (fun h hw => h (endsWell_mono (slope_le _ _)
      (slack_le _ _ hp₀0 hy hδ (by norm_num) le_rfl) hw)) id

end OrthoDFA
