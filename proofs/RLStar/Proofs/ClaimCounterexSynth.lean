import RLStar.Claims.CounterexSynth
import RLStar.Proofs.BigO
import RLStar.Proofs.Level4
import RLStar.Proofs.Level5

/-!
# `CounterexSynthClaim`: the proof

The witnesses are level 5's premises read off `toVec x`, the counts and the rates' reciprocals, and
are bounded there before being carried back to the record. Off the instances they take any value,
the tolerances `1` out of range. It takes level 5 at `f = 0` and splits its `δ/8` into `δ/16` for
two probes colliding with a wrong read and `δ/16` for the race tail, at `t = 1/W` and `t' = 2t`,
`W = δ/(16e²)`: when the probes' pairs weigh at most `1` and `κ (m + 2) ≤ W`, each is at most
`e² W = δ/16`.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real Random Random.Eps Random.Kappa

/-- `(|Q|, |Σ|, 1/ε, 1/δ, L)`. -/
private abbrev LevelVec := ℝ × ℝ × ℝ × ℝ × ℝ

private def vecRegion : Set LevelVec := Set.Ici (1, 1, 1, 2, 2)

private def lvSN (nQ nα : ℕ) : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1

private def lvKN (nQ nα : ℕ) : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)

private def lvMN (nQ nα : ℕ) (δ : ℝ) : ℝ :=
  16 / 5 * lvSN nQ nα * Real.log (8 * lvSN nQ nα * (lvSN nQ nα + 1) / δ) + 1
    + ((nQ : ℝ) + 2) * nα * (nQ + 1)

private def lvS (Q A : ℝ) : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1

private def lvProbes : LevelVec → ℕ
  | (Q, A, iε, iδ, _) =>
    ⌈36 * lvS Q A / (1 / iε) * Real.log (16 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + lvS Q A⌉₊

private def lvSlope : LevelVec → ℝ
  | (Q, _, iε, _, L) => (Q + 1) * (288 * iε + 16 * (2 + Nat.clog 2 ⌊L⌋₊))

private def lvLogK (Q A : ℝ) : ℝ :=
  Q * Real.log (1 + (Q + 1) ^ 3 * A) + (Q + 2) * A * Real.log (Q + 3)

private def lvSlack : LevelVec → ℝ
  | (Q, A, iε, iδ, _) =>
    (Real.log 16 + Real.log (Q + 1) + lvLogK Q A + Real.log iδ) * (48 * iε)

private def lvP₀ : LevelVec → ℝ
  | (Q, A, iε, iδ, L) =>
    1 / ((lvProbes (Q, A, iε, iδ, L) : ℝ) ^ 2 + 1)

private def lvM (Q A iδ : ℝ) : ℝ :=
  16 / 5 * lvS Q A * Real.log (8 * lvS Q A * (lvS Q A + 1) / (1 / iδ)) + 1 + (Q + 2) * A * (Q + 1)

private def lvκ : LevelVec → ℝ
  | (Q, A, _, iδ, _) =>
    if 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ then 1 / (16 * exp 2 * (lvM Q A iδ + 2) * iδ) else 1

private lemma lvS_cast (nQ nα : ℕ) : lvS nQ nα = lvSN nQ nα := by
  simp only [lvS, lvSN]; push_cast; ring

private lemma lvS_ge {Q A : ℝ} (hQ : 0 ≤ Q) (hA : 0 ≤ A) : 1 ≤ lvS Q A := by
  have := mul_nonneg hQ (show 0 ≤ (Q + 2) * A + 1 by positivity)
  unfold lvS; linarith

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

private lemma collision_le {B : ℕ} {p₀ Q A iε iδ L : ℝ} (hp₀0 : 0 ≤ p₀)
    (hB : B = lvProbes (Q, A, iε, iδ, L)) (hp₀ : p₀ ≤ lvP₀ (Q, A, iε, iδ, L)) :
    (B : ℝ) * ((B : ℝ) - 1) / 2 * p₀ ≤ 1 := by
  simp only [lvP₀, ← hB] at hp₀
  have hn : (0 : ℝ) ≤ B := Nat.cast_nonneg B
  rw [le_div_iff₀ (by positivity)] at hp₀
  have h1 : (B : ℝ) * ((B : ℝ) - 1) ≤ (B : ℝ) ^ 2 + 1 := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_right h1 hp₀0]

private lemma race_le (nQ nα : ℕ) {κ δ iε L : ℝ} (hδ : 0 < δ) (hδ1 : δ < 1) (hκ0 : 0 ≤ κ)
    (hκ : κ ≤ lvκ (nQ, nα, iε, 1 / δ, L)) :
    ∃ t t' : ℝ, 0 ≤ t ∧ 0 ≤ t' ∧ t * κ ≤ 1 - Real.exp (-(t' * κ))
      ∧ Real.exp (t' * (κ * lvMN nQ nα δ)) / (1 + t) ≤ δ / 16
      ∧ Real.exp (t' * (κ * lvMN nQ nα δ)) / (1 + t) ^ 2 ≤ δ / 16 := by
  have hm : lvM nQ nα (1 / δ) = lvMN nQ nα δ := by
    simp only [lvM, lvMN, lvS_cast, one_div_one_div]
  have hm0 : 0 ≤ lvMN nQ nα δ :=
    hm ▸ lvM_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _) (one_lt_one_div hδ hδ1)
  simp only [lvκ, if_pos (⟨Nat.cast_nonneg _, Nat.cast_nonneg _, one_lt_one_div hδ hδ1⟩ :
    (0 : ℝ) ≤ nQ ∧ (0 : ℝ) ≤ nα ∧ 1 < 1 / δ), hm] at hκ
  set m := lvMN nQ nα δ
  have he := Real.exp_pos 2
  set W := δ / (16 * Real.exp 2) with hW
  have hW0 : 0 < W := by positivity
  have hW1 : W ≤ 1 := by
    rw [hW, div_le_one (by positivity)]
    have := Real.add_one_le_exp (2 : ℝ)
    linarith
  have hκW : κ * (m + 2) ≤ W := by
    rw [show 16 * Real.exp 2 * (m + 2) * (1 / δ) = (m + 2) / W by rw [hW]; field_simp,
      one_div_div, le_div_iff₀ (by positivity)] at hκ
    exact hκ
  have hκm : 0 ≤ κ * m := mul_nonneg hκ0 hm0
  have hl0 : 0 ≤ W - 2 * κ := by linarith [show κ * (m + 2) = κ * m + 2 * κ by ring]
  obtain ⟨ht, ht', htc, htail⟩ := race_tail_choice hl0 hκ0 (by linarith)
  have hone := race_tail_one hl0 hκ0 (by linarith)
  rw [show W - 2 * κ + 2 * κ = W by ring] at ht ht' htc htail hone
  have hlam : κ * m ≤ W - 2 * κ := by linarith [show κ * (m + 2) = κ * m + 2 * κ by ring]
  have e : Real.exp 2 * W = δ / 16 := by rw [hW]; field_simp
  refine ⟨1 / W, 2 * (1 / W), ht, ht', htc, ?_, ?_⟩
  · calc Real.exp (2 * (1 / W) * (κ * m)) / (1 + 1 / W)
        ≤ Real.exp (2 * (1 / W) * (W - 2 * κ)) / (1 + 1 / W) := by
          gcongr
      _ ≤ δ / 16 := hone.trans e.le
  · have hsq : Real.exp 2 * W ^ 2 ≤ Real.exp 2 * W := by
      rw [sq, ← mul_assoc]; exact mul_le_of_le_one_right (by positivity) hW1
    calc Real.exp (2 * (1 / W) * (κ * m)) / (1 + 1 / W) ^ 2
        ≤ Real.exp (2 * (1 / W) * (W - 2 * κ)) / (1 + 1 / W) ^ 2 := by
          gcongr
      _ ≤ δ / 16 := (htail.trans hsq).trans e.le

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
    0 ≤ lvSlack (Q, A, iε, iδ, L) ∧ 0 ≤ (Q * A * Real.log (Q + 1) + Real.log iδ) * iε
      ∧ lvSlack (Q, A, iε, iδ, L)
        ≤ 2064 * ((Q * A * Real.log (Q + 1) + Real.log iδ) * iε) := by
  simp only [lvSlack, lvLogK]
  have hu1 : 1 ≤ Q * A := one_le_mul_of_one_le_of_one_le hQ hA
  have hQu : Q ≤ Q * A := le_mul_of_one_le_right (by linarith) hA
  have hℓ : 1 / 2 ≤ Real.log (Q + 1) := half_le_log (by linarith)
  set ℓ := Real.log (Q + 1) with hℓd
  have hw : ℓ ≤ Q * A * ℓ := le_mul_of_one_le_left (by linarith) hu1
  have hQw : Q * ℓ ≤ Q * A * ℓ := mul_le_mul_of_nonneg_right hQu (by linarith)
  have hu2 : Q * A ≤ 2 * (Q * A * ℓ) := by nlinarith
  have t1 : Real.log 16 ≤ 15 := by
    linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 16)]
  have t3 : Q * Real.log (1 + (Q + 1) ^ 3 * A) ≤ 6 * (Q * A * ℓ) := by
    have h1 : 1 ≤ (Q + 1) ^ 3 * A :=
      one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) hA
    have h2 : 1 + (Q + 1) ^ 3 * A ≤ (Q + 1) ^ 4 * A := by
      rw [show (Q + 1) ^ 4 * A = (Q + 1) ^ 3 * A * (Q + 1) by ring]; nlinarith
    have hl : Real.log (1 + (Q + 1) ^ 3 * A) ≤ 4 * ℓ + A := by
      have e : Real.log ((Q + 1) ^ 4 * A) = 4 * ℓ + Real.log A := by
        rw [Real.log_mul (by positivity) (by positivity), Real.log_pow]; push_cast; ring
      have := Real.log_le_log (by positivity) h2
      linarith [Real.log_le_sub_one_of_pos (by linarith : 0 < A)]
    have := mul_le_mul_of_nonneg_left hl (by linarith : 0 ≤ Q)
    nlinarith
  have t4 : (Q + 2) * A * Real.log (Q + 3) ≤ 6 * (Q * A * ℓ) := by
    have h1 : (Q + 2) * A ≤ 3 * (Q * A) := by
      nlinarith [mul_le_mul_of_nonneg_right hQ (by linarith : (0 : ℝ) ≤ A)]
    have h2 : Real.log (Q + 3) ≤ 2 * ℓ := by
      have := Real.log_le_log (by linarith) (by nlinarith : Q + 3 ≤ (Q + 1) ^ 2)
      rw [Real.log_pow] at this; push_cast at this; linarith
    calc (Q + 2) * A * Real.log (Q + 3) ≤ 3 * (Q * A) * (2 * ℓ) :=
          mul_le_mul h1 h2 (Real.log_nonneg (by linarith)) (by positivity)
      _ = 6 * (Q * A * ℓ) := by ring
  have hc0 : 0 ≤ Q * Real.log (1 + (Q + 1) ^ 3 * A) :=
    mul_nonneg (by linarith) (Real.log_nonneg (le_add_of_nonneg_right (by positivity)))
  have hd0 : 0 ≤ (Q + 2) * A * Real.log (Q + 3) :=
    mul_nonneg (by positivity) (Real.log_nonneg (by linarith))
  have h16 : 0 ≤ Real.log 16 := Real.log_nonneg (by norm_num)
  have hδ0 : 0 ≤ Real.log iδ := Real.log_nonneg (by linarith)
  have hsum : Real.log 16 + ℓ
        + (Q * Real.log (1 + (Q + 1) ^ 3 * A) + (Q + 2) * A * Real.log (Q + 3)) + Real.log iδ
      ≤ 43 * (Q * A * ℓ + Real.log iδ) := by
    linarith
  refine ⟨mul_nonneg (by linarith) (by linarith),
    mul_nonneg (by positivity) (by linarith), ?_⟩
  calc _ ≤ 43 * (Q * A * ℓ + Real.log iδ) * (48 * iε) :=
        mul_le_mul_of_nonneg_right hsum (by linarith)
    _ = _ := by ring

private lemma p₀_pt {Q A iε iδ : ℝ} (L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hε : 1 ≤ iε)
    (hδ : 2 ≤ iδ) :
    0 ≤ 1 / lvP₀ (Q, A, iε, iδ, L)
      ∧ 0 ≤ (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2
      ∧ 1 / lvP₀ (Q, A, iε, iδ, L)
        ≤ (3544 ^ 2 + 4) * (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2 := by
  obtain ⟨hn, hX1⟩ := probes_pt L hQ hA hε hδ
  simp only [lvP₀, one_div_one_div]
  have hn0 : (0 : ℝ) ≤ lvProbes (Q, A, iε, iδ, L) := Nat.cast_nonneg _
  refine ⟨by positivity, by positivity, ?_⟩
  set X := Q ^ 2 * A * iε * Real.log (Q * A * iδ)
  have h1 : ((lvProbes (Q, A, iε, iδ, L) : ℕ) : ℝ) ^ 2 ≤ 3544 ^ 2 * X ^ 2 := by
    rw [← mul_pow]; exact pow_le_pow_left₀ hn0 hn 2
  have h2 : 1 ≤ 4 * X ^ 2 := by nlinarith
  calc ((lvProbes (Q, A, iε, iδ, L) : ℕ) : ℝ) ^ 2 + 1
      ≤ 3544 ^ 2 * X ^ 2 + 4 * X ^ 2 := by gcongr
    _ = _ := by ring

private lemma κ_pt {Q A iδ : ℝ} (iε L : ℝ) (hQ : 1 ≤ Q) (hA : 1 ≤ A) (hδ : 2 ≤ iδ) :
    0 ≤ 1 / lvκ (Q, A, iε, iδ, L) ∧ 0 ≤ Q ^ 2 * A * Real.log (Q * A * iδ) * iδ
      ∧ 1 / lvκ (Q, A, iε, iδ, L)
        ≤ 16 * exp 2 * 344 * (Q ^ 2 * A * Real.log (Q * A * iδ) * iδ) := by
  simp only [lvκ, if_pos (⟨by linarith, by linarith, by linarith⟩ : 0 ≤ Q ∧ 0 ≤ A ∧ 1 < iδ),
    one_div_one_div]
  have hS1 := lvS_ge (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
  have hS7 := lvS_le hQ hA
  have hm0 := lvM_nonneg (by linarith : (0 : ℝ) ≤ Q) (by linarith : (0 : ℝ) ≤ A)
    (by linarith : 1 < iδ)
  have hQA : 1 ≤ Q ^ 2 * A := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hQ) hA
  have hz := two_le_qa_iδ hQ hA hδ
  have hℓ : 1 / 2 ≤ Real.log (Q * A * iδ) := half_le_log hz
  obtain ⟨hlog0, hlog⟩ := log_lvS_le_fourteen (c := 8) hQ hA hδ (by norm_num) (by norm_num)
  have h3 : Q ^ 2 * A ≤ 2 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by nlinarith
  have hm : lvM Q A iδ ≤ 340 * (Q ^ 2 * A * Real.log (Q * A * iδ)) := by
    have h1 : 16 / 5 * lvS Q A * Real.log (8 * lvS Q A * (lvS Q A + 1) * iδ)
        ≤ 16 / 5 * (7 * (Q ^ 2 * A)) * (14 * Real.log (Q * A * iδ)) := by gcongr
    have h2 : (Q + 2) * A * (Q + 1) ≤ 6 * (Q ^ 2 * A) := by
      have : (Q + 2) * (Q + 1) ≤ 6 * Q ^ 2 := by nlinarith
      nlinarith
    unfold lvM; rw [div_one_div']; linarith
  set X := Q ^ 2 * A * Real.log (Q * A * iδ)
  have hX : lvM Q A iδ + 2 ≤ 344 * X := by linarith
  have hs : 0 ≤ iδ := by linarith
  have he : 0 ≤ 16 * exp 2 := by positivity
  refine ⟨mul_nonneg (mul_nonneg he (by linarith)) hs, mul_nonneg (by linarith) hs, ?_⟩
  calc 16 * exp 2 * (lvM Q A iδ + 2) * iδ ≤ 16 * exp 2 * (344 * X) * iδ :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hX he) hs
    _ = _ := by ring

private lemma tol_pos (v : LevelVec) : 0 < lvP₀ v ∧ 0 < lvκ v := by
  obtain ⟨Q, A, iε, iδ, L⟩ := v
  simp only [lvP₀, lvκ]
  refine ⟨?_, ?_⟩
  · positivity
  · split_ifs with h
    · obtain ⟨hQ, hA, hδ⟩ := h
      have := lvM_nonneg hQ hA hδ
      exact div_pos one_pos (mul_pos (mul_pos (by positivity) (by linarith)) (by linarith))
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

private lemma lvSlope_isBigO : lvSlope =O[𝓟 vecRegion]
    (fun (Q, _, iε, _, L) => Q * (iε + Real.log L) : LevelVec → ℝ) := by
  refine isBigO_on 672 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, -, hε, -, hL⟩ := level_region hb
  exact slope_pt (A := A) (iδ := iδ) hQ hε hL

private lemma lvSlack_isBigO : lvSlack =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => (Q * A * Real.log (Q + 1) + Real.log iδ) * iε
      : LevelVec → ℝ) := by
  refine isBigO_on 2064 fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  exact slack_pt (L := L) hQ hA hδ hε

private lemma lvP₀_isBigO : (fun v => 1 / lvP₀ v) =O[𝓟 vecRegion]
    (fun (Q, A, iε, iδ, _) => (Q ^ 2 * A * iε * Real.log (Q * A * iδ)) ^ 2
      : LevelVec → ℝ) := by
  refine isBigO_on (3544 ^ 2 + 4) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, hε, hδ, -⟩ := level_region hb
  exact p₀_pt L hQ hA hε hδ

private lemma lvκ_isBigO : (fun v => 1 / lvκ v) =O[𝓟 vecRegion]
    (fun (Q, A, _, iδ, _) => Q ^ 2 * A * Real.log (Q * A * iδ) * iδ
      : LevelVec → ℝ) := by
  refine isBigO_on (16 * exp 2 * 344) fun b hb => ?_
  obtain ⟨Q, A, iε, iδ, L⟩ := b
  obtain ⟨hQ, hA, -, hδ, -⟩ := level_region hb
  exact κ_pt iε L hQ hA hδ

private def toVec (x : CounterexSynthInputs) : LevelVec := (x.Q, x.A, 1 / x.ε, 1 / x.δ, x.L)

private lemma transfer {f g : LevelVec → ℝ} {g' : CounterexSynthInputs → ℝ}
    (h : f =O[𝓟 vecRegion] g) (hg : ∀ x, g (toVec x) = g' x) :
    (fun x => f (toVec x)) =O[𝓟 counterexSynthRegion] g' := by
  refine (isBigO_comp_on h fun x hx => ?_).congr_right hg
  obtain ⟨hQ, hA, hδ, hδ2, hε0, hε, hL⟩ := hx
  simp only [toVec, vecRegion, Set.mem_Ici, Prod.mk_le_mk]
  exact ⟨by exact_mod_cast hQ, by exact_mod_cast hA, (le_div_iff₀ hε0).2 (by linarith),
    (le_div_iff₀ hδ).2 (by linarith), by exact_mod_cast hL⟩

theorem counterexSynthClaim_holds : CounterexSynthClaim := by
  refine ⟨fun x => lvProbes (toVec x), fun x => lvSlope (toVec x), fun x => lvSlack (toVec x),
    fun x => lvP₀ (toVec x), fun x => lvκ (toVec x),
    transfer lvProbes_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer lvSlope_isBigO fun x => by simp only [toVec],
    transfer lvSlack_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer lvP₀_isBigO fun x => by simp only [toVec]; ring_nf,
    transfer lvκ_isBigO fun x => by simp only [toVec]; ring_nf,
    fun x => tol_pos _, ?_⟩
  intro α _ _ σ _ M side U κ Ω _ μ _ R P C p₀
  dsimp only
  intro hx hP hpre hp₀ hκ0 hκ
  simp only [counterexSynthRegion, Set.mem_ofPred_eq] at hx
  obtain ⟨-, -, hδ, hδ2, hε0, hε, hL⟩ := hx
  have hyε : C.harvestRate = C.disagreementRate / 48 := rfl
  have hy : 0 < C.harvestRate := by rw [hyε]; positivity
  simp only [toVec] at hP hp₀ hκ ⊢
  have hδ1 : C.δ < 1 := by linarith
  have hp₀0 : 0 ≤ p₀ := measureReal_nonneg.trans (hpre 1)
  obtain ⟨t, t', ht, ht', htc, hone, hrace⟩ := race_le (Fintype.card σ) (Fintype.card α)
    (iε := 1 / C.disagreementRate) (L := P.L) hδ hδ1 hκ0 hκ
  have hfin : C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
      * min 1 (Real.exp (t' * (κ * lvMN (Fintype.card σ) (Fintype.card α) C.δ)) / (1 + t))
      + Real.exp (t' * (κ * lvMN (Fintype.card σ) (Fintype.card α) C.δ)) / (1 + t) ^ 2
        ≤ C.δ / 8 := by
    have hpair := collision_le hp₀0 hP hp₀
    have h0 : 0 ≤ Real.exp (t' * (κ * lvMN (Fintype.card σ) (Fintype.card α) C.δ)) / (1 + t) := by
      positivity
    have h1 := (mul_le_mul_of_nonneg_left (min_le_right 1 _) (pairs_nonneg _ hp₀0)).trans
      (mul_le_mul_of_nonneg_right hpair h0)
    linarith
  refine le_trans (lintegral_mono fun ω => measure_mono fun draws hd => ?_)
    (level5RoundCorrect_holds M side U κ μ R P C p₀ 0 t t' hpre hδ hδ1 hy hyε.le hε
      (budget_le (Fintype.card σ) (Fintype.card α) hP) hκ0 ht ht' htc
      hfin)
  exact hd.imp (fun h hw => h (endsWell_mono (slope_le _ _ hyε)
    (slack_le _ _ hp₀0 hy hyε hδ (by norm_num) le_rfl) hw)) id

end OrthoDFA
