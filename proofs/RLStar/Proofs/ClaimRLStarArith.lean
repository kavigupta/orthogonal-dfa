import RLStar.Claims.RLStar
import RLStar.Proofs.BigO
import RLStar.Proofs.Reads
import RLStar.Proofs.LoopReadsArith

noncomputable section

namespace OrthoDFA

open Real

/-! The knobs. -/

/-- `X = L + |Q|`. -/
def knX (r : RLStarRun) : ℝ := r.L + r.Q

def knZ (r : RLStarRun) : ℝ := knX r * r.A / (r.δ * r.ξc * r.pAP * r.s₀)

/-- The log every knob's log stays below: `knob_log_le`. -/
def knΛ (r : RLStarRun) : ℝ := 400 * log (knZ r) + 30

def knLg (r : RLStarRun) : ℝ := log (knX r + 1)

def knN (r : RLStarRun) : ℝ := ((r.L + r.Q + 3 : ℕ) : ℝ) + 1 - ((r.L + 1) / 2 : ℕ)

def knε (r : RLStarRun) : ℝ := r.ξc / (4 * (r.A + 1) * knN r)

def knεcov (r : RLStarRun) : ℝ := min (r.ξc / (10 * knN r)) (r.ξc / (8 * knN r))

def knδ' (r : RLStarRun) : ℝ := r.δ / (8 * (r.Q + 1))

def knS (r : RLStarRun) : ℝ := r.Q * ((r.Q + 2) * (r.A + 1) + 1) + 2 * (r.A + 1) + 1

def knE (r : RLStarRun) : ℕ := ⌈96 * r.L / knε r⌉₊ + 1

def knB (r : RLStarRun) : ℕ := ⌈192 * knΛ r / knε r⌉₊

def knB₃ (r : RLStarRun) : ℝ :=
  36 * knS r / knε r * log (16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r) + knS r + 1

def knRp (r : RLStarRun) : ℕ := Random.Kappa.probeSifts (r.L + r.Q + 3 + r.L)

/-- A bound on the lists' slope `lazySlope` free of the list size. -/
def knSlope (r : RLStarRun) : ℝ := 2 * r.Q * (r.Q + 3) * knE r * knRp r * (r.L + 3) * knB r / log 2

def knι (r : RLStarRun) : ℝ := min (knεcov r / 8) (1 / (48 * r.Q * knSlope r))

def knη₀ (r : RLStarRun) : ℝ := 1 / 2 - r.s₀ / 2

def knτ (r : RLStarRun) : ℝ := min ((knεcov r / 2) * (1 - 2 * knη₀ r) ^ 2) (knι r)

/-- A bound on `1/ι`. -/
def knιInv (r : RLStarRun) : ℝ :=
  2 ^ 41 * r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r

/-- A bound on the prefix count `RB`. -/
def knRBb (r : RLStarRun) : ℝ := 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2

/-- A bound on `1/capP`. -/
def knCapInv (r : RLStarRun) : ℝ :=
  2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
    + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2

def knNh (r : RLStarRun) : ℕ := ⌈2 * knCapInv r⌉₊

def knJ (r : RLStarRun) : ℕ := knNh r * (r.L + 1) + 1

def knNr (r : RLStarRun) : ℕ :=
  ⌈(knE r : ℝ) * ((knB₃ r + knJ r * knB r) * (knRp r * (r.Q + 3))
    + (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2)))⌉₊

def knW (r : RLStarRun) : ℝ :=
  max (knNr r : ℝ) (((r.L + r.Q + 3 : ℕ) + 1 : ℝ) * (r.Q + 2) / (r.ξc / 4))

def knTu (r : RLStarRun) : ℝ := 1 / (4 * knSlope r)

def knPB (r : RLStarRun) : ℝ :=
  (50 * (max (log (20 / knτ r)) (log (8 * knW r / knδ' r)) + 3) / (1 / 2 - knη₀ r) ^ 2
      + 20 / (1 / 2 - knη₀ r) + 28) / r.pAP
    + 120 * log (3 / (knδ' r / 2)) / r.pAP

def knRB (r : RLStarRun) : ℝ :=
  2 ^ 22 * (r.Q + 2) * log (128 * (knPB r + 1) / knδ' r) / ((1 / 2 - knη₀ r) ^ 4 * knι r ^ 2)
    + 2 ^ 27 * log ((r.Q + 2 + 2) * knPB r
      / (knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r)) / knτ r ^ 2

def knFV (r : RLStarRun) : ℝ :=
  (50 * (max (log (20 / knτ r)) (log (8 * knW r / knδ' r)) + 3) / (1 / 2 - knη₀ r) ^ 2
    + 20 / (1 / 2 - knη₀ r) + 28) / 4

/-- `lazy_loop_core`'s cap at the knobs. -/
def knCapP (r : RLStarRun) : ℝ :=
  min (min (min ((1 / 2 - knη₀ r) * min (knεcov r / 2 * (1 / 2 - knη₀ r)) (knι r)
          / (2 ^ 20 * (knRB r + 1)))
        (knι r ^ 2 / (2 ^ 28 * Real.log (128 * (r.Q + 2) * (knPB r + 1)
          * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r))))
      (knι r / 2 / (8 * (r.Q + 2) * (knRB r + 1))))
    (min (min (r.pAP / (2 ^ 18 * (knPB r + 1))) (knι r / (2 ^ 24 * (r.Q + 2) ^ 2 * (knRB r + 1))))
      (min (r.pAP / (2 * (r.Q + 2) * (knPB r + 2))) (1 / (6 * (r.Q + 2) * (knRB r + 1)))))

/-- The misread term's bound on `p₀`. -/
def knMcD (r : RLStarRun) : ℝ := r.ξc ^ 2 * r.s₀ ^ 2 / (2 ^ 16 * knΛ r * knX r * log r.A)

def knP₀ (r : RLStarRun) : ℝ :=
  min (min (knε r / (1536 * knCapInv r)) (1 / (12 * (r.Q + 2) * knRBb r)))
    (min (knε r / (768 * (2 * knCapInv r + 1))) (knMcD r))

/-! The totals. -/

def knPref (r : RLStarRun) : ℝ := (r.Q + 1) * (6 * (r.Q + 2) * knRB r)

def knSuff (r : RLStarRun) : ℝ := (r.Q + 1) * (knPB r + 1)

def knProbes (r : RLStarRun) : ℝ := (r.Q + 1) * (knE r * (knB₃ r + knJ r * knB r))

def knTrie (r : RLStarRun) : ℝ := 6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r)

def knReads (r : RLStarRun) : ℝ :=
  lazyReads (r.Q + 1) (6 * (r.Q + 2) * knRB r) (knPB r + 1) (knFV r) (knNr r) r.A
    (r.L * (6 * (r.Q + 2) * knRB r + 2 * (knNh r + knB r))) r.Q

/-! The fixed point: a log below its own argument's log. -/

/-- `log (P Λ^m) ≤ Λ` once `Λ ≥ 2 log P + 2m log(2m)`. -/
lemma log_mul_pow_le {P Λ : ℝ} {m : ℕ} (hP : 0 < P) (hm : 0 < m) (hΛ : 0 < Λ)
    (h : 2 * log P + 2 * m * log (2 * m) ≤ Λ) : log (P * Λ ^ m) ≤ Λ := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have h1 := Real.log_le_sub_one_of_pos (show 0 < Λ / (2 * m) by positivity)
  rw [Real.log_div hΛ.ne' (by positivity)] at h1
  rw [Real.log_mul hP.ne' (by positivity), Real.log_pow]
  have h2 : (m : ℝ) * log Λ ≤ m * (Λ / (2 * m) - 1 + log (2 * m)) :=
    mul_le_mul_of_nonneg_left (by linarith) hm'.le
  have e : (m : ℝ) * (Λ / (2 * m) - 1 + log (2 * m)) = Λ / 2 - m + m * log (2 * m) := by
    field_simp
  linarith

/-! The run's coordinates. -/

section base

variable {r : RLStarRun}

lemma ξc_mem (hr : r ∈ rlStarRunRegion) :
    0 < r.ξc ∧ r.ξc ≤ 1 ∧ r.ξc ≤ r.ξ ∧ r.ξc ≤ (r.L + r.Q) * r.β := by
  obtain ⟨hQ, -, -, -, hξ, hξ1, -, -, -, -, hβ⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  exact ⟨lt_min hξ (mul_pos (by linarith) hβ), (min_le_left _ _).trans hξ1, min_le_left _ _,
    min_le_right _ _⟩

lemma kn_base (hr : r ∈ rlStarRunRegion) :
    1 ≤ (r.Q : ℝ) ∧ 2 ≤ (r.A : ℝ) ∧ 1 ≤ knX r ∧ (r.Q : ℝ) ≤ knX r ∧ (r.L : ℝ) ≤ knX r
      ∧ 0 < r.δ ∧ r.δ ≤ 1 / 2 ∧ 0 < r.ξc ∧ r.ξc ≤ 1 ∧ 0 < r.pAP ∧ r.pAP ≤ 1
      ∧ 0 < r.s₀ ∧ r.s₀ ≤ 1 / 2 := by
  obtain ⟨hξ, hξ1, -, -⟩ := ξc_mem hr
  obtain ⟨hQ, hA, hδ, hδ2, -, -, hp, hp1, hs, hs1, -⟩ := hr
  have hQ' : (1 : ℝ) ≤ r.Q := by exact_mod_cast hQ
  have hA' : (2 : ℝ) ≤ r.A := by exact_mod_cast hA
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  unfold knX
  exact ⟨hQ', hA', by linarith, by linarith, by linarith, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩

lemma knN_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ knN r ∧ knN r ≤ 5 * knX r ∧ knX r ≤ 2 * knN r := by
  obtain ⟨hQ, -, hX, -⟩ := kn_base hr
  have h2 : 2 * ((r.L + 1) / 2) ≤ r.L + 1 := Nat.mul_div_le (r.L + 1) 2
  have h2' : (2 : ℝ) * (((r.L + 1) / 2 : ℕ) : ℝ) ≤ r.L + 1 := by exact_mod_cast h2
  have h0 : (0 : ℝ) ≤ (((r.L + 1) / 2 : ℕ) : ℝ) := Nat.cast_nonneg _
  have hL : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have e : knN r = (r.L : ℝ) + r.Q + 3 + 1 - (((r.L + 1) / 2 : ℕ) : ℝ) := by
    unfold knN; push_cast; ring
  rw [e]
  unfold knX at hX ⊢
  exact ⟨by linarith, by linarith, by linarith⟩

lemma knε_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knε r ∧ knε r ≤ r.ξc / 12 ∧ 1 / knε r ≤ 30 * r.A * knX r * (1 / r.ξc) := by
  obtain ⟨hN1, hN5, -⟩ := knN_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  have hN0 : 0 < knN r := by linarith
  unfold knε
  refine ⟨by positivity, ?_, ?_⟩
  · rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    have : (12 : ℝ) ≤ 4 * (r.A + 1) * knN r := by nlinarith
    nlinarith
  · have h : 4 * ((r.A : ℝ) + 1) * knN r ≤ 30 * r.A * knX r := by nlinarith
    calc 1 / (r.ξc / (4 * (r.A + 1) * knN r)) = 4 * (r.A + 1) * knN r * (1 / r.ξc) := by
          field_simp
      _ ≤ 30 * r.A * knX r * (1 / r.ξc) := by gcongr

lemma knεcov_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knεcov r ∧ knεcov r = r.ξc / (10 * knN r) ∧ 1 / knεcov r ≤ 50 * knX r * (1 / r.ξc)
      ∧ 2 * knεcov r ≤ r.ξc / knX r ∧ knεcov r ≤ 1 / 10 := by
  obtain ⟨hN1, hN5, hXN⟩ := knN_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have hN0 : 0 < knN r := by linarith
  have e : knεcov r = r.ξc / (10 * knN r) := by
    unfold knεcov
    exact min_eq_left (div_le_div_of_nonneg_left hξ.le (by positivity) (by linarith))
  rw [e]
  refine ⟨by positivity, rfl, ?_, ?_, ?_⟩
  · calc 1 / (r.ξc / (10 * knN r)) = 10 * knN r * (1 / r.ξc) := by field_simp
      _ ≤ 50 * knX r * (1 / r.ξc) := mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  · rw [mul_div_assoc', div_le_div_iff₀ (by positivity) (by linarith)]
    nlinarith
  · rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith

lemma knδ'_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knδ' r ∧ knδ' r ≤ 1 / 32 ∧ 1 / knδ' r ≤ 16 * r.Q * (1 / r.δ) := by
  obtain ⟨hQ, -, -, -, -, hδ, hδ2, -⟩ := kn_base hr
  unfold knδ'
  refine ⟨by positivity, ?_, ?_⟩
  · rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    nlinarith
  · calc 1 / (r.δ / (8 * (r.Q + 1))) = 8 * (r.Q + 1) * (1 / r.δ) := by field_simp
      _ ≤ 16 * r.Q * (1 / r.δ) := mul_le_mul_of_nonneg_right (by linarith) (by positivity)

/-- Every coordinate is at most `Z`. -/
lemma knZ_ge (hr : r ∈ rlStarRunRegion) :
    8 ≤ knZ r ∧ knX r ≤ knZ r ∧ (r.A : ℝ) ≤ knZ r ∧ 1 / r.δ ≤ knZ r ∧ 1 / r.ξc ≤ knZ r
      ∧ 1 / r.pAP ≤ knZ r ∧ 1 / r.s₀ ≤ knZ r ∧ (r.Q : ℝ) ≤ knZ r := by
  obtain ⟨hQ, hA, hX, hQX, -, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := kn_base hr
  have iδ : 2 ≤ 1 / r.δ := by rw [le_div_iff₀ hδ]; linarith
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have ip : 1 ≤ 1 / r.pAP := by rw [le_div_iff₀ hp]; linarith
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  set a := knX r * r.A with ha
  set b := (1 / r.δ) * (1 / r.ξc) with hb
  set c := (1 / r.pAP) * (1 / r.s₀) with hc
  have e : knZ r = a * b * c := by rw [ha, hb, hc]; unfold knZ; field_simp
  have ha2 : 2 ≤ a := by rw [ha]; nlinarith
  have hb2 : 2 ≤ b := by rw [hb]; nlinarith
  have hc2 : 2 ≤ c := by rw [hc]; nlinarith
  have hXa : knX r ≤ a := le_mul_of_one_le_right (by linarith) (by linarith)
  have hAa : (r.A : ℝ) ≤ a := le_mul_of_one_le_left (by linarith) hX
  have hδb : 1 / r.δ ≤ b := le_mul_of_one_le_right (by linarith) iξ
  have hξb : 1 / r.ξc ≤ b := le_mul_of_one_le_left (by linarith) (by linarith)
  have hpc : 1 / r.pAP ≤ c := le_mul_of_one_le_right (by linarith) (by linarith)
  have hsc : 1 / r.s₀ ≤ c := le_mul_of_one_le_left (by linarith) ip
  have haZ : a ≤ knZ r := by
    rw [e, mul_assoc]; exact le_mul_of_one_le_right (by linarith) (by nlinarith)
  have hbZ : b ≤ knZ r := by
    rw [e, mul_comm a b, mul_assoc]; exact le_mul_of_one_le_right (by linarith) (by nlinarith)
  have hcZ : c ≤ knZ r := by
    rw [e]; exact le_mul_of_one_le_left (by linarith) (by nlinarith)
  have h8 : (8 : ℝ) ≤ a * b * c :=
    calc (8 : ℝ) = 2 * 2 * 2 := by norm_num
      _ ≤ a * b * c := by gcongr
  refine ⟨by rw [e]; exact h8, by linarith, by linarith, by linarith, by linarith, by linarith,
    by linarith, by linarith⟩

lemma knΛ_bounds (hr : r ∈ rlStarRunRegion) : 30 ≤ knΛ r ∧ 0 ≤ log (knZ r) := by
  have h := (knZ_ge hr).1
  have : 0 ≤ log (knZ r) := Real.log_nonneg (by linarith)
  unfold knΛ
  exact ⟨by linarith, this⟩

lemma knLg_bounds (hr : r ∈ rlStarRunRegion) : 1 / 2 ≤ knLg r ∧ knLg r ≤ knX r := by
  obtain ⟨-, -, hX, -⟩ := kn_base hr
  unfold knLg
  refine ⟨half_le_log (by linarith), ?_⟩
  have := Real.log_le_sub_one_of_pos (show 0 < knX r + 1 by linarith)
  linarith

/-- `log y ≤ Λ` once `y ≤ Z²⁰⁰ Λ⁵`. -/
lemma knob_log_le (hr : r ∈ rlStarRunRegion) {y : ℝ} (hy : 0 < y)
    (h : y ≤ knZ r ^ 200 * knΛ r ^ 5) : log y ≤ knΛ r := by
  obtain ⟨hZ8, -⟩ := knZ_ge hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  refine (Real.log_le_log hy h).trans (log_mul_pow_le (by positivity) (by norm_num)
    (by linarith) ?_)
  rw [Real.log_pow]
  have h10 : log (2 * ((5 : ℕ) : ℝ)) ≤ 3 := by
    rw [Real.log_le_iff_le_exp (by norm_num)]
    have := Real.exp_one_gt_d9
    have e : Real.exp 3 = Real.exp 1 ^ 3 := by rw [← Real.exp_nat_mul]; norm_num
    rw [e]
    have h3 := pow_lt_pow_left₀ this (by norm_num) (by norm_num : (3 : ℕ) ≠ 0)
    push_cast
    norm_num at h3 ⊢
    linarith
  unfold knΛ
  push_cast at h10 ⊢
  nlinarith

/-- A monomial bound `c Zⁿ Λʲ` within `Z²⁰⁰ Λ⁵`. -/
lemma knob_pow_le (hr : r ∈ rlStarRunRegion) {y c : ℝ} {n j k : ℕ} (hc : c ≤ 8 ^ k)
    (hy : y ≤ c * knZ r ^ n * knΛ r ^ j) (hnk : n + k ≤ 200) (hj : j ≤ 5) :
    y ≤ knZ r ^ 200 * knΛ r ^ 5 := by
  obtain ⟨hZ8, -⟩ := knZ_ge hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hZ1 : 1 ≤ knZ r := by linarith
  have h1 : c * knZ r ^ n ≤ knZ r ^ 200 :=
    calc c * knZ r ^ n ≤ knZ r ^ k * knZ r ^ n := by
          gcongr
          exact hc.trans (pow_le_pow_left₀ (by norm_num) hZ8 k)
      _ = knZ r ^ (n + k) := by rw [← pow_add, add_comm]
      _ ≤ knZ r ^ 200 := pow_le_pow_right₀ hZ1 hnk
  have h2 : knΛ r ^ j ≤ knΛ r ^ 5 := pow_le_pow_right₀ (by linarith) hj
  calc y ≤ c * knZ r ^ n * knΛ r ^ j := hy
    _ ≤ knZ r ^ 200 * knΛ r ^ 5 := by gcongr

end base

/-! The knobs' bounds, in the inverse coordinates `1/ξc`, `1/δ`, `1/pAP`, `1/s₀`. -/

section bounds

variable {r : RLStarRun}

lemma knE_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ (knE r : ℝ) ∧ (knE r : ℝ) ≤ 2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)
      ∧ ∀ k : ℕ, k ≤ r.L → 2 * (k : ℝ) / (knε r / 48) + 1 ≤ knE r := by
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hQ, hA, hX, -, hLX, -, -, hξ, hξ1, -⟩ := kn_base hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hL0 : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hx : 0 ≤ 96 * (r.L : ℝ) / knε r := by positivity
  have hc := Nat.ceil_lt_add_one hx
  have hc' := Nat.le_ceil (96 * (r.L : ℝ) / knε r)
  have hc0 : (0 : ℝ) ≤ ⌈96 * (r.L : ℝ) / knε r⌉₊ := Nat.cast_nonneg _
  have e : (knE r : ℝ) = ⌈96 * (r.L : ℝ) / knε r⌉₊ + 1 := by unfold knE; push_cast; ring
  have h96 : 96 * (r.L : ℝ) / knε r ≤ 2880 * r.A * knX r ^ 2 * (1 / r.ξc) :=
    calc 96 * (r.L : ℝ) / knε r = 96 * r.L * (1 / knε r) := by ring
      _ ≤ 96 * knX r * (30 * r.A * knX r * (1 / r.ξc)) := by gcongr
      _ = 2880 * r.A * knX r ^ 2 * (1 / r.ξc) := by ring
  have hm : 2 ≤ r.A * knX r ^ 2 * (1 / r.ξc) := by
    have h1 : 1 ≤ knX r ^ 2 := one_le_pow₀ hX
    have h2 : 1 ≤ knX r ^ 2 * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le h1 iξ
    nlinarith
  refine ⟨by rw [e]; linarith, by rw [e]; nlinarith, fun k hk => ?_⟩
  have hk' : (k : ℝ) ≤ r.L := by exact_mod_cast hk
  rw [e, div_div_eq_mul_div]
  have : 2 * (k : ℝ) * 48 / knε r ≤ 96 * r.L / knε r :=
    div_le_div_of_nonneg_right (by linarith) hε.le
  linarith

lemma knB_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ (knB r : ℝ) ∧ (knB r : ℝ) ≤ 2 ^ 13 * r.A * knX r * (1 / r.ξc) * knΛ r
      ∧ 192 * knΛ r / knε r ≤ knB r := by
  obtain ⟨hε, hε12, hiε⟩ := knε_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hx : 0 ≤ 192 * knΛ r / knε r := by positivity
  have hc := Nat.ceil_lt_add_one hx
  have hc' := Nat.le_ceil (192 * knΛ r / knε r)
  have h1 : 1 ≤ 192 * knΛ r / knε r := by rw [le_div_iff₀ hε]; linarith
  have h192 : 192 * knΛ r / knε r ≤ 5760 * r.A * knX r * (1 / r.ξc) * knΛ r :=
    calc 192 * knΛ r / knε r = 192 * knΛ r * (1 / knε r) := by ring
      _ ≤ 192 * knΛ r * (30 * r.A * knX r * (1 / r.ξc)) := by gcongr
      _ = 5760 * r.A * knX r * (1 / r.ξc) * knΛ r := by ring
  have hm : 1 ≤ r.A * knX r * (1 / r.ξc) * knΛ r := by
    have h1 : 1 ≤ r.A * knX r := by nlinarith
    have h2 : 1 ≤ r.A * knX r * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le h1 iξ
    exact one_le_mul_of_one_le_of_one_le h2 (by linarith)
  unfold knB
  refine ⟨h1.trans hc', by linarith, hc'⟩

lemma knS_bounds (hr : r ∈ rlStarRunRegion) : 1 ≤ knS r ∧ knS r ≤ 9 * r.Q ^ 2 * r.A := by
  obtain ⟨hQ, hA, -⟩ := kn_base hr
  have h1 : (r.Q : ℝ) ≤ r.Q ^ 2 := le_self_pow₀ hQ (by norm_num)
  have h0 : (0 : ℝ) ≤ r.Q ^ 2 := by positivity
  have h2 : (r.Q : ℝ) * r.A ≤ r.Q ^ 2 * r.A := mul_le_mul_of_nonneg_right h1 (by linarith)
  have h3 : (r.Q : ℝ) ^ 2 * 2 ≤ r.Q ^ 2 * r.A := mul_le_mul_of_nonneg_left hA h0
  have h4 : (r.A : ℝ) ≤ r.Q ^ 2 * r.A :=
    le_mul_of_one_le_left (by linarith) (one_le_pow₀ hQ)
  unfold knS
  constructor
  · nlinarith
  · linarith

lemma knLg_ge_log_two (hr : r ∈ rlStarRunRegion) : log 2 ≤ knLg r := by
  obtain ⟨-, -, hX, -⟩ := kn_base hr
  exact Real.log_le_log (by norm_num) (by linarith)

lemma knRp_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ (knRp r : ℝ) ∧ (knRp r : ℝ) ≤ 16 * knLg r := by
  obtain ⟨hQ, -, hX, -, hLX, -⟩ := kn_base hr
  have hl2 := knLg_ge_log_two hr
  have hlt := Real.log_two_gt_d9
  set n := r.L + r.Q + 3 + r.L with hn
  have hn1 : 1 < n := by omega
  have hrp : knRp r = min (n + 1) (3 + Nat.clog 2 n) := rfl
  have hge : 1 ≤ knRp r := by rw [hrp]; exact le_min (by omega) (by omega)
  refine ⟨by exact_mod_cast hge, ?_⟩
  have hle : knRp r ≤ 3 + Nat.clog 2 n := by rw [hrp]; exact min_le_right _ _
  have hle' : (knRp r : ℝ) ≤ 3 + (Nat.clog 2 n : ℝ) := by exact_mod_cast hle
  have h1 := Nat.pow_pred_clog_lt_self (by norm_num : 1 < 2) hn1
  set c := Nat.clog 2 n
  have hc : c ≤ (c - 1) + 1 := by omega
  have hc' : (c : ℝ) ≤ ((c - 1 : ℕ) : ℝ) + 1 := by exact_mod_cast hc
  have h2 : (2 : ℝ) ^ (c - 1) < (n : ℝ) := by
    rw [Nat.pred_eq_sub_one] at h1
    exact_mod_cast h1
  have h3 : ((c - 1 : ℕ) : ℝ) * log 2 < log (n : ℝ) := by
    rw [← Real.log_pow]
    exact Real.log_lt_log (by positivity) h2
  have hnX : (n : ℝ) ≤ 2 ^ 2 * (knX r + 1) := by
    have : (n : ℝ) = 2 * r.L + r.Q + 3 := by rw [hn]; push_cast; ring
    unfold knX at hLX ⊢
    rw [this]; linarith
  have h4 : log (n : ℝ) ≤ 2 * log 2 + knLg r := by
    calc log (n : ℝ) ≤ log (2 ^ 2 * (knX r + 1)) := Real.log_le_log (by positivity) hnX
      _ = 2 * log 2 + knLg r := by
        rw [Real.log_mul (by norm_num) (by linarith), Real.log_pow]; push_cast; rfl
  -- `(3 + c) log 2 < 6 log 2 + lg ≤ 7 lg`
  have h5 : (3 + (c : ℝ)) * log 2 ≤ 7 * knLg r := by nlinarith
  have h6 : 3 + (c : ℝ) ≤ 16 * knLg r := by
    by_contra hcon
    push Not at hcon
    nlinarith
  linarith

lemma knSlope_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knSlope r ∧ knSlope r
      ≤ 2 ^ 35 * r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r := by
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hb1, hb, -⟩ := knB_bounds hr
  obtain ⟨hrp1, hrp⟩ := knRp_bounds hr
  obtain ⟨hQ, hA, hX, hQX, hLX, -, -, hξ, hξ1, -⟩ := kn_base hr
  have hlt := Real.log_two_gt_d9
  have hl0 : 0 < log 2 := by linarith
  have hil : 1 / log 2 ≤ 3 / 2 := by rw [div_le_div_iff₀ hl0 (by norm_num)]; linarith
  have hL0 : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hXL : (r.L : ℝ) + 3 ≤ 3 * knX r := by unfold knX at *; linarith
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hlg0 : 0 < knLg r := by linarith [(knLg_bounds hr).1]
  have e : knSlope r = 2 * r.Q * (r.Q + 3) * knE r * knRp r * (r.L + 3) * knB r * (1 / log 2) := by
    unfold knSlope; ring
  rw [e]
  refine ⟨by positivity, ?_⟩
  calc 2 * (r.Q : ℝ) * (r.Q + 3) * knE r * knRp r * (r.L + 3) * knB r * (1 / log 2)
      ≤ 2 * r.Q * (4 * r.Q) * (2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)) * (16 * knLg r)
        * (3 * knX r) * (2 ^ 13 * r.A * knX r * (1 / r.ξc) * knΛ r) * (3 / 2) := by
        gcongr
        linarith
    _ = 576 * 2 ^ 25 * r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r := by
        ring
    _ ≤ 2 ^ 35 * r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r := by
        have hΛ := (knΛ_bounds hr).1
        have hlg := (knLg_bounds hr).1
        gcongr
        norm_num

lemma knιInv_ge (hr : r ∈ rlStarRunRegion) : 2 ^ 46 * knX r * (1 / r.ξc) ≤ knιInv r := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨hlg, -⟩ := knLg_bounds hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  unfold knιInv
  calc 2 ^ 46 * knX r * (1 / r.ξc) ≤ 2 ^ 41 * 1 * 4 * knX r * (1 / r.ξc) * 30 * (1 / 2) := by
        nlinarith [mul_pos (by linarith : (0 : ℝ) < knX r) (by linarith : (0 : ℝ) < 1 / r.ξc)]
    _ ≤ 2 ^ 41 * r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r := by
        gcongr
        · exact one_le_pow₀ hQ
        · nlinarith
        · exact le_self_pow₀ hX (by norm_num)
        · exact le_self_pow₀ iξ (by norm_num)

lemma knι_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knι r ∧ knι r ≤ knεcov r / 8 ∧ 48 * r.Q * knSlope r * knι r ≤ 1
      ∧ 1 / knι r ≤ knιInv r ∧ knι r ≤ 1 / 80 := by
  obtain ⟨hεc, -, hiεc, -, hεc1⟩ := knεcov_bounds hr
  obtain ⟨hsl, hsl'⟩ := knSlope_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have hιI := knιInv_ge hr
  have h0 : 0 < 48 * (r.Q : ℝ) * knSlope r := by positivity
  have hι0 : 0 < knι r := lt_min (by positivity) (by positivity)
  have hι8 : knι r ≤ knεcov r / 8 := min_le_left _ _
  have hι48 : knι r ≤ 1 / (48 * r.Q * knSlope r) := min_le_right _ _
  refine ⟨hι0, hι8, ?_, ?_, by linarith⟩
  · rw [le_div_iff₀ h0] at hι48; linarith
  · calc 1 / knι r ≤ 1 / (knεcov r / 8) + 1 / (1 / (48 * r.Q * knSlope r)) :=
          one_div_min_le (by positivity) (by positivity)
      _ = 8 * (1 / knεcov r) + 48 * r.Q * knSlope r := by rw [one_div_one_div]; ring
      _ ≤ 8 * (50 * knX r * (1 / r.ξc))
          + 48 * r.Q * (2 ^ 35 * r.Q ^ 2 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r
            * knLg r) := by gcongr
      _ = 400 * knX r * (1 / r.ξc) + 3 / 4 * knιInv r := by unfold knιInv; ring
      _ ≤ knιInv r := by nlinarith [mul_pos (by linarith : (0 : ℝ) < knX r) (one_div_pos.2 hξ)]

lemma knτ_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knτ r ∧ knτ r ≤ knι r ∧ 1 / knτ r ≤ 2 * knιInv r * (1 / r.s₀) ^ 2 := by
  obtain ⟨hεc, -, hiεc, -, -⟩ := knεcov_bounds hr
  obtain ⟨hι0, -, -, hιI, -⟩ := knι_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  have hιI' := knιInv_ge hr
  have e : 1 - 2 * knη₀ r = r.s₀ := by unfold knη₀; ring
  have is : 1 ≤ (1 / r.s₀) ^ 2 := one_le_pow₀ (by rw [le_div_iff₀ hs]; linarith)
  have hτ : knτ r = min (knεcov r / 2 * r.s₀ ^ 2) (knι r) := by unfold knτ; rw [e]
  rw [hτ]
  refine ⟨lt_min (by positivity) hι0, min_le_right _ _, ?_⟩
  calc 1 / min (knεcov r / 2 * r.s₀ ^ 2) (knι r)
      ≤ 1 / (knεcov r / 2 * r.s₀ ^ 2) + 1 / knι r := one_div_min_le (by positivity) hι0
    _ = 2 * (1 / knεcov r) * (1 / r.s₀) ^ 2 + 1 / knι r := by field_simp
    _ ≤ 2 * (50 * knX r * (1 / r.ξc)) * (1 / r.s₀) ^ 2 + knιInv r := by gcongr
    _ ≤ knιInv r * (1 / r.s₀) ^ 2 + knιInv r * (1 / r.s₀) ^ 2 := by
        gcongr
        · nlinarith [mul_pos (by linarith : (0 : ℝ) < knX r) (one_div_pos.2 hξ)]
        · nlinarith [mul_pos (by linarith : (0 : ℝ) < knX r) (one_div_pos.2 hξ)]
    _ = 2 * knιInv r * (1 / r.s₀) ^ 2 := by ring

end bounds

/-! Every knob's log below `Λ`, and the knobs that sit inside those logs. -/

section logs

variable {r : RLStarRun}

lemma knιInv_ge' (hr : r ∈ rlStarRunRegion) :
    2 ^ 44 * r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 ≤ knιInv r := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨hlg, -⟩ := knLg_bounds hr
  have h : 0 ≤ r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 := by positivity
  have hl : 8 ≤ knΛ r * knLg r := by nlinarith
  have e : knιInv r = 2 ^ 41 * (r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2)
      * (knΛ r * knLg r) := by unfold knιInv; ring
  rw [e]
  nlinarith

lemma knιInv_Z (hr : r ∈ rlStarRunRegion) : knιInv r ≤ 2 ^ 41 * knZ r ^ 12 * knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, -, hξZ, -, -, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨hlg, hlgX⟩ := knLg_bounds hr
  unfold knιInv
  calc 2 ^ 41 * r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r
      ≤ 2 ^ 41 * knZ r ^ 3 * knZ r ^ 2 * knZ r ^ 4 * knZ r ^ 2 * knΛ r * knZ r := by
        gcongr
        linarith
    _ = 2 ^ 41 * knZ r ^ 12 * knΛ r := by ring

lemma knL1 (hr : r ∈ rlStarRunRegion) :
    log (16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r) ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, -, -, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -⟩ := kn_base hr
  obtain ⟨hS1, hS⟩ := knS_bounds hr
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hδ', -, hiδ'⟩ := knδ'_bounds hr
  refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 2 ^ 45) (n := 16) (j := 0)
    (k := 15) (by norm_num) ?_ (by norm_num) (by norm_num))
  calc 16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r
      = 16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) * (1 / knδ' r) := by ring
    _ ≤ 16 * (9 * r.Q ^ 2 * r.A) * (10 * r.Q ^ 2 * r.A)
        * ((2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)) * (2 ^ 13 * r.A * knX r ^ 2 * (1 / r.ξc)))
        * (16 * r.Q * (1 / r.δ)) := by
        gcongr
        · nlinarith
        · nlinarith
    _ ≤ 16 * (9 * knZ r ^ 2 * knZ r) * (10 * knZ r ^ 2 * knZ r)
        * ((2 ^ 12 * knZ r * knZ r ^ 2 * knZ r) * (2 ^ 13 * knZ r * knZ r ^ 2 * knZ r))
        * (16 * knZ r * knZ r) := by gcongr
    _ = 23040 * 2 ^ 25 * knZ r ^ 16 * knΛ r ^ 0 := by ring
    _ ≤ 2 ^ 45 * knZ r ^ 16 * knΛ r ^ 0 := by gcongr; norm_num

lemma knB₃_bounds (hr : r ∈ rlStarRunRegion) :
    0 ≤ knB₃ r ∧ knB₃ r ≤ 2 ^ 14 * r.Q ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc) * knΛ r := by
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hS1, hS⟩ := knS_bounds hr
  obtain ⟨hE1, -, -⟩ := knE_bounds hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hδ', hδ'1, -⟩ := knδ'_bounds hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hl := knL1 hr
  have hl0 : 0 ≤ log (16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ']
    have h1 : (1 : ℝ) ≤ knS r * (knS r + 1) := one_le_mul_of_one_le_of_one_le hS1 (by linarith)
    have h2 : (1 : ℝ) ≤ knE r * (knE r + 1) := one_le_mul_of_one_le_of_one_le hE1 (by linarith)
    have h3 := one_le_mul_of_one_le_of_one_le h1 h2
    linarith
  have e : knB₃ r = 36 * knS r * (1 / knε r)
      * log (16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r) + knS r + 1 := by
    unfold knB₃; ring
  rw [e]
  refine ⟨by positivity, ?_⟩
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  calc 36 * knS r * (1 / knε r)
        * log (16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r) + knS r + 1
      ≤ 36 * (9 * r.Q ^ 2 * r.A) * (30 * r.A * knX r * (1 / r.ξc)) * knΛ r
        + 9 * r.Q ^ 2 * r.A + 1 := by gcongr
    _ ≤ 2 ^ 14 * r.Q ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc) * knΛ r := by
        have h1 : (1 : ℝ) ≤ r.Q ^ 2 := one_le_pow₀ hQ
        have h2 : (2 : ℝ) ≤ r.Q ^ 2 * r.A := by nlinarith
        have h3 : 1 ≤ knX r * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le hX iξ
        have h4 : (2 : ℝ) ≤ r.Q ^ 2 * r.A * (knX r * (1 / r.ξc)) := by nlinarith
        have h5 : (4 : ℝ) ≤ r.Q ^ 2 * r.A * (knX r * (1 / r.ξc)) * r.A := by nlinarith
        have h6 : (4 : ℝ) * 30 ≤ r.Q ^ 2 * r.A * (knX r * (1 / r.ξc)) * r.A * knΛ r :=
          mul_le_mul h5 hΛ (by norm_num) (by linarith)
        nlinarith

lemma knιInv_one (hr : r ∈ rlStarRunRegion) : 1 ≤ knιInv r := by
  obtain ⟨-, -, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have h := knιInv_ge hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  nlinarith [one_le_mul_of_one_le_of_one_le hX iξ]

lemma knL2 (hr : r ∈ rlStarRunRegion) : log (20 / knτ r) ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hτ, -, hiτ⟩ := knτ_bounds hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, -, -, hs, -⟩ := kn_base hr
  have hι := knιInv_Z hr
  have hι0 : 0 ≤ knιInv r := by linarith [knιInv_one hr]
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 40 * 2 ^ 41) (n := 14) (j := 1)
    (k := 16) (by norm_num) ?_ (by norm_num) (by norm_num))
  calc 20 / knτ r = 20 * (1 / knτ r) := by ring
    _ ≤ 20 * (2 * (2 ^ 41 * knZ r ^ 12 * knΛ r) * knZ r ^ 2) := by
        gcongr; exact hiτ.trans (by gcongr)
    _ = 40 * 2 ^ 41 * knZ r ^ 14 * knΛ r ^ 1 := by ring

lemma knL4 (hr : r ∈ rlStarRunRegion) :
    log (3 / (knδ' r / 2)) ≤ knΛ r ∧ log (r.Q / (knδ' r / 4)) ≤ knΛ r
      ∧ log (16 / r.δ) ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, -, -, -, -, hδ, -⟩ := kn_base hr
  obtain ⟨hδ', -, hiδ'⟩ := knδ'_bounds hr
  refine ⟨knob_log_le hr (by positivity) (knob_pow_le hr (c := 96) (n := 2) (j := 0) (k := 3)
    (by norm_num) ?_ (by norm_num) (by norm_num)),
    knob_log_le hr (by positivity) (knob_pow_le hr (c := 64) (n := 3) (j := 0) (k := 2)
    (by norm_num) ?_ (by norm_num) (by norm_num)),
    knob_log_le hr (by positivity) (knob_pow_le hr (c := 16) (n := 1) (j := 0) (k := 2)
    (by norm_num) ?_ (by norm_num) (by norm_num))⟩
  · calc 3 / (knδ' r / 2) = 6 * (1 / knδ' r) := by field_simp; norm_num
      _ ≤ 6 * (16 * knZ r * knZ r) := by gcongr; exact hiδ'.trans (by gcongr)
      _ = 96 * knZ r ^ 2 * knΛ r ^ 0 := by ring
  · calc r.Q / (knδ' r / 4) = 4 * r.Q * (1 / knδ' r) := by field_simp
      _ ≤ 4 * knZ r * (16 * knZ r * knZ r) := by gcongr; exact hiδ'.trans (by gcongr)
      _ = 64 * knZ r ^ 3 * knΛ r ^ 0 := by ring
  · calc 16 / r.δ = 16 * (1 / r.δ) := by ring
      _ ≤ 16 * knZ r := by gcongr
      _ = 16 * knZ r ^ 1 * knΛ r ^ 0 := by ring

end logs

/-! The list size and what grows with it. -/

section lists

variable {r : RLStarRun}

lemma knCapInv_bounds (hr : r ∈ rlStarRunRegion) :
    2 ^ 64 * r.Q ^ 3 * knΛ r * knιInv r ≤ knCapInv r
      ∧ knCapInv r ≤ 2 ^ 185 * knZ r ^ 43 * knΛ r ^ 4 := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hι1 := knιInv_one hr
  have hιZ := knιInv_Z hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hZ1 : 1 ≤ knZ r := by linarith
  constructor
  · have h1 : knιInv r ≤ knιInv r ^ 3 := le_self_pow₀ hι1 (by norm_num)
    have h2 : 16 ≤ (1 / r.s₀) ^ 4 := by nlinarith [pow_le_pow_left₀ (by norm_num) is 4]
    have h3 : r.Q ^ 2 ≤ (1 / r.s₀) ^ 2 + r.Q ^ 2 := by nlinarith
    unfold knCapInv
    have h4 : 0 ≤ 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
    calc 2 ^ 64 * r.Q ^ 3 * knΛ r * knιInv r
        = 2 ^ 60 * r.Q * knΛ r * 16 * knιInv r * r.Q ^ 2 := by ring
      _ ≤ 2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2) := by
          gcongr
      _ ≤ _ := le_add_of_nonneg_right h4
  · unfold knCapInv
    calc 2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
          + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2
        ≤ 2 ^ 60 * knZ r * knΛ r * knZ r ^ 4 * (2 ^ 41 * knZ r ^ 12 * knΛ r) ^ 3
            * (knZ r ^ 2 + knZ r ^ 2)
          + 2 ^ 30 * knZ r * knΛ r * knZ r ^ 2 * knZ r ^ 2 := by
          gcongr
      _ = 2 ^ 184 * knZ r ^ 43 * knΛ r ^ 4 + 2 ^ 30 * knZ r ^ 5 * knΛ r := by ring
      _ ≤ 2 ^ 185 * knZ r ^ 43 * knΛ r ^ 4 := by
          have h1 : knZ r ^ 5 ≤ knZ r ^ 43 := pow_le_pow_right₀ hZ1 (by norm_num)
          have h2 : knΛ r ≤ knΛ r ^ 4 := le_self_pow₀ (by linarith) (by norm_num)
          have h3 : knZ r ^ 5 * knΛ r ≤ knZ r ^ 43 * knΛ r ^ 4 :=
            mul_le_mul h1 h2 (by linarith) (pow_nonneg (by linarith) _)
          have h5 : 0 ≤ knZ r ^ 43 * knΛ r ^ 4 :=
            mul_nonneg (pow_nonneg (by linarith) _) (pow_nonneg (by linarith) _)
          linarith

lemma knNh_bounds (hr : r ∈ rlStarRunRegion) :
    2 * knCapInv r ≤ knNh r ∧ (knNh r : ℝ) ≤ 2 * knCapInv r + 1 ∧ 1 ≤ (knNh r : ℝ)
      ∧ knB₃ r + (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2)) + 1 ≤ knNh r
      ∧ 6 * r.Q * knΛ r ≤ knNh r ∧ (knNh r : ℝ) ≤ 2 ^ 187 * knZ r ^ 43 * knΛ r ^ 4 := by
  obtain ⟨hcap, hcapZ⟩ := knCapInv_bounds hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨hZ8, -⟩ := knZ_ge hr
  obtain ⟨-, hB₃⟩ := knB₃_bounds hr
  have hι := knιInv_ge' hr
  have hι1 := knιInv_one hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hQ3 : (1 : ℝ) ≤ r.Q ^ 3 := one_le_pow₀ hQ
  have hc0 : 0 ≤ 2 * knCapInv r := by
    have : 0 ≤ (r.Q : ℝ) ^ 3 * knΛ r * knιInv r :=
      mul_nonneg (mul_nonneg (by linarith) (by linarith)) (by linarith)
    linarith
  have hc : (knNh r : ℝ) < 2 * knCapInv r + 1 := Nat.ceil_lt_add_one hc0
  have hc' : 2 * knCapInv r ≤ (knNh r : ℝ) := Nat.le_ceil (2 * knCapInv r)
  -- `2 capInv ≥ 2^109 M`, `M = Q⁶ A² X⁴ (1/ξc)² Λ`
  have hQ6 : (1 : ℝ) ≤ r.Q ^ 6 := one_le_pow₀ hQ
  have hA2 : (4 : ℝ) ≤ r.A ^ 2 := by nlinarith
  have hX4 : 1 ≤ knX r ^ 4 := one_le_pow₀ hX
  have hξ2 : 1 ≤ (1 / r.ξc) ^ 2 := one_le_pow₀ iξ
  set M := (r.Q : ℝ) ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r with hMd
  have hM : 2 ^ 109 * M ≤ 2 * knCapInv r := by
    have e : 2 ^ 109 * M = 2 * (2 ^ 64 * r.Q ^ 3 * knΛ r
        * (2 ^ 44 * r.Q ^ 3 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2)) := by rw [hMd]; ring
    have h0 : (0 : ℝ) ≤ 2 ^ 64 * r.Q ^ 3 * knΛ r := by
      have : (0 : ℝ) ≤ r.Q ^ 3 := by linarith
      have : (0 : ℝ) ≤ knΛ r := by linarith
      positivity
    have h1 := mul_le_mul_of_nonneg_left hι h0
    linarith
  -- each factor of `M` is at least one
  have f1 : (1 : ℝ) * 4 ≤ r.Q ^ 6 * r.A ^ 2 := mul_le_mul hQ6 hA2 (by norm_num) (by linarith)
  have f2 : (4 : ℝ) * 1 ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 :=
    mul_le_mul (by linarith) hX4 (by norm_num) (by linarith)
  have f3 : (4 : ℝ) * 1 ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 :=
    mul_le_mul (by linarith) hξ2 (by norm_num) (by linarith)
  have hM1 : (4 : ℝ) * 30 ≤ M := mul_le_mul (by linarith) hΛ (by norm_num) (by linarith)
  have hB₃M : knB₃ r ≤ 2 ^ 14 * M := by
    refine hB₃.trans ?_
    have g1 : (r.Q : ℝ) ^ 2 ≤ r.Q ^ 6 := pow_le_pow_right₀ hQ (by norm_num)
    have g2 : knX r ≤ knX r ^ 4 := le_self_pow₀ hX (by norm_num)
    have g3 : 1 / r.ξc ≤ (1 / r.ξc) ^ 2 := le_self_pow₀ iξ (by norm_num)
    have : (r.Q : ℝ) ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc) * knΛ r ≤ M := by
      rw [hMd]
      have a1 : (r.Q : ℝ) ^ 2 * r.A ^ 2 ≤ r.Q ^ 6 * r.A ^ 2 :=
        mul_le_mul_of_nonneg_right g1 (by positivity)
      have a2 : (r.Q : ℝ) ^ 2 * r.A ^ 2 * knX r ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 :=
        mul_le_mul a1 g2 (by linarith) (by linarith)
      have a3 : (r.Q : ℝ) ^ 2 * r.A ^ 2 * knX r * (1 / r.ξc)
          ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 :=
        mul_le_mul a2 g3 (by linarith) (by linarith)
      exact mul_le_mul_of_nonneg_right a3 (by linarith)
    linarith
  have hVM : ((r.Q : ℝ) + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2)) ≤ 41 * M := by
    have g1 : ((r.Q : ℝ) + 2) ^ 3 ≤ (3 * r.Q) ^ 3 :=
      pow_le_pow_left₀ (by linarith) (by linarith) 3
    have g2 : (r.A : ℝ) + 1 ≤ 3 / 2 * r.A := by linarith
    have g3 : ((r.Q : ℝ) + 2) ^ 3 * (r.A + 1) ≤ (3 * r.Q) ^ 3 * (3 / 2 * r.A) :=
      mul_le_mul g1 g2 (by linarith) (by positivity)
    have g4 : (r.Q : ℝ) ^ 3 * r.A ≤ M := by
      rw [hMd]
      have a1 : (r.Q : ℝ) ^ 3 ≤ r.Q ^ 6 := pow_le_pow_right₀ hQ (by norm_num)
      have a2 : (r.A : ℝ) ≤ r.A ^ 2 := le_self_pow₀ (by linarith) (by norm_num)
      have a3 : (r.Q : ℝ) ^ 3 * r.A ≤ r.Q ^ 6 * r.A ^ 2 :=
        mul_le_mul a1 a2 (by linarith) (by linarith)
      have a4 : (r.Q : ℝ) ^ 6 * r.A ^ 2
          ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r := by
        have b1 : (r.Q : ℝ) ^ 6 * r.A ^ 2 ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 :=
          le_mul_of_one_le_right (by linarith) hX4
        have b2 : (r.Q : ℝ) ^ 6 * r.A ^ 2 * knX r ^ 4
            ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 :=
          le_mul_of_one_le_right (by linarith) hξ2
        have b3 : (r.Q : ℝ) ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2
            ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r :=
          le_mul_of_one_le_right (by linarith) (by linarith)
        linarith
      linarith
    have e : ((r.Q : ℝ) + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2))
        = ((r.Q : ℝ) + 2) ^ 3 * (r.A + 1) := by
      ring
    have e2 : (3 * (r.Q : ℝ)) ^ 3 * (3 / 2 * r.A) = 81 / 2 * (r.Q ^ 3 * r.A) := by ring
    rw [e]
    linarith
  have hQM : 6 * (r.Q : ℝ) * knΛ r ≤ 6 * M := by
    have a1 : (r.Q : ℝ) ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 := by
      have : (r.Q : ℝ) ≤ r.Q ^ 6 := le_self_pow₀ hQ (by norm_num)
      have b1 : (r.Q : ℝ) ^ 6 ≤ r.Q ^ 6 * r.A ^ 2 :=
        le_mul_of_one_le_right (by linarith) (by linarith)
      have b2 : (r.Q : ℝ) ^ 6 * r.A ^ 2 ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 :=
        le_mul_of_one_le_right (by linarith) hX4
      have b3 : (r.Q : ℝ) ^ 6 * r.A ^ 2 * knX r ^ 4
          ≤ r.Q ^ 6 * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 :=
        le_mul_of_one_le_right (by linarith) hξ2
      linarith
    have := mul_le_mul_of_nonneg_right a1 (show 0 ≤ knΛ r by linarith)
    rw [hMd]
    linarith
  have hcZ2 : 2 * knCapInv r + 1 ≤ 2 ^ 187 * knZ r ^ 43 * knΛ r ^ 4 := by
    have hZ1 : 1 ≤ knZ r ^ 43 * knΛ r ^ 4 :=
      one_le_mul_of_one_le_of_one_le (one_le_pow₀ (by linarith)) (one_le_pow₀ (by linarith))
    have : 2 * knCapInv r ≤ 2 * (2 ^ 185 * knZ r ^ 43 * knΛ r ^ 4) := by linarith
    linarith
  refine ⟨hc', hc.le, by linarith, by linarith, by linarith, by linarith⟩

lemma knJ_bounds (hr : r ∈ rlStarRunRegion) :
    (knNh r : ℝ) ≤ knJ r ∧ (knJ r : ℝ) ≤ 2 * knX r * knNh r
      ∧ (knJ r : ℝ) ≤ knNh r * (r.L + 2)
      ∧ ∀ k : ℕ, k ≤ r.L → knNh r * (k + 1) + 1 ≤ knJ r := by
  obtain ⟨-, -, hnh1, -⟩ := knNh_bounds hr
  obtain ⟨hQ, -, hX, -, hLX, -⟩ := kn_base hr
  have hL0 : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have e : (knJ r : ℝ) = knNh r * (r.L + 1) + 1 := by unfold knJ; push_cast; ring
  rw [e]
  refine ⟨by nlinarith, ?_, by nlinarith, fun k hk => ?_⟩
  · unfold knX at *; nlinarith
  · have : (k : ℝ) ≤ r.L := by exact_mod_cast hk
    have : (knNh r : ℝ) * (k + 1) ≤ knNh r * (r.L + 1) := by gcongr
    exact_mod_cast (by linarith : (knNh r : ℝ) * (k + 1) + 1 ≤ knNh r * (r.L + 1) + 1)

lemma knNr_bounds (hr : r ∈ rlStarRunRegion) :
    (knNr r : ℝ) ≤ knE r * knRp r * (r.Q + 3) * knNh r * knB r * (r.L + 3)
      ∧ (knNr r : ℝ) ≤ 2 ^ 34 * r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r
        * knNh r
      ∧ 1 ≤ (knNr r : ℝ) := by
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hb1, hb, -⟩ := knB_bounds hr
  obtain ⟨hrp1, hrp⟩ := knRp_bounds hr
  obtain ⟨hB₃0, -⟩ := knB₃_bounds hr
  obtain ⟨-, -, hnh1, hnhB, -⟩ := knNh_bounds hr
  obtain ⟨hJnh, hJX, hJL, -⟩ := knJ_bounds hr
  obtain ⟨hQ, hA, hX, hQX, hLX, -, -, hξ, hξ1, -⟩ := kn_base hr
  have hΛ0 : 0 < knΛ r := by linarith [(knΛ_bounds hr).1]
  have hlg0 : 0 < knLg r := by linarith [(knLg_bounds hr).1]
  have hL0 : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hJ1 : (1 : ℝ) ≤ knJ r := hnh1.trans hJnh
  set u : ℝ := knRp r * (r.Q + 3) with hud
  have hu : 1 ≤ u := by
    have := mul_le_mul hrp1 (show (1 : ℝ) ≤ r.Q + 3 by linarith) zero_le_one (by linarith)
    linarith
  set V : ℝ := (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2)) with hVd
  have hV : 0 ≤ V := by positivity
  set T : ℝ := knE r * ((knB₃ r + knJ r * knB r) * u + V) with hTd
  have hJb : (1 : ℝ) ≤ knJ r * knB r := one_le_mul_of_one_le_of_one_le hJ1 hb1
  have hBu : (1 : ℝ) ≤ (knB₃ r + knJ r * knB r) * u :=
    one_le_mul_of_one_le_of_one_le (by linarith) hu
  have hT1 : 1 ≤ T := by
    have := mul_le_mul hE1 (show (1 : ℝ) ≤ (knB₃ r + knJ r * knB r) * u + V by linarith)
      zero_le_one (by linarith)
    rw [hTd]; linarith
  have hc' : T ≤ ((⌈T⌉₊ : ℕ) : ℝ) := Nat.le_ceil T
  have hc : ((⌈T⌉₊ : ℕ) : ℝ) < T + 1 := Nat.ceil_lt_add_one (by linarith)
  have hNr : knNr r = ⌈T⌉₊ := rfl
  rw [hNr]
  -- `T + 1 ≤ E u nh + E J b u`
  have h1 : T + 1 ≤ knE r * u * knNh r + knE r * knJ r * knB r * u := by
    have a1 : knE r * V ≤ knE r * u * V :=
      mul_le_mul_of_nonneg_right (le_mul_of_one_le_right (by linarith) hu) hV
    have a2 : (1 : ℝ) ≤ knE r * u := one_le_mul_of_one_le_of_one_le hE1 hu
    have a3 : knE r * u * (knB₃ r + V + 1) ≤ knE r * u * knNh r :=
      mul_le_mul_of_nonneg_left hnhB (by linarith)
    rw [hTd]; linarith
  have hEu : (0 : ℝ) ≤ knE r * u := by linarith [one_le_mul_of_one_le_of_one_le hE1 hu]
  refine ⟨?_, ?_, by linarith⟩
  · have b1 : knE r * u * knNh r ≤ knE r * u * knNh r * knB r :=
      le_mul_of_one_le_right (mul_nonneg hEu (by linarith)) hb1
    have b2 := mul_le_mul_of_nonneg_left hJL (show 0 ≤ knE r * u * knB r from
      mul_nonneg hEu (by linarith))
    have e : knE r * u * knNh r * knB r * (r.L + 3)
        = knE r * knRp r * (r.Q + 3) * knNh r * knB r * (r.L + 3) := by rw [hud]; ring
    rw [← e]
    linarith
  · have a1 : (knNh r : ℝ) ≤ knJ r * knB r := hJnh.trans (le_mul_of_one_le_right (by linarith) hb1)
    have a2 := mul_le_mul_of_nonneg_left a1 hEu
    have h3 : knE r * knJ r * knB r * u
        ≤ (2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)) * (2 * knX r * knNh r)
          * (2 ^ 13 * r.A * knX r * (1 / r.ξc) * knΛ r) * (16 * knLg r * (4 * r.Q)) := by
      rw [hud]
      gcongr
      linarith
    have hY0 : 0 ≤ knE r * knJ r * knB r * u := by positivity
    have e : 4 * ((2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)) * (2 * knX r * knNh r)
          * (2 ^ 13 * r.A * knX r * (1 / r.ξc) * knΛ r) * (16 * knLg r * (4 * r.Q)))
        = 2 ^ 34 * r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r * knNh r := by
      ring
    linarith

lemma knW_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ knW r ∧ (knNr r : ℝ) ≤ knW r
      ∧ (((r.L + r.Q + 3 : ℕ) : ℝ) + 1) * (r.Q + 2) ≤ r.ξc / 4 * knW r
      ∧ knW r ≤ knNr r + 60 * knX r * r.Q * (1 / r.ξc) := by
  obtain ⟨-, -, hNr1⟩ := knNr_bounds hr
  obtain ⟨hQ, hA, hX, hQX, hLX, -, -, hξ, -⟩ := kn_base hr
  have e : (((r.L + r.Q + 3 : ℕ) : ℝ) + 1) * (r.Q + 2) / (r.ξc / 4)
      = 4 * ((r.L : ℝ) + r.Q + 4) * (r.Q + 2) * (1 / r.ξc) := by
    push_cast; field_simp; ring
  have hw : (((r.L + r.Q + 3 : ℕ) : ℝ) + 1) * (r.Q + 2) / (r.ξc / 4) ≤ knW r := le_max_right _ _
  refine ⟨hNr1.trans (le_max_left _ _), le_max_left _ _, ?_, ?_⟩
  · rw [div_le_iff₀ (by positivity)] at hw
    linarith
  · unfold knW
    rw [e]
    refine max_le (by linarith [show 0 ≤ 60 * knX r * r.Q * (1 / r.ξc) by positivity]) ?_
    have h1 : 4 * ((r.L : ℝ) + r.Q + 4) * (r.Q + 2) ≤ 60 * knX r * r.Q := by
      unfold knX at *; nlinarith
    have : 4 * ((r.L : ℝ) + r.Q + 4) * (r.Q + 2) * (1 / r.ξc) ≤ 60 * knX r * r.Q * (1 / r.ξc) := by
      gcongr
    have : (0 : ℝ) ≤ knNr r := Nat.cast_nonneg _
    linarith

lemma knL3 (hr : r ∈ rlStarRunRegion) :
    log (8 * knW r / knδ' r) ≤ knΛ r
      ∧ log (4 * (knE r * knJ r) / knδ' r) ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨hlg, hlgX⟩ := knLg_bounds hr
  obtain ⟨hδ', -, hiδ'⟩ := knδ'_bounds hr
  obtain ⟨hW1, -, -, hW⟩ := knW_bounds hr
  obtain ⟨-, hNr, -⟩ := knNr_bounds hr
  obtain ⟨-, -, hnh1, -, -, hnhZ⟩ := knNh_bounds hr
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨-, hJX, -⟩ := knJ_bounds hr
  have hZ1 : 1 ≤ knZ r := by linarith
  have hZΛ : 1 ≤ knZ r ^ 43 * knΛ r ^ 4 :=
    one_le_mul_of_one_le_of_one_le (one_le_pow₀ hZ1) (one_le_pow₀ (by linarith))
  have hJ0 : (0 : ℝ) < knJ r := by linarith [knJ_bounds hr |>.1]
  have hΛ0 : 0 < knΛ r := by linarith
  have hlg0 : 0 < knLg r := by linarith
  constructor
  · refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 2 ^ 229) (n := 55) (j := 5)
      (k := 77) (by norm_num) ?_ (by norm_num) (by norm_num))
    have hNrZ : (knNr r : ℝ) ≤ 2 ^ 221 * knZ r ^ 53 * knΛ r ^ 5 := by
      calc (knNr r : ℝ)
          ≤ 2 ^ 34 * r.Q * r.A ^ 2 * knX r ^ 4 * (1 / r.ξc) ^ 2 * knΛ r * knLg r * knNh r := hNr
        _ ≤ 2 ^ 34 * knZ r * knZ r ^ 2 * knZ r ^ 4 * knZ r ^ 2 * knΛ r * knZ r
            * (2 ^ 187 * knZ r ^ 43 * knΛ r ^ 4) := by
            gcongr
            linarith
        _ = 2 ^ 221 * knZ r ^ 53 * knΛ r ^ 5 := by ring
    have h60 : 60 * knX r * r.Q * (1 / r.ξc) ≤ 2 ^ 221 * knZ r ^ 53 * knΛ r ^ 5 := by
      have a1 : 60 * knX r * r.Q * (1 / r.ξc) ≤ 60 * knZ r * knZ r * knZ r := by gcongr
      have a2 : knZ r ^ 3 ≤ knZ r ^ 53 := pow_le_pow_right₀ hZ1 (by norm_num)
      have a3 : knZ r ^ 53 ≤ knZ r ^ 53 * knΛ r ^ 5 :=
        le_mul_of_one_le_right (by positivity) (one_le_pow₀ (by linarith))
      have a4 : 0 ≤ knZ r ^ 53 * knΛ r ^ 5 := by positivity
      have e : 60 * knZ r * knZ r * knZ r = 60 * knZ r ^ 3 := by ring
      have e2 : 2 ^ 221 * knZ r ^ 53 * knΛ r ^ 5 = 2 ^ 221 * (knZ r ^ 53 * knΛ r ^ 5) := by ring
      rw [e2]
      linarith
    calc 8 * knW r / knδ' r = 8 * knW r * (1 / knδ' r) := by ring
      _ ≤ 8 * (2 ^ 221 * knZ r ^ 53 * knΛ r ^ 5 + 2 ^ 221 * knZ r ^ 53 * knΛ r ^ 5)
          * (16 * knZ r * knZ r) := by
          gcongr
          · linarith
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 229 * knZ r ^ 55 * knΛ r ^ 5 := by ring
  · refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 2 ^ 206) (n := 50) (j := 4)
      (k := 69) (by norm_num) ?_ (by norm_num) (by norm_num))
    calc 4 * (knE r * knJ r) / knδ' r = 4 * (knE r * knJ r) * (1 / knδ' r) := by ring
      _ ≤ 4 * ((2 ^ 12 * knZ r * knZ r ^ 2 * knZ r)
          * (2 * knZ r * (2 ^ 187 * knZ r ^ 43 * knΛ r ^ 4)))
          * (16 * knZ r * knZ r) := by
          gcongr
          · exact hE.trans (by gcongr)
          · exact hJX.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 206 * knZ r ^ 50 * knΛ r ^ 4 := by ring

end lists

/-! The pool, the prefix count and the cap. -/

section counts

variable {r : RLStarRun}

lemma knη₀_s (r : RLStarRun) : 1 / 2 - knη₀ r = r.s₀ / 2 := by unfold knη₀; ring

lemma knMax_bounds (hr : r ∈ rlStarRunRegion) :
    0 ≤ max (log (20 / knτ r)) (log (8 * knW r / knδ' r))
      ∧ max (log (20 / knτ r)) (log (8 * knW r / knδ' r)) ≤ knΛ r := by
  obtain ⟨hτ, hτι, -⟩ := knτ_bounds hr
  obtain ⟨-, -, -, -, hι80⟩ := knι_bounds hr
  have h20 : 0 ≤ log (20 / knτ r) :=
    Real.log_nonneg (by rw [le_div_iff₀ hτ]; linarith)
  exact ⟨h20.trans (le_max_left _ _), max_le (knL2 hr) (knL3 hr).1⟩

lemma knPB_bounds (hr : r ∈ rlStarRunRegion) :
    28 ≤ knPB r ∧ knPB r ≤ 2 ^ 10 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP)
      ∧ 0 ≤ knFV r ∧ knFV r ≤ 2 ^ 8 * knΛ r * (1 / r.s₀) ^ 2 := by
  obtain ⟨hm0, hm⟩ := knMax_bounds hr
  obtain ⟨hl3, -, -⟩ := knL4 hr
  obtain ⟨hδ', hδ'1, -⟩ := knδ'_bounds hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have ip : 1 ≤ 1 / r.pAP := by rw [le_div_iff₀ hp]; linarith
  have hl0 : 0 ≤ log (3 / (knδ' r / 2)) :=
    Real.log_nonneg (by rw [le_div_iff₀ (by positivity)]; linarith)
  have ePB : knPB r = (200 * (max (log (20 / knτ r)) (log (8 * knW r / knδ' r)) + 3)
        * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28) * (1 / r.pAP)
      + 120 * log (3 / (knδ' r / 2)) * (1 / r.pAP) := by
    unfold knPB; rw [knη₀_s]; ring
  have eFV : knFV r = (200 * (max (log (20 / knτ r)) (log (8 * knW r / knδ' r)) + 3)
      * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28) / 4 := by
    unfold knFV; rw [knη₀_s]; ring
  set m := max (log (20 / knτ r)) (log (8 * knW r / knδ' r)) with hmd
  set l := log (3 / (knδ' r / 2)) with hld
  have hs2 : 4 ≤ (1 / r.s₀) ^ 2 := by nlinarith
  have hcore : 200 * (m + 3) * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28
      ≤ 224 * knΛ r * (1 / r.s₀) ^ 2 := by nlinarith
  have hcore0 : 28 ≤ 200 * (m + 3) * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28 := by
    have : 0 ≤ 200 * (m + 3) * (1 / r.s₀) ^ 2 := by positivity
    linarith
  rw [ePB, eFV]
  refine ⟨?_, ?_, by linarith, by linarith⟩
  · have : 0 ≤ 120 * l * (1 / r.pAP) := by positivity
    nlinarith
  · have h1 : 120 * l ≤ 30 * knΛ r * (1 / r.s₀) ^ 2 := by nlinarith
    have h2 : (200 * (m + 3) * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28) + 120 * l
        ≤ 2 ^ 10 * knΛ r * (1 / r.s₀) ^ 2 := by nlinarith
    calc (200 * (m + 3) * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28) * (1 / r.pAP)
          + 120 * l * (1 / r.pAP)
        = ((200 * (m + 3) * (1 / r.s₀) ^ 2 + 40 * (1 / r.s₀) + 28) + 120 * l) * (1 / r.pAP) := by
          ring
      _ ≤ 2 ^ 10 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) := by gcongr

lemma knL56 (hr : r ∈ rlStarRunRegion) :
    0 ≤ log (128 * (knPB r + 1) / knδ' r) ∧ log (128 * (knPB r + 1) / knδ' r) ≤ knΛ r
      ∧ 0 ≤ log ((r.Q + 2 + 2) * knPB r / (knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r))
      ∧ log ((r.Q + 2 + 2) * knPB r / (knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r))
        ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hPB28, hPB, -⟩ := knPB_bounds hr
  obtain ⟨hδ', hδ'1, hiδ'⟩ := knδ'_bounds hr
  obtain ⟨hτ, hτι, hiτ⟩ := knτ_bounds hr
  obtain ⟨-, -, -, -, hι80⟩ := knι_bounds hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hιZ := knιInv_Z hr
  have hι1 := knιInv_one hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have ip : 1 ≤ 1 / r.pAP := by rw [le_div_iff₀ hp]; linarith
  have hZ1 : 1 ≤ knZ r := by linarith
  have hPBZ : knPB r + 1 ≤ 2 ^ 11 * knΛ r * knZ r ^ 3 := by
    have : 2 ^ 10 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ≤ 2 ^ 10 * knΛ r * knZ r ^ 2 * knZ r := by
      gcongr
    have : 1 ≤ knΛ r * knZ r ^ 3 := one_le_mul_of_one_le_of_one_le (by linarith) (one_le_pow₀ hZ1)
    have e : 2 ^ 10 * knΛ r * knZ r ^ 2 * knZ r = 2 ^ 10 * (knΛ r * knZ r ^ 3) := by ring
    nlinarith
  have hd : 0 < knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r := by positivity
  have hτ1 : knτ r ≤ 1 := by linarith
  refine ⟨Real.log_nonneg (by rw [le_div_iff₀ hδ']; linarith), knob_log_le hr (by positivity)
    (knob_pow_le hr (c := 2 ^ 22) (n := 5) (j := 1) (k := 8) (by norm_num) ?_ (by norm_num)
    (by norm_num)), Real.log_nonneg ?_, knob_log_le hr (by positivity)
    (knob_pow_le hr (c := 5 * 2 ^ 64) (n := 23) (j := 2) (k := 23) (by norm_num) ?_
    (by norm_num) (by norm_num))⟩
  · calc 128 * (knPB r + 1) / knδ' r = 128 * (knPB r + 1) * (1 / knδ' r) := by ring
      _ ≤ 128 * (2 ^ 11 * knΛ r * knZ r ^ 3) * (16 * knZ r * knZ r) := by
          gcongr
          exact hiδ'.trans (by gcongr)
      _ = 2 ^ 22 * knZ r ^ 5 * knΛ r ^ 1 := by ring
  · rw [le_div_iff₀ hd]
    have h1 : knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r ≤ 1 :=
      mul_le_one₀ (mul_le_one₀ (mul_le_one₀ (by linarith) (by positivity) (by linarith))
        (by positivity) (by linarith)) hτ.le hτ1
    have h2 : (1 : ℝ) ≤ (r.Q + 2 + 2) * knPB r :=
      one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
    linarith
  · have e : (r.Q + 2 + 2) * knPB r / (knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r)
        = 16 * (r.Q + 4) * knPB r * (1 / knδ' r) ^ 2 * (1 / r.pAP) * (1 / knτ r) := by
      field_simp; ring
    rw [e]
    calc 16 * (r.Q + 4) * knPB r * (1 / knδ' r) ^ 2 * (1 / r.pAP) * (1 / knτ r)
        ≤ 16 * (5 * knZ r) * (2 ^ 10 * knΛ r * knZ r ^ 2 * knZ r) * (16 * knZ r * knZ r) ^ 2
          * knZ r * (2 * (2 ^ 41 * knZ r ^ 12 * knΛ r) * knZ r ^ 2) := by
          gcongr
          · linarith
          · exact hPB.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
          · exact hiτ.trans (by gcongr)
      _ = 5 * 2 ^ 64 * knZ r ^ 23 * knΛ r ^ 2 := by ring

lemma knRB_bounds (hr : r ∈ rlStarRunRegion) : 0 ≤ knRB r ∧ knRB r ≤ knRBb r ∧ 1 ≤ knRBb r := by
  obtain ⟨l50, l5, l60, l6⟩ := knL56 hr
  obtain ⟨hι, -, -, hιI, -⟩ := knι_bounds hr
  obtain ⟨hτ, -, hiτ⟩ := knτ_bounds hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hι1 := knιInv_one hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have e : knRB r = 2 ^ 26 * (r.Q + 2) * log (128 * (knPB r + 1) / knδ' r) * (1 / r.s₀) ^ 4
        * (1 / knι r) ^ 2
      + 2 ^ 27 * log ((r.Q + 2 + 2) * knPB r / (knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r))
        * (1 / knτ r) ^ 2 := by
    unfold knRB; rw [knη₀_s]; ring
  set l5' := log (128 * (knPB r + 1) / knδ' r)
  set l6' := log ((r.Q + 2 + 2) * knPB r / (knδ' r / 2 * (knδ' r / 4) * (r.pAP / 2) * knτ r))
  have hRBb1 : 1 ≤ knRBb r := by
    unfold knRBb
    have : 1 ≤ (1 / r.s₀) ^ 4 := one_le_pow₀ (by linarith)
    have : 1 ≤ knιInv r ^ 2 := one_le_pow₀ hι1
    have : 1 ≤ r.Q * knΛ r := by nlinarith
    have : 1 ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 := by nlinarith
    have : 1 ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2 := by nlinarith
    nlinarith
  rw [e]
  refine ⟨by positivity, ?_, hRBb1⟩
  calc 2 ^ 26 * (r.Q + 2) * l5' * (1 / r.s₀) ^ 4 * (1 / knι r) ^ 2
        + 2 ^ 27 * l6' * (1 / knτ r) ^ 2
      ≤ 2 ^ 26 * (3 * r.Q) * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2
        + 2 ^ 27 * knΛ r * (2 * knιInv r * (1 / r.s₀) ^ 2) ^ 2 := by
        gcongr
        linarith
    _ = (3 * 2 ^ 26 * r.Q + 2 ^ 29) * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2 := by ring
    _ ≤ 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2 := by
        gcongr
        linarith

lemma knRBb_Z (hr : r ∈ rlStarRunRegion) : knRBb r ≤ 2 ^ 112 * knZ r ^ 29 * knΛ r ^ 3 := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hιZ := knιInv_Z hr
  have hι1 := knιInv_one hr
  unfold knRBb
  calc 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2
      ≤ 2 ^ 30 * knZ r * knΛ r * knZ r ^ 4 * (2 ^ 41 * knZ r ^ 12 * knΛ r) ^ 2 := by
        gcongr
    _ = 2 ^ 112 * knZ r ^ 29 * knΛ r ^ 3 := by ring

lemma knL7 (hr : r ∈ rlStarRunRegion) :
    0 ≤ log (128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r)
      ∧ log (128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r)
        ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hPB28, hPB, -⟩ := knPB_bounds hr
  obtain ⟨hδ', hδ'1, hiδ'⟩ := knδ'_bounds hr
  obtain ⟨hRB0, hRB, hRBb1⟩ := knRB_bounds hr
  have hRBZ := knRBb_Z hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hZ1 : 1 ≤ knZ r := by linarith
  have hlt := Real.log_two_gt_d9
  have hlb0 : 0 ≤ Real.logb 2 (knRB r + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  have hlb : 3 + 2 * Real.logb 2 (knRB r + 1) ≤ 6 * knRBb r := by
    have h1 : log (knRB r + 1) ≤ knRB r := by
      have := Real.log_le_sub_one_of_pos (show 0 < knRB r + 1 by linarith); linarith
    have h2 : Real.logb 2 (knRB r + 1) ≤ 3 / 2 * knRB r := by
      rw [Real.logb, div_le_iff₀ (by linarith)]
      nlinarith
    linarith
  have hy0 : 0 < 128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r := by
    positivity
  refine ⟨Real.log_nonneg ?_, knob_log_le hr hy0 (knob_pow_le hr (c := 2 ^ 139) (n := 35)
    (j := 4) (k := 47) (by norm_num) ?_ (by norm_num) (by norm_num))⟩
  · rw [le_div_iff₀ hδ']
    have h1 : (1 : ℝ) ≤ 128 * (r.Q + 2) * (knPB r + 1) :=
      one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
    have h2 := one_le_mul_of_one_le_of_one_le h1 (show (1 : ℝ) ≤ 3 + 2 * Real.logb 2 (knRB r + 1)
      by linarith)
    linarith
  · have hPBZ : knPB r + 1 ≤ 2 ^ 11 * knΛ r * knZ r ^ 3 := by
      have : 2 ^ 10 * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP)
          ≤ 2 ^ 10 * knΛ r * knZ r ^ 2 * knZ r := by
        gcongr
      have : 1 ≤ knΛ r * knZ r ^ 3 :=
        one_le_mul_of_one_le_of_one_le (by linarith) (one_le_pow₀ hZ1)
      have e : 2 ^ 10 * knΛ r * knZ r ^ 2 * knZ r = 2 ^ 10 * (knΛ r * knZ r ^ 3) := by ring
      nlinarith
    calc 128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r
        = 128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1))
          * (1 / knδ' r) := by ring
      _ ≤ 128 * (3 * knZ r) * (2 ^ 11 * knΛ r * knZ r ^ 3)
          * (6 * (2 ^ 112 * knZ r ^ 29 * knΛ r ^ 3)) * (16 * knZ r * knZ r) := by
          gcongr
          · linarith
          · exact hlb.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 134 * 18 * knZ r ^ 35 * knΛ r ^ 4 := by ring
      _ ≤ 2 ^ 139 * knZ r ^ 35 * knΛ r ^ 4 := by gcongr; norm_num

lemma inv_le_of_inv_le {t C : ℝ} (ht : 0 < t) (h : 1 / t ≤ C) : 1 / C ≤ t := by
  calc 1 / C ≤ 1 / (1 / t) := one_div_le_one_div_of_le (by positivity) h
    _ = t := one_div_one_div t

lemma knCapP_ge (hr : r ∈ rlStarRunRegion) : 1 / knCapInv r ≤ knCapP r := by
  unfold knCapP
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, hξ1, hp, hp1, hs, hs1⟩ := kn_base hr
  obtain ⟨hεc, -, hiεc, -, -⟩ := knεcov_bounds hr
  obtain ⟨hι, hι8, -, hιI, hι80⟩ := knι_bounds hr
  obtain ⟨hPB28, hPB, -⟩ := knPB_bounds hr
  obtain ⟨hRB0, hRB, hRBb1⟩ := knRB_bounds hr
  obtain ⟨l70, l7⟩ := knL7 hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hι1 := knιInv_one hr
  have hιX := knιInv_ge hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have ip : 1 ≤ 1 / r.pAP := by rw [le_div_iff₀ hp]; linarith
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  rw [knη₀_s]
  set l7' := Real.log (128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1))
    / knδ' r)
  have hl7 : 0 < l7' := by
    refine Real.log_pos ?_
    obtain ⟨hδ', hδ'1, -⟩ := knδ'_bounds hr
    have hlb0 : 0 ≤ Real.logb 2 (knRB r + 1) := Real.logb_nonneg (by norm_num) (by linarith)
    rw [lt_div_iff₀ hδ']
    have h1 : (1 : ℝ) ≤ 128 * (r.Q + 2) * (knPB r + 1) :=
      one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
    have h2 := one_le_mul_of_one_le_of_one_le h1 (show (1 : ℝ) ≤ 3 + 2 * Real.logb 2 (knRB r + 1)
      by linarith)
    linarith
  set Is := 1 / r.s₀ with hIs
  set Ip := 1 / r.pAP with hIp
  set ιI := knιInv r with hιId
  set Λ := knΛ r with hΛd
  set Q : ℝ := (r.Q : ℝ) with hQd
  set P1 := 2 ^ 60 * Q * Λ * Is ^ 4 * ιI ^ 3 * (Is ^ 2 + Q ^ 2) with hP1
  set P2 := 2 ^ 30 * Q * Λ * Is ^ 2 * Ip ^ 2 with hP2
  have hcap : knCapInv r = P1 + P2 := rfl
  have hP10 : 0 ≤ P1 := by positivity
  have hP20 : 0 ≤ P2 := by positivity
  have hRBe : knRBb r = 2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 2 := rfl
  set K := Q * Λ * Is ^ 4 * ιI ^ 3 with hK
  have hK0 : 0 ≤ K := by positivity
  have hP1K : P1 = 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K := by rw [hP1, hK]; ring
  have hRB2 : knRB r + 1 ≤ 2 * (2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 2) := by rw [← hRBe]; linarith
  have h4 : (4 : ℝ) ≤ Is ^ 2 := by
    calc (4 : ℝ) = 2 * 2 := by norm_num
      _ ≤ Is * Is := mul_le_mul is is (by norm_num) (by linarith)
      _ = Is ^ 2 := by ring
  have hPB1 : knPB r + 2 ≤ 2 ^ 11 * Λ * Is ^ 2 * Ip := by
    have : 1 ≤ Λ * Is ^ 2 := one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
    have : 1 ≤ Λ * Is ^ 2 * Ip := one_le_mul_of_one_le_of_one_le this ip
    linarith
  have hIs0 : 0 ≤ Is ^ 2 := sq_nonneg Is
  have hQs0 : 0 ≤ Q ^ 2 := sq_nonneg Q
  have hQQ : Q ≤ Q ^ 2 := le_self_pow₀ hQ (by norm_num)
  have hιι : ιI ^ 2 ≤ ιI ^ 3 := pow_le_pow_right₀ hι1 (by norm_num)
  have hKs : ∀ a : ℝ, a ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) → a * K ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K :=
    fun a h => mul_le_mul_of_nonneg_right h hK0
  have key : ∀ t : ℝ, 0 < t → 1 / t ≤ P1 + P2 → 1 / (P1 + P2) ≤ t := fun t ht h =>
    inv_le_of_inv_le ht h
  rw [hcap]
  have hm0 : 0 < min (knεcov r / 2 * (r.s₀ / 2)) (knι r) := lt_min (by positivity) hι
  have hRB0' : 0 < knRB r + 1 := by linarith
  have hPB0' : 0 < knPB r + 1 := by linarith
  refine le_min (le_min (le_min (key _ (by positivity) ?_) (key _ (by positivity) ?_))
    (key _ (by positivity) ?_)) (le_min (le_min (key _ (by positivity) ?_)
    (key _ (by positivity) ?_)) (le_min (key _ (by positivity) ?_) (key _ (by positivity) ?_)))
  · -- `1/t₁ ≤ 2⁵³ Is² K`
    have hmin : 1 / min (knεcov r / 2 * (r.s₀ / 2)) (knι r) ≤ 2 * ιI * Is := by
      calc 1 / min (knεcov r / 2 * (r.s₀ / 2)) (knι r)
          ≤ 1 / (knεcov r / 2 * (r.s₀ / 2)) + 1 / knι r := one_div_min_le (by positivity) hι
        _ = 4 * (1 / knεcov r) * Is + 1 / knι r := by rw [hIs]; field_simp; ring
        _ ≤ 4 * (50 * knX r * (1 / r.ξc)) * Is + ιI * Is := by
            gcongr
            exact hιI.trans (le_mul_of_one_le_right (by linarith) (by linarith))
        _ ≤ ιI * Is + ιI * Is := by
            gcongr
            linarith [mul_pos (by linarith : (0 : ℝ) < knX r) (one_div_pos.2 hξ)]
        _ = 2 * ιI * Is := by ring
    have e : 1 / (r.s₀ / 2 * min (knεcov r / 2 * (r.s₀ / 2)) (knι r) / (2 ^ 20 * (knRB r + 1)))
        = 2 ^ 21 * (knRB r + 1) * Is * (1 / min (knεcov r / 2 * (r.s₀ / 2)) (knι r)) := by
      rw [hIs]; field_simp
    rw [e]
    calc 2 ^ 21 * (knRB r + 1) * Is * (1 / min (knεcov r / 2 * (r.s₀ / 2)) (knι r))
        ≤ 2 ^ 21 * (2 * (2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 2)) * Is * (2 * ιI * Is) := by gcongr
      _ = 2 ^ 53 * Is ^ 2 * K := by rw [hK]; ring
      _ ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K := hKs _ (by linarith)
      _ ≤ P1 + P2 := by rw [hP1K]; linarith
  · have e : 1 / (knι r ^ 2 / (2 ^ 28 * l7')) = 2 ^ 28 * l7' * (1 / knι r) ^ 2 := by field_simp
    rw [e]
    calc 2 ^ 28 * l7' * (1 / knι r) ^ 2 ≤ 2 ^ 28 * Λ * ιI ^ 2 := by gcongr
      _ ≤ 2 ^ 28 * Λ * ιI ^ 2 * (Q * Is ^ 4 * ιI) := by
          refine le_mul_of_one_le_right (by positivity) ?_
          have : 1 ≤ Is ^ 4 := one_le_pow₀ (by linarith)
          have : 1 ≤ Q * Is ^ 4 := one_le_mul_of_one_le_of_one_le hQ this
          exact one_le_mul_of_one_le_of_one_le this hι1
      _ = 2 ^ 28 * K := by rw [hK]; ring
      _ ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K := hKs _ (by linarith)
      _ ≤ P1 + P2 := by rw [hP1K]; linarith
  · have e : 1 / (knι r / 2 / (8 * (Q + 2) * (knRB r + 1)))
        = 16 * (Q + 2) * (knRB r + 1) * (1 / knι r) := by field_simp; ring
    rw [e]
    calc 16 * (Q + 2) * (knRB r + 1) * (1 / knι r)
        ≤ 16 * (3 * Q) * (2 * (2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 2)) * ιI := by
          gcongr
          linarith
      _ = 96 * 2 ^ 30 * Q * K := by rw [hK]; ring
      _ ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K := hKs _ (by linarith)
      _ ≤ P1 + P2 := by rw [hP1K]; linarith
  · have e : 1 / (r.pAP / (2 ^ 18 * (knPB r + 1))) = 2 ^ 18 * (knPB r + 1) * Ip := by
      rw [hIp]; field_simp
    rw [e]
    calc 2 ^ 18 * (knPB r + 1) * Ip ≤ 2 ^ 18 * (2 ^ 11 * Λ * Is ^ 2 * Ip) * Ip := by
          gcongr; linarith
      _ = 2 ^ 29 * (Λ * Is ^ 2 * Ip ^ 2) := by ring
      _ ≤ 2 ^ 30 * Q * (Λ * Is ^ 2 * Ip ^ 2) := by gcongr; linarith
      _ = P2 := by rw [hP2]; ring
      _ ≤ P1 + P2 := by linarith
  · have e : 1 / (knι r / (2 ^ 24 * (Q + 2) ^ 2 * (knRB r + 1)))
        = 2 ^ 24 * (Q + 2) ^ 2 * (knRB r + 1) * (1 / knι r) := by field_simp
    rw [e]
    calc 2 ^ 24 * (Q + 2) ^ 2 * (knRB r + 1) * (1 / knι r)
        ≤ 2 ^ 24 * (3 * Q) ^ 2 * (2 * (2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 2)) * ιI := by
          gcongr
          linarith
      _ = 18 * 2 ^ 54 * Q ^ 2 * K := by rw [hK]; ring
      _ ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K := hKs _ (by linarith)
      _ ≤ P1 + P2 := by rw [hP1K]; linarith
  · have e : 1 / (r.pAP / (2 * (Q + 2) * (knPB r + 2))) = 2 * (Q + 2) * (knPB r + 2) * Ip := by
      rw [hIp]; field_simp
    rw [e]
    calc 2 * (Q + 2) * (knPB r + 2) * Ip ≤ 2 * (3 * Q) * (2 ^ 11 * Λ * Is ^ 2 * Ip) * Ip := by
          gcongr; linarith
      _ = 3 * 2 ^ 12 * (Q * Λ * Is ^ 2 * Ip ^ 2) := by ring
      _ ≤ 2 ^ 30 * (Q * Λ * Is ^ 2 * Ip ^ 2) := by gcongr; norm_num
      _ = P2 := by rw [hP2]; ring
      _ ≤ P1 + P2 := by linarith
  · rw [one_div_one_div]
    calc 6 * (Q + 2) * (knRB r + 1) ≤ 6 * (3 * Q) * (2 * (2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 2)) := by
          gcongr; linarith
      _ ≤ 6 * (3 * Q) * (2 * (2 ^ 30 * Q * Λ * Is ^ 4 * ιI ^ 3)) := by gcongr
      _ = 36 * 2 ^ 30 * Q * K := by rw [hK]; ring
      _ ≤ 2 ^ 60 * (Is ^ 2 + Q ^ 2) * K := hKs _ (by linarith)
      _ ≤ P1 + P2 := by rw [hP1K]; linarith

end counts

/-! The premises that pin `p₀`, and the lists' slope. -/

section premises

variable {r : RLStarRun}

lemma knMcD_pos (hr : r ∈ rlStarRunRegion) : 0 < knMcD r := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have : 0 < log (r.A : ℝ) := Real.log_pos (by linarith)
  unfold knMcD
  positivity

/-- The misread term: `2 (⌊FV⌋ + 1)³ (|Σ|^Ls + 1)^⌊FV⌋ e^{−2t²/p₀} ≤ δ/8` at `t = ξc/4`. -/
lemma knMcD_spec (hr : r ∈ rlStarRunRegion) {p₀ : ℝ} (hp0 : 0 < p₀) (hp : p₀ ≤ knMcD r) :
    2 * ((⌊knFV r⌋₊ : ℝ) + 1) ^ 3 * ((r.A : ℝ) ^ (r.L + r.Q + 3 + r.Q + 3) + 1) ^ ⌊knFV r⌋₊
      * Real.exp (-2 * (r.ξc / 4) ^ 2 / p₀) ≤ r.δ / 8 := by
  obtain ⟨hQ, hA, hX, hQX, hLX, hδ, hδ2, hξ, hξ1, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨-, -, hFV0, hFV⟩ := knPB_bounds hr
  obtain ⟨-, -, hl16⟩ := knL4 hr
  have hlA : log 2 ≤ log (r.A : ℝ) := Real.log_le_log (by norm_num) hA
  have hlt := Real.log_two_gt_d9
  have hlA0 : 0 < log (r.A : ℝ) := by linarith
  have hL0 : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  set F := ⌊knFV r⌋₊ with hFd
  have hF : (F : ℝ) ≤ 2 ^ 8 * knΛ r * (1 / r.s₀) ^ 2 := (Nat.floor_le hFV0).trans hFV
  have hF0 : (0 : ℝ) ≤ F := Nat.cast_nonneg _
  set n := r.L + r.Q + 3 + r.Q + 3 with hnd
  have hn : (n : ℝ) + 1 ≤ 10 * knX r := by
    have : (n : ℝ) = r.L + 2 * r.Q + 6 := by rw [hnd]; push_cast; ring
    unfold knX at *; rw [this]; linarith
  -- each factor as an exponential
  have h1 : ((F : ℝ) + 1) ^ 3 ≤ Real.exp (3 * F) := by
    rw [show (3 : ℝ) * F = ((3 : ℕ) : ℝ) * F by norm_num, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by positivity) (by linarith [Real.add_one_le_exp (F : ℝ)]) 3
  have h2 : (r.A : ℝ) ^ n + 1 ≤ Real.exp (10 * knX r * log r.A) := by
    have hAn : (1 : ℝ) ≤ (r.A : ℝ) ^ n := one_le_pow₀ (by linarith)
    have hA1 : (r.A : ℝ) ^ n + 1 ≤ (r.A : ℝ) ^ (n + 1) := by
      have := mul_le_mul_of_nonneg_left hA (by linarith : (0 : ℝ) ≤ (r.A : ℝ) ^ n)
      rw [pow_succ (r.A : ℝ) n]; linarith
    calc (r.A : ℝ) ^ n + 1 ≤ (r.A : ℝ) ^ (n + 1) := hA1
      _ = Real.exp (((n + 1 : ℕ) : ℝ) * log r.A) := by
          rw [← Real.log_pow, Real.exp_log (by positivity)]
      _ ≤ Real.exp (10 * knX r * log r.A) := by
          apply Real.exp_le_exp.2
          have h : ((n + 1 : ℕ) : ℝ) ≤ 10 * knX r := by push_cast; linarith
          exact mul_le_mul_of_nonneg_right h hlA0.le
  have h2' : ((r.A : ℝ) ^ n + 1) ^ F ≤ Real.exp (F * (10 * knX r * log r.A)) := by
    rw [Real.exp_nat_mul]
    exact pow_le_pow_left₀ (by positivity) h2 F
  -- the exponent
  have hpre : 2 ^ 16 * knΛ r * knX r * log r.A * p₀ ≤ r.ξc ^ 2 * r.s₀ ^ 2 := by
    have := hp
    unfold knMcD at this
    rw [le_div_iff₀ (by positivity)] at this
    linarith
  have hexp : 3 * (F : ℝ) + F * (10 * knX r * log r.A) + log (16 / r.δ)
      ≤ r.ξc ^ 2 / (8 * p₀) := by
    rw [le_div_iff₀ (by positivity)]
    have hs2 : (1 / r.s₀) ^ 2 * r.s₀ ^ 2 = 1 := by field_simp
    have hXA : log 2 ≤ knX r * log r.A := hlA.trans (le_mul_of_one_le_left hlA0.le hX)
    have hbig : 3 * (F : ℝ) + F * (10 * knX r * log r.A) + log (16 / r.δ)
        ≤ 2 ^ 13 * knΛ r * (1 / r.s₀) ^ 2 * (knX r * log r.A) := by
      have a1 : (F : ℝ) * (10 * knX r * log r.A)
          ≤ 2 ^ 8 * knΛ r * (1 / r.s₀) ^ 2 * (10 * (knX r * log r.A)) := by
        rw [show 10 * knX r * log r.A = 10 * (knX r * log r.A) by ring]
        exact mul_le_mul_of_nonneg_right hF (by nlinarith)
      have a2 : 4 ≤ (1 / r.s₀) ^ 2 := by nlinarith
      have hΛ0 : 0 < knΛ r := by linarith
      have a4 : knΛ r * (1 / r.s₀) ^ 2 * log 2 ≤ knΛ r * (1 / r.s₀) ^ 2 * (knX r * log r.A) :=
        mul_le_mul_of_nonneg_left hXA (by positivity)
      have a5 : knΛ r * (1 / r.s₀) ^ 2 * (1 / 2) ≤ knΛ r * (1 / r.s₀) ^ 2 * log 2 :=
        mul_le_mul_of_nonneg_left (by linarith) (by positivity)
      have a6 : knΛ r * 4 ≤ knΛ r * (1 / r.s₀) ^ 2 := mul_le_mul_of_nonneg_left a2 hΛ0.le
      linarith
    have hfin : 2 ^ 13 * knΛ r * (1 / r.s₀) ^ 2 * (knX r * log r.A) * (8 * p₀) ≤ r.ξc ^ 2 := by
      have e : 2 ^ 13 * knΛ r * (1 / r.s₀) ^ 2 * (knX r * log r.A) * (8 * p₀)
          = (1 / r.s₀) ^ 2 * (2 ^ 16 * knΛ r * knX r * log r.A * p₀) := by ring
      rw [e]
      calc (1 / r.s₀) ^ 2 * (2 ^ 16 * knΛ r * knX r * log r.A * p₀)
          ≤ (1 / r.s₀) ^ 2 * (r.ξc ^ 2 * r.s₀ ^ 2) := by gcongr
        _ = r.ξc ^ 2 * ((1 / r.s₀) ^ 2 * r.s₀ ^ 2) := by ring
        _ = r.ξc ^ 2 := by rw [hs2, mul_one]
    linarith [mul_le_mul_of_nonneg_right hbig (show 0 ≤ 8 * p₀ by positivity)]
  have e2 : -2 * (r.ξc / 4) ^ 2 / p₀ = -(r.ξc ^ 2 / (8 * p₀)) := by ring
  rw [e2]
  have hfac : 0 ≤ ((F : ℝ) + 1) ^ 3 := by positivity
  calc 2 * ((F : ℝ) + 1) ^ 3 * ((r.A : ℝ) ^ n + 1) ^ F * Real.exp (-(r.ξc ^ 2 / (8 * p₀)))
      ≤ 2 * Real.exp (3 * F) * Real.exp (F * (10 * knX r * log r.A))
        * Real.exp (-(r.ξc ^ 2 / (8 * p₀))) := by gcongr
    _ = 2 * Real.exp (3 * F + F * (10 * knX r * log r.A) - r.ξc ^ 2 / (8 * p₀)) := by
        rw [sub_eq_add_neg, Real.exp_add, Real.exp_add]; ring
    _ ≤ 2 * Real.exp (-log (16 / r.δ)) := by gcongr; linarith
    _ = r.δ / 8 := by
        rw [Real.exp_neg, Real.exp_log (by positivity), inv_div]
        ring

/-- The lists' slope and slack premise. -/
lemma knSlope_spec (hr : r ∈ rlStarRunRegion) :
    2 * (r.Q : ℝ) * (knNr r : ℝ) / ((knNh r : ℝ) * Real.log 2) * knTu r
      + (r.Q : ℝ) * Real.log ((r.Q : ℝ) / (knδ' r / 4)) / ((knNh r : ℝ) * Real.log 2)
      + (r.Q : ℝ) * (3 * knι r) / knTu r < 1 := by
  obtain ⟨hQ, -⟩ := kn_base hr
  obtain ⟨hsl, -⟩ := knSlope_bounds hr
  obtain ⟨hι, -, hι48, -, -⟩ := knι_bounds hr
  obtain ⟨hNr, -, -⟩ := knNr_bounds hr
  obtain ⟨-, -, hnh1, -, hnhΛ, -⟩ := knNh_bounds hr
  obtain ⟨-, hlQ, -⟩ := knL4 hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hlt := Real.log_two_gt_d9
  have hl0 : 0 < Real.log 2 := by linarith
  have hnh0 : (0 : ℝ) < knNh r := by linarith
  have hTu : knTu r = 1 / (4 * knSlope r) := rfl
  have e1 : knSlope r = 2 * r.Q * (r.Q + 3) * knE r * knRp r * (r.L + 3) * knB r / log 2 := rfl
  -- `Nr/nh` times `2 Q / log 2` is at most the slope bound
  have h1 : 2 * (r.Q : ℝ) * (knNr r : ℝ) / ((knNh r : ℝ) * Real.log 2) ≤ knSlope r := by
    rw [e1, div_le_div_iff₀ (by positivity) hl0]
    have := mul_le_mul_of_nonneg_left hNr (show (0 : ℝ) ≤ 2 * r.Q * log 2 by positivity)
    linarith
  have h1' : 2 * (r.Q : ℝ) * (knNr r : ℝ) / ((knNh r : ℝ) * Real.log 2) * knTu r ≤ 1 / 4 := by
    rw [hTu]
    calc 2 * (r.Q : ℝ) * (knNr r : ℝ) / ((knNh r : ℝ) * Real.log 2) * (1 / (4 * knSlope r))
        ≤ knSlope r * (1 / (4 * knSlope r)) := by gcongr
      _ = 1 / 4 := by field_simp; try ring
  have h2 : (r.Q : ℝ) * Real.log ((r.Q : ℝ) / (knδ' r / 4)) / ((knNh r : ℝ) * Real.log 2)
      ≤ 1 / 4 := by
    rw [div_le_div_iff₀ (by positivity) (by norm_num)]
    have := mul_le_mul_of_nonneg_left hlQ (show (0 : ℝ) ≤ r.Q by linarith)
    have := mul_le_mul_of_nonneg_left (show (2 : ℝ) / 3 ≤ Real.log 2 by linarith) hnh0.le
    linarith
  have h3 : (r.Q : ℝ) * (3 * knι r) / knTu r ≤ 1 / 4 := by
    rw [hTu, div_div_eq_mul_div, div_one]
    linarith
  linarith

lemma knP₀_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knP₀ r ∧ knP₀ r < 1 ∧ knP₀ r ≤ 1 / knCapInv r * knε r / 1536
      ∧ 6 * (r.Q + 2) * knRB r * knP₀ r ≤ 1 / 2
      ∧ 16 * knNh r * knP₀ r ≤ knε r / 48 ∧ knP₀ r ≤ knMcD r := by
  obtain ⟨hQ, -⟩ := kn_base hr
  obtain ⟨hε, hε12, -⟩ := knε_bounds hr
  obtain ⟨hξ, hξ1, -⟩ := ξc_mem hr
  obtain ⟨hcap, -⟩ := knCapInv_bounds hr
  obtain ⟨-, hnh, -⟩ := knNh_bounds hr
  obtain ⟨hRB0, hRB, hRBb1⟩ := knRB_bounds hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hMc := knMcD_pos hr
  have hι1 := knιInv_one hr
  have hc1 : 1 ≤ knCapInv r := by
    refine le_trans ?_ hcap
    have h1 : (1 : ℝ) ≤ r.Q ^ 3 := one_le_pow₀ hQ
    have h2 : (1 : ℝ) ≤ r.Q ^ 3 * knΛ r := one_le_mul_of_one_le_of_one_le h1 (by linarith)
    have h3 : (1 : ℝ) ≤ r.Q ^ 3 * knΛ r * knιInv r := one_le_mul_of_one_le_of_one_le h2 hι1
    linarith
  have hp1 : knP₀ r ≤ knε r / (1536 * knCapInv r) :=
    (min_le_left _ _).trans (min_le_left _ _)
  have hp2 : knP₀ r ≤ 1 / (12 * (r.Q + 2) * knRBb r) :=
    (min_le_left _ _).trans (min_le_right _ _)
  have hp3 : knP₀ r ≤ knε r / (768 * (2 * knCapInv r + 1)) :=
    (min_le_right _ _).trans (min_le_left _ _)
  have hp4 : knP₀ r ≤ knMcD r := (min_le_right _ _).trans (min_le_right _ _)
  have hpos : 0 < knP₀ r :=
    lt_min (lt_min (by positivity) (by positivity)) (lt_min (by positivity) hMc)
  refine ⟨hpos, ?_, ?_, ?_, ?_, hp4⟩
  · refine hp1.trans_lt ?_
    rw [div_lt_one (by positivity)]
    linarith
  · refine hp1.trans (le_of_eq ?_)
    ring
  · have : 0 < 12 * ((r.Q : ℝ) + 2) * knRBb r := by positivity
    rw [le_div_iff₀ this] at hp2
    linarith [mul_le_mul_of_nonneg_left hRB (show 0 ≤ 6 * ((r.Q : ℝ) + 2) * knP₀ r by positivity)]
  · have h0 : 0 < 768 * (2 * knCapInv r + 1) := by positivity
    rw [le_div_iff₀ h0] at hp3
    linarith [mul_le_mul_of_nonneg_left hnh (show 0 ≤ 16 * knP₀ r by positivity)]

end premises

/-! The remaining premises. -/

section rest

variable {r : RLStarRun}

lemma probeSifts_mono {m n : ℕ} (h : m ≤ n) :
    Random.Kappa.probeSifts m ≤ Random.Kappa.probeSifts n :=
  min_le_min (by omega) (by have := Nat.clog_mono_right 2 h; omega)

lemma knTu_pos (hr : r ∈ rlStarRunRegion) : 0 < knTu r := by
  have := (knSlope_bounds hr).1
  unfold knTu; positivity

lemma knb_spec (hr : r ∈ rlStarRunRegion) :
    (knE r : ℝ) * knJ r * (1 - knε r / 192) ^ knB r ≤ knδ' r / 4 := by
  obtain ⟨hε, hε12, -⟩ := knε_bounds hr
  obtain ⟨hξ, hξ1, -⟩ := ξc_mem hr
  obtain ⟨-, -, hb⟩ := knB_bounds hr
  obtain ⟨hE1, -, -⟩ := knE_bounds hr
  obtain ⟨hJ, -, -⟩ := knJ_bounds hr
  obtain ⟨-, -, hnh1, -⟩ := knNh_bounds hr
  obtain ⟨hδ', -, -⟩ := knδ'_bounds hr
  obtain ⟨-, hL9⟩ := knL3 hr
  have hJ1 : (1 : ℝ) ≤ knJ r := hnh1.trans hJ
  have hx : 0 ≤ 1 - knε r / 192 := by linarith
  have h1 : (1 - knε r / 192) ^ knB r ≤ Real.exp (-knΛ r) := by
    calc (1 - knε r / 192) ^ knB r ≤ Real.exp (-(knε r / 192)) ^ knB r :=
          pow_le_pow_left₀ hx (by linarith [Real.add_one_le_exp (-(knε r / 192))]) _
      _ = Real.exp (knB r * (-(knε r / 192))) := by rw [Real.exp_nat_mul]
      _ ≤ Real.exp (-knΛ r) := by
          gcongr
          rw [div_le_iff₀ hε] at hb
          linarith
  have h2 : 4 * (knE r * knJ r) / knδ' r ≤ Real.exp (knΛ r) := by
    rw [← Real.exp_log (show 0 < 4 * ((knE r : ℝ) * knJ r) / knδ' r by positivity)]
    exact Real.exp_le_exp.2 hL9
  rw [div_le_iff₀ hδ'] at h2
  have h3 : Real.exp (-knΛ r) * Real.exp (knΛ r) = 1 := by rw [← Real.exp_add]; simp
  have h4 : (knE r : ℝ) * knJ r * Real.exp (-knΛ r) ≤ knδ' r / 4 := by
    have h5 : Real.exp (-knΛ r) * (Real.exp (knΛ r) * knδ' r) = knδ' r := by
      rw [← mul_assoc, h3, one_mul]
    linarith [mul_le_mul_of_nonneg_left h2 (Real.exp_pos (-knΛ r)).le]
  calc (knE r : ℝ) * knJ r * (1 - knε r / 192) ^ knB r
      ≤ (knE r : ℝ) * knJ r * Real.exp (-knΛ r) := by gcongr
    _ ≤ knδ' r / 4 := h4

lemma knNr_spec (hr : r ∈ rlStarRunRegion) {k : ℕ} (hk : k ≤ r.L) :
    (knE r : ℝ) * ((knB₃ r + knJ r * knB r)
        * ((Random.Kappa.probeSifts (r.L + r.Q + 3 + k) : ℕ) * ((r.Q : ℝ) + 3))
      + (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2))) ≤ knNr r := by
  obtain ⟨hB0, -⟩ := knB₃_bounds hr
  have hp : Random.Kappa.probeSifts (r.L + r.Q + 3 + k) ≤ knRp r := probeSifts_mono (by omega)
  have hp' : ((Random.Kappa.probeSifts (r.L + r.Q + 3 + k) : ℕ) : ℝ) ≤ knRp r := by
    exact_mod_cast hp
  refine le_trans ?_ (Nat.le_ceil _)
  gcongr

lemma knξT_spec (hr : r ∈ rlStarRunRegion) :
    knε r + knε r / 96 + r.ξc / 4 + (1025 / 1024 + 3 / 8) * knεcov r + r.ξc / 4 ≤ r.ξ := by
  obtain ⟨hε, hε12, -⟩ := knε_bounds hr
  obtain ⟨hεc, hεce, -⟩ := knεcov_bounds hr
  obtain ⟨hN1, -⟩ := knN_bounds hr
  obtain ⟨hξ, hξ1, hξξ, -⟩ := ξc_mem hr
  have : knεcov r ≤ r.ξc / 10 := by
    rw [hεce, div_le_div_iff₀ (by positivity) (by norm_num)]
    linarith [mul_le_mul_of_nonneg_left hN1 hξ.le]
  linarith

end rest

end OrthoDFA

end
