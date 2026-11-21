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

/-- The walk start, held to `L`. -/
def knK (r : RLStarRun) : ℕ := min r.k r.L

def knZ (r : RLStarRun) : ℝ := knX r * r.A / (r.δ * r.ξc * r.pAP * r.s₀)

/-- The log every knob's log stays below: `knob_log_le`. -/
def knΛ (r : RLStarRun) : ℝ := 400 * log (knZ r) + 30

def knLg (r : RLStarRun) : ℝ := log (knX r + 1)

def knN (r : RLStarRun) : ℝ := ((r.L + r.Q + 3 : ℕ) : ℝ) + 1 - ((r.L + 1) / 2 : ℕ)

def knε (r : RLStarRun) : ℝ := r.ξc / (4 * (r.A + 1) * knN r)

def knεcov (r : RLStarRun) : ℝ := min (r.ξc / (10 * knN r)) (r.ξc / (8 * knN r))

def knδ' (r : RLStarRun) : ℝ := r.δ / (8 * (r.Q + 1))

def knS (r : RLStarRun) : ℝ := r.Q * ((r.Q + 2) * (r.A + 1) + 1) + 2 * (r.A + 1) + 1

def knE (r : RLStarRun) : ℕ := ⌈96 * (knK r : ℝ) / knε r⌉₊ + 1

def knB (r : RLStarRun) : ℕ := ⌈192 * knΛ r / knε r⌉₊

def knB₃ (r : RLStarRun) : ℝ :=
  36 * knS r / knε r * log (16 * knS r * (knS r + 1) * (knE r * (knE r + 1)) / knδ' r) + knS r + 1

def knRp (r : RLStarRun) : ℕ := Random.Kappa.probeSifts (r.L + r.Q + 3 + knK r)

/-- Level 3's slope at padded probes, at any walk start up to `L`. -/
def knSlope (r : RLStarRun) : ℝ :=
  2 * (r.Q + 1) * (288 / knε r + 16 * (2 + Nat.clog 2 (r.L + r.Q + 3 + r.L)))

def knι (r : RLStarRun) : ℝ := min (knεcov r / 8) (1 / (48 * r.Q * knSlope r))

def knη₀ (r : RLStarRun) : ℝ := 1 / 2 - r.s₀ / 2

def knτ (r : RLStarRun) : ℝ := min ((knεcov r / 2) * (1 - 2 * knη₀ r) ^ 2) (knι r)

/-- A bound on `1/ι`. -/
def knιInv (r : RLStarRun) : ℝ := 2 ^ 22 * r.Q ^ 2 * r.A * knX r * (1 / r.ξc)

/-- A bound on the prefix count `RB`. -/
def knRBb (r : RLStarRun) : ℝ := 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2

/-- A bound on `1/capP`. -/
def knCapInv (r : RLStarRun) : ℝ :=
  2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
    + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2

def knJ (r : RLStarRun) : ℕ := knK r + 2

def knNr (r : RLStarRun) : ℕ :=
  ⌈(knE r : ℝ) * ((knB₃ r + knJ r * knB r) * (knRp r * (r.Q + 3))
    + (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2)))⌉₊

/-- The slack `tw` on a round's disagreement. -/
def knTw (r : RLStarRun) : ℝ := knε r / 12288

def knW (r : RLStarRun) : ℝ :=
  max (knNr r : ℝ) (((r.L + r.Q + 3 : ℕ) + 1 : ℝ) * (r.Q + 2) / knTw r)

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

/-- The light-prefix level `θ`. -/
def knθ (r : RLStarRun) : ℝ := knCapP r * knε r / 192

/-- The heavy draws, missing a prefix heavier than `θ` with chance at most `δ'/4`. -/
def knNh (r : RLStarRun) : ℕ := ⌈8 * log (4 * knX r / (knθ r * knδ' r)) / knθ r⌉₊

def knSC (r : RLStarRun) : ℝ := 8 * (r.Q + 4) * knRB r + 16 * log (16 * (r.Q + 2) / knδ' r) + 2

/-- A bound on the sampler's probes per draw. -/
def knHR (r : RLStarRun) : ℝ := log (4 * (r.Q + 1) * (knSC r + 1) / knδ' r) / (knε r / 384) + 1

/-- The misread term's bound on `p₀`. -/
def knMcD (r : RLStarRun) : ℝ := r.ξc ^ 2 * r.s₀ ^ 2 / (2 ^ 16 * knΛ r * knX r * log r.A)

def knP₀ (r : RLStarRun) : ℝ := min (knCapP r * knε r / 12288) (knMcD r)

/-! The totals. -/

def knPref (r : RLStarRun) : ℝ := (r.Q + 1) * (6 * (r.Q + 2) * knRB r)

def knSuff (r : RLStarRun) : ℝ := (r.Q + 1) * (knPB r + 1)

def knProbes (r : RLStarRun) : ℝ := (r.Q + 1) * (knNh r + knE r * (knB₃ r + knJ r * knB r))

def knTrie (r : RLStarRun) : ℝ := 6 * (r.Q + 2) * knRB r + 2 / knθ r

def knReads (r : RLStarRun) : ℝ :=
  lazyReads (r.Q + 1) (6 * (r.Q + 2) * knRB r) (knPB r + 1) (knFV r)
    (knNr r + (r.Q + 1) * (knSC r * (knHR r * (knRp r * (r.Q + 1))))) r.A (knK r * knTrie r) r.Q

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

lemma knK_bounds (hr : r ∈ rlStarRunRegion) :
    knK r ≤ r.L ∧ (knK r : ℝ) ≤ r.L ∧ (knK r : ℝ) + 1 ≤ knX r := by
  obtain ⟨hQ, -⟩ := kn_base hr
  have h : knK r ≤ r.L := min_le_right _ _
  have h' : (knK r : ℝ) ≤ r.L := by exact_mod_cast h
  unfold knX
  exact ⟨h, h', by linarith⟩

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
      ∧ 2 * (knK r : ℝ) / (knε r / 48) + 1 ≤ knE r
      ∧ (knE r : ℝ) ≤ 2 ^ 12 * r.A * ((knK r : ℝ) + 1) * knX r * (1 / r.ξc) := by
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hQ, hA, hX, -, hLX, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨-, hKL, hKX⟩ := knK_bounds hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hK0 : (0 : ℝ) ≤ knK r := Nat.cast_nonneg _
  have hx : 0 ≤ 96 * (knK r : ℝ) / knε r := by positivity
  have hc := Nat.ceil_lt_add_one hx
  have hc' := Nat.le_ceil (96 * (knK r : ℝ) / knε r)
  have e : (knE r : ℝ) = ⌈96 * (knK r : ℝ) / knε r⌉₊ + 1 := by unfold knE; push_cast; ring
  have h96 : 96 * (knK r : ℝ) / knε r ≤ 2880 * r.A * knK r * knX r * (1 / r.ξc) :=
    calc 96 * (knK r : ℝ) / knε r = 96 * knK r * (1 / knε r) := by ring
      _ ≤ 96 * knK r * (30 * r.A * knX r * (1 / r.ξc)) := by gcongr
      _ = 2880 * r.A * knK r * knX r * (1 / r.ξc) := by ring
  have hm : 2 ≤ r.A * knX r * (1 / r.ξc) := by
    have h1 : 1 ≤ knX r * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le hX iξ
    nlinarith
  have hAXξ : 0 ≤ r.A * knX r * (1 / r.ξc) := by linarith
  have hKX' : (knK r : ℝ) ≤ knX r := by linarith
  have hKm : r.A * knX r * (1 / r.ξc) * knK r ≤ r.A * knX r * (1 / r.ξc) * knX r :=
    mul_le_mul_of_nonneg_left hKX' hAXξ
  have hXm : r.A * knX r * (1 / r.ξc) * 1 ≤ r.A * knX r * (1 / r.ξc) * knX r :=
    mul_le_mul_of_nonneg_left hX hAXξ
  have hKm0 : 0 ≤ r.A * knX r * (1 / r.ξc) * knK r := mul_nonneg hAXξ hK0
  refine ⟨by rw [e]; linarith, ?_, ?_, ?_⟩
  · rw [e]
    have e2 : 2880 * r.A * knK r * knX r * (1 / r.ξc)
        = 2880 * (r.A * knX r * (1 / r.ξc) * knK r) := by ring
    have e3 : 2 ^ 12 * r.A * knX r ^ 2 * (1 / r.ξc)
        = 2 ^ 12 * (r.A * knX r * (1 / r.ξc) * knX r) := by ring
    rw [e3]
    linarith
  · rw [e, div_div_eq_mul_div]
    have : 2 * (knK r : ℝ) * 48 / knε r = 96 * knK r / knε r := by ring
    linarith
  · rw [e]
    have e2 : 2880 * r.A * knK r * knX r * (1 / r.ξc)
        = 2880 * (r.A * knX r * (1 / r.ξc) * knK r) := by ring
    have e3 : 2 ^ 12 * r.A * ((knK r : ℝ) + 1) * knX r * (1 / r.ξc)
        = 2 ^ 12 * (r.A * knX r * (1 / r.ξc) * knK r) + 2 ^ 12 * (r.A * knX r * (1 / r.ξc)) := by
      ring
    rw [e3]
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
  obtain ⟨hKn, hKL, -⟩ := knK_bounds hr
  set n := r.L + r.Q + 3 + knK r with hn
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
    have : (n : ℝ) = r.L + r.Q + 3 + knK r := by rw [hn]; push_cast; ring
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

lemma clog_two_le_self (n : ℕ) : Nat.clog 2 n ≤ n :=
  Nat.clog_le_of_le_pow (by have := n.lt_two_pow_self; omega)

lemma knSlope_bounds (hr : r ∈ rlStarRunRegion) :
    2 ^ 10 * r.Q * r.A * knX r * (1 / r.ξc) ≤ knSlope r
      ∧ knSlope r ≤ 2 ^ 16 * r.Q * r.A * knX r * (1 / r.ξc) := by
  obtain ⟨-, -, hXN⟩ := knN_bounds hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hQ, hA, hX, hQX, hLX, -, -, hξ, hξ1, -⟩ := kn_base hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hL0 : (0 : ℝ) ≤ r.L := Nat.cast_nonneg _
  have hc : ((Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℕ) : ℝ) ≤ 2 * r.L + r.Q + 3 := by
    have h : ((Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℕ) : ℝ) ≤ ((r.L + r.Q + 3 + r.L : ℕ) : ℝ) := by
      exact_mod_cast clog_two_le_self _
    push_cast at h
    linarith
  have hc0 : (0 : ℝ) ≤ ((Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℕ) : ℝ) := Nat.cast_nonneg _
  set M : ℝ := r.A * knX r * (1 / r.ξc) with hM
  have hXM : 2 * knX r ≤ M := by
    have h1 : 2 * knX r ≤ r.A * knX r := by nlinarith
    have h2 : (r.A : ℝ) * knX r ≤ r.A * knX r * (1 / r.ξc) :=
      le_mul_of_one_le_right (by positivity) iξ
    rw [hM]; linarith
  have eε : 1 / knε r = 4 * (r.A + 1) * knN r * (1 / r.ξc) := by
    unfold knε; field_simp
  have hlo : 2 * M ≤ 1 / knε r := by
    rw [eε, hM]
    have : 2 * (r.A : ℝ) * knX r ≤ 4 * (r.A + 1) * knN r := by nlinarith
    have := mul_le_mul_of_nonneg_right this (show 0 ≤ 1 / r.ξc by positivity)
    linarith
  have hhi : 1 / knε r ≤ 30 * M := by rw [hM]; linarith
  have e : knSlope r
      = 2 * (r.Q + 1)
        * (288 * (1 / knε r) + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℝ))) := by
    unfold knSlope; ring
  have eQ : 2 ^ 10 * (r.Q : ℝ) * r.A * knX r * (1 / r.ξc) = 2 ^ 10 * r.Q * M := by rw [hM]; ring
  have eQ' : 2 ^ 16 * (r.Q : ℝ) * r.A * knX r * (1 / r.ξc) = 2 ^ 16 * r.Q * M := by rw [hM]; ring
  rw [e, eQ, eQ']
  have hM0 : 0 ≤ M := by linarith
  constructor
  · have h1 : 2 * (r.Q : ℝ) * (288 * (2 * M)) ≤ 2 * (r.Q + 1)
        * (288 * (1 / knε r) + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℝ))) := by
      gcongr
      · linarith
      · linarith
    nlinarith
  · have h16 : 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℝ)) ≤ 56 * M := by
      have hXe : knX r = r.L + r.Q := rfl
      linarith
    have h1 : 288 * (1 / knε r) + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + r.L) : ℝ))
        ≤ 8696 * M := by linarith
    have h2 : 2 * ((r.Q : ℝ) + 1) ≤ 4 * r.Q := by linarith
    have h3 := mul_le_mul h2 h1 (by positivity) (by positivity)
    nlinarith

lemma knιInv_ge (hr : r ∈ rlStarRunRegion) : 2 ^ 23 * knX r * (1 / r.ξc) ≤ knιInv r := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have h : (2 : ℝ) ≤ r.Q ^ 2 * r.A := by nlinarith [one_le_pow₀ hQ (n := 2)]
  have h0 : (0 : ℝ) ≤ knX r * (1 / r.ξc) := by positivity
  have := mul_le_mul_of_nonneg_right h h0
  unfold knιInv
  nlinarith

lemma knιInv_one (hr : r ∈ rlStarRunRegion) : 1 ≤ knιInv r := by
  obtain ⟨-, -, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have h := knιInv_ge hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  nlinarith [one_le_mul_of_one_le_of_one_le hX iξ]

lemma knι_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knι r ∧ knι r ≤ knεcov r / 8 ∧ 48 * r.Q * knSlope r * knι r ≤ 1
      ∧ 1 / knι r ≤ knιInv r ∧ knι r ≤ 1 / 80
      ∧ 2 ^ 15 * r.Q ^ 2 * r.A * knX r * (1 / r.ξc) * knι r ≤ 1 := by
  obtain ⟨hεc, -, hiεc, -, hεc1⟩ := knεcov_bounds hr
  obtain ⟨hslo, hsl⟩ := knSlope_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hP : 0 < knX r * (1 / r.ξc) := by positivity
  have hsl0 : 0 < knSlope r := lt_of_lt_of_le (by positivity) hslo
  have h0 : 0 < 48 * (r.Q : ℝ) * knSlope r := by positivity
  have hι0 : 0 < knι r := lt_min (by positivity) (by positivity)
  have hι8 : knι r ≤ knεcov r / 8 := min_le_left _ _
  have hι48 : knι r ≤ 1 / (48 * r.Q * knSlope r) := min_le_right _ _
  have h48 : 48 * r.Q * knSlope r * knι r ≤ 1 := by
    rw [le_div_iff₀ h0] at hι48; linarith
  refine ⟨hι0, hι8, h48, ?_, by linarith, ?_⟩
  · have hq : (2 : ℝ) ≤ r.Q ^ 2 * r.A := by nlinarith [one_le_pow₀ hQ (n := 2)]
    have hq' := mul_le_mul_of_nonneg_right hq hP.le
    calc 1 / knι r ≤ 1 / (knεcov r / 8) + 1 / (1 / (48 * r.Q * knSlope r)) :=
          one_div_min_le (by positivity) (by positivity)
      _ = 8 * (1 / knεcov r) + 48 * r.Q * knSlope r := by rw [one_div_one_div]; ring
      _ ≤ 8 * (50 * knX r * (1 / r.ξc))
          + 48 * r.Q * (2 ^ 16 * r.Q * r.A * knX r * (1 / r.ξc)) := by gcongr
      _ = 400 * (knX r * (1 / r.ξc)) + 3 * 2 ^ 20 * (r.Q ^ 2 * r.A * (knX r * (1 / r.ξc))) := by
          ring
      _ ≤ knιInv r := by unfold knιInv; nlinarith
  · have h1 : 2 ^ 15 * (r.Q : ℝ) ^ 2 * r.A * knX r * (1 / r.ξc) ≤ 48 * r.Q * knSlope r := by
      have := mul_le_mul_of_nonneg_left hslo (show (0 : ℝ) ≤ 48 * r.Q by positivity)
      nlinarith
    nlinarith [mul_le_mul_of_nonneg_right h1 hι0.le]

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

lemma knιInv_Z (hr : r ∈ rlStarRunRegion) : knιInv r ≤ 2 ^ 22 * knZ r ^ 5 := by
  obtain ⟨hZ8, hXZ, hAZ, -, hξZ, -, -, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  unfold knιInv
  calc 2 ^ 22 * (r.Q : ℝ) ^ 2 * r.A * knX r * (1 / r.ξc)
      ≤ 2 ^ 22 * knZ r ^ 2 * knZ r * knZ r * knZ r := by gcongr
    _ = 2 ^ 22 * knZ r ^ 5 := by ring

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

lemma knL2 (hr : r ∈ rlStarRunRegion) : log (20 / knτ r) ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hτ, -, hiτ⟩ := knτ_bounds hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, -, -, hs, -⟩ := kn_base hr
  have hι := knιInv_Z hr
  have hι0 : 0 ≤ knιInv r := by linarith [knιInv_one hr]
  refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 40 * 2 ^ 22) (n := 7) (j := 0)
    (k := 10) (by norm_num) ?_ (by norm_num) (by norm_num))
  calc 20 / knτ r = 20 * (1 / knτ r) := by ring
    _ ≤ 20 * (2 * (2 ^ 22 * knZ r ^ 5) * knZ r ^ 2) := by
        gcongr; exact hiτ.trans (by gcongr)
    _ = 40 * 2 ^ 22 * knZ r ^ 7 * knΛ r ^ 0 := by ring

lemma knL4 (hr : r ∈ rlStarRunRegion) :
    log (3 / (knδ' r / 2)) ≤ knΛ r ∧ log (16 * (r.Q + 2) / knδ' r) ≤ knΛ r
      ∧ log (16 / r.δ) ≤ knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, -, -, -, -, hδ, -⟩ := kn_base hr
  obtain ⟨hδ', -, hiδ'⟩ := knδ'_bounds hr
  refine ⟨knob_log_le hr (by positivity) (knob_pow_le hr (c := 96) (n := 2) (j := 0) (k := 3)
    (by norm_num) ?_ (by norm_num) (by norm_num)),
    knob_log_le hr (by positivity) (knob_pow_le hr (c := 768) (n := 3) (j := 0) (k := 4)
    (by norm_num) ?_ (by norm_num) (by norm_num)),
    knob_log_le hr (by positivity) (knob_pow_le hr (c := 16) (n := 1) (j := 0) (k := 2)
    (by norm_num) ?_ (by norm_num) (by norm_num))⟩
  · calc 3 / (knδ' r / 2) = 6 * (1 / knδ' r) := by field_simp; norm_num
      _ ≤ 6 * (16 * knZ r * knZ r) := by gcongr; exact hiδ'.trans (by gcongr)
      _ = 96 * knZ r ^ 2 * knΛ r ^ 0 := by ring
  · calc 16 * ((r.Q : ℝ) + 2) / knδ' r = 16 * (r.Q + 2) * (1 / knδ' r) := by ring
      _ ≤ 16 * (3 * knZ r) * (16 * knZ r * knZ r) := by
          gcongr
          · linarith
          · exact hiδ'.trans (by gcongr)
      _ = 768 * knZ r ^ 3 * knΛ r ^ 0 := by ring
  · calc 16 / r.δ = 16 * (1 / r.δ) := by ring
      _ ≤ 16 * knZ r := by gcongr
      _ = 16 * knZ r ^ 1 * knΛ r ^ 0 := by ring

end logs

/-! The rounds' sizes. -/

section sizes

variable {r : RLStarRun}

lemma knCapInv_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ knCapInv r ∧ knCapInv r ≤ 2 ^ 128 * knZ r ^ 22 * knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, -, hp, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hι1 := knιInv_one hr
  have hιZ := knιInv_Z hr
  have is : 2 ≤ 1 / r.s₀ := by rw [le_div_iff₀ hs]; linarith
  have hZ1 : 1 ≤ knZ r := by linarith
  constructor
  · have h1 : (1 : ℝ) ≤ r.Q * knΛ r := one_le_mul_of_one_le_of_one_le hQ (by linarith)
    have h2 : (1 : ℝ) ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 :=
      one_le_mul_of_one_le_of_one_le h1 (one_le_pow₀ (by linarith))
    have h3 : (1 : ℝ) ≤ r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 :=
      one_le_mul_of_one_le_of_one_le h2 (one_le_pow₀ hι1)
    have h4 : (1 : ℝ) ≤ (1 / r.s₀) ^ 2 + r.Q ^ 2 := by
      have := one_le_pow₀ hQ (n := 2)
      have : (0 : ℝ) ≤ (1 / r.s₀) ^ 2 := by positivity
      linarith
    have h5 := one_le_mul_of_one_le_of_one_le h3 h4
    have h6 : (0 : ℝ) ≤ 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2 := by positivity
    unfold knCapInv
    nlinarith
  · unfold knCapInv
    calc 2 ^ 60 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 3 * ((1 / r.s₀) ^ 2 + r.Q ^ 2)
          + 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 2 * (1 / r.pAP) ^ 2
        ≤ 2 ^ 60 * knZ r * knΛ r * knZ r ^ 4 * (2 ^ 22 * knZ r ^ 5) ^ 3
            * (knZ r ^ 2 + knZ r ^ 2)
          + 2 ^ 30 * knZ r * knΛ r * knZ r ^ 2 * knZ r ^ 2 := by
          gcongr
      _ = 2 ^ 127 * knZ r ^ 22 * knΛ r + 2 ^ 30 * knZ r ^ 5 * knΛ r := by ring
      _ ≤ 2 ^ 128 * knZ r ^ 22 * knΛ r := by
          have h1 : knZ r ^ 5 ≤ knZ r ^ 22 := pow_le_pow_right₀ hZ1 (by norm_num)
          have h3 := mul_le_mul_of_nonneg_right h1 (show 0 ≤ knΛ r by linarith)
          have h5 : (0 : ℝ) ≤ knZ r ^ 22 * knΛ r := by positivity
          nlinarith

lemma knJ_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ (knJ r : ℝ) ∧ (knJ r : ℝ) ≤ 3 * knX r ∧ knK r + 2 ≤ knJ r
      ∧ (knJ r : ℝ) ≤ 3 * ((knK r : ℝ) + 1) := by
  obtain ⟨-, -, hX, -⟩ := kn_base hr
  obtain ⟨-, -, hKX⟩ := knK_bounds hr
  have e : (knJ r : ℝ) = knK r + 2 := by unfold knJ; push_cast; ring
  have hK0 : (0 : ℝ) ≤ knK r := Nat.cast_nonneg _
  rw [e]
  exact ⟨by linarith, by linarith, le_rfl, by linarith⟩

/-- `E ((B₃ + J b) u + V) ≤ 2³⁵ Q³ A³ κ² X² ξi² Λ lg` from each factor's bound. -/
lemma nr_body_le {E B₃ J b u V Q A X κ ξi Λ lg : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X)
    (hκ : 1 ≤ κ) (hξi : 1 ≤ ξi) (hΛ : 30 ≤ Λ) (hlg : 1 / 2 ≤ lg)
    (hE1 : 1 ≤ E) (hE : E ≤ 2 ^ 12 * A * κ * X * ξi)
    (hB0 : 0 ≤ B₃) (hB : B₃ ≤ 2 ^ 14 * Q ^ 2 * A ^ 2 * X * ξi * Λ)
    (hJ1 : 1 ≤ J) (hJ : J ≤ 3 * κ) (hb1 : 1 ≤ b) (hb : b ≤ 2 ^ 13 * A * X * ξi * Λ)
    (hu1 : 1 ≤ u) (hu : u ≤ 64 * Q * lg) (hV0 : 0 ≤ V) (hV : V ≤ 41 * Q ^ 3 * A) :
    1 ≤ E * ((B₃ + J * b) * u + V)
      ∧ E * ((B₃ + J * b) * u + V) ≤ 2 ^ 35 * Q ^ 3 * A ^ 3 * κ ^ 2 * X ^ 2 * ξi ^ 2 * Λ * lg := by
  obtain ⟨m, hm⟩ : ∃ m : ℝ, m = κ * Q ^ 2 * A ^ 2 * X * ξi * Λ := ⟨_, rfl⟩
  have hQ2 : 1 ≤ Q ^ 2 := one_le_pow₀ hQ
  have hA2 : A ≤ A ^ 2 := le_self_pow₀ (by linarith) (by norm_num)
  have hm0 : 0 ≤ m := by rw [hm]; positivity
  have hBm : B₃ ≤ 2 ^ 14 * m := by
    have h1 : Q ^ 2 * A ^ 2 * X * ξi * Λ ≤ κ * (Q ^ 2 * A ^ 2 * X * ξi * Λ) :=
      le_mul_of_one_le_left (by positivity) hκ
    have e : m = κ * (Q ^ 2 * A ^ 2 * X * ξi * Λ) := by rw [hm]; ring
    rw [e]; linarith
  have hJbm : J * b ≤ 3 * 2 ^ 13 * m := by
    have h1 := mul_le_mul hJ hb (by linarith) (by linarith)
    have h3 : A ≤ Q ^ 2 * A ^ 2 := hA2.trans (le_mul_of_one_le_left (by positivity) hQ2)
    have h4 := mul_le_mul_of_nonneg_right h3 (show 0 ≤ κ * X * ξi * Λ by positivity)
    have e1 : 3 * κ * (2 ^ 13 * A * X * ξi * Λ) = 3 * 2 ^ 13 * (A * (κ * X * ξi * Λ)) := by ring
    have e2 : m = Q ^ 2 * A ^ 2 * (κ * X * ξi * Λ) := by rw [hm]; ring
    rw [e2]; linarith
  have hVm : V ≤ 2 ^ 22 * m * Q * lg := by
    have c1 : 1 ≤ κ * X * ξi * Λ :=
      one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le
        (one_le_mul_of_one_le_of_one_le hκ hX) hξi) (by linarith)
    have b1 := mul_le_mul hA2 c1 (by linarith) (by positivity)
    have a1 : Q ^ 3 * A ≤ Q * m := by
      have := mul_le_mul_of_nonneg_left b1 (show 0 ≤ Q ^ 3 by positivity)
      have e : Q * m = Q ^ 3 * (A ^ 2 * (κ * X * ξi * Λ)) := by rw [hm]; ring
      rw [e]; linarith
    have a2 : 0 ≤ Q * m := by positivity
    have a3 : Q * m ≤ Q * m * (2 * lg) := le_mul_of_one_le_right a2 (by linarith)
    have a4 : 0 ≤ Q * m * lg := by positivity
    nlinarith
  have h1 : (B₃ + J * b) * u ≤ (2 ^ 16 * m) * (64 * Q * lg) :=
    mul_le_mul (by linarith) hu (by linarith) (by positivity)
  have h2 : (B₃ + J * b) * u + V ≤ 2 ^ 23 * m * Q * lg := by nlinarith
  have h3 := mul_le_mul hE h2 (by positivity) (by positivity)
  have e : 2 ^ 12 * A * κ * X * ξi * (2 ^ 23 * m * Q * lg)
      = 2 ^ 35 * Q ^ 3 * A ^ 3 * κ ^ 2 * X ^ 2 * ξi ^ 2 * Λ * lg := by rw [hm]; ring
  refine ⟨?_, by linarith⟩
  have hJb : 1 ≤ J * b := one_le_mul_of_one_le_of_one_le hJ1 hb1
  have h4 : 1 ≤ (B₃ + J * b) * u := one_le_mul_of_one_le_of_one_le (by linarith) hu1
  exact one_le_mul_of_one_le_of_one_le hE1 (by linarith)

lemma knNr_bounds (hr : r ∈ rlStarRunRegion) :
    (knNr r : ℝ) ≤ 2 ^ 36 * r.Q ^ 3 * r.A ^ 3 * ((knK r : ℝ) + 1) ^ 2 * knX r ^ 2 * (1 / r.ξc) ^ 2
        * knΛ r * knLg r
      ∧ 1 ≤ (knNr r : ℝ) := by
  obtain ⟨hE1, -, -, hE⟩ := knE_bounds hr
  obtain ⟨hb1, hb, -⟩ := knB_bounds hr
  obtain ⟨hrp1, hrp⟩ := knRp_bounds hr
  obtain ⟨hB₃0, hB₃⟩ := knB₃_bounds hr
  obtain ⟨hJ1, -, -, hJκ⟩ := knJ_bounds hr
  obtain ⟨hQ, hA, hX, hQX, hLX, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  obtain ⟨hlg, -⟩ := knLg_bounds hr
  have hκ : (1 : ℝ) ≤ (knK r : ℝ) + 1 := by have := (knK r).cast_nonneg (α := ℝ); linarith
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hu1 : (1 : ℝ) ≤ knRp r * ((r.Q : ℝ) + 3) := by
    have := mul_le_mul hrp1 (show (1 : ℝ) ≤ r.Q + 3 by linarith) zero_le_one (by linarith)
    linarith
  have hu : (knRp r : ℝ) * ((r.Q : ℝ) + 3) ≤ 64 * r.Q * knLg r := by
    have := mul_le_mul hrp (show (r.Q : ℝ) + 3 ≤ 4 * r.Q by linarith) (by linarith)
      (by linarith)
    linarith
  have hV0 : (0 : ℝ) ≤ ((r.Q : ℝ) + 2) * (((r.Q : ℝ) + 2) * ((r.A : ℝ) + 1) * ((r.Q : ℝ) + 2)) := by
    positivity
  have hV : ((r.Q : ℝ) + 2) * (((r.Q : ℝ) + 2) * ((r.A : ℝ) + 1) * ((r.Q : ℝ) + 2))
      ≤ 41 * r.Q ^ 3 * r.A := by
    have g1 : ((r.Q : ℝ) + 2) ^ 3 ≤ (3 * r.Q) ^ 3 := pow_le_pow_left₀ (by linarith) (by linarith) 3
    have g2 : (r.A : ℝ) + 1 ≤ 3 / 2 * r.A := by linarith
    have g3 := mul_le_mul g1 g2 (by linarith) (by positivity)
    have : (0 : ℝ) ≤ r.Q ^ 3 * r.A := by positivity
    nlinarith
  obtain ⟨h1, h2⟩ := nr_body_le hQ hA hX hκ iξ hΛ hlg hE1 hE hB₃0 hB₃ hJ1 hJκ hb1 hb hu1 hu hV0 hV
  have hc : (knNr r : ℝ) < (knE r : ℝ) * ((knB₃ r + knJ r * knB r) * (knRp r * ((r.Q : ℝ) + 3))
      + ((r.Q : ℝ) + 2) * (((r.Q : ℝ) + 2) * ((r.A : ℝ) + 1) * ((r.Q : ℝ) + 2))) + 1 := by
    unfold knNr; exact Nat.ceil_lt_add_one (by linarith)
  have hc' : (knE r : ℝ) * ((knB₃ r + knJ r * knB r) * (knRp r * ((r.Q : ℝ) + 3))
      + ((r.Q : ℝ) + 2) * (((r.Q : ℝ) + 2) * ((r.A : ℝ) + 1) * ((r.Q : ℝ) + 2))) ≤ knNr r := by
    unfold knNr; exact Nat.le_ceil _
  constructor <;> linarith

lemma knW_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ knW r ∧ (knNr r : ℝ) ≤ knW r
      ∧ (((r.L + r.Q + 3 : ℕ) : ℝ) + 1) * (r.Q + 2) ≤ knTw r * knW r
      ∧ knW r ≤ knNr r + 2 ^ 23 * r.Q * r.A * knX r ^ 2 * (1 / r.ξc) := by
  obtain ⟨-, hNr1⟩ := knNr_bounds hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hQ, hA, hX, hQX, hLX, -, -, hξ, -⟩ := kn_base hr
  have htw : 0 < knTw r := by unfold knTw; positivity
  have hw : (((r.L + r.Q + 3 : ℕ) : ℝ) + 1) * (r.Q + 2) / knTw r ≤ knW r := le_max_right _ _
  refine ⟨hNr1.trans (le_max_left _ _), le_max_left _ _, ?_, ?_⟩
  · rw [div_le_iff₀ htw] at hw
    linarith
  · unfold knW
    refine max_le (by linarith [show 0 ≤ 2 ^ 23 * r.Q * r.A * knX r ^ 2 * (1 / r.ξc) by positivity])
      ?_
    have e : (((r.L + r.Q + 3 : ℕ) : ℝ) + 1) * (r.Q + 2) / knTw r
        = 12288 * ((r.L : ℝ) + r.Q + 4) * (r.Q + 2) * (1 / knε r) := by
      unfold knTw; push_cast; field_simp; ring
    rw [e]
    have h1 : ((r.L : ℝ) + r.Q + 4) * (r.Q + 2) ≤ (5 * knX r) * (3 * r.Q) := by
      have hXe : knX r = r.L + r.Q := rfl
      exact mul_le_mul (by linarith) (by linarith) (by positivity) (by positivity)
    have h2 := mul_le_mul h1 hiε (by positivity) (by positivity)
    have : (0 : ℝ) ≤ knNr r := Nat.cast_nonneg _
    have : (0 : ℝ) ≤ r.Q * r.A * knX r ^ 2 * (1 / r.ξc) := by positivity
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
  obtain ⟨hNr, -⟩ := knNr_bounds hr
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hJ1, hJX, -⟩ := knJ_bounds hr
  have hZ1 : 1 ≤ knZ r := by linarith
  have hΛ0 : 0 < knΛ r := by linarith
  constructor
  · refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 2 ^ 44) (n := 15) (j := 1)
      (k := 15) (by norm_num) ?_ (by norm_num) (by norm_num))
    obtain ⟨-, -, hKX⟩ := knK_bounds hr
    have hNrZ : (knNr r : ℝ) ≤ 2 ^ 36 * knZ r ^ 13 * knΛ r := by
      calc (knNr r : ℝ)
          ≤ 2 ^ 36 * r.Q ^ 3 * r.A ^ 3 * ((knK r : ℝ) + 1) ^ 2 * knX r ^ 2 * (1 / r.ξc) ^ 2
            * knΛ r * knLg r := hNr
        _ ≤ 2 ^ 36 * knZ r ^ 3 * knZ r ^ 3 * knZ r ^ 2 * knZ r ^ 2 * knZ r ^ 2 * knΛ r * knZ r := by
            gcongr
            · linarith
            · linarith
        _ = 2 ^ 36 * knZ r ^ 13 * knΛ r := by ring
    have h23 : 2 ^ 23 * r.Q * r.A * knX r ^ 2 * (1 / r.ξc) ≤ 2 ^ 36 * knZ r ^ 13 * knΛ r := by
      have a1 : 2 ^ 23 * r.Q * r.A * knX r ^ 2 * (1 / r.ξc)
          ≤ 2 ^ 23 * knZ r * knZ r * knZ r ^ 2 * knZ r := by
        gcongr
      have a2 : knZ r ^ 5 ≤ knZ r ^ 13 := pow_le_pow_right₀ hZ1 (by norm_num)
      have a3 : knZ r ^ 13 ≤ knZ r ^ 13 * knΛ r :=
        le_mul_of_one_le_right (by positivity) (by linarith)
      have e : 2 ^ 23 * knZ r * knZ r * knZ r ^ 2 * knZ r = 2 ^ 23 * knZ r ^ 5 := by ring
      have : (0 : ℝ) ≤ knZ r ^ 13 * knΛ r := by positivity
      linarith
    calc 8 * knW r / knδ' r = 8 * knW r * (1 / knδ' r) := by ring
      _ ≤ 8 * (2 ^ 36 * knZ r ^ 13 * knΛ r + 2 ^ 36 * knZ r ^ 13 * knΛ r)
          * (16 * knZ r * knZ r) := by
          gcongr
          · linarith
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 44 * knZ r ^ 15 * knΛ r ^ 1 := by ring
  · refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 3 * 2 ^ 18) (n := 7) (j := 0)
      (k := 7) (by norm_num) ?_ (by norm_num) (by norm_num))
    calc 4 * (knE r * knJ r) / knδ' r = 4 * (knE r * knJ r) * (1 / knδ' r) := by ring
      _ ≤ 4 * ((2 ^ 12 * knZ r * knZ r ^ 2 * knZ r) * (3 * knZ r)) * (16 * knZ r * knZ r) := by
          gcongr
          · exact hE.trans (by gcongr)
          · exact hJX.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
      _ = 3 * 2 ^ 18 * knZ r ^ 7 * knΛ r ^ 0 := by ring

end sizes

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
    (knob_pow_le hr (c := 5 * 2 ^ 45) (n := 16) (j := 1) (k := 16) (by norm_num) ?_
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
          * knZ r * (2 * (2 ^ 22 * knZ r ^ 5) * knZ r ^ 2) := by
          gcongr
          · linarith
          · exact hPB.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
          · exact hiτ.trans (by gcongr)
      _ = 5 * 2 ^ 45 * knZ r ^ 16 * knΛ r ^ 1 := by ring

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

lemma knRBb_Z (hr : r ∈ rlStarRunRegion) : knRBb r ≤ 2 ^ 74 * knZ r ^ 15 * knΛ r := by
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, hpZ, hsZ, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, -⟩ := kn_base hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hιZ := knιInv_Z hr
  have hι1 := knιInv_one hr
  unfold knRBb
  calc 2 ^ 30 * r.Q * knΛ r * (1 / r.s₀) ^ 4 * knιInv r ^ 2
      ≤ 2 ^ 30 * knZ r * knΛ r * knZ r ^ 4 * (2 ^ 22 * knZ r ^ 5) ^ 2 := by
        gcongr
    _ = 2 ^ 74 * knZ r ^ 15 * knΛ r := by ring

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
  refine ⟨Real.log_nonneg ?_, knob_log_le hr hy0 (knob_pow_le hr (c := 2 ^ 101) (n := 21)
    (j := 2) (k := 34) (by norm_num) ?_ (by norm_num) (by norm_num))⟩
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
          * (6 * (2 ^ 74 * knZ r ^ 15 * knΛ r)) * (16 * knZ r * knZ r) := by
          gcongr
          · linarith
          · exact hlb.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 96 * 18 * knZ r ^ 21 * knΛ r ^ 2 := by ring
      _ ≤ 2 ^ 101 * knZ r ^ 21 * knΛ r ^ 2 := by gcongr; norm_num

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

/-! The heavy draws and the sampler. -/

section draws

variable {r : RLStarRun}

lemma knCapP_pos (hr : r ∈ rlStarRunRegion) : 0 < knCapP r :=
  lt_of_lt_of_le (by have := (knCapInv_bounds hr).1; positivity) (knCapP_ge hr)

lemma knCapP_le_one (hr : r ∈ rlStarRunRegion) : knCapP r ≤ 1 := by
  obtain ⟨hRB0, -⟩ := knRB_bounds hr
  have h : knCapP r ≤ 1 / (6 * (r.Q + 2) * (knRB r + 1)) :=
    (min_le_right _ _).trans ((min_le_right _ _).trans (min_le_right _ _))
  refine h.trans ?_
  rw [div_le_one (by positivity)]
  have : (0 : ℝ) ≤ r.Q := Nat.cast_nonneg _
  nlinarith

lemma knθ_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knθ r ∧ knθ r ≤ 1 ∧ 1 / knθ r ≤ 2 ^ 13 * knCapInv r * r.A * knX r * (1 / r.ξc) := by
  have hc := knCapP_ge hr
  have hcp := knCapP_pos hr
  have hcp1 := knCapP_le_one hr
  obtain ⟨hc1, -⟩ := knCapInv_bounds hr
  obtain ⟨hε, hε12, hiε⟩ := knε_bounds hr
  obtain ⟨-, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  have hic : 1 / knCapP r ≤ knCapInv r := (one_div_le hcp (by linarith)).2 hc
  refine ⟨by unfold knθ; positivity, ?_, ?_⟩
  · unfold knθ
    rw [div_le_one (by norm_num)]
    nlinarith
  · have e : 1 / knθ r = 192 * (1 / knCapP r) * (1 / knε r) := by unfold knθ; field_simp
    rw [e]
    calc 192 * (1 / knCapP r) * (1 / knε r)
        ≤ 192 * knCapInv r * (30 * r.A * knX r * (1 / r.ξc)) := by gcongr
      _ = 5760 * knCapInv r * r.A * knX r * (1 / r.ξc) := by ring
      _ ≤ 2 ^ 13 * knCapInv r * r.A * knX r * (1 / r.ξc) := by
          gcongr
          norm_num

lemma knNh_bounds (hr : r ∈ rlStarRunRegion) :
    (knNh r : ℝ) ≤ 2 ^ 17 * knCapInv r * r.A * knX r * (1 / r.ξc) * knΛ r
      ∧ ∀ k : ℕ, k ≤ r.L → 8 * log (4 * k / (knθ r * knδ' r)) / knθ r ≤ knNh r := by
  obtain ⟨hθ, hθ1, hiθ⟩ := knθ_bounds hr
  obtain ⟨hδ', hδ'1, hiδ'⟩ := knδ'_bounds hr
  obtain ⟨hc1, hcZ⟩ := knCapInv_bounds hr
  obtain ⟨hQ, hA, hX, -, hLX, hδ, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, -, -, hQZ⟩ := knZ_ge hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hlog : log (4 * knX r / (knθ r * knδ' r)) ≤ knΛ r := by
    refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 2 ^ 147) (n := 28) (j := 1)
      (k := 49) (by norm_num) ?_ (by norm_num) (by norm_num))
    calc 4 * knX r / (knθ r * knδ' r) = 4 * knX r * (1 / knθ r) * (1 / knδ' r) := by
          field_simp
      _ ≤ 4 * knZ r * (2 ^ 13 * (2 ^ 128 * knZ r ^ 22 * knΛ r) * knZ r * knZ r * knZ r)
          * (16 * knZ r * knZ r) := by
          gcongr
          · exact hiθ.trans (by gcongr)
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 147 * knZ r ^ 28 * knΛ r ^ 1 := by ring
  have hlog0 : 0 ≤ log (4 * knX r / (knθ r * knδ' r)) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ (by positivity)]
    have : knθ r * knδ' r ≤ 1 := mul_le_one₀ hθ1 hδ'.le (by linarith)
    linarith
  have hx0 : 0 ≤ 8 * log (4 * knX r / (knθ r * knδ' r)) / knθ r := by positivity
  have hc := Nat.ceil_lt_add_one hx0
  have hc' := Nat.le_ceil (8 * log (4 * knX r / (knθ r * knδ' r)) / knθ r)
  have hN : knNh r = ⌈8 * log (4 * knX r / (knθ r * knδ' r)) / knθ r⌉₊ := rfl
  rw [hN]
  refine ⟨?_, fun k hk => ?_⟩
  · have h1 : 8 * log (4 * knX r / (knθ r * knδ' r)) / knθ r ≤ 8 * knΛ r * (1 / knθ r) := by
      rw [div_eq_mul_one_div]; gcongr
    have h2 := mul_le_mul_of_nonneg_left hiθ (show 0 ≤ 8 * knΛ r by linarith)
    have h3 : (1 : ℝ) ≤ 2 ^ 13 * knCapInv r * r.A * knX r * (1 / r.ξc) * knΛ r := by
      have a1 : (1 : ℝ) ≤ knCapInv r * r.A := one_le_mul_of_one_le_of_one_le hc1 (by linarith)
      have a2 : (1 : ℝ) ≤ knCapInv r * r.A * knX r := one_le_mul_of_one_le_of_one_le a1 hX
      have a3 : (1 : ℝ) ≤ knCapInv r * r.A * knX r * (1 / r.ξc) :=
        one_le_mul_of_one_le_of_one_le a2 iξ
      have a4 : (1 : ℝ) ≤ knCapInv r * r.A * knX r * (1 / r.ξc) * knΛ r :=
        one_le_mul_of_one_le_of_one_le a3 (by linarith)
      nlinarith
    nlinarith
  · rcases Nat.eq_zero_or_pos k with rfl | hk0
    · simp only [Nat.cast_zero, mul_zero, zero_div, Real.log_zero]
      exact Nat.cast_nonneg _
    refine le_trans ?_ hc'
    have hk' : (k : ℝ) ≤ knX r := (Nat.cast_le.2 hk).trans hLX
    have hk1 : (0 : ℝ) < k := by exact_mod_cast hk0
    gcongr

lemma knSC_bounds (hr : r ∈ rlStarRunRegion) : 0 ≤ knSC r ∧ knSC r ≤ 2 ^ 6 * r.Q * knRBb r := by
  obtain ⟨hRB0, hRB, hRBb1⟩ := knRB_bounds hr
  obtain ⟨-, hl, -⟩ := knL4 hr
  obtain ⟨hQ, -, -, -, -, -, -, -, -, -, -, hs, hs1⟩ := kn_base hr
  obtain ⟨hδ', hδ'1, -⟩ := knδ'_bounds hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have hι1 := knιInv_one hr
  have hl0 : 0 ≤ log (16 * (r.Q + 2) / knδ' r) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ']; nlinarith)
  have hΛR : 2 ^ 30 * knΛ r ≤ knRBb r := by
    have a1 : (1 : ℝ) ≤ r.Q * (1 / r.s₀) ^ 4 :=
      one_le_mul_of_one_le_of_one_le hQ (one_le_pow₀ (by rw [le_div_iff₀ hs]; linarith))
    have a2 : (1 : ℝ) ≤ r.Q * (1 / r.s₀) ^ 4 * knιInv r ^ 2 :=
      one_le_mul_of_one_le_of_one_le a1 (one_le_pow₀ hι1)
    have := mul_le_mul_of_nonneg_left a2 (show (0 : ℝ) ≤ 2 ^ 30 * knΛ r by positivity)
    unfold knRBb
    nlinarith
  unfold knSC
  constructor
  · positivity
  · have h1 : 8 * ((r.Q : ℝ) + 4) * knRB r ≤ 40 * r.Q * knRBb r := by
      have := mul_le_mul (show 8 * ((r.Q : ℝ) + 4) ≤ 40 * r.Q by linarith) hRB hRB0
        (by positivity)
      linarith
    have h2 : (r.Q : ℝ) * knRBb r ≥ knRBb r := le_mul_of_one_le_left (by linarith) hQ
    nlinarith

lemma knHR_bounds (hr : r ∈ rlStarRunRegion) :
    1 ≤ knHR r ∧ knHR r ≤ 2 ^ 14 * r.A * knX r * (1 / r.ξc) * knΛ r := by
  obtain ⟨hSC0, hSC⟩ := knSC_bounds hr
  obtain ⟨hRB0, hRB, hRBb1⟩ := knRB_bounds hr
  have hRBZ := knRBb_Z hr
  obtain ⟨hZ8, hXZ, hAZ, hδZ, hξZ, -, -, hQZ⟩ := knZ_ge hr
  obtain ⟨hQ, hA, hX, -, -, hδ, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hδ', hδ'1, hiδ'⟩ := knδ'_bounds hr
  obtain ⟨hε, -, hiε⟩ := knε_bounds hr
  obtain ⟨hΛ, -⟩ := knΛ_bounds hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hZ1 : 1 ≤ knZ r := by linarith
  have hl0 : 0 ≤ log (4 * (r.Q + 1) * (knSC r + 1) / knδ' r) := by
    refine Real.log_nonneg ?_
    rw [le_div_iff₀ hδ']
    have : (1 : ℝ) ≤ (r.Q + 1) * (knSC r + 1) :=
      one_le_mul_of_one_le_of_one_le (by linarith) (by linarith)
    nlinarith
  have hl : log (4 * (r.Q + 1) * (knSC r + 1) / knδ' r) ≤ knΛ r := by
    refine knob_log_le hr (by positivity) (knob_pow_le hr (c := 2 ^ 88) (n := 19) (j := 1)
      (k := 30) (by norm_num) ?_ (by norm_num) (by norm_num))
    have hS1 : knSC r + 1 ≤ 2 ^ 81 * knZ r ^ 16 * knΛ r := by
      have a1 : 2 ^ 6 * (r.Q : ℝ) * knRBb r ≤ 2 ^ 6 * knZ r * (2 ^ 74 * knZ r ^ 15 * knΛ r) := by
        gcongr
      have a2 : (1 : ℝ) ≤ knZ r ^ 16 * knΛ r :=
        one_le_mul_of_one_le_of_one_le (one_le_pow₀ hZ1) (by linarith)
      nlinarith
    calc 4 * ((r.Q : ℝ) + 1) * (knSC r + 1) / knδ' r
        = 4 * (r.Q + 1) * (knSC r + 1) * (1 / knδ' r) := by ring
      _ ≤ 4 * (2 * knZ r) * (2 ^ 81 * knZ r ^ 16 * knΛ r) * (16 * knZ r * knZ r) := by
          gcongr
          · linarith
          · exact hiδ'.trans (by gcongr)
      _ = 2 ^ 88 * knZ r ^ 19 * knΛ r ^ 1 := by ring
  have e : knHR r = 384 * log (4 * (r.Q + 1) * (knSC r + 1) / knδ' r) * (1 / knε r) + 1 := by
    unfold knHR; field_simp
  rw [e]
  have hpos : 0 ≤ 384 * log (4 * (r.Q + 1) * (knSC r + 1) / knδ' r) * (1 / knε r) := by
    positivity
  refine ⟨by linarith, ?_⟩
  have h1 := mul_le_mul hl hiε (by positivity) (by linarith)
  have h2 : (1 : ℝ) ≤ r.A * knX r * (1 / r.ξc) * knΛ r := by
    have a1 : (1 : ℝ) ≤ r.A * knX r := one_le_mul_of_one_le_of_one_le (by linarith) hX
    have a2 := one_le_mul_of_one_le_of_one_le a1 iξ
    exact one_le_mul_of_one_le_of_one_le a2 (by linarith)
  nlinarith

end draws

/-! The premises that pin `p₀`, and level 3's slope and slack. -/

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

/-- Level 3's slope premise at any walk start up to `L`. -/
lemma knSlope_spec (hr : r ∈ rlStarRunRegion) {k : ℕ} (hk : k ≤ r.L) :
    2 * ((r.Q : ℝ) + 1) * (288 / knε r + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + k) : ℝ)))
        * knTu r + r.Q * (3 * knι r) / knTu r < 7 / 8 := by
  obtain ⟨hQ, -⟩ := kn_base hr
  obtain ⟨hε, -⟩ := knε_bounds hr
  obtain ⟨hslo, -⟩ := knSlope_bounds hr
  obtain ⟨hι, -, hι48, -⟩ := knι_bounds hr
  have hsl0 : 0 < knSlope r := lt_of_lt_of_le (by
    obtain ⟨-, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr; positivity) hslo
  have hc : ((Nat.clog 2 (r.L + r.Q + 3 + k) : ℕ) : ℝ) ≤ Nat.clog 2 (r.L + r.Q + 3 + r.L) := by
    exact_mod_cast Nat.clog_mono_right 2 (by omega)
  have h1 : 2 * ((r.Q : ℝ) + 1) * (288 / knε r + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + k) : ℝ)))
      ≤ knSlope r := by
    unfold knSlope; gcongr
  have hTu : knTu r = 1 / (4 * knSlope r) := rfl
  have h2 : 2 * ((r.Q : ℝ) + 1) * (288 / knε r + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + k) : ℝ)))
      * knTu r ≤ 1 / 4 := by
    rw [hTu]
    calc 2 * ((r.Q : ℝ) + 1) * (288 / knε r + 16 * (2 + (Nat.clog 2 (r.L + r.Q + 3 + k) : ℝ)))
          * (1 / (4 * knSlope r)) ≤ knSlope r * (1 / (4 * knSlope r)) := by gcongr
      _ = 1 / 4 := by field_simp
  have h3 : (r.Q : ℝ) * (3 * knι r) / knTu r ≤ 1 / 4 := by
    rw [hTu, div_div_eq_mul_div, div_one]
    linarith
  linarith

/-- The log of the slack's count `(q + 1) Kt e (e + 1)`. -/
lemma log_kt_le {q a e K1 K2 : ℝ} {nq nm : ℕ} (hq : 1 ≤ q) (ha : 2 ≤ a) (he : 1 ≤ e)
    (hK1 : K1 = 1 + (q + 1) ^ 3 * (a + 1)) (hK2 : K2 = q + 3) (hnq : (nq : ℝ) = q)
    (hnm : (nm : ℝ) = (q + 2) * (a + 1)) :
    log ((q + 1) * (K1 ^ nq * K2 ^ nm) * (e * (e + 1))) ≤ 32 * q ^ 4 * a + 2 * e := by
  subst hK1 hK2
  have hK10 : 0 < 1 + (q + 1) ^ 3 * (a + 1) := by positivity
  have hK20 : 0 < q + 3 := by positivity
  rw [Real.log_mul (x := (q + 1) * ((1 + (q + 1) ^ 3 * (a + 1)) ^ nq * (q + 3) ^ nm))
      (y := e * (e + 1)) (by positivity) (by positivity),
    Real.log_mul (x := q + 1) (y := (1 + (q + 1) ^ 3 * (a + 1)) ^ nq * (q + 3) ^ nm)
      (by positivity) (by positivity),
    Real.log_mul (x := (1 + (q + 1) ^ 3 * (a + 1)) ^ nq) (y := (q + 3) ^ nm) (by positivity)
      (by positivity),
    Real.log_mul (x := e) (y := e + 1) (by positivity) (by positivity), Real.log_pow,
    Real.log_pow, hnq, hnm]
  have l1 := Real.log_le_sub_one_of_pos (show 0 < q + 1 by linarith)
  have l2 := Real.log_le_sub_one_of_pos hK10
  have l3 := Real.log_le_sub_one_of_pos hK20
  have l4 := Real.log_le_sub_one_of_pos (show 0 < e by linarith)
  have l5 := Real.log_le_sub_one_of_pos (show 0 < e + 1 by linarith)
  have t1 := mul_le_mul_of_nonneg_left l2 (show 0 ≤ q by linarith)
  have t2 := mul_le_mul_of_nonneg_left l3 (show 0 ≤ (q + 2) * (a + 1) by positivity)
  have p1 : (q + 1) ^ 3 ≤ 8 * q ^ 3 := by
    have := pow_le_pow_left₀ (by linarith) (show q + 1 ≤ 2 * q by linarith) 3
    linarith [show (2 * q) ^ 3 = 8 * q ^ 3 by ring]
  have p2 : q * ((q + 1) ^ 3 * (a + 1)) ≤ 12 * q ^ 4 * a := by
    have h1 := mul_le_mul p1 (show a + 1 ≤ 3 / 2 * a by linarith) (by positivity) (by positivity)
    have h2 := mul_le_mul_of_nonneg_left h1 (show 0 ≤ q by linarith)
    linarith [show q * (8 * q ^ 3 * (3 / 2 * a)) = 12 * q ^ 4 * a by ring]
  have p3 : (q + 2) * (a + 1) * (q + 3 - 1) ≤ 14 * q ^ 4 * a := by
    have h1 : (q + 2) * (q + 2) ≤ 9 * q ^ 2 := by nlinarith
    have h2 : q ^ 2 ≤ q ^ 4 := pow_le_pow_right₀ hq (by norm_num)
    have h3 := mul_le_mul (show (q + 2) * (q + 2) ≤ 9 * q ^ 4 by linarith)
      (show a + 1 ≤ 3 / 2 * a by linarith) (by positivity) (by positivity)
    have : 0 ≤ q ^ 4 * a := by positivity
    linarith [show (q + 2) * (a + 1) * (q + 3 - 1) = (q + 2) * (q + 2) * (a + 1) by ring]
  have p4 : q ≤ q ^ 4 * a := by
    have h1 : q ≤ q ^ 4 := le_self_pow₀ hq (by norm_num)
    have h2 : q ^ 4 ≤ q ^ 4 * a := le_mul_of_one_le_right (by positivity) (by linarith)
    linarith
  linarith

/-- `ι² log H ≤ 2⁻¹⁶` once `2¹⁵ Q² A X ξi ι ≤ 1` and `log H ≤ 2¹⁴ Q⁴ A X² ξi`. -/
lemma iota_sq_log_le {ι lH Q A X ξi : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hX : 1 ≤ X) (hξi : 1 ≤ ξi)
    (hι : 0 ≤ ι) (hm : 2 ^ 15 * Q ^ 2 * A * X * ξi * ι ≤ 1)
    (hlH : lH ≤ 2 ^ 14 * Q ^ 4 * A * X ^ 2 * ξi) : ι ^ 2 * lH ≤ 1 / 2 ^ 16 := by
  have hAξ : 2 ≤ A * ξi := by nlinarith
  have ht : 0 ≤ Q ^ 4 * A * X ^ 2 * ξi := by positivity
  have e : (2 ^ 15 * Q ^ 2 * A * X * ξi) ^ 2 = 2 ^ 30 * (Q ^ 4 * A * X ^ 2 * ξi) * (A * ξi) := by
    ring
  have hm2 : 2 ^ 17 * (2 ^ 14 * Q ^ 4 * A * X ^ 2 * ξi) ≤ (2 ^ 15 * Q ^ 2 * A * X * ξi) ^ 2 := by
    have := mul_le_mul_of_nonneg_left hAξ ht
    rw [e]; nlinarith
  have h0 : 0 ≤ 2 ^ 15 * Q ^ 2 * A * X * ξi * ι := by positivity
  have h1 : (2 ^ 15 * Q ^ 2 * A * X * ξi * ι) ^ 2 ≤ 1 := by nlinarith
  have a := mul_le_mul_of_nonneg_left hm2 (sq_nonneg ι)
  have b : ι ^ 2 * (2 ^ 15 * Q ^ 2 * A * X * ξi) ^ 2 = (2 ^ 15 * Q ^ 2 * A * X * ξi * ι) ^ 2 := by
    ring
  have c := mul_le_mul_of_nonneg_left hlH (sq_nonneg ι)
  rw [le_div_iff₀ (by norm_num)]
  nlinarith

/-- `cap L ≤ 1/8` from `cap ≤ ι²/(2²⁸ ly)` and `L ≤ ly + lH`. -/
lemma slack_final {cap ι ly lH L : ℝ} (hcap0 : 0 ≤ cap) (hcap : cap * (2 ^ 28 * ly) ≤ ι ^ 2)
    (hly : 1 ≤ ly) (hlH : 0 ≤ lH) (hι : ι ^ 2 ≤ 1 / 6400) (hιH : ι ^ 2 * lH ≤ 1 / 2 ^ 16)
    (hL : L ≤ ly + lH) : cap * L ≤ 1 / 8 := by
  have h1 : cap * ly ≤ ι ^ 2 / 2 ^ 28 := by rw [le_div_iff₀ (by norm_num)]; linarith
  have h2 : cap * lH ≤ cap * ly * lH := by
    have := mul_le_mul_of_nonneg_right hly (mul_nonneg hcap0 hlH)
    linarith
  have h3 : cap * ly * lH ≤ ι ^ 2 / 2 ^ 28 * lH := mul_le_mul_of_nonneg_right h1 hlH
  have h4 : cap * L ≤ cap * (ly + lH) := mul_le_mul_of_nonneg_left hL hcap0
  have h5 : ι ^ 2 / 2 ^ 28 * lH = ι ^ 2 * lH / 2 ^ 28 := by ring
  linarith

lemma slack_arg_le {q Kt E d c : ℝ} (hd : 0 < d) (hc : 64 ≤ c)
    (hH : 0 ≤ (q + 1) * Kt * (E * (E + 1))) :
    8 * (q + 1) * Kt * (E * (E + 1)) / (d / 8) ≤ c / d * ((q + 1) * Kt * (E * (E + 1))) := by
  have e1 : 8 * (q + 1) * Kt * (E * (E + 1)) / (d / 8)
      = 64 * ((q + 1) * Kt * (E * (E + 1)) / d) := by
    field_simp; ring
  have e2 : c / d * ((q + 1) * Kt * (E * (E + 1))) = c * ((q + 1) * Kt * (E * (E + 1)) / d) := by
    ring
  rw [e1, e2]
  exact mul_le_mul_of_nonneg_right hc (div_nonneg hH hd.le)

/-- Level 3's slack at `2θ`, at most `1/16`. -/
lemma knSlack_spec (hr : r ∈ rlStarRunRegion) :
    knCapP r * log (8 * ((r.Q : ℝ) + 1) * ((1 + ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1)) ^ r.Q
        * ((r.Q : ℝ) + 3) ^ ((r.Q + 2) * (r.A + 1))) * ((knE r : ℝ) * (knE r + 1))
        / (knδ' r / 8)) ≤ 1 / 8 := by
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, hξ1, -⟩ := kn_base hr
  obtain ⟨hι0, -, -, -, hι80, hιm⟩ := knι_bounds hr
  obtain ⟨hE1, hE, -⟩ := knE_bounds hr
  obtain ⟨hδ', -⟩ := knδ'_bounds hr
  obtain ⟨hPB28, -⟩ := knPB_bounds hr
  obtain ⟨hRB0, -⟩ := knRB_bounds hr
  have hcp := knCapP_pos hr
  have iξ : 1 ≤ 1 / r.ξc := by rw [le_div_iff₀ hξ]; linarith
  have hlb0 : 0 ≤ Real.logb 2 (knRB r + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  have hg1 : (1 : ℝ) ≤ (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1)) :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith) (by linarith))
      (by linarith)
  have hy3 : 3 ≤ 128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r := by
    rw [le_div_iff₀ hδ']
    obtain ⟨-, hδ'1, -⟩ := knδ'_bounds hr
    nlinarith
  have hl7 := one_le_log_of_three_le hy3
  have hcap : knCapP r ≤ knι r ^ 2 / (2 ^ 28 * log (128 * (r.Q + 2) * (knPB r + 1)
      * (3 + 2 * Real.logb 2 (knRB r + 1)) / knδ' r)) :=
    (min_le_left _ _).trans ((min_le_left _ _).trans (min_le_right _ _))
  rw [le_div_iff₀ (by positivity)] at hcap
  have hK1 : (1 : ℝ) ≤ 1 + ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1) := by
    have : (0 : ℝ) ≤ ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1) := by positivity
    linarith
  have hH1 : (1 : ℝ) ≤ ((r.Q : ℝ) + 1) * ((1 + ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1)) ^ r.Q
      * ((r.Q : ℝ) + 3) ^ ((r.Q + 2) * (r.A + 1))) * ((knE r : ℝ) * (knE r + 1)) :=
    one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le (by linarith)
      (one_le_mul_of_one_le_of_one_le (one_le_pow₀ hK1) (one_le_pow₀ (by linarith))))
      (one_le_mul_of_one_le_of_one_le hE1 (by linarith))
  have harg := slack_arg_le (q := (r.Q : ℝ))
    (Kt := (1 + ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1)) ^ r.Q
    * ((r.Q : ℝ) + 3) ^ ((r.Q + 2) * (r.A + 1))) (E := (knE r : ℝ)) (d := knδ' r)
    (c := 128 * (r.Q + 2) * (knPB r + 1) * (3 + 2 * Real.logb 2 (knRB r + 1))) hδ'
    (by linarith) (by linarith)
  have hlog := (Real.log_le_log (by positivity) harg).trans_eq
    (Real.log_mul (by positivity) (by positivity))
  have hlH := log_kt_le (q := (r.Q : ℝ)) (a := (r.A : ℝ)) (e := (knE r : ℝ))
    (K1 := 1 + ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1)) (K2 := (r.Q : ℝ) + 3) (nq := r.Q)
    (nm := (r.Q + 2) * (r.A + 1)) hQ hA hE1 rfl rfl rfl (by push_cast; ring)
  have hlH' : log (((r.Q : ℝ) + 1) * ((1 + ((r.Q : ℝ) + 1) ^ 3 * ((r.A : ℝ) + 1)) ^ r.Q
      * ((r.Q : ℝ) + 3) ^ ((r.Q + 2) * (r.A + 1))) * ((knE r : ℝ) * (knE r + 1)))
      ≤ 2 ^ 14 * r.Q ^ 4 * r.A * knX r ^ 2 * (1 / r.ξc) := by
    have a1 : (1 : ℝ) ≤ knX r ^ 2 * (1 / r.ξc) := one_le_mul_of_one_le_of_one_le (one_le_pow₀ hX) iξ
    have a2 : (r.Q : ℝ) ^ 4 * r.A ≤ r.Q ^ 4 * r.A * (knX r ^ 2 * (1 / r.ξc)) :=
      le_mul_of_one_le_right (by positivity) a1
    have a3 : (r.A : ℝ) * knX r ^ 2 * (1 / r.ξc) ≤ r.Q ^ 4 * (r.A * knX r ^ 2 * (1 / r.ξc)) :=
      le_mul_of_one_le_left (by positivity) (one_le_pow₀ hQ)
    linarith
  have hιH := iota_sq_log_le hQ hA hX iξ hι0.le hιm hlH'
  have hι2 : knι r ^ 2 ≤ 1 / 6400 := by nlinarith
  exact slack_final hcp.le hcap hl7 (Real.log_nonneg hH1) hι2 hιH hlog

lemma knP₀_bounds (hr : r ∈ rlStarRunRegion) :
    0 < knP₀ r ∧ knP₀ r < 1 ∧ knP₀ r ≤ knCapP r * knε r / 12288 ∧ knP₀ r ≤ knMcD r := by
  obtain ⟨hε, hε12, -⟩ := knε_bounds hr
  obtain ⟨-, hξ1, -⟩ := ξc_mem hr
  have hcp := knCapP_pos hr
  have hcp1 := knCapP_le_one hr
  have hMc := knMcD_pos hr
  have hp1 : knP₀ r ≤ knCapP r * knε r / 12288 := min_le_left _ _
  have hpos : 0 < knP₀ r := lt_min (by positivity) hMc
  refine ⟨hpos, hp1.trans_lt ?_, hp1, min_le_right _ _⟩
  have : knCapP r * knε r ≤ 1 := mul_le_one₀ hcp1 hε.le (by linarith)
  linarith

end premises

/-! The remaining premises. -/

section rest

variable {r : RLStarRun}

lemma probeSifts_mono {m n : ℕ} (h : m ≤ n) :
    Random.Kappa.probeSifts m ≤ Random.Kappa.probeSifts n :=
  min_le_min (by omega) (by have := Nat.clog_mono_right 2 h; omega)

lemma knTu_pos (hr : r ∈ rlStarRunRegion) : 0 < knTu r := by
  obtain ⟨hslo, -⟩ := knSlope_bounds hr
  obtain ⟨hQ, hA, hX, -, -, -, -, hξ, -⟩ := kn_base hr
  have : 0 < knSlope r := lt_of_lt_of_le (by positivity) hslo
  unfold knTu; positivity

lemma knb_spec (hr : r ∈ rlStarRunRegion) :
    (knE r : ℝ) * knJ r * (1 - knε r / 192) ^ knB r ≤ knδ' r / 4 := by
  obtain ⟨hε, hε12, -⟩ := knε_bounds hr
  obtain ⟨hξ, hξ1, -⟩ := ξc_mem hr
  obtain ⟨-, -, hb⟩ := knB_bounds hr
  obtain ⟨hE1, -, -⟩ := knE_bounds hr
  obtain ⟨hJ1, -, -⟩ := knJ_bounds hr
  obtain ⟨hδ', -, -⟩ := knδ'_bounds hr
  obtain ⟨-, hL9⟩ := knL3 hr
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

lemma knNr_spec (hr : r ∈ rlStarRunRegion) {k : ℕ} (hk : k ≤ knK r) :
    (knE r : ℝ) * ((knB₃ r + knJ r * knB r)
        * ((Random.Kappa.probeSifts (r.L + r.Q + 3 + k) : ℕ) * ((r.Q : ℝ) + 3))
      + (r.Q + 2) * ((r.Q + 2) * (r.A + 1) * (r.Q + 2))) ≤ knNr r := by
  obtain ⟨hB0, -⟩ := knB₃_bounds hr
  have hp : Random.Kappa.probeSifts (r.L + r.Q + 3 + k) ≤ knRp r := probeSifts_mono (by omega)
  have hp' : ((Random.Kappa.probeSifts (r.L + r.Q + 3 + k) : ℕ) : ℝ) ≤ knRp r := by
    exact_mod_cast hp
  refine le_trans ?_ (Nat.le_ceil _)
  gcongr

lemma knTw_spec (hr : r ∈ rlStarRunRegion) : 0 < knTw r ∧ knTw r ≤ knε r / 12288 := by
  obtain ⟨hε, -⟩ := knε_bounds hr
  exact ⟨by unfold knTw; positivity, le_rfl⟩

lemma knξT_spec (hr : r ∈ rlStarRunRegion) :
    knε r + knε r / 96 + knTw r + (1025 / 1024 + 3 / 8) * knεcov r + r.ξc / 4 ≤ r.ξ := by
  obtain ⟨hε, hε12, -⟩ := knε_bounds hr
  obtain ⟨hεc, hεce, -⟩ := knεcov_bounds hr
  obtain ⟨hN1, -⟩ := knN_bounds hr
  obtain ⟨hξ, hξ1, hξξ, -⟩ := ξc_mem hr
  have : knεcov r ≤ r.ξc / 10 := by
    rw [hεce, div_le_div_iff₀ (by positivity) (by norm_num)]
    linarith [mul_le_mul_of_nonneg_left hN1 hξ.le]
  unfold knTw
  linarith

end rest

end OrthoDFA

end
