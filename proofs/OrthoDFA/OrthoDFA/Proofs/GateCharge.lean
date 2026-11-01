import OrthoDFA.LowLevelClaims.Clustering

/-!
# What a side's cap charges a clean family

`wrongCap` reads a side's spread over `(1 − η₀ − c)²`.  On a side of `n` prefixes, `k` of them
cut wrongly, whose counts sit near their means, it is at most `k/gateKappa η₀` plus the
deviations at `gateDev η₀`.  With `a = (1 − 2η₀)/(1 − η₀)` and `κ = k/n`, the cap without
deviations is `κ(1 − κ)/(a − κ)²` until it reaches `1`, which happens at the root `κ*` of

    2κ² − (1 + 2a)κ + a² = 0,

so its ratio to `κ` is largest there.  `gateKappa` is a root bound below `κ*`.
-/

namespace OrthoDFA

/-- The spread identity behind `spreadBound`: a side of `n` prefixes, `k` of them wrong, its
right ones at rate `x` and its wrong ones at `y`. -/
lemma spread_identity (n k : ℝ) (hn : 0 < n) (x y c : ℝ) :
    (n - k) / n * (x - c) ^ 2 + k / n * (1 - y - c) ^ 2
      = c * (1 - c) + (1 - 2 * c) * (((n - k) * x + k * (1 - y)) / n - c)
        - ((n - k) * (x * (1 - x)) + k * (y * (1 - y))) / n := by
  field_simp
  ring

lemma spread_at_le (n k ag ap : ℕ) (hk : k ≤ n) (hn : 0 < n) {x y d t tu : ℝ} (hd : 0 ≤ d)
    (hlo : ((n : ℝ) - k) * x + k * (1 - y) - t ≤ ag)
    (hhi : (ag : ℝ) ≤ ((n : ℝ) - k) * x + k * (1 - y) + t)
    (hb : 2 * (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) - tu ≤ ap)
    (hc0 : 0 ≤ (ag : ℝ) / n) (hc1 : (ag : ℝ) / n ≤ 1) :
    spreadBound n ag ap d ((ag : ℝ) / n)
      ≤ (n - k) / n * (x - ag / n) ^ 2 + k / n * (1 - y - ag / n) ^ 2
        + (t + 3 / 2 * d + tu / 2) / n := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  set c : ℝ := (ag : ℝ) / n with hc
  set μ : ℝ := (((n : ℝ) - k) * x + k * (1 - y)) / n with hμ
  have hid := spread_identity (n : ℝ) k hnR x y c
  rw [← hμ] at hid
  have hcμ : |μ - c| ≤ t / n := by
    rw [hμ, hc, ← sub_div, abs_div, abs_of_pos hnR]
    exact div_le_div_of_nonneg_right (abs_le.2 ⟨by linarith, by linarith⟩) hnR.le
  have h12 : |1 - 2 * c| ≤ 1 := by rw [abs_le]; constructor <;> linarith
  have hmax : max ((1 - 2 * c) * (((ag : ℝ) - d) / n - c)) ((1 - 2 * c) * (((ag : ℝ) + d) / n - c))
      ≤ d / n := by
    have e1 : ((ag : ℝ) - d) / n - c = -(d / n) := by rw [hc]; field_simp; ring
    have e2 : ((ag : ℝ) + d) / n - c = d / n := by rw [hc]; field_simp; ring
    rw [e1, e2]
    have hdn : 0 ≤ d / n := div_nonneg hd hnR.le
    refine max_le ?_ ?_ <;> nlinarith [abs_le.1 h12]
  have hmid : -((1 - 2 * c) * (μ - c)) ≤ t / n := by
    have := abs_mul (1 - 2 * c) (μ - c)
    have h2 : |(1 - 2 * c) * (μ - c)| ≤ 1 * (t / n) :=
      by rw [this]; exact mul_le_mul h12 hcμ (abs_nonneg _) (by norm_num)
    have := neg_abs_le ((1 - 2 * c) * (μ - c))
    linarith
  have hfl : ((ap : ℝ) - d) / (2 * n)
      ≥ (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) / n - (tu + d) / (2 * n) := by
    have e : ((ap : ℝ) - d) / (2 * n)
        - ((((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) / n - (tu + d) / (2 * n))
        = ((ap : ℝ) - 2 * (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) + tu) / (2 * n) := by
      field_simp; ring
    have : 0 ≤ ((ap : ℝ) - 2 * (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) + tu)
        / (2 * n) := div_nonneg (by linarith) (by positivity)
    linarith
  unfold spreadBound
  have : (t + 3 / 2 * d + tu / 2) / n = t / n + d / n + (tu + d) / (2 * n) := by
    field_simp; ring
  rw [this]
  linarith

lemma wrongCap_le_one (n ag ap : ℕ) (d y : ℝ) : wrongCap n ag ap d y ≤ 1 := by
  unfold wrongCap; split_ifs
  · exact min_le_left _ _
  · exact le_rfl

lemma wrongCap_nonneg (n ag ap : ℕ) (d y : ℝ) : 0 ≤ wrongCap n ag ap d y := by
  unfold wrongCap; split_ifs
  · exact le_min (by norm_num) (div_nonneg (le_max_left _ _) (sq_nonneg _))
  · norm_num

/-- `a = (1 − 2η₀)/(1 − η₀)`, the least ratio of a right prefix's distance from the far rate
`1 − η₀` to the gap between the classes. -/
noncomputable def gateA (η₀ : ℝ) : ℝ := (1 - 2 * η₀) / (1 - η₀)

/-- A lower bound on the root `κ*`:

    gateKappa = a²(1 + 2a) / ((1 + 2a)² − 2a²) -/
noncomputable def gateKappa (η₀ : ℝ) : ℝ :=
  gateA η₀ ^ 2 * (1 + 2 * gateA η₀) / ((1 + 2 * gateA η₀) ^ 2 - 2 * gateA η₀ ^ 2)

/-- `H w`, with `H = 1 − η₀` and `w = a − gateKappa`: the least distance, near the cap's
saturation, of a side's mean from the far rate. -/
noncomputable def gateGap (η₀ : ℝ) : ℝ := (1 - η₀) * (gateA η₀ - gateKappa η₀)

/-- What the seed's and the references' deviations cost, per unit:

    gateDevT = 16/(5 H w) + 9/(4 (H w)²),   gateDevD = 27/(10 (H w)²),   gateDevU = 9/(10 (H w)²) -/
noncomputable def gateDevT (η₀ : ℝ) : ℝ := 16 / (5 * gateGap η₀) + 9 / (4 * gateGap η₀ ^ 2)

noncomputable def gateDevD (η₀ : ℝ) : ℝ := 27 / (10 * gateGap η₀ ^ 2)

noncomputable def gateDevU (η₀ : ℝ) : ℝ := 9 / (10 * gateGap η₀ ^ 2)

lemma gateA_pos {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) : 0 < gateA η₀ := by
  unfold gateA; exact div_pos (by linarith) (by linarith)

lemma gateA_le_one {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) : gateA η₀ ≤ 1 := by
  unfold gateA; rw [div_le_one (by linarith)]; linarith

lemma le_gateA {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) : 1 - 2 * η₀ ≤ gateA η₀ := by
  unfold gateA; rw [le_div_iff₀ (by linarith)]; nlinarith

/-- The facts about `κ = gateKappa` the cap's bound uses: it is positive, `3κ ≤ 2a`, and the
quadratic is nonnegative there, `(a − κ)² ≥ κ(1 − κ)`. -/
lemma gateKappa_facts {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) :
    0 < gateKappa η₀ ∧ 3 * gateKappa η₀ ≤ 2 * gateA η₀
      ∧ gateKappa η₀ * (1 - gateKappa η₀) ≤ (gateA η₀ - gateKappa η₀) ^ 2 := by
  have ha := gateA_pos h0 h1
  have ha1 := gateA_le_one h0 h1
  set a := gateA η₀
  have hQ : 0 < (1 + 2 * a) ^ 2 - 2 * a ^ 2 := by nlinarith
  have hk : gateKappa η₀ = a ^ 2 * (1 + 2 * a) / ((1 + 2 * a) ^ 2 - 2 * a ^ 2) := rfl
  refine ⟨?_, ?_, ?_⟩
  · rw [hk]; positivity
  · rw [hk, show 3 * (a ^ 2 * (1 + 2 * a) / ((1 + 2 * a) ^ 2 - 2 * a ^ 2))
        = 3 * a ^ 2 * (1 + 2 * a) / ((1 + 2 * a) ^ 2 - 2 * a ^ 2) by ring, div_le_iff₀ hQ]
    nlinarith
  · rw [hk]
    set Q := (1 + 2 * a) ^ 2 - 2 * a ^ 2
    have e : (a - a ^ 2 * (1 + 2 * a) / Q) ^ 2
        - a ^ 2 * (1 + 2 * a) / Q * (1 - a ^ 2 * (1 + 2 * a) / Q) = 4 * a ^ 6 / Q ^ 2 := by
      field_simp
      ring
    have : 0 ≤ 4 * a ^ 6 / Q ^ 2 := by positivity
    linarith

lemma gateKappa_pos {η₀ : ℝ} (h1 : η₀ < 1 / 2) : 0 < gateKappa η₀ := by
  have ha : 0 < gateA η₀ := by unfold gateA; exact div_pos (by linarith) (by linarith)
  unfold gateKappa
  have : 0 < (1 + 2 * gateA η₀) ^ 2 - 2 * gateA η₀ ^ 2 := by nlinarith
  positivity

lemma gateKappa_ge {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) :
    3 * gateA η₀ ^ 2 / 7 ≤ gateKappa η₀ := by
  have ha := gateA_pos h0 h1
  have ha1 := gateA_le_one h0 h1
  have hQ : 0 < (1 + 2 * gateA η₀) ^ 2 - 2 * gateA η₀ ^ 2 := by nlinarith
  unfold gateKappa
  rw [div_le_div_iff₀ (by norm_num) hQ]
  nlinarith [mul_nonneg (sq_nonneg (gateA η₀)) (mul_nonneg (by linarith : 0 ≤ 1 - gateA η₀)
    (by linarith : 0 ≤ 6 * gateA η₀ + 4))]

lemma gateWidth_pos {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) :
    gateA η₀ / 3 ≤ gateA η₀ - gateKappa η₀ := by
  obtain ⟨-, h3, -⟩ := gateKappa_facts h0 h1
  linarith

lemma gateGap_pos {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) : 0 < gateGap η₀ := by
  have hw := gateWidth_pos h0 h1
  have ha := gateA_pos h0 h1
  unfold gateGap
  exact mul_pos (by linarith) (by linarith)

lemma gateGap_ge {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) :
    4 * (1 - 2 * η₀) / 7 ≤ gateGap η₀ := by
  have ha := gateA_pos h0 h1
  have ha1 := gateA_le_one h0 h1
  have hQ : 0 < (1 + 2 * gateA η₀) ^ 2 - 2 * gateA η₀ ^ 2 := by nlinarith
  have hk : gateKappa η₀ ≤ 3 * gateA η₀ / 7 := by
    unfold gateKappa
    rw [div_le_div_iff₀ hQ (by norm_num)]
    nlinarith [mul_nonneg ha.le (mul_nonneg (by linarith : 0 ≤ 1 - gateA η₀)
      (by linarith : 0 ≤ 8 * gateA η₀ + 3))]
  have e : (1 - η₀) * gateA η₀ = 1 - 2 * η₀ := by
    have hne : (1 - η₀) ≠ 0 := by intro h; linarith
    unfold gateA; field_simp
  unfold gateGap
  nlinarith

lemma gateDev_pos {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) :
    0 < gateDevT η₀ ∧ 0 < gateDevD η₀ ∧ 0 < gateDevU η₀ := by
  have := gateGap_pos h0 h1
  refine ⟨?_, ?_, ?_⟩
  · unfold gateDevT; positivity
  · unfold gateDevD; positivity
  · unfold gateDevU; positivity

lemma inv_one_sub_sq_le {e : ℝ} (h0 : 0 ≤ e) (h4 : e ≤ 1 / 4) :
    1 / (1 - e) ^ 2 ≤ 1 + 16 / 5 * e := by
  rw [div_le_iff₀ (by nlinarith)]
  have hq : 0 ≤ (1 / 4 - e) * (3 / 2 - e) := mul_nonneg (by linarith) (by linarith)
  nlinarith [mul_nonneg h0 hq, mul_nonneg h0 h0]

set_option maxHeartbeats 4000000 in
/-- A side's cap against its wrong count: with the side's counts within `t` and `tu` of their
means, it charges each wrong prefix `1/gateKappa`, and the deviations `gateDev`. -/
theorem mul_wrongCap_le (n k ag ap : ℕ) (hk : k ≤ n) {x y η₀ d t tu : ℝ}
    (hd : 0 ≤ d) (ht : 0 ≤ t) (htu : 0 ≤ tu) (hx0 : 0 ≤ x) (hxη : x ≤ η₀) (hy0 : 0 ≤ y)
    (hyη : y ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hlo : ((n : ℝ) - k) * x + k * (1 - y) - t ≤ ag)
    (hhi : (ag : ℝ) ≤ ((n : ℝ) - k) * x + k * (1 - y) + t)
    (hb : 2 * (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) - tu ≤ ap) :
    n * wrongCap n ag ap d η₀
      ≤ k / gateKappa η₀ + gateDevT η₀ * t + gateDevD η₀ * d + gateDevU η₀ * tu := by
  have h0 : 0 ≤ η₀ := le_trans hx0 hxη
  obtain ⟨hκ0, hκ3, hq⟩ := gateKappa_facts h0 hη₀
  have hw := gateWidth_pos h0 hη₀
  have ha := gateA_pos h0 hη₀
  have ha1 := gateA_le_one h0 hη₀
  have haH : gateA η₀ * (1 - η₀) = 1 - 2 * η₀ := by
    unfold gateA; exact div_mul_cancel₀ _ (by linarith)
  set a := gateA η₀ with hadef
  set κl := gateKappa η₀ with hκdef
  set H := 1 - η₀ with hHdef
  set w := a - κl with hwdef
  have hH0 : 0 < H := by rw [hHdef]; linarith
  have hH1 : H ≤ 1 := by rw [hHdef]; linarith
  have hw0 : 0 < w := by linarith
  have hw1 : w ≤ 1 := by linarith
  have hHw : 0 < H * w := mul_pos hH0 hw0
  have hHw1 : H * w ≤ 1 := by nlinarith
  have hcap1 := wrongCap_le_one n ag ap d η₀
  have hcap0 := wrongCap_nonneg n ag ap d η₀
  obtain ⟨hT0, hD0, hU0⟩ := gateDev_pos h0 hη₀
  have hdev0 : 0 ≤ gateDevD η₀ * d + gateDevU η₀ * tu := by positivity
  have hdevT0 : 0 ≤ gateDevT η₀ * t := by positivity
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hkdiv0 : 0 ≤ (k : ℝ) / κl := div_nonneg hk0 hκ0.le
  rcases Nat.eq_zero_or_pos n with hn0 | hn
  · subst hn0; simp only [Nat.cast_zero, zero_mul]; linarith
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hgg : gateGap η₀ = H * w := rfl
  have hdevC : 4 * t / (H * w) ≤ gateDevT η₀ * t := by
    unfold gateDevT
    rw [hgg]
    have h1 : 4 / (H * w) ≤ 16 / (5 * (H * w)) + 9 / (4 * (H * w) ^ 2) := by
      have e1 : 16 / (5 * (H * w)) = 16 / 5 / (H * w) := by ring
      have e2 : 9 / (4 * (H * w) ^ 2) = 9 / 4 / (H * w) / (H * w) := by ring
      have h3 : 1 ≤ 1 / (H * w) := by rw [le_div_iff₀ hHw]; linarith
      have h4 : 9 / 4 / (H * w) ≤ 9 / 4 / (H * w) / (H * w) := by
        rw [le_div_iff₀ hHw]
        have : 0 ≤ 9 / 4 / (H * w) := by positivity
        have h7 := mul_le_mul_of_nonneg_left hHw1 this
        linarith only [h7]
      have h5 : 4 / (H * w) = 16 / 5 / (H * w) + 4 / 5 / (H * w) := by ring
      have h6 : 4 / 5 / (H * w) ≤ 9 / 4 / (H * w) :=
        div_le_div_of_nonneg_right (by norm_num) hHw.le
      linarith
    rw [show 4 * t / (H * w) = 4 / (H * w) * t by ring]
    exact mul_le_mul_of_nonneg_right h1 ht
  -- a side mostly wrong is charged by its size
  by_cases hA : κl * n ≤ k
  · have : (n : ℝ) ≤ k / κl := by rw [le_div_iff₀ hκ0]; linarith
    have h1 : (n : ℝ) * wrongCap n ag ap d η₀ ≤ n * 1 :=
      mul_le_mul_of_nonneg_left hcap1 hnR.le
    linarith only [h1, this, hdevT0, hdev0]
  push Not at hA
  -- a side too small for its deviations
  by_cases hC : n * (H * w) < 4 * t
  · have : (n : ℝ) ≤ 4 * t / (H * w) := by rw [le_div_iff₀ hHw]; linarith
    have h1 : (n : ℝ) * wrongCap n ag ap d η₀ ≤ n * 1 :=
      mul_le_mul_of_nonneg_left hcap1 hnR.le
    linarith only [h1, this, hdevC, hkdiv0, hdev0]
  push Not at hC
  set κ : ℝ := k / n with hκ
  set Δ : ℝ := 1 - x - y with hΔ
  set ε : ℝ := t / n with hε
  set c : ℝ := (ag : ℝ) / n with hc
  have hκn : κ * n = k := by rw [hκ]; field_simp
  have hεn : ε * n = t := by rw [hε]; field_simp
  have hκ0' : 0 ≤ κ := div_nonneg hk0 hnR.le
  have hκl : κ < κl := by rw [hκ, div_lt_iff₀ hnR]; linarith
  have hκ1 : κ ≤ 1 := by linarith
  have hε0 : 0 ≤ ε := div_nonneg ht hnR.le
  have hεHw : ε ≤ H * w / 4 := by rw [hε, div_le_iff₀ hnR]; linarith
  have hΔ1 : Δ ≤ 1 := by rw [hΔ]; linarith
  have hΔ0 : 0 < Δ := by rw [hΔ]; linarith
  -- the side's mean sits within `ε` of `x + κΔ`
  have hchi : c ≤ x + κ * Δ + ε := by
    rw [hc, hκ, hε, hΔ, div_le_iff₀ hnR]
    have : (x + k / n * (1 - x - y) + t / n) * n = ((n : ℝ) - k) * x + k * (1 - y) + t := by
      field_simp; ring
    linarith
  have hclo : x + κ * Δ - ε ≤ c := by
    rw [hc, hκ, hε, hΔ, le_div_iff₀ hnR]
    have : (x + k / n * (1 - x - y) - t / n) * n = ((n : ℝ) - k) * x + k * (1 - y) - t := by
      field_simp; ring
    linarith
  set z : ℝ := a - κ with hz
  have hzw : w ≤ z := by rw [hz, hwdef]; linarith
  have hz0 : 0 < z := by linarith
  -- the clean denominator, at least `Δz` and at least `Hz`
  set D₁ : ℝ := 1 - η₀ - x - κ * Δ with hD₁
  have hD₁Δ : Δ * z ≤ D₁ := by
    have h1 : a * (1 - x) ≤ 1 - η₀ - x := by
      have e : a * (1 - x) = (1 - 2 * η₀) - a * x + η₀ * a := by
        have : a = (1 - 2 * η₀) + a * η₀ := by linarith
        linarith only [this]
      have : a * x ≥ x * (1 - 2 * η₀) := by
        have := mul_le_mul_of_nonneg_left (le_gateA h0 hη₀) hx0
        linarith only [this]
      have := mul_nonneg (sub_nonneg.2 hxη) (sub_nonneg.2 ha1)
      linarith only [e, this]
    have h2 : a * Δ ≤ a * (1 - x) := mul_le_mul_of_nonneg_left (by rw [hΔ]; linarith) ha.le
    rw [hD₁, hz]
    linarith only [h1, h2]
  have hD₁H : H * z ≤ D₁ := by
    have e : D₁ - H * z = (η₀ - x) * (1 - κ) + κ * y := by
      rw [hD₁, hz, hHdef, hΔ]
      rw [hHdef] at haH
      linear_combination (-1 : ℝ) * haH
    have : 0 ≤ (η₀ - x) * (1 - κ) + κ * y := by
      have := mul_nonneg (show 0 ≤ η₀ - x by linarith) (show 0 ≤ 1 - κ by linarith)
      have := mul_nonneg hκ0' hy0
      linarith
    linarith
  have hHzw : H * w ≤ H * z := mul_le_mul_of_nonneg_left hzw hH0.le
  have hD₁0 : 0 < D₁ := by linarith
  set η' : ℝ := ε / D₁ with hη'
  have hη'0 : 0 ≤ η' := div_nonneg hε0 hD₁0.le
  have hη'4 : η' ≤ 1 / 4 := by
    rw [hη', div_le_iff₀ hD₁0]; linarith
  have hden : D₁ * (1 - η') ≤ 1 - η₀ - c := by
    have e : D₁ * (1 - η') = D₁ - ε := by rw [hη']; field_simp
    rw [e, hD₁]; linarith
  have hden0 : 0 < D₁ * (1 - η') := mul_pos hD₁0 (by linarith)
  have hdenpos : 0 < 1 - η₀ - c := lt_of_lt_of_le hden0 hden
  -- the spread
  have hc0 : 0 ≤ c := by rw [hc]; exact div_nonneg (Nat.cast_nonneg _) hnR.le
  have hc1 : c ≤ 1 := by linarith
  have hsp := spread_at_le n k ag ap hk hn hd hlo hhi hb hc0 hc1
  rw [← hc] at hsp
  have hsplit : ((n : ℝ) - k) / n * (x - c) ^ 2 + k / n * (1 - y - c) ^ 2
      = (c - x - κ * Δ) ^ 2 + κ * (1 - κ) * Δ ^ 2 := by
    rw [hκ, hΔ]; field_simp; ring
  have hsq : (c - x - κ * Δ) ^ 2 ≤ ε ^ 2 := by
    have : |c - x - κ * Δ| ≤ ε := abs_le.2 ⟨by linarith, by linarith⟩
    calc (c - x - κ * Δ) ^ 2 = |c - x - κ * Δ| ^ 2 := (sq_abs _).symm
      _ ≤ ε ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
  set Dv : ℝ := (t + 3 / 2 * d + tu / 2) / n with hDv
  have hDv0 : 0 ≤ Dv := by rw [hDv]; positivity
  have hSp : spreadBound n ag ap d c ≤ κ * (1 - κ) * Δ ^ 2 + (ε ^ 2 + Dv) := by
    rw [hsplit] at hsp; linarith
  have hA0' : 0 ≤ κ * (1 - κ) * Δ ^ 2 := by
    have : 0 ≤ 1 - κ := by linarith
    positivity
  have hnum0 : 0 ≤ κ * (1 - κ) * Δ ^ 2 + (ε ^ 2 + Dv) := by positivity
  have hcapS : wrongCap n ag ap d η₀
      ≤ (κ * (1 - κ) * Δ ^ 2 + (ε ^ 2 + Dv)) / (1 - η₀ - c) ^ 2 := by
    unfold wrongCap
    rw [if_pos ⟨hn, by rw [← hc]; exact hdenpos⟩, ← hc]
    refine le_trans (min_le_right _ _) ?_
    refine div_le_div_of_nonneg_right ?_ (sq_nonneg _)
    exact max_le hnum0 hSp
  have hcap2 : wrongCap n ag ap d η₀
      ≤ (κ * (1 - κ) * Δ ^ 2 + (ε ^ 2 + Dv)) / (D₁ * (1 - η')) ^ 2 :=
    le_trans hcapS (div_le_div_of_nonneg_left hnum0 (by positivity)
      (pow_le_pow_left₀ hden0.le hden 2))
  have hinv : 1 / (1 - η') ^ 2 ≤ 1 + 16 / 5 * η' := inv_one_sub_sq_le hη'0 hη'4
  -- the quadratic: `κ(1 − κ) ≤ (κ/κl)·z²`
  have hquad : κ * (1 - κ) ≤ κ / κl * z ^ 2 := by
    have hf : κl * (1 - κ) ≤ z ^ 2 := by
      have e : z ^ 2 - κl * (1 - κ)
          = (κl - κ) * (2 * a - κ - 2 * κl) + ((a - κl) ^ 2 - κl * (1 - κl)) := by
        rw [hz]; ring
      have h1 : 0 ≤ (κl - κ) * (2 * a - κ - 2 * κl) :=
        mul_nonneg (by linarith) (by linarith)
      have h2 : 0 ≤ (a - κl) ^ 2 - κl * (1 - κl) := by rw [← hwdef]; linarith
      linarith
    calc κ * (1 - κ) = κ / κl * (κl * (1 - κ)) := by field_simp
      _ ≤ κ / κl * z ^ 2 := mul_le_mul_of_nonneg_left hf (div_nonneg hκ0' hκ0.le)
  have hfin : wrongCap n ag ap d η₀
      ≤ κ / κl + 16 / 5 * η' + 9 / 5 * ((ε ^ 2 + Dv) / (H * w) ^ 2) := by
    have e : (κ * (1 - κ) * Δ ^ 2 + (ε ^ 2 + Dv)) / (D₁ * (1 - η')) ^ 2
        = (κ * (1 - κ) * Δ ^ 2 / D₁ ^ 2 + (ε ^ 2 + Dv) / D₁ ^ 2) * (1 / (1 - η') ^ 2) := by
      have : (1 - η') ≠ 0 := by linarith
      field_simp
    rw [e] at hcap2
    have hA1 : κ * (1 - κ) * Δ ^ 2 / D₁ ^ 2 ≤ κ / κl := by
      have h1 : κ * (1 - κ) * Δ ^ 2 / D₁ ^ 2 ≤ κ * (1 - κ) * Δ ^ 2 / (Δ * z) ^ 2 :=
        div_le_div_of_nonneg_left hA0' (by positivity) (pow_le_pow_left₀ (by positivity) hD₁Δ 2)
      have h2 : κ * (1 - κ) * Δ ^ 2 / (Δ * z) ^ 2 = κ * (1 - κ) / z ^ 2 := by
        field_simp
      have h3 : κ * (1 - κ) / z ^ 2 ≤ κ / κl := by
        rw [div_le_iff₀ (by positivity)]; exact hquad
      linarith
    have hκ1' : κ / κl ≤ 1 := by rw [div_le_one hκ0]; linarith
    have hB0 : 0 ≤ (ε ^ 2 + Dv) / D₁ ^ 2 := by positivity
    have hB1 : (ε ^ 2 + Dv) / D₁ ^ 2 ≤ (ε ^ 2 + Dv) / (H * w) ^ 2 :=
      div_le_div_of_nonneg_left (by positivity) (by positivity)
        (pow_le_pow_left₀ hHw.le (le_trans hHzw hD₁H) 2)
    have hA0 : 0 ≤ κ * (1 - κ) * Δ ^ 2 / D₁ ^ 2 := by positivity
    have hinv0 : 0 ≤ 1 / (1 - η') ^ 2 := by positivity
    calc wrongCap n ag ap d η₀
        ≤ (κ * (1 - κ) * Δ ^ 2 / D₁ ^ 2 + (ε ^ 2 + Dv) / D₁ ^ 2) * (1 / (1 - η') ^ 2) := hcap2
      _ ≤ (κ / κl + (ε ^ 2 + Dv) / (H * w) ^ 2) * (1 + 16 / 5 * η') :=
          mul_le_mul (by linarith) hinv hinv0 (by positivity)
      _ ≤ κ / κl + 16 / 5 * η' + 9 / 5 * ((ε ^ 2 + Dv) / (H * w) ^ 2) := by
          set A := κ / κl
          set B := (ε ^ 2 + Dv) / (H * w) ^ 2
          have hB2 : 0 ≤ B := le_trans hB0 hB1
          have e : (A + B) * (1 + 16 / 5 * η') = A + 16 / 5 * (A * η') + B + 16 / 5 * (B * η') := by
            ring
          have h1 : A * η' ≤ η' := mul_le_of_le_one_left hη'0 hκ1'
          have h2 : B * η' ≤ B / 4 := by
            have := mul_le_mul_of_nonneg_left hη'4 hB2
            linarith
          rw [e]; linarith
  -- scaled by the side's size
  have hη'n : n * η' ≤ t / (H * w) := by
    have e : (n : ℝ) * η' = t / D₁ := by rw [hη', ← hεn]; field_simp
    rw [e]
    exact div_le_div_of_nonneg_left ht hHw (le_trans hHzw hD₁H)
  have hDn : n * ((ε ^ 2 + Dv) / (H * w) ^ 2)
      ≤ (5 / 4 * t + 3 / 2 * d + tu / 2) / (H * w) ^ 2 := by
    rw [mul_div_assoc']
    refine div_le_div_of_nonneg_right ?_ (sq_nonneg _)
    have e1 : (n : ℝ) * Dv = t + 3 / 2 * d + tu / 2 := by rw [hDv]; field_simp
    have e2 : (n : ℝ) * ε ^ 2 = t * ε := by rw [← hεn]; ring
    have h3 : t * ε ≤ t / 4 := by
      have h4 := mul_le_mul_of_nonneg_left hεHw ht
      have h5 : t * (H * w / 4) ≤ t / 4 := by
        have := mul_le_mul_of_nonneg_left hHw1 ht
        linarith
      linarith
    rw [mul_add, e1, e2]
    linarith
  have hkey : (n : ℝ) * wrongCap n ag ap d η₀
      ≤ k / κl + 16 / 5 * (t / (H * w)) + 9 / 5 * ((5 / 4 * t + 3 / 2 * d + tu / 2) / (H * w) ^ 2)
      := by
    have h := mul_le_mul_of_nonneg_left hfin hnR.le
    have hk' : (n : ℝ) * (κ / κl) = k / κl := by rw [← hκn]; field_simp
    have e : (n : ℝ) * (κ / κl + 16 / 5 * η' + 9 / 5 * ((ε ^ 2 + Dv) / (H * w) ^ 2))
        = n * (κ / κl) + 16 / 5 * (n * η') + 9 / 5 * (n * ((ε ^ 2 + Dv) / (H * w) ^ 2)) := by
      ring
    rw [e, hk'] at h
    linarith
  refine le_trans hkey (le_of_eq ?_)
  unfold gateDevT gateDevD gateDevU
  rw [hgg]
  field_simp
  ring

lemma gateGap_le {η₀ : ℝ} (h0 : 0 ≤ η₀) (h1 : η₀ < 1 / 2) : gateGap η₀ ≤ 1 - 2 * η₀ := by
  have hκ := (gateKappa_facts h0 h1).1
  have e : (1 - η₀) * gateA η₀ = 1 - 2 * η₀ := by
    have hne : (1 - η₀) ≠ 0 := by intro h; linarith
    unfold gateA; field_simp
  unfold gateGap
  nlinarith

set_option maxHeartbeats 1000000 in
/-- `mul_wrongCap_le` at the side's own rates: a wrong prefix is charged
`4(1 − x − y)²/(1 − η₀ − x)²` rather than `1/gateKappa`, and the deviations twice. -/
theorem mul_wrongCap_le_rates (n k ag ap : ℕ) (hk : k ≤ n) {x y η₀ d t tu : ℝ}
    (hd : 0 ≤ d) (ht : 0 ≤ t) (htu : 0 ≤ tu) (hx0 : 0 ≤ x) (hxη : x ≤ η₀) (hy0 : 0 ≤ y)
    (hyη : y ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hlo : ((n : ℝ) - k) * x + k * (1 - y) - t ≤ ag)
    (hhi : (ag : ℝ) ≤ ((n : ℝ) - k) * x + k * (1 - y) + t)
    (hb : 2 * (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) - tu ≤ ap) :
    n * wrongCap n ag ap d η₀
      ≤ 4 * k * ((1 - x - y) ^ 2 / (1 - η₀ - x) ^ 2)
        + 2 * (gateDevT η₀ * t + gateDevD η₀ * d + gateDevU η₀ * tu) := by
  have h0 : 0 ≤ η₀ := le_trans hx0 hxη
  have hHw0 := gateGap_pos h0 hη₀
  have hHw1 := gateGap_le h0 hη₀
  set Hw := gateGap η₀ with hHw
  set D : ℝ := 1 - η₀ - x with hD
  set Δ : ℝ := 1 - x - y with hΔ
  have hD0 : 0 < D := by rw [hD]; linarith
  have hDHw : Hw ≤ D := by rw [hD]; linarith
  have hDΔ : D ≤ Δ := by rw [hD, hΔ]; linarith
  have hΔ1 : Δ ≤ 1 := by rw [hΔ]; linarith
  have hcap1 := wrongCap_le_one n ag ap d η₀
  have hcap0 := wrongCap_nonneg n ag ap d η₀
  obtain ⟨hT0, hD0', hU0⟩ := gateDev_pos h0 hη₀
  have hdev0 : 0 ≤ gateDevT η₀ * t + gateDevD η₀ * d + gateDevU η₀ * tu := by positivity
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hch0 : 0 ≤ 4 * k * (Δ ^ 2 / D ^ 2) :=
    mul_nonneg (by positivity) (div_nonneg (sq_nonneg _) (sq_nonneg _))
  rcases Nat.eq_zero_or_pos n with hn0 | hn
  · rw [hn0, Nat.cast_zero, zero_mul]; linarith
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hn1 : (n : ℝ) * wrongCap n ag ap d η₀ ≤ n := by nlinarith
  -- a side mostly wrong is charged by its size
  by_cases hA : (n : ℝ) * D ≤ 4 * k * Δ
  · have : (n : ℝ) ≤ 4 * k * (Δ ^ 2 / D ^ 2) := by
      rw [mul_div_assoc', le_div_iff₀ (by positivity)]
      have h1 : (n : ℝ) * D * D ≤ 4 * k * Δ * D := mul_le_mul_of_nonneg_right hA hD0.le
      have h2 : 4 * k * Δ * D ≤ 4 * k * Δ * Δ :=
        mul_le_mul_of_nonneg_left hDΔ (mul_nonneg (by positivity) (by linarith))
      nlinarith
    linarith
  push Not at hA
  -- a side too small for its deviations
  by_cases hC : 3 * (n : ℝ) * D < 16 * t
  · have h1 : (n : ℝ) ≤ 16 * t / (3 * Hw) := by
      rw [le_div_iff₀ (by positivity)]
      nlinarith [mul_le_mul_of_nonneg_left hDHw (by positivity : (0 : ℝ) ≤ 3 * n)]
    have h2 : 16 * t / (3 * Hw) ≤ 2 * (gateDevT η₀ * t) := by
      unfold gateDevT
      rw [← hHw]
      have e : 2 * ((16 / (5 * Hw) + 9 / (4 * Hw ^ 2)) * t)
          = 16 * t / (3 * Hw) + (16 / (15 * Hw) * t + 9 / (2 * Hw ^ 2) * t) := by
        field_simp; ring
      rw [e]
      have : 0 ≤ 16 / (15 * Hw) * t + 9 / (2 * Hw ^ 2) * t := by positivity
      linarith
    have h3 : 2 * (gateDevT η₀ * t) ≤ 2 * (gateDevT η₀ * t + gateDevD η₀ * d + gateDevU η₀ * tu) :=
      by have := mul_nonneg hD0'.le hd; have := mul_nonneg hU0.le htu; linarith
    linarith
  push Not at hC
  set κ : ℝ := k / n with hκ
  set ε : ℝ := t / n with hε
  set c : ℝ := (ag : ℝ) / n with hc
  have hκn : κ * n = k := by rw [hκ]; field_simp
  have hεn : ε * n = t := by rw [hε]; field_simp
  have hκ0 : 0 ≤ κ := div_nonneg hk0 hnR.le
  have hκ1 : κ ≤ 1 := by rw [hκ, div_le_one hnR]; exact_mod_cast hk
  have hε0 : 0 ≤ ε := div_nonneg ht hnR.le
  have hεD : 16 * ε ≤ 3 * D := by
    rw [hε, show 16 * (t / n) = 16 * t / n by ring, div_le_iff₀ hnR]
    linarith
  have hκΔ : 4 * (κ * Δ) < D := by
    rw [hκ, div_mul_eq_mul_div]
    rw [show 4 * (k * Δ / n) = 4 * k * Δ / n by ring, div_lt_iff₀ hnR]
    linarith
  have hchi : c ≤ x + κ * Δ + ε := by
    rw [hc, hκ, hε, hΔ, div_le_iff₀ hnR]
    have : (x + k / n * (1 - x - y) + t / n) * n = ((n : ℝ) - k) * x + k * (1 - y) + t := by
      field_simp; ring
    linarith
  have hclo : x + κ * Δ - ε ≤ c := by
    rw [hc, hκ, hε, hΔ, le_div_iff₀ hnR]
    have : (x + k / n * (1 - x - y) - t / n) * n = ((n : ℝ) - k) * x + k * (1 - y) - t := by
      field_simp; ring
    linarith
  -- the cap's denominator is at least `9D/16`
  have hden : 9 / 16 * D ≤ 1 - η₀ - c := by rw [hD] at hκΔ hεD ⊢; nlinarith
  have hden0 : 0 < 1 - η₀ - c := by linarith
  have hc0 : 0 ≤ c := by rw [hc]; exact div_nonneg (Nat.cast_nonneg _) hnR.le
  have hc1 : c ≤ 1 := by linarith
  have hsp := spread_at_le n k ag ap hk hn hd hlo hhi hb hc0 hc1
  rw [← hc] at hsp
  have hsplit : ((n : ℝ) - k) / n * (x - c) ^ 2 + k / n * (1 - y - c) ^ 2
      = (c - x - κ * Δ) ^ 2 + κ * (1 - κ) * Δ ^ 2 := by
    rw [hκ, hΔ]; field_simp; ring
  have hsq : (c - x - κ * Δ) ^ 2 ≤ ε ^ 2 := by
    have : |c - x - κ * Δ| ≤ ε := abs_le.2 ⟨by linarith, by linarith⟩
    calc (c - x - κ * Δ) ^ 2 = |c - x - κ * Δ| ^ 2 := (sq_abs _).symm
      _ ≤ ε ^ 2 := pow_le_pow_left₀ (abs_nonneg _) this 2
  set Dv : ℝ := (t + 3 / 2 * d + tu / 2) / n with hDv
  have hDv0 : 0 ≤ Dv := by rw [hDv]; positivity
  have hκ1' : κ * (1 - κ) * Δ ^ 2 ≤ κ * Δ ^ 2 := by
    have : 0 ≤ κ * κ * Δ ^ 2 := by positivity
    nlinarith
  have hSp : spreadBound n ag ap d c ≤ κ * Δ ^ 2 + (ε ^ 2 + Dv) := by
    rw [hsplit] at hsp; linarith
  have hnum0 : 0 ≤ κ * Δ ^ 2 + (ε ^ 2 + Dv) := by positivity
  have hcapS : wrongCap n ag ap d η₀ ≤ (κ * Δ ^ 2 + (ε ^ 2 + Dv)) / (1 - η₀ - c) ^ 2 := by
    unfold wrongCap
    rw [if_pos ⟨hn, by rw [← hc]; exact hden0⟩, ← hc]
    refine le_trans (min_le_right _ _) ?_
    refine div_le_div_of_nonneg_right ?_ (sq_nonneg _)
    exact max_le hnum0 hSp
  have hcap2 : wrongCap n ag ap d η₀ ≤ (κ * Δ ^ 2 + (ε ^ 2 + Dv)) / (9 / 16 * D) ^ 2 :=
    le_trans hcapS (div_le_div_of_nonneg_left hnum0 (by positivity)
      (pow_le_pow_left₀ (by positivity) hden 2))
  -- scaled by the side's size
  have e1 : (n : ℝ) * ((κ * Δ ^ 2 + (ε ^ 2 + Dv)) / (9 / 16 * D) ^ 2)
      = 256 / 81 * (k * (Δ ^ 2 / D ^ 2)) + 256 / 81 * (t * ε + (t + 3 / 2 * d + tu / 2)) / D ^ 2 := by
    have hn0 : (n : ℝ) ≠ 0 := hnR.ne'
    have hD0' : D ≠ 0 := hD0.ne'
    rw [hκ, hε, hDv]; field_simp; ring
  have hmain := mul_le_mul_of_nonneg_left hcap2 hnR.le
  rw [e1] at hmain
  have hε1 : ε ≤ 3 / 16 := by nlinarith
  have hte : t * ε ≤ 3 / 16 * t := (mul_le_mul_of_nonneg_left hε1 ht).trans_eq (by ring)
  have hdevD : 256 / 81 * (t * ε + (t + 3 / 2 * d + tu / 2)) / D ^ 2
      ≤ 2 * (gateDevT η₀ * t + gateDevD η₀ * d + gateDevU η₀ * tu) := by
    have hnum : 0 ≤ 256 / 81 * (t * ε + (t + 3 / 2 * d + tu / 2)) := by positivity
    refine le_trans (div_le_div_of_nonneg_left hnum (by positivity)
      (pow_le_pow_left₀ hHw0.le hDHw 2)) ?_
    unfold gateDevT gateDevD gateDevU
    rw [← hHw]
    rw [div_le_iff₀ (by positivity)]
    have e : 2 * ((16 / (5 * Hw) + 9 / (4 * Hw ^ 2)) * t + 27 / (10 * Hw ^ 2) * d
        + 9 / (10 * Hw ^ 2) * tu) * Hw ^ 2
        = 32 / 5 * Hw * t + 9 / 2 * t + 27 / 5 * d + 9 / 5 * tu := by
      field_simp; ring
    rw [e]
    have : 0 ≤ Hw * t := by positivity
    linarith
  have hk' : 256 / 81 * (k * (Δ ^ 2 / D ^ 2)) ≤ 4 * k * (Δ ^ 2 / D ^ 2) := by
    have : 0 ≤ k * (Δ ^ 2 / D ^ 2) := by positivity
    have e : 4 * (k : ℝ) * (Δ ^ 2 / D ^ 2) = 4 * (k * (Δ ^ 2 / D ^ 2)) := by ring
    rw [e]; linarith
  linarith

/-- What the rate-aware charge costs against the vote's margin on the wrong class:
with `m = 1/2 − y` that margin, `D = 1 − η₀ − x` and `v ≤ 4(1/2 − η₀)/5`,

    (1 − x − y)²/D² ≤ (5/(2(1/2 − η₀)²)) · 2m(m − v).

The charge is at most `m²/(1/2 − η₀)²`, and the margin keeps a fifth of `m`. -/
lemma charge_le_sig {x y η₀ v : ℝ} (hxη : x ≤ η₀) (hyη : y ≤ η₀)
    (hη₀ : η₀ < 1 / 2) (hv : v ≤ 4 * (1 / 2 - η₀) / 5) :
    (1 - x - y) ^ 2 / (1 - η₀ - x) ^ 2
      ≤ 5 / (2 * (1 / 2 - η₀) ^ 2) * (2 * (1 / 2 - y) * (1 / 2 - y - v)) := by
  set s : ℝ := 1 / 2 - η₀ with hs
  set mw : ℝ := 1 / 2 - y with hmw
  set mr : ℝ := 1 / 2 - x with hmr
  have hs0 : 0 < s := by rw [hs]; linarith
  have hmws : s ≤ mw := by rw [hmw, hs]; linarith
  have hmr0 : 0 ≤ mr := by rw [hmr]; linarith
  have hmw0 : 0 ≤ mw := le_trans hs0.le hmws
  have hD : 1 - η₀ - x = s + mr := by rw [hs, hmr]; ring
  have hΔ : 1 - x - y = mw + mr := by rw [hmw, hmr]; ring
  rw [hD, hΔ]
  have hD0 : 0 < s + mr := by linarith
  -- `(mw + mr)/(s + mr) ≤ mw/s`
  have hratio : (mw + mr) * s ≤ mw * (s + mr) := by nlinarith [mul_le_mul_of_nonneg_right hmws hmr0]
  have h1 : (mw + mr) ^ 2 / (s + mr) ^ 2 ≤ mw ^ 2 / s ^ 2 := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have := pow_le_pow_left₀ (by positivity) hratio 2
    nlinarith [this]
  have h2 : mw ^ 2 ≤ 5 / 2 * (2 * mw * (mw - v)) := by
    nlinarith [mul_le_mul_of_nonneg_left (show v ≤ 4 / 5 * mw by linarith) hmw0]
  refine le_trans h1 ?_
  rw [div_le_iff₀ (by positivity)]
  have e : 5 / (2 * s ^ 2) * (2 * mw * (mw - v)) * s ^ 2 = 5 / 2 * (2 * mw * (mw - v)) := by
    field_simp
  rw [e]
  exact h2

end OrthoDFA
