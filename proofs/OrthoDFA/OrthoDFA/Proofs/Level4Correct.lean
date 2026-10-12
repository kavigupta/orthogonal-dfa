import OrthoDFA.Proofs.Level4RiskBound
import OrthoDFA.Level4Family

/-!
# Level 4's claim

`Level4Correct` from `Level4RoundRisk` at `κs = 2κ` and `Level4RiskBound` bounding its
`roundRisk` by `δ`, at settings chosen as formulas of the problem. With `N = |Q| + S + 3`,
`L₁ = L + 1` and `s = √(|Σ| ρ/3)`: `maxLeaves = N`, `middleUndecidedRate = 1/4`,
`startUndecidedRate = 1/2`, `searchShare = 3εd/4`, `edgeUndecidedRate = 8s + 1/(L₁ N)`,
`edgeTraffic = 4s L₁/|Σ|`, `testLevel = e^{−G}`, `firstLook = ⌈(512 + 32/(9εd²)) G⌉`,
`edgeExcess = 6 L₁ G`, `recordsToMove` the records the caps ask for, and `probeBudget` the budget
`Level4RiskBound` asks for. `edgeUndecidedRate` and `edgeTraffic` balance the edges' share of the
records cap against the prefix mass the edge field admits.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- The tests' level at `a = e^{−G}`. -/
theorem level_le (P K b G δ : ℝ) (hP : 0 ≤ P) (hK : 100 ≤ K) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1)
    (hb0 : 0 ≤ b)
    (hG : G = Real.log (4 * (P + 7 * K ^ 2) / δ) + 2
      * Real.log (2 * Real.log (4 * (P + 7 * K ^ 2) / δ))) (hb : b ≤ K * G) :
    4 * (P + 7 * b ^ 2) * Real.exp (-G) ≤ δ := by
  set X := 4 * (P + 7 * K ^ 2) / δ
  set l := Real.log X
  have hX0 : 0 < X := by positivity
  have hδX : δ * X = 4 * (P + 7 * K ^ 2) := by simp only [X]; field_simp
  have he10 : Real.exp 10 ≤ X := by
    have h1 : Real.exp 10 = Real.exp 1 ^ 10 := by rw [← Real.exp_nat_mul]; norm_num
    have h2 : Real.exp 1 ^ 10 ≤ 3 ^ 10 :=
      pow_le_pow_left₀ (Real.exp_pos 1).le Real.exp_one_lt_three.le 10
    have h3 : (3 : ℝ) ^ 10 ≤ X := by
      simp only [X]
      rw [le_div_iff₀ hδ0]
      nlinarith [sq_nonneg K]
    linarith
  have hl : 10 ≤ l := (Real.le_log_iff_exp_le hX0).2 he10
  have hXl : Real.exp l = X := Real.exp_log hX0
  have h4 : 4 * l ^ 2 ≤ X := by
    have h := Real.pow_div_factorial_le_exp (x := l) (by linarith) 4
    rw [hXl] at h
    have hf : ((Nat.factorial 4 : ℕ) : ℝ) = 24 := by norm_num [Nat.factorial]
    rw [hf] at h
    nlinarith [sq_nonneg l, mul_le_mul hl hl (by norm_num) (by linarith)]
  have h2l : 0 < 2 * l := by linarith
  have hlog2 : 2 * Real.log (2 * l) = Real.log ((2 * l) ^ 2) := by
    rw [Real.log_pow]; push_cast; ring
  have hG2 : G ≤ 2 * l := by
    have : Real.log ((2 * l) ^ 2) ≤ l :=
      (Real.log_le_log (by positivity) (by nlinarith)).trans_eq rfl
    rw [hG, hlog2]; linarith
  have hexpG : Real.exp G = X * (2 * l) ^ 2 := by
    rw [hG, Real.exp_add, hlog2, Real.exp_log hX0, Real.exp_log (by positivity)]
  have hbsq : b ^ 2 ≤ K ^ 2 * (4 * l ^ 2) := by
    have h1 : b ≤ K * (2 * l) := hb.trans (mul_le_mul_of_nonneg_left hG2 (by linarith))
    have := mul_le_mul h1 h1 hb0 (by positivity)
    nlinarith [this]
  rw [Real.exp_neg, hexpG, ← div_eq_mul_inv, div_le_iff₀ (by positivity)]
  have : 4 * (P + 7 * b ^ 2) ≤ 4 * (P + 7 * K ^ 2) * (2 * l) ^ 2 := by
    nlinarith [mul_le_mul_of_nonneg_left (by nlinarith : (1 : ℝ) ≤ 4 * l ^ 2) hP]
  calc 4 * (P + 7 * b ^ 2) ≤ 4 * (P + 7 * K ^ 2) * (2 * l) ^ 2 := this
    _ = δ * (X * (2 * l) ^ 2) := by rw [← hδX]; ring

/-- The chosen budget is at most `K G`. -/
theorem budget_le (N1 A L1 εd G lo S1 Z n₀ m b : ℝ) (hN1 : 2 ≤ N1) (hA : 1 ≤ A) (hεd0 : 0 < εd)
    (hεd1 : εd ≤ 1) (hG : 1 ≤ G) (hlo0 : 0 ≤ lo) (hlo : lo ≤ G) (hS1 : 1 ≤ S1) (hZ0 : 0 < Z)
    (hZ : Z ≤ N1 * A) (hn₀ : n₀ ≤ (512 + 32 / (9 * εd ^ 2)) * G + 1)
    (hm : m ≤ εd * ((2 * (N1 + 2) * A + 1) * (128 * G / εd ^ 2 + 4 * n₀ / (3 * εd) + 2)
      + 32 * G / εd ^ 2 + 2) / (8 * Z) + εd * (32 * (N1 + 2) * A * L1 * G + n₀) / (32 * N1 * Z)
      + 4 * lo / S1 + 1)
    (hb : b ≤ N1 * ((2 * (N1 + 2) * A + 1) * (128 * G / εd ^ 2 + 4 * n₀ / (3 * εd) + 2)
      + 8 * Z * m / εd + 32 * G / εd ^ 2 + 1) + 1) :
    b ≤ (N1 * (2 * (N1 + 2) * A + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
      + 8 * (N1 + 2) * A * L1) * G := by
  obtain ⟨u, hu⟩ : ∃ u : ℝ, u = 1 / εd := ⟨_, rfl⟩
  have hu0 : 0 < u := by rw [hu]; positivity
  have hεd : εd = 1 / u := by rw [hu]; field_simp
  have hu1 : 1 ≤ u := by rw [hu, le_div_iff₀ hεd0]; linarith
  obtain ⟨N, hN⟩ : ∃ N : ℝ, N = N1 + 2 := ⟨_, rfl⟩
  rw [← hN] at hm hb ⊢
  subst hεd
  obtain ⟨P1, hP1⟩ : ∃ x : ℝ, x = 128 * G * u ^ 2 + 4 * n₀ / 3 * u + 2 := ⟨_, rfl⟩
  obtain ⟨B1, hB1⟩ : ∃ x : ℝ, x = (2 * N * A + 1) * P1 + 32 * G * u ^ 2 + 2 := ⟨_, rfl⟩
  have hm' : m ≤ B1 / (8 * Z * u) + (32 * N * A * L1 * G + n₀) / (32 * N1 * Z * u)
      + 4 * lo / S1 + 1 := by
    convert hm using 3
    rw [hB1, hP1]; field_simp
  have hb2 : b ≤ N1 * (B1 - 1) + N1 * (8 * Z * m * u) + 1 := by
    convert hb using 2
    rw [hB1, hP1]; field_simp; ring
  have hn₀' : n₀ ≤ (512 + 32 / 9 * u ^ 2) * G + 1 := by convert hn₀ using 3; field_simp
  rw [show (10 / (1 / u) ^ 3 + 270 / (1 / u) ^ 2 + 1400 / (1 / u))
    = 10 * u ^ 3 + 270 * u ^ 2 + 1400 * u by field_simp]
  -- the records' part of the budget
  have hZm : 8 * Z * m * u ≤ B1 + (32 * N * A * L1 * G + n₀) / (4 * N1)
      + 32 * Z * lo * u / S1 + 8 * Z * u := by
    have h8 : 0 ≤ 8 * Z * u := by positivity
    have := mul_le_mul_of_nonneg_left hm' h8
    have e : 8 * Z * u * (B1 / (8 * Z * u) + (32 * N * A * L1 * G + n₀) / (32 * N1 * Z * u)
        + 4 * lo / S1 + 1) = B1 + (32 * N * A * L1 * G + n₀) / (4 * N1)
          + 32 * Z * lo * u / S1 + 8 * Z * u := by
      field_simp; ring
    linarith [this, e]
  have hb' : b ≤ N1 * (B1 - 1) + N1 * (B1 + (32 * N * A * L1 * G + n₀) / (4 * N1)
      + 32 * Z * lo * u / S1 + 8 * Z * u) + 1 := by
    have : N1 * (8 * Z * m * u) ≤ N1 * (B1 + (32 * N * A * L1 * G + n₀) / (4 * N1)
      + 32 * Z * lo * u / S1 + 8 * Z * u) := mul_le_mul_of_nonneg_left hZm (by linarith)
    linarith
  obtain ⟨W, hW⟩ : ∃ x : ℝ, x = N1 * (2 * N * A + 1) := ⟨_, rfl⟩
  have hNA : N1 * A ≤ N * A := by rw [hN]; nlinarith only [hA]
  have hW2 : 2 * N1 * (N1 * A) ≤ W := by
    rw [hW]
    nlinarith only [mul_le_mul_of_nonneg_left hNA (by linarith : (0 : ℝ) ≤ N1), hN1]
  have hW9 : 9 * N1 ≤ W := by
    have h4 : 4 ≤ N := by rw [hN]; linarith
    have h8 : 8 ≤ 2 * N * A := by nlinarith only [h4, hA]
    rw [hW]; nlinarith only [mul_le_mul_of_nonneg_left h8 (by linarith : (0 : ℝ) ≤ N1)]
  have hW18 : 18 ≤ W := by linarith
  have hu2 : u ≤ u ^ 2 := by nlinarith only [hu1]
  have hGu : 0 ≤ G * u := by positivity
  obtain ⟨T1, hT1⟩ : ∃ t, t = W * G * u := ⟨_, rfl⟩
  obtain ⟨T2, hT2⟩ : ∃ t, t = W * G * u ^ 2 := ⟨_, rfl⟩
  obtain ⟨T3, hT3⟩ : ∃ t, t = W * G * u ^ 3 := ⟨_, rfl⟩
  have hW0 : 0 ≤ W := by linarith
  have hT12 : T1 ≤ T2 := by rw [hT1, hT2]; exact mul_le_mul_of_nonneg_left hu2 (by positivity)
  have hT1W : W ≤ T1 := by
    rw [hT1]; have : 1 ≤ G * u := by nlinarith only [hG, hu1]
    nlinarith only [this, hW0]
  have hT30 : 0 ≤ T3 := by rw [hT3]; positivity
  have hn₀u : n₀ * u ≤ 512 * (G * u) + 32 / 9 * (G * u ^ 3) + u := by
    have := mul_le_mul_of_nonneg_right hn₀' hu0.le
    have e : ((512 + 32 / 9 * u ^ 2) * G + 1) * u = 512 * (G * u) + 32 / 9 * (G * u ^ 3) + u := by
      ring
    linarith
  have p1 : 2 * N1 * ((2 * N * A + 1) * P1) ≤ 256 * T2 + 4096 / 3 * T1 + 256 / 27 * T3
      + 8 / 3 * (W * u) + 4 * W := by
    have e : 2 * N1 * ((2 * N * A + 1) * P1) = W * (256 * (G * u ^ 2) + 8 / 3 * (n₀ * u) + 4) := by
      rw [hW, hP1]; ring
    have e2 : 256 * T2 + 4096 / 3 * T1 + 256 / 27 * T3 + 8 / 3 * (W * u) + 4 * W
        = W * (256 * (G * u ^ 2) + 8 / 3 * (512 * (G * u) + 32 / 9 * (G * u ^ 3) + u) + 4) := by
      rw [hT1, hT2, hT3]; ring
    rw [e, e2]
    exact mul_le_mul_of_nonneg_left (by linarith) hW0
  have hGu2 : 0 ≤ G * u ^ 2 := by positivity
  have p2 : 2 * N1 * (32 * G * u ^ 2) ≤ 64 / 9 * T2 := by
    have := mul_le_mul_of_nonneg_right hW9 hGu2
    rw [hT2]; nlinarith only [this]
  have p3 : n₀ / 4 ≤ 128 * G + 8 / 9 * (G * u ^ 2) + 1 / 4 := by
    have e : (512 + 32 / 9 * u ^ 2) * G + 1 = 512 * G + 32 / 9 * (G * u ^ 2) + 1 := by ring
    linarith
  have p4 : 128 * G ≤ 128 / 18 * T1 := by
    have := mul_le_mul_of_nonneg_right hW18 hGu
    have h1 : G * 1 ≤ G * u := mul_le_mul_of_nonneg_left hu1 (by linarith)
    rw [hT1]; nlinarith only [this, h1]
  have p5 : 8 / 9 * (G * u ^ 2) ≤ 8 / 9 / 18 * T2 := by
    have := mul_le_mul_of_nonneg_right hW18 hGu2
    rw [hT2]; nlinarith only [this]
  have p6 : N1 * (32 * Z * lo * u / S1) ≤ 16 * T1 := by
    have h1 : 32 * Z * lo * u / S1 ≤ 32 * Z * lo * u := by
      have : 0 ≤ 32 * Z * lo * u := by positivity
      rw [div_le_iff₀ (by linarith)]; nlinarith only [this, hS1]
    have h2 : 32 * Z * lo * u ≤ 32 * (N1 * A) * (G * u) := by
      have := mul_le_mul hZ hlo hlo0 (by nlinarith only [hN1, hA])
      have e : 32 * (N1 * A) * (G * u) = 32 * u * ((N1 * A) * G) := by ring
      rw [e, show 32 * Z * lo * u = 32 * u * (Z * lo) by ring]
      exact mul_le_mul_of_nonneg_left this (by positivity)
    have h3 : N1 * (32 * (N1 * A) * (G * u)) ≤ 16 * T1 := by
      have := mul_le_mul_of_nonneg_right hW2 hGu
      rw [hT1]; nlinarith only [this]
    calc _ ≤ N1 * (32 * (N1 * A) * (G * u)) :=
          mul_le_mul_of_nonneg_left (h1.trans h2) (by linarith)
      _ ≤ _ := h3
  have hWu : W * u ≤ T1 := by
    rw [hT1]; nlinarith only [mul_le_mul_of_nonneg_left hG (by positivity : (0 : ℝ) ≤ W * u)]
  have p7 : N1 * (8 * Z * u) ≤ 4 * T1 := by
    have h1 : N1 * (8 * Z * u) ≤ N1 * (8 * (N1 * A) * u) :=
      mul_le_mul_of_nonneg_left (by nlinarith only [hZ, hu0]) (by linarith)
    have h2 : N1 * (8 * (N1 * A) * u) ≤ 4 * (W * u) := by
      nlinarith only [mul_le_mul_of_nonneg_right hW2 hu0.le]
    linarith
  have p9 : 3 * N1 + 5 / 4 ≤ 5 / 12 * T2 := by
    have : W ≤ T2 := hT1W.trans hT12
    linarith
  have hgoal : (N1 * (2 * N * A + 1) * (10 * u ^ 3 + 270 * u ^ 2 + 1400 * u) + 8 * N * A * L1) * G
      = 10 * T3 + 270 * T2 + 1400 * T1 + 8 * N * A * L1 * G := by
    rw [hT1, hT2, hT3, hW]; ring
  have e2 : N1 * (B1 - 1) + N1 * (B1 + (32 * N * A * L1 * G + n₀) / (4 * N1)
      + 32 * Z * lo * u / S1 + 8 * Z * u) + 1
      = 2 * N1 * ((2 * N * A + 1) * P1) + 2 * N1 * (32 * G * u ^ 2) + 3 * N1 + 1
        + 8 * N * A * L1 * G + n₀ / 4 + N1 * (32 * Z * lo * u / S1) + N1 * (8 * Z * u) := by
    rw [hB1]; field_simp; ring
  rw [hgoal]
  linarith [hb'.trans_eq e2, hT1W.trans hT12]

/-- A records cap, `16 X b ≤ (S + 1) m`, once `X ≤ ρ`. -/
theorem cap_half (X ρ b m N1 Z S1 εd B1 : ℝ) (hX : 0 ≤ X) (hXρ : X ≤ ρ)
    (hρ : 256 * ρ * N1 * Z = S1 * εd) (hεd : 0 < εd) (hZ : 0 < Z) (hN1 : 0 ≤ N1) (hm0 : 0 ≤ m)
    (hb : b ≤ N1 * (B1 + 8 * Z * m / εd)) (hm : εd * B1 / (8 * Z) ≤ m) :
    16 * X * b ≤ S1 * m := by
  have hB : B1 ≤ 8 * Z * m / εd := by
    rw [div_le_iff₀ (by positivity)] at hm
    rw [le_div_iff₀ hεd]; linarith
  have hb' : b ≤ N1 * (16 * Z * m / εd) := by
    refine hb.trans (mul_le_mul_of_nonneg_left ?_ hN1)
    have : 16 * Z * m / εd = 8 * Z * m / εd + 8 * Z * m / εd := by ring
    linarith
  have h1 : 16 * X * b ≤ 16 * X * (N1 * (16 * Z * m / εd)) :=
    mul_le_mul_of_nonneg_left hb' (by positivity)
  have h2 : 16 * X * (N1 * (16 * Z * m / εd)) ≤ 16 * ρ * (N1 * (16 * Z * m / εd)) :=
    mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  have h3 : 16 * ρ * (N1 * (16 * Z * m / εd)) = S1 * m := by
    rw [show 16 * ρ * (N1 * (16 * Z * m / εd)) = 256 * ρ * N1 * Z * m / εd by ring, hρ]
    field_simp
  linarith

/-- A run's ending, as `RunEnds` checks it, is `runProbes`'s. -/
theorem runEnds_iff {σ : Type*} (M : _root_.DFA α σ) (bad : σ → Prop) (D : Measure (FreeMonoid α))
    (C : Level4Settings) (a L : ℕ) (y : ℝ) (rd : FreeMonoid α → ARU) :
    ∀ (s : Level4State α) (xs : List (FreeMonoid α)),
      RunEnds (applyProbe C fun z => (rd z).cut) (Level4EndsWell M bad D C a L y rd) s xs ↔
        EndsWell M bad D C.disagreementRate
          (searchingProbes rd C.walkStart ⟨(runProbes C (fun z => (rd z).cut) s xs).1.tree,
            Function.uncurry (runProbes C (fun z => (rd z).cut) s xs).1.edges⟩) a L
          (level4Replay rd ⟨(runProbes C (fun z => (rd z).cut) s xs).1.tree,
            Function.uncurry (runProbes C (fun z => (rd z).cut) s xs).1.edges⟩) y
          (runProbes C (fun z => (rd z).cut) s xs).2
  | s, [] => by simp [RunEnds, runProbes, EndsWell]
  | s, x :: xs => by
    cases h : applyProbe C (fun z => (rd z).cut) s x with
    | inl s' => simpa only [RunEnds, runProbes, h] using runEnds_iff M bad D C a L y rd s' xs
    | inr es => obtain ⟨e, s'⟩ := es; simp only [RunEnds, runProbes, h]; rfl

theorem lintegral_level4_eq {σ : Type*} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (M : _root_.DFA α σ) (O : TriOracle μ M) (bad : σ → Prop) (P : PrefixDistribution α)
    (C : Level4Settings) (y : ℝ) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel4 C (O.read · ω) draws
          ¬ EndsWell M bad P.D C.disagreementRate
            (searchingProbes (O.read · ω) C.walkStart result.hypothesis)
            P.firstAnchor P.L (level4Replay (O.read · ω) result.hypothesis) y result.ending} ∂μ
      = ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (O.read z ω).cut)
          (Level4EndsWell M bad P.D C P.firstAnchor P.L y (O.read · ω))
          initialState (List.ofFn xs)} ∂μ := by
  congr 1; ext ω; congr 1; ext xs
  simp only [Set.mem_setOf_eq, runEnds_iff, indecisiveLStarLevel4]

section Guarantee

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

set_option maxHeartbeats 4000000 in -- the chosen settings meet every condition
theorem level4_guarantee (M : _root_.DFA α σ) (O : TriOracle μ M) (θ εd : ℝ)
    (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ) : Level4Guarantee M O θ εd P δ S k := by
  intro hδ0 hδ1 hεd0 hεd1 hL hθ hθc hκ hκc hε hεc
  generalize hρ : ((S : ℝ) + 1) * εd / (256 * ((Fintype.card σ : ℝ) + S + 1)
    * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)) = ρ at hκc hεc
  generalize hK : ((Fintype.card σ : ℝ) + S + 1) * (2 * ((Fintype.card σ : ℝ) + S + 3)
    * Fintype.card α + 1) * (10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd)
    + 8 * ((Fintype.card σ : ℝ) + S + 3) * Fintype.card α * (P.L + 1) = K
  generalize hG : Real.log (4 * (2 ^ (Fintype.card σ + S + 4) * (Fintype.card α : ℝ) + 7 * K ^ 2)
    / δ) + 2 * Real.log (2 * Real.log (4 * (2 ^ (Fintype.card σ + S + 4) * (Fintype.card α : ℝ)
    + 7 * K ^ 2) / δ)) = G
  -- the problem's sizes
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Fintype.card σ := by exact_mod_cast Fintype.card_pos
  have hS0 : (0 : ℝ) ≤ S := Nat.cast_nonneg _
  have hL1 : (1 : ℝ) ≤ P.L := by exact_mod_cast hL
  have hNr : (((Fintype.card σ + S + 3 : ℕ) : ℝ)) = ((Fintype.card σ : ℝ) + S + 1) + 2 := by
    push_cast; ring
  have hZ1 : (S : ℝ) + 1 ≤ (Fintype.card σ : ℝ) * Fintype.card α + S + 1 := by
    have : (0 : ℝ) ≤ (Fintype.card σ : ℝ) * Fintype.card α := by positivity
    linarith
  have hZN : (Fintype.card σ : ℝ) * Fintype.card α + S + 1
      ≤ ((Fintype.card σ : ℝ) + S + 1) * Fintype.card α := by nlinarith only [hA1, hS0]
  have hρ0 : 0 < ρ := by rw [← hρ]; positivity
  have hρe : 256 * ρ * ((Fintype.card σ : ℝ) + S + 1)
      * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) = (S + 1) * εd := by
    rw [← hρ]; field_simp
  have hρεd : 512 * ρ ≤ εd := by
    have h2 : (2 : ℝ) ≤ (Fintype.card σ : ℝ) + S + 1 := by linarith
    have hZ0 : (0 : ℝ) < (Fintype.card σ : ℝ) * Fintype.card α + S + 1 := by positivity
    nlinarith only [hρe, h2, hZ1, hρ0, hZ0, hεd0, mul_le_mul_of_nonneg_left hZ1 hρ0.le,
      mul_le_mul_of_nonneg_left h2 (mul_pos hρ0 hZ0).le]
  -- the edges' balance
  obtain ⟨s, hs⟩ : ∃ x : ℝ, x = Real.sqrt (Fintype.card α * ρ / 3) := ⟨_, rfl⟩
  rw [← hs] at hκc
  have hs0 : 0 ≤ s := hs ▸ Real.sqrt_nonneg _
  have hs2 : s ^ 2 = Fintype.card α * ρ / 3 := by rw [hs]; exact Real.sq_sqrt (by positivity)
  -- the level and its logarithm
  have hK100 : 100 ≤ K := by
    rw [← hK]
    have h1 : (1 : ℝ) ≤ (Fintype.card σ : ℝ) + S + 1 := by linarith
    have h2 : (1 : ℝ) ≤ 2 * ((Fintype.card σ : ℝ) + S + 3) * Fintype.card α + 1 := by
      nlinarith only [hA1, hS0, hQ1]
    have h3 : 1400 ≤ 10 / εd ^ 3 + 270 / εd ^ 2 + 1400 / εd := by
      have : 1400 ≤ 1400 / εd := by rw [le_div_iff₀ hεd0]; linarith
      have : 0 ≤ 10 / εd ^ 3 + 270 / εd ^ 2 := by positivity
      linarith
    have h4 : 0 ≤ 8 * ((Fintype.card σ : ℝ) + S + 3) * Fintype.card α * (P.L + 1) := by
      positivity
    nlinarith only [mul_le_mul h1 h2 zero_le_one (by linarith), h3, h4,
      mul_le_mul (mul_le_mul h1 h2 zero_le_one (by linarith)) h3 (by norm_num) (by positivity)]
  set Xl := 4 * (2 ^ (Fintype.card σ + S + 4) * (Fintype.card α : ℝ) + 7 * K ^ 2) / δ with hXl
  have hXl0 : 0 < Xl := by positivity
  have hlogX : 1 ≤ Real.log Xl := by
    rw [Real.le_log_iff_exp_le hXl0]
    refine Real.exp_one_lt_three.le.trans ?_
    rw [hXl, le_div_iff₀ hδ0]
    nlinarith only [hK100, hδ1, hδ0, sq_nonneg K,
      (by positivity : (0 : ℝ) ≤ 2 ^ (Fintype.card σ + S + 4) * (Fintype.card α : ℝ))]
  have hG1 : Real.log Xl ≤ G := by
    rw [← hG]
    have : 0 ≤ Real.log (2 * Real.log Xl) := Real.log_nonneg (by linarith)
    linarith
  have hlo : Real.log (8 / δ) ≤ G := by
    refine le_trans (Real.log_le_log (by positivity) ?_) hG1
    rw [hXl, div_le_div_iff_of_pos_right hδ0]
    nlinarith only [hK100, sq_nonneg K,
      (by positivity : (0 : ℝ) ≤ 2 ^ (Fintype.card σ + S + 4) * (Fintype.card α : ℝ))]
  have hlo0 : 0 ≤ Real.log (8 / δ) :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ0]; linarith)
  have hGge : 1 ≤ G := hlogX.trans hG1
  have hlogG : Real.log (1 / Real.exp (-G)) = G := by
    rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
  -- the counts
  obtain ⟨n₀, hn₀d⟩ : ∃ x : ℕ, x = ⌈(512 + 32 / (9 * εd ^ 2)) * G⌉₊ := ⟨_, rfl⟩
  have hn₀ : (n₀ : ℝ) ≤ (512 + 32 / (9 * εd ^ 2)) * G + 1 := by
    rw [hn₀d]; exact (Nat.ceil_lt_add_one (by positivity)).le
  have hn₀l : (512 + 32 / (9 * εd ^ 2)) * G ≤ (n₀ : ℝ) := by rw [hn₀d]; exact Nat.le_ceil _
  obtain ⟨m, hmd⟩ : ∃ x : ℕ, x = ⌈εd * ((2 * (((Fintype.card σ : ℝ) + S + 1) + 2)
      * Fintype.card α + 1) * (128 * G / εd ^ 2 + 4 * (n₀ : ℝ) / (3 * εd) + 2)
      + 32 * G / εd ^ 2 + 2) / (8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1))
    + εd * (32 * (((Fintype.card σ : ℝ) + S + 1) + 2) * Fintype.card α * (P.L + 1) * G
      + n₀) / (32 * ((Fintype.card σ : ℝ) + S + 1)
        * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1))
    + 4 * Real.log (8 / δ) / (S + 1)⌉₊ := ⟨_, rfl⟩
  have hm0 : (0 : ℝ) ≤ εd * ((2 * (((Fintype.card σ : ℝ) + S + 1) + 2)
      * Fintype.card α + 1) * (128 * G / εd ^ 2 + 4 * (n₀ : ℝ) / (3 * εd) + 2)
      + 32 * G / εd ^ 2 + 2) / (8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)) := by
    positivity
  have hm1 : (0 : ℝ) ≤ εd * (32 * (((Fintype.card σ : ℝ) + S + 1) + 2) * Fintype.card α
      * (P.L + 1) * G + n₀) / (32 * ((Fintype.card σ : ℝ) + S + 1)
        * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)) := by positivity
  have hm2 : (0 : ℝ) ≤ 4 * Real.log (8 / δ) / (S + 1) := by positivity
  have hmu := (Nat.ceil_lt_add_one (add_nonneg (add_nonneg hm0 hm1) hm2)).le
  have hml := Nat.le_ceil (εd * ((2 * (((Fintype.card σ : ℝ) + S + 1) + 2)
      * Fintype.card α + 1) * (128 * G / εd ^ 2 + 4 * (n₀ : ℝ) / (3 * εd) + 2)
      + 32 * G / εd ^ 2 + 2) / (8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1))
    + εd * (32 * (((Fintype.card σ : ℝ) + S + 1) + 2) * Fintype.card α * (P.L + 1) * G
      + n₀) / (32 * ((Fintype.card σ : ℝ) + S + 1)
        * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1))
    + 4 * Real.log (8 / δ) / (S + 1))
  rw [← hmd] at hmu hml
  obtain ⟨b, hbd⟩ : ∃ x : ℕ, x = ⌈((S : ℝ) + Fintype.card σ + 1)
      * ((2 * (((Fintype.card σ + S + 3 : ℕ) : ℝ)) * Fintype.card α + 1)
        * (8 * G / (εd ^ 2 * (1 / 4) ^ 2) + 4 * (n₀ : ℝ) / (3 * εd) + 2)
      + 8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd
      + 32 * G / εd ^ 2 + 1)⌉₊ := ⟨_, rfl⟩
  have hbl := Nat.le_ceil (((S : ℝ) + Fintype.card σ + 1)
      * ((2 * (((Fintype.card σ + S + 3 : ℕ) : ℝ)) * Fintype.card α + 1)
        * (8 * G / (εd ^ 2 * (1 / 4) ^ 2) + 4 * (n₀ : ℝ) / (3 * εd) + 2)
      + 8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd
      + 32 * G / εd ^ 2 + 1))
  have hbu := (Nat.ceil_lt_add_one (show (0 : ℝ) ≤ ((S : ℝ) + Fintype.card σ + 1)
      * ((2 * (((Fintype.card σ + S + 3 : ℕ) : ℝ)) * Fintype.card α + 1)
        * (8 * G / (εd ^ 2 * (1 / 4) ^ 2) + 4 * (n₀ : ℝ) / (3 * εd) + 2)
      + 8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd
      + 32 * G / εd ^ 2 + 1) by positivity)).le
  rw [← hbd] at hbl hbu
  -- the budget's bound
  have h8G : 8 * G / (εd ^ 2 * (1 / 4) ^ 2) = 128 * G / εd ^ 2 := by field_simp; ring
  have hbK : (b : ℝ) ≤ K * G := by
    rw [← hK]
    have := budget_le ((Fintype.card σ : ℝ) + S + 1) (Fintype.card α) (P.L + 1) εd G
      (Real.log (8 / δ)) (S + 1) ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) n₀ m b
      (by linarith) hA1 hεd0 hεd1 hGge hlo0 hlo (by linarith) (by positivity) hZN hn₀ hmu
      (by rw [hNr, h8G] at hbu; linarith)
    convert this using 2
    ring
  -- shared facts
  have hp₀ : 0 < prefixMax P.D k := prefixMax_pos P.D k
  have hΛ0 : 0 ≤ unionLog (α := α) σ S δ := by
    unfold unionLog
    refine mul_nonneg (by positivity) (Real.log_nonneg ?_)
    rw [le_div_iff₀ hδ0]
    have h3 : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by linarith
    nlinarith only [h3, hA1, hδ1, hδ0]
  have hpl0 : 0 ≤ prefixMax P.D k * unionLog (α := α) σ S δ := mul_nonneg hp₀.le hΛ0
  have hεN0 : 0 ≤ ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * O.ε := by positivity
  have hpl : 3 * (prefixMax P.D k * unionLog (α := α) σ S δ) ≤ ρ := by linarith
  have hερ : ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * O.ε ≤ ρ := by linarith
  have hsN : 0 ≤ 8 * ((Fintype.card σ : ℝ) + S + 3) * (P.L + 1) * s := by positivity
  have hκρ : 4 * O.κ ≤ ρ := by nlinarith only [hκc, hsN, hκ]
  have hn1 : (1 : ℝ) ≤ (Fintype.card σ : ℝ) + S + 1 := by linarith
  have hθn : ((Fintype.card σ : ℝ) + S + 1) * θ ≤ 1 / 768 := by
    have : 0 ≤ ((Fintype.card σ : ℝ) + S + 1) * θ := by positivity
    nlinarith only [hθc, this, hL1]
  have hZ0 : (0 : ℝ) < (Fintype.card σ : ℝ) * Fintype.card α + S + 1 := by positivity
  have hm0' : (0 : ℝ) ≤ m := Nat.cast_nonneg _
  have hcap : ∀ X : ℝ, 0 ≤ X → X ≤ ρ → 16 * X * b ≤ ((S : ℝ) + 1) * m := by
    intro X hX hXρ
    refine cap_half X ρ b m ((Fintype.card σ : ℝ) + S + 1)
      ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) ((S : ℝ) + 1) εd
      ((2 * (((Fintype.card σ : ℝ) + S + 1) + 2) * Fintype.card α + 1)
        * (128 * G / εd ^ 2 + 4 * (n₀ : ℝ) / (3 * εd) + 2) + 32 * G / εd ^ 2 + 2)
      hX hXρ hρe hεd0 hZ0 (by linarith) hm0' ?_ (by linarith)
    rw [hNr, h8G] at hbu
    have e : ((S : ℝ) + Fintype.card σ + 1) = (Fintype.card σ : ℝ) + S + 1 := by ring
    rw [e] at hbu
    nlinarith only [hbu, hn1]
  refine ⟨⟨k, m, Fintype.card σ + S + 3, 1 / 2,
    8 * s + 1 / ((P.L + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ))), 1 / 4, εd, Real.exp (-G), n₀,
    fun _ => 6 * (P.L + 1) * G, 4 * s * (P.L + 1) / Fintype.card α, 3 * εd / 4, b⟩,
    rfl, rfl, hbK, ?_⟩
  have hRB := level4_risk_bound_holds σ O.κ O.ε θ P.D P.L ⟨k, m, Fintype.card σ + S + 3, 1 / 2,
    8 * s + 1 / ((P.L + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ))), 1 / 4, εd, Real.exp (-G), n₀,
    fun _ => 6 * (P.L + 1) * G, 4 * s * (P.L + 1) / Fintype.card α, 3 * εd / 4, b⟩ (2 * O.κ) δ S
  dsimp only at hRB
  rw [hlogG] at hRB
  have hrisk := hRB hδ0 hδ1 hL le_rfl hκ le_rfl ?_ hε ?_ hθ ?_ ?_ ?_ (by norm_num) ?_ ?_ hεd1
    (by norm_num) (by linarith) ?_ ?_ ?_ ?_ ?_ (Real.exp_pos _) ?_
    (by rw [h8G] at hbl ⊢; exact hbl) ?_ ?_ ?_ ?_
  · refine (lintegral_level4_eq M O _ P _ _).trans_le ?_
    calc _ ≤ _ := level4_round_risk_holds M O θ P _ (2 * O.κ) δ le_rfl hδ0 hδ1
      _ ≤ ENNReal.ofReal δ + ENNReal.ofReal δ := add_le_add hrisk le_rfl
      _ = ENNReal.ofReal (2 * δ) := by rw [← ENNReal.ofReal_add hδ0.le hδ0.le]; ring_nf
  · -- κ against the length
    have hs1 : 0 < s := by
      rcases hs0.lt_or_eq with h | h
      · exact h
      · rw [← h] at hs2; have : 0 < (Fintype.card α : ℝ) * ρ / 3 := by positivity
        linarith
    set N : ℝ := (Fintype.card σ : ℝ) + S + 3
    have hN1 : 1 ≤ N := by simp only [N]; linarith
    have hρN : ρ ≤ 4 * N * s := by
      have h1 : ρ ^ 2 ≤ (4 * N * s) ^ 2 := by
        rw [mul_pow, hs2]
        have : ρ ≤ 16 * N ^ 2 / 3 * Fintype.card α := by
          have : (1 : ℝ) ≤ N ^ 2 := one_le_pow₀ hN1
          nlinarith only [this, hA1, hρεd, hεd1]
        nlinarith only [this, hρ0]
      exact (pow_le_pow_iff_left₀ hρ0.le (by positivity) (by norm_num)).1 h1
    have : 4 * O.κ * (8 * N * (P.L + 1) * s) ≤ 4 * N * s := by nlinarith only [hκc, hρN, hκ]
    have hNs : 0 < N * s := by positivity
    nlinarith only [this, hNs]
  · have e : 4 * O.ε * (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1))
        = 4 * (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * O.ε) := by ring
    rw [e]; linarith
  · rw [hNr]
    have h1 : 24 * θ ≤ 1 / (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1 + 2)) := by
      rw [le_div_iff₀ (by positivity)]
      have : 0 ≤ θ * ((P.L : ℝ) + 1) := by positivity
      nlinarith only [hθc, this, hn1]
    have : 0 ≤ 8 * s := by positivity
    linarith
  · linarith
  · have : 8 * (prefixMax P.D k * unionLog (α := α) σ S δ) ≤ 1 / 4 := by linarith
    linarith
  · have t1 : 4 * s * ((P.L : ℝ) + 1) / Fintype.card α * (8 * s)
        = 32 * ((P.L : ℝ) + 1) * ρ / 3 := by
      rw [show 4 * s * ((P.L : ℝ) + 1) / Fintype.card α * (8 * s)
        = 32 * ((P.L : ℝ) + 1) * s ^ 2 / Fintype.card α by ring, hs2]
      field_simp
    have t2 : 0 ≤ 4 * s * ((P.L : ℝ) + 1) / Fintype.card α
        * (1 / (((P.L : ℝ) + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ)))) := by positivity
    have e : 4 * s * ((P.L : ℝ) + 1) / Fintype.card α
        * (8 * s + 1 / (((P.L : ℝ) + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ))))
        = 4 * s * ((P.L : ℝ) + 1) / Fintype.card α * (8 * s)
          + 4 * s * ((P.L : ℝ) + 1) / Fintype.card α
            * (1 / (((P.L : ℝ) + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ)))) := by ring
    rw [e, t1]
    have := mul_le_mul_of_nonneg_left hpl (by positivity : (0 : ℝ) ≤ (P.L : ℝ) + 1)
    nlinarith only [this, t2]
  · linarith
  · have : 0 ≤ 32 / (9 * εd ^ 2) * G := by positivity
    nlinarith only [hn₀l, this, hGge]
  · have : 0 ≤ 32 / (9 * εd ^ 2) * G := by positivity
    nlinarith only [hn₀l, this, hGge]
  · have h1 : 32 / (9 * εd ^ 2) * G ≤ n₀ := by
      have : 0 ≤ 512 * G := by linarith
      nlinarith only [hn₀l, this]
    have h2 := mul_le_mul_of_nonneg_right h1 (sq_nonneg (3 * εd / 4))
    have e : 32 / (9 * εd ^ 2) * G * (3 * εd / 4) ^ 2 = 2 * G := by field_simp; ring
    linarith
  · intro _
    rw [hNr]
    have h12 : 12 * θ * ((Fintype.card σ : ℝ) + S + 1 + 2) ≤ 1 / 8 := by
      have : 0 ≤ θ := hθ
      nlinarith only [hθn, this, hn1]
    have hLG : 0 < ((P.L : ℝ) + 1) * G := by positivity
    nlinarith only [h12, hLG]
  · intro _
    rw [hNr]
    have : 0 ≤ 12 * θ * ((Fintype.card σ : ℝ) + S + 1 + 2) := by positivity
    have hLG : 0 < ((P.L : ℝ) + 1) * G := by positivity
    nlinarith only [this, hLG]
  · rw [show Fintype.card σ + S + 3 + 1 = Fintype.card σ + S + 4 by ring]
    exact level_le _ K b G δ (by positivity) hK100 hδ0 hδ1 (Nat.cast_nonneg _) hG.symm hbK
  · have : 4 * Real.log (8 / δ) / (S + 1) ≤ m := by linarith
    rw [div_le_iff₀ (by positivity)] at this
    linarith
  · rw [hNr]
    have hY : 0 ≤ 32 * ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α * (P.L + 1) * G
        + n₀ := by positivity
    have h1 : εd * (32 * ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α * (P.L + 1) * G
        + n₀) / (32 * ((Fintype.card σ : ℝ) + S + 1)
          * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)) ≤ m := by linarith
    have e : ((S : ℝ) + 1) * (εd * (32 * ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α
        * (P.L + 1) * G + n₀) / (32 * ((Fintype.card σ : ℝ) + S + 1)
          * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)))
        = 8 * ρ * (32 * ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α * (P.L + 1) * G
          + n₀) := by
      rw [← hρ]; field_simp; ring
    have h2 := mul_le_mul_of_nonneg_left h1 (by positivity : (0 : ℝ) ≤ (S : ℝ) + 1)
    rw [e] at h2
    have h12 : 12 * θ * ((Fintype.card σ : ℝ) + S + 1 + 2) ≤ 1 := by
      have : 0 ≤ θ := hθ
      nlinarith only [hθn, this, hn1]
    have hW : 0 ≤ ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α * (P.L + 1) * G := by
      positivity
    have h3 : 2 * ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α
        * (8 * (P.L + 1) * (1 + 12 * θ * ((Fintype.card σ : ℝ) + S + 1 + 2)) * G) + n₀
        ≤ 32 * ((Fintype.card σ : ℝ) + S + 1 + 2) * Fintype.card α * (P.L + 1) * G + n₀ := by
      nlinarith only [mul_le_mul_of_nonneg_left h12 hW]
    have h4 := mul_le_mul (by linarith : 16 * (2 * O.κ) ≤ 8 * ρ) h3 (by positivity)
      (by positivity)
    linarith
  · have hR : (8 * s + 1 / (((P.L : ℝ) + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ))))
        * (((P.L : ℝ) + 1) * (((Fintype.card σ + S + 3 : ℕ) : ℝ))) + 2 * (1 / 2)
        + 2 * (((Fintype.card σ + S + 3 : ℕ) : ℝ)) * Fintype.card α
          * (4 * s * (P.L + 1) / Fintype.card α)
        = 2 * (8 * ((Fintype.card σ : ℝ) + S + 3) * (P.L + 1) * s + 1) := by
      push_cast
      field_simp
      ring
    rw [hR]
    have := hcap (4 * O.κ * (8 * ((Fintype.card σ : ℝ) + S + 3) * (P.L + 1) * s + 1))
      (by positivity) hκc
    linarith
  · have := hcap (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * O.ε
      + 3 * (prefixMax P.D k * unionLog (α := α) σ S δ)) (by positivity) (by linarith)
    linarith

end Guarantee

theorem level4_correct_holds : Level4Correct := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ εd P δ S k
  exact level4_guarantee M O θ εd P δ S k

theorem level4_family_correct_holds : Level4FamilyCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ fam θ εd P δ S k
  exact level4_guarantee fam.M fam.oracle θ εd P δ S k

end OrthoDFA
