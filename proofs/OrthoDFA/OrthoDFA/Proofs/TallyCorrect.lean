import OrthoDFA.Proofs.TallyRiskBound
import OrthoDFA.TallyFamily

/-!
# The tally round's claim

`TallyRoundAtSettings`: at given settings, from `TallyRoundRisk` at `κs = 2κ`, `TallyRiskBound`
bounding its `roundRisk` by `δ`, and the yield's term at most `δ` once `16 q log(1/δ) ≤ 1.5θ c`;
each of `TallyRiskBound`'s conditions follows from fewer, looser ones.

`TallyRoundCorrect`: the settings chosen as formulas of the problem (`θpt = 1/4`,
`φpt = 3εd/4`, `θs = θe = 3εd/16`, `φe = 1`, `Lmax = |Q| + S + 3`, `a = e^{−G}`,
`n₀ = ⌈1000 G/εd²⌉`, `m = ⌈3100 G/εd²⌉`, `exc = 6(L + 1)(1 + 12θ Lmax) G` and the budget the
records cap asks for), which meet `TallyRoundAtSettings`'s conditions.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `TallyRoundAtSettings M O θ P C δ S`: the round over `C.budget` probes drawn from `P.D` fails
to end well with chance at most `2δ`, once the given settings `C` are wide enough for the laws'
`κ` and `ε`, the goodness level `θ`, `S` splits that are not genuine, `p₀`, the most `P.D` puts on
one length-`k` prefix, `Λ = tallyLog σ S δ` and `lg = log (1 / C.a)`. -/
def TallyRoundAtSettings [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ) (O : TriOracle μ M) (θ : ℝ)
    (P : PrefixDistribution α) (C : TallyCfg) (δ : ℝ) (S : ℕ) : Prop :=
    let n : ℝ := Fintype.card σ + S
    let A : ℝ := Fintype.card α
    let p₀ := prefixMax P.D C.k
    let Λ := tallyLog (α := α) σ S δ
    let lg := Real.log (1 / C.a)
    let bad := fun q => ¬ O.Good θ q
    0 < δ → δ ≤ 1 → 1 ≤ P.L → Fintype.card σ + S + 3 ≤ C.Lmax →
    0 < O.κ → 0 ≤ O.ε → 0 ≤ θ → 8 * (O.κ + O.ε * (n + 1)) * (P.L + 1) ≤ 1 →
    0 < C.φpt → 4 * C.φpt ≤ 3 * C.εd → C.εd ≤ 1 → 4 * C.θpt ≤ 1 → C.θs ≤ 1 → 24 * θ ≤ C.θe →
    256 * (P.L + 1) * ((n + 1) * θ + p₀ * Λ) ≤ min C.θs (min (C.θpt * C.φpt) (C.φe * C.θe)) →
    32 * lg ≤ C.n₀ * min C.θs (min C.θpt C.φpt) ^ 2 →
    (∀ j, 4 * (P.L + 1) * (1 + 12 * θ * C.Lmax) * lg < C.exc j
      ∧ C.exc j ≤ 8 * (P.L + 1) * (1 + 12 * θ * C.Lmax) * lg) →
    0 < C.a → 64 * (2 ^ C.Lmax * A + C.budget ^ 2) * C.a ≤ δ →
    32 * (n + 1) * (C.Lmax * A * (lg / (C.εd ^ 2 * C.θpt ^ 2) + C.n₀ / C.εd + 1)
      + (Fintype.card σ * A + S + 1) * C.m / C.εd) ≤ C.budget →
    4 * Real.log (8 / δ) + 32 * O.κ * (16 * (P.L + 1) * (1 + 12 * θ * C.Lmax) * C.Lmax * A * lg
      + C.n₀ + C.budget * (C.θe * ((P.L + 1) * C.Lmax) + 2 * C.θs + 2 * C.Lmax * A * C.φe))
      + 48 * C.budget * (O.ε * ((P.L + 1) * (n + 1)) + p₀ * Λ) ≤ (S + 1) * C.m →
    16 * P.q * Real.log (1 / δ) ≤ 3 / 2 * θ * coverage P.D P.start P.L M bad →
    ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => P.D)
        {xs | ¬ RunEnds (tallyStep C fun z => (O.read z ω).cut)
          (TallyEndsWell M bad P.D C P.start P.L
            (3 / 4 * θ * coverage P.D P.start P.L M bad / (P.L + 1 - P.start : ℕ)) (O.read · ω))
          tallyStart (List.ofFn xs)} ∂μ
      ≤ ENNReal.ofReal (2 * δ)

set_option maxHeartbeats 1000000 in -- every condition of `TallyRiskBound` from the claim's
theorem tally_round_correct_at [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*}
    [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ] (M : _root_.DFA α σ)
    (O : TriOracle μ M) (θ : ℝ) (P : PrefixDistribution α) (C : TallyCfg) (δ : ℝ) (S : ℕ) :
    TallyRoundAtSettings M O θ P C δ S := by
  dsimp only [TallyRoundAtSettings]
  intro hδ0 hδ1 hL hLmax hκ hε hθ hκεL hφpt hφεd hεd1 hθpt4 hθs1 hθe hmin hn₀ hexc ha0 ha hbud
    hcap hyield
  generalize hp₀ : prefixMax P.D C.k = p₀ at hmin hcap
  generalize hΛ : tallyLog (α := α) σ S δ = Λ at hmin hcap
  generalize hlg : Real.log (1 / C.a) = lg at hn₀ hexc hbud hcap
  have hp0 : 0 < p₀ := hp₀ ▸ prefixMax_pos P.D C.k
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hN0 : (0 : ℝ) ≤ Fintype.card σ + S := by positivity
  have hΛ0 : 0 < Λ := by
    rw [← hΛ]
    unfold tallyLog
    refine mul_pos (mul_pos (by positivity) (by linarith)) (Real.log_pos ?_)
    rw [lt_div_iff₀ hδ0]
    have h3 : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by linarith
    nlinarith only [h3, hA1, hδ1]
  have hpΛ : 0 < p₀ * Λ := mul_pos hp0 hΛ0
  have hL1 : (1 : ℝ) ≤ P.L := by exact_mod_cast hL
  have hb0 : (0 : ℝ) ≤ C.budget := Nat.cast_nonneg _
  have hLmax1 : (1 : ℝ) ≤ C.Lmax := by
    have : 1 ≤ C.Lmax := by omega
    exact_mod_cast this
  have hpow1 : (1 : ℝ) ≤ 2 ^ C.Lmax := one_le_pow₀ (by norm_num)
  have hpowA : (1 : ℝ) ≤ 2 ^ C.Lmax * Fintype.card α := by nlinarith only [hpow1, hA1]
  have ha64 : 64 * C.a ≤ δ := by nlinarith only [sq_nonneg (C.budget : ℝ), ha, hpowA, ha0]
  have ha1 : C.a ≤ 1 := by linarith
  have hlg0 : 0 ≤ lg := by
    rw [← hlg]; exact Real.log_nonneg (by rw [le_div_iff₀ ha0]; linarith)
  -- the thresholds
  obtain ⟨hm1, hm23⟩ := le_min_iff.1 hmin
  obtain ⟨hm2, hm3⟩ := le_min_iff.1 hm23
  have hnθ : 0 ≤ ((Fintype.card σ : ℝ) + S + 1) * θ := mul_nonneg (by positivity) hθ
  have hXθ : 0 ≤ ((P.L : ℝ) + 1) * (((Fintype.card σ : ℝ) + S + 1) * θ) :=
    mul_nonneg (by positivity) hnθ
  have hYp : 0 < ((P.L : ℝ) + 1) * (p₀ * Λ) := mul_pos (by positivity) hpΛ
  have hθs0 : 0 < C.θs := by nlinarith only [hm1, hXθ, hYp]
  have hθφ : 0 < C.θpt * C.φpt := by nlinarith only [hm2, hXθ, hYp]
  have hθpt0 : 0 < C.θpt := pos_of_mul_pos_left hθφ hφpt.le
  have hφeθe : 0 < C.φe * C.θe := by nlinarith only [hm3, hXθ, hYp]
  have hθe0 : 0 ≤ C.θe := by linarith
  have hθe1 : 0 < C.θe := by
    rcases hθe0.lt_or_eq with h | h
    · exact h
    · rw [← h, mul_zero] at hφeθe; exact absurd hφeθe (lt_irrefl 0)
  have hφe0 : 0 < C.φe := pos_of_mul_pos_left hφeθe hθe1.le
  have hφ34 : C.φpt ≤ 3 / 4 := by linarith
  have hθptφ : C.θpt * C.φpt ≤ C.θpt := by nlinarith only [hφ34, hθpt0]
  -- the minimum's square
  set mn := min C.θs (min C.θpt C.φpt)
  have hmn0 : 0 < mn := lt_min hθs0 (lt_min hθpt0 hφpt)
  have hmn_s : mn ≤ C.θs := min_le_left _ _
  have hmn_pt : mn ≤ C.θpt := (min_le_right _ _).trans (min_le_left _ _)
  have hmn_φ : mn ≤ C.φpt := (min_le_right _ _).trans (min_le_right _ _)
  have hn₀0 : (0 : ℝ) ≤ C.n₀ := Nat.cast_nonneg _
  have hsq : ∀ x, mn ≤ x → (C.n₀ : ℝ) * mn ^ 2 ≤ C.n₀ * x ^ 2 := fun x hx =>
    mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hmn0.le hx 2) hn₀0
  -- the records cap's parts
  have hlog8 : 0 < Real.log (8 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ0]; linarith)
  have hT2a : 0 ≤ 16 * ((P.L : ℝ) + 1) * (1 + 12 * θ * C.Lmax) * C.Lmax * Fintype.card α * lg
      + C.n₀ := by
    have : 0 ≤ θ * C.Lmax := mul_nonneg hθ (by positivity)
    have : 0 ≤ 1 + 12 * θ * C.Lmax := by linarith
    positivity
  have hT2b : 0 ≤ (C.budget : ℝ) * (C.θe * ((P.L + 1) * C.Lmax) + 2 * C.θs
      + 2 * C.Lmax * Fintype.card α * C.φe) := by
    have : 0 ≤ C.θe * ((P.L + 1) * C.Lmax) := mul_nonneg hθe0 (by positivity)
    have : 0 ≤ 2 * C.Lmax * Fintype.card α * C.φe := mul_nonneg (by positivity) hφe0.le
    exact mul_nonneg hb0 (by linarith)
  have hεN : 0 ≤ O.ε * (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1)) :=
    mul_nonneg hε (by positivity)
  have hT3 : 0 ≤ 48 * (C.budget : ℝ) * (O.ε * (((P.L : ℝ) + 1)
      * ((Fintype.card σ : ℝ) + S + 1)) + p₀ * Λ) :=
    mul_nonneg (by positivity) (by linarith)
  have hκT2a := mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 32) hκ.le) hT2a
  have hκT2b := mul_nonneg (mul_nonneg (by norm_num : (0 : ℝ) ≤ 32) hκ.le) hT2b
  -- the budget
  set X := lg / (C.εd ^ 2 * C.θpt ^ 2)
  have hεd0 : 0 < C.εd := by linarith
  have hX0 : 0 ≤ X := by positivity
  have hY0 : 0 ≤ (C.n₀ : ℝ) / C.εd := by positivity
  have hlgX : 32 * lg / C.εd ^ 2 ≤ 2 * X := by
    have hθsq : C.θpt ^ 2 ≤ 1 / 16 := by nlinarith only [hθpt4, hθpt0]
    have : lg / C.εd ^ 2 = X * C.θpt ^ 2 := by simp only [X]; field_simp
    rw [mul_div_assoc, this]
    nlinarith only [mul_le_mul_of_nonneg_left hθsq hX0]
  have hLA : (1 : ℝ) ≤ C.Lmax * Fintype.card α := by nlinarith only [hLmax1, hA1]
  have hbudget : ((S : ℝ) + Fintype.card σ + 1) * ((2 * C.Lmax * Fintype.card α + 1)
      * (8 * lg / (C.εd ^ 2 * C.θpt ^ 2) + 4 * C.n₀ / (3 * C.εd) + 2)
      + 8 * (Fintype.card σ * Fintype.card α + S + 1) * C.m / C.εd + 32 * lg / C.εd ^ 2 + 1)
      ≤ C.budget := by
    refine le_trans ?_ hbud
    have h8 : 8 * lg / (C.εd ^ 2 * C.θpt ^ 2) = 8 * X := by simp only [X]; ring
    have h4 : 4 * (C.n₀ : ℝ) / (3 * C.εd) = 4 / 3 * (C.n₀ / C.εd) := by field_simp
    have hZ : 8 * ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * C.m / C.εd
        = 8 * (((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * C.m / C.εd) := by ring
    have hZ0 : 0 ≤ ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * C.m / C.εd := by positivity
    rw [h8, h4, hZ]
    have hin : (2 * C.Lmax * Fintype.card α + 1 : ℝ) * (8 * X + 4 / 3 * (C.n₀ / C.εd) + 2)
        ≤ 3 * (C.Lmax * Fintype.card α) * (8 * X + 2 * (C.n₀ / C.εd) + 2) := by
      have hm := mul_nonneg (by linarith : (0 : ℝ) ≤ C.Lmax * Fintype.card α - 1)
        (by positivity : (0 : ℝ) ≤ 8 * X + 2 * (C.n₀ / C.εd) + 2)
      nlinarith only [hLA, hX0, hY0, hm]
    have hP0 : (0 : ℝ) ≤ (S : ℝ) + Fintype.card σ + 1 := by positivity
    have hXl : 2 * X ≤ 2 * (C.Lmax * Fintype.card α) * X := by nlinarith only [hLA, hX0]
    have hsum : (2 * C.Lmax * Fintype.card α + 1 : ℝ) * (8 * X + 4 / 3 * (C.n₀ / C.εd) + 2)
        + 8 * (((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * C.m / C.εd)
        + 32 * lg / C.εd ^ 2 + 1
        ≤ 32 * (C.Lmax * Fintype.card α * (X + C.n₀ / C.εd + 1)
          + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * C.m / C.εd) := by
      nlinarith only [hin, hlgX, hXl, hLA, hZ0, hX0, hY0]
    have := mul_le_mul_of_nonneg_left hsum hP0
    linarith
  -- the round's risk
  have hRB := tally_risk_bound_holds σ O.κ O.ε θ P.D P.L C (2 * O.κ) δ S
  dsimp only at hRB
  rw [hp₀, hΛ, hlg] at hRB
  have hRB' := hRB hδ0 hδ1 hL hLmax hκ le_rfl
    (by nlinarith only [hκεL, hεN, hκ, hL1, mul_nonneg hε (by positivity :
      (0 : ℝ) ≤ ((Fintype.card σ : ℝ) + S + 1) * ((P.L : ℝ) + 1))])
    hε
    (by nlinarith only [hκεL, hκ, hL1, mul_nonneg hε (by positivity :
      (0 : ℝ) ≤ ((Fintype.card σ : ℝ) + S + 1) * ((P.L : ℝ) + 1))])
    hθ hθe
    (by nlinarith only [hm2, hYp, hθptφ, hXθ])
    (by nlinarith only [hm1, hYp, hXθ, hL1, hnθ, hpΛ])
    hθs1
    (by nlinarith only [hm3, hXθ, hYp])
    (by nlinarith only [hm2, hXθ, hL1, hpΛ])
    hεd1 (by linarith) (by linarith)
    (by nlinarith only [hn₀, hsq _ hmn_s])
    (by nlinarith only [hn₀, hsq _ hmn_pt])
    (by nlinarith only [hn₀, hsq _ hmn_φ, hlg0])
    (fun j => (hexc j).1) (fun j => (hexc j).2) ha0
    (by
      have : (2 : ℝ) ^ (C.Lmax + 1) = 2 * 2 ^ C.Lmax := by rw [pow_succ]; ring
      rw [this]
      nlinarith only [ha, ha0, hpowA, sq_nonneg (C.budget : ℝ), hA1, hpow1])
    hbudget
    (by nlinarith only [hcap, hκT2a, hκT2b, hT3])
    (by nlinarith only [hcap, hκT2a, hκT2b, hT3, hlog8])
    (by nlinarith only [hcap, hκT2a, hκT2b, hT3, hlog8])
    (by nlinarith only [hcap, hκT2a, hκT2b, hεN, hpΛ, hb0, hlog8])
  -- the yield's term
  have hexp : Real.exp (-(3 / 2 * θ * coverage P.D P.start P.L M (fun q => ¬ O.Good θ q))
      / (16 * P.q)) ≤ δ := by
    have hq := P.q_pos
    have hlogδ : Real.log δ = -Real.log (1 / δ) := by rw [one_div, Real.log_inv, neg_neg]
    rw [← Real.exp_log hδ0, hlogδ]
    refine Real.exp_le_exp.2 ?_
    rw [neg_div, neg_le_neg_iff, le_div_iff₀ (by positivity)]
    linarith
  calc _ ≤ roundRisk σ O.κ O.ε θ P.D P.L C (2 * O.κ)
        + ENNReal.ofReal (Real.exp
          (-(3 / 2 * θ * coverage P.D P.start P.L M (fun q => ¬ O.Good θ q)) / (16 * P.q))) :=
        tally_round_holds M O θ P C (2 * O.κ) le_rfl
    _ ≤ ENNReal.ofReal δ + ENNReal.ofReal δ := add_le_add hRB' (ENNReal.ofReal_le_ofReal hexp)
    _ = ENNReal.ofReal (2 * δ) := by rw [← ENNReal.ofReal_add hδ0.le hδ0.le]; ring_nf

section Guarantee

variable [Nonempty α] {σ : Type*} [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ]

/-- The chosen budget is at most `K G`. -/
theorem budget_le (n1 A N Z G u n₀ m : ℝ) (hn1 : 1 ≤ n1) (hA : 1 ≤ A) (hN : N ≤ 3 * n1)
    (hZ : Z ≤ n1 * A) (hZ0 : 0 ≤ Z) (hG : 1 ≤ G) (hu : 1 ≤ u)
    (hn₀ : n₀ ≤ 1000 * G * u ^ 2 + 1) (hn₀0 : 0 ≤ n₀) (hm : m ≤ 3100 * G * u ^ 2 + 1)
    (hm0 : 0 ≤ m) :
    32 * n1 * (N * A * (16 * G * u ^ 2 + n₀ * u + 1) + Z * m * u) + 1
      ≤ 2 ^ 18 * n1 ^ 2 * A * u ^ 3 * G := by
  have hu2 : u ^ 2 ≤ u ^ 3 := pow_le_pow_right₀ hu (by norm_num)
  have hu1 : u ≤ u ^ 3 := by nlinarith
  have hu3 : 1 ≤ u ^ 3 := one_le_pow₀ hu
  have hGu : 1 ≤ G * u ^ 2 := by nlinarith [one_le_pow₀ (n := 2) hu]
  set W := n1 ^ 2 * A * u ^ 3 * G
  have hW1 : 1 ≤ W := by
    have : 1 ≤ n1 ^ 2 := one_le_pow₀ hn1
    have : 1 ≤ n1 ^ 2 * A := by nlinarith
    have : 1 ≤ u ^ 3 * G := by nlinarith
    calc (1 : ℝ) ≤ (n1 ^ 2 * A) * (u ^ 3 * G) := by nlinarith
      _ = W := by ring
  have hNA : N * A ≤ 3 * (n1 * A) := by nlinarith
  have hn1A : 0 ≤ n1 * A := by positivity
  -- each term against W
  have t1 : 32 * n1 * (N * A * (16 * G * u ^ 2)) ≤ 1536 * W := by
    have : 16 * G * u ^ 2 ≤ 16 * G * u ^ 3 := by nlinarith
    have h1 : N * A * (16 * G * u ^ 2) ≤ 3 * (n1 * A) * (16 * G * u ^ 3) :=
      mul_le_mul hNA this (by positivity) (by positivity)
    have : 32 * n1 * (N * A * (16 * G * u ^ 2)) ≤ 32 * n1 * (3 * (n1 * A) * (16 * G * u ^ 3)) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    simp only [W]; nlinarith
  have t2 : 32 * n1 * (N * A * (n₀ * u)) ≤ 96096 * W := by
    have h0 : n₀ * u ≤ 1001 * G * u ^ 3 := by
      have : n₀ * u ≤ (1000 * G * u ^ 2 + 1) * u := mul_le_mul_of_nonneg_right hn₀ (by linarith)
      nlinarith
    have h1 : N * A * (n₀ * u) ≤ 3 * (n1 * A) * (1001 * G * u ^ 3) :=
      mul_le_mul hNA h0 (by positivity) (by positivity)
    have : 32 * n1 * (N * A * (n₀ * u)) ≤ 32 * n1 * (3 * (n1 * A) * (1001 * G * u ^ 3)) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    simp only [W]; nlinarith
  have t3 : 32 * n1 * (N * A * 1) ≤ 96 * W := by
    have h1 : N * A * 1 ≤ 3 * (n1 * A) * (G * u ^ 3) := by
      have : 1 ≤ G * u ^ 3 := by nlinarith
      nlinarith
    have : 32 * n1 * (N * A * 1) ≤ 32 * n1 * (3 * (n1 * A) * (G * u ^ 3)) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    simp only [W]; nlinarith
  have t4 : 32 * n1 * (Z * m * u) ≤ 99232 * W := by
    have h0 : m * u ≤ 3101 * G * u ^ 3 := by
      have : m * u ≤ (3100 * G * u ^ 2 + 1) * u := mul_le_mul_of_nonneg_right hm (by linarith)
      nlinarith
    have h1 : Z * (m * u) ≤ (n1 * A) * (3101 * G * u ^ 3) :=
      mul_le_mul hZ h0 (by positivity) (by positivity)
    have : 32 * n1 * (Z * (m * u)) ≤ 32 * n1 * ((n1 * A) * (3101 * G * u ^ 3)) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    simp only [W]; nlinarith
  have hsplit : 32 * n1 * (N * A * (16 * G * u ^ 2 + n₀ * u + 1) + Z * m * u)
      = 32 * n1 * (N * A * (16 * G * u ^ 2)) + 32 * n1 * (N * A * (n₀ * u))
        + 32 * n1 * (N * A * 1) + 32 * n1 * (Z * m * u) := by ring
  rw [hsplit]
  have : (2 : ℝ) ^ 18 * n1 ^ 2 * A * u ^ 3 * G = 262144 * W := by simp only [W]; ring
  rw [this]
  linarith

/-- The records cap at the chosen settings. -/
theorem cap_le (n1 A L1 N Z G u εd κ ε θ pl b n₀ m S1 lo : ℝ) (hn1 : 1 ≤ n1) (hA : 1 ≤ A)
    (hL1 : 1 ≤ L1) (hN1 : 1 ≤ N) (hN : N ≤ 3 * n1) (hZ : Z ≤ n1 * A) (hZ0 : 0 ≤ Z) (hG : 1 ≤ G)
    (hu : 1 ≤ u) (hεd0 : 0 < εd) (hεd1 : εd ≤ 1) (hεu : εd * u = 1) (hκ : 0 ≤ κ) (hε : 0 ≤ ε)
    (hθ : 0 ≤ θ) (hpl : 0 ≤ pl) (hn₀0 : 0 ≤ n₀) (hS1 : 1 ≤ S1) (hm0 : 0 ≤ m)
    (hθc : 2 ^ 13 * L1 * n1 * θ ≤ εd) (hκc : 2 ^ 14 * n1 ^ 2 * A ^ 2 * L1 * N * κ ≤ εd)
    (hεc : 2 ^ 13 * L1 * n1 ^ 2 * A * (ε * n1 + pl) ≤ εd)
    (hb : b ≤ 32 * n1 * (N * A * (16 * G * u ^ 2 + n₀ * u + 1) + Z * m * u) + 1)
    (hn₀ : n₀ ≤ 1000 * G * u ^ 2 + 1) (hm : 3100 * G * u ^ 2 ≤ m) (hlo : 4 * lo ≤ 2 * G) :
    4 * lo + 32 * κ * (16 * L1 * (1 + 12 * θ * N) * N * A * G + n₀
      + b * (3 * εd / 16 * (L1 * N) + 2 * (3 * εd / 16) + 2 * N * A * 1))
      + 48 * b * (ε * (L1 * n1) + pl) ≤ S1 * m := by
  have hn1sq : 1 ≤ n1 ^ 2 := one_le_pow₀ hn1
  have hQ : 1 ≤ n1 ^ 2 * A := by nlinarith only [hn1sq, hA]
  have hu2 : 1 ≤ u ^ 2 := one_le_pow₀ hu
  have hGu : 1 ≤ G * u ^ 2 := by nlinarith only [hG, hu2]
  have huu : u ≤ u ^ 2 := by nlinarith only [hu]
  have hGu' : G * u ≤ G * u ^ 2 := mul_le_mul_of_nonneg_left huu (by linarith)
  have hGG : G ≤ G * u ^ 2 := by nlinarith only [hG, hu2]
  have hLN : 1 ≤ L1 * N := by nlinarith only [hL1, hN1]
  have hLNA : 1 ≤ L1 * N * A := by nlinarith only [hLN, hA]
  have h12 : 12 * θ * N ≤ 1 := by
    have h1 : 12 * θ * N ≤ 36 * n1 * θ := by nlinarith only [hθ, hN]
    have h2 : 36 * n1 * θ ≤ 2 ^ 13 * L1 * n1 * θ := by
      have : 0 ≤ n1 * θ := by positivity
      nlinarith only [this, hL1]
    linarith
  -- κ against its budget, through κ' = κ L1 N A
  obtain ⟨κ', hκ'd⟩ : ∃ x, x = κ * L1 * N * A := ⟨_, rfl⟩
  have hκ'0 : 0 ≤ κ' := by rw [hκ'd]; positivity
  have hκ' : 2 ^ 14 * (n1 ^ 2 * A) * κ' ≤ εd := by
    have : 2 ^ 14 * (n1 ^ 2 * A) * κ' = 2 ^ 14 * n1 ^ 2 * A ^ 2 * L1 * N * κ := by
      rw [hκ'd]; ring
    linarith
  have hκ'1 : 2 ^ 14 * κ' ≤ 1 := by nlinarith only [hκ', hQ, hκ'0, hεd1]
  have hκκ' : κ ≤ κ' := by
    rw [hκ'd]
    have : κ * 1 ≤ κ * (L1 * N * A) := mul_le_mul_of_nonneg_left hLNA hκ
    nlinarith only [this]
  have s3 : 32 * κ * (16 * L1 * (1 + 12 * θ * N) * N * A * G) ≤ G / 16 := by
    have e : 32 * κ * (16 * L1 * (1 + 12 * θ * N) * N * A * G)
        = 512 * (1 + 12 * θ * N) * (κ' * G) := by rw [hκ'd]; ring
    rw [e]
    have hκG : 0 ≤ κ' * G := by positivity
    have : 512 * (1 + 12 * θ * N) * (κ' * G) ≤ 1024 * (κ' * G) := by nlinarith only [h12, hκG]
    have : 1024 * (κ' * G) ≤ G / 16 := by nlinarith only [hκ'1, hG, hκ'0]
    linarith
  have s4 : 32 * κ * n₀ ≤ n₀ / 512 := by nlinarith only [hκκ', hκ'1, hn₀0, hκ]
  -- the cap's share of the budget
  obtain ⟨γ, hγd⟩ : ∃ x, x = 32 * κ * (3 * εd / 16 * (L1 * N) + 2 * (3 * εd / 16)
      + 2 * N * A * 1) + 48 * (ε * (L1 * n1) + pl) := ⟨_, rfl⟩
  have hX' : 3 * εd / 16 * (L1 * N) + 2 * (3 * εd / 16) + 2 * N * A * 1 ≤ 4 * (L1 * N * A) := by
    have h1 : L1 * N ≤ L1 * N * A := by nlinarith only [hLN, hA]
    have h3 : N * A ≤ L1 * N * A := by
      have : 0 ≤ N * A := by positivity
      nlinarith only [this, hL1]
    nlinarith only [h1, h3, hLNA, hεd1, hεd0, hLN]
  have hγ1 : 32 * κ * (3 * εd / 16 * (L1 * N) + 2 * (3 * εd / 16) + 2 * N * A * 1)
      ≤ 128 * κ' := by
    have := mul_le_mul_of_nonneg_left hX' (by positivity : 0 ≤ 32 * κ)
    rw [hκ'd]; nlinarith only [this]
  have hγ2 : 48 * (ε * (L1 * n1) + pl) ≤ 48 * (L1 * (ε * n1 + pl)) := by
    nlinarith only [hpl, hL1]
  have hγ : 64 * (n1 ^ 2 * A) * γ ≤ εd := by
    have hγle : γ ≤ 128 * κ' + 48 * (L1 * (ε * n1 + pl)) := by rw [hγd]; linarith
    have e2 : 2 ^ 13 * (n1 ^ 2 * A) * (L1 * (ε * n1 + pl)) ≤ εd := by
      have : 2 ^ 13 * (n1 ^ 2 * A) * (L1 * (ε * n1 + pl))
          = 2 ^ 13 * L1 * n1 ^ 2 * A * (ε * n1 + pl) := by ring
      linarith
    have := mul_le_mul_of_nonneg_left hγle (by positivity : 0 ≤ 64 * (n1 ^ 2 * A))
    nlinarith only [this, hκ', e2, hεd0]
  have hγ0 : 0 ≤ γ := by
    have : 0 ≤ 3 * εd / 16 * (L1 * N) + 2 * (3 * εd / 16) + 2 * N * A * 1 := by positivity
    rw [hγd]; positivity
  -- b γ
  obtain ⟨X, hXd⟩ : ∃ x, x = 32 * n1 * (N * A * (16 * G * u ^ 2 + n₀ * u + 1) + Z * m * u) :=
    ⟨_, rfl⟩
  rw [← hXd] at hb
  have hX0 : 0 ≤ X := by rw [hXd]; positivity
  have hbγ : b * γ ≤ m / 2 + 3 / 2 * (16 * G * u + n₀ + 1) + 1 / 64 := by
    have h1 : b * γ ≤ (X + 1) * γ := mul_le_mul_of_nonneg_right hb hγ0
    have h2 : (X + 1) * γ ≤ (X + 1) * (εd / (64 * (n1 ^ 2 * A))) := by
      refine mul_le_mul_of_nonneg_left ?_ (by positivity)
      rw [le_div_iff₀ (by positivity)]
      linarith
    have h3 : (X + 1) * (εd / (64 * (n1 ^ 2 * A)))
        = (N * A * (16 * G * u + n₀ + εd) + Z * m) / (2 * n1 * A) + εd / (64 * (n1 ^ 2 * A)) := by
      have hn1A : n1 * A ≠ 0 := by positivity
      have hu' : u = 1 / εd := by field_simp; linarith
      rw [hXd, hu']
      field_simp
      ring
    have h4 : (N * A * (16 * G * u + n₀ + εd) + Z * m) / (2 * n1 * A)
        ≤ 3 / 2 * (16 * G * u + n₀ + 1) + m / 2 := by
      rw [div_le_iff₀ (by positivity)]
      have hNA : N * A ≤ 3 * n1 * A := by nlinarith only [hN, hA]
      have hp1 : 16 * G * u + n₀ + εd ≤ 16 * G * u + n₀ + 1 := by linarith
      have hm1 : N * A * (16 * G * u + n₀ + εd) ≤ 3 * n1 * A * (16 * G * u + n₀ + 1) :=
        mul_le_mul hNA hp1 (by positivity) (by positivity)
      have hm2 : Z * m ≤ n1 * A * m := mul_le_mul_of_nonneg_right hZ hm0
      nlinarith only [hm1, hm2]
    have h5 : εd / (64 * (n1 ^ 2 * A)) ≤ 1 / 64 := by
      rw [div_le_iff₀ (by positivity)]; nlinarith only [hQ, hεd1]
    linarith
  have hfin : 4 * lo + 32 * κ * (16 * L1 * (1 + 12 * θ * N) * N * A * G) + 32 * κ * n₀
      + b * γ ≤ m := by
    have : 16 * G * u ≤ 16 * (G * u ^ 2) := by linarith
    have : n₀ / 512 + 3 / 2 * n₀ ≤ 1502 * (G * u ^ 2) + 2 := by nlinarith only [hn₀, hn₀0]
    linarith
  have hexp : 4 * lo + 32 * κ * (16 * L1 * (1 + 12 * θ * N) * N * A * G + n₀
      + b * (3 * εd / 16 * (L1 * N) + 2 * (3 * εd / 16) + 2 * N * A * 1))
      + 48 * b * (ε * (L1 * n1) + pl)
      = 4 * lo + 32 * κ * (16 * L1 * (1 + 12 * θ * N) * N * A * G) + 32 * κ * n₀ + b * γ := by
    rw [hγd]; ring
  rw [hexp]
  nlinarith only [hfin, hS1, hm0]

set_option maxHeartbeats 4000000 in -- the chosen settings meet every condition
theorem tally_round_guarantee (M : _root_.DFA α σ) (O : TriOracle μ M) (θ εd : ℝ)
    (P : PrefixDistribution α) (δ : ℝ) (S k : ℕ) : TallyRoundGuarantee M O θ εd P δ S k := by
  dsimp only [TallyRoundGuarantee]
  intro hδ0 hδ1 hεd0 hεd1 hL hθ hθc hκ hκc hε hεc hyield
  generalize hK : (2 : ℝ) ^ 18 * ((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α / εd ^ 3
    = K
  generalize hG : 2 * Real.log (64 * (2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ)
    + 8 * K ^ 2) / δ) = G
  -- the problem's sizes
  have hA1 : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have hn1 : (1 : ℝ) ≤ Fintype.card σ + S + 1 := by
    have : (0 : ℝ) ≤ Fintype.card σ + S := by positivity
    linarith
  have hL1 : (1 : ℝ) ≤ P.L := by exact_mod_cast hL
  have hNr : (((Fintype.card σ + S + 3 : ℕ) : ℝ)) = Fintype.card σ + S + 3 := by push_cast; ring
  have hN3 : (((Fintype.card σ + S + 3 : ℕ) : ℝ)) ≤ 3 * ((Fintype.card σ : ℝ) + S + 1) := by
    rw [hNr]; have : (0 : ℝ) ≤ Fintype.card σ + S := by positivity
    linarith
  have hN1 : (1 : ℝ) ≤ (((Fintype.card σ + S + 3 : ℕ) : ℝ)) := by rw [hNr]; linarith
  have hZ : (Fintype.card σ : ℝ) * Fintype.card α + S + 1
      ≤ ((Fintype.card σ : ℝ) + S + 1) * Fintype.card α := by
    have : (0 : ℝ) ≤ S := by positivity
    nlinarith only [this, hA1]
  have hZ0 : (0 : ℝ) ≤ (Fintype.card σ : ℝ) * Fintype.card α + S + 1 := by positivity
  -- u = 1 / εd
  obtain ⟨u, hud⟩ : ∃ x : ℝ, x = 1 / εd := ⟨_, rfl⟩
  have hεu : εd * u = 1 := by rw [hud]; field_simp
  have hu1 : 1 ≤ u := by rw [hud, le_div_iff₀ hεd0]; linarith
  have hK' : K = 2 ^ 18 * ((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α * u ^ 3 := by
    rw [← hK, hud]; field_simp
  have hK0 : 0 ≤ K := by rw [hK']; positivity
  -- G
  set X0 := 64 * (2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + 8 * K ^ 2) / δ
  have hpow : (8 : ℝ) ≤ 2 ^ (Fintype.card σ + S + 3) * Fintype.card α := by
    have : (8 : ℝ) ≤ 2 ^ (Fintype.card σ + S + 3) := by
      calc (8 : ℝ) = 2 ^ 3 := by norm_num
        _ ≤ 2 ^ (Fintype.card σ + S + 3) := pow_le_pow_right₀ (by norm_num) (by omega)
    nlinarith only [this, hA1]
  have hX0 : 512 / δ ≤ X0 := by
    simp only [X0]
    gcongr
    nlinarith only [hpow, sq_nonneg K]
  have hX0pos : 0 < X0 := lt_of_lt_of_le (by positivity) hX0
  have hX0e : 3 ≤ X0 := by
    have : 512 ≤ 512 / δ := by rw [le_div_iff₀ hδ0]; nlinarith only [hδ1]
    linarith
  have hexpG : Real.exp (G / 2) = X0 := by
    rw [← hG, show 2 * Real.log X0 / 2 = Real.log X0 by ring, Real.exp_log hX0pos]
  have hG1 : 1 ≤ G := by
    rw [← hG]
    have : 1 ≤ Real.log X0 := by
      rw [Real.le_log_iff_exp_le hX0pos]
      linarith [Real.exp_one_lt_three]
    linarith
  have hlo : 4 * Real.log (8 / δ) ≤ 2 * G := by
    rw [← hG]
    have : Real.log (8 / δ) ≤ Real.log X0 :=
      Real.log_le_log (by positivity) (le_trans (by gcongr; norm_num) hX0)
    linarith
  have hlogG : Real.log (1 / Real.exp (-G)) = G := by
    rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
  -- the settings
  obtain ⟨n₀, hn₀d⟩ : ∃ x : ℕ, x = ⌈1000 * G / εd ^ 2⌉₊ := ⟨_, rfl⟩
  obtain ⟨m, hmd⟩ : ∃ x : ℕ, x = ⌈3100 * G / εd ^ 2⌉₊ := ⟨_, rfl⟩
  have hGu2 : ∀ c : ℝ, c * G / εd ^ 2 = c * G * u ^ 2 := by
    intro c; rw [hud]; field_simp
  have hn₀ : (n₀ : ℝ) ≤ 1000 * G * u ^ 2 + 1 := by
    rw [hn₀d, ← hGu2]; exact (Nat.ceil_lt_add_one (by positivity)).le
  have hn₀l : 1000 * G * u ^ 2 ≤ (n₀ : ℝ) := by rw [hn₀d, ← hGu2]; exact Nat.le_ceil _
  have hm : (m : ℝ) ≤ 3100 * G * u ^ 2 + 1 := by
    rw [hmd, ← hGu2]; exact (Nat.ceil_lt_add_one (by positivity)).le
  have hml : 3100 * G * u ^ 2 ≤ (m : ℝ) := by rw [hmd, ← hGu2]; exact Nat.le_ceil _
  obtain ⟨b, hbd⟩ : ∃ x : ℕ, x = ⌈32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * (Fintype.card α : ℝ)
        * (Real.log (1 / Real.exp (-G)) / (εd ^ 2 * (1 / 4) ^ 2) + (n₀ : ℝ) / εd + 1)
      + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd)⌉₊ := ⟨_, rfl⟩
  have hbX : (b : ℝ) ≤ 32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * Fintype.card α
        * (16 * G * u ^ 2 + n₀ * u + 1) + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)
          * m * u) + 1 := by
    have h := Nat.ceil_lt_add_one (show (0 : ℝ) ≤ 32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * (Fintype.card α : ℝ)
        * (Real.log (1 / Real.exp (-G)) / (εd ^ 2 * (1 / 4) ^ 2) + (n₀ : ℝ) / εd + 1)
      + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd) by
        rw [hlogG]; positivity)
    rw [← hbd] at h
    have e : 32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * (Fintype.card α : ℝ)
        * (Real.log (1 / Real.exp (-G)) / (εd ^ 2 * (1 / 4) ^ 2) + (n₀ : ℝ) / εd + 1)
      + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd)
        = 32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * Fintype.card α
        * (16 * G * u ^ 2 + n₀ * u + 1) + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1)
          * m * u) := by
      rw [hlogG, hud]; field_simp; ring
    rw [e] at h
    exact h.le
  have hbK : (b : ℝ) ≤ K * G := by
    refine hbX.trans ?_
    rw [hK']
    have := budget_le ((Fintype.card σ : ℝ) + S + 1) (Fintype.card α)
      (((Fintype.card σ + S + 3 : ℕ) : ℝ)) ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) G u
      n₀ m hn1 hA1 hN3 hZ hZ0 hG1 hu1 hn₀ (Nat.cast_nonneg _) hm (Nat.cast_nonneg _)
    linarith
  refine ⟨TallyCfg.mk k m (Fintype.card σ + S + 3) (3 * εd / 16) (3 * εd / 16) (1 / 4) εd
    (Real.exp (-G)) n₀
    (fun _ => 6 * ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G)
    1 (3 * εd / 4) b, rfl, rfl, hbK, ?_⟩
  have h := tally_round_correct_at M O θ P (TallyCfg.mk k m (Fintype.card σ + S + 3)
    (3 * εd / 16) (3 * εd / 16) (1 / 4) εd (Real.exp (-G)) n₀
    (fun _ => 6 * ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G)
    1 (3 * εd / 4) b) δ S
  dsimp only [TallyRoundAtSettings] at h
  rw [hlogG] at h
  -- shared facts
  have hp₀ : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  have hΛ0 : 0 ≤ tallyLog (α := α) σ S δ := by
    unfold tallyLog
    refine mul_nonneg (by positivity) (Real.log_nonneg ?_)
    rw [le_div_iff₀ hδ0]
    have h3 : (3 : ℝ) ≤ Fintype.card σ + S + 3 := by
      have : (0 : ℝ) ≤ Fintype.card σ + S := by positivity
      linarith
    nlinarith only [h3, hA1, hδ1, hδ0]
  have hpl : 0 ≤ prefixMax P.D k * tallyLog (α := α) σ S δ := mul_nonneg hp₀ hΛ0
  have hn1sq : (1 : ℝ) ≤ ((Fintype.card σ : ℝ) + S + 1) ^ 2 := one_le_pow₀ hn1
  have hQ : (1 : ℝ) ≤ ((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α := by
    nlinarith only [hn1sq, hA1]
  have hL1' : (1 : ℝ) ≤ (P.L : ℝ) + 1 := by linarith
  -- (θ, p₀Λ) in the thresholds
  have hθ1 : 256 * ((P.L : ℝ) + 1) * (((Fintype.card σ : ℝ) + S + 1) * θ) ≤ εd / 32 := by
    have : 2 ^ 13 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * θ ≤ εd := by
      have e : (2 : ℝ) ^ 13 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * θ
          = 2 ^ 13 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) * θ := rfl
      exact hθc
    have hεd' : 0 ≤ εd := hεd0.le
    nlinarith only [this, hεd']
  have hpl1 : 256 * ((P.L : ℝ) + 1) * (prefixMax P.D k * tallyLog (α := α) σ S δ) ≤ εd / 32 := by
    have hc : 2 ^ 13 * ((P.L : ℝ) + 1) * (prefixMax P.D k * tallyLog (α := α) σ S δ) ≤ εd := by
      have h1 : ((P.L : ℝ) + 1) * (prefixMax P.D k * tallyLog (α := α) σ S δ)
          ≤ ((P.L : ℝ) + 1) * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α
            * (O.ε * ((Fintype.card σ : ℝ) + S + 1)
              + tallyLog (α := α) σ S δ * prefixMax P.D k)) := by
        refine mul_le_mul_of_nonneg_left ?_ (by positivity)
        have : 0 ≤ O.ε * ((Fintype.card σ : ℝ) + S + 1) := mul_nonneg hε (by positivity)
        have hp2 : prefixMax P.D k * tallyLog (α := α) σ S δ * 1
            ≤ prefixMax P.D k * tallyLog (α := α) σ S δ
              * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α) :=
          mul_le_mul_of_nonneg_left hQ hpl
        nlinarith only [hp2, this, hQ]
      have hεc2 : 2 ^ 13 * (((P.L : ℝ) + 1) * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α
            * (O.ε * ((Fintype.card σ : ℝ) + S + 1)
              + tallyLog (α := α) σ S δ * prefixMax P.D k))) ≤ εd := by
        have : (2 : ℝ) ^ 13 * (((P.L : ℝ) + 1) * (((Fintype.card σ : ℝ) + S + 1) ^ 2
            * Fintype.card α * (O.ε * ((Fintype.card σ : ℝ) + S + 1)
              + tallyLog (α := α) σ S δ * prefixMax P.D k)))
            = 2 ^ 13 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α
              * (O.ε * ((Fintype.card σ : ℝ) + S + 1)
                + tallyLog (α := α) σ S δ * prefixMax P.D k) := by ring
        linarith
      nlinarith only [h1, hεc2]
    have hεd' : 0 ≤ εd := hεd0.le
    nlinarith only [hc, hεd']
  -- the level a
  have hE2 : Real.exp (G / 2) ^ 2 * Real.exp (-G) = 1 := by
    rw [← Real.exp_nat_mul, ← Real.exp_add]
    rw [show ((2 : ℕ) : ℝ) * (G / 2) + -G = 0 by push_cast; ring, Real.exp_zero]
  have hGsq : G ^ 2 ≤ 8 * Real.exp (G / 2) := by
    have := Real.quadratic_le_exp_of_nonneg (show 0 ≤ G / 2 by linarith)
    nlinarith only [this, hG1]
  have he1 : 1 ≤ Real.exp (G / 2) := Real.one_le_exp (by linarith)
  have hb0 : (0 : ℝ) ≤ b := Nat.cast_nonneg _
  have hbsq : (b : ℝ) ^ 2 ≤ K ^ 2 * G ^ 2 := by
    have := mul_le_mul hbK hbK hb0 (by positivity)
    nlinarith only [this]
  have hlevel : 64 * (2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + (b : ℝ) ^ 2)
      * Real.exp (-G) ≤ δ := by
    have h1 : 2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + (b : ℝ) ^ 2
        ≤ (2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + 8 * K ^ 2) * Real.exp (G / 2) := by
      have : (b : ℝ) ^ 2 ≤ 8 * K ^ 2 * Real.exp (G / 2) := by
        have := mul_le_mul_of_nonneg_left hGsq (sq_nonneg K)
        nlinarith only [this, hbsq]
      have hp : 0 ≤ 2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) := by positivity
      nlinarith only [this, hp, he1, mul_nonneg hp (sub_nonneg.2 he1)]
    have hX : 64 * (2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + 8 * K ^ 2)
        = δ * Real.exp (G / 2) := by
      rw [hexpG]; simp only [X0]; field_simp
    have hepos : 0 < Real.exp (-G) := Real.exp_pos _
    calc 64 * (2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + (b : ℝ) ^ 2)
          * Real.exp (-G)
        ≤ 64 * ((2 ^ (Fintype.card σ + S + 3) * (Fintype.card α : ℝ) + 8 * K ^ 2)
          * Real.exp (G / 2)) * Real.exp (-G) := by gcongr
      _ = δ * (Real.exp (G / 2) ^ 2 * Real.exp (-G)) := by rw [← mul_assoc, hX]; ring
      _ = δ := by rw [hE2, mul_one]
  -- the budget as chosen
  have hbge : 32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * (Fintype.card α : ℝ)
        * (G / (εd ^ 2 * (1 / 4) ^ 2) + (n₀ : ℝ) / εd + 1)
      + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd) ≤ (b : ℝ) := by
    have := Nat.le_ceil (32 * ((Fintype.card σ : ℝ) + S + 1)
      * ((((Fintype.card σ + S + 3 : ℕ) : ℝ)) * (Fintype.card α : ℝ)
        * (Real.log (1 / Real.exp (-G)) / (εd ^ 2 * (1 / 4) ^ 2) + (n₀ : ℝ) / εd + 1)
      + ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) * (m : ℝ) / εd))
    rw [← hbd, hlogG] at this
    exact this
  -- the minima
  have hmin3 : min (3 * εd / 16) (min (1 / 4) (3 * εd / 4)) = 3 * εd / 16 :=
    min_eq_left (le_min (by linarith) (by linarith))
  have hεu2 : εd ^ 2 * u ^ 2 = 1 := by rw [← mul_pow, hεu]; norm_num
  have hn₀c : 32 * G ≤ (n₀ : ℝ) * min (3 * εd / 16) (min (1 / 4) (3 * εd / 4)) ^ 2 := by
    rw [hmin3]
    have : (n₀ : ℝ) * (3 * εd / 16) ^ 2 ≥ 1000 * G * u ^ 2 * (3 * εd / 16) ^ 2 :=
      mul_le_mul_of_nonneg_right hn₀l (sq_nonneg _)
    have e : 1000 * G * u ^ 2 * (3 * εd / 16) ^ 2 = 9000 / 256 * G * (εd ^ 2 * u ^ 2) := by ring
    rw [e, hεu2] at this
    linarith
  -- the exception counts
  have hexc : ∀ _j : ℕ, 4 * ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G
      < 6 * ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G
      ∧ 6 * ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G
      ≤ 8 * ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G := by
    intro _
    have : 0 < ((P.L : ℝ) + 1) * (1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ)) * G := by
      have : 0 ≤ θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ) := by positivity
      have : 0 < 1 + 12 * θ * ((Fintype.card σ + S + 3 : ℕ) : ℝ) := by linarith
      positivity
    constructor <;> linarith
  -- κ and ε small
  have hκε : 8 * (O.κ + O.ε * ((Fintype.card σ : ℝ) + S + 1)) * ((P.L : ℝ) + 1) ≤ 1 := by
    have hκ8 : 8 * O.κ * ((P.L : ℝ) + 1) ≤ 1 / 2 := by
      have hbig : (1 : ℝ) ≤ ((Fintype.card σ : ℝ) + S + 1) ^ 2 * (Fintype.card α : ℝ) ^ 2
          * ((Fintype.card σ : ℝ) + S + 3) := by
        have : (1 : ℝ) ≤ (Fintype.card α : ℝ) ^ 2 := one_le_pow₀ hA1
        have h3 : (1 : ℝ) ≤ (Fintype.card σ : ℝ) + S + 3 := by linarith
        have h4 := mul_le_mul hn1sq this zero_le_one (by positivity)
        nlinarith only [h4, h3]
      have hκL : 0 ≤ O.κ * ((P.L : ℝ) + 1) := by positivity
      have e : (2 : ℝ) ^ 14 * ((Fintype.card σ : ℝ) + S + 1) ^ 2 * (Fintype.card α : ℝ) ^ 2
          * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 3) * O.κ
          = 2 ^ 14 * (O.κ * ((P.L : ℝ) + 1)) * (((Fintype.card σ : ℝ) + S + 1) ^ 2
            * (Fintype.card α : ℝ) ^ 2 * ((Fintype.card σ : ℝ) + S + 3)) := by ring
      have hk2 := hκc
      rw [e] at hk2
      have := mul_le_mul_of_nonneg_left hbig
        (by positivity : (0 : ℝ) ≤ 2 ^ 14 * (O.κ * ((P.L : ℝ) + 1)))
      nlinarith only [this, hk2, hεd1]
    have hε8 : 8 * (O.ε * ((Fintype.card σ : ℝ) + S + 1)) * ((P.L : ℝ) + 1) ≤ 1 / 2 := by
      have h0 : 0 ≤ O.ε * ((Fintype.card σ : ℝ) + S + 1) * ((P.L : ℝ) + 1) := by positivity
      have h1 : O.ε * ((Fintype.card σ : ℝ) + S + 1) * ((P.L : ℝ) + 1)
          ≤ ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α
            * (O.ε * ((Fintype.card σ : ℝ) + S + 1)
              + tallyLog (α := α) σ S δ * prefixMax P.D k) := by
        have : 0 ≤ tallyLog (α := α) σ S δ * prefixMax P.D k := mul_nonneg hΛ0 hp₀
        have hε0 : 0 ≤ O.ε * ((Fintype.card σ : ℝ) + S + 1) := by positivity
        have hp3 : O.ε * ((Fintype.card σ : ℝ) + S + 1) * 1
            ≤ O.ε * ((Fintype.card σ : ℝ) + S + 1)
              * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α) :=
          mul_le_mul_of_nonneg_left hQ hε0
        have hp4 : ((P.L : ℝ) + 1) * (O.ε * ((Fintype.card σ : ℝ) + S + 1) * 1)
            ≤ ((P.L : ℝ) + 1) * (O.ε * ((Fintype.card σ : ℝ) + S + 1)
              * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α)
              + tallyLog (α := α) σ S δ * prefixMax P.D k
                * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α)) := by
          have : 0 ≤ tallyLog (α := α) σ S δ * prefixMax P.D k
              * (((Fintype.card σ : ℝ) + S + 1) ^ 2 * Fintype.card α) := by positivity
          exact mul_le_mul_of_nonneg_left (by linarith) (by positivity)
        nlinarith only [hp4]
      nlinarith only [h1, hεc, hεd1]
    nlinarith only [hκ8, hε8]
  -- the records cap
  have hκc' := hκc
  rw [show ((Fintype.card σ : ℝ) + S + 3) = ((Fintype.card σ + S + 3 : ℕ) : ℝ) from hNr.symm]
    at hκc'
  have hεc' := hεc
  rw [mul_comm (tallyLog (α := α) σ S δ) (prefixMax P.D k)] at hεc'
  have hcap := cap_le ((Fintype.card σ : ℝ) + S + 1) (Fintype.card α) ((P.L : ℝ) + 1)
    (((Fintype.card σ + S + 3 : ℕ) : ℝ)) ((Fintype.card σ : ℝ) * Fintype.card α + S + 1) G u εd
    O.κ O.ε θ (prefixMax P.D k * tallyLog (α := α) σ S δ) b n₀ m ((S : ℝ) + 1)
    (Real.log (8 / δ)) hn1 hA1 hL1' hN1 hN3 hZ hZ0 hG1 hu1 hεd0 hεd1 hεu hκ.le hε hθ hpl
    (Nat.cast_nonneg _) (by have : (0 : ℝ) ≤ S := by positivity
                            linarith) (Nat.cast_nonneg _) hθc hκc' hεc' hbX hn₀ hml hlo
  -- the yield
  have hy : 16 * P.q * Real.log (1 / δ)
      ≤ 3 / 2 * θ * coverage P.D P.start P.L M (fun q => ¬ O.Good θ q) := by
    have : 0 ≤ P.q * Real.log (1 / δ) :=
      mul_nonneg P.q_pos.le (Real.log_nonneg (by rw [le_div_iff₀ hδ0]; linarith))
    nlinarith only [this, hyield]
  exact h hδ0 hδ1 hL le_rfl hκ hε hθ hκε (by linarith) (by linarith) hεd1 (by norm_num)
    (by linarith) (by
      have : 24 * θ ≤ 256 * ((P.L : ℝ) + 1) * (((Fintype.card σ : ℝ) + S + 1) * θ) := by
        have : 0 ≤ ((Fintype.card σ : ℝ) + S + 1) * θ := by positivity
        nlinarith only [this, hL1', hn1, hθ]
      linarith)
    (le_min (by linarith) (le_min (by linarith) (by linarith))) hn₀c hexc (Real.exp_pos _)
    hlevel hbge hcap hy

end Guarantee

theorem tally_round_correct_holds : TallyRoundCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ εd P δ S k
  exact tally_round_guarantee M O θ εd P δ S k

theorem tally_round_family_holds : TallyRoundFamily := by
  intro α _ _ _ σ _ Ω _ μ _ fam θ εd P δ S k
  exact tally_round_guarantee fam.M fam.oracle θ εd P δ S k

end OrthoDFA
