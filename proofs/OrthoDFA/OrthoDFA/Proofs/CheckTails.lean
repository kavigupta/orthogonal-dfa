import OrthoDFA.Proofs.CheckNoise

/-!
# The check's two tails

On members of one label, the statistic has mean zero whatever the suffix.  Under an
accept-preserving suffix, a pair of members of different labels leans positive by the squared
gap of the two reads' means, and a pair differs in label with probability `2a(1-a)`.
-/

namespace OrthoDFA.CheckProof

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

omit [IsProbabilityMeasure μ] in
lemma measurable_leaning (O : Oracle μ S) (n : ℕ) (l : List S) (v : S) :
    Measurable (leaning O n l v) :=
  (measurable_leanN O n l v).comp (measurable_pi_lambda _ O.noise_meas)

lemma sinh_nonneg_of_nonneg {s : ℝ} (hs : 0 ≤ s) : 0 ≤ sinh s := by
  rw [sinh_eq]
  have := exp_le_exp.2 (show -s ≤ s by linarith)
  linarith

lemma eta_nonneg (O : Oracle μ S) : 0 ≤ O.η :=
  (O.rate_nonneg 1).trans (O.rate_le_eta 1)

theorem sound_tail (O : Oracle μ S) (v : S) (n : ℕ) (l : List S) (t : ℝ) (ht : 0 ≤ t)
    (hl : l.length = 2 * n) (hnd : l.Nodup) (hxz : ∀ x ∈ l, ∀ z ∈ l, x ≠ z * v)
    (hlab : ∀ x ∈ l, ∀ z ∈ l, (x ∈ O.L ↔ z ∈ O.L)) :
    μ {ω | 2 * t ≤ leaning O n l v ω} ≤ ENNReal.ofReal (exp (-2 * t ^ 2 / n)) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [CharP.cast_eq_zero, div_zero, exp_zero, ENNReal.ofReal_one]
    exact prob_le_one
  have hnr : (0 : ℝ) < n := by exact_mod_cast hn
  set s := 2 * t / n
  have hs : 0 ≤ s := by positivity
  have hcosh : cosh s ≤ exp (s ^ 2 / 2) := by
    simpa using cosh_add_mul_sinh_le s 0 (by simp)
  have hpair : ∀ a ∈ l, ∀ b ∈ l,
      ENNReal.ofReal (cosh s + pairMean O a b v * sinh s) ≤ ENNReal.ofReal (exp (s ^ 2 / 2)) := by
    intro a ha b hb
    have hm : mean O a = mean O b := by
      rw [mean_eq, mean_eq]
      by_cases hA : a ∈ O.L
      · rw [if_pos hA, if_pos ((hlab a ha b hb).1 hA)]
      · rw [if_neg hA, if_neg (fun hB => hA ((hlab a ha b hb).2 hB))]
    rw [pairMean, hm, sub_self, mul_zero, zero_mul, add_zero]
    exact ENNReal.ofReal_le_ofReal hcosh
  calc μ {ω | 2 * t ≤ leaning O n l v ω}
      ≤ μ {ω | s * (2 * t) ≤ s * leaning O n l v ω} :=
        measure_mono fun ω hω => mul_le_mul_of_nonneg_left hω hs
    _ ≤ ENNReal.ofReal (exp (-(s * (2 * t))))
          * ∫⁻ ω, ENNReal.ofReal (exp (s * leaning O n l v ω)) ∂μ :=
        measure_le_le_exp_mul μ _ (measurable_const.mul (measurable_leaning O n l v)) _
    _ ≤ ENNReal.ofReal (exp (-(s * (2 * t)))) * ENNReal.ofReal (exp (s ^ 2 / 2)) ^ n := by
        gcongr
        exact (lintegral_exp_leaning_le O s v n l hl hnd hxz).trans
          (pairProd_le _ _ n l hl hpair)
    _ = ENNReal.ofReal (exp (-2 * t ^ 2 / n)) := by
        rw [← ENNReal.ofReal_pow (exp_pos _).le, ← ENNReal.ofReal_mul (exp_pos _).le,
          ← exp_nat_mul, ← exp_add]
        congr 2
        simp only [s]
        field_simp
        ring

open scoped Classical in
lemma lintegral_ite_mem (ρ : Measure S) (P : Set S) (A B : ℝ≥0∞) :
    ∫⁻ x, (if x ∈ P then A else B) ∂ρ = ρ P * A + ρ Pᶜ * B := by
  have : (fun x => if x ∈ P then A else B)
      = fun x => P.indicator (fun _ => A) x + Pᶜ.indicator (fun _ => B) x := by
    funext x
    by_cases hx : x ∈ P <;> simp [hx]
  rw [this, lintegral_add_left Measurable.of_discrete,
    lintegral_indicator_const MeasurableSet.of_discrete,
    lintegral_indicator_const MeasurableSet.of_discrete, mul_comm A, mul_comm B]

lemma ofReal_nat_mul (k : ℕ) (r : ℝ) :
    (k : ℝ≥0∞) * ENNReal.ofReal r = ENNReal.ofReal (k * r) := by
  rw [ENNReal.ofReal_mul (Nat.cast_nonneg k), ENNReal.ofReal_natCast]

/-- The power tail, averaged over the members. -/
theorem power_avg (O : Oracle μ S) (ρ : Measure S) [IsProbabilityMeasure ρ] (v : S)
    (hv : ∀ p, p * v ∈ O.L ↔ p ∈ O.L) (G : List S → Prop) [DecidablePred G]
    (hG : ∀ l, G l → l.Nodup ∧ ∀ x ∈ l, ∀ z ∈ l, x ≠ z * v) (n : ℕ) (t η₀ w : ℝ)
    (ht : 0 ≤ t) (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (hw : w ≤ min (ρ.real O.L) (ρ.real O.Lᶜ)) (hmargin : 2 * t ≤ n * w * (1 - 2 * η₀) ^ 2) :
    listInt ρ (2 * n) (fun l => if G l then μ {ω | leaning O n l v ω < 2 * t} else 0)
      ≤ ENNReal.ofReal (exp (-(n * w * (1 - 2 * η₀) ^ 2 - 2 * t) ^ 2 / (2 * n))) := by
  classical
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [CharP.cast_eq_zero, mul_zero, div_zero, exp_zero, ENNReal.ofReal_one]
    refine listInt_le_one ρ _ fun l => ?_
    split_ifs
    · exact prob_le_one
    · exact zero_le_one
  have hnr : (0 : ℝ) < n := by exact_mod_cast hn
  set d := (1 - 2 * η₀) ^ 2
  have hη0 : 0 ≤ η₀ := (eta_nonneg O).trans hη
  have hd0 : 0 < d := pow_pos (by linarith) 2
  have hd1 : d ≤ 1 := by
    simp only [d]
    rw [sq_le_one_iff_abs_le_one, abs_le]
    constructor <;> linarith
  have hw0 : 0 ≤ w := by
    have h1 : 0 ≤ n * w * d := by linarith
    exact (mul_nonneg_iff_of_pos_left hnr).1 ((mul_nonneg_iff_of_pos_right hd0).1 h1)
  set α := ρ.real O.L
  have hαc : ρ.real O.Lᶜ = 1 - α := by
    rw [measureReal_compl O.L_meas, probReal_univ]
  have hα0 : 0 ≤ α := measureReal_nonneg
  have hα1 : α ≤ 1 := measureReal_le_one
  have hwα : w ≤ α := hw.trans (min_le_left _ _)
  have hwα' : w ≤ 1 - α := hαc ▸ hw.trans (min_le_right _ _)
  have h2α : w ≤ 2 * α * (1 - α) := by
    rcases le_total α (1 / 2) with h | h <;> nlinarith
  set s := (n * (w * d) - 2 * t) / n
  have hs : 0 ≤ s := by
    apply div_nonneg _ hnr.le
    nlinarith
  have hsh : 0 ≤ sinh s := sinh_nonneg_of_nonneg hs
  have hc1 : 0 ≤ cosh s - d * sinh s := by
    have := cosh_sub_sinh s
    nlinarith [exp_pos (-s)]
  set g' : S → S → ℝ≥0∞ := fun a b =>
    if (a ∈ O.L ↔ b ∈ O.L) then ENNReal.ofReal (cosh s)
    else ENNReal.ofReal (cosh s - d * sinh s)
  have hgg : ∀ a b, ENNReal.ofReal (cosh (-s) + pairMean O a b v * sinh (-s)) ≤ g' a b := by
    intro a b
    have hmv : ∀ p, mean O (p * v) = mean O p := fun p => by
      rw [mean_eq, mean_eq]
      by_cases hp : p ∈ O.L
      · rw [if_pos hp, if_pos ((hv p).2 hp)]
      · rw [if_neg hp, if_neg (fun h => hp ((hv p).1 h))]
    rw [cosh_neg, sinh_neg, pairMean, hmv, hmv]
    simp only [g']
    split_ifs with hab
    · refine ENNReal.ofReal_le_ofReal ?_
      nlinarith [sq_nonneg (mean O a - mean O b)]
    · refine ENNReal.ofReal_le_ofReal ?_
      have hgap : d ≤ (mean O a - mean O b) * (mean O a - mean O b) := by
        have hIn : O.ηIn ≤ η₀ := (le_max_left _ _).trans hη
        have hOut : O.ηOut ≤ η₀ := (le_max_right _ _).trans hη
        have hsq : d ≤ (1 - O.ηIn - O.ηOut) ^ 2 := by
          simp only [d]
          exact pow_le_pow_left₀ (by linarith) (by linarith) 2
        rw [mean_eq, mean_eq]
        by_cases ha : a ∈ O.L
        · have hb : b ∉ O.L := fun hb => hab ⟨fun _ => hb, fun _ => ha⟩
          rw [if_pos ha, if_neg hb]; nlinarith
        · have hb : b ∈ O.L := by
            by_contra hb; exact hab ⟨fun h => absurd h ha, fun h => absurd h hb⟩
          rw [if_neg ha, if_pos hb]; nlinarith
      nlinarith
  have hint : ∫⁻ a, ∫⁻ b, g' a b ∂ρ ∂ρ ≤ ENNReal.ofReal (cosh s - (w * d) * sinh s) := by
    have hinner : ∀ a, ∫⁻ b, g' a b ∂ρ
        = if a ∈ O.L
          then ρ O.L * ENNReal.ofReal (cosh s) + ρ O.Lᶜ * ENNReal.ofReal (cosh s - d * sinh s)
          else ρ O.L * ENNReal.ofReal (cosh s - d * sinh s) + ρ O.Lᶜ * ENNReal.ofReal (cosh s) := by
      intro a
      by_cases ha : a ∈ O.L
      · rw [if_pos ha, ← lintegral_ite_mem]
        refine lintegral_congr fun b => ?_
        by_cases hb : b ∈ O.L <;> simp [g', ha, hb]
      · rw [if_neg ha, ← lintegral_ite_mem]
        refine lintegral_congr fun b => ?_
        by_cases hb : b ∈ O.L <;> simp [g', ha, hb]
    simp_rw [hinner]
    rw [lintegral_ite_mem]
    have hL : ρ O.L = ENNReal.ofReal α := (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
    have hLc : ρ O.Lᶜ = ENNReal.ofReal (1 - α) := by
      rw [← hαc]; exact (ENNReal.ofReal_toReal (measure_ne_top _ _)).symm
    have hc0 : 0 ≤ cosh s := (cosh_pos s).le
    have h1α : 0 ≤ 1 - α := by linarith
    rw [hL, hLc, ← ENNReal.ofReal_mul hα0, ← ENNReal.ofReal_mul h1α, ← ENNReal.ofReal_mul hα0,
      ← ENNReal.ofReal_mul h1α, ← ENNReal.ofReal_add (by positivity) (mul_nonneg h1α hc1),
      ← ENNReal.ofReal_add (mul_nonneg hα0 hc1) (by positivity),
      ← ENNReal.ofReal_mul hα0, ← ENNReal.ofReal_mul h1α,
      ← ENNReal.ofReal_add (by positivity) (by positivity)]
    refine ENNReal.ofReal_le_ofReal ?_
    nlinarith [mul_le_mul_of_nonneg_right h2α (mul_nonneg hd0.le hsh)]
  have hwd : |-(w * d)| ≤ 1 := by
    rw [abs_neg, abs_le]
    constructor
    · nlinarith
    · nlinarith
  have htwo := cosh_add_mul_sinh_le s (-(w * d)) hwd
  calc listInt ρ (2 * n) (fun l => if G l then μ {ω | leaning O n l v ω < 2 * t} else 0)
      ≤ listInt ρ (2 * n) (fun l => ENNReal.ofReal (exp (s * (2 * t))) * pairProd g' l) := by
        refine listInt_mono ρ _ fun l hl => ?_
        split_ifs with hGl
        · obtain ⟨hnd, hxz⟩ := hG l hGl
          calc μ {ω | leaning O n l v ω < 2 * t}
              ≤ μ {ω | -s * (2 * t) ≤ -s * leaning O n l v ω} := measure_mono fun ω hω => by
                simp only [Set.mem_ofPred_eq] at hω ⊢; nlinarith
            _ ≤ ENNReal.ofReal (exp (-(-s * (2 * t))))
                  * ∫⁻ ω, ENNReal.ofReal (exp (-s * leaning O n l v ω)) ∂μ :=
                measure_le_le_exp_mul μ _ (measurable_const.mul (measurable_leaning O n l v)) _
            _ ≤ ENNReal.ofReal (exp (s * (2 * t))) * pairProd g' l := by
                rw [neg_mul, neg_neg]
                gcongr
                exact (lintegral_exp_leaning_le O (-s) v n l hl hnd hxz).trans
                  (pairProd_mono hgg l)
        · exact zero_le
    _ = ENNReal.ofReal (exp (s * (2 * t))) * (∫⁻ a, ∫⁻ b, g' a b ∂ρ ∂ρ) ^ n := by
        rw [listInt_const_mul, listInt_pairProd]
    _ ≤ ENNReal.ofReal (exp (s * (2 * t)))
          * ENNReal.ofReal (exp (-(w * d) * s + s ^ 2 / 2)) ^ n := by
        gcongr
        exact hint.trans (ENNReal.ofReal_le_ofReal (by linarith))
    _ = ENNReal.ofReal (exp (-(n * w * (1 - 2 * η₀) ^ 2 - 2 * t) ^ 2 / (2 * n))) := by
        rw [← ENNReal.ofReal_pow (exp_pos _).le, ← ENNReal.ofReal_mul (exp_pos _).le,
          ← exp_nat_mul, ← exp_add]
        congr 2
        simp only [s, d]
        field_simp
        ring

end OrthoDFA.CheckProof
