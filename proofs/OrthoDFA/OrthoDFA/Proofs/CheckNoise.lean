import OrthoDFA.Proofs.CheckDraws
import OrthoDFA.Proofs.Basics

/-!
# The check's statistic on fixed members

With the members and the suffix fixed, and every string read distinct, the pairs' terms are
independent and each lies in `[-1, 1]`, so the statistic has Hoeffding's moment bound.  Pinning
the noise on a set the reads avoid does not change their law.
-/

namespace OrthoDFA.CheckProof

open MeasureTheory ProbabilityTheory Real
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

section Hoeffding

lemma exp_mul_le_chord (y x : ℝ) (hx : x ∈ Set.Icc (-1 : ℝ) 1) :
    exp (y * x) ≤ cosh y + x * sinh y := by
  have h1 : (0 : ℝ) ≤ (1 + x) / 2 := by linarith [hx.1]
  have h2 : (0 : ℝ) ≤ (1 - x) / 2 := by linarith [hx.2]
  have h := convexOn_exp.2 (Set.mem_univ y) (Set.mem_univ (-y)) h1 h2 (by ring)
  simp only [smul_eq_mul] at h
  calc exp (y * x) = exp ((1 + x) / 2 * y + (1 - x) / 2 * -y) := by congr 1; ring
    _ ≤ _ := h
    _ = _ := by rw [cosh_eq, sinh_eq]; ring

/-- Hoeffding's lemma for a two-point law on `±1` with mean `c`. -/
lemma cosh_add_mul_sinh_le (y c : ℝ) (hc : |c| ≤ 1) :
    cosh y + c * sinh y ≤ exp (c * y + y ^ 2 / 2) := by
  have hc1 : 0 ≤ (1 + c) / 2 := by linarith [neg_abs_le c]
  have hc2 : 0 ≤ (1 - c) / 2 := by linarith [le_abs_self c]
  set ν : Measure ℝ := ENNReal.ofReal ((1 + c) / 2) • Measure.dirac 1
    + ENNReal.ofReal ((1 - c) / 2) • Measure.dirac (-1)
  have hν : IsProbabilityMeasure ν := by
    constructor
    simp only [ν, Measure.add_apply, Measure.smul_apply, measure_univ, smul_eq_mul, mul_one]
    rw [← ENNReal.ofReal_add hc1 hc2, show (1 + c) / 2 + (1 - c) / 2 = 1 by ring]
    simp
  have hint : ∀ f : ℝ → ℝ, ∫ x, f x ∂ν = (1 + c) / 2 * f 1 + (1 - c) / 2 * f (-1) := by
    intro f
    rw [integral_add_measure ((integrable_dirac (by simp)).smul_measure (by simp))
      ((integrable_dirac (by simp)).smul_measure (by simp)), integral_smul_measure,
      integral_smul_measure, integral_dirac, integral_dirac, ENNReal.toReal_ofReal hc1,
      ENNReal.toReal_ofReal hc2, smul_eq_mul, smul_eq_mul]
  have hb : ∀ᵐ x ∂ν, x ∈ Set.Icc (-1 : ℝ) 1 := by
    rw [ae_iff]
    have hm : MeasurableSet {a : ℝ | a ∉ Set.Icc (-1 : ℝ) 1} := measurableSet_Icc.compl
    simp only [ν, Measure.add_apply, Measure.smul_apply, Measure.dirac_apply' _ hm,
      smul_eq_mul]
    simp
  have hsub := hasSubgaussianMGF_of_mem_Icc (X := id) aemeasurable_id hb
  have hmean : ∫ x, x ∂ν = c := by
    rw [hint]; ring
  have hmgf := hsub.mgf_le y
  have hnorm : ((‖(1 : ℝ) - -1‖₊ / 2) ^ 2 : NNReal) = 1 := by
    rw [show (1 : ℝ) - -1 = 2 by norm_num]
    ext
    simp
  rw [hnorm, mgf, hint] at hmgf
  simp only [id, NNReal.coe_one, one_mul] at hmgf
  rw [hmean] at hmgf
  have hkey : (1 + c) / 2 * exp (y * (1 - c)) + (1 - c) / 2 * exp (y * (-1 - c))
      = exp (-(c * y)) * (cosh y + c * sinh y) := by
    rw [cosh_eq, sinh_eq, show y * (1 - c) = -(c * y) + y by ring,
      show y * (-1 - c) = -(c * y) + -y by ring, exp_add, exp_add]
    ring
  rw [hkey] at hmgf
  have hpos := exp_pos (c * y)
  calc cosh y + c * sinh y
      = exp (c * y) * (exp (-(c * y)) * (cosh y + c * sinh y)) := by
        rw [← mul_assoc, ← exp_add]; simp
    _ ≤ exp (c * y) * exp (y ^ 2 / 2) := mul_le_mul_of_nonneg_left hmgf hpos.le
    _ = exp (c * y + y ^ 2 / 2) := (exp_add _ _).symm

/-- Chernoff: a tail of `g` against its exponential moment. -/
lemma measure_le_le_exp_mul {α : Type*} [MeasurableSpace α] (ν : Measure α) (g : α → ℝ)
    (hg : Measurable g) (b : ℝ) :
    ν {x | b ≤ g x} ≤ ENNReal.ofReal (exp (-b)) * ∫⁻ x, ENNReal.ofReal (exp (g x)) ∂ν := by
  have hm : AEMeasurable (fun x => ENNReal.ofReal (exp (g x))) ν :=
    (ENNReal.measurable_ofReal.comp (measurable_exp.comp hg)).aemeasurable
  have h := mul_meas_ge_le_lintegral₀ hm (ENNReal.ofReal (exp b))
  have hsub : {x | b ≤ g x} ⊆ {x | ENNReal.ofReal (exp b) ≤ ENNReal.ofReal (exp (g x))} :=
    fun x hx => ENNReal.ofReal_le_ofReal (exp_le_exp.2 hx)
  calc ν {x | b ≤ g x}
      = ENNReal.ofReal (exp (-b)) * ENNReal.ofReal (exp b) * ν {x | b ≤ g x} := by
        rw [← ENNReal.ofReal_mul (exp_pos _).le, ← exp_add]; simp
    _ ≤ ENNReal.ofReal (exp (-b)) * ENNReal.ofReal (exp b)
          * ν {x | ENNReal.ofReal (exp b) ≤ ENNReal.ofReal (exp (g x))} := by
        gcongr
    _ ≤ _ := by rw [mul_assoc]; gcongr

end Hoeffding

section Blocks
variable (O : Oracle μ S)

/-- `z` as a noise assignment on all of `S`, zero off `W`. -/
noncomputable def extend (W : Finset S) (z : W → ℝ) : S → ℝ :=
  fun w => if hw : w ∈ W then z ⟨w, hw⟩ else 0

lemma measurable_extend (W : Finset S) : Measurable (extend W) := by
  refine measurable_pi_lambda _ fun w => ?_
  by_cases hw : w ∈ W
  · simp only [extend, dif_pos hw]; exact measurable_pi_apply _
  · simp only [extend, dif_neg hw]; exact measurable_const

noncomputable def noiseOn (W : Finset S) (ω : Ω) : W → ℝ := fun i => O.noise i ω

omit [IsProbabilityMeasure μ] in
lemma measurable_noiseOn (W : Finset S) : Measurable (noiseOn O W) :=
  measurable_pi_lambda _ fun i => O.noise_meas i

/-- `f` reads its argument only on `W`. -/
def ReadsOn (W : Finset S) {β : Type*} (f : (S → ℝ) → β) : Prop :=
  ∀ y y', (∀ w ∈ W, y w = y' w) → f y = f y'

omit [IsProbabilityMeasure μ] in
lemma readsOn_eq {W : Finset S} {β : Type*} {f : (S → ℝ) → β} (hf : ReadsOn W f) (ω : Ω) :
    f (fun w => O.noise w ω) = f (extend W (noiseOn O W ω)) :=
  hf _ _ fun w hw => by simp [extend, noiseOn, hw]

omit [IsProbabilityMeasure μ] in
lemma lintegral_mul_of_disjoint (S₁ S₂ : Finset S) (hd : Disjoint S₁ S₂)
    (f g : (S → ℝ) → ℝ≥0∞) (hf : Measurable f) (hg : Measurable g)
    (hf₁ : ReadsOn S₁ f) (hg₂ : ReadsOn S₂ g) :
    ∫⁻ ω, f (fun w => O.noise w ω) * g (fun w => O.noise w ω) ∂μ
      = (∫⁻ ω, f (fun w => O.noise w ω) ∂μ) * ∫⁻ ω, g (fun w => O.noise w ω) ∂μ := by
  have hind : IndepFun (fun ω => f (extend S₁ (noiseOn O S₁ ω)))
      (fun ω => g (extend S₂ (noiseOn O S₂ ω))) μ :=
    (O.noise_indep.indepFun_finset S₁ S₂ hd O.noise_meas).comp
      (hf.comp (measurable_extend S₁)) (hg.comp (measurable_extend S₂))
  simp_rw [readsOn_eq O hf₁, readsOn_eq O hg₂]
  exact lintegral_mul_eq_lintegral_mul_lintegral_of_indepFun
    ((hf.comp (measurable_extend S₁)).comp (measurable_noiseOn O S₁))
    ((hg.comp (measurable_extend S₂)).comp (measurable_noiseOn O S₂)) hind

omit [IsProbabilityMeasure μ] in
lemma measurableSet_pinned (U : Finset S) (b : S → ℝ) : MeasurableSet (pinned O U b) := by
  have : pinned O U b = ⋂ s ∈ U, {ω | O.noise s ω = b s} := by
    ext ω; simp [pinned]
  rw [this]
  exact Finset.measurableSet_biInter U fun s _ =>
    measurableSet_eq_fun (O.noise_meas s) measurable_const

omit [IsProbabilityMeasure μ] in
lemma cond_pinned_le_one (U : Finset S) (b : S → ℝ) (s : Set Ω) :
    μ[|pinned O U b] s ≤ 1 := by
  rw [cond_apply (measurableSet_pinned O U b)]
  calc (μ (pinned O U b))⁻¹ * μ (pinned O U b ∩ s)
      ≤ (μ (pinned O U b))⁻¹ * μ (pinned O U b) := by gcongr; exact Set.inter_subset_left
    _ ≤ 1 := ENNReal.inv_mul_le_one _

/-- An event read off the noise outside `U` is no likelier once the noise on `U` is pinned. -/
lemma cond_pinned_le (U : Finset S) (b : S → ℝ) (W : Finset S) (hW : Disjoint W U)
    (φ : (S → ℝ) → Prop) (hφ : MeasurableSet {y | φ y}) (hread : ReadsOn W φ) :
    μ[|pinned O U b] {ω | φ (fun w => O.noise w ω)} ≤ μ {ω | φ (fun w => O.noise w ω)} := by
  set B : Set (U → ℝ) := {z | ∀ i, z i = b i}
  have hB : MeasurableSet B := by
    simp only [B, Set.ofPred_forall]
    exact MeasurableSet.iInter fun i => measurableSet_eq_fun (measurable_pi_apply i)
      measurable_const
  have hP : pinned O U b = noiseOn O U ⁻¹' B := by
    ext ω; simp [B, pinned, noiseOn]
  have hE : {ω | φ (fun w => O.noise w ω)} = noiseOn O W ⁻¹' (extend W ⁻¹' {y | φ y}) := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_preimage]
    rw [readsOn_eq O hread ω]
  have hind : IndepFun (noiseOn O W) (noiseOn O U) μ :=
    O.noise_indep.indepFun_finset W U hW O.noise_meas
  rw [cond_apply (measurableSet_pinned O U b), Set.inter_comm, hE, hP,
    hind.measure_inter_preimage_eq_mul _ _ (measurable_extend W hφ) hB, mul_comm, mul_assoc]
  by_cases h0 : μ (noiseOn O U ⁻¹' B) = 0
  · rw [h0]; simp
  · rw [ENNReal.mul_inv_cancel h0 (measure_ne_top _ _), mul_one]

end Blocks

section Statistic
variable (O : Oracle μ S)

/-- `O.mq` with the noise given as an argument. -/
noncomputable def mqN (nz : S → ℝ) (w : S) : ℝ := O.label w + (1 - 2 * O.label w) * nz w

noncomputable def pairX (nz : S → ℝ) (a b v : S) : ℝ :=
  (mqN O nz (a * v) - mqN O nz (b * v)) * (mqN O nz a - mqN O nz b)

noncomputable def leanN (nz : S → ℝ) (n : ℕ) (l : List S) (v : S) : ℝ :=
  ∑ k ∈ Finset.range n, pairX O nz (l.getD (2 * k) 1) (l.getD (2 * k + 1) 1) v

omit [IsProbabilityMeasure μ] in
lemma leaning_eq (n : ℕ) (l : List S) (v : S) (ω : Ω) :
    leaning O n l v ω = leanN O (fun w => O.noise w ω) n l v := rfl

omit [IsProbabilityMeasure μ] in
lemma leanN_zero (nz : S → ℝ) (l : List S) (v : S) : leanN O nz 0 l v = 0 := by
  simp [leanN]

omit [IsProbabilityMeasure μ] in
lemma leanN_cons_cons (nz : S → ℝ) (n : ℕ) (a b : S) (l : List S) (v : S) :
    leanN O nz (n + 1) (a :: b :: l) v = pairX O nz a b v + leanN O nz n l v := by
  unfold leanN
  rw [Finset.sum_range_succ', add_comm]
  congr 1

omit [IsProbabilityMeasure μ] in
lemma measurable_mqN (w : S) : Measurable fun nz : S → ℝ => mqN O nz w :=
  measurable_const.add (measurable_const.mul (measurable_pi_apply w))

omit [IsProbabilityMeasure μ] in
lemma measurable_pairX (a b v : S) : Measurable fun nz : S → ℝ => pairX O nz a b v :=
  ((measurable_mqN O _).sub (measurable_mqN O _)).mul
    ((measurable_mqN O _).sub (measurable_mqN O _))

omit [IsProbabilityMeasure μ] in
lemma measurable_leanN (n : ℕ) (l : List S) (v : S) :
    Measurable fun nz : S → ℝ => leanN O nz n l v :=
  Finset.measurable_sum _ fun _ _ => measurable_pairX O _ _ _

/-- The strings the statistic reads on `l`. -/
def reads (l : List S) (v : S) : Finset S := (l ++ l.map (· * v)).toFinset

lemma mem_reads {l : List S} {v w : S} : w ∈ reads l v ↔ w ∈ l ∨ ∃ x ∈ l, x * v = w := by
  simp [reads]

omit [IsProbabilityMeasure μ] in
lemma leanN_readsOn (v : S) : ∀ (n : ℕ) (l : List S), l.length = 2 * n →
    ReadsOn (reads l v) (fun nz => leanN O nz n l v)
  | 0, _, _ => fun _ _ _ => by simp [leanN_zero]
  | n + 1, l, hl => by
      obtain ⟨a, b, l', rfl, hl'⟩ := exists_cons_cons hl
      intro y y' hy
      have hy' : ∀ w ∈ reads l' v, y w = y' w := fun w hw =>
        hy w (by
          rcases mem_reads.1 hw with hw | ⟨x, hx, rfl⟩
          · exact mem_reads.2 (Or.inl (by simp [hw]))
          · exact mem_reads.2 (Or.inr ⟨x, by simp [hx], rfl⟩))
      have h₁ : y a = y' a := hy a (mem_reads.2 (Or.inl (by simp)))
      have h₂ : y b = y' b := hy b (mem_reads.2 (Or.inl (by simp)))
      have h₃ : y (a * v) = y' (a * v) := hy _ (mem_reads.2 (Or.inr ⟨a, by simp, rfl⟩))
      have h₄ : y (b * v) = y' (b * v) := hy _ (mem_reads.2 (Or.inr ⟨b, by simp, rfl⟩))
      have hrest := leanN_readsOn v n l' hl' y y' hy'
      simp only at hrest
      simp only [leanN_cons_cons, pairX, mqN, h₁, h₂, h₃, h₄, hrest]

/-- The expected read at `w`. -/
noncomputable def mean (w : S) : ℝ := O.label w + (1 - 2 * O.label w) * O.rate w

open scoped Classical in
omit [IsProbabilityMeasure μ] in
lemma mean_eq (w : S) : mean O w = if w ∈ O.L then 1 - O.ηIn else O.ηOut := by
  by_cases hw : w ∈ O.L
  · simp [mean, Oracle.rate, Oracle.label, hw]; ring
  · simp [mean, Oracle.rate, Oracle.label, hw]

noncomputable def pairMean (a b v : S) : ℝ :=
  (mean O (a * v) - mean O (b * v)) * (mean O a - mean O b)

lemma mqN_icc (w : S) : ∀ᵐ ω ∂μ, mqN O (fun w => O.noise w ω) w ∈ Set.Icc (0 : ℝ) 1 := by
  filter_upwards [O.noise_bit w] with ω hω
  rcases O.label_bit w with hl | hl <;> rcases hω with h | h <;>
    norm_num [mqN, hl, h]

omit [IsProbabilityMeasure μ] in
lemma measurable_mq (w : S) : Measurable fun ω => mqN O (fun w => O.noise w ω) w :=
  measurable_const.add (measurable_const.mul (O.noise_meas w))

lemma integral_mqN (w : S) : ∫ ω, mqN O (fun w => O.noise w ω) w ∂μ = mean O w := by
  simp only [mqN]
  rw [integral_add (integrable_const _) ((O.noise_int w).const_mul _), integral_const,
    integral_const_mul, O.noise_mean w]
  simp [mean]

lemma integral_mqN_mul {w₁ w₂ : S} (hne : w₁ ≠ w₂) :
    ∫ ω, mqN O (fun w => O.noise w ω) w₁ * mqN O (fun w => O.noise w ω) w₂ ∂μ
      = mean O w₁ * mean O w₂ := by
  have hind : IndepFun (fun ω => mqN O (fun w => O.noise w ω) w₁)
      (fun ω => mqN O (fun w => O.noise w ω) w₂) μ :=
    (O.noise_indep.indepFun hne).comp
      (φ := fun x => O.label w₁ + (1 - 2 * O.label w₁) * x)
      (ψ := fun x => O.label w₂ + (1 - 2 * O.label w₂) * x) (by fun_prop) (by fun_prop)
  rw [← integral_mqN O w₁, ← integral_mqN O w₂]
  exact hind.integral_mul_eq_mul_integral (measurable_mq O w₁).aestronglyMeasurable
    (measurable_mq O w₂).aestronglyMeasurable

lemma integrable_mqN_mul (w₁ w₂ : S) :
    Integrable (fun ω => mqN O (fun w => O.noise w ω) w₁ * mqN O (fun w => O.noise w ω) w₂)
      μ := by
  refine Integrable.of_mem_Icc 0 1 ((measurable_mq O w₁).mul (measurable_mq O w₂)).aemeasurable ?_
  filter_upwards [mqN_icc O w₁, mqN_icc O w₂] with ω h₁ h₂
  exact ⟨mul_nonneg h₁.1 h₂.1, mul_le_one₀ h₁.2 h₂.1 h₂.2⟩

lemma pair_mgf_le (y : ℝ) {a b v : S} (haa : a ≠ a * v) (hab' : a ≠ b * v)
    (hba : b ≠ a * v) (hbb : b ≠ b * v) :
    ∫⁻ ω, ENNReal.ofReal (exp (y * pairX O (fun w => O.noise w ω) a b v)) ∂μ
      ≤ ENNReal.ofReal (cosh y + pairMean O a b v * sinh y) := by
  set X := fun ω => pairX O (fun w => O.noise w ω) a b v
  have hXm : Measurable X :=
    ((measurable_mq O _).sub (measurable_mq O _)).mul ((measurable_mq O _).sub (measurable_mq O _))
  have hXicc : ∀ᵐ ω ∂μ, X ω ∈ Set.Icc (-1 : ℝ) 1 := by
    filter_upwards [mqN_icc O (a * v), mqN_icc O (b * v), mqN_icc O a, mqN_icc O b]
      with ω h₁ h₂ h₃ h₄
    have e₁ : |mqN O (fun w => O.noise w ω) (a * v) - mqN O (fun w => O.noise w ω) (b * v)| ≤ 1 :=
      abs_sub_le_iff.2 ⟨by linarith [h₁.2, h₂.1], by linarith [h₁.1, h₂.2]⟩
    have e₂ : |mqN O (fun w => O.noise w ω) a - mqN O (fun w => O.noise w ω) b| ≤ 1 :=
      abs_sub_le_iff.2 ⟨by linarith [h₃.2, h₄.1], by linarith [h₃.1, h₄.2]⟩
    have : |X ω| ≤ 1 := by
      simp only [X, pairX, abs_mul]
      exact mul_le_one₀ e₁ (abs_nonneg _) e₂
    exact abs_le.1 this
  have hXint : Integrable X μ := Integrable.of_mem_Icc (-1) 1 hXm.aemeasurable hXicc
  have hmean : ∫ ω, X ω ∂μ = pairMean O a b v := by
    have hexp : X = fun ω =>
        mqN O (fun w => O.noise w ω) (a * v) * mqN O (fun w => O.noise w ω) a
          - mqN O (fun w => O.noise w ω) (a * v) * mqN O (fun w => O.noise w ω) b
          - mqN O (fun w => O.noise w ω) (b * v) * mqN O (fun w => O.noise w ω) a
          + mqN O (fun w => O.noise w ω) (b * v) * mqN O (fun w => O.noise w ω) b := by
      funext ω; simp only [X, pairX]; ring
    rw [hexp, integral_add, integral_sub, integral_sub, integral_mqN_mul O (Ne.symm haa),
      integral_mqN_mul O (Ne.symm hab'), integral_mqN_mul O (Ne.symm hba),
      integral_mqN_mul O (Ne.symm hbb), pairMean]
    · ring
    all_goals first
      | exact integrable_mqN_mul O _ _
      | exact (integrable_mqN_mul O _ _).sub (integrable_mqN_mul O _ _)
      | exact ((integrable_mqN_mul O _ _).sub (integrable_mqN_mul O _ _)).sub
          (integrable_mqN_mul O _ _)
  calc ∫⁻ ω, ENNReal.ofReal (exp (y * X ω)) ∂μ
      ≤ ∫⁻ ω, ENNReal.ofReal (cosh y + X ω * sinh y) ∂μ := by
        refine lintegral_mono_ae ?_
        filter_upwards [hXicc] with ω hω
        exact ENNReal.ofReal_le_ofReal (exp_mul_le_chord y (X ω) hω)
    _ = ENNReal.ofReal (∫ ω, (cosh y + X ω * sinh y) ∂μ) := by
        refine (ofReal_integral_eq_lintegral_ofReal
          ((integrable_const _).add (hXint.mul_const _)) ?_).symm
        filter_upwards [hXicc] with ω hω
        exact (exp_pos _).le.trans (exp_mul_le_chord y (X ω) hω)
    _ = ENNReal.ofReal (cosh y + pairMean O a b v * sinh y) := by
        rw [integral_add (integrable_const _) (hXint.mul_const _), integral_const,
          integral_mul_const, hmean]
        simp

/-- The pairs' moment bounds multiply. -/
theorem lintegral_exp_leaning_le (y : ℝ) (v : S) : ∀ (n : ℕ) (l : List S), l.length = 2 * n →
    l.Nodup → (∀ x ∈ l, ∀ z ∈ l, x ≠ z * v) →
    ∫⁻ ω, ENNReal.ofReal (exp (y * leaning O n l v ω)) ∂μ
      ≤ pairProd (fun a b => ENNReal.ofReal (cosh y + pairMean O a b v * sinh y)) l
  | 0, l, hl, _, _ => by
      obtain rfl : l = [] := List.length_eq_zero_iff.1 (by simpa using hl)
      simp [leaning_eq, leanN_zero, pairProd]
  | n + 1, l, hl, hnd, hxz => by
      obtain ⟨a, b, l', rfl, hl'⟩ := exists_cons_cons hl
      have hnd' : l'.Nodup := (List.nodup_cons.1 (List.nodup_cons.1 hnd).2).2
      have ha : a ∉ b :: l' := (List.nodup_cons.1 hnd).1
      have hb : b ∉ l' := (List.nodup_cons.1 (List.nodup_cons.1 hnd).2).1
      have hxz' : ∀ x ∈ l', ∀ z ∈ l', x ≠ z * v := fun x hx z hz =>
        hxz x (by simp [hx]) z (by simp [hz])
      have hcancel : ∀ x z : S, x ≠ z → x * v ≠ z * v := fun x z h e => h (mul_right_cancel e)
      have hab : a ≠ b := fun e => ha (by simp [e])
      have hd : Disjoint ({a, b, a * v, b * v} : Finset S) (reads l' v) := by
        rw [Finset.disjoint_left]
        intro w hw hw'
        rcases mem_reads.1 hw' with hw' | ⟨x, hx, rfl⟩
        · simp only [Finset.mem_insert, Finset.mem_singleton] at hw
          obtain e | e | e | e := hw <;> subst e
          · exact ha (by simp [hw'])
          · exact hb hw'
          · exact hxz (a * v) (by simp [hw']) a (by simp) rfl
          · exact hxz (b * v) (by simp [hw']) b (by simp) rfl
        · simp only [Finset.mem_insert, Finset.mem_singleton] at hw
          rcases hw with e | e | e | e
          · exact hxz a (by simp) x (by simp [hx]) e.symm
          · exact hxz b (by simp) x (by simp [hx]) e.symm
          · exact ha (List.mem_cons_of_mem _ (mul_right_cancel e ▸ hx))
          · exact hb (mul_right_cancel e ▸ hx)
      have hf₁ : ReadsOn ({a, b, a * v, b * v} : Finset S)
          (fun nz => ENNReal.ofReal (exp (y * pairX O nz a b v))) := by
        intro z z' hz
        simp only [pairX, mqN, hz a (by simp), hz b (by simp), hz (a * v) (by simp),
          hz (b * v) (by simp)]
      have hg₂ : ReadsOn (reads l' v) (fun nz => ENNReal.ofReal (exp (y * leanN O nz n l' v))) :=
        fun z z' hz => by
          have h := leanN_readsOn O v n l' hl' z z' hz
          simp only at h ⊢
          rw [h]
      calc ∫⁻ ω, ENNReal.ofReal (exp (y * leaning O (n + 1) (a :: b :: l') v ω)) ∂μ
          = ∫⁻ ω, ENNReal.ofReal (exp (y * pairX O (fun w => O.noise w ω) a b v))
              * ENNReal.ofReal (exp (y * leanN O (fun w => O.noise w ω) n l' v)) ∂μ := by
            refine lintegral_congr fun ω => ?_
            rw [leaning_eq, leanN_cons_cons, mul_add, exp_add, ENNReal.ofReal_mul (exp_pos _).le]
        _ = (∫⁻ ω, ENNReal.ofReal (exp (y * pairX O (fun w => O.noise w ω) a b v)) ∂μ)
              * ∫⁻ ω, ENNReal.ofReal (exp (y * leanN O (fun w => O.noise w ω) n l' v)) ∂μ :=
            lintegral_mul_of_disjoint O _ _ hd
              (fun nz => ENNReal.ofReal (exp (y * pairX O nz a b v)))
              (fun nz => ENNReal.ofReal (exp (y * leanN O nz n l' v)))
              (ENNReal.measurable_ofReal.comp
                (measurable_exp.comp (measurable_const.mul (measurable_pairX O a b v))))
              (ENNReal.measurable_ofReal.comp
                (measurable_exp.comp (measurable_const.mul (measurable_leanN O n l' v))))
              hf₁ hg₂
        _ ≤ ENNReal.ofReal (cosh y + pairMean O a b v * sinh y)
              * pairProd (fun a b => ENNReal.ofReal (cosh y + pairMean O a b v * sinh y)) l' :=
            mul_le_mul'
              (pair_mgf_le O y (hxz a (by simp) a (by simp)) (hxz a (by simp) b (by simp))
                (hxz b (by simp) a (by simp)) (hxz b (by simp) b (by simp)))
              (lintegral_exp_leaning_le y v n l' hl' hnd' hxz')
        _ = _ := (pairProd_cons_cons
            (fun a b => ENNReal.ofReal (cosh y + pairMean O a b v * sinh y)) a b l').symm

end Statistic

end OrthoDFA.CheckProof
