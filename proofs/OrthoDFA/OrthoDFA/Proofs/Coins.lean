import OrthoDFA.Proofs.Hoeffding
import Mathlib.Probability.ProductMeasure
import Mathlib.Probability.Independence.InfinitePi

/-!
# Fair coins on strings

Each string carries an independent fair coin.  The coins on a finite set of strings are
Hoeffding-concentrated, and a weighted sum of them over strings outside a finite set `U` stays
concentrated even when the set it sums over is chosen by the coins on `U`.
-/

namespace OrthoDFA.Thin

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

/-- A fair coin. -/
noncomputable def fairCoin : Measure Bool :=
  (2 : ℝ≥0∞)⁻¹ • Measure.dirac true + (2 : ℝ≥0∞)⁻¹ • Measure.dirac false

instance : IsProbabilityMeasure fairCoin := by
  constructor
  simp only [fairCoin, Measure.add_apply, Measure.smul_apply, measure_univ, smul_eq_mul, mul_one]
  exact ENNReal.inv_two_add_inv_two

lemma fairCoin_real (b : Bool) : fairCoin.real {b} = 1 / 2 := by
  cases b <;> simp [fairCoin, measureReal_def, ENNReal.toReal_inv]

variable {S : Type*}

/-- One fair coin per string. -/
noncomputable abbrev coins (S : Type*) : Measure (S → Bool) := Measure.infinitePi fun _ : S => fairCoin

/-- `+1` for a table-side coin, `-1` for a reserve-side one. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

lemma coins_indep (S : Type*) : iIndepFun (fun (s : S) (g : S → Bool) => g s) (coins S) :=
  iIndepFun_infinitePi (fun _ => measurable_id)

lemma integral_fairCoin (f : Bool → ℝ) : ∫ c, f c ∂fairCoin = (f true + f false) / 2 := by
  rw [integral_fintype (Integrable.of_finite), Fintype.sum_bool, fairCoin_real, fairCoin_real]
  simp only [smul_eq_mul]
  ring

lemma integral_coins_eval (s : S) (f : Bool → ℝ) :
    ∫ g, f (g s) ∂(coins S) = (f true + f false) / 2 := by
  have hmp := measurePreserving_eval_infinitePi (fun _ : S => fairCoin) s
  have h := integral_map (μ := coins S) hmp.measurable.aemeasurable
    (f := f) (measurable_of_countable f).aestronglyMeasurable
  rw [hmp.map_eq] at h
  rw [← integral_fairCoin, h]

lemma coins_indicator_mean (s : S) (b : Bool) :
    ∫ g, (if g s = b then (1 : ℝ) else 0) ∂(coins S) = 1 / 2 := by
  rw [integral_coins_eval s (fun c => if c = b then (1 : ℝ) else 0)]
  cases b <;> simp

/-- More than `|F|(1/2 + τ)` of the coins on `F` showing `b` has chance at most `exp(-2|F|τ²)`. -/
theorem coins_count_upper (F : Finset S) (b : Bool) {τ : ℝ} (hτ : 0 ≤ τ) :
    (coins S).real {g | (F.card : ℝ) * (1 / 2 + τ) ≤ ((F.filter (fun s => g s = b)).card : ℝ)}
      ≤ Real.exp (-2 * (F.card : ℝ) * τ ^ 2) := by
  classical
  set X : S → (S → Bool) → ℝ := fun s g => if g s = b then 1 else 0 with hX
  have hmeas : ∀ s, Measurable (X s) := fun s =>
    Measurable.ite (measurableSet_eq_fun (measurable_pi_apply s) measurable_const)
      measurable_const measurable_const
  have hind : iIndepFun X (coins S) :=
    (coins_indep S).comp (fun _ (c : Bool) => if c = b then (1 : ℝ) else 0) (fun _ => by fun_prop)
  have h := sumUpper_le X F (1 / 2) τ (fun s => (hmeas s).aemeasurable) hind
    (fun s => Filter.Eventually.of_forall (fun g => by
      simp only [hX]; split_ifs <;> simp))
    (by rw [Finset.sum_congr rfl (fun s _ => coins_indicator_mean s b), Finset.sum_const,
      nsmul_eq_mul]) hτ
  refine le_trans (measureReal_mono (fun g hg => ?_) (measure_ne_top _ _)) h
  simp only [Set.mem_setOf_eq] at hg ⊢
  rw [Finset.sum_boole]
  exact hg

/-- The weighted coin sum over a finite set strays `t` from zero with chance at most
`2 exp(-t²/(2 ∑ a²))`. -/
theorem coins_weighted_tail (F : Finset S) (a : S → ℝ) {t : ℝ} (ht : 0 < t) :
    (coins S).real {g | t ≤ |∑ s ∈ F, a s * sgn (g s)|}
      ≤ 2 * Real.exp (-t ^ 2 / (2 * ∑ s ∈ F, a s ^ 2)) := by
  classical
  have hmeasσ : Measurable sgn := measurable_of_countable _
  have hint : ∀ s, ∫ g, a s * sgn (g s) ∂(coins S) = 0 := by
    intro s
    rw [integral_coins_eval s (fun c => a s * sgn c)]
    simp [sgn]
  set c : S → ℝ≥0 := fun s => (‖|a s| - -|a s|‖₊ / 2) ^ 2 with hcdef
  have hsub : ∀ (ε : ℝ), |ε| = 1 → ∀ s ∈ F, HasSubgaussianMGF
      (fun g : S → Bool => ε * (a s * sgn (g s))) (c s) (coins S) := by
    intro ε hε s _
    have hm : AEMeasurable (fun g : S → Bool => ε * (a s * sgn (g s))) (coins S) :=
      (measurable_const.mul (measurable_const.mul
        (hmeasσ.comp (measurable_pi_apply s)))).aemeasurable
    have hIcc : ∀ᵐ g ∂(coins S), ε * (a s * sgn (g s)) ∈ Set.Icc (-|a s|) |a s| := by
      refine Filter.Eventually.of_forall (fun g => ?_)
      have h1 : |ε * (a s * sgn (g s))| = |a s| := by
        rw [abs_mul, abs_mul, hε]
        cases g s <;> simp [sgn]
      exact abs_le.1 (le_of_eq h1)
    have h := hasSubgaussianMGF_of_mem_Icc hm hIcc
    have hmean : ∫ g, ε * (a s * sgn (g s)) ∂(coins S) = 0 := by
      rw [integral_const_mul, hint s, mul_zero]
    simp only [hmean, sub_zero] at h
    exact h
  have hindε : ∀ ε : ℝ, iIndepFun (fun s (g : S → Bool) => ε * (a s * sgn (g s))) (coins S) :=
    fun ε => (coins_indep S).comp (fun s (c : Bool) => ε * (a s * sgn c))
      (fun _ => measurable_of_countable _)
  have hup := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun (hindε 1)
    (hsub 1 (by simp)) ht.le
  have hlo := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun (hindε (-1))
    (hsub (-1) (by simp)) ht.le
  have hc : ((∑ s ∈ F, c s : ℝ≥0) : ℝ) = ∑ s ∈ F, a s ^ 2 := by
    rw [NNReal.coe_sum]
    refine Finset.sum_congr rfl (fun s _ => ?_)
    simp only [hcdef, NNReal.coe_pow, NNReal.coe_div, coe_nnnorm, Real.norm_eq_abs]
    rw [sub_neg_eq_add, abs_of_nonneg (by positivity : (0 : ℝ) ≤ |a s| + |a s|)]
    rw [show ((2 : ℝ≥0) : ℝ) = 2 from rfl, ← sq_abs (a s)]
    ring
  rw [hc] at hup hlo
  have hsplit : {g : S → Bool | t ≤ |∑ s ∈ F, a s * sgn (g s)|}
      ⊆ {g | t ≤ ∑ s ∈ F, 1 * (a s * sgn (g s))} ∪ {g | t ≤ ∑ s ∈ F, -1 * (a s * sgn (g s))} := by
    intro g hg
    simp only [Set.mem_setOf_eq, Set.mem_union, one_mul, neg_one_mul, Finset.sum_neg_distrib] at hg ⊢
    rcases le_abs'.1 hg with h | h
    · right; linarith
    · left; exact h
  refine le_trans (measureReal_mono hsplit (measure_ne_top _ _))
    (le_trans (measureReal_union_le _ _) ?_)
  linarith

lemma exp_tail_mono {t x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) :
    Real.exp (-t ^ 2 / (2 * x)) ≤ Real.exp (-t ^ 2 / (2 * y)) := by
  refine Real.exp_le_exp.2 ?_
  rw [neg_div, neg_div, neg_le_neg_iff]
  exact div_le_div_of_nonneg_left (sq_nonneg t) (by positivity) (by linarith)

open scoped Classical in
/-- `coins_weighted_tail` over the strings of `F` outside `U` that a set `W` picks, where `W` reads
only the coins on `U`. -/
theorem coins_split_tail (U F : Finset S) (a : S → ℝ) (W : (S → Bool) → Set S)
    (hW : ∀ g g', (∀ s ∈ U, g s = g' s) → W g = W g') {t : ℝ} (ht : 0 < t) :
    (coins S).real {g | t ≤ |∑ s ∈ F.filter (fun s => s ∈ W g ∧ s ∉ U), a s * sgn (g s)|}
      ≤ 2 * Real.exp (-t ^ 2 / (2 * ∑ s ∈ F, a s ^ 2)) := by
  set ext : (U → Bool) → S → Bool := fun γ s => if h : s ∈ U then γ ⟨s, h⟩ else false with hext
  set Fγ : (U → Bool) → Finset S := fun γ => F.filter (fun s => s ∈ W (ext γ) ∧ s ∉ U) with hFγ
  set C : (U → Bool) → Set (S → Bool) := fun γ => {g | ∀ s : U, g s = γ s} with hC
  set B : (U → Bool) → Set (S → Bool) :=
    fun γ => {g | t ≤ |∑ s ∈ Fγ γ, a s * sgn (g s)|} with hB
  set b : ℝ := 2 * Real.exp (-t ^ 2 / (2 * ∑ s ∈ F, a s ^ 2)) with hb
  have hb0 : 0 ≤ b := by positivity
  have hsub : {g : S → Bool | t ≤ |∑ s ∈ F.filter (fun s => s ∈ W g ∧ s ∉ U), a s * sgn (g s)|}
      ⊆ ⋃ γ ∈ (Finset.univ : Finset (U → Bool)), (C γ ∩ B γ) := by
    intro g hg
    refine Set.mem_biUnion (Finset.mem_univ (fun s : U => g s)) ⟨fun s => rfl, ?_⟩
    have e : W g = W (ext (fun s : U => g s)) :=
      hW _ _ (fun s hs => by simp [hext, hs])
    simp only [hB, hFγ, Set.mem_setOf_eq, ← e]
    exact hg
  have hmeasU : Measurable (fun (g : S → Bool) (i : U) => g i) := by fun_prop
  have hmeasF : ∀ γ, Measurable (fun (g : S → Bool) (i : Fγ γ) => g i) := fun γ => by fun_prop
  have hind : ∀ γ, (C γ ∩ B γ) = (fun (g : S → Bool) (i : U) => g i) ⁻¹' {γ}
      ∩ (fun (g : S → Bool) (i : Fγ γ) => g i) ⁻¹'
        {h | t ≤ |∑ i : Fγ γ, a i * sgn (h i)|} := by
    intro γ
    ext g
    simp only [hC, hB, Set.mem_inter_iff, Set.mem_setOf_eq, Set.mem_preimage,
      Set.mem_singleton_iff, funext_iff]
    rw [← Finset.sum_coe_sort (Fγ γ) (fun s => a s * sgn (g s))]
  have hdisj : ∀ γ, Disjoint U (Fγ γ) := fun γ =>
    Finset.disjoint_right.2 (fun s hs => (Finset.mem_filter.1 hs).2.2)
  have hprod : ∀ γ, (coins S).real (C γ ∩ B γ) ≤ (coins S).real (C γ) * b := by
    intro γ
    have hI := (coins_indep S).indepFun_finset U (Fγ γ) (hdisj γ) (fun _ => measurable_pi_apply _)
    rw [hind γ, measureReal_def, hI.measure_inter_preimage_eq_mul _ _
      (Set.to_countable _).measurableSet (Set.to_countable _).measurableSet, ENNReal.toReal_mul,
      ← measureReal_def, ← measureReal_def]
    have hCγ : (fun (g : S → Bool) (i : U) => g i) ⁻¹' {γ} = C γ := by
      ext g
      simp [hC, funext_iff]
    rw [hCγ]
    refine mul_le_mul_of_nonneg_left ?_ measureReal_nonneg
    have hBγ : (fun (g : S → Bool) (i : Fγ γ) => g i) ⁻¹' {h | t ≤ |∑ i : Fγ γ, a i * sgn (h i)|}
        = B γ := by
      ext g
      simp only [hB, Set.mem_preimage, Set.mem_setOf_eq]
      rw [← Finset.sum_coe_sort (Fγ γ) (fun s => a s * sgn (g s))]
    rw [hBγ]
    by_cases hz : ∑ s ∈ Fγ γ, a s ^ 2 = 0
    · have h0 : ∀ s ∈ Fγ γ, a s = 0 := fun s hs =>
        pow_eq_zero_iff (n := 2) (by norm_num) |>.1
          ((Finset.sum_eq_zero_iff_of_nonneg (fun s _ => sq_nonneg (a s))).1 hz s hs)
      have hempty : B γ = ∅ := by
        ext g
        simp only [hB, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_le]
        rw [Finset.sum_eq_zero (fun s hs => by rw [h0 s hs, zero_mul]), abs_zero]
        exact ht
      rw [hempty, measureReal_empty]
      exact hb0
    · have hpos : 0 < ∑ s ∈ Fγ γ, a s ^ 2 :=
        lt_of_le_of_ne (Finset.sum_nonneg (fun s _ => sq_nonneg _)) (Ne.symm hz)
      refine le_trans (coins_weighted_tail (Fγ γ) a ht) ?_
      rw [hb]
      exact mul_le_mul_of_nonneg_left (exp_tail_mono hpos (Finset.sum_le_sum_of_subset_of_nonneg
        (Finset.filter_subset _ _) (fun s _ _ => sq_nonneg _))) (by norm_num)
  have hpart : ∑ γ ∈ (Finset.univ : Finset (U → Bool)), (coins S).real (C γ) = 1 := by
    have hU : (⋃ γ ∈ (Finset.univ : Finset (U → Bool)), C γ) = Set.univ := by
      ext g
      simp only [Set.mem_iUnion, Set.mem_univ, iff_true]
      exact ⟨fun s : U => g s, Finset.mem_univ _, fun s => rfl⟩
    have hd : Set.PairwiseDisjoint (↑(Finset.univ : Finset (U → Bool))) C := by
      intro γ _ γ' _ hne
      refine Set.disjoint_left.2 (fun g h1 h2 => hne (funext fun s => ?_))
      rw [← h1 s, ← h2 s]
    have hmC : ∀ γ, MeasurableSet (C γ) := by
      intro γ
      have e : C γ = (fun (g : S → Bool) (i : U) => g i) ⁻¹' {γ} := by
        ext g
        simp [hC, funext_iff]
      rw [e]
      exact hmeasU (measurableSet_singleton γ)
    rw [← measureReal_biUnion_finset hd (fun γ _ => hmC γ), hU,
      probReal_univ]
  calc (coins S).real {g | t ≤ |∑ s ∈ F.filter (fun s => s ∈ W g ∧ s ∉ U), a s * sgn (g s)|}
      ≤ (coins S).real (⋃ γ ∈ (Finset.univ : Finset (U → Bool)), (C γ ∩ B γ)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ γ ∈ (Finset.univ : Finset (U → Bool)), (coins S).real (C γ ∩ B γ) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ γ ∈ (Finset.univ : Finset (U → Bool)), (coins S).real (C γ) * b :=
        Finset.sum_le_sum (fun γ _ => hprod γ)
    _ = b := by rw [← Finset.sum_mul, hpart, one_mul]

end OrthoDFA.Thin
