import RLStar.Proofs.LoopAux
import RLStar.LowLevelClaims.Trie
import RLStar.Proofs.Hoeffding
import RLStar.Proofs.MiscutGate
import RLStar.Proofs.DFABridge

/-!
# The family's realized misreads over strings of one length

`T` puts its mass on the strings of length `L`, at most `a` on each. A family `F` read at the cut
`(kl, kh)` reads those strings off disjoint blocks of the oracle's answers, so the `T`-share it
misreads or leaves undecided exceeds its mean by `t` with chance at most `exp(-2t²/a)`
(`misFrac_tail`, from `sumUpper_weighted`). Every family of at most `FV` strings of length `Ls`,
at every cut, stays within `t` of its mean but for chance `2 (FV + 1)³ (|Σ|^Ls + 1)^FV exp(-2t²/a)`
(`misread_union_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

open scoped NNReal in
/-- For independent `X i ∈ [0, 1]` and weights `w i ≥ 0` with `∑ w i E[X i] ≤ m`:
    P(m + t ≤ ∑ w i X i) ≤ exp(-2t²/a)   where ∑ w i² ≤ a. -/
theorem sumUpper_weighted {ι : Type*} (X : ι → Ω → ℝ) (s : Finset ι) (w : ι → ℝ) (m t a : ℝ)
    (hmeas : ∀ i, AEMeasurable (X i) μ) (h_indep : iIndepFun X μ)
    (hIcc : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1) (hw : ∀ i, 0 ≤ w i)
    (hmean : ∑ i ∈ s, w i * μ[X i] ≤ m) (ht : 0 ≤ t)
    (hwa : ∑ i ∈ s, w i ^ 2 ≤ a) :
    μ.real {ω | m + t ≤ ∑ i ∈ s, w i * X i ω} ≤ Real.exp (-2 * t ^ 2 / a) := by
  rcases ht.eq_or_lt with rfl | ht0
  · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero, zero_div,
      Real.exp_zero]
    exact measureReal_le_one
  have hc : ∀ i, (((‖w i - 0‖₊ / 2) ^ 2 : ℝ≥0) : ℝ) = w i ^ 2 / 4 := fun i => by
    push_cast
    simp only [sub_zero, Real.norm_eq_abs, abs_of_nonneg (hw i)]
    ring
  rcases (Finset.sum_nonneg fun i _ => sq_nonneg (w i) : 0 ≤ ∑ i ∈ s, w i ^ 2).eq_or_lt with
    h0 | h0
  · have hw0 : ∀ i ∈ s, w i = 0 := fun i hi =>
      pow_eq_zero_iff (n := 2) (by norm_num) |>.1
        ((Finset.sum_eq_zero_iff_of_nonneg fun j _ => sq_nonneg (w j)).1 h0.symm i hi)
    have hm0 : 0 ≤ m := by
      rw [Finset.sum_eq_zero fun i hi => by rw [hw0 i hi, zero_mul]] at hmean
      exact hmean
    have hempty : {ω | m + t ≤ ∑ i ∈ s, w i * X i ω} = ∅ := by
      ext ω
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_le]
      rw [Finset.sum_eq_zero fun i hi => by rw [hw0 i hi, zero_mul]]
      linarith
    rw [hempty, measureReal_empty]
    exact (Real.exp_pos _).le
  set Z : ι → Ω → ℝ := fun i ω => w i * X i ω with hZ
  set Y : ι → Ω → ℝ := fun i ω => Z i ω - μ[Z i] with hY
  have hYindep : iIndepFun Y μ := by
    have h := h_indep.comp (fun i (x : ℝ) => w i * x - μ[Z i])
      fun i => (measurable_const_mul _).sub_const _
    exact h
  have hZI : ∀ i, μ[Z i] = w i * μ[X i] := fun i => integral_const_mul _ _
  have hZicc : ∀ i, ∀ᵐ ω ∂μ, Z i ω ∈ Set.Icc (0 : ℝ) (w i) := fun i =>
    (hIcc i).mono fun ω h => ⟨mul_nonneg (hw i) h.1, by nlinarith [h.2, hw i]⟩
  have hsub : ∀ i ∈ s, HasSubgaussianMGF (Y i) ((‖w i - 0‖₊ / 2) ^ 2) μ := fun i _ =>
    hasSubgaussianMGF_of_mem_Icc ((hmeas i).const_mul _) (hZicc i)
  have hevent : {ω | m + t ≤ ∑ i ∈ s, w i * X i ω} ⊆ {ω | t ≤ ∑ i ∈ s, Y i ω} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq] at hω ⊢
    have e : ∑ i ∈ s, Y i ω = ∑ i ∈ s, w i * X i ω - ∑ i ∈ s, w i * μ[X i] := by
      show ∑ i ∈ s, (Z i ω - μ[Z i]) = _
      rw [Finset.sum_sub_distrib]
      simp only [hZI]
      rfl
    linarith
  have hmain := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hYindep
    (c := fun i => (‖w i - 0‖₊ / 2) ^ 2) (s := s) hsub ht
  refine (measureReal_mono hevent).trans (hmain.trans ?_)
  have hsum : ((∑ i ∈ s, (‖w i - 0‖₊ / 2) ^ 2 : ℝ≥0) : ℝ) = (∑ i ∈ s, w i ^ 2) / 4 := by
    rw [NNReal.coe_sum, Finset.sum_div]
    exact Finset.sum_congr rfl fun i _ => hc i
  rw [hsum]
  refine Real.exp_le_exp.2 ?_
  rw [show -t ^ 2 / (2 * ((∑ i ∈ s, w i ^ 2) / 4)) = -(2 * t ^ 2) / ∑ i ∈ s, w i ^ 2 by
    field_simp; ring, neg_mul, neg_div, neg_div, neg_le_neg_iff]
  exact div_le_div_of_nonneg_left (by positivity) h0 hwa

omit [DecidableEq α] in
lemma eq_sum_dirac_of_length {T : Measure (FreeMonoid α)} {L : ℕ}
    (hT : T {x | x.toList.length ≠ L} = 0) :
    T = ∑ v : List.Vector α L,
      T {FreeMonoid.ofList v.toList} • Measure.dirac (FreeMonoid.ofList v.toList) := by
  classical
  set f : List.Vector α L → FreeMonoid α := fun v => FreeMonoid.ofList v.toList with hf
  have hfi : Function.Injective f := fun v v' h => by
    have := congrArg FreeMonoid.toList h
    simpa [hf] using List.Vector.toList_injective this
  ext s hs
  rw [Measure.coe_finsetSum, Finset.sum_apply]
  simp only [Measure.smul_apply, smul_eq_mul, Measure.dirac_apply' _ hs]
  have hsub : s ⊆ ↑((Finset.univ.filter fun v => f v ∈ s).image f)
      ∪ {x | x.toList.length ≠ L} := by
    intro x hx
    by_cases hl : x.toList.length = L
    · left
      refine Finset.mem_coe.2 (Finset.mem_image.2 ⟨⟨x.toList, hl⟩, ?_, ?_⟩)
      · exact Finset.mem_filter.2 ⟨Finset.mem_univ _, by simpa [hf] using hx⟩
      · simp [hf]
    · exact Or.inr hl
  have hsub' : (↑((Finset.univ.filter fun v => f v ∈ s).image f) : Set (FreeMonoid α)) ⊆ s := by
    intro x hx
    obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 (Finset.mem_coe.1 hx)
    exact (Finset.mem_filter.1 hv).2
  have hmeas : T s = T ↑((Finset.univ.filter fun v => f v ∈ s).image f) := by
    refine le_antisymm ((measure_mono hsub).trans ((measure_union_le _ _).trans ?_))
      (measure_mono hsub')
    rw [hT, add_zero]
  rw [hmeas, ← sum_measure_singleton, Finset.sum_image fun v _ v' _ h => hfi h, Finset.sum_filter]
  refine Finset.sum_congr rfl fun v _ => ?_
  by_cases hv : f v ∈ s
  · rw [if_pos hv, Set.indicator_of_mem hv, Pi.one_apply, mul_one]
  · rw [if_neg hv, Set.indicator_of_notMem hv, mul_zero]

omit [DecidableEq α] in
lemma integral_eq_sum_vector {T : Measure (FreeMonoid α)} [IsFiniteMeasure T] {L : ℕ}
    (hT : T {x | x.toList.length ≠ L} = 0) (g : FreeMonoid α → ℝ) :
    ∫ x, g x ∂T = ∑ v : List.Vector α L,
      T.real {FreeMonoid.ofList v.toList} * g (FreeMonoid.ofList v.toList) := by
  conv_lhs => rw [eq_sum_dirac_of_length hT]
  rw [integral_finsetSum_measure fun v _ =>
    (integrable_dirac (by simp)).smul_measure (measure_ne_top _ _)]
  simp only [integral_smul_measure, integral_dirac, smul_eq_mul, measureReal_def]


open scoped Classical in
/-- The `T`-share of the strings that `F`'s read at the noise `ω` misreads or leaves
undecided. -/
def misFrac (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (T : Measure (FreeMonoid α)) (ω : Ω) : ℝ :=
  T.real (Random.misreadSet (fun z => familyRead O.mq F kl kh z ω) O.L)

/-- Its mean over the noise. -/
def misMean (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (T : Measure (FreeMonoid α)) : ℝ :=
  ∫ x, (misreadProb O F kl kh x + readProb O F kl kh x .undecided) ∂T

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma familyRead_clamp (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (z : FreeMonoid α) (ω : Ω) :
    familyRead O.mq F kl kh z ω = familyRead O.mq F (min kl F.card) (min kh (F.card + 1)) z ω := by
  rw [familyRead_eq, familyRead_eq]
  have h := voteCount_le_card O.mq F z ω
  unfold readOf
  split_ifs <;> first | rfl | omega

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma misFrac_clamp (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (T : Measure (FreeMonoid α)) (ω : Ω) :
    misFrac O F kl kh T ω = misFrac O F (min kl F.card) (min kh (F.card + 1)) T ω := by
  have e : (fun z => familyRead O.mq F kl kh z ω)
      = fun z => familyRead O.mq F (min kl F.card) (min kh (F.card + 1)) z ω :=
    funext fun z => familyRead_clamp O F kl kh z ω
  unfold misFrac
  rw [e]

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma misMean_clamp (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (T : Measure (FreeMonoid α)) :
    misMean O F kl kh T = misMean O F (min kl F.card) (min kh (F.card + 1)) T := by
  unfold misMean misreadProb readProb
  simp only [familyRead_clamp O F kl kh]

open scoped Classical in
/-- The indicator that `F`'s read of `x` misreads or leaves it undecided. -/
def misVar (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (x : FreeMonoid α) (ω : Ω) : ℝ :=
  if familyRead O.mq F kl kh x ω ≠ (if x ∈ O.L then ARU.accept else ARU.reject) then 1 else 0

omit [DecidableEq α] in
lemma measurable_misVar (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (x : FreeMonoid α) :
    Measurable[noiseAlg O ↑(F.image fun v => x * v)] (misVar O F kl kh x) := by
  classical
  have hv := measurable_voteCount_block O F x (T := ↑(F.image fun v => x * v))
    fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv)
  have e : misVar O F kl kh x = (fun j : ℕ => if readOf kl kh j
      ≠ (if x ∈ O.L then ARU.accept else ARU.reject) then (1 : ℝ) else 0)
      ∘ fun ω => voteCount O.mq F x ω := by
    funext ω
    simp only [misVar, familyRead_eq, Function.comp]
  rw [e]
  exact (measurable_from_top (f := fun j : ℕ => if readOf kl kh j
      ≠ (if x ∈ O.L then ARU.accept else ARU.reject) then (1 : ℝ) else 0)).comp hv

omit [DecidableEq α] in
lemma integral_misVar (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (x : FreeMonoid α) :
    ∫ ω, misVar O F kl kh x ω ∂μ = misreadProb O F kl kh x + readProb O F kl kh x .undecided := by
  classical
  set E := {ω | familyRead O.mq F kl kh x ω ≠ (if x ∈ O.L then ARU.accept else ARU.reject)}
  have hm : ∀ v : ARU, MeasurableSet {ω | familyRead O.mq F kl kh x ω = v} := fun v => by
    simp only [familyRead_eq]
    exact (measurable_voteCount O F x) ((Set.to_countable {j | readOf kl kh j = v}).measurableSet)
  have hE : MeasurableSet E := (hm _).compl
  have e1 : misVar O F kl kh x = E.indicator 1 := by
    funext ω
    simp only [misVar, Set.indicator, E, Set.mem_ofPred_eq, Pi.one_apply]
  rw [e1, integral_indicator_one hE]
  have e2 : E = {ω | familyRead O.mq F kl kh x ω = if x ∈ O.L then ARU.reject else ARU.accept}
      ∪ {ω | familyRead O.mq F kl kh x ω = .undecided} := by
    ext ω
    simp only [E, Set.mem_ofPred_eq, Set.mem_union]
    split_ifs <;> rcases familyRead O.mq F kl kh x ω <;> simp
  rw [e2, measureReal_union (Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [Set.mem_ofPred_eq] at h1 h2
      rw [h1] at h2
      split_ifs at h2) (hm _)]
  rfl

omit [DecidableEq α] [IsProbabilityMeasure μ] in
lemma misFrac_eq (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    {T : Measure (FreeMonoid α)} [IsFiniteMeasure T] {L : ℕ} (hT : T {x | x.toList.length ≠ L} = 0)
    (ω : Ω) : misFrac O F kl kh T ω
      = ∑ v : List.Vector α L,
        T.real {FreeMonoid.ofList v.toList} * misVar O F kl kh (FreeMonoid.ofList v.toList) ω := by
  classical
  unfold misFrac
  rw [← integral_indicator_one MeasurableSpace.measurableSet_top, integral_eq_sum_vector hT]
  refine Finset.sum_congr rfl fun v _ => ?_
  simp only [Set.indicator, Random.misreadSet, Set.mem_ofPred_eq, Pi.one_apply, misVar]
  congr

omit [DecidableEq α] in
lemma misMean_eq (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    {T : Measure (FreeMonoid α)} [IsFiniteMeasure T] {L : ℕ}
    (hT : T {x | x.toList.length ≠ L} = 0) :
    misMean O F kl kh T
      = ∑ v : List.Vector α L, T.real {FreeMonoid.ofList v.toList}
        * ∫ ω, misVar O F kl kh (FreeMonoid.ofList v.toList) ω ∂μ := by
  unfold misMean
  rw [integral_eq_sum_vector hT]
  simp only [integral_misVar]

omit [DecidableEq α] in
lemma blocks_disjoint (F : Finset (FreeMonoid α)) (L : ℕ) (v v' : List.Vector α L) (h : v ≠ v') :
    Disjoint (F.image fun f => FreeMonoid.ofList v.toList * f)
      (F.image fun f => FreeMonoid.ofList v'.toList * f) := by
  classical
  refine Finset.disjoint_left.2 fun w hw hw' => h ?_
  obtain ⟨f, -, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨f', -, he⟩ := Finset.mem_image.1 hw'
  have he' := congrArg FreeMonoid.toList he
  simp only [FreeMonoid.toList_mul, FreeMonoid.toList_ofList] at he'
  exact List.Vector.toList_injective (List.append_inj he' (by simp)).1.symm

omit [DecidableEq α] in
/-- `F`'s misread `T`-share reaches its mean plus `t` with chance at most `exp(-2t²/a)`. -/
theorem misFrac_tail (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    {T : Measure (FreeMonoid α)} [IsProbabilityMeasure T] {L : ℕ}
    (hT : T {x | x.toList.length ≠ L} = 0) {a : ℝ}
    (hatom : ∀ v : List.Vector α L, T.real {FreeMonoid.ofList v.toList} ≤ a) {t : ℝ}
    (ht : 0 ≤ t) :
    μ.real {ω | misMean O F kl kh T + t ≤ misFrac O F kl kh T ω}
      ≤ Real.exp (-2 * t ^ 2 / a) := by
  classical
  have hY : ∀ v : List.Vector α L,
      Measurable[noiseAlg O ↑(F.image fun f => FreeMonoid.ofList v.toList * f)]
      (misVar O F kl kh (FreeMonoid.ofList v.toList)) := fun v => measurable_misVar O F kl kh _
  have hind : iIndepFun (fun v : List.Vector α L => misVar O F kl kh (FreeMonoid.ofList v.toList))
      μ := iIndepFun_of_blocks O
    (fun v : List.Vector α L => F.image fun f => FreeMonoid.ofList v.toList * f)
    (fun v v' h => blocks_disjoint F L v v' h)
    (fun v : List.Vector α L => misVar O F kl kh (FreeMonoid.ofList v.toList)) hY
  have hmeas : ∀ v : List.Vector α L,
      AEMeasurable (misVar O F kl kh (FreeMonoid.ofList v.toList)) μ := fun v =>
    ((hY v).mono (noiseAlg_le O _) le_rfl).aemeasurable
  have hIcc : ∀ v : List.Vector α L,
      ∀ᵐ ω ∂μ, misVar O F kl kh (FreeMonoid.ofList v.toList) ω ∈ Set.Icc (0 : ℝ) 1 := fun v =>
    Filter.Eventually.of_forall fun ω => by
      simp only [misVar]
      split_ifs <;> simp
  have hw : ∀ v : List.Vector α L, 0 ≤ T.real {FreeMonoid.ofList v.toList} := fun v =>
    measureReal_nonneg
  have hsum1 : ∑ v : List.Vector α L, T.real {FreeMonoid.ofList v.toList} = 1 := by
    have h := integral_eq_sum_vector hT (fun _ => (1 : ℝ))
    simp only [integral_const, smul_eq_mul, mul_one] at h
    exact h.symm.trans (by simp)
  have hwa : ∑ v : List.Vector α L, T.real {FreeMonoid.ofList v.toList} ^ 2 ≤ a := by
    calc ∑ v : List.Vector α L, T.real {FreeMonoid.ofList v.toList} ^ 2
        ≤ ∑ v : List.Vector α L, a * T.real {FreeMonoid.ofList v.toList} :=
          Finset.sum_le_sum fun v _ => by rw [sq]; exact mul_le_mul_of_nonneg_right (hatom v) (hw v)
      _ = a := by rw [← Finset.mul_sum, hsum1, mul_one]
  have h := sumUpper_weighted (fun v : List.Vector α L => misVar O F kl kh (FreeMonoid.ofList v.toList))
    Finset.univ (fun v => T.real {FreeMonoid.ofList v.toList}) (misMean O F kl kh T) t a hmeas hind
    hIcc hw (le_of_eq (misMean_eq O F kl kh hT).symm) ht hwa
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) h
  simp only [Set.mem_ofPred_eq] at hω ⊢
  rwa [misFrac_eq O F kl kh hT] at hω

/-- The families of at most `FV` strings of length `Ls`. -/
def smallFamilies (Ls FV : ℕ) : Finset (Finset (FreeMonoid α)) :=
  (Finset.range (FV + 1)).biUnion fun j =>
    ((Finset.univ : Finset (List.Vector α Ls)).image fun v => FreeMonoid.ofList v.toList).powersetCard j

omit [DecidableEq α] in
lemma mem_smallFamilies {Ls FV : ℕ} {F : Finset (FreeMonoid α)}
    (hF : ∀ v ∈ F, v.toList.length = Ls) (hc : F.card ≤ FV) : F ∈ smallFamilies Ls FV := by
  refine Finset.mem_biUnion.2 ⟨F.card, Finset.mem_range.2 (by omega),
    Finset.mem_powersetCard.2 ⟨fun v hv => ?_, rfl⟩⟩
  refine Finset.mem_image.2 ⟨⟨v.toList, hF v hv⟩, Finset.mem_univ _, ?_⟩
  simp

omit [DecidableEq α] in
lemma card_smallFamilies_le (Ls FV : ℕ) :
    ((smallFamilies (α := α) Ls FV).card : ℝ) ≤ (FV + 1) * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV := by
  classical
  set U := (Finset.univ : Finset (List.Vector α Ls)).image fun v => FreeMonoid.ofList v.toList
  have hU : U.card ≤ Fintype.card α ^ Ls :=
    Finset.card_image_le.trans (by simp [card_vector])
  have h1 : (smallFamilies (α := α) Ls FV).card ≤ ∑ j ∈ Finset.range (FV + 1), U.card.choose j :=
    Finset.card_biUnion_le.trans (le_of_eq (Finset.sum_congr rfl fun j _ =>
      Finset.card_powersetCard _ _))
  have h2 : ∀ j ∈ Finset.range (FV + 1), (U.card.choose j : ℝ)
      ≤ ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV := by
    intro j hj
    have hj' : j ≤ FV := by simp at hj; omega
    calc (U.card.choose j : ℝ) ≤ (U.card : ℝ) ^ j := by exact_mod_cast Nat.choose_le_pow _ _
      _ ≤ ((Fintype.card α : ℝ) ^ Ls + 1) ^ j := by
          gcongr
          have : (U.card : ℝ) ≤ (Fintype.card α : ℝ) ^ Ls := by exact_mod_cast hU
          linarith
      _ ≤ ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV :=
          pow_le_pow_right₀ (by have : (0 : ℝ) ≤ (Fintype.card α : ℝ) ^ Ls := by positivity
                                linarith) hj'
  calc ((smallFamilies (α := α) Ls FV).card : ℝ)
      ≤ ∑ j ∈ Finset.range (FV + 1), (U.card.choose j : ℝ) := by exact_mod_cast h1
    _ ≤ ∑ _j ∈ Finset.range (FV + 1), ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV :=
        Finset.sum_le_sum h2
    _ = (FV + 1) * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; push_cast; ring

omit [DecidableEq α] in
/-- Some family of at most `FV` strings of length `Ls`, at some cut, misreads more than its mean
plus `t` of `T` with chance at most `2 (FV + 1)³ (|Σ|^Ls + 1)^FV exp(-2t²/a)`. -/
theorem misread_union_le (O : Oracle μ (FreeMonoid α)) {T : Measure (FreeMonoid α)}
    [IsProbabilityMeasure T] {L : ℕ} (hT : T {x | x.toList.length ≠ L} = 0) {a : ℝ}
    (hatom : ∀ v : List.Vector α L, T.real {FreeMonoid.ofList v.toList} ≤ a) (Ls FV : ℕ) {t : ℝ}
    (ht : 0 ≤ t) :
    μ.real {ω | ∃ (F : Finset (FreeMonoid α)) (kl kh : ℕ), (∀ v ∈ F, v.toList.length = Ls)
      ∧ F.card ≤ FV ∧ misMean O F kl kh T + t < misFrac O F kl kh T ω}
      ≤ 2 * (FV + 1) ^ 3 * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV * Real.exp (-2 * t ^ 2 / a) := by
  classical
  set I := smallFamilies (α := α) Ls FV ×ˢ (Finset.range (FV + 1) ×ˢ Finset.range (FV + 2))
  have hsub : {ω | ∃ (F : Finset (FreeMonoid α)) (kl kh : ℕ), (∀ v ∈ F, v.toList.length = Ls)
      ∧ F.card ≤ FV ∧ misMean O F kl kh T + t < misFrac O F kl kh T ω}
      ⊆ ⋃ q ∈ I, {ω | misMean O q.1 q.2.1 q.2.2 T + t ≤ misFrac O q.1 q.2.1 q.2.2 T ω} := by
    rintro ω ⟨F, kl, kh, hF, hc, hlt⟩
    refine Set.mem_biUnion (x := (F, min kl F.card, min kh (F.card + 1)))
      (Finset.mem_product.2 ⟨mem_smallFamilies hF hc, Finset.mem_product.2
        ⟨Finset.mem_range.2 (Nat.lt_succ_of_le ((min_le_right _ _).trans hc)),
          Finset.mem_range.2 (Nat.lt_succ_of_le ((min_le_right _ _).trans (by omega)))⟩⟩) ?_
    simp only [Set.mem_ofPred_eq]
    rw [← misMean_clamp, ← misFrac_clamp]
    exact hlt.le
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans ?_)
  have hexp : 0 ≤ Real.exp (-2 * t ^ 2 / a) := (Real.exp_pos _).le
  calc ∑ q ∈ I, μ.real {ω | misMean O q.1 q.2.1 q.2.2 T + t ≤ misFrac O q.1 q.2.1 q.2.2 T ω}
      ≤ ∑ _q ∈ I, Real.exp (-2 * t ^ 2 / a) :=
        Finset.sum_le_sum fun q _ => misFrac_tail O q.1 q.2.1 q.2.2 hT hatom ht
    _ = I.card * Real.exp (-2 * t ^ 2 / a) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ 2 * (FV + 1) ^ 3 * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV
        * Real.exp (-2 * t ^ 2 / a) := by
        refine mul_le_mul_of_nonneg_right ?_ hexp
        have hI : (I.card : ℝ) = (smallFamilies (α := α) Ls FV).card * ((FV + 1) * (FV + 2)) := by
          simp only [I, Finset.card_product, Finset.card_range]; push_cast; ring
        rw [hI]
        have h1 := card_smallFamilies_le (α := α) Ls FV
        have h2 : (0 : ℝ) ≤ ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV := by positivity
        have h3 : ((FV : ℝ) + 1) * (FV + 2) ≤ (FV + 1) ^ 2 * 2 := by nlinarith
        calc ((smallFamilies (α := α) Ls FV).card : ℝ) * ((FV + 1) * (FV + 2))
            ≤ (FV + 1) * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV * ((FV + 1) ^ 2 * 2) :=
              mul_le_mul h1 h3 (by positivity) (by positivity)
          _ = 2 * (FV + 1) ^ 3 * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV := by ring

end OrthoDFA

end
