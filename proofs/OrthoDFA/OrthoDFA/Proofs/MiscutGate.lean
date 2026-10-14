import OrthoDFA.Proofs.Adaptive

/-!
# The gate's miscut bound is sound

`certifies` bounds how much of a population the family's cut misclassifies without knowing the
noise rates.  On a side of the cut, the seed reads against the side at `r_p` where the cut is
right and `1 − r_p` where it is wrong, so the excess of its count over `∑ r_p` is the miscut, at
least `1 − 2η` each.  `∑ r_p` itself is bounded below by an accept-preserving reference, whose
read of `p · v` disagrees with the seed's read of `p` at `2r_p(1 − r_p)`: `disagreeRate` is
convex, so the rates' sum is at least the side's size times `disagreeRate` of their mean.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J] {rule : Clusterer S}

/-! ### Independence across disjoint blocks -/

omit [IsProbabilityMeasure μ] in
open scoped Classical in
/-- σ-algebras independent one per string stay independent when grouped into disjoint blocks. -/
theorem iIndep_blocks {ι : Type*} (m : S → MeasurableSpace Ω) (hle : ∀ w, m w ≤ ‹_›)
    (hind : iIndep m μ) (blk : ι → Finset S)
    (hdisj : ∀ i j, i ≠ j → Disjoint (blk i) (blk j)) :
    iIndep (fun i => ⨆ w ∈ (blk i : Set S), m w) μ := by
  classical
  set π : ι → Set (Set Ω) := fun i =>
    piiUnionInter (fun w => {s | MeasurableSet[m w] s}) (blk i : Set S) with hπ
  refine iIndepSets.iIndep (fun i => iSup₂_le (fun w _ => hle w)) π
    (fun i => isPiSystem_piiUnionInter _
      (fun w => @MeasurableSpace.isPiSystem_measurableSet Ω (m w)) _)
    (fun i => (generateFrom_piiUnionInter_measurableSet m _).symm) ?_
  rw [iIndepSets_iff]
  intro t f hf
  have hex : ∀ i, ∃ q : Finset S, ∃ g : S → Set Ω, (i ∈ t → (↑q ⊆ (blk i : Set S)
      ∧ (∀ w ∈ q, MeasurableSet[m w] (g w)) ∧ f i = ⋂ w ∈ q, g w)) := by
    intro i
    by_cases hi : i ∈ t
    · obtain ⟨q, hq, g, hg, heq⟩ := hf i hi
      exact ⟨q, g, fun _ => ⟨hq, hg, heq⟩⟩
    · exact ⟨∅, fun _ => Set.univ, fun h => absurd h hi⟩
  choose q g hqg using hex
  set U : Finset S := t.biUnion q with hU
  set G : S → Set Ω := fun w => ⋂ i ∈ t.filter (fun i => w ∈ q i), g i w with hG
  have hGm : ∀ w ∈ U, MeasurableSet[m w] (G w) := by
    intro w _
    refine Finset.measurableSet_biInter _ (fun i hi => ?_)
    obtain ⟨hit, hwq⟩ := Finset.mem_filter.1 hi
    exact (hqg i hit).2.1 w hwq
  have hown : ∀ i ∈ t, ∀ w ∈ q i, G w = g i w := by
    intro i hi w hw
    have hone : t.filter (fun i' => w ∈ q i') = {i} := by
      ext i'
      simp only [Finset.mem_filter, Finset.mem_singleton]
      constructor
      · rintro ⟨hi', hw'⟩
        by_contra hne
        have hb := (hqg i hi).1 hw
        have hb' := (hqg i' hi').1 hw'
        exact Finset.disjoint_left.1 (hdisj i' i hne) hb' hb
      · rintro rfl
        exact ⟨hi, hw⟩
    simp only [hG, hone, Finset.mem_singleton, Set.iInter_iInter_eq_left]
  have hset : (⋂ i ∈ t, f i) = ⋂ w ∈ U, G w := by
    ext ω
    simp only [Set.mem_iInter, hU, Finset.mem_biUnion, hG, Finset.mem_filter]
    constructor
    · intro h w _ i ⟨hi, hw⟩
      have := h i hi
      rw [(hqg i hi).2.2] at this
      exact Set.mem_iInter₂.1 this w hw
    · intro h i hi
      rw [(hqg i hi).2.2]
      exact Set.mem_iInter₂.2 (fun w hw => h w ⟨i, hi, hw⟩ i ⟨hi, hw⟩)
  have hprodU : μ (⋂ w ∈ U, G w) = ∏ w ∈ U, μ (G w) :=
    hind.meas_biInter (fun w hw => hGm w hw)
  have hpd : (↑t : Set ι).PairwiseDisjoint q := by
    intro i hi i' hi' hne
    exact Finset.disjoint_of_subset_left (fun w hw => (hqg i hi).1 hw)
      (Finset.disjoint_of_subset_right (fun w hw => (hqg i' hi').1 hw) (hdisj i i' hne))
  rw [hset, hprodU, hU, Finset.prod_biUnion hpd]
  refine Finset.prod_congr rfl (fun i hi => ?_)
  rw [(hqg i hi).2.2, hind.meas_biInter (fun w hw => (hqg i hi).2.1 w hw)]
  exact Finset.prod_congr rfl (fun w hw => by rw [hown i hi w hw])

omit [IsProbabilityMeasure μ] in
/-- Reads that each depend only on their own block of the oracle's strings are independent when
the blocks are disjoint. -/
theorem iIndepFun_of_blocks (O : Oracle μ S) {ι : Type*} (blk : ι → Finset S)
    (hdisj : ∀ i j, i ≠ j → Disjoint (blk i) (blk j)) (Y : ι → Ω → ℝ)
    (hY : ∀ i, Measurable[noiseAlg O (blk i : Set S)] (Y i)) :
    iIndepFun Y μ := by
  rw [iIndepFun_iff_iIndep]
  have hnoise : iIndep (fun w => MeasurableSpace.comap (O.noise w) inferInstance) μ :=
    (iIndepFun_iff_iIndep _ _ _).1 O.noise_indep
  have hb := iIndep_blocks (μ := μ) _ (fun w => (O.noise_meas w).comap_le) hnoise blk hdisj
  refine iIndep_of_iIndep_of_le hb (fun i => ?_)
  exact (hY i).comap_le

/-! ### `disagreeRate` -/

lemma disagreeRate_mono : Monotone disagreeRate := by
  intro x y hxy
  unfold disagreeRate
  have := Real.sqrt_le_sqrt (show 1 - 2 * y ≤ 1 - 2 * x by linarith)
  linarith

/-- The Jensen step, from one tangent per term: with `u = √(1 − 2x̄)` and `v = 1 − 2r`,

    (1 − v)/2 ≥ (1 − u)/2 + (u² − v²)/(4u)

is `(u − v)² ≥ 0`, and the second terms sum to zero at the mean. -/
theorem card_mul_disagreeRate_le {ι : Type*} (A : Finset ι) (r : ι → ℝ)
    (hr : ∀ i ∈ A, r i < 1 / 2) :
    (A.card : ℝ) * disagreeRate ((∑ i ∈ A, 2 * r i * (1 - r i)) / A.card) ≤ ∑ i ∈ A, r i := by
  rcases A.eq_empty_or_nonempty with rfl | hA
  · simp
  have hn : (0 : ℝ) < A.card := by exact_mod_cast hA.card_pos
  set x : ℝ := (∑ i ∈ A, 2 * r i * (1 - r i)) / A.card with hx
  have hsum : ∑ i ∈ A, 2 * r i * (1 - r i) = A.card * x := by
    rw [hx]; field_simp
  have hlt : ∑ i ∈ A, 2 * r i * (1 - r i) < ∑ _i ∈ A, (1 / 2 : ℝ) :=
    Finset.sum_lt_sum_of_nonempty hA (fun i hi => by nlinarith [hr i hi, sq_nonneg (1 - 2 * r i)])
  rw [Finset.sum_const, nsmul_eq_mul, hsum] at hlt
  have hx2 : x < 1 / 2 := by nlinarith
  set u : ℝ := Real.sqrt (1 - 2 * x) with hu
  have hu0 : 0 < u := Real.sqrt_pos.2 (by linarith)
  have hu2 : u ^ 2 = 1 - 2 * x := Real.sq_sqrt (by linarith)
  have htan : ∀ i ∈ A, (1 - u) / 2 + (u ^ 2 - (1 - 2 * r i) ^ 2) / (4 * u) ≤ r i := by
    intro i _
    rw [div_add_div _ _ (by norm_num : (2 : ℝ) ≠ 0) (by positivity), div_le_iff₀ (by positivity)]
    nlinarith [sq_nonneg (u - (1 - 2 * r i))]
  have hsq : ∑ i ∈ A, (u ^ 2 - (1 - 2 * r i) ^ 2) = 0 := by
    have : ∀ i ∈ A, u ^ 2 - (1 - 2 * r i) ^ 2 = u ^ 2 - 1 + 2 * (2 * r i * (1 - r i)) :=
      fun i _ => by ring
    rw [Finset.sum_congr rfl this, Finset.sum_add_distrib, ← Finset.mul_sum, hsum,
      Finset.sum_const, nsmul_eq_mul, hu2]
    ring
  have hsum2 := Finset.sum_le_sum htan
  have h2 : ∑ i ∈ A, (u ^ 2 - (1 - 2 * r i) ^ 2) / (4 * u) = 0 := by
    rw [← Finset.sum_div, hsq, zero_div]
  rw [Finset.sum_add_distrib, h2, add_zero, Finset.sum_const, nsmul_eq_mul] at hsum2
  unfold disagreeRate
  rw [← hu]
  linarith

/-! ### One side of the cut -/

/-- A read is accept with chance `r + (1 − 2r)ℓ`, `r` the string's rate. -/
lemma prob_mq_one (O : Oracle μ S) (w : S) :
    μ.real {ω | O.mq w ω = 1} = O.rate w + (1 - 2 * O.rate w) * O.label w := by
  have hm : MeasurableSet {ω | O.mq w ω = 1} :=
    measurableSet_eq_fun (mq_meas O w) measurable_const
  have hae : O.mq w =ᵐ[μ] {ω | O.mq w ω = 1}.indicator 1 := by
    filter_upwards [mq_bit O w] with ω hω
    rcases hω with h | h
    · rw [Set.indicator_of_notMem (by simp [h]), h]
    · rw [Set.indicator_of_mem (by simp [h]), h]; rfl
  rw [← mq_mean O w, integral_congr_ae hae, integral_indicator_one hm]

lemma prob_mq_ne_one (O : Oracle μ S) (w : S) :
    μ.real {ω | ¬ O.mq w ω = 1} = 1 - (O.rate w + (1 - 2 * O.rate w) * O.label w) := by
  have hm : MeasurableSet {ω | O.mq w ω = 1} :=
    measurableSet_eq_fun (mq_meas O w) measurable_const
  rw [show {ω | ¬ O.mq w ω = 1} = {ω | O.mq w ω = 1}ᶜ from rfl, measureReal_compl hm,
    prob_mq_one, probReal_univ]

open scoped Classical in
/-- Whether the seed's read of `p` goes against the class `c` the cut gives its side. -/
noncomputable def againstVar (O : Oracle μ S) (c : ℝ) (p : S) (ω : Ω) : ℝ :=
  if (O.mq p ω = 1 ↔ c = 1) then 0 else 1

open scoped Classical in
/-- Whether a reference `v`'s read of `p · v` disagrees with the seed's read of `p`. -/
noncomputable def apartVar (O : Oracle μ S) (v p : S) (ω : Ω) : ℝ :=
  if ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)) then 0 else 1

lemma measurable_noise_of_mem (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T) :
    Measurable[noiseAlg O T] (O.noise w) := fun _ hs => measurableSet_noise_preimage O hw hs

lemma measurable_mq_of_mem (O : Oracle μ S) {T : Set S} {w : S} (hw : w ∈ T) :
    Measurable[noiseAlg O T] (O.mq w) := by
  have h := measurable_noise_of_mem O hw
  exact (measurable_const.add (measurable_const.mul measurable_id)).comp h

lemma againstVar_meas_alg (O : Oracle μ S) (c : ℝ) {T : Set S} {p : S} (hp : p ∈ T) :
    Measurable[noiseAlg O T] (againstVar O c p) := by
  classical
  have hset : MeasurableSet[noiseAlg O T] {ω | O.mq p ω = 1} :=
    measurableSet_mq_eq_one O hp
  have : againstVar O c p = fun ω => if ω ∈ {ω | O.mq p ω = 1} then
      (if c = 1 then 0 else 1) else (if c = 1 then 1 else 0) := by
    funext ω; unfold againstVar; by_cases h : O.mq p ω = 1 <;> by_cases hc : c = 1 <;> simp [h, hc]
  rw [this]
  exact Measurable.ite hset measurable_const measurable_const

lemma apartVar_meas_alg (O : Oracle μ S) {T : Set S} {v p : S} (hp : p ∈ T) (hpv : p * v ∈ T) :
    Measurable[noiseAlg O T] (apartVar O v p) := by
  classical
  have h1 : MeasurableSet[noiseAlg O T] {ω | O.mq (p * v) ω = 1} := measurableSet_mq_eq_one O hpv
  have h2 : MeasurableSet[noiseAlg O T] {ω | O.mq p ω = 1} := measurableSet_mq_eq_one O hp
  have : apartVar O v p = fun ω => if ω ∈ ({ω | O.mq (p * v) ω = 1} ∩ {ω | O.mq p ω = 1})
      ∪ ({ω | O.mq (p * v) ω = 1}ᶜ ∩ {ω | O.mq p ω = 1}ᶜ) then 0 else 1 := by
    funext ω; unfold apartVar
    by_cases ha : O.mq (p * v) ω = 1 <;> by_cases hb : O.mq p ω = 1 <;> simp [ha, hb]
  rw [this]
  exact Measurable.ite ((h1.inter h2).union (h1.compl.inter h2.compl)) measurable_const
    measurable_const

omit [IsProbabilityMeasure μ] in
lemma againstVar_icc (O : Oracle μ S) (c : ℝ) (p : S) (ω : Ω) :
    againstVar O c p ω ∈ Set.Icc (0 : ℝ) 1 := by
  unfold againstVar; split_ifs <;> norm_num

omit [IsProbabilityMeasure μ] in
lemma apartVar_icc (O : Oracle μ S) (v p : S) (ω : Ω) :
    apartVar O v p ω ∈ Set.Icc (0 : ℝ) 1 := by
  unfold apartVar; split_ifs <;> norm_num

/-- The seed reads against its side at its rate where the cut is right, and at one less it
where the cut is wrong. -/
lemma againstVar_mean (O : Oracle μ S) {c : ℝ} (hc : c = 0 ∨ c = 1) (p : S) :
    μ[againstVar O c p] = if O.label p = c then O.rate p else 1 - O.rate p := by
  classical
  have hm : MeasurableSet {ω | O.mq p ω = 1} :=
    measurableSet_eq_fun (mq_meas O p) measurable_const
  rcases hc with rfl | rfl
  · have : againstVar O 0 p = {ω | O.mq p ω = 1}.indicator 1 := by
      funext ω; unfold againstVar; by_cases h : O.mq p ω = 1 <;> simp [h]
    rw [this, integral_indicator_one hm, prob_mq_one]
    rcases O.label_bit p with hl | hl <;> rw [hl] <;> norm_num <;> ring
  · have : againstVar O 1 p = {ω | O.mq p ω = 1}ᶜ.indicator 1 := by
      funext ω; unfold againstVar; by_cases h : O.mq p ω = 1 <;> simp [h]
    rw [this, integral_indicator_one hm.compl, show {ω | O.mq p ω = 1}ᶜ = {ω | ¬ O.mq p ω = 1}
      from rfl, prob_mq_ne_one]
    rcases O.label_bit p with hl | hl <;> rw [hl] <;> norm_num <;> ring

/-- An accept-preserving reference disagrees with the seed at `2r(1 − r)`. -/
lemma apartVar_mean (O : Oracle μ S) {v p : S} (hap : O.label (p * v) = O.label p)
    (hne : p * v ≠ p) :
    μ[apartVar O v p] = 2 * O.rate p * (1 - O.rate p) := by
  classical
  set a : Set Ω := {ω | O.mq (p * v) ω = 1} with ha
  set b : Set Ω := {ω | O.mq p ω = 1} with hb
  have ham : MeasurableSet a := measurableSet_eq_fun (mq_meas O _) measurable_const
  have hbm : MeasurableSet b := measurableSet_eq_fun (mq_meas O _) measurable_const
  have hind : IndepFun (O.mq (p * v)) (O.mq p) μ := (mq_indep O).indepFun hne
  have hprod : ∀ s t : Set ℝ, MeasurableSet s → MeasurableSet t →
      μ.real (O.mq (p * v) ⁻¹' s ∩ O.mq p ⁻¹' t)
        = μ.real (O.mq (p * v) ⁻¹' s) * μ.real (O.mq p ⁻¹' t) := by
    intro s t hs ht
    simp only [measureReal_def,
      hind.measure_inter_preimage_eq_mul s t hs ht, ENNReal.toReal_mul]
  have h1 := hprod {1} {1}ᶜ (measurableSet_singleton 1) (measurableSet_singleton 1).compl
  have h2 := hprod {1}ᶜ {1} (measurableSet_singleton 1).compl (measurableSet_singleton 1)
  have hset : apartVar O v p = ((a ∩ bᶜ) ∪ (aᶜ ∩ b)).indicator 1 := by
    funext ω; unfold apartVar
    by_cases h1 : O.mq (p * v) ω = 1 <;> by_cases h2 : O.mq p ω = 1 <;> simp [ha, hb, h1, h2]
  have hdisj : Disjoint (a ∩ bᶜ) (aᶜ ∩ b) :=
    Set.disjoint_left.2 (fun ω h h' => h'.1 h.1)
  rw [hset, integral_indicator_one ((ham.inter hbm.compl).union (ham.compl.inter hbm)),
    measureReal_union hdisj (ham.compl.inter hbm)]
  have e1 : a ∩ bᶜ = O.mq (p * v) ⁻¹' {1} ∩ O.mq p ⁻¹' {1}ᶜ := rfl
  have e2 : aᶜ ∩ b = O.mq (p * v) ⁻¹' {1}ᶜ ∩ O.mq p ⁻¹' {1} := rfl
  rw [e1, e2, h1, h2]
  have pa : μ.real (O.mq (p * v) ⁻¹' {1}) = O.rate p + (1 - 2 * O.rate p) * O.label p := by
    rw [show O.mq (p * v) ⁻¹' {1} = {ω | O.mq (p * v) ω = 1} from rfl, prob_mq_one,
      hap, O.rate_eq_of_label_eq hap]
  have pb : μ.real (O.mq p ⁻¹' {1}) = O.rate p + (1 - 2 * O.rate p) * O.label p := prob_mq_one O p
  have pa' : μ.real (O.mq (p * v) ⁻¹' {1}ᶜ) = 1 - μ.real (O.mq (p * v) ⁻¹' {1}) := by
    rw [Set.preimage_compl, measureReal_compl (measurableSet_singleton 1 |>.preimage
      (mq_meas O _)), probReal_univ]
  have pb' : μ.real (O.mq p ⁻¹' {1}ᶜ) = 1 - μ.real (O.mq p ⁻¹' {1}) := by
    rw [Set.preimage_compl, measureReal_compl (measurableSet_singleton 1 |>.preimage
      (mq_meas O _)), probReal_univ]
  rw [pa', pb', pa, pb]
  rcases O.label_bit p with hl | hl <;> rw [hl] <;> ring

lemma exp_hoeffDev (n : ℕ) (hn : 0 < n) {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    Real.exp (-2 * (n : ℝ) * (hoeffDev n a / n) ^ 2) = a := by
  have hl : 0 ≤ Real.log (1 / a) := Real.log_nonneg (by rw [le_div_iff₀ ha0]; linarith)
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hsq : hoeffDev n a ^ 2 = n * Real.log (1 / a) / 2 :=
    Real.sq_sqrt (by positivity)
  rw [div_pow, hsq, show -2 * (n : ℝ) * (n * Real.log (1 / a) / 2 / n ^ 2) = - Real.log (1 / a)
    by field_simp, Real.exp_neg, Real.exp_log (by positivity), one_div, inv_inv]

omit [IsProbabilityMeasure μ] in
open scoped Classical in
lemma card_filter_eq_sum_againstVar (O : Oracle μ S) (c : ℝ) (A : Finset S) (ω : Ω) :
    ((A.filter (fun p => ¬ ((O.mq p ω = 1) ↔ c = 1))).card : ℝ)
      = ∑ p ∈ A, againstVar O c p ω := by
  rw [Finset.card_filter, Nat.cast_sum]
  refine Finset.sum_congr rfl (fun p _ => ?_)
  unfold againstVar; split_ifs <;> simp_all

omit [IsProbabilityMeasure μ] in
open scoped Classical in
lemma card_filter_eq_sum_apartVar (O : Oracle μ S) (v : S) (A : Finset S) (ω : Ω) :
    ((A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ)
      = ∑ p ∈ A, apartVar O v p ω := by
  rw [Finset.card_filter, Nat.cast_sum]
  refine Finset.sum_congr rfl (fun p _ => ?_)
  unfold apartVar; split_ifs <;> simp_all

open scoped Classical in
/-- One side of a fixed cut.  Unless the seed's count against the side falls a deviation below
its mean, or an accept-preserving reference's count of disagreements rises a deviation above its
mean, `sideBound` holds for every count of disagreements up to the reference's.  Each fails with
chance at most `a`. -/
theorem side_whp (O : Oracle μ S) (A : Finset S) {c : ℝ} (hc : c = 0 ∨ c = 1) {v : S}
    (hap : ∀ p, O.label (p * v) = O.label p) (hne : ∀ p ∈ A, p * v ≠ p)
    (hblk : ∀ p ∈ A, ∀ p' ∈ A, p ≠ p' → Disjoint ({p, p * v} : Finset S) {p', p' * v})
    {η₀ a : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (ha0 : 0 < a) (ha1 : a ≤ 1) :
    μ.real {ω | ¬ ∀ b : ℕ,
        (b : ℝ) ≤ ((A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ) →
        ((A.filter (fun p => O.label p ≠ c)).card : ℝ) * (1 - 2 * η₀)
          ≤ sideBound A.card (A.filter (fun p => ¬ ((O.mq p ω = 1) ↔ c = 1))).card b a}
      ≤ 2 * a := by
  classical
  rcases Nat.eq_zero_or_pos A.card with hA0 | hApos
  · have hAe : A = ∅ := Finset.card_eq_zero.1 hA0
    have hz : {ω | ¬ ∀ b : ℕ,
        (b : ℝ) ≤ ((A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ) →
        ((A.filter (fun p => O.label p ≠ c)).card : ℝ) * (1 - 2 * η₀)
          ≤ sideBound A.card (A.filter (fun p => ¬ ((O.mq p ω = 1) ↔ c = 1))).card b a}
        = (∅ : Set Ω) := by
      ext ω
      simp [hAe, sideBound, hoeffDev]
    rw [hz, measureReal_empty]
    positivity
  set n : ℕ := A.card with hndef
  have hnR : (0 : ℝ) < n := by exact_mod_cast hApos
  set d : ℝ := hoeffDev n a with hd
  set X : A → Ω → ℝ := fun i => againstVar O c i.1 with hX
  set Y : A → Ω → ℝ := fun i => apartVar O v i.1 with hY
  have hXind : iIndepFun X μ := by
    refine iIndepFun_of_blocks O (fun i : A => ({i.1} : Finset S)) ?_ X (fun i => ?_)
    · intro i j hij
      simp only [Finset.disjoint_singleton]
      exact fun h => hij (Subtype.ext h)
    · exact againstVar_meas_alg O c (by simp)
  have hYind : iIndepFun Y μ := by
    refine iIndepFun_of_blocks O (fun i : A => ({i.1, i.1 * v} : Finset S)) ?_ Y (fun i => ?_)
    · intro i j hij
      exact hblk i.1 i.2 j.1 j.2 (fun h => hij (Subtype.ext h))
    · exact apartVar_meas_alg O (by simp) (by simp)
  set EX : ℝ := ∑ p ∈ A, μ[againstVar O c p] with hEX
  set EY : ℝ := ∑ p ∈ A, μ[apartVar O v p] with hEY
  have hsumX : ∀ ω, ∑ i : A, X i ω = ∑ p ∈ A, againstVar O c p ω := fun ω =>
    Finset.sum_coe_sort A (fun p => againstVar O c p ω)
  have hsumY : ∀ ω, ∑ i : A, Y i ω = ∑ p ∈ A, apartVar O v p ω := fun ω =>
    Finset.sum_coe_sort A (fun p => apartVar O v p ω)
  have hcard : ((Finset.univ : Finset A).card : ℝ) = n := by simp [hndef]
  have hEXs : ∑ i : A, μ[X i] = EX := Finset.sum_coe_sort A (fun p => μ[againstVar O c p])
  have hEYs : ∑ i : A, μ[Y i] = EY := Finset.sum_coe_sort A (fun p => μ[apartVar O v p])
  have hB1 : μ.real {ω | ∑ i : A, X i ω ≤ (n : ℝ) * (EX / n - d / n)} ≤ a := by
    have h := sumLower_le X Finset.univ (EX / n) (d / n)
      (fun i => (againstVar_meas_alg O c (T := Set.univ) (Set.mem_univ _)).mono
        (noiseAlg_le O _) le_rfl |>.aemeasurable)
      hXind (fun i => Filter.Eventually.of_forall (fun ω => againstVar_icc O c _ ω))
      (by rw [hcard, hEXs, mul_div_cancel₀ _ hnR.ne'])
      (div_nonneg (Real.sqrt_nonneg _) hnR.le)
    rw [hcard, exp_hoeffDev n hApos ha0 ha1] at h
    exact h
  have hB2 : μ.real {ω | (n : ℝ) * (EY / n + d / n) ≤ ∑ i : A, Y i ω} ≤ a := by
    have h := sumUpper_le Y Finset.univ (EY / n) (d / n)
      (fun i => (apartVar_meas_alg O (T := Set.univ) (Set.mem_univ _) (Set.mem_univ _)).mono
        (noiseAlg_le O _) le_rfl |>.aemeasurable)
      hYind (fun i => Filter.Eventually.of_forall (fun ω => apartVar_icc O v _ ω))
      (by rw [hcard, hEYs, mul_div_cancel₀ _ hnR.ne'])
      (div_nonneg (Real.sqrt_nonneg _) hnR.le)
    rw [hcard, exp_hoeffDev n hApos ha0 ha1] at h
    exact h
  -- the deterministic step
  have hr : ∀ p ∈ A, O.rate p < 1 / 2 := fun p _ =>
    lt_of_le_of_lt (le_trans (O.rate_le_eta p) hη) hη₀
  have hEYeq : EY = ∑ p ∈ A, 2 * O.rate p * (1 - O.rate p) :=
    Finset.sum_congr rfl (fun p hp => apartVar_mean O (hap p) (hne p hp))
  have hEXge : ∑ p ∈ A, O.rate p
      + ((A.filter (fun p => O.label p ≠ c)).card : ℝ) * (1 - 2 * η₀) ≤ EX := by
    have hEXeq : EX = ∑ p ∈ A, (if O.label p = c then O.rate p else 1 - O.rate p) :=
      Finset.sum_congr rfl (fun p _ => againstVar_mean O hc p)
    rw [hEXeq, ← Finset.sum_filter_add_sum_filter_not A (fun p => O.label p = c)
        (fun p => if O.label p = c then O.rate p else 1 - O.rate p),
      ← Finset.sum_filter_add_sum_filter_not A (fun p => O.label p = c) O.rate]
    rw [Finset.sum_congr rfl (fun p hp => if_pos (Finset.mem_filter.1 hp).2),
      Finset.sum_congr rfl (fun p hp => if_neg (Finset.mem_filter.1 hp).2)]
    have hk : ((A.filter (fun p => O.label p ≠ c)).card : ℝ) * (1 - 2 * η₀)
        ≤ ∑ p ∈ A.filter (fun p => ¬ O.label p = c), (1 - 2 * O.rate p) := by
      have h := Finset.card_nsmul_le_sum (A.filter (fun p => ¬ O.label p = c))
        (fun p => 1 - 2 * O.rate p) (1 - 2 * η₀) (fun p _ => by linarith [O.rate_le_eta p])
      rw [nsmul_eq_mul] at h
      exact h
    have hsplit : ∑ p ∈ A.filter (fun p => ¬ O.label p = c), (1 - O.rate p)
        = ∑ p ∈ A.filter (fun p => ¬ O.label p = c), O.rate p
          + ∑ p ∈ A.filter (fun p => ¬ O.label p = c), (1 - 2 * O.rate p) := by
      rw [← Finset.sum_add_distrib]; exact Finset.sum_congr rfl (fun p _ => by ring)
    linarith
  have hjensen := card_mul_disagreeRate_le A O.rate hr
  rw [← hEYeq] at hjensen
  have hEY0 : 0 ≤ EY := by
    rw [hEYeq]
    exact Finset.sum_nonneg (fun p _ => by nlinarith [O.rate_nonneg p, hr p ‹_›])
  have hsub : {ω | ¬ ∀ b : ℕ,
        (b : ℝ) ≤ ((A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ) →
        ((A.filter (fun p => O.label p ≠ c)).card : ℝ) * (1 - 2 * η₀)
          ≤ sideBound A.card (A.filter (fun p => ¬ ((O.mq p ω = 1) ↔ c = 1))).card b a}
      ⊆ {ω | ∑ i : A, X i ω ≤ (n : ℝ) * (EX / n - d / n)}
        ∪ {ω | (n : ℝ) * (EY / n + d / n) ≤ ∑ i : A, Y i ω} := by
    intro ω hω
    by_contra hc'
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_le] at hc'
    obtain ⟨h1, h2⟩ := hc'
    apply hω
    intro b hb
    rw [hsumX] at h1
    rw [hsumY] at h2
    have e1 : (n : ℝ) * (EX / n - d / n) = EX - d := by field_simp
    have e2 : (n : ℝ) * (EY / n + d / n) = EY + d := by field_simp
    rw [e1] at h1
    rw [e2] at h2
    rw [card_filter_eq_sum_apartVar] at hb
    have hmax : max 0 ((b : ℝ) - d) ≤ EY := max_le hEY0 (by linarith)
    have hg : disagreeRate (max 0 ((b : ℝ) - d) / n) ≤ disagreeRate (EY / n) :=
      disagreeRate_mono (div_le_div_of_nonneg_right hmax hnR.le)
    unfold sideBound
    rw [card_filter_eq_sum_againstVar, ← hndef, ← hd]
    nlinarith [mul_le_mul_of_nonneg_left hg hnR.le]
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ a + a := add_le_add hB1 hB2
    _ = 2 * a := by ring

/-! ### Sides chosen by the votes -/

/-- `selection_read_bound` for a score that is any event the scored strings' bits decide, not
only a condition on which of them read accept. -/
theorem selection_congr_bound (O : Oracle μ S) (C Q : Finset S)
    (hdisj : Disjoint (↑C : Set S) (↑Q : Set S))
    {β : Type*} [DecidableEq β] (T : Finset β) (t₀ : β) (ht₀ : t₀ ∈ T)
    (sel : Ω → β) (hsel : ∀ ω, sel ω ∈ T)
    (hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel ω = sel ω')
    (Bad : β → Set Ω)
    (hBad : ∀ t ω ω', (∀ w ∈ C, O.noise w ω = O.noise w ω') → ω ∈ Bad t → ω' ∈ Bad t)
    (E : ℝ) (hE : 0 ≤ E) (hbad : ∀ t ∈ T, μ.real (Bad t) ≤ E) :
    μ.real {ω | ω ∈ Bad (sel ω)} ≤ E := by
  classical
  set Bad' : β → Set Ω := fun t => Bad t ∩ noiseClean O C with hBad'def
  have hmeasBadC : ∀ t, MeasurableSet[noiseAlg O ↑C] (Bad' t) := by
    intro t
    have h := measurableSet_side_clean O C (fun ω => decide (ω ∈ Bad t))
      (fun ω ω' h => by
        simp only [decide_eq_decide]
        exact ⟨hBad t ω ω' h, hBad t ω' ω (fun w hw => (h w hw).symm)⟩) true
    simpa [hBad'def] using h
  have hmeasBad : ∀ t, MeasurableSet (Bad' t) := fun t => noiseAlg_le O ↑C _ (hmeasBadC t)
  set sel' : Ω → β := fun ω => if ω ∈ noiseClean O Q then sel ω else t₀ with hsel'def
  have hsel2 : ∀ ω, sel' ω ∈ T := by
    intro ω
    rw [hsel'def]
    by_cases hc : ω ∈ noiseClean O Q
    · simpa [hc] using hsel ω
    · simpa [hc] using ht₀
  have hsplit : ∀ t, {ω | sel' ω = t}
      = ({ω | sel ω = t} ∩ noiseClean O Q) ∪ (if t = t₀ then (noiseClean O Q)ᶜ else ∅) := by
    intro t
    ext ω
    by_cases hc : ω ∈ noiseClean O Q <;> by_cases he : t = t₀ <;>
      simp [hsel'def, hc, he, eq_comm (a := t₀)]
  have hmeasSelQ : ∀ t, MeasurableSet[noiseAlg O ↑Q] {ω | sel' ω = t} := by
    intro t
    rw [hsplit t]
    refine MeasurableSet.union (measurableSet_side_clean O Q sel hcongr t) ?_
    split_ifs
    · exact (measurableSet_noiseClean O Q).compl
    · exact (noiseAlg O ↑Q).measurableSet_empty
  have hmeasSel : ∀ t, MeasurableSet {ω | sel' ω = t} := fun t =>
    noiseAlg_le O ↑Q _ (hmeasSelQ t)
  have hindep : ∀ t ∈ T,
      μ.real ({ω | sel' ω = t} ∩ Bad' t) = μ.real {ω | sel' ω = t} * μ.real (Bad' t) := by
    intro t _
    have hI := (indep_noiseAlg O hdisj.symm).indepSet_of_measurableSet (hmeasSelQ t)
      (hmeasBadC t)
    have := hI.measure_inter_eq_mul
    simp only [measureReal_def, this, ENNReal.toReal_mul]
  have hmain := measureReal_selection_le (μ := μ) T sel' hsel2 hmeasSel Bad' hmeasBad E
    hindep (fun t ht => le_trans (measureReal_mono Set.inter_subset_left (measure_ne_top _ _))
      (hbad t ht)) hE
  have hsub : {ω | ω ∈ Bad (sel ω)}
      ⊆ {ω | ω ∈ Bad' (sel' ω)} ∪ ((noiseClean O Q)ᶜ ∪ (noiseClean O C)ᶜ) := by
    intro ω hω
    by_cases hq : ω ∈ noiseClean O Q
    · by_cases hc : ω ∈ noiseClean O C
      · refine Or.inl ?_
        change ω ∈ Bad' (sel' ω)
        simp only [hsel'def, hq, if_pos]
        exact ⟨hω, hc⟩
      · exact Or.inr (Or.inr hc)
    · exact Or.inr (Or.inl hq)
  calc μ.real {ω | ω ∈ Bad (sel ω)}
      ≤ μ.real ({ω | ω ∈ Bad' (sel' ω)} ∪ ((noiseClean O Q)ᶜ ∪ (noiseClean O C)ᶜ)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ μ.real {ω | ω ∈ Bad' (sel' ω)} + (μ.real (noiseClean O Q)ᶜ + μ.real (noiseClean O C)ᶜ) :=
        le_trans (measureReal_union_le _ _) (add_le_add_right (measureReal_union_le _ _) _)
    _ = μ.real {ω | ω ∈ Bad' (sel' ω)} := by rw [noiseClean_ae O Q, noiseClean_ae O C]; ring
    _ ≤ E := hmain

open scoped Classical in
/-- The gate's count bound, with the certification sample `P`, the voters `F` and the references
`R` fixed and the cut chosen by the votes: the cut's wrong prefixes in `P`, at `1 − 2η₀` each,
exceed the two sides' bounds with chance at most `4a`. -/
theorem miscut_count_whp (O : Oracle μ S) (P F R : Finset S) (lo hi : ℕ) {v : S} (hvR : v ∈ R)
    (hap : ∀ p, O.label (p * v) = O.label p)
    (hinj : ∀ p ∈ P, ∀ p' ∈ P, ∀ u ∈ insert 1 (F ∪ R), ∀ u' ∈ insert 1 (F ∪ R),
      p * u = p' * u' → p = p' ∧ u = u')
    (hFR : Disjoint F R) (h1F : (1 : S) ∉ F) (h1R : (1 : S) ∉ R)
    {η₀ a : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (ha0 : 0 < a) (ha1 : a ≤ 1) :
    μ.real {ω | ¬ (((P.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card : ℝ) * (1 - 2 * η₀)
        ≤ sideBound (P.filter (fun p => hi < voteCount O.mq F p ω)).card
            ((P.filter (fun p => hi < voteCount O.mq F p ω)).filter
              (fun p => ¬ O.mq p ω = 1)).card
            (refApart O.mq (P.filter (fun p => hi < voteCount O.mq F p ω)) R ω) a
          + sideBound (P.filter (fun p => voteCount O.mq F p ω ≤ lo)).card
            ((P.filter (fun p => voteCount O.mq F p ω ≤ lo)).filter
              (fun p => O.mq p ω = 1)).card
            (refApart O.mq (P.filter (fun p => voteCount O.mq F p ω ≤ lo)) R ω) a)}
      ≤ 4 * a := by
  classical
  have hv1 : v ≠ 1 := fun h => h1R (h ▸ hvR)
  have hvmem : v ∈ insert 1 (F ∪ R) := Finset.mem_insert_of_mem (Finset.mem_union_right _ hvR)
  have h1mem : (1 : S) ∈ insert 1 (F ∪ R) := Finset.mem_insert_self _ _
  set C : Finset S := P ∪ (P ×ˢ R).image (fun q => q.1 * q.2) with hC
  set Q : Finset S := (P ×ˢ F).image (fun q => q.1 * q.2) with hQ
  have hdisj : Disjoint (↑C : Set S) (↑Q : Set S) := by
    rw [Finset.disjoint_coe]
    refine Finset.disjoint_left.2 (fun w hwC hwQ => ?_)
    obtain ⟨⟨p', u⟩, hpu, rfl⟩ := Finset.mem_image.1 hwQ
    obtain ⟨hp', hu⟩ := Finset.mem_product.1 hpu
    have humem : u ∈ insert 1 (F ∪ R) := Finset.mem_insert_of_mem (Finset.mem_union_left _ hu)
    rcases Finset.mem_union.1 hwC with hp | hw
    · have := (hinj _ hp p' hp' 1 h1mem u humem (by simpa using hp.symm ▸ (mul_one _).symm ▸ rfl)).2
      exact h1F (this ▸ hu)
    · obtain ⟨⟨p, r⟩, hpr, hpreq⟩ := Finset.mem_image.1 hw
      obtain ⟨hp, hr⟩ := Finset.mem_product.1 hpr
      have hrmem : r ∈ insert 1 (F ∪ R) := Finset.mem_insert_of_mem (Finset.mem_union_right _ hr)
      have := (hinj p hp p' hp' r hrmem u humem hpreq).2
      exact Finset.disjoint_left.1 hFR hu (this ▸ hr)
  set T : Finset (Finset S × Finset S) := P.powerset ×ˢ P.powerset with hT
  set sel : Ω → Finset S × Finset S := fun ω =>
    (P.filter (fun p => hi < voteCount O.mq F p ω), P.filter (fun p => voteCount O.mq F p ω ≤ lo))
    with hseldef
  have hsel : ∀ ω, sel ω ∈ T := fun ω =>
    Finset.mem_product.2 ⟨Finset.mem_powerset.2 (Finset.filter_subset _ _),
      Finset.mem_powerset.2 (Finset.filter_subset _ _)⟩
  have hcongr : ∀ ω ω', (∀ w ∈ Q, O.noise w ω = O.noise w ω') → sel ω = sel ω' := by
    intro ω ω' h
    have hvc : ∀ p ∈ P, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
      voteCount_congr O F p (fun u hu => by
        rw [mq_congr O (h _ (Finset.mem_image.2 ⟨(p, u), Finset.mem_product.2 ⟨hp, hu⟩, rfl⟩))])
    exact Prod.ext (Finset.filter_congr (fun p hp => by rw [hvc p hp]))
      (Finset.filter_congr (fun p hp => by rw [hvc p hp]))
  set bound : Finset S × Finset S → Ω → ℝ := fun t ω =>
    sideBound t.1.card (t.1.filter (fun p => ¬ O.mq p ω = 1)).card (refApart O.mq t.1 R ω) a
      + sideBound t.2.card (t.2.filter (fun p => O.mq p ω = 1)).card (refApart O.mq t.2 R ω) a
    with hbounddef
  set Bad : Finset S × Finset S → Set Ω := fun t =>
    if t.1 ⊆ P ∧ t.2 ⊆ P then
      {ω | ¬ ((((t.1.filter (fun p => O.label p ≠ 1)).card : ℝ)
        + ((t.2.filter (fun p => O.label p ≠ 0)).card : ℝ)) * (1 - 2 * η₀) ≤ bound t ω)}
    else ∅ with hBaddef
  have hBadcongr : ∀ t ω ω', (∀ w ∈ C, O.noise w ω = O.noise w ω') → ω ∈ Bad t → ω' ∈ Bad t := by
    intro t ω ω' h hω
    simp only [hBaddef] at hω ⊢
    split_ifs at hω ⊢ with ht
    · have hmq : ∀ p ∈ P, O.mq p ω = O.mq p ω' := fun p hp =>
        mq_congr O (h p (Finset.mem_union_left _ hp))
      have hmqR : ∀ p ∈ P, ∀ w ∈ R, O.mq (p * w) ω = O.mq (p * w) ω' := fun p hp w hw =>
        mq_congr O (h _ (Finset.mem_union_right _
          (Finset.mem_image.2 ⟨(p, w), Finset.mem_product.2 ⟨hp, hw⟩, rfl⟩)))
      have hap' : ∀ A ⊆ P, refApart O.mq A R ω = refApart O.mq A R ω' := by
        intro A hA
        unfold refApart
        split_ifs with hR
        · exact Finset.inf'_congr hR rfl (fun w hw => congrArg Finset.card
            (Finset.filter_congr (fun p hp => by rw [hmq p (hA hp), hmqR p (hA hp) w hw])))
        · rfl
      have hb : bound t ω = bound t ω' := by
        simp only [hbounddef]
        rw [hap' t.1 ht.1, hap' t.2 ht.2,
          Finset.filter_congr (fun p hp => by rw [hmq p (ht.1 hp)] :
            ∀ p ∈ t.1, (¬ O.mq p ω = 1) ↔ (¬ O.mq p ω' = 1)),
          Finset.filter_congr (fun p hp => by rw [hmq p (ht.2 hp)] :
            ∀ p ∈ t.2, (O.mq p ω = 1) ↔ (O.mq p ω' = 1))]
      simpa [hb] using hω
    · exact hω
  have hbad : ∀ t ∈ T, μ.real (Bad t) ≤ 4 * a := by
    rintro ⟨A₀, R₀⟩ ht
    obtain ⟨hA, hR⟩ := Finset.mem_product.1 ht
    have hAP := Finset.mem_powerset.1 hA
    have hRP := Finset.mem_powerset.1 hR
    have hside : ∀ D ⊆ P, (∀ p ∈ D, p * v ≠ p)
        ∧ ∀ p ∈ D, ∀ p' ∈ D, p ≠ p' → Disjoint ({p, p * v} : Finset S) {p', p' * v} := by
      intro D hD
      refine ⟨fun p hp heq => hv1 ?_, fun p hp p' hp' hne => ?_⟩
      · exact (hinj p (hD hp) p (hD hp) v hvmem 1 h1mem (by rw [mul_one]; exact heq)).2
      · rw [Finset.disjoint_left]
        intro w hw hw'
        simp only [Finset.mem_insert, Finset.mem_singleton] at hw hw'
        rcases hw with rfl | rfl <;> rcases hw' with h | h
        · exact hne h
        · exact hne (hinj _ (hD hp) p' (hD hp') 1 h1mem v hvmem (by rw [mul_one]; exact h)).1
        · exact hne (hinj p (hD hp) p' (hD hp') v hvmem 1 h1mem (by rw [mul_one]; exact h)).1
        · exact hne (hinj p (hD hp) p' (hD hp') v hvmem v hvmem h).1
    set E1 : Set Ω := {ω | ¬ ∀ b : ℕ,
        (b : ℝ) ≤ ((A₀.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ) →
        ((A₀.filter (fun p => O.label p ≠ 1)).card : ℝ) * (1 - 2 * η₀)
          ≤ sideBound A₀.card (A₀.filter (fun p => ¬ ((O.mq p ω = 1) ↔ (1 : ℝ) = 1))).card b a}
      with hE1
    set E2 : Set Ω := {ω | ¬ ∀ b : ℕ,
        (b : ℝ) ≤ ((R₀.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ) →
        ((R₀.filter (fun p => O.label p ≠ 0)).card : ℝ) * (1 - 2 * η₀)
          ≤ sideBound R₀.card (R₀.filter (fun p => ¬ ((O.mq p ω = 1) ↔ (0 : ℝ) = 1))).card b a}
      with hE2
    have hs1 : μ.real E1 ≤ 2 * a := side_whp (μ := μ) O A₀ (c := 1) (Or.inr rfl) hap
      (hside A₀ hAP).1 (hside A₀ hAP).2 hη hη₀ ha0 ha1
    have hs2 : μ.real E2 ≤ 2 * a := side_whp (μ := μ) O R₀ (c := 0) (Or.inl rfl) hap
      (hside R₀ hRP).1 (hside R₀ hRP).2 hη hη₀ ha0 ha1
    have hleR : ∀ D : Finset S, ∀ ω, (refApart O.mq D R ω : ℝ)
        ≤ ((D.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ) := by
      intro D ω
      have hRne : R.Nonempty := ⟨v, hvR⟩
      unfold refApart
      rw [dif_pos hRne]
      exact_mod_cast Finset.inf'_le _ hvR
    have hsub : Bad (A₀, R₀) ⊆ E1 ∪ E2 := by
      intro ω hω
      simp only [hBaddef, if_pos (And.intro hAP hRP), Set.mem_setOf_eq] at hω
      by_contra hc
      simp only [Set.mem_union, hE1, hE2, Set.mem_setOf_eq, not_or, not_not] at hc
      obtain ⟨h1, h2⟩ := hc
      have e1 := h1 (refApart O.mq A₀ R ω) (hleR A₀ ω)
      have e2 := h2 (refApart O.mq R₀ R ω) (hleR R₀ ω)
      simp only [iff_true] at e1
      simp only [show ((0 : ℝ) = 1) ↔ False from by norm_num, iff_false, not_not] at e2
      apply hω
      simp only [hbounddef]
      linarith
    exact le_trans (measureReal_mono hsub (measure_ne_top _ _))
      (le_trans (measureReal_union_le _ _) (by linarith))
  have hmain := selection_congr_bound O C Q hdisj T (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset P, Finset.empty_mem_powerset P⟩)
    sel hsel hcongr Bad hBadcongr (4 * a) (by positivity) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  simp only [Set.mem_setOf_eq] at hω ⊢
  have hselP : (sel ω).1 ⊆ P ∧ (sel ω).2 ⊆ P := ⟨Finset.filter_subset _ _, Finset.filter_subset _ _⟩
  simp only [hBaddef, if_pos hselP, Set.mem_setOf_eq]
  intro hle
  apply hω
  have hcount : ((P.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card : ℝ)
      ≤ (((sel ω).1.filter (fun p => O.label p ≠ 1)).card : ℝ)
        + (((sel ω).2.filter (fun p => O.label p ≠ 0)).card : ℝ) := by
    have hsub : P.filter (fun p => ¬ cutCorrect O lo hi F p ω)
        ⊆ (sel ω).1.filter (fun p => O.label p ≠ 1) ∪ (sel ω).2.filter (fun p => O.label p ≠ 0) := by
      intro p hp
      obtain ⟨hpP, hcc⟩ := Finset.mem_filter.1 hp
      by_contra hnot
      apply hcc
      constructor
      · intro h1
        by_contra hl
        exact hnot (Finset.mem_union_left _
          (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hpP, h1⟩, hl⟩))
      · intro h2
        by_contra hl
        exact hnot (Finset.mem_union_right _
          (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hpP, h2⟩, hl⟩))
    exact_mod_cast le_trans (Finset.card_le_card hsub) (Finset.card_union_le _ _)
  have hpos : 0 ≤ 1 - 2 * η₀ := by linarith
  simp only [hbounddef] at hle
  nlinarith [mul_le_mul_of_nonneg_right hcount hpos]

/-! ### At one state, over the run -/

open scoped Classical in
/-- An event the reads of a finite set of strings decide is measurable. -/
lemma measurableSet_of_readCongr (O : Oracle μ S) (C : Finset S) (Φ : Ω → Prop)
    (h : ∀ ω ω', (∀ w ∈ C, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) → Φ ω → Φ ω') :
    MeasurableSet {ω | Φ ω} := by
  set Pred : Finset S → Prop := fun U => ∃ ω', C.filter (fun w => O.mq w ω' = 1) = U ∧ Φ ω'
    with hPred
  have hcov : {ω | Φ ω} = {ω | Pred (C.filter (fun w => O.mq w ω = 1))} := by
    ext ω
    constructor
    · intro hω
      exact ⟨ω, rfl, hω⟩
    · rintro ⟨ω', hU, hΦ⟩
      refine h ω' ω (fun w hw => ?_) hΦ
      have := Finset.ext_iff.1 hU w
      simp only [Finset.mem_filter, hw, true_and] at this
      exact this
  rw [hcov]
  exact noiseAlg_le O Set.univ _ (measurableSet_filter_pred O (T := Set.univ) (by simp) Pred)

open scoped Classical in
lemma measurableSet_of_run_data_certAt (populations : Finset J) (j : J) (B : State) (m : ℕ)
    (R : Finset S → Finset S → Finset S → Set (Run Ω S J))
    (hR : ∀ P C A, MeasurableSet (R P C A)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (certOf (prefixesAt populations B.npref x) j m x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j m x)}
      = ⋃ z : Finset S × Finset S × Finset S,
          ((({x : Run Ω S J | prefixesAt populations B.npref x = z.1}
            ∩ {x : Run Ω S J | poolAt B.nsuff x = z.2.1})
            ∩ {x : Run Ω S J | certOf (prefixesAt populations B.npref x) j m x = z.2.2})
          ∩ R z.1 z.2.1 z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff]
    refine ⟨fun h => ⟨(prefixesAt populations B.npref x, poolAt B.nsuff x,
      certOf (prefixesAt populations B.npref x) j m x), ⟨⟨rfl, rfl⟩, rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C, A⟩, ⟨⟨hP, hC⟩, hA⟩, hx⟩
    simp only at hP hC hA
    rw [hA, hC, hP]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    (((measurableSet_prefixesAt populations B.npref z.1).inter
      (measurableSet_poolAt B.nsuff z.2.1)).inter
      (measurableSet_certOf populations B.npref j m z.2.2)).inter (hR _ _ _))

/-- The pool's suffixes outside the family: the gate's references. -/
noncomputable def refsBy (rule : Clusterer S) (mq : S → Ω → ℝ) (populations : Finset J)
    (x : Run Ω S J) (B : State) : Finset S :=
  poolAt B.nsuff x \ insert 1 (clusterBy rule mq populations x B)

open scoped Classical in
/-- At a look of `m` certification draws: the references hold an accept-preserving suffix, and
yet the cut's wrong count on the sample, at `1 − 2η₀` each, exceeds `cutBound`. -/
noncomputable def countFail (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (j : J) (B : State) (m : ℕ) (η₀ a : ℝ) : Set (Run Ω S J) :=
  {x | (∃ v ∈ refsBy rule O.mq populations x B, ∀ p, O.label (p * v) = O.label p)
    ∧ ¬ (((certOf (prefixesAt populations B.npref x) j m x).filter (fun p =>
          ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p (oracleNoise x))).card
        * (1 - 2 * η₀)
      ≤ cutBound O.mq B.lo B.hi (clusterBy rule O.mq populations x B)
          (refsBy rule O.mq populations x B) (certOf (prefixesAt populations B.npref x) j m x) a
          (oracleNoise x))}

open scoped Classical in
/-- The event `countFail` asks of the noise once the table `P`, the pool `C` and the sample `A`
are fixed. -/
noncomputable def countFailOf (rule : Clusterer S) (O : Oracle μ S) (B : State) (η₀ a : ℝ)
    (P C A : Finset S) (ω : Ω) : Prop :=
  (∃ v ∈ C \ insert 1 (clusterOf rule O B.sc B.scd P C B.k ω), ∀ p, O.label (p * v) = O.label p)
    ∧ ¬ ((A.filter (fun p =>
          ¬ cutCorrect O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k ω) p ω)).card
        * (1 - 2 * η₀)
      ≤ cutBound O.mq B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k ω)
          (C \ insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) A a ω)

lemma cutCorrect_congr (O : Oracle μ S) (lo hi : ℕ) (F : Finset S) (p : S) {ω ω' : Ω}
    (h : ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    (cutCorrect O lo hi F p ω ↔ cutCorrect O lo hi F p ω') := by
  unfold cutCorrect
  rw [voteCount_congr O F p h]

open scoped Classical in
lemma refApart_congr (O : Oracle μ S) (A R : Finset S) {ω ω' : Ω}
    (h : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1))
    (hR : ∀ p ∈ A, ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    refApart O.mq A R ω = refApart O.mq A R ω' := by
  unfold refApart
  split_ifs with hne
  · exact Finset.inf'_congr hne rfl (fun v hv => congrArg Finset.card
      (Finset.filter_congr (fun p hp => by rw [h p hp, hR p hp v hv])))
  · rfl

open scoped Classical in
lemma cutBound_congr (O : Oracle μ S) (lo hi : ℕ) (F R A : Finset S) (a : ℝ) {ω ω' : Ω}
    (hA : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1))
    (hF : ∀ p ∈ A, ∀ v ∈ F, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1))
    (hR : ∀ p ∈ A, ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    cutBound O.mq lo hi F R A a ω = cutBound O.mq lo hi F R A a ω' := by
  have hvc : ∀ p ∈ A, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_congr O F p (hF p hp)
  have hacc : A.filter (fun p => hi < voteCount O.mq F p ω)
      = A.filter (fun p => hi < voteCount O.mq F p ω') :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  have hrej : A.filter (fun p => voteCount O.mq F p ω ≤ lo)
      = A.filter (fun p => voteCount O.mq F p ω' ≤ lo) :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  unfold cutBound
  rw [hacc, hrej,
    refApart_congr O _ R (fun p hp => hA p (Finset.mem_filter.1 hp).1)
      (fun p hp v hv => hR p (Finset.mem_filter.1 hp).1 v hv),
    refApart_congr O _ R (fun p hp => hA p (Finset.mem_filter.1 hp).1)
      (fun p hp v hv => hR p (Finset.mem_filter.1 hp).1 v hv),
    Finset.filter_congr (fun p hp => by rw [hA p (Finset.mem_filter.1 hp).1] :
      ∀ p ∈ A.filter (fun p => hi < voteCount O.mq F p ω'),
        (¬ O.mq p ω = 1) ↔ (¬ O.mq p ω' = 1)),
    Finset.filter_congr (fun p hp => by rw [hA p (Finset.mem_filter.1 hp).1] :
      ∀ p ∈ A.filter (fun p => voteCount O.mq F p ω' ≤ lo),
        (O.mq p ω = 1) ↔ (O.mq p ω' = 1))]

open scoped Classical in
lemma countFailOf_congr (O : Oracle μ S) (B : State) (η₀ a : ℝ) (P C A : Finset S)
    (hone : (1 : S) ∈ C) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P C ∪ readSet A C, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    countFailOf rule O B η₀ a P C A ω → countFailOf rule O B η₀ a P C A ω' := by
  have hfam : clusterOf rule O B.sc B.scd P C B.k ω = clusterOf rule O B.sc B.scd P C B.k ω' :=
    clusterOf_congr_mq O B.sc B.scd P C B.k hone (fun w hw => h w (Finset.mem_union_left _ hw))
  have hAC : ∀ p ∈ A, ∀ v ∈ C, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1) := fun p hp v hv =>
    h _ (Finset.mem_union_right _ (mem_readSet hp hv))
  have hA : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun p hp => by
    simpa using hAC p hp 1 hone
  have hsub : clusterOf rule O B.sc B.scd P C B.k ω' ⊆ C :=
    clusterOf_subset O B.sc B.scd P C B.k ω' hone
  unfold countFailOf
  rw [hfam]
  rintro ⟨hv, hnot⟩
  refine ⟨hv, fun hle => hnot ?_⟩
  rw [cutBound_congr O B.lo B.hi _ _ A a hA (fun p hp v hv => hAC p hp v (hsub hv))
      (fun p hp v hv => hAC p hp v (Finset.mem_sdiff.1 hv).1),
    Finset.filter_congr (fun p hp => by
      rw [cutCorrect_congr O B.lo B.hi _ p (fun v hv => hAC p hp v (hsub hv))])]
  exact hle

open scoped Classical in
lemma measurableSet_countFail (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (m : ℕ) (η₀ a : ℝ) : MeasurableSet (countFail rule O populations j B m η₀ a) := by
  classical
  have hR : ∀ P C A : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω | countFailOf rule O B η₀ a P C A ω} else (∅ : Set (Run Ω S J))) := by
    intro P C A
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_readCongr O _ _
        (fun ω ω' h => countFailOf_congr O B η₀ a P C A hone h))
    · exact MeasurableSet.empty
  have hrw : countFail rule O populations j B m η₀ a
      = {x : Run Ω S J | x ∈ (fun P C A => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω | countFailOf rule O B η₀ a P C A ω} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j m x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_certAt populations j B m _ hR

open scoped Classical in
/-- `miscut_count_whp` over the run.  With the draws fixed the family is chosen by the table's
bits, and everything the count bound reads lies on the sample's strings, which the table never
reads; so conditioning on the family costs nothing. -/
theorem measureReal_countFail_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf]
    (hsupp : ∀ j ∈ populations, D j Preᶜ = 0) (hsuppSf : Dsf Sufᶜ = 0)
    (j : J) (hj : j ∈ populations) (B : State) (m : ℕ) {η₀ a : ℝ}
    (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2) (ha0 : 0 < a) (ha1 : a ≤ 1) :
    (runMeasure μ D Dsf).real (countFail rule O populations j B m η₀ a) ≤ 4 * a := by
  classical
  have hEnn : runMeasure μ D Dsf (countFail rule O populations j B m η₀ a)
      ≤ ENNReal.ofReal (4 * a) := by
    refine runMeasure_slice_le D Dsf _ (measurableSet_countFail O populations j B m η₀ a) _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_cert_mem_Pre D Dsf Pre populations hsupp, ae_sfx_mem_Suf D Dsf Suf hsuppSf]
      with d hdP hdC hdS
    set Pd : Finset S := populations.biUnion
      (fun j => freshDraws ∅ (fun i => d.1.2 j i) B.npref) with hPd
    set Cd : Finset S := insert 1 (freshDraws ∅ (fun i => d.1.1 i) B.nsuff) with hCd
    set Ad : Finset S := freshDraws Pd (fun i => d.2 j i) m with hAd
    have hV : ∀ v ∈ Cd, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    have hP : ∀ q ∈ Pd, q ∈ Pre := table_mem_Pre populations B.npref (fun j i => d.1.2 j i) hdP
    have hA : ∀ p ∈ Ad, p ∈ Pre := by
      intro p hp
      obtain ⟨i, rfl⟩ := mem_freshDraws hp
      exact hdC j hj i
    have hone : (1 : S) ∈ Cd := Finset.mem_insert_self _ _
    have hdisjPA : Disjoint Ad Pd := disjoint_freshDraws _ _ _
    have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ countFail rule O populations j B m η₀ a}
        = {ω | countFailOf rule O B η₀ a Pd Cd Ad ω} := rfl
    rw [hsec]
    set sel : Ω → Finset S := fun ω => clusterOf rule O B.sc B.scd Pd Cd B.k ω with hseldef
    set Bad : Finset S → Set Ω := fun F => if F ⊆ Cd ∧ (1 : S) ∉ F then
        {ω | (∃ v ∈ Cd \ insert 1 F, ∀ p, O.label (p * v) = O.label p)
          ∧ ¬ ((Ad.filter (fun p => ¬ cutCorrect O B.lo B.hi F p ω)).card * (1 - 2 * η₀)
            ≤ cutBound O.mq B.lo B.hi F (Cd \ insert 1 F) Ad a ω)}
      else ∅ with hBaddef
    have hdisj : Disjoint (↑(readSet Ad Cd) : Set S) (↑(readSet Pd Cd) : Set S) := by
      rw [Finset.disjoint_coe, Finset.disjoint_left]
      intro w hw hw'
      obtain ⟨⟨p, v⟩, hpv, rfl⟩ := Finset.mem_image.1 hw
      obtain ⟨⟨q, v'⟩, hqv, hq⟩ := Finset.mem_image.1 hw'
      obtain ⟨hp, hv⟩ := Finset.mem_product.1 hpv
      obtain ⟨hq', hv'⟩ := Finset.mem_product.1 hqv
      have := hflat q (hP q hq') p (hA p hp) v' (hV v' hv') v (hV v hv) hq
      exact Finset.disjoint_left.1 hdisjPA hp (this ▸ hq')
    have hmain := selection_congr_bound O (readSet Ad Cd) (readSet Pd Cd) hdisj Cd.powerset ∅
      (Finset.empty_mem_powerset _) sel
      (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.sc B.scd Pd Cd B.k ω hone))
      (fun ω ω' h => clusterOf_congr_mq O B.sc B.scd Pd Cd B.k hone
        (fun w hw => by rw [mq_congr O (h w hw)]))
      Bad (fun F ω ω' h hω => by
        simp only [hBaddef] at hω ⊢
        split_ifs at hω ⊢ with hF
        · have hAC : ∀ p ∈ Ad, ∀ v ∈ Cd, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1) :=
            fun p hp v hv => by rw [mq_congr O (h _ (mem_readSet hp hv))]
          have hAA : ∀ p ∈ Ad, (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun p hp => by
            simpa using hAC p hp 1 hone
          obtain ⟨hv, hnot⟩ := hω
          refine ⟨hv, fun hle => hnot ?_⟩
          rw [cutBound_congr O B.lo B.hi F _ Ad a hAA (fun p hp v hv => hAC p hp v (hF.1 hv))
              (fun p hp v hv => hAC p hp v (Finset.mem_sdiff.1 hv).1),
            Finset.filter_congr (fun p hp => by
              rw [cutCorrect_congr O B.lo B.hi F p (fun v hv => hAC p hp v (hF.1 hv))])]
          exact hle
        · exact hω)
      (4 * a) (by positivity) (fun F _ => by
        simp only [hBaddef]
        split_ifs with hF
        · by_cases hex : ∃ v ∈ Cd \ insert 1 F, ∀ p, O.label (p * v) = O.label p
          · obtain ⟨v, hvR, hap⟩ := hex
            refine le_trans (measureReal_mono (fun ω hω => hω.2) (measure_ne_top _ _)) ?_
            refine miscut_count_whp O Ad F (Cd \ insert 1 F) B.lo B.hi hvR hap ?_
              (Finset.disjoint_left.2 (fun u hu hu' =>
                (Finset.mem_sdiff.1 hu').2 (Finset.mem_insert_of_mem hu)))
              hF.2 (fun h1 => (Finset.mem_sdiff.1 h1).2 (Finset.mem_insert_self _ _))
              hη hη₀ ha0 ha1
            intro p hp p' hp' u hu u' hu' heq
            have hu1 : u ∈ insert 1 Suf := by
              rcases Finset.mem_insert.1 hu with rfl | hu
              · exact Set.mem_insert _ _
              · rcases Finset.mem_union.1 hu with hu | hu
                · exact hV u (hF.1 hu)
                · exact hV u (Finset.mem_sdiff.1 hu).1
            have hu1' : u' ∈ insert 1 Suf := by
              rcases Finset.mem_insert.1 hu' with rfl | hu'
              · exact Set.mem_insert _ _
              · rcases Finset.mem_union.1 hu' with hu' | hu'
                · exact hV u' (hF.1 hu')
                · exact hV u' (Finset.mem_sdiff.1 hu').1
            have hpp := hflat p (hA p hp) p' (hA p' hp') u hu1 u' hu1' heq
            subst hpp
            exact ⟨rfl, mul_left_cancel heq⟩
          · have hz : {ω | (∃ v ∈ Cd \ insert 1 F, ∀ p, O.label (p * v) = O.label p)
                ∧ ¬ ((Ad.filter (fun p => ¬ cutCorrect O B.lo B.hi F p ω)).card * (1 - 2 * η₀)
                  ≤ cutBound O.mq B.lo B.hi F (Cd \ insert 1 F) Ad a ω)} = (∅ : Set Ω) := by
              ext ω; simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
              exact fun h => hex h.1
            rw [hz, measureReal_empty]
            positivity
        · rw [measureReal_empty]; positivity)
    have hsub : {ω | countFailOf rule O B η₀ a Pd Cd Ad ω} ⊆ {ω | ω ∈ Bad (sel ω)} := by
      intro ω hω
      change ω ∈ Bad (sel ω)
      have hF : sel ω ⊆ Cd ∧ (1 : S) ∉ sel ω :=
        ⟨clusterOf_subset O B.sc B.scd Pd Cd B.k ω hone, Finset.notMem_erase _ _⟩
      simp only [hBaddef, if_pos hF]
      exact hω
    calc μ {ω | countFailOf rule O B η₀ a Pd Cd Ad ω} ≤ μ {ω | ω ∈ Bad (sel ω)} := measure_mono hsub
      _ = ENNReal.ofReal (μ.real {ω | ω ∈ Bad (sel ω)}) :=
          (ENNReal.ofReal_toReal (measure_ne_top μ _)).symm
      _ ≤ ENNReal.ofReal (4 * a) := ENNReal.ofReal_le_ofReal hmain
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (countFail rule O populations j B m η₀ a)).toReal
      ≤ (ENNReal.ofReal (4 * a)).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = 4 * a := ENNReal.toReal_ofReal (by positivity)

end OrthoDFA
