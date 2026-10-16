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

/-! ### What a side's caps bound -/

/-- The spread identity behind `spreadBound`: a side of `n` prefixes, `k` of them wrong, its
right ones at rate `x` and its wrong ones at `y`. -/
lemma spread_identity (n k : ℝ) (hn : 0 < n) (x y c : ℝ) :
    (n - k) / n * (x - c) ^ 2 + k / n * (1 - y - c) ^ 2
      = c * (1 - c) + (1 - 2 * c) * (((n - k) * x + k * (1 - y)) / n - c)
        - ((n - k) * (x * (1 - x)) + k * (y * (1 - y))) / n := by
  field_simp
  ring

/-- A side's cap bounds its wrong count, once its two counts sit within `d` of their means. -/
theorem le_mul_wrongCap (n k ag ap : ℕ) (hk : k ≤ n) {x y yb d : ℝ} (hy : y ≤ yb)
    (hlo : ((n : ℝ) - k) * x + k * (1 - y) - d ≤ ag)
    (hhi : (ag : ℝ) ≤ ((n : ℝ) - k) * x + k * (1 - y) + d)
    (hb : (ap : ℝ) ≤ 2 * (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) + d) :
    (k : ℝ) ≤ n * wrongCap n ag ap d yb := by
  rcases Nat.eq_zero_or_pos n with hn0 | hnpos
  · have : k = 0 := by omega
    subst this; subst hn0; simp
  have hnR : (0 : ℝ) < n := by exact_mod_cast hnpos
  have hkn : (k : ℝ) ≤ n := by exact_mod_cast hk
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  unfold wrongCap
  split_ifs with hden'
  · obtain ⟨-, hden⟩ := hden'
    set c : ℝ := (ag : ℝ) / n with hc
    set μ : ℝ := (((n : ℝ) - k) * x + k * (1 - y)) / n with hμ
    have hid := spread_identity (n : ℝ) k hnR x y c
    rw [← hμ] at hid
    have hμlo : ((ag : ℝ) - d) / n ≤ μ := by rw [hμ]; exact div_le_div_of_nonneg_right (by linarith) hnR.le
    have hμhi : μ ≤ ((ag : ℝ) + d) / n := by rw [hμ]; exact div_le_div_of_nonneg_right (by linarith) hnR.le
    have hmid : (1 - 2 * c) * (μ - c)
        ≤ max ((1 - 2 * c) * (((ag : ℝ) - d) / n - c)) ((1 - 2 * c) * (((ag : ℝ) + d) / n - c)) := by
      rcases le_total 0 (1 - 2 * c) with h | h
      · exact le_trans (mul_le_mul_of_nonneg_left (by linarith) h) (le_max_right _ _)
      · exact le_trans (mul_le_mul_of_nonpos_left (by linarith) h) (le_max_left _ _)
    have hfl : ((ap : ℝ) - d) / (2 * n) ≤ (((n : ℝ) - k) * (x * (1 - x)) + k * (y * (1 - y))) / n := by
      rw [div_le_div_iff₀ (by positivity) hnR]; nlinarith
    have hspread : (n - k) / n * (x - c) ^ 2 + k / n * (1 - y - c) ^ 2
        ≤ spreadBound n ag ap d c := by
      unfold spreadBound
      rw [hid]; linarith
    have h1 : (k : ℝ) / n * (1 - yb - c) ^ 2 ≤ k / n * (1 - y - c) ^ 2 :=
      mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hden.le (by linarith) 2) (by positivity)
    have h2 : 0 ≤ ((n : ℝ) - k) / n * (x - c) ^ 2 :=
      mul_nonneg (div_nonneg (by linarith) hnR.le) (sq_nonneg _)
    have hkappa : (k : ℝ) / n ≤ max 0 (spreadBound n ag ap d c) / (1 - yb - c) ^ 2 := by
      rw [le_div_iff₀ (by positivity)]
      linarith [le_max_right 0 (spreadBound n ag ap d c)]
    have hkappa1 : (k : ℝ) / n ≤ 1 := by rw [div_le_one hnR]; exact hkn
    have := le_min hkappa1 hkappa
    have h3 := mul_le_mul_of_nonneg_left this hnR.le
    rwa [mul_div_cancel₀ _ hnR.ne'] at h3
  · simpa using hkn

/-- A side's rate bound bounds its right class's noise rate. -/
theorem le_rateBound (n k ag : ℕ) {x y η₀ d cap : ℝ} (hx : x ≤ η₀) (hy1 : y ≤ 1) (hd : 0 ≤ d)
    (hcap : (k : ℝ) ≤ n * cap) (hcapn : n = 0 → 1 ≤ cap)
    (hhi : (ag : ℝ) ≥ ((n : ℝ) - k) * x + k * (1 - y) - d) :
    x ≤ rateBound η₀ ((ag + d) / n) cap := by
  unfold rateBound
  split_ifs with hc
  · refine le_min hx ?_
    have hm0 : 0 ≤ ((ag : ℝ) + d) / n := by positivity
    have hc1 : 0 < 1 - cap := by linarith
    rcases le_or_gt x 0 with hx0 | hx0
    · exact le_trans hx0 (div_nonneg hm0 hc1.le)
    rcases Nat.eq_zero_or_pos n with hn0 | hnpos
    · linarith [hcapn hn0]
    have hnR : (0 : ℝ) < n := by exact_mod_cast hnpos
    have hκ : (k : ℝ) / n ≤ cap := by rw [div_le_iff₀ hnR]; linarith
    have hk1 : 0 < 1 - (k : ℝ) / n := by linarith
    rw [le_div_iff₀ hc1]
    have h1 : x * (1 - (k : ℝ) / n) ≤ ((ag : ℝ) + d) / n := by
      rw [le_div_iff₀ hnR]
      have : x * (1 - (k : ℝ) / n) * n = ((n : ℝ) - k) * x := by field_simp
      rw [this]
      nlinarith [Nat.cast_nonneg (α := ℝ) k]
    nlinarith
  · exact hx


/-- The class a side holds rightly, as a bit: `1` on the accept side. -/
def sideClass (accept : Bool) : ℝ := if accept then 1 else 0

lemma sideClass_bit (accept : Bool) : sideClass accept = 0 ∨ sideClass accept = 1 := by
  cases accept <;> simp [sideClass]

omit [IsProbabilityMeasure μ] in
open scoped Classical in
lemma againstOn_eq_sum (O : Oracle μ S) (accept : Bool) (A : Finset S) (ω : Ω) :
    (againstOn O.mq accept A ω : ℝ) = ∑ p ∈ A, againstVar O (sideClass accept) p ω := by
  rw [← card_filter_eq_sum_againstVar]
  unfold againstOn
  congr 2
  refine Finset.filter_congr (fun p _ => ?_)
  cases accept <;> simp [sideClass]

open scoped Classical in
/-- A side's three counts sit within `hoeffDev #A a` of their means: the seed's count against
the side, both ways, and the reference `v`'s count of disagreements, from above. -/
def sideGood (O : Oracle μ S) (accept : Bool) (A : Finset S) (v : S) (a : ℝ) (ω : Ω) : Prop :=
  ∑ p ∈ A, μ[againstVar O (sideClass accept) p] - hoeffDev A.card a ≤ againstOn O.mq accept A ω
  ∧ (againstOn O.mq accept A ω : ℝ) ≤ ∑ p ∈ A, μ[againstVar O (sideClass accept) p]
      + hoeffDev A.card a
  ∧ ((A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ)
      ≤ ∑ p ∈ A, μ[apartVar O v p] + hoeffDev A.card a

open scoped Classical in
/-- Each of a side's three counts strays from its mean with chance at most `a`. -/
theorem side_tails (O : Oracle μ S) (accept : Bool) (A : Finset S) {v : S}
    (hne : ∀ p ∈ A, p * v ≠ p)
    (hblk : ∀ p ∈ A, ∀ p' ∈ A, p ≠ p' → Disjoint ({p, p * v} : Finset S) {p', p' * v})
    {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    μ.real {ω | ¬ sideGood O accept A v a ω} ≤ 3 * a := by
  classical
  set c : ℝ := sideClass accept with hcdef
  rcases Nat.eq_zero_or_pos A.card with hA0 | hApos
  · have hAe : A = ∅ := Finset.card_eq_zero.1 hA0
    have hz : {ω | ¬ sideGood O accept A v a ω} = (∅ : Set Ω) := by
      ext ω
      simp [sideGood, hAe, hoeffDev, againstOn]
    rw [hz, measureReal_empty]
    positivity
  set n : ℕ := A.card with hndef
  have hnR : (0 : ℝ) < n := by exact_mod_cast hApos
  set d : ℝ := hoeffDev n a with hd
  have hd0 : 0 ≤ d := Real.sqrt_nonneg _
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
  have hcard : ((Finset.univ : Finset A).card : ℝ) = n := by simp [hndef]
  have hEXs : ∑ i : A, μ[X i] = EX := Finset.sum_coe_sort A (fun p => μ[againstVar O c p])
  have hEYs : ∑ i : A, μ[Y i] = EY := Finset.sum_coe_sort A (fun p => μ[apartVar O v p])
  have hXm : ∀ i, AEMeasurable (X i) μ := fun i =>
    (againstVar_meas_alg O c (T := Set.univ) (Set.mem_univ _)).mono (noiseAlg_le O _) le_rfl
      |>.aemeasurable
  have hYm : ∀ i, AEMeasurable (Y i) μ := fun i =>
    (apartVar_meas_alg O (T := Set.univ) (Set.mem_univ _) (Set.mem_univ _)).mono
      (noiseAlg_le O _) le_rfl |>.aemeasurable
  have hXi : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1 := fun i =>
    Filter.Eventually.of_forall (fun ω => againstVar_icc O c _ ω)
  have hYi : ∀ i, ∀ᵐ ω ∂μ, Y i ω ∈ Set.Icc (0 : ℝ) 1 := fun i =>
    Filter.Eventually.of_forall (fun ω => apartVar_icc O v _ ω)
  have hdn : 0 ≤ d / n := div_nonneg hd0 hnR.le
  have hB1 : μ.real {ω | ∑ i : A, X i ω ≤ (n : ℝ) * (EX / n - d / n)} ≤ a := by
    have h := sumLower_le X Finset.univ (EX / n) (d / n) hXm hXind hXi
      (by rw [hcard, hEXs, mul_div_cancel₀ _ hnR.ne']) hdn
    rwa [hcard, exp_hoeffDev n hApos ha0 ha1] at h
  have hB2 : μ.real {ω | (n : ℝ) * (EX / n + d / n) ≤ ∑ i : A, X i ω} ≤ a := by
    have h := sumUpper_le X Finset.univ (EX / n) (d / n) hXm hXind hXi
      (by rw [hcard, hEXs, mul_div_cancel₀ _ hnR.ne']) hdn
    rwa [hcard, exp_hoeffDev n hApos ha0 ha1] at h
  have hB3 : μ.real {ω | (n : ℝ) * (EY / n + d / n) ≤ ∑ i : A, Y i ω} ≤ a := by
    have h := sumUpper_le Y Finset.univ (EY / n) (d / n) hYm hYind hYi
      (by rw [hcard, hEYs, mul_div_cancel₀ _ hnR.ne']) hdn
    rwa [hcard, exp_hoeffDev n hApos ha0 ha1] at h
  have hsub : {ω | ¬ sideGood O accept A v a ω}
      ⊆ {ω | ∑ i : A, X i ω ≤ (n : ℝ) * (EX / n - d / n)}
        ∪ ({ω | (n : ℝ) * (EX / n + d / n) ≤ ∑ i : A, X i ω}
          ∪ {ω | (n : ℝ) * (EY / n + d / n) ≤ ∑ i : A, Y i ω}) := by
    intro ω hω
    by_contra hc'
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_le] at hc'
    obtain ⟨h1, h2, h3⟩ := hc'
    apply hω
    have e1 : (n : ℝ) * (EX / n - d / n) = EX - d := by field_simp
    have e2 : (n : ℝ) * (EX / n + d / n) = EX + d := by field_simp
    have e3 : (n : ℝ) * (EY / n + d / n) = EY + d := by field_simp
    rw [e1, Finset.sum_coe_sort A (fun p => againstVar O c p ω)] at h1
    rw [e2, Finset.sum_coe_sort A (fun p => againstVar O c p ω)] at h2
    rw [e3, Finset.sum_coe_sort A (fun p => apartVar O v p ω)] at h3
    refine ⟨?_, ?_, ?_⟩
    · rw [againstOn_eq_sum]; exact h1.le
    · rw [againstOn_eq_sum]; exact h2.le
    · rw [card_filter_eq_sum_apartVar]; exact h3.le
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ _ := le_trans (measureReal_union_le _ _) (add_le_add_right (measureReal_union_le _ _) _)
    _ ≤ a + (a + a) := add_le_add hB1 (add_le_add hB2 hB3)
    _ = 3 * a := by ring

/-- The noise rate of a side's right class, and of its wrong one. -/
noncomputable def rightRate (O : Oracle μ S) (accept : Bool) : ℝ :=
  if accept then O.ηIn else O.ηOut

noncomputable def wrongRate (O : Oracle μ S) (accept : Bool) : ℝ :=
  if accept then O.ηOut else O.ηIn

omit [IsProbabilityMeasure μ] in
lemma rate_of_label (O : Oracle μ S) (accept : Bool) (p : S) :
    O.rate p = if O.label p = sideClass accept then rightRate O accept else wrongRate O accept := by
  classical
  unfold Oracle.rate rightRate wrongRate sideClass
  by_cases hp : p ∈ O.L <;> cases accept <;>
    simp [hp, Oracle.label, Set.indicator_of_mem, Set.indicator_of_notMem]

open scoped Classical in
/-- On a good side the means are those of `n − k` right prefixes and `k` wrong ones. -/
lemma side_means (O : Oracle μ S) (accept : Bool) (A : Finset S) {v : S}
    (hap : ∀ p, O.label (p * v) = O.label p) (hne : ∀ p ∈ A, p * v ≠ p) :
    ∑ p ∈ A, μ[againstVar O (sideClass accept) p]
        = ((A.card : ℝ) - (A.filter (fun p => O.label p ≠ sideClass accept)).card)
            * rightRate O accept
          + (A.filter (fun p => O.label p ≠ sideClass accept)).card * (1 - wrongRate O accept)
      ∧ ∑ p ∈ A, μ[apartVar O v p]
        = 2 * (((A.card : ℝ) - (A.filter (fun p => O.label p ≠ sideClass accept)).card)
            * (rightRate O accept * (1 - rightRate O accept))
          + (A.filter (fun p => O.label p ≠ sideClass accept)).card
            * (wrongRate O accept * (1 - wrongRate O accept))) := by
  classical
  set c := sideClass accept with hc
  set x := rightRate O accept with hx
  set y := wrongRate O accept with hy
  have hcnt : ((A.filter (fun p => O.label p = c)).card : ℝ)
      = (A.card : ℝ) - (A.filter (fun p => O.label p ≠ c)).card := by
    have := Finset.card_filter_add_card_filter_not (s := A) (fun p => O.label p = c)
    have : ((A.filter (fun p => O.label p = c)).card : ℝ)
        + ((A.filter (fun p => ¬ O.label p = c)).card : ℝ) = A.card := by exact_mod_cast this
    simp only [ne_eq]
    linarith
  have h1 : ∀ p ∈ A, μ[againstVar O c p] = if O.label p = c then x else 1 - y := by
    intro p _
    rw [againstVar_mean O (sideClass_bit accept) p, rate_of_label O accept p]
    split_ifs <;> rfl
  have h2 : ∀ p ∈ A, μ[apartVar O v p]
      = if O.label p = c then 2 * (x * (1 - x)) else 2 * (y * (1 - y)) := by
    intro p hp
    rw [apartVar_mean O (hap p) (hne p hp), rate_of_label O accept p]
    split_ifs <;> ring
  constructor
  · rw [Finset.sum_congr rfl h1, Finset.sum_ite, Finset.sum_const, Finset.sum_const,
      nsmul_eq_mul, nsmul_eq_mul, hcnt]
  · rw [Finset.sum_congr rfl h2, Finset.sum_ite, Finset.sum_const, Finset.sum_const,
      nsmul_eq_mul, nsmul_eq_mul, hcnt]
    simp only [ne_eq]
    ring

lemma refApart_le_of_mem (mq : S → Ω → ℝ) (A R : Finset S) {v : S} (hv : v ∈ R) (ω : Ω) :
    refApart mq A R ω ≤ (A.filter (fun p => ¬ ((mq (p * v) ω = 1) ↔ (mq p ω = 1)))).card := by
  classical
  unfold refApart
  rw [dif_pos ⟨v, hv⟩]
  exact Finset.inf'_le _ hv

open scoped Classical in
/-- On a good side whose wrong class's rate is at most `y`, the cap bounds the wrong count. -/
theorem wrong_le_sideCap (O : Oracle μ S) (accept : Bool) (A R : Finset S) {v : S} (hvR : v ∈ R)
    (hap : ∀ p, O.label (p * v) = O.label p) (hne : ∀ p ∈ A, p * v ≠ p) {a y : ℝ} {ω : Ω}
    (hgood : sideGood O accept A v a ω) (hy : wrongRate O accept ≤ y) :
    ((A.filter (fun p => O.label p ≠ sideClass accept)).card : ℝ)
      ≤ A.card * sideCap O.mq accept A R a y ω := by
  obtain ⟨hlo, hhi, hb⟩ := hgood
  obtain ⟨hEa, hEb⟩ := side_means O accept A hap hne
  rw [hEa] at hlo hhi
  rw [hEb] at hb
  have hap' := refApart_le_of_mem O.mq A R hvR ω
  unfold sideCap
  exact le_mul_wrongCap A.card _ _ _ (Finset.card_le_card (Finset.filter_subset _ _)) hy hlo hhi
    (le_trans (by exact_mod_cast hap') hb)

open scoped Classical in
/-- On a good side the rate bound bounds the right class's rate. -/
theorem rightRate_le_sideRate (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (accept : Bool) (A R : Finset S) {v : S} (hvR : v ∈ R)
    (hap : ∀ p, O.label (p * v) = O.label p) (hne : ∀ p ∈ A, p * v ≠ p) {a : ℝ} {ω : Ω}
    (hgood : sideGood O accept A v a ω) :
    rightRate O accept ≤ sideRate O.mq η₀ accept A R a ω := by
  have hηIn : O.ηIn ≤ η₀ := le_trans (le_max_left _ _) hη
  have hηOut : O.ηOut ≤ η₀ := le_trans (le_max_right _ _) hη
  have hx : rightRate O accept ≤ η₀ := by unfold rightRate; split_ifs <;> assumption
  have hy : wrongRate O accept ≤ η₀ := by unfold wrongRate; split_ifs <;> assumption
  have hcap := wrong_le_sideCap O accept A R hvR hap hne hgood hy
  obtain ⟨hlo, -, -⟩ := hgood
  obtain ⟨hEa, -⟩ := side_means O accept A hap hne
  rw [hEa] at hlo
  unfold sideRate
  refine le_rateBound A.card _ _ (y := wrongRate O accept) hx (by linarith) (Real.sqrt_nonneg _)
    hcap ?_ hlo
  intro h0
  unfold sideCap wrongCap
  rw [if_neg (fun h => by omega)]


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

/-! ### Spending a level over sample sizes -/

lemma sizeLevel_nonneg {a : ℝ} (ha : 0 ≤ a) (n : ℕ) : 0 ≤ sizeLevel a n := by
  unfold sizeLevel; have := ha; positivity

lemma sizeLevel_pos {a : ℝ} (ha : 0 < a) (n : ℕ) : 0 < sizeLevel a n := by
  unfold sizeLevel; positivity

lemma sizeLevel_le {a : ℝ} (ha : 0 ≤ a) (ha1 : a ≤ 1) (n : ℕ) : sizeLevel a n ≤ 1 := by
  unfold sizeLevel
  rw [div_le_one (by positivity)]
  nlinarith [(Nat.cast_nonneg n : (0 : ℝ) ≤ n)]

lemma sum_range_sizeLevel (a : ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range n, sizeLevel a i = a * (1 - 1 / ((n : ℝ) + 1)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih, sizeLevel]
    push_cast
    field_simp
    ring

/-- Sizes from `k` on spend at most `a` together. -/
lemma sum_range_sizeLevel_shift_le {a : ℝ} (ha : 0 ≤ a) (k n : ℕ) :
    ∑ i ∈ Finset.range n, sizeLevel a (k + i) ≤ a := by
  have h := Finset.sum_range_add (fun i => sizeLevel a i) k n
  rw [sum_range_sizeLevel] at h
  have h0 : 0 ≤ ∑ i ∈ Finset.range k, sizeLevel a i :=
    Finset.sum_nonneg (fun i _ => sizeLevel_nonneg ha i)
  have : 0 ≤ a * (1 / ((((k + n : ℕ) : ℝ)) + 1)) := by positivity
  linarith

/-! ### From the draws to the population -/

open scoped Classical in
/-- The cut misclassifies more than `lim` of the population, and the first `N` certification
draws show it at most `N·lim` less a deviation at `sizeLevel a N`. -/
noncomputable def rawShortAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (Dj : Measure S) (j : J) (B : State) (lim a : ℝ) (N : ℕ) : Set (Run Ω S J) :=
  {x | lim < Dj.real {p | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
        (oracleNoise x)}
    ∧ (((Finset.univ : Finset (Fin N)).filter (fun i =>
        ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) (certPrefix j i.val x)
          (oracleNoise x))).card : ℝ)
      ≤ N * lim - hoeffDev N (sizeLevel a N)}

open scoped Classical in
lemma measurableSet_of_run_data_certN (populations : Finset J) (j : J) (B : State) (N : ℕ)
    (R : Finset S → Finset S → (Fin N → S) → Set (Run Ω S J))
    (hR : ∀ P C t, MeasurableSet (R P C t)) :
    MeasurableSet {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
      (fun i : Fin N => certPrefix j i.val x)} := by
  classical
  have hcov : {x : Run Ω S J | x ∈ R (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin N => certPrefix j i.val x)}
      = ⋃ z : Finset S × Finset S × (Fin N → S),
          ((({x : Run Ω S J | prefixesAt populations B.npref x = z.1}
            ∩ {x : Run Ω S J | poolAt B.nsuff x = z.2.1})
            ∩ ⋂ i : Fin N, {x : Run Ω S J | certPrefix j i.val x = z.2.2 i})
          ∩ R z.1 z.2.1 z.2.2) := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_iInter]
    refine ⟨fun h => ⟨(prefixesAt populations B.npref x, poolAt B.nsuff x,
      fun i : Fin N => certPrefix j i.val x), ⟨⟨rfl, rfl⟩, fun _ => rfl⟩, h⟩, ?_⟩
    rintro ⟨⟨P, C, t⟩, ⟨⟨hP, hC⟩, ht⟩, hx⟩
    simp only at hP hC ht
    rw [hP, hC, show (fun i : Fin N => certPrefix j i.val x) = t from funext ht]
    exact hx
  rw [hcov]
  exact MeasurableSet.iUnion (fun z =>
    (((measurableSet_prefixesAt populations B.npref z.1).inter
      (measurableSet_poolAt B.nsuff z.2.1)).inter
        (MeasurableSet.iInter (fun i : Fin N =>
          measurableSet_eq_fun (measurable_cert j i.val) measurable_const))).inter
      (hR z.1 z.2.1 z.2.2))

open scoped Classical in
lemma measurableSet_rawShortAt (O : Oracle μ S) (populations : Finset J) (Dj : Measure S)
    (j : J) (B : State) (lim a : ℝ) (N : ℕ) :
    MeasurableSet (rawShortAt rule O populations Dj j B lim a N) := by
  classical
  have hR : ∀ (P C : Finset S) (tt : Fin N → S), MeasurableSet (if (1 : S) ∈ C then
      {x : Run Ω S J | lim < Dj.real {p | ¬ cutCorrect O B.lo B.hi
            (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
        ∧ (((Finset.univ : Finset (Fin N)).filter (fun i =>
            ¬ cutCorrect O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x))
                (tt i) (oracleNoise x))).card : ℝ)
          ≤ N * lim - hoeffDev N (sizeLevel a N)} else ∅) := by
    intro P C tt
    split_ifs with hone
    · have hω : MeasurableSet {ω : Ω |
          lim < Dj.real {p | ¬ cutCorrect O B.lo B.hi
              (clusterOf rule O B.sc B.scd P C B.k ω) p ω}
          ∧ (((Finset.univ : Finset (Fin N)).filter (fun i =>
              ¬ cutCorrect O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k ω) (tt i) ω)).card
                : ℝ) ≤ N * lim - hoeffDev N (sizeLevel a N)} :=
        measurableSet_of_fam (T := C.powerset)
          (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.sc B.scd P C B.k ω hone))
          (fun A₀ => measurableSet_clusterOf O B.sc B.scd P C B.k hone A₀)
          (fun A₀ => {ω | lim < Dj.real {p | ¬ cutCorrect O B.lo B.hi A₀ p ω}
            ∧ (((Finset.univ : Finset (Fin N)).filter (fun i =>
                ¬ cutCorrect O B.lo B.hi A₀ (tt i) ω)).card : ℝ)
                  ≤ N * lim - hoeffDev N (sizeLevel a N)})
          (fun A₀ => MeasurableSet.inter
            (measurableSet_lt measurable_const (measurable_badMassReal O Dj B.lo B.hi A₀))
            (measurableSet_card_filter_fin (fun i ω => ¬ cutCorrect O B.lo B.hi A₀ (tt i) ω)
              (fun i => noiseAlg_le O Set.univ _ (measurableSet_cutCorrect' O B.lo B.hi A₀ (tt i)))
              (fun n => (n : ℝ) ≤ N * lim - hoeffDev N (sizeLevel a N))))
      exact measurable_nz hω
    · exact MeasurableSet.empty
  have hrw : rawShortAt rule O populations Dj j B lim a N
      = {x : Run Ω S J | x ∈ (fun P C tt => if (1 : S) ∈ C then
          {x : Run Ω S J | lim < Dj.real {p | ¬ cutCorrect O B.lo B.hi
                (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x)) p (oracleNoise x)}
            ∧ (((Finset.univ : Finset (Fin N)).filter (fun i =>
                ¬ cutCorrect O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k (oracleNoise x))
                    (tt i) (oracleNoise x))).card : ℝ)
              ≤ N * lim - hoeffDev N (sizeLevel a N)} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (fun i : Fin N => certPrefix j i.val x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), rawShortAt,
      ← clusterAt_eq_clusterOf O populations B x]
  rw [hrw]
  exact measurableSet_of_run_data_certN populations j B N _ hR

open scoped Classical in
/-- At one count of draws: the family is fixed before the draws, so Hoeffding applies. -/
theorem measureReal_rawShortAt_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) {lim a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1)
    (N : ℕ) (hN : 0 < N) :
    (runMeasure μ D Dsf).real (rawShortAt rule O populations (D j) j B lim a N)
      ≤ sizeLevel a N := by
  classical
  set E : ℝ := sizeLevel a N with hEdef
  have hE0 : 0 ≤ E := sizeLevel_nonneg ha0.le N
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  have hEnn : runMeasure μ D Dsf (rawShortAt rule O populations (D j) j B lim a N)
      ≤ ENNReal.ofReal E := by
    refine runMeasure_slice_cert_le D Dsf _
      (measurableSet_rawShortAt O populations (D j) j B lim a N) _ ?_
    intro y
    set F : Finset S :=
      clusterBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B with hF
    set W : Set S := {p | ¬ cutCorrect O B.lo B.hi F p y.1} with hW
    have hFeq : ∀ c : J → ℕ → S,
        clusterBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : lim < (D j).real W
    · set t : ℝ := hoeffDev N (sizeLevel a N) / N with ht
      have ht0 : 0 ≤ t := div_nonneg (Real.sqrt_nonneg _) hNR.le
      have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ rawShortAt rule O populations (D j) j B lim a N}
          ⊆ {c | (((Finset.range N).filter (fun i => c j i ∈ W)).card : ℝ)
            ≤ (N : ℝ) * ((D j).real W - t)} := by
        rintro c ⟨-, hcount⟩
        have h := card_filter_fin_eq N (fun p => p ∈ W) (fun i => c j i)
        show (((Finset.range N).filter (fun i => c j i ∈ W)).card : ℝ)
          ≤ (N : ℝ) * ((D j).real W - t)
        rw [← h]
        have e : (N : ℝ) * ((D j).real W - t) = N * (D j).real W - hoeffDev N (sizeLevel a N) := by
          rw [ht]; field_simp
        rw [e]
        refine le_trans hcount ?_
        nlinarith
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top
          (Measure.pi fun j : J => Measure.infinitePi fun _ : ℕ => D j) _), ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_wrongSet D j N W ((D j).real W) t measureReal_nonneg ht0 le_rfl
      have hexp : Real.exp (-2 * (N : ℝ) * t ^ 2) = E :=
        exp_hoeffDev N hN (sizeLevel_pos ha0 N) (sizeLevel_le ha0.le ha1 N)
      rw [hexp] at hcert
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ rawShortAt rule O populations (D j) j B lim a N} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (runMeasure μ D Dsf (rawShortAt rule O populations (D j) j B lim a N)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

/-- Over every count of draws together. -/
theorem measureReal_rawShort_le (D : J → Measure S) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (D j)] [IsProbabilityMeasure Dsf] (O : Oracle μ S)
    (populations : Finset J) (j : J) (B : State) {lim a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    (runMeasure μ D Dsf).real (⋃ N : ℕ, rawShortAt rule O populations (D j) j B lim a (N + 1))
      ≤ a := by
  have hle : runMeasure μ D Dsf (⋃ N : ℕ, rawShortAt rule O populations (D j) j B lim a (N + 1))
      ≤ ENNReal.ofReal a := by
    refine le_trans (measure_iUnion_le _) ?_
    refine ENNReal.tsum_le_of_sum_range_le (fun n => ?_)
    calc ∑ i ∈ Finset.range n, runMeasure μ D Dsf (rawShortAt rule O populations (D j) j B lim a (i + 1))
        ≤ ∑ i ∈ Finset.range n, ENNReal.ofReal (sizeLevel a (1 + i)) := by
          refine Finset.sum_le_sum (fun i _ => ?_)
          rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def, add_comm 1 i]
          exact ENNReal.ofReal_le_ofReal
            (measureReal_rawShortAt_le D Dsf O populations j B ha0 ha1 (i + 1) (Nat.succ_pos i))
      _ = ENNReal.ofReal (∑ i ∈ Finset.range n, sizeLevel a (1 + i)) :=
          (ENNReal.ofReal_sum_of_nonneg (fun i _ => sizeLevel_nonneg ha0.le _)).symm
      _ ≤ ENNReal.ofReal a := ENNReal.ofReal_le_ofReal (sum_range_sizeLevel_shift_le ha0.le 1 n)
  rw [measureReal_def]
  exact le_trans (ENNReal.toReal_mono ENNReal.ofReal_ne_top hle)
    (le_of_eq (ENNReal.toReal_ofReal ha0.le))

/-! ### The gate at one state and one population -/

open scoped Classical in
/-- A count over the first `N` draws is a count over the sample they gave, plus the draws the
sampler redrew. -/
lemma card_range_filter_le (T : Finset S) (y : ℕ → S) (Q : S → Prop) [DecidablePred Q] (N : ℕ) :
    (((Finset.range N).filter (fun i => Q (y i))).card : ℝ)
      ≤ ((acceptedAt T y N).filter Q).card
        + ((N : ℝ) - (acceptedAt T y N).card) := by
  have h := sum_range_le_sum_acceptedAt T y (fun p => if Q p then (1 : ℝ) else 0)
    (fun a => by split_ifs <;> norm_num) (fun a => by split_ifs <;> norm_num) N
  rw [Finset.natCast_card_filter, Finset.natCast_card_filter]
  exact h

/-! ### A clean family is certified -/

/-- Any reference disagrees with the seed at least as often as an accept-preserving one:
reading the other class costs `(1 − 2r)(1 − r − r') ≥ 0` more. -/
lemma apartVar_mean_ge (O : Oracle μ S) (hη : O.η ≤ 1 / 2) {v p : S} (hne : p * v ≠ p) :
    2 * O.rate p * (1 - O.rate p) ≤ μ[apartVar O v p] := by
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
  have pa : μ.real (O.mq (p * v) ⁻¹' {1})
      = O.rate (p * v) + (1 - 2 * O.rate (p * v)) * O.label (p * v) := prob_mq_one O (p * v)
  have pb : μ.real (O.mq p ⁻¹' {1}) = O.rate p + (1 - 2 * O.rate p) * O.label p := prob_mq_one O p
  have pa' : μ.real (O.mq (p * v) ⁻¹' {1}ᶜ) = 1 - μ.real (O.mq (p * v) ⁻¹' {1}) := by
    rw [Set.preimage_compl, measureReal_compl (measurableSet_singleton 1 |>.preimage
      (mq_meas O _)), probReal_univ]
  have pb' : μ.real (O.mq p ⁻¹' {1}ᶜ) = 1 - μ.real (O.mq p ⁻¹' {1}) := by
    rw [Set.preimage_compl, measureReal_compl (measurableSet_singleton 1 |>.preimage
      (mq_meas O _)), probReal_univ]
  rw [pa', pb', pa, pb]
  have hr0 := O.rate_nonneg p
  have hr1 : O.rate p ≤ 1 / 2 := le_trans (O.rate_le_eta p) hη
  have hr1' : O.rate (p * v) ≤ 1 / 2 := le_trans (O.rate_le_eta _) hη
  have key : 0 ≤ (1 - 2 * O.rate p) * (1 - O.rate p - O.rate (p * v)) :=
    mul_nonneg (by linarith) (by linarith)
  rcases O.label_bit (p * v) with hl | hl <;> rcases O.label_bit p with hl' | hl' <;>
    rw [hl, hl']
  · rw [O.rate_eq_of_label_eq (hl.trans hl'.symm)]; nlinarith
  · nlinarith
  · nlinarith
  · rw [O.rate_eq_of_label_eq (hl.trans hl'.symm)]; nlinarith

lemma hoeffDev_mono {n n' : ℕ} (h : n ≤ n') {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    hoeffDev n a ≤ hoeffDev n' a := by
  have hl : 0 ≤ Real.log (1 / a) := Real.log_nonneg (by rw [le_div_iff₀ ha0]; linarith)
  unfold hoeffDev
  refine Real.sqrt_le_sqrt ?_
  have : (n : ℝ) ≤ n' := by exact_mod_cast h
  have := mul_le_mul_of_nonneg_right this hl
  linarith

lemma card_poolAt_le (M : ℕ) (x : Run Ω S J) : (poolAt M x).card ≤ M + 1 :=
  le_trans (Finset.card_insert_le _ _) (by have := card_freshDraws_le ∅ (fun i => suffixDraw i x) M; omega)

end OrthoDFA
