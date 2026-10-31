import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.GateCharge

/-!
# The gate's miscut bound is sound

`certifiesAt` bounds how much of a population the family's cut misclassifies, reading each
class's noise rate at `η₀`.  On a side of the cut, `p` reads against the side on the seed at
`t_p`: its noise rate `r_p` where the cut is right, `1 − r_p` where it is wrong.  An
accept-preserving reference disagrees with the seed at `2r_p(1 − r_p)`, which pins the noise's
share of the spread of the `t_p`; a wrong prefix adds at least `(1 − η₀ − t̄)²` to the rest.
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
/-- Both sides of a sample good for the reference `v`. -/
def sidesGood (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (v : S) (a : ℝ) (ω : Ω) : Prop :=
  sideGood O true (acceptSide O.mq hi F P ω) v a ω ∧ sideGood O false (rejectSide O.mq lo F P ω) v a ω

omit [IsProbabilityMeasure μ] in
open scoped Classical in
/-- The cut's wrong prefixes are the accept side's outside the language and the reject side's
inside it. -/
lemma miscut_le_sides (O : Oracle μ S) (lo hi : ℕ) (F P : Finset S) (ω : Ω) :
    ((P.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card : ℝ)
      ≤ (((acceptSide O.mq hi F P ω).filter (fun p => O.label p ≠ sideClass true)).card : ℝ)
        + (((rejectSide O.mq lo F P ω).filter (fun p => O.label p ≠ sideClass false)).card : ℝ) := by
  classical
  have hsub : P.filter (fun p => ¬ cutCorrect O lo hi F p ω)
      ⊆ (acceptSide O.mq hi F P ω).filter (fun p => O.label p ≠ sideClass true)
        ∪ (rejectSide O.mq lo F P ω).filter (fun p => O.label p ≠ sideClass false) := by
    intro p hp
    obtain ⟨hpP, hcc⟩ := Finset.mem_filter.1 hp
    by_contra hnot
    apply hcc
    constructor
    · intro h1
      by_contra hl
      exact hnot (Finset.mem_union_left _
        (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hpP, h1⟩, by simpa [sideClass] using hl⟩))
    · intro h2
      by_contra hl
      exact hnot (Finset.mem_union_right _
        (Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hpP, h2⟩, by simpa [sideClass] using hl⟩))
  have := le_trans (Finset.card_le_card hsub) (Finset.card_union_le _ _)
  exact_mod_cast this

omit [IsProbabilityMeasure μ] in
lemma wrongRate_le (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (accept : Bool) :
    wrongRate O accept ≤ η₀ := by
  have h1 : O.ηIn ≤ η₀ := le_trans (le_max_left _ _) hη
  have h2 : O.ηOut ≤ η₀ := le_trans (le_max_right _ _) hη
  unfold wrongRate; split_ifs <;> assumption

open scoped Classical in
/-- With both of the sample's sides good, the cut's wrong count on the sample is at most
`miscutBound`. -/
theorem miscut_le_miscutBound (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (lo hi : ℕ)
    (F R P : Finset S) {v : S} (hvR : v ∈ R) (hap : ∀ p, O.label (p * v) = O.label p)
    (hneP : ∀ p ∈ P, p * v ≠ p) {a : ℝ} {ω : Ω} (hP : sidesGood O lo hi F P v a ω) :
    ((P.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card : ℝ)
      ≤ miscutBound O.mq a lo hi F R P η₀ ω := by
  have hA := wrong_le_sideCap O true (acceptSide O.mq hi F P ω) R hvR hap
    (fun p hp => hneP p (Finset.mem_filter.1 hp).1) hP.1 (wrongRate_le O hη true)
  have hR := wrong_le_sideCap O false (rejectSide O.mq lo F P ω) R hvR hap
    (fun p hp => hneP p (Finset.mem_filter.1 hp).1) hP.2 (wrongRate_le O hη false)
  refine le_trans (miscut_le_sides O lo hi F P ω) ?_
  unfold miscutBound
  linarith

/-! ### Sides chosen by the votes -/

/-- A bound at every fixed index survives an index the scored strings' bits do not decide. -/
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

lemma sideGood_congr (O : Oracle μ S) (accept : Bool) (A : Finset S) (v : S) (a : ℝ) {ω ω' : Ω}
    (h : ∀ p ∈ A, O.noise p ω = O.noise p ω' ∧ O.noise (p * v) ω = O.noise (p * v) ω') :
    sideGood O accept A v a ω → sideGood O accept A v a ω' := by
  classical
  have h1 : ∀ p ∈ A, O.mq p ω = O.mq p ω' := fun p hp => mq_congr O (h p hp).1
  have h2 : ∀ p ∈ A, O.mq (p * v) ω = O.mq (p * v) ω' := fun p hp => mq_congr O (h p hp).2
  have ha : againstOn O.mq accept A ω = againstOn O.mq accept A ω' := by
    unfold againstOn
    exact congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [h1 p hp]))
  have hb : (A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card
      = (A.filter (fun p => ¬ ((O.mq (p * v) ω' = 1) ↔ (O.mq p ω' = 1)))).card :=
    congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [h1 p hp, h2 p hp]))
  unfold sideGood
  rw [ha, hb]
  exact id

open scoped Classical in
/-- Both sides of a sample good, with the sample, the voters and the reference fixed and the cut
chosen by the votes: chance at most `6a` they are not. -/
theorem sidesGood_whp (O : Oracle μ S) (P F : Finset S) (lo hi : ℕ) {v : S}
    (hinj : ∀ p ∈ P, ∀ p' ∈ P, ∀ u ∈ insert 1 (insert v F), ∀ u' ∈ insert 1 (insert v F),
      p * u = p' * u' → p = p' ∧ u = u')
    (hvF : v ∉ F) (h1F : (1 : S) ∉ F) (hv1 : v ≠ 1) {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    μ.real {ω | ¬ sidesGood O lo hi F P v a ω} ≤ 6 * a := by
  classical
  have h1mem : (1 : S) ∈ insert 1 (insert v F) := Finset.mem_insert_self _ _
  have hvmem : v ∈ insert 1 (insert v F) := Finset.mem_insert_of_mem (Finset.mem_insert_self _ _)
  have humem : ∀ u ∈ F, u ∈ insert 1 (insert v F) := fun u hu =>
    Finset.mem_insert_of_mem (Finset.mem_insert_of_mem hu)
  set C : Finset S := P ∪ P.image (fun p => p * v) with hC
  set Q : Finset S := (P ×ˢ F).image (fun q => q.1 * q.2) with hQ
  have hdisj : Disjoint (↑C : Set S) (↑Q : Set S) := by
    rw [Finset.disjoint_coe]
    refine Finset.disjoint_left.2 (fun w hwC hwQ => ?_)
    obtain ⟨⟨p', u⟩, hpu, rfl⟩ := Finset.mem_image.1 hwQ
    obtain ⟨hp', hu⟩ := Finset.mem_product.1 hpu
    rcases Finset.mem_union.1 hwC with hp | hw
    · have := (hinj _ hp p' hp' 1 h1mem u (humem u hu) (mul_one _)).2
      exact h1F (this ▸ hu)
    · obtain ⟨p, hp, hpeq⟩ := Finset.mem_image.1 hw
      have := (hinj p hp p' hp' v hvmem u (humem u hu) hpeq).2
      exact hvF (this ▸ hu)
  set T : Finset (Finset S × Finset S) := P.powerset ×ˢ P.powerset with hT
  set sel : Ω → Finset S × Finset S := fun ω =>
    (acceptSide O.mq hi F P ω, rejectSide O.mq lo F P ω) with hseldef
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
  set Bad : Finset S × Finset S → Set Ω := fun t =>
    if t.1 ⊆ P ∧ t.2 ⊆ P then
      {ω | ¬ (sideGood O true t.1 v a ω ∧ sideGood O false t.2 v a ω)}
    else ∅ with hBaddef
  have hBadcongr : ∀ t ω ω', (∀ w ∈ C, O.noise w ω = O.noise w ω') → ω ∈ Bad t → ω' ∈ Bad t := by
    intro t ω ω' h hω
    simp only [hBaddef] at hω ⊢
    split_ifs at hω ⊢ with ht
    · have hpp : ∀ D ⊆ P, ∀ p ∈ D, O.noise p ω' = O.noise p ω
          ∧ O.noise (p * v) ω' = O.noise (p * v) ω := fun D hD p hp =>
        ⟨(h p (Finset.mem_union_left _ (hD hp))).symm,
          (h _ (Finset.mem_union_right _ (Finset.mem_image_of_mem _ (hD hp)))).symm⟩
      intro hgood
      exact hω ⟨sideGood_congr O true t.1 v a (hpp t.1 ht.1) hgood.1,
        sideGood_congr O false t.2 v a (hpp t.2 ht.2) hgood.2⟩
    · exact hω
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
  have hbad : ∀ t ∈ T, μ.real (Bad t) ≤ 6 * a := by
    rintro ⟨A₀, R₀⟩ ht
    obtain ⟨hA, hR⟩ := Finset.mem_product.1 ht
    have hAP := Finset.mem_powerset.1 hA
    have hRP := Finset.mem_powerset.1 hR
    have hs1 := side_tails (μ := μ) O true A₀ (hside A₀ hAP).1 (hside A₀ hAP).2 ha0 ha1
    have hs2 := side_tails (μ := μ) O false R₀ (hside R₀ hRP).1 (hside R₀ hRP).2 ha0 ha1
    have hsub : Bad (A₀, R₀) ⊆ {ω | ¬ sideGood O true A₀ v a ω}
        ∪ {ω | ¬ sideGood O false R₀ v a ω} := by
      intro ω hω
      simp only [hBaddef, if_pos (And.intro hAP hRP), Set.mem_setOf_eq] at hω
      by_contra hc
      simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc
      exact hω hc
    exact le_trans (measureReal_mono hsub (measure_ne_top _ _))
      (le_trans (measureReal_union_le _ _) (by linarith))
  have hmain := selection_congr_bound O C Q hdisj T (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset P, Finset.empty_mem_powerset P⟩)
    sel hsel hcongr Bad hBadcongr (6 * a) (by positivity) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  simp only [Set.mem_setOf_eq] at hω ⊢
  have hselP : (sel ω).1 ⊆ P ∧ (sel ω).2 ⊆ P := ⟨Finset.filter_subset _ _, Finset.filter_subset _ _⟩
  simp only [hBaddef, if_pos hselP, Set.mem_setOf_eq]
  exact hω

/-! ### The sides' counts the other way -/

open scoped Classical in
/-- A side's counts within `t` and `tu` of their means the other way: the seed's count against
the side both ways, and every reference's count of disagreements from below. -/
def sideLive (O : Oracle μ S) (accept : Bool) (A R : Finset S) (t tu : ℝ) (ω : Ω) : Prop :=
  |(againstOn O.mq accept A ω : ℝ) - ∑ p ∈ A, μ[againstVar O (sideClass accept) p]| ≤ t
  ∧ ∀ v ∈ R, ∑ p ∈ A, μ[apartVar O v p] - tu
      ≤ ((A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card : ℝ)

open scoped Classical in
theorem side_live_tails (O : Oracle μ S) (accept : Bool) (A R : Finset S)
    (hne : ∀ v ∈ R, ∀ p ∈ A, p * v ≠ p)
    (hblk : ∀ v ∈ R, ∀ p ∈ A, ∀ p' ∈ A, p ≠ p' → Disjoint ({p, p * v} : Finset S) {p', p' * v})
    {t tu : ℝ} (ht : 0 ≤ t) (htu : 0 ≤ tu) (N : ℕ) (hN : A.card ≤ N) :
    μ.real {ω | ¬ sideLive O accept A R t tu ω}
      ≤ 2 * Real.exp (-2 * t ^ 2 / N) + R.card * Real.exp (-2 * tu ^ 2 / N) := by
  classical
  set c : ℝ := sideClass accept with hcdef
  rcases Nat.eq_zero_or_pos A.card with hA0 | hApos
  · have hAe : A = ∅ := Finset.card_eq_zero.1 hA0
    have hz : {ω | ¬ sideLive O accept A R t tu ω} = (∅ : Set Ω) := by
      ext ω
      simp only [sideLive, hAe, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_and,
        not_forall, Finset.sum_empty, Finset.filter_empty, Finset.card_empty, CharP.cast_eq_zero,
        againstOn, sub_zero, abs_zero, zero_sub]
      exact ⟨ht, fun ⟨v, _, h⟩ => h (by linarith)⟩
    rw [hz, measureReal_empty]
    positivity
  set n : ℕ := A.card with hndef
  have hnR : (0 : ℝ) < n := by exact_mod_cast hApos
  have hNR : (0 : ℝ) < N := lt_of_lt_of_le hnR (by exact_mod_cast hN)
  set X : A → Ω → ℝ := fun i => againstVar O c i.1 with hX
  have hXind : iIndepFun X μ := by
    refine iIndepFun_of_blocks O (fun i : A => ({i.1} : Finset S)) ?_ X (fun i => ?_)
    · intro i j hij
      simp only [Finset.disjoint_singleton]
      exact fun h => hij (Subtype.ext h)
    · exact againstVar_meas_alg O c (by simp)
  have hXm : ∀ i, AEMeasurable (X i) μ := fun i =>
    (againstVar_meas_alg O c (T := Set.univ) (Set.mem_univ _)).mono (noiseAlg_le O _) le_rfl
      |>.aemeasurable
  have hXi : ∀ i, ∀ᵐ ω ∂μ, X i ω ∈ Set.Icc (0 : ℝ) 1 := fun i =>
    Filter.Eventually.of_forall (fun ω => againstVar_icc O c _ ω)
  set EX : ℝ := ∑ p ∈ A, μ[againstVar O c p] with hEX
  have hcard : ((Finset.univ : Finset A).card : ℝ) = n := by simp [hndef]
  have hEXs : ∑ i : A, μ[X i] = EX := Finset.sum_coe_sort A (fun p => μ[againstVar O c p])
  have hexp : Real.exp (-2 * (n : ℝ) * (t / n) ^ 2) ≤ Real.exp (-2 * t ^ 2 / N) := by
    refine Real.exp_le_exp.2 ?_
    rw [div_pow, show -2 * (n : ℝ) * (t ^ 2 / (n : ℝ) ^ 2) = -(2 * t ^ 2 / n) by field_simp]
    have hle : 2 * t ^ 2 / N ≤ 2 * t ^ 2 / n :=
      div_le_div_of_nonneg_left (by positivity) hnR (by exact_mod_cast hN)
    rw [show -2 * t ^ 2 / (N : ℝ) = -(2 * t ^ 2 / N) by ring]
    linarith
  have hexpu : Real.exp (-2 * (n : ℝ) * (tu / n) ^ 2) ≤ Real.exp (-2 * tu ^ 2 / N) := by
    refine Real.exp_le_exp.2 ?_
    rw [div_pow, show -2 * (n : ℝ) * (tu ^ 2 / (n : ℝ) ^ 2) = -(2 * tu ^ 2 / n) by field_simp]
    have hle : 2 * tu ^ 2 / N ≤ 2 * tu ^ 2 / n :=
      div_le_div_of_nonneg_left (by positivity) hnR (by exact_mod_cast hN)
    rw [show -2 * tu ^ 2 / (N : ℝ) = -(2 * tu ^ 2 / N) by ring]
    linarith
  have htn : 0 ≤ t / n := div_nonneg ht hnR.le
  have htun : 0 ≤ tu / n := div_nonneg htu hnR.le
  have hB1 : μ.real {ω | ∑ i : A, X i ω ≤ (n : ℝ) * (EX / n - t / n)}
      ≤ Real.exp (-2 * t ^ 2 / N) := by
    have h := sumLower_le X Finset.univ (EX / n) (t / n) hXm hXind hXi
      (by rw [hcard, hEXs, mul_div_cancel₀ _ hnR.ne']) htn
    rw [hcard] at h
    exact le_trans h hexp
  have hB2 : μ.real {ω | (n : ℝ) * (EX / n + t / n) ≤ ∑ i : A, X i ω}
      ≤ Real.exp (-2 * t ^ 2 / N) := by
    have h := sumUpper_le X Finset.univ (EX / n) (t / n) hXm hXind hXi
      (by rw [hcard, hEXs, mul_div_cancel₀ _ hnR.ne']) htn
    rw [hcard] at h
    exact le_trans h hexp
  have hB3 : ∀ v ∈ R, μ.real {ω | ∑ i : A, apartVar O v i.1 ω
        ≤ (n : ℝ) * ((∑ p ∈ A, μ[apartVar O v p]) / n - tu / n)}
      ≤ Real.exp (-2 * tu ^ 2 / N) := by
    intro v hv
    have hYind : iIndepFun (fun i : A => apartVar O v i.1) μ := by
      refine iIndepFun_of_blocks O (fun i : A => ({i.1, i.1 * v} : Finset S)) ?_ _ (fun i => ?_)
      · intro i j hij
        exact hblk v hv i.1 i.2 j.1 j.2 (fun h => hij (Subtype.ext h))
      · exact apartVar_meas_alg O (by simp) (by simp)
    have hs : ∑ i : A, μ[apartVar O v i.1] = ∑ p ∈ A, μ[apartVar O v p] :=
      Finset.sum_coe_sort A (fun p => μ[apartVar O v p])
    have h := sumLower_le (fun i : A => apartVar O v i.1) Finset.univ
      ((∑ p ∈ A, μ[apartVar O v p]) / n) (tu / n)
      (fun i => (apartVar_meas_alg O (T := Set.univ) (Set.mem_univ _) (Set.mem_univ _)).mono
        (noiseAlg_le O _) le_rfl |>.aemeasurable)
      hYind (fun i => Filter.Eventually.of_forall (fun ω => apartVar_icc O v _ ω))
      (by rw [hcard, hs, mul_div_cancel₀ _ hnR.ne']) htun
    rw [hcard] at h
    exact le_trans h hexpu
  have hsub : {ω | ¬ sideLive O accept A R t tu ω}
      ⊆ ({ω | ∑ i : A, X i ω ≤ (n : ℝ) * (EX / n - t / n)}
          ∪ {ω | (n : ℝ) * (EX / n + t / n) ≤ ∑ i : A, X i ω})
        ∪ ⋃ v ∈ R, {ω | ∑ i : A, apartVar O v i.1 ω
            ≤ (n : ℝ) * ((∑ p ∈ A, μ[apartVar O v p]) / n - tu / n)} := by
    intro ω hω
    by_contra hc'
    simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_iUnion, not_or, not_exists, not_le]
      at hc'
    obtain ⟨⟨h1, h2⟩, h3⟩ := hc'
    apply hω
    have e1 : (n : ℝ) * (EX / n - t / n) = EX - t := by field_simp
    have e2 : (n : ℝ) * (EX / n + t / n) = EX + t := by field_simp
    rw [e1, Finset.sum_coe_sort A (fun p => againstVar O c p ω)] at h1
    rw [e2, Finset.sum_coe_sort A (fun p => againstVar O c p ω)] at h2
    refine ⟨?_, fun v hv => ?_⟩
    · rw [againstOn_eq_sum, abs_le]
      exact ⟨by linarith, by linarith⟩
    · have h := h3 v hv
      rw [Finset.sum_coe_sort A (fun p => apartVar O v p ω),
        show (n : ℝ) * ((∑ p ∈ A, μ[apartVar O v p]) / n - tu / n)
          = (∑ p ∈ A, μ[apartVar O v p]) - tu by field_simp,
        ← card_filter_eq_sum_apartVar] at h
      exact h.le
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ (Real.exp (-2 * t ^ 2 / N) + Real.exp (-2 * t ^ 2 / N))
          + ∑ v ∈ R, Real.exp (-2 * tu ^ 2 / N) :=
        add_le_add (le_trans (measureReal_union_le _ _) (add_le_add hB1 hB2))
          (le_trans (measureReal_biUnion_finset_le _ _)
            (Finset.sum_le_sum (fun v hv => hB3 v hv)))
    _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]; ring

open scoped Classical in
/-- Both sides of a sample live against the references `R`. -/
def sidesLive (O : Oracle μ S) (lo hi : ℕ) (F P R : Finset S) (t tu : ℝ) (ω : Ω) : Prop :=
  sideLive O true (acceptSide O.mq hi F P ω) R t tu ω
    ∧ sideLive O false (rejectSide O.mq lo F P ω) R t tu ω

lemma sideLive_congr_mq (O : Oracle μ S) (accept : Bool) (A R : Finset S) (t tu : ℝ) {ω ω' : Ω}
    (h : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1)
      ∧ ∀ v ∈ R, (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    sideLive O accept A R t tu ω → sideLive O accept A R t tu ω' := by
  classical
  have ha : againstOn O.mq accept A ω = againstOn O.mq accept A ω' := by
    unfold againstOn
    exact congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [(h p hp).1]))
  rintro ⟨h1, h2⟩
  refine ⟨by rw [← ha]; exact h1, fun v hv => ?_⟩
  have hb : (A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card
      = (A.filter (fun p => ¬ ((O.mq (p * v) ω' = 1) ↔ (O.mq p ω' = 1)))).card :=
    congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [(h p hp).1, (h p hp).2 v hv]))
  rw [← hb]
  exact h2 v hv

open scoped Classical in
/-- Both sides of a sample live, with the sample, the voters and the references fixed and the
cut chosen by the votes. -/
theorem sidesLive_whp (O : Oracle μ S) (P F R : Finset S) (lo hi : ℕ)
    (hinj : ∀ p ∈ P, ∀ p' ∈ P, ∀ u ∈ insert 1 (F ∪ R), ∀ u' ∈ insert 1 (F ∪ R),
      p * u = p' * u' → p = p' ∧ u = u')
    (hFR : Disjoint F R) (h1F : (1 : S) ∉ F) (h1R : (1 : S) ∉ R) {t tu : ℝ} (ht : 0 ≤ t)
    (htu : 0 ≤ tu) (N : ℕ) (hN : P.card ≤ N) :
    μ.real {ω | ¬ sidesLive O lo hi F P R t tu ω}
      ≤ 2 * (2 * Real.exp (-2 * t ^ 2 / N) + R.card * Real.exp (-2 * tu ^ 2 / N)) := by
  classical
  have h1mem : (1 : S) ∈ insert 1 (F ∪ R) := Finset.mem_insert_self _ _
  set E : ℝ := 2 * Real.exp (-2 * t ^ 2 / N) + R.card * Real.exp (-2 * tu ^ 2 / N) with hE
  have hE0 : 0 ≤ E := by positivity
  set C : Finset S := P ∪ (P ×ˢ R).image (fun q => q.1 * q.2) with hC
  set Q : Finset S := (P ×ˢ F).image (fun q => q.1 * q.2) with hQ
  have hdisj : Disjoint (↑C : Set S) (↑Q : Set S) := by
    rw [Finset.disjoint_coe]
    refine Finset.disjoint_left.2 (fun w hwC hwQ => ?_)
    obtain ⟨⟨p', u⟩, hpu, rfl⟩ := Finset.mem_image.1 hwQ
    obtain ⟨hp', hu⟩ := Finset.mem_product.1 hpu
    have humem : u ∈ insert 1 (F ∪ R) := Finset.mem_insert_of_mem (Finset.mem_union_left _ hu)
    rcases Finset.mem_union.1 hwC with hp | hw
    · have := (hinj _ hp p' hp' 1 h1mem u humem (mul_one _)).2
      exact h1F (this ▸ hu)
    · obtain ⟨⟨p, r⟩, hpr, hpreq⟩ := Finset.mem_image.1 hw
      obtain ⟨hp, hr⟩ := Finset.mem_product.1 hpr
      have hrmem : r ∈ insert 1 (F ∪ R) := Finset.mem_insert_of_mem (Finset.mem_union_right _ hr)
      have := (hinj p hp p' hp' r hrmem u humem hpreq).2
      exact Finset.disjoint_left.1 hFR hu (this ▸ hr)
  set T : Finset (Finset S × Finset S) := P.powerset ×ˢ P.powerset with hT
  set sel : Ω → Finset S × Finset S := fun ω =>
    (acceptSide O.mq hi F P ω, rejectSide O.mq lo F P ω) with hseldef
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
  set Bad : Finset S × Finset S → Set Ω := fun t' =>
    if t'.1 ⊆ P ∧ t'.2 ⊆ P then
      {ω | ¬ (sideLive O true t'.1 R t tu ω ∧ sideLive O false t'.2 R t tu ω)}
    else ∅ with hBaddef
  have hBadcongr : ∀ t' ω ω', (∀ w ∈ C, O.noise w ω = O.noise w ω') → ω ∈ Bad t' →
      ω' ∈ Bad t' := by
    intro t' ω ω' h hω
    simp only [hBaddef] at hω ⊢
    split_ifs at hω ⊢ with ht'
    · have hpp : ∀ D ⊆ P, ∀ p ∈ D, (O.mq p ω' = 1 ↔ O.mq p ω = 1)
          ∧ ∀ v ∈ R, (O.mq (p * v) ω' = 1 ↔ O.mq (p * v) ω = 1) := fun D hD p hp =>
        ⟨by rw [mq_congr O (h p (Finset.mem_union_left _ (hD hp)))],
          fun v hv => by rw [mq_congr O (h _ (Finset.mem_union_right _
            (Finset.mem_image.2 ⟨(p, v), Finset.mem_product.2 ⟨hD hp, hv⟩, rfl⟩)))]⟩
      intro hgood
      exact hω ⟨sideLive_congr_mq O true t'.1 R t tu (hpp t'.1 ht'.1) hgood.1,
        sideLive_congr_mq O false t'.2 R t tu (hpp t'.2 ht'.2) hgood.2⟩
    · exact hω
  have hside : ∀ D ⊆ P, (∀ v ∈ R, ∀ p ∈ D, p * v ≠ p)
      ∧ ∀ v ∈ R, ∀ p ∈ D, ∀ p' ∈ D, p ≠ p' → Disjoint ({p, p * v} : Finset S) {p', p' * v} := by
    intro D hD
    have hvmem : ∀ v ∈ R, v ∈ insert 1 (F ∪ R) := fun v hv =>
      Finset.mem_insert_of_mem (Finset.mem_union_right _ hv)
    refine ⟨fun v hv p hp heq => h1R ?_, fun v hv p hp p' hp' hne => ?_⟩
    · have := (hinj p (hD hp) p (hD hp) v (hvmem v hv) 1 h1mem (by rw [mul_one]; exact heq)).2
      exact this ▸ hv
    · rw [Finset.disjoint_left]
      intro w hw hw'
      simp only [Finset.mem_insert, Finset.mem_singleton] at hw hw'
      rcases hw with rfl | rfl <;> rcases hw' with h | h
      · exact hne h
      · exact hne (hinj _ (hD hp) p' (hD hp') 1 h1mem v (hvmem v hv) (by rw [mul_one]; exact h)).1
      · exact hne (hinj p (hD hp) p' (hD hp') v (hvmem v hv) 1 h1mem (by rw [mul_one]; exact h)).1
      · exact hne (hinj p (hD hp) p' (hD hp') v (hvmem v hv) v (hvmem v hv) h).1
  have hbad : ∀ t' ∈ T, μ.real (Bad t') ≤ 2 * E := by
    rintro ⟨A₀, R₀⟩ ht'
    obtain ⟨hA, hR⟩ := Finset.mem_product.1 ht'
    have hAP := Finset.mem_powerset.1 hA
    have hRP := Finset.mem_powerset.1 hR
    have hs1 := side_live_tails (μ := μ) O true A₀ R (hside A₀ hAP).1 (hside A₀ hAP).2 ht htu N
      (le_trans (Finset.card_le_card hAP) hN)
    have hs2 := side_live_tails (μ := μ) O false R₀ R (hside R₀ hRP).1 (hside R₀ hRP).2 ht htu N
      (le_trans (Finset.card_le_card hRP) hN)
    have hsub : Bad (A₀, R₀) ⊆ {ω | ¬ sideLive O true A₀ R t tu ω}
        ∪ {ω | ¬ sideLive O false R₀ R t tu ω} := by
      intro ω hω
      simp only [hBaddef, if_pos (And.intro hAP hRP), Set.mem_setOf_eq] at hω
      by_contra hc
      simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_not] at hc
      exact hω hc
    exact le_trans (measureReal_mono hsub (measure_ne_top _ _))
      (le_trans (measureReal_union_le _ _) (by linarith))
  have hmain := selection_congr_bound O C Q hdisj T (∅, ∅)
    (Finset.mem_product.2 ⟨Finset.empty_mem_powerset P, Finset.empty_mem_powerset P⟩)
    sel hsel hcongr Bad hBadcongr (2 * E) (by positivity) hbad
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) hmain
  simp only [Set.mem_setOf_eq] at hω ⊢
  have hselP : (sel ω).1 ⊆ P ∧ (sel ω).2 ⊆ P := ⟨Finset.filter_subset _ _, Finset.filter_subset _ _⟩
  simp only [hBaddef, if_pos hselP, Set.mem_setOf_eq]
  exact hω

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

lemma sideGood_congr_mq (O : Oracle μ S) (accept : Bool) (A : Finset S) (v : S) (a : ℝ)
    {ω ω' : Ω}
    (h : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1) ∧ (O.mq (p * v) ω = 1 ↔ O.mq (p * v) ω' = 1)) :
    sideGood O accept A v a ω → sideGood O accept A v a ω' := by
  classical
  have ha : againstOn O.mq accept A ω = againstOn O.mq accept A ω' := by
    unfold againstOn
    exact congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [(h p hp).1]))
  have hb : (A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card
      = (A.filter (fun p => ¬ ((O.mq (p * v) ω' = 1) ↔ (O.mq p ω' = 1)))).card :=
    congrArg Finset.card (Finset.filter_congr (fun p hp => by rw [(h p hp).1, (h p hp).2]))
  unfold sideGood
  rw [ha, hb]
  exact id

open scoped Classical in
/-- The sides of a sample `A` read against the family `F`, decided by the reads of `A` and
`A · C`, for `F` and `v` in the pool `C`. -/
lemma sidesGood_congr_mq (O : Oracle μ S) (lo hi : ℕ) (F A C : Finset S) (hone : (1 : S) ∈ C)
    (hF : F ⊆ C) {v : S} (hv : v ∈ C) (a : ℝ) {ω ω' : Ω}
    (h : ∀ w ∈ readSet A C, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    sidesGood O lo hi F A v a ω → sidesGood O lo hi F A v a ω' := by
  classical
  have hAC : ∀ p ∈ A, ∀ u ∈ C, (O.mq (p * u) ω = 1 ↔ O.mq (p * u) ω' = 1) := fun p hp u hu =>
    h _ (mem_readSet hp hu)
  have hA : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun p hp => by simpa using hAC p hp 1 hone
  have hvc : ∀ p ∈ A, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_congr O F p (fun u hu => hAC p hp u (hF hu))
  have hacc : acceptSide O.mq hi F A ω = acceptSide O.mq hi F A ω' :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  have hrej : rejectSide O.mq lo F A ω = rejectSide O.mq lo F A ω' :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  rintro ⟨h1, h2⟩
  refine ⟨?_, ?_⟩
  · rw [← hacc]
    exact sideGood_congr_mq O true _ v a (fun p hp =>
      ⟨hA p (Finset.mem_filter.1 hp).1, hAC p (Finset.mem_filter.1 hp).1 v hv⟩) h1
  · rw [← hrej]
    exact sideGood_congr_mq O false _ v a (fun p hp =>
      ⟨hA p (Finset.mem_filter.1 hp).1, hAC p (Finset.mem_filter.1 hp).1 v hv⟩) h2

open scoped Classical in
/-- At a look of `m` certification draws of population `j`: the references hold an
accept-preserving suffix, and for none of them are both of the sample's sides good. -/
noncomputable def sidesBad (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (j : J) (B : State) (m : ℕ) (a : ℝ) : Set (Run Ω S J) :=
  {x | (∃ v ∈ refsBy rule O.mq populations x B, ∀ p, O.label (p * v) = O.label p)
    ∧ ∀ v ∈ refsBy rule O.mq populations x B, (∀ p, O.label (p * v) = O.label p) →
      ¬ sidesGood O B.lo B.hi (clusterBy rule O.mq populations x B)
        (certOf (prefixesAt populations B.npref x) j m x) v a (oracleNoise x)}

open scoped Classical in
noncomputable def sidesBadOf (rule : Clusterer S) (O : Oracle μ S) (B : State) (a : ℝ)
    (P C A : Finset S) (ω : Ω) : Prop :=
  (∃ v ∈ C \ insert 1 (clusterOf rule O B.sc B.scd P C B.k ω), ∀ p, O.label (p * v) = O.label p)
    ∧ ∀ v ∈ C \ insert 1 (clusterOf rule O B.sc B.scd P C B.k ω),
      (∀ p, O.label (p * v) = O.label p) →
      ¬ sidesGood O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k ω) A v a ω

open scoped Classical in
lemma sidesBadOf_congr (O : Oracle μ S) (B : State) (a : ℝ) (P C A : Finset S)
    (hone : (1 : S) ∈ C) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P C ∪ readSet A C, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    sidesBadOf rule O B a P C A ω → sidesBadOf rule O B a P C A ω' := by
  have hfam : clusterOf rule O B.sc B.scd P C B.k ω = clusterOf rule O B.sc B.scd P C B.k ω' :=
    clusterOf_congr_mq O B.sc B.scd P C B.k hone (fun w hw => h w (Finset.mem_union_left _ hw))
  have hsub : clusterOf rule O B.sc B.scd P C B.k ω' ⊆ C :=
    clusterOf_subset O B.sc B.scd P C B.k ω' hone
  unfold sidesBadOf
  rw [hfam]
  rintro ⟨hex, hall⟩
  refine ⟨hex, fun v hv hap hgood => hall v hv hap ?_⟩
  exact sidesGood_congr_mq O B.lo B.hi _ A C hone hsub (Finset.mem_sdiff.1 hv).1 a
    (fun w hw => (h w (Finset.mem_union_right _ hw)).symm) hgood

open scoped Classical in
lemma measurableSet_sidesBad (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (m : ℕ) (a : ℝ) : MeasurableSet (sidesBad rule O populations j B m a) := by
  classical
  have hR : ∀ P C A : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω | sidesBadOf rule O B a P C A ω} else (∅ : Set (Run Ω S J))) := by
    intro P C A
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_readCongr O _ _
        (fun ω ω' h => sidesBadOf_congr O B a P C A hone h))
    · exact MeasurableSet.empty
  have hrw : sidesBad rule O populations j B m a
      = {x : Run Ω S J | x ∈ (fun P C A => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω | sidesBadOf rule O B a P C A ω} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j m x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_certAt populations j B m _ hR

open scoped Classical in
/-- `sidesGood_whp` over the run.  With the draws fixed the family is chosen by the table's
bits, and the sides read only the sample's strings, which the table never reads. -/
theorem measureReal_sidesBad_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (j : J) (hj : j ∈ populations) (B : State) (m : ℕ) {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    (streamMeasure μ Pp Pc Dsf).real (sidesBad rule O populations j B m a) ≤ 6 * a := by
  classical
  have hEnn : streamMeasure μ Pp Pc Dsf (sidesBad rule O populations j B m a)
      ≤ ENNReal.ofReal (6 * a) := by
    refine streamMeasure_slice_le Pp Pc Dsf _ (measurableSet_sidesBad O populations j B m a) _ ?_
    filter_upwards [ae_draws_mem_Pre Pp Pc Dsf Pre populations hsuppP,
      ae_cert_mem_Pre Pp Pc Dsf Pre populations hsuppC, ae_sfx_mem_Suf Pp Pc Dsf Suf hsuppSf]
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
    have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ sidesBad rule O populations j B m a}
        = {ω | sidesBadOf rule O B a Pd Cd Ad ω} := rfl
    rw [hsec]
    set sel : Ω → Finset S := fun ω => clusterOf rule O B.sc B.scd Pd Cd B.k ω with hseldef
    set Bad : Finset S → Set Ω := fun F =>
      if hF : F ⊆ Cd ∧ (1 : S) ∉ F ∧ ∃ v ∈ Cd \ insert 1 F, ∀ p, O.label (p * v) = O.label p then
        {ω | ¬ sidesGood O B.lo B.hi F Ad hF.2.2.choose a ω}
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
        · intro hgood
          exact hω (sidesGood_congr_mq O B.lo B.hi F Ad Cd hone hF.1
            (Finset.mem_sdiff.1 hF.2.2.choose_spec.1).1 a
            (fun w hw => by rw [mq_congr O (h w hw)]) hgood)
        · exact hω)
      (6 * a) (by positivity) (fun F _ => by
        simp only [hBaddef]
        split_ifs with hF
        · obtain ⟨hvR, hap⟩ := hF.2.2.choose_spec
          set v := hF.2.2.choose
          have hvC := (Finset.mem_sdiff.1 hvR).1
          have hvF : v ∉ insert 1 F := (Finset.mem_sdiff.1 hvR).2
          refine sidesGood_whp O Ad F B.lo B.hi ?_ (fun h => hvF (Finset.mem_insert_of_mem h))
            hF.2.1 (fun h => hvF (h ▸ Finset.mem_insert_self _ _)) ha0 ha1
          intro p hp p' hp' u hu u' hu' heq
          have hmem : ∀ u ∈ insert 1 (insert v F), u ∈ insert 1 Suf := by
            intro u hu
            rcases Finset.mem_insert.1 hu with rfl | hu
            · exact Set.mem_insert _ _
            · rcases Finset.mem_insert.1 hu with rfl | hu
              · exact hV _ hvC
              · exact hV u (hF.1 hu)
          have hpp := hflat p (hA p hp) p' (hA p' hp') u (hmem u hu) u' (hmem u' hu') heq
          subst hpp
          exact ⟨rfl, mul_left_cancel heq⟩
        · rw [measureReal_empty]; positivity)
    have hsub : {ω | sidesBadOf rule O B a Pd Cd Ad ω} ⊆ {ω | ω ∈ Bad (sel ω)} := by
      rintro ω ⟨hex, hall⟩
      change ω ∈ Bad (sel ω)
      have hF : sel ω ⊆ Cd ∧ (1 : S) ∉ sel ω
          ∧ ∃ v ∈ Cd \ insert 1 (sel ω), ∀ p, O.label (p * v) = O.label p :=
        ⟨clusterOf_subset O B.sc B.scd Pd Cd B.k ω hone, Finset.notMem_erase _ _, hex⟩
      simp only [hBaddef, dif_pos hF]
      exact hall _ hF.2.2.choose_spec.1 hF.2.2.choose_spec.2
    calc μ {ω | sidesBadOf rule O B a Pd Cd Ad ω} ≤ μ {ω | ω ∈ Bad (sel ω)} := measure_mono hsub
      _ = ENNReal.ofReal (μ.real {ω | ω ∈ Bad (sel ω)}) :=
          (ENNReal.ofReal_toReal (measure_ne_top μ _)).symm
      _ ≤ ENNReal.ofReal (6 * a) := ENNReal.ofReal_le_ofReal hmain
  rw [measureReal_def]
  calc (streamMeasure μ Pp Pc Dsf (sidesBad rule O populations j B m a)).toReal
      ≤ (ENNReal.ofReal (6 * a)).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = 6 * a := ENNReal.toReal_ofReal (by positivity)

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
theorem measureReal_rawShortAt_le (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (O : Oracle μ S) (populations : Finset J) (j : J) (Dj : Measure S) [IsProbabilityMeasure Dj]
    (hPc : Pc j = iidStream Dj) (B : State) {lim a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1)
    (N : ℕ) (hN : 0 < N) :
    (streamMeasure μ Pp Pc Dsf).real (rawShortAt rule O populations Dj j B lim a N)
      ≤ sizeLevel a N := by
  classical
  set E : ℝ := sizeLevel a N with hEdef
  have hE0 : 0 ≤ E := sizeLevel_nonneg ha0.le N
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  have hEnn : streamMeasure μ Pp Pc Dsf (rawShortAt rule O populations Dj j B lim a N)
      ≤ ENNReal.ofReal E := by
    refine streamMeasure_slice_cert_le Pp Pc Dsf _
      (measurableSet_rawShortAt O populations Dj j B lim a N) _ ?_
    intro y
    set F : Finset S :=
      clusterBy rule O.mq populations ((y.1, (y.2, fun _ _ => (1 : S))) : Run Ω S J) B with hF
    set W : Set S := {p | ¬ cutCorrect O B.lo B.hi F p y.1} with hW
    have hFeq : ∀ c : J → ℕ → S,
        clusterBy rule O.mq populations ((y.1, (y.2, c)) : Run Ω S J) B = F := fun c => rfl
    by_cases hmass : lim < Dj.real W
    · set t : ℝ := hoeffDev N (sizeLevel a N) / N with ht
      have ht0 : 0 ≤ t := div_nonneg (Real.sqrt_nonneg _) hNR.le
      have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
            ∈ rawShortAt rule O populations Dj j B lim a N}
          ⊆ {c | (((Finset.range N).filter (fun i => c j i ∈ W)).card : ℝ)
            ≤ (N : ℝ) * (Dj.real W - t)} := by
        rintro c ⟨-, hcount⟩
        have h := card_filter_fin_eq N (fun p => p ∈ W) (fun i => c j i)
        show (((Finset.range N).filter (fun i => c j i ∈ W)).card : ℝ)
          ≤ (N : ℝ) * (Dj.real W - t)
        rw [← h]
        have e : (N : ℝ) * (Dj.real W - t) = N * Dj.real W - hoeffDev N (sizeLevel a N) := by
          rw [ht]; field_simp
        rw [e]
        refine le_trans hcount ?_
        nlinarith
      refine le_trans (measure_mono hsec) ?_
      rw [← ENNReal.ofReal_toReal (measure_ne_top
          (Measure.pi Pc) _), ← measureReal_def]
      refine ENNReal.ofReal_le_ofReal ?_
      have hcert := cert_hits_wrongSet Pc j Dj hPc N W (Dj.real W) t measureReal_nonneg ht0 le_rfl
      have hexp : Real.exp (-2 * (N : ℝ) * t ^ 2) = E :=
        exp_hoeffDev N hN (sizeLevel_pos ha0 N) (sizeLevel_le ha0.le ha1 N)
      rw [hexp] at hcert
      convert hcert using 3
      funext c
      congr!
    · have hsec : {c : J → ℕ → S | ((y.1, (y.2, c)) : Run Ω S J)
          ∈ rawShortAt rule O populations Dj j B lim a N} = (∅ : Set (J → ℕ → S)) := by
        ext c
        simp only [Set.mem_empty_iff_false, iff_false]
        rintro ⟨hm, -⟩
        rw [hFeq c] at hm
        exact hmass hm
      simp [hsec]
  rw [measureReal_def]
  calc (streamMeasure μ Pp Pc Dsf (rawShortAt rule O populations Dj j B lim a N)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

/-- Over every count of draws together. -/
theorem measureReal_rawShort_le (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (O : Oracle μ S) (populations : Finset J) (j : J) (Dj : Measure S) [IsProbabilityMeasure Dj]
    (hPc : Pc j = iidStream Dj) (B : State) {lim a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    (streamMeasure μ Pp Pc Dsf).real (⋃ N : ℕ, rawShortAt rule O populations Dj j B lim a (N + 1))
      ≤ a := by
  have hle : streamMeasure μ Pp Pc Dsf (⋃ N : ℕ, rawShortAt rule O populations Dj j B lim a (N + 1))
      ≤ ENNReal.ofReal a := by
    refine le_trans (measure_iUnion_le _) ?_
    refine ENNReal.tsum_le_of_sum_range_le (fun n => ?_)
    calc ∑ i ∈ Finset.range n,
          streamMeasure μ Pp Pc Dsf (rawShortAt rule O populations Dj j B lim a (i + 1))
        ≤ ∑ i ∈ Finset.range n, ENNReal.ofReal (sizeLevel a (1 + i)) := by
          refine Finset.sum_le_sum (fun i _ => ?_)
          rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def, add_comm 1 i]
          exact ENNReal.ofReal_le_ofReal
            (measureReal_rawShortAt_le Pp Pc Dsf O populations j Dj hPc B ha0 ha1 (i + 1)
              (Nat.succ_pos i))
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

/-- A sample found within its first `n + r` draws finds every smaller one within its first
`L + r`. -/
lemma le_card_acceptedAt_of_le (T : Finset S) (y : ℕ → S) {L n r : ℕ} (hL : L ≤ n)
    (h : n ≤ (acceptedAt T y (n + r)).card) : L ≤ (acceptedAt T y (L + r)).card := by
  have := card_acceptedAt_add_le T y (show L + r ≤ n + r by omega)
  omega

open scoped Classical in
/-- The gate certifies, at some look, a family that misclassifies more than `lim + s` of
population `j`, though the references hold an accept-preserving suffix and the certification
draws redraw at most `r ≤ s·npref` times before their largest look.  Then some look's sides
failed, or the draws undercounted the population. -/
theorem falseCertify_subset (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀)
    (populations : Finset J) (Dj : Measure S) (j : J) (B : State) (lim s a : ℝ) (r : ℕ)
    (hs : 0 ≤ s) (hr : (r : ℝ) ≤ s * B.npref) :
    {x : Run Ω S J |
      (∃ e ≤ B.npref, certifiesAt O.mq η₀ lim a B.lo B.hi (clusterBy rule O.mq populations x B)
        (refsBy rule O.mq populations x B)
        (certOf (prefixesAt populations B.npref x) j (B.npref + e) x) (oracleNoise x))
      ∧ (∃ v ∈ refsBy rule O.mq populations x B, ∀ p, O.label (p * v) = O.label p)
      ∧ lim + s < Dj.real {p | ¬ cutCorrect O B.lo B.hi (clusterBy rule O.mq populations x B) p
          (oracleNoise x)}
      ∧ 2 * B.npref ≤ (acceptedAt (prefixesAt populations B.npref x)
          (fun i => certPrefix j i x) (2 * B.npref + r)).card}
      ⊆ (⋃ e ∈ Finset.range (B.npref + 1),
            sidesBad rule O populations j B (B.npref + e) (sizeLevel a (B.npref + e)))
          ∪ ⋃ N : ℕ, rawShortAt rule O populations Dj j B (lim + s) a (N + 1) := by
  classical
  rintro x ⟨⟨e, he, hpos, hineq⟩, hv, hmass, hdraws⟩
  set T : Finset S := prefixesAt populations B.npref x with hT
  set y : ℕ → S := fun i => certPrefix j i x with hy
  set m : ℕ := B.npref + e with hm
  set F : Finset S := clusterBy rule O.mq populations x B with hF
  set R : Finset S := refsBy rule O.mq populations x B with hR
  by_cases hpop : x ∈ sidesBad rule O populations j B m (sizeLevel a m)
  · exact Or.inl (Set.mem_biUnion (Finset.mem_range.2 (by omega)) hpop)
  refine Or.inr ?_
  have hgood : ∀ {m' : ℕ} {j' : J} {a' : ℝ},
      x ∉ sidesBad rule O populations j' B m' a' →
      ∃ w ∈ R, (∀ p, O.label (p * w) = O.label p)
        ∧ sidesGood O B.lo B.hi F (certOf T j' m' x) w a' (oracleNoise x) := by
    intro m' j' a' hb
    by_contra hc
    push_neg at hc
    exact hb ⟨hv, fun w hw hap => hc w hw hap⟩
  obtain ⟨v, hvR, hapv, hgv⟩ := hgood hpop
  have hne1 : ∀ u ∈ R, ∀ p : S, p * u ≠ p := fun u hu p h =>
    (Finset.mem_sdiff.1 hu).2 (by
      have : u = 1 := mul_left_cancel (a := p) (by rw [h, mul_one])
      rw [this]; exact Finset.mem_insert_self _ _)
  have hcount := miscut_le_miscutBound O hη B.lo B.hi F R (certOf T j m x) hvR hapv
    (fun p _ => hne1 v hvR p) hgv
  have hmr : m ≤ (acceptedAt T y (m + r)).card :=
    le_card_acceptedAt_of_le T y (show m ≤ 2 * B.npref by omega) hdraws
  have hex : ∃ n, m ≤ (acceptedAt T y n).card := ⟨m + r, hmr⟩
  have hP : certOf T j m x = acceptedAt T y (Nat.find hex) := freshDraws_eq T y m hex
  have hPcard : (certOf T j m x).card = m := card_freshDraws T y m hex
  set N : ℕ := Nat.find hex with hNdef
  have hNle : N ≤ m + r := Nat.find_min' hex hmr
  have hmN : m ≤ N := le_trans (Nat.find_spec hex) (card_acceptedAt_le T y N)
  have hraw := card_range_filter_le T y
    (fun p => ¬ cutCorrect O B.lo B.hi F p (oracleNoise x)) N
  rw [← hP, hPcard] at hraw
  have hsubN : (((Finset.range m).filter
        (fun i => ¬ cutCorrect O B.lo B.hi F (y i) (oracleNoise x))).card : ℝ)
      ≤ ((Finset.range N).filter
        (fun i => ¬ cutCorrect O B.lo B.hi F (y i) (oracleNoise x))).card := by
    exact_mod_cast Finset.card_le_card (Finset.filter_subset_filter _
      (Finset.range_subset_range.2 hmN))
  rw [hPcard] at hineq hpos
  have hNR : (N : ℝ) ≤ m + r := by exact_mod_cast hNle
  have hrm : (r : ℝ) ≤ s * m := by
    have : (B.npref : ℝ) ≤ m := by rw [hm]; push_cast; linarith [(Nat.cast_nonneg e : (0 : ℝ) ≤ e)]
    nlinarith [mul_le_mul_of_nonneg_left this hs]
  refine Set.mem_iUnion.2 ⟨m - 1, ?_⟩
  rw [show m - 1 + 1 = m by omega]
  refine ⟨hmass, ?_⟩
  have hfin := card_filter_fin_eq m (fun p => ¬ cutCorrect O B.lo B.hi F p (oracleNoise x)) y
  rw [hfin]
  have : ((m : ℕ) : ℝ) * (lim + s) - hoeffDev m (sizeLevel a m)
      = m * lim - hoeffDev m (sizeLevel a m) + s * m := by ring
  rw [this]
  linarith

/-- What one population's looks and draws cost the gate. -/
theorem measureReal_popGate_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (j : J) (hj : j ∈ populations) (Dj : Measure S) [IsProbabilityMeasure Dj]
    (hPc : Pc j = iidStream Dj) (B : State) {lim a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    (streamMeasure μ Pp Pc Dsf).real ((⋃ e ∈ Finset.range (B.npref + 1),
        sidesBad rule O populations j B (B.npref + e) (sizeLevel a (B.npref + e)))
      ∪ ⋃ N : ℕ, rawShortAt rule O populations Dj j B lim a (N + 1)) ≤ 7 * a := by
  refine le_trans (measureReal_union_le _ _) ?_
  have h1 : (streamMeasure μ Pp Pc Dsf).real (⋃ e ∈ Finset.range (B.npref + 1),
      sidesBad rule O populations j B (B.npref + e) (sizeLevel a (B.npref + e))) ≤ 6 * a := by
    refine le_trans (measureReal_biUnion_finset_le _ _) ?_
    calc ∑ e ∈ Finset.range (B.npref + 1), (streamMeasure μ Pp Pc Dsf).real
          (sidesBad rule O populations j B (B.npref + e) (sizeLevel a (B.npref + e)))
        ≤ ∑ e ∈ Finset.range (B.npref + 1), 6 * sizeLevel a (B.npref + e) :=
          Finset.sum_le_sum (fun e _ => measureReal_sidesBad_le hflat O populations Pp Pc Dsf
            hsuppP hsuppC hsuppSf j hj B _ (sizeLevel_pos ha0 _) (sizeLevel_le ha0.le ha1 _))
      _ = 6 * ∑ e ∈ Finset.range (B.npref + 1), sizeLevel a (B.npref + e) := by
          rw [Finset.mul_sum]
      _ ≤ 6 * a := by
          have := sum_range_sizeLevel_shift_le ha0.le B.npref (B.npref + 1)
          linarith
  have h2 := measureReal_rawShort_le (rule := rule) Pp Pc Dsf O populations j Dj hPc B (lim := lim)
    ha0 ha1
  linarith

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

open scoped Classical in
/-- A live side's counts, read against its `n − k` right prefixes and `k` wrong ones. -/
lemma live_counts (O : Oracle μ S) (hη : O.η ≤ 1 / 2) (accept : Bool) (A R : Finset S)
    (hR : R.Nonempty) (hne : ∀ v ∈ R, ∀ p ∈ A, p * v ≠ p) {t tu : ℝ} {ω : Ω}
    (h : sideLive O accept A R t tu ω) :
    ((A.card : ℝ) - (A.filter (fun p => O.label p ≠ sideClass accept)).card)
          * rightRate O accept
        + (A.filter (fun p => O.label p ≠ sideClass accept)).card * (1 - wrongRate O accept) - t
      ≤ againstOn O.mq accept A ω
    ∧ (againstOn O.mq accept A ω : ℝ)
      ≤ ((A.card : ℝ) - (A.filter (fun p => O.label p ≠ sideClass accept)).card)
          * rightRate O accept
        + (A.filter (fun p => O.label p ≠ sideClass accept)).card * (1 - wrongRate O accept) + t
    ∧ 2 * (((A.card : ℝ) - (A.filter (fun p => O.label p ≠ sideClass accept)).card)
            * (rightRate O accept * (1 - rightRate O accept))
          + (A.filter (fun p => O.label p ≠ sideClass accept)).card
            * (wrongRate O accept * (1 - wrongRate O accept))) - tu
      ≤ refApart O.mq A R ω := by
  classical
  set c := sideClass accept with hc
  set x := rightRate O accept with hx
  set y := wrongRate O accept with hy
  have hcnt : ((A.filter (fun p => O.label p = c)).card : ℝ)
      = (A.card : ℝ) - (A.filter (fun p => ¬ O.label p = c)).card := by
    have := Finset.card_filter_add_card_filter_not (s := A) (fun p => O.label p = c)
    have : ((A.filter (fun p => O.label p = c)).card : ℝ)
        + ((A.filter (fun p => ¬ O.label p = c)).card : ℝ) = A.card := by exact_mod_cast this
    linarith
  have h1 : ∀ p ∈ A, μ[againstVar O c p] = if O.label p = c then x else 1 - y := by
    intro p _
    rw [againstVar_mean O (sideClass_bit accept) p, rate_of_label O accept p]
    split_ifs <;> rfl
  have hEa : ∑ p ∈ A, μ[againstVar O c p]
      = ((A.card : ℝ) - (A.filter (fun p => ¬ O.label p = c)).card) * x
        + (A.filter (fun p => ¬ O.label p = c)).card * (1 - y) := by
    rw [Finset.sum_congr rfl h1, Finset.sum_ite, Finset.sum_const, Finset.sum_const,
      nsmul_eq_mul, nsmul_eq_mul, hcnt]
  obtain ⟨hag, hap⟩ := h
  simp only [ne_eq]
  rw [hEa] at hag
  obtain ⟨hagl, hagh⟩ := abs_le.1 hag
  refine ⟨by linarith, by linarith, ?_⟩
  obtain ⟨v, hv, hveq⟩ := Finset.exists_mem_eq_inf' hR
    (fun v => (A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card)
  have href : refApart O.mq A R ω
      = (A.filter (fun p => ¬ ((O.mq (p * v) ω = 1) ↔ (O.mq p ω = 1)))).card := by
    unfold refApart; rw [dif_pos hR]; exact hveq
  rw [href]
  refine le_trans ?_ (hap v hv)
  have h2 : ∀ p ∈ A, (if O.label p = c then 2 * (x * (1 - x)) else 2 * (y * (1 - y)))
      ≤ μ[apartVar O v p] := by
    intro p hp
    have := apartVar_mean_ge O hη (hne v hv p hp)
    rw [rate_of_label O accept p] at this
    split_ifs at this ⊢ <;> linarith
  have hsum := Finset.sum_le_sum h2
  rw [Finset.sum_ite, Finset.sum_const, Finset.sum_const, nsmul_eq_mul, nsmul_eq_mul, hcnt]
    at hsum
  linarith

omit [IsProbabilityMeasure μ] in
open scoped Classical in
/-- With the band in order the two sides' wrong prefixes are distinct prefixes the cut gets
wrong. -/
lemma sides_wrong_le_miscut (O : Oracle μ S) (lo hi : ℕ) (hlohi : lo ≤ hi) (F P : Finset S)
    (ω : Ω) :
    ((((acceptSide O.mq hi F P ω).filter (fun p => O.label p ≠ sideClass true)).card
        + ((rejectSide O.mq lo F P ω).filter (fun p => O.label p ≠ sideClass false)).card : ℕ)
          : ℝ)
      ≤ ((P.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card : ℝ) := by
  classical
  have hdisj : Disjoint ((acceptSide O.mq hi F P ω).filter (fun p => O.label p ≠ sideClass true))
      ((rejectSide O.mq lo F P ω).filter (fun p => O.label p ≠ sideClass false)) := by
    rw [Finset.disjoint_left]
    intro p hp hp'
    have h1 := (Finset.mem_filter.1 (Finset.mem_filter.1 hp).1).2
    have h2 := (Finset.mem_filter.1 (Finset.mem_filter.1 hp').1).2
    omega
  have hsub : (acceptSide O.mq hi F P ω).filter (fun p => O.label p ≠ sideClass true)
      ∪ (rejectSide O.mq lo F P ω).filter (fun p => O.label p ≠ sideClass false)
      ⊆ P.filter (fun p => ¬ cutCorrect O lo hi F p ω) := by
    intro p hp
    rcases Finset.mem_union.1 hp with hp | hp
    · obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
      obtain ⟨hpP, hv⟩ := Finset.mem_filter.1 hpA
      exact Finset.mem_filter.2 ⟨hpP, fun hc => hl (by simpa [sideClass] using hc.1 hv)⟩
    · obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
      obtain ⟨hpP, hv⟩ := Finset.mem_filter.1 hpA
      exact Finset.mem_filter.2 ⟨hpP, fun hc => hl (by simpa [sideClass] using hc.2 hv)⟩
  have := Finset.card_le_card hsub
  rw [Finset.card_union_of_disjoint hdisj] at this
  exact_mod_cast this

omit [IsProbabilityMeasure μ] in
lemma rightRate_le (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (accept : Bool) :
    rightRate O accept ≤ η₀ := by
  have h1 : O.ηIn ≤ η₀ := le_trans (le_max_left _ _) hη
  have h2 : O.ηOut ≤ η₀ := le_trans (le_max_right _ _) hη
  unfold rightRate; split_ifs <;> assumption

open scoped Classical in
/-- A live side's cap the other way: `1/gateKappa` per wrong prefix, and the deviations at
`gateDevT`, `gateDevD` and `gateDevU`.  A class with no prefix on the side has its rate read as
`0`, which changes no count. -/
theorem mul_sideCap_le (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (accept : Bool) (A R : Finset S) (hR : R.Nonempty)
    (hne : ∀ v ∈ R, ∀ p ∈ A, p * v ≠ p) {t tu a : ℝ} {ω : Ω} (ht : 0 ≤ t) (htu : 0 ≤ tu)
    (h : sideLive O accept A R t tu ω) :
    A.card * sideCap O.mq accept A R a η₀ ω
      ≤ (A.filter (fun p => O.label p ≠ sideClass accept)).card / gateKappa η₀
        + gateDevT η₀ * t + gateDevD η₀ * hoeffDev A.card a + gateDevU η₀ * tu := by
  classical
  obtain ⟨hlo, hhi, hb⟩ := live_counts O (by linarith [lt_of_le_of_lt hη hη₀]) accept A R hR hne h
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  set kk := (A.filter (fun p => O.label p ≠ sideClass accept)).card with hkk
  have hkA : kk ≤ A.card := Finset.card_le_card (Finset.filter_subset _ _)
  obtain ⟨x, hx0, hxη, hx1, hx2⟩ : ∃ x', 0 ≤ x' ∧ x' ≤ η₀
      ∧ ((A.card : ℝ) - kk) * x' = ((A.card : ℝ) - kk) * rightRate O accept
      ∧ ((A.card : ℝ) - kk) * (x' * (1 - x'))
        = ((A.card : ℝ) - kk) * (rightRate O accept * (1 - rightRate O accept)) := by
    by_cases hlt : kk < A.card
    · obtain ⟨p, hpA, hp⟩ : ∃ p ∈ A, ¬ (O.label p ≠ sideClass accept) := by
        by_contra hc
        push Not at hc
        have : A.filter (fun p => O.label p ≠ sideClass accept) = A := Finset.filter_true_of_mem hc
        have : kk = A.card := by rw [hkk, this]
        omega
      have hr := O.rate_nonneg p
      rw [rate_of_label O accept p, if_pos (not_not.1 hp)] at hr
      exact ⟨_, hr, rightRate_le O hη accept, rfl, rfl⟩
    · have h0 : ((A.card : ℝ) - kk) = 0 := by
        have : kk = A.card := le_antisymm hkA (not_lt.1 hlt)
        rw [this]; ring
      exact ⟨0, le_rfl, hη0, by rw [h0]; ring, by rw [h0]; ring⟩
  obtain ⟨y, hy0, hyη, hy1, hy2⟩ : ∃ y', 0 ≤ y' ∧ y' ≤ η₀
      ∧ (kk : ℝ) * (1 - y') = kk * (1 - wrongRate O accept)
      ∧ (kk : ℝ) * (y' * (1 - y')) = kk * (wrongRate O accept * (1 - wrongRate O accept)) := by
    by_cases hpos : 0 < kk
    · obtain ⟨p, hp⟩ : (A.filter (fun p => O.label p ≠ sideClass accept)).Nonempty :=
        Finset.card_pos.1 hpos
      have hr := O.rate_nonneg p
      rw [rate_of_label O accept p, if_neg (Finset.mem_filter.1 hp).2] at hr
      exact ⟨_, hr, wrongRate_le O hη accept, rfl, rfl⟩
    · have h0 : (kk : ℝ) = 0 := by
        have : kk = 0 := by omega
        rw [this]; simp
      exact ⟨0, le_rfl, hη0, by rw [h0]; ring, by rw [h0]; ring⟩
  rw [← hx1, ← hy1] at hlo hhi
  have hb' : 2 * (((A.card : ℝ) - kk) * (x * (1 - x)) + kk * (y * (1 - y))) - tu
      ≤ refApart O.mq A R ω := by rw [hx2, hy2]; exact hb
  unfold sideCap
  exact mul_wrongCap_le A.card _ _ _ hkA (Real.sqrt_nonneg _) ht htu hx0 hxη hy0 hyη hη₀
    hlo hhi hb'

open scoped Classical in
/-- A family whose cut is wrong on at most `M` of the sample is certified once the sample's sides
are live and `M/gateKappa` and the deviations fit under the limit. -/
theorem certifiesAt_of_live (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (lo hi : ℕ) (hlohi : lo ≤ hi) (F R C : Finset S)
    (hR : R.Nonempty) (hneC : ∀ v ∈ R, ∀ p ∈ C, p * v ≠ p) {n : ℕ} (hn : 0 < n)
    (hC : C.card = n) {lim a t M : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) (ht : 0 ≤ t)
    {ω : Ω} (hlive : sidesLive O lo hi F C R t t ω)
    (hmis : ((C.filter (fun p => ¬ cutCorrect O lo hi F p ω)).card : ℝ) ≤ M)
    (hgate : M / gateKappa η₀
        + 2 * ((gateDevT η₀ + gateDevU η₀) * t + gateDevD η₀ * hoeffDev n (sizeLevel a n))
        ≤ n * lim - hoeffDev n (sizeLevel a n)) :
    certifiesAt O.mq η₀ lim a lo hi F R C ω := by
  classical
  have h0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hκ := (gateKappa_facts h0 hη₀).1
  obtain ⟨hT, hD, hU⟩ := gateDev_pos h0 hη₀
  set b : ℝ := sizeLevel a n with hbdef
  have hb0 : 0 < b := sizeLevel_pos ha0 n
  have hb1 : b ≤ 1 := sizeLevel_le ha0.le ha1 n
  set A := acceptSide O.mq hi F C ω with hA
  set Rs := rejectSide O.mq lo F C ω with hRs
  have hAC : A.card ≤ n := hC ▸ Finset.card_le_card (Finset.filter_subset _ _)
  have hRC : Rs.card ≤ n := hC ▸ Finset.card_le_card (Finset.filter_subset _ _)
  have hw := sides_wrong_le_miscut O lo hi hlohi F C ω
  rw [← hA, ← hRs] at hw
  push_cast at hw
  have hcapA := mul_sideCap_le O hη hη₀ true A R hR
    (fun v hv p hp => hneC v hv p (Finset.mem_filter.1 hp).1) ht ht hlive.1 (a := b)
  have hcapR := mul_sideCap_le O hη hη₀ false Rs R hR
    (fun v hv p hp => hneC v hv p (Finset.mem_filter.1 hp).1) ht ht hlive.2 (a := b)
  have hdA : hoeffDev A.card b ≤ hoeffDev n b := hoeffDev_mono hAC hb0 hb1
  have hdR : hoeffDev Rs.card b ≤ hoeffDev n b := hoeffDev_mono hRC hb0 hb1
  have hsum : ((A.filter (fun p => O.label p ≠ sideClass true)).card : ℝ) / gateKappa η₀
      + ((Rs.filter (fun p => O.label p ≠ sideClass false)).card : ℝ) / gateKappa η₀
      ≤ M / gateKappa η₀ := by
    rw [← add_div]
    exact div_le_div_of_nonneg_right (by linarith) hκ.le
  have hdevs : gateDevD η₀ * hoeffDev A.card b + gateDevD η₀ * hoeffDev Rs.card b
      ≤ 2 * (gateDevD η₀ * hoeffDev n b) := by
    have := mul_le_mul_of_nonneg_left hdA hD.le
    have := mul_le_mul_of_nonneg_left hdR hD.le
    linarith
  refine ⟨hC ▸ hn, ?_⟩
  unfold miscutBound
  rw [hC, ← hbdef, ← hA, ← hRs]
  linarith

/-- What the gate charges a wrong prefix on side `accept` at the oracle's own rates:
`gap²/(1 − η₀ − x)²`, `x` the rate of the side's right class. -/
noncomputable def sideCharge (O : Oracle μ S) (η₀ : ℝ) (accept : Bool) : ℝ :=
  O.gap ^ 2 / (1 - η₀ - max (rightRate O accept) 0) ^ 2

omit [IsProbabilityMeasure μ] in
lemma sideCharge_nonneg (O : Oracle μ S) (η₀ : ℝ) (accept : Bool) : 0 ≤ sideCharge O η₀ accept :=
  div_nonneg (sq_nonneg _) (sq_nonneg _)

omit [IsProbabilityMeasure μ] in
lemma one_sub_side_rates (O : Oracle μ S) (accept : Bool) :
    1 - max (rightRate O accept) 0 - max (wrongRate O accept) 0 = O.gap := by
  unfold rightRate wrongRate Oracle.gap
  cases accept
  · simp only [Bool.false_eq_true, ↓reduceIte]; ring
  · simp only [↓reduceIte]

omit [IsProbabilityMeasure μ] in
lemma gap_eq_clsRate (O : Oracle μ S) (b : Bool) :
    O.gap = 1 - clsRate O (!b) - clsRate O b := by
  unfold Oracle.gap clsRate rightRate; cases b <;> simp <;> ring

/-- A wrong prefix of class `b` is charged, on the side it lands on, at most `5/(2 sig²)` times
its class's `σ_b` (`charge_le_sig`). -/
lemma sideCharge_le_clsSig (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (b : Bool) : sideCharge O η₀ (!b) ≤ 5 / (2 * sig η₀ ^ 2) * clsSig O η₀ b := by
  have h := charge_le_sig (x := clsRate O (!b)) (y := clsRate O b) (v := voteSlack η₀)
    (clsRate_le O hη _) (clsRate_le O hη _) hη₀ (by rw [voteSlack, sig])
  rw [← gap_eq_clsRate] at h
  unfold sideCharge clsSig sig
  exact h

lemma sideCharge_le (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (accept : Bool) : sideCharge O η₀ accept ≤ 1 / (1 - 2 * η₀) ^ 2 := by
  have hx := clsRate_le O hη accept
  have hg := gap_le_one O
  have hg0 := (gap_pos O (lt_of_le_of_lt hη hη₀)).le
  unfold sideCharge
  rw [show max (rightRate O accept) 0 = clsRate O accept from rfl]
  have hD : 1 - 2 * η₀ ≤ 1 - η₀ - clsRate O accept := by
    have := le_trans (eta_nonneg O) hη; linarith
  have hD0 : 0 < 1 - 2 * η₀ := by linarith
  calc O.gap ^ 2 / (1 - η₀ - clsRate O accept) ^ 2 ≤ 1 / (1 - η₀ - clsRate O accept) ^ 2 :=
        div_le_div_of_nonneg_right (by nlinarith) (sq_nonneg _)
    _ ≤ 1 / (1 - 2 * η₀) ^ 2 :=
        div_le_div_of_nonneg_left zero_le_one (by positivity) (pow_le_pow_left₀ hD0.le hD 2)

/-- A wrong prefix costs the gate at most `1/a²`: its charge is largest with the side's right class
at `η₀`. -/
lemma sideCharge_le_gateA (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (accept : Bool) : sideCharge O η₀ accept ≤ 1 / gateA η₀ ^ 2 := by
  have h0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hx := clsRate_le O hη accept
  have hx0 := clsRate_nonneg O accept
  have hy0 := clsRate_nonneg O (!accept)
  have hg := gap_eq_clsRate O (!accept)
  rw [Bool.not_not] at hg
  have hg0 := (gap_pos O (lt_of_le_of_lt hη hη₀)).le
  have hD : 0 < 1 - η₀ - clsRate O accept := by linarith
  have ha : 0 < gateA η₀ := gateA_pos h0 hη₀
  have key : O.gap * gateA η₀ ≤ 1 - η₀ - clsRate O accept := by
    unfold gateA
    rw [mul_div_assoc', div_le_iff₀ (by linarith)]
    nlinarith [mul_le_mul_of_nonneg_right (show O.gap ≤ 1 - clsRate O accept by linarith)
      (by linarith : (0 : ℝ) ≤ 1 - 2 * η₀)]
  have hsq : (O.gap * gateA η₀) ^ 2 ≤ (1 - η₀ - clsRate O accept) ^ 2 :=
    pow_le_pow_left₀ (mul_nonneg hg0 ha.le) key 2
  unfold sideCharge
  rw [show max (rightRate O accept) 0 = clsRate O accept from rfl,
    div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith [hsq]

open scoped Classical in
/-- `mul_sideCap_le` at the oracle's rates: `4·sideCharge` per wrong prefix, the deviations
twice. -/
theorem mul_sideCap_le_rates (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (accept : Bool) (A R : Finset S) (hR : R.Nonempty)
    (hne : ∀ v ∈ R, ∀ p ∈ A, p * v ≠ p) {t tu a : ℝ} {ω : Ω} (ht : 0 ≤ t) (htu : 0 ≤ tu)
    (h : sideLive O accept A R t tu ω) :
    A.card * sideCap O.mq accept A R a η₀ ω
      ≤ 4 * (A.filter (fun p => O.label p ≠ sideClass accept)).card * sideCharge O η₀ accept
        + 2 * (gateDevT η₀ * t + gateDevD η₀ * hoeffDev A.card a + gateDevU η₀ * tu) := by
  classical
  obtain ⟨hlo, hhi, hb⟩ := live_counts O (by linarith [lt_of_le_of_lt hη hη₀]) accept A R hR hne h
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  set kk := (A.filter (fun p => O.label p ≠ sideClass accept)).card with hkk
  have hkA : kk ≤ A.card := Finset.card_le_card (Finset.filter_subset _ _)
  set x := max (rightRate O accept) 0 with hx
  set y := max (wrongRate O accept) 0 with hy
  have hx0 : 0 ≤ x := le_max_right _ _
  have hy0 : 0 ≤ y := le_max_right _ _
  have hxη : x ≤ η₀ := max_le (rightRate_le O hη accept) hη0
  have hyη : y ≤ η₀ := max_le (wrongRate_le O hη accept) hη0
  have hx1 : ((A.card : ℝ) - kk) * x = ((A.card : ℝ) - kk) * rightRate O accept := by
    by_cases hlt : kk < A.card
    · obtain ⟨p, hpA, hp⟩ : ∃ p ∈ A, ¬ (O.label p ≠ sideClass accept) := by
        by_contra hc
        push Not at hc
        have : A.filter (fun p => O.label p ≠ sideClass accept) = A := Finset.filter_true_of_mem hc
        have : kk = A.card := by rw [hkk, this]
        omega
      have hr := O.rate_nonneg p
      rw [rate_of_label O accept p, if_pos (not_not.1 hp)] at hr
      rw [hx, max_eq_left hr]
    · have : kk = A.card := le_antisymm hkA (not_lt.1 hlt)
      rw [this]; ring
  have hy1 : (kk : ℝ) * (1 - y) = kk * (1 - wrongRate O accept) := by
    by_cases hpos : 0 < kk
    · obtain ⟨p, hp⟩ : (A.filter (fun p => O.label p ≠ sideClass accept)).Nonempty :=
        Finset.card_pos.1 hpos
      have hr := O.rate_nonneg p
      rw [rate_of_label O accept p, if_neg (Finset.mem_filter.1 hp).2] at hr
      rw [hy, max_eq_left hr]
    · have : kk = 0 := by omega
      rw [this]; simp
  have hx2 : ((A.card : ℝ) - kk) * (x * (1 - x))
      = ((A.card : ℝ) - kk) * (rightRate O accept * (1 - rightRate O accept)) := by
    by_cases hlt : kk < A.card
    · have h := hx1
      have hne0 : ((A.card : ℝ) - kk) ≠ 0 := by
        have : (kk : ℝ) < A.card := by exact_mod_cast hlt
        linarith
      have := mul_left_cancel₀ hne0 h
      rw [this]
    · have : kk = A.card := le_antisymm hkA (not_lt.1 hlt)
      rw [this]; ring
  have hy2 : (kk : ℝ) * (y * (1 - y)) = kk * (wrongRate O accept * (1 - wrongRate O accept)) := by
    by_cases hpos : 0 < kk
    · have hne0 : (kk : ℝ) ≠ 0 := by exact_mod_cast hpos.ne'
      have : 1 - y = 1 - wrongRate O accept := mul_left_cancel₀ hne0 hy1
      have : y = wrongRate O accept := by linarith
      rw [this]
    · have : kk = 0 := by omega
      rw [this]; simp
  rw [← hx1, ← hy1] at hlo hhi
  have hb' : 2 * (((A.card : ℝ) - kk) * (x * (1 - x)) + kk * (y * (1 - y))) - tu
      ≤ refApart O.mq A R ω := by rw [hx2, hy2]; exact hb
  unfold sideCap
  have key := mul_wrongCap_le_rates A.card _ _ _ hkA (Real.sqrt_nonneg _) ht htu hx0 hxη hy0 hyη
    hη₀ hlo hhi hb' (d := hoeffDev A.card a)
  have hg : 1 - x - y = O.gap := one_sub_side_rates O accept
  rw [hg] at key
  unfold sideCharge
  rw [← hx]
  linarith

open scoped Classical in
/-- `certifiesAt_of_live` at the oracle's rates.  `M b` bounds the sample's prefixes of class
`b` the cut gets wrong; each is charged `4·sideCharge` on the side it lands on. -/
theorem certifiesAt_of_live_rates (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) (hη₀ : η₀ < 1 / 2)
    (lo hi : ℕ) (F R C : Finset S)
    (hR : R.Nonempty) (hneC : ∀ v ∈ R, ∀ p ∈ C, p * v ≠ p) {n : ℕ} (hn : 0 < n)
    (hC : C.card = n) {lim a t : ℝ} (M : Bool → ℝ) (ha0 : 0 < a) (ha1 : a ≤ 1) (ht : 0 ≤ t)
    {ω : Ω} (hlive : sidesLive O lo hi F C R t t ω)
    (hmis : ∀ b, ((C.filter (fun p => O.label p = sideClass b ∧ ¬ cutCorrect O lo hi F p ω)).card
      : ℝ) ≤ M b)
    (hgate : 4 * (M false * sideCharge O η₀ true + M true * sideCharge O η₀ false)
        + 4 * ((gateDevT η₀ + gateDevU η₀) * t + gateDevD η₀ * hoeffDev n (sizeLevel a n))
        ≤ n * lim - hoeffDev n (sizeLevel a n)) :
    certifiesAt O.mq η₀ lim a lo hi F R C ω := by
  classical
  have h0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  obtain ⟨hT, hD, hU⟩ := gateDev_pos h0 hη₀
  set b : ℝ := sizeLevel a n with hbdef
  have hb0 : 0 < b := sizeLevel_pos ha0 n
  have hb1 : b ≤ 1 := sizeLevel_le ha0.le ha1 n
  set A := acceptSide O.mq hi F C ω with hA
  set Rs := rejectSide O.mq lo F C ω with hRs
  have hAC : A.card ≤ n := hC ▸ Finset.card_le_card (Finset.filter_subset _ _)
  have hRC : Rs.card ≤ n := hC ▸ Finset.card_le_card (Finset.filter_subset _ _)
  -- each side's wrong prefixes are the cut's mistakes on the other class
  have hwA : ((A.filter (fun p => O.label p ≠ sideClass true)).card : ℝ) ≤ M false := by
    refine le_trans ?_ (hmis false)
    refine Nat.cast_le.2 (Finset.card_le_card (fun p hp => ?_))
    obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpC, hv⟩ := Finset.mem_filter.1 hpA
    have hl0 : O.label p = 0 := by
      rcases O.label_bit p with h | h
      · exact h
      · exact absurd h (by simpa [sideClass] using hl)
    exact Finset.mem_filter.2 ⟨hpC, by simpa [sideClass] using hl0,
      fun hc => hl (by simpa [sideClass] using hc.1 hv)⟩
  have hwR : ((Rs.filter (fun p => O.label p ≠ sideClass false)).card : ℝ) ≤ M true := by
    refine le_trans ?_ (hmis true)
    refine Nat.cast_le.2 (Finset.card_le_card (fun p hp => ?_))
    obtain ⟨hpA, hl⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpC, hv⟩ := Finset.mem_filter.1 hpA
    have hl1 : O.label p = 1 := by
      rcases O.label_bit p with h | h
      · exact absurd h (by simpa [sideClass] using hl)
      · exact h
    exact Finset.mem_filter.2 ⟨hpC, by simpa [sideClass] using hl1,
      fun hc => hl (by simpa [sideClass] using hc.2 hv)⟩
  have hcapA := mul_sideCap_le_rates O hη hη₀ true A R hR
    (fun v hv p hp => hneC v hv p (Finset.mem_filter.1 hp).1) ht ht hlive.1 (a := b)
  have hcapR := mul_sideCap_le_rates O hη hη₀ false Rs R hR
    (fun v hv p hp => hneC v hv p (Finset.mem_filter.1 hp).1) ht ht hlive.2 (a := b)
  have hdA : hoeffDev A.card b ≤ hoeffDev n b := hoeffDev_mono hAC hb0 hb1
  have hdR : hoeffDev Rs.card b ≤ hoeffDev n b := hoeffDev_mono hRC hb0 hb1
  have hchA := sideCharge_nonneg O η₀ true
  have hchR := sideCharge_nonneg O η₀ false
  have h1 := mul_le_mul_of_nonneg_right hwA hchA
  have h2 := mul_le_mul_of_nonneg_right hwR hchR
  have h3 := mul_le_mul_of_nonneg_left hdA hD.le
  have h4 := mul_le_mul_of_nonneg_left hdR hD.le
  refine ⟨hC ▸ hn, ?_⟩
  unfold miscutBound
  rw [hC, ← hbdef, ← hA, ← hRs]
  linarith

/-! ### Liveness over the run -/

open scoped Classical in
lemma sidesLive_congr_reads (O : Oracle μ S) (lo hi : ℕ) (F A C R : Finset S) (t tu : ℝ)
    (hone : (1 : S) ∈ C) (hF : F ⊆ C) (hRC : R ⊆ C) {ω ω' : Ω}
    (h : ∀ w ∈ readSet A C, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    sidesLive O lo hi F A R t tu ω → sidesLive O lo hi F A R t tu ω' := by
  classical
  have hAC : ∀ p ∈ A, ∀ u ∈ C, (O.mq (p * u) ω = 1 ↔ O.mq (p * u) ω' = 1) := fun p hp u hu =>
    h _ (mem_readSet hp hu)
  have hA : ∀ p ∈ A, (O.mq p ω = 1 ↔ O.mq p ω' = 1) := fun p hp => by simpa using hAC p hp 1 hone
  have hvc : ∀ p ∈ A, voteCount O.mq F p ω = voteCount O.mq F p ω' := fun p hp =>
    voteCount_congr O F p (fun u hu => hAC p hp u (hF hu))
  have hacc : acceptSide O.mq hi F A ω = acceptSide O.mq hi F A ω' :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  have hrej : rejectSide O.mq lo F A ω = rejectSide O.mq lo F A ω' :=
    Finset.filter_congr (fun p hp => by rw [hvc p hp])
  rintro ⟨h1, h2⟩
  refine ⟨?_, ?_⟩
  · rw [← hacc]
    exact sideLive_congr_mq O true _ R t tu (fun p hp =>
      ⟨hA p (Finset.mem_filter.1 hp).1, fun v hv => hAC p (Finset.mem_filter.1 hp).1 v (hRC hv)⟩) h1
  · rw [← hrej]
    exact sideLive_congr_mq O false _ R t tu (fun p hp =>
      ⟨hA p (Finset.mem_filter.1 hp).1, fun v hv => hAC p (Finset.mem_filter.1 hp).1 v (hRC hv)⟩) h2

open scoped Classical in
/-- At a look of `m` certification draws of population `j`, a side of the sample is not live
against the references. -/
noncomputable def sidesDead (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J)
    (j : J) (B : State) (m : ℕ) (t : ℝ) : Set (Run Ω S J) :=
  {x | ¬ sidesLive O B.lo B.hi (clusterBy rule O.mq populations x B)
    (certOf (prefixesAt populations B.npref x) j m x) (refsBy rule O.mq populations x B) t t
    (oracleNoise x)}

open scoped Classical in
noncomputable def sidesDeadOf (rule : Clusterer S) (O : Oracle μ S) (B : State) (t : ℝ)
    (P C A : Finset S) (ω : Ω) : Prop :=
  ¬ sidesLive O B.lo B.hi (clusterOf rule O B.sc B.scd P C B.k ω) A
    (C \ insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) t t ω

open scoped Classical in
lemma sidesDeadOf_congr (O : Oracle μ S) (B : State) (t : ℝ) (P C A : Finset S)
    (hone : (1 : S) ∈ C) {ω ω' : Ω}
    (h : ∀ w ∈ readSet P C ∪ readSet A C, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    sidesDeadOf rule O B t P C A ω → sidesDeadOf rule O B t P C A ω' := by
  have hfam : clusterOf rule O B.sc B.scd P C B.k ω = clusterOf rule O B.sc B.scd P C B.k ω' :=
    clusterOf_congr_mq O B.sc B.scd P C B.k hone (fun w hw => h w (Finset.mem_union_left _ hw))
  have hsub : clusterOf rule O B.sc B.scd P C B.k ω' ⊆ C :=
    clusterOf_subset O B.sc B.scd P C B.k ω' hone
  unfold sidesDeadOf
  rw [hfam]
  intro hdead hlive
  exact hdead (sidesLive_congr_reads O B.lo B.hi _ A C _ t t hone hsub Finset.sdiff_subset
    (fun w hw => (h w (Finset.mem_union_right _ hw)).symm) hlive)

open scoped Classical in
lemma measurableSet_sidesDead (O : Oracle μ S) (populations : Finset J) (j : J) (B : State)
    (m : ℕ) (t : ℝ) : MeasurableSet (sidesDead rule O populations j B m t) := by
  classical
  have hR : ∀ P C A : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω | sidesDeadOf rule O B t P C A ω} else (∅ : Set (Run Ω S J))) := by
    intro P C A
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_readCongr O _ _
        (fun ω ω' h => sidesDeadOf_congr O B t P C A hone h))
    · exact MeasurableSet.empty
  have hrw : sidesDead rule O populations j B m t
      = {x : Run Ω S J | x ∈ (fun P C A => if (1 : S) ∈ C then
          oracleNoise ⁻¹' {ω | sidesDeadOf rule O B t P C A ω} else ∅)
        (prefixesAt populations B.npref x) (poolAt B.nsuff x)
        (certOf (prefixesAt populations B.npref x) j m x)} := by
    ext x
    simp only [Set.mem_setOf_eq, if_pos (one_mem_poolAt B.nsuff x), Set.mem_preimage]
    rfl
  rw [hrw]
  exact measurableSet_of_run_data_certAt populations j B m _ hR

open scoped Classical in
/-- `sidesLive_whp` over the run, at a deviation `t` for every count. -/
theorem measureReal_sidesDead_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (populations : Finset J) (Pp Pc : J → Measure (ℕ → S)) (Dsf : Measure S)
    [∀ j, IsProbabilityMeasure (Pp j)] [∀ j, IsProbabilityMeasure (Pc j)]
    [IsProbabilityMeasure Dsf]
    (hsuppP : ∀ j ∈ populations, ∀ᵐ y ∂(Pp j), ∀ i, y i ∈ Pre)
    (hsuppC : ∀ j ∈ populations, ∀ᵐ y ∂(Pc j), ∀ i, y i ∈ Pre) (hsuppSf : Dsf Sufᶜ = 0)
    (j : J) (hj : j ∈ populations) (B : State) (m : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    (streamMeasure μ Pp Pc Dsf).real (sidesDead rule O populations j B m t)
      ≤ 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / m)) := by
  classical
  set E : ℝ := 2 * (((B.nsuff : ℝ) + 3) * Real.exp (-2 * t ^ 2 / m)) with hEdef
  have hE0 : 0 ≤ E := by positivity
  have hEnn : streamMeasure μ Pp Pc Dsf (sidesDead rule O populations j B m t)
      ≤ ENNReal.ofReal E := by
    refine streamMeasure_slice_le Pp Pc Dsf _ (measurableSet_sidesDead O populations j B m t) _ ?_
    filter_upwards [ae_draws_mem_Pre Pp Pc Dsf Pre populations hsuppP,
      ae_cert_mem_Pre Pp Pc Dsf Pre populations hsuppC, ae_sfx_mem_Suf Pp Pc Dsf Suf hsuppSf]
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
    have hCdcard : (Cd.card : ℝ) ≤ (B.nsuff : ℝ) + 1 := by
      have := le_trans (Finset.card_insert_le (1 : S) _)
        (Nat.add_le_add_right (card_freshDraws_le ∅ (fun i => d.1.1 i) B.nsuff) 1)
      exact_mod_cast this
    have hdisjPA : Disjoint Ad Pd := disjoint_freshDraws _ _ _
    have hsec : {ω : Ω | ((ω, d) : Run Ω S J) ∈ sidesDead rule O populations j B m t}
        = {ω | sidesDeadOf rule O B t Pd Cd Ad ω} := rfl
    rw [hsec]
    set sel : Ω → Finset S := fun ω => clusterOf rule O B.sc B.scd Pd Cd B.k ω with hseldef
    set Bad : Finset S → Set Ω := fun F =>
      if F ⊆ Cd ∧ (1 : S) ∉ F then {ω | ¬ sidesLive O B.lo B.hi F Ad (Cd \ insert 1 F) t t ω}
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
        · intro hlive
          exact hω (sidesLive_congr_reads O B.lo B.hi F Ad Cd _ t t hone hF.1 Finset.sdiff_subset
            (fun w hw => by rw [mq_congr O (h w hw)]) hlive)
        · exact hω)
      E hE0 (fun F _ => by
        simp only [hBaddef]
        split_ifs with hF
        · have hsub : insert 1 (F ∪ (Cd \ insert 1 F)) ⊆ Cd :=
            Finset.insert_subset hone (Finset.union_subset hF.1 Finset.sdiff_subset)
          refine le_trans (sidesLive_whp O Ad F (Cd \ insert 1 F) B.lo B.hi ?_
            (Finset.disjoint_left.2 (fun v hv hv' =>
              (Finset.mem_sdiff.1 hv').2 (Finset.mem_insert_of_mem hv)))
            hF.2 (fun h => (Finset.mem_sdiff.1 h).2 (Finset.mem_insert_self _ _)) ht ht m
            (card_freshDraws_le _ _ _)) ?_
          · intro p hp p' hp' u hu u' hu' heq
            have hpp := hflat p (hA p hp) p' (hA p' hp') u (hV u (hsub hu)) u' (hV u' (hsub hu'))
              heq
            subst hpp
            exact ⟨rfl, mul_left_cancel heq⟩
          · have hR : ((Cd \ insert 1 F).card : ℝ) ≤ (B.nsuff : ℝ) + 1 :=
              le_trans (by exact_mod_cast Finset.card_le_card Finset.sdiff_subset) hCdcard
            have he := Real.exp_nonneg (-2 * t ^ 2 / m)
            rw [hEdef]
            nlinarith
        · rw [measureReal_empty]; exact hE0)
    have hsub : {ω | sidesDeadOf rule O B t Pd Cd Ad ω} ⊆ {ω | ω ∈ Bad (sel ω)} := by
      intro ω hω
      change ω ∈ Bad (sel ω)
      have hF : sel ω ⊆ Cd ∧ (1 : S) ∉ sel ω :=
        ⟨clusterOf_subset O B.sc B.scd Pd Cd B.k ω hone, Finset.notMem_erase _ _⟩
      simp only [hBaddef, if_pos hF]
      exact hω
    calc μ {ω | sidesDeadOf rule O B t Pd Cd Ad ω} ≤ μ {ω | ω ∈ Bad (sel ω)} := measure_mono hsub
      _ = ENNReal.ofReal (μ.real {ω | ω ∈ Bad (sel ω)}) :=
          (ENNReal.ofReal_toReal (measure_ne_top μ _)).symm
      _ ≤ ENNReal.ofReal E := ENNReal.ofReal_le_ofReal hmain
  rw [measureReal_def]
  calc (streamMeasure μ Pp Pc Dsf (sidesDead rule O populations j B m t)).toReal
      ≤ (ENNReal.ofReal E).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hEnn
    _ = E := ENNReal.toReal_ofReal hE0

end OrthoDFA
