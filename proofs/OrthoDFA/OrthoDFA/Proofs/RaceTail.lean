import OrthoDFA.Utils.RaceTail
import OrthoDFA.Proofs.RaceReads
import Mathlib.Probability.Independence.Integration

/-!
# How many rounds meet a fresh bad read: the proof

Dropping the rounds after the `f`-th fresh-bad one (`cut`) makes every round fixed by what it and
the rounds before it read (`Loc`). Then, with `c` the fresh-bad rounds before round `j`, `q` the
new strings and `v` those read in `V`,

    E (1 + t)^c e^{-t'(ε q + κ v)}

does not grow with `j`: a round is fresh-bad with chance at most what its new strings cost, since
that a string is new in round `j`, with the counts so far, does not change with its own read
(`newQ_congr`), and `e^{-t' w} + t w ≤ 1` for a round costing `w ≤ w̄`.
-/

namespace OrthoDFA

namespace RaceTail

open MeasureTheory ProbabilityTheory AdaptiveReads

variable {Z R : Type*} (Q : ℕ → (Z → R) → Finset Z) (B V : Z → Set R)

/-- Each round is fixed by what it and the rounds before it read. -/
def Loc : Prop := ∀ j r r', (∀ i ≤ j, ∀ z ∈ Q i r, r z = r' z) → Q j r = Q j r'

open scoped Classical in
/-- The new strings of round `j` read in `V`. -/
noncomputable def newV (j : ℕ) (r : Z → R) : Finset Z := (newQ Q j r).filter fun z => r z ∈ V z

/-- The fresh-bad rounds, the new strings and the new strings read in `V`, before round `j`. -/
noncomputable def hist (j : ℕ) (r : Z → R) : ℕ × ℕ × ℕ :=
  (badCount Q B j r, ∑ i ∈ Finset.range j, (newQ Q i r).card,
    ∑ i ∈ Finset.range j, (newV Q V i r).card)

open scoped Classical in
theorem badCount_succ (j : ℕ) (r : Z → R) :
    badCount Q B (j + 1) r = badCount Q B j r + if FreshBad Q B j r then 1 else 0 := by
  unfold badCount
  rw [Finset.range_add_one, Finset.filter_insert]
  split_ifs with h
  · rw [Finset.card_insert_of_notMem (by simp)]
  · rfl

theorem badCount_mono (r : Z → R) {i j : ℕ} (h : i ≤ j) : badCount Q B i r ≤ badCount Q B j r := by
  classical
  unfold badCount
  exact Finset.card_le_card (Finset.filter_subset_filter _ (Finset.range_subset_range.2 h))

open scoped Classical in
theorem hist_succ (j : ℕ) (r : Z → R) :
    hist Q B V (j + 1) r = (badCount Q B j r + (if FreshBad Q B j r then 1 else 0),
      (hist Q B V j r).2.1 + (newQ Q j r).card, (hist Q B V j r).2.2 + (newV Q V j r).card) := by
  simp only [hist, badCount_succ, Finset.sum_range_succ]

theorem hist_zero (r : Z → R) : hist Q B V 0 r = (0, 0, 0) := by
  simp [hist, badCount]

/-! ## Agreement on the rounds before `j` -/

section Congr

variable {Q} (hloc : Loc Q)
include hloc

theorem Q_lt_congr {j : ℕ} {r r' : Z → R} (h : ∀ y ∈ before Q j r, r y = r' y) :
    ∀ i < j, Q i r = Q i r' :=
  fun i hi => hloc i r r' fun l hl y hy => h y ((mem_before Q).2 ⟨l, by omega, hy⟩)

theorem newQ_lt_congr {j : ℕ} {r r' : Z → R} (h : ∀ y ∈ before Q j r, r y = r' y) :
    ∀ i < j, newQ Q i r = newQ Q i r' := by
  classical
  intro i hi
  have hQ := Q_lt_congr hloc h
  unfold newQ
  rw [hQ i hi]
  refine Finset.filter_congr fun z _ => ?_
  exact forall₂_congr fun l hl => by rw [hQ l (by omega)]

theorem freshBad_lt_congr {j : ℕ} {r r' : Z → R} (h : ∀ y ∈ before Q j r, r y = r' y) :
    ∀ i < j, (FreshBad Q B i r ↔ FreshBad Q B i r') := by
  classical
  intro i hi
  unfold FreshBad
  rw [← newQ_lt_congr hloc h i hi]
  exact exists_congr fun z => and_congr_right fun hz => by
    rw [h z ((mem_before Q).2 ⟨i, hi, (Finset.mem_filter.1 hz).1⟩)]

theorem newV_lt_congr {j : ℕ} {r r' : Z → R} (h : ∀ y ∈ before Q j r, r y = r' y) :
    ∀ i < j, newV Q V i r = newV Q V i r' := by
  classical
  intro i hi
  unfold newV
  rw [← newQ_lt_congr hloc h i hi]
  refine Finset.filter_congr fun z hz => ?_
  rw [h z ((mem_before Q).2 ⟨i, hi, (Finset.mem_filter.1 hz).1⟩)]

theorem badCount_congr {j : ℕ} {r r' : Z → R} (h : ∀ y ∈ before Q j r, r y = r' y) :
    badCount Q B j r = badCount Q B j r' := by
  classical
  unfold badCount
  congr 1
  exact Finset.filter_congr fun i hi =>
    freshBad_lt_congr (B := B) hloc h i (Finset.mem_range.1 hi)

theorem hist_congr {j : ℕ} {r r' : Z → R} (h : ∀ y ∈ before Q j r, r y = r' y) :
    hist Q B V j r = hist Q B V j r' := by
  simp only [hist, badCount_congr (B := B) hloc h]
  congr 2
  · exact Finset.sum_congr rfl fun i hi => by
      rw [newQ_lt_congr hloc h i (Finset.mem_range.1 hi)]
  · exact Finset.sum_congr rfl fun i hi => by
      rw [newV_lt_congr (V := V) hloc h i (Finset.mem_range.1 hi)]

omit hloc in
theorem mem_before_of_lt {j i : ℕ} (hi : i < j) {r : Z → R} {y : Z} (hy : y ∈ Q i r) :
    y ∈ before Q j r := (mem_before Q).2 ⟨i, hi, hy⟩

open scoped Classical in
/-- Reads agreeing with `r` on what rounds `0, …, j` query but `z` keep `z` new in round `j`. -/
theorem newQ_congr {j : ℕ} {z : Z} {r r' : Z → R} (hz : z ∈ newQ Q j r)
    (h : ∀ y ∈ before Q j r ∪ (Q j r).erase z, r y = r' y) : z ∈ newQ Q j r' := by
  classical
  obtain ⟨hzj, hzi⟩ := Finset.mem_filter.1 hz
  have hb : ∀ y ∈ before Q j r, r y = r' y := fun y hy => h y (Finset.mem_union_left _ hy)
  have hQ := Q_lt_congr hloc hb
  have hzb : z ∉ before Q j r := fun hzb => by
    obtain ⟨i, hi, hzi'⟩ := (mem_before Q).1 hzb
    exact hzi i hi hzi'
  set r'' := Function.update r' z (r z)
  have hQj : Q j r = Q j r'' := hloc j r r'' fun i hi y hy => by
    by_cases hyz : y = z
    · subst hyz; simp [r'']
    · rw [show r'' y = r' y from Function.update_of_ne hyz _ _]
      rcases Nat.lt_or_eq_of_le hi with hi | rfl
      · exact hb y (mem_before_of_lt hi hy)
      · exact h y (Finset.mem_union_right _ (Finset.mem_erase.2 ⟨hyz, hy⟩))
  have hzj' : z ∈ Q j r' := by
    by_contra hz'
    have := hloc j r' r'' fun i hi y hy => by
      have hyz : y ≠ z := by
        rintro rfl
        rcases Nat.lt_or_eq_of_le hi with hi | rfl
        · rw [← hQ i hi] at hy; exact hzi i hi hy
        · exact hz' hy
      simp [r'', Function.update_of_ne hyz]
    exact hz' (this ▸ hQj ▸ hzj)
  refine Finset.mem_filter.2 ⟨hzj', fun i hi hzi' => hzi i hi ?_⟩
  rw [hQ i hi]; exact hzi'

end Congr

/-! ## Dropping the rounds after the `f`-th fresh-bad one -/

section Cut

variable (f : ℕ)

open scoped Classical in
/-- The rounds that fewer than `f` fresh-bad rounds precede. -/
noncomputable def cut (j : ℕ) (r : Z → R) : Finset Z :=
  if badCount Q B j r < f then Q j r else ∅

variable {Q B}

theorem cut_eq {j : ℕ} {r : Z → R} (h : badCount Q B j r < f) : cut Q B f j r = Q j r := by
  simp [cut, h]

theorem cut_eq_empty {j : ℕ} {r : Z → R} (h : ¬ badCount Q B j r < f) : cut Q B f j r = ∅ := by
  simp [cut, h]

theorem newQ_cut {j : ℕ} {r : Z → R} (h : badCount Q B j r < f) :
    newQ (cut Q B f) j r = newQ Q j r := by
  classical
  unfold newQ
  rw [cut_eq f h]
  refine Finset.filter_congr fun z _ => forall₂_congr fun i hi => ?_
  rw [cut_eq f (lt_of_le_of_lt (badCount_mono Q B r hi.le) h)]

theorem newQ_cut_empty {j : ℕ} {r : Z → R} (h : ¬ badCount Q B j r < f) :
    newQ (cut Q B f) j r = ∅ := by
  classical
  unfold newQ
  rw [cut_eq_empty f h]
  rfl

theorem badCount_cut : ∀ (j : ℕ) (r : Z → R),
    badCount (cut Q B f) B j r = min (badCount Q B j r) f
  | 0, r => by simp [badCount]
  | j + 1, r => by
    classical
    rw [badCount_succ, badCount_succ, badCount_cut j r]
    by_cases h : badCount Q B j r < f
    · have hfb : FreshBad (cut Q B f) B j r ↔ FreshBad Q B j r := by
        unfold FreshBad; rw [newQ_cut f h]
      by_cases hb : FreshBad Q B j r
      · simp only [hfb, hb, if_true]; omega
      · simp only [hfb, hb, if_false]; omega
    · have hfb : ¬ FreshBad (cut Q B f) B j r := by
        unfold FreshBad; rw [newQ_cut_empty f h]; simp
      simp only [hfb, if_false]
      split_ifs <;> omega

theorem loc_cut
    (hloc : ∀ j r r', badCount Q B j r < f → (∀ i ≤ j, ∀ z ∈ Q i r, r z = r' z) →
      Q j r = Q j r') : Loc (cut Q B f) := by
  classical
  -- the count before `j` agrees while it is below `f`, and so does reaching `f`
  have key : ∀ j (r r' : Z → R), (∀ i < j, ∀ z ∈ cut Q B f i r, r z = r' z) →
      (badCount Q B j r < f ↔ badCount Q B j r' < f)
        ∧ (badCount Q B j r < f → ∀ i < j, Q i r = Q i r') := by
    intro j
    induction j with
    | zero => intro r r' _; simp [badCount]
    | succ j ih =>
      intro r r' h
      obtain ⟨h1, h2⟩ := ih r r' fun i hi => h i (by omega)
      by_cases hj : badCount Q B j r < f
      · have hj' : badCount Q B j r' < f := h1.1 hj
        have hQj : Q j r = Q j r' := hloc j r r' hj fun i hi y hy =>
          h i (by omega) y (by rw [cut_eq f (lt_of_le_of_lt (badCount_mono Q B r hi) hj)]; exact hy)
        have hall : ∀ i < j + 1, Q i r = Q i r' := fun i hi => by
          rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.1 hi) with hi | rfl
          · exact h2 hj i hi
          · exact hQj
        have hb : ∀ y ∈ before Q (j + 1) r, r y = r' y := fun y hy => by
          obtain ⟨i, hi, hyi⟩ := (mem_before Q).1 hy
          refine h i hi y ?_
          rw [cut_eq f (lt_of_le_of_lt (badCount_mono Q B r (by omega)) hj)]
          exact hyi
        have hcount : badCount Q B (j + 1) r = badCount Q B (j + 1) r' := by
          unfold badCount
          congr 1
          refine Finset.filter_congr fun i hi => ?_
          have hi' := Finset.mem_range.1 hi
          unfold FreshBad
          have hn : newQ Q i r = newQ Q i r' := by
            unfold newQ
            rw [hall i hi']
            exact Finset.filter_congr fun z _ => forall₂_congr fun l hl => by
              rw [hall l (by omega)]
          rw [← hn]
          exact exists_congr fun z => and_congr_right fun hz => by
            rw [hb z ((mem_before Q).2 ⟨i, hi', (Finset.mem_filter.1 hz).1⟩)]
        exact ⟨by rw [hcount], fun _ => hall⟩
      · have hj' : ¬ badCount Q B j r' < f := fun h' => hj (h1.2 h')
        refine ⟨⟨fun h' => absurd (lt_of_le_of_lt (badCount_mono Q B r (by omega)) h') hj,
          fun h' => absurd (lt_of_le_of_lt (badCount_mono Q B r' (by omega)) h') hj'⟩,
          fun h' => absurd (lt_of_le_of_lt (badCount_mono Q B r (by omega)) h') hj⟩
  intro j r r' h
  obtain ⟨h1, h2⟩ := key j r r' fun i hi => h i hi.le
  by_cases hj : badCount Q B j r < f
  · rw [cut_eq f hj, cut_eq f (h1.1 hj)]
    refine hloc j r r' hj fun i hi y hy => ?_
    refine h i hi y ?_
    rw [cut_eq f (lt_of_le_of_lt (badCount_mono Q B r hi) hj)]
    exact hy
  · rw [cut_eq_empty f hj, cut_eq_empty f fun h' => hj (h1.2 h')]

end Cut


/-! ## Measurability and independence -/

section Meas

variable [Countable Z] {Ω : Type*} [MeasurableSpace R] [Countable R] [MeasurableSingletonClass R]

omit [MeasurableSingletonClass R] in
/-- A property fixed by the reads of a finite set of strings in `Y` is measurable in those reads. -/
theorem cyl_measurableSet [MeasurableSingletonClass R] (read : Z → Ω → R) (Y : Set Z)
    (S : (Z → R) → Finset Z) (P : (Z → R) → Prop) (hY : ∀ r, P r → ∀ y ∈ S r, y ∈ Y)
    (hc : ∀ r r', P r → (∀ y ∈ S r, r y = r' y) → P r') :
    MeasurableSet[⨆ y ∈ Y, MeasurableSpace.comap (read y) inferInstance] {ω | P (read · ω)} := by
  classical
  set T : (S' : Finset Z) → Set (S' → R) := fun S' =>
    {a | ∃ r, (∀ y : S', r y = a y) ∧ S r = S' ∧ P r}
  have hset : {ω | P (read · ω)}
      = ⋃ (S' : Finset Z) (_ : ∀ y ∈ S', y ∈ Y), (fun ω (y : S') => read y ω) ⁻¹' T S' := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_preimage, exists_prop]
    constructor
    · intro h
      exact ⟨S (read · ω), hY _ h, (read · ω), fun _ => rfl, rfl, h⟩
    · rintro ⟨S', -, r, hr, hS, hP⟩
      exact hc r (read · ω) hP fun y hy => hr ⟨y, hS ▸ hy⟩
  rw [hset]
  refine MeasurableSet.iUnion fun S' => MeasurableSet.iUnion fun hS' => ?_
  have hm : Measurable[⨆ y ∈ Y, MeasurableSpace.comap (read y) inferInstance]
      fun ω (y : S') => read y ω := by
    let : MeasurableSpace Ω := ⨆ y ∈ Y, MeasurableSpace.comap (read y) inferInstance
    refine measurable_pi_iff.2 fun y => ?_
    refine Measurable.of_comap_le ?_
    exact le_iSup₂ (f := fun y (_ : y ∈ Y) => MeasurableSpace.comap (read y) inferInstance)
      y.1 (hS' y.1 y.2)
  exact hm ((Set.to_countable _).measurableSet)

variable {Q B V} (hloc : Loc Q)
include hloc

theorem hist_measurable [MeasurableSpace Ω] (read : Z → Ω → R) (hmeas : ∀ z, Measurable (read z))
    (j : ℕ) : Measurable fun ω => hist Q B V j (read · ω) := by
  refine measurable_to_countable' fun c => ?_
  have h := cyl_measurableSet read Set.univ (before Q j) (fun r => hist Q B V j r = c)
    (fun _ _ _ _ => trivial) fun r r' hr ha => (hist_congr (B := B) (V := V) hloc ha).symm.trans hr
  refine (iSup₂_le fun y _ => (hmeas y).comap_le : (⨆ y ∈ (Set.univ : Set Z),
    MeasurableSpace.comap (read y) inferInstance) ≤ _) _ h

open scoped Classical in
theorem newHist_measurableSet (read : Z → Ω → R) (j : ℕ) (z : Z) (c : ℕ × ℕ × ℕ) :
    MeasurableSet[others read z]
      {ω | z ∈ newQ Q j (read · ω) ∧ hist Q B V j (read · ω) = c} := by
  refine cyl_measurableSet read ({z}ᶜ : Set Z) (fun r => before Q j r ∪ (Q j r).erase z)
    (fun r => z ∈ newQ Q j r ∧ hist Q B V j r = c) (fun r hr y hy => ?_)
    fun r r' hr ha => ⟨newQ_congr hloc hr.1 ha,
      (hist_congr (B := B) (V := V) hloc fun y hy => ha y (Finset.mem_union_left _ hy)).symm.trans
        hr.2⟩
  simp only [Set.mem_compl_iff, Set.mem_singleton_iff]
  rintro rfl
  rcases Finset.mem_union.1 hy with hy | hy
  · obtain ⟨i, hi, hyi⟩ := (mem_before Q).1 hy
    exact (Finset.mem_filter.1 hr.1).2 i hi hyi
  · simp at hy

end Meas

section Indep

variable [Countable Z] {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
  [MeasurableSingletonClass R] (μ : Measure Ω) (read : Z → Ω → R)

open scoped ENNReal

omit [Countable Z] in
theorem lintegral_mul_read (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ) (z : Z)
    {F : Ω → ℝ≥0∞} (hF : Measurable[others read z] F) (A : Set R) :
    ∫⁻ ω, F ω * (read z ⁻¹' A).indicator 1 ω ∂μ = (∫⁻ ω, F ω ∂μ) * μ (read z ⁻¹' A) := by
  have hle : ∀ y,
      MeasurableSpace.comap (read y) inferInstance ≤ (inferInstance : MeasurableSpace Ω) :=
    fun y => (hmeas y).comap_le
  have hind' := (iIndepFun_iff_iIndep _ read μ).1 hind
  have hI := indep_iSup_of_disjoint hle hind' (S := {z}ᶜ) (T := {z})
    (Set.disjoint_compl_left_iff_subset.2 subset_rfl)
  have hI' : Indep (others read z) (MeasurableSpace.comap (read z) inferInstance) μ :=
    indep_of_indep_of_le_right hI (le_iSup₂ (f := fun y (_ : y ∈ ({z} : Set Z)) =>
      MeasurableSpace.comap (read y) inferInstance) z rfl)
  have hA : MeasurableSet (read z ⁻¹' A) := hmeas z ((Set.to_countable _).measurableSet)
  rw [lintegral_mul_eq_lintegral_mul_lintegral_of_independent_measurableSpace
    (iSup₂_le fun y _ => hle y) (hle z) hI' hF ?_, lintegral_indicator_one hA]
  let : MeasurableSpace Ω := MeasurableSpace.comap (read z) inferInstance
  exact measurable_one.indicator ⟨A, (Set.to_countable _).measurableSet, rfl⟩

end Indep


/-! ## One round -/

theorem tsum_indicator_measurable {ι Ω : Type*} [Countable ι] (m : MeasurableSpace Ω)
    (E : ι → Set Ω) (hE : ∀ c, MeasurableSet[m] (E c)) (h : ι → ENNReal) :
    Measurable[m] fun ω => ∑' c, h c * (E c).indicator 1 ω :=
  Measurable.ennreal_tsum fun c => measurable_const.mul (measurable_one.indicator (hE c))

section Main

open scoped ENNReal

/-- The weight `(1 + t)^c e^{-t'(ε q + κ v)}` of counts `(c, q, v)`. -/
noncomputable def hv (ε κ t t' : ℝ) (c : ℕ × ℕ × ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal ((1 + t) ^ c.1 * Real.exp (-(t' * (ε * c.2.1 + κ * c.2.2))))

open scoped Classical in
theorem tsum_ite_mem (A : Finset Z) (a : ℝ≥0∞) :
    ∑' z, (if z ∈ A then a else 0) = A.card * a := by
  rw [tsum_eq_sum (s := A) fun z hz => if_neg hz, Finset.sum_ite_of_true fun _ h => h,
    Finset.sum_const, nsmul_eq_mul]

/-- `t x ≤ 1 - e^{-t' x}` on `[0, w]` once it holds at `w`. -/
theorem tilt_le {t t' w x : ℝ} (hx0 : 0 ≤ x) (hxw : x ≤ w)
    (htw : t * w ≤ 1 - Real.exp (-(t' * w))) : t * x ≤ 1 - Real.exp (-(t' * x)) := by
  rcases eq_or_lt_of_le (hx0.trans hxw) with hw | hw
  · have : x = 0 := by linarith
    subst this; simp
  · have hb0 : 0 ≤ x / w := div_nonneg hx0 hw.le
    have hb1 : x / w ≤ 1 := (div_le_one hw).2 hxw
    have hc := convexOn_exp.2 (Set.mem_univ (0 : ℝ)) (Set.mem_univ (-(t' * w)))
      (sub_nonneg.2 hb1) hb0 (by ring)
    simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at hc
    have he : x / w * -(t' * w) = -(t' * x) := by field_simp
    rw [he] at hc
    have : t * x = x / w * (t * w) := by field_simp
    rw [this]
    nlinarith [mul_le_mul_of_nonneg_left htw hb0]

variable [Countable Z] {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
  [MeasurableSingletonClass R] (μ : Measure Ω) [IsProbabilityMeasure μ] (read : Z → Ω → R)
  {Q B V}

open scoped Classical in
/-- A round's weight does not grow the mean. -/
theorem step_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ) (hloc : Loc Q)
    {ε κ w t t' : ℝ} (hε : 0 ≤ ε) (hκ : 0 ≤ κ)
    (hB : ∀ z, μ.real {ω | read z ω ∈ B z} ≤ ε + κ * μ.real {ω | read z ω ∈ V z})
    (hw : ∀ r j, ε * (newQ Q j r).card + κ * (newV Q V j r).card ≤ w) (ht : 0 ≤ t)
    (ht' : 0 ≤ t') (htw : t * w ≤ 1 - Real.exp (-(t' * w))) (j : ℕ) :
    ∫⁻ ω, hv ε κ t t' (hist Q B V (j + 1) (read · ω)) ∂μ
      ≤ ∫⁻ ω, hv ε κ t t' (hist Q B V j (read · ω)) ∂μ := by
  have : Countable (ℕ × ℕ × ℕ) := inferInstance
  set H : Ω → ℕ × ℕ × ℕ := fun ω => hist Q B V j (read · ω)
  set H₁ : Ω → ℕ × ℕ × ℕ := fun ω => hist Q B V (j + 1) (read · ω)
  have hHm : Measurable H := hist_measurable hloc read hmeas j
  have hH₁m : Measurable H₁ := hist_measurable hloc read hmeas (j + 1)
  set P : ℕ × ℕ × ℕ → ℝ := fun c => (1 + t) ^ c.1 * Real.exp (-(t' * (ε * c.2.1 + κ * c.2.2)))
  have hP0 : ∀ c, 0 ≤ P c := fun c => by positivity
  set wr : Ω → ℝ := fun ω =>
    ε * (newQ Q j (read · ω)).card + κ * (newV Q V j (read · ω)).card
  have hwr0 : ∀ ω, 0 ≤ wr ω := fun ω => by positivity
  -- the next weight, the round left out
  set g : (ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ) → ℝ≥0∞ := fun p => ENNReal.ofReal (P p.1
    * Real.exp (-(t' * (ε * ((p.2.2.1 - p.1.2.1 : ℕ) : ℝ) + κ * ((p.2.2.2 - p.1.2.2 : ℕ) : ℝ)))))
  set a : Ω → ℝ≥0∞ := fun ω => g (H ω, H₁ ω)
  have ham : Measurable a := (measurable_of_countable g).comp (hHm.prodMk hH₁m)
  have hsucc := fun ω => hist_succ Q B V j (read · ω)
  have ha : ∀ ω, a ω = ENNReal.ofReal (P (H ω) * Real.exp (-(t' * wr ω))) := by
    intro ω
    simp only [a, g, H₁, H, hsucc ω, wr, Nat.add_sub_cancel_left]
  set Sfb : Set Ω := {ω | (H₁ ω).1 = (H ω).1 + 1}
  have hSm : MeasurableSet Sfb :=
    (hHm.prodMk hH₁m) ((Set.to_countable {p : (ℕ × ℕ × ℕ) × (ℕ × ℕ × ℕ) |
      p.2.1 = p.1.1 + 1}).measurableSet)
  have hSfb : ∀ ω, ω ∈ Sfb ↔ FreshBad Q B j (read · ω) := by
    intro ω
    simp only [Sfb, Set.mem_ofPred_eq, H₁, H]
    rw [hsucc ω, show (hist Q B V j (read · ω)).1 = badCount Q B j (read · ω) from rfl]
    by_cases h : FreshBad Q B j (read · ω) <;> simp [h]
  -- a fresh-bad round costs its new strings
  set F : Z → Ω → ℝ≥0∞ := fun z ω =>
    hv ε κ t t' (H ω) * if z ∈ newQ Q j (read · ω) then 1 else 0
  have hFm : ∀ z, Measurable[others read z] (F z) := by
    intro z
    have hFeq : F z = fun ω => ∑' c, hv ε κ t t' c *
        {ω | z ∈ newQ Q j (read · ω) ∧ hist Q B V j (read · ω) = c}.indicator 1 ω := by
      funext ω
      rw [tsum_eq_single (H ω) fun c hc => by
        rw [Set.indicator_of_notMem (fun h => hc h.2.symm), mul_zero]]
      by_cases hz : z ∈ newQ Q j (read · ω)
      · rw [Set.indicator_of_mem (show ω ∈ {ω' | z ∈ newQ Q j (read · ω')
          ∧ hist Q B V j (read · ω') = H ω} from ⟨hz, rfl⟩)]
        simp [F, hz]
      · rw [Set.indicator_of_notMem (fun h : ω ∈ {ω' | z ∈ newQ Q j (read · ω')
          ∧ hist Q B V j (read · ω') = H ω} => hz h.1)]
        simp [F, hz]
    rw [hFeq]
    exact tsum_indicator_measurable (others read z) _
      (fun c => newHist_measurableSet hloc read j z c) _
  have hoth : ∀ z, others read z ≤ (inferInstance : MeasurableSpace Ω) :=
    fun z => iSup₂_le fun y _ => (hmeas y).comap_le
  have hFm' : ∀ z, Measurable (F z) := fun z => (hFm z).mono (hoth z) le_rfl
  have hBm : ∀ z (A : Set R), MeasurableSet (read z ⁻¹' A) := fun z A =>
    hmeas z ((Set.to_countable _).measurableSet)
  have hBad : ∀ z, μ (read z ⁻¹' B z)
      ≤ ENNReal.ofReal ε + ENNReal.ofReal κ * μ (read z ⁻¹' V z) := fun z => by
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ (read z ⁻¹' B z)),
      ← ENNReal.ofReal_toReal (measure_ne_top μ (read z ⁻¹' V z)), ← ENNReal.ofReal_mul hκ,
      ← ENNReal.ofReal_add hε (by positivity)]
    exact ENNReal.ofReal_le_ofReal (hB z)
  have hkey : ∫⁻ ω, hv ε κ t t' (H ω) * Sfb.indicator 1 ω ∂μ
      ≤ ∫⁻ ω, hv ε κ t t' (H ω) * ENNReal.ofReal (wr ω) ∂μ := by
    calc _ ≤ ∫⁻ ω, ∑' z, F z ω * (read z ⁻¹' B z).indicator 1 ω ∂μ := by
          refine lintegral_mono fun ω => ?_
          by_cases hω : ω ∈ Sfb
          · obtain ⟨z, hz, hzB⟩ := (hSfb ω).1 hω
            refine le_trans (le_of_eq ?_) (ENNReal.le_tsum z)
            simp [F, hz, Set.indicator_of_mem hω, Set.indicator_of_mem (show ω ∈ read z ⁻¹' B z
              from hzB)]
          · rw [Set.indicator_of_notMem hω, mul_zero]; exact bot_le
      _ = ∑' z, ∫⁻ ω, F z ω * (read z ⁻¹' B z).indicator 1 ω ∂μ :=
          lintegral_tsum fun z => ((hFm' z).mul (measurable_one.indicator (hBm z _))).aemeasurable
      _ = ∑' z, (∫⁻ ω, F z ω ∂μ) * μ (read z ⁻¹' B z) := by
          congr 1; funext z; exact lintegral_mul_read μ read hmeas hind z (hFm z) _
      _ ≤ ∑' z, (ENNReal.ofReal ε * ∫⁻ ω, F z ω ∂μ
            + ENNReal.ofReal κ * ∫⁻ ω, F z ω * (read z ⁻¹' V z).indicator 1 ω ∂μ) := by
          refine ENNReal.tsum_le_tsum fun z => ?_
          rw [lintegral_mul_read μ read hmeas hind z (hFm z)]
          calc _ ≤ (∫⁻ ω, F z ω ∂μ) * (ENNReal.ofReal ε + ENNReal.ofReal κ * μ (read z ⁻¹' V z)) := by
                gcongr; exact hBad z
            _ = _ := by ring
      _ = ∫⁻ ω, ∑' z, (ENNReal.ofReal ε * F z ω
            + ENNReal.ofReal κ * (F z ω * (read z ⁻¹' V z).indicator 1 ω)) ∂μ := by
          rw [lintegral_tsum (f := fun z ω => ENNReal.ofReal ε * F z ω
            + ENNReal.ofReal κ * (F z ω * (read z ⁻¹' V z).indicator 1 ω)) fun z =>
            (((hFm' z).const_mul _).add (((hFm' z).mul
              (measurable_one.indicator (hBm z _))).const_mul _)).aemeasurable]
          congr 1; funext z
          rw [lintegral_add_left ((hFm' z).const_mul _),
            lintegral_const_mul' _ (fun ω => F z ω) ENNReal.ofReal_ne_top,
            lintegral_const_mul' _ (fun ω => F z ω * (read z ⁻¹' V z).indicator 1 ω)
              ENNReal.ofReal_ne_top]
      _ = _ := by
          congr 1; funext ω
          have hsum : ∀ z, ENNReal.ofReal ε * F z ω
              + ENNReal.ofReal κ * (F z ω * (read z ⁻¹' V z).indicator 1 ω)
              = hv ε κ t t' (H ω) * ((if z ∈ newQ Q j (read · ω) then ENNReal.ofReal ε else 0)
                + if z ∈ newV Q V j (read · ω) then ENNReal.ofReal κ else 0) := by
            intro z
            by_cases hz : z ∈ newQ Q j (read · ω) <;> by_cases hv' : read z ω ∈ V z
            · have : z ∈ newV Q V j (read · ω) := Finset.mem_filter.2 ⟨hz, hv'⟩
              simp [F, hz, this, Set.indicator_of_mem (show ω ∈ read z ⁻¹' V z from hv')]
              ring
            · have : z ∉ newV Q V j (read · ω) := fun h => hv' (Finset.mem_filter.1 h).2
              simp [F, hz, this, Set.indicator_of_notMem (show ω ∉ read z ⁻¹' V z from hv')]
              ring
            · have : z ∉ newV Q V j (read · ω) := fun h => hz (Finset.mem_filter.1 h).1
              simp [F, hz, this]
            · have : z ∉ newV Q V j (read · ω) := fun h => hz (Finset.mem_filter.1 h).1
              simp [F, hz, this]
          simp only [hsum]
          rw [ENNReal.tsum_mul_left, ENNReal.tsum_add, tsum_ite_mem, tsum_ite_mem]
          congr 1
          simp only [wr]
          rw [ENNReal.ofReal_add (by positivity) (by positivity), ENNReal.ofReal_mul hε,
            ENNReal.ofReal_mul hκ, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
          ring
  -- assemble
  have hpt : ∀ ω, hv ε κ t t' (H₁ ω)
      ≤ a ω + ENNReal.ofReal t * (hv ε κ t t' (H ω) * Sfb.indicator 1 ω) := by
    intro ω
    have hexp : Real.exp (-(t' * wr ω)) ≤ 1 := by
      rw [Real.exp_le_one_iff]; nlinarith [hwr0 ω]
    have hH₁ : hv ε κ t t' (H₁ ω) = ENNReal.ofReal (P (H ω)
        * (if FreshBad Q B j (read · ω) then 1 + t else 1) * Real.exp (-(t' * wr ω))) := by
      have e : ∀ q n v m : ℝ, Real.exp (-(t' * (ε * (q + n) + κ * (v + m))))
          = Real.exp (-(t' * (ε * q + κ * v))) * Real.exp (-(t' * (ε * n + κ * m))) := by
        intro q n v m; rw [← Real.exp_add]; ring_nf
      simp only [hv, H₁, H, P, wr]
      rw [hsucc ω]
      simp only [Nat.cast_add, e,
        show (hist Q B V j (read · ω)).1 = badCount Q B j (read · ω) from rfl]
      congr 1
      split_ifs <;> ring
    rw [hH₁, ha]
    by_cases hω : FreshBad Q B j (read · ω)
    · rw [if_pos hω, Set.indicator_of_mem ((hSfb ω).2 hω), Pi.one_apply, mul_one]
      simp only [hv]
      rw [← ENNReal.ofReal_mul ht, ← ENNReal.ofReal_add (by positivity) (by positivity)]
      refine ENNReal.ofReal_le_ofReal ?_
      have := hP0 (H ω)
      nlinarith [mul_le_mul_of_nonneg_left hexp this]
    · rw [if_neg hω, mul_one]
      exact le_self_add
  calc _ ≤ ∫⁻ ω, (a ω + ENNReal.ofReal t * (hv ε κ t t' (H ω) * Sfb.indicator 1 ω)) ∂μ :=
        lintegral_mono hpt
    _ = ∫⁻ ω, a ω ∂μ + ENNReal.ofReal t * ∫⁻ ω, hv ε κ t t' (H ω) * Sfb.indicator 1 ω ∂μ := by
        rw [lintegral_add_left ham, lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    _ ≤ ∫⁻ ω, a ω ∂μ + ENNReal.ofReal t * ∫⁻ ω, hv ε κ t t' (H ω) * ENNReal.ofReal (wr ω) ∂μ := by
        gcongr
    _ = ∫⁻ ω, a ω ∂μ + ∫⁻ ω, ENNReal.ofReal t * (hv ε κ t t' (H ω) * ENNReal.ofReal (wr ω)) ∂μ := by
        rw [lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
    _ ≤ ∫⁻ ω, (a ω + ENNReal.ofReal t * (hv ε κ t t' (H ω) * ENNReal.ofReal (wr ω))) ∂μ :=
        le_lintegral_add _ _
    _ ≤ _ := by
        refine lintegral_mono fun ω => ?_
        rw [ha]
        simp only [hv]
        rw [← ENNReal.ofReal_mul (hP0 _), ← ENNReal.ofReal_mul ht,
          ← ENNReal.ofReal_add (by positivity) (by positivity)]
        refine ENNReal.ofReal_le_ofReal ?_
        have h1 := tilt_le (hwr0 ω) (hw _ j) htw
        have := hP0 (H ω)
        nlinarith [mul_le_mul_of_nonneg_left h1 this]

/-- The mean weight stays at most its start, `1`. -/
theorem mean_le_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ) (hloc : Loc Q)
    {ε κ w t t' : ℝ} (hε : 0 ≤ ε) (hκ : 0 ≤ κ)
    (hB : ∀ z, μ.real {ω | read z ω ∈ B z} ≤ ε + κ * μ.real {ω | read z ω ∈ V z})
    (hw : ∀ r j, ε * (newQ Q j r).card + κ * (newV Q V j r).card ≤ w) (ht : 0 ≤ t)
    (ht' : 0 ≤ t') (htw : t * w ≤ 1 - Real.exp (-(t' * w))) :
    ∀ n, ∫⁻ ω, hv ε κ t t' (hist Q B V n (read · ω)) ∂μ ≤ 1
  | 0 => by simp [hist_zero, hv]
  | n + 1 => (step_le μ read hmeas hind hloc hε hκ hB hw ht ht' htw n).trans
      (mean_le_one hmeas hind hloc hε hκ hB hw ht ht' htw n)

end Main

end RaceTail

open MeasureTheory ProbabilityTheory RaceTail

open scoped ENNReal in
theorem raceTailBound_holds : RaceTailBound := by
  intro Z Ω R _ _ _ _ _ μ _ read B V ε κ Q f n b m w t t' hmeas hind hε hκ hB hloc hb hm hw hw0
    ht ht' htw
  classical
  have hlocf := loc_cut f hloc
  have hwf : ∀ r j, ε * (newQ (cut Q B f) j r).card + κ * (newV (cut Q B f) V j r).card ≤ w := by
    intro r j
    by_cases h : badCount Q B j r < f
    · simp only [newV, newQ_cut f h]; exact hw r j h
    · simp only [newV, newQ_cut_empty f h, Finset.filter_empty, Finset.card_empty,
        Nat.cast_zero, mul_zero, add_zero]
      exact hw0
  have hmean := mean_le_one μ read hmeas hind hlocf hε hκ hB hwf ht ht' htw n
  set x := ε * b + κ * m
  set thr := ENNReal.ofReal ((1 + t) ^ f * Real.exp (-(t' * x)))
  have hthr : 0 < (1 + t) ^ f * Real.exp (-(t' * x)) := by positivity
  have hsum : ∀ r : Z → R, ∑ i ∈ Finset.range n, (newQ (cut Q B f) i r).card
      = ∑ i ∈ (Finset.range n).filter (fun i => badCount Q B i r < f), (newQ Q i r).card := by
    intro r
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun i _ => ?_
    split_ifs with h
    · rw [newQ_cut f h]
    · rw [newQ_cut_empty f h]; rfl
  have hsumV : ∀ r : Z → R, ∑ i ∈ Finset.range n, (newV (cut Q B f) V i r).card
      = ∑ i ∈ (Finset.range n).filter (fun i => badCount Q B i r < f),
        ((newQ Q i r).filter fun z => r z ∈ V z).card := by
    intro r
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl fun i _ => ?_
    split_ifs with h
    · simp only [newV, newQ_cut f h]
    · simp only [newV, newQ_cut_empty f h]; rfl
  have hsubset : {ω | f ≤ badCount Q B n (read · ω)}
      ⊆ {ω | thr ≤ hv ε κ t t' (hist (cut Q B f) B V n (read · ω))} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq] at hω ⊢
    simp only [hv, hist, badCount_cut, min_eq_right hω, thr]
    refine ENNReal.ofReal_le_ofReal (mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_)
      (by positivity))
    have h1 := hb (read · ω)
    have h2 := hm (read · ω)
    rw [← Nat.cast_sum, ← hsum] at h1
    rw [← Nat.cast_sum, ← hsumV] at h2
    have : ε * (∑ i ∈ Finset.range n, (newQ (cut Q B f) i (read · ω)).card : ℕ)
        + κ * (∑ i ∈ Finset.range n, (newV (cut Q B f) V i (read · ω)).card : ℕ) ≤ x :=
      add_le_add (mul_le_mul_of_nonneg_left h1 hε) (mul_le_mul_of_nonneg_left h2 hκ)
    nlinarith [mul_le_mul_of_nonneg_left this ht']
  have hmeasf : Measurable fun ω => hv ε κ t t' (hist (cut Q B f) B V n (read · ω)) :=
    (measurable_of_countable (hv ε κ t t')).comp (hist_measurable (B := B) (V := V) hlocf read hmeas n)
  calc μ {ω | f ≤ badCount Q B n (read · ω)}
      ≤ μ {ω | thr ≤ hv ε κ t t' (hist (cut Q B f) B V n (read · ω))} := measure_mono hsubset
    _ ≤ (∫⁻ ω, hv ε κ t t' (hist (cut Q B f) B V n (read · ω)) ∂μ) / thr :=
        meas_ge_le_lintegral_div hmeasf.aemeasurable (by simpa [thr] using hthr)
          ENNReal.ofReal_ne_top
    _ ≤ 1 / thr := by gcongr
    _ = _ := by
        rw [one_div, ← ENNReal.ofReal_inv_of_pos hthr]
        congr 1
        rw [mul_inv, ← Real.exp_neg, neg_neg, div_eq_mul_inv, mul_comm]

end OrthoDFA
