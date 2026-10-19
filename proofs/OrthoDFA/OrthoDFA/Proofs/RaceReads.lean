import OrthoDFA.Utils.RaceReads
import OrthoDFA.Proofs.AdaptiveReads

/-!
# Bad reads against undecided ones: the proof

A bad query has a first round, and there a string no earlier round queried. That the string is
first queried in round `j`, with no bad read before (`First`), does not change with the string's
own read: changing it leaves the earlier rounds alone, and round `j`, fixed by its own queries'
reads, would otherwise not query the string. So it is fixed by the reads of the other strings that
rounds `0, …, j` query, and independent of the string's read, bad or in `V`.
-/

namespace OrthoDFA

namespace RaceReads

open MeasureTheory ProbabilityTheory AdaptiveReads

variable {Z R : Type*} (Q : ℕ → (Z → R) → Finset Z) (B : Z → Set R)

/-- The rounds' locality: a round is fixed by what it and the rounds before it read. -/
def WLocal : Prop :=
  ∀ j r r', (∀ i < j, ∀ z ∈ Q i r, r z ∉ B z) → (∀ i ≤ j, ∀ z ∈ Q i r, r z = r' z) →
    Q j r = Q j r'

variable {Q}

/-- Reads agreeing with `r` on what the rounds before `j` query, none of those reading bad under
`r`, agree with it on the rounds before `j`. -/
theorem Q_congr' (hloc : WLocal Q B) {j : ℕ} {r r' : Z → R}
    (hg : ∀ i < j, ∀ y ∈ Q i r, r y ∉ B y) (h : ∀ y ∈ before Q j r, r y = r' y) :
    ∀ i < j, Q i r = Q i r' :=
  fun i hi => hloc i r r' (fun l hl => hg l (by omega)) fun l hl y hy =>
    h y ((mem_before Q).2 ⟨l, by omega, hy⟩)

open scoped Classical in
/-- `First` does not change with the string's own read. -/
theorem first_update [DecidableEq Z] (hloc : WLocal Q B) {j : ℕ} {z : Z} {r : Z → R} (v : R)
    (hF : First Q B j z r) : First Q B j z (Function.update r z v) := by
  obtain ⟨h1, h2, h3⟩ := hF
  set r' := Function.update r z v
  have hag : ∀ y ∈ before Q j r, r y = r' y := fun y hy => by
    obtain ⟨i, hi, hyi⟩ := (mem_before Q).1 hy
    have hyz : y ≠ z := fun e => h3 i hi (e ▸ hyi)
    simp [r', Function.update_of_ne hyz]
  have hQ := Q_congr' B hloc h1 hag
  have h1' : ∀ i < j, ∀ y ∈ Q i r', r' y ∉ B y := fun i hi y hy => by
    rw [← hQ i hi] at hy
    rw [← hag y ((mem_before Q).2 ⟨i, hi, hy⟩)]
    exact h1 i hi y hy
  refine ⟨h1', ?_, fun i hi => hQ i hi ▸ h3 i hi⟩
  by_contra hz
  have := hloc j r' r h1' fun i hi y hy => by
    rcases Nat.lt_or_eq_of_le hi with hi | rfl
    · rw [← hQ i hi] at hy
      exact (hag y ((mem_before Q).2 ⟨i, hi, hy⟩)).symm
    · have hyz : y ≠ z := fun e => hz (e ▸ hy)
      simp [r', Function.update_of_ne hyz]
  exact hz (this ▸ h2)

open scoped Classical in
/-- Reads agreeing with `r` on what rounds `0, …, j` query but `z` keep `z` first queried in
round `j`. -/
theorem first_congr' (hloc : WLocal Q B) {j : ℕ} {z : Z} {r r' : Z → R}
    (h : ∀ y ∈ before Q j r ∪ (Q j r).erase z, r y = r' y) (hF : First Q B j z r) :
    First Q B j z r' := by
  set r'' := Function.update r' z (r z)
  have hag : ∀ y ∈ before Q j r, r y = r'' y := fun y hy => by
    obtain ⟨i, hi, hyi⟩ := (mem_before Q).1 hy
    have hyz : y ≠ z := fun e => hF.2.2 i hi (e ▸ hyi)
    simp only [r'', Function.update_of_ne hyz]
    exact h y (Finset.mem_union_left _ hy)
  have hQ := Q_congr' B hloc hF.1 hag
  have hQj : Q j r = Q j r'' := hloc j r r'' hF.1 fun i hi y hy => by
    rcases Nat.lt_or_eq_of_le hi with hi | rfl
    · exact hag y ((mem_before Q).2 ⟨i, hi, hy⟩)
    · by_cases hyz : y = z
      · subst hyz; simp [r'']
      · simp only [r'', Function.update_of_ne hyz]
        exact h y (Finset.mem_union_right _ (Finset.mem_erase.2 ⟨hyz, hy⟩))
  have hF'' : First Q B j z r'' := by
    refine ⟨fun i hi y hy => ?_, hQj ▸ hF.2.1, fun i hi => hQ i hi ▸ hF.2.2 i hi⟩
    rw [← hQ i hi] at hy
    rw [← hag y ((mem_before Q).2 ⟨i, hi, hy⟩)]
    exact hF.1 i hi y hy
  have := first_update B hloc (r' z) hF''
  simpa [r''] using this

theorem sum_lintegral_le {ι Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} (s : Finset ι)
    (f : ι → Ω → ENNReal) : ∑ j ∈ s, ∫⁻ ω, f j ω ∂μ ≤ ∫⁻ ω, ∑ j ∈ s, f j ω ∂μ := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert j s hj ih =>
    rw [Finset.sum_insert hj]
    simp_rw [Finset.sum_insert hj]
    exact (add_le_add le_rfl ih).trans (le_lintegral_add _ _)

variable [Countable Z] {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
  [MeasurableSingletonClass R]

omit [MeasurableSpace Ω] in
/-- That `z` is first queried in round `j` is fixed by the other strings' reads. -/
theorem first_measurable' (read : Z → Ω → R) (hloc : WLocal Q B) (j : ℕ) (z : Z) :
    MeasurableSet[others read z] {ω | First Q B j z (read · ω)} := by
  classical
  set T : (S : Finset Z) → Set (S → R) := fun S =>
    {a | ∃ r, (∀ y : S, r y = a y) ∧ before Q j r ∪ (Q j r).erase z = S ∧ First Q B j z r}
  have hset : {ω | First Q B j z (read · ω)}
      = ⋃ (S : Finset Z) (_ : z ∉ S), (fun ω (y : S) => read y ω) ⁻¹' T S := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_preimage, exists_prop]
    constructor
    · intro h
      refine ⟨before Q j (read · ω) ∪ (Q j (read · ω)).erase z, fun hz => ?_, (read · ω),
        fun _ => rfl, rfl, h⟩
      rcases Finset.mem_union.1 hz with hz | hz
      · obtain ⟨i, hi, hzi⟩ := (mem_before Q).1 hz
        exact h.2.2 i hi hzi
      · simp at hz
    · rintro ⟨S, -, r, hr, hS, hF⟩
      exact first_congr' B hloc (fun y hy => hr ⟨y, hS ▸ hy⟩) hF
  rw [hset]
  refine MeasurableSet.iUnion fun S => MeasurableSet.iUnion fun hzS => ?_
  have hm : Measurable[others read z] fun ω (y : S) => read y ω := by
    let : MeasurableSpace Ω := others read z
    refine measurable_pi_iff.2 fun y => ?_
    refine Measurable.of_comap_le ?_
    refine le_trans ?_ (le_iSup₂ (f := fun y (_ : y ∈ ({z}ᶜ : Set Z)) =>
      MeasurableSpace.comap (read y) inferInstance) y.1 ?_)
    · exact le_rfl
    · exact fun h => hzS (h ▸ y.2)
  exact hm ((Set.to_countable _).measurableSet)

end RaceReads

open MeasureTheory ProbabilityTheory AdaptiveReads RaceReads

theorem raceReadsBound_holds : RaceReadsBound := by
  intro Z Ω R _ _ _ _ _ μ _ read B V ε κ Q b m n hmeas hind hε hκ hB hloc hb hm
  classical
  have hle : ∀ z,
      MeasurableSpace.comap (read z) inferInstance ≤ (inferInstance : MeasurableSpace Ω) :=
    fun z => (hmeas z).comap_le
  have hoth : ∀ z, others read z ≤ (inferInstance : MeasurableSpace Ω) :=
    fun z => iSup₂_le fun y _ => hle y
  set F : ℕ → Z → Set Ω := fun j z => {ω | First Q B j z (read · ω)}
  set Bad : Z → Set Ω := fun z => read z ⁻¹' B z
  set Vs : Z → Set Ω := fun z => read z ⁻¹' V z
  have hFm : ∀ j z, MeasurableSet (F j z) := fun j z =>
    hoth z _ (first_measurable' B read hloc j z)
  have hVm : ∀ z, MeasurableSet (Vs z) := fun z =>
    hmeas z ((Set.to_countable _).measurableSet)
  have hind' := (iIndepFun_iff_iIndep _ read μ).1 hind
  have hmul : ∀ j z (A : Set R), μ (F j z ∩ read z ⁻¹' A) = μ (F j z) * μ (read z ⁻¹' A) := by
    intro j z A
    have hI := indep_iSup_of_disjoint hle hind' (S := {z}ᶜ) (T := {z})
      (Set.disjoint_compl_left_iff_subset.2 subset_rfl)
    refine (Indep_iff _ _ μ).1 hI _ _ (first_measurable' B read hloc j z) ?_
    refine le_iSup₂ (f := fun y (_ : y ∈ ({z} : Set Z)) =>
      MeasurableSpace.comap (read y) inferInstance) z rfl _ ?_
    exact ⟨A, (Set.to_countable _).measurableSet, rfl⟩
  have hBad : ∀ z, μ (Bad z) ≤ ENNReal.ofReal ε + ENNReal.ofReal κ * μ (Vs z) := fun z => by
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ (Bad z)),
      ← ENNReal.ofReal_toReal (measure_ne_top μ (Vs z)), ← ENNReal.ofReal_mul hκ,
      ← ENNReal.ofReal_add hε (by positivity)]
    exact ENNReal.ofReal_le_ofReal (hB z)
  set G : (Z → R) → ℕ → Prop := fun r j => ∀ i < j, ∀ z ∈ Q i r, r z ∉ B z
  have hpt : ∀ (A : Z → Set R) (c : ℝ),
      (∀ r, ∑ j ∈ (Finset.range n).filter (G r),
        (((Q j r).filter fun z => r z ∈ A z).card : ℝ) ≤ c) →
      ∑ j ∈ Finset.range n, ∑' z, μ (F j z ∩ read z ⁻¹' A z) ≤ ENNReal.ofReal c := by
    intro A c hc
    have hAm : ∀ j z, MeasurableSet (F j z ∩ read z ⁻¹' A z) := fun j z =>
      (hFm j z).inter (hmeas z ((Set.to_countable _).measurableSet))
    calc _ = ∑ j ∈ Finset.range n, ∫⁻ ω, ∑' z, (F j z ∩ read z ⁻¹' A z).indicator 1 ω ∂μ := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [lintegral_tsum fun z => (measurable_one.indicator (hAm j z)).aemeasurable]
          simp_rw [lintegral_indicator_one (hAm j _)]
      _ ≤ ∫⁻ ω, ∑ j ∈ Finset.range n, ∑' z, (F j z ∩ read z ⁻¹' A z).indicator 1 ω ∂μ :=
          sum_lintegral_le _ _
      _ ≤ ∫⁻ _, ENNReal.ofReal c ∂μ := lintegral_mono fun ω => ?_
      _ = ENNReal.ofReal c := by simp
    have hj : ∀ j, ∑' z, (F j z ∩ read z ⁻¹' A z).indicator (1 : Ω → ENNReal) ω
        ≤ if G (read · ω) j then
            ENNReal.ofReal (((Q j (read · ω)).filter fun z => read z ω ∈ A z).card : ℝ) else 0 := by
      intro j
      rw [tsum_eq_sum (s := Q j (read · ω)) fun z hz =>
        Set.indicator_of_notMem (fun h => hz h.1.2.1) _]
      split_ifs with hG
      · rw [Finset.card_filter, Nat.cast_sum, ENNReal.ofReal_sum_of_nonneg fun _ _ => by positivity]
        refine Finset.sum_le_sum fun z _ => ?_
        by_cases hzA : read z ω ∈ A z
        · simp only [hzA, if_true, Nat.cast_one, ENNReal.ofReal_one]
          unfold Set.indicator; split_ifs <;> simp
        · rw [Set.indicator_of_notMem (fun h => hzA h.2)]
          exact bot_le
      · exact (Finset.sum_eq_zero fun z _ => Set.indicator_of_notMem (fun h => hG h.1.1) _).le
    calc _ ≤ ∑ j ∈ Finset.range n, if G (read · ω) j then
            ENNReal.ofReal (((Q j (read · ω)).filter fun z => read z ω ∈ A z).card : ℝ) else 0 :=
          Finset.sum_le_sum fun j _ => hj j
      _ = ENNReal.ofReal (∑ j ∈ (Finset.range n).filter (G (read · ω)),
            (((Q j (read · ω)).filter fun z => read z ω ∈ A z).card : ℝ)) := by
          rw [← Finset.sum_filter, ENNReal.ofReal_sum_of_nonneg fun _ _ => by positivity]
      _ ≤ _ := ENNReal.ofReal_le_ofReal (hc _)
  have hsub : {ω | ∃ j < n, ∃ z ∈ Q j (read · ω), read z ω ∈ B z}
      ⊆ ⋃ j ∈ Finset.range n, ⋃ z, F j z ∩ Bad z := by
    intro ω ⟨j, hj, hex⟩
    have hP : ∃ j, j < n ∧ ∃ z ∈ Q j (read · ω), read z ω ∈ B z := ⟨j, hj, hex⟩
    set j₀ := Nat.find hP
    obtain ⟨hj₀, z, hz, hzB⟩ := Nat.find_spec hP
    have hmin : ∀ i < j₀, ∀ y ∈ Q i (read · ω), read y ω ∉ B y := fun i hi y hy hyB =>
      Nat.find_min hP hi ⟨by omega, y, hy, hyB⟩
    refine Set.mem_biUnion (Finset.mem_range.2 hj₀) (Set.mem_iUnion.2 ⟨z, ?_, hzB⟩)
    exact ⟨hmin, hz, fun i hi hzi => hmin i hi z hzi hzB⟩
  have hall := hpt (fun _ => Set.univ) b (fun r => by simpa using hb r)
  have hV := hpt V m hm
  simp only [Set.preimage_univ, Set.inter_univ] at hall
  calc _ ≤ μ (⋃ j ∈ Finset.range n, ⋃ z, F j z ∩ Bad z) := measure_mono hsub
    _ ≤ ∑ j ∈ Finset.range n, μ (⋃ z, F j z ∩ Bad z) := measure_biUnion_finset_le _ _
    _ ≤ ∑ j ∈ Finset.range n, ∑' z, (ENNReal.ofReal ε * μ (F j z)
          + ENNReal.ofReal κ * μ (F j z ∩ Vs z)) :=
        Finset.sum_le_sum fun j _ => (measure_iUnion_le _).trans
          (ENNReal.tsum_le_tsum fun z => by
            rw [hmul, hmul j z (V z)]
            calc μ (F j z) * μ (Bad z)
                ≤ μ (F j z) * (ENNReal.ofReal ε + ENNReal.ofReal κ * μ (Vs z)) := by
                  gcongr; exact hBad z
              _ = _ := by ring)
    _ = ENNReal.ofReal ε * ∑ j ∈ Finset.range n, ∑' z, μ (F j z)
          + ENNReal.ofReal κ * ∑ j ∈ Finset.range n, ∑' z, μ (F j z ∩ Vs z) := by
        rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [ENNReal.tsum_add, ENNReal.tsum_mul_left, ENNReal.tsum_mul_left]
    _ ≤ ENNReal.ofReal ε * ENNReal.ofReal b + ENNReal.ofReal κ * ENNReal.ofReal m := by
        gcongr
    _ = _ := by
        obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
        have hb0 : 0 ≤ b := (Finset.sum_nonneg fun _ _ => by positivity).trans (hb (read · ω₀))
        have hm0 : 0 ≤ m := (Finset.sum_nonneg fun _ _ => by positivity).trans (hm (read · ω₀))
        rw [← ENNReal.ofReal_mul hε, ← ENNReal.ofReal_mul hκ,
          ENNReal.ofReal_add (by positivity) (by positivity)]

end OrthoDFA
