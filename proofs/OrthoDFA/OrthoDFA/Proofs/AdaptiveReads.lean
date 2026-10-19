import OrthoDFA.Utils.AdaptiveReads

/-!
# A union bound over adaptively chosen reads: the proof

A bad query has a first round, and there a string no earlier round queried. That the string is
first queried in round `j`, with no bad read before (`First`), is fixed by the reads of the
finitely many strings queried before, none of them the string itself, so it is independent of its
read.
-/

namespace OrthoDFA

namespace AdaptiveReads

open MeasureTheory ProbabilityTheory

variable {Z R : Type*} (Q : ℕ → (Z → R) → Finset Z) (B : Z → Set R)

open scoped Classical in
/-- The strings the rounds before `j` query. -/
noncomputable def before (j : ℕ) (r : Z → R) : Finset Z := (Finset.range j).biUnion fun i => Q i r

/-- `z` is first queried in round `j`, and no earlier query reads bad. -/
def First (j : ℕ) (z : Z) (r : Z → R) : Prop :=
  (∀ i < j, ∀ y ∈ Q i r, r y ∉ B y) ∧ z ∈ Q j r ∧ ∀ i < j, z ∉ Q i r

theorem mem_before {j : ℕ} {r : Z → R} {y : Z} : y ∈ before Q j r ↔ ∃ i < j, y ∈ Q i r := by
  classical
  simp [before]

/-- The rounds' locality. -/
def Local : Prop :=
  ∀ j r r', (∀ i < j, ∀ z ∈ Q i r, r z ∉ B z) → (∀ i < j, ∀ z ∈ Q i r, r z = r' z) →
    Q j r = Q j r'

variable {Q}

/-- Reads agreeing with `r` on what the rounds before `j` query, where none of those read bad
under `r`, agree with it on rounds up to `j`. -/
theorem Q_congr (hloc : Local Q B) {j : ℕ} {r r' : Z → R} (hg : ∀ i < j, ∀ y ∈ Q i r, r y ∉ B y)
    (h : ∀ y ∈ before Q j r, r y = r' y) {i : ℕ} (hi : i ≤ j) : Q i r = Q i r' :=
  hloc i r r' (fun l hl => hg l (by omega)) fun l hl y hy =>
    h y ((mem_before Q).2 ⟨l, by omega, hy⟩)

theorem first_congr (hloc : Local Q B) {j : ℕ} {z : Z} {r r' : Z → R}
    (h : ∀ y ∈ before Q j r, r y = r' y) (hF : First Q B j z r) : First Q B j z r' := by
  obtain ⟨h1, h2, h3⟩ := hF
  have hQ : ∀ i ≤ j, Q i r = Q i r' := fun i hi => Q_congr B hloc h1 h hi
  refine ⟨fun i hi y hy => ?_, hQ j le_rfl ▸ h2, fun i hi => hQ i hi.le ▸ h3 i hi⟩
  rw [← hQ i hi.le] at hy
  rw [← h y ((mem_before Q).2 ⟨i, hi, hy⟩)]
  exact h1 i hi y hy

variable [Countable Z] {Ω : Type*} [MeasurableSpace Ω] [MeasurableSpace R] [Countable R]
  [MeasurableSingletonClass R]

/-- The reads of every string but `z`. -/
abbrev others (read : Z → Ω → R) (z : Z) : MeasurableSpace Ω :=
  ⨆ y ∈ ({z}ᶜ : Set Z), MeasurableSpace.comap (read y) inferInstance

omit [MeasurableSpace Ω] in
/-- That `z` is first queried in round `j` is fixed by the other strings' reads. -/
theorem first_measurable (read : Z → Ω → R)
    (hloc : Local Q B) (j : ℕ) (z : Z) :
    MeasurableSet[others read z] {ω | First Q B j z (read · ω)} := by
  classical
  set T : (S : Finset Z) → Set (S → R) := fun S =>
    {a | ∃ r, (∀ y : S, r y = a y) ∧ before Q j r = S ∧ First Q B j z r}
  have hset : {ω | First Q B j z (read · ω)}
      = ⋃ (S : Finset Z) (_ : z ∉ S), (fun ω (y : S) => read y ω) ⁻¹' T S := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_preimage, exists_prop]
    constructor
    · intro h
      refine ⟨before Q j (read · ω), fun hz => ?_, (read · ω), fun _ => rfl, rfl, h⟩
      obtain ⟨i, hi, hzi⟩ := (mem_before Q).1 hz
      exact h.2.2 i hi hzi
    · rintro ⟨S, -, r, hr, hS, hF⟩
      have hag : ∀ y ∈ before Q j r, r y = read y ω := fun y hy => hr ⟨y, hS ▸ hy⟩
      exact first_congr B hloc hag hF
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

end AdaptiveReads

open MeasureTheory ProbabilityTheory AdaptiveReads

theorem adaptiveReadsBound_holds : AdaptiveReadsBound := by
  intro Z Ω R _ _ _ _ _ μ _ read B ε Q b n hmeas hind hB hloc hb
  classical
  have hle : ∀ z,
      MeasurableSpace.comap (read z) inferInstance ≤ (inferInstance : MeasurableSpace Ω) :=
    fun z => (hmeas z).comap_le
  have hoth : ∀ z, others read z ≤ (inferInstance : MeasurableSpace Ω) :=
    fun z => iSup₂_le fun y _ => hle y
  set F : ℕ → Z → Set Ω := fun j z => {ω | First Q B j z (read · ω)}
  set Bad : Z → Set Ω := fun z => read z ⁻¹' B z
  have hFm : ∀ j z, MeasurableSet (F j z) := fun j z =>
    hoth z _ (first_measurable B read hloc j z)
  have hBad : ∀ z, μ (Bad z) ≤ ENNReal.ofReal ε := fun z => by
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ (Bad z))]
    exact ENNReal.ofReal_le_ofReal (hB z)
  have hind' := (iIndepFun_iff_iIndep _ read μ).1 hind
  have hmul : ∀ j z, μ (F j z ∩ Bad z) = μ (F j z) * μ (Bad z) := by
    intro j z
    have hI := indep_iSup_of_disjoint hle hind' (S := {z}ᶜ) (T := {z})
      (Set.disjoint_compl_left_iff_subset.2 subset_rfl)
    refine (Indep_iff _ _ μ).1 hI _ _ (first_measurable B read hloc j z) ?_
    refine le_iSup₂ (f := fun y (_ : y ∈ ({z} : Set Z)) =>
      MeasurableSpace.comap (read y) inferInstance) z rfl _ ?_
    exact ⟨B z, (Set.to_countable _).measurableSet, rfl⟩
  have hcount : ∀ j, ∑' z, μ (F j z) ≤ b j := by
    intro j
    calc ∑' z, μ (F j z) = ∑' z, ∫⁻ ω, (F j z).indicator 1 ω ∂μ := by
          simp_rw [lintegral_indicator_one (hFm j _)]
      _ = ∫⁻ ω, ∑' z, (F j z).indicator 1 ω ∂μ :=
          (lintegral_tsum fun z => (measurable_one.indicator (hFm j z)).aemeasurable).symm
      _ ≤ ∫⁻ _, (b j : ENNReal) ∂μ := lintegral_mono fun ω => ?_
      _ = b j := by simp
    rw [tsum_eq_sum (s := Q j (read · ω)) fun z hz =>
      Set.indicator_of_notMem (fun h => hz h.2.1) _]
    by_cases hG : ∀ i < j, ∀ y ∈ Q i (read · ω), read y ω ∉ B y
    · calc _ ≤ ∑ _z ∈ Q j (read · ω), (1 : ENNReal) :=
            Finset.sum_le_sum fun z _ => by unfold Set.indicator; split_ifs <;> simp
        _ ≤ b j := by
            simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
            exact_mod_cast hb j _ hG
    · rw [Finset.sum_eq_zero fun z _ => Set.indicator_of_notMem (fun h => hG h.1) _]
      exact bot_le
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
  calc _ ≤ μ (⋃ j ∈ Finset.range n, ⋃ z, F j z ∩ Bad z) := measure_mono hsub
    _ ≤ ∑ j ∈ Finset.range n, μ (⋃ z, F j z ∩ Bad z) := measure_biUnion_finset_le _ _
    _ ≤ ∑ j ∈ Finset.range n, ∑' z, μ (F j z) * ENNReal.ofReal ε :=
        Finset.sum_le_sum fun j _ => (measure_iUnion_le _).trans
          (ENNReal.tsum_le_tsum fun z => by rw [hmul]; gcongr; exact hBad z)
    _ = ∑ j ∈ Finset.range n, (∑' z, μ (F j z)) * ENNReal.ofReal ε := by
        simp_rw [ENNReal.tsum_mul_right]
    _ ≤ ∑ j ∈ Finset.range n, (b j : ENNReal) * ENNReal.ofReal ε := by
        gcongr with j
        exact hcount j
    _ ≤ _ := by
        by_cases hε : 0 ≤ ε
        · rw [ENNReal.ofReal_mul hε, ENNReal.ofReal_sum_of_nonneg (fun _ _ => by positivity),
            Finset.mul_sum]
          simp only [ENNReal.ofReal_natCast, mul_comm]
          exact le_rfl
        · rw [ENNReal.ofReal_of_nonpos (by linarith)]
          simp

end OrthoDFA
