import OrthoDFA.Proofs.Level5EdgeNoise

/-!
# Fixing the reads of a finite set

`ovr Z rd c` reads the strings of `Z` as `c` and every other string as `rd`. Over independent
reads, fixing those of `Z` leaves the rest independent with their laws (`iIndepFun_ovr`), and an
event read through finitely many strings has at most the chance it has at the worst fixing of
`Z`'s reads (`measure_le_of_fixed`): it is a mixture of its chances at each fixing, the reads of
`Z` independent of the rest.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `rd` with the strings of `Z` read as `c`. -/
def ovr (Z : Finset (FreeMonoid α)) (rd : FreeMonoid α → ARU) (c : Z → ARU) :
    FreeMonoid α → ARU :=
  fun z => if h : z ∈ Z then c ⟨z, h⟩ else rd z

theorem ovr_of_notMem {Z : Finset (FreeMonoid α)} {rd : FreeMonoid α → ARU} {c : Z → ARU}
    {z : FreeMonoid α} (hz : z ∉ Z) : ovr Z rd c z = rd z := by
  simp only [ovr, dif_neg hz]

theorem ovr_self (Z : Finset (FreeMonoid α)) (rd : FreeMonoid α → ARU) :
    ovr Z rd (fun z => rd z) = rd := by
  funext z; simp only [ovr]; split_ifs <;> rfl

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU}

theorem measurable_ovr (hmeas : ∀ z, Measurable (read z)) (Z : Finset (FreeMonoid α))
    (c : Z → ARU) (z : FreeMonoid α) : Measurable fun ω => ovr Z (read · ω) c z := by
  by_cases hz : z ∈ Z
  · simp only [ovr, dif_pos hz]; exact measurable_const
  · simp only [ovr, dif_neg hz]; exact hmeas z

omit [IsProbabilityMeasure μ] in
/-- Fixing the reads of `Z` keeps the reads independent. -/
theorem iIndepFun_ovr (hind : iIndepFun read μ) (Z : Finset (FreeMonoid α)) (c : Z → ARU) :
    iIndepFun (fun z ω => ovr Z (read · ω) c z) μ := by
  have := hind.comp (fun z (a : ARU) => if h : z ∈ Z then c ⟨z, h⟩ else a)
    fun _ => measurable_of_countable _
  have h : (fun z ω => ovr Z (read · ω) c z)
      = fun z => (fun a => if h : z ∈ Z then c ⟨z, h⟩ else a) ∘ read z := by
    funext z ω; simp [ovr]
  rw [h]; exact this

/-- An event read through the strings `W` has at most the chance it has at the worst fixing of
the reads of `Z`. -/
theorem measure_le_of_fixed (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (Z W : Finset (FreeMonoid α)) (F : (FreeMonoid α → ARU) → Prop)
    (hF : ∀ rd rd', (∀ z ∈ W, rd z = rd' z) → (F rd ↔ F rd')) (b : ENNReal)
    (hb : ∀ c : Z → ARU, μ {ω | F (ovr Z (read · ω) c)} ≤ b) :
    μ {ω | F (read · ω)} ≤ b := by
  classical
  have : Fintype ARU := Fintype.ofFinite ARU
  set R := W \ Z
  have hdisj : Disjoint R Z := Finset.sdiff_disjoint
  have hI := hind.indepFun_finset R Z hdisj hmeas
  set A : (Z → ARU) → Set Ω := fun c => {ω | F (ovr Z (read · ω) c)}
  set B : (Z → ARU) → Set Ω := fun c => {ω | ∀ z : Z, read z ω = c z}
  set Sc : (Z → ARU) → Set (R → ARU) := fun c => {v | F (ovr Z (rdOf R v) c)}
  have hA : ∀ c, A c = (fun ω (i : R) => read i ω) ⁻¹' Sc c := by
    intro c
    ext ω
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, A, Sc]
    refine hF _ _ fun z hz => ?_
    by_cases hzZ : z ∈ Z
    · simp only [ovr, dif_pos hzZ]
    · have hzR : z ∈ R := Finset.mem_sdiff.2 ⟨hz, hzZ⟩
      simp only [ovr, dif_neg hzZ, rdOf, dif_pos hzR]
  have hB : ∀ c, B c = (fun ω (i : Z) => read i ω) ⁻¹' {c} := by
    intro c
    ext ω
    simp only [Set.mem_preimage, Set.mem_singleton_iff, B, Set.mem_ofPred_eq, funext_iff]
  have hmul : ∀ c, μ (A c ∩ B c) = μ (A c) * μ (B c) := by
    intro c
    rw [hA, hB]
    exact hI.measure_inter_preimage_eq_mul _ _ (Set.to_countable _).measurableSet
      (measurableSet_singleton _)
  have hsub : {ω | F (read · ω)} ⊆ ⋃ c, A c ∩ B c := by
    intro ω hω
    refine Set.mem_iUnion.2 ⟨fun z => read z ω, ?_, fun z => rfl⟩
    simp only [A, Set.mem_ofPred_eq, ovr_self Z (read · ω)]
    exact hω
  have hBm : ∀ c, MeasurableSet (B c) := fun c => by
    rw [hB]; exact (measurable_pi_lambda (fun ω (i : Z) => read i ω) fun i => hmeas i)
      (measurableSet_singleton c)
  have hBdisj : Pairwise (Function.onFun Disjoint B) := by
    intro c c' hne
    refine Set.disjoint_left.2 fun ω h1 h2 => hne ?_
    funext z
    rw [← h1 z, ← h2 z]
  calc μ {ω | F (read · ω)} ≤ μ (⋃ c, A c ∩ B c) := measure_mono hsub
    _ ≤ ∑' c, μ (A c ∩ B c) := measure_iUnion_le _
    _ = ∑' c, μ (A c) * μ (B c) := by simp_rw [hmul]
    _ ≤ ∑' c, b * μ (B c) := ENNReal.tsum_le_tsum fun c => by gcongr; exact hb c
    _ = b * μ (⋃ c, B c) := by rw [ENNReal.tsum_mul_left, measure_iUnion hBdisj hBm]
    _ ≤ b * 1 := by gcongr; exact prob_le_one
    _ = b := mul_one b

end OrthoDFA
