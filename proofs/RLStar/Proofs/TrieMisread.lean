import RLStar.Proofs.TrieStage
import RLStar.Proofs.Hoeffding
import RLStar.Proofs.MiscutGate
import RLStar.Proofs.DFABridge

/-!
# The family's realized misreads over strings of one length

A family `F` read at the cut `(kl, kh)` reads the strings of length `L` off disjoint blocks of
the oracle's answers, so the share it misreads or leaves undecided exceeds its mean by `t` with
chance at most `exp(-2|Σ|^L t²)` (`misFrac_tail`). Every family of at most `FV` strings of length
`Ls`, at every cut, stays within `t` of its mean but for chance
`2 (FV + 1)³ (|Σ|^Ls + 1)^FV exp(-2|Σ|^L t²)` (`misread_union_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ]

open scoped Classical in
/-- The share of the strings of length `L` that `F`'s read at the noise `ω` misreads or leaves
undecided. -/
def misFrac (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ) (ω : Ω) : ℝ :=
  (uniformStrings L).real (Random.misreadSet (fun z => familyRead O.mq F kl kh z ω) O.L)

/-- Its mean over the noise. -/
def misMean (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ) : ℝ :=
  ∫ x, (misreadProb O F kl kh x + readProb O F kl kh x .undecided) ∂(uniformStrings L)

omit [DecidableEq α] [Nonempty α] [IsProbabilityMeasure μ] in
lemma familyRead_clamp (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (z : FreeMonoid α) (ω : Ω) :
    familyRead O.mq F kl kh z ω = familyRead O.mq F (min kl F.card) (min kh (F.card + 1)) z ω := by
  rw [familyRead_eq, familyRead_eq]
  have h := voteCount_le_card O.mq F z ω
  unfold readOf
  split_ifs <;> first | rfl | omega

lemma misFrac_clamp (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ)
    (ω : Ω) : misFrac O F kl kh L ω = misFrac O F (min kl F.card) (min kh (F.card + 1)) L ω := by
  have e : (fun z => familyRead O.mq F kl kh z ω)
      = fun z => familyRead O.mq F (min kl F.card) (min kh (F.card + 1)) z ω :=
    funext fun z => familyRead_clamp O F kl kh z ω
  unfold misFrac
  rw [e]

lemma misMean_clamp (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ) :
    misMean O F kl kh L = misMean O F (min kl F.card) (min kh (F.card + 1)) L := by
  unfold misMean misreadProb readProb
  simp only [familyRead_clamp O F kl kh]

open scoped Classical in
/-- The indicator that `F`'s read of `x` misreads or leaves it undecided. -/
def misVar (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh : ℕ)
    (x : FreeMonoid α) (ω : Ω) : ℝ :=
  if familyRead O.mq F kl kh x ω ≠ (if x ∈ O.L then ARU.accept else ARU.reject) then 1 else 0

omit [DecidableEq α] [Nonempty α] in
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

omit [DecidableEq α] [Nonempty α] in
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
    simp only [misVar, Set.indicator, E, Set.mem_setOf_eq, Pi.one_apply]
  rw [e1, integral_indicator_one hE]
  have e2 : E = {ω | familyRead O.mq F kl kh x ω = if x ∈ O.L then ARU.reject else ARU.accept}
      ∪ {ω | familyRead O.mq F kl kh x ω = .undecided} := by
    ext ω
    simp only [E, Set.mem_setOf_eq, Set.mem_union]
    split_ifs <;> rcases familyRead O.mq F kl kh x ω <;> simp
  rw [e2, measureReal_union (Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [Set.mem_setOf_eq] at h1 h2
      rw [h1] at h2
      split_ifs at h2) (hm _)]
  rfl

omit [IsProbabilityMeasure μ] in
lemma misFrac_eq (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ)
    (ω : Ω) : misFrac O F kl kh L ω
      = (∑ v : List.Vector α L, misVar O F kl kh (FreeMonoid.ofList v.toList) ω)
        / (Fintype.card α : ℝ) ^ L := by
  classical
  unfold misFrac
  rw [uniformStrings_real_uavg]
  unfold uavg
  congr 1
  refine Finset.sum_congr rfl fun v _ => ?_
  simp only [Set.indicator, Random.misreadSet, Set.mem_setOf_eq, Pi.one_apply, misVar]

lemma misMean_eq (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ) :
    misMean O F kl kh L
      = (∑ v : List.Vector α L, ∫ ω, misVar O F kl kh (FreeMonoid.ofList v.toList) ω ∂μ)
        / (Fintype.card α : ℝ) ^ L := by
  unfold misMean
  rw [integral_uniformStrings]
  unfold uavg
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

/-- `F`'s misread share over the strings of length `L` reaches its mean plus `t` with chance at
most `exp(-2|Σ|^L t²)`. -/
theorem misFrac_tail (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (kl kh L : ℕ)
    {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | misMean O F kl kh L + t ≤ misFrac O F kl kh L ω}
      ≤ Real.exp (-2 * (Fintype.card α : ℝ) ^ L * t ^ 2) := by
  classical
  have hs : ((Finset.univ : Finset (List.Vector α L)).card : ℝ) = (Fintype.card α : ℝ) ^ L := by
    simp [card_vector]
  have hA : (0 : ℝ) < (Fintype.card α : ℝ) ^ L := by positivity
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
  have hmean : ∑ v ∈ (Finset.univ : Finset (List.Vector α L)),
      ∫ ω, misVar O F kl kh (FreeMonoid.ofList v.toList) ω ∂μ
      ≤ ((Finset.univ : Finset (List.Vector α L)).card : ℝ) * misMean O F kl kh L := by
    rw [hs, misMean_eq (μ := μ), mul_div_cancel₀ _ hA.ne']
  have h := sumUpper_le (fun v : List.Vector α L => misVar O F kl kh (FreeMonoid.ofList v.toList))
    Finset.univ (misMean O F kl kh L) t hmeas hind hIcc hmean ht
  rw [hs] at h
  refine le_trans (measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)) h
  simp only [Set.mem_setOf_eq] at hω ⊢
  rw [misFrac_eq, le_div_iff₀ hA] at hω
  linarith

/-- The families of at most `FV` strings of length `Ls`. -/
def smallFamilies (Ls FV : ℕ) : Finset (Finset (FreeMonoid α)) :=
  (Finset.range (FV + 1)).biUnion fun j =>
    ((Finset.univ : Finset (List.Vector α Ls)).image fun v => FreeMonoid.ofList v.toList).powersetCard j

omit [Nonempty α] in
lemma mem_smallFamilies {Ls FV : ℕ} {F : Finset (FreeMonoid α)}
    (hF : ∀ v ∈ F, v.toList.length = Ls) (hc : F.card ≤ FV) : F ∈ smallFamilies Ls FV := by
  refine Finset.mem_biUnion.2 ⟨F.card, Finset.mem_range.2 (by omega),
    Finset.mem_powersetCard.2 ⟨fun v hv => ?_, rfl⟩⟩
  refine Finset.mem_image.2 ⟨⟨v.toList, hF v hv⟩, Finset.mem_univ _, ?_⟩
  simp

omit [Nonempty α] in
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

/-- Some family of at most `FV` strings of length `Ls`, at some cut, misreads more than its mean
plus `t` of the strings of length `L` with chance at most
`2 (FV + 1)³ (|Σ|^Ls + 1)^FV exp(-2|Σ|^L t²)`. -/
theorem misread_union_le (O : Oracle μ (FreeMonoid α)) (L Ls FV : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    μ.real {ω | ∃ (F : Finset (FreeMonoid α)) (kl kh : ℕ), (∀ v ∈ F, v.toList.length = Ls)
      ∧ F.card ≤ FV ∧ misMean O F kl kh L + t < misFrac O F kl kh L ω}
      ≤ 2 * (FV + 1) ^ 3 * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV
        * Real.exp (-2 * (Fintype.card α : ℝ) ^ L * t ^ 2) := by
  classical
  set I := smallFamilies (α := α) Ls FV ×ˢ (Finset.range (FV + 1) ×ˢ Finset.range (FV + 2))
  have hsub : {ω | ∃ (F : Finset (FreeMonoid α)) (kl kh : ℕ), (∀ v ∈ F, v.toList.length = Ls)
      ∧ F.card ≤ FV ∧ misMean O F kl kh L + t < misFrac O F kl kh L ω}
      ⊆ ⋃ q ∈ I, {ω | misMean O q.1 q.2.1 q.2.2 L + t ≤ misFrac O q.1 q.2.1 q.2.2 L ω} := by
    rintro ω ⟨F, kl, kh, hF, hc, hlt⟩
    refine Set.mem_biUnion (x := (F, min kl F.card, min kh (F.card + 1)))
      (Finset.mem_product.2 ⟨mem_smallFamilies hF hc, Finset.mem_product.2
        ⟨Finset.mem_range.2 (Nat.lt_succ_of_le ((min_le_right _ _).trans hc)),
          Finset.mem_range.2 (Nat.lt_succ_of_le ((min_le_right _ _).trans (by omega)))⟩⟩) ?_
    simp only [Set.mem_setOf_eq]
    rw [← misMean_clamp, ← misFrac_clamp]
    exact hlt.le
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans
    ((measureReal_biUnion_finset_le _ _).trans ?_)
  have hexp : 0 ≤ Real.exp (-2 * (Fintype.card α : ℝ) ^ L * t ^ 2) := (Real.exp_pos _).le
  calc ∑ q ∈ I, μ.real {ω | misMean O q.1 q.2.1 q.2.2 L + t ≤ misFrac O q.1 q.2.1 q.2.2 L ω}
      ≤ ∑ _q ∈ I, Real.exp (-2 * (Fintype.card α : ℝ) ^ L * t ^ 2) :=
        Finset.sum_le_sum fun q _ => misFrac_tail O q.1 q.2.1 q.2.2 L ht
    _ = I.card * Real.exp (-2 * (Fintype.card α : ℝ) ^ L * t ^ 2) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ 2 * (FV + 1) ^ 3 * ((Fintype.card α : ℝ) ^ Ls + 1) ^ FV
        * Real.exp (-2 * (Fintype.card α : ℝ) ^ L * t ^ 2) := by
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
