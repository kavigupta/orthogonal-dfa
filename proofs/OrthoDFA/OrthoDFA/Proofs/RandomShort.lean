import OrthoDFA.Proofs.RandomGood

/-!
# What the probes harvest at good read-states, past a depth `kd`

A probe walked from `k` under `T` reads strings `u m`, `u` a prefix of it and `m` a midfix of
`T`. Those with `|u| < kd` (`headOf k kd T`) are shared by probes of different length-`kd`
prefixes. An event has at most its chance at the worst fixing of their reads, the rest staying
independent (`measure_le_of_fixed`); with them fixed, what the probes harvest at good read-states
off them (`goodOff`) concentrates over the length-`kd` prefixes as `block_tail` says. Where none
of them at a good read-state reads undecided, that is all they harvest there
(`goodAt_subset_off`).
-/

namespace OrthoDFA

namespace Random

open MeasureTheory ProbabilityTheory

variable {α : Type*}

/-! ## Fixing the reads of a finite set -/

section Fix

variable [Fintype α]

open scoped Classical in
/-- `rd` with the strings of `Z` read as `c`. -/
noncomputable def ovr (Z : Finset (FreeMonoid α)) (rd : FreeMonoid α → ARU) (c : Z → ARU) :
    FreeMonoid α → ARU :=
  fun z => if h : z ∈ Z then c ⟨z, h⟩ else rd z

theorem ovr_self (Z : Finset (FreeMonoid α)) (rd : FreeMonoid α → ARU) :
    ovr Z rd (fun z => rd z) = rd := by
  funext z; simp only [ovr]; split_ifs <;> rfl

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {read : FreeMonoid α → Ω → ARU}

theorem measurable_ovr (hmeas : ∀ z, Measurable (read z)) (Z : Finset (FreeMonoid α))
    (c : Z → ARU) (z : FreeMonoid α) : Measurable fun ω => ovr Z (read · ω) c z := by
  by_cases hz : z ∈ Z
  · simp only [ovr, dif_pos hz]; exact measurable_const
  · simp only [ovr, dif_neg hz]; exact hmeas z

theorem iIndepFun_ovr (hind : iIndepFun read μ) (Z : Finset (FreeMonoid α)) (c : Z → ARU) :
    iIndepFun (fun z ω => ovr Z (read · ω) c z) μ := by
  have := hind.comp (fun z (a : ARU) => if h : z ∈ Z then c ⟨z, h⟩ else a)
    fun _ => measurable_of_countable _
  have h : (fun z ω => ovr Z (read · ω) c z)
      = fun z => (fun a => if h : z ∈ Z then c ⟨z, h⟩ else a) ∘ read z := by
    funext z ω; simp [ovr]
  rw [h]; exact this

variable [IsProbabilityMeasure μ]

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
    change F (ovr Z (read · ω) fun z => read z ω)
    rw [show ovr Z (read · ω) (fun z => read z ω) = (read · ω) from ovr_self Z _]
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

end Fix

/-! ## The strings before `kd` -/

section Head

variable [Fintype α]

/-- The strings `u m` with `k ≤ |u| < kd` and `m` a midfix of `T`. -/
noncomputable def headOf (k kd : ℕ) (T : DTree α) : Finset (FreeMonoid α) :=
  (Finset.Ico k kd).biUnion fun i => (lenK i).biUnion fun u => (mids T).image (u * ·)

theorem card_lenK (i : ℕ) : (lenK i : Finset (FreeMonoid α)).card ≤ Fintype.card α ^ i := by
  unfold lenK
  exact Finset.card_image_le.trans (by simp)

theorem card_headOf (k kd : ℕ) (T : DTree α) :
    ((headOf k kd T).card : ℝ)
      ≤ (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (mids T).card := by
  have h : (headOf k kd T).card ≤ (∑ i ∈ Finset.Ico k kd, Fintype.card α ^ i) * (mids T).card := by
    unfold headOf
    refine Finset.card_biUnion_le.trans ?_
    rw [Finset.sum_mul]
    refine Finset.sum_le_sum fun i _ => Finset.card_biUnion_le.trans ?_
    calc _ ≤ ∑ _u ∈ lenK i, (mids T).card := Finset.sum_le_sum fun u _ => Finset.card_image_le
      _ = (lenK i : Finset (FreeMonoid α)).card * (mids T).card := by
          rw [Finset.sum_const, smul_eq_mul]
      _ ≤ _ := Nat.mul_le_mul_right _ (card_lenK i)
  exact_mod_cast h

theorem head_mem {k kd p : ℕ} {x m : FreeMonoid α} {T : DTree α} (hkp : k ≤ p) (hpd : p < kd)
    (hpx : p ≤ x.toList.length) (hm : m ∈ mids T) : prefixOf x p * m ∈ headOf k kd T :=
  Finset.mem_biUnion.2 ⟨p, Finset.mem_Ico.2 ⟨hkp, hpd⟩, Finset.mem_biUnion.2 ⟨prefixOf x p,
    mem_lenK.2 (by simp [prefixOf]; omega), Finset.mem_image.2 ⟨m, hm, rfl⟩⟩⟩

/-- A string a probe of `x` walked from `k` can read under `T`, off `headOf k kd T`, has `x`'s
length-`kd` prefix. -/
theorem pot_deep {k kd : ℕ} {x z : FreeMonoid α} {T : DTree α} (hz : z ∈ pot k x T)
    (hzh : z ∉ headOf k kd T) : prefixOf z kd = prefixOf x kd := by
  classical
  unfold pot at hz
  obtain ⟨p, hp, hz⟩ := Finset.mem_biUnion.1 hz
  obtain ⟨hkp, hpx⟩ := Finset.mem_Icc.1 hp
  obtain ⟨m, hm, rfl⟩ := Finset.mem_image.1 hz
  by_cases hpd : p < kd
  · exact absurd (head_mem hkp hpd hpx hm) hzh
  simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList]
  rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left (by omega)]

end Head

/-! ## Off a set of strings -/

variable [Fintype α] [DecidableEq α] {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ)

/-- The probes that harvest at the site a string off `Z` at a good read-state. -/
def goodOff (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (Z : Finset (FreeMonoid α)) (site : Level3Site) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ siteHarvest read k H site x, z ∉ Z ∧ ¬ BadAt U θ (M.eval z.toList)}

omit [DecidableEq α] in
/-- Where no string of `Z` at a good read-state reads undecided, every probe harvesting at a good
read-state harvests off `Z`. -/
theorem goodAt_subset_off (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (Z : Finset (FreeMonoid α)) (site : Level3Site)
    (h : ∀ z ∈ Z, ¬ BadAt U θ (M.eval z.toList) → read z ≠ .undecided) :
    goodAt read k H M U θ site ⊆ goodOff M U θ read k H Z site := by
  rintro x ⟨z, hz, hg⟩
  exact ⟨z, hz, fun hzZ => h z hzZ hg (siteHarvest_pot read k H site x z hz).2, hg⟩

section One

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU} (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)
  (Z : Finset (FreeMonoid α))

omit [Fintype α] [DecidableEq α] in
theorem goodOff_congr {rd rd' : FreeMonoid α → ARU} (site : Level3Site)
    (h : ∀ z ∈ pot k x T, rd z = rd' z) :
    x ∈ goodOff M U θ rd k ⟨T, E⟩ Z site ↔ x ∈ goodOff M U θ rd' k ⟨T, E⟩ Z site := by
  cases site <;> simp only [goodOff, siteHarvest, Set.mem_ofPred_eq,
    probeOrAgree_congr rd T E k x h]

/-- `mid_mean_le` off `Z`, the reads off `Z` at good read-states undecided with chance at most
`1.5θ`. -/
theorem mid_mean_off (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hgood : ∀ z ∉ Z, ¬ BadAt U θ (M.eval z.toList) →
      μ.real {ω | read z ω = .undecided} ≤ 3 / 2 * θ) (hθ : 0 ≤ θ) :
    μ.real {ω | x ∈ goodOff M U θ (read · ω) k ⟨T, E⟩ Z .middles}
      ≤ (pot k x T).card * (3 / 2 * θ)
        * μ.real {ω | searches (probeOrAgree (read · ω) T E k x) = true} := by
  classical
  set Srch := {ω | searches (probeOrAgree (read · ω) T E k x) = true}
  set Zs := (pot k x T).filter fun z => z ∉ Z ∧ ¬ BadAt U θ (M.eval z.toList)
  have hsub : {ω | x ∈ goodOff M U θ (read · ω) k ⟨T, E⟩ Z .middles}
      ⊆ ⋃ z ∈ Zs, {ω | searches (probeOrAgree (read · ω) T E k x) = true
        ∧ read z ω = .undecided} := by
    intro ω hω
    obtain ⟨z, hz, hzZ, hg⟩ := hω
    obtain ⟨hzp, hzu⟩ := probeOrAgree_pt (read · ω) T x E k z hz
    exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hzp, hzZ, hg⟩)
      ⟨searches_of_ptStr (List.ne_nil_of_mem hz), hzu⟩
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top μ _)
    _ ≤ ∑ z ∈ Zs, μ.real {ω | searches (probeOrAgree (read · ω) T E k x) = true
          ∧ read z ω = .undecided} := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _z ∈ Zs, μ.real Srch * (3 / 2 * θ) := by
        refine Finset.sum_le_sum fun z hz => (search_und_le T E k x hmeas hind z).trans ?_
        obtain ⟨-, hzZ, hg⟩ := Finset.mem_filter.1 hz
        exact mul_le_mul_of_nonneg_left (hgood z hzZ hg) measureReal_nonneg
    _ = Zs.card * (3 / 2 * θ) * μ.real Srch := by rw [Finset.sum_const, nsmul_eq_mul]; ring
    _ ≤ _ := by
        have : (Zs.card : ℝ) ≤ (pot k x T).card := by exact_mod_cast Finset.card_filter_le _ _
        gcongr

/-- `ends_mean_le` off `Z`, the reads off `Z` at good read-states undecided with chance at most
`1.5θ`. -/
theorem ends_mean_off (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hgood : ∀ z ∉ Z, ¬ BadAt U θ (M.eval z.toList) →
      μ.real {ω | read z ω = .undecided} ≤ 3 / 2 * θ) (hθ : 0 ≤ θ) {n : ℕ}
    (hT : (mids T).card ≤ n + 1) :
    μ.real {ω | x ∈ goodOff M U θ (read · ω) k ⟨T, E⟩ Z .ends} ≤ 3 * (n + 1) * (3 / 2 * θ) := by
  classical
  set β := 3 / 2 * θ
  set Sk := (mids T).image (prefixOf x k * ·)
  set hiOf : List Bool → ℕ := fun a => k + (walk E a (x.toList.drop k)).length - 1
  set Se : List Bool → Finset (FreeMonoid α) := fun a =>
    (mids T).image (prefixOf x (hiOf a) * ·) ∪ (mids T).image (prefixOf x (hiOf a + 1) * ·)
  set good : FreeMonoid α → Prop := fun z => z ∉ Z ∧ ¬ BadAt U θ (M.eval z.toList)
  set St : List Bool → Set Ω := fun a => {ω | T.sift (read · ω) (prefixOf x k) = .inl a}
  have hsub : {ω | x ∈ goodOff M U θ (read · ω) k ⟨T, E⟩ Z .ends}
      ⊆ (⋃ z ∈ Sk.filter good, {ω | read z ω = .undecided})
        ∪ ⋃ a ∈ T.leaves.toFinset, ⋃ z ∈ (Se a).filter good,
          {ω | T.sift (read · ω) (prefixOf x k) = .inl a ∧ read z ω = .undecided} := by
    intro ω hω
    obtain ⟨z, hz, hg⟩ := hω
    obtain ⟨-, h⟩ := probe_ends_cases (read · ω) T E k x hz
    rcases h with h | ⟨a, ha, h⟩
    · obtain ⟨m, hm, rfl⟩ := sift_inr_mids (read · ω) T _ _ h
      exact Or.inl (Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_image_of_mem _ hm, hg⟩)
        (DTree.sift_inr (read · ω) T _ _ h))
    · refine Or.inr (Set.mem_biUnion (List.mem_toFinset.2 (DTree.sift_inl_mem _ T _ _ ha)) ?_)
      rcases h with h | h
      · obtain ⟨m, hm, rfl⟩ := sift_inr_mids (read · ω) T _ _ h
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_union_left _
          (Finset.mem_image_of_mem _ hm), hg⟩) ⟨ha, DTree.sift_inr (read · ω) T _ _ h⟩
      · obtain ⟨m, hm, rfl⟩ := sift_inr_mids (read · ω) T _ _ h
        exact Set.mem_biUnion (Finset.mem_filter.2 ⟨Finset.mem_union_right _
          (Finset.mem_image_of_mem _ hm), hg⟩) ⟨ha, DTree.sift_inr (read · ω) T _ _ h⟩
  have hβ : 0 ≤ β := by positivity
  have hcardk : ((Sk.filter good).card : ℝ) ≤ n + 1 := by
    have : (Sk.filter good).card ≤ n + 1 :=
      (Finset.card_filter_le _ _).trans (Finset.card_image_le.trans hT)
    exact_mod_cast this
  have hcarde : ∀ a, (((Se a).filter good).card : ℝ) ≤ 2 * (n + 1) := by
    intro a
    have : ((Se a).filter good).card ≤ 2 * (n + 1) :=
      (Finset.card_filter_le _ _).trans ((Finset.card_union_le _ _).trans
        (by have := (Finset.card_image_le (s := mids T) (f := (prefixOf x (hiOf a) * ·))).trans hT
            have := (Finset.card_image_le (s := mids T)
              (f := (prefixOf x (hiOf a + 1) * ·))).trans hT
            omega))
    exact_mod_cast this
  have hStm : ∀ a, MeasurableSet (St a) := by
    intro a
    have := measurable_of_strings hmeas Sk (fun rd => decide (T.sift rd (prefixOf x k) = .inl a))
      fun rd rd' h => by
        simp only [sift_congr_mids T _ fun m hm => h _ (Finset.mem_image_of_mem _ hm)]
    convert this (measurableSet_singleton true) using 1
    ext ω; simp [St]
  have hSt1 : ∑ a ∈ T.leaves.toFinset, μ.real (St a) ≤ 1 := by
    rw [← measureReal_biUnion_finset (fun a _ b _ hab => Set.disjoint_left.2 fun ω h1 h2 => by
      simp only [St, Set.mem_ofPred_eq] at h1 h2
      rw [h1] at h2; exact hab (Sum.inl.inj h2)) fun a _ => hStm a]
    exact measureReal_le_one
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top μ _)
    _ ≤ μ.real (⋃ z ∈ Sk.filter good, {ω | read z ω = .undecided})
        + μ.real (⋃ a ∈ T.leaves.toFinset, ⋃ z ∈ (Se a).filter good,
          {ω | T.sift (read · ω) (prefixOf x k) = .inl a ∧ read z ω = .undecided}) :=
        measureReal_union_le _ _
    _ ≤ ∑ _z ∈ Sk.filter good, β
        + ∑ a ∈ T.leaves.toFinset, ∑ _z ∈ (Se a).filter good, μ.real (St a) * β := by
        gcongr
        · refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun z hz => ?_)
          obtain ⟨-, hzZ, hg⟩ := Finset.mem_filter.1 hz
          exact hgood z hzZ hg
        · refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun a _ => ?_)
          refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun z hz => ?_)
          refine (start_und_le T k x hmeas hind a z).trans ?_
          obtain ⟨-, hzZ, hg⟩ := Finset.mem_filter.1 hz
          exact mul_le_mul_of_nonneg_left (hgood z hzZ hg) measureReal_nonneg
    _ = (Sk.filter good).card * β
        + ∑ a ∈ T.leaves.toFinset, ((Se a).filter good).card * (μ.real (St a) * β) := by
        simp only [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (n + 1) * β + ∑ a ∈ T.leaves.toFinset, 2 * (n + 1) * β * μ.real (St a) := by
        refine add_le_add (mul_le_mul_of_nonneg_right hcardk hβ)
          (Finset.sum_le_sum fun a _ => ?_)
        calc _ ≤ 2 * (n + 1) * (μ.real (St a) * β) :=
              mul_le_mul_of_nonneg_right (hcarde a) (mul_nonneg measureReal_nonneg hβ)
          _ = _ := by ring
    _ = (n + 1) * β + 2 * (n + 1) * β * ∑ a ∈ T.leaves.toFinset, μ.real (St a) := by
        rw [Finset.mul_sum]
    _ ≤ (n + 1) * β + 2 * (n + 1) * β * 1 := by gcongr
    _ = _ := by ring

variable (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (kd : ℕ) {L n : ℕ} {p₀ η : ℝ}
  (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
  (hpre : ∀ u, D.real {x | prefixOf x kd = u} ≤ p₀) (hT : T ∈ (classSet n : Finset (DTree α)))
include hL hp₀ hpre hT

omit x Z in
open scoped Classical in
/-- `mid_tail_one` off `headOf k kd T`, the tail at scale `p₀` bounding the length-`kd`
prefixes. -/
theorem mid_tail_off (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ)
    (hθ8 : 8 * ((L + 1) * (n + 1) * (3 / 2 * θ)) ≤ 1) :
    μ {ω | 2 * ((L + 1) * (n + 1) * (3 / 2 * θ))
          * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η
        < D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .middles)}
      ≤ ENNReal.ofReal (Real.exp (-(η / p₀))) := by
  set Z : Finset (FreeMonoid α) := headOf k kd T
  set θ₀ : ℝ := (L + 1) * (n + 1) * (3 / 2 * θ)
  have hθ₀ : 0 ≤ θ₀ := by positivity
  have h01 : ∀ (P : Prop) [Decidable P], 0 ≤ (if P then (1 : ℝ) else 0)
      ∧ (if P then (1 : ℝ) else 0) ≤ 1 := fun P _ => by split_ifs <;> norm_num
  set F : (FreeMonoid α → ARU) → Prop := fun rd =>
    2 * θ₀ * ∑ x ∈ upTo L, D.real {x}
        * (if searches (probeOrAgree rd T E k x) = true then 1 else 0) + η
      < ∑ x ∈ upTo L, D.real {x} * (if x ∈ goodOff M U θ rd k ⟨T, E⟩ Z .middles then 1 else 0)
  have hF : ∀ rd rd', (∀ z ∈ (upTo L).biUnion fun x => pot k x T, rd z = rd' z) →
      (F rd ↔ F rd') := by
    intro rd rd' h
    have h' : ∀ x ∈ upTo L, ∀ z ∈ pot k x T, rd z = rd' z :=
      fun x hx z hz => h z (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
    have e1 : ∑ x ∈ upTo L, D.real {x}
          * (if searches (probeOrAgree rd T E k x) = true then (1 : ℝ) else 0)
        = ∑ x ∈ upTo L, D.real {x}
          * (if searches (probeOrAgree rd' T E k x) = true then (1 : ℝ) else 0) :=
      Finset.sum_congr rfl fun x hx => by rw [probeOrAgree_congr rd T E k x (h' x hx)]
    have e2 : ∑ x ∈ upTo L, D.real {x}
          * (if x ∈ goodOff M U θ rd k ⟨T, E⟩ Z .middles then (1 : ℝ) else 0)
        = ∑ x ∈ upTo L, D.real {x}
          * (if x ∈ goodOff M U θ rd' k ⟨T, E⟩ Z .middles then (1 : ℝ) else 0) :=
      Finset.sum_congr rfl fun x hx => by
        simp only [goodOff_congr M U θ T E k x Z .middles (h' x hx)]
    simp only [F, e1, e2]
  have hev : {ω | 2 * θ₀ * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η
      < D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ Z .middles)} ⊆ {ω | F (read · ω)} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq] at hω
    rw [measureReal_eq_sum_upTo D hL, measureReal_eq_sum_upTo D hL] at hω
    simpa only [F, Set.mem_ofPred_eq] using hω
  refine (measure_mono hev).trans (measure_le_of_fixed hmeas hind Z _ F hF _ fun c => ?_)
  have hmeas' : ∀ z, Measurable fun ω => ovr Z (read · ω) c z := measurable_ovr hmeas Z c
  have hind' : iIndepFun (fun z ω => ovr Z (read · ω) c z) μ := iIndepFun_ovr hind Z c
  have hovr : ∀ x rd rd', (∀ z ∈ pot k x T \ Z, rd z = rd' z) →
      ∀ z ∈ pot k x T, ovr Z rd c z = ovr Z rd' c z := by
    intro x rd rd' h z hz
    by_cases hzZ : z ∈ Z
    · simp only [ovr, dif_pos hzZ]
    · simp only [ovr, dif_neg hzZ]; exact h z (Finset.mem_sdiff.2 ⟨hz, hzZ⟩)
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if x ∈ goodOff M U θ (ovr Z rd c) k ⟨T, E⟩ Z .middles then 1 else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if searches (probeOrAgree (ovr Z rd c) T E k x) = true then 1 else 0
  have hmean : ∀ x ∈ upTo L, ∫ ω, uf (read · ω) x ∂μ ≤ θ₀ * ∫ ω, tf (read · ω) x ∂μ := by
    intro x hx
    have e1 := integral_ite_strings (μ := μ) (fun z ω => ovr Z (read · ω) c z) hmeas' (pot k x T)
      (fun rd => x ∈ goodOff M U θ rd k ⟨T, E⟩ Z .middles) fun rd rd' h =>
        goodOff_congr M U θ T E k x Z .middles h
    have e2 := integral_ite_strings (μ := μ) (fun z ω => ovr Z (read · ω) c z) hmeas' (pot k x T)
      (fun rd => searches (probeOrAgree rd T E k x) = true) fun rd rd' h => by
        rw [probeOrAgree_congr rd T E k x h]
    have hgood : ∀ z ∉ Z, ¬ BadAt U θ (M.eval z.toList) →
        μ.real {ω | ovr Z (read · ω) c z = .undecided} ≤ 3 / 2 * θ := by
      intro z hz hg
      simp only [ovr, dif_neg hz]
      exact undecided_le_of_good M U θ hU hg
    have hcard : ((pot k x T).card : ℝ) ≤ (L + 1) * (n + 1) := by
      have h1 := card_pot k x T
      have h2 := mem_upTo.1 hx
      have : (pot k x T).card ≤ (L + 1) * (n + 1) :=
        h1.trans (Nat.mul_le_mul (by omega) (mids_le_of_class hT))
      exact_mod_cast this
    calc ∫ ω, uf (read · ω) x ∂μ
        = μ.real {ω | x ∈ goodOff M U θ (fun z => ovr Z (read · ω) c z) k ⟨T, E⟩ Z
            .middles} := e1
      _ ≤ (pot k x T).card * (3 / 2 * θ) * μ.real {ω | searches (probeOrAgree
            (fun z => ovr Z (read · ω) c z) T E k x) = true} :=
          mid_mean_off M U θ T E k x Z hmeas' hind' hgood hθ
      _ ≤ θ₀ * μ.real {ω | searches (probeOrAgree
            (fun z => ovr Z (read · ω) c z) T E k x) = true} := by
          gcongr
          exact mul_le_mul_of_nonneg_right hcard (by positivity)
      _ = θ₀ * ∫ ω, tf (read · ω) x ∂μ := by rw [← e2]
  have hTB := block_tail hmeas hind D hp₀ hpre hθ₀ hθ8 (fun x => pot k x T \ Z)
    (fun x z hz => pot_deep (Finset.mem_sdiff.1 hz).1 (Finset.mem_sdiff.1 hz).2) uf tf
    (fun x rd rd' h => by simp only [uf, goodOff_congr M U θ T E k x Z .middles (hovr x rd rd' h)])
    (fun x rd rd' h => by
      simp only [tf, probeOrAgree_congr (ovr Z rd c) T E k x (hovr x rd rd' h)])
    (fun rd x _ => h01 _) (fun rd x _ => h01 _) (η := η) (L := L) hmean
  refine le_trans (measure_mono fun ω hω => ?_) hTB
  simpa only [F, uf, tf, Set.mem_ofPred_eq] using hω

omit x Z in
open scoped Classical in
/-- `ends_tail_one` off `headOf k kd T`, the tail at scale `p₀` bounding the length-`kd`
prefixes. -/
theorem ends_tail_off (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ)
    (hθ8 : 8 * (3 * (n + 1) * (3 / 2 * θ)) ≤ 1) :
    μ {ω | 2 * (3 * (n + 1) * (3 / 2 * θ)) + η
        < D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .ends)}
      ≤ ENNReal.ofReal (Real.exp (-(η / p₀))) := by
  set Z : Finset (FreeMonoid α) := headOf k kd T
  set θ₀ : ℝ := 3 * (n + 1) * (3 / 2 * θ)
  have hθ₀ : 0 ≤ θ₀ := by positivity
  have hs : ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) ≤ 1 := by
    simp only [mul_one]
    rw [sum_measureReal_singleton]
    exact measureReal_le_one
  set F : (FreeMonoid α → ARU) → Prop := fun rd =>
    2 * θ₀ * ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) + η
      < ∑ x ∈ upTo L, D.real {x} * (if x ∈ goodOff M U θ rd k ⟨T, E⟩ Z .ends then 1 else 0)
  have hF : ∀ rd rd', (∀ z ∈ (upTo L).biUnion fun x => pot k x T, rd z = rd' z) →
      (F rd ↔ F rd') := by
    intro rd rd' h
    have h' : ∀ x ∈ upTo L, ∀ z ∈ pot k x T, rd z = rd' z :=
      fun x hx z hz => h z (Finset.mem_biUnion.2 ⟨x, hx, hz⟩)
    have e2 : ∑ x ∈ upTo L, D.real {x}
          * (if x ∈ goodOff M U θ rd k ⟨T, E⟩ Z .ends then (1 : ℝ) else 0)
        = ∑ x ∈ upTo L, D.real {x}
          * (if x ∈ goodOff M U θ rd' k ⟨T, E⟩ Z .ends then (1 : ℝ) else 0) :=
      Finset.sum_congr rfl fun x hx => by
        simp only [goodOff_congr M U θ T E k x Z .ends (h' x hx)]
    simp only [F, e2]
  have hev : {ω | 2 * θ₀ + η < D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ Z .ends)}
      ⊆ {ω | F (read · ω)} := by
    intro ω hω
    simp only [Set.mem_ofPred_eq] at hω ⊢
    rw [measureReal_eq_sum_upTo D hL] at hω
    have : 2 * θ₀ * ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) ≤ 2 * θ₀ := by nlinarith
    simp only [F]
    linarith
  refine (measure_mono hev).trans (measure_le_of_fixed hmeas hind Z _ F hF _ fun c => ?_)
  have hmeas' : ∀ z, Measurable fun ω => ovr Z (read · ω) c z := measurable_ovr hmeas Z c
  have hind' : iIndepFun (fun z ω => ovr Z (read · ω) c z) μ := iIndepFun_ovr hind Z c
  have hovr : ∀ x rd rd', (∀ z ∈ pot k x T \ Z, rd z = rd' z) →
      ∀ z ∈ pot k x T, ovr Z rd c z = ovr Z rd' c z := by
    intro x rd rd' h z hz
    by_cases hzZ : z ∈ Z
    · simp only [ovr, dif_pos hzZ]
    · simp only [ovr, dif_neg hzZ]; exact h z (Finset.mem_sdiff.2 ⟨hz, hzZ⟩)
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if x ∈ goodOff M U θ (ovr Z rd c) k ⟨T, E⟩ Z .ends then 1 else 0
  have hmean : ∀ x ∈ upTo L, ∫ ω, uf (read · ω) x ∂μ ≤ θ₀ * ∫ _ω, (1 : ℝ) ∂μ := by
    intro x _
    have e1 := integral_ite_strings (μ := μ) (fun z ω => ovr Z (read · ω) c z) hmeas' (pot k x T)
      (fun rd => x ∈ goodOff M U θ rd k ⟨T, E⟩ Z .ends) fun rd rd' h =>
        goodOff_congr M U θ T E k x Z .ends h
    have hgood : ∀ z ∉ Z, ¬ BadAt U θ (M.eval z.toList) →
        μ.real {ω | ovr Z (read · ω) c z = .undecided} ≤ 3 / 2 * θ := by
      intro z hz hg
      simp only [ovr, dif_neg hz]
      exact undecided_le_of_good M U θ hU hg
    simp only [integral_const, probReal_univ, smul_eq_mul, mul_one]
    calc ∫ ω, uf (read · ω) x ∂μ
        = μ.real {ω | x ∈ goodOff M U θ (fun z => ovr Z (read · ω) c z) k ⟨T, E⟩ Z
            .ends} := e1
      _ ≤ θ₀ := ends_mean_off M U θ T E k x Z hmeas' hind' hgood hθ (mids_le_of_class hT)
  have hTB := block_tail hmeas hind D hp₀ hpre hθ₀ hθ8 (fun x => pot k x T \ Z)
    (fun x z hz => pot_deep (Finset.mem_sdiff.1 hz).1 (Finset.mem_sdiff.1 hz).2) uf
    (fun _ _ => 1)
    (fun x rd rd' h => by simp only [uf, goodOff_congr M U θ T E k x Z .ends (hovr x rd rd' h)])
    (fun _ _ _ _ => rfl) (fun rd x _ => by simp only [uf]; split_ifs <;> norm_num)
    (fun rd x _ => by norm_num) (η := η) (L := L) hmean
  refine le_trans (measure_mono fun ω hω => ?_) hTB
  simpa only [F, uf, Set.mem_ofPred_eq] using hω

end One

/-! ## Over the trees and edge maps -/

section Union

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {read : FreeMonoid α → Ω → ARU} (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
  (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k kd : ℕ) {L n : ℕ} {p₀ η : ℝ}
  (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
  (hpre : ∀ u, D.real {x | prefixOf x kd = u} ≤ p₀)
include hmeas hind hU hθ hL hp₀ hpre

/-- `midNoise_le` off `headOf k kd T`, the prefix mass read at `kd`. -/
theorem midNoise_off (hθ8 : 8 * ((L + 1) * (n + 1) * (3 / 2 * θ)) ≤ 1) {θgpt : ℝ}
    (hθg : 2 * ((L + 1) * (n + 1) * (3 / 2 * θ)) ≤ θgpt) :
    μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)), ∀ E, EdgesInto T E →
        D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .middles)
          ≤ θgpt * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η}
      ≤ ENNReal.ofReal (tailRisk n (Fintype.card α) p₀ η) := by
  set B : DTree α → Edges α → Set Ω := fun T E => {ω | 2 * ((L + 1) * (n + 1) * (3 / 2 * θ))
      * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η
    < D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .middles)}
  refine le_trans (measure_mono fun ω hω => ?_) ((union_le n B fun T hT E _ =>
    mid_tail_off M U θ T E k D kd hL hp₀ hpre hT hmeas hind hU hθ hθ8).trans
    (le_of_eq (by unfold tailRisk; push_cast; ring_nf)))
  simp only [Set.mem_ofPred_eq, not_forall, not_le] at hω
  obtain ⟨T, hT, E, hE, hlt⟩ := hω
  obtain ⟨E', hE', hag⟩ := exists_edgeMaps hE
  refine Set.mem_biUnion hT (Set.mem_biUnion hE' ?_)
  have he : ∀ x, probeOrAgree (read · ω) T E k x = probeOrAgree (read · ω) T E' k x :=
    fun x => probe_econgr _ T k x hE hag
  have h1 : goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .middles
      = goodOff M U θ (read · ω) k ⟨T, E'⟩ (headOf k kd T) .middles := by
    ext x; simp only [goodOff, siteHarvest, Set.mem_ofPred_eq, he]
  have h2 : {x | searches (probeOrAgree (read · ω) T E k x) = true}
      = {x | searches (probeOrAgree (read · ω) T E' k x) = true} := by
    ext x; simp only [Set.mem_ofPred_eq, he]
  simp only [B, Set.mem_ofPred_eq]
  rw [← h1, ← h2]
  have : 0 ≤ D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} := measureReal_nonneg
  have := mul_le_mul_of_nonneg_right hθg this
  linarith

/-- `endsNoise_le` off `headOf k kd T`, the prefix mass read at `kd`. -/
theorem endsNoise_off (hθ8 : 8 * (3 * (n + 1) * (3 / 2 * θ)) ≤ 1) :
    μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)), ∀ E, EdgesInto T E →
        D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .ends)
          ≤ 9 * θ * (n + 1) + η}
      ≤ ENNReal.ofReal (tailRisk n (Fintype.card α) p₀ η) := by
  set B : DTree α → Edges α → Set Ω := fun T E => {ω | 2 * (3 * (n + 1) * (3 / 2 * θ)) + η
    < D.real (goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .ends)}
  refine le_trans (measure_mono fun ω hω => ?_) ((union_le n B fun T hT E _ =>
    ends_tail_off M U θ T E k D kd hL hp₀ hpre hT hmeas hind hU hθ hθ8).trans
    (le_of_eq (by unfold tailRisk; push_cast; ring_nf)))
  simp only [Set.mem_ofPred_eq, not_forall, not_le] at hω
  obtain ⟨T, hT, E, hE, hlt⟩ := hω
  obtain ⟨E', hE', hag⟩ := exists_edgeMaps hE
  refine Set.mem_biUnion hT (Set.mem_biUnion hE' ?_)
  have he : ∀ x, probeOrAgree (read · ω) T E k x = probeOrAgree (read · ω) T E' k x :=
    fun x => probe_econgr _ T k x hE hag
  have h1 : goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) .ends
      = goodOff M U θ (read · ω) k ⟨T, E'⟩ (headOf k kd T) .ends := by
    ext x; simp only [goodOff, siteHarvest, Set.mem_ofPred_eq, he]
  simp only [B, Set.mem_ofPred_eq]
  rw [← h1]
  linarith

end Union

end Random

end OrthoDFA
