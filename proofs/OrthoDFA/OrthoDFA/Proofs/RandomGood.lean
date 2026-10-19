import OrthoDFA.Proofs.RandomHarvest

/-!
# What the probes harvest at good read-states

Fix a tree and an edge map. A search reads only decided sifts, so it and a string read undecided
have chance at most the product of theirs (`forced_le`): a probe stops at an undecided middle at a
good read-state with chance at most `(L + 1)(|Q| + 1) · 1.5θ` of its chance of searching. A probe
stops undecided at its start or end at a string the start sift reads, or, its start sifted to a
leaf, one the sift at the walk's end or just past it reads, and the sift to that leaf is likewise
independent of each later string: it stops so at a good read-state with chance at most
`3(|Q| + 1) · 1.5θ`. `block_tail` bounds both over the probes, and the union runs over `classSet`
and `edgeMaps`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ)
  {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  (read : FreeMonoid α → Ω → ARU)

omit [DecidableEq α] in
theorem integral_ite_strings (hmeas : ∀ z, Measurable (read z)) (S : Finset (FreeMonoid α))
    (P : (FreeMonoid α → ARU) → Prop) [DecidablePred P]
    (hP : ∀ rd rd', (∀ z ∈ S, rd z = rd' z) → (P rd ↔ P rd')) :
    ∫ ω, (if P (read · ω) then (1 : ℝ) else 0) ∂μ = μ.real {ω | P (read · ω)} := by
  have hm : MeasurableSet {ω | P (read · ω)} := by
    have := measurable_of_strings hmeas S (fun rd => decide (P rd)) fun rd rd' h => by
      simp only [hP rd rd' h]
    convert this (measurableSet_singleton true) using 1
    ext ω; simp
  rw [← integral_indicator_one hm]
  congr 1
  funext ω
  by_cases h : P (read · ω) <;> simp [h]

section One

variable {read} (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α)

theorem undecided_le_of_good
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) {z : FreeMonoid α}
    (hz : ¬ BadAt U θ (M.eval z.toList)) :
    μ.real {ω | read z ω = .undecided} ≤ 3 / 2 * θ := by
  rw [hU z]; simpa [BadAt] using hz

open scoped Classical in
theorem search_und_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (z : FreeMonoid α) :
    μ.real {ω | searches (probeOrAgree (read · ω) T E k x) = true ∧ read z ω = .undecided}
      ≤ μ.real {ω | searches (probeOrAgree (read · ω) T E k x) = true}
        * μ.real {ω | read z ω = .undecided} :=
  forced_le hmeas hind (pot k x T) (fun rd => searches (probeOrAgree rd T E k x) = true)
    (fun rd rd' h => by rw [probeOrAgree_congr rd T E k x h]) z
    fun rd h => searches_update rd T E k x z h

open scoped Classical in
theorem start_und_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (a : List Bool) (z : FreeMonoid α) :
    μ.real {ω | T.sift (read · ω) (prefixOf x k) = .inl a ∧ read z ω = .undecided}
      ≤ μ.real {ω | T.sift (read · ω) (prefixOf x k) = .inl a}
        * μ.real {ω | read z ω = .undecided} :=
  forced_le hmeas hind ((mids T).image (prefixOf x k * ·))
    (fun rd => T.sift rd (prefixOf x k) = .inl a)
    (fun rd rd' h => by
      rw [sift_congr_mids T _ fun m hm => h _ (Finset.mem_image_of_mem _ hm)]) z
    fun rd h => sift_update_inl rd z T _ _ h

/-- A probe stops at an undecided middle at a good read-state with chance at most its strings'
count times `1.5θ` of its chance of searching. -/
theorem mid_mean_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ) :
    μ.real {ω | x ∈ goodAt (read · ω) k ⟨T, E⟩ M U θ .middles}
      ≤ (pot k x T).card * (3 / 2 * θ)
        * μ.real {ω | searches (probeOrAgree (read · ω) T E k x) = true} := by
  classical
  set Srch := {ω | searches (probeOrAgree (read · ω) T E k x) = true}
  set Zs := (pot k x T).filter fun z => ¬ BadAt U θ (M.eval z.toList)
  have hsub : {ω | x ∈ goodAt (read · ω) k ⟨T, E⟩ M U θ .middles}
      ⊆ ⋃ z ∈ Zs, {ω | searches (probeOrAgree (read · ω) T E k x) = true
        ∧ read z ω = .undecided} := by
    intro ω hω
    obtain ⟨z, hz, hg⟩ := hω
    obtain ⟨hzp, hzu⟩ := probeOrAgree_pt (read · ω) T x E k z hz
    exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hzp, hg⟩)
      ⟨searches_of_ptStr (List.ne_nil_of_mem hz), hzu⟩
  calc _ ≤ _ := measureReal_mono hsub (measure_ne_top μ _)
    _ ≤ ∑ z ∈ Zs, μ.real {ω | searches (probeOrAgree (read · ω) T E k x) = true
          ∧ read z ω = .undecided} := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _z ∈ Zs, μ.real Srch * (3 / 2 * θ) := by
        refine Finset.sum_le_sum fun z hz => (search_und_le T E k x hmeas hind z).trans ?_
        exact mul_le_mul_of_nonneg_left
          (undecided_le_of_good M U θ hU (Finset.mem_filter.1 hz).2) measureReal_nonneg
    _ = Zs.card * (3 / 2 * θ) * μ.real Srch := by rw [Finset.sum_const, nsmul_eq_mul]; ring
    _ ≤ _ := by
        have : (Zs.card : ℝ) ≤ (pot k x T).card := by exact_mod_cast Finset.card_filter_le _ _
        gcongr

/-- A probe stops undecided at its start or end at a good read-state with chance at most three
times the midfixes' count times `1.5θ`. -/
theorem ends_mean_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ) {n : ℕ}
    (hT : (mids T).card ≤ n + 1) :
    μ.real {ω | x ∈ goodAt (read · ω) k ⟨T, E⟩ M U θ .ends} ≤ 3 * (n + 1) * (3 / 2 * θ) := by
  classical
  set β := 3 / 2 * θ
  set Sk := (mids T).image (prefixOf x k * ·)
  set hiOf : List Bool → ℕ := fun a => k + (walk E a (x.toList.drop k)).length - 1
  set Se : List Bool → Finset (FreeMonoid α) := fun a =>
    (mids T).image (prefixOf x (hiOf a) * ·) ∪ (mids T).image (prefixOf x (hiOf a + 1) * ·)
  set good : FreeMonoid α → Prop := fun z => ¬ BadAt U θ (M.eval z.toList)
  set St : List Bool → Set Ω := fun a => {ω | T.sift (read · ω) (prefixOf x k) = .inl a}
  have hsub : {ω | x ∈ goodAt (read · ω) k ⟨T, E⟩ M U θ .ends}
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
          exact undecided_le_of_good M U θ hU (Finset.mem_filter.1 hz).2
        · refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun a _ => ?_)
          refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun z hz => ?_)
          refine (start_und_le T k x hmeas hind a z).trans ?_
          exact mul_le_mul_of_nonneg_left
            (undecided_le_of_good M U θ hU (Finset.mem_filter.1 hz).2)
            measureReal_nonneg
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

open scoped Classical in
theorem goodAt_congr {rd rd' : FreeMonoid α → ARU} (site : Level3Site)
    (h : ∀ z ∈ pot k x T, rd z = rd' z) :
    x ∈ goodAt rd k ⟨T, E⟩ M U θ site ↔ x ∈ goodAt rd' k ⟨T, E⟩ M U θ site := by
  cases site <;> simp only [goodAt, siteHarvest, Set.mem_ofPred_eq, probeOrAgree_congr rd T E k x h]

variable (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {L n : ℕ} {p₀ η : ℝ}
  (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
  (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hT : (mids T).card ≤ n + 1)
include hL hp₀ hpre hT

omit x in
open scoped Classical in
/-- At one tree and edge map, the probes stopping at an undecided middle at a good read-state
exceed twice `θ₀ = (L + 1)(|Q| + 1) · 1.5θ` of the searches by `η` with chance at most
`exp(−η / (2 p₀ (1 + 4θ₀)))`. -/
theorem mid_tail_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ) :
    μ {ω | 2 * ((L + 1) * (n + 1) * (3 / 2 * θ))
          * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η
        < D.real (goodAt (read · ω) k ⟨T, E⟩ M U θ .middles)}
      ≤ ENNReal.ofReal
        (Real.exp (-(η / (2 * p₀ * (1 + 4 * ((L + 1) * (n + 1) * (3 / 2 * θ))))))) := by
  set θ₀ : ℝ := (L + 1) * (n + 1) * (3 / 2 * θ)
  have hθ₀ : 0 ≤ θ₀ := by positivity
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if x ∈ goodAt rd k ⟨T, E⟩ M U θ .middles then 1 else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if searches (probeOrAgree rd T E k x) = true then 1 else 0
  have h01 : ∀ (P : Prop) [Decidable P], 0 ≤ (if P then (1 : ℝ) else 0)
      ∧ (if P then (1 : ℝ) else 0) ≤ 1 := fun P _ => by split_ifs <;> norm_num
  have hTB := block_tail hmeas hind D hp₀ hpre hθ₀ one_pos zero_le_one (fun x => pot k x T)
    (fun x z hz => (pot_pre hz).2) uf tf
    (fun x rd rd' h => by simp only [uf, goodAt_congr M U θ T E k x .middles h])
    (fun x rd rd' h => by simp only [tf, probeOrAgree_congr rd T E k x h])
    (fun rd x _ => h01 _) (fun rd x _ => by rw [mul_one]; exact h01 _) (η := η) (L := L)
    fun x hx => by
      simp only [uf, tf]
      rw [integral_ite_strings read hmeas (pot k x T)
          (fun rd => x ∈ goodAt rd k ⟨T, E⟩ M U θ .middles) fun rd rd' h =>
          goodAt_congr M U θ T E k x .middles h,
        integral_ite_strings read hmeas (pot k x T)
          (fun rd => searches (probeOrAgree rd T E k x) = true) fun rd rd' h => by
          rw [probeOrAgree_congr rd T E k x h]]
      refine (mid_mean_le M U θ T E k x hmeas hind hU hθ).trans ?_
      gcongr
      have h1 := card_pot k x T
      have h2 := mem_upTo.1 hx
      have : (pot k x T).card ≤ (L + 1) * (n + 1) := h1.trans (Nat.mul_le_mul (by omega) hT)
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast this) (by positivity)
  refine le_trans (measure_mono fun ω hω => ?_) (hTB.trans (le_of_eq (by ring_nf)))
  simp only [Set.mem_ofPred_eq] at hω ⊢
  rw [measureReal_eq_sum_upTo D hL, measureReal_eq_sum_upTo D hL] at hω
  simpa only [uf, tf, Set.mem_ofPred_eq] using hω

omit x in
open scoped Classical in
/-- At one tree and edge map, the probes stopping undecided at their start or end at a good
read-state exceed twice `3(|Q| + 1) · 1.5θ` by `η` with chance at most
`exp(−η / (2 p₀ (1 + 4 · 3(|Q| + 1) · 1.5θ)))`. -/
theorem ends_tail_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ) :
    μ {ω | 2 * (3 * (n + 1) * (3 / 2 * θ)) + η
        < D.real (goodAt (read · ω) k ⟨T, E⟩ M U θ .ends)}
      ≤ ENNReal.ofReal
        (Real.exp (-(η / (2 * p₀ * (1 + 4 * (3 * (n + 1) * (3 / 2 * θ))))))) := by
  set θ₀ : ℝ := 3 * (n + 1) * (3 / 2 * θ)
  have hθ₀ : 0 ≤ θ₀ := by positivity
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if x ∈ goodAt rd k ⟨T, E⟩ M U θ .ends then 1 else 0
  have hTB := block_tail hmeas hind D hp₀ hpre hθ₀ one_pos zero_le_one (fun x => pot k x T)
    (fun x z hz => (pot_pre hz).2) uf (fun _ _ => 1)
    (fun x rd rd' h => by simp only [uf, goodAt_congr M U θ T E k x .ends h])
    (fun _ _ _ _ => rfl) (fun rd x _ => by simp only [uf]; split_ifs <;> norm_num)
    (fun rd x _ => by norm_num) (η := η) (L := L)
    fun x _ => by
      simp only [uf]
      rw [integral_ite_strings read hmeas (pot k x T)
          (fun rd => x ∈ goodAt rd k ⟨T, E⟩ M U θ .ends) fun rd rd' h =>
          goodAt_congr M U θ T E k x .ends h]
      simp only [integral_const, probReal_univ, smul_eq_mul, mul_one]
      exact ends_mean_le M U θ T E k x hmeas hind hU hθ hT
  refine le_trans (measure_mono fun ω hω => ?_) (hTB.trans (le_of_eq (by ring_nf)))
  simp only [Set.mem_ofPred_eq] at hω ⊢
  rw [measureReal_eq_sum_upTo D hL] at hω
  have hs : ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) ≤ 1 := by
    simp only [mul_one]
    rw [sum_measureReal_singleton]
    exact measureReal_le_one
  have : 2 * θ₀ * ∑ x ∈ upTo L, D.real {x} * (1 : ℝ) ≤ 2 * θ₀ := by
    nlinarith
  linarith

end One

/-! ## Over the trees and edge maps -/

section Union

theorem union_le (n : ℕ) (B : DTree α → Edges α → Set Ω) {b : ℝ}
    (hone : ∀ T ∈ (classSet n : Finset (DTree α)), ∀ E ∈ edgeMaps T,
      μ (B T E) ≤ ENNReal.ofReal b) :
    μ (⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ E ∈ edgeMaps T, B T E)
      ≤ ENNReal.ofReal (((1 + (n + 1) ^ 3 * Fintype.card α) ^ n
        * (n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) * b) := by
  calc _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)), ∑ E ∈ edgeMaps T, μ (B T E) := by
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun T _ => ?_)
        exact measure_biUnion_finset_le _ _
    _ ≤ ∑ _T ∈ (classSet n : Finset (DTree α)),
          (((n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) : ENNReal) * ENNReal.ofReal b := by
        refine Finset.sum_le_sum fun T hT => ?_
        have hp := classSet_leaves n T hT
        calc ∑ E ∈ edgeMaps T, μ (B T E) ≤ ∑ _E ∈ edgeMaps T, ENNReal.ofReal b :=
              Finset.sum_le_sum fun E hE => hone T hT E hE
          _ = ((edgeMaps T).card : ENNReal) * ENNReal.ofReal b := by
              simp only [Finset.sum_const, nsmul_eq_mul]
          _ ≤ _ := by
              refine mul_le_mul' ?_ le_rfl
              have h1 : (edgeMaps T).card ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                calc (edgeMaps T).card
                    ≤ (T.leaves.length + 1) ^ (T.leaves.length * Fintype.card α) :=
                      edgeMaps_card T
                  _ ≤ (n + 3) ^ (T.leaves.length * Fintype.card α) :=
                    Nat.pow_le_pow_left (by omega) _
                  _ ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                    Nat.pow_le_pow_right (by omega) (Nat.mul_le_mul_right _ hp)
              exact_mod_cast h1
    _ = ((classSet n : Finset (DTree α)).card : ENNReal)
          * ((((n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) : ENNReal) * ENNReal.ofReal b) := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ (((1 + (n + 1) ^ 3 * Fintype.card α) ^ n : ℕ) : ENNReal)
          * ((((n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) : ENNReal) * ENNReal.ofReal b) := by
        gcongr
        exact_mod_cast classSet_card n
    _ = _ := by
        rw [← mul_assoc, ← Nat.cast_mul, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_mul (Nat.cast_nonneg _)]

omit [DecidableEq α] in
theorem mids_le_of_class {n : ℕ} {T : DTree α} (hT : T ∈ (classSet n : Finset (DTree α))) :
    (mids T).card ≤ n + 1 := by
  have := mids_card T; have := classSet_leaves n T hT; omega

variable {read} (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
  (hU : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) {L n : ℕ} {p₀ η : ℝ}
  (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) (hp₀ : 0 < p₀)
  (hpre : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀)
include hmeas hind hU hθ hL hp₀ hpre

/-- Over every tree of `classSet n` and edge map into its leaves, the probes stopping at an
undecided middle at a good read-state are at most `θgpt` of the searches plus `η`, but for
`tailRisk`. -/
theorem midNoise_le {θgpt : ℝ} (hθg : 2 * ((L + 1) * (n + 1) * (3 / 2 * θ)) ≤ θgpt) :
    μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)), ∀ E, EdgesInto T E →
        D.real (goodAt (read · ω) k ⟨T, E⟩ M U θ .middles)
          ≤ θgpt * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η}
      ≤ ENNReal.ofReal (tailRisk n (Fintype.card α) p₀ ((L + 1) * (n + 1) * (3 / 2 * θ)) η) := by
  set B : DTree α → Edges α → Set Ω := fun T E => {ω | 2 * ((L + 1) * (n + 1) * (3 / 2 * θ))
      * D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} + η
    < D.real (goodAt (read · ω) k ⟨T, E⟩ M U θ .middles)}
  refine le_trans (measure_mono fun ω hω => ?_) ((union_le n B fun T hT E _ =>
    mid_tail_one M U θ T E k D hL hp₀ hpre (mids_le_of_class hT) hmeas hind hU hθ).trans
    (le_of_eq (by unfold tailRisk; push_cast; ring_nf)))
  simp only [Set.mem_ofPred_eq, not_forall, not_le] at hω
  obtain ⟨T, hT, E, hE, hlt⟩ := hω
  obtain ⟨E', hE', hag⟩ := exists_edgeMaps hE
  refine Set.mem_biUnion hT (Set.mem_biUnion hE' ?_)
  have he : ∀ x, probeOrAgree (read · ω) T E k x = probeOrAgree (read · ω) T E' k x :=
    fun x => probe_econgr _ T k x hE hag
  have h1 : goodAt (read · ω) k ⟨T, E⟩ M U θ .middles
      = goodAt (read · ω) k ⟨T, E'⟩ M U θ .middles := by
    ext x; simp only [goodAt, siteHarvest, Set.mem_ofPred_eq, he]
  have h2 : {x | searches (probeOrAgree (read · ω) T E k x) = true}
      = {x | searches (probeOrAgree (read · ω) T E' k x) = true} := by
    ext x; simp only [Set.mem_ofPred_eq, he]
  simp only [B, Set.mem_ofPred_eq]
  rw [← h1, ← h2]
  have : 0 ≤ D.real {x | searches (probeOrAgree (read · ω) T E k x) = true} := measureReal_nonneg
  have := mul_le_mul_of_nonneg_right hθg this
  linarith

/-- Over every tree of `classSet n` and edge map into its leaves, the probes stopping undecided
at their start or end at a good read-state are at most `9(n + 1)θ + η`, but for `tailRisk`. -/
theorem endsNoise_le :
    μ {ω | ¬ ∀ T ∈ (classSet n : Finset (DTree α)), ∀ E, EdgesInto T E →
        D.real (goodAt (read · ω) k ⟨T, E⟩ M U θ .ends) ≤ 9 * θ * (n + 1) + η}
      ≤ ENNReal.ofReal (tailRisk n (Fintype.card α) p₀ (3 * (n + 1) * (3 / 2 * θ)) η) := by
  set B : DTree α → Edges α → Set Ω := fun T E => {ω | 2 * (3 * (n + 1) * (3 / 2 * θ)) + η
    < D.real (goodAt (read · ω) k ⟨T, E⟩ M U θ .ends)}
  refine le_trans (measure_mono fun ω hω => ?_) ((union_le n B fun T hT E _ =>
    ends_tail_one M U θ T E k D hL hp₀ hpre (mids_le_of_class hT) hmeas hind hU hθ).trans
    (le_of_eq (by unfold tailRisk; push_cast; ring_nf)))
  simp only [Set.mem_ofPred_eq, not_forall, not_le] at hω
  obtain ⟨T, hT, E, hE, hlt⟩ := hω
  obtain ⟨E', hE', hag⟩ := exists_edgeMaps hE
  refine Set.mem_biUnion hT (Set.mem_biUnion hE' ?_)
  have he : ∀ x, probeOrAgree (read · ω) T E k x = probeOrAgree (read · ω) T E' k x :=
    fun x => probe_econgr _ T k x hE hag
  have h1 : goodAt (read · ω) k ⟨T, E⟩ M U θ .ends
      = goodAt (read · ω) k ⟨T, E'⟩ M U θ .ends := by
    ext x; simp only [goodAt, siteHarvest, Set.mem_ofPred_eq, he]
  simp only [B, Set.mem_ofPred_eq]
  rw [← h1]
  linarith

end Union

end Random

end OrthoDFA
