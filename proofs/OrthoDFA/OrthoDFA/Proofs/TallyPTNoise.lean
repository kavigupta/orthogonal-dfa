import OrthoDFA.Proofs.TallyEdgeNoise

/-!
# The noise event's middles field

A probe searches only if every sift its walk check makes is decided, and so never reads a string
undecided there. Making one string undecided can then only stop a search, never start one, and
given that string undecided the search is that of the reads with it undecided, independent of its
own read. So a search stops at an undecided middle at a good read-state with chance at most the
positions' and midfixes' counts times `1.5θ` of the searches, and over the probes' length-`k`
prefixes the field's tail is bounded as the edge field's is.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Update

variable {cut : FreeMonoid α → Option Bool} {z : FreeMonoid α}

/-- `cut` with `z` undecided. -/
def cutWithout (cut : FreeMonoid α → Option Bool) (z : FreeMonoid α) : FreeMonoid α → Option Bool :=
  fun y => if y = z then none else cut y

theorem sift_without_inl {T : DTree α} {u : FreeMonoid α} {p : List Bool}
    (h : T.sift (cutWithout cut z) u = .inl p) : T.sift cut u = .inl p := by
  obtain ⟨L, hL, hall⟩ := (sift_inl_iff u T p).1 h
  refine (sift_inl_iff u T p).2 ⟨L, hL, fun mb hmb => ?_⟩
  have := hall mb hmb
  simp only [cutWithout] at this
  split_ifs at this
  exact this

theorem walkCheckBy_without_inr {T : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {r : List (List Bool) × ℕ} (h : walkCheckBy (cutWithout cut z) T edges k x = .inr r) :
    walkCheckBy cut T edges k x = .inr r := by
  unfold walkCheckBy kWalkBy at h ⊢
  rcases ha : T.sift (cutWithout cut z) (prefixOf x k) with p₀ | b
  swap
  · simp [ha] at h
  have hw : walkToBy (cutWithout cut z) T edges k x = walkToBy cut T edges k x := by
    funext j; simp only [walkToBy, ha, sift_without_inl ha]
  rw [sift_without_inl ha]
  simp only [ha] at h
  rcases hf : follow edges p₀ (x.toList.drop k) with ps | ⟨s, c, i⟩ <;> simp only [hf] at h ⊢
  · rcases hx : T.sift (cutWithout cut z) x with a | b <;> simp only [hx, reduceCtorEq] at h
    · rw [sift_without_inl hx]; exact h
  · rcases h1 : T.sift (cutWithout cut z) (prefixOf x (k + i + 1)) with q₁ | b <;>
      simp only [h1, reduceCtorEq] at h
    rw [sift_without_inl h1]
    rcases h2 : T.sift (cutWithout cut z) (prefixOf x (k + i)) with q₂ | b <;>
      simp only [h2, reduceCtorEq] at h
    rw [sift_without_inl h2]
    simp only []
    rw [← hw]; exact h

end Update

section PTExp

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- The strings a probe of `x` from `k` can read: its prefixes from `k` on, followed by the
tree's midfixes. -/
noncomputable def probeStrings (T : DTree α) (k : ℕ) (x : FreeMonoid α) : Finset (FreeMonoid α) :=
  (Finset.Icc k x.toList.length).biUnion fun j => T.midfixes.image (prefixOf x j * ·)

theorem prefAgreeK_of_strings {cut cut' : FreeMonoid α → Option Bool} {T : DTree α} {k : ℕ}
    {x : FreeMonoid α} (hkx : k ≤ x.toList.length)
    (h : ∀ z ∈ probeStrings T k x, cut z = cut' z) : PrefAgreeK cut cut' T x k := by
  intro j hj
  have hj' : ∃ j' ∈ Finset.Icc k x.toList.length, prefixOf x j = prefixOf x j' := by
    by_cases hjl : j ≤ x.toList.length
    · exact ⟨j, Finset.mem_Icc.2 ⟨hj, hjl⟩, rfl⟩
    · refine ⟨x.toList.length, Finset.mem_Icc.2 ⟨hkx, le_rfl⟩, ?_⟩
      simp only [prefixOf]
      rw [List.take_of_length_le (by omega), List.take_of_length_le le_rfl]
  obtain ⟨j', hj'm, hjj⟩ := hj'
  rw [hjj]
  exact sift_congr_mid T _ fun m hm =>
    h _ (Finset.mem_biUnion.2 ⟨j', hj'm, Finset.mem_image_of_mem _ hm⟩)

open scoped Classical in
/-- A probe searches with a given string undecided with no more than the chance it searches,
times that string's chance of being undecided. -/
theorem search_und_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (T : DTree α) (edges : Edges α) {k : ℕ} {x : FreeMonoid α} (hkx : k ≤ x.toList.length)
    (z : FreeMonoid α) :
    μ.real {ω | (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch
        ∧ read z ω = .undecided}
      ≤ μ.real {ω | (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch}
        * μ.real {ω | read z ω = .undecided} := by
  set R := (probeStrings T k x).erase z
  set S' : Set Ω := {ω | (probeBy (cutWithout (fun y => (read y ω).cut) z) T edges k x).IsSearch}
  have hS'sub : S' ⊆ {ω | (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch} := by
    intro ω hω
    obtain ⟨ps, hi, hw, hb⟩ := probeBy_search _ rfl hω
    have hw' := walkCheckBy_without_inr hw
    show (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch
    unfold probeBy
    rw [hw']
    exact bracketAt_isSearch _ _ _ _ _
  have heqset : {ω | (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch
      ∧ read z ω = .undecided} = S' ∩ {ω | read z ω = .undecided} := by
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_inter_iff, S']
    constructor
    · rintro ⟨h1, h2⟩
      refine ⟨?_, h2⟩
      have : cutWithout (fun y => (read y ω).cut) z = fun y => (read y ω).cut := by
        funext y; simp only [cutWithout]; split_ifs with hy
        · subst hy; rw [h2]; rfl
        · rfl
      rw [this]; exact h1
    · rintro ⟨h1, h2⟩
      have : cutWithout (fun y => (read y ω).cut) z = fun y => (read y ω).cut := by
        funext y; simp only [cutWithout]; split_ifs with hy
        · subst hy; rw [h2]; rfl
        · rfl
      rw [this] at h1; exact ⟨h1, h2⟩
  have hS' : S' = (fun ω (i : R) => read i ω) ⁻¹' {v | (probeBy (fun y => (rdOf R v y).cut)
      T edges k x).IsSearch} := by
    ext ω
    simp only [Set.mem_preimage, Set.mem_setOf_eq, S']
    rw [probeBy_congrK (edges := edges) (prefAgreeK_of_strings hkx fun y hy => ?_)]
    simp only [cutWithout, rdOf]
    by_cases hyz : y = z
    · rw [if_pos hyz, dif_neg (by simp [R, hyz])]; rfl
    · rw [if_neg hyz, dif_pos (Finset.mem_erase.2 ⟨hyz, hy⟩)]
  have hB : {ω | read z ω = .undecided}
      = (fun ω (i : ({z} : Finset (FreeMonoid α))) => read i ω) ⁻¹'
        {v | v ⟨z, Finset.mem_singleton_self z⟩ = .undecided} := by
    ext ω; simp
  have hdisj : Disjoint R {z} := Finset.disjoint_singleton_right.2 (by simp [R])
  have hI := hind.indepFun_finset R {z} hdisj hmeas
  have heq := hI.measure_inter_preimage_eq_mul
    {v : R → ARU | (probeBy (fun y => (rdOf R v y).cut) T edges k x).IsSearch}
    {v : ({z} : Finset (FreeMonoid α)) → ARU | v ⟨z, Finset.mem_singleton_self z⟩ = .undecided}
    (Set.to_countable _).measurableSet (Set.to_countable _).measurableSet
  rw [← hS', ← hB] at heq
  have hmul : μ.real (S' ∩ {ω | read z ω = .undecided})
      = μ.real S' * μ.real {ω | read z ω = .undecided} := by
    simp only [measureReal_def, heq, ENNReal.toReal_mul]
  rw [heqset, hmul]
  exact mul_le_mul_of_nonneg_right (measureReal_mono hS'sub) measureReal_nonneg

end PTExp

section PTMean

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
/-- At one probe, a search stops at an undecided middle at a good read-state with chance at most
its positions' and the midfixes' counts times `1.5θ` of its chance of searching. -/
theorem pt_mean_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (T : DTree α) (edges : Edges α) {k : ℕ} {x : FreeMonoid α} (hkx : k ≤ x.toList.length) :
    μ.real {ω | x ∈ G.ptAt (read · ω) k T edges G.Good}
      ≤ (x.toList.length + 1) * T.midfixes.card * (3 / 2 * G.θ)
        * μ.real {ω | x ∈ ReadModel.searchAt (read · ω) k T edges} := by
  set J := Finset.Icc k x.toList.length
  set Srch : Set Ω := {ω | (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch}
  have hsrch : {ω | x ∈ ReadModel.searchAt (read · ω) k T edges} = Srch := by
    ext ω; simp only [Set.mem_setOf_eq, Srch]; exact mem_searchAt
  have hsub : {ω | x ∈ G.ptAt (read · ω) k T edges G.Good}
      ⊆ ⋃ j ∈ J, ⋃ m ∈ T.midfixes.filter fun m => G.Good (G.M.eval (prefixOf x j * m).toList),
        Srch ∩ {ω | read (prefixOf x j * m) ω = .undecided} := by
    intro ω hω
    obtain ⟨b, hb, hg⟩ := hω
    unfold ptHarvBy at hb
    rcases ho : probeBy (fun y => (read y ω).cut) T edges k x with _ | _ | _ | j | _ | j | _ <;>
      rw [ho] at hb <;> simp only [reduceCtorEq] at hb
    all_goals first
      | (simp at hb)
      | skip
    all_goals
      rcases hs : T.sift (fun y => (read y ω).cut) (prefixOf x j) with q | b' <;> rw [hs] at hb <;>
        simp only [Sum.elim_inl, Sum.elim_inr, List.cons.injEq, reduceCtorEq, and_true,
          List.nil_eq] at hb
      subst hb
      obtain ⟨m, hm, rfl, hcm⟩ := sift_inr T (prefixOf x j) b' hs
      obtain ⟨ps, hi, -, hbr⟩ := probeBy_search _ ho trivial
      have hjk : k < j := by
        first
          | exact bracketAt_pair_gt _ _ _ _ _ _ hbr
          | exact bracketAt_triple_gt _ _ _ _ _ _ hbr
      set j' := min j x.toList.length
      have hpj : prefixOf x j = prefixOf x j' := by
        simp only [prefixOf, j']
        rcases le_total j x.toList.length with h | h
        · rw [min_eq_left h]
        · rw [min_eq_right h, List.take_of_length_le h, List.take_of_length_le le_rfl]
      have hj' : j' ∈ J := Finset.mem_Icc.2 ⟨le_min hjk.le hkx, min_le_right _ _⟩
      rw [hpj] at hg hcm
      refine Set.mem_biUnion hj' (Set.mem_biUnion (Finset.mem_filter.2 ⟨hm, hg⟩) ⟨?_, ?_⟩)
      · show (probeBy (fun y => (read y ω).cut) T edges k x).IsSearch
        rw [ho]; trivial
      · simp only [Set.mem_setOf_eq]
        revert hcm
        cases read (prefixOf x j' * m) ω <;> simp [ARU.cut]
  rw [hsrch]
  calc μ.real {ω | x ∈ G.ptAt (read · ω) k T edges G.Good}
      ≤ μ.real (⋃ j ∈ J, ⋃ m ∈ T.midfixes.filter fun m =>
          G.Good (G.M.eval (prefixOf x j * m).toList),
          Srch ∩ {ω | read (prefixOf x j * m) ω = .undecided}) :=
        measureReal_mono hsub (measure_ne_top μ _)
    _ ≤ ∑ j ∈ J, ∑ m ∈ T.midfixes.filter fun m => G.Good (G.M.eval (prefixOf x j * m).toList),
          μ.real (Srch ∩ {ω | read (prefixOf x j * m) ω = .undecided}) := by
        refine (measureReal_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun j _ => ?_)
        exact measureReal_biUnion_finset_le _ _
    _ ≤ ∑ j ∈ J, ∑ m ∈ T.midfixes.filter fun m => G.Good (G.M.eval (prefixOf x j * m).toList),
          μ.real Srch * (3 / 2 * G.θ) := by
        refine Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun m hm => ?_
        refine (search_und_le read hmeas hind T edges hkx _).trans ?_
        rw [hlaw]
        exact mul_le_mul_of_nonneg_left (Finset.mem_filter.1 hm).2 measureReal_nonneg
    _ ≤ ∑ _j ∈ J, ∑ _m ∈ T.midfixes, μ.real Srch * (3 / 2 * G.θ) := by
        refine Finset.sum_le_sum fun j _ => Finset.sum_le_sum_of_subset_of_nonneg
          (Finset.filter_subset _ _) fun _ _ _ => mul_nonneg measureReal_nonneg (by positivity)
    _ = J.card * T.midfixes.card * (3 / 2 * G.θ) * μ.real Srch := by
        simp only [Finset.sum_const, nsmul_eq_mul]; ring
    _ ≤ _ := by
        have hJ : (J.card : ℝ) ≤ x.toList.length + 1 := by
          simp only [J, Nat.card_Icc]; exact_mod_cast (by omega : _ ≤ x.toList.length + 1)
        gcongr

end PTMean

section PTOne

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
/-- At one tree and edge map, the probes whose search stops at an undecided middle at a good
read-state exceed twice `θ₀` of the probes that search by `η` with chance at most
`exp(−η / (2 p₀ (1 + 4θ₀)))`, where `θ₀ ≥ (L + 1)` times the midfixes' count times `1.5θ`. -/
theorem pt_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η θ₀ : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η)
    (T : DTree α) (hT : (L + 1) * T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀) (edges : Edges α) :
    μ {ω | 2 * θ₀ * D.real (ReadModel.searchAt (read · ω) k T edges) + η
        < D.real (G.ptAt (read · ω) k T edges G.Good)}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * p₀ * (1 + 4 * θ₀ * 1))))) := by
  have hθ₀ : 0 ≤ θ₀ := le_trans (by positivity) hT
  set c : ℝ := 1 / (2 * p₀ * (1 + 4 * θ₀ * 1))
  have hc : 0 < c := by positivity
  set X : Finset (FreeMonoid α) := lenRange k L
  set U : Finset (FreeMonoid α) := lenK k
  set Xu : FreeMonoid α → Finset (FreeMonoid α) := fun u => X.filter fun x => prefixOf x k = u
  set w : FreeMonoid α → ℝ := fun x => D.real {x}
  set Sb : FreeMonoid α → Finset (FreeMonoid α) := fun u => (Xu u).biUnion fun x =>
    (Finset.Icc k x.toList.length).biUnion fun j => T.midfixes.image (prefixOf x j * ·)
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if x ∈ G.ptAt rd k T edges G.Good then 1 else 0
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    if x ∈ ReadModel.searchAt rd k T edges then 1 else 0
  have hpa : ∀ u, ∀ x ∈ Xu u, ∀ ω, PrefAgreeK (fun z => (read z ω).cut)
      (fun z => (rdOf (Sb u) (fun i => read i ω) z).cut) T x k := by
    intro u x hx ω
    have hxX := (Finset.mem_filter.1 hx).1
    refine prefAgreeK_of_strings (mem_lenRange.1 hxX).1 fun z hz => ?_
    have : z ∈ Sb u := Finset.mem_biUnion.2 ⟨x, hx, hz⟩
    simp only [rdOf, dif_pos this]
  have huf : ∀ u, ∀ x ∈ Xu u, ∀ ω, uf (read · ω) x = uf (rdOf (Sb u) fun i => read i ω) x := by
    intro u x hx ω
    simp only [uf, ReadModel.ptAt, Set.mem_setOf_eq]
    rw [ptHarvBy_congrK (edges := edges) (hpa u x hx ω)]
  have htf : ∀ u, ∀ x ∈ Xu u, ∀ ω, tf (read · ω) x = tf (rdOf (Sb u) fun i => read i ω) x := by
    intro u x hx ω
    simp only [tf, ReadModel.searchAt, Set.mem_setOf_eq]
    rw [probeBy_congrK (edges := edges) (hpa u x hx ω)]
  have hb : ∀ (P : Prop) [Decidable P], 0 ≤ (if P then (1 : ℝ) else 0)
      ∧ (if P then (1 : ℝ) else 0) ≤ 1 := by
    intro P _; split_ifs <;> norm_num
  have hmapsto : ∀ x ∈ X, prefixOf x k ∈ U := by
    intro x hx
    refine mem_lenK.2 ?_
    simp [prefixOf, (mem_lenRange.1 hx).1]
  have hsplit : ∀ S : Set (FreeMonoid α), D.real S
      = ∑ u ∈ U, ∑ x ∈ Xu u, w x * if x ∈ S then 1 else 0 := by
    intro S
    have h1 : D.real S = ∫ x, (if x ∈ S then (1 : ℝ) else 0) ∂D := by
      rw [← integral_indicator_one (Set.to_countable S).measurableSet]
      congr 1
    rw [h1, integral_eq_sum_lenRange D hlen hk _ fun x _ => by
      rw [abs_of_nonneg (hb (x ∈ S)).1]; linarith [(hb (x ∈ S)).2]]
    exact (Finset.sum_fiberwise_of_maps_to hmapsto _).symm
  set Y : FreeMonoid α → Ω → ℝ := fun u ω =>
    ∑ x ∈ Xu u, w x * (uf (read · ω) x - 2 * θ₀ * tf (read · ω) x)
  have hev : {ω | 2 * θ₀ * D.real (ReadModel.searchAt (read · ω) k T edges) + η
      < D.real (G.ptAt (read · ω) k T edges G.Good)} ⊆ {ω | η < ∑ u ∈ U, Y u ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [hsplit, hsplit] at hω
    have : ∑ u ∈ U, Y u ω = ∑ u ∈ U, ∑ x ∈ Xu u, w x * uf (read · ω) x
        - 2 * θ₀ * ∑ u ∈ U, ∑ x ∈ Xu u, w x * tf (read · ω) x := by
      simp only [Y, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun x _ => ?_
      ring
    rw [this]
    simp only [uf, tf]
    linarith
  have hw0 : ∀ x, 0 ≤ w x := fun x => measureReal_nonneg
  have hwsum : ∀ u, ∑ x ∈ Xu u, w x ≤ p₀ := by
    intro u
    refine le_trans ?_ (hpmax u)
    rw [← measureReal_biUnion_finset (fun x _ x' _ h => Set.disjoint_singleton.2 h)
      (fun x _ => measurableSet_singleton x)]
    refine measureReal_mono ?_
    intro y hy
    simp only [Set.mem_iUnion, Set.mem_singleton_iff, exists_prop] at hy
    obtain ⟨x, hx, rfl⟩ := hy
    exact (Finset.mem_filter.1 hx).2
  have htuple : ∀ u, Measurable fun ω (i : Sb u) => read i ω := fun u =>
    measurable_pi_lambda _ fun i => hmeas i
  have hmuf : ∀ u, ∀ x ∈ Xu u, Measurable fun ω => uf (read · ω) x := by
    intro u x hx
    have : (fun ω => uf (read · ω) x)
        = (fun v => uf (rdOf (Sb u) v) x) ∘ fun ω (i : Sb u) => read i ω := by
      funext ω; exact huf u x hx ω
    rw [this]; exact (measurable_of_countable _).comp (htuple u)
  have hmtf : ∀ u, ∀ x ∈ Xu u, Measurable fun ω => tf (read · ω) x := by
    intro u x hx
    have : (fun ω => tf (read · ω) x)
        = (fun v => tf (rdOf (Sb u) v) x) ∘ fun ω (i : Sb u) => read i ω := by
      funext ω; exact htf u x hx ω
    rw [this]; exact (measurable_of_countable _).comp (htuple u)
  have hXuX : ∀ u, ∀ x ∈ Xu u, x ∈ X := fun u x hx => (Finset.mem_filter.1 hx).1
  have hblock : ∀ u, ∫⁻ ω, ENNReal.ofReal (Real.exp (c * Y u ω)) ∂μ ≤ 1 := by
    intro u
    set Uf : Ω → ℝ := fun ω => ∑ x ∈ Xu u, w x * uf (read · ω) x
    set Nf : Ω → ℝ := fun ω => ∑ x ∈ Xu u, w x * tf (read · ω) x
    have hY : ∀ ω, c * Y u ω = c * (Uf ω - 2 * θ₀ * Nf ω) := by
      intro ω
      have : Y u ω = Uf ω - 2 * θ₀ * Nf ω := by
        simp only [Y, Uf, Nf, Finset.mul_sum, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun x _ => by ring
      rw [this]
    simp_rw [hY]
    have hUm : Measurable Uf := Finset.measurable_sum _ fun x hx =>
      measurable_const.mul (hmuf u x hx)
    have hNm : Measurable Nf := Finset.measurable_sum _ fun x hx =>
      measurable_const.mul (hmtf u x hx)
    refine mgf_real Uf Nf hUm hNm hθ₀ hp₀.le zero_le_one hc ?_ ?_ ?_ ?_
    · simp only [c]; field_simp; linarith
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x _ => mul_nonneg (hw0 x) (hb _).1
      · calc Uf ω ≤ ∑ x ∈ Xu u, w x * 1 := Finset.sum_le_sum fun x _ =>
              mul_le_mul_of_nonneg_left (hb _).2 (hw0 x)
          _ ≤ p₀ := by simpa using hwsum u
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x _ => mul_nonneg (hw0 x) (hb _).1
      · calc Nf ω ≤ ∑ x ∈ Xu u, w x * 1 := Finset.sum_le_sum fun x _ =>
              mul_le_mul_of_nonneg_left (hb _).2 (hw0 x)
          _ ≤ p₀ * 1 := by simpa using hwsum u
    · have hint : ∀ (f : FreeMonoid α → Ω → ℝ), (∀ x ∈ Xu u, Measurable (f x)) →
          (∀ x ∈ Xu u, ∀ ω, |f x ω| ≤ 1) →
          ∫ ω, ∑ x ∈ Xu u, w x * f x ω ∂μ = ∑ x ∈ Xu u, w x * ∫ ω, f x ω ∂μ := by
        intro f hfm hfb
        rw [integral_finset_sum _ fun x hx => (Integrable.of_bound (hfm x hx).aestronglyMeasurable
          1 (Filter.Eventually.of_forall fun ω => by
            rw [Real.norm_eq_abs]; exact hfb x hx ω)).const_mul _]
        exact Finset.sum_congr rfl fun x _ => integral_const_mul _ _
      have habs : ∀ (P : Prop) [Decidable P], |(if P then (1 : ℝ) else 0)| ≤ 1 := by
        intro P _; split_ifs <;> norm_num
      rw [hint (fun x ω => uf (read · ω) x) (hmuf u) (fun x _ ω => habs _),
        hint (fun x ω => tf (read · ω) x) (hmtf u) (fun x _ ω => habs _), Finset.mul_sum]
      refine Finset.sum_le_sum fun x hx => ?_
      have hxX := hXuX u x hx
      have hset : ∀ (A : FreeMonoid α → Ω → Prop), (∀ x ∈ Xu u, Measurable fun ω =>
          (if A x ω then (1 : ℝ) else 0)) → ∫ ω, (if A x ω then (1 : ℝ) else 0) ∂μ
            = μ.real {ω | A x ω} := by
        intro A hAm
        have hms : MeasurableSet {ω | A x ω} := by
          have := (hAm x hx) (measurableSet_singleton (1 : ℝ))
          convert this using 1
          ext ω; simp only [Set.mem_setOf_eq, Set.mem_preimage, Set.mem_singleton_iff]
          split_ifs <;> simp_all
        rw [← integral_indicator_one hms]
        congr 1
      rw [hset (fun x ω => x ∈ G.ptAt (read · ω) k T edges G.Good) (hmuf u),
        hset (fun x ω => x ∈ ReadModel.searchAt (read · ω) k T edges) (hmtf u)]
      have h1 := pt_mean_le G read hmeas hind hlaw hθ T edges (mem_lenRange.1 hxX).1
        (x := x)
      have hxL : (x.toList.length : ℝ) + 1 ≤ L + 1 := by
        have h := (mem_lenRange.1 hxX).2
        have : (x.toList.length : ℝ) ≤ L := Nat.cast_le.2 h
        linarith
      have h2 : ((x.toList.length : ℝ) + 1) * T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀ :=
        le_trans (mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_right hxL
          (Nat.cast_nonneg _)) (by positivity)) hT
      have h3 := mul_le_mul_of_nonneg_right h2 (measureReal_nonneg (μ := μ)
        (s := {ω | x ∈ ReadModel.searchAt (read · ω) k T edges}))
      nlinarith [hw0 x]
  have hdisj : ∀ u ∈ U, ∀ u' ∈ U, u ≠ u' → Disjoint (Sb u) (Sb u') := by
    intro u hu u' hu' hne
    rw [Finset.disjoint_left]
    intro z h1 h2
    apply hne
    have key : ∀ v ∈ U, z ∈ Sb v → prefixOf z k = v := by
      intro v hv hz
      obtain ⟨x, hx, hz⟩ := Finset.mem_biUnion.1 hz
      obtain ⟨j, hj, hz⟩ := Finset.mem_biUnion.1 hz
      obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
      obtain ⟨hxX, hxv⟩ := Finset.mem_filter.1 hx
      rw [← hxv]
      have hjk := (Finset.mem_Icc.1 hj).1
      have hjx := (Finset.mem_Icc.1 hj).2
      apply FreeMonoid.toList.injective
      simp only [prefixOf, FreeMonoid.toList_ofList, FreeMonoid.toList_mul]
      rw [List.take_append_of_le_length (by simp; omega), List.take_take, min_eq_left hjk]
    rw [← key u hu h1, key u' hu' h2]
  set gb : ∀ u, (Sb u → ARU) → ENNReal := fun u v => ENNReal.ofReal (Real.exp (c *
    ∑ x ∈ Xu u, w x * (uf (rdOf (Sb u) v) x - 2 * θ₀ * tf (rdOf (Sb u) v) x)))
  have hgb : ∀ u, ∀ ω, gb u (fun i => read i ω) = ENNReal.ofReal (Real.exp (c * Y u ω)) := by
    intro u ω
    simp only [gb, Y]
    congr 3
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [huf u x hx ω, htf u x hx ω]
  have hprod : ∫⁻ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) ∂μ ≤ 1 := by
    have he : ∀ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω))
        = ∏ u ∈ U, gb u (fun i => read i ω) := by
      intro ω
      rw [Finset.mul_sum, Real.exp_sum, ENNReal.ofReal_prod_of_nonneg fun u _ =>
        (Real.exp_pos _).le]
      exact Finset.prod_congr rfl fun u _ => (hgb u ω).symm
    simp_rw [he]
    rw [lintegral_prod_blocks read hmeas hind Sb gb U hdisj]
    calc ∏ u ∈ U, ∫⁻ ω, gb u (fun i => read i ω) ∂μ ≤ ∏ _u ∈ U, (1 : ENNReal) := by
          refine Finset.prod_le_prod' fun u _ => ?_
          simp_rw [hgb]
          exact hblock u
      _ = 1 := Finset.prod_const_one
  have hYm : Measurable fun ω => ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) := by
    refine ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp (measurable_const.mul
      (Finset.measurable_sum _ fun u _ => Finset.measurable_sum _ fun x hx =>
        measurable_const.mul ((hmuf u x hx).sub (measurable_const.mul (hmtf u x hx))))))
  calc μ _ ≤ μ {ω | η < ∑ u ∈ U, Y u ω} := measure_mono hev
    _ ≤ μ {ω | ENNReal.ofReal (Real.exp (c * η))
          ≤ ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω))} := by
        refine measure_mono fun ω hω => ?_
        simp only [Set.mem_setOf_eq] at hω ⊢
        exact ENNReal.ofReal_le_ofReal (Real.exp_le_exp.2
          (mul_le_mul_of_nonneg_left hω.le hc.le))
    _ ≤ (∫⁻ ω, ENNReal.ofReal (Real.exp (c * ∑ u ∈ U, Y u ω)) ∂μ)
          / ENNReal.ofReal (Real.exp (c * η)) :=
        meas_ge_le_lintegral_div hYm.aemeasurable (by simp [Real.exp_pos]) ENNReal.ofReal_ne_top
    _ ≤ 1 / ENNReal.ofReal (Real.exp (c * η)) := by gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_div_of_pos (Real.exp_pos _), one_div,
          ← Real.exp_neg]
        congr 2
        simp only [c]
        field_simp

end PTOne

section PTUnion

variable {σ : Type*} [Fintype σ] (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

theorem goodPT_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k L S : ℕ) {p₀ η θgpt : ℝ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η)
    (hθg : 2 * ((L + 1) * (Fintype.card σ + S + 1) * (3 / 2 * G.θ)) ≤ θgpt) :
    μ {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
        D.real (G.ptAt (read · ω) k T edges G.Good)
          ≤ θgpt * D.real (ReadModel.searchAt (read · ω) k T edges) + η}
      ≤ ENNReal.ofReal ((classSet (Fintype.card σ + S) : Finset (DTree α)).card
        * (Fintype.card σ + S + 3) ^ ((Fintype.card σ + S + 2) * Fintype.card α)
        * Real.exp (-(η / (2 * p₀ * (1 + 4 * ((L + 1) * (Fintype.card σ + S + 1)
          * (3 / 2 * G.θ)) * 1))))) := by
  set n := Fintype.card σ + S
  set θ₀ : ℝ := (L + 1) * (n + 1) * (3 / 2 * G.θ)
  have hθ₀ : 0 ≤ θ₀ := by positivity
  have h2θ : 2 * θ₀ ≤ θgpt := by simp only [θ₀, n]; push_cast; linarith
  set ε : ℝ := Real.exp (-(η / (2 * p₀ * (1 + 4 * θ₀ * 1))))
  set B : DTree α → Edges α → Set Ω := fun T e => {ω |
    2 * θ₀ * D.real (ReadModel.searchAt (read · ω) k T e) + η
      < D.real (G.ptAt (read · ω) k T e G.Good)}
  have hsub : {ω | ¬ ∀ T edges, G.InClass S T → EdgesInto T edges →
      D.real (G.ptAt (read · ω) k T edges G.Good)
        ≤ θgpt * D.real (ReadModel.searchAt (read · ω) k T edges) + η}
      ⊆ ⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T, B T e := by
    intro ω hω
    simp only [Set.mem_setOf_eq, not_forall, not_le] at hω
    obtain ⟨T, edges, hT, he, hlt⟩ := hω
    obtain ⟨e', he'm, hagr, -⟩ := exists_edgeMaps he
    have hP : G.ptAt (read · ω) k T edges G.Good = G.ptAt (read · ω) k T e' G.Good := by
      ext x
      simp only [ReadModel.ptAt, Set.mem_setOf_eq]
      rw [ptHarvBy_econgr (cut := fun z => (read z ω).cut) (k := k) (x := x) he hagr]
    have hS : ReadModel.searchAt (read · ω) k T edges = ReadModel.searchAt (read · ω) k T e' := by
      ext x
      simp only [ReadModel.searchAt, Set.mem_setOf_eq]
      rw [probeBy_econgr (cut := fun z => (read z ω).cut) (k := k) (x := x) he hagr]
    refine Set.mem_biUnion (inClass_mem G hT) (Set.mem_biUnion he'm ?_)
    simp only [B, Set.mem_setOf_eq]
    rw [hP, hS] at hlt
    have : 0 ≤ D.real (ReadModel.searchAt (read · ω) k T e') := measureReal_nonneg
    nlinarith [mul_le_mul_of_nonneg_right h2θ this]
  have hone : ∀ T ∈ (classSet n : Finset (DTree α)), ∀ e, μ (B T e) ≤ ENNReal.ofReal ε := by
    intro T hT e
    have hm : (T.midfixes.card : ℝ) ≤ n + 1 := by
      have := midfixes_card T
      have := classSet_paths n T hT
      exact_mod_cast (by omega : T.midfixes.card ≤ n + 1)
    have hT' : ((L : ℝ) + 1) * T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀ :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hm (by positivity)) (by positivity)
    exact pt_one G read hmeas hind hlaw hθ D hlen hk hp₀ hpmax hη T hT' e
  calc μ _ ≤ μ (⋃ T ∈ (classSet n : Finset (DTree α)), ⋃ e ∈ edgeMaps T, B T e) :=
        measure_mono hsub
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)), ∑ e ∈ edgeMaps T, μ (B T e) := by
        refine (measure_biUnion_finset_le _ _).trans (Finset.sum_le_sum fun T _ => ?_)
        exact measure_biUnion_finset_le _ _
    _ ≤ ∑ T ∈ (classSet n : Finset (DTree α)),
          (((n + 3) ^ ((n + 2) * Fintype.card α) : ℕ) : ENNReal) * ENNReal.ofReal ε := by
        refine Finset.sum_le_sum fun T hT => ?_
        have hp := classSet_paths n T hT
        calc ∑ e ∈ edgeMaps T, μ (B T e) ≤ ∑ e ∈ edgeMaps T, ENNReal.ofReal ε :=
              Finset.sum_le_sum fun e _ => hone T hT e
          _ = ((edgeMaps T).card : ENNReal) * ENNReal.ofReal ε := by
              simp only [Finset.sum_const, nsmul_eq_mul]
          _ ≤ _ := by
              refine mul_le_mul' ?_ le_rfl
              have h1 : (edgeMaps T).card ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                calc (edgeMaps T).card
                    ≤ (T.paths.length + 1) ^ (T.paths.length * Fintype.card α) := edgeMaps_card T
                  _ ≤ (n + 3) ^ (T.paths.length * Fintype.card α) :=
                    Nat.pow_le_pow_left (by omega) _
                  _ ≤ (n + 3) ^ ((n + 2) * Fintype.card α) :=
                    Nat.pow_le_pow_right (by omega) (Nat.mul_le_mul_right _ hp)
              exact_mod_cast h1
    _ = _ := by
        rw [Finset.sum_const, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_natCast ((n + 3) ^ ((n + 2) * Fintype.card α)),
          ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity)]
        congr 1
        simp only [n, ε, θ₀]
        push_cast
        ring

end PTUnion

end OrthoDFA
