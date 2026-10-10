import OrthoDFA.Proofs.TallyCongr

/-!
# The noise event's edge field

A probe's positions are charged to edges by its walk, which the cut fixes only through the sift of
its start. Given that sift's leaf, which only reads strings decided on the way there, every other
string's read is independent, so each position charged to an edge stops undecided at a good
read-state with chance at most the midfixes' count times `1.5θ`. Over the probes' length-`k`
prefixes, which read disjoint strings, the good read-states' undecided strings at an edge then
exceed twice that rate of its positions by `η` with chance at most `exp(−η/(2 L p₀ (1 + 8 θ₀)))`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section PathCond

/-- The reads that place a string at the leaf `p`: each node's midfix on the way, and its side. -/
def pathCond : DTree α → List Bool → Option (List (FreeMonoid α × Bool))
  | .leaf, [] => some []
  | .node m r _, false :: p => (pathCond r p).map ((m, false) :: ·)
  | .node m _ a, true :: p => (pathCond a p).map ((m, true) :: ·)
  | _, _ => none

omit [Fintype α] [DecidableEq α] in
theorem map_cons_eq_inl {s : List Bool ⊕ FreeMonoid α} {b : Bool} {q : List Bool} :
    s.map (b :: ·) id = .inl q ↔ ∃ p, q = b :: p ∧ s = .inl p := by
  rcases s with p | z <;> simp [eq_comm]

theorem sift_inl_iff {cut : FreeMonoid α → Option Bool} (u : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), T.sift cut u = .inl p ↔
      ∃ L, pathCond T p = some L ∧ ∀ mb ∈ L, cut (u * mb.1) = some mb.2
  | .leaf, p => by
    rcases p with _ | ⟨b, p⟩ <;> simp [DTree.sift, DTree.route, pathCond]
  | .node m r a, p => by
    have ihr := sift_inl_iff (cut := cut) u r
    have iha := sift_inl_iff (cut := cut) u a
    simp only [DTree.sift] at ihr iha ⊢
    simp only [DTree.route]
    rcases hc : cut (u * m) with _ | b
    · simp only [reduceCtorEq, false_iff, not_exists, not_and]
      intro L hL hall
      rcases p with _ | ⟨_ | _, p⟩ <;> simp only [pathCond, reduceCtorEq] at hL
      all_goals
        obtain ⟨L', -, rfl⟩ := Option.map_eq_some_iff.1 hL
        have := hall _ List.mem_cons_self
        rw [hc] at this; cases this
    · cases b
      · simp only [map_cons_eq_inl]
        constructor
        · rintro ⟨p', rfl, hp'⟩
          obtain ⟨L, hL, hall⟩ := (ihr p').1 hp'
          refine ⟨(m, false) :: L, by simp [pathCond, hL], ?_⟩
          intro mb hmb
          rcases List.mem_cons.1 hmb with rfl | h
          · exact hc
          · exact hall mb h
        · rintro ⟨L, hL, hall⟩
          rcases p with _ | ⟨_ | _, p⟩ <;> simp only [pathCond, reduceCtorEq] at hL
          · obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.1 hL
            exact ⟨p, rfl, (ihr p).2 ⟨L', hL', fun mb h => hall mb (List.mem_cons_of_mem _ h)⟩⟩
          · obtain ⟨L', -, rfl⟩ := Option.map_eq_some_iff.1 hL
            have := hall _ List.mem_cons_self
            rw [hc] at this; cases this
      · simp only [map_cons_eq_inl]
        constructor
        · rintro ⟨p', rfl, hp'⟩
          obtain ⟨L, hL, hall⟩ := (iha p').1 hp'
          refine ⟨(m, true) :: L, by simp [pathCond, hL], ?_⟩
          intro mb hmb
          rcases List.mem_cons.1 hmb with rfl | h
          · exact hc
          · exact hall mb h
        · rintro ⟨L, hL, hall⟩
          rcases p with _ | ⟨_ | _, p⟩ <;> simp only [pathCond, reduceCtorEq] at hL
          · obtain ⟨L', -, rfl⟩ := Option.map_eq_some_iff.1 hL
            have := hall _ List.mem_cons_self
            rw [hc] at this; cases this
          · obtain ⟨L', hL', rfl⟩ := Option.map_eq_some_iff.1 hL
            exact ⟨p, rfl, (iha p).2 ⟨L', hL', fun mb h => hall mb (List.mem_cons_of_mem _ h)⟩⟩

end PathCond

section Anchor

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
/-- Given that a string's sift ends at a leaf, which only fixes the strings decided on the way, any
string is undecided with no more than its own chance. -/
theorem anchor_und_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (T : DTree α) (u : FreeMonoid α) (p : List Bool) (z : FreeMonoid α) :
    μ.real {ω | T.sift (fun y => (read y ω).cut) u = .inl p ∧ read z ω = .undecided}
      ≤ μ.real {ω | T.sift (fun y => (read y ω).cut) u = .inl p}
        * μ.real {ω | read z ω = .undecided} := by
  have hnn : 0 ≤ μ.real {ω | T.sift (fun y => (read y ω).cut) u = .inl p}
      * μ.real {ω | read z ω = .undecided} := mul_nonneg measureReal_nonneg measureReal_nonneg
  rcases hL : pathCond T p with _ | L
  · have : {ω | T.sift (fun y => (read y ω).cut) u = .inl p ∧ read z ω = .undecided} = ∅ := by
      ext ω
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
      intro h
      obtain ⟨L, hL', -⟩ := (sift_inl_iff u T p).1 h
      rw [hL] at hL'; cases hL'
    rw [this, measureReal_empty]; exact hnn
  set R : Finset (FreeMonoid α) := (L.map fun mb => u * mb.1).toFinset
  have hmemR : ∀ mb ∈ L, u * mb.1 ∈ R := fun mb h =>
    List.mem_toFinset.2 (List.mem_map.2 ⟨mb, h, rfl⟩)
  have hA : {ω | T.sift (fun y => (read y ω).cut) u = .inl p}
      = (fun ω (i : R) => read i ω) ⁻¹'
        {v | ∀ mb (h : mb ∈ L), (v ⟨u * mb.1, hmemR mb h⟩).cut = some mb.2} := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_preimage, sift_inl_iff u T p, hL, Option.some.injEq,
      exists_eq_left']
  by_cases hz : z ∈ R
  · have : {ω | T.sift (fun y => (read y ω).cut) u = .inl p ∧ read z ω = .undecided} = ∅ := by
      ext ω
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
      intro h hu
      obtain ⟨L', hL', hall⟩ := (sift_inl_iff u T p).1 h
      rw [hL] at hL'
      cases hL'
      obtain ⟨mb, hmb, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hz)
      have := hall mb hmb
      rw [hu] at this
      cases this
    rw [this, measureReal_empty]; exact hnn
  have hdisj : Disjoint R {z} := Finset.disjoint_singleton_right.2 hz
  have hI := hind.indepFun_finset R {z} hdisj hmeas
  have hB : {ω | read z ω = .undecided}
      = (fun ω (i : ({z} : Finset (FreeMonoid α))) => read i ω) ⁻¹'
        {v | v ⟨z, Finset.mem_singleton_self z⟩ = .undecided} := by
    ext ω; simp
  have hmA : MeasurableSet {v : R → ARU | ∀ mb (h : mb ∈ L), (v ⟨u * mb.1, hmemR mb h⟩).cut
      = some mb.2} := (Set.to_countable _).measurableSet
  have hmB : MeasurableSet {v : ({z} : Finset (FreeMonoid α)) → ARU |
      v ⟨z, Finset.mem_singleton_self z⟩ = .undecided} := (Set.to_countable _).measurableSet
  have heq := hI.measure_inter_preimage_eq_mul _ _ hmA hmB
  rw [← hA, ← hB] at heq
  have hset : {ω | T.sift (fun y => (read y ω).cut) u = .inl p ∧ read z ω = .undecided}
      = {ω | T.sift (fun y => (read y ω).cut) u = .inl p} ∩ {ω | read z ω = .undecided} := rfl
  rw [hset, measureReal_def, heq, ENNReal.toReal_mul, ← measureReal_def, ← measureReal_def]

end Anchor

section From

/-- The walk from the leaf `p` at the start, to position `j`. -/
def walkFrom (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (p : List Bool) (j : ℕ) :
    List (List Bool) :=
  (follow edges p ((x.toList.drop k).take (j - k))).elim id fun _ => []

/-- The edge position `i` is charged to, the walk starting at the leaf `p`. -/
def posEdgeFrom (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (p : List Bool) (i : ℕ) :
    Option (List Bool × α) :=
  let at' := fun i => match (walkFrom edges k x p i).getLast?, x.toList[i]? with
    | some q, some c => some (q, c)
    | _, _ => none
  (at' i).orElse fun _ => at' (i - 1)

theorem posEdgeBy_of_inl {cut : FreeMonoid α → Option Bool} {T : DTree α} {edges : Edges α}
    {k : ℕ} {x : FreeMonoid α} {p : List Bool} (h : T.sift cut (prefixOf x k) = .inl p) (i : ℕ) :
    posEdgeBy cut T edges k x i = posEdgeFrom edges k x p i := by
  simp only [posEdgeBy, edgeAtBy, walkToBy, h, posEdgeFrom, walkFrom]
  rfl

theorem posEdgeBy_of_inr {cut : FreeMonoid α → Option Bool} {T : DTree α} {edges : Edges α}
    {k : ℕ} {x : FreeMonoid α} {b : FreeMonoid α} (h : T.sift cut (prefixOf x k) = .inr b)
    (i : ℕ) : posEdgeBy cut T edges k x i = none := by
  simp [posEdgeBy, edgeAtBy, walkToBy, h]

omit [Fintype α] [DecidableEq α] in
theorem length_filter_filterMap_le {β γ : Type*} (f : β → Option γ) (P : γ → Prop)
    [DecidablePred P] : ∀ L : List β, ((L.filterMap f).filter fun c => P c).length
      ≤ (L.map fun i => if ∃ c, f i = some c ∧ P c then 1 else 0).sum
  | [] => by simp
  | i :: L => by
    have := length_filter_filterMap_le f P L
    rcases h : f i with _ | c
    · simp only [List.filterMap_cons, h, List.map_cons, List.sum_cons]
      have : ¬ ∃ c, (none : Option γ) = some c ∧ P c := by simp
      rw [if_neg (by simpa [h] using this)]
      omega
    · simp only [List.filterMap_cons, h, List.map_cons, List.sum_cons]
      by_cases hc : P c
      · rw [List.filter_cons_of_pos (by simpa using hc), if_pos ⟨c, rfl, hc⟩]
        simp only [List.length_cons]; omega
      · rw [List.filter_cons_of_neg (by simpa using hc)]
        omega

omit [Fintype α] [DecidableEq α] in
theorem length_filter_range (P : ℕ → Prop) [DecidablePred P] :
    ∀ n, (((List.range n).filter fun i => P i).length : ℝ)
      = ∑ i ∈ Finset.range n, if P i then (1 : ℝ) else 0
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.filter_append, List.length_append, Finset.sum_range_succ,
      Nat.cast_add, length_filter_range P n]
    by_cases h : P n <;> simp [h]

end From

section EdgeExp

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

open scoped Classical in
theorem anchorSet_measurable (hmeas : ∀ z, Measurable (read z)) (T : DTree α) (u : FreeMonoid α)
    (p : List Bool) :
    MeasurableSet {ω | T.sift (fun y => (read y ω).cut) u = .inl p} := by
  set R : Finset (FreeMonoid α) := T.midfixes.image (u * ·)
  have hmA : Measurable fun ω (i : R) => read i ω := measurable_pi_lambda _ fun i => hmeas i
  have : {ω | T.sift (fun y => (read y ω).cut) u = .inl p}
      = (fun ω (i : R) => read i ω) ⁻¹' {v | T.sift (fun y => if h : y ∈ R then (v ⟨y, h⟩).cut
        else none) u = .inl p} := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_preimage]
    rw [sift_congr_mid (cut := fun y => if h : y ∈ R then ((fun ω (i : R) => read i ω) ω ⟨y, h⟩).cut
      else none) (cut' := fun y => (read y ω).cut) T u fun m hm => by
        simp only []
        rw [dif_pos (Finset.mem_image_of_mem (u * ·) hm)]]
  rw [this]
  exact hmA (Set.to_countable _).measurableSet

open scoped Classical in
/-- At one probe, the good read-states' undecided strings charged to an edge average at most the
midfixes' count times `1.5θ` of its positions. -/
theorem undec_mean_le (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (T : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) (d : List Bool × α) :
    ∫ ω, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂μ
      ≤ T.midfixes.card * (3 / 2 * G.θ)
        * ∫ ω, (travBy (fun y => (read y ω).cut) T edges k x d : ℝ) ∂μ := by
  set Pp := T.paths.toFinset
  set Rg := Finset.range (x.toList.length + 1)
  set A : List Bool → Set Ω := fun p => {ω | T.sift (fun y => (read y ω).cut) (prefixOf x k)
    = .inl p}
  set Z : ℕ → FreeMonoid α → FreeMonoid α := fun i m => prefixOf x i * m
  set c : List Bool → ℕ → Prop := fun p i => k < i ∧ posEdgeFrom edges k x p i = some d
  have hAm : ∀ p, MeasurableSet (A p) := fun p => anchorSet_measurable read hmeas T _ p
  have hUm : ∀ z, MeasurableSet {ω | read z ω = .undecided} := fun z =>
    hmeas z (Set.to_countable ({ARU.undecided} : Set ARU)).measurableSet
  -- the positions charged to `d`, by the start's leaf
  have hN : ∀ ω, (travBy (fun y => (read y ω).cut) T edges k x d : ℝ)
      = ∑ i ∈ Rg, ∑ p ∈ Pp, if c p i then (A p).indicator 1 ω else 0 := by
    intro ω
    simp only [travBy]
    rw [length_filter_range]
    refine Finset.sum_congr rfl fun i _ => ?_
    rcases ha : T.sift (fun y => (read y ω).cut) (prefixOf x k) with p₀ | b
    · have hp₀ : p₀ ∈ Pp := List.mem_toFinset.2 (DTree.sift_mem_paths _ _ _ ha)
      rw [posEdgeBy_of_inl ha, Finset.sum_eq_single p₀]
      · simp only [c, Set.indicator, A, Set.mem_setOf_eq, ha, Pi.one_apply]
        split_ifs <;> simp_all
      · intro p _ hne
        have : ω ∉ A p := by
          simp only [A, Set.mem_setOf_eq, ha, Sum.inl.injEq]; exact fun h => hne h.symm
        simp [Set.indicator_of_notMem this]
      · exact fun h => absurd hp₀ h
    · rw [posEdgeBy_of_inr ha]
      have : ∀ p, ω ∉ A p := fun p => by simp [A, ha]
      simp [Set.indicator_of_notMem (this _)]
  have hU : ∀ ω, (G.undecAt (read · ω) k T edges d G.Good x : ℝ)
      ≤ ∑ i ∈ Rg, ∑ p ∈ Pp, ∑ m ∈ T.midfixes,
        if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
          (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator 1 ω else 0 := by
    intro ω
    set cut : FreeMonoid α → Option Bool := fun y => (read y ω).cut
    set f : ℕ → Option (FreeMonoid α) := fun i =>
      if posEdgeBy cut T edges k x i = some d then (T.sift cut (prefixOf x i)).elim
        (fun _ => none) some else none
    set g : ℕ → ℕ := fun i =>
      if ∃ b, f i = some b ∧ G.Good (G.M.eval b.toList) then 1 else 0
    have h1 : G.undecAt (read · ω) k T edges d G.Good x
        ≤ ((siftsBy cut T edges k x).map g).sum := by
      unfold ReadModel.undecAt edgeHarvBy
      exact length_filter_filterMap_le f _ _
    have hnd : (siftsBy cut T edges k x).Nodup := by unfold siftsBy; exact List.nodup_dedup _
    have h2 : ((siftsBy cut T edges k x).map g).sum
        = ∑ i ∈ (siftsBy cut T edges k x).toFinset, g i := by
      rw [List.sum_toFinset _ hnd]
    have hsub : (siftsBy cut T edges k x).toFinset ⊆ Rg.filter (k < ·) := by
      intro i hi
      obtain ⟨h1, h2⟩ := siftsBy_range cut (List.mem_toFinset.1 hi)
      simp [Rg]; omega
    have h3 : ∑ i ∈ (siftsBy cut T edges k x).toFinset, (g i : ℝ)
        ≤ ∑ i ∈ Rg.filter (k < ·), (g i : ℝ) :=
      Finset.sum_le_sum_of_subset_of_nonneg hsub fun _ _ _ => Nat.cast_nonneg _
    have h4 : ∀ i ∈ Rg.filter (k < ·), (g i : ℝ) ≤ ∑ p ∈ Pp, ∑ m ∈ T.midfixes,
        if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
          (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator 1 ω else 0 := by
      intro i hi
      have hki : k < i := (Finset.mem_filter.1 hi).2
      by_cases hg : ∃ b, f i = some b ∧ G.Good (G.M.eval b.toList)
      · obtain ⟨b, hb, hgood⟩ := hg
        simp only [g, if_pos (⟨b, hb, hgood⟩ : ∃ b, f i = some b ∧ G.Good (G.M.eval b.toList)),
          Nat.cast_one]
        simp only [f] at hb
        split_ifs at hb with hpe
        rcases hs : T.sift cut (prefixOf x i) with q | b' <;> rw [hs] at hb
        · simp at hb
        simp only [Sum.elim_inr, Option.some.injEq] at hb
        subst hb
        obtain ⟨m, hm, rfl, hcm⟩ := sift_inr T (prefixOf x i) b' hs
        rcases ha : T.sift cut (prefixOf x k) with p₀ | b''
        swap
        · rw [posEdgeBy_of_inr ha] at hpe; cases hpe
        rw [posEdgeBy_of_inl ha] at hpe
        have hp₀ : p₀ ∈ Pp := List.mem_toFinset.2 (DTree.sift_mem_paths _ _ _ ha)
        have hterm : (1 : ℝ) ≤ if c p₀ i ∧ G.Good (G.M.eval (Z i m).toList) then
            (A p₀ ∩ {ω | read (Z i m) ω = .undecided}).indicator 1 ω else 0 := by
          have hmem : ω ∈ A p₀ ∩ {ω | read (Z i m) ω = .undecided} := by
            refine ⟨ha, ?_⟩
            simp only [Set.mem_setOf_eq, Z]
            simp only [cut] at hcm
            revert hcm
            cases read (prefixOf x i * m) ω <;> simp [ARU.cut]
          rw [if_pos ⟨⟨hki, hpe⟩, hgood⟩, Set.indicator_of_mem hmem, Pi.one_apply]
        have hnn : ∀ p m, (0 : ℝ) ≤ if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
            (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator 1 ω else 0 := by
          intro p m
          split_ifs
          · exact Set.indicator_nonneg (fun _ _ => zero_le_one) _
          · exact le_rfl
        refine hterm.trans ((Finset.single_le_sum (fun m _ => hnn p₀ m) hm).trans ?_)
        exact Finset.single_le_sum (f := fun p => ∑ m ∈ T.midfixes, _)
          (fun p _ => Finset.sum_nonneg fun m _ => hnn p m) hp₀
      · simp only [g, if_neg hg, Nat.cast_zero]
        exact Finset.sum_nonneg fun p _ => Finset.sum_nonneg fun m _ => by
          split_ifs
          · exact Set.indicator_nonneg (fun _ _ => zero_le_one) _
          · exact le_rfl
    have hnn' : ∀ i p m, (0 : ℝ) ≤ if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
        (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator 1 ω else 0 := by
      intro i p m
      split_ifs
      · exact Set.indicator_nonneg (fun _ _ => zero_le_one) _
      · exact le_rfl
    calc (G.undecAt (read · ω) k T edges d G.Good x : ℝ)
        ≤ ((siftsBy cut T edges k x).map g).sum := by exact_mod_cast h1
      _ = ∑ i ∈ (siftsBy cut T edges k x).toFinset, (g i : ℝ) := by rw [h2]; push_cast; rfl
      _ ≤ ∑ i ∈ Rg.filter (k < ·), (g i : ℝ) := h3
      _ ≤ ∑ i ∈ Rg.filter (k < ·), ∑ p ∈ Pp, ∑ m ∈ T.midfixes,
          if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
            (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator 1 ω else 0 :=
          Finset.sum_le_sum h4
      _ ≤ _ := Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
          fun i _ _ => Finset.sum_nonneg fun p _ => Finset.sum_nonneg fun m _ => hnn' i p m
  -- integrate
  have hint1 : ∀ (S : Set Ω), MeasurableSet S → Integrable (S.indicator (1 : Ω → ℝ)) μ :=
    fun S hS => (integrable_const (1 : ℝ)).indicator hS
  have hIU : Integrable (fun ω => ∑ i ∈ Rg, ∑ p ∈ Pp, ∑ m ∈ T.midfixes,
      if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
        (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator (1 : Ω → ℝ) ω else 0) μ := by
    refine integrable_finset_sum _ fun i _ => integrable_finset_sum _ fun p _ =>
      integrable_finset_sum _ fun m _ => ?_
    split_ifs
    · exact hint1 _ ((hAm p).inter (hUm _))
    · exact integrable_const 0
  have hIN : ∫ ω, (travBy (fun y => (read y ω).cut) T edges k x d : ℝ) ∂μ
      = ∑ i ∈ Rg, ∑ p ∈ Pp, if c p i then μ.real (A p) else 0 := by
    simp_rw [hN]
    rw [integral_finset_sum _ fun i _ => integrable_finset_sum _ fun p _ => by
      split_ifs; exacts [hint1 _ (hAm p), integrable_const 0]]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [integral_finset_sum _ fun p _ => by
      split_ifs; exacts [hint1 _ (hAm p), integrable_const 0]]
    refine Finset.sum_congr rfl fun p _ => ?_
    split_ifs
    · rw [integral_indicator_one (hAm p)]
    · simp
  calc ∫ ω, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂μ
      ≤ ∫ ω, (∑ i ∈ Rg, ∑ p ∈ Pp, ∑ m ∈ T.midfixes,
          if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
            (A p ∩ {ω | read (Z i m) ω = .undecided}).indicator (1 : Ω → ℝ) ω else 0) ∂μ :=
        integral_mono_of_nonneg (Filter.Eventually.of_forall fun ω => Nat.cast_nonneg _) hIU
          (Filter.Eventually.of_forall hU)
    _ = ∑ i ∈ Rg, ∑ p ∈ Pp, ∑ m ∈ T.midfixes,
          if c p i ∧ G.Good (G.M.eval (Z i m).toList) then
            μ.real (A p ∩ {ω | read (Z i m) ω = .undecided}) else 0 := by
        rw [integral_finset_sum _ fun i _ => integrable_finset_sum _ fun p _ =>
          integrable_finset_sum _ fun m _ => by
            split_ifs; exacts [hint1 _ ((hAm p).inter (hUm _)), integrable_const 0]]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [integral_finset_sum _ fun p _ => integrable_finset_sum _ fun m _ => by
            split_ifs; exacts [hint1 _ ((hAm p).inter (hUm _)), integrable_const 0]]
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [integral_finset_sum _ fun m _ => by
            split_ifs; exacts [hint1 _ ((hAm p).inter (hUm _)), integrable_const 0]]
        refine Finset.sum_congr rfl fun m _ => ?_
        split_ifs
        · rw [integral_indicator_one ((hAm p).inter (hUm _))]
        · simp
    _ ≤ ∑ i ∈ Rg, ∑ p ∈ Pp, ∑ _m ∈ T.midfixes,
          if c p i then μ.real (A p) * (3 / 2 * G.θ) else 0 := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun p _ =>
          Finset.sum_le_sum fun m _ => ?_
        by_cases h1 : c p i
        · rw [if_pos h1]
          split_ifs with h2
          · refine (anchor_und_le read hmeas hind T _ p _).trans ?_
            rw [hlaw]
            exact mul_le_mul_of_nonneg_left h2.2 measureReal_nonneg
          · exact mul_nonneg measureReal_nonneg (by positivity)
        · rw [if_neg h1, if_neg (fun h => h1 h.1)]
    _ = _ := by
        rw [hIN, Finset.mul_sum]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun p _ => ?_
        rw [Finset.sum_const, nsmul_eq_mul]
        split_ifs <;> ring

end EdgeExp

end OrthoDFA
