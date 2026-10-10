import OrthoDFA.Proofs.TallyCongr
import OrthoDFA.Proofs.TallyHarvest

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

section Mgf

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- A bounded increment whose mean is at most `θ` of a bounded count's, less twice that, has
exponential moment at most `1` at any scale `c` with `c B (1 + 4θR) ≤ 1/2`. -/
theorem mgf_real (U N : Ω → ℝ) (hUm : Measurable U) (hNm : Measurable N) {θ B R c : ℝ}
    (hθ : 0 ≤ θ) (hB : 0 ≤ B) (hR : 0 ≤ R) (hc : 0 < c) (hcB : c * B * (1 + 4 * θ * R) ≤ 1 / 2)
    (hU : ∀ ω, 0 ≤ U ω ∧ U ω ≤ B) (hN : ∀ ω, 0 ≤ N ω ∧ N ω ≤ B * R)
    (hmean : ∫ ω, U ω ∂μ ≤ θ * ∫ ω, N ω ∂μ) :
    ∫⁻ ω, ENNReal.ofReal (Real.exp (c * (U ω - 2 * θ * N ω))) ∂μ ≤ 1 := by
  have hcBθR : 0 ≤ c * B * (θ * R) := mul_nonneg (mul_nonneg hc.le hB) (mul_nonneg hθ hR)
  set A := c + c ^ 2 * B
  set B' := 2 * θ * c - 4 * θ ^ 2 * c ^ 2 * B * R
  set g : Ω → ℝ := fun ω => 1 + A * U ω - B' * N ω
  have hUi : Integrable U μ := Integrable.of_bound hUm.aestronglyMeasurable B
    (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hU ω).1]; exact (hU ω).2)
  have hNi : Integrable N μ := Integrable.of_bound hNm.aestronglyMeasurable (B * R)
    (Filter.Eventually.of_forall fun ω => by
      rw [Real.norm_eq_abs, abs_of_nonneg (hN ω).1]; exact (hN ω).2)
  have hgi : Integrable g μ := ((integrable_const 1).add (hUi.const_mul A)).sub (hNi.const_mul B')
  have hcBR : c * B ≤ 1 / 2 := by nlinarith
  have hpt : ∀ ω, Real.exp (c * (U ω - 2 * θ * N ω)) ≤ g ω := by
    intro ω
    obtain ⟨hU0, hU1⟩ := hU ω
    obtain ⟨hN0, hN1⟩ := hN ω
    set y := c * (U ω - 2 * θ * N ω)
    have hy1 : |y| ≤ 1 := by
      rw [abs_le]
      constructor
      · have : c * (2 * θ * N ω) ≤ 1 := by
          calc c * (2 * θ * N ω) ≤ c * (2 * θ * (B * R)) := by gcongr
            _ ≤ c * B * (1 + 4 * θ * R) := by nlinarith
            _ ≤ 1 := by linarith
        nlinarith [mul_nonneg hc.le hU0]
      · have : c * U ω ≤ 1 := by
          calc c * U ω ≤ c * B := by gcongr
            _ ≤ 1 := by linarith
        nlinarith [mul_nonneg hc.le (mul_nonneg hθ hN0)]
    have hexp := Real.abs_exp_sub_one_sub_id_le hy1
    have hsq : y ^ 2 ≤ c ^ 2 * (B * U ω + 4 * θ ^ 2 * (B * R) * N ω) := by
      simp only [y]
      rw [mul_pow]
      gcongr
      nlinarith [mul_le_mul_of_nonneg_left hU1 hU0, mul_le_mul_of_nonneg_left hN1 hN0,
        mul_nonneg hθ (mul_nonneg hU0 hN0)]
    have h1 : Real.exp y ≤ 1 + y + y ^ 2 := by
      have := (abs_le.1 hexp).2
      linarith
    have h2 : 1 + y + c ^ 2 * (B * U ω + 4 * θ ^ 2 * (B * R) * N ω) = g ω := by
      simp only [g, A, B', y]; ring
    linarith
  calc ∫⁻ ω, ENNReal.ofReal (Real.exp (c * (U ω - 2 * θ * N ω))) ∂μ
      ≤ ∫⁻ ω, ENNReal.ofReal (g ω) ∂μ := lintegral_mono fun ω => ENNReal.ofReal_le_ofReal (hpt ω)
    _ = ENNReal.ofReal (∫ ω, g ω ∂μ) := by
        rw [ofReal_integral_eq_lintegral_ofReal hgi]
        exact Filter.Eventually.of_forall fun ω => (Real.exp_pos _).le.trans (hpt ω)
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        have hint : ∫ ω, g ω ∂μ = 1 + A * ∫ ω, U ω ∂μ - B' * ∫ ω, N ω ∂μ := by
          have e1 := integral_sub (μ := μ) (f := fun ω => 1 + A * U ω)
            (g := fun ω => B' * N ω) ((integrable_const 1).add (hUi.const_mul A))
            (hNi.const_mul B')
          have e2 := integral_add (μ := μ) (f := fun _ => (1 : ℝ)) (g := fun ω => A * U ω)
            (integrable_const 1) (hUi.const_mul A)
          simp only [g]
          rw [e1, e2, integral_const_mul, integral_const_mul, integral_const]
          simp
        have hEN : 0 ≤ ∫ ω, N ω ∂μ := integral_nonneg fun ω => (hN ω).1
        have hA : 0 ≤ A := by positivity
        rw [hint]
        have h1 := mul_le_mul_of_nonneg_left hmean hA
        have hkey : A * θ - B' ≤ 0 := by
          simp only [A, B']
          nlinarith [mul_nonneg (mul_nonneg hc.le hc.le) hθ, mul_nonneg hθ hc.le]
        nlinarith

end Mgf

section One

variable {σ : Type*} (G : ReadModel α σ) {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)

/-- The probes of length from `k` to `L`. -/
noncomputable def lenRange (k L : ℕ) : Finset (FreeMonoid α) :=
  (Finset.Icc k L).biUnion lenK

theorem mem_lenRange {k L : ℕ} {x : FreeMonoid α} :
    x ∈ lenRange k L ↔ k ≤ x.toList.length ∧ x.toList.length ≤ L := by
  simp [lenRange, mem_lenK]

/-- The read of `z` from the reads of the block `S`, undecided off it. -/
def rdOf (S : Finset (FreeMonoid α)) (v : S → ARU) (z : FreeMonoid α) : ARU :=
  if h : z ∈ S then v ⟨z, h⟩ else .undecided

theorem integral_eq_sum_lenRange (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {k L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (f : FreeMonoid α → ℝ) (hf : ∀ x, x.toList.length ≤ L → |f x| ≤ L + 1) :
    ∫ x, f x ∂D = ∑ x ∈ lenRange k L, D.real {x} * f x := by
  have hae : ∀ᵐ x ∂D, x ∈ ((lenRange k L : Finset (FreeMonoid α)) : Set (FreeMonoid α)) := by
    filter_upwards [hlen, hk] with x h1 h2
    exact mem_lenRange.2 ⟨h2, h1⟩
  have hi : Integrable f D := Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable
    (L + 1) (by filter_upwards [hlen] with x hx; rw [Real.norm_eq_abs]; exact hf x hx)
  calc ∫ x, f x ∂D = ∫ x in ((lenRange k L : Finset (FreeMonoid α)) : Set (FreeMonoid α)),
        f x ∂D := by rw [Measure.restrict_eq_self_of_ae_mem hae]
    _ = _ := by rw [setIntegral_finset _ hi.integrableOn]; simp [smul_eq_mul]

open scoped Classical in
/-- At one tree, edge map and edge, the good read-states' undecided strings exceed twice the
midfixes' count times `1.5θ` of its positions by `η` with chance at most
`exp(−η / (2 L p₀ (1 + 8 θ₀)))`, `θ₀` that rate. -/
theorem goodEdge_one (hmeas : ∀ z, Measurable (read z)) (hind : iIndepFun read μ)
    (hlaw : ∀ z r, μ.real {ω | read z ω = r} = G.dist (G.M.eval z.toList) r) (hθ : 0 ≤ G.θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ} {p₀ η θ₀ : ℝ}
    (hL : 1 ≤ L) (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hk : ∀ᵐ x ∂D, k ≤ x.toList.length)
    (hp₀ : 0 < p₀) (hpmax : ∀ u, D.real {x | prefixOf x k = u} ≤ p₀) (hη : 0 ≤ η)
    (T : DTree α) (hT : T.midfixes.card * (3 / 2 * G.θ) ≤ θ₀) (edges : Edges α)
    (d : List Bool × α) :
    μ {ω | 2 * θ₀ * ∫ x, (travBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η
        < ∫ x, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂D}
      ≤ ENNReal.ofReal (Real.exp (-(η / (2 * L * p₀ * (1 + 4 * θ₀ * 2))))) := by
  have hθ₀ : 0 ≤ θ₀ := le_trans (by positivity) hT
  set c : ℝ := 1 / (2 * L * p₀ * (1 + 4 * θ₀ * 2))
  have hLr : (1 : ℝ) ≤ L := by exact_mod_cast hL
  have hc : 0 < c := by positivity
  set X : Finset (FreeMonoid α) := lenRange k L
  set U : Finset (FreeMonoid α) := lenK k
  set Xu : FreeMonoid α → Finset (FreeMonoid α) := fun u => X.filter fun x => prefixOf x k = u
  set w : FreeMonoid α → ℝ := fun x => D.real {x}
  set Sb : FreeMonoid α → Finset (FreeMonoid α) := fun u => (Xu u).biUnion fun x =>
    (Finset.Icc k x.toList.length).biUnion fun j => T.midfixes.image (prefixOf x j * ·)
  -- the probe's counts through a block's reads
  set uf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    (G.undecAt rd k T edges d G.Good x : ℝ)
  set tf : (FreeMonoid α → ARU) → FreeMonoid α → ℝ := fun rd x =>
    (travBy (fun z => (rd z).cut) T edges k x d : ℝ)
  have hpa : ∀ u, ∀ x ∈ Xu u, ∀ ω, PrefAgreeK (fun z => (read z ω).cut)
      (fun z => (rdOf (Sb u) (fun i => read i ω) z).cut) T x k := by
    intro u x hx ω j hj
    have hxX := (Finset.mem_filter.1 hx).1
    have hxl := (mem_lenRange.1 hxX).1
    have hj' : ∃ j' ∈ Finset.Icc k x.toList.length, prefixOf x j = prefixOf x j' := by
      by_cases hjl : j ≤ x.toList.length
      · exact ⟨j, Finset.mem_Icc.2 ⟨hj, hjl⟩, rfl⟩
      · refine ⟨x.toList.length, Finset.mem_Icc.2 ⟨hxl, le_rfl⟩, ?_⟩
        simp only [prefixOf]
        rw [List.take_of_length_le (by omega), List.take_of_length_le le_rfl]
    obtain ⟨j', hj'm, hjj⟩ := hj'
    rw [hjj]
    refine sift_congr_mid T _ fun m hm => ?_
    have : prefixOf x j' * m ∈ Sb u := Finset.mem_biUnion.2 ⟨x, hx, Finset.mem_biUnion.2
      ⟨j', hj'm, Finset.mem_image_of_mem _ hm⟩⟩
    simp only [rdOf, dif_pos this]
  have huf : ∀ u, ∀ x ∈ Xu u, ∀ ω, uf (read · ω) x = uf (rdOf (Sb u) fun i => read i ω) x := by
    intro u x hx ω
    simp only [uf, ReadModel.undecAt]
    rw [edgeHarvBy_congrK (edges := edges) (hpa u x hx ω)]
  have htf : ∀ u, ∀ x ∈ Xu u, ∀ ω, tf (read · ω) x = tf (rdOf (Sb u) fun i => read i ω) x := by
    intro u x hx ω
    simp only [tf]
    rw [travBy_congrK (edges := edges) (hpa u x hx ω)]
  have hb_uf : ∀ rd x, x.toList.length ≤ L → 0 ≤ uf rd x ∧ uf rd x ≤ L := by
    intro rd x hx
    refine ⟨Nat.cast_nonneg _, ?_⟩
    have h1 := List.length_filter_le (fun b => decide (G.Good (G.M.eval b.toList)))
      (edgeHarvBy (fun z => (rd z).cut) T edges k x d)
    have h2 := edgeHarvBy_length_le (fun z => (rd z).cut) (t := T) (edges := edges) (k := k)
      (x := x) d
    simp only [uf, ReadModel.undecAt]
    exact_mod_cast h1.trans (h2.trans hx)
  have hb_tf : ∀ rd x, x.toList.length ≤ L → 0 ≤ tf rd x ∧ tf rd x ≤ L + 1 := by
    intro rd x hx
    refine ⟨Nat.cast_nonneg _, ?_⟩
    have := travBy_le (fun z => (rd z).cut) (t := T) (edges := edges) (k := k) (x := x) d
    simp only [tf]
    exact_mod_cast this.trans (by omega)
  have hXl : ∀ x ∈ X, x.toList.length ≤ L := fun x hx => (mem_lenRange.1 hx).2
  have hmapsto : ∀ x ∈ X, prefixOf x k ∈ U := by
    intro x hx
    refine mem_lenK.2 ?_
    simp [prefixOf, (mem_lenRange.1 hx).1]
  -- the field's sides as sums over prefixes
  have hsplit : ∀ (f : FreeMonoid α → ℝ), (∀ x, x.toList.length ≤ L → |f x| ≤ L + 1) →
      ∫ x, f x ∂D = ∑ u ∈ U, ∑ x ∈ Xu u, w x * f x := by
    intro f hf
    rw [integral_eq_sum_lenRange D hlen hk f hf]
    exact (Finset.sum_fiberwise_of_maps_to hmapsto _).symm
  set Y : FreeMonoid α → Ω → ℝ := fun u ω =>
    ∑ x ∈ Xu u, w x * (uf (read · ω) x - 2 * θ₀ * tf (read · ω) x)
  have hev : {ω | 2 * θ₀ * ∫ x, (travBy (fun z => (read z ω).cut) T edges k x d : ℝ) ∂D + η
      < ∫ x, (G.undecAt (read · ω) k T edges d G.Good x : ℝ) ∂D}
      ⊆ {ω | η < ∑ u ∈ U, Y u ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have e1 := hsplit (uf (read · ω)) fun x hx => by
      rw [abs_of_nonneg (hb_uf (read · ω) x hx).1]; linarith [(hb_uf (read · ω) x hx).2]
    have e2 := hsplit (tf (read · ω)) fun x hx => by
      rw [abs_of_nonneg (hb_tf (read · ω) x hx).1]; exact (hb_tf (read · ω) x hx).2
    simp only [uf, tf] at e1 e2
    rw [e1, e2] at hω
    have : ∑ u ∈ U, Y u ω = ∑ u ∈ U, ∑ x ∈ Xu u, w x * uf (read · ω) x
        - 2 * θ₀ * ∑ u ∈ U, ∑ x ∈ Xu u, w x * tf (read · ω) x := by
      simp only [Y, Finset.mul_sum, ← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun u _ => Finset.sum_congr rfl fun x _ => ?_
      ring
    rw [this]
    simp only [uf, tf]
    linarith
  -- one prefix's block
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
    have hp₀L : 0 ≤ (L : ℝ) * p₀ := by positivity
    refine mgf_real Uf Nf hUm hNm hθ₀ hp₀L zero_le_two hc ?_ ?_ ?_ ?_
    · simp only [c]
      field_simp
      linarith
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x hx => mul_nonneg (hw0 x) (hb_uf (read · ω) x (hXl x (hXuX u x hx))).1
      · calc Uf ω ≤ ∑ x ∈ Xu u, w x * L := Finset.sum_le_sum fun x hx =>
              mul_le_mul_of_nonneg_left (hb_uf (read · ω) x (hXl x (hXuX u x hx))).2 (hw0 x)
          _ = (∑ x ∈ Xu u, w x) * L := (Finset.sum_mul _ _ _).symm
          _ ≤ p₀ * L := mul_le_mul_of_nonneg_right (hwsum u) (by positivity)
          _ = L * p₀ := mul_comm _ _
    · intro ω
      constructor
      · exact Finset.sum_nonneg fun x hx => mul_nonneg (hw0 x) (hb_tf (read · ω) x (hXl x (hXuX u x hx))).1
      · calc Nf ω ≤ ∑ x ∈ Xu u, w x * (L + 1) := Finset.sum_le_sum fun x hx =>
              mul_le_mul_of_nonneg_left (hb_tf (read · ω) x (hXl x (hXuX u x hx))).2 (hw0 x)
          _ = (∑ x ∈ Xu u, w x) * (L + 1) := (Finset.sum_mul _ _ _).symm
          _ ≤ p₀ * (L + 1) := mul_le_mul_of_nonneg_right (hwsum u) (by positivity)
          _ ≤ L * p₀ * 2 := by nlinarith
    · have hint : ∀ (f : FreeMonoid α → Ω → ℝ), (∀ x ∈ Xu u, Measurable (f x)) →
          (∀ x ∈ Xu u, ∀ ω, |f x ω| ≤ L + 1) →
          ∫ ω, ∑ x ∈ Xu u, w x * f x ω ∂μ = ∑ x ∈ Xu u, w x * ∫ ω, f x ω ∂μ := by
        intro f hfm hfb
        rw [integral_finset_sum _ fun x hx => (Integrable.of_bound (hfm x hx).aestronglyMeasurable
          (L + 1) (Filter.Eventually.of_forall fun ω => by
            rw [Real.norm_eq_abs]; exact hfb x hx ω)).const_mul _]
        exact Finset.sum_congr rfl fun x _ => integral_const_mul _ _
      rw [hint (fun x ω => uf (read · ω) x) (hmuf u) (fun x hx ω => by
          rw [abs_of_nonneg (hb_uf (read · ω) x (hXl x (hXuX u x hx))).1]
          linarith [(hb_uf (read · ω) x (hXl x (hXuX u x hx))).2]),
        hint (fun x ω => tf (read · ω) x) (hmtf u) (fun x hx ω => by
          rw [abs_of_nonneg (hb_tf (read · ω) x (hXl x (hXuX u x hx))).1]
          exact (hb_tf (read · ω) x (hXl x (hXuX u x hx))).2), Finset.mul_sum]
      refine Finset.sum_le_sum fun x _ => ?_
      have h1 := undec_mean_le G read hmeas hind hlaw hθ T edges k x d
      have h2 : 0 ≤ ∫ ω, tf (read · ω) x ∂μ := integral_nonneg fun ω => Nat.cast_nonneg _
      have h3 : (T.midfixes.card : ℝ) * (3 / 2 * G.θ) * ∫ ω, tf (read · ω) x ∂μ
          ≤ θ₀ * ∫ ω, tf (read · ω) x ∂μ := mul_le_mul_of_nonneg_right hT h2
      simp only [uf, tf] at h1 h3 ⊢
      nlinarith [hw0 x]
  -- the blocks are disjoint
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
  -- the exponential moment of the sum, and Markov
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

end One

end OrthoDFA
