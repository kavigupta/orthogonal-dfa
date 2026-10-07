import OrthoDFA.Stage

/-!
# The harvest's replay is spread

Every string a `Walked` replay reads extends a prefix of the probe at least as long as its earliest anchor
(`take_of_mem_replayReads`), so reading a fixed `t` needs the probe to start as `t` does, up to
some point at or past the anchor.

A draw the gate counts against the hypothesis is one whose replay from the start harvests
(`replayHarvest_of_gateDisagrees`): a read the band decides is on the side of the middle of the
band, so a replay that decides every read follows the gate's own reading, and parts from the walk
where the gate does.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

theorem siftQueries_prefix (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) {x q : FreeMonoid α}, q ∈ t.siftQueries cut x → ∃ m, q = x * m
  | .leaf, x, q, h => by simp [siftQueries] at h
  | .node m r a, x, q, h => by
    simp only [siftQueries, List.mem_cons] at h
    rcases h with rfl | h
    · exact ⟨m, rfl⟩
    · split at h
      · simp at h
      · exact siftQueries_prefix cut a h
      · exact siftQueries_prefix cut r h

end DTree

theorem bisectionSifts_bounds (R : CutReads α) (t : DTree α) (w : FreeMonoid α)
    (walk : ℕ → List Bool) :
    ∀ (fuel lo hi j : ℕ), j ∈ bisectionSifts R t w walk fuel lo hi → lo < j ∧ j < hi
  | 0, lo, hi, j, h => by simp [bisectionSifts] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bisectionSifts] at h
    split_ifs at h with hlh
    · simp only [List.mem_cons] at h
      rcases h with rfl | h
      · omega
      · split at h
        · simp at h
        · split_ifs at h
          · have := bisectionSifts_bounds R t w walk fuel _ hi j h
            omega
          · have := bisectionSifts_bounds R t w walk fuel lo _ j h
            omega
    · simp at h

theorem anchoredWalkFrom_start (R : CutReads α) (t : DTree α)
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) (w : FreeMonoid α) (e : ℕ)
    {start : ℕ} {walk : List (List Bool)}
    (h : anchoredWalkFrom R t edges w e = some (start, walk)) :
    e ≤ start ∧ start ≤ w.toList.length := by
  unfold anchoredWalkFrom at h
  split at h
  · simp at h
  · rename_i i p hfind
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, -⟩ := h
    obtain ⟨a, ha, hfa⟩ := List.exists_of_findSome?_eq_some hfind
    have hai : a = i := by
      split at hfa
      · simp only [Option.some.injEq, Prod.mk.injEq] at hfa
        exact hfa.1
      · simp at hfa
    subst hai
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at ha
    omega

theorem replaySifts_bounds (R : CutReads α) (s : PassState α) (w : FreeMonoid α) (e : ℕ)
    {i : ℕ} (h : i ∈ replaySifts R s w e) : e ≤ i ∧ i ≤ w.toList.length := by
  simp only [replaySifts, List.mem_append] at h
  rcases h with (h | h) | h
  · have := (List.takeWhile_sublist _).subset h
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at this
    omega
  · have := List.mem_of_mem_drop (List.mem_of_mem_take h)
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at this
    omega
  · split at h
    · simp at h
    · rename_i start walk hw
      have hs := anchoredWalkFrom_start R _ _ _ _ hw
      simp only [List.mem_cons] at h
      rcases h with rfl | h
      · omega
      · split at h
        · simp at h
        · split_ifs at h
          · simp at h
          · have := bisectionSifts_bounds R _ _ _ _ _ _ _ h
            omega

/-- A string a replay reads starts as the probe does, up to some point at or past the earliest
anchor. -/
theorem take_of_mem_replayReads (R : CutReads α) (s : PassState α) (w : FreeMonoid α) (e : ℕ)
    {x : FreeMonoid α} (h : x ∈ replayReads R s w e) :
    ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
  obtain ⟨i, hi, q, hq, v, -, rfl⟩ := h
  obtain ⟨hei, hin⟩ := replaySifts_bounds R s w e hi
  obtain ⟨m, rfl⟩ := DTree.siftQueries_prefix R.cut _ hq
  have hlen : (prefixOf w i).toList.length = i := by
    simp [prefixOf, List.length_take, hin]
  refine ⟨i, hei, ?_, ?_⟩
  · simp only [FreeMonoid.toList_mul, List.length_append, hlen]
    omega
  · simp only [FreeMonoid.toList_mul, List.append_assoc]
    rw [List.take_left' hlen]
    simp [prefixOf]

instance (L : ℕ) : IsFiniteMeasure (uniformAnchor L) := by
  constructor
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp [uniformAnchor]
  · simp only [uniformAnchor, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
      measure_univ, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one, smul_eq_mul]
    rw [ENNReal.inv_mul_cancel (by exact_mod_cast hL.ne') (by simp)]
    exact ENNReal.one_lt_top

instance (L : ℕ) : IsFiniteMeasure (anchorLaw L) := by
  constructor
  simp only [anchorLaw, Measure.add_apply, Measure.smul_apply, smul_eq_mul, measure_univ]
  exact ENNReal.add_lt_top.2 ⟨ENNReal.mul_lt_top (by simp) (by simp),
    ENNReal.mul_lt_top (by simp) (measure_lt_top _ _)⟩

theorem uniformAnchor_real_le (L i : ℕ) :
    (uniformAnchor L).real {e | e ≤ i} = ((min (i + 1) L : ℕ) : ℝ) / L := by
  have hcount : ∑ k ∈ Finset.range L, (Measure.dirac k : Measure ℕ) {e | e ≤ i}
      = ((min (i + 1) L : ℕ) : ℝ≥0∞) := by
    simp only [Measure.dirac_apply, Set.indicator_apply, Set.mem_setOf_eq, Pi.one_apply]
    rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
    have hf : ((Finset.range L).filter fun k => k ≤ i) = Finset.range (min (i + 1) L) := by
      ext k
      simp only [Finset.mem_filter, Finset.mem_range]
      omega
    rw [hf, Finset.card_range]
  simp only [measureReal_def, uniformAnchor, Measure.smul_apply, Measure.coe_finsetSum,
    Finset.sum_apply, smul_eq_mul, hcount, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast]
  ring

theorem anchorLaw_real_le (L i : ℕ) :
    (anchorLaw L).real {e | e ≤ i} = 1 / 2 + ((min (i + 1) L : ℕ) : ℝ) / (2 * L) := by
  have hu := uniformAnchor_real_le L i
  rw [measureReal_def] at hu
  have hd : (Measure.dirac 0 : Measure ℕ) {e | e ≤ i} = 1 :=
    Measure.dirac_apply_of_mem (by simp)
  rw [measureReal_def, anchorLaw, Measure.add_apply, Measure.smul_apply, Measure.smul_apply,
    smul_eq_mul, smul_eq_mul, hd, ENNReal.toReal_add (by simp)
      (ENNReal.mul_ne_top (by simp) (measure_ne_top _ _)), ENNReal.toReal_mul,
    ENNReal.toReal_mul, hu]
  simp only [ENNReal.toReal_inv, ENNReal.toReal_ofNat, ENNReal.toReal_one]
  ring

/-- A replay anchored no earlier than a point drawn from `A` reads `t` with chance at most
`∑_{i ≤ |t|} A(e ≤ i) · D(the probe's first i letters are t's)`. -/
theorem replay_spread_of (R : CutReads α) (s : PassState α) (D : Measure (FreeMonoid α))
    [IsFiniteMeasure D] (A : Measure ℕ) [IsFiniteMeasure A] (t : FreeMonoid α) :
    (D.prod A).real {q | t ∈ replayReads R s q.1 q.2}
      ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
          A.real {e | e ≤ i} * D.real {p | p.toList.take i = t.toList.take i} := by
  have hsub : {q : FreeMonoid α × ℕ | t ∈ replayReads R s q.1 q.2} ⊆
      ⋃ i ∈ Finset.range (t.toList.length + 1),
        {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i} := by
    rintro ⟨p, e⟩ hq
    obtain ⟨i, hei, hit, htake⟩ := take_of_mem_replayReads R s p e hq
    simp only [Set.mem_iUnion, Finset.mem_range, Set.mem_prod, Set.mem_setOf_eq]
    exact ⟨i, by omega, htake, hei⟩
  calc (D.prod A).real {q | t ∈ replayReads R s q.1 q.2}
      ≤ (D.prod A).real (⋃ i ∈ Finset.range (t.toList.length + 1),
          {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ i ∈ Finset.range (t.toList.length + 1), (D.prod A).real
          ({p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i}) :=
        measureReal_biUnion_finset_le _ _
    _ = _ := by
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [measureReal_prod_prod]
        ring

theorem replay_spread_holds : ReplaySpread := by
  intro α _ _ R s D _ L t
  refine (replay_spread_of R s D (anchorLaw L) t).trans_eq (Finset.sum_congr rfl fun i _ => ?_)
  rw [anchorLaw_real_le]

instance [Nonempty α] (L : ℕ) : IsProbabilityMeasure (uniformStrings α L) := by
  constructor
  have hc : (Fintype.card α : ℝ≥0∞) ^ L ≠ 0 := pow_ne_zero _ (by simp)
  simp only [uniformStrings, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
    measure_univ, Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin,
    nsmul_eq_mul, mul_one, smul_eq_mul, Nat.cast_pow]
  exact ENNReal.inv_mul_cancel hc (by simp)

omit [DecidableEq α] in
open scoped Classical in
theorem uniformStrings_real (L : ℕ) (S : Set (FreeMonoid α)) :
    (uniformStrings α L).real S
      = ((Finset.univ.filter fun f : Fin L → α => FreeMonoid.ofList (List.ofFn f) ∈ S).card : ℝ)
        / (Fintype.card α : ℝ) ^ L := by
  have hsum : ∑ f : Fin L → α, (Measure.dirac (FreeMonoid.ofList (List.ofFn f))) S
      = ((Finset.univ.filter fun f : Fin L → α => FreeMonoid.ofList (List.ofFn f) ∈ S).card
          : ℝ≥0∞) := by
    simp only [Measure.dirac_apply', MeasurableSpace.measurableSet_top, Set.indicator_apply,
      Pi.one_apply]
    rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
  simp only [measureReal_def, uniformStrings, Measure.smul_apply, Measure.coe_finsetSum,
    Finset.sum_apply, smul_eq_mul, hsum, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_pow, ENNReal.toReal_natCast]
  rw [div_eq_inv_mul]

/-- At most `|α|^(L - i)` length-`L` strings start with a given `i` letters. -/
theorem card_take_le (L i : ℕ) (u : List α) (hu : u.length = i) :
    (Finset.univ.filter fun f : Fin L → α => (List.ofFn f).take i = u).card
      ≤ Fintype.card α ^ (L - i) := by
  classical
  by_cases hiL : i ≤ L
  · rw [← Fintype.card_fin (L - i), ← Fintype.card_fun]
    refine Finset.card_le_card_of_injOn
      (fun f j => f ⟨i + j.val, by omega⟩) (fun _ _ => Finset.mem_univ _) ?_
    intro f hf g hg hfg
    simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_setOf_eq] at hf hg
    funext j
    by_cases hj : j.val < i
    · have h1 : (List.ofFn f).take i = (List.ofFn g).take i := hf.trans hg.symm
      have := congrArg (fun l => l[j.val]?) h1
      simp only [List.getElem?_take, hj, if_true, List.getElem?_ofFn] at this
      simpa [List.ofFnNthVal, j.isLt] using this
    · have := congrFun hfg ⟨j.val - i, by omega⟩
      simp only at this
      have hj' : i + (j.val - i) = j.val := by omega
      simpa [hj'] using this
  · have : (Finset.univ.filter fun f : Fin L → α => (List.ofFn f).take i = u) = ∅ := by
      ext f
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, iff_false]
      intro h
      have := congrArg List.length h
      simp [List.length_take] at this
      omega
    simp [this]

theorem uniformStrings_take_le [Nonempty α] (L i : ℕ) (u : List α) (hu : u.length = i) :
    (uniformStrings α L).real {p | p.toList.take i = u} ≤ ((Fintype.card α : ℝ)⁻¹) ^ i := by
  classical
  have hc : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [uniformStrings_real]
  simp only [Set.mem_setOf_eq, FreeMonoid.toList_ofList]
  have hcard := card_take_le (α := α) L i u hu
  by_cases hiL : i ≤ L
  · calc ((Finset.univ.filter fun f : Fin L → α => (List.ofFn f).take i = u).card : ℝ)
          / (Fintype.card α : ℝ) ^ L
        ≤ (Fintype.card α : ℝ) ^ (L - i) / (Fintype.card α : ℝ) ^ L := by
          gcongr
          exact_mod_cast hcard
      _ = ((Fintype.card α : ℝ)⁻¹) ^ i := by
          rw [← Nat.sub_add_cancel hiL, pow_add, inv_pow, Nat.add_sub_cancel]
          field_simp
  · have hz : (Finset.univ.filter fun f : Fin L → α => (List.ofFn f).take i = u).card = 0 := by
      have := card_take_le (α := α) L i u hu
      rw [Finset.card_eq_zero]
      ext f
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, iff_false]
      intro h
      have := congrArg List.length h
      simp [List.length_take] at this
      omega
    rw [hz]
    simp only [Nat.cast_zero, zero_div]
    positivity

theorem replay_spread_uniform_holds : ReplaySpreadUniform := by
  intro α _ _ _ R s L t
  refine (replay_spread_holds R s (uniformStrings α L) L t).trans
    (Finset.sum_le_sum fun i hi => ?_)
  have hti : (t.toList.take i).length = i := by
    simp only [Finset.mem_range] at hi
    simp [List.length_take]
    omega
  apply mul_le_mul
  · gcongr
    exact_mod_cast min_le_left _ _
  · exact uniformStrings_take_le L i _ hti
  · exact measureReal_nonneg
  · positivity

theorem replay_spread_anchored_holds : ReplaySpreadAnchored := by
  intro α _ _ _ R s L t
  refine (replay_spread_of R s (uniformStrings α L) (uniformAnchor L) t).trans
    (Finset.sum_le_sum fun i hi => ?_)
  have hti : (t.toList.take i).length = i := by
    simp only [Finset.mem_range] at hi
    simp [List.length_take]
    omega
  rw [uniformAnchor_real_le]
  apply mul_le_mul
  · gcongr
    exact_mod_cast min_le_left _ _
  · exact uniformStrings_take_le L i _ hti
  · exact measureReal_nonneg
  · positivity

/-! ## The round's outcome -/

/-- A read the band decides is on the side of the middle of the band. -/
theorem midCut_of_cut (R : CutReads α) (hb : R.B.lo ≤ R.B.hi) {y : FreeMonoid α} {b : Bool}
    (h : R.cut y = some b) : midCut R y = b := by
  simp only [CutReads.cut, cutOn] at h
  simp only [midCut]
  split_ifs at h with h1 h2
  · cases h
    simp only [decide_eq_true_eq]
    omega
  · cases h
    simp only [decide_eq_false_iff_not, not_lt]
    omega

namespace DTree

theorem classify_of_sift {cut : FreeMonoid α → Option Bool} {g : FreeMonoid α → Bool}
    (hg : ∀ y b, cut y = some b → g y = b) :
    ∀ (t : DTree α) {x : FreeMonoid α} {p : List Bool}, t.sift cut x = .inl p → t.classify g x = p
  | .leaf, x, p, h => by simpa [sift, classify] using h
  | .node m r a, x, p, h => by
    simp only [sift] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> rw [hc] at h
    · simp at h
    · rcases hr : r.sift cut x with q | q <;> rw [hr] at h <;> simp at h
      subst h
      simp [classify, hg _ _ hc, classify_of_sift hg r hr]
    · rcases ha : a.sift cut x with q | q <;> rw [ha] at h <;> simp at h
      subst h
      simp [classify, hg _ _ hc, classify_of_sift hg a ha]

end DTree

theorem scanl_getD_length {β γ : Type*} (f : β → γ → β) :
    ∀ (l : List γ) (a d : β), (l.scanl f a).getD l.length d = l.foldl f a
  | [], a, d => by simp
  | c :: l, a, d => by
    simp only [List.scanl_cons, List.length_cons, List.foldl_cons]
    rw [List.getD_cons_succ, scanl_getD_length f l]

theorem prefixOf_zero (w : FreeMonoid α) : prefixOf w 0 = 1 := by
  simp [prefixOf]

theorem anchoredWalkFrom_zero (R : CutReads α) (t : DTree α)
    (edges : List Bool → α → Option (List Bool × FreeMonoid α)) (w : FreeMonoid α)
    {p : List Bool} (h : t.sift R.cut 1 = .inl p) :
    anchoredWalkFrom R t edges w 0 = some (0, w.toList.scanl (stepPath t edges) p) := by
  unfold anchoredWalkFrom
  rw [List.filter_eq_self.mpr (by simp), List.range_succ_eq_map, List.findSome?_cons,
    prefixOf_zero, h]
  simp

/-- A draw the gate counts against the hypothesis is one whose replay from the start harvests. -/
theorem replayHarvest_of_gateDisagrees (R : CutReads α) (s : PassState α) (hb : R.B.lo ≤ R.B.hi)
    {x : FreeMonoid α} (hd : GateDisagrees R s x) : replayHarvest R s x 0 ≠ [] := by
  have hg : ∀ y b, R.cut y = some b → midCut R y = b := fun y b h => midCut_of_cut R hb h
  rcases hε : s.tree.sift R.cut 1 with p | b
  · have hp : s.tree.classify (midCut R) 1 = p := DTree.classify_of_sift hg _ hε
    have hw := anchoredWalkFrom_zero R s.tree s.edges x hε
    have hend : (x.toList.scanl (stepPath s.tree s.edges) p).getD (x.toList.length - 0) []
        = gateEnd R s x := by
      rw [Nat.sub_zero, scanl_getD_length, gateEnd, hp]
    rcases hx : s.tree.sift R.cut x with actual | b
    · have ha : actual ≠ gateEnd R s x := by
        unfold GateDisagrees at hd
        rw [DTree.classify_of_sift hg _ hx] at hd
        exact fun h => hd h.symm
      simp only [replayHarvest, walkedHarvest, disagreedHarvest, hw, hx, hend, ha, if_false]
      split <;> simp
    · simp [replayHarvest, walkedHarvest, hw, hx]
  · simp [replayHarvest, walkedHarvest, List.range_succ_eq_map, prefixOf_zero, hε,
      List.takeWhile_cons]

instance : Countable (FreeMonoid α) := inferInstanceAs (Countable (List α))

instance : MeasurableSingletonClass (FreeMonoid α) := ⟨fun _ => trivial⟩

theorem replay_yield_holds : ReplayYield := by
  intro α _ _ R s D _ L hb
  set S := {q : FreeMonoid α × ℕ | replayHarvest R s q.1 q.2 ≠ []}
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hpair : Measurable fun x : FreeMonoid α => (x, (0 : ℕ)) :=
    measurable_id.prodMk measurable_const
  have hle : (2 : ℝ≥0∞)⁻¹ * D {x | GateDisagrees R s x} ≤ (D.prod (anchorLaw L)) S :=
    calc (2 : ℝ≥0∞)⁻¹ * D {x | GateDisagrees R s x}
        ≤ (2 : ℝ≥0∞)⁻¹ * D ((fun x : FreeMonoid α => (x, (0 : ℕ))) ⁻¹' S) := by
          gcongr
          intro x hx
          exact replayHarvest_of_gateDisagrees R s hb hx
      _ = (D.prod ((2 : ℝ≥0∞)⁻¹ • Measure.dirac 0)) S := by
          rw [Measure.prod_smul_right, Measure.prod_dirac, Measure.smul_apply,
            Measure.map_apply hpair hS, smul_eq_mul]
      _ ≤ (D.prod ((2 : ℝ≥0∞)⁻¹ • Measure.dirac 0)) S
            + (D.prod ((2 : ℝ≥0∞)⁻¹ • uniformAnchor L)) S := le_self_add
      _ = (D.prod (anchorLaw L)) S := by
          rw [anchorLaw, Measure.prod_add, Measure.add_apply]
  have hfin : (D.prod (anchorLaw L)) S ≠ ⊤ := measure_ne_top _ _
  calc D.real {x | GateDisagrees R s x} / 2
      = ((2 : ℝ≥0∞)⁻¹ * D {x | GateDisagrees R s x}).toReal := by
        rw [ENNReal.toReal_mul, measureReal_def, ENNReal.toReal_inv, ENNReal.toReal_ofNat]
        ring
    _ ≤ (D.prod (anchorLaw L)).real S := ENNReal.toReal_mono hfin hle

end OrthoDFA
