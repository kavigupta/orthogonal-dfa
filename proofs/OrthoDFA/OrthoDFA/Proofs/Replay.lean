import OrthoDFA.Stage

/-!
# The round's outcome

Every string a replay reads extends a prefix of the draw at least as long as its earliest anchor
(`take_of_mem_replayReads`), so reading a fixed `t` needs the draw to start as `t` does, up to
some point at or past the anchor.

A draw the gate counts against the hypothesis is one whose replay harvests, anchored no earlier
than any point before the gate's reading and the walk first part (`replay_harvests`): a read the
band decides is on the side of the middle of the band, so a replay that decides every read
follows the gate's own reading, and parts from the walk where the gate does.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

theorem route_queries (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) {x q : FreeMonoid α}, q ∈ (t.route cut x).1 → ∃ m, q = x * m
  | .leaf, x, q, h => by simp [route] at h
  | .node m r a, x, q, h => by
    simp only [route] at h
    split at h
    · simp only [List.mem_singleton] at h
      exact ⟨m, h⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, h⟩
      · exact route_queries cut a h
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, h⟩
      · exact route_queries cut r h

theorem sift_node_none {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    (h : cut (x * m) = none) : (node m r a).sift cut x = .inr (x * m) := by
  simp [sift, route, h]

theorem sift_node_some {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    {b : Bool} (h : cut (x * m) = some b) :
    (node m r a).sift cut x = ((if b then a else r).sift cut x).map (b :: ·) id := by
  cases b <;> simp [sift, route, h]

/-- Where a cut places `x`, any total reading that agrees with it on what it decides does too. -/
theorem sift_of_sift {cut : FreeMonoid α → Option Bool} {g : FreeMonoid α → Bool}
    (hg : ∀ y b, cut y = some b → g y = b) :
    ∀ (t : DTree α) {x : FreeMonoid α} {p : List Bool}, t.sift cut x = .inl p →
      t.sift (fun y => some (g y)) x = .inl p
  | .leaf, x, p, h => by simpa [sift, route] using h
  | .node m r a, x, p, h => by
    rcases hc : cut (x * m) with _ | b
    · rw [sift_node_none hc] at h
      simp at h
    · rw [sift_node_some hc] at h
      rw [sift_node_some (b := b) (by rw [hg _ _ hc])]
      cases b
      · simp only [Bool.false_eq_true, ↓reduceIte] at h ⊢
        rcases hr : r.sift cut x with q | q <;> rw [hr] at h <;> simp at h
        subst h
        rw [sift_of_sift hg r hr]
        rfl
      · simp only [↓reduceIte] at h ⊢
        rcases ha : a.sift cut x with q | q <;> rw [ha] at h <;> simp at h
        subst h
        rw [sift_of_sift hg a ha]
        rfl

/-- What a node cannot place is the string extended by its midfix. -/
theorem sift_inr {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) {x b : FreeMonoid α}, t.sift cut x = .inr b → ∃ m, b = x * m
  | .leaf, x, b, h => by simp [sift, route] at h
  | .node m r a, x, b, h => by
    rcases hc : cut (x * m) with _ | c
    · rw [sift_node_none hc] at h
      simp only [Sum.inr.injEq] at h
      exact ⟨m, h.symm⟩
    · rw [sift_node_some hc] at h
      cases c
      · simp only [Bool.false_eq_true, ↓reduceIte] at h
        rcases hr : r.sift cut x with q | q <;> rw [hr] at h <;> simp at h
        subst h
        exact sift_inr r hr
      · simp only [↓reduceIte] at h
        rcases ha : a.sift cut x with q | q <;> rw [ha] at h <;> simp at h
        subst h
        exact sift_inr a ha

end DTree

variable (R : CutReads α)

theorem anchorSearch_mem (t : DTree α) (w : FreeMonoid α) :
    ∀ is : List ℕ, (∀ i ∈ (anchorSearch R t w is).1, i ∈ is)
      ∧ ∀ s p, (anchorSearch R t w is).2 = some (s, p) → s ∈ is
  | [] => by simp [anchorSearch]
  | i :: is => by
    have ih := anchorSearch_mem t w is
    simp only [anchorSearch]
    split
    · simp
    · refine ⟨fun j hj => ?_, fun s p h => List.mem_cons_of_mem _ (ih.2 s p h)⟩
      simp only [List.mem_cons] at hj ⊢
      rcases hj with rfl | hj
      · exact Or.inl rfl
      · exact Or.inr (ih.1 j hj)

theorem anchorSearch_cons_of_sift (t : DTree α) {w : FreeMonoid α} {i : ℕ} {is : List ℕ}
    {p : List Bool} (h : t.sift R.cut (prefixOf w i) = .inl p) :
    anchorSearch R t w (i :: is) = ([], some (i, p)) := by
  simp only [anchorSearch, h]

theorem anchorSearch_some (t : DTree α) (w : FreeMonoid α) :
    ∀ (is : List ℕ) {s : ℕ} {p : List Bool}, (anchorSearch R t w is).2 = some (s, p) →
      t.sift R.cut (prefixOf w s) = .inl p
  | [], s, p, h => by simp [anchorSearch] at h
  | i :: is, s, p, h => by
    simp only [anchorSearch] at h
    split at h
    · rename_i q hq
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact hq
    · exact anchorSearch_some t w is h

/-- Between an index where the sift agrees with the walk and a later one, the bisection either
meets a prefix strictly between them it cannot place, or settles on an index past the first. -/
theorem bisect_result (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ∀ (fuel lo hi : ℕ), lo < hi →
      (∀ b, (bisect R t w walk fuel lo hi).2 = .inl b →
          ∃ j, lo < j ∧ j < hi ∧ ∃ m, b = prefixOf w j * m)
        ∧ ∀ fd, (bisect R t w walk fuel lo hi).2 = .inr fd → lo < fd ∧ fd ≤ hi
  | 0, lo, hi, hlh => by
    refine ⟨fun b h => by simp [bisect] at h, fun fd h => ?_⟩
    simp only [bisect, Sum.inr.injEq] at h
    omega
  | fuel + 1, lo, hi, hlh => by
    simp only [bisect]
    split_ifs with h1
    · split
      · rename_i b hb
        refine ⟨fun b' h => ?_, fun fd h => by simp at h⟩
        simp only [Sum.inl.injEq] at h
        subst h
        obtain ⟨m, hm⟩ := DTree.sift_inr t hb
        exact ⟨_, by omega, by omega, m, hm⟩
      · split_ifs
        · have ih := bisect_result t w walk fuel ((lo + hi) / 2) hi (by omega)
          refine ⟨fun b h => ?_, fun fd h => ?_⟩
          · obtain ⟨j, hj1, hj2, hj3⟩ := ih.1 b h
            exact ⟨j, by omega, hj2, hj3⟩
          · have := ih.2 fd h
            omega
        · have ih := bisect_result t w walk fuel lo ((lo + hi) / 2) (by omega)
          refine ⟨fun b h => ?_, fun fd h => ?_⟩
          · obtain ⟨j, hj1, hj2, hj3⟩ := ih.1 b h
            exact ⟨j, hj1, by omega, hj3⟩
          · have := ih.2 fd h
            omega
    · refine ⟨fun b h => by simp at h, fun fd h => ?_⟩
      simp only [Sum.inr.injEq] at h
      omega

theorem bisect_bounds (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ∀ (fuel lo hi j : ℕ), j ∈ (bisect R t w walk fuel lo hi).1 → lo < j ∧ j < hi
  | 0, lo, hi, j, h => by simp [bisect] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bisect] at h
    split_ifs at h with hlh
    · split at h
      · simp only [List.mem_singleton] at h
        omega
      · simp only [List.mem_cons] at h
        rcases h with rfl | h
        · omega
        · split_ifs at h
          · have := bisect_bounds t w walk fuel _ hi j h
            omega
          · have := bisect_bounds t w walk fuel lo _ j h
            omega
    · simp at h

theorem replay_bounds (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ) {i : ℕ}
    (h : i ∈ (replay R H w e).1) : e ≤ i ∧ i ≤ w.toList.length := by
  have hc : ∀ j ∈ (List.range (w.toList.length + 1)).filter (e ≤ ·),
      e ≤ j ∧ j ≤ w.toList.length := by
    intro j hj
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at hj
    omega
  have hm := anchorSearch_mem R H.tree w ((List.range (w.toList.length + 1)).filter (e ≤ ·))
  simp only [replay] at h
  split at h
  · exact hc _ (hm.1 _ h)
  · rename_i start p hs
    have hstart := hc _ (hm.2 _ _ hs)
    split at h
    · simp only [List.mem_append, List.mem_cons] at h
      rcases h with h | rfl | rfl | h
      · exact hc _ (hm.1 _ h)
      · exact hstart
      · omega
      · simp at h
    · split_ifs at h
      · simp only [List.mem_append, List.mem_cons] at h
        rcases h with h | rfl | rfl | h
        · exact hc _ (hm.1 _ h)
        · exact hstart
        · omega
        · simp at h
      · simp only [List.mem_append, List.mem_cons] at h
        rcases h with (h | rfl | rfl | h) | h
        · exact hc _ (hm.1 _ h)
        · exact hstart
        · omega
        · simp at h
        · have := bisect_bounds R _ _ _ _ _ _ _ h
          omega

/-- A string a replay reads starts as the draw does, up to some point at or past the earliest
anchor. -/
theorem take_of_mem_replayReads (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ)
    {x : FreeMonoid α} (h : x ∈ replayReads R H w e) :
    ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
  obtain ⟨i, hi, q, hq, v, -, rfl⟩ := h
  obtain ⟨hei, hin⟩ := replay_bounds R H w e hi
  obtain ⟨m, rfl⟩ := DTree.route_queries R.cut _ hq
  have hlen : (prefixOf w i).toList.length = i := by
    simp [prefixOf, List.length_take, hin]
  refine ⟨i, hei, ?_, ?_⟩
  · simp only [FreeMonoid.toList_mul, List.length_append, hlen]
    omega
  · simp only [FreeMonoid.toList_mul, List.append_assoc]
    rw [List.take_left' hlen]
    simp [prefixOf]

theorem take_of_prefix_mul {w : FreeMonoid α} {i : ℕ} (hin : i ≤ w.toList.length)
    (m : FreeMonoid α) :
    i ≤ (prefixOf w i * m).toList.length ∧ w.toList.take i = (prefixOf w i * m).toList.take i := by
  have hlen : (prefixOf w i).toList.length = i := by
    simp [prefixOf, List.length_take, hin]
  refine ⟨?_, ?_⟩
  · simp only [FreeMonoid.toList_mul, List.length_append, hlen]
    omega
  · simp only [FreeMonoid.toList_mul]
    rw [List.take_left' hlen]
    simp [prefixOf]

/-- A string a replay harvests starts as the draw does, up to some point at or past the earliest
anchor: a prefix it cannot place, extended by a midfix, or the prefix before the disagreeing
edge. -/
theorem take_of_mem_harvest (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ)
    {x : FreeMonoid α} (h : x ∈ (replay R H w e).2) :
    ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
  have hc : ∀ j ∈ (List.range (w.toList.length + 1)).filter (e ≤ ·),
      e ≤ j ∧ j ≤ w.toList.length := by
    intro j hj
    simp only [List.mem_filter, List.mem_range, decide_eq_true_eq] at hj
    omega
  have hm := anchorSearch_mem R H.tree w ((List.range (w.toList.length + 1)).filter (e ≤ ·))
  have hpre : ∀ i, e ≤ i → i ≤ w.toList.length → ∀ m, x = prefixOf w i * m →
      ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
    rintro i hei hin m rfl
    exact ⟨i, hei, take_of_prefix_mul hin m⟩
  have hmissed : x ∈ (anchorSearch R H.tree w
      ((List.range (w.toList.length + 1)).filter (e ≤ ·))).1.filterMap
        (fun i => (H.tree.sift R.cut (prefixOf w i)).getRight?) →
      ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
    intro hx
    obtain ⟨i, hi, hgr⟩ := List.mem_filterMap.mp hx
    obtain ⟨hei, hin⟩ := hc _ (hm.1 _ hi)
    have hs : H.tree.sift R.cut (prefixOf w i) = .inr x := by
      rcases hq : H.tree.sift R.cut (prefixOf w i) with q | q <;> rw [hq] at hgr <;>
        simp at hgr
      rw [hgr]
    obtain ⟨m, hm'⟩ := DTree.sift_inr _ hs
    exact hpre i hei hin m hm'
  have hwn : prefixOf w w.toList.length = w := by simp [prefixOf]
  simp only [replay] at h
  split at h
  · exact hmissed h
  · rename_i start p hs
    obtain ⟨hes, hsn⟩ := hc _ (hm.2 _ _ hs)
    split at h
    · rename_i b hb
      simp only [List.mem_append, List.mem_singleton] at h
      rcases h with h | rfl
      · exact hmissed h
      · obtain ⟨m, hm'⟩ := DTree.sift_inr _ hb
        exact hpre _ (hes.trans hsn) le_rfl m (by rw [hwn]; exact hm')
    · rename_i actual hact
      split_ifs at h with hagree
      · exact hmissed h
      · have hlt : start < w.toList.length := by
          by_contra hge
          have heq : start = w.toList.length := by omega
          apply hagree
          have h1 := anchorSearch_some R H.tree w _ hs
          subst heq
          rw [hwn, hact, Sum.inl.injEq] at h1
          subst h1
          simp
        simp only [List.mem_append] at h
        rcases h with h | h
        · exact hmissed h
        · split at h
          · rename_i b hb'
            simp only [List.mem_singleton] at h
            subst h
            obtain ⟨j, hj1, hj2, m, hm'⟩ := (bisect_result R H.tree w _ _ _ _ hlt).1 _ hb'
            exact hpre j (by omega) (by omega) m hm'
          · rename_i fd hfd
            simp only [List.mem_singleton] at h
            subst h
            obtain ⟨h1, h2⟩ := (bisect_result R H.tree w _ _ _ _ hlt).2 fd hfd
            exact hpre (fd - 1) (by omega) (by omega) 1 (by simp)

instance (L : ℕ) : IsFiniteMeasure (anchorLaw L) := by
  constructor
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp [anchorLaw]
  · simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
      measure_univ, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one, smul_eq_mul]
    rw [ENNReal.inv_mul_cancel (by exact_mod_cast hL.ne') (by simp)]
    exact ENNReal.one_lt_top

theorem anchorLaw_real_le (L i : ℕ) :
    (anchorLaw L).real {e | e ≤ i} = ((min (i + 1) L : ℕ) : ℝ) / L := by
  have hcount : ∑ k ∈ Finset.range L, (Measure.dirac k : Measure ℕ) {e | e ≤ i}
      = ((min (i + 1) L : ℕ) : ℝ≥0∞) := by
    simp only [Measure.dirac_apply, Set.indicator_apply, Set.mem_setOf_eq, Pi.one_apply]
    rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
    have hf : ((Finset.range L).filter fun k => k ≤ i) = Finset.range (min (i + 1) L) := by
      ext k
      simp only [Finset.mem_filter, Finset.mem_range]
      omega
    rw [hf, Finset.card_range]
  simp only [measureReal_def, anchorLaw, Measure.smul_apply, Measure.coe_finsetSum,
    Finset.sum_apply, smul_eq_mul, hcount, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_natCast]
  ring

/-- If whatever a replay from `e` reads or harvests starts as the draw does up to some point at or
past `e`, a replay from an anchor drawn from `A` takes `t` with chance at most
`∑_{i ≤ |t|} A(e ≤ i) · D(the draw's first i letters are t's)`. -/
theorem spread_of (P : FreeMonoid α → ℕ → FreeMonoid α → Prop)
    (hP : ∀ w e x, P w e x → ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i)
    (D : Measure (FreeMonoid α)) [IsFiniteMeasure D] (A : Measure ℕ) [IsFiniteMeasure A]
    (t : FreeMonoid α) :
    (D.prod A).real {q | P q.1 q.2 t}
      ≤ ∑ i ∈ Finset.range (t.toList.length + 1),
          A.real {e | e ≤ i} * D.real {p | p.toList.take i = t.toList.take i} := by
  have hsub : {q : FreeMonoid α × ℕ | P q.1 q.2 t} ⊆
      ⋃ i ∈ Finset.range (t.toList.length + 1),
        {p : FreeMonoid α | p.toList.take i = t.toList.take i} ×ˢ {e : ℕ | e ≤ i} := by
    rintro ⟨p, e⟩ hq
    obtain ⟨i, hei, hit, htake⟩ := hP p e t hq
    simp only [Set.mem_iUnion, Finset.mem_range, Set.mem_prod, Set.mem_setOf_eq]
    exact ⟨i, by omega, htake, hei⟩
  calc (D.prod A).real {q | P q.1 q.2 t}
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

/-- A read the band decides is on the side of the middle of the band. -/
theorem mid_of_cut (hb : R.B.lo ≤ R.B.hi) {y : FreeMonoid α} {b : Bool}
    (h : R.cut y = some b) : decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f y) = b := by
  simp only [CutReads.cut] at h
  split_ifs at h with h1 h2
  · cases h
    simp only [decide_eq_true_eq]
    omega
  · cases h
    simp only [decide_eq_false_iff_not, not_lt]
    omega

theorem midPath_of_sift (H : Hypothesis α) (hb : R.B.lo ≤ R.B.hi) {x : FreeMonoid α}
    {p : List Bool} (h : H.tree.sift R.cut x = .inl p) : midPath R H x = p := by
  have := DTree.sift_of_sift
    (g := fun y => decide (R.B.lo + R.B.hi < 2 * acceptsOn R.F R.f y))
    (fun y b hy => mid_of_cut R hb hy) H.tree h
  simp only [midPath, this, Sum.elim_inl, id]

theorem scanl_getD_length {β γ : Type*} (f : β → γ → β) :
    ∀ (l : List γ) (a d : β), (l.scanl f a).getD l.length d = l.foldl f a
  | [], a, d => by simp
  | c :: l, a, d => by
    simp only [List.scanl_cons, List.length_cons, List.foldl_cons]
    rw [List.getD_cons_succ, scanl_getD_length f l]

theorem prefixOf_zero (w : FreeMonoid α) : prefixOf w 0 = 1 := by
  simp [prefixOf]

theorem filter_range_ge {e n : ℕ} (h : e ≤ n) :
    ∃ rest, (List.range (n + 1)).filter (e ≤ ·) = e :: rest := by
  obtain ⟨k, hk⟩ : ∃ k, n + 1 = e + (k + 1) := ⟨n - e, by omega⟩
  rw [hk, List.range_add, List.filter_append,
    List.filter_eq_nil_iff.mpr (by simp only [List.mem_range, decide_eq_true_eq]; omega),
    List.nil_append, List.filter_eq_self.mpr (by simp), List.range_succ_eq_map]
  exact ⟨_, by rw [List.map_cons, Nat.add_zero]⟩

/-- What a replay harvests begins with what its anchor search could not place. -/
theorem replay_snd (H : Hypothesis α) (w : FreeMonoid α) (e : ℕ) :
    ∃ l, (replay R H w e).2 =
      ((anchorSearch R H.tree w ((List.range (w.toList.length + 1)).filter (e ≤ ·))).1.filterMap
        fun i => (H.tree.sift R.cut (prefixOf w i)).getRight?) ++ l := by
  simp only [replay]
  split
  · exact ⟨[], by simp⟩
  · split
    · exact ⟨_, rfl⟩
    · split_ifs
      · exact ⟨[], by simp⟩
      · exact ⟨_, rfl⟩

/-- A replay anchored no earlier than `e`, of a draw the gate counts against the hypothesis,
harvests, if the middle-of-band reading of the draw's first `e` letters is where the walk from
the hypothesis's start is after them.  That holds at `e = 0`, and at every `e` before the first
place the two part. -/
theorem replay_harvests (H : Hypothesis α) (hb : R.B.lo ≤ R.B.hi) {x : FreeMonoid α} {e : ℕ}
    (he : e ≤ x.toList.length)
    (hag : midPath R H (prefixOf x e) = (x.toList.take e).foldl H.step (midPath R H 1))
    (hd : DFAandDTDisagree R H x) : (replay R H x e).2 ≠ [] := by
  obtain ⟨rest, hrest⟩ := filter_range_ge he
  rcases hε : H.tree.sift R.cut (prefixOf x e) with p | b
  · have hp : midPath R H (prefixOf x e) = p := midPath_of_sift R H hb hε
    have hs := anchorSearch_cons_of_sift R H.tree (is := rest) hε
    have hend : ((x.toList.drop e).scanl H.step p).getD (x.toList.length - e) []
        = x.toList.foldl H.step (midPath R H 1) := by
      rw [← List.length_drop, scanl_getD_length, ← hp, hag, ← List.foldl_append,
        List.take_append_drop]
    rcases hx : H.tree.sift R.cut x with actual | b
    · have ha : actual ≠ x.toList.foldl H.step (midPath R H 1) := by
        intro h
        apply hd
        rw [midPath_of_sift R H hb hx, h]
      simp only [replay, hrest, hs, hx, hend, ha, if_false]
      split <;> simp
    · simp [replay, hrest, hs, hx]
  · obtain ⟨l, hl⟩ := replay_snd R H x e
    rw [hl, hrest]
    simp [anchorSearch, hε]

instance : Countable (FreeMonoid α) := inferInstanceAs (Countable (List α))

instance : MeasurableSingletonClass (FreeMonoid α) := ⟨fun _ => trivial⟩

theorem replay_yield (H : Hypothesis α) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (L : ℕ) (hb : R.B.lo ≤ R.B.hi) :
    D.real {x | DFAandDTDisagree R H x} / L
      ≤ (D.prod (anchorLaw L)).real {q | (replay R H q.1 q.2).2 ≠ []} := by
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp only [Nat.cast_zero, div_zero]
    exact measureReal_nonneg
  obtain ⟨m, rfl⟩ : ∃ m, L = m + 1 := ⟨L - 1, by omega⟩
  set S := {q : FreeMonoid α × ℕ | (replay R H q.1 q.2).2 ≠ []}
  have hS : MeasurableSet S := S.to_countable.measurableSet
  have hpair : Measurable fun x : FreeMonoid α => (x, (0 : ℕ)) :=
    measurable_id.prodMk measurable_const
  have hle : ((m + 1 : ℕ) : ℝ≥0∞)⁻¹ * D {x | DFAandDTDisagree R H x}
      ≤ (D.prod (anchorLaw (m + 1))) S :=
    calc ((m + 1 : ℕ) : ℝ≥0∞)⁻¹ * D {x | DFAandDTDisagree R H x}
        ≤ ((m + 1 : ℕ) : ℝ≥0∞)⁻¹ * D ((fun x : FreeMonoid α => (x, (0 : ℕ))) ⁻¹' S) := by
          gcongr
          intro x hx
          refine replay_harvests R H hb (Nat.zero_le _) ?_ hx
          simp [prefixOf_zero]
      _ = (D.prod (((m + 1 : ℕ) : ℝ≥0∞)⁻¹ • Measure.dirac 0)) S := by
          rw [Measure.prod_smul_right, Measure.prod_dirac, Measure.smul_apply,
            Measure.map_apply hpair hS, smul_eq_mul]
      _ ≤ (D.prod (((m + 1 : ℕ) : ℝ≥0∞)⁻¹
              • ∑ k ∈ Finset.range m, Measure.dirac (k + 1))) S
            + (D.prod (((m + 1 : ℕ) : ℝ≥0∞)⁻¹ • Measure.dirac 0)) S := le_add_self
      _ = (D.prod (anchorLaw (m + 1))) S := by
          rw [anchorLaw, Finset.sum_range_succ', smul_add, Measure.prod_add, Measure.add_apply]
  have hfin : (D.prod (anchorLaw (m + 1))) S ≠ ⊤ := measure_ne_top _ _
  calc D.real {x | DFAandDTDisagree R H x} / ((m + 1 : ℕ) : ℝ)
      = (((m + 1 : ℕ) : ℝ≥0∞)⁻¹ * D {x | DFAandDTDisagree R H x}).toReal := by
        rw [ENNReal.toReal_mul, measureReal_def, ENNReal.toReal_inv, ENNReal.toReal_natCast]
        ring
    _ ≤ (D.prod (anchorLaw (m + 1))).real S := ENNReal.toReal_mono hfin hle

theorem round_outcome_holds : RoundOutcome := by
  intro α _ _ R H D _ L hb
  refine ⟨replay_yield R H D L hb, fun t => ?_, fun t => ?_⟩
  · exact (spread_of (fun w e x => x ∈ (replay R H w e).2)
      (fun w e x h => take_of_mem_harvest R H w e h) D (anchorLaw L) t).trans_eq
      (Finset.sum_congr rfl fun i _ => by rw [anchorLaw_real_le])
  · exact (spread_of (fun w e x => x ∈ replayReads R H w e)
      (fun w e x h => take_of_mem_replayReads R H w e h) D (anchorLaw L) t).trans_eq
      (Finset.sum_congr rfl fun i _ => by rw [anchorLaw_real_le])

end OrthoDFA
