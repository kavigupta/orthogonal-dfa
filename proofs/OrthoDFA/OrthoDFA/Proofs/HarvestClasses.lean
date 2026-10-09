import OrthoDFA.Proofs.HarvestBound
import OrthoDFA.Proofs.Visits

/-!
# The harvest classes

Each class of `QualityHolds` as a computation and harvest meeting `HarvestSpec`: the ends' sifts
with the root's read untagged, the reads at an unlearned edge, and the search's middles.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Ends

/-- The sift of `w`, the root's read untagged: the read it leaves undecided below the root. -/
def qSiftDeep (w : FreeMonoid α) : DTree α → Qry α (Option (FreeMonoid α))
  | .leaf => .pure none
  | .node m r a => .ask (w * m) false fun o =>
    match o with
    | none => .pure none
    | some true => (qSift w true a).map Sum.getRight?
    | some false => (qSift w true r).map Sum.getRight?

variable (cut : FreeMonoid α → Option Bool)

theorem qSiftDeep_of_deep (w b : FreeMonoid α) :
    ∀ t : DTree α, t.sift cut w = .inr b → 2 ≤ (t.route cut w).1.length →
      (qSiftDeep w t).run cut = some b
  | .leaf, h, _ => by simp [DTree.sift, DTree.route] at h
  | .node m r a, h, hl => by
    simp only [DTree.sift, DTree.route] at h hl
    simp only [qSiftDeep, Qry.run]
    rcases hc : cut (w * m) with _ | _ | _ <;> simp only [hc] at h hl ⊢
    · simp at hl
    · rw [Qry.run_map, qSift_run]
      rcases hs : (r.route cut w).2 with _ | b' <;> rw [hs] at h
      · simp at h
      · simp only [Sum.map_inr, Sum.inr.injEq, id_eq] at h
        subst h
        simp [DTree.sift, hs]
    · rw [Qry.run_map, qSift_run]
      rcases hs : (a.route cut w).2 with _ | b' <;> rw [hs] at h
      · simp at h
      · simp only [Sum.map_inr, Sum.inr.injEq, id_eq] at h
        subst h
        simp [DTree.sift, hs]

theorem qSiftDeep_some (w b : FreeMonoid α) :
    ∀ t : DTree α, (qSiftDeep w t).run cut = some b →
      t.sift cut w = .inr b ∧ cut b = none
      ∧ ∃ pre blk, (qSiftDeep w t).trace cut = pre ++ blk ++ []
        ∧ (∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) ∧ (∀ e ∈ blk, e.2 = true) ∧ (b, true) ∈ blk
  | .leaf, h => by simp [qSiftDeep, Qry.run] at h
  | .node m r a, h => by
    simp only [qSiftDeep, Qry.run] at h
    simp only [DTree.sift, DTree.route, qSiftDeep, Qry.trace]
    rcases hc : cut (w * m) with _ | v <;> simp only [hc] at h ⊢
    · simp [Qry.run] at h
    have hpre : ∀ e ∈ [(w * m, false)], e.2 = false → cut e.1 ≠ none := by simp [hc]
    cases v
    · simp only [Qry.run_map, qSift_run] at h
      simp only [Qry.trace_map, qSift_trace]
      rcases hs : r.sift cut w with _ | b' <;> rw [hs] at h
      · simp at h
      obtain rfl : b' = b := by simpa using h
      obtain ⟨hmem, hcut⟩ := route_undecided cut r w b' hs
      refine ⟨by simp [DTree.sift] at hs; simp [hs], hcut, [(w * m, false)],
        (r.route cut w).1.map (·, true), by simp, hpre,
        fun e he => by obtain ⟨y, -, rfl⟩ := List.mem_map.1 he; rfl,
        List.mem_map.2 ⟨b', hmem, rfl⟩⟩
    · simp only [Qry.run_map, qSift_run] at h
      simp only [Qry.trace_map, qSift_trace]
      rcases hs : a.sift cut w with _ | b' <;> rw [hs] at h
      · simp at h
      obtain rfl : b' = b := by simpa using h
      obtain ⟨hmem, hcut⟩ := route_undecided cut a w b' hs
      refine ⟨by simp [DTree.sift] at hs; simp [hs], hcut, [(w * m, false)],
        (a.route cut w).1.map (·, true), by simp, hpre,
        fun e he => by obtain ⟨y, -, rfl⟩ := List.mem_map.1 he; rfl,
        List.mem_map.2 ⟨b', hmem, rfl⟩⟩

theorem qSiftDeep_countP (w : FreeMonoid α) :
    ∀ t : DTree α, ((qSiftDeep w t).trace cut).countP (fun e => e.2) ≤ t.depth - 1
  | .leaf => by simp [qSiftDeep, Qry.trace]
  | .node m r a => by
    simp only [qSiftDeep, Qry.trace, DTree.depth]
    rcases cut (w * m) with _ | _ | _
    · simp [Qry.trace]
    · simp only [Qry.trace_map, List.countP_cons, Bool.false_eq_true, if_false, add_zero]
      have h1 := qSift_length cut w true r
      have h2 := List.countP_le_length (p := fun e : FreeMonoid α × Bool => e.2)
        (l := (qSift w true r).trace cut)
      omega
    · simp only [Qry.trace_map, List.countP_cons, Bool.false_eq_true, if_false, add_zero]
      have h1 := qSift_length cut w true a
      have h2 := List.countP_le_length (p := fun e : FreeMonoid α × Bool => e.2)
        (l := (qSift w true a).trace cut)
      omega

theorem qSiftDeep_asksIn (w : FreeMonoid α) :
    ∀ t : DTree α, (qSiftDeep w t).AsksIn fun y => ∃ m ∈ t.mids, y = w * m
  | .leaf => trivial
  | .node m r a => by
    refine ⟨⟨m, by simp [DTree.mids], rfl⟩, fun o => ?_⟩
    rcases o with _ | _ | _
    · trivial
    · exact Qry.asksIn_map _ _ (Qry.asksIn_mono (fun y ⟨m', hm', he⟩ =>
        ⟨m', by simp [DTree.mids, hm'], he⟩) _ (qSift_asksIn w true r))
    · exact Qry.asksIn_map _ _ (Qry.asksIn_mono (fun y ⟨m', hm', he⟩ =>
        ⟨m', by simp [DTree.mids, hm'], he⟩) _ (qSift_asksIn w true a))

theorem qSiftDeep_form (w b : FreeMonoid α) (t : DTree α) (h : (qSiftDeep w t).run cut = some b) :
    ∃ m, b = w * m :=
  route_inr_form t w b (qSiftDeep_some cut w b t h).1

omit [Fintype α] [DecidableEq α] in
theorem prefixOf_max (x : FreeMonoid α) (k : ℕ) : prefixOf x (max k x.toList.length) = x := by
  apply FreeMonoid.toList.injective
  simp [prefixOf, List.take_of_length_le (le_max_right _ _)]

/-- The ends' classes: a draw's first `k` letters, and the whole draw. -/
theorem ends_spec (k : ℕ) (start : Bool) :
    HarvestSpec (fun t (_ : Edges α) x =>
      qSiftDeep (if start then prefixOf x k else x) t) Option.toList k where
  asks t _ x := Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => by
      refine ⟨if start then k else max k x.toList.length, ?_, m, hm, ?_⟩
      · split_ifs <;> simp
      · rw [he]; split_ifs <;> simp [prefixOf_max]) _ (qSiftDeep_asksIn _ t)
  first cut t _ x h := by
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 (by simpa using h)
    obtain ⟨-, hcut, pre, blk, htr, hpre, hblk, hmem⟩ := qSiftDeep_some cut _ b t hb
    refine ⟨b, by simp [hb], hcut, ?_⟩
    rw [htr]
    exact first_of_split cut hpre hblk hmem hcut
  form cut t _ x b hb := by
    have hb' : (qSiftDeep (if start then prefixOf x k else x) t).run cut = some b := by
      simpa using hb
    obtain ⟨m, rfl⟩ := qSiftDeep_form cut _ b t hb'
    refine ⟨if start then k else max k x.toList.length, by split_ifs <;> simp, m, ?_⟩
    split_ifs <;> simp [prefixOf_max]

end Ends

section Blocked

/-- The walk from `k` to an unlearned edge, the reads there tagged: the read it leaves
undecided. -/
def qWalkB (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Qry α (Option (FreeMonoid α)) :=
  (qSift (prefixOf x k) false t).bind fun s0 =>
    match s0 with
    | .inr _ => .pure none
    | .inl p =>
      match follow edges p (x.toList.drop k) with
      | .inl _ => .pure none
      | .inr (_, _, i) => (qSift (prefixOf x (k + i + 1)) true t).bind fun s1 =>
        match s1 with
        | .inr b => .pure (some b)
        | .inl _ => (qSift (prefixOf x (k + i)) true t).map Sum.getRight?

variable (cut : FreeMonoid α → Option Bool)

theorem qWalkB_some (t : DTree α) (edges : Edges α) (k : ℕ) (x b : FreeMonoid α)
    (h : (qWalkB t edges k x).run cut = some b) :
    cut b = none ∧ (∃ i, k ≤ i ∧ ∃ m, b = prefixOf x i * m)
      ∧ ∃ pre blk, (qWalkB t edges k x).trace cut = pre ++ blk ++ []
        ∧ (∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) ∧ (∀ e ∈ blk, e.2 = true)
        ∧ (b, true) ∈ blk := by
  unfold qWalkB at h ⊢
  rw [Qry.run_bind, qSift_run] at h
  rw [Qry.trace_bind, qSift_run]
  have hd0 := qSift_decided cut (prefixOf x k) false t
  rcases hk : t.sift cut (prefixOf x k) with p | _ <;> simp only [hk] at h ⊢
  · rcases hf : follow edges p (x.toList.drop k) with _ | ⟨s, c, i⟩ <;> simp only [hf] at h ⊢
    · simp [Qry.run] at h
    rw [Qry.run_bind, qSift_run] at h
    rw [Qry.trace_bind, qSift_run]
    have hpre0 : ∀ e ∈ (qSift (prefixOf x k) false t).trace cut, e.2 = false → cut e.1 ≠ none :=
      fun e he _ => hd0 (by simp [hk]) e he
    rcases h1 : t.sift cut (prefixOf x (k + i + 1)) with p1 | b1 <;> simp only [h1] at h ⊢
    · rw [Qry.run_map, qSift_run] at h
      rcases h2 : t.sift cut (prefixOf x (k + i)) with _ | b2 <;> rw [h2] at h
      · simp at h
      obtain rfl : b2 = b := by simpa using h
      obtain ⟨hmem, hcut⟩ := route_undecided cut t _ b2 h2
      obtain ⟨m, hm⟩ := route_inr_form t _ b2 h2
      refine ⟨hcut, ⟨k + i, by omega, m, hm⟩,
        (qSift (prefixOf x k) false t).trace cut ++ (qSift (prefixOf x (k + i + 1)) true t).trace cut,
        (qSift (prefixOf x (k + i)) true t).trace cut, by simp [Qry.trace_map], ?_,
        qSift_tag cut _ true t, by rw [qSift_trace]; exact List.mem_map.2 ⟨b2, hmem, rfl⟩⟩
      intro e he _
      rcases List.mem_append.1 he with he | he
      · exact hd0 (by simp [hk]) e he
      · exact qSift_decided cut _ true t (by simp [h1]) e he
    · simp only [Qry.run] at h
      obtain rfl : b1 = b := by simpa using h
      obtain ⟨hmem, hcut⟩ := route_undecided cut t _ b1 h1
      obtain ⟨m, hm⟩ := route_inr_form t _ b1 h1
      refine ⟨hcut, ⟨k + i + 1, by omega, m, hm⟩,
        (qSift (prefixOf x k) false t).trace cut, (qSift (prefixOf x (k + i + 1)) true t).trace cut,
        by simp [Qry.trace], hpre0, qSift_tag cut _ true t,
        by rw [qSift_trace]; exact List.mem_map.2 ⟨b1, hmem, rfl⟩⟩
  · simp [Qry.run] at h

theorem qWalkB_of_blocked (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ)
    (x w b : FreeMonoid α) (hB : IsBlocked R t edges k x)
    (hw : probeOutcome R t edges k x = .endUndecided w) (hb : t.sift R.cut w = .inr b) :
    (qWalkB t edges k x).run R.cut = some b := by
  have hwc : walkCheck R t edges k x = .inl (.endUndecided w) := by
    unfold probeOutcome at hw
    rcases hc : walkCheck R t edges k x with o | d <;> rw [hc] at hw
    · simp only [Sum.elim_inl, id] at hw; rw [hw]
    · have := bracketAt_isSearch (α := α) (agreesAt R t x fun j => d.1.getD (j - k) []) d.1
        (d.2 - k) k d.2
      simp only [Sum.elim_inr] at hw
      rw [hw] at this
      exact this.elim
  obtain ⟨⟨s, c, j, hkw⟩, -⟩ := hB
  unfold kWalk at hkw
  rcases hk : t.sift R.cut (prefixOf x k) with p | _ <;> rw [hk] at hkw
  swap
  · simp at hkw
  simp only [] at hkw
  rcases hf : follow edges p (x.toList.drop k) with _ | ⟨s', c', i⟩ <;> rw [hf] at hkw
  · simp at hkw
  simp only [KWalk.edge.injEq] at hkw
  obtain ⟨rfl, rfl, rfl⟩ := hkw
  unfold walkCheck kWalk at hwc
  rw [hk] at hwc
  simp only [hf] at hwc
  unfold qWalkB
  rw [Qry.run_bind, qSift_run, hk]
  simp only [hf]
  rw [Qry.run_bind, qSift_run]
  rcases h1 : t.sift R.cut (prefixOf x (k + i + 1)) with p1 | b1 <;> rw [h1] at hwc
  · simp only [] at hwc ⊢
    rw [Qry.run_map, qSift_run]
    rcases h2 : t.sift R.cut (prefixOf x (k + i)) with p2 | b2 <;> rw [h2] at hwc
    · simp only [] at hwc
      split_ifs at hwc <;> simp at hwc
    · simp only [Sum.inl.injEq, Outcome.endUndecided.injEq] at hwc
      subst hwc
      rw [h2] at hb
      simpa using hb
  · simp only [Sum.inl.injEq, Outcome.endUndecided.injEq] at hwc
    subst hwc
    rw [h1] at hb
    simp only [Qry.run]
    simpa using hb

theorem qWalkB_countP (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    ((qWalkB t edges k x).trace cut).countP (fun e => e.2) ≤ 2 * t.depth := by
  unfold qWalkB
  rw [Qry.trace_bind, List.countP_append]
  have h0 : ((qSift (prefixOf x k) false t).trace cut).countP (fun e => e.2) = 0 :=
    List.countP_eq_zero.2 fun e he => by simp [qSift_tag cut _ false t e he]
  rw [h0, zero_add]
  have hl : ∀ i, ((qSift (prefixOf x i) true t).trace cut).countP (fun e => e.2) ≤ t.depth :=
    fun i => List.countP_le_length.trans (qSift_length cut _ true t)
  rcases (qSift (prefixOf x k) false t).run cut with p | _
  · simp only []
    rcases follow edges p (x.toList.drop k) with _ | ⟨s, c, i⟩
    · simp [Qry.trace]
    · simp only []
      rw [Qry.trace_bind, List.countP_append]
      have := hl (k + i + 1)
      rcases (qSift (prefixOf x (k + i + 1)) true t).run cut with _ | _
      · simp only [Qry.trace_map]
        have := hl (k + i)
        omega
      · simp [Qry.trace]; omega
  · simp [Qry.trace]

theorem qWalkB_asksIn (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qWalkB t edges k x).AsksIn fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := by
  have hs : ∀ i g, k ≤ i → (qSift (prefixOf x i) g t).AsksIn
      fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := fun i g hi =>
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, hi, m, hm, he⟩) _ (qSift_asksIn _ g t)
  unfold qWalkB
  refine Qry.asksIn_bind (fun s0 => ?_) _ (hs k false le_rfl)
  rcases s0 with p | _
  · simp only []
    split
    · trivial
    · refine Qry.asksIn_bind (fun s1 => ?_) _ (hs _ true (by omega))
      rcases s1 with _ | _
      · exact Qry.asksIn_map _ _ (hs _ true (by omega))
      · trivial
  · trivial

/-- The class of the reads an unlearned edge leaves undecided. -/
theorem blocked_spec (k : ℕ) :
    HarvestSpec (fun (t : DTree α) (edges : Edges α) (x : FreeMonoid α) => qWalkB t edges k x)
      Option.toList k where
  asks t edges x := qWalkB_asksIn t edges k x
  first cut t edges x h := by
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 (by simpa using h)
    obtain ⟨hcut, -, pre, blk, htr, hpre, hblk, hmem⟩ := qWalkB_some cut t edges k x b hb
    refine ⟨b, by simp [hb], hcut, ?_⟩
    rw [htr]
    exact first_of_split cut hpre hblk hmem hcut
  form cut t edges x b hb := (qWalkB_some cut t edges k x b (by simpa using hb)).2.1

end Blocked

end OrthoDFA
