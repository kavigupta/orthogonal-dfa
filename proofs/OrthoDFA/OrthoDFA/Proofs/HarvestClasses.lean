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

/-- The ends' classes: the sift of a prefix `w x` of each draw at least `k` long. -/
theorem ends_spec (k : ℕ) (w : FreeMonoid α → FreeMonoid α)
    (hw : ∀ x, ∃ i, k ≤ i ∧ w x = prefixOf x i) :
    HarvestSpec (fun _ o => o = none)
      (fun t (_ : Edges α) x => qSiftDeep (w x) t) Option.toList k where
  asks t _ x := Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => by
      obtain ⟨i, hi, hwx⟩ := hw x
      exact ⟨i, hi, m, hm, by rw [he, hwx]⟩) _ (qSiftDeep_asksIn _ t)
  first R t _ x h := by
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 (by simpa using h)
    obtain ⟨-, hcut, pre, blk, htr, hpre, hblk, hmem⟩ := qSiftDeep_some R.cut _ b t hb
    refine ⟨b, by simp [hb], hcut, ?_⟩
    rw [htr]
    exact first_of_split R.cut hpre hblk hmem hcut
  form R t _ x b hb := by
    obtain ⟨m, rfl⟩ := qSiftDeep_form R.cut _ b t (by simpa using hb)
    obtain ⟨i, hi, hwx⟩ := hw x
    exact ⟨i, hi, m, by rw [hwx]⟩

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
        (qSift (prefixOf x k) false t).trace cut
          ++ (qSift (prefixOf x (k + i + 1)) true t).trace cut,
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
      split_ifs at hwc
      simp at hwc
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
    HarvestSpec (fun _ o => o = none)
      (fun (t : DTree α) (edges : Edges α) (x : FreeMonoid α) => qWalkB t edges k x)
      Option.toList k where
  asks t edges x := qWalkB_asksIn t edges k x
  first R t edges x h := by
    obtain ⟨b, hb⟩ := Option.ne_none_iff_exists'.1 (by simpa using h)
    obtain ⟨hcut, -, pre, blk, htr, hpre, hblk, hmem⟩ := qWalkB_some R.cut t edges k x b hb
    refine ⟨b, by simp [hb], hcut, ?_⟩
    rw [htr]
    exact first_of_split R.cut hpre hblk hmem hcut
  form R t edges x b hb := (qWalkB_some R.cut t edges k x b (by simpa using hb)).2.1

end Blocked

section Probe

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_pair_gt (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .pair j → lo < j
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h; simp at h
    rw [if_pos hlh] at h
    have hl0 : (lo + hi) / 2 - 1 = lo → (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1))
          = some true := fun e => if_pos e
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h hl0
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      all_goals first
        | (have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega)
        | (obtain rfl := Outcome.pair.inj h; omega)
        | (obtain rfl := Outcome.pair.inj h
           have hne : (lo + hi) / 2 - 1 ≠ lo := fun he => by simpa using hl0 he
           omega)
    · have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega

variable (cut : FreeMonoid α → Option Bool)

/-- A search ending at a pair reads the middle beside it first, tagged, after reads that are
tagged or decided. -/
theorem qBracket_pair (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (ps : List (List Bool)) :
    ∀ fuel lo hi j, (qBracket (qAgrees t x walkAt) ps fuel lo hi).run cut = .pair j →
      ∃ b, (t.sift cut (prefixOf x j) = .inr b ∨ t.sift cut (prefixOf x (j + 1)) = .inr b)
        ∧ ∃ pre blk post, (qBracket (qAgrees t x walkAt) ps fuel lo hi).trace cut
            = pre ++ blk ++ post
          ∧ (∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) ∧ (∀ e ∈ blk, e.2 = true)
          ∧ (b, true) ∈ blk
  | 0, _, _, _, h => by simp [qBracket, Qry.run] at h
  | fuel + 1, lo, hi, j, h => by
    by_cases hlh : lo + 1 < hi
    swap
    · rw [qBracket, if_neg hlh] at h; simp [Qry.run] at h
    rw [qBracket_unfold t x walkAt ps fuel lo hi hlh] at h ⊢
    rw [Qry.run_bind] at h
    simp only [Qry.trace_bind]
    have hmid := qGuard_tag cut t x walkAt lo hi ((lo + hi) / 2) true
    have hdm := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2) true
    -- a recursive call, after reads tagged or decided
    have hrec : ∀ (T : List (FreeMonoid α × Bool)) lo' hi',
        (∀ e ∈ T, e.2 = false → cut e.1 ≠ none) →
        (qBracket (qAgrees t x walkAt) ps fuel lo' hi').run cut = .pair j →
        ∃ b, (t.sift cut (prefixOf x j) = .inr b ∨ t.sift cut (prefixOf x (j + 1)) = .inr b)
          ∧ ∃ pre blk post, T ++ (qBracket (qAgrees t x walkAt) ps fuel lo' hi').trace cut
              = pre ++ blk ++ post
            ∧ (∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) ∧ (∀ e ∈ blk, e.2 = true)
            ∧ (b, true) ∈ blk := by
      intro T lo' hi' hT hr
      obtain ⟨b, hb, pre, blk, post, htr, hpre, hblk, hmem⟩ := qBracket_pair t x walkAt ps fuel
        lo' hi' j hr
      refine ⟨b, hb, T ++ pre, blk, post, by rw [htr]; simp, fun e he hg => ?_, hblk, hmem⟩
      rcases List.mem_append.1 he with he | he
      · exact hT e he hg
      · exact hpre e he hg
    rcases hv : (qGuard t x walkAt lo hi ((lo + hi) / 2) true).run cut with _ | v
    · simp only [hv] at h ⊢
      obtain ⟨b, hb, hbm⟩ := qGuard_undecided cut t x walkAt lo hi ((lo + hi) / 2) true hv
      rw [Qry.run_bind] at h
      simp only [Qry.trace_bind]
      have hl := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2 - 1) false
      rcases hlv : (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).run cut with _ | lv
      · simp only [hlv, Qry.run, Outcome.pair.injEq] at h ⊢
        subst h
        refine ⟨b, .inr (by rw [show (lo + hi) / 2 - 1 + 1 = (lo + hi) / 2 by omega]; exact hb),
          [], (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut,
          (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut, by simp [Qry.trace],
          by simp, hmid, hbm⟩
      simp only [hlv] at h ⊢
      rw [Qry.run_bind] at h
      simp only [Qry.trace_bind]
      have hr := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2 + 1) false
      rcases hrv : (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).run cut with _ | rv
      · have hj : j = (lo + hi) / 2 := by
          rcases lv <;> simpa [hrv, Qry.run] using h.symm
        subst hj
        refine ⟨b, .inl hb, [], (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut,
          (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut
            ++ (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).trace cut,
          by rcases lv <;> simp [Qry.trace], by simp, hmid, hbm⟩
      have hT : ∀ e ∈ (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut
          ++ ((qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut
            ++ (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).trace cut),
          e.2 = false → cut e.1 ≠ none := by
        intro e he hg
        rcases List.mem_append.1 he with he | he
        · rw [hmid e he] at hg; exact absurd hg (by simp)
        rcases List.mem_append.1 he with he | he
        · exact hl hlv e he
        · exact hr hrv e he
      rcases lv <;> rcases rv <;> simp only [hrv] at h ⊢
      · obtain ⟨b', hb', pre, blk, post, htr, hpre, hblk, hmem⟩ := hrec _ _ _ hT h
        exact ⟨b', hb', pre, blk, post, by rw [← htr]; simp, hpre, hblk, hmem⟩
      · obtain ⟨b', hb', pre, blk, post, htr, hpre, hblk, hmem⟩ := hrec _ _ _ hT h
        exact ⟨b', hb', pre, blk, post, by rw [← htr]; simp, hpre, hblk, hmem⟩
      · simp [Qry.run] at h
      · obtain ⟨b', hb', pre, blk, post, htr, hpre, hblk, hmem⟩ := hrec _ _ _ hT h
        exact ⟨b', hb', pre, blk, post, by rw [← htr]; simp, hpre, hblk, hmem⟩
    · have hT : ∀ e ∈ (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut,
          e.2 = false → cut e.1 ≠ none := fun e he _ => hdm hv e he
      cases v <;> simp only [hv] at h ⊢ <;> exact hrec _ _ _ hT h

/-- Where a probe's search ends: a triple's or a pair's index. -/
def Outcome.at : Outcome α → Option ℕ
  | .triple j => some j
  | .pair j => some j
  | _ => none

/-- A probe's computation, then the reads of its triple or pair again, untagged. -/
def qProbeH (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    Qry α (Outcome α × Option (FreeMonoid α) × Option (FreeMonoid α)) :=
  (qProbe t edges k x).bind fun o =>
    match o.at with
    | some j =>
      if k ≤ j then (qSift (prefixOf x j) false t).bind fun s =>
        (qSift (prefixOf x (j + 1)) false t).map fun s' => (o, s.getRight?, s'.getRight?)
      else .pure (o, none, none)
    | none => .pure (o, none, none)

theorem qProbeH_run (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α)
    {j : ℕ} (hj : (probeOutcome R t edges k x).at = some j) (hk : k ≤ j) :
    (qProbeH t edges k x).run R.cut
      = (probeOutcome R t edges k x, tripleRead R t x j, tripleRead R t x (j + 1)) := by
  unfold qProbeH
  rw [Qry.run_bind, qProbe_run, hj]
  simp only [if_pos hk]
  rw [Qry.run_bind, Qry.run_map, qSift_run, qSift_run]
  rfl

theorem qProbeH_run_fst (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ)
    (x : FreeMonoid α) : ((qProbeH t edges k x).run R.cut).1 = probeOutcome R t edges k x := by
  unfold qProbeH
  rw [Qry.run_bind, qProbe_run]
  rcases (probeOutcome R t edges k x).at with _ | j
  · rfl
  · simp only []
    split_ifs
    · rw [Qry.run_bind, Qry.run_map]
    · rfl

theorem qProbeH_trace (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    ∃ E, (qProbeH t edges k x).trace cut = (qProbe t edges k x).trace cut ++ E
      ∧ ∀ e ∈ E, e.2 = false := by
  unfold qProbeH
  rw [Qry.trace_bind]
  refine ⟨_, rfl, ?_⟩
  rcases ((qProbe t edges k x).run cut).at with _ | j
  · simp [Qry.trace]
  · simp only []
    split_ifs
    · intro e he
      rw [Qry.trace_bind, Qry.trace_map] at he
      rcases List.mem_append.1 he with he | he
      · exact qSift_tag cut _ false t e he
      · exact qSift_tag cut _ false t e he
    · simp [Qry.trace]

theorem qProbeH_asksIn (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (qProbeH t edges k x).AsksIn fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := by
  have hs : ∀ i g, k ≤ i → (qSift (prefixOf x i) g t).AsksIn
      fun y => ∃ i, k ≤ i ∧ ∃ m ∈ t.mids, y = prefixOf x i * m := fun i g hi =>
    Qry.asksIn_mono (fun y ⟨m, hm, he⟩ => ⟨i, hi, m, hm, he⟩) _ (qSift_asksIn _ g t)
  unfold qProbeH
  refine Qry.asksIn_bind (fun o => ?_) _ (qProbe_asksIn_ge t edges k x)
  rcases o.at with _ | j
  · trivial
  · simp only []
    split_ifs with hk
    · exact Qry.asksIn_bind (fun s => Qry.asksIn_map _ _ (hs _ false (by omega))) _
        (hs _ false hk)
    · trivial

theorem qProbeH_countP (R : CutReads α) (t : DTree α) (edges : Edges α) (k : ℕ)
    (x : FreeMonoid α) :
    ((qProbeH t edges k x).trace R.cut).countP (fun e => e.2) ≤ t.depth * visits R t edges k x := by
  obtain ⟨E, hE, hEt⟩ := qProbeH_trace R.cut t edges k x
  have h0 : E.countP (fun e => e.2) = 0 := List.countP_eq_zero.2 fun e he => by simp [hEt e he]
  rw [hE, List.countP_append, h0, add_zero]
  exact qProbe_countP R t edges k x

/-- A triple's harvest. -/
def harvTriple : Outcome α × Option (FreeMonoid α) × Option (FreeMonoid α) → List (FreeMonoid α)
  | (.triple _, some b, _) => [b]
  | _ => []

/-- A pair's harvest: both its reads. -/
def harvPair : Outcome α × Option (FreeMonoid α) × Option (FreeMonoid α) → List (FreeMonoid α)
  | (.pair _, some b, some b') => [b, b']
  | _ => []

theorem probeOutcome_pair_gt {R : CutReads α} {t : DTree α} {edges : Edges α} {k : ℕ}
    {x : FreeMonoid α} {j : ℕ} (h : probeOutcome R t edges k x = .pair j) : k < j := by
  obtain ⟨ps, hi, -, hb⟩ := probeOutcome_search R h trivial
  exact bracketAt_pair_gt _ ps _ _ _ _ hb

theorem tripleRead_form {R : CutReads α} {t : DTree α} {x b : FreeMonoid α} {j : ℕ}
    (h : tripleRead R t x j = some b) : R.cut b = none ∧ ∃ m, b = prefixOf x j * m := by
  simp only [tripleRead] at h
  rcases hs : t.sift R.cut (prefixOf x j) with _ | b' <;> rw [hs] at h
  · simp at h
  obtain rfl : b' = b := by simpa using h
  exact ⟨(route_undecided R.cut t _ b' hs).2, route_inr_form t _ b' hs⟩

/-- The run of the probe computation, read off its harvest. -/
theorem qProbeH_harv {R : CutReads α} {t : DTree α} {edges : Edges α} {k : ℕ}
    {x : FreeMonoid α} {o : Outcome α} {b₁ b₂ : Option (FreeMonoid α)}
    (h : (qProbeH t edges k x).run R.cut = (o, b₁, b₂)) :
    o = probeOutcome R t edges k x
      ∧ ∀ j, o.at = some j → k ≤ j → b₁ = tripleRead R t x j ∧ b₂ = tripleRead R t x (j + 1) := by
  have h1 := qProbeH_run_fst R t edges k x
  simp only [h] at h1
  refine ⟨h1, fun j hj hk => ?_⟩
  rw [h1] at hj
  have := qProbeH_run R t edges k x hj hk
  rw [h] at this
  simp only [Prod.mk.injEq] at this
  exact ⟨this.2.1, this.2.2⟩

/-- The triples' class. -/
theorem triple_spec (k : ℕ) :
    HarvestSpec (fun _ o => o = none)
      (fun (t : DTree α) (edges : Edges α) (x : FreeMonoid α) => qProbeH t edges k x)
      harvTriple k where
  asks t edges x := qProbeH_asksIn t edges k x
  first R t edges x h := by
    rcases hr : (qProbeH t edges k x).run R.cut with ⟨o, b₁, b₂⟩
    rw [hr] at h
    rcases o <;> rcases b₁ with _ | b <;> simp [harvTriple] at h ⊢
    rename_i j
    obtain ⟨ho, hb⟩ := qProbeH_harv hr
    have hk := probeOutcome_triple_gt ho.symm
    have hb1 := (hb j rfl hk.le).1
    obtain ⟨hcut, hfirst⟩ := triple_first_read R ho.symm hb1.symm
    obtain ⟨E, hE, -⟩ := qProbeH_trace R.cut t edges k x
    refine ⟨hcut, ?_⟩
    rw [hE]
    exact first_append hfirst
  form R t edges x b hb := by
    rcases hr : (qProbeH t edges k x).run R.cut with ⟨o, b₁, b₂⟩
    rw [hr] at hb
    rcases o <;> rcases b₁ with _ | b' <;> simp [harvTriple] at hb
    rename_i j
    subst hb
    obtain ⟨ho, hbb⟩ := qProbeH_harv hr
    have hk := probeOutcome_triple_gt ho.symm
    obtain ⟨-, m, hm⟩ := tripleRead_form (hbb j rfl hk.le).1.symm
    exact ⟨j, hk.le, m, hm⟩

/-- The pairs' class. -/
theorem pair_spec (k : ℕ) :
    HarvestSpec (fun _ o => o = none)
      (fun (t : DTree α) (edges : Edges α) (x : FreeMonoid α) => qProbeH t edges k x)
      harvPair k where
  asks t edges x := qProbeH_asksIn t edges k x
  first R t edges x h := by
    rcases hr : (qProbeH t edges k x).run R.cut with ⟨o, b₁, b₂⟩
    rw [hr] at h
    rcases o <;> rcases b₁ with _ | b₁ <;> rcases b₂ with _ | b₂ <;> simp [harvPair] at h ⊢
    rename_i j
    obtain ⟨ho, hb⟩ := qProbeH_harv hr
    have hk := probeOutcome_pair_gt ho.symm
    obtain ⟨hb1, hb2⟩ := hb j rfl hk.le
    obtain ⟨ps, hi, hw, hbr⟩ := probeOutcome_search R ho.symm trivial
    set walkAt : ℕ → List Bool := fun j => ps.getD (j - k) []
    have hbq : (qBracket (qAgrees t x walkAt) ps (hi - k) k hi).run R.cut = .pair j := by
      rw [qBracket_run R.cut _ _ (qAgrees_run R t x walkAt)]; exact hbr
    obtain ⟨b, hbs, pre, blk, post, htr, hpre, hblk, hmem⟩ :=
      qBracket_pair R.cut t x walkAt ps (hi - k) k hi j hbq
    have hLeq : (qProbe t edges k x).trace R.cut = (qWalk t edges k x).trace R.cut
        ++ (qBracket (qAgrees t x walkAt) ps (hi - k) k hi).trace R.cut := by
      rw [qProbe, Qry.trace_bind, qWalk_run, hw]
      rfl
    have hwd := (qWalk_facts R t edges k x).2 (by simp [hw])
    obtain ⟨E, hE, -⟩ := qProbeH_trace R.cut t edges k x
    have hcut : R.cut b = none := by
      rcases hbs with hbs | hbs
      · exact (route_undecided R.cut t _ b hbs).2
      · exact (route_undecided R.cut t _ b hbs).2
    have hsplit : (qProbeH t edges k x).trace R.cut
        = ((qWalk t edges k x).trace R.cut ++ pre) ++ blk ++ (post ++ E) := by
      rw [hE, hLeq, htr]; simp
    have hfirst := first_of_split R.cut (pre := (qWalk t edges k x).trace R.cut ++ pre)
      (post := post ++ E) (fun e he hg => by
        rcases List.mem_append.1 he with he | he
        · exact hwd e he
        · exact hpre e he hg) hblk hmem hcut
    rw [← hsplit] at hfirst
    have hbin : b = b₁ ∨ b = b₂ := by
      rcases hbs with hbs | hbs
      · left; simpa [tripleRead, hbs, eq_comm] using hb1
      · right; simpa [tripleRead, hbs, eq_comm] using hb2
    rcases hbin with rfl | rfl
    · exact .inl ⟨hcut, hfirst⟩
    · exact .inr ⟨hcut, hfirst⟩
  form R t edges x b hb := by
    rcases hr : (qProbeH t edges k x).run R.cut with ⟨o, b₁, b₂⟩
    rw [hr] at hb
    rcases o <;> rcases b₁ with _ | b₁ <;> rcases b₂ with _ | b₂ <;> simp [harvPair] at hb
    rename_i j
    obtain ⟨ho, hbb⟩ := qProbeH_harv hr
    have hk := probeOutcome_pair_gt ho.symm
    obtain ⟨hb1, hb2⟩ := hbb j rfl hk.le
    rcases hb with rfl | rfl
    · obtain ⟨-, m, hm⟩ := tripleRead_form hb1.symm
      exact ⟨j, hk.le, m, hm⟩
    · obtain ⟨-, m, hm⟩ := tripleRead_form hb2.symm
      exact ⟨j + 1, by omega, m, hm⟩

end Probe

end OrthoDFA
