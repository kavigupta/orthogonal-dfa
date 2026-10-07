import OrthoDFA.Stage

/-!
# The pass's dichotomy

Every quiet probe is accounted for (`probeStep_accounted`): the fall-throughs of
`_act_on_disagreement` other than a missing edge and the evidence's no split never fire, since a
sift is a function of the string and the edges are re-voted after every probe.  A missing edge is
one none of its leaf's members can place, and `settle` harvested those.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

variable {cut : FreeMonoid α → Option Bool}

theorem sift_mem_paths : ∀ (t : DTree α) {x : FreeMonoid α} {p : List Bool},
    t.sift cut x = .inl p → p ∈ t.paths
  | .leaf, x, p, h => by simp [sift] at h; simp [paths, ← h]
  | .node m r a, x, p, h => by
    simp only [sift] at h
    split at h
    · simp at h
    · rcases ha : a.sift cut x with q | b <;> rw [ha] at h <;> simp at h
      subst h; simp [paths, sift_mem_paths a ha]
    · rcases hr : r.sift cut x with q | b <;> rw [hr] at h <;> simp at h
      subst h; simp [paths, sift_mem_paths r hr]

theorem firstDisagreement_some : ∀ (t : DTree α) {x y pre d : FreeMonoid α},
    t.firstDisagreement cut x y pre = some d →
      ∃ a a', cut (x * d) = some a ∧ cut (y * d) = some a' ∧ a ≠ a'
  | .leaf, x, y, pre, d, h => by simp [firstDisagreement] at h
  | .node m r a, x, y, pre, d, h => by
    simp only [firstDisagreement] at h
    rcases hcx : cut (x * (pre * m)) with _ | bx
    · simp [hcx] at h
    rcases hcy : cut (y * (pre * m)) with _ | by'
    · simp [hcx, hcy] at h
    simp only [hcx, hcy] at h
    cases bx <;> cases by' <;> simp only at h
    · exact firstDisagreement_some r h
    · simp only [Option.some.injEq] at h; subst h; exact ⟨false, true, hcx, hcy, by simp⟩
    · simp only [Option.some.injEq] at h; subst h; exact ⟨true, false, hcx, hcy, by simp⟩
    · exact firstDisagreement_some a h

theorem firstDisagreement_isSome : ∀ (t : DTree α) {x y pre : FreeMonoid α} {p q : List Bool},
    t.sift cut (x * pre) = .inl p → t.sift cut (y * pre) = .inl q → p ≠ q →
      ∃ d, t.firstDisagreement cut x y pre = some d
  | .leaf, x, y, pre, p, q, hx, hy, hne => by
    simp [sift] at hx hy; exact absurd (hx.trans hy.symm) hne
  | .node m r a, x, y, pre, p, q, hx, hy, hne => by
    simp only [sift, mul_assoc] at hx hy
    simp only [firstDisagreement]
    rcases hcx : cut (x * (pre * m)) with _ | bx <;> rw [hcx] at hx
    · simp at hx
    rcases hcy : cut (y * (pre * m)) with _ | by' <;> rw [hcy] at hy
    · simp at hy
    cases bx <;> cases by' <;> simp only
    · rcases hrx : r.sift cut (x * pre) with p' | _ <;> rw [hrx] at hx <;> simp at hx
      rcases hry : r.sift cut (y * pre) with q' | _ <;> rw [hry] at hy <;> simp at hy
      subst hx hy
      exact firstDisagreement_isSome r hrx hry (fun h => hne (by rw [h]))
    · exact ⟨_, rfl⟩
    · exact ⟨_, rfl⟩
    · rcases hax : a.sift cut (x * pre) with p' | _ <;> rw [hax] at hx <;> simp at hx
      rcases hay : a.sift cut (y * pre) with q' | _ <;> rw [hay] at hy <;> simp at hy
      subst hx hy
      exact firstDisagreement_isSome a hax hay (fun h => hne (by rw [h]))

theorem sift_splitAt_of_ne (d : FreeMonoid α) : ∀ (t : DTree α) (s1 : List Bool)
    {x : FreeMonoid α} {p : List Bool},
    t.sift cut x = .inl p → p ≠ s1 → (t.splitAt d s1).sift cut x = .inl p
  | .leaf, [], x, p, h, hne => by simp [sift] at h; exact absurd h hne
  | .leaf, _ :: _, x, p, h, hne => by simpa [splitAt] using h
  | .node m r a, [], x, p, h, hne => by simpa [splitAt] using h
  | .node m r a, b :: s1, x, p, h, hne => by
    rcases hc : cut (x * m) with _ | c
    · simp [sift, hc] at h
    cases b <;> cases c <;> simp only [splitAt, sift, hc] at h ⊢
    · rcases hr : r.sift cut x with q | _ <;> rw [hr] at h <;> simp at h
      subst h
      rw [sift_splitAt_of_ne d r s1 hr (fun h' => hne (by rw [h']))]
      rfl
    · exact h
    · exact h
    · rcases ha : a.sift cut x with q | _ <;> rw [ha] at h <;> simp at h
      subst h
      rw [sift_splitAt_of_ne d a s1 ha (fun h' => hne (by rw [h']))]
      rfl

theorem sift_splitAt_self (d : FreeMonoid α) : ∀ (t : DTree α) (s1 : List Bool)
    {x : FreeMonoid α} {b : Bool},
    t.sift cut x = .inl s1 → cut (x * d) = some b →
      (t.splitAt d s1).sift cut x = .inl (s1 ++ [b])
  | .leaf, [], x, b, h, hb => by cases b <;> simp [splitAt, sift, hb]
  | .leaf, _ :: _, x, b, h, hb => by simp [sift] at h
  | .node m r a, [], x, b, h, hb => by
    rcases hc : cut (x * m) with _ | c
    · simp [sift, hc] at h
    cases c
    · simp only [sift, hc] at h
      rcases hr : r.sift cut x with q | _ <;> rw [hr] at h <;> simp at h
    · simp only [sift, hc] at h
      rcases ha : a.sift cut x with q | _ <;> rw [ha] at h <;> simp at h
  | .node m r a, c :: s1, x, b, h, hb => by
    rcases hc : cut (x * m) with _ | c'
    · simp [sift, hc] at h
    cases c'
    · simp only [sift, hc] at h
      rcases hr : r.sift cut x with q | _ <;> rw [hr] at h <;> simp at h
      obtain ⟨rfl, rfl⟩ := h
      simp only [splitAt, sift, hc]
      rw [sift_splitAt_self d r _ hr hb]
      rfl
    · simp only [sift, hc] at h
      rcases ha : a.sift cut x with q | _ <;> rw [ha] at h <;> simp at h
      obtain ⟨rfl, rfl⟩ := h
      simp only [splitAt, sift, hc]
      rw [sift_splitAt_self d a _ ha hb]
      rfl

omit [Fintype α] [DecidableEq α] in
theorem mem_paths_splitAt (d : FreeMonoid α) : ∀ (t : DTree α) (s1 p : List Bool),
    p ∈ (t.splitAt d s1).paths → (p ∈ t.paths ∧ p ≠ s1) ∨ ∃ b : Bool, p = s1 ++ [b]
  | .leaf, [], p, h => by
    right; simp [splitAt, paths] at h
    rcases h with rfl | rfl
    · exact ⟨false, rfl⟩
    · exact ⟨true, rfl⟩
  | .leaf, _ :: _, p, h => by
    left; simp [splitAt, paths] at h; subst h; simp [paths]
  | .node m r a, [], p, h => by
    left; simp only [splitAt] at h
    refine ⟨h, ?_⟩
    rintro rfl
    simp [paths] at h
  | .node m r a, false :: s1, p, h => by
    simp only [splitAt, paths, List.mem_append, List.mem_map] at h
    rcases h with ⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩
    · rcases mem_paths_splitAt d r s1 q hq with ⟨hq', hne⟩ | ⟨b, rfl⟩
      · left; simp [paths, hq', hne]
      · right; exact ⟨b, rfl⟩
    · left; simp [paths, hq]
  | .node m r a, true :: s1, p, h => by
    simp only [splitAt, paths, List.mem_append, List.mem_map] at h
    rcases h with ⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩
    · left; simp [paths, hq]
    · rcases mem_paths_splitAt d a s1 q hq with ⟨hq', hne⟩ | ⟨b, rfl⟩
      · left; simp [paths, hq', hne]
      · right; exact ⟨b, rfl⟩

end DTree

omit [Fintype α] [DecidableEq α] in
theorem prefixOf_length (w : FreeMonoid α) : prefixOf w w.toList.length = w := by
  simp [prefixOf]

theorem prefixOf_succ {w : FreeMonoid α} {i : ℕ} {c : α} (h : w.toList[i]? = some c) :
    prefixOf w i * FreeMonoid.of c = prefixOf w (i + 1) := by
  apply FreeMonoid.toList.injective
  simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList, FreeMonoid.toList_of,
    List.take_add_one, h, Option.toList_some]

section Pass

variable (K : StageKnobs α) (R : CutReads α)

/-- Every edge's witness sifts to its leaf, and to its target once extended by the letter; every
member of a leaf whose edge is open, extended by the letter, is harvested undecided. -/
def Settled (s : PassState α) : Prop :=
  (∀ p c q x, s.edges p c = some (q, x) →
      s.tree.sift R.cut x = .inl p ∧ s.tree.sift R.cut (x * FreeMonoid.of c) = .inl q)
    ∧ ∀ p c, s.edges p c = none → ∀ m ∈ members K R s.tree s.pool p, ∃ b,
      s.tree.sift R.cut (m * FreeMonoid.of c) = .inr b ∧ b ∈ s.boundary

/-- Every leaf holds a string of the population. -/
def Populated (s : PassState α) : Prop :=
  ∀ p ∈ s.tree.paths, ∃ x ∈ s.pool, s.tree.sift R.cut x = .inl p

theorem mem_members {t : DTree α} {pool : List (FreeMonoid α)} {p : List Bool}
    {m : FreeMonoid α} (h : m ∈ members K R t pool p) : m ∈ pool ∧ t.sift R.cut m = .inl p := by
  simp only [members] at h
  have := List.mem_of_mem_take h
  simpa using this

theorem members_ne_nil (hK : 0 < K.memberLimit) {t : DTree α} {pool : List (FreeMonoid α)}
    {p : List Bool} {x : FreeMonoid α} (hx : x ∈ pool) (hs : t.sift R.cut x = .inl p) :
    members K R t pool p ≠ [] := by
  simp only [members, ne_eq, List.take_eq_nil_iff, not_or]
  refine ⟨by omega, ?_⟩
  rw [← ne_eq, List.ne_nil_iff_exists_cons]
  have : x ∈ pool.filter fun s => decide (t.sift R.cut s = .inl p) := by simp [hx, hs]
  obtain ⟨l₁, l₂, h⟩ := List.append_of_mem this
  cases l₁ with
  | nil => exact ⟨x, l₂, h⟩
  | cons y l₁ => exact ⟨y, l₁ ++ x :: l₂, h⟩

omit [Fintype α] [DecidableEq α] in
/-- A fold that only ever keeps its best so far or takes the next ends at one of them, or at
none only when it started there with nothing to take. -/
theorem foldl_keep_or_take {β : Type*} (f : Option β → β → Option β)
    (hf : ∀ best v, f best v = some v ∨ ∃ b, best = some b ∧ f best v = some b) :
    ∀ (l : List β) (init : Option β), (l.foldl f init = none → init = none ∧ l = [])
      ∧ ∀ v, l.foldl f init = some v → init = some v ∨ v ∈ l
  | [], init => ⟨fun h => ⟨h, rfl⟩, fun v h => Or.inl h⟩
  | u :: us, init => by
    have ih := foldl_keep_or_take f hf us (f init u)
    simp only [List.foldl_cons]
    refine ⟨fun h => ?_, fun v h => ?_⟩
    · have := (ih.1 h).1
      rcases hf init u with h' | ⟨b, -, h'⟩ <;> rw [h'] at this <;> simp at this
    · rcases ih.2 v h with h' | h'
      · rcases hf init u with h'' | ⟨b, rfl, h''⟩ <;> rw [h''] at h' <;>
          simp only [Option.some.injEq] at h'
        · exact Or.inr (by simp [h'])
        · exact Or.inl (by rw [h'])
      · exact Or.inr (List.mem_cons_of_mem _ h')

theorem decisiveTarget_some {t : DTree α} {pool : List (FreeMonoid α)} {p : List Bool} {c : α}
    {cur : Option (List Bool)} {q : List Bool} {x : FreeMonoid α}
    (h : decisiveTarget K R t pool p c cur = some (q, x)) :
    x ∈ members K R t pool p ∧ t.sift R.cut (x * FreeMonoid.of c) = .inl q := by
  simp only [decisiveTarget] at h
  rcases (foldl_keep_or_take _ (fun best v => by
      cases best with
      | none => simp
      | some b => simp only; split_ifs <;> simp) _ none).2 _ h with h' | h'
  · simp at h'
  · rw [List.mem_filterMap] at h'
    obtain ⟨m, hm, hv⟩ := h'
    split at hv
    · rename_i p' hs
      simp only [Option.some.injEq, Prod.mk.injEq] at hv
      obtain ⟨rfl, rfl⟩ := hv
      exact ⟨hm, hs⟩
    · simp at hv

theorem decisiveTarget_none {t : DTree α} {pool : List (FreeMonoid α)} {p : List Bool} {c : α}
    {cur : Option (List Bool)} (h : decisiveTarget K R t pool p c cur = none) :
    ∀ m ∈ members K R t pool p, ∃ b, t.sift R.cut (m * FreeMonoid.of c) = .inr b := by
  simp only [decisiveTarget] at h
  have hv := ((foldl_keep_or_take _ (fun best v => by
      cases best with
      | none => simp
      | some b => simp only; split_ifs <;> simp) _ none).1 h).2
  intro m hm
  have := List.filterMap_eq_nil_iff.mp hv m hm
  rcases hs : t.sift R.cut (m * FreeMonoid.of c) with p' | b
  · rw [hs] at this; simp at this
  · exact ⟨b, rfl⟩

theorem mem_edgeMisses {t : DTree α} {c : α} :
    ∀ {ms : List (FreeMonoid α)}, (∀ m ∈ ms, ∃ b, t.sift R.cut (m * FreeMonoid.of c) = .inr b) →
      ∀ m ∈ ms, ∃ b, t.sift R.cut (m * FreeMonoid.of c) = .inr b ∧ b ∈ edgeMisses R t c ms
  | [], _, m, hm => by simp at hm
  | m' :: ms, h, m, hm => by
    obtain ⟨b', hb'⟩ := h m' (by simp)
    have ih := mem_edgeMisses (ms := ms) (fun m hm => h m (by simp [hm]))
    simp only [edgeMisses, hb']
    rcases List.mem_cons.mp hm with rfl | hm
    · exact ⟨b', hb', by simp⟩
    · obtain ⟨b, hb, hmem⟩ := ih m hm
      exact ⟨b, hb, by simp [hmem]⟩

/-- `settle` leaves the pass settled, whatever edges it re-votes, as long as their witnesses
already sift as they should. -/
theorem settle_settled {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : List Bool → α → Option (List Bool × FreeMonoid α)} {st un : ℕ}
    {bd dg : List (FreeMonoid α)}
    (hE : ∀ p c q x, edges p c = some (q, x) →
      t.sift R.cut x = .inl p ∧ t.sift R.cut (x * FreeMonoid.of c) = .inl q) :
    Settled K R (settle K R t pool edges st un bd dg) := by
  refine ⟨fun p c q x h => ?_, fun p c h m hm => ?_⟩
  · simp only [settle, closeEdges] at h
    rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', x'⟩
    · rw [hd] at h; exact hE p c q x h
    · rw [hd] at h
      simp only [Option.orElse_some, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hm, hs⟩ := decisiveTarget_some K R hd
      exact ⟨(mem_members K R hm).2, hs⟩
  · simp only [settle, closeEdges] at h hm ⊢
    rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', x'⟩
    · obtain ⟨b, hb, hmem⟩ := mem_edgeMisses R (decisiveTarget_none K R hd) m hm
      refine ⟨b, hb, List.mem_append_right _ ?_⟩
      simp only [List.mem_flatMap, Finset.mem_toList, Finset.mem_univ, true_and]
      exact ⟨p, DTree.sift_mem_paths t (mem_members K R hm).2, c, hmem⟩
    · rw [hd] at h; simp at h

theorem firstDisagreeingEdge_inl (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ∀ (fuel lo hi : ℕ) (b : FreeMonoid α), firstDisagreeingEdge R t w walk fuel lo hi = .inl b →
      ∃ i, lo < i ∧ i < hi ∧ t.sift R.cut (prefixOf w i) = .inr b
  | 0, lo, hi, b, h => by simp [firstDisagreeingEdge] at h
  | fuel + 1, lo, hi, b, h => by
    simp only [firstDisagreeingEdge] at h
    split_ifs at h with hlt
    · rcases hs : t.sift R.cut (prefixOf w ((lo + hi) / 2)) with p | b'
      · simp only [hs] at h
        split_ifs at h
        · obtain ⟨i, h1, h2, h3⟩ := firstDisagreeingEdge_inl t w walk fuel _ hi b h
          exact ⟨i, by omega, h2, h3⟩
        · obtain ⟨i, h1, h2, h3⟩ := firstDisagreeingEdge_inl t w walk fuel lo _ b h
          exact ⟨i, h1, by omega, h3⟩
      · simp only [hs, Sum.inl.injEq] at h; subst h; exact ⟨_, by omega, by omega, hs⟩

theorem firstDisagreeingEdge_inr (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ∀ (fuel lo hi fd : ℕ), lo < hi → hi ≤ lo + fuel + 1 →
      t.sift R.cut (prefixOf w lo) = .inl (walk lo) →
      (∃ q, t.sift R.cut (prefixOf w hi) = .inl q ∧ q ≠ walk hi) →
      firstDisagreeingEdge R t w walk fuel lo hi = .inr fd →
      lo < fd ∧ fd ≤ hi ∧ t.sift R.cut (prefixOf w (fd - 1)) = .inl (walk (fd - 1))
        ∧ ∃ q, t.sift R.cut (prefixOf w fd) = .inl q ∧ q ≠ walk fd
  | 0, lo, hi, fd, hlt, hle, hlo, hhi, h => by
    simp only [firstDisagreeingEdge, Sum.inr.injEq] at h; subst h
    have : hi - 1 = lo := by omega
    rw [this]; exact ⟨hlt, le_rfl, hlo, hhi⟩
  | fuel + 1, lo, hi, fd, hlt, hle, hlo, hhi, h => by
    simp only [firstDisagreeingEdge] at h
    split_ifs at h with hlt'
    · rcases hs : t.sift R.cut (prefixOf w ((lo + hi) / 2)) with p | b'
      · simp only [hs] at h
        split_ifs at h with hp
        · have := firstDisagreeingEdge_inr t w walk fuel _ hi fd (by omega) (by omega)
            (by rw [hs, hp]) hhi h
          exact ⟨by omega, this.2.1, this.2.2⟩
        · have := firstDisagreeingEdge_inr t w walk fuel lo _ fd (by omega) (by omega) hlo
            ⟨p, hs, hp⟩ h
          exact ⟨this.1, by omega, this.2.2.1, this.2.2.2⟩
      · simp [hs] at h
    · simp only [Sum.inr.injEq] at h; subst h
      have : hi - 1 = lo := by omega
      rw [this]; exact ⟨hlt, le_rfl, hlo, hhi⟩

theorem anchoredWalk_some {t : DTree α} {edges : List Bool → α → Option (List Bool × FreeMonoid α)}
    {w : FreeMonoid α} {start : ℕ} {walk : List (List Bool)}
    (h : anchoredWalk R t edges w = some (start, walk)) :
    start ≤ w.toList.length ∧ ∃ p, t.sift R.cut (prefixOf w start) = .inl p
      ∧ walk = (w.toList.drop start).scanl (stepPath t edges) p := by
  simp only [anchoredWalk] at h
  split at h
  · simp at h
  · rename_i i p hf
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain ⟨j, hj, hjf⟩ := List.exists_of_findSome?_eq_some hf
    rcases hs : t.sift R.cut (prefixOf w j) with p' | _ <;> rw [hs] at hjf <;> simp at hjf
    obtain ⟨rfl, rfl⟩ := hjf
    exact ⟨by simpa [Nat.lt_succ_iff] using hj, p', hs, rfl⟩

theorem anchoredWalk_none {t : DTree α} {edges : List Bool → α → Option (List Bool × FreeMonoid α)}
    {w : FreeMonoid α} (h : anchoredWalk R t edges w = none) :
    ∃ b, t.sift R.cut (prefixOf w 0) = .inr b ∧ b ∈ anchorMisses R t w := by
  simp only [anchoredWalk] at h
  split at h
  · rename_i hf
    have h0 := List.findSome?_eq_none_iff.mp hf 0 (by simp)
    rcases hs : t.sift R.cut (prefixOf w 0) with p | b
    · rw [hs] at h0; simp at h0
    · refine ⟨b, rfl, ?_⟩
      simp [anchorMisses, List.range_succ_eq_map, hs, List.takeWhile_cons]
  · simp at h

omit [Fintype α] [DecidableEq α] in
theorem getD_scanl_succ {β γ : Type*} {g : β → γ → β} {p : β} {L : List γ} {k : ℕ} {c : γ}
    {d : β} (hc : L[k]? = some c) :
    (L.scanl g p).getD (k + 1) d = g ((L.scanl g p).getD k d) c := by
  have hk : k < L.length := (List.getElem?_eq_some_iff.mp hc).1
  obtain ⟨a, ha⟩ : ∃ a, (L.scanl g p)[k]? = some a :=
    ⟨_, List.getElem?_eq_getElem (by rw [List.length_scanl]; omega)⟩
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_succ_scanl, ha, hc]
  rfl

omit [Fintype α] [DecidableEq α] in
theorem walk_succ {t : DTree α} {edges : List Bool → α → Option (List Bool × FreeMonoid α)}
    {w : FreeMonoid α} {start j : ℕ} {p : List Bool} {c : α} (hj : start ≤ j)
    (hc : w.toList[j]? = some c) :
    ((w.toList.drop start).scanl (stepPath t edges) p).getD (j + 1 - start) []
      = stepPath t edges (((w.toList.drop start).scanl (stepPath t edges) p).getD (j - start) [])
        c := by
  rw [show j + 1 - start = (j - start) + 1 by omega]
  apply getD_scanl_succ
  rw [List.getElem?_drop, show start + (j - start) = j by omega, hc]

theorem mem_settle_boundary {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : List Bool → α → Option (List Bool × FreeMonoid α)} {st un : ℕ}
    {bd dg : List (FreeMonoid α)} {b : FreeMonoid α} (h : b ∈ bd) :
    b ∈ (settle K R t pool edges st un bd dg).boundary :=
  List.mem_append_left _ h

/-- What a probe leaves the pass at: a settled state whose edges' witnesses sift as they should
and whose leaves each hold a string of its population. -/
theorem onEdge_shape (s : PassState α) (w : FreeMonoid α) (walkAt : ℕ → List Bool)
    (pool boundary : List (FreeMonoid α)) (fd : ℕ) (hS : Settled K R s) (hP : Populated R s)
    (hpool : ∀ x ∈ s.pool, x ∈ pool) :
    ∃ t pool' edges st un dg, onEdge K R s w walkAt pool boundary fd
        = settle K R t pool' edges st un boundary dg
      ∧ (st = s.streak + 1 ∨ st = 0)
      ∧ (∀ p c q x, edges p c = some (q, x) →
        t.sift R.cut x = .inl p ∧ t.sift R.cut (x * FreeMonoid.of c) = .inl q)
      ∧ ∀ p ∈ t.paths, ∃ x ∈ pool', t.sift R.cut x = .inl p := by
  have hpop : ∀ pool' : List (FreeMonoid α), (∀ x ∈ pool, x ∈ pool') →
      ∀ p ∈ s.tree.paths, ∃ x ∈ pool', s.tree.sift R.cut x = .inl p := by
    intro pool' h p hp
    obtain ⟨x, hx, hs⟩ := hP p hp
    exact ⟨x, h x (hpool x hx), hs⟩
  have hclean := hpop pool (fun x hx => hx)
  simp only [onEdge]
  rcases hc : w.toList[fd - 1]? with _ | c
  · exact ⟨_, _, _, _, _, _, rfl, Or.inl rfl, hS.1, hclean⟩
  simp only
  rcases he : s.edges (walkAt (fd - 1)) c with _ | ⟨s2, x⟩
  · exact ⟨_, _, _, _, _, _, rfl, Or.inl rfl, hS.1, hclean⟩
  simp only
  split_ifs with hcond
  · exact ⟨_, _, _, _, _, _, rfl, Or.inl rfl, hS.1, hclean⟩
  simp only [ne_eq, not_or, not_not] at hcond
  obtain ⟨-, hx, hsp⟩ := hcond
  rcases hd : s.tree.firstDisagreement R.cut x (prefixOf w (fd - 1)) (FreeMonoid.of c) with _ | d
  · exact ⟨_, _, _, _, _, _, rfl, Or.inl rfl, hS.1, hclean⟩
  simp only
  have hkept : ∀ y ∈ pool, y ∈ prefixOf w (fd - 1) :: pool.filter (· ≠ prefixOf w (fd - 1)) := by
    intro y hy
    by_cases h : y = prefixOf w (fd - 1)
    · simp [h]
    · simp [hy, h]
  rcases hv : verdict K R s.tree pool (walkAt (fd - 1)) d
      (s.tree.paths.length * Fintype.card α) with _ | _ | _ <;> simp only
  · refine ⟨_, _, _, _, _, _, rfl, Or.inr rfl, ?_, ?_⟩
    · intro p c' q y h
      rcases hpe : s.edges p c' with _ | ⟨q', y'⟩ <;> rw [hpe] at h
      · simp at h
      · simp only at h
        split_ifs at h with hne
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        simp only [not_or] at hne
        obtain ⟨h1, h2⟩ := hS.1 p c' _ _ hpe
        exact ⟨DTree.sift_splitAt_of_ne d _ _ h1 hne.1,
          DTree.sift_splitAt_of_ne d _ _ h2 hne.2⟩
    · intro p hp
      have hmem : ∀ y ∈ pool, y ∈ pool ++ [x, prefixOf w (fd - 1)].filter (· ∉ pool) :=
        fun y hy => List.mem_append_left _ hy
      have hnew : ∀ y ∈ [x, prefixOf w (fd - 1)],
          y ∈ pool ++ [x, prefixOf w (fd - 1)].filter (· ∉ pool) := by
        intro y hy
        by_cases h : y ∈ pool
        · exact List.mem_append_left _ h
        · exact List.mem_append_right _ (List.mem_filter.mpr ⟨hy, by simpa using h⟩)
      rcases DTree.mem_paths_splitAt d _ _ _ hp with ⟨hp', hne⟩ | ⟨b, rfl⟩
      · obtain ⟨y, hy, hs⟩ := hP p hp'
        exact ⟨y, hmem y (hpool y hy), DTree.sift_splitAt_of_ne d _ _ hs hne⟩
      · obtain ⟨a, a', ha, ha', hne⟩ := DTree.firstDisagreement_some _ hd
        by_cases hb : a = b
        · subst hb
          exact ⟨x, hnew x (by simp), DTree.sift_splitAt_self d _ _ hx ha⟩
        · have : a' = b := by cases a <;> cases a' <;> cases b <;> simp_all
          subst this
          exact ⟨_, hnew _ (by simp), DTree.sift_splitAt_self d _ _ hsp ha'⟩
  · exact ⟨_, _, _, _, _, _, rfl, Or.inl rfl, hS.1, hpop _ hkept⟩
  · exact ⟨_, _, _, _, _, _, rfl, Or.inr rfl, hS.1, hpop _ hkept⟩

theorem probeStep_shape (s : PassState α) (w : FreeMonoid α) (hS : Settled K R s)
    (hP : Populated R s) :
    ∃ t pool' edges st un bd dg, probeStep K R s w = settle K R t pool' edges st un bd dg
      ∧ (st = s.streak + 1 ∨ st = 0) ∧ (∀ b ∈ s.boundary, b ∈ bd)
      ∧ (∀ p c q x, edges p c = some (q, x) →
        t.sift R.cut x = .inl p ∧ t.sift R.cut (x * FreeMonoid.of c) = .inl q)
      ∧ ∀ p ∈ t.paths, ∃ x ∈ pool', t.sift R.cut x = .inl p := by
  simp only [probeStep]
  rcases hw : anchoredWalk R s.tree s.edges w with _ | ⟨start, walk⟩
  · exact ⟨_, _, _, _, _, _, _, rfl, Or.inl rfl, fun b hb => List.mem_append_left _ hb, hS.1, hP⟩
  simp only
  have hpool : ∀ x ∈ s.pool,
      x ∈ (if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]) := by
    intro x hx; split_ifs <;> simp [hx]
  have hpop : ∀ p ∈ s.tree.paths, ∃ x ∈ (if prefixOf w start ∈ s.pool then s.pool
      else s.pool ++ [prefixOf w start]), s.tree.sift R.cut x = .inl p := by
    intro p hp
    obtain ⟨x, hx, hs⟩ := hP p hp
    exact ⟨x, hpool x hx, hs⟩
  generalize (if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]) = pool
    at hpool hpop ⊢
  rcases hsw : s.tree.sift R.cut w with actual | b
  · simp only
    by_cases hact : actual = walk.getD (w.toList.length - start) []
    · simp only [hact, ↓reduceIte]
      exact ⟨_, _, _, _, _, _, _, rfl, Or.inl rfl, fun b hb => List.mem_append_left _ hb, hS.1,
        hpop⟩
    simp only [hact, ↓reduceIte]
    rcases firstDisagreeingEdge R s.tree w (fun j => walk.getD (j - start) [])
        w.toList.length start w.toList.length with b | fd <;> simp only
    · exact ⟨_, _, _, _, _, _, _, rfl, Or.inl rfl, fun b hb => by simp [hb], hS.1, hpop⟩
    · obtain ⟨t, pool', edges, st, un, dg, heq, hst, hE, hP'⟩ :=
        onEdge_shape K R s w _ _ (s.boundary ++ anchorMisses R s.tree w) fd hS hP hpool
      exact ⟨t, pool', edges, st, un, _, dg, heq, hst,
        fun b hb => List.mem_append_left _ hb, hE, hP'⟩
  · exact ⟨_, _, _, _, _, _, _, rfl, Or.inl rfl, fun b hb => by simp [hb], hS.1, hpop⟩

/-- A probe leaves the pass settled and populated, with its streak one longer or reset, and
its harvest only grown. -/
theorem probeStep_inv (s : PassState α) (w : FreeMonoid α) (hS : Settled K R s)
    (hP : Populated R s) :
    Settled K R (probeStep K R s w) ∧ Populated R (probeStep K R s w)
      ∧ ((probeStep K R s w).streak = s.streak + 1 ∨ (probeStep K R s w).streak = 0)
      ∧ ∀ b ∈ s.boundary, b ∈ (probeStep K R s w).boundary := by
  obtain ⟨t, pool', edges, st, un, bd, dg, heq, hst, hbd, hE, hP'⟩ := probeStep_shape K R s w hS hP
  rw [heq]
  exact ⟨settle_settled K R hE, hP', hst, fun b hb => mem_settle_boundary K R (hbd b hb)⟩

theorem onEdge_accounted (hK : 0 < K.memberLimit) (s : PassState α) (w : FreeMonoid α)
    (hS : Settled K R s) (hP : Populated R s) {start : ℕ} {walk : List (List Bool)}
    (hw : anchoredWalk R s.tree s.edges w = some (start, walk))
    (pool boundary : List (FreeMonoid α)) (hbd : ∀ b ∈ s.boundary, b ∈ boundary) {fd : ℕ}
    (hlt : start < fd)
    (hle : fd ≤ w.toList.length)
    (hs1 : s.tree.sift R.cut (prefixOf w (fd - 1)) = .inl (walk.getD (fd - 1 - start) []))
    (hq : ∃ q, s.tree.sift R.cut (prefixOf w fd) = .inl q ∧ q ≠ walk.getD (fd - start) [])
    (hquiet : (onEdge K R s w (fun j => walk.getD (j - start) []) pool boundary fd).streak ≠ 0) :
    (∃ i c, start ≤ i ∧ w.toList[i]? = some c ∧ s.edges (walk.getD (i - start) []) c = none
      ∧ members K R s.tree s.pool (walk.getD (i - start) []) ≠ []
      ∧ ∀ m ∈ members K R s.tree s.pool (walk.getD (i - start) []), ∃ b,
        s.tree.sift R.cut (m * FreeMonoid.of c) = .inr b
          ∧ b ∈ (onEdge K R s w (fun j => walk.getD (j - start) []) pool boundary fd).boundary)
    ∨ ∃ i < w.toList.length, prefixOf w i
      ∈ (onEdge K R s w (fun j => walk.getD (j - start) []) pool boundary fd).disagreements := by
  obtain ⟨q, hq, hne⟩ := hq
  obtain ⟨-, p0, -, hwalk⟩ := anchoredWalk_some R hw
  obtain ⟨c, hc⟩ : ∃ c, w.toList[fd - 1]? = some c := ⟨_, List.getElem?_eq_getElem (by omega)⟩
  have hstep : walk.getD (fd - start) []
      = stepPath s.tree s.edges (walk.getD (fd - 1 - start) []) c := by
    have := walk_succ (t := s.tree) (edges := s.edges) (p := p0) (show start ≤ fd - 1 by omega) hc
    rw [show fd - 1 + 1 = fd by omega] at this
    rw [hwalk]; exact this
  rcases he : s.edges (walk.getD (fd - 1 - start) []) c with _ | ⟨s2, x⟩
  · left
    have ho : onEdge K R s w (fun j => walk.getD (j - start) []) pool boundary fd
        = settle K R s.tree pool s.edges (s.streak + 1) s.unchecked boundary s.disagreements := by
      simp only [onEdge, hc, he]
    refine ⟨fd - 1, c, by omega, hc, he, ?_, ?_⟩
    · obtain ⟨y, hy, hsy⟩ := hP _ (DTree.sift_mem_paths _ hs1)
      exact members_ne_nil K R hK hy hsy
    · intro m hm
      obtain ⟨b, hb, hmem⟩ := hS.2 _ _ he m hm
      exact ⟨b, hb, by rw [ho]; exact mem_settle_boundary K R (hbd b hmem)⟩
  · obtain ⟨hx, hxc⟩ := hS.1 _ _ _ _ he
    have hs2 : walk.getD (fd - start) [] = s2 := by
      rw [hstep]; simp only [stepPath, he, DTree.sift_mem_paths _ hxc, ↓reduceIte]
    have hsc : s.tree.sift R.cut (prefixOf w (fd - 1) * FreeMonoid.of c) = .inl q := by
      rw [prefixOf_succ hc, show fd - 1 + 1 = fd by omega]; exact hq
    obtain ⟨d, hd⟩ := DTree.firstDisagreement_isSome s.tree hxc hsc
      (fun h => hne (by rw [hs2]; exact h.symm))
    rcases hv : verdict K R s.tree pool (walk.getD (fd - 1 - start) []) d
        (s.tree.paths.length * Fintype.card α) with _ | _ | _
    · exfalso; apply hquiet
      simp only [onEdge, hc, he, hs2, hx, hs1, hd, hv, settle, ne_eq,
        not_true_eq_false, or_self, ↓reduceIte]
    · right
      refine ⟨fd - 1, by omega, ?_⟩
      simp only [onEdge, hc, he, hs2, hx, hs1, hd, hv, settle, ne_eq,
        not_true_eq_false, or_self, ↓reduceIte]
      simp
    · exfalso; apply hquiet
      simp only [onEdge, hc, he, hs2, hx, hs1, hd, hv, settle, ne_eq,
        not_true_eq_false, or_self, ↓reduceIte]

/-- Every quiet probe is accounted for. -/
theorem probeStep_accounted (hK : 0 < K.memberLimit) (s : PassState α) (w : FreeMonoid α)
    (hS : Settled K R s) (hP : Populated R s) (hquiet : (probeStep K R s w).streak ≠ 0) :
    Accounted K R s w := by
  simp only [Accounted]
  rcases hw : anchoredWalk R s.tree s.edges w with _ | ⟨start, walk⟩
  · right; left
    obtain ⟨b, hb, hmem⟩ := anchoredWalk_none R hw
    refine ⟨0, Nat.zero_le _, b, hb, ?_⟩
    have ho : probeStep K R s w = settle K R s.tree s.pool s.edges (s.streak + 1)
        (s.unchecked + 1) (s.boundary ++ anchorMisses R s.tree w) s.disagreements := by
      simp only [probeStep, hw]
    rw [ho]; exact mem_settle_boundary K R (List.mem_append_right _ hmem)
  obtain ⟨hstart, p0, hp0, hwalk⟩ := anchoredWalk_some R hw
  have hw0 : walk.getD 0 [] = p0 := by
    rw [hwalk, List.getD_eq_getElem?_getD, List.getElem?_scanl_zero]; rfl
  rcases hsw : s.tree.sift R.cut w with actual | b
  · by_cases hact : actual = walk.getD (w.toList.length - start) []
    · left; exact ⟨start, walk, rfl, by rw [hact]⟩
    rcases hf : firstDisagreeingEdge R s.tree w (fun j => walk.getD (j - start) [])
        w.toList.length start w.toList.length with b | fd
    · right; left
      obtain ⟨i, -, hi, hb⟩ := firstDisagreeingEdge_inl R _ _ _ _ _ _ _ hf
      refine ⟨i, hi.le, b, hb, ?_⟩
      have ho : probeStep K R s w = settle K R s.tree
          (if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]) s.edges
          (s.streak + 1) (s.unchecked + 1) (s.boundary ++ anchorMisses R s.tree w ++ [b])
          s.disagreements := by
        simp only [probeStep, hw, hsw, hact, hf, ↓reduceIte]
      rw [ho]; exact mem_settle_boundary K R (by simp)
    · have hlt : start < w.toList.length := by
        rcases Nat.lt_or_ge start w.toList.length with h | h
        · exact h
        exfalso; apply hact
        have hsn : start = w.toList.length := le_antisymm hstart h
        rw [← hsn, Nat.sub_self, hw0]
        rw [hsn, prefixOf_length, hsw] at hp0
        simpa using hp0
      obtain ⟨hfd1, hfd2, hfd3, hfd4⟩ := firstDisagreeingEdge_inr R s.tree w
        (fun j => walk.getD (j - start) []) _ _ _ fd hlt (by omega)
        (by simp only [Nat.sub_self, hw0]; exact hp0)
        ⟨actual, by rw [prefixOf_length]; exact hsw, hact⟩ hf
      have ho : probeStep K R s w = onEdge K R s w (fun j => walk.getD (j - start) [])
          (if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start])
          (s.boundary ++ anchorMisses R s.tree w) fd := by
        simp only [probeStep, hw, hsw, hact, hf, ↓reduceIte]
      rw [ho] at hquiet ⊢
      rcases onEdge_accounted K R hK s w hS hP hw _ _ (fun b hb => List.mem_append_left _ hb)
          hfd1 hfd2 hfd3 hfd4 hquiet with ⟨i, c, hi, hc, he, hne, hall⟩ | ⟨i, hi, hmem⟩
      · right; right; left; exact ⟨start, walk, i, c, rfl, hi, hc, he, hne, hall⟩
      · right; right; right; exact ⟨i, hi, hmem⟩
  · right; left
    refine ⟨w.toList.length, le_rfl, b, by rw [prefixOf_length]; exact hsw, ?_⟩
    have ho : probeStep K R s w = settle K R s.tree
        (if prefixOf w start ∈ s.pool then s.pool else s.pool ++ [prefixOf w start]) s.edges
        (s.streak + 1) (s.unchecked + 1) (s.boundary ++ anchorMisses R s.tree w ++ [b])
        s.disagreements := by
      simp only [probeStep, hw, hsw]
    rw [ho]; exact mem_settle_boundary K R (by simp)

theorem QuietRun.snoc {s s' : PassState α} {ws : List (FreeMonoid α)}
    (h : QuietRun K R s ws s') {w : FreeMonoid α} (hq : (probeStep K R s' w).streak ≠ 0)
    (ha : Accounted K R s' w) : QuietRun K R s (ws ++ [w]) (probeStep K R s' w) := by
  induction h with
  | nil s => exact QuietRun.cons _ _ _ _ hq ha (QuietRun.nil _)
  | cons s w' ws s' h1 h2 _ ih => exact QuietRun.cons _ _ _ _ h1 h2 (ih hq ha)

/-- What a pass carries over the probes `done`: settled and populated, its streak within
patience, and its last `streak` probes a quiet run, all of `done` while it is still probing. -/
def PassInv (s : PassState α) (done : List (FreeMonoid α)) : Prop :=
  Settled K R s ∧ Populated R s ∧ s.streak ≤ K.patience
    ∧ ∃ s₀ pre ws, pre ++ ws <+: done ∧ (s.streak < K.patience → pre ++ ws = done)
      ∧ ws.length = s.streak ∧ QuietRun K R s₀ ws s

theorem passInv_foldl (hK : 0 < K.memberLimit) :
    ∀ (ps : List (FreeMonoid α)) (s : PassState α) (done : List (FreeMonoid α)),
    PassInv K R s done →
      PassInv K R (ps.foldl (fun s w => if K.patience ≤ s.streak then s else probeStep K R s w) s)
        (done ++ ps)
  | [], s, done, h => by simpa using h
  | w :: ps, s, done, h => by
    rw [List.foldl_cons, show done ++ w :: ps = (done ++ [w]) ++ ps by simp]
    apply passInv_foldl hK ps
    obtain ⟨hS, hP, hle, s₀, pre, ws, hpre, hdone, hlen, hrun⟩ := h
    split_ifs with hstop
    · exact ⟨hS, hP, hle, s₀, pre, ws, hpre.trans (List.prefix_append _ _),
        fun h => absurd hstop (by omega), hlen, hrun⟩
    · have hlt : s.streak < K.patience := by omega
      obtain ⟨hS', hP', hst, -⟩ := probeStep_inv K R s w hS hP
      rcases hst with hst | hst
      · have hq : (probeStep K R s w).streak ≠ 0 := by omega
        refine ⟨hS', hP', by omega, s₀, pre, ws ++ [w], ?_, fun _ => ?_, by simp [hlen, hst],
          hrun.snoc K R hq (probeStep_accounted K R hK s w hS hP hq)⟩
        · rw [← List.append_assoc, hdone hlt]
        · rw [← List.append_assoc, hdone hlt]
      · exact ⟨hS', hP', by omega, probeStep K R s w, done ++ [w], [], by simp,
          fun _ => by simp, by simp [hst], QuietRun.nil _⟩

end Pass

theorem pass_dichotomy : PassDichotomy := by
  intro α _ _ K R seed probes hK hseed hstop
  have h0 : PassInv K R (initialState K R seed) [] := by
    refine ⟨settle_settled K R (by simp), ?_, by simp [initialState, settle],
      initialState K R seed, [], [], by simp, fun _ => by simp, by simp [initialState, settle],
      QuietRun.nil _⟩
    intro p hp
    simp only [initialState, settle, DTree.paths, List.map_cons, List.map_nil, List.cons_append,
      List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · obtain ⟨x, hx, hc⟩ := hseed false
      exact ⟨x, hx, by simp [initialState, settle, DTree.sift, hc]⟩
    · obtain ⟨x, hx, hc⟩ := hseed true
      exact ⟨x, hx, by simp [initialState, settle, DTree.sift, hc]⟩
  obtain ⟨-, -, hle, s₀, pre, ws, hpre, -, hlen, hrun⟩ := passInv_foldl K R hK probes _ [] h0
  obtain ⟨rest, hrest⟩ := hpre
  refine ⟨s₀, ws, ⟨pre, rest, by simpa using hrest⟩, ?_, hrun⟩
  rw [hlen]; exact le_antisymm hle hstop

end OrthoDFA
