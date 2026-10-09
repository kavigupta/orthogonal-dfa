import OrthoDFA.Round
import OrthoDFA.Proofs.Replay
import Mathlib.Data.List.Basic

/-!
# What the pass reads

Every read the counterexample pass makes asks the family about a string `b·e₁·e₂·m`: `b` a seed
string, a witness or a probe's prefix, `e₁` and `e₂` each empty or a letter, and `m` a midfix of
the tree the pass holds before the probe.  So two oracles that agree there take the pass through
the same probe (`probeStep_congr`), and the state after `k` probes is decided by the oracle on
those strings, against the tree after `k − 1` (`phase_congr`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

/-- The nodes' midfixes. -/
def mids : DTree α → List (FreeMonoid α)
  | .leaf => []
  | .node m r a => m :: (r.mids ++ a.mids)

theorem route_congr_mids {c₁ c₂ : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α), (∀ m ∈ t.mids, c₁ (x * m) = c₂ (x * m)) →
      t.route c₁ x = t.route c₂ x
  | .leaf, _, _ => rfl
  | .node m r a, x, h => by
    have hm := h m (by simp [mids])
    have hr := route_congr_mids r x fun m' hm' => h m' (by simp [mids, hm'])
    have ha := route_congr_mids a x fun m' hm' => h m' (by simp [mids, hm'])
    simp only [route, hm, hr, ha]

theorem halfway_congr {c₁ c₂ : FreeMonoid α → Option Bool} {m₁ m₂ : FreeMonoid α → Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α),
      (∀ m ∈ t.mids, c₁ (x * m) = c₂ (x * m) ∧ m₁ (x * m) = m₂ (x * m)) →
        t.halfway c₁ m₁ x = t.halfway c₂ m₂ x
  | .leaf, _, _ => rfl
  | .node m r a, x, h => by
    obtain ⟨hc, hm⟩ := h m (by simp [mids])
    have hr := halfway_congr r x fun m' hm' => h m' (by simp [mids, hm'])
    have ha := halfway_congr a x fun m' hm' => h m' (by simp [mids, hm'])
    simp only [halfway, hc, hm, hr, ha]

theorem mids_splitAt (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool) (m : FreeMonoid α),
      m ∈ (t.splitAt d p).mids ↔ m ∈ t.mids ∨ (m = d ∧ (t.splitAt d p).mids ≠ t.mids)
  | .leaf, [], m => by simp [splitAt, mids]
  | .leaf, _ :: _, m => by simp [splitAt]
  | .node _ _ _, [], m => by simp [splitAt]
  | .node n r a, false :: p, m => by
    have := mids_splitAt d r p m
    by_cases he : (r.splitAt d p).mids = r.mids
    · simp only [splitAt, mids, he]; simp
    · simp only [he, ne_eq, not_false_eq_true, and_true] at this
      simp only [splitAt, mids, List.mem_cons, List.mem_append, this, ne_eq, List.cons.injEq,
        true_and, List.append_cancel_right_eq, he, not_false_eq_true, and_true]
      tauto
  | .node n r a, true :: p, m => by
    have := mids_splitAt d a p m
    by_cases he : (a.splitAt d p).mids = a.mids
    · simp only [splitAt, mids, he]; simp
    · simp only [he, ne_eq, not_false_eq_true, and_true] at this
      simp only [splitAt, mids, List.mem_cons, List.mem_append, this, ne_eq, List.cons.injEq,
        true_and, List.append_cancel_left_eq, he, not_false_eq_true, and_true]
      tauto

theorem mem_mids_splitAt {d m : FreeMonoid α} {t : DTree α} {p : List Bool} (h : m ∈ t.mids) :
    m ∈ (t.splitAt d p).mids :=
  (mids_splitAt d t p m).2 (.inl h)

theorem mids_splitAt_cases {d m : FreeMonoid α} {t : DTree α} {p : List Bool}
    (h : m ∈ (t.splitAt d p).mids) : m ∈ t.mids ∨ m = d :=
  ((mids_splitAt d t p m).1 h).imp id And.left

theorem firstDisagreement_mids {cut : FreeMonoid α → Option Bool} {x y pre d : FreeMonoid α} :
    ∀ t : DTree α, t.firstDisagreement cut x y pre = some d → ∃ m ∈ t.mids, d = pre * m
  | .leaf, h => by simp [firstDisagreement] at h
  | .node m r a, h => by
    simp only [firstDisagreement] at h
    split at h
    · obtain ⟨m', hm', rfl⟩ := firstDisagreement_mids a h
      exact ⟨m', by simp [mids, hm'], rfl⟩
    · obtain ⟨m', hm', rfl⟩ := firstDisagreement_mids r h
      exact ⟨m', by simp [mids, hm'], rfl⟩
    · exact ⟨m, by simp [mids], (Option.some.inj h).symm⟩
    · simp at h

theorem firstDisagreement_congr {c₁ c₂ : FreeMonoid α → Option Bool} {x y pre : FreeMonoid α} :
    ∀ t : DTree α, (∀ m ∈ t.mids, c₁ (x * (pre * m)) = c₂ (x * (pre * m))
        ∧ c₁ (y * (pre * m)) = c₂ (y * (pre * m))) →
      t.firstDisagreement c₁ x y pre = t.firstDisagreement c₂ x y pre
  | .leaf, _ => rfl
  | .node m r a, h => by
    have hm := h m (by simp [mids])
    have hr := firstDisagreement_congr r fun m' hm' => h m' (by simp [mids, hm'])
    have ha := firstDisagreement_congr a fun m' hm' => h m' (by simp [mids, hm'])
    simp only [firstDisagreement, hm.1, hm.2, hr, ha]

end DTree

theorem findSome?_congr {β γ : Type*} {f g : β → Option γ} :
    ∀ {l : List β}, (∀ x ∈ l, f x = g x) → l.findSome? f = l.findSome? g
  | [], _ => rfl
  | x :: xs, h => by
    simp only [List.findSome?_cons, h x (by simp),
      findSome?_congr fun y hy => h y (List.mem_cons_of_mem _ hy)]

theorem foldl_congr_mem {β γ : Type*} {f g : γ → β → γ} :
    ∀ {l : List β} (init : γ), (∀ a, ∀ x ∈ l, f a x = g a x) → l.foldl f init = l.foldl g init
  | [], _, _ => rfl
  | x :: xs, init, h => by
    simp only [List.foldl_cons, h init x (by simp)]
    exact foldl_congr_mem _ fun a y hy => h a y (List.mem_cons_of_mem _ hy)

section Congr

variable (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)

/-- The family `F` cut at `B`, read through `f`. -/
abbrev rd (f : FreeMonoid α → ℝ) : CutReads α := ⟨B, F, f⟩

/-- `f₁` and `f₂` agree on what a read of `y` asks: `y` followed by each suffix of the family or
its training half. -/
def Agree (y : FreeMonoid α) : Prop := ∀ v ∈ F ∪ K.train F, f₁ (y * v) = f₂ (y * v)

/-- `e` as a string: empty, or the letter. -/
def ext : Option α → FreeMonoid α
  | none => 1
  | some c => FreeMonoid.of c

/-- They agree on every read of `b` extended by up to a letter, against `t`. -/
def AgreeOne (t : DTree α) (b : FreeMonoid α) : Prop :=
  ∀ e : Option α, ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * ext e * m)

/-- They agree on every read of `b` extended by up to two letters, against `t`. -/
def AgreeDeep (t : DTree α) (b : FreeMonoid α) : Prop :=
  ∀ e₁ e₂ : Option α, ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * ext e₁ * ext e₂ * m)

variable {K B F f₁ f₂}

theorem acceptsOn_congr {W : Finset (FreeMonoid α)} (hW : W ⊆ F ∪ K.train F)
    {y : FreeMonoid α} (h : Agree K F f₁ f₂ y) : acceptsOn W f₁ y = acceptsOn W f₂ y := by
  classical
  unfold acceptsOn
  congr 1
  exact Finset.filter_congr fun v hv => by rw [h v (hW hv)]

theorem cut_congr {y : FreeMonoid α} (h : Agree K F f₁ f₂ y) :
    (rd B F f₁).cut y = (rd B F f₂).cut y := by
  simp only [CutReads.cut, rd, acceptsOn_congr Finset.subset_union_left h]

theorem sift_congr {t : DTree α} {x : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (x * m)) :
    t.sift (rd B F f₁).cut x = t.sift (rd B F f₂).cut x := by
  unfold DTree.sift
  rw [DTree.route_congr_mids t x fun m hm => cut_congr (h m hm)]

theorem route_congr' {t : DTree α} {x : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (x * m)) :
    t.route (rd B F f₁).cut x = t.route (rd B F f₂).cut x :=
  DTree.route_congr_mids t x fun m hm => cut_congr (h m hm)

theorem AgreeOne.tree {t : DTree α} {b : FreeMonoid α} (h : AgreeOne K F f₁ f₂ t b) :
    ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * m) := fun m hm => by
  simpa [ext] using h none m hm

theorem AgreeOne.letter {t : DTree α} {b : FreeMonoid α} (h : AgreeOne K F f₁ f₂ t b) (c : α) :
    ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * FreeMonoid.of c * m) := fun m hm => by
  simpa [ext] using h (some c) m hm

theorem AgreeOne.letter' {t : DTree α} {b : FreeMonoid α} (h : AgreeOne K F f₁ f₂ t b) (c : α) :
    ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * (FreeMonoid.of c * m)) := fun m hm => by
  simpa [ext, mul_assoc] using h (some c) m hm

theorem AgreeDeep.one {t : DTree α} {b : FreeMonoid α} (h : AgreeDeep K F f₁ f₂ t b) :
    AgreeOne K F f₁ f₂ t b := fun e m hm => by
  simpa [ext] using h e none m hm

/-- After a split on `ofc'·m'`, a midfix of `t` extended by a letter, the agreement against the
new tree on `b` and `b` extended by a letter. -/
theorem AgreeDeep.split {t : DTree α} {b m' : FreeMonoid α} {c' : α} {p : List Bool}
    (h : AgreeDeep K F f₁ f₂ t b) (hm' : m' ∈ t.mids) :
    AgreeOne K F f₁ f₂ (t.splitAt (FreeMonoid.of c' * m') p) b := fun e m hm => by
  rcases DTree.mids_splitAt_cases hm with hm | rfl
  · simpa [ext] using h e none m hm
  · simpa [ext, mul_assoc] using h e (some c') m' hm'

theorem members_mem {R : CutReads α} {t : DTree α} {pool : List (FreeMonoid α)}
    {path : List Bool} {b : FreeMonoid α} (h : b ∈ members K R t pool path) : b ∈ pool :=
  (List.mem_filter.1 (List.mem_of_mem_take h)).1

theorem members_congr {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    (h : ∀ b ∈ pool, ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * m)) :
    members K (rd B F f₁) t pool path = members K (rd B F f₂) t pool path := by
  unfold members
  congr 1
  apply List.filter_congr
  intro b hb
  simp only [sift_congr (B := B) (h b hb)]

theorem decisiveTarget_congr {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {c : α} {cur : Option (List Bool)} (h : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b) :
    decisiveTarget K (rd B F f₁) t pool path c cur
      = decisiveTarget K (rd B F f₂) t pool path c cur := by
  unfold decisiveTarget
  rw [members_congr fun b hb => (h b hb).tree]
  generalize hV₁ : List.filterMap _ (members K (rd B F f₂) t pool path) = V₁
  generalize hV₂ : List.filterMap _ (members K (rd B F f₂) t pool path) = V₂
  have : V₁ = V₂ := by
    rw [← hV₁, ← hV₂]
    apply List.filterMap_congr
    intro b hb
    simp only [sift_congr (B := B) ((h b (members_mem hb)).letter c)]
  subst this
  rfl

theorem edgeMisses_congr {t : DTree α} {c : α} :
    ∀ {ms : List (FreeMonoid α)}, (∀ b ∈ ms, AgreeOne K F f₁ f₂ t b) →
      edgeMisses (rd B F f₁) t c ms = edgeMisses (rd B F f₂) t c ms
  | [], _ => rfl
  | b :: ms, h => by
    simp only [edgeMisses, sift_congr (B := B) ((h b (by simp)).letter c),
      edgeMisses_congr fun b' hb' => h b' (List.mem_cons_of_mem _ hb')]

theorem closeEdges_congr {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : List Bool → α → Option (List Bool × FreeMonoid α)}
    (h : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b) :
    closeEdges K (rd B F f₁) t pool edges = closeEdges K (rd B F f₂) t pool edges := by
  funext path c
  simp only [closeEdges, decisiveTarget_congr h]

theorem settle_congr {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : List Bool → α → Option (List Bool × FreeMonoid α)} {st un : ℕ}
    {bd hd : List (FreeMonoid α)} (h : ∀ b ∈ pool, AgreeOne K F f₁ f₂ t b) :
    settle K (rd B F f₁) t pool edges st un bd hd
      = settle K (rd B F f₂) t pool edges st un bd hd := by
  simp only [settle, closeEdges_congr h, PassState.mk.injEq, true_and, and_true]
  congr 1
  refine List.flatMap_congr fun p _ => List.flatMap_congr fun c _ => ?_
  rw [members_congr fun b hb => (h b hb).tree]
  exact edgeMisses_congr fun b hb => h b (members_mem hb)

theorem anchoredWalk_congr {t : DTree α}
    {edges : List Bool → α → Option (List Bool × FreeMonoid α)} {w : FreeMonoid α}
    (h : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf w i * m)) :
    anchoredWalk (rd B F f₁) t edges w = anchoredWalk (rd B F f₂) t edges w := by
  unfold anchoredWalk
  generalize hV₁ : List.findSome? _ (List.range (w.toList.length + 1)) = V₁
  generalize hV₂ : List.findSome? _ (List.range (w.toList.length + 1)) = V₂
  have : V₁ = V₂ := by
    rw [← hV₁, ← hV₂]
    apply findSome?_congr
    intro i _
    simp only [sift_congr (B := B) (h i)]
  subst this
  rfl

theorem anchorMisses_congr {t : DTree α} {w : FreeMonoid α}
    (h : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf w i * m)) :
    anchorMisses (rd B F f₁) t w = anchorMisses (rd B F f₂) t w := by
  unfold anchorMisses
  congr 2
  apply List.map_congr_left
  intro i _
  exact sift_congr (h i)

theorem firstDisagreeingEdge_congr {t : DTree α} {w : FreeMonoid α} {walk : ℕ → List Bool}
    (h : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf w i * m)) :
    ∀ fuel lo hi, firstDisagreeingEdge (rd B F f₁) t w walk fuel lo hi
      = firstDisagreeingEdge (rd B F f₂) t w walk fuel lo hi
  | 0, _, _ => rfl
  | fuel + 1, lo, hi => by
    simp only [firstDisagreeingEdge, sift_congr (B := B) (h ((lo + hi) / 2)),
      firstDisagreeingEdge_congr h fuel]

theorem bisect_congr {t : DTree α} {w : FreeMonoid α} {walk : ℕ → List Bool}
    (h : ∀ i, ∀ m ∈ t.mids, Agree K F f₁ f₂ (prefixOf w i * m)) :
    ∀ fuel lo hi, bisect (rd B F f₁) t w walk fuel lo hi = bisect (rd B F f₂) t w walk fuel lo hi
  | 0, _, _ => rfl
  | fuel + 1, lo, hi => by
    simp only [bisect, sift_congr (B := B) (h ((lo + hi) / 2)), bisect_congr h fuel]

theorem tally_congr {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {d : FreeMonoid α} (h : ∀ b ∈ pool, ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * m))
    (hd : ∀ b ∈ pool, Agree K F f₁ f₂ (b * d)) :
    tally K (rd B F f₁) t pool path d = tally K (rd B F f₂) t pool path d := by
  unfold tally
  rw [members_congr h]
  refine foldl_congr_mem _ fun acc b hb => ?_
  have hb' := hd b (members_mem hb)
  simp only [rd, acceptsOn_congr Finset.subset_union_right hb',
    acceptsOn_congr (Finset.sdiff_subset.trans Finset.subset_union_left) hb']

theorem verdict_congr {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {d : FreeMonoid α} {tests : ℕ} (h : ∀ b ∈ pool, ∀ m ∈ t.mids, Agree K F f₁ f₂ (b * m))
    (hd : ∀ b ∈ pool, Agree K F f₁ f₂ (b * d)) :
    verdict K (rd B F f₁) t pool path d tests = verdict K (rd B F f₂) t pool path d tests := by
  unfold verdict
  rw [tally_congr h hd]

theorem prefixOf_length (w : FreeMonoid α) : prefixOf w w.toList.length = w := by
  simp [prefixOf]

theorem onEdge_congr {s : PassState α} {w : FreeMonoid α} {walkAt : ℕ → List Bool}
    {pool boundary held : List (FreeMonoid α)} {fd : ℕ}
    (hpool : ∀ b ∈ pool, AgreeDeep K F f₁ f₂ s.tree b)
    (hwit : ∀ p c q y, s.edges p c = some (q, y) → AgreeDeep K F f₁ f₂ s.tree y)
    (hw : ∀ i, AgreeDeep K F f₁ f₂ s.tree (prefixOf w i)) :
    onEdge K (rd B F f₁) s w walkAt pool boundary held fd
      = onEdge K (rd B F f₂) s w walkAt pool boundary held fd := by
  have hone : ∀ b ∈ pool, AgreeOne K F f₁ f₂ s.tree b := fun b hb => (hpool b hb).one
  simp only [onEdge]
  rcases hc : w.toList[fd - 1]? with _ | c
  · exact settle_congr hone
  · simp only []
    rcases he : s.edges (walkAt (fd - 1)) c with _ | ⟨s2, x⟩
    · exact settle_congr hone
    · simp only []
      have hx := hwit _ _ _ _ he
      have hsp := hw (fd - 1)
      rw [sift_congr (B := B) hx.one.tree, sift_congr (B := B) hsp.one.tree]
      split_ifs with hplace hcond
      · exact settle_congr hone
      · exact settle_congr hone
      · rw [DTree.firstDisagreement_congr (c₁ := (rd B F f₁).cut) (c₂ := (rd B F f₂).cut) s.tree
          fun m hm => ⟨cut_congr (hx.one.letter' c m hm), cut_congr (hsp.one.letter' c m hm)⟩]
        rcases hd : s.tree.firstDisagreement (rd B F f₂).cut x (prefixOf w (fd - 1))
            (FreeMonoid.of c) with _ | d
        · exact settle_congr hone
        · obtain ⟨m', hm', rfl⟩ := DTree.firstDisagreement_mids _ hd
          simp only []
          rw [verdict_congr (fun b hb => (hone b hb).tree)
            (fun b hb => (hone b hb).letter' c m' hm')]
          rcases hv : verdict K (rd B F f₂) s.tree pool (walkAt (fd - 1)) (FreeMonoid.of c * m')
              (s.tree.paths.length * Fintype.card α) with _ | _ | _
          · refine settle_congr fun b hb => ?_
            rcases List.mem_append.1 hb with hb | hb
            · exact (hpool b hb).split hm'
            · rcases List.mem_cons.1 (List.mem_of_mem_filter hb) with rfl | hb
              · exact hx.split hm'
              · rw [List.mem_singleton.1 hb]
                exact hsp.split hm'
          · refine settle_congr fun b hb => ?_
            rcases List.mem_cons.1 hb with rfl | hb
            · exact hsp.one
            · exact hone b (List.mem_of_mem_filter hb)
          · refine settle_congr fun b hb => ?_
            rcases List.mem_cons.1 hb with rfl | hb
            · exact hsp.one
            · exact hone b (List.mem_of_mem_filter hb)

theorem mid_congr {y : FreeMonoid α} (h : Agree K F f₁ f₂ y) :
    (rd B F f₁).mid y = (rd B F f₂).mid y := by
  simp only [CutReads.mid, rd, acceptsOn_congr Finset.subset_union_left h]

theorem halfway_congr' {t : DTree α} {x : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (x * m)) :
    t.halfway (rd B F f₁).cut (rd B F f₁).mid x = t.halfway (rd B F f₂).cut (rd B F f₂).mid x :=
  DTree.halfway_congr t x fun m hm => ⟨cut_congr (h m hm), mid_congr (h m hm)⟩

theorem midLeaf_congr {t : DTree α} {x : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (x * m)) :
    midLeaf (rd B F f₁) t x = midLeaf (rd B F f₂) t x := by
  have := DTree.route_congr_mids (c₁ := fun y => some ((rd B F f₁).mid y))
    (c₂ := fun y => some ((rd B F f₂).mid y)) t x fun m hm => by simp only [mid_congr (h m hm)]
  simp only [midLeaf, DTree.sift, this]

theorem probeWalk_congr {s : PassState α} {w : FreeMonoid α}
    (h : ∀ i, ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (prefixOf w i * m)) :
    probeWalk (rd B F f₁) s w = probeWalk (rd B F f₂) s w := by
  have h1 : ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (1 * m) := by
    simpa [prefixOf_zero] using h 0
  unfold probeWalk
  rw [anchoredWalk_congr (B := B) h, midLeaf_congr (B := B) (by simpa using h1)]

theorem probeStep_congr {s : PassState α} {w : FreeMonoid α}
    (hpool : ∀ b ∈ s.pool, AgreeDeep K F f₁ f₂ s.tree b)
    (hwit : ∀ p c q y, s.edges p c = some (q, y) → AgreeDeep K F f₁ f₂ s.tree y)
    (hw : ∀ i, AgreeDeep K F f₁ f₂ s.tree (prefixOf w i)) :
    probeStep K (rd B F f₁) s w = probeStep K (rd B F f₂) s w := by
  have hpw : ∀ i, ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (prefixOf w i * m) :=
    fun i => (hw i).one.tree
  have hwhole : ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (w * m) := by
    simpa [prefixOf_length] using hpw w.toList.length
  have h1 : ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (1 * m) := by
    simpa [prefixOf_zero] using hpw 0
  simp only [probeStep]
  rw [show probePool (rd B F f₁) s w = probePool (rd B F f₂) s w by
      unfold probePool; rw [anchoredWalk_congr (B := B) hpw],
    anchorMisses_congr (B := B) hpw, probeWalk_congr (B := B) hpw, halfway_congr' (B := B) h1, halfway_congr' (B := B) hwhole,
    sift_congr (B := B) hwhole, firstDisagreeingEdge_congr (B := B) hpw]
  have hpool' : ∀ b ∈ probePool (rd B F f₂) s w, AgreeDeep K F f₁ f₂ s.tree b := by
    intro b hb
    unfold probePool at hb
    rcases ha : anchoredWalk (rd B F f₂) s.tree s.edges w with _ | ⟨start, walk⟩ <;>
      rw [ha] at hb
    · exact hpool b hb
    · simp only [] at hb
      split_ifs at hb
      · exact hpool b hb
      · rcases List.mem_append.1 hb with hb | hb
        · exact hpool b hb
        · rw [List.mem_singleton.1 hb]; exact hw start
  generalize probePool (rd B F f₂) s w = pool at hpool' ⊢
  generalize probeWalk (rd B F f₂) s w = pw
  rcases hs : s.tree.sift (rd B F f₂).cut w with actual | b
  · simp only []
    split_ifs
    · exact settle_congr fun b hb => (hpool' b hb).one
    · rcases hf : firstDisagreeingEdge (rd B F f₂) s.tree w _ _ _ _ with b | fd
      · exact settle_congr fun b hb => (hpool' b hb).one
      · exact onEdge_congr hpool' hwit hw
  · simp only []
    split_ifs
    · exact settle_congr fun b hb => (hpool' b hb).one
    · exact settle_congr fun b hb => (hpool' b hb).one

theorem siftReads_congr {t : DTree α} {x : FreeMonoid α}
    (h : ∀ m ∈ t.mids, Agree K F f₁ f₂ (x * m)) :
    siftReads (rd B F f₁) t x = siftReads (rd B F f₂) t x := by
  unfold siftReads
  rw [route_congr' h]

theorem probeReads_congr {s : PassState α} {w : FreeMonoid α}
    (hwit : ∀ p c q y, s.edges p c = some (q, y) → AgreeDeep K F f₁ f₂ s.tree y)
    (hw : ∀ i, AgreeDeep K F f₁ f₂ s.tree (prefixOf w i)) :
    probeReads (rd B F f₁) s w = probeReads (rd B F f₂) s w := by
  have hpw : ∀ i, ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (prefixOf w i * m) :=
    fun i => (hw i).one.tree
  have hsift : ∀ i, s.tree.sift (rd B F f₁).cut (prefixOf w i)
      = s.tree.sift (rd B F f₂).cut (prefixOf w i) := fun i => sift_congr (hpw i)
  have hreads : ∀ i, siftReads (rd B F f₁) s.tree (prefixOf w i)
      = siftReads (rd B F f₂) s.tree (prefixOf w i) := fun i => siftReads_congr (hpw i)
  have hwhole : ∀ m ∈ s.tree.mids, Agree K F f₁ f₂ (w * m) := by
    simpa [prefixOf_length] using hpw w.toList.length
  simp only [probeReads]
  rw [probeWalk_congr (B := B) hpw]
  simp only [hsift, hreads]
  rw [sift_congr (B := B) hwhole, siftReads_congr (B := B) hwhole, bisect_congr (B := B) hpw]
  generalize probeWalk (rd B F f₂) s w = pw
  rcases hs : s.tree.sift (rd B F f₂).cut w with actual | b
  · simp only []
    split_ifs
    · rfl
    · rcases hb : (bisect (rd B F f₂) s.tree w pw.2 w.toList.length pw.1 w.toList.length).2
          with b | fd
      · rfl
      · simp only []
        rcases hc : w.toList[fd - 1]? with _ | c
        · rfl
        · simp only []
          rcases he : s.edges (pw.2 (fd - 1)) c with _ | ⟨s2, x⟩
          · rfl
          · simp only []
            have hx := (hwit _ _ _ _ he).one.tree
            rw [sift_congr (B := B) hx, siftReads_congr (B := B) hx]
  · rfl

/-- One step of the pass, as `runPass` takes it. -/
noncomputable def runStep (K : StageKnobs α) (R : CutReads α) (s : PassState α)
    (w : FreeMonoid α) : PassState α :=
  if K.patience ≤ s.streak then s
  else
    let s' := probeStep K R s w
    { s' with reads := if s'.streak = 0 then 0 else s.reads + probeReads R s w }

theorem runStep_congr {s : PassState α} {w : FreeMonoid α}
    (hpool : ∀ b ∈ s.pool, AgreeDeep K F f₁ f₂ s.tree b)
    (hwit : ∀ p c q y, s.edges p c = some (q, y) → AgreeDeep K F f₁ f₂ s.tree y)
    (hw : ∀ i, AgreeDeep K F f₁ f₂ s.tree (prefixOf w i)) :
    runStep K (rd B F f₁) s w = runStep K (rd B F f₂) s w := by
  simp only [runStep, probeStep_congr hpool hwit hw, probeReads_congr hwit hw]

end Congr


section Phases

variable (K : StageKnobs α) (R : CutReads α)

theorem foldl_best_mem {β : Type*} (g : Option β → β → Option β)
    (hg : ∀ o v r, g o v = some r → r = v ∨ o = some r) :
    ∀ (l : List β) (o : Option β) (r : β), l.foldl g o = some r → r ∈ l ∨ o = some r
  | [], o, r, h => .inr h
  | v :: vs, o, r, h => by
    rcases foldl_best_mem g hg vs (g o v) r h with hr | hr
    · exact .inl (List.mem_cons_of_mem _ hr)
    · rcases hg o v r hr with rfl | ho
      · exact .inl (by simp)
      · exact .inr ho

theorem decisiveTarget_mem {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool} {c : α}
    {cur : Option (List Bool)} {q : List Bool} {y : FreeMonoid α}
    (h : decisiveTarget K R t pool path c cur = some (q, y)) : y ∈ members K R t pool path := by
  unfold decisiveTarget at h
  dsimp only at h
  generalize hV : List.filterMap _ (members K R t pool path) = V at h
  rcases foldl_best_mem _ (fun o v r hr => by
      rcases o with _ | b
      · exact .inl (Option.some.inj hr).symm
      · dsimp only at hr
        split_ifs at hr
        · exact .inl (Option.some.inj hr).symm
        · exact .inr hr) V none (q, y) h with hm | hm
  · rw [← hV] at hm
    obtain ⟨a, ha, hfa⟩ := List.mem_filterMap.1 hm
    split at hfa
    · obtain ⟨-, rfl⟩ := Prod.mk.inj (Option.some.inj hfa)
      exact ha
    · simp at hfa
  · simp at hm

theorem mem_append_single {Bs : Set (FreeMonoid α)} {l : List (FreeMonoid α)} {a : FreeMonoid α}
    (hl : ∀ b ∈ l, b ∈ Bs) (ha : a ∈ Bs) : ∀ b ∈ l ++ [a], b ∈ Bs := fun b hb => by
  rcases List.mem_append.1 hb with hb | hb
  · exact hl b hb
  · rw [List.mem_singleton.1 hb]; exact ha

theorem mem_cons_filter {Bs : Set (FreeMonoid α)} {l : List (FreeMonoid α)} {a : FreeMonoid α}
    {p : FreeMonoid α → Bool} (ha : a ∈ Bs) (hl : ∀ b ∈ l, b ∈ Bs) :
    ∀ b ∈ a :: l.filter p, b ∈ Bs := fun b hb => by
  rcases List.mem_cons.1 hb with rfl | hb
  · exact ha
  · exact hl b (List.mem_of_mem_filter hb)

theorem mem_append_filter {Bs : Set (FreeMonoid α)} {l l' : List (FreeMonoid α)}
    {p : FreeMonoid α → Bool} (hl : ∀ b ∈ l, b ∈ Bs) (hl' : ∀ b ∈ l', b ∈ Bs) :
    ∀ b ∈ l ++ l'.filter p, b ∈ Bs := fun b hb => by
  rcases List.mem_append.1 hb with hb | hb
  · exact hl b hb
  · exact hl' b (List.mem_of_mem_filter hb)

theorem mem_pair {Bs : Set (FreeMonoid α)} {a a' : FreeMonoid α} (ha : a ∈ Bs) (ha' : a' ∈ Bs) :
    ∀ b ∈ [a, a'], b ∈ Bs := fun b hb => by
  rcases List.mem_cons.1 hb with rfl | hb
  · exact ha
  · rw [List.mem_singleton.1 hb]; exact ha'

/-- The pool and every edge's witness lie in `Bs`. -/
def PoolIn (Bs : Set (FreeMonoid α)) (s : PassState α) : Prop :=
  (∀ b ∈ s.pool, b ∈ Bs) ∧ ∀ p c q y, s.edges p c = some (q, y) → y ∈ Bs

theorem settle_poolIn {Bs : Set (FreeMonoid α)} {t : DTree α} {pool : List (FreeMonoid α)}
    {edges : List Bool → α → Option (List Bool × FreeMonoid α)} {st un : ℕ}
    {bd hd : List (FreeMonoid α)} (hp : ∀ b ∈ pool, b ∈ Bs)
    (he : ∀ p c q y, edges p c = some (q, y) → y ∈ Bs) :
    PoolIn Bs (settle K R t pool edges st un bd hd) := by
  refine ⟨hp, fun p c q y h => ?_⟩
  simp only [settle, closeEdges] at h
  rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', y'⟩
  · rw [hd] at h
    exact he _ _ _ _ h
  · rw [hd] at h
    dsimp only at h
    split_ifs at h
    · exact he _ _ _ _ h
    obtain ⟨-, rfl⟩ := Prod.mk.inj (Option.some.inj h)
    exact hp _ (members_mem (decisiveTarget_mem K R hd))

theorem probePool_in {Bs : Set (FreeMonoid α)} {s : PassState α} {w : FreeMonoid α}
    (hp : ∀ b ∈ s.pool, b ∈ Bs) (hw : ∀ i, prefixOf w i ∈ Bs) :
    ∀ b ∈ probePool R s w, b ∈ Bs := by
  unfold probePool
  split
  · exact hp
  · split
    · exact hp
    · exact mem_append_single hp (hw _)

theorem probeStep_poolIn {Bs : Set (FreeMonoid α)} {s : PassState α} {w : FreeMonoid α}
    (hs : PoolIn Bs s) (hw : ∀ i, prefixOf w i ∈ Bs) : PoolIn Bs (probeStep K R s w) := by
  obtain ⟨hp, he⟩ := hs
  have hP := probePool_in R hp hw
  simp only [probeStep, onEdge]
  generalize probePool R s w = pool at hP ⊢
  repeat' split
  all_goals (try have hx := he _ _ _ _ ‹s.edges _ _ = some (_, _)›)
  all_goals refine settle_poolIn K R (fun b hb => ?_) (fun p c q y hy => ?_)
  all_goals first
    | exact hP b hb
    | exact he _ _ _ _ hy
    | exact mem_cons_filter (hw _) hP b hb
    | exact mem_append_filter hP (mem_pair hx (hw _)) b hb
    | (rcases hE : s.edges p c with _ | ⟨q', y'⟩ <;> simp [hE] at hy
       obtain ⟨-, rfl, rfl⟩ := hy
       exact he _ _ _ _ hE)

theorem runStep_poolIn {Bs : Set (FreeMonoid α)} {s : PassState α} {w : FreeMonoid α}
    (hs : PoolIn Bs s) (hw : ∀ i, prefixOf w i ∈ Bs) : PoolIn Bs (runStep K R s w) := by
  unfold runStep
  split
  · exact hs
  · exact probeStep_poolIn K R hs hw

theorem probeStep_tree_cases (s : PassState α) (w : FreeMonoid α) :
    (probeStep K R s w).tree = s.tree ∨ ∃ d p, (probeStep K R s w).tree = s.tree.splitAt d p := by
  simp only [probeStep, onEdge, settle, probePool]
  repeat' split
  all_goals first | exact .inl rfl | exact .inr ⟨_, _, rfl⟩

theorem runStep_tree_cases (s : PassState α) (w : FreeMonoid α) :
    (runStep K R s w).tree = s.tree ∨ ∃ d p, (runStep K R s w).tree = s.tree.splitAt d p := by
  unfold runStep
  split
  · exact .inl rfl
  · exact probeStep_tree_cases K R s w

theorem runStep_mids_mono (s : PassState α) (w : FreeMonoid α) :
    ∀ m ∈ s.tree.mids, m ∈ (runStep K R s w).tree.mids := fun m hm => by
  rcases runStep_tree_cases K R s w with h | ⟨d, p, h⟩
  · rw [h]; exact hm
  · rw [h]; exact DTree.mem_mids_splitAt hm

theorem runStep_mids_new (s : PassState α) (w : FreeMonoid α) :
    ∃ d, ∀ m ∈ (runStep K R s w).tree.mids, m ∈ s.tree.mids ∨ m = d := by
  rcases runStep_tree_cases K R s w with h | ⟨d, p, h⟩
  · exact ⟨1, fun m hm => .inl (h ▸ hm)⟩
  · exact ⟨d, fun m hm => DTree.mids_splitAt_cases (h ▸ hm)⟩

theorem runPass_eq_foldl (s : PassState α) (ws : List (FreeMonoid α)) :
    runPass K R s ws = ws.foldl (runStep K R) s := rfl

/-- The pass's state after the first `k` probes of `ws`. -/
noncomputable def phase (seed ws : List (FreeMonoid α)) (k : ℕ) : PassState α :=
  runPass K R (initialState K R seed) (ws.take k)

theorem phase_zero (seed ws : List (FreeMonoid α)) :
    phase K R seed ws 0 = initialState K R seed := by
  simp [phase, runPass_eq_foldl]

theorem phase_zero_tree (seed ws : List (FreeMonoid α)) :
    (phase K R seed ws 0).tree = .node 1 .leaf .leaf := by
  rw [phase_zero]; rfl

theorem phase_succ (seed ws : List (FreeMonoid α)) {k : ℕ} (hk : k < ws.length) :
    phase K R seed ws (k + 1) = runStep K R (phase K R seed ws k) ws[k] := by
  simp only [phase, runPass_eq_foldl, List.take_succ, List.getElem?_eq_getElem hk,
    Option.toList_some, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem phase_of_le (seed ws : List (FreeMonoid α)) {k : ℕ} (hk : ws.length ≤ k) :
    phase K R seed ws k = phase K R seed ws ws.length := by
  simp only [phase, List.take_of_length_le hk, List.take_length]

theorem phase_succ_cases (seed ws : List (FreeMonoid α)) (k : ℕ) :
    phase K R seed ws (k + 1) = phase K R seed ws k
      ∨ ∃ h : k < ws.length, phase K R seed ws (k + 1) = runStep K R (phase K R seed ws k) ws[k] := by
  by_cases hk : k < ws.length
  · exact .inr ⟨hk, phase_succ K R seed ws hk⟩
  · left
    rw [phase_of_le K R seed ws (k := k + 1) (by omega), phase_of_le K R seed ws (k := k) (by omega)]

theorem phase_mids_mono (seed ws : List (FreeMonoid α)) (k : ℕ) :
    ∀ m ∈ (phase K R seed ws k).tree.mids, m ∈ (phase K R seed ws (k + 1)).tree.mids := by
  rcases phase_succ_cases K R seed ws k with h | ⟨hk, h⟩
  · rw [h]; exact fun m hm => hm
  · rw [h]; exact runStep_mids_mono K R _ _

theorem phase_mids_mono' (seed ws : List (FreeMonoid α)) {j k : ℕ} (hjk : j ≤ k) :
    ∀ m ∈ (phase K R seed ws j).tree.mids, m ∈ (phase K R seed ws k).tree.mids := by
  induction hjk with
  | refl => exact fun m hm => hm
  | step _ ih => exact fun m hm => phase_mids_mono K R seed ws _ m (ih m hm)

theorem phase_mids_new (seed ws : List (FreeMonoid α)) (k : ℕ) :
    ∃ d, ∀ m ∈ (phase K R seed ws (k + 1)).tree.mids,
      m ∈ (phase K R seed ws k).tree.mids ∨ m = d := by
  rcases phase_succ_cases K R seed ws k with h | ⟨hk, h⟩
  · exact ⟨1, fun m hm => .inl (h ▸ hm)⟩
  · rw [h]; exact runStep_mids_new K R _ _

theorem phase_poolIn {Bs : Set (FreeMonoid α)} {seed ws : List (FreeMonoid α)}
    (hseed : ∀ b ∈ seed, b ∈ Bs) (hws : ∀ w ∈ ws, ∀ i, prefixOf w i ∈ Bs) :
    ∀ k, PoolIn Bs (phase K R seed ws k)
  | 0 => by
    rw [phase_zero]
    exact settle_poolIn K R hseed (fun _ _ _ _ h => by simp at h)
  | k + 1 => by
    rcases phase_succ_cases K R seed ws k with h | ⟨hk, h⟩
    · rw [h]; exact phase_poolIn hseed hws k
    · rw [h]; exact runStep_poolIn K R (phase_poolIn hseed hws k) (hws _ (List.getElem_mem hk))

end Phases

section PhaseCongr

variable (K : StageKnobs α) (B : State) (F : Finset (FreeMonoid α)) (f₁ f₂ : FreeMonoid α → ℝ)

omit K B F f₁ f₂ in
theorem AgreeDeep.mono {K : StageKnobs α} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}
    {t t' : DTree α} {b : FreeMonoid α} (h : AgreeDeep K F f₁ f₂ t' b)
    (ht : ∀ m ∈ t.mids, m ∈ t'.mids) : AgreeDeep K F f₁ f₂ t b :=
  fun e₁ e₂ m hm => h e₁ e₂ m (ht m hm)

theorem phase_succ_congr_of {Bs : Set (FreeMonoid α)} {seed ws : List (FreeMonoid α)}
    (hseed : ∀ b ∈ seed, b ∈ Bs) (hws : ∀ w ∈ ws, ∀ i, prefixOf w i ∈ Bs) {k : ℕ}
    (heq : phase K (rd B F f₁) seed ws k = phase K (rd B F f₂) seed ws k)
    (h : ∀ b ∈ Bs, AgreeDeep K F f₁ f₂ (phase K (rd B F f₁) seed ws k).tree b) :
    phase K (rd B F f₁) seed ws (k + 1) = phase K (rd B F f₂) seed ws (k + 1) := by
  by_cases hk : k < ws.length
  · rw [phase_succ K _ seed ws hk, phase_succ K _ seed ws hk, ← heq]
    obtain ⟨hp, he⟩ := phase_poolIn K (rd B F f₁) hseed hws k
    exact runStep_congr (fun b hb => h b (hp b hb)) (fun p c q y hy => h y (he p c q y hy))
      (fun i => h _ (hws _ (List.getElem_mem hk) i))
  · rw [phase_of_le K _ seed ws (k := k + 1) (by omega), phase_of_le K _ seed ws (k := k + 1)
      (by omega)]
    rwa [phase_of_le K _ seed ws (k := k) (by omega), phase_of_le K _ seed ws (k := k)
      (by omega)] at heq

/-- Two oracles agreeing on every read of the bases against the tree after `k` probes take the
pass to the same state after `k` probes, and after `k + 1`. -/
theorem phase_congr {Bs : Set (FreeMonoid α)} {seed ws : List (FreeMonoid α)}
    (hseed : ∀ b ∈ seed, b ∈ Bs) (hws : ∀ w ∈ ws, ∀ i, prefixOf w i ∈ Bs) :
    ∀ k, (∀ b ∈ Bs, AgreeDeep K F f₁ f₂ (phase K (rd B F f₁) seed ws k).tree b) →
      phase K (rd B F f₁) seed ws k = phase K (rd B F f₂) seed ws k
  | 0, h => by
    rw [phase_zero, phase_zero]
    exact settle_congr fun b hb => by
      have := (h b (hseed b hb)).one
      rwa [phase_zero_tree] at this
  | k + 1, h => by
    have h' : ∀ b ∈ Bs, AgreeDeep K F f₁ f₂ (phase K (rd B F f₁) seed ws k).tree b :=
      fun b hb => (h b hb).mono (phase_mids_mono K _ seed ws k)
    exact phase_succ_congr_of K B F f₁ f₂ hseed hws (phase_congr hseed hws k h') h'

end PhaseCongr

section Reads

variable {Ω : Type*} [MeasurableSpace Ω] {μ : MeasureTheory.Measure Ω} {Q : Type*}

/-- The strings `b·e₁·e₂·m` a pass reads against `t`, `b` from `Bs`. -/
noncomputable def nodeReads (Bs : Finset (FreeMonoid α)) (t : DTree α) : Finset (FreeMonoid α) :=
  Bs.biUnion fun b => (Finset.univ : Finset (Option α)).biUnion fun e₁ =>
    (Finset.univ : Finset (Option α)).biUnion fun e₂ =>
      t.mids.toFinset.image fun m => b * ext e₁ * ext e₂ * m

theorem mem_nodeReads {Bs : Finset (FreeMonoid α)} {t : DTree α} {b m : FreeMonoid α}
    (e₁ e₂ : Option α) (hb : b ∈ Bs) (hm : m ∈ t.mids) :
    b * ext e₁ * ext e₂ * m ∈ nodeReads Bs t := by
  simp only [nodeReads, Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_image,
    List.mem_toFinset]
  exact ⟨b, hb, e₁, e₂, m, hm, rfl⟩

theorem card_nodeReads (Bs : Finset (FreeMonoid α)) (t : DTree α) :
    (nodeReads Bs t).card
      ≤ Bs.card * ((Fintype.card α + 1) * ((Fintype.card α + 1) * t.mids.length)) := by
  unfold nodeReads
  calc _ ≤ ∑ _b ∈ Bs, ∑ _e₁ : Option α, ∑ _e₂ : Option α, t.mids.length := by
        refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun b _ => ?_)
        refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun e₁ _ => ?_)
        refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun e₂ _ => ?_)
        exact Finset.card_image_le.trans (List.toFinset_card_le _)
    _ = _ := by simp [Finset.sum_const, Fintype.card_option]

/-- A pass's bases: the seed, and every prefix of a probe no longer than `L`. -/
noncomputable def passBases (seed ws : List (FreeMonoid α)) (L : ℕ) : Finset (FreeMonoid α) :=
  seed.toFinset ∪ ws.toFinset.biUnion fun w => (Finset.range (L + 1)).image (prefixOf w)

theorem seed_mem_passBases {seed ws : List (FreeMonoid α)} {L : ℕ} {b : FreeMonoid α}
    (hb : b ∈ seed) : b ∈ passBases seed ws L :=
  Finset.mem_union_left _ (List.mem_toFinset.2 hb)

theorem prefixOf_mem_passBases {seed ws : List (FreeMonoid α)} {L : ℕ} {w : FreeMonoid α}
    (hw : w ∈ ws) (hL : w.toList.length ≤ L) (i : ℕ) : prefixOf w i ∈ passBases seed ws L := by
  refine Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨w, List.mem_toFinset.2 hw, ?_⟩)
  refine Finset.mem_image.2 ⟨min i L, Finset.mem_range.2 (by omega), ?_⟩
  unfold prefixOf
  rcases le_total i L with h | h
  · rw [min_eq_left h]
  · rw [min_eq_right h, List.take_of_length_le hL, List.take_of_length_le (hL.trans h)]

theorem card_passBases (seed ws : List (FreeMonoid α)) (L : ℕ) :
    (passBases seed ws L).card ≤ seed.length + ws.length * (L + 1) := by
  unfold passBases
  refine (Finset.card_union_le _ _).trans (add_le_add (List.toFinset_card_le _) ?_)
  calc _ ≤ ∑ _w ∈ ws.toFinset, (L + 1) := Finset.card_biUnion_le.trans
        (Finset.sum_le_sum fun w _ => Finset.card_image_le.trans (by simp))
    _ = ws.toFinset.card * (L + 1) := by simp
    _ ≤ _ := Nat.mul_le_mul_right _ (List.toFinset_card_le _)

theorem agree_of_noise (K : StageKnobs α) (O : Oracle μ (FreeMonoid α))
    (F : Finset (FreeMonoid α)) {ω ω' : Ω} {y : FreeMonoid α}
    (h : ∀ v ∈ F ∪ K.train F, O.noise (y * v) ω = O.noise (y * v) ω') :
    Agree K F (fun w => O.mq w ω) (fun w => O.mq w ω') y := fun v hv => by
  simp only [Oracle.mq, h v hv]

theorem roundEnd_eq_phase (K : StageKnobs α) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (seed : List (FreeMonoid α)) {N : ℕ}
    (θ : Ω × (Fin N → FreeMonoid α)) :
    roundEnd K O B F seed θ
      = phase K (rd B F fun w => O.mq w θ.1) seed (List.ofFn θ.2) N := by
  simp only [roundEnd, phase, readsAt, rd]
  rw [List.take_of_length_le (by simp)]

/-- The oracle's bits a pass reads at the strings `Y`: each followed by every suffix of the
family or its training half. -/
noncomputable def readBits (K : StageKnobs α) (F : Finset (FreeMonoid α))
    (Y : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  Y.biUnion fun y => (F ∪ K.train F).image (y * ·)

/-- After `k` probes, the round's state is decided by the oracle's bits at what the pass reads
against the tree after `k`, and so is the state after `k + 1`. -/
theorem phase_determined (K : StageKnobs α) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (seed : List (FreeMonoid α)) {N L : ℕ}
    (p : Fin N → FreeMonoid α) (hp : ∀ i, (p i).toList.length ≤ L) (ω ω' : Ω) (k : ℕ)
    (h : ∀ x ∈ readBits K F (nodeReads (passBases seed (List.ofFn p) L)
        (phase K (rd B F fun w => O.mq w ω) seed (List.ofFn p) k).tree),
      O.noise x ω = O.noise x ω') :
    phase K (rd B F fun w => O.mq w ω') seed (List.ofFn p) k
        = phase K (rd B F fun w => O.mq w ω) seed (List.ofFn p) k
      ∧ phase K (rd B F fun w => O.mq w ω') seed (List.ofFn p) (k + 1)
        = phase K (rd B F fun w => O.mq w ω) seed (List.ofFn p) (k + 1) := by
  have hseed : ∀ b ∈ seed, b ∈ (passBases seed (List.ofFn p) L : Set (FreeMonoid α)) :=
    fun b hb => seed_mem_passBases hb
  have hws : ∀ w ∈ List.ofFn p, ∀ i,
      prefixOf w i ∈ (passBases seed (List.ofFn p) L : Set (FreeMonoid α)) := by
    intro w hw i
    obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hw
    exact prefixOf_mem_passBases hw (hp j) i
  have hag : ∀ b ∈ (passBases seed (List.ofFn p) L : Set (FreeMonoid α)),
      AgreeDeep K F (fun w => O.mq w ω) (fun w => O.mq w ω')
        (phase K (rd B F fun w => O.mq w ω) seed (List.ofFn p) k).tree b :=
    fun b hb e₁ e₂ m hm => agree_of_noise K O F fun v hv => h _
      (Finset.mem_biUnion.2 ⟨_, mem_nodeReads e₁ e₂ hb hm, Finset.mem_image.2 ⟨v, hv, rfl⟩⟩)
  have h0 := phase_congr K B F _ _ hseed hws k hag
  exact ⟨h0.symm, (phase_succ_congr_of K B F _ _ hseed hws h0 hag).symm⟩

/-- The round's pass is decided by the oracle's bits at what it reads against its final tree. -/
theorem roundEnd_determined (K : StageKnobs α) (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (seed : List (FreeMonoid α)) {N L : ℕ}
    (p : Fin N → FreeMonoid α) (hp : ∀ i, (p i).toList.length ≤ L) (ω ω' : Ω)
    (h : ∀ x ∈ readBits K F (nodeReads (passBases seed (List.ofFn p) L)
        (roundEnd K O B F seed (ω, p)).tree), O.noise x ω = O.noise x ω') :
    roundEnd K O B F seed (ω', p) = roundEnd K O B F seed (ω, p) := by
  rw [roundEnd_eq_phase, roundEnd_eq_phase] at *
  exact (phase_determined K O B F seed p hp ω ω' N h).1

end Reads

end OrthoDFA
