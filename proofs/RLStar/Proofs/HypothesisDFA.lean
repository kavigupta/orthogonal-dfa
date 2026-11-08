import RLStar.LowLevelClaims.HypothesisDFA
import RLStar.Proofs.DTree

/-!
# The walk in a hypothesis DFA, and prefixes

A walk starts at its leaf, follows one learned edge per letter, and stops short only at an
unlearned one; `walkEnd` is its last leaf when it follows every letter.
-/

namespace OrthoDFA

variable {α : Type*}

section Walk

variable (E : Edges α)

theorem walk_cons (s : List Bool) (cs : List α) : ∃ l, walk E s cs = s :: l := by
  cases cs <;> simp [walk]

theorem walk_getD_zero (s : List Bool) (cs : List α) : (walk E s cs).getD 0 [] = s := by
  obtain ⟨l, hl⟩ := walk_cons E s cs; simp [hl]

theorem length_walk_pos (s : List Bool) (cs : List α) : 0 < (walk E s cs).length := by
  obtain ⟨l, hl⟩ := walk_cons E s cs; simp [hl]

theorem length_walk_le : ∀ (s : List Bool) (cs : List α), (walk E s cs).length ≤ cs.length + 1
  | s, [] => by simp [walk]
  | s, c :: cs => by
    simp only [walk, List.length_cons, add_le_add_iff_right]
    split
    · exact length_walk_le _ cs
    · simp

theorem walk_succ : ∀ (s : List Bool) (cs : List α) (i : ℕ), i + 1 < (walk E s cs).length →
    ∃ c w, cs[i]? = some c ∧
      E ((walk E s cs).getD i [], c) = some ((walk E s cs).getD (i + 1) [], w)
  | s, [], i, h => by simp [walk] at h
  | s, c :: cs, i, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · simp [walk, hE] at h
    · have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      rw [hw] at h ⊢
      rcases i with _ | i
      · obtain ⟨l, hl⟩ := walk_cons E t cs
        exact ⟨c, w, rfl, by simpa [hl] using hE⟩
      · simp only [List.length_cons, add_lt_add_iff_right] at h
        obtain ⟨c', w', h1, h2⟩ := walk_succ t cs i h
        exact ⟨c', w', by simpa using h1, by simpa using h2⟩

theorem walk_stop : ∀ (s : List Bool) (cs : List α), (walk E s cs).length ≤ cs.length →
    ∃ c, cs[(walk E s cs).length - 1]? = some c ∧
      E ((walk E s cs).getD ((walk E s cs).length - 1) [], c) = none
  | s, [], h => by simp [walk] at h
  | s, c :: cs, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · have hw : walk E s (c :: cs) = [s] := by simp [walk, hE]
      rw [hw]
      exact ⟨c, rfl, by simpa using hE⟩
    · have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      rw [hw] at h ⊢
      simp only [List.length_cons, add_le_add_iff_right] at h
      obtain ⟨c', h1, h2⟩ := walk_stop t cs h
      have hpos := length_walk_pos E t cs
      refine ⟨c', ?_, ?_⟩
      · simp only [List.length_cons, Nat.add_sub_cancel]
        rw [show (walk E t cs).length = (walk E t cs).length - 1 + 1 by omega]
        simpa using h1
      · simp only [List.length_cons, Nat.add_sub_cancel]
        rw [show (walk E t cs).length = (walk E t cs).length - 1 + 1 by omega]
        simpa using h2

theorem walk_mem {L : List (List Bool)} (hE : ∀ q ∈ L, ∀ c t w, E (q, c) = some (t, w) → t ∈ L) :
    ∀ (s : List Bool) (cs : List α), s ∈ L → ∀ q ∈ walk E s cs, q ∈ L
  | s, [], hs, q, hq => by simp only [walk, List.mem_singleton] at hq; rwa [hq]
  | s, c :: cs, hs, q, hq => by
    rcases hE' : E (s, c) with _ | ⟨t, w⟩
    · simp only [walk, hE', List.mem_cons, List.not_mem_nil, or_false] at hq; rwa [hq]
    · simp only [walk, hE', List.mem_cons] at hq
      rcases hq with rfl | hq
      · exact hs
      · exact walk_mem hE t cs (hE s hs c t w hE') q hq

theorem walkEnd_walk : ∀ (s : List Bool) (cs : List α) (q : List Bool), walkEnd E s cs = some q →
    (walk E s cs).length = cs.length + 1 ∧ (walk E s cs).getD cs.length [] = q
  | s, [], q, h => by simp only [walkEnd, Option.some.injEq] at h; simp [walk, h]
  | s, c :: cs, q, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · simp [walkEnd, hE] at h
    · simp only [walkEnd, hE, Option.bind_some] at h
      have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      obtain ⟨h1, h2⟩ := walkEnd_walk t cs q h
      rw [hw]
      exact ⟨by simp [h1], by simpa using h2⟩

theorem walk_walkEnd : ∀ (s : List Bool) (cs : List α) (q : List Bool),
    (walk E s cs).length = cs.length + 1 → (walk E s cs).getLast? = some q →
      walkEnd E s cs = some q
  | s, [], q, _, h => by simpa [walk, walkEnd] using h
  | s, c :: cs, q, hl, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · simp [walk, hE] at hl
    · have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      rw [hw] at hl h
      have hpos := length_walk_pos E t cs
      have hne : walk E t cs ≠ [] := List.ne_nil_of_length_pos hpos
      simp only [walkEnd, hE, Option.bind_some]
      refine walk_walkEnd t cs q (by simpa using hl) ?_
      rw [List.getLast?_cons] at h
      rcases hlast : (walk E t cs).getLast? with _ | v
      · exact absurd (List.getLast?_eq_none_iff.1 hlast) hne
      · rw [hlast] at h
        simpa using h

end Walk

theorem prefixOf_succ {x : FreeMonoid α} {p : ℕ} {c : α} (h : x.toList[p]? = some c) :
    prefixOf x (p + 1) = prefixOf x p * FreeMonoid.of c := by
  simp only [prefixOf, List.take_add_one, h, Option.toList_some]
  rfl

theorem prefixOf_of_le {x : FreeMonoid α} {p : ℕ} (h : x.toList.length ≤ p) : prefixOf x p = x := by
  simp [prefixOf, List.take_of_length_le h]

theorem prefixOf_length (w : FreeMonoid α) : prefixOf w w.toList.length = w := by
  simp [prefixOf]

theorem walkCorrect : WalkCorrect := by
  intro α
  refine ⟨fun H s cs => ⟨?_, length_walk_le H.edges s cs, walk_succ H.edges s cs,
    walk_stop H.edges s cs, ?_, fun hE hs => walk_mem H.edges hE s cs hs⟩,
    fun x i => ⟨fun c h => prefixOf_succ h, prefixOf_of_le⟩⟩
  · obtain ⟨l, hl⟩ := walk_cons H.edges s cs
    simp [hl]
  · intro q
    constructor
    · intro h
      obtain ⟨h1, h2⟩ := walkEnd_walk H.edges s cs q h
      refine ⟨h1, ?_⟩
      rw [List.getLast?_eq_getElem?, h1, Nat.add_sub_cancel, ← h2, List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem (by omega), Option.getD_some]
    · rintro ⟨h1, h2⟩
      exact walk_walkEnd H.edges s cs q h1 h2

section Edges

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α)

variable {T E}

theorem EdgesOK.restrict (hE : EdgesOK read T E) (P : List Bool × α → Prop) [DecidablePred P] :
    EdgesOK read T fun e => if P e then E e else none := fun q hq c t w h => by
  dsimp only at h
  by_cases hP : P (q, c)
  · rw [if_pos hP] at h
    exact hE q hq c t w h
  · rw [if_neg hP] at h
    cases h

theorem edgesOK_retarget (hE : EdgesOK read T E) {p : List Bool}
    (hp : p ∈ T.leaves) (d : FreeMonoid α) :
    EdgesOK read (T.splitAt d p) (retarget read (T.splitAt d p) p E) := by
  intro q hq c t w h
  simp only [retarget_eq] at h
  cases hpq : p.isPrefixOf q
  · obtain ⟨hq', hne⟩ := DTree.prefix_of_mem_leaves_splitAt hp hq hpq
    simp only [hpq, Bool.false_eq_true, if_false] at h
    rcases hEq : E (q, c) with _ | ⟨t₁, w₁⟩ <;> rw [hEq] at h
    · simp at h
    · obtain ⟨hw, hwc⟩ := hE q hq' c t₁ w₁ hEq
      simp only at h
      split at h
      · rename_i ht₁
        rcases hs : (T.splitAt d p).sift read (w₁ * FreeMonoid.of c) with t' | z <;>
          rw [hs] at h <;> simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq,
            Prod.mk.injEq, reduceCtorEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨DTree.sift_splitAt_ne read hw hne, hs⟩
      · rename_i ht₁
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨DTree.sift_splitAt_ne read hw hne, DTree.sift_splitAt_ne read hwc ht₁⟩
  · simp [hpq] at h

theorem retarget_eq_none_of_not_mem {p : List Bool} (hp : p ∈ T.leaves) {d : FreeMonoid α}
    {e : List Bool × α} (he : e.1 ∈ (T.splitAt d p).leaves) (hn : e.1 ∉ T.leaves) :
    retarget read (T.splitAt d p) p E e = none := by
  rw [retarget_eq, if_pos]
  rcases (DTree.mem_leaves_splitAt d T p hp e.1).1 he with ⟨h, -⟩ | h | h
  · exact absurd h hn
  all_goals rw [h]; exact List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)

variable [DecidableEq α]

section Split

variable [Fintype α] {H : HypothesisDFA α} {p : List Bool}

theorem HypothesisDFA.split_tree (d : FreeMonoid α) :
    (H.split read d p).tree = H.tree.splitAt d p := by
  rw [HypothesisDFA.split_eq]

theorem edgesOK_split (hE : EdgesOK read H.tree H.edges) (hp : p ∈ H.tree.leaves)
    (d : FreeMonoid α) : EdgesOK read (H.split read d p).tree (H.split read d p).edges := by
  rw [HypothesisDFA.split_eq]
  exact (edgesOK_retarget read hE hp d).restrict _ _

/-- At the split tree's leaves the split's edges are every edge retargeted. -/
theorem split_edges_of_mem (hp : p ∈ H.tree.leaves) (d : FreeMonoid α) {e : List Bool × α}
    (he : e.1 ∈ (H.tree.splitAt d p).leaves) :
    (H.split read d p).edges e = retarget read (H.tree.splitAt d p) p H.edges e := by
  rw [HypothesisDFA.split_eq]
  dsimp only
  split_ifs with h
  · rfl
  · exact (retarget_eq_none_of_not_mem read hp he h).symm

theorem edgesAtLeaves_split (d : FreeMonoid α) : EdgesAtLeaves (H.split read d p) := by
  intro e t w he
  rw [HypothesisDFA.split_eq] at he ⊢
  dsimp only at he ⊢
  by_cases hq : e.1 ∈ H.tree.leaves
  swap
  · simp [hq] at he
  rw [if_pos hq, retarget_eq] at he
  by_cases hpe : p.isPrefixOf e.1 = true
  · simp [hpe] at he
  exact DTree.mem_leaves_splitAt_of_ne d H.tree p e.1 hq fun h => hpe (by
    rw [h]; exact List.isPrefixOf_iff_prefix.2 (List.prefix_refl _))

end Split

theorem edgesOK_upd (hE : EdgesOK read T E) {p t : List Bool} {c : α}
    {u : FreeMonoid α} (hu : T.sift read u = .inl p)
    (huc : T.sift read (u * FreeMonoid.of c) = .inl t) :
    EdgesOK read T (Function.update E (p, c) (some (t, u))) := by
  intro q hq c' t' w h
  by_cases hqc : (q, c') = (p, c)
  · simp only [Prod.mk.injEq] at hqc
    obtain ⟨rfl, rfl⟩ := hqc
    rw [Function.update_self] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨hu, huc⟩
  · rw [Function.update_of_ne hqc] at h
    exact hE q hq c' t' w h

end Edges

theorem splitAndLearnCorrect : SplitAndLearnCorrect := by
  intro α _ _ read H hE
  exact ⟨fun d p hp => edgesOK_split read hE hp d,
    fun u s t c hu huc => edgesOK_upd read hE hu huc⟩

end OrthoDFA
