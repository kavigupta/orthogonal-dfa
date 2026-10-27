import OrthoDFA.LowLevelClaims.DTree
import Mathlib.Tactic

/-!
# Sifting and splitting are correct

By induction on the tree: a node reads `z` followed by its midfix, and a decided read hands the
rest of the sift to the child it picks. A split rebuilds only the path to its leaf.
-/

namespace OrthoDFA

namespace DTree

variable {α : Type*} (cut : FreeMonoid α → Option Bool)

theorem siftWithPath_reads : ∀ (T : DTree α) (z : FreeMonoid α) (i : ℕ),
    i < (T.siftWithPath cut z).1.length →
    (T.siftWithPath cut z).1[i]?
      = some (z * T.midfixAt (((T.siftWithPath cut z).1.filterMap cut).take i))
  | .leaf, z, i, h => by simp [siftWithPath] at h
  | .node m r a, z, i, h => by
    rcases hc : cut (z * m) with _ | _ | _
    · simp only [siftWithPath, hc] at h ⊢
      rcases i with _ | i
      · simp [midfixAt]
      · simp at h
    · simp only [siftWithPath, hc] at h ⊢
      rcases i with _ | i
      · simp [midfixAt]
      · simp only [List.getElem?_cons_succ, List.filterMap_cons, hc, List.take_succ_cons, midfixAt]
        exact siftWithPath_reads r z i (by simpa using h)
    · simp only [siftWithPath, hc] at h ⊢
      rcases i with _ | i
      · simp [midfixAt]
      · simp only [List.getElem?_cons_succ, List.filterMap_cons, hc, List.take_succ_cons, midfixAt]
        exact siftWithPath_reads a z i (by simpa using h)

theorem siftWithPath_length : ∀ (T : DTree α) (z : FreeMonoid α),
    (T.siftWithPath cut z).1.length ≤ T.depth
  | .leaf, z => by simp [siftWithPath, depth]
  | .node m r a, z => by
    simp only [siftWithPath, depth]
    split
    · have := siftWithPath_length a z; simp only [List.length_cons]; omega
    · have := siftWithPath_length r z; simp only [List.length_cons]; omega
    · simp

theorem siftWithPath_decided : ∀ (T : DTree α) (z : FreeMonoid α) (p : List Bool),
    (T.siftWithPath cut z).2 = some p →
      p ∈ T.leaves ∧ p = (T.siftWithPath cut z).1.filterMap cut
        ∧ ∀ w ∈ (T.siftWithPath cut z).1, cut w ≠ none
  | .leaf, z, p, h => by
    simp only [siftWithPath, Option.some.injEq] at h
    subst h
    simp [leaves, siftWithPath]
  | .node m r a, z, p, h => by
    simp only [siftWithPath] at h ⊢
    split at h
    · rename_i hc
      obtain ⟨p', hp', rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := siftWithPath_decided a z p' hp'
      refine ⟨by simp [leaves, h1], by simp [hc, ← h2], ?_⟩
      intro w hw
      rcases List.mem_cons.1 hw with rfl | hw
      · simp [hc]
      · exact h3 w hw
    · rename_i hc
      obtain ⟨p', hp', rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := siftWithPath_decided r z p' hp'
      refine ⟨by simp [leaves, h1], by simp [hc, ← h2], ?_⟩
      intro w hw
      rcases List.mem_cons.1 hw with rfl | hw
      · simp [hc]
      · exact h3 w hw
    · simp at h

theorem siftWithPath_undecided : ∀ (T : DTree α) (z : FreeMonoid α),
    (T.siftWithPath cut z).2 = none →
    ∃ w, (T.siftWithPath cut z).1.getLast? = some w ∧ cut w = none
      ∧ ∀ w' ∈ (T.siftWithPath cut z).1.dropLast, cut w' ≠ none
  | .leaf, z, h => by simp [siftWithPath] at h
  | .node m r a, z, h => by
    simp only [siftWithPath] at h ⊢
    split at h
    · rename_i hc
      have h' : (a.siftWithPath cut z).2 = none := by simpa using h
      obtain ⟨w, hw, hcw, hrest⟩ := siftWithPath_undecided a z h'
      have hne : (a.siftWithPath cut z).1 ≠ [] := by rintro he; simp [he] at hw
      refine ⟨w, by rw [List.getLast?_cons, hw]; rfl, hcw, ?_⟩
      rw [List.dropLast_cons_of_ne_nil hne]
      intro w' hw'
      rcases List.mem_cons.1 hw' with rfl | hw'
      · simp [hc]
      · exact hrest w' hw'
    · rename_i hc
      have h' : (r.siftWithPath cut z).2 = none := by simpa using h
      obtain ⟨w, hw, hcw, hrest⟩ := siftWithPath_undecided r z h'
      have hne : (r.siftWithPath cut z).1 ≠ [] := by rintro he; simp [he] at hw
      refine ⟨w, by rw [List.getLast?_cons, hw]; rfl, hcw, ?_⟩
      rw [List.dropLast_cons_of_ne_nil hne]
      intro w' hw'
      rcases List.mem_cons.1 hw' with rfl | hw'
      · simp [hc]
      · exact hrest w' hw'
    · rename_i hc
      simp [hc]

end DTree

theorem siftWithPathCorrect : SiftWithPathCorrect := fun cut T z =>
  ⟨fun i h => by
    have := DTree.siftWithPath_reads cut T z i h
    rwa [List.getElem?_eq_getElem h, Option.some.injEq] at this,
  DTree.siftWithPath_length cut T z, DTree.siftWithPath_decided cut T z,
  DTree.siftWithPath_undecided cut T z⟩


namespace DTree

variable {α : Type*} (read : FreeMonoid α → ARU)

theorem sift_inl_mem : ∀ (T : DTree α) (z : FreeMonoid α) (p : List Bool),
    T.sift read z = .inl p → p ∈ T.leaves
  | .leaf, z, p, h => by simp only [sift, Sum.inl.injEq] at h; simp [leaves, ← h]
  | .node m r a, z, p, h => by
    simp only [sift] at h
    split at h
    · rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      subst h
      simp [leaves, sift_inl_mem a z q hq]
    · rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      subst h
      simp [leaves, sift_inl_mem r z q hq]
    · simp at h

theorem sift_inr : ∀ (T : DTree α) (z z' : FreeMonoid α),
    T.sift read z = .inr z' → read z' = .undecided
  | .leaf, z, z', h => by simp [sift] at h
  | .node m r a, z, z', h => by
    simp only [sift] at h
    split at h
    · rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inr.injEq, reduceCtorEq] at h
      subst h; exact sift_inr a z q hq
    · rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inr.injEq, reduceCtorEq] at h
      subst h; exact sift_inr r z q hq
    · rename_i hu; simp only [Sum.inr.injEq] at h; subst h; exact hu

theorem leaves_nodup : ∀ T : DTree α, T.leaves.Nodup
  | .leaf => by simp [leaves]
  | .node _ r a => by
    simp only [leaves]
    refine List.Nodup.append ((leaves_nodup r).map (fun _ _ h => List.cons_injective h))
      ((leaves_nodup a).map (fun _ _ h => List.cons_injective h)) ?_
    simp [List.disjoint_left]

theorem leaves_ne_nil : ∀ T : DTree α, T.leaves ≠ []
  | .leaf => by simp [leaves]
  | .node _ r a => by simp [leaves, leaves_ne_nil r]

/-- A split replaces its leaf's sifts by the new node's. -/
theorem sift_splitAt (d : FreeMonoid α) : ∀ (T : DTree α) (p : List Bool) (z : FreeMonoid α),
    (T.splitAt d p).sift read z = (T.sift read z).elim
      (fun q => if q = p then ((DTree.node d .leaf .leaf).sift read z).map (p ++ ·) id
        else .inl q) .inr
  | .leaf, [], z => by cases hd : read (z * d) <;> simp [splitAt, sift, hd]
  | .leaf, b :: p, z => by simp [splitAt, sift]
  | .node m r a, [], z => by
    simp only [splitAt]
    rcases h : (DTree.node m r a).sift read z with q | q
    · have hq : q ≠ [] := by
        have := sift_inl_mem read _ z q h
        rintro rfl
        simp [leaves] at this
      simp [hq]
    · simp
  | .node m r a, false :: p, z => by
    simp only [splitAt, sift]
    split
    · rcases a.sift read z with q | q <;> simp
    · rw [sift_splitAt d r p z]
      rcases r.sift read z with q | q
      · by_cases hq : q = p
        · subst hq
          cases hd : read (z * d) <;> simp [sift, hd]
        · simp [hq]
      · simp
    · simp
  | .node m r a, true :: p, z => by
    simp only [splitAt, sift]
    split
    · rw [sift_splitAt d a p z]
      rcases a.sift read z with q | q
      · by_cases hq : q = p
        · subst hq
          cases hd : read (z * d) <;> simp [sift, hd]
        · simp [hq]
      · simp
    · rcases r.sift read z with q | q <;> simp
    · simp

theorem mem_leaves_splitAt (d : FreeMonoid α) : ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves →
    ∀ q, q ∈ (T.splitAt d p).leaves ↔
      (q ∈ T.leaves ∧ q ≠ p) ∨ q = p ++ [false] ∨ q = p ++ [true]
  | .leaf, p, hp, q => by
    simp only [leaves, List.mem_singleton] at hp
    subst hp
    simp [splitAt, leaves]
  | .node m r a, [], hp, q => by simp [leaves] at hp
  | .node m r a, false :: p, hp, q => by
    have hp' : p ∈ r.leaves := by simpa [leaves] using hp
    simp only [splitAt, leaves, List.mem_append, List.mem_map]
    constructor
    · rintro (⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩)
      · rcases (mem_leaves_splitAt d r p hp' q').1 hq' with ⟨h1, h2⟩ | h | h
        · exact Or.inl ⟨Or.inl ⟨q', h1, rfl⟩, by simpa using h2⟩
        · exact Or.inr (Or.inl (by simp [h]))
        · exact Or.inr (Or.inr (by simp [h]))
      · exact Or.inl ⟨Or.inr ⟨q', hq', rfl⟩, by simp⟩
    · rintro (⟨⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩, hne⟩ | rfl | rfl)
      · exact Or.inl ⟨q', (mem_leaves_splitAt d r p hp' q').2 (Or.inl ⟨hq', by simpa using hne⟩),
          rfl⟩
      · exact Or.inr ⟨q', hq', rfl⟩
      · exact Or.inl ⟨p ++ [false], (mem_leaves_splitAt d r p hp' _).2 (Or.inr (Or.inl rfl)),
          rfl⟩
      · exact Or.inl ⟨p ++ [true], (mem_leaves_splitAt d r p hp' _).2 (Or.inr (Or.inr rfl)),
          rfl⟩
  | .node m r a, true :: p, hp, q => by
    have hp' : p ∈ a.leaves := by simpa [leaves] using hp
    simp only [splitAt, leaves, List.mem_append, List.mem_map]
    constructor
    · rintro (⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩)
      · exact Or.inl ⟨Or.inl ⟨q', hq', rfl⟩, by simp⟩
      · rcases (mem_leaves_splitAt d a p hp' q').1 hq' with ⟨h1, h2⟩ | h | h
        · exact Or.inl ⟨Or.inr ⟨q', h1, rfl⟩, by simpa using h2⟩
        · exact Or.inr (Or.inl (by simp [h]))
        · exact Or.inr (Or.inr (by simp [h]))
    · rintro (⟨⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩, hne⟩ | rfl | rfl)
      · exact Or.inl ⟨q', hq', rfl⟩
      · exact Or.inr ⟨q', (mem_leaves_splitAt d a p hp' q').2 (Or.inl ⟨hq', by simpa using hne⟩),
          rfl⟩
      · exact Or.inr ⟨p ++ [false], (mem_leaves_splitAt d a p hp' _).2 (Or.inr (Or.inl rfl)),
          rfl⟩
      · exact Or.inr ⟨p ++ [true], (mem_leaves_splitAt d a p hp' _).2 (Or.inr (Or.inr rfl)),
          rfl⟩

theorem length_leaves_splitAt (d : FreeMonoid α) : ∀ (T : DTree α) (p : List Bool),
    p ∈ T.leaves → (T.splitAt d p).leaves.length = T.leaves.length + 1
  | .leaf, p, hp => by
    simp only [leaves, List.mem_singleton] at hp
    subst hp
    simp [splitAt, leaves]
  | .node m r a, [], hp => by simp [leaves] at hp
  | .node m r a, false :: p, hp => by
    have hp' : p ∈ r.leaves := by simpa [leaves] using hp
    simp only [splitAt, leaves, List.length_append, List.length_map,
      length_leaves_splitAt d r p hp']
    omega
  | .node m r a, true :: p, hp => by
    have hp' : p ∈ a.leaves := by simpa [leaves] using hp
    simp only [splitAt, leaves, List.length_append, List.length_map,
      length_leaves_splitAt d a p hp']
    omega

/-- Two strings at different leaves read decidedly apart at the node where the paths part. -/
theorem sift_part : ∀ (T : DTree α) (z₁ z₂ : FreeMonoid α) (p₁ p₂ : List Bool),
    T.sift read z₁ = .inl p₁ → T.sift read z₂ = .inl p₂ → p₁ ≠ p₂ →
    read (z₁ * T.midfixAt (commonPrefix p₁ p₂)) ≠ .undecided
      ∧ read (z₂ * T.midfixAt (commonPrefix p₁ p₂)) ≠ .undecided
      ∧ read (z₁ * T.midfixAt (commonPrefix p₁ p₂)) ≠ read (z₂ * T.midfixAt (commonPrefix p₁ p₂))
  | .leaf, z₁, z₂, p₁, p₂, h₁, h₂, hne => by
    simp only [sift, Sum.inl.injEq] at h₁ h₂; subst h₁ h₂; exact absurd rfl hne
  | .node m r a, z₁, z₂, p₁, p₂, h₁, h₂, hne => by
    simp only [sift] at h₁ h₂
    split at h₁ <;> rename_i hr₁ <;> split at h₂ <;> rename_i hr₂
    · rcases hq₁ : a.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : a.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      have hq : q₁ ≠ q₂ := fun h => hne (by rw [h])
      simpa [commonPrefix, midfixAt] using sift_part a z₁ z₂ q₁ q₂ hq₁ hq₂ hq
    · rcases hq₁ : a.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : r.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      simp [commonPrefix, midfixAt, hr₁, hr₂]
    · simp at h₂
    · rcases hq₁ : r.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : a.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      simp [commonPrefix, midfixAt, hr₁, hr₂]
    · rcases hq₁ : r.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : r.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      have hq : q₁ ≠ q₂ := fun h => hne (by rw [h])
      simpa [commonPrefix, midfixAt] using sift_part r z₁ z₂ q₁ q₂ hq₁ hq₂ hq
    · simp at h₂
    · simp at h₁
    · simp at h₁
    · simp at h₁

/-- Where every read of a string followed by a midfix is the same for `z` and `z'`, they sift
to the same leaf. -/
theorem sift_congr (z z' : FreeMonoid α) (h : ∀ m, read (z * m) = read (z' * m)) :
    ∀ (T : DTree α) (p : List Bool), T.sift read z = .inl p → T.sift read z' = .inl p
  | .leaf, p, hp => by simpa [sift] using hp
  | .node m r a, p, hp => by
    simp only [sift, ← h m] at hp ⊢
    split at hp
    · rcases hq : a.sift read z with q | q <;> rw [hq] at hp <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at hp
      rw [sift_congr z z' h a q hq]; simpa using hp
    · rcases hq : r.sift read z with q | q <;> rw [hq] at hp <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at hp
      rw [sift_congr z z' h r q hq]; simpa using hp
    · simp at hp

theorem sift_inr_mul : ∀ (T : DTree α) (z z' : FreeMonoid α),
    T.sift read z = .inr z' → ∃ m, z' = z * m
  | .leaf, z, z', h => by simp [sift] at h
  | .node m r a, z, z', h => by
    simp only [sift] at h
    split at h
    · rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inr.injEq, reduceCtorEq] at h
      subst h; exact sift_inr_mul a z q hq
    · rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inr.injEq, reduceCtorEq] at h
      subst h; exact sift_inr_mul r z q hq
    · simp only [Sum.inr.injEq] at h; exact ⟨m, h.symm⟩

theorem sift_splitAt_ne {T : DTree α} {d z : FreeMonoid α} {p q : List Bool}
    (h : T.sift read z = .inl q) (hne : q ≠ p) : (T.splitAt d p).sift read z = .inl q := by
  rw [sift_splitAt, h]; simp [hne]

theorem sift_splitAt_eq {T : DTree α} {d z : FreeMonoid α} {p : List Bool}
    (h : T.sift read z = .inl p) :
    (T.splitAt d p).sift read z = ((DTree.node d .leaf .leaf).sift read z).map (p ++ ·) id := by
  rw [sift_splitAt, h]; simp

theorem prefix_of_mem_leaves_splitAt {T : DTree α} {p q : List Bool} {d : FreeMonoid α}
    (hp : p ∈ T.leaves) (hq : q ∈ (T.splitAt d p).leaves) (hpq : p.isPrefixOf q = false) :
    q ∈ T.leaves ∧ q ≠ p := by
  rcases (mem_leaves_splitAt d T p hp q).1 hq with h | rfl | rfl
  · exact h
  · rw [List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)] at hpq; simp at hpq
  · rw [List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)] at hpq; simp at hpq

theorem nil_not_mem_leaves_node {m : FreeMonoid α} {r a : DTree α} :
    [] ∉ (DTree.node m r a).leaves := by
  simp [leaves]

theorem mem_leaves_splitAt_of_ne (d : FreeMonoid α) :
    ∀ (T : DTree α) (p q : List Bool), q ∈ T.leaves → q ≠ p → q ∈ (T.splitAt d p).leaves
  | .leaf, [], q, hq, hne => by simp only [leaves, List.mem_singleton] at hq; exact absurd hq hne
  | .leaf, _ :: _, q, hq, _ => by simpa [splitAt] using hq
  | .node _ _ _, [], q, hq, _ => by simpa [splitAt] using hq
  | .node n r a, false :: p, q, hq, hne => by
    simp only [splitAt, leaves, List.mem_append, List.mem_map] at hq ⊢
    rcases hq with ⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩
    · exact .inl ⟨q', mem_leaves_splitAt_of_ne d r p q' hq' (fun h => hne (by rw [h])), rfl⟩
    · exact .inr ⟨q', hq', rfl⟩
  | .node n r a, true :: p, q, hq, hne => by
    simp only [splitAt, leaves, List.mem_append, List.mem_map] at hq ⊢
    rcases hq with ⟨q', hq', rfl⟩ | ⟨q', hq', rfl⟩
    · exact .inl ⟨q', hq', rfl⟩
    · exact .inr ⟨q', mem_leaves_splitAt_of_ne d a p q' hq' (fun h => hne (by rw [h])), rfl⟩

theorem splitAt_ne {d : FreeMonoid α} {T : DTree α} {p : List Bool} (hp : p ∈ T.leaves) :
    T.splitAt d p ≠ T := fun h => by
  have := length_leaves_splitAt d T p hp
  rw [h] at this
  omega

theorem splitAt_inj {d d' : FreeMonoid α} :
    ∀ {T : DTree α} {p p' : List Bool}, p ∈ T.leaves → p' ∈ T.leaves →
      T.splitAt d p = T.splitAt d' p' → p = p' ∧ d = d'
  | .leaf, p, p', hp, hp', h => by
    simp only [leaves, List.mem_singleton] at hp hp'
    subst hp hp'
    simp only [splitAt, node.injEq] at h
    exact ⟨rfl, h.1⟩
  | .node n r a, [], _, hp, _, _ => absurd hp nil_not_mem_leaves_node
  | .node n r a, _, [], _, hp', _ => absurd hp' nil_not_mem_leaves_node
  | .node n r a, b :: p, b' :: p', hp, hp', h => by
    simp only [leaves, List.mem_append, List.mem_map, List.cons.injEq] at hp hp'
    rcases hp with ⟨q, hq, rfl, rfl⟩ | ⟨q, hq, rfl, rfl⟩ <;>
      rcases hp' with ⟨q', hq', rfl, rfl⟩ | ⟨q', hq', rfl, rfl⟩ <;>
      simp only [splitAt, node.injEq, true_and, and_true] at h
    · obtain ⟨rfl, rfl⟩ := splitAt_inj hq hq' h; exact ⟨rfl, rfl⟩
    · exact absurd h.1 (splitAt_ne hq)
    · exact absurd h.2 (splitAt_ne hq)
    · obtain ⟨rfl, rfl⟩ := splitAt_inj hq hq' h; exact ⟨rfl, rfl⟩

theorem length_lt_of_mem_leaves :
    ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves → p.length < T.leaves.length
  | .leaf, p, h => by simp [leaves] at h; simp [h, leaves]
  | .node _ r a, p, h => by
    simp only [leaves, List.mem_append, List.mem_map, List.length_append,
      List.length_map] at h ⊢
    have hr := List.length_pos_of_ne_nil (leaves_ne_nil r)
    have ha := List.length_pos_of_ne_nil (leaves_ne_nil a)
    rcases h with ⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩
    · have := length_lt_of_mem_leaves r q hq; simp; omega
    · have := length_lt_of_mem_leaves a q hq; simp; omega

theorem snoc_not_mem_leaves : ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves → ∀ b : Bool,
    p ++ [b] ∉ T.leaves
  | .leaf, p, hp, b => by simp [leaves] at hp ⊢
  | .node m r a, [], hp, b => by simp [leaves] at hp
  | .node m r a, false :: p, hp, b => by
    have hp' : p ∈ r.leaves := by simpa [leaves] using hp
    simpa [leaves] using snoc_not_mem_leaves r p hp' b
  | .node m r a, true :: p, hp, b => by
    have hp' : p ∈ a.leaves := by simpa [leaves] using hp
    simpa [leaves] using snoc_not_mem_leaves a p hp' b

theorem midfixAt_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves → (T.splitAt d p).midfixAt p = d
  | .leaf, p, hp => by
    simp only [leaves, List.mem_singleton] at hp
    subst hp
    simp [splitAt, midfixAt]
  | .node m r a, [], hp => absurd hp nil_not_mem_leaves_node
  | .node m r a, false :: p, hp => by
    simp only [splitAt, midfixAt]
    exact midfixAt_splitAt d r p (by simpa [leaves] using hp)
  | .node m r a, true :: p, hp => by
    simp only [splitAt, midfixAt]
    exact midfixAt_splitAt d a p (by simpa [leaves] using hp)

theorem sift_eq_siftWithPath_of (c : FreeMonoid α → Option Bool) (hc : ∀ w, c w = (read w).cut) :
    ∀ (T : DTree α) (z : FreeMonoid α), T.sift read z = match (T.siftWithPath c z).2 with
      | some p => .inl p
      | none => .inr ((T.siftWithPath c z).1.getLastD 1)
  | .leaf, z => by simp [sift, siftWithPath]
  | .node m r a, z => by
    have hm := hc (z * m)
    simp only [sift, siftWithPath]
    rcases hr : read (z * m) with _ | _ | _ <;> rw [hr] at hm <;> simp only [ARU.cut] at hm <;>
      simp only [hm]
    · rw [sift_eq_siftWithPath_of c hc a z]
      rcases hp : (a.siftWithPath c z).2 with _ | q
      · obtain ⟨w, hw, -, -⟩ := siftWithPath_undecided _ a z hp
        simp [List.getLastD_eq_getLast?, List.getLast?_cons, hw]
      · simp
    · rw [sift_eq_siftWithPath_of c hc r z]
      rcases hp : (r.siftWithPath c z).2 with _ | q
      · obtain ⟨w, hw, -, -⟩ := siftWithPath_undecided _ r z hp
        simp [List.getLastD_eq_getLast?, List.getLast?_cons, hw]
      · simp
    · simp

/-- `sift` is `siftWithPath` under the read's cut: its leaf, or its last read. -/
theorem sift_eq_siftWithPath (T : DTree α) (z : FreeMonoid α) :
    T.sift read z = match (T.siftWithPath (fun w => (read w).cut) z).2 with
      | some p => .inl p
      | none => .inr ((T.siftWithPath (fun w => (read w).cut) z).1.getLastD 1) :=
  sift_eq_siftWithPath_of read _ (fun _ => rfl) T z

theorem sift_eq_inl_iff (T : DTree α) (z : FreeMonoid α) (p : List Bool) :
    T.sift read z = .inl p ↔ (T.siftWithPath (fun w => (read w).cut) z).2 = some p := by
  rw [sift_eq_siftWithPath]
  rcases (T.siftWithPath (fun w => (read w).cut) z).2 with _ | q <;> simp

theorem sift_eq_inr_iff (T : DTree α) (z w : FreeMonoid α) :
    T.sift read z = .inr w ↔ (T.siftWithPath (fun w => (read w).cut) z).2 = none
      ∧ (T.siftWithPath (fun w => (read w).cut) z).1.getLast? = some w := by
  rw [sift_eq_siftWithPath]
  rcases hp : (T.siftWithPath (fun w => (read w).cut) z).2 with _ | q
  · obtain ⟨w', hw', -, -⟩ := siftWithPath_undecided _ T z hp
    simp [List.getLastD_eq_getLast?, hw']
  · simp

end DTree

theorem siftAndSplitCorrect : SiftAndSplitCorrect := fun read T z =>
  ⟨DTree.sift_eq_inl_iff read T z, DTree.sift_eq_inr_iff read T z,
    fun z' p p' h h' hne => DTree.sift_part read T z z' p p' h h' hne,
    fun d p hp => ⟨DTree.mem_leaves_splitAt d T p hp, DTree.length_leaves_splitAt d T p hp,
      DTree.midfixAt_splitAt d T p hp, DTree.sift_splitAt read d T p z⟩⟩

end OrthoDFA
