import OrthoDFA.IdealRound

/-!
# The tree under ideal reads

Sifting lands on a leaf or on an undecided read; a split changes only the sifts that landed on
its leaf; two strings at different leaves read decidedly apart at the node where their paths
part; and with reads fixed by the target state, so is the leaf a string sifts to.
-/

namespace OrthoDFA

namespace Ideal

variable {α : Type*} (read : FreeMonoid α → ARU)

namespace DTree

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
    read (z₁ * T.midAt (lcp p₁ p₂)) ≠ .undecided ∧ read (z₂ * T.midAt (lcp p₁ p₂)) ≠ .undecided
      ∧ read (z₁ * T.midAt (lcp p₁ p₂)) ≠ read (z₂ * T.midAt (lcp p₁ p₂))
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
      simpa [lcp, midAt] using sift_part a z₁ z₂ q₁ q₂ hq₁ hq₂ hq
    · rcases hq₁ : a.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : r.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      simp [lcp, midAt, hr₁, hr₂]
    · simp at h₂
    · rcases hq₁ : r.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : a.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      simp [lcp, midAt, hr₁, hr₂]
    · rcases hq₁ : r.sift read z₁ with q₁ | q₁ <;> rw [hq₁] at h₁ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₁
      rcases hq₂ : r.sift read z₂ with q₂ | q₂ <;> rw [hq₂] at h₂ <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h₂
      subst h₁ h₂
      have hq : q₁ ≠ q₂ := fun h => hne (by rw [h])
      simpa [lcp, midAt] using sift_part r z₁ z₂ q₁ q₂ hq₁ hq₂ hq
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

end DTree

/-- Under ideal reads a read is fixed by the target state. -/
theorem read_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hI : IdealReads read M side)
    {z z' : FreeMonoid α} (h : M.eval z.toList = M.eval z'.toList) : read z = read z' := by
  rcases hI (M.eval z.toList) with hq | hq
  · rw [hq z rfl, hq z' h.symm]
  · rw [hq z rfl, hq z' h.symm]

theorem read_mul_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hI : IdealReads read M side)
    {z z' : FreeMonoid α} (h : M.eval z.toList = M.eval z'.toList) (m : FreeMonoid α) :
    read (z * m) = read (z' * m) := by
  apply read_congr read hI
  simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
  exact congrArg (M.evalFrom · m.toList) h

end Ideal

end OrthoDFA
