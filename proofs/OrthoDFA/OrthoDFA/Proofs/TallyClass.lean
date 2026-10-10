import OrthoDFA.Proofs.TallyTrue

/-!
# The class is finite

Every tree of the class is reached from the root's cut by splits each at a leaf, on a letter
followed by the midfix where two leaves part. `classSet n` collects the trees `n` such splits
reach; a tree of the class with `f ≤ S` splits that are not genuine is in `classSet (|Q| + S)`,
and `classSet n` has at most `(1 + (n + 1)³ |Σ|)ⁿ` trees.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

open scoped Classical in
/-- The trees at most `n` splits reach. -/
noncomputable def classSet : ℕ → Finset (DTree α)
  | 0 => {.node 1 .leaf .leaf}
  | n + 1 => classSet n ∪ (classSet n).biUnion fun T =>
      (T.paths.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.paths.toFinset ×ˢ T.paths.toFinset).image
        fun q => T.splitAt (FreeMonoid.of q.2.1 * T.midAt (lcp q.2.2.1 q.2.2.2)) q.1

theorem classSet_mono (n : ℕ) : (classSet n : Finset (DTree α)) ⊆ classSet (n + 1) := by
  classical
  intro T h
  simp only [classSet]
  exact Finset.mem_union_left _ h

theorem classSet_mono' {n n' : ℕ} (h : n ≤ n') :
    (classSet n : Finset (DTree α)) ⊆ classSet n' := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (classSet_mono _)

theorem classSet_split {n : ℕ} {T : DTree α} (hT : T ∈ classSet n) {p t t₀ : List Bool} {c : α}
    (hp : p ∈ T.paths) (ht : t ∈ T.paths) (ht₀ : t₀ ∈ T.paths) :
    T.splitAt (FreeMonoid.of c * T.midAt (lcp t t₀)) p ∈ classSet (n + 1) := by
  classical
  simp only [classSet]
  refine Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨T, hT, Finset.mem_image.2
    ⟨(p, c, t, t₀), ?_, rfl⟩⟩)
  simp [hp, ht, ht₀]

theorem classSet_paths : ∀ (n : ℕ) (T : DTree α), T ∈ classSet n → T.paths.length ≤ n + 2
  | 0, T, h => by
    simp only [classSet, Finset.mem_singleton] at h
    subst h
    simp [DTree.paths]
  | n + 1, T, h => by
    classical
    simp only [classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T', hT', ⟨p, c, t, t₀⟩, hq, rfl⟩
    · have := classSet_paths n T h; omega
    · simp only [Finset.mem_product, List.mem_toFinset] at hq
      rw [DTree.splitAt_paths_length _ _ _ hq.1]
      have := classSet_paths n T' hT'
      omega

theorem classSet_card : ∀ n : ℕ,
    (classSet n : Finset (DTree α)).card ≤ (1 + (n + 1) ^ 3 * Fintype.card α) ^ n
  | 0 => by simp [classSet]
  | n + 1 => by
    classical
    simp only [classSet]
    refine (Finset.card_union_le _ _).trans ?_
    have hstep : ∀ T ∈ (classSet n : Finset (DTree α)),
        ((T.paths.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.paths.toFinset ×ˢ
          T.paths.toFinset).image fun q => T.splitAt (FreeMonoid.of q.2.1 *
            T.midAt (lcp q.2.2.1 q.2.2.2)) q.1).card ≤ (n + 2) ^ 3 * Fintype.card α := by
      intro T hT
      refine Finset.card_image_le.trans ?_
      have h := (List.toFinset_card_le T.paths).trans (classSet_paths n T hT)
      simp only [Finset.card_product, Finset.card_univ]
      calc _ ≤ (n + 2) * (Fintype.card α * ((n + 2) * (n + 2))) := by gcongr
        _ = _ := by ring
    have hb := Finset.card_biUnion_le_card_mul (classSet n) _ _ hstep
    have ih := classSet_card n
    have h1 : (n + 2) ^ 3 * Fintype.card α ≤ (n + 1 + 1) ^ 3 * Fintype.card α := le_rfl
    have h2 : (1 + (n + 1) ^ 3 * Fintype.card α) ^ n
        ≤ (1 + (n + 1 + 1) ^ 3 * Fintype.card α) ^ n := by
      gcongr <;> omega
    calc _ ≤ (classSet n).card + (classSet n).card * ((n + 2) ^ 3 * Fintype.card α) :=
          Nat.add_le_add_left hb _
      _ = (classSet n).card * (1 + (n + 2) ^ 3 * Fintype.card α) := by ring
      _ ≤ (1 + (n + 1 + 1) ^ 3 * Fintype.card α) ^ n * (1 + (n + 2) ^ 3 * Fintype.card α) :=
          Nat.mul_le_mul_right _ (ih.trans h2)
      _ = _ := by ring

theorem grown_two {σ : Type*} (G : ReadModel α σ) {T : DTree α} {f : ℕ} (h : G.Grown T f) :
    2 ≤ T.paths.length := by
  induction h with
  | start => simp [DTree.paths]
  | @real T f p t t₀ c hg ht ht₀ hgen ih =>
    have hp : p ∈ T.paths := by
      obtain ⟨q₁, -, hl, -⟩ := hgen
      exact hl ▸ G.leafOf_mem_paths _ _
    rw [DTree.splitAt_paths_length _ _ _ hp]; omega
  | @fake T f p t t₀ c hg hp ht ht₀ hng ih =>
    rw [DTree.splitAt_paths_length _ _ _ hp]; omega

theorem grown_mem {σ : Type*} (G : ReadModel α σ) {T : DTree α} {f : ℕ}
    (h : G.Grown T f) : T ∈ classSet (T.paths.length - 2) := by
  induction h with
  | start => simp [classSet, DTree.paths]
  | @real T f p t t₀ c hg ht ht₀ hgen ih =>
    have hp : p ∈ T.paths := by
      obtain ⟨q₁, -, hl, -⟩ := hgen
      exact hl ▸ G.leafOf_mem_paths _ _
    rw [DTree.splitAt_paths_length _ _ _ hp]
    have h2 := grown_two G hg
    rw [show T.paths.length + 1 - 2 = T.paths.length - 2 + 1 by omega]
    exact classSet_split ih hp ht ht₀
  | @fake T f p t t₀ c hg hp ht ht₀ hng ih =>
    rw [DTree.splitAt_paths_length _ _ _ hp]
    have h2 := grown_two G hg
    rw [show T.paths.length + 1 - 2 = T.paths.length - 2 + 1 by omega]
    exact classSet_split ih hp ht ht₀

/-- A tree of the class is in `classSet (|Q| + S)`. -/
theorem inClass_mem {σ : Type*} [Fintype σ] (G : ReadModel α σ) {S : ℕ} {T : DTree α}
    (h : G.InClass S T) : T ∈ classSet (Fintype.card σ + S) := by
  obtain ⟨f, hf, hg⟩ := h
  have := G.grown_paths hg
  exact classSet_mono' (by omega) (grown_mem G hg)

end OrthoDFA
