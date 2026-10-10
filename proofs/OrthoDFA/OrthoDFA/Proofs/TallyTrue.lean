import OrthoDFA.Proofs.TallyWalk

/-!
# True leaves and genuine trees

A split at `p` leaves every other state's true leaf where it was, and sends the states at `p` to
the side their read of the new midfix is on. A genuine tree has at most `|Q| + 2` leaves: each
genuine split adds one leaf and one more true leaf. Two true records at one edge with different
targets make the split they trigger genuine.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

theorem leafOf_mem_paths : ∀ (T : DTree α) (q : σ), G.leafOf T q ∈ T.paths
  | .leaf, q => by simp [leafOf, DTree.paths]
  | .node m r a, q => by
    simp only [leafOf, DTree.paths, List.mem_append, List.mem_map]
    split
    · exact .inr ⟨_, leafOf_mem_paths a q, rfl⟩
    · exact .inl ⟨_, leafOf_mem_paths r q, rfl⟩

theorem leafOf_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), p ∈ T.paths → ∀ q : σ,
      (G.leafOf T q ≠ p → G.leafOf (T.splitAt d p) q = G.leafOf T q)
      ∧ (G.leafOf T q = p → G.leafOf (T.splitAt d p) q = p ++ [G.side (G.at' q d)])
  | .leaf, [], _, q => by
    refine ⟨fun h => absurd rfl h, fun _ => ?_⟩
    simp only [DTree.splitAt, leafOf, List.nil_append]
    split <;> simp_all
  | .leaf, _ :: _, h, _ => by simp [DTree.paths] at h
  | .node _ _ _, [], h, _ => absurd h DTree.paths_ne_nil_of_node
  | .node n r a, false :: p, h, q => by
    have hp : p ∈ r.paths := by simpa [DTree.paths] using h
    have ih := leafOf_splitAt d r p hp q
    simp only [DTree.splitAt, leafOf]
    by_cases hs : G.side (G.at' q n)
    · simp [hs]
    · simp only [hs, if_false, Bool.false_eq_true, List.cons.injEq, true_and, ne_eq,
        List.cons_append]
      exact ⟨fun h => by rw [ih.1 h], fun h => by rw [ih.2 h]⟩
  | .node n r a, true :: p, h, q => by
    have hp : p ∈ a.paths := by simpa [DTree.paths] using h
    have ih := leafOf_splitAt d a p hp q
    simp only [DTree.splitAt, leafOf]
    by_cases hs : G.side (G.at' q n)
    · simp only [hs, if_true, List.cons.injEq, true_and, ne_eq, List.cons_append]
      exact ⟨fun h => by rw [ih.1 h], fun h => by rw [ih.2 h]⟩
    · simp [hs]

open scoped Classical in
/-- The states' true leaves. -/
noncomputable def trueLeaves [Fintype σ] (T : DTree α) : Finset (List Bool) :=
  (Finset.univ : Finset σ).image (G.leafOf T)

theorem genuine_paths_aux [Fintype σ] {T : DTree α} (hT : G.Genuine T) :
    T.paths.length ≤ (G.trueLeaves T).card + 2 := by
  classical
  induction hT with
  | start => simp [DTree.paths]
  | @split T p d hT hs ih =>
    obtain ⟨q₁, q₂, hl₁, hl₂, hne⟩ := hs
    have hp : p ∈ T.paths := hl₁ ▸ G.leafOf_mem_paths T q₁
    rw [DTree.splitAt_paths_length d T p hp]
    have hpW : p ∈ G.trueLeaves T := Finset.mem_image.2 ⟨q₁, Finset.mem_univ _, hl₁⟩
    have hsub : insert (p ++ [G.side (G.at' q₁ d)]) (insert (p ++ [G.side (G.at' q₂ d)])
        ((G.trueLeaves T).erase p)) ⊆ G.trueLeaves (T.splitAt d p) := by
      intro r hr
      rcases Finset.mem_insert.1 hr with rfl | hr
      · exact Finset.mem_image.2 ⟨q₁, Finset.mem_univ _, (G.leafOf_splitAt d T p hp q₁).2 hl₁⟩
      rcases Finset.mem_insert.1 hr with rfl | hr
      · exact Finset.mem_image.2 ⟨q₂, Finset.mem_univ _, (G.leafOf_splitAt d T p hp q₂).2 hl₂⟩
      obtain ⟨hrp, hr⟩ := Finset.mem_erase.1 hr
      obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hr
      exact Finset.mem_image.2 ⟨q, Finset.mem_univ _, (G.leafOf_splitAt d T p hp q).1 hrp⟩
    have hW : ∀ b, p ++ [b] ∉ G.trueLeaves T := by
      intro b hb
      obtain ⟨q, -, hq⟩ := Finset.mem_image.1 hb
      exact DTree.append_not_mem_paths T p b hp (hq ▸ G.leafOf_mem_paths T q)
    have hcard := Finset.card_le_card hsub
    rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem, Finset.card_erase_of_mem hpW]
      at hcard
    · have := Finset.card_pos.2 ⟨p, hpW⟩
      omega
    · intro h
      exact hW _ (Finset.mem_of_mem_erase h)
    · intro h
      rcases Finset.mem_insert.1 h with h | h
      · exact hne (by simpa using h)
      · exact hW _ (Finset.mem_of_mem_erase h)

theorem genuine_paths [Fintype σ] {T : DTree α} (hT : G.Genuine T) :
    T.paths.length ≤ Fintype.card σ + 2 := by
  classical
  have h1 := G.genuine_paths_aux hT
  have h2 : (G.trueLeaves T).card ≤ Fintype.card σ := Finset.card_image_le.trans_eq rfl
  omega

/-- Two states at different true leaves read the midfix where their leaves part on different
sides. -/
theorem split_sides : ∀ (T : DTree α) (q₁ q₂ : σ), G.leafOf T q₁ ≠ G.leafOf T q₂ →
    G.side (G.at' q₁ (T.midAt (lcp (G.leafOf T q₁) (G.leafOf T q₂))))
      ≠ G.side (G.at' q₂ (T.midAt (lcp (G.leafOf T q₁) (G.leafOf T q₂))))
  | .leaf, _, _, h => absurd rfl h
  | .node m r a, q₁, q₂, hne => by
    simp only [leafOf] at hne ⊢
    by_cases s₁ : G.side (G.at' q₁ m) <;> by_cases s₂ : G.side (G.at' q₂ m)
    · simp only [s₁, s₂, if_true, ne_eq, List.cons.injEq, true_and] at hne ⊢
      simpa [lcp, DTree.midAt] using split_sides a q₁ q₂ hne
    · simp [lcp, DTree.midAt, s₁, s₂]
    · simp [lcp, DTree.midAt, s₁, s₂]
    · simp only [s₁, s₂, Bool.false_eq_true, if_false, ne_eq, List.cons.injEq, true_and]
        at hne ⊢
      simpa [lcp, DTree.midAt] using split_sides r q₁ q₂ hne

theorem eval_mul_of (sp : FreeMonoid α) (c : α) :
    G.M.eval (sp * FreeMonoid.of c).toList = G.M.step (G.M.eval sp.toList) c := by
  simp [DFA.eval_append_singleton]

theorem at'_of_mul (q : σ) (c : α) (m : FreeMonoid α) :
    G.at' q (FreeMonoid.of c * m) = G.at' (G.M.step q c) m := by
  simp [at', DFA.evalFrom_cons]

/-- True records at one edge with different targets make the split they trigger genuine. -/
theorem genuineSplit_of_recs {T : DTree α} {p : List Bool} {c : α} {t t₀ : List Bool}
    {sp₁ sp₂ : FreeMonoid α} (h₁ : G.TrueRec T (p, c, t) sp₁) (h₂ : G.TrueRec T (p, c, t₀) sp₂)
    (hne : t ≠ t₀) : G.GenuineSplit T p (FreeMonoid.of c * T.midAt (lcp t t₀)) := by
  obtain ⟨l₁, l₁'⟩ := h₁
  obtain ⟨l₂, l₂'⟩ := h₂
  simp only [G.eval_mul_of] at l₁' l₂'
  refine ⟨_, _, l₁, l₂, ?_⟩
  rw [G.at'_of_mul, G.at'_of_mul]
  have := G.split_sides T (G.M.step (G.M.eval sp₁.toList) c) (G.M.step (G.M.eval sp₂.toList) c)
    (by rw [l₁', l₂']; exact hne)
  rw [l₁', l₂'] at this
  exact this

end ReadModel

end OrthoDFA
