import OrthoDFA.Proofs.TallyWalk

/-!
# True leaves and genuine trees

A split at `p` leaves every other state's true leaf and path where they were, and sends the
states at `p` to the side their read of the new midfix is on. A genuine tree has at most `|Q| + 2`
leaves: each genuine split adds one leaf and one more leaf a path-good state reaches. Two true
records at one edge with different targets make the split they trigger genuine.
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
      (G.leafOf T q ≠ p → G.leafOf (T.splitAt d p) q = G.leafOf T q
        ∧ (G.PathGood (T.splitAt d p) q ↔ G.PathGood T q))
      ∧ (G.leafOf T q = p → G.leafOf (T.splitAt d p) q = p ++ [G.side (G.at' q d)]
        ∧ (G.PathGood (T.splitAt d p) q ↔ G.PathGood T q ∧ G.Good (G.at' q d)))
  | .leaf, [], _, q => by
    refine ⟨fun h => absurd rfl h, fun _ => ?_⟩
    simp only [DTree.splitAt, leafOf, PathGood, List.nil_append]
    split <;> simp_all [leafOf, PathGood]
  | .leaf, _ :: _, h, _ => by simp [DTree.paths] at h
  | .node _ _ _, [], h, _ => absurd h DTree.paths_ne_nil_of_node
  | .node n r a, false :: p, h, q => by
    have hp : p ∈ r.paths := by simpa [DTree.paths] using h
    have ih := leafOf_splitAt d r p hp q
    simp only [DTree.splitAt, leafOf, PathGood]
    by_cases hs : G.side (G.at' q n)
    · simp [hs]
    · simp only [hs, if_false, Bool.false_eq_true, List.cons.injEq, true_and, ne_eq,
        List.cons_append]
      exact ⟨fun h => ⟨by rw [(ih.1 h).1], by rw [(ih.1 h).2]⟩,
        fun h => ⟨by rw [(ih.2 h).1], by rw [(ih.2 h).2, and_assoc]⟩⟩
  | .node n r a, true :: p, h, q => by
    have hp : p ∈ a.paths := by simpa [DTree.paths] using h
    have ih := leafOf_splitAt d a p hp q
    simp only [DTree.splitAt, leafOf, PathGood]
    by_cases hs : G.side (G.at' q n)
    · simp only [hs, if_true, List.cons.injEq, true_and, ne_eq, List.cons_append]
      exact ⟨fun h => ⟨by rw [(ih.1 h).1], by rw [(ih.1 h).2]⟩,
        fun h => ⟨by rw [(ih.2 h).1], by rw [(ih.2 h).2, and_assoc]⟩⟩
    · simp [hs]

open scoped Classical in
/-- The leaves path-good states reach. -/
noncomputable def goodLeaves [Fintype σ] (T : DTree α) : Finset (List Bool) :=
  (Finset.univ.filter fun q : σ => G.PathGood T q).image (G.leafOf T)

theorem genuine_paths_aux [Fintype σ] {T : DTree α} (hT : G.Genuine T) :
    T.paths.length ≤ (G.goodLeaves T).card + 2 := by
  classical
  induction hT with
  | start => simp [DTree.paths]
  | @split T p d hT hs ih =>
    obtain ⟨q₁, q₂, hg₁, hg₂, hl₁, hl₂, hd₁, hd₂, hne⟩ := hs
    have hp : p ∈ T.paths := hl₁ ▸ G.leafOf_mem_paths T q₁
    rw [DTree.splitAt_paths_length d T p hp]
    have hpW : p ∈ G.goodLeaves T := by
      simp only [goodLeaves, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
      exact ⟨q₁, hg₁, hl₁⟩
    have hsub : insert (p ++ [G.side (G.at' q₁ d)]) (insert (p ++ [G.side (G.at' q₂ d)])
        ((G.goodLeaves T).erase p)) ⊆ G.goodLeaves (T.splitAt d p) := by
      intro r hr
      simp only [goodLeaves, Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
      rcases Finset.mem_insert.1 hr with rfl | hr
      · have := (G.leafOf_splitAt d T p hp q₁).2 hl₁
        exact ⟨q₁, this.2.2 ⟨hg₁, hd₁⟩, this.1⟩
      rcases Finset.mem_insert.1 hr with rfl | hr
      · have := (G.leafOf_splitAt d T p hp q₂).2 hl₂
        exact ⟨q₂, this.2.2 ⟨hg₂, hd₂⟩, this.1⟩
      obtain ⟨hrp, hr⟩ := Finset.mem_erase.1 hr
      simp only [goodLeaves, Finset.mem_image, Finset.mem_filter, Finset.mem_univ,
        true_and] at hr
      obtain ⟨q, hq, rfl⟩ := hr
      have := (G.leafOf_splitAt d T p hp q).1 hrp
      exact ⟨q, this.2.2 hq, this.1⟩
    have hW : ∀ b, p ++ [b] ∉ G.goodLeaves T := by
      intro b hb
      simp only [goodLeaves, Finset.mem_image, Finset.mem_filter, Finset.mem_univ,
        true_and] at hb
      obtain ⟨q, -, hq⟩ := hb
      exact DTree.append_not_mem_paths T p b hp (hq ▸ G.leafOf_mem_paths T q)
    have hcard := Finset.card_le_card hsub
    have hb : G.side (G.at' q₁ d) ≠ G.side (G.at' q₂ d) := hne
    rw [Finset.card_insert_of_notMem, Finset.card_insert_of_notMem, Finset.card_erase_of_mem hpW]
      at hcard
    · have := Finset.card_pos.2 ⟨p, hpW⟩
      omega
    · intro h
      exact hW _ (Finset.mem_of_mem_erase h)
    · intro h
      rcases Finset.mem_insert.1 h with h | h
      · exact hb (by simpa using h)
      · exact hW _ (Finset.mem_of_mem_erase h)

theorem genuine_paths [Fintype σ] {T : DTree α} (hT : G.Genuine T) :
    T.paths.length ≤ Fintype.card σ + 2 := by
  classical
  have h1 := G.genuine_paths_aux hT
  have h2 : (G.goodLeaves T).card ≤ Fintype.card σ :=
    Finset.card_image_le.trans (Finset.card_filter_le _ _)
  omega

/-- Two path-good states at different leaves read the midfix where their leaves part on
different sides, both at good read-states. -/
theorem split_sides : ∀ (T : DTree α) (q₁ q₂ : σ), G.PathGood T q₁ → G.PathGood T q₂ →
    G.leafOf T q₁ ≠ G.leafOf T q₂ →
    G.Good (G.at' q₁ (T.midAt (lcp (G.leafOf T q₁) (G.leafOf T q₂))))
      ∧ G.Good (G.at' q₂ (T.midAt (lcp (G.leafOf T q₁) (G.leafOf T q₂))))
      ∧ G.side (G.at' q₁ (T.midAt (lcp (G.leafOf T q₁) (G.leafOf T q₂))))
        ≠ G.side (G.at' q₂ (T.midAt (lcp (G.leafOf T q₁) (G.leafOf T q₂))))
  | .leaf, _, _, _, _, h => absurd rfl h
  | .node m r a, q₁, q₂, h₁, h₂, hne => by
    simp only [PathGood] at h₁ h₂
    simp only [leafOf] at hne ⊢
    by_cases s₁ : G.side (G.at' q₁ m) <;> by_cases s₂ : G.side (G.at' q₂ m)
    · simp only [s₁, s₂, if_true, ne_eq, List.cons.injEq, true_and] at h₁ h₂ hne ⊢
      simpa [lcp, DTree.midAt] using split_sides a q₁ q₂ h₁.2 h₂.2 hne
    · simp only [s₁, s₂, if_true, Bool.false_eq_true, if_false] at h₁ h₂ ⊢
      simp [lcp, DTree.midAt, s₁, s₂, h₁.1, h₂.1]
    · simp only [s₁, s₂, if_true, Bool.false_eq_true, if_false] at h₁ h₂ ⊢
      simp [lcp, DTree.midAt, s₁, s₂, h₁.1, h₂.1]
    · simp only [s₁, s₂, Bool.false_eq_true, if_false, ne_eq, List.cons.injEq, true_and]
        at h₁ h₂ hne ⊢
      simpa [lcp, DTree.midAt] using split_sides r q₁ q₂ h₁.2 h₂.2 hne

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
  obtain ⟨g₁, l₁, g₁', l₁'⟩ := h₁
  obtain ⟨g₂, l₂, g₂', l₂'⟩ := h₂
  simp only [G.eval_mul_of] at g₁' l₁' g₂' l₂'
  refine ⟨_, _, g₁, g₂, l₁, l₂, ?_⟩
  rw [G.at'_of_mul, G.at'_of_mul]
  have := G.split_sides T _ _ g₁' g₂' (by rw [l₁', l₂']; exact hne)
  rw [l₁', l₂'] at this
  exact this

end ReadModel

end OrthoDFA
