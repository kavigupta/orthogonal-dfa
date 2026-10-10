import OrthoDFA.Proofs.TallyWalk

/-!
# True leaves and genuine trees

A split at `p` leaves every other state's true leaf where it was, and sends the states at `p` to
the side their read of the new midfix is on. So no split lowers the count of true leaves, at most
`|Q|`, and a genuine one raises it. Two true records at one edge with different targets make the
split they trigger genuine. A split is fixed by its tree.
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

theorem trueLeaves_card_le [Fintype σ] (T : DTree α) :
    (G.trueLeaves T).card ≤ Fintype.card σ :=
  Finset.card_image_le.trans_eq rfl

theorem trueLeaves_splitAt [Fintype σ] {T : DTree α} {p : List Bool} (hp : p ∈ T.paths)
    (d : FreeMonoid α) :
    (G.trueLeaves T).card ≤ (G.trueLeaves (T.splitAt d p)).card
      ∧ (G.GenuineSplit T p d →
        (G.trueLeaves T).card + 1 ≤ (G.trueLeaves (T.splitAt d p)).card) := by
  classical
  have hW : ∀ b, p ++ [b] ∉ G.trueLeaves T := by
    intro b hb
    obtain ⟨q, -, hq⟩ := Finset.mem_image.1 hb
    exact DTree.append_not_mem_paths T p b hp (hq ▸ G.leafOf_mem_paths T q)
  have hrest : (G.trueLeaves T).erase p ⊆ G.trueLeaves (T.splitAt d p) := by
    intro r hr
    obtain ⟨hrp, hr⟩ := Finset.mem_erase.1 hr
    obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hr
    exact Finset.mem_image.2 ⟨q, Finset.mem_univ _, (G.leafOf_splitAt d T p hp q).1 hrp⟩
  have hnew : ∀ q, G.leafOf T q = p →
      p ++ [G.side (G.at' q d)] ∈ G.trueLeaves (T.splitAt d p) := fun q hq =>
    Finset.mem_image.2 ⟨q, Finset.mem_univ _, (G.leafOf_splitAt d T p hp q).2 hq⟩
  constructor
  · by_cases hpW : p ∈ G.trueLeaves T
    · obtain ⟨q, -, hq⟩ := Finset.mem_image.1 hpW
      have hsub : insert (p ++ [G.side (G.at' q d)]) ((G.trueLeaves T).erase p)
          ⊆ G.trueLeaves (T.splitAt d p) := Finset.insert_subset (hnew q hq) hrest
      have := Finset.card_le_card hsub
      rw [Finset.card_insert_of_notMem (fun h => hW _ (Finset.mem_of_mem_erase h)),
        Finset.card_erase_of_mem hpW] at this
      have := Finset.card_pos.2 ⟨p, hpW⟩
      omega
    · rw [← Finset.erase_eq_of_notMem hpW]
      exact Finset.card_le_card hrest
  · rintro ⟨q₁, q₂, hl₁, hl₂, hne⟩
    have hpW : p ∈ G.trueLeaves T := Finset.mem_image.2 ⟨q₁, Finset.mem_univ _, hl₁⟩
    have hsub : insert (p ++ [G.side (G.at' q₁ d)]) (insert (p ++ [G.side (G.at' q₂ d)])
        ((G.trueLeaves T).erase p)) ⊆ G.trueLeaves (T.splitAt d p) :=
      Finset.insert_subset (hnew q₁ hl₁) (Finset.insert_subset (hnew q₂ hl₂) hrest)
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

open scoped Classical in
theorem trueLeaves_splitAt_le [Fintype σ] {T : DTree α} {p : List Bool} (hp : p ∈ T.paths)
    (d : FreeMonoid α) :
    (G.trueLeaves (T.splitAt d p)).card
      ≤ (G.trueLeaves T).card + if G.GenuineSplit T p d then 1 else 0 := by
  classical
  set A := (Finset.univ.filter fun q => G.leafOf T q = p).image fun q => p ++ [G.side (G.at' q d)]
  have hsub : G.trueLeaves (T.splitAt d p) ⊆ (G.trueLeaves T).erase p ∪ A := by
    intro r hr
    obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hr
    by_cases hq : G.leafOf T q = p
    · rw [(G.leafOf_splitAt d T p hp q).2 hq]
      exact Finset.mem_union_right _ (Finset.mem_image.2 ⟨q, by simp [hq], rfl⟩)
    · rw [(G.leafOf_splitAt d T p hp q).1 hq]
      exact Finset.mem_union_left _ (Finset.mem_erase.2 ⟨hq, Finset.mem_image.2
        ⟨q, Finset.mem_univ _, rfl⟩⟩)
  refine (Finset.card_le_card hsub).trans ((Finset.card_union_le _ _).trans ?_)
  by_cases hpW : p ∈ G.trueLeaves T
  · rw [Finset.card_erase_of_mem hpW]
    have hpos := Finset.card_pos.2 ⟨p, hpW⟩
    have hA : A.card ≤ 1 + if G.GenuineSplit T p d then 1 else 0 := by
      split_ifs with hg
      · refine (Finset.card_le_card (t := {p ++ [true], p ++ [false]}) ?_).trans
          (Finset.card_le_two)
        intro r hr
        obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hr
        cases G.side (G.at' q d) <;> simp
      · obtain ⟨q₀, -, hq₀⟩ := Finset.mem_image.1 hpW
        refine Finset.card_le_one.2 fun r₁ h₁ r₂ h₂ => ?_
        obtain ⟨q₁, hq₁, rfl⟩ := Finset.mem_image.1 h₁
        obtain ⟨q₂, hq₂, rfl⟩ := Finset.mem_image.1 h₂
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hq₁ hq₂
        by_contra hne
        exact hg ⟨q₁, q₂, hq₁, hq₂, fun h => hne (by rw [h])⟩
    omega
  · have hA : A = ∅ := by
      refine Finset.image_eq_empty.2 (Finset.filter_eq_empty_iff.2 fun q _ hq => hpW ?_)
      exact Finset.mem_image.2 ⟨q, Finset.mem_univ _, hq⟩
    rw [hA, Finset.erase_eq_of_notMem hpW]
    simp

/-- The class's trees: each split adds a leaf, a genuine one a true leaf too, so the count of
splits that are not genuine is fixed by the tree. -/
theorem grown_count [Fintype σ] {T : DTree α} {f : ℕ} (h : G.Grown T f) :
    T.paths.length + (G.trueLeaves (.node 1 .leaf .leaf)).card = (G.trueLeaves T).card + 2 + f := by
  classical
  induction h with
  | start => simp [DTree.paths]; omega
  | @real T f p t t₀ c hT ht ht₀ hg ih =>
    have hp : p ∈ T.paths := by
      obtain ⟨q₁, -, hl, -⟩ := hg
      exact hl ▸ G.leafOf_mem_paths T q₁
    have h1 := (G.trueLeaves_splitAt hp (FreeMonoid.of c * T.midAt (lcp t t₀))).2 hg
    have h2 := G.trueLeaves_splitAt_le hp (FreeMonoid.of c * T.midAt (lcp t t₀))
    rw [if_pos hg] at h2
    rw [DTree.splitAt_paths_length _ _ _ hp]
    omega
  | @fake T f p t t₀ c hT hp ht ht₀ hg ih =>
    have h1 := (G.trueLeaves_splitAt hp (FreeMonoid.of c * T.midAt (lcp t t₀))).1
    have h2 := G.trueLeaves_splitAt_le hp (FreeMonoid.of c * T.midAt (lcp t t₀))
    rw [if_neg hg] at h2
    rw [DTree.splitAt_paths_length _ _ _ hp]
    omega

theorem grown_unique [Fintype σ] {T : DTree α} {f f' : ℕ} (h : G.Grown T f)
    (h' : G.Grown T f') : f = f' := by
  have := G.grown_count h
  have := G.grown_count h'
  omega

theorem grown_paths [Fintype σ] {T : DTree α} {f : ℕ} (h : G.Grown T f) :
    T.paths.length ≤ Fintype.card σ + 2 + f := by
  have := G.grown_count h
  have := G.trueLeaves_card_le T
  omega

end ReadModel

namespace DTree

omit [Fintype α] [DecidableEq α] in
theorem splitAt_ne {d : FreeMonoid α} {T : DTree α} {p : List Bool} (hp : p ∈ T.paths) :
    T.splitAt d p ≠ T := fun h => by
  have := splitAt_paths_length d T p hp
  rw [h] at this
  omega

omit [Fintype α] [DecidableEq α] in
theorem splitAt_inj {d d' : FreeMonoid α} :
    ∀ {T : DTree α} {p p' : List Bool}, p ∈ T.paths → p' ∈ T.paths →
      T.splitAt d p = T.splitAt d' p' → p = p' ∧ d = d'
  | .leaf, p, p', hp, hp', h => by
    simp only [paths, List.mem_singleton] at hp hp'
    subst hp hp'
    simp only [splitAt, node.injEq] at h
    exact ⟨rfl, h.1⟩
  | .node n r a, [], _, hp, _, _ => absurd hp paths_ne_nil_of_node
  | .node n r a, _, [], _, hp', _ => absurd hp' paths_ne_nil_of_node
  | .node n r a, b :: p, b' :: p', hp, hp', h => by
    simp only [paths, List.mem_append, List.mem_map, List.cons.injEq] at hp hp'
    rcases hp with ⟨q, hq, rfl, rfl⟩ | ⟨q, hq, rfl, rfl⟩ <;>
      rcases hp' with ⟨q', hq', rfl, rfl⟩ | ⟨q', hq', rfl, rfl⟩ <;>
      simp only [splitAt, node.injEq, true_and, and_true] at h
    · obtain ⟨rfl, rfl⟩ := splitAt_inj hq hq' h; exact ⟨rfl, rfl⟩
    · exact absurd h.1 (splitAt_ne hq)
    · exact absurd h.2 (splitAt_ne hq)
    · obtain ⟨rfl, rfl⟩ := splitAt_inj hq hq' h; exact ⟨rfl, rfl⟩

end DTree

namespace ReadModel

variable {σ : Type*} (G : ReadModel α σ)

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
