import RLStar.Proofs.TrieSound
import RLStar.Proofs.QueryLevel3
import RLStar.Proofs.RunReads

/-!
# The trie round's tree, and the strings it reads

A trie round changes its tree and edges as level 3 does, so under reads that are never wrong its
final tree is in `classSet |Q|` with at most `|Q| + 2` leaves (`trie_final_tree`). Its probes each
read at most `(3 + ⌈log₂ L⌉)(|Q| + 1)` strings and its splits, at most `|Q|` of them, each re-sift
at most `(|Q| + 2)|Σ|(|Q| + 2)` (`card_reads_level3TrieQ_le`).
-/

noncomputable section

namespace OrthoDFA

namespace Random

open Query Eps Kappa Spur

variable {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (y : ℝ) (cap : ℕ)
  (read : FreeMonoid α → ARU)

/-- The trie state's tree and edges as a level-3 state. -/
def trieL3 (s : TrieState α) : Level3State α := newHypothesis s.tree s.edges 0

omit [DecidableEq α] in
theorem rinv_congr {s₁ s₂ : Level3State α} (ht : s₂.tree = s₁.tree) (he : s₂.edges = s₁.edges)
    (h : RInv read s₁) : RInv read s₂ := by
  obtain ⟨⟨h1, h2, h3⟩, h4⟩ := h
  exact ⟨⟨ht ▸ he ▸ h1, ht ▸ h2, ht ▸ h3⟩, ht ▸ h4⟩

theorem splitTrie_tree_edges (s : TrieState α) (p : List Bool) (c : α) (t t₀ : List Bool) :
    ((splitTrieLeafQ s p c t t₀).eval read).tree = (splitLeaf read (trieL3 s) p c t t₀).tree
      ∧ ((splitTrieLeafQ s p c t t₀).eval read).edges
        = (splitLeaf read (trieL3 s) p c t t₀).edges := by
  simp [splitTrieLeafQ, splitLeaf, splitLeafQ, eval_map, trieL3, newHypothesis,
    newTrieHypothesis]

variable {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

open scoped Classical in
/-- A trie step keeps `RInv`; one that adds no leaf reads only its probe's strings, and one that
adds one adds exactly one. -/
theorem trie_step_inv (hW : NoWrong M side read) {s s' : TrieState α} {x : FreeMonoid α}
    (hs : RInv read (trieL3 s)) (h : (applyTrieProbeQ C y cap s x).eval read = .inl s') :
    RInv read (trieL3 s')
      ∧ ((s'.tree.leaves.length = s.tree.leaves.length
          ∧ (applyTrieProbeQ C y cap s x).reads read
            ⊆ probeReads read s.tree s.edges (walkStartIn s.expanded x) x)
        ∨ (s'.tree.leaves.length = s.tree.leaves.length + 1
          ∧ (applyTrieProbeQ C y cap s x).reads read
            ⊆ probeReads read s.tree s.edges (walkStartIn s.expanded x) x
              ∪ splitReads read { C with walkStart := walkStartIn s.expanded x } (trieL3 s) x))
    := by
  set k := walkStartIn s.expanded x
  have hOK := probeOrAgree_ok read s.tree s.edges k hs.1.edges x
  have hR : (probeOrAgreeQ s.tree s.edges k x).reads read ⊆ probeReads read s.tree s.edges k x :=
    reads_probeOrAgreeQ_sub read s.tree x s.edges k
  unfold applyTrieProbeQ at h ⊢
  rw [eval_bind] at h
  rw [reads_bind]
  have hev : (probeOrAgreeQ s.tree s.edges k x).eval read = probeOrAgree read s.tree s.edges k x :=
    rfl
  rw [hev] at h ⊢
  generalize ho : probeOrAgree read s.tree s.edges k x = o at h hOK ⊢
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr, eval_pure, Sum.inl.injEq] at h
    subst h
    simp only [if_pos hr, reads_pure, Finset.union_empty]
    exact ⟨rinv_congr read rfl rfl hs, Or.inl ⟨rfl, hR⟩⟩
  rw [if_neg hr] at h
  simp only [if_neg hr]
  have hc : ∀ o', applyTrieTests C y (countTrieOutcome s o' (prefixOf x k)) = .inl s' →
      RInv read (trieL3 s') ∧ ((s'.tree.leaves.length = s.tree.leaves.length
          ∧ (probeOrAgreeQ s.tree s.edges k x).reads read ∪ ∅
            ⊆ probeReads read s.tree s.edges k x) ∨ False) := by
    intro o' h'
    rw [applyTrieTests_inl C y h']
    exact ⟨rinv_congr read rfl rfl hs, Or.inl ⟨rfl, by simpa using hR⟩⟩
  cases o with
  | unlearnedEdge p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    simp only [eval_pure, Sum.inl.injEq] at h
    subst h
    simp only [reads_pure, Finset.union_empty]
    exact ⟨⟨⟨edgesOK_upd read hs.1.edges hu huc, hs.1.cls, hs.1.two⟩, hs.2⟩, Or.inl ⟨rfl, hR⟩⟩
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, hne⟩ := hOK
    simp only [hE] at h ⊢
    rw [eval_map, Sum.inl.injEq] at h
    subst h
    obtain ⟨hw₀, hw₀c⟩ := hs.1.edges p hp c t₀ w₀ hE
    obtain ⟨hinv, -⟩ := split_ok (σ := σ) read hW hs.1 hs.2 hp hu huc hw₀ hw₀c hne
    obtain ⟨ht, he⟩ := splitTrie_tree_edges read s p c t t₀
    refine ⟨rinv_congr read ht he hinv, Or.inr ⟨?_, ?_⟩⟩
    · rw [ht, splitLeaf_eq]
      exact DTree.length_leaves_splitAt _ _ _ hp
    · rw [reads_map]
      refine Finset.union_subset_union hR ?_
      have hs' := reads_splitLeafQ_sub read (trieL3 s) p c t t₀
      simp only [splitLeafQ, splitTrieLeafQ, reads_map] at hs' ⊢
      refine hs'.trans (le_of_eq ?_)
      unfold splitReads
      simp only [trieL3, newHypothesis]
      rw [ho]
      simp only [hE]
  | agree =>
    rw [eval_pure] at h
    obtain ⟨h1, h2⟩ := hc _ h
    exact ⟨h1, Or.inl (by simpa using h2)⟩
  | undecidedStart z =>
    rw [eval_pure] at h
    obtain ⟨h1, h2⟩ := hc _ h
    exact ⟨h1, Or.inl (by simpa using h2)⟩
  | undecided zs cl =>
    rw [eval_pure] at h
    obtain ⟨h1, h2⟩ := hc _ h
    exact ⟨h1, Or.inl (by simpa using h2)⟩

open scoped Classical in
/-- A step that ends the round reads only its probe's strings. -/
theorem trie_step_inr_reads {s : TrieState α} {x : FreeMonoid α} {e : RoundEnd TrieSite}
    (h : (applyTrieProbeQ C y cap s x).eval read = .inr e) :
    (applyTrieProbeQ C y cap s x).reads read
      ⊆ probeReads read s.tree s.edges (walkStartIn s.expanded x) x := by
  have hR := reads_probeOrAgreeQ_sub read s.tree x s.edges (walkStartIn s.expanded x)
  unfold applyTrieProbeQ at h ⊢
  rw [eval_bind] at h
  rw [reads_bind]
  have hev : (probeOrAgreeQ s.tree s.edges (walkStartIn s.expanded x) x).eval read
      = probeOrAgree read s.tree s.edges (walkStartIn s.expanded x) x := rfl
  rw [hev] at h ⊢
  refine Finset.union_subset hR ?_
  generalize probeOrAgree read s.tree s.edges (walkStartIn s.expanded x) x = o at h ⊢
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr]; simp
  rw [if_neg hr] at h
  simp only [if_neg hr]
  cases o with
  | wrongEdge p c u t =>
    simp only at h ⊢
    split at h
    · rw [eval_map] at h; cases h
    · simp
  | _ => simp

theorem trie_run_inv (hW : NoWrong M side read) : ∀ (s : TrieState α) (xs : List (FreeMonoid α)),
    RInv read (trieL3 s) →
      RInv read (trieL3 ((runTrieProbesQ C y cap s xs).eval read).1)
        ∧ s.tree.leaves.length ≤ ((runTrieProbesQ C y cap s xs).eval read).1.tree.leaves.length
  | s, [], hs => by simp [runTrieProbesQ, hs]
  | s, x :: xs, hs => by
    unfold runTrieProbesQ
    rw [eval_bind]
    rcases ho : (applyTrieProbeQ C y cap s x).eval read with s' | e
    · obtain ⟨hs', hlen⟩ := trie_step_inv C y cap read hW hs ho
      obtain ⟨h1, h2⟩ := trie_run_inv hW s' xs hs'
      refine ⟨h1, le_trans ?_ h2⟩
      rcases hlen with ⟨h, -⟩ | ⟨h, -⟩ <;> omega
    · simp [hs]

omit [Fintype α] [DecidableEq α] in
theorem trieL3_initial : trieL3 (trieInitialState : TrieState α) = initialState := rfl

/-- Under reads that are never wrong, the round's final tree is in `classSet |Q|`, with at most
`|Q| + 2` leaves and its edges witnessed. -/
theorem trie_final_tree (hW : NoWrong M side read) {P : ℕ} (draws : Fin P → FreeMonoid α) :
    let H := (indecisiveLStarLevel3Trie C y cap read draws).hypothesis
    H.tree ∈ classSet (Fintype.card σ) ∧ H.tree.leaves.length ≤ Fintype.card σ + 2
      ∧ EdgesOK read H.tree H.edges := by
  have h := (trie_run_inv C y cap read hW trieInitialState (List.ofFn draws)
    (by rw [trieL3_initial]; exact ⟨inv_start read, reached_start read⟩)).1
  simp only [indecisiveLStarLevel3Trie, level3TrieQ, eval_map]
  exact ⟨cls_mem read hW h, leaves_le_of_reached' read hW h.2, h.1.edges⟩

/-- A run from `s` over probes at most `L` long, its final tree at most `n + 2` leaves, reads at
most `probeSifts L (n + 1)` strings in each probe and `(n + 2)|Σ|(n + 2)` in each split. -/
theorem card_reads_runTrieProbesQ_le (hW : NoWrong M side read) {L : ℕ} (n : ℕ) :
    ∀ (xs : List (FreeMonoid α)) (s : TrieState α), (∀ z ∈ xs, z.toList.length ≤ L) →
      RInv read (trieL3 s) →
      ((runTrieProbesQ C y cap s xs).eval read).1.tree.leaves.length ≤ n + 2 →
      ((runTrieProbesQ C y cap s xs).reads read).card
        ≤ xs.length * (probeSifts L * (n + 1))
          + (n + 2 - s.tree.leaves.length) * ((n + 2) * Fintype.card α * (n + 2))
  | [], _, _, _, _ => by simp [runTrieProbesQ]
  | x :: xs, s, hL, hs, hfin => by
    have hle := (trie_run_inv C y cap read hW s (x :: xs) hs).2
    have hc : s.tree.leaves.length ≤ n + 2 := hle.trans hfin
    have hxL : x.toList.length ≤ L := hL x (by simp)
    have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; omega
    have hprobe : ∀ k, (probeReads read s.tree s.edges k x).card ≤ probeSifts L * (n + 1) :=
      fun k => (probeReads_card_mids read hxL).trans (Nat.mul_le_mul_left _ hmid)
    have hsplit : ∀ C' : Level3Settings,
        (splitReads read C' (trieL3 s) x).card ≤ (n + 2) * Fintype.card α * (n + 2) :=
      fun C' => (splitReads_card_le C' read (trieL3 s) x).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_right _ hc) hc)
    have hL' : ∀ z ∈ xs, z.toList.length ≤ L := fun z hz => hL z (by simp [hz])
    unfold runTrieProbesQ at hfin ⊢
    rw [eval_bind] at hfin
    rw [reads_bind]
    refine (Finset.card_union_le _ _).trans ?_
    rcases ho : (applyTrieProbeQ C y cap s x).eval read with s' | e
    · rw [ho] at hfin
      simp only at hfin ⊢
      obtain ⟨hs', hcase⟩ := trie_step_inv C y cap read hW hs ho
      have hrec := card_reads_runTrieProbesQ_le hW n xs s' hL' hs' hfin
      generalize probeSifts L * (n + 1) = Pp at hprobe hrec ⊢
      generalize (n + 2) * Fintype.card α * (n + 2) = Rr at hsplit hrec ⊢
      rcases hcase with ⟨hlen, hsub⟩ | ⟨hlen, hsub⟩
      · have h1 := (Finset.card_le_card hsub).trans (hprobe _)
        rw [hlen] at hrec
        simp only [List.length_cons]
        nlinarith
      · have h1 := (Finset.card_le_card hsub).trans ((Finset.card_union_le _ _).trans
          (Nat.add_le_add (hprobe _) (hsplit _)))
        have hs'le := (trie_run_inv C y cap read hW s' xs hs').2.trans hfin
        obtain ⟨m, hm⟩ : ∃ m, n + 2 - s.tree.leaves.length = m + 1 :=
          ⟨n + 1 - s.tree.leaves.length, by omega⟩
        rw [hlen, show n + 2 - (s.tree.leaves.length + 1) = m by omega] at hrec
        rw [hm]
        simp only [List.length_cons]
        nlinarith
    · simp only [reads_pure, Finset.card_empty, add_zero]
      have h1 := (Finset.card_le_card (trie_step_inr_reads C y cap read ho)).trans (hprobe _)
      calc _ ≤ probeSifts L * (n + 1) := h1
        _ ≤ (xs.length + 1) * (probeSifts L * (n + 1)) := Nat.le_mul_of_pos_left _ (by omega)
        _ ≤ _ := by simp only [List.length_cons]; omega

/-- The round over `P` probes at most `L` long reads, under reads that are never wrong, at most
`P probeSifts L (|Q| + 1) + |Q| (|Q| + 2)|Σ|(|Q| + 2)` strings. -/
theorem card_reads_level3TrieQ_le (hW : NoWrong M side read) {L P : ℕ}
    (draws : Fin P → FreeMonoid α) (hL : ∀ i, (draws i).toList.length ≤ L) :
    ((level3TrieQ C y cap draws).reads read).card
      ≤ P * (probeSifts L * (Fintype.card σ + 1))
        + Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α * (Fintype.card σ + 2)) := by
  have hfin := (trie_final_tree C y cap read hW draws).2.1
  simp only [indecisiveLStarLevel3Trie, level3TrieQ, eval_map] at hfin
  unfold level3TrieQ
  rw [reads_map]
  have h := card_reads_runTrieProbesQ_le C y cap read hW (Fintype.card σ) (List.ofFn draws)
    trieInitialState (fun z hz => by obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hz; exact hL i)
    (by rw [trieL3_initial]; exact ⟨inv_start read, reached_start read⟩) hfin
  have h0 : (trieInitialState : TrieState α).tree.leaves.length = 2 := by
    simp [trieInitialState, newTrieHypothesis, DTree.leaves]
  rwa [List.length_ofFn, h0, show Fintype.card σ + 2 - 2 = Fintype.card σ by omega] at h

end Random

end OrthoDFA

end
