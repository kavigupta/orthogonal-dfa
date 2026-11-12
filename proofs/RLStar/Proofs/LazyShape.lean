import RLStar.LowLevelClaims.LazyRound
import RLStar.Proofs.RandomProbe
import RLStar.Proofs.TrieMass

/-!
# A round with a lazy trie: its shape

A probe harvests only strings it read (`mem_reads_siteHarvestQ`), each read undecided, so every
listed string is one the round read undecided. A block expands the walk start, shorter than `K`, of
a string it repeats, and every shorter prefix of that string is in `X`, so the trie stays well
formed (`trieOK_union`) and grows only by prefixes shorter than `K` of listed strings.
-/

noncomputable section

namespace OrthoDFA

open Query

namespace Random

section Probe

variable {β : Type*} (r : FreeMonoid β → ARU)

private lemma mem_undecidedStrings_of_ends {o : Outcome β} {z : FreeMonoid β}
    (h : z ∈ endsString o) : z ∈ o.undecidedStrings := by
  rcases o with _ | _ | _ | z' | ⟨zs, _ | _⟩ <;>
    simp only [endsString, Outcome.undecidedStrings, List.not_mem_nil] at h ⊢
  · exact h
  · exact List.mem_of_mem_take h

private lemma mem_undecidedStrings_of_middle {o : Outcome β} {z : FreeMonoid β}
    (h : z ∈ middleString o) : z ∈ o.undecidedStrings := by
  rcases o with _ | _ | _ | z' | ⟨zs, _ | _⟩ <;>
    simp only [middleString, Outcome.undecidedStrings, List.not_mem_nil] at h ⊢
  exact List.mem_of_mem_take h

private lemma eval_siteHarvestQ' (k : ℕ) (H : HypothesisDFA β) (s : Level3Site)
    (x : FreeMonoid β) : (siteHarvestQ k H s x).eval r = siteHarvest r k H s x := by
  cases s <;> simp [siteHarvestQ, siteHarvest, probeOrAgree]

variable [Countable β]

private lemma mem_reads_siftQ : ∀ (T : DTree β) (v z : FreeMonoid β),
    (T.siftQ v).eval r = .inr z → z ∈ (T.siftQ v).reads r
  | .leaf, _, _, h => by simp [DTree.siftQ] at h
  | .node m rj ac, v, z, h => by
    simp only [DTree.siftQ, eval_ask, reads_ask] at h ⊢
    rcases hr : r (v * m) with _ | _ | _ <;>
      simp only [hr, eval_map, reads_map, eval_pure] at h ⊢
    · rcases hs : (ac.siftQ v).eval r with a | z' <;> rw [hs] at h
      · simp at h
      · obtain rfl : z' = z := by simpa using h
        exact Finset.mem_insert_of_mem (mem_reads_siftQ ac v _ hs)
    · rcases hs : (rj.siftQ v).eval r with a | z' <;> rw [hs] at h
      · simp at h
      · obtain rfl : z' = z := by simpa using h
        exact Finset.mem_insert_of_mem (mem_reads_siftQ rj v _ hs)
    · simp only [Sum.inr.injEq] at h
      simp [h]

private lemma mem_reads_undecidedAtQ (T : DTree β) (x : FreeMonoid β) (p : ℕ) :
    ∀ z ∈ (undecidedAtQ T x p).eval r, z ∈ (undecidedAtQ T x p).reads r := by
  intro z hz
  simp only [undecidedAtQ, eval_map, reads_map] at hz ⊢
  rcases hs : (T.siftQ (prefixOf x p)).eval r with a | z' <;> rw [hs] at hz
  · simp at hz
  · obtain rfl : z = z' := by simpa using hz
    exact mem_reads_siftQ r T _ _ hs

private lemma mem_reads_searchOutcomeQ (T : DTree β) (x : FreeMonoid β) (st : ℕ → List Bool)
    (un : Option (β × List Bool)) (hi : ℕ) (res : SearchResult) :
    ∀ z ∈ ((searchOutcomeQ T x st un hi res).eval r).undecidedStrings,
      z ∈ (searchOutcomeQ T x st un hi res).reads r := by
  intro z hz
  cases res with
  | undecidedHi =>
    simp only [searchOutcomeQ, eval_map, reads_map, Outcome.undecidedStrings] at hz ⊢
    exact mem_reads_undecidedAtQ r T x hi z hz
  | undecidedMid p =>
    simp only [searchOutcomeQ, eval_map, reads_map, Outcome.undecidedStrings] at hz ⊢
    exact mem_reads_undecidedAtQ r T x p z hz
  | allAgree =>
    simp only [searchOutcomeQ, eval_pure] at hz
    split at hz <;> simp [Outcome.undecidedStrings] at hz
  | edge p =>
    simp only [searchOutcomeQ] at hz
    split at hz
    · simp only [eval_map] at hz
      split at hz <;> simp [Outcome.undecidedStrings] at hz
    · simp [Outcome.undecidedStrings] at hz
  | _ => simp [searchOutcomeQ, Outcome.undecidedStrings] at hz

private lemma mem_reads_probeQ (T : DTree β) (E : Edges β) (k : ℕ) (x : FreeMonoid β) :
    ∀ z ∈ ((probeQ T E k x).eval r).undecidedStrings, z ∈ (probeQ T E k x).reads r := by
  intro z hz
  simp only [probeQ, eval_bind, reads_bind] at hz ⊢
  revert hz
  rcases hs : (T.siftQ (prefixOf x k)).eval r with a | z₀ <;> intro hz <;> dsimp only at hz ⊢
  · refine Finset.mem_union_right _ ?_
    split at hz
    · simp only [eval_bind, reads_bind] at hz ⊢
      revert hz
      rcases ht : (T.siftQ (prefixOf x _)).eval r with t | z₁ <;> intro hz <;> dsimp only at hz ⊢
      · simp only [eval_bind, reads_bind] at hz ⊢
        exact Finset.mem_union_right _ (Finset.mem_union_right _
          (mem_reads_searchOutcomeQ r T x _ _ _ _ z hz))
      · simp only [eval_pure, Outcome.undecidedStrings, List.mem_singleton] at hz
        subst hz
        exact Finset.mem_union_left _ (mem_reads_siftQ r T _ _ ht)
    · simp only [eval_bind, reads_bind] at hz ⊢
      exact Finset.mem_union_right _ (mem_reads_searchOutcomeQ r T x _ _ _ _ z hz)
  · simp only [eval_pure, Outcome.undecidedStrings, List.mem_singleton] at hz
    subst hz
    exact Finset.mem_union_left _ (mem_reads_siftQ r T _ _ hs)

private lemma mem_reads_siteHarvestQ (k : ℕ) (H : HypothesisDFA β) (s : Level3Site)
    (x : FreeMonoid β) :
    ∀ z ∈ (siteHarvestQ k H s x).eval r, z ∈ (siteHarvestQ k H s x).reads r := by
  have hp : ∀ z ∈ ((probeOrAgreeQ H.tree H.edges k x).eval r).undecidedStrings,
      z ∈ (probeOrAgreeQ H.tree H.edges k x).reads r := by
    unfold probeOrAgreeQ
    split_ifs
    · simp [Outcome.undecidedStrings]
    · exact mem_reads_probeQ r _ _ k x
  intro z hz
  cases s <;> simp only [siteHarvestQ, eval_map, reads_map] at hz ⊢
  · exact hp z (mem_undecidedStrings_of_ends hz)
  · exact hp z (mem_undecidedStrings_of_middle hz)

end Probe

variable {α : Type*} [Fintype α]

omit [Fintype α] in
private lemma blockQ_mem [Countable α] (read : FreeMonoid α → ARU) (K : ℕ)
    (H : HypothesisDFA (Option α)) (s : Level3Site) (X : Finset (FreeMonoid α)) :
    ∀ xs : List (FreeMonoid α),
    ∀ w ∈ (blockQ K H s X xs).eval read,
      w ∈ (blockQ K H s X xs).reads read ∧ read w = .undecided
  | [] => by simp [blockQ]
  | x :: xs => by
    intro w hw
    simp only [blockQ, eval_bind, eval_map, reads_bind, reads_map, List.mem_append,
      List.mem_map] at hw ⊢
    rcases hw with ⟨z, hz, rfl⟩ | hw
    · rw [eval_subst] at hz
      change z ∈ (siteHarvestQ K H s (padIn K X x)).eval (padRead read) at hz
      refine ⟨Finset.mem_union_left _ ?_, ?_⟩
      · rw [reads_subst]
        exact Finset.mem_biUnion.2 ⟨z, mem_reads_siteHarvestQ _ K H s _ z hz, by simp [unpadQ]⟩
      · rw [eval_siteHarvestQ'] at hz
        cases s
        · exact (probeOrAgree_ends (padRead read) H.tree (padIn K X x) H.edges K z hz).2
        · exact (probeOrAgree_pt (padRead read) H.tree (padIn K X x) H.edges K z hz).2
    · obtain ⟨h1, h2⟩ := blockQ_mem read K H s X xs w hw
      exact ⟨Finset.mem_union_right _ h1, h2⟩

private lemma blockUpdate_spec (K : ℕ) (X : Finset (FreeMonoid α)) :
    ∀ ws ℓ : List (FreeMonoid α),
      ℓ <+: (blockUpdate K X ℓ ws).1 ∧ (ℓ.Nodup → (blockUpdate K X ℓ ws).1.Nodup)
        ∧ (∀ w ∈ (blockUpdate K X ℓ ws).1, w ∈ ℓ ∨ w ∈ ws)
        ∧ ∀ u ∈ (blockUpdate K X ℓ ws).2, ∃ w ∈ (blockUpdate K X ℓ ws).1,
          walkStartIn X w < K ∧ u = prefixOf w (walkStartIn X w)
  | [], ℓ => by simp [blockUpdate]
  | w :: ws, ℓ => by
    rw [blockUpdate]
    split_ifs with hw hk
    · obtain ⟨h1, h2, h3, h4⟩ := blockUpdate_spec K X ws ℓ
      refine ⟨h1, h2, fun v hv => (h3 v hv).imp_right (List.mem_cons_of_mem _), fun u hu => ?_⟩
      rcases Finset.mem_insert.1 hu with rfl | hu
      · exact ⟨w, h1.subset hw, hk, rfl⟩
      · exact h4 u hu
    · obtain ⟨h1, h2, h3, h4⟩ := blockUpdate_spec K X ws ℓ
      exact ⟨h1, h2, fun v hv => (h3 v hv).imp_right (List.mem_cons_of_mem _), h4⟩
    · obtain ⟨h1, h2, h3, h4⟩ := blockUpdate_spec K X ws (ℓ ++ [w])
      refine ⟨(List.prefix_append ℓ [w]).trans h1, fun hℓ => h2 ?_, fun v hv => ?_, h4⟩
      · exact hℓ.append (List.nodup_singleton w) (List.disjoint_singleton.2 hw)
      rcases h3 v hv with hv | hv
      · rcases List.mem_append.1 hv with hv | hv
        · exact Or.inl hv
        · exact Or.inr (by simp_all)
      · exact Or.inr (List.mem_cons_of_mem _ hv)

omit [Fintype α] in
private lemma prefixOf_prefixOf_min (w : FreeMonoid α) (k i : ℕ) :
    prefixOf (prefixOf w k) i = prefixOf w (min i k) := by
  simp [prefixOf, List.take_take]

omit [Fintype α] in
private lemma trieOK_union [Countable α] {K : ℕ} {X Y : Finset (FreeMonoid α)} (hX : TrieOK K X)
    (hY : ∀ u ∈ Y, ∃ w : FreeMonoid α, walkStartIn X w < K ∧ u = prefixOf w (walkStartIn X w)) :
    TrieOK K (X ∪ Y) := by
  refine ⟨fun u hu => ?_, fun u hu i => ?_⟩
  · rcases Finset.mem_union.1 hu with hu | hu
    · exact hX.1 u hu
    · obtain ⟨w, hk, rfl⟩ := hY u hu
      simp only [prefixOf, FreeMonoid.toList_ofList, List.length_take]
      omega
  · rcases Finset.mem_union.1 hu with hu | hu
    · exact Finset.mem_union_left _ (hX.2 u hu i)
    · obtain ⟨w, hk, rfl⟩ := hY _ hu
      rw [prefixOf_prefixOf_min]
      by_cases hi : i < walkStartIn X w
      · rw [min_eq_left hi.le]
        exact Finset.mem_union_left _ ((walkStartIn_spec X w).2.1 i hi)
      · rw [min_eq_right (not_lt.1 hi)]
        exact Finset.mem_union_right _ hu

/-- What a top-up from `X` and `ℓ` may end with, where it reads `R`. -/
private def TopUpOK (read : FreeMonoid α → ARU) (K nh : ℕ) (X : Finset (FreeMonoid α))
    (ℓ : List (FreeMonoid α)) (R : Finset (FreeMonoid α)) {b : ℕ} :
    Option (TopUpEnd α × List (Fin b → FreeMonoid α)) → Prop
  | none => True
  | some (.full ℓ', _) => ℓ <+: ℓ' ∧ ℓ'.Nodup ∧ nh ≤ ℓ'.length
      ∧ ∀ w ∈ ℓ', w ∈ ℓ ∨ w ∈ R ∧ read w = .undecided
  | some (.retract ℓ' X', _) => (TrieOK K X' ∧ X ⊆ X' ∧ ℓ'.Nodup ∧ ℓ'.length < nh ∧ ℓ <+: ℓ'
      ∧ ∀ u ∈ X', u ∈ X ∨ ∃ w ∈ ℓ', ∃ i < K, u = prefixOf w i)
      ∧ ∀ w ∈ ℓ', w ∈ ℓ ∨ w ∈ R ∧ read w = .undecided

private lemma topUpQ_ok (read : FreeMonoid α → ARU) (H : HypothesisDFA (Option α))
    (s : Level3Site) (K nh : ℕ) {b : ℕ} :
    ∀ (Bs : List (Fin b → FreeMonoid α)) (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)),
      TrieOK K X → ℓ.Nodup →
      TopUpOK read K nh X ℓ ((topUpQ K nh H s Bs X ℓ).reads read)
        ((topUpQ K nh H s Bs X ℓ).eval read)
  | [], _, _, _, _ => trivial
  | B :: Bs, X, ℓ, hX, hℓ => by
    have hB := blockQ_mem read K H s X (List.ofFn B)
    simp only [topUpQ, eval_bind, reads_bind]
    generalize (blockQ K H s X (List.ofFn B)).eval read = ws at hB ⊢
    obtain ⟨h1, h2, h3, h4⟩ := blockUpdate_spec K X ws ℓ
    set RB := (blockQ K H s X (List.ofFn B)).reads read
    have hws : ∀ w ∈ (blockUpdate K X ℓ ws).1, w ∈ ℓ ∨ w ∈ RB ∧ read w = .undecided :=
      fun w hw => (h3 w hw).imp_right (hB w)
    split_ifs with hfull hret
    · exact ⟨h1, h2 hℓ, hfull, fun w hw =>
        (hws w hw).imp_right fun h => ⟨Finset.mem_union_left _ h.1, h.2⟩⟩
    · exact ⟨⟨hX, subset_rfl, h2 hℓ, by omega, h1, fun u hu => Or.inl hu⟩, fun w hw =>
        (hws w hw).imp_right fun h => ⟨Finset.mem_union_left _ h.1, h.2⟩⟩
    · set r := blockUpdate K X ℓ ws
      have ih := topUpQ_ok read H s K nh Bs (X ∪ r.2) r.1
        (trieOK_union hX fun u hu => (h4 u hu).imp fun w hw => hw.2) (h2 hℓ)
      generalize (topUpQ K nh H s Bs (X ∪ r.2) r.1).eval read = o at ih ⊢
      set R' := (topUpQ K nh H s Bs (X ∪ r.2) r.1).reads read
      have hR : ∀ w, w ∈ r.1 ∨ w ∈ R' ∧ read w = .undecided →
          w ∈ ℓ ∨ w ∈ RB ∪ R' ∧ read w = .undecided := by
        rintro w (hw | hw)
        · exact (hws w hw).imp_right fun h => ⟨Finset.mem_union_left _ h.1, h.2⟩
        · exact Or.inr ⟨Finset.mem_union_right _ hw.1, hw.2⟩
      rcases o with _ | ⟨_ | _, rest⟩
      · trivial
      · rename_i ℓ'
        obtain ⟨g1, g2, g3, g4⟩ := ih
        exact ⟨h1.trans g1, g2, g3, fun w hw => hR w (g4 w hw)⟩
      · rename_i ℓ' X'
        obtain ⟨⟨g1, g2, g3, g4, g5, g6⟩, g7⟩ := ih
        refine ⟨⟨g1, Finset.subset_union_left.trans g2, g3, g4, h1.trans g5, fun u hu => ?_⟩,
          fun w hw => hR w (g7 w hw)⟩
        rcases g6 u hu with hu | hu
        · rcases Finset.mem_union.1 hu with hu | hu
          · exact Or.inl hu
          · obtain ⟨w, hw, hk, rfl⟩ := h4 u hu
            exact Or.inr ⟨w, g5.subset hw, _, hk, rfl⟩
        · exact Or.inr hu

theorem topUpQ_retract_inv (read : FreeMonoid α → ARU) (H : HypothesisDFA (Option α))
    (s : Level3Site) (K nh : ℕ) {b : ℕ} (Bs : List (Fin b → FreeMonoid α))
    (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)) (hX : TrieOK K X) (hℓ : ℓ.Nodup)
    (_hlen : ℓ.length < nh) {ℓ' : List (FreeMonoid α)} {X' : Finset (FreeMonoid α)}
    {rest : List (Fin b → FreeMonoid α)}
    (h : (topUpQ K nh H s Bs X ℓ).eval read = some (.retract ℓ' X', rest)) :
    TrieOK K X' ∧ X ⊆ X' ∧ ℓ'.Nodup ∧ ℓ'.length < nh ∧ ℓ <+: ℓ'
      ∧ ∀ u ∈ X', u ∈ X ∨ ∃ w ∈ ℓ', ∃ i < K, u = prefixOf w i := by
  have := topUpQ_ok read H s K nh Bs X ℓ hX hℓ
  rw [h] at this
  exact this.1

theorem topUpQ_full_inv (read : FreeMonoid α → ARU) (H : HypothesisDFA (Option α))
    (s : Level3Site) (K nh : ℕ) {b : ℕ} (Bs : List (Fin b → FreeMonoid α))
    (X : Finset (FreeMonoid α)) (ℓ : List (FreeMonoid α)) (hX : TrieOK K X) (hℓ : ℓ.Nodup)
    {ℓ' : List (FreeMonoid α)} {rest : List (Fin b → FreeMonoid α)}
    (h : (topUpQ K nh H s Bs X ℓ).eval read = some (.full ℓ', rest)) :
    ℓ'.Nodup ∧ nh ≤ ℓ'.length ∧ ℓ <+: ℓ' := by
  have := topUpQ_ok read H s K nh Bs X ℓ hX hℓ
  rw [h] at this
  exact ⟨this.2.1, this.2.2.1, this.1⟩

omit [Fintype α] in
open scoped Classical in
private lemma upd_self (ls : LazyLists α) (s : Level3Site) (ℓ : List (FreeMonoid α)) :
    Function.update ls s ℓ s = ℓ :=
  Function.update_self ..

omit [Fintype α] in
open scoped Classical in
private lemma upd_ne (ls : LazyLists α) {s s' : Level3Site} (ℓ : List (FreeMonoid α))
    (h : s' ≠ s) : Function.update ls s ℓ s' = ls s' :=
  Function.update_of_ne h ..

/-- What epochs from `X` and `ls` may end with, where they read `R`. -/
private def EpochsOK (read : FreeMonoid α → ARU) (K nh : ℕ) (X : Finset (FreeMonoid α))
    (ls : LazyLists α) (R : Finset (FreeMonoid α)) (r : LazyResult α) : Prop :=
  TrieOK K r.expanded ∧ X ⊆ r.expanded
    ∧ (∀ u ∈ r.expanded, u ∈ X ∨ ∃ s, ∃ w ∈ r.lists s, ∃ i < K, u = prefixOf w i)
    ∧ (∀ s, ls s <+: r.lists s) ∧ (∀ s, (r.lists s).Nodup)
    ∧ (r.ending = .consistent → ∀ s, (r.lists s).length < nh)
    ∧ (∀ s ℓ, r.ending = .harvest s ℓ → r.lists s = ℓ ∧ nh ≤ ℓ.length)
    ∧ ∀ s, ∀ w ∈ r.lists s, w ∈ ls s ∨ w ∈ R ∧ read w = .undecided

omit [Fintype α] in
private lemma epochsOK_stay (read : FreeMonoid α → ARU) {K nh : ℕ} {X : Finset (FreeMonoid α)}
    {ls : LazyLists α} (R : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α))
    (hX : TrieOK K X) (hn : ∀ s, (ls s).Nodup) (hl : ∀ s, (ls s).length < nh)
    {en : LazyEnd α} (hen : ∀ s ℓ, en ≠ .harvest s ℓ) :
    EpochsOK read K nh X ls R ⟨H, X, ls, en⟩ :=
  ⟨hX, subset_rfl, fun _ hu => Or.inl hu, fun _ => List.prefix_refl _, hn, fun _ => hl,
    fun s ℓ h => absurd h (hen s ℓ), fun _ _ hw => Or.inl hw⟩

private lemma lazyEpochsQ_ok [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
    (K nh : ℕ) {J b : ℕ} : ∀ (e : ℕ) (ds : List (EpochDraws α C.probeBudget J b))
      (X : Finset (FreeMonoid α)) (ls : LazyLists α),
      TrieOK K X → (∀ s, (ls s).Nodup) → (∀ s, (ls s).length < nh) →
      EpochsOK read K nh X ls ((lazyEpochsQ C K nh e ds X ls).reads read)
        ((lazyEpochsQ C K nh e ds X ls).eval read)
  | _, [], X, ls, hX, hn, hl => epochsOK_stay read _ _ hX hn hl fun _ _ h => by cases h
  | e, d :: ds, X, ls, hX, hn, hl => by
    simp only [lazyEpochsQ, eval_bind, reads_bind]
    generalize (epochQ (epochSettings C e) K X d.1).eval read = res
    set RE := (epochQ (epochSettings C e) K X d.1).reads read
    rcases res with ⟨hyp, _ | s | _⟩
    · exact epochsOK_stay read _ _ hX hn hl fun _ _ h => by cases h
    · simp only [eval_bind, reads_bind]
      have hT := topUpQ_ok read hyp s K nh (List.ofFn d.2) X (ls s) hX (hn s)
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval read = o at hT ⊢
      set RT := (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).reads read
      rcases o with _ | ⟨_ | _, rest⟩ <;> dsimp only [eval_pure, reads_pure]
      · exact epochsOK_stay read _ _ hX hn hl fun _ _ h => by cases h
      · rename_i ℓ
        obtain ⟨g1, g2, g3, g4⟩ := hT
        refine ⟨hX, subset_rfl, fun _ hu => Or.inl hu, fun s' => ?_, fun s' => ?_,
          fun h => ?_, fun s' ℓ' h => ?_, fun s' w hw => ?_⟩
        · by_cases hs : s' = s
          · subst hs; simpa only [upd_self] using g1
          · simp only [upd_ne _ _ hs, List.prefix_refl]
        · by_cases hs : s' = s
          · subst hs; simpa only [upd_self] using g2
          · simpa only [upd_ne _ _ hs] using hn s'
        · cases h
        · cases h
          exact ⟨upd_self _ _ _, g3⟩
        · by_cases hs : s' = s
          · subst hs
            simp only [upd_self] at hw
            exact (g4 w hw).imp_right fun h =>
              ⟨Finset.mem_union_right _ (Finset.mem_union_left _ h.1), h.2⟩
          · simp only [upd_ne _ _ hs] at hw
            exact Or.inl hw
      · rename_i ℓ X'
        obtain ⟨⟨g1, g2, g3, g4, g5, g6⟩, g7⟩ := hT
        classical
        set ls' := Function.update ls s ℓ
        have hs' : ls' s = ℓ := upd_self _ _ _
        have hne : ∀ s', s' ≠ s → ls' s' = ls s' := fun s' hs => upd_ne _ _ hs
        have hn' : ∀ s', (ls' s').Nodup := fun s' => by
          by_cases hs : s' = s
          · subst hs; rw [hs']; exact g3
          · rw [hne s' hs]; exact hn s'
        have hl' : ∀ s', (ls' s').length < nh := fun s' => by
          by_cases hs : s' = s
          · subst hs; rw [hs']; exact g4
          · rw [hne s' hs]; exact hl s'
        have hpre : ∀ s', ls s' <+: ls' s' := fun s' => by
          by_cases hs : s' = s
          · subst hs; rw [hs']; exact g5
          · rw [hne s' hs]
        obtain ⟨i1, i2, i3, i4, i5, i6, i7, i8⟩ :=
          lazyEpochsQ_ok read C K nh (e + 1) ds X' ls' g1 hn' hl'
        refine ⟨i1, g2.trans i2, fun u hu => ?_, fun s' => (hpre s').trans (i4 s'), i5, i6, i7,
          fun s' w hw => ?_⟩
        · rcases i3 u hu with hu | hu
          · rcases g6 u hu with hu | ⟨w, hw, i, hi, rfl⟩
            · exact Or.inl hu
            · exact Or.inr ⟨s, w, (i4 s).subset (hs' ▸ hw), i, hi, rfl⟩
          · exact Or.inr hu
        · rcases i8 s' w hw with hw | hw
          · by_cases hs : s' = s
            · subst hs
              rw [hs'] at hw
              exact (g7 w hw).imp_right fun h =>
                ⟨Finset.mem_union_right _ (Finset.mem_union_left _ h.1), h.2⟩
            · rw [hne s' hs] at hw
              exact Or.inl hw
          · exact Or.inr ⟨Finset.mem_union_right _ (Finset.mem_union_right _ hw.1), hw.2⟩
    · exact epochsOK_stay read _ _ hX hn hl fun _ _ h => by cases h

end Random

open Random

theorem lazyRoundShape_holds : LazyRoundShape := by
  intro α _ _ read C K nh E J b X₀ draws hnh hX₀
  obtain ⟨h1, h2, h3, -, h5, h6, h7, h8⟩ := lazyEpochsQ_ok read C K nh 0 (List.ofFn draws) X₀
    (fun _ => []) hX₀ (fun _ => List.nodup_nil) fun _ => hnh
  simp only [lazyRoundQ]
  set r := (lazyEpochsQ C K nh 0 (List.ofFn draws) X₀ fun _ => []).eval read
  refine ⟨h1, h2, ?_, h5, h6, h7, fun s w hw => (h8 s w hw).resolve_left List.not_mem_nil⟩
  set L := r.lists .ends ++ r.lists .middles
  have hsub : r.expanded
      ⊆ X₀ ∪ L.toFinset.biUnion fun w => (Finset.range K).image (prefixOf w) := by
    intro u hu
    rcases h3 u hu with hu | ⟨s, w, hw, i, hi, rfl⟩
    · exact Finset.mem_union_left _ hu
    · refine Finset.mem_union_right _ (Finset.mem_biUnion.2
        ⟨w, ?_, Finset.mem_image.2 ⟨i, Finset.mem_range.2 hi, rfl⟩⟩)
      cases s <;> simp [L, hw]
  calc r.expanded.card
      ≤ X₀.card + (L.toFinset.biUnion fun w => (Finset.range K).image (prefixOf w)).card :=
        (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)
    _ ≤ X₀.card + L.length * K := by
        refine Nat.add_le_add_left ((Finset.card_biUnion_le_card_mul _ _ _ fun w _ =>
          Finset.card_image_le.trans (Finset.card_range K).le).trans ?_) _
        exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)
    _ = X₀.card + K * ((r.lists .ends).length + (r.lists .middles).length) := by
        rw [List.length_append, Nat.mul_comm]

end OrthoDFA
