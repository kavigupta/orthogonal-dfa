import RLStar.LowLevelClaims.LazyTable
import RLStar.Proofs.Trie
import RLStar.Proofs.TrieTable
import Mathlib.Data.List.ReduceOption

/-!
# The lazy trie's DFA: the proofs

A probe `x` at least `K` long, padded, walks from `K` past the complete trie, and its first `K`
letters are its key `u` padded, so the table classifies `x` as `TrieClassifies` does the padded
probe (`lazyClassifies_iff`). `trieSinglePieceCorrect` over the padded law then bounds the error.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Query

namespace Random

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [Fintype α] [DecidableEq α] in
private theorem unpad_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    unpad (padIn K X x) = x := by
  simp only [unpad, padIn, FreeMonoid.toList_ofList, List.reduceOption_append,
    List.reduceOption_replicate_none, List.nil_append]
  simp [List.reduceOption]

omit [Fintype α] [DecidableEq α] in
private theorem le_length_padIn (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    K ≤ (padIn K X x).toList.length := by
  have := walkStartIn_le X x
  simp only [padIn, FreeMonoid.toList_ofList, List.length_append, List.length_replicate,
    List.length_map]
  omega

omit [Fintype α] [DecidableEq α] in
private theorem walkStartIn_key (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    walkStartIn X (prefixOf x (walkStartIn X x)) = walkStartIn X x := by
  obtain ⟨hle, hin, -⟩ := walkStartIn_spec X x
  refine walkStartIn_eq (by rw [length_prefixOf']; omega) (fun j hj => ?_)
    (fun h => by rw [length_prefixOf'] at h; omega)
  have : prefixOf (prefixOf x (walkStartIn X x)) j = prefixOf x j := by
    simp only [prefixOf, FreeMonoid.toList_ofList, List.take_take]
    rw [min_eq_left hj.le]
  rw [this]
  exact hin j hj

omit [Fintype α] [DecidableEq α] in
private theorem padIn_key (K : ℕ) {X : Finset (FreeMonoid α)} {x : FreeMonoid α}
    (hK : walkStartIn X x ≤ K) :
    padIn K X (prefixOf x (walkStartIn X x)) = prefixOf (padIn K X x) K := by
  rw [padIn, walkStartIn_key]
  simp only [padIn, prefixOf, FreeMonoid.toList_ofList, List.take_append, List.length_replicate,
    List.take_replicate, ← List.map_take]
  rw [min_eq_right (Nat.sub_le K _), Nat.sub_sub_self hK]

omit [Fintype α] [DecidableEq α] in
private theorem drop_padIn (K : ℕ) {X : Finset (FreeMonoid α)} {x : FreeMonoid α}
    (hK : walkStartIn X x ≤ K) :
    (padIn K X x).toList.drop K = (x.toList.drop (walkStartIn X x)).map some := by
  simp only [padIn, FreeMonoid.toList_ofList, List.drop_append, List.length_replicate,
    List.drop_replicate, ← List.map_drop]
  rw [show K - (K - walkStartIn X x) = walkStartIn X x by omega,
    show K - walkStartIn X x - K = 0 by omega]
  simp

omit [DecidableEq α] in
private theorem eval_lazyTableQ_of_mem (read : FreeMonoid α → ARU) (K : ℕ)
    (T : DTree (Option α)) {X : Finset (FreeMonoid α)} {w : FreeMonoid α}
    (hw : w ∈ trieFrontier X) :
    (lazyTableQ K T X).eval read w = T.sift (padRead read) (padIn K X w) := by
  simp [lazyTableQ, eval_piList, hw, DTree.sift, unpadQ, Query.query]
  rfl

omit [DecidableEq α] in
private theorem lazyClassifies_iff (read : FreeMonoid α → ARU) (H : HypothesisDFA (Option α))
    {K : ℕ} {X : Finset (FreeMonoid α)} (hX : ∀ u ∈ X, u.toList.length < K) {x : FreeMonoid α}
    (hx : K ≤ x.toList.length) (b : Prop) :
    H.LazyClassifies ((lazyTableQ K H.tree X).eval read) X x b
      ↔ H.TrieClassifies (padRead read) (completeTrie K) (padIn K X x) b := by
  obtain ⟨-, hin, -⟩ := walkStartIn_spec X x
  have hK : walkStartIn X x ≤ K := by
    by_contra h
    have := hX _ (hin K (by omega))
    rw [length_prefixOf'] at this
    omega
  unfold HypothesisDFA.LazyClassifies HypothesisDFA.TrieClassifies
  rw [walkStartIn_completeTrie (le_length_padIn K X x), drop_padIn K hK, ← padIn_key K hK,
    eval_lazyTableQ_of_mem read K H.tree
      (prefixOf_walkStartIn_mem_trieFrontier fun u hu => (hX u hu).trans_le hx)]

end Random

open Random

theorem lazyTableCorrect_holds : LazyTableCorrect := by
  intro α _ _ read H K X hX hroot L D _ hlen
  have hπ : Measurable (padIn K X) := measurable_of_countable _
  have h := trieSinglePieceCorrect (padRead read) H (completeTrie K) hroot (unpad ⁻¹' L)
    (D.map (padIn K X))
  rw [map_measureReal_apply hπ (Set.to_countable _).measurableSet,
    map_measureReal_apply hπ (Set.to_countable _).measurableSet,
    map_measureReal_apply hπ (Set.to_countable _).measurableSet] at h
  have hdis : padIn K X ⁻¹' trieDisagreeSet (padRead read) (completeTrie K) H
      = lazyDisagree read K X H := by
    ext x
    simp only [trieDisagreeSet, lazyDisagree, Set.mem_preimage, Set.mem_ofPred_eq]
    rw [walkStartIn_completeTrie (le_length_padIn K X x)]
  have hmis : padIn K X ⁻¹' misreadSet (padRead read) (unpad ⁻¹' L) = misreadSet read L := by
    ext x
    have hL : padIn K X x ∈ unpad ⁻¹' L ↔ x ∈ L := by
      change unpad (padIn K X x) ∈ L ↔ x ∈ L
      rw [unpad_padIn]
    simp only [misreadSet, Set.mem_preimage, Set.mem_ofPred_eq, padRead, unpad_padIn]
    split_ifs with h₁ h₂ h₂
    · rfl
    · exact absurd (hL.1 h₁) h₂
    · exact absurd (hL.2 h₂) h₁
    · rfl
  rw [hdis, hmis] at h
  refine le_trans (le_of_eq (measureReal_congr ?_)) h
  filter_upwards [hlen] with x hx
  change (¬ _) = (¬ _)
  rw [lazyClassifies_iff read H hX hx, Set.mem_preimage, unpad_padIn]

theorem lazyTableReads_holds : LazyTableReads := by
  intro α _ _ read K T X
  have h : ((lazyTableQ K T X).reads read).card ≤ (Fintype.card α * X.card + 1) * T.depth := by
    rw [lazyTableQ]
    refine (card_reads_subst_le _ _ _ (n := 1) fun q _ => by
      simp [unpadQ, Query.query, reads]).trans ?_
    rw [mul_one, reads_piList, Finset.toList_toFinset]
    exact (Finset.card_biUnion_le_card_mul _ _ _ fun z _ =>
      DTree.card_reads_siftQ_le _ T _).trans (Nat.mul_le_mul_right _ (card_trieFrontier_le X))
  exact_mod_cast h

end OrthoDFA

end
