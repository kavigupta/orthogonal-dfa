import RLStar.LowLevelClaims.TrieTable
import RLStar.Proofs.Query

/-!
# The trie DFA's walk-start sifts as a table: the proofs

Where `X`'s strings are shorter than `x`, `x`'s walk start is `ε` or a child of a string of `X`,
outside `X`, so it is on the frontier (`prefixOf_walkStartIn_mem_trieFrontier`) and the table holds
its sift. A sift asks at most one string per level (`DTree.card_reads_siftQ_le`).
-/

namespace OrthoDFA

open Query

variable {α : Type*}

namespace Random

theorem walkStartIn_spec (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    walkStartIn X x ≤ x.toList.length ∧ (∀ j < walkStartIn X x, prefixOf x j ∈ X)
      ∧ (walkStartIn X x < x.toList.length → prefixOf x (walkStartIn X x) ∉ X) := by
  classical
  unfold walkStartIn
  rcases h : (List.range x.toList.length).find? (fun k => decide (prefixOf x k ∉ X)) with _ | k
  · rw [List.find?_eq_none] at h
    refine ⟨le_rfl, fun j hj => ?_, fun h' => absurd h' (lt_irrefl _)⟩
    simpa using h j (List.mem_range.2 hj)
  · rw [List.find?_eq_some_iff_getElem] at h
    obtain ⟨hk, i, hi, hik, hbefore⟩ := h
    simp only [List.getElem_range] at hik
    subst hik
    simp only [Option.getD_some]
    exact ⟨by simpa using hi.le, fun j hj => by simpa using hbefore j hj,
      fun _ => by simpa using hk⟩

end Random

theorem DTree.card_reads_siftQ_le [DecidableEq (FreeMonoid α)] (read : FreeMonoid α → ARU) :
    ∀ (T : DTree α) (z : FreeMonoid α), ((T.siftQ z).reads read).card ≤ T.depth
  | .leaf, _ => by simp [DTree.siftQ]
  | .node m r a, z => by
    simp only [DTree.siftQ, reads_ask, DTree.depth]
    refine (Finset.card_insert_le _ _).trans (Nat.succ_le_succ ?_)
    rcases h : read (z * m) with _ | _ | _ <;> simp only [reads_map, reads_pure]
    · exact (DTree.card_reads_siftQ_le read a z).trans (le_max_right _ _)
    · exact (DTree.card_reads_siftQ_le read r z).trans (le_max_left _ _)
    · simp

variable [Fintype α]

theorem prefixOf_walkStartIn_mem_trieFrontier {X : Finset (FreeMonoid α)} {x : FreeMonoid α}
    (hX : ∀ u ∈ X, u.toList.length < x.toList.length) :
    prefixOf x (Random.walkStartIn X x) ∈ trieFrontier X := by
  obtain ⟨hle, hin, hout⟩ := Random.walkStartIn_spec X x
  generalize Random.walkStartIn X x = k at hle hin hout
  unfold trieFrontier
  rw [Finset.mem_filter, Finset.mem_insert, Finset.mem_image]
  refine ⟨?_, fun hmem => ?_⟩
  · rcases k with _ | j
    · left
      simp [prefixOf]
    · right
      have hj : j < x.toList.length := by omega
      refine ⟨(prefixOf x j, x.toList[j]), Finset.mem_product.2 ⟨hin j (by omega),
        Finset.mem_univ _⟩, ?_⟩
      change FreeMonoid.ofList (x.toList.take j) * FreeMonoid.of x.toList[j]
        = FreeMonoid.ofList (x.toList.take (j + 1))
      rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some,
        FreeMonoid.ofList_append, FreeMonoid.ofList_singleton]
  · rcases lt_or_eq_of_le hle with hlt | heq
    · exact hout hlt hmem
    · have := hX _ hmem
      simp [prefixOf, heq] at this

theorem eval_trieTableQ_of_mem (read : FreeMonoid α → ARU) {T : DTree α}
    {X : Finset (FreeMonoid α)} {w : FreeMonoid α} (hw : w ∈ trieFrontier X) :
    (trieTableQ T X).eval read w = T.sift read w := by
  simp [trieTableQ, eval_piList, hw, DTree.sift]

theorem trieTableCorrect_holds : TrieTableCorrect := by
  intro α _ _ read H X x b hX
  unfold HypothesisDFA.TableClassifies HypothesisDFA.TrieClassifies
  rw [eval_trieTableQ_of_mem read (prefixOf_walkStartIn_mem_trieFrontier hX)]

theorem card_trieFrontier_le (X : Finset (FreeMonoid α)) :
    (trieFrontier X).card ≤ Fintype.card α * X.card + 1 := by
  unfold trieFrontier
  refine (Finset.card_filter_le _ _).trans ((Finset.card_insert_le _ _).trans ?_)
  refine Nat.add_le_add_right (Finset.card_image_le.trans ?_) 1
  rw [Finset.card_product, Finset.card_univ, mul_comm]

theorem trieTableReads_holds : TrieTableReads := by
  intro α _ _ read T X
  have h : ((trieTableQ T X).reads read).card ≤ (Fintype.card α * X.card + 1) * T.depth := by
    rw [trieTableQ, reads_piList, Finset.toList_toFinset]
    exact (Finset.card_biUnion_le_card_mul _ _ _ fun z _ =>
      DTree.card_reads_siftQ_le read T z).trans (Nat.mul_le_mul_right _ (card_trieFrontier_le X))
  exact_mod_cast h

end OrthoDFA
