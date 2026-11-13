import RLStar.Proofs.TrieTable
import RLStar.Proofs.LazyTable
import RLStar.LowLevelClaims.LazyRound
import RLStar.LowLevelClaims.Trie

/-!
# The lazy round's DFA as a trie DFA over padded strings

The walk start's sift read off `lazyTableQ` classifies `x` as the trie DFA over `Option α`, its
trie every string shorter than `K`, classifies `padIn K X x` (`lazyClassifies_iff`), so
`lazyDisagree` is `trieDisagreeSet` at the padded string (`mem_lazyDisagree_iff`).
-/

noncomputable section

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem lazyClassifies_iff (read : FreeMonoid α → ARU) (K : ℕ) (H : HypothesisDFA (Option α))
    (X : Finset (FreeMonoid α)) (hX : Random.TrieOK K X) (x : FreeMonoid α) (b : Prop) :
    H.LazyClassifies ((Random.lazyTableQ K H.tree X).eval read) X x b
      ↔ H.TrieClassifies (Random.padRead read) (Random.completeTrie K) (Random.padIn K X x) b := by
  sorry -- false at x ∈ X; given K ≤ |x| it is Random.lazyClassifies_iff read H hX.1 hx b

theorem mem_lazyDisagree_iff (read : FreeMonoid α → ARU) (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) (x : FreeMonoid α) :
    x ∈ Random.lazyDisagree read K X H
      ↔ Random.padIn K X x
        ∈ Random.trieDisagreeSet (Random.padRead read) (Random.completeTrie K) H := by
  simp only [Random.lazyDisagree, Random.trieDisagreeSet, Set.mem_ofPred_eq]
  rw [Random.walkStartIn_completeTrie (Random.le_length_padIn K X x)]

end OrthoDFA

end
