import RLStar.Proofs.TrieTable
import RLStar.Proofs.LazyTable
import RLStar.LowLevelClaims.LazyRound
import RLStar.LowLevelClaims.Trie

/-!
# The lazy round's DFA as a trie DFA over padded strings

Past the trie the padded string's walk start is `K`, so `lazyDisagree` is `trieDisagreeSet` at the
padded string (`mem_lazyDisagree_iff`). The table side is `Random.lazyClassifies_iff`, in
`LazyTable`.
-/

noncomputable section

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [DecidableEq α] in
theorem mem_lazyDisagree_iff (read : FreeMonoid α → ARU) (K : ℕ) (X : Finset (FreeMonoid α))
    (H : HypothesisDFA (Option α)) (x : FreeMonoid α) :
    x ∈ Random.lazyDisagree read K X H
      ↔ Random.padIn K X x
        ∈ Random.trieDisagreeSet (Random.padRead read) (Random.completeTrie K) H := by
  simp only [Random.lazyDisagree, Random.trieDisagreeSet, Set.mem_ofPred_eq]
  rw [Random.walkStartIn_completeTrie (Random.le_length_padIn K X x)]

end OrthoDFA

end
