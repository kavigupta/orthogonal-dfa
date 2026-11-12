import RLStar.Code.TrieTable

/-!
# The trie DFA's walk-start sifts as a table

`TrieTableCorrect`: the table classifies a string longer than every expanded prefix as
`TrieClassifies` does. `TrieTableReads`: building it asks at most one string per tree level per
frontier string.
-/

namespace OrthoDFA

/-- Where every string of `X` is shorter than `x`, the table classifies `x` as `TrieClassifies`
does. -/
def TrieTableCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (H : HypothesisDFA α)
    (X : Finset (FreeMonoid α)) (x : FreeMonoid α) (b : Prop),
    (∀ u ∈ X, u.toList.length < x.toList.length) →
    (H.TableClassifies ((trieTableQ H.tree X).eval read) X x b ↔ H.TrieClassifies read X x b)

/-- Building the table asks at most one string per tree level per frontier string:

    |reads| ≤ (|Σ| |X| + 1) depth(T). -/
def TrieTableReads : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (T : DTree α)
    (X : Finset (FreeMonoid α)),
    (((trieTableQ T X).reads read).card : ℝ) ≤ (Fintype.card α * X.card + 1) * T.depth

end OrthoDFA
