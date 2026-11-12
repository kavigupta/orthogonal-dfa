import RLStar.Code.TrieDFA

/-! The trie DFA with every walk start's sift read in one program, as a table. -/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- `ε` and the children of `X`'s strings, outside `X`. -/
noncomputable def trieFrontier (X : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (insert 1 ((X ×ˢ (Finset.univ : Finset α)).image fun p => p.1 * FreeMonoid.of p.2)).filter
    (· ∉ X)

/-- The tree's sift of every frontier string of `X`. -/
noncomputable def trieTableQ (T : DTree α) (X : Finset (FreeMonoid α)) :
    Query (FreeMonoid α) ARU (FreeMonoid α → List Bool ⊕ FreeMonoid α) :=
  Query.piList T.siftQ (trieFrontier X).toList

/-- `TrieClassifies` with the walk start's sift read off the table `tab`. -/
def HypothesisDFA.TableClassifies (H : HypothesisDFA α)
    (tab : FreeMonoid α → List Bool ⊕ FreeMonoid α) (X : Finset (FreeMonoid α)) (x : FreeMonoid α)
    (b : Prop) : Prop :=
  ∃ a, tab (prefixOf x (Random.walkStartIn X x)) = .inl a
    ∧ H.Classifies a (FreeMonoid.ofList (x.toList.drop (Random.walkStartIn X x))) b

end OrthoDFA
