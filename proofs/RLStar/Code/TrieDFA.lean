import RLStar.Code.CoveredStart
import RLStar.Code.Level3Trie

/-! The DFA a round with local walk starts outputs: one state per expanded prefix, each of its
children outside the expanded prefixes going to the leaf that child sifts to, and the hypothesis's
leaves and learned edges. -/

noncomputable section

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- Run past the expanded prefixes `X` of `x`, then from the leaf its first prefix outside them
sifts to, `H` follows learned edges along the rest of `x` to a leaf, which is on the root's accept
side exactly when `b`.

Known modelling gap. The Python's DFA sifts those prefixes with the tree's base family at its
decision boundary, where the round read them with its gated cut, `read` here; it also runs an
unlearned edge as a loop on its leaf and an undecided sift to its start leaf, where here both count
as wrong. -/
def HypothesisDFA.TrieClassifies (H : HypothesisDFA α) (read : FreeMonoid α → ARU)
    (X : Finset (FreeMonoid α)) (x : FreeMonoid α) (b : Prop) : Prop :=
  ∃ a, H.tree.sift read (prefixOf x (Random.walkStartIn X x)) = .inl a
    ∧ H.Classifies a (FreeMonoid.ofList (x.toList.drop (Random.walkStartIn X x))) b

end OrthoDFA

end
