import OrthoDFA.Level3Claim
import OrthoDFA.Utils.CoveredStart

/-! What a probe that does not disagree promises. -/

namespace OrthoDFA

open Random

/-- A probe walked from `k` that neither searches nor stops undecided at its start or end sift
sifts its first `k` letters to a leaf from which the edges walk the rest of it to the leaf it sifts
to. Walked from `0`, under a tree whose root reads the string itself, the hypothesis started at the
leaf `ε` sifts to classifies the probe as its read, which is decided. -/
def ProbeAgreeCorrect : Prop :=
  ∀ {α : Type*} [DecidableEq α] (read : FreeMonoid α → ARU) (H : HypothesisDFA α) (k : ℕ)
    (x : FreeMonoid α), k ≤ x.toList.length →
    x ∉ searchSet read k H ∪ endsSet read k H →
    (∃ a e, H.tree.sift read (prefixOf x k) = .inl a
      ∧ walkEnd H.edges a (x.toList.drop k) = some e ∧ H.tree.sift read x = .inl e)
    ∧ (k = 0 → (∃ r a, H.tree = .node 1 r a) →
      ∃ s, H.tree.sift read 1 = .inl s ∧ read x ≠ .undecided
        ∧ H.Classifies s x (read x = .accept))

end OrthoDFA
