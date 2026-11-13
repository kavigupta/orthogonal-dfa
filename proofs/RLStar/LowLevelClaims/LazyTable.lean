import RLStar.LowLevelClaims.LazyRound
import RLStar.LowLevelClaims.Trie

/-!
# The lazy trie's DFA

The DFA reads `x`'s walk-start sift off the padded table and walks the hypothesis along the rest
of `x`. On padded probes that is the hypothesis walked from `K` past the complete trie, so its
error is the round's disagreement plus the misread strings (`LazyTableCorrect`). Building the table
asks at most one string per tree level per frontier string (`LazyTableReads`).
-/

namespace OrthoDFA

open MeasureTheory Random

/-- Where every string of `X` is shorter than `K`, the probes are at least `K` long and the tree's
root is the empty string's split, the DFA errs on at most the round's disagreement and the misread
strings:

    D{x : DFA wrong on x} ≤ D(lazyDisagree) + D(misread). -/
def LazyTableCorrect : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
    (H : HypothesisDFA (Option α)) (K : ℕ) (X : Finset (FreeMonoid α)),
    (∀ u ∈ X, u.toList.length < K) → (∃ r a, H.tree = .node 1 r a) →
    ∀ (L : Set (FreeMonoid α)) (D : Measure (FreeMonoid α)) [IsFiniteMeasure D],
    (∀ᵐ x ∂D, K ≤ x.toList.length) →
    D.real {x | ¬ H.LazyClassifies ((lazyTableQ K H.tree X).eval read) X x (x ∈ L)}
      ≤ D.real (lazyDisagree read K X H) + D.real (misreadSet read L)

/-- Building the table asks at most one string per tree level per frontier string:

    |reads| ≤ (|Σ| |X| + 1) depth(T). -/
def LazyTableReads : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (K : ℕ)
    (T : DTree (Option α)) (X : Finset (FreeMonoid α)),
    (((lazyTableQ K T X).reads read).card : ℝ) ≤ (Fintype.card α * X.card + 1) * T.depth

end OrthoDFA
