import OrthoDFA.Stage

/-!
# One node of a sift
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

theorem siftCut_node_none {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    (h : cut (x * m) = none) : (node m r a).siftCut cut x = .inr (x * m) := by
  simp [siftCut, route, h]

theorem siftCut_node_some {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    {b : Bool} (h : cut (x * m) = some b) :
    (node m r a).siftCut cut x = ((if b then a else r).siftCut cut x).map (b :: ·) id := by
  cases b <;> simp [siftCut, route, h]

/-- What a node cannot place is the string extended by its midfix. -/
theorem siftCut_inr {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) {x b : FreeMonoid α}, t.siftCut cut x = .inr b → ∃ m, b = x * m
  | .leaf, x, b, h => by simp [siftCut, route] at h
  | .node m r a, x, b, h => by
    rcases hc : cut (x * m) with _ | c
    · rw [siftCut_node_none hc] at h
      simp only [Sum.inr.injEq] at h
      exact ⟨m, h.symm⟩
    · rw [siftCut_node_some hc] at h
      cases c
      · simp only [Bool.false_eq_true, ↓reduceIte] at h
        rcases hr : r.siftCut cut x with q | q <;> rw [hr] at h <;> simp at h
        subst h
        exact siftCut_inr r hr
      · simp only [↓reduceIte] at h
        rcases ha : a.siftCut cut x with q | q <;> rw [ha] at h <;> simp at h
        subst h
        exact siftCut_inr a ha

end DTree

instance : Countable (FreeMonoid α) := inferInstanceAs (Countable (List α))

instance : MeasurableSingletonClass (FreeMonoid α) := ⟨fun _ => trivial⟩

end OrthoDFA
