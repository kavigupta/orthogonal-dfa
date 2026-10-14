import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Proofs.MidBound
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.HypothesisDFA

/-!
# Leaves' paths
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

omit [Fintype α] [DecidableEq α] in
theorem siftCut_mem_paths {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α) (p : List Bool), t.siftCut cut x = .inl p → p ∈ t.leaves
  | .leaf, x, p, h => by
    simp only [siftCut, route, Sum.inl.injEq] at h
    simp [leaves, ← h]
  | .node m r a, x, p, h => by
    simp only [siftCut, route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> simp only [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [leaves, List.mem_append, List.mem_map]
        exact .inl ⟨q, siftCut_mem_paths r x q hr, rfl⟩
      · simp at h
    · rcases hr : (a.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [leaves, List.mem_append, List.mem_map]
        exact .inr ⟨q, siftCut_mem_paths a x q hr, rfl⟩
      · simp at h

end DTree

end OrthoDFA
