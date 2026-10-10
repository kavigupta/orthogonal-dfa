import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base
import OrthoDFA.Proofs.HarvestClasses

/-!
# Leaves' paths
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

omit [Fintype α] [DecidableEq α] in
theorem sift_mem_paths {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α) (p : List Bool), t.sift cut x = .inl p → p ∈ t.paths
  | .leaf, x, p, h => by
    simp only [sift, route, Sum.inl.injEq] at h
    simp [paths, ← h]
  | .node m r a, x, p, h => by
    simp only [sift, route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> simp only [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [paths, List.mem_append, List.mem_map]
        exact .inl ⟨q, sift_mem_paths r x q hr, rfl⟩
      · simp at h
    · rcases hr : (a.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [paths, List.mem_append, List.mem_map]
        exact .inr ⟨q, sift_mem_paths a x q hr, rfl⟩
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem paths_ne_nil_of_node {m : FreeMonoid α} {r a : DTree α} :
    [] ∉ (DTree.node m r a).paths := by
  simp [paths]

omit [Fintype α] [DecidableEq α] in
theorem splitAt_paths_length (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), p ∈ t.paths →
      (t.splitAt d p).paths.length = t.paths.length + 1
  | .leaf, [], _ => by simp [splitAt, paths]
  | .leaf, _ :: _, h => by simp [paths] at h
  | .node _ _ _, [], h => absurd h paths_ne_nil_of_node
  | .node n r a, false :: p, h => by
    have : p ∈ r.paths := by simpa [paths] using h
    simp only [splitAt, paths, List.length_append, List.length_map,
      splitAt_paths_length d r p this]
    omega
  | .node n r a, true :: p, h => by
    have : p ∈ a.paths := by simpa [paths] using h
    simp only [splitAt, paths, List.length_append, List.length_map,
      splitAt_paths_length d a p this]
    omega

omit [Fintype α] [DecidableEq α] in
theorem append_not_mem_paths :
    ∀ (t : DTree α) (p : List Bool) (b : Bool), p ∈ t.paths → p ++ [b] ∉ t.paths
  | .leaf, p, b, h => by simp_all [paths]
  | .node _ r a, [], b, h => absurd h paths_ne_nil_of_node
  | .node _ r a, c :: p, b, h => by
    simp only [paths, List.mem_append, List.mem_map, List.cons_append] at h ⊢
    rintro (⟨q, hq, he⟩ | ⟨q, hq, he⟩) <;> rcases h with ⟨q', hq', he'⟩ | ⟨q', hq', he'⟩ <;>
      simp only [List.cons.injEq] at he he' <;> obtain ⟨rfl, rfl⟩ := he'
    · obtain ⟨-, rfl⟩ := he; exact append_not_mem_paths r q' b hq' hq
    · simp at he
    · simp at he
    · obtain ⟨-, rfl⟩ := he; exact append_not_mem_paths a q' b hq' hq

end DTree

end OrthoDFA
