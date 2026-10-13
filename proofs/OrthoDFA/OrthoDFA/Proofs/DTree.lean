import OrthoDFA.DTree
import Mathlib.Tactic

/-!
# Sifting with a path is correct

By induction on the tree: a node reads `z` followed by its midfix, and a decided read hands the
rest of the sift to the child it picks.
-/

namespace OrthoDFA

namespace DTree

variable {α : Type*} (cut : FreeMonoid α → Option Bool)

theorem siftWithPath_reads : ∀ (T : DTree α) (z : FreeMonoid α) (i : ℕ),
    i < (T.siftWithPath cut z).1.length →
    (T.siftWithPath cut z).1[i]?
      = some (z * T.midfixAt (((T.siftWithPath cut z).1.filterMap cut).take i))
  | .leaf, z, i, h => by simp [siftWithPath] at h
  | .node m r a, z, i, h => by
    rcases hc : cut (z * m) with _ | _ | _
    · simp only [siftWithPath, hc] at h ⊢
      rcases i with _ | i
      · simp [midfixAt]
      · simp at h
    · simp only [siftWithPath, hc] at h ⊢
      rcases i with _ | i
      · simp [midfixAt]
      · simp only [List.getElem?_cons_succ, List.filterMap_cons, hc, List.take_succ_cons, midfixAt]
        exact siftWithPath_reads r z i (by simpa using h)
    · simp only [siftWithPath, hc] at h ⊢
      rcases i with _ | i
      · simp [midfixAt]
      · simp only [List.getElem?_cons_succ, List.filterMap_cons, hc, List.take_succ_cons, midfixAt]
        exact siftWithPath_reads a z i (by simpa using h)

theorem siftWithPath_length : ∀ (T : DTree α) (z : FreeMonoid α),
    (T.siftWithPath cut z).1.length ≤ T.depth
  | .leaf, z => by simp [siftWithPath, depth]
  | .node m r a, z => by
    simp only [siftWithPath, depth]
    split
    · have := siftWithPath_length a z; simp only [List.length_cons]; omega
    · have := siftWithPath_length r z; simp only [List.length_cons]; omega
    · simp

theorem siftWithPath_decided : ∀ (T : DTree α) (z : FreeMonoid α) (p : List Bool),
    (T.siftWithPath cut z).2 = some p →
      p ∈ T.leaves ∧ p = (T.siftWithPath cut z).1.filterMap cut
        ∧ ∀ w ∈ (T.siftWithPath cut z).1, cut w ≠ none
  | .leaf, z, p, h => by
    simp only [siftWithPath, Option.some.injEq] at h
    subst h
    simp [leaves, siftWithPath]
  | .node m r a, z, p, h => by
    simp only [siftWithPath] at h ⊢
    split at h
    · rename_i hc
      obtain ⟨p', hp', rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := siftWithPath_decided a z p' hp'
      refine ⟨by simp [leaves, h1], by simp [hc, ← h2], ?_⟩
      intro w hw
      rcases List.mem_cons.1 hw with rfl | hw
      · simp [hc]
      · exact h3 w hw
    · rename_i hc
      obtain ⟨p', hp', rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := siftWithPath_decided r z p' hp'
      refine ⟨by simp [leaves, h1], by simp [hc, ← h2], ?_⟩
      intro w hw
      rcases List.mem_cons.1 hw with rfl | hw
      · simp [hc]
      · exact h3 w hw
    · simp at h

theorem siftWithPath_undecided : ∀ (T : DTree α) (z : FreeMonoid α),
    (T.siftWithPath cut z).2 = none →
    ∃ w, (T.siftWithPath cut z).1.getLast? = some w ∧ cut w = none
      ∧ ∀ w' ∈ (T.siftWithPath cut z).1.dropLast, cut w' ≠ none
  | .leaf, z, h => by simp [siftWithPath] at h
  | .node m r a, z, h => by
    simp only [siftWithPath] at h ⊢
    split at h
    · rename_i hc
      have h' : (a.siftWithPath cut z).2 = none := by simpa using h
      obtain ⟨w, hw, hcw, hrest⟩ := siftWithPath_undecided a z h'
      have hne : (a.siftWithPath cut z).1 ≠ [] := by rintro he; simp [he] at hw
      refine ⟨w, by rw [List.getLast?_cons, hw]; rfl, hcw, ?_⟩
      rw [List.dropLast_cons_of_ne_nil hne]
      intro w' hw'
      rcases List.mem_cons.1 hw' with rfl | hw'
      · simp [hc]
      · exact hrest w' hw'
    · rename_i hc
      have h' : (r.siftWithPath cut z).2 = none := by simpa using h
      obtain ⟨w, hw, hcw, hrest⟩ := siftWithPath_undecided r z h'
      have hne : (r.siftWithPath cut z).1 ≠ [] := by rintro he; simp [he] at hw
      refine ⟨w, by rw [List.getLast?_cons, hw]; rfl, hcw, ?_⟩
      rw [List.dropLast_cons_of_ne_nil hne]
      intro w' hw'
      rcases List.mem_cons.1 hw' with rfl | hw'
      · simp [hc]
      · exact hrest w' hw'
    · rename_i hc
      simp [hc]

end DTree

theorem siftWithPathCorrect : SiftWithPathCorrect := fun cut T z =>
  ⟨fun i h => by
    have := DTree.siftWithPath_reads cut T z i h
    rwa [List.getElem?_eq_getElem h, Option.some.injEq] at this,
  DTree.siftWithPath_length cut T z, DTree.siftWithPath_decided cut T z,
  DTree.siftWithPath_undecided cut T z⟩

end OrthoDFA
