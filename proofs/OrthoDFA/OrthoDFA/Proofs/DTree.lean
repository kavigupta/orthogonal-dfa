import OrthoDFA.DTree
import Mathlib.Tactic

/-!
# Routing is correct

By induction on the tree: a node reads `z` followed by its midfix, and a decided read hands the
rest of the route to the child it picks.
-/

namespace OrthoDFA

namespace DTree

variable {α : Type*} (cut : FreeMonoid α → Option Bool)

theorem route_reads : ∀ (T : DTree α) (z : FreeMonoid α) (i : ℕ),
    i < (T.route cut z).1.length →
    (T.route cut z).1[i]? = some (z * T.midAt (((T.route cut z).1.filterMap cut).take i))
  | .leaf, z, i, h => by simp [route] at h
  | .node m r a, z, i, h => by
    rcases hc : cut (z * m) with _ | _ | _
    · simp only [route, hc] at h ⊢
      rcases i with _ | i
      · simp [midAt]
      · simp at h
    · simp only [route, hc] at h ⊢
      rcases i with _ | i
      · simp [midAt]
      · simp only [List.getElem?_cons_succ, List.filterMap_cons, hc, List.take_succ_cons, midAt]
        exact route_reads r z i (by simpa using h)
    · simp only [route, hc] at h ⊢
      rcases i with _ | i
      · simp [midAt]
      · simp only [List.getElem?_cons_succ, List.filterMap_cons, hc, List.take_succ_cons, midAt]
        exact route_reads a z i (by simpa using h)

theorem route_length : ∀ (T : DTree α) (z : FreeMonoid α), (T.route cut z).1.length ≤ T.depth
  | .leaf, z => by simp [route, depth]
  | .node m r a, z => by
    simp only [route, depth]
    split
    · have := route_length a z; simp only [List.length_cons]; omega
    · have := route_length r z; simp only [List.length_cons]; omega
    · simp

theorem route_decided : ∀ (T : DTree α) (z : FreeMonoid α) (p : List Bool),
    (T.route cut z).2 = some p →
      p ∈ T.leaves ∧ p = (T.route cut z).1.filterMap cut ∧ ∀ w ∈ (T.route cut z).1, cut w ≠ none
  | .leaf, z, p, h => by
    simp only [route, Option.some.injEq] at h
    subst h
    simp [leaves, route]
  | .node m r a, z, p, h => by
    simp only [route] at h ⊢
    split at h
    · rename_i hc
      obtain ⟨p', hp', rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := route_decided a z p' hp'
      refine ⟨by simp [leaves, h1], by simp [hc, ← h2], ?_⟩
      intro w hw
      rcases List.mem_cons.1 hw with rfl | hw
      · simp [hc]
      · exact h3 w hw
    · rename_i hc
      obtain ⟨p', hp', rfl⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨h1, h2, h3⟩ := route_decided r z p' hp'
      refine ⟨by simp [leaves, h1], by simp [hc, ← h2], ?_⟩
      intro w hw
      rcases List.mem_cons.1 hw with rfl | hw
      · simp [hc]
      · exact h3 w hw
    · simp at h

theorem route_undecided : ∀ (T : DTree α) (z : FreeMonoid α), (T.route cut z).2 = none →
    ∃ w, (T.route cut z).1.getLast? = some w ∧ cut w = none
      ∧ ∀ w' ∈ (T.route cut z).1.dropLast, cut w' ≠ none
  | .leaf, z, h => by simp [route] at h
  | .node m r a, z, h => by
    simp only [route] at h ⊢
    split at h
    · rename_i hc
      have h' : (a.route cut z).2 = none := by simpa using h
      obtain ⟨w, hw, hcw, hrest⟩ := route_undecided a z h'
      have hne : (a.route cut z).1 ≠ [] := by rintro he; simp [he] at hw
      refine ⟨w, by rw [List.getLast?_cons, hw]; rfl, hcw, ?_⟩
      rw [List.dropLast_cons_of_ne_nil hne]
      intro w' hw'
      rcases List.mem_cons.1 hw' with rfl | hw'
      · simp [hc]
      · exact hrest w' hw'
    · rename_i hc
      have h' : (r.route cut z).2 = none := by simpa using h
      obtain ⟨w, hw, hcw, hrest⟩ := route_undecided r z h'
      have hne : (r.route cut z).1 ≠ [] := by rintro he; simp [he] at hw
      refine ⟨w, by rw [List.getLast?_cons, hw]; rfl, hcw, ?_⟩
      rw [List.dropLast_cons_of_ne_nil hne]
      intro w' hw'
      rcases List.mem_cons.1 hw' with rfl | hw'
      · simp [hc]
      · exact hrest w' hw'
    · rename_i hc
      simp [hc]

end DTree

theorem routeCorrect : RouteCorrect := fun cut T z =>
  ⟨fun i h => by
    have := DTree.route_reads cut T z i h
    rwa [List.getElem?_eq_getElem h, Option.some.injEq] at this,
  DTree.route_length cut T z, DTree.route_decided cut T z, DTree.route_undecided cut T z⟩

end OrthoDFA
