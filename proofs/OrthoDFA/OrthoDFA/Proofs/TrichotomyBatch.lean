import OrthoDFA.Proofs.HarvestClasses

/-!
# The trichotomy's batch claims

The gate's tests settle on the wrong side of `acc` for some start with chance at most `a` per
look. A gate settling below then has every start disagreeing on more than `1 − acc`, so a covering
start leaves the classes and live edges more than `need`, and below `τ₀` every class fires on its
first hit: halving needs the refusal sample to miss that mass.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Combinatorics

omit [Fintype α] [DecidableEq α] in
theorem foldl_best (score : List Bool → ℕ) :
    ∀ (qs : List (List Bool)) (b₀ : List Bool),
      (qs.foldl (fun b q => if score b < score q then q else b) b₀ = b₀
        ∨ qs.foldl (fun b q => if score b < score q then q else b) b₀ ∈ qs)
      ∧ score b₀ ≤ score (qs.foldl (fun b q => if score b < score q then q else b) b₀)
      ∧ ∀ q ∈ qs, score q ≤ score (qs.foldl (fun b q => if score b < score q then q else b) b₀)
  | [], b₀ => by simp
  | q :: qs, b₀ => by
    simp only [List.foldl_cons]
    set b₁ := if score b₀ < score q then q else b₀
    obtain ⟨h1, h2, h3⟩ := foldl_best score qs b₁
    have hb₁ : (b₁ = q ∨ b₁ = b₀) ∧ score b₀ ≤ score b₁ ∧ score q ≤ score b₁ := by
      simp only [b₁]; split_ifs with h <;> simp <;> omega
    refine ⟨?_, hb₁.2.1.trans h2, ?_⟩
    · rcases h1 with h1 | h1
      · rw [h1]; rcases hb₁.1 with h | h
        · exact .inr (by simp [h])
        · exact .inl h
      · exact .inr (List.mem_cons_of_mem _ h1)
    · intro q' hq'
      rcases List.mem_cons.1 hq' with rfl | hq'
      · exact hb₁.2.2.trans h2
      · exact h3 q' hq'

omit [Fintype α] [DecidableEq α] in
theorem paths_ne_nil : ∀ t : DTree α, t.paths ≠ []
  | .leaf => by simp [DTree.paths]
  | .node _ r _ => by
    simp only [DTree.paths, ne_eq, List.append_eq_nil_iff, List.map_eq_nil_iff, not_and]
    exact fun h => absurd h (paths_ne_nil r)

end Combinatorics

section Outcomes

variable (R : CutReads α)

end Outcomes

section Need

variable (R : CutReads α) {Q : Type*}

end Need

section Stops

end Stops

section Rates

end Rates

section Batch

variable (R : CutReads α) {Q : Type*}

end Batch

end OrthoDFA
