import OrthoDFA.Proofs.DTree

/-! Level 2 keeps every leaf past the first two reached by some string, and a split whose leaf
holds two strings reading decidedly apart at its midfix keeps that. -/

namespace OrthoDFA

section Reached

variable {α : Type*} (read : FreeMonoid α → ARU)

/-- Every leaf past the first two is reached by some string. -/
def Reached (T : DTree α) : Prop :=
  ∀ p ∈ T.leaves, p = [false] ∨ p = [true] ∨ ∃ w, T.sift read w = .inl p

/-- A split whose leaf holds two strings reading decidedly apart at its midfix keeps every leaf
past the first two reached. -/
theorem reached_splitAt {T : DTree α} (hr : Reached read T) {p : List Bool} (hp : p ∈ T.leaves)
    {d w₁ w₂ : FreeMonoid α} (h₁ : T.sift read w₁ = .inl p) (h₂ : T.sift read w₂ = .inl p)
    (hd₁ : read (w₁ * d) ≠ .undecided) (hd₂ : read (w₂ * d) ≠ .undecided)
    (hd : read (w₁ * d) ≠ read (w₂ * d)) : Reached read (T.splitAt d p) := by
  have hchild : ∀ b, ∃ w, (T.splitAt d p).sift read w = .inl (p ++ [b]) := by
    have key : ∀ w, T.sift read w = .inl p → read (w * d) ≠ .undecided →
        (T.splitAt d p).sift read w = .inl (p ++ [decide (read (w * d) = .accept)]) := by
      intro w hw hu
      rw [DTree.sift_splitAt_eq read hw]
      cases hrd : read (w * d) <;> simp_all [DTree.sift]
    intro b
    cases hb : decide (read (w₁ * d) = .accept) with
    | true =>
      cases b
      · refine ⟨w₂, ?_⟩
        rw [key w₂ h₂ hd₂]
        have : read (w₁ * d) = .accept := by simpa using hb
        cases hr2 : read (w₂ * d) <;> simp_all
      · exact ⟨w₁, by rw [key w₁ h₁ hd₁, hb]⟩
    | false =>
      cases b
      · exact ⟨w₁, by rw [key w₁ h₁ hd₁, hb]⟩
      · refine ⟨w₂, ?_⟩
        rw [key w₂ h₂ hd₂]
        have : read (w₁ * d) ≠ .accept := by simpa using hb
        cases hr1 : read (w₁ * d) <;> cases hr2 : read (w₂ * d) <;> simp_all
  intro q hq
  rcases (DTree.mem_leaves_splitAt d T p hp q).1 hq with ⟨hq, hne⟩ | rfl | rfl
  · rcases hr q hq with h | h | ⟨w, hw⟩
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr ⟨w, DTree.sift_splitAt_ne read hw hne⟩)
  · exact Or.inr (Or.inr (hchild false))
  · exact Or.inr (Or.inr (hchild true))

end Reached

end OrthoDFA
