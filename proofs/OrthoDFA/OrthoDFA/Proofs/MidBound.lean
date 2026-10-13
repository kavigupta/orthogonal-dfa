import OrthoDFA.Proofs.Triple

/-!
# Where a bisection's undecided midpoint lands
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_mid (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .undecidedMid j →
      lo < j ∧ j < hi ∧ agrees j = none
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    split_ifs at h with hlh
    · rcases hm : agrees ((lo + hi) / 2) with _ | _ | _ <;> simp only [hm] at h
      · obtain rfl := Outcome.undecidedMid.inj h
        exact ⟨by omega, by omega, hm⟩
      · have := bracketAt_mid agrees ps fuel _ _ j h
        exact ⟨by omega, by omega, this.2.2⟩
      · have := bracketAt_mid agrees ps fuel _ _ j h
        exact ⟨by omega, by omega, this.2.2⟩

end OrthoDFA
