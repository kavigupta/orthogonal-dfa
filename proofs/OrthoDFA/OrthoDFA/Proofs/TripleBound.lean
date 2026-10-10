import OrthoDFA.Proofs.Triple

/-!
# The triples' claim

Given the pass's bits, draws with different first `k` letters read different strings, so their
contributions are independent, each pulled down on average by `fresh_first_le`. Chebyshev over
the pass's cells bounds the fluctuation by the most any one first `k` letters carry.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_triple_gt (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .triple j → lo < j
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h; simp at h
    rw [if_pos hlh] at h
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
      · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
      · obtain rfl := Outcome.triple.inj h; omega
      · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_triple_gt agrees ps fuel _ _ j h; omega

section Cells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

variable {Q : Type*}

end Cells

section PassCells

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α))

end PassCells

section Tail

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {Q : Type*}
variable (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (K : StageKnobs α)
  (k : ℕ) (seed probes : List (FreeMonoid α))

end Tail

end OrthoDFA
