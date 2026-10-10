import OrthoDFA.Proofs.TripleBound
import OrthoDFA.StartAtK
import OrthoDFA.Pass
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# The harvest classes

Each class of `QualityHolds` as a computation and harvest meeting `HarvestSpec`: the ends' sifts
with the root's read untagged, the reads at an unlearned edge, and the search's middles.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Probe

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_pair_gt (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .pair j → lo < j
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h; simp at h
    rw [if_pos hlh] at h
    have hl0 : (lo + hi) / 2 - 1 = lo → (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1))
          = some true := fun e => if_pos e
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h hl0
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      all_goals first
        | (have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega)
        | (obtain rfl := Outcome.pair.inj h; omega)
        | (obtain rfl := Outcome.pair.inj h
           have hne : (lo + hi) / 2 - 1 ≠ lo := fun he => by simpa using hl0 he
           omega)
    · have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega

variable (cut : FreeMonoid α → Option Bool)

/-- Where a probe's search ends: a triple's or a pair's index. -/
def Outcome.at : Outcome α → Option ℕ
  | .triple j => some j
  | .pair j => some j
  | _ => none

end Probe

end OrthoDFA
