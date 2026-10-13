import OrthoDFA.Level2Claim
import OrthoDFA.Proofs.DTree
import OrthoDFA.Proofs.HypothesisDFA

/-!
# The tree under ideal reads

With reads fixed by the target state, so is the leaf a string sifts to. The tree's own lemmas are
main's, in `Proofs/DTree.lean`.
-/

namespace OrthoDFA

variable {α : Type*} (read : FreeMonoid α → ARU)

namespace Ideal

/-- Under ideal reads a read is fixed by the target state. -/
theorem read_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hI : IdealReads read M side)
    {z z' : FreeMonoid α} (h : M.eval z.toList = M.eval z'.toList) : read z = read z' := by
  rcases hI (M.eval z.toList) with hq | hq
  · rw [hq z rfl, hq z' h.symm]
  · rw [hq z rfl, hq z' h.symm]

theorem read_mul_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hI : IdealReads read M side)
    {z z' : FreeMonoid α} (h : M.eval z.toList = M.eval z'.toList) (m : FreeMonoid α) :
    read (z * m) = read (z' * m) := by
  apply read_congr read hI
  simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
  exact congrArg (M.evalFrom · m.toList) h

end Ideal

end OrthoDFA
