import OrthoDFA.Code.HypothesisDFA

/-! How a hypothesis DFA, run from a leaf, classifies a string. -/

noncomputable section

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- From the leaf `s`, `H` follows learned edges along all of `x` to a leaf, which is on the root's
accept side exactly when `b`. -/
def HypothesisDFA.Classifies (H : HypothesisDFA α) (s : List Bool) (x : FreeMonoid α) (b : Prop) :
    Prop :=
  ∃ p, walkEnd H.edges s x.toList = some p ∧ (p.head? = some true ↔ b)

end OrthoDFA

end
