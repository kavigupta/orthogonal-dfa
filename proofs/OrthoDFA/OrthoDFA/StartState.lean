import OrthoDFA.Pass

/-!
# A start state that works

The round's gate judges where a hypothesis walks from position `k`; where to start it is a separate
question.  `StartExists`: a DFA that agrees with the target on a set `S` of target states, read
through a map `h`, works from `h q` as well as the target re-rooted at `q` does on the draws whose
re-rooted run stays in `S`.  So if some `q` has that share at least `1 − η`, the all-starts
certificate has a start of error at most `η` to find.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Q P : Type*}

/-- `H` agrees with the target `A` on the states `S`, each read as `H`'s state `h q`: a step by a
letter from a state of `S` to a state of `S` is `H`'s step, and acceptance is `H`'s. -/
def MatchesOn (A : DFA (FreeMonoid α) Q) (H : DFA (FreeMonoid α) P) (S : Set Q) (h : Q → P) :
    Prop :=
  (∀ q ∈ S, ∀ c : α, A.step q (FreeMonoid.of c) ∈ S →
      H.step (h q) (FreeMonoid.of c) = h (A.step q (FreeMonoid.of c)))
    ∧ ∀ q ∈ S, (h q ∈ H.accept ↔ q ∈ A.accept)

/-- The target's run from `q` along `x` stays in `S`. -/
def StaysIn (A : DFA (FreeMonoid α) Q) (S : Set Q) (q : Q) (x : FreeMonoid α) : Prop :=
  ∀ i ≤ x.toList.length, A.step q (prefixOf x i) ∈ S

/-- `StartExists`: if `H` agrees with the target on `S`, then started at `h q` it misjudges at most
`η` of `D`'s draws whenever the target re-rooted at `q` both stays in `S` and agrees with the
target on at least `1 − η` of them. -/
def StartExists : Prop :=
  ∀ {α : Type*} [Fintype α] [DecidableEq α] {Q P : Type*} (A : DFA (FreeMonoid α) Q)
    (H : DFA (FreeMonoid α) P) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (S : Set Q)
    (h : Q → P) (q : Q) (η : ℝ),
    MatchesOn A H S h →
    1 - η ≤ D.real {x | StaysIn A S q x ∧ (A.step q x ∈ A.accept ↔ A.state x ∈ A.accept)} →
    D.real {x | ¬ (H.step (h q) x ∈ H.accept ↔ A.state x ∈ A.accept)} ≤ η

end OrthoDFA
