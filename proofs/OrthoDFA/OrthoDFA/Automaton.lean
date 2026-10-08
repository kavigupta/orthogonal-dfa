import OrthoDFA.Clustering

/-!
# The target as a DFA
-/

namespace OrthoDFA

variable {S : Type*} [Stringlike S]

/-- A deterministic automaton over the string monoid, acting on the right. -/
structure DFA (S : Type*) [Monoid S] (Q : Type*) where
  step : Q → S → Q
  step_one : ∀ q, step q 1 = q
  step_mul : ∀ q a b, step q (a * b) = step (step q a) b
  start : Q
  accept : Set Q

variable {Q : Type*}

/-- The state a prefix reaches. -/
def DFA.state (A : DFA S Q) (p : S) : Q := A.step A.start p

end OrthoDFA
