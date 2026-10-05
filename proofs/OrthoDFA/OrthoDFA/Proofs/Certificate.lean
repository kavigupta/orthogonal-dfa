import OrthoDFA.Certificate

/-!
# The certificate is sound

Past some look fixed by `|R|`, `s`, `α` and `e` alone, the intervals are too narrow for the
certificate to go on.  Before it, each look's intervals all hold but for `lookLevel α k`, and
while they hold, a bound of at most `e` puts `H`'s reads apart.  A read is fresh unless its
string was drawn before or its noise was pinned.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem certifies_sound (R : Type*) [Fintype R] (s α e : ℝ) (hs : 0 < s) (hα : 0 < α)
    (he : 0 < e) :
    ∃ n : ℕ, ∀ {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
      {S : Type*} [Stringlike S] (O : Oracle μ S) (H : DFA S R) (Dsamp : Measure S)
      [IsProbabilityMeasure Dsamp] (U : Finset S) (b : S → ℝ) (κ : ℝ),
      (∀ a, Dsamp.real {a} ≤ κ) →
      ((μ[|pinned O U b]).prod (Measure.infinitePi fun _ : ℕ => Dsamp)).real
          {q | certifies O H s α e q.2 q.1
            ∧ e ≤ Dsamp.real {x | H.state x ∈ H.accept}
              * (1 - Dsamp.real {x | H.state x ∈ H.accept})
            ∧ ¬ readsApart O H Dsamp s e}
        ≤ α + n * (n + U.card) * κ := by
  sorry

end OrthoDFA
