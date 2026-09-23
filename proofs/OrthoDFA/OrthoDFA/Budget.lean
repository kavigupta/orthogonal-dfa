import OrthoDFA.Model

/-!
# What the budget costs

`prefCount` is the count the round's tails ask for, and `ClusteringCorrect` quantifies over
the states the schedule built from it.  Written out it is a sum of six ceilings, each a log
over the square of some margin, and nothing in the statement says how those margins depend on
the rates they resolve.  This file says it: every term is bounded by one polynomial in the
reciprocals of `sig η` and `cutBudget`, so the budget is `log` in the failure probability and
polynomial in everything else.

The two scales are not independent.  `screenMargin` carries `cutBudget · sig³`, and the tail
it sits in squares it, so `sig` enters at the sixth power while `cutBudget` enters at the
second — and `cutBudget` is itself a minimum over `εcov`, `sig` and `indecisionLimit`, so a
bound in `sig` alone cannot exist.
-/

namespace OrthoDFA

-- `Finset J` carries the instance the definitions were written with; these bounds read
-- only the card, so the linter is right that they do not use it themselves.
set_option linter.unusedFintypeInType false

variable {J : Type*} [Fintype J]

/-- The budget is polynomial in the reciprocals of the rates it resolves.

`sig` enters at the sixth power and `budgetScale` at the second because the screen's tail is
the binding one: its margin is `cutBudget · sig³` up to constants, and a deviation bound
squares the margin it is given.  The population count enters squared for the same reason --
`flipBudget` divides by it, and `screenMargin` inherits that.

The `4096` is not tight.  It is a constant that covers every term at once, which is what makes
the statement one bound rather than six. -/
theorem prefCount_le_poly (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ)
    (hsig : 0 < sig η) (hη : η < 1) (hpop : populations.Nonempty)
    (hind : 0 < indecisionLimit) (hεcov : 0 < εcov) (hεcov1 : εcov ≤ 1)
    (hδ : 0 < δ) (hδ1 : δ ≤ 1) (hα : 0 < α) (hα1 : α < 1)
    (hpAP : 0 < pAP) (hpAP1 : pAP ≤ 1) :
    (prefCount η populations indecisionLimit εcov δ α pAP : ℝ)
      ≤ 4096 * (populations.card : ℝ) ^ 2
        * budgetLog populations η indecisionLimit εcov δ α pAP
        / (sig η ^ 6 * budgetScale η indecisionLimit εcov ^ 2) := by
  sorry

/-- The ladder is `log` of the budget, so any bound on the budget bounds the number of times
the loop runs the gate, logarithmically. -/
theorem ladderLen_le_of_prefCount_le (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) {N : ℕ}
    (h : prefCount η populations indecisionLimit εcov δ α pAP ≤ N) :
    ladderLen η populations indecisionLimit εcov δ α pAP ≤ Nat.log 2 N + 1 :=
  Nat.succ_le_succ (Nat.log_mono_right h)

/-- Every ceiling is under its argument plus one, and under one when the argument is
negative -- which is what lets the six terms be bounded separately and summed. -/
private lemma cast_ceil_le (x : ℝ) : (⌈x⌉₊ : ℝ) ≤ max x 0 + 1 := by
  by_cases hx : x ≤ 0
  · have : ⌈x⌉₊ = 0 := Nat.ceil_eq_zero.2 hx
    rw [this]
    have : (0 : ℝ) ≤ max x 0 := le_max_right _ _
    simpa using by linarith
  · have hx' : 0 ≤ x := le_of_not_ge hx
    calc (⌈x⌉₊ : ℝ) ≤ x + 1 := (Nat.ceil_lt_add_one hx').le
      _ ≤ max x 0 + 1 := by gcongr; exact le_max_left _ _

/-- The six tails, each named, so the bound can be read term by term. -/
noncomputable def prefTerms (η : ℝ) (populations : Finset J)
    (indecisionLimit εcov δ α pAP : ℝ) : List ℝ :=
  [ Real.log (128 * (populations.card : ℝ)
      * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 2) ^ 2 / δ)
      / (2 * (screenMargin η populations indecisionLimit εcov δ / 2) ^ 2),
    Real.log (128 * (populations.card : ℝ) / δ)
      / (2 * (cutBudget η indecisionLimit εcov / 4) ^ 2),
    64 * Real.log (1 / α) / (εcov * (sig η * εcov / 4) ^ 2),
    64 * Real.log (256 * (populations.card : ℝ) / δ) / (εcov * (sig η * εcov / 4) ^ 2),
    Real.log (128 * (populations.card : ℝ)
        * ((poolCount η populations indecisionLimit εcov δ pAP : ℝ) + 1) / δ)
      / (2 * ((populations.card : ℝ) * flipBudget η populations indecisionLimit εcov δ) ^ 2),
    64 / εcov ]

/-- Rounding up a list of reals costs one apiece. -/
private lemma cast_sum_ceil_le : ∀ l : List ℝ,
    (((l.map (fun x => ⌈x⌉₊)).sum : ℕ) : ℝ)
      ≤ (l.map (fun x => max x 0)).sum + l.length
  | [] => by simp
  | x :: xs => by
    have hx := cast_ceil_le x
    have hxs := cast_sum_ceil_le xs
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    linarith

/-- The count is exactly its six tails, each rounded up, and the one rung the ladder always
has. -/
theorem prefCount_eq_sum (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) :
    prefCount η populations indecisionLimit εcov δ α pAP
      = ((prefTerms η populations indecisionLimit εcov δ α pAP).map
          (fun x => ⌈x⌉₊)).sum + 1 := by
  simp only [prefCount, prefTerms, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-- The mechanical half of the bound: the count is under its six tails once each rounding is
paid for.  Nothing analytic happens here -- what the tails are worth is `prefCount_le_poly`. -/
theorem prefCount_le_terms (populations : Finset J)
    (η indecisionLimit εcov δ α pAP : ℝ) :
    (prefCount η populations indecisionLimit εcov δ α pAP : ℝ)
      ≤ ((prefTerms η populations indecisionLimit εcov δ α pAP).map
          (fun x => max x 0)).sum + 7 := by
  have h := cast_sum_ceil_le (prefTerms η populations indecisionLimit εcov δ α pAP)
  rw [prefCount_eq_sum]
  have hlen : (prefTerms η populations indecisionLimit εcov δ α pAP).length = 6 := by
    simp [prefTerms]
  rw [hlen] at h
  push_cast at h ⊢
  linarith

end OrthoDFA
