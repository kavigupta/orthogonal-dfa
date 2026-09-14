import Mathlib.MeasureTheory.Constructions.Pi

/-!
# Termination: the "eventually" bound

Each round draws fresh sampling randomness and, by findability, proposes a family
that is accept-preserving on every population with probability ≥ `p`.  The draws
are independent across rounds (an accept-preserving suffix is a property of the
target language, so "this round drew one" depends only on that round's coins).  So
the probability that all `N` rounds miss is at most `(1-p)^N` — geometric decay,
the source of "eventually with probability > 1-ε".

Modeled with a product measure over the rounds: "all miss" is the rectangle
`∏ (Gᵣ)ᶜ`, whose measure is the product of the per-round miss probabilities.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

/-- **Geometric termination.**  Over `N` independent rounds (a product measure),
if each round lands in its "good" set `Gᵣ` with probability at least `p`, then the
probability that every round misses is at most `(1-p)^N`. -/
theorem geometric_miss {N : ℕ} {α : Fin N → Type*} [∀ r, MeasurableSpace (α r)]
    (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    (G : ∀ r, Set (α r)) (hG : ∀ r, MeasurableSet (G r))
    (p : ℝ) (hp1 : p ≤ 1) (hp : ∀ r, p ≤ (ν r).real (G r)) :
    (Measure.pi ν).real {x | ∀ r, x r ∉ G r} ≤ (1 - p) ^ N := by
  have hset : {x : ∀ r, α r | ∀ r, x r ∉ G r} = Set.univ.pi (fun r => (G r)ᶜ) := by
    ext x; simp [Set.mem_pi]
  have hle1 : ∀ r, (ν r).real (G r) ≤ 1 := by
    intro r
    have h := ENNReal.toReal_mono (ENNReal.one_ne_top) (prob_le_one (μ := ν r) (s := G r))
    simpa [measureReal_def] using h
  have hfac : ∀ r, ((ν r) ((G r)ᶜ)).toReal = 1 - (ν r).real (G r) := by
    intro r
    rw [prob_compl_eq_one_sub (hG r),
      ENNReal.toReal_sub_of_le (prob_le_one (μ := ν r) (s := G r)) ENNReal.one_ne_top,
      ENNReal.toReal_one]
    rfl
  rw [hset, measureReal_def, Measure.pi_pi, ENNReal.toReal_prod]
  calc ∏ r, ((ν r) ((G r)ᶜ)).toReal
      = ∏ r, (1 - (ν r).real (G r)) := Finset.prod_congr rfl (fun r _ => hfac r)
    _ ≤ ∏ _r : Fin N, (1 - p) := by
        refine Finset.prod_le_prod (fun r _ => ?_) (fun r _ => by linarith [hp r])
        linarith [hle1 r]
    _ = (1 - p) ^ N := by rw [Finset.prod_const, Finset.card_univ, Fintype.card_fin]

#print axioms geometric_miss

end OrthoDFA
