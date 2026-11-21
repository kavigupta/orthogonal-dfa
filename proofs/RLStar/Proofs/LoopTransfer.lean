import RLStar.Proofs.LoopRuns
import RLStar.Proofs.LoopLaw
import RLStar.Proofs.LoopRounds

/-!
# A round's end, and the avoiding probes

`LooseEndsWell` is what the loop needs of a round's end. The round draws its probes from `D`
conditioned on avoiding a set `K₀` of first `k` letters, which removes `m` of `D`; a claim of at
most half of a set at the avoiding probes is one at all of `D` up to `m`
(`real_inter_le_of_avoiding`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A round's end is good: a consistent round has at most `ε` of `D` in `disagree`, and a harvest's
list puts at most `A·T + sl` on the states read undecided at most `T` of the time. -/
def LooseEndsWell {σ : Type*} (M : _root_.DFA α σ) (U : σ → ℝ) (A sl : ℝ)
    (D : Measure (FreeMonoid α)) (ε : ℝ) (disagree : Set (FreeMonoid α)) : LazyEnd α → Prop
  | .consistent => D.real disagree ≤ ε
  | .harvest _ ℓ => ∀ T, 0 ≤ T → (listLaw ℓ).real {z | U (M.eval z.toList) ≤ T} ≤ A * T + sl
  | .failed => False

/-- The probes the round may draw. -/
def avoiding (k : ℕ) (K₀ : Set (FreeMonoid α)) : Set (FreeMonoid α) := {x | Avoids k K₀ x}

variable {k : ℕ} {K₀ : Set (FreeMonoid α)}

omit [DecidableEq α] in
/-- At most half of `E` and `u` more in `S'` at the avoiding probes, `S'` holding every avoiding
probe of `S`, is at most half of `E` and `u + m` more in `S` at all of `D`. -/
theorem real_inter_le_of_avoiding (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    {E S S' : Set (FreeMonoid α)} (hS : ∀ x ∈ avoiding k K₀, x ∈ S → x ∈ S') {u : ℝ} (hu : 0 ≤ u)
    (hm : D.real (avoiding k K₀)ᶜ ≤ 1 / 2)
    (h : (popLaw (D.restrict (avoiding k K₀))).real (E ∩ S')
      ≤ (popLaw (D.restrict (avoiding k K₀))).real E / 2 + u) :
    D.real (E ∩ S) ≤ D.real E / 2 + (u + D.real (avoiding k K₀)ᶜ) := by
  classical
  set A := avoiding (α := α) k K₀
  set m := D.real Aᶜ with hmd
  have hA : D.real A = 1 - m := by
    rw [hmd, probReal_compl_eq_one_sub (measurableSet_of_countable A)]; ring
  have hApos : 0 < D.real A := by rw [hA]; linarith
  have hA0 : D.restrict A Set.univ ≠ 0 := by
    rw [Measure.restrict_apply_univ]
    intro h0
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hApos
    exact lt_irrefl _ hApos
  have hAtop : D.restrict A Set.univ ≠ ⊤ := measure_ne_top _ _
  have hD' : ∀ X, (popLaw (D.restrict A)).real X = D.real (X ∩ A) / D.real A := by
    intro X
    unfold popLaw
    rw [if_neg (by tauto)]
    simp only [measureReal_def, Measure.smul_apply, smul_eq_mul, ENNReal.toReal_mul,
      ENNReal.toReal_inv, Measure.restrict_apply_univ,
      Measure.restrict_apply (measurableSet_of_countable X)]
    rw [div_eq_inv_mul]
  rw [hD', hD', div_le_iff₀ hApos, add_mul] at h
  have h1 : D.real (E ∩ S) ≤ D.real (E ∩ S' ∩ A) + m := by
    have hsplit := measureReal_inter_add_sdiff₀ (μ := D) (s := E ∩ S)
      (measurableSet_of_countable A).nullMeasurableSet (measure_ne_top _ _)
    have hinter : D.real (E ∩ S ∩ A) ≤ D.real (E ∩ S' ∩ A) :=
      measureReal_mono (fun x ⟨⟨h1, h2⟩, h3⟩ => ⟨⟨h1, hS x h3 h2⟩, h3⟩) (measure_ne_top _ _)
    have hdiff : D.real ((E ∩ S) \ A) ≤ m :=
      measureReal_mono (fun x hx => hx.2) (measure_ne_top _ _)
    linarith
  have h2 : D.real (E ∩ A) ≤ D.real E :=
    measureReal_mono Set.inter_subset_left (measure_ne_top _ _)
  have h3 : D.real A ≤ 1 := measureReal_le_one
  have h5 : D.real (E ∩ A) / D.real A / 2 * D.real A = D.real (E ∩ A) / 2 := by
    field_simp
  nlinarith [mul_le_mul_of_nonneg_left h3 hu]

end OrthoDFA
