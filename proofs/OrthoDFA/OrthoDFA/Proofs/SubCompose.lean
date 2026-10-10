import OrthoDFA.Proofs.TallyProb
import OrthoDFA.TallyRound

/-!
# Rounds made of sub-rounds

A run over fresh draws is cut into segments: each draw continues the segment or ends it with a
kind and a state. `seg_le`: the chance of an event that, once the segment ends, becomes an event
of the remaining draws is at most the average, over the segment, of a bound on the latter.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

section Segment

variable {X S K : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (D : Measure X) [IsProbabilityMeasure D] (seg : S → X → S ⊕ (K × S))

theorem lintegral_pi_succ (T : ℕ) (f : (Fin (T + 1) → X) → ENNReal) :
    ∫⁻ xs, f xs ∂(Measure.pi fun _ : Fin (T + 1) => D)
      = ∫⁻ x, ∫⁻ xs, f (Fin.cons x xs) ∂(Measure.pi fun _ : Fin T => D) ∂D := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (T + 1) => D) 0
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (T + 1) => X) 0
  have hf : ∀ xs, f xs = (fun p : X × (Fin T → X) => f (Fin.cons p.1 p.2)) (e xs) := by
    intro xs
    simp only [e, piFinSuccAbove_zero', Fin.cons_self_tail]
  calc ∫⁻ xs, f xs ∂(Measure.pi fun _ : Fin (T + 1) => D)
      = ∫⁻ xs, (fun p : X × (Fin T → X) => f (Fin.cons p.1 p.2)) (e xs)
          ∂(Measure.pi fun _ : Fin (T + 1) => D) := lintegral_congr fun xs => hf xs
    _ = ∫⁻ p, f (Fin.cons p.1 p.2) ∂(D.prod (Measure.pi fun _ : Fin T => D)) :=
      (MeasurePreserving.lintegral_map_equiv (fun p : X × (Fin T → X) => f (Fin.cons p.1 p.2))
        e hmp).symm
    _ = _ := lintegral_prod _ (measurable_of_countable _).aemeasurable

/-- The bound `w` at the segment's end, or `1` where it does not end within `j` draws. -/
noncomputable def segVal (w : K → S → ℕ → ENNReal) : S → ℕ → (T : ℕ) → (Fin T → X) → ENNReal
  | _, 0, _, _ => 1
  | _, _ + 1, 0, _ => 1
  | s, j + 1, T + 1, xs => match seg s (xs 0) with
    | .inl s' => segVal w s' j T (Fin.tail xs)
    | .inr (k, s') => w k s' T

theorem seg_le (F : S → (T : ℕ) → (Fin T → X) → Prop)
    (V : K → S → (T : ℕ) → (Fin T → X) → Prop) (w : K → S → ℕ → ENNReal)
    (hF : ∀ s T x xs, F s (T + 1) (Fin.cons x xs)
      ↔ (seg s x).elim (fun s' => F s' T xs) fun p => V p.1 p.2 T xs)
    (hV : ∀ k s T, (Measure.pi fun _ : Fin T => D) {xs | V k s T xs} ≤ w k s T) :
    ∀ T s j, (Measure.pi fun _ : Fin T => D) {xs | F s T xs}
      ≤ ∫⁻ xs, segVal seg w s j T xs ∂(Measure.pi fun _ : Fin T => D) := by
  intro T
  induction T with
  | zero =>
    intro s j
    rcases j with _ | j <;> simp only [segVal, lintegral_const, measure_univ, mul_one] <;>
      exact prob_le_one
  | succ T ih =>
    intro s j
    rcases j with _ | j
    · simp only [segVal, lintegral_const, measure_univ, mul_one]; exact prob_le_one
    rw [pi_succ_apply, lintegral_pi_succ]
    refine lintegral_mono fun x => ?_
    simp only [segVal, Fin.cons_zero, Fin.tail_cons]
    rcases h : seg s x with s' | ⟨k, s'⟩
    · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | F s (T + 1) xs}}
          = {xs | F s' T xs} := by
        ext xs; simp only [Set.mem_ofPred_eq, hF, h, Sum.elim_inl, Sum.elim_inr]
      rw [this]
      exact ih s' j
    · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | F s (T + 1) xs}}
          = {xs | V k s' T xs} := by
        ext xs; simp only [Set.mem_ofPred_eq, hF, h, Sum.elim_inl, Sum.elim_inr]
      rw [this, lintegral_const, measure_univ, mul_one]
      exact hV k s' T

end Segment

section Bound

theorem binomSfGe_one : ∀ n j : ℕ, j ≤ n → binomSfGe n 1 j = 1
  | n, 0, _ => binomSfGe_zero_right n 1
  | 0, _ + 1, h => absurd h (by omega)
  | n + 1, j + 1, h => by
    rw [binomSfGe_succ, binomSfGe_one n j (by omega)]; ring

theorem binomSfGe_le_one {n : ℕ} {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (j : ℕ) :
    binomSfGe n p j ≤ 1 := by
  have := binomSfGe_antitone' hp0 hp1 (n := n) (Nat.zero_le j)
  rwa [binomSfGe_zero_right] at this

theorem binomSfGe_top {n : ℕ} {p : ℝ} : binomSfGe n p (n + 1) = 0 := by
  rw [binomSfGe_eq_range]
  exact Finset.sum_eq_zero fun i hi => by
    simp only [Finset.mem_range] at hi; rw [if_neg (by omega)]

end Bound

end OrthoDFA
