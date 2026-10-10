import OrthoDFA.Proofs.TallyProb
import OrthoDFA.TallyRound

/-!
# Rounds made of sub-rounds

A run over fresh draws is cut into segments: each draw continues the segment or ends it with a
kind and a state. `seg_le`: the chance of an event that, once the segment ends, becomes an event
of the remaining draws is at most the average, over the segment, of a bound on the latter.

`roundW k g` bounds a round that fails once more than `k` segments are fake, where at most `g` are
real, each fake with chance at most `c` times its ending real or good plus `δ`, and `δ'` for each
that is bad or unfinished: `round_step` is its one-segment step.
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

variable {c δ δ' : ℝ}

/-- The fake chance of a segment: `pf ≤ c (1 − pf) + δ`, and at most `1`. -/
theorem lp_le {A B pf pr pg pb : ℝ} (hc : 0 ≤ c) (hB : 0 ≤ B) (hBA : B ≤ A)
    (hf : pf ≤ c * (pg + pr) + δ) (hb : pb ≤ δ') (hr0 : 0 ≤ pr) (hg0 : 0 ≤ pg)
    (h1 : pf + pr + pg + pb ≤ 1) (hb0 : 0 ≤ pb) :
    A * pf + B * pr + pb
      ≤ min 1 ((c + δ) / (1 + c)) * A + (1 - min 1 ((c + δ) / (1 + c))) * B + δ' := by
  have hpf : pf ≤ min 1 ((c + δ) / (1 + c)) := by
    refine le_min (by linarith) ?_
    rw [le_div_iff₀ (by linarith)]; nlinarith
  have : A * pf + B * pr ≤ B + (A - B) * pf := by nlinarith
  nlinarith [mul_le_mul_of_nonneg_left hpf (sub_nonneg.2 hBA)]

theorem roundW_mono_g (hc : 0 ≤ c) (hδ : 0 ≤ δ) (hδ' : 0 ≤ δ') (k : ℕ) {g g' : ℕ}
    (h : g ≤ g') : roundW c δ δ' k g ≤ roundW c δ δ' k g' := by
  unfold roundW
  have hp0 : 0 ≤ min 1 ((c + δ) / (1 + c)) := le_min zero_le_one (by positivity)
  have h1 := binomSfGe_mono_left hp0 (min_le_left _ _) (show k + g + 1 ≤ k + g' + 1 by omega)
    (k + 1)
  have h2 : (g : ℝ) ≤ g' := by exact_mod_cast h
  nlinarith

theorem roundW_nonneg (hc : 0 ≤ c) (hδ : 0 ≤ δ) (hδ' : 0 ≤ δ') (k g : ℕ) :
    0 ≤ roundW c δ δ' k g := by
  unfold roundW
  have := binomSfGe_nonneg (n := k + g + 1) (le_min zero_le_one (by positivity) :
    0 ≤ min 1 ((c + δ) / (1 + c))) (min_le_left _ _) (k + 1)
  positivity

section Recursion

variable (c δ δ')

theorem roundW_eq (k g : ℕ) : roundW c δ δ' k g
    = binomSfGe (k + g + 1) (min 1 ((c + δ) / (1 + c))) (k + 1) + (k + g + 1) * δ' := rfl

theorem roundW_succ_succ (k g : ℕ) : roundW c δ δ' (k + 1) (g + 1)
    = min 1 ((c + δ) / (1 + c)) * roundW c δ δ' k (g + 1)
      + (1 - min 1 ((c + δ) / (1 + c))) * roundW c δ δ' (k + 1) g + δ' := by
  simp only [roundW_eq]
  rw [show k + 1 + (g + 1) + 1 = (k + g + 2) + 1 by omega, binomSfGe_succ,
    show k + (g + 1) + 1 = k + g + 2 by omega, show k + 1 + g + 1 = k + g + 2 by omega]
  push_cast; ring

theorem roundW_zero_succ (hδ' : 0 ≤ δ') (hp0 : 0 ≤ min 1 ((c + δ) / (1 + c))) (g : ℕ) :
    min 1 ((c + δ) / (1 + c)) + (1 - min 1 ((c + δ) / (1 + c))) * roundW c δ δ' 0 g + δ'
      ≤ roundW c δ δ' 0 (g + 1) := by
  set P := min 1 ((c + δ) / (1 + c))
  have hp1 : P ≤ 1 := min_le_left _ _
  have h := binomSfGe_succ (g + 1) P 0
  rw [binomSfGe_zero_right] at h
  simp only [roundW_eq, zero_add]
  have hg : (0 : ℝ) ≤ g := Nat.cast_nonneg g
  push_cast
  rw [h]
  nlinarith [mul_nonneg hp0 (mul_nonneg (by linarith : (0 : ℝ) ≤ g + 1) hδ')]

theorem roundW_succ_zero (hδ' : 0 ≤ δ') (hp0 : 0 ≤ min 1 ((c + δ) / (1 + c))) (k : ℕ) :
    min 1 ((c + δ) / (1 + c)) * roundW c δ δ' k 0 + δ' ≤ roundW c δ δ' (k + 1) 0 := by
  set P := min 1 ((c + δ) / (1 + c))
  have hp1 : P ≤ 1 := min_le_left _ _
  have h := binomSfGe_succ (k + 1) P (k + 1)
  rw [binomSfGe_top] at h
  simp only [roundW_eq, add_zero]
  have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  push_cast
  rw [h]
  nlinarith [mul_nonneg (sub_nonneg.2 hp1) (mul_nonneg (by linarith : (0 : ℝ) ≤ k + 1) hδ')]

theorem roundW_zero_zero : roundW c δ δ' 0 0 = min 1 ((c + δ) / (1 + c)) + δ' := by
  simp only [roundW_eq]
  rw [show 0 + 0 + 1 = 0 + 1 by omega, binomSfGe_succ, binomSfGe_zero_right,
    binomSfGe_zero_left]
  push_cast; ring

theorem roundW_zero_mono (hδ : 0 ≤ δ) (hc : 0 ≤ c) (g : ℕ) :
    roundW c δ δ' 0 g + δ' ≤ roundW c δ δ' 0 (g + 1) := by
  simp only [roundW_eq]
  have hp0 : 0 ≤ min 1 ((c + δ) / (1 + c)) := le_min zero_le_one (by positivity)
  have := binomSfGe_mono_left hp0 (min_le_left _ _) (show 0 + g + 1 ≤ 0 + (g + 1) + 1 by omega)
    (0 + 1)
  push_cast
  nlinarith

theorem roundW_cross (hδ : 0 ≤ δ) (hc : 0 ≤ c) (k g : ℕ) :
    roundW c δ δ' (k + 1) g ≤ roundW c δ δ' k (g + 1) := by
  simp only [roundW_eq]
  have hp0 : 0 ≤ min 1 ((c + δ) / (1 + c)) := le_min zero_le_one (by positivity)
  rw [show k + 1 + g + 1 = k + (g + 1) + 1 by omega]
  have := binomSfGe_antitone hp0 (min_le_left _ _) (n := k + (g + 1) + 1) (k + 1)
  push_cast
  nlinarith

end Recursion

/-- One segment: fake leaves `k - 1` more (or fails, at `k = 0`), real leaves `g - 1`, and the
round's bound absorbs it. -/
theorem round_step (hc : 0 ≤ c) (hδ : 0 ≤ δ) (hδ' : 0 ≤ δ') (k g : ℕ) {pf pr pg pb : ℝ}
    (hf : pf ≤ c * (pg + pr) + δ) (hb : pb ≤ δ') (h0 : 0 ≤ pf) (hr0 : 0 ≤ pr) (hg0 : 0 ≤ pg)
    (h1 : pf + pr + pg + pb ≤ 1) (hb0 : 0 ≤ pb) :
    (if k = 0 then 1 else roundW c δ δ' (k - 1) g) * pf
      + (if g = 0 then 0 else roundW c δ δ' k (g - 1)) * pr + pb ≤ roundW c δ δ' k g := by
  have hp0 : 0 ≤ min 1 ((c + δ) / (1 + c)) := le_min zero_le_one (by positivity)
  have hnn := roundW_nonneg hc hδ hδ'
  rcases k with _ | k <;> rcases g with _ | g
  · simp only [if_true, zero_mul, add_zero, roundW_zero_zero]
    have := lp_le (A := 1) (B := 0) hc le_rfl zero_le_one hf hb hr0 hg0 h1 hb0
    linarith
  · simp only [if_true, add_eq_zero, one_ne_zero, and_false, if_false, add_tsub_cancel_right]
    by_cases hBA : roundW c δ δ' 0 g ≤ 1
    · have := lp_le (A := 1) hc (hnn 0 g) hBA hf hb hr0 hg0 h1 hb0
      have := roundW_zero_succ c δ δ' hδ' hp0 g
      linarith
    · push_neg at hBA
      have := roundW_zero_mono c δ δ' hδ hc g
      nlinarith
  · simp only [add_eq_zero, one_ne_zero, and_false, if_false, if_true, add_tsub_cancel_right,
      zero_mul, add_zero]
    have := lp_le (A := roundW c δ δ' k 0) (B := 0) hc le_rfl (hnn _ _) hf hb hr0 hg0 h1 hb0
    have := roundW_succ_zero c δ δ' hδ' hp0 k
    linarith
  · simp only [add_eq_zero, one_ne_zero, and_false, if_false, add_tsub_cancel_right]
    have := lp_le hc (hnn _ _) (roundW_cross c δ δ' hδ hc k g) hf hb hr0 hg0 h1 hb0
    rw [roundW_succ_succ]
    linarith


end Bound

end OrthoDFA
