import OrthoDFA.Clustering
import Mathlib.Probability.Independence.Basic

/-!
# The count of a batch is binomial

Over independent draws, the number of a batch's draws satisfying a predicate of chance `p`
reaches `j` with chance exactly `binomSfGe n p j`, and that tail grows with `p`.  So an exact
binomial test against a rate `θ` errs, at any one look, with at most its own failure chance.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem binomSfGe_eq_range (n : ℕ) (p : ℝ) (j : ℕ) :
    binomSfGe n p j = ∑ i ∈ Finset.range (n + 1),
      if j ≤ i then (n.choose i : ℝ) * p ^ i * (1 - p) ^ (n - i) else 0 := by
  unfold binomSfGe
  rw [← Finset.sum_filter]
  congr 1
  ext i
  simp only [Finset.mem_Icc, Finset.mem_filter, Finset.mem_range]
  omega

theorem binomSfGe_zero_right (n : ℕ) (p : ℝ) : binomSfGe n p 0 = 1 := by
  rw [binomSfGe_eq_range]
  simp only [zero_le, if_true]
  have h := (add_pow p (1 - p) n).symm
  rw [add_sub_cancel, one_pow] at h
  exact (Finset.sum_congr rfl fun i _ => by ring).trans h

theorem binomSfGe_zero_left (p : ℝ) (j : ℕ) : binomSfGe 0 p (j + 1) = 0 := by
  rw [binomSfGe_eq_range]
  simp

theorem binomSfGe_succ (n : ℕ) (p : ℝ) (j : ℕ) :
    binomSfGe (n + 1) p (j + 1) = p * binomSfGe n p j + (1 - p) * binomSfGe n p (j + 1) := by
  set q := 1 - p
  set a : ℕ → ℝ := fun i => if j ≤ i then (n.choose i : ℝ) * p ^ i * q ^ (n - i) else 0 with ha
  set b : ℕ → ℝ := fun i => if j + 1 ≤ i then (n.choose i : ℝ) * p ^ i * q ^ (n - i) else 0
    with hb
  set f : ℕ → ℝ := fun i =>
    if j + 1 ≤ i then ((n + 1).choose i : ℝ) * p ^ i * q ^ (n + 1 - i) else 0 with hf
  have hterm : ∀ i, f (i + 1) = p * a i + q * b (i + 1) := by
    intro i
    simp only [hf, ha, hb, Nat.choose_succ_succ', Nat.cast_add]
    by_cases hji : j ≤ i
    · have h1 : j + 1 ≤ i + 1 := by omega
      simp only [hji, h1, if_true]
      rcases Nat.lt_or_ge i n with hin | hin
      · have he : n + 1 - (i + 1) = n - i := by omega
        have he2 : n - i = n - (i + 1) + 1 := by omega
        rw [he, he2, pow_succ, pow_succ]
        ring
      · rw [Nat.choose_eq_zero_of_lt (show n < i + 1 by omega)]
        have he : n + 1 - (i + 1) = n - i := by omega
        rw [he]
        ring
    · have h1 : ¬ j + 1 ≤ i + 1 := by omega
      simp [hji, h1]
  have hb0 : b 0 = 0 := by simp [hb]
  have hbn : b (n + 1) = 0 := by simp [hb, Nat.choose_eq_zero_of_lt (Nat.lt_succ_self n)]
  have hf0 : f 0 = 0 := by simp [hf]
  have hsumb : ∑ i ∈ Finset.range (n + 1), b (i + 1) = ∑ i ∈ Finset.range (n + 1), b i := by
    have h1 := Finset.sum_range_succ' b (n + 1)
    rw [Finset.sum_range_succ, hbn, hb0] at h1
    simpa using h1.symm
  simp only [binomSfGe_eq_range]
  change ∑ i ∈ Finset.range (n + 1 + 1), f i
    = p * ∑ i ∈ Finset.range (n + 1), a i + q * ∑ i ∈ Finset.range (n + 1), b i
  rw [Finset.sum_range_succ', hf0, add_zero, Finset.sum_congr rfl fun i _ => hterm i,
    Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsumb]

theorem binomSfGe_nonneg {n : ℕ} {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (j : ℕ) :
    0 ≤ binomSfGe n p j := by
  unfold binomSfGe
  exact Finset.sum_nonneg fun i _ => by
    have : 0 ≤ 1 - p := by linarith
    positivity

theorem binomSfGe_antitone {n : ℕ} {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (j : ℕ) :
    binomSfGe n p (j + 1) ≤ binomSfGe n p j := by
  simp only [binomSfGe_eq_range]
  refine Finset.sum_le_sum fun i _ => ?_
  have : 0 ≤ 1 - p := by linarith
  split_ifs <;> first | exact le_rfl | positivity | omega

theorem binomSfGe_antitone' {n : ℕ} {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) {j j' : ℕ}
    (h : j ≤ j') : binomSfGe n p j' ≤ binomSfGe n p j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact (binomSfGe_antitone hp0 hp1 _).trans ih

theorem binomSfGe_mono {p θ : ℝ} (hp0 : 0 ≤ p) (hθ1 : θ ≤ 1) (hpθ : p ≤ θ) :
    ∀ n j, binomSfGe n p j ≤ binomSfGe n θ j
  | n, 0 => by rw [binomSfGe_zero_right, binomSfGe_zero_right]
  | 0, j + 1 => by rw [binomSfGe_zero_left, binomSfGe_zero_left]
  | n + 1, j + 1 => by
    have hp1 : p ≤ 1 := hpθ.trans hθ1
    have hθ0 : 0 ≤ θ := hp0.trans hpθ
    rw [binomSfGe_succ, binomSfGe_succ]
    have h1 := binomSfGe_mono hp0 hθ1 hpθ n j
    have h2 := binomSfGe_mono hp0 hθ1 hpθ n (j + 1)
    have h3 := binomSfGe_antitone hp0 hp1 (n := n) j
    have : 0 ≤ 1 - θ := by linarith
    nlinarith

section Law

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

omit [Countable X] [MeasurableSingletonClass X] in
theorem card_filter_subtype {N : ℕ} (S : Finset (Fin N)) (q : Fin N → Prop) [DecidablePred q] :
    (Finset.univ.filter fun i : S => q i).card = (S.filter q).card := by
  rw [Finset.univ_eq_attach, Finset.filter_attach, Finset.card_map, Finset.card_attach]

open scoped Classical in
/-- Over independent draws, how many of the coordinates in `S` satisfy `P` is binomial. -/
theorem pi_count_ge (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) {N : ℕ}
    (S : Finset (Fin N)) (j : ℕ) :
    (Measure.pi fun _ : Fin N => D).real {b | j ≤ (S.filter fun i => P (b i)).card}
      = binomSfGe S.card (D.real {x | P x}) j := by
  set ν := Measure.pi fun _ : Fin N => D
  have hmeas : ∀ s : Set (Fin N → X), MeasurableSet s := fun s => (Set.to_countable s).measurableSet
  have hmarg : ∀ (i : Fin N) (A : Set X), ν.real ((fun b => b i) ⁻¹' A) = D.real A := by
    intro i A
    rw [measureReal_def, measureReal_def, ← Measure.map_apply (measurable_pi_apply i)
      (Set.to_countable A).measurableSet, (measurePreserving_eval _ i).map_eq]
  have hind : iIndepFun (fun i (b : Fin N → X) => b i) ν :=
    iIndepFun_pi (X := fun _ (x : X) => x) fun _ => measurable_id.aemeasurable
  induction S using Finset.induction_on generalizing j with
  | empty =>
    rcases j with _ | j
    · simp [binomSfGe_zero_right]
    · simp [binomSfGe_zero_left]
  | insert a S ha ih =>
    rw [Finset.card_insert_of_notMem ha]
    rcases j with _ | j
    · simp [binomSfGe_zero_right]
    have hS : ∀ b : Fin N → X, ((insert a S).filter fun i => P (b i)).card
        = (if P (b a) then 1 else 0) + (S.filter fun i => P (b i)).card := by
      intro b
      rw [Finset.filter_insert]
      split_ifs with h
      · rw [Finset.card_insert_of_notMem (fun hm => ha (Finset.mem_filter.1 hm).1)]
        ring
      · simp
    have h2 := hind.indepFun_finset {a} S (by simpa using ha) (fun i => measurable_pi_apply i)
    have h3 : IndepFun (fun b : Fin N → X => b a) (fun b (i : S) => b i) ν :=
      h2.comp (measurable_pi_apply (⟨a, Finset.mem_singleton_self a⟩ : ({a} : Finset _)))
        measurable_id
    have hcount : ∀ k, {b : Fin N → X | k ≤ (S.filter fun i => P (b i)).card}
        = (fun b (i : S) => b i) ⁻¹' {g | k ≤ (Finset.univ.filter fun i : S => P (g i)).card} := by
      intro k
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_preimage]
      rw [card_filter_subtype S (fun i => P (b i))]
    have hprod : ∀ (A : Set X) k, ν.real ((fun b => b a) ⁻¹' A
        ∩ {b | k ≤ (S.filter fun i => P (b i)).card})
        = D.real A * binomSfGe S.card (D.real {x | P x}) k := by
      intro A k
      rw [hcount, measureReal_def, h3.measure_inter_preimage_eq_mul _ _
        (Set.to_countable A).measurableSet (Set.to_countable _).measurableSet,
        ENNReal.toReal_mul, ← measureReal_def, ← measureReal_def, hmarg, ← hcount, ih]
    have hsplit : {b : Fin N → X | j + 1 ≤ ((insert a S).filter fun i => P (b i)).card}
        = ((fun b => b a) ⁻¹' {x | P x} ∩ {b | j ≤ (S.filter fun i => P (b i)).card})
          ∪ ((fun b => b a) ⁻¹' {x | ¬ P x}
            ∩ {b | j + 1 ≤ (S.filter fun i => P (b i)).card}) := by
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_union, Set.mem_inter_iff, Set.mem_preimage, hS]
      by_cases h : P (b a) <;> simp [h] <;> omega
    rw [hsplit, measureReal_union (Set.disjoint_left.2 fun b h1 h2 => h2.1 h1.1) (hmeas _),
      hprod, hprod, binomSfGe_succ]
    have hc : D.real {x | ¬ P x} = 1 - D.real {x | P x} := by
      rw [show {x | ¬ P x} = {x | P x}ᶜ from rfl,
        measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
    rw [hc]

end Law

end OrthoDFA
