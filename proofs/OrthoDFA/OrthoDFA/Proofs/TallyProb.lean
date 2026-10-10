import OrthoDFA.Proofs.FixTime
import OrthoDFA.Proofs.Freedman

/-!
# Hits along a round of fresh draws

`hits_le`: where each state's hit set has chance at most `ρ`, the hits along `T` draws reach `j`
with chance at most `P(Bin(T, ρ) ≥ j)`, however the states are chosen. `somewhere_le`: an event
of chance at most `c` from every state happens at some point of `T` draws with chance at most
`T · c`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {X S : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

section Defs

variable (step : S → X → Option S)

open scoped Classical in
/-- How many of the draws `xs` from `s` fall in the hit set of the state they are drawn at. -/
noncomputable def hitsAlong (A : S → Set X) : S → (T : ℕ) → (Fin T → X) → ℕ
  | _, 0, _ => 0
  | s, _ + 1, xs => (if xs 0 ∈ A s then 1 else 0) + match step s (xs 0) with
    | some s' => hitsAlong A s' _ (Fin.tail xs)
    | none => 0

/-- At some point of the draws `xs` from `s`, the round is at a state from which `F` holds of the
remaining draws. -/
def Somewhere (F : S → (T : ℕ) → (Fin T → X) → Prop) : S → (T : ℕ) → (Fin T → X) → Prop
  | _, 0, _ => False
  | s, _ + 1, xs => F s _ xs ∨ ∃ s', step s (xs 0) = some s' ∧ Somewhere F s' _ (Fin.tail xs)

end Defs

variable (D : Measure X) [IsProbabilityMeasure D] (step : S → X → Option S)

open scoped Classical in
theorem hits_le (A : S → Set X) {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hA : ∀ s, D.real (A s) ≤ ρ) :
    ∀ (T : ℕ) (s : S) (j : ℕ), (Measure.pi fun _ : Fin T => D) {xs | j ≤ hitsAlong step A s T xs}
      ≤ ENNReal.ofReal (binomSfGe T ρ j) := by
  intro T
  induction T with
  | zero =>
    intro s j
    rcases j with _ | j
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    · simp [hitsAlong]
  | succ T ih =>
    intro s j
    rcases j with _ | j
    · rw [binomSfGe_zero_right, ENNReal.ofReal_one]; exact prob_le_one
    set c₁ := binomSfGe T ρ j
    set c₀ := binomSfGe T ρ (j + 1)
    have hc₀ : 0 ≤ c₀ := binomSfGe_nonneg hρ0 hρ1 _
    have hc₀₁ : c₀ ≤ c₁ := binomSfGe_antitone hρ0 hρ1 j
    set d := D.real (A s)
    have hd : d ≤ ρ := hA s
    have hd0 : 0 ≤ d := measureReal_nonneg
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D)
        {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | j + 1 ≤ hitsAlong step A s (T + 1) xs}}
        ≤ (A s).indicator (fun _ => ENNReal.ofReal c₁) x
          + (A s)ᶜ.indicator (fun _ => ENNReal.ofReal c₀) x := by
      intro x
      by_cases hxA : x ∈ A s
      · rw [Set.indicator_of_mem hxA, Set.indicator_of_notMem (by simpa using hxA), add_zero]
        rcases hx : step s x with _ | s'
        · rcases j with _ | j
          · exact prob_le_one.trans (by
              rw [show c₁ = 1 from binomSfGe_zero_right T ρ, ENNReal.ofReal_one])
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X |
                j + 1 + 1 ≤ hitsAlong step A s (T + 1) xs}} = ∅ := by
              ext xs; simp [hitsAlong, hx, hxA]
            rw [this, measure_empty]; exact zero_le
        · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X |
              j + 1 ≤ hitsAlong step A s (T + 1) xs}} = {xs | j ≤ hitsAlong step A s' T xs} := by
            ext xs; simp [hitsAlong, hx, hxA]; omega
          rw [this]; exact ih s' j
      · rw [Set.indicator_of_notMem hxA, Set.indicator_of_mem (by simpa using hxA), zero_add]
        rcases hx : step s x with _ | s'
        · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X |
              j + 1 ≤ hitsAlong step A s (T + 1) xs}} = ∅ := by
            ext xs; simp [hitsAlong, hx, hxA]
          rw [this, measure_empty]; exact zero_le
        · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X |
              j + 1 ≤ hitsAlong step A s (T + 1) xs}}
              = {xs | j + 1 ≤ hitsAlong step A s' T xs} := by
            ext xs; simp [hitsAlong, hx, hxA]
          rw [this]; exact ih s' (j + 1)
    rw [pi_succ_apply]
    refine (lintegral_mono hsec).trans ?_
    have hc₁ : 0 ≤ c₁ := hc₀.trans hc₀₁
    rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
        (Set.to_countable _).measurableSet, lintegral_indicator (Set.to_countable _).measurableSet,
      setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
      ← ENNReal.ofReal_mul hc₁, ← ENNReal.ofReal_mul hc₀,
      ← ENNReal.ofReal_add (mul_nonneg hc₁ hd0)
        (mul_nonneg hc₀ (by simp only [d] at hd ⊢; linarith))]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [binomSfGe_succ]
    change c₁ * d + c₀ * (1 - d) ≤ ρ * c₁ + (1 - ρ) * c₀
    nlinarith

theorem somewhere_le (F : S → (T : ℕ) → (Fin T → X) → Prop) (c : ENNReal) (T₀ : ℕ)
    (hF : ∀ s T, T ≤ T₀ → (Measure.pi fun _ : Fin T => D) {xs | F s T xs} ≤ c) :
    ∀ (T : ℕ), T ≤ T₀ → ∀ s : S,
      (Measure.pi fun _ : Fin T => D) {xs | Somewhere step F s T xs} ≤ T * c := by
  intro T
  induction T with
  | zero => intro _ s; simp [Somewhere]
  | succ T ih =>
    intro hT s
    have h2 : (Measure.pi fun _ : Fin (T + 1) => D)
        {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
          ∧ Somewhere step F s' T (Fin.tail xs)} ≤ T * c := by
      rw [pi_succ_apply]
      calc ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
              {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
                ∧ Somewhere step F s' T (Fin.tail xs)}} ∂D
          ≤ ∫⁻ _, (T : ENNReal) * c ∂D := by
            refine lintegral_mono fun x => ?_
            rcases hx : step s x with _ | s'
            · simp [hx]
            · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
                  ∧ Somewhere step F s' T (Fin.tail xs)}}
                  = {xs | Somewhere step F s' T xs} := by
                ext xs; simp [hx]
              rw [this]
              exact ih (by omega) s'
        _ = T * c := by rw [lintegral_const, measure_univ, mul_one]
    have hsplit : {xs | Somewhere step F s (T + 1) xs}
        = {xs | F s (T + 1) xs}
          ∪ {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
            ∧ Somewhere step F s' T (Fin.tail xs)} := by
      ext xs; simp [Somewhere]
    rw [hsplit]
    refine (measure_union_le _ _).trans ?_
    calc _ ≤ c + T * c := add_le_add (hF s (T + 1) hT) h2
      _ = (T + 1 : ℕ) * c := by push_cast; ring

section Tests

open scoped Classical in
/-- An exact binomial test against `θ`, on the first `j + 1` of `T` fresh draws, fires where the
rate is at most `θ` with chance below `a`. -/
theorem binom_test_le (A : Set X) {θ a : ℝ} (hθ1 : θ ≤ 1) (hA : D.real A ≤ θ) {T j : ℕ}
    (hj : j < T) :
    (Measure.pi fun _ : Fin T => D)
        {xs | binomSfGe (j + 1) θ (∑ i : Fin T, if (i : ℕ) ≤ j ∧ xs i ∈ A then 1 else 0) < a}
      ≤ ENNReal.ofReal a := by
  by_cases hex : ∃ h, binomSfGe (j + 1) θ h < a
  swap
  · push_neg at hex
    have : {xs : Fin T → X | binomSfGe (j + 1) θ (∑ i : Fin T,
        if (i : ℕ) ≤ j ∧ xs i ∈ A then 1 else 0) < a} = ∅ := by
      ext xs; simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_lt]
      exact hex _
    rw [this, measure_empty]; exact zero_le
  set S := Finset.univ.filter fun i : Fin T => (i : ℕ) ≤ j
  have hS : S.card = j + 1 := by
    rw [show S = (Finset.range (j + 1)).attachFin (fun i hi => by
      simp only [Finset.mem_range] at hi; omega) from by
        ext i; simp [S, Finset.mem_attachFin, Nat.lt_succ_iff]]
    simp
  have hcnt : ∀ xs : Fin T → X, (∑ i : Fin T, if (i : ℕ) ≤ j ∧ xs i ∈ A then 1 else 0)
      = (S.filter fun i => xs i ∈ A).card := by
    intro xs
    rw [Finset.card_eq_sum_ones, Finset.sum_filter, Finset.sum_filter]
    exact Finset.sum_congr rfl fun i _ => by split_ifs <;> simp_all
  have hsub : {xs : Fin T → X | binomSfGe (j + 1) θ (∑ i : Fin T,
      if (i : ℕ) ≤ j ∧ xs i ∈ A then 1 else 0) < a}
      ⊆ {xs | Nat.find hex ≤ (S.filter fun i => xs i ∈ A).card} := by
    intro xs hxs
    simp only [Set.mem_ofPred_eq, hcnt] at hxs ⊢
    exact Nat.find_min' hex hxs
  refine (measure_mono hsub).trans ?_
  have hreal := pi_count_ge D (fun x => x ∈ A) S (Nat.find hex)
  rw [hS] at hreal
  rw [← ofReal_measureReal, hreal]
  refine ENNReal.ofReal_le_ofReal ((binomSfGe_mono measureReal_nonneg hθ1 hA _ _).trans ?_)
  exact (Nat.find_spec hex).le

theorem pi_bad_null {T : ℕ} {B : Set X} (hB : D B = 0) :
    (Measure.pi fun _ : Fin T => D) {xs | ∃ i, xs i ∈ B} = 0 := by
  have : {xs : Fin T → X | ∃ i, xs i ∈ B} = ⋃ i, (fun xs => xs i) ⁻¹' B := by
    ext xs; simp
  rw [this]
  refine measure_iUnion_null fun i => ?_
  rw [← Measure.map_apply (measurable_pi_apply i) (Set.to_countable B).measurableSet,
    (measurePreserving_eval (fun _ : Fin T => D) i).map_eq, hB]

/-- Bennett's tail for a test on the first `j + 1` of `T` fresh draws: their counts `U` exceed
`θ` times their counts `N` by `c`, both at most `R` almost surely, where `U` averages at most `θ`
times `N`. -/
theorem excess_test_le (U N : X → ℕ) {θ R c : ℝ} (hθ : 0 < θ) (hR : 0 < R) (hc : 0 ≤ c)
    (hb : ∀ᵐ x ∂D, (U x : ℝ) ≤ R ∧ (N x : ℝ) ≤ R)
    (hmean : ∫ x, (U x : ℝ) ∂D ≤ θ * ∫ x, (N x : ℝ) ∂D) {T j : ℕ} (hj : j < T) :
    (Measure.pi fun _ : Fin T => D)
        {xs | c ≤ (∑ i : Fin T, if (i : ℕ) ≤ j then (U (xs i) : ℝ) else 0)
          - θ * ∑ i : Fin T, if (i : ℕ) ≤ j then (N (xs i) : ℝ) else 0}
      ≤ ENNReal.ofReal (Real.exp (-((c + (j + 1 : ℕ) * θ * R)
          * Real.log ((c + (j + 1 : ℕ) * θ * R) / ((j + 1 : ℕ) * θ * R))
          - (c + (j + 1 : ℕ) * θ * R) + (j + 1 : ℕ) * θ * R) / ((1 + θ) * R))) := by
  set b := (1 + θ) * R
  set g : X → ℝ := fun x => (U x : ℝ) - θ * N x + θ * R
  set gc : X → ℝ := fun x => max 0 (min (g x) b)
  set B := {x : X | ¬ ((U x : ℝ) ≤ R ∧ (N x : ℝ) ≤ R)}
  have hB : D B = 0 := by
    rw [ae_iff] at hb; exact hb
  have hgc : ∀ x, x ∉ B → gc x = g x := by
    intro x hx
    simp only [B, Set.mem_ofPred_eq, not_not] at hx
    have h0 : 0 ≤ g x := by
      simp only [g]; nlinarith [hx.2, (Nat.cast_nonneg (U x) : (0 : ℝ) ≤ U x)]
    have h1 : g x ≤ b := by
      simp only [g, b]; nlinarith [hx.1, (Nat.cast_nonneg (N x) : (0 : ℝ) ≤ N x)]
    simp only [gc, min_eq_left h1, max_eq_right h0]
  have hgc_ae : gc =ᵐ[D] g := by
    rw [Filter.EventuallyEq, ae_iff]
    exact measure_mono_null (fun x (hx : ¬ gc x = g x) => by
      by_contra hxB; exact hx (hgc x hxB)) hB
  have hUi : Integrable (fun x => (U x : ℝ)) D := by
    refine Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable R ?_
    filter_upwards [hb] with x hx
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.1
  have hNi : Integrable (fun x => (N x : ℝ)) D := by
    refine Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable R ?_
    filter_upwards [hb] with x hx
    rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.2
  have hEg : ∫ x, gc x ∂D ≤ θ * R := by
    rw [integral_congr_ae hgc_ae]
    have h1 : ∫ x, g x ∂D = ∫ x, (U x : ℝ) ∂D - θ * ∫ x, (N x : ℝ) ∂D + θ * R := by
      simp only [g]
      rw [integral_add (f := fun x => (U x : ℝ) - θ * N x) (g := fun _ => θ * R)
        (hUi.sub (hNi.const_mul θ)) (integrable_const _),
        integral_sub (f := fun x => (U x : ℝ)) (g := fun x => θ * (N x : ℝ)) hUi
        (hNi.const_mul θ), integral_const_mul, integral_const]
      simp
    rw [h1]
    linarith
  set S := Finset.univ.filter fun i : Fin T => (i : ℕ) ≤ j
  have hS : S.card = j + 1 := by
    rw [show S = (Finset.range (j + 1)).attachFin (fun i hi => by
      simp only [Finset.mem_range] at hi; omega) from by
        ext i; simp [S, Finset.mem_attachFin, Nat.lt_succ_iff]]
    simp
  set A := ((j + 1 : ℕ) : ℝ) * θ * R
  have hA : 0 < A := by positivity
  have hsub : {xs : Fin T → X | c ≤ (∑ i : Fin T, if (i : ℕ) ≤ j then (U (xs i) : ℝ) else 0)
        - θ * ∑ i : Fin T, if (i : ℕ) ≤ j then (N (xs i) : ℝ) else 0}
      ⊆ {xs | c + A ≤ ∑ i ∈ S, gc (xs i)} ∪ {xs | ∃ i, xs i ∈ B} := by
    intro xs hxs
    by_cases hbad : ∃ i, xs i ∈ B
    · exact .inr hbad
    left
    push_neg at hbad
    simp only [Set.mem_ofPred_eq] at hxs ⊢
    have : ∑ i ∈ S, gc (xs i) = (∑ i : Fin T, if (i : ℕ) ≤ j then (U (xs i) : ℝ) else 0)
        - θ * (∑ i : Fin T, if (i : ℕ) ≤ j then (N (xs i) : ℝ) else 0) + A := by
      rw [Finset.sum_congr rfl fun i _ => hgc (xs i) (hbad i)]
      simp only [g, S, Finset.sum_filter, A]
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      have hcard : (∑ i : Fin T, if (i : ℕ) ≤ j then θ * R else 0) = ((j + 1 : ℕ) : ℝ) * θ * R := by
        rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
        change (S.card : ℝ) * (θ * R) = _
        rw [hS]; ring
      rw [← hcard, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun i _ => by split_ifs <;> ring
    rw [this]; linarith
  refine (measure_mono hsub).trans ((measure_union_le _ _).trans ?_)
  rw [pi_bad_null D hB, add_zero]
  have hind : iIndepFun (fun (i : Fin T) (xs : Fin T → X) => gc (xs i))
      (Measure.pi fun _ : Fin T => D) :=
    iIndepFun_pi (X := fun _ => gc) fun _ => (measurable_of_countable _).aemeasurable
  have hmean' : ∀ i ∈ S, (Measure.pi fun _ : Fin T => D)[fun xs => gc (xs i)] ≤ θ * R := by
    intro i _
    have : ∫ xs, gc (xs i) ∂(Measure.pi fun _ : Fin T => D) = ∫ x, gc x ∂D := by
      conv_rhs => rw [← (measurePreserving_eval (fun _ : Fin T => D) i).map_eq]
      rw [integral_map (measurable_pi_apply i).aemeasurable
          (measurable_of_countable _).aestronglyMeasurable]
    rw [this]; exact hEg
  have hbound := bennett_indep (μ := Measure.pi fun _ : Fin T => D)
    (fun i xs => gc (xs i)) (fun _ => θ * R) (b := b) (A := A) (s := c + A) S hind
    (fun i => measurable_of_countable _) (fun i xs => le_max_left _ _)
    (fun i xs => max_le (by positivity) (min_le_right _ _)) hmean'
    (by rw [Finset.sum_const, hS, nsmul_eq_mul]; simp only [A]; ring_nf; rfl) hA (by linarith)
  rw [← ofReal_measureReal]
  exact ENNReal.ofReal_le_ofReal hbound

end Tests

end OrthoDFA
