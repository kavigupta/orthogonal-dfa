import OrthoDFA.Proofs.BinomLaw
import Mathlib.Probability.ProductMeasure

/-!
# A significant edge is fixed in time

A round driven by fresh draws: `step s x` is the state after the draw `x`, or `none` once the
round has ended. Each state has a hypothesis `key s`. Where the hypothesis is significant, a
fresh draw lies in its record set `A (key s)` with chance at least `q`; each such draw raises the
counter `cnt` while the hypothesis stays, and the counter never reaches `m`, since `m` records
force a fix.

`fix_in_time`: over `T` draws, the chance that at some point a significant hypothesis outlasts
the next `n` draws, the round still running, is at most `T · P(Bin(n, q) < m)`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {X S H : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

section Defs

variable (step : S → X → Option S) (key : S → H)

/-- The round from `s` runs the first `n` of the draws `xs` with its hypothesis unchanged. -/
def Stays : ℕ → S → (T : ℕ) → (Fin T → X) → Prop
  | 0, _, _, _ => True
  | _ + 1, _, 0, _ => False
  | n + 1, s, _ + 1, xs =>
    ∃ s', step s (xs 0) = some s' ∧ key s' = key s ∧ Stays n s' _ (Fin.tail xs)

/-- At some point of the draws `xs` from `s`, the round reaches a significant hypothesis that then
runs `n` more draws unchanged. -/
def Lingers (sig : H → Prop) (n : ℕ) : S → (T : ℕ) → (Fin T → X) → Prop
  | _, 0, _ => False
  | s, _ + 1, xs => (sig (key s) ∧ Stays step key n s _ xs)
    ∨ ∃ s', step s (xs 0) = some s' ∧ Lingers sig n s' _ (Fin.tail xs)

end Defs

omit [Countable X] [MeasurableSingletonClass X] in
theorem piFinSuccAbove_zero' {n : ℕ} (d : Fin (n + 1) → X) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => X) 0 d = (d 0, Fin.tail d) := by
  ext j
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove, Fin.tail]

/-- The chance of `E` over `T + 1` draws, as the first draw's average of its section. -/
theorem pi_succ_apply (D : Measure X) [IsProbabilityMeasure D] (T : ℕ)
    (E : Set (Fin (T + 1) → X)) :
    (Measure.pi fun _ : Fin (T + 1) => D) E
      = ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈ E} ∂D := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (T + 1) => D) 0
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (T + 1) => X) 0
  have hE : E = e ⁻¹' {p | Fin.cons p.1 p.2 ∈ E} := by
    ext d
    simp only [Set.mem_preimage, Set.mem_ofPred_eq, e, piFinSuccAbove_zero', Fin.cons_self_tail]
  rw [hE, hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet,
    Measure.prod_apply (Set.to_countable _).measurableSet]
  rfl

variable (D : Measure X) [IsProbabilityMeasure D] (step : S → X → Option S) (key : S → H)
  (A : H → Set X) (cnt : S → ℕ) (sig : H → Prop) (q : ℝ) (m : ℕ)

open scoped Classical in
/-- From a significant hypothesis with the counter at `cnt s`, the hypothesis outlasts `n` draws
with chance at most `P(Bin(n, q) < m − cnt s)`. -/
theorem stays_le (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hA : ∀ h, sig h → q ≤ D.real (A h))
    (hcnt : ∀ s x s', step s x = some s' → key s' = key s →
      cnt s + (if x ∈ A (key s) then 1 else 0) ≤ cnt s')
    (hcap : ∀ s, cnt s < m) :
    ∀ (n T : ℕ) (s : S), sig (key s) →
      (Measure.pi fun _ : Fin T => D) {xs | Stays step key n s T xs}
        ≤ ENNReal.ofReal (1 - binomSfGe n q (m - cnt s)) := by
  intro n
  induction n with
  | zero =>
    intro T s _
    have hj : m - cnt s = (m - cnt s - 1) + 1 := by have := hcap s; omega
    rw [hj, binomSfGe_zero_left, sub_zero, ENNReal.ofReal_one]
    exact prob_le_one
  | succ n ih =>
    intro T s hs
    rcases T with _ | T
    · simp [Stays]
    obtain ⟨j, hj⟩ : ∃ j, m - cnt s = j + 1 := ⟨m - cnt s - 1, by have := hcap s; omega⟩
    set c₁ := 1 - binomSfGe n q j
    set c₀ := 1 - binomSfGe n q (j + 1)
    have hS1 : binomSfGe n q j ≤ 1 :=
      (binomSfGe_antitone' hq0 hq1 (Nat.zero_le j)).trans (binomSfGe_zero_right n q).le
    have hc₁ : 0 ≤ c₁ := by simp only [c₁]; linarith
    have hc₁₀ : c₁ ≤ c₀ := by
      simp only [c₁, c₀]; linarith [binomSfGe_antitone (n := n) hq0 hq1 j]
    set d := D.real (A (key s))
    have hd : q ≤ d := hA _ hs
    have hd1 : d ≤ 1 := measureReal_le_one
    -- the section at each first draw
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D)
        {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | Stays step key (n + 1) s (T + 1) xs}}
        ≤ (A (key s)).indicator (fun _ => ENNReal.ofReal c₁) x
          + (A (key s))ᶜ.indicator (fun _ => ENNReal.ofReal c₀) x := by
      intro x
      rcases hx : step s x with _ | s'
      · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | Stays step key (n + 1) s (T + 1) xs}}
            = ∅ := by
          ext xs; simp [Stays, hx]
        rw [this, measure_empty]; exact zero_le
      by_cases hk : key s' = key s
      · have hset : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | Stays step key (n + 1) s (T + 1)
            xs}} = {xs | Stays step key n s' T xs} := by
          ext xs; simp [Stays, hx, hk]
        rw [hset]
        refine (ih T s' (hk ▸ hs)).trans ?_
        have hc := hcnt s x s' hx hk
        by_cases hxA : x ∈ A (key s)
        · rw [Set.indicator_of_mem hxA, Set.indicator_of_notMem (by simpa using hxA), add_zero]
          rw [if_pos hxA] at hc
          refine ENNReal.ofReal_le_ofReal ?_
          simp only [c₁]
          linarith [binomSfGe_antitone' (n := n) hq0 hq1 (show m - cnt s' ≤ j by omega)]
        · rw [Set.indicator_of_notMem hxA, Set.indicator_of_mem (by simpa using hxA), zero_add]
          rw [if_neg hxA] at hc
          refine ENNReal.ofReal_le_ofReal ?_
          simp only [c₀]
          linarith [binomSfGe_antitone' (n := n) hq0 hq1 (show m - cnt s' ≤ j + 1 by omega)]
      · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | Stays step key (n + 1) s (T + 1) xs}}
            = ∅ := by
          ext xs; simp [Stays, hx, hk]
        rw [this, measure_empty]; exact zero_le
    rw [pi_succ_apply]
    refine (lintegral_mono hsec).trans ?_
    rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator
        (Set.to_countable _).measurableSet, lintegral_indicator (Set.to_countable _).measurableSet,
      setLIntegral_const, setLIntegral_const, ← ofReal_measureReal, ← ofReal_measureReal,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ,
      ← ENNReal.ofReal_mul hc₁, ← ENNReal.ofReal_mul (hc₁.trans hc₁₀),
      ← ENNReal.ofReal_add (mul_nonneg hc₁ measureReal_nonneg)
        (mul_nonneg (hc₁.trans hc₁₀) (by simp only [d] at hd1 ⊢; linarith))]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [hj, binomSfGe_succ]
    change c₁ * d + c₀ * (1 - d) ≤ 1 - (q * binomSfGe n q j + (1 - q) * binomSfGe n q (j + 1))
    have : 1 - (q * binomSfGe n q j + (1 - q) * binomSfGe n q (j + 1)) = q * c₁ + (1 - q) * c₀ := by
      simp only [c₁, c₀]; ring
    rw [this]
    nlinarith

open scoped Classical in
/-- `fix_in_time`: over `T` fresh draws, a significant hypothesis somewhere outlasts the next `n`
draws, the round still running, with chance at most `T · P(Bin(n, q) < m)`. -/
theorem fix_in_time (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hA : ∀ h, sig h → q ≤ D.real (A h))
    (hcnt : ∀ s x s', step s x = some s' → key s' = key s →
      cnt s + (if x ∈ A (key s) then 1 else 0) ≤ cnt s')
    (hcap : ∀ s, cnt s < m) (n : ℕ) :
    ∀ (T : ℕ) (s : S), (Measure.pi fun _ : Fin T => D) {xs | Lingers step key sig n s T xs}
      ≤ T * ENNReal.ofReal (1 - binomSfGe n q m) := by
  intro T
  induction T with
  | zero => intro s; simp [Lingers]
  | succ T ih =>
    intro s
    set c := ENNReal.ofReal (1 - binomSfGe n q m)
    have h1 : (Measure.pi fun _ : Fin (T + 1) => D)
        {xs | sig (key s) ∧ Stays step key n s (T + 1) xs} ≤ c := by
      by_cases hs : sig (key s)
      · simp only [hs, true_and]
        refine (stays_le D step key A cnt sig q m hq0 hq1 hA hcnt hcap n (T + 1) s hs).trans ?_
        exact ENNReal.ofReal_le_ofReal (by
          linarith [binomSfGe_antitone' (n := n) hq0 hq1 (show m - cnt s ≤ m by omega)])
      · simp [hs]
    have h2 : (Measure.pi fun _ : Fin (T + 1) => D)
        {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
          ∧ Lingers step key sig n s' T (Fin.tail xs)} ≤ T * c := by
      rw [pi_succ_apply]
      calc ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
              {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
                ∧ Lingers step key sig n s' T (Fin.tail xs)}} ∂D
          ≤ ∫⁻ _, (T : ENNReal) * c ∂D := by
            refine lintegral_mono fun x => ?_
            rcases hx : step s x with _ | s'
            · simp [hx]
            · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
                  ∧ Lingers step key sig n s' T (Fin.tail xs)}}
                  = {xs | Lingers step key sig n s' T xs} := by
                ext xs; simp [hx]
              rw [this]
              exact ih s'
        _ = T * c := by rw [lintegral_const, measure_univ, mul_one]
    have hsplit : {xs | Lingers step key sig n s (T + 1) xs}
        = {xs | sig (key s) ∧ Stays step key n s (T + 1) xs}
          ∪ {xs : Fin (T + 1) → X | ∃ s', step s (xs 0) = some s'
            ∧ Lingers step key sig n s' T (Fin.tail xs)} := by
      ext xs; simp [Lingers]
    rw [hsplit]
    refine (measure_union_le _ _).trans ?_
    calc _ ≤ c + T * c := add_le_add h1 h2
      _ = (T + 1 : ℕ) * c := by push_cast; ring

end OrthoDFA
