import OrthoDFA.Proofs.Level5Clean
import OrthoDFA.Proofs.FixTime

/-!
# A test errs at most once a hypothesis

Every count starts afresh at each change of hypothesis, and within a hypothesis its probes are
fresh draws at a fixed tree and edges. A test's weight, its likelihood ratio or an exponentiated
count less twice its running mean, is a product of one factor a probe; where the factors average
at most one, the weight times `a` passing one is Ville's event, of chance at most `a`
(`evL_le`). A round that ends at a site where its hypothesis's sampler is not good has passed one
such weight within the version it ends in (`bad_end_skip`), and the versions number at most
`versionBound`, so each test errs with chance at most `versionBound · a` over the round.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Mgf

/-- One probe's exponentiated increment averages at most `1`. -/
theorem mgf_step {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (U N : X → ℕ) {θ L R : ℝ} (hθ : 0 ≤ θ) (hL : 0 < L)
    (hR : 0 ≤ R) (hb : ∀ᵐ x ∂D, (U x : ℝ) ≤ L ∧ (N x : ℝ) ≤ L * R) {η : ℝ} (hη : 0 ≤ η)
    (hmean : ∫ x, (U x : ℝ) ∂D ≤ θ * ∫ x, (N x : ℝ) ∂D + η) :
    ∫⁻ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x - 2 * η) / (2 * L * (1 + 4 * θ * R)))) ∂D
      ≤ 1 := by
  have hK : 0 < 2 * L * (1 + 4 * θ * R) := by positivity
  have hsplit : ∀ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x - 2 * η)
      / (2 * L * (1 + 4 * θ * R)))) = ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x)
        / (2 * L * (1 + 4 * θ * R)))) * ENNReal.ofReal (Real.exp (-(2 * η)
          / (2 * L * (1 + 4 * θ * R)))) := by
    intro x
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    congr 2
    ring
  simp_rw [hsplit]
  rw [lintegral_mul_const _ (measurable_of_countable _)]
  set l := 1 / (2 * L * (1 + 4 * θ * R))
  have hl : 0 < l := by positivity
  have hlL : l * L * (1 + 4 * θ * R) = 1 / 2 := by
    simp only [l]; field_simp
  set A := l + l ^ 2 * L
  set B := 2 * θ * l - 4 * θ ^ 2 * l ^ 2 * L * R
  set g : X → ℝ := fun x => 1 + A * U x - B * N x
  have hUi : Integrable (fun x => (U x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable L (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.1)
  have hNi : Integrable (fun x => (N x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable (L * R) (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx.2)
  have hgi : Integrable g D := ((integrable_const 1).add (hUi.const_mul A)).sub (hNi.const_mul B)
  have hpt : ∀ᵐ x ∂D, Real.exp ((U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R))) ≤ g x := by
    filter_upwards [hb] with x ⟨hU, hN⟩
    have hU0 : (0 : ℝ) ≤ U x := Nat.cast_nonneg _
    have hN0 : (0 : ℝ) ≤ N x := Nat.cast_nonneg _
    set y := (U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R))
    have hy : y = l * (U x - 2 * θ * N x) := by simp only [y, l]; ring
    have hy1 : |y| ≤ 1 := by
      rw [hy, abs_le]
      constructor
      · have : l * (2 * θ * N x) ≤ 1 := by
          calc l * (2 * θ * N x) ≤ l * (2 * θ * (L * R)) := by gcongr
            _ ≤ l * L * (1 + 4 * θ * R) := by
              nlinarith [mul_nonneg (mul_nonneg hl.le hL.le) (mul_nonneg hθ hR)]
            _ ≤ 1 := by rw [hlL]; norm_num
        nlinarith
      · have : l * U x ≤ 1 := by
          calc l * U x ≤ l * L := by gcongr
            _ ≤ l * L * (1 + 4 * θ * R) := by
              nlinarith [mul_nonneg (mul_nonneg hl.le hL.le) (mul_nonneg hθ hR)]
            _ ≤ 1 := by rw [hlL]; norm_num
        nlinarith [mul_nonneg hl.le (mul_nonneg hθ hN0)]
    have hexp := Real.abs_exp_sub_one_sub_id_le hy1
    have hsq : y ^ 2 ≤ l ^ 2 * (L * U x + 4 * θ ^ 2 * (L * R) * N x) := by
      rw [hy, mul_pow]
      gcongr
      nlinarith [mul_le_mul_of_nonneg_left hU hU0, mul_le_mul_of_nonneg_left hN hN0,
        mul_nonneg hθ (mul_nonneg hU0 hN0)]
    have h1 : Real.exp y ≤ 1 + y + y ^ 2 := by
      have := (abs_le.1 hexp).2
      linarith
    have h2 : 1 + y + l ^ 2 * (L * U x + 4 * θ ^ 2 * (L * R) * N x) = g x := by
      simp only [g, A, B]; rw [hy]; ring
    linarith
  calc (∫⁻ x, ENNReal.ofReal (Real.exp ((U x - 2 * θ * N x) / (2 * L * (1 + 4 * θ * R)))) ∂D)
        * ENNReal.ofReal (Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R))))
      ≤ (∫⁻ x, ENNReal.ofReal (g x) ∂D)
        * ENNReal.ofReal (Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R)))) := by
        exact mul_le_mul' (lintegral_mono_ae (by
          filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx)) le_rfl
    _ = ENNReal.ofReal (∫ x, g x ∂D)
        * ENNReal.ofReal (Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R)))) := by
        rw [ofReal_integral_eq_lintegral_ofReal hgi]
        filter_upwards [hpt] with x hx using (Real.exp_pos _).le.trans hx
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_mul' (Real.exp_pos _).le, ← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        suffices h : ∫ x, g x ∂D ≤ 1 + 2 * η / (2 * L * (1 + 4 * θ * R)) by
          have h1 := Real.add_one_le_exp (2 * η / (2 * L * (1 + 4 * θ * R)))
          have h2 : Real.exp (2 * η / (2 * L * (1 + 4 * θ * R)))
              * Real.exp (-(2 * η) / (2 * L * (1 + 4 * θ * R))) = 1 := by
            rw [← Real.exp_add, show 2 * η / (2 * L * (1 + 4 * θ * R))
              + -(2 * η) / (2 * L * (1 + 4 * θ * R)) = 0 by ring, Real.exp_zero]
          have h3 := (Real.exp_pos (-(2 * η) / (2 * L * (1 + 4 * θ * R)))).le
          nlinarith
        have hint : ∫ x, g x ∂D = 1 + A * ∫ x, (U x : ℝ) ∂D - B * ∫ x, (N x : ℝ) ∂D := by
          have e1 := integral_sub (μ := D) (f := fun x => 1 + A * (U x : ℝ))
            (g := fun x => B * (N x : ℝ)) ((integrable_const 1).add (hUi.const_mul A))
            (hNi.const_mul B)
          have e2 := integral_add (μ := D) (f := fun _ => (1 : ℝ)) (g := fun x => A * (U x : ℝ))
            (integrable_const 1) (hUi.const_mul A)
          simp only [g]
          rw [e1, e2, integral_const_mul, integral_const_mul, integral_const]
          simp
        have hEN : 0 ≤ ∫ x, (N x : ℝ) ∂D := integral_nonneg fun _ => Nat.cast_nonneg _
        have hA : 0 ≤ A := by positivity
        rw [hint]
        have h1 := mul_le_mul_of_nonneg_left hmean hA
        have hkey : A * θ - B = -(θ * l) / 2 := by
          simp only [A, B]
          have : l * L = 1 / 2 / (1 + 4 * θ * R) := by
            rw [← hlL]; field_simp
          field_simp
          nlinarith [hlL]
        have hAl : A ≤ 2 * l := by
          have : l * L ≤ 1 / 2 := by
            rw [← hlL]
            nlinarith [mul_nonneg (mul_nonneg hl.le hL.le) (mul_nonneg hθ hR)]
          simp only [A]; nlinarith
        have hl' : 2 * η / (2 * L * (1 + 4 * θ * R)) = 2 * η * l := by simp only [l]; ring
        rw [hl']
        nlinarith [mul_nonneg (mul_nonneg hθ hl.le) hEN, mul_le_mul_of_nonneg_left hAl hη]

/-- A count's halved mean exceeds it by much only rarely. -/
theorem mgf_lower {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (D : Measure X) [IsProbabilityMeasure D] (N : X → ℕ) {B : ℝ} (hB : 0 < B)
    (hb : ∀ᵐ x ∂D, (N x : ℝ) ≤ B) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (((∫ y, (N y : ℝ) ∂D) / 2 - N x) / (2 * B))) ∂D ≤ 1 := by
  set r := ∫ y, (N y : ℝ) ∂D
  have hNi : Integrable (fun x => (N x : ℝ)) D :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable B (by
      filter_upwards [hb] with x hx
      rw [Real.norm_eq_abs, abs_of_nonneg (Nat.cast_nonneg _)]; exact hx)
  have hpt : ∀ᵐ x ∂D, Real.exp (-(N x : ℝ) / (2 * B)) ≤ 1 - (N x : ℝ) / (4 * B) := by
    filter_upwards [hb] with x hx
    have hN0 : (0 : ℝ) ≤ N x := Nat.cast_nonneg _
    set y := (N x : ℝ) / (2 * B) with hy
    have hy0 : 0 ≤ y := by positivity
    have hy1 : y ≤ 1 / 2 := by rw [hy, div_le_iff₀ (by positivity)]; linarith
    have habs : |-y| ≤ 1 := by rw [abs_neg, abs_of_nonneg hy0]; linarith
    have h1 := (abs_le.1 (Real.abs_exp_sub_one_sub_id_le habs)).2
    have h2 : -(N x : ℝ) / (2 * B) = -y := by rw [hy]; ring
    have h3 : (N x : ℝ) / (4 * B) = y / 2 := by rw [hy]; field_simp; ring
    rw [h2, h3]
    nlinarith
  have hsplit : ∀ x, ENNReal.ofReal (Real.exp ((r / 2 - N x) / (2 * B)))
      = ENNReal.ofReal (Real.exp (r / (4 * B)))
        * ENNReal.ofReal (Real.exp (-(N x : ℝ) / (2 * B))) := by
    intro x
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
    congr 2
    field_simp
    ring
  simp_rw [hsplit]
  rw [lintegral_const_mul _ (measurable_of_countable _)]
  have hgi : Integrable (fun x => 1 - (N x : ℝ) / (4 * B)) D :=
    (integrable_const 1).sub (hNi.div_const _)
  have hg0 : ∀ᵐ x ∂D, 0 ≤ 1 - (N x : ℝ) / (4 * B) := by
    filter_upwards [hb] with x hx
    have : (N x : ℝ) / (4 * B) ≤ 1 / 4 := by rw [div_le_iff₀ (by positivity)]; linarith
    linarith
  have hint : ∫ x, (1 - (N x : ℝ) / (4 * B)) ∂D = 1 - r / (4 * B) := by
    rw [integral_sub (integrable_const 1) (hNi.div_const _), integral_const, integral_div]
    simp [r]
  calc ENNReal.ofReal (Real.exp (r / (4 * B)))
        * ∫⁻ x, ENNReal.ofReal (Real.exp (-(N x : ℝ) / (2 * B))) ∂D
      ≤ ENNReal.ofReal (Real.exp (r / (4 * B)))
        * ∫⁻ x, ENNReal.ofReal (1 - (N x : ℝ) / (4 * B)) ∂D := by
        exact mul_le_mul' le_rfl (lintegral_mono_ae (by
          filter_upwards [hpt] with x hx using ENNReal.ofReal_le_ofReal hx))
    _ = ENNReal.ofReal (Real.exp (r / (4 * B)) * (1 - r / (4 * B))) := by
        rw [← ofReal_integral_eq_lintegral_ofReal hgi hg0, hint,
          ENNReal.ofReal_mul (Real.exp_pos _).le]
    _ ≤ 1 := by
        rw [← ENNReal.ofReal_one]
        refine ENNReal.ofReal_le_ofReal ?_
        have h1 := Real.add_one_le_exp (-(r / (4 * B)))
        have h2 : Real.exp (r / (4 * B)) * Real.exp (-(r / (4 * B))) = 1 := by
          rw [← Real.exp_add]; simp
        have h3 := (Real.exp_pos (r / (4 * B))).le
        nlinarith

end Mgf

section Ville

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- The run reaches version `v` from `s`, and `Q` holds there of the remaining draws. -/
def SkipV (Q : Level5State α → List (FreeMonoid α) → Prop) (v : ℕ) :
    Level5State α → List (FreeMonoid α) → Prop
  | s, [] => s.version = v ∧ Q s []
  | s, x :: xs => if s.version = v then Q s (x :: xs) else
      match applyProbe C cut s x with
      | .inl s' => SkipV Q v s' xs
      | .inr _ => False

theorem skipV_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : Level5State α → Prop)
    (hP : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → P s')
    (Q : Level5State α → List (FreeMonoid α) → Prop) (v : ℕ) {c : ENNReal}
    (hQ : ∀ (T : ℕ) (s : Level5State α), P s → s.version = v →
      (Measure.pi fun _ : Fin T => D) {xs | Q s (List.ofFn xs)} ≤ c) :
    ∀ (T : ℕ) (s : Level5State α), P s →
      (Measure.pi fun _ : Fin T => D) {xs | SkipV C cut Q v s (List.ofFn xs)} ≤ c := by
  intro T
  induction T with
  | zero =>
    intro s hs
    by_cases hv : s.version = v
    · refine (measure_mono fun xs hxs => ?_).trans (hQ 0 s hs hv)
      simp only [Set.mem_ofPred_eq, List.ofFn_zero, SkipV] at hxs ⊢; exact hxs.2
    · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
      simp only [Set.mem_ofPred_eq, List.ofFn_zero, SkipV] at hxs; exact hv hxs.1
  | succ T ih =>
    intro s hs
    by_cases hv : s.version = v
    · refine (measure_mono fun xs hxs => ?_).trans (hQ (T + 1) s hs hv)
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, SkipV, hv, if_true] at hxs ⊢
      exact hxs
    rw [pi_succ_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
          {xs : Fin (T + 1) → FreeMonoid α | SkipV C cut Q v s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ _, c ∂D := lintegral_mono fun x => by
          rcases hst : applyProbe C cut s x with s' | _
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                SkipV C cut Q v s (List.ofFn xs)}} = {xs | SkipV C cut Q v s' (List.ofFn xs)} := by
              ext xs; simp only [Set.mem_ofPred_eq, List.ofFn_cons, SkipV, hv, if_false, hst]
            rw [this]; exact ih s' (hP s s' x hs hst)
          · have : {xs | Fin.cons x xs ∈ {xs : Fin (T + 1) → FreeMonoid α |
                SkipV C cut Q v s (List.ofFn xs)}} = ∅ := by
              ext xs
              simp only [Set.mem_ofPred_eq, List.ofFn_cons, SkipV, hv, if_false, hst,
                Set.mem_empty_iff_false]
            rw [this, measure_empty]; exact zero_le
      _ = c := by rw [lintegral_const, measure_univ, mul_one]

/-- From `s`, within its version, the weight `r` times its probes' factors reaches `1`, every
state on the way satisfying `ok`. -/
def EvL (ok : Level5State α → Prop) (f : Level5State α → FreeMonoid α → ℝ) :
    ℝ → Level5State α → List (FreeMonoid α) → Prop
  | _, _, [] => False
  | r, s, x :: xs => ok s ∧ (1 ≤ r * f s x ∨ match applyProbe C cut s x with
      | .inl s' => s'.version = s.version ∧ EvL ok f (r * f s x) s' xs
      | .inr _ => False)

/-- Ville's inequality along a version: where every probe's factor averages at most `1`, the
weight `r` times the factors reaches `1` with chance at most `r`. -/
theorem evL_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (ok : Level5State α → Prop) (f : Level5State α → FreeMonoid α → ℝ) (hf0 : ∀ s x, 0 ≤ f s x)
    (hf : ∀ s, ok s → ∫⁻ x, ENNReal.ofReal (f s x) ∂D ≤ 1) :
    ∀ (T : ℕ) (r : ℝ) (s : Level5State α), 0 ≤ r →
      (Measure.pi fun _ : Fin T => D) {xs | EvL C cut ok f r s (List.ofFn xs)}
        ≤ ENNReal.ofReal r := by
  intro T
  induction T with
  | zero => intro r s _; simp [EvL]
  | succ T ih =>
    intro r s hr
    by_cases hok : ok s
    swap
    · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, EvL] at hxs
      exact hok hxs.1
    rw [pi_succ_apply]
    have hsec : ∀ x, (Measure.pi fun _ : Fin T => D) {xs | Fin.cons x xs ∈
        {xs : Fin (T + 1) → FreeMonoid α | EvL C cut ok f r s (List.ofFn xs)}}
        ≤ ENNReal.ofReal (r * f s x) := by
      intro x
      by_cases h1 : 1 ≤ r * f s x
      · exact prob_le_one.trans (by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal h1)
      rcases hst : applyProbe C cut s x with s' | _
      · refine le_trans (measure_mono fun xs hxs => ?_)
          (ih (r * f s x) s' (mul_nonneg hr (hf0 s x)))
        simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvL, hst] at hxs
        rcases hxs.2 with h | h
        · exact absurd h h1
        · exact h.2
      · refine (measure_mono (t := ∅) fun xs hxs => ?_).trans (by simp)
        simp only [Set.mem_ofPred_eq, List.ofFn_cons, EvL, hst] at hxs
        rcases hxs.2 with h | h
        · exact h1 h
        · exact h
    refine (lintegral_mono hsec).trans ?_
    have : ∀ x, ENNReal.ofReal (r * f s x) = ENNReal.ofReal r * ENNReal.ofReal (f s x) :=
      fun x => ENNReal.ofReal_mul hr
    simp_rw [this]
    rw [lintegral_const_mul _ (measurable_of_countable _)]
    calc ENNReal.ofReal r * ∫⁻ x, ENNReal.ofReal (f s x) ∂D ≤ ENNReal.ofReal r * 1 := by
          gcongr; exact hf s hok
      _ = _ := mul_one _

/-- A round from `s` ending badly at a site whose weight, one at each new version and multiplied
by a probe's factor within one, passed `1/a` at an `ok` hypothesis: then from some later version
on, or within `s`'s version from the weight it holds, Ville's event happens. -/
theorem bad_end_skip (P : Level5State α → Prop)
    (hP : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → P s')
    (W : Level5State α → ℝ) (f : Level5State α → FreeMonoid α → ℝ) (ok : Level5State α → Prop)
    (a : ℝ) (Bad : Level5Ending α → Level5State α → Prop)
    (hnew : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → s'.version ≠ s.version → W s' = 1)
    (hkeep : ∀ s s' x, P s → applyProbe C cut s x = .inl s' → s'.version = s.version →
      W s' = W s * f s x ∧ (ok s' ↔ ok s))
    (hbad : ∀ s s' x e, P s → applyProbe C cut s x = .inr (e, s') → Bad e s' →
      ok s ∧ 1 ≤ a * (W s * f s x)) :
    ∀ (l : List (FreeMonoid α)) (s : Level5State α), P s →
      RunEnds (applyProbe C cut) Bad s l →
        (∃ v, s.version < v ∧ SkipV C cut (EvL C cut ok f a) v s l)
          ∨ EvL C cut ok f (a * W s) s l
  | [], _, _, h => h.elim
  | x :: xs, s, hs, h => by
    simp only [RunEnds] at h
    rcases hst : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [hst] at h
    · have hs' := hP s s' x hs hst
      rcases bad_end_skip P hP W f ok a Bad hnew hkeep hbad xs s' hs' h with ⟨v, hv, hsk⟩ | hev
      · by_cases hver : s'.version = s.version
        · refine .inl ⟨v, hver ▸ hv, ?_⟩
          simp only [SkipV, show s.version ≠ v by omega, if_false, hst]
          exact hsk
        · have hlt : s.version < s'.version :=
            lt_of_le_of_ne (applyProbe_version C cut hst) (Ne.symm hver)
          refine .inl ⟨v, by omega, ?_⟩
          simp only [SkipV, show s.version ≠ v by omega, if_false, hst]
          exact hsk
      · by_cases hver : s'.version = s.version
        · obtain ⟨hW, hok⟩ := hkeep s s' x hs hst hver
          refine .inr ⟨?_, .inr ?_⟩
          · rcases xs with _ | ⟨y, ys⟩
            · exact hev.elim
            · exact hok.1 hev.1
          · rw [hst]
            refine ⟨hver, ?_⟩
            rw [hW, ← mul_assoc] at hev
            exact hev
        · have hW := hnew s s' x hs hst hver
          rw [hW, mul_one] at hev
          refine .inl ⟨s'.version, lt_of_le_of_ne (applyProbe_version C cut hst) (Ne.symm hver), ?_⟩
          · simp only [SkipV, show s.version ≠ s'.version from fun h => hver h.symm, if_false, hst]
            rcases xs with _ | ⟨y, ys⟩
            · exact hev.elim
            · simp only [SkipV, if_true]
              exact hev
    · obtain ⟨hok, h1⟩ := hbad s s' x e hs hst h
      exact .inr ⟨hok, .inl (by rw [mul_assoc]; exact h1)⟩

end Ville

section Shape

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- A probe that stops at an unlearned edge it could learn. -/
def WalkOutcome.IsMember : WalkOutcome α → Prop
  | .member _ => True
  | _ => False

/-- A probe that stops at an unlearned edge stops where its prefix and the prefix's successor sift
decidedly, the prefix to the edge's leaf. -/
theorem probeBy_member_learns {T : DTree α} {edges : EdgeTable α} {k : ℕ} {x u : FreeMonoid α}
    (h : probeBy cut T edges k x = .member u) :
    ∃ c p t, T.siftCut cut u = .inl p ∧ x.toList[u.toList.length]? = some c
      ∧ T.siftCut cut (u * FreeMonoid.of c) = .inl t ∧ edges p c = none := by
  rw [probeBy_eq] at h
  split_ifs at h with hs
  rcases hw : walkCheckBy cut T edges k x with o | ⟨ps, hi⟩ <;> rw [hw] at h
  swap
  · simp only [Sum.elim_inr] at h
    have := bracketAt_isSearch (α := α) (agreesAtBy cut T x fun j => ps.getD (j - k) []) ps
      (hi - k) k hi
    rw [h] at this
    exact this.elim
  simp only [Sum.elim_inl, id] at h
  subst h
  unfold walkCheckBy at hw
  rcases hk : kWalkBy cut T edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hk] at hw
  · simp at hw
  · simp only [] at hw
    rcases h1 : T.siftCut cut (prefixOf x (j + 1)) with t | b <;> rw [h1] at hw
    · simp only [] at hw
      rcases h2 : T.siftCut cut (prefixOf x j) with p | b <;> rw [h2] at hw
      · simp only [] at hw
        split_ifs at hw with hps
        · simp only [Sum.inl.injEq, WalkOutcome.member.injEq] at hw
          subst hw hps
          unfold kWalkBy at hk
          rcases h3 : T.siftCut cut (prefixOf x k) with p₀ | b <;> rw [h3] at hk
          · simp only [] at hk
            rcases hf : follow edges p₀ (x.toList.drop k) with ps | ⟨s', c', i⟩ <;> rw [hf] at hk
            · simp at hk
            · simp only [KWalk.edge.injEq] at hk
              obtain ⟨rfl, rfl, rfl⟩ := hk
              obtain ⟨hi, hc, hn, -⟩ := follow_inr _ _ _ _ _ hf
              simp only [List.length_drop] at hi
              have hlen : (prefixOf x (k + i)).toList.length = k + i := by
                simp [prefixOf]; omega
              have hxc : x.toList[k + i]? = some c' := by
                rw [List.getElem?_eq_getElem (by omega)]
                simp only [← hc, List.getElem_drop]
              refine ⟨c', s', t, h2, by rw [hlen]; exact hxc, ?_, hn⟩
              rw [← prefixOf_succ hxc]
              exact h1
          · simp at hk
      · simp at hw
    · simp at hw
  · simp only [] at hw
    split at hw
    · simp at hw
    · split_ifs at hw <;> simp at hw

/-- A probe that learns an edge moves to a fresh hypothesis. -/
theorem recordProbe_of_member (s : Level5State α) {x : FreeMonoid α}
    (h : (probeBy cut s.tree s.edges C.walkStart x).IsMember) :
    (recordProbe C cut s x).version = s.version + 1
      ∧ ∃ T E, recordProbe C cut s x = freshState T E (s.version + 1) := by
  rcases ho : probeBy cut s.tree s.edges C.walkStart x with _ | _ | _ | _ | _ | u <;>
    rw [ho] at h <;> simp only [WalkOutcome.IsMember] at h
  obtain ⟨c, p, t, hp, hc, ht, he⟩ := probeBy_member_learns cut ho
  unfold recordProbe
  simp only [ho, hc, hp, ht, he]
  exact ⟨rfl, _, _, rfl⟩

/-- A step that keeps the hypothesis learns no edge. -/
theorem not_member_of_inl {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') (hv : s'.version = s.version) :
    ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember := by
  intro hm
  obtain ⟨hv', T, E, hf⟩ := recordProbe_of_member C cut s hm
  rcases (applyProbe_inl C cut h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, hrec, rfl, -⟩
  · omega
  · have := splitState_version cut (recordProbe C cut s x) p c t
    omega


open scoped Classical in
/-- A probe that keeps the hypothesis: its counts added, its search counted. -/
noncomputable def keepProbe (s : Level5State α) (x : FreeMonoid α) : Level5State α :=
  { s.countProbe cut C.walkStart x with
    searches := s.searches
      + if (probeBy cut s.tree s.edges C.walkStart x).IsSearch then 1 else 0 }

/-- A step that goes on moves to a fresh hypothesis with a higher version, or keeps the
hypothesis. -/
theorem applyProbe_inl_shape {s s' : Level5State α} {x : FreeMonoid α}
    (h : applyProbe C cut s x = .inl s') :
    (s.version < s'.version ∧ ∃ T E, s' = freshState T E s'.version) ∨ s' = keepProbe C cut s x := by
  rcases (applyProbe_inl C cut h).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, hrec, rfl, -⟩
  · rcases recordProbe_spec C cut s x with h1 | ⟨u, c, p, t, -, -, -, -, -, h1⟩ <;> rw [h1]
    · exact .inr rfl
    · exact .inl ⟨Nat.lt_succ_self _, _, _, rfl⟩
  · obtain ⟨-, -, -, -, -, -, t₀, w, hpe, -⟩ := recordBy_spec cut hrec
    have hrp := recordProbe_of_record C cut s x hrec
    have hpe' : (recordProbe C cut s x).edges p c = some (t₀, w) := by rw [hrp]; exact hpe
    rw [splitState_of_some cut _ hpe']
    exact .inl ⟨by rw [hrp]; exact Nat.lt_succ_self _, _, _, rfl⟩

open scoped Classical in
/-- A fresh hypothesis's tests are silent. -/
theorem runTests_fresh (ha : 0 < C.testLevel) (ha1 : C.testLevel < 1) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) (T : DTree α) (E : EdgeTable α) (v : ℕ) :
    runTests C (freshState T E v) = none := by
  have h1 := not_seqRate_zero (p₀ := C.startUndecidedRate) (p₁ := C.startAltRate) ha ha1
  have h4 := not_seqRate_zero (p₀ := C.disagreementRate) (p₁ := C.disagreementAltRate) ha ha1
  unfold runTests
  rw [if_neg (by simpa [freshState] using h1), dif_neg, if_neg, if_neg (by simpa [freshState] using h4)]
  · rintro ⟨-, h, -⟩
    simp only [freshState, Nat.cast_le, Nat.le_zero] at h
    omega
  · rintro ⟨e, -, -, h⟩
    have := hexc 0
    simp only [freshState, List.length_nil, Nat.cast_zero, mul_zero, sub_zero] at h
    linarith

/-- A step whose tests fire learns no edge. -/
theorem not_member_of_inr (ha : 0 < C.testLevel) (ha1 : C.testLevel < 1)
    (hn₀ : 1 ≤ C.firstLook) (hexc : ∀ j, 0 < C.edgeExcess j) {s s' : Level5State α}
    {x : FreeMonoid α} {e : Level5Ending α} (h : applyProbe C cut s x = .inr (e, s'))
    (he : e ≠ .tooBig) : ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember := by
  intro hm
  obtain ⟨-, T, E, hf⟩ := recordProbe_of_member C cut s hm
  rcases applyProbe_inr C cut h with ⟨hr, -⟩ | ⟨rfl, -⟩
  · rw [hf, runTests_fresh C ha ha1 hn₀ hexc] at hr
    simp at hr
  · exact he rfl

/-- A step that ends is past `maxLeaves` leaves, or its tests fired at a probe that kept the
hypothesis. -/
theorem applyProbe_inr_shape (ha : 0 < C.testLevel) (ha1 : C.testLevel < 1)
    (hn₀ : 1 ≤ C.firstLook) (hexc : ∀ j, 0 < C.edgeExcess j) {s s' : Level5State α}
    {x : FreeMonoid α} {e : Level5Ending α} (h : applyProbe C cut s x = .inr (e, s')) :
    e = .tooBig ∨ (s' = keepProbe C cut s x ∧ runTests C s' = some e) := by
  rcases applyProbe_inr C cut h with ⟨hr, rfl⟩ | ⟨rfl, -⟩
  · rcases recordProbe_spec C cut s x with h1 | ⟨u, c, p, t, -, -, -, -, -, h1⟩
    · rw [h1] at hr ⊢
      exact .inr ⟨rfl, hr⟩
    · rw [h1] at hr
      rw [Level5State.setEdge, runTests_fresh C ha ha1 hn₀ hexc] at hr
      simp at hr
  · exact .inl rfl

theorem keepProbe_hyp (s : Level5State α) (x : FreeMonoid α) :
    (keepProbe C cut s x).tree = s.tree ∧ (keepProbe C cut s x).edges = s.edges
      ∧ (keepProbe C cut s x).version = s.version := ⟨rfl, rfl, rfl⟩

/-- The counts every reached state keeps in order: the start's undecided strings and the
searches at most the probes, the middles at most the searches, each edge's undecided strings at
most its reads. -/
def CountInv (s : Level5State α) : Prop :=
  s.startUndecided.length ≤ s.probes ∧ s.middles.length ≤ s.searches ∧ s.searches ≤ s.probes
    ∧ ∀ p c, (s.edgeUndecided p c).length ≤ s.edgeReads p c

theorem countInv_fresh (T : DTree α) (E : EdgeTable α) (v : ℕ) : CountInv (freshState T E v) := by
  simp [CountInv, freshState]

open scoped Classical in
theorem countInv_keep {s : Level5State α} (hs : CountInv s) (x : FreeMonoid α) :
    CountInv (keepProbe C cut s x) := by
  obtain ⟨h1, h2, h3, h4⟩ := hs
  have hst : (startHarvestBy cut s.tree C.walkStart x).length ≤ 1 := by
    unfold startHarvestBy
    split_ifs
    · simp
    · rcases s.tree.siftCut cut (prefixOf x C.walkStart) with _ | _ <;> simp
  have hmid := middleHarvestBy_length cut s.tree s.edges C.walkStart x
  have hd : (middleHarvestBy cut s.tree s.edges C.walkStart x).length
      ≤ if (probeBy cut s.tree s.edges C.walkStart x).IsSearch then 1 else 0 := by
    rw [hmid]
    rcases probeBy cut s.tree s.edges C.walkStart x with _ | _ | _ | _ | _ | _ <;>
      simp [WalkOutcome.IsSearch]
  have hi : (if (probeBy cut s.tree s.edges C.walkStart x).IsSearch then 1 else 0) ≤ 1 := by
    split <;> omega
  simp only [CountInv, keepProbe, Level5State.countProbe, List.length_append]
  refine ⟨by omega, by omega, by omega, fun p c => ?_⟩
  have := edgeHarvestBy_le_edgeReadsBy cut s.tree s.edges C.walkStart x (p, c)
  have := h4 p c
  omega

theorem countInv_step {s s' : Level5State α} {x : FreeMonoid α} (hs : CountInv s)
    (h : applyProbe C cut s x = .inl s') : CountInv s' := by
  rcases applyProbe_inl_shape C cut h with ⟨-, T, E, he⟩ | he <;> rw [he]
  · exact countInv_fresh _ _ _
  · exact countInv_keep C cut hs x

end Shape

section Site

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- What every reached state keeps. -/
def RunInv (s : Level5State α) : Prop := VerInv C s ∧ CountInv s

theorem runInv_step {s s' : Level5State α} {x : FreeMonoid α} (hs : RunInv C s)
    (h : applyProbe C cut s x = .inl s') : RunInv C s' :=
  ⟨verInv_step C cut hs.1 h, countInv_step C cut hs.2 h⟩

theorem runInv_start (h : 2 ≤ C.maxLeaves) : RunInv C (initialState : Level5State α) :=
  ⟨verInv_start C h, countInv_fresh _ _ _⟩

theorem skipV_runInv (Q : Level5State α → List (FreeMonoid α) → Prop) (v : ℕ) :
    ∀ (l : List (FreeMonoid α)) (s : Level5State α), RunInv C s → SkipV C cut Q v s l →
      ∃ s' : Level5State α, RunInv C s' ∧ s'.version = v
  | [], s, hs, h => ⟨s, hs, h.1⟩
  | x :: xs, s, hs, h => by
    simp only [SkipV] at h
    split_ifs at h with hv
    · exact ⟨s, hs, hv⟩
    · rcases hst : applyProbe C cut s x with s' | _ <;> rw [hst] at h
      · exact skipV_runInv Q v xs s' (runInv_step C cut hs hst) h
      · exact h.elim

/-- A test whose weight is one at a fresh hypothesis and multiplies by a probe's factor while the
hypothesis keeps, at hypotheses where the factor averages at most one, ends the round badly
with chance at most `versionBound · a`. -/
theorem site_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hL : 2 ≤ C.maxLeaves)
    (ha : 0 < C.testLevel) (ha1 : C.testLevel < 1) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j)
    (W : Level5State α → ℝ) (f : Level5State α → FreeMonoid α → ℝ) (ok : Level5State α → Prop)
    (Bad : Level5Ending α → Level5State α → Prop)
    (hf0 : ∀ s x, 0 ≤ f s x) (hf : ∀ s, ok s → ∫⁻ x, ENNReal.ofReal (f s x) ∂D ≤ 1)
    (hfresh : ∀ T E v, W (freshState T E v) = 1)
    (hkeep : ∀ s x, CountInv s → ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember →
      W (keepProbe C cut s x) = W s * f s x ∧ (ok (keepProbe C cut s x) ↔ ok s))
    (htoo : ∀ s, ¬ Bad .tooBig s)
    (hbad : ∀ s x e, CountInv s → ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember →
      Bad e (keepProbe C cut s x) →
      runTests C (keepProbe C cut s x) = some e → ok s ∧ 1 ≤ C.testLevel * W (keepProbe C cut s x))
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C cut) Bad initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set Q := EvL C cut ok f C.testLevel
  have hsub : {xs : Fin T → FreeMonoid α | RunEnds (applyProbe C cut) Bad initialState
      (List.ofFn xs)} ⊆ ⋃ v ∈ Finset.range (versionBound C (Fintype.card α)),
        {xs | SkipV C cut Q v initialState (List.ofFn xs)} := by
    intro xs hxs
    have key := bad_end_skip C cut (RunInv C) (fun s s' x hs h => runInv_step C cut hs h) W f ok
      C.testLevel Bad
      (fun s s' x hs h hv => by
        rcases applyProbe_inl_shape C cut h with ⟨-, T, E, he⟩ | he
        · rw [he]; exact hfresh _ _ _
        · exact absurd (by rw [he]; rfl) hv)
      (fun s s' x hs h hv => by
        have hm := not_member_of_inl C cut h hv
        rcases applyProbe_inl_shape C cut h with ⟨hlt, -⟩ | he
        · omega
        · rw [he]; exact hkeep s x hs.2 hm)
      (fun s s' x e hs h hb => by
        have hm := not_member_of_inr C cut ha ha1 hn₀ hexc h fun he => htoo _ (he ▸ hb)
        rcases applyProbe_inr_shape C cut ha ha1 hn₀ hexc h with rfl | ⟨rfl, hr⟩
        · exact absurd hb (htoo _)
        · obtain ⟨hok, h1⟩ := hbad s x e hs.2 hm hb hr
          exact ⟨hok, by rw [← (hkeep s x hs.2 hm).1]; exact h1⟩)
      (List.ofFn xs) initialState (runInv_start C hL) hxs
    have hW : W initialState = 1 := hfresh _ _ _
    obtain ⟨v, hsk⟩ : ∃ v, SkipV C cut Q v initialState (List.ofFn xs) := by
      rcases key with ⟨v, -, hsk⟩ | hev
      · exact ⟨v, hsk⟩
      · rw [hW, mul_one] at hev
        refine ⟨0, ?_⟩
        rcases hl : List.ofFn xs with _ | ⟨y, ys⟩ <;> rw [hl] at hev
        · exact hev.elim
        · simp only [SkipV, show (initialState : Level5State α).version = 0 from rfl, if_true]
          exact hev
    obtain ⟨s', hs', hv⟩ := skipV_runInv C cut Q v _ _ (runInv_start C hL) hsk
    exact Set.mem_biUnion (Finset.mem_range.2 (hv ▸ version_lt C hs'.1 hL)) hsk
  refine (measure_mono hsub).trans ((measure_biUnion_finset_le _ _).trans ?_)
  have hQ : ∀ v ∈ Finset.range (versionBound C (Fintype.card α)),
      (Measure.pi fun _ : Fin T => D) {xs | SkipV C cut Q v initialState (List.ofFn xs)}
        ≤ ENNReal.ofReal C.testLevel := fun v _ =>
    skipV_le C cut D (RunInv C) (fun s s' x hs h => runInv_step C cut hs h) Q v
      (fun T' s _ _ => evL_le C cut D ok f hf0 hf T' C.testLevel s ha.le) T _ (runInv_start C hL)
  refine (Finset.sum_le_sum hQ).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, ← ENNReal.ofReal_natCast,
    ← ENNReal.ofReal_mul (Nat.cast_nonneg _)]

/-- A sequential test whose counts start afresh at each hypothesis and, while it keeps, advance at
the probes `act` picks, with a hit at those `hit` picks: where its factor averages at most one at
`ok` hypotheses, it fires at one with chance at most `versionBound · a` over the round. -/
theorem ratio_site_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hL : 2 ≤ C.maxLeaves)
    (ha : 0 < C.testLevel) (ha1 : C.testLevel < 1) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) {p₀ p₁ : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁)
    (hp₁1 : p₁ < 1) (N H : Level5State α → ℕ) (act hit : Level5State α → FreeMonoid α → Bool)
    (ok : Level5State α → Prop) (Bad : Level5Ending α → Level5State α → Prop)
    (hf : ∀ s, ok s → ∫⁻ x, ENNReal.ofReal
      (if act s x then ratioFactor p₀ p₁ (hit s x) else 1) ∂D ≤ 1)
    (hfresh : ∀ T E v, N (freshState T E v) = 0 ∧ H (freshState T E v) = 0)
    (hle : ∀ s, CountInv s → H s ≤ N s)
    (hkeep : ∀ s x, CountInv s → ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember →
      N (keepProbe C cut s x) = N s + (if act s x then 1 else 0)
      ∧ H (keepProbe C cut s x) = H s + (if hit s x then 1 else 0) ∧ (hit s x → act s x)
      ∧ (ok (keepProbe C cut s x) ↔ ok s))
    (htoo : ∀ s, ¬ Bad .tooBig s)
    (hbad : ∀ s x e, CountInv s → ¬ (probeBy cut s.tree s.edges C.walkStart x).IsMember →
      Bad e (keepProbe C cut s x) →
      runTests C (keepProbe C cut s x) = some e →
        ok s ∧ seqRate p₀ p₁ C.testLevel (N (keepProbe C cut s x)) (H (keepProbe C cut s x)))
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C cut) Bad initialState (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  have hrf : ∀ b, 0 ≤ ratioFactor p₀ p₁ b := fun b => by
    unfold ratioFactor; split
    · exact div_nonneg hp₁.le hp₀.le
    · exact div_nonneg (by linarith) (by linarith)
  refine site_le C cut D hL ha ha1 hn₀ hexc (fun s => likelihoodRatio p₀ p₁ (N s) (H s))
    (fun s x => if act s x then ratioFactor p₀ p₁ (hit s x) else 1) ok Bad
    (fun s x => by split <;> [exact hrf _; norm_num]) hf
    (fun T E v => by simp only [(hfresh T E v).1, (hfresh T E v).2, likelihoodRatio_zero])
    (fun s x hs hm => ⟨?_, (hkeep s x hs hm).2.2.2⟩) htoo (fun s x e hs hm hb hr => ?_) T
  · obtain ⟨h1, h2, h3, -⟩ := hkeep s x hs hm
    rw [h1, h2]
    by_cases hact : act s x
    · simp only [hact, if_true]
      rw [likelihoodRatio_succ (hle s hs), mul_comm]
    · have hh : hit s x = false := by
        by_contra hc
        exact hact (h3 (by simpa using hc))
      simp [hact, hh]
  · obtain ⟨hok, hs'⟩ := hbad s x e hs hm hb hr
    refine ⟨hok, ?_⟩
    unfold seqRate at hs'
    rw [div_le_iff₀ ha] at hs'
    linarith

end Site



section Tests

open scoped Classical

variable (C : Level5Settings) (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
  (rd : FreeMonoid α → ARU)

open scoped Classical in
theorem runTests_start' {s : Level5State α} (h : runTests C s = some .harvestStart) :
    seqRate C.startUndecidedRate C.startAltRate C.testLevel s.probes s.startUndecided.length := by
  unfold runTests at h
  split_ifs at h with h1 <;> first | exact h1 | simp at h

open scoped Classical in
theorem runTests_pt' {s : Level5State α} (h : runTests C s = some .harvestPT) :
    C.searchShare * s.probes ≤ s.searches ∧ C.firstLook ≤ s.searches
      ∧ seqRate C.middleUndecidedRate C.middleAltRate C.testLevel s.searches s.middles.length := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 <;> first | exact h3 | simp at h

open scoped Classical in
theorem runTests_success' {s : Level5State α} (h : runTests C s = some .success) :
    seqRate C.disagreementRate C.disagreementAltRate C.testLevel s.probes s.searches := by
  unfold runTests at h
  split_ifs at h with h1 h2 h3 h4 <;> first | exact h4 | simp at h

open scoped Classical in
theorem runTests_harvest' {s : Level5State α} {e : List Bool × α}
    (h : runTests C s = some (.harvest e)) :
    e.1 ∈ s.tree.leaves ∧ C.edgeTraffic * s.probes ≤ s.edgeReads e.1 e.2
      ∧ C.edgeExcess (s.edgeReads e.1 e.2)
        ≤ ((s.edgeUndecided e.1 e.2).length : ℝ) - C.edgeUndecidedRate * s.edgeReads e.1 e.2 := by
  unfold runTests at h
  split_ifs at h with h1 h2
  · simp at h
  · simp only [Option.some.injEq, Level5Ending.harvest.injEq] at h
    subst h
    exact h2.choose_spec
  all_goals simp at h

/-- The probes whose start reads undecided at `T`. -/
def startSet (T : DTree α) : Set (FreeMonoid α) :=
  {x | startHarvestBy (fun z => (rd z).cut) T C.walkStart x ≠ []}

/-- A harvest at the start of a hypothesis whose start reads undecided at most `θs` of the
time. -/
def StartBad (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .harvestStart ∧ D.real (startSet C rd s.tree) ≤ C.startUndecidedRate

theorem startHarvestBy_length (T : DTree α) (x : FreeMonoid α) :
    (startHarvestBy (fun z => (rd z).cut) T C.walkStart x).length
      = if decide (x ∈ startSet C rd T) then 1 else 0 := by
  unfold startSet startHarvestBy
  by_cases h1 : x.toList.length < C.walkStart
  · simp [h1]
  · rcases hs : T.siftCut (fun z => (rd z).cut) (prefixOf x C.walkStart) with p | b <;>
      simp [h1, hs]

theorem start_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (StartBad C D rd) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  have hp₁ : 0 < C.startAltRate := hSq.θs_pos.trans hSq.θs_alt
  refine ratio_site_le C _ D hL hSq.a_pos hSq.a_lt hn₀ hexc hSq.θs_pos
    (hSq.θs_alt.trans hSq.θs_alt_lt) hp₁ hSq.θs_alt_lt (fun s => s.probes)
    (fun s => s.startUndecided.length) (fun _ _ => true)
    (fun s x => decide (x ∈ startSet C rd s.tree))
    (fun s => D.real (startSet C rd s.tree) ≤ C.startUndecidedRate) (StartBad C D rd)
    (fun s hs => ?_) (fun T E v => ⟨rfl, rfl⟩) (fun s hs => hs.1)
    (fun s x hs _ => ⟨by simp [keepProbe, Level5State.countProbe], ?_, fun _ => rfl, Iff.rfl⟩)
    (fun s h => by simp [StartBad] at h) (fun s x e hs _ hb hr => ?_) T
  · simp only [if_true]
    refine lintegral_ratioFactor_le D (· ∈ startSet C rd s.tree) hSq.θs_pos
      (hSq.θs_alt.trans hSq.θs_alt_lt) hp₁.le hSq.θs_alt_lt.le ?_
    rw [Set.setOf_mem_eq]
    exact mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith [hSq.θs_alt])
  · simp only [keepProbe, Level5State.countProbe, List.length_append]
    rw [startHarvestBy_length]
  · obtain ⟨rfl, hok⟩ := hb
    exact ⟨hok, runTests_start' C hr⟩


/-- The probes that search or learn, at a tree and curried edges. -/
def missAt (T : DTree α) (edges : EdgeTable α) : Set (FreeMonoid α) :=
  searchingProbes rd C.walkStart ⟨T, Function.uncurry edges⟩

/-- A harvest's success at a hypothesis searching or learning on at least `εd` of the probes. -/
def SuccessBad (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .success ∧ C.disagreementRate ≤ D.real (missAt C rd s.tree s.edges)

theorem keep_searches (s : Level5State α) (x : FreeMonoid α) :
    (keepProbe C (fun z => (rd z).cut) s x).searches
      = s.searches + if decide (x ∈ searchAt rd C.walkStart s.tree s.edges) then 1 else 0 := by
  simp only [keepProbe]
  congr 1
  by_cases h : (probeBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x).IsSearch
  · rw [if_pos h, if_pos (by simpa using mem_searchAt.2 h)]
  · rw [if_neg h, if_neg (by simpa using fun h' => h (mem_searchAt.1 h'))]

theorem keep_misses (s : Level5State α) (x : FreeMonoid α)
    (hm : ¬ (probeBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x).IsMember) :
    (keepProbe C (fun z => (rd z).cut) s x).searches
      = s.searches + if decide (x ∈ missAt C rd s.tree s.edges) then 1 else 0 := by
  rw [keep_searches]
  congr 2
  have e : Function.curry (Function.uncurry s.edges) = s.edges := rfl
  simp only [missAt, searchingProbes, searchAt_eq, Set.mem_setOf_eq, e]
  rcases ho : probeBy (fun z => (rd z).cut) s.tree s.edges C.walkStart x with
    _ | _ | _ | _ | _ | u <;> simp_all [WalkOutcome.IsMember]

theorem success_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (SuccessBad C D rd) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  refine ratio_site_le C _ D hL hSq.a_pos hSq.a_lt hn₀ hexc hSq.εd_pos hSq.εd_lt hSq.εd_alt_pos
    (hSq.εd_alt.trans hSq.εd_lt) (fun s => s.probes) (fun s => s.searches) (fun _ _ => true)
    (fun s x => decide (x ∈ missAt C rd s.tree s.edges))
    (fun s => C.disagreementRate ≤ D.real (missAt C rd s.tree s.edges))
    (SuccessBad C D rd) (fun s hs => ?_) (fun T E v => ⟨rfl, rfl⟩) (fun s hs => hs.2.2.1)
    (fun s x hs hm => ⟨by simp [keepProbe, Level5State.countProbe], keep_misses C rd s x hm,
      fun _ => rfl, Iff.rfl⟩)
    (fun s h => by simp [SuccessBad] at h) (fun s x e hs _ hb hr => ?_) T
  · simp only [if_true]
    refine lintegral_ratioFactor_le D (· ∈ missAt C rd s.tree s.edges) hSq.εd_pos
      hSq.εd_lt hSq.εd_alt_pos.le (hSq.εd_alt.trans hSq.εd_lt).le ?_
    rw [Set.setOf_mem_eq]
    exact mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith [hSq.εd_alt])
  · obtain ⟨rfl, hok⟩ := hb
    exact ⟨hok, runTests_success' C hr⟩

/-- A harvest of the middles at a hypothesis searching on at most `φpt/2` of the probes. -/
def GateBad (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .harvestPT ∧ D.real (searchAt rd C.walkStart s.tree s.edges) ≤ C.searchShare / 2

theorem gate_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) (hgate : 6 * Real.log (1 / C.testLevel) ≤ C.firstLook)
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (GateBad C D rd) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  have hφ := hSq.φ_pos
  have hφ1 := hSq.φ_lt
  refine ratio_site_le C _ D hL hSq.a_pos hSq.a_lt hn₀ hexc (p₀ := C.searchShare / 2)
    (p₁ := C.searchShare) (by linarith) (by linarith) hφ hφ1 (fun s => s.probes)
    (fun s => s.searches) (fun _ _ => true)
    (fun s x => decide (x ∈ searchAt rd C.walkStart s.tree s.edges))
    (fun s => D.real (searchAt rd C.walkStart s.tree s.edges) ≤ C.searchShare / 2)
    (GateBad C D rd) (fun s hs => ?_) (fun T E v => ⟨rfl, rfl⟩) (fun s hs => hs.2.2.1)
    (fun s x hs _ => ⟨by simp [keepProbe, Level5State.countProbe], keep_searches C rd s x,
      fun _ => rfl, Iff.rfl⟩)
    (fun s h => by simp [GateBad] at h) (fun s x e hs _ hb hr => ?_) T
  · simp only [if_true]
    refine lintegral_ratioFactor_le D (· ∈ searchAt rd C.walkStart s.tree s.edges)
      (by linarith) (by linarith) hφ.le hφ1.le ?_
    rw [Set.setOf_mem_eq]
    exact mul_nonpos_of_nonpos_of_nonneg (by linarith) (by linarith)
  · obtain ⟨rfl, hok⟩ := hb
    refine ⟨hok, ?_⟩
    obtain ⟨hg, hf, -⟩ := runTests_pt' C hr
    have hci := countInv_keep C (fun z => (rd z).cut) hs x
    exact seqRate_share hφ hφ1 hSq.a_pos hgate hf hci.2.2.1 hg

/-- The probes whose search stops at an undecided middle. -/
def ptSet (T : DTree α) (E : EdgeTable α) : Set (FreeMonoid α) :=
  {x | middleHarvestBy (fun z => (rd z).cut) T E C.walkStart x ≠ []}

/-- A harvest of the middles at a hypothesis whose searches stop at an undecided middle at most
`θpt` of the time. -/
def MiddlesBad (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .harvestPT ∧ D.real (ptSet C rd s.tree s.edges)
    ≤ C.middleUndecidedRate * D.real (searchAt rd C.walkStart s.tree s.edges)

theorem ptSet_sub (T : DTree α) (E : EdgeTable α) : ptSet C rd T E ⊆ searchAt rd C.walkStart T E := by
  intro x hx
  rw [mem_searchAt]
  have := middleHarvestBy_length (fun z => (rd z).cut) T E C.walkStart x
  simp only [ptSet, Set.mem_setOf_eq] at hx
  rcases ho : probeBy (fun z => (rd z).cut) T E C.walkStart x with _ | _ | _ | _ | _ | _ <;>
    rw [ho] at this <;> simp_all [WalkOutcome.IsSearch]

/-- At hits `A` among the active draws `B`, at most `p₀` of them, a factor `p₁/p₀` at a hit,
`(1 − p₁)/(1 − p₀)` at another active draw and `1` elsewhere averages at most one. -/
theorem lintegral_actFactor_le {A B : Set (FreeMonoid α)} (hAB : A ⊆ B) {p₀ p₁ : ℝ}
    (hp₀ : 0 < p₀) (hp₀₁ : p₀ < p₁) (hp₁1 : p₁ < 1) (hs : D.real A ≤ p₀ * D.real B) :
    ∫⁻ x, ENNReal.ofReal (if decide (x ∈ B) then ratioFactor p₀ p₁ (decide (x ∈ A)) else 1) ∂D
      ≤ 1 := by
  set a₁ : ℝ := p₁ / p₀
  set a₂ : ℝ := (1 - p₁) / (1 - p₀)
  have ha₁ : 0 ≤ a₁ := div_nonneg (by linarith) hp₀.le
  have ha₂ : 0 ≤ a₂ := div_nonneg (by linarith) (by linarith)
  have hpt : ∀ x, ENNReal.ofReal (if decide (x ∈ B) then ratioFactor p₀ p₁ (decide (x ∈ A)) else 1)
      = A.indicator (fun _ => ENNReal.ofReal a₁) x + (B \ A).indicator (fun _ => ENNReal.ofReal a₂) x
        + Bᶜ.indicator (fun _ => 1) x := by
    intro x
    unfold ratioFactor
    by_cases hxA : x ∈ A
    · have hxB := hAB hxA
      simp [hxA, hxB, Set.indicator, a₁]
    by_cases hxB : x ∈ B
    · simp [hxA, hxB, Set.indicator, a₂]
    · simp [hxA, hxB, Set.indicator]
  simp_rw [hpt]
  have hmeas : ∀ E : Set (FreeMonoid α), MeasurableSet E := fun E =>
    (Set.to_countable _).measurableSet
  rw [lintegral_add_left (measurable_of_countable _), lintegral_add_left
      (measurable_of_countable _), lintegral_indicator (hmeas _), lintegral_indicator (hmeas _),
    lintegral_indicator (hmeas _), setLIntegral_const, setLIntegral_const, setLIntegral_const,
    ← ofReal_measureReal, ← ofReal_measureReal, ← ofReal_measureReal,
    measureReal_compl (hmeas _), probReal_univ, measureReal_sdiff hAB (hmeas _),
    ← ENNReal.ofReal_mul ha₁, ← ENNReal.ofReal_mul ha₂, one_mul]
  have hA0 : 0 ≤ D.real A := measureReal_nonneg
  have hAB' : D.real A ≤ D.real B := measureReal_mono hAB
  have hB1 : D.real B ≤ 1 := measureReal_le_one
  rw [← ENNReal.ofReal_add (by positivity) (mul_nonneg ha₂ (by linarith)),
    ← ENNReal.ofReal_add (by positivity) (by linarith), ← ENNReal.ofReal_one]
  refine ENNReal.ofReal_le_ofReal ?_
  have e1 : p₀ * a₁ = p₁ := by simp only [a₁]; field_simp
  have e2 : (1 - p₀) * a₂ = 1 - p₁ := by
    simp only [a₂]; field_simp [show (1 - p₀) ≠ 0 by linarith]
  have hgap : a₂ ≤ a₁ := by
    simp only [a₁, a₂]
    rw [div_le_div_iff₀ (by linarith) hp₀]
    nlinarith
  have h1 : D.real A * a₁ + (D.real B - D.real A) * a₂
      ≤ p₀ * D.real B * a₁ + (D.real B - p₀ * D.real B) * a₂ := by
    nlinarith [mul_le_mul_of_nonneg_right hs (sub_nonneg.2 hgap)]
  have h2 : p₀ * D.real B * a₁ + (D.real B - p₀ * D.real B) * a₂ = D.real B := by
    have : p₀ * D.real B * a₁ + (D.real B - p₀ * D.real B) * a₂
        = D.real B * (p₀ * a₁) + D.real B * ((1 - p₀) * a₂) := by ring
    rw [this, e1, e2]; ring
  linarith

theorem middles_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (MiddlesBad C D rd) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  refine ratio_site_le C _ D hL hSq.a_pos hSq.a_lt hn₀ hexc hSq.θpt_pos
    (hSq.θpt_alt.trans hSq.θpt_alt_lt) (hSq.θpt_pos.trans hSq.θpt_alt) hSq.θpt_alt_lt
    (fun s => s.searches) (fun s => s.middles.length)
    (fun s x => decide (x ∈ searchAt rd C.walkStart s.tree s.edges))
    (fun s x => decide (x ∈ ptSet C rd s.tree s.edges))
    (fun s => D.real (ptSet C rd s.tree s.edges)
      ≤ C.middleUndecidedRate * D.real (searchAt rd C.walkStart s.tree s.edges))
    (MiddlesBad C D rd) (fun s hs => lintegral_actFactor_le D (ptSet_sub C rd _ _) hSq.θpt_pos
      hSq.θpt_alt hSq.θpt_alt_lt hs) (fun T E v => ⟨rfl, rfl⟩) (fun s hs => hs.2.1)
    (fun s x hs _ => ⟨keep_searches C rd s x, ?_, fun h => ?_, Iff.rfl⟩)
    (fun s h => by simp [MiddlesBad] at h) (fun s x e hs _ hb hr => ?_) T
  · simp only [keepProbe, Level5State.countProbe, List.length_append]
    congr 1
    have := middleHarvestBy_length (fun z => (rd z).cut) s.tree s.edges C.walkStart x
    unfold ptSet
    rcases middleHarvestBy_cases (fun z => (rd z).cut) s.tree s.edges C.walkStart x with h | ⟨b, h⟩ <;>
      simp [h]
  · simp only [decide_eq_true_eq] at h ⊢
    exact ptSet_sub C rd _ _ h
  · obtain ⟨rfl, hok⟩ := hb
    exact ⟨hok, (runTests_pt' C hr).2.2⟩


/-- The edge out of a tree's `i`-th leaf by `c`. -/
def edgeOf (T : DTree α) (i : ℕ) (c : α) : List Bool × α := (T.leaves.getD i [], c)

/-- A probe's mean count of undecided strings at the edge `e`. -/
noncomputable def undMean (T : DTree α) (E : EdgeTable α) (e : List Bool × α) : ℝ :=
  ∫ x, ((edgeHarvestBy (fun z => (rd z).cut) T E C.walkStart x e).length : ℝ) ∂D

/-- A probe's mean count of first reads at the edge `e`. -/
noncomputable def readMean (T : DTree α) (E : EdgeTable α) (e : List Bool × α) : ℝ :=
  ∫ x, (edgeReadsBy (fun z => (rd z).cut) T E C.walkStart x e : ℝ) ∂D

/-- The `i`-th leaf's edge by `c`: its undecided strings less twice their running mean,
exponentiated. -/
noncomputable def undW (L i : ℕ) (c : α) (s : Level5State α) : ℝ :=
  Real.exp ((((s.edgeUndecided (edgeOf s.tree i c).1 c).length : ℝ)
    - 2 * s.probes * undMean C D rd s.tree s.edges (edgeOf s.tree i c)) / (2 * (L + 1)))

/-- Half its reads' running mean less its reads, exponentiated. -/
noncomputable def readW (L i : ℕ) (c : α) (s : Level5State α) : ℝ :=
  Real.exp ((s.probes * readMean C D rd s.tree s.edges (edgeOf s.tree i c) / 2
    - s.edgeReads (edgeOf s.tree i c).1 c) / (2 * ((L + 1) * C.maxLeaves)))

/-- A harvest at the `i`-th leaf's edge by `c`, its undecided strings past twice their running
mean by `2 (L + 1) log (1/a)`. -/
def UndBad (L i : ℕ) (c : α) (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .harvest (edgeOf s.tree i c) ∧ 1 ≤ C.testLevel * undW C D rd L i c s

/-- A harvest at the `i`-th leaf's edge by `c`, its reads short of half their running mean by
`2 (L + 1) maxLeaves log (1/a)`. -/
def ReadBad (L i : ℕ) (c : α) (e : Level5Ending α) (s : Level5State α) : Prop :=
  e = .harvest (edgeOf s.tree i c) ∧ s.tree.leaves.length ≤ C.maxLeaves
    ∧ 1 ≤ C.testLevel * readW C D rd L i c s

theorem und_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (i : ℕ) (c : α)
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (UndBad C D rd L i c) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  refine site_le C cut D hL hSq.a_pos hSq.a_lt hn₀ hexc (undW C D rd L i c)
    (fun s x => Real.exp ((((edgeHarvestBy cut s.tree s.edges C.walkStart x
      (edgeOf s.tree i c)).length : ℝ) - 2 * undMean C D rd s.tree s.edges (edgeOf s.tree i c)) / (2 * (L + 1))))
    (fun _ => True) (UndBad C D rd L i c) (fun s x => (Real.exp_pos _).le) (fun s _ => ?_)
    (fun T E v => by simp [undW, freshState]) (fun s x hs _ => ⟨?_, Iff.rfl⟩)
    (fun s h => by simp [UndBad] at h) (fun s x e hs _ hb hr => ⟨trivial, hb.2⟩) T
  · have h0 : 0 ≤ undMean C D rd s.tree s.edges (edgeOf s.tree i c) :=
      integral_nonneg fun _ => Nat.cast_nonneg _
    have := mgf_step D
      (fun x => (edgeHarvestBy cut s.tree s.edges C.walkStart x (edgeOf s.tree i c)).length)
      (fun _ => 0) (θ := 0) (L := (L : ℝ) + 1) (R := 0) le_rfl (by positivity) le_rfl
      (by
        filter_upwards [hlen] with x hx
        refine ⟨?_, by simp⟩
        have := edgeHarvestBy_length_le cut s.tree s.edges C.walkStart x (edgeOf s.tree i c)
        exact_mod_cast (by omega : (edgeHarvestBy cut s.tree s.edges C.walkStart x
          (edgeOf s.tree i c)).length ≤ L + 1))
      h0 (by simp only [zero_mul, zero_add]; exact le_rfl)
    refine le_trans (le_of_eq (lintegral_congr fun x => ?_)) this
    congr 2
    push_cast
    ring
  · simp only [undW, keepProbe, Level5State.countProbe, List.length_append, Nat.cast_add,
      Nat.cast_one]
    rw [← Real.exp_add]
    congr 1
    simp only [edgeOf]
    ring

theorem read_le (hSq : C.SeqOK) (hL : 2 ≤ C.maxLeaves) (hn₀ : 1 ≤ C.firstLook)
    (hexc : ∀ j, 0 < C.edgeExcess j) {L : ℕ} (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (i : ℕ) (c : α)
    (T : ℕ) :
    (Measure.pi fun _ : Fin T => D)
        {xs | RunEnds (applyProbe C fun z => (rd z).cut) (ReadBad C D rd L i c) initialState
          (List.ofFn xs)}
      ≤ ENNReal.ofReal (versionBound C (Fintype.card α) * C.testLevel) := by
  set cut : FreeMonoid α → Option Bool := fun z => (rd z).cut
  refine site_le C cut D hL hSq.a_pos hSq.a_lt hn₀ hexc (readW C D rd L i c)
    (fun s x => Real.exp ((readMean C D rd s.tree s.edges (edgeOf s.tree i c) / 2
      - edgeReadsBy cut s.tree s.edges C.walkStart x (edgeOf s.tree i c))
        / (2 * ((L + 1) * C.maxLeaves))))
    (fun s => s.tree.leaves.length ≤ C.maxLeaves) (ReadBad C D rd L i c)
    (fun s x => (Real.exp_pos _).le) (fun s hs => ?_)
    (fun T E v => by simp [readW, freshState]) (fun s x hs _ => ⟨?_, Iff.rfl⟩)
    (fun s h => by simp [ReadBad] at h) (fun s x e hs _ hb hr => ⟨hb.2.1, hb.2.2⟩) T
  · have hB : (0 : ℝ) < (L + 1) * C.maxLeaves := by
      have : (0 : ℝ) < C.maxLeaves := by exact_mod_cast (by omega : 0 < C.maxLeaves)
      positivity
    have := mgf_lower D
      (fun x => edgeReadsBy cut s.tree s.edges C.walkStart x (edgeOf s.tree i c)) hB (by
        filter_upwards [hlen] with x hx
        have h1 := edgeReadsBy_le cut s.tree s.edges C.walkStart x (edgeOf s.tree i c)
        have hm := midfixes_card s.tree
        have : edgeReadsBy cut s.tree s.edges C.walkStart x (edgeOf s.tree i c)
            ≤ (L + 1) * C.maxLeaves := h1.trans (Nat.mul_le_mul (by omega) (by omega))
        exact_mod_cast this)
    refine le_trans (le_of_eq ?_) this
    rfl
  · simp only [readW, keepProbe, Level5State.countProbe, Nat.cast_add, Nat.cast_one]
    rw [← Real.exp_add]
    congr 1
    simp only [edgeOf]
    ring

end Tests

end OrthoDFA
