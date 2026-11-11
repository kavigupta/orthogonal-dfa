import RLStar.Proofs.RandomCounts

/-!
# The sequential tests on counts missing a few draws

A test counts a sublist `l` of the first `n` draws, missing at most `d` of them. Each test fires
where a sum `Σ_{x ∈ l} g(x)` of per-draw log factors reaches `log(1/a)`, and with `g ≥ −c` the sum
over all the first `n` draws is at least that over `l` less `c d` (`sum_sublist_ge`). So it fires,
at some look, with chance at most `a e^{c d}` (`pi_sum_del`), where Ville bounds the full sum. A
test silent on `l`, with `g ≤ c'`, is silent on the first `n` draws at level `a e^{−c' d}`
(`sum_sublist_le`).
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

section Sums

variable {X : Type*} (g : X → ℝ)

theorem sum_sublist_ge {c : ℝ} (hlow : ∀ x, -c ≤ g x) {l L : List X} (h : l.Sublist L) :
    (l.map g).sum - c * ((L.length : ℝ) - l.length) ≤ (L.map g).sum := by
  induction h with
  | slnil => simp
  | cons x _ ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    linarith [hlow x]
  | cons₂ x _ ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons, Nat.cast_add, Nat.cast_one]
    linarith

theorem sum_sublist_le {c : ℝ} (hup : ∀ x, g x ≤ c) {l L : List X} (h : l.Sublist L) :
    (L.map g).sum ≤ (l.map g).sum + c * ((L.length : ℝ) - l.length) := by
  have := sum_sublist_ge (fun x => -g x) (c := c) (fun x => by linarith [hup x]) h
  simp only [List.sum_map_neg] at this
  linarith

end Sums

section Draws

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (D : Measure X) [IsProbabilityMeasure D]

/-- Ville's inequality on sublists: over fresh draws, a sum with `g ≥ −c` whose exponentiated step
averages at most `1` reaches `a`, over some sublist of the first `n` draws missing at most `d`, with
chance at most `exp(−a + c d)`. -/
theorem pi_sum_del (g : X → ℝ) {c : ℝ} (hc : 0 ≤ c) (hlow : ∀ x, -c ≤ g x)
    (hg : ∫⁻ x, ENNReal.ofReal (Real.exp (g x / 1)) ∂D ≤ 1) (N d : ℕ) (a : ℝ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | ∃ n ≤ N, ∃ l : List X, l.Sublist ((List.ofFn xs).take n) ∧ n ≤ l.length + d
          ∧ a ≤ (l.map g).sum}
      ≤ ENNReal.ofReal (Real.exp (-a + c * d)) := by
  have hsub : {xs : Fin N → X | ∃ n ≤ N, ∃ l : List X, l.Sublist ((List.ofFn xs).take n)
        ∧ n ≤ l.length + d ∧ a ≤ (l.map g).sum}
      ⊆ {xs | ∃ n ≤ N, a - c * d ≤ sumFirst g n (List.ofFn xs)} := by
    rintro xs ⟨n, hn, l, hl, hld, ha⟩
    refine ⟨n, hn, ?_⟩
    have hlen : ((List.ofFn xs).take n).length = n := by simp [hn]
    have h1 := sum_sublist_ge g hlow hl
    rw [hlen] at h1
    have h2 : ((n : ℝ) - l.length) ≤ d := by
      have : (n : ℝ) ≤ l.length + d := by exact_mod_cast hld
      linarith
    simp only [sumFirst]
    nlinarith
  refine (measure_mono hsub).trans ((pi_sum_any D g one_pos hg N _).trans (le_of_eq ?_))
  congr 2
  ring

/-- A sequential test fires on a list where its sum of log factors reaches `log(1/a)`. -/
theorem seqRate_list_iff (f : X → Bool) {p₀ p₁ a : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1)
    (hp₁ : 0 < p₁) (hp₁1 : p₁ < 1) (ha : 0 < a) (l : List X) :
    seqRate p₀ p₁ a l.length (l.countP f) ↔ Real.log (1 / a)
      ≤ (l.map fun x => if f x then Real.log (p₁ / p₀)
          else Real.log ((1 - p₁) / (1 - p₀))).sum := by
  have hc : l.countP f ≤ l.length := List.countP_le_length
  have hlr : 0 < likelihoodRatio p₀ p₁ l.length (l.countP f) := by
    unfold likelihoodRatio
    have : 0 < (1 - p₁) / (1 - p₀) := div_pos (by linarith) (by linarith)
    have : 0 < p₁ / p₀ := div_pos hp₁ hp₀
    positivity
  unfold seqRate
  rw [← Real.log_le_log_iff (by positivity) hlr, log_likelihoodRatio hp₀ hp₀1 hp₁ hp₁1 hc,
    sum_map_ite, Nat.cast_sub hc]

/-- The larger of a test's two per-draw factors' inverses: `max(p₀/p₁, (1 − p₀)/(1 − p₁))`. -/
noncomputable def delFactor (p₀ p₁ : ℝ) : ℝ := max (p₀ / p₁) ((1 - p₀) / (1 - p₁))

theorem one_le_delFactor {p₀ p₁ : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁)
    (hp₁1 : p₁ < 1) : 1 ≤ delFactor p₀ p₁ := by
  unfold delFactor
  rcases le_total p₀ p₁ with h | h
  · exact le_max_of_le_right (by rw [le_div_iff₀ (by linarith)]; linarith)
  · exact le_max_of_le_left (by rw [le_div_iff₀ hp₁]; linarith)

/-- The sequential test, counting any sublist of the first `n` draws missing at most `d` of them,
fires at some look with chance at most `a Rᵈ`, `R = delFactor p₀ p₁`, on the far side of `p₀` from
`p₁`. -/
theorem seq_any_del (f : X → Bool) {p₀ p₁ a : ℝ} (hp₀ : 0 < p₀) (hp₀1 : p₀ < 1) (hp₁ : 0 < p₁)
    (hp₁1 : p₁ < 1) (ha : 0 < a) (hq : (D.real {x | f x} - p₀) * (p₁ - p₀) ≤ 0) (N d : ℕ) :
    (Measure.pi fun _ : Fin N => D)
        {xs | ∃ n ≤ N, ∃ l : List X, l.Sublist ((List.ofFn xs).take n) ∧ n ≤ l.length + d
          ∧ seqRate p₀ p₁ a l.length (l.countP f)}
      ≤ ENNReal.ofReal (a * delFactor p₀ p₁ ^ d) := by
  set g : X → ℝ := fun x => if f x then Real.log (p₁ / p₀) else Real.log ((1 - p₁) / (1 - p₀))
  set R := delFactor p₀ p₁
  have hR1 : 1 ≤ R := one_le_delFactor hp₀ hp₀1 hp₁ hp₁1
  have hR0 : 0 < R := by linarith
  have hlow : ∀ x, -Real.log R ≤ g x := by
    intro x
    simp only [g]
    split_ifs
    · rw [neg_le, ← Real.log_inv, inv_div]
      exact Real.log_le_log (div_pos hp₀ hp₁) (le_max_left _ _)
    · rw [neg_le, ← Real.log_inv, inv_div]
      exact Real.log_le_log (div_pos (by linarith) (by linarith)) (le_max_right _ _)
  have hg : ∫⁻ x, ENNReal.ofReal (Real.exp (g x / 1)) ∂D ≤ 1 := by
    have := lintegral_ratioFactor_le D (fun x => f x = true) hp₀ hp₀1 hp₁.le hp₁1.le
      (by simpa using hq)
    refine le_trans (le_of_eq (lintegral_congr fun x => ?_)) this
    simp only [g, div_one, ratioFactor]
    cases f x <;> simp [Real.exp_log (div_pos hp₁ hp₀),
      Real.exp_log (div_pos (by linarith : 0 < 1 - p₁) (by linarith : 0 < 1 - p₀))]
  have hsub : {xs : Fin N → X | ∃ n ≤ N, ∃ l : List X, l.Sublist ((List.ofFn xs).take n)
        ∧ n ≤ l.length + d ∧ seqRate p₀ p₁ a l.length (l.countP f)}
      ⊆ {xs | ∃ n ≤ N, ∃ l : List X, l.Sublist ((List.ofFn xs).take n) ∧ n ≤ l.length + d
          ∧ Real.log (1 / a) ≤ (l.map g).sum} := by
    rintro xs ⟨n, hn, l, hl, hld, h⟩
    exact ⟨n, hn, l, hl, hld, (seqRate_list_iff f hp₀ hp₀1 hp₁ hp₁1 ha l).1 h⟩
  refine (measure_mono hsub).trans ((pi_sum_del D g (Real.log_nonneg hR1) hlow hg N d _).trans (le_of_eq ?_))
  congr 1
  rw [Real.log_div one_ne_zero ha.ne', Real.log_one, zero_sub, neg_neg, Real.exp_add,
    Real.exp_log ha, ← Real.log_pow, Real.exp_log (pow_pos hR0 d)]

end Draws

end Random

theorem seqRateRobustSound : SeqRateRobustSound := by
  intro X _ _ _ D _ P p₀ p₁ a N d hp₀ hp₀1 hp₁ hp₁1 ha hq
  classical
  have hR1 := Random.one_le_delFactor hp₀ hp₀1 hp₁ hp₁1
  have h := Random.seq_any_del D (fun x => decide (P x)) hp₀ hp₀1 hp₁ hp₁1 ha
    (by simpa using hq) N d
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) h

end OrthoDFA
