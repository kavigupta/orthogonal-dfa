import OrthoDFA.Proofs.KappaCount
import OrthoDFA.Proofs.RaceReads
import OrthoDFA.Proofs.EpsRound

/-!
# Level 5: the proof

Given the reads, the round fails to end well only where a stretch risks it, as at level 3, or
where some step reads a string wrong among what it reads (`ReadsWrong`), as at level 4
(`eps_round_le_of`). Given the probes, the rounds of what the steps read are each fixed by what
they and the earlier ones read, and while nothing read is wrong they read at most `Mb` strings,
`Mu` of them undecided (`count_le`); so by `RaceReadsBound` some step reads a string wrong with
chance at most `εw Mb + κ Mu` (`kappa_wrong_le`).
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open MeasureTheory Eps

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (C : Level3Settings)

omit [Fintype σ] in
/-- A run with a step reading a string wrong reads it in one of the rounds. -/
theorem anyB_rounds' (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s₀ : Level3State α),
      AnyB (applyProbe read C) (ReadsWrong M side read C) s₀ xs →
      ∃ j < xs.length, ∃ s x, stateAt C read s₀ xs j = some s ∧ xs[j]? = some x
        ∧ ReadsWrong M side read C s x
  | [], _, h => by simp [AnyB] at h
  | x :: xs, s₀, h => by
    rcases h with h | ⟨s', hs', h⟩
    · exact ⟨0, by simp, s₀, x, rfl, rfl, h⟩
    · obtain ⟨j, hj, s, y, h1, h2, h3⟩ := anyB_rounds' read xs s' h
      refine ⟨j + 1, by simp; omega, s, y, ?_, by simpa using h2, h3⟩
      simp only [stateAt, hs']
      exact h1

/-- Given probes no longer than `L`, some step reads a string wrong with chance at most
`εw Mb + κ Mu`. -/
theorem wrong_le' {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (U : σ → ℝ) (εw κ : ℝ) (R : KappaReadOracle μ M side U εw κ) (xs : List (FreeMonoid α))
    (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L) (hεw : 0 ≤ εw) (hκ : 0 ≤ κ)
    (hy0 : 0 < C.harvestRate) (hy1 : C.harvestRate ≤ 1 / 4) (hε : 0 ≤ C.disagreementRate)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    μ {ω | AnyB (applyProbe (R.read · ω) C) (ReadsWrong M side (R.read · ω) C) initialState xs}
      ≤ ENNReal.ofReal (εw * Mb (Fintype.card σ) (Fintype.card α) L xs.length
        + κ * Mu (Fintype.card σ) (Fintype.card α) C xs.length) := by
  classical
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z => {if side (M.eval z.toList) then .reject else .accept}
  set V : FreeMonoid α → Set ARU := fun _ => {.undecided}
  have hG : ∀ r j, (∀ i < j, ∀ z ∈ rounds' C xs i r, r z ∉ B z) ↔ Good' M side C xs r j := by
    intro r j; simp [Good', B]
  have hmain := raceReadsBound_holds μ R.read B V εw κ (fun i r => rounds' C xs i r)
    (Mb (Fintype.card σ) (Fintype.card α) L xs.length)
    (Mu (Fintype.card σ) (Fintype.card α) C xs.length) (2 * xs.length) R.measurable R.indep
    hεw hκ (fun z => by
      have h1 := R.wrong z
      have h2 := R.undecided z
      simp only [B, V, Set.mem_singleton_iff]
      rw [h2]; exact h1)
    (fun i r r' hg ha => rounds'_local M side C xs i r r' ((hG r i).1 hg) ha)
    (fun r => by
      have := (count_le M side xs r L hL hy0 hy1 hε hδ0 hδ1).2
      unfold readCount at this
      convert this using 2
      ext j; simp only [Finset.mem_filter]; exact and_congr_right fun _ => hG r j)
    (fun r => by
      have := (count_le M side xs r L hL hy0 hy1 hε hδ0 hδ1).1
      unfold undCount at this
      convert this using 2 with j
      · ext j; simp only [Finset.mem_filter]; exact and_congr_right fun _ => hG r j
      · congr 2; ext z; simp [V])
  refine (measure_mono fun ω hω => ?_).trans hmain
  obtain ⟨j, hj, s, x, h1, h2, z, hz, hw⟩ := anyB_rounds' M side C (R.read · ω) xs _ hω
  rcases Finset.mem_union.1 hz with hz | hz
  · exact ⟨2 * j, by omega, z, by rw [rounds'_even C xs _ h1 h2]; exact hz, by simpa [B] using hw⟩
  · exact ⟨2 * j + 1, by omega, z, by rw [rounds'_odd C xs _ h1 h2]; exact hz,
      by simpa [B] using hw⟩

/-- Over the probes and the reads, some step reads a string wrong with chance at most
`εw Mb + κ Mu`. -/
theorem pi_wrong_le' {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (U : σ → ℝ) (εw κ : ℝ) (R : KappaReadOracle μ M side U εw κ) (P : PrefixDistribution α)
    (hεw : 0 ≤ εw) (hκ : 0 ≤ κ) (hy0 : 0 < C.harvestRate) (hy1 : C.harvestRate ≤ 1 / 4)
    (hε : 0 ≤ C.disagreementRate) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {d | AnyB (applyProbe (R.read · ω) C) (ReadsWrong M side (R.read · ω) C) initialState
          (List.ofFn d)} ∂μ
      ≤ ENNReal.ofReal (εw * Mb (Fintype.card σ) (Fintype.card α) P.L C.probeBudget
        + κ * Mu (Fintype.card σ) (Fintype.card α) C C.probeBudget) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  refine lintegral_le_of_points μ _ _ _ fun d hd => ?_
  have hL : ∀ x ∈ List.ofFn d, x.toList.length ≤ P.L := by
    intro x hx
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    by_contra hlt
    apply hd
    have hpi : (Measure.pi fun _ : Fin C.probeBudget => P.D) {d}
        = ∏ j, P.D {d j} := by
      rw [← Measure.pi_pi]
      congr 1
      ext d'
      simp [funext_iff]
    rw [hpi]
    refine Finset.prod_eq_zero (Finset.mem_univ i) (measure_mono_null ?_
      (ae_iff.1 P.length_le))
    intro x hx
    rw [Set.mem_singleton_iff] at hx
    subst hx
    exact hlt
  have := wrong_le' M side C μ U εw κ R (List.ofFn d) P.L hL hεw hκ hy0 hy1 hε hδ0 hδ1
  rw [List.length_ofFn] at this
  exact this

theorem kappaRoundCorrect_holds : KappaRoundCorrect := by
  intro α _ _ σ _ M side U θ εw κ Ω _ μ _ R P C p₀
  dsimp only
  intro hpre hθ hδ hθL hGy hyρ hρ1 hP hεw hκ hw
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1 with hS
  set K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) with hK
  set Λ := Real.log (16 * (K : ℝ) / δ) with hΛ
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
  push Not at hδ1
  have hp₀ : 0 < p₀ :=
    (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  have hK1 : (1 : ℝ) ≤ K := by
    have : 1 ≤ K := Nat.one_le_iff_ne_zero.2 (by positivity)
    exact_mod_cast this
  have hΛ0 : 0 < Λ := Real.log_pos (by rw [one_lt_div hδ]; linarith)
  have hΛ' : Λ = Real.log (8 * (K : ℝ) / (δ / 2)) := by
    rw [hΛ]; congr 1; field_simp; ring
  obtain ⟨kn, hA, hGe, hηm, hsum⟩ := knobs_exist (σ := σ) (α := α) C θ p₀ Λ P.L S hS hθ hθL hp₀
    hΛ0 hδ hδ1 hGy hyρ hρ1 hP
  have hy0 : 0 < C.harvestRate := by
    have : 0 < 2 * (9 * θ * (nQ + 1) + 2 * p₀ * Λ) := by positivity
    linarith
  have hε0 : 0 ≤ C.disagreementRate := by linarith
  have hy1 : C.harvestRate ≤ 1 / 4 := by linarith
  refine (eps_round_le_of C M side U θ _ μ R.read R.measurable R.indep R.undecided
    (fun rd => ReadsWrong M side rd C)
    (fun rd _ _ _ hs hg hx h hne => kappa_step_change M side rd C hs hg hx h hne) P _ p₀ kn hA
    hp₀ hpre).trans ?_
  refine (add_le_add le_rfl (pi_wrong_le' M side C μ U εw κ R P hεw hκ hy0 hy1 hε0 hδ
    hδ1.le)).trans ?_
  have h1 : tailRisk nQ nα p₀ (kn.Ge - 9 * θ * (nQ + 1)) = δ / 2 / 8 := by
    rw [hGe, show 9 * θ * (nQ + 1) + p₀ * Λ - 9 * θ * (nQ + 1) = p₀ * Λ by ring, hΛ']
    exact tailRisk_eq nQ nα hp₀ (by positivity)
  have h2 : tailRisk nQ nα p₀ kn.ηm = δ / 2 / 8 := by
    rw [hηm, hΛ']
    exact tailRisk_eq nQ nα hp₀ (by positivity)
  have hround : roundBoundAt (α := α) σ θ p₀ P.L C kn ≤ 7 * δ / 8 := by
    unfold roundBoundAt
    linarith
  have hSst : Level2.stretches nQ nα = S := rfl
  have hcost := sum_costAt_le C S hδ hδ1.le
  have hMu : Mu nQ nα C C.probeBudget ≤ 3 * S * Real.log (8 * S * (S + 1) / δ)
      + (3 * C.harvestRate + C.disagreementRate / 6) * C.probeBudget + 1
      + nQ * (nQ + 1) ^ 2 * nα := by
    unfold Mu wSplit
    rw [hSst]
    push_cast
    nlinarith
  have hMb : Mb nQ nα P.L C.probeBudget = C.probeBudget * (P.L + 1) * nQ
      + nQ * (nQ + 1) ^ 2 * nα := by
    unfold Mb wSplit
    push_cast
    ring
  have hwr : εw * Mb nQ nα P.L C.probeBudget + κ * Mu nQ nα C C.probeBudget ≤ δ / 8 := by
    rw [hMb]
    have := mul_le_mul_of_nonneg_left hMu hκ
    linarith
  calc _ ≤ ENNReal.ofReal (7 * δ / 8) + ENNReal.ofReal (δ / 8) := by
        gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1
        ring

end Kappa

end Random

end OrthoDFA
