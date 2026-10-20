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
chance at most `εw Mb + κ Mu` (`pi_wrong_le'`).
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
    (hy0 : 0 < C.harvestRate) (hyε : C.harvestRate ≤ C.disagreementRate / 48)
    (hε1 : C.disagreementRate < 1) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    μ {ω | AnyB (applyProbe (R.read · ω) C) (ReadsWrong M side (R.read · ω) C) initialState xs}
      ≤ ENNReal.ofReal (εw * Mb (Fintype.card σ) (Fintype.card α) L xs.length
        + κ * Mu (Fintype.card σ) (Fintype.card α) C) := by
  classical
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z => {if side (M.eval z.toList) then .reject else .accept}
  set V : FreeMonoid α → Set ARU := fun _ => {.undecided}
  have hG : ∀ r j, (∀ i < j, ∀ z ∈ rounds' C xs i r, r z ∉ B z) ↔ Good' M side C xs r j := by
    intro r j; simp [Good', B]
  have hmain := raceReadsBound_holds μ R.read B V εw κ (fun i r => rounds' C xs i r)
    (Mb (Fintype.card σ) (Fintype.card α) L xs.length)
    (Mu (Fintype.card σ) (Fintype.card α) C) (2 * xs.length) R.measurable R.indep
    hεw hκ (fun z => by
      have h1 := R.wrong z
      have h2 := R.undecided z
      simp only [B, V, Set.mem_singleton_iff]
      rw [h2]; exact h1)
    (fun i r r' hg ha => rounds'_local M side C xs i r r' ((hG r i).1 hg) ha)
    (fun r => by
      have := (count_le M side xs r L hL hy0 hyε hε1 hδ0 hδ1).2
      unfold readCount at this
      convert this using 2
      ext j; simp only [Finset.mem_filter]; exact and_congr_right fun _ => hG r j)
    (fun r => by
      have := (count_le M side xs r L hL hy0 hyε hε1 hδ0 hδ1).1
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
    (hεw : 0 ≤ εw) (hκ : 0 ≤ κ) (hy0 : 0 < C.harvestRate)
    (hyε : C.harvestRate ≤ C.disagreementRate / 48) (hε1 : C.disagreementRate < 1)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {d | AnyB (applyProbe (R.read · ω) C) (ReadsWrong M side (R.read · ω) C) initialState
          (List.ofFn d)} ∂μ
      ≤ ENNReal.ofReal (εw * Mb (Fintype.card σ) (Fintype.card α) P.L C.probeBudget
        + κ * Mu (Fintype.card σ) (Fintype.card α) C) := by
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
  have := wrong_le' M side C μ U εw κ R (List.ofFn d) P.L hL hεw hκ hy0 hyε hε1 hδ0 hδ1
  rw [List.length_ofFn] at this
  exact this

theorem level5RoundCorrect_holds : Level5RoundCorrect := by
  intro α _ _ σ _ M side U εw κ Ω _ μ _ R P C p₀
  dsimp only
  intro hpre hδ hy0 hyρ hρ1 hP hεw hκ hw
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1 with hS
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
  push Not at hδ1
  have hp₀ : 0 < p₀ :=
    (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  obtain ⟨kn, hA, hθg, hηm0, hη0, hAe, hAm, hround⟩ := round_knobs (σ := σ) (α := α) C P.L p₀ hp₀
    hδ hδ1 hy0 hyρ hρ1 (by exact_mod_cast hP)
  refine (eps_round_le_of C M side U _ _ _ μ R.read R.measurable R.indep R.undecided
    (fun rd => ReadsWrong M side rd C)
    (fun rd _ _ _ hs hg hx h hne => kappa_step_change M side rd C hs hg hx h hne) P _ p₀ _ kn hA
    hp₀ hpre hy0 (by linarith) hη0 hAe hAm le_rfl hθg hηm0).trans ?_
  refine (add_le_add le_rfl (pi_wrong_le' M side C μ U εw κ R P hεw hκ hy0 hyρ hρ1 hδ
    hδ1.le)).trans ?_
  have hSst : Level2.stretches nQ nα = S := rfl
  have hcost := sum_costAt_le C S hδ hδ1.le
  have hMu : Mu nQ nα C ≤ 5 * S * Real.log (8 * S * (S + 1) / δ) + 1
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
  have hwr : εw * Mb nQ nα P.L C.probeBudget + κ * Mu nQ nα C ≤ δ / 8 := by
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
