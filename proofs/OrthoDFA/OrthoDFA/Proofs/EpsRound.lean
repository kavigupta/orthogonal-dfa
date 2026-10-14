import OrthoDFA.Level4Claim
import OrthoDFA.Proofs.RandomRound

/-!
# Level 3 with rare wrong reads: the proof

Given the reads, `probe_level` bounds the round's failure by `stretches` times what a stretch
risks, a probe that can read a string wrong counted as a bad step, at most `Ns · Gw` a stretch.
Over the reads, the probes that can read a good read-state undecided are at most `G` of the
probes but for `noiseRisk`, and those that can read a string wrong at most `Gw` but for
`wrongRisk`.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

theorem epsRoundCorrect_holds : EpsRoundCorrect := by
  intro α _ _ σ _ M side U θ εw Ω _ μ _ R P C ε y
  obtain ⟨D, _, L, hL⟩ := P
  obtain ⟨read, hmeas, hind, hU, hwrong⟩ := R
  have hεw : 0 ≤ εw := measureReal_nonneg.trans (hwrong 1)
  simp only [epsRoundRisk]
  refine le_iInf₂ fun kn hA => ?_
  obtain ⟨⟨Ns, N₁, G, θr, εd', θpt'⟩, Gw⟩ := kn
  obtain ⟨⟨hN₁, hN₁s, hθ, hG0, ha, hθs0, hθs, hθs1, hθe, hθpt0, hθpt, hθpt1, hεd0, hεd, hεd1,
    hε, hθr0, hθr1, hεd'0, hεd'1, hθpt'0, hθpt'1, hsep, hP, hGt, hyt, hps, hfall⟩, hGw0⟩ := hA
  simp only at hN₁ hN₁s hG0 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP hGt hps hfall hGw0
  unfold epsRoundBound roundBound
  simp only
  set P := C.probeBudget
  set p₀ := prefixMax D C.walkStart
  have hp₀ : 0 < p₀ := prefixMax_pos D C.walkStart
  have hpre : ∀ u : FreeMonoid α, u.toList.length = C.walkStart
      → D.real {x | prefixOf x C.walkStart = u} ≤ p₀ :=
    fun u _ => le_prefixMax D C.walkStart u
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set Nn := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotGood M U θ C.walkStart (read · ω) T) ≤ G}
  set Nw := {ω | Gw < D.real (PotWrong M side C.walkStart (Fintype.card σ) (read · ω))}
  have hNn := noise_le M U θ μ read D C.walkStart L p₀ G hmeas hind hU hL hp₀ hpre hθ
  set K := ∑ j ∈ Finset.range (Fintype.card σ + 1), Fintype.card α ^ j with hK
  have hK1 : 1 ≤ K := by rw [hK, Finset.sum_range_succ']; simp
  have hNw : μ Nw
      ≤ ENNReal.ofReal (wrongRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw) := by
    have h := noise_hit_one μ read D C.walkStart L p₀ Gw εw
      (fun z r => r = if side (M.eval z.toList) then .reject else .accept) hmeas hind hwrong hεw
      hL hp₀ hpre (upTo (Fintype.card σ)) (K - 1)
      (by have := card_upTo (α := α) (Fintype.card σ); omega)
    have he : ((K - 1 : ℕ) : ℝ) + 1
        = ∑ j ∈ Finset.range (Fintype.card σ + 1), (Fintype.card α : ℝ) ^ j := by
      rw [Nat.cast_sub hK1, hK]
      push_cast
      ring
    rw [he] at h
    exact h
  set c := ENNReal.ofReal ((Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1) + 2 * Fintype.card α + 1)
    * stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' + P * Gw)
  set B := toMeasurable μ (Nn ∪ Nw)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
          C.walkStart (harvestLaw (read · ω) D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
        ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ Nn ∪ Nw := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or] at hω'
      obtain ⟨hωN, hωW⟩ := hω'
      simp only [Nn, Nw, Set.mem_ofPred_eq, not_not, not_lt] at hωN hωW
      refine (probe_level (read · ω) C M U θ D ε Ns y (side := side) P N₁ G Gw θr εd' θpt'
        hωN hωW hGw0 hN₁ hN₁s hG0 hGt hyt ha hθs0 hθs hθs1 hps hθe hθpt0 hθpt hθpt1 hfall hεd0
        hεd hεd1 hε hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP).trans
        (ENNReal.ofReal_le_ofReal ?_)
      have hP' : ((Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns : ℕ) : ℝ) ≤ P := by
        exact_mod_cast hP
      unfold Level2.stretches at hP' ⊢
      push_cast at hP' ⊢
      nlinarith
  have hnn :
      0 ≤ stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' :=
    stretchRisk_nonneg C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' ha.le hθr0 hθr1
      hεd'0 hεd'1 hθpt'0 hθpt'1
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ Nn + μ Nw) := by
        gcongr
        rw [measure_toMeasurable]
        exact measure_union_le _ _
    _ ≤ c + (ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G)
          + ENNReal.ofReal (wrongRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw)) := by
        gcongr
    _ = _ := by
        have h1 : 0 ≤ noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G := by
          unfold noiseRisk; positivity
        have h2 : 0 ≤ wrongRisk (Fintype.card σ) (Fintype.card α) L εw p₀ Gw := by
          unfold wrongRisk; positivity
        have h3 : 0 ≤ (Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
            + 2 * Fintype.card α + 1 : ℝ)
            * stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' + P * Gw := by
          positivity
        rw [← ENNReal.ofReal_add h1 h2, ← ENNReal.ofReal_add h3 (by positivity)]
        congr 1
        ring

end Eps

end Random

end OrthoDFA
