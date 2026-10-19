import OrthoDFA.Proofs.KappaRound
import OrthoDFA.Proofs.RandomShort

/-!
# Level 5 with the prefix mass read past the walk's start: the proof

The good states' bounds feed only a harvest's ending; a consistent ending's test reads nothing
of them, so with no state good (`fun _ => 1` undecided, `θ = 0`) the round fails other than by a
harvest with chance at most what the stretches and the wrong reads risk
(`eps_round_consistent_le`). With the prefix mass read at `kd`, `NoiseBad` has chance at most the
two tails at `kd` plus the chance that some string of `headStrings k kd |Q|` at a good state reads
undecided (`noiseBad_le_at`), at most `1.5θ` for each.
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open MeasureTheory Eps

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) (C : Level3Settings)

omit [Fintype α] [DecidableEq α] [Fintype σ] in
theorem not_endsWell_of_not_harvest [Countable α] {H : Type*} (s : RoundOutcome α σ H)
    (bad : σ → Prop) (h : ¬ EndsWell s) (hh : ∀ h, s.ending ≠ .harvest h) :
    ¬ EndsWell { s with badStates := bad } := by
  rcases s with ⟨_, _, _, _, _, _, _, _, ending⟩
  cases ending with
  | consistent => exact h
  | harvest h' => exact absurd rfl (hh h')
  | failed => exact h

/-- `NoiseBad` with the prefix mass read at `kd`: the two tails, plus `1.5θ` for each string of
`headStrings k kd |Q|`. -/
theorem noiseBad_le_at {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : ProbabilityTheory.iIndepFun read μ)
    (hund : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList)) (hθ : 0 ≤ θ)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k kd : ℕ) {L : ℕ}
    (hL : ∀ᵐ x ∂D, x.toList.length ≤ L) {p₀ : ℝ} (hp₀ : 0 < p₀)
    (hpre : ∀ u, D.real {x | prefixOf x kd = u} ≤ p₀) {Ge θgpt ηm : ℝ}
    (hθ8 : 8 * ((L + 1) * (Fintype.card σ + 1) * (3 / 2 * θ)) ≤ 1)
    (hθ8e : 8 * (3 * (Fintype.card σ + 1) * (3 / 2 * θ)) ≤ 1)
    (hθg : 2 * ((L + 1) * (Fintype.card σ + 1) * (3 / 2 * θ)) ≤ θgpt) :
    μ {ω | NoiseBad (read · ω) M U θ D k Ge θgpt ηm}
      ≤ ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀
          (Ge - 9 * θ * (Fintype.card σ + 1)))
        + ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀ ηm)
        + ENNReal.ofReal (3 / 2 * θ * ((∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i)
          * ∑ j ∈ Finset.range (Fintype.card σ + 1), (Fintype.card α : ℝ) ^ j)) := by
  classical
  set n := Fintype.card σ
  set Z : Finset (FreeMonoid α) := headStrings k kd n
  set good : FreeMonoid α → Prop := fun z => ¬ BadAt U θ (M.eval z.toList)
  set Sh := {ω | ∃ z ∈ Z, good z ∧ read z ω = .undecided}
  have hNe := endsNoise_off M U θ hmeas hind hund hθ D k kd (n := n)
    (η := Ge - 9 * θ * (n + 1)) hL hp₀ hpre hθ8e
  have hNm := midNoise_off M U θ hmeas hind hund hθ D k kd (n := n) (η := ηm) hL hp₀ hpre hθ8 hθg
  have hSh : μ Sh ≤ ENNReal.ofReal (3 / 2 * θ * ((∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i)
      * ∑ j ∈ Finset.range (n + 1), (Fintype.card α : ℝ) ^ j)) := by
    calc μ Sh ≤ ∑ z ∈ Z.filter good, μ {ω | read z ω = .undecided} := by
          refine (measure_mono ?_).trans (measure_biUnion_finset_le _ _)
          rintro ω ⟨z, hz, hg, hu⟩
          exact Set.mem_biUnion (Finset.mem_filter.2 ⟨hz, hg⟩) hu
      _ = ENNReal.ofReal (∑ z ∈ Z.filter good, μ.real {ω | read z ω = .undecided}) := by
          rw [ENNReal.ofReal_sum_of_nonneg fun _ _ => measureReal_nonneg]
          exact Finset.sum_congr rfl fun z _ => (ofReal_measureReal (measure_ne_top _ _)).symm
      _ ≤ _ := by
          refine ENNReal.ofReal_le_ofReal ?_
          calc _ ≤ ∑ _z ∈ Z.filter good, 3 / 2 * θ := Finset.sum_le_sum fun z hz =>
                undecided_le_of_good M U θ hund (Finset.mem_filter.1 hz).2
            _ = (Z.filter good).card * (3 / 2 * θ) := by rw [Finset.sum_const, nsmul_eq_mul]
            _ ≤ Z.card * (3 / 2 * θ) := mul_le_mul_of_nonneg_right
                (by exact_mod_cast Finset.card_filter_le _ _) (by positivity)
            _ ≤ _ := by
                rw [mul_comm]
                exact mul_le_mul_of_nonneg_left (card_headStrings k kd n) (by positivity)
  refine (measure_mono fun ω hω => ?_).trans ((measure_union_le _ _).trans
    (add_le_add ((measure_union_le _ _).trans (add_le_add hNe hNm)) hSh))
  by_cases hs : ω ∈ Sh
  · exact Or.inr hs
  have hsub : ∀ H site, goodAt (read · ω) k H M U θ site ⊆ goodOff M U θ (read · ω) k H Z site :=
    fun H site => goodAt_subset_off M U θ _ k H Z site fun z hz hg hu => hs ⟨z, hz, hg, hu⟩
  simp only [NoiseBad, Set.mem_ofPred_eq] at hω
  simp only [Set.mem_union, Set.mem_ofPred_eq]
  left
  rcases hω with h | h
  · refine Or.inl fun hall => h fun T hT E hE => ?_
    have := hall T hT E hE
    exact (measureReal_mono (hsub _ _) (measure_ne_top _ _)).trans (by linarith)
  · refine Or.inr fun hall => h fun T hT E hE => ?_
    exact (measureReal_mono (hsub _ _) (measure_ne_top _ _)).trans (hall T hT E hE)

/-- Over the reads, the round fails to end well other than by a harvest with chance at most what
the stretches risk at admissible knobs, plus the chance some step is `W`. -/
theorem eps_round_consistent_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU) (ε : ℝ)
    (W : (FreeMonoid α → ARU) → Level3State α → FreeMonoid α → Prop)
    (hW : ∀ rd, KeepsInv rd C M side (W rd)) (P : PrefixDistribution α) (y : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) θ P.L C ε y) (hηm : 0 ≤ kn.ηm) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (read · ω) draws
          ¬ EndsWell
            { target := M,
              badStates := BadAt U θ,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (read · ω) C.walkStart result.hypothesis
                ∪ endsSet (read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }
            ∧ ∀ h, result.ending ≠ .harvest h} ∂μ
      ≤ ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
          stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt')
        + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)} ∂μ := by
  obtain ⟨Ns, N₁, Ge, θgpt, ηm, pM, θr, εd', θe', θpt'⟩ := kn
  obtain ⟨hN₁, hN₁s, hθ, hθs, hGe0, hδ, hθs0, hθs1, hθpt0, hθpt1, hεd0, hεd1,
    hε, hθr0, hθr1, hεd'0, hεd'1, hθe'0, hθe'1, hθpt'0, hθpt'1, hsep, hP, hps, hθg, hθg4,
    hpM0, hyM, hηm', hfall⟩ := hA
  simp only at hN₁ hN₁s hGe0 hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP hps hfall ⊢
  simp only at hθg hθg4 hpM0 hyM hηm' hηm
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  have hnone : ∀ (rd : FreeMonoid α → ARU) (H : HypothesisDFA α) (site : Level3Site),
      goodAt rd C.walkStart H M (fun _ => (1 : ℝ)) 0 site = ∅ := by
    intro rd H site; ext x; simp [goodAt, BadAt]
  have hθgpt : 0 ≤ θgpt := le_trans (by positivity) hθg
  set c := ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
    stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set g : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin C.probeBudget => P.D)
    {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)}
  have hpt : ∀ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
        ¬ EndsWell
          { target := M,
            badStates := BadAt U θ,
            probes := P.D,
            disagreeRate := ε,
            disagree := searchSet (read · ω) C.walkStart ⟨r.1.tree, r.1.edges⟩
              ∪ endsSet (read · ω) C.walkStart ⟨r.1.tree, r.1.edges⟩,
            walkStart := C.walkStart,
            harvestLaw := harvestLaw (read · ω) P.D C.walkStart ⟨r.1.tree, r.1.edges⟩,
            yield := y,
            ending := r.2 }
          ∧ ∀ h, r.2 ≠ .harvest h}
        ≤ c + g ω := by
    intro ω
    refine (measure_mono fun d hd => not_endsWell_of_not_harvest _ (BadAt (fun _ => (1 : ℝ)) 0)
      hd.1 hd.2).trans ?_
    exact eps_probe_level (read · ω) C M side (fun _ => 1) 0 P.D ε Ns y (W (read · ω))
      (hW (read · ω)) C.probeBudget N₁ Ge θgpt ηm pM θr εd' θe' θpt'
      (fun T _ E _ => by rw [hnone, measureReal_empty]; exact hGe0)
      (fun T _ E _ => by rw [hnone, measureReal_empty]; positivity)
      hN₁ hN₁s hδ hθs0 hθs1 hps hθpt0 hθpt1 hθg4 hpM0 hyM hηm' hfall hεd0 hεd1 hε
      hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP
  calc _ ≤ ∫⁻ ω, (c + g ω) ∂μ := lintegral_mono hpt
    _ = _ := by rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]

theorem kappaRoundCorrectAt_holds : KappaRoundCorrectAt := by
  intro α _ _ σ _ M side U θ εw κ Ω _ μ _ R P C p₀ kd
  dsimp only
  intro hpre hθ hδ hθL hGy hyρ hρ1 hP hεw hκ hw
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1 with hS
  set K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) with hK
  set Λ := Real.log (16 * (K : ℝ) / δ) with hΛ
  have hHW : 0 ≤ 3 / 2 * θ * ((∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i)
      * ∑ j ∈ Finset.range (nQ + 1), (nα : ℝ) ^ j) := by positivity
  by_cases hδ1 : 1 ≤ δ
  · have h1 : ∀ F : Ω → Set (Fin C.probeBudget → FreeMonoid α),
        ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D) (F ω) ∂μ ≤ ENNReal.ofReal δ :=
      fun F => calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
        _ = 1 := by simp
        _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
    refine ⟨h1 _, (h1 _).trans (ENNReal.ofReal_le_ofReal ?_)⟩
    nlinarith
  push Not at hδ1
  have hp₀ : 0 < p₀ := (prefixMax_pos P.D kd).trans_le (ciSup_le hpre)
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
  have hSst : Level2.stretches nQ nα = S := rfl
  have hMu : Mu nQ nα C ≤ 5 * S * Real.log (8 * S * (S + 1) / δ) + 1
      + nQ * (nQ + 1) ^ 2 * nα := by
    have hcost := sum_costAt_le C S hδ hδ1.le
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
  have hwrong := pi_wrong_le' M side C μ U εw κ R P hεw hκ hy0 hyρ hρ1 hδ hδ1.le
  have hstr : ∑ j ∈ Finset.range (Level2.stretches nQ nα),
      stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt' ≤ 3 * δ / 4 := by
    rw [hSst]; exact hsum
  have hW' : ∀ rd, KeepsInv rd C M side (ReadsWrong M side rd C) :=
    fun rd _ _ _ hs hg hx h hne => kappa_step_change M side rd C hs hg hx h hne
  refine ⟨?_, ?_⟩
  · have hc := eps_round_consistent_le M side U θ C μ R.read C.disagreementRate
      (fun rd => ReadsWrong M side rd C) hW' P C.harvestRate kn hA
      (by rw [hηm]; exact (mul_pos hp₀ hΛ0).le)
    have a4 : (0 : ℝ) ≤ 3 * δ / 4 := by positivity
    have a6 : (0 : ℝ) ≤ δ / 8 := by positivity
    have e1 := ENNReal.ofReal_le_ofReal hstr
    have e2 := hwrong.trans (ENNReal.ofReal_le_ofReal hwr)
    have hfin : ENNReal.ofReal (3 * δ / 4) + ENNReal.ofReal (δ / 8) ≤ ENNReal.ofReal δ := by
      rw [← ENNReal.ofReal_add a4 a6]
      exact ENNReal.ofReal_le_ofReal (by linarith only [hδ])
    exact (hc.trans (add_le_add e1 e2)).trans hfin
  · refine (eps_round_le_noise C M side U θ _ μ R.read (fun rd => ReadsWrong M side rd C) hW' P _
      kn hA).trans ?_
    have hθs := hA.θ_small
    have hL0 : (0 : ℝ) ≤ P.L := Nat.cast_nonneg _
    have hn0 : (0 : ℝ) ≤ nQ := Nat.cast_nonneg _
    have hN := noiseBad_le_at M U θ μ R.read R.measurable R.indep R.undecided hθ P.D C.walkStart
      kd P.length_le hp₀ hpre (Ge := kn.Ge) (θgpt := kn.θgpt) (ηm := kn.ηm)
      (by nlinarith) (by nlinarith) hA.θgpt_ge
    have h1 : tailRisk nQ nα p₀ (kn.Ge - 9 * θ * (nQ + 1)) = δ / 2 / 8 := by
      rw [hGe, show 9 * θ * (nQ + 1) + p₀ * Λ - 9 * θ * (nQ + 1) = p₀ * Λ by ring, hΛ']
      exact tailRisk_eq nQ nα hp₀ (by positivity)
    have h2 : tailRisk nQ nα p₀ kn.ηm = δ / 2 / 8 := by
      rw [hηm, hΛ']
      exact tailRisk_eq nQ nα hp₀ (by positivity)
    rw [h1, h2] at hN
    calc _ ≤ ENNReal.ofReal (3 * δ / 4)
            + (ENNReal.ofReal (δ / 2 / 8) + ENNReal.ofReal (δ / 2 / 8)
              + ENNReal.ofReal (3 / 2 * θ * ((∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i)
                * ∑ j ∈ Finset.range (nQ + 1), (nα : ℝ) ^ j)))
            + ENNReal.ofReal (δ / 8) := by
          gcongr
          exact hwrong.trans (ENNReal.ofReal_le_ofReal hwr)
      _ = _ := by
          have a1 : (0 : ℝ) ≤ δ / 2 / 8 := by positivity
          have a4 : (0 : ℝ) ≤ 3 * δ / 4 := by positivity
          have a6 : (0 : ℝ) ≤ δ / 8 := by positivity
          have a3 := add_nonneg (add_nonneg a1 a1) hHW
          rw [← ENNReal.ofReal_add a1 a1, ← ENNReal.ofReal_add (add_nonneg a1 a1) hHW,
            ← ENNReal.ofReal_add a4 a3, ← ENNReal.ofReal_add (add_nonneg a4 a3) a6]
          generalize (∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i) = A
          generalize (∑ j ∈ Finset.range (nQ + 1), (nα : ℝ) ^ j) = B
          rw [show 3 * δ / 4 + (δ / 2 / 8 + δ / 2 / 8 + 3 / 2 * θ * (A * B)) + δ / 8
            = δ + 3 / 2 * θ * A * B by ring]

end Kappa

end Random

end OrthoDFA
