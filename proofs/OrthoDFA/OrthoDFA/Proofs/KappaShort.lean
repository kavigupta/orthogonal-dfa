import OrthoDFA.Proofs.KappaHead

/-!
# Level 5 with the prefix mass read past the walk's start: the proof

The good states' bounds feed only a harvest's ending; a consistent ending's test reads nothing
of them, so with no state good (`fun _ => 1` undecided, `θ = 0`) the round fails other than by a
harvest with chance at most what the stretches and the wrong reads risk
(`eps_round_consistent_le`). Otherwise a step at a tree with a string of `headOf k kd` at a good
read-state reading undecided counts as bad (`HeadUnd`): off such trees the good states' bounds
hold but for the two tails at `kd` (`noiseBad_le_at`), and such a step, before any wrong read,
has chance at most `1.5θ (Mb + H (|Q| + 1))` (`head_le`).
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

/-- `NoiseBad` off the trees with a string of `headOf k kd` at a good read-state reading
undecided, with the prefix mass read at `kd`: the two tails. -/
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
    μ {ω | NoiseBad (read · ω) M U θ D k (HeadUnd M U θ k kd (read · ω)) Ge θgpt ηm}
      ≤ ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀
          (Ge - 9 * θ * (Fintype.card σ + 1)))
        + ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀ ηm) := by
  classical
  set n := Fintype.card σ
  have hNe := endsNoise_off M U θ hmeas hind hund hθ D k kd (n := n)
    (η := Ge - 9 * θ * (n + 1)) hL hp₀ hpre hθ8e
  have hNm := midNoise_off M U θ hmeas hind hund hθ D k kd (n := n) (η := ηm) hL hp₀ hpre hθ8 hθg
  refine (measure_mono fun ω hω => ?_).trans ((measure_union_le _ _).trans (add_le_add hNe hNm))
  have hsub : ∀ T, ¬ HeadUnd M U θ k kd (read · ω) T → ∀ E site,
      goodAt (read · ω) k ⟨T, E⟩ M U θ site
        ⊆ goodOff M U θ (read · ω) k ⟨T, E⟩ (headOf k kd T) site :=
    fun T hT E site => goodAt_subset_off M U θ _ k _ _ site fun z hz hg hu => hT ⟨z, hz, hg, hu⟩
  simp only [NoiseBad, Set.mem_ofPred_eq] at hω
  simp only [Set.mem_union, Set.mem_ofPred_eq]
  rcases hω with h | h
  · refine Or.inl fun hall => h fun T hT hsp E hE => ?_
    have := hall T hT E hE
    exact (measureReal_mono (hsub T hsp E _) (measure_ne_top _ _)).trans (by linarith)
  · refine Or.inr fun hall => h fun T hT hsp E hE => ?_
    exact (measureReal_mono (hsub T hsp E _) (measure_ne_top _ _)).trans (hall T hT E hE)

/-- Over the probes and the reads, some step reads a string wrong, or is at a tree with a
string of `headOf k kd` at a good read-state reading undecided, with chance at most
`εw Mb + κ Mu`, plus `1.5θ (Mb + H (|Q| + 1))` where `k < kd`. -/
theorem pi_wrong_head_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (εw κ : ℝ) (R : KappaReadOracle μ M side U εw κ) (P : PrefixDistribution α) (k kd : ℕ)
    (hεw : 0 ≤ εw) (hκ : 0 ≤ κ) (hθ : 0 ≤ θ) (hy0 : 0 < C.harvestRate)
    (hyε : C.harvestRate ≤ C.disagreementRate / 48) (hε1 : C.disagreementRate < 1)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {d | AnyB (applyProbe (R.read · ω) C) (fun s x => ReadsWrong M side (R.read · ω) C s x
          ∨ HeadUnd M U θ k kd (R.read · ω) s.tree) initialState (List.ofFn d)} ∂μ
      ≤ ENNReal.ofReal (εw * Mb (Fintype.card σ) (Fintype.card α) P.L C.probeBudget
        + κ * Mu (Fintype.card σ) (Fintype.card α) C)
        + ENNReal.ofReal (if k < kd then 3 / 2 * θ
          * (Mb (Fintype.card σ) (Fintype.card α) P.L C.probeBudget
          + (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1))
        else 0) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  refine lintegral_le_of_points μ _ _ _ fun d hd => ?_
  have hL : ∀ x ∈ List.ofFn d, x.toList.length ≤ P.L := by
    intro x hx
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    by_contra hlt
    apply hd
    have hpi : (Measure.pi fun _ : Fin C.probeBudget => P.D) {d} = ∏ j, P.D {d j} := by
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
  have hw := wrong_le' M side C μ U εw κ R (List.ofFn d) P.L hL hεw hκ hy0 hyε hε1 hδ0 hδ1
  rw [List.length_ofFn] at hw
  have hh : μ {ω | ∃ j < (List.ofFn d).length, Good' M side C (List.ofFn d) (R.read · ω) (2 * j)
      ∧ HeadUnd M U θ k kd (R.read · ω) (treeAt C (List.ofFn d) (R.read · ω) j)}
      ≤ ENNReal.ofReal (if k < kd then 3 / 2 * θ
        * (Mb (Fintype.card σ) (Fintype.card α) P.L C.probeBudget
          + (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1)) else 0) := by
    by_cases hk : k < kd
    · rw [if_pos hk]
      have := head_le M side U θ C μ εw κ R k kd (List.ofFn d) P.L hL hθ hy0 hyε hε1 hδ0 hδ1
      rw [List.length_ofFn] at this ⊢
      exact this
    · rw [if_neg hk]
      refine le_of_eq_of_le (measure_mono_null (fun ω hω => ?_) measure_empty) bot_le
      obtain ⟨j, -, -, z, hz, -⟩ := hω
      simp only [headOf, Finset.mem_biUnion, Finset.mem_Ico] at hz
      obtain ⟨i, hi, -⟩ := hz
      omega
  refine (measure_mono fun ω hω => anyB_head U θ (R.read · ω) k kd (List.ofFn d) hω).trans
    ((measure_union_le _ _).trans (add_le_add hw hh))

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
      (hW (read · ω)) (fun _ => False) (fun _ _ h => h.elim) C.probeBudget N₁ Ge θgpt ηm pM θr
      εd' θe' θpt'
      (fun T _ _ E _ => by rw [hnone, measureReal_empty]; exact hGe0)
      (fun T _ _ E _ => by rw [hnone, measureReal_empty]; positivity)
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
  set ex : ℝ := if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1) * nQ
    + nQ * (nQ + 1) ^ 2 * nα + (∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i) * (nQ + 1)) else 0
  have hHW : 0 ≤ ex := by simp only [ex]; split_ifs <;> positivity
  by_cases hδ1 : 1 ≤ δ
  · have h1 : ∀ F : Ω → Set (Fin C.probeBudget → FreeMonoid α),
        ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D) (F ω) ∂μ ≤ ENNReal.ofReal δ :=
      fun F => calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
        _ = 1 := by simp
        _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
    exact ⟨h1 _, (h1 _).trans (ENNReal.ofReal_le_ofReal (by linarith only [hHW]))⟩
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
  · have hW'' : ∀ rd, KeepsInv rd C M side (fun s x => ReadsWrong M side rd C s x
        ∨ HeadUnd M U θ C.walkStart kd rd s.tree) :=
      fun rd _ _ _ hs hg hx h hne => kappa_step_change M side rd C hs hg (fun hr => hx (Or.inl hr))
        h hne
    refine (eps_round_le_noise C M side U θ _ μ R.read _ hW''
      (fun rd T => HeadUnd M U θ C.walkStart kd rd T) (fun _ _ _ h => Or.inr h) P _ kn hA).trans ?_
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
    have hH := pi_wrong_head_le M side U θ C μ εw κ R P C.walkStart kd hεw hκ hθ hy0 hyρ hρ1 hδ
      hδ1.le
    have hex : (if C.walkStart < kd then 3 / 2 * θ * (Mb nQ nα P.L C.probeBudget
        + (∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i) * (nQ + 1)) else 0) = ex := by
      simp only [ex, hMb]
    rw [hex] at hH
    have hH' := hH.trans (add_le_add (ENNReal.ofReal_le_ofReal hwr) le_rfl)
    have e1 := ENNReal.ofReal_le_ofReal hstr
    have a1 : (0 : ℝ) ≤ δ / 2 / 8 := by positivity
    have a4 : (0 : ℝ) ≤ 3 * δ / 4 := by positivity
    have a5 : (0 : ℝ) ≤ δ / 8 := by positivity
    have a6 : (0 : ℝ) ≤ δ / 8 + ex := add_nonneg a5 hHW
    have hfin : ENNReal.ofReal (3 * δ / 4)
        + (ENNReal.ofReal (δ / 2 / 8) + ENNReal.ofReal (δ / 2 / 8))
        + (ENNReal.ofReal (δ / 8) + ENNReal.ofReal ex) = ENNReal.ofReal (δ + ex) := by
      rw [← ENNReal.ofReal_add a1 a1, ← ENNReal.ofReal_add a4 (add_nonneg a1 a1),
        ← ENNReal.ofReal_add a5 hHW, ← ENNReal.ofReal_add (add_nonneg a4 (add_nonneg a1 a1)) a6]
      generalize ex = A
      rw [show 3 * δ / 4 + (δ / 2 / 8 + δ / 2 / 8) + (δ / 8 + A) = δ + A by ring]
    exact ((add_le_add (add_le_add e1 hN) hH').trans (le_of_eq hfin))

end Kappa

end Random

end OrthoDFA
