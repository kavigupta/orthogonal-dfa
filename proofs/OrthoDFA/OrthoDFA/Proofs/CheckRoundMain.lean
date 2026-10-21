import OrthoDFA.Proofs.CheckHead

/-!
# The checked round ends well: the proof

Given the reads, the checked round fails to end well only where a stretch risks it, as at level 3,
or at a hazard (`eps_check_level`); over the reads the stretches risk what level 4's do
(`eps_check_round_le_of`). A hazard reads a string wrong (`pi_wrongC_le`), or, reading nothing
wrong, has a check fail without a fix (`anyB_waste_le`), which needs the check strings to read
undecided at most `q` of the time at some tree or a silent check; the former has chance at most
`|heavy| u + exp(−η_c/p_h)` (`head_le`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings) (K : CheckSettings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (A sl : ℝ) (ε : ℝ)

/-- The checked round's failure chance is at most `roundBoundAt` at admissible knobs, plus the
chance of a hazard. -/
theorem eps_check_round_le_of {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : ProbabilityTheory.iIndepFun read μ)
    (hund : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList))
    (W : (FreeMonoid α → ARU) → Level3State α → FreeMonoid α → Prop)
    (hW : ∀ rd, KeepsInv rd C M side (W rd)) (P : PrefixDistribution α)
    (Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure Dc] (y p₀ η : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) C ε y) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀)
    (hy : 0 < y) (hy4 : y ≤ 1 / 4) (hη : 0 ≤ η)
    (hAe : 6 * (Fintype.card σ + 1) / y ≤ A)
    (hAm : 16 * ((P.L + 1) * (Fintype.card σ + 1)) ≤ A) (hsl : η / y ≤ sl)
    (hθg : 1 / 16 ≤ kn.θgpt) (hηm0 : 0 ≤ kn.ηm) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget =>
        P.D.prod (Measure.pi fun _ : Fin K.size => Dc))
        {draws | let result := indecisiveLStarCheck C K (read · ω) draws
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := sl,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (read · ω) C.walkStart result.hypothesis
                ∪ endsSet (read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (roundBoundAt (α := α) σ η p₀ C kn)
        + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget =>
          P.D.prod (Measure.pi fun _ : Fin K.size => Dc))
          {d | AnyB (applyCheck (read · ω) C K) (CheckHazard (read · ω) C K M side (W (read · ω)))
            initialState (List.ofFn d)} ∂μ := by
  obtain ⟨N, hNmeas, hN, hgood⟩ := harvest_noise M U μ read hmeas hind hund P C ε y p₀ η A sl hp₀
    hpre hy hy4 hη hAe hAm hsl hA.Ge_nonneg hθg hηm0
  obtain ⟨Ns, N₁, Ge, θgpt, ηm, pM, θr, εd', θe', θpt'⟩ := kn
  obtain ⟨hN₁, hN₁s, hGe0, hδ, hθs0, hθs1, hθpt0, hθpt1, hεd0, hεd1,
    hε, hθr0, hθr1, hεd'0, hεd'1, hθe'0, hθe'1, hθpt'0, hθpt'1, hsep, hP, hps,
    hθg4, hpM0, hyM, hηm, hfall⟩ := hA
  simp only at hN₁ hN₁s hGe0 hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP hps hfall ⊢
  simp only at hθg hθg4 hpM0 hyM hηm hgood
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set c := ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
    stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set g : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin C.probeBudget =>
    P.D.prod (Measure.pi fun _ : Fin K.size => Dc))
    {d | AnyB (applyCheck (read · ω) C K) (CheckHazard (read · ω) C K M side (W (read · ω)))
      initialState (List.ofFn d)}
  have hpt : ∀ ω, (Measure.pi fun _ : Fin C.probeBudget =>
      P.D.prod (Measure.pi fun _ : Fin K.size => Dc))
      {ds | let r := runChecked (read · ω) C K initialState (List.ofFn ds)
        ¬ EndsWell
          { target := M,
            undecided := U,
            harvestSlope := A,
            harvestSlack := sl,
            probes := P.D,
            disagreeRate := ε,
            disagree := searchSet (read · ω) C.walkStart ⟨r.1.tree, r.1.edges⟩
              ∪ endsSet (read · ω) C.walkStart ⟨r.1.tree, r.1.edges⟩,
            walkStart := C.walkStart,
            harvestLaw := harvestLaw (read · ω) P.D C.walkStart ⟨r.1.tree, r.1.edges⟩,
            yield := y,
            ending := r.2 }}
        ≤ (c + N.indicator 1 ω) + g ω := by
    intro ω
    by_cases hω : ω ∈ N
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans (le_add_self.trans le_self_add)
    · rw [Set.indicator_of_notMem hω, add_zero]
      exact eps_check_level (read · ω) C K M side U A sl P.D ε Ns y (W (read · ω)) (hW (read · ω))
        Dc C.probeBudget N₁ Ge θgpt ηm pM
        θr εd' θe' θpt' (hgood ω hω).1 (hgood ω hω).2
        hN₁ hN₁s hδ hθs0 hθs1 hps hθpt0 hθpt1 hθg4 hpM0 hyM hηm hfall hεd0 hεd1 hε
        hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP
  have hnn : 0 ≤ ∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
      stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt' :=
    Finset.sum_nonneg fun j _ => stretchRisk_nonneg C (level_pos C hδ j) hεd0 hεd1 hθpt0 hθpt1
      hθs0 hθs1 Ns N₁ hθr1 hεd'0 hεd'1 hθe'0 hθe'1 (le_trans (by
        show 0 ≤ 2 * C.middleRate; linarith) hθpt'0) hθpt'1
  have htr0 : 0 ≤ tailRisk (Fintype.card σ) (Fintype.card α) p₀ η := by
    unfold tailRisk; positivity
  calc _ ≤ ∫⁻ ω, ((c + N.indicator 1 ω) + g ω) ∂μ := lintegral_mono hpt
    _ = c + μ N + ∫⁻ ω, g ω ∂μ := by
        rw [lintegral_add_left ((measurable_one.indicator hNmeas).const_add c),
          lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one hNmeas]
    _ ≤ c + ENNReal.ofReal (2 * (Fintype.card σ + 1)
          * tailRisk (Fintype.card σ) (Fintype.card α) p₀ η) + ∫⁻ ω, g ω ∂μ := by
        gcongr
    _ = _ := by
        rw [add_comm c, ← ENNReal.ofReal_add (by positivity) hnn]
        rfl

omit [DecidableEq α] in
theorem card_upTo_le [Nonempty α] (n : ℕ) :
    ((upTo (α := α) n).card : ℝ) ≤ (n + 1) * (Fintype.card α : ℝ) ^ n := by
  classical
  have hA : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
  have h1 : (upTo (α := α) n).card ≤ ∑ k ∈ Finset.range (n + 1), Fintype.card α ^ k := by
    unfold upTo
    refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun k _ => ?_)
    unfold lenK
    refine Finset.card_image_le.trans ?_
    simp
  calc ((upTo (α := α) n).card : ℝ)
      ≤ ((∑ k ∈ Finset.range (n + 1), Fintype.card α ^ k : ℕ) : ℝ) := by exact_mod_cast h1
    _ = ∑ k ∈ Finset.range (n + 1), (Fintype.card α : ℝ) ^ k := by push_cast; rfl
    _ ≤ ∑ _k ∈ Finset.range (n + 1), (Fintype.card α : ℝ) ^ n :=
        Finset.sum_le_sum fun k hk => pow_le_pow_right₀ hA (by
          have := Finset.mem_range.1 hk; omega)
    _ = _ := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; push_cast; ring

omit [DecidableEq α] in
theorem card_heavySet_le [Nonempty α] (h n : ℕ) :
    ((heavySet (α := α) h n).card : ℝ)
      ≤ (h + 1) * (Fintype.card α : ℝ) ^ h * ((n + 1) * (Fintype.card α : ℝ) ^ n) := by
  classical
  have : (heavySet (α := α) h n).card ≤ (upTo (α := α) h).card * (upTo (α := α) n).card := by
    unfold heavySet
    exact Finset.card_image_le.trans (by rw [Finset.card_product])
  calc ((heavySet (α := α) h n).card : ℝ)
      ≤ ((upTo (α := α) h).card : ℝ) * (upTo (α := α) n).card := by exact_mod_cast this
    _ ≤ _ := mul_le_mul (card_upTo_le h) (card_upTo_le n) (Nat.cast_nonneg _) (by positivity)

end Eps

end Random

open MeasureTheory Random Random.Eps

theorem level4CheckRoundCorrect_holds : Level4CheckRoundCorrect := by
  intro α _ _ _ σ _ M side U εw Ω _ μ _ R P Dc _ Lc h C K p₀ u ph ηc
  dsimp only
  intro hpre hδ hy0 hyρ hρ1 hP hDc hU hu0 hph hph0 hhL hηc hθ8 hq1 hr0 hr1 hK hw
  classical
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set Ms : ℝ := (nQ + 1) * nα ^ nQ
  set q : ℝ := 2 * ((Lc + 1) * Ms * u) + ηc
  set S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
  have hq0 : 0 ≤ q := by positivity
  have hsb : 0 ≤ K.silentBase q := by
    unfold CheckSettings.silentBase
    have : (0 : ℝ) ≤ K.rate / (K.rate / 2) := by positivity
    have : (0 : ℝ) ≤ (1 - K.rate) / (1 - K.rate / 2) := div_nonneg (by linarith) (by linarith)
    nlinarith
  have hlev : ∀ j, 0 < K.level j := fun j => by unfold CheckSettings.level; positivity
  set Wsum : ℝ := ∑ j ∈ Finset.range S, K.silentBase q ^ K.size / K.level j
  have hWsum0 : 0 ≤ Wsum := Finset.sum_nonneg fun j _ => div_nonneg (pow_nonneg hsb _) (hlev j).le
  have hMs : ((upTo (α := α) nQ).card : ℝ) ≤ Ms := card_upTo_le nQ
  set heavy : ℝ := (h + 1) * nα ^ h * Ms * u
  have hheavy0 : 0 ≤ heavy := by positivity
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by
          rw [← ENNReal.ofReal_one]
          exact ENNReal.ofReal_le_ofReal (by have := (Real.exp_pos (-(ηc / ph))).le; linarith)
  push Not at hδ1
  have hp₀ : 0 < p₀ := (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  obtain ⟨kn, hA, hθg, hηm0, hη0, hAe, hAm, hround⟩ := round_knobs (σ := σ) (α := α) C P.L p₀ hp₀
    hδ hδ1 hy0 hyρ hρ1 (by exact_mod_cast hP)
  have hmain := eps_check_round_le_of C K M side U _ _ _ μ R.read R.measurable R.indep R.undecided
    (fun rd => WrongStep M side rd C)
    (fun rd _ _ _ hs hg hx h hne => eps_step_change M side rd C hs hg hx h hne) P Dc _ p₀ _ kn hA
    hp₀ hpre hy0 (by linarith) hη0 hAe hAm le_rfl hθg hηm0
  refine hmain.trans ?_
  -- the hazards
  set Dd := P.D.prod (Measure.pi fun _ : Fin K.size => Dc)
  set Bad : Set Ω := {ω | ∃ T : DTree α, (∀ m ∈ mids T, m.toList.length ≤ nQ) ∧
    2 * ((Lc + 1) * (upTo (α := α) nQ).card * u) + ηc
      < Dc.real {x | Utype (R.read · ω) T x = true}}
  have hBad : μ Bad ≤ ENNReal.ofReal (heavy + Real.exp (-(ηc / ph))) := by
    refine (head_le R.read R.measurable R.indep (Lh := Lc + nQ)
      (fun z hz => by rw [R.undecided z]; exact hU z hz) Dc hDc le_rfl hph0 hph (by omega)
      (le_trans (by gcongr) hθ8) hu0).trans (ENNReal.ofReal_le_ofReal ?_)
    have := card_heavySet_le (α := α) h nQ
    have : ((heavySet (α := α) h nQ).card : ℝ) * u ≤ heavy := by
      simp only [heavy]
      have := mul_le_mul_of_nonneg_right this hu0
      linarith
    linarith
  set g₁ : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin C.probeBudget => Dd)
    {d | AnyB (applyCheck (R.read · ω) C K) (WrongC M side C K (R.read · ω)) initialState
      (List.ofFn d)}
  set c₂ : ENNReal := ∑ j ∈ Finset.range (Level2.stretches nQ nα),
    ENNReal.ofReal (K.silentBase q ^ K.size / K.level j)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin C.probeBudget => Dd)
      {d | AnyB (applyCheck (R.read · ω) C K)
        (CheckHazard (R.read · ω) C K M side (WrongStep M side (R.read · ω) C)) initialState
        (List.ofFn d)}
      ≤ g₁ ω + (c₂ + (toMeasurable μ Bad).indicator 1 ω) := by
    intro ω
    have hsplit : {d : Fin C.probeBudget → CheckDraw α K | AnyB (applyCheck (R.read · ω) C K)
        (CheckHazard (R.read · ω) C K M side (WrongStep M side (R.read · ω) C)) initialState
        (List.ofFn d)}
        ⊆ {d | AnyB (applyCheck (R.read · ω) C K) (WrongC M side C K (R.read · ω)) initialState
            (List.ofFn d)}
          ∪ {d | AnyB (stopAt (applyCheck (R.read · ω) C K) (WrongC M side C K (R.read · ω))
            .failed) (WasteC (R.read · ω) C K) initialState (List.ofFn d)} := by
      intro d hd
      rcases anyB_split (applyCheck (R.read · ω) C K) (WrongC M side C K (R.read · ω)) .failed _
        (WasteC (R.read · ω) C K) (fun s x hx => by
          rcases hx with hx | ⟨hf, hx | hx⟩
          · exact Or.inr (Or.inl hx)
          · exact Or.inr (Or.inr ⟨hf, hx⟩)
          · exact Or.inl ⟨hf, hx⟩) _ _ hd with h1 | h1
      · exact Or.inr h1
      · exact Or.inl h1
    refine (measure_mono hsplit).trans ((measure_union_le _ _).trans (add_le_add le_rfl ?_))
    by_cases hω : ω ∈ Bad
    · rw [Set.indicator_of_mem (subset_toMeasurable μ Bad hω), Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · refine le_trans ?_ le_self_add
      refine anyB_waste_le (R.read · ω) C K Dc M side P.D hq1 hr0 hr1 hK (fun T hT => ?_)
        C.probeBudget
      by_contra hlt
      push Not at hlt
      apply hω
      refine ⟨T, midsLe_classSet nQ T hT, lt_of_le_of_lt ?_ hlt⟩
      have : (Lc + 1) * ((upTo (α := α) nQ).card : ℝ) * u ≤ (Lc + 1) * Ms * u := by gcongr
      linarith
  have hg₁ := pi_wrongC_le M side C K μ U εw R P Dc Lc hDc
  have hc₂ : c₂ = ENNReal.ofReal Wsum := by
    rw [ENNReal.ofReal_sum_of_nonneg fun j _ => div_nonneg (pow_nonneg hsb _) (hlev j).le]
    rfl
  have hwr : εw * (C.probeBudget * ((P.L + 1) * nQ + K.size * ((Lc + 1) * nQ)
      + (K.size + 1) * ((nQ + 1) * nα * (nQ + 1)))) ≤ δ / 8 := by
    convert hw using 1
    ring
  calc _ ≤ ENNReal.ofReal (7 * δ / 8) + ∫⁻ ω, (g₁ ω + (c₂ + (toMeasurable μ Bad).indicator 1 ω)) ∂μ :=
        add_le_add (ENNReal.ofReal_le_ofReal hround) (lintegral_mono hpt)
    _ = ENNReal.ofReal (7 * δ / 8) + (∫⁻ ω, g₁ ω ∂μ + (c₂ + μ (toMeasurable μ Bad))) := by
        rw [lintegral_add_right (fun ω => g₁ ω) (show Measurable fun ω =>
            c₂ + (toMeasurable μ Bad).indicator 1 ω from measurable_const.add
              (measurable_one.indicator (measurableSet_toMeasurable _ _))),
          lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ ENNReal.ofReal (7 * δ / 8) + (ENNReal.ofReal (δ / 8) + (ENNReal.ofReal Wsum
          + ENNReal.ofReal (heavy + Real.exp (-(ηc / ph))))) := by
        rw [measure_toMeasurable, hc₂]
        gcongr
        exact hg₁.trans (ENNReal.ofReal_le_ofReal hwr)
    _ = _ := by
        have e : 7 * δ / 8 + (δ / 8 + (Wsum + (heavy + Real.exp (-(ηc / ph)))))
            = δ + Wsum + heavy + Real.exp (-(ηc / ph)) := by ring
        have h1 : 0 ≤ heavy + Real.exp (-(ηc / ph)) := by positivity
        rw [← e, ENNReal.ofReal_add (p := 7 * δ / 8) (by positivity) (by positivity),
          ENNReal.ofReal_add (p := δ / 8) (by positivity) (by positivity),
          ENNReal.ofReal_add (p := Wsum) hWsum0 h1]

end OrthoDFA
