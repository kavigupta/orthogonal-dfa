import OrthoDFA.Proofs.LoopRenoise
import OrthoDFA.Proofs.KappaShort

/-!
# Level 5's guarantee through the family, read at a depth

`level3Fail_le` with `KappaRoundCorrectAt` in place of level 5's claim, for a walk starting before
the depth `kd` at which the probes' prefix mass is read.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]
  {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'} [IsProbabilityMeasure μ']

open Random.Kappa in
/-- `level3Fail_le` with the probes' prefix mass read at a depth `kd`: the bound gains
`KappaRoundCorrectAt`'s term for the strings before `kd`. -/
theorem level3Fail_le_at (fam : SuffixFamily μ' α σ) (hε : 0 ≤ fam.ε) (hκ : 0 ≤ fam.κ)
    (P : PrefixDistribution α) (C : Random.Level3Settings) (θ p₀ u : ℝ) (kd : ℕ) (hu : 0 < u) :
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα)
    let G := 9 * θ * (nQ + 1) + 2 * p₀ * Real.log (16 * K / C.δ)
    (∀ u, P.D.real {x | prefixOf x kd = u} ≤ p₀) → 0 ≤ θ → 0 < C.δ → C.δ ≤ 1 →
    48 * (P.L + 1) * (nQ + 1) * θ ≤ 1 → 2 * G ≤ C.harvestRate →
    C.harvestRate ≤ C.disagreementRate / 48 → C.disagreementRate < 1 →
    36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget →
    fam.ε * (C.probeBudget * (P.L + 1) * nQ + nQ * (nQ + 1) ^ 2 * nα)
      + fam.κ * (5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1 + nQ * (nQ + 1) ^ 2 * nα)
        ≤ C.δ / 8 →
    (μ'.prod (Measure.pi fun _ : Fin C.probeBudget => P.D)) (level3Fail fam C P.D θ u)
      ≤ ENNReal.ofReal (C.δ + (if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1)
          * nQ + nQ * (nQ + 1) ^ 2 * nα + (∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i)
            * (nQ + 1)) else 0) + C.δ / 8
        + (shortStrings (α := α) (Fintype.card σ)).card * (fam.ε + fam.κ) / u) := by
  intro nQ nα S K G hpre hθ hδ hδ1 hθL hGy hyρ hρ1 hP hw
  classical
  set side : σ → Bool := fun q => readSide (fam.readDist q)
  set U : σ → ℝ := fun q => fam.readDist q .undecided
  obtain ⟨R, hR⟩ := familyKappaOracle_holds fam hε hκ
  have hRr : ∀ z ω, R.read z ω = fam.read ω z := fun z ω => congrFun (congrFun hR z) ω
  set k := C.walkStart
  set T := C.probeBudget
  set π := Measure.pi fun _ : Fin T => P.D
  set wrong : FreeMonoid α → ARU := fun z => if side (fam.M.eval z.toList) then .reject else .accept
  -- level 3 ends well
  set ex : ℝ := if C.walkStart < kd then 3 / 2 * θ * (C.probeBudget * (P.L + 1) * nQ
    + nQ * (nQ + 1) ^ 2 * nα + (∑ i ∈ Finset.Ico C.walkStart kd, (nα : ℝ) ^ i) * (nQ + 1)) else 0
  have hex0 : 0 ≤ ex := by simp only [ex]; split_ifs <;> positivity
  have hK := (Random.Kappa.kappaRoundCorrectAt_holds fam.M side U θ fam.ε fam.κ μ' R P C p₀ kd
    hpre hθ hδ hθL hGy hyρ hρ1 hP hε hκ hw).2
  -- no step reads wrong
  have hp₀ : 0 < p₀ := (Random.prefixMax_pos P.D kd).trans_le (ciSup_le hpre)
  have hK1 : (1 : ℝ) ≤ K := by
    have : 1 ≤ K := Nat.one_le_iff_ne_zero.2 (by positivity)
    exact_mod_cast this
  have hΛ0 : 0 < Real.log (16 * (K : ℝ) / C.δ) := Real.log_pos (by
    rw [one_lt_div hδ]; linarith)
  have hy0 : 0 < C.harvestRate := by
    have : 0 < 2 * G := by
      have : 0 ≤ 9 * θ * ((nQ : ℝ) + 1) := by positivity
      have : 0 < 2 * p₀ * Real.log (16 * (K : ℝ) / C.δ) := by positivity
      show 0 < 2 * (9 * θ * ((nQ : ℝ) + 1) + 2 * p₀ * Real.log (16 * (K : ℝ) / C.δ))
      linarith
    linarith
  have hW := pi_wrong_le' fam.M side C μ' U fam.ε fam.κ R P hε hκ hy0 hyρ hρ1 hδ hδ1
  have hSst : Level2.stretches nQ nα = S := rfl
  have hcost := sum_costAt_le C S hδ hδ1
  have hMu : Mu nQ nα C ≤ 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1
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
  have hwr : fam.ε * Mb nQ nα P.L C.probeBudget + fam.κ * Mu nQ nα C ≤ C.δ / 8 := by
    rw [hMb]
    have := mul_le_mul_of_nonneg_left hMu hκ
    linarith
  have hW' := hW.trans (ENNReal.ofReal_le_ofReal hwr)
  -- the probes reading wrong past a short midfix weigh at most `u`
  set Mk := {ω | u < P.D.real (wrongSet (fun z => fam.read ω z) wrong nQ)}
  have hMk : MeasurableSet Mk :=
    measurableSet_lt measurable_const (ENNReal.measurable_toReal.comp
      (measurable_wrongSet (fun z ω => fam.read ω z) fam.read_measurable wrong P.D nQ))
  have hMkle : μ' Mk ≤ ENNReal.ofReal ((shortStrings (α := α) nQ).card * (fam.ε + fam.κ) / u) :=
    wrongSet_markov (fun z ω => fam.read ω z) fam.read_measurable wrong (by positivity) (fun z => by
      have h1 := R.wrong z
      simp only [hRr] at h1
      have h2 : fam.κ * U (fam.M.eval z.toList) ≤ fam.κ :=
        mul_le_of_le_one_right hκ (readDist_le_one fam _ _)
      exact h1.trans (by linarith)) P.D nQ hu
  -- the three events
  set E₁ := {d : Ω' × (Fin T → FreeMonoid α) |
    let result := Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2
    ¬ EndsWell
      { target := fam.M,
        badStates := Random.BadAt U θ,
        probes := P.D,
        disagreeRate := C.disagreementRate,
        disagree := Random.searchSet (fun z => fam.read d.1 z) k result.hypothesis
          ∪ Random.endsSet (fun z => fam.read d.1 z) k result.hypothesis,
        walkStart := k,
        harvestLaw := Random.harvestLaw (fun z => fam.read d.1 z) P.D k result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending }}
  have hE₁ : MeasurableSet E₁ := measurableSet_of_result fam C (fun ω result => ¬ EndsWell
    { target := fam.M,
      badStates := Random.BadAt U θ,
      probes := P.D,
      disagreeRate := C.disagreementRate,
      disagree := Random.searchSet (fun z => fam.read ω z) k result.hypothesis
        ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis,
      walkStart := k,
      harvestLaw := Random.harvestLaw (fun z => fam.read ω z) P.D k result.hypothesis,
      yield := C.harvestRate,
      ending := result.ending }) fun res =>
      measurableSet_not_endsWell_searchEnds fam.O fam.F _ _ k fam.M _ P.D _ _ _ _
  set E₂ := {d : Ω' × (Fin T → FreeMonoid α) |
    Random.AnyB (Random.applyProbe (fun z => fam.read d.1 z) C)
      (Random.Kappa.ReadsWrong fam.M side (fun z => fam.read d.1 z) C) Random.initialState
      (List.ofFn d.2)}
  have hE₂ : MeasurableSet E₂ := measurableSet_anyB fam C side
  have hbad : Random.BadAt U θ = fun q => ¬ fam.readDist q .undecided ≤ 3 / 2 * θ :=
    funext fun q => propext not_le.symm
  -- off the three, the round ends well
  have hsub : ∀ ω, Prod.mk ω ⁻¹' level3Fail fam C P.D θ u
      ⊆ Prod.mk ω ⁻¹' E₁ ∪ Prod.mk ω ⁻¹' E₂ ∪ (if ω ∈ Mk then Set.univ else ∅) := by
    intro ω xs hxs
    by_contra hno
    simp only [Set.mem_union, Set.mem_preimage, not_or] at hno
    obtain ⟨⟨h1, h2⟩, h3⟩ := hno
    have hM : ω ∉ Mk := fun h => h3 (by rw [if_pos h]; trivial)
    apply hxs
    set result := Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs
    have hEW : EndsWell
      { target := fam.M,
        badStates := Random.BadAt U θ,
        probes := P.D,
        disagreeRate := C.disagreementRate,
        disagree := Random.searchSet (fun z => fam.read ω z) k result.hypothesis
          ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis,
        walkStart := k,
        harvestLaw := Random.harvestLaw (fun z => fam.read ω z) P.D k result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending } := by
      by_contra h; exact h1 h
    have hcls : result.hypothesis.tree ∈ Random.classSet nQ :=
      Random.tree_classSet_of_good fam.M side (fun z => fam.read ω z) C xs h2
    refine ⟨?_, hcls⟩
    rw [← hbad]
    rcases he : result.ending with _ | site | _
    · rw [he] at hEW
      show P.D.real (disagreeSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis)
        ≤ C.disagreementRate + u
      have hr : rarerSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis.tree
          ⊆ wrongSet (fun z => fam.read ω z) wrong nQ := by
        rintro y ⟨-, m, hm, hw⟩
        exact ⟨m, mem_shortStrings.2 ((Random.classSet_props _ _ hcls).1 m hm), by exact hw⟩
      have hMr : P.D.real (wrongSet (fun z => fam.read ω z) wrong nQ) ≤ u := by
        by_contra h; exact hM (lt_of_not_ge h)
      calc _ ≤ P.D.real (Random.searchSet (fun z => fam.read ω z) k result.hypothesis
              ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis)
            + P.D.real (rarerSet fam.M fam.readDist (fun z => fam.read ω z) k
              result.hypothesis.tree) := measureReal_union_le _ _
        _ ≤ C.disagreementRate + u := add_le_add hEW
            ((measureReal_mono hr (measure_ne_top _ _)).trans hMr)
    · rw [he] at hEW; exact hEW
    · rw [he] at hEW; exact hEW
  -- the integral
  rw [Measure.prod_apply (measurableSet_level3Fail fam C P.D θ u)]
  calc ∫⁻ ω, π (Prod.mk ω ⁻¹' level3Fail fam C P.D θ u) ∂μ'
      ≤ ∫⁻ ω, (π (Prod.mk ω ⁻¹' E₁) + π (Prod.mk ω ⁻¹' E₂) + Mk.indicator 1 ω) ∂μ' := by
        refine lintegral_mono fun ω => (measure_mono (hsub ω)).trans
          ((measure_union_le _ _).trans (add_le_add (measure_union_le _ _) ?_))
        by_cases h : ω ∈ Mk
        · rw [if_pos h, Set.indicator_of_mem h]; exact prob_le_one
        · rw [if_neg h, Set.indicator_of_notMem h]; simp
    _ = ∫⁻ ω, π (Prod.mk ω ⁻¹' E₁) ∂μ' + ∫⁻ ω, π (Prod.mk ω ⁻¹' E₂) ∂μ' + μ' Mk := by
        rw [lintegral_add_right _ (measurable_one.indicator hMk),
          lintegral_add_left (measurable_measure_prodMk_left hE₁),
          lintegral_indicator_one hMk]
    _ ≤ ENNReal.ofReal (C.δ + ex) + ENNReal.ofReal (C.δ / 8)
          + ENNReal.ofReal ((shortStrings (α := α) nQ).card * (fam.ε + fam.κ) / u) := by
        simp only [hRr] at hK hW'
        gcongr
        · exact hK
        · exact hW'
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity)]

end OrthoDFA

end
