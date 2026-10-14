import OrthoDFA.Proofs.RandomStretch
import OrthoDFA.Proofs.RandomNoise
import OrthoDFA.Proofs.Level2Round

/-!
# The random round: the proofs

A run with no bad step ends well within `stretches · Ns` probes: each stretch is shorter than
`Ns` probes, and each change of hypothesis lowers `Ψr`. A probe that can read a string wrong
(`PotWrong`) is a bad step, so every change of hypothesis is on real targets and keeps `GInv`.
The chance of a bad step is at most what the current stretch risks plus `Ψr` times what a fresh
one does (`anyB_le`), the chance part of #417 with its runs of clean probes replaced by stretches.
Over the reads, the wrong ones are null and the read field is good but for `noiseRisk`.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

section Compose

variable {S X E : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)
  (D : Measure X) [IsProbabilityMeasure D] (I F : S → Prop) (Ψ : S → ℕ) (b : ENNReal)

/-- The chance of a bad step is at most the current segment's, plus `Ψ` times a fresh one's. -/
theorem anyB_le
    (hsame : ∀ s x s', I s → step s x = .inl s' → same s s' → ¬ B s x → I s' ∧ Ψ s' = Ψ s)
    (hchg : ∀ s x s', I s → step s x = .inl s' → ¬ same s s' → ¬ B s x → I s' ∧ F s' ∧ Ψ s' < Ψ s)
    (hseg : ∀ s, I s → F s → ∀ N, (Measure.pi fun _ : Fin N => D)
      {xs | SegB step B same s (List.ofFn xs)} ≤ b) :
    ∀ N s, I s → (Measure.pi fun _ : Fin N => D) {xs | AnyB step B s (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin N => D) {xs | SegB step B same s (List.ofFn xs)} + Ψ s * b := by
  intro N
  induction N with
  | zero =>
    intro s _
    have : {xs : Fin 0 → X | AnyB step B s (List.ofFn xs)} = ∅ := by ext xs; simp [AnyB]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s hs
    rw [pi_cons_apply, pi_cons_apply]
    calc ∫⁻ x, (Measure.pi fun _ : Fin N => D)
          {xs | Fin.cons x xs ∈ {xs : Fin (N + 1) → X | AnyB step B s (List.ofFn xs)}} ∂D
        ≤ ∫⁻ x, ((Measure.pi fun _ : Fin N => D)
          {xs | Fin.cons x xs ∈ {xs : Fin (N + 1) → X | SegB step B same s (List.ofFn xs)}}
            + Ψ s * b) ∂D := lintegral_mono fun x => ?_
      _ = _ := by
          rw [lintegral_add_right _ measurable_const, lintegral_const, measure_univ, mul_one]
    simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, AnyB, SegB]
    by_cases hB : B s x
    · simp only [hB, true_or, Set.ofPred_true, measure_univ]
      exact le_add_right le_rfl
    · simp only [hB, false_or]
      rcases hst : step s x with s' | e
      · simp only [Sum.inl.injEq, exists_eq_left']
        by_cases hsm : same s s'
        · obtain ⟨hs', hΨ⟩ := hsame s x s' hs hst hsm hB
          simp only [hsm, true_and]
          rw [← hΨ]
          exact ih s' hs'
        · obtain ⟨hs', hF, hΨ⟩ := hchg s x s' hs hst hsm hB
          simp only [hsm, false_and, Set.ofPred_false, measure_empty, zero_add]
          calc _ ≤ _ := ih s' hs'
            _ ≤ b + Ψ s' * b := add_le_add (hseg s' hs' hF N) le_rfl
            _ = ((Ψ s' + 1 : ℕ) : ENNReal) * b := by push_cast; ring
            _ ≤ Ψ s * b := by gcongr; exact_mod_cast hΨ
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le

end Compose

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ) (y : ℝ)

/-- A run with no bad step ends well, given enough probes. -/
theorem ends_of_not_anyB {side : σ → Bool} :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), Inv read s → GInv read M side s →
      s.probeCount < Ns → Ψr (Fintype.card σ) s * Ns + (Ns - s.probeCount) ≤ xs.length →
      ¬ AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y
        fun _ x => x ∈ PotWrong M side C.walkStart (Fintype.card σ) read) s xs →
      EndsWell M (BadAt U θ) D ε
        {w | Disagrees read ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
            C.walkStart w}
        C.walkStart (harvestLaw read D C.walkStart
          ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩) y
        (runProbes read C s xs).2
  | [], s, _, _, hn, hlen, _ => by simp at hlen; omega
  | x :: xs, s, hs, hg, hn, hlen, hb => by
    simp only [AnyB, not_or, not_exists, not_and] at hb
    obtain ⟨hb1, hb2⟩ := hb
    rcases hst : applyProbe read C s x with s' | e
    · simp only [runProbes, hst]
      by_cases hsm : Same s s'
      · obtain ⟨hs', hΨ, hn'⟩ := step_same read C hs hst hsm.1 hsm.2
        have hlt : s'.probeCount < Ns := by
          have : s'.probeCount ≠ Ns := fun h => hb1 (Or.inr ⟨s', hst, hsm, h⟩)
          omega
        refine ends_of_not_anyB xs s' hs' (gInv_congr read M side hg hsm.1 hsm.2) hlt ?_
          (hb2 s' hst)
        rw [hΨ, hn']
        simp only [List.length_cons] at hlen
        omega
      · have hx : x ∉ PotWrong M side C.walkStart (Fintype.card σ) read :=
          fun h => hb1 (Or.inl ⟨hn, Or.inr h⟩)
        obtain ⟨hs', hg', hf, hΨ⟩ := step_change read C M side hs hg hx hst hsm
        have h0 : s'.probeCount = 0 := by rw [hf]; rfl
        refine ends_of_not_anyB xs s' hs' hg' (by omega) ?_ (hb2 s' hst)
        rw [h0, Nat.sub_zero]
        simp only [List.length_cons] at hlen
        have : (Ψr (Fintype.card σ) s' + 1) * Ns ≤ Ψr (Fintype.card σ) s * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    · simp only [runProbes, hst]
      by_contra hne
      exact hb1 (Or.inl ⟨hn, Or.inl ⟨e, hst, hne⟩⟩)

/-- Given the reads, the round fails to end well with chance at most `stretches` times what a
stretch risks, where at most `Gw` of the probes can read a string wrong. -/
theorem probe_level {side : σ → Bool} [IsProbabilityMeasure D]
    (P N₁ : ℕ) (G Gw θr εd' θpt' : ℝ)
    (hN : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      D.real (PotGood M U θ C.walkStart read T) ≤ G)
    (hNw : D.real (PotWrong M side C.walkStart (Fintype.card σ) read) ≤ Gw) (hGw0 : 0 ≤ Gw)
    (hN₁ : 1 ≤ N₁) (hN₁s : N₁ ≤ Ns) (hG0 : 0 ≤ G)
    (hGt : 4 * G ≤ C.edgeRate * C.edgeReadShare)
    (hyt : 2 * y ≤ C.edgeRate * C.edgeReadShare) (ha : 0 < C.testLevel)
    (hθs0 : 0 < C.startRate) (hθs : C.startRate < C.startAlternative)
    (hθs1 : C.startAlternative < 1) (hps : max (2 * G) y ≤ C.startRate) (hθe : 0 ≤ C.edgeRate)
    (hθpt0 : 0 < C.middleRate) (hθpt : C.middleRate < C.middleAlternative)
    (hθpt1 : C.middleAlternative < 1)
    (hfall : max (2 * G) y * (C.middleAlternative / C.middleRate)
        * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
      + (1 - max (2 * G) y) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare
      ≤ 1)
    (hεd0 : 0 < C.disagreementAlternative) (hεd : C.disagreementAlternative < C.disagreementRate)
    (hεd1 : C.disagreementRate < 1) (hε : C.disagreementRate ≤ ε) (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1)
    (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1) (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : θr ≤ (1 - θpt') * εd')
    (hP : Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns ≤ P) :
    (Measure.pi fun _ : Fin P => D)
        {xs | let r := runProbes read C initialState (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees read ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
            C.walkStart (harvestLaw read D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
      ≤ ENNReal.ofReal (Level2.stretches (Fintype.card σ) (Fintype.card α)
          * (stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' + Ns * Gw)) := by
  have hNs : 0 < Ns := by omega
  have hΨ := Ψr_start (α := α) (Fintype.card σ)
  set PW := PotWrong M side C.walkStart (Fintype.card σ) read
  set X : Level3State α → FreeMonoid α → Prop := fun _ x => x ∈ PW
  set b := ENNReal.ofReal
    (stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' + Ns * Gw)
  set I : Level3State α → Prop := fun s => Inv read s ∧ GInv read M side s ∧ s.probeCount < Ns
  have hI0 : I initialState := ⟨inv_start read, gInv_start read M side,
    by simp only [initialState, newHypothesis]; omega⟩
  have hseg : ∀ s, I s → s = newHypothesis s.tree s.edges → ∀ N,
      (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
            ≤ b := by
    rintro s ⟨hs, hg, -⟩ hf N
    refine seg_le read C M U θ D ε Ns y X (fun zs => ∃ w ∈ zs, w ∈ PW) (Ns * Gw) N₁ G θr εd'
      θpt' hs hf (cls_mem read M side hs hg) (by have := leaves_le read M side hg; omega) hN
      (fun _ _ x _ _ hx => ⟨x, by simp, hx⟩) (fun N => ?_) hN₁ hN₁s hG0 hGt hyt ha hθs0 hθs
      hθs1 hps hθe hθpt0 hθpt hθpt1 hfall hεd0 hεd hεd1 hε hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1
      hsep N
    calc _ ≤ (Measure.pi fun _ : Fin N => D).real {xs | ∃ w ∈ (List.ofFn xs).take Ns, w ∈ PW} :=
          measureReal_mono fun xs hxs => by
            obtain ⟨n, hn, -, w, hw, hwP⟩ := hxs
            refine ⟨w, ?_, hwP⟩
            rw [← min_eq_left (Finset.mem_Icc.1 hn).2, ← List.take_take] at hw
            exact List.mem_of_mem_take hw
      _ ≤ Ns * D.real PW := pi_mem_le D PW N Ns
      _ ≤ _ := by gcongr
  have hany := anyB_le (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same D I
    (fun s => s = newHypothesis s.tree s.edges) (Ψr (Fintype.card σ)) b
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨h1, h2, h3⟩ := step_same read C hs h hsm.1 hsm.2
      refine ⟨⟨h1, gInv_congr read M side hg hsm.1 hsm.2, ?_⟩, h2 _⟩
      have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h, hsm, hc⟩)
      omega)
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      have hx : x ∉ PW := fun hx => hb (Or.inl ⟨hn, Or.inr hx⟩)
      obtain ⟨h1, h2, h3, h4⟩ := step_change read C M side hs hg hx h hsm
      refine ⟨⟨h1, h2, ?_⟩, h3, h4⟩
      rw [h3]; simp only [newHypothesis]; omega)
    hseg P initialState hI0
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        {xs | AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) initialState
            (List.ofFn xs)} := by
        refine measure_mono fun xs hxs => ?_
        by_contra hb
        have hn0 : (initialState : Level3State α).probeCount = 0 := rfl
        refine hxs (ends_of_not_anyB read C M U θ D ε Ns y _ initialState hI0.1 hI0.2.1
          (by rw [hn0]; exact hNs) ?_ hb)
        rw [hn0, Nat.sub_zero, List.length_ofFn]
        have : (Ψr (Fintype.card σ) (initialState : Level3State α) + 1) * Ns
            ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    _ ≤ b + Ψr (Fintype.card σ) (initialState : Level3State α) * b :=
        hany.trans (add_le_add (hseg _ hI0 rfl P) le_rfl)
    _ = ((Ψr (Fintype.card σ) (initialState : Level3State α) + 1 : ℕ) : ENNReal) * b := by
        push_cast; ring
    _ ≤ (Level2.stretches (Fintype.card σ) (Fintype.card α) : ENNReal) * b := by gcongr
    _ = _ := by
        rw [ENNReal.ofReal_mul (by positivity), ENNReal.ofReal_natCast]

theorem le_prefixMax [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (u : FreeMonoid α) : D.real {x | prefixOf x k = u} ≤ prefixMax D k :=
  le_ciSup (f := fun u => D.real {x | prefixOf x k = u}) ⟨1, by
    rintro _ ⟨v, rfl⟩; exact measureReal_le_one⟩ u

theorem prefixMax_pos [Fintype α] (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) :
    0 < prefixMax D k := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  by_contra h
  push_neg at h
  have h0 : ∀ u : FreeMonoid α, D {x | prefixOf x k = u} = 0 := by
    intro u
    have h1 := (le_prefixMax D k u).trans h
    have h2 : D.real {x | prefixOf x k = u} = 0 := le_antisymm h1 measureReal_nonneg
    rwa [measureReal_eq_zero_iff] at h2
  have : (Set.univ : Set (FreeMonoid α)) = ⋃ u, {x | prefixOf x k = u} := by
    ext x; simp
  have hu := measure_iUnion_null (μ := D) h0
  rw [← this, measure_univ] at hu
  exact one_ne_zero hu


theorem randomRoundCorrect_holds : RandomRoundCorrect := by
  intro α _ _ σ _ M side U θ Ω _ μ _ R P C ε y
  obtain ⟨D, _, L, hL⟩ := P
  obtain ⟨read, hmeas, hind, hU, hwrong⟩ := R
  simp only [roundRisk]
  refine le_iInf₂ fun kn hA => ?_
  obtain ⟨Ns, N₁, G, θr, εd', θpt'⟩ := kn
  obtain ⟨hN₁, hN₁s, hθ, hG0, ha, hθs0, hθs, hθs1, hθe, hθpt0, hθpt, hθpt1, hεd0, hεd, hεd1,
    hε, hθr0, hθr1, hεd'0, hεd'1, hθpt'0, hθpt'1, hsep, hP, hGt, hyt, hps, hfall⟩ := hA
  simp only at hN₁ hN₁s hG0 hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP hGt hps hfall
  unfold roundBound
  simp only
  set P := C.probeBudget
  set p₀ := prefixMax D C.walkStart
  have hp₀ : 0 < p₀ := prefixMax_pos D C.walkStart
  have hpre : ∀ u : FreeMonoid α, u.toList.length = C.walkStart
      → D.real {x | prefixOf x C.walkStart = u} ≤ p₀ :=
    fun u _ => le_prefixMax D C.walkStart u
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set W := {ω | ¬ NoWrong M side (read · ω)}
  set Nn := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
    D.real (PotGood M U θ C.walkStart (read · ω) T) ≤ G}
  have hW : μ W = 0 := by
    have : W ⊆ ⋃ z, {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} := by
      intro ω hω
      simp only [W, NoWrong, Set.mem_ofPred_eq, not_forall, not_not] at hω
      obtain ⟨z, hz⟩ := hω
      exact Set.mem_iUnion.2 ⟨z, hz⟩
    exact measure_mono_null this (measure_iUnion_null hwrong)
  have hNn := noise_le M U θ μ read D C.walkStart L p₀ G hmeas hind hU hL hp₀ hpre hθ
  set c := ENNReal.ofReal (Level2.stretches (Fintype.card σ) (Fintype.card α)
    * stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt')
  set B := toMeasurable μ (W ∪ Nn)
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε {x | Disagrees (read · ω) ⟨r.1.tree, r.1.edges⟩ C.walkStart x}
          C.walkStart (harvestLaw (read · ω) D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
        ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ W ∪ Nn := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or] at hω'
      obtain ⟨hωW, hωN⟩ := hω'
      simp only [W, Nn, Set.mem_ofPred_eq, not_not] at hωW hωN
      have := probe_level (read · ω) C M U θ D ε Ns y (side := side) P N₁ G 0 θr εd' θpt'
        hωN (by rw [potWrong_empty hωW]; simp) le_rfl
        hN₁ hN₁s hG0 hGt hyt ha hθs0 hθs hθs1 hps hθe hθpt0 hθpt hθpt1 hfall hεd0 hεd hεd1 hε
        hθr0 hθr1 hεd'0 hεd'1 hθpt'0 hθpt'1 hsep hP
      rwa [mul_zero, add_zero] at this
  have hnn :
      0 ≤ stretchRisk C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' :=
    stretchRisk_nonneg C (Fintype.card σ) (Fintype.card α) Ns N₁ θr εd' θpt' ha.le hθr0 hθr1
      hεd'0 hεd'1 hθpt'0 hθpt'1
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ W + μ Nn) := by
        gcongr
        rw [measure_toMeasurable]
        exact measure_union_le _ _
    _ ≤ c + ENNReal.ofReal (noiseRisk (Fintype.card σ) (Fintype.card α) L (3 / 2 * θ) p₀ G) := by
        rw [hW, zero_add]
        gcongr
    _ = _ := by
        rw [add_comm c, ← ENNReal.ofReal_add (by unfold noiseRisk; positivity) (by positivity)]
        unfold Level2.stretches
        push_cast
        ring_nf

end Random

end OrthoDFA
