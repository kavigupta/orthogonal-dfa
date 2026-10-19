import OrthoDFA.Proofs.RandomStretch
import OrthoDFA.Proofs.RandomGood
import OrthoDFA.Proofs.Level2Round

/-!
# The random round: the proofs

A run with no bad step ends well within `stretches · Ns` probes: each stretch is shorter than
`Ns` probes, and each change of hypothesis lowers `Ψr`. The chance of a bad step is at most what
the current stretch risks plus what each of the at most `Ψr` fresh ones that can follow does, at
its own level (`anyB_le`). Over the reads, the wrong ones are null and the probes harvest at good
read-states within their bounds but for `tailRisk` at the ends and at the middles.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

section Compose

variable {S X E : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
  (step : S → X → S ⊕ E) (B : S → X → Prop) (same : S → S → Prop)
  (D : Measure X) [IsProbabilityMeasure D] (I F : S → Prop) (Ψ : S → ℕ)

/-- The chance of a bad step is at most the current segment's, plus a fresh one's at each of the
next `Ψ` indices. -/
theorem anyB_le (idx : S → ℕ) (b : ℕ → ENNReal)
    (hsame : ∀ s x s', I s → step s x = .inl s' → same s s' → ¬ B s x →
      I s' ∧ Ψ s' = Ψ s ∧ idx s' = idx s)
    (hchg : ∀ s x s', I s → step s x = .inl s' → ¬ same s s' → ¬ B s x →
      I s' ∧ F s' ∧ Ψ s' < Ψ s ∧ idx s' = idx s + 1)
    (hseg : ∀ s, I s → F s → ∀ N, (Measure.pi fun _ : Fin N => D)
      {xs | SegB step B same s (List.ofFn xs)} ≤ b (idx s)) :
    ∀ N s, I s → (Measure.pi fun _ : Fin N => D) {xs | AnyB step B s (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin N => D) {xs | SegB step B same s (List.ofFn xs)}
        + ∑ i ∈ Finset.Ico (idx s + 1) (idx s + 1 + Ψ s), b i := by
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
            + ∑ i ∈ Finset.Ico (idx s + 1) (idx s + 1 + Ψ s), b i) ∂D :=
          lintegral_mono fun x => ?_
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
        · obtain ⟨hs', hΨ, hi⟩ := hsame s x s' hs hst hsm hB
          simp only [hsm, true_and]
          rw [← hΨ, ← hi]
          exact ih s' hs'
        · obtain ⟨hs', hF, hΨ, hi⟩ := hchg s x s' hs hst hsm hB
          simp only [hsm, false_and, Set.ofPred_false, measure_empty, zero_add]
          have hih := ih s' hs'
          have hsg := hseg s' hs' hF N
          rw [hi, show idx s + 1 + 1 = idx s + 2 by omega] at hih
          rw [hi] at hsg
          calc _ ≤ _ := hih
            _ ≤ b (idx s + 1) + ∑ i ∈ Finset.Ico (idx s + 2) (idx s + 2 + Ψ s'), b i :=
                add_le_add hsg le_rfl
            _ ≤ _ := Level2.sum_Ico_succ_le b (fun _ => bot_le) (by omega)
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le

end Compose

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (D : Measure (FreeMonoid α))
  (ε : ℝ) (Ns : ℕ) (y : ℝ)

/-- A run with no bad step ends well, given enough probes. -/
theorem ends_of_not_anyB {side : σ → Bool} (hW : NoWrong M side read)
    (X : Level3State α → FreeMonoid α → Prop) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), RInv read s → s.probeCount < Ns →
      Ψr (Fintype.card σ) s * Ns + (Ns - s.probeCount) ≤ xs.length →
      ¬ AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) s xs →
      EndsWell M (BadAt U θ) D ε
        (searchSet read C.walkStart
            ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
          ∪ endsSet read C.walkStart
            ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩)
        C.walkStart (harvestLaw read D C.walkStart
          ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩) y
        (runProbes read C s xs).2
  | [], s, _, hn, hlen, _ => by simp at hlen; omega
  | x :: xs, s, hs, hn, hlen, hb => by
    simp only [AnyB, not_or, not_exists, not_and] at hb
    obtain ⟨hb1, hb2⟩ := hb
    rcases hst : applyProbe read C s x with s' | e
    · simp only [runProbes, hst]
      by_cases hsm : Same s s'
      · obtain ⟨hs', hΨ, hn', -⟩ := step_same read C hs.1 hst hsm.1 hsm.2
        replace hs' : RInv read s' := ⟨hs', hsm.1 ▸ hs.2⟩
        have hlt : s'.probeCount < Ns := by
          have : s'.probeCount ≠ Ns := fun h => hb1 (Or.inr ⟨s', hst, hsm, h⟩)
          omega
        refine ends_of_not_anyB hW X xs s' hs' hlt ?_ (hb2 s' hst)
        rw [hΨ, hn']
        simp only [List.length_cons] at hlen
        omega
      · obtain ⟨hs', hf, -, hΨ⟩ := step_change read C hW hs hst hsm
        have h0 : s'.probeCount = 0 := by rw [hf]; rfl
        refine ends_of_not_anyB hW X xs s' hs' (by omega) ?_ (hb2 s' hst)
        rw [h0, Nat.sub_zero]
        simp only [List.length_cons] at hlen
        have : (Ψr (Fintype.card σ) s' + 1) * Ns ≤ Ψr (Fintype.card σ) s * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    · simp only [runProbes, hst]
      by_contra hne
      exact hb1 (Or.inl ⟨hn, Or.inl ⟨e, hst, hne⟩⟩)

theorem stretchRisk_nonneg {a : ℝ} (ha : 0 < a) (hρ0 : 0 < C.disagreementRate)
    (hρ1 : C.disagreementRate < 1) (hμ0 : 0 < C.middleRate) (hμ1 : C.middleRate < 1 / 2)
    (hθ0 : 0 < C.endsRate) (hθ1 : C.endsRate < 1 / 2)
    (Ns N₁ : ℕ) {θr εd' θe' θpt' : ℝ} (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1)
    (hθe'0 : 0 ≤ θe') (hθe'1 : θe' ≤ εd') (hθpt'0 : 0 ≤ θpt') (hθpt'1 : θpt' ≤ 1) :
    0 ≤ stretchRisk C a Ns N₁ θr εd' θe' θpt' := by
  unfold stretchRisk Level3Settings.disagreementAlternative Level3Settings.middleAlternative
    Level3Settings.endsAlternative
  have h9 : 0 ≤ C.endsRate / (2 * C.endsRate) := by positivity
  have h10 : 0 ≤ (1 - C.endsRate) / (1 - 2 * C.endsRate) :=
    div_nonneg (by linarith) (by linarith)
  have h11 : 0 ≤ 1 - θe' := by linarith
  have h1 : 0 ≤ C.disagreementRate / (C.disagreementRate / 2) := by positivity
  have h2 : 0 ≤ (1 - C.disagreementRate) / (1 - C.disagreementRate / 2) :=
    div_nonneg (by linarith) (by linarith)
  have h3 : 0 ≤ C.middleRate / (2 * C.middleRate) := by positivity
  have h4 : 0 ≤ (1 - C.middleRate) / (1 - 2 * C.middleRate) :=
    div_nonneg (by linarith) (by linarith)
  have h5 : 0 ≤ 1 - εd' := by linarith
  have h6 : 0 ≤ 1 - θpt' := by linarith
  have h7 : 0 ≤ 1 - θr := by linarith
  have h8 : 0 ≤ 1 - (εd' - θe') / 2 := by linarith
  positivity

theorem level_pos (hδ : 0 < C.δ) (j : ℕ) : 0 < C.level j := by
  unfold Level3Settings.level
  positivity

theorem probe_level {side : σ → Bool} (hW : NoWrong M side read) [IsProbabilityMeasure D]
    (P N₁ : ℕ) (Ge θgpt ηm pM θr εd' θe' θpt' : ℝ)
    (hNe : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
      D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .ends) ≤ Ge)
    (hNm : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
      D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .middles)
        ≤ θgpt * D.real {x | searches (probeOrAgree read T E C.walkStart x) = true} + ηm)
    (hN₁ : 1 ≤ N₁) (hN₁s : N₁ ≤ Ns)
    (hδ : 0 < C.δ)
    (hθs0 : 0 < C.endsRate) (hθs1 : C.endsRate < 1 / 2) (hps : max (2 * Ge) y ≤ C.endsRate)
    (hθpt0 : 0 < C.middleRate) (hθpt1 : C.middleRate < 1 / 2) (hθg : 4 * θgpt ≤ C.middleRate)
    (hpM0 : 0 ≤ pM) (hyM : y ≤ pM) (hηm : 4 * ηm ≤ pM)
    (hfall : pM * (C.middleAlternative / C.middleRate)
        * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ (C.searchShare - 1)
      + (1 - pM) * ((1 - C.middleAlternative) / (1 - C.middleRate)) ^ C.searchShare ≤ 1)
    (hεd0 : 0 < C.disagreementRate) (hεd1 : C.disagreementRate < 1) (hε : C.disagreementRate ≤ ε)
    (hθr0 : 0 ≤ θr) (hθr1 : θr ≤ 1) (hεd'0 : 0 ≤ εd') (hεd'1 : εd' ≤ 1)
    (hθe'0 : 0 ≤ θe') (hθe'1 : θe' ≤ εd')
    (hθpt'0 : C.middleAlternative ≤ θpt') (hθpt'1 : θpt' ≤ 1)
    (hsep : θr ≤ (1 - θpt') * (εd' - θe'))
    (hP : Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns ≤ P) :
    (Measure.pi fun _ : Fin P => D)
        {xs | let r := runProbes read C initialState (List.ofFn xs)
          ¬ EndsWell M (BadAt U θ) D ε
            (searchSet read C.walkStart ⟨r.1.tree, r.1.edges⟩
              ∪ endsSet read C.walkStart ⟨r.1.tree, r.1.edges⟩)
            C.walkStart (harvestLaw read D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
      ≤ ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
          stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt') := by
  have hNs : 0 < Ns := by omega
  have hst : RInv read (initialState : Level3State α) := ⟨inv_start read, reached_start read⟩
  have hΨ := Ψr_start (α := α) (Fintype.card σ)
  have hm : C.middleAlternative = 2 * C.middleRate := rfl
  have hd : C.disagreementAlternative = C.disagreementRate / 2 := rfl
  have hsa : C.endsAlternative = 2 * C.endsRate := rfl
  set b : ℕ → ENNReal := fun j => ENNReal.ofReal (stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set X : Level3State α → FreeMonoid α → Prop := fun _ _ => False
  have hseg : ∀ s, RInv read s → s = newHypothesis s.tree s.edges s.hypotheses → ∀ N,
      (Measure.pi fun _ : Fin N => D)
        {xs | SegB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same s (List.ofFn xs)}
            ≤ b s.hypotheses := by
    intro s hs hf N
    have := seg_le read C M U θ D ε Ns y X (fun _ => False) 0 N₁ Ge θgpt ηm pM θr εd' θe' θpt'
      hs.1 hf (cls_mem read hW hs) hNe hNm (fun _ _ _ _ _ h => h)
      (fun N => by simp) hN₁ hN₁s (level_pos C hδ _) hθs0 (by rw [hsa]; linarith)
      (by rw [hsa]; linarith) hps hθpt0 (by rw [hm]; linarith) (by rw [hm]; linarith) hθg hpM0
      hyM hηm hfall
      (by rw [hd]; linarith) (by rw [hd]; linarith) hεd1 hε hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1
      hθpt'0 hθpt'1 hsep N
    rwa [add_zero] at this
  have hany := anyB_le (applyProbe read C) (BadStep read C M U θ D ε Ns y X) Same D (RInv read)
    (fun s => s = newHypothesis s.tree s.edges s.hypotheses) (Ψr (Fintype.card σ))
    Level3State.hypotheses b
    (fun s x s' hs h hsm _ => by
      obtain ⟨h1, h2, -, h4⟩ := step_same read C hs.1 h hsm.1 hsm.2
      exact ⟨⟨h1, hsm.1 ▸ hs.2⟩, h2 _, h4⟩)
    (fun s x s' hs h hsm _ => by
      obtain ⟨h1, h2, h3, h4⟩ := step_change read C hW hs h hsm
      exact ⟨h1, h2, h4, h3⟩) hseg P initialState hst
  have hnn : ∀ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
      0 ≤ stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt' := fun j _ =>
    stretchRisk_nonneg C (level_pos C hδ j) hεd0 hεd1 hθpt0 hθpt1 hθs0 hθs1 Ns N₁ hθr1 hεd'0
      hεd'1 hθe'0 hθe'1 (le_trans (by rw [hm]; linarith) hθpt'0) hθpt'1
  have h0 : (initialState : Level3State α).hypotheses = 0 := rfl
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        {xs | AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y X) initialState
            (List.ofFn xs)} := by
        refine measure_mono fun xs hxs => ?_
        by_contra hb
        have hn0 : (initialState : Level3State α).probeCount = 0 := rfl
        refine hxs (ends_of_not_anyB read C M U θ D ε Ns y hW X _ initialState hst
          (by rw [hn0]; exact hNs) ?_ hb)
        rw [hn0, Nat.sub_zero, List.length_ofFn]
        have : (Ψr (Fintype.card σ) (initialState : Level3State α) + 1) * Ns
            ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    _ ≤ b 0 + ∑ i ∈ Finset.Ico 1 (1 + Ψr (Fintype.card σ) (initialState : Level3State α)), b i := by
        refine hany.trans (add_le_add ?_ ?_)
        · exact h0 ▸ hseg _ hst rfl P
        · rw [h0]
    _ = ∑ i ∈ Finset.range (1 + Ψr (Fintype.card σ) (initialState : Level3State α)), b i := by
        rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot
          (show 0 < 1 + Ψr (Fintype.card σ) (initialState : Level3State α) by omega)]
    _ ≤ ∑ i ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)), b i :=
        Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 (by omega))
    _ = _ := (ENNReal.ofReal_sum_of_nonneg hnn).symm

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


/-- The round's failure chance is at most `roundBoundAt` at admissible knobs, with no
length-`walkStart` prefix above `p₀`. -/
theorem round_le_at {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (θ : ℝ)
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (R : ReadOracle μ M side U) (P : PrefixDistribution α) (C : Level3Settings) (ε y p₀ : ℝ)
    (kn : Knobs) (hA : kn.Admissible (σ := σ) (α := α) θ P.L C ε y) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell M (BadAt U θ) P.D ε
            (searchSet (R.read · ω) C.walkStart result.hypothesis
              ∪ endsSet (R.read · ω) C.walkStart result.hypothesis)
            C.walkStart (harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis) y
            result.ending} ∂μ
      ≤ ENNReal.ofReal (roundBoundAt (α := α) σ θ p₀ P.L C kn) := by
  obtain ⟨D, _, L, hL⟩ := P
  obtain ⟨read, hmeas, hind, hU, hwrong⟩ := R
  obtain ⟨Ns, N₁, Ge, θgpt, ηm, pM, θr, εd', θe', θpt'⟩ := kn
  obtain ⟨hN₁, hN₁s, hθ, hGe0, hδ, hθs0, hθs1, hθpt0, hθpt1, hεd0, hεd1,
    hε, hθr0, hθr1, hεd'0, hεd'1, hθe'0, hθe'1, hθpt'0, hθpt'1, hsep, hP, hps, hθg, hθg4,
    hpM0, hyM, hηm, hfall⟩ := hA
  simp only at hN₁ hN₁s hGe0 hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP hps
  simp only at hθg hθg4 hpM0 hyM hηm
  simp only at hfall hpre ⊢
  set P := C.probeBudget
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set W := {ω | ¬ NoWrong M side (read · ω)}
  set Ne := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
    D.real (goodAt (read · ω) C.walkStart ⟨T, E⟩ M U θ .ends)
      ≤ 9 * θ * (Fintype.card σ + 1) + (Ge - 9 * θ * (Fintype.card σ + 1))}
  set Nm := {ω | ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
    D.real (goodAt (read · ω) C.walkStart ⟨T, E⟩ M U θ .middles)
      ≤ θgpt * D.real {x | searches (probeOrAgree (read · ω) T E C.walkStart x) = true} + ηm}
  have hW : μ W = 0 := by
    have : W ⊆ ⋃ z, {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} := by
      intro ω hω
      simp only [W, NoWrong, Set.mem_ofPred_eq, not_forall, not_not] at hω
      obtain ⟨z, hz⟩ := hω
      exact Set.mem_iUnion.2 ⟨z, hz⟩
    exact measure_mono_null this (measure_iUnion_null hwrong)
  have hNe := endsNoise_le M U θ hmeas hind hU hθ D C.walkStart (n := Fintype.card σ)
    (η := Ge - 9 * θ * (Fintype.card σ + 1)) hL hp₀ hpre
  have hNm := midNoise_le M U θ hmeas hind hU hθ D C.walkStart (n := Fintype.card σ) (η := ηm)
    hL hp₀ hpre hθg
  set c := ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
    stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set B := toMeasurable μ (W ∪ (Ne ∪ Nm))
  have hpt : ∀ ω, (Measure.pi fun _ : Fin P => D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
        ¬ EndsWell M (BadAt U θ) D ε
          (searchSet (read · ω) C.walkStart ⟨r.1.tree, r.1.edges⟩
            ∪ endsSet (read · ω) C.walkStart ⟨r.1.tree, r.1.edges⟩)
          C.walkStart (harvestLaw (read · ω) D C.walkStart ⟨r.1.tree, r.1.edges⟩) y r.2}
        ≤ c + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · have hω' : ω ∉ W ∪ (Ne ∪ Nm) := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Set.mem_union, not_or] at hω'
      obtain ⟨hωW, hωE, hωM⟩ := hω'
      simp only [W, Ne, Nm, Set.mem_ofPred_eq, not_not] at hωW hωE hωM
      exact probe_level (read · ω) C M U θ D ε Ns y hωW P N₁ Ge θgpt ηm pM θr εd' θe' θpt'
        (fun T hT E hE => (hωE T hT E hE).trans (by linarith)) hωM
        hN₁ hN₁s hδ hθs0 hθs1 hps hθpt0 hθpt1 hθg4 hpM0 hyM hηm hfall hεd0 hεd1 hε
        hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP
  have hnn : 0 ≤ ∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
      stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt' :=
    Finset.sum_nonneg fun j _ => stretchRisk_nonneg C (level_pos C hδ j) hεd0 hεd1 hθpt0 hθpt1
      hθs0 hθs1 Ns N₁ hθr1 hεd'0 hεd'1 hθe'0 hθe'1 (le_trans (by
        show 0 ≤ 2 * C.middleRate; linarith) hθpt'0) hθpt'1
  calc _ ≤ ∫⁻ ω, (c + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = c + μ B := by
        rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ c + (μ W + (μ Ne + μ Nm)) := by
        gcongr
        rw [measure_toMeasurable]
        exact (measure_union_le _ _).trans (add_le_add le_rfl (measure_union_le _ _))
    _ ≤ c + (ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀
            (3 * (Fintype.card σ + 1) * (3 / 2 * θ)) (Ge - 9 * θ * (Fintype.card σ + 1)))
          + ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀
            ((L + 1) * (Fintype.card σ + 1) * (3 / 2 * θ)) ηm)) := by
        rw [hW, zero_add]
        gcongr
    _ = _ := by
        rw [add_comm c, ← ENNReal.ofReal_add (by unfold tailRisk; positivity)
          (by unfold tailRisk; positivity),
          ← ENNReal.ofReal_add (by unfold tailRisk; positivity) hnn]
        rfl

theorem roundRiskBound_holds : RoundRiskBound := by
  intro α _ _ σ _ M side U θ Ω _ μ _ R P C ε y
  refine le_iInf₂ fun kn hA => ?_
  exact round_le_at M side U θ μ R P C ε y _ kn hA (prefixMax_pos P.D C.walkStart)
    fun u => le_prefixMax P.D C.walkStart u


end Random

end OrthoDFA
