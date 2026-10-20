import OrthoDFA.Proofs.EpsReads

/-!
# Level 3 with rare wrong reads: the proof

Given the reads, the round fails to end well only where a stretch risks it, as at level 3, or
where some step reads a string wrong (`eps_probe_level`): a step that reads nothing wrong keeps
`WitInv`, so the tree stays in the class level 3's stretches need. The run stopped at the first
step that reads a string wrong (`stopAt`) has level 3's stretches. Over the reads, the probes
harvest within their bounds but for the tails at every state's rate (`harvest_noise`), and some
step reads a string wrong with chance at most
`εw · probeBudget · ((L + 1)|Q| + (|Q| + 1)²|Σ|)` (`pi_wrong_le`).

`Level4RoundCorrect` takes level 3's knobs at `log(16(|Q| + 1)K/δ)`: the tails risk `δ/16` at each
site, the stretches `3δ/4` and the wrong reads `δ/8`.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

section Stop

variable {S X E : Type*} (step : S → X → S ⊕ E) (B₂ : S → X → Prop) (e₀ : E)

open scoped Classical in
/-- `step`, ending at a step `B₂`. -/
noncomputable def stopAt (s : S) (x : X) : S ⊕ E := if B₂ s x then .inr e₀ else step s x

theorem stopAt_inl {s s' : S} {x : X} (h : stopAt step B₂ e₀ s x = .inl s') :
    ¬ B₂ s x ∧ step s x = .inl s' := by
  classical
  unfold stopAt at h
  split_ifs at h with hB
  exact ⟨hB, h⟩

theorem segB_stopAt (B : S → X → Prop) (same : S → S → Prop) :
    ∀ s xs, SegB (stopAt step B₂ e₀) B same s xs → SegB step B same s xs
  | _, [], h => h.elim
  | _, _ :: xs, h => h.elim Or.inl fun ⟨s', hs', hsm, h⟩ =>
      Or.inr ⟨s', (stopAt_inl step B₂ e₀ hs').2, hsm, segB_stopAt B same s' xs h⟩

/-- A run with a bad step, each bad step `B₁` or `B₂`, has a `B₁` step before any `B₂`, or a `B₂`
step. -/
theorem anyB_split (B B₁ : S → X → Prop) (hB : ∀ s x, B s x → B₁ s x ∨ B₂ s x) :
    ∀ s xs, AnyB step B s xs → AnyB (stopAt step B₂ e₀) B₁ s xs ∨ AnyB step B₂ s xs
  | _, [], h => h.elim
  | s, x :: xs, h => by
    classical
    rcases h with h | ⟨s', hs', h⟩
    · rcases hB s x h with h | h
      · exact Or.inl (Or.inl h)
      · exact Or.inr (Or.inl h)
    · by_cases h₂ : B₂ s x
      · exact Or.inr (Or.inl h₂)
      · rcases anyB_split B B₁ hB s' xs h with h | h
        · exact Or.inl (Or.inr ⟨s', by simp [stopAt, h₂, hs'], h⟩)
        · exact Or.inr (Or.inr ⟨s', hs', h⟩)

end Stop

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (A sl : ℝ)
  (D : Measure (FreeMonoid α)) (ε : ℝ) (Ns : ℕ) (y : ℝ)

/-- The steps `W` that a step changing the hypothesis must avoid to keep the invariants. -/
def KeepsInv (W : Level3State α → FreeMonoid α → Prop) : Prop :=
  ∀ ⦃s s' : Level3State α⦄ ⦃x : FreeMonoid α⦄, Inv read s → WitInv M side s → ¬ W s x →
    applyProbe read C s x = .inl s' → ¬ (s'.tree = s.tree ∧ s'.edges = s.edges) →
    Inv read s' ∧ WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s

/-- A run with no bad step, a step `W` counted as bad, ends well, given enough probes. -/
theorem eps_ends_of_not_anyB (W : Level3State α → FreeMonoid α → Prop)
    (hW : KeepsInv read C M side W) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), Inv read s → WitInv M side s →
      s.probeCount < Ns → Ψr (Fintype.card σ) s * Ns + (Ns - s.probeCount) ≤ xs.length →
      ¬ AnyB (applyProbe read C) (BadStep read C M U A sl D ε Ns y W) s xs →
      EndsWell
        { target := M,
          undecided := U,
          harvestSlope := A,
          harvestSlack := sl,
          probes := D,
          disagreeRate := ε,
          disagree := searchSet read C.walkStart
            ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩
            ∪ endsSet read C.walkStart
            ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩,
          walkStart := C.walkStart,
          harvestLaw := harvestLaw read D C.walkStart
            ⟨(runProbes read C s xs).1.tree, (runProbes read C s xs).1.edges⟩,
          yield := y,
          ending := (runProbes read C s xs).2 }
  | [], s, _, _, hn, hlen, _ => by simp at hlen; omega
  | x :: xs, s, hs, hg, hn, hlen, hb => by
    simp only [AnyB, not_or, not_exists, not_and] at hb
    obtain ⟨hb1, hb2⟩ := hb
    rcases hst : applyProbe read C s x with s' | e
    · simp only [runProbes, hst]
      by_cases hsm : Same s s'
      · obtain ⟨hs', hΨ, hn', -⟩ := step_same read C hs hst hsm.1 hsm.2
        have hlt : s'.probeCount < Ns := by
          have : s'.probeCount ≠ Ns := fun h => hb1 (Or.inr ⟨s', hst, hsm, h⟩)
          omega
        refine eps_ends_of_not_anyB W hW xs s' hs' (witInv_congr M side hg hsm.1 hsm.2) hlt ?_
          (hb2 s' hst)
        rw [hΨ, hn']
        simp only [List.length_cons] at hlen
        omega
      · have hx : ¬ W s x := fun h => hb1 (Or.inl ⟨hn, Or.inr h⟩)
        obtain ⟨hs', hg', hf, -, hΨ⟩ := hW hs hg hx hst hsm
        have h0 : s'.probeCount = 0 := by rw [hf]; rfl
        refine eps_ends_of_not_anyB W hW xs s' hs' hg' (by omega) ?_ (hb2 s' hst)
        rw [h0, Nat.sub_zero]
        simp only [List.length_cons] at hlen
        have : (Ψr (Fintype.card σ) s' + 1) * Ns ≤ Ψr (Fintype.card σ) s * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    · simp only [runProbes, hst]
      by_contra hne
      exact hb1 (Or.inl ⟨hn, Or.inl ⟨e, hst, hne⟩⟩)

/-- Given the reads, the round fails to end well with chance at most what the stretches risk at
their levels, plus the chance that some step is `W`. -/
theorem eps_probe_level (W : Level3State α → FreeMonoid α → Prop) (hW : KeepsInv read C M side W)
    [IsProbabilityMeasure D] (P N₁ : ℕ)
    (Ge θgpt ηm pM θr εd' θe' θpt' : ℝ)
    (hNe : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
      ¬ EndsWell (outcomeAt read C M U A sl D ε y T E (.harvest .ends)) →
        (harvestLaw read D C.walkStart ⟨T, E⟩ .ends).real Set.univ ≤ max (2 * Ge) y)
    (hNm : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ∀ E, EdgesInto T E →
      ¬ EndsWell (outcomeAt read C M U A sl D ε y T E (.harvest .middles)) →
        (harvestLaw read D C.walkStart ⟨T, E⟩ .middles).real Set.univ
          ≤ max (2 * (θgpt * D.real {x | searches (probeOrAgree read T E C.walkStart x) = true}
            + ηm)) y)
    (hN₁ : 1 ≤ N₁) (hN₁s : N₁ ≤ Ns) (hδ : 0 < C.δ)
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
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := sl,
              probes := D,
              disagreeRate := ε,
              disagree := searchSet read C.walkStart ⟨r.1.tree, r.1.edges⟩
                ∪ endsSet read C.walkStart ⟨r.1.tree, r.1.edges⟩,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw read D C.walkStart ⟨r.1.tree, r.1.edges⟩,
              yield := y,
              ending := r.2 }}
      ≤ ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
          stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
        + (Measure.pi fun _ : Fin P => D)
          {xs | AnyB (applyProbe read C) W initialState (List.ofFn xs)} := by
  have hNs : 0 < Ns := by omega
  have hΨ := Ψr_start (α := α) (Fintype.card σ)
  have hm : C.middleAlternative = 2 * C.middleRate := rfl
  have hd : C.disagreementAlternative = C.disagreementRate / 2 := rfl
  have hsa : C.endsAlternative = 2 * C.endsRate := rfl
  set step₂ := stopAt (applyProbe read C) W (RoundEnd.failed : RoundEnd Level3Site)
  set B₁ := BadStep read C M U A sl D ε Ns y fun _ _ => False
  set b : ℕ → ENNReal := fun j => ENNReal.ofReal (stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set I : Level3State α → Prop := fun s => Inv read s ∧ WitInv M side s ∧ s.probeCount < Ns
  have hI0 : I initialState := ⟨inv_start read, witInv_start M side,
    by simp only [initialState, newHypothesis]; omega⟩
  have hseg : ∀ s, I s → s = newHypothesis s.tree s.edges s.hypotheses → ∀ N,
      (Measure.pi fun _ : Fin N => D) {xs | SegB step₂ B₁ Same s (List.ofFn xs)}
        ≤ b s.hypotheses := by
    rintro s ⟨hs, hg, -⟩ hf N
    refine (measure_mono fun xs hxs => segB_stopAt _ _ _ _ _ _ _ hxs).trans ?_
    have := seg_le read C M U A sl D ε Ns y (fun _ _ => False) (fun _ => False) 0 N₁ Ge θgpt ηm pM
      θr εd' θe' θpt' hs hf (cls_mem' M side read hs hg) hNe hNm (fun _ _ _ _ _ h => h.elim)
      (fun N => by simp) hN₁ hN₁s (level_pos C hδ _) hθs0 (by rw [hsa]; linarith)
      (by rw [hsa]; linarith) hps hθpt0 (by rw [hm]; linarith) (by rw [hm]; linarith) hθg hpM0
      hyM hηm hfall
      (by rw [hd]; linarith) (by rw [hd]; linarith) hεd1 hε hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1
      hθpt'0 hθpt'1 hsep N
    rw [add_zero] at this
    exact this
  have hany := anyB_le step₂ B₁ Same D I
    (fun s => s = newHypothesis s.tree s.edges s.hypotheses) (Ψr (Fintype.card σ))
    Level3State.hypotheses b
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      have h := (stopAt_inl _ _ _ h).2
      obtain ⟨h1, h2, h3, h4⟩ := step_same read C hs h hsm.1 hsm.2
      refine ⟨⟨h1, witInv_congr M side hg hsm.1 hsm.2, ?_⟩, h2 _, h4⟩
      have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h, hsm, hc⟩)
      omega)
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨hW', h⟩ := stopAt_inl _ _ _ h
      obtain ⟨h1, h2, h3, h4, h5⟩ := hW hs hg hW' h hsm
      refine ⟨⟨h1, h2, ?_⟩, h3, h5, h4⟩
      rw [h3]; simp only [newHypothesis]; omega)
    hseg P initialState hI0
  have hnn : ∀ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
      0 ≤ stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt' := fun j _ =>
    stretchRisk_nonneg C (level_pos C hδ j) hεd0 hεd1 hθpt0 hθpt1 hθs0 hθs1 Ns N₁ hθr1 hεd'0
      hεd'1 hθe'0 hθe'1 (le_trans (by rw [hm]; linarith) hθpt'0) hθpt'1
  have h0 : (initialState : Level3State α).hypotheses = 0 := rfl
  calc _ ≤ (Measure.pi fun _ : Fin P => D)
        ({xs | AnyB step₂ B₁ initialState (List.ofFn xs)}
          ∪ {xs | AnyB (applyProbe read C) W initialState (List.ofFn xs)}) := by
        refine measure_mono fun xs hxs => ?_
        have hb : AnyB (applyProbe read C) (BadStep read C M U A sl D ε Ns y W) initialState
            (List.ofFn xs) := by
          by_contra hb
          have hn0 : (initialState : Level3State α).probeCount = 0 := rfl
          refine hxs (eps_ends_of_not_anyB read C M side U A sl D ε Ns y W hW _ initialState hI0.1
            hI0.2.1 (by rw [hn0]; exact hNs) ?_ hb)
          rw [hn0, Nat.sub_zero, List.length_ofFn]
          have : (Ψr (Fintype.card σ) (initialState : Level3State α) + 1) * Ns
              ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns :=
            Nat.mul_le_mul_right _ hΨ
          rw [Nat.add_mul, one_mul] at this
          omega
        refine anyB_split (applyProbe read C) W _ _ B₁ (fun s x h => ?_) _ _ hb
        rcases h with ⟨hn, h | h⟩ | h
        · exact Or.inl (Or.inl ⟨hn, Or.inl h⟩)
        · exact Or.inr h
        · exact Or.inl (Or.inr h)
    _ ≤ (Measure.pi fun _ : Fin P => D) {xs | AnyB step₂ B₁ initialState (List.ofFn xs)}
          + (Measure.pi fun _ : Fin P => D)
            {xs | AnyB (applyProbe read C) W initialState (List.ofFn xs)} :=
        measure_union_le _ _
    _ ≤ _ := by
        gcongr
        calc _ ≤ (Measure.pi fun _ : Fin P => D) {xs | SegB step₂ B₁ Same initialState
                (List.ofFn xs)}
              + ∑ i ∈ Finset.Ico 1 (1 + Ψr (Fintype.card σ) (initialState : Level3State α)),
                b i := by
              have := hany
              rw [h0] at this
              exact this
          _ ≤ b 0 + ∑ i ∈ Finset.Ico 1 (1 + Ψr (Fintype.card σ)
                (initialState : Level3State α)), b i := by
              gcongr
              exact h0 ▸ hseg _ hI0 rfl P
          _ = ∑ i ∈ Finset.range (1 + Ψr (Fintype.card σ) (initialState : Level3State α)),
                b i := by
              rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot
                (show 0 < 1 + Ψr (Fintype.card σ) (initialState : Level3State α) by omega)]
          _ ≤ ∑ i ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)), b i :=
              Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 (by omega))
          _ = _ := (ENNReal.ofReal_sum_of_nonneg hnn).symm

/-- The round's failure chance is at most `roundBoundAt` at admissible knobs, with no
length-`walkStart` prefix above `p₀`, plus the chance some step is `W`. -/
theorem eps_round_le_of {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : ProbabilityTheory.iIndepFun read μ)
    (hund : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList))
    (W : (FreeMonoid α → ARU) → Level3State α → FreeMonoid α → Prop)
    (hW : ∀ rd, KeepsInv rd C M side (W rd)) (P : PrefixDistribution α) (y p₀ η : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) C ε y) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀)
    (hy : 0 < y) (hy4 : y ≤ 1 / 4) (hη : 0 ≤ η)
    (hAe : 6 * (Fintype.card σ + 1) / y ≤ A)
    (hAm : 16 * ((P.L + 1) * (Fintype.card σ + 1)) ≤ A) (hsl : η / y ≤ sl)
    (hθg : 1 / 16 ≤ kn.θgpt) (hηm0 : 0 ≤ kn.ηm) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (read · ω) draws
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
        + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)} ∂μ := by
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
  set g : Ω → ENNReal := fun ω => (Measure.pi fun _ : Fin C.probeBudget => P.D)
    {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)}
  have hpt : ∀ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
      {xs | let r := runProbes (read · ω) C initialState (List.ofFn xs)
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
      exact eps_probe_level (read · ω) C M side U A sl P.D ε Ns y (W (read · ω)) (hW (read · ω))
        C.probeBudget N₁ Ge θgpt ηm pM
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

/-- The round's failure chance is at most `roundBoundAt` at admissible knobs, with no
length-`walkStart` prefix above `p₀`, plus the chance some step reads a string wrong. -/
theorem eps_round_le_at {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (εw : ℝ) (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α) (y p₀ η : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) C ε y) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀)
    (hy : 0 < y) (hy4 : y ≤ 1 / 4) (hη : 0 ≤ η)
    (hAe : 6 * (Fintype.card σ + 1) / y ≤ A)
    (hAm : 16 * ((P.L + 1) * (Fintype.card σ + 1)) ≤ A) (hsl : η / y ≤ sl)
    (hθg : 1 / 16 ≤ kn.θgpt) (hηm0 : 0 ≤ kn.ηm) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := sl,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (roundBoundAt (α := α) σ η p₀ C kn)
        + ENNReal.ofReal (εw * (C.probeBudget
          * ((P.L + 1) * Fintype.card σ + (Fintype.card σ + 1) * Fintype.card α
            * (Fintype.card σ + 1)))) :=
  (eps_round_le_of C M side U A sl ε μ R.read R.measurable R.indep R.undecided
    (fun rd => WrongStep M side rd C)
    (fun rd _ _ _ hs hg hx h hne => eps_step_change M side rd C hs hg hx h hne) P y p₀ η kn hA hp₀
    hpre hy hy4 hη hAe hAm hsl hθg hηm0).trans (by gcongr; exact pi_wrong_le M side C μ U εw R P)

/-- The level-3 knobs and the sampler's bounds at `η = p₀ log(16(|Q| + 1)K/δ)`: the tails risk
`δ/8` in all, the stretches `3δ/4`. -/
theorem round_knobs {α : Type*} [Fintype α] {σ : Type*} [Fintype σ] (C : Level3Settings) (L : ℕ)
    (p₀ : ℝ) (hp₀ : 0 < p₀) (hδ : 0 < C.δ) (hδ1 : C.δ < 1) (hy0 : 0 < C.harvestRate)
    (hyρ : C.harvestRate ≤ C.disagreementRate / 48) (hρ1 : C.disagreementRate < 1)
    (hP : 36 * ((Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
        + 2 * Fintype.card α + 1 : ℕ) : ℝ) / C.disagreementRate
      * Real.log (16 * ((Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
        + 2 * Fintype.card α + 1 : ℕ) : ℝ) * (((Fintype.card σ * ((Fintype.card σ + 2)
        * Fintype.card α + 1) + 2 * Fintype.card α + 1 : ℕ) : ℝ) + 1) / C.δ)
      + ((Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α + 1)
        + 2 * Fintype.card α + 1 : ℕ) : ℝ) ≤ C.probeBudget) :
    let nQ := Fintype.card σ
    let K : ℕ := (1 + (nQ + 1) ^ 3 * Fintype.card α) ^ nQ * (nQ + 3) ^ ((nQ + 2) * Fintype.card α)
    let η := p₀ * Real.log (16 * (nQ + 1) * K / C.δ)
    ∃ kn : Knobs, kn.Admissible (σ := σ) (α := α) C C.disagreementRate C.harvestRate
      ∧ 1 / 16 ≤ kn.θgpt ∧ 0 ≤ kn.ηm ∧ 0 ≤ η
      ∧ 6 * (nQ + 1) / C.harvestRate ≤ (nQ + 1) * (6 / C.harvestRate + 16 * (L + 1))
      ∧ 16 * ((L + 1) * (nQ + 1)) ≤ (nQ + 1) * (6 / C.harvestRate + 16 * (L + 1))
      ∧ roundBoundAt (α := α) σ η p₀ C kn ≤ 7 * C.δ / 8 := by
  intro nQ K η
  have hK1 : (1 : ℝ) ≤ K := by
    have : 1 ≤ K := Nat.one_le_iff_ne_zero.2 (by positivity)
    exact_mod_cast this
  have hnQ : (0 : ℝ) ≤ nQ := Nat.cast_nonneg _
  obtain ⟨kn, hA, hθg, hηm, hsum⟩ := knobs_exist (σ := σ) (α := α) C _ rfl hδ hδ1 hy0 hyρ hρ1 hP
  have hη0 : 0 ≤ η := mul_nonneg hp₀.le (Real.log_nonneg (by
    rw [le_div_iff₀ hδ]; nlinarith))
  refine ⟨kn, hA, hθg.ge, hηm.ge, hη0, ?_, ?_, ?_⟩
  · have : 0 ≤ (nQ + 1) * (16 * ((L : ℝ) + 1)) := by positivity
    have e : (nQ + 1) * (6 / C.harvestRate + 16 * ((L : ℝ) + 1))
        = 6 * (nQ + 1) / C.harvestRate + (nQ + 1) * (16 * ((L : ℝ) + 1)) := by ring
    rw [e]; linarith
  · have : 0 ≤ (nQ + 1) * (6 / C.harvestRate) := by positivity
    nlinarith
  · unfold roundBoundAt
    have h1 := tailRisk_eq nQ (Fintype.card α) hp₀ hδ 16 (by norm_num)
    have : 2 * ((nQ : ℝ) + 1) * tailRisk nQ (Fintype.card α) p₀ η = 2 * C.δ / 16 := h1
    linarith

theorem level4RoundCorrect_holds : Level4RoundCorrect := by
  intro α _ _ σ _ M side U εw Ω _ μ _ R P C p₀
  dsimp only
  intro hpre hδ hy0 hyρ hρ1 hP hw
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
  push Not at hδ1
  have hp₀ : 0 < p₀ :=
    (prefixMax_pos P.D C.walkStart).trans_le (ciSup_le hpre)
  obtain ⟨kn, hA, hθg, hηm0, hη0, hAe, hAm, hround⟩ := round_knobs (σ := σ) (α := α) C P.L p₀ hp₀
    hδ hδ1 hy0 hyρ hρ1 (by exact_mod_cast hP)
  refine (eps_round_le_at C M side U _ _ _ μ εw R P _ p₀ _ kn hA hp₀ hpre hy0 (by linarith) hη0
    hAe hAm le_rfl hθg hηm0).trans ?_
  have hwr : εw * (C.probeBudget * ((P.L + 1) * nQ + (nQ + 1) * nα * (nQ + 1))) ≤ δ / 8 := by
    convert hw using 1
    ring
  calc _ ≤ ENNReal.ofReal (7 * δ / 8) + ENNReal.ofReal (δ / 8) := by
        gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity)]
        congr 1
        ring

end Eps

end Random

end OrthoDFA
