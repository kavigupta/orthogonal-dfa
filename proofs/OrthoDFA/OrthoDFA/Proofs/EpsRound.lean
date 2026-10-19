import OrthoDFA.Proofs.EpsReads

/-!
# Level 3 with rare wrong reads: the proof

Given the reads, the round fails to end well only where a stretch risks it, as at level 3, or
where some step reads a string wrong (`eps_probe_level`): a step that reads nothing wrong keeps
`WitInv`, so the tree stays in the class level 3's stretches need. The run stopped at the first
step that reads a string wrong (`stopAt`) has level 3's stretches. Over the reads, the probes
harvest at good read-states within their bounds but for `tailRisk` at the ends and at the middles,
and some step reads a string wrong with chance at most
`εw · probeBudget · ((L + 1)|Q| + (|Q| + 1)²|Σ|)` (`pi_wrong_le`).

`EpsRoundCorrect` takes level 3's knobs at `log(16K/δ)`: each `tailRisk` risks `δ/16`, the
stretches `3δ/4` and the wrong reads `δ/8`.
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

theorem stopAt_inr {s : S} {x : X} (h : B₂ s x) : stopAt step B₂ e₀ s x = .inr e₀ := by
  classical
  unfold stopAt
  rw [if_pos h]

theorem segB_mono (B B' : S → X → Prop) (same : S → S → Prop) (h : ∀ s x, B s x → B' s x) :
    ∀ s xs, SegB step B same s xs → SegB step B' same s xs
  | _, [], hs => hs.elim
  | s, x :: xs, hs => hs.elim (fun hb => Or.inl (h s x hb)) fun ⟨s', hs', hsm, hr⟩ =>
      Or.inr ⟨s', hs', hsm, segB_mono B B' same h s' xs hr⟩

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
  {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ) (θ : ℝ)
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
      ¬ AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y W) s xs →
      EndsWell
        { target := M,
          badStates := BadAt U θ,
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
their levels, plus the chance that some step is `W`; the good states' bounds are needed only at
the trees off `Sp`, every step at a tree of `Sp` being `W`. -/
theorem eps_probe_level (W : Level3State α → FreeMonoid α → Prop) (hW : KeepsInv read C M side W)
    (Sp : DTree α → Prop) (hSp : ∀ s x, Sp s.tree → W s x)
    [IsProbabilityMeasure D] (P N₁ : ℕ)
    (Ge θgpt ηm pM θr εd' θe' θpt' : ℝ)
    (hNe : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ¬ Sp T → ∀ E, EdgesInto T E →
      D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .ends) ≤ Ge)
    (hNm : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ¬ Sp T → ∀ E, EdgesInto T E →
      D.real (goodAt read C.walkStart ⟨T, E⟩ M U θ .middles)
        ≤ θgpt * D.real {x | searches (probeOrAgree read T E C.walkStart x) = true} + ηm)
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
              badStates := BadAt U θ,
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
  set B₀ := BadStep read C M U θ D ε Ns y fun _ _ => False
  set B₁ : Level3State α → FreeMonoid α → Prop := fun s x => B₀ s x ∧ ¬ W s x
  set b : ℕ → ENNReal := fun j => ENNReal.ofReal (stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set I : Level3State α → Prop := fun s => Inv read s ∧ WitInv M side s ∧ s.probeCount < Ns
  have hI0 : I initialState := ⟨inv_start read, witInv_start M side,
    by simp only [initialState, newHypothesis]; omega⟩
  have hseg : ∀ s, I s → s = newHypothesis s.tree s.edges s.hypotheses → ∀ N,
      (Measure.pi fun _ : Fin N => D) {xs | SegB step₂ B₁ Same s (List.ofFn xs)}
        ≤ b s.hypotheses := by
    rintro s ⟨hs, hg, -⟩ hf N
    by_cases hsp : Sp s.tree
    · refine le_of_eq_of_le (measure_mono_null (fun xs hxs => ?_) measure_empty) bot_le
      change SegB step₂ B₁ Same s (List.ofFn xs) at hxs
      rcases hl : List.ofFn xs with _ | ⟨x, l⟩
      · rw [hl] at hxs; exact hxs.elim
      · rw [hl] at hxs
        rcases hxs with h | ⟨s', h, -⟩
        · exact h.2 (hSp s x hsp)
        · have h' : step₂ s x = .inr .failed := stopAt_inr _ _ _ (hSp s x hsp)
          rw [h'] at h; cases h
    refine (measure_mono fun xs hxs => segB_stopAt _ _ _ _ _ _ _
      (segB_mono _ B₁ B₀ Same (fun _ _ h => h.1) _ _ hxs)).trans ?_
    have hT := cls_mem' M side read hs hg
    have := seg_le read C M U θ D ε Ns y (fun _ _ => False) (fun _ => False) 0 N₁ Ge θgpt ηm pM
      θr εd' θe' θpt' hs hf (hNe _ hT hsp) (hNm _ hT hsp) (fun _ _ _ _ _ h => h.elim)
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
      obtain ⟨hW', h⟩ := stopAt_inl _ _ _ h
      obtain ⟨h1, h2, h3, h4⟩ := step_same read C hs h hsm.1 hsm.2
      refine ⟨⟨h1, witInv_congr M side hg hsm.1 hsm.2, ?_⟩, h2 _, h4⟩
      have : s'.probeCount ≠ Ns := fun hc => hb ⟨Or.inr ⟨s', h, hsm, hc⟩, hW'⟩
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
        have hb : AnyB (applyProbe read C) (BadStep read C M U θ D ε Ns y W) initialState
            (List.ofFn xs) := by
          by_contra hb
          have hn0 : (initialState : Level3State α).probeCount = 0 := rfl
          refine hxs (eps_ends_of_not_anyB read C M side U θ D ε Ns y W hW _ initialState hI0.1
            hI0.2.1 (by rw [hn0]; exact hNs) ?_ hb)
          rw [hn0, Nat.sub_zero, List.length_ofFn]
          have : (Ψr (Fintype.card σ) (initialState : Level3State α) + 1) * Ns
              ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns :=
            Nat.mul_le_mul_right _ hΨ
          rw [Nat.add_mul, one_mul] at this
          omega
        refine anyB_split (applyProbe read C) W _ _ B₁ (fun s x h => ?_) _ _ hb
        by_cases hw : W s x
        · exact Or.inr hw
        rcases h with ⟨hn, h | h⟩ | h
        · exact Or.inl ⟨Or.inl ⟨hn, Or.inl h⟩, hw⟩
        · exact absurd h hw
        · exact Or.inl ⟨Or.inr h, hw⟩
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

/-- For some tree of `classSet |Q|` off `Sp` and edge map into its leaves, the probes harvesting
at a good read-state exceed `Ge` at the ends, or `θgpt` of the searches plus `ηm` at the
middles. -/
def NoiseBad (k : ℕ) (Sp : DTree α → Prop) (Ge θgpt ηm : ℝ) : Prop :=
  (¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ¬ Sp T → ∀ E, EdgesInto T E →
      D.real (goodAt read k ⟨T, E⟩ M U θ .ends) ≤ Ge)
  ∨ ¬ ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)), ¬ Sp T → ∀ E, EdgesInto T E →
      D.real (goodAt read k ⟨T, E⟩ M U θ .middles)
        ≤ θgpt * D.real {x | searches (probeOrAgree read T E k x) = true} + ηm

/-- The round's failure chance is at most what the stretches risk at admissible knobs, plus the
chance of `NoiseBad` off `Sp`, plus the chance some step is `W`, every step at a tree of `Sp`
being `W`. -/
theorem eps_round_le_noise {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [IsProbabilityMeasure μ] (read : FreeMonoid α → Ω → ARU)
    (W : (FreeMonoid α → ARU) → Level3State α → FreeMonoid α → Prop)
    (hW : ∀ rd, KeepsInv rd C M side (W rd)) (Sp : (FreeMonoid α → ARU) → DTree α → Prop)
    (hSp : ∀ rd s x, Sp rd s.tree → W rd s x) (P : PrefixDistribution α) (y : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) θ P.L C ε y) :
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
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
          stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt')
        + μ {ω | NoiseBad (read · ω) M U θ P.D C.walkStart (Sp (read · ω)) kn.Ge kn.θgpt kn.ηm}
        + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)} ∂μ := by
  obtain ⟨Ns, N₁, Ge, θgpt, ηm, pM, θr, εd', θe', θpt'⟩ := kn
  obtain ⟨hN₁, hN₁s, hθ, hθs, hGe0, hδ, hθs0, hθs1, hθpt0, hθpt1, hεd0, hεd1,
    hε, hθr0, hθr1, hεd'0, hεd'1, hθe'0, hθe'1, hθpt'0, hθpt'1, hsep, hP, hps, hθg, hθg4,
    hpM0, hyM, hηm, hfall⟩ := hA
  simp only at hN₁ hN₁s hGe0 hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP hps hfall ⊢
  simp only at hθg hθg4 hpM0 hyM hηm
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set Nn := {ω | NoiseBad (read · ω) M U θ P.D C.walkStart (Sp (read · ω)) Ge θgpt ηm}
  set c := ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
    stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set B := toMeasurable μ Nn
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
            ending := r.2 }}
        ≤ (c + B.indicator 1 ω) + g ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans (le_add_self.trans le_self_add)
    · have hω' : ω ∉ Nn := fun h => hω (subset_toMeasurable _ _ h)
      rw [Set.indicator_of_notMem hω, add_zero]
      simp only [Nn, NoiseBad, Set.mem_ofPred_eq, not_or, not_not] at hω'
      exact eps_probe_level (read · ω) C M side U θ P.D ε Ns y (W (read · ω)) (hW (read · ω))
        (Sp (read · ω)) (hSp (read · ω)) C.probeBudget N₁ Ge θgpt ηm pM θr εd' θe' θpt' hω'.1 hω'.2
        hN₁ hN₁s hδ hθs0 hθs1 hps hθpt0 hθpt1 hθg4 hpM0 hyM hηm hfall hεd0 hεd1 hε
        hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1 hθpt'0 hθpt'1 hsep hP
  calc _ ≤ ∫⁻ ω, ((c + B.indicator 1 ω) + g ω) ∂μ := lintegral_mono hpt
    _ = c + μ B + ∫⁻ ω, g ω ∂μ := by
        rw [lintegral_add_left ((measurable_one.indicator
          (measurableSet_toMeasurable μ Nn)).const_add c),
          lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ = _ := by rw [measure_toMeasurable]

/-- The round's failure chance is at most `roundBoundAt` at admissible knobs, with no
length-`walkStart` prefix above `p₀`, plus the chance some step is `W`. -/
theorem eps_round_le_of {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (read : FreeMonoid α → Ω → ARU) (hmeas : ∀ z, Measurable (read z))
    (hind : ProbabilityTheory.iIndepFun read μ)
    (hund : ∀ z, μ.real {ω | read z ω = .undecided} = U (M.eval z.toList))
    (W : (FreeMonoid α → ARU) → Level3State α → FreeMonoid α → Prop)
    (hW : ∀ rd, KeepsInv rd C M side (W rd)) (P : PrefixDistribution α) (y p₀ : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) θ P.L C ε y) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) :
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
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (roundBoundAt (α := α) σ θ p₀ P.L C kn)
        + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
          {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)} ∂μ := by
  refine (eps_round_le_noise C M side U θ ε μ read W hW (fun _ _ => False)
    (fun _ _ _ h => h.elim) P y kn hA).trans ?_
  have hθs := hA.θ_small
  have hθg := hA.θgpt_ge
  have hL0 : (0 : ℝ) ≤ P.L := Nat.cast_nonneg _
  have hn0 : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
  have hNe := endsNoise_le M U θ hmeas hind hund hA.θ_nonneg P.D C.walkStart
    (n := Fintype.card σ) (η := kn.Ge - 9 * θ * (Fintype.card σ + 1)) P.length_le hp₀ hpre
    (by nlinarith [hA.θ_nonneg])
  have hNm := midNoise_le M U θ hmeas hind hund hA.θ_nonneg P.D C.walkStart
    (n := Fintype.card σ) (η := kn.ηm) P.length_le hp₀ hpre (by nlinarith [hA.θ_nonneg]) hθg
  have hnn : 0 ≤ ∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
      stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt' :=
    Finset.sum_nonneg fun j _ => stretchRisk_nonneg C (level_pos C hA.δ_pos j)
      hA.disagreementRate_pos hA.disagreementRate_lt hA.middleRate_pos hA.middleRate_lt
      hA.endsRate_pos hA.endsRate_lt kn.Ns kn.N₁ hA.θr_le hA.εd'_nonneg hA.εd'_le hA.θe'_nonneg
      hA.θe'_le (le_trans (by show 0 ≤ 2 * C.middleRate; linarith [hA.middleRate_pos])
        hA.θpt'_ge) hA.θpt'_le
  have hN : μ {ω | NoiseBad (read · ω) M U θ P.D C.walkStart (fun _ => False) kn.Ge kn.θgpt kn.ηm}
      ≤ ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀
          (kn.Ge - 9 * θ * (Fintype.card σ + 1)))
        + ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀ kn.ηm) := by
    refine (measure_mono fun ω hω => ?_).trans ((measure_union_le _ _).trans (add_le_add hNe hNm))
    simp only [NoiseBad, Set.mem_ofPred_eq, not_false_eq_true, forall_const] at hω
    rcases hω with h | h
    · exact Or.inl (by simpa only [Set.mem_ofPred_eq, add_sub_cancel] using h)
    · exact Or.inr h
  calc _ ≤ ENNReal.ofReal (∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ)
            (Fintype.card α)), stretchRisk C (C.level j) kn.Ns kn.N₁ kn.θr kn.εd' kn.θe' kn.θpt')
          + (ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀
              (kn.Ge - 9 * θ * (Fintype.card σ + 1)))
            + ENNReal.ofReal (tailRisk (Fintype.card σ) (Fintype.card α) p₀ kn.ηm))
          + ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
            {d | AnyB (applyProbe (read · ω) C) (W (read · ω)) initialState (List.ofFn d)} ∂μ := by
        gcongr
    _ = _ := by
        rw [add_comm (ENNReal.ofReal _) (_ + _), ← ENNReal.ofReal_add
          (by unfold tailRisk; positivity) (by unfold tailRisk; positivity),
          ← ENNReal.ofReal_add (by unfold tailRisk; positivity) hnn]
        rfl

/-- The round's failure chance is at most `roundBoundAt` at admissible knobs, with no
length-`walkStart` prefix above `p₀`, plus the chance some step reads a string wrong. -/
theorem eps_round_le_at {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (εw : ℝ) (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α) (y p₀ : ℝ) (kn : Knobs)
    (hA : kn.Admissible (σ := σ) (α := α) θ P.L C ε y) (hp₀ : 0 < p₀)
    (hpre : ∀ u : FreeMonoid α, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel3 C (R.read · ω) draws
          ¬ EndsWell
            { target := M,
              badStates := BadAt U θ,
              probes := P.D,
              disagreeRate := ε,
              disagree := searchSet (R.read · ω) C.walkStart result.hypothesis
                ∪ endsSet (R.read · ω) C.walkStart result.hypothesis,
              walkStart := C.walkStart,
              harvestLaw := harvestLaw (R.read · ω) P.D C.walkStart result.hypothesis,
              yield := y,
              ending := result.ending }} ∂μ
      ≤ ENNReal.ofReal (roundBoundAt (α := α) σ θ p₀ P.L C kn)
        + ENNReal.ofReal (εw * (C.probeBudget
          * ((P.L + 1) * Fintype.card σ + (Fintype.card σ + 1) * Fintype.card α
            * (Fintype.card σ + 1)))) :=
  (eps_round_le_of C M side U θ ε μ R.read R.measurable R.indep R.undecided
    (fun rd => WrongStep M side rd C)
    (fun rd _ _ _ hs hg hx h hne => eps_step_change M side rd C hs hg hx h hne) P y p₀ kn hA hp₀
    hpre).trans (by gcongr; exact pi_wrong_le M side C μ U εw R P)

theorem epsRoundCorrect_holds : EpsRoundCorrect := by
  intro α _ _ σ _ M side U θ εw Ω _ μ _ R P C p₀
  dsimp only
  intro hpre hθ hδ hθL hGy hyρ hρ1 hP hw
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
  refine (eps_round_le_at C M side U θ _ μ εw R P _ p₀ kn hA hp₀ hpre).trans ?_
  have h1 : tailRisk nQ nα p₀ (kn.Ge - 9 * θ * (nQ + 1)) = δ / 2 / 8 := by
    rw [hGe, show 9 * θ * (nQ + 1) + p₀ * Λ - 9 * θ * (nQ + 1) = p₀ * Λ by ring, hΛ']
    exact tailRisk_eq nQ nα hp₀ (by positivity)
  have h2 : tailRisk nQ nα p₀ kn.ηm = δ / 2 / 8 := by
    rw [hηm, hΛ']
    exact tailRisk_eq nQ nα hp₀ (by positivity)
  have hround : roundBoundAt (α := α) σ θ p₀ P.L C kn ≤ 7 * δ / 8 := by
    unfold roundBoundAt
    linarith
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
