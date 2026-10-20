import OrthoDFA.Proofs.CheckSteps

/-!
# The checked round, given the reads

Within a stretch the checked round's steps are level 3's on the draws' probes
(`segB_fst`), so a stretch risks what level 3's does. A step changing the hypothesis that avoids
the hazard keeps the invariants (`check_keeps`), so a run with no bad step and no hazard ends well
(`eps_ends_check`): where the check passes the success test fired at a hypothesis level 3's claim
admits, and the hazard rules out a check that fails without a fix. The round fails to end well
with chance at most the stretches' risk plus the chance of a hazard (`eps_check_level`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  (K : CheckSettings) {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool) (U : σ → ℝ)
  (A sl : ℝ) (D : Measure (FreeMonoid α)) (ε : ℝ) (Ns : ℕ) (y : ℝ)

omit [Fintype σ] in
/-- A stretch's bad step of the checked round is level 3's on its draws' probes. -/
theorem segB_fst (B : Level3State α → FreeMonoid α → Prop) :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α), Inv read s →
      SegB (applyCheck read C K) (fun s d => B s d.1) Same s ds →
      SegB (applyProbe read C) B Same s (ds.map Prod.fst)
  | [], _, _, h => h.elim
  | d :: ds, s, hs, h => by
    rcases h with h | ⟨s', hs', hsm, h⟩
    · exact Or.inl h
    · have ha := applyCheck_same read C K hs hs' hsm
      exact Or.inr ⟨s', ha, hsm,
        segB_fst B ds s' (step_same read C hs ha hsm.1 hsm.2).1 h⟩

/-- A run with no bad step of level 3's kinds on its probes and no hazard ends well, given enough
draws. -/
theorem eps_ends_check (W : Level3State α → FreeMonoid α → Prop) (hW : KeepsInv read C M side W) :
    ∀ (ds : List (CheckDraw α K)) (s : Level3State α), Inv read s → WitInv M side s →
      s.probeCount < Ns → Ψr (Fintype.card σ) s * Ns + (Ns - s.probeCount) ≤ ds.length →
      ¬ AnyB (applyCheck read C K)
        (fun s d => BadStep read C M U A sl D ε Ns y (fun _ _ => False) s d.1
          ∨ CheckHazard read C K M side W s d) s ds →
      EndsWell
        { target := M,
          undecided := U,
          harvestSlope := A,
          harvestSlack := sl,
          probes := D,
          disagreeRate := ε,
          disagree := searchSet read C.walkStart
            ⟨(runChecked read C K s ds).1.tree, (runChecked read C K s ds).1.edges⟩
            ∪ endsSet read C.walkStart
            ⟨(runChecked read C K s ds).1.tree, (runChecked read C K s ds).1.edges⟩,
          walkStart := C.walkStart,
          harvestLaw := harvestLaw read D C.walkStart
            ⟨(runChecked read C K s ds).1.tree, (runChecked read C K s ds).1.edges⟩,
          yield := y,
          ending := (runChecked read C K s ds).2 }
  | [], s, _, _, hn, hlen, _ => by simp at hlen; omega
  | d :: ds, s, hs, hg, hn, hlen, hb => by
    simp only [AnyB, not_or, not_exists, not_and] at hb
    obtain ⟨⟨hb1, hb3⟩, hb2⟩ := hb
    rcases hst : applyCheck read C K s d with s' | e
    · simp only [runChecked, hst]
      by_cases hsm : Same s s'
      · have ha := applyCheck_same read C K hs hst hsm
        obtain ⟨hs', hΨ, hn', -⟩ := step_same read C hs ha hsm.1 hsm.2
        have hlt : s'.probeCount < Ns := by
          have : s'.probeCount ≠ Ns := fun h => hb1 (Or.inr ⟨s', ha, hsm, h⟩)
          omega
        refine eps_ends_check W hW ds s' hs' (witInv_congr M side hg hsm.1 hsm.2) hlt ?_
          (hb2 s' hst)
        rw [hΨ, hn']
        simp only [List.length_cons] at hlen
        omega
      · obtain ⟨hs', hg', hf, -, hΨ⟩ := check_keeps read C K M side W hW hs hg hb3 hst hsm
        have h0 : s'.probeCount = 0 := by rw [hf]; rfl
        refine eps_ends_check W hW ds s' hs' hg' (by omega) ?_ (hb2 s' hst)
        rw [h0, Nat.sub_zero]
        simp only [List.length_cons] at hlen
        have : (Ψr (Fintype.card σ) s' + 1) * Ns ≤ Ψr (Fintype.card σ) s * Ns :=
          Nat.mul_le_mul_right _ hΨ
        rw [Nat.add_mul, one_mul] at this
        omega
    · simp only [runChecked, hst]
      rcases applyCheck_cases read C K s d with ⟨he, -⟩ | ⟨hf, he⟩
      · rw [he] at hst
        by_contra hne
        exact hb1 (Or.inl ⟨hn, Or.inl ⟨e, hst, hne⟩⟩)
      · rw [he] at hst
        rcases checkStep_inr read K hst with ⟨rfl, -⟩ | ⟨rfl, -⟩
        · by_contra hne
          exact hb1 (Or.inl ⟨hn, Or.inl ⟨_, hf, hne⟩⟩)
        · exact (hb3 (Or.inr ⟨hf, Or.inr hst⟩)).elim

/-- Given the reads, the checked round fails to end well with chance at most what the stretches
risk at their levels, plus the chance of a hazard. -/
theorem eps_check_level (W : Level3State α → FreeMonoid α → Prop) (hW : KeepsInv read C M side W)
    [IsProbabilityMeasure D] (Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure Dc] (P N₁ : ℕ)
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
    (Measure.pi fun _ : Fin P => D.prod (Measure.pi fun _ : Fin K.size => Dc))
        {ds | let r := runChecked read C K initialState (List.ofFn ds)
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
        + (Measure.pi fun _ : Fin P => D.prod (Measure.pi fun _ : Fin K.size => Dc))
          {ds | AnyB (applyCheck read C K) (CheckHazard read C K M side W) initialState
            (List.ofFn ds)} := by
  have hNs : 0 < Ns := by omega
  have hΨ := Ψr_start (α := α) (Fintype.card σ)
  have hm : C.middleAlternative = 2 * C.middleRate := rfl
  have hd : C.disagreementAlternative = C.disagreementRate / 2 := rfl
  have hsa : C.endsAlternative = 2 * C.endsRate := rfl
  set Dd := D.prod (Measure.pi fun _ : Fin K.size => Dc)
  set Hz := CheckHazard read C K M side W
  set step₂ := stopAt (applyCheck read C K) Hz (RoundEnd.failed : RoundEnd Level3Site)
  set B₀ := BadStep read C M U A sl D ε Ns y fun _ _ => False
  set B₁ : Level3State α → CheckDraw α K → Prop := fun s d => B₀ s d.1
  set b : ℕ → ENNReal := fun j => ENNReal.ofReal (stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt')
  set I : Level3State α → Prop := fun s => Inv read s ∧ WitInv M side s ∧ s.probeCount < Ns
  have hI0 : I initialState := ⟨inv_start read, witInv_start M side,
    by simp only [initialState, newHypothesis]; omega⟩
  have hproj : ∀ N, MeasurePreserving (fun (ds : Fin N → CheckDraw α K) i => (ds i).1)
      (Measure.pi fun _ : Fin N => Dd) (Measure.pi fun _ : Fin N => D) := fun N =>
    measurePreserving_pi _ _ fun _ => measurePreserving_fst
  have hseg : ∀ s, I s → s = newHypothesis s.tree s.edges s.hypotheses → ∀ N,
      (Measure.pi fun _ : Fin N => Dd) {ds | SegB step₂ B₁ Same s (List.ofFn ds)}
        ≤ b s.hypotheses := by
    rintro s ⟨hs, hg, -⟩ hf N
    have hsub : {ds : Fin N → CheckDraw α K | SegB step₂ B₁ Same s (List.ofFn ds)}
        ⊆ (fun (ds : Fin N → CheckDraw α K) i => (ds i).1) ⁻¹'
          {xs | SegB (applyProbe read C) B₀ Same s (List.ofFn xs)} := by
      intro ds hds
      have h1 := segB_fst read C K B₀ (List.ofFn ds) s hs (segB_stopAt _ _ _ _ _ _ _ hds)
      simp only [Set.mem_preimage, Set.mem_ofPred_eq]
      rwa [List.map_ofFn] at h1
    refine (measure_mono hsub).trans ?_
    rw [(hproj N).measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet]
    have := seg_le read C M U A sl D ε Ns y (fun _ _ => False) (fun _ => False) 0 N₁ Ge θgpt ηm pM
      θr εd' θe' θpt' hs hf (cls_mem' M side read hs hg) hNe hNm (fun _ _ _ _ _ h => h.elim)
      (fun N => by simp) hN₁ hN₁s (level_pos C hδ _) hθs0 (by rw [hsa]; linarith)
      (by rw [hsa]; linarith) hps hθpt0 (by rw [hm]; linarith) (by rw [hm]; linarith) hθg hpM0
      hyM hηm hfall
      (by rw [hd]; linarith) (by rw [hd]; linarith) hεd1 hε hθr0 hθr1 hεd'0 hεd'1 hθe'0 hθe'1
      hθpt'0 hθpt'1 hsep N
    rw [add_zero] at this
    exact this
  have hany := anyB_le step₂ B₁ Same Dd I
    (fun s => s = newHypothesis s.tree s.edges s.hypotheses) (Ψr (Fintype.card σ))
    Level3State.hypotheses b
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      have h := applyCheck_same read C K hs (stopAt_inl _ _ _ h).2 hsm
      obtain ⟨h1, h2, h3, h4⟩ := step_same read C hs h hsm.1 hsm.2
      refine ⟨⟨h1, witInv_congr M side hg hsm.1 hsm.2, ?_⟩, h2 _, h4⟩
      have : s'.probeCount ≠ Ns := fun hc => hb (Or.inr ⟨s', h, hsm, hc⟩)
      omega)
    (fun s x s' hI h hsm hb => by
      obtain ⟨hs, hg, hn⟩ := hI
      obtain ⟨hW', h⟩ := stopAt_inl _ _ _ h
      obtain ⟨h1, h2, h3, h4, h5⟩ := check_keeps read C K M side W hW hs hg hW' h hsm
      refine ⟨⟨h1, h2, ?_⟩, h3, h5, h4⟩
      rw [h3]; simp only [newHypothesis]; omega)
    hseg P initialState hI0
  have hnn : ∀ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
      0 ≤ stretchRisk C (C.level j) Ns N₁ θr εd' θe' θpt' := fun j _ =>
    stretchRisk_nonneg C (level_pos C hδ j) hεd0 hεd1 hθpt0 hθpt1 hθs0 hθs1 Ns N₁ hθr1 hεd'0
      hεd'1 hθe'0 hθe'1 (le_trans (by rw [hm]; linarith) hθpt'0) hθpt'1
  have h0 : (initialState : Level3State α).hypotheses = 0 := rfl
  calc _ ≤ (Measure.pi fun _ : Fin P => Dd)
        ({ds | AnyB step₂ B₁ initialState (List.ofFn ds)}
          ∪ {ds | AnyB (applyCheck read C K) Hz initialState (List.ofFn ds)}) := by
        refine measure_mono fun ds hds => ?_
        have hb : AnyB (applyCheck read C K) (fun s d => B₁ s d ∨ Hz s d) initialState
            (List.ofFn ds) := by
          by_contra hb
          have hn0 : (initialState : Level3State α).probeCount = 0 := rfl
          refine hds (eps_ends_check read C K M side U A sl D ε Ns y W hW _ initialState hI0.1
            hI0.2.1 (by rw [hn0]; exact hNs) ?_ hb)
          rw [hn0, Nat.sub_zero, List.length_ofFn]
          have : (Ψr (Fintype.card σ) (initialState : Level3State α) + 1) * Ns
              ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) * Ns :=
            Nat.mul_le_mul_right _ hΨ
          rw [Nat.add_mul, one_mul] at this
          omega
        exact anyB_split (applyCheck read C K) Hz _ _ B₁ (fun s x h => h) _ _ hb
    _ ≤ (Measure.pi fun _ : Fin P => Dd) {ds | AnyB step₂ B₁ initialState (List.ofFn ds)}
          + (Measure.pi fun _ : Fin P => Dd)
            {ds | AnyB (applyCheck read C K) Hz initialState (List.ofFn ds)} :=
        measure_union_le _ _
    _ ≤ _ := by
        gcongr
        calc _ ≤ (Measure.pi fun _ : Fin P => Dd) {ds | SegB step₂ B₁ Same initialState
                (List.ofFn ds)}
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

end Eps

end Random

end OrthoDFA
