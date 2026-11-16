import RLStar.LowLevelClaims.Level3NoWrong
import RLStar.Proofs.RandomCorrect

/-!
# `Level3NoWrongCorrect`: the proof

`probe_level` at the knobs `knobs_exist` picks, with harvest slope `0` and slack `1`: a harvest
that is spread and yields `y` then ends well. Reads never wrong keep `RInv`, so the tree stays
within `|Q| + 2` leaves.
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  {σ : Type*} {M : DFA α σ} {side : σ → Bool}

theorem rinv_runProbes [Finite σ] (hW : NoWrong M side read) (xs : List (FreeMonoid α)) :
    ∀ s : Level3State α, RInv read s → RInv read (runProbes read C s xs).1 := by
  have := Fintype.ofFinite σ
  induction xs with
  | nil => intro s hs; rwa [runProbes_nil]
  | cons x xs ih =>
    intro s hs
    rcases hst : applyProbe read C s x with s' | e
    · simp only [runProbes_cons, hst]
      refine ih s' ?_
      by_cases hsm : Same s s'
      · exact ⟨(step_same read C hs.1 hst hsm.1 hsm.2).1, hsm.1 ▸ hs.2⟩
      · exact (step_change read C hW hs hst hsm).1
    · simp only [runProbes_cons, hst]
      exact hs

omit [DecidableEq α] in
theorem harvest_lt_of_not_endsWell (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (U : σ → ℝ) (ε : ℝ) {y : ℝ} (hy : 0 < y) (T : DTree α) (E : Edges α) (site : Level3Site)
    (h : ¬ EndsWell (outcomeAt read C M U 0 1 D ε y T E (.harvest site))) :
    (harvestLaw read D C.walkStart ⟨T, E⟩ site).real Set.univ < y := by
  by_contra hge
  push Not at hge
  have hfin : harvestLaw read D C.walkStart ⟨T, E⟩ site Set.univ ≠ ⊤ := fun htop => by
    rw [measureReal_def, htop, ENNReal.toReal_top] at hge
    linarith
  refine h ⟨fun t _ => ?_, harvestLaw_spread read D C.walkStart ⟨T, E⟩ site, hge⟩
  change _ ≤ (0 * t + 1) * _
  rw [zero_mul, zero_add, one_mul]
  exact measureReal_mono (Set.subset_univ _) hfin

end Random

open MeasureTheory Random

theorem level3NoWrongCorrect_holds : Level3NoWrongCorrect := by
  intro α _ _ σ _ M side read C D _
  dsimp only
  intro hW hδ hδ1 hε0 hε1 hP
  have hy : 0 < C.harvestRate := by unfold Level3Settings.harvestRate; positivity
  obtain ⟨kn, hA, -, -, hsum⟩ := knobs_exist (σ := σ) (α := α) C _ rfl hδ hδ1 hy le_rfl hε1
    (by exact_mod_cast hP)
  have hN : ∀ site T E, ¬ EndsWell (outcomeAt read C M (fun _ => 0) 0 1 D C.disagreementRate
      C.harvestRate T E (.harvest site)) →
      (harvestLaw read D C.walkStart ⟨T, E⟩ site).real Set.univ < C.harvestRate :=
    fun site T E => harvest_lt_of_not_endsWell read C D _ _ hy T E site
  have hle := probe_level read C M (fun _ => 0) 0 1 D C.disagreementRate kn.Ns C.harvestRate hW
    C.probeBudget kn.N₁ kn.Ge kn.θgpt kn.ηm kn.pM kn.θr kn.εd' kn.θe' kn.θpt'
    (fun T _ E _ h => (hN .ends T E h).le.trans (le_max_right _ _))
    (fun T _ E _ h => (hN .middles T E h).le.trans (le_max_right _ _))
    hA.N₁_pos hA.N₁_le hδ hA.endsRate_pos hA.endsRate_lt hA.harvest_le_ends hA.middleRate_pos
    hA.middleRate_lt hA.θgpt_le hA.pM_nonneg hA.y_le_pM hA.ηm_le hA.middles_fall
    hA.disagreementRate_pos hA.disagreementRate_lt hA.disagreementRate_le_ε hA.θr_nonneg hA.θr_le
    hA.εd'_nonneg hA.εd'_le hA.θe'_nonneg hA.θe'_le hA.θpt'_ge hA.θpt'_le hA.sep
    hA.probeBudget_ge
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal hδ.le
    (le_trans (measure_mono ?_) (hle.trans (ENNReal.ofReal_le_ofReal (hsum.trans (by linarith)))))
  intro draws hd
  simp only [Set.mem_ofPred_eq, indecisiveLStarLevel3_eq] at hd ⊢
  intro hew
  rcases hd with hl | hd
  · exact absurd hl (not_lt.2 (leaves_le_of_reached' read hW
      (rinv_runProbes read C hW _ _ ⟨inv_start read, reached_start read⟩).2))
  · unfold EndsWell at hew
    rcases he : (runProbes read C initialState (List.ofFn draws)).2 with _ | s | _ <;>
      simp only [he] at hd hew
    · exact absurd hew (not_le.2 hd)
    · exact absurd hew.2.2 (not_le.2 hd)

end OrthoDFA
