import OrthoDFA.Proofs.TallyEvent
import OrthoDFA.Proofs.TallySpurNoise
import OrthoDFA.Proofs.TallyKeep
import OrthoDFA.Proofs.TallyRace
import OrthoDFA.Proofs.TallySuccess
import OrthoDFA.Proofs.Yield

/-!
# The tally round's risk

`TallyRoundGiven` with its noise event's chance bounded by `tallyE_le` and `spurious_le`, the
records field at the coefficient `κs` and the read model's own `κ`, gives `TallyRound` at every
admissible choice of knobs.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ]

theorem le_prefixMax (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (u : FreeMonoid α) : D.real {x | prefixOf x k = u} ≤ prefixMax D k :=
  le_ciSup (f := fun u => D.real {x | prefixOf x k = u}) ⟨1, by
    rintro _ ⟨v, rfl⟩; exact measureReal_le_one⟩ u

theorem prefixMax_pos (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ) :
    0 < prefixMax D k := by
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

omit [Fintype σ] in
theorem leafOf_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided)) :
    ∀ (T : DTree α) (q : σ), ({ G with κ := κ', rare := h } : ReadModel α σ).leafOf T q
      = G.leafOf T q
  | .leaf, _ => rfl
  | .node m r a, q => by
    simp only [ReadModel.leafOf, ReadModel.side, ReadModel.at', leafOf_with G κ' h r q,
      leafOf_with G κ' h a q]
    congr

omit [Fintype σ] in
theorem grown_of_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    {T : DTree α} {f : ℕ} (hg : ({ G with κ := κ', rare := h } : ReadModel α σ).Grown T f) :
    G.Grown T f := by
  induction hg with
  | start => exact .start
  | real _ ht ht₀ hs ih =>
    refine .real ih ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 8 <;> exact decide_eq_decide.2 Iff.rfl
  | fake _ hp ht ht₀ hs ih =>
    refine .fake ih hp ht ht₀ ?_
    simp only [ReadModel.GenuineSplit, leafOf_with, ReadModel.side, ReadModel.at'] at hs ⊢
    convert hs using 9 <;> exact decide_eq_decide.2 Iff.rfl

omit [Fintype σ] in
theorem untrueAt_with (G : ReadModel α σ) (κ' : ℝ)
    (h : ∀ r, min (G.dist r .accept) (G.dist r .reject) ≤ max G.ε (κ' * G.dist r .undecided))
    (rd : FreeMonoid α → ARU) (k : ℕ) (T : DTree α) (edges : Edges α) :
    ({ G with κ := κ', rare := h } : ReadModel α σ).untrueAt rd k T edges
      = G.untrueAt rd k T edges := by
  simp only [ReadModel.untrueAt, ReadModel.TrueRec, leafOf_with]

theorem tally_round_holds : TallyRound := by
  intro α _ _ σ _ Ω _ μ _ G R D _ L hlen C κs q hκ hq hpreq
  obtain ⟨read, hmeas, hind, hlaw⟩ := R
  simp only [roundRisk, ENNReal.iInf_add]
  refine le_iInf₂ fun kn hA => ?_
  have hold : ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => D)
      {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C L 0 (read · ω))
        tallyStart (List.ofFn xs)} ∂μ ≤ roundBound G D L C kn := by
    have hp₀ := prefixMax_pos D C.k
    have hpmax := le_prefixMax D C.k
    have hrare : ∀ r, min (G.dist r .accept) (G.dist r .reject)
        ≤ max G.ε (κs * G.dist r .undecided) := fun r =>
      (G.rare r).trans (max_le_max le_rfl (mul_le_mul_of_nonneg_right
        (by linarith [hA.κ_pos]) (G.nonneg r _)))
    set G₂ : ReadModel α σ := { G with κ := κs, rare := hrare }
    have hθg0 : 0 ≤ kn.θg := le_trans (by linarith [hA.θ_nonneg]) hA.θg_ge
    have hθgpt0 : 0 ≤ kn.θgpt := le_trans (by have := hA.θ_nonneg; positivity) hA.θgpt_ge
    have hρ0 : 0 ≤ kn.ρ := le_trans (by have := hA.ε_nonneg; positivity) hA.records_slack
    have hTR := tally_round_of sub_round_holds fake_race_holds harvest_good_holds
      success_sound_holds (μ := μ) G₂ read D C kn.S kn.nEnd kn.nRec kn.hP kn.hS L C.budget kn.ρ
      kn.θg kn.θgs kn.θgpt kn.θpt' kn.θr kn.εd' kn.Xe kn.θs' kn.Xs kn.η kn.ν
      hlen hρ0 hA.m_pos hA.Lmax_ge hA.n₀_le_nEnd hA.n₀_le_hS hA.θpt_nonneg hA.θpt_le
      hA.εd_nonneg hA.εd_le hA.θpt'_nonneg hA.θpt'_le hA.θr_nonneg hA.θr_le_one hA.εd'_nonneg
      hA.εd'_le hA.θr_le hA.pt_level hA.success_level hA.success_reach hA.φpt_nEnd hA.L_pos
      hA.a_pos hA.a_le hθg0 hA.θe_ge hA.φe_nonneg hA.exc_gt hA.θgs_nonneg hA.θgs_le hA.start_test
      hθgpt0 hA.θgpt_le hA.pt_test hA.φpt_nonneg hA.φpt_le hA.φpt_test (by linarith [hA.θe_ge])
      hA.Xe_nonneg hA.exc_le hA.θs'_nonneg hA.n₀_le_Xs hA.start_lin hA.η_nonneg hA.ν_nonneg
      hA.race_side hA.budget_ge
    have hE := tallyE_le (ρ := kn.ρ) (θgs := kn.θgs) G₂ read hmeas hind hlaw hA.θ_nonneg D C kn.S L
      hlen hp₀ hpmax hA.l_nonneg hA.φe_nonneg hA.φpt_nonneg hA.θe_ge hA.θg_ge hA.θgpt_ge
    have hSp := spurious_le G read hmeas hind hlaw hA.κ_pos hA.ε_nonneg D C.k L kn.S hlen hp₀ hpmax
      hκ hA.records_slack
    have hcard : ((classSet (Fintype.card σ + kn.S) : Finset (DTree α)).card : ℝ)
        ≤ classCount (α := α) σ kn.S := by
      unfold classCount
      exact_mod_cast classSet_card (Fintype.card σ + kn.S)
    refine hTR.trans ?_
    unfold roundBound testLevels
    refine add_le_add (add_le_add (add_le_add (hE.trans ?_) le_rfl) le_rfl) le_rfl
    unfold recordsTail edgeTail startTail middlesTail edgeMapCount
    refine add_le_add (add_le_add (add_le_add ((measure_mono ?_).trans (hSp.trans ?_)) ?_) ?_) ?_
    · intro ω hω
      simp only [Set.mem_setOf_eq, not_forall] at hω ⊢
      obtain ⟨T, edges, ⟨f, hf, hg⟩, he, hlt⟩ := hω
      refine ⟨T, edges, ⟨f, hf, grown_of_with G κs hrare hg⟩, he, ?_⟩
      rwa [untrueAt_with] at hlt
    all_goals
      refine ENNReal.ofReal_le_ofReal ?_
    · gcongr ?_ * _ * _
    · gcongr ?_ * _ * _
    · gcongr ?_ * _
    · simp only [mul_one]
      gcongr ?_ * _ * _
  set m₀ := 3 / 2 * G.θ * coverage D L G.M (fun r => ¬ G.Good r) / (L + 1 - startLength L : ℕ)
  set Ny := {ω | (D.prod (anchorLaw L)).real {r | r.2 ≤ r.1.toList.length ∧
      read (FreeMonoid.ofList (r.1.toList.take r.2)) ω = .undecided} < m₀ / 2}
  have hm₀ : m₀ ≤ 3 / 2 * G.θ * (D.prod (anchorLaw L)).real {r | r.2 ≤ r.1.toList.length ∧
      ¬ G.Good (G.M.eval (FreeMonoid.ofList (r.1.toList.take r.2)).toList)} := by
    have hcov : coverage D L G.M (fun r => ¬ G.Good r) / (L + 1 - startLength L : ℕ)
        ≤ (D.prod (anchorLaw L)).real {r | r.2 ≤ r.1.toList.length ∧
          ¬ G.Good (G.M.eval (FreeMonoid.ofList (r.1.toList.take r.2)).toList)} :=
      coverage_le_anchor D L G.M (fun r => ¬ G.Good r)
    simp only [m₀, mul_div_assoc]
    exact mul_le_mul_of_nonneg_left hcov (by linarith [hA.θ_nonneg])
  have hNy : μ Ny ≤ ENNReal.ofReal
      (Real.exp (-(3 / 2 * G.θ * coverage D L G.M (fun r => ¬ G.Good r)) / (16 * q))) := by
    rw [← ENNReal.ofReal_toReal (measure_ne_top μ Ny)]
    refine ENNReal.ofReal_le_ofReal ((anchor_reads read hmeas hind D L
      (fun u => ¬ G.Good (G.M.eval u.toList)) (3 / 2 * G.θ) (fun u hu => ?_) m₀ q hm₀ hq
      hpreq).trans_eq ?_)
    · rw [hlaw]
      exact (not_le.1 hu).le
    · have hn : ((L + 1 - startLength L : ℕ) : ℝ) ≠ 0 := by
        exact_mod_cast (anchorCount_pos L).ne'
      simp only [m₀]
      congr 1
      field_simp
  set B := toMeasurable μ Ny
  have hpt : ∀ ω, (Measure.pi fun _ : Fin C.budget => D)
      {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut)
        (TallyEndsWell G D C L (3 / 4 * G.θ * coverage D L G.M (fun r => ¬ G.Good r)
          / (L + 1 - startLength L : ℕ)) (read · ω)) tallyStart (List.ofFn xs)}
      ≤ (Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C L 0 (read · ω))
          tallyStart (List.ofFn xs)} + B.indicator 1 ω := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans le_add_self
    · rw [Set.indicator_of_notMem hω, add_zero]
      have hωY : ¬ ω ∈ Ny := fun h => hω (subset_toMeasurable _ _ h)
      simp only [Ny, Set.mem_ofPred_eq, not_lt] at hωY
      refine measure_mono fun xs hxs h => hxs ?_
      have h' := runEnds_inv (fun s : TState α => RootOne s.tree)
        (fun s x hs => tallyStep_rootOne C _ hs x) tallyStart _ ⟨_, _, rfl⟩ h
      refine runEnds_mono (fun e s' hs => ?_) _ _ h'
      obtain ⟨hQ, hroot⟩ := hs
      refine EndsWell.respread (tallyReplay_spread _ _ _ D L) ?_ hQ
      refine le_trans (le_of_eq ?_) (hωY.trans (measureReal_mono ?_ (measure_ne_top _ _)))
      · simp only [m₀]; ring
      · rintro ⟨w, e⟩ ⟨he, hu⟩
        exact tallyReplay_root _ hroot _ he hu
  calc _ ≤ ∫⁻ ω, ((Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C L 0 (read · ω))
          tallyStart (List.ofFn xs)} + B.indicator 1 ω) ∂μ := lintegral_mono hpt
    _ = ∫⁻ ω, (Measure.pi fun _ : Fin C.budget => D)
        {xs | ¬ RunEnds (tallyStep C fun z => (read z ω).cut) (TallyEndsWell G D C L 0 (read · ω))
          tallyStart (List.ofFn xs)} ∂μ + μ B := by
        rw [lintegral_add_right _ (measurable_one.indicator (measurableSet_toMeasurable _ _)),
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ _ := by
        rw [measure_toMeasurable]
        exact add_le_add hold hNy

end OrthoDFA
