import OrthoDFA.Proofs.LoopUniRound
import OrthoDFA.Proofs.Level3Sound

/-!
# A round's consistent ending, given everything before its probes

Round `r`'s level 3 reads a fixed function of the noise and of what reached it (`stage4`), and its
probes are fresh uniforms independent of both. So at each noise, given the earlier rounds and the
round's clustering, level 3 ends consistent at a hypothesis disagreeing on more than its rate, over
its probes' law, with chance at most `δ/8` (`Level3ConsistentSound`), and so over everything
(`loop_consistent_round_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {K : ℕ}

/-- Level 3, reading `F` cut at `kl` and `kh` at the noise `ω` over the probes `xs`, ends consistent
at a hypothesis whose `D'`-probes search or stop undecided at their ends more than its rate. -/
def FalseConsistent (O : Oracle μ (FreeMonoid α)) (C : Random.Level3Settings)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (D' : Measure (FreeMonoid α)) (ω : Ω)
    (xs : Fin C.probeBudget → FreeMonoid α) : Prop :=
  (Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs).ending = .consistent
    ∧ C.disagreementRate < D'.real
      (Random.searchSet (fun z => familyRead O.mq F kl kh z ω) C.walkStart
          (Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs).hypothesis
        ∪ Random.endsSet (fun z => familyRead O.mq F kl kh z ω) C.walkStart
          (Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs).hypothesis)

theorem measurableSet_falseConsistent (O : Oracle μ (FreeMonoid α)) (C : Random.Level3Settings)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) (D' : Measure (FreeMonoid α)) [IsFiniteMeasure D'] :
    MeasurableSet {q : Ω × (Fin C.probeBudget → FreeMonoid α) |
      FalseConsistent O C F kl kh D' q.1 q.2} := by
  classical
  refine measurableSet_prod_countable_right _ fun xs => ?_
  set A := readStrings (shortStrings (probesLen xs + 1 + C.probeBudget + 2)) F
  set g : Ω → Finset (FreeMonoid α) := fun ω => A.filter fun w => O.mq w ω = 1
  have hg : ∀ U, MeasurableSet {ω | g ω = U} := fun U =>
    measurableSet_of_reads O A _ fun ω ω' h hω => by
      simp only [Set.mem_ofPred_eq, g] at hω ⊢
      rw [← hω]
      exact Finset.filter_congr fun w hw => (h w hw).symm
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ
  set res : Ω → Random.Level3Result α := fun ω =>
    Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs
  set Φ : Finset (FreeMonoid α) → Random.Level3Result α := fun U =>
    if h : ∃ ω, g ω = U then res h.choose else res ω₀
  have hΦ : ∀ ω, res ω = Φ (g ω) := fun ω => by
    have hex : ∃ ω', g ω' = g ω := ⟨ω, rfl⟩
    simp only [Φ, dif_pos hex]
    refine level3_result_congr O C F kl kh xs fun w hw => ?_
    have := Finset.ext_iff.1 hex.choose_spec w
    simp only [g, Finset.mem_filter] at this
    exact ⟨fun h => (this.2 ⟨hw, h⟩).2, fun h => (this.1 ⟨hw, h⟩).2⟩
  have hR : ∀ R : Random.Level3Result α, MeasurableSet {ω | R.ending = .consistent
      ∧ C.disagreementRate < D'.real
        (Random.searchSet (fun z => familyRead O.mq F kl kh z ω) C.walkStart R.hypothesis
          ∪ Random.endsSet (fun z => familyRead O.mq F kl kh z ω) C.walkStart R.hypothesis)} := by
    intro R
    by_cases he : R.ending = .consistent
    · have := measurableSet_not_endsWell_searchEnds (σ := Unit) O F kl kh C.walkStart
        (_root_.DFA.mk (fun _ _ => ()) () ∅) (fun _ => 0) (fun _ => le_rfl) (le_refl (0 : ℝ))
        (le_refl (0 : ℝ)) D' C.disagreementRate 0
        R.hypothesis .consistent
      simp only [EndsWell, not_le] at this
      simp only [he, true_and]
      exact this
    · simp only [he, false_and, Set.ofPred_false]
      exact MeasurableSet.empty
  have := measurableSet_of_data g hg (fun U => {ω | (Φ U).ending = .consistent
      ∧ C.disagreementRate < D'.real
        (Random.searchSet (fun z => familyRead O.mq F kl kh z ω) C.walkStart (Φ U).hypothesis
          ∪ Random.endsSet (fun z => familyRead O.mq F kl kh z ω) C.walkStart
            (Φ U).hypothesis)}) fun U => hR _
  convert this using 1
  ext ω
  simp only [Set.mem_ofPred_eq, FalseConsistent]
  rw [show Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs
    = Φ (g ω) from hΦ ω]

section Round

variable (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
  (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]

/-- Round `r`, where every earlier clustering and its own returned, ends consistent at a
hypothesis disagreeing, over its probes' law, on more than its rate with chance at most `δ/8`. -/
theorem loop_consistent_round_le (N : ℕ) (hN : P.L + settings.level3.probeBudget + 2 ≤ N)
    (hδ : 0 < settings.level3.δ) (hε0 : 0 < settings.level3.disagreementRate)
    (hε1 : settings.level3.disagreementRate < 1) (r : ℕ) (hr : r < K) :
    (loopMeasure μ α K settings.level3.probeBudget).real
      {p | ∃ t, stage4 settings O P.D Dsf N r hr p.1 p.2 = some t
        ∧ FalseConsistent O settings.level3 t.2.2.1 t.2.1.lo (t.2.1.hi + 1)
          (avoidLaw P.D settings.avoidDepth t.2.2.2) p.1
          (probesOf settings P.D t.2.2.2 (p.2 ⟨r, hr⟩).2)}
      ≤ settings.level3.δ / 8 := by
  classical
  set T := settings.level3.probeBudget
  set C := settings.level3
  set rK : Fin K := ⟨r, hr⟩
  set Y := List (HarvestCode α) × State × Finset (FreeMonoid α) × Finset (FreeMonoid α)
  set Bd : Y → Set (Ω × (Fin T → unitInterval)) := fun t =>
    (fun q : Ω × (Fin T → unitInterval) => (q.1, probesOf settings P.D t.2.2.2 q.2)) ⁻¹'
      {q | FalseConsistent O C t.2.2.1 t.2.1.lo (t.2.1.hi + 1)
        (avoidLaw P.D settings.avoidDepth t.2.2.2) q.1 q.2}
  have hBd : ∀ t, MeasurableSet (Bd t) := fun t =>
    (measurable_fst.prodMk ((measurable_probesOf settings P.D t.2.2.2).comp measurable_snd))
      (measurableSet_falseConsistent O C _ _ _ _)
  set S := {p : Ω × (Fin K → RoundDraws α K T) | ∃ t, stage4 settings O P.D Dsf N r hr p.1 p.2
    = some t ∧ FalseConsistent O C t.2.2.1 t.2.1.lo (t.2.1.hi + 1)
      (avoidLaw P.D settings.avoidDepth t.2.2.2) p.1 (probesOf settings P.D t.2.2.2 (p.2 rK).2)}
  have hS : MeasurableSet S := by
    have : S = ⋃ t : Y, {p | stage4 settings O P.D Dsf N r hr p.1 p.2 = some t}
        ∩ (fun p : Ω × (Fin K → RoundDraws α K T) => (p.1, (p.2 rK).2)) ⁻¹' Bd t := by
      ext p
      simp only [S, Bd, Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Set.mem_preimage]
      rfl
    rw [this]
    exact MeasurableSet.iUnion fun t =>
      (measurableSet_stage4 settings O P.D Dsf N P.length_le hN r hr (some t)).inter
        ((measurable_fst.prodMk (measurable_snd.comp ((measurable_pi_apply rK).comp
          measurable_snd))) (hBd t))
  have hsec : ∀ ω, (Measure.pi fun _ : Fin K => roundLaw α K T) (Prod.mk ω ⁻¹' S)
      ≤ ENNReal.ofReal (C.δ / 8) := by
    intro ω
    have hσ : MeasurePreserving (id : RoundDraws α K T → RoundDraws α K T) (roundLaw α K T)
        ((restLaw α K).prod (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))) :=
      MeasurePreserving.id _
    set Bad : Option Y → Set (Fin T → unitInterval) := fun y => match y with
      | some t => {u | (ω, u) ∈ Bd t}
      | none => ∅
    have hBad : ∀ y, (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) (Bad y)
        ≤ ENNReal.ofReal (C.δ / 8) := by
      intro y
      rcases y with _ | t
      · simp [Bad]
      · set D' := avoidLaw P.D settings.avoidDepth t.2.2.2
        have hpr : MeasurePreserving (probesOf settings P.D t.2.2.2)
            (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))
            (Measure.pi fun _ => D') :=
          measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
        have hX : MeasurableSet {xs : Fin T → FreeMonoid α |
            FalseConsistent O C t.2.2.1 t.2.1.lo (t.2.1.hi + 1) D' ω xs} :=
          (Set.to_countable _).measurableSet
        have hsound := level3ConsistentSound_holds
          (fun z => familyRead O.mq t.2.2.1 t.2.1.lo (t.2.1.hi + 1) z ω) D' C hδ hε0 hε1
        show (Measure.pi fun _ : Fin T => (volume : Measure unitInterval))
          (probesOf settings P.D t.2.2.2 ⁻¹' {xs | FalseConsistent O C t.2.2.1 t.2.1.lo
            (t.2.1.hi + 1) D' ω xs}) ≤ _
        rw [hpr.measure_preimage hX.nullMeasurableSet, ← ofReal_measureReal]
        exact ENNReal.ofReal_le_ofReal hsound
    have h := round_stage_le (roundLaw α K T) (restLaw α K)
      (Measure.pi fun _ : Fin T => (volume : Measure unitInterval)) id hσ id measurable_id
      (fun w => rfl) rK (stage4 settings O P.D Dsf N r hr ω)
      (fun x x' hxx' h1 => stage4_congr_draws settings O P.D Dsf N r hr ω x x'
        (fun i hi => hxx' i hi) h1)
      (fun y => measurable_prodMk_left
        (measurableSet_stage4 settings O P.D Dsf N P.length_le hN r hr y))
      Bad (ENNReal.ofReal (C.δ / 8)) hBad
    refine (measure_mono fun g hg => ?_).trans h
    obtain ⟨t, ht, hf⟩ := hg
    show (g rK).2 ∈ Bad (stage4 settings O P.D Dsf N r hr ω g)
    rw [ht]
    exact hf
  have hle : (loopMeasure μ α K T) S ≤ ENNReal.ofReal (C.δ / 8) := by
    unfold loopMeasure
    rw [Measure.prod_apply hS]
    calc ∫⁻ ω, (Measure.pi fun _ : Fin K => roundLaw α K T) (Prod.mk ω ⁻¹' S) ∂μ
        ≤ ∫⁻ _ω, ENNReal.ofReal (C.δ / 8) ∂μ := lintegral_mono hsec
      _ = ENNReal.ofReal (C.δ / 8) := by rw [lintegral_const, measure_univ, mul_one]
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) hle

end Round

end OrthoDFA

end
