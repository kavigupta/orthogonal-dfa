import OrthoDFA.Level3CheckSound
import OrthoDFA.Proofs.SeqTest

/-!
# A passing check certifies its rate: the proof

Given the first block's probes, the first run's hypothesis is fixed, and its check passes it while
it fails more than the rate with chance at most its level (`seqRateSound`). The round goes on only
through a fix, at the next index, so the chance from the `i`-th check on is at most the levels from
`i` on, `δ/(8(i + 1))` (`checked_le`).
-/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU)
  (D Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure D] [IsProbabilityMeasure Dc]
  (C : Level3Settings) (K : CheckSettings)

/-- The law of a block: its probes from `D`, its check's strings from `Dc`. -/
noncomputable def blockLaw : Measure (CheckBlock α C K) :=
  (Measure.pi fun _ : Fin C.probeBudget => D).prod (Measure.pi fun _ : Fin K.size => Dc)

instance : IsProbabilityMeasure (blockLaw D Dc C K) := by
  unfold blockLaw; infer_instance

/-- The blocks as the round's lists. -/
def blockList {n : ℕ} (bs : Fin n → CheckBlock α C K) :
    List (List (FreeMonoid α) × List (FreeMonoid α)) :=
  List.ofFn fun i => (List.ofFn (bs i).1, List.ofFn (bs i).2)

omit [Fintype α] [DecidableEq α] in
theorem blockList_cons {n : ℕ} (b : CheckBlock α C K) (bs : Fin n → CheckBlock α C K) :
    blockList C K (Fin.cons b bs : Fin (n + 1) → CheckBlock α C K)
      = (List.ofFn b.1, List.ofFn b.2) :: blockList C K bs := by
  simp [blockList, List.ofFn_succ]

/-- The hypothesis fails more than the check's rate from `0`. -/
def CheckBad (s : Level3State α) : Prop :=
  K.rate < Dc.real {x | startFails read s.tree s.edges x = true}

omit [DecidableEq α] [IsProbabilityMeasure D] in
/-- A check that passes a bad hypothesis is its level's risk. -/
theorem check_pass_le (hK : 0 < K.δ) (hr0 : 0 < K.rate) (hr1 : K.rate < 1) (i : ℕ)
    (s : Level3State α) (hs : CheckBad read Dc K s) :
    (Measure.pi fun _ : Fin K.size => Dc)
        {c | checkPasses read K i s.tree s.edges (List.ofFn c)}
      ≤ ENNReal.ofReal (K.level i) := by
  classical
  have h := seqRateSound Dc (fun x => startFails read s.tree s.edges x = true) K.rate (K.rate / 2)
    (K.level i) K.size hr0 hr1 (by linarith) (by linarith)
    (by unfold CheckSettings.level; positivity)
    (by
      have : 0 ≤ Dc.real {x | startFails read s.tree s.edges x = true} - K.rate := by
        unfold CheckBad at hs; linarith
      nlinarith)
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  refine ENNReal.ofReal_le_ofReal ((measureReal_mono fun c hc => ?_).trans h)
  obtain ⟨n, hn, hf⟩ := hc
  rw [List.length_ofFn] at hn
  exact ⟨n, hn, by simpa using hf⟩

omit [IsProbabilityMeasure D] in
/-- Given its probes, a run's check passes a bad hypothesis at most its level's risk. -/
theorem pass_bad_le (hK : 0 < K.δ) (hr0 : 0 < K.rate) (hr1 : K.rate < 1) (i : ℕ)
    (s : Level3State α) (p : Fin C.probeBudget → FreeMonoid α) :
    (Measure.pi fun _ : Fin K.size => Dc)
        {c | (runProbes read C s (List.ofFn p)).2 = .consistent
          ∧ checkPasses read K i (runProbes read C s (List.ofFn p)).1.tree
            (runProbes read C s (List.ofFn p)).1.edges (List.ofFn c)
          ∧ CheckBad read Dc K (runProbes read C s (List.ofFn p)).1}
      ≤ ENNReal.ofReal (K.level i) := by
  set r := runProbes read C s (List.ofFn p)
  by_cases hb : CheckBad read Dc K r.1
  · exact (measure_mono fun c hc => hc.2.1).trans (check_pass_le read Dc K hK hr0 hr1 i _ hb)
  · have : {c : Fin K.size → FreeMonoid α | r.2 = .consistent
        ∧ checkPasses read K i r.1.tree r.1.edges (List.ofFn c) ∧ CheckBad read Dc K r.1} = ∅ := by
      ext c
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      exact fun h => hb h.2.2
    rw [this, measure_empty]
    exact bot_le

theorem checked_le (hK : 0 < K.δ) (hr0 : 0 < K.rate) (hr1 : K.rate < 1) :
    ∀ (n i : ℕ) (s : Level3State α),
      (Measure.pi fun _ : Fin n => blockLaw D Dc C K)
          {bs | (checkedRun read C K (blockList C K bs) i s).2 = .consistent
            ∧ CheckBad read Dc K (checkedRun read C K (blockList C K bs) i s).1}
        ≤ ENNReal.ofReal (K.δ / (8 * (i + 1))) := by
  classical
  intro n
  induction n with
  | zero =>
    intro i s
    have : {bs : Fin 0 → CheckBlock α C K |
        (checkedRun read C K (blockList C K bs) i s).2 = .consistent
          ∧ CheckBad read Dc K (checkedRun read C K (blockList C K bs) i s).1} = ∅ := by
      ext bs
      simp [blockList, checkedRun]
    rw [this, measure_empty]
    exact bot_le
  | succ n ih =>
    intro i s
    rw [pi_cons_apply]
    set E : CheckBlock α C K → Set (Fin n → CheckBlock α C K) := fun b =>
      {bs | (Fin.cons b bs : Fin (n + 1) → CheckBlock α C K) ∈
        {bs : Fin (n + 1) → CheckBlock α C K |
          (checkedRun read C K (blockList C K bs) i s).2 = .consistent
            ∧ CheckBad read Dc K (checkedRun read C K (blockList C K bs) i s).1}}
    set P : CheckBlock α C K → Prop := fun b =>
      (runProbes read C s (List.ofFn b.1)).2 = .consistent
        ∧ checkPasses read K i (runProbes read C s (List.ofFn b.1)).1.tree
          (runProbes read C s (List.ofFn b.1)).1.edges (List.ofFn b.2)
        ∧ CheckBad read Dc K (runProbes read C s (List.ofFn b.1)).1
    have hb : ∀ b, (Measure.pi fun _ : Fin n => blockLaw D Dc C K) (E b)
        ≤ Set.indicator {b | P b} 1 b + ENNReal.ofReal (K.δ / (8 * (i + 1 + 1))) := by
      intro b
      by_cases hP : P b
      · rw [Set.indicator_of_mem (show b ∈ {b | P b} from hP)]
        exact le_add_right prob_le_one
      rw [Set.indicator_of_notMem (show b ∉ {b | P b} from hP), zero_add]
      simp only [E, Set.mem_ofPred_eq, blockList_cons]
      set r := runProbes read C s (List.ofFn b.1) with hr
      simp only [checkedRun, ← hr]
      rcases hr2 : r.2 with _ | e | _
      · simp only
        by_cases hpass : checkPasses read K i r.1.tree r.1.edges (List.ofFn b.2)
        · simp only [hpass, ↓reduceIte]
          have : {bs : Fin n → CheckBlock α C K | r.2 = .consistent ∧ CheckBad read Dc K r.1}
              = ∅ := by
            ext bs
            simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_and]
            exact fun h1 h2 => hP ⟨h1, hpass, h2⟩
          rw [this, measure_empty]
          exact bot_le
        · simp only [hpass, ↓reduceIte]
          rcases hfix : firstFix read r.1 (List.ofFn b.2) with _ | s'
          · simp
          · simp only
            have := ih (i + 1) s'
            push_cast at this
            exact this
      · simp
      · simp
    calc ∫⁻ b, (Measure.pi fun _ : Fin n => blockLaw D Dc C K) (E b) ∂blockLaw D Dc C K
        ≤ ∫⁻ b, (Set.indicator {b | P b} 1 b + ENNReal.ofReal (K.δ / (8 * (i + 1 + 1))))
            ∂blockLaw D Dc C K := lintegral_mono hb
      _ = blockLaw D Dc C K {b | P b} + ENNReal.ofReal (K.δ / (8 * (i + 1 + 1))) := by
          rw [lintegral_add_right _ measurable_const, lintegral_const, measure_univ, mul_one,
            lintegral_indicator_one (Set.to_countable _).measurableSet]
      _ ≤ ENNReal.ofReal (K.level i) + ENNReal.ofReal (K.δ / (8 * (i + 1 + 1))) := by
          gcongr
          unfold blockLaw
          rw [Measure.prod_apply (Set.to_countable _).measurableSet]
          calc ∫⁻ p, (Measure.pi fun _ : Fin K.size => Dc) (Prod.mk p ⁻¹' {b | P b})
                ∂(Measure.pi fun _ : Fin C.probeBudget => D)
              ≤ ∫⁻ _, ENNReal.ofReal (K.level i) ∂(Measure.pi fun _ : Fin C.probeBudget => D) :=
                lintegral_mono fun p => pass_bad_le read Dc C K hK hr0 hr1 i s p
            _ = _ := by rw [lintegral_const, measure_univ, mul_one]
      _ ≤ _ := by
          rw [← ENNReal.ofReal_add (by unfold CheckSettings.level; positivity) (by positivity)]
          apply ENNReal.ofReal_le_ofReal
          unfold CheckSettings.level
          rw [div_add_div _ _ (by positivity) (by positivity), div_le_div_iff₀ (by positivity)
            (by positivity)]
          ring_nf
          nlinarith [sq_nonneg (i : ℝ), hK]

end Random

open MeasureTheory Random

theorem level3CheckSound_holds : Level3CheckSound := by
  intro α _ _ read D Dc _ _ C K hK hr0 hr1
  classical
  have h := checked_le read D Dc C K hK hr0 hr1 K.runs 0 initialState
  have hsub : {blocks : Fin K.runs → CheckBlock α C K |
      (indecisiveLStarCheck C K read blocks).ending = .consistent
        ∧ K.rate < Dc.real {x | startFails read (indecisiveLStarCheck C K read blocks).hypothesis.tree
          (indecisiveLStarCheck C K read blocks).hypothesis.edges x = true}}
      = {bs | (checkedRun read C K (blockList C K bs) 0 initialState).2 = .consistent
          ∧ CheckBad read Dc K (checkedRun read C K (blockList C K bs) 0 initialState).1} := rfl
  rw [show (Measure.pi fun _ : Fin K.runs => (Measure.pi fun _ : Fin C.probeBudget => D).prod
      (Measure.pi fun _ : Fin K.size => Dc)) = Measure.pi fun _ : Fin K.runs => blockLaw D Dc C K
    from rfl, measureReal_def, hsub]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) (h.trans (le_of_eq ?_))
  congr 1
  push_cast
  ring

end OrthoDFA
