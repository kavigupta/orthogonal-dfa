import OrthoDFA.Proofs.CheckReads

/-!
# A check that fails without a fix

A failing check string whose probe from `0` gives no fix stopped at an undecided read of one of
its prefixes followed by a midfix (`fail_nofix`). So a check failing without a fix leaves its
test silent on the strings that read undecided (`checkStep_failed_silent`), and where fewer than
`q` of the check's strings do, that has chance at most `b(q)^n / a`, `n` the check's size and `a`
its level (`waste_le`), with

    b(q) = 2q + (1 − q)(1 − ρ)/(1 − ρ/2),   ρ the check's rate.

Within a stretch the check runs at most once, so a stretch risks at most that (`seg_waste_le`),
and the round at most the sum over the stretches (`anyB_waste_le`).
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)
  (K : CheckSettings)

open scoped Classical in
/-- Some prefix of `x` followed by a midfix of `T` reads undecided. -/
noncomputable def Utype (T : DTree α) (x : FreeMonoid α) : Bool :=
  decide (∃ z ∈ pot 0 x T, read z = .undecided)

/-- A failing check string giving no fix read undecided. -/
theorem fail_nofix {s : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (hf : startFails read s.tree s.edges x = true) (hn : fixOf read s x = none) :
    Utype read s.tree x = true := by
  classical
  have hp : probeOrAgree read s.tree s.edges 0 x = probe read s.tree s.edges 0 x := by
    simp [probeOrAgree]
  have hOK := probeOrAgree_ok read s.tree s.edges 0 hs.edges x
  have hE := probeOrAgree_ends read s.tree x s.edges 0
  have hM := probeOrAgree_pt read s.tree x s.edges 0
  rw [hp] at hOK hE hM
  unfold startFails at hf
  unfold fixOf at hn
  simp only [Utype, decide_eq_true_eq]
  generalize ho : probe read s.tree s.edges 0 x = o at hf hn hOK hE hM
  cases o with
  | agree => simp at hf
  | wrongEdge p c u t =>
    obtain ⟨-, -, -, t₀, w₀, hE', -⟩ := hOK
    simp only [hE'] at hn
    cases hn
  | unlearnedEdge p c u t => cases hn
  | undecidedStart z =>
    obtain ⟨hz, hu⟩ := hE z (by simp [endsString])
    exact ⟨z, hz, hu⟩
  | undecided zs cl =>
    rcases zs with _ | ⟨z, zs⟩
    · cases cl
      · exact absurd rfl (hOK.1 rfl)
      · simp at hf
    · cases cl
      · obtain ⟨hz, hu⟩ := hM z (by simp [middleString])
        exact ⟨z, hz, hu⟩
      · obtain ⟨hz, hu⟩ := hE z (by simp [endsString])
        exact ⟨z, hz, hu⟩

omit [Fintype α] in
theorem firstFix_none {s : Level3State α} :
    ∀ {xs : List (FreeMonoid α)}, firstFix read s xs = none → ∀ x ∈ xs, fixOf read s x = none
  | [], _, x, hx => by simp at hx
  | y :: ys, hy, x, hx => by
    simp only [firstFix] at hy
    rcases hf : fixOf read s y with _ | t
    · rw [hf] at hy
      simp only [Option.elim] at hy
      rcases List.mem_cons.1 hx with rfl | hx
      · exact hf
      · exact firstFix_none hy x hx
    · rw [hf] at hy; simp at hy

/-- A check failing without a fix leaves the test silent on its strings that read undecided. -/
theorem checkStep_failed_silent {s : Level3State α} (hs : Inv read s)
    {b : Fin K.size → FreeMonoid α} (h : checkStep read K s b = .inr .failed) (hr0 : 0 < K.rate)
    (hr1 : K.rate < 1) :
    ¬ seqRate K.rate (K.rate / 2) (K.level s.hypotheses) K.size
      (cnt (Utype read s.tree) K.size (List.ofFn b)) := by
  classical
  rcases checkStep_inr read K h with ⟨h1, -⟩ | ⟨-, hn⟩
  · cases h1
  have hnp : ¬ checkPasses read K s.hypotheses s.tree s.edges (List.ofFn b) := by
    unfold checkStep at h
    split_ifs at h with hp
    · cases h
    · exact hp
  have hall : ∀ x ∈ List.ofFn b, fixOf read s x = none := firstFix_none read hn
  have hle : (List.ofFn b).countP (startFails read s.tree s.edges)
      ≤ (List.ofFn b).countP (Utype read s.tree) :=
    List.countP_mono_left fun x hx hfx => fail_nofix read hs hfx (hall x hx)
  have hc : cnt (Utype read s.tree) K.size (List.ofFn b) = (List.ofFn b).countP (Utype read s.tree)
      := by simp only [cnt, List.take_of_length_le (le_of_eq List.length_ofFn)]
  intro hsil
  apply hnp
  refine ⟨K.size, by simp, ?_⟩
  rw [List.take_of_length_le (by simp)]
  rw [hc] at hsil
  unfold seqRate at hsil ⊢
  refine le_trans hsil (likelihoodRatio_anti hr0 (by linarith) (by linarith) hr1 hle ?_)
  exact (List.countP_le_length).trans (by simp)

/-- The check's silent base at rate `q`. -/
noncomputable def silentBase (K : CheckSettings) (q : ℝ) : ℝ :=
  q * (K.rate / (K.rate / 2)) + (1 - q) * ((1 - K.rate) / (1 - K.rate / 2))

variable (Dc : Measure (FreeMonoid α)) [IsProbabilityMeasure Dc]

/-- Where at most `q` of the check's strings read undecided, it fails without a fix with chance at
most `b(q)^n / a`. -/
theorem waste_le {s : Level3State α} (hs : Inv read s) {q : ℝ}
    (hq : Dc.real {x | Utype read s.tree x = true} ≤ q) (hq1 : q ≤ 1) (hr0 : 0 < K.rate)
    (hr1 : K.rate < 1) (hK : 0 < K.δ) :
    (Measure.pi fun _ : Fin K.size => Dc) {b | checkStep read K s b = .inr .failed}
      ≤ ENNReal.ofReal (silentBase K q ^ K.size / K.level s.hypotheses) := by
  classical
  have ha : 0 < K.level s.hypotheses := by unfold CheckSettings.level; positivity
  set a := K.level s.hypotheses
  set qU := Dc.real {x | Utype read s.tree x = true}
  have hqU0 : 0 ≤ qU := measureReal_nonneg
  have hsub : {b : Fin K.size → FreeMonoid α | checkStep read K s b = .inr .failed}
      ⊆ {b | ¬ seqRate K.rate (K.rate / 2) a K.size (cnt (Utype read s.tree) K.size (List.ofFn b))}
      := fun b hb => checkStep_failed_silent read K hs hb hr0 hr1
  rw [← ENNReal.ofReal_toReal (measure_ne_top _ _), ← measureReal_def]
  refine ENNReal.ofReal_le_ofReal ((measureReal_mono hsub).trans ?_)
  rw [pi_cnt Dc (Utype read s.tree) K.size K.size le_rfl
    (fun h => ¬ seqRate K.rate (K.rate / 2) a K.size h)]
  have hmono : ∀ h h', h ≤ h' → h' ≤ K.size →
      (if ¬ seqRate K.rate (K.rate / 2) a K.size h then (1 : ℝ) else 0)
        ≤ if ¬ seqRate K.rate (K.rate / 2) a K.size h' then (1 : ℝ) else 0 := by
    intro h h' hh hh'
    by_cases hs' : seqRate K.rate (K.rate / 2) a K.size h'
    · have : seqRate K.rate (K.rate / 2) a K.size h :=
        le_trans hs' (likelihoodRatio_anti hr0 (by linarith) (by linarith) hr1 hh hh')
      simp [this, hs']
    · simp only [hs', not_false_eq_true, if_true]
      split_ifs <;> norm_num
  calc _ ≤ binE K.size q fun h => if ¬ seqRate K.rate (K.rate / 2) a K.size h then 1 else 0 :=
        binE_le_of_le_on hqU0 hq hq1 K.size _ hmono
    _ = binE K.size q fun h => if seqRate K.rate (K.rate / 2) a K.size h then 0 else 1 :=
        binE_congr fun h _ => by split_ifs <;> simp_all
    _ ≤ _ := binE_silent_le hr0 hr1 (by linarith) (by linarith) ha (hqU0.trans hq) hq1 K.size

variable {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

/-- The success test fires and the check fails without a fix. -/
def WasteC (s : Level3State α) (d : CheckDraw α K) : Prop :=
  applyProbe read C s d.1 = .inr .consistent ∧ checkStep read K s d.2 = .inr .failed

/-- A step that changes the hypothesis reading nothing wrong keeps the invariants and lowers
`Ψr`. -/
theorem wrongC_keeps {s s' : Level3State α} {d : CheckDraw α K} (hs : Inv read s)
    (hg : WitInv M side s) (hx : ¬ WrongC M side C K read s d)
    (h : applyCheck read C K s d = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ WitInv M side s' ∧ s' = newHypothesis s'.tree s'.edges s'.hypotheses
      ∧ s'.hypotheses = s.hypotheses + 1 ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s := by
  refine check_keeps read C K M side (WrongStep M side read C)
    (fun _ _ _ hs hg hx h hne => eps_step_change M side read C hs hg hx h hne) hs hg ?_ h hne
  rintro (hw | ⟨hf, hw | hw⟩)
  · exact hx (Or.inl hw)
  · exact hx (Or.inr ⟨hf, hw⟩)
  · rcases applyCheck_cases read C K s d with ⟨-, hne'⟩ | ⟨-, he⟩
    · exact hne' hf
    · rw [he, hw] at h; cases h

variable (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]

/-- Within a stretch, a check failing without a fix has chance at most its bound. -/
theorem seg_waste_le {q : ℝ} (hq1 : q ≤ 1) (hr0 : 0 < K.rate) (hr1 : K.rate < 1) (hK : 0 < K.δ)
    (hHead : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      Dc.real {x | Utype read T x = true} ≤ q) (N : ℕ) :
    ∀ s : Level3State α, Inv read s → WitInv M side s →
      (Measure.pi fun _ : Fin N => D.prod (Measure.pi fun _ : Fin K.size => Dc))
          {ds | SegB (stopAt (applyCheck read C K) (WrongC M side C K read) .failed)
            (WasteC read C K) Same s (List.ofFn ds)}
        ≤ ENNReal.ofReal (silentBase K q ^ K.size / K.level s.hypotheses) := by
  classical
  induction N with
  | zero =>
    intro s _ _
    have : {ds : Fin 0 → CheckDraw α K | SegB (stopAt (applyCheck read C K)
        (WrongC M side C K read) .failed) (WasteC read C K) Same s (List.ofFn ds)} = ∅ := by
      ext ds; simp [SegB]
    rw [this, measure_empty]
    exact bot_le
  | succ N ih =>
    intro s hs hg
    rw [pi_cons_apply]
    set lv := ENNReal.ofReal (silentBase K q ^ K.size / K.level s.hypotheses)
    set fire : FreeMonoid α → Prop := fun x => applyProbe read C s x = .inr .consistent
    have hpt : ∀ d : CheckDraw α K,
        (Measure.pi fun _ : Fin N => D.prod (Measure.pi fun _ : Fin K.size => Dc))
          {ds | (Fin.cons d ds : Fin (N + 1) → CheckDraw α K) ∈ {ds : Fin (N + 1) → CheckDraw α K |
            SegB (stopAt (applyCheck read C K) (WrongC M side C K read) .failed)
              (WasteC read C K) Same s (List.ofFn ds)}}
        ≤ Set.indicator {d | WasteC read C K s d} 1 d
          + Set.indicator {d | ¬ fire d.1} (fun _ => lv) d := by
      intro d
      simp only [Set.mem_ofPred_eq, List.ofFn_succ, Fin.cons_zero, Fin.cons_succ, SegB]
      by_cases hB : WasteC read C K s d
      · simp only [hB, true_or, Set.ofPred_true, measure_univ]
        rw [Set.indicator_of_mem (show d ∈ {d | WasteC read C K s d} from hB)]
        exact le_add_right le_rfl
      simp only [hB, false_or]
      rw [Set.indicator_of_notMem (show d ∉ {d | WasteC read C K s d} from hB), zero_add]
      rcases hst : stopAt (applyCheck read C K) (WrongC M side C K read) .failed s d with s' | e
      · simp only [Sum.inl.injEq, exists_eq_left']
        by_cases hsm : Same s s'
        · simp only [hsm, true_and]
          have ha := applyCheck_same read C K hs (stopAt_inl _ _ _ hst).2 hsm
          have hnf : ¬ fire d.1 := fun hf => by
            simp only [fire] at hf
            rw [hf] at ha
            cases ha
          rw [Set.indicator_of_mem (show d ∈ {d | ¬ fire d.1} from hnf)]
          obtain ⟨hs', -, -, hh⟩ := step_same read C hs ha hsm.1 hsm.2
          have := ih s' hs' (witInv_congr M side hg hsm.1 hsm.2)
          rwa [hh] at this
        · simp only [hsm, false_and, Set.ofPred_false, measure_empty]
          exact bot_le
      · simp only [reduceCtorEq, false_and, exists_false, Set.ofPred_false, measure_empty]
        exact bot_le
    have hcls := cls_mem' M side read hs hg
    calc ∫⁻ d, _ ∂(D.prod (Measure.pi fun _ : Fin K.size => Dc))
        ≤ ∫⁻ d, (Set.indicator {d | WasteC read C K s d} 1 d
            + Set.indicator {d | ¬ fire d.1} (fun _ => lv) d)
            ∂(D.prod (Measure.pi fun _ : Fin K.size => Dc)) := lintegral_mono hpt
      _ = (D.prod (Measure.pi fun _ : Fin K.size => Dc)) {d | WasteC read C K s d}
            + lv * (D.prod (Measure.pi fun _ : Fin K.size => Dc)) {d | ¬ fire d.1} := by
          rw [lintegral_add_left (measurable_one.indicator (Set.to_countable _).measurableSet),
            lintegral_indicator_one (Set.to_countable _).measurableSet,
            lintegral_indicator_const (Set.to_countable _).measurableSet]
      _ ≤ lv * D {x | fire x} + lv * D {x | ¬ fire x} := by
          gcongr
          · rw [Measure.prod_apply (Set.to_countable _).measurableSet]
            calc ∫⁻ x, (Measure.pi fun _ : Fin K.size => Dc)
                  (Prod.mk x ⁻¹' {d | WasteC read C K s d}) ∂D
                ≤ ∫⁻ x, Set.indicator {x | fire x} (fun _ => lv) x ∂D := lintegral_mono fun x => by
                  by_cases hf : fire x
                  · rw [Set.indicator_of_mem (show x ∈ {x | fire x} from hf)]
                    exact (measure_mono fun b hb => hb.2).trans
                      (waste_le read K Dc hs (hHead _ hcls) hq1 hr0 hr1 hK)
                  · rw [Set.indicator_of_notMem (show x ∉ {x | fire x} from hf)]
                    have : Prod.mk x ⁻¹' {d | WasteC read C K s d} = ∅ := by
                      ext b
                      simp only [Set.mem_preimage, Set.mem_ofPred_eq, Set.mem_empty_iff_false,
                        iff_false]
                      exact fun h => hf h.1
                    rw [this, measure_empty]
              _ = lv * D {x | fire x} := by
                  rw [lintegral_indicator_const (Set.to_countable _).measurableSet]
          · have : {d : CheckDraw α K | ¬ fire d.1} = {x | ¬ fire x} ×ˢ Set.univ := by
              ext d; simp
            rw [this, Measure.prod_prod, measure_univ, mul_one]
      _ = lv := by
          rw [← mul_add, show {x | ¬ fire x} = {x | fire x}ᶜ from rfl,
            measure_add_measure_compl (Set.to_countable _).measurableSet, measure_univ, mul_one]

/-- Over the round, reading nothing wrong, a check failing without a fix has chance at most the
sum of the stretches' bounds. -/
theorem anyB_waste_le {q : ℝ} (hq1 : q ≤ 1) (hr0 : 0 < K.rate) (hr1 : K.rate < 1) (hK : 0 < K.δ)
    (hHead : ∀ T ∈ (classSet (Fintype.card σ) : Finset (DTree α)),
      Dc.real {x | Utype read T x = true} ≤ q) (P : ℕ) :
    (Measure.pi fun _ : Fin P => D.prod (Measure.pi fun _ : Fin K.size => Dc))
        {ds | AnyB (stopAt (applyCheck read C K) (WrongC M side C K read) .failed)
          (WasteC read C K) initialState (List.ofFn ds)}
      ≤ ∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)),
        ENNReal.ofReal (silentBase K q ^ K.size / K.level j) := by
  classical
  set w : ℕ → ENNReal := fun j => ENNReal.ofReal (silentBase K q ^ K.size / K.level j)
  set stepW := stopAt (applyCheck read C K) (WrongC M side C K read) (RoundEnd.failed :
    RoundEnd Level3Site)
  have hΨ := Ψr_start (α := α) (Fintype.card σ)
  have hany := anyB_le stepW (WasteC read C K) Same (D.prod (Measure.pi fun _ : Fin K.size => Dc))
    (fun s => Inv read s ∧ WitInv M side s) (fun _ => True) (Ψr (Fintype.card σ))
    Level3State.hypotheses w
    (fun s d s' hI h hsm _ => by
      obtain ⟨hs, hg⟩ := hI
      have ha := applyCheck_same read C K hs (stopAt_inl _ _ _ h).2 hsm
      obtain ⟨h1, h2, -, h4⟩ := step_same read C hs ha hsm.1 hsm.2
      exact ⟨⟨h1, witInv_congr M side hg hsm.1 hsm.2⟩, h2 _, h4⟩)
    (fun s d s' hI h hsm _ => by
      obtain ⟨hs, hg⟩ := hI
      obtain ⟨hW', h⟩ := stopAt_inl _ _ _ h
      obtain ⟨h1, h2, -, h4, h5⟩ := wrongC_keeps read C K M side hs hg hW' h hsm
      exact ⟨⟨h1, h2⟩, trivial, h5, h4⟩)
    (fun s hI _ N => seg_waste_le read C K Dc M side D hq1 hr0 hr1 hK hHead N s hI.1 hI.2)
    P initialState ⟨inv_start read, witInv_start M side⟩
  have h0 : (initialState : Level3State α).hypotheses = 0 := rfl
  have hseg0 := seg_waste_le read C K Dc M side D hq1 hr0 hr1 hK hHead P initialState
    (inv_start read) (witInv_start M side)
  rw [h0] at hany hseg0
  calc _ ≤ _ := hany
    _ ≤ w 0 + ∑ i ∈ Finset.Ico (0 + 1) (0 + 1 + Ψr (Fintype.card σ) (initialState : Level3State α)),
          w i := add_le_add hseg0 le_rfl
    _ = ∑ i ∈ Finset.range (1 + Ψr (Fintype.card σ) (initialState : Level3State α)), w i := by
        rw [Finset.range_eq_Ico, Finset.sum_eq_sum_Ico_succ_bot
          (show 0 < 1 + Ψr (Fintype.card σ) (initialState : Level3State α) by omega)]
    _ ≤ _ := Finset.sum_le_sum_of_subset (Finset.range_subset_range.2 (by omega))

end Eps

end Random

end OrthoDFA
