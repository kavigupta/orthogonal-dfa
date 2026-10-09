import OrthoDFA.RoundLevel
import OrthoDFA.Proofs.TrichotomyBatch

/-!
# `RoundTrichotomyLevel`

A reading ends the round badly with chance at most its gate's spent failure chance and the
refusal sample's and certificate's (`reading_bad_le`), whatever the readings before it; summing
over the readings the round makes gives the bound (`round_bad_aux`).
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Q : Type*}

section Reading

variable (C : RoundCfg α) (R : CutReads α) (A : DFA (FreeMonoid α) Q)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (CertGood : KState α → Prop)

/-- The per-reading failure chance: the gate's at reading `j`, the refusal sample's miss and the
certificate's. -/
noncomputable def readBound (j : ℕ) (αc ν : ℝ) : ℝ :=
  2 * (Nat.log 2 (C.ng / 30) + 2) * (C.a / 2 ^ j) + ((1 - ν) ^ C.nr + αc)

open scoped Classical in
/-- Given its probes, a reading ends the round badly with chance at most `readBound`. -/
theorem reading_section_le {αc η minCov ν : ℝ} (hacc0 : 0 ≤ C.acc) (hacc1 : C.acc ≤ 1)
    (hf : 0 ≤ C.f) (ha : 0 ≤ C.a) (hν : ν ≤ 1)
    (hcert : ∀ R s, (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert R s cs = true ∧ ¬ CertGood s} ≤ αc)
    (hist : List C.Draws) (j : ℕ) (s₀ : KState α)
    (first : List (FreeMonoid α)) (pr : Fin C.np → FreeMonoid α) :
    ((Measure.pi fun _ : Fin C.ng => D).prod ((Measure.pi fun _ : Fin C.nr => D).prod
        (Measure.pi fun _ : Fin C.nc => D))).real
      {z | ∃ e, readingStep C R hist j s₀ first (pr, z) = .done e
        ∧ ¬ RoundEndHolds C R A D CertGood η minCov ν e}
      ≤ readBound C j αc ν := by
  set s := runPassK C.K R C.k s₀ (first ++ List.ofFn pr) with hs
  set aj := C.a / 2 ^ j
  have haj : 0 ≤ aj := by positivity
  have hαc : 0 ≤ αc := (measureReal_nonneg).trans (hcert R s)
  have hmiss0 : 0 ≤ (1 - ν) ^ C.nr := pow_nonneg (by linarith) _
  have hRB0 : 0 ≤ readBound C j αc ν := by unfold readBound; positivity
  by_cases hP : C.Pmax < s.tree.paths.length
  · refine le_of_eq_of_le ?_ hRB0
    rw [show {z | ∃ e, readingStep C R hist j s₀ first (pr, z) = .done e
        ∧ ¬ RoundEndHolds C R A D CertGood η minCov ν e} = ∅ from ?_, measureReal_empty]
    refine Set.eq_empty_of_forall_notMem fun z ⟨e, he, hne⟩ => ?_
    simp only [readingStep, ← hs, if_pos hP, ReadStep.done.injEq] at he
    subst he
    exact hne trivial
  push Not at hP
  obtain ⟨Bad, hBad, hgood⟩ := gate_settled R s D C.ng hacc0 hacc1 haj
  set gu := C.gu R hist
  set Miss := {br : Fin C.nr → FreeMonoid α | ν < D.real {x | NAOff R s.tree s.edges C.k gu x}
    ∧ ∀ i, ¬ NAOff R s.tree s.edges C.k gu (br i)}
  set CB := {cs : Fin C.nc → FreeMonoid α | C.cert R s cs = true ∧ ¬ CertGood s}
  have hsub : {z | ∃ e, readingStep C R hist j s₀ first (pr, z) = .done e
      ∧ ¬ RoundEndHolds C R A D CertGood η minCov ν e}
      ⊆ Bad ×ˢ Set.univ ∪ Set.univ ×ˢ (Miss ×ˢ Set.univ) ∪ Set.univ ×ˢ (Set.univ ×ˢ CB) := by
    rintro ⟨bg, br, cs⟩ ⟨e, he, hne⟩
    by_cases hbad : bg ∈ Bad
    · exact .inl (.inl ⟨hbad, trivial⟩)
    obtain ⟨hpass1, hrefuse⟩ := hgood bg hbad
    simp only [readingStep, ← hs, if_neg (not_lt.2 hP)] at he
    split_ifs at he with hgc hlv hfire
    · simp only [ReadStep.done.injEq] at he
      subst he
      simp only [RoundEndHolds, not_and_or] at hne
      rcases hne with hne | hne
      · exact absurd (hpass1 hgc.1) hne
      · exact .inr ⟨trivial, trivial, hgc.2, hne⟩
    · simp only [ReadStep.done.injEq] at he
      subst he
      exact absurd trivial hne
    · simp only [ReadStep.done.injEq] at he
      subst he
      simp only [RoundEndHolds, not_or, not_and_or, not_le] at hne
      obtain ⟨htau, hM, hne⟩ := hne
      have hB : ¬ ∃ i : Fin C.nr, (i : ℕ) < refusalStop br C.a
          (harvestTests R s.tree s.edges C.k C.L C.f C.c C.θM) (LiveEdge R s.tree s.edges C.k gu)
          ∧ LiveEdge R s.tree s.edges C.k gu (br i) := by
        rintro ⟨i, hi, hl⟩
        apply hlv
        rw [Ne, List.map_eq_nil_iff, List.filter_eq_nil_iff]
        push Not
        refine ⟨i, List.mem_finRange _, ?_⟩
        first | exact ⟨hi, hl⟩ | exact decide_eq_true ⟨hi, hl⟩
      have hclear := refusal_clear R s.tree s.edges C.k C.L hf ha gu br htau hM hB hfire
      refine .inl (.inr ⟨trivial, ?_, trivial⟩)
      refine ⟨?_, hclear⟩
      rcases hne with hne | hne
      · exact hne
      · simp only [_root_.not_imp, not_forall, decide_eq_true_eq] at hne
        obtain ⟨hgr, q₀, h, hq₀, hhq, hcov, hh, hneed⟩ := hne
        have hall := hrefuse hgr
        have herr : 1 - C.acc < D.real {x | StartDis R s.edges (h q₀) x} := by
          have := hall _ hhq
          rw [show {x | ¬ StartDis R s.edges (h q₀) x} = {x | StartDis R s.edges (h q₀) x}ᶜ
            from rfl, measureReal_compl (Set.to_countable _).measurableSet, probReal_univ] at this
          linarith
        have hNA := need_lt R A D _ h s.tree s.edges C.k q₀ gu hcov hh herr
        linarith
  set ν₁ := Measure.pi fun _ : Fin C.ng => D
  set ν₂ := Measure.pi fun _ : Fin C.nr => D
  set ν₃ := Measure.pi fun _ : Fin C.nc => D
  have hMiss := miss_all_le D (NAOff R s.tree s.edges C.k gu) C.nr hν
  have hCB := hcert R s
  calc (ν₁.prod (ν₂.prod ν₃)).real {z | ∃ e, readingStep C R hist j s₀ first (pr, z) = .done e
        ∧ ¬ RoundEndHolds C R A D CertGood η minCov ν e}
      ≤ (ν₁.prod (ν₂.prod ν₃)).real
          (Bad ×ˢ Set.univ ∪ Set.univ ×ˢ (Miss ×ˢ Set.univ) ∪ Set.univ ×ˢ (Set.univ ×ˢ CB)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ (ν₁.prod (ν₂.prod ν₃)).real (Bad ×ˢ Set.univ)
        + (ν₁.prod (ν₂.prod ν₃)).real (Set.univ ×ˢ (Miss ×ˢ Set.univ))
        + (ν₁.prod (ν₂.prod ν₃)).real (Set.univ ×ˢ (Set.univ ×ˢ CB)) :=
        (measureReal_union_le _ _).trans (add_le_add (measureReal_union_le _ _) le_rfl)
    _ = ν₁.real Bad + ν₂.real Miss + ν₃.real CB := by
        simp only [measureReal_prod_prod, probReal_univ, mul_one, one_mul]
    _ ≤ readBound C j αc ν := by
        unfold readBound
        linarith

open scoped Classical in
/-- A reading ends the round badly with chance at most `readBound`, whatever came before it. -/
theorem reading_bad_le {αc η minCov ν : ℝ} (hacc0 : 0 ≤ C.acc) (hacc1 : C.acc ≤ 1)
    (hf : 0 ≤ C.f) (ha : 0 ≤ C.a) (hν : ν ≤ 1)
    (hcert : ∀ R s, (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert R s cs = true ∧ ¬ CertGood s} ≤ αc)
    (hist : List C.Draws) (j : ℕ) (s₀ : KState α)
    (first : List (FreeMonoid α)) :
    C.drawMeasure D {y | ∃ e, readingStep C R hist j s₀ first y = .done e
        ∧ ¬ RoundEndHolds C R A D CertGood η minCov ν e}
      ≤ ENNReal.ofReal (readBound C j αc ν) := by
  unfold RoundCfg.drawMeasure
  rw [Measure.prod_apply (Set.to_countable _).measurableSet]
  calc ∫⁻ pr, _ ∂_ ≤ ∫⁻ _pr, ENNReal.ofReal (readBound C j αc ν)
        ∂(Measure.pi fun _ : Fin C.np => D) := by
        refine lintegral_mono fun pr => ?_
        have := reading_section_le C R A D CertGood (η := η) (minCov := minCov) hacc0 hacc1 hf ha
          hν hcert hist j s₀ first pr
        rw [← ENNReal.ofReal_toReal (measure_ne_top _ _)]
        exact ENNReal.ofReal_le_ofReal this
    _ = ENNReal.ofReal (readBound C j αc ν) := by simp

end Reading

instance RoundCfg.drawMeasure_prob (C : RoundCfg α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] : IsProbabilityMeasure (C.drawMeasure D) := by
  unfold RoundCfg.drawMeasure; infer_instance

section Round

variable (C : RoundCfg α) (R : CutReads α) (A : DFA (FreeMonoid α) Q)
  (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (CertGood : KState α → Prop)

omit [Fintype α] [DecidableEq α] in
theorem piFinSuccAbove_zero {n : ℕ} {Y : Type*} [MeasurableSpace Y] (d : Fin (n + 1) → Y) :
    MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => Y) 0 d = (d 0, Fin.tail d) := by
  ext j
  · rfl
  · simp [MeasurableEquiv.piFinSuccAbove, Fin.removeNth, Fin.tail]

theorem roundAux_le (n j : ℕ) (hist : List C.Draws) (s : KState α)
    (first : List (FreeMonoid α)) (d : Fin n → C.Draws) :
    (roundAux C R n j hist s first d).2 ≤ n := by
  induction n generalizing j hist s first with
  | zero => simp [roundAux]
  | succ n ih =>
    simp only [roundAux]
    split
    · omega
    · exact Nat.succ_le_succ (ih ..)

open scoped Classical in
/-- Over `n` readings from reading `j`, the round ends badly with chance at most the expected sum
of `readBound` over the readings it makes. -/
theorem round_bad_aux {αc η minCov ν : ℝ} (hacc0 : 0 ≤ C.acc) (hacc1 : C.acc ≤ 1)
    (hf : 0 ≤ C.f) (ha : 0 ≤ C.a) (hν : ν ≤ 1)
    (hcert : ∀ R s, (Measure.pi fun _ : Fin C.nc => D).real
      {cs | C.cert R s cs = true ∧ ¬ CertGood s} ≤ αc) :
    ∀ (n j : ℕ) (hist : List C.Draws) (s : KState α) (first : List (FreeMonoid α)),
      (Measure.pi fun _ : Fin n => C.drawMeasure D)
          {d | ¬ RoundEndHolds C R A D CertGood η minCov ν
            (roundAux C R n j hist s first d).1}
        ≤ ∫⁻ d, ∑ i ∈ Finset.range (roundAux C R n j hist s first d).2,
            ENNReal.ofReal (readBound C (j + i) αc ν)
          ∂(Measure.pi fun _ : Fin n => C.drawMeasure D) := by
  intro n
  induction n with
  | zero =>
    intro j hist s first
    rw [show {d : Fin 0 → C.Draws | ¬ RoundEndHolds C R A D CertGood η minCov ν
        (roundAux C R 0 j hist s first d).1} = ∅ from
      Set.eq_empty_of_forall_notMem fun d hd => hd (by simp [roundAux, RoundEndHolds])]
    simp
  | succ n ih =>
    intro j hist s first
    set ρm := C.drawMeasure D
    set μn := Measure.pi fun _ : Fin n => ρm
    set β : ℕ → ℝ≥0∞ := fun i => ENNReal.ofReal (readBound C i αc ν)
    have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (n + 1) => ρm) 0
    set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (n + 1) => C.Draws) 0
    -- the round from its first reading's draws and the rest's
    set F : C.Draws × (Fin n → C.Draws) → RoundEnd α × ℕ := fun p =>
      match readingStep C R hist j s first p.1 with
      | .done e => (e, 1)
      | .rerun s' first' =>
        ((roundAux C R n (j + 1) (hist ++ [p.1]) s' first' p.2).1,
          (roundAux C R n (j + 1) (hist ++ [p.1]) s' first' p.2).2 + 1)
    have hF : ∀ d, roundAux C R (n + 1) j hist s first d = F (e d) := by
      intro d
      rw [piFinSuccAbove_zero]
      rfl
    set G : C.Draws → ℝ≥0∞ := fun y => match readingStep C R hist j s first y with
      | .done _ => 0
      | .rerun s' first' => ∫⁻ d', ∑ i ∈ Finset.range
          (roundAux C R n (j + 1) (hist ++ [y]) s' first' d').2, β (j + 1 + i) ∂μn
    set Bad := {y | ∃ e, readingStep C R hist j s first y = .done e
      ∧ ¬ RoundEndHolds C R A D CertGood η minCov ν e}
    have hcount : ∀ (T : Set (C.Draws × (Fin n → C.Draws))), MeasurableSet T :=
      fun T => (Set.to_countable T).measurableSet
    -- the left side, reading by reading
    have hL : (Measure.pi fun _ : Fin (n + 1) => ρm)
        {d | ¬ RoundEndHolds C R A D CertGood η minCov ν (roundAux C R (n + 1) j hist s
          first d).1}
        ≤ ∫⁻ y, Bad.indicator 1 y + G y ∂ρm := by
      have hpre : {d | ¬ RoundEndHolds C R A D CertGood η minCov ν
          (roundAux C R (n + 1) j hist s first d).1}
          = e ⁻¹' {p | ¬ RoundEndHolds C R A D CertGood η minCov ν (F p).1} := by
        ext d; simp only [Set.mem_ofPred_eq, Set.mem_preimage, hF]
      rw [hpre, hmp.measure_preimage (hcount _).nullMeasurableSet,
        Measure.prod_apply (hcount _)]
      refine lintegral_mono fun y => ?_
      simp only [F, G, Set.preimage, Set.mem_ofPred_eq]
      rcases hstep : readingStep C R hist j s first y with e' | ⟨s', first'⟩
      · simp only []
        by_cases hb : RoundEndHolds C R A D CertGood η minCov ν e'
        · simp [hb]
        · have hy : y ∈ Bad := ⟨e', hstep, hb⟩
          simp only [hb, not_false_eq_true, Set.setOf_true, measure_univ, add_zero,
            Set.indicator_of_mem hy, Pi.one_apply, le_refl]
      · simp only []
        refine le_trans ?_ (le_add_self)
        exact ih (j + 1) _ s' first'
    -- the right side, reading by reading
    set Hf : C.Draws × (Fin n → C.Draws) → ℝ≥0∞ := fun p =>
      ∑ i ∈ Finset.range (F p).2, β (j + i)
    have hR : ∫⁻ d, ∑ i ∈ Finset.range (roundAux C R (n + 1) j hist s first d).2, β (j + i)
          ∂(Measure.pi fun _ : Fin (n + 1) => ρm)
        = ∫⁻ y, β j + G y ∂ρm := by
      calc ∫⁻ d, ∑ i ∈ Finset.range (roundAux C R (n + 1) j hist s first d).2, β (j + i)
            ∂(Measure.pi fun _ : Fin (n + 1) => ρm)
          = ∫⁻ d, Hf (e d) ∂(Measure.pi fun _ : Fin (n + 1) => ρm) :=
            lintegral_congr fun d => by simp only [Hf, hF]
        _ = ∫⁻ p, Hf p ∂(ρm.prod μn) := hmp.lintegral_comp (measurable_of_countable Hf)
        _ = ∫⁻ y, ∫⁻ d', Hf (y, d') ∂μn ∂ρm :=
            lintegral_prod _ (measurable_of_countable Hf).aemeasurable
        _ = ∫⁻ y, β j + G y ∂ρm := by
          refine lintegral_congr fun y => ?_
          simp only [Hf, F, G]
          rcases readingStep C R hist j s first y with e' | ⟨s', first'⟩
          · simp
          · simp only []
            rw [← (show ∫⁻ _d' : Fin n → C.Draws, β j ∂μn = β j by simp),
              ← lintegral_add_left measurable_const]
            refine lintegral_congr fun d' => ?_
            rw [Finset.sum_range_succ', add_comm]
            congr 1
            refine Finset.sum_congr rfl fun i _ => ?_
            ring_nf
    have hbad := reading_bad_le C R A D CertGood (η := η) (minCov := minCov) hacc0 hacc1 hf ha
      hν hcert hist j s first
    calc _ ≤ ∫⁻ y, Bad.indicator 1 y + G y ∂ρm := hL
      _ = ρm Bad + ∫⁻ y, G y ∂ρm := by
          rw [lintegral_add_left (measurable_of_countable _), lintegral_indicator_one
            (Set.to_countable _).measurableSet]
      _ ≤ β j + ∫⁻ y, G y ∂ρm := add_le_add hbad le_rfl
      _ = ∫⁻ y, β j + G y ∂ρm := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ = _ := hR.symm

end Round

theorem round_trichotomy_level : RoundTrichotomyLevel := by
  intro α _ _ Q C A D _ seed CertGood αc η minCov ν Rmax hacc0 hacc1 hf ha hν hcert R
  set P := Measure.pi fun _ : Fin Rmax => C.drawMeasure D
  set N : (Fin Rmax → C.Draws) → ℕ := fun d => (roundAux C R Rmax 0 [] (initialK C.K R seed) [] d).2
  set γ : ℝ := (1 - ν) ^ C.nr + αc
  set c₀ : ℝ := 2 * (Nat.log 2 (C.ng / 30) + 2)
  have hc₀ : 0 ≤ c₀ := by positivity
  have hαc : 0 ≤ αc := measureReal_nonneg.trans (hcert R (initialK C.K R seed))
  have hγ : 0 ≤ γ := by
    have := pow_nonneg (by linarith : (0 : ℝ) ≤ 1 - ν) C.nr
    positivity
  have hNle : ∀ d, N d ≤ Rmax := fun d => roundAux_le C R _ _ _ _ _ d
  have hNint : Integrable (fun d => (N d : ℝ)) P :=
    Integrable.of_bound (measurable_of_countable _).aestronglyMeasurable Rmax
      (ae_of_all _ fun d => by
        rw [Real.norm_of_nonneg (Nat.cast_nonneg _)]; exact_mod_cast hNle d)
  have hEN : 0 ≤ ∫ d, (N d : ℝ) ∂P := integral_nonneg fun d => Nat.cast_nonneg _
  have hrhs0 : 0 ≤ 2 * c₀ * C.a + (∫ d, (N d : ℝ) ∂P) * γ := by positivity
  rw [show 4 * ((Nat.log 2 (C.ng / 30) : ℝ) + 2) * C.a = 2 * c₀ * C.a by simp only [c₀]; ring]
  have h := round_bad_aux C R A D CertGood (η := η) (minCov := minCov) hacc0 hacc1 hf ha
    hν hcert Rmax 0 [] (initialK C.K R seed) []
  have hrb : ∀ i, readBound C i αc ν = c₀ * (C.a / 2 ^ i) + γ := fun i => rfl
  have hpt : ∀ d, ∑ i ∈ Finset.range (N d), ENNReal.ofReal (readBound C (0 + i) αc ν)
      ≤ ENNReal.ofReal (2 * c₀ * C.a + N d * γ) := by
    intro d
    have hnn : ∀ i, 0 ≤ readBound C (0 + i) αc ν := fun i => by
      rw [hrb]; positivity
    rw [← ENNReal.ofReal_sum_of_nonneg (fun i _ => hnn i)]
    refine ENNReal.ofReal_le_ofReal ?_
    simp only [hrb, zero_add, Finset.sum_add_distrib, Finset.sum_const, Finset.card_range,
      nsmul_eq_mul]
    have hgeo : ∑ i ∈ Finset.range (N d), c₀ * (C.a / 2 ^ i)
        = c₀ * C.a * ∑ i ∈ Finset.range (N d), (1 / 2 : ℝ) ^ i := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [one_div_pow]; ring
    have := sum_geometric_two_le (N d)
    rw [hgeo]
    nlinarith [mul_nonneg hc₀ ha]
  have hint : Integrable (fun d => 2 * c₀ * C.a + (N d : ℝ) * γ) P :=
    (integrable_const _).add (hNint.mul_const _)
  have hlin : ∫⁻ d, ENNReal.ofReal (2 * c₀ * C.a + N d * γ) ∂P
      = ENNReal.ofReal (2 * c₀ * C.a + (∫ d, (N d : ℝ) ∂P) * γ) := by
    rw [← ofReal_integral_eq_lintegral_ofReal hint (ae_of_all _ fun d => by positivity),
      integral_add (integrable_const _) (hNint.mul_const _), integral_const, integral_mul_const]
    simp
  have hle : P {d | ¬ RoundEndHolds C R A D CertGood η minCov ν
      (roundAux C R Rmax 0 [] (initialK C.K R seed) [] d).1}
      ≤ ENNReal.ofReal (2 * c₀ * C.a + (∫ d, (N d : ℝ) ∂P) * γ) :=
    h.trans ((lintegral_mono hpt).trans hlin.le)
  rw [measureReal_def]
  exact ENNReal.toReal_le_of_le_ofReal hrhs0 hle

section Quality

variable (C : RoundCfg α) (R : CutReads α)

theorem runPassK_append (K : StageKnobs α) (k : ℕ) (s : KState α) (l₁ l₂ : List (FreeMonoid α)) :
    runPassK K R k s (l₁ ++ l₂) = runPassK K R k (runPassK K R k s l₁) l₂ :=
  List.foldl_append ..

open scoped Classical in
/-- The rerun lists a refusal sample `br` can give: its draws picked by a set of positions. -/
noncomputable def sublistsOf (br : Fin C.nr → FreeMonoid α) : Finset (List (FreeMonoid α)) :=
  (Finset.univ : Finset (Finset (Fin C.nr))).image fun S =>
    ((List.finRange C.nr).filter fun i => decide (i ∈ S)).map br

theorem card_sublistsOf (br : Fin C.nr → FreeMonoid α) : (sublistsOf C br).card ≤ 2 ^ C.nr := by
  unfold sublistsOf
  refine Finset.card_image_le.trans (le_of_eq ?_)
  rw [Finset.card_univ, Fintype.card_finset, Fintype.card_fin]

open scoped Classical in
theorem mem_sublistsOf (br : Fin C.nr → FreeMonoid α) (q : Fin C.nr → Bool) :
    ((List.finRange C.nr).filter q).map br ∈ sublistsOf C br :=
  Finset.mem_image.2 ⟨Finset.univ.filter fun i => q i = true, Finset.mem_univ _, by
    congr 1
    exact List.filter_congr fun i _ => by simp⟩

open scoped Classical in
theorem readingStep_spec (hist : List C.Draws) (j : ℕ) (s₀ : KState α)
    (first : List (FreeMonoid α)) (y : C.Draws) :
    (∀ e, readingStep C R hist j s₀ first y = .done e
      → e.state = runPassK C.K R C.k s₀ (first ++ List.ofFn y.1))
    ∧ ∀ s' first', readingStep C R hist j s₀ first y = .rerun s' first'
      → s' = runPassK C.K R C.k s₀ (first ++ List.ofFn y.1) ∧ first' ∈ sublistsOf C y.2.2.1 := by
  constructor
  · intro e h
    simp only [readingStep] at h
    split_ifs at h <;> simp only [ReadStep.done.injEq, reduceCtorEq] at h <;> subst h <;> rfl
  · intro s' first' h
    simp only [readingStep] at h
    split_ifs at h with h1 h2 h3
    simp only [ReadStep.rerun.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨rfl, mem_sublistsOf C _ _⟩

theorem roundAux_state :
    ∀ (n j : ℕ) (hist : List C.Draws) (s : KState α) (first : List (FreeMonoid α))
      (d : Fin n → C.Draws),
      (roundAux C R n j hist s first d).1.state
        = runPassK C.K R C.k s (roundProbes C R n j hist s first d)
  | 0, _, _, _, _, _ => rfl
  | n + 1, j, hist, s, first, d => by
    obtain ⟨hd, hr⟩ := readingStep_spec C R hist j s first (d 0)
    simp only [roundAux, roundProbes]
    split
    · rename_i e he
      exact hd e he
    · rename_i s' first' he
      rw [roundAux_state n, runPassK_append, ← (hr s' first' he).1]

/-- The probe lists a round over the draws `d` can take in `r` readings, from first probes among
`firsts`. -/
noncomputable def candR : (n : ℕ) → (Fin n → C.Draws) → ℕ → Finset (List (FreeMonoid α))
    → Finset (List (FreeMonoid α))
  | _, _, 0, _ => {[]}
  | 0, _, _ + 1, _ => ∅
  | n + 1, d, r + 1, firsts => firsts.biUnion fun f =>
    insert (f ++ List.ofFn (d 0).1)
      ((candR n (Fin.tail d) r (sublistsOf C (d 0).2.2.1)).image
        ((f ++ List.ofFn (d 0).1) ++ ·))

theorem roundProbes_mem :
    ∀ (n j : ℕ) (hist : List C.Draws) (s : KState α) (first : List (FreeMonoid α))
      (d : Fin n → C.Draws) (firsts : Finset (List (FreeMonoid α))), first ∈ firsts →
      roundProbes C R n j hist s first d
        ∈ candR C n d (roundAux C R n j hist s first d).2 firsts
  | 0, _, _, _, _, _, _, _ => by simp [roundProbes, roundAux, candR]
  | n + 1, j, hist, s, first, d, firsts, hf => by
    obtain ⟨-, hr⟩ := readingStep_spec C R hist j s first (d 0)
    simp only [roundProbes, roundAux]
    split
    · simp only [candR]
      exact Finset.mem_biUnion.2 ⟨first, hf, Finset.mem_insert_self _ _⟩
    · rename_i s' first' he
      simp only [candR]
      exact Finset.mem_biUnion.2 ⟨first, hf, Finset.mem_insert_of_mem (Finset.mem_image.2
        ⟨_, roundProbes_mem n (j + 1) _ s' first' (Fin.tail d) _ (hr s' first' he).2, rfl⟩)⟩

theorem card_candR :
    ∀ (n : ℕ) (d : Fin n → C.Draws) (r : ℕ) (firsts : Finset (List (FreeMonoid α))),
      (candR C n d r firsts).card ≤ max 1 firsts.card * 2 ^ ((C.nr + 1) * r)
  | _, _, 0, firsts => by simp [candR]
  | 0, _, _ + 1, firsts => by simp [candR]
  | n + 1, d, r + 1, firsts => by
    have ih := card_candR n (Fin.tail d) r (sublistsOf C (d 0).2.2.1)
    have hS : max 1 (sublistsOf C (d 0).2.2.1).card ≤ 2 ^ C.nr :=
      max_le (Nat.one_le_two_pow) (card_sublistsOf C _)
    have hone : ∀ f ∈ firsts, (insert (f ++ List.ofFn (d 0).1)
        ((candR C n (Fin.tail d) r (sublistsOf C (d 0).2.2.1)).image
          ((f ++ List.ofFn (d 0).1) ++ ·))).card ≤ 2 ^ ((C.nr + 1) * (r + 1)) := by
      intro f _
      refine (Finset.card_insert_le _ _).trans ?_
      have h1 := (Finset.card_image_le (s := candR C n (Fin.tail d) r (sublistsOf C (d 0).2.2.1))
        (f := ((f ++ List.ofFn (d 0).1) ++ ·))).trans (ih.trans (Nat.mul_le_mul_right _ hS))
      have h2 : 2 ^ ((C.nr + 1) * (r + 1)) = 2 * (2 ^ C.nr * 2 ^ ((C.nr + 1) * r)) := by ring
      have h3 : 1 ≤ 2 ^ C.nr * 2 ^ ((C.nr + 1) * r) := Nat.one_le_iff_ne_zero.2 (by positivity)
      omega
    simp only [candR]
    refine (Finset.card_biUnion_le).trans ((Finset.sum_le_sum hone).trans ?_)
    rw [Finset.sum_const, smul_eq_mul]
    exact Nat.mul_le_mul_right _ (le_max_right _ _)

theorem prefixMax_pos (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {k L : ℕ}
    (hkL : k ≤ L) (hlen : ∀ᵐ x ∂D, x.toList.length = L) : 0 < prefixMax D k := by
  classical
  have h1 : ∑ x ∈ wordsOf (α := α) L, D.real {x} = 1 := by
    have := real_eq_sum_words D hlen Set.univ
    simpa using this.symm
  obtain ⟨x, hx, hpos⟩ : ∃ x ∈ wordsOf (α := α) L, 0 < D.real {x} := by
    by_contra h
    push Not at h
    have : ∑ x ∈ wordsOf (α := α) L, D.real {x} ≤ 0 := Finset.sum_nonpos h
    linarith
  have hle : D.real {x} ≤ D.real {y | (prefixOf x k).toList <+: y.toList} :=
    measureReal_mono (fun y hy => by
      rw [Set.mem_singleton_iff.1 hy]; simp only [Set.mem_ofPred_eq, prefixOf,
        FreeMonoid.toList_ofList]; exact List.take_prefix _ _) (measure_ne_top _ _)
  refine lt_of_lt_of_le (hpos.trans_le hle) ?_
  refine le_trans (le_of_eq ?_) (le_ciSup (f := fun p : FreeMonoid α =>
    if p.toList.length = k then D.real {x | p.toList <+: x.toList} else 0)
    ⟨1, by rintro _ ⟨p, rfl⟩; simp only []; split_ifs <;> simp [measureReal_le_one]⟩
    (prefixOf x k))
  rw [if_pos (length_prefixOf (by rw [mem_wordsOf.1 hx]; exact hkL))]

end Quality

theorem round_quality_level : RoundQualityLevel := by
  intro α _ _ Ω _ μ _ Q C A O B F D _ seed δ Rmax d hf hc hδ hδ1 hkL hlen hV
  have hpm := prefixMax_pos D hkL hlen
  have hlog : 0 < Real.log (5 / δ) := Real.log_pos (by rw [lt_div_iff₀ hδ]; linarith)
  have heps : ∀ r, 0 < roundEps C D δ r := fun r => by
    unfold roundEps
    refine Real.sqrt_pos.2 (div_pos (mul_pos hpm ?_) two_pos)
    have : 0 ≤ (((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2 :=
      mul_nonneg (Nat.cast_nonneg _) (Real.log_nonneg (by norm_num))
    linarith
  have hq := fun (r : ℕ) (probes : List (FreeMonoid α)) => quality_holds (μ := μ) A O B F C.K D
    hkL seed probes hf hc (heps r) hlen hV
  choose Ef hEf hgood using hq
  set E : Set Ω := ⋃ r ∈ Finset.range (Rmax + 1), ⋃ p ∈ candR C Rmax d r {[]}, Ef r p
  refine ⟨E, ?_, fun ω hω => ?_⟩
  · have hterm : ∀ r, ∑ p ∈ candR C Rmax d r {[]}, μ.real (Ef r p) ≤ δ / 2 ^ (r + 1) := by
      intro r
      have hcard : ((candR C Rmax d r {[]}).card : ℝ) ≤ 2 ^ ((C.nr + 1) * r) := by
        have := card_candR C Rmax d r {[]}
        simp only [Finset.card_singleton, max_self, one_mul] at this
        exact_mod_cast this
      have hexp : 5 * Real.exp (-2 * roundEps C D δ r ^ 2 / prefixMax D C.k)
          = δ / 2 ^ ((C.nr + 1) * r + r + 1) := by
        unfold roundEps
        rw [Real.sq_sqrt (by
          have : 0 ≤ (((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2 :=
            mul_nonneg (Nat.cast_nonneg _) (Real.log_nonneg (by norm_num))
          positivity)]
        rw [show -2 * (prefixMax D C.k * ((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2
            + Real.log (5 / δ)) / 2) / prefixMax D C.k
            = -((((C.nr + 1) * r + r + 1 : ℕ) : ℝ) * Real.log 2) - Real.log (5 / δ) by
          field_simp; ring]
        rw [Real.exp_sub, Real.exp_neg, ← Real.log_rpow two_pos, Real.exp_log (by positivity),
          Real.exp_log (by positivity), Real.rpow_natCast]
        field_simp
      calc ∑ p ∈ candR C Rmax d r {[]}, μ.real (Ef r p)
          ≤ ∑ _p ∈ candR C Rmax d r {[]}, δ / 2 ^ ((C.nr + 1) * r + r + 1) :=
            Finset.sum_le_sum fun p _ => (hEf r p).trans (le_of_eq hexp)
        _ = (candR C Rmax d r {[]}).card * (δ / 2 ^ ((C.nr + 1) * r + r + 1)) := by
            rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ 2 ^ ((C.nr + 1) * r) * (δ / 2 ^ ((C.nr + 1) * r + r + 1)) :=
            mul_le_mul_of_nonneg_right hcard (by positivity)
        _ = δ / 2 ^ (r + 1) := by
            rw [pow_add, pow_add (2 : ℝ) ((C.nr + 1) * r)]
            field_simp
            ring
    calc μ.real E ≤ ∑ r ∈ Finset.range (Rmax + 1), μ.real (⋃ p ∈ candR C Rmax d r {[]}, Ef r p) :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ r ∈ Finset.range (Rmax + 1), δ / 2 ^ (r + 1) :=
          Finset.sum_le_sum fun r _ => (measureReal_biUnion_finset_le _ _).trans (hterm r)
      _ ≤ δ := by
          have h := sum_geometric_two_le (Rmax + 1)
          have : ∑ r ∈ Finset.range (Rmax + 1), δ / 2 ^ (r + 1)
              = δ / 2 * ∑ r ∈ Finset.range (Rmax + 1), (1 / 2 : ℝ) ^ r := by
            rw [Finset.mul_sum]
            refine Finset.sum_congr rfl fun r _ => ?_
            rw [one_div_pow, pow_succ]; ring
          rw [this]
          nlinarith
  · simp only []
    set R := readsAt O B F ω
    have hmem := roundProbes_mem C R Rmax 0 [] (initialK C.K R seed) [] d {[]}
      (Finset.mem_singleton_self _)
    have hr : (roundAux C R Rmax 0 [] (initialK C.K R seed) [] d).2 ∈ Finset.range (Rmax + 1) :=
      Finset.mem_range.2 (Nat.lt_succ_of_le (roundAux_le C R _ _ _ _ _ d))
    simp only [E, Set.mem_iUnion, not_exists] at hω
    rw [roundAux_state]
    exact hgood _ _ ω (hω _ hr _ hmem)

end OrthoDFA
