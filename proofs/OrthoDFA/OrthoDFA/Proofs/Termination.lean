import OrthoDFA.Termination

/-!
# The learner stops

Count, for each round, the target states some population so far is heavy enough on to force
them cut well, once for each of `qualityBound`'s two thresholds.  The count never falls and is
at most `2·|Q|`.  On the event that every round meets its three specs, a failing state's own
population is heavy on a state the family cuts badly, which the specs say no earlier population
was; so each failing round raises the count.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
variable {S : Type*} [Stringlike S] {Q R : Type*}

lemma measurableSet_stringlike (s : Set S) : MeasurableSet s := s.to_countable.measurableSet

lemma reaching_eq_cond {H : DFA S R} {Dsamp : Measure S} {h : R}
    (hne : Dsamp {w | H.state w = h} ≠ 0) :
    reaching H Dsamp h = Dsamp[|{w | H.state w = h}] := if_neg hne

lemma reaching_real_le {H : DFA S R} {Dsamp : Measure S} [IsFiniteMeasure Dsamp] {h : R}
    (hpos : 0 < Dsamp.real {w | H.state w = h}) (t : Set S) :
    (reaching H Dsamp h).real t ≤ Dsamp.real t / Dsamp.real {w | H.state w = h} := by
  have hne : Dsamp {w | H.state w = h} ≠ 0 := fun h0 => by
    simp [measureReal_def, h0] at hpos
  rw [reaching_eq_cond hne, measureReal_def, cond_apply (measurableSet_stringlike _),
    ENNReal.toReal_mul, ENNReal.toReal_inv, ← measureReal_def, ← measureReal_def,
    inv_mul_eq_div]
  exact div_le_div_of_nonneg_right (measureReal_mono Set.inter_subset_right) hpos.le

lemma minorityShare_le_off (A : DFA S Q) {H : DFA S R} {Dsamp : Measure S} {h : R}
    (hne : Dsamp {w | H.state w = h} ≠ 0) :
    minorityShare A H Dsamp h ≤
      (reaching H Dsamp h).real {p | A.state p ∈ A.accept ↔ H.state p ∉ H.accept} := by
  have key : ∀ X Y : Set S, {w | H.state w = h} ∩ X = {w | H.state w = h} ∩ Y →
      (reaching H Dsamp h).real X = (reaching H Dsamp h).real Y := by
    intro X Y hXY
    rw [reaching_eq_cond hne, measureReal_def, measureReal_def,
      cond_apply (measurableSet_stringlike _), cond_apply (measurableSet_stringlike _), hXY]
  unfold minorityShare
  by_cases hh : h ∈ H.accept
  · refine (min_le_right _ _).trans (key _ _ ?_).le
    ext p
    by_cases hp : H.state p = h <;> simp [hp, hh]
  · refine (min_le_left _ _).trans (key _ _ ?_).le
    ext p
    by_cases hp : H.state p = h <;> simp [hp, hh]

/-- A failing state's population puts at least `(wₛ − ζ·|R|/ε)/|Q|` on some state the family
cuts badly. -/
lemma exists_heavy_badState [Fintype Q] [Fintype R] (A : DFA S Q) (O : Oracle μ S)
    {Dsamp : Measure S} [IsProbabilityMeasure Dsamp] (H : DFA S R) (h : R) (B : State)
    (F : Finset S) {tolerance ε ζ wₛ : ℝ} (hε : 0 < ε) (hζ : 0 ≤ ζ)
    (hx : 0 < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q)
    (hmass : ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h})
    (hwell : mislabelledWellCut A O tolerance B F H Dsamp ≤ ζ)
    (hmin : wₛ ≤ minorityShare A H Dsamp h) :
    ∃ q, (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q ≤ stateMass A (reaching H Dsamp h) q ∧
      ∃ p, A.state p = q ∧ (tolerance < miscutProb O B.lo B.hi F p
        ∨ tolerance < undecidedProb O B.lo B.hi F p) := by
  classical
  have : Nonempty Q := ⟨A.start⟩
  have hR : 0 < (Fintype.card R : ℝ) := Nat.cast_pos.2 (Fintype.card_pos_iff.2 ⟨h⟩)
  have hQ : 0 < (Fintype.card Q : ℝ) := Nat.cast_pos.2 Fintype.card_pos
  have hm : 0 < Dsamp.real {v | H.state v = h} := (div_pos hε hR).trans_le hmass
  have hne : Dsamp {v | H.state v = h} ≠ 0 := fun h0 => by simp [measureReal_def, h0] at hm
  have : IsProbabilityMeasure (reaching H Dsamp h) := by
    rw [reaching_eq_cond hne]; exact cond_isProbabilityMeasure hne
  set ν := reaching H Dsamp h
  set x := wₛ - ζ * Fintype.card R / ε
  let M : Set S := {p | (A.state p ∈ A.accept ↔ H.state p ∉ H.accept)
    ∧ miscutProb O B.lo B.hi F p ≤ tolerance ∧ undecidedProb O B.lo B.hi F p ≤ tolerance}
  let T : Set S := {p | (A.state p ∈ A.accept ↔ H.state p ∉ H.accept)
    ∧ ¬ (miscutProb O B.lo B.hi F p ≤ tolerance ∧ undecidedProb O B.lo B.hi F p ≤ tolerance)}
  have hM : ν.real M ≤ ζ * Fintype.card R / ε :=
    calc ν.real M ≤ Dsamp.real M / Dsamp.real {v | H.state v = h} := reaching_real_le hm M
      _ ≤ ζ / (ε / Fintype.card R) := div_le_div₀ hζ hwell (div_pos hε hR) hmass
      _ = ζ * Fintype.card R / ε := by field_simp
  have hT : x ≤ ν.real T := by
    have hoff : ν.real {p | A.state p ∈ A.accept ↔ H.state p ∉ H.accept} ≤ ν.real M + ν.real T :=
      (measureReal_mono fun p (hp : A.state p ∈ A.accept ↔ H.state p ∉ H.accept) => by
        by_cases hw : miscutProb O B.lo B.hi F p ≤ tolerance
            ∧ undecidedProb O B.lo B.hi F p ≤ tolerance
        exacts [Or.inl ⟨hp, hw⟩, Or.inr ⟨hp, hw⟩]).trans (measureReal_union_le _ _)
    have := minorityShare_le_off A hne
    linarith
  have hsum : ν.real T ≤ ∑ q, ν.real (T ∩ {p | A.state p = q}) :=
    calc ν.real T ≤ ν.real (⋃ q ∈ (Finset.univ : Finset Q), T ∩ {p | A.state p = q}) :=
          measureReal_mono (fun p hp =>
            Set.mem_biUnion (Finset.mem_coe.2 (Finset.mem_univ (A.state p))) ⟨hp, rfl⟩)
            (measure_ne_top _ _)
      _ ≤ _ := measureReal_biUnion_finset_le _ _
  obtain ⟨q, -, hq⟩ := Finset.exists_le_of_sum_le Finset.univ_nonempty
    (f := fun _ => x / Fintype.card Q) (g := fun q => ν.real (T ∩ {p | A.state p = q})) (by
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
      rw [mul_div_cancel₀ _ hQ.ne']
      exact hT.trans hsum)
  obtain ⟨p, hpT, hpq⟩ : (T ∩ {p | A.state p = q}).Nonempty := by
    by_contra he
    rw [Set.not_nonempty_iff_eq_empty.1 he, measureReal_empty] at hq
    exact absurd hq (not_le.2 hx)
  refine ⟨q, hq.trans (measureReal_mono Set.inter_subset_right), p, hpq, ?_⟩
  exact (not_and_or.1 hpT.2).imp not_le.1 not_le.1

/-- `Termination`, with the specs asked of the first `2·|Q| + 1` rounds only. -/
theorem termination_le [Fintype Q] [Fintype R] {J Θ : Type*} [MeasurableSpace Θ]
    (P : Measure Θ) [IsProbabilityMeasure P] (A : DFA S Q) (O : Oracle μ S) (Dsamp : Measure S)
    (populations : Finset J) (D : J → Measure S) (family : ℕ → Θ → State × Finset S)
    (hyp : ℕ → Θ → DFA S R) (fails : ℕ → Θ → Finset R)
    {tolerance εcov indecisionLimit ε ζ wₛ δc δs δa : ℝ}
    (hD : IsProbabilityMeasure Dsamp) (hε : 0 < ε) (hζ : 0 ≤ ζ) (htol : 0 < tolerance)
    (hεcov : 0 ≤ εcov)
    (hcC : 2 * εcov / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q)
    (hcU : 4 * indecisionLimit / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q)
    (hC : ∀ r < 2 * Fintype.card Q + 1, P.real {θ | ∃ q,
      ((∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
          2 * εcov / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < miscutProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
      ∨ ((∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
          4 * indecisionLimit / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)}
      ≤ δc)
    (hS : ∀ r < 2 * Fintype.card Q + 1, P.real {θ | ζ < mislabelledWellCut A O tolerance
      (family r θ).1 (family r θ).2 (hyp r θ) Dsamp} ≤ δs)
    (hA : ∀ r < 2 * Fintype.card Q + 1,
      P.real {θ | ∃ h ∈ fails r θ, minorityShare A (hyp r θ) Dsamp h < wₛ} ≤ δa)
    (hmass : ∀ r θ, ∀ h ∈ fails r θ, ε / Fintype.card R ≤ Dsamp.real {v | (hyp r θ).state v = h}) :
    P.real {θ | ∀ r < 2 * Fintype.card Q + 1, (fails r θ).Nonempty}
      ≤ (2 * Fintype.card Q + 1) * (δc + δs + δa) := by
  classical
  set x := (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q
  have hx : 0 < x := lt_of_le_of_lt (div_nonneg (by linarith) htol.le) hcC
  let Cb : ℕ → Set Θ := fun r => {θ | ∃ q,
      ((∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
          2 * εcov / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < miscutProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
      ∨ ((∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
          4 * indecisionLimit / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)}
  let Sb : ℕ → Set Θ := fun r => {θ | ζ < mislabelledWellCut A O tolerance (family r θ).1
      (family r θ).2 (hyp r θ) Dsamp}
  let Ab : ℕ → Set Θ := fun r => {θ | ∃ h ∈ fails r θ, minorityShare A (hyp r θ) Dsamp h < wₛ}
  have hsub : {θ | ∀ r < 2 * Fintype.card Q + 1, (fails r θ).Nonempty}
      ⊆ ⋃ r ∈ Finset.range (2 * Fintype.card Q + 1), (Cb r ∪ Sb r ∪ Ab r) := by
    intro θ hall
    replace hall : ∀ r < 2 * Fintype.card Q + 1, (fails r θ).Nonempty := hall
    by_contra hnot
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists] at hnot
    have hgood : ∀ r < 2 * Fintype.card Q + 1, θ ∉ Cb r ∧ θ ∉ Sb r ∧ θ ∉ Ab r := fun r hr =>
      ⟨fun h => hnot r hr (Or.inl (Or.inl h)), fun h => hnot r hr (Or.inl (Or.inr h)),
        fun h => hnot r hr (Or.inr h)⟩
    let covC : ℕ → Finset Q := fun r => Finset.univ.filter fun q =>
      ∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ, 2 * εcov / tolerance < stateMass A D' q
    let covU : ℕ → Finset Q := fun r => Finset.univ.filter fun q =>
      ∃ D' ∈ poolsAt populations D Dsamp hyp fails r θ,
        4 * indecisionLimit / tolerance < stateMass A D' q
    have hpools : ∀ r, poolsAt populations D Dsamp hyp fails r θ
        ⊆ poolsAt populations D Dsamp hyp fails (r + 1) θ := by
      rintro r D' (h | ⟨i, hi, h'⟩)
      exacts [Or.inl h, Or.inr ⟨i, Nat.lt_succ_of_lt hi, h'⟩]
    have hmonoC : ∀ r, covC r ⊆ covC (r + 1) := fun r q hq => by
      obtain ⟨D', hD', hm⟩ := (Finset.mem_filter.1 hq).2
      exact Finset.mem_filter.2 ⟨Finset.mem_univ _, D', hpools r hD', hm⟩
    have hmonoU : ∀ r, covU r ⊆ covU (r + 1) := fun r q hq => by
      obtain ⟨D', hD', hm⟩ := (Finset.mem_filter.1 hq).2
      exact Finset.mem_filter.2 ⟨Finset.mem_univ _, D', hpools r hD', hm⟩
    have hgrow : ∀ r ≤ 2 * Fintype.card Q + 1, r ≤ (covC r).card + (covU r).card := by
      intro r
      induction r with
      | zero => intro _; exact Nat.zero_le _
      | succ r ih =>
        intro hr
        have hr' : r < 2 * Fintype.card Q + 1 := hr
        have ih := ih hr'.le
        obtain ⟨hCr, hSr, hAr⟩ := hgood r hr'
        obtain ⟨h, hh⟩ := hall r hr'
        have hwell : mislabelledWellCut A O tolerance (family r θ).1 (family r θ).2 (hyp r θ)
            Dsamp ≤ ζ := not_lt.1 hSr
        have hmin : wₛ ≤ minorityShare A (hyp r θ) Dsamp h := not_lt.1 fun hlt => hAr ⟨h, hh, hlt⟩
        obtain ⟨q, hq, p, hpq, hbad⟩ := exists_heavy_badState A O (hyp r θ) h (family r θ).1
          (family r θ).2 hε hζ hx (hmass r θ h hh) hwell hmin
        have hnew : reaching (hyp r θ) Dsamp h ∈ poolsAt populations D Dsamp hyp fails (r + 1) θ :=
          Or.inr ⟨r, Nat.lt_succ_self r, h, hh, rfl⟩
        rcases hbad with hbad | hbad
        · have hout : q ∉ covC r := fun hin =>
            hCr ⟨q, Or.inl ⟨(Finset.mem_filter.1 hin).2, p, hpq, hbad⟩⟩
          have hin : q ∈ covC (r + 1) :=
            Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hnew, hcC.trans_le hq⟩
          have h1 := Finset.card_lt_card
            ((Finset.ssubset_iff_of_subset (hmonoC r)).2 ⟨q, hin, hout⟩)
          have h2 := Finset.card_le_card (hmonoU r)
          omega
        · have hout : q ∉ covU r := fun hin =>
            hCr ⟨q, Or.inr ⟨(Finset.mem_filter.1 hin).2, p, hpq, hbad⟩⟩
          have hin : q ∈ covU (r + 1) :=
            Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hnew, hcU.trans_le hq⟩
          have h1 := Finset.card_lt_card
            ((Finset.ssubset_iff_of_subset (hmonoU r)).2 ⟨q, hin, hout⟩)
          have h2 := Finset.card_le_card (hmonoC r)
          omega
    have h1 := hgrow _ le_rfl
    have h2 := Finset.card_le_univ (covC (2 * Fintype.card Q + 1))
    have h3 := Finset.card_le_univ (covU (2 * Fintype.card Q + 1))
    omega
  calc P.real {θ | ∀ r < 2 * Fintype.card Q + 1, (fails r θ).Nonempty}
      ≤ P.real (⋃ r ∈ Finset.range (2 * Fintype.card Q + 1), (Cb r ∪ Sb r ∪ Ab r)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (2 * Fintype.card Q + 1), P.real (Cb r ∪ Sb r ∪ Ab r) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (2 * Fintype.card Q + 1), (δc + δs + δa) := by
        refine Finset.sum_le_sum fun r hr => ?_
        have hr := Finset.mem_range.1 hr
        have h1 := measureReal_union_le (μ := P) (Cb r ∪ Sb r) (Ab r)
        have h2 := measureReal_union_le (μ := P) (Cb r) (Sb r)
        have h3 : P.real (Cb r) ≤ δc := hC r hr
        have h4 : P.real (Sb r) ≤ δs := hS r hr
        have h5 : P.real (Ab r) ≤ δa := hA r hr
        linarith
    _ = (2 * Fintype.card Q + 1) * (δc + δs + δa) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

theorem termination_holds : Termination := by
  intro Ω _ μ _ S _ Q R _ _ J Θ _ P _ A O Dsamp populations D family hyp fails tolerance εcov
    indecisionLimit ε ζ wₛ δc δs δa hD hε hζ htol hεcov hcC hcU hC hS hA hmass
  exact termination_le P A O Dsamp populations D family hyp fails hD hε hζ htol hεcov hcC hcU
    (fun r _ => hC r) (fun r _ => hS r) (fun r _ => hA r) hmass

end OrthoDFA
