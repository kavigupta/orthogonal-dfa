import OrthoDFA.Termination

/-!
# The learner stops

Count, for each round, the target states some population so far is heavy enough on to force
them cut well, once for each of `qualityBound`'s two thresholds.  The count never falls and is
at most `2·|Q|`.  On the event that every round meets its four specs, a failing state's own
population, or after a refusal some heavy state's, is heavy on a state the family cuts badly,
which the specs say no earlier population was; so each round that does not return raises the
count.
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


lemma le_reaching_real {H : DFA S R} {Dsamp : Measure S} [IsProbabilityMeasure Dsamp] {h : R}
    (hne : Dsamp {w | H.state w = h} ≠ 0) (t : Set S) :
    Dsamp.real ({w | H.state w = h} ∩ t) ≤ (reaching H Dsamp h).real t := by
  have e : (Dsamp[|{w | H.state w = h}]).real t
      = (Dsamp.real {w | H.state w = h})⁻¹ * Dsamp.real ({w | H.state w = h} ∩ t) := by
    rw [measureReal_def, cond_apply (measurableSet_stringlike _), ENNReal.toReal_mul,
      ENNReal.toReal_inv]
    rfl
  rw [reaching_eq_cond hne, e]
  have h0 : 0 < Dsamp.real {w | H.state w = h} :=
    ENNReal.toReal_pos hne (measure_ne_top _ _)
  exact le_mul_of_one_le_left measureReal_nonneg ((one_le_inv₀ h0).2 measureReal_le_one)

/-- When the family cuts at least `x₀` of the sampler badly, some state carrying `ε/|R|` of it
puts `(x₀ − ε)/(|R|·|Q|)` on a target state the family cuts badly. -/
lemma exists_heavy_badlyCut [Fintype Q] [Fintype R] (A : DFA S Q) (O : Oracle μ S)
    {Dsamp : Measure S} [IsProbabilityMeasure Dsamp] (H : DFA S R) (B : State) (F : Finset S)
    {tolerance ε x₀ : ℝ} (hε : 0 < ε)
    (hx : 0 < (x₀ - ε) / (Fintype.card R * Fintype.card Q))
    (hbad : x₀ ≤ badlyCut O tolerance B F Dsamp) :
    ∃ h, ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} ∧
      ∃ q, (x₀ - ε) / (Fintype.card R * Fintype.card Q) ≤ stateMass A (reaching H Dsamp h) q ∧
        ∃ p, A.state p = q ∧ (tolerance < miscutProb O B.lo B.hi F p
          ∨ tolerance < undecidedProb O B.lo B.hi F p) := by
  classical
  set c := (x₀ - ε) / (Fintype.card R * Fintype.card Q) with hc
  have hR0 : (Fintype.card R : ℝ) ≠ 0 := fun h0 => by simp [c, h0] at hx
  have hRpos : (0 : ℝ) < Fintype.card R := lt_of_le_of_ne (Nat.cast_nonneg _) (Ne.symm hR0)
  have hRQ : (0 : ℝ) < Fintype.card R * Fintype.card Q := by
    rcases (Nat.cast_nonneg (Fintype.card Q) : (0 : ℝ) ≤ _).lt_or_eq with h | h
    · positivity
    · simp [c, ← h] at hx
  have hQ0 : (Fintype.card Q : ℝ) ≠ 0 := (pos_of_mul_pos_right hRQ hRpos.le).ne'
  have hxε : 0 < x₀ - ε := (div_pos_iff_of_pos_right hRQ).1 hx
  set Bad : Set S := {p | tolerance < miscutProb O B.lo B.hi F p
    ∨ tolerance < undecidedProb O B.lo B.hi F p}
  set Hv := Finset.univ.filter fun h : R => ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h}
  set m : R × Q → ℝ := fun hq =>
    Dsamp.real (Bad ∩ {v | H.state v = hq.1} ∩ {v | A.state v = hq.2})
  have hcover : Bad ⊆ (⋃ hq ∈ Hv ×ˢ (Finset.univ : Finset Q),
        Bad ∩ {v | H.state v = hq.1} ∩ {v | A.state v = hq.2})
      ∪ ⋃ h ∈ Finset.univ.filter (fun h => h ∉ Hv), {v | H.state v = h} := by
    intro p hp
    by_cases hh : H.state p ∈ Hv
    · exact Or.inl (Set.mem_biUnion (x := (H.state p, A.state p))
        (Finset.mem_coe.2 (Finset.mem_product.2 ⟨hh, Finset.mem_univ _⟩)) ⟨⟨hp, rfl⟩, rfl⟩)
    · exact Or.inr (Set.mem_biUnion (x := H.state p)
        (Finset.mem_coe.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, hh⟩)) rfl)
  have hlight : ∑ h ∈ Finset.univ.filter (fun h => h ∉ Hv), Dsamp.real {v | H.state v = h}
      ≤ ε := by
    calc ∑ h ∈ Finset.univ.filter (fun h => h ∉ Hv), Dsamp.real {v | H.state v = h}
        ≤ ∑ _h ∈ Finset.univ.filter (fun h => h ∉ Hv), ε / Fintype.card R :=
          Finset.sum_le_sum fun h hh => by
            have := (Finset.mem_filter.1 hh).2
            simp only [Hv, Finset.mem_filter, Finset.mem_univ, true_and, not_le] at this
            exact this.le
      _ ≤ ∑ _h : R, ε / Fintype.card R :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            fun _ _ _ => by positivity
      _ = ε := by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; field_simp
  have hsum : x₀ - ε ≤ ∑ hq ∈ Hv ×ˢ (Finset.univ : Finset Q), m hq := by
    have h1 := (measureReal_mono (μ := Dsamp) hcover (measure_ne_top _ _)).trans
      ((measureReal_union_le _ _).trans (add_le_add (measureReal_biUnion_finset_le _ _)
        (measureReal_biUnion_finset_le _ _)))
    have h2 : x₀ ≤ Dsamp.real Bad := hbad
    linarith
  have hne : (Hv ×ˢ (Finset.univ : Finset Q)).Nonempty := by
    by_contra he
    rw [Finset.not_nonempty_iff_eq_empty.1 he, Finset.sum_empty] at hsum
    linarith
  obtain ⟨⟨h, q⟩, hhq, hm⟩ := Finset.exists_le_of_sum_le hne
    (f := fun _ => c) (g := m) (by
      refine le_trans ?_ hsum
      rw [Finset.sum_const, nsmul_eq_mul, Finset.card_product, Finset.card_univ]
      calc ((Hv.card * Fintype.card Q : ℕ) : ℝ) * c
          ≤ ((Fintype.card R * Fintype.card Q : ℕ) : ℝ) * c := by
            gcongr
            exact Finset.card_le_univ _
        _ = x₀ - ε := by push_cast; rw [hc]; field_simp)
  have hh : h ∈ Hv := (Finset.mem_product.1 hhq).1
  have hheavy : ε / Fintype.card R ≤ Dsamp.real {v | H.state v = h} := (Finset.mem_filter.1 hh).2
  have hne0 : Dsamp {v | H.state v = h} ≠ 0 := fun h0 => by
    rw [measureReal_def, h0, ENNReal.toReal_zero] at hheavy
    exact absurd hheavy (not_le.2 (div_pos hε hRpos))
  obtain ⟨p, ⟨⟨hpB, -⟩, hpq⟩⟩ : (Bad ∩ {v | H.state v = h} ∩ {v | A.state v = q}).Nonempty := by
    by_contra he
    have hm' : c ≤ 0 := by
      have := hm
      simp only [m] at this
      rwa [Set.not_nonempty_iff_eq_empty.1 he, measureReal_empty] at this
    linarith
  refine ⟨h, hheavy, q, hm.trans ?_, p, hpq, hpB⟩
  calc m (h, q) ≤ Dsamp.real ({v | H.state v = h} ∩ {v | A.state v = q}) :=
        measureReal_mono (fun v hv => ⟨hv.1.2, hv.2⟩) (measure_ne_top _ _)
    _ ≤ stateMass A (reaching H Dsamp h) q := le_reaching_real hne0 _

/-- `Termination`, with the specs asked of the first `2·|Q| + 1` rounds only. -/
theorem termination_le [Fintype Q] [Fintype R] {J Θ : Type*} [MeasurableSpace Θ]
    (P : Measure Θ) [IsProbabilityMeasure P] (A : DFA S Q) (O : Oracle μ S) (Dsamp : Measure S)
    (populations : Finset J) (D : J → Measure S) (family : ℕ → Θ → State × Finset S)
    (hyp : ℕ → Θ → DFA S R) (gate : ℕ → Θ → Prop) (fails : ℕ → Θ → Finset R)
    {tolerance εcov indecisionLimit ε ζ x₀ wₛ δc δs δg δa : ℝ}
    (hD : IsProbabilityMeasure Dsamp) (hε : 0 < ε) (hζ : 0 ≤ ζ) (htol : 0 < tolerance)
    (hεcov : 0 ≤ εcov)
    (hcC : 2 * εcov / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q)
    (hcU : 4 * indecisionLimit / tolerance < (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q)
    (hgC : 2 * εcov / tolerance < (x₀ - ε) / (Fintype.card R * Fintype.card Q))
    (hgU : 4 * indecisionLimit / tolerance < (x₀ - ε) / (Fintype.card R * Fintype.card Q))
    (hC : ∀ r < 2 * Fintype.card Q + 1, P.real {θ | ∃ q,
      ((∃ D' ∈ poolsAt populations D Dsamp ε hyp r θ,
          2 * εcov / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < miscutProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
      ∨ ((∃ D' ∈ poolsAt populations D Dsamp ε hyp r θ,
          4 * indecisionLimit / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)}
      ≤ δc)
    (hS : ∀ r < 2 * Fintype.card Q + 1, P.real {θ | gate r θ ∧ ζ < mislabelledWellCut A O
      tolerance (family r θ).1 (family r θ).2 (hyp r θ) Dsamp} ≤ δs)
    (hG : ∀ r < 2 * Fintype.card Q + 1, P.real {θ | ¬ gate r θ
      ∧ badlyCut O tolerance (family r θ).1 (family r θ).2 Dsamp < x₀} ≤ δg)
    (hA : ∀ r < 2 * Fintype.card Q + 1,
      P.real {θ | ∃ h ∈ fails r θ, minorityShare A (hyp r θ) Dsamp h < wₛ} ≤ δa)
    (hmass : ∀ r θ, ∀ h ∈ fails r θ, ε / Fintype.card R ≤ Dsamp.real {v | (hyp r θ).state v = h}) :
    P.real {θ | ∀ r < 2 * Fintype.card Q + 1, ¬ gate r θ ∨ (fails r θ).Nonempty}
      ≤ (2 * Fintype.card Q + 1) * (δc + δs + δg + δa) := by
  classical
  set x := (wₛ - ζ * Fintype.card R / ε) / Fintype.card Q
  have hx : 0 < x := lt_of_le_of_lt (div_nonneg (by linarith) htol.le) hcC
  have hx₀ : 0 < (x₀ - ε) / (Fintype.card R * Fintype.card Q) :=
    lt_of_le_of_lt (div_nonneg (by linarith) htol.le) hgC
  let Cb : ℕ → Set Θ := fun r => {θ | ∃ q,
      ((∃ D' ∈ poolsAt populations D Dsamp ε hyp r θ,
          2 * εcov / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < miscutProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)
      ∨ ((∃ D' ∈ poolsAt populations D Dsamp ε hyp r θ,
          4 * indecisionLimit / tolerance < stateMass A D' q)
        ∧ ∃ p, A.state p = q
          ∧ tolerance < undecidedProb O (family r θ).1.lo (family r θ).1.hi (family r θ).2 p)}
  let Sb : ℕ → Set Θ := fun r => {θ | gate r θ ∧ ζ < mislabelledWellCut A O tolerance
      (family r θ).1 (family r θ).2 (hyp r θ) Dsamp}
  let Gb : ℕ → Set Θ := fun r => {θ | ¬ gate r θ
      ∧ badlyCut O tolerance (family r θ).1 (family r θ).2 Dsamp < x₀}
  let Ab : ℕ → Set Θ := fun r => {θ | ∃ h ∈ fails r θ, minorityShare A (hyp r θ) Dsamp h < wₛ}
  have hsub : {θ | ∀ r < 2 * Fintype.card Q + 1, ¬ gate r θ ∨ (fails r θ).Nonempty}
      ⊆ ⋃ r ∈ Finset.range (2 * Fintype.card Q + 1), (Cb r ∪ Sb r ∪ Gb r ∪ Ab r) := by
    intro θ hall
    replace hall : ∀ r < 2 * Fintype.card Q + 1, ¬ gate r θ ∨ (fails r θ).Nonempty := hall
    by_contra hnot
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists] at hnot
    have hgood : ∀ r < 2 * Fintype.card Q + 1,
        θ ∉ Cb r ∧ θ ∉ Sb r ∧ θ ∉ Gb r ∧ θ ∉ Ab r := fun r hr =>
      ⟨fun h => hnot r hr (Or.inl (Or.inl (Or.inl h))),
        fun h => hnot r hr (Or.inl (Or.inl (Or.inr h))),
        fun h => hnot r hr (Or.inl (Or.inr h)), fun h => hnot r hr (Or.inr h)⟩
    let covC : ℕ → Finset Q := fun r => Finset.univ.filter fun q =>
      ∃ D' ∈ poolsAt populations D Dsamp ε hyp r θ, 2 * εcov / tolerance < stateMass A D' q
    let covU : ℕ → Finset Q := fun r => Finset.univ.filter fun q =>
      ∃ D' ∈ poolsAt populations D Dsamp ε hyp r θ,
        4 * indecisionLimit / tolerance < stateMass A D' q
    have hpools : ∀ r, poolsAt populations D Dsamp ε hyp r θ
        ⊆ poolsAt populations D Dsamp ε hyp (r + 1) θ := by
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
        obtain ⟨hCr, hSr, hGr, hAr⟩ := hgood r hr'
        -- A new pool at round `r + 1`, heavy on a state the family cut badly at round `r`.
        obtain ⟨D', hnew, q, hqC, hqU, p, hpq, hbad⟩ : ∃ D' ∈ poolsAt populations D Dsamp ε hyp
            (r + 1) θ, ∃ q, 2 * εcov / tolerance < stateMass A D' q
              ∧ 4 * indecisionLimit / tolerance < stateMass A D' q
              ∧ ∃ p, A.state p = q ∧ (tolerance < miscutProb O (family r θ).1.lo
                (family r θ).1.hi (family r θ).2 p ∨ tolerance < undecidedProb O
                (family r θ).1.lo (family r θ).1.hi (family r θ).2 p) := by
          by_cases hg : gate r θ
          · obtain ⟨h, hh⟩ := (hall r hr').resolve_left (not_not.2 hg)
            have hwell : mislabelledWellCut A O tolerance (family r θ).1 (family r θ).2
                (hyp r θ) Dsamp ≤ ζ := not_lt.1 fun hlt => hSr ⟨hg, hlt⟩
            have hmin : wₛ ≤ minorityShare A (hyp r θ) Dsamp h :=
              not_lt.1 fun hlt => hAr ⟨h, hh, hlt⟩
            obtain ⟨q, hq, p, hpq, hbad⟩ := exists_heavy_badState A O (hyp r θ) h
              (family r θ).1 (family r θ).2 hε hζ hx (hmass r θ h hh) hwell hmin
            exact ⟨_, Or.inr ⟨r, Nat.lt_succ_self r, h, hmass r θ h hh, rfl⟩, q,
              hcC.trans_le hq, hcU.trans_le hq, p, hpq, hbad⟩
          · have hbc : x₀ ≤ badlyCut O tolerance (family r θ).1 (family r θ).2 Dsamp :=
              not_lt.1 fun hlt => hGr ⟨hg, hlt⟩
            obtain ⟨h, hh, q, hq, p, hpq, hbad⟩ := exists_heavy_badlyCut A O (hyp r θ)
              (family r θ).1 (family r θ).2 hε hx₀ hbc
            exact ⟨_, Or.inr ⟨r, Nat.lt_succ_self r, h, hh, rfl⟩, q, hgC.trans_le hq,
              hgU.trans_le hq, p, hpq, hbad⟩
        rcases hbad with hbad | hbad
        · have hout : q ∉ covC r := fun hin =>
            hCr ⟨q, Or.inl ⟨(Finset.mem_filter.1 hin).2, p, hpq, hbad⟩⟩
          have hin : q ∈ covC (r + 1) := Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hnew, hqC⟩
          have h1 := Finset.card_lt_card
            ((Finset.ssubset_iff_of_subset (hmonoC r)).2 ⟨q, hin, hout⟩)
          have h2 := Finset.card_le_card (hmonoU r)
          omega
        · have hout : q ∉ covU r := fun hin =>
            hCr ⟨q, Or.inr ⟨(Finset.mem_filter.1 hin).2, p, hpq, hbad⟩⟩
          have hin : q ∈ covU (r + 1) := Finset.mem_filter.2 ⟨Finset.mem_univ _, _, hnew, hqU⟩
          have h1 := Finset.card_lt_card
            ((Finset.ssubset_iff_of_subset (hmonoU r)).2 ⟨q, hin, hout⟩)
          have h2 := Finset.card_le_card (hmonoC r)
          omega
    have h1 := hgrow _ le_rfl
    have h2 := Finset.card_le_univ (covC (2 * Fintype.card Q + 1))
    have h3 := Finset.card_le_univ (covU (2 * Fintype.card Q + 1))
    omega
  calc P.real {θ | ∀ r < 2 * Fintype.card Q + 1, ¬ gate r θ ∨ (fails r θ).Nonempty}
      ≤ P.real (⋃ r ∈ Finset.range (2 * Fintype.card Q + 1), (Cb r ∪ Sb r ∪ Gb r ∪ Ab r)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range (2 * Fintype.card Q + 1), P.real (Cb r ∪ Sb r ∪ Gb r ∪ Ab r) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range (2 * Fintype.card Q + 1), (δc + δs + δg + δa) := by
        refine Finset.sum_le_sum fun r hr => ?_
        have hr := Finset.mem_range.1 hr
        have h1 := measureReal_union_le (μ := P) (Cb r ∪ Sb r ∪ Gb r) (Ab r)
        have h2 := measureReal_union_le (μ := P) (Cb r ∪ Sb r) (Gb r)
        have h3 := measureReal_union_le (μ := P) (Cb r) (Sb r)
        have h4 : P.real (Cb r) ≤ δc := hC r hr
        have h5 : P.real (Sb r) ≤ δs := hS r hr
        have h6 : P.real (Gb r) ≤ δg := hG r hr
        have h7 : P.real (Ab r) ≤ δa := hA r hr
        linarith
    _ = (2 * Fintype.card Q + 1) * (δc + δs + δg + δa) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        push_cast
        ring

theorem termination_holds : Termination := by
  intro Ω _ μ _ S _ Q R _ _ J Θ _ P _ A O Dsamp populations D family hyp gate fails tolerance
    εcov indecisionLimit ε ζ x₀ wₛ δc δs δg δa hD hε hζ htol hεcov hcC hcU hgC hgU hC hS hG hA
    hmass
  exact termination_le P A O Dsamp populations D family hyp gate fails hD hε hζ htol hεcov hcC
    hcU hgC hgU (fun r _ => hC r) (fun r _ => hS r) (fun r _ => hG r) (fun r _ => hA r) hmass

end OrthoDFA
