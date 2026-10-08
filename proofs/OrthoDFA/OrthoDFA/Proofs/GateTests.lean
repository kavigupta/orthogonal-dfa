import OrthoDFA.Proofs.BinomLaw

/-!
# The gate's tests

The gate stops where its agreement's test settles and reads the batch's share there, so it errs
at a look only where the test settles on either side with the share on the wrong one. A count at
or past the mean is likelier under the rate at the mean than under any rate short of it, so that
too costs at most the failure chance.

The pair test counts pairs among the searched draws read. Which draws are read depends only on
which agree, and a searched draw never agrees, so given which draws were searched their pairs are
independent, each with chance `D(pair)/D(searched)`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem binom_term_le {n h : ℕ} {p θ : ℝ} (hp0 : 0 ≤ p) (hpθ : p ≤ θ) (hθ1 : θ ≤ 1)
    (hh : θ * n ≤ h) (hhn : h ≤ n) :
    p ^ h * (1 - p) ^ (n - h) ≤ θ ^ h * (1 - θ) ^ (n - h) := by
  rcases eq_or_lt_of_le hθ1 with rfl | hθ1'
  · have : h = n := le_antisymm hhn (by exact_mod_cast (by simpa using hh : (n : ℝ) ≤ h))
    subst this
    simpa using pow_le_one₀ hp0 hpθ
  rcases eq_or_lt_of_le (hp0.trans hpθ) with hθ0 | hθ0
  · obtain rfl : p = 0 := le_antisymm (hθ0 ▸ hpθ) hp0
    rw [← hθ0]
  set A := p / θ with hAdef
  set B := (1 - p) / (1 - θ) with hBdef
  have hA : 0 ≤ A := div_nonneg hp0 hθ0.le
  have hB : 0 ≤ B := div_nonneg (by linarith) (by linarith)
  have hpA : p ^ h = θ ^ h * A ^ h := by
    rw [← mul_pow, hAdef, mul_div_cancel₀ _ hθ0.ne']
  have hpB : (1 - p) ^ (n - h) = (1 - θ) ^ (n - h) * B ^ (n - h) := by
    rw [← mul_pow, hBdef, mul_div_cancel₀ _ (by linarith : (1 - θ) ≠ 0)]
  have hexp : ∀ (x : ℝ) (m : ℕ), 0 ≤ x → x ^ m ≤ Real.exp (m * (x - 1)) := fun x m hx =>
    calc x ^ m ≤ Real.exp (x - 1) ^ m :=
          pow_le_pow_left₀ hx (by linarith [Real.add_one_le_exp (x - 1)]) m
      _ = Real.exp (m * (x - 1)) := by rw [← Real.exp_nat_mul]
  have hsum : (h : ℝ) * (A - 1) + ((n - h : ℕ) : ℝ) * (B - 1)
      = (θ - p) * (θ * n - h) / (θ * (1 - θ)) := by
    rw [Nat.cast_sub hhn, hAdef, hBdef]
    field_simp
    ring
  have key : A ^ h * B ^ (n - h) ≤ 1 := by
    calc A ^ h * B ^ (n - h)
        ≤ Real.exp (h * (A - 1)) * Real.exp ((n - h : ℕ) * (B - 1)) :=
          mul_le_mul (hexp A h hA) (hexp B _ hB) (by positivity) (by positivity)
      _ = Real.exp ((h : ℝ) * (A - 1) + ((n - h : ℕ) : ℝ) * (B - 1)) := (Real.exp_add _ _).symm
      _ ≤ Real.exp 0 := by
          rw [hsum]
          refine Real.exp_le_exp.2 (div_nonpos_of_nonpos_of_nonneg ?_ ?_)
          · exact mul_nonpos_of_nonneg_of_nonpos (by linarith) (by linarith)
          · exact mul_nonneg hθ0.le (by linarith)
      _ = 1 := Real.exp_zero
  rw [hpA, hpB]
  have : 0 ≤ θ ^ h * (1 - θ) ^ (n - h) := by
    have : 0 ≤ 1 - θ := by linarith
    positivity
  nlinarith

theorem binom_term_le' {n h : ℕ} {p θ : ℝ} (hθ0 : 0 ≤ θ) (hθp : θ ≤ p) (hp1 : p ≤ 1)
    (hh : (h : ℝ) ≤ θ * n) (hhn : h ≤ n) :
    p ^ h * (1 - p) ^ (n - h) ≤ θ ^ h * (1 - θ) ^ (n - h) := by
  have := binom_term_le (n := n) (h := n - h) (p := 1 - p) (θ := 1 - θ) (by linarith)
    (by linarith) (by linarith) (by rw [Nat.cast_sub hhn]; nlinarith) (Nat.sub_le n h)
  rw [Nat.sub_sub_self hhn, sub_sub_cancel, sub_sub_cancel] at this
  linarith [mul_comm (p ^ h) ((1 - p) ^ (n - h)), mul_comm (θ ^ h) ((1 - θ) ^ (n - h))]

/-- The binomial term at `h`. -/
noncomputable def binomTerm (n : ℕ) (p : ℝ) (h : ℕ) : ℝ :=
  (n.choose h : ℝ) * p ^ h * (1 - p) ^ (n - h)

theorem binomSfGe_sub (n : ℕ) (p : ℝ) (h : ℕ) :
    binomSfGe n p h - binomSfGe n p (h + 1) = binomTerm n p h := by
  unfold binomSfGe binomTerm
  rcases Nat.lt_or_ge n h with hnh | hhn
  · rw [Nat.choose_eq_zero_of_lt hnh, Finset.Icc_eq_empty (by omega),
      Finset.Icc_eq_empty (by omega)]
    simp
  · rw [← Finset.insert_Icc_add_one_left_eq_Icc hhn, Finset.sum_insert (by simp)]
    ring

theorem binomTerm_nonneg {n : ℕ} {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (h : ℕ) :
    0 ≤ binomTerm n p h := by
  unfold binomTerm
  have : 0 ≤ 1 - p := by linarith
  positivity

theorem sum_binomTerm_range (n : ℕ) (p : ℝ) (M : ℕ) :
    ∑ h ∈ Finset.range M, binomTerm n p h = 1 - binomSfGe n p M := by
  induction M with
  | zero => simp [binomSfGe_zero_right]
  | succ M ih => rw [Finset.sum_range_succ, ih, ← binomSfGe_sub]; ring

theorem sum_binomTerm_Ico (n : ℕ) (p : ℝ) (m : ℕ) :
    ∑ h ∈ Finset.Ico m (n + 1), binomTerm n p h = binomSfGe n p m := by
  rcases Nat.lt_or_ge (n + 1) m with hm | hm
  · rw [Finset.Ico_eq_empty (by omega)]
    unfold binomSfGe
    rw [Finset.Icc_eq_empty (by omega)]
    simp
  · rw [Finset.sum_Ico_eq_sub _ hm, sum_binomTerm_range, sum_binomTerm_range]
    have : binomSfGe n p (n + 1) = 0 := by
      unfold binomSfGe; rw [Finset.Icc_eq_empty (by omega)]; simp
    linarith

section Law

variable {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]

open scoped Classical in
/-- At one look of an exact binomial test against `θ`, a rate of at most `θ` reads above it with
chance at most the test's failure chance. -/
theorem look_above_le (D : Measure X) [IsProbabilityMeasure D]
    (P : X → Prop) {N : ℕ} (S : Finset (Fin N)) {θ a : ℝ} (hθ1 : θ ≤ 1)
    (hp : D.real {x | P x} ≤ θ) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin N => D).real
      {b | binomSfGe S.card θ (S.filter fun i => P (b i)).card < a} ≤ a := by
  by_cases hex : ∃ j, binomSfGe S.card θ j < a
  · refine le_trans (measureReal_mono (s₂ := {b | Nat.find hex ≤ (S.filter fun i => P (b i)).card})
      fun b hb => Nat.find_min' hex (show binomSfGe S.card θ (S.filter fun i => P (b i)).card < a
        from hb)) ?_
    rw [pi_count_ge]
    exact ((binomSfGe_mono measureReal_nonneg hθ1 hp _ _).trans_lt (Nat.find_spec hex)).le
  · push Not at hex
    rw [show {b : Fin N → X | binomSfGe S.card θ (S.filter fun i => P (b i)).card < a}
      = ∅ from Set.eq_empty_of_forall_notMem fun b hb => (not_lt.2 (hex _)) hb]
    simpa using ha

open scoped Classical in
/-- At one look, a rate of at least `θ` reads below it with chance at most the failure chance. -/
theorem look_below_le (D : Measure X) [IsProbabilityMeasure D]
    (P : X → Prop) {N : ℕ} (S : Finset (Fin N)) {θ a : ℝ} (hθ0 : 0 ≤ θ)
    (hp : θ ≤ D.real {x | P x}) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin N => D).real
      {b | 1 - binomSfGe S.card θ ((S.filter fun i => P (b i)).card + 1) < a} ≤ a := by
  set ν := Measure.pi fun _ : Fin N => D
  set c : (Fin N → X) → ℕ := fun b => (S.filter fun i => P (b i)).card
  set T := (Finset.range (S.card + 1)).filter fun h => 1 - binomSfGe S.card θ (h + 1) < a
  have hcle : ∀ b, c b ≤ S.card := fun b => Finset.card_filter_le _ _
  by_cases hT : T.Nonempty
  · have hmax := Finset.mem_filter.1 (T.max'_mem hT)
    have hsub : {b | 1 - binomSfGe S.card θ (c b + 1) < a} ⊆ {b | T.max' hT + 1 ≤ c b}ᶜ := by
      intro b hb
      have : c b ∈ T := Finset.mem_filter.2 ⟨Finset.mem_range.2 (by have := hcle b; omega), hb⟩
      have := T.le_max' _ this
      simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, not_le]
      omega
    refine le_trans (measureReal_mono hsub) ?_
    rw [measureReal_compl (Set.to_countable _).measurableSet, probReal_univ, pi_count_ge]
    have := binomSfGe_mono hθ0 measureReal_le_one hp S.card (T.max' hT + 1)
    linarith [hmax.2]
  · rw [Finset.not_nonempty_iff_eq_empty] at hT
    rw [show {b | 1 - binomSfGe S.card θ (c b + 1) < a} = ∅ from
      Set.eq_empty_of_forall_notMem fun b hb => by
        have : c b ∈ T := Finset.mem_filter.2 ⟨Finset.mem_range.2 (by have := hcle b; omega), hb⟩
        simp [hT] at this]
    simpa using ha

open scoped Classical in
theorem pi_count_eq (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) {N : ℕ}
    (S : Finset (Fin N)) (h : ℕ) :
    (Measure.pi fun _ : Fin N => D).real {b | (S.filter fun i => P (b i)).card = h}
      = binomTerm S.card (D.real {x | P x}) h := by
  have hset : {b : Fin N → X | (S.filter fun i => P (b i)).card = h}
      = {b | h ≤ (S.filter fun i => P (b i)).card}
        \ {b | h + 1 ≤ (S.filter fun i => P (b i)).card} := by
    ext b; simp only [Set.mem_ofPred_eq, Set.mem_sdiff]; omega
  rw [hset, measureReal_sdiff (fun b hb => by simp only [Set.mem_ofPred_eq] at hb ⊢; omega)
    (Set.to_countable _).measurableSet, pi_count_ge, pi_count_ge, binomSfGe_sub]

open scoped Classical in
theorem pi_count_mem (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) {N : ℕ}
    (S : Finset (Fin N)) (H : Finset ℕ) :
    (Measure.pi fun _ : Fin N => D).real {b | (S.filter fun i => P (b i)).card ∈ H}
      = ∑ h ∈ H, binomTerm S.card (D.real {x | P x}) h := by
  have hset : {b : Fin N → X | (S.filter fun i => P (b i)).card ∈ H}
      = ⋃ h ∈ H, {b | (S.filter fun i => P (b i)).card = h} := by
    ext b; simp
  rw [hset, measureReal_biUnion_finset (fun h _ h' _ hh' => Set.disjoint_left.2
    fun b hb hb' => hh' (hb.symm.trans hb')) (fun _ _ => (Set.to_countable _).measurableSet)]
  exact Finset.sum_congr rfl fun h _ => pi_count_eq D P S h

open scoped Classical in
/-- At one look, a rate of at most `θ` settles, either way, with the share at or past `θ` with
chance at most twice the failure chance. -/
theorem look_pass_le (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) {N : ℕ}
    (S : Finset (Fin N)) {θ a : ℝ} (hθ1 : θ ≤ 1) (hp : D.real {x | P x} ≤ θ) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin N => D).real {b | θ * S.card ≤ (S.filter fun i => P (b i)).card
      ∧ (binomSfGe S.card θ (S.filter fun i => P (b i)).card < a
        ∨ 1 - binomSfGe S.card θ ((S.filter fun i => P (b i)).card + 1) < a)} ≤ 2 * a := by
  set ν := Measure.pi fun _ : Fin N => D
  set c : (Fin N → X) → ℕ := fun b => (S.filter fun i => P (b i)).card
  set p := D.real {x | P x}
  have hp0 : 0 ≤ p := measureReal_nonneg
  set H := (Finset.range (S.card + 1)).filter
    fun h : ℕ => θ * S.card ≤ h ∧ 1 - binomSfGe S.card θ (h + 1) < a
  have hcle : ∀ b, c b ≤ S.card := fun b => Finset.card_filter_le _ _
  have hsub : {b | θ * S.card ≤ c b ∧ (binomSfGe S.card θ (c b) < a
      ∨ 1 - binomSfGe S.card θ (c b + 1) < a)}
      ⊆ {b | binomSfGe S.card θ (c b) < a} ∪ {b | c b ∈ H} := by
    rintro b ⟨h1, h2 | h2⟩
    · exact .inl h2
    · exact .inr (Finset.mem_filter.2 ⟨Finset.mem_range.2 (by have := hcle b; omega), h1, h2⟩)
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have h1 := look_above_le D P S hθ1 hp ha
  have h2 : ν.real {b | c b ∈ H} ≤ a := by
    rw [pi_count_mem]
    by_cases hH : H.Nonempty
    · have hmax := Finset.mem_filter.1 (H.max'_mem hH)
      calc ∑ h ∈ H, binomTerm S.card p h
          ≤ ∑ h ∈ H, binomTerm S.card θ h := Finset.sum_le_sum fun h hh => by
            have hh' := Finset.mem_filter.1 hh
            unfold binomTerm
            rw [mul_assoc, mul_assoc]
            exact mul_le_mul_of_nonneg_left (binom_term_le hp0 hp hθ1 hh'.2.1
              (by have := Finset.mem_range.1 hh'.1; omega)) (by positivity)
        _ ≤ ∑ h ∈ Finset.range (H.max' hH + 1), binomTerm S.card θ h := by
            refine Finset.sum_le_sum_of_subset_of_nonneg (fun h hh => Finset.mem_range.2
              (Nat.lt_succ_of_le (H.le_max' h hh))) fun h _ _ => ?_
            exact binomTerm_nonneg (hp0.trans hp) hθ1 h
        _ = 1 - binomSfGe S.card θ (H.max' hH + 1) := sum_binomTerm_range _ _ _
        _ ≤ a := hmax.2.2.le
    · rw [Finset.not_nonempty_iff_eq_empty.1 hH, Finset.sum_empty]
      exact ha
  linarith

open scoped Classical in
/-- And a rate of at least `θ` settles with the share short of `θ` with chance at most twice the
failure chance. -/
theorem look_refuse_le (D : Measure X) [IsProbabilityMeasure D] (P : X → Prop) {N : ℕ}
    (S : Finset (Fin N)) {θ a : ℝ} (hθ0 : 0 ≤ θ) (hp : θ ≤ D.real {x | P x}) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin N => D).real {b | ((S.filter fun i => P (b i)).card : ℝ) < θ * S.card
      ∧ (binomSfGe S.card θ (S.filter fun i => P (b i)).card < a
        ∨ 1 - binomSfGe S.card θ ((S.filter fun i => P (b i)).card + 1) < a)} ≤ 2 * a := by
  set ν := Measure.pi fun _ : Fin N => D
  set c : (Fin N → X) → ℕ := fun b => (S.filter fun i => P (b i)).card
  set p := D.real {x | P x}
  have hp1 : p ≤ 1 := measureReal_le_one
  set H := (Finset.range (S.card + 1)).filter
    fun h : ℕ => (h : ℝ) < θ * S.card ∧ binomSfGe S.card θ h < a
  have hcle : ∀ b, c b ≤ S.card := fun b => Finset.card_filter_le _ _
  have hsub : {b | (c b : ℝ) < θ * S.card ∧ (binomSfGe S.card θ (c b) < a
      ∨ 1 - binomSfGe S.card θ (c b + 1) < a)}
      ⊆ {b | 1 - binomSfGe S.card θ (c b + 1) < a} ∪ {b | c b ∈ H} := by
    rintro b ⟨h1, h2 | h2⟩
    · exact .inr (Finset.mem_filter.2 ⟨Finset.mem_range.2 (by have := hcle b; omega), h1, h2⟩)
    · exact .inl h2
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have h1 := look_below_le D P S hθ0 hp ha
  have h2 : ν.real {b | c b ∈ H} ≤ a := by
    rw [pi_count_mem]
    by_cases hH : H.Nonempty
    · have hmin := Finset.mem_filter.1 (H.min'_mem hH)
      calc ∑ h ∈ H, binomTerm S.card p h
          ≤ ∑ h ∈ H, binomTerm S.card θ h := Finset.sum_le_sum fun h hh => by
            have hh' := Finset.mem_filter.1 hh
            unfold binomTerm
            rw [mul_assoc, mul_assoc]
            exact mul_le_mul_of_nonneg_left (binom_term_le' hθ0 hp hp1 hh'.2.1.le
              (by have := Finset.mem_range.1 hh'.1; omega)) (by positivity)
        _ ≤ ∑ h ∈ Finset.Ico (H.min' hH) (S.card + 1), binomTerm S.card θ h := by
            refine Finset.sum_le_sum_of_subset_of_nonneg (fun h hh => Finset.mem_Ico.2
              ⟨H.min'_le h hh, Finset.mem_range.1 (Finset.mem_filter.1 hh).1⟩) fun h _ _ => ?_
            exact binomTerm_nonneg hθ0 (hp.trans hp1) h
        _ = binomSfGe S.card θ (H.min' hH) := sum_binomTerm_Ico _ _ _
        _ ≤ a := hmin.2.2.le
    · rw [Finset.not_nonempty_iff_eq_empty.1 hH, Finset.sum_empty]
      exact ha
  linarith

open scoped Classical in
/-- Over independent draws, every coordinate of `S` satisfying `B` and at least `j` of them `Q`,
`Q` inside `B`, has chance `D(B)^|S|` times the binomial tail at `D(Q)/D(B)`. -/
theorem pi_count_cond_ge (D : Measure X) [IsProbabilityMeasure D] (B Q : X → Prop)
    (hQB : ∀ x, Q x → B x) {N : ℕ} (S : Finset (Fin N)) (j : ℕ) :
    (Measure.pi fun _ : Fin N => D).real
      {b | (∀ i ∈ S, B (b i)) ∧ j ≤ (S.filter fun i => Q (b i)).card}
      = D.real {x | B x} ^ S.card
        * binomSfGe S.card (D.real {x | Q x} / D.real {x | B x}) j := by
  set ν := Measure.pi fun _ : Fin N => D
  set pB := D.real {x | B x}
  set pQ := D.real {x | Q x}
  have hmeas : ∀ s : Set (Fin N → X), MeasurableSet s := fun s => (Set.to_countable s).measurableSet
  have hmarg : ∀ (i : Fin N) (A : Set X), ν.real ((fun b => b i) ⁻¹' A) = D.real A := by
    intro i A
    rw [measureReal_def, measureReal_def, ← Measure.map_apply (measurable_pi_apply i)
      (Set.to_countable A).measurableSet, (measurePreserving_eval _ i).map_eq]
  have hind : iIndepFun (fun i (b : Fin N → X) => b i) ν :=
    iIndepFun_pi (X := fun _ (x : X) => x) fun _ => measurable_id.aemeasurable
  have hQle : pQ ≤ pB := measureReal_mono (fun x hx => hQB x hx)
  have hdiff : D.real {x | B x ∧ ¬ Q x} = pB - pQ := by
    rw [show {x | B x ∧ ¬ Q x} = {x | B x} \ {x | Q x} from rfl,
      measureReal_sdiff (s₁ := {x | B x}) (s₂ := {x | Q x}) (fun x hx => hQB x hx)
        (Set.to_countable _).measurableSet]
  induction S using Finset.induction_on generalizing j with
  | empty =>
    rcases j with _ | j
    · simp [binomSfGe_zero_right]
    · simp [binomSfGe_zero_left]
  | insert a S ha ih =>
    rw [Finset.card_insert_of_notMem ha]
    have hS : ∀ b : Fin N → X, ((insert a S).filter fun i => Q (b i)).card
        = (if Q (b a) then 1 else 0) + (S.filter fun i => Q (b i)).card := by
      intro b
      rw [Finset.filter_insert]
      split_ifs with h
      · rw [Finset.card_insert_of_notMem (fun hm => ha (Finset.mem_filter.1 hm).1)]
        ring
      · simp
    have h2 := hind.indepFun_finset {a} S (by simpa using ha) (fun i => measurable_pi_apply i)
    have h3 : IndepFun (fun b : Fin N → X => b a) (fun b (i : S) => b i) ν :=
      h2.comp (measurable_pi_apply (⟨a, Finset.mem_singleton_self a⟩ : ({a} : Finset _)))
        measurable_id
    have hcount : ∀ k, {b : Fin N → X | (∀ i ∈ S, B (b i)) ∧ k ≤ (S.filter fun i => Q (b i)).card}
        = (fun b (i : S) => b i) ⁻¹' {g | (∀ i : S, B (g i))
          ∧ k ≤ (Finset.univ.filter fun i : S => Q (g i)).card} := by
      intro k
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_preimage]
      rw [card_filter_subtype S (fun i => Q (b i))]
      simp
    have hprod : ∀ (A : Set X) k, ν.real ((fun b => b a) ⁻¹' A
        ∩ {b | (∀ i ∈ S, B (b i)) ∧ k ≤ (S.filter fun i => Q (b i)).card})
        = D.real A * (pB ^ S.card * binomSfGe S.card (pQ / pB) k) := by
      intro A k
      rw [hcount, measureReal_def, h3.measure_inter_preimage_eq_mul _ _
        (Set.to_countable A).measurableSet (Set.to_countable _).measurableSet,
        ENNReal.toReal_mul, ← measureReal_def, ← measureReal_def, hmarg, ← hcount, ih]
    rcases j with _ | j
    · have hset : {b : Fin N → X | (∀ i ∈ insert a S, B (b i))
          ∧ 0 ≤ ((insert a S).filter fun i => Q (b i)).card}
          = (fun b => b a) ⁻¹' {x | B x}
            ∩ {b | (∀ i ∈ S, B (b i)) ∧ 0 ≤ (S.filter fun i => Q (b i)).card} := by
        ext b; simp
      rw [hset, hprod, binomSfGe_zero_right, binomSfGe_zero_right]
      ring
    have hsplit : {b : Fin N → X | (∀ i ∈ insert a S, B (b i))
        ∧ j + 1 ≤ ((insert a S).filter fun i => Q (b i)).card}
        = ((fun b => b a) ⁻¹' {x | Q x}
            ∩ {b | (∀ i ∈ S, B (b i)) ∧ j ≤ (S.filter fun i => Q (b i)).card})
          ∪ ((fun b => b a) ⁻¹' {x | B x ∧ ¬ Q x}
            ∩ {b | (∀ i ∈ S, B (b i)) ∧ j + 1 ≤ (S.filter fun i => Q (b i)).card}) := by
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_union, Set.mem_inter_iff, Set.mem_preimage, hS,
        Finset.forall_mem_insert]
      by_cases h : Q (b a)
      · have e : ∀ c : ℕ, j + 1 ≤ 1 + c ↔ j ≤ c := fun c => by omega
        simp [h, hQB _ h, e]
      · simp [h, and_assoc]
    rw [hsplit, measureReal_union (Set.disjoint_left.2 fun b h1 h2 => h2.1.2 h1.1) (hmeas _),
      hprod, hprod, binomSfGe_succ, hdiff]
    rcases eq_or_lt_of_le (show 0 ≤ pB from measureReal_nonneg) with hB0 | hB0
    · have h0 : pB = 0 := hB0.symm
      have h1 : pQ = 0 := le_antisymm (h0 ▸ hQle) measureReal_nonneg
      have h1' : D.real {x | Q x} = 0 := h1
      simp only [h0, h1, h1', pow_succ, mul_zero, zero_mul, sub_self, add_zero]
    · have hq : pB * (pQ / pB) = pQ := mul_div_cancel₀ _ hB0.ne'
      linear_combination (-(pB ^ S.card * (binomSfGe S.card (pQ / pB) j
        - binomSfGe S.card (pQ / pB) (j + 1)))) * hq

open scoped Classical in
/-- The pair test: read up to a look `T` that does not see which draws satisfying `B` were drawn,
it counts `Q` among them. It reads more than `θ` with chance at most its failure chance where
`D(Q) ≤ θ·D(B)`. -/
theorem pair_test_le (D : Measure X) [IsProbabilityMeasure D] (B Q : X → Prop)
    (hQB : ∀ x, Q x → B x) {N : ℕ} (T : (Fin N → X) → ℕ)
    (hT : ∀ b b' : Fin N → X, (∀ i, B (b i) ↔ B (b' i)) → (∀ i, ¬ B (b i) → b' i = b i) →
      T b = T b') {θ a : ℝ} (hθ0 : 0 ≤ θ)
    (hθ1 : θ ≤ 1) (ha : 0 ≤ a) (h : D.real {x | Q x} ≤ θ * D.real {x | B x}) :
    (Measure.pi fun _ : Fin N => D).real
      {b | binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < T b ∧ B (b i)).card θ
        ((Finset.univ.filter fun i : Fin N => (i : ℕ) < T b ∧ B (b i)).filter
          fun i => Q (b i)).card < a} ≤ a := by
  set ν := Measure.pi fun _ : Fin N => D
  set Sh : (Fin N → X) → Finset (Fin N) := fun b =>
    Finset.univ.filter fun i : Fin N => (i : ℕ) < T b ∧ B (b i)
  set E := {b : Fin N → X | binomSfGe (Sh b).card θ ((Sh b).filter fun i => Q (b i)).card < a}
  have hmeas : ∀ s : Set (Fin N → X), MeasurableSet s := fun s => (Set.to_countable s).measurableSet
  rcases le_or_gt 1 a with ha1 | ha1
  · exact measureReal_le_one.trans ha1
  set pB := D.real {x | B x}
  set pQ := D.real {x | Q x}
  have hq0 : 0 ≤ pQ / pB := div_nonneg measureReal_nonneg measureReal_nonneg
  have hqθ : pQ / pB ≤ θ := by
    rcases eq_or_lt_of_le (show 0 ≤ pB from measureReal_nonneg) with hB0 | hB0
    · have h0 : pB = 0 := hB0.symm
      have : pQ = 0 := le_antisymm (by have := h; rw [h0, mul_zero] at this; exact this)
        measureReal_nonneg
      rw [h0, this, div_zero]
      exact hθ0
    · rwa [div_le_iff₀ hB0]
  by_cases hex : ∃ x, B x
  swap
  · push Not at hex
    rw [show E = ∅ from Set.eq_empty_of_forall_notMem fun b hb => by
      have : Sh b = ∅ := Finset.filter_eq_empty_iff.2 fun i _ hi => hex _ hi.2
      simp only [E, Set.mem_ofPred_eq, this, Finset.card_empty, Finset.filter_empty,
        binomSfGe_zero_right] at hb
      linarith]
    simpa using ha
  obtain ⟨xB, hxB⟩ := hex
  -- `repl S b` puts `xB` at the coordinates of `S`.
  set repl : Finset (Fin N) → (Fin N → X) → (Fin N → X) := fun S b i =>
    if i ∈ S then xB else b i
  have hShrepl : ∀ S b, (∀ i ∈ S, B (b i)) → Sh (repl S b) = Sh b := by
    intro S b hb
    have hTe : T (repl S b) = T b := (hT b (repl S b) (fun i => by
      simp only [repl]
      split_ifs with hi
      · exact ⟨fun _ => hxB, fun _ => hb i hi⟩
      · rfl) (fun i hi => by
      simp only [repl]
      split_ifs with hiS
      · exact absurd (hb i hiS) hi
      · rfl)).symm
    ext i
    simp only [Sh, Finset.mem_filter, Finset.mem_univ, true_and, hTe, repl]
    split_ifs with hi
    · simp [hxB, hb i hi]
    · rfl
  have hsplitS : ∀ S, {b | Sh b = S} = {b | ∀ i ∈ S, B (b i)} ∩ {b | Sh (repl S b) = S} := by
    intro S
    ext b
    simp only [Set.mem_ofPred_eq, Set.mem_inter_iff]
    constructor
    · intro hb
      have hB : ∀ i ∈ S, B (b i) := fun i hi => by
        rw [← hb] at hi; exact (Finset.mem_filter.1 hi).2.2
      exact ⟨hB, (hShrepl S b hB).trans hb⟩
    · rintro ⟨hB, hb⟩
      exact (hShrepl S b hB).symm.trans hb
  have hind : iIndepFun (fun i (b : Fin N → X) => b i) ν :=
    iIndepFun_pi (X := fun _ (x : X) => x) fun _ => measurable_id.aemeasurable
  have hfactor : ∀ S (k : ℕ), ν.real ({b | Sh b = S} ∩ {b | k ≤ (S.filter fun i => Q (b i)).card})
      = pB ^ S.card * binomSfGe S.card (pQ / pB) k * ν.real {b | Sh (repl S b) = S} := by
    intro S k
    have h2 := hind.indepFun_finset S Sᶜ disjoint_compl_right
      (fun i => measurable_pi_apply i)
    have hin : {b : Fin N → X | (∀ i ∈ S, B (b i)) ∧ k ≤ (S.filter fun i => Q (b i)).card}
        = (fun b (i : S) => b i) ⁻¹' {g | (∀ i : S, B (g i))
          ∧ k ≤ (Finset.univ.filter fun i : S => Q (g i)).card} := by
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_preimage]
      rw [card_filter_subtype S (fun i => Q (b i))]
      simp
    have hout : {b : Fin N → X | Sh (repl S b) = S}
        = (fun b (i : (Sᶜ : Finset (Fin N))) => b i) ⁻¹'
          {g | Sh (fun i => if h : i ∈ S then xB else g ⟨i, Finset.mem_compl.2 h⟩) = S} := by
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_preimage]
      congr! 3
    rw [hsplitS, Set.inter_comm, ← Set.inter_assoc,
      show {b | k ≤ (S.filter fun i => Q (b i)).card} ∩ {b : Fin N → X | ∀ i ∈ S, B (b i)}
        = {b | (∀ i ∈ S, B (b i)) ∧ k ≤ (S.filter fun i => Q (b i)).card} from by
          ext b; simp [and_comm],
      hin, hout, measureReal_def, h2.measure_inter_preimage_eq_mul _ _
        (Set.to_countable _).measurableSet (Set.to_countable _).measurableSet,
      ENNReal.toReal_mul, ← measureReal_def, ← measureReal_def, ← hin, ← hout,
      pi_count_cond_ge D B Q hQB]
  have hper : ∀ S, ν.real (E ∩ {b | Sh b = S}) ≤ a * ν.real {b | Sh b = S} := by
    intro S
    have hall := hfactor S 0
    simp only [zero_le, Set.ofPred_true, Set.inter_univ, binomSfGe_zero_right, mul_one] at hall
    by_cases hj : ∃ j, binomSfGe S.card θ j < a
    · have hsub : E ∩ {b | Sh b = S}
          ⊆ {b | Sh b = S} ∩ {b | Nat.find hj ≤ (S.filter fun i => Q (b i)).card} := by
        rintro b ⟨hb, hbS⟩
        refine ⟨hbS, Nat.find_min' hj ?_⟩
        simp only [E, Set.mem_ofPred_eq] at hb hbS
        rwa [hbS] at hb
      refine (measureReal_mono hsub).trans ?_
      rw [hfactor, hall]
      have hsf := (binomSfGe_mono hq0 hθ1 hqθ S.card (Nat.find hj)).trans
        (Nat.find_spec hj).le
      have : 0 ≤ pB ^ S.card * ν.real {b | Sh (repl S b) = S} :=
        mul_nonneg (pow_nonneg measureReal_nonneg _) measureReal_nonneg
      nlinarith
    · push Not at hj
      rw [show E ∩ {b | Sh b = S} = ∅ from Set.eq_empty_of_forall_notMem fun b hb => by
        obtain ⟨hb, hbS⟩ := hb
        simp only [E, Set.mem_ofPred_eq] at hb hbS
        rw [hbS] at hb
        exact (not_lt.2 (hj _)) hb]
      simp only [measureReal_empty]
      exact mul_nonneg ha measureReal_nonneg
  have hcover : E = ⋃ S ∈ (Finset.univ : Finset (Finset (Fin N))), E ∩ {b | Sh b = S} := by
    ext b; simp
  have hcover' : (Set.univ : Set (Fin N → X))
      = ⋃ S ∈ (Finset.univ : Finset (Finset (Fin N))), {b | Sh b = S} := by
    ext b; simp
  have hdisj : ∀ (F : Finset (Fin N) → Set (Fin N → X)), (∀ S, F S ⊆ {b | Sh b = S}) →
      Set.PairwiseDisjoint (↑(Finset.univ : Finset (Finset (Fin N)))) F :=
    fun F hF S _ S' _ hSS' => Set.disjoint_left.2 fun b h1 h2 =>
      hSS' ((hF S h1).symm.trans (hF S' h2))
  rw [hcover, measureReal_biUnion_finset (hdisj _ fun S => Set.inter_subset_right)
    fun _ _ => hmeas _]
  calc ∑ S, ν.real (E ∩ {b | Sh b = S}) ≤ ∑ S, a * ν.real {b | Sh b = S} :=
        Finset.sum_le_sum fun S _ => hper S
    _ = a * ν.real (⋃ S ∈ (Finset.univ : Finset (Finset (Fin N))), {b | Sh b = S}) := by
        rw [← Finset.mul_sum, measureReal_biUnion_finset (hdisj _ fun S => subset_rfl)
          fun _ _ => hmeas _]
    _ = a := by rw [← hcover', probReal_univ, mul_one]

end Law

end OrthoDFA
