import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.GateFlip
import OrthoDFA.Proofs.BinomLaw

/-!
# `RoundAtK`, `WalkYield` and `SourceSpread`

A decision on a batch against a fixed hypothesis is a Hoeffding tail over the batch.  A refusal's
probes reach the split test or stop at a string the cut cannot place, because the pass keeps its
edges learned: every edge's witness sits at its leaf and its extension at its target.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Batch

open scoped Classical in
/-- The share of a batch satisfying `P`. -/
noncomputable def share {n : ℕ} (b : Fin n → FreeMonoid α) (P : FreeMonoid α → Prop) : ℝ :=
  ((Finset.univ.filter fun i => P (b i)).card : ℝ) / n


omit [Fintype α] [DecidableEq α] in
open scoped Classical in
/-- The batch's count of `P`, as the sum of its indicators. -/
theorem sum_indicator_eq {n : ℕ} (P : FreeMonoid α → Prop) (b : Fin n → FreeMonoid α) :
    ∑ i, (if P (b i) then (1 : ℝ) else 0) = ((Finset.univ.filter fun i => P (b i)).card : ℝ) := by
  rw [Finset.sum_boole]

open scoped Classical in
theorem batch_indicator_facts (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) :
    (∀ i, AEMeasurable (fun b : Fin n → FreeMonoid α => if P (b i) then (1 : ℝ) else 0)
        (Measure.pi fun _ : Fin n => D))
      ∧ iIndepFun (fun i (b : Fin n → FreeMonoid α) => if P (b i) then (1 : ℝ) else 0)
        (Measure.pi fun _ : Fin n => D)
      ∧ (∀ i, ∀ᵐ b ∂(Measure.pi fun _ : Fin n => D),
          (if P (b i) then (1 : ℝ) else 0) ∈ Set.Icc (0 : ℝ) 1)
      ∧ ∀ i, (Measure.pi fun _ : Fin n => D)[fun b => if P (b i) then (1 : ℝ) else 0]
          = D.real {x | P x} := by
  refine ⟨fun i => (measurable_of_countable _).aemeasurable, ?_, fun i => ?_, fun i => ?_⟩
  · exact iIndepFun_pi (μ := fun _ : Fin n => D)
      (X := fun _ (y : FreeMonoid α) => if P y then (1 : ℝ) else 0)
      fun _ => (measurable_of_countable _).aemeasurable
  · exact Filter.Eventually.of_forall fun b => by split_ifs <;> simp
  · have hS : MeasurableSet {x : FreeMonoid α | P x} := MeasurableSpace.measurableSet_top
    have heq : (fun b : Fin n → FreeMonoid α => if P (b i) then (1 : ℝ) else 0)
        = (Function.eval i ⁻¹' {x | P x}).indicator 1 := by
      ext b; simp [Set.indicator_apply]
    rw [heq, integral_indicator_one ((measurable_pi_apply i) hS),
      measureReal_def, measureReal_def,
      ← Measure.map_apply (measurable_pi_apply i) hS, (measurePreserving_eval _ i).map_eq]

/-- A predicate holding on at most `θ − δ` of draws holds on more than `θ` of a batch of `n`
with chance at most `exp(−2nδ²)`. -/
theorem share_gt_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {θ δ : ℝ} (hδ : 0 ≤ δ)
    (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin n => D).real {b | θ < share b P} ≤ Real.exp (-2 * n * δ ^ 2) := by
  classical
  obtain ⟨hm, hi, hI, hmean⟩ := batch_indicator_facts D P n
  have key := sumUpper_le _ Finset.univ (θ - δ) δ hm hi hI (by
    simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact le_of_eq rfl |>.trans (by gcongr)) hδ
  simp only [Finset.card_univ, Fintype.card_fin, sub_add_cancel] at key
  refine le_trans (measureReal_mono fun b hb => ?_) key
  simp only [Set.mem_ofPred_eq, share] at hb ⊢
  rw [sum_indicator_eq]
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp at hb ⊢
  · have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    rw [lt_div_iff₀ hn'] at hb
    linarith

/-- One holding on at least `ε + δ` of draws holds on at most `ε` of a batch with chance at most
`exp(−2nδ²)`. -/
theorem share_le_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {ε δ : ℝ} (hδ : 0 ≤ δ)
    (h : ε + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin n => D).real {b | share b P ≤ ε} ≤ Real.exp (-2 * n * δ ^ 2) := by
  classical
  obtain ⟨hm, hi, hI, hmean⟩ := batch_indicator_facts D P n
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp only [CharP.cast_eq_zero, mul_zero, zero_mul, Real.exp_zero]
    exact measureReal_le_one
  have key := sumLower_le _ Finset.univ (ε + δ) δ hm hi hI (by
    simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    gcongr) hδ
  simp only [Finset.card_univ, Fintype.card_fin, add_sub_cancel_right] at key
  refine le_trans (measureReal_mono fun b hb => ?_) key
  simp only [Set.mem_ofPred_eq, share] at hb ⊢
  rw [sum_indicator_eq]
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  rwa [div_le_iff₀ hn', mul_comm] at hb

omit [DecidableEq α] in
open scoped Classical in
/-- At one look of an exact binomial test against `θ`, a rate of at most `θ` reads above it with
chance at most the test's failure chance. -/
theorem look_above_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) {N : ℕ} (S : Finset (Fin N)) {θ a : ℝ} (hθ1 : θ ≤ 1)
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
    rw [show {b : Fin N → FreeMonoid α | binomSfGe S.card θ (S.filter fun i => P (b i)).card < a}
      = ∅ from Set.eq_empty_of_forall_notMem fun b hb => (not_lt.2 (hex _)) hb]
    simpa using ha

omit [DecidableEq α] in
open scoped Classical in
/-- At one look, a rate of at least `θ` reads below it with chance at most the failure chance. -/
theorem look_below_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) {N : ℕ} (S : Finset (Fin N)) {θ a : ℝ} (hθ0 : 0 ≤ θ)
    (hp : θ ≤ D.real {x | P x}) (ha : 0 ≤ a) :
    (Measure.pi fun _ : Fin N => D).real
      {b | 1 - binomSfGe S.card θ ((S.filter fun i => P (b i)).card + 1) < a} ≤ a := by
  set ν := Measure.pi fun _ : Fin N => D
  set c : (Fin N → FreeMonoid α) → ℕ := fun b => (S.filter fun i => P (b i)).card
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

omit [DecidableEq α] in
open scoped Classical in
theorem hitsIn_eq {N : ℕ} (b : Fin N → FreeMonoid α) (P : FreeMonoid α → Prop) (n : ℕ) :
    hitsIn b P n
      = ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card := by
  simp only [hitsIn, Finset.filter_filter]

omit [DecidableEq α] in
theorem card_lt_filter {N n : ℕ} (hn : n ≤ N) :
    (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card = n := by
  rw [Fin.card_filter_val_lt, min_eq_right hn]

omit [DecidableEq α] in
theorem seqAbove_cases {θ a : ℝ} {n₀ N : ℕ} {b : Fin N → FreeMonoid α}
    {P : FreeMonoid α → Prop} (h : seqAbove θ a n₀ b P) :
    (∃ n ∈ Finset.Ico 1 (N + 1), n₀ ≤ n ∧ binomSfGe n θ (hitsIn b P n) < a)
      ∨ θ * N < hitsIn b P N := by
  unfold seqAbove at h
  split at h
  · rename_i s hs
    obtain ⟨n, hn, hns⟩ := List.exists_of_findSome?_eq_some hs
    subst h
    refine .inl ⟨n, ?_, ?_⟩
    · simp only [List.mem_range'_1] at hn
      simp only [Finset.mem_Ico]
      omega
    · unfold rateSide at hns
      split_ifs at hns with h1 h2 <;> simp_all
  · exact .inr h

omit [DecidableEq α] in
theorem not_seqAbove_cases {θ a : ℝ} {n₀ N : ℕ} {b : Fin N → FreeMonoid α}
    {P : FreeMonoid α → Prop} (h : ¬ seqAbove θ a n₀ b P) :
    (∃ n ∈ Finset.Ico 1 (N + 1), n₀ ≤ n ∧ 1 - binomSfGe n θ (hitsIn b P n + 1) < a)
      ∨ (hitsIn b P N : ℝ) ≤ θ * N := by
  unfold seqAbove at h
  split at h
  · rename_i s hs
    obtain ⟨n, hn, hns⟩ := List.exists_of_findSome?_eq_some hs
    have hs' : s = false := by simpa using h
    subst hs'
    refine .inl ⟨n, ?_, ?_⟩
    · simp only [List.mem_range'_1] at hn
      simp only [Finset.mem_Ico]
      omega
    · unfold rateSide at hns
      split_ifs at hns with h1 h2 h3 <;> simp_all
  · exact .inr (not_lt.1 h)

omit [DecidableEq α] in
open scoped Classical in
/-- `SequentialRate` reads a rate of at most `θ − δ` as above `θ` with chance at most
`N·a + exp(−2Nδ²)`: the test's failure chance at each of its looks, and Hoeffding's tail if it
never settles. -/
theorem seqAbove_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (N n₀ : ℕ) {θ a δ : ℝ} (hθ1 : θ ≤ 1) (ha : 0 ≤ a) (hδ : 0 ≤ δ)
    (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin N => D).real {b | seqAbove θ a n₀ b P}
      ≤ N * a + Real.exp (-2 * N * δ ^ 2) := by
  set ν := Measure.pi fun _ : Fin N => D
  have hsub : {b | seqAbove θ a n₀ b P}
      ⊆ (⋃ n ∈ Finset.Ico 1 (N + 1), {b : Fin N → FreeMonoid α |
          binomSfGe ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card) θ
            (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card)
            < a})
        ∪ {b | θ < share b P} := by
    intro b hb
    rcases seqAbove_cases hb with ⟨n, hn, -, hlt⟩ | hfin
    · refine .inl (Set.mem_biUnion hn ?_)
      have hnN : n ≤ N := by simp only [Finset.mem_Ico] at hn; omega
      simp only [Set.mem_ofPred_eq, card_lt_filter hnN, ← hitsIn_eq]
      exact hlt
    · refine .inr ?_
      rcases Nat.eq_zero_or_pos N with rfl | hN
      · simp [hitsIn] at hfin
      · have hN' : (0 : ℝ) < N := by exact_mod_cast hN
        simp only [Set.mem_ofPred_eq, share]
        rw [lt_div_iff₀ hN']
        have : hitsIn b P N = (Finset.univ.filter fun i => P (b i)).card := by
          simp [hitsIn]
        rw [← this]
        linarith
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have hu := measureReal_biUnion_finset_le (μ := ν) (Finset.Ico 1 (N + 1)) fun n =>
    {b : Fin N → FreeMonoid α |
      binomSfGe ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card) θ
        (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card) < a}
  have hlook : ∀ n ∈ Finset.Ico 1 (N + 1), ν.real {b : Fin N → FreeMonoid α |
      binomSfGe ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card) θ
        (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card) < a}
      ≤ a := fun n _ => look_above_le D P _ hθ1 (by linarith) ha
  have hsum := Finset.sum_le_sum hlook
  simp only [Finset.sum_const, Nat.card_Ico, add_tsub_cancel_right, nsmul_eq_mul] at hsum
  have hfin := share_gt_le D P N hδ h
  linarith

omit [DecidableEq α] in
open scoped Classical in
/-- And a rate of at least `θ + δ` as not above `θ` with chance at most the same. -/
theorem not_seqAbove_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (N n₀ : ℕ) {θ a δ : ℝ} (hθ0 : 0 ≤ θ) (ha : 0 ≤ a) (hδ : 0 ≤ δ)
    (h : θ + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin N => D).real {b | ¬ seqAbove θ a n₀ b P}
      ≤ N * a + Real.exp (-2 * N * δ ^ 2) := by
  set ν := Measure.pi fun _ : Fin N => D
  have hsub : {b | ¬ seqAbove θ a n₀ b P}
      ⊆ (⋃ n ∈ Finset.Ico 1 (N + 1), {b : Fin N → FreeMonoid α |
          1 - binomSfGe ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card) θ
            (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card
              + 1) < a})
        ∪ {b | share b P ≤ θ} := by
    intro b hb
    rcases not_seqAbove_cases hb with ⟨n, hn, -, hlt⟩ | hfin
    · refine .inl (Set.mem_biUnion hn ?_)
      have hnN : n ≤ N := by simp only [Finset.mem_Ico] at hn; omega
      simp only [Set.mem_ofPred_eq, card_lt_filter hnN, ← hitsIn_eq]
      exact hlt
    · refine .inr ?_
      rcases Nat.eq_zero_or_pos N with rfl | hN
      · simp [share, hθ0]
      · have hN' : (0 : ℝ) < N := by exact_mod_cast hN
        simp only [Set.mem_ofPred_eq, share]
        rw [div_le_iff₀ hN']
        have : hitsIn b P N = (Finset.univ.filter fun i => P (b i)).card := by
          simp [hitsIn]
        rw [← this]
        linarith
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have hu := measureReal_biUnion_finset_le (μ := ν) (Finset.Ico 1 (N + 1)) fun n =>
    {b : Fin N → FreeMonoid α |
      1 - binomSfGe ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card) θ
        (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card + 1)
        < a}
  have hlook : ∀ n ∈ Finset.Ico 1 (N + 1), ν.real {b : Fin N → FreeMonoid α |
      1 - binomSfGe ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card) θ
        (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card + 1)
        < a} ≤ a := fun n _ => look_below_le D P _ hθ0 (by linarith) ha
  have hsum := Finset.sum_le_sum hlook
  simp only [Finset.sum_const, Nat.card_Ico, add_tsub_cancel_right, nsmul_eq_mul] at hsum
  have hfin := share_le_le D P N hδ h
  linarith

end Batch

section Sources

variable (R : CutReads α)

/-- The walk is blocked: it does not reach the end. -/
def KWalk.isBlocked : KWalk α → Prop
  | .reached _ => False
  | _ => True

theorem walkOutput_isSome_iff (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (walkOutput R t edges k x).isSome
      ↔ (kWalk R t edges k x).isBlocked ∧ ¬ wrongEarlier R t edges k x := by
  unfold walkOutput wrongEarlier
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps
  · simp [KWalk.isBlocked]
  · have key : (∃ s' c' j' p, KWalk.edge s c j = KWalk.edge s' c' j'
        ∧ (t.sift R.cut (prefixOf x (j' + 1))).isLeft
        ∧ t.sift R.cut (prefixOf x j') = .inl p ∧ p ≠ s')
        ↔ (t.sift R.cut (prefixOf x (j + 1))).isLeft
          ∧ ∃ p, t.sift R.cut (prefixOf x j) = .inl p ∧ p ≠ s :=
      ⟨fun ⟨_, _, _, p, he, h1, h2, h3⟩ => by cases he; exact ⟨h1, p, h2, h3⟩,
        fun ⟨h1, p, h2, h3⟩ => ⟨s, c, j, p, rfl, h1, h2, h3⟩⟩
    rw [key]
    simp only [KWalk.isBlocked, true_and]
    rcases h1 : t.sift R.cut (prefixOf x (j + 1)) with p1 | b1
    · rcases h2 : t.sift R.cut (prefixOf x j) with p2 | b2
      · by_cases hp : p2 = s
        · simp [hp]
        · simp [hp]
      · simp
    · simp
  · simp [KWalk.isBlocked]

theorem checkOutput_isSome_iff (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (checkOutput R t edges k x).isSome
      ↔ (kCheck R t edges k x).isBlocked ∧ ¬ wrongEarlier R t edges k x := by
  have hwalk := walkOutput_isSome_iff R t edges k x
  unfold checkOutput kCheck at *
  unfold wrongEarlier at *
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps <;> rw [hw] at hwalk
  · simpa [KCheck.isBlocked, KWalk.isBlocked] using hwalk
  · simpa [KCheck.isBlocked, KWalk.isBlocked] using hwalk
  · simp only []
    rcases hs : t.sift R.cut x with a | b
    · simp only []
      split_ifs <;> simp [KCheck.isBlocked]
    · simp [KCheck.isBlocked]

theorem check_yield_holds : CheckYield := by
  intro α _ _ R D _ t edges k
  have hsub : {x | wrongEarlier R t edges k x} ⊆ {x | (kCheck R t edges k x).isBlocked} := by
    rintro x ⟨s, c, j, p, hw, -⟩
    simp [kCheck, hw, KCheck.isBlocked]
  have heq : {x | (checkOutput R t edges k x).isSome}
      = {x | (kCheck R t edges k x).isBlocked} \ {x | wrongEarlier R t edges k x} := by
    ext x
    simp only [Set.mem_ofPred_eq, Set.mem_sdiff]
    exact checkOutput_isSome_iff R t edges k x
  rw [heq, measureReal_sdiff hsub MeasurableSpace.measurableSet_top]

theorem route_inr {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x b : FreeMonoid α), (t.route cut x).2 = .inr b → ∃ m, b = x * m
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, x, b, h => by
    simp only [DTree.route] at h
    split at h
    · exact ⟨m, (Sum.inr.inj h).symm⟩
    · rcases ha : (a.route cut x).2 with p | b' <;> rw [ha] at h
      · simp at h
      · exact route_inr a x b' ha |>.imp fun m hm => by simp at h; rw [← h, hm]
    · rcases hr : (r.route cut x).2 with p | b' <;> rw [hr] at h
      · simp at h
      · exact route_inr r x b' hr |>.imp fun m hm => by simp at h; rw [← h, hm]

theorem take_prefixOf {x : FreeMonoid α} {i k : ℕ} (hk : k ≤ i) :
    (prefixOf x i).toList.take k = x.toList.take k := by
  simp [prefixOf, List.take_take, min_eq_left hk]

theorem walkOutput_prefix {t : DTree α} {edges : Edges α} {k : ℕ} {x u : FreeMonoid α}
    (h : walkOutput R t edges k x = some u) : u.toList.take k = x.toList.take k := by
  unfold walkOutput at h
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps <;> rw [hw] at h
  · simp only [Option.some.injEq] at h
    rw [← h, take_prefixOf le_rfl]
  · have hj : k ≤ j := by
      unfold kWalk at hw
      split at hw
      · simp at hw
      · split at hw
        · simp at hw
        · simp only [KWalk.edge.injEq] at hw
          omega
    simp only [] at h
    rcases h1 : t.sift R.cut (prefixOf x (j + 1)) with p1 | b1 <;> rw [h1] at h
    · simp only [] at h
      rcases h2 : t.sift R.cut (prefixOf x j) with p2 | b2 <;> rw [h2] at h
      · simp only [] at h
        split_ifs at h
        simp only [Option.some.injEq] at h
        rw [← h, take_prefixOf hj]
      · simp only [Option.some.injEq] at h
        rw [← h, take_prefixOf hj]
    · simp only [Option.some.injEq] at h
      rw [← h, take_prefixOf (by omega)]
  · simp at h

theorem source_spread_holds : SourceSpread := by
  intro α _ _ R D _ t edges k L u hkL hlen
  rw [measureReal_def, measureReal_def]
  refine ENNReal.toReal_mono (measure_ne_top _ _) (measure_mono_ae ?_)
  filter_upwards [hlen] with x hx hu
  change checkOutput R t edges k x = some u at hu
  change u.toList.take k <+: x.toList
  have hpre : ∀ v : FreeMonoid α, walkOutput R t edges k x = some v →
      v.toList.take k <+: x.toList := fun v hv => by
    rw [walkOutput_prefix R hv]; exact List.take_prefix _ _
  unfold checkOutput kCheck at hu
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps <;> rw [hw] at hu
  · exact hpre u (by simpa using hu)
  · exact hpre u (by simpa using hu)
  · simp only [] at hu
    rcases hs : t.sift R.cut x with a | b <;> rw [hs] at hu
    · simp only [] at hu
      split_ifs at hu <;> simp at hu
    · simp only [Option.some.injEq] at hu
      subst hu
      obtain ⟨m, rfl⟩ := route_inr _ _ _ hs
      simp only [FreeMonoid.toList_mul]
      rw [List.take_append_of_le_length (by omega)]
      exact List.take_prefix _ _

end Sources

section Learned

variable (K : StageKnobs α) (R : CutReads α)

/-- Every edge's witness sits at the edge's leaf, and its extension by the letter at the edge's
target. -/
def Learned (t : DTree α) (edges : Edges α) : Prop :=
  ∀ p c q y, edges p c = some (q, y) →
    t.sift R.cut y = .inl p ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q

omit [Fintype α] in
theorem route_splitAt {cut : FreeMonoid α → Option Bool} (d : FreeMonoid α) :
    ∀ (t : DTree α) (x : FreeMonoid α) (s p : List Bool), (t.route cut x).2 = .inl p → p ≠ s →
      ((t.splitAt d s).route cut x).2 = .inl p
  | .leaf, x, s, p, h, hps => by
    simp only [DTree.route, Sum.inl.injEq] at h
    subst h
    rcases s with _ | ⟨b, s⟩
    · exact absurd rfl hps
    · rfl
  | .node m r a, x, s, p, h, hps => by
    simp only [DTree.route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> rw [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with p' | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        rcases s with _ | ⟨_ | _, s⟩
        · simp [DTree.splitAt, DTree.route, hc, hr]
        · have := route_splitAt d r x s p' hr (fun he => hps (by rw [he]))
          simp [DTree.splitAt, DTree.route, hc, this]
        · simp [DTree.splitAt, DTree.route, hc, hr]
      · simp at h
    · rcases ha : (a.route cut x).2 with p' | b <;> rw [ha] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        rcases s with _ | ⟨_ | _, s⟩
        · simp [DTree.splitAt, DTree.route, hc, ha]
        · simp [DTree.splitAt, DTree.route, hc, ha]
        · have := route_splitAt d a x s p' ha (fun he => hps (by rw [he]))
          simp [DTree.splitAt, DTree.route, hc, this]
      · simp at h

omit [Fintype α] in
theorem sift_splitAt {cut : FreeMonoid α → Option Bool} {t : DTree α} {x : FreeMonoid α}
    {d : FreeMonoid α} {s p : List Bool} (h : t.sift cut x = .inl p) (hps : p ≠ s) :
    (t.splitAt d s).sift cut x = .inl p :=
  route_splitAt d t x s p h hps

theorem decisiveTarget_learned {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {c : α} {cur : Option (List Bool)} {q : List Bool} {y : FreeMonoid α}
    (h : decisiveTarget K R t pool path c cur = some (q, y)) :
    t.sift R.cut y = .inl path ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q := by
  refine ⟨?_, ?_⟩
  · have hm := decisiveTarget_mem K R h
    have := (List.mem_filter.1 (List.mem_of_mem_take hm)).2
    simpa using this
  · unfold decisiveTarget at h
    dsimp only at h
    generalize hV : List.filterMap _ (members K R t pool path) = V at h
    rcases foldl_best_mem _ (fun o v r hr => by
        rcases o with _ | b
        · exact .inl (Option.some.inj hr).symm
        · dsimp only at hr
          split_ifs at hr
          · exact .inl (Option.some.inj hr).symm
          · exact .inr hr) V none (q, y) h with hm | hm
    · rw [← hV] at hm
      obtain ⟨a, -, hfa⟩ := List.mem_filterMap.1 hm
      split at hfa
      · rename_i p hp
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hfa)
        exact hp
      · simp at hfa
    · simp at hm

theorem closeEdges_learned {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (h : Learned R t edges) : Learned R t (closeEdges K R t pool edges) := by
  intro p c q y he
  simp only [closeEdges] at he
  rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', y'⟩ <;>
    rw [hd] at he
  · exact h p c q y (by simpa using he)
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj (by simpa using he))
    exact decisiveTarget_learned K R hd

theorem probeStepK_learned {k : ℕ} {s : KState α} {x : FreeMonoid α}
    (h : Learned R s.tree s.edges) :
    Learned R (probeStepK K R k s x).tree (probeStepK K R k s x).edges := by
  simp only [probeStepK]
  split
  · split
    swap
    · exact closeEdges_learned K R h
    split
    · rename_i d s1 y sprime _
      refine closeEdges_learned K R fun p c q w he => ?_
      rcases hE : s.edges p c with _ | ⟨q', w'⟩ <;> simp only [hE] at he
      · simp at he
      · split_ifs at he with hcl
        all_goals first
          | (simp at he; done)
          | (simp only [Option.some.injEq, Prod.mk.injEq] at he
             obtain ⟨rfl, rfl⟩ := he
             have hcl' : p ≠ s1 ∧ q' ≠ s1 := by tauto
             obtain ⟨h1, h2⟩ := h p c q' w' hE
             exact ⟨sift_splitAt h1 hcl'.1, sift_splitAt h2 hcl'.2⟩)
    · exact closeEdges_learned K R h
    · exact closeEdges_learned K R h
  · exact closeEdges_learned K R h

theorem runPassK_learned (k : ℕ) (seed probes : List (FreeMonoid α)) :
    Learned R (runPassK K R k (initialK K R seed) probes).tree
      (runPassK K R k (initialK K R seed) probes).edges := by
  unfold runPassK
  have h0 : Learned R (initialK K R seed).tree (initialK K R seed).edges :=
    closeEdges_learned K R fun _ _ _ _ he => by simp at he
  generalize initialK K R seed = s at h0 ⊢
  induction probes generalizing s with
  | nil => exact h0
  | cons x xs ih =>
    simp only [List.foldl_cons]
    refine ih _ ?_
    split
    · exact h0
    · exact probeStepK_learned K R h0

theorem parting_none {cut : FreeMonoid α → Option Bool} {x y pre : FreeMonoid α} :
    ∀ t : DTree α, t.parting cut x y pre = none →
      ∃ p, t.sift cut (x * pre) = .inl p ∧ t.sift cut (y * pre) = .inl p
  | .leaf, _ => ⟨[], rfl, rfl⟩
  | .node m r a, h => by
    simp only [DTree.parting] at h
    unfold DTree.sift DTree.route
    simp only [mul_assoc]
    rcases hx : cut (x * (pre * m)) with _ | _ | _ <;>
      rcases hy : cut (y * (pre * m)) with _ | _ | _ <;>
      simp only [hx, hy, reduceCtorEq] at h
    · obtain ⟨p, h1, h2⟩ := parting_none r h
      unfold DTree.sift at h1 h2
      exact ⟨false :: p, by simp [h1], by simp [h2]⟩
    · obtain ⟨p, h1, h2⟩ := parting_none a h
      unfold DTree.sift at h1 h2
      exact ⟨true :: p, by simp [h1], by simp [h2]⟩

omit [Fintype α] [DecidableEq α] in
theorem follow_inl {edges : Edges α} :
    ∀ (cs : List α) (p : List Bool) (ps : List (List Bool)), follow edges p cs = .inl ps →
      ps.length = cs.length + 1 ∧ ps.getD 0 [] = p
        ∧ ∀ i (hi : i < cs.length), ∃ y, edges (ps.getD i []) cs[i] = some (ps.getD (i + 1) [], y)
  | [], p, ps, h => by
    simp only [follow, Sum.inl.injEq] at h
    subst h
    exact ⟨rfl, rfl, fun i hi => absurd hi (by simp)⟩
  | c :: cs, p, ps, h => by
    simp only [follow] at h
    rcases he : edges p c with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp at h
    · simp only [] at h
      rcases hf : follow edges q cs with ps' | r <;> rw [hf] at h
      · simp only [Sum.inl.injEq] at h
        subst h
        obtain ⟨hl, hh, hs⟩ := follow_inl cs q ps' hf
        refine ⟨by simp [hl], rfl, fun i hi => ?_⟩
        rcases i with _ | i
        · exact ⟨y, by simp only [List.getD_cons_zero, List.getD_cons_succ, hh,
            List.getElem_cons_zero]; exact he⟩
        · obtain ⟨y', hy'⟩ := hs i (by simpa using hi)
          exact ⟨y', by simpa using hy'⟩
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_edge (agrees : ℕ → Option Bool) :
    ∀ fuel lo hi j, lo < hi → hi - lo ≤ fuel → agrees lo = some true → agrees hi = some false →
      bracketAt agrees fuel lo hi = .edge j →
      lo < j ∧ j ≤ hi ∧ agrees (j - 1) = some true ∧ agrees j = some false
  | 0, lo, hi, j, hlt, hf, _, _, _ => by omega
  | fuel + 1, lo, hi, j, hlt, hf, hlo, hhi, h => by
    have hag : ∀ p, (if p = lo then some true else if p = hi then some false else agrees p)
        = agrees p := fun p => by split_ifs <;> simp_all
    simp only [bracketAt, hag] at h
    split_ifs at h with h1
    · rcases hm : agrees ((lo + hi) / 2) with _ | _ | _ <;> simp only [hm] at h
      · rcases hl : agrees ((lo + hi) / 2 - 1) with _ | _ | _ <;>
          rcases hr : agrees ((lo + hi) / 2 + 1) with _ | _ | _ <;>
          simp only [hl, hr, reduceCtorEq] at h
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees fuel lo _ j this (by omega) hlo hl h
          exact ⟨this.1, by omega, this.2.2⟩
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees fuel lo _ j this (by omega) hlo hl h
          exact ⟨this.1, by omega, this.2.2⟩
        · have : (lo + hi) / 2 + 1 < hi := by
            rcases Nat.lt_or_ge ((lo + hi) / 2 + 1) hi with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 + 1 = hi by omega, hhi] at hr; simp at hr
          have := bracketAt_edge agrees fuel _ hi j this (by omega) hr hhi h
          exact ⟨by omega, this.2⟩
      · have := bracketAt_edge agrees fuel lo _ j (by omega) (by omega) hlo hm h
        exact ⟨this.1, by omega, this.2.2⟩
      · have := bracketAt_edge agrees fuel _ hi j (by omega) (by omega) hm hhi h
        exact ⟨by omega, this.2⟩
    · obtain rfl := Bracket.edge.inj h
      obtain rfl : hi = lo + 1 := by omega
      exact ⟨by omega, le_rfl, by simpa using hlo, hhi⟩

theorem kCheck_disagree {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} (h : kCheck R t edges k x = .disagree ps) :
    ∃ p₀ a, t.sift R.cut (prefixOf x k) = .inl p₀ ∧ follow edges p₀ (x.toList.drop k) = .inl ps
      ∧ t.sift R.cut x = .inl a ∧ some a ≠ ps.getLast? := by
  unfold kCheck at h
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hw] at h
  · simp at h
  · simp at h
  · simp only [] at h
    rcases hs : t.sift R.cut x with a | b <;> rw [hs] at h
    · simp only [] at h
      split_ifs at h with ha
      simp only [KCheck.disagree.injEq] at h
      subst h
      unfold kWalk at hw
      rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
      · simp only [] at hw
        rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | r <;> rw [hf] at hw
        · simp only [KWalk.reached.injEq] at hw
          subst hw
          exact ⟨p₀, a, rfl, hf, rfl, ha⟩
        · simp at hw
      · simp at hw
    · simp at h

theorem seedStep_ne_dropped {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (hl : Learned R t edges) {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)}
    (h : kCheck R t edges k x = .disagree ps) {fd : ℕ} (hb : bracketOf R t k x ps = .edge fd) :
    seedStep K R t pool edges k x ps fd ≠ .dropped := by
  obtain ⟨p₀, a, hk, hf, hx, hne⟩ := kCheck_disagree R h
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  set n := x.toList.length with hn
  set walkAt : ℕ → List Bool := fun j => ps.getD (j - k) [] with hwalk
  have hlast : ps.getLast? = some (ps.getD (n - k) []) := by
    simp only [List.length_drop] at hlen
    have hidx : n - k < ps.length := by omega
    rw [List.getLast?_eq_getElem?, show ps.length - 1 = n - k by omega,
      List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hidx]
    rfl
  have hkn : k < n := by
    by_contra hkn
    have hxk : prefixOf x k = x := by
      apply FreeMonoid.toList.injective
      simp [prefixOf, List.take_of_length_le (not_lt.1 hkn)]
    rw [hxk, hx] at hk
    obtain rfl := Sum.inl.inj hk
    simp only [List.length_drop] at hlen
    have : ps.getD (n - k) [] = a := by
      rw [show n - k = 0 by omega, hhead]
    exact hne (by rw [hlast, this])
  have hpk : agreesAt R t x walkAt k = some true := by
    simp only [agreesAt, hk, Sum.elim_inl, hwalk, Nat.sub_self, hhead, decide_true]
  have hpn : agreesAt R t x walkAt n = some false := by
    have : prefixOf x n = x := prefixOf_length x
    simp only [agreesAt, this, hx, Sum.elim_inl, hwalk, Option.some.injEq, decide_eq_false_iff_not]
    intro he
    exact hne (by rw [hlast, he])
  obtain ⟨hfd1, hfd2, hfd3, hfd4⟩ :=
    bracketAt_edge (agreesAt R t x walkAt) (n - k) k n fd hkn le_rfl hpk hpn hb
  unfold seedStep
  simp only []
  have hfdn : fd - 1 < n := by omega
  rw [List.getElem?_eq_getElem hfdn]
  simp only []
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) (by simp; omega)
  have hidx : (x.toList.drop k)[fd - 1 - k]'(by simp; omega) = x.toList[fd - 1] := by
    simp only [List.getElem_drop]
    congr 1
    omega
  rw [hidx, show fd - 1 - k + 1 = fd - k by omega] at hy
  rw [hy]
  simp only [ne_eq, not_true_eq_false, if_false]
  obtain ⟨hy1, hy2⟩ := hl _ _ _ _ hy
  rcases hsp : t.sift R.cut (prefixOf x (fd - 1)) with p | b
  · simp only []
    have hp : p = ps.getD (fd - 1 - k) [] := by
      have := hfd3
      simp only [agreesAt, hsp, Sum.elim_inl, hwalk, Option.some.injEq, decide_eq_true_eq] at this
      exact this
    subst hp
    simp only [ne_eq, not_true_eq_false, false_or, hy1, not_true_eq_false, if_false]
    rcases hpt : t.parting R.cut y (prefixOf x (fd - 1)) (FreeMonoid.of x.toList[fd - 1]) with
      _ | d | b
    · exfalso
      obtain ⟨q, h1, h2⟩ := parting_none t hpt
      rw [hy2] at h1
      rw [← prefixOf_succ hfdn, show fd - 1 + 1 = fd by omega] at h2
      simp only [agreesAt, ← h1, h2, Sum.elim_inl, hwalk, Option.some.injEq,
        decide_eq_false_iff_not, not_true_eq_false] at hfd4
    · simp only []
      split <;> simp
    · simp
  · simp

end Learned

/-- A draw the gate counts against the hypothesis is one the check source outputs a string for,
or one a refusal can carry. -/
theorem gateDisagrees_cover {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : gateDisagrees R t edges k x) :
    (checkOutput R t edges k x).isSome ∨ Carried R t edges k x := by
  unfold checkOutput Carried
  unfold kCheck
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps
  · left
    have : walkOutput R t edges k x = some (prefixOf x k) := by simp [walkOutput, hw]
    simp [this]
  · by_cases hwe : wrongEarlier R t edges k x
    · exact .inr (.inr hwe)
    · left
      have := (walkOutput_isSome_iff R t edges k x).2 ⟨by simp [hw, KWalk.isBlocked], hwe⟩
      simpa using this
  · simp only []
    rcases hs : t.sift R.cut x with a | b
    · simp only []
      split_ifs with ha
      · exfalso
        unfold kWalk at hw
        rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
        · simp only [] at hw
          rcases hf : follow edges p₀ (x.toList.drop k) with ps' | r <;> rw [hf] at hw
          · simp only [KWalk.reached.injEq] at hw
            subst hw
            unfold gateDisagrees at h
            have hp0 : place R t x k = p₀ := by simp [place, hk]
            have hpn : place R t x x.toList.length = a := by
              simp [place, prefixOf_length, hs]
            rw [hp0, hf, hpn] at h
            exact h ha.symm
          · simp at hw
        · simp at hw
      · exact .inr (.inl ⟨ps, rfl⟩)
    · left; simp

omit [DecidableEq α] in
open scoped Classical in
/-- A batch whose first `n₀` draws all miss a set of mass above `δ` comes up with chance at most
`exp(−min(n₀, n)·δ)`. -/
theorem miss_first_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (C : FreeMonoid α → Prop) (n n₀ : ℕ) {δ : ℝ} (hδ : 0 ≤ δ) :
    (Measure.pi fun _ : Fin n => D).real
      {b | (∀ i : Fin n, (i : ℕ) < n₀ → ¬ C (b i)) ∧ δ < D.real {x | C x}}
      ≤ Real.exp (-(min n₀ n : ℕ) * δ) := by
  by_cases h : δ < D.real {x | C x}
  · have hset : {b : Fin n → FreeMonoid α | (∀ i : Fin n, (i : ℕ) < n₀ → ¬ C (b i))
        ∧ δ < D.real {x | C x}}
        = Set.univ.pi fun i : Fin n => if (i : ℕ) < n₀ then {x | ¬ C x} else Set.univ := by
      ext b
      simp only [Set.mem_ofPred_eq, Set.mem_pi, Set.mem_univ, true_implies, h, and_true]
      refine forall_congr' fun i => ?_
      split_ifs <;> simp_all
    rw [hset, measureReal_def, Measure.pi_pi]
    have hc : D {x | ¬ C x} = ENNReal.ofReal (1 - D.real {x | C x}) := by
      rw [show {x | ¬ C x} = {x | C x}ᶜ from rfl, ← ENNReal.ofReal_toReal (measure_ne_top D _),
        ← measureReal_def, measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
    have hp1 : D.real {x | C x} ≤ 1 := measureReal_le_one
    have hterm : ∀ i : Fin n, D (if (i : ℕ) < n₀ then {x | ¬ C x} else Set.univ)
        = if (i : ℕ) < n₀ then ENNReal.ofReal (1 - D.real {x | C x}) else 1 := by
      intro i; split_ifs <;> simp [hc]
    simp only [hterm]
    rw [      Finset.prod_ite, Finset.prod_const_one, mul_one, Finset.prod_const,
      ENNReal.toReal_pow, ENNReal.toReal_ofReal (by linarith)]
    have hcard : (Finset.univ.filter fun i : Fin n => (i : ℕ) < n₀).card = min n₀ n := by
      rw [Fin.card_filter_val_lt, min_comm]
    rw [hcard]
    calc (1 - D.real {x | C x}) ^ min n₀ n ≤ Real.exp (-D.real {x | C x}) ^ min n₀ n :=
          pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-D.real {x | C x})]) _
      _ = Real.exp (-(min n₀ n : ℕ) * D.real {x | C x}) := by
          rw [← Real.exp_nat_mul]; ring_nf
      _ ≤ Real.exp (-(min n₀ n : ℕ) * δ) := Real.exp_le_exp.2 (by
          have : (0 : ℝ) ≤ (min n₀ n : ℕ) := Nat.cast_nonneg _
          nlinarith)
  · rw [show {b : Fin n → FreeMonoid α | (∀ i : Fin n, (i : ℕ) < n₀ → ¬ C (b i))
        ∧ δ < D.real {x | C x}} = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
    simp only [measureReal_empty]
    positivity

theorem gate_bad_le (K : StageKnobs α) (R : CutReads α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k ng n₀ : ℕ) (sg : KState α) (hl : Learned R sg.tree sg.edges)
    {θc acc a δ : ℝ} (hθc1 : θc ≤ 1) (hacc0 : 0 ≤ acc) (hacc1 : acc ≤ 1) (ha : 0 ≤ a)
    (hδ : 0 ≤ δ) :
    (Measure.pi fun _ : Fin ng => D).real {bg | ¬ RoundAtKHolds K R sg D k θc acc a δ n₀ bg}
      ≤ 2 * (ng * a + Real.exp (-2 * ng * δ ^ 2)) + Real.exp (-(min n₀ ng : ℕ) * δ) := by
  classical
  set ν := Measure.pi fun _ : Fin ng => D
  set Pc := fun x => (kCheck R sg.tree sg.edges k x).isBlocked
  set Pd := fun x => gateDisagrees R sg.tree sg.edges k x
  set B1 := {bg : Fin ng → FreeMonoid α | seqAbove θc a n₀ bg Pc ∧ D.real {x | Pc x} < θc - δ}
  set B2 := {bg : Fin ng → FreeMonoid α |
    seqAbove acc a n₀ bg (fun x => ¬ Pd x) ∧ 1 - acc + δ < D.real {x | Pd x}}
  set B3 := {bg : Fin ng → FreeMonoid α |
    ¬ seqAbove acc a n₀ bg (fun x => ¬ Pd x) ∧ D.real {x | Pd x} < 1 - acc - δ}
  set C := fun x => Carried R sg.tree sg.edges k x
  set B4 := {bg : Fin ng → FreeMonoid α |
    (∀ i : Fin ng, (i : ℕ) < n₀ → ¬ C (bg i)) ∧ δ < D.real {x | C x}}
  have hcov : D.real {x | Pd x}
      ≤ D.real {x | (checkOutput R sg.tree sg.edges k x).isSome} + D.real {x | C x} :=
    (measureReal_mono fun x hx => gateDisagrees_cover hx).trans (measureReal_union_le _ _)
  have hsub : {bg | ¬ RoundAtKHolds K R sg D k θc acc a δ n₀ bg} ⊆ (B1 ∪ (B2 ∪ B3)) ∪ B4 := by
    intro bg hb
    simp only [Set.mem_ofPred_eq, RoundAtKHolds, not_and_or, Classical.not_imp] at hb
    rcases hb with ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩
    · exact .inl (.inl ⟨h1, not_le.1 h2⟩)
    · exact .inl (.inr (.inl ⟨h1, not_le.1 h2⟩))
    · rcases h2 with h2 | h2 | ⟨h3, h4⟩
      · exact .inl (.inr (.inr ⟨h1, not_le.1 h2⟩))
      · exfalso
        simp only [not_forall, not_not] at h2
        obtain ⟨i, ps, fd, hc, hb, hd⟩ := h2
        exact seedStep_ne_dropped K R hl hc hb hd
      · exact .inr ⟨h3, by linarith [not_le.1 h4]⟩
  have hcomp : D.real {x | ¬ Pd x} = 1 - D.real {x | Pd x} := by
    rw [show {x | ¬ Pd x} = {x | Pd x}ᶜ from rfl,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
  have hB1 : ν.real B1 ≤ ng * a + Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h : D.real {x | Pc x} < θc - δ
    · exact (measureReal_mono fun b hb => hb.1).trans (seqAbove_le D Pc ng n₀ hθc1 ha hδ h.le)
    · rw [show B1 = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h hb.2]
      simp only [measureReal_empty]
      positivity
  have hB23 : ν.real (B2 ∪ B3) ≤ ng * a + Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h2 : 1 - acc + δ < D.real {x | Pd x}
    · have hB3 : B3 = ∅ := Set.eq_empty_of_forall_notMem fun b hb => by
        have := hb.2; linarith
      rw [hB3, Set.union_empty]
      refine (measureReal_mono fun b hb => hb.1).trans
        (seqAbove_le D (fun x => ¬ Pd x) ng n₀ hacc1 ha hδ (by rw [hcomp]; linarith))
    · have hB2 : B2 = ∅ := Set.eq_empty_of_forall_notMem fun b hb => h2 hb.2
      rw [hB2, Set.empty_union]
      by_cases h3 : D.real {x | Pd x} < 1 - acc - δ
      · refine (measureReal_mono fun b hb => hb.1).trans
          (not_seqAbove_le D (fun x => ¬ Pd x) ng n₀ hacc0 ha hδ (by rw [hcomp]; linarith))
      · rw [show B3 = ∅ from Set.eq_empty_of_forall_notMem fun b hb => h3 hb.2]
        simp only [measureReal_empty]
        positivity
  have hB4 := miss_first_le D C ng n₀ hδ
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have := measureReal_union_le (μ := ν) B1 (B2 ∪ B3)
  linarith

theorem round_at_k_holds : RoundAtK := by
  intro α _ _ K R D _ k ng n₀ seed probes θc acc a δ hθc1 hacc0 hacc1 ha hδ
  exact gate_bad_le K R D k ng n₀ _ (runPassK_learned K R k seed probes) hθc1 hacc0 hacc1 ha hδ

end OrthoDFA
