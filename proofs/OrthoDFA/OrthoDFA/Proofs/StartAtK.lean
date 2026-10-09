import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.GateFlip
import OrthoDFA.Proofs.GateTests

/-!
# `RoundAtK`

A decision on a batch against a fixed hypothesis is an exact binomial test at the look where the
gate stops, or a Hoeffding tail at the batch's end.  An edge reaches the split test or stops at a
string the cut cannot place, because the pass keeps its edges learned: every edge's witness sits
at its leaf and its extension at its target.
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
theorem hitsIn_eq {N : ℕ} (b : Fin N → FreeMonoid α) (P : FreeMonoid α → Prop) (n : ℕ) :
    hitsIn b P n
      = ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card := by
  simp only [hitsIn, Finset.filter_filter]

omit [DecidableEq α] in
theorem card_lt_filter {N n : ℕ} (hn : n ≤ N) :
    (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card = n := by
  rw [Fin.card_filter_val_lt, min_eq_right hn]

/-- One holding on at most `θ − δ` of draws holds on at least `θ` of a batch with chance at most
`exp(−2nδ²)`. -/
theorem share_ge_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {θ δ : ℝ} (hδ : 0 ≤ δ)
    (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin n => D).real {b | θ * n ≤ hitsIn b P n}
      ≤ Real.exp (-2 * n * δ ^ 2) := by
  classical
  obtain ⟨hm, hi, hI, hmean⟩ := batch_indicator_facts D P n
  have key := sumUpper_le _ Finset.univ (θ - δ) δ hm hi hI (by
    simp only [hmean, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    exact le_of_eq rfl |>.trans (by gcongr)) hδ
  simp only [Finset.card_univ, Fintype.card_fin, sub_add_cancel] at key
  refine le_trans (measureReal_mono fun b hb => ?_) key
  simp only [Set.mem_ofPred_eq] at hb ⊢
  rw [sum_indicator_eq]
  have : hitsIn b P n = (Finset.univ.filter fun i => P (b i)).card := by simp [hitsIn]
  rw [← this, mul_comm]
  exact hb

theorem hits_gt_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {θ δ : ℝ} (hδ : 0 ≤ δ)
    (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin n => D).real {b | θ * n < hitsIn b P n}
      ≤ Real.exp (-2 * n * δ ^ 2) :=
  le_trans (measureReal_mono (s₁ := {b : Fin n → FreeMonoid α | θ * n < hitsIn b P n})
    (s₂ := {b | θ * n ≤ hitsIn b P n}) (fun b (hb : θ * n < hitsIn b P n) => le_of_lt hb)
    (measure_ne_top _ _)) (share_ge_le D P n hδ h)

open scoped Classical in
theorem hits_le_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P : FreeMonoid α → Prop) (n : ℕ) {θ δ : ℝ} (hθ0 : 0 ≤ θ) (hδ : 0 ≤ δ)
    (h : θ + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin n => D).real {b | (hitsIn b P n : ℝ) ≤ θ * n}
      ≤ Real.exp (-2 * n * δ ^ 2) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact measureReal_le_one.trans (by simp)
  refine (measureReal_mono (fun b hb => ?_) (measure_ne_top _ _)).trans (share_le_le D P n hδ h)
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  simp only [Set.mem_ofPred_eq, share] at hb ⊢
  rw [div_le_iff₀ hn']
  have : hitsIn b P n = (Finset.univ.filter fun i => P (b i)).card := by simp [hitsIn]
  rw [← this]
  linarith

omit [Fintype α] [DecidableEq α] in
theorem lookSet_card (n₀ N : ℕ) : (lookSet n₀ N).card ≤ Nat.log 2 (N / n₀) + 2 := by
  unfold lookSet
  refine (Finset.card_insert_le _ _).trans ?_
  have := (Finset.card_filter_le
    ((Finset.range (Nat.log 2 (N / n₀) + 1)).image (n₀ * 2 ^ ·)) (· ≤ N)).trans
    (Finset.card_image_le.trans (Finset.card_range _).le)
  omega

omit [Fintype α] [DecidableEq α] in
theorem mem_lookSet {n₀ N n : ℕ} (h : n ∈ lookSet n₀ N) : n ≤ N ∧ min n₀ N ≤ n := by
  unfold lookSet at h
  rcases Finset.mem_insert.1 h with rfl | h
  · exact ⟨le_rfl, min_le_right _ _⟩
  · obtain ⟨h1, h2⟩ := Finset.mem_filter.1 h
    obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 h1
    exact ⟨h2, (min_le_left _ _).trans (Nat.le_mul_of_pos_right _ (Nat.two_pow_pos i))⟩

omit [DecidableEq α] in
theorem stopLook_spec {θ θ' a : ℝ} {n₀ N : ℕ} (b : Fin N → FreeMonoid α)
    (P P' : FreeMonoid α → Prop) :
    stopLook θ θ' a n₀ b P P' ∈ lookSet n₀ N ∧ stopLook θ θ' a n₀ b P P' ≤ N
      ∧ min n₀ N ≤ stopLook θ θ' a n₀ b P P'
      ∧ (stopLook θ θ' a n₀ b P P' = N
        ∨ ((rateSide θ a n₀ (stopLook θ θ' a n₀ b P P')
              (hitsIn b P (stopLook θ θ' a n₀ b P P'))).isSome
          ∧ (rateSide θ' a n₀ (stopLook θ θ' a n₀ b P P')
              (hitsIn b P' (stopLook θ θ' a n₀ b P P'))).isSome)) := by
  have hN : N ∈ lookSet n₀ N := Finset.mem_insert_self _ _
  unfold stopLook
  rcases hf : ((lookSet n₀ N).sort (· ≤ ·)).find?
      (fun n => (rateSide θ a n₀ n (hitsIn b P n)).isSome
      && (rateSide θ' a n₀ n (hitsIn b P' n)).isSome) with _ | n
  · simp only [Option.getD_none]
    exact ⟨hN, le_rfl, min_le_right _ _, by simp⟩
  · have hmem : n ∈ lookSet n₀ N := (Finset.mem_sort _).1 (List.mem_of_find?_eq_some hf)
    have hp := List.find?_some hf
    simp only [Bool.and_eq_true] at hp
    simp only [Option.getD_some]
    exact ⟨hmem, (mem_lookSet hmem).1, (mem_lookSet hmem).2, .inr hp⟩

omit [DecidableEq α] in
theorem rateSide_isSome {θ a : ℝ} {n₀ n h : ℕ} (hs : (rateSide θ a n₀ n h).isSome) :
    binomSfGe n θ h < a ∨ 1 - binomSfGe n θ (h + 1) < a := by
  unfold rateSide at hs
  split_ifs at hs with h1 h2 h3 <;> simp_all

omit [DecidableEq α] in
open scoped Classical in
theorem look_set {N n : ℕ} (hn : n ≤ N) (b : Fin N → FreeMonoid α) (P : FreeMonoid α → Prop) :
    hitsIn b P n = ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card
      ∧ (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card = n :=
  ⟨hitsIn_eq b P n, card_lt_filter hn⟩

omit [DecidableEq α] in
theorem sum_looks_le {N n₀ : ℕ} {c : ℝ} (hc : 0 ≤ c) (g : ℕ → ℝ)
    (hg : ∀ n ∈ lookSet n₀ N, g n ≤ c) :
    ∑ n ∈ lookSet n₀ N, g n ≤ (Nat.log 2 (N / n₀) + 2) * c := by
  refine (Finset.sum_le_sum hg).trans ?_
  rw [Finset.sum_const, nsmul_eq_mul]
  have : ((lookSet n₀ N).card : ℝ) ≤ Nat.log 2 (N / n₀) + 2 := by exact_mod_cast lookSet_card n₀ N
  nlinarith

omit [DecidableEq α] in
open scoped Classical in
/-- Passing at the look where the gate stops, its share at least `θ`, on a rate of at most
`θ − δ`: at most twice the failure chance per look and Hoeffding's tail at the end. -/
theorem gate_pass_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P P' : FreeMonoid α → Prop) (N n₀ : ℕ) {θ θ' a δ : ℝ} (hθ1 : θ ≤ 1) (ha : 0 ≤ a)
    (hδ : 0 ≤ δ) (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin N => D).real {b | θ * stopLook θ θ' a n₀ b P P'
        ≤ hitsIn b P (stopLook θ θ' a n₀ b P P')}
      ≤ 2 * (Nat.log 2 (N / n₀) + 2) * a + Real.exp (-2 * N * δ ^ 2) := by
  set ν := Measure.pi fun _ : Fin N => D
  set L : ℕ → Set (Fin N → FreeMonoid α) := fun n =>
    {b | θ * (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card
        ≤ ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card
      ∧ (binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card θ
          ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card < a
        ∨ 1 - binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card θ
          (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card + 1)
          < a)}
  have hsub : {b | θ * stopLook θ θ' a n₀ b P P' ≤ hitsIn b P (stopLook θ θ' a n₀ b P P')}
      ⊆ (⋃ n ∈ lookSet n₀ N, L n) ∪ {b | θ * N ≤ hitsIn b P N} := by
    intro b hb
    simp only [Set.mem_ofPred_eq] at hb
    obtain ⟨hmem, hle, -, hT | ⟨hs, -⟩⟩ :=
      stopLook_spec (θ := θ) (θ' := θ') (a := a) (n₀ := n₀) b P P'
    · exact .inr (by simpa [hT] using hb)
    · refine .inl (Set.mem_biUnion hmem ?_)
      obtain ⟨he, hc⟩ := look_set hle b P
      simp only [L, Set.mem_ofPred_eq, hc, ← he]
      exact ⟨hb, rateSide_isSome hs⟩
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have hu := measureReal_biUnion_finset_le (μ := ν) (lookSet n₀ N) L
  have hsum := sum_looks_le (n₀ := n₀) (N := N) (by linarith : (0 : ℝ) ≤ 2 * a)
    (fun n => ν.real (L n)) fun n _ => look_pass_le D P _ hθ1 (by linarith) ha
  have hfin := share_ge_le D P N hδ h
  nlinarith

omit [DecidableEq α] in
open scoped Classical in
/-- Refusing at the stop, its share short of `θ`, on a rate of at least `θ + δ`. -/
theorem gate_refuse_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P P' : FreeMonoid α → Prop) (N n₀ : ℕ) {θ θ' a δ : ℝ} (hθ0 : 0 ≤ θ) (ha : 0 ≤ a)
    (hδ : 0 ≤ δ) (h : θ + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin N => D).real {b | (hitsIn b P (stopLook θ θ' a n₀ b P P') : ℝ)
        < θ * stopLook θ θ' a n₀ b P P'}
      ≤ 2 * (Nat.log 2 (N / n₀) + 2) * a + Real.exp (-2 * N * δ ^ 2) := by
  set ν := Measure.pi fun _ : Fin N => D
  set L : ℕ → Set (Fin N → FreeMonoid α) := fun n =>
    {b | (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card : ℝ)
        < θ * (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card
      ∧ (binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card θ
          ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card < a
        ∨ 1 - binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card θ
          (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card + 1)
          < a)}
  have hsub : {b | (hitsIn b P (stopLook θ θ' a n₀ b P P') : ℝ) < θ * stopLook θ θ' a n₀ b P P'}
      ⊆ (⋃ n ∈ lookSet n₀ N, L n) ∪ {b | (hitsIn b P N : ℝ) ≤ θ * N} := by
    intro b hb
    simp only [Set.mem_ofPred_eq] at hb
    obtain ⟨hmem, hle, -, hT | ⟨hs, -⟩⟩ :=
      stopLook_spec (θ := θ) (θ' := θ') (a := a) (n₀ := n₀) b P P'
    · refine .inr ?_
      rw [hT] at hb
      exact hb.le
    · refine .inl (Set.mem_biUnion hmem ?_)
      obtain ⟨he, hc⟩ := look_set hle b P
      simp only [L, Set.mem_ofPred_eq, hc, ← he]
      exact ⟨hb, rateSide_isSome hs⟩
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have hu := measureReal_biUnion_finset_le (μ := ν) (lookSet n₀ N) L
  have hsum := sum_looks_le (n₀ := n₀) (N := N) (by linarith : (0 : ℝ) ≤ 2 * a)
    (fun n => ν.real (L n)) fun n _ => look_refuse_le D P _ hθ0 (by linarith) ha
  have hfin := hits_le_le D P N hθ0 hδ h
  nlinarith

omit [DecidableEq α] in
open scoped Classical in
/-- The ends test, read where the gate stops, reading above `θ` on a rate of at most `θ − δ`. -/
theorem side_above_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P P' : FreeMonoid α → Prop) (N n₀ : ℕ) {θ θ' a δ : ℝ} (hθ1 : θ ≤ 1) (ha : 0 ≤ a)
    (hδ : 0 ≤ δ) (h : D.real {x | P x} ≤ θ - δ) :
    (Measure.pi fun _ : Fin N => D).real
        {b | sideAt θ a n₀ b P (stopLook θ' θ a n₀ b P' P)}
      ≤ (Nat.log 2 (N / n₀) + 2) * a + Real.exp (-2 * N * δ ^ 2) := by
  set ν := Measure.pi fun _ : Fin N => D
  set L : ℕ → Set (Fin N → FreeMonoid α) := fun n =>
    {b | binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card θ
          ((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card < a}
  have hsub : {b | sideAt θ a n₀ b P (stopLook θ' θ a n₀ b P' P)}
      ⊆ (⋃ n ∈ lookSet n₀ N, L n) ∪ {b | θ * N < hitsIn b P N} := by
    intro b hb
    obtain ⟨hmem, hle, -, hT⟩ := stopLook_spec (θ := θ') (θ' := θ) (a := a) (n₀ := n₀) b P' P
    simp only [Set.mem_ofPred_eq, sideAt] at hb
    rcases hr : rateSide θ a n₀ (stopLook θ' θ a n₀ b P' P)
      (hitsIn b P (stopLook θ' θ a n₀ b P' P)) with _ | s <;> rw [hr] at hb
    · rcases hT with hT | ⟨-, hs⟩
      · exact .inr (by simpa [hT] using hb)
      · simp [hr] at hs
    · subst hb
      refine .inl (Set.mem_biUnion hmem ?_)
      obtain ⟨he, hc⟩ := look_set hle b P
      simp only [L, Set.mem_ofPred_eq, hc, ← he]
      unfold rateSide at hr
      split_ifs at hr with h1 h2 h3 <;> simp_all
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have hu := measureReal_biUnion_finset_le (μ := ν) (lookSet n₀ N) L
  have hsum := sum_looks_le (n₀ := n₀) (N := N) ha
    (fun n => ν.real (L n)) fun n _ => look_above_le D P _ hθ1 (by linarith) ha
  have hfin := hits_gt_le D P N hδ h
  linarith

omit [DecidableEq α] in
open scoped Classical in
/-- And reading not above `θ` on a rate of at least `θ + δ`. -/
theorem side_below_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (P P' : FreeMonoid α → Prop) (N n₀ : ℕ) {θ θ' a δ : ℝ} (hθ0 : 0 ≤ θ) (ha : 0 ≤ a)
    (hδ : 0 ≤ δ) (h : θ + δ ≤ D.real {x | P x}) :
    (Measure.pi fun _ : Fin N => D).real
        {b | ¬ sideAt θ a n₀ b P (stopLook θ' θ a n₀ b P' P)}
      ≤ (Nat.log 2 (N / n₀) + 2) * a + Real.exp (-2 * N * δ ^ 2) := by
  set ν := Measure.pi fun _ : Fin N => D
  set L : ℕ → Set (Fin N → FreeMonoid α) := fun n =>
    {b | 1 - binomSfGe (Finset.univ.filter fun i : Fin N => (i : ℕ) < n).card θ
          (((Finset.univ.filter fun i : Fin N => (i : ℕ) < n).filter fun i => P (b i)).card + 1)
          < a}
  have hsub : {b | ¬ sideAt θ a n₀ b P (stopLook θ' θ a n₀ b P' P)}
      ⊆ (⋃ n ∈ lookSet n₀ N, L n) ∪ {b | (hitsIn b P N : ℝ) ≤ θ * N} := by
    intro b hb
    obtain ⟨hmem, hle, -, hT⟩ := stopLook_spec (θ := θ') (θ' := θ) (a := a) (n₀ := n₀) b P' P
    simp only [Set.mem_ofPred_eq, sideAt] at hb
    rcases hr : rateSide θ a n₀ (stopLook θ' θ a n₀ b P' P)
      (hitsIn b P (stopLook θ' θ a n₀ b P' P)) with _ | s <;> rw [hr] at hb
    · rcases hT with hT | ⟨-, hs⟩
      · refine .inr ?_
        rw [hT] at hb
        exact not_lt.1 hb
      · simp [hr] at hs
    · have hs : s = false := by simpa using hb
      subst hs
      refine .inl (Set.mem_biUnion hmem ?_)
      obtain ⟨he, hc⟩ := look_set hle b P
      simp only [L, Set.mem_ofPred_eq, hc, ← he]
      unfold rateSide at hr
      split_ifs at hr with h1 h2 h3 <;> simp_all
  refine (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans ?_)
  have hu := measureReal_biUnion_finset_le (μ := ν) (lookSet n₀ N) L
  have hsum := sum_looks_le (n₀ := n₀) (N := N) ha
    (fun n => ν.real (L n)) fun n _ => look_below_le D P _ hθ0 (by linarith) ha
  have hfin := hits_le_le D P N hθ0 hδ h
  linarith

end Batch

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
  · exact h p c q y he
  · dsimp only at he
    split_ifs at he
    · exact h p c q y he
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj he)
    exact decisiveTarget_learned K R hd

theorem probeStepK_learned {k : ℕ} {s : KState α} {x : FreeMonoid α}
    (h : Learned R s.tree s.edges) :
    Learned R (probeStepK K R k s x).tree (probeStepK K R k s x).edges := by
  simp only [probeStepK]
  split
  · split
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
theorem follow_inr {edges : Edges α} :
    ∀ (cs : List α) (p s : List Bool) (c : α) (i : ℕ), follow edges p cs = .inr (s, c, i) →
      ∃ (hi : i < cs.length), cs[i] = c ∧ edges s c = none
        ∧ ∃ ps, follow edges p (cs.take i) = .inl ps ∧ ps.getLast? = some s
  | [], p, s, c, i, h => by simp [follow] at h
  | c' :: cs, p, s, c, i, h => by
    simp only [follow] at h
    rcases he : edges p c' with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨by simp, rfl, he, [p], by simp [follow], rfl⟩
    · simp only [] at h
      rcases hf : follow edges q cs with ps' | ⟨s', c'', i'⟩ <;> rw [hf] at h
      · simp at h
      · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        obtain ⟨hi, hc, hn, ps, hps, hlast⟩ := follow_inr cs q s' c'' i' hf
        refine ⟨by simpa using hi, by simpa using hc, hn, p :: ps, ?_, ?_⟩
        · simp [follow, he, hps]
        · have hne : ps ≠ [] := by
            intro h0; subst h0; obtain ⟨hl, -⟩ := follow_inl _ _ _ hps; simp at hl
          rw [List.getLast?_cons, hlast]
          simp [hne]

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_edge (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi ps' j, lo < hi → hi - lo ≤ fuel → agrees lo = some true →
      agrees hi = some false → bracketAt (α := α) agrees ps fuel lo hi = .edge ps' j →
      ps' = ps ∧ lo < j ∧ j ≤ hi ∧ agrees (j - 1) = some true ∧ agrees j = some false
  | 0, lo, hi, ps', j, hlt, hf, _, _, _ => by omega
  | fuel + 1, lo, hi, ps', j, hlt, hf, hlo, hhi, h => by
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
          have := bracketAt_edge agrees ps fuel lo _ ps' j this (by omega) hlo hl h
          exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees ps fuel lo _ ps' j this (by omega) hlo hl h
          exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
        · have : (lo + hi) / 2 + 1 < hi := by
            rcases Nat.lt_or_ge ((lo + hi) / 2 + 1) hi with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 + 1 = hi by omega, hhi] at hr; simp at hr
          have := bracketAt_edge agrees ps fuel _ hi ps' j this (by omega) hr hhi h
          exact ⟨this.1, by omega, this.2.2⟩
      · have := bracketAt_edge agrees ps fuel lo _ ps' j (by omega) (by omega) hlo hm h
        exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
      · have := bracketAt_edge agrees ps fuel _ hi ps' j (by omega) (by omega) hm hhi h
        exact ⟨this.1, by omega, this.2.2⟩
    · obtain ⟨rfl, rfl⟩ := Outcome.edge.inj h
      obtain rfl : hi = lo + 1 := by omega
      exact ⟨rfl, by omega, le_rfl, by simpa using hlo, hhi⟩

/-- The outcomes a search ends at. -/
def Outcome.IsSearch : Outcome α → Prop
  | .pair _ | .edge _ _ | .triple _ => True
  | _ => False

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_isSearch (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi, (bracketAt (α := α) agrees ps fuel lo hi).IsSearch
  | 0, _, _ => trivial
  | fuel + 1, lo, hi => by
    simp only [bracketAt]
    split
    · split
      · exact bracketAt_isSearch agrees ps fuel _ _
      · exact bracketAt_isSearch agrees ps fuel _ _
      · split
        all_goals first | trivial | exact bracketAt_isSearch agrees ps fuel _ _
    · trivial

theorem walkCheck_inl {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α} {o : Outcome α}
    (h : walkCheck R t edges k x = .inl o) : ¬ o.IsSearch := by
  unfold walkCheck at h
  split at h
  · obtain rfl := Sum.inl.inj h; exact id
  · split at h
    · obtain rfl := Sum.inl.inj h; exact id
    · split at h
      · obtain rfl := Sum.inl.inj h; exact id
      · split_ifs at h
        · obtain rfl := Sum.inl.inj h; exact id
  · split at h
    · obtain rfl := Sum.inl.inj h; exact id
    · split_ifs at h
      · obtain rfl := Sum.inl.inj h; exact id

/-- A search outcome comes from a decided disagreement. -/
theorem probeOutcome_search {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {o : Outcome α} (h : probeOutcome R t edges k x = o) (ho : o.IsSearch) :
    ∃ ps hi, walkCheck R t edges k x = .inr (ps, hi)
      ∧ bracketAt (agreesAt R t x fun j => ps.getD (j - k) []) ps (hi - k) k hi = o := by
  unfold probeOutcome at h
  rcases hw : walkCheck R t edges k x with o' | ⟨ps, hi⟩ <;> rw [hw] at h
  · simp only [Sum.elim_inl, id] at h
    subst h
    exact absurd ho (walkCheck_inl R hw)
  · exact ⟨ps, hi, rfl, h⟩

theorem walkCheck_inr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {hi : ℕ} (h : walkCheck R t edges k x = .inr (ps, hi)) :
    ∃ p₀, t.sift R.cut (prefixOf x k) = .inl p₀
      ∧ follow edges p₀ ((x.toList.drop k).take (hi - k)) = .inl ps ∧ k < hi
      ∧ hi ≤ x.toList.length
      ∧ agreesAt R t x (fun j => ps.getD (j - k) []) hi = some false := by
  unfold walkCheck at h
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hw] at h
  · simp at h
  · unfold kWalk at hw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
    swap
    · simp at hw
    simp only [] at hw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | ⟨s', c', i⟩ <;> rw [hf] at hw
    · simp at hw
    simp only [KWalk.edge.injEq] at hw
    obtain ⟨rfl, rfl, rfl⟩ := hw
    obtain ⟨hi', hc, hn, qs, hqs, hlast⟩ := follow_inr _ _ _ _ _ hf
    simp only [] at h
    rcases h1 : t.sift R.cut (prefixOf x (k + i + 1)) with _ | _ <;> rw [h1] at h
    swap
    · simp at h
    simp only [] at h
    rcases h2 : t.sift R.cut (prefixOf x (k + i)) with p | _ <;> rw [h2] at h
    swap
    · simp at h
    simp only [] at h
    split_ifs at h with hps
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hwt : walkTo R t edges k x (k + i) = qs := by
      simp [walkTo, hk, show k + i - k = i by omega, hqs]
    rw [hwt]
    obtain ⟨hl, -, -⟩ := follow_inl _ _ _ hqs
    simp only [List.length_take, List.length_drop] at hi' hl
    refine ⟨p₀, rfl, by rw [show k + i - k = i by omega]; exact hqs, ?_, by omega, ?_⟩
    · rcases Nat.eq_zero_or_pos i with rfl | hi0
      · exfalso
        simp only [add_zero, List.take_zero] at hqs h2
        simp [follow] at hqs
        subst hqs
        rw [hk] at h2
        obtain rfl := Sum.inl.inj h2
        simp at hlast
        first | exact hps hlast | exact hps hlast.symm
      · omega
    · have hget : qs.getD (k + i - k) [] = s' := by
        rw [show k + i - k = i by omega]
        rw [List.getLast?_eq_getElem?] at hlast
        rw [List.getD_eq_getElem?_getD, show i = qs.length - 1 by omega, hlast]
        rfl
      simp only [agreesAt, h2, Sum.elim_inl, hget, Option.some.injEq, decide_eq_false_iff_not]
      exact hps
  · simp only [] at h
    rcases hs : t.sift R.cut x with a | b <;> rw [hs] at h
    swap
    · simp at h
    simp only [] at h
    split_ifs at h with ha
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    unfold kWalk at hw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
    swap
    · simp at hw
    simp only [] at hw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | r <;> rw [hf] at hw
    swap
    · simp at hw
    simp only [KWalk.reached.injEq] at hw
    subst hw
    obtain ⟨hlen, hhead, -⟩ := follow_inl _ _ _ hf
    set n := x.toList.length
    simp only [List.length_drop] at hlen
    have hlast : ps''.getLast? = some (ps''.getD (n - k) []) := by
      have hidx : n - k < ps''.length := by omega
      rw [List.getLast?_eq_getElem?, show ps''.length - 1 = n - k by omega,
        List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hidx]
      rfl
    have hkn : k < n := by
      by_contra hkn
      have hxk : prefixOf x k = x := by
        apply FreeMonoid.toList.injective
        simp [prefixOf, List.take_of_length_le (not_lt.1 hkn)]
      rw [hxk, hs] at hk
      obtain rfl := Sum.inl.inj hk
      have : ps''.getD (n - k) [] = a := by rw [show n - k = 0 by omega, hhead]
      exact ha (by rw [hlast, this])
    refine ⟨p₀, rfl, by rw [List.take_of_length_le (by simp; omega)]; exact hf, hkn, le_rfl, ?_⟩
    have : prefixOf x n = x := prefixOf_length x
    simp only [agreesAt, this, hs, Sum.elim_inl, Option.some.injEq, decide_eq_false_iff_not]
    intro he
    exact ha (by rw [hlast, he])

theorem seedStep_ne_dropped {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {T : Tested α}
    (hl : Learned R t edges) {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (h : probeOutcome R t edges k x = .edge ps fd) :
    seedStep K R t pool edges T k x ps fd ≠ .dropped := by
  obtain ⟨ps₀, hi, hw, hb⟩ := probeOutcome_search R h trivial
  obtain ⟨p₀, hk, hf, hkh, hhn, hpn⟩ := walkCheck_inr R hw
  set walkAt : ℕ → List Bool := fun j => ps₀.getD (j - k) [] with hwalk
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  have hpk : agreesAt R t x walkAt k = some true := by
    simp only [agreesAt, hk, Sum.elim_inl, hwalk, Nat.sub_self, hhead, decide_true]
  obtain ⟨rfl, hfd1, hfd2, hfd3, hfd4⟩ :=
    bracketAt_edge (agreesAt R t x walkAt) ps₀ (hi - k) k hi ps fd hkh le_rfl hpk hpn hb
  unfold seedStep
  simp only []
  have hfdn : fd - 1 < x.toList.length := by omega
  rw [List.getElem?_eq_getElem hfdn]
  simp only []
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) (by simp; omega)
  have hidx : ((x.toList.drop k).take (hi - k))[fd - 1 - k]'(by simp; omega)
      = x.toList[fd - 1] := by
    simp only [List.getElem_take, List.getElem_drop]
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

/-- A member is placed at a leaf whose edge by some letter is unlearned, and placed followed by
that letter. -/
theorem member_spec {t : DTree α} {edges : Edges α} {k : ℕ} {x u : FreeMonoid α}
    (h : probeOutcome R t edges k x = .member u) :
    ∃ p c, edges p c = none ∧ t.sift R.cut u = .inl p
      ∧ (t.sift R.cut (u * FreeMonoid.of c)).isLeft := by
  unfold probeOutcome at h
  rcases hw : walkCheck R t edges k x with o | ⟨ps, hi⟩ <;> rw [hw] at h
  swap
  · have := bracketAt_isSearch (α := α) (agreesAt R t x fun j => ps.getD (j - k) []) ps (hi - k)
      k hi
    simp only [Sum.elim_inr] at h
    rw [h] at this
    exact this.elim
  simp only [Sum.elim_inl, id] at h
  subst h
  unfold walkCheck at hw
  rcases hkw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hkw] at hw
  · simp at hw
  · unfold kWalk at hkw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hkw
    swap
    · simp at hkw
    simp only [] at hkw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | ⟨s', c', i⟩ <;> rw [hf] at hkw
    · simp at hkw
    simp only [KWalk.edge.injEq] at hkw
    obtain ⟨rfl, rfl, rfl⟩ := hkw
    obtain ⟨hi', hc, hn, -⟩ := follow_inr _ _ _ _ _ hf
    simp only [] at hw
    rcases h1 : t.sift R.cut (prefixOf x (k + i + 1)) with q | _ <;> rw [h1] at hw
    swap
    · simp at hw
    simp only [] at hw
    rcases h2 : t.sift R.cut (prefixOf x (k + i)) with p | _ <;> rw [h2] at hw
    swap
    · simp at hw
    simp only [] at hw
    split_ifs at hw with hps
    · obtain rfl := Outcome.member.inj (Sum.inl.inj hw)
      simp only [List.length_drop] at hi'
      have hlt : k + i < x.toList.length := by omega
      refine ⟨p, c', hps ▸ hn, h2, ?_⟩
      have hxc : x.toList[k + i] = c' := by rw [← hc]; simp
      rw [← hxc, ← prefixOf_succ hlt, h1]
      rfl
  · simp only [] at hw
    split at hw
    · simp at hw
    · split_ifs at hw <;> simp at hw

end Learned

section Gate

variable (K : StageKnobs α) (R : CutReads α)

/-- A searched draw is one the gate's reading disagrees on. -/
theorem bisected_gateDisagrees {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : Bisected R t edges k x) : gateDisagrees R t edges k x := by
  unfold Bisected at h
  rcases hw : walkCheck R t edges k x with o | ⟨ps, hi⟩ <;> rw [hw] at h
  · simp at h
  clear h
  unfold walkCheck at hw
  unfold gateDisagrees
  rcases hkw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hkw] at hw
  · simp at hw
  · unfold kWalk at hkw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hkw
    swap
    · simp at hkw
    simp only [] at hkw
    have hp0 : place R t x k = p₀ := by simp [place, hk]
    rw [hp0]
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | r <;> rw [hf] at hkw
    · simp at hkw
    · simp
  · simp only [] at hw
    rcases hs : t.sift R.cut x with a | b <;> rw [hs] at hw
    swap
    · simp at hw
    simp only [] at hw
    split_ifs at hw with ha
    unfold kWalk at hkw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hkw
    swap
    · simp at hkw
    simp only [] at hkw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | r <;> rw [hf] at hkw
    swap
    · simp at hkw
    simp only [KWalk.reached.injEq] at hkw
    subst hkw
    have hp0 : place R t x k = p₀ := by simp [place, hk]
    have hpn : place R t x x.toList.length = a := by simp [place, prefixOf_length, hs]
    rw [hp0, hf, hpn]
    exact fun he => ha he.symm

theorem not_bisected_of_ends {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : EndsDeep R t edges k x) : ¬ Bisected R t edges k x := by
  obtain ⟨w, hw, -⟩ := h
  unfold Bisected
  rcases hc : walkCheck R t edges k x with o | ⟨ps, hi⟩
  · simp
  · exfalso
    have hs := bracketAt_isSearch (α := α) (agreesAt R t x fun j => ps.getD (j - k) []) ps
      (hi - k) k hi
    have : probeOutcome R t edges k x
        = bracketAt (agreesAt R t x fun j => ps.getD (j - k) []) ps (hi - k) k hi := by
      simp [probeOutcome, hc]
    rw [← this] at hs
    rcases hw with hw | hw <;> rw [hw] at hs <;> exact hs

theorem bisected_of_pair {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    (h : ∃ j, probeOutcome R t edges k x = .pair j) : Bisected R t edges k x := by
  obtain ⟨j, hj⟩ := h
  obtain ⟨ps, hi, hw, -⟩ := probeOutcome_search R hj trivial
  simp [Bisected, hw]

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
    rw [Finset.prod_ite, Finset.prod_const_one, mul_one, Finset.prod_const,
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

omit [Fintype α] [DecidableEq α] in
open scoped Classical in
theorem hitsIn_congr {N : ℕ} {b b' : Fin N → FreeMonoid α} {P : FreeMonoid α → Prop}
    (h : ∀ i, P (b i) ↔ P (b' i)) (n : ℕ) : hitsIn b P n = hitsIn b' P n := by
  unfold hitsIn
  congr 1
  ext i
  simp [h i]

theorem gate_bad_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k ng n₀ : ℕ)
    (sg : KState α) (hl : Learned R sg.tree sg.edges) {acc θp f a δ : ℝ} (hacc0 : 0 ≤ acc)
    (hacc1 : acc ≤ 1) (hθp0 : 0 ≤ θp) (hθp1 : θp ≤ 1) (hf : 0 ≤ f) (ha : 0 ≤ a) (hδ : 0 ≤ δ) :
    (Measure.pi fun _ : Fin ng => D).real {bg | ¬ RoundAtKHolds K R sg D k acc θp f a δ n₀ bg}
      ≤ (3 * (Nat.log 2 (ng / n₀) + 2) + 1) * a + 2 * Real.exp (-2 * ng * δ ^ 2)
        + Real.exp (-(min n₀ ng : ℕ) * δ) := by
  classical
  set ν := Measure.pi fun _ : Fin ng => D
  set t := sg.tree
  set e := sg.edges
  set Pd := fun x => gateDisagrees R t e k x
  set Ag := fun x => ¬ gateDisagrees R t e k x
  set En := EndsDeep R t e k
  set Bi := Bisected R t e k
  set Pr := fun x => ∃ j, probeOutcome R t e k x = .pair j
  set θe := min (2 * ((t.depth - 1 : ℕ) : ℝ) * f) 1
  have hθe0 : 0 ≤ θe := le_min (by positivity) zero_le_one
  have hθe1 : θe ≤ 1 := min_le_right _ _
  set T := fun bg : Fin ng → FreeMonoid α => stopLook acc θe a n₀ bg Ag En
  set B12 := {bg : Fin ng → FreeMonoid α |
    (acc * T bg ≤ hitsIn bg Ag (T bg) ∧ 1 - acc + δ < D.real {x | Pd x})
      ∨ ((hitsIn bg Ag (T bg) : ℝ) < acc * T bg ∧ D.real {x | Pd x} < 1 - acc - δ)}
  set B34 := {bg : Fin ng → FreeMonoid α |
    (sideAt θe a n₀ bg En (T bg) ∧ D.real {x | En x} < θe - δ)
      ∨ (¬ sideAt θe a n₀ bg En (T bg) ∧ θe + δ < D.real {x | En x})}
  set B5 := {bg : Fin ng → FreeMonoid α |
    binomSfGe (hitsIn bg Bi (T bg)) θp (hitsIn bg Pr (T bg)) < a
      ∧ D.real {x | Pr x} ≤ θp * D.real {x | Bi x}}
  set B6 := {bg : Fin ng → FreeMonoid α |
    (∀ i : Fin ng, (i : ℕ) < n₀ → ¬ Bi (bg i)) ∧ δ < D.real {x | Bi x}}
  have hsub : {bg | ¬ RoundAtKHolds K R sg D k acc θp f a δ n₀ bg} ⊆ (B12 ∪ B34) ∪ (B5 ∪ B6) := by
    intro bg hb
    simp only [Set.mem_ofPred_eq, RoundAtKHolds, not_and_or, Classical.not_imp, not_le,
      not_lt] at hb
    rcases hb with (⟨h1, h2⟩ | ⟨h1, h2⟩) | (⟨h1, h2⟩ | ⟨h1, h2⟩) | (⟨h1, h2⟩ | ⟨h1, h2, h3⟩) |
      h4 | h5
    · exact .inl (.inl (.inl ⟨h1, h2⟩))
    · exact .inl (.inl (.inr ⟨h1, h2⟩))
    · exact .inl (.inr (.inl ⟨h1, by linarith⟩))
    · exact .inl (.inr (.inr ⟨h1, h2⟩))
    · exact .inr (.inl ⟨h1, h2⟩)
    · refine .inr (.inr ⟨fun i hi => h2 i ?_, h3⟩)
      have := (stopLook_spec (θ := acc) (θ' := θe) (a := a) (n₀ := n₀) bg Ag En).2.2.1
      have hi' : (i : ℕ) < ng := i.2
      exact lt_of_lt_of_le (lt_min hi hi') this
    · exfalso
      simp only [not_forall] at h4
      obtain ⟨i, ps, fd, ho, hd⟩ := h4
      exact hd (seedStep_ne_dropped K R hl ho)
    · exfalso
      simp only [not_forall, not_exists, not_and] at h5
      obtain ⟨i, u, ho, hm⟩ := h5
      obtain ⟨p, c, h1, h2, h3⟩ := member_spec R ho
      exact hm p c h1 h2 h3
  have hcomp : D.real {x | Ag x} = 1 - D.real {x | Pd x} := by
    rw [show {x | Ag x} = {x | Pd x}ᶜ from rfl,
      measureReal_compl (Set.to_countable _).measurableSet, probReal_univ]
  have hB12 : ν.real B12 ≤ 2 * (Nat.log 2 (ng / n₀) + 2) * a + Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h2 : 1 - acc + δ < D.real {x | Pd x}
    · refine le_trans (measureReal_mono fun bg hb => ?_)
        (gate_pass_le D Ag En ng n₀ (θ' := θe) hacc1 ha hδ (by rw [hcomp]; linarith))
      rcases hb with hb | hb
      · exact hb.1
      · exfalso; linarith [hb.2]
    · by_cases h3 : D.real {x | Pd x} < 1 - acc - δ
      · refine le_trans (measureReal_mono fun bg hb => ?_)
          (gate_refuse_le D Ag En ng n₀ (θ' := θe) hacc0 ha hδ (by rw [hcomp]; linarith))
        rcases hb with hb | hb
        · exact absurd hb.2 h2
        · exact hb.1
      · rw [show B12 = ∅ from Set.eq_empty_of_forall_notMem fun bg hb => by
          rcases hb with hb | hb
          · exact h2 hb.2
          · exact h3 hb.2]
        simp only [measureReal_empty]
        positivity
  have hB34 : ν.real B34 ≤ (Nat.log 2 (ng / n₀) + 2) * a + Real.exp (-2 * ng * δ ^ 2) := by
    by_cases h2 : D.real {x | En x} < θe - δ
    · refine le_trans (measureReal_mono fun bg hb => ?_)
        (side_above_le D En Ag ng n₀ (θ' := acc) hθe1 ha hδ h2.le)
      rcases hb with hb | hb
      · exact hb.1
      · exfalso; linarith [hb.2]
    · by_cases h3 : θe + δ < D.real {x | En x}
      · refine le_trans (measureReal_mono fun bg hb => ?_)
          (side_below_le D En Ag ng n₀ (θ' := acc) hθe0 ha hδ h3.le)
        rcases hb with hb | hb
        · exact absurd hb.2 h2
        · exact hb.1
      · rw [show B34 = ∅ from Set.eq_empty_of_forall_notMem fun bg hb => by
          rcases hb with hb | hb
          · exact h2 hb.2
          · exact h3 hb.2]
        simp only [measureReal_empty]
        positivity
  have hB5 : ν.real B5 ≤ a := by
    by_cases h : D.real {x | Pr x} ≤ θp * D.real {x | Bi x}
    · refine le_trans (measureReal_mono fun bg hb => ?_)
        (pair_test_le D Bi Pr (fun x hx => bisected_of_pair R hx) T (fun b b' h1 h2 => ?_) hθp0
          hθp1 ha h)
      · have hP : ∀ (P : FreeMonoid α → Prop), (∀ x, Bi x → ¬ P x) →
            ∀ i, P (bg i) ↔ P (bg i) := fun _ _ _ => Iff.rfl
        have e1 : hitsIn bg Bi (T bg)
            = (Finset.univ.filter fun i : Fin ng => (i : ℕ) < T bg ∧ Bi (bg i)).card := rfl
        have e2 : hitsIn bg Pr (T bg)
            = ((Finset.univ.filter fun i : Fin ng => (i : ℕ) < T bg ∧ Bi (bg i)).filter
                fun i => Pr (bg i)).card := by
          unfold hitsIn
          rw [Finset.filter_filter]
          congr 1
          ext i
          simp only [Finset.mem_filter, Finset.mem_univ, true_and]
          exact ⟨fun h => ⟨⟨h.1, bisected_of_pair R h.2⟩, h.2⟩, fun h => ⟨h.1.1, h.2⟩⟩
        simp only [Set.mem_ofPred_eq]
        rw [← e1, ← e2]
        exact hb.1
      · have hc : ∀ (P : FreeMonoid α → Prop), (∀ x, Bi x → ¬ P x) →
            ∀ i, P (b i) ↔ P (b' i) := by
          intro P hP i
          by_cases hi : Bi (b i)
          · exact ⟨fun h => absurd h (hP _ hi), fun h => absurd h (hP _ ((h1 i).1 hi))⟩
          · rw [h2 i hi]
        have hag := hc Ag fun x hx hx' => hx' (bisected_gateDisagrees R hx)
        have hen := hc En fun x hx hx' => not_bisected_of_ends R hx' hx
        simp only [T, stopLook, hitsIn_congr hag, hitsIn_congr hen]
    · rw [show B5 = ∅ from Set.eq_empty_of_forall_notMem fun bg hb => h hb.2]
      simp only [measureReal_empty]
      exact ha
  have hB6 := miss_first_le D Bi ng n₀ hδ
  have h1 := measureReal_union_le (μ := ν) B12 B34
  have h2 := measureReal_union_le (μ := ν) B5 B6
  have h3 := measureReal_union_le (μ := ν) (B12 ∪ B34) (B5 ∪ B6)
  refine (measureReal_mono hsub).trans ?_
  nlinarith

end Gate

theorem round_at_k_batch (K : StageKnobs α) (R : CutReads α) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (k ng n₀ : ℕ) (seed probes : List (FreeMonoid α))
    {acc θp f a δ : ℝ} (hacc0 : 0 ≤ acc) (hacc1 : acc ≤ 1) (hθp0 : 0 ≤ θp) (hθp1 : θp ≤ 1)
    (hf : 0 ≤ f) (ha : 0 ≤ a) (hδ : 0 ≤ δ) :
    (Measure.pi fun _ : Fin ng => D).real
        {bg | ¬ RoundAtKHolds K R (runPassK K R k (initialK K R seed) probes) D k acc θp f a δ
          n₀ bg}
      ≤ (3 * (Nat.log 2 (ng / n₀) + 2) + 1) * a + 2 * Real.exp (-2 * ng * δ ^ 2)
        + Real.exp (-(min n₀ ng : ℕ) * δ) :=
  gate_bad_le K R D k ng n₀ _ (runPassK_learned K R k seed probes) hacc0 hacc1 hθp0 hθp1 hf ha hδ

end OrthoDFA
