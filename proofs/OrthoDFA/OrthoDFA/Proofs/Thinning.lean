import OrthoDFA.LowLevelClaims.Clustering
import Mathlib.Probability.ConditionalProbability
import Mathlib.Data.List.GetD

/-!
# Splitting an i.i.d. stream by a set

The draws of an i.i.d. stream that fall in `A`, and those that fall in a disjoint `R`, are two
independent i.i.d. streams, of laws `D[|A]` and `D[|R]`.
-/

namespace OrthoDFA.Thin

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {S : Type*} [Stringlike S]

section Lists

/-- `s`, then `y`. -/
def consStream (s : S) (y : ℕ → S) : ℕ → S := fun n => Nat.casesOn n s y

lemma entries_zero (A : Set S) (w : ℕ → S) : drawsIn A w 0 = [] := by simp [drawsIn]

open scoped Classical in
lemma entries_succ (A : Set S) (w : ℕ → S) (n : ℕ) :
    drawsIn A w (n + 1) = drawsIn A w n ++ (if w n ∈ A then [w n] else []) := by
  simp only [drawsIn, List.range_succ, List.map_append, List.map_cons, List.map_nil,
    List.filter_append, List.filter_cons, List.filter_nil]
  by_cases h : w n ∈ A <;> simp [h]

lemma entries_prefix (A : Set S) (w : ℕ → S) {n n' : ℕ} (h : n ≤ n') :
    drawsIn A w n <+: drawsIn A w n' := by
  induction n', h using Nat.le_induction with
  | base => exact List.prefix_refl _
  | succ k _ ih =>
    rw [entries_succ]
    exact ih.trans (List.prefix_append _ _)

open scoped Classical in
lemma entries_cons (A : Set S) (s : S) (y : ℕ → S) (n : ℕ) :
    drawsIn A (consStream s y) (n + 1) = (if s ∈ A then [s] else []) ++ drawsIn A y n := by
  induction n with
  | zero => by_cases h : s ∈ A <;> simp [entries_succ, entries_zero, consStream, h]
  | succ k ih =>
    rw [entries_succ, ih, entries_succ, List.append_assoc]
    rfl

lemma entries_eq_ofFn (A : Set S) (w : ℕ → S) (n : ℕ) :
    drawsIn A w n = drawsIn A (fun k => if h : k < n then w k else w k) n := by
  simp

lemma sub_eq_getD (A : Set S) (d : S) (w : ℕ → S) {k n : ℕ}
    (hk : k < (drawsIn A w n).length) : nthDrawIn A d w k = (drawsIn A w n).getD k d := by
  classical
  have h : ∃ n, k < (drawsIn A w n).length := ⟨n, hk⟩
  rw [nthDrawIn, dif_pos h]
  have hmin := Nat.find_spec h
  rcases le_total (Nat.find h) n with hle | hle
  · obtain ⟨t, ht⟩ := entries_prefix A w hle
    rw [← ht, List.getD_append _ _ _ _ hmin]
  · obtain ⟨t, ht⟩ := entries_prefix A w hle
    rw [← ht, List.getD_append _ _ _ _ hk]

/-- `α` and `β` open the draws in `A` and in `R`. -/
def opens (A R : Set S) (α β : List S) : Set (ℕ → S) :=
  {w | ∃ n, α <+: drawsIn A w n ∧ β <+: drawsIn R w n}

open scoped Classical in
/-- What is left of `α` once a draw `s` has been read. -/
noncomputable def step (A : Set S) (s : S) (α : List S) : Option (List S) :=
  if s ∈ A then
    match α with
    | [] => some []
    | a :: t => if a = s then some t else none
  else some α

lemma prefix_step_iff (A : Set S) (s : S) (α E : List S) [Decidable (s ∈ A)] :
    α <+: (if s ∈ A then [s] else []) ++ E ↔ ∃ α', step A s α = some α' ∧ α' <+: E := by
  classical
  by_cases hs : s ∈ A
  · simp only [hs, if_true, List.singleton_append, step]
    cases α with
    | nil => simp
    | cons a t =>
      rw [List.cons_prefix_cons]
      by_cases ha : a = s <;> simp [ha]
  · simp [hs, step]

lemma opens_cons (A R : Set S) (hAR : Disjoint A R) (α β : List S) (s : S) (y : ℕ → S) :
    consStream s y ∈ opens A R α β ↔ ∃ α' β', step A s α = some α' ∧ step R s β = some β'
      ∧ y ∈ opens A R α' β' := by
  classical
  constructor
  · rintro ⟨n, hα, hβ⟩
    cases n with
    | zero =>
      rw [entries_zero, List.prefix_nil] at hα hβ
      subst hα; subst hβ
      refine ⟨[], [], ?_, ?_, 0, by simp [entries_zero], by simp [entries_zero]⟩ <;>
        · unfold step; split_ifs <;> rfl
    | succ k =>
      rw [entries_cons] at hα hβ
      obtain ⟨α', h1, h2⟩ := (prefix_step_iff A s α _).1 hα
      obtain ⟨β', h3, h4⟩ := (prefix_step_iff R s β _).1 hβ
      exact ⟨α', β', h1, h3, k, h2, h4⟩
  · rintro ⟨α', β', h1, h3, n, h2, h4⟩
    refine ⟨n + 1, ?_, ?_⟩
    · rw [entries_cons]; exact (prefix_step_iff A s α _).2 ⟨α', h1, h2⟩
    · rw [entries_cons]; exact (prefix_step_iff R s β _).2 ⟨β', h3, h4⟩

end Lists

section Measure


/-- The law of an i.i.d. stream. -/
noncomputable abbrev iidS (D : Measure S) : Measure (ℕ → S) := Measure.infinitePi fun _ : ℕ => D

lemma measurableSet_of_countable' (W : Set S) : MeasurableSet W :=
  (Set.to_countable W).measurableSet

/-- Every event about the first `n` draws is measurable: they take countably many values. -/
lemma measurableSet_firstN {X : Type*} [MeasurableSpace X] [MeasurableSingletonClass X]
    [Countable X] (n : ℕ) (P : (Fin n → X) → Prop) :
    MeasurableSet {w : ℕ → X | P (fun i : Fin n => w i.val)} := by
  have hmeas : Measurable (fun (w : ℕ → X) (i : Fin n) => w i.val) := by fun_prop
  exact hmeas (Set.to_countable {t : Fin n → X | P t}).measurableSet

lemma entries_congr (A : Set S) {w w' : ℕ → S} {n : ℕ} (h : ∀ k < n, w k = w' k) :
    drawsIn A w n = drawsIn A w' n := by
  unfold drawsIn
  congr 1
  exact List.map_congr_left (fun k hk => h k (List.mem_range.1 hk))

lemma measurableSet_opens (A R : Set S) (α β : List S) : MeasurableSet (opens A R α β) := by
  classical
  have hcov : opens A R α β = ⋃ n : ℕ, {w : ℕ → S | (fun t : Fin n → S =>
      ∃ w' : ℕ → S, (∀ i : Fin n, w' i.val = t i)
        ∧ α <+: drawsIn A w' n ∧ β <+: drawsIn R w' n) (fun i : Fin n => w i.val)} := by
    ext w
    simp only [opens, Set.mem_iUnion]
    refine exists_congr (fun n => ⟨fun h => ⟨w, fun _ => rfl, h⟩, ?_⟩)
    rintro ⟨w', hw', hα, hβ⟩
    have e : ∀ B : Set S, drawsIn B w' n = drawsIn B w n :=
      fun B => entries_congr B (fun k hk => hw' ⟨k, hk⟩)
    rw [e A] at hα
    rw [e R] at hβ
    exact ⟨hα, hβ⟩
  rw [hcov]
  exact MeasurableSet.iUnion (fun n => measurableSet_firstN n (fun t : Fin n → S =>
    ∃ w' : ℕ → S, (∀ i : Fin n, w' i.val = t i) ∧ α <+: drawsIn A w' n ∧ β <+: drawsIn R w' n))

lemma measurable_consStream : Measurable (fun p : S × (ℕ → S) => consStream p.1 p.2) := by
  refine measurable_pi_lambda _ (fun n => ?_)
  cases n with
  | zero => exact measurable_fst
  | succ k => exact (measurable_pi_apply k).comp measurable_snd

/-- An i.i.d. stream is its first draw, then an independent i.i.d. stream. -/
lemma map_consStream (D : Measure S) [IsProbabilityMeasure D] :
    Measure.map (fun p : S × (ℕ → S) => consStream p.1 p.2) (D.prod (iidS D)) = iidS D := by
  classical
  refine Measure.eq_infinitePi _ (fun s t ht => ?_)
  rw [Measure.map_apply measurable_consStream
    (MeasurableSet.pi s.countable_toSet (fun i _ => ht i))]
  have hpre : (fun p : S × (ℕ → S) => consStream p.1 p.2) ⁻¹' (Set.pi ↑s t)
      = (if 0 ∈ s then t 0 else Set.univ)
        ×ˢ Set.pi ↑(s.preimage Nat.succ Nat.succ_injective.injOn) (fun j => t (j + 1)) := by
    ext ⟨a, y⟩
    simp only [Set.mem_preimage, Set.mem_pi, Finset.mem_coe, Set.mem_prod,
      Finset.mem_preimage]
    constructor
    · intro h
      refine ⟨?_, fun j hj => h (j + 1) hj⟩
      split_ifs with h0
      · exact h 0 h0
      · trivial
    · rintro ⟨h0, hs⟩ i hi
      cases i with
      | zero => show a ∈ t 0; simpa [hi] using h0
      | succ j => exact hs j hi
  rw [hpre, Measure.prod_prod, Measure.infinitePi_pi _ (fun i _ => ht _)]
  rw [← Finset.prod_filter_mul_prod_filter_not s (fun i => i = 0)]
  congr 1
  · by_cases h0 : 0 ∈ s
    · rw [if_pos h0, Finset.filter_eq' s 0, if_pos h0, Finset.prod_singleton]
    · rw [if_neg h0, measure_univ, Finset.filter_eq' s 0, if_neg h0, Finset.prod_empty]
  · have hset : s.filter (fun i => ¬ i = 0)
        = (s.preimage Nat.succ Nat.succ_injective.injOn).map ⟨Nat.succ, Nat.succ_injective⟩ := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_map, Finset.mem_preimage,
        Function.Embedding.coeFn_mk]
      constructor
      · rintro ⟨hi, hne⟩
        have e : (i - 1).succ = i := by omega
        exact ⟨i - 1, by rw [e]; exact hi, e⟩
      · rintro ⟨j, hj, rfl⟩
        exact ⟨hj, Nat.succ_ne_zero j⟩
    rw [hset, Finset.prod_map]
    rfl

lemma step_of_mem {A R : Set S} (hAR : Disjoint A R) {s : S} (hs : s ∈ A) (β : List S) :
    step R s β = some β := by
  classical
  have : s ∉ R := Set.disjoint_left.1 hAR hs
  simp [step, this]

lemma step_of_not_mem {A : Set S} {s : S} (hs : s ∉ A) (α : List S) :
    step A s α = some α := by
  classical
  simp [step, hs]

open scoped Classical in
/-- What a first draw `s` leaves of the chance of opening on `α`, given that of each rest. -/
noncomputable def hval (s : S) (α : List S) (g : List S → ℝ≥0∞) : ℝ≥0∞ :=
  match α with
  | [] => g []
  | a :: t => if a = s then g t else 0

/-- The part of that chance carried by first draws in `A`. -/
noncomputable def aterm (D : Measure S) (A : Set S) (α : List S) (g : List S → ℝ≥0∞) : ℝ≥0∞ :=
  match α with
  | [] => D A * g []
  | a :: t => A.indicator (fun s => D {s}) a * g t

lemma tsum_aterm (D : Measure S) (A : Set S) (α : List S) (g : List S → ℝ≥0∞)
    (H : S → ℝ≥0∞) (hH : ∀ s ∈ A, H s = hval s α g) :
    ∑' s, A.indicator (fun s => H s * D {s}) s = aterm D A α g := by
  classical
  have hmA : MeasurableSet A := measurableSet_of_countable' A
  cases α with
  | nil =>
    have : ∀ s, A.indicator (fun s => H s * D {s}) s = g [] * A.indicator (fun s => D {s}) s := by
      intro s
      by_cases hs : s ∈ A
      · rw [Set.indicator_of_mem hs, Set.indicator_of_mem hs, hH s hs, hval, mul_comm]
      · simp [hs]
    simp only [this, ENNReal.tsum_mul_left, Measure.tsum_indicator_apply_singleton D A hmA, aterm]
    exact mul_comm _ _
  | cons a t =>
    rw [tsum_eq_single a]
    · by_cases ha : a ∈ A
      · rw [Set.indicator_of_mem ha, hH a ha]
        simp [hval, aterm, ha, mul_comm]
      · simp [ha, aterm]
    · intro s hs
      by_cases hsA : s ∈ A
      · rw [Set.indicator_of_mem hsA, hH s hsA, hval, if_neg (Ne.symm hs), zero_mul]
      · simp [hsA]

/-- The chance a stream opens on `α` and `β`, read off its first draw. -/
lemma measure_opens_eq (D : Measure S) [IsProbabilityMeasure D] (A R : Set S)
    (hAR : Disjoint A R) (α β : List S) :
    iidS D (opens A R α β)
      = aterm D A α (fun α' => iidS D (opens A R α' β))
        + aterm D R β (fun β' => iidS D (opens A R α β'))
        + D (A ∪ R)ᶜ * iidS D (opens A R α β) := by
  classical
  set H : S → ℝ≥0∞ := fun s => iidS D {y | consStream s y ∈ opens A R α β} with hH
  have hsplit : iidS D (opens A R α β) = ∑' s, H s * D {s} := by
    rw [← map_consStream D, Measure.map_apply measurable_consStream (measurableSet_opens _ _ _ _),
      Measure.prod_apply (measurable_consStream (measurableSet_opens _ _ _ _)),
      lintegral_countable']
    rfl
  have hHA : ∀ s ∈ A, H s = hval s α (fun α' => iidS D (opens A R α' β)) := by
    intro s hs
    simp only [hH]
    cases α with
    | nil =>
      refine congrArg (iidS D) (Set.ext fun y => ?_)
      rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_mem hAR hs β]
      simp [step, hs]
    | cons a t =>
      by_cases ha : a = s
      · simp only [hval, if_pos ha]
        refine congrArg (iidS D) (Set.ext fun y => ?_)
        rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_mem hAR hs β]
        simp [step, hs, ha]
      · simp only [hval, if_neg ha]
        rw [show {y : ℕ → S | consStream s y ∈ opens A R (a :: t) β} = ∅ by
          ext y
          rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_mem hAR hs β]
          simp [step, hs, ha]]
        exact measure_empty
  have hHR : ∀ s ∈ R, H s = hval s β (fun β' => iidS D (opens A R α β')) := by
    intro s hs
    have hsA : s ∉ A := fun h => Set.disjoint_left.1 hAR h hs
    simp only [hH]
    cases β with
    | nil =>
      refine congrArg (iidS D) (Set.ext fun y => ?_)
      rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_not_mem hsA α]
      simp [step, hs]
    | cons b u =>
      by_cases hb : b = s
      · simp only [hval, if_pos hb]
        refine congrArg (iidS D) (Set.ext fun y => ?_)
        rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_not_mem hsA α]
        simp [step, hs, hb]
      · simp only [hval, if_neg hb]
        rw [show {y : ℕ → S | consStream s y ∈ opens A R α (b :: u)} = ∅ by
          ext y
          rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_not_mem hsA α]
          simp [step, hs, hb]]
        exact measure_empty
  have hHZ : ∀ s ∈ (A ∪ R)ᶜ, H s = iidS D (opens A R α β) := by
    intro s hs
    simp only [Set.mem_compl_iff, Set.mem_union, not_or] at hs
    simp only [hH]
    refine congrArg (iidS D) (Set.ext fun y => ?_)
    rw [Set.mem_setOf_eq, opens_cons A R hAR, step_of_not_mem hs.1 α, step_of_not_mem hs.2 β]
    simp
  have hpoint : ∀ s, H s * D {s} = A.indicator (fun s => H s * D {s}) s
      + R.indicator (fun s => H s * D {s}) s + (A ∪ R)ᶜ.indicator (fun s => H s * D {s}) s := by
    intro s
    by_cases hA : s ∈ A
    · have hR : s ∉ R := Set.disjoint_left.1 hAR hA
      simp [hA, hR]
    · by_cases hR : s ∈ R
      · simp [hA, hR]
      · have hZ : s ∈ (A ∪ R)ᶜ := by simp [hA, hR]
        rw [Set.indicator_of_notMem hA, Set.indicator_of_notMem hR, Set.indicator_of_mem hZ]
        simp
  have hmZ : MeasurableSet (A ∪ R)ᶜ := measurableSet_of_countable' _
  have hsumZ : ∑' s, (A ∪ R)ᶜ.indicator (fun s => H s * D {s}) s
      = D (A ∪ R)ᶜ * iidS D (opens A R α β) := by
    have : ∀ s, (A ∪ R)ᶜ.indicator (fun s => H s * D {s}) s
        = iidS D (opens A R α β) * (A ∪ R)ᶜ.indicator (fun s => D {s}) s := by
      intro s
      by_cases hs : s ∈ (A ∪ R)ᶜ
      · rw [Set.indicator_of_mem hs, Set.indicator_of_mem hs, hHZ s hs, mul_comm]
      · rw [Set.indicator_of_notMem hs, Set.indicator_of_notMem hs, mul_zero]
    simp only [this, ENNReal.tsum_mul_left, Measure.tsum_indicator_apply_singleton D _ hmZ]
    exact mul_comm _ _
  conv_lhs => rw [hsplit]
  rw [tsum_congr hpoint, ENNReal.tsum_add, ENNReal.tsum_add, tsum_aterm D A α _ H hHA,
    tsum_aterm D R β _ H hHR, hsumZ]

lemma indicator_eq_mul_cond (D : Measure S) (A : Set S) (hA0 : D A ≠ 0) (hAt : D A ≠ ∞)
    (a : S) : A.indicator (fun s => D {s}) a = D A * D[|A] {a} := by
  rw [cond_apply (measurableSet_of_countable' A)]
  by_cases ha : a ∈ A
  · rw [Set.indicator_of_mem ha, Set.inter_eq_right.2 (Set.singleton_subset_iff.2 ha),
      ← mul_assoc, ENNReal.mul_inv_cancel hA0 hAt, one_mul]
  · rw [Set.indicator_of_notMem ha, Set.inter_singleton_eq_empty.2 ha, measure_empty, mul_zero,
      mul_zero]

/-- The chance the stream opens on `α` and `β`: their draws' conditional chances. -/
noncomputable def openChance (D : Measure S) (A R : Set S) (α β : List S) : ℝ≥0∞ :=
  (α.map fun a => D[|A] {a}).prod * (β.map fun b => D[|R] {b}).prod

lemma opens_nil_nil (A R : Set S) : opens A R [] [] = Set.univ := by
  ext w
  exact ⟨fun _ => trivial, fun _ => ⟨0, List.nil_prefix, List.nil_prefix⟩⟩

theorem measure_opens (D : Measure S) [IsProbabilityMeasure D] (A R : Set S)
    (hAR : Disjoint A R) (hA : D A ≠ 0) (hR : D R ≠ 0) (α β : List S) :
    iidS D (opens A R α β) = openChance D A R α β := by
  have hAt : D A ≠ ∞ := measure_ne_top D A
  have hRt : D R ≠ ∞ := measure_ne_top D R
  have hsum : D A + D R + D (A ∪ R)ᶜ = 1 := by
    rw [← measure_union hAR (measurableSet_of_countable' R),
      measure_add_measure_compl (measurableSet_of_countable' _), measure_univ]
  have hfin : ∀ α β, iidS D (opens A R α β) ≠ ∞ := fun _ _ => measure_ne_top _ _
  have hconsA : ∀ a t β, openChance D A R (a :: t) β = D[|A] {a} * openChance D A R t β := by
    intro a t β
    simp only [openChance, List.map_cons, List.prod_cons, mul_assoc]
  have hconsR : ∀ α b u, openChance D A R α (b :: u) = D[|R] {b} * openChance D A R α u := by
    intro α b u
    simp only [openChance, List.map_cons, List.prod_cons]
    ring
  induction h : α.length + β.length using Nat.strong_induction_on generalizing α β with
  | _ n ih =>
  have heq := measure_opens_eq D A R hAR α β
  set G := iidS D (opens A R α β) with hG
  cases α with
  | nil =>
    cases β with
    | nil =>
      rw [hG, opens_nil_nil, measure_univ]
      simp [openChance]
    | cons b u =>
      have hIH := ih ([].length + u.length) (by simp at h ⊢; omega) [] u rfl
      simp only [aterm] at heq
      rw [indicator_eq_mul_cond D R hR hRt, hIH] at heq
      have h1 : D R * G + (D A * G + D (A ∪ R)ᶜ * G) = G := by
        rw [← add_assoc, ← add_mul, ← add_mul, add_comm (D R), hsum, one_mul]
      have h2 : D R * G = D R * (D[|R] {b} * openChance D A R [] u) := by
        have h3 : D R * G + (D A * G + D (A ∪ R)ᶜ * G)
            = D R * (D[|R] {b} * openChance D A R [] u) + (D A * G + D (A ∪ R)ᶜ * G) := by
          rw [h1]; conv_lhs => rw [heq]
          ring
        exact (ENNReal.add_left_inj (by
          exact ENNReal.add_ne_top.2 ⟨ENNReal.mul_ne_top hAt (hfin _ _),
            ENNReal.mul_ne_top (measure_ne_top _ _) (hfin _ _)⟩)).1 h3
      rw [hconsR, ← (ENNReal.mul_right_inj hR hRt).1 h2]
  | cons a t =>
    cases β with
    | nil =>
      have hIH := ih (t.length + [].length) (by simp at h ⊢; omega) t [] rfl
      simp only [aterm] at heq
      rw [indicator_eq_mul_cond D A hA hAt, hIH] at heq
      have h1 : D A * G + (D R * G + D (A ∪ R)ᶜ * G) = G := by
        rw [← add_assoc, ← add_mul, ← add_mul, hsum, one_mul]
      have h2 : D A * G = D A * (D[|A] {a} * openChance D A R t []) := by
        have h3 : D A * G + (D R * G + D (A ∪ R)ᶜ * G)
            = D A * (D[|A] {a} * openChance D A R t []) + (D R * G + D (A ∪ R)ᶜ * G) := by
          rw [h1]; conv_lhs => rw [heq]
          ring
        exact (ENNReal.add_left_inj (by
          exact ENNReal.add_ne_top.2 ⟨ENNReal.mul_ne_top hRt (hfin _ _),
            ENNReal.mul_ne_top (measure_ne_top _ _) (hfin _ _)⟩)).1 h3
      rw [hconsA, ← (ENNReal.mul_right_inj hA hAt).1 h2]
    | cons b u =>
      have hIH1 := ih (t.length + (b :: u).length) (by simp at h ⊢; omega) t (b :: u) rfl
      have hIH2 := ih ((a :: t).length + u.length) (by simp at h ⊢; omega) (a :: t) u rfl
      simp only [aterm] at heq
      rw [indicator_eq_mul_cond D A hA hAt, indicator_eq_mul_cond D R hR hRt, hIH1, hIH2] at heq
      have hf1 := hconsA a t (b :: u)
      have hf2 := hconsR (a :: t) b u
      have h1 : (D A + D R) * G + D (A ∪ R)ᶜ * G = G := by
        rw [← add_mul, hsum, one_mul]
      have h2 : (D A + D R) * G = (D A + D R) * openChance D A R (a :: t) (b :: u) := by
        have h3 : (D A + D R) * G + D (A ∪ R)ᶜ * G
            = (D A + D R) * openChance D A R (a :: t) (b :: u) + D (A ∪ R)ᶜ * G := by
          rw [h1]; conv_lhs => rw [heq]
          have e1 : D A * openChance D A R (a :: t) (b :: u)
              = D A * (D[|A] {a} * openChance D A R t (b :: u)) := by rw [hf1]
          have e2 : D R * openChance D A R (a :: t) (b :: u)
              = D R * (D[|R] {b} * openChance D A R (a :: t) u) := by rw [hf2]
          rw [add_mul, e1, e2]
          ring
        exact (ENNReal.add_left_inj (ENNReal.mul_ne_top (measure_ne_top _ _) (hfin _ _))).1 h3
      exact (ENNReal.mul_right_inj (by simp [hA]) (ENNReal.add_ne_top.2 ⟨hAt, hRt⟩)).1 h2

open scoped Classical in
lemma length_entries_succ (A : Set S) (w : ℕ → S) (n : ℕ) :
    (drawsIn A w (n + 1)).length = (drawsIn A w n).length + if w n ∈ A then 1 else 0 := by
  rw [entries_succ, List.length_append]
  split_ifs <;> simp

/-- A stream whose draws land in `A` infinitely often has every count of them. -/
lemma entries_unbounded (A : Set S) (w : ℕ → S) (hinf : ∀ N, ∃ i, N ≤ i ∧ w i ∈ A) (k : ℕ) :
    ∃ n, k < (drawsIn A w n).length := by
  classical
  induction k with
  | zero =>
    obtain ⟨i, -, hi⟩ := hinf 0
    exact ⟨i + 1, by rw [length_entries_succ, if_pos hi]; omega⟩
  | succ k ih =>
    obtain ⟨n, hn⟩ := ih
    obtain ⟨i, hi, hiA⟩ := hinf n
    refine ⟨i + 1, ?_⟩
    have hmono := (entries_prefix A w hi).length_le
    rw [length_entries_succ, if_pos hiA]
    omega

/-- With positive mass on `A`, a stream lands in `A` infinitely often. -/
lemma ae_entries_unbounded (D : Measure S) [IsProbabilityMeasure D] (A : Set S) (hA : D A ≠ 0) :
    ∀ᵐ w ∂(iidS D), ∀ k, ∃ n, k < (drawsIn A w n).length := by
  have hlt : D Aᶜ < 1 := by
    rw [measure_compl (measurableSet_of_countable' A) (measure_ne_top D A), measure_univ]
    exact ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hA
  have hN : ∀ N : ℕ, iidS D {w | ∀ i, N ≤ i → w i ∉ A} = 0 := by
    intro N
    refine le_antisymm (ge_of_tendsto' (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hlt)
      (fun K => ?_)) bot_le
    calc iidS D {w | ∀ i, N ≤ i → w i ∉ A}
        ≤ iidS D (Set.pi ↑(Finset.Ico N (N + K)) (fun _ => Aᶜ)) := by
          refine measure_mono (fun w hw i hi => hw i (Finset.mem_Ico.1 hi).1)
      _ = D Aᶜ ^ K := by
          rw [Measure.infinitePi_pi _ (fun _ _ => measurableSet_of_countable' _),
            Finset.prod_const, Nat.card_Ico]
          congr 1; omega
  have hnull : iidS D (⋃ N : ℕ, {w | ∀ i, N ≤ i → w i ∉ A}) = 0 :=
    measure_iUnion_null hN
  rw [ae_iff]
  refine measure_mono_null (fun w hw => ?_) hnull
  simp only [Set.mem_setOf_eq, not_forall] at hw
  by_contra hcon
  simp only [Set.mem_iUnion, Set.mem_setOf_eq, not_exists, not_forall, not_not] at hcon
  obtain ⟨k, hk⟩ := hw
  obtain ⟨n, hn⟩ := entries_unbounded A w (fun N => by
    obtain ⟨i, hi, hiA⟩ := hcon N
    exact ⟨i, hi, hiA⟩) k
  exact hk ⟨n, hn⟩

lemma getD_iff_prefix (E : List S) (d : S) {n : ℕ} (hn : n ≤ E.length) (α : Fin n → S) :
    (∀ i : Fin n, E.getD i d = α i) ↔ List.ofFn α <+: E := by
  rw [List.prefix_iff_eq_take]
  constructor
  · intro h
    refine List.ext_getElem (by simp; omega) (fun i h1 h2 => ?_)
    simp only [List.length_ofFn] at h1
    rw [List.getElem_ofFn, List.getElem_take, ← h ⟨i, h1⟩, List.getD_eq_getElem]
  · intro h i
    have hi : (i : ℕ) < E.length := lt_of_lt_of_le i.isLt hn
    rw [List.getD_eq_getElem _ _ hi]
    have := congrArg (fun l : List S => l[(i : ℕ)]?) h
    simp only [List.getElem?_ofFn, List.getElem?_take, i.isLt] at this
    simpa [List.getElem?_eq_getElem hi] using this.symm

/-- The first `n` draws in `A` and in `R` take given values with their conditional chances. -/
theorem measure_sub_eq (D : Measure S) [IsProbabilityMeasure D] (A R : Set S)
    (hAR : Disjoint A R) (hA : D A ≠ 0) (hR : D R ≠ 0) (d₁ d₂ : S) (n : ℕ)
    (α β : Fin n → S) :
    iidS D {w | ∀ i : Fin n, nthDrawIn A d₁ w i = α i ∧ nthDrawIn R d₂ w i = β i}
      = openChance D A R (List.ofFn α) (List.ofFn β) := by
  rw [← measure_opens D A R hAR hA hR]
  refine measure_congr ?_
  filter_upwards [ae_entries_unbounded D A hA, ae_entries_unbounded D R hR] with w hwA hwR
  obtain ⟨NA, hNA⟩ := hwA n
  obtain ⟨NR, hNR⟩ := hwR n
  set N := max NA NR
  have hlA : n ≤ (drawsIn A w N).length :=
    le_trans hNA.le ((entries_prefix A w (le_max_left _ _)).length_le)
  have hlR : n ≤ (drawsIn R w N).length :=
    le_trans hNR.le ((entries_prefix R w (le_max_right _ _)).length_le)
  have hsubA : ∀ i : Fin n, nthDrawIn A d₁ w i = (drawsIn A w N).getD i d₁ :=
    fun i => sub_eq_getD A d₁ w (lt_of_lt_of_le i.isLt hlA)
  have hsubR : ∀ i : Fin n, nthDrawIn R d₂ w i = (drawsIn R w N).getD i d₂ :=
    fun i => sub_eq_getD R d₂ w (lt_of_lt_of_le i.isLt hlR)
  have hopen : w ∈ opens A R (List.ofFn α) (List.ofFn β)
      ↔ List.ofFn α <+: drawsIn A w N ∧ List.ofFn β <+: drawsIn R w N := by
    constructor
    · rintro ⟨M, hM1, hM2⟩
      have k1 : ∀ B : Set S, ∀ (γ : List S), γ.length ≤ (drawsIn B w N).length →
          γ <+: drawsIn B w M → γ <+: drawsIn B w N := by
        intro B γ hγ hM
        rcases le_total M N with hle | hle
        · exact hM.trans (entries_prefix B w hle)
        · exact List.prefix_of_prefix_length_le hM (entries_prefix B w hle) hγ
      exact ⟨k1 A _ (by simpa using hlA) hM1, k1 R _ (by simpa using hlR) hM2⟩
    · exact fun h => ⟨N, h⟩
  apply propext
  show (∀ i : Fin n, nthDrawIn A d₁ w i = α i ∧ nthDrawIn R d₂ w i = β i)
    ↔ w ∈ opens A R (List.ofFn α) (List.ofFn β)
  rw [hopen, ← getD_iff_prefix _ d₁ hlA, ← getD_iff_prefix _ d₂ hlR]
  simp only [hsubA, hsubR]
  exact forall_and

/-- Two finite measures on streams of a countable type agree once their prefixes do. -/
lemma eq_of_prefix {X : Type*} [MeasurableSpace X] [MeasurableSingletonClass X] [Countable X]
    (μ ν : Measure (ℕ → X)) [IsFiniteMeasure μ] [IsFiniteMeasure ν]
    (h : ∀ n (t : Fin n → X), μ {w | ∀ i : Fin n, w i = t i} = ν {w | ∀ i : Fin n, w i = t i}) :
    μ = ν := by
  have hπ : ∀ n, Measurable (fun (w : ℕ → X) (i : Fin n) => w i.val) := fun n => by fun_prop
  have hmar : ∀ n, μ.map (fun (w : ℕ → X) (i : Fin n) => w i.val)
      = ν.map (fun (w : ℕ → X) (i : Fin n) => w i.val) := by
    intro n
    refine Measure.ext_of_singleton (fun t => ?_)
    rw [Measure.map_apply (hπ n) (measurableSet_singleton t),
      Measure.map_apply (hπ n) (measurableSet_singleton t)]
    have e : (fun (w : ℕ → X) (i : Fin n) => w i.val) ⁻¹' {t} = {w | ∀ i : Fin n, w i = t i} := by
      ext w
      simp [funext_iff]
    rw [e]
    exact h n t
  refine ext_of_generate_finite (measurableCylinders _) generateFrom_measurableCylinders.symm
    isPiSystem_measurableCylinders (fun c hc => ?_) ?_
  · obtain ⟨I, B, -, rfl⟩ := (mem_measurableCylinders _).mp hc
    have hI : ∀ i ∈ I, i < I.sup id + 1 := fun i hi =>
      Nat.lt_succ_of_le (Finset.le_sup (f := id) hi)
    have hcyl : cylinder I B = (fun (w : ℕ → X) (i : Fin (I.sup id + 1)) => w i.val) ⁻¹'
        {t | (fun i : I => t ⟨i, hI i i.2⟩) ∈ B} := by
      ext w
      rfl
    have hm : MeasurableSet {t : Fin (I.sup id + 1) → X | (fun i : I => t ⟨i, hI i i.2⟩) ∈ B} :=
      (Set.to_countable _).measurableSet
    rw [hcyl, ← Measure.map_apply (hπ _) hm, hmar, Measure.map_apply (hπ _) hm]
  · have h0 := h 0 Fin.elim0
    have e : {w : ℕ → X | ∀ i : Fin 0, w i = Fin.elim0 i} = Set.univ := by
      ext w; simp
    rwa [e] at h0

lemma measurable_sub (A : Set S) (d : S) : Measurable (nthDrawIn A d) := by
  classical
  refine measurable_pi_lambda _ (fun k => measurable_to_countable' (fun s => ?_))
  have hcov : (fun w => nthDrawIn A d w k) ⁻¹' {s}
      = (⋃ n : ℕ, {w : ℕ → S | k < (drawsIn A w n).length ∧ (drawsIn A w n).getD k d = s})
        ∪ ({w | ∀ n, ¬ k < (drawsIn A w n).length} ∩ {_w | d = s}) := by
    ext w
    simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_union, Set.mem_iUnion,
      Set.mem_inter_iff]
    constructor
    · intro hw
      by_cases h : ∃ n, k < (drawsIn A w n).length
      · obtain ⟨n, hn⟩ := h
        exact Or.inl ⟨n, hn, (sub_eq_getD A d w hn).symm.trans hw⟩
      · refine Or.inr ⟨fun n hn => h ⟨n, hn⟩, ?_⟩
        rw [← hw, nthDrawIn, dif_neg h]
        rfl
    · rintro (⟨n, hn, he⟩ | ⟨hno, he⟩)
      · rw [sub_eq_getD A d w hn, he]
      · rw [nthDrawIn, dif_neg (fun ⟨n, hn⟩ => hno n hn)]
        exact he
  have hfin : ∀ (n : ℕ) (P : List S → Prop),
      MeasurableSet {w : ℕ → S | P (drawsIn A w n)} := by
    intro n P
    have e : {w : ℕ → S | P (drawsIn A w n)} = {w : ℕ → S | (fun t : Fin n → S =>
        ∃ w' : ℕ → S, (∀ i : Fin n, w' i.val = t i) ∧ P (drawsIn A w' n))
        (fun i : Fin n => w i.val)} := by
      ext w
      refine ⟨fun h => ⟨w, fun _ => rfl, h⟩, ?_⟩
      rintro ⟨w', hw', hP⟩
      rwa [entries_congr A (fun k hk => hw' ⟨k, hk⟩)] at hP
    rw [e]
    exact measurableSet_firstN n (fun t : Fin n → S =>
      ∃ w' : ℕ → S, (∀ i : Fin n, w' i.val = t i) ∧ P (drawsIn A w' n))
  have hall : {w : ℕ → S | ∀ n, ¬ k < (drawsIn A w n).length}
      = ⋂ n, {w : ℕ → S | ¬ k < (drawsIn A w n).length} := by
    ext w; simp
  rw [hcov, hall]
  refine (MeasurableSet.iUnion (fun n => hfin n (fun E => k < E.length ∧ E.getD k d = s))).union
    ((MeasurableSet.iInter (fun n => hfin n (fun E => ¬ k < E.length))).inter ?_)
  by_cases hds : d = s
  · simp [hds]
  · simp [hds]

/-- The draws of an i.i.d. stream in `A` and those in a disjoint `R` are independent i.i.d.
streams of the conditional laws. -/
theorem map_sub_pair (D : Measure S) [IsProbabilityMeasure D] (A R : Set S)
    (hAR : Disjoint A R) (hA : D A ≠ 0) (hR : D R ≠ 0) (d₁ d₂ : S) :
    Measure.map (fun w => (nthDrawIn A d₁ w, nthDrawIn R d₂ w)) (iidS D)
      = (iidS (D[|A])).prod (iidS (D[|R])) := by
  have := cond_isProbabilityMeasure (μ := D) hA
  have := cond_isProbabilityMeasure (μ := D) hR
  set e := MeasurableEquiv.arrowProdEquivProdArrow S S ℕ with he
  have hF : Measurable (fun w => (nthDrawIn A d₁ w, nthDrawIn R d₂ w)) :=
    (measurable_sub A d₁).prodMk (measurable_sub R d₂)
  have key : Measure.map e.symm
      (Measure.map (fun w => (nthDrawIn A d₁ w, nthDrawIn R d₂ w)) (iidS D))
      = Measure.map e.symm ((iidS (D[|A])).prod (iidS (D[|R]))) := by
    refine eq_of_prefix _ _ (fun n t => ?_)
    have hset : MeasurableSet {z : ℕ → S × S | ∀ i : Fin n, z i = t i} :=
      measurableSet_firstN (X := S × S) n (fun u => ∀ i, u i = t i)
    rw [Measure.map_apply e.symm.measurable hset, Measure.map_apply e.symm.measurable hset,
      Measure.map_apply hF (e.symm.measurable hset)]
    have e1 : (fun w => (nthDrawIn A d₁ w, nthDrawIn R d₂ w))
        ⁻¹' (e.symm ⁻¹' {z | ∀ i : Fin n, z i = t i})
        = {w | ∀ i : Fin n, nthDrawIn A d₁ w i = (t i).1 ∧ nthDrawIn R d₂ w i = (t i).2} := by
      ext w
      simp [he, MeasurableEquiv.arrowProdEquivProdArrow, Equiv.arrowProdEquivProdArrow,
        Prod.ext_iff]
    have e2 : e.symm ⁻¹' {z | ∀ i : Fin n, z i = t i}
        = Set.pi ↑(Finset.range n) (fun k => if h : k < n then {(t ⟨k, h⟩).1} else Set.univ)
          ×ˢ Set.pi ↑(Finset.range n)
            (fun k => if h : k < n then {(t ⟨k, h⟩).2} else Set.univ) := by
      ext ⟨u, v⟩
      simp only [he, MeasurableEquiv.arrowProdEquivProdArrow, Equiv.arrowProdEquivProdArrow,
        Set.mem_preimage, Set.mem_setOf_eq, Set.mem_prod, Set.mem_pi, Finset.coe_range,
        Set.mem_Iio, MeasurableEquiv.symm_mk, MeasurableEquiv.coe_mk, Equiv.coe_fn_symm_mk]
      constructor
      · intro h
        exact ⟨fun k hk => by rw [dif_pos hk]; simp [← h ⟨k, hk⟩],
          fun k hk => by rw [dif_pos hk]; simp [← h ⟨k, hk⟩]⟩
      · rintro ⟨h1, h2⟩ i
        have a := h1 i i.isLt
        have b := h2 i i.isLt
        rw [dif_pos i.isLt] at a b
        exact Prod.ext a b
    rw [e1, e2, measure_sub_eq D A R hAR hA hR d₁ d₂ n, Measure.prod_prod,
      Measure.infinitePi_pi _ (fun _ _ => measurableSet_of_countable' _),
      Measure.infinitePi_pi _ (fun _ _ => measurableSet_of_countable' _), openChance,
      List.map_ofFn, List.map_ofFn, List.prod_ofFn, List.prod_ofFn,
      ← Fin.prod_univ_eq_prod_range, ← Fin.prod_univ_eq_prod_range]
    simp
  have hinj := congrArg (Measure.map e) key
  rwa [Measure.map_map e.measurable e.symm.measurable, Measure.map_map e.measurable
    e.symm.measurable, e.self_comp_symm, Measure.map_id, Measure.map_id] at hinj

lemma ae_forall_mem (D : Measure S) [IsProbabilityMeasure D] (A : Set S) (hA : D Aᶜ = 0) :
    ∀ᵐ w ∂(iidS D), ∀ n, w n ∈ A := by
  refine ae_all_iff.2 (fun n => ?_)
  have hmp := (measurePreserving_eval_infinitePi (fun _ : ℕ => D) n)
  rw [ae_iff]
  have e : {w : ℕ → S | ¬ w n ∈ A} = (fun w : ℕ → S => w n) ⁻¹' Aᶜ := rfl
  rw [e, hmp.measure_preimage (measurableSet_of_countable' _).nullMeasurableSet, hA]

/-- With no mass on `A` a stream has no draw there, so its `A`-subsequence is `d` throughout. -/
lemma sub_ae_const (D : Measure S) [IsProbabilityMeasure D] (A : Set S) (hA : D A = 0) (d : S) :
    ∀ᵐ w ∂(iidS D), nthDrawIn A d w = fun _ => d := by
  filter_upwards [ae_forall_mem D Aᶜ (by rwa [compl_compl])] with w hw
  funext k
  have hnil : ∀ n, drawsIn A w n = [] := by
    intro n
    induction n with
    | zero => exact entries_zero A w
    | succ n ih => rw [entries_succ, ih, if_neg (hw n)]; rfl
  rw [nthDrawIn, dif_neg (by simp [hnil])]

/-- On a set of full mass a stream's subsequence is the stream itself. -/
lemma sub_ae_self (D : Measure S) [IsProbabilityMeasure D] (A : Set S) (hA : D Aᶜ = 0) (d : S) :
    ∀ᵐ w ∂(iidS D), nthDrawIn A d w = w := by
  filter_upwards [ae_forall_mem D A hA] with w hw
  funext k
  have hall : ∀ n, drawsIn A w n = (List.range n).map w := by
    intro n
    unfold drawsIn
    exact List.filter_eq_self.2 (fun s hs => by
      obtain ⟨i, -, rfl⟩ := List.mem_map.1 hs
      simpa using hw i)
  have hk : k < (drawsIn A w (k + 1)).length := by rw [hall]; simp
  rw [sub_eq_getD A d w hk, hall, List.getD_eq_getElem _ _ (by simp)]
  simp

lemma cond_eq_self (D : Measure S) [IsProbabilityMeasure D] (A : Set S) (hA : D Aᶜ = 0) :
    D[|A] = D := by
  ext t _
  have hAm := measurableSet_of_countable' A
  have h1 : D A = 1 := by
    have := measure_add_measure_compl (μ := D) hAm
    rwa [hA, add_zero, measure_univ] at this
  rw [cond_apply hAm, h1, inv_one, one_mul, Set.inter_comm, measure_inter_conull hA]

/-- The draws of an i.i.d. stream in a set of positive mass are i.i.d. of the conditional law. -/
theorem map_sub (D : Measure S) [IsProbabilityMeasure D] (R : Set S) (hR : D R ≠ 0) (d : S) :
    Measure.map (nthDrawIn R d) (iidS D) = iidS (D[|R]) := by
  by_cases hc : D Rᶜ = 0
  · rw [cond_eq_self D R hc, Measure.map_congr (sub_ae_self D R hc d), Measure.map_id']
  · have := cond_isProbabilityMeasure (μ := D) hR
    have := cond_isProbabilityMeasure (μ := D) hc
    have h := congrArg (Measure.map Prod.snd)
      (map_sub_pair D Rᶜ R disjoint_compl_left hc hR d d)
    rwa [Measure.map_map measurable_snd ((measurable_sub _ d).prodMk (measurable_sub _ d)),
      Measure.map_snd_prod, measure_univ, one_smul] at h

/-- The two subsequences are independent whatever the masses, and i.i.d. of the conditional laws
where the mass is positive. -/
theorem map_sub_pair_prod (D : Measure S) [IsProbabilityMeasure D] (A R : Set S)
    (hAR : Disjoint A R) (d₁ d₂ : S) :
    ∃ μ₁ μ₂ : Measure (ℕ → S), IsProbabilityMeasure μ₁ ∧ IsProbabilityMeasure μ₂
      ∧ Measure.map (fun w => (nthDrawIn A d₁ w, nthDrawIn R d₂ w)) (iidS D) = μ₁.prod μ₂
      ∧ (D A ≠ 0 → μ₁ = iidS (D[|A])) ∧ (D R ≠ 0 → μ₂ = iidS (D[|R])) := by
  have hF := (measurable_sub A d₁).prodMk (measurable_sub R d₂)
  by_cases hA : D A = 0
  · refine ⟨Measure.dirac (fun _ => d₁), Measure.map (nthDrawIn R d₂) (iidS D), inferInstance,
      Measure.isProbabilityMeasure_map (measurable_sub R d₂).aemeasurable, ?_, fun h => absurd hA h,
      fun h => map_sub D R h d₂⟩
    rw [Measure.map_congr (g := fun w => ((fun _ => d₁), nthDrawIn R d₂ w))
      (by filter_upwards [sub_ae_const D A hA d₁] with w hw; rw [hw])]
    rw [show (fun w : ℕ → S => ((fun _ : ℕ => d₁), nthDrawIn R d₂ w))
        = (fun y => ((fun _ : ℕ => d₁), y)) ∘ nthDrawIn R d₂ from rfl,
      ← Measure.map_map (by fun_prop) (measurable_sub R d₂)]
    exact (Measure.dirac_prod _).symm
  by_cases hR : D R = 0
  · refine ⟨Measure.map (nthDrawIn A d₁) (iidS D), Measure.dirac (fun _ => d₂),
      Measure.isProbabilityMeasure_map (measurable_sub A d₁).aemeasurable, inferInstance, ?_,
      fun h => map_sub D A h d₁, fun h => absurd hR h⟩
    rw [Measure.map_congr (g := fun w => (nthDrawIn A d₁ w, (fun _ => d₂)))
      (by filter_upwards [sub_ae_const D R hR d₂] with w hw; rw [hw])]
    rw [show (fun w : ℕ → S => (nthDrawIn A d₁ w, (fun _ : ℕ => d₂)))
        = (fun y => (y, (fun _ : ℕ => d₂))) ∘ nthDrawIn A d₁ from rfl,
      ← Measure.map_map (by fun_prop) (measurable_sub A d₁)]
    exact (Measure.prod_dirac _).symm
  · have := cond_isProbabilityMeasure (μ := D) hA
    have := cond_isProbabilityMeasure (μ := D) hR
    exact ⟨_, _, inferInstance, inferInstance, map_sub_pair D A R hAR hA hR d₁ d₂,
      fun _ => rfl, fun _ => rfl⟩

end Measure

end OrthoDFA.Thin
