import OrthoDFA.ReturnAccuracy
import OrthoDFA.Proofs.Adaptive

/-!
# The draws that reach a state

The draws of an i.i.d. stream that reach `h` are i.i.d. from the sampler conditioned on
reaching `h`.  The proof peels off the first draw: when it reaches `h` it is the first hit and
the rest are the tail's hits, and otherwise every hit is the tail's.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

/-! ## Shifting `Nat.nth` -/

section Nth
variable (p : ℕ → Prop)

lemma infinite_succ_iff : {i | p (i + 1)}.Infinite ↔ {i | p i}.Infinite := by
  constructor
  · intro h
    refine (h.image Nat.succ_injective.injOn).mono ?_
    rintro _ ⟨i, hi, rfl⟩
    exact hi
  · intro h hfin
    refine h ((hfin.image Nat.succ).insert 0 |>.subset ?_)
    intro i hi
    cases i with
    | zero => exact Set.mem_insert _ _
    | succ j => exact Set.mem_insert_of_mem _ ⟨j, hi, rfl⟩

lemma nth_shift (hinf : {i | p i}.Infinite) (k : ℕ) :
    Nat.nth p (k + (@ite _ (p 0) (Classical.propDecidable _) 1 0))
      = Nat.nth (fun i => p (i + 1)) k + 1 := by
  classical
  have hq : {i | p (i + 1)}.Infinite := (infinite_succ_iff p).2 hinf
  have hm := Nat.nth_mem_of_infinite hq k
  have hc := Nat.count_nth_of_infinite hq k
  have h1 : Nat.count p (Nat.nth (fun i => p (i + 1)) k + 1)
      = k + (@ite _ (p 0) (Classical.propDecidable _) 1 0) := by
    rw [Nat.count_succ', hc]
  rw [← h1, Nat.nth_count hm]

lemma nth_succ_of_head (hinf : {i | p i}.Infinite) (h0 : p 0) (k : ℕ) :
    Nat.nth p (k + 1) = Nat.nth (fun i => p (i + 1)) k + 1 := by
  have := nth_shift p hinf k
  rwa [if_pos h0] at this

lemma nth_of_not_head (hinf : {i | p i}.Infinite) (h0 : ¬ p 0) (k : ℕ) :
    Nat.nth p k = Nat.nth (fun i => p (i + 1)) k + 1 := by
  have := nth_shift p hinf k
  rwa [if_neg h0, add_zero] at this

end Nth

/-! ## The events -/

variable {S : Type*} [Stringlike S] {R : Type*}

/-- Infinitely many draws of `u` reach `h`, and the first `n` of them are `s`. -/
def firstHits (H : DFA S R) (h : R) (n : ℕ) (s : Fin n → S) : Set (ℕ → S) :=
  {u | {i | H.state (u i) = h}.Infinite
    ∧ ∀ i : Fin n, u (Nat.nth (fun j => H.state (u j) = h) i) = s i}

lemma firstHits_succ (H : DFA S R) (h : R) (n : ℕ) (s : Fin (n + 1) → S) :
    firstHits H h (n + 1) s
      = {u | H.state (u 0) = h ∧ u 0 = s 0 ∧ (fun i => u (i + 1)) ∈ firstHits H h n (Fin.tail s)}
        ∪ {u | H.state (u 0) ≠ h ∧ (fun i => u (i + 1)) ∈ firstHits H h (n + 1) s} := by
  ext u
  have hiff := infinite_succ_iff (fun i => H.state (u i) = h)
  simp only [firstHits, Set.mem_ofPred_eq, Set.mem_union]
  by_cases h0 : H.state (u 0) = h
  · simp only [h0, true_and, ne_eq, not_true_eq_false, false_and, or_false]
    constructor
    · rintro ⟨hinf, hall⟩
      refine ⟨?_, hiff.2 hinf, fun i => ?_⟩
      · have := hall 0
        rwa [Fin.val_zero, Nat.nth_zero_of_zero (p := fun j => H.state (u j) = h) h0] at this
      · have := hall i.succ
        rwa [Fin.val_succ, nth_succ_of_head _ hinf h0] at this
    · rintro ⟨hs0, hinf, hall⟩
      have hinf' := hiff.1 hinf
      refine ⟨hinf', fun i => ?_⟩
      refine Fin.cases ?_ (fun i => ?_) i
      · rw [Fin.val_zero, Nat.nth_zero_of_zero (p := fun j => H.state (u j) = h) h0]
        exact hs0
      · rw [Fin.val_succ, nth_succ_of_head _ hinf' h0]
        exact hall i
  · simp only [h0, false_and, false_or, ne_eq, not_false_eq_true, true_and]
    constructor
    · rintro ⟨hinf, hall⟩
      refine ⟨hiff.2 hinf, fun i => ?_⟩
      have := hall i
      rwa [nth_of_not_head _ hinf h0] at this
    · rintro ⟨hinf, hall⟩
      have hinf' := hiff.1 hinf
      refine ⟨hinf', fun i => ?_⟩
      rw [nth_of_not_head _ hinf' h0]
      exact hall i

/-! ## Measurability -/

lemma measurableSet_reach (H : DFA S R) (h : R) (i : ℕ) :
    MeasurableSet {u : ℕ → S | H.state (u i) = h} :=
  (measurableSet_of_countable {v : S | H.state v = h}).preimage (measurable_pi_apply i)

lemma measurableSet_infinite (H : DFA S R) (h : R) :
    MeasurableSet {u : ℕ → S | {i | H.state (u i) = h}.Infinite} := by
  have : {u : ℕ → S | {i | H.state (u i) = h}.Infinite}
      = ⋂ m : ℕ, ⋃ i : ℕ, ⋃ (_ : m ≤ i), {u : ℕ → S | H.state (u i) = h} := by
    ext u
    simp only [Set.mem_ofPred_eq, Set.mem_iInter, Set.mem_iUnion, exists_prop]
    rw [← Nat.frequently_atTop_iff_infinite, Filter.frequently_atTop]
  rw [this]
  exact .iInter fun m => .iUnion fun i => .iUnion fun _ => measurableSet_reach H h i

open scoped Classical in
lemma measurable_count (H : DFA S R) (h : R) (m : ℕ) :
    Measurable (fun u : ℕ → S => Nat.count (fun j => H.state (u j) = h) m) := by
  classical
  induction m with
  | zero => simp only [Nat.count_zero]; exact measurable_const
  | succ m ih =>
    simp only [Nat.count_succ]
    refine (measurable_of_countable (fun q : ℕ × ℕ => q.1 + q.2)).comp (ih.prodMk ?_)
    exact Measurable.ite (measurableSet_reach H h m) measurable_const measurable_const

lemma measurableSet_firstHits (H : DFA S R) (h : R) (n : ℕ) (s : Fin n → S) :
    MeasurableSet (firstHits H h n s) := by
  classical
  have : firstHits H h n s = {u : ℕ → S | {i | H.state (u i) = h}.Infinite}
      ∩ ⋂ i : Fin n, ⋃ m : ℕ, ({u : ℕ → S | H.state (u m) = h}
        ∩ {u | Nat.count (fun j => H.state (u j) = h) m = i} ∩ {u | u m = s i}) := by
    ext u
    simp only [firstHits, Set.mem_ofPred_eq, Set.mem_inter_iff, Set.mem_iInter, Set.mem_iUnion]
    constructor
    · rintro ⟨hinf, hall⟩
      exact ⟨hinf, fun i => ⟨_, ⟨Nat.nth_mem_of_infinite hinf i,
        Nat.count_nth_of_infinite hinf i⟩, hall i⟩⟩
    · rintro ⟨hinf, hall⟩
      refine ⟨hinf, fun i => ?_⟩
      obtain ⟨m, ⟨hm, hc⟩, hu⟩ := hall i
      rw [← hc, Nat.nth_count hm]
      exact hu
  rw [this]
  refine (measurableSet_infinite H h).inter (.iInter fun i => .iUnion fun m => ?_)
  refine ((measurableSet_reach H h m).inter ?_).inter ?_
  · exact measurable_count H h m (measurableSet_singleton _)
  · exact measurable_pi_apply m (measurableSet_singleton _)

/-! ## The law -/

lemma isProbabilityMeasure_reaching (H : DFA S R) (Dsamp : Measure S)
    [IsProbabilityMeasure Dsamp] (h : R) : IsProbabilityMeasure (reaching H Dsamp h) := by
  unfold reaching
  split_ifs with h0
  · infer_instance
  · exact cond_isProbabilityMeasure h0

section Law
variable (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]

lemma measurePreserving_headTail :
    MeasurePreserving (fun u : ℕ → S => (u 0, fun i => u (i + 1)))
      (Measure.infinitePi fun _ : ℕ => Dsamp)
      (Dsamp.prod (Measure.infinitePi fun _ : ℕ => Dsamp)) := by
  set P := Measure.infinitePi fun _ : ℕ => Dsamp
  have hmeas : Measurable (fun u : ℕ → S => fun i => u (i + 1)) := by fun_prop
  refine ⟨by fun_prop, ?_⟩
  set m : ℕ → MeasurableSpace (ℕ → S) := fun i =>
    MeasurableSpace.comap (fun u : ℕ → S => u i) inferInstance
  have hind : iIndep m P := by
    have := iIndepFun_infinitePi (P := fun _ : ℕ => Dsamp) (X := fun _ => id)
      (fun _ => measurable_id)
    exact (iIndepFun_iff_iIndep _ _ _).1 this
  have hle : ∀ i, m i ≤ MeasurableSpace.pi := fun i => (measurable_pi_apply i).comap_le
  have hST : Disjoint ({0} : Set ℕ) {i | i ≠ 0} := by
    rw [Set.disjoint_left]; intro i hi hi'; exact hi' hi
  have h2 := indep_iSup_of_disjoint hle hind hST
  have hIF : IndepFun (fun u : ℕ → S => u 0) (fun u : ℕ → S => fun i => u (i + 1)) P := by
    rw [IndepFun_iff_Indep]
    refine indep_of_indep_of_le_right (indep_of_indep_of_le_left h2 ?_) ?_
    · exact le_iSup₂ (f := fun i (_ : i ∈ ({0} : Set ℕ)) => m i) 0 rfl
    · show MeasurableSpace.comap _ MeasurableSpace.pi ≤ _
      rw [MeasurableSpace.pi, MeasurableSpace.comap_iSup]
      refine iSup_le fun i => ?_
      rw [MeasurableSpace.comap_comp]
      exact le_iSup₂ (f := fun i (_ : i ∈ {i : ℕ | i ≠ 0}) => m i) (i + 1) (Nat.succ_ne_zero i)
  rw [hIF.map_prod_eq_prod_map_map (measurable_pi_apply 0).aemeasurable hmeas.aemeasurable,
    Measure.infinitePi_map_eval,
    Measure.map_infinitePi_infinitePi_of_inj (f := fun i => i + 1) Nat.succ_injective]

variable {Dsamp}

lemma infinite_ae (H : DFA S R) (h : R) (hp : Dsamp {v | H.state v = h} ≠ 0) :
    (Measure.infinitePi fun _ : ℕ => Dsamp) {u : ℕ → S | {i | H.state (u i) = h}.Infinite}ᶜ
      = 0 := by
  set P := Measure.infinitePi fun _ : ℕ => Dsamp
  set A := {v : S | H.state v = h}
  have hlt : Dsamp Aᶜ < 1 := by
    rw [measure_compl (measurableSet_of_countable A) (measure_ne_top _ _), measure_univ]
    exact ENNReal.sub_lt_self ENNReal.one_ne_top one_ne_zero hp
  have htail : ∀ m, P {u : ℕ → S | ∀ i, m ≤ i → H.state (u i) ≠ h} = 0 := by
    intro m
    refine le_antisymm (ge_of_tendsto' (ENNReal.tendsto_pow_atTop_nhds_zero_of_lt_one hlt)
      fun k => ?_) zero_le
    calc P {u : ℕ → S | ∀ i, m ≤ i → H.state (u i) ≠ h}
        ≤ P (Set.pi ↑(Finset.Ico m (m + k)) fun _ => Aᶜ) := by
          refine measure_mono fun u hu i hi => ?_
          simp only [Finset.coe_Ico, Set.mem_Ico] at hi
          exact hu i hi.1
      _ = Dsamp Aᶜ ^ k := by
          rw [Measure.infinitePi_pi _ fun _ _ => (measurableSet_of_countable A).compl,
            Finset.prod_const, Nat.card_Ico, Nat.add_sub_cancel_left]
  refine measure_mono_null (fun u hu => ?_) (measure_iUnion_null htail)
  simp only [Set.mem_compl_iff, Set.mem_ofPred_eq, Set.not_infinite] at hu
  obtain ⟨m, hm⟩ := hu.bddAbove
  refine Set.mem_iUnion.2 ⟨m + 1, fun i hi hh => ?_⟩
  have := hm hh
  omega

lemma measure_firstHits (H : DFA S R) (h : R) (hp : Dsamp {v | H.state v = h} ≠ 0) (n : ℕ)
    (s : Fin n → S) :
    (Measure.infinitePi fun _ : ℕ => Dsamp) (firstHits H h n s)
      = ∏ i, reaching H Dsamp h {s i} := by
  set P := Measure.infinitePi fun _ : ℕ => Dsamp
  set A := {v : S | H.state v = h}
  have hA : MeasurableSet A := measurableSet_of_countable A
  have hρ : ∀ a, reaching H Dsamp h {a} = (Dsamp A)⁻¹ * Dsamp (A ∩ {a}) := fun a => by
    rw [reaching, if_neg hp, cond_apply hA]
  induction n with
  | zero =>
    have : firstHits H h 0 s = {u : ℕ → S | {i | H.state (u i) = h}.Infinite} := by
      ext u; simp [firstHits]
    rw [this, Finset.univ_eq_empty, Finset.prod_empty, ← prob_compl_eq_zero_iff
      (measurableSet_infinite H h)]
    exact infinite_ae H h hp
  | succ n ih =>
    set x := P (firstHits H h (n + 1) s)
    set E := firstHits H h n (Fin.tail s)
    have hmp := measurePreserving_headTail Dsamp
    have hset : firstHits H h (n + 1) s
        = (fun u : ℕ → S => (u 0, fun i => u (i + 1))) ⁻¹'
          ((A ∩ {s 0}) ×ˢ E ∪ Aᶜ ×ˢ firstHits H h (n + 1) s) := by
      conv_lhs => rw [firstHits_succ]
      ext u
      simp only [Set.mem_union, Set.mem_ofPred_eq, Set.mem_preimage, Set.mem_prod,
        Set.mem_inter_iff, Set.mem_singleton_iff, Set.mem_compl_iff, A, E]
      tauto
    have hmE := measurableSet_firstHits H h n (Fin.tail s)
    have hmF := measurableSet_firstHits H h (n + 1) s
    have hdisj : Disjoint ((A ∩ {s 0}) ×ˢ E) (Aᶜ ×ˢ firstHits H h (n + 1) s) := by
      rw [Set.disjoint_left]
      rintro ⟨a, u⟩ ⟨⟨ha, _⟩, _⟩ ⟨ha', _⟩
      exact ha' ha
    have hx : x = Dsamp (A ∩ {s 0}) * P E + (1 - Dsamp A) * x := by
      conv_lhs => rw [show x = P (firstHits H h (n + 1) s) from rfl, hset]
      rw [hmp.measure_preimage (((hA.inter (measurableSet_singleton _)).prod hmE).union
          (hA.compl.prod hmF)).nullMeasurableSet,
        measure_union hdisj (hA.compl.prod hmF), Measure.prod_prod, Measure.prod_prod,
        measure_compl hA (measure_ne_top _ _), measure_univ]
    have hxtop : x ≠ ∞ := measure_ne_top _ _
    have hp1 : Dsamp A ≤ 1 := prob_le_one
    have hkey : Dsamp A * x = Dsamp (A ∩ {s 0}) * P E := by
      have h1 : x - (1 - Dsamp A) * x = Dsamp (A ∩ {s 0}) * P E :=
        ENNReal.sub_eq_of_eq_add (ENNReal.mul_ne_top (by finiteness) hxtop) hx
      calc Dsamp A * x = (1 - (1 - Dsamp A)) * x := by
            rw [ENNReal.sub_sub_cancel ENNReal.one_ne_top hp1]
        _ = 1 * x - (1 - Dsamp A) * x := ENNReal.sub_mul fun _ _ => hxtop
        _ = _ := by rw [one_mul, h1]
    have hxe : x = (Dsamp A)⁻¹ * Dsamp (A ∩ {s 0}) * P E := by
      rw [mul_assoc, ← hkey, ENNReal.inv_mul_cancel_left hp (measure_ne_top _ _)]
    rw [hxe, ih (Fin.tail s), Fin.prod_univ_succ, hρ]
    rfl

open scoped Classical in
/-- The first `n` draws of an i.i.d. `Dsamp` stream that reach `h` are i.i.d. from
`reaching`. -/
theorem measurePreserving_hitsOf (H : DFA S R) (h : R)
    (hp : Dsamp {v | H.state v = h} ≠ 0) (n : ℕ) :
    MeasurePreserving (fun (u : ℕ → S) (i : Fin n) => hitsOf H h u i)
      (Measure.infinitePi fun _ : ℕ => Dsamp) (Measure.pi fun _ : Fin n => reaching H Dsamp h) := by
  have := isProbabilityMeasure_reaching H Dsamp h
  set P := Measure.infinitePi fun _ : ℕ => Dsamp
  set Inf := {u : ℕ → S | {i | H.state (u i) = h}.Infinite}
  have hpre : ∀ s : Fin n → S, (fun (u : ℕ → S) (i : Fin n) => hitsOf H h u i) ⁻¹' {s}
      = firstHits H h n s ∪ (Infᶜ ∩ {u | ∀ i : Fin n, u i = s i}) := by
    intro s
    ext u
    by_cases hinf : u ∈ Inf
    · have hinf' : {i | H.state (u i) = h}.Infinite := hinf
      simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_union, Set.mem_inter_iff,
        Set.mem_compl_iff, hinf, not_true_eq_false, false_and, or_false, firstHits,
        Set.mem_ofPred_eq, hinf', true_and, hitsOf, ↓reduceIte, funext_iff]
    · have hinf' : ¬ {i | H.state (u i) = h}.Infinite := hinf
      simp only [Set.mem_preimage, Set.mem_singleton_iff, Set.mem_union, Set.mem_inter_iff,
        Set.mem_compl_iff, hinf, not_false_eq_true, true_and, firstHits, Set.mem_ofPred_eq,
        hinf', false_and, false_or, hitsOf, ↓reduceIte, funext_iff]
  have hmInf := measurableSet_infinite H h
  have hmeas : Measurable (fun (u : ℕ → S) (i : Fin n) => hitsOf H h u i) := by
    refine measurable_to_countable' fun s => ?_
    rw [hpre]
    refine (measurableSet_firstHits H h n s).union (hmInf.compl.inter ?_)
    simp only [Set.ofPred_forall]
    exact .iInter fun i => measurable_pi_apply _ (measurableSet_singleton _)
  refine ⟨hmeas, Measure.ext_iff_singleton.2 fun s => ?_⟩
  rw [Measure.map_apply hmeas (measurableSet_singleton s), hpre, Measure.pi_singleton,
    ← measure_firstHits H h hp n s]
  refine le_antisymm ((measure_union_le _ _).trans ?_) (measure_mono Set.subset_union_left)
  rw [measure_mono_null Set.inter_subset_left (infinite_ae H h hp), add_zero]

end Law

end OrthoDFA
