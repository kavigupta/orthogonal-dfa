import OrthoDFA.Proofs.LearnerReads

/-!
# The hits among a round's draws

A population's prefixes are the draws among `M` from the sampler that reach its state.  When
there are at least `P` of them, the first `P` are no more likely than `P` i.i.d. draws from
`reaching` to take any given values; there are fewer with probability at most a Hoeffding tail.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {S : Type*} [Stringlike S] {R : Type*}

namespace LearnerProof

open scoped Classical in
/-- How many of the draws reach `h`. -/
noncomputable def hitCount (H : DFA S R) (h : R) {M : ℕ} (a : Fin M → S) : ℕ :=
  ((List.ofFn a).filter fun p => H.state p = h).length

open scoped Classical in
lemma hitCount_cons (H : DFA S R) (h : R) {M : ℕ} (a : Fin (M + 1) → S) :
    hitCount H h a = (if H.state (a 0) = h then 1 else 0) + hitCount H h (Fin.tail a) := by
  unfold hitCount
  rw [List.ofFn_succ]
  by_cases h0 : H.state (a 0) = h
  · simp [h0, add_comm]
    rfl
  · simp [h0]
    rfl

open scoped Classical in
lemma hitCount_eq_card (H : DFA S R) (h : R) {M : ℕ} (a : Fin M → S) :
    hitCount H h a = (Finset.univ.filter fun i => a i ∈ {v | H.state v = h}).card := by
  induction M with
  | zero => simp [hitCount]
  | succ M ih =>
    rw [hitCount_cons, ih, Finset.card_filter, Finset.card_filter, Fin.sum_univ_succ]
    rfl

open scoped Classical in
lemma hitsIn_cons_hit (H : DFA S R) (h : R) {M : ℕ} (a : Fin (M + 1) → S)
    (h0 : H.state (a 0) = h) (i : ℕ) :
    hitsIn H h a (i + 1) = hitsIn H h (Fin.tail a) i ∧ hitsIn H h a 0 = a 0 := by
  unfold hitsIn
  rw [List.ofFn_succ, List.filter_cons, if_pos (by simpa using h0)]
  simp
  rfl

open scoped Classical in
lemma hitsIn_cons_miss (H : DFA S R) (h : R) {M : ℕ} (a : Fin (M + 1) → S)
    (h0 : H.state (a 0) ≠ h) : hitsIn H h a = hitsIn H h (Fin.tail a) := by
  funext i
  unfold hitsIn
  rw [List.ofFn_succ, List.filter_cons, if_neg (by simpa using h0)]
  rfl

variable (Dsamp : Measure S) [IsProbabilityMeasure Dsamp]

/-- When at least `P` draws reach `h`, the first `P` of them take the values `t` no more often
than `P` i.i.d. draws from `reaching` do. -/
theorem pi_hitsIn_le (H : DFA S R) (h : R) (hp : Dsamp {v | H.state v = h} ≠ 0) (M : ℕ) :
    ∀ (P : ℕ) (t : Fin P → S),
      (Measure.pi fun _ : Fin M => Dsamp)
          {a | P ≤ hitCount H h a ∧ (fun i : Fin P => hitsIn H h a i) = t}
        ≤ ∏ i, reaching H Dsamp h {t i} := by
  classical
  set A := {v : S | H.state v = h}
  have hA : MeasurableSet A := measurableSet_of_countable A
  have hρ : ∀ a, reaching H Dsamp h {a} = (Dsamp A)⁻¹ * Dsamp (A ∩ {a}) := fun a => by
    rw [reaching, if_neg hp, cond_apply hA]
  induction M with
  | zero =>
    intro P t
    rcases P with _ | P
    · exact prob_le_one.trans (by simp)
    · refine le_of_eq_of_le (measure_mono_null (fun a ha => ?_) measure_empty) zero_le
      have := ha.1
      simp [hitCount] at this
  | succ M ih =>
    intro P t
    set π := Measure.pi fun _ : Fin M => Dsamp
    have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (M + 1) => Dsamp) 0
    rcases P with _ | P
    · exact prob_le_one.trans (by simp)
    set E1 := {a : Fin M → S | P ≤ hitCount H h a
      ∧ (fun i : Fin P => hitsIn H h a i) = Fin.tail t}
    set E2 := {a : Fin M → S | P + 1 ≤ hitCount H h a
      ∧ (fun i : Fin (P + 1) => hitsIn H h a i) = t}
    have hset : {a : Fin (M + 1) → S | P + 1 ≤ hitCount H h a
          ∧ (fun i : Fin (P + 1) => hitsIn H h a i) = t}
        ⊆ MeasurableEquiv.piFinSuccAbove (fun _ : Fin (M + 1) => S) 0 ⁻¹'
          ((A ∩ {t 0}) ×ˢ E1 ∪ Aᶜ ×ˢ E2) := by
      rintro a ⟨hc, ht⟩
      simp only [Set.mem_preimage, MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv,
        Equiv.coe_fn_symm_mk, Fin.removeNth_zero]
      rw [hitCount_cons] at hc
      by_cases h0 : H.state (a 0) = h
      · left
        have hcons := hitsIn_cons_hit H h a h0
        refine ⟨⟨h0, ?_⟩, ?_, ?_⟩
        · have := congrFun ht 0
          simp only [Fin.val_zero, (hcons 0).2] at this
          simpa using this
        · rw [if_pos h0] at hc
          change P ≤ hitCount H h (Fin.tail a)
          omega
        · funext i
          have := congrFun ht i.succ
          simp only [Fin.val_succ, (hcons i).1] at this
          simpa [Fin.tail] using this
      · right
        have hmiss := hitsIn_cons_miss H h a h0
        refine ⟨h0, ?_, ?_⟩
        · rw [if_neg h0] at hc
          simpa using hc
        · rw [hmiss] at ht
          simpa using ht
    have hE1 := ih P (Fin.tail t)
    have hE2 := ih (P + 1) t
    calc (Measure.pi fun _ : Fin (M + 1) => Dsamp) {a | P + 1 ≤ hitCount H h a
          ∧ (fun i : Fin (P + 1) => hitsIn H h a i) = t}
        ≤ (Dsamp.prod π) ((A ∩ {t 0}) ×ˢ E1 ∪ Aᶜ ×ˢ E2) := by
          refine (measure_mono hset).trans (le_of_eq ?_)
          exact hmp.measure_preimage (Set.to_countable _).measurableSet.nullMeasurableSet
      _ ≤ Dsamp (A ∩ {t 0}) * π E1 + Dsamp Aᶜ * π E2 := by
          refine (measure_union_le _ _).trans (le_of_eq ?_)
          rw [Measure.prod_prod, Measure.prod_prod]
      _ ≤ Dsamp (A ∩ {t 0}) * ∏ i, reaching H Dsamp h {Fin.tail t i}
          + Dsamp Aᶜ * ∏ i, reaching H Dsamp h {t i} := by gcongr
      _ = ∏ i, reaching H Dsamp h {t i} := by
          rw [Fin.prod_univ_succ, hρ (t 0)]
          have hAt : Dsamp A ≠ ⊤ := measure_ne_top _ _
          have h1 : Dsamp (A ∩ {t 0}) = Dsamp A * ((Dsamp A)⁻¹ * Dsamp (A ∩ {t 0})) := by
            rw [← mul_assoc, ENNReal.mul_inv_cancel hp hAt, one_mul]
          have h2 : Dsamp A + Dsamp Aᶜ = 1 := by
            rw [measure_add_measure_compl hA, measure_univ]
          change Dsamp (A ∩ {t 0}) * ∏ i : Fin P, reaching H Dsamp h {t i.succ} + _ = _
          set c := (Dsamp A)⁻¹ * Dsamp (A ∩ {t 0})
          rw [h1, mul_assoc, ← add_mul, h2, one_mul]

end LearnerProof

end OrthoDFA
