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

end Round

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

end Quality

end OrthoDFA
