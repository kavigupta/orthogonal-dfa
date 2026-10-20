import OrthoDFA.Proofs.LoopSoundRound
import OrthoDFA.Proofs.ProbeAgree
import OrthoDFA.Proofs.LoopDFA

/-!
# A consistent round's DFA, from its rate and its family's cut

Walked from `0`, a probe that neither searches nor stops undecided at its ends is classified as
read by the DFA started where `ε` sifts (`ProbeAgreeCorrect`); a decided read on the wrong side is
a seeded miscut or a seeded indecision (`misread_cut`). So the DFA misjudges at most the probes'
disagreements plus the family's miscut and indecision of the uniform pool
(`dfa_of_reads`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Random

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω}

omit [DecidableEq α] in
open scoped Classical in
/-- A read on the wrong side is, with the seed's vote, a miscut or undecided. -/
lemma misread_cut (O : Oracle μ (FreeMonoid α)) {F : Finset (FreeMonoid α)} (hF : 1 ∉ F)
    {lo hi : ℕ} (hlh : lo ≤ hi) (p : FreeMonoid α) (ω : Ω)
    (hω : familyRead O.mq F lo (hi + 1) p ω = if p ∈ O.L then .reject else .accept) :
    ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω ∨ ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω := by
  classical
  simp only [familyRead] at hω
  simp only [cutCorrect, decided]
  have hb := voteCount_insert O.mq hF p ω
  have hb01 : (if O.mq (p * 1) ω = 1 then 1 else 0) ≤ 1 := by split_ifs <;> omega
  set v := voteCount O.mq F p ω
  set vf := voteCount O.mq (insert 1 F) p ω
  by_cases hp : p ∈ O.L
  · have hl : O.label p = 1 := by simp [Oracle.label, hp]
    rw [if_pos hp] at hω
    have hv : v ≤ lo := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h2
    by_cases hlo : vf ≤ lo
    · left
      intro hcc
      have := hcc.2 hlo
      rw [hl] at this
      norm_num at this
    · right
      omega
  · have hl : O.label p = 0 := by simp [Oracle.label, hp]
    rw [if_neg hp] at hω
    have hv : hi + 1 ≤ v := by
      unfold readOf at hω
      split_ifs at hω with h1 h2
      · exact h1
    by_cases hhi : hi + 1 < vf
    · left
      intro hcc
      have := hcc.1 hhi
      rw [hl] at this
      norm_num at this
    · right
      omega

/-- Off its avoided probes, a law is at most its conditioned law plus what it avoids. -/
lemma real_le_avoidLaw (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k : ℕ)
    (K₀ : Finset (FreeMonoid α)) (S : Set (FreeMonoid α)) :
    D.real S ≤ (avoidLaw D k K₀).real S + D.real (avoiding k ↑K₀)ᶜ := by
  classical
  set A := avoiding (α := α) k ↑K₀
  have hsplit : D.real S ≤ D.real (S ∩ A) + D.real Aᶜ :=
    (measureReal_mono (fun x hx => by
      by_cases h : x ∈ A
      · exact Or.inl ⟨hx, h⟩
      · exact Or.inr h)).trans (measureReal_union_le _ _)
  refine hsplit.trans (add_le_add ?_ le_rfl)
  by_cases h0 : D A = 0
  · rw [measureReal_def, measure_mono_null Set.inter_subset_right h0, ENNReal.toReal_zero]
    exact measureReal_nonneg
  · rw [avoidLaw_eq D k K₀]
    unfold popLaw
    rw [if_neg (by rw [Measure.restrict_apply_univ]; exact fun h => h.elim h0 (measure_ne_top _ _))]
    have e : (((D.restrict A) Set.univ)⁻¹ • D.restrict A).real S
        = (D A).toReal⁻¹ * (D (S ∩ A)).toReal := by
      rw [measureReal_def, Measure.smul_apply, smul_eq_mul, Measure.restrict_apply
        (measurableSet_of_countable S), Measure.restrict_apply_univ, ENNReal.toReal_mul,
        ENNReal.toReal_inv]
    rw [e]
    have hA1 : (D A).toReal ≤ 1 := measureReal_le_one
    have hA0 : 0 < (D A).toReal := ENNReal.toReal_pos h0 (measure_ne_top _ _)
    rw [le_inv_mul_iff₀ hA0]
    exact mul_le_of_le_one_left ENNReal.toReal_nonneg hA1

theorem rootOne_runProbes (read : FreeMonoid α → ARU) (C : Level3Settings) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α), RootOne s.tree →
      RootOne (runProbes read C s xs).1.tree
  | [], s, h => h
  | x :: xs, s, h => by
    simp only [runProbes]
    rcases hst : applyProbe read C s x with s' | e
    · refine rootOne_runProbes read C xs s' ?_
      unfold applyProbe at hst
      split at hst
      · simp only [Sum.inl.injEq] at hst
        rw [← hst]
        exact h
      · split at hst
        · simp only [Sum.inl.injEq] at hst
          rw [← hst]
          exact rootOne_splitAt h _ _
        · unfold applyTests at hst
          split at hst
          · simp at hst
          · simp only [Sum.inl.injEq] at hst
            rw [← hst]
            exact h
      · unfold applyTests at hst
        split at hst
        · simp at hst
        · simp only [Sum.inl.injEq] at hst
          rw [← hst]
          exact h
    · exact h

theorem rootOne_level3 (read : FreeMonoid α → ARU) (C : Level3Settings) {T : ℕ}
    (xs : Fin T → FreeMonoid α) : RootOne (indecisiveLStarLevel3 C read xs).hypothesis.tree :=
  rootOne_runProbes read C (List.ofFn xs) initialState ⟨_, _, rfl⟩

/-- Where at most `e < 1` of `D`'s probes walked from `0` search or stop undecided at their ends,
and off the pool's prefixes `T` the family cuts all but `c₁` of `D` and leaves at most `c₂`
undecided with its seed, the DFA started where `ε` sifts misjudges at most
`e + c₁ + c₂ + D(T)` of `D`. -/
theorem dfa_of_reads {σ : Type*} (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α))
    (hL : ∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) {F : Finset (FreeMonoid α)} (hF : 1 ∉ F)
    {lo hi : ℕ} (hlh : lo ≤ hi) (ω : Ω) (H : HypothesisDFA α) (hroot : RootOne H.tree)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (T : Finset (FreeMonoid α))
    (hT : D.real T ≤ 1 / 2) {e c₁ c₂ : ℝ}
    (hdis : D.real (searchSet (fun z => familyRead O.mq F lo (hi + 1) z ω) 0 H
      ∪ endsSet (fun z => familyRead O.mq F lo (hi + 1) z ω) 0 H) ≤ e) (he : e < 1)
    (hcut : (offLaw D T).real {p | ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω} ≤ c₁)
    (hund : (offLaw D T).real {p | ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω} ≤ c₂) :
    ∃ s, H.tree.sift (fun z => familyRead O.mq F lo (hi + 1) z ω) 1 = .inl s
      ∧ D.real {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)} ≤ e + c₁ + c₂ + D.real T := by
  classical
  set read := fun z => familyRead O.mq F lo (hi + 1) z ω
  set Dis := searchSet read 0 H ∪ endsSet read 0 H
  have hpa := probeAgreeCorrect (α := α)
  -- some probe agrees, so `ε` sifts to a leaf
  obtain ⟨x₀, hx₀⟩ : ∃ x, x ∉ Dis := by
    by_contra hall
    push Not at hall
    have : D.real Dis = 1 := by
      rw [show Dis = Set.univ from Set.eq_univ_of_forall hall, probReal_univ]
    linarith
  obtain ⟨-, h0⟩ := hpa read H 0 x₀ (Nat.zero_le _) hx₀
  obtain ⟨s, hs, -⟩ := h0 rfl hroot
  refine ⟨s, hs, ?_⟩
  set Bad := {p : FreeMonoid α | ¬ cutCorrect O lo (hi + 1) (insert 1 F) p ω}
    ∪ {p | ¬ decided O.mq lo (hi + 1) (insert 1 F) p ω}
  have hsub : {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)} ⊆ Dis ∪ Bad := by
    intro x hx
    by_contra hn
    simp only [Set.mem_union, not_or] at hn
    obtain ⟨hnd, hnb⟩ := hn
    obtain ⟨-, h1⟩ := hpa read H 0 x (Nat.zero_le _) hnd
    obtain ⟨s', hs', hdec, hcl⟩ := h1 rfl hroot
    rw [hs] at hs'
    cases hs'
    have hright : (read x = .accept ↔ M.eval x.toList ∈ M.accept) := by
      rw [← hL x]
      by_contra hw
      have hwrong : familyRead O.mq F lo (hi + 1) x ω
          = if x ∈ O.L then .reject else .accept := by
        show read x = _
        rcases hr : read x with _ | _ | _
        · rw [hr] at hw; split_ifs with hx' <;> simp_all
        · rw [hr] at hw; split_ifs with hx' <;> simp_all
        · exact absurd hr hdec
      rcases misread_cut O hF hlh x ω hwrong with h | h
      · exact hnb (Or.inl h)
      · exact hnb (Or.inr h)
    apply hx
    obtain ⟨q, hq, hqb⟩ := hcl
    exact ⟨q, hq, hqb.trans hright⟩
  have hBad : D.real Bad ≤ c₁ + c₂ + D.real T := by
    have h1 := le_offLaw_real hT Bad
    have h2 : (offLaw D T).real Bad ≤ c₁ + c₂ :=
      (measureReal_union_le _ _).trans (add_le_add hcut hund)
    linarith
  calc D.real {x | ¬ H.Classifies s x (M.eval x.toList ∈ M.accept)}
      ≤ D.real (Dis ∪ Bad) := measureReal_mono hsub
    _ ≤ D.real Dis + D.real Bad := measureReal_union_le _ _
    _ ≤ e + (c₁ + c₂ + D.real T) := add_le_add hdis hBad
    _ = e + c₁ + c₂ + D.real T := by ring

end OrthoDFA

end
