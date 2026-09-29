import OrthoDFA.Termination

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem termination_holds : Termination := by
  intro Ω _ μ _ S _ Q _ J Θ _ P _ A O populations D family gate fails harvest side kept sideKept
    θU cU cM φ ρ δc δs δa hφ hρ hc hs ha
  classical
  set K := 2 * 2 ^ Fintype.card Q + 1 with hK
  set U : ℕ → Θ → Set Q := fun r θ => undecidedStates A O θU (family r θ).1 (family r θ).2
  set M : ℕ → Θ → Set Q := fun r θ => misreadStates A O (family r θ).1 (family r θ).2
  set pools := poolsAt populations D harvest side kept sideKept
  set Bc : ℕ → Set Θ := fun r => {θ | ∃ D' ∈ pools r θ,
      cU < D'.real {p | A.state p ∈ U r θ}
      ∨ cM < D'.real {p | A.state p ∈ M r θ}}
  set Bs : ℕ → Set Θ := fun r => {θ | ¬ gate r θ ∧ ¬ (kept r θ ∧ 1 - φ < (harvest r θ).real
      {p | A.state p ∈ U r θ})}
  set Ba : ℕ → Set Θ := fun r => {θ | gate r θ ∧ fails r θ ∧ ¬ (sideKept r θ ∧ ρ < (side r θ).real
      {p | A.state p ∈ M r θ})}
  have hsub : {θ | ∀ r < K, ¬ gate r θ ∨ fails r θ}
      ⊆ ⋃ r ∈ Finset.range K, (Bc r ∪ Bs r ∪ Ba r) := by
    intro θ hθ
    by_contra hG
    simp only [Set.mem_iUnion, Finset.mem_range, not_exists, Set.mem_union, not_or] at hG
    have hc' : ∀ r < K, ∀ D' ∈ pools r θ,
        D'.real {p | A.state p ∈ U r θ} ≤ cU
        ∧ D'.real {p | A.state p ∈ M r θ} ≤ cM := by
      intro r hr D' hD'
      have := (hG r hr).1.1
      exact ⟨not_lt.mp fun h => this ⟨D', hD', Or.inl h⟩,
        not_lt.mp fun h => this ⟨D', hD', Or.inr h⟩⟩
    have hs' : ∀ r < K, ¬ gate r θ →
        kept r θ ∧ 1 - φ < (harvest r θ).real {p | A.state p ∈ U r θ} := by
      intro r hr hg
      have := (hG r hr).1.2
      exact not_not.mp fun h => this ⟨hg, h⟩
    have ha' : ∀ r < K, gate r θ → fails r θ →
        sideKept r θ ∧ ρ < (side r θ).real {p | A.state p ∈ M r θ} := by
      intro r hr hg hf
      have := (hG r hr).2
      exact not_not.mp fun h => this ⟨hg, hf, h⟩
    set R := (Finset.range K).filter (fun r => ¬ gate r θ)
    set F := (Finset.range K).filter (fun r => gate r θ ∧ fails r θ)
    have hRne : ∀ r ∈ R, ∀ s ∈ R, r < s → U r θ ≠ U s θ := by
      intro r hr s hs hrs hU
      simp only [R, Finset.mem_filter, Finset.mem_range] at hr hs
      obtain ⟨hk, hlt⟩ := hs' r hr.1 hr.2
      have hmem : harvest r θ ∈ pools s θ := Or.inr (Or.inl ⟨r, hrs, hk, rfl⟩)
      have := (hc' s hs.1 _ hmem).1
      rw [← hU] at this
      linarith
    have hFne : ∀ r ∈ F, ∀ s ∈ F, r < s → M r θ ≠ M s θ := by
      intro r hr s hs hrs hM
      simp only [F, Finset.mem_filter, Finset.mem_range] at hr hs
      obtain ⟨hk, hlt⟩ := ha' r hr.1 hr.2.1 hr.2.2
      have hmem : side r θ ∈ pools s θ := Or.inr (Or.inr ⟨r, hrs, hk, rfl⟩)
      have := (hc' s hs.1 _ hmem).2
      rw [← hM] at this
      linarith
    have inj {T : Finset ℕ} {f : ℕ → Set Q} (h : ∀ r ∈ T, ∀ s ∈ T, r < s → f r ≠ f s) :
        T.card ≤ 2 ^ Fintype.card Q := by
      rw [← Fintype.card_set, ← Finset.card_univ]
      refine Finset.card_le_card_of_injOn f (fun _ _ => Finset.mem_univ _) ?_
      intro r hr s hs hf
      rcases lt_trichotomy r s with hrs | hrs | hrs
      · exact absurd hf (h r hr s hs hrs)
      · exact hrs
      · exact absurd hf.symm (h s hs r hr hrs)
    have hcover : Finset.range K ⊆ R ∪ F := by
      intro r hr
      have hr' := Finset.mem_range.mp hr
      simp only [R, F, Finset.mem_union, Finset.mem_filter]
      by_cases hg : gate r θ
      · exact Or.inr ⟨hr, hg, (hθ r hr').resolve_left (not_not.mpr hg)⟩
      · exact Or.inl ⟨hr, hg⟩
    have := (Finset.card_le_card hcover).trans (Finset.card_union_le R F)
    rw [Finset.card_range] at this
    have := inj hRne
    have := inj hFne
    omega
  calc P.real {θ | ∀ r < K, ¬ gate r θ ∨ fails r θ}
      ≤ P.real (⋃ r ∈ Finset.range K, (Bc r ∪ Bs r ∪ Ba r)) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ r ∈ Finset.range K, P.real (Bc r ∪ Bs r ∪ Ba r) := measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _r ∈ Finset.range K, (δc + δs + δa) := by
        refine Finset.sum_le_sum fun r _ => ?_
        calc P.real (Bc r ∪ Bs r ∪ Ba r) ≤ P.real (Bc r ∪ Bs r) + P.real (Ba r) :=
              measureReal_union_le _ _
          _ ≤ P.real (Bc r) + P.real (Bs r) + P.real (Ba r) := by
              gcongr; exact measureReal_union_le _ _
          _ ≤ δc + δs + δa := by gcongr <;> [exact hc r; exact hs r; exact ha r]
    _ = (2 * 2 ^ Fintype.card Q + 1) * (δc + δs + δa) := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hK]
        push_cast
        ring

end OrthoDFA
