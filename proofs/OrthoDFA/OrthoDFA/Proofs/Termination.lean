import OrthoDFA.Termination

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem termination_holds : Termination := by
  intro Ω _ μ _ S _ Q _ J Θ mΘ P _ ℱ A O populations D family gate fails harvest side kept
    sideKept θU cU cM φ ρ δc δs δa W hφ hρ hc hIm hI ha
  classical
  set K := 2 * 2 ^ Fintype.card Q + W with hK
  set U : ℕ → Θ → Set Q := fun r θ => undecidedStates A O θU (family r θ).1 (family r θ).2
  set M : ℕ → Θ → Set Q := fun r θ => misreadStates A O (family r θ).1 (family r θ).2
  set pools := poolsAt populations D harvest side kept sideKept
  set I := idleRefusal A O family gate kept harvest θU φ
  set Bc : ℕ → Set Θ := fun r => {θ | ∃ D' ∈ pools r θ,
      cU < D'.real {p | A.state p ∈ U r θ}
      ∨ cM < D'.real {p | A.state p ∈ M r θ}}
  set Ba : ℕ → Set Θ := fun r => {θ | gate r θ ∧ fails r θ ∧ ¬ (sideKept r θ ∧ ρ < (side r θ).real
      {p | A.state p ∈ M r θ})}
  set Bw : Set Θ := {θ | W ≤ ((Finset.range K).filter (fun r => θ ∈ I r)).card}
  have hδs : 0 ≤ δs := by
    have := hI 0 Set.univ MeasurableSet.univ
    simp only [Set.univ_inter, probReal_univ, mul_one] at this
    exact measureReal_nonneg.trans this
  have hprod : ∀ T : Finset ℕ, P.real (⋂ r ∈ T, I r) ≤ δs ^ T.card := by
    intro T
    induction T using Finset.induction_on_max with
    | empty => simp
    | insert a T hlt ih =>
      have haT : a ∉ T := fun h => lt_irrefl a (hlt a h)
      have hE : MeasurableSet[ℱ a] (⋂ r ∈ T, I r) :=
        MeasurableSet.biInter T.countable_toSet fun r hr =>
          ℱ.mono (Nat.succ_le_of_lt (hlt r hr)) _ (hIm r)
      calc P.real (⋂ r ∈ insert a T, I r) = P.real ((⋂ r ∈ T, I r) ∩ I a) := by
            rw [Finset.set_biInter_insert, Set.inter_comm]
        _ ≤ δs * P.real (⋂ r ∈ T, I r) := hI a _ hE
        _ ≤ δs * δs ^ T.card := mul_le_mul_of_nonneg_left ih hδs
        _ = δs ^ (insert a T).card := by
            rw [Finset.card_insert_of_notMem haT, pow_succ, mul_comm]
  have hBw : P.real Bw ≤ (K.choose W : ℝ) * δs ^ W := by
    have hsub : Bw ⊆ ⋃ T ∈ (Finset.range K).powersetCard W, ⋂ r ∈ T, I r := by
      intro θ hθ
      obtain ⟨T, hT, hTc⟩ := Finset.exists_subset_card_eq hθ
      refine Set.mem_iUnion₂.mpr ⟨T, Finset.mem_powersetCard.mpr
        ⟨hT.trans (Finset.filter_subset _ _), hTc⟩, Set.mem_iInter₂.mpr fun r hr => ?_⟩
      exact (Finset.mem_filter.mp (hT hr)).2
    calc P.real Bw ≤ P.real (⋃ T ∈ (Finset.range K).powersetCard W, ⋂ r ∈ T, I r) :=
          measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ ∑ T ∈ (Finset.range K).powersetCard W, P.real (⋂ r ∈ T, I r) :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ _T ∈ (Finset.range K).powersetCard W, δs ^ W := by
          refine Finset.sum_le_sum fun T hT => ?_
          have := hprod T
          rwa [(Finset.mem_powersetCard.mp hT).2] at this
      _ = (K.choose W : ℝ) * δs ^ W := by
          rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_range, nsmul_eq_mul]
  have hsub : {θ | ∀ r < K, ¬ gate r θ ∨ fails r θ}
      ⊆ (⋃ r ∈ Finset.range K, (Bc r ∪ Ba r)) ∪ Bw := by
    intro θ hθ
    by_contra hG
    have hG1 : ∀ r < K, θ ∉ Bc r ∧ θ ∉ Ba r := fun r hr =>
      ⟨fun h => hG (Or.inl (Set.mem_iUnion₂.mpr ⟨r, Finset.mem_range.mpr hr, Or.inl h⟩)),
        fun h => hG (Or.inl (Set.mem_iUnion₂.mpr ⟨r, Finset.mem_range.mpr hr, Or.inr h⟩))⟩
    have hG2 : θ ∉ Bw := fun h => hG (Or.inr h)
    have hc' : ∀ r < K, ∀ D' ∈ pools r θ,
        D'.real {p | A.state p ∈ U r θ} ≤ cU
        ∧ D'.real {p | A.state p ∈ M r θ} ≤ cM := by
      intro r hr D' hD'
      have := (hG1 r hr).1
      exact ⟨not_lt.mp fun h => this ⟨D', hD', Or.inl h⟩,
        not_lt.mp fun h => this ⟨D', hD', Or.inr h⟩⟩
    have hs' : ∀ r, ¬ gate r θ → θ ∉ I r →
        kept r θ ∧ 1 - φ < (harvest r θ).real {p | A.state p ∈ U r θ} :=
      fun r hg hi => not_not.mp fun h => hi ⟨hg, h⟩
    have ha' : ∀ r < K, gate r θ → fails r θ →
        sideKept r θ ∧ ρ < (side r θ).real {p | A.state p ∈ M r θ} := by
      intro r hr hg hf
      have := (hG1 r hr).2
      exact not_not.mp fun h => this ⟨hg, hf, h⟩
    set R := (Finset.range K).filter (fun r => ¬ gate r θ ∧ θ ∉ I r)
    set F := (Finset.range K).filter (fun r => gate r θ ∧ fails r θ)
    set Z := (Finset.range K).filter (fun r => θ ∈ I r)
    have hZ : Z.card < W := not_le.mp hG2
    have hRne : ∀ r ∈ R, ∀ s ∈ R, r < s → U r θ ≠ U s θ := by
      intro r hr s hs hrs hU
      simp only [R, Finset.mem_filter, Finset.mem_range] at hr hs
      obtain ⟨hk, hlt⟩ := hs' r hr.2.1 hr.2.2
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
    have hcover : Finset.range K ⊆ R ∪ F ∪ Z := by
      intro r hr
      have hr' := Finset.mem_range.mp hr
      simp only [R, F, Z, Finset.mem_union, Finset.mem_filter]
      by_cases hg : gate r θ
      · exact Or.inl (Or.inr ⟨hr, hg, (hθ r hr').resolve_left (not_not.mpr hg)⟩)
      · by_cases hi : θ ∈ I r
        · exact Or.inr ⟨hr, hi⟩
        · exact Or.inl (Or.inl ⟨hr, hg, hi⟩)
    have := (Finset.card_le_card hcover).trans
      ((Finset.card_union_le _ _).trans (Nat.add_le_add_right (Finset.card_union_le R F) _))
    rw [Finset.card_range] at this
    have := inj hRne
    have := inj hFne
    omega
  calc P.real {θ | ∀ r < K, ¬ gate r θ ∨ fails r θ}
      ≤ P.real ((⋃ r ∈ Finset.range K, (Bc r ∪ Ba r)) ∪ Bw) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ P.real (⋃ r ∈ Finset.range K, (Bc r ∪ Ba r)) + P.real Bw := measureReal_union_le _ _
    _ ≤ ∑ _r ∈ Finset.range K, (δc + δa) + (K.choose W : ℝ) * δs ^ W := by
        gcongr
        calc P.real (⋃ r ∈ Finset.range K, (Bc r ∪ Ba r))
            ≤ ∑ r ∈ Finset.range K, P.real (Bc r ∪ Ba r) := measureReal_biUnion_finset_le _ _
          _ ≤ ∑ _r ∈ Finset.range K, (δc + δa) := Finset.sum_le_sum fun r _ =>
              (measureReal_union_le _ _).trans (add_le_add (hc r) (ha r))
    _ = (2 * 2 ^ Fintype.card Q + W) * (δc + δa) + (K.choose W : ℝ) * δs ^ W := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hK]
        push_cast
        ring

end OrthoDFA
