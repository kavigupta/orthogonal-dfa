import OrthoDFA.Termination

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

theorem termination_holds : Termination := by
  intro Ω _ μ _ S _ Q _ J Θ mΘ P _ ℱ A O populations D family gate fails harvest side kept
    sideKept halves halvings kstar θs cap cM ρ δc δs δa W hρ hstep hhalf hcap hc hIm hI ha
  classical
  set K := (θs.card + 1) * 2 ^ Fintype.card Q + kstar + W with hK
  set U : ℝ → ℕ → Θ → Set Q := fun t r θ => undecidedStates A O t (family r θ).1 (family r θ).2
  set M : ℕ → Θ → Set Q := fun r θ => misreadStates A O (family r θ).1 (family r θ).2
  set pools := poolsAt populations D harvest side kept sideKept
  set I := idleRefusal A O family gate kept halves halvings kstar harvest θs cap
  set Bc : ℕ → Set Θ := fun r => {θ | ∃ D' ∈ pools r θ,
      (∃ t ∈ θs, cap (halvings r θ) t < D'.real {p | A.state p ∈ U t r θ})
      ∨ cM < D'.real {p | A.state p ∈ M r θ}}
  set Ba : ℕ → Set Θ := fun r => {θ | gate r θ ∧ fails r θ ∧ ¬ (sideKept r θ ∧ ρ < (side r θ).real
      {p | A.state p ∈ M r θ})}
  set Bw : Set Θ := {θ | W ≤ ((Finset.range K).filter (fun r => θ ∈ I r)).card}
  -- Halvings only accumulate, and each halving round raises the count past it.
  have hmono : ∀ θ {r s : ℕ}, r ≤ s → halvings r θ ≤ halvings s θ := fun θ _ _ h =>
    monotone_nat_of_le_succ (fun r => hstep r θ) h
  have hraise : ∀ θ {r s : ℕ}, r < s → halves r θ → halvings r θ < halvings s θ := by
    intro θ r s hrs hr
    have h1 : halvings r θ < halvings (r + 1) θ := by rw [hhalf r θ hr]; omega
    exact h1.trans_le (hmono θ hrs)
  have hcapm : ∀ t {k k' : ℕ}, k ≤ k' → cap k' t ≤ cap k t := fun t _ _ h =>
    antitone_nat_of_succ_le (f := fun k => cap k t) (fun k => hcap k t) h
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
        (∀ t ∈ θs, D'.real {p | A.state p ∈ U t r θ} ≤ cap (halvings r θ) t)
        ∧ D'.real {p | A.state p ∈ M r θ} ≤ cM := by
      intro r hr D' hD'
      have := (hG1 r hr).1
      exact ⟨fun t ht => not_lt.mp fun h => this ⟨D', hD', Or.inl ⟨t, ht, h⟩⟩,
        not_lt.mp fun h => this ⟨D', hD', Or.inr h⟩⟩
    have hs' : ∀ r, ¬ gate r θ → θ ∉ I r →
        (kept r θ ∧ ∃ t ∈ θs, cap (halvings r θ) t < (harvest r θ).real
            {p | A.state p ∈ U t r θ})
        ∨ (halves r θ ∧ halvings r θ < kstar) := by
      intro r hg hi
      by_contra h
      exact hi ⟨hg, fun hk => h (Or.inl hk), fun hh => h (Or.inr hh)⟩
    have ha' : ∀ r < K, gate r θ → fails r θ →
        sideKept r θ ∧ ρ < (side r θ).real {p | A.state p ∈ M r θ} := by
      intro r hr hg hf
      have := (hG1 r hr).2
      exact not_not.mp fun h => this ⟨hg, hf, h⟩
    set R : ℝ → Finset ℕ := fun t => (Finset.range K).filter (fun r =>
      ¬ gate r θ ∧ kept r θ ∧ cap (halvings r θ) t < (harvest r θ).real
        {p | A.state p ∈ U t r θ})
    set F := (Finset.range K).filter (fun r => gate r θ ∧ fails r θ)
    set H := (Finset.range K).filter (fun r => halves r θ ∧ halvings r θ < kstar)
    set Z := (Finset.range K).filter (fun r => θ ∈ I r)
    have hZ : Z.card < W := not_le.mp hG2
    have hRne : ∀ t ∈ θs, ∀ r ∈ R t, ∀ s ∈ R t, r < s → U t r θ ≠ U t s θ := by
      intro t ht r hr s hs hrs hU
      simp only [R, Finset.mem_filter, Finset.mem_range] at hr hs
      have hmem : harvest r θ ∈ pools s θ := Or.inr (Or.inl ⟨r, hrs, hr.2.2.1, rfl⟩)
      have := (hc' s hs.1 _ hmem).1 t ht
      rw [← hU] at this
      have hle := hcapm t (hmono θ hrs.le)
      linarith [hr.2.2.2]
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
    have hHcard : H.card ≤ kstar := by
      rw [← Finset.card_range kstar]
      refine Finset.card_le_card_of_injOn (fun r => halvings r θ) (fun r hr => ?_) ?_
      · exact Finset.mem_range.mpr (Finset.mem_filter.mp hr).2.2
      · intro r hr s hs hf
        have hr' := (Finset.mem_filter.mp hr).2.1
        have hs' := (Finset.mem_filter.mp hs).2.1
        rcases lt_trichotomy r s with hrs | hrs | hrs
        · exact absurd hf (hraise θ hrs hr').ne
        · exact hrs
        · exact absurd hf.symm (hraise θ hrs hs').ne
    have hcover : Finset.range K ⊆ θs.biUnion R ∪ F ∪ H ∪ Z := by
      intro r hr
      have hr' := Finset.mem_range.mp hr
      simp only [R, F, H, Z, Finset.mem_union, Finset.mem_filter, Finset.mem_biUnion]
      by_cases hg : gate r θ
      · exact Or.inl (Or.inl (Or.inr ⟨hr, hg, (hθ r hr').resolve_left (not_not.mpr hg)⟩))
      · by_cases hi : θ ∈ I r
        · exact Or.inr ⟨hr, hi⟩
        · rcases hs' r hg hi with ⟨hk, t, ht, hlt⟩ | hh
          · exact Or.inl (Or.inl (Or.inl ⟨t, ht, hr, hg, hk, hlt⟩))
          · exact Or.inl (Or.inr ⟨hr, hh⟩)
    have hRcard : (θs.biUnion R).card ≤ θs.card * 2 ^ Fintype.card Q :=
      (Finset.card_biUnion_le).trans (by
        rw [← smul_eq_mul, ← Finset.sum_const]
        exact Finset.sum_le_sum fun t ht => inj (hRne t ht))
    have hcard := (Finset.card_le_card hcover).trans
      ((Finset.card_union_le _ _).trans (Nat.add_le_add_right
        ((Finset.card_union_le _ _).trans (Nat.add_le_add_right
          (Finset.card_union_le (θs.biUnion R) F) _)) _))
    rw [Finset.card_range] at hcard
    have := inj hFne
    have : K = θs.card * 2 ^ Fintype.card Q + 2 ^ Fintype.card Q + kstar + W := by rw [hK]; ring
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
    _ = ((θs.card + 1) * 2 ^ Fintype.card Q + kstar + W) * (δc + δa)
          + (K.choose W : ℝ) * δs ^ W := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, hK]
        push_cast
        ring

end OrthoDFA
