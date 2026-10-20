import OrthoDFA.Proofs.ZeroDFA
import OrthoDFA.Proofs.ZeroSimple
import OrthoDFA.Proofs.ZeroSelect

/-!
# The learner's DFA, walked from `0`: the proof

Accuracy at every length up to `|Q|` with `η |Σ|^|Q| < 1` makes the start equivalent to the
covered start (`start_agree_of_accuracy`). `zero_simple_checks` then leaves a first round, the
last, with at most `2ε + ε/96` of the probes disagreeing; `dfa_of_checks_at` gives a leaf
misjudging at most `η`, and `loop_select_zero` the pick within `ξ` more.
-/

noncomputable section

namespace OrthoDFA

open Random (prefixMax le_prefixMax prefixMax_pos)

open MeasureTheory

set_option maxHeartbeats 8000000 in -- the claim's settings meet `zero_simple_checks`'s conditions
theorem learner_loop_dfa_from_zero_simple_holds : LearnerLoopDFAFromZeroSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ c γ δ pAP ξ Q A k L₁ N ε εcov δ' s S B₃ W Kc kd H V ι τ J
    PB RB cap a n hL hη hη₀ hc hδ hδ1 hpAP hξ hA2 hpAPu hceil hηA hkk hk1 hk2 hk3
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - k := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  have hℓN : ((L₁ : ℝ) + 1) ≤ 2 * N := by
    rw [hNe]
    have : (2 * k : ℕ) ≤ L₁ + 1 := by show 2 * ((L + 1) / 2) ≤ L + Fintype.card σ + 3 + 1; omega
    have : (2 : ℝ) * k ≤ L₁ + 1 := by exact_mod_cast this
    linarith
  obtain ⟨q₀, hq₀, hq₀acc⟩ := hceil
  have hc1 : c ≤ 1 := hq₀.trans measureReal_le_one
  obtain ⟨hε0, hε12, hεc0, hεcε, -, -, hbad, hcN, -⟩ :=
    dfa_rates (ℓ := (L₁ : ℝ) + 1) (ι := min (εcov / 8) (ε / (13824 * Q * (Q + 1)))) hQ1 hA2' hN1
      (by positivity) hℓN hc hc1 hξ rfl rfl rfl
  have hεcov1 : εcov ≤ 1 := by linarith
  have hδ'0 : 0 < δ' := by show 0 < δ / 16; positivity
  have hδ'1 : δ' ≤ 1 := by show δ / 16 ≤ 1; linarith
  have hKc1 : (1 : ℝ) ≤ Kc := by
    show (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ
      * (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α)
    have h1 : (1 : ℝ) ≤ (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ :=
      one_le_pow₀ (by nlinarith [pow_nonneg (show (0 : ℝ) ≤ Q + 1 by linarith) 3])
    have h2 : (1 : ℝ) ≤ (Q + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) :=
      one_le_pow₀ (by linarith)
    nlinarith
  have hlogK : 0 ≤ Real.log (16 * Kc / δ') :=
    Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith)
  have hkL : k ≤ L := by
    have : (L + 1) / 2 ≤ L ∨ L = 0 := by omega
    rcases this with h | h
    · exact h
    · exfalso; have : k = 0 := by show (L + 1) / 2 = 0; omega
      omega
  -- the window
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  have : IsProbabilityMeasure P.D := windowStrings_isProbability hkL₁
  have hpm : ∀ d, d ≤ k → prefixMax P.D d ≤ (A ^ d)⁻¹ := fun d hd => by
    have := prefixMax_windowStrings_le (α := α) hd hkL₁
    rwa [one_div] at this
  -- the start is the covered start's equivalent
  have hstart : ∀ w : List α, M.evalFrom q₀ w ∈ M.accept ↔ M.evalFrom M.start w ∈ M.accept :=
    start_agree_of_accuracy M hηA (fun ℓ hℓ => hq₀acc ℓ (by
      have : Fintype.card σ + 2 ≤ (L + 1) / 2 := by omega
      omega))
  -- the goodness level and the indecision limit
  have hV0 : 0 < V := by show 0 < 2 * N * A ^ Fintype.card σ / c; positivity
  set X : ℝ := B₃ * (L₁ + 1) * Q + Q * (Q + 1) ^ 2 * A + H * (Q + 1) with hXd
  have hS1 : (1 : ℝ) ≤ S := by
    show 1 ≤ Q * ((Q + 2) * A + 1) + 2 * A + 1
    have : 0 ≤ Q * ((Q + 2) * A + 1) := by positivity
    linarith
  have hB0 : 0 ≤ B₃ := by
    show 0 ≤ 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    have : 0 ≤ Real.log (16 * S * (S + 1) / δ') := Real.log_nonneg (by
      rw [le_div_iff₀ hδ'0]; nlinarith)
    positivity
  have hH0 : 0 ≤ H := Finset.sum_nonneg fun i _ => by positivity
  have hX0 : 0 < X := by
    have : 0 < Q * (Q + 1) ^ 2 * A := by positivity
    have : 0 ≤ B₃ * (L₁ + 1) * Q := by positivity
    have : 0 ≤ H * (Q + 1) := by positivity
    linarith
  have hι0 : 0 < ι := by
    show 0 < min (min (εcov / 8) (ε / (1728 * V * (Q + 1))))
      (min (1 / (48 * V * (L₁ + 1) * (Q + 1))) (δ' / (12 * V * X)))
    refine lt_min (lt_min (by positivity) (by positivity)) (lt_min (by positivity) ?_)
    positivity
  have hι8 : ι ≤ εcov / 8 := (min_le_left _ _).trans (min_le_left _ _)
  have hιε : ι ≤ ε / (1728 * V * (Q + 1)) := (min_le_left _ _).trans (min_le_right _ _)
  have hιL : ι ≤ 1 / (48 * V * (L₁ + 1) * (Q + 1)) := (min_le_right _ _).trans (min_le_left _ _)
  have hιδ : ι ≤ δ' / (12 * V * X) := (min_le_right _ _).trans (min_le_right _ _)
  set θ : ℝ := V * ι with hθd
  have hθ0 : 0 ≤ θ := by positivity
  have hθc : 3 * ι * N * A ^ Fintype.card σ / c ≤ 3 / 2 * θ := by
    rw [hθd]
    show _ ≤ 3 / 2 * (2 * N * A ^ Fintype.card σ / c * ι)
    rw [div_le_iff₀ hc]
    field_simp
    nlinarith [le_refl (3 * ι * N * A ^ Fintype.card σ)]
  have hθL : 48 * ((L₁ : ℕ) + 1 : ℝ) * (Q + 1) * θ ≤ 1 := by
    have h := mul_le_mul_of_nonneg_left hιL (show 0 ≤ 48 * ((L₁ : ℕ) + 1 : ℝ) * (Q + 1) * V by
      positivity)
    have e : 48 * ((L₁ : ℕ) + 1 : ℝ) * (Q + 1) * V * (1 / (48 * V * (L₁ + 1) * (Q + 1))) = 1 := by
      field_simp
    calc 48 * ((L₁ : ℕ) + 1 : ℝ) * (Q + 1) * θ
        = 48 * ((L₁ : ℕ) + 1 : ℝ) * (Q + 1) * V * ι := by rw [hθd]; ring
      _ ≤ _ := h
      _ = 1 := e
  have hθε : 18 * θ * (Q + 1) ≤ ε / 96 := by
    have h := mul_le_mul_of_nonneg_left hιε (show 0 ≤ 18 * V * (Q + 1) by positivity)
    have e : 18 * V * (Q + 1) * (ε / (1728 * V * (Q + 1))) = ε / 96 := by field_simp; ring
    calc 18 * θ * (Q + 1) = 18 * V * (Q + 1) * ι := by rw [hθd]; ring
      _ ≤ _ := h
      _ = ε / 96 := e
  have hθδ : 3 / 2 * θ * X ≤ δ' / 8 := by
    have h := mul_le_mul_of_nonneg_left hιδ (show 0 ≤ 3 / 2 * V * X by positivity)
    have e : 3 / 2 * V * X * (δ' / (12 * V * X)) = δ' / 8 := by field_simp; ring
    calc 3 / 2 * θ * X = 3 / 2 * V * X * ι := by rw [hθd]; ring
      _ ≤ _ := h
      _ = δ' / 8 := e
  -- the depths
  have hkdk : kd ≤ k := by omega
  have hak : a + Fintype.card σ < k := by show k - Fintype.card σ - 1 + Fintype.card σ < k; omega
  have ha1 : 1 ≤ a := by show 1 ≤ k - Fintype.card σ - 1; omega
  have hak' : a ≤ k := by omega
  have hkd : 8 * prefixMax P.D kd * Real.log (16 * Kc / δ') ≤ ε / 96 := by
    have hcl := Nat.le_pow_clog (b := Fintype.card α) (by omega)
      ⌈768 * Real.log (16 * Kc / δ') / ε⌉₊
    have h1 : 768 * Real.log (16 * Kc / δ') / ε ≤ A ^ kd := by
      refine (Nat.le_ceil _).trans ?_
      have := (Nat.cast_le (α := ℝ)).2 hcl
      push_cast at this
      exact this
    have hAk : (0 : ℝ) < A ^ kd := by positivity
    have h2 : 768 * Real.log (16 * Kc / δ') ≤ ε * A ^ kd := by
      rw [div_le_iff₀ hε0] at h1; linarith
    have h3 : prefixMax P.D kd ≤ (A ^ kd)⁻¹ := hpm kd hkdk
    have h4 : 0 ≤ prefixMax P.D kd := (prefixMax_pos P.D kd).le
    have h5 : prefixMax P.D kd * A ^ kd ≤ 1 := by
      have := mul_le_mul_of_nonneg_right h3 hAk.le
      rwa [inv_mul_cancel₀ hAk.ne'] at this
    nlinarith
  have hcap0 : 0 < cap := by
    have h1 : 0 < (A ^ k)⁻¹ := by positivity
    linarith
  have hDat : ∀ t, P.D.real {t} ≤ cap := fun t =>
    (real_singleton_le_prefixMax P.D k t).trans ((hpm k le_rfl).trans hk1)
  have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
  have hs1 : s ≤ 1 / 2 := by
    show 1 / 2 - η₀ ≤ 1 / 2
    linarith
  have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
  have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
    (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
  have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
  have hRB0 : 0 ≤ RB := by
    have hJ : 0 < J := by show 0 < Q + 2; linarith
    have hPB : 1 ≤ (J + 2) * PB / (δ' / 2 * (δ' / 4) * pAP * τ) := by
      have hl2 := Real.log_two_gt_d9
      have hlτ : Real.log 2 ≤ Real.log (2 / τ) :=
        Real.log_le_log (by norm_num) (by rw [le_div_iff₀ hτ0]; linarith)
      have hW1 : 1 ≤ W := by
        show 1 ≤ B₃ * (L₁ + 1) * Q + Q * (Q + 1) ^ 2 * A + (Q + 1) * A ^ Fintype.card σ / ε
        have h1 : (1 : ℝ) ≤ (Q + 1) ^ 2 := one_le_pow₀ (by linarith)
        have : 1 ≤ Q * (Q + 1) ^ 2 * A :=
          one_le_mul_of_one_le_of_one_le (one_le_mul_of_one_le_of_one_le hQ1 h1) (by linarith)
        have : 0 ≤ B₃ * (L₁ + 1) * Q := by positivity
        have : 0 ≤ (Q + 1) * A ^ Fintype.card σ / ε := by positivity
        linarith
      have hlW : 0 ≤ Real.log (8 * W / δ') := Real.log_nonneg (by
        rw [le_div_iff₀ hδ'0]; linarith)
      have hsp : s ^ 2 * pAP ≤ 1 / 4 := by
        have : s ^ 2 ≤ 1 / 4 := by nlinarith
        nlinarith [sq_nonneg s]
      have h1 : 1 ≤ 128 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP) := by
        rw [le_div_iff₀ (by positivity)]
        nlinarith
      have h2 : 0 ≤ 16 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2 := by
        refine div_nonneg (mul_nonneg (by norm_num) (Real.log_nonneg ?_)) (by positivity)
        rw [le_div_iff₀ (by positivity)]
        show 1 * (δ' / 2) ≤ Q + 2 + 2
        linarith
      have hPB1 : 1 ≤ PB := by
        show 1 ≤ 128 * (Real.log (2 / τ) + 4 * Real.log (8 * W / δ')) / (s ^ 2 * pAP)
          + 16 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
        linarith
      have hD1 : δ' / 2 * (δ' / 4) * pAP * τ ≤ 1 := by
        have : δ' / 2 * (δ' / 4) * pAP ≤ 1 := by nlinarith
        nlinarith
      rw [le_div_iff₀ (by positivity), one_mul]
      nlinarith
    show 0 ≤ 60000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * pAP * τ))
      / (s ^ 4 * τ ^ 2)
    exact div_nonneg (mul_nonneg (by positivity) (Real.log_nonneg hPB)) (by positivity)
  have hmr : 6 * J * (RB + 1) * prefixMax P.D a ≤ ε / 96 := by
    refine le_trans ?_ hk2
    have hJ : 0 < J := by show 0 < Q + 2; linarith
    have hRB1 : 0 ≤ 6 * J * (RB + 1) := by positivity
    exact mul_le_mul_of_nonneg_left (hpm a hak') hRB1
  have hH : H = ∑ i ∈ Finset.Ico 0 kd, A ^ i := by
    show ∑ i ∈ Finset.range kd, A ^ i = _
    rw [Finset.range_eq_Ico]
  obtain ⟨settings, hwalk, hdis, hbud, hzd, hnp, hmain⟩ := zero_simple_checks M O P L η₀ εcov ε δ'
    pAP c ι θ kd a q₀ hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP rfl rfl hc hq₀ hstart hι0 hι8
    hθ0 hθc hθL hθε (by rw [← hH]; exact hθδ) hkd ha1 hak hDat hmr
  refine ⟨settings, hwalk, hdis, hbud, ?_⟩
  have hatom : ∀ v : FreeMonoid α,
      (uniformStrings (L₁ + Fintype.card σ + 3)).real {v} ≤ cap := fun v => by
    refine (uniformStrings_singleton_le _ v).trans ?_
    have h1 : 1 / A ^ (L₁ + Fintype.card σ + 3) ≤ (A ^ k)⁻¹ := by
      rw [one_div]
      exact inv_anti₀ (by positivity) (pow_le_pow_right₀ (by linarith) (by omega))
    exact h1.trans hk1
  set Dsf : Measure (FreeMonoid α) := uniformStrings (L₁ + Fintype.card σ + 3) with hDsfd
  have hD := hmain Dsf inferInstance (uniformStrings_length _) hpAPu hatom
  set T := settings.level3.probeBudget
  set μL := loopMeasure μ α (Fintype.card σ + 1) T
  set β := Measure.pi fun _ : Fin n => uniformStrings (α := α) L
  set Nn := P.L + T + 2
  set Ls := P.L + Fintype.card σ + 3
  set t : ℝ := ξ / 4 with htd
  set Bloop : Set (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T)) :=
    {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
      ¬ ((rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.Checks M O P.D 0 (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε + ε / 96))
        ∧ RoundsFresh settings O P.D Dsf Nn Ls p)} with hBloopd
  set Sel : ℕ → Set ((Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
      × (Fin n → FreeMonoid α)) := fun r =>
    if hr : r < Fintype.card σ + 1 then
      selFailZ settings O P.D Dsf Nn Ls L n (Fintype.card σ + 2) (Fintype.card σ) t r hr else ∅
    with hSeld
  have hsub : {q : (Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
        × (Fin n → FreeMonoid α) |
      let rounds := indecisiveLStarLoop settings O (windowStrings k L₁)
        (uniformStrings (L₁ + Fintype.card σ + 3)) (persistDraws q.1)
      ¬ (rounds.length ≤ Fintype.card σ + 1 ∧ ∃ o ∈ rounds.getLast?,
          o.result.ending = .consistent ∧ (uniformStrings L).real {x | ¬
            o.result.hypothesis.Classifies (pickLeaf o.result.hypothesis o.read q.2) x
              (M.eval x.toList ∈ M.accept)} ≤ γ + ξ)}
      ⊆ (Prod.fst ⁻¹' Bloop) ∪ ⋃ r ∈ Finset.range (Fintype.card σ + 1), Sel r := by
    intro q hq
    by_cases hlp : q.1 ∈ Bloop
    · exact Or.inl hlp
    right
    have hlp' := not_not.1 hlp
    obtain ⟨⟨hlen, o, ho, hch⟩, hfr⟩ := hlp'
    have hcons : o.result.ending = .consistent := hch.1.1
    obtain ⟨r, hr, -, tt, hst, hG, hOk, hB, hF, hread, hhyp⟩ :=
      stageSel_last settings O P Dsf Ls q.1 hfr o ho hcons
    have hleaves : o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 :=
      (Random.classSet_props _ _ hch.2.2).2
    have hGZ : SelGZ (Fintype.card σ) r (some (tt, resCode o.result)) := by
      obtain ⟨t', h', he, hl, hc'⟩ := hG
      obtain ⟨e1, e2⟩ := Prod.mk.inj (Option.some.inj he)
      refine ⟨tt, resCode o.result, rfl, by rw [e1]; exact hl, by rw [e2]; exact hc', hleaves⟩
    refine Set.mem_iUnion₂.2 ⟨r, Finset.mem_range.2 hr, ?_⟩
    simp only [hSeld, dif_pos hr]
    by_contra hnot
    apply hq
    refine ⟨hlen, o, ho, hcons, ?_⟩
    have hnot' : (q.1.1, q.2) ∉ SelBad O L n (Fintype.card σ + 2) t (some (tt, resCode o.result))
        ∧ (q.1.2, q.2) ∈ SelDkZ settings P.D L n r hr (some (tt, resCode o.result)) := by
      constructor
      · intro h
        apply hnot
        show SelGZ (Fintype.card σ) r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hGZ, hOk, Or.inl h⟩
      · by_contra h
        apply hnot
        show SelGZ (Fintype.card σ) r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hGZ, hOk, Or.inr h⟩
    obtain ⟨hnb, hdk⟩ := hnot'
    obtain ⟨hinj, -⟩ := hdk
    have hsf : SuffixFree o.family := by
      have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
        hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
      intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    have hgood : (q.1.1, q.2) ∉ scoreBad O o.result.hypothesis o.family o.state.lo
        (o.state.hi + 1) L n t := by
      intro h
      apply hnb
      show (q.1.1, q.2) ∈ (if SuffixFree tt.2.2.1
          ∧ (resCode o.result).hyp.tree.leaves.length ≤ Fintype.card σ + 2
        then scoreBad O (resCode o.result).hyp tt.2.2.1 tt.2.1.lo (tt.2.1.hi + 1) L n t else ∅)
      rw [hF, hB, hhyp, if_pos ⟨hsf, hleaves⟩]
      exact h
    have hs := dfa_of_checks_at M O hL o L 0 (Nat.zero_le _) hch ?_ ?_
      ⟨q₀, hq₀, hq₀acc L (by omega)⟩
    · have hn : 0 < n := by
        refine Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos ?_)) (by positivity))
        rw [one_lt_div hδ]
        nlinarith only [hQ0, hδ1, mul_nonneg hQ0 hQ0]
      have hεcov_ξ : εcov ≤ ξ / (8 * N) := min_le_right _ _
      have hNε : N * εcov ≤ ξ / 8 := by
        have h := mul_le_mul_of_nonneg_left hεcov_ξ (by linarith : (0 : ℝ) ≤ N)
        rw [show N * (ξ / (8 * N)) = ξ / 8 by field_simp] at h
        exact h
      have hu : ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
          ∂(uniformStrings L) ≤ 3 / 8 * (N * εcov) := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact (mul_le_mul_of_nonneg_left hch.1.2.2 (zero_le_one.trans hN1)).trans
          (le_of_eq (by ring))
      have hm : ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(uniformStrings L)
          ≤ 1025 / 1024 * (N * εcov) := by
        refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
          fun x => measureReal_nonneg).trans ?_
        exact (mul_le_mul_of_nonneg_left hch.1.2.1 (zero_le_one.trans hN1)).trans
          (le_of_eq (by ring))
      have h := pickLeaf_le M O o.result.hypothesis o.family o.state.lo (o.state.hi + 1) hL L n hn
        t (3 / 8 * (N * εcov)) (1025 / 1024 * (N * εcov)) γ q.1.1 q.2 hinj hgood hu hm hs
      rw [hread]
      have : γ + 2 * t + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) ≤ γ + ξ := by
        have h0 : 0 ≤ N * εcov := mul_nonneg (zero_le_one.trans hN1) hεc0.le
        rw [htd]; linarith
      exact h.trans this
    · show (A + 1) * N * (ε + ε + ε / 96) < c
      exact hbad
    · show (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov) * N < c
      have h0 : 0 ≤ εcov * N := mul_nonneg hεc0.le (zero_le_one.trans hN1)
      linarith only [hcN, h0]
  have hBl : (μL.prod β).real (Prod.fst ⁻¹' Bloop) ≤ δ / 2 := by
    have e : (μL.prod β).real (Prod.fst ⁻¹' Bloop) = μL.real Bloop := by
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    refine hD.trans ?_
    show 3 * (δ / 16) ≤ δ / 2
    linarith
  have hsel : ∀ r ∈ Finset.range (Fintype.card σ + 1), (μL.prod β).real (Sel r)
      ≤ δ / (2 * (Q + 1)) := by
    intro r hr
    have hr' := Finset.mem_range.1 hr
    simp only [hSeld, dif_pos hr']
    refine (loop_select_zero settings O P Dsf Ls L n (Fintype.card σ + 2) (Fintype.card σ)
      ⌊RB⌋₊ t (by positivity) (by rw [hzd]; omega) (by rw [hwalk]; exact Nat.zero_le _) hnp
      r hr').trans ?_
    rw [hzd]
    have e1 : ((Fintype.card σ + 2 : ℕ) : ℝ) = Q + 2 := by push_cast; rfl
    have e2 : ((Fintype.card σ + 1 : ℕ) : ℝ) + 1 = Q + 2 := by push_cast; ring
    rw [e1, e2, htd]
    exact dfa_select_arith hQ0 hA2' hδ hξ (show a ≤ L by omega) (Nat.le_ceil _) hbud
      (Nat.floor_le hRB0) hk3
  refine (measureReal_mono hsub (measure_ne_top _ _)).trans ?_
  refine (measureReal_union_le _ _).trans ?_
  refine (add_le_add hBl ((measureReal_biUnion_finset_le _ _).trans
    (Finset.sum_le_sum hsel))).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  show δ / 2 + ((Fintype.card σ + 1 : ℕ) : ℝ) * (δ / (2 * (Q + 1))) = δ
  push_cast
  field_simp
  ring

end OrthoDFA

end
