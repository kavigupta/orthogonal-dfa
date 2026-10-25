import OrthoDFA.Proofs.TSRBridge

/-!
# The learner's DFA under typical-start replication, probes to twice the length: the proof

`loop_simple_checks` at the window `⌈L/2⌉` to `2L + |Q| + 3` leaves a last round as in
`learner_loop_dfa_simple_holds`, with the misled probes counted among each state's strings shorter
than `2L + 1`. `dfa_of_checks_tsrb` gives a leaf misjudging at most
`γ + N(2((|Σ|+1) + |Q|)(ε + ε/96) + 2.4 εcov)/m`, at most `γ + ξ/2` less the pick's misreads
(`tsrb_harm`), and the pick gets `ξ/2` as in `learner_loop_dfa_simple_holds`. The closed form
`L₀` bounds `log ℓ` for `ℓ = 2L + |Q| + 4` (`dfa_pow_twice`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

open Random (prefixMax le_prefixMax prefixMax_pos)

/-- The claim's rates, from the start's visit rate `m`. -/
lemma tsrb_rates {Q A N ℓ m ξ ε εcov ι : ℝ} (hQ : 1 ≤ Q) (hA : 2 ≤ A) (hN : 1 ≤ N)
    (hℓ : ℓ ≤ 2 * N) (hm : 0 < m) (hm1 : m ≤ 1) (hξ : 0 < ξ) (hξ1 : ξ ≤ 1)
    (he : ε = ξ * m / (8 * (A + Q + 3) * N)) (hεc : εcov = min ε (ξ / (8 * N)))
    (hι : ι = min (εcov / 8) (ε / (4608 * Q * (Q + 1)))) :
    0 < ε ∧ ε ≤ 1 / 2 ∧ 0 < εcov ∧ εcov ≤ ε ∧ 0 < ι ∧ ι ≤ εcov / 8 ∧ ℓ * ε < 3 / 8 := by
  have hP : 0 < 8 * (A + Q + 3) := by positivity
  have he0 : 0 < ε := by rw [he]; positivity
  have hεN : ε * N ≤ 1 / 48 := by
    have e : ε * N = ξ * m / (8 * (A + Q + 3)) := by rw [he]; field_simp
    rw [e, div_le_iff₀ hP]
    have : ξ * m ≤ 1 := by nlinarith
    nlinarith
  have hε48 : ε ≤ 1 / 48 := by nlinarith
  have hεc0 : 0 < εcov := by rw [hεc]; exact lt_min he0 (by positivity)
  have hεcε : εcov ≤ ε := by rw [hεc]; exact min_le_left _ _
  have hι0 : 0 < ι := by rw [hι]; exact lt_min (by positivity) (by positivity)
  have hι8 : ι ≤ εcov / 8 := by rw [hι]; exact min_le_left _ _
  refine ⟨he0, by linarith, hεc0, hεcε, hι0, hι8, ?_⟩
  have : ℓ * ε ≤ 2 * N * ε := mul_le_mul_of_nonneg_right hℓ he0.le
  nlinarith

/-- The wrong states' share of the start's runs, and the pick's misreads, cost at most `ξ/2`. -/
lemma tsrb_harm {A Q N m ξ ε εcov : ℝ} (hA : 0 ≤ A) (hQ : 0 ≤ Q) (hN : 0 < N) (hm : 0 < m)
    (hm1 : m ≤ 1) (he : ε = ξ * m / (8 * (A + Q + 3) * N)) (hεc0 : 0 ≤ εcov)
    (hεcε : εcov ≤ ε) :
    N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
        + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)) / m
      + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) ≤ ξ / 2 := by
  have hP : 0 < 8 * (A + Q + 3) := by positivity
  have hu : N * ε / m = ξ / (8 * (A + Q + 3)) := by rw [he]; field_simp
  have hu0 : 0 ≤ N * ε := mul_nonneg hN.le (hεc0.trans hεcε)
  have hv : N * εcov ≤ N * ε := mul_le_mul_of_nonneg_left hεcε hN.le
  have hv0 : 0 ≤ N * εcov := mul_nonneg hN.le hεc0
  have huu : N * ε ≤ N * ε / m := le_div_self hu0 hm hm1
  set c : ℝ := 2 * (1025 / 1024) + 3 / 8 with hcd
  have e1 : N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
        + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov))
      = 97 / 48 * (A + Q + 1) * (N * ε) + c * (N * εcov) := by rw [hcd]; ring
  have e2 : 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) = c * (N * εcov) := by
    rw [hcd]; ring
  rw [e1, add_assoc, e2]
  have h1 : (97 / 48 * (A + Q + 1) * (N * ε) + c * (N * εcov)) / m
      ≤ (97 / 48 * (A + Q + 1) + c) * (N * ε / m) := by
    rw [mul_div_assoc', div_le_div_iff_of_pos_right hm]
    nlinarith
  have h2 : c * (N * εcov) ≤ c * (N * ε / m) :=
    mul_le_mul_of_nonneg_left (hv.trans huu) (by rw [hcd]; norm_num)
  have h3 : (97 / 48 * (A + Q + 1) + c) * (N * ε / m) + c * (N * ε / m)
      = (97 / 48 * (A + Q + 1) + 2 * c) * (ξ / (8 * (A + Q + 3))) := by rw [hu]; ring
  have hξ0 : 0 ≤ ξ / (8 * (A + Q + 3)) := by
    rw [← hu]; exact div_nonneg hu0 hm.le
  have h4 : (97 / 48 * (A + Q + 1) + 2 * c) * (ξ / (8 * (A + Q + 3)))
      ≤ 4 * (A + Q + 3) * (ξ / (8 * (A + Q + 3))) :=
    mul_le_mul_of_nonneg_right (by rw [hcd]; nlinarith) hξ0
  have h5 : 4 * (A + Q + 3) * (ξ / (8 * (A + Q + 3))) = ξ / 2 := by field_simp; ring
  linarith

set_option maxHeartbeats 8000000 in -- the claim's settings meet `loop_simple_checks`'s conditions
theorem learner_loop_dfa_tsrb_simple_holds : LearnerLoopDFATSRBSimple := by
  intro α _ _ _ σ _ Ω _ μ _ M O L η₀ m γ δ pAP ξ Q A k L₁ N ε εcov δ' ι s τ J S B₃ Wm W PB RB
    cap Kc n hL hη hη₀ hm hδ hδ1 hpAP hξ hξ1 hA2 hpAPu hstart hk1 hk2 hk3 hk4
  classical
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hη
  have hA2' : (2 : ℝ) ≤ A := by show (2 : ℝ) ≤ (Fintype.card α : ℝ); exact_mod_cast hA2
  have : Nonempty σ := ⟨M.start⟩
  have hQ1 : (1 : ℝ) ≤ Q := by
    show (1 : ℝ) ≤ (Fintype.card σ : ℝ); exact_mod_cast Fintype.card_pos
  have hQ0 : (0 : ℝ) ≤ Q := by linarith
  have hkL₁ : k ≤ L₁ := by show (L + 1) / 2 ≤ 2 * L + Fintype.card σ + 3; omega
  have hNe : N = ((L₁ : ℕ) : ℝ) + 1 - k := rfl
  have hN1 : (1 : ℝ) ≤ N := by
    have : (k : ℝ) ≤ L₁ := by exact_mod_cast hkL₁
    rw [hNe]; linarith
  obtain ⟨q₀, hq₀w, hq₀e⟩ := hstart
  have hm1 : m ≤ 1 := hq₀w.trans measureReal_le_one
  have hwN : 2 + (Nat.clog 2 L₁ : ℝ) ≤ 2 * N := by
    have h1 : 1 ≤ L₁ := by change 1 ≤ 2 * L + Fintype.card σ + 3; omega
    have h2 : Nat.clog 2 L₁ ≤ L₁ - 1 :=
      Nat.clog_le_of_le_pow (by have := (L₁ - 1).lt_two_pow_self; omega)
    have : 2 + Nat.clog 2 L₁ ≤ L₁ + 1 := by omega
    have : 2 + (Nat.clog 2 L₁ : ℝ) ≤ L₁ + 1 := by exact_mod_cast this
    have h3 : (2 * k : ℕ) ≤ L₁ + 1 := by
      show 2 * ((L + 1) / 2) ≤ 2 * L + Fintype.card σ + 3 + 1; omega
    have : (2 : ℝ) * k ≤ L₁ + 1 := by exact_mod_cast h3
    rw [hNe]; linarith
  obtain ⟨hε0, hε12, hεc0, hεcε, hι0, hι8, hιL⟩ :=
    tsrb_rates (ℓ := 2 + (Nat.clog 2 L₁ : ℝ)) hQ1 hA2' hN1 hwN hm hm1 hξ hξ1 rfl rfl rfl
  have hεcov1 : εcov ≤ 1 := by linarith
  have hδ'0 : 0 < δ' := by show 0 < δ / (8 * (Q + 1)); positivity
  have hδ'1' : δ' < 1 := by
    show δ / (8 * (Q + 1)) < 1
    rw [div_lt_one (by positivity)]
    linarith
  have hδ'1 : δ' ≤ 1 := hδ'1'.le
  have hKc1 : (1 : ℝ) ≤ Kc := kc_ge_one hQ0 (by linarith)
  have hlogK : 2 ≤ Real.log (16 * (Q + 1) * Kc / δ') := by
    rw [Real.le_log_iff_exp_le (by positivity)]
    have he := Real.exp_one_lt_d9
    have : Real.exp 2 ≤ 16 := by
      calc Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
        _ ≤ 3 * 3 := by gcongr <;> linarith
        _ ≤ 16 := by norm_num
    refine this.trans ?_
    rw [le_div_iff₀ hδ'0]
    nlinarith only [hδ'1, hQ0, hKc1, mul_nonneg hQ0 (sub_nonneg.2 hKc1)]
  -- the walk starts after at least one letter
  have hk0 : 1 ≤ k := by
    by_contra h
    have hk : k = 0 := by omega
    rw [hk, pow_zero, inv_one, mul_one] at hk2
    linarith
  have hkL : k ≤ L := by
    have : (L + 1) / 2 ≤ L ∨ L = 0 := by omega
    rcases this with h | h
    · exact h
    · exfalso; have : k = 0 := by show (L + 1) / 2 = 0; omega
      omega
  let P : PrefixDistribution α := ⟨windowStrings k L₁, windowStrings_isProbability hkL₁, L₁,
    (windowStrings_length k L₁).mono fun x hx => hx.2⟩
  have hpm : prefixMax P.D k ≤ (A ^ k)⁻¹ := by
    have := prefixMax_windowStrings_le (α := α) (le_refl k) hkL₁
    rwa [one_div] at this
  have : IsProbabilityMeasure P.D := windowStrings_isProbability hkL₁
  have hpm0 : 0 ≤ prefixMax P.D k := (prefixMax_pos P.D k).le
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hmain⟩ :=
    loop_simple_checks M O P (shortAt M (2 * L + 1)) η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1' hpAP
    (by show 1 ≤ 2 * L + Fintype.card σ + 3; omega)
    hk0 hιL (hpm.trans hk1)
    ((mul_le_mul_of_nonneg_left hpm (mul_nonneg (by norm_num) (by linarith [hlogK]))).trans hk2)
    ((mul_le_mul_of_nonneg_left hpm (by positivity)).trans hk4)
  refine ⟨settings, hwalk, hdis, hbud, ?_⟩
  have hcap0 : 0 < cap := by
    have h1 : 0 < (A ^ k)⁻¹ := by positivity
    by_contra hc0
    push Not at hc0
    have : cap * ε / 1536 ≤ 0 := by
      have := mul_nonpos_of_nonpos_of_nonneg hc0 hε0.le
      linarith
    linarith
  have hatom : ∀ v : FreeMonoid α,
      (uniformStrings (L₁ + Fintype.card σ + 3)).real {v} ≤ cap := fun v => by
    refine (uniformStrings_singleton_le _ v).trans ?_
    have h1 : 1 / A ^ (L₁ + Fintype.card σ + 3) ≤ (A ^ k)⁻¹ := by
      rw [one_div]
      exact inv_anti₀ (by positivity) (pow_le_pow_right₀ (by linarith) (by omega))
    refine h1.trans (hk1.trans ?_)
    have : cap * ε ≤ cap * 1 := mul_le_mul_of_nonneg_left (by linarith) hcap0.le
    linarith
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
          o.Checks M O P.D k (1025 / 1024 * εcov) (3 / 8 * εcov) (ε + ε / 96)
            (shortAt M (2 * L + 1)) (ε + ε / 96))
        ∧ RoundsFresh settings O P.D Dsf Nn Ls p)} with hBloopd
  set Sel : ℕ → Set ((Ω × (Fin (Fintype.card σ + 1) → RoundDraws α (Fintype.card σ + 1) T))
      × (Fin n → FreeMonoid α)) := fun r =>
    if hr : r < Fintype.card σ + 1 then
      selFail settings O P.D Dsf Nn Ls L n (Fintype.card σ + 2) t r hr else ∅
    with hSeld
  -- the loop's last round, consistent and fresh, picks well unless its scores stray or a
  -- held-out string collides
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
    refine Set.mem_iUnion₂.2 ⟨r, Finset.mem_range.2 hr, ?_⟩
    simp only [hSeld, dif_pos hr]
    by_contra hnot
    apply hq
    refine ⟨hlen, o, ho, hcons, ?_⟩
    have hnot' : (q.1.1, q.2) ∉ SelBad O L n (Fintype.card σ + 2) t (some (tt, resCode o.result))
        ∧ (q.1.2, q.2) ∈ SelDk settings P.D L n r hr (some (tt, resCode o.result)) := by
      constructor
      · intro h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inl h⟩
      · by_contra h
        apply hnot
        show SelG r (stageSel settings O P.D Dsf Nn r hr q.1.1 q.1.2) ∧ _
        rw [hst]
        exact ⟨hG, hOk, Or.inr h⟩
    obtain ⟨hnb, hdk⟩ := hnot'
    obtain ⟨hinj, -⟩ := hdk
    have hsf : SuffixFree o.family := by
      have hlenF : ∀ v ∈ o.family, v.toList.length = Ls := fun v hv =>
        hOk.1.1 ⟨r, hr⟩ le_rfl v (hOk.2.1 (hF ▸ hv))
      intro v hv v' hv' hsuf
      apply FreeMonoid.toList.injective
      exact hsuf.eq_of_length ((hlenF v hv).trans (hlenF v' hv').symm)
    have hleaves : o.result.hypothesis.tree.leaves.length ≤ Fintype.card σ + 2 :=
      (Random.classSet_props _ _ hch.2.2.1).2
    have hgood : (q.1.1, q.2) ∉ scoreBad O o.result.hypothesis o.family o.state.lo
        (o.state.hi + 1) L n t := by
      intro h
      apply hnb
      show (q.1.1, q.2) ∈ (if SuffixFree tt.2.2.1
          ∧ (resCode o.result).hyp.tree.leaves.length ≤ Fintype.card σ + 2
        then scoreBad O (resCode o.result).hyp tt.2.2.1 tt.2.1.lo (tt.2.1.hi + 1) L n t else ∅)
      rw [hF, hB, hhyp, if_pos ⟨hsf, hleaves⟩]
      exact h
    -- the start's true leaf, and the family's reads on the strings of length `L`
    have hs := dfa_of_checks_tsrb M O hL o L hch (by linarith) hm ⟨q₀, hq₀w, hq₀e⟩
    have hn : 0 < n := by
      refine Nat.ceil_pos.2 (div_pos (mul_pos (by norm_num) (Real.log_pos ?_)) (by positivity))
      rw [one_lt_div hδ]
      nlinarith only [hQ0, hδ1, mul_nonneg hQ0 hQ0]
    have hu : ∫ x, readProb O o.family o.state.lo (o.state.hi + 1) x .undecided
        ∂(uniformStrings L) ≤ 3 / 8 * (N * εcov) := by
      refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
        fun x => measureReal_nonneg).trans ?_
      exact (mul_le_mul_of_nonneg_left hch.1.2.2 (zero_le_one.trans hN1)).trans
        (le_of_eq (by ring))
    have hmr : ∫ x, misreadProb O o.family o.state.lo (o.state.hi + 1) x ∂(uniformStrings L)
        ≤ 1025 / 1024 * (N * εcov) := by
      refine (integral_uniform_le_window (L₁ := L₁) hkL (by omega) _
        fun x => measureReal_nonneg).trans ?_
      exact (mul_le_mul_of_nonneg_left hch.1.2.1 (zero_le_one.trans hN1)).trans
        (le_of_eq (by ring))
    have h := pickLeaf_le M O o.result.hypothesis o.family o.state.lo (o.state.hi + 1) hL L n hn
      t (3 / 8 * (N * εcov)) (1025 / 1024 * (N * εcov)) _ q.1.1 q.2 hinj hgood hu hmr hs
    rw [hread]
    have hharm := tsrb_harm (A := A) (Q := Q) (N := N) (by linarith) hQ0 (by linarith) hm hm1 rfl
      hεc0.le hεcε
    have h' : (uniformStrings L).real {x | ¬ o.result.hypothesis.Classifies
        (pickLeaf o.result.hypothesis (fun z => familyRead O.mq o.family o.state.lo
          (o.state.hi + 1) z q.1.1) q.2) x (M.eval x.toList ∈ M.accept)}
        ≤ γ + N * (2 * ((A + 1) * (ε + ε / 96) + Q * (ε + ε / 96))
          + (2 * (1025 / 1024 * εcov) + 3 / 8 * εcov)) / m
          + 2 * t + 3 / 8 * (N * εcov) + 2 * (1025 / 1024 * (N * εcov)) := h
    refine h'.trans ?_
    rw [htd]
    linarith
  -- the loop's failure, and each round's selection
  have hBl : (μL.prod β).real (Prod.fst ⁻¹' Bloop) ≤ δ / 2 := by
    have e : (μL.prod β).real (Prod.fst ⁻¹' Bloop) = μL.real Bloop := by
      rw [← Set.prod_univ, measureReal_def, measureReal_def, Measure.prod_prod, measure_univ,
        mul_one]
    rw [e]
    refine hD.trans ?_
    show (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) ≤ δ / 2
    have e : (Q + 1) * (5 * (δ / (8 * (Q + 1))) / 2) = 5 * δ / 16 := by field_simp; ring
    rw [e]; linarith
  have hsel : ∀ r ∈ Finset.range (Fintype.card σ + 1), (μL.prod β).real (Sel r)
      ≤ δ / (2 * (Q + 1)) := by
    intro r hr
    have hr' := Finset.mem_range.1 hr
    simp only [hSeld, dif_pos hr']
    have hs0 : 0 < s := by show 0 < 1 / 2 - η₀; linarith
    have hs1 : s ≤ 1 / 2 := by
      show 1 / 2 - η₀ ≤ 1 / 2
      linarith
    have hp1 : pAP ≤ 1 := hpAPu.trans measureReal_le_one
    have hτ0 : 0 < τ := lt_min (mul_pos (div_pos hεc0 (by norm_num))
      (pow_pos (gateA_pos hη0 hη₀) 2)) hι0
    have hτ1 : τ ≤ 1 := (min_le_right _ _).trans (by linarith)
    have hW1 : 1 ≤ W := le_max_right _ _
    have hRB0 : 0 ≤ RB := rb_nonneg hQ0 hs0 hs1 hpAP hp1 hδ'0 hδ'1 hτ0 hτ1
      (Real.log_nonneg (by rw [le_div_iff₀ hδ'0]; nlinarith))
    refine (loop_select_round_le settings O P Dsf Ls L n (Fintype.card σ + 2) ⌊RB⌋₊ t
      (by positivity) (hwalk ▸ hkL) hnp r hr').trans ?_
    rw [hwalk]
    have e1 : ((Fintype.card σ + 2 : ℕ) : ℝ) = Q + 2 := by push_cast; rfl
    have e2 : ((Fintype.card σ + 1 : ℕ) : ℝ) + 1 = Q + 2 := by push_cast; ring
    rw [e1, e2, htd]
    exact dfa_select_arith hQ0 hA2' hδ hξ hkL (Nat.le_ceil _) hbud (Nat.floor_le hRB0) hk3
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
