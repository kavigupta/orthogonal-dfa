import RLStar.Proofs.ClaimRLStar
import RLStar.Proofs.TrieTol
import RLStar.Proofs.ClaimRLStarTrieArith

/-!
# The learner with local walk starts, asymptotically: the proof

`trie_loop_tol` at main's rates with `R = 1` and `β = 1/|Σ|`, its walk start the least `K` with
`2³⁰ loopTolInv⁹ ≤ |Σ|^K`. As in `rlStarClaimUniform_of_rlStarClaim`, `loopTolInv` is at most
`C₀ (L + |Q|)⁶ loopG`, so `L ≥ L₀` gives `K ≤ ⌈L/2⌉` (`pow_of_tangent_m` at `m = 9`). The totals
are main's `distPref`, `distSuff`, `distProbes` and `distReads` at `R = 1`, `β = 1/|Σ|`.
-/

noncomputable section

set_option maxHeartbeats 400000

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

def trieInputs (x : RLStarUniformInputs) : RLStarInputs := ⟨x.Q, x.A, x.δ, 1, x.ξ, x.pAP, x.s₀⟩

def trieRun (r : RLStarUniformRun) : RLStarDistRun :=
  { toRLStarDistInputs := toDist (trieInputs r.toRLStarUniformInputs), L := r.L }

def trieRunR (r : RLStarUniformRun) : RLStarRun :=
  { toRLStarInputs := trieInputs r.toRLStarUniformInputs, L := r.L }

lemma trieInputs_mem {x : RLStarUniformInputs} (hx : x ∈ rlStarUniformRegion) :
    trieInputs x ∈ rlStarRegion := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := hx
  exact ⟨h1, h2, h3, h4, le_rfl, h5, h6, h7, h8, h9, h10⟩

lemma learnerReads_zero_le {K a b f T q B lg A n : ℝ} (hn : 0 ≤ n) (hf : 0 ≤ f) :
    learnerReads K a b f T q B lg A 0 ≤ learnerReads K a b f T q B lg A n := by
  unfold learnerReads readsStart
  nlinarith [mul_nonneg hn hf]

theorem rlStarClaimUniform_holds : RLStarClaimUniform := by
  obtain ⟨C, hC⟩ := isBigO_principal.1 distP₀Tol_isBigO
  set C₁ : ℝ := 256 * max 1 C with hC₁d
  have hC₁ : 1 ≤ C₁ := by rw [hC₁d]; linarith [le_max_left 1 C]
  have hmapsR : Set.MapsTo trieRun rlStarUniformRunRegion rlStarDistRunRegion :=
    fun r hr => toDist_mem (trieInputs_mem hr)
  refine ⟨fun x => 1 + 4 * 9 * (log (C₁ * loopG (trieInputs x)) + 6 * log (x.Q + 4)
      + 6 * log 9 + 24) / log x.A,
    fun r => distPref (trieRun r), fun r => distSuff (trieRun r),
    fun r => distProbes (trieRun r), fun r => distReads (trieRun r), ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine isBigO_on (9 * (263 + 6 * log (9 ^ 6 * C₁))) fun x hx => ?_
    obtain ⟨hQ, hA, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
    have hQ' : (1 : ℝ) ≤ x.Q := by exact_mod_cast hQ
    have hA' : (2 : ℝ) ≤ x.A := by exact_mod_cast hA
    have hC9 : 1 ≤ 9 ^ 6 * C₁ := by linarith
    obtain ⟨hG, hle⟩ := loopL0_le (R := 1) (iδ := 1 / x.δ) (iξ := 1 / x.ξ) (ip := 1 / x.pAP)
      (is := 1 / x.s₀) hQ' hA' le_rfl ((le_div_iff₀ hδ).2 (by linarith))
      ((le_div_iff₀ hξ).2 (by linarith)) ((le_div_iff₀ hp).2 (by linarith))
      ((le_div_iff₀ hs).2 (by linarith)) hC9
    change 1 ≤ loopG (trieInputs x) at hG
    change 1 + 4 * (log (9 ^ 6 * C₁ * loopG (trieInputs x)) + 6 * log ((x.Q : ℝ) + 4) + 24)
        / log x.A ≤ (263 + 6 * log (9 ^ 6 * C₁))
          * (1 + log ((x.Q : ℝ) * 1 * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀))
            / log x.A) at hle
    have hlA : 0 < log (x.A : ℝ) := Real.log_pos (by linarith)
    have hl9 : log (9 ^ 6 * C₁ * loopG (trieInputs x))
        = 6 * log 9 + log (C₁ * loopG (trieInputs x)) := by
      rw [mul_assoc, Real.log_mul (by norm_num)
        (ne_of_gt (by linarith [one_le_mul_of_one_le_of_one_le hC₁ hG])), Real.log_pow]
      push_cast; ring
    have e : (x.Q : ℝ) * 1 * (1 / x.δ) * (1 / x.ξ) * (1 / x.pAP) * (1 / x.s₀)
        = x.Q / (x.δ * x.ξ * x.pAP * x.s₀) := by ring
    rw [hl9, e] at hle
    have hlCG : 0 ≤ log (C₁ * loopG (trieInputs x)) :=
      Real.log_nonneg (one_le_mul_of_one_le_of_one_le hC₁ hG)
    have hlQ : 0 ≤ log ((x.Q : ℝ) + 4) := Real.log_nonneg (by linarith)
    have hl9' : 0 ≤ log (9 : ℝ) := Real.log_nonneg (by norm_num)
    have hZ : 1 ≤ (x.Q : ℝ) / (x.δ * x.ξ * x.pAP * x.s₀) := by
      rw [le_div_iff₀ (by positivity), one_mul]
      have : x.δ * x.ξ * x.pAP * x.s₀ ≤ 1 := by
        have : x.δ * x.ξ ≤ 1 := by nlinarith
        have : x.δ * x.ξ * x.pAP ≤ 1 := by nlinarith
        nlinarith
      linarith
    have hT0 : 0 ≤ 1 + log ((x.Q : ℝ) / (x.δ * x.ξ * x.pAP * x.s₀)) / log x.A :=
      add_nonneg zero_le_one (div_nonneg (Real.log_nonneg hZ) hlA.le)
    set u := 4 * (6 * log 9 + log (C₁ * loopG (trieInputs x)) + 6 * log ((x.Q : ℝ) + 4) + 24)
      / log x.A with hud
    have hu0 : 0 ≤ u := div_nonneg (by linarith) hlA.le
    have eL : 1 + 4 * 9 * (log (C₁ * loopG (trieInputs x)) + 6 * log ((x.Q : ℝ) + 4)
        + 6 * log 9 + 24) / log x.A = 1 + 9 * u := by rw [hud]; ring
    refine ⟨?_, hT0, ?_⟩
    · show 0 ≤ 1 + 4 * 9 * (log (C₁ * loopG (trieInputs x)) + 6 * log ((x.Q : ℝ) + 4)
        + 6 * log 9 + 24) / log x.A
      rw [eL]; linarith
    · show 1 + 4 * 9 * (log (C₁ * loopG (trieInputs x)) + 6 * log ((x.Q : ℝ) + 4)
        + 6 * log 9 + 24) / log x.A ≤ _
      rw [eL]
      linarith
  · refine (isBigO_comp_on distPref_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, trieRun, trieInputs, toDist, one_div_one_div]
    obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs
    rw [e, e']
    simp only [mul_one, one_pow]
  · refine (isBigO_comp_on distSuff_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, trieRun, trieInputs, toDist]
    rw [(uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs).2]
    simp only [mul_one]
  · refine (isBigO_comp_on distProbes_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    simp only [Function.comp, trieRun, trieInputs, toDist, one_div_one_div]
    ring
  · refine (isBigO_comp_on distReads_isBigO hmapsR).congr' EventuallyEq.rfl
      (eventually_principal.2 fun r hr => ?_)
    obtain ⟨-, hA, hδ, -, hξ, -, hp, -, hs, -⟩ := hr
    have hA' : (0 : ℝ) < r.A := by exact_mod_cast (show 0 < r.A by omega)
    simp only [Function.comp, trieRun, trieInputs, toDist, shCluster, shSearch, shHarvest, shStart,
      shPref, shSuff, shProbe, shHeld, shFam, shRho, shLg, one_div_one_div]
    obtain ⟨e, e'⟩ := uniform_rewrite (X := (r.L : ℝ) + r.Q) (R := 1) hA' hδ hξ hp hs
    rw [e, e']
    ring
  -- the run
  intro α _ _ _ σ _ Ω _ μ _ x
  obtain ⟨Q, A, δ, ξ, pAP, s₀⟩ := x
  intro M O L Ls hx hQ hA hLang hη hpAPu hL₀
  subst hQ hA
  have hx' := hx
  obtain ⟨hQ1, hA2, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1⟩ := hx
  have hQ' : (1 : ℝ) ≤ Fintype.card σ := by exact_mod_cast hQ1
  have hA' : (2 : ℝ) ≤ Fintype.card α := by exact_mod_cast hA2
  set xi := trieInputs ⟨Fintype.card σ, Fintype.card α, δ, ξ, pAP, s₀⟩ with hxid
  set r : RLStarDistRun := trieRun ⟨⟨Fintype.card σ, Fintype.card α, δ, ξ, pAP, s₀⟩, L⟩ with hrd
  have hr : r ∈ rlStarDistRunRegion :=
    @hmapsR ⟨⟨Fintype.card σ, Fintype.card α, δ, ξ, pAP, s₀⟩, L⟩ hx'
  have hxi : xi ∈ rlStarRegion := trieInputs_mem hx'
  have hG1 : 1 ≤ loopG xi := loopG_ge_one hxi
  have hη₀ : 1 / 2 - s₀ < 1 / 2 := by linarith
  have hc0 : 0 < ξ / (2 * 1) := by positivity
  have hc1 : ξ / (2 * 1) ≤ 1 := by linarith
  set T := loopTolInv ((L : ℝ) + Fintype.card σ + 4) (Fintype.card σ) (Fintype.card α)
    (1 / (1 / (Fintype.card α : ℝ))) (Fintype.card σ) (Fintype.card α) δ (ξ / (2 * 1)) (ξ / 4)
    pAP (1 / 2 - s₀) with hTd
  have hT1 : 1 ≤ T := (loopTolInv_le _ _ (by have : (0 : ℝ) ≤ L := Nat.cast_nonneg _; linarith)
    (Nat.cast_nonneg _) (Nat.cast_nonneg _) (by rw [one_div_one_div]; exact hA') hδ (by linarith)
    hc0 hc1 (by positivity) hp hp1 (by linarith) hη₀).1
  -- `T` within `16 max 1 C (L + |Q|)⁶ loopG`
  have hTle : T ≤ 16 * max 1 C * (((L : ℝ) + Fintype.card σ) ^ 6 * loopG xi) := by
    have hc : ‖1 / distP₀Tol r‖ ≤ C * ‖p₀Shape r‖ := hC r hr
    have e : 1 / distP₀Tol r = max 1 T := by
      show 1 / (1 / max 1 T) = max 1 T
      rw [one_div_one_div]
    obtain ⟨hs0, hsle⟩ := p₀Shape_le hr
    obtain ⟨-, hsh⟩ := shape_toDist_le hxi L
    rw [e, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_pos (lt_max_of_lt_left one_pos),
      abs_of_nonneg hs0] at hc
    have h8 : p₀Shape r ≤ 8 * (2 * (((L : ℝ) + Fintype.card σ) ^ 6 * loopG xi)) :=
      hsle.trans (mul_le_mul_of_nonneg_left hsh (by norm_num))
    have hCC : C ≤ max 1 C := le_max_right _ _
    have hm : T ≤ max 1 T := le_max_right _ _
    have := (mul_le_mul_of_nonneg_right hCC hs0).trans
      (mul_le_mul_of_nonneg_left h8 (by linarith [le_max_left 1 C] : (0 : ℝ) ≤ max 1 C))
    linarith
  have hZ : 2 ^ 30 * T ^ 9 ≤ (((L : ℝ) + Fintype.card σ) ^ 6 * (C₁ * loopG xi)) ^ 9 := by
    have hy : 0 ≤ 16 * max 1 C * (((L : ℝ) + Fintype.card σ) ^ 6 * loopG xi) := by
      have : (0 : ℝ) ≤ max 1 C := by linarith [le_max_left 1 C]
      have : (0 : ℝ) ≤ loopG xi := by linarith
      positivity
    calc 2 ^ 30 * T ^ 9
        ≤ 2 ^ 30 * (16 * max 1 C * (((L : ℝ) + Fintype.card σ) ^ 6 * loopG xi)) ^ 9 :=
          mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by linarith) hTle 9) (by norm_num)
      _ ≤ 16 ^ 9 * (16 * max 1 C * (((L : ℝ) + Fintype.card σ) ^ 6 * loopG xi)) ^ 9 :=
          mul_le_mul_of_nonneg_right (by norm_num) (pow_nonneg hy 9)
      _ = _ := by rw [hC₁d]; ring
  have hpow := pow_of_tangent_m (A := (Fintype.card α : ℝ)) (Q := (Fintype.card σ : ℝ))
    (E := C₁ * loopG xi) (L := L) (m := 9) hA' hQ' (one_le_mul_of_one_le_of_one_le hC₁ hG1)
    (by norm_num) (by simpa only [Nat.cast_ofNat] using hL₀)
  -- the walk start
  have hA1 : 1 < Fintype.card α := by
    have : 2 ≤ Fintype.card α := hA2
    omega
  obtain ⟨K, hKd⟩ : ∃ K, K = Nat.clog (Fintype.card α) ⌈(2 : ℝ) ^ 30 * T ^ 9⌉₊ := ⟨_, rfl⟩
  have hK1 : 2 ^ 30 * T ^ 9 ≤ (Fintype.card α : ℝ) ^ K := by
    have h : ⌈(2 : ℝ) ^ 30 * T ^ 9⌉₊ ≤ Fintype.card α ^ K := hKd ▸ Nat.le_pow_clog hA1 _
    have h' : ((⌈(2 : ℝ) ^ 30 * T ^ 9⌉₊ : ℕ) : ℝ) ≤ ((Fintype.card α ^ K : ℕ) : ℝ) :=
      Nat.cast_le.2 h
    rw [Nat.cast_pow] at h'
    exact (Nat.le_ceil _).trans h'
  have hK2 : K ≤ (L + 1) / 2 := by
    rw [hKd]
    refine Nat.clog_le_of_le_pow (Nat.ceil_le.2 ?_)
    rw [Nat.cast_pow]
    exact hZ.trans hpow
  have hLs : L + Fintype.card σ + 3 + Fintype.card σ + 3 = L + 2 * Fintype.card σ + 6 := by omega
  obtain ⟨settings, -, hprobe, hprob⟩ := trie_loop_tol (Ω := Ω) (μ := μ) M O L K ((L + 1) / 2)
    (1 / 2 - s₀) (ξ / (2 * 1)) δ pAP (ξ / 4) ξ (1 / (1 / (Fintype.card α : ℝ))) hLang hη hη₀
    hc0 hc1 hδ (by linarith) hp (by positivity) hA2 (le_of_eq (one_div_one_div _).symm)
    (by omega) (by rw [hLs]; exact hpAPu) hK1 (by linarith)
  refine ⟨settings, fun p rounds hrnd => hprobe p rounds (by rw [hLs]; exact hrnd), ?_⟩
  refine (measureReal_mono fun p hp => ?_).trans hprob
  intro h
  rw [hLs] at h
  exact hp (trieSucceeds_mono (learnerReads_zero_le (Nat.cast_nonneg _) (fam_pw hr).1) h)

theorem rlStarClaimUniformSimple_holds : RLStarClaimUniformSimple := by
  obtain ⟨L₀, pref, suff, probes, reads, hL₀, hp, hs, hpr, hrd, hw⟩ := rlStarClaimUniform_holds
  refine ⟨L₀, pref, suff, probes, reads, hL₀,
    hp.trans (isBigO_on 4 fun r hr => ?_), hs.trans (isBigO_on 2 fun r hr => ?_), hpr,
    hrd.trans (isBigO_on 32 fun r hr => ?_), hw⟩
  · have h := loop_pref_dom (r := trieRunR r) (trieInputs_mem hr)
    simp only [trieRunR, trieInputs, mul_one, one_pow] at h
    exact h
  · have h := loop_suff_dom (r := trieRunR r) (trieInputs_mem hr)
    simp only [trieRunR, trieInputs, mul_one] at h
    exact h
  have hr' : trieRunR r ∈ rlStarRunRegion := trieInputs_mem hr
  obtain ⟨hp0, hp0', hp1⟩ := loop_pref_dom hr'
  obtain ⟨hs0, hs0', hs1⟩ := loop_suff_dom hr'
  obtain ⟨hQ, hF, hPr, hlg, hρ, hHd⟩ := uniform_factors hr'
  obtain ⟨hPrle, hHdle, hlg2⟩ := uniform_search_start_le hr'
  have h4 := reads_dom hQ hp0 hp1 hs0 hs1 hF hPr hlg hρ hHd
  have h2 := reads_two_term hQ hp0' hs0' hF hPr hlg2 hρ hHd hPrle hHdle
  simp only [trieRunR, trieInputs, mul_one, one_mul, one_pow] at h4 h2
  dsimp only
  refine ⟨h4.1, h2.2.1, h4.2.2.trans ?_⟩
  linarith [h2.2.2]

end OrthoDFA

end
