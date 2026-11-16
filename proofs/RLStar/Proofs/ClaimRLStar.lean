import RLStar.Claims.RLStar
import RLStar.Proofs.TrieTol
import RLStar.Proofs.ClaimRLStarUniform

/-!
# The learner's costs, asymptotically: the proof

`trie_loop_tol` bounds, for every draw, the loop's reads by `(|Q| + 1)` rounds of
`6 (|Q| + 2) trieCount` prefixes and `triePool + 1` suffixes, and its probes by
`(|Q| + 1) trieBudget`, all at `ℓ = L + |Q| + 4`, once `D` holds at most `1/(2⁴ trieTolInv⁵)` on
any first `k` letters; but for chance `δ` the run succeeds, asking the oracle at most
`learnerReads` strings.
The claim runs it at `η₀ = 1/2 − s₀/2`, the rate `ret` reads a declared `1/2 − s₀` at, so the
gap `2 s₀` is `4 (1/2 − η₀)`, and at `c = ξc/2` and `ξc/4`, `ξc = min(ξ, (L + |Q|) β)`, so the
clustering's accuracy `ξc/4` is within `(L + |Q|) β`. `ClaimRLStarArith` and `ClaimReadsArith`
bound each total within a constant of its shape (`run_tc`).
`RLStarClaimUniform` and `RLStarClaimUniformSimple` follow (`ClaimRLStarUniform`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Asymptotics Filter Real

def runTol (r : RLStarRun) : ℝ :=
  trieTolInv (r.L + r.Q + 4) r.Q r.A r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP (1 / 2 - r.s₀ / 2)

def runPref (r : RLStarRun) : ℝ :=
  (r.Q + 1) * (6 * (r.Q + 2)
    * trieCount (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP (1 / 2 - r.s₀ / 2))

def runSuff (r : RLStarRun) : ℝ :=
  (r.Q + 1) * (triePool (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP
    (1 / 2 - r.s₀ / 2) + 1)

def runProbes (r : RLStarRun) : ℝ := (r.Q + 1) * trieBudget (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2)

def runReads (r : RLStarRun) : ℝ :=
  learnerReads (r.Q + 1)
    (6 * (r.Q + 2)
      * trieCount (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP (1 / 2 - r.s₀ / 2))
    (triePool (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP (1 / 2 - r.s₀ / 2) + 1)
    (trieFam (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) (1 / 2 - r.s₀ / 2))
    (trieBudget (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2)) r.Q
    ((⌈trieHB (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP (1 / 2 - r.s₀ / 2)⌉₊ : ℕ)
      : ℝ)
    ((3 + Nat.clog 2 (r.L + r.Q + 3) : ℕ) : ℝ) r.A 0

lemma tol_bigO : (fun r => 1 / ((1 / max 1 (runTol r)) ^ 5 / 2 ^ 4)) =O[𝓟 rlStarRunRegion]
    (fun r => shTol r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀ ^ 5) := by
  refine isBigO_on (2 ^ 4 * (2 ^ 138) ^ 5) fun r hr => ?_
  have ht := run_tc hr
  obtain ⟨hT1, hT⟩ := tc_tol ht r.Q r.A rfl rfl
  obtain ⟨-, hs0, -⟩ := tc_tolPow ht
  change 1 ≤ runTol r at hT1
  change runTol r ≤ _ at hT
  beta_reduce
  rw [max_eq_right hT1]
  have e : 1 / ((1 / runTol r) ^ 5 / 2 ^ 4) = 2 ^ 4 * runTol r ^ 5 := by
    have : runTol r ≠ 0 := by linarith
    field_simp
  rw [e]
  have hp := pow_le_pow_left₀ (by linarith) hT 5
  refine ⟨by positivity, by positivity, ?_⟩
  calc 2 ^ 4 * runTol r ^ 5 ≤ 2 ^ 4 * (2 ^ 138 * shTol r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀) ^ 5
        := by gcongr
    _ = 2 ^ 4 * (2 ^ 138) ^ 5 * shTol r.Q r.A (r.L + r.Q) r.δ r.ξc r.pAP r.s₀ ^ 5 := by ring

lemma reads_bigO : runReads =O[𝓟 rlStarRunRegion] RLStarRun.nReads := by
  refine isBigO_on (2 ^ 112) fun r hr => ?_
  have ht := run_tc hr
  obtain ⟨hHB1, -⟩ := tc_hb ht
  have hQ : 1 ≤ r.Q := hr.1
  have hB : ((⌈trieHB (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP
      (1 / 2 - r.s₀ / 2)⌉₊ : ℕ) : ℝ)
        ≤ trieHB (r.L + r.Q + 4) r.Q r.A r.δ (r.ξc / 2) (r.ξc / 4) r.pAP (1 / 2 - r.s₀ / 2) + 1 :=
    (Nat.ceil_lt_add_one (by linarith)).le
  exact tc_reads ht _ _ (Nat.cast_nonneg _) (lg_le (L := r.L) hQ) (Nat.cast_nonneg _) hB

theorem rlStarClaim_holds : RLStarClaim := by
  refine ⟨fun r => (1 / max 1 (runTol r)) ^ 5 / 2 ^ 4, runPref, runSuff, runProbes, runReads,
    tol_bigO, isBigO_on (2 ^ 77) fun r hr => tc_prefs (run_tc hr),
    isBigO_on (2 ^ 15) fun r hr => tc_suffs (run_tc hr),
    isBigO_on (2 ^ 18) fun r hr => tc_probes (run_tc hr), reads_bigO,
    fun r => div_pos (pow_pos (one_div_pos.2 (lt_of_lt_of_le one_pos (le_max_left _ _))) 5)
      (by norm_num), ?_⟩
  -- the run
  intro α _ _ _ σ _ Ω _ μ _ x
  obtain ⟨Q, A, δ, ξ, pAP, s₀, β⟩ := x
  intro M O L k D _ L₁ Ls hx hQ hA hLang hgap hbL hbN hlong hkL hpre hpAPu
  subst hQ hA
  have ht := run_tc (r := ⟨⟨Fintype.card σ, Fintype.card α, δ, ξ, pAP, s₀, β⟩, L⟩) hx
  have hT1 := (tc_tol ht (Fintype.card σ) (Fintype.card α) rfl rfl).1
  obtain ⟨hQ1, hA2, hδ, hδ2, hξ, hξ1, hp, hp1, hs, hs1, hβ⟩ := hx
  have hgap' : 2 * s₀ ≤ O.gap := hgap
  have hbL' : β ≤ (prefixLaw D L).real {p | p ∈ O.L} := hbL
  have hbN' : β ≤ (prefixLaw D L).real {p | p ∉ O.L} := hbN
  have hLQ : (1 : ℝ) ≤ (L : ℝ) + Fintype.card σ := by
    have : (1 : ℝ) ≤ Fintype.card σ := by exact_mod_cast hQ1
    have : (0 : ℝ) ≤ L := Nat.cast_nonneg _
    linarith
  set ξc : ℝ := min ξ (((L : ℝ) + Fintype.card σ) * β) with hξcd
  have hξc0 : 0 < ξc := lt_min hξ (mul_pos (by linarith) hβ)
  have hξc1 : ξc ≤ 1 := (min_le_left _ _).trans hξ1
  have hξcξ : ξc ≤ ξ := min_le_left _ _
  have hξcβ : ξc ≤ ((L : ℝ) + Fintype.card σ) * β := min_le_right _ _
  have hbL'' := mul_le_mul_of_nonneg_left hbL' (by linarith : (0 : ℝ) ≤ (L : ℝ) + Fintype.card σ)
  have hbN'' := mul_le_mul_of_nonneg_left hbN' (by linarith : (0 : ℝ) ≤ (L : ℝ) + Fintype.card σ)
  have hη₀ : 1 / 2 - s₀ / 2 < 1 / 2 := by linarith
  set T := trieTolInv ((L : ℝ) + Fintype.card σ + 4) (Fintype.card σ) (Fintype.card α)
      (Fintype.card σ) (Fintype.card α) δ (ξc / 2) (ξc / 4) pAP (1 / 2 - s₀ / 2) with hTd
  change 1 ≤ T at hT1
  have hTm : max 1 T = T := max_eq_right hT1
  -- `D`'s first `k` letters weigh at most `1/Ak`
  have hpre' : ∀ u, D.real {w | prefixOf w k = u} ≤ (2 ^ 4 * T ^ 5)⁻¹ := by
    intro u
    refine (hpre u).trans (le_of_eq ?_)
    change (1 / max 1 T) ^ 5 / 2 ^ 4 = _
    rw [hTm, one_div_pow, div_div, one_div, mul_comm]
  obtain ⟨settings, hwk, hprobe, hprob⟩ := trie_loop_tol (Ω := Ω) (μ := μ) M O L k k
    (1 / 2 - s₀ / 2) (ξc / 2) δ pAP (ξc / 4) ξ (2 ^ 4 * T ^ 5) D hLang (by linarith) hη₀
    (by linarith) (by linarith) (by positivity) (by linarith) hδ (by linarith) hp (by positivity)
    hA2 hkL le_rfl hlong hpre' hpAPu le_rfl (by linarith)
  exact ⟨settings, hwk, hprobe, hprob⟩

theorem rlStarClaimUniform_holds : RLStarClaimUniform :=
  rlStarClaimUniform_of_rlStarClaim rlStarClaim_holds

theorem rlStarClaimUniformSimple_holds : RLStarClaimUniformSimple :=
  rlStarClaimUniformSimple_of_claim rlStarClaimUniform_holds

end OrthoDFA

end
