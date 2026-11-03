import OrthoDFA.Proofs.LoopSimple
import OrthoDFA.Proofs.LoopQueries
import OrthoDFA.LowLevelClaims.LoopQueries

/-!
# `learner_loop_total_queries`: the loop's membership queries

A companion to `learner_loop_dfa_simple_holds`, counting oracle calls (invocations, with
multiplicity), not distinct strings. The loop asks the oracle at three kinds of place, each a
`familyRead` of `|F|` calls:

* its counterexample search's level-3 run, `level3Count` (`round_level3_count_le`);
* its clustering's sifts, `schedReads` (`schedReads_card_le`, `≤ |prefixes| · (|suffixes| + 1)`);
* generating each round's prefixes, which sifts a draw through the prior hypothesis
  (`round_harvest_card_le`, `≤ probeSifts L · (|Q| + 2) · |F|` per draw).

On the loop's success event (probability `≥ 1 - (|Q| + 1) · 5δ'/2`) every round's hypothesis tree is
in `classSet |Q|` (at most `|Q| + 2` leaves), so by `round_level3_count_le` every round makes at
most the level-3 polynomial of calls. It sums the three sources over the at
most `|Q| + 1` rounds, each round's level-3 calls bounded by that polynomial and its harvest draw
count `3·|pops|·∑ npref` by the schedule bound `3(|Q| + 2)·2⌊RB⌋`.
-/

namespace OrthoDFA

open Random MeasureTheory


theorem learner_loop_total_queries_holds : LearnerLoopTotalQueries := by
  intro α _ _ _ σ _ Ω _ μ _ M O P E η₀ εcov ε δ' pAP k
  intro Q A L ι s τ J S B₃ Wm W PB RB cap Kc
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2 hk3
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hreads, hmain⟩ :=
    loop_simple_checks M O P E η₀ εcov ε δ' pAP k
      hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2 hk3
  refine ⟨settings, hwalk, hdis, hbud, hnp, hreads, ?_⟩
  intro Dsf hDsf hlen hpAPD hatom
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (hmain Dsf hDsf hlen hpAPD hatom).2
  intro p hp
  set rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p) with hrounds
  simp only [Set.mem_setOf_eq, not_forall, _root_.not_imp, not_le] at hp ⊢
  obtain ⟨xs, hreal, hsum⟩ := hp
  obtain ⟨i, -, hlt⟩ := Finset.exists_lt_of_sum_lt hsum
  have hharv : 3 * (loopPopulations (Fintype.card σ + 1) i.val).card
        * (∑ B ∈ settings.schedule (loopPopulations (Fintype.card σ + 1) i.val), B.npref)
      ≤ 3 * (Fintype.card σ + 2) * (2 * ⌊RB⌋₊) :=
    Nat.mul_le_mul (Nat.mul_le_mul (le_refl 3) ((Finset.card_le_univ _).trans_eq (by simp)))
      (hnp _)
  have hlevel : settings.level3.probeBudget * (Kappa.probeSifts P.L * (Fintype.card σ + 2)
        + (Fintype.card σ + 2) * Fintype.card α * (Fintype.card σ + 2)) * (rounds.get i).family.card
      < level3Count O (rounds.get i).family (rounds.get i).state.lo
          ((rounds.get i).state.hi + 1) settings.level3 (xs i) p.1 := by
    by_contra hle
    exact absurd (Nat.add_le_add (Nat.add_le_add (not_lt.1 hle) le_rfl)
      (Nat.mul_le_mul hharv le_rfl)) (not_le.2 hlt)
  refine ⟨i.val, Nat.lt_succ_iff.1 (Nat.lt_of_lt_of_le i.isLt
      (length_loopRounds_le settings O P.D Dsf (persistDraws p) (Fintype.card σ + 1))), i.isLt,
    fun htree => ?_⟩
  exact absurd (round_level3_count_le O settings.level3 _ _ _ (xs i) p.1 (hreal i).1
    (by rw [← (hreal i).2]; exact htree)) (not_le.2 hlevel)

open Asymptotics Filter in
/-- As the state count grows, a round's level-3 query bound is `O(|Q|²)` (the probe budget,
alphabet, family size and probe-length sift count held fixed). -/
theorem level3_query_isBigO (T A F L : ℕ) :
    (fun nQ : ℕ => (T * (Kappa.probeSifts L * (nQ + 2) + (nQ + 2) * A * (nQ + 2)) * F : ℝ))
      =O[atTop] (fun nQ : ℕ => ((nQ : ℝ)) ^ 2) := by
  refine Asymptotics.IsBigO.of_bound (T * F * (Kappa.probeSifts L + A) * 9) ?_
  filter_upwards [Filter.eventually_ge_atTop 1] with nQ hnQ
  have hnQ1 : (1 : ℝ) ≤ (nQ : ℝ) := by exact_mod_cast hnQ
  have hps : (0 : ℝ) ≤ (Kappa.probeSifts L : ℝ) := by positivity
  rw [Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity)]
  have hinner : (Kappa.probeSifts L : ℝ) * ((nQ : ℝ) + 2)
        + ((nQ : ℝ) + 2) * (A : ℝ) * ((nQ : ℝ) + 2)
      ≤ ((Kappa.probeSifts L : ℝ) + (A : ℝ)) * 9 * (nQ : ℝ) ^ 2 := by
    nlinarith [mul_nonneg hps (by nlinarith [hnQ1] : (0:ℝ) ≤ 9 * (nQ:ℝ)^2 - ((nQ:ℝ) + 2)),
      mul_nonneg (by positivity : (0:ℝ) ≤ (A:ℝ))
        (by nlinarith [hnQ1] : (0:ℝ) ≤ 9 * (nQ:ℝ)^2 - ((nQ:ℝ) + 2) * ((nQ:ℝ) + 2)),
      hnQ1, hps]
  calc ((T : ℝ) * ((Kappa.probeSifts L : ℝ) * ((nQ : ℝ) + 2)
          + ((nQ : ℝ) + 2) * (A : ℝ) * ((nQ : ℝ) + 2)) * (F : ℝ))
      = ((T : ℝ) * (F : ℝ)) * ((Kappa.probeSifts L : ℝ) * ((nQ : ℝ) + 2)
          + ((nQ : ℝ) + 2) * (A : ℝ) * ((nQ : ℝ) + 2)) := by ring
    _ ≤ ((T : ℝ) * (F : ℝ)) * (((Kappa.probeSifts L : ℝ) + (A : ℝ)) * 9 * (nQ : ℝ) ^ 2) :=
        mul_le_mul_of_nonneg_left hinner (by positivity)
    _ = (T : ℝ) * (F : ℝ) * ((Kappa.probeSifts L : ℝ) + (A : ℝ)) * 9 * (nQ : ℝ) ^ 2 := by ring

end OrthoDFA
