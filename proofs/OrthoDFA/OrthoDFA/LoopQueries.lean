import OrthoDFA.Proofs.LoopSimple
import OrthoDFA.Proofs.LoopQueries

/-!
# `learner_loop_queries`: the loop's membership queries

A companion to `learner_loop_dfa_simple_holds`, counting oracle calls (invocations, with
multiplicity), not distinct strings. The loop asks the oracle at three kinds of place, each a
`familyRead` of `|F|` calls:

* its counterexample search's level-3 run, `level3Count` (`round_level3_count_le`);
* its clustering's sifts, `schedReads` (`schedReads_card_le`, `≤ |prefixes| · (|suffixes| + 1)`);
* generating each round's prefixes, which sifts a draw through the prior hypothesis
  (`harvest_reads_card_le`, `≤ (3 + ⌈log₂ L⌉)·|mids|·|F|` per draw).

The last is the model's form of the paper's resampling term and is a real oracle cost, not free: the
measure-based loop draws from the normalized population rather than by rejection, so there is no
literal `1/ρ`, but the sift that produces each drawn prefix is counted.

On the loop's success event (probability `≥ 1 - (|Q| + 1) · 5δ'/2`) every round's hypothesis tree
has at most `|Q| + 2` leaves, so by `round_level3_count_le` every round `i ≤ |Q|` and every probe
sequence `xs` realizing its result with probes at most `P.L` long makes at most

    T · ((3 + ⌈log₂ P.L⌉)(|Q| + 2) + (|Q| + 2)|Σ|(|Q| + 2)) · |F_i|

level-3 calls, `T` the probe budget and `F_i` the round's family. Summed with the clustering and
prefix-generation over the at most `|Q| + 1` rounds this is the loop's total.

`learner_loop_total_queries` is that sum: on the same event it bounds

    ∑ᵢ [ level-3 calls + clustering reads + harvest sifts ]

over the rounds, each round's level-3 calls by the polynomial above and each round's harvest draw
count `3·|pops|·∑ npref` by the schedule bound `3(|Q| + 2)·2⌊RB⌋`. The clustering factor and the
per-draw harvest cost stay the round's own cardinalities.
-/

namespace OrthoDFA

open Random MeasureTheory

/-- Every round's level-3 counterexample search makes at most the level-3 polynomial of oracle
calls, with probability at least `1 - (|Q| + 1) · 5δ'/2`. Clustering (`schedReads_card_le`) and
prefix generation (`harvest_reads_card_le`) add their own calls per round. -/
theorem learner_loop_queries {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov ε δ' pAP : ℝ) (k : ℕ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
    let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) (1 + (Q + 1) * δ' / (7 * ε))
    let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 2)
    let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
          (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
        (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
          (ι / 2 / (8 * J * (RB + 1)))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ * (Q + 3) ^ ((Fintype.card σ + 2)
      * Fintype.card α)
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 → 0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' < 1 → 0 < pAP → 1 ≤ P.L → 1 ≤ k →
    (2 + Nat.clog 2 P.L) * ε < 3 / 8 →
    prefixMax P.D k ≤ cap * ε / 1536 →
    1536 * Real.log (16 * (Q + 1) * Kc / δ') * prefixMax P.D k ≤ ε →
    16 * B₃ ^ 2 * prefixMax P.D k ≤ δ' →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * ⌊RB⌋₊) ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).real
            {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
              ¬ ∀ (i : ℕ) (hi : i < rounds.length), i ≤ Fintype.card σ →
                  ∀ xs : Fin settings.level3.probeBudget → FreeMonoid α,
                    (∀ j, (xs j).toList.length ≤ P.L) →
                    rounds[i].result = indecisiveLStarLevel3 settings.level3
                      (fun z => familyRead O.mq rounds[i].family rounds[i].state.lo
                        (rounds[i].state.hi + 1) z p.1) xs →
                    level3Count O rounds[i].family rounds[i].state.lo
                        (rounds[i].state.hi + 1) settings.level3 xs p.1
                      ≤ settings.level3.probeBudget * ((3 + Nat.clog 2 P.L) * (Fintype.card σ + 2)
                        + (Fintype.card σ + 2) * Fintype.card α * (Fintype.card σ + 2))
                        * rounds[i].family.card}
          ≤ (Fintype.card σ + 1 : ℝ) * (5 * δ' / 2) := by
  intro Q A ι s τ J S B₃ Wm W PB RB cap Kc
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2 hk3
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hmain⟩ := loop_simple_checks M O P η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2 hk3
  refine ⟨settings, hwalk, hdis, hbud, hnp, ?_⟩
  intro Dsf hDsf hlen hpAPD hatom
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (hmain Dsf hDsf hlen hpAPD hatom).2
  intro p hp
  simp only [Set.mem_setOf_eq, not_forall, not_le] at hp ⊢
  obtain ⟨i, hi, hiσ, xs, hxsL, hres, hcard⟩ := hp
  refine ⟨i, hiσ, hi, fun htree => ?_⟩
  exact absurd (round_level3_count_le O settings.level3 _ _ _ xs p.1 hxsL (by rw [← hres]; exact htree))
    (not_le.2 hcard)

/-- The loop's total membership queries: summed over its at most `|Q| + 1` rounds, the oracle calls
of all three sources. With probability at least `1 - (|Q| + 1) · 5δ'/2`, for every round's realizing
probe sequence, the total

    ∑ᵢ [ level-3 calls + clustering reads + harvest sifts ]

is at most the same sum with each round's level-3 calls replaced by the level-3 polynomial and each
round's harvest draws replaced by the schedule bound `3(|Q| + 2)·2⌊RB⌋`. The clustering term
(`|prefixes|·(|suffixes| + 1)`) and the per-draw harvest cost
(`(3 + ⌈log₂ P.L⌉)(|Q| + 2)·|F_i|`, from the tracked population leaf count) are the loop's own
quantities; the draw count and the level-3 term reduce to the settings. -/
theorem learner_loop_total_queries {α : Type*} [Fintype α] [DecidableEq α] [Nonempty α] {σ : Type*}
    [Fintype σ] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    (M : _root_.DFA α σ) (O : Oracle μ (FreeMonoid α)) (P : PrefixDistribution α)
    (η₀ εcov ε δ' pAP : ℝ) (k : ℕ) :
    let Q : ℝ := Fintype.card σ
    let A : ℝ := Fintype.card α
    let ι : ℝ := min (εcov / 8) (ε / (4608 * Q * (Q + 1)))
    let s : ℝ := 1 / 2 - η₀
    let τ : ℝ := min ((εcov / 2) * ((1 - 2 * η₀) / (1 - η₀)) ^ 2) ι
    let J : ℝ := Q + 2
    let S : ℝ := Q * ((Q + 2) * A + 1) + 2 * A + 1
    let B₃ : ℝ := 36 * S / ε * Real.log (16 * S * (S + 1) / δ') + S + 1
    let Wm : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / δ') + 3 + (Q + 2) * A * (Q + 1)
    let W : ℝ := max (Real.exp 1 * Real.sqrt δ' * Wm / 2) (1 + (Q + 1) * δ' / (7 * ε))
    let PB : ℝ := (50 * max (Real.log (20 / τ)) (Real.log (8 * W / δ')) / s ^ 2 + 20 / s + 28) / pAP
      + 64 * Real.log ((J + 2) / (δ' / 2)) / pAP ^ 2
    let RB : ℝ := 161000 * J ^ 2 * Real.log ((J + 2) * PB / (δ' / 2 * (δ' / 4) * (pAP / 2) * τ))
      / (s ^ 4 * τ ^ 2)
    let cap : ℝ := min (min (min (τ * s / (2 ^ 19 * (J + 3) * (RB + 1)))
          (δ' * min (εcov / 2) ι ^ 2 / (2 ^ 35 * J * (3 + 2 * Real.logb 2 (RB + 1)))))
        (min (δ' * (ι / 2) ^ 2 / (2 ^ 5 * (J + 1) * (3 + 2 * Real.logb 2 (RB + 1))))
          (ι / 2 / (8 * J * (RB + 1)))))
      (min (min (δ' * pAP / (2 ^ 15 * (J + 1) * (PB + 1))) (ι / (12 * J ^ 2 * (RB + 1))))
        (min (pAP / (2 * J * (PB + 2))) (1 / (6 * J * (RB + 1)))))
    let Kc : ℝ := (1 + (Q + 1) ^ 3 * A) ^ Fintype.card σ * (Q + 3) ^ ((Fintype.card σ + 2)
      * Fintype.card α)
    (∀ w, w ∈ O.L ↔ M.eval w.toList ∈ M.accept) →
    O.η ≤ η₀ → η₀ < 1 / 2 →
    0 < εcov → εcov ≤ 1 → 0 < ε → ε ≤ 1 / 2 → 0 < δ' → δ' < 1 → 0 < pAP → 1 ≤ P.L → 1 ≤ k →
    (2 + Nat.clog 2 P.L) * ε < 3 / 8 →
    prefixMax P.D k ≤ cap * ε / 1536 →
    1536 * Real.log (16 * (Q + 1) * Kc / δ') * prefixMax P.D k ≤ ε →
    16 * B₃ ^ 2 * prefixMax P.D k ≤ δ' →
    ∃ settings : LoopSettings (Fintype.card σ + 1),
      settings.level3.walkStart = k ∧ settings.level3.disagreementRate = ε ∧
      (settings.level3.probeBudget : ℝ) ≤ B₃ ∧
      (∀ pops, ∑ B ∈ settings.schedule pops, B.npref ≤ 2 * ⌊RB⌋₊) ∧
      ∀ Dsf : Measure (FreeMonoid α), IsProbabilityMeasure Dsf →
        Dsf {v | v.toList.length ≠ P.L + Fintype.card σ + 3} = 0 →
        pAP ≤ Dsf.real {v | ∀ p, p * v ∈ O.L ↔ p ∈ O.L} →
        (∀ v, Dsf.real {v} ≤ cap) →
        (loopMeasure μ α (Fintype.card σ + 1) settings.level3.probeBudget).real
            {p | let rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p)
              ¬ ∀ xs : (i : Fin rounds.length) → Fin settings.level3.probeBudget → FreeMonoid α,
                  (∀ i, (∀ j, (xs i j).toList.length ≤ P.L) ∧
                    (rounds.get i).result = indecisiveLStarLevel3 settings.level3
                      (fun z => familyRead O.mq (rounds.get i).family (rounds.get i).state.lo
                        ((rounds.get i).state.hi + 1) z p.1) (xs i)) →
                  ∑ i : Fin rounds.length,
                      (level3Count O (rounds.get i).family (rounds.get i).state.lo
                          ((rounds.get i).state.hi + 1) settings.level3 (xs i) p.1
                        + (rounds.get i).prefs.card * ((rounds.get i).suffs.card + 1)
                        + 3 * (loopPopulations (Fintype.card σ + 1) i.val).card
                            * (∑ B ∈ settings.schedule (loopPopulations (Fintype.card σ + 1) i.val),
                                B.npref)
                            * ((3 + Nat.clog 2 P.L) * (Fintype.card σ + 2)
                                * (rounds.get i).family.card))
                    ≤ ∑ i : Fin rounds.length,
                      (settings.level3.probeBudget * ((3 + Nat.clog 2 P.L) * (Fintype.card σ + 2)
                            + (Fintype.card σ + 2) * Fintype.card α * (Fintype.card σ + 2))
                            * (rounds.get i).family.card
                        + (rounds.get i).prefs.card * ((rounds.get i).suffs.card + 1)
                        + 3 * (Fintype.card σ + 2) * (2 * ⌊RB⌋₊)
                            * ((3 + Nat.clog 2 P.L) * (Fintype.card σ + 2)
                                * (rounds.get i).family.card))}
          ≤ (Fintype.card σ + 1 : ℝ) * (5 * δ' / 2) := by
  intro Q A ι s τ J S B₃ Wm W PB RB cap Kc
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2 hk3
  obtain ⟨settings, hwalk, hdis, hbud, hnp, hmain⟩ := loop_simple_checks M O P η₀ εcov ε δ' pAP k
    hL hη hη₀ hεc0 hεcov1 hε0 hε12 hδ'0 hδ'1 hpAP hL1 hk0 hLε hk1 hk2 hk3
  refine ⟨settings, hwalk, hdis, hbud, hnp, ?_⟩
  intro Dsf hDsf hlen hpAPD hatom
  refine le_trans (measureReal_mono ?_ (measure_ne_top _ _)) (hmain Dsf hDsf hlen hpAPD hatom).2
  intro p hp
  set rounds := indecisiveLStarLoop settings O P.D Dsf (persistDraws p) with hrounds
  simp only [Set.mem_setOf_eq, not_forall, _root_.not_imp, not_le] at hp ⊢
  obtain ⟨xs, hreal, hsum⟩ := hp
  -- The clustering term is identical and the harvest draws reduce unconditionally, so a round where
  -- the summed bound fails is one whose level-3 term exceeds its polynomial.
  obtain ⟨i, -, hlt⟩ := Finset.exists_lt_of_sum_lt hsum
  have hharv : 3 * (loopPopulations (Fintype.card σ + 1) i.val).card
        * (∑ B ∈ settings.schedule (loopPopulations (Fintype.card σ + 1) i.val), B.npref)
      ≤ 3 * (Fintype.card σ + 2) * (2 * ⌊RB⌋₊) :=
    Nat.mul_le_mul (Nat.mul_le_mul (le_refl 3) ((Finset.card_le_univ _).trans_eq (by simp)))
      (hnp _)
  have hlevel : settings.level3.probeBudget * ((3 + Nat.clog 2 P.L) * (Fintype.card σ + 2)
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

/-- The per-round level-3 query bound, factored to show its shape:
`T · (|Q| + 2) · (3 + ⌈log₂ L⌉ + (|Q| + 2)|Σ|) · |F|`. -/
theorem level3_query_bound_factored (T A F L nQ : ℕ) :
    T * ((3 + Nat.clog 2 L) * (nQ + 2) + (nQ + 2) * A * (nQ + 2)) * F
      = T * (nQ + 2) * (3 + Nat.clog 2 L + (nQ + 2) * A) * F := by ring

open Asymptotics Filter in
/-- As the state count grows, a round's level-3 query bound is `O(|Q|²)` (the probe budget,
alphabet, family size and probe length held fixed). -/
theorem level3_query_isBigO (T A F L : ℕ) :
    (fun nQ : ℕ => (T * ((3 + Nat.clog 2 L) * (nQ + 2) + (nQ + 2) * A * (nQ + 2)) * F : ℝ))
      =O[atTop] (fun nQ : ℕ => ((nQ : ℝ)) ^ 2) := by
  refine Asymptotics.IsBigO.of_bound (T * F * (3 + Nat.clog 2 L + A) * 9) ?_
  filter_upwards [Filter.eventually_ge_atTop 1] with nQ hnQ
  have hnQ1 : (1 : ℝ) ≤ (nQ : ℝ) := by exact_mod_cast hnQ
  have hclog : (0 : ℝ) ≤ (Nat.clog 2 L : ℝ) := by positivity
  rw [Real.norm_of_nonneg (by positivity), Real.norm_of_nonneg (by positivity)]
  have hinner : (3 + (Nat.clog 2 L : ℝ)) * ((nQ : ℝ) + 2)
        + ((nQ : ℝ) + 2) * (A : ℝ) * ((nQ : ℝ) + 2)
      ≤ (3 + (Nat.clog 2 L : ℝ) + (A : ℝ)) * 9 * (nQ : ℝ) ^ 2 := by
    nlinarith [mul_nonneg hclog (by nlinarith [hnQ1] : (0:ℝ) ≤ 9 * (nQ:ℝ)^2 - ((nQ:ℝ) + 2)),
      mul_nonneg (by positivity : (0:ℝ) ≤ (A:ℝ))
        (by nlinarith [hnQ1] : (0:ℝ) ≤ 9 * (nQ:ℝ)^2 - ((nQ:ℝ) + 2) * ((nQ:ℝ) + 2)),
      hnQ1, hclog]
  calc ((T : ℝ) * ((3 + (Nat.clog 2 L : ℝ)) * ((nQ : ℝ) + 2)
          + ((nQ : ℝ) + 2) * (A : ℝ) * ((nQ : ℝ) + 2)) * (F : ℝ))
      = ((T : ℝ) * (F : ℝ)) * ((3 + (Nat.clog 2 L : ℝ)) * ((nQ : ℝ) + 2)
          + ((nQ : ℝ) + 2) * (A : ℝ) * ((nQ : ℝ) + 2)) := by ring
    _ ≤ ((T : ℝ) * (F : ℝ)) * ((3 + (Nat.clog 2 L : ℝ) + (A : ℝ)) * 9 * (nQ : ℝ) ^ 2) :=
        mul_le_mul_of_nonneg_left hinner (by positivity)
    _ = (T : ℝ) * (F : ℝ) * (3 + (Nat.clog 2 L : ℝ) + (A : ℝ)) * 9 * (nQ : ℝ) ^ 2 := by ring

end OrthoDFA
