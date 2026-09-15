import OrthoDFA.Liveness
import OrthoDFA.Complexity

/-!
# The single fully-fused clustering theorem

`clustering_pac` is the whole guarantee in one statement.  Its premises are only the
primitives: the oracle model, findability, the target, and the confidence + one
explicit sample-size threshold.  Everything else is derived in the body:

* the greedy is **defined** (`leastLossSubset`), so its subset/cardinality/least-loss
  properties are lemmas, not hypotheses;
* `good`/`bad` are **defined** from the oracle's `flip` (`good` = accept-preserving =
  flips nothing; `bad` = flips `≥ εcov`), so no separability hypotheses appear;
* the per-round good-pass and false-admit probabilities are derived (`greedy_picks_good`
  + `tail_le`, and `cert_sound`), with the soundness level set to `δ/(2·#cands)`.

Single-shot: confidence comes from the number of reads `m` (selection) and the
certification level, *not* from repeating rounds — the multi-round discovery loop is
the outer learner, out of scope.

Conclusion (PAC): with probability `≥ 1 − δ`, the greedy proposes a family that avoids
the bad set **and** no bad candidate clears certification.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

open scoped Classical in
/-- **The clustering algorithm is PAC-correct — one theorem, primitive premises.** -/
theorem clustering_pac
    {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {S : Type*} [MeasurableSpace S] [Mul S] [IsRightCancelMul S]
    -- ORACLE MODEL — one persistent RCN oracle, with signal (η < ½)
    (O : Oracle μ S) (hsig : O.η < 1 / 2)
    -- ALGORITHM — selection prefixes `pref` (m), certification prefixes `cpref` (n)
    (m n : ℕ) (hn : 0 < n)
    (pref cpref : ℕ → S) (hpref : Function.Injective pref) (hcpref : Function.Injective cpref)
    (cands : Finset S) (k : ℕ) (hcands : 0 < cands.card)
    -- TARGET — accuracy εcov
    (εcov : ℝ) (hεcov : 0 < εcov)
    -- FINDABILITY — ≥ k accept-preserving (flip-free) candidates
    (find : k ≤ (cands.filter (fun v => ∑ i ∈ Finset.range m, O.flip v (pref i) = 0)).card)
    -- CONFIDENCE + explicit sample complexity (in the number of selection reads m)
    (δ : ℝ) (hδ : 0 < δ) (hδ1 : δ ≤ 1)
    (hm : Real.log (2 * cands.card / δ) / (2 * ((1 / 2 - O.η) * εcov) ^ 2) ≤ (m : ℝ)) :
    μ.real
        ({ω | ¬ ∀ w ∈ leastLossSubset (fun v => ∑ i ∈ Finset.range m, O.read pref v i ω) cands k,
              ¬ ((m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, O.flip w (pref i))}
          ∪ {ω | ∃ v ∈ cands,
              ((n : ℝ) * εcov ≤ ∑ j ∈ Finset.range n, O.flip v (cpref j))
              ∧ (n : ℝ) * (((1 - O.η) - (1 - 2 * O.η) * εcov)
                    + certMargin n (δ / (2 * cands.card)))
                  ≤ ∑ j ∈ Finset.range n, (1 - O.read cpref v j ω)})
      ≤ δ := by
  have hk : k ≤ cands.card := find.trans (Finset.card_filter_le _ _)
  have ht : 0 < (1 / 2 - O.η) * εcov := mul_pos (by linarith) hεcov
  have hcardR : 0 < (cands.card : ℝ) := by exact_mod_cast hcands
  have hcard1 : (1 : ℝ) ≤ (cands.card : ℝ) := by exact_mod_cast hcands
  -- m ≥ 1, from the sample-size threshold being positive
  have h2cd : 1 < 2 * (cands.card : ℝ) / δ := by
    rw [lt_div_iff₀ hδ]; nlinarith [hcard1, hδ1]
  have hmpos : 0 < (m : ℝ) :=
    lt_of_lt_of_le (div_pos (Real.log_pos h2cd) (by positivity)) hm
  have hm1 : (1 : ℝ) ≤ (m : ℝ) := by
    have : 0 < m := by exact_mod_cast hmpos
    exact_mod_cast this
  -- LIVENESS: the greedy avoids the bad set, w.p. ≥ 1 - δ/2
  have hlive : μ.real
      {ω | ¬ ∀ w ∈ leastLossSubset (fun v => ∑ i ∈ Finset.range m, O.read pref v i ω) cands k,
          ¬ ((m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, O.flip w (pref i))} ≤ δ / 2 := by
    have hg := greedy_picks_good O pref hpref
      (fun v => ∑ i ∈ Finset.range m, O.flip v (pref i) = 0)
      (fun v => (m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, O.flip v (pref i))
      (fun v hb hg => by rw [hg] at hb; nlinarith [hb, hεcov, hm1])
      cands k m εcov
      (fun v _ hg => hg) (fun v _ hb => hb) hεcov.le find
      (fun ω => leastLossSubset (fun v => ∑ i ∈ Finset.range m, O.read pref v i ω) cands k)
      (fun ω => leastLossSubset_subset _ cands k hk)
      (fun ω => leastLossSubset_card _ cands k hk)
      (fun ω => leastLossSubset_least _ cands k hk)
    refine hg.trans ?_
    have hconv : Real.log ((cands.card : ℝ) / (δ / 2)) / (2 * ((1 / 2 - O.η) * εcov) ^ 2)
        ≤ (m : ℝ) := by
      rwa [show (cands.card : ℝ) / (δ / 2) = 2 * cands.card / δ by rw [div_div_eq_mul_div]; ring]
    exact tail_le ht hcardR (by linarith) hconv
  -- SOUNDNESS: no bad candidate clears certification, w.p. ≥ 1 - δ/2
  have hα0 : 0 < δ / (2 * cands.card) := by positivity
  have hα1 : δ / (2 * cands.card) ≤ 1 := by
    rw [div_le_one (by positivity)]; nlinarith [hcard1, hδ1]
  have hsound : μ.real
      {ω | ∃ v ∈ cands,
          ((n : ℝ) * εcov ≤ ∑ j ∈ Finset.range n, O.flip v (cpref j))
          ∧ (n : ℝ) * (((1 - O.η) - (1 - 2 * O.η) * εcov)
                + certMargin n (δ / (2 * cands.card)))
              ≤ ∑ j ∈ Finset.range n, (1 - O.read cpref v j ω)} ≤ δ / 2 := by
    refine (O.cert_sound cpref hcpref cands n εcov (δ / (2 * cands.card)) hn hα0 hα1 hεcov.le).trans
      (le_of_eq ?_)
    have hcne : (cands.card : ℝ) ≠ 0 := ne_of_gt hcardR
    field_simp
  calc μ.real (_ ∪ _)
      ≤ _ + _ := measureReal_union_le _ _
    _ ≤ δ / 2 + δ / 2 := add_le_add hlive hsound
    _ = δ := by ring

#print axioms clustering_pac

end OrthoDFA
