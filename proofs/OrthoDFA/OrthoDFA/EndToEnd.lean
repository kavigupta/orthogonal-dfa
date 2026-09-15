import OrthoDFA.PAC
import OrthoDFA.Liveness
import OrthoDFA.Discharge

/-!
# The single fully-fused end-to-end theorem

`clustering_end_to_end` is the whole guarantee in one statement, with **no**
abstract `trigger`/`himplies`/`p`/`b` premises: those are all discharged internally.
Its only inputs are

* the **oracle model** — per round `r` a persistent random-classification-noise
  `Oracle (ν r) S` (uniform noise level `η`), over a right-cancellative string type
  (`FreeMonoid`/`List` an instance);
* **findability** — each round the greedy proposes a least-loss `k`-subset of at
  least `k` good candidates (`goodCount`), the pool bounded by `C`;
* the **target** — accuracy `εcov` (a `bad` suffix flips `≥ εcov` of the prefix
  mass, a `good` one flips nothing) and the gate's certification of a drifted side
  (`Xc`, average mean `≤ β+τ`).

Everything else is derived:

* the per-round good-pass probability `p = 1 − C·exp(-2m((½−η)εcov)²)` comes from
  `sepFail_prob` (liveness, via `denoised_loss` ∘ `read_disagreement_mean`);
* the block-local trigger is the reads' separation event `(sepFail)ᶜ`
  (`sepFail_measurable`), and it forces a good round (`not_bad_of_not_mem_sepFail`);
* the per-round false-admit `≤ a` comes from `certErr_bound`;
* the run-level failure is driven below `δ` by the explicit sample sizes
  (`clustering_pac` ∘ `geom_le`).

Conclusion (PAC): with probability `≥ 1 − δ` some round produces an all-good
(accept-preserving, avoids-`bad`) family and no round admits a drifted one.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

/-- **The clustering algorithm is PAC-correct — one theorem, clean premises.** -/
theorem clustering_end_to_end
    {N : ℕ} {α : Fin N → Type*} [∀ r, MeasurableSpace (α r)]
    {S : Type*} [DecidableEq S] [Mul S] [IsRightCancelMul S]
    -- the oracle model: per-round persistent RCN oracles, uniform noise level η
    (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    (O : ∀ r, Oracle (ν r) S) (η : ℝ) (hη : ∀ r, (O r).η = η)
    -- the target language: `good`/`bad` suffixes, disjoint
    (good bad : S → Prop) [DecidablePred good] [DecidablePred bad]
    (hdisj : ∀ v, bad v → ¬ good v)
    -- findability + the greedy's selection spec, per round
    (pref : Fin N → ℕ → S) (hpref : ∀ r, Function.Injective (pref r))
    (cands : Fin N → Finset S) (k : Fin N → ℕ) (m : ℕ) (εcov : ℝ) (hεcov0 : 0 ≤ εcov)
    (C : ℕ) (hC : ∀ r, (cands r).card ≤ C)
    (hgoodflip : ∀ r, ∀ v ∈ cands r, good v →
        ∑ i ∈ Finset.range m, (O r).flip v (pref r i) = 0)
    (hbadflip : ∀ r, ∀ v ∈ cands r, bad v →
        (m : ℝ) * εcov ≤ ∑ i ∈ Finset.range m, (O r).flip v (pref r i))
    (goodCount : ∀ r, k r ≤ ((cands r).filter good).card)
    (chosen : ∀ r, α r → Finset S)
    (hsub : ∀ r ω, chosen r ω ⊆ cands r) (hcard : ∀ r ω, (chosen r ω).card = k r)
    (hleast : ∀ r ω, ∀ v ∈ chosen r ω, ∀ w ∈ cands r, w ∉ chosen r ω →
        (∑ i ∈ Finset.range m, (O r).read (pref r) v i ω)
          ≤ ∑ i ∈ Finset.range m, (O r).read (pref r) w i ω)
    -- soundness: the gate certifies a drifted side (oracle reads, average mean ≤ β+τ)
    (β : ℝ) (n : ℕ) (τ a : ℝ) (hn : 0 < n) (ha0 : 0 < a) (ha1 : a ≤ 1)
    (Xc : ∀ r, ℕ → α r → ℝ) (hXmeas : ∀ r i, Measurable (Xc r i))
    (hXindep : ∀ r, iIndepFun (Xc r) (ν r))
    (hXicc : ∀ r i, ∀ᵐ ω ∂(ν r), Xc r i ω ∈ Set.Icc (0 : ℝ) 1)
    (hXmean : ∀ r, ∑ i ∈ Finset.range n, (ν r)[Xc r i] ≤ (n : ℝ) * (β + τ))
    -- PAC target + explicit sample sizes
    (δ : ℝ) (hδ : 0 < δ)
    (hlive : 0 < 1 - (C : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2))
    (hN : Real.log (2 / δ) / (1 - (C : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2))
        ≤ (N : ℝ))
    (hNb : (N : ℝ) * a ≤ δ / 2) :
    (Measure.pi ν).real
        ({x | ∀ r, ¬ ∀ w ∈ chosen r (x r), ¬ bad w}
          ∪ (⋃ r, {x | (n : ℝ) * ((β + τ) + certMargin n a)
              ≤ ∑ i ∈ Finset.range n, Xc r i (x r)}))
      ≤ δ := by
  set p : ℝ := 1 - (C : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2) with hp_def
  have hEnn : (0 : ℝ) ≤ (C : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2) :=
    mul_nonneg (Nat.cast_nonneg C) (Real.exp_pos _).le
  refine clustering_pac ν
    (fun r => (O r).sepFail (pref r) good bad (cands r) m εcov)ᶜ ?_ -- htrig
    (fun r => {x | ¬ ∀ w ∈ chosen r (x r), ¬ bad w})
    (fun r => {x | x r ∈ {ω | (n : ℝ) * ((β + τ) + certMargin n a)
        ≤ ∑ i ∈ Finset.range n, Xc r i ω}})
    ?_ -- himplies
    p a δ hlive (by linarith [hEnn]) ?_ ?_ hδ hN hNb
  · -- htrig: the trigger (complement of sepFail) is measurable
    exact fun r => ((O r).sepFail_measurable (pref r) good bad (cands r) m εcov).compl
  · -- himplies: off sepFail the greedy avoids bad
    exact fun r x hx hcontra => hcontra ((O r).not_bad_of_not_mem_sepFail (pref r) good bad
      hdisj (cands r) (k r) m εcov (goodCount r) (chosen r (x r)) (hsub r (x r)) (hcard r (x r))
      (hleast r (x r)) hx)
  · -- hp: the trigger fires with probability ≥ p
    intro r
    have hsfle : (ν r).real ((O r).sepFail (pref r) good bad (cands r) m εcov)
        ≤ (C : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2) := by
      calc (ν r).real ((O r).sepFail (pref r) good bad (cands r) m εcov)
          ≤ ((cands r).card : ℝ)
              * Real.exp (-2 * (m : ℝ) * ((1 / 2 - (O r).η) * εcov) ^ 2) :=
            (O r).sepFail_prob (pref r) (hpref r) good bad hdisj (cands r) m εcov hεcov0
              (hgoodflip r) (hbadflip r)
        _ = ((cands r).card : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2) := by
            rw [hη r]
        _ ≤ (C : ℝ) * Real.exp (-2 * (m : ℝ) * ((1 / 2 - η) * εcov) ^ 2) :=
            mul_le_mul_of_nonneg_right (by exact_mod_cast hC r) (Real.exp_pos _).le
    have huniv : (ν r).real Set.univ = 1 := by simp [measureReal_def, measure_univ]
    have hcompl : (ν r).real ((O r).sepFail (pref r) good bad (cands r) m εcov)ᶜ
        = 1 - (ν r).real ((O r).sepFail (pref r) good bad (cands r) m εcov) := by
      rw [measureReal_compl ((O r).sepFail_measurable (pref r) good bad (cands r) m εcov), huniv]
    show p ≤ (ν r).real ((O r).sepFail (pref r) good bad (cands r) m εcov)ᶜ
    rw [hcompl, hp_def]; linarith [hsfle]
  · -- hbad: a drifted side is admitted with probability ≤ a
    intro r
    have hcm : MeasurableSet {ω : α r | (n : ℝ) * ((β + τ) + certMargin n a)
        ≤ ∑ i ∈ Finset.range n, Xc r i ω} :=
      measurableSet_le measurable_const (Finset.measurable_sum _ (fun i _ => hXmeas r i))
    rw [pi_coord_real ν r _ hcm]
    exact certErr_bound (Xc r) n β τ a (fun i => (hXmeas r i).aemeasurable) (hXindep r)
      (hXicc r) (hXmean r) hn ha0 ha1

#print axioms clustering_end_to_end

end OrthoDFA
