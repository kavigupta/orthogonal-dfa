import OrthoDFA.Termination
import OrthoDFA.Top

/-!
# The gate-driven algorithm over N rounds (joint measure, fully constructed)

Each round draws fresh strings, so the rounds are independent: the joint
randomness is the product measure `Measure.pi ν` over the `N` rounds, `ν r` being
round `r`'s own probability space (its suffix/prefix draws and their oracle reads).

Per round two coordinate-local events matter:
* `goodPass r` — the round proposes an accept-preserving family that passes the
  gate (probability ≥ `p`, by findability + the gate accepting a clean family);
* `badPass r` — the round's family passes the gate while being bad
  (probability ≤ `b`, by `certErr_bound`).

The failure event is "no round is a good pass, or some round is a bad pass".  Off
it, at least one good family passed and no bad family did, so the algorithm's
output (any passing family) is good.  Its probability is bounded by geometric
termination plus a union bound over rounds:

    P[failure] ≤ (1 - p)^N + N · b.

This is fully constructed — the product measure is built, `geometric_miss` gives
the `(1-p)^N`, and the coordinate marginal transfers the per-round `certErr`
bounds to the joint.  The per-round bounds `hgood`/`hbad` are exactly the outputs
of the discharged ingredient lemmas applied to `ν r`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {N : ℕ} {α : Fin N → Type*} [∀ r, MeasurableSpace (α r)]

/-- Coordinate marginal: a round-`r`-local event has the same probability under the
joint product as under round `r`'s own measure. -/
theorem pi_coord_real (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    (r : Fin N) (B : Set (α r)) (hB : MeasurableSet B) :
    (Measure.pi ν).real {x | x r ∈ B} = (ν r).real B := by
  rw [measureReal_def, measureReal_def,
    show {x : ∀ r, α r | x r ∈ B} = Function.eval r ⁻¹' B from rfl,
    (measurePreserving_eval ν r).measure_preimage hB.nullMeasurableSet]

/-- **The algorithm is correct.**  With the per-round bounds (findability `hgood`,
certification `hbad`), the failure event has probability at most `(1-p)^N + N·b`;
equivalently, with probability at least `1 - (1-p)^N - N·b` some round produced a
good family that passed the gate and no round produced a passing bad family, so the
output is accept-preserving with the target FNR/misclassification on every
population. -/
theorem algorithm_correct (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    (goodPass badPass : ∀ r, Set (α r))
    (hgoodmeas : ∀ r, MeasurableSet (goodPass r))
    (hbadmeas : ∀ r, MeasurableSet (badPass r))
    (p b : ℝ) (hp1 : p ≤ 1)
    (hgood : ∀ r, p ≤ (ν r).real (goodPass r))
    (hbad : ∀ r, (ν r).real (badPass r) ≤ b) :
    (Measure.pi ν).real
        ({x | ∀ r, x r ∉ goodPass r} ∪ {x | ∃ r, x r ∈ badPass r})
      ≤ (1 - p) ^ N + (N : ℝ) * b := by
  have hgeo : (Measure.pi ν).real {x | ∀ r, x r ∉ goodPass r} ≤ (1 - p) ^ N :=
    geometric_miss ν goodPass hgoodmeas p hp1 hgood
  have hb : (Measure.pi ν).real {x | ∃ r, x r ∈ badPass r} ≤ (N : ℝ) * b := by
    have hunion : {x : ∀ r, α r | ∃ r, x r ∈ badPass r}
        = ⋃ r ∈ (Finset.univ : Finset (Fin N)), {x | x r ∈ badPass r} := by
      ext x; simp
    rw [hunion]
    calc (Measure.pi ν).real (⋃ r ∈ (Finset.univ : Finset (Fin N)), {x | x r ∈ badPass r})
        ≤ ∑ r ∈ Finset.univ, (Measure.pi ν).real {x | x r ∈ badPass r} :=
          measureReal_biUnion_le _ _
      _ = ∑ r ∈ Finset.univ, (ν r).real (badPass r) :=
          Finset.sum_congr rfl (fun r _ => pi_coord_real ν r (badPass r) (hbadmeas r))
      _ ≤ ∑ _r ∈ (Finset.univ : Finset (Fin N)), b := Finset.sum_le_sum (fun r _ => hbad r)
      _ = (N : ℝ) * b := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc (Measure.pi ν).real
          ({x | ∀ r, x r ∉ goodPass r} ∪ {x | ∃ r, x r ∈ badPass r})
      ≤ (Measure.pi ν).real {x | ∀ r, x r ∉ goodPass r}
          + (Measure.pi ν).real {x | ∃ r, x r ∈ badPass r} := measureReal_union_le _ _
    _ ≤ (1 - p) ^ N + (N : ℝ) * b := add_le_add hgeo hb

#print axioms algorithm_correct

/-- **Capstone.**  The clustering algorithm over `N` rounds, with the certification
half fully discharged from `certErr_bound`.

Inputs are exactly the oracle model and findability:
* per round `r`, the certification reads `Xc r` of a drifted side — independent,
  `[0,1]`, with average mean at most `β+τ` (drifted past tolerance);
* findability: each round is a "good pass" (proposes an accept-preserving family
  that clears the gate) with probability at least `p`.

`badPass r` is the certification admit event; `hbad` is `certErr_bound`, so a bad
family is admitted with probability at most the level `a`.  Then the failure
probability over the joint product is at most `(1-p)^N + N·a`: with probability at
least `1 - (1-p)^N - N·a`, some round produced an accept-preserving family that
passed the gate and no round admitted a drifted one — the algorithm's output is
accept-preserving.  ("Good pass" bundles the gate-accept factor, which
`cleanAdmit_le`/`apLowFNR_le` bound; the remaining input is proposal findability.) -/
theorem clustering_algorithm_correct
    (ν : ∀ r, Measure (α r)) [∀ r, IsProbabilityMeasure (ν r)]
    (β : ℝ) (n : ℕ) (τ a : ℝ) (hn : 0 < n) (ha0 : 0 < a) (ha1 : a ≤ 1)
    (Xc : ∀ r, ℕ → α r → ℝ)
    (hcmeas : ∀ r i, AEMeasurable (Xc r i) (ν r))
    (hcindep : ∀ r, iIndepFun (Xc r) (ν r))
    (hcIcc : ∀ r i, ∀ᵐ ω ∂(ν r), Xc r i ω ∈ Set.Icc (0 : ℝ) 1)
    (hcmean : ∀ r, ∑ i ∈ Finset.range n, (ν r)[Xc r i] ≤ (n : ℝ) * (β + τ))
    (goodPass : ∀ r, Set (α r)) (hgoodmeas : ∀ r, MeasurableSet (goodPass r))
    (p : ℝ) (hp1 : p ≤ 1) (hgood : ∀ r, p ≤ (ν r).real (goodPass r))
    (hbadmeas : ∀ r, MeasurableSet
      {ω | (n : ℝ) * ((β + τ) + certMargin n a) ≤ ∑ i ∈ Finset.range n, Xc r i ω}) :
    (Measure.pi ν).real
        ({x | ∀ r, x r ∉ goodPass r}
          ∪ {x | ∃ r, x r ∈ {ω | (n : ℝ) * ((β + τ) + certMargin n a)
                ≤ ∑ i ∈ Finset.range n, Xc r i ω}})
      ≤ (1 - p) ^ N + (N : ℝ) * a :=
  algorithm_correct ν goodPass
    (fun r => {ω | (n : ℝ) * ((β + τ) + certMargin n a)
      ≤ ∑ i ∈ Finset.range n, Xc r i ω})
    hgoodmeas hbadmeas p a hp1 hgood
    (fun r => certErr_bound (Xc r) n β τ a (hcmeas r) (hcindep r) (hcIcc r)
      (hcmean r) hn ha0 ha1)

#print axioms clustering_algorithm_correct

end OrthoDFA
