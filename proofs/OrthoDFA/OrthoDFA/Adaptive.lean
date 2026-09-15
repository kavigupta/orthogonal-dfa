import OrthoDFA.Distributional

/-!
# The adaptive clustering loop: the theorem we actually want

`sample_suffix_family` is a retry loop: cluster, measure the FNR, and if it is too high
grow the pool and try again, stopping at a data-dependent time.  This file states the
guarantee **for that loop**, with no fixed budget:

> with probability `≥ 1 − δ` the loop **terminates**, and **whatever** family it returns
> preserves acceptance on `≥ 1 − εcov` of **each** prefix population.

It is proved by the two-part decomposition:

* `validity_of_returned` — whatever is returned is valid, *whenever* it is returned
  (uniformly over the round), except w.p. `δ/2`;
* `loop_terminates` — the loop returns at some round, except w.p. `δ/2`;
* `sound_and_terminating` (in `Distributional.lean`) composes them by a union bound.

The composition is proved.  The two halves are `sorry` for now: they are believed true
and are the remaining work — deliberately *not* adjustments to the statement.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S] [IsCancelMul S] [MeasurableMul S]
  [Countable S] [MeasurableSingletonClass S] [DecidableEq S]

section Loop

/-! ## The loop

A run carries, for each round `t`: the candidate suffix pool the loop has accumulated,
the prefixes it has drawn for each population, how many, its persistent noise sample,
whether the FNR test passed (so the loop returns), and the family it proposes.  `famAt`
is the clustering output — the least-loss `k`-subset of the pool against the round's
reads, summed over the populations. -/

-- `O`: the persistent oracle.  `populations`/`D`: the collection of prefix populations.
-- `Dsf`: the suffix distribution.  `k`: the family size (`suffix_family_size`).
-- `ν` on `Ξ`: the run.  `nz`: the run's persistent noise sample.
-- `pool t`: round `t`'s accumulated candidate suffixes.  `pre t j`: its prefixes for
-- population `j`, `mcount t` of them.  `ret t`: the loop returns at round `t`.
variable (O : Oracle μ S) {J : Type*} (populations : Finset J) (D : J → Measure S)
  (Dsf : Measure S) (k : ℕ)
  {Ξ : Type*} [MeasurableSpace Ξ] (ν : Measure Ξ)
  (nz : Ξ → Ω)
  (pool : ℕ → Ξ → Finset S)
  (pre : ℕ → J → ℕ → Ξ → S)
  (mcount : ℕ → ℕ)
  (ret : ℕ → Set Ξ)

/-- The family proposed at round `t`: the least-loss `k`-subset of the accumulated pool,
scored by the summed oracle reads over the round's prefixes across all populations. -/
noncomputable def famAt (x : Ξ) (t : ℕ) : Finset S :=
  leastLossSubset
    (fun v => ∑ j ∈ populations, ∑ i ∈ Finset.range (mcount t),
      O.read (fun i' : ℕ => pre t j i' x) v i (nz x))
    (pool t x) k

/-- Round `t`'s family is **invalid**: on some population it fails to preserve acceptance
on a `1 − εcov` fraction. -/
def FailAt (εcov : ℝ) (t : ℕ) : Set Ξ :=
  {x | ¬ ∀ j ∈ populations, 1 - εcov
        ≤ (D j).real {p | ∀ v ∈ famAt O populations k nz pool pre mcount x t,
            O.label (p * v) = O.label p}}

/-- **Part 1 — whatever is returned is valid, whenever it is returned.**

Uniformly over the round at which the loop stops: except with probability `δ/2`, no
round's family is invalid.  Hence under *any* stopping rule — the FNR test included —
the family the loop returns is valid, and the stopping rule never has to be modelled.

Proof plan (the pieces are in `Distributional.lean`): per round this is
`clustering_budget` at that round's budget — the selection avoids suffixes of high
distributional flip-mass (`ploss_good_upper`/`ploss_bad_lower`: two-level concentration
against the *persistent* oracle, plus the collision term), and `coverage_of_summed_flip`
turns per-suffix flip control into the per-population fraction.  The rounds are then
union-bounded at `δ/2^(t+2)`, which is summable, so no bound on the number of rounds is
needed. -/
theorem validity_of_returned (εcov δ : ℝ) :
    ν.real (⋃ t, FailAt O populations D k nz pool pre mcount εcov t) ≤ δ / 2 :=
  sorry

/-- **Part 2 — the loop terminates.**

Except with probability `δ/2`, some round's FNR test passes and the loop returns.

Proof plan: each round draws fresh suffixes, and by findability a drawn suffix is
accept-preserving with probability `≥ pAP`; once the pool holds enough accept-preserving
suffixes and enough prefixes, the family is decisive on all but `fnr_limit` of each
population, so the test passes.  The per-round trigger is block-local (it depends on the
round's fresh draws), so `geometric_miss_triggered` gives `(1 − p)^N` and `geom_le`
drives it under `δ/2`. -/
theorem loop_terminates (δ : ℝ) :
    ν.real {x | ∀ t, x ∉ ret t} ≤ δ / 2 :=
  sorry

/-- **The E-L\* clustering algorithm is PAC-correct.**

With probability `≥ 1 − δ` the adaptive loop **terminates**, and **whatever** family it
returns preserves acceptance on `≥ 1 − εcov` of **each** prefix population.

No fixed budget: the rounds range over all of `ℕ`, the pool and the prefix counts grow
with the round, and the stopping time is data-dependent and arbitrary. -/
theorem clustering_correct (εcov δ : ℝ) [IsProbabilityMeasure ν] :
    1 - δ ≤ ν.real
      {x | (∃ t, x ∈ ret t) ∧
        ∀ t, ∀ j ∈ populations, 1 - εcov
          ≤ (D j).real {p | ∀ v ∈ famAt O populations k nz pool pre mcount x t,
              O.label (p * v) = O.label p}} := by
  have h := sound_and_terminating ν (FailAt O populations D k nz pool pre mcount εcov) ret δ
    (validity_of_returned O populations D k ν nz pool pre mcount εcov δ)
    (loop_terminates ν ret δ)
  refine le_trans h (le_of_eq ?_)
  congr 1
  ext x
  simp only [Set.mem_setOf_eq, FailAt, not_not]

#print axioms validity_of_returned
#print axioms loop_terminates
#print axioms clustering_correct

end Loop

end OrthoDFA
