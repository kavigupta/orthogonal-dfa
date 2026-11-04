import OrthoDFA.Code.Level3Harvest

/-! # The bounded-rejection harvest sampler

Level 3's harvest draws probes one at a time and stops at the first that leaves a string undecided
at the site, giving up after a cap `N`.  This is the operational counterpart of `harvestLaw`: its
output law (conditioned on a hit) is `popLaw (harvestLaw …)`, and it reads the oracle only along
the at most `N` probes it walks.  Defined here as a standalone object; the loop is not yet rewired
to use it. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

/-- The first of `xs`, walked from `k` under `H`, that harvests a string at the site — `none` if all
of them miss.  (`siteHarvest … x` has length ≤ 1, so `head?` is the harvested string.) -/
noncomputable def harvestFirst (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) (xs : List (FreeMonoid α)) : Option (FreeMonoid α) :=
  xs.findSome? fun x => (siteHarvest read k H site x).head?

/-- The (unnormalised) law of the bounded-rejection sampler on the event it hits: draw `N` probes
i.i.d. from `D`, keep the first harvested string.  `popLaw` of this is the sampler's output
distribution; `harvestMiss` carries the all-miss mass. -/
noncomputable def harvestCodeLaw [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) (N : ℕ) :
    Measure (FreeMonoid α) :=
  ((Measure.pi fun _ : Fin N => D).restrict
      {xs | (harvestFirst read k H site (List.ofFn xs)).isSome = true}).map
    fun xs => (harvestFirst read k H site (List.ofFn xs)).getD 1

/-- The all-miss mass of the bounded-rejection sampler: the chance none of the `N` probes harvests
at the site. -/
noncomputable def harvestMiss [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) (N : ℕ) :
    ℝ≥0∞ :=
  (Measure.pi fun _ : Fin N => D) {xs | harvestFirst read k H site (List.ofFn xs) = none}

/-- A probe cap that drives the all-miss probability below `δₛ` when the per-probe hit rate is at
least `y`:  `N = ⌈log (1/δₛ) / y⌉` gives `(1 - y) ^ N ≤ exp (-y N) ≤ δₛ`.  With
`y = harvestRate = ε/48` this is the `O((1/ε)·log(1/δₛ))` harvest probe budget. -/
noncomputable def harvestN (y δₛ : ℝ) : ℕ := ⌈Real.log (1 / δₛ) / y⌉₊

end Random

end OrthoDFA
