import RLStar.Code.Level3Harvest

/-! # The bounded-rejection harvest sampler

Level 3's harvest draws probes one at a time and stops at the first that leaves a string undecided
at the site, outside a set `T` it rejects, giving up after a cap `N`.  This is the operational
counterpart of `harvestLaw` off `T`: its output law, conditioned on a hit, is
`popLaw ((harvestLaw …).restrict Tᶜ)`, and it reads the oracle only along the at most `N` probes it
walks. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

open scoped Classical in
/-- The string harvested from the first of `xs`, walked from `k` under `H`, that harvests one at
the site outside `T`; `none` if none of them does.  (`siteHarvest … x` has length at most `1`.) -/
noncomputable def harvestFirst (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) (T : Set (FreeMonoid α)) (xs : List (FreeMonoid α)) :
    Option (FreeMonoid α) :=
  xs.findSome? fun x => (siteHarvest read k H site x).head?.filter fun z => z ∉ T

open scoped Classical in
/-- The probes in turn, up to the first that harvests. -/
noncomputable def harvestFirstQ (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) (xs : List (FreeMonoid α)) :
    Query (FreeMonoid α) ARU (Option (FreeMonoid α)) :=
  Query.findSome (fun x => (siteHarvestQ k H site x).map fun l => l.head?.filter fun z => z ∉ T) xs

/-- The (unnormalised) law of the sampler on the event it hits: draw `N` probes i.i.d. from `D`,
keep the first string harvested outside `T`. -/
noncomputable def harvestCodeLaw [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) (N : ℕ) : Measure (FreeMonoid α) :=
  ((Measure.pi fun _ : Fin N => D).restrict
      {xs | (harvestFirst read k H site T (List.ofFn xs)).isSome = true}).map
    fun xs => (harvestFirst read k H site T (List.ofFn xs)).getD 1

/-- The chance none of the `N` probes harvests a string outside `T`. -/
noncomputable def harvestMiss [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) (N : ℕ) : ℝ≥0∞ :=
  (Measure.pi fun _ : Fin N => D) {xs | harvestFirst read k H site T (List.ofFn xs) = none}

/-- The probe cap at which a per-probe hit rate of at least `y` misses with chance at most `δₛ`:
    (1 - y)^N ≤ exp(-yN) ≤ δₛ. -/
noncomputable def harvestN (y δₛ : ℝ) : ℕ := ⌈Real.log (1 / δₛ) / y⌉₊

end Random

end OrthoDFA
