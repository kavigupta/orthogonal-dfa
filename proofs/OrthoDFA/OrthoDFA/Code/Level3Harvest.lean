import OrthoDFA.Code.Level3

/-! What level 3's harvest takes from a probe, and the law of what its sampler draws. -/

namespace OrthoDFA

namespace Random

open MeasureTheory

variable {α : Type*}

/-- The strings a probe of `x` walked from `k` under `H` reads undecided at the site, as the tests
count them. -/
noncomputable def siteHarvest (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α) :
    Level3Site → FreeMonoid α → List (FreeMonoid α)
  | .ends, x => endsString (probeOrAgree read H.tree H.edges k x)
  | .middles, x => middleString (probeOrAgree read H.tree H.edges k x)

/-- The law of what the harvest's sampler finds, each string as often as a probe drawn from `D`
and walked under `H` finds it at the site. -/
noncomputable def harvestLaw [Countable α] (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α))
    (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) :
    Measure (FreeMonoid α) :=
  D.bind fun x => ((siteHarvest read k H site x).map Measure.dirac).sum

/-- The law of what an `m`-probe sample finds at the site: each drawn probe, walked under `H`,
contributes the strings it leaves undecided there. Unlike `harvestLaw`, which integrates over all
of `D`, this reads the oracle only at the `m` sampled probes. -/
noncomputable def harvestSample [Countable α] (read : FreeMonoid α → ARU) (k : ℕ)
    (H : HypothesisDFA α) (site : Level3Site) {m : ℕ} (xs : Fin m → FreeMonoid α) :
    Measure (FreeMonoid α) :=
  ∑ i, ((siteHarvest read k H site (xs i)).map Measure.dirac).sum

/-- The law of what a list of probes finds at the site, each walked under `H`. Stored by a round
as a list (so its length need not sit in the type); `harvestSampleList_ofFn` identifies it with
`harvestSample` on the probes indexed by `Fin`. -/
noncomputable def harvestSampleList [Countable α] (read : FreeMonoid α → ARU) (k : ℕ)
    (H : HypothesisDFA α) (site : Level3Site) (l : List (FreeMonoid α)) :
    Measure (FreeMonoid α) :=
  (l.map fun x => ((siteHarvest read k H site x).map Measure.dirac).sum).sum

theorem harvestSampleList_ofFn [Countable α] (read : FreeMonoid α → ARU) (k : ℕ)
    (H : HypothesisDFA α) (site : Level3Site) {m : ℕ} (xs : Fin m → FreeMonoid α) :
    harvestSampleList read k H site (List.ofFn xs) = harvestSample read k H site xs := by
  unfold harvestSampleList harvestSample
  rw [List.map_ofFn, List.sum_ofFn]
  rfl

/-- `harvestLaw` is `harvestSample` on a single draw, averaged over that draw from `D`. -/
theorem harvestLaw_eq_bind_sample [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) :
    harvestLaw read D k H site
      = D.bind fun x => harvestSample read k H site (fun _ : Fin 1 => x) := by
  unfold harvestLaw harvestSample
  simp

end Random

end OrthoDFA
