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

end Random

end OrthoDFA
