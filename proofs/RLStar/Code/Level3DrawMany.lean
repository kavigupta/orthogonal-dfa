import RLStar.Code.Level3HarvestCode

/-! # Drawing many harvest samples in the error monad

`m` draws, each a bounded-rejection `harvestFirst` off `T` over its own block of `N'` probes,
threaded through `Option`: the whole draw is `none` as soon as any one misses.  The loop's round is
`drawMany >>= clustering`, the clustering seeing only the draws.

`drawManyLaw` is the output law conditioned on all `m` draws hitting; `drawManyMiss` is the chance
some draw misses. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

open scoped Classical in
/-- `m` draws, the `i`-th the string `harvestFirst` finds off `T` in the block `yss i`; `none` if
some block finds none. -/
noncomputable def drawManyFirst (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) (T : Set (FreeMonoid α)) {m N' : ℕ}
    (yss : Fin m → (Fin N' → FreeMonoid α)) : Option (Fin m → FreeMonoid α) :=
  if ∀ i, (harvestFirst read k H site T (List.ofFn (yss i))).isSome = true then
    some fun i => (harvestFirst read k H site T (List.ofFn (yss i))).getD 1
  else none

noncomputable def drawManyFirstQ (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) {m N' : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α)) :
    Query (FreeMonoid α) ARU (Option (Fin m → FreeMonoid α)) :=
  (Query.pi fun i => harvestFirstQ k H site T (List.ofFn (yss i))).map fun g =>
    if ∀ i, (g i).isSome = true then some fun i => (g i).getD 1 else none

/-- The law of the `m` draws over i.i.d. probes from `D`, conditioned on all hitting, as a
probability (a point mass where the hit event has no or infinite mass, as `popLaw`). -/
noncomputable def drawManyLaw [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) (N' m : ℕ) : Measure (Fin m → FreeMonoid α) :=
  let raw : Measure (Fin m → FreeMonoid α) :=
    ((Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N' => D).restrict
        {yss | (drawManyFirst read k H site T yss).isSome = true}).map
      fun yss => (drawManyFirst read k H site T yss).getD 1
  if raw Set.univ = 0 ∨ raw Set.univ = ∞ then Measure.dirac 1 else (raw Set.univ)⁻¹ • raw

/-- The chance some one of the `m` draws misses: all `N'` of its probes find nothing off `T`. -/
noncomputable def drawManyMiss [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site)
    (T : Set (FreeMonoid α)) (N' m : ℕ) : ℝ≥0∞ :=
  (Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N' => D)
    {yss | drawManyFirst read k H site T yss = none}

end Random

end OrthoDFA
