import OrthoDFA.Code.Level3HarvestCode

/-! # Drawing many harvest samples in the error monad

A round draws many harvest samples and feeds them to clustering.  Each sample is a bounded-rejection
`harvestFirst` over its own block of `N'` probes; the `m` samples are threaded through `Option`, so
the whole draw is `none` as soon as any one sample misses.  The round is then `drawMany >>=
clustering`, with clustering — the continuation — never seeing the draws or the miss.

`drawManyLaw` is the output law conditioned on all `m` samples hitting; `drawManyMiss` carries the
chance some sample misses.  Standalone: the loop is not yet rewired to use it. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*}

open scoped Classical in
/-- `m` bounded-rejection harvest draws, each over its own `N'`-probe block `yss i`, threaded
through `Option`: `none` as soon as one block misses, otherwise the `m` harvested strings. -/
noncomputable def drawManyFirst (read : FreeMonoid α → ARU) (k : ℕ) (H : HypothesisDFA α)
    (site : Level3Site) {m N' : ℕ} (yss : Fin m → (Fin N' → FreeMonoid α)) :
    Option (Fin m → FreeMonoid α) :=
  if ∀ i, (harvestFirst read k H site (List.ofFn (yss i))).isSome = true then
    some fun i => (harvestFirst read k H site (List.ofFn (yss i))).getD 1
  else none

/-- The law of `m` draws conditioned on all hitting: draw `m` blocks of `N'` i.i.d. probes, keep the
`m` harvested strings, and normalise to a probability (a point mass where the hit event has no or
infinite mass, matching `popLaw`).  This is the `m`-fold population clustering consumes. -/
noncomputable def drawManyLaw [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) (N' m : ℕ) :
    Measure (Fin m → FreeMonoid α) :=
  let raw : Measure (Fin m → FreeMonoid α) :=
    ((Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N' => D).restrict
        {yss | (drawManyFirst read k H site yss).isSome = true}).map
      fun yss => (drawManyFirst read k H site yss).getD 1
  if raw Set.univ = 0 ∨ raw Set.univ = ∞ then Measure.dirac 1 else (raw Set.univ)⁻¹ • raw

/-- The chance some one of the `m` draws misses entirely (all `N'` of its probes miss). -/
noncomputable def drawManyMiss [Countable α] (read : FreeMonoid α → ARU)
    (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α) (site : Level3Site) (N' m : ℕ) :
    ℝ≥0∞ :=
  (Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N' => D)
    {yss | drawManyFirst read k H site yss = none}

end Random

end OrthoDFA
