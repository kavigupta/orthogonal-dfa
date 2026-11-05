import OrthoDFA.Proofs.HarvestCode
import OrthoDFA.Proofs.LoopLaw

/-! # The operational harvest draw matches the sampler law

`harvestDraw` draws `N` probes from `D` by uniforms and runs the bounded-rejection sampler,
reading the oracle only along them.  Drawing the probes coordinatewise from `D` is measure
preserving (`map_drawFrom`), so the operational draw's law over its uniforms is exactly the
sampler's `Option` law:

* `harvestCodeLaw_eq_harvestDraw` — the hit-part equals `harvestCodeLaw`, so `popLaw` of the hits
  is the operational population (`popLaw_harvestDraw`);
* `harvestDraw_miss` — the all-miss mass equals `harvestMiss`;
* `harvestDraw_congr` — per run, the draw depends on the reader only through the bounded
  `harvestFirstReads` of the probes it drew.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α]
variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)
  (site : Random.Level3Site) (N : ℕ)

/-- The probes a harvest draw makes are the sampler's i.i.d. draws from `D`: pushing the uniforms
through the hit-restricted, output-mapped draw reproduces `harvestCodeLaw`. -/
theorem harvestCodeLaw_eq_harvestDraw [IsProbabilityMeasure D] :
    Random.harvestCodeLaw read D k H site N
      = ((Measure.pi fun _ : Fin N => (volume : Measure unitInterval)).restrict
            {u | (harvestDraw read D k H site N u).isSome = true}).map
          fun u => (harvestDraw read D k H site N u).getD 1 := by
  classical
  set g : (Fin N → unitInterval) → (Fin N → FreeMonoid α) := fun u i => drawFrom D (u i) with hg_def
  have hg : MeasurePreserving g (Measure.pi fun _ : Fin N => (volume : Measure unitInterval))
      (Measure.pi fun _ : Fin N => D) :=
    measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
  set f : (Fin N → FreeMonoid α) → FreeMonoid α :=
    fun xs => (Random.harvestFirst read k H site (List.ofFn xs)).getD 1 with hf_def
  set S : Set (Fin N → FreeMonoid α) :=
    {xs | (Random.harvestFirst read k H site (List.ofFn xs)).isSome = true} with hS_def
  have hSmeas : MeasurableSet S := (Set.to_countable _).measurableSet
  have hfmeas : Measurable f := measurable_of_countable _
  have hcomp : (fun u => (harvestDraw read D k H site N u).getD 1) = f ∘ g := by funext u; rfl
  have hset : {u | (harvestDraw read D k H site N u).isSome = true} = g ⁻¹' S := by ext u; rfl
  have key : ((Measure.pi fun _ : Fin N => (volume : Measure unitInterval)).restrict (g ⁻¹' S)).map
        (f ∘ g)
      = ((Measure.pi fun _ : Fin N => D).restrict S).map f := by
    rw [← Measure.map_map hfmeas hg.measurable, ← Measure.restrict_map hg.measurable hSmeas,
      hg.map_eq]
  rw [hcomp, hset]
  exact key.symm

/-- `popLaw` of the harvest draw's hits is the operational population. -/
theorem popLaw_harvestDraw [IsProbabilityMeasure D] :
    popLaw (((Measure.pi fun _ : Fin N => (volume : Measure unitInterval)).restrict
          {u | (harvestDraw read D k H site N u).isSome = true}).map
        fun u => (harvestDraw read D k H site N u).getD 1)
      = popLaw (Random.harvestCodeLaw read D k H site N) :=
  congrArg popLaw (harvestCodeLaw_eq_harvestDraw read D k H site N).symm

/-- The chance a harvest draw misses on all `N` probes is the sampler's `harvestMiss`. -/
theorem harvestDraw_miss [IsProbabilityMeasure D] :
    (Measure.pi fun _ : Fin N => (volume : Measure unitInterval))
        {u | harvestDraw read D k H site N u = none}
      = Random.harvestMiss read D k H site N := by
  classical
  set g : (Fin N → unitInterval) → (Fin N → FreeMonoid α) := fun u i => drawFrom D (u i) with hg_def
  have hg : MeasurePreserving g (Measure.pi fun _ : Fin N => (volume : Measure unitInterval))
      (Measure.pi fun _ : Fin N => D) :=
    measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
  set T : Set (Fin N → FreeMonoid α) :=
    {xs | Random.harvestFirst read k H site (List.ofFn xs) = none} with hT_def
  have hTmeas : MeasurableSet T := (Set.to_countable _).measurableSet
  have hset : {u | harvestDraw read D k H site N u = none} = g ⁻¹' T := by ext u; rfl
  rw [hset, ← Measure.map_apply hg.measurable hTmeas, hg.map_eq]
  rfl

/-- At the cap `N = harvestN y δₛ`, a harvest draw misses on all its probes with chance at most
`δₛ`: the round's per-draw failure mass, ready to thread as `δₛ'`. -/
theorem harvestDraw_miss_le [IsProbabilityMeasure D] {y δₛ : ℝ}
    (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₛ)
    (hyield : y ≤ (Random.harvestLaw read D k H site).real Set.univ) :
    (Measure.pi fun _ : Fin (Random.harvestN y δₛ) => (volume : Measure unitInterval))
        {u | harvestDraw read D k H site (Random.harvestN y δₛ) u = none}
      ≤ ENNReal.ofReal δₛ := by
  rw [harvestDraw_miss]
  exact Random.harvestCodeLaw_miss_le read D k H site hy0 hy1 hδ hyield

/-- Per-run locality of the draw: with the uniforms fixed, the drawn probe list is fixed (the
probes come from `D`, not the oracle), so `harvestDraw` depends on the reader only through the
bounded `harvestFirstReads` of that list. -/
theorem harvestDraw_congr {read read' : FreeMonoid α → ARU} (D : Measure (FreeMonoid α)) (k : ℕ)
    (H : HypothesisDFA α) (site : Random.Level3Site) (N : ℕ) (u : Fin N → unitInterval)
    (h : ∀ w ∈ Random.harvestFirstReads read H.tree H.edges k (List.ofFn fun i => drawFrom D (u i)),
        read w = read' w) :
    harvestDraw read D k H site N u = harvestDraw read' D k H site N u :=
  Random.harvestFirst_congr k H site _ h

/-- Conditioning a normalized law off `T` equals conditioning the raw measure off `T`: the
normalization scalar cancels under `popLaw`.  So the abstract harvested draw
`offLaw populationOp prefs` is `popLaw` of `harvestCodeLaw` restricted off the prefixes. -/
lemma offLaw_popLaw {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] (T : Finset (FreeMonoid α)) :
    offLaw (popLaw ν) T = popLaw (ν.restrict (↑T)ᶜ) := by
  classical
  unfold offLaw
  by_cases h0 : ν Set.univ = 0
  · have hν : ν = 0 := by rwa [← Measure.measure_univ_eq_zero]
    subst hν
    have hp0 : popLaw (0 : Measure (FreeMonoid α)) = Measure.dirac 1 := by
      unfold popLaw; rw [if_pos (Or.inl (by simp))]
    rw [Measure.restrict_zero, hp0, restrict_dirac' (measurableSet_of_countable _)]
    split_ifs with h1
    · unfold popLaw; simp
    · exact hp0
  · have htop : ν Set.univ ≠ ∞ := measure_ne_top ν Set.univ
    have hpop : popLaw ν = (ν Set.univ)⁻¹ • ν := by
      unfold popLaw; rw [if_neg (not_or.2 ⟨h0, htop⟩)]
    rw [hpop, Measure.restrict_smul]
    exact Random.popLaw_smul (ENNReal.inv_ne_zero.2 htop) (ENNReal.inv_ne_top.2 h0)

end OrthoDFA
