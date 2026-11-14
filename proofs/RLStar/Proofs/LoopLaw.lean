import RLStar.Proofs.LoopAux
import RLStar.Proofs.RawRun
import RLStar.Proofs.DrawFrom
import RLStar.Proofs.DrawMany
import RLStar.Proofs.LoopRuns
import Mathlib.Probability.ProductMeasure

/-!
# Coupling a round's draws with i.i.d. draws

A harvested population's draw reads a uniform and a block of `N`. `fillDraw` is the draw with a
miss filled by the uniform's draw from the population off the earlier prefixes: a hit's law is a
multiple of the population off them, so the filled draw is drawn from it
(`measurePreserving_fillDraw`). `idealDraws` fills every draw and lets the named populations
certify on draws of their own; it is drawn by `runMeasure` at the populations off the earlier
prefixes (`measurePreserving_idealDraws`). Its first `M` draws of each population are the round's
tables (`padList_popTable`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## One draw -/

omit [DecidableEq α] in
/-- A probability that is a multiple of `ν` plus a multiple of its normalisation is its
normalisation. -/
lemma eq_popLaw_of_combo {ν ρ : Measure (FreeMonoid α)} [IsFiniteMeasure ν]
    [IsProbabilityMeasure ρ] {c m : ℝ≥0∞} (h : ρ = c • ν + m • popLaw ν) : ρ = popLaw ν := by
  by_cases h0 : ν Set.univ = 0
  · have hν : ν = 0 := Measure.measure_univ_eq_zero.1 h0
    have hp0 : popLaw (0 : Measure (FreeMonoid α)) = Measure.dirac 1 := by
      unfold popLaw; simp
    rw [hν, smul_zero, zero_add, hp0] at h
    rw [hν, hp0]
    have h1 := congrArg (fun ρ' : Measure (FreeMonoid α) => ρ' Set.univ) h
    simp only [measure_univ, Measure.smul_apply, smul_eq_mul, mul_one] at h1
    rw [h, ← h1, one_smul]
  · have htop : ν Set.univ ≠ ∞ := measure_ne_top _ _
    have hp : popLaw ν = (ν Set.univ)⁻¹ • ν := by
      unfold popLaw; rw [if_neg (not_or.2 ⟨h0, htop⟩)]
    rw [hp, smul_smul, ← add_smul] at h
    have h1 := congrArg (fun ρ' : Measure (FreeMonoid α) => ρ' Set.univ) h
    simp only [measure_univ, Measure.smul_apply, smul_eq_mul] at h1
    rw [hp, h, ENNReal.eq_inv_of_mul_eq_one_left h1.symm]

omit [DecidableEq α] in
lemma offLaw_dirac_one (T : Finset (FreeMonoid α)) :
    offLaw (Measure.dirac (1 : FreeMonoid α)) T = Measure.dirac 1 := by
  classical
  unfold offLaw popLaw
  rw [restrict_dirac' ((Set.to_countable _).measurableSet)]
  split_ifs with h1 h2 <;> simp_all

omit [DecidableEq α] in
lemma drawFrom_dirac_one (u : unitInterval) :
    drawFrom (Measure.dirac (1 : FreeMonoid α)) u = 1 := by
  set z := drawFrom (Measure.dirac (1 : FreeMonoid α)) u
  rcases (drawFrom_eq_iff _ u z).1 rfl with ⟨h1, h2⟩ | ⟨h, -⟩
  · by_contra hne
    rw [massBelow_succ] at h2
    have hs : {z' : FreeMonoid α | strCode z' = strCode z} = {z} := by
      ext z'; simp [strCode_injective.eq_iff]
    have h0 : (Measure.dirac (1 : FreeMonoid α)).real {z} = 0 := by
      rw [measureReal_def, Measure.dirac_apply' _ (measurableSet_singleton _),
        Set.indicator_of_notMem (by simpa [eq_comm] using hne)]
      rfl
    rw [hs, h0, add_zero] at h2
    linarith
  · exact h

omit [DecidableEq α] in
/-- Conditioning a normalised law off `T` conditions the measure off `T`. -/
lemma offLaw_popLaw {ν : Measure (FreeMonoid α)} [IsFiniteMeasure ν] (T : Finset (FreeMonoid α)) :
    offLaw (popLaw ν) T = popLaw (ν.restrict (↑T)ᶜ) := by
  classical
  unfold offLaw
  by_cases h0 : ν Set.univ = 0
  · have hν : ν = 0 := by rwa [← Measure.measure_univ_eq_zero]
    subst hν
    have hp0 : popLaw (0 : Measure (FreeMonoid α)) = Measure.dirac 1 := by
      unfold popLaw; rw [if_pos (Or.inl (by simp))]
    rw [Measure.restrict_zero, hp0, restrict_dirac' ((Set.to_countable _).measurableSet)]
    split_ifs with h1
    · unfold popLaw; simp
    · exact hp0
  · have htop : ν Set.univ ≠ ∞ := measure_ne_top ν Set.univ
    have hpop : popLaw ν = (ν Set.univ)⁻¹ • ν := by
      unfold popLaw; rw [if_neg (not_or.2 ⟨h0, htop⟩)]
    rw [hpop, Measure.restrict_smul]
    exact Random.popLaw_smul (ENNReal.inv_ne_zero.2 htop) (ENNReal.inv_ne_top.2 h0)

instance (N : ℕ) : IsProbabilityMeasure (drawLaw N) := by unfold drawLaw; infer_instance

omit [DecidableEq α] in
lemma measurable_getD {β : Type*} [MeasurableSpace β] {f : β → Option (FreeMonoid α)}
    {d : β → FreeMonoid α} (hS : MeasurableSet {b | (f b).isSome = true})
    (h1 : Measurable fun b => (f b).getD 1) (h2 : Measurable d) :
    Measurable fun b => (f b).getD (d b) := by
  classical
  have : (fun b => (f b).getD (d b))
      = {b | (f b).isSome = true}.piecewise (fun b => (f b).getD 1) d := by
    funext b
    by_cases hb : (f b).isSome = true
    · rw [Set.piecewise_eq_of_mem (s := {b | (f b).isSome = true}) _ _ hb]
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hb
      rw [hv]; rfl
    · rw [Set.piecewise_eq_of_notMem (s := {b | (f b).isSome = true}) _ _ hb]
      rw [Option.not_isSome_iff_eq_none.1 hb]; rfl
  rw [this]
  exact h1.piecewise hS h2

section Sampler

variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (Kw : ℕ)
  (X : Finset (FreeMonoid α)) (H : HypothesisDFA (Option α)) (s : Random.Level3Site)

instance [IsProbabilityMeasure D] : IsProbabilityMeasure (D.map (Random.padIn Kw X)) :=
  Measure.isProbabilityMeasure_map (measurable_of_countable _).aemeasurable

lemma lazyHarvestLaw_univ_ne_top [IsProbabilityMeasure D] :
    Random.lazyHarvestLaw read D Kw X H s Set.univ ≠ ⊤ := by
  unfold Random.lazyHarvestLaw
  rw [Measure.map_apply (measurable_of_countable _) MeasurableSet.univ, Set.preimage_univ]
  exact Random.harvestLaw_univ_ne_top _ _ _ _ _

instance [IsProbabilityMeasure D] : IsFiniteMeasure (Random.lazyHarvestLaw read D Kw X H s) :=
  ⟨lt_top_iff_ne_top.2 (lazyHarvestLaw_univ_ne_top read D Kw X H s)⟩

/-- The sampler over `N` probes the block `u.2` draws from `D`, its miss filled by the uniform
`u.1`'s draw from `τ`. -/
def fillSampler {N : ℕ} (T : Finset (FreeMonoid α)) (τ : Measure (FreeMonoid α))
    (u : unitInterval × (Fin N → unitInterval)) : FreeMonoid α :=
  (Random.lazyHarvestFirst read Kw X H s T (List.ofFn fun t => drawFrom D (u.2 t))).getD
    (drawFrom τ u.1)

omit [Fintype α] [DecidableEq α] in
lemma lazyHarvestFirst_ofFn (T : Finset (FreeMonoid α)) {N : ℕ} (xs : Fin N → FreeMonoid α) :
    Random.lazyHarvestFirst read Kw X H s T (List.ofFn xs)
      = (Random.harvestFirst (Random.padRead read) Kw H s (Random.unpad ⁻¹' ↑T)
          (List.ofFn fun t => Random.padIn Kw X (xs t))).map Random.unpad := by
  simp only [Random.lazyHarvestFirst, List.map_ofFn]
  rfl

omit [DecidableEq α] in
lemma measurable_fillSampler [IsProbabilityMeasure D] (N : ℕ) (T : Finset (FreeMonoid α))
    (τ : Measure (FreeMonoid α)) [IsFiniteMeasure τ] :
    Measurable (fillSampler read D Kw X H s (N := N) T τ) := by
  have hpr : Measurable fun u : unitInterval × (Fin N → unitInterval) =>
      fun t => drawFrom D (u.2 t) :=
    measurable_pi_lambda _ fun t => (measurable_drawFrom D).comp
      ((measurable_pi_apply t).comp measurable_snd)
  exact measurable_getD (hpr ((Set.to_countable {xs : Fin N → FreeMonoid α |
      (Random.lazyHarvestFirst read Kw X H s T (List.ofFn xs)).isSome = true}).measurableSet))
    ((measurable_of_countable (fun xs : Fin N → FreeMonoid α =>
      (Random.lazyHarvestFirst read Kw X H s T (List.ofFn xs)).getD 1)).comp hpr)
    ((measurable_drawFrom τ).comp measurable_fst)

/-- A harvested draw with its miss filled is drawn from the harvested population off `T`. -/
theorem map_fillSampler [IsProbabilityMeasure D] (N : ℕ) (T : Finset (FreeMonoid α)) :
    (drawLaw N).map (fillSampler read D Kw X H s T
        (offLaw (popLaw (Random.lazyHarvestLaw read D Kw X H s)) T))
      = offLaw (popLaw (Random.lazyHarvestLaw read D Kw X H s)) T := by
  classical
  set D' := D.map (Random.padIn Kw X)
  set T' : Set (FreeMonoid (Option α)) := Random.unpad ⁻¹' ↑T
  set rd := Random.padRead read
  set ν := Random.lazyHarvestLaw read D Kw X H s
  set ν' := ν.restrict (↑T)ᶜ
  set τ := offLaw (popLaw ν) T
  have hτ : τ = popLaw ν' := offLaw_popLaw T
  set P := Measure.pi fun _ : Fin N => (volume : Measure unitInterval)
  set pr : (Fin N → unitInterval) → (Fin N → FreeMonoid (Option α)) :=
    fun b t => Random.padIn Kw X (drawFrom D (b t))
  have hone : MeasurePreserving (fun v : unitInterval => Random.padIn Kw X (drawFrom D v))
      volume D' :=
    ⟨(measurable_of_countable (Random.padIn Kw X)).comp (measurable_drawFrom D), by
      rw [show (fun v => Random.padIn Kw X (drawFrom D v)) = Random.padIn Kw X ∘ drawFrom D
        from rfl, ← Measure.map_map (measurable_of_countable (Random.padIn Kw X))
          (measurable_drawFrom D), map_drawFrom]⟩
  have hpr : MeasurePreserving pr P (Measure.pi fun _ : Fin N => D') :=
    measurePreserving_pi _ _ fun _ => hone
  set hf : (Fin N → FreeMonoid (Option α)) → Option (FreeMonoid (Option α)) :=
    fun xs => Random.harvestFirst rd Kw H s T' (List.ofFn xs)
  set g : (Fin N → FreeMonoid (Option α)) → Option (FreeMonoid α) :=
    fun xs => (hf xs).map Random.unpad
  set fill : unitInterval × (Fin N → unitInterval) → FreeMonoid α :=
    fun u => (g (pr u.2)).getD (drawFrom τ u.1)
  have hfun : fillSampler read D Kw X H s T τ = fill := by
    funext u
    simp only [fillSampler, fill, g, hf, pr, lazyHarvestFirst_ofFn]
    rfl
  have hfill : Measurable fill := hfun ▸ measurable_fillSampler read D Kw X H s N T τ
  rw [hfun]
  have hdec : (drawLaw N).map fill
      = (Random.harvestCodeLaw rd D' Kw H s T' N).map Random.unpad
        + Random.harvestMiss rd D' Kw H s T' N • τ := by
    ext A hA
    set S1 : Set (Fin N → FreeMonoid (Option α)) :=
      (fun z => (g z).getD 1) ⁻¹' A ∩ {z | (g z).isSome = true}
    set S2 : Set (Fin N → FreeMonoid (Option α)) := {z | g z = none}
    have hS1 : MeasurableSet S1 := (Set.to_countable _).measurableSet
    have hS2 : MeasurableSet S2 := (Set.to_countable _).measurableSet
    have hset : fill ⁻¹' A = (Set.univ ×ˢ (pr ⁻¹' S1)) ∪ ((drawFrom τ ⁻¹' A) ×ˢ (pr ⁻¹' S2)) := by
      ext ⟨u1, b⟩
      simp only [Set.mem_preimage, Set.mem_union, Set.mem_prod, Set.mem_univ, true_and, S1, S2,
        Set.mem_inter_iff, Set.mem_setOf_eq, fill]
      rcases g (pr b) with _ | v <;> simp
    have hdisj : Disjoint (Set.univ ×ˢ (pr ⁻¹' S1)) ((drawFrom τ ⁻¹' A) ×ˢ (pr ⁻¹' S2)) := by
      rw [Set.disjoint_left]
      rintro ⟨u1, b⟩ ⟨-, h1⟩ ⟨-, h2⟩
      simp only [Set.mem_preimage, S1, S2, Set.mem_inter_iff, Set.mem_setOf_eq] at h1 h2
      rw [h2] at h1
      exact absurd h1.2 (by simp)
    have hS1' : S1 = (fun z => (hf z).getD 1) ⁻¹' (Random.unpad ⁻¹' A)
        ∩ {z | (hf z).isSome = true} := by
      ext z
      simp only [S1, g, Set.mem_inter_iff, Set.mem_preimage, Set.mem_setOf_eq]
      rcases hf z with _ | v <;> simp
    have hS2' : S2 = {z | hf z = none} := by
      ext z
      simp only [S2, g, Set.mem_setOf_eq, Option.map_eq_none_iff]
    rw [Measure.map_apply hfill hA, hset,
      measure_union hdisj ((measurable_drawFrom τ hA).prod (hpr.measurable hS2)), drawLaw,
      Measure.prod_prod, Measure.prod_prod, measure_univ, one_mul,
      ← Measure.map_apply hpr.measurable hS1, ← Measure.map_apply hpr.measurable hS2, hpr.map_eq,
      ← Measure.map_apply (measurable_drawFrom τ) hA, map_drawFrom, Measure.add_apply,
      Measure.smul_apply, smul_eq_mul,
      Measure.map_apply (measurable_of_countable _) hA, Random.harvestCodeLaw_apply', mul_comm,
      hS1', hS2']
    rfl
  obtain ⟨c, hc⟩ : ∃ c : ℝ≥0∞, (Random.harvestCodeLaw rd D' Kw H s T' N).map Random.unpad
      = c • ν' := by
    rcases Nat.eq_zero_or_pos N with rfl | hN
    · refine ⟨0, ?_⟩
      rw [zero_smul]
      unfold Random.harvestCodeLaw
      have : {xs : Fin 0 → FreeMonoid (Option α) |
          (Random.harvestFirst rd Kw H s T' (List.ofFn xs)).isSome = true} = ∅ := by
        ext xs; simp [Random.harvestFirst]
      rw [this, Measure.restrict_empty, Measure.map_zero, Measure.map_zero]
    · obtain ⟨c, -, -, h⟩ := Random.harvestCodeLaw_smul rd D' Kw H s T' N hN
      refine ⟨c, ?_⟩
      rw [h, Measure.map_smul, show T'ᶜ = Random.unpad ⁻¹' ((↑T : Set (FreeMonoid α))ᶜ) from rfl,
        ← Measure.restrict_map (measurable_of_countable _) (Set.to_countable _).measurableSet]
      rfl
  haveI : IsProbabilityMeasure ((drawLaw N).map fill) :=
    Measure.isProbabilityMeasure_map hfill.aemeasurable
  rw [hc, hτ] at hdec
  rw [hτ]
  exact eq_popLaw_of_combo hdec

end Sampler

/-- A filled draw is drawn from its population off `T`. -/
theorem measurePreserving_fillDraw {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (Kw N : ℕ) (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K)) :
    MeasurePreserving (fillDraw D Kw N pops T j) (drawLaw N)
      (offLaw (loopDistributions D Kw pops j) T) := by
  classical
  have hfst : MeasurePreserving (Prod.fst : unitInterval × (Fin N → unitInterval) → unitInterval)
      (drawLaw N) volume := measurePreserving_fst
  have hmiss : (∀ u, popHit D Kw N pops T j u = none) →
      MeasurePreserving (fillDraw D Kw N pops T j) (drawLaw N)
        (offLaw (loopDistributions D Kw pops j) T) := fun h0 => by
    have hf : fillDraw D Kw N pops T j
        = fun u => drawFrom (offLaw (loopDistributions D Kw pops j) T) u.1 := by
      funext u; simp only [fillDraw, h0 u, Option.getD_none]
    rw [hf]
    exact (⟨measurable_drawFrom _, map_drawFrom _⟩ : MeasurePreserving
      (drawFrom (offLaw (loopDistributions D Kw pops j) T)) volume _).comp hfst
  rcases j with _ | i
  · exact hmiss fun u => rfl
  rcases hs : pops[i.val]? with _ | s
  · exact hmiss fun u => by simp only [popHit, hs]
  · have hf : fillDraw D Kw N pops T (some i) = fillSampler s.read D Kw s.trie s.hyp s.site T
        (offLaw (popLaw (Random.lazyHarvestLaw s.read D Kw s.trie s.hyp s.site)) T) := by
      funext u; simp only [fillDraw, popHit, hs, loopDistributions, fillSampler]
    have hl : loopDistributions D Kw pops (some i)
        = popLaw (Random.lazyHarvestLaw s.read D Kw s.trie s.hyp s.site) := by
      simp only [loopDistributions, hs]
    rw [hf, hl]
    exact ⟨measurable_fillSampler _ D Kw _ _ _ N T _, map_fillSampler _ D Kw _ _ _ N T⟩

omit [DecidableEq α] in
/-- The tables' draws are the filled draws. -/
lemma padList_popTable {K : ℕ} (D : Measure (FreeMonoid α)) (Kw N M : ℕ)
    (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) {n : ℕ} (hn : n < M) :
    padList (popTable D Kw N M pops T j u) n = fillDraw D Kw N pops T j (u n) := by
  rcases j with _ | i
  · simp only [popTable, padList, uniDraws]
    rw [List.getD_eq_getElem _ _ (by simpa using hn), List.getElem_ofFn]
    rfl
  rcases hs : pops[i.val]? with _ | s
  · simp only [popTable, hs, padList, List.getD_nil, fillDraw, popHit, loopDistributions,
      offLaw_dirac_one, drawFrom_dirac_one, Option.getD_none]
  · simp only [popTable, hs, padList]
    rw [List.getD_eq_getElem _ _ (by simpa using hn), List.getElem_ofFn]

/-! ## A round's draws -/

/-- A stream of draws, each measure preserving, is measure preserving. -/
lemma measurePreserving_stream {β γ : Type*} [MeasurableSpace β] [MeasurableSpace γ]
    {ρ : Measure β} {τ : Measure γ} [IsProbabilityMeasure ρ] [IsProbabilityMeasure τ] {f : β → γ}
    (hf : MeasurePreserving f ρ τ) :
    MeasurePreserving (fun u : ℕ → β => fun n => f (u n))
      (Measure.infinitePi fun _ : ℕ => ρ) (Measure.infinitePi fun _ : ℕ => τ) :=
  ⟨measurable_pi_lambda _ fun n => hf.measurable.comp (measurable_pi_apply n), by
    rw [Measure.infinitePi_map_pi (μ := fun _ : ℕ => ρ) (hf := fun _ => hf.measurable), hf.map_eq]⟩

/-- Each population's stream, each measure preserving, is measure preserving. -/
lemma measurePreserving_streams {K : ℕ} {β : Type*} [MeasurableSpace β] {ρ : Measure β}
    [IsProbabilityMeasure ρ] (L : Option (Fin K) → Measure (FreeMonoid α))
    [∀ j, IsProbabilityMeasure (L j)] (f : Option (Fin K) → β → FreeMonoid α)
    (hf : ∀ j, MeasurePreserving (f j) ρ (L j)) :
    MeasurePreserving (fun U : Option (Fin K) → ℕ → β => fun j n => f j (U j n))
      (Measure.pi fun _ : Option (Fin K) => Measure.infinitePi fun _ : ℕ => ρ)
      (Measure.pi fun j : Option (Fin K) => Measure.infinitePi fun _ : ℕ => L j) := by
  refine ⟨measurable_pi_lambda _ fun j => (measurePreserving_stream (hf j)).measurable.comp
    (measurable_pi_apply j), ?_⟩
  rw [Measure.pi_map_pi (fun j => (measurePreserving_stream (hf j)).measurable.aemeasurable)]
  congr 1
  funext j
  exact (measurePreserving_stream (hf j)).map_eq

variable {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}

/-- The round's draws after the history `h` with every miss filled, the named populations
certifying on the uniforms of their own. -/
def idealDraws (D Dsf : Measure (FreeMonoid α)) (Kw N : ℕ) (h : SpecHist α)
    (c : ClusterDraws Ω α K N) : Draws Ω (FreeMonoid α) (Option (Fin K)) :=
  ((c.1, ((fun n => drawFrom (offLaw Dsf h.suffs) (c.2.1 n),
      fun j n => fillDraw D Kw N h.pops h.prefs j (c.2.2.1 j n)),
    fun j n => drawFrom (offLaw (loopDistributions D Kw h.pops j) h.prefs) (c.2.2.2.1 j n))),
    c.2.2.2.2)

/-- The run the clustering reads off `idealDraws`. -/
def idealRun (D Dsf : Measure (FreeMonoid α)) (Kw N : ℕ) (h : SpecHist α)
    (c : ClusterDraws Ω α K N) : Run Ω (FreeMonoid α) (Option (Fin K)) :=
  streams (fun _ => ∅) none (idealDraws D Dsf Kw N h c)

/-- After the history `h`, the filled draws are drawn by `runMeasure` at the populations and
suffixes off what the earlier rounds read. -/
theorem measurePreserving_idealDraws {μ : Measure Ω} [IsProbabilityMeasure μ]
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (Kw N : ℕ) (h : SpecHist α) :
    MeasurePreserving (idealDraws (Ω := Ω) (K := K) D Dsf Kw N h) (clusterLaw μ α K N)
      (runMeasure μ (fun j => offLaw (loopDistributions D Kw h.pops j) h.prefs)
        (offLaw Dsf h.suffs)) := by
  set L : Option (Fin K) → Measure (FreeMonoid α) :=
    fun j => offLaw (loopDistributions D Kw h.pops j) h.prefs
  have hS := measurePreserving_stream
    (⟨measurable_drawFrom _, map_drawFrom _⟩ : MeasurePreserving (drawFrom (offLaw Dsf h.suffs))
      volume (offLaw Dsf h.suffs))
  have hP := measurePreserving_streams (ρ := drawLaw N) L _
    fun j => measurePreserving_fillDraw D Kw N h.pops h.prefs j
  have hC := measurePreserving_streams (ρ := (volume : Measure unitInterval)) L _
    fun j => (⟨measurable_drawFrom _, map_drawFrom _⟩ : MeasurePreserving (drawFrom (L j))
      volume (L j))
  set US := Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)
  set UP := Measure.pi fun _ : Option (Fin K) => Measure.infinitePi fun _ : ℕ => drawLaw N
  set UC := Measure.pi fun _ : Option (Fin K) =>
    Measure.infinitePi fun _ : ℕ => (volume : Measure unitInterval)
  set C := Measure.infinitePi fun _ : FreeMonoid α => fairCoin
  have e2 := (measurePreserving_prodAssoc US UP (UC.prod C)).symm
  have e3 := (measurePreserving_prodAssoc (US.prod UP) UC C).symm
  have hin := (((hS.prod hP).prod hC).prod (MeasurePreserving.id C)).comp (e3.comp e2)
  have hall := (MeasurePreserving.id μ).prod hin
  exact (measurePreserving_prodAssoc μ (((Measure.infinitePi fun _ : ℕ => offLaw Dsf h.suffs).prod
      (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => L j)).prod
      (Measure.pi fun j => Measure.infinitePi fun _ : ℕ => L j)) C).symm.comp hall

lemma measurable_idealDraws (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (Kw N : ℕ)
    (h : SpecHist α) : Measurable (idealDraws (Ω := Ω) (K := K) D Dsf Kw N h) := by
  rcases isEmpty_or_nonempty Ω with hΩ | hΩ
  · exact Subsingleton.measurable
  · exact (measurePreserving_idealDraws (μ := Measure.dirac hΩ.some) D Dsf Kw N h).measurable

lemma measurable_idealRun (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (Kw N : ℕ)
    (h : SpecHist α) : Measurable (idealRun (Ω := Ω) (K := K) D Dsf Kw N h) :=
  (measurable_streams _ none).comp (measurable_idealDraws D Dsf Kw N h)

end OrthoDFA

end
