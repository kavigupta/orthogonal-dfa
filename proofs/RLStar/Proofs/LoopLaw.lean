import RLStar.Proofs.HarvestDraw
import RLStar.Proofs.LoopAux
import RLStar.Proofs.RawRun
import Mathlib.Probability.ProductMeasure

/-!
# Coupling a round's draws with i.i.d. draws

A harvested population's draw reads a uniform and a block of `N`.  `fillDraw` is the draw with a
miss filled by the uniform's draw from the population off the earlier prefixes: a hit's law is a
multiple of the population off them, so the filled draw is drawn from it
(`measurePreserving_fillDraw`).  `idealDraws` fills every draw and lets the named populations
certify on draws of their own; it is drawn by `runMeasure` at the populations off the earlier
prefixes (`measurePreserving_idealDraws`).  Where `clusterDraws` hits, it agrees with
`idealDraws` on each stream's first `M` draws (`clusterDraws_agree`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## One draw -/

/-- Population `j`'s draw by `u`, a harvested population's miss filled by the uniform's draw from
the population off `T`. -/
def fillDraw {K : ℕ} (D : Measure (FreeMonoid α)) (k N : ℕ) (pops : List (SpecSampler α))
    (T : Finset (FreeMonoid α)) (j : Option (Fin K)) (u : unitInterval × (Fin N → unitInterval)) :
    FreeMonoid α :=
  match j with
  | none => drawFrom (offLaw D T) u.1
  | some i =>
    match pops[i.val]? with
    | some s => (Random.harvestFirst s.read k s.hyp s.site ↑T
        (List.ofFn fun t => drawFrom D (u.2 t))).getD
          (drawFrom (offLaw (popLaw (Random.harvestLaw s.read D k s.hyp s.site)) T) u.1)
    | none => 1

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

lemma offLaw_dirac_one (T : Finset (FreeMonoid α)) :
    offLaw (Measure.dirac (1 : FreeMonoid α)) T = Measure.dirac 1 := by
  classical
  unfold offLaw popLaw
  rw [restrict_dirac' ((Set.to_countable _).measurableSet)]
  split_ifs with h1 h2 <;> simp_all

instance (N : ℕ) : IsProbabilityMeasure (Measure.pi fun _ : Fin N => (volume : Measure unitInterval)) :=
  inferInstance

omit [DecidableEq α] in
lemma measurable_getD {β : Type*} [MeasurableSpace β] {f : β → Option (FreeMonoid α)}
    {d : β → FreeMonoid α} (hS : MeasurableSet {b | (f b).isSome = true})
    (h1 : Measurable fun b => (f b).getD 1) (h2 : Measurable d) :
    Measurable fun b => (f b).getD (d b) := by
  classical
  have : (fun b => (f b).getD (d b)) = {b | (f b).isSome = true}.piecewise (fun b => (f b).getD 1) d := by
    funext b
    by_cases hb : (f b).isSome = true
    · rw [Set.piecewise_eq_of_mem (s := {b | (f b).isSome = true}) _ _ hb]
      obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hb
      rw [hv]; rfl
    · rw [Set.piecewise_eq_of_notMem (s := {b | (f b).isSome = true}) _ _ hb]
      rw [Option.not_isSome_iff_eq_none.1 hb]; rfl
  rw [this]
  exact h1.piecewise hS h2

omit [DecidableEq α] in
lemma measurable_fillSampler (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N : ℕ)
    (s : SpecSampler α) (T : Finset (FreeMonoid α)) (τ : Measure (FreeMonoid α))
    [IsFiniteMeasure τ] :
    Measurable fun u : unitInterval × (Fin N → unitInterval) =>
      (Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn fun t => drawFrom D (u.2 t))).getD
        (drawFrom τ u.1) := by
  have hpr : Measurable fun u : unitInterval × (Fin N → unitInterval) =>
      fun t => drawFrom D (u.2 t) :=
    measurable_pi_lambda _ fun t => (measurable_drawFrom D).comp
      ((measurable_pi_apply t).comp measurable_snd)
  exact measurable_getD (hpr ((Set.to_countable {xs : Fin N → FreeMonoid α |
      (Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn xs)).isSome = true}).measurableSet))
    ((measurable_of_countable (fun xs : Fin N → FreeMonoid α =>
      (Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn xs)).getD 1)).comp hpr)
    ((measurable_drawFrom τ).comp measurable_fst)

omit [DecidableEq α] in
/-- A harvested draw with its miss filled is drawn from the harvested population off `T`. -/
theorem map_fillSampler (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N : ℕ)
    (s : SpecSampler α) (T : Finset (FreeMonoid α)) :
    (drawLaw N).map (fun u : unitInterval × (Fin N → unitInterval) =>
        (Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn fun t => drawFrom D (u.2 t))).getD
          (drawFrom (offLaw (popLaw (Random.harvestLaw s.read D k s.hyp s.site)) T) u.1))
      = offLaw (popLaw (Random.harvestLaw s.read D k s.hyp s.site)) T := by
  classical
  set ν := Random.harvestLaw s.read D k s.hyp s.site
  haveI : IsFiniteMeasure ν := ⟨lt_top_iff_ne_top.2 (Random.harvestLaw_univ_ne_top _ _ _ _ _)⟩
  set ν' := ν.restrict (↑T)ᶜ
  set τ := offLaw (popLaw ν) T
  have hτ : τ = popLaw ν' := offLaw_popLaw T
  set P := Measure.pi fun _ : Fin N => (volume : Measure unitInterval)
  set pr : (Fin N → unitInterval) → (Fin N → FreeMonoid α) := fun b t => drawFrom D (b t)
  have hpr : MeasurePreserving pr P (Measure.pi fun _ : Fin N => D) :=
    measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩
  set g : (Fin N → FreeMonoid α) → Option (FreeMonoid α) :=
    fun xs => Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn xs)
  set fill : unitInterval × (Fin N → unitInterval) → FreeMonoid α :=
    fun u => (g (pr u.2)).getD (drawFrom τ u.1)
  have hfill : Measurable fill := measurable_fillSampler D k N s T τ
  have hdec : (drawLaw N).map fill
      = Random.harvestCodeLaw s.read D k s.hyp s.site ↑T N
        + Random.harvestMiss s.read D k s.hyp s.site ↑T N • τ := by
    ext A hA
    set S1 : Set (Fin N → FreeMonoid α) := (fun z => (g z).getD 1) ⁻¹' A ∩ {z | (g z).isSome = true}
    set S2 : Set (Fin N → FreeMonoid α) := {z | g z = none}
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
    rw [Measure.map_apply hfill hA, hset,
      measure_union hdisj ((measurable_drawFrom τ hA).prod (hpr.measurable hS2)), drawLaw,
      Measure.prod_prod, Measure.prod_prod, measure_univ, one_mul,
      ← Measure.map_apply hpr.measurable hS1, ← Measure.map_apply hpr.measurable hS2, hpr.map_eq,
      ← Measure.map_apply (measurable_drawFrom τ) hA, map_drawFrom, Measure.add_apply,
      Measure.smul_apply, smul_eq_mul, Random.harvestCodeLaw_apply', mul_comm]
    rfl
  obtain ⟨c, hc⟩ : ∃ c : ℝ≥0∞, Random.harvestCodeLaw s.read D k s.hyp s.site ↑T N = c • ν' := by
    rcases Nat.eq_zero_or_pos N with rfl | hN
    · refine ⟨0, ?_⟩
      rw [zero_smul]
      unfold Random.harvestCodeLaw
      have : {xs : Fin 0 → FreeMonoid α |
          (Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn xs)).isSome = true} = ∅ := by
        ext xs; simp [Random.harvestFirst]
      rw [this, Measure.restrict_empty, Measure.map_zero]
    · obtain ⟨c, -, -, h⟩ := Random.harvestCodeLaw_smul s.read D k s.hyp s.site (↑T) N hN
      exact ⟨c, h⟩
  haveI : IsProbabilityMeasure ((drawLaw N).map fill) := Measure.isProbabilityMeasure_map hfill.aemeasurable
  rw [hc, hτ] at hdec
  rw [hτ]
  exact eq_popLaw_of_combo hdec

/-- A filled draw is drawn from its population off `T`. -/
theorem measurePreserving_fillDraw {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N : ℕ) (pops : List (SpecSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K)) :
    MeasurePreserving (fillDraw D k N pops T j) (drawLaw N)
      (offLaw (loopDistributions D k pops j) T) := by
  classical
  have hfst : MeasurePreserving (Prod.fst : unitInterval × (Fin N → unitInterval) → unitInterval)
      (drawLaw N) volume := measurePreserving_fst
  rcases j with _ | i
  · exact (⟨measurable_drawFrom _, map_drawFrom _⟩ : MeasurePreserving (drawFrom (offLaw D T))
      volume (offLaw D T)).comp hfst
  rcases hs : pops[i.val]? with _ | s
  · have hf : fillDraw D k N pops T (some i) = fun _ => 1 := by
      funext u; simp only [fillDraw, hs]
    have hl : loopDistributions D k pops (some i) = Measure.dirac 1 := by
      simp only [loopDistributions, hs]
    rw [hf, hl, offLaw_dirac_one]
    refine ⟨measurable_const, ?_⟩
    rw [Measure.map_const, measure_univ, one_smul]
  · have hf : fillDraw D k N pops T (some i) = fun u =>
        (Random.harvestFirst s.read k s.hyp s.site ↑T (List.ofFn fun t => drawFrom D (u.2 t))).getD
          (drawFrom (offLaw (popLaw (Random.harvestLaw s.read D k s.hyp s.site)) T) u.1) := by
      funext u; simp only [fillDraw, hs]
    have hl : loopDistributions D k pops (some i)
        = popLaw (Random.harvestLaw s.read D k s.hyp s.site) := by
      simp only [loopDistributions, hs]
    rw [hf, hl]
    exact ⟨measurable_fillSampler D k N s T _, map_fillSampler D k N s T⟩

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
def idealDraws (D Dsf : Measure (FreeMonoid α)) (k N : ℕ) (h : SpecHist α)
    (c : ClusterDraws Ω α K N) : Draws Ω (FreeMonoid α) (Option (Fin K)) :=
  ((c.1, ((fun n => drawFrom (offLaw Dsf h.suffs) (c.2.1 n),
      fun j n => fillDraw D k N h.pops h.prefs j (c.2.2.1 j n)),
    fun j n => drawFrom (offLaw (loopDistributions D k h.pops j) h.prefs) (c.2.2.2.1 j n))),
    c.2.2.2.2)

/-- The run the clustering reads off `idealDraws`. -/
def idealRun (D Dsf : Measure (FreeMonoid α)) (k N : ℕ) (h : SpecHist α)
    (c : ClusterDraws Ω α K N) : Run Ω (FreeMonoid α) (Option (Fin K)) :=
  streams (fun _ => ∅) none (idealDraws D Dsf k N h c)

/-- After the history `h`, the filled draws are drawn by `runMeasure` at the populations and
suffixes off what the earlier rounds read. -/
theorem measurePreserving_idealDraws {μ : Measure Ω} [IsProbabilityMeasure μ]
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N : ℕ) (h : SpecHist α) :
    MeasurePreserving (idealDraws (Ω := Ω) (K := K) D Dsf k N h) (clusterLaw μ α K N)
      (runMeasure μ (fun j => offLaw (loopDistributions D k h.pops j) h.prefs)
        (offLaw Dsf h.suffs)) := by
  set L : Option (Fin K) → Measure (FreeMonoid α) :=
    fun j => offLaw (loopDistributions D k h.pops j) h.prefs
  have hS := measurePreserving_stream
    (⟨measurable_drawFrom _, map_drawFrom _⟩ : MeasurePreserving (drawFrom (offLaw Dsf h.suffs))
      volume (offLaw Dsf h.suffs))
  have hP := measurePreserving_streams (ρ := drawLaw N) L _
    fun j => measurePreserving_fillDraw D k N h.pops h.prefs j
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

lemma measurable_idealDraws (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N : ℕ)
    (h : SpecHist α) : Measurable (idealDraws (Ω := Ω) (K := K) D Dsf k N h) := by
  rcases isEmpty_or_nonempty Ω with hΩ | hΩ
  · exact Subsingleton.measurable
  · exact (measurePreserving_idealDraws (μ := Measure.dirac hΩ.some) D Dsf k N h).measurable

lemma measurable_idealRun (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k N : ℕ)
    (h : SpecHist α) : Measurable (idealRun (Ω := Ω) (K := K) D Dsf k N h) :=
  (measurable_streams _ none).comp (measurable_idealDraws D Dsf k N h)

/-! ## The round's draws where they hit -/

/-- Two round's draws agree on the noise, the suffixes, the hash, the first `n j` draws of each
population `j`'s stream and the first `n uni` of the probes' certification stream. -/
def AgreeTo {J : Type*} (uni : J) (n : J → ℕ) (x y : Draws Ω (FreeMonoid α) J) : Prop :=
  x.1.1 = y.1.1 ∧ x.1.2.1.1 = y.1.2.1.1 ∧ x.2 = y.2
    ∧ (∀ j, ∀ m < n j, x.1.2.1.2 j m = y.1.2.1.2 j m) ∧ ∀ m < n uni, x.1.2.2 uni m = y.1.2.2 uni m

/-- Where a population's `M` draws hit, they are its filled draws. -/
lemma popDraws_eq_some {D : Measure (FreeMonoid α)} {k N M : ℕ} {pops : List (SpecSampler α)}
    {T : Finset (FreeMonoid α)} {j : Option (Fin K)}
    {u : ℕ → unitInterval × (Fin N → unitInterval)} {f : Fin M → FreeMonoid α}
    (hf : popDraws D k N M pops T j u = some f) (n : Fin M) :
    f n = fillDraw D k N pops T j (u n) := by
  classical
  rcases j with _ | i
  · simp only [popDraws, Option.some.injEq] at hf
    rw [← hf]; rfl
  rcases hs : pops[i.val]? with _ | s
  · simp only [popDraws, hs, Option.some.injEq] at hf
    rw [← hf]; simp only [fillDraw, hs]
  · simp only [popDraws, hs] at hf
    unfold Random.drawManyFirst at hf
    split_ifs at hf with hall
    simp only [Option.some.injEq] at hf
    rw [← hf]
    simp only [fillDraw, hs]
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 (hall n)
    rw [hv]
    rfl

/-- The round's draws hit where every population's `M` draws hit. -/
lemma clusterDraws_isSome_iff {D Dsf : Measure (FreeMonoid α)} {k N M : ℕ} {h : SpecHist α}
    {c : ClusterDraws Ω α K N} :
    (clusterDraws D Dsf k N M h c).isSome
      ↔ ∀ j, (popDraws D k N M h.pops h.prefs j (c.2.2.1 j)).isSome := by
  classical
  unfold clusterDraws allSome
  split_ifs with hall
  · simp only [Option.map_some, Option.isSome_some, true_iff]; exact hall
  · simp only [Option.map_none, Option.isSome_none, Bool.false_eq_true, false_iff]; exact hall

/-- Where population `i`'s first `M` draws hit, its draws are the first of its filled draws,
stopped where `stop` first holds or at `M`. -/
lemma popDrawList_of_hit {D : Measure (FreeMonoid α)} {k N M : ℕ} {pops : List (SpecSampler α)}
    {T : Finset (FreeMonoid α)} (stop : List (FreeMonoid α) → Prop) {i : Fin K}
    {s : SpecSampler α} (hs : pops[i.val]? = some s)
    {u : ℕ → unitInterval × (Fin N → unitInterval)}
    (hhit : (popDraws D k N M pops T (some i) u).isSome) :
    ∃ l, popDrawList D k N M pops T stop (some i) u = some l ∧ l.length ≤ M
      ∧ (∀ n (hn : n < l.length), l[n] = fillDraw D k N pops T (some i) (u n))
      ∧ (stop l ∨ l.length = M) ∧ ∀ m < l.length, ¬ stop (l.take m) := by
  classical
  have hstep : ∀ n < M, (Random.harvestFirst s.read k s.hyp s.site ↑T
      (List.ofFn fun t : Fin N => drawFrom D ((u n).2 t))).isSome := by
    intro n hn
    simp only [popDraws, hs] at hhit
    unfold Random.drawManyFirst at hhit
    split_ifs at hhit with hall
    · exact hall ⟨n, hn⟩
    · simp at hhit
  have hsome : (popDrawList D k N M pops T stop (some i) u).isSome := by
    simp only [popDrawList, hs]
    exact growList_isSome fun m hm => by simpa using hstep m hm
  obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 hsome
  refine ⟨l, hl, ?_⟩
  simp only [popDrawList, hs] at hl
  obtain ⟨e, rfl, hlen, hfe, hstop, hmin⟩ := growList_eq_some hl
  simp only [List.nil_append, List.length_nil, Nat.zero_add] at hfe hstop hmin ⊢
  refine ⟨hlen, fun n hn => ?_, hstop, hmin⟩
  simp only [fillDraw, hs, hfe n hn, Option.getD_some]

/-- Where every population's first `M` draws hit, the round's draws are the first of its filled
draws: all `M` of the probes' population's, and a harvested population's stopped where they first
hold every scheduled state's samples, or at `M`. -/
theorem drawLists_of_hit {D Dsf : Measure (FreeMonoid α)} {k N M : ℕ} {sched : Finset State}
    {h : SpecHist α} {c : ClusterDraws Ω α K N} (hhit : (clusterDraws D Dsf k N M h c).isSome) :
    ∃ tabs, drawLists D k N M sched h c.2 = some tabs
      ∧ AgreeTo none (fun j => (tabs j).length) (drawsOf c.1 D Dsf M h.prefs h.suffs c.2 tabs)
        (idealDraws D Dsf k N h c)
      ∧ tabs none = uniDraws D M h.prefs (c.2.2.1 none)
      ∧ ∀ i s, h.pops[i.val]? = some s → (tabs (some i)).length ≤ M
        ∧ (DrawsHold sched c.2.2.2.2 (uniTable D M h.prefs (c.2.2.1 none)) (tabs (some i))
          ∨ (tabs (some i)).length = M)
        ∧ ∀ m < (tabs (some i)).length,
          ¬ DrawsHold sched c.2.2.2.2 (uniTable D M h.prefs (c.2.2.1 none)) ((tabs (some i)).take m) := by
  classical
  set stop := DrawsHold sched c.2.2.2.2 (uniTable D M h.prefs (c.2.2.1 none))
  have hhit' := clusterDraws_isSome_iff.1 hhit
  have hall : ∀ j, (popDrawList D k N M h.pops h.prefs stop j (c.2.2.1 j)).isSome := by
    intro j
    rcases j with _ | i
    · rfl
    rcases hs : h.pops[i.val]? with _ | s
    · simp [popDrawList, hs]
    · obtain ⟨l, hl, -⟩ := popDrawList_of_hit stop hs (hhit' (some i))
      rw [hl]; rfl
  set tabs : Option (Fin K) → List (FreeMonoid α) := fun j => (popDrawList D k N M h.pops h.prefs
    stop j (c.2.2.1 j)).get (hall j) with htabs
  have hj : ∀ j, popDrawList D k N M h.pops h.prefs stop j (c.2.2.1 j) = some (tabs j) :=
    fun j => (Option.some_get (hall j)).symm
  have hdl : drawLists D k N M sched h c.2 = some tabs := by
    unfold drawLists allSome
    rw [dif_pos hall]
  have hnone : tabs none = uniDraws D M h.prefs (c.2.2.1 none) := by
    have := hj none
    simp only [popDrawList, Option.some.injEq] at this
    exact this.symm
  refine ⟨tabs, hdl, ⟨rfl, rfl, rfl, fun j n hn => ?_, fun n hn => ?_⟩, hnone, fun i s hs => ?_⟩
  · simp only [drawsOf, idealDraws, padList]
    rcases j with _ | i
    · have hn' : n < (uniDraws D M h.prefs (c.2.2.1 none)).length := by rw [← hnone]; exact hn
      simp only [hnone]
      rw [List.getD_eq_getElem _ _ hn']
      simp only [uniDraws, List.getElem_ofFn, fillDraw]
    rw [List.getD_eq_getElem _ _ hn]
    rcases hs : h.pops[i.val]? with _ | s
    · have := hj (some i)
      simp only [popDrawList, hs, Option.some.injEq] at this
      simp [← this] at hn
    · obtain ⟨l, hl, -, hle, -⟩ := popDrawList_of_hit stop hs (hhit' (some i))
      have e : l = tabs (some i) := Option.some.inj (hl.symm.trans (hj (some i)))
      subst e
      exact hle n hn
  · simp only at hn
    rw [hnone, uniDraws, List.length_ofFn] at hn
    simp only [drawsOf, idealDraws, padDraws, dif_pos hn, loopDistributions]
  · obtain ⟨l, hl, hlen, -, hstop, hmin⟩ := popDrawList_of_hit stop hs (hhit' (some i))
    have e : l = tabs (some i) := Option.some.inj (hl.symm.trans (hj (some i)))
    subst e
    exact ⟨hlen, hstop, hmin⟩

/-- The event that the round's draws hit is measurable. -/
lemma measurableSet_clusterDraws_isSome (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (h : SpecHist α) :
    MeasurableSet {c : ClusterDraws Ω α K N | (clusterDraws D Dsf k N M h c).isSome} := by
  have : {c : ClusterDraws Ω α K N | (clusterDraws D Dsf k N M h c).isSome}
      = ⋂ j, (fun c : ClusterDraws Ω α K N => c.2.2.1 j) ⁻¹'
        {u | (popDraws D k N M h.pops h.prefs j u).isSome} := by
    ext c; simp only [Set.mem_setOf_eq, Set.mem_iInter, Set.mem_preimage, clusterDraws_isSome_iff]
  rw [this]
  refine MeasurableSet.iInter fun j => ?_
  exact ((measurable_pi_apply j).comp (measurable_fst.comp (measurable_snd.comp measurable_snd)))
    ((measurable_popDraws_isSome D k N M h.pops h.prefs j) (measurableSet_singleton true))

end OrthoDFA

end
