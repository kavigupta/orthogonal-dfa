import OrthoDFA.Proofs.HarvestCode
import OrthoDFA.Proofs.DrawFrom
import OrthoDFA.Proofs.DrawMany
import OrthoDFA.Proofs.LoopRuns

/-! # The loop's harvest draws

`popDraws` takes a population's first `M` draws, a harvested population's each by its sampler over
`N` probes drawn by its own block of uniforms.

* `popDraws_miss_le`: at the cap `harvestN y (δₛ'/M)` the `M` draws miss with chance at most `δₛ'`;
* `popDraws_eq_some_law`: where they all hit, they are i.i.d. from the population off `T`;
* `popDraws_congr`, `popDrawReads_card_le`: with the uniforms fixed, the draws read only along
  their probes, at most `M · N · (3 + clog₂ L) · |mids|` strings.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α]
variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)
  (site : Random.Level3Site) (T : Set (FreeMonoid α)) (N : ℕ)

/-- A draw lands on a string its law gives mass, or on `1`. -/
lemma drawFrom_pos_or (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] (u : unitInterval) :
    0 < ν {drawFrom ν u} ∨ drawFrom ν u = 1 := by
  classical
  by_cases h : ∃ n, (u : ℝ) < massBelow ν (n + 1)
  · have hz := (drawFrom_eq_iff ν u (drawFrom ν u)).1 rfl
    rcases hz with ⟨h1, h2⟩ | ⟨h1, -⟩
    · left
      have hs := massBelow_succ ν (strCode (drawFrom ν u))
      have hset : {z' : FreeMonoid α | strCode z' = strCode (drawFrom ν u)} = {drawFrom ν u} := by
        ext z'; simp [strCode_injective.eq_iff]
      rw [hset] at hs
      have hpos : 0 < ν.real {drawFrom ν u} := by linarith
      exact (ENNReal.toReal_pos_iff.1 hpos).1
    · exact Or.inr h1
  · right
    unfold drawFrom
    rw [dif_neg h]

/-- Every draw from a law on strings at most `L` long is at most `L` long. -/
lemma drawFrom_length_le (ν : Measure (FreeMonoid α)) [IsProbabilityMeasure ν] {L : ℕ}
    (hν : ∀ᵐ x ∂ν, x.toList.length ≤ L) (u : unitInterval) :
    (drawFrom ν u).toList.length ≤ L := by
  rcases drawFrom_pos_or ν u with h | h
  · by_contra hlt
    have : ν {drawFrom ν u} = 0 := measure_mono_null (fun z hz => by
      rw [Set.mem_singleton_iff.1 hz]; exact hlt) (ae_iff.1 hν)
    rw [this] at h
    exact lt_irrefl _ h
  · rw [h]; simp

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

/-! ## A population's draws -/

/-- The probes a population's first `M` draws walk, draw `n`'s drawn from `D` by its block. -/
noncomputable def probeBlocks (D : Measure (FreeMonoid α)) (N M : ℕ)
    (u : ℕ → unitInterval × (Fin N → unitInterval)) : Fin M → Fin N → FreeMonoid α :=
  fun n t => drawFrom D ((u n).2 t)

instance (N : ℕ) : IsProbabilityMeasure (drawLaw N) := by unfold drawLaw; infer_instance

omit [Fintype α] in
/-- The first `M` coordinates of an i.i.d. sequence are i.i.d. -/
lemma map_firstM {β : Type*} [MeasurableSpace β] (ν : Measure β) [IsProbabilityMeasure ν] (M : ℕ) :
    (Measure.infinitePi fun _ : ℕ => ν).map (fun u (n : Fin M) => u n)
      = Measure.pi fun _ : Fin M => ν := by
  classical
  refine (Measure.pi_eq fun s hs => ?_).symm
  rw [Measure.map_apply (by fun_prop) (MeasurableSet.univ_pi hs)]
  set t : ℕ → Set β := fun i => if h : i < M then s ⟨i, h⟩ else Set.univ
  have hpre : (fun u (n : Fin M) => u n) ⁻¹' Set.univ.pi s = Set.pi ↑(Finset.range M) t := by
    ext u
    simp only [Set.mem_preimage, Set.mem_univ_pi, Set.mem_pi, Finset.coe_range, Set.mem_Iio, t]
    constructor
    · intro h i hi; rw [dif_pos hi]; exact h ⟨i, hi⟩ (Set.mem_univ _)
    · intro h n _; have := h n n.2; rwa [dif_pos n.2] at this
  rw [hpre, Measure.infinitePi_pi _ fun i _ => by
    simp only [t]; split_ifs; exacts [hs _, .univ]]
  rw [← Fin.prod_univ_eq_prod_range (fun i => ν (t i)) M]
  refine Finset.prod_congr rfl fun n _ => ?_
  simp only [t, dif_pos n.2]

lemma measurable_probeBlocks (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (N M : ℕ) :
    Measurable (probeBlocks D N M) :=
  measurable_pi_lambda _ fun n => measurable_pi_lambda _ fun t =>
    (measurable_drawFrom D).comp ((measurable_pi_apply t).comp
      (measurable_snd.comp (measurable_pi_apply _)))

/-- The probes the blocks draw are i.i.d. from `D`. -/
theorem map_probeBlocks (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (N M : ℕ) :
    (Measure.infinitePi fun _ : ℕ => drawLaw N).map (probeBlocks D N M)
      = Measure.pi fun _ : Fin M => Measure.pi fun _ : Fin N => D := by
  have hblock : MeasurePreserving (fun p : unitInterval × (Fin N → unitInterval) =>
      fun t => drawFrom D (p.2 t)) (drawLaw N) (Measure.pi fun _ : Fin N => D) :=
    (measurePreserving_pi _ _ fun _ => ⟨measurable_drawFrom _, map_drawFrom _⟩).comp
      (measurePreserving_snd (μ := (volume : Measure unitInterval)))
  have hfirst : MeasurePreserving (fun u (n : Fin M) => u n)
      (Measure.infinitePi fun _ : ℕ => drawLaw N) (Measure.pi fun _ : Fin M => drawLaw N) :=
    ⟨measurable_pi_lambda _ fun n => measurable_pi_apply _, map_firstM _ M⟩
  exact ((measurePreserving_pi _ _ fun _ => hblock).comp hfirst).map_eq

/-- `popDraws` at a harvested population: the sampler over its probe blocks. -/
lemma popDraws_some {K : ℕ} (M : ℕ) (pops : List (HarvestSampler α))
    (T : Finset (FreeMonoid α)) (i : Fin K) (s : HarvestSampler α) (hs : pops[i.val]? = some s)
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    popDraws D k N M pops T (some i) u
      = Random.drawManyFirst s.read k s.hyp s.site ↑T (probeBlocks D N M u) := by
  simp only [popDraws, hs]
  rfl

/-- A harvested population's `M` draws miss with the sampler's chance. -/
theorem popDraws_miss {K : ℕ} [IsProbabilityMeasure D] (M : ℕ) (pops : List (HarvestSampler α))
    (T : Finset (FreeMonoid α)) (i : Fin K) (s : HarvestSampler α) (hs : pops[i.val]? = some s) :
    (Measure.infinitePi fun _ : ℕ => drawLaw N)
        {u | popDraws D k N M pops T (some i) u = none}
      = Random.drawManyMiss s.read D k s.hyp s.site ↑T N M := by
  have hset : {u | popDraws D k N M pops T (some i) u = none}
      = probeBlocks D N M ⁻¹' {yss | Random.drawManyFirst s.read k s.hyp s.site ↑T yss = none} := by
    ext u; simp only [Set.mem_setOf_eq, Set.mem_preimage, popDraws_some D k N M pops T i s hs]
  rw [hset, ← Measure.map_apply (measurable_probeBlocks D N M) (Set.to_countable _).measurableSet,
    map_probeBlocks]
  rfl

/-- At the cap `harvestN y (δₛ'/M)`, with `y` at most the chance a probe harvests a string off
`T`, a harvested population's `M` draws miss with chance at most `δₛ'`. -/
theorem popDraws_miss_le {K : ℕ} [IsProbabilityMeasure D] (M : ℕ) (pops : List (HarvestSampler α))
    (T : Finset (FreeMonoid α)) (i : Fin K) (s : HarvestSampler α) (hs : pops[i.val]? = some s)
    {y δₛ' : ℝ} (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₛ') (hM : 0 < M)
    (hyield : y ≤ (Random.harvestLaw s.read D k s.hyp s.site).real (↑T)ᶜ) :
    (Measure.infinitePi fun _ : ℕ => drawLaw (Random.harvestN y (δₛ' / M)))
        {u | popDraws D k (Random.harvestN y (δₛ' / M)) M pops T (some i) u = none}
      ≤ ENNReal.ofReal δₛ' := by
  rw [popDraws_miss D k _ M pops T i s hs]
  exact Random.drawManyMiss_le s.read D k s.hyp s.site _ hy0 hy1 hδ hM hyield

/-- Where a harvested population's `M` draws all hit, they are i.i.d. from it off `T`. -/
theorem popDraws_eq_some_law {K : ℕ} [IsProbabilityMeasure D] (M : ℕ) (hN : 0 < N)
    (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (i : Fin K) (s : HarvestSampler α)
    (hs : pops[i.val]? = some s) :
    let raw := ((Measure.infinitePi fun _ : ℕ => drawLaw N).restrict
        {u | (popDraws D k N M pops T (some i) u).isSome = true}).map
      fun u => (popDraws D k N M pops T (some i) u).getD 1
    (if raw Set.univ = 0 ∨ raw Set.univ = ∞ then Measure.dirac 1 else (raw Set.univ)⁻¹ • raw)
      = Measure.pi fun _ : Fin M => offLaw (popLaw (Random.harvestLaw s.read D k s.hyp s.site)) T := by
  intro raw
  classical
  haveI : IsFiniteMeasure (Random.harvestLaw s.read D k s.hyp s.site) :=
    ⟨lt_top_iff_ne_top.2 (Random.harvestLaw_univ_ne_top s.read D k s.hyp s.site)⟩
  have hraw : raw = ((Measure.pi fun _ : Fin M => Measure.pi fun _ : Fin N => D).restrict
      {yss | (Random.drawManyFirst s.read k s.hyp s.site ↑T yss).isSome = true}).map
      fun yss => (Random.drawManyFirst s.read k s.hyp s.site ↑T yss).getD 1 := by
    have hfun : (fun u => (popDraws D k N M pops T (some i) u).getD 1)
        = (fun yss => (Random.drawManyFirst s.read k s.hyp s.site ↑T yss).getD 1)
          ∘ probeBlocks D N M := by
      funext u; simp only [Function.comp, popDraws_some D k N M pops T i s hs]
    have hset : {u | (popDraws D k N M pops T (some i) u).isSome = true}
        = probeBlocks D N M ⁻¹'
          {yss | (Random.drawManyFirst s.read k s.hyp s.site ↑T yss).isSome = true} := by
      ext u; simp only [Set.mem_setOf_eq, Set.mem_preimage, popDraws_some D k N M pops T i s hs]
    simp only [raw]
    rw [hfun, hset, ← Measure.map_map (measurable_of_countable _) (measurable_probeBlocks D N M),
      ← Measure.restrict_map (measurable_probeBlocks D N M) (Set.to_countable _).measurableSet,
      map_probeBlocks]
  have h := Random.drawManyLaw_eq_pi s.read D k s.hyp s.site (↑T) (m := M) hN
  unfold Random.drawManyLaw at h
  simp only at h
  rw [hraw, h]
  congr 1
  funext _
  rw [offLaw_popLaw]

/-! ## Measurability -/

lemma measurable_popDraws_isSome {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K)) :
    Measurable fun u => (popDraws D k N M pops T j u).isSome := by
  rcases j with _ | i
  · exact measurable_const
  · rcases hs : pops[i.val]? with _ | s
    · simp only [popDraws, hs]; exact measurable_const
    · have : (fun u => (popDraws D k N M pops T (some i) u).isSome)
          = (fun yss => (Random.drawManyFirst s.read k s.hyp s.site ↑T yss).isSome)
            ∘ probeBlocks D N M := by
        funext u; simp only [Function.comp, popDraws_some D k N M pops T i s hs]
      rw [this]
      exact (measurable_of_countable _).comp (measurable_probeBlocks D N M)

lemma measurable_popDraws_getD {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (d : Fin M → FreeMonoid α) :
    Measurable fun u => (popDraws D k N M pops T j u).getD d := by
  rcases j with _ | i
  · simp only [popDraws, Option.getD_some]
    exact measurable_pi_lambda _ fun n => (measurable_drawFrom _).comp
      (measurable_fst.comp (measurable_pi_apply _))
  · rcases hs : pops[i.val]? with _ | s
    · simp only [popDraws, hs]; exact measurable_const
    · have : (fun u => (popDraws D k N M pops T (some i) u).getD d)
          = (fun yss => (Random.drawManyFirst s.read k s.hyp s.site ↑T yss).getD d)
            ∘ probeBlocks D N M := by
        funext u; simp only [Function.comp, popDraws_some D k N M pops T i s hs]
      rw [this]
      exact (measurable_of_countable _).comp (measurable_probeBlocks D N M)

/-! ## Reads -/

open scoped Classical in
/-- The strings population `j`'s `M` draws read through its sampler: none for the probes'
population. -/
noncomputable def popDrawReads {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ)
    (pops : List (HarvestSampler α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval)) : Finset (FreeMonoid α) :=
  match j with
  | none => ∅
  | some i =>
    match pops[i.val]? with
    | some s => Random.drawManyReads s.read s.hyp.tree s.hyp.edges k (probeBlocks D N M u)
    | none => ∅

/-- With the uniforms fixed, a population's draws depend on its sampler's read only through
`popDrawReads`. -/
theorem popDraws_congr {K : ℕ} (D : Measure (FreeMonoid α)) (k N M : ℕ)
    (pops pops' : List (HarvestSampler α)) (T : Finset (FreeMonoid α)) (j : Option (Fin K))
    (u : ℕ → unitInterval × (Fin N → unitInterval))
    (hlen : pops.length = pops'.length)
    (hs : ∀ i : ℕ, ∀ s s', pops[i]? = some s → pops'[i]? = some s' →
      s.hyp = s'.hyp ∧ s.site = s'.site)
    (h : ∀ w ∈ popDrawReads D k N M pops j u, ∀ i : Fin K, j = some i → ∀ s s',
      pops[i.val]? = some s → pops'[i.val]? = some s' → s.read w = s'.read w) :
    popDraws D k N M pops T j u = popDraws D k N M pops' T j u := by
  classical
  rcases j with _ | i
  · rfl
  rcases hp : pops[i.val]? with _ | s
  · have hp' : pops'[i.val]? = none := by
      rw [List.getElem?_eq_none_iff] at hp ⊢; omega
    simp only [popDraws, hp, hp']
  obtain ⟨s', hp'⟩ : ∃ s', pops'[i.val]? = some s' := by
    refine Option.ne_none_iff_exists'.1 fun h' => ?_
    rw [List.getElem?_eq_none_iff] at h'
    have := (List.getElem?_eq_some_iff.1 hp).1
    omega
  obtain ⟨hhyp, hsite⟩ := hs i s s' hp hp'
  rw [popDraws_some D k N M pops T i s hp, popDraws_some D k N M pops' T i s' hp', ← hhyp, ← hsite]
  unfold Random.drawManyFirst
  have hi : ∀ n, Random.harvestFirst s.read k s.hyp s.site ↑T
        (List.ofFn (probeBlocks D N M u n))
      = Random.harvestFirst s'.read k s.hyp s.site ↑T (List.ofFn (probeBlocks D N M u n)) := by
    intro n
    refine Random.harvestFirst_congr k s.hyp s.site _ _ fun w hw => ?_
    refine h w ?_ i rfl s s' hp hp'
    simp only [popDrawReads, hp, Random.drawManyReads]
    exact Finset.mem_biUnion.2 ⟨n, Finset.mem_univ _, hw⟩
  simp only [hi]

/-- Where `D`'s probes are at most `L` long, a population's `M` draws read at most
`M · N · (3 + clog₂ L) · |mids|` strings, `mids` its sampler's hypothesis's. -/
theorem popDrawReads_card_le {K : ℕ} (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N M : ℕ) (pops : List (HarvestSampler α)) (i : Fin K) (s : HarvestSampler α)
    (hs : pops[i.val]? = some s) {L : ℕ} (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (u : ℕ → unitInterval × (Fin N → unitInterval)) :
    (popDrawReads D k N M pops (some i) u).card
      ≤ M * (N * ((3 + Nat.clog 2 L) * (Random.mids s.hyp.tree).card)) := by
  simp only [popDrawReads, hs]
  refine Random.drawManyReads_card_le s.read k s.hyp _ fun n x hx => ?_
  obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx
  exact drawFrom_length_le D hD _

end OrthoDFA
