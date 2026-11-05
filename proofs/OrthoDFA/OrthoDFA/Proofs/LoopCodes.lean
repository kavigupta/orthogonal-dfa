import OrthoDFA.Proofs.LoopScan

/-!
# A round's sampler from finite data

Level 3's edge tables only ever hold edges out of the tree's leaves, finitely many, and after `T`
probes its midfixes are at most `T` long, so a probe at most `L` long walked under the round's last
hypothesis reads only strings at most `L + T` long. So the sampler a round hands on is fixed, on
probes at most `L` long, by finite data: its harvest's site, its last hypothesis's tree and edge
graph, and the round's reads of the strings at most `N` long (`decodePop_alike`). The data range
over a countable type, so a round can be split off from the rounds before it; and alike samplers
draw alike, run by run (`idealRun_alike`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

/-! ## Countable data -/

/-- The strings at most `N` long. -/
def shortStrings (N : ℕ) : Finset (FreeMonoid α) :=
  (Finset.range (N + 1)).biUnion fun n =>
    (Finset.univ : Finset (Fin n → α)).image fun f => FreeMonoid.ofList (List.ofFn f)

theorem mem_shortStrings {N : ℕ} {z : FreeMonoid α} :
    z ∈ shortStrings N ↔ z.toList.length ≤ N := by
  constructor
  · intro h
    obtain ⟨n, hn, hz⟩ := Finset.mem_biUnion.1 h
    obtain ⟨f, -, rfl⟩ := Finset.mem_image.1 hz
    simp only [Finset.mem_range] at hn
    simp only [FreeMonoid.toList_ofList, List.length_ofFn]
    omega
  · intro h
    refine Finset.mem_biUnion.2 ⟨z.toList.length, Finset.mem_range.2 (by omega),
      Finset.mem_image.2 ⟨fun i => z.toList.get i, Finset.mem_univ _, ?_⟩⟩
    apply FreeMonoid.toList.injective
    simp


namespace DTree

/-- The tree read in preorder, a leaf as `none`. -/
def code : DTree α → List (Option (FreeMonoid α))
  | .leaf => [none]
  | .node m r a => some m :: (code r ++ code a)

theorem code_append_inj : ∀ (t t' : DTree α) (l l' : List (Option (FreeMonoid α))),
    t.code ++ l = t'.code ++ l' → t = t' ∧ l = l'
  | .leaf, .leaf, l, l', h => by simpa [code] using h
  | .leaf, .node _ _ _, _, _, h => by simp [code] at h
  | .node _ _ _, .leaf, _, _, h => by simp [code] at h
  | .node m r a, .node m' r' a', l, l', h => by
    simp only [code, List.cons_append, List.cons.injEq, Option.some.injEq,
      List.append_assoc] at h
    obtain ⟨hm, h1⟩ := h
    obtain ⟨rfl, h2⟩ := code_append_inj r r' _ _ h1
    obtain ⟨rfl, rfl⟩ := code_append_inj a a' _ _ h2
    exact ⟨by rw [hm], rfl⟩

theorem code_injective : Function.Injective (code (α := α)) := fun t t' h =>
  (code_append_inj t t' [] [] (by simpa using h)).1

end DTree

instance : Countable (DTree α) := DTree.code_injective.countable

instance : Countable Random.Level3Site :=
  Function.Injective.countable (f := fun s : Random.Level3Site => match s with
    | .ends => false
    | .middles => true) fun a b h => by
    cases a <;> cases b <;> simp_all

/-! ## Finitely many edges -/

/-- The table holds finitely many edges. -/
abbrev EdgesFin (E : Edges α) : Prop := Set.Finite {qc : List Bool × α | E qc ≠ none}

omit [Fintype α] [DecidableEq α] in
theorem EdgesFin.mono {E E' : Edges α} (hE : EdgesFin E) (pc : List Bool × α)
    (h : ∀ q c, E' (q, c) ≠ none → E (q, c) ≠ none ∨ (q, c) = pc) : EdgesFin E' := by
  refine (hE.union (Set.finite_singleton pc)).subset fun qc hqc => ?_
  rcases h qc.1 qc.2 hqc with h | h
  · exact .inl h
  · exact .inr h

/-- A round's last hypothesis holds edges only out of its leaves, so finitely many. -/
theorem result_fin (C : Random.Level3Settings) (read : FreeMonoid α → ARU) {T : ℕ}
    (draws : Fin T → FreeMonoid α) :
    EdgesFin (Random.indecisiveLStarLevel3 C read draws).hypothesis.edges := by
  classical
  have hk := (Random.runProbes_tree read C (List.ofFn draws)).1.keys
  refine ((Random.runProbes read C Random.initialState (List.ofFn draws)).1.tree.leaves.toFinset
    ×ˢ (Finset.univ : Finset α)).finite_toSet.subset fun qc hqc => ?_
  rcases he : (Random.runProbes read C Random.initialState (List.ofFn draws)).1.edges qc with
    _ | ⟨t, w⟩
  · exact absurd he hqc
  · simp only [Finset.coe_product, Finset.coe_univ, Set.prod_univ, Set.mem_preimage,
      Finset.mem_coe, List.mem_toFinset]
    exact hk qc t w he

/-! ## Graphs -/

open scoped Classical in
/-- An edge table's graph, where it holds finitely many edges. -/
def edgeGraph (E : Edges α) : Finset ((List Bool × α) × (List Bool × FreeMonoid α)) :=
  if h : EdgesFin E then h.toFinset.image fun qc => (qc, (E qc).getD ([], 1)) else ∅

open scoped Classical in
/-- The table a graph draws. -/
def decodeEdges (F : Finset ((List Bool × α) × (List Bool × FreeMonoid α))) : Edges α :=
  fun qc => if h : ∃ v, (qc, v) ∈ F then some h.choose else none

theorem decodeEdges_edgeGraph {E : Edges α} (hE : EdgesFin E) :
    decodeEdges (edgeGraph E) = E := by
  have hg : edgeGraph E = hE.toFinset.image fun qc => (qc, (E qc).getD ([], 1)) := by
    unfold edgeGraph; exact dif_pos hE
  rw [hg]
  funext ⟨q, c⟩
  unfold decodeEdges
  rcases he : E (q, c) with _ | v
  · rw [dif_neg]
    rintro ⟨v, hv⟩
    obtain ⟨qc, hqc, h⟩ := Finset.mem_image.1 hv
    simp only [Prod.mk.injEq] at h
    rw [Set.Finite.mem_toFinset, Set.mem_setOf_eq, h.1, he] at hqc
    exact hqc rfl
  · have hmem : ((q, c), v) ∈ hE.toFinset.image fun qc => (qc, (E qc).getD ([], 1)) := by
      refine Finset.mem_image.2 ⟨(q, c), ?_, by simp [he]⟩
      rw [Set.Finite.mem_toFinset]
      simp [he]
    have hex : ∃ v', ((q, c), v') ∈ hE.toFinset.image
        fun qc => (qc, (E qc).getD ([], 1)) := ⟨v, hmem⟩
    rw [dif_pos hex]
    obtain ⟨qc, -, h⟩ := Finset.mem_image.1 hex.choose_spec
    simp only [Prod.mk.injEq] at h
    rw [← h.2, h.1, he, Option.getD_some]

/-- A read's graph on the strings at most `N` long. -/
def readGraph (N : ℕ) (rd : FreeMonoid α → ARU) : Finset (FreeMonoid α × ARU) :=
  (shortStrings N).image fun z => (z, rd z)

open scoped Classical in
/-- The reads a graph draws, undecided off it. -/
def decodeReads (R : Finset (FreeMonoid α × ARU)) (z : FreeMonoid α) : ARU :=
  if h : ∃ a, (z, a) ∈ R then h.choose else .undecided

theorem decodeReads_readGraph {N : ℕ} (rd : FreeMonoid α → ARU) {z : FreeMonoid α}
    (hz : z.toList.length ≤ N) : decodeReads (readGraph N rd) z = rd z := by
  unfold decodeReads
  have hex : ∃ a, (z, a) ∈ readGraph N rd :=
    ⟨rd z, Finset.mem_image.2 ⟨z, mem_shortStrings.2 hz, rfl⟩⟩
  rw [dif_pos hex]
  obtain ⟨z', -, h⟩ := Finset.mem_image.1 hex.choose_spec
  simp only [Prod.mk.injEq] at h
  rw [← h.2, h.1]

/-! ## A round's sampler from its data -/

/-- A harvest as finite data: its site, its hypothesis's tree and edge graph, and reads, and the
prefixes and suffixes its round's clustering read. -/
abbrev HarvestCode (α : Type*) :=
  Random.Level3Site × (DTree α × Finset ((List Bool × α) × (List Bool × FreeMonoid α)))
    × Finset (FreeMonoid α × ARU) × Finset (FreeMonoid α) × Finset (FreeMonoid α)

/-- A round's harvest, with its reads of the strings at most `N` long. -/
def popCode (N : ℕ) (o : SpecRound α) : Option (HarvestCode α) :=
  o.result.ending.harvested.map fun site =>
    (site, (o.result.hypothesis.tree, edgeGraph o.result.hypothesis.edges),
      readGraph N o.read, o.prefs, o.suffs)

/-- The sampler the data hand on. -/
def decodePop (c : HarvestCode α) : SpecSampler α :=
  ⟨decodeReads c.2.2.1, ⟨c.2.1.1, decodeEdges c.2.1.2⟩, c.1⟩

/-- The history the data hand on. -/
def decodeHist (cs : List (HarvestCode α)) : SpecHist α :=
  ⟨cs.map decodePop, cs.foldr (fun c T => c.2.2.2.1 ∪ T) ∅, cs.foldr (fun c T => c.2.2.2.2 ∪ T) ∅⟩

/-- The data of the rounds' harvests, in order. -/
def codesOf (N : ℕ) (rounds : List (SpecRound α)) : List (HarvestCode α) :=
  rounds.filterMap (popCode N)

/-- Two samplers alike on probes at most `L` long: the same hypothesis and site, and the same
read of every string at most `L` long followed by a midfix of the hypothesis. -/
def SamplerAlike (L : ℕ) (s s' : SpecSampler α) : Prop :=
  s.hyp = s'.hyp ∧ s.site = s'.site
    ∧ ∀ y : FreeMonoid α, y.toList.length ≤ L → ∀ m ∈ Random.mids s.hyp.tree,
      s.read (y * m) = s'.read (y * m)

/-- Two histories alike on probes at most `L` long. -/
def HistAlike (L : ℕ) (h h' : SpecHist α) : Prop :=
  List.Forall₂ (SamplerAlike L) h.pops h'.pops ∧ h.prefs = h'.prefs ∧ h.suffs = h'.suffs

/-- A round's sampler is alike the one its data hand on, on probes at most `L` long, where
`N ≥ L + T`, `T` its probes. -/
theorem decodePop_alike {L N : ℕ} (C : Random.Level3Settings) (hN : L + C.probeBudget ≤ N)
    (o : SpecRound α) (xs : Fin C.probeBudget → FreeMonoid α)
    (ho : o.result = Random.indecisiveLStarLevel3 C o.read xs) {c : HarvestCode α}
    {s : SpecSampler α} (hc : popCode N o = some c) (hs : o.sampler = some s) :
    SamplerAlike L (decodePop c) s := by
  unfold popCode at hc
  unfold SpecRound.sampler at hs
  rcases hh : o.result.ending.harvested with _ | site
  · rw [hh] at hc; simp at hc
  rw [hh] at hc hs
  simp only [Option.map_some, Option.some.injEq] at hc hs
  subst hc hs
  have hfin := result_fin C o.read xs
  have hmid := (Random.runProbes_tree o.read C (List.ofFn xs)).2.2
  rw [List.length_ofFn] at hmid
  rw [← ho] at hfin
  have hv : (⟨o.result.hypothesis.tree,
      decodeEdges (edgeGraph o.result.hypothesis.edges)⟩ : HypothesisDFA α)
      = o.result.hypothesis := by
    rw [decodeEdges_edgeGraph hfin]
  refine ⟨hv, rfl, fun y hy m hm => ?_⟩
  have hmT : m.toList.length ≤ C.probeBudget := by
    have : o.result.hypothesis.tree
        = (Random.runProbes o.read C Random.initialState (List.ofFn xs)).1.tree := by
      rw [ho]; rfl
    simp only [decodePop] at hm
    rw [this] at hm
    exact hmid m hm
  refine decodeReads_readGraph o.read ?_
  simp only [FreeMonoid.toList_mul, List.length_append]
  omega

/-- Where every round harvested, the rounds' history is alike the one their data hand on. -/
theorem decodeHist_alike {L N : ℕ} (C : Random.Level3Settings) (hN : L + C.probeBudget ≤ N)
    (rounds : List (SpecRound α))
    (hr : ∀ o ∈ rounds, ∃ xs : Fin C.probeBudget → FreeMonoid α,
      o.result = Random.indecisiveLStarLevel3 C o.read xs)
    (hall : ∀ o ∈ rounds, o.result.ending.harvested ≠ none) :
    HistAlike L (decodeHist (codesOf N rounds)) (specHistOf rounds) := by
  induction rounds with
  | nil => exact ⟨List.Forall₂.nil, rfl, rfl⟩
  | cons o rounds ih =>
    obtain ⟨site, hh⟩ := Option.ne_none_iff_exists'.1 (hall o (by simp))
    have hc : popCode N o = some (site, (o.result.hypothesis.tree,
        edgeGraph o.result.hypothesis.edges), readGraph N o.read, o.prefs, o.suffs) := by
      simp [popCode, hh]
    have hs : o.sampler = some ⟨o.read, o.result.hypothesis, site⟩ := by
      simp [SpecRound.sampler, hh]
    obtain ⟨xs, ho⟩ := hr o (by simp)
    obtain ⟨ih1, ih2, ih3⟩ := ih (fun o' ho' => hr o' (by simp [ho']))
      (fun o' ho' => hall o' (by simp [ho']))
    simp only [decodeHist, codesOf, specHistOf, List.filterMap_cons, hc, hs, List.map_cons,
      List.foldr_cons] at ih1 ih2 ih3 ⊢
    exact ⟨List.Forall₂.cons (decodePop_alike C hN o xs ho hc hs) ih1, by rw [ih2], by rw [ih3]⟩

/-! ## Alike samplers draw alike -/

lemma harvestFirst_congr_zone {s s' : SpecSampler α} {L : ℕ} (hs : SamplerAlike L s s')
    (k : ℕ) (T : Set (FreeMonoid α)) (xs : List (FreeMonoid α))
    (hxs : ∀ x ∈ xs, x.toList.length ≤ L) :
    Random.harvestFirst s.read k s.hyp s.site T xs
      = Random.harvestFirst s'.read k s'.hyp s'.site T xs := by
  obtain ⟨hH, hsite, hr⟩ := hs
  have hx : ∀ x ∈ xs, Random.siteHarvest s.read k s.hyp s.site x
      = Random.siteHarvest s'.read k s'.hyp s'.site x := by
    intro x hx
    rw [← hH, ← hsite]
    have hp := Random.probeOrAgree_congr_zone (r := s.read) (r' := s'.read) s.hyp.edges k
      (x := x) fun j _ hj m hm => hr _ ((by simp [prefixOf] :
        (prefixOf x j).toList.length ≤ x.toList.length).trans (hxs x hx)) m hm
    cases s.site <;> simp only [Random.siteHarvest, hp]
  unfold Random.harvestFirst
  induction xs with
  | nil => rfl
  | cons x xs ih =>
    rw [List.findSome?_cons, List.findSome?_cons, hx x (by simp),
      ih (fun y hy => hxs y (by simp [hy])) (fun y hy => hx y (by simp [hy]))]

lemma harvestLaw_alike {s s' : SpecSampler α} {L : ℕ} (hs : SamplerAlike L s s')
    (D : Measure (FreeMonoid α)) (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) :
    Random.harvestLaw s.read D k s.hyp s.site = Random.harvestLaw s'.read D k s'.hyp s'.site := by
  obtain ⟨hH, hsite, hr⟩ := hs
  rw [← hH, ← hsite]
  refine Random.harvestLaw_congr_zone D k s.hyp {x | x.toList.length ≤ L}
    (by rw [ae_iff] at hD; exact hD) (fun x hx j _ hj m hm => ?_) s.site
  exact hr _ ((by simp [prefixOf] : (prefixOf x j).toList.length ≤ x.toList.length).trans hx) m hm

variable {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}

omit [DecidableEq α] in
lemma SamplerAlike.of_forall₂ {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') {i : ℕ} (hi : i < pops.length) :
    SamplerAlike L pops[i] (pops'[i]'(hp.length_eq ▸ hi)) := by
  have := (List.forall₂_iff_get.1 hp).2 i hi (hp.length_eq ▸ hi)
  simpa using this

lemma loopDistributions_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) (j : Option (Fin K)) :
    loopDistributions D k pops j = loopDistributions D k pops' j := by
  rcases j with _ | i
  · rfl
  simp only [loopDistributions]
  have hlen := hp.length_eq
  by_cases hi : i.val < pops.length
  · rw [List.getElem?_eq_getElem hi, List.getElem?_eq_getElem (hlen ▸ hi)]
    exact congrArg popLaw (harvestLaw_alike (SamplerAlike.of_forall₂ hp hi) D hD k)
  · rw [List.getElem?_eq_none (not_lt.1 hi), List.getElem?_eq_none (hlen ▸ not_lt.1 hi)]

lemma fillDraw_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (k N : ℕ)
    (T : Finset (FreeMonoid α)) (j : Option (Fin K)) (u : unitInterval × (Fin N → unitInterval)) :
    fillDraw D k N pops T j u = fillDraw D k N pops' T j u := by
  rcases j with _ | i
  · rfl
  simp only [fillDraw]
  have hlen := hp.length_eq
  by_cases hi : i.val < pops.length
  · rw [List.getElem?_eq_getElem hi, List.getElem?_eq_getElem (hlen ▸ hi)]
    have hs := SamplerAlike.of_forall₂ hp hi
    simp only
    rw [harvestFirst_congr_zone hs k (↑T) (List.ofFn fun t => drawFrom D (u.2 t)) fun x hx => by
        obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx; exact drawFrom_length_le D hD _,
      harvestLaw_alike hs D hD k]
  · rw [List.getElem?_eq_none (not_lt.1 hi), List.getElem?_eq_none (hlen ▸ not_lt.1 hi)]

/-- Alike histories make the same filled draws. -/
theorem idealDraws_alike {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h')
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (k N : ℕ) (c : ClusterDraws Ω α K N) :
    idealDraws D Dsf k N h c = idealDraws D Dsf k N h' c := by
  obtain ⟨hp, hpr, hsu⟩ := hh
  unfold idealDraws
  rw [hpr, hsu]
  simp only [fillDraw_alike hp D hD, loopDistributions_alike hp D hD]

theorem idealRun_alike {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h')
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (k N : ℕ) (c : ClusterDraws Ω α K N) :
    idealRun D Dsf k N h c = idealRun D Dsf k N h' c := by
  unfold idealRun
  rw [idealDraws_alike hh D Dsf hD k N c]

theorem HistAlike.length {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h') :
    h.pops.length = h'.pops.length :=
  hh.1.length_eq

end OrthoDFA
