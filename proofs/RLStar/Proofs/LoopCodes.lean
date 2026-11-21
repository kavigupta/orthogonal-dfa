import RLStar.Proofs.LoopScan
import RLStar.Proofs.LoopRuns

/-!
# A round's sampler from finite data

Level 3's edge tables only ever hold edges out of the tree's leaves, finitely many, and after `T`
probes its midfixes are at most `T` long (`lazyRound_hyp`), so a probe at most `L` long, padded
and walked under the round's last hypothesis, reads only strings at most `L + T` long. So the
sampler a round hands on is fixed, on probes at most `L` long, by finite data: its harvest's site,
its last hypothesis's tree and edge graph, its trie, and the round's reads of the strings at most
`N` long (`decodePop_alike`), `N` at least `lenBound D + T`, `lenBound D` the length `D`'s strings
almost surely keep within. The data range over a countable type, so a round can be split off from
the rounds before it; and alike samplers draw alike, run by run (`idealRun_alike`).
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
  · rw [Set.mem_ofPred_eq, Random.indecisiveLStarLevel3_edges] at hqc
    exact absurd he hqc
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

/-! ## The lazy round's last hypothesis -/

section LazyHyp

open Random Query

omit [DecidableEq α] in
lemma edgesFin_leaf :
    EdgesFin (fun _ : List Bool × α => (none : Option (List Bool × FreeMonoid α))) :=
  Set.finite_empty.subset fun _ h => h rfl

lemma epochQ_hyp (r : FreeMonoid α → ARU) (C : Level3Settings) (K : ℕ)
    (X : Finset (FreeMonoid α)) {P : ℕ} (xs : Fin P → FreeMonoid α) :
    EdgesFin ((epochQ C K X xs).eval r).hypothesis.edges
      ∧ MidsLe P ((epochQ C K X xs).eval r).hypothesis.tree := by
  simp only [epochQ, eval_subst, unpadQ, eval_query]
  have := (runProbes_tree (fun z => r (unpad z)) { C with walkStart := K }
    (List.ofFn fun i => padIn K X (xs i))).2.2
  rw [List.length_ofFn] at this
  exact ⟨result_fin _ _ _, (indecisiveLStarLevel3_tree (fun z => r (unpad z))
    { C with walkStart := K } (fun i => padIn K X (xs i))) ▸ this⟩

private lemma lazyEpochsQ_hyp (r : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ)
    {J b : ℕ} : ∀ (e pk : ℕ) (ds : List (EpochDraws α C.probeBudget J b))
      (X : Finset (FreeMonoid α)) (ls : LazyLists α),
      EdgesFin ((lazyEpochsQ C K nh e pk ds X ls).eval r).hypothesis.edges
        ∧ MidsLe C.probeBudget ((lazyEpochsQ C K nh e pk ds X ls).eval r).hypothesis.tree
  | _, _, [], _, _ => ⟨edgesFin_leaf, fun m hm => by
      change m ∈ mids (DTree.leaf : DTree (Option α)) at hm
      simp [mids] at hm⟩
  | e, pk, d :: ds, X, ls => by
    have h0 := epochQ_hyp r (epochSettings C e) K X d.1
    simp only [lazyEpochsQ, eval_bind]
    generalize (epochQ (epochSettings C e) K X d.1).eval r = res at h0 ⊢
    rcases res with ⟨hyp, _ | s | _⟩
    · exact h0
    · simp only [eval_bind]
      generalize (topUpQ K nh hyp s (List.ofFn d.2) X (ls s)).eval r = o
      rcases o with _ | ⟨_ | _, rest⟩
      · exact h0
      · exact h0
      · exact lazyEpochsQ_hyp r C K nh _ _ ds _ _
    · exact h0

/-- The lazy round's last hypothesis holds finitely many edges, and its midfixes are at most the
probe budget long. -/
theorem lazyRound_hyp (r : FreeMonoid α → ARU) (C : Level3Settings) (K nh : ℕ) {E J b : ℕ}
    (X₀ : Finset (FreeMonoid α)) (d : Fin E → EpochDraws α C.probeBudget J b) :
    EdgesFin ((lazyRoundQ C K nh X₀ d).eval r).hypothesis.edges
      ∧ MidsLe C.probeBudget ((lazyRoundQ C K nh X₀ d).eval r).hypothesis.tree :=
  lazyEpochsQ_hyp r C K nh _ _ _ _ _

end LazyHyp

/-! ## Pads -/

omit [DecidableEq α] in
lemma unpad_mul_eq (a b : FreeMonoid (Option α)) :
    Random.unpad (a * b) = Random.unpad a * Random.unpad b := by
  apply FreeMonoid.toList.injective
  simp [Random.unpad, List.reduceOption, List.filterMap_append]

lemma unpad_padIn_eq (K : ℕ) (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    Random.unpad (Random.padIn K X x) = x := by
  apply FreeMonoid.toList.injective
  simp [Random.unpad, Random.padIn, List.reduceOption, List.filterMap_append]

lemma length_unpad_le_length (m : FreeMonoid (Option α)) :
    (Random.unpad m).toList.length ≤ m.toList.length := by
  simpa [Random.unpad, List.reduceOption] using List.length_filterMap_le id m.toList

lemma length_unpad_prefixOf_le (z : FreeMonoid (Option α)) (j : ℕ) :
    (Random.unpad (prefixOf z j)).toList.length ≤ (Random.unpad z).toList.length := by
  have hp : (z.toList.take j).reduceOption <+: z.toList.reduceOption :=
    (List.take_prefix j _).filterMap id
  simpa [Random.unpad, prefixOf] using hp.length_le

/-! ## A round's sampler from its data -/

/-- A harvest as finite data: its site, its hypothesis's tree and edge graph, its reads, its trie,
and the prefixes and suffixes its round's clustering read. -/
abbrev HarvestCode (α : Type*) :=
  Random.Level3Site × (DTree (Option α)
      × Finset ((List Bool × Option α) × (List Bool × FreeMonoid (Option α))))
    × Finset (FreeMonoid α × ARU) × Finset (FreeMonoid α) × Finset (FreeMonoid α)
    × Finset (FreeMonoid α)

/-- A round's harvest, with its reads of the strings at most `N` long. -/
def popCode (N : ℕ) (o : SpecRound α) : Option (HarvestCode α) :=
  match o.result.ending with
  | .harvest s _ => some (s, (o.result.hypothesis.tree, edgeGraph o.result.hypothesis.edges),
      readGraph N o.read, o.result.expanded, o.prefs, o.suffs)
  | _ => none

/-- The sampler the data hand on. -/
def decodePop (c : HarvestCode α) : SpecSampler α :=
  ⟨decodeReads c.2.2.1, ⟨c.2.1.1, decodeEdges c.2.1.2⟩, c.2.2.2.1, c.1⟩

/-- The history the data hand on. -/
def decodeHist (cs : List (HarvestCode α)) : SpecHist α :=
  ⟨cs.map decodePop, cs.foldr (fun c T => c.2.2.2.2.1 ∪ T) ∅,
    cs.foldr (fun c T => c.2.2.2.2.2 ∪ T) ∅⟩

/-- The data of the rounds' harvests, in order. -/
def codesOf (N : ℕ) (rounds : List (SpecRound α)) : List (HarvestCode α) :=
  rounds.filterMap (popCode N)

/-- Two samplers alike on probes at most `L` long: the same hypothesis, trie and site, and the
same read of every string at most `L` long followed by a midfix of the hypothesis, its pads
dropped. -/
def SamplerAlike (L : ℕ) (s s' : SpecSampler α) : Prop :=
  s.hyp = s'.hyp ∧ s.trie = s'.trie ∧ s.site = s'.site
    ∧ ∀ y : FreeMonoid α, y.toList.length ≤ L → ∀ m ∈ Random.mids s.hyp.tree,
      s.read (y * Random.unpad m) = s'.read (y * Random.unpad m)

/-- Two histories alike on probes at most `L` long. -/
def HistAlike (L : ℕ) (h h' : SpecHist α) : Prop :=
  List.Forall₂ (SamplerAlike L) h.pops h'.pops ∧ h.prefs = h'.prefs ∧ h.suffs = h'.suffs

/-- A round's sampler is alike the one its data hand on, on probes at most `L` long, where its
last hypothesis holds finitely many edges and midfixes at most `T` long, and `N ≥ L + T`. -/
theorem decodePop_alike {L N T : ℕ} (hN : L + T ≤ N) (o : SpecRound α)
    (hE : EdgesFin o.result.hypothesis.edges) (hM : Random.MidsLe T o.result.hypothesis.tree)
    {c : HarvestCode α} {s : SpecSampler α} (hc : popCode N o = some c)
    (hs : o.sampler = some s) : SamplerAlike L (decodePop c) s := by
  unfold popCode at hc
  unfold SpecRound.sampler at hs
  rcases he : o.result.ending with _ | ⟨site, ℓ⟩ | _ <;> rw [he] at hc hs
  · simp at hc
  · simp only [Option.some.injEq] at hc hs
    subst hc hs
    have hv : (⟨o.result.hypothesis.tree,
        decodeEdges (edgeGraph o.result.hypothesis.edges)⟩ : HypothesisDFA (Option α))
        = o.result.hypothesis := by
      rw [decodeEdges_edgeGraph hE]
    refine ⟨hv, rfl, rfl, fun y hy m hm => ?_⟩
    have hmT : m.toList.length ≤ T := hM m hm
    refine decodeReads_readGraph o.read ?_
    simp only [FreeMonoid.toList_mul, List.length_append]
    have := length_unpad_le_length m
    omega
  · simp at hc

/-- Where every round harvested, its last hypothesis holding finitely many edges and midfixes at
most `T` long, the rounds' history is alike the one their data hand on. -/
theorem decodeHist_alike {L N T : ℕ} (hN : L + T ≤ N) (rounds : List (SpecRound α))
    (hr : ∀ o ∈ rounds, EdgesFin o.result.hypothesis.edges
      ∧ Random.MidsLe T o.result.hypothesis.tree)
    (hall : ∀ o ∈ rounds, o.result.ending.IsHarvest) :
    HistAlike L (decodeHist (codesOf N rounds)) (specHistOf rounds) := by
  induction rounds with
  | nil => exact ⟨List.Forall₂.nil, rfl, rfl⟩
  | cons o rounds ih =>
    have ho := hall o List.mem_cons_self
    obtain ⟨site, ℓ, hh⟩ : ∃ site ℓ, o.result.ending = .harvest site ℓ := by
      revert ho
      cases o.result.ending <;> simp [Random.LazyEnd.IsHarvest]
    have hc : popCode N o = some (site, (o.result.hypothesis.tree,
        edgeGraph o.result.hypothesis.edges), readGraph N o.read, o.result.expanded, o.prefs,
        o.suffs) := by
      simp [popCode, hh]
    have hs : o.sampler = some ⟨o.read, o.result.hypothesis, o.result.expanded, site⟩ := by
      simp [SpecRound.sampler, hh]
    obtain ⟨hE, hM⟩ := hr o List.mem_cons_self
    obtain ⟨ih1, ih2, ih3⟩ := ih (fun o' ho' => hr o' (List.mem_cons_of_mem _ ho'))
      (fun o' ho' => hall o' (List.mem_cons_of_mem _ ho'))
    simp only [decodeHist, codesOf, specHistOf, List.filterMap_cons, hc, hs, List.map_cons,
      List.foldr_cons] at ih1 ih2 ih3 ⊢
    exact ⟨List.Forall₂.cons (decodePop_alike hN o hE hM hc hs) ih1, by rw [ih2], by rw [ih3]⟩

/-- The least `L` with `D`'s strings almost surely at most `L` long; `0` where none is. -/
noncomputable def lenBound (D : Measure (FreeMonoid α)) : ℕ :=
  sInf {L : ℕ | ∀ᵐ x ∂D, x.toList.length ≤ L}

theorem ae_lenBound {D : Measure (FreeMonoid α)} {L : ℕ} (h : ∀ᵐ x ∂D, x.toList.length ≤ L) :
    ∀ᵐ x ∂D, x.toList.length ≤ lenBound D := by
  have hne : ({L : ℕ | ∀ᵐ x ∂D, x.toList.length ≤ L} : Set ℕ).Nonempty := ⟨L, h⟩
  exact Nat.sInf_mem hne

omit [DecidableEq α] in
/-- Every draw from a law on strings at most `L` long is at most `L` long. -/
lemma length_drawFrom_le_of_ae (ν : Measure (FreeMonoid α)) [IsFiniteMeasure ν] {L : ℕ}
    (hν : ∀ᵐ x ∂ν, x.toList.length ≤ L) (u : unitInterval) :
    (drawFrom ν u).toList.length ≤ L := by
  rcases (drawFrom_eq_iff ν u (drawFrom ν u)).1 rfl with ⟨h1, h2⟩ | ⟨h, -⟩
  · by_contra hlt
    rw [massBelow_succ] at h2
    have hs : {z : FreeMonoid α | strCode z = strCode (drawFrom ν u)} = {drawFrom ν u} := by
      ext z; simp [strCode_injective.eq_iff]
    have h0 : ν.real {drawFrom ν u} = 0 := by
      rw [measureReal_def, measure_mono_null (fun z hz => by
        rw [Set.mem_singleton_iff.1 hz]; exact hlt) (ae_iff.1 hν), ENNReal.toReal_zero]
    rw [hs, h0, add_zero] at h2
    linarith
  · rw [h]; simp

/-! ## Alike samplers draw alike -/

lemma lazyHarvestFirst_congr_zone {s s' : SpecSampler α} {L : ℕ} (hs : SamplerAlike L s s')
    (Kw : ℕ) (T : Finset (FreeMonoid α)) (xs : List (FreeMonoid α))
    (hxs : ∀ x ∈ xs, x.toList.length ≤ L) :
    Random.lazyHarvestFirst s.read Kw s.trie s.hyp s.site T xs
      = Random.lazyHarvestFirst s'.read Kw s'.trie s'.hyp s'.site T xs := by
  obtain ⟨hH, hX, hsite, hr⟩ := hs
  unfold Random.lazyHarvestFirst
  rw [← hH, ← hX, ← hsite]
  congr 1
  have hx : ∀ z ∈ xs.map (Random.padIn Kw s.trie),
      Random.siteHarvest (Random.padRead s.read) Kw s.hyp s.site z
        = Random.siteHarvest (Random.padRead s'.read) Kw s.hyp s.site z := by
    intro z hz
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hz
    have hp := Random.probeOrAgree_congr_zone (r := Random.padRead s.read)
      (r' := Random.padRead s'.read) s.hyp.edges Kw (x := Random.padIn Kw s.trie x)
      fun j _ _ m hm => by
        simp only [Random.padRead, unpad_mul_eq]
        exact hr _ ((length_unpad_prefixOf_le _ j).trans
          (by rw [unpad_padIn_eq]; exact hxs x hx)) m hm
    cases s.site <;> simp only [Random.siteHarvest, hp]
  unfold Random.harvestFirst
  generalize xs.map (Random.padIn Kw s.trie) = zs at hx
  induction zs with
  | nil => rfl
  | cons z zs ih =>
    rw [List.findSome?_cons, List.findSome?_cons, hx z List.mem_cons_self,
      ih (fun y hy => hx y (List.mem_cons_of_mem _ hy))]

lemma lazyHarvestLaw_alike {s s' : SpecSampler α} {L : ℕ} (hs : SamplerAlike L s s')
    (D : Measure (FreeMonoid α)) (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (Kw : ℕ) :
    Random.lazyHarvestLaw s.read D Kw s.trie s.hyp s.site
      = Random.lazyHarvestLaw s'.read D Kw s'.trie s'.hyp s'.site := by
  obtain ⟨hH, hX, hsite, hr⟩ := hs
  unfold Random.lazyHarvestLaw
  rw [← hH, ← hX, ← hsite]
  congr 1
  refine Random.harvestLaw_congr_zone (D.map (Random.padIn Kw s.trie)) Kw s.hyp
    {z | (Random.unpad z).toList.length ≤ L} ?_ (fun z hz j _ _ m hm => ?_) s.site
  · rw [Measure.map_apply (measurable_of_countable _) (Set.to_countable _).measurableSet]
    rw [ae_iff] at hD
    convert hD using 2
    ext x
    simp [unpad_padIn_eq]
  · simp only [Random.padRead, unpad_mul_eq]
    exact hr _ ((length_unpad_prefixOf_le z j).trans hz) m hm

variable {Ω : Type*} [MeasurableSpace Ω] {K : ℕ}

omit [DecidableEq α] in
lemma SamplerAlike.of_forall₂ {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') {i : ℕ} (hi : i < pops.length) :
    SamplerAlike L pops[i] (pops'[i]'(hp.length_eq ▸ hi)) := by
  have := (List.forall₂_iff_get.1 hp).2 i hi (hp.length_eq ▸ hi)
  simpa using this

lemma loopDistributions_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (Kw : ℕ) (j : Option (Fin K)) :
    loopDistributions D Kw pops j = loopDistributions D Kw pops' j := by
  rcases j with _ | i
  · rfl
  simp only [loopDistributions]
  have hlen := hp.length_eq
  by_cases hi : i.val < pops.length
  · rw [List.getElem?_eq_getElem hi, List.getElem?_eq_getElem (hlen ▸ hi)]
    exact congrArg popLaw (lazyHarvestLaw_alike (SamplerAlike.of_forall₂ hp hi) D hD Kw)
  · rw [List.getElem?_eq_none (not_lt.1 hi), List.getElem?_eq_none (hlen ▸ not_lt.1 hi)]

lemma popHit_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (Kw N : ℕ)
    (T : Finset (FreeMonoid α)) (j : Option (Fin K)) (u : unitInterval × (Fin N → unitInterval)) :
    popHit D Kw N pops T j u = popHit D Kw N pops' T j u := by
  rcases j with _ | i
  · rfl
  simp only [popHit]
  have hlen := hp.length_eq
  by_cases hi : i.val < pops.length
  · rw [List.getElem?_eq_getElem hi, List.getElem?_eq_getElem (hlen ▸ hi)]
    exact lazyHarvestFirst_congr_zone (SamplerAlike.of_forall₂ hp hi) Kw T _ fun x hx => by
      obtain ⟨t, rfl⟩ := List.mem_ofFn.1 hx; exact length_drawFrom_le_of_ae D hD _
  · rw [List.getElem?_eq_none (not_lt.1 hi), List.getElem?_eq_none (hlen ▸ not_lt.1 hi)]

lemma fillDraw_alike {L : ℕ} {pops pops' : List (SpecSampler α)}
    (hp : List.Forall₂ (SamplerAlike L) pops pops') (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L) (Kw N : ℕ)
    (T : Finset (FreeMonoid α)) (j : Option (Fin K)) (u : unitInterval × (Fin N → unitInterval)) :
    fillDraw D Kw N pops T j u = fillDraw D Kw N pops' T j u := by
  unfold fillDraw
  rw [popHit_alike hp D hD Kw N T j u, loopDistributions_alike hp D hD Kw j]

/-- Alike histories make the same filled draws. -/
theorem idealDraws_alike {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h')
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (Kw N : ℕ) (c : ClusterDraws Ω α K N) :
    idealDraws D Dsf Kw N h c = idealDraws D Dsf Kw N h' c := by
  obtain ⟨hp, hpr, hsu⟩ := hh
  unfold idealDraws
  rw [hpr, hsu]
  simp only [fillDraw_alike hp D hD, loopDistributions_alike hp D hD]

theorem idealRun_alike {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h')
    (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (Kw N : ℕ) (c : ClusterDraws Ω α K N) :
    idealRun D Dsf Kw N h c = idealRun D Dsf Kw N h' c := by
  unfold idealRun
  rw [idealDraws_alike hh D Dsf hD Kw N c]

omit [DecidableEq α] in
theorem HistAlike.length {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h') :
    h.pops.length = h'.pops.length :=
  hh.1.length_eq

/-- Alike histories' harvested draws hit alike. -/
lemma popsHit_alike {L : ℕ} {h h' : SpecHist α} (hh : HistAlike L h h')
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (hD : ∀ᵐ x ∂D, x.toList.length ≤ L)
    (Kw N M : ℕ) (c : ClusterRest α K N) :
    PopsHit D Kw N M h.pops h.prefs c ↔ PopsHit D Kw N M h'.pops h'.prefs c := by
  unfold PopsHit
  rw [hh.length, hh.2.1]
  simp only [popHit_alike hh.1 D hD]

end OrthoDFA
