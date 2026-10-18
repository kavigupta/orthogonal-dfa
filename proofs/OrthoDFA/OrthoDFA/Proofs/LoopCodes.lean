import OrthoDFA.Proofs.Level5Short
import OrthoDFA.Proofs.Level5Reads
import OrthoDFA.Proofs.LoopTermination

/-!
# A round's population from finite data

Level 5's edge tables only ever hold finitely many edges, and a probe at most `L` long walked
under a short tree reads only strings at most `L + maxLeaves` long. So the population a round
hands on is fixed by finite data: its harvest's site, its last hypothesis's tree and edge graph,
and the round's reads of the strings at most `N` long (`decodePop_popCode`). The data range over a
countable type, so a round can be split off from the rounds before it.
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

instance : Countable (Level5Site α) :=
  Function.Injective.countable (f := fun s : Level5Site α => match s with
    | .start => (none : Option (Option (List Bool × α)))
    | .edge e => some (some e)
    | .middles => some none) fun a b h => by
    cases a <;> cases b <;> simp_all

/-! ## Finitely many edges -/

/-- The table holds finitely many edges. -/
abbrev EdgesFin (E : EdgeTable α) : Prop := Set.Finite {qc : List Bool × α | E qc.1 qc.2 ≠ none}

omit [Fintype α] [DecidableEq α] in
theorem EdgesFin.mono {E E' : EdgeTable α} (hE : EdgesFin E) (pc : List Bool × α)
    (h : ∀ q c, E' q c ≠ none → E q c ≠ none ∨ (q, c) = pc) : EdgesFin E' := by
  refine (hE.union (Set.finite_singleton pc)).subset fun qc hqc => ?_
  rcases h qc.1 qc.2 hqc with h | h
  · exact .inl h
  · exact .inr h

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

theorem recordProbe_edgesFin {s : Level5State α} (hs : EdgesFin s.edges) (x : FreeMonoid α) :
    EdgesFin (recordProbe C cut s x).edges := by
  rcases recordProbe_spec C cut s x with h | ⟨u, c, p, t, -, -, -, -, -, h⟩ <;> rw [h]
  · exact hs
  · refine hs.mono (p, c) fun q c' h' => ?_
    rw [setEdge_eq] at h'
    split_ifs at h' with hqc
    · exact .inr hqc
    · exact .inl h'

theorem splitState_edgesFin {s : Level5State α} (hs : EdgesFin s.edges) (p : List Bool) (c : α)
    (t : List Bool) : EdgesFin (splitState cut s p c t).edges := by
  refine hs.mono (p, c) fun q c' h' => .inl fun hn => ?_
  rcases he : (splitState cut s p c t).edges q c' with _ | ⟨t', w⟩
  · exact h' he
  · obtain ⟨t₁, h₁⟩ := splitState_witness cut he
    rw [hn] at h₁
    cases h₁

theorem runProbes_fin : ∀ {s : Level5State α} {xs : List (FreeMonoid α)}, EdgesFin s.edges →
    EdgesFin (runProbes C cut s xs).1.edges
  | s, [], hs => hs
  | s, x :: xs, hs => by
    have hr := recordProbe_edgesFin C cut hs x
    simp only [runProbes]
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> dsimp only
    · refine runProbes_fin ?_
      rcases (applyProbe_inl C cut ha).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, -, rfl, -⟩
      · exact hr
      · exact splitState_edgesFin cut hr p c t
    · rcases applyProbe_inr C cut ha with ⟨-, rfl⟩ | ⟨-, p, c, t, sp, -, rfl, -⟩
      · exact hr
      · exact splitState_edgesFin cut hr p c t

/-- A round's last hypothesis holds finitely many edges. -/
theorem result_fin (read : FreeMonoid α → ARU) {T : ℕ} (draws : Fin T → FreeMonoid α) :
    EdgesFin (Function.curry (indecisiveLStarLevel5 C read draws).hypothesis.edges) :=
  runProbes_fin C (fun z => (read z).cut) (s := initialState) (xs := List.ofFn draws)
    (by simp [EdgesFin, initialState, freshState])

/-! ## Graphs -/

open scoped Classical in
/-- An edge table's graph, where it holds finitely many edges. -/
def edgeGraph (E : EdgeTable α) : Finset ((List Bool × α) × (List Bool × FreeMonoid α)) :=
  if h : EdgesFin E then h.toFinset.image fun qc => (qc, (E qc.1 qc.2).getD ([], 1)) else ∅

open scoped Classical in
/-- The table a graph draws. -/
def decodeEdges (F : Finset ((List Bool × α) × (List Bool × FreeMonoid α))) : EdgeTable α :=
  fun q c => if h : ∃ v, ((q, c), v) ∈ F then some h.choose else none

theorem decodeEdges_edgeGraph {E : EdgeTable α} (hE : EdgesFin E) :
    decodeEdges (edgeGraph E) = E := by
  have hg : edgeGraph E = hE.toFinset.image fun qc => (qc, (E qc.1 qc.2).getD ([], 1)) := by
    unfold edgeGraph; exact dif_pos hE
  rw [hg]
  funext q c
  unfold decodeEdges
  rcases he : E q c with _ | v
  · rw [dif_neg]
    rintro ⟨v, hv⟩
    obtain ⟨qc, hqc, h⟩ := Finset.mem_image.1 hv
    simp only [Prod.mk.injEq] at h
    rw [Set.Finite.mem_toFinset, Set.mem_setOf_eq, h.1, he] at hqc
    exact hqc rfl
  · have hmem : ((q, c), v) ∈ hE.toFinset.image fun qc => (qc, (E qc.1 qc.2).getD ([], 1)) := by
      refine Finset.mem_image.2 ⟨(q, c), ?_, by simp [he]⟩
      rw [Set.Finite.mem_toFinset]
      simp [he]
    have hex : ∃ v', ((q, c), v') ∈ hE.toFinset.image
        fun qc => (qc, (E qc.1 qc.2).getD ([], 1)) := ⟨v, hmem⟩
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

/-! ## A round's population from its data -/

/-- A harvest as finite data: its site, its hypothesis's tree and edge graph, and reads, and the
prefixes and suffixes its round's clustering read. -/
abbrev HarvestCode (α : Type*) :=
  Level5Site α × (DTree α × Finset ((List Bool × α) × (List Bool × FreeMonoid α)))
    × Finset (FreeMonoid α × ARU) × Finset (FreeMonoid α) × Finset (FreeMonoid α)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}

/-- A round's harvest, with its reads of the strings at most `N` long. -/
def popCode (N : ℕ) (o : LoopRound Ω α K) : Option (HarvestCode α) :=
  o.result.ending.harvested.map fun site =>
    (site, (o.result.hypothesis.tree, edgeGraph (Function.curry o.result.hypothesis.edges)),
      readGraph N o.read, o.prefs, o.suffs)

/-- The population the data hand on. -/
def decodePop (D : Measure (FreeMonoid α)) (k : ℕ) (c : HarvestCode α) :
    Measure (FreeMonoid α) :=
  popLaw (harvestLaw D (fun z => (decodeReads c.2.2.1 z).cut) k
    ⟨c.2.1.1, Function.uncurry (decodeEdges c.2.1.2)⟩ c.1)

/-- The history the data hand on. -/
def decodeHist (D : Measure (FreeMonoid α)) (k : ℕ) (cs : List (HarvestCode α)) : LoopHist α :=
  ⟨cs.map (decodePop D k), cs.foldr (fun c T => c.2.2.2.1 ∪ T) ∅,
    cs.foldr (fun c T => c.2.2.2.2 ∪ T) ∅⟩

omit [MeasurableSpace Ω] in
/-- Over draws at most `L` long, reads that agree on the strings at most `L + M` long find the
same, under a hypothesis whose midfixes are at most `M` long. -/
theorem harvestLaw_congr_up {D : Measure (FreeMonoid α)} {L M N : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hN : L + M ≤ N)
    {cut' : FreeMonoid α → Option Bool} (hcut : AgreeUpTo N cut cut') (k : ℕ)
    {H : HypothesisDFA α} (hT : MidLe M H.tree) (site : Level5Site α) :
    harvestLaw D cut k H site = harvestLaw D cut' k H site := by
  unfold harvestLaw
  ext A hA
  rw [Measure.bind_apply hA (measurable_of_countable _).aemeasurable,
    Measure.bind_apply hA (measurable_of_countable _).aemeasurable]
  refine lintegral_congr_ae ?_
  filter_upwards [hlen] with x hx
  have hxN : x.toList.length + M ≤ N := by omega
  have : siteHarvest cut k site (H.tree, Function.curry H.edges) x
      = siteHarvest cut' k site (H.tree, Function.curry H.edges) x := by
    cases site with
    | start => exact startHarvestBy_congr_up hcut hT hxN k
    | edge e =>
      show edgeHarvestBy cut H.tree _ k x e = edgeHarvestBy cut' H.tree _ k x e
      rw [edgeHarvestBy_congr_pref (prefStrings_agree_up hcut hT hxN)]
    | middles => exact middleHarvestBy_congrK (prefAgreeK_up hcut hT hxN k)
  rw [this]

omit [MeasurableSpace Ω] in
/-- A round's population is the one its data hand on, where its probes and replays are at most
`L` long and `N ≥ L + maxLeaves`. -/
theorem decodePop_popCode {D : Measure (FreeMonoid α)} {L N : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hC : 2 ≤ C.maxLeaves) (hN : L + C.maxLeaves ≤ N)
    (k : ℕ) (o : LoopRound Ω α K) {T : ℕ} (xs : Fin T → FreeMonoid α)
    (ho : o.result = indecisiveLStarLevel5 C o.read xs) :
    (popCode N o).map (decodePop D k) = o.population D k := by
  unfold popCode LoopRound.population
  rcases hh : o.result.ending.harvested with _ | site
  · rfl
  have hend : (indecisiveLStarLevel5 C o.read xs).ending = .harvest site := by
    rw [← ho]
    revert hh
    cases o.result.ending <;> simp [RoundEnd.harvested]
  have hfin := result_fin C o.read xs
  have hshort := harvest_short C hC o.read xs hend
  rw [← ho] at hfin hshort
  simp only [Option.map_some, decodePop]
  congr 1
  have hv : (⟨o.result.hypothesis.tree, Function.uncurry
      (decodeEdges (edgeGraph (Function.curry o.result.hypothesis.edges)))⟩ : HypothesisDFA α)
      = o.result.hypothesis := by
    rw [decodeEdges_edgeGraph hfin, Function.uncurry_curry]
  rw [hv]
  refine congrArg popLaw (harvestLaw_congr_up _ hlen hN (fun z hz => ?_) k
    (fun m hm => ?_) site)
  · show (decodeReads (readGraph N o.read) z).cut = (o.read z).cut
    rw [decodeReads_readGraph o.read hz]
  · obtain ⟨-, hL, hmid⟩ := hshort
    have := hmid m hm
    omega

/-- The data of the rounds' harvests, in order. -/
def codesOf (N : ℕ) (rounds : List (LoopRound Ω α K)) : List (HarvestCode α) :=
  rounds.filterMap (popCode N)

omit [MeasurableSpace Ω] in
theorem codesOf_decode {D : Measure (FreeMonoid α)} {L N : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hC : 2 ≤ C.maxLeaves) (hN : L + C.maxLeaves ≤ N)
    (k : ℕ) (rounds : List (LoopRound Ω α K))
    (hr : ∀ o ∈ rounds, ∃ T, ∃ xs : Fin T → FreeMonoid α,
      o.result = indecisiveLStarLevel5 C o.read xs) :
    (codesOf N rounds).map (decodePop D k) = harvestsOf D k rounds := by
  induction rounds with
  | nil => rfl
  | cons o rounds ih =>
    obtain ⟨T, xs, ho⟩ := hr o (by simp)
    have h1 := decodePop_popCode C hlen hC hN k o xs ho
    have h2 := ih fun o' ho' => hr o' (by simp [ho'])
    unfold codesOf harvestsOf at h2 ⊢
    rw [List.filterMap_cons, List.filterMap_cons]
    rcases hp : popCode N o with _ | c <;> rw [hp] at h1 <;> simp only [Option.map] at h1
    · rw [← h1]; exact h2
    · rw [← h1, List.map_cons, h2]

omit [MeasurableSpace Ω] in
/-- Where every round harvested, the data hand on the rounds' history. -/
theorem decodeHist_codesOf {D : Measure (FreeMonoid α)} {L N : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hC : 2 ≤ C.maxLeaves) (hN : L + C.maxLeaves ≤ N)
    (k : ℕ) (rounds : List (LoopRound Ω α K))
    (hr : ∀ o ∈ rounds, ∃ T, ∃ xs : Fin T → FreeMonoid α,
      o.result = indecisiveLStarLevel5 C o.read xs)
    (hall : ∀ o ∈ rounds, o.result.ending.harvested ≠ none) :
    decodeHist D k (codesOf N rounds) = histOf D k rounds := by
  have hp := codesOf_decode C hlen hC hN k rounds hr
  unfold decodeHist histOf
  congr 1
  · induction rounds with
    | nil => rfl
    | cons o rounds ih =>
      obtain ⟨h, hh⟩ := Option.ne_none_iff_exists'.1 (hall o (by simp))
      have hc : popCode N o = some (h, (o.result.hypothesis.tree,
          edgeGraph (Function.curry o.result.hypothesis.edges)),
          readGraph N o.read, o.prefs, o.suffs) := by simp [popCode, hh]
      simp only [codesOf, List.filterMap_cons, hc, List.foldr_cons]
      congr 1
      exact ih (fun o' ho' => hr o' (by simp [ho'])) (fun o' ho' => hall o' (by simp [ho']))
        (codesOf_decode C hlen hC hN k rounds fun o' ho' => hr o' (by simp [ho']))
  · induction rounds with
    | nil => rfl
    | cons o rounds ih =>
      obtain ⟨h, hh⟩ := Option.ne_none_iff_exists'.1 (hall o (by simp))
      have hc : popCode N o = some (h, (o.result.hypothesis.tree,
          edgeGraph (Function.curry o.result.hypothesis.edges)),
          readGraph N o.read, o.prefs, o.suffs) := by simp [popCode, hh]
      simp only [codesOf, List.filterMap_cons, hc, List.foldr_cons]
      congr 1
      exact ih (fun o' ho' => hr o' (by simp [ho'])) (fun o' ho' => hall o' (by simp [ho']))
        (codesOf_decode C hlen hC hN k rounds fun o' ho' => hr o' (by simp [ho']))

end OrthoDFA
