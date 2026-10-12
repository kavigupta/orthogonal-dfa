import OrthoDFA.Proofs.Level4Short
import OrthoDFA.Proofs.Level4Reads
import OrthoDFA.Proofs.LoopTermination

/-!
# A round's population from finite data

Level 4's edge tables only ever hold finitely many edges, and a probe at most `L` long walked
under a short tree reads only strings at most `L + maxLeaves` long. So the population a round
hands on is fixed by finite data: its harvest's site, the hypotheses' trees and edge graphs, and
the round's reads of the strings at most `N` long (`decodePop_popCode`). The data range over a
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

instance : Countable (Level4Site α) :=
  Function.Injective.countable (f := fun s : Level4Site α => match s with
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

variable (C : Level4Settings) (cut : FreeMonoid α → Option Bool)

theorem fixEdge_edges (m : ℕ) (s : Level4State α) (p : List Bool) (c : α) (t : List Bool)
    (q : List Bool) (e : α) (h : (fixEdge cut m s p c t).edges q e ≠ none) :
    s.edges q e ≠ none ∨ (q, e) = (p, c) := by
  unfold fixEdge at h
  simp only [] at h
  split at h
  · split_ifs at h
    · dsimp only at h
      split_ifs at h with hpq
      · exact absurd rfl h
      · left
        unfold retargetBy at h
        rcases hs : s.edges q e with _ | ⟨s0, w⟩
        · rw [hs] at h; exact absurd rfl h
        · simp
    · rw [setEdge_eq] at h
      split_ifs at h with hqc
      · exact .inr hqc
      · exact .inl h
  · rw [setEdge_eq] at h
    split_ifs at h with hqc
    · exact .inr hqc
    · exact .inl h

theorem recordProbe_edgesFin {s : Level4State α} (hs : EdgesFin s.edges) (x : FreeMonoid α) :
    EdgesFin (recordProbe C cut s x).edges := by
  rcases (recordProbe_cases cut C s x).2 with ⟨-, he, -⟩ | ⟨-, -, -, p, c, e, -, -, he⟩
  · rw [he]; exact hs
  · refine hs.mono (p, c) fun q c' h => ?_
    rw [he q c'] at h
    split_ifs at h with hqc
    · exact .inr hqc
    · exact .inl h

theorem applyProbe_edgesFin {s s' : Level4State α} {x : FreeMonoid α}
    (hs : EdgesFin (recordProbe C cut s x).edges) (ha : applyProbe C cut s x = .inl s') :
    EdgesFin s'.edges := by
  unfold applyProbe at ha
  simp only [] at ha
  split at ha
  · cases ha
  unfold settleOne at ha
  split_ifs at ha with hv
  · simp only [Sum.inl.injEq] at ha
    subst ha
    exact hs.mono _ fun q e h => fixEdge_edges cut _ _ _ _ _ q e h
  · simp only [Sum.inl.injEq] at ha
    subst ha
    exact hs

/-- The table, and every hypothesis kept, hold finitely many edges. -/
def FinInv (s : Level4State α) : Prop := EdgesFin s.edges ∧ ∀ v ∈ s.versions, EdgesFin v.2

theorem runProbes_fin {s : Level4State α} {xs : List (FreeMonoid α)} {h : Level4Harvest α}
    (hs : FinInv s) (hz : (runProbes C cut s xs).2 = .harvest h) :
    ∀ v ∈ h.versions, EdgesFin v.2 := by
  induction xs generalizing s with
  | nil => simp [runProbes] at hz
  | cons x xs ih =>
    have hr : FinInv (recordProbe C cut s x) := by
      refine ⟨recordProbe_edgesFin C cut hs.1 x, fun v hv => ?_⟩
      rw [recordProbe_versions] at hv
      rcases List.mem_append.1 hv with hv | hv
      · exact hs.2 v hv
      · rw [List.mem_singleton] at hv
        subst hv
        exact hs.1
    simp only [runProbes] at hz
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at hz
    · exact ih ⟨applyProbe_edgesFin C cut hr.1 ha, by
        rw [applyProbe_versions C cut ha]; exact hr.2⟩ hz
    · have he : e ≠ .tooBig := by rintro rfl; simp [toRoundEnd] at hz
      obtain ⟨rfl, -⟩ := applyProbe_inr_pre C cut ha he
      cases e with
      | success => simp [toRoundEnd] at hz
      | tooBig => simp [toRoundEnd] at hz
      | harvestStart =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hr.2
      | harvestPT =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]
        intro v hv
        rw [List.mem_singleton] at hv
        subst hv
        exact hr.1
      | harvest e =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hr.2

/-- Every hypothesis a harvest kept holds finitely many edges. -/
theorem harvest_fin (read : FreeMonoid α → ARU) {T : ℕ} (draws : Fin T → FreeMonoid α)
    {h : Level4Harvest α} (hh : (indecisiveLStarLevel4 C read draws).ending = .harvest h) :
    ∀ v ∈ h.versions, EdgesFin v.2 :=
  runProbes_fin C (fun z => (read z).cut) (s := initialState) (xs := List.ofFn draws)
    ⟨by simp [EdgesFin, initialState], by simp [initialState]⟩ hh

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

/-- A harvest as finite data: its site, its hypotheses' trees and edge graphs, and reads. -/
abbrev HarvestCode (α : Type*) :=
  Level4Site α × List (DTree α × Finset ((List Bool × α) × (List Bool × FreeMonoid α)))
    × Finset (FreeMonoid α × ARU)

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {K : ℕ}

/-- A round's harvest, with its reads of the strings at most `N` long. -/
def popCode (N : ℕ) (o : LoopRound Ω α K) : Option (HarvestCode α) :=
  o.result.ending.harvested.map fun h =>
    (h.site, h.versions.map fun v => (v.1, edgeGraph v.2), readGraph N o.read)

/-- The population the data hand on. -/
def decodePop (D : Measure (FreeMonoid α)) (k : ℕ) (c : HarvestCode α) :
    Measure (FreeMonoid α) :=
  popLaw (harvestLaw D (fun z => (decodeReads c.2.2 z).cut) k
    ⟨c.1, c.2.1.map fun v => (v.1, decodeEdges v.2)⟩)

omit [MeasurableSpace Ω] in
/-- Over draws at most `L` long, reads that agree on the strings at most `L + M` long find the
same, under hypotheses whose midfixes are at most `M` long. -/
theorem harvestLaw_congr_up {D : Measure (FreeMonoid α)} {L M N : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hN : L + M ≤ N)
    {cut' : FreeMonoid α → Option Bool} (hcut : AgreeUpTo N cut cut') (k : ℕ)
    {h : Level4Harvest α} (hmid : ∀ v ∈ h.versions, MidLe M v.1) :
    harvestLaw D cut k h = harvestLaw D cut' k h := by
  unfold harvestLaw
  congr 2
  refine List.map_congr_left fun v hv => ?_
  ext A hA
  rw [Measure.bind_apply hA (measurable_of_countable _).aemeasurable,
    Measure.bind_apply hA (measurable_of_countable _).aemeasurable]
  refine lintegral_congr_ae ?_
  filter_upwards [hlen] with x hx
  have hxN : x.toList.length + M ≤ N := by omega
  have hT := hmid v hv
  have : siteHarvest cut k h.site v x = siteHarvest cut' k h.site v x := by
    cases h.site with
    | start => exact startHarvestBy_congr_up hcut hT hxN k
    | edge e =>
      show edgeHarvestBy cut v.1 v.2 k x e = edgeHarvestBy cut' v.1 v.2 k x e
      rw [edgeHarvestBy_congr_pref (prefStrings_agree_up hcut hT hxN)]
    | middles => exact middleHarvestBy_congrK (prefAgreeK_up hcut hT hxN k)
  rw [this]

omit [MeasurableSpace Ω] in
/-- A round's population is the one its data hand on, where its probes and replays are at most
`L` long and `N ≥ L + maxLeaves`. -/
theorem decodePop_popCode {D : Measure (FreeMonoid α)} {L N : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (hC : 2 ≤ C.maxLeaves) (hN : L + C.maxLeaves ≤ N)
    (k : ℕ) (o : LoopRound Ω α K) {T : ℕ} (xs : Fin T → FreeMonoid α)
    (ho : o.result = indecisiveLStarLevel4 C o.read xs) :
    (popCode N o).map (decodePop D k) = o.population D k := by
  unfold popCode LoopRound.population
  rcases hh : o.result.ending.harvested with _ | h
  · rfl
  have hend : (indecisiveLStarLevel4 C o.read xs).ending = .harvest h := by
    rw [← ho]
    revert hh
    cases o.result.ending <;> simp [RoundEnd.harvested]
  have hfin := harvest_fin C o.read xs hend
  have hshort := harvest_short C hC o.read xs hend
  simp only [Option.map_some, decodePop, List.map_map, Function.comp_def]
  congr 1
  have hv : (h.versions.map fun v => (v.1, decodeEdges (edgeGraph v.2))) = h.versions :=
    (List.map_congr_left fun v hv => by rw [decodeEdges_edgeGraph (hfin v hv)]).trans
      (List.map_id' _)
  rw [hv]
  refine congrArg popLaw (harvestLaw_congr_up _ hlen hN (fun z hz => ?_) k
    (h := ⟨h.site, h.versions⟩) fun v hvm m hm => ?_)
  · show (decodeReads (readGraph N o.read) z).cut = (o.read z).cut
    rw [decodeReads_readGraph o.read hz]
  · obtain ⟨-, hL, hmid⟩ := hshort v hvm
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
      o.result = indecisiveLStarLevel4 C o.read xs) :
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

end OrthoDFA
