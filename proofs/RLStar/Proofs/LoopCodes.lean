import RLStar.Proofs.LoopScan
import RLStar.Proofs.LoopRuns

/-!
# A round's harvest from finite data

A harvest hands on its list and what its clustering read, data over a countable type, so a round
can be split off from the rounds before it (`decodeHist_codesOf`). Level 3's edge tables only ever
hold edges out of the tree's leaves, finitely many (`result_fin`).
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

/-! ## A round's harvest from its data -/

/-- A harvest as data: its list, and the prefixes and suffixes its round's clustering read. -/
abbrev HarvestCode (α : Type*) :=
  List (FreeMonoid α) × Finset (FreeMonoid α) × Finset (FreeMonoid α)

/-- A round's harvest. -/
def popCode (o : SpecRound α) : Option (HarvestCode α) :=
  o.population.map fun ℓ => (ℓ, o.prefs, o.suffs)

/-- The history the data hand on. -/
def decodeHist (cs : List (HarvestCode α)) : SpecHist α :=
  ⟨cs.map (·.1), cs.foldr (fun c T => c.2.1 ∪ T) ∅, cs.foldr (fun c T => c.2.2 ∪ T) ∅⟩

/-- The data of the rounds' harvests, in order. -/
def codesOf (rounds : List (SpecRound α)) : List (HarvestCode α) :=
  rounds.filterMap popCode

/-- Where every round harvested, their data hand on their history. -/
theorem decodeHist_codesOf (rounds : List (SpecRound α))
    (hall : ∀ o ∈ rounds, o.result.ending.IsHarvest) :
    decodeHist (codesOf rounds) = specHistOf rounds := by
  sorry -- closes by: induction on rounds, popCode, SpecRound.population

/-- The least `L` with `D`'s strings almost surely at most `L` long; `0` where none is. -/
noncomputable def lenBound (D : Measure (FreeMonoid α)) : ℕ :=
  sInf {L : ℕ | ∀ᵐ x ∂D, x.toList.length ≤ L}

theorem ae_lenBound {D : Measure (FreeMonoid α)} {L : ℕ} (h : ∀ᵐ x ∂D, x.toList.length ≤ L) :
    ∀ᵐ x ∂D, x.toList.length ≤ lenBound D := by
  have hne : ({L : ℕ | ∀ᵐ x ∂D, x.toList.length ≤ L} : Set ℕ).Nonempty := ⟨L, h⟩
  exact Nat.sInf_mem hne

end OrthoDFA
