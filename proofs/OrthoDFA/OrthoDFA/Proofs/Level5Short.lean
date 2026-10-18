import OrthoDFA.Proofs.Level5RiskBound

/-!
# A harvest's strings are short

A split adds a midfix one letter longer than one already there, and a leaf, so a midfix is two
shorter than the leaves count, which a round that does not end too big keeps within `maxLeaves`.
A string a probe harvests is its prefix followed by a midfix, so it is at most
`L + maxLeaves − 2` long where the probe is at most `L` long (`siteHarvest_short`), and so is
every string a harvest's sampler finds over draws that short (`harvestLaw_short`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A node's midfix is at most the leaves less two letters long. -/
def ShortMids (T : DTree α) : Prop := ∀ m ∈ T.midfixes, m.toList.length + 2 ≤ T.leaves.length

theorem midfixAt_mem : ∀ (T : DTree α) (p : List Bool), T.midfixAt p = 1 ∨ T.midfixAt p ∈ T.midfixes
  | .leaf, _ => .inl rfl
  | .node m r a, [] => .inr (by simp [DTree.midfixAt, DTree.midfixes])
  | .node m r a, false :: p => by
    rcases midfixAt_mem r p with h | h
    · exact .inl h
    · exact .inr (by simp [DTree.midfixAt, DTree.midfixes, h])
  | .node m r a, true :: p => by
    rcases midfixAt_mem a p with h | h
    · exact .inl h
    · exact .inr (by simp [DTree.midfixAt, DTree.midfixes, h])

theorem shortMids_start : ShortMids (.node 1 .leaf .leaf : DTree α) := by
  intro m hm
  simp [DTree.midfixes] at hm
  subst hm
  simp [DTree.leaves]

theorem shortMids_split {T : DTree α} (hT : ShortMids T) (h2 : 2 ≤ T.leaves.length)
    {p : List Bool} (hp : p ∈ T.leaves) (c : α) (q : List Bool) :
    ShortMids (T.splitAt (FreeMonoid.of c * T.midfixAt q) p) := by
  intro m hm
  rw [DTree.length_leaves_splitAt _ _ _ hp]
  rcases Finset.mem_insert.1 (midfixes_splitAt _ T p hm) with rfl | hm
  · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
      List.length_singleton]
    rcases midfixAt_mem T q with h | h
    · rw [h]; simp; omega
    · have := hT _ h; omega
  · have := hT m hm; omega

omit [Fintype α] [DecidableEq α] in
theorem length_prefixOf_le (x : FreeMonoid α) (j : ℕ) :
    (prefixOf x j).toList.length ≤ x.toList.length := by
  simp [prefixOf]

/-- A string a sift of a probe's prefix reads is short, where the tree's midfixes are. -/
theorem route_short {cut : FreeMonoid α → Option Bool} {T : DTree α} {Lmax L : ℕ}
    (hT : ∀ m ∈ T.midfixes, m.toList.length + 2 ≤ T.leaves.length)
    (hTL : T.leaves.length ≤ Lmax) {x : FreeMonoid α} (hx : x.toList.length ≤ L) {j : ℕ}
    {z : FreeMonoid α} (hz : z ∈ (T.siftWithPath cut (prefixOf x j)).1) :
    z.toList.length + 2 ≤ L + Lmax := by
  obtain ⟨m, hm, rfl⟩ := reads_mid cut T _ z hz
  have h1 := hT m hm
  have h2 := length_prefixOf_le x j
  simp only [FreeMonoid.toList_mul, List.length_append]
  omega

/-- The tree's leaves within `maxLeaves`, and its midfixes two shorter than its leaves. -/
def TreeShort (C : Level5Settings) (T : DTree α) : Prop :=
  2 ≤ T.leaves.length ∧ T.leaves.length ≤ C.maxLeaves
    ∧ ∀ m ∈ T.midfixes, m.toList.length + 2 ≤ T.leaves.length

variable (C : Level5Settings) (cut : FreeMonoid α → Option Bool)

/-- A string a probe of at most `L` letters harvests under a short tree is at most
`L + maxLeaves − 2` long. -/
theorem siteHarvest_short {L k : ℕ} (site : Level5Site α) {v : DTree α × Edges α}
    (hv : TreeShort C v.1) {x : FreeMonoid α} (hx : x.toList.length ≤ L) :
    ∀ z ∈ siteHarvest cut k site v x, z.toList.length + 2 ≤ L + C.maxLeaves := by
  obtain ⟨-, hL, hm⟩ := hv
  have hr : ∀ {j z}, z ∈ (v.1.siftWithPath cut (prefixOf x j)).1 →
      z.toList.length + 2 ≤ L + C.maxLeaves := fun hz => route_short hm hL hx hz
  intro z hz
  cases site with
  | start =>
    change z ∈ startHarvestBy cut v.1 k x at hz
    unfold startHarvestBy at hz
    split_ifs at hz
    · simp at hz
    rcases hb : v.1.sift (ofCut cut) (prefixOf x k) with q | b <;> rw [hb] at hz
    · simp at hz
    · simp only [Sum.elim_inr, List.mem_singleton] at hz
      subst hz
      exact hr (sift_inr_reads cut _ _ _ hb).1
  | edge e =>
    change z ∈ edgeHarvestBy cut v.1 v.2 k x e at hz
    unfold edgeHarvestBy at hz
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hz
    exact hr (firstsBy_pos cut _ _ _ _ (List.mem_of_mem_filter hq)).1
  | middles =>
    change z ∈ middleHarvestBy cut v.1 v.2 k x at hz
    unfold middleHarvestBy at hz
    rcases ho : probeOrAgree (ofCut cut) v.1 v.2 k x with _ | _ | _ | _ | ⟨zs, _ | _⟩ <;>
      rw [ho] at hz <;> try simp at hz
    obtain ⟨j, b, -, -, hb, rfl⟩ := probe_middle _ v.1 v.2 k x ho
    simp only [List.mem_singleton] at hz
    subst hz
    exact hr (sift_inr_reads cut _ _ _ hb).1

theorem treeShort_step {s s' : Level5State α} {x : FreeMonoid α} (h : TreeShort C s.tree)
    (ha : applyProbe C cut s x = .inl s') : TreeShort C s'.tree := by
  rcases (applyProbe_inl C cut ha).2 with ⟨-, rfl⟩ | ⟨p, c, t, sp, hrb, rfl, hL'⟩
  · rw [recordProbe_tree]; exact h
  obtain ⟨h2, -, hm⟩ := h
  obtain ⟨-, -, -, -, t₀, w, hpe, -⟩ := recordBy_spec cut hrb
  obtain ⟨fd, -, -, -, -, hsp, -⟩ := recordBy_parts cut hrb
  have hp := DTree.sift_inl_mem _ _ _ _ hsp
  have hpe' : (recordProbe C cut s x).edges (p, c) = some (t₀, w) := by
    rcases recordProbe_spec C cut s x with h1 | ⟨u, c', p', t', -, -, -, -, hn, h1⟩
    · rw [h1]; exact hpe
    · rw [h1, setEdge_eq]
      split_ifs with hq
      · simp only [Prod.mk.injEq] at hq
        obtain ⟨rfl, rfl⟩ := hq
        rw [hn] at hpe; cases hpe
      · exact hpe
  have htr := recordProbe_tree C cut s x
  refine ⟨?_, hL', ?_⟩ <;> rw [splitState_of_some cut _ hpe'] <;> simp only [freshState] <;>
    rw [htr]
  · rw [DTree.length_leaves_splitAt _ _ _ hp]; omega
  · exact shortMids_split hm h2 hp c _

theorem runProbes_short {s : Level5State α} {xs : List (FreeMonoid α)} {site : Level5Site α}
    (hs : TreeShort C s.tree) (hz : (runProbes C cut s xs).2 = .harvest site) :
    TreeShort C (runProbes C cut s xs).1.tree := by
  induction xs generalizing s with
  | nil => simp [runProbes] at hz
  | cons x xs ih =>
    simp only [runProbes] at hz ⊢
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at hz <;> dsimp only at hz ⊢
    · exact ih (treeShort_step C cut hs ha) hz
    · rcases applyProbe_inr C cut ha with ⟨-, rfl⟩ | ⟨rfl, -⟩
      · rw [recordProbe_tree]; exact hs
      · simp [toRoundEnd] at hz

/-- A harvest's hypothesis is short. -/
theorem harvest_short (hC : 2 ≤ C.maxLeaves) (read : FreeMonoid α → ARU) {T : ℕ}
    (draws : Fin T → FreeMonoid α) {site : Level5Site α}
    (hh : (indecisiveLStarLevel5 C read draws).ending = .harvest site) :
    TreeShort C (indecisiveLStarLevel5 C read draws).hypothesis.tree := by
  refine runProbes_short C (fun z => (read z).cut) (s := initialState) (xs := List.ofFn draws)
    ⟨?_, ?_, ?_⟩ hh
  · simp [initialState, freshState, DTree.leaves]
  · simpa [initialState, freshState, DTree.leaves] using hC
  · exact shortMids_start

open MeasureTheory in
open scoped Classical in
/-- Over draws at most `L` long, a harvest under a short hypothesis finds no string longer than
`L + maxLeaves − 2`. -/
theorem harvestLaw_short {D : Measure (FreeMonoid α)} {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) {H : HypothesisDFA α} (site : Level5Site α)
    (hv : TreeShort C H.tree) :
    harvestLaw D cut k H site {z | L + C.maxLeaves < z.toList.length + 2} = 0 := by
  rw [harvestLaw_apply]
  refine lintegral_eq_zero_of_ae_eq_zero ?_
  filter_upwards [hlen] with x hx
  simp
  exact fun a ha => siteHarvest_short C cut site hv hx a ha

end OrthoDFA
