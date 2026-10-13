import OrthoDFA.Proofs.Level4RiskBound

/-!
# A harvest's strings are short

A split adds a midfix one letter longer than one already there, and a leaf, so a midfix is two
shorter than the leaves count, which the round keeps within `maxLeaves`, at every hypothesis it
keeps. A string a probe harvests is its prefix followed by a midfix, so it is at most
`L + maxLeaves − 2` long where the probe is at most `L` long (`siteHarvest_short`), and so is
every string a harvest's sampler finds over draws that short (`harvestLaw_short`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem midfixAt_mem_or_one : ∀ (T : DTree α) (p : List Bool), T.midfixAt p ∈ T.midfixes ∨ T.midfixAt p = 1
  | .leaf, _ => .inr rfl
  | .node m r a, [] => .inl (by simp [DTree.midfixAt, DTree.midfixes])
  | .node m r a, false :: p => by
    rcases midfixAt_mem_or_one r p with h | h
    · left; simp [DTree.midfixAt, DTree.midfixes, h]
    · right; simpa [DTree.midfixAt] using h
  | .node m r a, true :: p => by
    rcases midfixAt_mem_or_one a p with h | h
    · left; simp [DTree.midfixAt, DTree.midfixes, h]
    · right; simpa [DTree.midfixAt] using h

theorem midfixes_splitAt (d : FreeMonoid α) : ∀ (T : DTree α) (p : List Bool) (m : FreeMonoid α),
    m ∈ (T.splitAt d p).midfixes → m ∈ T.midfixes ∨ m = d
  | .leaf, [], m, h => by simp [DTree.splitAt, DTree.midfixes] at h; exact .inr h
  | .leaf, _ :: _, m, h => .inl (by simpa [DTree.splitAt] using h)
  | .node n r a, [], m, h => .inl (by simpa [DTree.splitAt] using h)
  | .node n r a, false :: p, m, h => by
    simp only [DTree.splitAt, DTree.midfixes, Finset.mem_insert, Finset.mem_union] at h
    rcases h with h | h | h
    · left; simp [DTree.midfixes, h]
    · rcases midfixes_splitAt d r p m h with h' | h'
      · left; simp [DTree.midfixes, h']
      · exact .inr h'
    · left; simp [DTree.midfixes, h]
  | .node n r a, true :: p, m, h => by
    simp only [DTree.splitAt, DTree.midfixes, Finset.mem_insert, Finset.mem_union] at h
    rcases h with h | h | h
    · left; simp [DTree.midfixes, h]
    · left; simp [DTree.midfixes, h]
    · rcases midfixes_splitAt d a p m h with h' | h'
      · left; simp [DTree.midfixes, h']
      · exact .inr h'

omit [Fintype α] [DecidableEq α] in
theorem length_prefixOf_le (x : FreeMonoid α) (j : ℕ) :
    (prefixOf x j).toList.length ≤ x.toList.length := by
  simp [prefixOf]

/-- A string a sift of a probe's prefix reads is short, where the tree's midfixes are. -/
theorem route_short {cut : FreeMonoid α → Option Bool} {T : DTree α} {Lmax L : ℕ}
    (hT : ∀ m ∈ T.midfixes, m.toList.length + 2 ≤ T.leaves.length)
    (hTL : T.leaves.length ≤ Lmax) {x : FreeMonoid α} (hx : x.toList.length ≤ L) {j : ℕ}
    {z : FreeMonoid α} (hz : z ∈ (T.route cut (prefixOf x j)).1) :
    z.toList.length + 2 ≤ L + Lmax := by
  obtain ⟨m, hm, rfl⟩ := route_mid cut T _ z hz
  have h1 := hT m hm
  have h2 := length_prefixOf_le x j
  simp only [FreeMonoid.toList_mul, List.length_append]
  omega

/-- The tree's leaves within `maxLeaves`, and its midfixes two shorter than its leaves. -/
def TreeShort (C : Level4Settings) (T : DTree α) : Prop :=
  2 ≤ T.leaves.length ∧ T.leaves.length ≤ C.maxLeaves
    ∧ ∀ m ∈ T.midfixes, m.toList.length + 2 ≤ T.leaves.length

/-- The tree, and every hypothesis the round kept, short. -/
def ShortInv (C : Level4Settings) (s : Level4State α) : Prop :=
  TreeShort C s.tree ∧ ∀ v ∈ s.versions, TreeShort C v.1

variable (C : Level4Settings) (cut : FreeMonoid α → Option Bool)

/-- A string a probe of at most `L` letters harvests under a short tree is at most
`L + maxLeaves − 2` long. -/
theorem siteHarvest_short {L k : ℕ} (site : Level4Site α) {v : DTree α × EdgeTable α}
    (hv : TreeShort C v.1) {x : FreeMonoid α} (hx : x.toList.length ≤ L) :
    ∀ z ∈ siteHarvest cut k site v x, z.toList.length + 2 ≤ L + C.maxLeaves := by
  obtain ⟨-, hL, hm⟩ := hv
  have hr : ∀ {j z}, z ∈ (v.1.route cut (prefixOf x j)).1 →
      z.toList.length + 2 ≤ L + C.maxLeaves := fun hz => route_short hm hL hx hz
  intro z hz
  cases site with
  | start =>
    change z ∈ startHarvestBy cut v.1 k x at hz
    unfold startHarvestBy at hz
    split_ifs at hz
    · simp at hz
    rcases hb : v.1.siftCut cut (prefixOf x k) with q | b <;> rw [hb] at hz
    · simp at hz
    · simp only [Sum.elim_inr, List.mem_singleton] at hz
      subst hz
      exact hr (route_undecided cut _ _ _ hb).1
  | edge e =>
    change z ∈ edgeHarvestBy cut v.1 v.2 k x e at hz
    unfold edgeHarvestBy at hz
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hz
    exact hr (firstsBy_pos cut _ _ _ _ (List.mem_of_mem_filter hq)).1
  | middles =>
    change z ∈ middleHarvestBy cut v.1 v.2 k x at hz
    unfold middleHarvestBy at hz
    rcases ho : probeBy cut v.1 v.2 k x with _ | _ | _ | j | _ | j | _ <;>
      rw [ho] at hz <;> try simp at hz
    all_goals
      rcases hb : v.1.siftCut cut (prefixOf x j) with q | b <;> rw [hb] at hz
      · simp at hz
      · simp only [Sum.elim_inr, List.mem_singleton] at hz
        subst hz
        exact hr (route_undecided cut _ _ _ hb).1

theorem recordProbe_short {s : Level4State α} {x : FreeMonoid α} (h : ShortInv C s) :
    ShortInv C (recordProbe C cut s x) := by
  have htree := (recordProbe_cases cut C s x).1
  refine ⟨htree ▸ h.1, fun v hv => ?_⟩
  rw [recordProbe_versions] at hv
  rcases List.mem_append.1 hv with hv | hv
  · exact h.2 v hv
  · rw [List.mem_singleton] at hv
    subst hv
    exact h.1

theorem applyProbe_short {s s' : Level4State α} {x : FreeMonoid α}
    (h : ShortInv C (recordProbe C cut s x)) (ha : applyProbe C cut s x = .inl s') :
    ShortInv C s' := by
  have hvers := applyProbe_versions C cut ha
  obtain ⟨⟨h2, hL, hm⟩, hv⟩ := h
  refine ⟨?_, hvers ▸ hv⟩
  -- the tree: kept, or split within `maxLeaves`
  unfold applyProbe at ha
  simp only [] at ha
  split at ha
  · cases ha
  unfold settleOne at ha
  split_ifs at ha with hv' hbig
  · simp only [Sum.inl.injEq] at ha
    subst ha
    refine ⟨?_, not_lt.1 hbig, ?_⟩
    · rcases (fixEdge_cases cut C hv'.choose_spec.choose_spec.choose_spec).2 with ⟨ht, -⟩ |
        ⟨t₀, w₀, -, -, ht⟩
      · rw [ht]; exact h2
      · rw [ht, DTree.splitAt_paths_length _ _ _ hv'.choose_spec.choose_spec.choose_spec.1]
        omega
    · rcases (fixEdge_cases cut C hv'.choose_spec.choose_spec.choose_spec).2 with ⟨ht, -⟩ |
        ⟨t₀, w₀, -, -, ht⟩
      · rw [ht]; exact hm
      · rw [ht, DTree.splitAt_paths_length _ _ _ hv'.choose_spec.choose_spec.choose_spec.1]
        intro m hmm
        rcases midfixes_splitAt _ _ _ m hmm with hm' | rfl
        · have := hm m hm'; omega
        · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
            List.length_singleton]
          rcases midfixAt_mem_or_one (recordProbe C cut s x).tree
              (commonPrefix hv'.choose_spec.choose_spec.choose t₀)
            with h' | h'
          · have := hm _ h'; omega
          · rw [h']; simp; omega
  · simp only [Sum.inl.injEq] at ha
    subst ha
    exact ⟨h2, hL, hm⟩

theorem runProbes_short {s : Level4State α} {xs : List (FreeMonoid α)} {h : Level4Harvest α}
    (hs : ShortInv C s) (hz : (runProbes C cut s xs).2 = .harvest h) :
    ∀ v ∈ h.versions, TreeShort C v.1 := by
  induction xs generalizing s with
  | nil => simp [runProbes] at hz
  | cons x xs ih =>
    have hr := recordProbe_short C cut (x := x) hs
    simp only [runProbes] at hz
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at hz
    · exact ih (applyProbe_short C cut hr ha) hz
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

/-- Every hypothesis a harvest kept is short. -/
theorem harvest_short (hC : 2 ≤ C.maxLeaves) (read : FreeMonoid α → ARU) {T : ℕ}
    (draws : Fin T → FreeMonoid α) {h : Level4Harvest α}
    (hh : (indecisiveLStarLevel4 C read draws).ending = .harvest h) :
    ∀ v ∈ h.versions, TreeShort C v.1 := by
  refine runProbes_short C (fun z => (read z).cut) (s := initialState) (xs := List.ofFn draws)
    ⟨⟨?_, ?_, ?_⟩, by simp [initialState]⟩ hh
  · simp [initialState, DTree.leaves]
  · simpa [initialState, DTree.leaves] using hC
  · simp [initialState, DTree.leaves, DTree.midfixes]

open MeasureTheory in
open scoped Classical in
/-- Over draws at most `L` long, a harvest whose hypotheses are short finds no string longer
than `L + maxLeaves − 2`. -/
theorem harvestLaw_short {D : Measure (FreeMonoid α)} {L : ℕ}
    (hlen : ∀ᵐ x ∂D, x.toList.length ≤ L) (k : ℕ) {h : Level4Harvest α}
    (hv : ∀ v ∈ h.versions, TreeShort C v.1) :
    harvestLaw D cut k h {z | L + C.maxLeaves < z.toList.length + 2} = 0 := by
  rw [harvestLaw_apply]
  refine mul_eq_zero_of_right _ (List.sum_eq_zero fun a ha => ?_)
  obtain ⟨v, hvm, rfl⟩ := List.mem_map.1 ha
  refine lintegral_eq_zero_of_ae_eq_zero ?_
  filter_upwards [hlen] with x hx
  simp
  exact fun a ha => siteHarvest_short C cut h.site (hv v hvm) hx a ha

end OrthoDFA
