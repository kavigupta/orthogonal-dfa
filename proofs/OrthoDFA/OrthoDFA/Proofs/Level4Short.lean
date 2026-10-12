import OrthoDFA.Proofs.Level4Distinct

/-!
# A harvest's strings are short

A split adds a midfix one letter longer than one already there, and a leaf, so a midfix is two
shorter than the leaves count, which the round keeps within `maxLeaves`. A harvested string is a
probe's prefix followed by a midfix, so it is at most `L + maxLeaves − 2` long where the probes are
at most `L` long (`harvest_short`).
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem midAt_mem_or_one : ∀ (T : DTree α) (p : List Bool), T.midAt p ∈ T.midfixes ∨ T.midAt p = 1
  | .leaf, _ => .inr rfl
  | .node m r a, [] => .inl (by simp [DTree.midAt, DTree.midfixes])
  | .node m r a, false :: p => by
    rcases midAt_mem_or_one r p with h | h
    · left; simp [DTree.midAt, DTree.midfixes, h]
    · right; simpa [DTree.midAt] using h
  | .node m r a, true :: p => by
    rcases midAt_mem_or_one a p with h | h
    · left; simp [DTree.midAt, DTree.midfixes, h]
    · right; simpa [DTree.midAt] using h

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

/-- The tree's leaves and midfixes, and the harvest lists' strings, within the round's lengths. -/
def ShortInv (C : Level4Settings) (L : ℕ) (s : Level4State α) : Prop :=
  2 ≤ s.tree.leaves.length ∧ s.tree.leaves.length ≤ C.maxLeaves
    ∧ (∀ m ∈ s.tree.midfixes, m.toList.length + 2 ≤ s.tree.leaves.length)
    ∧ (∀ z ∈ s.startUndecided, z.toList.length + 2 ≤ L + C.maxLeaves)
    ∧ (∀ p c, ∀ z ∈ s.edgeUndecided p c, z.toList.length + 2 ≤ L + C.maxLeaves)
    ∧ (∀ z ∈ s.stretchMiddles, z.toList.length + 2 ≤ L + C.maxLeaves)

variable (C : Level4Settings) (cut : FreeMonoid α → Option Bool)

theorem recordProbe_short {L : ℕ} {s : Level4State α} {x : FreeMonoid α} (h : ShortInv C L s)
    (hx : x.toList.length ≤ L) : ShortInv C L (recordProbe C cut s x) := by
  obtain ⟨h2, hL, hm, hs, he, hp⟩ := h
  have htree := (recordProbe_cases cut C s x).1
  obtain ⟨hE, -, hS, -⟩ := recordProbe_countProbe C cut s x
  have hr : ∀ {j z}, z ∈ (s.tree.route cut (prefixOf x j)).1 →
      z.toList.length + 2 ≤ L + C.maxLeaves := fun hz => route_short hm hL hx hz
  refine ⟨htree ▸ h2, htree ▸ hL, htree ▸ hm, ?_, fun p c => ?_, ?_⟩
  · rw [hS]
    intro z hz
    rcases List.mem_append.1 hz with hz | hz
    · exact hs z hz
    unfold startHarvestBy at hz
    split_ifs at hz
    · simp at hz
    rcases hb : s.tree.siftCut cut (prefixOf x C.walkStart) with q | b <;> rw [hb] at hz
    · simp at hz
    · simp only [Sum.elim_inr, List.mem_singleton] at hz
      subst hz
      exact hr (route_undecided cut _ _ _ hb).1
  · rw [hE]
    intro z hz
    rcases List.mem_append.1 hz with hz | hz
    · exact he p c z hz
    unfold edgeHarvestBy at hz
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hz
    exact hr (firstsBy_pos cut _ _ _ _ (List.mem_of_mem_filter hq)).1
  · rcases recordProbe_stretch C cut s x with ⟨-, hM, -⟩ | ⟨-, hM, -, -⟩
    · rw [hM]
      intro z hz
      rcases List.mem_append.1 hz with hz | hz
      · exact hp z hz
      unfold middleHarvestBy at hz
      rcases ho : probeBy cut s.tree s.edges C.walkStart x with _ | _ | _ | j | _ | j | _ <;>
        rw [ho] at hz <;> try simp at hz
      all_goals
        rcases hb : s.tree.siftCut cut (prefixOf x j) with q | b <;> rw [hb] at hz
        · simp at hz
        · simp only [Sum.elim_inr, List.mem_singleton] at hz
          subst hz
          exact hr (route_undecided cut _ _ _ hb).1
    · rw [hM]; simp

theorem applyProbe_short {L : ℕ} {s s' : Level4State α} {x : FreeMonoid α}
    (h : ShortInv C L (recordProbe C cut s x)) (ha : applyProbe C cut s x = .inl s') :
    ShortInv C L s' := by
  obtain ⟨hE, -, hS, -⟩ := applyProbe_counts C cut ha
  obtain ⟨h2, hL, hm, hs, he, hp⟩ := h
  have hmid : ∀ z ∈ s'.stretchMiddles, z.toList.length + 2 ≤ L + C.maxLeaves := by
    rcases applyProbe_newStretch C cut ha with ⟨-, -, hM⟩ | ⟨rfl, -⟩
    · rw [hM]; simp
    · exact hp
  have hlists : (∀ z ∈ s'.startUndecided, z.toList.length + 2 ≤ L + C.maxLeaves)
      ∧ (∀ p c, ∀ z ∈ s'.edgeUndecided p c, z.toList.length + 2 ≤ L + C.maxLeaves) :=
    ⟨hS ▸ hs, fun p c => hE ▸ he p c⟩
  -- the tree: kept, or split within `maxLeaves`
  unfold applyProbe at ha
  simp only [] at ha
  split at ha
  · cases ha
  unfold settleOne at ha
  split_ifs at ha with hv hbig
  · simp only [Sum.inl.injEq] at ha
    subst ha
    refine ⟨?_, not_lt.1 hbig, ?_, hlists.1, hlists.2, hmid⟩
    · rcases (fixEdge_cases cut C hv.choose_spec.choose_spec.choose_spec).2 with ⟨ht, -⟩ |
        ⟨t₀, w₀, -, -, ht⟩
      · rw [ht]; exact h2
      · rw [ht, DTree.splitAt_paths_length _ _ _ hv.choose_spec.choose_spec.choose_spec.1]
        omega
    · rcases (fixEdge_cases cut C hv.choose_spec.choose_spec.choose_spec).2 with ⟨ht, -⟩ |
        ⟨t₀, w₀, -, -, ht⟩
      · rw [ht]; exact hm
      · rw [ht, DTree.splitAt_paths_length _ _ _ hv.choose_spec.choose_spec.choose_spec.1]
        intro m hmm
        rcases midfixes_splitAt _ _ _ m hmm with hm' | rfl
        · have := hm m hm'; omega
        · simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
            List.length_singleton]
          rcases midAt_mem_or_one (recordProbe C cut s x).tree
              (commonPrefix hv.choose_spec.choose_spec.choose t₀)
            with h' | h'
          · have := hm _ h'; omega
          · rw [h']; simp; omega
  · simp only [Sum.inl.injEq] at ha
    subst ha
    exact ⟨h2, hL, hm, hlists.1, hlists.2, hmid⟩

theorem runProbes_short {L : ℕ} {s : Level4State α} {xs zs : List (FreeMonoid α)}
    (h : ShortInv C L s) (hxs : ∀ x ∈ xs, x.toList.length ≤ L)
    (hz : (runProbes C cut s xs).2 = .harvest zs) :
    ∀ z ∈ zs, z.toList.length + 2 ≤ L + C.maxLeaves := by
  induction xs generalizing s with
  | nil => simp [runProbes] at hz
  | cons x xs ih =>
    have hr := recordProbe_short C cut h (hxs x List.mem_cons_self)
    simp only [runProbes] at hz
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at hz
    · exact ih (applyProbe_short C cut hr ha) (fun x' hx' => hxs x' (List.mem_cons_of_mem _ hx')) hz
    · have he : e ≠ .tooBig := by rintro rfl; simp [toRoundEnd] at hz
      obtain ⟨rfl, -⟩ := applyProbe_inr_pre C cut ha he
      obtain ⟨-, -, -, hs, hE, hp⟩ := hr
      cases e with
      | success => simp [toRoundEnd] at hz
      | tooBig => simp [toRoundEnd] at hz
      | harvestStart =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hs
      | harvestPT =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hp
      | harvest e =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hE e.1 e.2

/-- Where the probes are at most `L` long, a harvested string is at most `L + maxLeaves − 2`
long. -/
theorem harvest_short (L : ℕ) (hC : 2 ≤ C.maxLeaves) (read : FreeMonoid α → ARU) {T : ℕ}
    (draws : Fin T → FreeMonoid α) (hdraws : ∀ i, (draws i).toList.length ≤ L)
    {zs : List (FreeMonoid α)} (h : (indecisiveLStarLevel4 C read draws).ending = .harvest zs) :
    ∀ z ∈ zs, z.toList.length + 2 ≤ L + C.maxLeaves := by
  refine runProbes_short C (fun z => (read z).cut) (s := initialState) (xs := List.ofFn draws)
    ⟨?_, ?_, ?_, by simp [initialState], by simp [initialState], by simp [initialState]⟩
    (fun x hx => ?_) h
  · simp [initialState, DTree.leaves]
  · simpa [initialState, DTree.leaves] using hC
  · simp [initialState, DTree.leaves, DTree.midfixes]
  · obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    exact hdraws i

end OrthoDFA
