import OrthoDFA.RoundStrong
import OrthoDFA.Proofs.RoundLevel

/-!
# The strengthened round's deterministic claims
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

/-- Every node's path, the root's first. -/
def nodes : DTree α → List (List Bool)
  | .leaf => [[]]
  | .node _ r a => [] :: (r.nodes.map (false :: ·) ++ a.nodes.map (true :: ·))

omit [Fintype α] [DecidableEq α] in
theorem paths_sub_nodes : ∀ (t : DTree α), ∀ p ∈ t.paths, p ∈ t.nodes
  | .leaf, p, h => by simpa [paths, nodes] using h
  | .node _ r a, p, h => by
    simp only [paths, List.mem_append, List.mem_map] at h
    simp only [nodes, List.mem_cons, List.mem_append, List.mem_map]
    rcases h with ⟨q, hq, rfl⟩ | ⟨q, hq, rfl⟩
    · exact .inr (.inl ⟨q, paths_sub_nodes r q hq, rfl⟩)
    · exact .inr (.inr ⟨q, paths_sub_nodes a q hq, rfl⟩)

omit [Fintype α] [DecidableEq α] in
theorem nodes_length : ∀ t : DTree α, t.nodes.length + 1 = 2 * t.paths.length
  | .leaf => by simp [nodes, paths]
  | .node _ r a => by
    have := nodes_length r
    have := nodes_length a
    simp only [nodes, paths, List.length_cons, List.length_append, List.length_map]
    omega

omit [Fintype α] [DecidableEq α] in
theorem sift_mem_paths {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α) (p : List Bool), t.sift cut x = .inl p → p ∈ t.paths
  | .leaf, x, p, h => by
    simp only [sift, route, Sum.inl.injEq] at h
    simp [paths, ← h]
  | .node m r a, x, p, h => by
    simp only [sift, route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> simp only [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [paths, List.mem_append, List.mem_map]
        exact .inl ⟨q, sift_mem_paths r x q hr, rfl⟩
      · simp at h
    · rcases hr : (a.route cut x).2 with q | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        simp only [paths, List.mem_append, List.mem_map]
        exact .inr ⟨q, sift_mem_paths a x q hr, rfl⟩
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem paths_ne_nil_of_node {m : FreeMonoid α} {r a : DTree α} :
    [] ∉ (DTree.node m r a).paths := by
  simp [paths]

omit [Fintype α] [DecidableEq α] in
theorem splitAt_paths_length (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), p ∈ t.paths →
      (t.splitAt d p).paths.length = t.paths.length + 1
  | .leaf, [], _ => by simp [splitAt, paths]
  | .leaf, _ :: _, h => by simp [paths] at h
  | .node _ _ _, [], h => absurd h paths_ne_nil_of_node
  | .node n r a, false :: p, h => by
    have : p ∈ r.paths := by simpa [paths] using h
    simp only [splitAt, paths, List.length_append, List.length_map,
      splitAt_paths_length d r p this]
    omega
  | .node n r a, true :: p, h => by
    have : p ∈ a.paths := by simpa [paths] using h
    simp only [splitAt, paths, List.length_append, List.length_map,
      splitAt_paths_length d a p this]
    omega

omit [Fintype α] [DecidableEq α] in
theorem nodes_splitAt (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), ∀ q ∈ t.nodes, q ∈ (t.splitAt d p).nodes
  | .leaf, [], q, h => by simp_all [splitAt, nodes]
  | .leaf, _ :: _, q, h => h
  | .node _ _ _, [], q, h => h
  | .node n r a, false :: p, q, h => by
    simp only [nodes, List.mem_cons, List.mem_append, List.mem_map, splitAt] at h ⊢
    rcases h with h | ⟨q', hq, rfl⟩ | ⟨q', hq, rfl⟩
    · exact .inl h
    · exact .inr (.inl ⟨q', nodes_splitAt d r p q' hq, rfl⟩)
    · exact .inr (.inr ⟨q', hq, rfl⟩)
  | .node n r a, true :: p, q, h => by
    simp only [nodes, List.mem_cons, List.mem_append, List.mem_map, splitAt] at h ⊢
    rcases h with h | ⟨q', hq, rfl⟩ | ⟨q', hq, rfl⟩
    · exact .inl h
    · exact .inr (.inl ⟨q', hq, rfl⟩)
    · exact .inr (.inr ⟨q', nodes_splitAt d a p q' hq, rfl⟩)

omit [Fintype α] [DecidableEq α] in
theorem append_not_mem_paths :
    ∀ (t : DTree α) (p : List Bool) (b : Bool), p ∈ t.paths → p ++ [b] ∉ t.paths
  | .leaf, p, b, h => by simp_all [paths]
  | .node _ r a, [], b, h => absurd h paths_ne_nil_of_node
  | .node _ r a, c :: p, b, h => by
    simp only [paths, List.mem_append, List.mem_map, List.cons_append] at h ⊢
    rintro (⟨q, hq, he⟩ | ⟨q, hq, he⟩) <;> rcases h with ⟨q', hq', he'⟩ | ⟨q', hq', he'⟩ <;>
      simp only [List.cons.injEq] at he he' <;> obtain ⟨rfl, rfl⟩ := he'
    · obtain ⟨-, rfl⟩ := he; exact append_not_mem_paths r q' b hq' hq
    · simp at he
    · simp at he
    · obtain ⟨-, rfl⟩ := he; exact append_not_mem_paths a q' b hq' hq

end DTree

section MajPath

theorem sidePath_mem_paths (side : FreeMonoid α → Bool) :
    ∀ (t : DTree α) (x : FreeMonoid α), sidePath side t x ∈ t.paths
  | .leaf, x => by simp [sidePath, DTree.paths]
  | .node m r a, x => by
    simp only [sidePath, DTree.paths, List.mem_append, List.mem_map]
    split
    · exact .inr ⟨_, sidePath_mem_paths side a x, rfl⟩
    · exact .inl ⟨_, sidePath_mem_paths side r x, rfl⟩

theorem sidePath_splitAt (side : FreeMonoid α → Bool) (d : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), p ∈ t.paths → ∀ x : FreeMonoid α,
      sidePath side (t.splitAt d p) x
        = if sidePath side t x = p then p ++ [side (x * d)] else sidePath side t x
  | .leaf, [], _, x => by
    simp only [DTree.splitAt, sidePath, if_true, List.nil_append]
    split <;> simp_all
  | .leaf, _ :: _, h, _ => by simp [DTree.paths] at h
  | .node _ _ _, [], h, _ => absurd h DTree.paths_ne_nil_of_node
  | .node n r a, false :: p, h, x => by
    have hp : p ∈ r.paths := by simpa [DTree.paths] using h
    simp only [DTree.splitAt, sidePath]
    split
    · simp
    · rw [sidePath_splitAt side d r p hp x]
      split <;> simp_all
  | .node n r a, true :: p, h, x => by
    have hp : p ∈ a.paths := by simpa [DTree.paths] using h
    simp only [DTree.splitAt, sidePath]
    split
    · rw [sidePath_splitAt side d a p hp x]
      split <;> simp_all
    · simp

variable {Q : Type*} [Fintype Q]

open scoped Classical in
/-- How many leaves the reference's states reach. -/
noncomputable def classCount (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (t : DTree α) : ℕ :=
  (Finset.univ.image fun q => sidePath side t (rep q)).card

theorem classCount_le (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α) (t : DTree α) :
    classCount side rep t ≤ Fintype.card Q := by
  classical
  exact Finset.card_image_le.trans (by simp)

open scoped Classical in
/-- A split at a leaf raises the count by one where it separates two states, else keeps it. -/
theorem classCount_splitAt (side : FreeMonoid α → Bool) (rep : Q → FreeMonoid α)
    (r : SplitRec α) (hp : r.leaf ∈ r.tree.paths) :
    classCount side rep r.tree + (if r.Separates side rep then 1 else 0)
      ≤ classCount side rep (r.tree.splitAt r.d r.leaf) := by
  classical
  set p := r.leaf
  set g : Q → List Bool := fun q => sidePath side r.tree (rep q)
  set g' : Q → List Bool := fun q => sidePath side (r.tree.splitAt r.d p) (rep q)
  set π : List Bool → List Bool := fun z => if z = p ++ [true] ∨ z = p ++ [false] then p else z
  have hg' : ∀ q, g' q = if g q = p then p ++ [side (rep q * r.d)] else g q := fun q =>
    sidePath_splitAt side r.d r.tree p hp (rep q)
  have hπ : ∀ q, π (g' q) = g q := fun q => by
    rw [hg']
    split_ifs with h
    · simp only [π]
      rw [if_pos (by cases side (rep q * r.d) <;> simp)]
      exact h.symm
    · have hm : g q ∈ r.tree.paths := sidePath_mem_paths side r.tree (rep q)
      simp only [π]
      rw [if_neg]
      rintro (he | he) <;> rw [he] at hm
      · exact DTree.append_not_mem_paths r.tree p true hp hm
      · exact DTree.append_not_mem_paths r.tree p false hp hm
  have himg : Finset.univ.image g = (Finset.univ.image g').image π := by
    rw [Finset.image_image]
    congr 1
    funext q
    exact (hπ q).symm
  unfold classCount
  change (Finset.univ.image g).card + _ ≤ (Finset.univ.image g').card
  rw [himg]
  split_ifs with hs
  · obtain ⟨q, q', hq, hq', hne⟩ := hs
    have hlt : ((Finset.univ.image g').image π).card < (Finset.univ.image g').card := by
      refine lt_of_le_of_ne Finset.card_image_le fun he => ?_
      have hinj := Finset.card_image_iff.1 he
      have h1 : g' q ∈ Finset.univ.image g' := Finset.mem_image_of_mem _ (Finset.mem_univ _)
      have h2 : g' q' ∈ Finset.univ.image g' := Finset.mem_image_of_mem _ (Finset.mem_univ _)
      have := hinj h1 h2 (by rw [hπ, hπ]; exact hq.trans hq'.symm)
      rw [hg', hg', if_pos hq, if_pos hq'] at this
      exact hne (by simpa using this)
    omega
  · simpa using Finset.card_image_le

end MajPath

section SplitFacts

/-- What a recorded split read: its two strings sift to the leaf, and part at the distinguisher,
both decided. -/
def SplitOK (R : CutReads α) (r : SplitRec α) : Prop :=
  r.tree.sift R.cut r.y = .inl r.leaf ∧ r.tree.sift R.cut r.sprime = .inl r.leaf
    ∧ ∃ b, R.cut (r.y * r.d) = some b ∧ R.cut (r.sprime * r.d) = some (!b)

theorem parting_inl {cut : FreeMonoid α → Option Bool} {x y pre d : FreeMonoid α} :
    ∀ t : DTree α, t.parting cut x y pre = some (.inl d) →
      ∃ b, cut (x * d) = some b ∧ cut (y * d) = some (!b)
  | .leaf, h => by simp [DTree.parting] at h
  | .node m r a, h => by
    simp only [DTree.parting] at h
    rcases hx : cut (x * (pre * m)) with _ | _ | _ <;>
      rcases hy : cut (y * (pre * m)) with _ | _ | _ <;>
      simp only [hx, hy, reduceCtorEq, Option.some.injEq, Sum.inl.injEq] at h
    · exact parting_inl r h
    · subst h; exact ⟨false, hx, hy⟩
    · subst h; exact ⟨true, hx, hy⟩
    · exact parting_inl a h

theorem seedStep_split_facts (K : StageKnobs α) (R : CutReads α) {t : DTree α}
    {pool : List (FreeMonoid α)} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} {d : FreeMonoid α} {s1 : List Bool} {y sprime : FreeMonoid α}
    (h : seedStep K R t pool edges k x ps fd = .split d s1 y sprime) :
    SplitOK R ⟨t, s1, d, y, sprime⟩ := by
  unfold seedStep at h
  simp only [] at h
  split at h
  · simp at h
  split at h
  · simp at h
  split_ifs at h
  rename_i hsp
  split at h
  · simp at h
  rename_i p hp
  split_ifs at h with hg
  push Not at hg
  split at h
  · simp at h
  · simp at h
  rename_i d' hpt
  split at h
  · simp only [SeedResult.split.injEq] at h
    obtain ⟨rfl, rfl, rfl, rfl⟩ := h
    exact ⟨hg.2, hg.1 ▸ hp, parting_inl t hpt⟩
  · simp at h

theorem faithfulRoute_sift {R : CutReads α} {side : FreeMonoid α → Bool} {z : FreeMonoid α} :
    ∀ (t : DTree α) (x : FreeMonoid α), FaithfulRoute R side z t x →
      t.sift R.cut x = .inl (sidePath side t z)
  | .leaf, _, _ => rfl
  | .node m r a, x, ⟨hc, hrest⟩ => by
    unfold DTree.sift DTree.route
    simp only [hc, sidePath]
    cases hs : side (z * m) <;> simp only [hs, if_true] at hrest ⊢
    · have := faithfulRoute_sift r x hrest
      unfold DTree.sift at this
      simp [this]
    · have := faithfulRoute_sift a x hrest
      unfold DTree.sift at this
      simp [this]

variable {Q : Type*} (R : CutReads α) (A : DFA (FreeMonoid α) Q) (side : FreeMonoid α → Bool)
  (rep : Q → FreeMonoid α)

theorem noisy_of_not_separates {r : SplitRec α} (hok : SplitOK R r)
    (hs : ¬ r.Separates side rep) : r.Noisy R A side rep := by
  by_contra hn
  simp only [SplitRec.Noisy, not_or, not_not] at hn
  obtain ⟨h1, h2, h3, h4⟩ := hn
  obtain ⟨hy, hsp, b, hby, hbs⟩ := hok
  refine hs ⟨A.state r.y, A.state r.sprime, ?_, ?_, fun he => ?_⟩
  · have := faithfulRoute_sift r.tree r.y h1
    rw [hy] at this
    exact (Sum.inl.inj this).symm
  · have := faithfulRoute_sift r.tree r.sprime h2
    rw [hsp] at this
    exact (Sum.inl.inj this).symm
  · unfold Faithful at h3 h4
    rw [hby] at h3
    rw [hbs] at h4
    rw [← Option.some.inj h3, ← Option.some.inj h4] at he
    cases b <;> simp at he

theorem same_state_unfaithful {r : SplitRec α} (hok : SplitOK R r)
    (hq : A.state r.y = A.state r.sprime) :
    ¬ Faithful R A side rep r.y r.d ∨ ¬ Faithful R A side rep r.sprime r.d := by
  by_contra hn
  simp only [not_or, not_not] at hn
  obtain ⟨h3, h4⟩ := hn
  obtain ⟨-, -, b, hby, hbs⟩ := hok
  unfold Faithful at h3 h4
  rw [hby] at h3
  rw [hbs, ← hq] at h4
  have := (Option.some.inj h3).trans (Option.some.inj h4).symm
  cases b <;> simp at this

end SplitFacts

section Step

variable (C : StrongCfg α) (R : CutReads α)

theorem edge_source_mem {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {fd : ℕ} (h : probeOutcome R t edges k x = .edge ps fd) :
    ps.getD (fd - 1 - k) [] ∈ t.paths := by
  obtain ⟨ps₀, hi, hw, hb⟩ := probeOutcome_search R h trivial
  obtain ⟨p₀, hk, hf, hkh, hhn, hpn⟩ := walkCheck_inr R hw
  set walkAt : ℕ → List Bool := fun j => ps₀.getD (j - k) [] with hwalk
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  have hpk : agreesAt R t x walkAt k = some true := by
    simp only [agreesAt, hk, Sum.elim_inl, hwalk, Nat.sub_self, hhead, decide_true]
  obtain ⟨rfl, hfd1, hfd2, hfd3, hfd4⟩ :=
    bracketAt_edge (agreesAt R t x walkAt) ps₀ (hi - k) k hi ps fd hkh le_rfl hpk hpn hb
  simp only [agreesAt] at hfd3
  rcases hs : t.sift R.cut (prefixOf x (fd - 1)) with p | b <;> rw [hs] at hfd3
  · simp only [Sum.elim_inl, Option.some.injEq, decide_eq_true_eq] at hfd3
    rw [show ps.getD (fd - 1 - k) [] = p by simp only [hwalk] at hfd3; exact hfd3.symm]
    exact DTree.sift_mem_paths t _ p hs
  · simp at hfd3

theorem edgeAt_mem {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {e : List Bool × α} (h : edgeAt R t edges k x = some e) : e.1 ∈ t.paths := by
  unfold edgeAt at h
  split at h
  · rename_i ps fd ho
    obtain ⟨c, -, rfl⟩ := Option.map_eq_some_iff.1 h
    exact edge_source_mem R ho
  · simp at h

theorem split_leaf_mem {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α} {k : ℕ}
    (hl : Learned R t edges) {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    {d y sp : FreeMonoid α} {s1 : List Bool}
    (h : seedStep C.K R t pool edges k x ps fd = .split d s1 y sp) : s1 ∈ t.paths := by
  obtain ⟨⟨c, s2, he⟩, -⟩ := seedStep_split_spec C.K R h
  exact DTree.sift_mem_paths t _ _ (hl _ _ _ _ he).1

theorem strongStep_state (A : RoundAcc α) (x : FreeMonoid α) :
    (strongStep C R A x).s.tree = (probeStepK C.K R C.k A.s x).tree
      ∧ (strongStep C R A x).s.edges = (probeStepK C.K R C.k A.s x).edges
      ∧ (strongStep C R A x).s.pool = (probeStepK C.K R C.k A.s x).pool
      ∧ (strongStep C R A x).used = A.used + 1 ∧ (strongStep C R A x).certs = A.certs := by
  unfold strongStep
  simp only []
  repeat' split
  all_goals simp

/-- A step counted an end of the split test against an edge not given up. -/
def LiveInc (A A' : RoundAcc α) : Prop :=
  ∃ e, ¬ givenUp C A e ∧ e.1 ∈ A.s.tree.paths ∧ A'.att e = A.att e + 1

theorem seedStep_of_edgeAt_none {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    {k : ℕ} {x : FreeMonoid α} {ps : List (List Bool)} {fd : ℕ}
    (ho : probeOutcome R t edges k x = .edge ps fd) (he : edgeAt R t edges k x = none) :
    seedStep C.K R t pool edges k x ps fd = .dropped := by
  simp only [edgeAt, ho, Option.map_eq_none_iff] at he
  simp [seedStep, he]

theorem strongStep_cases (A : RoundAcc α) (x : FreeMonoid α)
    (hl : Learned R A.s.tree A.s.edges) :
    ((strongStep C R A x).splits = A.splits ∧ (strongStep C R A x).s.tree = A.s.tree
      ∧ (∀ e, A.att e ≤ (strongStep C R A x).att e)
      ∧ ((strongStep C R A x).s.streak = A.s.streak + 1
        ∨ ((strongStep C R A x).s.streak = 0 ∧ LiveInc C A (strongStep C R A x)))
      ∧ (LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x → LiveInc C A (strongStep C R A x)))
    ∨ ∃ r : SplitRec α, (strongStep C R A x).splits = A.splits ++ [r] ∧ r.tree = A.s.tree
      ∧ r.leaf ∈ A.s.tree.paths ∧ (strongStep C R A x).s.tree = A.s.tree.splitAt r.d r.leaf
      ∧ (strongStep C R A x).att = A.att ∧ (strongStep C R A x).s.streak = 0 ∧ SplitOK R r := by
  by_cases hE : ∃ ps fd, probeOutcome R A.s.tree A.s.edges C.k x = .edge ps fd
  swap
  · left
    have hL : ¬ LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x := by
      rintro ⟨e, he, -⟩
      unfold edgeAt at he
      split at he
      · exact hE ⟨_, _, by assumption⟩
      · simp at he
    have hst : strongStep C R A x = { A with s := probeStepK C.K R C.k A.s x, used := A.used + 1 }
        := by
      unfold strongStep
      simp only []
      split
      · exact absurd ⟨_, _, by assumption⟩ hE
      · rfl
    have hp : (probeStepK C.K R C.k A.s x).tree = A.s.tree
        ∧ (probeStepK C.K R C.k A.s x).streak = A.s.streak + 1 := by
      unfold probeStepK
      simp only []
      split
      · exact absurd ⟨_, _, by assumption⟩ hE
      all_goals simp [closeK]
    rw [hst]
    exact ⟨rfl, hp.1, fun _ => le_rfl, .inl hp.2, fun h => absurd h hL⟩
  obtain ⟨ps, fd, ho⟩ := hE
  rcases hs : seedStep C.K R A.s.tree A.s.pool A.s.edges C.k x ps fd with
    ⟨d, s1, y, sp⟩ | ⟨s1, sp⟩ | b | _
  · right
    refine ⟨⟨A.s.tree, s1, d, y, sp⟩, ?_, rfl, split_leaf_mem C R hl hs, ?_, ?_, ?_,
      seedStep_split_facts C.K R hs⟩ <;>
      simp [strongStep, probeStepK, ho, hs, closeK]
  all_goals
    left
    rcases he : edgeAt R A.s.tree A.s.edges C.k x with _ | e
    · have hd := seedStep_of_edgeAt_none C R (pool := A.s.pool) ho he
      rw [hs] at hd
      cases hd
      all_goals
        have hL : ¬ LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x := by
          rintro ⟨e, he', -⟩
          rw [he] at he'
          cases he'
        refine ⟨?_, ?_, fun _ => ?_, .inl ?_, fun h => absurd h hL⟩ <;>
          simp [strongStep, probeStepK, ho, hs, he, closeK]
    have hmem := edgeAt_mem R he
    have hatt : ∀ e', A.att e' ≤ (strongStep C R A x).att e' := fun e' => by
      by_cases h : e' = e
      · subst h; by_cases hg : givenUp C A e' <;> simp [strongStep, ho, hs, he, hg]
      · by_cases hg : givenUp C A e <;> simp [strongStep, ho, hs, he, hg, Function.update_of_ne h]
    by_cases hg : givenUp C A e
    · have hL : ¬ LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x := by
        rintro ⟨e', he', hg'⟩
        rw [he] at he'
        cases he'
        exact hg' hg
      refine ⟨?_, ?_, hatt, .inl ?_, fun h => absurd h hL⟩ <;>
        simp [strongStep, probeStepK, ho, hs, he, closeK, hg]
    · have hinc : LiveInc C A (strongStep C R A x) := by
        refine ⟨e, hg, hmem, ?_⟩
        simp [strongStep, ho, hs, he, hg]
      refine ⟨?_, ?_, hatt, ?_, fun _ => hinc⟩
      · simp [strongStep, probeStepK, ho, hs, he, closeK, hg]
      · simp [strongStep, probeStepK, ho, hs, he, closeK, hg]
      · have h0 : (strongStep C R A x).s.streak = (probeStepK C.K R C.k A.s x).streak := by
          simp [strongStep, ho, hs, he, hg]
        rw [h0]
        simp [probeStepK, ho, hs, closeK, hinc]

end Step

section Count

variable (C : StrongCfg α) (R : CutReads α)

/-- The edge keys counted: every node's path by every letter. -/
noncomputable def keys (t : DTree α) : Finset (List Bool × α) :=
  t.nodes.toFinset ×ˢ Finset.univ

/-- The ends of the split test counted against edges while not given up. -/
noncomputable def phi (A : RoundAcc α) : ℕ :=
  ∑ e ∈ keys A.s.tree, min (A.att e) (C.mmax A.s.tree.paths.length)

/-- What the round has done that its readings are counted by. -/
noncomputable def ev (A : RoundAcc α) : ℕ := A.splits.length + phi C A

omit [DecidableEq α] in
theorem leaves_pos (t : DTree α) : 1 ≤ t.paths.length :=
  List.length_pos_iff.2 (paths_ne_nil t)

theorem phi_le (A : RoundAcc α) :
    phi C A ≤ C.mmax A.s.tree.paths.length * Fintype.card α
      * (2 * (A.s.tree.paths.length - 1) + 1) := by
  classical
  have hn := DTree.nodes_length A.s.tree
  have h1 := leaves_pos A.s.tree
  calc phi C A ≤ ∑ _e ∈ keys A.s.tree, C.mmax A.s.tree.paths.length :=
        Finset.sum_le_sum fun _ _ => min_le_right _ _
    _ = (keys A.s.tree).card * C.mmax A.s.tree.paths.length := by simp
    _ ≤ (A.s.tree.nodes.length * Fintype.card α) * C.mmax A.s.tree.paths.length := by
        refine Nat.mul_le_mul_right _ ?_
        simp only [keys, Finset.card_product, Finset.card_univ]
        exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)
    _ = _ := by
        rw [show A.s.tree.nodes.length = 2 * (A.s.tree.paths.length - 1) + 1 by omega]
        ring

theorem phi_mono (hm : Monotone C.mmax) {A A' : RoundAcc α}
    (hn : A.s.tree.paths.length ≤ A'.s.tree.paths.length)
    (hnodes : ∀ q ∈ A.s.tree.nodes, q ∈ A'.s.tree.nodes) (hatt : ∀ e, A.att e ≤ A'.att e) :
    phi C A ≤ phi C A' := by
  classical
  have hsub : keys A.s.tree ⊆ keys A'.s.tree := fun e he => by
    simp only [keys, Finset.mem_product, List.mem_toFinset, Finset.mem_univ, and_true] at he ⊢
    exact hnodes _ he
  calc phi C A ≤ ∑ e ∈ keys A.s.tree, min (A'.att e) (C.mmax A'.s.tree.paths.length) :=
        Finset.sum_le_sum fun e _ => min_le_min (hatt e) (hm hn)
    _ ≤ phi C A' := Finset.sum_le_sum_of_subset hsub

theorem phi_inc (hm : Monotone C.mmax) {A A' : RoundAcc α}
    (hn : A.s.tree.paths.length ≤ A'.s.tree.paths.length)
    (hnodes : ∀ q ∈ A.s.tree.nodes, q ∈ A'.s.tree.nodes) (hatt : ∀ e, A.att e ≤ A'.att e)
    (hinc : LiveInc C A A') : phi C A + 1 ≤ phi C A' := by
  classical
  obtain ⟨e, hg, hmem, he⟩ := hinc
  have hsub : keys A.s.tree ⊆ keys A'.s.tree := fun e he => by
    simp only [keys, Finset.mem_product, List.mem_toFinset, Finset.mem_univ, and_true] at he ⊢
    exact hnodes _ he
  have hek : e ∈ keys A.s.tree := by
    simp only [keys, Finset.mem_product, List.mem_toFinset, Finset.mem_univ, and_true]
    exact DTree.paths_sub_nodes _ _ hmem
  have hg' : A.att e < C.mmax A.s.tree.paths.length := Nat.lt_of_not_le hg
  calc phi C A + 1 ≤ ∑ e ∈ keys A.s.tree, min (A'.att e) (C.mmax A'.s.tree.paths.length) := by
        refine Finset.sum_lt_sum (fun e _ => min_le_min (hatt e) (hm hn)) ⟨e, hek, ?_⟩
        rw [he, min_eq_left hg'.le]
        exact lt_min (Nat.lt_succ_self _) (lt_of_lt_of_le hg' (hm hn))
    _ ≤ phi C A' := Finset.sum_le_sum_of_subset hsub

/-- What every state the round reaches satisfies. -/
def StrongInv (A : RoundAcc α) : Prop :=
  Learned R A.s.tree A.s.edges ∧ A.splits.length + 2 = A.s.tree.paths.length

theorem strongStep_ev (hm : Monotone C.mmax) (A : RoundAcc α) (x : FreeMonoid α)
    (hI : StrongInv R A) :
    StrongInv R (strongStep C R A x) ∧ ev C A ≤ ev C (strongStep C R A x)
      ∧ ((strongStep C R A x).s.streak = A.s.streak + 1
        ∨ ((strongStep C R A x).s.streak = 0 ∧ ev C A + 1 ≤ ev C (strongStep C R A x)))
      ∧ (LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x
        → ev C A + 1 ≤ ev C (strongStep C R A x))
      ∧ (strongStep C R A x).used = A.used + 1 ∧ (strongStep C R A x).certs = A.certs
      ∧ A.s.tree.paths.length ≤ (strongStep C R A x).s.tree.paths.length := by
  obtain ⟨hl, hlen⟩ := hI
  obtain ⟨htr, hed, -, hus, hce⟩ := strongStep_state C R A x
  have hlearn : Learned R (strongStep C R A x).s.tree (strongStep C R A x).s.edges := by
    rw [htr, hed]
    exact probeStepK_learned C.K R hl
  rcases strongStep_cases C R A x hl with ⟨hsp, ht, hatt, hstr, hlive⟩ |
    ⟨r, hsp, hrt, hrl, ht, hatt, hstr, -⟩
  · have hn : A.s.tree.paths.length ≤ (strongStep C R A x).s.tree.paths.length := by rw [ht]
    have hnodes : ∀ q ∈ A.s.tree.nodes, q ∈ (strongStep C R A x).s.tree.nodes := by
      rw [ht]; exact fun q h => h
    have hinc := fun h => phi_inc C hm hn hnodes hatt h
    refine ⟨⟨hlearn, by rw [hsp, ht]; exact hlen⟩, ?_, ?_, ?_, hus, hce, hn⟩
    · unfold ev; rw [hsp]; exact Nat.add_le_add_left (phi_mono C hm hn hnodes hatt) _
    · rcases hstr with h | ⟨h, hi⟩
      · exact .inl h
      · refine .inr ⟨h, ?_⟩
        unfold ev; rw [hsp]; have := hinc hi; omega
    · intro h
      unfold ev; rw [hsp]; have := hinc (hlive h); omega
  · have hn : A.s.tree.paths.length + 1 = (strongStep C R A x).s.tree.paths.length := by
      rw [ht, DTree.splitAt_paths_length _ _ _ hrl]
    have hnodes : ∀ q ∈ A.s.tree.nodes, q ∈ (strongStep C R A x).s.tree.nodes := by
      rw [ht]; exact DTree.nodes_splitAt _ _ _
    have hph := phi_mono C hm (by omega) hnodes (fun e => by rw [hatt])
    have hev : ev C A + 1 ≤ ev C (strongStep C R A x) := by
      unfold ev; rw [hsp, List.length_append]; simp only [List.length_singleton]; omega
    refine ⟨⟨hlearn, by rw [hsp, List.length_append]; simp only [List.length_singleton]; omega⟩,
      by omega, .inr ⟨hstr, hev⟩, fun _ => hev, hus, hce, by omega⟩

/-- One probe of the pass. -/
noncomputable def passBody (A : RoundAcc α) (x : FreeMonoid α) : RoundAcc α :=
  if C.K.patience ≤ A.s.streak ∨ budgetOf C A.s.tree.paths.length ≤ A.used then A
  else strongStep C R A x

theorem strongPass_eq (A : RoundAcc α) (probes : List (FreeMonoid α)) :
    strongPass C R A probes
      = probes.foldl (passBody C R) { A with s := { A.s with streak := 0 } } :=
  rfl

/-- Within a pass from `X`, each probe spends one, and a probe that resets the quiet streak is
paid for by `patience` of what the round has done. -/
theorem fold_inv (hm : Monotone C.mmax) :
    ∀ (probes : List (FreeMonoid α)) (X : RoundAcc α) (u0 e0 : ℕ), StrongInv R X →
      X.s.streak ≤ C.K.patience → e0 ≤ ev C X →
      X.used + C.K.patience * e0 ≤ u0 + X.s.streak + C.K.patience * ev C X →
      let Y := probes.foldl (passBody C R) X
      StrongInv R Y ∧ Y.s.streak ≤ C.K.patience ∧ e0 ≤ ev C Y
        ∧ Y.used + C.K.patience * e0 ≤ u0 + Y.s.streak + C.K.patience * ev C Y
        ∧ Y.certs = X.certs ∧ X.s.tree.paths.length ≤ Y.s.tree.paths.length
  | [], X, _, _, hI, hs, he, hu => ⟨hI, hs, he, hu, rfl, le_rfl⟩
  | x :: xs, X, u0, e0, hI, hs, he, hu => by
    simp only [List.foldl_cons]
    unfold passBody
    split_ifs with hg
    · exact fold_inv hm xs X u0 e0 hI hs he hu
    · push Not at hg
      obtain ⟨hI', hev, hstr, -, hus, hce, hn⟩ := strongStep_ev C R hm X x hI
      have := fold_inv hm xs (strongStep C R X x) u0 e0 hI' ?_ (he.trans hev) ?_
      · exact ⟨this.1, this.2.1, this.2.2.1, this.2.2.2.1, this.2.2.2.2.1.trans hce,
          hn.trans this.2.2.2.2.2⟩
      · rcases hstr with h | ⟨h, -⟩ <;> omega
      · rw [hus]
        rcases hstr with h | ⟨h, hi⟩
        · rw [h]
          have := Nat.mul_le_mul_left C.K.patience hev
          omega
        · rw [h]
          have := Nat.mul_le_mul_left C.K.patience hi
          rw [Nat.mul_add] at this
          omega

theorem strongPass_inv (hm : Monotone C.mmax) (A : RoundAcc α) (probes : List (FreeMonoid α))
    (hI : StrongInv R A) :
    let Y := strongPass C R A probes
    StrongInv R Y ∧ Y.s.streak ≤ C.K.patience ∧ ev C A ≤ ev C Y
      ∧ Y.used + C.K.patience * ev C A ≤ A.used + C.K.patience + C.K.patience * ev C Y
      ∧ Y.certs = A.certs ∧ A.s.tree.paths.length ≤ Y.s.tree.paths.length := by
  rw [strongPass_eq]
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := fold_inv C R hm probes { A with s := { A.s with streak := 0 } }
    A.used (ev C A) hI (Nat.zero_le _) le_rfl (by simp [ev, phi])
  exact ⟨h1, h2, h3, by omega, h5, h6⟩

/-- A pass whose first probe ends at an edge not given up does something counted. -/
theorem strongPass_live (hm : Monotone C.mmax) (A : RoundAcc α) (x : FreeMonoid α)
    (rest : List (FreeMonoid α)) (hI : StrongInv R A) (hp : 1 ≤ C.K.patience)
    (hb : A.used < budgetOf C A.s.tree.paths.length)
    (hlive : LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x) :
    ev C A + 1 ≤ ev C (strongPass C R A (x :: rest)) := by
  rw [strongPass_eq, List.foldl_cons]
  set A₀ : RoundAcc α := { A with s := { A.s with streak := 0 } }
  have hI₀ : StrongInv R A₀ := hI
  have hb0 : passBody C R A₀ x = strongStep C R A₀ x := by
    unfold passBody
    rw [if_neg]
    push Not
    exact ⟨hp, hb⟩
  rw [hb0]
  obtain ⟨hI', hev, hstr, hl, -⟩ := strongStep_ev C R hm A₀ x hI₀
  have h1 : ev C A + 1 ≤ ev C (strongStep C R A₀ x) := hl hlive
  have h0 : A₀.s.streak = 0 := rfl
  have := (fold_inv C R hm rest (strongStep C R A₀ x) (strongStep C R A₀ x).used
    (ev C (strongStep C R A₀ x)) hI' ?_ le_rfl (by omega)).2.2.1
  · omega
  · rcases hstr with h | ⟨h, -⟩ <;> omega

theorem strongReading_fst (j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (y : C.Draws) :
    (strongReading C R j A first y).1.s = (strongPass C R A (first ++ List.ofFn y.1)).s
      ∧ (strongReading C R j A first y).1.att = (strongPass C R A (first ++ List.ofFn y.1)).att
      ∧ (strongReading C R j A first y).1.splits
        = (strongPass C R A (first ++ List.ofFn y.1)).splits
      ∧ (strongReading C R j A first y).1.used
        = (strongPass C R A (first ++ List.ofFn y.1)).used := by
  unfold strongReading
  simp only []
  split_ifs <;> simp

theorem strongReading_exhausted {j : ℕ} {A : RoundAcc α} {first : List (FreeMonoid α)}
    {y : C.Draws} (h : (strongReading C R j A first y).2 = .inl .exhausted) :
    budgetOf C (strongPass C R A (first ++ List.ofFn y.1)).s.tree.paths.length
      ≤ (strongPass C R A (first ++ List.ofFn y.1)).used := by
  unfold strongReading at h
  simp only [] at h
  split_ifs at h with h1 h2 h3 h4 <;> first | assumption | simp at h

theorem exists_cons_of_ne_nil {β : Type*} {L : List β} {P : β → Prop} (h : L ≠ [])
    (hP : ∀ x ∈ L, P x) : ∃ x rest, L = x :: rest ∧ P x := by
  cases L with
  | nil => exact absurd rfl h
  | cons x rest => exact ⟨x, rest, rfl, hP x (by simp)⟩

theorem strongReading_rerun {j : ℕ} {A : RoundAcc α} {first lv : List (FreeMonoid α)}
    {y : C.Draws} (h : (strongReading C R j A first y).2 = .inr lv) :
    ∃ x rest, lv = x :: rest
      ∧ LiveEdge R (strongPass C R A (first ++ List.ofFn y.1)).s.tree
        (strongPass C R A (first ++ List.ofFn y.1)).s.edges C.k
        (givenUp C (strongPass C R A (first ++ List.ofFn y.1))) x := by
  unfold strongReading at h
  simp only [] at h
  split_ifs at h with h1 h2 h3 h4 <;> simp only [Sum.inr.injEq] at h
  all_goals
    subst h
    exact exists_cons_of_ne_nil h2 fun x hx => by
      obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hx
      have := (List.mem_filter.1 hi).2
      simp only [decide_eq_true_eq] at this
      exact this.2

theorem ev_congr {A B : RoundAcc α} (hs : A.s = B.s) (ha : A.att = B.att)
    (hsp : A.splits = B.splits) : ev C A = ev C B := by
  simp only [ev, phi, hs, ha, hsp]

theorem ev_le (A : RoundAcc α) (hI : StrongInv R A) :
    ev C A ≤ A.s.tree.paths.length - 2
      + C.mmax A.s.tree.paths.length * Fintype.card α * (2 * (A.s.tree.paths.length - 1) + 1) := by
  have := phi_le C A
  have := hI.2
  unfold ev
  omega

theorem budget_gap (hp1 : 1 ≤ C.K.patience) (hpn : C.K.patience ≤ C.nr) {n u r e : ℕ}
    (hn : 2 ≤ n) (hu : u ≤ C.K.patience * r + C.K.patience * e) (hr : r ≤ e + 1)
    (he : e ≤ n - 2 + C.mmax n * Fintype.card α * (2 * (n - 1) + 1)) : u < budgetOf C n := by
  obtain ⟨n', rfl⟩ : ∃ n', n = n' + 2 := ⟨n - 2, by omega⟩
  have h1 : n' + 2 - 1 = n' + 1 := by omega
  have h2 : n' + 2 - 2 = n' := by omega
  rw [h2, h1] at he
  unfold budgetOf readStar
  rw [h1, h2]
  set M := C.mmax (n' + 2) * Fintype.card α * (2 * (n' + 1) + 1)
  set p := C.K.patience
  have key : p * M < C.nr * (n' + 1 + M) :=
    calc p * M ≤ C.nr * M := Nat.mul_le_mul_right _ hpn
      _ < C.nr * (n' + 1 + M) := Nat.mul_lt_mul_of_pos_left (by omega) (by omega)
  refine lt_of_lt_of_le ?_ (le_max_right _ _)
  have hpr := Nat.mul_le_mul_left p hr
  have hpe := Nat.mul_le_mul_left p he
  nlinarith

theorem givenUp_congr {A B : RoundAcc α} (hs : A.s = B.s) (ha : A.att = B.att) :
    givenUp C A = givenUp C B := by
  funext e
  simp only [givenUp, hs, ha]

/-- The round's count: the budget holds at every gate, and the readings are at most what it has
done, plus one. -/
theorem round_count (hp1 : 1 ≤ C.K.patience) (hpn : C.K.patience ≤ C.nr)
    (hm : Monotone C.mmax) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws) (m : ℕ),
      StrongInv R A → A.used ≤ C.K.patience * m + C.K.patience * ev C A → m ≤ ev C A + 1 →
      (1 ≤ m → (∃ x rest, first = x :: rest
          ∧ LiveEdge R A.s.tree A.s.edges C.k (givenUp C A) x)
        ∧ A.used < budgetOf C A.s.tree.paths.length) →
      (strongRound C R n j A first d).1 ≠ .exhausted
        ∧ (∀ p ∈ (strongRound C R n j A first d).2.2, p.2 < budgetOf C p.1)
        ∧ m + (strongRound C R n j A first d).2.2.length
          ≤ ev C (strongRound C R n j A first d).2.1 + 1
        ∧ StrongInv R (strongRound C R n j A first d).2.1
  | 0, j, A, first, d, m, hI, _, hme, _ => by
    simp only [strongRound]
    exact ⟨by simp, by simp, by simpa using hme, hI⟩
  | n + 1, j, A, first, d, m, hI, hu0, hme, hfirst => by
    obtain ⟨hI', -, hev, hu, -, -⟩ := strongPass_inv C R hm A (first ++ List.ofFn (d 0).1) hI
    obtain ⟨hs, hat, hsp, hus⟩ := strongReading_fst C R j A first (d 0)
    have hevm : m ≤ ev C (strongPass C R A (first ++ List.ofFn (d 0).1)) := by
      rcases Nat.eq_zero_or_pos m with h0 | hm1
      · omega
      obtain ⟨⟨x, rest, rfl, hlive⟩, hb⟩ := hfirst hm1
      have := strongPass_live C R hm A x (rest ++ List.ofFn (d 0).1) hI hp1 hb hlive
      simp only [List.cons_append]
      omega
    set A' := strongPass C R A (first ++ List.ofFn (d 0).1) with hA'
    have hused : A'.used ≤ C.K.patience * (m + 1) + C.K.patience * ev C A' := by
      rw [Nat.mul_succ]
      omega
    have hn2 : 2 ≤ A'.s.tree.paths.length := by rw [← hI'.2]; omega
    have hb' : A'.used < budgetOf C A'.s.tree.paths.length :=
      budget_gap C hp1 hpn hn2 hused (by omega) (ev_le C R A' hI')
    have hevR := ev_congr C hs hat hsp
    have hIR : StrongInv R (strongReading C R j A first (d 0)).1 := by
      unfold StrongInv; rw [hs, hsp]; exact hI'
    have hex : (strongReading C R j A first (d 0)).2 ≠ .inl .exhausted := fun h => by
      have := strongReading_exhausted C R h
      rw [← hA'] at this
      omega
    have hrr := fun lv (h : (strongReading C R j A first (d 0)).2 = .inr lv) =>
      strongReading_rerun C R h
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩
    all_goals
      rw [hR] at hevR hIR hs hus hat hex
      dsimp only at hevR hIR hs hus hat hex
      simp only [strongRound, hR]
    · refine ⟨fun he => hex (by rw [he]), ?_, ?_, hIR⟩
      · simp only [List.mem_singleton]
        rintro p rfl
        simp only [hs, hus]
        exact hb'
      · simp only [List.length_singleton]
        rw [hevR]
        omega
    · obtain ⟨x, rest, hlv, hlive⟩ := hrr lv (by rw [hR])
      rw [← hA'] at hlive
      have ih := round_count hp1 hpn hm n (j + 1) A'' lv (Fin.tail d) (m + 1) hIR
        (by rw [hevR, hus]; exact hused) (by rw [hevR]; omega)
        (fun _ => ⟨⟨x, rest, hlv, by rw [hs, givenUp_congr C hs hat]; exact hlive⟩,
          by rw [hus, hs]; exact hb'⟩)
      refine ⟨ih.1, ?_, ?_, ih.2.2.2⟩
      · simp only [List.mem_cons]
        rintro p (rfl | hp)
        · simp only [hs, hus]
          exact hb'
        · exact ih.2.1 p hp
      · simp only [List.length_cons]
        have := ih.2.2.1
        omega

end Count

end OrthoDFA

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Round

variable (C : StrongCfg α) (R : CutReads α)

theorem startAcc_inv (seed : List (FreeMonoid α)) : StrongInv R (startAcc C R seed) :=
  ⟨closeEdges_learned C.K R fun _ _ _ _ he => by simp at he,
    by simp [startAcc, initialK, closeK, DTree.paths]⟩

theorem strongStep_inv (A : RoundAcc α) (x : FreeMonoid α) (hI : StrongInv R A) :
    StrongInv R (strongStep C R A x) := by
  obtain ⟨hl, hlen⟩ := hI
  obtain ⟨htr, hed, -⟩ := strongStep_state C R A x
  refine ⟨by rw [htr, hed]; exact probeStepK_learned C.K R hl, ?_⟩
  rcases strongStep_cases C R A x hl with ⟨hsp, ht, -⟩ | ⟨r, hsp, -, hrl, ht, -⟩
  · rw [hsp, ht]; exact hlen
  · rw [hsp, ht, DTree.splitAt_paths_length _ _ _ hrl, List.length_append]
    simp only [List.length_singleton]
    omega

/-- A property of the tree, edges and splits that every step keeps holds of the round's end. -/
theorem strongRound_preserves (P : RoundAcc α → Prop)
    (hcongr : ∀ A B : RoundAcc α, A.s.tree = B.s.tree → A.s.edges = B.s.edges →
      A.splits = B.splits → P A → P B)
    (hstep : ∀ A x, StrongInv R A → P A → P (strongStep C R A x)) :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws),
      StrongInv R A → P A →
        StrongInv R (strongRound C R n j A first d).2.1
          ∧ P (strongRound C R n j A first d).2.1 := by
  have hfold : ∀ (probes : List (FreeMonoid α)) (A : RoundAcc α), StrongInv R A → P A →
      StrongInv R (probes.foldl (passBody C R) A) ∧ P (probes.foldl (passBody C R) A) := by
    intro probes
    induction probes with
    | nil => exact fun A h1 h2 => ⟨h1, h2⟩
    | cons x xs ih =>
      intro A h1 h2
      simp only [List.foldl_cons]
      unfold passBody
      split_ifs
      · exact ih A h1 h2
      · exact ih _ (strongStep_inv C R A x h1) (hstep A x h1 h2)
  have hread : ∀ j A first (y : C.Draws), StrongInv R A → P A →
      StrongInv R (strongReading C R j A first y).1 ∧ P (strongReading C R j A first y).1 := by
    intro j A first y h1 h2
    obtain ⟨hs, -, hsp, -⟩ := strongReading_fst C R j A first y
    rw [strongPass_eq] at hs hsp
    obtain ⟨g1, g2⟩ := hfold (first ++ List.ofFn y.1) { A with s := { A.s with streak := 0 } } h1
      (hcongr A _ rfl rfl rfl h2)
    refine ⟨⟨?_, ?_⟩, hcongr _ _ (by rw [hs]) (by rw [hs]) (by rw [hsp]) g2⟩
    · rw [hs]; exact g1.1
    · rw [hs, hsp]; exact g1.2
  intro n
  induction n with
  | zero => exact fun _ A _ _ h1 h2 => ⟨h1, h2⟩
  | succ n ih =>
    intro j A first d h1 h2
    obtain ⟨g1, g2⟩ := hread j A first (d 0) h1 h2
    rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;>
      rw [hR] at g1 g2 <;> simp only [strongRound, hR]
    · exact ⟨g1, g2⟩
    · exact ih _ _ _ _ g1 g2

theorem round_strong_budget : RoundStrongBudget := by
  intro α _ _ C R seed Rmax d hp1 hpn hm
  obtain ⟨h1, h2, -⟩ := round_count C R hp1 hpn hm Rmax 0 (startAcc C R seed) [] d 0
    (startAcc_inv C R seed) (by simp [startAcc]) (Nat.zero_le _) (fun h => absurd h (by omega))
  exact ⟨h1, h2⟩

theorem round_strong_readings : RoundStrongReadings := by
  intro α _ _ C R seed Rmax d hp1 hpn hm
  obtain ⟨-, -, h3, h4⟩ := round_count C R hp1 hpn hm Rmax 0 (startAcc C R seed) [] d 0
    (startAcc_inv C R seed) (by simp [startAcc]) (Nat.zero_le _) (fun h => absurd h (by omega))
  have := ev_le C R _ h4
  have := h4.2
  unfold strongRun readStar
  omega

theorem round_strong_leaf_paths : RoundStrongLeafPaths := by
  intro α _ _ C R seed Rmax d
  have hl := (strongRound_preserves C R (fun _ => True) (fun _ _ _ _ _ _ => trivial)
    (fun _ _ _ _ => trivial) Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed) trivial).1.1
  intro p c q w he
  obtain ⟨h1, h2⟩ := hl p c q w he
  exact ⟨DTree.sift_mem_paths _ _ _ h1, DTree.sift_mem_paths _ _ _ h2⟩

theorem round_strong_same_state : RoundStrongSameState := by
  intro α _ _ Q C R A side rep seed Rmax d
  have h := (strongRound_preserves C R (fun B => ∀ r ∈ B.splits, SplitOK R r)
    (fun A B _ _ hsp hA => hsp ▸ hA) (fun A x hI hA => by
      rcases strongStep_cases C R A x hI.1 with ⟨hsp, -⟩ | ⟨r, hsp, -, -, -, -, -, hok⟩
      · rw [hsp]; exact hA
      · rw [hsp]
        intro r' hr'
        rcases List.mem_append.1 hr' with h | h
        · exact hA r' h
        · rw [List.mem_singleton.1 h]; exact hok)
    Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed) (by simp [startAcc])).2
  exact fun r hr hq => same_state_unfaithful R A side rep (h r hr) hq

theorem round_strong_leaves : RoundStrongLeaves := by
  classical
  intro α _ _ Q _ C R A side rep seed Rmax d
  have h := (strongRound_preserves C R
    (fun B => (∀ r ∈ B.splits, SplitOK R r) ∧ B.s.tree.paths.length
      ≤ classCount side rep B.s.tree + 2 + (B.splits.filter fun r => ¬ r.Separates side rep).length)
    (fun A B ht _ hsp hA => by rw [← ht, ← hsp]; exact hA) (fun A x hI hA => by
      obtain ⟨hok, hc⟩ := hA
      rcases strongStep_cases C R A x hI.1 with ⟨hsp, ht, -⟩ |
        ⟨r, hsp, hrt, hrl, ht, -, -, hokr⟩
      · rw [hsp, ht]; exact ⟨hok, hc⟩
      · rw [hsp, ht]
        refine ⟨fun r' hr' => ?_, ?_⟩
        · rcases List.mem_append.1 hr' with h | h
          · exact hok r' h
          · rw [List.mem_singleton.1 h]; exact hokr
        · have hcs := classCount_splitAt side rep r (hrt ▸ hrl)
          rw [hrt] at hcs
          rw [DTree.splitAt_paths_length _ _ _ hrl, List.filter_append, List.length_append]
          by_cases hs : r.Separates side rep
          · rw [if_pos hs] at hcs
            simp only [hs, not_true_eq_false, decide_false, List.filter_cons_of_neg,
              List.filter_nil, List.length_nil, Bool.false_eq_true, not_false_eq_true]
            omega
          · rw [if_neg hs] at hcs
            simp only [hs, not_false_eq_true, decide_true, List.filter_singleton, cond_true,
              List.length_singleton]
            omega)
    Rmax 0 (startAcc C R seed) [] d (startAcc_inv C R seed)
    ⟨by simp [startAcc], by simp [startAcc, initialK, closeK, DTree.paths]⟩).2
  obtain ⟨hok, hc⟩ := h
  have hcq := classCount_le side rep (strongRun C R seed Rmax d).2.1.s.tree
  have hle : ((strongRun C R seed Rmax d).2.1.splits.filter fun r => ¬ r.Separates side rep).length
      ≤ noisySplits R A side rep (strongRun C R seed Rmax d).2.1.splits := by
    unfold noisySplits
    rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
    refine List.countP_mono_left fun r hr h => ?_
    simp only [decide_eq_true_eq] at h ⊢
    exact noisy_of_not_separates R A side rep (hok r hr) h
  unfold strongRun at hle hcq ⊢
  omega

end Round

end OrthoDFA
