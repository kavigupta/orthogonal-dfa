import RLStar.LowLevelClaims.Trie
import RLStar.Proofs.DFABridge
import RLStar.Proofs.LoopDFA

/-!
# Local walk starts: the proofs

A probe neither searching nor stopping undecided at its start or end sift walks, learned all the
way, from its start's sift to its own (`agree_of_unchecked`), so the DFA, which walks it from the
same start, ends at the leaf its own sift reaches, on the root's side of its read. Every step of a
round keeps the tree's root, or splits below it.
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory Query

variable {α : Type*}

namespace Random

theorem walkStartIn_le (X : Finset (FreeMonoid α)) (x : FreeMonoid α) :
    walkStartIn X x ≤ x.toList.length := by
  classical
  unfold walkStartIn
  rcases h : (List.range x.toList.length).find? (fun k => decide (prefixOf x k ∉ X)) with _ | k
  · simp
  · have := List.mem_range.1 (List.mem_of_find?_eq_some h)
    simp only [Option.getD_some]
    omega

end Random

open Random

variable [Fintype α] [DecidableEq α]

omit [DecidableEq α] in
/-- The root of a tree reading `ε` decides a string the way its read does. -/
theorem head_of_sift {read : FreeMonoid α → ARU} {r a : DTree α} {x : FreeMonoid α}
    {e : List Bool} (h : (DTree.node 1 r a).sift read x = .inl e) :
    (e.head? = some true ↔ read x = .accept) := by
  rw [DTree.sift_node, mul_one] at h
  rcases hr : read x with _ | _ | _ <;> rw [hr] at h
  · rcases ha : a.sift read x with l | z <;> rw [ha] at h <;> simp at h
    subst h; simp
  · rcases hr' : r.sift read x with l | z <;> rw [hr'] at h <;> simp at h
    subst h; simp
  · simp at h

theorem trieProbeAgreeCorrect : TrieProbeAgreeCorrect := by
  intro α _ _ read H X hroot x hx
  obtain ⟨r, a, hT⟩ := hroot
  simp only [trieDisagreeSet, Set.mem_setOf_eq, Set.mem_union, not_or] at hx
  obtain ⟨p, e, hp, hf, he, hl⟩ :=
    agree_of_unchecked read H (walkStartIn_le X x) hx.1 hx.2
  refine ⟨p, hp, e, ?_, ?_⟩
  · rw [FreeMonoid.toList_ofList]
    exact walk_walkEnd H.edges p _ e hf hl
  · rw [hT] at he
    exact head_of_sift he

theorem trieSinglePieceCorrect : TrieSinglePieceCorrect := by
  intro α _ _ read H X hroot L D _
  classical
  have hsub : {x | ¬ H.TrieClassifies read X x (x ∈ L)}
      ⊆ trieDisagreeSet read X H ∪ misreadSet read L := by
    intro x hx
    by_contra hc
    simp only [Set.mem_union, not_or] at hc
    obtain ⟨a, ha, q, hq, hiff⟩ := trieProbeAgreeCorrect read H X hroot x hc.1
    have hm := hc.2
    simp only [misreadSet, Set.mem_setOf_eq, not_not] at hm
    refine hx ⟨a, ha, q, hq, ?_⟩
    rw [hiff, hm]
    by_cases hL : x ∈ L <;> simp [hL]
  exact (measureReal_mono hsub).trans (measureReal_union_le _ _)

section RootOne

variable (C : Level3Settings) (cap : ℕ) (read : FreeMonoid α → ARU)

omit [Fintype α] [DecidableEq α] in
theorem rootOne_applyTrieTests {s s' : TrieState α} (h : applyTrieTests C s = .inl s') :
    s'.tree = s.tree := by
  unfold applyTrieTests at h
  split at h
  · cases h
  · cases h; rfl

theorem rootOne_applyTrieProbe {s s' : TrieState α} (x : FreeMonoid α) (hs : RootOne s.tree)
    (h : (applyTrieProbeQ C cap s x).eval read = .inl s') : RootOne s'.tree := by
  unfold applyTrieProbeQ at h
  rw [eval_bind] at h
  generalize (probeOrAgreeQ s.tree s.edges (walkStartIn s.expanded x) x).eval read = o at h
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr, eval_pure, Sum.inl.injEq] at h
    subst h
    exact hs
  rw [if_neg hr] at h
  cases o with
  | unlearnedEdge p c u t => simp only [eval_pure, Sum.inl.injEq] at h; subst h; exact hs
  | wrongEdge p c u t =>
    simp only at h
    split at h
    · rw [eval_map, splitTrieLeafQ, eval_map, Sum.inl.injEq] at h
      subst h
      change RootOne (HypothesisDFA.split read ⟨s.tree, s.edges⟩ _ p).tree
      rw [HypothesisDFA.split_tree]
      exact rootOne_splitAt hs _ _
    · rw [eval_pure] at h; rw [rootOne_applyTrieTests C h]; exact hs
  | _ => rw [eval_pure] at h; rw [rootOne_applyTrieTests C h]; exact hs

theorem rootOne_runTrieProbes : ∀ (s : TrieState α) (xs : List (FreeMonoid α)),
    RootOne s.tree → RootOne ((runTrieProbesQ C cap s xs).eval read).1.tree
  | s, [], hs => hs
  | s, x :: xs, hs => by
    unfold runTrieProbesQ
    rw [eval_bind]
    rcases ho : (applyTrieProbeQ C cap s x).eval read with s' | e
    · exact rootOne_runTrieProbes s' xs (rootOne_applyTrieProbe C cap read x hs ho)
    · exact hs

end RootOne

section Expanded

variable (C : Level3Settings) (cap : ℕ) (read : FreeMonoid α → ARU)

open scoped Classical in
/-- The prefixes shorter than `cap` of the strings of `W`. -/
def shortPrefixes (W : Finset (FreeMonoid α)) : Finset (FreeMonoid α) :=
  (W ×ˢ Finset.range cap).image fun p => prefixOf p.1 p.2

omit [DecidableEq α] in
theorem mem_shortPrefixes {W : Finset (FreeMonoid α)} {x : FreeMonoid α} (hx : x ∈ W) {k : ℕ}
    (h : (prefixOf x k).toList.length < cap) : prefixOf x k ∈ shortPrefixes cap W := by
  unfold shortPrefixes
  refine Finset.mem_image.2 ⟨(x, (prefixOf x k).toList.length),
    Finset.mem_product.2 ⟨hx, Finset.mem_range.2 h⟩, ?_⟩
  simp only [prefixOf, FreeMonoid.toList_ofList]
  rw [← List.prefix_iff_eq_take.1 (List.take_prefix k x.toList)]

/-- The expanded prefixes are short prefixes of `W`, and every walk start seen is a prefix of a
string of `W`. -/
def TrieExpandedIn (W : Finset (FreeMonoid α)) (s : TrieState α) : Prop :=
  s.expanded ⊆ shortPrefixes cap W ∧ ∀ p ∈ s.seen, ∃ x ∈ W, ∃ k, p.2 = prefixOf x k

omit [Fintype α] [DecidableEq α] in
theorem applyTrieTests_inl {s s' : TrieState α} (h : applyTrieTests C s = .inl s') : s' = s := by
  unfold applyTrieTests at h
  split at h
  · cases h
  · cases h; rfl

omit [DecidableEq α] in
theorem expandedIn_skipRepeat {W : Finset (FreeMonoid α)} {s : TrieState α} {x : FreeMonoid α}
    (hx : x ∈ W) (o : Outcome α) (k : ℕ) (hs : TrieExpandedIn cap W s) :
    TrieExpandedIn cap W (s.skipRepeat cap o (prefixOf x k)) := by
  refine ⟨?_, hs.2⟩
  intro v hv
  simp only [TrieState.skipRepeat, Finset.mem_union, List.mem_toFinset, List.mem_filter,
    List.mem_cons, List.mem_map, decide_eq_true_eq] at hv
  rcases hv with hv | ⟨hv, hlen⟩
  · exact hs.1 hv
  rcases hv with rfl | ⟨p, hp, rfl⟩
  · exact mem_shortPrefixes cap hx hlen
  · obtain ⟨y, hy, j, hj⟩ := hs.2 p hp.1
    rw [hj] at hlen ⊢
    exact mem_shortPrefixes cap hy hlen

theorem expandedIn_applyTrieProbe {W : Finset (FreeMonoid α)} {s s' : TrieState α}
    {x : FreeMonoid α} (hx : x ∈ W) (hs : TrieExpandedIn cap W s)
    (h : (applyTrieProbeQ C cap s x).eval read = .inl s') : TrieExpandedIn cap W s' := by
  unfold applyTrieProbeQ at h
  rw [eval_bind] at h
  generalize (probeOrAgreeQ s.tree s.edges (walkStartIn s.expanded x) x).eval read = o at h
  by_cases hr : ∃ z ∈ harvestString o, z ∈ s.seen.map Prod.fst
  · rw [if_pos hr, eval_pure, Sum.inl.injEq] at h
    subst h
    exact expandedIn_skipRepeat cap hx o _ hs
  rw [if_neg hr] at h
  have hc : ∀ o', applyTrieTests C (countTrieOutcome s o' (prefixOf x (walkStartIn s.expanded x)))
      = .inl s' → TrieExpandedIn cap W s' := by
    intro o' h'
    rw [applyTrieTests_inl C h']
    refine ⟨hs.1, ?_⟩
    intro p hp
    simp only [countTrieOutcome, List.mem_append, List.mem_map] at hp
    rcases hp with hp | ⟨z, -, rfl⟩
    · exact hs.2 p hp
    · exact ⟨x, hx, _, rfl⟩
  cases o with
  | unlearnedEdge p c u t =>
    simp only [eval_pure, Sum.inl.injEq] at h; subst h
    exact ⟨hs.1, by simp [newTrieHypothesis]⟩
  | wrongEdge p c u t =>
    simp only at h
    split at h
    · rw [eval_map, splitTrieLeafQ, eval_map, Sum.inl.injEq] at h
      subst h
      exact ⟨hs.1, by simp [newTrieHypothesis]⟩
    · rw [eval_pure] at h; exact hc _ h
  | _ => rw [eval_pure] at h; exact hc _ h

theorem expandedIn_runTrieProbes (W : Finset (FreeMonoid α)) :
    ∀ (s : TrieState α) (xs : List (FreeMonoid α)), (∀ x ∈ xs, x ∈ W) →
      TrieExpandedIn cap W s → TrieExpandedIn cap W ((runTrieProbesQ C cap s xs).eval read).1
  | s, [], _, hs => hs
  | s, x :: xs, hW, hs => by
    unfold runTrieProbesQ
    rw [eval_bind]
    rcases ho : (applyTrieProbeQ C cap s x).eval read with s' | e
    · exact expandedIn_runTrieProbes W s' xs (fun y hy => hW y (List.mem_cons_of_mem _ hy))
        (expandedIn_applyTrieProbe C cap read (hW x List.mem_cons_self) hs ho)
    · exact hs

end Expanded

theorem level3TrieRootOne : Level3TrieRootOne := by
  intro α _ _ C cap read P draws
  have h := rootOne_runTrieProbes C cap read trieInitialState (List.ofFn draws) ⟨_, _, rfl⟩
  unfold indecisiveLStarLevel3Trie level3TrieQ
  rw [eval_map]
  exact h

theorem level3TrieExpandedCard : Level3TrieExpandedCard := by
  intro α _ _ C cap read P draws
  classical
  let W : Finset (FreeMonoid α) := Finset.univ.image draws
  have h := expandedIn_runTrieProbes C cap read W trieInitialState (List.ofFn draws)
    (fun x hx => by simpa [W] using hx)
    ⟨by simp [trieInitialState, newTrieHypothesis], by simp [trieInitialState, newTrieHypothesis]⟩
  have hsub : (indecisiveLStarLevel3Trie C cap read draws).expanded ⊆ shortPrefixes cap W := by
    unfold indecisiveLStarLevel3Trie level3TrieQ
    rw [eval_map]
    exact h.1
  calc _ ≤ (shortPrefixes cap W).card := Finset.card_le_card hsub
    _ ≤ (W ×ˢ Finset.range cap).card := Finset.card_image_le
    _ = W.card * cap := by rw [Finset.card_product, Finset.card_range]
    _ ≤ P * cap := Nat.mul_le_mul_right _ (Finset.card_image_le.trans (by simp))

end OrthoDFA

end
