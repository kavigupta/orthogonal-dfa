import RLStar.LowLevelClaims.Trie
import RLStar.Proofs.DFABridge
import RLStar.Proofs.LazyArith
import RLStar.Proofs.TrieAgree

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

variable (C : Level3Settings) (read : FreeMonoid α → ARU)

omit [Fintype α] in
theorem applyTests_inl {s s' : Level3State α} (h : applyTests C s = .inl s') : s' = s := by
  unfold applyTests at h
  split at h
  · cases h
  · cases h; rfl

theorem rootOne_level3Step {s : Level3State α} {s' : Level3State α} (o : Outcome α)
    (hs : RootOne s.tree) (h : (level3StepQ C s o).eval read = .inl s') : RootOne s'.tree := by
  unfold level3StepQ at h
  cases o with
  | unlearnedEdge p c u t => simp only [eval_pure, Sum.inl.injEq] at h; subst h; exact hs
  | wrongEdge p c u t =>
    simp only at h
    split at h
    · rw [eval_map, splitLeafQ, eval_map, Sum.inl.injEq] at h
      subst h
      change RootOne (HypothesisDFA.split read ⟨s.tree, s.edges⟩ _ p).tree
      rw [HypothesisDFA.split_tree]
      exact rootOne_splitAt hs _ _
    · rw [eval_pure] at h; rw [applyTests_inl C h]; exact hs
  | _ => rw [eval_pure] at h; rw [applyTests_inl C h]; exact hs

theorem rootOne_applyTrieProbe {s s' : TrieState α} (x : FreeMonoid α) (hs : RootOne s.base.tree)
    (h : (applyTrieProbeQ C s x).eval read = .inl s') : RootOne s'.base.tree := by
  unfold applyTrieProbeQ at h
  rw [eval_bind, eval_map] at h
  generalize (probeOrAgreeQ s.base.tree s.base.edges (walkStartIn s.expanded x) x).eval read = o
    at h
  rcases hb : (level3StepQ C s.base o).eval read with b | e
  · rw [hb, Sum.map_inl, Sum.inl.injEq] at h
    subst h
    exact rootOne_level3Step C read o hs hb
  · rw [hb] at h; cases h

theorem rootOne_runTrieProbes : ∀ (s : TrieState α) (xs : List (FreeMonoid α)),
    RootOne s.base.tree → RootOne ((runTrieProbesQ C s xs).eval read).1.base.tree
  | s, [], hs => hs
  | s, x :: xs, hs => by
    unfold runTrieProbesQ
    rw [eval_bind]
    rcases ho : (applyTrieProbeQ C s x).eval read with s' | e
    · exact rootOne_runTrieProbes s' xs (rootOne_applyTrieProbe C read x hs ho)
    · exact hs

end RootOne

theorem level3TrieRootOne : Level3TrieRootOne := by
  intro α _ _ C read P draws
  have h := rootOne_runTrieProbes C read (trieInitialState C) (List.ofFn draws)
    ⟨_, _, rfl⟩
  unfold indecisiveLStarLevel3Trie level3TrieQ
  rw [eval_map]
  exact h

end OrthoDFA

end
