import RLStar.Proofs.RandomProbe
import RLStar.Proofs.RandomNoise
import RLStar.Code.LearnerLoopTrie

/-!
# A round with local walk starts is level 3's

Every string shorter than `walkStart` is expanded, so a probe at least that long walks from
`walkStart`, as level 3's does, and every step is level 3's (`runTrieProbes_agree`).
-/

noncomputable section

namespace OrthoDFA

open Query

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

theorem length_prefixOf' (x : FreeMonoid α) (k : ℕ) :
    (prefixOf x k).toList.length = min k x.toList.length := by
  simp [prefixOf]

/-- The walk start is `K` where every prefix shorter than `K` is expanded and the one of length `K`,
if shorter than `x`, is not. -/
theorem walkStartIn_eq {X : Finset (FreeMonoid α)} {x : FreeMonoid α} {K : ℕ}
    (hK : K ≤ x.toList.length) (hin : ∀ j < K, prefixOf x j ∈ X)
    (hout : K < x.toList.length → prefixOf x K ∉ X) : walkStartIn X x = K := by
  classical
  unfold walkStartIn
  rcases hlt : decide (K < x.toList.length) with _ | _
  · have hKe : K = x.toList.length := by
      have := of_decide_eq_false hlt; omega
    have hn : (List.range x.toList.length).find? (fun k => decide (prefixOf x k ∉ X)) = none := by
      rw [List.find?_eq_none]
      intro j hj
      simp only [decide_eq_true_eq, not_not]
      exact hin j (hKe ▸ List.mem_range.1 hj)
    rw [hn, Option.getD_none, hKe]
  · have hK' := of_decide_eq_true hlt
    have hs : (List.range x.toList.length).find? (fun k => decide (prefixOf x k ∉ X)) = some K := by
      rw [List.find?_eq_some_iff_getElem]
      refine ⟨by simpa using hout hK', K, by simpa using hK', by simp, fun j hj => ?_⟩
      simpa using hin j hj
    rw [hs, Option.getD_some]

variable [Fintype α]

theorem mem_completeTrie {K : ℕ} {u : FreeMonoid α} : u ∈ completeTrie K ↔ u.toList.length < K := by
  classical
  unfold completeTrie
  rw [Finset.mem_biUnion]
  constructor
  · rintro ⟨j, hj, hu⟩
    obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 hu
    simpa using Finset.mem_range.1 hj
  · intro h
    refine ⟨u.toList.length, Finset.mem_range.2 h, ?_⟩
    exact (mem_lenK (k := u.toList.length) (u := u)).2 rfl

theorem walkStartIn_completeTrie {K : ℕ} {x : FreeMonoid α} (hK : K ≤ x.toList.length) :
    walkStartIn (completeTrie K) x = K :=
  walkStartIn_eq hK (fun j hj => mem_completeTrie.2 (by rw [length_prefixOf']; omega))
    (fun h hm => by have := mem_completeTrie.1 hm; rw [length_prefixOf'] at this; omega)

theorem card_completeTrie_le [Nonempty α] (K : ℕ) :
    (completeTrie (α := α) K).card ≤ K * Fintype.card α ^ K := by
  classical
  unfold completeTrie
  refine (Finset.card_biUnion_le).trans ?_
  calc _ ≤ ∑ _j ∈ Finset.range K, Fintype.card α ^ K := Finset.sum_le_sum fun j hj => by
          refine Finset.card_image_le.trans ?_
          rw [Finset.card_univ, Fintype.card_fun, Fintype.card_fin]
          exact Nat.pow_le_pow_right Fintype.card_pos (Finset.mem_range.1 hj).le
    _ = K * Fintype.card α ^ K := by rw [Finset.sum_const, Finset.card_range, smul_eq_mul]

variable [DecidableEq α]

variable (C : Level3Settings) (read : FreeMonoid α → ARU)

theorem applyProbeQ_eq (s : Level3State α) (x : FreeMonoid α) :
    applyProbeQ C s x = (probeOrAgreeQ s.tree s.edges C.walkStart x).bind (level3StepQ C s) := rfl

/-- A probe at least `walkStart` long, past the complete trie, steps as level 3's. -/
theorem applyTrieProbe_agree (t : TrieState α) (x : FreeMonoid α)
    (hX : t.expanded = completeTrie C.walkStart) (hxK : C.walkStart ≤ x.toList.length) :
    (applyTrieProbeQ C t x).eval read
        = Sum.map (fun b => ⟨b, t.expanded⟩) id ((applyProbeQ C t.base x).eval read)
      ∧ (applyTrieProbeQ C t x).reads read = (applyProbeQ C t.base x).reads read := by
  have hw : walkStartIn t.expanded x = C.walkStart := by
    rw [hX]; exact walkStartIn_completeTrie hxK
  refine ⟨?_, ?_⟩
  · simp only [applyTrieProbeQ, applyProbeQ_eq, eval_bind, eval_map, hw]
  · simp only [applyTrieProbeQ, applyProbeQ_eq, reads_bind, reads_map, hw]

/-- Probes at least `walkStart` long, past the complete trie, run as level 3's. -/
theorem runTrieProbes_agree :
    ∀ (xs : List (FreeMonoid α)) (t : TrieState α),
      t.expanded = completeTrie C.walkStart → (∀ x ∈ xs, C.walkStart ≤ x.toList.length) →
      ((runTrieProbesQ C t xs).eval read).1.base = ((runProbesQ C t.base xs).eval read).1
        ∧ ((runTrieProbesQ C t xs).eval read).1.expanded = completeTrie C.walkStart
        ∧ ((runTrieProbesQ C t xs).eval read).2 = ((runProbesQ C t.base xs).eval read).2
        ∧ (runTrieProbesQ C t xs).reads read = (runProbesQ C t.base xs).reads read
  | [], t, hX, _ => ⟨rfl, hX, rfl, rfl⟩
  | x :: xs, t, hX, hxs => by
    obtain ⟨hA, hR⟩ := applyTrieProbe_agree C read t x hX (hxs x List.mem_cons_self)
    have e1r : (runTrieProbesQ C t (x :: xs)).reads read
        = (applyTrieProbeQ C t x).reads read ∪ match (applyTrieProbeQ C t x).eval read with
          | .inl s' => (runTrieProbesQ C s' xs).reads read
          | .inr _ => ∅ := by
      simp only [runTrieProbesQ, reads_bind]
      rcases (applyTrieProbeQ C t x).eval read <;> rfl
    have e2r : (runProbesQ C t.base (x :: xs)).reads read
        = (applyProbeQ C t.base x).reads read ∪ match (applyProbeQ C t.base x).eval read with
          | .inl s' => (runProbesQ C s' xs).reads read
          | .inr _ => ∅ := by
      simp only [runProbesQ, reads_bind]
      rcases (applyProbeQ C t.base x).eval read <;> rfl
    have e1 : (runTrieProbesQ C t (x :: xs)).eval read
        = match (applyTrieProbeQ C t x).eval read with
          | .inl s' => (runTrieProbesQ C s' xs).eval read
          | .inr e => (t, e) := by
      simp only [runTrieProbesQ, eval_bind]
      rcases (applyTrieProbeQ C t x).eval read <;> rfl
    have e2 : (runProbesQ C t.base (x :: xs)).eval read
        = match (applyProbeQ C t.base x).eval read with
          | .inl s' => (runProbesQ C s' xs).eval read
          | .inr e => (t.base, e) := by
      simp only [runProbesQ, eval_bind]
      rcases (applyProbeQ C t.base x).eval read <;> rfl
    rw [e1r, e2r, e1, e2, hR, hA]
    rcases (applyProbeQ C t.base x).eval read with b | e
    · have ih := runTrieProbes_agree xs ⟨b, t.expanded⟩ hX
        fun y hy => hxs y (List.mem_cons_of_mem _ hy)
      simp only [Sum.map_inl]
      exact ⟨ih.1, ih.2.1, ih.2.2.1, by rw [ih.2.2.2]⟩
    · exact ⟨rfl, hX, rfl, rfl⟩

/-- Probes at least `walkStart` long run as level 3's, with nothing expanded past the complete
trie. -/
theorem level3Trie_agree {P : ℕ} (draws : Fin P → FreeMonoid α)
    (hlen : ∀ i, C.walkStart ≤ (draws i).toList.length) :
    indecisiveLStarLevel3Trie C read draws
      = ⟨(indecisiveLStarLevel3 C read draws).hypothesis, completeTrie C.walkStart,
        (indecisiveLStarLevel3 C read draws).ending⟩ := by
  have h := runTrieProbes_agree C read (List.ofFn draws) (trieInitialState C) rfl
    (fun x hx => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      exact hlen i)
  unfold indecisiveLStarLevel3Trie level3TrieQ indecisiveLStarLevel3 level3Q
  simp only [eval_map]
  obtain ⟨h1, h2, h3, -⟩ := h
  rw [h1, h2, h3]
  rfl

end Random

end OrthoDFA

end
