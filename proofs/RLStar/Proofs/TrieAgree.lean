import RLStar.Proofs.RandomProbe
import RLStar.Proofs.RandomNoise
import RLStar.Code.LearnerLoopTrie

/-!
# A round with local walk starts is level 3's where its probes' first letters differ

Every string shorter than `walkStart` is expanded, so a probe at least that long walks from
`walkStart`, as level 3's does. Its harvest strings begin with its own first `walkStart` letters
(`pot_pre`), so where the probes' first `walkStart` letters are distinct no probe repeats a string,
nothing more is expanded, and every step is level 3's (`runTrieProbes_agree`).
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

/-- A harvest string of a probe walked from `k` begins with the probe's first `k` letters. -/
theorem harvestString_pre (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) {k : ℕ}
    {x z : FreeMonoid α} (hz : z ∈ harvestString (probeOrAgree read T E k x)) :
    prefixOf z k = prefixOf x k := by
  rcases List.mem_append.1 hz with h | h
  · exact (pot_pre (probeOrAgree_ends read T x E k z h).1).2
  · exact (pot_pre (probeOrAgree_pt read T x E k z h).1).2

variable [DecidableEq α]

variable (C : Level3Settings) (cap : ℕ) (read : FreeMonoid α → ARU)

theorem applyProbeQ_eq (s : Level3State α) (x : FreeMonoid α) :
    applyProbeQ C s x = (probeOrAgreeQ s.tree s.edges C.walkStart x).bind (level3StepQ C s) := rfl

/-- Probes at least `walkStart` long, each with first `walkStart` letters of its own and of none
of the strings read so far, run as level 3's. -/
theorem runTrieProbes_agree (K : ℕ) (hK : C.walkStart = K) :
    ∀ (xs : List (FreeMonoid α)) (t : TrieState α) (used : Finset (FreeMonoid α)),
      t.expanded = completeTrie K → (∀ r ∈ t.reads, prefixOf r.1 K ∈ used) →
      (∀ x ∈ xs, K ≤ x.toList.length ∧ prefixOf x K ∉ used) →
      (xs.map fun x => prefixOf x K).Nodup →
      ((runTrieProbesQ C cap t xs).eval read).1.base = ((runProbesQ C t.base xs).eval read).1
        ∧ ((runTrieProbesQ C cap t xs).eval read).1.expanded = completeTrie K
        ∧ ((runTrieProbesQ C cap t xs).eval read).2 = ((runProbesQ C t.base xs).eval read).2
        ∧ (runTrieProbesQ C cap t xs).reads read = (runProbesQ C t.base xs).reads read
  | [], t, _, hX, _, _, _ => ⟨rfl, hX, rfl, rfl⟩
  | x :: xs, t, used, hX, hr, hxs, hnd => by
    classical
    obtain ⟨hxK, hxu⟩ := hxs x List.mem_cons_self
    have hw : walkStartIn t.expanded x = C.walkStart := by
      rw [hX, hK]; exact walkStartIn_completeTrie hxK
    set o := probeOrAgree read t.base.tree t.base.edges C.walkStart x with ho
    have hnr : ¬ t.Repeats o := by
      rintro ⟨z, hz, hzr⟩
      obtain ⟨r, hr', rfl⟩ := List.mem_map.1 hzr
      have h1 := hr r hr'
      have h2 := harvestString_pre read _ _ hz
      rw [hK] at h2
      rw [h2] at h1
      exact hxu h1
    have hT : (applyTrieProbeQ C cap t x).eval read
        = Sum.map (t.record o (prefixOf x C.walkStart)) id
          ((level3StepQ C t.base o).eval read) := by
      have ho' : (probeOrAgreeQ t.base.tree t.base.edges C.walkStart x).eval read = o := rfl
      simp only [applyTrieProbeQ, eval_bind, hw, ho']
      rw [if_neg hnr, eval_map]
    have hM : (applyProbeQ C t.base x).eval read = (level3StepQ C t.base o).eval read := by
      rw [applyProbeQ_eq, eval_bind]
      rfl
    have hTr : (applyTrieProbeQ C cap t x).reads read
        = (probeOrAgreeQ t.base.tree t.base.edges C.walkStart x).reads read
          ∪ (level3StepQ C t.base o).reads read := by
      have ho' : (probeOrAgreeQ t.base.tree t.base.edges C.walkStart x).eval read = o := rfl
      simp only [applyTrieProbeQ, reads_bind, hw, ho']
      rw [if_neg hnr, reads_map]
    have hMr : (applyProbeQ C t.base x).reads read
        = (probeOrAgreeQ t.base.tree t.base.edges C.walkStart x).reads read
          ∪ (level3StepQ C t.base o).reads read := by
      rw [applyProbeQ_eq, reads_bind]
      rfl
    have e1r : (runTrieProbesQ C cap t (x :: xs)).reads read
        = (applyTrieProbeQ C cap t x).reads read ∪ match (applyTrieProbeQ C cap t x).eval read with
          | .inl s' => (runTrieProbesQ C cap s' xs).reads read
          | .inr _ => ∅ := by
      simp only [runTrieProbesQ, reads_bind]
      rcases (applyTrieProbeQ C cap t x).eval read <;> rfl
    have e2r : (runProbesQ C t.base (x :: xs)).reads read
        = (applyProbeQ C t.base x).reads read ∪ match (applyProbeQ C t.base x).eval read with
          | .inl s' => (runProbesQ C s' xs).reads read
          | .inr _ => ∅ := by
      simp only [runProbesQ, reads_bind]
      rcases (applyProbeQ C t.base x).eval read <;> rfl
    rw [e1r, e2r, hTr, hMr, hT, hM]
    have e1 : (runTrieProbesQ C cap t (x :: xs)).eval read
        = match (applyTrieProbeQ C cap t x).eval read with
          | .inl s' => (runTrieProbesQ C cap s' xs).eval read
          | .inr e => (t, e) := by
      simp only [runTrieProbesQ, eval_bind]
      rcases (applyTrieProbeQ C cap t x).eval read <;> rfl
    have e2 : (runProbesQ C t.base (x :: xs)).eval read
        = match (applyProbeQ C t.base x).eval read with
          | .inl s' => (runProbesQ C s' xs).eval read
          | .inr e => (t.base, e) := by
      simp only [runProbesQ, eval_bind]
      rcases (applyProbeQ C t.base x).eval read <;> rfl
    rw [e1, e2, hT, hM]
    rcases hstep : (level3StepQ C t.base o).eval read with b | e
    · simp only [Sum.map_inl]
      have hxs' : ∀ y ∈ xs, K ≤ y.toList.length ∧ prefixOf y K ∉ insert (prefixOf x K) used := by
        intro y hy
        refine ⟨(hxs y (List.mem_cons_of_mem _ hy)).1, ?_⟩
        rw [Finset.mem_insert, not_or]
        refine ⟨fun he => ?_, (hxs y (List.mem_cons_of_mem _ hy)).2⟩
        rw [List.map_cons, List.nodup_cons] at hnd
        exact hnd.1 (List.mem_map.2 ⟨y, hy, he⟩)
      have hnd' : (xs.map fun x => prefixOf x K).Nodup := (List.nodup_cons.1 hnd).2
      have hrec := runTrieProbes_agree K hK xs (t.record o (prefixOf x C.walkStart) b)
        (insert (prefixOf x K) used) (by unfold TrieState.record; split_ifs <;> exact hX)
        (by
          intro r hr'
          unfold TrieState.record at hr'
          split_ifs at hr' with hh
          · rcases List.mem_append.1 hr' with h | h
            · exact Finset.mem_insert_of_mem (hr r h)
            · obtain ⟨z, hz, rfl⟩ := List.mem_map.1 h
              rw [Finset.mem_insert]
              left
              have h2 := harvestString_pre read _ _ hz
              rw [hK] at h2
              exact h2
          · simp at hr')
        hxs' hnd'
      have hb : (t.record o (prefixOf x C.walkStart) b).base = b := by
        unfold TrieState.record; split_ifs <;> rfl
      rw [hb] at hrec
      exact ⟨hrec.1, hrec.2.1, hrec.2.2.1, by rw [hrec.2.2.2]⟩
    · exact ⟨rfl, hX, rfl, rfl⟩

/-- Probes at least `walkStart` long with distinct first `walkStart` letters run as level 3's, with
nothing expanded past the complete trie. -/
theorem level3Trie_agree {P : ℕ} (draws : Fin P → FreeMonoid α)
    (hlen : ∀ i, C.walkStart ≤ (draws i).toList.length)
    (hinj : Function.Injective fun i => prefixOf (draws i) C.walkStart) :
    indecisiveLStarLevel3Trie C cap read draws
      = ⟨(indecisiveLStarLevel3 C read draws).hypothesis, completeTrie C.walkStart,
        (indecisiveLStarLevel3 C read draws).ending⟩ := by
  have h := runTrieProbes_agree C cap read C.walkStart rfl (List.ofFn draws)
    (trieInitialState C) ∅ rfl (by simp [trieInitialState])
    (fun x hx => by
      obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
      exact ⟨hlen i, Finset.notMem_empty _⟩)
    (by
      rw [List.map_ofFn]
      exact List.nodup_ofFn.2 hinj)
  unfold indecisiveLStarLevel3Trie level3TrieQ indecisiveLStarLevel3 level3Q
  simp only [eval_map]
  obtain ⟨h1, h2, h3, -⟩ := h
  rw [h1, h2, h3]
  rfl

end Random

end OrthoDFA

end
