import OrthoDFA.Utils.CoveredStart
import OrthoDFA.Proofs.RandomTree
import OrthoDFA.Proofs.TripartiteSearch

/-!
# True leaves

A state's true leaf is where its strings sift when no read on the way is undecided or wrong. A
split between two real targets parts two states at its leaf, adding a true leaf, so the leaves
past the first are at most `|Q|`.
-/

namespace OrthoDFA

namespace Random

variable {α : Type*}

theorem midfixAt_mem : ∀ (T : DTree α) (q : List Bool), T.midfixAt q = 1 ∨ T.midfixAt q ∈ mids T
  | .leaf, _ => Or.inl rfl
  | .node m r a, [] => Or.inr (by classical simp [DTree.midfixAt, mids])
  | .node m r a, false :: q => by
    classical
    rcases midfixAt_mem r q with h | h
    · exact Or.inl h
    · exact Or.inr (by simp [DTree.midfixAt, mids, h])
  | .node m r a, true :: q => by
    classical
    rcases midfixAt_mem a q with h | h
    · exact Or.inl h
    · exact Or.inr (by simp [DTree.midfixAt, mids, h])

open scoped Classical in
theorem mids_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), mids (T.splitAt d p) ⊆ insert d (mids T)
  | .leaf, [] => by classical simp [DTree.splitAt, mids]
  | .leaf, _ :: _ => by classical intro m hm; simp_all [DTree.splitAt, mids]
  | .node m r a, [] => by classical intro m' hm; simp_all [DTree.splitAt, mids]
  | .node m r a, false :: p => by
    classical
    intro m' hm
    have := mids_splitAt d r p
    simp only [DTree.splitAt, mids, Finset.mem_insert, Finset.mem_union] at hm ⊢
    rcases hm with h | h | h
    · exact Or.inr (Or.inl h)
    · rcases Finset.mem_insert.1 (this h) with h' | h'
      · exact Or.inl h'
      · exact Or.inr (Or.inr (Or.inl h'))
    · exact Or.inr (Or.inr (Or.inr h))
  | .node m r a, true :: p => by
    classical
    intro m' hm
    have := mids_splitAt d a p
    simp only [DTree.splitAt, mids, Finset.mem_insert, Finset.mem_union] at hm ⊢
    rcases hm with h | h | h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (Or.inl h))
    · rcases Finset.mem_insert.1 (this h) with h' | h'
      · exact Or.inl h'
      · exact Or.inr (Or.inr (Or.inr h'))

section True

variable {σ : Type*} (M : DFA α σ) (side : σ → Bool)

/-- Some state at the leaf `p` reaches the target `t` under `c`. -/
def RealT (T : DTree α) (p : List Bool) (c : α) (t : List Bool) : Prop :=
  ∃ q, leafOf M side T q = p ∧ leafOf M side T (M.evalFrom q [c]) = t

/-- Two states at the leaf `p` part at the midfix `d`. -/
def GenuineSplit (T : DTree α) (p : List Bool) (d : FreeMonoid α) : Prop :=
  ∃ q₁ q₂, leafOf M side T q₁ = p ∧ leafOf M side T q₂ = p ∧
    side (M.evalFrom q₁ d.toList) ≠ side (M.evalFrom q₂ d.toList)

theorem eval_mul (z m : FreeMonoid α) :
    M.eval (z * m).toList = M.evalFrom (M.eval z.toList) m.toList := by
  simp [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]

theorem leafOf_mem : ∀ (T : DTree α) (q : σ), leafOf M side T q ∈ T.leaves
  | .leaf, q => by simp [leafOf, DTree.leaves]
  | .node m r a, q => by
    unfold leafOf
    split_ifs
    · simp [DTree.leaves, leafOf_mem a q]
    · simp [DTree.leaves, leafOf_mem r q]

/-- A sift with no wrong read on its way that ends at a leaf ends at the true leaf. -/
theorem sift_true (read : FreeMonoid α → ARU) : ∀ (T : DTree α) (z : FreeMonoid α) (p : List Bool),
    T.sift read z = .inl p →
    (∀ m ∈ mids T, read (z * m) ≠ if side (M.eval (z * m).toList) then .reject else .accept) →
    p = leafOf M side T (M.eval z.toList)
  | .leaf, z, p, h, _ => by simp only [DTree.sift, Sum.inl.injEq] at h; simp [leafOf, ← h]
  | .node m r a, z, p, h, hw => by
    classical
    have hm := hw m (by simp [mids])
    have hsub : ∀ T', (T' = r ∨ T' = a) → ∀ m' ∈ mids T', read (z * m') ≠
        if side (M.eval (z * m').toList) then .reject else .accept :=
      fun T' hT' m' hm' => hw m' (by rcases hT' with rfl | rfl <;> simp [mids, hm'])
    simp only [DTree.sift] at h
    unfold leafOf
    rw [← eval_mul]
    split at h
    · rename_i hr
      have hs : side (M.eval (z * m).toList) = true := by
        cases hsd : side (M.eval (z * m).toList)
        · exact absurd (by rw [hr, hsd]; simp) hm
        · rfl
      rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rw [if_pos hs, ← h, sift_true read a z q hq (hsub a (Or.inr rfl))]
    · rename_i hr
      have hs : side (M.eval (z * m).toList) = false := by
        cases hsd : side (M.eval (z * m).toList)
        · rfl
        · exact absurd (by rw [hr, hsd]; simp) hm
      rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rw [if_neg (by rw [hs]; exact Bool.false_ne_true), ← h,
        sift_true read r z q hq (hsub r (Or.inl rfl))]
    · simp at h

/-- Two states at different true leaves part at the node where their paths part. -/
theorem leafOf_part : ∀ (T : DTree α) (q₁ q₂ : σ), leafOf M side T q₁ ≠ leafOf M side T q₂ →
    side (M.evalFrom q₁
        (T.midfixAt (commonPrefix (leafOf M side T q₁) (leafOf M side T q₂))).toList)
      ≠ side (M.evalFrom q₂ (T.midfixAt (commonPrefix (leafOf M side T q₁)
          (leafOf M side T q₂))).toList)
  | .leaf, q₁, q₂, h => by simp [leafOf] at h
  | .node m r a, q₁, q₂, h => by
    unfold leafOf at h ⊢
    by_cases h1 : side (M.evalFrom q₁ m.toList) <;> by_cases h2 : side (M.evalFrom q₂ m.toList)
    · rw [if_pos h1, if_pos h2] at h ⊢
      have h' : leafOf M side a q₁ ≠ leafOf M side a q₂ := fun e => h (by rw [e])
      simpa [commonPrefix, DTree.midfixAt] using leafOf_part a q₁ q₂ h'
    · rw [if_pos h1, if_neg h2]; simp [commonPrefix, DTree.midfixAt, h1, h2]
    · rw [if_neg h1, if_pos h2]; simp [commonPrefix, DTree.midfixAt, h1, h2]
    · rw [if_neg h1, if_neg h2] at h ⊢
      have h' : leafOf M side r q₁ ≠ leafOf M side r q₂ := fun e => h (by rw [e])
      simpa [commonPrefix, DTree.midfixAt] using leafOf_part r q₁ q₂ h'

/-- A split between two real targets is genuine. -/
theorem genuine_of_real {T : DTree α} {p t t₀ : List Bool} {c : α} (ht : RealT M side T p c t)
    (ht₀ : RealT M side T p c t₀) (hne : t ≠ t₀) :
    GenuineSplit M side T p (FreeMonoid.of c * T.midfixAt (commonPrefix t t₀)) := by
  obtain ⟨q₁, h1, h1'⟩ := ht
  obtain ⟨q₂, h2, h2'⟩ := ht₀
  refine ⟨q₁, q₂, h1, h2, ?_⟩
  have := leafOf_part M side T (M.evalFrom q₁ [c]) (M.evalFrom q₂ [c]) (by rw [h1', h2']; exact hne)
  rw [h1', h2'] at this
  simpa [FreeMonoid.toList_mul, DFA.evalFrom_of_append] using this

theorem leafOf_splitAt (d : FreeMonoid α) : ∀ (T : DTree α) (p : List Bool), p ∈ T.leaves →
    ∀ q, leafOf M side (T.splitAt d p) q = if leafOf M side T q = p
      then p ++ [side (M.evalFrom q d.toList)] else leafOf M side T q
  | .leaf, p, hp, q => by
    simp only [DTree.leaves, List.mem_singleton] at hp
    subst hp
    cases h : side (M.evalFrom q d.toList) <;> simp [DTree.splitAt, leafOf, h]
  | .node m r a, [], hp, q => by simp [DTree.leaves] at hp
  | .node m r a, false :: p, hp, q => by
    have hp' : p ∈ r.leaves := by simpa [DTree.leaves] using hp
    simp only [DTree.splitAt, leafOf]
    by_cases hs : side (M.evalFrom q m.toList)
    · simp [hs]
    · simp only [hs, Bool.false_eq_true, if_false, List.cons.injEq, true_and,
        leafOf_splitAt d r p hp' q]
      split_ifs <;> simp
  | .node m r a, true :: p, hp, q => by
    have hp' : p ∈ a.leaves := by simpa [DTree.leaves] using hp
    simp only [DTree.splitAt, leafOf]
    by_cases hs : side (M.evalFrom q m.toList)
    · simp only [hs, if_true, List.cons.injEq, true_and, leafOf_splitAt d a p hp' q]
      split_ifs <;> simp
    · simp [hs]

variable [Fintype σ]

open scoped Classical in
/-- The true leaves. -/
noncomputable def img (T : DTree α) : Finset (List Bool) :=
  (Finset.univ : Finset σ).image (leafOf M side T)

theorem img_card_le (T : DTree α) : (img M side T).card ≤ Fintype.card σ :=
  Finset.card_image_le.trans (by simp)

/-- A genuine split adds a true leaf. -/
theorem img_card_splitAt {T : DTree α} {p : List Bool} {d : FreeMonoid α} (hp : p ∈ T.leaves)
    (hg : GenuineSplit M side T p d) :
    (img M side T).card + 1 ≤ (img M side (T.splitAt d p)).card := by
  classical
  obtain ⟨q₁, q₂, h1, h2, h12⟩ := hg
  have hpimg : p ∈ img M side T := Finset.mem_image.2 ⟨q₁, Finset.mem_univ _, h1⟩
  have hsub : ((img M side T).erase p) ∪ {p ++ [true], p ++ [false]}
      ⊆ img M side (T.splitAt d p) := by
    intro y hy
    rcases Finset.mem_union.1 hy with hy | hy
    · obtain ⟨hyp, hy⟩ := Finset.mem_erase.1 hy
      obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hy
      refine Finset.mem_image.2 ⟨q, Finset.mem_univ _, ?_⟩
      rw [leafOf_splitAt M side d T p hp q, if_neg hyp]
    · have hq : ∀ q, leafOf M side T q = p →
          p ++ [side (M.evalFrom q d.toList)] ∈ img M side (T.splitAt d p) := fun q hq =>
        Finset.mem_image.2 ⟨q, Finset.mem_univ _, by
          rw [leafOf_splitAt M side d T p hp q, if_pos hq]⟩
      simp only [Finset.mem_insert, Finset.mem_singleton] at hy
      have hb : ∀ b : Bool, ∃ q, leafOf M side T q = p ∧ side (M.evalFrom q d.toList) = b := by
        intro b
        by_cases hs1 : side (M.evalFrom q₁ d.toList) = b
        · exact ⟨q₁, h1, hs1⟩
        · refine ⟨q₂, h2, ?_⟩
          cases b <;> cases h1' : side (M.evalFrom q₁ d.toList) <;>
            cases h2' : side (M.evalFrom q₂ d.toList) <;> simp_all
      rcases hy with rfl | rfl
      · obtain ⟨q, hq1, hq2⟩ := hb true
        rw [← hq2]; exact hq q hq1
      · obtain ⟨q, hq1, hq2⟩ := hb false
        rw [← hq2]; exact hq q hq1
  have hdisj : Disjoint ((img M side T).erase p) {p ++ [true], p ++ [false]} := by
    rw [Finset.disjoint_left]
    intro y hy hy'
    obtain ⟨-, hy⟩ := Finset.mem_erase.1 hy
    obtain ⟨q, -, rfl⟩ := Finset.mem_image.1 hy
    have := leafOf_mem M side T q
    simp only [Finset.mem_insert, Finset.mem_singleton] at hy'
    rcases hy' with h | h <;> rw [h] at this <;> exact DTree.snoc_not_mem_leaves T p hp _ this
  have h2 : ({p ++ [true], p ++ [false]} : Finset (List Bool)).card = 2 := by
    rw [Finset.card_insert_of_notMem (by simp), Finset.card_singleton]
  have := Finset.card_le_card hsub
  rw [Finset.card_union_of_disjoint hdisj, h2, Finset.card_erase_of_mem hpimg] at this
  have := Finset.card_pos.2 ⟨p, hpimg⟩
  omega

end True

section Probe

variable [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (read : FreeMonoid α → ARU)

omit [DecidableEq α] [Fintype α] [Fintype σ] in
theorem searchOutcome_prefix (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)
    (un : Option (α × List Bool)) (lo hi : ℕ) (hle : lo ≤ hi) {p : List Bool} {c : α}
    {u : FreeMonoid α} {t : List Bool}
    (h : searchOutcome read T x st un hi (tripartiteSearch false (siftAgrees read T x st) lo hi).1
      = .wrongEdge p c u t
      ∨ searchOutcome read T x st un hi (tripartiteSearch false (siftAgrees read T x st) lo hi).1
      = .unlearnedEdge p c u t) :
    (∃ q, lo < q ∧ q ≤ hi ∧ u = prefixOf x (q - 1) ∧ x.toList[q - 1]? = some c)
      ∨ (un = some (c, t) ∧ u = prefixOf x hi) := by
  have hs := tripartiteSearch_ok (recover := false) (agrees := siftAgrees read T x st) hle
  generalize (tripartiteSearch false (siftAgrees read T x st) lo hi).1 = r at hs h
  cases r with
  | edge q =>
    obtain ⟨h1, h2, -, -⟩ := hs
    simp only [searchOutcome] at h
    split at h
    · rename_i c' t' hc ht
      simp only [Outcome.wrongEdge.injEq, reduceCtorEq, or_false] at h
      obtain ⟨-, rfl, rfl, -⟩ := h
      exact Or.inl ⟨q, h1, h2, rfl, hc⟩
    · simp at h
  | allAgree =>
    simp only [searchOutcome] at h
    split at h
    · rename_i c' t'
      simp only [reduceCtorEq, Outcome.unlearnedEdge.injEq, false_or] at h
      obtain ⟨-, rfl, rfl, rfl⟩ := h
      exact Or.inr ⟨rfl, rfl⟩
    · simp at h
  | _ => simp [searchOutcome] at h

omit [DecidableEq α] [Fintype α] [Fintype σ] in
/-- The prefix and letter of an edge a probe learns or disagrees at are the probe's, past its
first `k` letters. -/
theorem probe_prefix (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) {p : List Bool}
    {c : α} {u : FreeMonoid α} {t : List Bool}
    (h : probeOrAgree read T E k x = .wrongEdge p c u t
      ∨ probeOrAgree read T E k x = .unlearnedEdge p c u t) :
    ∃ i, k ≤ i ∧ i < x.toList.length ∧ u = prefixOf x i ∧ x.toList[i]? = some c := by
  have hk : k ≤ x.toList.length := by
    by_contra hc
    simp [probeOrAgree, show x.toList.length < k by omega] at h
  have h' : probe read T E k x = .wrongEdge p c u t
      ∨ probe read T E k x = .unlearnedEdge p c u t := by
    simpa [probeOrAgree, show ¬ x.toList.length < k by omega] using h
  clear h
  unfold probe at h'
  rcases ha : T.sift read (prefixOf x k) with a | z₀
  · rw [ha] at h'
    simp only at h'
    have hpos := length_walk_pos E a (x.toList.drop k)
    have hlt : ∀ q, x.toList[q]? = some c → q < x.toList.length := fun q hc => by
      by_contra hcon
      rw [List.getElem?_eq_none (by omega)] at hc
      simp at hc
    split at h'
    · rename_i c' hc'
      split at h'
      · simp at h'
      · rcases searchOutcome_prefix read T x _ _ k _ (by omega) h' with
          ⟨q, h1, -, h3, h4⟩ | ⟨h3, h4⟩
        · exact ⟨q - 1, by omega, hlt _ h4, h3, h4⟩
        · simp only [Option.some.injEq, Prod.mk.injEq] at h3
          obtain ⟨rfl, -⟩ := h3
          exact ⟨_, by omega, hlt _ hc', h4, hc'⟩
    · rcases searchOutcome_prefix read T x _ _ k _ (by omega) h' with
        ⟨q, h1, -, h3, h4⟩ | ⟨h3, -⟩
      · exact ⟨q - 1, by omega, hlt _ h4, h3, h4⟩
      · simp at h3
  · rw [ha] at h'
    simp at h'

end Probe

end Random

end OrthoDFA
