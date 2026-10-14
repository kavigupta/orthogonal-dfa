import OrthoDFA.Level3Claim
import OrthoDFA.Proofs.IdealSteps
import OrthoDFA.Proofs.TripartiteSearch

/-!
# The tree and the steps

A state's true leaf is where its strings sift when no read on the way is undecided or wrong. A
target is real at an edge where some state at the edge's leaf reaches it under the edge's letter,
and a split between two real targets parts two states at its leaf, adding a true leaf (`GInv`), so
the leaves past the first are at most `|Q|`. A string is clean where no read of it followed by a
string of length at most `|Q|` is wrong, and every midfix of a tree in `classSet |Q|` is that
short. A probe that cannot read a string wrong (outside `PotWrong`) has clean prefixes, so the
edge it learns or the split it makes has a real target; every learned edge keeps a clean witness,
so its target stays real across splits. Every split is on a letter and the midfix where two
leaves part, so the tree is in `classSet |Q|`.

A step that keeps the hypothesis counts one more probe into the stretch; one that changes it
starts a fresh stretch and lowers `Ψ`, #417's count of the splits and learnings left.
-/

namespace OrthoDFA

namespace Random

open OrthoDFA.Ideal (probe probeOrAgree Outcome retarget EdgesOK OutcomeOK searchOutcome siftAgrees)

variable {α : Type*}

/-- No read is on its state's wrong side. -/
def NoWrong {σ : Type*} (M : DFA α σ) (side : σ → Bool) (read : FreeMonoid α → ARU) : Prop :=
  ∀ z, read z ≠ if side (M.eval z.toList) then .reject else .accept

open scoped Classical in
/-- The midfixes of the tree's nodes. -/
noncomputable def mids : DTree α → Finset (FreeMonoid α)
  | .leaf => ∅
  | .node m r a => insert m (mids r ∪ mids a)

open scoped Classical in
/-- The strings a probe of `x` can read: a prefix of length at least `k` followed by a string of
`Ms`. -/
noncomputable def potM (k : ℕ) (x : FreeMonoid α) (Ms : Finset (FreeMonoid α)) :
    Finset (FreeMonoid α) :=
  (Finset.Icc k x.toList.length).biUnion fun p => Ms.image (prefixOf x p * ·)

/-- The strings a probe of `x` can read under `T`. -/
noncomputable def pot (k : ℕ) (x : FreeMonoid α) (T : DTree α) : Finset (FreeMonoid α) :=
  potM k x (mids T)

theorem mem_potM {k p : ℕ} {x : FreeMonoid α} {Ms : Finset (FreeMonoid α)} {m : FreeMonoid α}
    (h1 : k ≤ p) (h2 : p ≤ x.toList.length) (hm : m ∈ Ms) : prefixOf x p * m ∈ potM k x Ms := by
  classical
  unfold potM
  exact Finset.mem_biUnion.2 ⟨p, Finset.mem_Icc.2 ⟨h1, h2⟩, Finset.mem_image.2 ⟨m, hm, rfl⟩⟩

/-- The probes that can read a string at a good read-state undecided. -/
def PotGood {σ : Type*} (M : DFA α σ) (U : σ → ℝ) (θ : ℝ) (k : ℕ) (read : FreeMonoid α → ARU)
    (T : DTree α) : Set (FreeMonoid α) :=
  {x | ∃ z ∈ pot k x T, ¬ BadAt U θ (M.eval z.toList) ∧ read z = .undecided}

open scoped Classical in
/-- The trees at most `n` splits reach, each at a leaf on a letter and the midfix where two
leaves part. -/
noncomputable def classSet [Fintype α] : ℕ → Finset (DTree α)
  | 0 => {.node 1 .leaf .leaf}
  | n + 1 => classSet n ∪ (classSet n).biUnion fun T =>
      (T.leaves.toFinset ×ˢ (Finset.univ : Finset α) ×ˢ T.leaves.toFinset ×ˢ
        T.leaves.toFinset).image
        fun q => T.splitAt (FreeMonoid.of q.2.1 * T.midfixAt (commonPrefix q.2.2.1 q.2.2.2)) q.1

/-! ## Short strings -/

section Short

variable [Fintype α]

/-- The strings of length `k`. -/
noncomputable def lenK (k : ℕ) : Finset (FreeMonoid α) :=
  (Finset.univ : Finset (Fin k → α)).image fun v => FreeMonoid.ofList (List.ofFn v)

theorem mem_lenK {k : ℕ} {u : FreeMonoid α} : u ∈ lenK k ↔ u.toList.length = k := by
  constructor
  · intro h
    obtain ⟨v, -, rfl⟩ := Finset.mem_image.1 h
    simp
  · intro h
    refine Finset.mem_image.2 ⟨fun i => u.toList.get ⟨i, by omega⟩, Finset.mem_univ _, ?_⟩
    apply FreeMonoid.toList.injective
    simp only [FreeMonoid.toList_ofList]
    apply List.ext_get (by simp [h])
    intro i h1 h2
    simp

/-- The strings of length at most `L`. -/
noncomputable def upTo (L : ℕ) : Finset (FreeMonoid α) := (Finset.range (L + 1)).biUnion lenK

theorem mem_upTo {L : ℕ} {u : FreeMonoid α} : u ∈ upTo L ↔ u.toList.length ≤ L := by
  classical
  simp only [upTo, Finset.mem_biUnion, Finset.mem_range, mem_lenK]
  constructor
  · rintro ⟨j, hj, rfl⟩; omega
  · intro h; exact ⟨_, by omega, rfl⟩

theorem card_upTo (n : ℕ) :
    (upTo n : Finset (FreeMonoid α)).card ≤ ∑ j ∈ Finset.range (n + 1), Fintype.card α ^ j := by
  classical
  refine Finset.card_biUnion_le.trans (Finset.sum_le_sum fun j _ => ?_)
  refine Finset.card_image_le.trans ?_
  simp

/-- The probes that can read a string, a prefix of length at least `k` followed by a string of
length at most `n`, on its state's wrong side. -/
def PotWrong {σ : Type*} (M : DFA α σ) (side : σ → Bool) (k n : ℕ) (read : FreeMonoid α → ARU) :
    Set (FreeMonoid α) :=
  {x | ∃ z ∈ potM k x (upTo n), read z = if side (M.eval z.toList) then .reject else .accept}

/-- No read of the string followed by one of length at most `n` is on its state's wrong side. -/
def Clean {σ : Type*} (M : DFA α σ) (side : σ → Bool) (n : ℕ) (read : FreeMonoid α → ARU)
    (z : FreeMonoid α) : Prop :=
  ∀ m ∈ upTo n, read (z * m) ≠ if side (M.eval (z * m).toList) then .reject else .accept

theorem clean_of_not_potWrong {σ : Type*} {M : DFA α σ} {side : σ → Bool} {k n : ℕ}
    {read : FreeMonoid α → ARU} {x : FreeMonoid α} (hx : x ∉ PotWrong M side k n read) {i : ℕ}
    (h1 : k ≤ i) (h2 : i ≤ x.toList.length) : Clean M side n read (prefixOf x i) :=
  fun _ hm hw => hx ⟨_, mem_potM h1 h2 hm, hw⟩

theorem potWrong_empty {σ : Type*} {M : DFA α σ} {side : σ → Bool} {read : FreeMonoid α → ARU}
    (hW : NoWrong M side read) (k n : ℕ) : PotWrong M side k n read = ∅ :=
  Set.eq_empty_of_forall_notMem fun _ ⟨z, _, hz⟩ => hW z hz

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

/-- A tree `n` splits reach has midfixes of length at most `n`. -/
theorem mids_short : ∀ (n : ℕ) (T : DTree α), T ∈ classSet n → ∀ m ∈ mids T, m ∈ upTo n
  | 0, T, h, m, hm => by
    classical
    simp only [classSet, Finset.mem_singleton] at h
    subst h
    simp only [mids, Finset.mem_insert, Finset.notMem_empty, Finset.union_empty, or_false] at hm
    subst hm
    simp [mem_upTo]
  | n + 1, T, h, m, hm => by
    classical
    simp only [classSet, Finset.mem_union, Finset.mem_biUnion, Finset.mem_image] at h
    rcases h with h | ⟨T', hT', ⟨p, c, t, t₀⟩, -, rfl⟩
    · have := mem_upTo.1 (mids_short n T h m hm)
      exact mem_upTo.2 (by omega)
    · rcases Finset.mem_insert.1 (mids_splitAt _ T' p hm) with rfl | h
      · rw [mem_upTo]
        simp only [FreeMonoid.toList_mul, FreeMonoid.toList_of, List.length_append,
          List.length_singleton]
        rcases midfixAt_mem T' (commonPrefix t t₀) with h1 | h1
        · rw [h1]; simp
        · have := mem_upTo.1 (mids_short n T' hT' _ h1)
          omega
      · have := mem_upTo.1 (mids_short n T' hT' m h)
        exact mem_upTo.2 (by omega)

end Short

/-! ## True leaves -/

section True

variable {σ : Type*} (M : DFA α σ) (side : σ → Bool)

/-- The leaf a state's strings sift to where no read on the way is undecided or wrong. -/
def leafOf : DTree α → σ → List Bool
  | .leaf, _ => []
  | .node m r a, q =>
    if side (M.evalFrom q m.toList) then true :: leafOf a q else false :: leafOf r q

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

variable [DecidableEq α] (read : FreeMonoid α → ARU) (C : Level3Settings)

/-- The splits and learnings left, as #417 counts them, within `nσ + 2` leaves. -/
def Ψr [Fintype α] (nσ : ℕ) (s : Level3State α) : ℕ :=
  Ideal.Ψ nσ (Ideal.newHypothesis s.tree s.edges)

structure Inv [Fintype α] (s : Level3State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  cls : s.tree ∈ classSet (s.tree.leaves.length - 2)
  two : 2 ≤ s.tree.leaves.length

/-! ## The tree -/

section Tree

theorem classSet_mono [Fintype α] (n : ℕ) :
    (classSet n : Finset (DTree α)) ⊆ classSet (n + 1) := by
  classical
  intro T h
  simp only [classSet]
  exact Finset.mem_union_left _ h

theorem classSet_mono' [Fintype α] {n n' : ℕ} (h : n ≤ n') :
    (classSet n : Finset (DTree α)) ⊆ classSet n' := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (classSet_mono _)

theorem classSet_split [Fintype α] {n : ℕ} {T : DTree α} (hT : T ∈ classSet n) {p t t₀ : List Bool}
    {c : α} (hp : p ∈ T.leaves) (ht : t ∈ T.leaves) (ht₀ : t₀ ∈ T.leaves) :
    T.splitAt (FreeMonoid.of c * T.midfixAt (commonPrefix t t₀)) p ∈ classSet (n + 1) := by
  classical
  simp only [classSet]
  refine Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨T, hT, Finset.mem_image.2
    ⟨(p, c, t, t₀), ?_, rfl⟩⟩)
  simp [hp, ht, ht₀]

end Tree

/-! ## The steps -/

/-- Where a probe disagrees at decided reads: the edge and the target there. -/
def recOf : Outcome α → Option (List Bool × α × List Bool)
  | .wrongEdge p c _ t => some (p, c, t)
  | _ => none

section Steps

variable [Fintype α] {σ : Type*} [Fintype σ] (M : DFA α σ) (side : σ → Bool)

omit [DecidableEq α] in
theorem inv_start : Inv read (initialState : Level3State α) where
  edges := fun _ _ _ _ _ h => by simp [initialState, newHypothesis] at h
  cls := by simp [initialState, newHypothesis, DTree.leaves, classSet]
  two := by simp [initialState, newHypothesis, DTree.leaves]

theorem Ψr_start (nσ : ℕ) :
    Ψr nσ (initialState : Level3State α) + 1 ≤ Ideal.stretches nσ (Fintype.card α) :=
  Ideal.Ψ_start nσ

omit [DecidableEq α] in
theorem finish_inl {t s' : Level3State α} (h : applyTests C t = .inl s') : s' = t
    ∧ runTests C t = none := by
  unfold applyTests at h
  split at h
  · simp at h
  · rename_i hl
    exact ⟨by simpa using h.symm, hl⟩

theorem countProbe_tree (s : Level3State α) (x : FreeMonoid α) :
    (countProbe read C s x).tree = s.tree := rfl
theorem countProbe_edges (s : Level3State α) (x : FreeMonoid α) :
    (countProbe read C s x).edges = s.edges := rfl
theorem countProbe_probeCount (s : Level3State α) (x : FreeMonoid α) :
    (countProbe read C s x).probeCount = s.probeCount + 1 := rfl

/-- The steps that keep the hypothesis: a look after counting a probe that does not disagree at
decided reads. -/
theorem step_cases {s s' : Level3State α} {x : FreeMonoid α} (hs : Inv read s)
    (h : applyProbe read C s x = .inl s') :
    (∃ p c u t, probeOrAgree false read s.tree s.edges C.walkStart x = .unlearnedEdge p c u t
        ∧ s.edges (p, c) = none ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        p ∈ s.tree.leaves ∧
        s' = newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u))))
    ∨ (∃ p c u t t₀ w₀, probeOrAgree false read s.tree s.edges C.walkStart x = .wrongEdge p c u t ∧
        s.edges (p, c) = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        s' = splitLeaf read s p c t t₀)
    ∨ (s' = countProbe read C s x ∧ runTests C s' = none
        ∧ recOf (probeOrAgree false read s.tree s.edges C.walkStart x) = none) := by
  have hOK := Ideal.probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe at h
  generalize hg : probeOrAgree false read s.tree s.edges C.walkStart x = o at h hOK
  cases o with
  | unlearnedEdge p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    exact Or.inl ⟨p, c, u, t, rfl, hE, hu, huc, hp, by simpa using h.symm⟩
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, hne⟩ := hOK
    simp only [hE, Sum.inl.injEq] at h
    exact Or.inr (Or.inl ⟨p, c, u, t, t₀, w₀, rfl, hE, hne, hp, hu, huc, h.symm⟩)
  | agree | undecidedStart _ | undecided _ _ =>
    obtain ⟨h1, h2⟩ := finish_inl C h
    exact Or.inr (Or.inr ⟨h1, h1 ▸ h2, rfl⟩)

/-- A step that keeps the hypothesis counts one more probe. -/
theorem step_same {s s' : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (h : applyProbe read C s x = .inl s') (ht : s'.tree = s.tree) (he : s'.edges = s.edges) :
    Inv read s' ∧ (∀ nσ, Ψr nσ s' = Ψr nσ s) ∧ s'.probeCount = s.probeCount + 1 := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, rfl⟩ |
    ⟨rfl, -, -⟩
  · exfalso
    have := congrFun he (p, c)
    simp only [newHypothesis, Function.update_self, hE] at this
    exact absurd this (by simp)
  · exfalso
    have := congrArg List.length (congrArg DTree.leaves ht)
    simp only [splitLeaf, newHypothesis] at this
    rw [DTree.length_leaves_splitAt _ _ _ hp] at this
    omega
  · exact ⟨⟨hs.edges, hs.cls, hs.two⟩, fun _ => rfl, rfl⟩

/-- The tree has a true leaf for each leaf past the first, and every learned edge a clean witness
and successor. -/
structure GInv (s : Level3State α) : Prop where
  leaves : s.tree.leaves.length ≤ (img M side s.tree).card + 1
  clean : ∀ p c t w, s.edges (p, c) = some (t, w) →
    Clean M side (Fintype.card σ) read w ∧ Clean M side (Fintype.card σ) read (w * FreeMonoid.of c)

omit [DecidableEq α] in
theorem gInv_start : GInv read M side (initialState : Level3State α) where
  leaves := by
    have : 0 < (img M side (initialState : Level3State α).tree).card :=
      Finset.card_pos.2 ⟨_, Finset.mem_image_of_mem _ (Finset.mem_univ M.start)⟩
    simp only [initialState, newHypothesis, DTree.leaves] at this ⊢
    simp only [List.map_cons, List.map_nil, List.cons_append, List.nil_append, List.length_cons,
      List.length_nil]
    omega
  clean := fun _ _ _ _ h => by simp [initialState, newHypothesis] at h

omit [DecidableEq α] in
theorem gInv_congr {s s' : Level3State α} (hg : GInv read M side s) (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) : GInv read M side s' :=
  ⟨ht ▸ hg.leaves, he ▸ hg.clean⟩

omit [DecidableEq α] in
theorem leaves_le {s : Level3State α} (hg : GInv read M side s) :
    s.tree.leaves.length ≤ Fintype.card σ + 1 :=
  hg.leaves.trans (by have := img_card_le M side s.tree; omega)

theorem cls_mem {s : Level3State α} (hs : Inv read s) (hg : GInv read M side s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := leaves_le read M side hg; omega) hs.cls

omit [DecidableEq α] in
/-- A clean string that sifts to a leaf of a tree `|Q|` splits reach sifts to its true leaf. -/
theorem true_leaf {T : DTree α} (hT : T ∈ classSet (Fintype.card σ)) {z : FreeMonoid α}
    (hz : Clean M side (Fintype.card σ) read z) {p : List Bool} (h : T.sift read z = .inl p) :
    p = leafOf M side T (M.eval z.toList) :=
  sift_true M side read T z p h fun m hm => hz m (mids_short _ T hT m hm)

omit [DecidableEq α] in
theorem real_of_clean {T : DTree α} (hT : T ∈ classSet (Fintype.card σ)) {w : FreeMonoid α}
    {p t : List Bool} {c : α} (hw : Clean M side (Fintype.card σ) read w)
    (hwc : Clean M side (Fintype.card σ) read (w * FreeMonoid.of c)) (h1 : T.sift read w = .inl p)
    (h2 : T.sift read (w * FreeMonoid.of c) = .inl t) : RealT M side T p c t := by
  refine ⟨M.eval w.toList, (true_leaf read M side hT hw h1).symm, ?_⟩
  rw [true_leaf read M side hT hwc h2, eval_mul]
  rfl

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
    (h : probeOrAgree false read T E k x = .wrongEdge p c u t
      ∨ probeOrAgree false read T E k x = .unlearnedEdge p c u t) :
    ∃ i, k ≤ i ∧ i < x.toList.length ∧ u = prefixOf x i ∧ x.toList[i]? = some c := by
  have hk : k ≤ x.toList.length := by
    by_contra hc
    simp [probeOrAgree, show x.toList.length < k by omega] at h
  have h' : probe false read T E k x = .wrongEdge p c u t
      ∨ probe false read T E k x = .unlearnedEdge p c u t := by
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

omit [DecidableEq α] in
/-- The prefix and its successor of an edge a probe that cannot read a string wrong learns or
disagrees at are clean. -/
theorem clean_of_probe {T : DTree α} {E : Edges α} {k : ℕ} {x : FreeMonoid α}
    (hx : x ∉ PotWrong M side k (Fintype.card σ) read) {p : List Bool} {c : α} {u : FreeMonoid α}
    {t : List Bool}
    (h : probeOrAgree false read T E k x = .wrongEdge p c u t
      ∨ probeOrAgree false read T E k x = .unlearnedEdge p c u t) :
    Clean M side (Fintype.card σ) read u
      ∧ Clean M side (Fintype.card σ) read (u * FreeMonoid.of c) := by
  obtain ⟨i, h1, h2, rfl, hc⟩ := probe_prefix read T E k x h
  rw [← prefixOf_succ hc]
  exact ⟨clean_of_not_potWrong hx h1 h2.le, clean_of_not_potWrong hx (by omega) h2⟩

/-- A split between the edge's target and a real one keeps the invariants and the tree in the
class, and lowers `Ψr`. -/
theorem split_ok {s : Level3State α} (hs : Inv read s) (hg : GInv read M side s)
    {p t t₀ : List Bool} {c : α} {u w₀ : FreeMonoid α} (hp : p ∈ s.tree.leaves)
    (hu : s.tree.sift read u = .inl p) (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t)
    (hcu : Clean M side (Fintype.card σ) read u)
    (hcuc : Clean M side (Fintype.card σ) read (u * FreeMonoid.of c))
    (hE : s.edges (p, c) = some (t₀, w₀)) (hne : t₀ ≠ t) :
    Inv read (splitLeaf read s p c t t₀) ∧ GInv read M side (splitLeaf read s p c t t₀)
      ∧ Ψr (Fintype.card σ) (splitLeaf read s p c t t₀) < Ψr (Fintype.card σ) s := by
  obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
  have hT := cls_mem read M side hs hg
  have ht := real_of_clean read M side hT hcu hcuc hu huc
  obtain ⟨hcw, hcwc⟩ := hg.clean p c t₀ w₀ hE
  have ht₀ := real_of_clean read M side hT hcw hcwc hw₀ hw₀c
  have hgen := genuine_of_real M side ht ht₀ (Ne.symm hne)
  have himg := img_card_splitAt M side hp hgen
  have hlen :=
    DTree.length_leaves_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
  have ht' := DTree.sift_inl_mem read _ _ t huc
  have ht₀' := DTree.sift_inl_mem read _ _ t₀ hw₀c
  have hleaves : (splitLeaf read s p c t t₀).tree.leaves.length
      ≤ (img M side (splitLeaf read s p c t t₀).tree).card + 1 := by
    simp only [splitLeaf, newHypothesis]
    have := hg.leaves
    omega
  refine ⟨⟨Ideal.edgesOK_retarget read hs.edges hp _, ?_,
    by simp only [splitLeaf, newHypothesis]; rw [hlen]; have := hs.two; omega⟩,
    ⟨hleaves, fun p' c' t' w h => ?_⟩, ?_⟩
  · simp only [splitLeaf, newHypothesis]
    rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
      have := hs.two; omega]
    exact classSet_split hs.cls hp ht' ht₀'
  · simp only [splitLeaf, newHypothesis] at h
    unfold retarget at h
    split_ifs at h
    split at h
    · rename_i t₁ w₁ hE₁
      have hc := hg.clean p' c' t₁ w₁ hE₁
      split_ifs at h
      · rcases hs' : (s.tree.splitAt _ p).sift read (w₁ * FreeMonoid.of c') with a | z <;>
          rw [hs'] at h <;> simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq,
            Prod.mk.injEq, reduceCtorEq] at h
        obtain ⟨-, rfl⟩ := h
        exact hc
      · simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨-, rfl⟩ := h
        exact hc
    · simp at h
  · exact Ideal.Ψ_lt_split _ (s := Ideal.newHypothesis s.tree s.edges) hp
      (by
        have := img_card_le M side (splitLeaf read s p c t t₀).tree
        simp only [splitLeaf, newHypothesis] at this hleaves
        simp only [Ideal.newHypothesis]
        omega) _

/-- A step that changes the hypothesis on a probe that cannot read a string wrong starts a fresh
stretch, keeps the invariants and lowers `Ψr`. -/
theorem step_change {s s' : Level3State α} (hs : Inv read s) (hg : GInv read M side s)
    {x : FreeMonoid α} (hx : x ∉ PotWrong M side C.walkStart (Fintype.card σ) read)
    (h : applyProbe read C s x = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read s' ∧ GInv read M side s' ∧ s' = newHypothesis s'.tree s'.edges
      ∧ Ψr (Fintype.card σ) s' < Ψr (Fintype.card σ) s := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, ho, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu, huc, rfl⟩ | ⟨rfl, -, -⟩
  · obtain ⟨hcu, hcuc⟩ := clean_of_probe read M side hx (Or.inr ho)
    refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, hs.cls, hs.two⟩,
      ⟨hg.leaves, fun p' c' t' w h => ?_⟩, rfl,
      Ideal.Ψ_lt_learn _ (s := Ideal.newHypothesis s.tree s.edges) hp hE _⟩
    simp only [newHypothesis] at h
    by_cases hpc : (p', c') = (p, c)
    · simp only [Prod.mk.injEq] at hpc
      obtain ⟨rfl, rfl⟩ := hpc
      simp only [Function.update_self, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact ⟨hcu, hcuc⟩
    · rw [Function.update_of_ne hpc] at h
      exact hg.clean p' c' t' w h
  · obtain ⟨hcu, hcuc⟩ := clean_of_probe read M side hx (Or.inl ho)
    obtain ⟨hinv, hg', hΨ⟩ := split_ok read M side hs hg hp hu huc hcu hcuc hE htt
    exact ⟨hinv, hg', rfl, hΨ⟩
  · exact absurd ⟨rfl, rfl⟩ hne

omit [Fintype σ] in
theorem applyProbe_ne_failed (s : Level3State α) (x : FreeMonoid α) :
    applyProbe read C s x ≠ .inr .failed := by
  have hlook : ∀ t : Level3State α, applyTests C t ≠ .inr .failed := by
    intro t ht
    unfold applyTests runTests at ht
    split_ifs at ht <;> simp_all
  unfold applyProbe
  split
  · simp
  · split
    · simp
    · exact hlook _
  · exact hlook _

end Steps

end Random

end OrthoDFA
