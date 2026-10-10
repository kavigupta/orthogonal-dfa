import OrthoDFA.RandomRound
import OrthoDFA.Proofs.IdealSteps

/-!
# The tree and the steps under reads that are never wrong

A decided read is on its state's side, so two strings of one state that both sift decidedly
reach the same leaf, and the leaves past the first two are at most `|Q|`. Every split is on a
letter and the midfix where two leaves part, so the tree is in `classSet |Q|`.

A step that keeps the hypothesis counts one more probe into the stretch; one that changes it
starts a fresh stretch and lowers `Ψ`, #417's count of the splits, learnings and redirects left.
-/

namespace OrthoDFA

namespace Random

open OrthoDFA.Ideal (DTree Edges walk pre search agrees probe Outcome upd2 lcp retarget Disagrees
  EdgesOK Reached OutcomeOK)

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
/-- The strings a probe of `x` can read: a prefix of length at least `k` followed by a midfix. -/
noncomputable def pot (k : ℕ) (x : FreeMonoid α) (T : DTree α) : Finset (FreeMonoid α) :=
  (Finset.Icc k x.toList.length).biUnion fun p => (mids T).image (pre x p * ·)

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
        fun q => T.splitAt (FreeMonoid.of q.2.1 * T.midAt (lcp q.2.2.1 q.2.2.2)) q.1

variable [DecidableEq α] (read : FreeMonoid α → ARU) (C : Cfg)

/-- #417's settings with this round's cap, for its potential. -/
def idealCfg (C : Cfg) : Ideal.RoundCfg := ⟨C.k, C.m, 1, 1, C.Lmax⟩

/-- The splits, learnings and redirects left, as #417 counts them. -/
def Ψr [Fintype α] (s : RState α) : ℕ := Ideal.Ψ (idealCfg C) (Ideal.fresh s.tree s.edges s.moved)

structure Inv [Fintype α] (s : RState α) : Prop where
  edges : EdgesOK read s.tree s.edges
  recs_lt : ∀ r, s.recs.count r < C.m
  size : s.tree.leaves.length ≤ C.Lmax
  reached : Reached read s.tree
  cls : s.tree ∈ classSet (s.tree.leaves.length - 2)
  two : 2 ≤ s.tree.leaves.length

/-! ## The tree -/

section Tree

omit [DecidableEq α] in
/-- Two strings of one state that both sift decidedly reach the same leaf. -/
theorem sift_inl_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hW : NoWrong M side read)
    {z z' : FreeMonoid α} (hz : M.eval z.toList = M.eval z'.toList) :
    ∀ (T : DTree α) (p p' : List Bool), T.sift read z = .inl p → T.sift read z' = .inl p' →
      p = p'
  | .leaf, p, p', h, h' => by
    simp only [DTree.sift, Sum.inl.injEq] at h h'; rw [← h, ← h']
  | .node m r a, p, p', h, h' => by
    have hs : M.eval (z * m).toList = M.eval (z' * m).toList := by
      simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
      exact congrArg (M.evalFrom · m.toList) hz
    have hw := hW (z * m)
    have hw' := hW (z' * m)
    rw [hs] at hw
    simp only [DTree.sift] at h h'
    split at h <;> rename_i hr <;> split at h' <;> rename_i hr'
    · rcases hq : a.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rcases hq' : a.sift read z' with q' | q' <;> rw [hq'] at h' <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h'
      rw [← h, ← h', sift_inl_congr hW hz a q q' hq hq']
    · exfalso
      rw [hr] at hw; rw [hr'] at hw'
      cases hsd : side (M.eval (z' * m).toList) <;> simp_all
    · simp at h'
    · exfalso
      rw [hr] at hw; rw [hr'] at hw'
      cases hsd : side (M.eval (z' * m).toList) <;> simp_all
    · rcases hq : r.sift read z with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h
      rcases hq' : r.sift read z' with q' | q' <;> rw [hq'] at h' <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inl.injEq, reduceCtorEq] at h'
      rw [← h, ← h', sift_inl_congr hW hz r q q' hq hq']
    · simp at h'
    · simp at h
    · simp at h
    · simp at h

omit [DecidableEq α] in
theorem leaves_le_of_reached' {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}
    (hW : NoWrong M side read) {T : DTree α} (h : Reached read T) :
    T.leaves.length ≤ Fintype.card σ + 2 := by
  classical
  rw [← List.toFinset_card_of_nodup (DTree.leaves_nodup T)]
  set A := T.leaves.toFinset.filter fun p => ∃ w, T.sift read w = .inl p
  have hsub : T.leaves.toFinset ⊆ A ∪ {[false], [true]} := by
    intro p hp
    rcases h p (List.mem_toFinset.1 hp) with h1 | h1 | h1
    · simp [h1]
    · simp [h1]
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨hp, h1⟩)
  have hA : A.card ≤ Fintype.card σ := by
    let f : List Bool → σ := fun p =>
      if hp : ∃ w, T.sift read w = .inl p then M.eval hp.choose.toList else M.eval []
    refine Finset.card_le_card_of_injOn f (fun _ _ => Finset.mem_univ _) ?_
    intro p hp p' hp' hf
    have h1 := (Finset.mem_filter.1 hp).2
    have h2 := (Finset.mem_filter.1 hp').2
    simp only [f, h1, h2, dite_true] at hf
    exact sift_inl_congr read hW hf T p p' h1.choose_spec h2.choose_spec
  calc T.leaves.toFinset.card ≤ (A ∪ {[false], [true]}).card := Finset.card_le_card hsub
    _ ≤ A.card + ({[false], [true]} : Finset (List Bool)).card := Finset.card_union_le _ _
    _ ≤ Fintype.card σ + 2 := by
      have : ({[false], [true]} : Finset (List Bool)).card ≤ 2 := Finset.card_le_two
      omega

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

theorem classSet_split [Fintype α] {n : ℕ} {T : DTree α} (hT : T ∈ classSet n)
    {p t t₀ : List Bool} {c : α} (hp : p ∈ T.leaves) (ht : t ∈ T.leaves) (ht₀ : t₀ ∈ T.leaves) :
    T.splitAt (FreeMonoid.of c * T.midAt (lcp t t₀)) p ∈ classSet (n + 1) := by
  classical
  simp only [classSet]
  refine Finset.mem_union_right _ (Finset.mem_biUnion.2 ⟨T, hT, Finset.mem_image.2
    ⟨(p, c, t, t₀), ?_, rfl⟩⟩)
  simp [hp, ht, ht₀]

end Tree

/-! ## The steps -/

section Steps

variable [Fintype α] {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem inv_start (hm : 1 ≤ C.m) (h2 : 2 ≤ C.Lmax) : Inv read C (start : RState α) where
  edges := fun _ _ _ _ _ h => by simp [start, fresh] at h
  recs_lt := fun _ => by simp [start, fresh]; omega
  size := by simp [start, fresh, DTree.leaves]; omega
  reached := by
    intro p hp
    simp only [start, fresh, DTree.leaves, List.map_cons, List.map_nil, List.cons_append,
      List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · exact Or.inl rfl
    · exact Or.inr (Or.inl rfl)
  cls := by simp [start, fresh, DTree.leaves, classSet]
  two := by simp [start, fresh, DTree.leaves]

omit [DecidableEq α] in
theorem leaves_root : (DTree.node 1 .leaf .leaf : DTree α).leaves = [[false], [true]] := by
  simp [DTree.leaves]

theorem Ψr_start (h2 : 2 ≤ C.Lmax) :
    Ψr C (start : RState α) + 1 ≤ stretches C (Fintype.card α) := by
  have hΦ := Ideal.Φ_le (Ideal.fresh (start : RState α).tree (start : RState α).edges
    (start : RState α).moved)
  unfold Ψr stretches Ideal.Ψ
  simp only [start, fresh, Ideal.fresh] at hΦ ⊢
  rw [leaves_root] at hΦ ⊢
  simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hΦ ⊢
  have hX : Ideal.X (α := α) (idealCfg C) = 2 * C.Lmax * Fintype.card α + 1 := rfl
  rw [hX]
  have hL : C.Lmax * (2 * C.Lmax * Fintype.card α + 1)
      = (C.Lmax - 2) * (2 * C.Lmax * Fintype.card α + 1) + 2 * (2 * C.Lmax * Fintype.card α + 1) := by
    rw [← Nat.add_mul, Nat.sub_add_cancel h2]
  have : 4 * Fintype.card α ≤ 2 * (2 * C.Lmax * Fintype.card α + 1) := by
    have := Nat.mul_le_mul_right (Fintype.card α) h2
    nlinarith
  have hC : (idealCfg C).Lmax = C.Lmax := rfl
  rw [hC]
  omega

theorem probeR_ok {T : DTree α} {E : Ideal.Edges α} (hE : EdgesOK read T E) (k : ℕ)
    (x : FreeMonoid α) : OutcomeOK read T E (probeR read T E k x) := by
  unfold probeR
  split_ifs
  · trivial
  · exact Ideal.probe_ok read T E k hE x

theorem finish_inl {t s' : RState α} (h : finish C t = .inl s') : s' = t := by
  unfold finish at h
  split at h
  · simp at h
  · simpa using h.symm

theorem charge_tree (s : RState α) (x : FreeMonoid α) : (charge read C s x).tree = s.tree := rfl
theorem charge_edges (s : RState α) (x : FreeMonoid α) : (charge read C s x).edges = s.edges := rfl
theorem charge_moved (s : RState α) (x : FreeMonoid α) : (charge read C s x).moved = s.moved := rfl
theorem charge_recs (s : RState α) (x : FreeMonoid α) : (charge read C s x).recs = s.recs := rfl
theorem charge_n (s : RState α) (x : FreeMonoid α) : (charge read C s x).n = s.n + 1 := rfl

/-- The steps that keep the hypothesis: a look after counting the probe. -/
theorem step_cases {s s' : RState α} {x : FreeMonoid α} (hs : Inv read C s)
    (h : step read C s x = .inl s') :
    (∃ p c u t, probeR read s.tree s.edges C.k x = .member p c u t ∧ s.edges p c = none ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        p ∈ s.tree.leaves ∧
        s' = fresh s.tree (upd2 s.edges p c (some (t, u))) (upd2 s.moved p c false))
    ∨ (∃ p c u t t₀ w₀, probeR read s.tree s.edges C.k x = .edge p c u t ∧
        s.edges p c = some (t₀, w₀) ∧ t₀ ≠ t ∧ p ∈ s.tree.leaves ∧
        s.tree.sift read u = .inl p ∧ s.tree.sift read (u * FreeMonoid.of c) = .inl t ∧
        C.m ≤ (s.recs ++ [(p, c, t)]).count (p, c, t) ∧
        fix read C s p c u t = .inl s')
    ∨ (s' = charge read C { s with recs := s.recs ++ (match probeR read s.tree s.edges C.k x with
          | .edge p c _ t => [(p, c, t)]
          | _ => []) } x
        ∧ ∀ p c u t, probeR read s.tree s.edges C.k x = .edge p c u t →
          ¬ C.m ≤ (s.recs ++ [(p, c, t)]).count (p, c, t)) := by
  have hOK := probeR_ok read hs.edges C.k x
  unfold step at h
  generalize hg : probeR read s.tree s.edges C.k x = o at h hOK
  cases o with
  | member p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    exact Or.inl ⟨p, c, u, t, rfl, hE, hu, huc, hp, by simpa using h.symm⟩
  | edge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, hne⟩ := hOK
    simp only at h
    split_ifs at h with hc
    · exact Or.inr (Or.inl ⟨p, c, u, t, t₀, w₀, rfl, hE, hne, hp, hu, huc, hc, h⟩)
    · refine Or.inr (Or.inr ⟨finish_inl C h, ?_⟩)
      intro p' c' u' t' he
      cases he
      exact hc
  | agree => exact Or.inr (Or.inr ⟨by simpa using finish_inl C h, fun _ _ _ _ he => by cases he⟩)
  | startU z => exact Or.inr (Or.inr ⟨by simpa using finish_inl C h, fun _ _ _ _ he => by cases he⟩)
  | endU z => exact Or.inr (Or.inr ⟨by simpa using finish_inl C h, fun _ _ _ _ he => by cases he⟩)
  | triple k zs =>
    exact Or.inr (Or.inr ⟨by simpa using finish_inl C h, fun _ _ _ _ he => by cases he⟩)
  | pair k zs =>
    exact Or.inr (Or.inr ⟨by simpa using finish_inl C h, fun _ _ _ _ he => by cases he⟩)

/-- A step that keeps the hypothesis counts one more probe. -/
theorem step_same (hW : NoWrong M side read) {s s' : RState α} (hs : Inv read C s)
    {x : FreeMonoid α} (h : step read C s x = .inl s') (ht : s'.tree = s.tree)
    (he : s'.edges = s.edges) : Inv read C s' ∧ Ψr C s' = Ψr C s ∧ s'.n = s.n + 1 := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, -, -, -, rfl⟩ | ⟨p, c, u, t, t₀, w₀, -, hE, hne, hp, -, -, -, hfix⟩ |
    ⟨rfl, hlt⟩
  · exfalso
    have := congrFun (congrFun he p) c
    simp only [fresh, Ideal.upd2_same, hE] at this
    exact absurd this (by simp)
  · exfalso
    unfold fix at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · unfold split at hfix
      simp only at hfix
      split_ifs at hfix
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrArg List.length (congrArg DTree.leaves ht)
      simp only [fresh] at this
      rw [DTree.length_leaves_splitAt _ _ _ hp] at this
      omega
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      have := congrFun (congrFun he p) c
      simp only [fresh, Ideal.upd2_same, hE, Option.some.injEq, Prod.mk.injEq] at this
      exact hne this.1.symm
  · refine ⟨⟨hs.edges, ?_, hs.size, hs.reached, hs.cls, hs.two⟩, rfl, rfl⟩
    intro r
    simp only [charge_recs]
    split
    · rename_i p c u t hg
      by_cases hr : r = (p, c, t)
      · subst hr; have := hlt p c u t hg; omega
      · simp only [List.count_append, List.count_singleton]
        have := hs.recs_lt r
        simp [Ne.symm hr, this]
    · simpa using hs.recs_lt r

/-- A split on a letter and the midfix where its targets part keeps the tree in the class and
every leaf past the first two reached. -/
theorem split_ok (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.Lmax) (hm : 1 ≤ C.m)
    {s : RState α} (hs : Inv read C s) {p t t₀ : List Bool} {c : α} {u w₀ : FreeMonoid α}
    (hp : p ∈ s.tree.leaves) (hu : s.tree.sift read u = .inl p)
    (huc : s.tree.sift read (u * FreeMonoid.of c) = .inl t)
    (hw₀ : s.tree.sift read w₀ = .inl p) (hw₀c : s.tree.sift read (w₀ * FreeMonoid.of c) = .inl t₀)
    (hne : t₀ ≠ t) :
    let T' := s.tree.splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) p
    T'.leaves.length ≤ C.Lmax ∧ Inv read C (fresh T' (retarget read T' p s.edges) fun _ _ => false)
      ∧ Ψr C (fresh T' (retarget read T' p s.edges) fun _ _ => false) < Ψr C s := by
  intro T'
  have hpart := DTree.sift_part read s.tree _ _ t t₀ huc hw₀c (Ne.symm hne)
  have hreach : Reached read T' := by
    refine Ideal.reached_splitAt read hs.reached hp hu hw₀ ?_ ?_ ?_
    · simpa [mul_assoc] using hpart.1
    · simpa [mul_assoc] using hpart.2.1
    · simpa [mul_assoc] using hpart.2.2
  have hsz := (leaves_le_of_reached' read hW hreach).trans hcap
  have hlen := DTree.length_leaves_splitAt (FreeMonoid.of c * s.tree.midAt (lcp t t₀)) s.tree p hp
  have ht := DTree.sift_inl_mem read _ _ t huc
  have ht₀ := DTree.sift_inl_mem read _ _ t₀ hw₀c
  refine ⟨hsz, ⟨Ideal.edgesOK_retarget read hs.edges hp _, by simp [fresh]; omega, hsz, hreach,
    ?_, by simp only [fresh, T']; rw [hlen]; have := hs.two; omega⟩, ?_⟩
  · simp only [fresh, T']
    rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
      have := hs.two; omega]
    exact classSet_split hs.cls hp ht ht₀
  · exact Ideal.Ψ_lt_split (idealCfg C) (s := Ideal.fresh s.tree s.edges s.moved) hp hsz _ _

/-- A step that changes the hypothesis starts a fresh stretch and lowers `Ψr`. -/
theorem step_change (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) {s s' : RState α} (hs : Inv read C s) {x : FreeMonoid α}
    (h : step read C s x = .inl s') (hne : ¬ (s'.tree = s.tree ∧ s'.edges = s.edges)) :
    Inv read C s' ∧ s' = fresh s'.tree s'.edges s'.moved ∧ Ψr C s' < Ψr C s := by
  rcases step_cases read C hs h with
    ⟨p, c, u, t, -, hE, hu, huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, -, hE, htt, hp, hu, huc, -, hfix⟩ | ⟨rfl, -⟩
  · refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, fun r => by simp [fresh]; omega, hs.size,
      hs.reached, hs.cls, hs.two⟩, rfl, ?_⟩
    exact Ideal.Ψ_lt_upd (idealCfg C) (s := Ideal.fresh s.tree s.edges s.moved) hp _ _
      (by simp [Ideal.wt, Ideal.fresh, Ideal.upd2_same, hE])
  · obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
    unfold fix at hfix
    rw [hE] at hfix
    simp only at hfix
    split_ifs at hfix with hmv
    · obtain ⟨hsz, hinv, hΨ⟩ := split_ok read C hW hcap hm hs hp hu huc hw₀ hw₀c htt
      unfold split at hfix
      simp only at hfix
      rw [if_neg (by omega)] at hfix
      simp only [Sum.inl.injEq] at hfix
      subst hfix
      exact ⟨hinv, rfl, hΨ⟩
    · simp only [Sum.inl.injEq] at hfix
      subst hfix
      refine ⟨⟨Ideal.edgesOK_upd read hs.edges hu huc, fun r => by simp [fresh]; omega, hs.size,
        hs.reached, hs.cls, hs.two⟩, rfl, ?_⟩
      exact Ideal.Ψ_lt_upd (idealCfg C) (s := Ideal.fresh s.tree s.edges s.moved) hp _ _
        (by simp [Ideal.wt, Ideal.fresh, Ideal.upd2_same, hE, hmv])
  · exact absurd ⟨rfl, rfl⟩ hne

theorem step_ne_tooBig (hW : NoWrong M side read) (hcap : Fintype.card σ + 2 ≤ C.Lmax)
    (hm : 1 ≤ C.m) {s : RState α} (hs : Inv read C s) (x : FreeMonoid α) :
    step read C s x ≠ .inr .tooBig := by
  intro h
  have hOK := probeR_ok read hs.edges C.k x
  have hlook : ∀ t : RState α, finish C t ≠ .inr .tooBig := by
    intro t ht
    unfold finish look at ht
    split_ifs at ht <;> simp_all
  unfold step at h
  generalize probeR read s.tree s.edges C.k x = o at h hOK
  cases o with
  | member p c u t => simp at h
  | edge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, htt⟩ := hOK
    simp only at h
    split_ifs at h
    · obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
      unfold fix at h
      rw [hE] at h
      simp only at h
      split_ifs at h
      · obtain ⟨hsz, -, -⟩ := split_ok read C hW hcap hm hs hp hu huc hw₀ hw₀c htt
        unfold split at h
        dsimp only at h
        split_ifs at h with hlt
        omega
    · exact hlook _ h
  | agree => exact hlook _ h
  | startU z => exact hlook _ h
  | endU z => exact hlook _ h
  | triple k zs => exact hlook _ h
  | pair k zs => exact hlook _ h

theorem cls_mem (hW : NoWrong M side read) {s : RState α} (hs : Inv read C s) :
    s.tree ∈ classSet (Fintype.card σ) :=
  classSet_mono' (by have := leaves_le_of_reached' read hW hs.reached; omega) hs.cls

end Steps

end Random

end OrthoDFA
