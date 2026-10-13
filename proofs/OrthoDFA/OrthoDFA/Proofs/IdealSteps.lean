import OrthoDFA.Proofs.IdealProbe
import OrthoDFA.Proofs.BinomLaw
import OrthoDFA.Proofs.Adaptive

/-!
# A round's steps under ideal reads

The invariant `Inv`: every learned edge out of a leaf points where its witness's successor sifts,
every leaf past the first two is reached by some string, the stretch's disagreements are at most
its undecided reads and its probes, and it is shorter than `ends`, the look by which, with
`harvestRate < disagreementRate`, one of the rate tests decides whatever the counts.

A disagreement at decided reads at the edge `(p, c)` has two strings at `p`, the probe's prefix and
the edge's witness, whose successors sift to different leaves. They read decidedly apart at `c`
and the midfix where those leaves part, so both new leaves of the split are reached, and with
ideal reads the reached leaves are at most `|Q|`. Each step that does not end the round lowers
`Θ`: a split or a learned edge lowers `Ψ`, which counts the splits left and the unlearned edges,
and a probe that keeps the hypothesis adds to the stretch, which stays below `ends`.
-/

namespace OrthoDFA

namespace Ideal

open Finset

variable {α : Type*} (read : FreeMonoid α → ARU) (C : Level2Settings)

/-- Every leaf past the first two is reached by some string. -/
def Reached (T : DTree α) : Prop :=
  ∀ p ∈ T.leaves, p = [false] ∨ p = [true] ∨ ∃ w, T.sift read w = .inl p

/-! ## The tree's size -/

theorem leaves_le_of_reached {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}
    (hI : IdealReads read M side) {T : DTree α} (h : Reached read T) :
    T.leaves.length ≤ Fintype.card σ + 2 := by
  classical
  rw [← List.toFinset_card_of_nodup (DTree.leaves_nodup T)]
  set A := T.leaves.toFinset.filter fun p => ∃ w, T.sift read w = .inl p
  have hsub : T.leaves.toFinset ⊆ A ∪ {[false], [true]} := by
    intro p hp
    rcases h p (List.mem_toFinset.1 hp) with h1 | h1 | h1
    · simp [h1]
    · simp [h1]
    · exact mem_union_left _ (mem_filter.2 ⟨hp, h1⟩)
  have hA : A.card ≤ Fintype.card σ := by
    let f : List Bool → σ := fun p =>
      if hp : ∃ w, T.sift read w = .inl p then M.eval hp.choose.toList else M.eval []
    refine card_le_card_of_injOn f (fun _ _ => mem_univ _) ?_
    intro p hp p' hp' hf
    have h1 := (mem_filter.1 hp).2
    have h2 := (mem_filter.1 hp').2
    simp only [f, h1, h2, dite_true] at hf
    have := DTree.sift_congr read _ _ (read_mul_congr read hI hf) T p h1.choose_spec
    rw [h2.choose_spec] at this
    exact (Sum.inl.inj this).symm
  calc T.leaves.toFinset.card ≤ (A ∪ {[false], [true]}).card := card_le_card hsub
    _ ≤ A.card + ({[false], [true]} : Finset (List Bool)).card := card_union_le _ _
    _ ≤ Fintype.card σ + 2 := by
      have : ({[false], [true]} : Finset (List Bool)).card ≤ 2 := card_le_two
      omega

/-! ## Splits -/

theorem sift_splitAt_ne {T : DTree α} {d z : FreeMonoid α} {p q : List Bool}
    (h : T.sift read z = .inl q) (hne : q ≠ p) : (T.splitAt d p).sift read z = .inl q := by
  rw [DTree.sift_splitAt, h]; simp [hne]

theorem sift_splitAt_eq {T : DTree α} {d z : FreeMonoid α} {p : List Bool}
    (h : T.sift read z = .inl p) :
    (T.splitAt d p).sift read z = ((DTree.node d .leaf .leaf).sift read z).map (p ++ ·) id := by
  rw [DTree.sift_splitAt, h]; simp

/-- A split whose leaf holds two strings reading decidedly apart at its midfix keeps every leaf
past the first two reached. -/
theorem reached_splitAt {T : DTree α} (hr : Reached read T) {p : List Bool} (hp : p ∈ T.leaves)
    {d w₁ w₂ : FreeMonoid α} (h₁ : T.sift read w₁ = .inl p) (h₂ : T.sift read w₂ = .inl p)
    (hd₁ : read (w₁ * d) ≠ .undecided) (hd₂ : read (w₂ * d) ≠ .undecided)
    (hd : read (w₁ * d) ≠ read (w₂ * d)) : Reached read (T.splitAt d p) := by
  have hchild : ∀ b, ∃ w, (T.splitAt d p).sift read w = .inl (p ++ [b]) := by
    have key : ∀ w, T.sift read w = .inl p → read (w * d) ≠ .undecided →
        (T.splitAt d p).sift read w = .inl (p ++ [decide (read (w * d) = .accept)]) := by
      intro w hw hu
      rw [sift_splitAt_eq read hw]
      cases hrd : read (w * d) <;> simp_all [DTree.sift]
    intro b
    cases hb : decide (read (w₁ * d) = .accept) with
    | true =>
      cases b
      · refine ⟨w₂, ?_⟩
        rw [key w₂ h₂ hd₂]
        have : read (w₁ * d) = .accept := by simpa using hb
        cases hr2 : read (w₂ * d) <;> simp_all
      · exact ⟨w₁, by rw [key w₁ h₁ hd₁, hb]⟩
    | false =>
      cases b
      · exact ⟨w₁, by rw [key w₁ h₁ hd₁, hb]⟩
      · refine ⟨w₂, ?_⟩
        rw [key w₂ h₂ hd₂]
        have : read (w₁ * d) ≠ .accept := by simpa using hb
        cases hr1 : read (w₁ * d) <;> cases hr2 : read (w₂ * d) <;> simp_all
  intro q hq
  rcases (DTree.mem_leaves_splitAt d T p hp q).1 hq with ⟨hq, hne⟩ | rfl | rfl
  · rcases hr q hq with h | h | ⟨w, hw⟩
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr ⟨w, sift_splitAt_ne read hw hne⟩)
  · exact Or.inr (Or.inr (hchild false))
  · exact Or.inr (Or.inr (hchild true))

theorem prefix_of_mem_leaves_splitAt {T : DTree α} {p q : List Bool} {d : FreeMonoid α}
    (hp : p ∈ T.leaves) (hq : q ∈ (T.splitAt d p).leaves) (hpq : p.isPrefixOf q = false) :
    q ∈ T.leaves ∧ q ≠ p := by
  rcases (DTree.mem_leaves_splitAt d T p hp q).1 hq with h | rfl | rfl
  · exact h
  · rw [List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)] at hpq; simp at hpq
  · rw [List.isPrefixOf_iff_prefix.2 (List.prefix_append _ _)] at hpq; simp at hpq

theorem edgesOK_retarget {T : DTree α} {E : Edges α} (hE : EdgesOK read T E) {p : List Bool}
    (hp : p ∈ T.leaves) (d : FreeMonoid α) :
    EdgesOK read (T.splitAt d p) (retarget read (T.splitAt d p) p E) := by
  intro q hq c t w h
  unfold retarget at h
  cases hpq : p.isPrefixOf q
  · obtain ⟨hq', hne⟩ := prefix_of_mem_leaves_splitAt hp hq hpq
    simp only [hpq, Bool.false_eq_true, if_false] at h
    rcases hEq : E (q, c) with _ | ⟨t₁, w₁⟩ <;> rw [hEq] at h
    · simp at h
    · obtain ⟨hw, hwc⟩ := hE q hq' c t₁ w₁ hEq
      simp only at h
      split at h
      · rename_i ht₁
        rcases hs : (T.splitAt d p).sift read (w₁ * FreeMonoid.of c) with t' | z <;>
          rw [hs] at h <;> simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq,
            Prod.mk.injEq, reduceCtorEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨sift_splitAt_ne read hw hne, hs⟩
      · rename_i ht₁
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        exact ⟨sift_splitAt_ne read hw hne, sift_splitAt_ne read hwc ht₁⟩
  · simp [hpq] at h

variable [DecidableEq α]

theorem edgesOK_upd {T : DTree α} {E : Edges α} (hE : EdgesOK read T E) {p t : List Bool} {c : α}
    {u : FreeMonoid α} (hu : T.sift read u = .inl p)
    (huc : T.sift read (u * FreeMonoid.of c) = .inl t) :
    EdgesOK read T (Function.update E (p, c) (some (t, u))) := by
  intro q hq c' t' w h
  by_cases hqc : (q, c') = (p, c)
  · simp only [Prod.mk.injEq] at hqc
    obtain ⟨rfl, rfl⟩ := hqc
    rw [Function.update_self] at h
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    exact ⟨hu, huc⟩
  · rw [Function.update_of_ne hqc] at h
    exact hE q hq c' t' w h

/-- The rates are chances, the undecided reads' below the disagreements', and the level strictly
between 0 and 1. -/
structure TestsOK : Prop where
  harvestRate_nonneg : 0 ≤ C.harvestRate
  harvestRate_lt : C.harvestRate < C.disagreementRate
  disagreementRate_le : C.disagreementRate ≤ 1
  testLevel_pos : 0 < C.testLevel
  testLevel_lt : C.testLevel < 1

/-- The probes by which a stretch ends. -/
noncomputable def Level2Settings.ends : ℕ :=
  stretchLength C.harvestRate C.disagreementRate C.testLevel

theorem one_sub_binomSfGe_succ {n d : ℕ} (hd : d ≤ n) (p : ℝ) :
    1 - binomSfGe n p (d + 1) = binomSfGe n (1 - p) (n - d) := by
  have h := binomSfGe_zero_right n p
  unfold binomSfGe at h ⊢
  rw [show Finset.Icc 0 n = Finset.Icc 0 d ∪ Finset.Icc (d + 1) n by ext i; simp; omega,
    Finset.sum_union (by rw [Finset.disjoint_left]; intro i; simp; omega)] at h
  rw [← eq_sub_of_add_eq h]
  refine Finset.sum_nbij' (fun i => n - i) (fun j => n - j) ?_ ?_ ?_ ?_ ?_
  · intro i hi; simp at hi ⊢; omega
  · intro j hj; simp at hj ⊢; omega
  · intro i hi; simp at hi; omega
  · intro j hj; simp at hj; omega
  · intro i hi
    simp only [Finset.mem_Icc] at hi
    rw [Nat.choose_symm (by omega), Nat.sub_sub_self (by omega), sub_sub_cancel]
    ring

/-- Hoeffding's lower tail. -/
theorem one_sub_binomSfGe_le {n d : ℕ} {p τ : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hτ : 0 ≤ τ)
    (hd : d ≤ n) (h : (d : ℝ) + n * τ ≤ n * p) :
    1 - binomSfGe n p (d + 1) ≤ Real.exp (-2 * n * τ ^ 2) := by
  rw [one_sub_binomSfGe_succ hd]
  refine binomSfGe_le n (n - d) (1 - p) τ (by linarith) (by linarith) hτ ?_
  rw [Nat.cast_sub hd]
  linarith

variable {C} in
/-- At look `ends`, whatever the disagreements counted, one of the tests decides. -/
theorem ends_spec (hC : TestsOK C) :
    ∀ d ≤ C.ends, binomSfGe C.ends C.harvestRate d < C.testLevel
      ∨ rateSide C.disagreementRate C.testLevel 0 C.ends d = some false := by
  have hy0 := hC.harvestRate_nonneg
  have hε1 := hC.disagreementRate_le
  have hyε := hC.harvestRate_lt
  have ha0 := hC.testLevel_pos
  have ha1 := hC.testLevel_lt
  have hne : {n | ∀ d ≤ n, binomSfGe n C.harvestRate d < C.testLevel
      ∨ rateSide C.disagreementRate C.testLevel 0 n d = some false}.Nonempty := by
    set τ := (C.disagreementRate - C.harvestRate) / 2 with hτdef
    have hτ : 0 < τ := by linarith
    have hm : 0 < min C.testLevel (1 - C.testLevel) := lt_min ha0 (by linarith)
    obtain ⟨n, hn⟩ := exists_nat_gt (-Real.log (min C.testLevel (1 - C.testLevel)) / (2 * τ ^ 2))
    have hexp : Real.exp (-2 * n * τ ^ 2) < min C.testLevel (1 - C.testLevel) := by
      rw [← Real.exp_log hm]
      refine Real.exp_lt_exp.2 ?_
      have h2 : 0 < 2 * τ ^ 2 := by positivity
      rw [div_lt_iff₀ h2] at hn
      linarith
    have hexpa := hexp.trans_le (min_le_left _ _)
    have hexpb := hexp.trans_le (min_le_right _ _)
    refine ⟨n, fun d hd => ?_⟩
    by_cases hdy : (n : ℝ) * (C.harvestRate + τ) ≤ d
    · exact Or.inl ((binomSfGe_le n d C.harvestRate τ hy0 (by linarith) hτ.le hdy).trans_lt hexpa)
    · push_neg at hdy
      have hε : (n : ℝ) * C.disagreementRate = n * C.harvestRate + 2 * (n * τ) := by
        rw [hτdef]; ring
      have hlow := one_sub_binomSfGe_le (p := C.disagreementRate) (by linarith) hε1 hτ.le hd
        (by nlinarith)
      right
      unfold rateSide
      rw [if_pos (Nat.zero_le n), if_neg, if_pos (hlow.trans_lt hexpa)]
      have := binomSfGe_antitone (n := n) (by linarith) hε1 d
      linarith
  intro d hd
  exact Nat.sInf_mem hne d hd

variable {C} in
theorem one_le_ends (hC : TestsOK C) : 1 ≤ C.ends := by
  by_contra h
  have h0 : C.ends = 0 := by omega
  rcases ends_spec hC 0 (by omega) with h1 | h1 <;> rw [h0] at h1
  · rw [binomSfGe_zero_right] at h1
    linarith [hC.testLevel_lt]
  · unfold rateSide at h1
    rw [if_pos le_rfl, binomSfGe_zero_right, if_neg (by linarith [hC.testLevel_lt]),
      binomSfGe_zero_left, if_neg (by linarith [hC.testLevel_lt])] at h1
    cases h1

/-- The rate test calls the undecided reads counted above `harvestRate`. -/
def Fires (s : Level2State α) : Prop :=
  binomSfGe s.probes C.harvestRate s.undecided < C.testLevel

/-- The rate test calls the disagreements counted below `disagreementRate`. -/
def Below (s : Level2State α) : Prop :=
  rateSide C.disagreementRate C.testLevel 0 s.probes s.disagreements = some false

structure Inv (s : Level2State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  reached : Reached read s.tree
  dis_le_undecided : s.disagreements ≤ s.undecided
  dis_le_probes : s.disagreements ≤ s.probes
  probes_lt : s.probes < C.ends

/-- The probe disagrees at decided reads, or reaches an unlearned edge: it changes the
hypothesis. -/
def Changes : Outcome α → Prop
  | .wrongEdge .. => True
  | .unlearnedEdge .. => True
  | _ => False

/-- `s` with an outcome that keeps the hypothesis counted. -/
def counted (s : Level2State α) (o : Outcome α) : Level2State α :=
  ⟨s.tree, s.edges, s.undecided + o.undecidedStrings.length,
    s.disagreements + (if o.clean then 0 else 1), s.probes + 1⟩

/-! ## The potential -/

section Potential

variable [Fintype α] (nσ : ℕ)

/-- The leaves' edges not yet learned. -/
def unlearnedOf (T : DTree α) (E : Edges α) : ℕ :=
  ((T.leaves.toFinset ×ˢ (univ : Finset α)).filter fun qc => E qc = none).card

def unlearned (s : Level2State α) : ℕ := unlearnedOf s.tree s.edges

def X : ℕ := (nσ + 2) * Fintype.card α + 1

def Ψ (s : Level2State α) : ℕ := (nσ + 2 - s.tree.leaves.length) * X (α := α) nσ + unlearned s

noncomputable def Θ (s : Level2State α) : ℕ := Ψ nσ s * C.ends + (C.ends - 1 - s.probes)

omit [DecidableEq α] in
theorem unlearned_le (s : Level2State α) : unlearned s ≤ s.tree.leaves.length * Fintype.card α := by
  unfold unlearned unlearnedOf
  refine (card_filter_le _ _).trans ?_
  rw [card_product, card_univ]
  exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)

/-- Learning an edge of a leaf lowers `Ψ`. -/
theorem Ψ_lt_learn {s : Level2State α} {p : List Bool} {c : α} (hp : p ∈ s.tree.leaves)
    (hE : s.edges (p, c) = none) (v : List Bool × FreeMonoid α) :
    Ψ nσ (newHypothesis s.tree (Function.update s.edges (p, c) (some v))) < Ψ nσ s := by
  have : unlearnedOf s.tree (Function.update s.edges (p, c) (some v))
      < unlearnedOf s.tree s.edges := by
    unfold unlearnedOf
    have hsub : (s.tree.leaves.toFinset ×ˢ (univ : Finset α)).filter
          (fun qc => Function.update s.edges (p, c) (some v) qc = none)
        ⊆ (s.tree.leaves.toFinset ×ˢ (univ : Finset α)).filter fun qc => s.edges qc = none := by
      rw [Finset.subset_iff]
      intro qc hqc
      rw [mem_filter] at hqc ⊢
      refine ⟨hqc.1, ?_⟩
      by_cases hq : qc = (p, c)
      · subst hq; simp at hqc
      · rw [Function.update_of_ne hq] at hqc; exact hqc.2
    refine card_lt_card ((ssubset_iff_of_subset hsub).2
      ⟨(p, c), mem_filter.2 ⟨by simp [hp], hE⟩, ?_⟩)
    simp
  unfold Ψ unlearned
  simp only [newHypothesis] at this ⊢
  omega

/-- A split within `|Q| + 2` leaves lowers `Ψ`, whatever the new edges. -/
theorem Ψ_lt_split {s : Level2State α} {p : List Bool} (hp : p ∈ s.tree.leaves) {d : FreeMonoid α}
    (hsz : (s.tree.splitAt d p).leaves.length ≤ nσ + 2) (E : Edges α) :
    Ψ nσ (newHypothesis (s.tree.splitAt d p) E) < Ψ nσ s := by
  unfold Ψ
  have hlen := DTree.length_leaves_splitAt d s.tree p hp
  have hu := unlearned_le (newHypothesis (s.tree.splitAt d p) E)
  simp only [newHypothesis] at hu ⊢
  rw [hlen] at hu hsz ⊢
  have h1 : (s.tree.leaves.length + 1) * Fintype.card α < X (α := α) nσ := by
    unfold X
    have := Nat.mul_le_mul_right (Fintype.card α) hsz
    omega
  have h2 : nσ + 2 - s.tree.leaves.length = nσ + 2 - (s.tree.leaves.length + 1) + 1 := by omega
  rw [h2, Nat.add_mul, one_mul]
  omega

theorem Θ_lt_same {s s' : Level2State α} (ht : s'.tree = s.tree) (he : s'.edges = s.edges)
    (hp : s'.probes = s.probes + 1) (hlt : s'.probes < C.ends) : Θ C nσ s' < Θ C nσ s := by
  have hΨ : Ψ nσ s' = Ψ nσ s := by unfold Ψ unlearned; rw [ht, he]
  unfold Θ
  rw [hΨ]
  omega

theorem Θ_lt_change {s : Level2State α} (hs : s.probes < C.ends) (T : DTree α)
    (E : Edges α) (hΨ : Ψ nσ (newHypothesis T E) < Ψ nσ s) :
    Θ C nσ (newHypothesis T E) < Θ C nσ s := by
  unfold Θ
  have := Nat.mul_le_mul_right C.ends (show Ψ nσ (newHypothesis T E) + 1 ≤ Ψ nσ s by omega)
  simp only [newHypothesis] at this ⊢
  rw [Nat.add_mul, one_mul] at this
  omega

omit [DecidableEq α] in
theorem leaves_start : (initialState : Level2State α).tree.leaves = [[false], [true]] := by
  simp [initialState, newHypothesis, DTree.leaves]

theorem Ψ_start : Ψ nσ (initialState : Level2State α) + 1 ≤ stretches nσ (Fintype.card α) := by
  have hu := unlearned_le (initialState : Level2State α)
  rw [leaves_start] at hu
  unfold Ψ stretches X
  rw [leaves_start]
  simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hu ⊢
  rw [show nσ + 2 - 2 = nσ by omega]
  omega

theorem Θ_start (hn : 1 ≤ C.ends) :
    Θ C nσ (initialState : Level2State α) < stretches nσ (Fintype.card α) * C.ends := by
  have hu := unlearned_le (initialState : Level2State α)
  rw [leaves_start] at hu
  have hΨ : Ψ nσ (initialState : Level2State α) ≤ stretches nσ (Fintype.card α) - 1 := by
    unfold Ψ stretches X
    rw [leaves_start]
    simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hu ⊢
    rw [show nσ + 2 - 2 = nσ by omega]
    omega
  have hs : 1 ≤ stretches nσ (Fintype.card α) := by unfold stretches; omega
  unfold Θ
  have := Nat.mul_le_mul_right C.ends hΨ
  rw [Nat.sub_mul, one_mul] at this
  have : C.ends ≤ stretches nσ (Fintype.card α) * C.ends :=
    Nat.le_mul_of_pos_left _ hs
  omega

end Potential

/-! ## One step -/

section Step

variable {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}

theorem inv_fresh {T : DTree α} {E : Edges α} (hn : 1 ≤ C.ends) (hE : EdgesOK read T E)
    (hr : Reached read T) : Inv read C (newHypothesis T E) :=
  ⟨hE, hr, le_rfl, le_rfl, by simp [newHypothesis]; omega⟩

omit [DecidableEq α] in
theorem leaves_le_two {T : DTree α} [Fintype α] (hI : IdealReads read M side)
    (hr : Reached read T) : T.leaves.length ≤ Fintype.card σ + 2 :=
  leaves_le_of_reached read hI hr

/-- A step that changes the hypothesis: a split or a learned edge, keeping `Inv`, lowering `Ψ`. -/
theorem step_change [Fintype α] (hI : IdealReads read M side) (hn : 1 ≤ C.ends)
    {s : Level2State α} (hs : Inv read C s) {x : FreeMonoid α}
    (hc : Changes (probeOrAgree false read s.tree s.edges C.walkStart x)) :
    ∃ T E, applyProbe read C s x = .inl (newHypothesis T E) ∧ Inv read C (newHypothesis T E)
      ∧ Ψ (Fintype.card σ) (newHypothesis T E) < Ψ (Fintype.card σ) s := by
  have hOK := probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe
  revert hc hOK
  generalize probeOrAgree false read s.tree s.edges C.walkStart x = o
  intro hc hOK
  cases o with
  | agree => exact hc.elim
  | undecidedStart z => exact hc.elim
  | undecided zs cl => exact hc.elim
  | unlearnedEdge p c u t =>
    obtain ⟨hp, hu, huc, hE⟩ := hOK
    exact ⟨_, _, rfl, inv_fresh read C hn (edgesOK_upd read hs.edges hu huc) hs.reached,
      Ψ_lt_learn _ hp hE _⟩
  | wrongEdge p c u t =>
    obtain ⟨hp, hu, huc, t₀, w₀, hE, ht₀⟩ := hOK
    obtain ⟨hw₀, hw₀c⟩ := hs.edges p hp c t₀ w₀ hE
    have hpart := DTree.sift_part read s.tree _ _ t t₀ huc hw₀c (Ne.symm ht₀)
    have hreach : Reached read
        (s.tree.splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) p) := by
      refine reached_splitAt read hs.reached hp hu hw₀ ?_ ?_ ?_
      · simpa [mul_assoc] using hpart.1
      · simpa [mul_assoc] using hpart.2.1
      · simpa [mul_assoc] using hpart.2.2
    simp only [hE]
    exact ⟨_, _, rfl, inv_fresh read C hn (edgesOK_retarget read hs.edges hp _) hreach,
      Ψ_lt_split _ hp (leaves_le_two read hI hreach) _⟩

omit [DecidableEq α] in
theorem rateSide_zero (θ a : ℝ) (n h : ℕ) :
    rateSide θ a 0 n h = some true ↔ binomSfGe n θ h < a := by
  unfold rateSide
  simp only [zero_le, if_true]
  split_ifs with h1 h2 <;> simp [*]

omit [DecidableEq α] in
/-- Counting a probe that keeps the hypothesis keeps `Inv` where neither test decides. -/
theorem inv_counted (hC : TestsOK C) {s : Level2State α} (hs : Inv read C s) {o : Outcome α}
    (hok : OutcomeOK read s.tree s.edges o) (hc : ¬ Changes o) (hF : ¬ Fires C (counted s o))
    (hB : ¬ Below C (counted s o)) : Inv read C (counted s o) := by
  have hdu : s.disagreements + (if o.clean then 0 else 1)
      ≤ s.undecided + o.undecidedStrings.length := by
    have := hs.dis_le_undecided
    cases o with
    | wrongEdge => exact hc.elim trivial
    | unlearnedEdge => exact hc.elim trivial
    | agree => simp [Outcome.clean]; omega
    | undecidedStart z => simp [Outcome.clean]; omega
    | undecided zs cl =>
      cases cl
      · have : zs ≠ [] := hok.1 rfl
        have := List.length_pos_iff.2 this
        simp [Outcome.clean, Outcome.undecidedStrings]; omega
      · simp [Outcome.clean]; omega
  have hdp : s.disagreements + (if o.clean then 0 else 1) ≤ s.probes + 1 := by
    have := hs.dis_le_probes; split_ifs <;> omega
  refine ⟨hs.edges, hs.reached, hdu, hdp, ?_⟩
  show s.probes + 1 < C.ends
  have hlt := hs.probes_lt
  by_contra hge
  have heq : s.probes + 1 = C.ends := by omega
  rcases ends_spec hC _ (heq ▸ hdp) with h | h
  · apply hF
    show binomSfGe (s.probes + 1) C.harvestRate (s.undecided + o.undecidedStrings.length)
      < C.testLevel
    rw [heq]
    exact lt_of_le_of_lt (binomSfGe_antitone' hC.harvestRate_nonneg
      (by linarith [hC.harvestRate_lt, hC.disagreementRate_le]) hdu) h
  · exact hB (by show rateSide _ _ 0 (s.probes + 1) _ = some false; rw [heq]; exact h)

/-- A step that keeps the hypothesis counts the probe: the round goes on with neither test
deciding, harvests with the undecided reads' test firing, or ends consistent with the
disagreements' test calling them below. -/
theorem step_same (hC : TestsOK C) {s : Level2State α} (hs : Inv read C s) {x : FreeMonoid α}
    (hc : ¬ Changes (probeOrAgree false read s.tree s.edges C.walkStart x)) :
    let s' := counted s (probeOrAgree false read s.tree s.edges C.walkStart x)
    (applyProbe read C s x = .inl s' ∧ Inv read C s' ∧ ¬ Fires C s')
      ∨ (applyProbe read C s x = .inr (.harvest ()) ∧ Fires C s')
      ∨ (applyProbe read C s x = .inr .consistent ∧ Below C s' ∧ ¬ Fires C s') := by
  intro s'
  have hOK := probeOrAgree_ok false read s.tree s.edges C.walkStart hs.edges x
  have hs' : s' = counted s (probeOrAgree false read s.tree s.edges C.walkStart x) := rfl
  clear_value s'
  unfold applyProbe
  revert hc hs' hOK
  generalize probeOrAgree false read s.tree s.edges C.walkStart x = o
  intro hc hOK hs'
  subst hs'
  rcases o with _ | _ | _ | z | ⟨zs, _ | _⟩
  all_goals first | exact absurd trivial hc | skip
  all_goals
    simp only
    unfold keep
    dsimp only
    split_ifs with h1 h2
    · exact Or.inr (Or.inl ⟨rfl, (rateSide_zero _ _ _ _).1 h1⟩)
    · exact Or.inr (Or.inr ⟨rfl, h2, fun h => h1 ((rateSide_zero _ _ _ _).2 h)⟩)
    · exact Or.inl ⟨rfl, inv_counted read C hC hs hOK hc
        (fun h => h1 ((rateSide_zero _ _ _ _).2 h)) h2,
        fun h => h1 ((rateSide_zero _ _ _ _).2 h)⟩

end Step

end Ideal

end OrthoDFA
