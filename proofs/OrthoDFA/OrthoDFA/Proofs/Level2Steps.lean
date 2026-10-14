import OrthoDFA.Level2Claim
import OrthoDFA.Proofs.Probe
import OrthoDFA.Proofs.SeqTest

/-!
# Level 2's steps

The invariant `Inv`: every learned edge out of a leaf points where its witness's successor sifts,
every leaf past the first two is reached by some string, the stretch's disagreements are at most
its undecided reads, which are at most its probes, and it is shorter than `ends`, the look by
which, with `harvestRate < alternativeRate < disagreementRate`, one of the sequential tests fires
whatever the counts.

A disagreement at decided reads at the edge `(p, c)` has two strings at `p`, the probe's prefix and
the edge's witness, whose successors sift to different leaves. They read decidedly apart at `c`
and the midfix where those leaves part, so both new leaves of the split are reached, and with
reads fixed by the target state the reached leaves are at most `|Q|`. Each step that does not end
the round lowers `Θ`: a split or a learned edge lowers `Ψ`, which counts the splits left and the
unlearned edges, and a probe that keeps the hypothesis adds to the stretch, which stays below
`ends`.
-/

namespace OrthoDFA

namespace Level2

open Finset

variable {α : Type*} (read : FreeMonoid α → ARU) (C : Settings)

/-- A read is fixed by the target state. -/
theorem read_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hI : ReadsByState read M side)
    {z z' : FreeMonoid α} (h : M.eval z.toList = M.eval z'.toList) : read z = read z' := by
  rcases hI (M.eval z.toList) with hq | hq
  · rw [hq z rfl, hq z' h.symm]
  · rw [hq z rfl, hq z' h.symm]

theorem read_mul_congr {σ : Type*} {M : DFA α σ} {side : σ → Bool} (hI : ReadsByState read M side)
    {z z' : FreeMonoid α} (h : M.eval z.toList = M.eval z'.toList) (m : FreeMonoid α) :
    read (z * m) = read (z' * m) := by
  apply read_congr read hI
  simp only [FreeMonoid.toList_mul, DFA.eval, DFA.evalFrom_of_append]
  exact congrArg (M.evalFrom · m.toList) h

/-! ## The tree's size -/

theorem leaves_le_of_reached {σ : Type*} [Fintype σ] {M : DFA α σ} {side : σ → Bool}
    (hI : ReadsByState read M side) {T : DTree α} (h : Reached read T) :
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

variable [DecidableEq α]

/-- The rates are chances, in the order `harvestRate < alternativeRate < disagreementRate`, and the
level is strictly between 0 and 1. -/
structure TestsOK : Prop where
  harvestRate_pos : 0 < C.harvestRate
  harvestRate_lt : C.harvestRate < C.alternativeRate
  alternativeRate_lt : C.alternativeRate < C.disagreementRate
  disagreementRate_lt : C.disagreementRate < 1
  testLevel_pos : 0 < C.testLevel
  testLevel_lt : C.testLevel < 1

/-- The probes by which a stretch ends. -/
noncomputable def Settings.ends : ℕ :=
  stretchLength C.harvestRate C.alternativeRate C.disagreementRate C.testLevel

variable {C} in
/-- At look `ends`, whatever the disagreements counted, one of the tests fires. -/
theorem ends_spec (hC : TestsOK C) :
    ∀ d ≤ C.ends, seqRate C.harvestRate C.alternativeRate C.testLevel C.ends d
      ∨ seqRate C.disagreementRate C.alternativeRate C.testLevel C.ends d := by
  have hne : {n | ∀ d ≤ n, seqRate C.harvestRate C.alternativeRate C.testLevel n d
      ∨ seqRate C.disagreementRate C.alternativeRate C.testLevel n d}.Nonempty :=
    ⟨_, seqRate_decides hC.harvestRate_pos hC.harvestRate_lt hC.alternativeRate_lt
      hC.disagreementRate_lt hC.testLevel_pos hC.testLevel_lt le_rfl⟩
  intro d hd
  exact Nat.sInf_mem hne d hd

variable {C} in
theorem one_le_ends (hC : TestsOK C) : 1 ≤ C.ends := by
  by_contra h
  have h0 : C.ends = 0 := by omega
  have ha : 1 < 1 / C.testLevel := by
    rw [lt_div_iff₀ hC.testLevel_pos]; linarith [hC.testLevel_lt]
  rcases ends_spec hC 0 (by omega) with h1 | h1 <;> rw [h0] at h1 <;>
    simp only [seqRate, likelihoodRatio_zero] at h1 <;> linarith

/-- The undecided reads' test fires. -/
def Fires (s : State α) : Prop :=
  seqRate C.harvestRate C.alternativeRate C.testLevel s.probes s.undecided

/-- The disagreements' test fires. -/
def Below (s : State α) : Prop :=
  seqRate C.disagreementRate C.alternativeRate C.testLevel s.probes s.disagreements

structure Inv (s : State α) : Prop where
  edges : EdgesOK read s.tree s.edges
  reached : Reached read s.tree
  dis_le_undecided : s.disagreements ≤ s.undecided
  undecided_le_probes : s.undecided ≤ s.probes
  probes_lt : s.probes < C.ends

/-- The probe disagrees at decided reads, or reaches an unlearned edge: it changes the
hypothesis. -/
def Changes : Outcome α → Prop
  | .wrongEdge .. => True
  | .unlearnedEdge .. => True
  | _ => False

/-- `s` with an outcome that keeps the hypothesis counted. -/
def counted (s : State α) (o : Outcome α) : State α :=
  ⟨s.tree, s.edges, s.undecided + o.undecidedStrings.length,
    s.disagreements + (if o.clean then 0 else 1), s.probes + 1⟩

/-! ## The potential -/

section Potential

variable [Fintype α] (nσ : ℕ)

/-- The leaves' edges not yet learned. -/
def unlearnedOf (T : DTree α) (E : Edges α) : ℕ :=
  ((T.leaves.toFinset ×ˢ (univ : Finset α)).filter fun qc => E qc = none).card

def unlearned (s : State α) : ℕ := unlearnedOf s.tree s.edges

def X : ℕ := (nσ + 2) * Fintype.card α + 1

def Ψ (s : State α) : ℕ := (nσ + 2 - s.tree.leaves.length) * X (α := α) nσ + unlearned s

noncomputable def Θ (s : State α) : ℕ := Ψ nσ s * C.ends + (C.ends - 1 - s.probes)

omit [DecidableEq α] in
theorem unlearned_le (s : State α) : unlearned s ≤ s.tree.leaves.length * Fintype.card α := by
  unfold unlearned unlearnedOf
  refine (card_filter_le _ _).trans ?_
  rw [card_product, card_univ]
  exact Nat.mul_le_mul_right _ (List.toFinset_card_le _)

/-- Learning an edge of a leaf lowers `Ψ`. -/
theorem Ψ_lt_learn {s : State α} {p : List Bool} {c : α} (hp : p ∈ s.tree.leaves)
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
theorem Ψ_lt_split {s : State α} {p : List Bool} (hp : p ∈ s.tree.leaves) {d : FreeMonoid α}
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

theorem Θ_lt_same {s s' : State α} (ht : s'.tree = s.tree) (he : s'.edges = s.edges)
    (hp : s'.probes = s.probes + 1) (hlt : s'.probes < C.ends) : Θ C nσ s' < Θ C nσ s := by
  have hΨ : Ψ nσ s' = Ψ nσ s := by unfold Ψ unlearned; rw [ht, he]
  unfold Θ
  rw [hΨ]
  omega

theorem Θ_lt_change {s : State α} (hs : s.probes < C.ends) (T : DTree α)
    (E : Edges α) (hΨ : Ψ nσ (newHypothesis T E) < Ψ nσ s) :
    Θ C nσ (newHypothesis T E) < Θ C nσ s := by
  unfold Θ
  have := Nat.mul_le_mul_right C.ends (show Ψ nσ (newHypothesis T E) + 1 ≤ Ψ nσ s by omega)
  simp only [newHypothesis] at this ⊢
  rw [Nat.add_mul, one_mul] at this
  omega

omit [DecidableEq α] in
theorem leaves_start : (initialState : State α).tree.leaves = [[false], [true]] := by
  simp [initialState, newHypothesis, DTree.leaves]

theorem Ψ_start : Ψ nσ (initialState : State α) + 1 ≤ stretches nσ (Fintype.card α) := by
  have hu := unlearned_le (initialState : State α)
  rw [leaves_start] at hu
  unfold Ψ stretches X
  rw [leaves_start]
  simp only [List.length_cons, List.length_nil, zero_add, Nat.reduceAdd] at hu ⊢
  rw [show nσ + 2 - 2 = nσ by omega]
  omega

theorem Θ_start (hn : 1 ≤ C.ends) :
    Θ C nσ (initialState : State α) < stretches nσ (Fintype.card α) * C.ends := by
  have hu := unlearned_le (initialState : State α)
  rw [leaves_start] at hu
  have hΨ : Ψ nσ (initialState : State α) ≤ stretches nσ (Fintype.card α) - 1 := by
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

/-- A step that changes the hypothesis: a split or a learned edge, keeping `Inv`, lowering `Ψ`. -/
theorem step_change [Fintype α] (hI : ReadsByState read M side) (hn : 1 ≤ C.ends)
    {s : State α} (hs : Inv read C s) {x : FreeMonoid α}
    (hc : Changes (probeOrAgree read s.tree s.edges C.walkStart x)) :
    ∃ T E, applyProbe read C s x = .inl (newHypothesis T E) ∧ Inv read C (newHypothesis T E)
      ∧ Ψ (Fintype.card σ) (newHypothesis T E) < Ψ (Fintype.card σ) s := by
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.edges x
  unfold applyProbe
  revert hc hOK
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o
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
      Ψ_lt_split _ hp (leaves_le_of_reached read hI hreach) _⟩

omit [DecidableEq α] in
/-- Counting a probe that keeps the hypothesis keeps `Inv` where neither test fires. -/
theorem inv_counted (hC : TestsOK C) {s : State α} (hs : Inv read C s) {o : Outcome α}
    (hok : OutcomeOK read s.tree s.edges o) (hc : ¬ Changes o)
    (hlen : o.undecidedStrings.length ≤ 1) (hF : ¬ Fires C (counted s o))
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
  have hup : s.undecided + o.undecidedStrings.length ≤ s.probes + 1 := by
    have := hs.undecided_le_probes; omega
  refine ⟨hs.edges, hs.reached, hdu, hup, ?_⟩
  show s.probes + 1 < C.ends
  have hlt := hs.probes_lt
  by_contra hge
  have heq : s.probes + 1 = C.ends := by omega
  rcases ends_spec hC _ (heq ▸ hdu.trans hup) with h | h
  · apply hF
    show seqRate _ _ _ (s.probes + 1) (s.undecided + o.undecidedStrings.length)
    rw [heq]
    unfold seqRate at h ⊢
    exact h.trans (likelihoodRatio_mono hC.harvestRate_pos (by linarith [hC.harvestRate_lt,
      hC.alternativeRate_lt, hC.disagreementRate_lt]) hC.harvestRate_lt.le
      (by linarith [hC.alternativeRate_lt, hC.disagreementRate_lt]) hdu (heq ▸ hup))
  · exact hB (by show seqRate _ _ _ (s.probes + 1) _; rw [heq]; exact h)

/-- A step that keeps the hypothesis counts the probe: the round goes on with neither test firing,
harvests with the undecided reads' test firing, or ends consistent with the disagreements'. -/
theorem step_same (hC : TestsOK C) {s : State α} (hs : Inv read C s) {x : FreeMonoid α}
    (hc : ¬ Changes (probeOrAgree read s.tree s.edges C.walkStart x))
    (hlen : (probeOrAgree read s.tree s.edges C.walkStart x).undecidedStrings.length ≤ 1) :
    let s' := counted s (probeOrAgree read s.tree s.edges C.walkStart x)
    (applyProbe read C s x = .inl s' ∧ Inv read C s' ∧ ¬ Fires C s')
      ∨ (applyProbe read C s x = .inr (.harvest ()) ∧ Fires C s')
      ∨ (applyProbe read C s x = .inr .consistent ∧ Below C s' ∧ ¬ Fires C s') := by
  intro s'
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.edges x
  have hs' : s' = counted s (probeOrAgree read s.tree s.edges C.walkStart x) := rfl
  clear_value s'
  unfold applyProbe
  revert hc hs' hOK hlen
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o
  intro hc hlen hOK hs'
  subst hs'
  rcases o with _ | _ | _ | z | ⟨zs, _ | _⟩
  all_goals first | exact absurd trivial hc | skip
  all_goals
    simp only
    unfold keep
    dsimp only
    split_ifs with h1 h2
    · exact Or.inr (Or.inl ⟨rfl, h1⟩)
    · exact Or.inr (Or.inr ⟨rfl, h2, h1⟩)
    · exact Or.inl ⟨rfl, inv_counted read C hC hs hOK hc hlen h1 h2, h1⟩

end Step

end Level2

end OrthoDFA
