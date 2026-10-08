import OrthoDFA.Round
import OrthoDFA.Proofs.Replay

/-!
# The edge gap holds or a chain advances

Read at the middle of the band with every state read cleanly, an edge's reading lands off it
either rarely or nearly always, since only a flipped node read on the two paths moves it.  A
badly read state is reached from a visited string, which advances an indecision chain at some
rung of a geometric ladder.  So the round's hypothesis need not be assumed to meet the edge gap.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
variable {μ : Measure Ω} {Q : Type*}

namespace DTree

/-- Nodes on the longest root-to-leaf path. -/
def depth : DTree α → ℕ
  | .leaf => 0
  | .node _ r a => max r.depth a.depth + 1

omit [Fintype α] [DecidableEq α] in
theorem route_length_le_depth (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) (x : FreeMonoid α), (t.route cut x).1.length ≤ t.depth
  | .leaf, x => by simp [route]
  | .node m r a, x => by
    have hr := route_length_le_depth cut r x
    have ha := route_length_le_depth cut a x
    simp only [route, depth]
    split <;> simp <;> omega

omit [Fintype α] [DecidableEq α] in
theorem route_congr {c₁ c₂ : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) (x : FreeMonoid α), (∀ z ∈ (t.route c₁ x).1, c₁ z = c₂ z) →
      t.route c₁ x = t.route c₂ x
  | .leaf, _, _ => rfl
  | .node m r a, x, h => by
    have hm : c₁ (x * m) = c₂ (x * m) := h _ (by simp only [route]; split <;> simp)
    rcases hc : c₁ (x * m) with _ | ⟨_ | _⟩
    · simp only [route, hc, ← hm]
    · have := route_congr r x fun z hz => h z (by simp only [route, hc]; exact .tail _ hz)
      simp only [route, hc, ← hm, this]
    · have := route_congr a x fun z hz => h z (by simp only [route, hc]; exact .tail _ hz)
      simp only [route, hc, ← hm, this]

omit [Fintype α] [DecidableEq α] in
theorem splitAt_depth_le (m : FreeMonoid α) :
    ∀ (t : DTree α) (p : List Bool), (t.splitAt m p).depth ≤ t.depth + 1
  | .leaf, [] => by simp [splitAt, depth]
  | .leaf, _ :: _ => by simp [splitAt]
  | .node _ _ _, [] => by simp [splitAt]
  | .node n r a, false :: p => by
    have := splitAt_depth_le m r p
    simp only [splitAt, depth]
    omega
  | .node n r a, true :: p => by
    have := splitAt_depth_le m a p
    simp only [splitAt, depth]
    omega

end DTree

section Depth

variable (K : StageKnobs α) (R : CutReads α)

theorem probeStep_tree (s : PassState α) (w : FreeMonoid α) :
    (probeStep K R s w).tree = s.tree ∨ ∃ d p, (probeStep K R s w).tree = s.tree.splitAt d p := by
  simp only [probeStep, onEdge, settle]
  repeat' split
  all_goals first | exact .inl rfl | exact .inr ⟨_, _, rfl⟩

theorem probeStep_depth_le (s : PassState α) (w : FreeMonoid α) :
    (probeStep K R s w).tree.depth ≤ s.tree.depth + 1 := by
  rcases probeStep_tree K R s w with h | ⟨d, p, h⟩
  · rw [h]; omega
  · rw [h]; exact DTree.splitAt_depth_le d _ p

theorem runPass_depth_le :
    ∀ (s : PassState α) (probes : List (FreeMonoid α)),
      (runPass K R s probes).tree.depth ≤ s.tree.depth + probes.length
  | s, [] => by simp [runPass]
  | s, w :: ws => by
    have step : runPass K R s (w :: ws) = runPass K R (if K.patience ≤ s.streak then s else
        let s' := probeStep K R s w
        { s' with reads := if s'.streak = 0 then 0 else s.reads + probeReads R s w }) ws := rfl
    rw [step]
    refine (runPass_depth_le _ ws).trans ?_
    split
    · simp
    · have := probeStep_depth_le K R s w
      simp only [List.length_cons]
      omega

theorem roundEnd_depth_le (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (seed : List (FreeMonoid α)) {N : ℕ} (θ : Ω × (Fin N → FreeMonoid α)) :
    (roundEnd K O B F seed θ).tree.depth ≤ N + 1 := by
  have := runPass_depth_le K (readsAt O B F θ.1) (initialState K (readsAt O B F θ.1) seed)
    (List.ofFn θ.2)
  simp only [List.length_ofFn] at this
  have h0 : (initialState K (readsAt O B F θ.1) seed).tree.depth = 1 := rfl
  unfold roundEnd
  omega

end Depth

noncomputable instance (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (ω : Ω)
    (z : FreeMonoid α) : Decidable (midRead O B F ω z) :=
  inferInstanceAs (Decidable (B.lo + B.hi < 2 * _))

omit [Fintype α] [DecidableEq α] in
theorem measurableSet_midRead (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (z : FreeMonoid α) : MeasurableSet {ω | midRead O B F ω z} := by
  have hf : Measurable fun ω => acceptsOn F (fun w => O.mq w ω) z := by
    unfold acceptsOn
    simp_rw [Finset.card_filter]
    exact Finset.measurable_sum _ fun v _ => Measurable.ite
      (measurableSet_eq_fun (measurable_const.add (measurable_const.mul (O.noise_meas _)))
        measurable_const) measurable_const measurable_const
  exact hf (MeasurableSet.of_discrete (s := {n : ℕ | B.lo + B.hi < 2 * n}))

/-- The side a node read of `z` at the middle of the band lands on more often than not. -/
noncomputable def majRead (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (z : FreeMonoid α) : Bool :=
  decide (1 / 2 ≤ μ.real {ω | midRead O B F ω z})

/-- The node reads `x`'s sift makes when every read lands on its likelier side. -/
noncomputable def majRoute (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (H : Hypothesis α) (x : FreeMonoid α) : List (FreeMonoid α) :=
  (H.tree.route (fun y => some (majRead O B F y)) x).1

/-- Where `x` sifts when every read lands on its likelier side. -/
noncomputable def majPath (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (H : Hypothesis α) (x : FreeMonoid α) : List Bool :=
  (H.tree.sift (fun y => some (majRead O B F y)) x).elim id fun _ => []

theorem flip_le [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ : ℝ}
    (hflip : MidFlipPremise A O B F uHi φ) {z : FreeMonoid α}
    (hz : stateIndecision A O B F (A.state z) < uHi) :
    μ.real {ω | decide (midRead O B F ω z) ≠ majRead O B F z} ≤ φ := by
  have hm := measurableSet_midRead O B F z
  have hc : μ.real {ω | midRead O B F ω z}ᶜ = 1 - μ.real {ω | midRead O B F ω z} := by
    rw [measureReal_compl hm, probReal_univ]
  by_cases hp : 1 / 2 ≤ μ.real {ω | midRead O B F ω z}
  · have hmaj : majRead O B F z = true := decide_eq_true hp
    have : {ω | decide (midRead O B F ω z) ≠ majRead O B F z} = {ω | midRead O B F ω z}ᶜ := by
      ext ω; simp [hmaj]
    rw [this, hc]
    rcases hflip z hz with h | h <;> linarith
  · have hmaj : majRead O B F z = false := decide_eq_false hp
    have : {ω | decide (midRead O B F ω z) ≠ majRead O B F z} = {ω | midRead O B F ω z} := by
      ext ω; simp [hmaj]
    rw [this]
    rcases hflip z hz with h | h
    · exact h
    · linarith

omit [Fintype α] [DecidableEq α] in
theorem midPath_eq_majPath {O : Oracle μ (FreeMonoid α)} {B : State}
    {F : Finset (FreeMonoid α)} {H : Hypothesis α} {x : FreeMonoid α} {ω : Ω}
    (h : ∀ z ∈ majRoute O B F H x, decide (midRead O B F ω z) = majRead O B F z) :
    midPath (readsAt O B F ω) H x = majPath O B F H x := by
  have := DTree.route_congr (c₁ := fun y => some (majRead O B F y))
    (c₂ := fun y => some (decide (midRead O B F ω y))) H.tree x
    fun z hz => by rw [h z hz]
  unfold midPath majPath DTree.sift
  rw [this]
  rfl

theorem midPath_ne_majPath_le [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ : ℝ}
    (hflip : MidFlipPremise A O B F uHi φ) (H : Hypothesis α) (x : FreeMonoid α)
    (hclean : ∀ z ∈ majRoute O B F H x, stateIndecision A O B F (A.state z) < uHi) :
    μ.real {ω | midPath (readsAt O B F ω) H x ≠ majPath O B F H x}
      ≤ (majRoute O B F H x).length * φ := by
  classical
  calc μ.real {ω | midPath (readsAt O B F ω) H x ≠ majPath O B F H x}
      ≤ μ.real (⋃ z ∈ (majRoute O B F H x).toFinset,
          {ω | decide (midRead O B F ω z) ≠ majRead O B F z}) := by
        refine measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)
        by_contra hn
        simp only [Set.mem_iUnion, List.mem_toFinset, Set.mem_ofPred_eq, not_exists,
          not_not] at hn
        exact hω (midPath_eq_majPath hn)
    _ ≤ ∑ z ∈ (majRoute O B F H x).toFinset,
          μ.real {ω | decide (midRead O B F ω z) ≠ majRead O B F z} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _z ∈ (majRoute O B F H x).toFinset, φ :=
        Finset.sum_le_sum fun z hz => flip_le hflip (hclean z (List.mem_toFinset.1 hz))
    _ ≤ (majRoute O B F H x).length * φ := by
        rw [Finset.sum_const, nsmul_eq_mul]
        by_cases h0 : majRoute O B F H x = []
        · simp [h0]
        · obtain ⟨z, hz⟩ := List.exists_mem_of_ne_nil _ h0
          have hφ : 0 ≤ φ := measureReal_nonneg.trans (flip_le hflip (hclean z hz))
          exact mul_le_mul_of_nonneg_right
            (by exact_mod_cast List.toFinset_card_le _) hφ

/-- With every node read on both majority paths read cleanly, an edge's reading lands off it with
chance at most `ℓ·φ` or at least `1 − ℓ·φ`, `ℓ` the two paths' reads. -/
theorem edgeDisagreeProb_bimodal [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ : ℝ}
    (hflip : MidFlipPremise A O B F uHi φ) (H : Hypothesis α) (x : FreeMonoid α) (c : α)
    (hclean : ∀ z, stateIndecision A O B F (A.state z) < uHi) :
    edgeDisagreeProb O B F H x c
        ≤ ((majRoute O B F H x).length + (majRoute O B F H (x * FreeMonoid.of c)).length) * φ
      ∨ 1 - ((majRoute O B F H x).length
          + (majRoute O B F H (x * FreeMonoid.of c)).length) * φ
        ≤ edgeDisagreeProb O B F H x c := by
  set bad := {ω | midPath (readsAt O B F ω) H x ≠ majPath O B F H x}
    ∪ {ω | midPath (readsAt O B F ω) H (x * FreeMonoid.of c)
      ≠ majPath O B F H (x * FreeMonoid.of c)}
  have hbad : μ.real bad
      ≤ ((majRoute O B F H x).length + (majRoute O B F H (x * FreeMonoid.of c)).length) * φ := by
    refine (measureReal_union_le _ _).trans ?_
    rw [add_mul]
    exact add_le_add (midPath_ne_majPath_le hflip H x fun z _ => hclean z)
      (midPath_ne_majPath_le hflip H _ fun z _ => hclean z)
  by_cases hm : majPath O B F H (x * FreeMonoid.of c) = H.step (majPath O B F H x) c
  · refine .inl ((measureReal_mono (fun ω hω => ?_) (measure_ne_top _ _)).trans hbad)
    by_contra hn
    simp only [bad, Set.mem_union, Set.mem_ofPred_eq, not_or, not_not] at hn
    exact hω (by rw [hn.1, hn.2, hm])
  · refine .inr ?_
    have hsub : badᶜ ⊆ {ω | midPath (readsAt O B F ω) H (x * FreeMonoid.of c)
        ≠ H.step (midPath (readsAt O B F ω) H x) c} := fun ω hω => by
      simp only [bad, Set.mem_compl_iff, Set.mem_union, Set.mem_ofPred_eq, not_or, not_not] at hω
      simp only [Set.mem_ofPred_eq]
      rw [hω.1, hω.2]
      exact hm
    have h1 : (1 : ℝ) ≤ μ.real bad + μ.real badᶜ := by
      have hu : μ.real (Set.univ : Set Ω) = 1 := by simp
      rw [← hu, ← Set.union_compl_self bad]
      exact measureReal_union_le _ _
    have h2 := measureReal_mono (μ := μ) hsub (measure_ne_top μ _)
    unfold edgeDisagreeProb
    linarith

/-- With every state read cleanly, an edge read off its edge no more often than not is read off
it at most `2dφ` of the time, `d` the tree's depth. -/
theorem edgeDisagreeProb_le_of_clean [IsProbabilityMeasure μ] {A : DFA (FreeMonoid α) Q}
    {O : Oracle μ (FreeMonoid α)} {B : State} {F : Finset (FreeMonoid α)} {uHi φ : ℝ} {d : ℕ}
    (hflip : MidFlipPremise A O B F uHi φ) (hall : ∀ q, stateIndecision A O B F q < uHi)
    {H : Hypothesis α} (hdepth : H.tree.depth ≤ d) (hφ : 0 ≤ φ) (hd : 4 * d * φ < 1)
    {x : FreeMonoid α} {c : α} (hhalf : edgeDisagreeProb O B F H x c ≤ 1 / 2) :
    edgeDisagreeProb O B F H x c ≤ 2 * d * φ := by
  have h1 := (DTree.route_length_le_depth (fun y => some (majRead O B F y)) H.tree
    x).trans hdepth
  have h2 := (DTree.route_length_le_depth (fun y => some (majRead O B F y)) H.tree
    (x * FreeMonoid.of c)).trans hdepth
  have hl : (((majRoute O B F H x).length
      + (majRoute O B F H (x * FreeMonoid.of c)).length : ℕ) : ℝ) ≤ 2 * d := by
    unfold majRoute
    push_cast
    have : ((H.tree.route (fun y => some (majRead O B F y)) x).1.length : ℝ) ≤ d := by
      exact_mod_cast h1
    have : ((H.tree.route (fun y => some (majRead O B F y))
        (x * FreeMonoid.of c)).1.length : ℝ) ≤ d := by exact_mod_cast h2
    linarith
  have hlφ := mul_le_mul_of_nonneg_right hl hφ
  push_cast at hlφ
  rcases edgeDisagreeProb_bimodal hflip H x c (fun z => hall _) with h | h
  · linarith
  · linarith

end OrthoDFA
