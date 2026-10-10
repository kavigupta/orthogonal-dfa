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

end Depth

noncomputable instance (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α)) (ω : Ω)
    (z : FreeMonoid α) : Decidable (midRead O B F ω z) :=
  inferInstanceAs (Decidable (B.lo + B.hi < 2 * _))

end OrthoDFA
