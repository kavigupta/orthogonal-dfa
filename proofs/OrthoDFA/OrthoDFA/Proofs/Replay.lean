import OrthoDFA.Stage

/-!
# The round's outcome

Every string a replay reads extends a prefix of the draw at least as long as its earliest anchor
(`take_of_mem_replayReads`), so reading a fixed `t` needs the draw to start as `t` does, up to
some point at or past the anchor.

A draw the gate counts against the hypothesis is one whose replay harvests, anchored no earlier
than any point before the gate's reading and the walk first part (`replay_harvests`): a read the
band decides is on the side of the middle of the band, so a replay that decides every read
follows the gate's own reading, and parts from the walk where the gate does.
-/

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

namespace DTree

theorem sift_node_none {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    (h : cut (x * m) = none) : (node m r a).sift cut x = .inr (x * m) := by
  simp [sift, route, h]

theorem sift_node_some {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    {b : Bool} (h : cut (x * m) = some b) :
    (node m r a).sift cut x = ((if b then a else r).sift cut x).map (b :: ·) id := by
  cases b <;> simp [sift, route, h]

/-- What a node cannot place is the string extended by its midfix. -/
theorem sift_inr {cut : FreeMonoid α → Option Bool} :
    ∀ (t : DTree α) {x b : FreeMonoid α}, t.sift cut x = .inr b → ∃ m, b = x * m
  | .leaf, x, b, h => by simp [sift, route] at h
  | .node m r a, x, b, h => by
    rcases hc : cut (x * m) with _ | c
    · rw [sift_node_none hc] at h
      simp only [Sum.inr.injEq] at h
      exact ⟨m, h.symm⟩
    · rw [sift_node_some hc] at h
      cases c
      · simp only [Bool.false_eq_true, ↓reduceIte] at h
        rcases hr : r.sift cut x with q | q <;> rw [hr] at h <;> simp at h
        subst h
        exact sift_inr r hr
      · simp only [↓reduceIte] at h
        rcases ha : a.sift cut x with q | q <;> rw [ha] at h <;> simp at h
        subst h
        exact sift_inr a ha

end DTree

variable (R : CutReads α)

instance (L : ℕ) : IsFiniteMeasure (anchorLaw L) := by
  constructor
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp [anchorLaw]
  · simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
      measure_univ, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one, smul_eq_mul]
    rw [ENNReal.inv_mul_cancel (by exact_mod_cast hL.ne') (by simp)]
    exact ENNReal.one_lt_top

instance : Countable (FreeMonoid α) := inferInstanceAs (Countable (List α))

instance : MeasurableSingletonClass (FreeMonoid α) := ⟨fun _ => trivial⟩

end OrthoDFA
