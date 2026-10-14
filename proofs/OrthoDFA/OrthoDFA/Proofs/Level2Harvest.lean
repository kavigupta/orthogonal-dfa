import OrthoDFA.Proofs.Level2Steps
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 2's sampler

Every undecided string a probe walked from `k` turns up is a prefix of the probe at least `k`
long, extended by a midfix, so the sampler is spread from `k`, whatever the hypothesis. It finds at
most one string a probe, read undecided, so with reads fixed by the target state all it finds is
at bad states, and it finds one exactly when the round would count one.
-/

namespace OrthoDFA

namespace Level2

open MeasureTheory
open scoped ENNReal

variable {α : Type*} (read : FreeMonoid α → ARU)

variable [DecidableEq α] [Countable α] (D : Measure (FreeMonoid α)) (H : HypothesisDFA α) (k : ℕ)

theorem stuckString_fresh (x : FreeMonoid α) : (stuckString read H k x).Nodup
    ∧ ∀ z ∈ stuckString read H k x, z.toList.take k = x.toList.take k := by
  unfold stuckString probeOrAgree
  split_ifs with hkx
  · simp [Outcome.undecidedStrings]
  · refine ⟨?_, fun z hz => preFrom_take (by omega) (probe_pre read H.tree H.edges k x z hz)⟩
    have := probe_length read H.tree H.edges k x
    rcases h : (probe read H.tree H.edges k x).undecidedStrings with _ | ⟨a, _ | ⟨b, l⟩⟩
    · simp
    · simp
    · rw [h] at this; simp at this

theorem stuckString_length (x : FreeMonoid α) : (stuckString read H k x).length ≤ 1 := by
  unfold stuckString probeOrAgree
  split_ifs
  · simp [Outcome.undecidedStrings]
  · exact probe_length read H.tree H.edges k x

theorem stuckString_undecided (hE : EdgesOK read H.tree H.edges) (x : FreeMonoid α) :
    ∀ z ∈ stuckString read H k x, read z = .undecided := by
  have hOK := probeOrAgree_ok read H.tree H.edges k hE x
  unfold stuckString
  intro z hz
  revert hOK hz
  generalize probeOrAgree read H.tree H.edges k x = o
  rcases o with _ | _ | _ | z' | ⟨zs, cl⟩ <;>
    simp only [Outcome.undecidedStrings, List.not_mem_nil, false_imp_iff, imp_true_iff]
  · intro hOK hz
    rw [List.mem_singleton.1 hz]
    exact hOK
  · intro hOK hz
    exact hOK.2 z hz

theorem harvestLaw_apply (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    harvestLaw read D k H A
      = ∫⁻ x, (((stuckString read H k x).filter (· ∈ A)).length : ℝ≥0∞) ∂D :=
  bind_diracSum_apply D _ A

theorem harvestLaw_univ_le [IsProbabilityMeasure D] : harvestLaw read D k H Set.univ ≤ 1 := by
  classical
  rw [harvestLaw_apply]
  calc _ ≤ ∫⁻ _, (1 : ℝ≥0∞) ∂D := lintegral_mono fun x => by
          have := (List.length_filter_le (· ∈ Set.univ) (stuckString read H k x)).trans
            (stuckString_length read H k x)
          calc _ ≤ ((1 : ℕ) : ℝ≥0∞) := Nat.cast_le.2 this
            _ = 1 := by norm_num
    _ = 1 := by simp

theorem harvestLaw_spread [IsFiniteMeasure D] : Spread D k (harvestLaw read D k H) :=
  spread_of_fresh D k _ (fun x => (stuckString_fresh read H k x).1)
    (fun x => (stuckString_fresh read H k x).2)

/-- The sampler finds a string as often as a probe stops at an undecided read. -/
theorem stuck_le_harvestLaw :
    D {x | stuckString read H k x ≠ []} ≤ harvestLaw read D k H Set.univ := by
  classical
  rw [harvestLaw_apply, ← lintegral_indicator_one (Set.to_countable _).measurableSet]
  refine lintegral_mono fun x => ?_
  by_cases hx : stuckString read H k x = []
  · simp [hx]
  · rw [Set.indicator_of_mem hx, Pi.one_apply, List.filter_eq_self.2 (by simp)]
    exact Nat.one_le_cast.2 (List.length_pos_iff.2 hx)

/-- With reads fixed by the target state, all the sampler finds is at bad states, so the harvest
ends well once the sampler yields at least `y > 0`. -/
theorem endsWell_harvest {σ : Type*} {M : DFA α σ} {side : σ → Bool}
    (hI : ReadsByState read M side) [IsProbabilityMeasure D] (hE : EdgesOK read H.tree H.edges)
    (ε : ℝ) (dis : Set (FreeMonoid α)) {y : ℝ} (hy : 0 < y)
    (hlaw : y ≤ (harvestLaw read D k H).real Set.univ) :
    EndsWell M (AllUndecided read M) D ε dis k (fun _ => harvestLaw read D k H) y
      (.harvest ()) := by
  classical
  have hbad : harvestLaw read D k H {z | AllUndecided read M (M.eval z.toList)}
      = harvestLaw read D k H Set.univ := by
    refine bind_diracSum_of_forall D _ _ fun x z hz => ?_
    rcases hI (M.eval z.toList) with h | h
    · exact h
    · have := h z rfl
      rw [stuckString_undecided read H k hE x z hz] at this
      split_ifs at this
  refine ⟨?_, harvestLaw_spread read D H k, hlaw⟩
  rw [measureReal_def, measureReal_def, hbad]
  rw [measureReal_def] at hlaw
  linarith

end Level2

end OrthoDFA
