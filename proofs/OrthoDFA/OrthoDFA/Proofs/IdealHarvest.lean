import OrthoDFA.Proofs.IdealSteps
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 2's sampler

Every undecided string a probe walked from `k` turns up is a prefix of the probe at least `k`
long, extended by a midfix, so the sampler is spread from `k`, whatever the hypothesis. It finds at
most one string a probe, read undecided, so with ideal reads all it finds is at bad states, and it
finds one exactly when the round would count one.
-/

namespace OrthoDFA

namespace Ideal

open MeasureTheory
open scoped ENNReal

variable {α : Type*} (recover : Bool) (read : FreeMonoid α → ARU)

variable (T : DTree α) (E : Edges α)

/-- A string the probe turns up undecided starts as the probe does, up to `p ≥ lo`. -/
def PreFrom (x : FreeMonoid α) (lo : ℕ) (z : FreeMonoid α) : Prop :=
  ∃ p, lo ≤ p ∧ ∃ m, z = prefixOf x p * m

theorem stuckAt_pre (x : FreeMonoid α) (p : ℕ) :
    ∀ z ∈ undecidedAt read T x p, ∃ m, z = prefixOf x p * m := by
  intro z hz
  unfold undecidedAt at hz
  rcases hs : T.sift read (prefixOf x p) with l | y <;> rw [hs] at hz
  · simp at hz
  · simp only [Sum.elim_inr, List.mem_singleton] at hz
    subst hz
    exact DTree.sift_inr_mul read T _ _ hs

theorem searchOutcome_pre (x : FreeMonoid α) (st : ℕ → List Bool) (un : Option (α × List Bool))
    (lo hi : ℕ) (hle : lo ≤ hi) :
    ∀ z ∈ (searchOutcome read T x st un hi
      (tripartiteSearch recover (siftAgrees read T x st) lo hi).1).undecidedStrings,
      PreFrom x lo z := by
  have hs := tripartiteSearch_ok (recover := recover) (agrees := siftAgrees read T x st) hle
  generalize (tripartiteSearch recover (siftAgrees read T x st) lo hi).1 = r at hs
  intro z hz
  have hat : ∀ p, lo ≤ p → z ∈ undecidedAt read T x p → PreFrom x lo z := fun p hp hz =>
    have ⟨m, hm⟩ := stuckAt_pre read T x p z hz
    ⟨p, hp, m, hm⟩
  cases r with
  | undecidedHi => exact hat hi hle hz
  | allAgree =>
    simp only [searchOutcome] at hz
    split at hz <;> simp [Outcome.undecidedStrings] at hz
  | edge p =>
    simp only [searchOutcome] at hz
    split at hz <;> simp [Outcome.undecidedStrings] at hz
  | triple p => exact hat p hs.1.le hz
  | undecidedMid p => exact hat p hs.2.1.le hz
  | pair p => exact hat p hs.1 hz
  | undecidedLo => simp [searchOutcome, Outcome.undecidedStrings] at hz
  | disagreeLo => simp [searchOutcome, Outcome.undecidedStrings] at hz

theorem probe_pre (k : ℕ) (x : FreeMonoid α) :
    ∀ z ∈ (probe recover read T E k x).undecidedStrings, PreFrom x k z := by
  intro z hz
  unfold probe at hz
  rcases ha : T.sift read (prefixOf x k) with a | y <;> rw [ha] at hz
  · simp only at hz
    have hpos := length_walk_pos E a (x.toList.drop k)
    split at hz
    · split at hz
      · rename_i y hy
        simp only [Outcome.undecidedStrings, List.mem_singleton] at hz
        subst hz
        obtain ⟨m, hm⟩ := DTree.sift_inr_mul read T _ _ hy
        exact ⟨_, by omega, m, hm⟩
      · exact searchOutcome_pre recover read T x _ _ k _ (by omega) z hz
    · exact searchOutcome_pre recover read T x _ _ k _ (by omega) z hz
  · simp only [Outcome.undecidedStrings, List.mem_singleton] at hz
    subst hz
    obtain ⟨m, hm⟩ := DTree.sift_inr_mul read T _ _ ha
    exact ⟨k, le_rfl, m, hm⟩

theorem preFrom_take {x z : FreeMonoid α} {k : ℕ} (hk : k ≤ x.toList.length)
    (h : PreFrom x k z) : z.toList.take k = x.toList.take k := by
  obtain ⟨p, hp, m, rfl⟩ := h
  have hlen : k ≤ (prefixOf x p).toList.length := by simp [prefixOf]; omega
  rw [FreeMonoid.toList_mul, List.take_append_of_le_length hlen]
  simp [prefixOf, List.take_take, min_eq_left hp]

theorem undecidedAt_length (x : FreeMonoid α) (p : ℕ) : (undecidedAt read T x p).length ≤ 1 := by
  unfold undecidedAt
  rcases T.sift read (prefixOf x p) with _ | _ <;> simp

theorem searchOutcome_length (x : FreeMonoid α) (st : ℕ → List Bool) (un : Option (α × List Bool))
    (hi : ℕ) (r : SearchResult) :
    (searchOutcome read T x st un hi r).undecidedStrings.length ≤ 1 := by
  have h := undecidedAt_length read T x
  cases r with
  | undecidedHi => exact h hi
  | allAgree =>
    simp only [searchOutcome]
    split <;> simp [Outcome.undecidedStrings]
  | edge p =>
    simp only [searchOutcome]
    split <;> simp [Outcome.undecidedStrings]
  | triple p => exact h _
  | undecidedMid p => exact h _
  | pair p => exact h _
  | undecidedLo => simp [searchOutcome, Outcome.undecidedStrings]
  | disagreeLo => simp [searchOutcome, Outcome.undecidedStrings]

theorem probe_length (k : ℕ) (x : FreeMonoid α) :
    (probe recover read T E k x).undecidedStrings.length ≤ 1 := by
  rcases ha : T.sift read (prefixOf x k) with a | y
  · simp only [probe, ha]
    split
    · split
      · simp [Outcome.undecidedStrings]
      · exact searchOutcome_length read T x _ _ _ _
    · exact searchOutcome_length read T x _ _ _ _
  · simp [probe, ha, Outcome.undecidedStrings]

variable [DecidableEq α] [Countable α] (D : Measure (FreeMonoid α)) (H : Hypothesis α) (k : ℕ)

theorem stuckString_fresh (x : FreeMonoid α) : (stuckString read H k x).Nodup
    ∧ ∀ z ∈ stuckString read H k x, z.toList.take k = x.toList.take k := by
  unfold stuckString probeOrAgree
  split_ifs with hkx
  · simp [Outcome.undecidedStrings]
  · refine ⟨?_, fun z hz => preFrom_take (by omega) (probe_pre false read H.tree H.edges k x z hz)⟩
    have := probe_length false read H.tree H.edges k x
    rcases h : (probe false read H.tree H.edges k x).undecidedStrings with _ | ⟨a, _ | ⟨b, l⟩⟩
    · simp
    · simp
    · rw [h] at this; simp at this

theorem stuckString_length (x : FreeMonoid α) : (stuckString read H k x).length ≤ 1 := by
  unfold stuckString probeOrAgree
  split_ifs
  · simp [Outcome.undecidedStrings]
  · exact probe_length false read H.tree H.edges k x

theorem stuckString_undecided (hE : EdgesOK read H.tree H.edges) (x : FreeMonoid α) :
    ∀ z ∈ stuckString read H k x, read z = .undecided := by
  have hOK := probeOrAgree_ok false read H.tree H.edges k hE x
  unfold stuckString
  intro z hz
  revert hOK hz
  generalize probeOrAgree false read H.tree H.edges k x = o
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

/-- With ideal reads, all the sampler finds is at bad states, so the harvest ends well once the
sampler yields at least `y > 0`. -/
theorem endsWell_harvest {σ : Type*} {M : DFA α σ} {side : σ → Bool}
    (hI : IdealReads read M side) [IsProbabilityMeasure D] (hE : EdgesOK read H.tree H.edges)
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

end Ideal

end OrthoDFA
