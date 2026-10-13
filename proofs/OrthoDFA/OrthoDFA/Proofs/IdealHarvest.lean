import OrthoDFA.Level2Claim
import OrthoDFA.Proofs.IdealProbe
import OrthoDFA.Proofs.RoundEnd

/-!
# Level 2's sampler

Every undecided string a probe walked from `k` turns up is a prefix of the probe at least `k`
long, extended by a midfix, so the sampler is spread from `k`, whatever the hypothesis. It finds at
most two strings a probe, each read undecided, so with ideal reads all it finds is at bad states,
and it finds a string at least as often as a probe is charged one at its site.
-/

namespace OrthoDFA

namespace Ideal

open MeasureTheory
open scoped ENNReal

variable {α : Type*} (read : FreeMonoid α → ARU)

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
      (tripartiteSearch true (siftAgrees read T x st) lo hi).1).undecidedStrings,
      PreFrom x lo z := by
  have hs := tripartiteSearch_ok (recover := true) (agrees := siftAgrees read T x st) hle
  generalize (tripartiteSearch true (siftAgrees read T x st) lo hi).1 = r at hs
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
  | pair p =>
    rcases List.mem_append.1 hz with hz | hz
    · exact hat p hs.1 hz
    · exact hat (p + 1) (by have := hs.1; omega) hz
  | undecidedLo => simp [searchOutcome, Outcome.undecidedStrings] at hz
  | disagreeLo => simp [searchOutcome, Outcome.undecidedStrings] at hz

theorem probe_pre (k : ℕ) (x : FreeMonoid α) :
    ∀ z ∈ (probe read T E k x).undecidedStrings, PreFrom x k z := by
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
      · exact searchOutcome_pre read T x _ _ k _ (by omega) z hz
    · exact searchOutcome_pre read T x _ _ k _ (by omega) z hz
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
    (searchOutcome read T x st un hi r).undecidedStrings.length ≤ 2 := by
  have h := undecidedAt_length read T x
  cases r with
  | undecidedHi => exact (h hi).trans (by norm_num)
  | allAgree =>
    simp only [searchOutcome]
    split <;> simp [Outcome.undecidedStrings]
  | edge p =>
    simp only [searchOutcome]
    split <;> simp [Outcome.undecidedStrings]
  | triple p => exact (h _).trans (by norm_num)
  | undecidedMid p => exact (h _).trans (by norm_num)
  | pair p =>
    simp only [searchOutcome, Outcome.undecidedStrings, List.length_append]
    have := h p
    have := h (p + 1)
    omega
  | undecidedLo => simp [searchOutcome, Outcome.undecidedStrings]
  | disagreeLo => simp [searchOutcome, Outcome.undecidedStrings]

theorem probe_length (k : ℕ) (x : FreeMonoid α) :
    (probe read T E k x).undecidedStrings.length ≤ 2 := by
  rcases ha : T.sift read (prefixOf x k) with a | y
  · simp only [probe, ha]
    split
    · split
      · simp [Outcome.undecidedStrings]
      · exact searchOutcome_length read T x _ _ _ _
    · exact searchOutcome_length read T x _ _ _ _
  · simp [probe, ha, Outcome.undecidedStrings]

variable [DecidableEq α] [Countable α] (D : Measure (FreeMonoid α)) (H : Hypothesis α) (k : ℕ)
  (site : UndecidedSite α)

theorem siteHarvest_of (x : FreeMonoid α) :
    ∃ zs, siteHarvest read H k site x = zs.dedup ∧ (zs = [] ∨ (k ≤ x.toList.length ∧
      zs = (probe read H.tree H.edges k x).undecidedStrings)) := by
  unfold siteHarvest probeOrAgree
  split_ifs with hkx
  · exact ⟨[], rfl, Or.inl rfl⟩
  · rcases ho : probe read H.tree H.edges k x with _ | _ | _ | ⟨k', zs, cl⟩
    all_goals try exact ⟨[], rfl, Or.inl rfl⟩
    simp only
    split_ifs
    · exact ⟨zs, rfl, Or.inr ⟨by omega, by simp [Outcome.undecidedStrings]⟩⟩
    · exact ⟨[], rfl, Or.inl rfl⟩

theorem siteHarvest_fresh (x : FreeMonoid α) : (siteHarvest read H k site x).Nodup
    ∧ ∀ z ∈ siteHarvest read H k site x, z.toList.take k = x.toList.take k := by
  obtain ⟨zs, hzs, hor⟩ := siteHarvest_of read H k site x
  rw [hzs]
  refine ⟨List.nodup_dedup zs, fun z hz => ?_⟩
  rw [List.mem_dedup] at hz
  rcases hor with rfl | ⟨hk, rfl⟩
  · simp at hz
  · exact preFrom_take hk (probe_pre read H.tree H.edges k x z hz)

theorem siteHarvest_length (x : FreeMonoid α) : (siteHarvest read H k site x).length ≤ 2 := by
  obtain ⟨zs, hzs, hor⟩ := siteHarvest_of read H k site x
  rw [hzs]
  refine (List.dedup_sublist zs).length_le.trans ?_
  rcases hor with rfl | ⟨-, rfl⟩
  · simp
  · exact probe_length read H.tree H.edges k x

theorem siteHarvest_undecided (hE : EdgesOK read H.tree H.edges) (x : FreeMonoid α) :
    ∀ z ∈ siteHarvest read H k site x, read z = .undecided := by
  have hOK := probeOrAgree_ok read H.tree H.edges k hE x
  unfold siteHarvest
  intro z hz
  revert hOK hz
  generalize probeOrAgree read H.tree H.edges k x = o
  rcases o with _ | _ | _ | ⟨k', zs, cl⟩ <;> simp only [List.not_mem_nil, false_imp_iff,
    imp_true_iff]
  intro hOK hz
  split_ifs at hz
  · exact hOK.2.2 z (List.mem_dedup.1 hz)
  · simp at hz

theorem harvestLaw_apply (A : Set (FreeMonoid α)) [DecidablePred (· ∈ A)] :
    harvestLaw read D k H site A
      = ∫⁻ x, (((siteHarvest read H k site x).filter (· ∈ A)).length : ℝ≥0∞) ∂D :=
  bind_diracSum_apply D _ A

theorem harvestLaw_univ_le [IsProbabilityMeasure D] : harvestLaw read D k H site Set.univ ≤ 2 := by
  classical
  rw [harvestLaw_apply]
  calc _ ≤ ∫⁻ _, (2 : ℝ≥0∞) ∂D := lintegral_mono fun x => by
          have := (List.length_filter_le (· ∈ Set.univ) (siteHarvest read H k site x)).trans
            (siteHarvest_length read H k site x)
          calc _ ≤ ((2 : ℕ) : ℝ≥0∞) := Nat.cast_le.2 this
            _ = 2 := by norm_num
    _ = 2 := by simp

theorem harvestLaw_spread [IsFiniteMeasure D] : Spread D k (harvestLaw read D k H site) :=
  spread_of_fresh D k _ (fun x => (siteHarvest_fresh read H k site x).1)
    (fun x => (siteHarvest_fresh read H k site x).2)

/-- The sampler finds a string at least as often as a probe is charged one at the site. -/
theorem charged_le_harvestLaw :
    D {x | siteHarvest read H k site x ≠ []} ≤ harvestLaw read D k H site Set.univ := by
  classical
  rw [harvestLaw_apply, ← lintegral_indicator_one (Set.to_countable _).measurableSet]
  refine lintegral_mono fun x => ?_
  by_cases hx : siteHarvest read H k site x = []
  · simp [hx]
  · rw [Set.indicator_of_mem hx, Pi.one_apply, List.filter_eq_self.2 (by simp)]
    exact Nat.one_le_cast.2 (List.length_pos_iff.2 hx)

/-- With ideal reads, all the sampler finds is at bad states, so the harvest ends well once the
sampler yields more than `y ≥ 0`. -/
theorem endsWell_harvest {σ : Type*} {M : DFA α σ} {side : σ → Bool}
    (hI : IdealReads read M side) [IsProbabilityMeasure D] (hE : EdgesOK read H.tree H.edges)
    (ε : ℝ) (dis : Set (FreeMonoid α)) {y : ℝ} (hy : 0 ≤ y)
    (hlaw : y < (harvestLaw read D k H site).real Set.univ) :
    EndsWell M (AllUndecided read M) D ε dis k (harvestLaw read D k H) y (.harvest site) := by
  classical
  have hbad : harvestLaw read D k H site {z | AllUndecided read M (M.eval z.toList)}
      = harvestLaw read D k H site Set.univ := by
    refine bind_diracSum_of_forall D _ _ fun x z hz => ?_
    rcases hI (M.eval z.toList) with h | h
    · exact h
    · have := h z rfl
      rw [siteHarvest_undecided read H k site hE x z hz] at this
      split_ifs at this
  refine ⟨?_, harvestLaw_spread read D H k site, hlaw.le⟩
  rw [measureReal_def, measureReal_def, hbad]
  rw [measureReal_def] at hlaw
  linarith

end Ideal

end OrthoDFA
