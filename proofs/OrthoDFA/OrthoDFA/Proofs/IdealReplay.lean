import OrthoDFA.Proofs.IdealProbe
import OrthoDFA.Proofs.RoundEnd

/-!
# The level-2 replay is spread, and yields

Every undecided string a probe walked from `k` turns up is a prefix of the probe at least `k`
long, extended by a midfix. So the replay, walked from its anchor, keeps a prefix of its draw past
the anchor, whatever the hypothesis.

The root of every tree the round holds reads its string alone, so a replay finds the anchor's
prefix whenever it reads undecided: whenever the draw is at a bad state there.
-/

namespace OrthoDFA

namespace Ideal

variable {α : Type*} (read : FreeMonoid α → ARU)

theorem search_ge (ag : ℕ → Option Bool) : ∀ (n lo hi : ℕ), hi - lo = n → ∀ p,
    (bisect ag lo hi = .triple p ∨ bisect ag lo hi = .pair p) → lo ≤ p := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn p hp
  rw [bisect] at hp
  split at hp
  · simp at hp
  · rename_i hgap
    split at hp
    · exact (ih _ (by omega) _ _ rfl p hp).trans' (by omega)
    · exact ih _ (by omega) _ _ rfl p hp
    · split at hp
      · simp only [reduceCtorEq, SearchResult.pair.injEq, false_or] at hp; omega
      · simp only [reduceCtorEq, SearchResult.pair.injEq, false_or] at hp; omega
      · simp only [SearchResult.triple.injEq, reduceCtorEq, or_false] at hp; omega
      · exact ih _ (by omega) _ _ rfl p hp
      · exact (ih _ (by omega) _ _ rfl p hp).trans' (by omega)

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

theorem located_pre (x : FreeMonoid α) (st : ℕ → List Bool) (lo hi : ℕ) :
    ∀ z ∈ (locate read T x st lo hi).undecidedStrings, PreFrom x lo z := by
  have hs := search_ge (siftAgrees read T x st) _ lo hi rfl
  unfold locate
  revert hs
  generalize bisect (siftAgrees read T x st) lo hi = f
  intro hs z hz
  cases f with
  | edge p =>
    simp only at hz
    split at hz <;> simp [Outcome.undecidedStrings] at hz
  | triple p =>
    obtain ⟨m, hm⟩ := stuckAt_pre read T x p z hz
    exact ⟨p, hs p (.inl rfl), m, hm⟩
  | pair p =>
    have hp := hs p (.inr rfl)
    rcases List.mem_append.1 hz with hz | hz
    · obtain ⟨m, hm⟩ := stuckAt_pre read T x p z hz
      exact ⟨p, hp, m, hm⟩
    · obtain ⟨m, hm⟩ := stuckAt_pre read T x (p + 1) z hz
      exact ⟨p + 1, by omega, m, hm⟩

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
      · rename_i y hy _
        simp only [Outcome.undecidedStrings, List.mem_singleton] at hz
        subst hz
        obtain ⟨m, hm⟩ := DTree.sift_inr_mul read T _ _ hy
        exact ⟨_, by omega, m, hm⟩
      · split at hz
        · simp [Outcome.undecidedStrings] at hz
        · exact (located_pre read T x _ _ _ z hz).imp fun p h => ⟨by omega, h.2⟩
    · split at hz
      · rename_i y hy
        simp only [Outcome.undecidedStrings, List.mem_singleton] at hz
        subst hz
        obtain ⟨m, hm⟩ := DTree.sift_inr_mul read T _ _ hy
        exact ⟨max k x.toList.length, le_max_left _ _, m, by
          rw [pre_of_le (le_max_right _ _)]; exact hm⟩
      · split at hz
        · simp [Outcome.undecidedStrings] at hz
        · exact (located_pre read T x _ _ _ z hz).imp fun p h => ⟨by omega, h.2⟩
  · simp only [Outcome.undecidedStrings, List.mem_singleton] at hz
    subst hz
    obtain ⟨m, hm⟩ := DTree.sift_inr_mul read T _ _ ha
    exact ⟨k, le_rfl, m, hm⟩

theorem replay_prefix (w : FreeMonoid α) (e : ℕ) (x : FreeMonoid α)
    (hx : x ∈ replay read ⟨T, E⟩ w e) :
    ∃ i, e ≤ i ∧ i ≤ x.toList.length ∧ w.toList.take i = x.toList.take i := by
  unfold replay at hx
  split_ifs at hx with hw
  · simp at hx
  obtain ⟨p, hep, m, rfl⟩ := probe_pre read T E e w x hx
  have hi : prefixOf w p = prefixOf w (min p w.toList.length) := by
    rcases le_total p w.toList.length with h | h
    · rw [min_eq_left h]
    · rw [min_eq_right h, pre_of_le h, pre_of_le le_rfl]
  have hlen : (prefixOf w (min p w.toList.length)).toList.length = min p w.toList.length := by
    simp [prefixOf, List.length_take]
  refine ⟨min p w.toList.length, by omega, ?_, ?_⟩
  · rw [hi, FreeMonoid.toList_mul, List.length_append, hlen]; omega
  · rw [hi, FreeMonoid.toList_mul, List.take_left' hlen]
    simp only [prefixOf, FreeMonoid.toList_ofList]

/-- The tree's root reads its string alone. -/
def RootOne (T : DTree α) : Prop := ∃ r a, T = .node 1 r a

theorem splitAt_root {T : DTree α} (hT : RootOne T) (d : FreeMonoid α) (p : List Bool) :
    RootOne (T.splitAt d p) := by
  obtain ⟨r, a, rfl⟩ := hT
  rcases p with _ | ⟨_ | _, p⟩
  · exact ⟨r, a, rfl⟩
  · exact ⟨_, a, rfl⟩
  · exact ⟨r, _, rfl⟩

section Steps

variable [DecidableEq α] (C : Level2Settings)

omit [DecidableEq α] in
theorem tick_tree {s s' : Level2State α} (h : extendCleanRun C s = .inl s') : s'.tree = s.tree := by
  unfold extendCleanRun at h
  split_ifs at h
  cases h; rfl

theorem charge_tree {s s' : Level2State α} {k : UndecidedSite α} {zs : List (FreeMonoid α)}
    {cl : Bool} (h : addUndecided C s k zs cl = .inl s') : s'.tree = s.tree := by
  unfold addUndecided at h
  simp only at h
  split_ifs at h
  · have := tick_tree C h; exact this
  · cases h; rfl

theorem record_root {s s' : Level2State α} {p : List Bool} {c : α} {u : FreeMonoid α}
    {t : List Bool} (h : recordDisagreement read C s p c u t = .inl s') (hs : RootOne s.tree) :
    RootOne s'.tree := by
  unfold recordDisagreement splitLeaf at h
  simp only at h
  repeat' (first | split_ifs at h | split at h)
  all_goals first
    | (injection h with h; rw [← h]; exact splitAt_root hs _ _)
    | (injection h with h; rw [← h]; exact hs)

theorem step_root {s s' : Level2State α} {x : FreeMonoid α} (h : applyProbe read C s x = .inl s')
    (hs : RootOne s.tree) : RootOne s'.tree := by
  unfold applyProbe at h
  split at h
  · exact tick_tree C h ▸ hs
  · exact charge_tree C h ▸ hs
  · exact record_root read C h hs
  · injection h with h; rw [← h]; exact hs

end Steps

variable [DecidableEq α] (C : Level2Settings) in
theorem round_root : ∀ (xs : List (FreeMonoid α)) (s : Level2State α), RootOne s.tree →
    RootOne (runProbes read C s xs).1.tree
  | [], s, hs => hs
  | x :: xs, s, hs => by
    simp only [runProbes]
    split
    · rename_i s' h; exact round_root xs s' (step_root read C h hs)
    · exact hs

theorem replay_root {T : DTree α} (hT : RootOne T) (E : Edges α) {w : FreeMonoid α} {e : ℕ}
    (he : e ≤ w.toList.length) (hu : read (prefixOf w e) = .undecided) :
    replay read ⟨T, E⟩ w e ≠ [] := by
  obtain ⟨r, a, rfl⟩ := hT
  have hs : (DTree.node 1 r a).sift read (prefixOf w e) = .inr (prefixOf w e) := by
    simp [DTree.sift, hu]
  unfold replay probe
  dsimp only
  rw [if_neg (by omega), hs]
  simp [Outcome.undecidedStrings]

theorem replay_yields [Countable α] {T : DTree α} (hT : RootOne T) (E : Edges α)
    (D : MeasureTheory.Measure (FreeMonoid α)) [MeasureTheory.IsFiniteMeasure D] (s L : ℕ)
    {σ : Type*} (M : DFA α σ) :
    Yields D s L (replay read ⟨T, E⟩)
      (coverage D s L M (AllUndecided read M) / (L + 1 - s : ℕ)) := by
  refine (coverage_le_anchor D s L M _).trans (MeasureTheory.measureReal_mono ?_
    (MeasureTheory.measure_ne_top _ _))
  rintro ⟨w, e⟩ ⟨he, hb⟩
  exact replay_root read hT E he (hb _ (by simp [prefixOf]))

theorem replay_spread [Countable α] (D : MeasureTheory.Measure (FreeMonoid α))
    [MeasureTheory.IsFiniteMeasure D]
    (s L : ℕ) : Spread D s L (replay read ⟨T, E⟩) :=
  spread_of_prefix D s L _ (replay_prefix read T E)

end Ideal

end OrthoDFA
