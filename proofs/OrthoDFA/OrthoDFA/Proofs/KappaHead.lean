import OrthoDFA.Proofs.KappaRound
import OrthoDFA.Proofs.RandomShort

/-!
# The strings before `kd` at the trees a round reaches

A run reaches a tree only through what its steps read, and each step's tree holds the midfixes of
the one before. So, given the probes, the race (`RaceReadsBound`) can read, before each step, the
strings of `headOf k kd` its tree adds (`headRound`), then what the step reads: at most
`H (|Q| + 1)` strings before `kd` in all, `H` the strings of length `k` to `kd − 1`, besides the
round's `Mb`. Each at a good read-state reads undecided with chance at most `1.5θ`, so some string
of `headOf k kd` at a tree the round reaches, before it reads anything wrong, reads so with chance
at most `1.5θ (Mb + H (|Q| + 1))` (`head_le`).
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open MeasureTheory Eps

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (U : σ → ℝ) (θ : ℝ) (C : Level3Settings)

/-- Some string of `headOf k kd T` at a good read-state reads undecided. -/
def HeadUnd (k kd : ℕ) (rd : FreeMonoid α → ARU) (T : DTree α) : Prop :=
  ∃ z ∈ headOf k kd T, ¬ BadAt U θ (M.eval z.toList) ∧ rd z = .undecided

omit [Fintype α] [DecidableEq α] in
theorem mids_subset_splitAt (d : FreeMonoid α) :
    ∀ (T : DTree α) (p : List Bool), mids T ⊆ mids (T.splitAt d p)
  | .leaf, [] => by classical simp [mids]
  | .leaf, _ :: _ => subset_rfl
  | .node m r a, [] => subset_rfl
  | .node m r a, false :: p => by
    classical
    intro z hz
    simp only [DTree.splitAt, mids, Finset.mem_insert, Finset.mem_union] at hz ⊢
    rcases hz with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl (mids_subset_splitAt d r p h))
    · exact Or.inr (Or.inr h)
  | .node m r a, true :: p => by
    classical
    intro z hz
    simp only [DTree.splitAt, mids, Finset.mem_insert, Finset.mem_union] at hz ⊢
    rcases hz with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl h)
    · exact Or.inr (Or.inr (mids_subset_splitAt d a p h))

/-- A step keeps the tree's midfixes. -/
theorem mids_step {read : FreeMonoid α → ARU} {s s' : Level3State α} {x : FreeMonoid α}
    (hs : Inv read s) (h : applyProbe read C s x = .inl s') : mids s.tree ⊆ mids s'.tree := by
  rcases step_cases read C hs h with ⟨-, -, -, -, -, -, -, -, -, rfl⟩ |
    ⟨-, c, -, t, t₀, -, -, -, -, -, -, -, -, rfl⟩ | ⟨rfl, -, -⟩
  · exact subset_rfl
  · exact mids_subset_splitAt _ _ _
  · exact subset_rfl

omit [DecidableEq α] in
theorem headOf_mono {k kd : ℕ} {T T' : DTree α} (h : mids T ⊆ mids T') :
    headOf k kd T ⊆ headOf k kd T' := by
  intro z hz
  simp only [headOf, Finset.mem_biUnion, Finset.mem_image] at hz ⊢
  obtain ⟨i, hi, u, hu, m, hm, rfl⟩ := hz
  exact ⟨i, hi, u, hu, m, h hm, rfl⟩

/-! ## A step reading a string before `kd` undecided, before any wrong read -/

section Steps

variable {S X E : Type*} (step : S → X → S ⊕ E)

/-- The states a run reaches, step by step. -/
def reach : S → List X → ℕ → Option S
  | s, _, 0 => some s
  | _, [], _ + 1 => none
  | s, x :: xs, j + 1 =>
    match step s x with
    | .inl s' => reach s' xs j
    | .inr _ => none

theorem anyB_reach (B : S → X → Prop) :
    ∀ (xs : List X) (s₀ : S), AnyB step B s₀ xs →
      ∃ j < xs.length, ∃ s x, reach step s₀ xs j = some s ∧ xs[j]? = some x ∧ B s x
  | [], _, h => by simp [AnyB] at h
  | x :: xs, s₀, h => by
    rcases h with h | ⟨s', hs', h⟩
    · exact ⟨0, by simp, s₀, x, rfl, rfl, h⟩
    · obtain ⟨j, hj, s, y, h1, h2, h3⟩ := anyB_reach B xs s' h
      refine ⟨j + 1, by simp; omega, s, y, ?_, by simpa using h2, h3⟩
      simp only [reach, hs']
      exact h1

theorem reach_anyB (B : S → X → Prop) :
    ∀ (xs : List X) (s₀ : S) (j : ℕ) (s : S) (x : X), reach step s₀ xs j = some s →
      xs[j]? = some x → B s x → AnyB step B s₀ xs
  | [], _, 0, _, _, _, hx, _ => by simp at hx
  | [], _, _ + 1, _, _, h, _, _ => by simp [reach] at h
  | y :: xs, s₀, 0, s, x, h, hx, hB => by
    simp only [reach, Option.some.injEq] at h
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
    subst h hx
    exact Or.inl hB
  | y :: xs, s₀, j + 1, s, x, h, hx, hB => by
    simp only [reach] at h
    rcases hst : step s₀ y with s' | e <;> rw [hst] at h
    · exact Or.inr ⟨s', hst, reach_anyB B xs s' j s x h (by simpa using hx) hB⟩
    · cases h

end Steps

omit [Fintype α] in
theorem stateAt_eq_reach (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α) (j : ℕ),
      stateAt C read s xs j = reach (applyProbe read C) s xs j
  | _, _, 0 => by simp [stateAt, reach]
  | [], _, _ + 1 => by simp [stateAt, reach]
  | x :: xs, s, j + 1 => by
    simp only [stateAt, reach]
    rcases applyProbe read C s x with s' | e
    · exact stateAt_eq_reach read xs s' j
    · rfl

omit [Fintype σ] in
/-- A run whose steps read nothing wrong reads nothing wrong in its rounds before step `j`. -/
theorem good'_of_not_anyB (read : FreeMonoid α → ARU) (xs : List (FreeMonoid α))
    (h : ¬ AnyB (applyProbe read C) (ReadsWrong M side read C) initialState xs) (n : ℕ) :
    Good' M side C xs read n := by
  intro i _ z hz hw
  unfold rounds' at hz
  rcases hs : stateAt C read initialState xs (i / 2) with _ | s <;>
    rcases hx : xs[i / 2]? with _ | x <;> simp only [hs, hx] at hz
  · exact absurd hz (Finset.notMem_empty z)
  · exact absurd hz (Finset.notMem_empty z)
  · exact absurd hz (Finset.notMem_empty z)
  refine h (reach_anyB _ _ xs initialState (i / 2) s x
    (by rw [← stateAt_eq_reach]; exact hs) hx ⟨z, ?_, hw⟩)
  split_ifs at hz
  · exact Finset.mem_union_left _ hz
  · exact Finset.mem_union_right _ hz

/-! ## The race -/

section Race

variable (k kd : ℕ) (xs : List (FreeMonoid α))

/-- The tree after `j` steps, a leaf where the run has ended. -/
noncomputable def treeAt (r : FreeMonoid α → ARU) (j : ℕ) : DTree α :=
  ((stateAt C r initialState xs j).map Level3State.tree).getD .leaf

open scoped Classical in
/-- Round `i`, read only while no earlier round reads a string wrong. -/
noncomputable def gated (r : FreeMonoid α → ARU) (i : ℕ) : Finset (FreeMonoid α) :=
  if Good' M side C xs r i then rounds' C xs i r else ∅

open scoped Classical in
/-- The strings of `headOf k kd` at step `j`'s tree and not at step `j − 1`'s, read only while no
earlier round reads a string wrong. -/
noncomputable def headRound (r : FreeMonoid α → ARU) : ℕ → Finset (FreeMonoid α)
  | 0 => headOf k kd (treeAt C xs r 0)
  | j + 1 => if Good' M side C xs r (2 * (j + 1)) then
      headOf k kd (treeAt C xs r (j + 1)) \ headOf k kd (treeAt C xs r j) else ∅

/-- At step `j`, rounds `3j`, `3j + 1` and `3j + 2`: the strings before `kd` the tree adds, then
the step's two rounds. -/
noncomputable def raceRound (r : FreeMonoid α → ARU) (i : ℕ) : Finset (FreeMonoid α) :=
  if i % 3 = 0 then headRound M side C k kd xs r (i / 3)
  else if i % 3 = 1 then gated M side C xs r (2 * (i / 3))
  else gated M side C xs r (2 * (i / 3) + 1)

variable {M side C k kd xs}

omit [Fintype σ] in
theorem good'_succ_iff {r : FreeMonoid α → ARU} {n : ℕ} :
    Good' M side C xs r (n + 1) ↔ Good' M side C xs r n ∧ ∀ z ∈ rounds' C xs n r,
      r z ≠ if side (M.eval z.toList) then .reject else .accept :=
  ⟨fun h => ⟨good'_mono M side C xs r h (by omega), h n (by omega)⟩,
    fun ⟨h1, h2⟩ i hi => if hin : i < n then h1 i hin else (show i = n by omega) ▸ h2⟩

theorem good'_congr {r r' : FreeMonoid α → ARU} :
    ∀ n, (∀ l < n, ∀ z ∈ gated M side C xs r l, r z = r' z) →
      (Good' M side C xs r n ↔ Good' M side C xs r' n)
  | 0, _ => ⟨fun _ i hi => absurd hi (by omega), fun _ i hi => absurd hi (by omega)⟩
  | n + 1, h => by
    have ih := good'_congr n fun l hl => h l (by omega)
    rw [good'_succ_iff, good'_succ_iff]
    by_cases hg : Good' M side C xs r n
    · have hag : ∀ l ≤ n, ∀ z ∈ rounds' C xs l r, r z = r' z := fun l hl z hz =>
        h l (by omega) z (by
          simp only [gated, if_pos (good'_mono M side C xs r hg hl)]; exact hz)
      have hr := rounds'_local M side C xs n r r' hg hag
      rw [← hr]
      constructor
      · rintro ⟨-, h2⟩
        refine ⟨ih.1 hg, fun z hz => ?_⟩
        rw [← hag n le_rfl z hz]
        exact h2 z hz
      · rintro ⟨-, h2⟩
        refine ⟨hg, fun z hz => ?_⟩
        rw [hag n le_rfl z hz]
        exact h2 z hz
    · have hg' : ¬ Good' M side C xs r' n := fun h' => hg (ih.2 h')
      exact ⟨fun h' => absurd h'.1 hg, fun h' => absurd h'.1 hg'⟩

theorem gated_congr {r r' : FreeMonoid α → ARU} (n : ℕ)
    (h : ∀ l ≤ n, ∀ z ∈ gated M side C xs r l, r z = r' z) :
    gated M side C xs r n = gated M side C xs r' n := by
  classical
  have hiff := good'_congr n fun l hl => h l (by omega)
  by_cases hg : Good' M side C xs r n
  · have hag : ∀ l ≤ n, ∀ z ∈ rounds' C xs l r, r z = r' z := fun l hl z hz =>
      h l hl z (by simp only [gated, if_pos (good'_mono M side C xs r hg hl)]; exact hz)
    simp only [gated, if_pos hg, if_pos (hiff.1 hg)]
    exact rounds'_local M side C xs n r r' hg hag
  · simp only [gated, if_neg hg, if_neg (fun h' => hg (hiff.2 h'))]

theorem treeAt_congr {r r' : FreeMonoid α → ARU} (j : ℕ) (hg : Good' M side C xs r (2 * j))
    (h : ∀ l < 2 * j, ∀ z ∈ gated M side C xs r l, r z = r' z) :
    treeAt C xs r j = treeAt C xs r' j := by
  unfold treeAt
  rw [stateAt_congr' M side C xs j hg fun i hi z hz => h i hi z (by
    simp only [gated, if_pos (good'_mono M side C xs r hg hi.le)]; exact hz)]

theorem headRound_congr {r r' : FreeMonoid α → ARU} (j : ℕ)
    (h : ∀ l < 2 * j, ∀ z ∈ gated M side C xs r l, r z = r' z) :
    headRound M side C k kd xs r j = headRound M side C k kd xs r' j := by
  classical
  cases j with
  | zero => simp [headRound, treeAt, stateAt]
  | succ j =>
    have hiff := good'_congr (2 * (j + 1)) h
    by_cases hg : Good' M side C xs r (2 * (j + 1))
    · simp only [headRound, if_pos hg, if_pos (hiff.1 hg)]
      rw [treeAt_congr (j + 1) hg h, treeAt_congr j (good'_mono M side C xs r hg (by omega))
        fun l hl => h l (by omega)]
    · simp only [headRound, if_neg hg, if_neg (fun h' => hg (hiff.2 h'))]

omit [Fintype σ] in
theorem raceRound_gated (r : FreeMonoid α → ARU) (l : ℕ) :
    raceRound M side C k kd xs r (3 * (l / 2) + 1 + l % 2) = gated M side C xs r l := by
  unfold raceRound
  rcases Nat.mod_two_eq_zero_or_one l with h | h
  · rw [if_neg (by omega), if_pos (by omega)]
    congr 1; omega
  · rw [if_neg (by omega), if_neg (by omega)]
    congr 1; omega

theorem raceRound_local {r r' : FreeMonoid α → ARU} (i : ℕ)
    (h : ∀ i' ≤ i, ∀ z ∈ raceRound M side C k kd xs r i', r z = r' z) :
    raceRound M side C k kd xs r i = raceRound M side C k kd xs r' i := by
  have hg : ∀ l, 3 * (l / 2) + 1 + l % 2 ≤ i → ∀ z ∈ gated M side C xs r l, r z = r' z :=
    fun l hl z hz => h _ hl z (by rw [raceRound_gated]; exact hz)
  unfold raceRound
  split_ifs with h0 h1
  · exact headRound_congr _ fun l hl => hg l (by omega)
  · exact gated_congr _ fun l hl => hg l (by omega)
  · exact gated_congr _ fun l hl => hg l (by omega)

theorem sum_range_three (f : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range (3 * n), f i
      = ∑ j ∈ Finset.range n, (f (3 * j) + f (3 * j + 1) + f (3 * j + 2)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show 3 * (n + 1) = 3 * n + 1 + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_succ, ih, Finset.sum_range_succ]
    ring

theorem sum_range_two (f : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range (2 * n), f i = ∑ j ∈ Finset.range n, (f (2 * j) + f (2 * j + 1)) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ, ih,
      Finset.sum_range_succ]
    ring

theorem treeAt_mids {r : FreeMonoid α → ARU} {j : ℕ} (hg : Good' M side C xs r (2 * j)) :
    (mids (treeAt C xs r j)).card ≤ Fintype.card σ + 1 := by
  unfold treeAt
  rcases hs : stateAt C r initialState xs j with _ | s
  · simp [mids]
  · obtain ⟨hi, hw⟩ := inv_of_good' M side C xs r j s hs hg
    exact mids_le_of_class (cls_mem' M side r hi hw)

/-- The head rounds number at most `H (|Q| + 1)` strings. -/
theorem sum_headRound (r : FreeMonoid α → ARU) :
    ∀ n, (∑ j ∈ Finset.range (n + 1), ((headRound M side C k kd xs r j).card : ℝ)
        ≤ (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1))
      ∧ (Good' M side C xs r (2 * n) → (stateAt C r initialState xs n).isSome →
        ∑ j ∈ Finset.range (n + 1), ((headRound M side C k kd xs r j).card : ℝ)
          ≤ (headOf k kd (treeAt C xs r n)).card) := by
  classical
  have hH : 0 ≤ ∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i := by positivity
  have hcard : ∀ j, Good' M side C xs r (2 * j) → ((headOf k kd (treeAt C xs r j)).card : ℝ)
      ≤ (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1) := fun j hg =>
    (card_headOf k kd _).trans (mul_le_mul_of_nonneg_left (by exact_mod_cast treeAt_mids hg) hH)
  intro n
  induction n with
  | zero =>
    have hg0 : Good' M side C xs r (2 * 0) := fun i hi => absurd hi (by omega)
    simp only [zero_add, Finset.sum_range_one, headRound]
    exact ⟨hcard 0 hg0, fun _ _ => le_rfl⟩
  | succ n ih =>
    rw [Finset.sum_range_succ]
    by_cases hg : Good' M side C xs r (2 * (n + 1))
    · have hgn : Good' M side C xs r (2 * n) := good'_mono M side C xs r hg (by omega)
      rcases hs : stateAt C r initialState xs (n + 1) with _ | s'
      · have hA : headOf k kd (treeAt C xs r (n + 1)) = ∅ := by
          unfold treeAt; rw [hs]
          simp only [headOf, mids, Option.map_none, Option.getD_none, Finset.image_empty]
          ext z
          simp
        have h0 : headRound M side C k kd xs r (n + 1) = ∅ := by
          simp only [headRound, if_pos hg, hA, Finset.empty_sdiff]
        rw [h0, Finset.card_empty, Nat.cast_zero, add_zero]
        exact ⟨ih.1, fun _ h => by simp at h⟩
      · have hs' := hs
        rw [stateAt_succ] at hs'
        rcases hs₀ : stateAt C r initialState xs n with _ | s <;> rw [hs₀] at hs'
        · simp at hs'
        rcases hx : xs[n]? with _ | x <;>
          simp only [hx, Option.bind_some, Option.bind_none, reduceCtorEq] at hs'
        rcases hst : applyProbe r C s x with s'' | e <;> rw [hst] at hs'
        · simp only [Sum.elim_inl, Option.some.injEq] at hs'
          subst hs'
          have hI := (inv_of_good' M side C xs r n s hs₀ hgn).1
          have ht0 : treeAt C xs r n = s.tree := by unfold treeAt; rw [hs₀]; rfl
          have ht1 : treeAt C xs r (n + 1) = s''.tree := by unfold treeAt; rw [hs]; rfl
          have hsub : headOf k kd (treeAt C xs r n) ⊆ headOf k kd (treeAt C xs r (n + 1)) := by
            rw [ht0, ht1]; exact headOf_mono (mids_step C hI hst)
          have hsum := ih.2 hgn (by rw [hs₀]; rfl)
          have hnew : ((headRound M side C k kd xs r (n + 1)).card : ℝ)
              = (headOf k kd (treeAt C xs r (n + 1))).card
                - (headOf k kd (treeAt C xs r n)).card := by
            simp only [headRound, if_pos hg]
            rw [Finset.card_sdiff_of_subset hsub, Nat.cast_sub (Finset.card_le_card hsub)]
          have h2 : ∑ j ∈ Finset.range (n + 1), ((headRound M side C k kd xs r j).card : ℝ)
              + (headRound M side C k kd xs r (n + 1)).card
              ≤ (headOf k kd (treeAt C xs r (n + 1))).card := by
            rw [hnew]; linarith
          exact ⟨h2.trans (hcard _ hg), fun _ _ => h2⟩
        · simp at hs'
    · simp only [headRound, if_neg hg, Finset.card_empty, Nat.cast_zero, add_zero]
      exact ⟨ih.1, fun h => absurd h hg⟩

omit [Fintype σ] in
theorem raceRound_three (r : FreeMonoid α → ARU) (j : ℕ) :
    raceRound M side C k kd xs r (3 * j) = headRound M side C k kd xs r j
      ∧ raceRound M side C k kd xs r (3 * j + 1) = gated M side C xs r (2 * j)
      ∧ raceRound M side C k kd xs r (3 * j + 2) = gated M side C xs r (2 * j + 1) := by
  refine ⟨?_, ?_, ?_⟩ <;> unfold raceRound
  · rw [if_pos (by omega)]; congr 1; omega
  · rw [if_neg (by omega), if_pos (by omega)]; congr 2; omega
  · rw [if_neg (by omega), if_neg (by omega)]; congr 2; omega

end Race

variable {M side C}

omit [Fintype σ] in
/-- A step reading a string wrong, or at a tree with a string of `headOf k kd` at a good
read-state read undecided: some step reads a string wrong, or the latter holds at a tree reached
before any wrong read. -/
theorem anyB_head (read : FreeMonoid α → ARU) (k kd : ℕ) (xs : List (FreeMonoid α))
    (h : AnyB (applyProbe read C) (fun s x => ReadsWrong M side read C s x
      ∨ HeadUnd M U θ k kd read s.tree) initialState xs) :
    AnyB (applyProbe read C) (ReadsWrong M side read C) initialState xs
      ∨ ∃ j < xs.length, Good' M side C xs read (2 * j)
        ∧ HeadUnd M U θ k kd read (treeAt C xs read j) := by
  by_cases hw : AnyB (applyProbe read C) (ReadsWrong M side read C) initialState xs
  · exact Or.inl hw
  right
  obtain ⟨j, hj, s, x, h1, h2, h3⟩ := anyB_reach _ _ xs initialState h
  rcases h3 with h3 | h3
  · exact absurd (reach_anyB _ _ xs initialState j s x h1 h2 h3) hw
  refine ⟨j, hj, good'_of_not_anyB M side C read xs hw _, ?_⟩
  have : treeAt C xs read j = s.tree := by unfold treeAt; rw [stateAt_eq_reach, h1]; rfl
  rwa [this]

variable (M side C)

/-- Given probes no longer than `L`, some string of `headOf k kd` at a good read-state at a tree
the round reaches before reading anything wrong reads undecided with chance at most
`1.5θ (Mb + H (|Q| + 1))`. -/
theorem head_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (εw κ : ℝ) (R : KappaReadOracle μ M side U εw κ) (k kd : ℕ) (xs : List (FreeMonoid α))
    (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L) (hθ : 0 ≤ θ) (hy0 : 0 < C.harvestRate)
    (hyε : C.harvestRate ≤ C.disagreementRate / 48) (hε1 : C.disagreementRate < 1)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    μ {ω | ∃ j < xs.length, Good' M side C xs (R.read · ω) (2 * j)
        ∧ HeadUnd M U θ k kd (R.read · ω) (treeAt C xs (R.read · ω) j)}
      ≤ ENNReal.ofReal (3 / 2 * θ * (Mb (Fintype.card σ) (Fintype.card α) L xs.length
        + (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1))) := by
  classical
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z => {a | ¬ BadAt U θ (M.eval z.toList) ∧ a = .undecided}
  set Hs := (∑ i ∈ Finset.Ico k kd, (Fintype.card α : ℝ) ^ i) * (Fintype.card σ + 1)
  have hHs : 0 ≤ Hs := by positivity
  have hb : ∀ (r : FreeMonoid α → ARU) (P : ℕ → Prop) (dP : DecidablePred P),
      ∑ j ∈ @Finset.filter ℕ P dP (Finset.range (3 * xs.length)),
        ((raceRound M side C k kd xs r j).card : ℝ)
      ≤ Mb (Fintype.card σ) (Fintype.card α) L xs.length + Hs := by
    intro r P dP
    have hhead : ∑ j ∈ Finset.range xs.length, ((headRound M side C k kd xs r j).card : ℝ)
        ≤ Hs := by
      rcases hn : xs.length with _ | n
      · simp [hHs]
      · exact (sum_headRound r n).1
    have hgate : ∑ l ∈ Finset.range (2 * xs.length), ((gated M side C xs r l).card : ℝ)
        = readCount M side C xs r xs.length := by
      unfold readCount
      rw [Finset.sum_filter]
      refine Finset.sum_congr rfl fun l _ => ?_
      by_cases hg : Good' M side C xs r l
      · simp only [gated, if_pos hg]
      · simp only [gated, if_neg hg, Finset.card_empty, Nat.cast_zero]
    calc _ ≤ ∑ j ∈ Finset.range (3 * xs.length), ((raceRound M side C k kd xs r j).card : ℝ) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            fun _ _ _ => Nat.cast_nonneg _
      _ = ∑ j ∈ Finset.range xs.length, ((headRound M side C k kd xs r j).card : ℝ)
          + ∑ l ∈ Finset.range (2 * xs.length), ((gated M side C xs r l).card : ℝ) := by
          rw [sum_range_three, sum_range_two, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun j _ => ?_
          obtain ⟨h0, h1, h2⟩ := raceRound_three (M := M) (side := side) (C := C) (k := k)
            (kd := kd) (xs := xs) r j
          rw [h0, h1, h2]
          ring
      _ ≤ Hs + Mb (Fintype.card σ) (Fintype.card α) L xs.length := by
          rw [hgate]
          exact add_le_add hhead (count_le M side xs r L hL hy0 hyε hε1 hδ0 hδ1).2
      _ = _ := add_comm _ _
  have hrate : ∀ z, μ.real {ω | R.read z ω ∈ B z}
      ≤ 3 / 2 * θ + 0 * μ.real {ω | R.read z ω ∈ (∅ : Set ARU)} := by
    intro z
    rw [zero_mul, add_zero]
    by_cases hg : BadAt U θ (M.eval z.toList)
    · have hE : {ω | R.read z ω ∈ B z} = ∅ := by
        ext ω
        simp only [B, Set.mem_ofPred_eq, hg, not_true_eq_false, false_and,
          Set.mem_empty_iff_false]
      rw [hE, measureReal_empty]
      linarith only [hθ]
    · have hE : {ω | R.read z ω ∈ B z} = {ω | R.read z ω = .undecided} := by
        ext ω; simp only [B, Set.mem_ofPred_eq, hg, not_false_eq_true, true_and]
      rw [hE]
      exact undecided_le_of_good M U θ R.undecided hg
  have hθ' : (0 : ℝ) ≤ 3 / 2 * θ := by linarith only [hθ]
  have hm0 : ∀ (r : FreeMonoid α → ARU) (P : ℕ → Prop) (dP : DecidablePred P)
      (Q : FreeMonoid α → Prop) (dQ : DecidablePred Q), (∀ z, ¬ Q z) →
      ∑ j ∈ @Finset.filter ℕ P dP (Finset.range (3 * xs.length)),
        ((@Finset.filter _ Q dQ (raceRound M side C k kd xs r j)).card : ℝ) ≤ 0 := by
    intro r P dP Q dQ hQ
    refine le_of_eq (Finset.sum_eq_zero fun j _ => ?_)
    rw [Finset.filter_false_of_mem fun z _ => hQ z, Finset.card_empty, Nat.cast_zero]
  have hmain := raceReadsBound_holds μ R.read B (fun _ => ∅) (3 / 2 * θ) 0
    (fun i r => raceRound M side C k kd xs r i) (Mb (Fintype.card σ) (Fintype.card α) L
      xs.length + Hs) 0 (3 * xs.length) R.measurable R.indep hθ' le_rfl hrate
    (fun i r r' _ ha => raceRound_local i ha) (fun r => hb r _ _)
    (fun r => hm0 r _ _ _ _ fun z h => h)
  refine (measure_mono fun ω hω => ?_).trans (hmain.trans (le_of_eq ?_))
  · obtain ⟨j, hj, hg, z, hz, hgood, hu⟩ := hω
    have hex : ∃ i, i ≤ j ∧ z ∈ headOf k kd (treeAt C xs (R.read · ω) i) := ⟨j, le_rfl, hz⟩
    obtain ⟨i, ⟨hij, hzi⟩, hmin⟩ : ∃ i, (i ≤ j ∧ z ∈ headOf k kd (treeAt C xs (R.read · ω) i))
        ∧ ∀ i' < i, ¬ (i' ≤ j ∧ z ∈ headOf k kd (treeAt C xs (R.read · ω) i')) :=
      ⟨Nat.find hex, Nat.find_spec hex, fun i' h => Nat.find_min hex h⟩
    refine ⟨3 * i, by omega, z, ?_, show ¬ BadAt U θ (M.eval z.toList) ∧ _ from ⟨hgood, hu⟩⟩
    show z ∈ raceRound M side C k kd xs (R.read · ω) (3 * i)
    rw [(raceRound_three (M := M) (side := side) (C := C) (k := k) (kd := kd) (xs := xs)
      _ i).1]
    cases i with
    | zero => exact hzi
    | succ i =>
      have hgi : Good' M side C xs (R.read · ω) (2 * (i + 1)) :=
        good'_mono M side C xs _ hg (by omega)
      simp only [headRound, if_pos hgi]
      exact Finset.mem_sdiff.2 ⟨hzi, fun h => hmin i (by omega) ⟨by omega, h⟩⟩
  · congr 1
    ring

end Kappa

end Random

end OrthoDFA
