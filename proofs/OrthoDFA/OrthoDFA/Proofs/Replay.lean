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

theorem route_queries (cut : FreeMonoid α → Option Bool) :
    ∀ (t : DTree α) {x q : FreeMonoid α}, q ∈ (t.route cut x).1 → ∃ m, q = x * m
  | .leaf, x, q, h => by simp [route] at h
  | .node m r a, x, q, h => by
    simp only [route] at h
    split at h
    · simp only [List.mem_singleton] at h
      exact ⟨m, h⟩
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, h⟩
      · exact route_queries cut a h
    · simp only [List.mem_cons] at h
      rcases h with h | h
      · exact ⟨m, h⟩
      · exact route_queries cut r h

theorem sift_node_none {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    (h : cut (x * m) = none) : (node m r a).sift cut x = .inr (x * m) := by
  simp [sift, route, h]

theorem sift_node_some {cut : FreeMonoid α → Option Bool} {m x : FreeMonoid α} {r a : DTree α}
    {b : Bool} (h : cut (x * m) = some b) :
    (node m r a).sift cut x = ((if b then a else r).sift cut x).map (b :: ·) id := by
  cases b <;> simp [sift, route, h]

/-- Where a cut places `x`, any total reading that agrees with it on what it decides does too. -/
theorem sift_of_sift {cut : FreeMonoid α → Option Bool} {g : FreeMonoid α → Bool}
    (hg : ∀ y b, cut y = some b → g y = b) :
    ∀ (t : DTree α) {x : FreeMonoid α} {p : List Bool}, t.sift cut x = .inl p →
      t.sift (fun y => some (g y)) x = .inl p
  | .leaf, x, p, h => by simpa [sift, route] using h
  | .node m r a, x, p, h => by
    rcases hc : cut (x * m) with _ | b
    · rw [sift_node_none hc] at h
      simp at h
    · rw [sift_node_some hc] at h
      rw [sift_node_some (b := b) (by rw [hg _ _ hc])]
      cases b
      · simp only [Bool.false_eq_true, ↓reduceIte] at h ⊢
        rcases hr : r.sift cut x with q | q <;> rw [hr] at h <;> simp at h
        subst h
        rw [sift_of_sift hg r hr]
        rfl
      · simp only [↓reduceIte] at h ⊢
        rcases ha : a.sift cut x with q | q <;> rw [ha] at h <;> simp at h
        subst h
        rw [sift_of_sift hg a ha]
        rfl

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

theorem anchorSearch_mem (t : DTree α) (w : FreeMonoid α) :
    ∀ is : List ℕ, (∀ i ∈ (anchorSearch R t w is).1, i ∈ is)
      ∧ ∀ s p, (anchorSearch R t w is).2 = some (s, p) → s ∈ is
  | [] => by simp [anchorSearch]
  | i :: is => by
    have ih := anchorSearch_mem t w is
    simp only [anchorSearch]
    split
    · simp
    · refine ⟨fun j hj => ?_, fun s p h => List.mem_cons_of_mem _ (ih.2 s p h)⟩
      simp only [List.mem_cons] at hj ⊢
      rcases hj with rfl | hj
      · exact Or.inl rfl
      · exact Or.inr (ih.1 j hj)

theorem anchorSearch_some (t : DTree α) (w : FreeMonoid α) :
    ∀ (is : List ℕ) {s : ℕ} {p : List Bool}, (anchorSearch R t w is).2 = some (s, p) →
      t.sift R.cut (prefixOf w s) = .inl p
  | [], s, p, h => by simp [anchorSearch] at h
  | i :: is, s, p, h => by
    simp only [anchorSearch] at h
    split at h
    · rename_i q hq
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact hq
    · exact anchorSearch_some t w is h

/-- Between an index where the sift agrees with the walk and a later one, the bisection either
meets a prefix strictly between them it cannot place, or settles on an index past the first. -/
theorem bisect_result (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ∀ (fuel lo hi : ℕ), lo < hi →
      (∀ b, (bisect R t w walk fuel lo hi).2 = .inl b →
          ∃ j, lo < j ∧ j < hi ∧ ∃ m, b = prefixOf w j * m)
        ∧ ∀ fd, (bisect R t w walk fuel lo hi).2 = .inr fd → lo < fd ∧ fd ≤ hi
  | 0, lo, hi, hlh => by
    refine ⟨fun b h => by simp [bisect] at h, fun fd h => ?_⟩
    simp only [bisect, Sum.inr.injEq] at h
    omega
  | fuel + 1, lo, hi, hlh => by
    simp only [bisect]
    split_ifs with h1
    · split
      · rename_i b hb
        refine ⟨fun b' h => ?_, fun fd h => by simp at h⟩
        simp only [Sum.inl.injEq] at h
        subst h
        obtain ⟨m, hm⟩ := DTree.sift_inr t hb
        exact ⟨_, by omega, by omega, m, hm⟩
      · split_ifs
        · have ih := bisect_result t w walk fuel ((lo + hi) / 2) hi (by omega)
          refine ⟨fun b h => ?_, fun fd h => ?_⟩
          · obtain ⟨j, hj1, hj2, hj3⟩ := ih.1 b h
            exact ⟨j, by omega, hj2, hj3⟩
          · have := ih.2 fd h
            omega
        · have ih := bisect_result t w walk fuel lo ((lo + hi) / 2) (by omega)
          refine ⟨fun b h => ?_, fun fd h => ?_⟩
          · obtain ⟨j, hj1, hj2, hj3⟩ := ih.1 b h
            exact ⟨j, hj1, by omega, hj3⟩
          · have := ih.2 fd h
            omega
    · refine ⟨fun b h => by simp at h, fun fd h => ?_⟩
      simp only [Sum.inr.injEq] at h
      omega

theorem bisect_bounds (t : DTree α) (w : FreeMonoid α) (walk : ℕ → List Bool) :
    ∀ (fuel lo hi j : ℕ), j ∈ (bisect R t w walk fuel lo hi).1 → lo < j ∧ j < hi
  | 0, lo, hi, j, h => by simp [bisect] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bisect] at h
    split_ifs at h with hlh
    · split at h
      · simp only [List.mem_singleton] at h
        omega
      · simp only [List.mem_cons] at h
        rcases h with rfl | h
        · omega
        · split_ifs at h
          · have := bisect_bounds t w walk fuel _ hi j h
            omega
          · have := bisect_bounds t w walk fuel lo _ j h
            omega
    · simp at h

instance (L : ℕ) : IsFiniteMeasure (anchorLaw L) := by
  constructor
  rcases Nat.eq_zero_or_pos L with rfl | hL
  · simp [anchorLaw]
  · simp only [anchorLaw, Measure.smul_apply, Measure.coe_finsetSum, Finset.sum_apply,
      measure_univ, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one, smul_eq_mul]
    rw [ENNReal.inv_mul_cancel (by exact_mod_cast hL.ne') (by simp)]
    exact ENNReal.one_lt_top

theorem scanl_getD_length {β γ : Type*} (f : β → γ → β) :
    ∀ (l : List γ) (a d : β), (l.scanl f a).getD l.length d = l.foldl f a
  | [], a, d => by simp
  | c :: l, a, d => by
    simp only [List.scanl_cons, List.length_cons, List.foldl_cons]
    rw [List.getD_cons_succ, scanl_getD_length f l]

theorem prefixOf_zero (w : FreeMonoid α) : prefixOf w 0 = 1 := by
  simp [prefixOf]

instance : Countable (FreeMonoid α) := inferInstanceAs (Countable (List α))

instance : MeasurableSingletonClass (FreeMonoid α) := ⟨fun _ => trivial⟩

end OrthoDFA
