import OrthoDFA.Proofs.Query
import OrthoDFA.Proofs.StartAtK

/-!
# The triples' harvest

A triple's harvest is a read the cut leaves undecided. In a probe whose search ends at a triple,
every read outside the search's middles is decided, so the harvested string was first read, and
left undecided, at one of the middles (`triple_first_read`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Trace

variable (cut : FreeMonoid α → Option Bool)

omit [Fintype α] [DecidableEq α] in
theorem route_decided : ∀ (t : DTree α) (w : FreeMonoid α), (t.route cut w).2.isLeft →
    ∀ y ∈ (t.route cut w).1, cut y ≠ none
  | .leaf, _, _, y, hy => by simp [DTree.route] at hy
  | .node m r a, w, h, y, hy => by
    simp only [DTree.route] at h hy
    rcases hc : cut (w * m) with _ | _ | _ <;> rw [hc] at h hy
    · simp at h
    · simp only [Sum.isLeft_map, List.mem_cons] at h hy
      rcases hy with rfl | hy
      · rw [hc]; simp
      · exact route_decided r w h y hy
    · simp only [Sum.isLeft_map, List.mem_cons] at h hy
      rcases hy with rfl | hy
      · rw [hc]; simp
      · exact route_decided a w h y hy

omit [Fintype α] [DecidableEq α] in
theorem route_undecided : ∀ (t : DTree α) (w b : FreeMonoid α), (t.route cut w).2 = .inr b →
    b ∈ (t.route cut w).1 ∧ cut b = none
  | .leaf, _, _, h => by simp [DTree.route] at h
  | .node m r a, w, b, h => by
    simp only [DTree.route] at h ⊢
    rcases hc : cut (w * m) with _ | _ | _ <;> simp only [hc] at h ⊢
    · obtain rfl := Sum.inr.inj h
      exact ⟨by simp, hc⟩
    · rcases hr : (r.route cut w).2 with _ | b' <;> rw [hr] at h
      · simp at h
      · obtain rfl := Sum.inr.inj h
        obtain ⟨h1, h2⟩ := route_undecided r w b' hr
        exact ⟨by simp [h1], h2⟩
    · rcases ha : (a.route cut w).2 with _ | b' <;> rw [ha] at h
      · simp at h
      · obtain rfl := Sum.inr.inj h
        obtain ⟨h1, h2⟩ := route_undecided a w b' ha
        exact ⟨by simp [h1], h2⟩

theorem qSift_tag (w : FreeMonoid α) (g : Bool) (t : DTree α) :
    ∀ e ∈ (qSift w g t).trace cut, e.2 = g := by
  intro e he
  rw [qSift_trace] at he
  obtain ⟨y, -, rfl⟩ := List.mem_map.1 he
  rfl

theorem qSift_decided (w : FreeMonoid α) (g : Bool) (t : DTree α) (h : (t.sift cut w).isLeft) :
    ∀ e ∈ (qSift w g t).trace cut, cut e.1 ≠ none := by
  intro e he
  rw [qSift_trace] at he
  obtain ⟨y, hy, rfl⟩ := List.mem_map.1 he
  exact route_decided cut t w h y hy

theorem qSift_length (w : FreeMonoid α) (g : Bool) (t : DTree α) :
    ((qSift w g t).trace cut).length ≤ t.depth := by
  rw [qSift_trace, List.length_map]
  exact DTree.route_length_le_depth cut t w

/-- The reads of `bracketAt`'s guard on `p`: none at `lo` or `hi`, else `qAgrees`'s. -/
def qGuard (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (lo hi p : ℕ) (g : Bool) :
    Qry α (Option Bool) :=
  if p = lo then .pure (some true) else if p = hi then .pure (some false)
  else qAgrees t x walkAt p g

theorem qGuard_tag (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (lo hi p : ℕ)
    (g : Bool) : ∀ e ∈ (qGuard t x walkAt lo hi p g).trace cut, e.2 = g := by
  unfold qGuard
  split_ifs
  · simp [Qry.trace]
  · simp [Qry.trace]
  · intro e he
    simp only [qAgrees, Qry.trace_map] at he
    exact qSift_tag cut _ g t e he

theorem qGuard_decided (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (lo hi p : ℕ)
    (g : Bool) : ∀ ⦃v : Bool⦄, (qGuard t x walkAt lo hi p g).run cut = some v →
    ∀ e ∈ (qGuard t x walkAt lo hi p g).trace cut, cut e.1 ≠ none := by
  intro v h
  unfold qGuard at h ⊢
  split_ifs at h ⊢
  · simp [Qry.trace]
  · simp [Qry.trace]
  · simp only [qAgrees, Qry.trace_map, Qry.run_map, qSift_run] at h ⊢
    refine qSift_decided cut _ g t ?_
    rcases hs : t.sift cut (prefixOf x p) with _ | _ <;> simp_all

theorem qGuard_undecided (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (lo hi p : ℕ) (g : Bool) (h : (qGuard t x walkAt lo hi p g).run cut = none) :
    ∃ b, t.sift cut (prefixOf x p) = .inr b ∧ (b, g) ∈ (qGuard t x walkAt lo hi p g).trace cut := by
  unfold qGuard at h ⊢
  split_ifs at h ⊢ with h1 h2
  · simp [Qry.run] at h
  · simp [Qry.run] at h
  · simp only [qAgrees, Qry.trace_map, Qry.run_map, qSift_run] at h ⊢
    rcases hs : t.sift cut (prefixOf x p) with _ | b <;> rw [hs] at h
    · simp at h
    · refine ⟨b, rfl, ?_⟩
      rw [qSift_trace]
      exact List.mem_map.2 ⟨b, (route_undecided cut t _ b hs).1, rfl⟩

theorem qGuard_length (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (lo hi p : ℕ)
    (g : Bool) : ((qGuard t x walkAt lo hi p g).trace cut).length ≤ t.depth := by
  unfold qGuard
  split_ifs
  · simp [Qry.trace]
  · simp [Qry.trace]
  · simp only [qAgrees, Qry.trace_map]
    exact qSift_length cut _ g t

theorem qBracket_unfold (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (ps : List (List Bool)) (fuel lo hi : ℕ) (h : lo + 1 < hi) :
    qBracket (qAgrees t x walkAt) ps (fuel + 1) lo hi =
      (qGuard t x walkAt lo hi ((lo + hi) / 2) true).bind fun o =>
        match o with
        | some true => qBracket (qAgrees t x walkAt) ps fuel ((lo + hi) / 2) hi
        | some false => qBracket (qAgrees t x walkAt) ps fuel lo ((lo + hi) / 2)
        | none => (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).bind fun l =>
          match l with
          | none => .pure (.pair ((lo + hi) / 2 - 1))
          | some lv => (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).bind fun r =>
            match lv, r with
            | _, none => .pure (.pair ((lo + hi) / 2))
            | true, some false => .pure (.triple ((lo + hi) / 2))
            | true, some true => qBracket (qAgrees t x walkAt) ps fuel ((lo + hi) / 2 + 1) hi
            | false, some _ => qBracket (qAgrees t x walkAt) ps fuel lo ((lo + hi) / 2 - 1) := by
  rw [qBracket, if_pos h]
  rfl

/-- A search ending at a triple decides every read but its middles', and its harvest is a
middle's read. -/
theorem qBracket_triple (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (ps : List (List Bool)) :
    ∀ fuel lo hi j, (qBracket (qAgrees t x walkAt) ps fuel lo hi).run cut = .triple j →
      (∀ e ∈ (qBracket (qAgrees t x walkAt) ps fuel lo hi).trace cut, e.2 = false →
        cut e.1 ≠ none)
      ∧ ∃ b, t.sift cut (prefixOf x j) = .inr b
        ∧ (b, true) ∈ (qBracket (qAgrees t x walkAt) ps fuel lo hi).trace cut
  | 0, _, _, _, h => by simp [qBracket, Qry.run] at h
  | fuel + 1, lo, hi, j, h => by
    by_cases hlh : lo + 1 < hi
    swap
    · rw [qBracket, if_neg hlh] at h; simp [Qry.run] at h
    rw [qBracket_unfold t x walkAt ps fuel lo hi hlh] at h ⊢
    set G := fun p g => qGuard t x walkAt lo hi p g
    rw [Qry.run_bind] at h
    simp only [Qry.trace_bind]
    have hmid := qGuard_tag cut t x walkAt lo hi ((lo + hi) / 2) true
    rcases hv : (qGuard t x walkAt lo hi ((lo + hi) / 2) true).run cut with _ | _ | _ <;>
      simp only [hv] at h
    · simp only [] at h ⊢
      rw [Qry.run_bind] at h
      simp only [Qry.trace_bind]
      have hl := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2 - 1) false
      rcases hlv : (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).run cut with _ | lv <;>
        simp only [hlv] at h
      · simp [Qry.run] at h
      simp only [] at h ⊢
      rw [Qry.run_bind] at h
      simp only [Qry.trace_bind]
      have hr := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2 + 1) false
      rcases hrv : (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).run cut with _ | rv <;>
        simp only [hrv] at h
      · simp [Qry.run] at h
      have hdec : ∀ e ∈ (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut
          ++ ((qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut
            ++ (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).trace cut),
          e.2 = false → cut e.1 ≠ none := by
        intro e he hg
        rcases List.mem_append.1 he with he | he
        · rw [hmid e he] at hg; exact absurd hg (by simp)
        rcases List.mem_append.1 he with he | he
        · exact hl hlv e he
        · exact hr hrv e he
      obtain ⟨b, hb, hbm⟩ := qGuard_undecided cut t x walkAt lo hi ((lo + hi) / 2) true hv
      rcases lv <;> rcases rv
      · -- false, some false
        obtain ⟨h1, h2⟩ := qBracket_triple t x walkAt ps fuel lo ((lo + hi) / 2 - 1) j h
        refine ⟨fun e he hg => ?_, ?_⟩
        · simp only [List.append_assoc, List.mem_append] at he
          rcases he with he | he | he | he
          · exact hdec e (by simp [he]) hg
          · exact hdec e (by simp [he]) hg
          · exact hdec e (by simp [he]) hg
          · exact h1 e he hg
        · obtain ⟨b', hb', hm⟩ := h2
          exact ⟨b', hb', by simp [hm]⟩
      · obtain ⟨h1, h2⟩ := qBracket_triple t x walkAt ps fuel lo ((lo + hi) / 2 - 1) j h
        refine ⟨fun e he hg => ?_, ?_⟩
        · simp only [List.append_assoc, List.mem_append] at he
          rcases he with he | he | he | he
          · exact hdec e (by simp [he]) hg
          · exact hdec e (by simp [he]) hg
          · exact hdec e (by simp [he]) hg
          · exact h1 e he hg
        · obtain ⟨b', hb', hm⟩ := h2
          exact ⟨b', hb', by simp [hm]⟩
      · -- true, some false: the triple
        simp only [Qry.run, Outcome.triple.injEq] at h
        subst h
        refine ⟨fun e he hg => ?_, b, hb, by simp [hbm]⟩
        simp only [Qry.trace, List.append_nil] at he
        exact hdec e (by simpa using he) hg
      · obtain ⟨h1, h2⟩ := qBracket_triple t x walkAt ps fuel ((lo + hi) / 2 + 1) hi j h
        refine ⟨fun e he hg => ?_, ?_⟩
        · simp only [List.append_assoc, List.mem_append] at he
          rcases he with he | he | he | he
          · exact hdec e (by simp [he]) hg
          · exact hdec e (by simp [he]) hg
          · exact hdec e (by simp [he]) hg
          · exact h1 e he hg
        · obtain ⟨b', hb', hm⟩ := h2
          exact ⟨b', hb', by simp [hm]⟩
    · obtain ⟨h1, h2⟩ := qBracket_triple t x walkAt ps fuel lo ((lo + hi) / 2) j h
      refine ⟨fun e he hg => ?_, ?_⟩
      · rcases List.mem_append.1 he with he | he
        · rw [hmid e he] at hg; exact absurd hg (by simp)
        · exact h1 e he hg
      · obtain ⟨b', hb', hm⟩ := h2
        exact ⟨b', hb', by simp [hm]⟩
    · obtain ⟨h1, h2⟩ := qBracket_triple t x walkAt ps fuel ((lo + hi) / 2) hi j h
      refine ⟨fun e he hg => ?_, ?_⟩
      · rcases List.mem_append.1 he with he | he
        · rw [hmid e he] at hg; exact absurd hg (by simp)
        · exact h1 e he hg
      · obtain ⟨b', hb', hm⟩ := h2
        exact ⟨b', hb', by simp [hm]⟩

end Trace

section Counts

variable (R : CutReads α)

theorem qGuard_run (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (lo hi p : ℕ)
    (g : Bool) : (qGuard t x walkAt lo hi p g).run R.cut
      = if p = lo then some true else if p = hi then some false else agreesAt R t x walkAt p := by
  unfold qGuard
  split_ifs <;> simp [Qry.run, qAgrees_run]

theorem qGuard_countP (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool) (lo hi p : ℕ)
    (g : Bool) : ((qGuard t x walkAt lo hi p g).trace R.cut).countP (fun e => e.2)
      ≤ if g then t.depth else 0 := by
  have htag := qGuard_tag R.cut t x walkAt lo hi p g
  cases g
  · simp only [Bool.false_eq_true, if_false, Nat.le_zero, List.countP_eq_zero]
    intro e he; simp [htag e he]
  · exact List.countP_le_length.trans (qGuard_length R.cut t x walkAt lo hi p true)

/-- The search's tagged reads are at most the depth for each middle it visits. -/
theorem qBracket_countP (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (ps : List (List Bool)) : ∀ fuel lo hi,
    ((qBracket (qAgrees t x walkAt) ps fuel lo hi).trace R.cut).countP (fun e => e.2)
      ≤ t.depth * visited (agreesAt R t x walkAt) fuel lo hi
  | 0, _, _ => by simp [qBracket, Qry.trace]
  | fuel + 1, lo, hi => by
    by_cases hlh : lo + 1 < hi
    swap
    · simp [qBracket, hlh, Qry.trace]
    have e : ∀ p g, (if p = lo then some true else if p = hi then some false
        else agreesAt R t x walkAt p) = (qGuard t x walkAt lo hi p g).run R.cut :=
      fun p g => (qGuard_run R t x walkAt lo hi p g).symm
    rw [qBracket_unfold t x walkAt ps fuel lo hi hlh, visited, if_pos hlh]
    simp only []
    rw [e ((lo + hi) / 2) true, e ((lo + hi) / 2 - 1) false, e ((lo + hi) / 2 + 1) false]
    simp only [Qry.trace_bind, List.countP_append]
    have hm := qGuard_countP R t x walkAt lo hi ((lo + hi) / 2) true
    have hl := qGuard_countP R t x walkAt lo hi ((lo + hi) / 2 - 1) false
    have hr := qGuard_countP R t x walkAt lo hi ((lo + hi) / 2 + 1) false
    simp only [if_true, if_false, Bool.false_eq_true] at hm hl hr
    rcases (qGuard t x walkAt lo hi ((lo + hi) / 2) true).run R.cut with _ | _ | _
    · simp only [Qry.trace_bind, List.countP_append]
      rcases (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).run R.cut with _ | _ | _
      · simp [Qry.trace]; nlinarith
      all_goals
        simp only [Qry.trace_bind, List.countP_append]
        rcases (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).run R.cut with _ | _ | _
      all_goals first
        | (simp [Qry.trace]; nlinarith)
        | (have := qBracket_countP t x walkAt ps fuel lo ((lo + hi) / 2 - 1)
           simp only [Nat.mul_add, mul_one]; omega)
        | (have := qBracket_countP t x walkAt ps fuel ((lo + hi) / 2 + 1) hi
           simp only [Nat.mul_add, mul_one]; omega)
    · have := qBracket_countP t x walkAt ps fuel lo ((lo + hi) / 2)
      simp only [Nat.mul_add, mul_one]; omega
    · have := qBracket_countP t x walkAt ps fuel ((lo + hi) / 2) hi
      simp only [Nat.mul_add, mul_one]; omega

end Counts

section Fresh

open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

end Fresh

end OrthoDFA
