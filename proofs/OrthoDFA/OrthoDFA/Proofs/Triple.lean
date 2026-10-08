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

theorem qWalk_facts (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    (∀ e ∈ (qWalk t edges k x).trace R.cut, e.2 = false)
      ∧ ((walkCheck R t edges k x).isRight →
        ∀ e ∈ (qWalk t edges k x).trace R.cut, R.cut e.1 ≠ none) := by
  have hw := qWalk_run R t edges k x
  unfold qWalk at hw ⊢
  simp only [Qry.trace_bind, Qry.run_bind, qSift_run] at hw ⊢
  rcases hk : t.sift R.cut (prefixOf x k) with p | b
  · simp only [hk] at hw ⊢
    rcases hf : follow edges p (x.toList.drop k) with ps | ⟨s, c, i⟩
    · simp only [hf, Qry.trace_bind, Qry.run_bind, qSift_run] at hw ⊢
      rcases hs : t.sift R.cut x with a | b'
      · simp only [hs, Qry.trace, List.append_nil] at hw ⊢
        refine ⟨fun e he => ?_, fun _ e he => ?_⟩
        · rcases List.mem_append.1 he with he | he
          · exact qSift_tag R.cut _ false t e he
          · exact qSift_tag R.cut _ false t e he
        · rcases List.mem_append.1 he with he | he
          · exact qSift_decided R.cut _ false t (by simp [hk]) e he
          · exact qSift_decided R.cut _ false t (by simp [hs]) e he
      · simp only [hs, Qry.trace, List.append_nil, Qry.run] at hw ⊢
        refine ⟨fun e he => ?_, fun hr => ?_⟩
        · rcases List.mem_append.1 he with he | he
          · exact qSift_tag R.cut _ false t e he
          · exact qSift_tag R.cut _ false t e he
        · rw [← hw] at hr; simp at hr
    · simp only [hf, Qry.trace_bind, Qry.run_bind, qSift_run] at hw ⊢
      rcases hs1 : t.sift R.cut (prefixOf x (k + i + 1)) with _ | b'
      · simp only [hs1, Qry.trace_bind, Qry.run_bind, qSift_run] at hw ⊢
        rcases hs2 : t.sift R.cut (prefixOf x (k + i)) with p' | b''
        · simp only [hs2, Qry.trace, List.append_nil] at hw ⊢
          refine ⟨fun e he => ?_, fun _ e he => ?_⟩
          · simp only [List.mem_append] at he
            rcases he with he | he | he <;> exact qSift_tag R.cut _ false t e he
          · simp only [List.mem_append] at he
            rcases he with he | he | he
            · exact qSift_decided R.cut _ false t (by simp [hk]) e he
            · exact qSift_decided R.cut _ false t (by simp [hs1]) e he
            · exact qSift_decided R.cut _ false t (by simp [hs2]) e he
        · simp only [hs2, Qry.trace, List.append_nil, Qry.run] at hw ⊢
          refine ⟨fun e he => ?_, fun hr => ?_⟩
          · simp only [List.mem_append] at he
            rcases he with he | he | he <;> exact qSift_tag R.cut _ false t e he
          · rw [← hw] at hr; simp at hr
      · simp only [hs1, Qry.trace, List.append_nil, Qry.run] at hw ⊢
        refine ⟨fun e he => ?_, fun hr => ?_⟩
        · rcases List.mem_append.1 he with he | he <;> exact qSift_tag R.cut _ false t e he
        · rw [← hw] at hr; simp at hr
  · simp only [hk, Qry.trace, List.append_nil, Qry.run] at hw ⊢
    refine ⟨fun e he => qSift_tag R.cut _ false t e he, fun hr => ?_⟩
    rw [← hw] at hr; simp at hr

/-- A probe's tagged reads are at most the depth for each middle its search visits. -/
theorem qProbe_countP (t : DTree α) (edges : Edges α) (k : ℕ) (x : FreeMonoid α) :
    ((qProbe t edges k x).trace R.cut).countP (fun e => e.2)
      ≤ t.depth * visits R t edges k x := by
  unfold qProbe visits
  rw [Qry.trace_bind, List.countP_append, qWalk_run]
  have hw : ((qWalk t edges k x).trace R.cut).countP (fun e => e.2) = 0 :=
    List.countP_eq_zero.2 fun e he => by simp [(qWalk_facts R t edges k x).1 e he]
  rw [hw, zero_add]
  rcases walkCheck R t edges k x with o | d
  · simp [Qry.trace]
  · exact qBracket_countP R t x _ d.1 _ _ _

/-- A triple's harvest is left undecided, and first read, tagged, at one of the search's
middles. -/
theorem triple_first_read {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α} {j : ℕ}
    {b : FreeMonoid α} (h : probeOutcome R t edges k x = .triple j)
    (hb : tripleRead R t x j = some b) :
    R.cut b = none ∧ ∃ r, ∃ hr : r < ((qProbe t edges k x).trace R.cut).length,
      ((qProbe t edges k x).trace R.cut)[r] = (b, true)
      ∧ ∀ i, ∀ hi : i < r, (((qProbe t edges k x).trace R.cut)[i]'(hi.trans hr)).1 ≠ b := by
  obtain ⟨ps, hi, hw, hbr⟩ := probeOutcome_search R h trivial
  set walkAt : ℕ → List Bool := fun j => ps.getD (j - k) []
  have hbq : (qBracket (qAgrees t x walkAt) ps (hi - k) k hi).run R.cut = .triple j := by
    rw [qBracket_run R.cut _ _ (qAgrees_run R t x walkAt)]; exact hbr
  obtain ⟨hdec, b', hb', hmem⟩ := qBracket_triple R.cut t x walkAt ps (hi - k) k hi j hbq
  have hbb : b' = b := by
    simp only [tripleRead, hb', Sum.getRight?_inr, Option.some.injEq] at hb; exact hb
  subst hbb
  have hcut : R.cut b' = none := by
    have := route_undecided R.cut t (prefixOf x j) b' hb'
    exact this.2
  set L := (qProbe t edges k x).trace R.cut with hL
  have hLeq : L = (qWalk t edges k x).trace R.cut
      ++ (qBracket (qAgrees t x walkAt) ps (hi - k) k hi).trace R.cut := by
    rw [hL, qProbe, Qry.trace_bind, qWalk_run, hw]
    rfl
  have hwd := (qWalk_facts R t edges k x).2 (by simp [hw])
  have hall : ∀ e ∈ L, e.2 = false → R.cut e.1 ≠ none := by
    intro e he hg
    rw [hLeq] at he
    rcases List.mem_append.1 he with he | he
    · exact hwd e he
    · exact hdec e he hg
  have hin : (b', true) ∈ L := by rw [hLeq]; exact List.mem_append_right _ hmem
  have hex : ∃ e ∈ L, (fun e : FreeMonoid α × Bool => decide (e.1 = b')) e := ⟨_, hin, by simp⟩
  set r := L.findIdx fun e => decide (e.1 = b')
  have hr : r < L.length := List.findIdx_lt_length_of_exists hex
  have hrb : (L[r]).1 = b' := by
    have := List.findIdx_getElem (w := hr)
    simpa using this
  refine ⟨hcut, r, hr, ?_, fun i hi => ?_⟩
  · have hg : (L[r]).2 = true := by
      by_contra hg
      exact hall _ (List.getElem_mem hr) (by simpa using hg) (hrb ▸ hcut)
    exact Prod.ext hrb hg
  · have := List.not_of_lt_findIdx hi
    simpa using this

end Counts

section Fresh

open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- `cell_bound`, relative: the chance some string of `Z` off `T` meets an event of its own fresh
bits is at most `φ` times how many such strings there are on average. -/
theorem cell_bound_rel [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α))
    {V : Finset (FreeMonoid α)} (hV : SuffixFree V) (T Z : Ω → Finset (FreeMonoid α))
    (hTZ : ∀ ω ω', (∀ x ∈ vBits V (T ω), O.noise x ω = O.noise x ω') →
      T ω' = T ω ∧ Z ω' = Z ω)
    {W : Finset (FreeMonoid α)} (hW : W ⊆ V)
    (Fl : FreeMonoid α → Set Ω)
    (hFl : ∀ z, MeasurableSet[noiseAlg O ↑(W.image (z * ·))] (Fl z)) {φ : ℝ≥0∞}
    (hφ : ∀ z, μ (Fl z) ≤ φ) :
    μ {ω | ∃ z ∈ Z ω, z ∉ T ω ∧ ω ∈ Fl z} ≤ φ * ∫⁻ ω, ((Z ω \ T ω).card : ℝ≥0∞) ∂μ := by
  classical
  set PC : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    {ω | noisePattern O (vBits V c.1) ω = c.2} ∩ noiseClean O (vBits V c.1) with hPC
  set C : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Set Ω := fun c =>
    PC c ∩ cleanAll O ∩ {ω | T ω = c.1} with hCdef
  have hagree : ∀ c ω ω', ω ∈ C c → ω' ∈ PC c →
      ∀ x ∈ vBits V (T ω), O.noise x ω = O.noise x ω' := by
    intro c ω ω' hω hω' x hx
    obtain ⟨⟨⟨hp, hc⟩, -⟩, hT⟩ := hω
    rw [show T ω = c.1 from hT] at hx
    exact noise_eq_of_pattern O hc hω'.2 (hp.trans hω'.1.symm) x hx
  have hC_eq : ∀ c, (C c).Nonempty → C c = PC c ∩ cleanAll O := by
    intro c ⟨ω₀, h₀⟩
    refine Set.Subset.antisymm (fun ω hω => hω.1) fun ω hω => ⟨hω, ?_⟩
    have := (hTZ ω₀ ω (hagree c ω₀ ω h₀ hω.1)).1
    exact this.trans h₀.2
  set rep : Finset (FreeMonoid α) × Finset (FreeMonoid α) → Finset (FreeMonoid α) := fun c =>
    if h : (C c).Nonempty then Z h.some else ∅ with hrep
  have hZ_eq : ∀ c ω, ω ∈ C c → Z ω = rep c := by
    intro c ω hω
    have hne : (C c).Nonempty := ⟨ω, hω⟩
    simp only [hrep, dif_pos hne]
    exact (hTZ _ ω (hagree c _ ω hne.some_mem hω.1.1)).2
  have hmeasPC : ∀ c, MeasurableSet (PC c) := fun c =>
    noiseAlg_le O _ _ ((measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _))
  have hmeasC : ∀ c, MeasurableSet (C c) := by
    intro c
    by_cases hne : (C c).Nonempty
    · rw [hC_eq c hne]; exact (hmeasPC c).inter (measurableSet_cleanAll O)
    · rw [Set.not_nonempty_iff_eq_empty.1 hne]; exact MeasurableSet.empty
  have hdisj : Pairwise (Function.onFun Disjoint C) := by
    intro c c' hcc'
    rw [Function.onFun, Set.disjoint_left]
    intro ω hω hω'
    apply hcc'
    have h1 : c.1 = c'.1 := (show T ω = c.1 from hω.2).symm.trans (show T ω = c'.1 from hω'.2)
    have h2 : c.2 = c'.2 := by
      have a : noisePattern O (vBits V c.1) ω = c.2 := hω.1.1.1
      have b : noisePattern O (vBits V c'.1) ω = c'.2 := hω'.1.1.1
      rw [h1] at a
      exact a.symm.trans b
    exact Prod.ext h1 h2
  have hcover : {ω | ∃ z ∈ Z ω, z ∉ T ω ∧ ω ∈ Fl z}
      ⊆ (cleanAll O)ᶜ ∪ ⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z) := by
    intro ω ⟨z, hz, hzT, hzF⟩
    by_cases hcl : ω ∈ cleanAll O
    · right
      set c := (T ω, noisePattern O (vBits V (T ω)) ω)
      have hωC : ω ∈ C c := ⟨⟨⟨rfl, fun x _ => hcl x⟩, hcl⟩, rfl⟩
      simp only [Set.mem_iUnion]
      refine ⟨c, z, ?_, hωC, hzF⟩
      rw [← hZ_eq c ω hωC]
      exact Finset.mem_sdiff.2 ⟨hz, hzT⟩
    · exact Or.inl hcl
  have hcell : ∀ c, μ (⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) ≤ φ * ((rep c \ c.1).card * μ (C c)) := by
    intro c
    calc μ (⋃ z ∈ rep c \ c.1, (C c ∩ Fl z))
        ≤ ∑ z ∈ rep c \ c.1, μ (C c ∩ Fl z) := measure_biUnion_finset_le _ _
      _ ≤ ∑ _z ∈ rep c \ c.1, φ * μ (C c) := by
          refine Finset.sum_le_sum fun z hz => ?_
          by_cases hne : (C c).Nonempty
          · have hzT := (Finset.mem_sdiff.1 hz).2
            have hind := indep_noiseAlg O (disjoint_vBits hV hW hzT)
            have hPCm : MeasurableSet[noiseAlg O ↑(vBits V c.1)] (PC c) :=
              (measurableSet_noisePattern O _ c.2).inter (measurableSet_noiseClean O _)
            calc μ (C c ∩ Fl z) ≤ μ (PC c ∩ Fl z) :=
                  measure_mono (Set.inter_subset_inter_left _ fun ω hω => hω.1.1)
              _ = μ (PC c) * μ (Fl z) := (Indep_iff _ _ μ).1 hind _ _ hPCm (hFl z)
              _ ≤ μ (PC c) * φ := by gcongr; exact hφ z
              _ = φ * μ (C c) := by
                  rw [mul_comm, hC_eq c hne, measure_inter_conull (measure_cleanAll_compl O)]
          · rw [Set.not_nonempty_iff_eq_empty.1 hne]
            simp
      _ = (rep c \ c.1).card * (φ * μ (C c)) := by rw [Finset.sum_const, nsmul_eq_mul]
      _ = φ * ((rep c \ c.1).card * μ (C c)) := by ring
  have hset : ∀ c, ∫⁻ ω in C c, ((Z ω \ T ω).card : ℝ≥0∞) ∂μ = (rep c \ c.1).card * μ (C c) := by
    intro c
    rw [← setLIntegral_const]
    refine setLIntegral_congr_fun (hmeasC c) fun ω hω => ?_
    rw [hZ_eq c ω hω, show T ω = c.1 from hω.2]
  have hsum : ∑' c, (rep c \ c.1).card * μ (C c) ≤ ∫⁻ ω, ((Z ω \ T ω).card : ℝ≥0∞) ∂μ := by
    simp_rw [← hset]
    rw [← lintegral_iUnion hmeasC hdisj]
    exact setLIntegral_le_lintegral _ _
  calc μ {ω | ∃ z ∈ Z ω, z ∉ T ω ∧ ω ∈ Fl z}
      ≤ μ ((cleanAll O)ᶜ ∪ ⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := measure_mono hcover
    _ ≤ μ (cleanAll O)ᶜ + μ (⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := measure_union_le _ _
    _ = μ (⋃ c, ⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := by
        rw [measure_cleanAll_compl O, zero_add]
    _ ≤ ∑' c, μ (⋃ z ∈ rep c \ c.1, (C c ∩ Fl z)) := measure_iUnion_le _
    _ ≤ ∑' c, φ * ((rep c \ c.1).card * μ (C c)) := ENNReal.tsum_le_tsum hcell
    _ = φ * ∑' c, (rep c \ c.1).card * μ (C c) := ENNReal.tsum_mul_left
    _ ≤ φ * ∫⁻ ω, ((Z ω \ T ω).card : ℝ≥0∞) ∂μ := by gcongr

theorem cut_readsAt_congr (O : Oracle μ (FreeMonoid α)) (B : State) {F V : Finset (FreeMonoid α)}
    (hFV : F ⊆ V) {ω ω' : Ω} {w : FreeMonoid α}
    (h : ∀ y ∈ vBits V {w}, O.noise y ω = O.noise y ω') :
    (readsAt O B F ω').cut w = (readsAt O B F ω).cut w := by
  have hc : acceptsOn F (fun w => O.mq w ω') w = acceptsOn F (fun w => O.mq w ω) w := by
    unfold acceptsOn
    congr 1
    apply Finset.filter_congr
    intro v hv
    have := h (w * v) (by
      simp only [vBits, Finset.singleton_biUnion, Finset.mem_image]
      exact ⟨v, hFV hv, rfl⟩)
    simp only [Oracle.mq, this]
  have key : ∀ (R R' : CutReads α), R.B = R'.B → acceptsOn R.F R.f w = acceptsOn R'.F R'.f w →
      R.cut w = R'.cut w := by
    intro R R' hB hA
    unfold CutReads.cut
    rw [hB, hA]
  exact key _ _ rfl hc

theorem measurableSet_cut_none [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F : Finset (FreeMonoid α)) (z : FreeMonoid α) :
    MeasurableSet[noiseAlg O ↑(F.image (z * ·))] {ω | (readsAt O B F ω).cut z = none} := by
  classical
  have hcard : ∀ ω, (F.filter fun v => O.mq (z * v) ω = 1).card
      = acceptsOn F (fun w => O.mq w ω) z := fun ω => by
    unfold acceptsOn
    congr
  have hs : {ω | (readsAt O B F ω).cut z = none}
      = {ω | (fun U : Finset (FreeMonoid α) => ¬ B.hi < U.card ∧ ¬ U.card ≤ B.lo)
          (F.filter fun v => O.mq (z * v) ω = 1)} := by
    ext ω
    simp only [Set.mem_ofPred_eq]
    rw [hcard ω]
    show (if B.hi < acceptsOn F (fun w => O.mq w ω) z then some true
      else if acceptsOn F (fun w => O.mq w ω) z ≤ B.lo then some false else none) = none ↔ _
    split_ifs <;> simp <;> omega
  rw [hs]
  exact measurableSet_filter_pred_map O (T := ↑(F.image (z * ·))) (A := F) (z * ·)
    (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv))
    (fun U => ¬ B.hi < U.card ∧ ¬ U.card ≤ B.lo)

omit [Fintype α] [DecidableEq α] in
theorem sum_lintegral_le (f : ℕ → Ω → ℝ≥0∞) (n : ℕ) :
    ∑ i ∈ Finset.range n, ∫⁻ ω, f i ω ∂μ ≤ ∫⁻ ω, ∑ i ∈ Finset.range n, f i ω ∂μ := by
  induction n with
  | zero => simp
  | succ n ih =>
    simp only [Finset.sum_range_succ]
    exact (add_le_add ih le_rfl).trans
      (le_lintegral_add (fun ω => ∑ i ∈ Finset.range n, f i ω) (f n))

omit [Fintype α] [DecidableEq α] in
/-- Summing lower integrals never exceeds the lower integral of the sum. -/
theorem tsum_lintegral_le (f : ℕ → Ω → ℝ≥0∞) :
    ∑' i, ∫⁻ ω, f i ω ∂μ ≤ ∫⁻ ω, ∑' i, f i ω ∂μ :=
  ENNReal.tsum_le_of_sum_range_le fun n =>
    (sum_lintegral_le f n).trans (lintegral_mono fun ω => ENNReal.sum_le_tsum _)

omit [Fintype α] [DecidableEq α] in
theorem tsum_tagged {β : Type*} (L : List (β × Bool)) :
    ∑' r : ℕ, (if (L[r]?.any fun e : β × Bool => e.2) = true then (1 : ℝ≥0∞) else 0)
      = L.countP fun e => e.2 := by
  induction L with
  | nil => simp
  | cons e L ih =>
    rw [tsum_eq_zero_add' ENNReal.summable]
    simp only [List.getElem?_cons_zero, Option.any_some, List.getElem?_cons_succ, ih,
      List.countP_cons]
    rcases e with ⟨_, _ | _⟩ <;> simp [add_comm]

theorem vBits_mono {V Y Y' : Finset (FreeMonoid α)} (h : Y ⊆ Y') : vBits V Y ⊆ vBits V Y' :=
  Finset.biUnion_subset_biUnion_of_subset_left _ h

/-- The probe's reads, in order, on the hypothesis `st ω` under the reads of noise `ω`. -/
noncomputable def probeTrace (O : Oracle μ (FreeMonoid α)) (B : State) (F : Finset (FreeMonoid α))
    (st : Ω → DTree α × Edges α) (k : ℕ) (x : FreeMonoid α) (ω : Ω) :
    List (FreeMonoid α × Bool) :=
  (qProbe (st ω).1 (st ω).2 k x).trace (readsAt O B F ω).cut

/-- Some tagged read in `L` is of a string off `Tp` read for the first time, which the cut leaves
undecided and which `good` holds of. -/
def FirstBad (L : List (FreeMonoid α × Bool)) (Tp : Finset (FreeMonoid α))
    (cut : FreeMonoid α → Option Bool) (good : FreeMonoid α → Prop) : Prop :=
  ∃ r, ∃ hr : r < L.length, L[r].2 = true ∧ L[r].1 ∉ Tp
    ∧ (∀ i, ∀ hi : i < r, (L[i]'(hi.trans hr)).1 ≠ L[r].1) ∧ cut L[r].1 = none ∧ good L[r].1

/-- One draw's fresh reads: with the hypothesis and the strings `Tp` it was built on decided by
the bits `Tp` asks, a probe makes a first undecided read of a string off `Tp` at which `good`
holds, inside `C₀`, with chance at most `u` times its expected tagged reads there. -/
theorem fresh_first_le [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) (B : State)
    (F V : Finset (FreeMonoid α)) (hV : SuffixFree V) (hFV : F ⊆ V)
    (st : Ω → DTree α × Edges α) (Tp : Ω → Finset (FreeMonoid α))
    (hst : ∀ ω ω', (∀ y ∈ vBits V (Tp ω), O.noise y ω = O.noise y ω') →
      st ω' = st ω ∧ Tp ω' = Tp ω)
    (C₀ : Set Ω)
    (hC₀ : ∀ ω ω', (∀ y ∈ vBits V (Tp ω), O.noise y ω = O.noise y ω') → (ω' ∈ C₀ ↔ ω ∈ C₀))
    (k : ℕ) (x : FreeMonoid α) (good : FreeMonoid α → Prop) {u : ℝ≥0∞}
    (hgood : ∀ z, good z → μ {ω | (readsAt O B F ω).cut z = none} ≤ u) :
    μ {ω | ω ∈ C₀ ∧ FirstBad (probeTrace O B F st k x ω) (Tp ω) (readsAt O B F ω).cut good}
      ≤ u * ∫⁻ ω, C₀.indicator
          (fun ω => (((probeTrace O B F st k x ω).countP (·.2) : ℕ) : ℝ≥0∞)) ω ∂μ := by
  classical
  set L := probeTrace O B F st k x with hLdef
  set T : ℕ → Ω → Finset (FreeMonoid α) := fun r ω =>
    Tp ω ∪ (((L ω).take r).map Prod.fst).toFinset
  set Z : ℕ → Ω → Finset (FreeMonoid α) := fun r ω =>
    if ω ∈ C₀ then (((L ω)[r]?.filter (·.2)).map Prod.fst).toFinset else ∅
  set Fl : FreeMonoid α → Set Ω := fun z =>
    if good z then {ω | (readsAt O B F ω).cut z = none} else ∅
  have hFl : ∀ z, MeasurableSet[noiseAlg O ↑(F.image (z * ·))] (Fl z) := by
    intro z
    by_cases hz : good z
    · rw [show Fl z = {ω | (readsAt O B F ω).cut z = none} from if_pos hz]
      exact measurableSet_cut_none O B F z
    · rw [show Fl z = ∅ from if_neg hz]
      exact @MeasurableSet.empty Ω (noiseAlg O ↑(F.image (z * ·)))
  have hφ : ∀ z, μ (Fl z) ≤ u := by
    intro z
    by_cases hz : good z
    · simp only [Fl, if_pos hz]; exact hgood z hz
    · simp [Fl, if_neg hz]
  have hTZ : ∀ r ω ω', (∀ y ∈ vBits V (T r ω), O.noise y ω = O.noise y ω') →
      T r ω' = T r ω ∧ Z r ω' = Z r ω := by
    intro r ω ω' hag
    have hT : ∀ y ∈ vBits V (Tp ω), O.noise y ω = O.noise y ω' :=
      fun y hy => hag y (vBits_mono Finset.subset_union_left hy)
    obtain ⟨hst', hTp'⟩ := hst ω ω' hT
    have hC := hC₀ ω ω' hT
    have htake : (L ω').take (r + 1) = (L ω).take (r + 1) := by
      simp only [hLdef, probeTrace, hst']
      refine Qry.trace_congr _ r fun i hi hir => ?_
      have hi' : i < (L ω).length := hi
      refine cut_readsAt_congr O B hFV fun y hy => hag y (vBits_mono ?_ hy)
      intro w hw
      rw [Finset.mem_singleton] at hw
      subst hw
      refine Finset.mem_union_right _ ?_
      simp only [List.mem_toFinset, List.mem_map]
      exact ⟨_, List.mem_iff_getElem.2 ⟨i, by simp; omega, by simp [List.getElem_take]; rfl⟩, rfl⟩
    have hsplit : ∀ l : List (FreeMonoid α × Bool), l.take r = (l.take (r + 1)).take r :=
      fun l => by rw [List.take_take, min_eq_left (by omega)]
    have h1 : (L ω').take r = (L ω).take r := by
      rw [hsplit (L ω'), htake, ← hsplit]
    have h2 : (L ω')[r]? = (L ω)[r]? := by
      rw [← List.getElem?_take_of_lt (Nat.lt_succ_self r), htake,
        List.getElem?_take_of_lt (Nat.lt_succ_self r)]
    refine ⟨by simp only [T, hTp', h1], ?_⟩
    simp only [Z, h2]
    by_cases hω : ω ∈ C₀
    · rw [if_pos hω, if_pos (hC.2 hω)]
    · rw [if_neg hω, if_neg fun h => hω (hC.1 h)]
  have hsub : {ω | ω ∈ C₀ ∧ FirstBad (L ω) (Tp ω) (readsAt O B F ω).cut good}
      ⊆ ⋃ r, {ω | ∃ z ∈ Z r ω, z ∉ T r ω ∧ ω ∈ Fl z} := by
    rintro ω ⟨hC, r, hr, htag, hTp, hfirst, hcut, hgd⟩
    refine Set.mem_iUnion.2 ⟨r, (L ω)[r].1, ?_, ?_, ?_⟩
    · simp only [Z, if_pos hC, List.getElem?_eq_getElem hr, Option.filter, htag, if_true,
        Option.map_some, Option.toFinset_some, Finset.mem_singleton]
    · simp only [T, Finset.mem_union, List.mem_toFinset, List.mem_map, not_or, not_exists,
        not_and]
      refine ⟨hTp, fun e he hfe => ?_⟩
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 he
      simp only [List.length_take] at hi
      have := hfirst i (by omega)
      simp only [List.getElem_take] at hfe
      exact this hfe
    · simp only [Fl, if_pos hgd]; exact hcut
  have hcard : ∀ r ω, ((Z r ω \ T r ω).card : ℝ≥0∞)
      ≤ C₀.indicator (fun ω => if ((L ω)[r]?.any fun e : FreeMonoid α × Bool => e.2) = true
          then (1 : ℝ≥0∞) else 0) ω := by
    intro r ω
    refine (Nat.cast_le.2 (Finset.card_le_card Finset.sdiff_subset)).trans ?_
    by_cases hC : ω ∈ C₀
    · rw [Set.indicator_of_mem hC]
      simp only [Z, if_pos hC]
      rcases (L ω)[r]? with _ | ⟨w, _ | _⟩ <;> simp [Option.filter]
    · simp [Z, hC]
  calc μ {ω | ω ∈ C₀ ∧ FirstBad (L ω) (Tp ω) (readsAt O B F ω).cut good}
      ≤ μ (⋃ r, {ω | ∃ z ∈ Z r ω, z ∉ T r ω ∧ ω ∈ Fl z}) := measure_mono hsub
    _ ≤ ∑' r, μ {ω | ∃ z ∈ Z r ω, z ∉ T r ω ∧ ω ∈ Fl z} := measure_iUnion_le _
    _ ≤ ∑' r, u * ∫⁻ ω, ((Z r ω \ T r ω).card : ℝ≥0∞) ∂μ := ENNReal.tsum_le_tsum fun r =>
        cell_bound_rel O hV (T r) (Z r) (hTZ r) hFV Fl hFl hφ
    _ = u * ∑' r, ∫⁻ ω, ((Z r ω \ T r ω).card : ℝ≥0∞) ∂μ := ENNReal.tsum_mul_left
    _ ≤ u * ∫⁻ ω, ∑' r, C₀.indicator
          (fun ω => if ((L ω)[r]?.any fun e : FreeMonoid α × Bool => e.2) = true
            then (1 : ℝ≥0∞) else 0) ω ∂μ := by
        gcongr
        exact (ENNReal.tsum_le_tsum fun r => lintegral_mono (hcard r)).trans
          (tsum_lintegral_le _)
    _ = u * ∫⁻ ω, C₀.indicator
          (fun ω => (((L ω).countP (·.2) : ℕ) : ℝ≥0∞)) ω ∂μ := by
        congr 1
        refine lintegral_congr fun ω => ?_
        by_cases hC : ω ∈ C₀
        · simp only [Set.indicator_of_mem hC, tsum_tagged]
        · simp [Set.indicator_of_notMem hC]

end Fresh

end OrthoDFA
