import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.EdgeGap
import OrthoDFA.Proofs.PassReads
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.BinomLaw

/-!
# `RoundAtK`

A decision on a batch against a fixed hypothesis is an exact binomial test at the look where the
gate stops, or a Hoeffding tail at the batch's end.  An edge reaches the split test or stops at a
string the cut cannot place, because the pass keeps its edges learned: every edge's witness sits
at its leaf and its extension at its target.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Batch

open scoped Classical in
/-- The share of a batch satisfying `P`. -/
noncomputable def share {n : ℕ} (b : Fin n → FreeMonoid α) (P : FreeMonoid α → Prop) : ℝ :=
  ((Finset.univ.filter fun i => P (b i)).card : ℝ) / n

end Batch

section Learned

variable (K : StageKnobs α) (R : CutReads α)

/-- Every edge's witness sits at the edge's leaf, and its extension by the letter at the edge's
target. -/
def Learned (t : DTree α) (edges : Edges α) : Prop :=
  ∀ p c q y, edges p c = some (q, y) →
    t.sift R.cut y = .inl p ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q

omit [Fintype α] in
theorem route_splitAt {cut : FreeMonoid α → Option Bool} (d : FreeMonoid α) :
    ∀ (t : DTree α) (x : FreeMonoid α) (s p : List Bool), (t.route cut x).2 = .inl p → p ≠ s →
      ((t.splitAt d s).route cut x).2 = .inl p
  | .leaf, x, s, p, h, hps => by
    simp only [DTree.route, Sum.inl.injEq] at h
    subst h
    rcases s with _ | ⟨b, s⟩
    · exact absurd rfl hps
    · rfl
  | .node m r a, x, s, p, h, hps => by
    simp only [DTree.route] at h
    rcases hc : cut (x * m) with _ | _ | _ <;> rw [hc] at h
    · simp at h
    · rcases hr : (r.route cut x).2 with p' | b <;> rw [hr] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        rcases s with _ | ⟨_ | _, s⟩
        · simp [DTree.splitAt, DTree.route, hc, hr]
        · have := route_splitAt d r x s p' hr (fun he => hps (by rw [he]))
          simp [DTree.splitAt, DTree.route, hc, this]
        · simp [DTree.splitAt, DTree.route, hc, hr]
      · simp at h
    · rcases ha : (a.route cut x).2 with p' | b <;> rw [ha] at h
      · simp only [Sum.map_inl, Sum.inl.injEq] at h
        subst h
        rcases s with _ | ⟨_ | _, s⟩
        · simp [DTree.splitAt, DTree.route, hc, ha]
        · simp [DTree.splitAt, DTree.route, hc, ha]
        · have := route_splitAt d a x s p' ha (fun he => hps (by rw [he]))
          simp [DTree.splitAt, DTree.route, hc, this]
      · simp at h

omit [Fintype α] in
theorem sift_splitAt {cut : FreeMonoid α → Option Bool} {t : DTree α} {x : FreeMonoid α}
    {d : FreeMonoid α} {s p : List Bool} (h : t.sift cut x = .inl p) (hps : p ≠ s) :
    (t.splitAt d s).sift cut x = .inl p :=
  route_splitAt d t x s p h hps

theorem decisiveTarget_learned {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {c : α} {cur : Option (List Bool)} {q : List Bool} {y : FreeMonoid α}
    (h : decisiveTarget K R t pool path c cur = some (q, y)) :
    t.sift R.cut y = .inl path ∧ t.sift R.cut (y * FreeMonoid.of c) = .inl q := by
  refine ⟨?_, ?_⟩
  · have hm := decisiveTarget_mem K R h
    have := (List.mem_filter.1 (List.mem_of_mem_take hm)).2
    simpa using this
  · unfold decisiveTarget at h
    dsimp only at h
    generalize hV : List.filterMap _ (members K R t pool path) = V at h
    rcases foldl_best_mem _ (fun o v r hr => by
        rcases o with _ | b
        · exact .inl (Option.some.inj hr).symm
        · dsimp only at hr
          split_ifs at hr
          · exact .inl (Option.some.inj hr).symm
          · exact .inr hr) V none (q, y) h with hm | hm
    · rw [← hV] at hm
      obtain ⟨a, -, hfa⟩ := List.mem_filterMap.1 hm
      split at hfa
      · rename_i p hp
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hfa)
        exact hp
      · simp at hfa
    · simp at hm

theorem closeEdges_learned {t : DTree α} {pool : List (FreeMonoid α)} {edges : Edges α}
    (h : Learned R t edges) : Learned R t (closeEdges K R t pool edges) := by
  intro p c q y he
  simp only [closeEdges] at he
  rcases hd : decisiveTarget K R t pool p c ((edges p c).map Prod.fst) with _ | ⟨q', y'⟩ <;>
    rw [hd] at he
  · exact h p c q y he
  · dsimp only at he
    split_ifs at he
    · exact h p c q y he
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj he)
    exact decisiveTarget_learned K R hd

theorem probeStepK_learned {k : ℕ} {s : KState α} {x : FreeMonoid α}
    (h : Learned R s.tree s.edges) :
    Learned R (probeStepK K R k s x).tree (probeStepK K R k s x).edges := by
  simp only [probeStepK]
  split
  · split
    · rename_i d s1 y sprime _
      refine closeEdges_learned K R fun p c q w he => ?_
      rcases hE : s.edges p c with _ | ⟨q', w'⟩ <;> simp only [hE] at he
      · simp at he
      · split_ifs at he with hcl
        all_goals first
          | (simp at he; done)
          | (simp only [Option.some.injEq, Prod.mk.injEq] at he
             obtain ⟨rfl, rfl⟩ := he
             have hcl' : p ≠ s1 ∧ q' ≠ s1 := by tauto
             obtain ⟨h1, h2⟩ := h p c q' w' hE
             exact ⟨sift_splitAt h1 hcl'.1, sift_splitAt h2 hcl'.2⟩)
    · exact closeEdges_learned K R h
    · exact closeEdges_learned K R h
  · exact closeEdges_learned K R h
  · exact closeEdges_learned K R h

theorem parting_none {cut : FreeMonoid α → Option Bool} {x y pre : FreeMonoid α} :
    ∀ t : DTree α, t.parting cut x y pre = none →
      ∃ p, t.sift cut (x * pre) = .inl p ∧ t.sift cut (y * pre) = .inl p
  | .leaf, _ => ⟨[], rfl, rfl⟩
  | .node m r a, h => by
    simp only [DTree.parting] at h
    unfold DTree.sift DTree.route
    simp only [mul_assoc]
    rcases hx : cut (x * (pre * m)) with _ | _ | _ <;>
      rcases hy : cut (y * (pre * m)) with _ | _ | _ <;>
      simp only [hx, hy, reduceCtorEq] at h
    · obtain ⟨p, h1, h2⟩ := parting_none r h
      unfold DTree.sift at h1 h2
      exact ⟨false :: p, by simp [h1], by simp [h2]⟩
    · obtain ⟨p, h1, h2⟩ := parting_none a h
      unfold DTree.sift at h1 h2
      exact ⟨true :: p, by simp [h1], by simp [h2]⟩

omit [Fintype α] [DecidableEq α] in
theorem follow_inl {edges : Edges α} :
    ∀ (cs : List α) (p : List Bool) (ps : List (List Bool)), follow edges p cs = .inl ps →
      ps.length = cs.length + 1 ∧ ps.getD 0 [] = p
        ∧ ∀ i (hi : i < cs.length), ∃ y, edges (ps.getD i []) cs[i] = some (ps.getD (i + 1) [], y)
  | [], p, ps, h => by
    simp only [follow, Sum.inl.injEq] at h
    subst h
    exact ⟨rfl, rfl, fun i hi => absurd hi (by simp)⟩
  | c :: cs, p, ps, h => by
    simp only [follow] at h
    rcases he : edges p c with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp at h
    · simp only [] at h
      rcases hf : follow edges q cs with ps' | r <;> rw [hf] at h
      · simp only [Sum.inl.injEq] at h
        subst h
        obtain ⟨hl, hh, hs⟩ := follow_inl cs q ps' hf
        refine ⟨by simp [hl], rfl, fun i hi => ?_⟩
        rcases i with _ | i
        · exact ⟨y, by simp only [List.getD_cons_zero, List.getD_cons_succ, hh,
            List.getElem_cons_zero]; exact he⟩
        · obtain ⟨y', hy'⟩ := hs i (by simpa using hi)
          exact ⟨y', by simpa using hy'⟩
      · simp at h

omit [Fintype α] [DecidableEq α] in
theorem follow_inr {edges : Edges α} :
    ∀ (cs : List α) (p s : List Bool) (c : α) (i : ℕ), follow edges p cs = .inr (s, c, i) →
      ∃ (hi : i < cs.length), cs[i] = c ∧ edges s c = none
        ∧ ∃ ps, follow edges p (cs.take i) = .inl ps ∧ ps.getLast? = some s
  | [], p, s, c, i, h => by simp [follow] at h
  | c' :: cs, p, s, c, i, h => by
    simp only [follow] at h
    rcases he : edges p c' with _ | ⟨q, y⟩ <;> rw [he] at h
    · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨by simp, rfl, he, [p], by simp [follow], rfl⟩
    · simp only [] at h
      rcases hf : follow edges q cs with ps' | ⟨s', c'', i'⟩ <;> rw [hf] at h
      · simp at h
      · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl, rfl⟩ := h
        obtain ⟨hi, hc, hn, ps, hps, hlast⟩ := follow_inr cs q s' c'' i' hf
        refine ⟨by simpa using hi, by simpa using hc, hn, p :: ps, ?_, ?_⟩
        · simp [follow, he, hps]
        · have hne : ps ≠ [] := by
            intro h0; subst h0; obtain ⟨hl, -⟩ := follow_inl _ _ _ hps; simp at hl
          rw [List.getLast?_cons, hlast]
          simp [hne]

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_edge (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi ps' j, lo < hi → hi - lo ≤ fuel → agrees lo = some true →
      agrees hi = some false → bracketAt (α := α) agrees ps fuel lo hi = .edge ps' j →
      ps' = ps ∧ lo < j ∧ j ≤ hi ∧ agrees (j - 1) = some true ∧ agrees j = some false
  | 0, lo, hi, ps', j, hlt, hf, _, _, _ => by omega
  | fuel + 1, lo, hi, ps', j, hlt, hf, hlo, hhi, h => by
    have hag : ∀ p, (if p = lo then some true else if p = hi then some false else agrees p)
        = agrees p := fun p => by split_ifs <;> simp_all
    simp only [bracketAt, hag] at h
    split_ifs at h with h1
    · rcases hm : agrees ((lo + hi) / 2) with _ | _ | _ <;> simp only [hm] at h
      · rcases hl : agrees ((lo + hi) / 2 - 1) with _ | _ | _ <;>
          rcases hr : agrees ((lo + hi) / 2 + 1) with _ | _ | _ <;>
          simp only [hl, hr, reduceCtorEq] at h
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees ps fuel lo _ ps' j this (by omega) hlo hl h
          exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees ps fuel lo _ ps' j this (by omega) hlo hl h
          exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
        · have : (lo + hi) / 2 + 1 < hi := by
            rcases Nat.lt_or_ge ((lo + hi) / 2 + 1) hi with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 + 1 = hi by omega, hhi] at hr; simp at hr
          have := bracketAt_edge agrees ps fuel _ hi ps' j this (by omega) hr hhi h
          exact ⟨this.1, by omega, this.2.2⟩
      · have := bracketAt_edge agrees ps fuel lo _ ps' j (by omega) (by omega) hlo hm h
        exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
      · have := bracketAt_edge agrees ps fuel _ hi ps' j (by omega) (by omega) hm hhi h
        exact ⟨this.1, by omega, this.2.2⟩
    · obtain ⟨rfl, rfl⟩ := Outcome.edge.inj h
      obtain rfl : hi = lo + 1 := by omega
      exact ⟨rfl, by omega, le_rfl, by simpa using hlo, hhi⟩

/-- The outcomes a search ends at. -/
def Outcome.IsSearch : Outcome α → Prop
  | .pair _ | .edge _ _ | .triple _ => True
  | _ => False

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_isSearch (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi, (bracketAt (α := α) agrees ps fuel lo hi).IsSearch
  | 0, _, _ => trivial
  | fuel + 1, lo, hi => by
    simp only [bracketAt]
    split
    · split
      · exact bracketAt_isSearch agrees ps fuel _ _
      · exact bracketAt_isSearch agrees ps fuel _ _
      · split
        all_goals first | trivial | exact bracketAt_isSearch agrees ps fuel _ _
    · trivial

theorem walkCheck_inl {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α} {o : Outcome α}
    (h : walkCheck R t edges k x = .inl o) : ¬ o.IsSearch := by
  unfold walkCheck at h
  split at h
  · obtain rfl := Sum.inl.inj h; exact id
  · split at h
    · obtain rfl := Sum.inl.inj h; exact id
    · split at h
      · obtain rfl := Sum.inl.inj h; exact id
      · split_ifs at h
        · obtain rfl := Sum.inl.inj h; exact id
  · split at h
    · obtain rfl := Sum.inl.inj h; exact id
    · split_ifs at h
      · obtain rfl := Sum.inl.inj h; exact id

/-- A search outcome comes from a decided disagreement. -/
theorem probeOutcome_search {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {o : Outcome α} (h : probeOutcome R t edges k x = o) (ho : o.IsSearch) :
    ∃ ps hi, walkCheck R t edges k x = .inr (ps, hi)
      ∧ bracketAt (agreesAt R t x fun j => ps.getD (j - k) []) ps (hi - k) k hi = o := by
  unfold probeOutcome at h
  rcases hw : walkCheck R t edges k x with o' | ⟨ps, hi⟩ <;> rw [hw] at h
  · simp only [Sum.elim_inl, id] at h
    subst h
    exact absurd ho (walkCheck_inl R hw)
  · exact ⟨ps, hi, rfl, h⟩

theorem walkCheck_inr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {hi : ℕ} (h : walkCheck R t edges k x = .inr (ps, hi)) :
    ∃ p₀, t.sift R.cut (prefixOf x k) = .inl p₀
      ∧ follow edges p₀ ((x.toList.drop k).take (hi - k)) = .inl ps ∧ k < hi
      ∧ hi ≤ x.toList.length
      ∧ agreesAt R t x (fun j => ps.getD (j - k) []) hi = some false := by
  unfold walkCheck at h
  rcases hw : kWalk R t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hw] at h
  · simp at h
  · unfold kWalk at hw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
    swap
    · simp at hw
    simp only [] at hw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | ⟨s', c', i⟩ <;> rw [hf] at hw
    · simp at hw
    simp only [KWalk.edge.injEq] at hw
    obtain ⟨rfl, rfl, rfl⟩ := hw
    obtain ⟨hi', hc, hn, qs, hqs, hlast⟩ := follow_inr _ _ _ _ _ hf
    simp only [] at h
    rcases h1 : t.sift R.cut (prefixOf x (k + i + 1)) with _ | _ <;> rw [h1] at h
    swap
    · simp at h
    simp only [] at h
    rcases h2 : t.sift R.cut (prefixOf x (k + i)) with p | _ <;> rw [h2] at h
    swap
    · simp at h
    simp only [] at h
    split_ifs at h with hps
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hwt : walkTo R t edges k x (k + i) = qs := by
      simp [walkTo, hk, show k + i - k = i by omega, hqs]
    rw [hwt]
    obtain ⟨hl, -, -⟩ := follow_inl _ _ _ hqs
    simp only [List.length_take, List.length_drop] at hi' hl
    refine ⟨p₀, rfl, by rw [show k + i - k = i by omega]; exact hqs, ?_, by omega, ?_⟩
    · rcases Nat.eq_zero_or_pos i with rfl | hi0
      · exfalso
        simp only [add_zero, List.take_zero] at hqs h2
        simp [follow] at hqs
        subst hqs
        rw [hk] at h2
        obtain rfl := Sum.inl.inj h2
        simp at hlast
        first | exact hps hlast | exact hps hlast.symm
      · omega
    · have hget : qs.getD (k + i - k) [] = s' := by
        rw [show k + i - k = i by omega]
        rw [List.getLast?_eq_getElem?] at hlast
        rw [List.getD_eq_getElem?_getD, show i = qs.length - 1 by omega, hlast]
        rfl
      simp only [agreesAt, h2, Sum.elim_inl, hget, Option.some.injEq, decide_eq_false_iff_not]
      exact hps
  · simp only [] at h
    rcases hs : t.sift R.cut x with a | b <;> rw [hs] at h
    swap
    · simp at h
    simp only [] at h
    split_ifs at h with ha
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    unfold kWalk at hw
    rcases hk : t.sift R.cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
    swap
    · simp at hw
    simp only [] at hw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | r <;> rw [hf] at hw
    swap
    · simp at hw
    simp only [KWalk.reached.injEq] at hw
    subst hw
    obtain ⟨hlen, hhead, -⟩ := follow_inl _ _ _ hf
    set n := x.toList.length
    simp only [List.length_drop] at hlen
    have hlast : ps''.getLast? = some (ps''.getD (n - k) []) := by
      have hidx : n - k < ps''.length := by omega
      rw [List.getLast?_eq_getElem?, show ps''.length - 1 = n - k by omega,
        List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hidx]
      rfl
    have hkn : k < n := by
      by_contra hkn
      have hxk : prefixOf x k = x := by
        apply FreeMonoid.toList.injective
        simp [prefixOf, List.take_of_length_le (not_lt.1 hkn)]
      rw [hxk, hs] at hk
      obtain rfl := Sum.inl.inj hk
      have : ps''.getD (n - k) [] = a := by rw [show n - k = 0 by omega, hhead]
      exact ha (by rw [hlast, this])
    refine ⟨p₀, rfl, by rw [List.take_of_length_le (by simp; omega)]; exact hf, hkn, le_rfl, ?_⟩
    have : prefixOf x n = x := prefixOf_length x
    simp only [agreesAt, this, hs, Sum.elim_inl, Option.some.injEq, decide_eq_false_iff_not]
    intro he
    exact ha (by rw [hlast, he])

end Learned

section Gate

variable (K : StageKnobs α) (R : CutReads α)

end Gate

end OrthoDFA
