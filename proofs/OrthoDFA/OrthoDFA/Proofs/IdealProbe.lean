import OrthoDFA.Proofs.IdealTree
import OrthoDFA.Proofs.TripartiteSearch

/-!
# A probe under ideal reads

With every learned edge out of a leaf witnessed, a probe's outcome is what it claims: an edge
outcome is a string at a leaf whose successor sifts off the edge's target, an unlearned edge's is
one at a leaf with the edge unlearned, and every string an undecided outcome carries reads
undecided. A probe on which the edges and the tree disagree never comes out clean.
-/

namespace OrthoDFA

namespace Ideal

variable {α : Type*}

/-! ## The walk -/

section Walk

variable (E : Edges α)

theorem walk_cons (s : List Bool) (cs : List α) : ∃ l, walk E s cs = s :: l := by
  cases cs <;> simp [walk]

theorem walk_getD_zero (s : List Bool) (cs : List α) : (walk E s cs).getD 0 [] = s := by
  obtain ⟨l, hl⟩ := walk_cons E s cs; simp [hl]

theorem length_walk_pos (s : List Bool) (cs : List α) : 0 < (walk E s cs).length := by
  obtain ⟨l, hl⟩ := walk_cons E s cs; simp [hl]

theorem length_walk_le : ∀ (s : List Bool) (cs : List α), (walk E s cs).length ≤ cs.length + 1
  | s, [] => by simp [walk]
  | s, c :: cs => by
    simp only [walk, List.length_cons, add_le_add_iff_right]
    split
    · exact length_walk_le _ cs
    · simp

theorem walk_succ : ∀ (s : List Bool) (cs : List α) (i : ℕ), i + 1 < (walk E s cs).length →
    ∃ c w, cs[i]? = some c ∧
      E ((walk E s cs).getD i [], c) = some ((walk E s cs).getD (i + 1) [], w)
  | s, [], i, h => by simp [walk] at h
  | s, c :: cs, i, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · simp [walk, hE] at h
    · have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      rw [hw] at h ⊢
      rcases i with _ | i
      · obtain ⟨l, hl⟩ := walk_cons E t cs
        exact ⟨c, w, rfl, by simpa [hl] using hE⟩
      · simp only [List.length_cons, add_lt_add_iff_right] at h
        obtain ⟨c', w', h1, h2⟩ := walk_succ t cs i h
        exact ⟨c', w', by simpa using h1, by simpa using h2⟩

theorem walk_stop : ∀ (s : List Bool) (cs : List α), (walk E s cs).length ≤ cs.length →
    ∃ c, cs[(walk E s cs).length - 1]? = some c ∧
      E ((walk E s cs).getD ((walk E s cs).length - 1) [], c) = none
  | s, [], h => by simp [walk] at h
  | s, c :: cs, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · have hw : walk E s (c :: cs) = [s] := by simp [walk, hE]
      rw [hw]
      exact ⟨c, rfl, by simpa using hE⟩
    · have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      rw [hw] at h ⊢
      simp only [List.length_cons, add_le_add_iff_right] at h
      obtain ⟨c', h1, h2⟩ := walk_stop t cs h
      have hpos := length_walk_pos E t cs
      refine ⟨c', ?_, ?_⟩
      · simp only [List.length_cons, Nat.add_sub_cancel]
        rw [show (walk E t cs).length = (walk E t cs).length - 1 + 1 by omega]
        simpa using h1
      · simp only [List.length_cons, Nat.add_sub_cancel]
        rw [show (walk E t cs).length = (walk E t cs).length - 1 + 1 by omega]
        simpa using h2

theorem walk_mem {L : List (List Bool)} (hE : ∀ q ∈ L, ∀ c t w, E (q, c) = some (t, w) → t ∈ L) :
    ∀ (s : List Bool) (cs : List α), s ∈ L → ∀ q ∈ walk E s cs, q ∈ L
  | s, [], hs, q, hq => by simp only [walk, List.mem_singleton] at hq; rwa [hq]
  | s, c :: cs, hs, q, hq => by
    rcases hE' : E (s, c) with _ | ⟨t, w⟩
    · simp only [walk, hE', List.mem_cons, List.not_mem_nil, or_false] at hq; rwa [hq]
    · simp only [walk, hE', List.mem_cons] at hq
      rcases hq with rfl | hq
      · exact hs
      · exact walk_mem hE t cs (hE s hs c t w hE') q hq

theorem run_walk : ∀ (s : List Bool) (cs : List α) (q : List Bool), walkEnd E s cs = some q →
    (walk E s cs).length = cs.length + 1 ∧ (walk E s cs).getD cs.length [] = q
  | s, [], q, h => by simp only [walkEnd, Option.some.injEq] at h; simp [walk, h]
  | s, c :: cs, q, h => by
    rcases hE : E (s, c) with _ | ⟨t, w⟩
    · simp [walkEnd, hE] at h
    · simp only [walkEnd, hE, Option.bind_some] at h
      have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      obtain ⟨h1, h2⟩ := run_walk t cs q h
      rw [hw]
      exact ⟨by simp [h1], by simpa using h2⟩

end Walk

/-! ## Prefixes -/

theorem pre_succ {x : FreeMonoid α} {p : ℕ} {c : α} (h : x.toList[p]? = some c) :
    prefixOf x (p + 1) = prefixOf x p * FreeMonoid.of c := by
  simp only [prefixOf, List.take_add_one, h, Option.toList_some]
  rfl

theorem pre_of_le {x : FreeMonoid α} {p : ℕ} (h : x.toList.length ≤ p) : prefixOf x p = x := by
  simp [prefixOf, List.take_of_length_le h]

/-! ## The outcome -/

/-- An undecided site at the start, the stop, or a leaf's edge. -/
def SiteOK (T : DTree α) : UndecidedSite α → Prop
  | .start => True
  | .stop => True
  | .edge s _ => s ∈ T.leaves

/-- What an outcome promises against the hypothesis `(T, E)`. -/
def OutcomeOK (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) : Outcome α → Prop
  | .agree => True
  | .wrongEdge s c u t => s ∈ T.leaves ∧ T.sift read u = .inl s ∧
      T.sift read (u * FreeMonoid.of c) = .inl t ∧ ∃ t₀ w₀, E (s, c) = some (t₀, w₀) ∧ t₀ ≠ t
  | .unlearnedEdge s c u t => s ∈ T.leaves ∧ T.sift read u = .inl s ∧
      T.sift read (u * FreeMonoid.of c) = .inl t ∧ E (s, c) = none
  | .undecided k zs cl => SiteOK T k ∧ (cl = false → zs ≠ []) ∧ ∀ z ∈ zs, read z = .undecided

/-- The outcomes that do not disagree. -/
def Outcome.clean : Outcome α → Bool
  | .agree => true
  | .undecided _ _ cl => cl
  | _ => false

theorem outcomeOK_one {read : FreeMonoid α → ARU} {T : DTree α} {E : Edges α} {k : UndecidedSite α}
    (hk : SiteOK T k) {z : FreeMonoid α} (h : read z = .undecided) :
    OutcomeOK read T E (.undecided k [z] true) :=
  ⟨hk, by simp, by simpa using h⟩

section Located

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

theorem agrees_inl {p : ℕ} {l : List Bool} (h : T.sift read (prefixOf x p) = .inl l) :
    siftAgrees read T x st p = some (decide (l = st p)) := by
  simp only [siftAgrees, h, Sum.elim_inl]

theorem agrees_true {p : ℕ} (h : siftAgrees read T x st p = some true) :
    T.sift read (prefixOf x p) = .inl (st p) := by
  unfold siftAgrees at h
  rcases hs : T.sift read (prefixOf x p) with l | z <;> rw [hs] at h <;> simp_all

theorem agrees_false {p : ℕ} (h : siftAgrees read T x st p = some false) :
    ∃ l, T.sift read (prefixOf x p) = .inl l ∧ l ≠ st p := by
  unfold siftAgrees at h
  rcases hs : T.sift read (prefixOf x p) with l | z <;> rw [hs] at h <;> simp_all

theorem stuckAt_ok {p : ℕ} (h : siftAgrees read T x st p = none) :
    undecidedAt read T x p ≠ [] ∧ ∀ z ∈ undecidedAt read T x p, read z = .undecided := by
  unfold siftAgrees at h
  unfold undecidedAt
  rcases hs : T.sift read (prefixOf x p) with l | z <;> rw [hs] at h
  · simp at h
  · simpa using DTree.sift_inr read T _ z hs

/-- The search between `k` and `j` along a walk whose leaves are `st`, `un` the unlearned edge
the walk stopped at, if any. -/
theorem searchOutcome_ok (E : Edges α) (k j : ℕ) (hkj : k ≤ j)
    (hmem : ∀ p, k ≤ p → p ≤ j → st p ∈ T.leaves)
    (hstep : ∀ p, k ≤ p → p < j →
      ∃ c w, x.toList[p]? = some c ∧ E (st p, c) = some (st (p + 1), w))
    (un : Option (α × List Bool))
    (hun : ∀ c t, un = some (c, t) → x.toList[j]? = some c
      ∧ T.sift read (prefixOf x (j + 1)) = .inl t ∧ E (st j, c) = none) :
    OutcomeOK read T E
      (searchOutcome read T x st un j (tripartiteSearch true (siftAgrees read T x st) k j).1) := by
  have hs := tripartiteSearch_ok (recover := true) (agrees := siftAgrees read T x st) hkj
  generalize (tripartiteSearch true (siftAgrees read T x st) k j).1 = r at hs
  have hcell : ∀ p, k ≤ p → p ≤ j → SiteOK T (siteAt x st p) := by
    intro p h1 h2
    unfold siteAt
    split
    · exact hmem p h1 h2
    · trivial
  cases r with
  | undecidedLo => exact ⟨trivial, by simp, by simp⟩
  | disagreeLo => exact ⟨trivial, by simp, by simp⟩
  | undecidedHi =>
    exact ⟨trivial, (fun h => by cases h), (stuckAt_ok read T x st hs.2).2⟩
  | allAgree =>
    rcases un with _ | ⟨c, t⟩
    · trivial
    · obtain ⟨hc, ht, hE⟩ := hun c t rfl
      refine ⟨hmem j hkj le_rfl, agrees_true read T x st hs.2, ?_, hE⟩
      rw [← pre_succ hc, ht]
  | edge p =>
    obtain ⟨h1, h2, h3, h4⟩ := hs
    obtain ⟨c, w, hc, hE⟩ := hstep (p - 1) (by omega) (by omega)
    obtain ⟨t, ht, htne⟩ := agrees_false read T x st h4
    have hp : p - 1 + 1 = p := by omega
    simp only [searchOutcome, hc, ht]
    refine ⟨hmem _ (by omega) (by omega), agrees_true read T x st h3, ?_, st (p - 1 + 1), w, hE, ?_⟩
    · rw [← pre_succ hc, hp, ht]
    · rw [hp]; exact fun h => htne h.symm
  | triple p =>
    obtain ⟨h1, h2, -, h3, -⟩ := hs
    obtain ⟨n3, u3⟩ := stuckAt_ok read T x st h3
    exact ⟨hcell p (by omega) (by omega), fun _ => n3, u3⟩
  | undecidedMid p => exact absurd hs.1 (by simp)
  | pair p =>
    obtain ⟨h1, h2, h3, h4⟩ := hs
    obtain ⟨n3, u3⟩ := stuckAt_ok read T x st h3
    obtain ⟨_, u4⟩ := stuckAt_ok read T x st h4
    refine ⟨hcell p (by omega) (by omega), by simp [searchOutcome, n3], ?_⟩
    intro z hz
    rcases List.mem_append.1 hz with hz | hz
    · exact u3 z hz
    · exact u4 z hz

/-- Where the walk agrees at `k` and disagrees at `j`, the search's outcome is not clean. -/
theorem searchOutcome_clean (un : Option (α × List Bool)) (k j : ℕ) (hkj : k ≤ j)
    (hk : siftAgrees read T x st k = some true) (hj : siftAgrees read T x st j = some false) :
    (searchOutcome read T x st un j
      (tripartiteSearch true (siftAgrees read T x st) k j).1).clean = false := by
  have hs := tripartiteSearch_ok (recover := true) (agrees := siftAgrees read T x st) hkj
  generalize (tripartiteSearch true (siftAgrees read T x st) k j).1 = r at hs
  cases r with
  | undecidedLo => simp [SearchOK, hk] at hs
  | disagreeLo => simp [SearchOK, hk] at hs
  | undecidedHi => simp [SearchOK, hj] at hs
  | allAgree => simp [SearchOK, hj] at hs
  | edge p => simp only [searchOutcome]; split <;> rfl
  | triple p => rfl
  | undecidedMid p => rfl
  | pair p => rfl

end Located

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)

/-- Every learned edge out of a leaf is witnessed. -/
def EdgesOK : Prop :=
  ∀ q ∈ T.leaves, ∀ c t w, E (q, c) = some (t, w) →
    T.sift read w = .inl q ∧ T.sift read (w * FreeMonoid.of c) = .inl t

theorem probe_ok (hE : EdgesOK read T E) (x : FreeMonoid α) :
    OutcomeOK read T E (probe read T E k x) := by
  unfold probe
  rcases ha : T.sift read (prefixOf x k) with a | z
  · simp only
    have hL : ∀ q ∈ T.leaves, ∀ c t w, E (q, c) = some (t, w) → t ∈ T.leaves :=
      fun q hq c t w h => DTree.sift_inl_mem read T _ t (hE q hq c t w h).2
    generalize hcs : x.toList.drop k = cs
    generalize hss : walk E a cs = ss
    have hpos : 0 < ss.length := hss ▸ length_walk_pos E a cs
    have hle : ss.length ≤ cs.length + 1 := hss ▸ length_walk_le E a cs
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    have hmem : ∀ p, k ≤ p → p ≤ k + ss.length - 1 → ss.getD (p - k) [] ∈ T.leaves := by
      intro p h1 h2
      have hin : p - k < ss.length := by omega
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hin, Option.getD_some]
      have hw := walk_mem E hL a cs (DTree.sift_inl_mem read T _ a ha)
      rw [hss] at hw
      exact hw _ (List.getElem_mem hin)
    have hstep : ∀ p, k ≤ p → p < k + ss.length - 1 → ∃ c w, x.toList[p]? = some c ∧
        E (ss.getD (p - k) [], c) = some (ss.getD (p + 1 - k) [], w) := by
      intro p h1 h2
      obtain ⟨c, w, hc, hw⟩ := walk_succ E a cs (p - k) (by rw [hss]; omega)
      rw [hss] at hw
      refine ⟨c, w, ?_, by rwa [show p + 1 - k = p - k + 1 by omega]⟩
      rw [← hcs, List.getElem?_drop] at hc
      rwa [show k + (p - k) = p by omega] at hc
    split
    · -- the walk stopped at an unlearned edge
      rename_i c hc
      have hjx : k + ss.length - 1 < x.toList.length := by
        by_contra hcon
        rw [List.getElem?_eq_none (by omega)] at hc
        simp at hc
      have hstop : ss.length ≤ cs.length := by omega
      obtain ⟨c', hc', hE'⟩ := walk_stop E a cs (by rw [hss]; exact hstop)
      rw [hss] at hc' hE'
      rw [← hcs, List.getElem?_drop, show k + (ss.length - 1) = k + ss.length - 1 by omega,
        hc] at hc'
      cases hc'
      have hj : k + ss.length - 1 - k = ss.length - 1 := by omega
      rcases ht : T.sift read (prefixOf x (k + ss.length - 1 + 1)) with t | z
      · refine searchOutcome_ok read T x _ E k _ (by omega) hmem hstep _ fun c' t' h => ?_
        cases h
        exact ⟨hc, ht, by rw [hj]; exact hE'⟩
      · exact outcomeOK_one (k := .stop) trivial (DTree.sift_inr read T _ z ht)
    · exact searchOutcome_ok read T x _ E k _ (by omega) hmem hstep _ fun _ _ h => by cases h
  · exact outcomeOK_one (k := .start) trivial (DTree.sift_inr read T _ z ha)

theorem probeOrAgree_ok (hE : EdgesOK read T E) (x : FreeMonoid α) :
    OutcomeOK read T E (probeOrAgree read T E k x) := by
  unfold probeOrAgree
  split_ifs
  · trivial
  · exact probe_ok read T E k hE x

theorem probe_disagrees {x : FreeMonoid α} (h : Disagrees read ⟨T, E⟩ k x) :
    (probe read T E k x).clean = false := by
  obtain ⟨a, q, e, ha, hq, he, hne⟩ := h
  dsimp only at ha hq he
  obtain ⟨hlen, hlast⟩ := run_walk E a _ q hq
  unfold probe
  simp only [ha]
  generalize hcs : x.toList.drop k = cs at hlen hlast
  generalize hss : walk E a cs = ss at hlen hlast
  have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
  by_cases hkx : k ≤ x.toList.length
  · have hj : k + ss.length - 1 = x.toList.length := by omega
    have hst0 : ss.getD (k - k) [] = a := by
      rw [Nat.sub_self, ← hss]
      obtain ⟨l, hl⟩ := walk_cons E a cs
      simp [hl]
    rw [hj, List.getElem?_eq_none le_rfl]
    refine searchOutcome_clean read T x _ _ k _ hkx ?_ ?_
    · rw [agrees_inl read T x _ ha]
      simp only [hst0, decide_true]
    · rw [agrees_inl read T x _ (by rw [pre_of_le le_rfl]; exact he)]
      rw [show x.toList.length - k = cs.length by omega, hlast]
      simpa using fun h => hne h.symm
  · exfalso
    have hcs0 : cs = [] := by rw [← hcs]; exact List.drop_eq_nil_of_le (by omega)
    subst hcs0
    rw [pre_of_le (by omega), he] at ha
    cases ha
    rw [← hss] at hlast
    simp [walk] at hlast
    exact hne hlast.symm

theorem disagrees_long {x : FreeMonoid α} (h : Disagrees read ⟨T, E⟩ k x) :
    k ≤ x.toList.length := by
  obtain ⟨a, q, e, ha, hq, he, hne⟩ := h
  by_contra hkx
  dsimp only at ha hq he
  rw [pre_of_le (by omega), he, Sum.inl.injEq] at ha
  rw [List.drop_eq_nil_of_le (by omega)] at hq
  simp only [walkEnd, Option.some.injEq] at hq
  exact hne (hq.symm.trans ha.symm)

theorem probeOrAgree_disagrees {x : FreeMonoid α} (h : Disagrees read ⟨T, E⟩ k x) :
    (probeOrAgree read T E k x).clean = false := by
  unfold probeOrAgree
  rw [if_neg (by have := disagrees_long read T E k h; omega)]
  exact probe_disagrees read T E k h

end Ideal

end OrthoDFA
