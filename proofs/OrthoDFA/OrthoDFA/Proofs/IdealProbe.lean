import OrthoDFA.Proofs.IdealTree

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
      E ((walk E s cs).getD i []) c = some ((walk E s cs).getD (i + 1) [], w)
  | s, [], i, h => by simp [walk] at h
  | s, c :: cs, i, h => by
    rcases hE : E s c with _ | ⟨t, w⟩
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
      E ((walk E s cs).getD ((walk E s cs).length - 1) []) c = none
  | s, [], h => by simp [walk] at h
  | s, c :: cs, h => by
    rcases hE : E s c with _ | ⟨t, w⟩
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

theorem walk_mem {L : List (List Bool)} (hE : ∀ q ∈ L, ∀ c t w, E q c = some (t, w) → t ∈ L) :
    ∀ (s : List Bool) (cs : List α), s ∈ L → ∀ q ∈ walk E s cs, q ∈ L
  | s, [], hs, q, hq => by simp only [walk, List.mem_singleton] at hq; rwa [hq]
  | s, c :: cs, hs, q, hq => by
    rcases hE' : E s c with _ | ⟨t, w⟩
    · simp only [walk, hE', List.mem_cons, List.not_mem_nil, or_false] at hq; rwa [hq]
    · simp only [walk, hE', List.mem_cons] at hq
      rcases hq with rfl | hq
      · exact hs
      · exact walk_mem hE t cs (hE s hs c t w hE') q hq

theorem run_walk : ∀ (s : List Bool) (cs : List α) (q : List Bool), run E s cs = some q →
    (walk E s cs).length = cs.length + 1 ∧ (walk E s cs).getD cs.length [] = q
  | s, [], q, h => by simp only [run, Option.some.injEq] at h; simp [walk, h]
  | s, c :: cs, q, h => by
    rcases hE : E s c with _ | ⟨t, w⟩
    · simp [run, hE] at h
    · simp only [run, hE, Option.bind_some] at h
      have hw : walk E s (c :: cs) = s :: walk E t cs := by simp [walk, hE]
      obtain ⟨h1, h2⟩ := run_walk t cs q h
      rw [hw]
      exact ⟨by simp [h1], by simpa using h2⟩

end Walk

/-! ## The search -/

/-- What the search's answer promises between `lo` and `hi`. -/
def FoundOK (ag : ℕ → Option Bool) (lo hi : ℕ) : Found → Prop
  | .edge p => lo < p ∧ p ≤ hi ∧ ag (p - 1) = some true ∧ ag p = some false
  | .triple p => lo < p ∧ p < hi ∧ ag p = none
  | .pair p => lo ≤ p ∧ p + 1 ≤ hi ∧ ag p = none ∧ ag (p + 1) = none

theorem FoundOK.mono {ag : ℕ → Option Bool} {lo hi lo' hi' : ℕ} {f : Found}
    (h : FoundOK ag lo' hi' f) (hlo : lo ≤ lo') (hhi : hi' ≤ hi) : FoundOK ag lo hi f := by
  cases f with
  | edge p => obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨by omega, by omega, h3, h4⟩
  | triple p => obtain ⟨h1, h2, h3⟩ := h; exact ⟨by omega, by omega, h3⟩
  | pair p => obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨by omega, by omega, h3, h4⟩

theorem search_ok (ag : ℕ → Option Bool) : ∀ (n lo hi : ℕ), hi - lo = n → lo ≤ hi →
    ag lo = some true → ag hi = some false → FoundOK ag lo hi (search ag lo hi) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn hle hlo hhi
  have hne : lo ≠ hi := by rintro rfl; rw [hlo] at hhi; simp at hhi
  rw [search]
  split
  · refine ⟨by omega, le_rfl, ?_, hhi⟩
    rwa [show hi - 1 = lo by omega]
  · rename_i hgap
    split
    · rename_i hm
      exact (ih _ (by omega) _ _ rfl (by omega) hm hhi).mono (by omega) le_rfl
    · rename_i hm
      exact (ih _ (by omega) _ _ rfl (by omega) hlo hm).mono le_rfl (by omega)
    · rename_i hm
      split
      · rename_i hl
        refine ⟨by omega, by omega, hl, ?_⟩
        rwa [show (lo + hi) / 2 - 1 + 1 = (lo + hi) / 2 by omega]
      · rename_i hr _
        exact ⟨by omega, by omega, hm, hr⟩
      · exact ⟨by omega, by omega, hm⟩
      · rename_i hl _
        have : (lo + hi) / 2 - 1 ≠ lo := by rintro h; rw [h, hlo] at hl; simp at hl
        exact (ih _ (by omega) _ _ rfl (by omega) hlo hl).mono le_rfl (by omega)
      · rename_i _ hr
        have : (lo + hi) / 2 + 1 ≠ hi := by rintro h; rw [h, hhi] at hr; simp at hr
        exact (ih _ (by omega) _ _ rfl (by omega) hr hhi).mono (by omega) le_rfl

/-! ## Prefixes -/

theorem pre_succ {x : FreeMonoid α} {p : ℕ} {c : α} (h : x.toList[p]? = some c) :
    pre x (p + 1) = pre x p * FreeMonoid.of c := by
  simp only [pre, List.take_add_one, h, Option.toList_some]
  rfl

theorem pre_of_le {x : FreeMonoid α} {p : ℕ} (h : x.toList.length ≤ p) : pre x p = x := by
  simp [pre, List.take_of_length_le h]

/-! ## The outcome -/

/-- An undecided class at the start, the end, or a leaf's edge. -/
def ClsOK (T : DTree α) : Cls α → Prop
  | .start => True
  | .stop => True
  | .cell s _ => s ∈ T.leaves

/-- What an outcome promises against the hypothesis `(T, E)`. -/
def OutcomeOK (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) : Outcome α → Prop
  | .agree => True
  | .startU z => read z = .undecided
  | .endU z => read z = .undecided
  | .edge s c u t => s ∈ T.leaves ∧ T.sift read u = .inl s ∧
      T.sift read (u * FreeMonoid.of c) = .inl t ∧ ∃ t₀ w₀, E s c = some (t₀, w₀) ∧ t₀ ≠ t
  | .triple k zs => ClsOK T k ∧ zs ≠ [] ∧ ∀ z ∈ zs, read z = .undecided
  | .pair k zs => ClsOK T k ∧ zs ≠ [] ∧ ∀ z ∈ zs, read z = .undecided
  | .member s c u t => s ∈ T.leaves ∧ T.sift read u = .inl s ∧
      T.sift read (u * FreeMonoid.of c) = .inl t ∧ E s c = none

/-- The outcomes that do not disagree. -/
def Outcome.clean : Outcome α → Bool
  | .agree => true
  | .startU _ => true
  | .endU _ => true
  | _ => false

section Located

variable (read : FreeMonoid α → ARU) (T : DTree α) (x : FreeMonoid α) (st : ℕ → List Bool)

theorem agrees_inl {p : ℕ} {l : List Bool} (h : T.sift read (pre x p) = .inl l) :
    agrees read T x st p = some (decide (l = st p)) := by
  simp only [agrees, h, Sum.elim_inl]

theorem agrees_true {p : ℕ} (h : agrees read T x st p = some true) :
    T.sift read (pre x p) = .inl (st p) := by
  unfold agrees at h
  rcases hs : T.sift read (pre x p) with l | z <;> rw [hs] at h <;> simp_all

theorem agrees_false {p : ℕ} (h : agrees read T x st p = some false) :
    ∃ l, T.sift read (pre x p) = .inl l ∧ l ≠ st p := by
  unfold agrees at h
  rcases hs : T.sift read (pre x p) with l | z <;> rw [hs] at h <;> simp_all

theorem stuckAt_ok {p : ℕ} (h : agrees read T x st p = none) :
    stuckAt read T x p ≠ [] ∧ ∀ z ∈ stuckAt read T x p, read z = .undecided := by
  unfold agrees at h
  unfold stuckAt
  rcases hs : T.sift read (pre x p) with l | z <;> rw [hs] at h
  · simp at h
  · simpa using DTree.sift_inr read T _ z hs

theorem located_clean (lo hi : ℕ) : (located read T x st lo hi).clean = false := by
  unfold located
  split
  · split <;> rfl
  · rfl
  · rfl

/-- The search between `k` and `j` along a walk whose leaves are `st`. -/
theorem located_ok (E : Edges α) (k j : ℕ) (hkj : k ≤ j)
    (hk : agrees read T x st k = some true) (hj : agrees read T x st j = some false)
    (hmem : ∀ p, k ≤ p → p ≤ j → st p ∈ T.leaves)
    (hstep : ∀ p, k ≤ p → p < j →
      ∃ c w, x.toList[p]? = some c ∧ E (st p) c = some (st (p + 1), w)) :
    OutcomeOK read T E (located read T x st k j) := by
  have hf := search_ok (agrees read T x st) _ k j rfl hkj hk hj
  unfold located
  revert hf
  generalize search (agrees read T x st) k j = f
  intro hf
  have hcell : ∀ p, k ≤ p → p ≤ j → ClsOK T (cellAt x st p) := by
    intro p h1 h2
    unfold cellAt
    split
    · exact hmem p h1 h2
    · trivial
  cases f with
  | edge p =>
    obtain ⟨h1, h2, h3, h4⟩ := hf
    obtain ⟨c, w, hc, hE⟩ := hstep (p - 1) (by omega) (by omega)
    obtain ⟨t, ht, htne⟩ := agrees_false read T x st h4
    have hp : p - 1 + 1 = p := by omega
    simp only [hc, ht]
    refine ⟨hmem _ (by omega) (by omega), agrees_true read T x st h3, ?_, st (p - 1 + 1), w, hE, ?_⟩
    · rw [← pre_succ hc, hp, ht]
    · rw [hp]; exact fun h => htne h.symm
  | triple p =>
    obtain ⟨h1, h2, h3⟩ := hf
    exact ⟨hcell p (by omega) (by omega), stuckAt_ok read T x st h3⟩
  | pair p =>
    obtain ⟨h1, h2, h3, h4⟩ := hf
    obtain ⟨n3, u3⟩ := stuckAt_ok read T x st h3
    obtain ⟨_, u4⟩ := stuckAt_ok read T x st h4
    refine ⟨hcell p (by omega) (by omega), by simp [n3], ?_⟩
    intro z hz
    rcases List.mem_append.1 hz with hz | hz
    · exact u3 z hz
    · exact u4 z hz

end Located

variable (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α) (k : ℕ)

/-- Every learned edge out of a leaf is witnessed. -/
def EdgesOK : Prop :=
  ∀ q ∈ T.leaves, ∀ c t w, E q c = some (t, w) →
    T.sift read w = .inl q ∧ T.sift read (w * FreeMonoid.of c) = .inl t

theorem probe_ok (hE : EdgesOK read T E) (x : FreeMonoid α) :
    OutcomeOK read T E (probe read T E k x) := by
  unfold probe
  rcases ha : T.sift read (pre x k) with a | z
  · simp only
    have hL : ∀ q ∈ T.leaves, ∀ c t w, E q c = some (t, w) → t ∈ T.leaves :=
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
        E (ss.getD (p - k) []) c = some (ss.getD (p + 1 - k) [], w) := by
      intro p h1 h2
      obtain ⟨c, w, hc, hw⟩ := walk_succ E a cs (p - k) (by rw [hss]; omega)
      rw [hss] at hw
      refine ⟨c, w, ?_, by rwa [show p + 1 - k = p - k + 1 by omega]⟩
      rw [← hcs, List.getElem?_drop] at hc
      rwa [show k + (p - k) = p by omega] at hc
    have hst0 : ss.getD (k - k) [] = a := by
      rw [Nat.sub_self, ← hss]
      obtain ⟨l, hl⟩ := walk_cons E a cs
      simp [hl]
    have hagk : agrees read T x (fun p => ss.getD (p - k) []) k = some true := by
      rw [agrees_inl read T x _ ha]
      simp only [hst0, decide_true]
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
      rcases ht : T.sift read (pre x (k + ss.length - 1 + 1)) with t | z
      · rcases hs : T.sift read (pre x (k + ss.length - 1)) with s | z
        · simp only
          split
          · rename_i hseq
            refine ⟨DTree.sift_inl_mem read T _ s hs, hs, ?_, ?_⟩
            · rw [← pre_succ hc, ht]
            · rw [hseq, hj]; exact hE'
          · rename_i hsne
            refine located_ok read T x _ E k _ (by omega) hagk ?_ hmem hstep
            rw [agrees_inl read T x _ hs]
            simp only [hsne, decide_false]
        · exact DTree.sift_inr read T _ z hs
      · exact DTree.sift_inr read T _ z ht
    · rename_i hc
      have hjx : x.toList.length ≤ k + ss.length - 1 := by
        by_contra hcon
        rw [List.getElem?_eq_getElem (by omega)] at hc
        simp at hc
      split
      · rename_i z hz; exact DTree.sift_inr read T _ z hz
      · rename_i e he
        split
        · trivial
        · rename_i hne
          have hkx : k ≤ x.toList.length := by
            by_contra hcon
            have hcs0 : cs = [] := by rw [← hcs]; exact List.drop_eq_nil_of_le (by omega)
            have hss1 : ss.length = 1 := by
              rw [← hss, hcs0]; simp [walk]
            rw [pre_of_le (by omega)] at ha
            rw [ha] at he
            cases he
            apply hne
            rw [show k + ss.length - 1 = k by omega, hst0]
          refine located_ok read T x _ E k _ (by omega) hagk ?_ hmem hstep
          rw [show k + ss.length - 1 = x.toList.length by omega] at hne ⊢
          rw [agrees_inl read T x _ (by rw [pre_of_le le_rfl]; exact he)]
          simp only [hne, decide_false]
  · exact DTree.sift_inr read T _ z ha

theorem probe_disagrees {x : FreeMonoid α} (h : Disagrees read T E k x) :
    (probe read T E k x).clean = false := by
  obtain ⟨a, q, e, ha, hq, he, hne⟩ := h
  obtain ⟨hlen, hlast⟩ := run_walk E a _ q hq
  unfold probe
  simp only [ha]
  generalize hcs : x.toList.drop k = cs at hlen hlast
  generalize hss : walk E a cs = ss at hlen hlast
  have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
  by_cases hkx : k ≤ x.toList.length
  · have hj : k + ss.length - 1 = x.toList.length := by omega
    rw [hj, List.getElem?_eq_none le_rfl, pre_of_le le_rfl] at *
    simp only [he]
    rw [if_neg]
    · exact located_clean read T x _ k _
    · rw [show x.toList.length - k = cs.length by omega, hlast]
      exact fun h => hne h.symm
  · exfalso
    have hcs0 : cs = [] := by rw [← hcs]; exact List.drop_eq_nil_of_le (by omega)
    subst hcs0
    rw [pre_of_le (by omega), he] at ha
    cases ha
    rw [← hss] at hlast
    simp [walk] at hlast
    exact hne hlast.symm

end Ideal

end OrthoDFA
