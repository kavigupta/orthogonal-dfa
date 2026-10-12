import OrthoDFA.Proofs.RandomTree

/-!
# What a probe harvests

Every string a probe harvests, at its start, at an undecided middle or charged to an edge, was
read undecided at a prefix of length at least `k` followed by a midfix: it is in `pot k x T`. A
probe charges at most `|x| - k` positions, and a probe the hypothesis disagrees on searches.
-/

namespace OrthoDFA

namespace Random

open OrthoDFA.Ideal (bisect siftAgrees probe Outcome locate undecidedAt SearchResult walk_cons
  length_walk_le run_walk pre_of_le)

variable {α : Type*} (read : FreeMonoid α → ARU)

theorem sift_inr_mids : ∀ (T : DTree α) (v z : FreeMonoid α), T.sift read v = .inr z →
    ∃ m ∈ mids T, z = v * m
  | .leaf, _, _, h => by simp [DTree.sift] at h
  | .node m r a, v, z, h => by
    classical
    simp only [DTree.sift] at h
    split at h
    · rcases hq : a.sift read v with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inr.injEq, reduceCtorEq] at h
      obtain ⟨m', hm', rfl⟩ := sift_inr_mids a v q hq
      exact ⟨m', by simp [mids, hm'], h.symm⟩
    · rcases hq : r.sift read v with q | q <;> rw [hq] at h <;> simp only [Sum.map_inl,
        Sum.map_inr, Sum.inr.injEq, reduceCtorEq] at h
      obtain ⟨m', hm', rfl⟩ := sift_inr_mids r v q hq
      exact ⟨m', by simp [mids, hm'], h.symm⟩
    · simp only [Sum.inr.injEq] at h
      exact ⟨m, by simp [mids], h.symm⟩

theorem mem_pot {k p : ℕ} {x : FreeMonoid α} {T : DTree α} {m : FreeMonoid α} (h1 : k ≤ p)
    (h2 : p ≤ x.toList.length) (hm : m ∈ mids T) : prefixOf x p * m ∈ pot k x T := by
  classical
  unfold pot
  exact Finset.mem_biUnion.2 ⟨p, Finset.mem_Icc.2 ⟨h1, h2⟩, Finset.mem_image.2 ⟨m, hm, rfl⟩⟩

/-- A string read undecided by the sift of a prefix of length between `k` and `|x|`. -/
theorem pot_of_sift {T : DTree α} {k p : ℕ} {x z : FreeMonoid α} (h1 : k ≤ p)
    (h2 : p ≤ x.toList.length) (h : T.sift read (prefixOf x p) = .inr z) :
    z ∈ pot k x T ∧ read z = .undecided := by
  obtain ⟨m, hm, rfl⟩ := sift_inr_mids read T _ z h
  exact ⟨mem_pot h1 h2 hm, DTree.sift_inr read T _ _ h⟩

/-! ## The bisection's stopping points -/

/-- Where the bisection's answer lies between `lo` and `hi`. -/
def PosIn (lo hi : ℕ) : SearchResult → Prop
  | .edge _ => True
  | .triple p => lo ≤ p ∧ p ≤ hi
  | .pair p => lo ≤ p ∧ p + 1 ≤ hi

theorem PosIn.mono {lo hi lo' hi' : ℕ} {f : SearchResult} (h : PosIn lo' hi' f) (hlo : lo ≤ lo')
    (hhi : hi' ≤ hi) : PosIn lo hi f := by
  cases f with
  | edge p => trivial
  | triple p => obtain ⟨h1, h2⟩ := h; exact ⟨by omega, by omega⟩
  | pair p => obtain ⟨h1, h2⟩ := h; exact ⟨by omega, by omega⟩

theorem search_pos (ag : ℕ → Option Bool) : ∀ (n lo hi : ℕ), hi - lo = n →
    PosIn lo hi (bisect ag lo hi) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn
  rw [bisect]
  split
  · trivial
  · rename_i hgap
    split
    · exact (ih _ (by omega) _ _ rfl).mono (by omega) le_rfl
    · exact (ih _ (by omega) _ _ rfl).mono le_rfl (by omega)
    · split
      · exact ⟨by omega, by omega⟩
      · exact ⟨by omega, by omega⟩
      · exact ⟨by omega, by omega⟩
      · exact (ih _ (by omega) _ _ rfl).mono le_rfl (by omega)
      · exact (ih _ (by omega) _ _ rfl).mono (by omega) le_rfl

variable (T : DTree α) (x : FreeMonoid α)

theorem located_pt (st : ℕ → List Bool) (lo hi : ℕ) :
    ∀ z ∈ middleString (locate read T x st lo hi), ∃ p, lo ≤ p ∧ p ≤ hi ∧
      T.sift read (prefixOf x p) = .inr z := by
  have hpos := search_pos (siftAgrees read T x st) _ lo hi rfl
  unfold locate
  revert hpos
  generalize bisect (siftAgrees read T x st) lo hi = f
  intro hpos z hz
  have hst : ∀ p, z ∈ undecidedAt read T x p → T.sift read (prefixOf x p) = .inr z := by
    intro p h
    unfold undecidedAt at h
    rcases hs : T.sift read (prefixOf x p) with l | z' <;> rw [hs] at h <;> simp_all
  cases f with
  | edge p =>
    simp only at hz
    split at hz <;> simp [middleString] at hz
  | triple p =>
    simp only [middleString] at hz
    exact ⟨p, hpos.1, hpos.2, hst p (List.mem_of_mem_take hz)⟩
  | pair p =>
    simp only [middleString] at hz
    rcases List.mem_append.1 (List.mem_of_mem_take hz) with h | h
    · exact ⟨p, hpos.1, by have := hpos.2; omega, hst p h⟩
    · exact ⟨p + 1, by have := hpos.1; omega, hpos.2, hst _ h⟩

theorem located_searches (st : ℕ → List Bool) (lo hi : ℕ) :
    searches (locate read T x st lo hi) = true ∧ startString (locate read T x st lo hi) = [] := by
  have hs : ∀ p zs, startString (.undecided (Ideal.siteAt x st p) zs false) = [] := by
    intro p zs
    unfold Ideal.siteAt
    split <;> rfl
  unfold locate
  split
  · split <;> exact ⟨rfl, rfl⟩
  · exact ⟨rfl, hs _ _⟩
  · exact ⟨rfl, hs _ _⟩

variable (E : Edges α) (k : ℕ)

/-- The undecided middle a probe stops at was read at a prefix between `k` and `|x|`. -/
theorem probe_pt (hk : k ≤ x.toList.length) :
    ∀ z ∈ middleString (probe read T E k x), ∃ p, k ≤ p ∧ p ≤ x.toList.length ∧
      T.sift read (prefixOf x p) = .inr z := by
  unfold probe
  rcases ha : T.sift read (prefixOf x k) with a | z₀
  · simp only
    generalize hcs : x.toList.drop k = cs
    generalize hss : walk E a cs = ss
    have hle : ss.length ≤ cs.length + 1 := hss ▸ length_walk_le E a cs
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    split
    · rename_i c hc
      have hj : k + ss.length - 1 < x.toList.length := by
        by_contra hcon
        rw [List.getElem?_eq_none (by omega)] at hc
        simp at hc
      split
      · simp [middleString]
      · simp [middleString]
      · split_ifs
        · simp [middleString]
        · intro z hz
          obtain ⟨p, h1, h2, h3⟩ := located_pt read T x _ k _ z hz
          exact ⟨p, h1, by omega, h3⟩
    · split
      · simp [middleString]
      · split_ifs
        · simp [middleString]
        · intro z hz
          obtain ⟨p, h1, h2, h3⟩ := located_pt read T x _ k _ z hz
          exact ⟨p, h1, by omega, h3⟩
  · simp [middleString]

/-- The undecided read a probe stops at at its start is its first `k` letters' sift's. -/
theorem probe_start : ∀ z ∈ startString (probe read T E k x), T.sift read (prefixOf x k)
    = .inr z := by
  unfold probe
  rcases ha : T.sift read (prefixOf x k) with a | z₀
  · simp only
    split
    · split
      · simp [startString]
      · simp [startString]
      · split_ifs
        · simp [startString]
        · simp [(located_searches read T x _ k _).2]
    · split
      · simp [startString]
      · split_ifs
        · simp [startString]
        · simp [(located_searches read T x _ k _).2]
  · simp [startString]

/-- A probe the hypothesis disagrees on is long enough to walk, and searches. -/
theorem disagrees_searches (h : Disagrees read ⟨T, E⟩ k x) :
    k ≤ x.toList.length ∧ searches (probeOrAgree read T E k x) = true := by
  obtain ⟨a, q, e, ha, hq, he, hne⟩ := h
  obtain ⟨hlen, hlast⟩ := run_walk E a _ q hq
  by_cases hkx : k ≤ x.toList.length
  · refine ⟨hkx, ?_⟩
    unfold probeOrAgree
    rw [if_neg (by omega)]
    unfold probe
    simp only [ha]
    generalize hcs : x.toList.drop k = cs at hlen hlast
    generalize hss : walk E a cs = ss at hlen hlast
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    have hj : k + ss.length - 1 = x.toList.length := by omega
    rw [hj, List.getElem?_eq_none le_rfl, pre_of_le le_rfl] at *
    simp only [he]
    rw [if_neg]
    · exact (located_searches read T x _ k _).1
    · rw [show x.toList.length - k = cs.length by omega, hlast]
      exact fun h => hne h.symm
  · exfalso
    generalize hcs : x.toList.drop k = cs at hlen hlast
    have hcs0 : cs = [] := by rw [← hcs]; exact List.drop_eq_nil_of_le (by omega)
    subst hcs0
    rw [pre_of_le (by omega), he] at ha
    cases ha
    simp [walk] at hlast
    exact hne hlast.symm

/-! ## The counted strings -/

theorem middleString_length (o : Outcome α) : (middleString o).length ≤ 1 := by
  unfold middleString; split <;> simp

theorem startString_length (o : Outcome α) : (startString o).length ≤ 1 := by
  unfold startString; split <;> simp

theorem searches_of_ptStr {o : Outcome α} (h : middleString o ≠ []) : searches o = true := by
  unfold middleString at h
  split at h
  · rfl
  · exact absurd rfl h

theorem probeOrAgree_pt : ∀ z ∈ middleString (probeOrAgree read T E k x), z ∈ pot k x T
    ∧ read z = .undecided := by
  intro z hz
  unfold probeOrAgree at hz
  split_ifs at hz with hs
  · simp [middleString] at hz
  · obtain ⟨p, h1, h2, h3⟩ := probe_pt read T x E k (by omega) z hz
    exact pot_of_sift read h1 h2 h3

theorem probeOrAgree_start :
    ∀ z ∈ startString (probeOrAgree read T E k x), z ∈ pot k x T ∧ read z = .undecided := by
  intro z hz
  unfold probeOrAgree at hz
  split_ifs at hz with hs
  · simp [startString] at hz
  · exact pot_of_sift read le_rfl (by omega) (probe_start read T x E k z hz)

theorem positionReads_und :
    ∀ e ∈ positionReads read T E k x, ∀ z, e.2.2.2 = some z
        → z ∈ pot k x T ∧ read z = .undecided := by
  intro e he z hz
  unfold positionReads at he
  split_ifs at he
  · simp at he
  · obtain ⟨i, hi, hie⟩ := List.mem_filterMap.1 he
    have hi' := List.mem_filter.1 (List.mem_dedup.1 hi)
    simp only [decide_eq_true_eq] at hi'
    obtain ⟨e', -, rfl⟩ := Option.map_eq_some_iff.1 hie
    simp only at hz
    rcases hs : T.sift read (prefixOf x i) with l | z' <;> rw [hs] at hz <;> simp at hz
    subst hz
    exact pot_of_sift read hi'.2.1.le hi'.2.2 hs

theorem positionReads_length : (positionReads read T E k x).length ≤ x.toList.length - k := by
  classical
  unfold positionReads
  split_ifs
  · simp
  · refine (List.length_filterMap_le _ _).trans ?_
    set l := ((probePositions read T E k x).2.filter fun i => k < i ∧ i ≤ x.toList.length).dedup
    have hnd : l.Nodup := List.nodup_dedup _
    rw [← List.toFinset_card_of_nodup hnd, ← Nat.card_Ioc]
    refine Finset.card_le_card fun i hi => ?_
    have := List.mem_filter.1 (List.mem_dedup.1 (List.mem_toFinset.1 hi))
    simp only [decide_eq_true_eq] at this
    exact Finset.mem_Ioc.2 this.2

variable [DecidableEq α]

theorem undecidedAtEdge_sub (ch : List (List Bool × α × ℕ × Option (FreeMonoid α))) (p : List Bool)
    (c : α) : ∀ z ∈ undecidedAtEdge ch p c, ∃ e ∈ ch, e.2.2.2 = some z := by
  intro z hz
  obtain ⟨e, he, hez⟩ := List.mem_filterMap.1 hz
  exact ⟨e, (List.mem_filter.1 he).1, hez⟩

theorem undecidedAtEdge_length (ch : List (List Bool × α × ℕ × Option (FreeMonoid α)))
    (p : List Bool)
    (c : α) : (undecidedAtEdge ch p c).length ≤ ch.length :=
  (List.length_filterMap_le _ _).trans (List.length_filter_le _ _)

end Random

end OrthoDFA
