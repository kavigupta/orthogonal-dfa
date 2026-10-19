import OrthoDFA.Proofs.RandomTree

/-!
# What a probe harvests

Every string a probe harvests, at its start or end sift or at an undecided middle, was read
undecided at a prefix of length at least `k` followed by a midfix: it is in `pot k x T`.
-/

namespace OrthoDFA

namespace Random


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

variable (T : DTree α) (x : FreeMonoid α)

theorem searchOutcome_pt (st : ℕ → List Bool) (un : Option (α × List Bool)) (lo hi : ℕ)
    (hle : lo ≤ hi) :
    ∀ z ∈ middleString (searchOutcome read T x st un hi
      (tripartiteSearch false (siftAgrees read T x st) lo hi).1), ∃ p, lo ≤ p ∧ p ≤ hi ∧
      T.sift read (prefixOf x p) = .inr z := by
  have hs := tripartiteSearch_ok (recover := false) (agrees := siftAgrees read T x st) hle
  generalize (tripartiteSearch false (siftAgrees read T x st) lo hi).1 = r at hs
  intro z hz
  have hst : ∀ p, z ∈ undecidedAt read T x p → T.sift read (prefixOf x p) = .inr z := by
    intro p h
    unfold undecidedAt at h
    rcases hs : T.sift read (prefixOf x p) with l | z' <;> rw [hs] at h <;> simp_all
  cases r with
  | edge p =>
    simp only [searchOutcome] at hz
    split at hz <;> simp [middleString] at hz
  | allAgree =>
    simp only [searchOutcome] at hz
    split at hz <;> simp [middleString] at hz
  | undecidedMid p =>
    simp only [searchOutcome, middleString] at hz
    exact ⟨p, hs.2.1.le, hs.2.2.1.le, hst p (List.mem_of_mem_take hz)⟩
  | pair p | triple p => simp [searchOutcome, middleString] at hz
  | undecidedHi => simp [searchOutcome, middleString] at hz
  | undecidedLo => simp [searchOutcome, middleString] at hz
  | disagreeLo => simp [searchOutcome, middleString] at hz

theorem searchOutcome_ends (st : ℕ → List Bool) (un : Option (α × List Bool)) (hi : ℕ)
    (r : SearchResult) :
    ∀ z ∈ endsString (searchOutcome read T x st un hi r), T.sift read (prefixOf x hi) = .inr z := by
  intro z hz
  cases r with
  | undecidedHi =>
    simp only [searchOutcome, endsString] at hz
    have hz' := List.mem_of_mem_take hz
    unfold undecidedAt at hz'
    rcases hs : T.sift read (prefixOf x hi) with l | z' <;> rw [hs] at hz' <;> simp_all
  | allAgree => simp only [searchOutcome] at hz; split at hz <;> simp [endsString] at hz
  | edge p => simp only [searchOutcome] at hz; split at hz <;> simp [endsString] at hz
  | undecidedMid p => simp [searchOutcome, endsString] at hz
  | pair p | triple p => simp [searchOutcome, endsString] at hz
  | undecidedLo => simp [searchOutcome, endsString] at hz
  | disagreeLo => simp [searchOutcome, endsString] at hz

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
    have hpos : 0 < ss.length := hss ▸ length_walk_pos E a cs
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    split
    · rename_i c hc
      have hj : k + ss.length - 1 < x.toList.length := by
        by_contra hcon
        rw [List.getElem?_eq_none (by omega)] at hc
        simp at hc
      split
      · simp [middleString]
      · intro z hz
        obtain ⟨p, h1, h2, h3⟩ := searchOutcome_pt read T x _ _ k _ (by omega) z hz
        exact ⟨p, h1, by omega, h3⟩
    · intro z hz
      obtain ⟨p, h1, h2, h3⟩ := searchOutcome_pt read T x _ _ k _ (by omega) z hz
      exact ⟨p, h1, by omega, h3⟩
  · simp [middleString]

/-- The undecided read a probe stops at at its start or end is a sift's of a prefix between `k`
and `|x|`. -/
theorem probe_ends (hk : k ≤ x.toList.length) :
    ∀ z ∈ endsString (probe read T E k x), ∃ p, k ≤ p ∧ p ≤ x.toList.length ∧
      T.sift read (prefixOf x p) = .inr z := by
  unfold probe
  rcases ha : T.sift read (prefixOf x k) with a | z₀
  · simp only
    generalize hcs : x.toList.drop k = cs
    generalize hss : walk E a cs = ss
    have hle : ss.length ≤ cs.length + 1 := hss ▸ length_walk_le E a cs
    have hpos : 0 < ss.length := hss ▸ length_walk_pos E a cs
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    split
    · rename_i c hc
      have hj : k + ss.length - 1 < x.toList.length := by
        by_contra hcon
        rw [List.getElem?_eq_none (by omega)] at hc
        simp at hc
      split
      · rename_i z hz
        intro z' hz'
        simp only [endsString, List.take_succ_cons, List.take_zero, List.mem_singleton] at hz'
        subst hz'
        exact ⟨_, by omega, by omega, hz⟩
      · intro z hz
        exact ⟨_, by omega, by omega, searchOutcome_ends read T x _ _ _ _ z hz⟩
    · intro z hz
      exact ⟨_, by omega, by omega, searchOutcome_ends read T x _ _ _ _ z hz⟩
  · intro z hz
    simp only [endsString, List.mem_singleton] at hz
    subst hz
    exact ⟨k, le_rfl, hk, ha⟩

/-! ## The counted strings -/

theorem middleString_length (o : Outcome α) : (middleString o).length ≤ 1 := by
  unfold middleString; split <;> simp

theorem endsString_length (o : Outcome α) : (endsString o).length ≤ 1 := by
  unfold endsString; split <;> simp

theorem searches_of_ptStr {o : Outcome α} (h : middleString o ≠ []) : searches o = true := by
  unfold middleString at h
  split at h
  · rfl
  · exact absurd rfl h

theorem probeOrAgree_pt :
    ∀ z ∈ middleString (probeOrAgree read T E k x), z ∈ pot k x T ∧ read z = .undecided := by
  intro z hz
  unfold probeOrAgree at hz
  split_ifs at hz with hs
  · simp [middleString] at hz
  · obtain ⟨p, h1, h2, h3⟩ := probe_pt read T x E k (by omega) z hz
    exact pot_of_sift read h1 h2 h3

theorem probeOrAgree_ends :
    ∀ z ∈ endsString (probeOrAgree read T E k x), z ∈ pot k x T ∧ read z = .undecided := by
  intro z hz
  unfold probeOrAgree at hz
  split_ifs at hz with hs
  · simp [endsString] at hz
  · obtain ⟨p, h1, h2, h3⟩ := probe_ends read T x E k (by omega) z hz
    exact pot_of_sift read h1 h2 h3

end Random

end OrthoDFA
