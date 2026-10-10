import OrthoDFA.TallyRound
import OrthoDFA.Proofs.StartAtK
import OrthoDFA.Proofs.RoundStrong

/-!
# The tally loop's walks

A record's edge is out of a leaf on the walk and points elsewhere than the leaf its next prefix
sifts to. The positions a probe sifts lie between its start and its end.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

section Walk

variable (cut : FreeMonoid α → Option Bool)

theorem walkCheckBy_inl {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {o : Outcome α} (h : walkCheckBy cut t edges k x = .inl o) : ¬ o.IsSearch := by
  unfold walkCheckBy at h
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

theorem probeBy_search {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {o : Outcome α} (h : probeBy cut t edges k x = o) (ho : o.IsSearch) :
    ∃ ps hi, walkCheckBy cut t edges k x = .inr (ps, hi)
      ∧ bracketAt (agreesAtBy cut t x fun j => ps.getD (j - k) []) ps (hi - k) k hi = o := by
  unfold probeBy at h
  rcases hw : walkCheckBy cut t edges k x with o' | ⟨ps, hi⟩ <;> rw [hw] at h
  · simp only [Sum.elim_inl, id] at h
    subst h
    exact absurd ho (walkCheckBy_inl cut hw)
  · exact ⟨ps, hi, rfl, h⟩

theorem walkCheckBy_inr {t : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {ps : List (List Bool)} {hi : ℕ} (h : walkCheckBy cut t edges k x = .inr (ps, hi)) :
    ∃ p₀, t.sift cut (prefixOf x k) = .inl p₀
      ∧ follow edges p₀ ((x.toList.drop k).take (hi - k)) = .inl ps ∧ k < hi
      ∧ hi ≤ x.toList.length
      ∧ agreesAtBy cut t x (fun j => ps.getD (j - k) []) hi = some false := by
  unfold walkCheckBy at h
  rcases hw : kWalkBy cut t edges k x with _ | ⟨s, c, j⟩ | ps' <;> rw [hw] at h
  · simp at h
  · unfold kWalkBy at hw
    rcases hk : t.sift cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
    swap
    · simp at hw
    simp only [] at hw
    rcases hf : follow edges p₀ (x.toList.drop k) with ps'' | ⟨s', c', i⟩ <;> rw [hf] at hw
    · simp at hw
    simp only [KWalk.edge.injEq] at hw
    obtain ⟨rfl, rfl, rfl⟩ := hw
    obtain ⟨hi', hc, hn, qs, hqs, hlast⟩ := follow_inr _ _ _ _ _ hf
    simp only [] at h
    rcases h1 : t.sift cut (prefixOf x (k + i + 1)) with _ | _ <;> rw [h1] at h
    swap
    · simp at h
    simp only [] at h
    rcases h2 : t.sift cut (prefixOf x (k + i)) with p | _ <;> rw [h2] at h
    swap
    · simp at h
    simp only [] at h
    split_ifs at h with hps
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    have hwt : walkToBy cut t edges k x (k + i) = qs := by
      simp [walkToBy, hk, show k + i - k = i by omega, hqs]
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
      simp only [agreesAtBy, h2, Sum.elim_inl, hget, Option.some.injEq, decide_eq_false_iff_not]
      exact hps
  · simp only [] at h
    rcases hs : t.sift cut x with a | b <;> rw [hs] at h
    swap
    · simp at h
    simp only [] at h
    split_ifs at h with ha
    simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    unfold kWalkBy at hw
    rcases hk : t.sift cut (prefixOf x k) with p₀ | b <;> rw [hk] at hw
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
    simp only [agreesAtBy, this, hs, Sum.elim_inl, Option.some.injEq, decide_eq_false_iff_not]
    intro he
    exact ha (by rw [hlast, he])

theorem follow_mem_paths {T : DTree α} {edges : Edges α} (he : EdgesInto T edges) :
    ∀ (cs : List α) (p : List Bool) (ps : List (List Bool)), p ∈ T.paths →
      follow edges p cs = .inl ps → ∀ q ∈ ps, q ∈ T.paths
  | [], p, ps, hp, h, q, hq => by
    simp only [follow, Sum.inl.injEq] at h
    subst h
    simp_all
  | c :: cs, p, ps, hp, h, q, hq => by
    simp only [follow] at h
    rcases hpc : edges p c with _ | ⟨p', y⟩ <;> rw [hpc] at h
    · simp at h
    · simp only [] at h
      rcases hf : follow edges p' cs with ps' | r <;> rw [hf] at h
      · simp only [Sum.inl.injEq] at h
        subst h
        rcases List.mem_cons.1 hq with rfl | hq
        · exact hp
        · exact follow_mem_paths he cs p' ps' (he _ _ _ _ hpc) hf q hq
      · simp at h

/-- What a record says about its walk: the edge it is at, out of a leaf on the walk, points at the
walk's next leaf, which is not the leaf the next prefix sifts to. -/
theorem recordBy_spec {T : DTree α} {edges : Edges α} {k : ℕ} {x : FreeMonoid α}
    {p : List Bool} {c : α} {t : List Bool} {sp : FreeMonoid α}
    (h : recordBy cut k (T, edges) x = some ((p, c, t), sp)) :
    ∃ p₀ ps, T.sift cut (prefixOf x k) = .inl p₀
      ∧ follow edges p₀ ((x.toList.drop k).take (ps.length - 1)) = .inl ps ∧ p ∈ ps
      ∧ T.sift cut (sp * FreeMonoid.of c) = .inl t ∧ (edges p c).map Prod.fst ≠ some t := by
  unfold recordBy at h
  simp only [] at h
  rcases ho : probeBy cut T edges k x with _ | _ | _ | _ | ⟨ps, fd⟩ | _ | _ <;> rw [ho] at h <;>
    simp only [reduceCtorEq] at h
  obtain ⟨ps₀, hi, hw, hb⟩ := probeBy_search cut ho trivial
  obtain ⟨p₀, hk, hf, hkh, hhx, hag⟩ := walkCheckBy_inr cut hw
  obtain ⟨hlen, hhead, hstep⟩ := follow_inl _ _ _ hf
  simp only [List.length_take, List.length_drop] at hlen
  have hlo : agreesAtBy cut T x (fun j => ps₀.getD (j - k) []) k = some true := by
    simp [agreesAtBy, hk, ← hhead, List.getD_eq_getElem?_getD]
  obtain ⟨rfl, hfd1, hfd2, -, hfd⟩ := bracketAt_edge _ ps₀ (hi - k) k hi ps fd hkh le_rfl hlo hag hb
  rcases hc : x.toList[fd - 1]? with _ | c₁ <;> rw [hc] at h
  · simp at h
  rcases ht : T.sift cut (prefixOf x fd) with t' | b <;> rw [ht] at h
  swap
  · simp at h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨⟨hp, hcc, htt⟩, hsp⟩ := h
  subst hp htt hsp hcc
  have hi_lt : fd - 1 - k < ((x.toList.drop k).take (hi - k)).length := by
    simp only [List.length_take, List.length_drop]; omega
  obtain ⟨y, hy⟩ := hstep (fd - 1 - k) hi_lt
  have hcs : ((x.toList.drop k).take (hi - k))[fd - 1 - k] = c₁ := by
    simp only [List.getElem_take, List.getElem_drop]
    have : k + (fd - 1 - k) = fd - 1 := by omega
    rw [List.getElem?_eq_some_iff] at hc
    obtain ⟨hlt, hc⟩ := hc
    simp only [this, hc]
  rw [hcs] at hy
  have hidx : fd - 1 - k < ps.length := by omega
  have hpi : ps.getD (fd - 1 - k) [] = ps[fd - 1 - k] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hidx]; rfl
  refine ⟨p₀, ps, hk, by rw [show ps.length - 1 = hi - k by omega]; exact hf,
    hpi ▸ List.getElem_mem hidx, ?_, ?_⟩
  · have : prefixOf x (fd - 1) * FreeMonoid.of c₁ = prefixOf x fd := by
      apply FreeMonoid.toList.injective
      rw [List.getElem?_eq_some_iff] at hc
      obtain ⟨hlt, hc⟩ := hc
      simp only [prefixOf, FreeMonoid.toList_mul, FreeMonoid.toList_ofList, FreeMonoid.toList_of]
      rw [show fd = fd - 1 + 1 by omega, List.take_add_one, List.getElem?_eq_getElem hlt, hc]
      simp
    rw [this, ht]
  · rw [hy]
    simp only [Option.map_some, ne_eq, Option.some.injEq]
    have h2 : fd - 1 - k + 1 = fd - k := by omega
    rw [h2]
    simp only [agreesAtBy, ht, Sum.elim_inl, Option.some.injEq, decide_eq_false_iff_not] at hfd
    exact fun he => hfd he.symm

end Walk

end OrthoDFA
