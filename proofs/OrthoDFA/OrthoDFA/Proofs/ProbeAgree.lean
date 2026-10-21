import OrthoDFA.Utils.ProbeAgree
import OrthoDFA.Proofs.Probe

/-!
# What a probe that does not disagree promises: the proof

A probe that neither searches nor stops undecided at its start or end sift comes out `agree`, or
undecided with no string and clean (`quiet_cases`). The latter needs a search result that the
search, starting at an agreeing `k` and never recovering, does not return, so the search agreed at
both ends and the walk followed every letter (`searchOutcome_quiet`). The root reading the string
itself puts the string's read at the head of its leaf.
-/

namespace OrthoDFA

open Random

variable {α : Type*} (read : FreeMonoid α → ARU) (T : DTree α) (E : Edges α)

theorem quiet_cases {o : Outcome α}
    (hS : ¬ (match o with
      | .wrongEdge .. | .unlearnedEdge .. | .undecided _ false => True
      | _ => False))
    (hE : endsString o = []) : o = .agree ∨ o = .undecided [] true := by
  cases o with
  | agree => exact Or.inl rfl
  | wrongEdge => exact absurd trivial hS
  | unlearnedEdge => exact absurd trivial hS
  | undecidedStart z => simp [endsString] at hE
  | undecided zs cl =>
    cases cl
    · exact absurd trivial hS
    · cases zs with
      | nil => exact Or.inr rfl
      | cons z zs => simp [endsString] at hE

theorem searchOutcome_quiet (x : FreeMonoid α) (st : ℕ → List Bool)
    (un : Option (α × List Bool)) {k j : ℕ} (hkj : k ≤ j)
    (hk : siftAgrees read T x st k = some true)
    (h : searchOutcome read T x st un j (tripartiteSearch false (siftAgrees read T x st) k j).1
        = .agree
      ∨ searchOutcome read T x st un j (tripartiteSearch false (siftAgrees read T x st) k j).1
        = .undecided [] true) :
    un = none ∧ siftAgrees read T x st j = some true := by
  have hs := tripartiteSearch_ok (recover := false) (agrees := siftAgrees read T x st) hkj
  have hn := tripartiteSearch_noRecovery (agrees := siftAgrees read T x st) (lo := k) (hi := j)
  generalize (tripartiteSearch false (siftAgrees read T x st) k j).1 = r at hs hn h
  cases r with
  | undecidedLo => simp [SearchOK, hk] at hs
  | disagreeLo => simp [SearchOK, hk] at hs
  | undecidedHi =>
    have hne := (stuckAt_ok read T x st hs.2).1
    simp only [searchOutcome, reduceCtorEq, false_or, Outcome.undecided.injEq, and_true] at h
    exact absurd h hne
  | allAgree =>
    rcases un with _ | ⟨c, t⟩
    · exact ⟨rfl, hs.2⟩
    · simp [searchOutcome] at h
  | edge p =>
    simp only [searchOutcome] at h
    split at h <;> simp at h
  | undecidedMid p => simp [searchOutcome] at h
  | pair p => exact absurd rfl (hn p).1
  | triple p => exact absurd rfl (hn p).2

theorem probe_quiet (k : ℕ) (x : FreeMonoid α) (hk : k ≤ x.toList.length)
    (h : probe read T E k x = .agree ∨ probe read T E k x = .undecided [] true) :
    ∃ a e, T.sift read (prefixOf x k) = .inl a ∧ walkEnd E a (x.toList.drop k) = some e
      ∧ T.sift read x = .inl e := by
  unfold probe at h
  rcases ha : T.sift read (prefixOf x k) with a | z
  · simp only [ha] at h
    generalize hcs : x.toList.drop k = cs at h
    generalize hss : walk E a cs = ss at h
    have hpos : 0 < ss.length := hss ▸ length_walk_pos E a cs
    have hle : ss.length ≤ cs.length + 1 := hss ▸ length_walk_le E a cs
    have hcsl : cs.length = x.toList.length - k := by rw [← hcs, List.length_drop]
    have hst0 : ss.getD (k - k) [] = a := by
      rw [Nat.sub_self, ← hss]
      exact walk_getD_zero E a cs
    have hagk : siftAgrees read T x (fun p => ss.getD (p - k) []) k = some true := by
      rw [agrees_inl read T x _ ha, hst0]; simp
    split at h
    · exfalso
      split at h
      · rcases h with h | h <;> simp at h
      · rename_i t _
        exact absurd (searchOutcome_quiet read T x _ _ (by omega) hagk h).1 (by simp)
    · rename_i hnone
      obtain ⟨-, hj⟩ := searchOutcome_quiet read T x _ _ (by omega) hagk h
      have hhi : k + ss.length - 1 = x.toList.length := by
        have : x.toList.length ≤ k + ss.length - 1 := by
          by_contra hlt
          rw [List.getElem?_eq_getElem (by omega)] at hnone
          simp at hnone
        omega
      have hlen : ss.length = cs.length + 1 := by omega
      have hsx := agrees_true read T x _ hj
      rw [hhi, prefixOf_of_le le_rfl] at hsx
      refine ⟨a, ss.getD (x.toList.length - k) [], rfl, ?_, hsx⟩
      apply walk_walkEnd E a _ _ (by rw [hss, hlen])
      rw [hss, List.getLast?_eq_getElem?, List.getD_eq_getElem?_getD]
      rw [show ss.length - 1 = x.toList.length - k by omega]
      rw [List.getElem?_eq_getElem (by omega), Option.getD_some]
  · simp only [ha] at h
    rcases h with h | h <;> simp at h

theorem classifies_of_sift {H : HypothesisDFA α} (hroot : ∃ r a, H.tree = .node 1 r a)
    {x : FreeMonoid α} {s e : List Bool} (hw : walkEnd H.edges s x.toList = some e)
    (hx : H.tree.sift read x = .inl e) :
    read x ≠ .undecided ∧ H.Classifies s x (read x = .accept) := by
  obtain ⟨r, a, hT⟩ := hroot
  rw [hT] at hx
  simp only [DTree.sift, mul_one] at hx
  refine ⟨(fun hu => by rw [hu] at hx; cases hx), e, hw, ?_⟩
  rcases hr : read x with _ | _ | _ <;> rw [hr] at hx
  · rcases hs : a.sift read x with l | z <;> rw [hs] at hx <;> simp at hx
    rw [← hx]; simp
  · rcases hs : r.sift read x with l | z <;> rw [hs] at hx <;> simp at hx
    rw [← hx]; simp
  · cases hx

theorem probeAgreeCorrect : ProbeAgreeCorrect := by
  intro α _ read H k x hk hx
  simp only [Set.mem_union, not_or, searchSet, endsSet, Set.mem_ofPred_eq] at hx
  obtain ⟨hS, hE⟩ := hx
  unfold probeOrAgree at hS hE
  rw [if_neg (by omega)] at hS hE
  have hq := probe_quiet read H.tree H.edges k x hk (quiet_cases hS (by simpa using hE))
  refine ⟨hq, fun hk0 hroot => ?_⟩
  subst hk0
  obtain ⟨a, e, ha, hw, hsx⟩ := hq
  refine ⟨a, by simpa [prefixOf] using ha, classifies_of_sift read hroot (by simpa using hw) hsx⟩

end OrthoDFA
