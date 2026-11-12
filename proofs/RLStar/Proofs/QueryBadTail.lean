import RLStar.LowLevelClaims.QueryBadTail
import RLStar.Proofs.RaceTail
import RLStar.Proofs.Query

/-!
# Bad reads among the strings a program asks: the proof

`RaceTailBound` with one round per distinct question: round `j` queries the `j`-th distinct string
`p` asks (`nthAsk`), which the answers to the earlier distinct strings fix
(`newAsks_getElem?_congr`), and each round costs `u`.
-/

namespace OrthoDFA

namespace Query

variable {Z : Type*} {R : Type*} {β : Type*} [DecidableEq Z]

/-- The questions `p` asks against `o` outside `s`, each where first asked. -/
def newAsks (o : Z → R) : Finset Z → Query Z R β → List Z
  | _, pure _ => []
  | s, ask q k => if q ∈ s then newAsks o s (k (o q)) else q :: newAsks o (insert q s) (k (o q))

theorem newAsks_spec (o : Z → R) (p : Query Z R β) : ∀ s : Finset Z,
    (newAsks o s p).Nodup ∧ (newAsks o s p).toFinset = p.reads o \ s := by
  induction p with
  | pure b => intro s; simp [newAsks]
  | ask q k ih =>
    intro s
    by_cases hq : q ∈ s
    · simp only [newAsks, if_pos hq, reads_ask]
      exact ⟨(ih (o q) s).1, by rw [(ih (o q) s).2, Finset.insert_sdiff_of_mem _ hq]⟩
    · simp only [newAsks, if_neg hq, reads_ask]
      obtain ⟨h1, h2⟩ := ih (o q) (insert q s)
      refine ⟨List.nodup_cons.2 ⟨fun h => ?_, h1⟩, ?_⟩
      · have := List.mem_toFinset.2 h
        rw [h2] at this
        simp at this
      · rw [List.toFinset_cons, h2]
        ext z
        by_cases hz : z = q <;> simp [hz, hq]

theorem newAsks_getElem?_congr {o o' : Z → R} (p : Query Z R β) : ∀ (s : Finset Z) (j : ℕ),
    (∀ z ∈ s, o' z = o z) → (∀ i < j, ∀ z ∈ (newAsks o s p)[i]?, o' z = o z) →
    (newAsks o' s p)[j]? = (newAsks o s p)[j]? := by
  induction p with
  | pure b => intro s j _ _; rfl
  | ask q k ih =>
    intro s j hs h
    by_cases hq : q ∈ s
    · simp only [newAsks, if_pos hq] at h ⊢
      rw [hs q hq]
      exact ih (o q) s j hs h
    · simp only [newAsks, if_neg hq] at h ⊢
      cases j with
      | zero => rfl
      | succ j =>
        have hq' : o' q = o q := h 0 (Nat.succ_pos j) q (by simp)
        rw [hq', List.getElem?_cons_succ, List.getElem?_cons_succ]
        refine ih (o q) (insert q s) j ?_ fun i hi z hz =>
          h (i + 1) (by omega) z (by simpa using hz)
        intro z hz
        rcases Finset.mem_insert.1 hz with rfl | hz
        exacts [hq', hs z hz]

/-- The `j`-th distinct question `p` asks against `r`, if it asks that many. -/
def nthAsk (p : Query Z R β) (j : ℕ) (r : Z → R) : Finset Z :=
  ((newAsks r ∅ p)[j]?).toFinset

theorem mem_nthAsk {p : Query Z R β} {j : ℕ} {r : Z → R} {z : Z} :
    z ∈ nthAsk p j r ↔ (newAsks r ∅ p)[j]? = some z := Option.mem_toFinset

theorem card_nthAsk_le (p : Query Z R β) (j : ℕ) (r : Z → R) : (nthAsk p j r).card ≤ 1 := by
  unfold nthAsk
  rw [Option.card_toFinset]
  cases (newAsks r ∅ p)[j]? <;> simp

end Query

open MeasureTheory ProbabilityTheory RaceTail Query

theorem RaceTail.mem_newQ {Z R : Type*} {Q : ℕ → (Z → R) → Finset Z} {j : ℕ} {r : Z → R} {z : Z} :
    z ∈ newQ Q j r ↔ z ∈ Q j r ∧ ∀ i < j, z ∉ Q i r := by
  classical
  exact Finset.mem_filter

theorem queryBadTail_holds : QueryBadTail := by
  intro Z Ω R β _ _ _ _ _ _ μ _ read B u p N f t t' hmeas hind hu hB hN ht ht' htw
  classical
  set Q : ℕ → (Z → R) → Finset Z := nthAsk p with hQ
  have hnew : ∀ j r, (newQ Q j r).card ≤ 1 := fun j r =>
    (Finset.card_le_card fun _ hz => (mem_newQ.1 hz).1).trans (card_nthAsk_le p j r)
  have hloc : ∀ j r r', badCount Q B j r < f → (∀ i ≤ j, ∀ z ∈ Q i r, r z = r' z) →
      Q j r = Q j r' := by
    intro j r r' _ h
    ext z
    simp only [hQ, mem_nthAsk]
    rw [newAsks_getElem?_congr p ∅ j (by simp) fun i hi z hz =>
      (h i hi.le z (mem_nthAsk.2 hz)).symm]
  have hb : ∀ r, ∑ j ∈ (Finset.range N).filter (fun j => badCount Q B j r < f),
      ((newQ Q j r).card : ℝ) ≤ N := by
    intro r
    calc ∑ j ∈ (Finset.range N).filter (fun j => badCount Q B j r < f), ((newQ Q j r).card : ℝ)
        ≤ ∑ j ∈ (Finset.range N).filter (fun j => badCount Q B j r < f), (1 : ℝ) :=
          Finset.sum_le_sum fun j _ => by exact_mod_cast hnew j r
      _ ≤ N := by
          rw [Finset.sum_const, nsmul_eq_mul, mul_one]
          exact_mod_cast (Finset.card_filter_le _ _).trans (Finset.card_range N).le
  have hcount : ∀ r : Z → R,
      ((p.reads r).filter fun z => r z ∈ B z).card ≤ badCount Q B N r := by
    intro r
    obtain ⟨hnd, hfs⟩ := newAsks_spec r p ∅
    rw [Finset.sdiff_empty] at hfs
    have hmem : ∀ z ∈ p.reads r, z ∈ newAsks r ∅ p := fun z hz =>
      List.mem_toFinset.1 (hfs ▸ hz)
    have hlen : (newAsks r ∅ p).length ≤ N := by
      rw [← List.toFinset_card_of_nodup hnd, hfs]
      exact hN r
    unfold badCount
    refine Finset.card_le_card_of_injOn (fun z => (newAsks r ∅ p).idxOf z) ?_ ?_
    · intro z hz
      rw [Finset.mem_coe, Finset.mem_filter] at hz
      have hzL := hmem z hz.1
      refine Finset.mem_filter.2 ⟨Finset.mem_range.2
        ((List.idxOf_lt_length_of_mem hzL).trans_le hlen), z, mem_newQ.2 ⟨?_, ?_⟩, hz.2⟩
      · exact mem_nthAsk.2 (List.getElem?_idxOf hzL)
      · intro i (hi : i < (newAsks r ∅ p).idxOf z) hzi
        obtain ⟨hil, hiz⟩ := List.getElem?_eq_some_iff.1 (mem_nthAsk.1 hzi)
        have := hnd.idxOf_getElem i hil
        rw [hiz] at this
        omega
    · intro z₁ hz₁ z₂ _ h
      exact (List.idxOf_inj (hmem z₁ (Finset.mem_filter.1 hz₁).1)).1 h
  have hrt := raceTailBound_holds μ read B (fun _ => ∅) u 0 Q f N N 0 u t t' hmeas hind hu le_rfl
    (fun z => by rw [zero_mul, add_zero]; exact hB z) hloc hb (fun r => by simp)
    (fun r j _ => by
      rw [zero_mul, add_zero]
      exact mul_le_of_le_one_right hu (by exact_mod_cast hnew j r))
    hu ht ht' htw
  calc μ {ω | f ≤ ((p.reads fun z => read z ω).filter fun z => read z ω ∈ B z).card}
      ≤ μ {ω | f ≤ badCount Q B N (read · ω)} := measure_mono fun ω hω => le_trans hω (hcount _)
    _ ≤ _ := hrt
    _ = _ := by rw [mul_zero, add_zero]

end OrthoDFA
