import RLStar.Proofs.Query

/-!
# A program cut off at its `N`-th new question

`p.cutoff N` runs `p` but returns `none` rather than ask an `N + 1`-th distinct question. It asks
at most `N` strings (`card_reads_cutoff_le`), and where `p` asks at most `N` it is `p`
(`eval_cutoff`).
-/

namespace OrthoDFA

namespace Query

variable {Q R β : Type*} [DecidableEq Q]

/-- `p`, having asked the questions `S`, cut off at its `N + 1`-th distinct question. -/
def cutoffFrom (N : ℕ) : Finset Q → Query Q R β → Query Q R (Option β)
  | _, pure b => pure (some b)
  | S, ask q k =>
    if q ∈ S ∨ S.card < N then ask q fun r => cutoffFrom N (insert q S) (k r) else pure none

/-- `p` cut off at its `N + 1`-th distinct question. -/
def cutoff (N : ℕ) (p : Query Q R β) : Query Q R (Option β) := cutoffFrom N ∅ p

private theorem card_union_reads_cutoffFrom_le (N : ℕ) (o : Q → R) (p : Query Q R β) :
    ∀ S : Finset Q, S.card ≤ N → (S ∪ (cutoffFrom N S p).reads o).card ≤ N := by
  induction p with
  | pure b => intro S hS; simpa [cutoffFrom] using hS
  | ask q k ih =>
    intro S hS
    rw [cutoffFrom]
    split_ifs with hq
    · rw [reads_ask, Finset.union_insert, ← Finset.insert_union]
      refine ih (o q) _ ?_
      rcases hq with hq | hq
      · rwa [Finset.insert_eq_of_mem hq]
      · exact (Finset.card_insert_le q S).trans hq
    · simpa using hS

theorem card_reads_cutoff_le (N : ℕ) (p : Query Q R β) (o : Q → R) :
    ((p.cutoff N).reads o).card ≤ N := by
  simpa [cutoff] using card_union_reads_cutoffFrom_le N o p ∅ (by simp)

private theorem eval_cutoffFrom (N : ℕ) (o : Q → R) (p : Query Q R β) :
    ∀ S : Finset Q, (S ∪ p.reads o).card ≤ N →
      (cutoffFrom N S p).eval o = some (p.eval o) ∧ (cutoffFrom N S p).reads o = p.reads o := by
  induction p with
  | pure b => intro S _; exact ⟨rfl, rfl⟩
  | ask q k ih =>
    intro S h
    rw [reads_ask, Finset.union_insert] at h
    have hq : q ∈ S ∨ S.card < N := by
      by_cases hqS : q ∈ S
      · exact Or.inl hqS
      · refine Or.inr (lt_of_lt_of_le (S.card_lt_card (Finset.ssubset_insert hqS)) ?_)
        exact (Finset.card_le_card (Finset.insert_subset_insert q Finset.subset_union_left)).trans h
    rw [cutoffFrom, if_pos hq, eval_ask, reads_ask, eval_ask, reads_ask]
    obtain ⟨h1, h2⟩ := ih (o q) (insert q S) (by rwa [Finset.insert_union])
    exact ⟨h1, by rw [h2]⟩

theorem eval_cutoff {N : ℕ} {p : Query Q R β} {o : Q → R} (h : (p.reads o).card ≤ N) :
    (p.cutoff N).eval o = some (p.eval o) ∧ (p.cutoff N).reads o = p.reads o :=
  eval_cutoffFrom N o p ∅ (by simpa using h)

private theorem eval_cutoffFrom_eq_some (N : ℕ) (o : Q → R) (p : Query Q R β) :
    ∀ (S : Finset Q) (b : β), (cutoffFrom N S p).eval o = some b →
      b = p.eval o ∧ (cutoffFrom N S p).reads o = p.reads o := by
  induction p with
  | pure b' =>
    intro S b h
    simp only [cutoffFrom, eval_pure, Option.some.injEq] at h
    exact ⟨h.symm, rfl⟩
  | ask q k ih =>
    intro S b h
    rw [cutoffFrom] at h ⊢
    split_ifs at h ⊢ with hq
    · rw [eval_ask] at h
      obtain ⟨h1, h2⟩ := ih (o q) (insert q S) b h
      exact ⟨h1, by rw [reads_ask, reads_ask, h2]⟩
    · simp at h

theorem eval_cutoff_eq_some {N : ℕ} {p : Query Q R β} {o : Q → R} {b : β}
    (h : (p.cutoff N).eval o = some b) :
    b = p.eval o ∧ (p.cutoff N).reads o = p.reads o :=
  eval_cutoffFrom_eq_some N o p ∅ b h

end Query

end OrthoDFA
