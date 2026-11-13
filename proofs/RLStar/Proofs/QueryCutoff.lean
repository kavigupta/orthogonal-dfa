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

theorem card_reads_cutoff_le (N : ℕ) (p : Query Q R β) (o : Q → R) :
    ((p.cutoff N).reads o).card ≤ N := by
  sorry -- closes by: induction on p, generalizing S ⊇ reads so far with S.card ≤ N

theorem eval_cutoff {N : ℕ} {p : Query Q R β} {o : Q → R} (h : (p.reads o).card ≤ N) :
    (p.cutoff N).eval o = some (p.eval o) ∧ (p.cutoff N).reads o = p.reads o := by
  sorry -- closes by: induction on p, generalizing S ⊆ reads asked so far

theorem eval_cutoff_eq_some {N : ℕ} {p : Query Q R β} {o : Q → R} {b : β}
    (h : (p.cutoff N).eval o = some b) :
    b = p.eval o ∧ (p.cutoff N).reads o = p.reads o := by
  sorry -- closes by: induction on p; a cut returns none

end Query

end OrthoDFA
