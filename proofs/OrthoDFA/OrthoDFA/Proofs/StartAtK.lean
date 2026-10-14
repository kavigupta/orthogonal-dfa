import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.PassReads
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.BinomLaw

/-!
# Walks and bisections

What `follow` returns, and where `bracketAt` ends: at an edge between an agreeing and a
disagreeing position, or at a search outcome.
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

variable (R : CutReads α)

omit [Fintype α] [DecidableEq α] in
theorem follow_inl {edges : EdgeTable α} :
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
theorem follow_inr {edges : EdgeTable α} :
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
    simp only [bracketAt] at h
    split_ifs at h with h1
    · rcases hm : agrees ((lo + hi) / 2) with _ | _ | _ <;> simp only [hm, reduceCtorEq] at h
      · have := bracketAt_edge agrees ps fuel lo _ ps' j (by omega) (by omega) hlo hm h
        exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
      · have := bracketAt_edge agrees ps fuel _ hi ps' j (by omega) (by omega) hm hhi h
        exact ⟨this.1, by omega, this.2.2⟩
    · obtain ⟨rfl, rfl⟩ := WalkOutcome.edge.inj h
      obtain rfl : hi = lo + 1 := by omega
      exact ⟨rfl, by omega, le_rfl, by simpa using hlo, hhi⟩

omit [Fintype α] [DecidableEq α] in
/-- Level 5's search is main's bisection without recovery. -/
theorem bracketAt_eq_bisect (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi, hi - lo ≤ fuel →
      bracketAt (α := α) agrees ps fuel lo hi
        = bisectOutcome ps (bisectBoundary false agrees lo hi).1
  | 0, lo, hi, h => by
    rw [bisectBoundary, dif_pos (by omega)]
    rfl
  | fuel + 1, lo, hi, h => by
    rw [bisectBoundary]
    simp only [bracketAt]
    by_cases h1 : lo + 1 < hi
    · rw [if_pos h1, dif_neg (by omega)]
      rcases hm : agrees ((lo + hi) / 2) with _ | _ | _
      · rfl
      · exact bracketAt_eq_bisect agrees ps fuel lo _ (by omega)
      · exact bracketAt_eq_bisect agrees ps fuel _ hi (by omega)
    · rw [if_neg h1, dif_pos (by omega)]
      rfl

/-- The outcomes a search ends at. -/
def WalkOutcome.IsSearch : WalkOutcome α → Prop
  | .undecidedMid _ | .edge _ _ => True
  | _ => False

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_isSearch (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi, (bracketAt (α := α) agrees ps fuel lo hi).IsSearch
  | 0, _, _ => trivial
  | fuel + 1, lo, hi => by
    simp only [bracketAt]
    split
    · rcases agrees ((lo + hi) / 2) with _ | _ | _
      · trivial
      · exact bracketAt_isSearch agrees ps fuel _ _
      · exact bracketAt_isSearch agrees ps fuel _ _
    · trivial

end Learned

end OrthoDFA
