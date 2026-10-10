import OrthoDFA.StartAtK
import OrthoDFA.Proofs.Hoeffding
import OrthoDFA.Proofs.Replay
import OrthoDFA.Proofs.PassReads
import OrthoDFA.Proofs.Adaptive
import OrthoDFA.Proofs.BinomLaw

/-!
# `RoundAtK`

A decision on a batch against a fixed hypothesis is an exact binomial test at the look where the
gate stops, or a Hoeffding tail at the batch's end.  An edge reaches the split test or stops at a
string the cut cannot place, because the pass keeps its edges learned: every edge's witness sits
at its leaf and its extension at its target.
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
theorem follow_inl {edges : Edges α} :
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
theorem follow_inr {edges : Edges α} :
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
    have hag : ∀ p, (if p = lo then some true else if p = hi then some false else agrees p)
        = agrees p := fun p => by split_ifs <;> simp_all
    simp only [bracketAt, hag] at h
    split_ifs at h with h1
    · rcases hm : agrees ((lo + hi) / 2) with _ | _ | _ <;> simp only [hm] at h
      · rcases hl : agrees ((lo + hi) / 2 - 1) with _ | _ | _ <;>
          rcases hr : agrees ((lo + hi) / 2 + 1) with _ | _ | _ <;>
          simp only [hl, hr, reduceCtorEq] at h
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees ps fuel lo _ ps' j this (by omega) hlo hl h
          exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
        · have : lo < (lo + hi) / 2 - 1 := by
            rcases Nat.lt_or_ge lo ((lo + hi) / 2 - 1) with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 - 1 = lo by omega, hlo] at hl; simp at hl
          have := bracketAt_edge agrees ps fuel lo _ ps' j this (by omega) hlo hl h
          exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
        · have : (lo + hi) / 2 + 1 < hi := by
            rcases Nat.lt_or_ge ((lo + hi) / 2 + 1) hi with h' | h'
            · exact h'
            · rw [show (lo + hi) / 2 + 1 = hi by omega, hhi] at hr; simp at hr
          have := bracketAt_edge agrees ps fuel _ hi ps' j this (by omega) hr hhi h
          exact ⟨this.1, by omega, this.2.2⟩
      · have := bracketAt_edge agrees ps fuel lo _ ps' j (by omega) (by omega) hlo hm h
        exact ⟨this.1, this.2.1, by omega, this.2.2.2⟩
      · have := bracketAt_edge agrees ps fuel _ hi ps' j (by omega) (by omega) hm hhi h
        exact ⟨this.1, by omega, this.2.2⟩
    · obtain ⟨rfl, rfl⟩ := Outcome.edge.inj h
      obtain rfl : hi = lo + 1 := by omega
      exact ⟨rfl, by omega, le_rfl, by simpa using hlo, hhi⟩

/-- The outcomes a search ends at. -/
def Outcome.IsSearch : Outcome α → Prop
  | .pair _ | .edge _ _ | .triple _ => True
  | _ => False

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_isSearch (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi, (bracketAt (α := α) agrees ps fuel lo hi).IsSearch
  | 0, _, _ => trivial
  | fuel + 1, lo, hi => by
    simp only [bracketAt]
    split
    · split
      · exact bracketAt_isSearch agrees ps fuel _ _
      · exact bracketAt_isSearch agrees ps fuel _ _
      · split
        all_goals first | trivial | exact bracketAt_isSearch agrees ps fuel _ _
    · trivial

end Learned

end OrthoDFA
