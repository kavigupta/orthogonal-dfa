import OrthoDFA.Proofs.TripleBound
import OrthoDFA.Proofs.Visits

/-!
# The harvest classes

Each class of `QualityHolds` as a computation and harvest meeting `HarvestSpec`: the ends' sifts
with the root's read untagged, the reads at an unlearned edge, and the search's middles.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

section Ends

variable (cut : FreeMonoid α → Option Bool)

end Ends

section Blocked

variable (cut : FreeMonoid α → Option Bool)

end Blocked

section Probe

omit [Fintype α] [DecidableEq α] in
theorem bracketAt_pair_gt (agrees : ℕ → Option Bool) (ps : List (List Bool)) :
    ∀ fuel lo hi j, bracketAt (α := α) agrees ps fuel lo hi = .pair j → lo < j
  | 0, _, _, _, h => by simp [bracketAt] at h
  | fuel + 1, lo, hi, j, h => by
    simp only [bracketAt] at h
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh] at h; simp at h
    rw [if_pos hlh] at h
    have hl0 : (lo + hi) / 2 - 1 = lo → (if (lo + hi) / 2 - 1 = lo then some true
        else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1))
          = some true := fun e => if_pos e
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v at h
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l at h hl0
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r at h
    rcases v with _ | _ | _
    · rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;> simp only [reduceCtorEq] at h
      all_goals first
        | (have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega)
        | (obtain rfl := Outcome.pair.inj h; omega)
        | (obtain rfl := Outcome.pair.inj h
           have hne : (lo + hi) / 2 - 1 ≠ lo := fun he => by simpa using hl0 he
           omega)
    · have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega
    · have := bracketAt_pair_gt agrees ps fuel _ _ j h; omega

variable (cut : FreeMonoid α → Option Bool)

/-- A search ending at a pair reads the middle beside it first, tagged, after reads that are
tagged or decided. -/
theorem qBracket_pair (t : DTree α) (x : FreeMonoid α) (walkAt : ℕ → List Bool)
    (ps : List (List Bool)) :
    ∀ fuel lo hi j, (qBracket (qAgrees t x walkAt) ps fuel lo hi).run cut = .pair j →
      ∃ b, (t.sift cut (prefixOf x j) = .inr b ∨ t.sift cut (prefixOf x (j + 1)) = .inr b)
        ∧ ∃ pre blk post, (qBracket (qAgrees t x walkAt) ps fuel lo hi).trace cut
            = pre ++ blk ++ post
          ∧ (∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) ∧ (∀ e ∈ blk, e.2 = true)
          ∧ (b, true) ∈ blk
  | 0, _, _, _, h => by simp [qBracket, Qry.run] at h
  | fuel + 1, lo, hi, j, h => by
    by_cases hlh : lo + 1 < hi
    swap
    · rw [qBracket, if_neg hlh] at h; simp [Qry.run] at h
    rw [qBracket_unfold t x walkAt ps fuel lo hi hlh] at h ⊢
    rw [Qry.run_bind] at h
    simp only [Qry.trace_bind]
    have hmid := qGuard_tag cut t x walkAt lo hi ((lo + hi) / 2) true
    have hdm := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2) true
    -- a recursive call, after reads tagged or decided
    have hrec : ∀ (T : List (FreeMonoid α × Bool)) lo' hi',
        (∀ e ∈ T, e.2 = false → cut e.1 ≠ none) →
        (qBracket (qAgrees t x walkAt) ps fuel lo' hi').run cut = .pair j →
        ∃ b, (t.sift cut (prefixOf x j) = .inr b ∨ t.sift cut (prefixOf x (j + 1)) = .inr b)
          ∧ ∃ pre blk post, T ++ (qBracket (qAgrees t x walkAt) ps fuel lo' hi').trace cut
              = pre ++ blk ++ post
            ∧ (∀ e ∈ pre, e.2 = false → cut e.1 ≠ none) ∧ (∀ e ∈ blk, e.2 = true)
            ∧ (b, true) ∈ blk := by
      intro T lo' hi' hT hr
      obtain ⟨b, hb, pre, blk, post, htr, hpre, hblk, hmem⟩ := qBracket_pair t x walkAt ps fuel
        lo' hi' j hr
      refine ⟨b, hb, T ++ pre, blk, post, by rw [htr]; simp, fun e he hg => ?_, hblk, hmem⟩
      rcases List.mem_append.1 he with he | he
      · exact hT e he hg
      · exact hpre e he hg
    rcases hv : (qGuard t x walkAt lo hi ((lo + hi) / 2) true).run cut with _ | v
    · simp only [hv] at h ⊢
      obtain ⟨b, hb, hbm⟩ := qGuard_undecided cut t x walkAt lo hi ((lo + hi) / 2) true hv
      rw [Qry.run_bind] at h
      simp only [Qry.trace_bind]
      have hl := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2 - 1) false
      rcases hlv : (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).run cut with _ | lv
      · simp only [hlv, Qry.run, Outcome.pair.injEq] at h ⊢
        subst h
        refine ⟨b, .inr (by rw [show (lo + hi) / 2 - 1 + 1 = (lo + hi) / 2 by omega]; exact hb),
          [], (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut,
          (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut, by simp [Qry.trace],
          by simp, hmid, hbm⟩
      simp only [hlv] at h ⊢
      rw [Qry.run_bind] at h
      simp only [Qry.trace_bind]
      have hr := qGuard_decided cut t x walkAt lo hi ((lo + hi) / 2 + 1) false
      rcases hrv : (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).run cut with _ | rv
      · have hj : j = (lo + hi) / 2 := by
          rcases lv <;> simpa [hrv, Qry.run] using h.symm
        subst hj
        refine ⟨b, .inl hb, [], (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut,
          (qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut
            ++ (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).trace cut,
          by rcases lv <;> simp [Qry.trace], by simp, hmid, hbm⟩
      have hT : ∀ e ∈ (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut
          ++ ((qGuard t x walkAt lo hi ((lo + hi) / 2 - 1) false).trace cut
            ++ (qGuard t x walkAt lo hi ((lo + hi) / 2 + 1) false).trace cut),
          e.2 = false → cut e.1 ≠ none := by
        intro e he hg
        rcases List.mem_append.1 he with he | he
        · rw [hmid e he] at hg; exact absurd hg (by simp)
        rcases List.mem_append.1 he with he | he
        · exact hl hlv e he
        · exact hr hrv e he
      rcases lv <;> rcases rv <;> simp only [hrv] at h ⊢
      · obtain ⟨b', hb', pre, blk, post, htr, hpre, hblk, hmem⟩ := hrec _ _ _ hT h
        exact ⟨b', hb', pre, blk, post, by rw [← htr]; simp, hpre, hblk, hmem⟩
      · obtain ⟨b', hb', pre, blk, post, htr, hpre, hblk, hmem⟩ := hrec _ _ _ hT h
        exact ⟨b', hb', pre, blk, post, by rw [← htr]; simp, hpre, hblk, hmem⟩
      · simp [Qry.run] at h
      · obtain ⟨b', hb', pre, blk, post, htr, hpre, hblk, hmem⟩ := hrec _ _ _ hT h
        exact ⟨b', hb', pre, blk, post, by rw [← htr]; simp, hpre, hblk, hmem⟩
    · have hT : ∀ e ∈ (qGuard t x walkAt lo hi ((lo + hi) / 2) true).trace cut,
          e.2 = false → cut e.1 ≠ none := fun e he _ => hdm hv e he
      cases v <;> simp only [hv] at h ⊢ <;> exact hrec _ _ _ hT h

/-- Where a probe's search ends: a triple's or a pair's index. -/
def Outcome.at : Outcome α → Option ℕ
  | .triple j => some j
  | .pair j => some j
  | _ => none

end Probe

end OrthoDFA
