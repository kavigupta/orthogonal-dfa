import OrthoDFA.Proofs.TripleBound
import OrthoDFA.Trichotomy

/-!
# How many middles a search visits

A search halves its bracket at each middle it visits, so it visits at most `⌈log₂(hi − lo)⌉`.
-/

namespace OrthoDFA

variable {α : Type*} [Fintype α] [DecidableEq α]

omit [Fintype α] [DecidableEq α] in
theorem visited_le_clog (agrees : ℕ → Option Bool) :
    ∀ fuel lo hi, visited agrees fuel lo hi ≤ Nat.clog 2 (hi - lo)
  | 0, _, _ => Nat.zero_le _
  | fuel + 1, lo, hi => by
    simp only [visited]
    by_cases hlh : lo + 1 < hi
    swap
    · rw [if_neg hlh]; omega
    rw [if_pos hlh]
    have hc : Nat.clog 2 (hi - lo) = Nat.clog 2 ((hi - lo + 1) / 2) + 1 := by
      rw [Nat.clog_of_two_le (by norm_num) (by omega),
        show hi - lo + 2 - 1 = hi - lo + 1 by omega]
    have hm : ∀ m, m ≤ (hi - lo + 1) / 2 → Nat.clog 2 m ≤ Nat.clog 2 ((hi - lo + 1) / 2) :=
      fun m hm => Nat.clog_mono_right 2 hm
    have h1 := (visited_le_clog agrees fuel ((lo + hi) / 2) hi).trans (hm _ (by omega))
    have h2 := (visited_le_clog agrees fuel lo ((lo + hi) / 2)).trans (hm _ (by omega))
    have h3 := (visited_le_clog agrees fuel ((lo + hi) / 2 + 1) hi).trans (hm _ (by omega))
    have h4 := (visited_le_clog agrees fuel lo ((lo + hi) / 2 - 1)).trans (hm _ (by omega))
    generalize (if (lo + hi) / 2 = lo then some true else if (lo + hi) / 2 = hi then some false
      else agrees ((lo + hi) / 2)) = v
    generalize (if (lo + hi) / 2 - 1 = lo then some true
      else if (lo + hi) / 2 - 1 = hi then some false else agrees ((lo + hi) / 2 - 1)) = l
    generalize (if (lo + hi) / 2 + 1 = lo then some true
      else if (lo + hi) / 2 + 1 = hi then some false else agrees ((lo + hi) / 2 + 1)) = r
    rcases v with _ | _ | _ <;> rcases l with _ | _ | _ <;> rcases r with _ | _ | _ <;>
      simp only [] <;> omega

end OrthoDFA
