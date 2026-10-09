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

theorem clog_le_logb (n : ℕ) : (Nat.clog 2 n : ℝ) ≤ Real.logb 2 n + 1 := by
  rcases Nat.lt_or_ge n 2 with hn | hn
  · interval_cases n <;> simp
  have hlt := Nat.pow_pred_clog_lt_self (b := 2) (by norm_num) (by omega : 1 < n)
  have hpos : 0 < Nat.clog 2 n := Nat.clog_pos (by norm_num) (by omega)
  have hlt' : (2 : ℝ) ^ ((Nat.clog 2 n - 1 : ℕ) : ℝ) < n := by
    rw [Real.rpow_natCast]; exact_mod_cast hlt
  have := (Real.lt_logb_iff_rpow_lt (by norm_num) (by positivity)).2 hlt'
  rw [Nat.cast_sub hpos] at this
  push_cast at this
  linarith

open scoped Classical in
/-- A searched draw of length `L` visits at most `searchSteps L k` middles, and an unsearched
one none. -/
theorem visits_le_steps (R : CutReads α) (t : DTree α) (edges : Edges α) {k L : ℕ}
    {x : FreeMonoid α} (hx : x.toList.length = L) :
    (visits R t edges k x : ℝ)
      ≤ searchSteps L k * if Searched R t edges k x then 1 else 0 := by
  unfold visits Searched
  rcases hw : walkCheck R t edges k x with o | ⟨ps, hi⟩
  · simp
  · obtain ⟨-, -, -, -, hhi, -⟩ := walkCheck_inr R hw
    simp only [Sum.elim_inr, Sum.isRight_inr, if_true, mul_one]
    refine le_trans ?_ (clog_le_logb (L - k))
    exact_mod_cast (visited_le_clog _ _ _ _).trans (Nat.clog_mono_right 2 (by omega))

end OrthoDFA
