import OrthoDFA.TripartiteSearch
import Mathlib.Tactic

/-!
# The tripartite search is correct

Each result says what `agrees` is at the positions it names. Every queried position lies between
the endpoints, and there are at most `2 + 3⌈log₂(hi − lo)⌉` queries: the two endpoints, and at
most three per halving of the interval.
-/

namespace OrthoDFA

variable (recover : Bool) (agrees : ℕ → Option Bool)

/-! ## The bisection -/

/-- What a bisection's result promises between `lo` and `hi`. -/
def BoundaryOK (lo hi : ℕ) : SearchResult → Prop
  | .edge j => lo < j ∧ j ≤ hi ∧ agrees (j - 1) = some true ∧ agrees j = some false
  | .pair j => lo ≤ j ∧ j + 1 ≤ hi ∧ agrees j = none ∧ agrees (j + 1) = none
  | .triple j => lo < j ∧ j < hi ∧ agrees (j - 1) = some true ∧ agrees j = none
      ∧ agrees (j + 1) = some false
  | .undecidedMid j => recover = false ∧ lo < j ∧ j < hi ∧ agrees j = none
  | _ => False

variable {recover agrees} in
theorem BoundaryOK.mono {lo hi lo' hi' : ℕ} {r : SearchResult}
    (h : BoundaryOK recover agrees lo' hi' r) (hlo : lo ≤ lo') (hhi : hi' ≤ hi) :
    BoundaryOK recover agrees lo hi r := by
  cases r with
  | edge j => obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨by omega, by omega, h3, h4⟩
  | pair j => obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨by omega, by omega, h3, h4⟩
  | triple j => obtain ⟨h1, h2, h3, h4, h5⟩ := h; exact ⟨by omega, by omega, h3, h4, h5⟩
  | undecidedMid j => obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, by omega, by omega, h4⟩
  | undecidedLo => exact h.elim
  | disagreeLo => exact h.elim
  | undecidedHi => exact h.elim
  | allAgree => exact h.elim

theorem bisectBoundary_ok : ∀ (n lo hi : ℕ), hi - lo = n → lo ≤ hi →
    agrees lo = some true → agrees hi = some false →
    BoundaryOK recover agrees lo hi (bisectBoundary recover agrees lo hi).1 := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn hle hlo hhi
  have hne : lo ≠ hi := by rintro rfl; rw [hlo] at hhi; simp at hhi
  rw [bisectBoundary]
  split
  · refine ⟨by omega, le_rfl, ?_, hhi⟩
    rwa [show hi - 1 = lo by omega]
  · rename_i hgap
    split
    · rename_i hm
      exact (ih _ (by omega) _ _ rfl (by omega) hm hhi).mono (by omega) le_rfl
    · rename_i hm
      exact (ih _ (by omega) _ _ rfl (by omega) hlo hm).mono le_rfl (by omega)
    · rename_i hm
      split
      · rename_i hrec
        split
        · rename_i hl
          refine ⟨by omega, by omega, hl, ?_⟩
          rwa [show (lo + hi) / 2 - 1 + 1 = (lo + hi) / 2 by omega]
        · rename_i left hl
          dsimp only
          split
          · rename_i hr
            exact ⟨by omega, by omega, hm, hr⟩
          · rename_i hr
            exact ⟨by omega, by omega, hl, hm, hr⟩
          · rename_i hr
            have : (lo + hi) / 2 - 1 ≠ lo := by rintro h; rw [h, hlo] at hl; simp at hl
            exact (ih _ (by omega) _ _ rfl (by omega) hlo hl).mono le_rfl (by omega)
          · rename_i hr
            have : (lo + hi) / 2 + 1 ≠ hi := by rintro h; rw [h, hhi] at hr; simp at hr
            exact (ih _ (by omega) _ _ rfl (by omega) hr hhi).mono (by omega) le_rfl
      · rename_i hrec
        exact ⟨by simpa using hrec, by omega, by omega, hm⟩

theorem bisectBoundary_length : ∀ (n lo hi : ℕ), hi - lo = n →
    (bisectBoundary recover agrees lo hi).2.length ≤ 3 * Nat.clog 2 (hi - lo) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn
  rw [bisectBoundary]
  split
  · simp
  · rename_i hgap
    have hc : Nat.clog 2 (hi - lo) = Nat.clog 2 ((hi - lo + 2 - 1) / 2) + 1 :=
      Nat.clog_of_two_le (by norm_num) (by omega)
    have hmono : ∀ m, m ≤ (hi - lo + 2 - 1) / 2 →
        Nat.clog 2 m ≤ Nat.clog 2 ((hi - lo + 2 - 1) / 2) := fun m h => Nat.clog_mono_right 2 h
    split
    · have := ih _ (by omega) ((lo + hi) / 2) hi rfl
      have := hmono (hi - (lo + hi) / 2) (by omega)
      simp only [List.length_cons]
      omega
    · have := ih _ (by omega) lo ((lo + hi) / 2) rfl
      have := hmono ((lo + hi) / 2 - lo) (by omega)
      simp only [List.length_cons]
      omega
    · split
      · split
        · simp; omega
        · dsimp only
          split
          · simp; omega
          · simp; omega
          · have := ih _ (by omega) lo ((lo + hi) / 2 - 1) rfl
            have := hmono ((lo + hi) / 2 - 1 - lo) (by omega)
            simp only [List.length_append, List.length_cons, List.length_nil]
            omega
          · have := ih _ (by omega) ((lo + hi) / 2 + 1) hi rfl
            have := hmono (hi - ((lo + hi) / 2 + 1)) (by omega)
            simp only [List.length_append, List.length_cons, List.length_nil]
            omega
      · simp; omega

theorem bisectBoundary_mem : ∀ (n lo hi : ℕ), hi - lo = n →
    ∀ p ∈ (bisectBoundary recover agrees lo hi).2, lo ≤ p ∧ p ≤ hi := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn p hp
  rw [bisectBoundary] at hp
  split at hp
  · simp at hp
  · rename_i hgap
    have hq : ∀ p ∈ [(lo + hi) / 2, (lo + hi) / 2 - 1, (lo + hi) / 2 + 1], lo ≤ p ∧ p ≤ hi := by
      intro p hp; simp only [List.mem_cons, List.not_mem_nil, or_false] at hp; omega
    split at hp
    · rcases List.mem_cons.1 hp with rfl | hp
      · omega
      · have := ih _ (by omega) _ _ rfl p hp; omega
    · rcases List.mem_cons.1 hp with rfl | hp
      · omega
      · have := ih _ (by omega) _ _ rfl p hp; omega
    · split at hp
      · split at hp
        · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp; omega
        · dsimp only at hp
          split at hp
          · exact hq p hp
          · exact hq p hp
          · rcases List.mem_append.1 hp with hp | hp
            · exact hq p hp
            · have := ih _ (by omega) _ _ rfl p hp; omega
          · rcases List.mem_append.1 hp with hp | hp
            · exact hq p hp
            · have := ih _ (by omega) _ _ rfl p hp; omega
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp; omega

/-! ## The search -/

variable {recover agrees} {lo hi : ℕ}

theorem tripartiteSearch_bisect (hlo : agrees lo = some true) (hhi : agrees hi = some false) :
    (tripartiteSearch recover agrees lo hi).1 = (bisectBoundary recover agrees lo hi).1 := by
  simp [tripartiteSearch, hlo, hhi]

/-- What the search's result promises between `lo` and `hi`. -/
def SearchOK (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ) : SearchResult → Prop
  | .undecidedLo => agrees lo = none
  | .disagreeLo => agrees lo = some false
  | .undecidedHi => agrees lo = some true ∧ agrees hi = none
  | .allAgree => agrees lo = some true ∧ agrees hi = some true
  | r => BoundaryOK recover agrees lo hi r

theorem tripartiteSearch_ok (hle : lo ≤ hi) :
    SearchOK recover agrees lo hi (tripartiteSearch recover agrees lo hi).1 := by
  rcases hlo : agrees lo with _ | _ | _
  · simp [tripartiteSearch, hlo, SearchOK]
  · simp [tripartiteSearch, hlo, SearchOK]
  · rcases hhi : agrees hi with _ | _ | _
    · simp [tripartiteSearch, hlo, hhi, SearchOK]
    · rw [tripartiteSearch_bisect hlo hhi]
      have := bisectBoundary_ok recover agrees _ lo hi rfl hle hlo hhi
      generalize (bisectBoundary recover agrees lo hi).1 = r at this
      cases r <;> first | exact this.elim | exact this
    · simp [tripartiteSearch, hlo, hhi, SearchOK]

variable (recover agrees lo hi)

theorem tripartiteSearch_length :
    (tripartiteSearch recover agrees lo hi).2.length ≤ 2 + 3 * Nat.clog 2 (hi - lo) := by
  have := bisectBoundary_length recover agrees _ lo hi rfl
  unfold tripartiteSearch
  split
  · simp only [List.length_cons, List.length_nil]; omega
  · simp only [List.length_cons, List.length_nil]; omega
  · split
    · simp only [List.length_cons, List.length_nil]; omega
    · simp only [List.length_cons, List.length_nil]; omega
    · simp only [List.length_cons]; omega

theorem tripartiteSearch_mem (hle : lo ≤ hi) :
    ∀ p ∈ (tripartiteSearch recover agrees lo hi).2, lo ≤ p ∧ p ≤ hi := by
  have := bisectBoundary_mem recover agrees _ lo hi rfl
  intro p hp
  unfold tripartiteSearch at hp
  split at hp
  · simp only [List.mem_singleton] at hp; omega
  · simp only [List.mem_singleton] at hp; omega
  · split at hp
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp; omega
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hp; omega
    · rcases List.mem_cons.1 hp with rfl | hp
      · omega
      rcases List.mem_cons.1 hp with rfl | hp
      · omega
      exact this p hp

theorem tripartiteSearch_monotone {t : ℕ} (hlt : lo < t) (hth : t ≤ hi)
    (hmono : ∀ i, lo ≤ i → i ≤ hi → agrees i = some (decide (i < t))) :
    (tripartiteSearch recover agrees lo hi).1 = .edge t := by
  have hlo : agrees lo = some true := by rw [hmono lo le_rfl (by omega)]; simp [hlt]
  have hhi : agrees hi = some false := by rw [hmono hi (by omega) le_rfl]; simp; omega
  have := tripartiteSearch_ok (recover := recover) (agrees := agrees) (by omega : lo ≤ hi)
  generalize (tripartiteSearch recover agrees lo hi).1 = r at this
  cases r with
  | undecidedLo => simp [SearchOK, hlo] at this
  | disagreeLo => simp [SearchOK, hlo] at this
  | undecidedHi => simp [SearchOK, hhi] at this
  | allAgree => simp [SearchOK, hhi] at this
  | edge j =>
    obtain ⟨h1, h2, h3, h4⟩ := this
    rw [hmono _ (by omega) (by omega)] at h3 h4
    simp only [Option.some.injEq, decide_eq_true_eq, decide_eq_false_iff_not, not_lt] at h3 h4
    congr 1; omega
  | pair j =>
    obtain ⟨h1, h2, h3, -⟩ := this
    rw [hmono _ (by omega) (by omega)] at h3; cases h3
  | triple j =>
    obtain ⟨h1, h2, -, h4, -⟩ := this
    rw [hmono _ (by omega) (by omega)] at h4; cases h4
  | undecidedMid j =>
    obtain ⟨-, h1, h2, h4⟩ := this
    rw [hmono _ (by omega) (by omega)] at h4; cases h4

theorem tripartiteSearch_correct : TripartiteSearchCorrect := by
  intro recover agrees lo hi hle
  have hok := tripartiteSearch_ok (recover := recover) (agrees := agrees) hle
  refine ⟨fun h => ?_, fun h => ?_, fun h => ?_, fun h => ?_, fun j h => ?_, fun j h => ?_,
    fun j h => ?_, fun j h => ?_, tripartiteSearch_mem recover agrees lo hi hle,
    tripartiteSearch_length recover agrees lo hi,
    fun t h1 h2 h3 => tripartiteSearch_monotone recover agrees lo hi h1 h2 h3⟩ <;>
    rw [h] at hok <;> exact hok

end OrthoDFA
