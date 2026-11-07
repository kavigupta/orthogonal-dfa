import OrthoDFA.LowLevelClaims.TripartiteSearch
import OrthoDFA.Proofs.Query
import Mathlib.Tactic

/-!
# The tripartite search is correct

Each result says what `agrees` is at the positions it names. Every position the search asks lies
between the endpoints, and it asks at most `2 + 3⌈log₂(hi − lo)⌉`: the two endpoints, and at most
three per halving of the interval, one without recovery.
-/

namespace OrthoDFA

open Query

variable (recover : Bool) (agrees : ℕ → Option Bool)

/-! ## The bisection -/

theorem bisectQ_eval (lo hi : ℕ) : (bisectQ recover lo hi).eval agrees =
    if hi ≤ lo + 1 then .edge hi
    else match agrees ((lo + hi) / 2) with
      | some true => (bisectQ recover ((lo + hi) / 2) hi).eval agrees
      | some false => (bisectQ recover lo ((lo + hi) / 2)).eval agrees
      | none =>
        if recover then
          match agrees ((lo + hi) / 2 - 1) with
          | none => .pair ((lo + hi) / 2 - 1)
          | some left =>
            match agrees ((lo + hi) / 2 + 1), left with
            | none, _ => .pair ((lo + hi) / 2)
            | some false, true => .triple ((lo + hi) / 2)
            | some _, false => (bisectQ recover lo ((lo + hi) / 2 - 1)).eval agrees
            | some true, true => (bisectQ recover ((lo + hi) / 2 + 1) hi).eval agrees
        else .undecidedMid ((lo + hi) / 2) := by
  rw [bisectQ]
  by_cases h : hi ≤ lo + 1
  · rw [dif_pos h, if_pos h]; rfl
  rw [dif_neg h, if_neg h, eval_ask]
  rcases agrees ((lo + hi) / 2) with _ | _ | _
  · cases recover
    · rfl
    simp only [↓reduceIte, eval_ask]
    rcases agrees ((lo + hi) / 2 - 1) with _ | left
    · rfl
    simp only [eval_ask]
    rcases agrees ((lo + hi) / 2 + 1) with _ | right
    · rfl
    cases left <;> cases right <;> rfl
  · rfl
  · rfl

theorem bisectQ_reads (lo hi : ℕ) : (bisectQ recover lo hi).reads agrees =
    if hi ≤ lo + 1 then ∅
    else insert ((lo + hi) / 2) (match agrees ((lo + hi) / 2) with
      | some true => (bisectQ recover ((lo + hi) / 2) hi).reads agrees
      | some false => (bisectQ recover lo ((lo + hi) / 2)).reads agrees
      | none =>
        if recover then
          insert ((lo + hi) / 2 - 1) (match agrees ((lo + hi) / 2 - 1) with
          | none => ∅
          | some left => insert ((lo + hi) / 2 + 1) (match agrees ((lo + hi) / 2 + 1), left with
            | none, _ => ∅
            | some false, true => ∅
            | some _, false => (bisectQ recover lo ((lo + hi) / 2 - 1)).reads agrees
            | some true, true => (bisectQ recover ((lo + hi) / 2 + 1) hi).reads agrees))
        else ∅) := by
  rw [bisectQ]
  by_cases h : hi ≤ lo + 1
  · rw [dif_pos h, if_pos h]; rfl
  rw [dif_neg h, if_neg h, reads_ask]
  congr 1
  rcases agrees ((lo + hi) / 2) with _ | _ | _
  · cases recover
    · rfl
    simp only [↓reduceIte, reads_ask]
    congr 1
    rcases agrees ((lo + hi) / 2 - 1) with _ | left
    · rfl
    simp only [reads_ask]
    congr 1
    rcases agrees ((lo + hi) / 2 + 1) with _ | right
    · rfl
    cases left <;> cases right <;> rfl
  · rfl
  · rfl

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

theorem bisectQ_ok : ∀ (n lo hi : ℕ), hi - lo = n → lo ≤ hi →
    agrees lo = some true → agrees hi = some false →
    BoundaryOK recover agrees lo hi ((bisectQ recover lo hi).eval agrees) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn hle hlo hhi
  have hne : lo ≠ hi := by rintro rfl; rw [hlo] at hhi; simp at hhi
  rw [bisectQ_eval]
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

private theorem card_insert_le_of {a : ℕ} {S : Finset ℕ} {n : ℕ} (h : S.card ≤ n) :
    (insert a S).card ≤ n + 1 :=
  (Finset.card_insert_le a S).trans (by omega)

theorem bisectQ_card : ∀ (n lo hi : ℕ), hi - lo = n →
    ((bisectQ recover lo hi).reads agrees).card ≤ 3 * Nat.clog 2 (hi - lo) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn
  rw [bisectQ_reads]
  split
  · simp
  · rename_i hgap
    have hc : Nat.clog 2 (hi - lo) = Nat.clog 2 ((hi - lo + 2 - 1) / 2) + 1 :=
      Nat.clog_of_two_le (by norm_num) (by omega)
    have hmono : ∀ m, m ≤ (hi - lo + 2 - 1) / 2 →
        Nat.clog 2 m ≤ Nat.clog 2 ((hi - lo + 2 - 1) / 2) := fun m h => Nat.clog_mono_right 2 h
    refine (Finset.card_insert_le _ _).trans ?_
    split
    · have := ih _ (by omega) ((lo + hi) / 2) hi rfl
      have := hmono (hi - (lo + hi) / 2) (by omega)
      omega
    · have := ih _ (by omega) lo ((lo + hi) / 2) rfl
      have := hmono ((lo + hi) / 2 - lo) (by omega)
      omega
    · split
      · suffices h : ∀ S : Finset ℕ, S.card ≤ 2 + 3 * Nat.clog 2 ((hi - lo + 2 - 1) / 2) →
            S.card + 1 ≤ 3 * Nat.clog 2 (hi - lo) by
          refine h _ (card_insert_le_of (n := 1 + 3 * Nat.clog 2 ((hi - lo + 2 - 1) / 2)) ?_
            |>.trans (by omega))
          split
          · simp
          · refine card_insert_le_of (n := 3 * Nat.clog 2 ((hi - lo + 2 - 1) / 2)) ?_
              |>.trans (by omega)
            split
            · simp
            · simp
            · have := ih _ (by omega) lo ((lo + hi) / 2 - 1) rfl
              have := hmono ((lo + hi) / 2 - 1 - lo) (by omega)
              omega
            · have := ih _ (by omega) ((lo + hi) / 2 + 1) hi rfl
              have := hmono (hi - ((lo + hi) / 2 + 1)) (by omega)
              omega
        intro S hS
        omega
      · simp; omega

theorem bisectQ_mem : ∀ (n lo hi : ℕ), hi - lo = n →
    ∀ p ∈ (bisectQ recover lo hi).reads agrees, lo ≤ p ∧ p ≤ hi := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn p hp
  rw [bisectQ_reads] at hp
  split at hp
  · simp at hp
  · rename_i hgap
    rcases Finset.mem_insert.1 hp with rfl | hp
    · omega
    split at hp
    · have := ih _ (by omega) _ _ rfl p hp; omega
    · have := ih _ (by omega) _ _ rfl p hp; omega
    · split at hp
      · rcases Finset.mem_insert.1 hp with rfl | hp
        · omega
        split at hp
        · simp at hp
        · rcases Finset.mem_insert.1 hp with rfl | hp
          · omega
          split at hp
          · simp at hp
          · simp at hp
          · have := ih _ (by omega) _ _ rfl p hp; omega
          · have := ih _ (by omega) _ _ rfl p hp; omega
      · simp at hp

omit recover in
/-- Without recovery the bisection never reports a pair or a triple. -/
theorem bisectQ_noRecovery : ∀ (n lo hi : ℕ), hi - lo = n → ∀ j,
    (bisectQ false lo hi).eval agrees ≠ .pair j
      ∧ (bisectQ false lo hi).eval agrees ≠ .triple j := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn j
  rw [bisectQ_eval]
  split
  · simp
  · split
    · exact ih _ (by omega) _ _ rfl j
    · exact ih _ (by omega) _ _ rfl j
    · simp

omit recover in
/-- Without recovery the bisection asks one position per halving. -/
theorem bisectQ_card_noRecovery : ∀ (n lo hi : ℕ), hi - lo = n →
    ((bisectQ false lo hi).reads agrees).card ≤ Nat.clog 2 (hi - lo) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn
  rw [bisectQ_reads]
  split
  · simp
  · rename_i hgap
    have hc : Nat.clog 2 (hi - lo) = Nat.clog 2 ((hi - lo + 2 - 1) / 2) + 1 :=
      Nat.clog_of_two_le (by norm_num) (by omega)
    have hmono : ∀ m, m ≤ (hi - lo + 2 - 1) / 2 →
        Nat.clog 2 m ≤ Nat.clog 2 ((hi - lo + 2 - 1) / 2) := fun m h => Nat.clog_mono_right 2 h
    refine (Finset.card_insert_le _ _).trans ?_
    split
    · have := ih _ (by omega) ((lo + hi) / 2) hi rfl
      have := hmono (hi - (lo + hi) / 2) (by omega)
      omega
    · have := ih _ (by omega) lo ((lo + hi) / 2) rfl
      have := hmono ((lo + hi) / 2 - lo) (by omega)
      omega
    · simp; omega

omit recover in
/-- The undecided midpoint a bisection without recovery stops at is one it asked. -/
theorem bisectQ_undecidedMid_mem : ∀ (n lo hi : ℕ), hi - lo = n → ∀ j,
    (bisectQ false lo hi).eval agrees = .undecidedMid j →
      j ∈ (bisectQ false lo hi).reads agrees := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn j hj
  rw [bisectQ_eval] at hj
  rw [bisectQ_reads]
  split at hj
  · simp at hj
  · rename_i hgap
    rw [if_neg hgap]
    split at hj
    · exact Finset.mem_insert_of_mem (ih _ (by omega) _ _ rfl j hj)
    · exact Finset.mem_insert_of_mem (ih _ (by omega) _ _ rfl j hj)
    · simp only [Bool.false_eq_true, ↓reduceIte, SearchResult.undecidedMid.injEq] at hj
      subst hj
      exact Finset.mem_insert_self _ _

omit recover in
/-- The edge a bisection without recovery finds has each side asked, or at an end. -/
theorem bisectQ_edge_mem : ∀ (n lo hi : ℕ), hi - lo = n → lo < hi → ∀ j,
    (bisectQ false lo hi).eval agrees = .edge j →
      (j = hi ∨ j ∈ (bisectQ false lo hi).reads agrees)
        ∧ (j - 1 = lo ∨ j - 1 ∈ (bisectQ false lo hi).reads agrees) := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn hlt j hj
  rw [bisectQ_eval] at hj
  rw [bisectQ_reads]
  split at hj
  · rename_i hgap
    simp only [SearchResult.edge.injEq] at hj
    subst hj
    exact ⟨Or.inl rfl, Or.inl (by omega)⟩
  · rename_i hgap
    rw [if_neg hgap]
    split at hj
    · obtain ⟨h1, h2⟩ := ih _ (by omega) _ _ rfl (by omega) j hj
      refine ⟨h1.imp_right Finset.mem_insert_of_mem, Or.inr ?_⟩
      rcases h2 with h2 | h2
      · rw [h2]; exact Finset.mem_insert_self _ _
      · exact Finset.mem_insert_of_mem h2
    · obtain ⟨h1, h2⟩ := ih _ (by omega) _ _ rfl (by omega) j hj
      refine ⟨Or.inr ?_, h2.imp_right Finset.mem_insert_of_mem⟩
      rcases h1 with h1 | h1
      · rw [h1]; exact Finset.mem_insert_self _ _
      · exact Finset.mem_insert_of_mem h1
    · simp at hj

omit agrees in
/-- A bisection without recovery under `ag`, which decides as `ag'` wherever `ag'` decides,
asks every position the one under `ag'` asks. -/
theorem bisectQ_reads_mono {ag ag' : ℕ → Option Bool}
    (h : ∀ i b, ag' i = some b → ag i = some b) : ∀ (n lo hi : ℕ), hi - lo = n →
    (bisectQ false lo hi).reads ag' ⊆ (bisectQ false lo hi).reads ag := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
  intro lo hi hn p hp
  rw [bisectQ_reads] at hp ⊢
  split at hp
  · simp at hp
  · rename_i hgap
    rw [if_neg hgap]
    rcases Finset.mem_insert.1 hp with rfl | hp
    · exact Finset.mem_insert_self _ _
    refine Finset.mem_insert_of_mem ?_
    split at hp
    · rename_i hm
      simp only [h _ _ hm]
      exact ih _ (by omega) _ _ rfl hp
    · rename_i hm
      simp only [h _ _ hm]
      exact ih _ (by omega) _ _ rfl hp
    · simp at hp

/-! ## The search -/

theorem tripartiteSearch_eq (lo hi : ℕ) : tripartiteSearch recover agrees lo hi =
    match agrees lo with
    | none => .undecidedLo
    | some false => .disagreeLo
    | some true =>
      match agrees hi with
      | none => .undecidedHi
      | some true => .allAgree
      | some false => (bisectQ recover lo hi).eval agrees := by
  change (tripartiteQ recover lo hi).eval agrees = _
  rw [tripartiteQ, eval_ask]
  rcases agrees lo with _ | _ | _
  · rfl
  · rfl
  · simp only [eval_ask]
    rcases agrees hi with _ | _ | _ <;> rfl

theorem tripartiteQ_reads (lo hi : ℕ) : (tripartiteQ recover lo hi).reads agrees =
    insert lo (match agrees lo with
    | some true =>
      insert hi (match agrees hi with
        | some false => (bisectQ recover lo hi).reads agrees
        | _ => ∅)
    | _ => ∅) := by
  rw [tripartiteQ, reads_ask]
  congr 1
  rcases agrees lo with _ | _ | _
  · rfl
  · rfl
  · simp only [reads_ask]
    congr 1
    rcases agrees hi with _ | _ | _ <;> rfl

variable {recover agrees} {lo hi : ℕ}

theorem tripartiteSearch_bisect (hlo : agrees lo = some true) (hhi : agrees hi = some false) :
    tripartiteSearch recover agrees lo hi = (bisectQ recover lo hi).eval agrees := by
  simp [tripartiteSearch_eq, hlo, hhi]

/-- What the search's result promises between `lo` and `hi`. -/
def SearchOK (recover : Bool) (agrees : ℕ → Option Bool) (lo hi : ℕ) : SearchResult → Prop
  | .undecidedLo => agrees lo = none
  | .disagreeLo => agrees lo = some false
  | .undecidedHi => agrees lo = some true ∧ agrees hi = none
  | .allAgree => agrees lo = some true ∧ agrees hi = some true
  | r => BoundaryOK recover agrees lo hi r

theorem tripartiteSearch_ok (hle : lo ≤ hi) :
    SearchOK recover agrees lo hi (tripartiteSearch recover agrees lo hi) := by
  rcases hlo : agrees lo with _ | _ | _
  · simp [tripartiteSearch_eq, hlo, SearchOK]
  · simp [tripartiteSearch_eq, hlo, SearchOK]
  · rcases hhi : agrees hi with _ | _ | _
    · simp [tripartiteSearch_eq, hlo, hhi, SearchOK]
    · rw [tripartiteSearch_bisect hlo hhi]
      have := bisectQ_ok recover agrees _ lo hi rfl hle hlo hhi
      generalize (bisectQ recover lo hi).eval agrees = r at this
      cases r <;> first | exact this.elim | exact this
    · simp [tripartiteSearch_eq, hlo, hhi, SearchOK]

omit recover in
/-- Without recovery the search never reports a pair or a triple. -/
theorem tripartiteSearch_noRecovery (j : ℕ) :
    tripartiteSearch false agrees lo hi ≠ .pair j
      ∧ tripartiteSearch false agrees lo hi ≠ .triple j := by
  rw [tripartiteSearch_eq]
  split
  · simp
  · simp
  · split
    · simp
    · simp
    · exact bisectQ_noRecovery agrees _ lo hi rfl j

variable (recover agrees lo hi)

theorem tripartiteQ_card :
    ((tripartiteQ recover lo hi).reads agrees).card ≤ 2 + 3 * Nat.clog 2 (hi - lo) := by
  have := bisectQ_card recover agrees _ lo hi rfl
  rw [tripartiteQ_reads]
  refine (Finset.card_insert_le _ _).trans ?_
  split
  · refine Nat.add_le_add_right (Finset.card_insert_le _ _) 1 |>.trans ?_
    split <;> simp <;> omega
  · simp <;> omega

omit recover in
theorem tripartiteQ_card_noRecovery :
    ((tripartiteQ false lo hi).reads agrees).card ≤ 2 + Nat.clog 2 (hi - lo) := by
  have := bisectQ_card_noRecovery agrees _ lo hi rfl
  rw [tripartiteQ_reads]
  refine (Finset.card_insert_le _ _).trans ?_
  split
  · refine Nat.add_le_add_right (Finset.card_insert_le _ _) 1 |>.trans ?_
    split <;> simp <;> omega
  · simp <;> omega

omit recover in
theorem tripartiteSearch_undecidedMid_mem (j : ℕ)
    (hj : tripartiteSearch false agrees lo hi = .undecidedMid j) :
    j ∈ (tripartiteQ false lo hi).reads agrees := by
  rw [tripartiteSearch_eq] at hj
  rw [tripartiteQ_reads]
  split at hj
  · simp at hj
  · simp at hj
  · rename_i hlo
    rw [hlo]
    split at hj
    · simp at hj
    · simp at hj
    · rename_i hhi
      rw [hhi]
      exact Finset.mem_insert_of_mem (Finset.mem_insert_of_mem
        (bisectQ_undecidedMid_mem agrees _ lo hi rfl j hj))

omit recover in
theorem tripartiteSearch_lo_mem : lo ∈ (tripartiteQ false lo hi).reads agrees := by
  rw [tripartiteQ_reads]
  exact Finset.mem_insert_self _ _

omit recover in
theorem tripartiteSearch_undecidedHi_mem
    (h : tripartiteSearch false agrees lo hi = .undecidedHi) :
    hi ∈ (tripartiteQ false lo hi).reads agrees := by
  rw [tripartiteSearch_eq] at h
  rw [tripartiteQ_reads]
  split at h
  · simp at h
  · simp at h
  · rename_i hlo
    rw [hlo]
    exact Finset.mem_insert_of_mem (Finset.mem_insert_self _ _)

omit recover in
/-- Both sides of the edge a search without recovery finds are positions it asked. -/
theorem tripartiteSearch_edge_mem (hle : lo ≤ hi) (j : ℕ)
    (hj : tripartiteSearch false agrees lo hi = .edge j) :
    j ∈ (tripartiteQ false lo hi).reads agrees
      ∧ j - 1 ∈ (tripartiteQ false lo hi).reads agrees := by
  rw [tripartiteSearch_eq] at hj
  rw [tripartiteQ_reads]
  split at hj
  · simp at hj
  · simp at hj
  · rename_i hlo
    rw [hlo]
    split at hj
    · simp at hj
    · simp at hj
    · rename_i hhi
      rw [hhi]
      have hlt : lo < hi := by
        rcases Nat.lt_or_ge lo hi with h | h
        · exact h
        · have e : lo = hi := by omega
          subst e
          rw [hlo] at hhi
          cases hhi
      obtain ⟨h1, h2⟩ := bisectQ_edge_mem agrees _ lo hi rfl hlt j hj
      refine ⟨?_, ?_⟩
      · rcases h1 with h1 | h1
        · rw [h1]; exact Finset.mem_insert_of_mem (Finset.mem_insert_self _ _)
        · exact Finset.mem_insert_of_mem (Finset.mem_insert_of_mem h1)
      · rcases h2 with h2 | h2
        · rw [h2]; exact Finset.mem_insert_self _ _
        · exact Finset.mem_insert_of_mem (Finset.mem_insert_of_mem h2)

omit recover agrees in
/-- A search without recovery under `ag`, which decides as `ag'` wherever `ag'` decides, asks
every position the one under `ag'` asks. -/
theorem tripartiteQ_reads_mono {ag ag' : ℕ → Option Bool}
    (h : ∀ i b, ag' i = some b → ag i = some b) :
    (tripartiteQ false lo hi).reads ag' ⊆ (tripartiteQ false lo hi).reads ag := by
  intro p hp
  rw [tripartiteQ_reads] at hp ⊢
  rcases Finset.mem_insert.1 hp with rfl | hp
  · exact Finset.mem_insert_self _ _
  refine Finset.mem_insert_of_mem ?_
  split at hp
  · rename_i hlo
    simp only [h _ _ hlo]
    rcases Finset.mem_insert.1 hp with rfl | hp
    · exact Finset.mem_insert_self _ _
    refine Finset.mem_insert_of_mem ?_
    split at hp
    · rename_i hhi
      simp only [h _ _ hhi]
      exact bisectQ_reads_mono h _ lo hi rfl hp
    · simp at hp
  · simp at hp

theorem tripartiteQ_mem (hle : lo ≤ hi) :
    ∀ p ∈ (tripartiteQ recover lo hi).reads agrees, lo ≤ p ∧ p ≤ hi := by
  have := bisectQ_mem recover agrees _ lo hi rfl
  intro p hp
  rw [tripartiteQ_reads] at hp
  rcases Finset.mem_insert.1 hp with rfl | hp
  · omega
  split at hp
  · rcases Finset.mem_insert.1 hp with rfl | hp
    · omega
    split at hp
    · exact this p hp
    · simp at hp
  · simp at hp

/-- Under any `agrees'` agreeing with `agrees` between `lo` and `hi`, the search returns the same
and asks the same. -/
theorem tripartiteSearch_congr {agrees' : ℕ → Option Bool} (hle : lo ≤ hi)
    (h : ∀ i, lo ≤ i → i ≤ hi → agrees' i = agrees i) :
    tripartiteSearch recover agrees' lo hi = tripartiteSearch recover agrees lo hi
      ∧ (tripartiteQ recover lo hi).reads agrees' = (tripartiteQ recover lo hi).reads agrees :=
  eval_congr agrees _ fun i hi' => by
    obtain ⟨h1, h2⟩ := tripartiteQ_mem recover agrees lo hi hle i hi'
    exact h i h1 h2

theorem tripartiteSearch_monotone {t : ℕ} (hlt : lo < t) (hth : t ≤ hi)
    (hmono : ∀ i, lo ≤ i → i ≤ hi → agrees i = some (decide (i < t))) :
    tripartiteSearch recover agrees lo hi = .edge t := by
  have hlo : agrees lo = some true := by rw [hmono lo le_rfl (by omega)]; simp [hlt]
  have hhi : agrees hi = some false := by rw [hmono hi (by omega) le_rfl]; simp; omega
  have := tripartiteSearch_ok (recover := recover) (agrees := agrees) (by omega : lo ≤ hi)
  generalize tripartiteSearch recover agrees lo hi = r at this
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
    fun j h => ?_, fun j h => ?_, tripartiteQ_mem recover agrees lo hi hle,
    tripartiteQ_card recover agrees lo hi,
    fun h => by subst h; exact tripartiteQ_card_noRecovery agrees lo hi,
    fun t h1 h2 h3 => tripartiteSearch_monotone recover agrees lo hi h1 h2 h3⟩ <;>
    rw [h] at hok <;> exact hok

end OrthoDFA
