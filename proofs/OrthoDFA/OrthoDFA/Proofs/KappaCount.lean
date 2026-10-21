import OrthoDFA.Proofs.KappaSteps
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The undecided reads a round makes

Under the `j`-th hypothesis, while no test fires, every search stops at an undecided middle, and
the three tests together keep the probes stopping undecided, at their start or end or a middle, to
at most `5 log(1/a_j)`, whatever the stretch's length (`quiet_cost`): the ends test bounds the
first by `(log(1/a_j) + 2yn)/log 2` after `n` probes, the middles test or the search share bounds
the second, and the success test, not firing, makes `n` at most of order `log(1/a_j)/ε`. A probe
reads at most the one string it stops at undecided, and a split re-sifts at most `(|Q| + 1)²|Σ|`
strings. So while nothing read is wrong, the round reads at most
`Σ_{j < S} 5 log(1/a_j) + 1 + |Q|(|Q| + 1)²|Σ|` strings undecided, and at most
`min(L + 1, 3 + ⌈log₂ L⌉)|Q| · |xs| + |Q|(|Q| + 1)²|Σ|` in all (`count_le`), a probe sifting
at most `min(L + 1, 3 + ⌈log₂ L⌉)` of its prefixes.
-/

namespace OrthoDFA

namespace Random

namespace Kappa

open Eps

/-! ## A stretch's undecided reads -/

section Quiet

variable {α : Type*} [Fintype α] [DecidableEq α] (C : Level3Settings)

/-- The probes counted stopping undecided, at their start or end or a middle. -/
def cost (t : Level3State α) : ℝ := t.endsUndecided.length + t.middleUndecided.length

/-- What a stretch under the `j`-th hypothesis may cost. -/
noncomputable def costAt (j : ℕ) : ℝ := 5 * Real.log (1 / C.level j)

theorem costAt_nonneg {j : ℕ} (ha0 : 0 < C.level j) (ha1 : C.level j ≤ 1) : 0 ≤ costAt C j := by
  unfold costAt
  have := Real.log_nonneg (show 1 ≤ 1 / C.level j by rw [le_div_iff₀ ha0]; linarith)
  positivity

/-- A quiet look's counts. -/
theorem quiet_cost (t : Level3State α) (hn : runTests C t = none)
    (hsm : t.searchCount = t.middleUndecided.length) (hy0 : 0 < C.harvestRate)
    (hyε : C.harvestRate ≤ C.disagreementRate / 48) (hε1 : C.disagreementRate < 1)
    (ha0 : 0 < C.level t.hypotheses) (ha1 : C.level t.hypotheses ≤ 1) :
    cost t ≤ costAt C t.hypotheses := by
  obtain ⟨he, hm, hs⟩ := look_none C hn
  set a := C.level t.hypotheses
  set y := C.harvestRate
  set ε := C.disagreementRate
  set e := t.endsUndecided.length
  set k := t.middleUndecided.length
  set n := t.probeCount
  set l := Real.log 2
  set Λ := Real.log (1 / a)
  have hl0 : 2 / 3 ≤ l := by simp only [l]; linarith [Real.log_two_gt_d9]
  have hl1 : l ≤ 7 / 10 := by simp only [l]; linarith [Real.log_two_lt_d9]
  have hla : 0 ≤ Λ := Real.log_nonneg (show 1 ≤ 1 / a by rw [le_div_iff₀ ha0]; linarith)
  have hε0 : 0 < ε := by linarith
  have hy1 : y ≤ 1 / 4 := by linarith
  have he0 : (0 : ℝ) ≤ e := Nat.cast_nonneg _
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg _
  -- the ends
  have hq0 : 0 < (1 - 2 * y) / (1 - y) := div_pos (by linarith) (by linarith)
  have hq1 : (1 - 2 * y) / (1 - y) ≤ 1 := by rw [div_le_one (by linarith)]; linarith
  have hlq : -(2 * y) ≤ Real.log ((1 - 2 * y) / (1 - y)) := by
    have h1 := Real.log_le_sub_one_of_pos (inv_pos.2 hq0)
    rw [Real.log_inv] at h1
    have h2 : ((1 - 2 * y) / (1 - y))⁻¹ - 1 ≤ 2 * y := by
      rw [inv_div, div_sub_one (by linarith), div_le_iff₀ (by linarith)]
      nlinarith
    linarith
  have hE : e * l < Λ + 2 * y * n := by
    have hlt : (2 : ℝ) ^ e * ((1 - 2 * y) / (1 - y)) ^ n < 1 / a := by
      have hlr : likelihoodRatio C.endsRate C.endsAlternative n e < 1 / a := lt_of_not_ge he
      unfold likelihoodRatio at hlr
      have h2 : C.endsAlternative / C.endsRate = 2 := by
        simp only [Level3Settings.endsAlternative, Level3Settings.endsRate]
        rw [mul_div_assoc, div_self (ne_of_gt hy0), mul_one]
      simp only [Level3Settings.endsAlternative, Level3Settings.endsRate] at hlr h2
      rw [h2] at hlr
      refine lt_of_le_of_lt (mul_le_mul_of_nonneg_left ?_ (by positivity)) hlr
      exact pow_le_pow_of_le_one hq0.le hq1 (Nat.sub_le n e)
    have hpos : 0 < (2 : ℝ) ^ e * ((1 - 2 * y) / (1 - y)) ^ n := by positivity
    have := Real.log_lt_log hpos hlt
    rw [Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow] at this
    nlinarith
  -- the middles
  have hK : k * l < Λ ∨ (k : ℝ) < ε / 6 * n := by
    by_cases hsh : C.searchShare * n ≤ t.searchCount
    · left
      have hnot : ¬ seqRate C.middleRate C.middleAlternative a t.searchCount k :=
        fun h => hm ⟨hsh, h⟩
      have hlt : likelihoodRatio C.middleRate C.middleAlternative t.searchCount k < 1 / a :=
        lt_of_not_ge hnot
      unfold likelihoodRatio at hlt
      simp only [Level3Settings.middleAlternative, Level3Settings.middleRate, hsm, Nat.sub_self,
        pow_zero, mul_one] at hlt
      norm_num at hlt
      have := Real.log_lt_log (by positivity) hlt
      rw [Real.log_pow] at this
      simpa [Λ, one_div, Real.log_inv] using this
    · right
      simp only [Level3Settings.searchShare, not_le] at hsh
      rw [hsm] at hsh
      linarith
  -- success
  have hr1 : 1 ≤ (1 - ε / 2) / (1 - ε) := by rw [le_div_iff₀ (by linarith)]; linarith
  have hlr : ε / 2 ≤ Real.log ((1 - ε / 2) / (1 - ε)) := by
    have h1 := Real.one_sub_inv_le_log_of_pos (show 0 < (1 - ε / 2) / (1 - ε) by linarith)
    have h2 : ε / 2 ≤ 1 - ((1 - ε / 2) / (1 - ε))⁻¹ := by
      rw [inv_div, one_sub_div (by linarith), le_div_iff₀ (by linarith)]
      nlinarith
    linarith
  have hS : ((n : ℝ) - k - e) * (ε / 2) - (k + e) * l < Λ := by
    have hlt : likelihoodRatio C.disagreementRate C.disagreementAlternative n
        (t.searchCount + e) < 1 / a := lt_of_not_ge hs
    unfold likelihoodRatio at hlt
    have h2 : C.disagreementAlternative / C.disagreementRate = 1 / 2 := by
      simp only [Level3Settings.disagreementAlternative]
      have : C.disagreementRate ≠ 0 := ne_of_gt hε0
      field_simp
    rw [h2, hsm] at hlt
    simp only [Level3Settings.disagreementAlternative] at hlt
    have hpos : 0 < (1 / 2 : ℝ) ^ (k + e) * ((1 - ε / 2) / (1 - ε)) ^ (n - (k + e)) := by
      have : 0 < (1 - ε / 2) / (1 - ε) := by linarith
      positivity
    have := Real.log_lt_log hpos hlt
    rw [Real.log_mul (by positivity) (by positivity), Real.log_pow, Real.log_pow,
      one_div, Real.log_inv] at this
    have hsub : (n : ℝ) - k - e ≤ ((n - (k + e) : ℕ) : ℝ) := by
      have : (n : ℝ) - (k + e) ≤ ((n - (k + e) : ℕ) : ℝ) := by
        rcases le_total (k + e) n with h | h
        · rw [Nat.cast_sub h]; push_cast; rfl
        · rw [Nat.sub_eq_zero_of_le h]; push_cast
          have : (n : ℝ) ≤ k + e := by exact_mod_cast h
          linarith
      linarith
    have hm1 : ((n : ℝ) - k - e) * (ε / 2)
        ≤ ((n - (k + e) : ℕ) : ℝ) * Real.log ((1 - ε / 2) / (1 - ε)) := by
      rcases le_total 0 ((n : ℝ) - k - e) with h | h
      · exact mul_le_mul hsub hlr (by linarith) (Nat.cast_nonneg _)
      · have : 0 ≤ ((n - (k + e) : ℕ) : ℝ) * Real.log ((1 - ε / 2) / (1 - ε)) :=
          mul_nonneg (Nat.cast_nonneg _) (by linarith)
        nlinarith
    push_cast at this
    nlinarith
  -- the cases
  have hyn : y * n ≤ ε / 48 * n := mul_le_mul_of_nonneg_right hyε hn0
  have hel : e * (2 / 3) ≤ e * l := mul_le_mul_of_nonneg_left hl0 he0
  have hkel : (k + e) * l ≤ (k + e) * (7 / 10) := mul_le_mul_of_nonneg_left hl1 (by positivity)
  have hkee : (k + e) * ε ≤ (k + e) * 1 := mul_le_mul_of_nonneg_left hε1.le (by positivity)
  have hexp : ((n : ℝ) - k - e) * (ε / 2) = (n : ℝ) * ε / 2 - (k + e) * ε / 2 := by ring
  have hSe : (n : ℝ) * ε / 2 < Λ + (k + e) * (7 / 10) + (k + e) / 2 := by linarith
  have hEe : (e : ℝ) < 3 / 2 * Λ + ε * n / 16 := by linarith
  unfold cost costAt
  rcases hK with hK | hK
  · have hkl : k * (2 / 3) ≤ k * l := mul_le_mul_of_nonneg_left hl0 hk0
    have hk : (k : ℝ) < 3 / 2 * Λ := by linarith
    linarith
  · linarith

/-- The stretches' costs over the first `H` hypotheses. -/
theorem sum_costAt_le (H : ℕ) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    ∑ j ∈ Finset.range H, costAt C j ≤ 5 * H * Real.log (8 * H * (H + 1) / C.δ) := by
  calc _ ≤ ∑ _j ∈ Finset.range H, 5 * Real.log (8 * H * (H + 1) / C.δ) := by
        refine Finset.sum_le_sum fun j hj => ?_
        have hj' : (j : ℝ) + 1 ≤ H := by exact_mod_cast Finset.mem_range.1 hj
        have hlev : 1 / C.level j = 8 * (j + 1) * (j + 2) / C.δ := by
          unfold Level3Settings.level; field_simp
        have h1 : 1 ≤ 1 / C.level j := by
          rw [hlev, le_div_iff₀ hδ0]; nlinarith
        have h0 := Real.log_nonneg h1
        have hle : Real.log (1 / C.level j) ≤ Real.log (8 * H * (H + 1) / C.δ) := by
          refine Real.log_le_log (by linarith) ?_
          rw [hlev]
          have : (8 : ℝ) * (j + 1) * (j + 2) ≤ 8 * H * (H + 1) := by nlinarith
          exact div_le_div_of_nonneg_right this hδ0.le
        unfold costAt
        linarith
    _ = _ := by simp; ring

end Quiet

/-! ## A step's reads -/

section Step

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (read : FreeMonoid α → ARU) (C : Level3Settings)

omit [Fintype σ] [DecidableEq α] in
theorem splitReads_empty {s : Level3State α} {x : FreeMonoid α}
    (h : ∀ p c u t, probeOrAgree read s.tree s.edges C.walkStart x ≠ .wrongEdge p c u t) :
    splitReads read C s x = ∅ := by
  unfold splitReads
  rcases ho : probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
  · rfl
  · exact absurd ho (h p c u t)
  all_goals rfl

omit [DecidableEq α] in
/-- A split re-sifts at most `(|Q| + 1)²|Σ|` strings. -/
theorem splitReads_card {s : Level3State α} (hW : WitInv M side s) (x : FreeMonoid α) :
    (splitReads read C s x).card
      ≤ (Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1) := by
  have hlv : s.tree.leaves.length ≤ Fintype.card σ + 1 := by
    have := hW.leaves; have := img_card_le M side s.tree; omega
  have hmid : (mids s.tree).card ≤ Fintype.card σ := by
    have := mids_card s.tree; omega
  unfold splitReads
  rcases hpo : probeOrAgree read s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
  · simp
  · simp only
    split
    · rename_i t₀ w₀ _
      refine (resifts_card s _).trans ?_
      have hm := Finset.card_le_card
        (mids_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p)
      have := Finset.card_insert_le (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀))
        (mids s.tree)
      gcongr
      omega
    · simp
  all_goals simp

/-- The most prefixes a probe at most `L` long sifts. -/
def probeSifts (L : ℕ) : ℕ := min (L + 1) (3 + Nat.clog 2 L)

omit [DecidableEq α] in
theorem probeReads_card {s : Level3State α} (hW : WitInv M side s) {x : FreeMonoid α} {L : ℕ}
    (hx : x.toList.length ≤ L) :
    (probeReads read s.tree s.edges C.walkStart x).card ≤ probeSifts L * Fintype.card σ := by
  have hlv : s.tree.leaves.length ≤ Fintype.card σ + 1 := by
    have := hW.leaves; have := img_card_le M side s.tree; omega
  have hmid : (mids s.tree).card ≤ Fintype.card σ := by
    have := mids_card s.tree; omega
  have h1 : (probeReads read s.tree s.edges C.walkStart x).card ≤ (L + 1) * Fintype.card σ :=
    calc _ ≤ (pot C.walkStart x s.tree).card := Finset.card_le_card (probeReads_pot _ _)
      _ ≤ (x.toList.length + 1) * (mids s.tree).card := card_pot _ x _
      _ ≤ _ := Nat.mul_le_mul (by omega) hmid
  have h2 : (probeReads read s.tree s.edges C.walkStart x).card
      ≤ (3 + Nat.clog 2 L) * Fintype.card σ := by
    obtain ⟨ps, hps, hsub⟩ := probeReads_sub (rd := read) (T := s.tree) s.edges C.walkStart hx
    calc _ ≤ (posReads read s.tree x ps).card := Finset.card_le_card hsub
      _ ≤ ps.toFinset.card * (mids s.tree).card := posReads_card _ _ _ _
      _ ≤ _ := Nat.mul_le_mul hps hmid
  unfold probeSifts
  rcases le_total (L + 1) (3 + Nat.clog 2 L) with h | h
  · rw [min_eq_left h]; exact h1
  · rw [min_eq_right h]; exact h2

omit [Fintype α] [DecidableEq α] in
theorem ends_mid_le_one (o : Outcome α) : (endsString o).length + (middleString o).length ≤ 1 := by
  rcases o with _ | _ | _ | _ | ⟨zs, _ | _⟩ <;> simp [endsString, middleString, List.length_take]

omit [Fintype α] [DecidableEq α] in
/-- A probe reads undecided at most the strings it stops at. -/
theorem probeReads_und_card (T : DTree α) (E : Edges α) (k : ℕ) (x : FreeMonoid α) :
    ((probeReads read T E k x).filter fun z => read z = .undecided).card
      ≤ (endsString (probeOrAgree read T E k x)).length
        + (middleString (probeOrAgree read T E k x)).length := by
  classical
  calc _ ≤ (endsString (probeOrAgree read T E k x)
        ++ middleString (probeOrAgree read T E k x)).toFinset.card := by
        refine Finset.card_le_card fun z hz => ?_
        obtain ⟨hz, hu⟩ := Finset.mem_filter.1 hz
        rw [List.mem_toFinset, List.mem_append]
        exact probeReads_undecided E k hz hu
    _ ≤ _ := (List.toFinset_card_le _).trans (by simp)

omit [DecidableEq α] in
/-- A counted probe searches exactly where it stops at an undecided middle. -/
theorem counted_search {s : Level3State α} (hs : Inv read s) {x : FreeMonoid α}
    (hc : changes (probeOrAgree read s.tree s.edges C.walkStart x) = false) :
    (if searches (probeOrAgree read s.tree s.edges C.walkStart x) then 1 else 0)
      = (middleString (probeOrAgree read s.tree s.edges C.walkStart x)).length := by
  have hOK := probeOrAgree_ok read s.tree s.edges C.walkStart hs.edges x
  generalize probeOrAgree read s.tree s.edges C.walkStart x = o at hc hOK ⊢
  rcases o with _ | _ | _ | _ | ⟨zs, _ | _⟩
  · rfl
  · simp [changes] at hc
  · simp [changes] at hc
  · rfl
  · obtain ⟨h1, -⟩ := hOK
    have := h1 rfl
    rcases zs with _ | ⟨z, zs⟩
    · exact absurd rfl this
    · simp [searches, middleString]
  · simp [searches, middleString]

end Step

/-! ## The round's reads -/

section Round

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (C : Level3Settings) (xs : List (FreeMonoid α)) (r : FreeMonoid α → ARU)

open scoped Classical in
/-- Over the rounds before `2n` that no wrong read precedes, the strings read undecided. -/
noncomputable def undCount (n : ℕ) : ℝ :=
  ∑ i ∈ (Finset.range (2 * n)).filter (Good' M side C xs r),
    (((rounds' C xs i r).filter fun z => r z = .undecided).card : ℝ)

open scoped Classical in
/-- Over the rounds before `2n` that no wrong read precedes, the strings read. -/
noncomputable def readCount (n : ℕ) : ℝ :=
  ∑ i ∈ (Finset.range (2 * n)).filter (Good' M side C xs r), ((rounds' C xs i r).card : ℝ)

theorem sum_filter_two (P : ℕ → Prop) [DecidablePred P] (f : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ (Finset.range (2 * (n + 1))).filter P, f i
      = ∑ i ∈ (Finset.range (2 * n)).filter P, f i + (if P (2 * n) then f (2 * n) else 0)
        + (if P (2 * n + 1) then f (2 * n + 1) else 0) := by
  rw [Finset.sum_filter, Finset.sum_filter, show 2 * (n + 1) = 2 * n + 1 + 1 by ring,
    Finset.sum_range_succ, Finset.sum_range_succ]

omit [Fintype σ] in
theorem good'_mono {m n : ℕ} (h : Good' M side C xs r n) (hmn : m ≤ n) :
    Good' M side C xs r m := fun i hi => h i (by omega)

/-- The invariant a run keeps while nothing read is wrong. -/
structure CInv (s : Level3State α) : Prop where
  inv : Inv r s
  wit : WitInv M side s
  hyp : s.hypotheses + Ψr (Fintype.card σ) s ≤ Ψr (Fintype.card σ) (initialState : Level3State α)
  search : s.searchCount = s.middleUndecided.length
  cost : cost s ≤ costAt C s.hypotheses

/-- What a split may re-sift. -/
def wSplit (nQ nα : ℕ) : ℕ := (nQ + 1) * nα * (nQ + 1)

/-- What the rounds so far may have read undecided, given the state reached. -/
noncomputable def Φu (nQ nα : ℕ) (C : Level3Settings) (s : Level3State α) : ℝ :=
  ∑ j ∈ Finset.range s.hypotheses, costAt C j + cost s
    + (wSplit nQ nα : ℝ) * ((s.tree.leaves.length : ℝ) - 2)

/-- What the rounds so far may have read, given the state reached after `n` probes. -/
noncomputable def Φb (nQ nα L : ℕ) (s : Level3State α) (n : ℕ) : ℝ :=
  ((probeSifts L * nQ : ℕ) : ℝ) * n + (wSplit nQ nα : ℝ) * ((s.tree.leaves.length : ℝ) - 2)

/-- The bound on what the round reads undecided. -/
noncomputable def Mu (nQ nα : ℕ) (C : Level3Settings) : ℝ :=
  ∑ j ∈ Finset.range (Level2.stretches nQ nα), costAt C j + 1 + (wSplit nQ nα : ℝ) * nQ

/-- The bound on what the round reads. -/
noncomputable def Mb (nQ nα L N : ℕ) : ℝ :=
  ((probeSifts L * nQ : ℕ) : ℝ) * N + (wSplit nQ nα : ℝ) * nQ

variable {C}

omit [Fintype σ] [Fintype α] [DecidableEq α] in
theorem level_le_one (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) (j : ℕ) : 0 < C.level j ∧ C.level j ≤ 1 := by
  unfold Level3Settings.level
  refine ⟨by positivity, ?_⟩
  rw [div_le_one (by positivity)]
  have : (1 : ℝ) ≤ 8 * (j + 1) * (j + 2) := by nlinarith
  linarith

/-- The state reached bounds what the rounds so far read, with room for one more step. -/
theorem phi_le {s : Level3State α} (hs : CInv M side C r s) {n N : ℕ} (hnN : n + 1 ≤ N) (L : ℕ)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    Φu (Fintype.card σ) (Fintype.card α) C s + 1 + wSplit (Fintype.card σ) (Fintype.card α)
        ≤ Mu (Fintype.card σ) (Fintype.card α) C
      ∧ Φb (Fintype.card σ) (Fintype.card α) L s n + ((probeSifts L * Fintype.card σ : ℕ) : ℝ)
        + wSplit (Fintype.card σ) (Fintype.card α) ≤ Mb (Fintype.card σ) (Fintype.card α) L N := by
  have hlv : s.tree.leaves.length ≤ Fintype.card σ + 1 := by
    have := hs.wit.leaves; have := img_card_le M side s.tree; omega
  have hS := Ψr_start (α := α) (Fintype.card σ)
  have hhyp : s.hypotheses + 1 ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) := by
    have := hs.hyp; omega
  have hlv' : (s.tree.leaves.length : ℝ) - 2 + 1 ≤ Fintype.card σ := by
    have : (s.tree.leaves.length : ℝ) ≤ Fintype.card σ + 1 := by exact_mod_cast hlv
    linarith
  have hW0 : (0 : ℝ) ≤ wSplit (Fintype.card σ) (Fintype.card α) := Nat.cast_nonneg _
  have hsum : ∑ j ∈ Finset.range s.hypotheses, costAt C j + costAt C s.hypotheses
      ≤ ∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)), costAt C j := by
    rw [← Finset.sum_range_succ]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 hhyp)
      fun j _ _ => costAt_nonneg C (level_le_one hδ0 hδ1 j).1 (level_le_one hδ0 hδ1 j).2
  have hcost := hs.cost
  have hn : (n : ℝ) + 1 ≤ N := by exact_mod_cast hnN
  constructor
  · unfold Φu Mu
    have : (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2)
        + wSplit (Fintype.card σ) (Fintype.card α)
        ≤ (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) * Fintype.card σ := by
      rw [← mul_add_one]; gcongr
    linarith
  · unfold Φb Mb
    have : (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2)
        + wSplit (Fintype.card σ) (Fintype.card α)
        ≤ (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) * Fintype.card σ := by
      rw [← mul_add_one]; gcongr
    have : ((probeSifts L * Fintype.card σ : ℕ) : ℝ) * n + ((probeSifts L * Fintype.card σ : ℕ) : ℝ)
        ≤ ((probeSifts L * Fintype.card σ : ℕ) : ℝ) * N := by
      rw [← mul_add_one]; gcongr
    linarith

omit [DecidableEq α] [Fintype α] [Fintype σ] in
theorem phiU_fresh (nQ nα : ℕ) {s s' : Level3State α}
    (hh : s'.hypotheses = s.hypotheses + 1) (hc0 : cost s' = 0)
    (hcost : cost s ≤ costAt C s.hypotheses) :
    Φu nQ nα C s + (wSplit nQ nα : ℝ) * ((s'.tree.leaves.length : ℝ) - s.tree.leaves.length)
      ≤ Φu nQ nα C s' := by
  unfold Φu
  rw [hh, Finset.sum_range_succ, hc0]
  have e3 : (wSplit nQ nα : ℝ) * ((s'.tree.leaves.length : ℝ) - 2)
      = (wSplit nQ nα : ℝ) * ((s.tree.leaves.length : ℝ) - 2)
        + (wSplit nQ nα : ℝ) * ((s'.tree.leaves.length : ℝ) - s.tree.leaves.length) := by ring
  rw [e3]
  linarith

omit [DecidableEq α] [Fintype α] [Fintype σ] in
theorem phiU_count (nQ nα : ℕ) {s s' : Level3State α} (hh : s'.hypotheses = s.hypotheses)
    (hl : s'.tree = s.tree) : Φu nQ nα C s' = Φu nQ nα C s + (cost s' - cost s) := by
  unfold Φu
  rw [hh, hl]
  ring

omit [DecidableEq α] [Fintype α] [Fintype σ] in
theorem phiB_step (nQ nα L : ℕ) {s s' : Level3State α} (n : ℕ) :
    Φb nQ nα L s' (n + 1) = Φb nQ nα L s n + ((probeSifts L * nQ : ℕ) : ℝ)
      + (wSplit nQ nα : ℝ) * ((s'.tree.leaves.length : ℝ) - s.tree.leaves.length) := by
  unfold Φb
  push_cast
  ring

variable (C)

/-- After `n` probes, the state reached bounds what the rounds so far read, while nothing read is
wrong; once something is, or the run ends, the bounds stay. -/
def Claim (L N n : ℕ) : Prop :=
  (Good' M side C xs r (2 * n) ∧ ∃ s, stateAt C r initialState xs n = some s
      ∧ CInv M side C r s
      ∧ undCount M side C xs r n ≤ Φu (Fintype.card σ) (Fintype.card α) C s
      ∧ readCount M side C xs r n ≤ Φb (Fintype.card σ) (Fintype.card α) L s n)
    ∨ ((¬ Good' M side C xs r (2 * n) ∨ stateAt C r initialState xs n = none)
      ∧ undCount M side C xs r n ≤ Mu (Fintype.card σ) (Fintype.card α) C
      ∧ readCount M side C xs r n ≤ Mb (Fintype.card σ) (Fintype.card α) L N)

variable {C}

theorem claim_zero (L N : ℕ) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) : Claim M side C xs r L N 0 := by
  have h0 := costAt_nonneg C (level_le_one hδ0 hδ1 0).1 (level_le_one hδ0 hδ1 0).2
  refine Or.inl ⟨fun i hi => absurd hi (by omega), initialState, by simp only [stateAt],
    ⟨inv_start r, witInv_start M side, by simp [initialState, newHypothesis], rfl, ?_⟩, ?_, ?_⟩
  · simp [cost, initialState, newHypothesis, h0]
  · simp [undCount, Φu, cost, initialState, newHypothesis, DTree.leaves]
  · simp [readCount, Φb, initialState, newHypothesis, DTree.leaves]

theorem claim_succ (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L) {n : ℕ} (hn : n < xs.length)
    (hy0 : 0 < C.harvestRate) (hyε : C.harvestRate ≤ C.disagreementRate / 48)
    (hε1 : C.disagreementRate < 1) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1)
    (h : Claim M side C xs r L xs.length n) : Claim M side C xs r L xs.length (n + 1) := by
  classical
  have hsu : undCount M side C xs r (n + 1) = undCount M side C xs r n
      + (if Good' M side C xs r (2 * n) then
          (((rounds' C xs (2 * n) r).filter fun z => r z = .undecided).card : ℝ) else 0)
      + (if Good' M side C xs r (2 * n + 1) then
          (((rounds' C xs (2 * n + 1) r).filter fun z => r z = .undecided).card : ℝ) else 0) := by
    unfold undCount; exact sum_filter_two _ _ n
  have hsb : readCount M side C xs r (n + 1) = readCount M side C xs r n
      + (if Good' M side C xs r (2 * n) then ((rounds' C xs (2 * n) r).card : ℝ) else 0)
      + (if Good' M side C xs r (2 * n + 1) then ((rounds' C xs (2 * n + 1) r).card : ℝ)
        else 0) := by
    unfold readCount; exact sum_filter_two _ _ n
  have hW0 : (0 : ℝ) ≤ wSplit (Fintype.card σ) (Fintype.card α) := Nat.cast_nonneg _
  rcases h with ⟨hg, s, hs, hI, hu, hb⟩ | ⟨hstop, hu, hb⟩
  swap
  · -- stopped: nothing more is counted
    have hz : ∀ i, i = 2 * n ∨ i = 2 * n + 1 →
        (¬ Good' M side C xs r i ∨ rounds' C xs i r = ∅) := by
      rintro i hi
      rcases hstop with hstop | hstop
      · exact Or.inl fun hgi => hstop (good'_mono M side C xs r hgi (by omega))
      · right
        have : i / 2 = n := by omega
        simp [rounds', this, hstop]
    have e1 : ∀ i, i = 2 * n ∨ i = 2 * n + 1 →
        (if Good' M side C xs r i then
          (((rounds' C xs i r).filter fun z => r z = .undecided).card : ℝ) else 0) = 0 := by
      intro i hi
      rcases hz i hi with h | h
      · rw [if_neg h]
      · simp [h]
    have e2 : ∀ i, i = 2 * n ∨ i = 2 * n + 1 →
        (if Good' M side C xs r i then ((rounds' C xs i r).card : ℝ) else 0) = 0 := by
      intro i hi
      rcases hz i hi with h | h
      · rw [if_neg h]
      · simp [h]
    refine Or.inr ⟨?_, ?_, ?_⟩
    · rcases hstop with hstop | hstop
      · exact Or.inl fun hgi => hstop (good'_mono M side C xs r hgi (by omega))
      · right; rw [stateAt_succ, hstop]; rfl
    · rw [hsu, e1 _ (Or.inl rfl), e1 _ (Or.inr rfl)]; linarith
    · rw [hsb, e2 _ (Or.inl rfl), e2 _ (Or.inr rfl)]; linarith
  -- running
  obtain ⟨x, hx⟩ : ∃ x, xs[n]? = some x := ⟨_, List.getElem?_eq_getElem hn⟩
  have hxL : x.toList.length ≤ L := hL x (List.mem_of_getElem? hx)
  have hr0 := rounds'_even C xs r hs hx
  have hr1 := rounds'_odd C xs r hs hx
  have hu0 := probeReads_und_card r s.tree s.edges C.walkStart x
  have hu01 := ends_mid_le_one (probeOrAgree r s.tree s.edges C.walkStart x)
  have hb0 := probeReads_card M side r C hI.wit hxL
  have hb1 := splitReads_card M side r C hI.wit x
  have hu1 := Finset.card_filter_le (splitReads r C s x) fun z => r z = .undecided
  obtain ⟨o, ho_def⟩ : ∃ o, probeOrAgree r s.tree s.edges C.walkStart x = o := ⟨_, rfl⟩
  rw [ho_def] at hu0 hu01
  have hU : undCount M side C xs r (n + 1) ≤ undCount M side C xs r n
      + (((probeReads r s.tree s.edges C.walkStart x).filter fun z => r z = .undecided).card : ℝ)
      + (((splitReads r C s x).filter fun z => r z = .undecided).card : ℝ) := by
    rw [hsu, hr0, hr1]
    gcongr
    · split_ifs <;> simp
    · split_ifs <;> simp
  have hB : readCount M side C xs r (n + 1) ≤ readCount M side C xs r n
      + ((probeReads r s.tree s.edges C.walkStart x).card : ℝ)
      + ((splitReads r C s x).card : ℝ) := by
    rw [hsb, hr0, hr1]
    gcongr
    · split_ifs <;> simp
    · split_ifs <;> simp
  have hu0r : (((probeReads r s.tree s.edges C.walkStart x).filter
      fun z => r z = .undecided).card : ℝ) ≤ (endsString o).length + (middleString o).length := by
    exact_mod_cast hu0
  have hu01r : ((endsString o).length : ℝ) + (middleString o).length ≤ 1 := by
    exact_mod_cast hu01
  have hb0r : ((probeReads r s.tree s.edges C.walkStart x).card : ℝ)
      ≤ ((probeSifts L * Fintype.card σ : ℕ) : ℝ) := by exact_mod_cast hb0
  have hb1r : ((splitReads r C s x).card : ℝ) ≤ (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) := by
    simp only [wSplit]; exact_mod_cast hb1
  have hu1r : (((splitReads r C s x).filter fun z => r z = .undecided).card : ℝ)
      ≤ ((splitReads r C s x).card : ℝ) := by exact_mod_cast hu1
  have hs0U : splitReads r C s x = ∅ →
      (((splitReads r C s x).filter fun z => r z = .undecided).card : ℝ) = 0 := fun h => by
    simp [h]
  have hs0B : splitReads r C s x = ∅ → ((splitReads r C s x).card : ℝ) = 0 := fun h => by
    simp [h]
  generalize (((probeReads r s.tree s.edges C.walkStart x).filter
    fun z => r z = .undecided).card : ℝ) = U0 at hU hu0r
  generalize (((splitReads r C s x).filter fun z => r z = .undecided).card : ℝ) = U1
    at hU hu1r hs0U
  generalize ((probeReads r s.tree s.edges C.walkStart x).card : ℝ) = B0 at hB hb0r
  generalize ((splitReads r C s x).card : ℝ) = B1 at hB hb1r hu1r hs0B
  obtain ⟨hpu, hpb⟩ := phi_le M side r hI (n := n) (N := xs.length) (by omega) L hδ0 hδ1
  have hstopU : undCount M side C xs r (n + 1)
      ≤ Mu (Fintype.card σ) (Fintype.card α) C := by linarith
  have hstopB : readCount M side C xs r (n + 1)
      ≤ Mb (Fintype.card σ) (Fintype.card α) L xs.length := by linarith
  rcases hap : applyProbe r C s x with s' | e
  swap
  · refine Or.inr ⟨Or.inr ?_, hstopU, hstopB⟩
    rw [stateAt_succ, hs]
    simp [hx, hap]
  have hst : stateAt C r initialState xs (n + 1) = some s' := by
    rw [stateAt_succ, hs]
    simp [hx, hap]
  by_cases hg2 : Good' M side C xs r (2 * (n + 1))
  swap
  · exact Or.inr ⟨Or.inl hg2, hstopU, hstopB⟩
  have hnw : ¬ ReadsWrong M side r C s x := by
    rintro ⟨z, hz, hw⟩
    rcases Finset.mem_union.1 hz with hz | hz
    · exact hg2 (2 * n) (by omega) z (by rw [hr0]; exact hz) hw
    · exact hg2 (2 * n + 1) (by omega) z (by rw [hr1]; exact hz) hw
  refine Or.inl ⟨hg2, s', hst, ?_⟩
  have hcost := hI.cost
  have hpB := phiB_step (Fintype.card σ) (Fintype.card α) L (s := s) (s' := s') n
  rcases step_cases r C hI.inv hap with
    ⟨p, c, u, t, ho, hE, -, -, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, -, -, rfl⟩ | ⟨rfl, hrt, hch⟩
  · -- learn
    have hne : ¬ ((newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
        (s.hypotheses + 1)).tree = s.tree ∧ (newHypothesis s.tree
          (Function.update s.edges (p, c) (some (t, u))) (s.hypotheses + 1)).edges = s.edges) := by
      rintro ⟨-, he⟩
      have := congrFun he (p, c)
      simp [newHypothesis, hE] at this
    obtain ⟨hI1, hW1, -, -, hΨ⟩ := kappa_step_change M side r C hI.inv hI.wit hnw hap hne
    have hs0 : splitReads r C s x = ∅ := splitReads_empty r C fun p' c' u' t' h' => by
      rw [ho] at h'; cases h'
    rw [ho_def] at ho
    subst ho
    have hU1 := hs0U hs0
    have hB1 := hs0B hs0
    have hU0 : U0 ≤ 0 := by simpa [endsString, middleString] using hu0r
    have hA := costAt_nonneg C (level_le_one hδ0 hδ1 (s.hypotheses + 1)).1
      (level_le_one hδ0 hδ1 (s.hypotheses + 1)).2
    have hph := phiU_fresh (C := C) (Fintype.card σ) (Fintype.card α)
      (s' := newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
        (s.hypotheses + 1)) rfl (by simp [cost, newHypothesis]) hcost
    simp only [newHypothesis, sub_self, mul_zero, add_zero] at hph hpB
    refine ⟨⟨hI1, hW1, ?_, rfl, ?_⟩, ?_, ?_⟩
    · have := hI.hyp; simp only [newHypothesis] at hΨ ⊢; omega
    · simp [cost, newHypothesis, hA]
    · simp only [newHypothesis]; linarith
    · simp only [newHypothesis]; linarith
  · -- split
    have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have hne : ¬ ((splitLeaf r s p c t t₀).tree = s.tree
        ∧ (splitLeaf r s p c t t₀).edges = s.edges) := by
      rintro ⟨ht, -⟩
      have := congrArg (fun T => T.leaves.length) ht
      simp only [splitLeaf, newHypothesis] at this
      omega
    obtain ⟨hI1, hW1, -, -, hΨ⟩ := kappa_step_change M side r C hI.inv hI.wit hnw hap hne
    rw [ho_def] at ho
    subst ho
    have hU0 : U0 ≤ 0 := by simpa [endsString, middleString] using hu0r
    have hl : ((splitLeaf r s p c t t₀).tree.leaves.length : ℝ) - s.tree.leaves.length = 1 := by
      simp only [splitLeaf, newHypothesis]; rw [hlen]; push_cast; ring
    have hA := costAt_nonneg C (level_le_one hδ0 hδ1 (s.hypotheses + 1)).1
      (level_le_one hδ0 hδ1 (s.hypotheses + 1)).2
    have hph := phiU_fresh (C := C) (Fintype.card σ) (Fintype.card α)
      (s' := splitLeaf r s p c t t₀) rfl (by simp [cost, splitLeaf, newHypothesis]) hcost
    rw [hl] at hph hpB
    refine ⟨⟨hI1, hW1, ?_, rfl, ?_⟩, ?_, ?_⟩
    · have := hI.hyp; simp only [splitLeaf, newHypothesis] at hΨ ⊢; omega
    · simp [cost, splitLeaf, newHypothesis, hA]
    · linarith
    · linarith
  · -- counted
    have hsame := step_same r C hI.inv hap rfl rfl
    have hs0 : splitReads r C s x = ∅ := splitReads_empty r C fun p' c' u' t' h' => by
      rw [h'] at hch; simp [changes] at hch
    have hsc := counted_search r C hI.inv hch
    have hsearch : (countProbe r C s x).searchCount
        = (countProbe r C s x).middleUndecided.length := by
      simp only [countProbe, List.length_append, hI.search]
      omega
    have hcount : cost (countProbe r C s x) - cost s
        = (endsString o).length + (middleString o).length := by
      simp only [cost, countProbe, List.length_append, Nat.cast_add, ho_def]
      ring
    have hU1 := hs0U hs0
    have hB1 := hs0B hs0
    have hph := phiU_count (C := C) (Fintype.card σ) (Fintype.card α) (s := s)
      (s' := countProbe r C s x) rfl rfl
    simp only [countProbe_tree, sub_self, mul_zero, add_zero] at hpB
    refine ⟨⟨hsame.1, witInv_congr M side hI.wit rfl rfl, ?_, hsearch, ?_⟩, ?_, ?_⟩
    · have := hI.hyp; have := hsame.2.1 (Fintype.card σ)
      simp only [countProbe] at this ⊢; omega
    · exact quiet_cost C _ hrt hsearch hy0 hyε hε1 (level_le_one hδ0 hδ1 _).1
        (level_le_one hδ0 hδ1 _).2
    · linarith
    · linarith

/-- The state reached bounds what the rounds so far read. -/
theorem phi_le0 {s : Level3State α} (hs : CInv M side C r s) {n N : ℕ} (hnN : n ≤ N) (L : ℕ)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    Φu (Fintype.card σ) (Fintype.card α) C s ≤ Mu (Fintype.card σ) (Fintype.card α) C
      ∧ Φb (Fintype.card σ) (Fintype.card α) L s n ≤ Mb (Fintype.card σ) (Fintype.card α) L N := by
  have hlv : s.tree.leaves.length ≤ Fintype.card σ + 1 := by
    have := hs.wit.leaves; have := img_card_le M side s.tree; omega
  have hS := Ψr_start (α := α) (Fintype.card σ)
  have hhyp : s.hypotheses + 1 ≤ Level2.stretches (Fintype.card σ) (Fintype.card α) := by
    have := hs.hyp; omega
  have hlv' : (s.tree.leaves.length : ℝ) - 2 ≤ Fintype.card σ := by
    have : (s.tree.leaves.length : ℝ) ≤ Fintype.card σ + 1 := by exact_mod_cast hlv
    linarith
  have hW0 : (0 : ℝ) ≤ wSplit (Fintype.card σ) (Fintype.card α) := Nat.cast_nonneg _
  have hsum : ∑ j ∈ Finset.range s.hypotheses, costAt C j + costAt C s.hypotheses
      ≤ ∑ j ∈ Finset.range (Level2.stretches (Fintype.card σ) (Fintype.card α)), costAt C j := by
    rw [← Finset.sum_range_succ]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 hhyp)
      fun j _ _ => costAt_nonneg C (level_le_one hδ0 hδ1 j).1 (level_le_one hδ0 hδ1 j).2
  have hcost := hs.cost
  have hn : (n : ℝ) ≤ N := by exact_mod_cast hnN
  have hWl : (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2)
      ≤ (wSplit (Fintype.card σ) (Fintype.card α) : ℝ) * Fintype.card σ := by gcongr
  constructor
  · unfold Φu Mu
    linarith
  · unfold Φb Mb
    have : ((probeSifts L * Fintype.card σ : ℕ) : ℝ) * n
        ≤ ((probeSifts L * Fintype.card σ : ℕ) : ℝ) * N := by
      gcongr
    linarith

/-- While nothing read is wrong, the round reads at most `Mu` strings undecided and `Mb` in all. -/
theorem count_le (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L) (hy0 : 0 < C.harvestRate)
    (hyε : C.harvestRate ≤ C.disagreementRate / 48) (hε1 : C.disagreementRate < 1)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    undCount M side C xs r xs.length ≤ Mu (Fintype.card σ) (Fintype.card α) C
      ∧ readCount M side C xs r xs.length ≤ Mb (Fintype.card σ) (Fintype.card α) L xs.length := by
  have hall : ∀ n ≤ xs.length, Claim M side C xs r L xs.length n := by
    intro n
    induction n with
    | zero => exact fun _ => claim_zero M side xs r L _ hδ0 hδ1
    | succ n ih =>
      intro hn
      exact claim_succ M side xs r L hL (by omega) hy0 hyε hε1 hδ0 hδ1 (ih (by omega))
  rcases hall xs.length le_rfl with ⟨-, s, -, hI, hu, hb⟩ | ⟨-, hu, hb⟩
  · obtain ⟨h1, h2⟩ := phi_le0 M side r hI le_rfl L hδ0 hδ1
    exact ⟨hu.trans h1, hb.trans h2⟩
  · exact ⟨hu, hb⟩

end Round

end Kappa

end Random

end OrthoDFA
