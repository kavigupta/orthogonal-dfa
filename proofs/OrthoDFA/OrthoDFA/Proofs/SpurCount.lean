import OrthoDFA.Proofs.SpurRounds
import OrthoDFA.Proofs.KappaCount

/-!
# What the run's rounds read

While the states are small, at most `n + 2` leaves, a probe round reads at most
`min(L + 1, 3 + ⌈log₂ L⌉)(n + 1)` strings and at most one undecided, and a split's re-sift rounds,
one per edge of `edgeList`, read at most `n + 2` strings each and one undecided. With the stretch
costs of `quiet_cost` and `Ψr` at `n`, the rounds read at most

    Σ_{j < S} (16/5) log(1/a_j) + 1 + (n + 2)|Σ|(n + 1)

strings undecided, `S` the hypotheses within `n + 2` leaves (`count6`).
-/

namespace OrthoDFA

namespace Random

namespace Spur

open Eps Kappa

variable {α : Type*} [Fintype α] [DecidableEq α]

section Round

variable (r : FreeMonoid α → ARU) (C : Level3Settings)

omit [Fintype α] [DecidableEq α] in
theorem siftReads_und_le (T : DTree α) (z : FreeMonoid α) :
    ((siftReads r T z).filter fun w => r w = .undecided).card ≤ 1 := by
  classical
  refine Finset.card_le_one.2 fun w₁ h₁ w₂ h₂ => ?_
  obtain ⟨h₁, u₁⟩ := Finset.mem_filter.1 h₁
  obtain ⟨h₂, u₂⟩ := Finset.mem_filter.1 h₂
  have e₁ := siftReads_undecided T z w₁ h₁ u₁
  have e₂ := siftReads_undecided T z w₂ h₂ u₂
  rw [e₁] at e₂
  exact Sum.inr.inj e₂

omit [DecidableEq α] in
theorem probeReads_card_mids {T : DTree α} {E : Edges α} {k : ℕ} {x : FreeMonoid α} {L : ℕ}
    (hx : x.toList.length ≤ L) :
    (probeReads r T E k x).card ≤ probeSifts L * (mids T).card := by
  have h1 : (probeReads r T E k x).card ≤ (L + 1) * (mids T).card :=
    calc _ ≤ (pot k x T).card := Finset.card_le_card (probeReads_pot _ _)
      _ ≤ (x.toList.length + 1) * (mids T).card := card_pot _ x _
      _ ≤ _ := Nat.mul_le_mul_right _ (by omega)
  have h2 : (probeReads r T E k x).card ≤ (3 + Nat.clog 2 L) * (mids T).card := by
    obtain ⟨ps, hps, hsub⟩ := probeReads_sub (rd := r) (T := T) E k hx
    calc _ ≤ (posReads r T x ps).card := Finset.card_le_card hsub
      _ ≤ ps.toFinset.card * (mids T).card := posReads_card _ _ _ _
      _ ≤ _ := Nat.mul_le_mul_right _ hps
  unfold probeSifts
  rcases le_total (L + 1) (3 + Nat.clog 2 L) with h | h
  · rw [min_eq_left h]; exact h1
  · rw [min_eq_right h]; exact h2

omit [DecidableEq α] in
theorem splitOf_some {s : Level3State α} {x : FreeMonoid α} {p : List Bool} {T' : DTree α}
    (hs : Inv r s) (h : splitOf r C s x = some (p, T')) :
    p ∈ s.tree.leaves ∧ T'.leaves.length = s.tree.leaves.length + 1 := by
  have hOK := probeOrAgree_ok r s.tree s.edges C.walkStart hs.edges x
  unfold splitOf at h
  generalize probeOrAgree r s.tree s.edges C.walkStart x = o at h hOK
  rcases o with _ | ⟨p', c, u, t⟩ | _ | _ | _
  all_goals try simp at h
  obtain ⟨hp, -⟩ := hOK
  rcases hE : s.edges (p', c) with _ | ⟨t₀, w₀⟩ <;> rw [hE] at h
  · simp at h
  simp only [Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  exact ⟨hp, DTree.length_leaves_splitAt _ _ _ hp⟩

omit [DecidableEq α] in
/-- A re-sift round reads at most the new tree's midfixes, and at most one string undecided. -/
theorem resiftOf_card {s : Level3State α} {x : FreeMonoid α} (hs : Inv r s)
    (e : List Bool × α) :
    (resiftOf r C s x e).card ≤ s.tree.leaves.length
      ∧ ((resiftOf r C s x e).filter fun z => r z = .undecided).card ≤ 1 := by
  classical
  unfold resiftOf
  rcases hso : splitOf r C s x with _ | ⟨p, T'⟩
  · simp
  rcases hE : s.edges e with _ | ⟨t, w⟩
  · simp
  dsimp only
  split_ifs
  · refine ⟨(siftReads_card r T' _).trans ?_, siftReads_und_le r T' _⟩
    have := mids_card T'
    have := (splitOf_some r C hs hso).2
    omega
  · simp

omit [DecidableEq α] in
open scoped Classical in
/-- What a step's rounds read undecided: what its probe stops at, and one per edge of a split. -/
theorem stepRound_und {s : Level3State α} {x : FreeMonoid α} (hs : Inv r s) (R : ℕ) :
    ∑ i ∈ Finset.range (R + 1), ((stepRound r C s x i).filter fun z => r z = .undecided).card
      ≤ (endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length
        + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length
        + if (splitOf r C s x).isSome then (edgeList s).length else 0 := by
  rw [Finset.sum_range_succ']
  have h0 : ((stepRound r C s x 0).filter fun z => r z = .undecided).card
      ≤ (endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length
        + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length := by
    simpa [stepRound] using probeReads_und_card r s.tree s.edges C.walkStart x
  have hrest : ∑ i ∈ Finset.range R, ((stepRound r C s x (i + 1)).filter
      fun z => r z = .undecided).card ≤ if (splitOf r C s x).isSome then (edgeList s).length
        else 0 := by
    split_ifs with hsome
    · calc _ ≤ ∑ i ∈ Finset.range R, if i < (edgeList s).length then 1 else 0 := by
            refine Finset.sum_le_sum fun i _ => ?_
            simp only [stepRound, Nat.add_sub_cancel, if_neg (Nat.succ_ne_zero i)]
            split_ifs with hi
            · rw [List.getElem?_eq_getElem hi]
              exact (resiftOf_card r C hs _).2
            · rw [List.getElem?_eq_none (by omega)]; simp
        _ = ((Finset.range R).filter fun i => i < (edgeList s).length).card := by
            rw [Finset.card_filter]
        _ ≤ (Finset.range (edgeList s).length).card := Finset.card_le_card fun i hi => by
            simp only [Finset.mem_filter, Finset.mem_range] at hi ⊢; exact hi.2
        _ = _ := Finset.card_range _
    · have hnone : splitOf r C s x = none := by
        cases h : splitOf r C s x
        · rfl
        · simp [h] at hsome
      refine le_of_eq (Finset.sum_eq_zero fun i _ => ?_)
      simp only [stepRound, if_neg (Nat.succ_ne_zero i)]
      rcases (edgeList s)[i + 1 - 1]? with _ | e
      · simp
      · simp [resiftOf, hnone]
  omega

omit [DecidableEq α] in
open scoped Classical in
/-- What a step's rounds read: its probe's reads, and the new tree's midfixes per edge of a
split. -/
theorem stepRound_card {s : Level3State α} {x : FreeMonoid α} (hs : Inv r s) (R : ℕ) :
    ∑ i ∈ Finset.range (R + 1), (stepRound r C s x i).card
      ≤ (probeReads r s.tree s.edges C.walkStart x).card
        + if (splitOf r C s x).isSome then (edgeList s).length * s.tree.leaves.length else 0 := by
  rw [Finset.sum_range_succ']
  have hrest : ∑ i ∈ Finset.range R, (stepRound r C s x (i + 1)).card
      ≤ if (splitOf r C s x).isSome then (edgeList s).length * s.tree.leaves.length else 0 := by
    split_ifs with hsome
    · calc _ ≤ ∑ i ∈ Finset.range R,
              if i < (edgeList s).length then s.tree.leaves.length else 0 := by
            refine Finset.sum_le_sum fun i _ => ?_
            simp only [stepRound, Nat.add_sub_cancel, if_neg (Nat.succ_ne_zero i)]
            split_ifs with hi
            · rw [List.getElem?_eq_getElem hi]
              exact (resiftOf_card r C hs _).1
            · rw [List.getElem?_eq_none (by omega)]; simp
        _ = ((Finset.range R).filter fun i => i < (edgeList s).length).card
              * s.tree.leaves.length := by
            rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
        _ ≤ (edgeList s).length * s.tree.leaves.length := by
            refine Nat.mul_le_mul_right _ ?_
            calc _ ≤ (Finset.range (edgeList s).length).card :=
                  Finset.card_le_card fun i hi => by
                    simp only [Finset.mem_filter, Finset.mem_range] at hi ⊢; exact hi.2
              _ = _ := Finset.card_range _
    · have hnone : splitOf r C s x = none := by
        cases h : splitOf r C s x
        · rfl
        · simp [h] at hsome
      refine le_of_eq (Finset.sum_eq_zero fun i _ => ?_)
      simp only [stepRound, if_neg (Nat.succ_ne_zero i)]
      rcases (edgeList s)[i + 1 - 1]? with _ | e
      · simp
      · simp [resiftOf, hnone]
  have h0 : (stepRound r C s x 0).card = (probeReads r s.tree s.edges C.walkStart x).card := by
    simp [stepRound]
  omega

end Round

/-! ## The run -/

section Run

variable (r : FreeMonoid α → ARU) (C : Level3Settings) (xs : List (FreeMonoid α)) (n R L : ℕ)

open scoped Classical in
/-- The strings the rounds of the first `j` steps read undecided. -/
noncomputable def undC6 (j : ℕ) : ℝ :=
  ∑ ρ ∈ Finset.range (j * (R + 1)),
    (((rounds6 C xs n R ρ r).filter fun z => r z = .undecided).card : ℝ)

/-- The strings the rounds of the first `j` steps read. -/
noncomputable def readC6 (j : ℕ) : ℝ :=
  ∑ ρ ∈ Finset.range (j * (R + 1)), ((rounds6 C xs n R ρ r).card : ℝ)

/-- The edges a state of at most `n + 2` leaves can hold. -/
def edgesMax (nα : ℕ) : ℕ := (n + 2) * nα

/-- What the rounds so far may have read undecided, given the state reached. -/
noncomputable def Φu6 (s : Level3State α) : ℝ :=
  ∑ i ∈ Finset.range s.hypotheses, costAt C i + cost s
    + (edgesMax n (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2)

/-- What the rounds so far may have read, given the state reached after `j` probes. -/
noncomputable def Φb6 (s : Level3State α) (j : ℕ) : ℝ :=
  ((probeSifts L * (n + 1) : ℕ) : ℝ) * j
    + (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s.tree.leaves.length : ℝ) - 2)

/-- The bound on what the rounds read undecided. -/
noncomputable def Mu6 (nα : ℕ) : ℝ :=
  ∑ i ∈ Finset.range (Level2.stretches n nα), costAt C i + 1 + (edgesMax n nα : ℝ) * (n + 1)

/-- The bound on what the rounds read, over `N` probes. -/
noncomputable def Mb6 (nα N : ℕ) : ℝ :=
  ((probeSifts L * (n + 1) : ℕ) : ℝ) * N + (edgesMax n nα : ℝ) * (n + 2) * (n + 1)

/-- The invariant a run keeps while its states are small. -/
structure CInv6 (s : Level3State α) : Prop where
  inv : Inv r s
  small : s.tree.leaves.length ≤ n + 2
  hyp : s.hypotheses + Ψr n s ≤ Ψr n (initialState : Level3State α)
  search : s.searchCount = s.middleUndecided.length
  cost : cost s ≤ costAt C s.hypotheses

/-- After `j` probes, the state reached bounds what the rounds so far read, while the states
are small; once they are not, or the run ends, the bounds stay. -/
def Claim6 (N j : ℕ) : Prop :=
  (∃ s, stateAt C r initialState xs j = some s ∧ Small r C xs n j ∧ CInv6 r C n s
      ∧ undC6 r C xs n R j ≤ Φu6 C n s ∧ readC6 r C xs n R j ≤ Φb6 n L s j)
    ∨ ((stateAt C r initialState xs j = none ∨ ¬ Small r C xs n j)
      ∧ undC6 r C xs n R j ≤ Mu6 C n (Fintype.card α)
      ∧ readC6 r C xs n R j ≤ Mb6 n L (Fintype.card α) N)

variable {r C xs n R L}

theorem phi6_le {s : Level3State α} (hs : CInv6 r C n s) {j N : ℕ} (hjN : j + 1 ≤ N)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    Φu6 C n s + 1 + edgesMax n (Fintype.card α) ≤ Mu6 C n (Fintype.card α)
      ∧ Φb6 n L s j + ((probeSifts L * (n + 1) : ℕ) : ℝ)
          + (edgesMax n (Fintype.card α) : ℝ) * (n + 2)
        ≤ Mb6 n L (Fintype.card α) N := by
  have hS := Ψr_start (α := α) n
  have hhyp : s.hypotheses + 1 ≤ Level2.stretches n (Fintype.card α) := by
    have := hs.hyp; omega
  have hlv : (s.tree.leaves.length : ℝ) - 2 + 1 ≤ n + 1 := by
    have : (s.tree.leaves.length : ℝ) ≤ n + 2 := by exact_mod_cast hs.small
    linarith
  have hlv0 : 0 ≤ (s.tree.leaves.length : ℝ) - 2 + 1 := by
    have := hs.inv.two
    have : (2 : ℝ) ≤ s.tree.leaves.length := by exact_mod_cast this
    linarith
  have hW0 : (0 : ℝ) ≤ edgesMax n (Fintype.card α) := Nat.cast_nonneg _
  have hsum : ∑ i ∈ Finset.range s.hypotheses, costAt C i + costAt C s.hypotheses
      ≤ ∑ i ∈ Finset.range (Level2.stretches n (Fintype.card α)), costAt C i := by
    rw [← Finset.sum_range_succ]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 hhyp)
      fun i _ _ => costAt_nonneg C (level_le_one hδ0 hδ1 i).1 (level_le_one hδ0 hδ1 i).2
  have hcost := hs.cost
  have hj : (j : ℝ) + 1 ≤ N := by exact_mod_cast hjN
  constructor
  · unfold Φu6 Mu6
    have := mul_le_mul_of_nonneg_left hlv hW0
    nlinarith
  · unfold Φb6 Mb6
    have h1 := mul_le_mul_of_nonneg_left hlv
      (by positivity : (0 : ℝ) ≤ edgesMax n (Fintype.card α) * (n + 2))
    have h2 : ((probeSifts L * (n + 1) : ℕ) : ℝ) * j + ((probeSifts L * (n + 1) : ℕ) : ℝ)
        ≤ ((probeSifts L * (n + 1) : ℕ) : ℝ) * N := by
      rw [← mul_add_one]; gcongr
    nlinarith

theorem phi6_le0 {s : Level3State α} (hs : CInv6 r C n s) {j N : ℕ} (hjN : j ≤ N)
    (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    Φu6 C n s ≤ Mu6 C n (Fintype.card α) ∧ Φb6 n L s j ≤ Mb6 n L (Fintype.card α) N := by
  have hS := Ψr_start (α := α) n
  have hhyp : s.hypotheses + 1 ≤ Level2.stretches n (Fintype.card α) := by
    have := hs.hyp; omega
  have hlv : (s.tree.leaves.length : ℝ) - 2 ≤ n + 1 := by
    have : (s.tree.leaves.length : ℝ) ≤ n + 2 := by exact_mod_cast hs.small
    linarith
  have hW0 : (0 : ℝ) ≤ edgesMax n (Fintype.card α) := Nat.cast_nonneg _
  have hsum : ∑ i ∈ Finset.range s.hypotheses, costAt C i + costAt C s.hypotheses
      ≤ ∑ i ∈ Finset.range (Level2.stretches n (Fintype.card α)), costAt C i := by
    rw [← Finset.sum_range_succ]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 hhyp)
      fun i _ _ => costAt_nonneg C (level_le_one hδ0 hδ1 i).1 (level_le_one hδ0 hδ1 i).2
  have hcost := hs.cost
  have hj : (j : ℝ) ≤ N := by exact_mod_cast hjN
  constructor
  · unfold Φu6 Mu6
    have := mul_le_mul_of_nonneg_left hlv hW0
    nlinarith
  · unfold Φb6 Mb6
    have h1 := mul_le_mul_of_nonneg_left hlv
      (by positivity : (0 : ℝ) ≤ edgesMax n (Fintype.card α) * (n + 2))
    have h2 : ((probeSifts L * (n + 1) : ℕ) : ℝ) * j ≤ ((probeSifts L * (n + 1) : ℕ) : ℝ) * N := by
      gcongr
    nlinarith

theorem claim6_zero (N : ℕ) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) : Claim6 r C xs n R L N 0 := by
  have h0 := costAt_nonneg C (level_le_one hδ0 hδ1 0).1 (level_le_one hδ0 hδ1 0).2
  by_cases hsm : (2 : ℕ) ≤ n + 2
  · refine Or.inl ⟨initialState, by simp only [stateAt], fun i hi s hs => ?_,
      ⟨inv_start r, by simp [initialState, newHypothesis, DTree.leaves],
        by simp [initialState, newHypothesis], rfl, ?_⟩, ?_, ?_⟩
    · have : i = 0 := by omega
      subst this
      simp only [stateAt, Option.some.injEq] at hs
      subst hs
      simp [initialState, newHypothesis, DTree.leaves]
    · simp [cost, initialState, newHypothesis, h0]
    · simp [undC6, Φu6, cost, initialState, newHypothesis, DTree.leaves]
    · simp [readC6, Φb6, initialState, newHypothesis, DTree.leaves]
  · omega

theorem claim6_succ (hL : ∀ x ∈ xs, x.toList.length ≤ L)
    {j : ℕ} (hj : j < xs.length)
    (hy0 : 0 < C.harvestRate) (hyε : C.harvestRate ≤ C.disagreementRate / 48)
    (hε1 : C.disagreementRate < 1) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1)
    (h : Claim6 r C xs n R L xs.length j) : Claim6 r C xs n R L xs.length (j + 1) := by
  classical
  have hjj : (j + 1) * (R + 1) = j * (R + 1) + (R + 1) := by ring
  have hsu : undC6 r C xs n R (j + 1) = undC6 r C xs n R j
      + ∑ i ∈ Finset.range (R + 1), (((rounds6 C xs n R (j * (R + 1) + i) r).filter
          fun z => r z = .undecided).card : ℝ) := by
    unfold undC6; rw [hjj, Finset.sum_range_add]
  have hsb : readC6 r C xs n R (j + 1) = readC6 r C xs n R j
      + ∑ i ∈ Finset.range (R + 1), ((rounds6 C xs n R (j * (R + 1) + i) r).card : ℝ) := by
    unfold readC6; rw [hjj, Finset.sum_range_add]
  have hdiv : ∀ i < R + 1, (j * (R + 1) + i) / (R + 1) = j := fun i hi => by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hi, zero_add]
  have hmod : ∀ i < R + 1, (j * (R + 1) + i) % (R + 1) = i := fun i hi => by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hi]
  have hW0 : (0 : ℝ) ≤ edgesMax n (Fintype.card α) := Nat.cast_nonneg _
  rcases h with ⟨s, hs, hS, hI, hu, hb⟩ | ⟨hstop, hu, hb⟩
  swap
  · -- stopped: the step's rounds are empty
    have hz : ∀ i < R + 1, rounds6 C xs n R (j * (R + 1) + i) r = ∅ := by
      intro i hi
      rcases hstop with hstop | hstop
      · unfold rounds6
        rw [hdiv i hi, hstop]
        split_ifs <;> rfl
      · exact rounds6_of_not_small (by rw [hdiv i hi]; exact hstop)
    refine Or.inr ⟨?_, ?_, ?_⟩
    · rcases hstop with hstop | hstop
      · left; rw [stateAt_succ, hstop]; rfl
      · right; exact fun h' => hstop fun i hi s hs => h' i (by omega) s hs
    · rw [hsu, Finset.sum_eq_zero fun i hi => by rw [hz i (Finset.mem_range.1 hi)]; simp]
      linarith
    · rw [hsb, Finset.sum_eq_zero fun i hi => by rw [hz i (Finset.mem_range.1 hi)]; simp]
      linarith
  -- running
  obtain ⟨x, hx⟩ : ∃ x, xs[j]? = some x := ⟨_, List.getElem?_eq_getElem hj⟩
  have hxL : x.toList.length ≤ L := hL x (List.mem_of_getElem? hx)
  have hround : ∀ i < R + 1, rounds6 C xs n R (j * (R + 1) + i) r = stepRound r C s x i :=
    fun i hi => by
      rw [rounds6_eq (s := s) (x := x) (by rw [hdiv i hi]; exact hS) (by rw [hdiv i hi]; exact hs)
        (by rw [hdiv i hi]; exact hx), hmod i hi]
  have hsU : ∑ i ∈ Finset.range (R + 1), (((rounds6 C xs n R (j * (R + 1) + i) r).filter
      fun z => r z = .undecided).card : ℝ) = ∑ i ∈ Finset.range (R + 1),
        (((stepRound r C s x i).filter fun z => r z = .undecided).card : ℝ) :=
    Finset.sum_congr rfl fun i hi => by rw [hround i (Finset.mem_range.1 hi)]
  have hsB : ∑ i ∈ Finset.range (R + 1), ((rounds6 C xs n R (j * (R + 1) + i) r).card : ℝ)
      = ∑ i ∈ Finset.range (R + 1), ((stepRound r C s x i).card : ℝ) :=
    Finset.sum_congr rfl fun i hi => by rw [hround i (Finset.mem_range.1 hi)]
  have hlenE : (edgeList s).length ≤ edgesMax n (Fintype.card α) := by
    rw [length_edgeList]; exact Nat.mul_le_mul_right _ hI.small
  have hstU := stepRound_und r C (x := x) hI.inv R
  have hstB := stepRound_card r C (x := x) hI.inv R
  have hmid : (mids s.tree).card ≤ n + 1 := by have := mids_card s.tree; have := hI.small; omega
  have hpr : (probeReads r s.tree s.edges C.walkStart x).card ≤ probeSifts L * (n + 1) :=
    (probeReads_card_mids r hxL).trans (Nat.mul_le_mul_left _ hmid)
  have hu01 := ends_mid_le_one (probeOrAgree r s.tree s.edges C.walkStart x)
  have hUstep : ∑ i ∈ Finset.range (R + 1),
      (((stepRound r C s x i).filter fun z => r z = .undecided).card : ℝ)
        ≤ (endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length
          + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length
          + if (splitOf r C s x).isSome then (edgesMax n (Fintype.card α) : ℝ) else 0 := by
    have h1 : (∑ i ∈ Finset.range (R + 1),
        ((stepRound r C s x i).filter fun z => r z = .undecided).card : ℕ)
          ≤ (endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length
            + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length
            + if (splitOf r C s x).isSome then edgesMax n (Fintype.card α) else 0 := by
      refine hstU.trans (Nat.add_le_add_left ?_ _)
      split_ifs <;> omega
    have h2 := (Nat.cast_le (α := ℝ)).2 h1
    push_cast at h2
    exact h2
  have hBstep : ∑ i ∈ Finset.range (R + 1), ((stepRound r C s x i).card : ℝ)
        ≤ ((probeSifts L * (n + 1) : ℕ) : ℝ)
          + if (splitOf r C s x).isSome then (edgesMax n (Fintype.card α) : ℝ) * (n + 2)
            else 0 := by
    have h1 : (∑ i ∈ Finset.range (R + 1), (stepRound r C s x i).card : ℕ)
          ≤ probeSifts L * (n + 1)
            + if (splitOf r C s x).isSome then edgesMax n (Fintype.card α) * (n + 2) else 0 := by
      refine hstB.trans (Nat.add_le_add hpr ?_)
      split_ifs
      · exact Nat.mul_le_mul hlenE hI.small
      · rfl
    exact_mod_cast h1
  have hu01r : ((endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length : ℝ)
      + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length ≤ 1 := by
    exact_mod_cast hu01
  obtain ⟨hpu, hpb⟩ := phi6_le (L := L) hI (j := j) (N := xs.length) (by omega) hδ0 hδ1
  have hsplitW : (if (splitOf r C s x).isSome then (edgesMax n (Fintype.card α) : ℝ) else 0)
      ≤ edgesMax n (Fintype.card α) := by split_ifs <;> linarith
  have hsplitB : (if (splitOf r C s x).isSome then (edgesMax n (Fintype.card α) : ℝ) * (n + 2)
      else 0) ≤ (edgesMax n (Fintype.card α) : ℝ) * (n + 2) := by
    split_ifs
    · exact le_rfl
    · positivity
  have hstopU : undC6 r C xs n R (j + 1) ≤ Mu6 C n (Fintype.card α) := by
    rw [hsu, hsU]; linarith
  have hstopB : readC6 r C xs n R (j + 1) ≤ Mb6 n L (Fintype.card α) xs.length := by
    rw [hsb, hsB]; linarith
  rcases hap : applyProbe r C s x with s' | e
  swap
  · refine Or.inr ⟨Or.inl ?_, hstopU, hstopB⟩
    rw [stateAt_succ, hs]
    simp [hx, hap]
  have hst : stateAt C r initialState xs (j + 1) = some s' := by
    rw [stateAt_succ, hs]
    simp [hx, hap]
  by_cases hsm' : s'.tree.leaves.length ≤ n + 2
  swap
  · exact Or.inr ⟨Or.inr fun h' => hsm' (h' (j + 1) le_rfl s' hst), hstopU, hstopB⟩
  have hS' : Small r C xs n (j + 1) := by
    intro i hi s₂ hs₂
    rcases Nat.lt_or_eq_of_le hi with hi | rfl
    · exact hS i (by omega) s₂ hs₂
    · rw [hst] at hs₂; cases hs₂; exact hsm'
  refine Or.inl ⟨s', hst, hS', ?_⟩
  have hcost := hI.cost
  have hphiU : Φu6 C n s = ∑ i ∈ Finset.range s.hypotheses, costAt C i + cost s
      + (edgesMax n (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2) := rfl
  have hphiB : Φb6 n L s j = ((probeSifts L * (n + 1) : ℕ) : ℝ) * j
      + (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s.tree.leaves.length : ℝ) - 2) := rfl
  rcases step_cases r C hI.inv hap with
    ⟨p, c, u, t, ho, hE, hu', huc, hp, rfl⟩ |
    ⟨p, c, u, t, t₀, w₀, ho, hE, htt, hp, hu', huc, rfl⟩ | ⟨rfl, hrt, hch⟩
  · -- learn
    have hso : splitOf r C s x = none := by unfold splitOf; rw [ho]
    have hU0 : ((endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length : ℝ)
        + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length = 0 := by
      rw [ho]; simp [endsString, middleString]
    set s₁ := newHypothesis s.tree (Function.update s.edges (p, c) (some (t, u)))
      (s.hypotheses + 1)
    have hΨ : Ψr n s₁ < Ψr n s :=
      Level2.Ψ_lt_learn n (s := Level2.newHypothesis s.tree s.edges 0) hp hE (t, u) 0
    have hA := costAt_nonneg C (level_le_one hδ0 hδ1 (s.hypotheses + 1)).1
      (level_le_one hδ0 hδ1 (s.hypotheses + 1)).2
    have hh : s₁.hypotheses = s.hypotheses + 1 := rfl
    have htr : s₁.tree = s.tree := rfl
    have hc1 : cost s₁ = 0 := by simp [s₁, cost, newHypothesis]
    have hphiU' : Φu6 C n s₁ = ∑ i ∈ Finset.range (s.hypotheses + 1), costAt C i + 0
        + (edgesMax n (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2) := by
      unfold Φu6; rw [hc1, hh, htr]
    rw [Finset.sum_range_succ] at hphiU'
    rw [hso] at hUstep hBstep
    simp only [Option.isSome_none, Bool.false_eq_true, if_false, add_zero] at hUstep hBstep
    refine ⟨⟨⟨edgesOK_upd r hI.inv.edges hu' huc, hI.inv.cls, hI.inv.two⟩, hsm', ?_, rfl, ?_⟩,
      ?_, ?_⟩
    · have := hI.hyp; omega
    · rw [hc1, hh]; exact hA
    · rw [hsu, hsU, hphiU']; rw [hphiU] at hu; linarith
    · rw [hsb, hsB]
      have : Φb6 n L s₁ (j + 1) = ((probeSifts L * (n + 1) : ℕ) : ℝ) * (j + 1 : ℕ)
          + (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s.tree.leaves.length : ℝ) - 2) := by
        unfold Φb6; rw [htr]
      have e2 : ((probeSifts L * (n + 1) : ℕ) : ℝ) * ((j + 1 : ℕ) : ℝ)
          = ((probeSifts L * (n + 1) : ℕ) : ℝ) * j + ((probeSifts L * (n + 1) : ℕ) : ℝ) := by
        push_cast; ring
      rw [this, e2]; rw [hphiB] at hb; linarith
  · -- split
    have hlen := DTree.length_leaves_splitAt
      (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p hp
    have hU0 : ((endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length : ℝ)
        + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length = 0 := by
      rw [ho]; simp [endsString, middleString]
    set s₁ := splitLeaf r s p c t t₀
    have hhs : s₁.tree.leaves.length = s.tree.leaves.length + 1 := hlen
    have hΨ : Ψr n s₁ < Ψr n s :=
      Level2.Ψ_lt_split n (s := Level2.newHypothesis s.tree s.edges 0) hp
        (d := FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) hsm' _ 0
    have hA := costAt_nonneg C (level_le_one hδ0 hδ1 (s.hypotheses + 1)).1
      (level_le_one hδ0 hδ1 (s.hypotheses + 1)).2
    have hh : s₁.hypotheses = s.hypotheses + 1 := rfl
    have hc1 : cost s₁ = 0 := by simp [s₁, cost, splitLeaf, newHypothesis]
    have ht := DTree.sift_inl_mem r _ _ t huc
    have ht₀ := DTree.sift_inl_mem r _ _ t₀ (hI.inv.edges p hp c t₀ w₀ hE).2
    have hl : (s₁.tree.leaves.length : ℝ) = s.tree.leaves.length + 1 := by
      rw [hhs]; push_cast; ring
    have hphiU' : Φu6 C n s₁ = ∑ i ∈ Finset.range (s.hypotheses + 1), costAt C i + 0
        + (edgesMax n (Fintype.card α) : ℝ) * ((s₁.tree.leaves.length : ℝ) - 2) := by
      unfold Φu6; rw [hc1, hh]
    rw [Finset.sum_range_succ, hl] at hphiU'
    refine ⟨⟨⟨edgesOK_retarget r hI.inv.edges hp _, ?_, ?_⟩, hsm', ?_, rfl, ?_⟩, ?_, ?_⟩
    · show (s.tree.splitAt _ p) ∈ classSet ((s.tree.splitAt _ p).leaves.length - 2)
      rw [hlen, show s.tree.leaves.length + 1 - 2 = s.tree.leaves.length - 2 + 1 by
        have := hI.inv.two; omega]
      exact classSet_split hI.inv.cls hp ht ht₀
    · show 2 ≤ (s.tree.splitAt _ p).leaves.length
      rw [hlen]; have := hI.inv.two; omega
    · have := hI.hyp; omega
    · rw [hc1, hh]; exact hA
    · rw [hsu, hsU, hphiU']; rw [hphiU] at hu
      have hso : (splitOf r C s x).isSome := by unfold splitOf; rw [ho]; simp [hE]
      rw [if_pos hso] at hUstep
      have e : (edgesMax n (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) + 1 - 2)
          = (edgesMax n (Fintype.card α) : ℝ) * ((s.tree.leaves.length : ℝ) - 2)
            + edgesMax n (Fintype.card α) := by ring
      rw [e]
      linarith
    · rw [hsb, hsB]
      have : Φb6 n L s₁ (j + 1) = ((probeSifts L * (n + 1) : ℕ) : ℝ) * (j + 1 : ℕ)
          + (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s₁.tree.leaves.length : ℝ) - 2) := by
        unfold Φb6; rfl
      have hso : (splitOf r C s x).isSome := by unfold splitOf; rw [ho]; simp [hE]
      rw [if_pos hso] at hBstep
      have e : (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s.tree.leaves.length : ℝ) + 1 - 2)
          = (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s.tree.leaves.length : ℝ) - 2)
            + (edgesMax n (Fintype.card α) : ℝ) * (n + 2) := by ring
      have e2 : ((probeSifts L * (n + 1) : ℕ) : ℝ) * ((j + 1 : ℕ) : ℝ)
          = ((probeSifts L * (n + 1) : ℕ) : ℝ) * j + ((probeSifts L * (n + 1) : ℕ) : ℝ) := by
        push_cast; ring
      rw [this, hl, e, e2]; rw [hphiB] at hb
      linarith
  · -- counted
    have hsame := step_same r C hI.inv hap rfl rfl
    have hso : splitOf r C s x = none := by
      unfold splitOf
      rcases hO : probeOrAgree r s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
      all_goals try rfl
      rw [hO] at hch; simp [changes] at hch
    rw [hso] at hUstep hBstep
    simp only [Option.isSome_none, Bool.false_eq_true, if_false, add_zero] at hUstep hBstep
    have hsc := counted_search r C hI.inv hch
    have hsearch : (countProbe r C s x).searchCount
        = (countProbe r C s x).middleUndecided.length := by
      simp only [countProbe, List.length_append, hI.search]
      omega
    have hcount : cost (countProbe r C s x) - cost s
        = (endsString (probeOrAgree r s.tree s.edges C.walkStart x)).length
          + (middleString (probeOrAgree r s.tree s.edges C.walkStart x)).length := by
      simp only [cost, countProbe, List.length_append, Nat.cast_add]
      ring
    refine ⟨⟨hsame.1, hsm', ?_, hsearch, ?_⟩, ?_, ?_⟩
    · have := hI.hyp; have := hsame.2.1 n
      simp only [countProbe] at this ⊢; omega
    · exact quiet_cost C _ hrt hsearch hy0 hyε hε1 (level_le_one hδ0 hδ1 _).1
        (level_le_one hδ0 hδ1 _).2
    · rw [hsu, hsU]
      have e1 : Φu6 C n (countProbe r C s x)
          = Φu6 C n s + (cost (countProbe r C s x) - cost s) := by
        rw [hphiU]; unfold Φu6; simp only [countProbe]; ring
      rw [e1, hcount]
      linarith
    · rw [hsb, hsB]
      have : Φb6 n L (countProbe r C s x) (j + 1)
          = ((probeSifts L * (n + 1) : ℕ) : ℝ) * j + ((probeSifts L * (n + 1) : ℕ) : ℝ)
            + (edgesMax n (Fintype.card α) : ℝ) * (n + 2) * ((s.tree.leaves.length : ℝ) - 2) := by
        unfold Φb6; simp only [countProbe]; push_cast; ring
      rw [this]; rw [hphiB] at hb; linarith

/-- While the states are small, the rounds read at most `Mu6` strings undecided and `Mb6` in
all. -/
theorem count6 (hL : ∀ x ∈ xs, x.toList.length ≤ L)
    (hy0 : 0 < C.harvestRate) (hyε : C.harvestRate ≤ C.disagreementRate / 48)
    (hε1 : C.disagreementRate < 1) (hδ0 : 0 < C.δ) (hδ1 : C.δ ≤ 1) :
    undC6 r C xs n R xs.length ≤ Mu6 C n (Fintype.card α)
      ∧ readC6 r C xs n R xs.length ≤ Mb6 n L (Fintype.card α) xs.length := by
  have hall : ∀ j ≤ xs.length, Claim6 r C xs n R L xs.length j := by
    intro j
    induction j with
    | zero => exact fun _ => claim6_zero xs.length hδ0 hδ1
    | succ j ih =>
      intro hj
      exact claim6_succ hL (by omega) hy0 hyε hε1 hδ0 hδ1 (ih (by omega))
  rcases hall xs.length le_rfl with ⟨s, -, -, hI, hu, hb⟩ | ⟨-, hu, hb⟩
  · obtain ⟨h1, h2⟩ := phi6_le0 (L := L) hI (j := xs.length) (N := xs.length) le_rfl hδ0 hδ1
    exact ⟨hu.trans h1, hb.trans h2⟩
  · exact ⟨hu, hb⟩

end Run

end Spur

end Random

end OrthoDFA
