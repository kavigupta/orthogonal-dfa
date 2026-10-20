import OrthoDFA.Proofs.LoopAux

/-!
# The loop's rounds

Round `i` is run on the populations the rounds before it harvested, every one of which harvested,
and the loop stops growing at the first round that does not harvest or at `K` rounds.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
  (D Dsf : Measure (FreeMonoid α)) (draws : Fin K → LoopRoundDraws Ω α K settings.level3.probeBudget)

/-- The populations `rounds` harvested, in order, their samplers drawing from `D` walked from
`k`. -/
noncomputable def harvestsOf (D : Measure (FreeMonoid α)) (k : ℕ)
    (rounds : List (LoopRound Ω α K)) :
    List (Measure (FreeMonoid α)) :=
  rounds.filterMap fun o => o.population D k

@[simp] lemma histOf_pops (D : Measure (FreeMonoid α)) (k : ℕ) (rounds : List (LoopRound Ω α K)) :
    (histOf D k rounds).pops = harvestsOf D k rounds := rfl

/-- Every round harvested. -/
abbrev AllHarvested (rounds : List (LoopRound Ω α K)) : Prop :=
  ∀ o ∈ rounds, o.result.ending.harvested ≠ none

lemma loopRounds_succ (r : ℕ) :
    loopRounds settings O D Dsf draws (r + 1) =
      if h : r < K ∧ AllHarvested (loopRounds settings O D Dsf draws r) then
        loopRounds settings O D Dsf draws r
          ++ [loopRound settings O D Dsf
            (histOf D settings.level3.walkStart (loopRounds settings O D Dsf draws r))
            (draws ⟨r, h.1⟩)]
      else loopRounds settings O D Dsf draws r := by
  rw [loopRounds]

lemma length_loopRounds_le (r : ℕ) : (loopRounds settings O D Dsf draws r).length ≤ r := by
  induction r with
  | zero => simp [loopRounds]
  | succ r ih =>
    rw [loopRounds_succ]
    split_ifs
    · simp; omega
    · omega

lemma length_eq_of_allHarvested {r : ℕ} (hr : r ≤ K)
    (h : AllHarvested (loopRounds settings O D Dsf draws r)) :
    (loopRounds settings O D Dsf draws r).length = r := by
  induction r with
  | zero => simp [loopRounds]
  | succ r ih =>
    rw [loopRounds_succ] at h ⊢
    split_ifs at h ⊢ with hc
    · have hall : AllHarvested (loopRounds settings O D Dsf draws r) := fun o ho => h o (by simp [ho])
      simp [ih (by omega) hall]
    · exact absurd ⟨by omega, h⟩ hc

lemma loopRounds_prefix (r : ℕ) :
    loopRounds settings O D Dsf draws r <+: loopRounds settings O D Dsf draws (r + 1) := by
  rw [loopRounds_succ]
  split_ifs
  · exact List.prefix_append _ _
  · exact List.prefix_refl _

/-- Every round but the last harvested. -/
lemma harvested_of_lt (r i : ℕ) (hi : i + 1 < (loopRounds settings O D Dsf draws r).length) :
    (loopRounds settings O D Dsf draws r)[i].result.ending.harvested ≠ none := by
  induction r with
  | zero => simp [loopRounds] at hi
  | succ r ih =>
    have hpre := loopRounds_prefix settings O D Dsf draws r
    simp only [loopRounds_succ] at hi ⊢
    split_ifs at hi ⊢ with hc
    · simp only [List.length_append, List.length_singleton] at hi
      rw [List.getElem_append_left (by omega)]
      exact hc.2 _ (List.getElem_mem _)
    · exact ih hi

/-- Round `i` is run on the harvests of the rounds before it, from round `i`'s draws. -/
lemma getElem_loopRounds (r i : ℕ) (hi : i < (loopRounds settings O D Dsf draws r).length) :
    ∃ hK : i < K, (loopRounds settings O D Dsf draws r)[i] = loopRound settings O D Dsf
      (histOf D settings.level3.walkStart ((loopRounds settings O D Dsf draws r).take i))
        (draws ⟨i, hK⟩) := by
  induction r with
  | zero => simp [loopRounds] at hi
  | succ r ih =>
    simp only [loopRounds_succ] at hi ⊢
    split_ifs at hi ⊢ with hc
    · simp only [List.length_append, List.length_singleton] at hi
      have hlen := length_eq_of_allHarvested settings O D Dsf draws (r := r) hc.1.le hc.2
      rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi' | hi'
      · obtain ⟨hK, e⟩ := ih hi'
        refine ⟨hK, ?_⟩
        rw [List.getElem_append_left hi', List.take_append_of_le_length hi'.le]
        exact e
      · refine ⟨by omega, ?_⟩
        rw [List.getElem_append_right (by omega)]
        simp only [hi', Nat.sub_self, List.getElem_singleton]
        rw [List.take_append_of_le_length le_rfl, List.take_length]
        congr 2
        exact Fin.ext hlen.symm
    · exact ih hi

/-- At `K` rounds the loop has run all `K`, or its last round did not harvest. -/
lemma loopRounds_stops :
    (loopRounds settings O D Dsf draws K).length = K
      ∨ ∃ o ∈ (loopRounds settings O D Dsf draws K).getLast?, o.result.ending.harvested = none := by
  by_cases h : AllHarvested (loopRounds settings O D Dsf draws K)
  · exact Or.inl (length_eq_of_allHarvested settings O D Dsf draws le_rfl h)
  · right
    simp only [AllHarvested, not_forall, not_not] at h
    obtain ⟨o, ho, hn⟩ := h
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
    have hlast : i + 1 = (loopRounds settings O D Dsf draws K).length := by
      by_contra hne
      exact harvested_of_lt settings O D Dsf draws K i (by omega) hn
    refine ⟨_, ?_, hn⟩
    rw [List.getLast?_eq_getElem?, ← hlast]
    simp

/-- The first `r` rounds are run on the first `r` rounds' draws. -/
lemma loopRounds_congr (draws' : Fin K → LoopRoundDraws Ω α K settings.level3.probeBudget) :
    ∀ r : ℕ, (∀ i : Fin K, i.val < r → draws i = draws' i) →
      loopRounds settings O D Dsf draws r = loopRounds settings O D Dsf draws' r
  | 0, _ => rfl
  | r + 1, h => by
    rw [loopRounds_succ settings O D Dsf draws r, loopRounds_succ settings O D Dsf draws' r,
      loopRounds_congr draws' r (fun i hi => h i (by omega))]
    split_ifs with hc
    · rw [h ⟨r, hc.1⟩ (by simp)]
    · rfl

lemma take_loopRounds {r s : ℕ} (hrs : r ≤ s) :
    (loopRounds settings O D Dsf draws s).take r = loopRounds settings O D Dsf draws r := by
  induction s, hrs using Nat.le_induction with
  | base => exact List.take_of_length_le (length_loopRounds_le settings O D Dsf draws r)
  | succ s hrs ih =>
    rw [loopRounds_succ]
    split_ifs with hc
    · have hlen := length_eq_of_allHarvested settings O D Dsf draws hc.1.le hc.2
      rw [List.take_append_of_le_length (by omega), ih]
    · exact ih

omit [MeasurableSpace Ω] in
lemma harvestsOf_take_getD {k : ℕ} {rounds : List (LoopRound Ω α K)} {j i : ℕ} (hij : i < j)
    (hj : j ≤ rounds.length) (hall : AllHarvested (rounds.take j)) {ν : Measure (FreeMonoid α)}
    (hν : (rounds[i]'(by omega)).population D k = some ν) :
    (harvestsOf D k (rounds.take j)).getD i (Measure.dirac 1) = ν := by
  induction rounds generalizing j i with
  | nil => simp at hj; omega
  | cons o rounds ih =>
    obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
    have ho : o.result.ending.harvested ≠ none := hall o (by simp)
    obtain ⟨zo, hzo⟩ := Option.ne_none_iff_exists'.1 ho
    have hpo : o.population D k
        = some (popLaw (Random.harvestLaw o.read D k o.result.hypothesis zo)) := by
      simp [LoopRound.population, hzo]
    simp only [List.take_succ_cons, harvestsOf, List.filterMap_cons, hpo] at hν ⊢
    rcases i with _ | i
    · simp only [List.getElem_cons_zero, hpo, Option.some.injEq] at hν
      simp [hν]
    · simp only [List.getD_cons_succ]
      exact ih (j := j) (i := i) (by omega) (by simp at hj; omega)
        (fun o' ho' => hall o' (by simp [ho'])) hν

end OrthoDFA
