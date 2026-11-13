import RLStar.Proofs.LoopIdeal

/-!
# The loop's rounds

Round `i` is run on the populations the rounds before it harvested, every one of which harvested,
and the loop stops growing at the first round that does not harvest or at `K` rounds.
-/

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} {K : ℕ} (settings : LoopSettings K) (O : Oracle μ (FreeMonoid α))
  (D Dsf : Measure (FreeMonoid α)) (draws : Fin K → LoopRoundDraws Ω α settings)

/-- Round `r`'s clustering run on the filled draws. -/
noncomputable def idealRunAt (r : Fin K) : Run Ω (FreeMonoid α) (Option (Fin K)) :=
  idealRun D Dsf
    (specHistOf (idealRounds settings O D Dsf draws r)) ((draws r).1, (draws r).2.1)

/-- Every round harvested. -/
abbrev AllHarvested (rounds : List (SpecRound α)) : Prop :=
  ∀ o ∈ rounds, o.result.ending.IsHarvest

lemma idealRounds_succ (r : ℕ) :
    idealRounds settings O D Dsf draws (r + 1) =
      if h : r < K ∧ AllHarvested (idealRounds settings O D Dsf draws r) then
        idealRounds settings O D Dsf draws r
          ++ [idealRound settings O D Dsf
            (specHistOf (idealRounds settings O D Dsf draws r))
            (draws ⟨r, h.1⟩)]
      else idealRounds settings O D Dsf draws r := by
  rw [idealRounds]

lemma length_idealRounds_le (r : ℕ) : (idealRounds settings O D Dsf draws r).length ≤ r := by
  induction r with
  | zero => simp [idealRounds]
  | succ r ih =>
    rw [idealRounds_succ]
    split_ifs
    · simp; omega
    · omega

lemma length_eq_of_allHarvested {r : ℕ} (hr : r ≤ K)
    (h : AllHarvested (idealRounds settings O D Dsf draws r)) :
    (idealRounds settings O D Dsf draws r).length = r := by
  induction r with
  | zero => simp [idealRounds]
  | succ r ih =>
    rw [idealRounds_succ] at h ⊢
    split_ifs at h ⊢ with hc
    · have hall : AllHarvested (idealRounds settings O D Dsf draws r) :=
        fun o ho => h o (by simp [ho])
      simp [ih (by omega) hall]
    · exact absurd ⟨by omega, h⟩ hc

lemma idealRounds_prefix (r : ℕ) :
    idealRounds settings O D Dsf draws r <+: idealRounds settings O D Dsf draws (r + 1) := by
  rw [idealRounds_succ]
  split_ifs
  · exact List.prefix_append _ _
  · exact List.prefix_refl _

/-- Every round but the last harvested. -/
lemma harvested_of_lt (r i : ℕ) (hi : i + 1 < (idealRounds settings O D Dsf draws r).length) :
    (idealRounds settings O D Dsf draws r)[i].result.ending.IsHarvest := by
  induction r with
  | zero => simp [idealRounds] at hi
  | succ r ih =>
    have hpre := idealRounds_prefix settings O D Dsf draws r
    simp only [idealRounds_succ] at hi ⊢
    split_ifs at hi ⊢ with hc
    · simp only [List.length_append, List.length_singleton] at hi
      rw [List.getElem_append_left (by omega)]
      exact hc.2 _ (List.getElem_mem _)
    · exact ih hi

/-- Round `i` is run on the harvests of the rounds before it, from round `i`'s draws. -/
lemma getElem_idealRounds (r i : ℕ) (hi : i < (idealRounds settings O D Dsf draws r).length) :
    ∃ hK : i < K, (idealRounds settings O D Dsf draws r)[i] = idealRound settings O D Dsf
      (specHistOf ((idealRounds settings O D Dsf draws r).take i))
        (draws ⟨i, hK⟩) := by
  induction r with
  | zero => simp [idealRounds] at hi
  | succ r ih =>
    simp only [idealRounds_succ] at hi ⊢
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
lemma idealRounds_stops :
    (idealRounds settings O D Dsf draws K).length = K
      ∨ ∃ o ∈ (idealRounds settings O D Dsf draws K).getLast?,
        ¬ o.result.ending.IsHarvest := by
  by_cases h : AllHarvested (idealRounds settings O D Dsf draws K)
  · exact Or.inl (length_eq_of_allHarvested settings O D Dsf draws le_rfl h)
  · right
    simp only [AllHarvested, not_forall, not_not] at h
    obtain ⟨o, ho, hn⟩ := h
    obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem ho
    have hlast : i + 1 = (idealRounds settings O D Dsf draws K).length := by
      by_contra hne
      exact hn (harvested_of_lt settings O D Dsf draws K i (by omega))
    refine ⟨_, ?_, hn⟩
    rw [List.getLast?_eq_getElem?, ← hlast]
    simp

/-- The first `r` rounds are run on the first `r` rounds' draws. -/
lemma idealRounds_congr (draws' : Fin K → LoopRoundDraws Ω α settings) :
    ∀ r : ℕ, (∀ i : Fin K, i.val < r → draws i = draws' i) →
      idealRounds settings O D Dsf draws r = idealRounds settings O D Dsf draws' r
  | 0, _ => rfl
  | r + 1, h => by
    rw [idealRounds_succ settings O D Dsf draws r, idealRounds_succ settings O D Dsf draws' r,
      idealRounds_congr draws' r (fun i hi => h i (by omega))]
    split_ifs with hc
    · rw [h ⟨r, hc.1⟩ (by simp)]
    · rfl

lemma take_idealRounds {r s : ℕ} (hrs : r ≤ s) :
    (idealRounds settings O D Dsf draws s).take r = idealRounds settings O D Dsf draws r := by
  induction s, hrs using Nat.le_induction with
  | base => exact List.take_of_length_le (length_idealRounds_le settings O D Dsf draws r)
  | succ s hrs ih =>
    rw [idealRounds_succ]
    split_ifs with hc
    · have hlen := length_eq_of_allHarvested settings O D Dsf draws hc.1.le hc.2
      rw [List.take_append_of_le_length (by omega), ih]
    · exact ih

omit [MeasurableSpace Ω] [DecidableEq α] in
lemma pops_take_getD {rounds : List (SpecRound α)} {j i : ℕ} (hij : i < j)
    (hj : j ≤ rounds.length) (hall : AllHarvested (rounds.take j)) {ℓ : List (FreeMonoid α)}
    (hℓ : (rounds[i]'(by omega)).population = some ℓ) :
    (specHistOf (rounds.take j)).pops.getD i [] = ℓ := by
  sorry -- closes by: main's harvestsOf_take_getD, induction on rounds

end OrthoDFA
