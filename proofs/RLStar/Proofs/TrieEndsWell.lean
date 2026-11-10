import RLStar.Proofs.TrieNoise

/-!
# The trie round ends well, given that its harvests yield

`level3TrieRoundCorrect_of_yields`: `Level3TrieYields` gives `Level3TrieRoundCorrect`. Its
`δ` splits as
- `δ / 8`: a consistent ending above its rate (`trieConsistent_le`);
- `δ / 4`: a harvest yielding less than `κ y` (`Level3TrieYields`);
- `δ / 8`: the light probes above their bound at some state and tree (`light_noise_le`);
- `δ / 8`: some string the round reads, or some heavy string, read undecided at a state read
  undecided at most `T₀` of the time (`heavy_le`).
-/

noncomputable section

namespace OrthoDFA

namespace Random

open MeasureTheory ProbabilityTheory Query
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (U : σ → ℝ) (C : Level3Settings) (y : ℝ) (cap : ℕ)

/-- The round, then the heavy strings of its tree. -/
def trieHeavyQ (k : ℕ) {P : ℕ} (draws : Fin P → FreeMonoid α) :
    Query (FreeMonoid α) ARU (FreeMonoid α → ARU) :=
  (level3TrieQ C y cap draws).bind fun r => Query.table (heavySet k r.hypothesis.tree).toList

/-- The value on the state's wrong side, and undecided where the state is read undecided at most
`t` of the time. -/
def badRead (t : ℝ) (z : FreeMonoid α) : Set ARU :=
  {v | v = (if side (M.eval z.toList) then .reject else .accept)
    ∨ (v = .undecided ∧ U (M.eval z.toList) ≤ t)}

omit [Fintype α] [DecidableEq α] [Fintype σ] in
theorem rightSide_notMem_badRead (t : ℝ) (z : FreeMonoid α) :
    (if side (M.eval z.toList) then ARU.accept else ARU.reject) ∉ badRead M side U t z := by
  unfold badRead
  cases side (M.eval z.toList) <;> simp

/-- The bound on the strings `trieHeavyQ` reads. -/
def heavyReads (L k : ℕ) (P : ℕ) : ℕ :=
  P * ((3 + Nat.clog 2 L) * (Fintype.card σ + 1))
    + Fintype.card σ * ((Fintype.card σ + 2) * Fintype.card α * (Fintype.card σ + 2))
    + (∑ j ∈ Finset.range k, Fintype.card α ^ j) * (Fintype.card σ + 1)

theorem card_reads_trieHeavyQ_le (k L : ℕ) {P : ℕ} (draws : Fin P → FreeMonoid α)
    (hL : ∀ i, (draws i).toList.length ≤ L) (t : ℝ) (o : FreeMonoid α → ARU)
    (ho : ∀ z ∈ (trieHeavyQ C y cap k draws).reads o, o z ∉ badRead M side U t z) :
    ((trieHeavyQ C y cap k draws).reads o).card ≤ heavyReads (σ := σ) (α := α) L k P := by
  classical
  set p := trieHeavyQ C y cap k draws
  set o' : FreeMonoid α → ARU := fun z =>
    if z ∈ p.reads o then o z else if side (M.eval z.toList) then .accept else .reject
  have hW : NoWrong M side o' := by
    intro z
    by_cases hz : z ∈ p.reads o
    · simp only [o', if_pos hz]
      exact fun h => ho z hz (Or.inl h)
    · simp only [o', if_neg hz]
      cases side (M.eval z.toList) <;> simp
  rw [← (eval_congr o (o' := o') p fun z hz => if_pos hz).2]
  simp only [p, trieHeavyQ, reads_bind, reads_table]
  refine (Finset.card_union_le _ _).trans ?_
  have h1 := card_reads_level3TrieQ_le C y cap o' hW draws hL
  have hT := (trie_final_tree C y cap o' hW draws).1
  simp only [indecisiveLStarLevel3Trie] at hT
  have h2 := (card_heavySet k ((level3TrieQ C y cap draws).eval o').hypothesis.tree).trans
    (Nat.mul_le_mul_left _ (mids_le_of_class hT))
  rw [Finset.toList_toFinset]
  have h3 : Kappa.probeSifts L ≤ 3 + Nat.clog 2 L := min_le_right _ _
  unfold heavyReads
  have := Nat.mul_le_mul_left P (Nat.mul_le_mul_right (Fintype.card σ + 1) h3)
  omega

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- Some string the round or the heavy strings read is read badly at `t` with chance at most `t`
times the strings read. -/
theorem heavy_le (R : ReadOracle μ M side U) (k L : ℕ) {P : ℕ} (draws : Fin P → FreeMonoid α)
    (hL : ∀ i, (draws i).toList.length ≤ L) {t : ℝ} (ht : 0 ≤ t) :
    μ {ω | ∃ z ∈ (trieHeavyQ C y cap k draws).reads (R.read · ω),
        R.read z ω ∈ badRead M side U t z}
      ≤ ENNReal.ofReal (t * heavyReads (σ := σ) (α := α) L k P) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  have : Countable ARU := by
    refine Function.Injective.countable (f := fun v : ARU => match v with
      | .accept => (0 : ℕ) | .reject => 1 | .undecided => 2) ?_
    intro a b h; cases a <;> cases b <;> simp_all
  refine query_reads_bad_le μ R.read R.measurable R.indep (badRead M side U t) t (fun z => ?_)
    (fun z => ⟨_, rightSide_notMem_badRead M side U t z⟩) _ _
    (card_reads_trieHeavyQ_le M side U C y cap k L draws hL t)
  have hw : μ.real {ω | R.read z ω = if side (M.eval z.toList) then .reject else .accept} = 0 := by
    simp [Measure.real, R.wrong z]
  by_cases hU : U (M.eval z.toList) ≤ t
  · calc _ ≤ μ.real ({ω | R.read z ω = if side (M.eval z.toList) then .reject else .accept}
          ∪ {ω | R.read z ω = .undecided}) :=
          measureReal_mono (fun ω hω => by
            rcases hω with h | ⟨h, -⟩
            · exact Or.inl h
            · exact Or.inr h) (measure_ne_top _ _)
      _ ≤ _ := measureReal_union_le _ _
      _ ≤ t := by rw [hw, R.undecided z, zero_add]; exact hU
  · calc _ ≤ μ.real {ω | R.read z ω = if side (M.eval z.toList) then .reject else .accept} :=
          measureReal_mono (fun ω hω => by
            rcases hω with h | ⟨-, h⟩
            · exact h
            · exact absurd h hU) (measure_ne_top _ _)
      _ ≤ t := by rw [hw]; exact ht

end Random

open MeasureTheory ProbabilityTheory Query Random
open scoped ENNReal

universe u v w

theorem level3TrieRoundCorrect_of_yields (hY : Level3TrieYields.{u, v, w}) :
    Level3TrieRoundCorrect.{u, v, w} := by
  obtain ⟨κ, hκ, hY⟩ := hY
  refine ⟨κ, hκ, ?_⟩
  intro α _ _ σ _ M side U Ω _ μ _ R P C y cap kw pc pw
  dsimp only
  intro hpw hpc hcap hδ hε0 hε1 hy0 hy4
  classical
  have hYb := hY M side U μ R P C y cap pc hpc hcap hδ hy0 hy4
  set nQ := Fintype.card σ
  set nα := Fintype.card α
  set δ := C.δ
  set ε := C.disagreementRate
  set y₁ := κ nα * y
  set K : ℕ := (1 + (nQ + 1) ^ 3 * nα) ^ nQ
  set η := pw * Real.log (8 * (nQ + 1) * K / δ)
  set N : ℕ := C.probeBudget * ((3 + Nat.clog 2 P.L) * (nQ + 1))
    + nQ * ((nQ + 2) * nα * (nQ + 2)) + (∑ j ∈ Finset.range kw, nα ^ j) * (nQ + 1)
  have hN : heavyReads (σ := σ) (α := α) P.L kw C.probeBudget = N := rfl
  set T₀ : ℝ := δ / (8 * (N + 1))
  set A := 8 * (P.L + 1) * (nQ + 1) / y₁ + 1 / T₀
  set D := P.D
  set L := P.L
  set read := R.read
  by_cases hδ1 : 1 ≤ δ
  · calc _ ≤ ∫⁻ _, 1 ∂μ := lintegral_mono fun _ => prob_le_one
      _ = 1 := by simp
      _ ≤ _ := by rw [← ENNReal.ofReal_one]; exact ENNReal.ofReal_le_ofReal hδ1
  push Not at hδ1
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  have hpw0 : 0 < pw := (prefixMax_pos D kw).trans_le (ciSup_le hpw)
  have hy₁ : 0 < y₁ := mul_pos (hκ nα) hy0
  have hnQ : (0 : ℝ) ≤ nQ := Nat.cast_nonneg _
  have hK1 : (1 : ℝ) ≤ K := by
    have : 1 ≤ K := Nat.one_le_iff_ne_zero.2 (by positivity)
    exact_mod_cast this
  have hη0 : 0 ≤ η := mul_nonneg hpw0.le (Real.log_nonneg (by
    rw [le_div_iff₀ hδ]; nlinarith))
  have hT₀ : 0 < T₀ := by positivity
  have hU0 : ∀ z : FreeMonoid α, 0 ≤ U (M.eval z.toList) := fun z => by
    rw [← R.undecided z]; exact measureReal_nonneg
  -- the light probes
  set Nl := {ω | ∃ q, 0 ≤ U q ∧ 8 * ((L + 1) * (nQ + 1) * U q) ≤ 1 ∧
      ∃ T ∈ (classSet nQ : Finset (DTree α)), 2 * ((L + 1) * (nQ + 1) * U q) + η
        < D.real {x | ∃ z ∈ pot kw x T, read z ω = .undecided ∧ U (M.eval z.toList) ≤ U q}}
  have hNl : μ Nl ≤ ENNReal.ofReal (δ / 8) := by
    refine (light_noise_le D M U read R.measurable R.indep R.undecided P.length_le hpw0
      hpw).trans (ENNReal.ofReal_le_ofReal ?_)
    have he : Real.exp (-(η / pw)) = δ / (8 * (nQ + 1) * K) := by
      rw [mul_div_cancel_left₀ _ hpw0.ne', Real.exp_neg, Real.exp_log (by positivity), inv_div]
    have hK : ((1 + (nQ + 1) ^ 3 * nα) ^ nQ : ℝ) = K := by simp [K]
    rw [hK, he]
    rw [show (nQ : ℝ) * K * (δ / (8 * (nQ + 1) * K)) = δ / 8 * (nQ / (nQ + 1)) by
      field_simp]
    have : (nQ : ℝ) / (nQ + 1) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
    nlinarith
  -- the reads on their state's wrong side
  set W := {ω | ¬ NoWrong M side (read · ω)}
  have hW : μ W = 0 := by
    have : W ⊆ ⋃ z, {ω | read z ω = if side (M.eval z.toList) then .reject else .accept} := by
      intro ω hω
      simp only [W, NoWrong, Set.mem_ofPred_eq, not_forall, not_not] at hω
      obtain ⟨z, hz⟩ := hω
      exact Set.mem_iUnion.2 ⟨z, hz⟩
    exact measure_mono_null this (measure_iUnion_null R.wrong)
  set B := toMeasurable μ (W ∪ Nl)
  -- the heavy strings
  set Pm := Measure.pi fun _ : Fin C.probeBudget => D
  set G : (Fin C.probeBudget → FreeMonoid α) → Set Ω := fun d => toMeasurable μ
    {ω | ∃ z ∈ (trieHeavyQ C y cap kw d).reads (read · ω), read z ω ∈ badRead M side U T₀ z}
  set f : Ω → ℝ≥0∞ := fun ω => ∑' d, Pm {d} * (G d).indicator 1 ω
  have hfd : ∀ d, Measurable fun ω => Pm {d} * (G d).indicator 1 ω := fun d =>
    (measurable_one.indicator (measurableSet_toMeasurable _ _)).const_mul _
  have hfm : Measurable f := Measurable.tsum hfd
  have hTN : T₀ * N ≤ δ / 8 := by
    have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg _
    rw [show T₀ * N = δ / 8 * (N / (N + 1)) by simp only [T₀]; field_simp]
    have : (N : ℝ) / (N + 1) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
    nlinarith
  have hf : ∫⁻ ω, f ω ∂μ ≤ ENNReal.ofReal (δ / 8) := by
    rw [show ∫⁻ ω, f ω ∂μ = ∑' d, ∫⁻ ω, Pm {d} * (G d).indicator 1 ω ∂μ from
      lintegral_tsum fun d => (hfd d).aemeasurable]
    have hd : ∀ d, ∫⁻ ω, Pm {d} * (G d).indicator 1 ω ∂μ ≤ Pm {d} * ENNReal.ofReal (δ / 8) := by
      intro d
      rw [lintegral_const_mul _ (measurable_one.indicator (measurableSet_toMeasurable _ _)),
        lintegral_indicator_one (measurableSet_toMeasurable _ _), measure_toMeasurable]
      by_cases hdL : ∀ i, (d i).toList.length ≤ L
      · gcongr
        exact (heavy_le M side U C y cap R kw L d hdL hT₀.le).trans
          (ENNReal.ofReal_le_ofReal (by rw [hN]; exact hTN))
      · push Not at hdL
        obtain ⟨i, hi⟩ := hdL
        have h0 : D {d i} = 0 := measure_mono_null (fun x hx => by
          simp only [Set.mem_singleton_iff] at hx
          subst hx
          simp only [Set.mem_ofPred_eq, not_le]
          exact hi) (ae_iff.1 P.length_le)
        have : Pm {d} = 0 := by
          simp only [Pm, Measure.pi_singleton]
          exact Finset.prod_eq_zero (Finset.mem_univ i) h0
        simp [this]
    calc _ ≤ ∑' d, Pm {d} * ENNReal.ofReal (δ / 8) := ENNReal.tsum_le_tsum hd
      _ = Pm Set.univ * ENNReal.ofReal (δ / 8) := by
          rw [ENNReal.tsum_mul_right, ← Measure.tsum_indicator_apply_singleton Pm Set.univ
            MeasurableSet.univ]
          simp
      _ = _ := by rw [measure_univ, one_mul]
  -- each read, the round ends well but on the probes in the named events
  set Y := fun ω => {draws : Fin C.probeBudget → FreeMonoid α |
    let r := indecisiveLStarLevel3Trie C y cap (read · ω) draws
    ∃ site, r.ending = .harvest site
      ∧ (trieHarvestLaw (read · ω) D r.expanded r.hypothesis site).real Set.univ
        < κ (Fintype.card α) * y}
  have hpt : ∀ ω, Pm {draws | let r := indecisiveLStarLevel3Trie C y cap (read · ω) draws
          r.ending ≠ .failed ∧
          ¬ EndsWell
            { target := M,
              undecided := U,
              harvestSlope := A,
              harvestSlack := η / y₁,
              probes := D,
              disagreeRate := ε,
              disagree := trieDisagreeSet (read · ω) r.expanded r.hypothesis,
              walkStart := 0,
              harvestLaw := trieHarvestLaw (read · ω) D r.expanded r.hypothesis,
              yield := y₁,
              ending := r.ending }}
        ≤ (ENNReal.ofReal (δ / 8) + B.indicator 1 ω + f ω) + Pm (Y ω) := by
    intro ω
    by_cases hω : ω ∈ B
    · rw [Set.indicator_of_mem hω, Pi.one_apply]
      exact prob_le_one.trans (le_add_right (le_add_right le_add_self))
    have hω' : ω ∉ W ∪ Nl := fun h => hω (subset_toMeasurable _ _ h)
    rw [Set.indicator_of_notMem hω, add_zero]
    simp only [Set.mem_union, not_or] at hω'
    obtain ⟨hωW, hωN⟩ := hω'
    simp only [W, Set.mem_ofPred_eq, not_not] at hωW
    set Gs := {d : Fin C.probeBudget → FreeMonoid α | ω ∈ G d}
    have hGs : Pm Gs ≤ f ω := by
      rw [← Measure.tsum_indicator_apply_singleton Pm Gs (Set.to_countable Gs).measurableSet]
      refine ENNReal.tsum_le_tsum fun d => ?_
      by_cases hd : d ∈ Gs
      · rw [Set.indicator_of_mem hd, Set.indicator_of_mem (show ω ∈ G d from hd), Pi.one_apply,
          mul_one]
      · rw [Set.indicator_of_notMem hd]
        exact bot_le
    set Cs := {draws : Fin C.probeBudget → FreeMonoid α |
      let r := indecisiveLStarLevel3Trie C y cap (read · ω) draws
      r.ending = .consistent
        ∧ C.disagreementRate < D.real (trieDisagreeSet (read · ω) r.expanded r.hypothesis)}
    have hCs := trieConsistent_le C y cap (read · ω) D hδ hε0 hε1
    refine le_trans (measure_mono (show _ ⊆ (Cs ∪ Gs) ∪ Y ω from fun d hd => ?_))
      ((measure_union_le _ _).trans (add_le_add ((measure_union_le _ _).trans
        (add_le_add hCs hGs)) le_rfl))
    obtain ⟨hfail, hnot⟩ := hd
    by_contra hout
    simp only [Set.mem_union, not_or] at hout
    obtain ⟨⟨hdC, hdG⟩, hdY⟩ := hout
    set r := indecisiveLStarLevel3Trie C y cap (read · ω) d with hr
    cases he : r.ending with
    | failed => exact hfail he
    | consistent =>
      simp only [EndsWell, he, not_le] at hnot
      exact hdC ⟨he, hnot⟩
    | harvest site =>
      simp only [EndsWell, he] at hnot
      set ν := trieHarvestLaw (read · ω) D r.expanded r.hypothesis site
      have hνy : y₁ ≤ ν.real Set.univ := not_lt.1 fun h => hdY ⟨site, he, h⟩
      have hν1 : ν.real Set.univ ≤ 1 := trieHarvestLaw_univ_le _ D _ _ _
      refine hnot ⟨sublevel_of_states M U ν (by positivity) (by positivity) hU0 fun q hq0 => ?_,
        trieHarvestLaw_spread _ D _ _ _, hνy⟩
      have hAy : 8 * (L + 1) * (nQ + 1) ≤ A * y₁ := by
        have : 0 ≤ 1 / T₀ * y₁ := by positivity
        have e : 8 * ((L : ℝ) + 1) * (nQ + 1) / y₁ * y₁ = 8 * (L + 1) * (nQ + 1) :=
          div_mul_cancel₀ _ hy₁.ne'
        simp only [A, add_mul]
        linarith
      have hsl : 0 ≤ η / y₁ := by positivity
      by_cases hq : U q ≤ T₀ ∧ 8 * ((L + 1) * (nQ + 1) * U q) ≤ 1
      · have hheavy : ∀ z ∈ heavySet kw r.hypothesis.tree,
            ¬ (read z ω = .undecided ∧ U (M.eval z.toList) ≤ U q) := by
          rintro z hz ⟨hund, hzU⟩
          refine hdG (subset_toMeasurable _ _ ⟨z, ?_, Or.inr ⟨hund, hzU.trans hq.1⟩⟩)
          simp only [trieHeavyQ, reads_bind, reads_table, Finset.mem_union, Finset.mem_toList,
            List.mem_toFinset]
          exact Or.inr hz
        have hT := (trie_final_tree C y cap (read · ω) hωW d).1
        have hlow := trieHarvestLaw_low_le (read · ω) D r.expanded r.hypothesis site M U kw (U q)
          hheavy
        have hlight : D.real {x | ∃ z ∈ pot kw x r.hypothesis.tree, read z ω = .undecided
            ∧ U (M.eval z.toList) ≤ U q} ≤ 2 * ((L + 1) * (nQ + 1) * U q) + η := by
          by_contra hgt
          exact hωN ⟨q, hq0, hq.2, _, hT, lt_of_not_ge hgt⟩
        calc _ ≤ 2 * ((L + 1) * (nQ + 1) * U q) + η := hlow.trans hlight
          _ ≤ (A * U q + η / y₁) * y₁ := by
              rw [show (A * U q + η / y₁) * y₁ = A * y₁ * U q + η by
                rw [add_mul, div_mul_cancel₀ _ hy₁.ne']; ring]
              have h0 : 0 ≤ ((L : ℝ) + 1) * (nQ + 1) * U q := by positivity
              nlinarith [mul_le_mul_of_nonneg_right hAy hq0]
          _ ≤ _ := mul_le_mul_of_nonneg_left hνy (by positivity)
      · have hA1 : 1 ≤ A * U q := by
          rcases not_and_or.1 hq with h | h
          · push Not at h
            have : U q / T₀ ≥ 1 := by rw [ge_iff_le, le_div_iff₀ hT₀]; linarith
            have h8 : 0 ≤ 8 * ((L : ℝ) + 1) * (nQ + 1) / y₁ * U q := by positivity
            simp only [A, add_mul]
            rw [one_div_mul_eq_div]
            linarith
          · push Not at h
            have hy1 : y₁ ≤ 1 := hνy.trans hν1
            have : 8 * ((L : ℝ) + 1) * (nQ + 1) * U q ≤ A * y₁ * U q :=
              mul_le_mul_of_nonneg_right hAy hq0
            nlinarith
        calc _ ≤ ν.real Set.univ := measureReal_mono (Set.subset_univ _) (measure_ne_top _ _)
          _ ≤ (A * U q + η / y₁) * ν.real Set.univ := by
              have : 0 ≤ ν.real Set.univ := measureReal_nonneg
              nlinarith
  calc _ ≤ ∫⁻ ω, ((ENNReal.ofReal (δ / 8) + B.indicator 1 ω + f ω) + Pm (Y ω)) ∂μ :=
        lintegral_mono hpt
    _ = ENNReal.ofReal (δ / 8) + μ B + ∫⁻ ω, f ω ∂μ + ∫⁻ ω, Pm (Y ω) ∂μ := by
        have hBm : Measurable fun ω => ENNReal.ofReal (δ / 8) + B.indicator 1 ω :=
          (measurable_const (a := ENNReal.ofReal (δ / 8))).add
            (measurable_one.indicator (measurableSet_toMeasurable μ (W ∪ Nl)))
        have e1 : ∫⁻ ω, ((ENNReal.ofReal (δ / 8) + B.indicator 1 ω + f ω) + Pm (Y ω)) ∂μ
            = ∫⁻ ω, (ENNReal.ofReal (δ / 8) + B.indicator 1 ω + f ω) ∂μ + ∫⁻ ω, Pm (Y ω) ∂μ :=
          lintegral_add_left (hBm.add hfm) _
        have e2 : ∫⁻ ω, (ENNReal.ofReal (δ / 8) + B.indicator 1 ω + f ω) ∂μ
            = ∫⁻ ω, (ENNReal.ofReal (δ / 8) + B.indicator 1 ω) ∂μ + ∫⁻ ω, f ω ∂μ :=
          lintegral_add_left hBm _
        have e3 : ∫⁻ ω, (ENNReal.ofReal (δ / 8) + B.indicator 1 ω) ∂μ
            = ENNReal.ofReal (δ / 8) + μ B := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one,
            lintegral_indicator_one (measurableSet_toMeasurable _ _)]
        rw [e1, e2, e3]
    _ ≤ ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8) + ENNReal.ofReal (δ / 8)
        + ENNReal.ofReal (δ / 4) := by
        gcongr
        · rw [measure_toMeasurable]
          exact (measure_union_le _ _).trans (by rw [hW, zero_add]; exact hNl)
    _ ≤ _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity)]
        exact ENNReal.ofReal_le_ofReal (by linarith)

end OrthoDFA

end
