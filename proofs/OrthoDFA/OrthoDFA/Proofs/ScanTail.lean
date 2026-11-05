import OrthoDFA.Proofs.LoopScan
import OrthoDFA.Proofs.RawEvents
import OrthoDFA.Proofs.Coins

/-!
# The scan's tail

Fewer than `T` distinct values among `M` i.i.d. draws, each value of mass at most `ρ`, has chance
at most `C(M, T) (Tρ)^(M-T)` (`measureReal_card_image_lt`): some `T` of the draws then take every
value the draws take.  Where each population's two sides offer `(2 + |pops|) NP` distinct strings
among the first `M` draws, every state reading at most `NP` prefixes scans within `M`
(`scanWithin_of_card`); a harvested population's sides split its one stream by the hash, which
past `T ≥ 4 (2 + |pops|) NP` distinct strings leaves a side short with chance at most
`2 exp(-T/8)` (`runMeasure_not_scanWithin_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory ProbabilityTheory Thin
open scoped ENNReal

section Distinct

variable {S : Type*} [Stringlike S]

/-- Fewer than `T ≤ M` values among `M` draws are all taken at some `T` of them. -/
lemma exists_card_eq_of_card_image_lt {M T : ℕ} (hT : T ≤ M) (y : Fin M → S)
    (h : (Finset.univ.image y).card < T) :
    ∃ F ∈ (Finset.univ : Finset (Fin M)).powersetCard T, ∀ i, y i ∈ F.image y := by
  classical
  have : Nonempty (Fin M) := ⟨⟨0, by omega⟩⟩
  have hpick : ∀ s ∈ Finset.univ.image y, ∃ i, y i = s := fun s hs => by simpa using hs
  choose! pick hpk using hpick
  obtain ⟨F, hF0, -, hF⟩ := Finset.exists_subsuperset_card_eq
    (Finset.subset_univ ((Finset.univ.image y).image pick))
    (Finset.card_image_le.trans h.le) (by simpa using hT)
  refine ⟨F, Finset.mem_powersetCard.2 ⟨Finset.subset_univ _, hF⟩, fun i => ?_⟩
  have hi : y i ∈ Finset.univ.image y := Finset.mem_image_of_mem y (Finset.mem_univ i)
  exact Finset.mem_image.2 ⟨pick (y i), hF0 (Finset.mem_image_of_mem pick hi), hpk _ hi⟩

/-- The chance every draw takes a value the draws at `F` take:

    P(∀ i, y i ∈ y(F)) ≤ (|F| ρ)^(M - |F|). -/
lemma pi_forall_mem_image_le (ν : Measure S) [IsProbabilityMeasure ν] {ρ : ℝ} (hρ0 : 0 ≤ ρ)
    (hρ : ∀ s, ν.real {s} ≤ ρ) {M : ℕ} (F : Finset (Fin M)) :
    (Measure.pi fun _ : Fin M => ν) {y | ∀ i, y i ∈ F.image y}
      ≤ ENNReal.ofReal ((F.card * ρ) ^ (M - F.card)) := by
  classical
  set P := Measure.pi fun _ : Fin M => ν
  set c : ℝ≥0∞ := ENNReal.ofReal (F.card * ρ)
  let Q : (F → S) → Set (Fin M → S) := fun v =>
    Set.univ.pi fun i => if h : i ∈ F then {v ⟨i, h⟩} else Set.univ
  let R : (F → S) → Set (Fin M → S) := fun v =>
    Set.univ.pi fun i => if h : i ∈ F then {v ⟨i, h⟩} else Set.range v
  have hsub : {y : Fin M → S | ∀ i, y i ∈ F.image y} ⊆ ⋃ v, R v := by
    intro y hy
    refine Set.mem_iUnion.2 ⟨fun f => y f.1, fun i _ => ?_⟩
    by_cases h : i ∈ F
    · simp only [h, dif_pos, Set.mem_singleton_iff]
    · obtain ⟨j, hj, hji⟩ := Finset.mem_image.1 (hy i)
      simp only [h, dif_neg, not_false_eq_true]
      exact ⟨⟨j, hj⟩, hji⟩
  have hrange : ∀ v : F → S, ν (Set.range v) ≤ c := by
    intro v
    calc ν (Set.range v) = ν (⋃ f, {v f}) := by rw [← Set.range_eq_iUnion]
      _ ≤ ∑ f, ν {v f} := measure_iUnion_fintype_le ν _
      _ ≤ ∑ _f : F, ENNReal.ofReal ρ := Finset.sum_le_sum fun f _ => by
          rw [← ofReal_measureReal]; exact ENNReal.ofReal_le_ofReal (hρ _)
      _ = c := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_coe, nsmul_eq_mul,
            ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]
  have hcompl : (Finset.univ.filter fun i : Fin M => ¬ i ∈ F).card = M - F.card := by
    rw [Finset.filter_not, Finset.filter_mem_eq_inter, Finset.univ_inter, Finset.card_sdiff,
      Finset.card_univ, Fintype.card_fin]
    · simp
  have hR : ∀ v, P (R v) ≤ P (Q v) * c ^ (M - F.card) := by
    intro v
    simp only [P, R, Q, Measure.pi_pi]
    have hi : ∀ i, ν (if h : i ∈ F then {v ⟨i, h⟩} else Set.range v)
        ≤ ν (if h : i ∈ F then {v ⟨i, h⟩} else Set.univ) * (if i ∈ F then 1 else c) := by
      intro i
      by_cases h : i ∈ F
      · simp [h]
      · simp only [h, dif_neg, not_false_eq_true, if_false, measure_univ, one_mul]
        exact hrange v
    calc ∏ i, ν (if h : i ∈ F then {v ⟨i, h⟩} else Set.range v)
        ≤ ∏ i, (ν (if h : i ∈ F then {v ⟨i, h⟩} else Set.univ) * (if i ∈ F then 1 else c)) :=
          Finset.prod_le_prod' fun i _ => hi i
      _ = (∏ i, ν (if h : i ∈ F then {v ⟨i, h⟩} else Set.univ)) * c ^ (M - F.card) := by
          rw [Finset.prod_mul_distrib, Finset.prod_ite, Finset.prod_const_one, one_mul,
            Finset.prod_const, hcompl]
  have hQd : Pairwise (Function.onFun Disjoint Q) := by
    intro v w hvw
    obtain ⟨f, hf⟩ := Function.ne_iff.1 hvw
    refine Set.disjoint_left.2 fun y hy hy' => hf ?_
    have h1 := hy f.1 (Set.mem_univ _)
    have h2 := hy' f.1 (Set.mem_univ _)
    dsimp only at h1 h2
    rw [dif_pos f.2] at h1 h2
    exact (Set.mem_singleton_iff.1 h1).symm.trans (Set.mem_singleton_iff.1 h2)
  have hQm : ∀ v, MeasurableSet (Q v) := fun v =>
    MeasurableSet.univ_pi fun i => by
      by_cases h : i ∈ F
      · rw [dif_pos h]; exact measurableSet_singleton _
      · rw [dif_neg h]; exact MeasurableSet.univ
  have hQ1 : ∑' v, P (Q v) ≤ 1 := by
    rw [← measure_iUnion hQd hQm]; exact prob_le_one
  calc P {y | ∀ i, y i ∈ F.image y} ≤ P (⋃ v, R v) := measure_mono hsub
    _ ≤ ∑' v, P (R v) := measure_iUnion_le _
    _ ≤ ∑' v, P (Q v) * c ^ (M - F.card) := ENNReal.tsum_le_tsum hR
    _ = (∑' v, P (Q v)) * c ^ (M - F.card) := ENNReal.tsum_mul_right
    _ ≤ 1 * c ^ (M - F.card) := by gcongr
    _ = ENNReal.ofReal ((F.card * ρ) ^ (M - F.card)) := by
        rw [one_mul, ENNReal.ofReal_pow (by positivity)]

/-- Fewer than `T ≤ M` distinct values among `M` i.i.d. draws, each value of mass at most `ρ`:

    P(|{y i}| < T) ≤ C(M, T) (Tρ)^(M-T). -/
theorem measureReal_card_image_lt (ν : Measure S) [IsProbabilityMeasure ν] {ρ : ℝ}
    (hρ : ∀ s, ν.real {s} ≤ ρ) {M T : ℕ} (hT : T ≤ M) :
    (Measure.pi fun _ : Fin M => ν).real {y | (Finset.univ.image y).card < T}
      ≤ M.choose T * (T * ρ) ^ (M - T) := by
  classical
  have hρ0 : 0 ≤ ρ := le_trans measureReal_nonneg (hρ 1)
  have hsub : {y : Fin M → S | (Finset.univ.image y).card < T}
      ⊆ ⋃ F ∈ (Finset.univ : Finset (Fin M)).powersetCard T, {y | ∀ i, y i ∈ F.image y} := by
    intro y hy
    obtain ⟨F, hF, h⟩ := exists_card_eq_of_card_image_lt hT y hy
    exact Set.mem_biUnion hF h
  rw [measureReal_def]
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) ?_
  calc _ ≤ _ := measure_mono hsub
    _ ≤ ∑ F ∈ (Finset.univ : Finset (Fin M)).powersetCard T,
          (Measure.pi fun _ : Fin M => ν) {y | ∀ i, y i ∈ F.image y} :=
        measure_biUnion_finset_le _ _
    _ ≤ ∑ _F ∈ (Finset.univ : Finset (Fin M)).powersetCard T,
          ENNReal.ofReal ((T * ρ) ^ (M - T)) := Finset.sum_le_sum fun F hF => by
        have h := pi_forall_mem_image_le ν hρ0 hρ F
        rwa [(Finset.mem_powersetCard.1 hF).2] at h
    _ = ENNReal.ofReal (M.choose T * (T * ρ) ^ (M - T)) := by
        rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_mul (by positivity)]

lemma choose_mul_pow_le {M T : ℕ} {ρ : ℝ} (hρ0 : 0 ≤ ρ) (hM : 2 * T ≤ M)
    (hρ : 16 * T * ρ ≤ 1) : M.choose T * (T * ρ) ^ (M - T) ≤ (1 / 2) ^ M := by
  have h1 : (M.choose T : ℝ) ≤ 2 ^ M := by exact_mod_cast Nat.choose_le_two_pow M T
  have h2 : (T * ρ) ^ (M - T) ≤ (1 / 16 : ℝ) ^ (M - T) :=
    pow_le_pow_left₀ (by positivity) (by linarith) _
  have h3 : (1 / 16 : ℝ) ^ (M - T) ≤ (1 / 4) ^ M := by
    rw [show (1 / 16 : ℝ) = (1 / 4) ^ 2 by norm_num, ← pow_mul]
    exact pow_le_pow_of_le_one (by norm_num) (by norm_num) (by omega)
  calc _ ≤ (2 : ℝ) ^ M * (1 / 4) ^ M :=
        mul_le_mul h1 (h2.trans h3) (by positivity) (by positivity)
    _ = (1 / 2) ^ M := by rw [← mul_pow]; norm_num

end Distinct

section Streams

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S] {J : Type*} [Fintype J]

lemma measurableSet_card_image_lt (M T : ℕ) :
    MeasurableSet {y : Fin M → S | (Finset.univ.image y).card < T} :=
  (Set.to_countable _).measurableSet

/-- Fewer than `T ≤ M` distinct strings among the first `M` of population `j`'s table draws. -/
lemma runMeasure_table_lt (L : J → Measure S) [∀ j, IsProbabilityMeasure (L j)] (Dsf : Measure S)
    [IsProbabilityMeasure Dsf] (j : J) {ρ : ℝ} (hρ : ∀ s, (L j).real {s} ≤ ρ) {M T : ℕ}
    (hT : T ≤ M) :
    (runMeasure μ L Dsf).real
        {x | (Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).card < T}
      ≤ M.choose T * (T * ρ) ^ (M - T) := by
  have hm : Measurable (fun x : Run Ω S J => fun i : Fin M => prefixDraw j i.val x) :=
    measurable_pi_lambda _ fun i => measurable_prf j i.val
  refine runMeasure_base_le L Dsf _ (hm (measurableSet_card_image_lt M T)) ?_
  rw [measureReal_def, ← Measure.map_apply hm (measurableSet_card_image_lt M T),
    map_prefixBlock _ _ Dsf j (L j) rfl M, ← measureReal_def]
  exact measureReal_card_image_lt (L j) hρ hT

/-- Fewer than `T ≤ M` distinct strings among the first `M` of population `j`'s certification
draws. -/
lemma runMeasure_cert_lt (L : J → Measure S) [∀ j, IsProbabilityMeasure (L j)] (Dsf : Measure S)
    [IsProbabilityMeasure Dsf] (j : J) {ρ : ℝ} (hρ : ∀ s, (L j).real {s} ≤ ρ) {M T : ℕ}
    (hT : T ≤ M) :
    (runMeasure μ L Dsf).real
        {x | (Finset.univ.image fun i : Fin M => x.1.2.2 j i).card < T}
      ≤ M.choose T * (T * ρ) ^ (M - T) := by
  have hm : Measurable (fun x : Run Ω S J => fun i : Fin M => certPrefix j i.val x) :=
    measurable_pi_lambda _ fun i => measurable_cert j i.val
  refine runMeasure_base_le L Dsf _ (hm (measurableSet_card_image_lt M T)) ?_
  rw [measureReal_def, ← Measure.map_apply hm (measurableSet_card_image_lt M T),
    map_certBlock _ _ Dsf j (L j) rfl M, ← measureReal_def]
  exact measureReal_card_image_lt (L j) hρ hT

omit [Stringlike S] in
lemma measurable_card_filter (F : Finset S) (b : Bool) :
    Measurable fun g : S → Bool => (F.filter fun s => g s = b).card := by
  simp only [Finset.card_filter]
  exact Finset.measurable_sum _ fun s _ =>
    Measurable.ite (measurableSet_eq_fun (measurable_pi_apply s) measurable_const)
      measurable_const measurable_const

omit [IsProbabilityMeasure μ] in
lemma measurableSet_sideShort {M T A : ℕ} (j : J) :
    MeasurableSet {x : Draws Ω S J | T ≤ (Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).card
      ∧ ¬ (A ≤ ((Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).filter
            fun s => x.2 s = true).card
        ∧ A ≤ ((Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).filter
            fun s => x.2 s = false).card)} := by
  classical
  have hslice : ∀ y : Fin M → S, Measurable fun g : S → Bool =>
      T ≤ (Finset.univ.image y).card ∧ ¬ (A ≤ ((Finset.univ.image y).filter fun s => g s = true).card
        ∧ A ≤ ((Finset.univ.image y).filter fun s => g s = false).card) := fun y =>
    measurable_const.and (Measurable.not
      ((measurableSet_setOfPred.1 (measurable_card_filter _ true (MeasurableSet.of_discrete
          (s := {n | A ≤ n})))).and
        (measurableSet_setOfPred.1 (measurable_card_filter _ false (MeasurableSet.of_discrete
          (s := {n | A ≤ n}))))))
  have hW : MeasurableSet {p : (Fin M → S) × (S → Bool) |
      T ≤ (Finset.univ.image p.1).card ∧ ¬ (A ≤ ((Finset.univ.image p.1).filter
        fun s => p.2 s = true).card ∧ A ≤ ((Finset.univ.image p.1).filter
        fun s => p.2 s = false).card)} :=
    measurableSet_setOfPred.2 (measurable_from_prod_countable_right fun y => hslice y)
  have hm : Measurable fun x : Draws Ω S J => ((fun i : Fin M => x.1.2.1.2 j i), x.2) := by
    fun_prop
  exact hm hW

/-- Past `T ≥ 4A` distinct strings among the first `M` of population `j`'s table draws, the hash
leaves either side fewer than `A` of them with chance at most `2 exp(-T/8)`. -/
lemma runMeasure_side_lt (L : J → Measure S) [∀ j, IsProbabilityMeasure (L j)] (Dsf : Measure S)
    [IsProbabilityMeasure Dsf] (j : J) {M T A : ℕ} (hA : 4 * A ≤ T) :
    (runMeasure μ L Dsf).real
        {x | T ≤ (Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).card
          ∧ ¬ (A ≤ ((Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).filter
                fun s => x.2 s = true).card
            ∧ A ≤ ((Finset.univ.image fun i : Fin M => x.1.2.1.2 j i).filter
                fun s => x.2 s = false).card)}
      ≤ 2 * Real.exp (-(T / 8 : ℝ)) := by
  classical
  set Bad : (Fin M → S) → (S → Bool) → Prop := fun y g =>
    T ≤ (Finset.univ.image y).card ∧ ¬ (A ≤ ((Finset.univ.image y).filter fun s => g s = true).card
      ∧ A ≤ ((Finset.univ.image y).filter fun s => g s = false).card) with hBad
  have hslice : ∀ y, Measurable fun g => Bad y g := fun y =>
    measurable_const.and (Measurable.not
      ((measurableSet_setOfPred.1 (measurable_card_filter _ true (MeasurableSet.of_discrete
          (s := {n | A ≤ n})))).and
        (measurableSet_setOfPred.1 (measurable_card_filter _ false (MeasurableSet.of_discrete
          (s := {n | A ≤ n}))))))
  have hW : MeasurableSet {p : (Fin M → S) × (S → Bool) | Bad p.1 p.2} :=
    measurableSet_setOfPred.2 (measurable_from_prod_countable_right fun y => hslice y)
  have hm : Measurable fun x : Draws Ω S J => ((fun i : Fin M => x.1.2.1.2 j i), x.2) := by
    fun_prop
  refine runMeasure_coins_le L Dsf _ (hm hW) (by positivity) fun x => ?_
  set F := Finset.univ.image fun i : Fin M => x.2.1.2 j i
  by_cases hTF : T ≤ F.card
  swap
  · have : {g : S → Bool | (x, g) ∈ ((fun x : Draws Ω S J => ((fun i : Fin M => x.1.2.1.2 j i),
        x.2)) ⁻¹' {p | Bad p.1 p.2})} = ∅ := by
      ext g; simp only [Set.mem_preimage, Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      exact fun h => hTF h.1
    rw [this, measureReal_empty]; positivity
  have hF : (T : ℝ) ≤ F.card := by exact_mod_cast hTF
  have hsub : {g : S → Bool | (x, g) ∈ ((fun x : Draws Ω S J => ((fun i : Fin M => x.1.2.1.2 j i),
        x.2)) ⁻¹' {p | Bad p.1 p.2})}
      ⊆ {g | (F.card : ℝ) * (1 / 2 + 1 / 4) ≤ ((F.filter fun s => g s = false).card : ℝ)}
        ∪ {g | (F.card : ℝ) * (1 / 2 + 1 / 4) ≤ ((F.filter fun s => g s = true).card : ℝ)} := by
    intro g hg
    obtain ⟨-, hg⟩ := hg
    have hsplit := Finset.card_filter_add_card_filter_not (s := F)
      (fun s => g s = true)
    have hneg : (F.filter fun s => ¬ g s = true) = F.filter fun s => g s = false :=
      Finset.filter_congr fun s _ => by simp
    rw [hneg] at hsplit
    have hA' : 4 * (A : ℝ) ≤ F.card := le_trans (by exact_mod_cast hA) hF
    by_cases ht : A ≤ (F.filter fun s => g s = true).card
    · have hf : (F.filter fun s => g s = false).card < A := by
        by_contra h; exact hg ⟨ht, not_lt.1 h⟩
      right
      have : ((F.filter fun s => g s = true).card : ℝ)
          = F.card - (F.filter fun s => g s = false).card := by
        rw [← hsplit]; push_cast; ring
      have hf' : ((F.filter fun s => g s = false).card : ℝ) ≤ A := by exact_mod_cast hf.le
      change (F.card : ℝ) * (1 / 2 + 1 / 4) ≤ _
      rw [this]; linarith
    · left
      have : ((F.filter fun s => g s = false).card : ℝ)
          = F.card - (F.filter fun s => g s = true).card := by
        rw [← hsplit]; push_cast; ring
      have ht' : ((F.filter fun s => g s = true).card : ℝ) ≤ A := by
        exact_mod_cast (not_le.1 ht).le
      change (F.card : ℝ) * (1 / 2 + 1 / 4) ≤ _
      rw [this]; linarith
  have hexp : Real.exp (-2 * (F.card : ℝ) * (1 / 4) ^ 2) ≤ Real.exp (-(T / 8 : ℝ)) :=
    Real.exp_le_exp.2 (by nlinarith)
  calc _ ≤ _ := measureReal_mono hsub
    _ ≤ _ := measureReal_union_le _ _
    _ ≤ Real.exp (-(T / 8 : ℝ)) + Real.exp (-(T / 8 : ℝ)) :=
        add_le_add ((coins_count_upper F false (by norm_num)).trans hexp)
          ((coins_count_upper F true (by norm_num)).trans hexp)
    _ = 2 * Real.exp (-(T / 8 : ℝ)) := by ring

end Streams

section Scan

variable {α : Type*} [Fintype α] {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
  [IsProbabilityMeasure μ] {K : ℕ}

omit [IsProbabilityMeasure μ] in
/-- Each population's sides offering `(2 + |pops|) NP` distinct strings, every state reading at
most `NP` prefixes scans within `M`. -/
lemma scanWithin_of_card {M NP : ℕ} (sched : Finset State) (pops : Finset (Option (Fin K)))
    (hNP : ∀ B ∈ sched, B.npref ≤ NP) (dr : Draws Ω (FreeMonoid α) (Option (Fin K)))
    (h : ∀ j ∈ pops, (2 + pops.card) * NP ≤ (seqsOf M dr j).1.toFinset.card
      ∧ (2 + pops.card) * NP ≤ (seqsOf M dr j).2.toFinset.card) :
    ScanWithin sched pops M dr := by
  intro B hB j hj
  obtain ⟨h1, h2⟩ := h j hj
  have hn := hNP B hB
  have hP := card_prefixesAt_le pops B.npref (streams (fun _ => ∅) none dr)
  have hsd := Finset.le_card_sdiff (prefixesAt pops B.npref (streams (fun _ => ∅) none dr))
    (seqsOf M dr j).2.toFinset
  have hm : pops.card * B.npref ≤ pops.card * NP := Nat.mul_le_mul_left _ hn
  rw [add_mul] at h1 h2
  generalize pops.card * B.npref = a at *
  generalize pops.card * NP = b at *
  exact ⟨by omega, fun e he => by omega⟩

omit [Fintype α] in
lemma toFinset_map_range {β : Type*} [DecidableEq β] (M : ℕ) (f : ℕ → β) :
    ((List.range M).map f).toFinset = Finset.univ.image fun i : Fin M => f i := by
  ext s
  simp only [List.mem_toFinset, List.mem_map, List.mem_range, Finset.mem_image, Finset.mem_univ,
    true_and]
  exact ⟨fun ⟨a, ha, e⟩ => ⟨⟨a, ha⟩, e⟩, fun ⟨a, e⟩ => ⟨a.1, a.2, e⟩⟩

/-- Population `j`'s sides fall short among its first `M` draws: fewer than `T` distinct table
draws; for the probes' population fewer than `T` distinct certification draws, for a harvested one
fewer than `A` on a side of the hash past `T` distinct. -/
def popShort (M T A : ℕ) : Option (Fin K) → Set (Draws Ω (FreeMonoid α) (Option (Fin K)))
  | none => {x | (Finset.univ.image fun i : Fin M => x.1.2.1.2 none i).card < T}
      ∪ {x | (Finset.univ.image fun i : Fin M => x.1.2.2 none i).card < T}
  | some i => {x | (Finset.univ.image fun i' : Fin M => x.1.2.1.2 (some i) i').card < T}
      ∪ {x | T ≤ (Finset.univ.image fun i' : Fin M => x.1.2.1.2 (some i) i').card
        ∧ ¬ (A ≤ ((Finset.univ.image fun i' : Fin M => x.1.2.1.2 (some i) i').filter
              fun s => x.2 s = true).card
          ∧ A ≤ ((Finset.univ.image fun i' : Fin M => x.1.2.1.2 (some i) i').filter
              fun s => x.2 s = false).card)}

/-- Some population's sides fall short. -/
def ScanShort (pops : Finset (Option (Fin K))) (M T A : ℕ)
    (dr : Draws Ω (FreeMonoid α) (Option (Fin K))) : Prop :=
  ∃ j ∈ pops, dr ∈ popShort M T A j

omit [IsProbabilityMeasure μ] in
/-- Where no population's sides fall short of `A`, `(2 + |pops|) NP ≤ A ≤ T`, every state reading
at most `NP` prefixes scans within `M`. -/
lemma scanWithin_of_not_scanShort {M T A NP : ℕ} (sched : Finset State)
    (pops : Finset (Option (Fin K))) (hNP : ∀ B ∈ sched, B.npref ≤ NP)
    (hNA : (2 + pops.card) * NP ≤ A) (hAT : A ≤ T) (dr : Draws Ω (FreeMonoid α) (Option (Fin K)))
    (h : ¬ ScanShort pops M T A dr) : ScanWithin sched pops M dr := by
  classical
  refine scanWithin_of_card sched pops hNP dr fun j hj => ?_
  have hj' : dr ∉ popShort M T A j := fun h' => h ⟨j, hj, h'⟩
  rcases j with _ | i
  · simp only [popShort, Set.mem_union, Set.mem_ofPred_eq, not_or, not_lt] at hj'
    simp only [seqsOf, toFinset_map_range]
    exact ⟨(hNA.trans hAT).trans hj'.1, (hNA.trans hAT).trans hj'.2⟩
  · simp only [popShort, Set.mem_union, Set.mem_ofPred_eq, not_or, not_lt] at hj'
    obtain ⟨h1, h2⟩ := hj'
    have h3 := not_not.1 (fun h => h2 ⟨h1, h⟩)
    simp only [seqsOf, List.toFinset_filter, toFinset_map_range]
    refine ⟨hNA.trans h3.1, ?_⟩
    have : ((Finset.univ.image fun i' : Fin M => dr.1.2.1.2 (some i) i').filter
        fun s => (!dr.2 s) = true)
        = (Finset.univ.image fun i' : Fin M => dr.1.2.1.2 (some i) i').filter
          fun s => dr.2 s = false := Finset.filter_congr fun s _ => by simp
    rw [this]; exact hNA.trans h3.2

omit [IsProbabilityMeasure μ] in
lemma measurableSet_popShort (M T A : ℕ) (j : Option (Fin K)) :
    MeasurableSet (popShort (Ω := Ω) (α := α) M T A j) := by
  classical
  have htab : ∀ j : Option (Fin K), Measurable fun x : Draws Ω (FreeMonoid α) (Option (Fin K)) =>
      fun i : Fin M => x.1.2.1.2 j i := fun j => by fun_prop
  rcases j with _ | i
  · have hcert : Measurable fun x : Draws Ω (FreeMonoid α) (Option (Fin K)) =>
        fun i : Fin M => x.1.2.2 none i := by fun_prop
    exact (htab none (measurableSet_card_image_lt M T)).union
      (hcert (measurableSet_card_image_lt M T))
  · refine (htab (some i) (measurableSet_card_image_lt M T)).union ?_
    exact measurableSet_sideShort (some i)

omit [IsProbabilityMeasure μ] in
lemma measurableSet_scanShort (pops : Finset (Option (Fin K))) (M T A : ℕ) :
    MeasurableSet {dr : Draws Ω (FreeMonoid α) (Option (Fin K)) | ScanShort pops M T A dr} := by
  have : {dr : Draws Ω (FreeMonoid α) (Option (Fin K)) | ScanShort pops M T A dr}
      = ⋃ j ∈ pops, popShort M T A j := by
    ext dr; simp [ScanShort]
  rw [this]
  exact Finset.measurableSet_biUnion _ fun j _ => measurableSet_popShort M T A j

/-- Some population's sides fall short, at the populations' laws `L`, every string holding at
most `ρ` of each, `4A ≤ T ≤ M`:

    P ≤ |pops| (2 C(M, T) (Tρ)^(M-T) + 2 exp(-T/8)). -/
theorem runMeasure_scanShort_le (L : Option (Fin K) → Measure (FreeMonoid α))
    [∀ j, IsProbabilityMeasure (L j)] (Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure Dsf]
    (pops : Finset (Option (Fin K))) {M T A : ℕ} {ρ : ℝ}
    (hρ : ∀ j ∈ pops, ∀ s, (L j).real {s} ≤ ρ) (hA : 4 * A ≤ T) (hT : T ≤ M) :
    (runMeasure μ L Dsf).real {dr | ScanShort pops M T A dr}
      ≤ pops.card * (2 * (M.choose T * (T * ρ) ^ (M - T)) + 2 * Real.exp (-(T / 8 : ℝ))) := by
  classical
  set q : ℝ := M.choose T * (T * ρ) ^ (M - T)
  have hρ0 : ∀ j ∈ pops, 0 ≤ ρ := fun j hj => le_trans measureReal_nonneg (hρ j hj 1)
  have hsub : {dr | ScanShort pops M T A dr}
      ⊆ ⋃ j ∈ pops, popShort (Ω := Ω) (α := α) M T A j := by
    rintro dr ⟨j, hj, h⟩; exact Set.mem_biUnion hj h
  have hj : ∀ j ∈ pops, (runMeasure μ L Dsf).real (popShort (Ω := Ω) (α := α) M T A j)
      ≤ 2 * q + 2 * Real.exp (-(T / 8 : ℝ)) := by
    intro j hj
    have hq0 : 0 ≤ q := by have := hρ0 j hj; positivity
    have he0 : 0 ≤ Real.exp (-(T / 8 : ℝ)) := (Real.exp_pos _).le
    rcases j with _ | i
    · refine (measureReal_union_le _ _).trans ?_
      have h1 := runMeasure_table_lt (μ := μ) L Dsf none (hρ none hj) hT
      have h2 := runMeasure_cert_lt (μ := μ) L Dsf none (hρ none hj) hT
      exact (add_le_add h1 h2).trans (by linarith)
    · refine (measureReal_union_le _ _).trans ?_
      have h1 := runMeasure_table_lt (μ := μ) L Dsf (some i) (hρ (some i) hj) hT
      have h2 := runMeasure_side_lt (μ := μ) L Dsf (some i) (M := M) hA
      exact (add_le_add h1 h2).trans (by linarith)
  calc _ ≤ (runMeasure μ L Dsf).real (⋃ j ∈ pops, popShort (Ω := Ω) (α := α) M T A j) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ j ∈ pops, (runMeasure μ L Dsf).real (popShort (Ω := Ω) (α := α) M T A j) :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _j ∈ pops, (2 * q + 2 * Real.exp (-(T / 8 : ℝ))) := Finset.sum_le_sum hj
    _ = _ := by rw [Finset.sum_const, nsmul_eq_mul]

/-- After the history `h`, some population's filled draws fall short with the chance of
`runMeasure_scanShort_le`, every string holding at most `ρ` of each population off the earlier
rounds' prefixes. -/
theorem clusterLaw_scanShort_le (D Dsf : Measure (FreeMonoid α)) [IsProbabilityMeasure D]
    (k N : ℕ) (h : LoopHist α) (pops : Finset (Option (Fin K))) {M T A : ℕ} {ρ : ℝ}
    (hρ : ∀ j ∈ pops, ∀ s, (offLaw (loopDistributions D k h.pops j) h.prefs).real {s} ≤ ρ)
    (hA : 4 * A ≤ T) (hT : T ≤ M) :
    (clusterLaw μ α K N).real {c | ScanShort pops M T A (idealDraws D Dsf k N h c)}
      ≤ pops.card * (2 * (M.choose T * (T * ρ) ^ (M - T)) + 2 * Real.exp (-(T / 8 : ℝ))) := by
  classical
  have hmp := measurePreserving_idealDraws (μ := μ) (K := K) D Dsf k N h
  have e := hmp.measureReal_preimage (measurableSet_scanShort pops M T A).nullMeasurableSet
  exact (le_of_eq e).trans (runMeasure_scanShort_le _ _ pops hρ hA hT)

end Scan

end OrthoDFA
