import OrthoDFA.ClusteringQuality
import OrthoDFA.Proofs.Budget

/-!
# The quality of the returned family

`clustering_correct` bounds what the family miscuts, or leaves undecided, under the one noise
draw the run used.  `quality` asks the same of fresh noise, one DFA state at a time.

The two meet because the family reads the noise only at the table's strings.  At any other
population prefix its verdict is a fresh draw, independent across prefixes, so a weighted
Hoeffding bound ties the realized mass to its mean; the table's own prefixes carry little mass
once the collision mass is small.  The mean groups by state because the vote's law at a prefix
depends only on the state it reaches.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {J : Type*} [Fintype J]

/-! ## The vote's law depends only on the state -/

lemma measureReal_mq_eq_one (O : Oracle μ S) (w : S) :
    μ.real {ω | O.mq w ω = 1} = μ[O.mq w] := by
  have hmeas : MeasurableSet {ω | O.mq w ω = 1} :=
    measurableSet_eq_fun (mq_meas O w) measurable_const
  have hae : O.mq w =ᵐ[μ] ({ω | O.mq w ω = 1} : Set Ω).indicator 1 := by
    filter_upwards [mq_bit O w] with ω hω
    rcases hω with h | h
    · simp [h]
    · simp [h]
  rw [integral_congr_ae hae, integral_indicator_one hmeas]

lemma measure_mq_eq_one_congr (O : Oracle μ S) {w w' : S} (h : O.label w = O.label w') :
    μ (O.mq w ⁻¹' {1}) = μ (O.mq w' ⁻¹' {1}) := by
  have hr : μ.real {ω | O.mq w ω = 1} = μ.real {ω | O.mq w' ω = 1} := by
    rw [measureReal_mq_eq_one, measureReal_mq_eq_one, mq_mean, mq_mean,
      O.rate_eq_of_label_eq h, h]
  rw [measureReal_def, measureReal_def] at hr
  exact (ENNReal.toReal_eq_toReal_iff' (measure_ne_top _ _) (measure_ne_top _ _)).1 hr

open scoped Classical in
/-- Two prefixes whose extensions carry the same labels read the same family's votes with the
same law. -/
lemma measureReal_filter_congr (O : Oracle μ S) (G : Finset S) {p p' : S}
    (hlab : ∀ v, O.label (p * v) = O.label (p' * v)) (Φ : Finset S → Prop) :
    μ.real {ω | Φ (G.filter (fun v => O.mq (p * v) ω = 1))}
      = μ.real {ω | Φ (G.filter (fun v => O.mq (p' * v) ω = 1))} := by
  classical
  have hfib : ∀ (r : S) (U : Finset S), U ⊆ G →
      {ω | G.filter (fun v => O.mq (r * v) ω = 1) = U}
        = ⋂ v ∈ G, (fun ω => O.mq (r * v) ω) ⁻¹'
            (if v ∈ U then ({1} : Set ℝ) else ({1} : Set ℝ)ᶜ) := by
    intro r U hU
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iInter, Set.mem_preimage]
    constructor
    · rintro rfl v hv
      by_cases h : O.mq (r * v) ω = 1 <;> simp [h, hv]
    · intro h
      ext v
      simp only [Finset.mem_filter]
      constructor
      · rintro ⟨hvG, h1⟩
        by_contra hvU
        have := h v hvG
        simp only [hvU, if_false, Set.mem_compl_iff, Set.mem_singleton_iff] at this
        exact this h1
      · intro hvU
        refine ⟨hU hvU, ?_⟩
        have := h v (hU hvU)
        simpa [hvU] using this
  have hprob : ∀ U ∈ G.powerset, μ {ω | G.filter (fun v => O.mq (p * v) ω = 1) = U}
      = μ {ω | G.filter (fun v => O.mq (p' * v) ω = 1) = U} := by
    intro U hU
    have hU' := Finset.mem_powerset.1 hU
    have hsets : ∀ v, v ∈ G →
        MeasurableSet (if v ∈ U then ({1} : Set ℝ) else ({1} : Set ℝ)ᶜ) := by
      intro v _
      split_ifs
      · exact measurableSet_singleton 1
      · exact (measurableSet_singleton 1).compl
    rw [hfib p U hU', hfib p' U hU',
      (mq_indep_shift O p).measure_inter_preimage_eq_mul G hsets,
      (mq_indep_shift O p').measure_inter_preimage_eq_mul G hsets]
    refine Finset.prod_congr rfl (fun v _ => ?_)
    have h1 := measure_mq_eq_one_congr O (hlab v)
    split_ifs
    · exact h1
    · rw [Set.preimage_compl, Set.preimage_compl,
        prob_compl_eq_one_sub ((mq_meas O _) (measurableSet_singleton 1)),
        prob_compl_eq_one_sub ((mq_meas O _) (measurableSet_singleton 1)), h1]
  have hcov : ∀ r : S, {ω | Φ (G.filter (fun v => O.mq (r * v) ω = 1))}
      = ⋃ U ∈ G.powerset.filter Φ, {ω | G.filter (fun v => O.mq (r * v) ω = 1) = U} := by
    intro r
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Finset.mem_filter, Finset.mem_powerset,
      exists_prop]
    refine ⟨fun h => ⟨_, ⟨Finset.filter_subset _ _, h⟩, rfl⟩, ?_⟩
    rintro ⟨U, ⟨-, hU⟩, rfl⟩
    exact hU
  have hdisj : ∀ r : S, (↑(G.powerset.filter Φ) : Set (Finset S)).PairwiseDisjoint
      (fun U => {ω | G.filter (fun v => O.mq (r * v) ω = 1) = U}) := by
    intro r a _ b _ hab
    simp only [Function.onFun, Set.disjoint_left, Set.mem_ofPred_eq]
    exact fun ω ha hb => hab (ha.symm.trans hb)
  have hmeas : ∀ (r : S) (U : Finset S),
      MeasurableSet {ω | G.filter (fun v => O.mq (r * v) ω = 1) = U} := fun r U =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred_map O (T := Set.univ) (fun v => r * v)
      (by simp) (fun W => W = U))
  rw [hcov p, hcov p', measureReal_biUnion_finset (hdisj p) (fun U _ => hmeas p U),
    measureReal_biUnion_finset (hdisj p') (fun U _ => hmeas p' U)]
  refine Finset.sum_congr rfl (fun U hU => ?_)
  rw [measureReal_def, measureReal_def,
    hprob U (Finset.mem_of_mem_filter _ hU)]

variable {Q : Type*}

lemma label_mul_congr (A : DFA S Q) (O : Oracle μ S) (hL : O.L = {w | A.state w ∈ A.accept})
    {p p' : S} (h : A.state p = A.state p') (v : S) :
    O.label (p * v) = O.label (p' * v) := by
  refine (O.label_eq_iff _ _).2 ?_
  rw [hL]
  simp only [Set.mem_ofPred_eq, DFA.state, A.step_mul]
  simp only [DFA.state] at h
  rw [h]

lemma miscutProb_congr (A : DFA S Q) (O : Oracle μ S) (hL : O.L = {w | A.state w ∈ A.accept})
    (lo hi : ℕ) (G : Finset S) {p p' : S} (h : A.state p = A.state p') :
    miscutProb O lo hi G p = miscutProb O lo hi G p' := by
  classical
  have hl : O.label p = O.label p' := by simpa using label_mul_congr A O hL h 1
  have e := measureReal_filter_congr O G (label_mul_congr A O hL h)
    (fun U => ¬ ((hi < U.card → O.label p' = 1) ∧ (U.card ≤ lo → O.label p' = 0)))
  unfold miscutProb cutCorrect voteCount
  rw [hl]
  convert e using 1

lemma undecidedProb_congr (A : DFA S Q) (O : Oracle μ S) (hL : O.L = {w | A.state w ∈ A.accept})
    (lo hi : ℕ) (G : Finset S) {p p' : S} (h : A.state p = A.state p') :
    undecidedProb O lo hi G p = undecidedProb O lo hi G p' := by
  classical
  have e := measureReal_filter_congr O G (label_mul_congr A O hL h)
    (fun U => ¬ (hi < U.card ∨ U.card ≤ lo))
  unfold undecidedProb decided voteCount
  convert e using 1

/-! ## Weighted concentration at a fixed family -/

open scoped Classical in
/-- Weighted Hoeffding for failures decided by disjoint blocks: the weighted failure mass falls
`t` below its mean with chance at most `∑ w²/t²`. -/
theorem weighted_dev_le (O : Oracle μ S) (C : Finset S) (blk : S → Finset S)
    (hblk : ∀ p ∈ C, ∀ q ∈ C, p ≠ q → Disjoint (blk p) (blk q))
    (Bad : S → Set Ω) (hmeasB : ∀ p, MeasurableSet[noiseAlg O ↑(blk p)] (Bad p))
    (w : S → ℝ) (hw : ∀ p, 0 ≤ w p) (t : ℝ) (ht : 0 < t) :
    μ.real {ω | t ≤ ∑ p ∈ C, w p * μ.real (Bad p)
        - ∑ p ∈ C.filter (fun p => ω ∈ Bad p), w p}
      ≤ (∑ p ∈ C, w p ^ 2) / t ^ 2 := by
  classical
  by_cases hz : ∑ p ∈ C, w p ^ 2 = 0
  · have hw0 : ∀ p ∈ C, w p = 0 := fun p hp =>
      pow_eq_zero_iff (n := 2) (by norm_num) |>.1
        ((Finset.sum_eq_zero_iff_of_nonneg (fun p _ => sq_nonneg (w p))).1 hz p hp)
    have hempty : {ω | t ≤ ∑ p ∈ C, w p * μ.real (Bad p)
        - ∑ p ∈ C.filter (fun p => ω ∈ Bad p), w p} = ∅ := by
      ext ω
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false, not_le]
      rw [Finset.sum_eq_zero (fun p hp => by rw [hw0 p hp, zero_mul]),
        Finset.sum_eq_zero (fun p hp => hw0 p (Finset.mem_filter.1 hp).1)]
      simpa using ht
    rw [hempty, hz]
    simp
  have hpos : 0 < ∑ p ∈ C, w p ^ 2 :=
    lt_of_le_of_ne (Finset.sum_nonneg (fun p _ => sq_nonneg _)) (Ne.symm hz)
  have hmeasA : ∀ p, MeasurableSet (Bad p) := fun p => noiseAlg_le O _ _ (hmeasB p)
  set Z : {p // p ∈ C} → Ω → ℝ := fun i ω => -(w i.val) * (Bad i.val).indicator 1 ω with hZ
  have hindZ : iIndepFun Z μ := by
    refine iIndepFun_blocks (X := O.noise) O.noise_meas O.noise_indep
      (fun i : {p // p ∈ C} => blk i.val) ?_ Z (fun i => ?_)
    · intro a b hab
      exact hblk a.val a.property b.val b.property (fun h => hab (Subtype.ext h))
    · have hsup : (⨆ w ∈ blk i.val, MeasurableSpace.comap (O.noise w) inferInstance)
          = noiseAlg O ↑(blk i.val) := by
        unfold noiseAlg
        exact iSup_congr (fun w => by simp)
      rw [hsup]
      exact (measurable_const.indicator (hmeasB i.val)).const_mul _
  set Y : {p // p ∈ C} → Ω → ℝ := fun i ω => Z i ω - μ[Z i] with hY
  have hindY : iIndepFun Y μ := by
    have h := hindZ.comp (fun (i : {p // p ∈ C}) (x : ℝ) => x - μ[Z i])
      (fun i => measurable_id.sub_const _)
    exact h
  have hsub : ∀ i ∈ (Finset.univ : Finset {p // p ∈ C}),
      HasSubgaussianMGF (Y i) ((‖(0 : ℝ) - -w i.val‖₊ / 2) ^ 2) μ := by
    intro i _
    refine hasSubgaussianMGF_of_mem_Icc
      ((measurable_const.indicator (hmeasA i.val)).const_mul _).aemeasurable
      (Filter.Eventually.of_forall (fun ω => ?_))
    by_cases h : ω ∈ Bad i.val <;> simp [hZ, h, hw i.val]
  have hmain := HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun hindY hsub ht.le
  have hint : ∀ i, μ[Z i] = -(w i.val) * μ.real (Bad i.val) := by
    intro i
    simp only [hZ]
    rw [integral_const_mul, integral_indicator_one (hmeasA i.val)]
  have hsumY : ∀ ω, ∑ i ∈ (Finset.univ : Finset {p // p ∈ C}), Y i ω
      = ∑ p ∈ C, w p * μ.real (Bad p) - ∑ p ∈ C.filter (fun p => ω ∈ Bad p), w p := by
    intro ω
    have e1 : ∑ i ∈ (Finset.univ : Finset {p // p ∈ C}), Y i ω
        = ∑ p ∈ C, (w p * μ.real (Bad p) - (if ω ∈ Bad p then w p else 0)) := by
      rw [← Finset.sum_coe_sort C]
      refine Finset.sum_congr rfl (fun i _ => ?_)
      simp only [hY]
      rw [hint i]
      simp only [hZ, Set.indicator_apply, Pi.one_apply]
      split_ifs <;> ring
    rw [e1, Finset.sum_sub_distrib, Finset.sum_filter]
  have hc : ((∑ i ∈ (Finset.univ : Finset {p // p ∈ C}),
      (‖(0 : ℝ) - -w i.val‖₊ / 2) ^ 2 : ℝ≥0) : ℝ) = (∑ p ∈ C, w p ^ 2) / 4 := by
    push_cast
    rw [Finset.sum_div, ← Finset.sum_coe_sort C (fun p => w p ^ 2 / 4)]
    refine Finset.sum_congr rfl (fun i _ => ?_)
    rw [Real.norm_eq_abs, zero_sub, neg_neg, div_pow, sq_abs]
    ring
  have hset : {ω | t ≤ ∑ p ∈ C, w p * μ.real (Bad p)
      - ∑ p ∈ C.filter (fun p => ω ∈ Bad p), w p}
      = {ω | t ≤ ∑ i ∈ (Finset.univ : Finset {p // p ∈ C}), Y i ω} := by
    ext ω
    simp only [Set.mem_ofPred_eq, hsumY ω]
  rw [hset]
  refine le_trans hmain ?_
  rw [hc]
  have hexp : ∀ y : ℝ, 0 < y → Real.exp (-y) ≤ 1 / y := by
    intro y hy
    rw [Real.exp_neg, one_div]
    exact inv_anti₀ hy (by linarith [Real.add_one_le_exp y])
  have hy : 0 < 2 * t ^ 2 / ∑ p ∈ C, w p ^ 2 := by positivity
  calc Real.exp (-t ^ 2 / (2 * ((∑ p ∈ C, w p ^ 2) / 4)))
      = Real.exp (-(2 * t ^ 2 / ∑ p ∈ C, w p ^ 2)) := by
        congr 1
        field_simp
        ring
    _ ≤ 1 / (2 * t ^ 2 / ∑ p ∈ C, w p ^ 2) := hexp _ hy
    _ ≤ (∑ p ∈ C, w p ^ 2) / t ^ 2 := by
        rw [one_div_div, div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith

/-! ## The family is chosen by noise it does not score -/

open scoped Classical in
/-- `weighted_dev_le` for the family the table selects.  The selection reads `readSet P cands`,
and on a flat alphabet the prefixes of `C` are read nowhere in it. -/
theorem selected_dev_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (P cands C : Finset S) (hP : ∀ q ∈ P, q ∈ Pre) (hV : ∀ v ∈ cands, v ∈ insert 1 Suf)
    (hCPre : ∀ p ∈ C, p ∈ Pre) (hPC : Disjoint P C) (fam : Ω → Finset S)
    (hfam : ∀ ω, fam ω ⊆ cands)
    (hcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') → fam ω = fam ω')
    (Ev : Finset S → S → Ω → Prop)
    (hEv : ∀ (A₀ : Finset S) (p : S) (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | Ev A₀ p ω})
    (w : S → ℝ) (hw : ∀ p, 0 ≤ w p) (t : ℝ) (ht : 0 < t) :
    μ.real {ω | t ≤ ∑ p ∈ C, w p * μ.real {ω' | Ev (fam ω) p ω'}
        - ∑ p ∈ C.filter (fun p => Ev (fam ω) p ω), w p}
      ≤ (∑ p ∈ C, w p ^ 2) / t ^ 2 := by
  classical
  set R : Finset S := C.biUnion (fun p => cands.image (fun v => p * v)) with hR
  have hdisjR : Disjoint (↑R : Set S) (↑(readSet P cands) : Set S) := by
    rw [Finset.disjoint_coe, hR, Finset.disjoint_biUnion_left]
    intro p hp
    exact Finset.disjoint_coe.1 (disjoint_image_readSet hflat hP hV (hCPre p hp)
      (Finset.disjoint_right.1 hPC hp))
  set Bad : Finset S → Set Ω := fun A₀ => {ω | t ≤ ∑ p ∈ C, w p * μ.real {ω' | Ev A₀ p ω'}
      - ∑ p ∈ C.filter (fun p => Ev A₀ p ω), w p} with hBad
  have hmeasU : ∀ A₀, MeasurableSet (Bad A₀) := fun A₀ =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O (fun p ω => Ev A₀ p ω)
      (fun p _ => hEv A₀ p Set.univ (fun _ _ => Set.mem_univ _))
      (fun U => t ≤ ∑ p ∈ C, w p * μ.real {ω' | Ev A₀ p ω'} - ∑ p ∈ U, w p))
  have hmeasR : ∀ A₀ ∈ cands.powerset, MeasurableSet[noiseAlg O ↑R] (Bad A₀) := fun A₀ hA₀ =>
    measurableSet_filter_pred' O (fun p ω => Ev A₀ p ω)
      (fun p hp => hEv A₀ p ↑R (fun v hv => Finset.mem_coe.2 (Finset.mem_biUnion.2
        ⟨p, hp, Finset.mem_image_of_mem _ (Finset.mem_powerset.1 hA₀ hv)⟩)))
      (fun U => t ≤ ∑ p ∈ C, w p * μ.real {ω' | Ev A₀ p ω'} - ∑ p ∈ U, w p)
  have hfixed : ∀ A₀ ∈ cands.powerset, μ.real (Bad A₀) ≤ (∑ p ∈ C, w p ^ 2) / t ^ 2 := by
    intro A₀ hA₀
    refine weighted_dev_le O C (fun p => A₀.image (fun v => p * v)) ?_
      (fun p => {ω | Ev A₀ p ω})
      (fun p => hEv A₀ p _ (fun v hv => Finset.mem_coe.2 (Finset.mem_image_of_mem _ hv)))
      w hw t ht
    intro p hp q hq hpq
    rw [Finset.disjoint_left]
    intro z hz hz'
    obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hz
    obtain ⟨v', hv'A, hv'⟩ := Finset.mem_image.1 hz'
    exact hpq (hflat p (hCPre p hp) q (hCPre q hq) v (hV v (Finset.mem_powerset.1 hA₀ hv)) v'
      (hV v' (Finset.mem_powerset.1 hA₀ hv'A)) hv'.symm)
  exact selection_block_bound O R (readSet P cands) hdisjR cands.powerset ∅
    (Finset.empty_mem_powerset _) fam (fun ω => Finset.mem_powerset.2 (hfam ω)) hcongr Bad
    hmeasU hmeasR _ (div_nonneg (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) (sq_nonneg _))
    hfixed

/-! ## Over the run -/

open scoped Classical in
/-- How far the family's realized failure mass on the population prefixes of `F` the table
does not hold falls below its mean under fresh noise. -/
noncomputable def devAt (O : Oracle μ S) (populations : Finset J) (B : State) (Dj : Measure S)
    (F : Finset S) (Ev : Finset S → S → Ω → Prop) (x : Run Ω S J) : ℝ :=
  ∑ p ∈ F \ prefixesAt populations B.npref x,
      Dj.real {p} * μ.real {ω | Ev (clusterAt O.mq populations x B) p ω}
    - ∑ p ∈ (F \ prefixesAt populations B.npref x).filter
        (fun p => Ev (clusterAt O.mq populations x B) p (oracleNoise x)), Dj.real {p}

open scoped Classical in
theorem runMeasure_dev_le {Pre Suf : Set S} (hflat : Flat Pre Suf) (O : Oracle μ S)
    (D : J → Measure S) (Dsf : Measure S) [∀ j, IsProbabilityMeasure (D j)]
    [IsProbabilityMeasure Dsf] (populations : Finset J) (hsupp : ∀ j ∈ populations, D j Preᶜ = 0)
    (hsuppSf : Dsf Sufᶜ = 0)
    (B : State) (Dj : Measure S) (F : Finset S) (hF : ∀ p ∈ F, p ∈ Pre)
    (Ev : Finset S → S → Ω → Prop)
    (hEv : ∀ (A₀ : Finset S) (p : S) (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | Ev A₀ p ω})
    (t : ℝ) (ht : 0 < t) (ρ : ℝ) (hρ : ∑ p ∈ F, Dj.real {p} ^ 2 ≤ ρ) :
    (runMeasure μ D Dsf).real {x | t ≤ devAt O populations B Dj F Ev x} ≤ ρ / t ^ 2 := by
  classical
  have hR' : ∀ (P : Finset S) (A₀ : Finset S), MeasurableSet {ω : Ω | t ≤
      ∑ p ∈ F \ P, Dj.real {p} * μ.real {ω' | Ev A₀ p ω'}
        - ∑ p ∈ (F \ P).filter (fun p => Ev A₀ p ω), Dj.real {p}} := fun P A₀ =>
    noiseAlg_le O Set.univ _ (measurableSet_filter_pred' O (fun p ω => Ev A₀ p ω)
      (fun p _ => hEv A₀ p Set.univ (fun _ _ => Set.mem_univ _))
      (fun U => t ≤ ∑ p ∈ F \ P, Dj.real {p} * μ.real {ω' | Ev A₀ p ω'} - ∑ p ∈ U, Dj.real {p}))
  have hR : ∀ P C : Finset S, MeasurableSet (if (1 : S) ∈ C then
      oracleNoise ⁻¹' {ω : Ω | t ≤
        ∑ p ∈ F \ P, Dj.real {p}
            * μ.real {ω' | Ev (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω'}
          - ∑ p ∈ (F \ P).filter (fun p => Ev (clusterOf O B.cn B.cd B.sc B.scd P C B.k ω) p ω),
            Dj.real {p}}
      else (∅ : Set (Run Ω S J))) := by
    intro P C
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2 (clusterOf_subset O B.cn B.cd B.sc B.scd P C B.k ω hone))
        (fun A₀ => measurableSet_clusterOf O B.cn B.cd B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {ω : Ω | t ≤ ∑ p ∈ F \ P, Dj.real {p} * μ.real {ω' | Ev A₀ p ω'}
          - ∑ p ∈ (F \ P).filter (fun p => Ev A₀ p ω), Dj.real {p}}) (hR' P))
    · exact MeasurableSet.empty
  have hmeas : MeasurableSet {x : Run Ω S J | t ≤ devAt O populations B Dj F Ev x} := by
    have h := measurableSet_of_run_data populations B _ hR
    convert h using 1
    ext x
    simp only [Set.mem_ofPred_eq, if_pos (one_mem_poolAt B.nsuff x)]
    rfl
  have hEnn : runMeasure μ D Dsf {x | t ≤ devAt O populations B Dj F Ev x}
      ≤ ENNReal.ofReal (ρ / t ^ 2) := by
    refine runMeasure_slice_le D Dsf _ hmeas _ ?_
    filter_upwards [ae_draws_mem_Pre D Dsf Pre populations hsupp,
      ae_sfx_mem_Suf D Dsf Suf hsuppSf] with d hd hdS
    set P : Finset S :=
      populations.biUnion (fun j => (Finset.range B.npref).image (fun i => d.1.2 j i)) with hPdef
    set cands : Finset S := insert 1 ((Finset.range B.nsuff).image (fun i => d.1.1 i))
      with hcands
    have hV : ∀ v ∈ cands, v ∈ insert 1 Suf := pool_mem_Suf B.nsuff (fun i => d.1.1 i) hdS
    have hP : ∀ q ∈ P, q ∈ Pre := by
      intro q hq
      obtain ⟨j, hj, hq'⟩ := Finset.mem_biUnion.1 hq
      obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hq'
      exact hd j hj i
    have h := selected_dev_le hflat O P cands (F \ P) hP hV
      (fun p hp => hF p (Finset.mem_sdiff.1 hp).1) Finset.disjoint_sdiff
      (fun ω => clusterAt O.mq populations ((ω, d) : Run Ω S J) B)
      (fun ω => clusterAt_subset O populations B _)
      (fun ω ω' h => clusterAt_congr O populations B d h) Ev hEv (fun p => Dj.real {p})
      (fun p => measureReal_nonneg) t ht
    have hle : (∑ p ∈ F \ P, Dj.real {p} ^ 2) / t ^ 2 ≤ ρ / t ^ 2 := by
      gcongr
      exact le_trans (Finset.sum_le_sum_of_subset_of_nonneg Finset.sdiff_subset
        (fun _ _ _ => sq_nonneg _)) hρ
    exact (ENNReal.le_ofReal_iff_toReal_le (measure_ne_top _ _)
      (div_nonneg (le_trans (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) hρ) (sq_nonneg _))).2
      (le_trans h hle)
  rw [measureReal_def]
  exact ENNReal.toReal_le_of_le_ofReal
    (div_nonneg (le_trans (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) hρ) (sq_nonneg _)) hEnn

/-! ## Mass bookkeeping -/

/-- A population leaves little mass outside some finite part of its support. -/
lemma exists_finset_tail (Dj : Measure S) [IsProbabilityMeasure Dj] (Pre : Set S) {η : ℝ}
    (hη : 0 < η) : ∃ F : Finset S, (∀ p ∈ F, p ∈ Pre) ∧ Dj.real (Pre \ ↑F) ≤ η := by
  classical
  have htot : ∑' a : S, Dj {a} ≠ ∞ := by
    have h := measure_setOf_eq_tsum Dj Set.univ
    rw [Set.indicator_univ] at h
    rw [← h]
    exact measure_ne_top _ _
  obtain ⟨s, hs⟩ := ((ENNReal.tendsto_tsum_compl_atTop_zero htot).eventually
    (gt_mem_nhds (ENNReal.ofReal_pos.2 hη))).exists
  refine ⟨s.filter (· ∈ Pre), fun p hp => (Finset.mem_filter.1 hp).2, ?_⟩
  have hsub : Pre \ ↑(s.filter (· ∈ Pre)) ⊆ (↑s : Set S)ᶜ := by
    rintro p ⟨hpPre, hpF⟩ hps
    exact hpF (Finset.mem_coe.2 (Finset.mem_filter.2 ⟨hps, hpPre⟩))
  have hcompl : Dj ((↑s : Set S)ᶜ) = ∑' b : {x // x ∉ s}, Dj {(b : S)} := by
    rw [← tsum_singleton_eq Dj ((↑s : Set S)ᶜ)]
    rfl
  calc Dj.real (Pre \ ↑(s.filter (· ∈ Pre))) ≤ Dj.real ((↑s : Set S)ᶜ) :=
        measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ η := by
        rw [measureReal_def, hcompl]
        exact ENNReal.toReal_le_of_le_ofReal hη.le hs.le

lemma sum_sq_le_collisionMass (Dj : Measure S) [IsProbabilityMeasure Dj] (F : Finset S) :
    ∑ p ∈ F, Dj.real {p} ^ 2 ≤ collisionMass Dj :=
  (summable_singleton_sq Dj).sum_le_tsum F (fun _ _ => sq_nonneg _)

/-- A finite set's mass is paid for by its size and the collision mass. -/
lemma measureReal_finset_le (Dj : Measure S) [IsProbabilityMeasure Dj] (P : Finset S) {τ : ℝ}
    (hτ : 0 < τ) : Dj.real ↑P ≤ (P.card : ℝ) * τ + collisionMass Dj / τ := by
  rw [← sum_measureReal_singleton]
  have hpt : ∀ a ∈ P, Dj.real {a} ≤ τ + Dj.real {a} ^ 2 / τ := by
    intro a _
    have h0 : 0 ≤ Dj.real {a} := measureReal_nonneg
    have h1 : 2 * Dj.real {a} - τ ≤ Dj.real {a} ^ 2 / τ := by
      rw [le_div_iff₀ hτ]
      nlinarith [sq_nonneg (Dj.real {a} - τ)]
    linarith
  calc ∑ a ∈ P, Dj.real {a} ≤ ∑ a ∈ P, (τ + Dj.real {a} ^ 2 / τ) := Finset.sum_le_sum hpt
    _ = (P.card : ℝ) * τ + (∑ a ∈ P, Dj.real {a} ^ 2) / τ := by
        rw [Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul, Finset.sum_div]
    _ ≤ (P.card : ℝ) * τ + collisionMass Dj / τ := by
        gcongr
        exact sum_sq_le_collisionMass Dj P

open scoped Classical in
/-- A state's mass times its failure rate is at most the realized failure mass, the deviation,
and the mass outside the prefixes the deviation covers. -/
lemma stateMass_mul_le (Dj : Measure S) [IsProbabilityMeasure Dj] {Pre : Set S}
    (hsupp : Dj Preᶜ = 0) (F P : Finset S) (Sq : Set S) (bad : S → Prop) (m : S → ℝ) (c : ℝ)
    (hc0 : 0 ≤ c) (hc1 : c ≤ 1) (hm0 : ∀ p, 0 ≤ m p) (hm : ∀ p ∈ Sq, m p = c) (dev : ℝ)
    (hdev : dev = ∑ p ∈ F \ P, Dj.real {p} * m p
      - ∑ p ∈ (F \ P).filter (fun p => bad p), Dj.real {p}) :
    Dj.real Sq * c ≤ Dj.real {p | bad p} + dev + Dj.real (Pre \ ↑F) + Dj.real ↑P := by
  classical
  have hR0 : 0 ≤ Dj.real (Pre \ ↑F) + Dj.real ↑P := add_nonneg measureReal_nonneg measureReal_nonneg
  set CS := (F \ P).filter (fun p => p ∈ Sq) with hCS
  have h1 : Dj.real Sq ≤ Dj.real ↑CS + (Dj.real (Pre \ ↑F) + Dj.real ↑P) := by
    have hsub : Sq ⊆ ((↑CS ∪ (Pre \ ↑F)) ∪ ↑P) ∪ Preᶜ := by
      intro p hp
      by_cases hpre : p ∈ Pre
      · by_cases hF : p ∈ F
        · by_cases hP : p ∈ P
          · exact Or.inl (Or.inr hP)
          · exact Or.inl (Or.inl (Or.inl (Finset.mem_coe.2
              (Finset.mem_filter.2 ⟨Finset.mem_sdiff.2 ⟨hF, hP⟩, hp⟩))))
        · exact Or.inl (Or.inl (Or.inr ⟨hpre, hF⟩))
      · exact Or.inr hpre
    have hnull : Dj.real Preᶜ = 0 := by rw [measureReal_def, hsupp, ENNReal.toReal_zero]
    calc Dj.real Sq ≤ Dj.real (((↑CS ∪ (Pre \ ↑F)) ∪ ↑P) ∪ Preᶜ) :=
          measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ Dj.real ((↑CS ∪ (Pre \ ↑F)) ∪ ↑P) + Dj.real Preᶜ := measureReal_union_le _ _
      _ ≤ Dj.real (↑CS ∪ (Pre \ ↑F)) + Dj.real ↑P + 0 := by
          rw [hnull]
          gcongr
          exact measureReal_union_le _ _
      _ ≤ Dj.real ↑CS + Dj.real (Pre \ ↑F) + Dj.real ↑P + 0 := by
          gcongr
          exact measureReal_union_le _ _
      _ = _ := by ring
  have h2 : Dj.real ↑CS * c ≤ ∑ p ∈ F \ P, Dj.real {p} * m p := by
    rw [← sum_measureReal_singleton, Finset.sum_mul]
    calc ∑ p ∈ CS, Dj.real {p} * c = ∑ p ∈ CS, Dj.real {p} * m p :=
          Finset.sum_congr rfl (fun p hp => by rw [hm p (Finset.mem_filter.1 hp).2])
      _ ≤ ∑ p ∈ F \ P, Dj.real {p} * m p :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
            (fun p _ _ => mul_nonneg measureReal_nonneg (hm0 p))
  have h3 : ∑ p ∈ (F \ P).filter (fun p => bad p), Dj.real {p} ≤ Dj.real {p | bad p} := by
    rw [sum_measureReal_singleton]
    exact measureReal_mono (fun p hp => (Finset.mem_filter.1 (Finset.mem_coe.1 hp)).2)
      (measure_ne_top _ _)
  have h4 : Dj.real Sq * c ≤ Dj.real ↑CS * c + (Dj.real (Pre \ ↑F) + Dj.real ↑P) := by
    calc Dj.real Sq * c ≤ (Dj.real ↑CS + (Dj.real (Pre \ ↑F) + Dj.real ↑P)) * c :=
          mul_le_mul_of_nonneg_right h1 hc0
      _ = Dj.real ↑CS * c + (Dj.real (Pre \ ↑F) + Dj.real ↑P) * c := by ring
      _ ≤ _ := by
          gcongr
          exact mul_le_of_le_one_right hR0 hc1
  rw [hdev]
  linarith

lemma card_le_sub_of_forall_notMem [Fintype Q] {s t : Finset Q} (h : ∀ q ∈ s, q ∉ t) :
    s.card ≤ Fintype.card Q - t.card := by
  classical
  rw [← Finset.card_compl]
  exact Finset.card_le_card (fun q hq => Finset.mem_compl.2 (h q hq))

/-! ## The theorem -/

theorem clustering_quality_guarantee_holds : ClusteringQualityGuarantee := by
  intro Ω _ μ _ S _ J _ Q _ A O populations Pre Suf η₀ indecisionLimit εcov α δ pAP tolerance
    hL hηle hη₀ hpop hflat hpAP hind hind1 hα hα1 hε hε1 hδ hδ1 htolerance
  classical
  set δ' : ℝ := δ / 2 with hδ'def
  have hδ' : 0 < δ' := by positivity
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  set ε₁ : ℝ := min εcov (2 * indecisionLimit) with hε₁def
  have hε₁ : 0 < ε₁ := lt_min hε (by linarith)
  have hε₁c : ε₁ ≤ εcov := min_le_left _ _
  have hε₁u : ε₁ ≤ 2 * indecisionLimit := min_le_right _ _
  set N : ℝ := (populations.card : ℝ)
    * (prefCount η₀ populations indecisionLimit εcov δ' α pAP : ℝ) with hNdef
  have hN0 : 0 ≤ N := by positivity
  set L : ℝ := (ladderLen η₀ populations indecisionLimit εcov δ' α pAP : ℝ) with hLdef
  set Jc : ℝ := (populations.card : ℝ) with hJcdef
  have hL0 : 0 ≤ L := Nat.cast_nonneg _
  have hJc0 : 0 ≤ Jc := Nat.cast_nonneg _
  refine ⟨min (collisionCap η₀ populations indecisionLimit εcov δ' α pAP)
      (min (ε₁ ^ 2 / (64 * (N + 1))) (δ * ε₁ ^ 2 / (16 * (L + 1) * (Jc + 1)))), ?_, ?_⟩
  · refine lt_min ?_ (lt_min (by positivity) (by positivity))
    simp only [collisionCap]
    positivity
  intro D Dsf hD hDsf hsupp hsuppSf hpAPBound ρ hρ hρcap hρsf
  have := hD
  have := hDsf
  have hρ1 : ρ ≤ ε₁ ^ 2 / (64 * (N + 1)) :=
    le_trans hρcap (le_trans (min_le_right _ _) (min_le_left _ _))
  have hρ2 : ρ ≤ δ * ε₁ ^ 2 / (16 * (L + 1) * (Jc + 1)) :=
    le_trans hρcap (le_trans (min_le_right _ _) (min_le_right _ _))
  have hρ0 : 0 ≤ ρ :=
    le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ hpop.choose hpop.choose_spec)
  set states := stoppable η₀ populations indecisionLimit εcov δ' α pAP ρ (collisionMass Dsf)
    with hstates
  refine ⟨states, ?_⟩
  have hcc := clustering_correct O populations D Dsf Pre Suf η₀ indecisionLimit εcov α δ' ρ pAP
    524288 hηle hη₀ hpop hflat hsupp hsuppSf hρ hpAP hpAPBound hind hind1 hα hα1 hε hε1 hδ'
    (prefCount_le_poly populations η₀ indecisionLimit εcov δ' α pAP hsig hη0 hpop hind hε hε1
      hδ' (by linarith) hα (by linarith) hpAP (le_trans hpAPBound measureReal_le_one))
    (le_trans hρcap (min_le_left _ _)) (le_trans hρsf (min_le_left _ _))
  -- A finite part of each population's support, off which it has little mass.
  have hF : ∀ j, ∃ F : Finset S, (∀ p ∈ F, p ∈ Pre) ∧ (D j).real (Pre \ ↑F) ≤ ε₁ / 4 :=
    fun j => exists_finset_tail (D j) Pre (by positivity)
  choose F hFPre hFtail using hF
  set EvC : State → Finset S → S → Ω → Prop :=
    fun B G p ω => ¬ cutCorrect O B.lo B.hi G p ω with hEvC
  set EvU : State → Finset S → S → Ω → Prop :=
    fun B G p ω => ¬ decided O.mq B.lo B.hi G p ω with hEvU
  have hEvCm : ∀ B A₀ p (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | EvC B A₀ p ω} := fun B A₀ p U hU =>
    measurableSet_filter_pred_map O (T := U) (fun v => p * v) hU
      (fun W => ¬ ((B.hi < Finset.card W → O.label p = 1) ∧ (Finset.card W ≤ B.lo → O.label p = 0)))
  have hEvUm : ∀ B A₀ p (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | EvU B A₀ p ω} := fun B A₀ p U hU =>
    measurableSet_filter_pred_map O (T := U) (fun v => p * v) hU
      (fun W => ¬ (B.hi < Finset.card W ∨ Finset.card W ≤ B.lo))
  set badC : State → J → Set (Run Ω S J) :=
    fun B j => {x | εcov / 2 ≤ devAt O populations B (D j) (F j) (EvC B) x} with hbadC
  set badU : State → J → Set (Run Ω S J) :=
    fun B j => {x | indecisionLimit ≤ devAt O populations B (D j) (F j) (EvU B) x} with hbadU
  set Bad : Set (Run Ω S J) := ⋃ B ∈ states, ⋃ j ∈ populations, (badC B j ∪ badU B j)
    with hBaddef
  have hsq : ∀ j ∈ populations, ∑ p ∈ F j, (D j).real {p} ^ 2 ≤ ρ := fun j hj =>
    le_trans (sum_sq_le_collisionMass (D j) (F j)) (hρ j hj)
  have hper : ∀ B, ∀ j ∈ populations,
      (runMeasure μ D Dsf).real (badC B j ∪ badU B j) ≤ 8 * ρ / ε₁ ^ 2 := by
    intro B j hj
    have h1 := runMeasure_dev_le hflat O D Dsf populations hsupp hsuppSf B (D j) (F j) (hFPre j)
      (EvC B) (hEvCm B) (εcov / 2) (by positivity) ρ (hsq j hj)
    have h2 := runMeasure_dev_le hflat O D Dsf populations hsupp hsuppSf B (D j) (F j) (hFPre j)
      (EvU B) (hEvUm B) indecisionLimit hind ρ (hsq j hj)
    have e1 : ρ / (εcov / 2) ^ 2 ≤ 4 * ρ / ε₁ ^ 2 := by
      calc ρ / (εcov / 2) ^ 2 = 4 * ρ / εcov ^ 2 := by field_simp; ring
        _ ≤ 4 * ρ / ε₁ ^ 2 := div_le_div_of_nonneg_left (by positivity) (by positivity)
          (pow_le_pow_left₀ hε₁.le hε₁c 2)
    have e2 : ρ / indecisionLimit ^ 2 ≤ 4 * ρ / ε₁ ^ 2 := by
      calc ρ / indecisionLimit ^ 2 = 4 * ρ / (2 * indecisionLimit) ^ 2 := by field_simp; ring
        _ ≤ 4 * ρ / ε₁ ^ 2 := div_le_div_of_nonneg_left (by positivity) (by positivity)
          (pow_le_pow_left₀ hε₁.le hε₁u 2)
    calc (runMeasure μ D Dsf).real (badC B j ∪ badU B j)
        ≤ (runMeasure μ D Dsf).real (badC B j) + (runMeasure μ D Dsf).real (badU B j) :=
          measureReal_union_le _ _
      _ ≤ 4 * ρ / ε₁ ^ 2 + 4 * ρ / ε₁ ^ 2 := add_le_add (h1.trans e1) (h2.trans e2)
      _ = 8 * ρ / ε₁ ^ 2 := by ring
  have hcardst : (states.card : ℝ) ≤ L := by
    have : states.card ≤ ladderLen η₀ populations indecisionLimit εcov δ' α pAP :=
      le_trans (Finset.card_le_card (Finset.filter_subset _ _))
        (le_trans Finset.card_image_le (by simp))
    rw [hLdef]
    exact_mod_cast this
  have hBad : (runMeasure μ D Dsf).real Bad ≤ δ / 2 := by
    have hρ2' : 16 * (L + 1) * (Jc + 1) * ρ ≤ δ * ε₁ ^ 2 := by
      rw [le_div_iff₀ (by positivity)] at hρ2
      linarith
    calc (runMeasure μ D Dsf).real Bad
        ≤ ∑ B ∈ states, (runMeasure μ D Dsf).real
            (⋃ j ∈ populations, (badC B j ∪ badU B j)) := measureReal_biUnion_finset_le _ _
      _ ≤ ∑ B ∈ states, ∑ j ∈ populations, (runMeasure μ D Dsf).real (badC B j ∪ badU B j) :=
          Finset.sum_le_sum (fun B _ => measureReal_biUnion_finset_le _ _)
      _ ≤ ∑ _B ∈ states, ∑ _j ∈ populations, 8 * ρ / ε₁ ^ 2 :=
          Finset.sum_le_sum (fun B _ => Finset.sum_le_sum (fun j hj => hper B j hj))
      _ = (states.card : ℝ) * (Jc * (8 * ρ / ε₁ ^ 2)) := by
          simp only [Finset.sum_const, nsmul_eq_mul, hJcdef]
      _ ≤ L * (Jc * (8 * ρ / ε₁ ^ 2)) := mul_le_mul_of_nonneg_right hcardst (by positivity)
      _ ≤ δ / 2 := by
          rw [show L * (Jc * (8 * ρ / ε₁ ^ 2)) = 8 * L * Jc * ρ / ε₁ ^ 2 by ring,
            div_le_iff₀ (by positivity)]
          nlinarith [mul_nonneg (by positivity : (0 : ℝ) ≤ L + Jc + 1) hρ0,
            mul_nonneg (mul_nonneg hL0 hJc0) hρ0]
  have key : ∀ T : Set (Run Ω S J), {x | (∃ B : {B : State // B ∈ states},
          x ∈ ret O.mq populations indecisionLimit α B.val) ∧
        ∀ B : {B : State // B ∈ states}, x ∈ ret O.mq populations indecisionLimit α B.val →
          ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | cutCorrect O B.val.lo B.val.hi
                (clusterAt O.mq populations x B.val) p (oracleNoise x)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo B.val.hi
                (clusterAt O.mq populations x B.val) p (oracleNoise x)}
              ≤ 2 * indecisionLimit} ⊆ T ∪ Bad →
      1 - δ ≤ (runMeasure μ D Dsf).real T := by
    intro T hT
    have := le_trans hcc (le_trans (measureReal_mono hT (measure_ne_top _ _))
      (measureReal_union_le T Bad))
    linarith
  apply key
  intro x hx
  by_cases hxb : x ∈ Bad
  · exact Or.inr hxb
  refine Or.inl ⟨hx.1, fun B hret => ?_⟩
  have hgood := hx.2 B hret
  have hnb : ∀ j ∈ populations, x ∉ badC B.val j ∧ x ∉ badU B.val j := by
    intro j hj
    have : x ∉ badC B.val j ∪ badU B.val j := fun h =>
      hxb (Set.mem_biUnion B.property (Set.mem_biUnion hj h))
    exact ⟨fun h => this (Or.inl h), fun h => this (Or.inr h)⟩
  set G := clusterAt O.mq populations x B.val with hG
  set P := prefixesAt populations B.val.npref x with hP
  have hPmass : ∀ j ∈ populations, (D j).real ↑P ≤ ε₁ / 4 := by
    intro j hj
    have hnpref : B.val.npref ≤ prefCount η₀ populations indecisionLimit εcov δ' α pAP := by
      have hsched := Finset.mem_of_mem_filter _ B.property
      obtain ⟨i, _, hi⟩ := Finset.mem_image.1 hsched
      rw [← hi]
      exact Nat.div_le_self _ _
    have hcard : (P.card : ℝ) ≤ N := by
      have h1 := card_prefixesAt_le populations B.val.npref x
      have h2 : populations.card * B.val.npref
          ≤ populations.card * prefCount η₀ populations indecisionLimit εcov δ' α pAP :=
        Nat.mul_le_mul_left _ hnpref
      have h3 : ((prefixesAt populations B.val.npref x).card : ℝ) ≤ (populations.card : ℝ)
          * (prefCount η₀ populations indecisionLimit εcov δ' α pAP : ℝ) := by
        exact_mod_cast le_trans h1 h2
      exact h3
    set τ : ℝ := ε₁ / (8 * (N + 1)) with hτ
    have hτ0 : 0 < τ := by positivity
    have a1 : (P.card : ℝ) * τ ≤ ε₁ / 8 := by
      calc (P.card : ℝ) * τ ≤ N * τ := mul_le_mul_of_nonneg_right hcard hτ0.le
        _ ≤ ε₁ / 8 := by
          rw [hτ, mul_div_assoc', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith
    have a2 : collisionMass (D j) / τ ≤ ε₁ / 8 := by
      rw [div_le_iff₀ hτ0]
      calc collisionMass (D j) ≤ ρ := hρ j hj
        _ ≤ ε₁ ^ 2 / (64 * (N + 1)) := hρ1
        _ = ε₁ / 8 * τ := by rw [hτ]; field_simp; ring
    linarith [measureReal_finset_le (D j) P hτ0]
  unfold quality qualityBound
  refine Prod.mk_le_mk.2 ⟨?_, ?_⟩
  · apply card_le_sub_of_forall_notMem
    intro q hq hq2
    obtain ⟨p₀, hp₀, htolerancep⟩ := (Finset.mem_filter.1 hq).2
    obtain ⟨j, hj, hmass⟩ := (Finset.mem_filter.1 hq2).2
    have hkey := stateMass_mul_le (D j) (hsupp j hj) (F j) P {p | A.state p = q}
      (fun p => EvC B.val G p (oracleNoise x))
      (fun p => miscutProb O B.val.lo B.val.hi G p) (miscutProb O B.val.lo B.val.hi G p₀)
      measureReal_nonneg measureReal_le_one (fun p => measureReal_nonneg)
      (fun p hp => miscutProb_congr A O hL _ _ G (hp.trans hp₀.symm))
      (devAt O populations B.val (D j) (F j) (EvC B.val) x) rfl
    have hdev : devAt O populations B.val (D j) (F j) (EvC B.val) x < εcov / 2 :=
      not_le.1 (hnb j hj).1
    have hreal : (D j).real {p | ¬ cutCorrect O B.val.lo B.val.hi G p (oracleNoise x)}
        ≤ εcov := by
      rw [← Set.compl_ofPred, measureReal_compl (measurableSet_of_countable _),
        probReal_univ]
      linarith [(hgood j hj).1]
    have hsm : stateMass A (D j) q * miscutProb O B.val.lo B.val.hi G p₀ ≤ 2 * εcov := by
      have := hFtail j
      have := hPmass j hj
      unfold stateMass
      linarith
    have h1 : 2 * εcov < tolerance * stateMass A (D j) q := by
      rw [div_lt_iff₀ htolerance] at hmass
      linarith
    have h2 : 0 ≤ stateMass A (D j) q := measureReal_nonneg
    have h3 := mul_le_mul_of_nonneg_left htolerancep.le h2
    linarith
  · apply card_le_sub_of_forall_notMem
    intro q hq hq2
    obtain ⟨p₀, hp₀, htolerancep⟩ := (Finset.mem_filter.1 hq).2
    obtain ⟨j, hj, hmass⟩ := (Finset.mem_filter.1 hq2).2
    have hkey := stateMass_mul_le (D j) (hsupp j hj) (F j) P {p | A.state p = q}
      (fun p => EvU B.val G p (oracleNoise x))
      (fun p => undecidedProb O B.val.lo B.val.hi G p) (undecidedProb O B.val.lo B.val.hi G p₀)
      measureReal_nonneg measureReal_le_one (fun p => measureReal_nonneg)
      (fun p hp => undecidedProb_congr A O hL _ _ G (hp.trans hp₀.symm))
      (devAt O populations B.val (D j) (F j) (EvU B.val) x) rfl
    have hdev : devAt O populations B.val (D j) (F j) (EvU B.val) x < indecisionLimit :=
      not_le.1 (hnb j hj).2
    have hreal := (hgood j hj).2
    have hsm : stateMass A (D j) q * undecidedProb O B.val.lo B.val.hi G p₀
        ≤ 4 * indecisionLimit := by
      have := hFtail j
      have := hPmass j hj
      unfold stateMass
      linarith
    have h1 : 4 * indecisionLimit < tolerance * stateMass A (D j) q := by
      rw [div_lt_iff₀ htolerance] at hmass
      linarith
    have h2 : 0 ≤ stateMass A (D j) q := measureReal_nonneg
    have h3 := mul_le_mul_of_nonneg_left htolerancep.le h2
    linarith

end OrthoDFA
