import OrthoDFA.ClusteringQuality
import OrthoDFA.Proofs.Budget

/-!
# The quality of the returned family

`clustering_correct` bounds what the family miscuts, or leaves undecided, under the one noise
draw the run used.  `miscutProb` and `undecidedProb` ask the same of fresh noise, averaged over
the population.

The two meet because the family reads the noise only at the table's strings.  At any other
population prefix its verdict is a fresh draw, independent across prefixes, so a weighted
Hoeffding bound ties the realized mass to its mean; the table's own prefixes carry little mass
once the collision mass is small.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]
variable {rule : Clusterer S}
variable {J : Type*} [Fintype J]

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
/-- How far the returned family's realized failure mass on the population prefixes of `F` the table
does not hold falls below its mean under fresh noise. -/
noncomputable def devAt (rule : Clusterer S) (O : Oracle μ S) (populations : Finset J) (B : State) (Dj : Measure S)
    (F : Finset S) (Ev : Finset S → S → Ω → Prop) (x : Run Ω S J) : ℝ :=
  ∑ p ∈ F \ prefixesAt populations B.npref x,
      Dj.real {p} * μ.real {ω | Ev (familyBy rule O.mq populations x B) p ω}
    - ∑ p ∈ (F \ prefixesAt populations B.npref x).filter
        (fun p => Ev (familyBy rule O.mq populations x B) p (oracleNoise x)), Dj.real {p}

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
    (runMeasure μ D Dsf).real {x | t ≤ devAt rule O populations B Dj F Ev x} ≤ ρ / t ^ 2 := by
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
            * μ.real {ω' | Ev (insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) p ω'}
          - ∑ p ∈ (F \ P).filter
              (fun p => Ev (insert 1 (clusterOf rule O B.sc B.scd P C B.k ω)) p ω),
            Dj.real {p}}
      else (∅ : Set (Run Ω S J))) := by
    intro P C
    split_ifs with hone
    · exact measurable_nz (measurableSet_of_fam (T := C.powerset)
        (fun ω => Finset.mem_powerset.2
          (Finset.insert_subset hone (clusterOf_subset O B.sc B.scd P C B.k ω hone)))
        (fun A₀ => measurableSet_insert_clusterOf O B.sc B.scd P C B.k hone A₀)
        (fun A₀ => {ω : Ω | t ≤ ∑ p ∈ F \ P, Dj.real {p} * μ.real {ω' | Ev A₀ p ω'}
          - ∑ p ∈ (F \ P).filter (fun p => Ev A₀ p ω), Dj.real {p}}) (hR' P))
    · exact MeasurableSet.empty
  have hmeas : MeasurableSet {x : Run Ω S J | t ≤ devAt rule O populations B Dj F Ev x} := by
    have h := measurableSet_of_run_data populations B _ hR
    convert h using 1
    ext x
    simp only [Set.mem_ofPred_eq, if_pos (one_mem_poolAt B.nsuff x)]
    rfl
  have hEnn : runMeasure μ D Dsf {x | t ≤ devAt rule O populations B Dj F Ev x}
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
    have hfsub : ∀ ω, familyBy rule O.mq populations ((ω, d) : Run Ω S J) B ⊆ cands := by
      intro ω
      have hc : clusterBy rule O.mq populations ((ω, d) : Run Ω S J) B ⊆ cands :=
        clusterAt_subset O populations B _
      exact Finset.insert_subset (Finset.mem_insert_self _ _) hc
    have hfcongr : ∀ ω ω', (∀ w ∈ readSet P cands, O.noise w ω = O.noise w ω') →
        familyBy rule O.mq populations ((ω, d) : Run Ω S J) B
          = familyBy rule O.mq populations ((ω', d) : Run Ω S J) B := by
      intro ω ω' h
      unfold familyBy
      rw [clusterAt_congr O populations B d h]
    have h := selected_dev_le hflat O P cands (F \ P) hP hV
      (fun p hp => hF p (Finset.mem_sdiff.1 hp).1) Finset.disjoint_sdiff
      (fun ω => familyBy rule O.mq populations ((ω, d) : Run Ω S J) B) hfsub hfcongr Ev hEv
      (fun p => Dj.real {p})
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
/-- A failure rate's mean over the population is at most the realized failure mass, the
deviation, and the mass outside the prefixes the deviation covers. -/
lemma integral_le_realized (Dj : Measure S) [IsProbabilityMeasure Dj] {Pre : Set S}
    (hsupp : Dj Preᶜ = 0) (F P : Finset S) (bad : S → Prop) (m : S → ℝ)
    (hm0 : ∀ p, 0 ≤ m p) (hm1 : ∀ p, m p ≤ 1) (dev : ℝ)
    (hdev : dev = ∑ p ∈ F \ P, Dj.real {p} * m p
      - ∑ p ∈ (F \ P).filter (fun p => bad p), Dj.real {p}) :
    ∫ p, m p ∂Dj ≤ Dj.real {p | bad p} + dev + Dj.real (Pre \ ↑F) + Dj.real ↑P := by
  classical
  have hint : Integrable m Dj :=
    Integrable.of_bound (measurable_of_countable m).aestronglyMeasurable 1
      (ae_of_all _ (fun p => by rw [Real.norm_eq_abs, abs_of_nonneg (hm0 p)]; exact hm1 p))
  rw [← integral_add_compl (measurableSet_of_countable (↑(F \ P) : Set S)) hint,
    setIntegral_finset (F \ P) hint.integrableOn]
  have h1 : ∫ p in (↑(F \ P) : Set S)ᶜ, m p ∂Dj ≤ Dj.real (Pre \ ↑F) + Dj.real ↑P := by
    have hsub : (↑(F \ P) : Set S)ᶜ ⊆ ((Pre \ ↑F) ∪ ↑P) ∪ Preᶜ := by
      intro p hp
      by_cases hpre : p ∈ Pre
      · by_cases hF : p ∈ F
        · refine Or.inl (Or.inr (Finset.mem_coe.2 ?_))
          by_contra hP
          exact hp (Finset.mem_coe.2 (Finset.mem_sdiff.2 ⟨hF, hP⟩))
        · exact Or.inl (Or.inl ⟨hpre, hF⟩)
      · exact Or.inr hpre
    have hnull : Dj.real Preᶜ = 0 := by rw [measureReal_def, hsupp, ENNReal.toReal_zero]
    calc ∫ p in (↑(F \ P) : Set S)ᶜ, m p ∂Dj ≤ ∫ p in (↑(F \ P) : Set S)ᶜ, (1 : ℝ) ∂Dj :=
          setIntegral_mono hint.integrableOn (integrableOn_const (measure_ne_top _ _)) hm1
      _ = Dj.real (↑(F \ P) : Set S)ᶜ := by rw [setIntegral_const, smul_eq_mul, mul_one]
      _ ≤ Dj.real (((Pre \ ↑F) ∪ ↑P) ∪ Preᶜ) := measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ Dj.real ((Pre \ ↑F) ∪ ↑P) + Dj.real Preᶜ := measureReal_union_le _ _
      _ ≤ Dj.real (Pre \ ↑F) + Dj.real ↑P := by
          rw [hnull, add_zero]
          exact measureReal_union_le _ _
  have h3 : ∑ p ∈ (F \ P).filter (fun p => bad p), Dj.real {p} ≤ Dj.real {p | bad p} := by
    rw [sum_measureReal_singleton]
    exact measureReal_mono (fun p hp => (Finset.mem_filter.1 (Finset.mem_coe.1 hp)).2)
      (measure_ne_top _ _)
  simp only [smul_eq_mul]
  rw [hdev]
  linarith

/-! ## The theorem -/

theorem clustering_quality_guarantee_holds : ClusteringQualityGuarantee := by
  intro Ω _ μ _ S _ J _ O populations uni Pre Suf η₀ indecisionLimit εcov₀ α δ pAP crossLimit
    slack a v hηle hη₀ huni hflat hpAP hind hind1 hα hα1 hε₀ hε1₀ hδ hδ1 hstr hstr1 hslack hneed
    hv0
  have hpop : populations.Nonempty := ⟨uni, huni⟩
  classical
  simp only [ret_eq, familyAt_eq, clusterAt_eq]
  generalize (lloydClusterer : Clusterer S) = rule
  set δ' : ℝ := δ / 2 with hδ'def
  have hδ' : 0 < δ' := by positivity
  have hsig : 0 < sig η₀ := by simp only [sig]; linarith
  have hη0 : 0 ≤ η₀ := le_trans (eta_nonneg O) hηle
  -- the proof runs at a cut budget small against the signal and the veto's share of `δ'`
  obtain ⟨εcov, hεdef⟩ : ∃ e : ℝ, e = min εcov₀ (min (1 / 2 - η₀)
      (δ' / (2 * (populations.card : ℝ) * v))) := ⟨_, rfl⟩
  have hcardR : (0 : ℝ) < (populations.card : ℝ) := by exact_mod_cast Finset.card_pos.2 hpop
  have hvR : (0 : ℝ) < (v : ℝ) := by exact_mod_cast hv0
  have hε : 0 < εcov := by
    rw [hεdef]; exact lt_min hε₀ (lt_min (by linarith) (by positivity))
  have hε1 : εcov ≤ 1 := by rw [hεdef]; exact le_trans (min_le_left _ _) hε1₀
  have hεle : εcov ≤ εcov₀ := by rw [hεdef]; exact min_le_left _ _
  have hveto : 2 * (populations.card : ℝ) * v * εcov ≤ δ' := by
    have h : εcov ≤ δ' / (2 * (populations.card : ℝ) * v) := by
      rw [hεdef]; exact min_le_of_right_le (min_le_right _ _)
    rw [le_div_iff₀ (by positivity)] at h
    linarith
  set N : ℝ := (populations.card : ℝ)
    * (prefCount η₀ populations indecisionLimit εcov δ' α pAP crossLimit : ℝ) with hNdef
  have hN0 : 0 ≤ N := by positivity
  set L : ℝ := (ladderLen η₀ populations indecisionLimit εcov δ' α pAP crossLimit : ℝ) with hLdef
  set Jc : ℝ := (populations.card : ℝ) with hJcdef
  have hL0 : 0 ≤ L := Nat.cast_nonneg _
  have hJc0 : 0 ≤ Jc := Nat.cast_nonneg _
  refine ⟨min (collisionCap η₀ populations indecisionLimit εcov δ' α pAP crossLimit)
      (min (slack ^ 2 / (64 * (N + 1))) (δ * slack ^ 2 / (16 * (L + 1) * (Jc + 1)))), ?_, ?_⟩
  · refine lt_min ?_ (lt_min (by positivity) (by positivity))
    simp only [collisionCap]
    positivity
  intro D Dsf hD hDsf hsupp hsuppSf hpAPBound ρ hρ hρcap hρsf
  have := hD
  have := hDsf
  have hρ1 : ρ ≤ slack ^ 2 / (64 * (N + 1)) :=
    le_trans hρcap (le_trans (min_le_right _ _) (min_le_left _ _))
  have hρ2 : ρ ≤ δ * slack ^ 2 / (16 * (L + 1) * (Jc + 1)) :=
    le_trans hρcap (le_trans (min_le_right _ _) (min_le_right _ _))
  have hρ0 : 0 ≤ ρ :=
    le_trans (tsum_nonneg (fun a => sq_nonneg _)) (hρ hpop.choose hpop.choose_spec)
  set states := stoppable η₀ populations indecisionLimit εcov δ' α pAP crossLimit ρ
    with hstates
  refine ⟨states, fun B hB =>
    cross_of_mem_schedule O hη0 hη₀ hstr (Finset.mem_of_mem_filter _ hB), ?_⟩
  have hcc := clustering_correct O rule populations uni D Dsf Pre Suf η₀ indecisionLimit εcov α δ'
    ρ pAP crossLimit 2048 a v hηle hη₀ huni hflat hsupp hsuppSf hρ hpAP hpAPBound hind hind1 hα
    hα1 hε hε1 hδ'
    (prefCount_le_poly populations η₀ indecisionLimit εcov δ' α pAP crossLimit hsig hη0 hpop hind
      hε hε1 hδ' (by linarith) hα (by linarith) hpAP (le_trans hpAPBound measureReal_le_one))
    (le_trans hρcap (min_le_left _ _)) (le_trans hρsf (min_le_left _ _))
    (certBudget_covers populations η₀ indecisionLimit εcov₀ εcov δ' α pAP crossLimit a v hv0 hη0
      hη₀ hpop hind hε₀ hε1₀ hεdef hδ' (by linarith) hα (by linarith) hpAP
      (le_trans hpAPBound measureReal_le_one) hstr hstr1 hneed)
    hveto
  -- A finite part of each population's support, off which it has little mass.
  have hF : ∀ j, ∃ F : Finset S, (∀ p ∈ F, p ∈ Pre) ∧ (D j).real (Pre \ ↑F) ≤ slack / 4 :=
    fun j => exists_finset_tail (D j) Pre (by positivity)
  choose F hFPre hFtail using hF
  set EvC : State → Finset S → S → Ω → Prop :=
    fun B G p ω => ¬ cutCorrect O B.lo (B.hi + 1) G p ω with hEvC
  set EvU : State → Finset S → S → Ω → Prop :=
    fun B G p ω => ¬ decided O.mq B.lo (B.hi + 1) G p ω with hEvU
  have hEvCm : ∀ B A₀ p (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | EvC B A₀ p ω} := fun B A₀ p U hU =>
    measurableSet_filter_pred_map O (T := U) (fun v => p * v) hU
      (fun W => ¬ ((B.hi + 1 < Finset.card W → O.label p = 1)
        ∧ (Finset.card W ≤ B.lo → O.label p = 0)))
  have hEvUm : ∀ B A₀ p (U : Set S), (∀ v ∈ A₀, p * v ∈ U) →
      MeasurableSet[noiseAlg O U] {ω | EvU B A₀ p ω} := fun B A₀ p U hU =>
    measurableSet_filter_pred_map O (T := U) (fun v => p * v) hU
      (fun W => ¬ (B.hi + 1 < Finset.card W ∨ Finset.card W ≤ B.lo))
  set badC : State → J → Set (Run Ω S J) :=
    fun B j => {x | slack / 2 ≤ devAt rule O populations B (D j) (F j) (EvC B) x} with hbadC
  set badU : State → J → Set (Run Ω S J) :=
    fun B j => {x | slack / 2 ≤ devAt rule O populations B (D j) (F j) (EvU B) x} with hbadU
  set Bad : Set (Run Ω S J) := ⋃ B ∈ states, ⋃ j ∈ populations, (badC B j ∪ badU B j)
    with hBaddef
  have hsq : ∀ j ∈ populations, ∑ p ∈ F j, (D j).real {p} ^ 2 ≤ ρ := fun j hj =>
    le_trans (sum_sq_le_collisionMass (D j) (F j)) (hρ j hj)
  have hper : ∀ B, ∀ j ∈ populations,
      (runMeasure μ D Dsf).real (badC B j ∪ badU B j) ≤ 8 * ρ / slack ^ 2 := by
    intro B j hj
    have h1 := runMeasure_dev_le (rule := rule) hflat O D Dsf populations hsupp hsuppSf B (D j) (F j) (hFPre j)
      (EvC B) (hEvCm B) (slack / 2) (by positivity) ρ (hsq j hj)
    have h2 := runMeasure_dev_le (rule := rule) hflat O D Dsf populations hsupp hsuppSf B (D j) (F j) (hFPre j)
      (EvU B) (hEvUm B) (slack / 2) (by positivity) ρ (hsq j hj)
    have e : ρ / (slack / 2) ^ 2 = 4 * ρ / slack ^ 2 := by field_simp; ring
    calc (runMeasure μ D Dsf).real (badC B j ∪ badU B j)
        ≤ (runMeasure μ D Dsf).real (badC B j) + (runMeasure μ D Dsf).real (badU B j) :=
          measureReal_union_le _ _
      _ ≤ 4 * ρ / slack ^ 2 + 4 * ρ / slack ^ 2 := add_le_add (h1.trans e.le) (h2.trans e.le)
      _ = 8 * ρ / slack ^ 2 := by ring
  have hcardst : (states.card : ℝ) ≤ L := by
    have : states.card ≤ ladderLen η₀ populations indecisionLimit εcov δ' α pAP crossLimit :=
      le_trans (Finset.card_le_card (Finset.filter_subset _ _))
        (le_trans Finset.card_image_le (by simp))
    rw [hLdef]
    exact_mod_cast this
  have hBad : (runMeasure μ D Dsf).real Bad ≤ δ / 2 := by
    have hρ2' : 16 * (L + 1) * (Jc + 1) * ρ ≤ δ * slack ^ 2 := by
      rw [le_div_iff₀ (by positivity)] at hρ2
      linarith
    calc (runMeasure μ D Dsf).real Bad
        ≤ ∑ B ∈ states, (runMeasure μ D Dsf).real
            (⋃ j ∈ populations, (badC B j ∪ badU B j)) := measureReal_biUnion_finset_le _ _
      _ ≤ ∑ B ∈ states, ∑ j ∈ populations, (runMeasure μ D Dsf).real (badC B j ∪ badU B j) :=
          Finset.sum_le_sum (fun B _ => measureReal_biUnion_finset_le _ _)
      _ ≤ ∑ _B ∈ states, ∑ _j ∈ populations, 8 * ρ / slack ^ 2 :=
          Finset.sum_le_sum (fun B _ => Finset.sum_le_sum (fun j hj => hper B j hj))
      _ = (states.card : ℝ) * (Jc * (8 * ρ / slack ^ 2)) := by
          simp only [Finset.sum_const, nsmul_eq_mul, hJcdef]
      _ ≤ L * (Jc * (8 * ρ / slack ^ 2)) := mul_le_mul_of_nonneg_right hcardst (by positivity)
      _ ≤ δ / 2 := by
          rw [show L * (Jc * (8 * ρ / slack ^ 2)) = 8 * L * Jc * ρ / slack ^ 2 by ring,
            div_le_iff₀ (by positivity)]
          nlinarith [mul_nonneg (by positivity : (0 : ℝ) ≤ L + Jc + 1) hρ0,
            mul_nonneg (mul_nonneg hL0 hJc0) hρ0]
  have key : ∀ T : Set (Run Ω S J), {x | (∃ B : {B : State // B ∈ states},
          x ∈ retBy rule O.mq populations uni indecisionLimit α v (certSize a B.val) B.val) ∧
        ∀ B : {B : State // B ∈ states},
          x ∈ retBy rule O.mq populations uni indecisionLimit α v (certSize a B.val) B.val →
          ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | cutCorrect O B.val.lo (B.val.hi + 1)
                (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
            ∧ (D j).real {p | ¬ decided O.mq B.val.lo (B.val.hi + 1)
                (familyBy rule O.mq populations x B.val) p (oracleNoise x)}
              ≤ 2 * indecisionLimit} ⊆ T ∪ Bad →
      1 - δ - α ≤ (runMeasure μ D Dsf).real T := by
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
  set G := familyBy rule O.mq populations x B.val with hG
  set P := prefixesAt populations B.val.npref x with hP
  have hPmass : ∀ j ∈ populations, (D j).real ↑P ≤ slack / 4 := by
    intro j hj
    have hnpref :
        B.val.npref ≤ prefCount η₀ populations indecisionLimit εcov δ' α pAP crossLimit := by
      have hsched := Finset.mem_of_mem_filter _ B.property
      obtain ⟨i, _, hi⟩ := Finset.mem_image.1 hsched
      rw [← hi]
      exact Nat.div_le_self _ _
    have hcard : (P.card : ℝ) ≤ N := by
      have h1 := card_prefixesAt_le populations B.val.npref x
      have h2 : populations.card * B.val.npref
          ≤ populations.card * prefCount η₀ populations indecisionLimit εcov δ' α pAP crossLimit :=
        Nat.mul_le_mul_left _ hnpref
      have h3 : ((prefixesAt populations B.val.npref x).card : ℝ) ≤ (populations.card : ℝ)
          * (prefCount η₀ populations indecisionLimit εcov δ' α pAP crossLimit : ℝ) := by
        exact_mod_cast le_trans h1 h2
      exact h3
    set τ : ℝ := slack / (8 * (N + 1)) with hτ
    have hτ0 : 0 < τ := by positivity
    have a1 : (P.card : ℝ) * τ ≤ slack / 8 := by
      calc (P.card : ℝ) * τ ≤ N * τ := mul_le_mul_of_nonneg_right hcard hτ0.le
        _ ≤ slack / 8 := by
          rw [hτ, mul_div_assoc', div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith
    have a2 : collisionMass (D j) / τ ≤ slack / 8 := by
      rw [div_le_iff₀ hτ0]
      calc collisionMass (D j) ≤ ρ := hρ j hj
        _ ≤ slack ^ 2 / (64 * (N + 1)) := hρ1
        _ = slack / 8 * τ := by rw [hτ]; field_simp; ring
    linarith [measureReal_finset_le (D j) P hτ0]
  intro j hj
  have hC := integral_le_realized (D j) (hsupp j hj) (F j) P
    (fun p => EvC B.val G p (oracleNoise x)) (fun p => miscutProb O B.val.lo (B.val.hi + 1) G p)
    (fun p => measureReal_nonneg) (fun p => measureReal_le_one)
    (devAt rule O populations B.val (D j) (F j) (EvC B.val) x) rfl
  have hU := integral_le_realized (D j) (hsupp j hj) (F j) P
    (fun p => EvU B.val G p (oracleNoise x)) (fun p => undecidedProb O B.val.lo (B.val.hi + 1) G p)
    (fun p => measureReal_nonneg) (fun p => measureReal_le_one)
    (devAt rule O populations B.val (D j) (F j) (EvU B.val) x) rfl
  have hdevC : devAt rule O populations B.val (D j) (F j) (EvC B.val) x < slack / 2 :=
    not_le.1 (hnb j hj).1
  have hdevU : devAt rule O populations B.val (D j) (F j) (EvU B.val) x < slack / 2 :=
    not_le.1 (hnb j hj).2
  have hrealC : (D j).real {p | ¬ cutCorrect O B.val.lo (B.val.hi + 1) G p (oracleNoise x)} ≤ εcov := by
    rw [← Set.compl_ofPred, measureReal_compl (measurableSet_of_countable _), probReal_univ]
    linarith [(hgood j hj).1]
  have hrealU := (hgood j hj).2
  have := hFtail j
  have := hPmass j hj
  exact ⟨by linarith, by linarith⟩

end OrthoDFA
