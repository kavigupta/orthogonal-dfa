import OrthoDFA.Proofs.AnchorFamily

/-!
# A majority-pattern anchor resolves its atoms' states: the proof

At atom `i` and pattern `b`, with `x ≥ 1/2 ≥ y` the two chances that `z_i·v` reads `b_i` and `X_v`
the indicator that it does, the term

    f_v = y(x − X_v) on `z_i`'s side of `v`,   x(y − X_v) off it,

counted where `v` reads `b` at the other atoms, has mean `0` and lies in `[−x(1 − y), xy]`. The
terms are independent across candidates, so Hoeffding bounds their sum below at `−t`; and
`ρ_v X_v − (x + y − 2xy)X_v = 2xy(X_v − 1/2) + f_v` for each `v`. Over the `m 2^m` pairs `(i, b)`
and the count of the class-preserving candidates reading their own classes, the union bound
spends `δ`.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [Stringlike S]

lemma exp_hoeffDev_le {n : ℕ} (hn : 0 < n) {a : ℝ} (ha : 0 < a) :
    Real.exp (-2 * hoeffDev n a ^ 2 / n) ≤ a := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  unfold hoeffDev
  by_cases hl : 0 ≤ Real.log (1 / a)
  · rw [Real.sq_sqrt (by positivity)]
    have e : -2 * (n * Real.log (1 / a) / 2) / n = Real.log a := by
      rw [one_div, Real.log_inv]; field_simp
    rw [e, Real.exp_log ha]
  · push Not at hl
    rw [Real.sqrt_eq_zero_of_nonpos (by nlinarith)]
    have h1 : 1 / a < 1 := (Real.log_neg_iff (by positivity)).1 hl
    rw [div_lt_one ha] at h1
    simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero, zero_div,
      Real.exp_zero]
    exact h1.le

lemma readsAs_prob (O : Oracle μ S) (s : S) (β : Bool) :
    μ.real {ω | readsAs O.mq s β ω} = if β then O.acceptRate s else 1 - O.acceptRate s := by
  have h1 := measureReal_mq_one O s
  cases β
  · have : {ω | readsAs O.mq s false ω} = {ω | O.mq s ω = 1}ᶜ := by ext ω; simp [readsAs]
    rw [this, measureReal_compl (measurableSet_eq_fun (mq_meas O s) measurable_const),
      probReal_univ, h1]
    simp
  · have : {ω | readsAs O.mq s true ω} = {ω | O.mq s ω = 1} := by ext ω; simp [readsAs]
    rw [this, h1]
    simp

open scoped Classical in
lemma exists_rates (O : Oracle μ S) {η₀ : ℝ} (hη : O.η ≤ η₀) :
    ∃ eI eO : ℝ, 0 ≤ eI ∧ eI ≤ η₀ ∧ 0 ≤ eO ∧ eO ≤ η₀ ∧
      ∀ s, O.acceptRate s = if s ∈ O.L then 1 - eI else eO := by
  classical
  have h0 : 0 ≤ η₀ := (eta_nonneg O).trans hη
  by_cases hin : ∃ s, s ∈ O.L
  · by_cases hout : ∃ s, s ∉ O.L
    · obtain ⟨s, hs⟩ := hin
      obtain ⟨s', hs'⟩ := hout
      have r1 := O.rate_nonneg s
      have r2 := O.rate_nonneg s'
      unfold Oracle.rate at r1 r2
      rw [if_pos hs] at r1
      rw [if_neg hs'] at r2
      refine ⟨O.ηIn, O.ηOut, r1, (le_max_left _ _).trans hη, r2, (le_max_right _ _).trans hη,
        fun u => ?_⟩
      unfold Oracle.acceptRate
      rfl
    · push Not at hout
      obtain ⟨s, hs⟩ := hin
      have r1 := O.rate_nonneg s
      unfold Oracle.rate at r1
      rw [if_pos hs] at r1
      refine ⟨O.ηIn, 0, r1, (le_max_left _ _).trans hη, le_rfl, h0, fun u => ?_⟩
      unfold Oracle.acceptRate
      rw [if_pos (hout u), if_pos (hout u)]
  · push Not at hin
    by_cases hout : ∃ s, s ∉ O.L
    · obtain ⟨s', hs'⟩ := hout
      have r2 := O.rate_nonneg s'
      unfold Oracle.rate at r2
      rw [if_neg hs'] at r2
      refine ⟨0, O.ηOut, le_rfl, h0, r2, (le_max_right _ _).trans hη, fun u => ?_⟩
      unfold Oracle.acceptRate
      rw [if_neg (hin u), if_neg (hin u)]
    · exact absurd ⟨1, hin 1⟩ hout

/-! ## Blocks of the noise -/

lemma measurableSet_readsAs_alg (O : Oracle μ S) {T : Set S} {s : S} (hs : s ∈ T) (β : Bool) :
    MeasurableSet[noiseAlg O T] {ω | readsAs O.mq s β ω} := by
  have h1 := measurableSet_mq_eq_one O hs
  cases β
  · have : {ω | readsAs O.mq s false ω} = {ω | O.mq s ω = 1}ᶜ := by ext ω; simp [readsAs]
    rw [this]; exact h1.compl
  · have : {ω | readsAs O.mq s true ω} = {ω | O.mq s ω = 1} := by ext ω; simp [readsAs]
    rw [this]; exact h1

omit [IsProbabilityMeasure μ] in
/-- Functions of disjoint blocks of the noise, one per candidate, are independent. -/
lemma iIndepFun_atomBlocks (O : Oracle μ S) (G : Finset S) (blk : S → Finset S)
    (hdisj : ∀ v ∈ G, ∀ v' ∈ G, v ≠ v' → Disjoint (blk v) (blk v')) (Y : S → Ω → ℝ)
    (hY : ∀ v ∈ G, Measurable[noiseAlg O ↑(blk v)] (Y v)) :
    iIndepFun (fun v ω => if v ∈ G then Y v ω else 0) μ := by
  classical
  set A : S → Finset S := fun v => if v ∈ G then blk v else ∅ with hA
  have hle : ∀ u, MeasurableSpace.comap (O.noise u) (inferInstance : MeasurableSpace ℝ)
      ≤ ‹MeasurableSpace Ω› := fun u => (O.noise_meas u).comap_le
  have hdA : Pairwise fun v v' => Disjoint (A v) (A v') := by
    intro v v' hne
    simp only [hA]
    by_cases hv : v ∈ G
    · by_cases hv' : v' ∈ G
      · rw [if_pos hv, if_pos hv']; exact hdisj v hv v' hv' hne
      · rw [if_neg hv']; exact Finset.disjoint_empty_right _
    · rw [if_neg hv]; exact Finset.disjoint_empty_left _
  have hgroup := iIndep_biSup_of_disjoint hle
    ((iIndepFun_iff_iIndep (fun _ => (inferInstance : MeasurableSpace ℝ)) O.noise μ).1
      O.noise_indep) A hdA
  refine (iIndepFun_iff_iIndep (fun _ => (inferInstance : MeasurableSpace ℝ)) _ μ).2 ?_
  refine iIndep_of_iIndep_of_le hgroup (fun v => ?_)
  have hsup : (⨆ u ∈ A v, MeasurableSpace.comap (O.noise u) inferInstance)
      = noiseAlg O ↑(A v) := by
    unfold noiseAlg
    exact iSup_congr (fun u => by simp)
  rw [hsup]
  by_cases hv : v ∈ G
  · have hm : Measurable[noiseAlg O ↑(A v)] (fun ω => if v ∈ G then Y v ω else 0) := by
      have hAv : A v = blk v := if_pos hv
      rw [hAv]
      simp only [if_pos hv]
      exact hY v hv
    exact hm.comap_le
  · have hm : Measurable[noiseAlg O ↑(A v)] (fun ω => if v ∈ G then Y v ω else 0) := by
      simp only [if_neg hv]
      exact measurable_const
    exact hm.comap_le

/-! ## The deviation at one atom and pattern -/

/-- `v` reads `b` at every atom but `i`. -/
def otherReads {m : ℕ} (mq : S → Ω → ℝ) (z : Fin m → S) (i : Fin m) (b : Fin m → Bool) (v : S)
    (ω : Ω) : Prop :=
  ∀ j, j ≠ i → readsAs mq (z j * v) (b j) ω

open scoped Classical in
/-- `f_v` where `v` reads `b` at the other atoms, `0` elsewhere. -/
noncomputable def anchorDev {m : ℕ} (O : Oracle μ S) (z : Fin m → S) (i : Fin m)
    (b : Fin m → Bool) (x y : ℝ) (v : S) (ω : Ω) : ℝ :=
  if otherReads O.mq z i b v ω then
    (if readsAs O.mq (z i * v) (b i) ω then
      (if (z i * v ∈ O.L ↔ b i = true) then y * (x - 1) else x * (y - 1)) else x * y)
  else 0

/-- The strings `z_j·v`. -/
noncomputable def atomBlock {m : ℕ} (z : Fin m → S) (v : S) : Finset S :=
  Finset.univ.image fun j => z j * v

lemma mem_atomBlock {m : ℕ} (z : Fin m → S) (v : S) (j : Fin m) :
    z j * v ∈ (↑(atomBlock z v) : Set S) :=
  Finset.mem_coe.2 (Finset.mem_image_of_mem _ (Finset.mem_univ j))

lemma measurableSet_otherReads {m : ℕ} (O : Oracle μ S) (z : Fin m → S) (i : Fin m)
    (b : Fin m → Bool) (v : S) {T : Set S} (hT : ∀ j, j ≠ i → z j * v ∈ T) :
    MeasurableSet[noiseAlg O T] {ω | otherReads O.mq z i b v ω} := by
  have : {ω | otherReads O.mq z i b v ω}
      = ⋂ j, {ω | j ≠ i → readsAs O.mq (z j * v) (b j) ω} := by
    ext ω; simp [otherReads]
  rw [this]
  refine MeasurableSet.iInter fun j => ?_
  by_cases h : j = i
  · have : {ω | j ≠ i → readsAs O.mq (z j * v) (b j) ω} = Set.univ := by
      ext ω; simp [h]
    rw [this]; exact MeasurableSet.univ
  · have : {ω | j ≠ i → readsAs O.mq (z j * v) (b j) ω} = {ω | readsAs O.mq (z j * v) (b j) ω} := by
      ext ω; simp [h]
    rw [this]; exact measurableSet_readsAs_alg O (hT j h) _

lemma anchorDev_meas {m : ℕ} (O : Oracle μ S) (z : Fin m → S) (i : Fin m) (b : Fin m → Bool)
    (x y : ℝ) (v : S) : Measurable[noiseAlg O ↑(atomBlock z v)] (anchorDev O z i b x y v) := by
  classical
  unfold anchorDev
  exact Measurable.ite (measurableSet_otherReads O z i b v fun j _ => mem_atomBlock z v j)
    (Measurable.ite (measurableSet_readsAs_alg O (mem_atomBlock z v i) _) measurable_const
      measurable_const) measurable_const

lemma measurable_of_alg (O : Oracle μ S) {T : Set S} {f : Ω → ℝ}
    (h : Measurable[noiseAlg O T] f) : Measurable f :=
  h.mono (noiseAlg_le O T) le_rfl

omit [IsProbabilityMeasure μ] in
lemma anchorDev_range {m : ℕ} (O : Oracle μ S) (z : Fin m → S) (i : Fin m) (b : Fin m → Bool)
    {x y : ℝ} (hx : x ≤ 1) (hy : 0 ≤ y) (hyx : y ≤ x) (v : S) (ω : Ω) :
    anchorDev O z i b x y v ω ∈ Set.Icc (-(x * (1 - y))) (x * y) := by
  classical
  have hx0 : 0 ≤ x := hy.trans hyx
  unfold anchorDev
  split_ifs <;> constructor <;> nlinarith [mul_nonneg hx0 hy]

open scoped Classical in
lemma anchorDev_mean {m : ℕ} (O : Oracle μ S) (z : Fin m → S) (i : Fin m) (b : Fin m → Bool)
    (x y : ℝ) (v : S) (hdist : ∀ j, j ≠ i → z j * v ≠ z i * v)
    (hρ : μ.real {ω | readsAs O.mq (z i * v) (b i) ω}
      = if (z i * v ∈ O.L ↔ b i = true) then x else y) :
    μ[anchorDev O z i b x y v] = 0 := by
  classical
  set T1 : Set S := {s | ∃ j, j ≠ i ∧ s = z j * v} with hT1
  have hdisj : Disjoint T1 {z i * v} := by
    rw [Set.disjoint_singleton_right]
    rintro ⟨j, hj, he⟩
    exact hdist j hj he.symm
  set E := {ω | otherReads O.mq z i b v ω} with hE
  set H := {ω | readsAs O.mq (z i * v) (b i) ω} with hH
  have hEm : MeasurableSet[noiseAlg O T1] E :=
    measurableSet_otherReads O z i b v fun j hj => ⟨j, hj, rfl⟩
  have hHm : MeasurableSet[noiseAlg O {z i * v}] H :=
    measurableSet_readsAs_alg O (Set.mem_singleton _) _
  have hind := indep_noiseAlg O hdisj
  have hEH := measureReal_inter_of_indep hind hEm hHm
  have hEHc := measureReal_inter_of_indep hind hEm hHm.compl
  have hHc : μ.real Hᶜ = 1 - μ.real H := by
    rw [measureReal_compl (noiseAlg_le O _ _ hHm), probReal_univ]
  have hEa := noiseAlg_le O _ _ hEm
  have hHa := noiseAlg_le O _ _ hHm
  set f1 : ℝ := if (z i * v ∈ O.L ↔ b i = true) then y * (x - 1) else x * (y - 1) with hf1
  have hpt : anchorDev O z i b x y v
      = fun ω => (E ∩ H).indicator (fun _ => f1) ω + (E ∩ Hᶜ).indicator (fun _ => x * y) ω := by
    funext ω
    unfold anchorDev
    by_cases h1 : ω ∈ E <;> by_cases h2 : ω ∈ H
    · have h1' : otherReads O.mq z i b v ω := h1
      have h2' : readsAs O.mq (z i * v) (b i) ω := h2
      simp [h1', h2', Set.indicator_of_mem (Set.mem_inter h1 h2),
        Set.indicator_of_notMem (fun h : ω ∈ E ∩ Hᶜ => h.2 h2), hf1]
    · have h1' : otherReads O.mq z i b v ω := h1
      have h2' : ¬ readsAs O.mq (z i * v) (b i) ω := h2
      simp [h1', h2', Set.indicator_of_notMem (fun h : ω ∈ E ∩ H => h2 h.2),
        Set.indicator_of_mem (Set.mem_inter h1 (show ω ∈ Hᶜ from h2))]
    · have h1' : ¬ otherReads O.mq z i b v ω := h1
      simp [h1', Set.indicator_of_notMem (fun h : ω ∈ E ∩ H => h1 h.1),
        Set.indicator_of_notMem (fun h : ω ∈ E ∩ Hᶜ => h1 h.1)]
    · have h1' : ¬ otherReads O.mq z i b v ω := h1
      simp [h1', Set.indicator_of_notMem (fun h : ω ∈ E ∩ H => h1 h.1),
        Set.indicator_of_notMem (fun h : ω ∈ E ∩ Hᶜ => h1 h.1)]
  rw [hpt, integral_add ((integrable_const _).indicator (hEa.inter hHa))
      ((integrable_const _).indicator (hEa.inter hHa.compl)),
    integral_indicator_const _ (hEa.inter hHa), integral_indicator_const _ (hEa.inter hHa.compl),
    hEH, hEHc, hHc, hρ]
  simp only [smul_eq_mul, hf1]
  split_ifs <;> ring

/-! ## The claim -/

theorem anchorResolves_holds : AnchorResolves := by
  classical
  intro Ω _ μ _ S _ O m z G η₀ pAP δ hη hη2 hδ hinj hAP κ₀ t
  obtain ⟨eI, eO, heI0, heI, heO0, heO, hrate⟩ := exists_rates O hη
  set xr : Bool → ℝ := fun β => if β then 1 - eI else 1 - eO with hxr
  set yr : Bool → ℝ := fun β => if β then eO else eI with hyr
  have hprob : ∀ s β, μ.real {ω | readsAs O.mq s β ω}
      = if (s ∈ O.L ↔ β = true) then xr β else yr β := by
    intro s β
    rw [readsAs_prob, hrate]
    by_cases hs : s ∈ O.L <;> cases β <;> simp [hs, hxr, hyr]
  have hxy : ∀ β, xr β ≤ 1 ∧ 0 ≤ yr β ∧ yr β ≤ xr β := by
    intro β; cases β <;> simp only [hxr, hyr] <;> refine ⟨?_, ?_, ?_⟩ <;> norm_num <;> linarith
  have hK : ∀ β, 1 / 2 + κ₀ ≤ xr β + yr β - 2 * xr β * yr β := by
    have h1 : 0 ≤ 1 - 2 * η₀ := by linarith
    have h2 : 1 - 2 * η₀ ≤ 1 - 2 * eI := by linarith
    have h3 : 1 - 2 * η₀ ≤ 1 - 2 * eO := by linarith
    have key : (1 - 2 * η₀) ^ 2 ≤ (1 - 2 * eI) * (1 - 2 * eO) := by
      rw [sq]; exact mul_le_mul h2 h3 h1 (h1.trans h2)
    intro β; cases β <;> simp only [hxr, hyr, κ₀] <;> norm_num <;> nlinarith [key]
  have hdist : ∀ v ∈ G, ∀ i j, j ≠ i → z j * v ≠ z i * v :=
    fun v hv i j hji he => hji (hinj j i v hv v hv he).1
  have hblk : ∀ v ∈ G, ∀ v' ∈ G, v ≠ v' → Disjoint (atomBlock z v) (atomBlock z v') := by
    intro v hv v' hv' hne
    rw [Finset.disjoint_left]
    intro u hu hu'
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.1 hu
    obtain ⟨j', -, he⟩ := Finset.mem_image.1 hu'
    exact hne (hinj j' j v' hv' v hv he).2.symm
  set lvl : ℝ := δ / (m * 2 ^ m + 1) with hlvl
  have hlvl0 : 0 < lvl := by positivity
  have ht0 : 0 ≤ t := Real.sqrt_nonneg _
  have htail : ∀ k : ℕ, 0 < k → k ≤ G.card → Real.exp (-2 * k * (t / k) ^ 2) ≤ lvl := by
    intro k hk hkn
    have hkR : (0 : ℝ) < k := by exact_mod_cast hk
    have hnR : (k : ℝ) ≤ G.card := by exact_mod_cast hkn
    refine le_trans (Real.exp_le_exp.2 ?_) (exp_hoeffDev_le (lt_of_lt_of_le hk hkn) hlvl0)
    rw [div_pow, show -2 * (k : ℝ) * (t ^ 2 / k ^ 2) = -2 * t ^ 2 / k by field_simp]
    rw [div_le_div_iff₀ hkR (lt_of_lt_of_le hkR hnR)]
    nlinarith [sq_nonneg t]
  -- the bad events
  set Bp : Fin m × (Fin m → Bool) → Set Ω := fun p =>
    {ω | ∑ v ∈ G, anchorDev O z p.1 p.2 (xr (p.2 p.1)) (yr (p.2 p.1)) v ω < -t} with hBp
  set tstar : Fin m → Bool := fun i => decide (z i ∈ O.L) with htstar
  set GAP := G.filter (fun v => ∀ p, p * v ∈ O.L ↔ p ∈ O.L) with hGAP
  set V : S → Ω → ℝ := fun v ω => if (∀ i, readsAs O.mq (z i * v) (tstar i) ω) then 1 else 0
    with hV
  set B0 : Set Ω := {ω | ∑ v ∈ GAP, V v ω < (1 - η₀) ^ m * GAP.card - t} with hB0
  have hBp_le : ∀ p, μ.real (Bp p) ≤ lvl := by
    rintro ⟨i, b⟩
    set x := xr (b i)
    set y := yr (b i)
    obtain ⟨hx1, hy0, hyx⟩ := hxy (b i)
    rcases Nat.eq_zero_or_pos G.card with hn | hn
    · have : Bp (i, b) = ∅ := by
        ext ω
        simp only [hBp, Finset.card_eq_zero.1 hn, Finset.sum_empty, Set.mem_ofPred_eq,
          Set.mem_empty_iff_false, iff_false, not_lt]
        linarith
      rw [this, measureReal_empty]; exact hlvl0.le
    have hnR : (0 : ℝ) < G.card := by exact_mod_cast hn
    set c : ℝ := x * (1 - y)
    set U : S → Ω → ℝ := fun v ω => if v ∈ G then anchorDev O z i b x y v ω + c else 0 with hU
    have hUind : iIndepFun U μ := iIndepFun_atomBlocks O G (atomBlock z) hblk
      (fun v ω => anchorDev O z i b x y v ω + c)
      (fun v _ => (anchorDev_meas O z i b x y v).add_const c)
    have hUm : ∀ v, Measurable (U v) := by
      intro v
      by_cases hv : v ∈ G
      · simp only [hU, if_pos hv]
        exact (measurable_of_alg O (anchorDev_meas O z i b x y v)).add_const c
      · simp only [hU, if_neg hv]; exact measurable_const
    have hUI : ∀ v, ∀ᵐ ω ∂μ, U v ω ∈ Set.Icc (0 : ℝ) 1 := by
      intro v
      refine ae_of_all _ fun ω => ?_
      by_cases hv : v ∈ G
      · simp only [hU, if_pos hv]
        have := anchorDev_range O z i b hx1 hy0 hyx v ω
        constructor <;> nlinarith [this.1, this.2]
      · simp only [hU, if_neg hv]; norm_num
    have hmean : (G.card : ℝ) * c ≤ ∑ v ∈ G, μ[U v] := by
      have : ∀ v ∈ G, μ[U v] = c := by
        intro v hv
        have hint : Integrable (anchorDev O z i b x y v) μ :=
          MeasureTheory.Integrable.of_mem_Icc _ _
            (measurable_of_alg O (anchorDev_meas O z i b x y v)).aemeasurable
            (ae_of_all _ fun ω => anchorDev_range O z i b hx1 hy0 hyx v ω)
        have e : U v = fun ω => anchorDev O z i b x y v ω + c := by
          funext ω; simp [hU, hv]
        rw [e, integral_add hint (integrable_const c), integral_const,
          anchorDev_mean O z i b x y v (hdist v hv i) (hprob _ _)]
        simp
      rw [Finset.sum_congr rfl this, Finset.sum_const, nsmul_eq_mul]
    have h := sumLower_le U G c (t / G.card) (fun v => (hUm v).aemeasurable) hUind hUI hmean
      (div_nonneg ht0 hnR.le)
    refine le_trans (measureReal_mono ?_) (h.trans (htail G.card hn le_rfl))
    intro ω hω
    simp only [hBp, Set.mem_ofPred_eq] at hω ⊢
    have e : ∑ v ∈ G, U v ω = ∑ v ∈ G, anchorDev O z i b x y v ω + G.card * c := by
      have e1 : ∑ v ∈ G, U v ω = ∑ v ∈ G, (anchorDev O z i b x y v ω + c) :=
        Finset.sum_congr rfl (fun v hv => by simp only [hU, if_pos hv])
      rw [e1, Finset.sum_add_distrib, Finset.sum_const, nsmul_eq_mul]
    rw [e, mul_sub, mul_div_cancel₀ _ hnR.ne']
    linarith
  have hB0_le : μ.real B0 ≤ lvl := by
    rcases Nat.eq_zero_or_pos GAP.card with ha | ha
    · have : B0 = ∅ := by
        ext ω
        simp only [hB0, Finset.card_eq_zero.1 ha, Finset.sum_empty, Finset.card_empty,
          Nat.cast_zero, mul_zero, zero_sub, Set.mem_ofPred_eq, Set.mem_empty_iff_false,
          iff_false, not_lt]
        linarith
      rw [this, measureReal_empty]; exact hlvl0.le
    have haR : (0 : ℝ) < GAP.card := by exact_mod_cast ha
    have hVm : ∀ v, Measurable[noiseAlg O ↑(atomBlock z v)] (V v) := by
      intro v
      have : {ω | ∀ i, readsAs O.mq (z i * v) (tstar i) ω}
          = ⋂ i, {ω | readsAs O.mq (z i * v) (tstar i) ω} := by ext; simp
      refine Measurable.ite ?_ measurable_const measurable_const
      rw [this]
      exact MeasurableSet.iInter fun i => measurableSet_readsAs_alg O (mem_atomBlock z v i) _
    set U : S → Ω → ℝ := fun v ω => if v ∈ G then V v ω else 0 with hU
    have hUind : iIndepFun U μ := iIndepFun_atomBlocks O G (atomBlock z) hblk V (fun v _ => hVm v)
    have hUm : ∀ v, Measurable (U v) := by
      intro v
      by_cases hv : v ∈ G
      · simp only [hU, if_pos hv]; exact measurable_of_alg O (hVm v)
      · simp only [hU, if_neg hv]; exact measurable_const
    have hUI : ∀ v, ∀ᵐ ω ∂μ, U v ω ∈ Set.Icc (0 : ℝ) 1 := by
      intro v
      refine ae_of_all _ fun ω => ?_
      simp only [hU, hV]
      split_ifs <;> norm_num
    have hη1 : 0 ≤ 1 - η₀ := by linarith
    have hmean : (GAP.card : ℝ) * (1 - η₀) ^ m ≤ ∑ v ∈ GAP, μ[U v] := by
      have : ∀ v ∈ GAP, (1 - η₀) ^ m ≤ μ[U v] := by
        intro v hv
        have hvG : v ∈ G := (Finset.mem_filter.1 hv).1
        have hAPv := (Finset.mem_filter.1 hv).2
        have e : U v = ({ω | ∀ i, readsAs O.mq (z i * v) (tstar i) ω} : Set Ω).indicator 1 := by
          funext ω; simp only [hU, hV, if_pos hvG]; split_ifs with h <;> simp [h]
        have hinjv : Function.Injective fun i : Fin m => z i * v :=
          fun i j he => (hinj i j v hvG v hvG he).1
        have hind := (mq_indep O).precomp hinjv
        set sets : Fin m → Set ℝ := fun i => {r : ℝ | r = 1 ↔ tstar i = true} with hsets
        have hsm : ∀ i, MeasurableSet (sets i) := by
          intro i
          cases h : tstar i
          · have : sets i = {1}ᶜ := by ext r; simp [hsets, h]
            rw [this]; exact (measurableSet_singleton 1).compl
          · have : sets i = {1} := by ext r; simp [hsets, h]
            rw [this]; exact measurableSet_singleton 1
        have hprod := hind.measure_inter_preimage_eq_mul Finset.univ (sets := sets)
          (fun i _ => hsm i)
        have hset : {ω | ∀ i, readsAs O.mq (z i * v) (tstar i) ω}
            = ⋂ i ∈ (Finset.univ : Finset (Fin m)), (fun i => O.mq (z i * v)) i ⁻¹' sets i := by
          ext ω; simp [readsAs, hsets]
        have hmeasS : MeasurableSet {ω | ∀ i, readsAs O.mq (z i * v) (tstar i) ω} := by
          rw [hset]
          exact Finset.measurableSet_biInter _ fun i _ => (mq_meas O _) (hsm i)
        rw [e, integral_indicator_one hmeasS, measureReal_def, hset, hprod, ENNReal.toReal_prod]
        have hfac : ∀ i, (1 - η₀) ≤ (μ ((fun i => O.mq (z i * v)) i ⁻¹' sets i)).toReal := by
          intro i
          have h1 : (μ ((fun i => O.mq (z i * v)) i ⁻¹' sets i)).toReal
              = μ.real {ω | readsAs O.mq (z i * v) (tstar i) ω} := by
            rw [measureReal_def]; rfl
          have hg : (z i * v ∈ O.L ↔ tstar i = true) := by
            rw [hAPv (z i)]; simp [htstar]
          rw [h1, hprob, if_pos hg]
          cases tstar i <;> simp only [hxr] <;> norm_num <;> linarith
        calc (1 - η₀) ^ m = ∏ _i : Fin m, (1 - η₀) := by simp
          _ ≤ ∏ i : Fin m, (μ ((fun i => O.mq (z i * v)) i ⁻¹' sets i)).toReal :=
            Finset.prod_le_prod (fun _ _ => hη1) (fun i _ => hfac i)
      calc (GAP.card : ℝ) * (1 - η₀) ^ m = ∑ _v ∈ GAP, (1 - η₀) ^ m := by
            rw [Finset.sum_const, nsmul_eq_mul]
        _ ≤ ∑ v ∈ GAP, μ[U v] := Finset.sum_le_sum this
    have h := sumLower_le U GAP ((1 - η₀) ^ m) (t / GAP.card) (fun v => (hUm v).aemeasurable)
      hUind hUI hmean (div_nonneg ht0 haR.le)
    refine le_trans (measureReal_mono ?_)
      (h.trans (htail GAP.card ha (Finset.card_filter_le _ _)))
    intro ω hω
    simp only [hB0, Set.mem_ofPred_eq] at hω ⊢
    have e : ∑ v ∈ GAP, U v ω = ∑ v ∈ GAP, V v ω :=
      Finset.sum_congr rfl fun v hv => by simp only [hU, if_pos (Finset.mem_filter.1 hv).1]
    rw [e, mul_sub, mul_div_cancel₀ _ haR.ne']
    linarith
  set bad : Set Ω := B0 ∪ ⋃ p ∈ (Finset.univ : Finset (Fin m × (Fin m → Bool))), Bp p with hbad
  have hbad_le : μ.real bad ≤ δ := by
    refine (measureReal_union_le _ _).trans ?_
    have h2 := measureReal_biUnion_finset_le (μ := μ)
      (Finset.univ : Finset (Fin m × (Fin m → Bool))) Bp
    have h3 : ∑ p ∈ (Finset.univ : Finset (Fin m × (Fin m → Bool))), μ.real (Bp p)
        ≤ (m * 2 ^ m : ℝ) * lvl := by
      refine (Finset.sum_le_sum fun p _ => hBp_le p).trans (le_of_eq ?_)
      simp [Finset.card_univ, Fintype.card_prod, Fintype.card_fin]
    have hδ' : lvl + (m * 2 ^ m : ℝ) * lvl = δ := by
      rw [hlvl]; field_simp; ring
    linarith [hB0_le]
  refine (one_sub_le_compl_real μ bad δ hbad_le).trans (measureReal_mono ?_)
  intro ω hω
  simp only [hbad, Set.mem_compl_iff, Set.mem_union, not_or, Set.mem_iUnion, not_exists] at hω
  obtain ⟨hω0, hωp⟩ := hω
  simp only [Set.mem_ofPred_eq]
  intro b hmax
  set C := patternSet O.mq z G b ω with hC
  -- the pattern's size
  have hsize : (1 - η₀) ^ m * pAP * G.card - t ≤ C.card := by
    have h0 : (1 - η₀) ^ m * GAP.card - t ≤ ∑ v ∈ GAP, V v ω := by
      have := hω0; simp only [hB0, Set.mem_ofPred_eq, not_lt] at this; exact this
    have hsum : ∑ v ∈ GAP, V v ω
        = ((GAP.filter fun v => ∀ i, readsAs O.mq (z i * v) (tstar i) ω).card : ℝ) := by
      simp only [hV]; rw [Finset.sum_boole]
    have hsub : (GAP.filter fun v => ∀ i, readsAs O.mq (z i * v) (tstar i) ω)
        ⊆ patternSet O.mq z G tstar ω := by
      intro v hv
      simp only [patternSet, Finset.mem_filter, hGAP] at hv ⊢
      exact ⟨hv.1.1, hv.2⟩
    have hc1 := Finset.card_le_card hsub
    have hc2 := hmax tstar
    have hpow : 0 ≤ (1 - η₀) ^ m := pow_nonneg (by linarith) _
    have hmul := mul_le_mul_of_nonneg_left hAP hpow
    have : ((GAP.filter fun v => ∀ i, readsAs O.mq (z i * v) (tstar i) ω).card : ℝ)
        ≤ C.card := by exact_mod_cast hc1.trans hc2
    rw [mul_assoc]
    linarith
  refine ⟨hsize, fun i => ?_⟩
  set x := xr (b i) with hx
  set y := yr (b i) with hy
  obtain ⟨hx1, hy0, hyx⟩ := hxy (b i)
  set ρ : S → ℝ := fun v => μ.real {ω' | readsAs O.mq (z i * v) (b i) ω'} with hρdef
  -- the margin
  have hmargin : (1 / 2 + κ₀) * C.card - t ≤ ∑ v ∈ C, ρ v := by
    have hD : -t ≤ ∑ v ∈ G, anchorDev O z i b x y v ω := by
      have := hωp (i, b) (Finset.mem_univ _)
      simp only [hBp, Set.mem_ofPred_eq, not_lt] at this
      exact this
    set A := G.filter fun v => otherReads O.mq z i b v ω with hA
    set Hp : S → Prop := fun v => readsAs O.mq (z i * v) (b i) ω with hHp
    have hCA : C = A.filter Hp := by
      ext v
      rw [hC, hA, patternSet, Finset.mem_filter, Finset.mem_filter, Finset.mem_filter]
      simp only [hHp, otherReads]
      constructor
      · rintro ⟨hv, h⟩; exact ⟨⟨hv, fun j _ => h j⟩, h i⟩
      · rintro ⟨⟨hv, h⟩, hi⟩
        refine ⟨hv, fun j => ?_⟩
        by_cases hj : j = i
        · subst hj; exact hi
        · exact h j hj
    have hflip : patternSet O.mq z G (Function.update b i (!b i)) ω = A.filter fun v => ¬ Hp v := by
      ext v
      rw [hA, patternSet, Finset.mem_filter, Finset.mem_filter, Finset.mem_filter]
      simp only [hHp, otherReads]
      have hneg : ∀ s β, readsAs O.mq s (!β) ω ↔ ¬ readsAs O.mq s β ω := by
        intro s β; cases β <;> simp [readsAs]
      constructor
      · rintro ⟨hv, h⟩
        refine ⟨⟨hv, fun j hj => ?_⟩, ?_⟩
        · have := h j; rwa [Function.update_of_ne hj] at this
        · have := h i; rwa [Function.update_self, hneg] at this
      · rintro ⟨⟨hv, h⟩, hi⟩
        refine ⟨hv, fun j => ?_⟩
        by_cases hj : j = i
        · subst hj; rw [Function.update_self, hneg]; exact hi
        · rw [Function.update_of_ne hj]; exact h j hj
    have hcard : A.card = C.card + (A.filter fun v => ¬ Hp v).card := by
      rw [hCA]; exact (Finset.card_filter_add_card_filter_not _).symm
    have hle : (A.filter fun v => ¬ Hp v).card ≤ C.card := by
      rw [← hflip]; exact hmax _
    have hsumD : ∑ v ∈ G, anchorDev O z i b x y v ω
        = ∑ v ∈ A, (if Hp v then
            (if (z i * v ∈ O.L ↔ b i = true) then y * (x - 1) else x * (y - 1)) else x * y) := by
      rw [hA, Finset.sum_filter]
      refine Finset.sum_congr rfl fun v _ => ?_
      simp only [anchorDev, hHp]
    have hsumρ : ∑ v ∈ C, ρ v = ∑ v ∈ A, (if Hp v then ρ v else 0) := by
      rw [hCA, Finset.sum_filter]
    have hcardC : (C.card : ℝ) = ∑ v ∈ A, (if Hp v then (1 : ℝ) else 0) := by
      rw [hCA, Finset.sum_boole]
    have hcardA : (A.card : ℝ) = ∑ v ∈ A, (1 : ℝ) := by simp
    have hpt : ∀ v ∈ A, (if Hp v then ρ v else 0)
        = (x + y - 2 * x * y) * (if Hp v then (1 : ℝ) else 0)
          + 2 * x * y * ((if Hp v then (1 : ℝ) else 0) - 1 / 2)
          + (if Hp v then
              (if (z i * v ∈ O.L ↔ b i = true) then y * (x - 1) else x * (y - 1)) else x * y) := by
      intro v _
      have hr := hprob (z i * v) (b i)
      simp only [hρdef] at hr ⊢
      by_cases h1 : Hp v <;> by_cases h2 : (z i * v ∈ O.L ↔ b i = true) <;>
        simp only [h1, h2, if_true, if_false, hr] <;> ring
    have hid : ∑ v ∈ C, ρ v = (x + y - 2 * x * y) * C.card
        + 2 * x * y * (C.card - A.card / 2) + ∑ v ∈ G, anchorDev O z i b x y v ω := by
      rw [hsumρ, Finset.sum_congr rfl hpt, Finset.sum_add_distrib, Finset.sum_add_distrib,
        ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_sub_distrib, ← hcardC, hsumD]
      rw [hcardA]
      simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
      ring
    have hAC : (A.card : ℝ) ≤ 2 * C.card := by
      have : A.card ≤ 2 * C.card := by omega
      exact_mod_cast this
    have hxy0 : 0 ≤ 2 * x * y := by have := hy0.trans hyx; positivity
    have hKb := hK (b i)
    have hC0 : (0 : ℝ) ≤ C.card := Nat.cast_nonneg _
    rw [hid]
    nlinarith [mul_le_mul_of_nonneg_right hKb hC0,
      mul_nonneg hxy0 (by linarith : (0 : ℝ) ≤ C.card - A.card / 2)]
  refine ⟨hmargin, fun w hw W kl kh g hg hk h1 h2 => ?_⟩
  have hρw : ∀ v ∈ C, μ.real {ω' | readsAs O.mq (w * v) (b i) ω'} = ρ v := by
    intro v hv
    have hvG : v ∈ G := (Finset.mem_filter.1 hv).1
    simp only [hρdef]
    rw [hprob, hprob, propext (hw v hvG)]
  have hr : (1 / 2 + κ₀ - t / C.card) * C.card
      ≤ ∑ v ∈ C, μ.real {ω' | readsAs O.mq (w * v) (b i) ω'} := by
    rw [Finset.sum_congr rfl hρw]
    rcases Nat.eq_zero_or_pos C.card with hc | hc
    · rw [Finset.card_eq_zero.1 hc]; simp
    · have hcR : (0 : ℝ) < C.card := by exact_mod_cast hc
      rw [sub_mul, div_mul_cancel₀ _ hcR.ne']
      exact hmargin
  refine family_off_side_le O C w (b i) (1 / 2 + κ₀ - t / C.card) g W kl kh hr hg hk h1 ?_
  have e : (1 : ℝ) - (1 / 2 + κ₀ - t / C.card) + g = 1 / 2 - κ₀ + t / C.card + g := by ring
  rw [e]; exact h2

end OrthoDFA
