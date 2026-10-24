import OrthoDFA.Proofs.FamilyOracle
import OrthoDFA.Proofs.Level5
import OrthoDFA.Proofs.LoopAvoid
import OrthoDFA.Proofs.Renoise
set_option profiler true
set_option profiler.threshold 2000

/-!
# Level 3 read through a re-noised family

`SuffixFamily.renoise` reads the family through `O.renoise R`: a suffix family on two copies of
the noise, with the same read distributions, which agrees with the family off `R` at the first
copy (`renoise_read_of_notMem`). Level 5's guarantee holds of it (`level3Fail_le`): level 3 fails
to end well within `|Q| + 2` leaves, or for some state `q` and letter `a` the probes of a fixed set
`E q` reading past one of its tree's at most `|Q| + 1` midfixes, themselves or extended by `a`, on
their state's rarer side weigh more than half of `E q` plus `u`, with chance at most
`δ + |Q||Σ| misledTail T p₀ (2(|Q| + 1)κ) u`. Level 3 reads only strings beginning with a probe's
first `k` letters (`level3_congr_zone'`), so a probe whose first `k` letters are no probe's reads
independently of the tree, and of every probe with other first `k` letters; the others weigh at most
`T p₀`. Given the tree's midfixes the rest concentrate by Chebyshev (`meas_gt_le_of_pairs`,
`misled_tail`). A second copy can be fixed at a point where the bound still holds
(`exists_noise_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}

/-- The family read through `O.renoise R`. -/
def SuffixFamily.renoise (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) :
    SuffixFamily (μ.prod μ) α σ :=
  ⟨fam.O.renoise R, fam.M, fam.F, fam.kl, fam.kh, fam.κ, fam.lang, fam.suffixFree, fam.band⟩

lemma SuffixFamily.renoise_readDist (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) :
    (fam.renoise R).readDist = fam.readDist := by
  classical
  funext q
  unfold SuffixFamily.readDist
  show (if h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q then
      readProb (fam.O.renoise R) fam.F fam.kl fam.kh h.choose else fun _ => 0)
    = (if h : ∃ w : FreeMonoid α, fam.M.eval w.toList = q then
      readProb fam.O fam.F fam.kl fam.kh h.choose else fun _ => 0)
  split_ifs with h
  · funext rd
    rw [readProb_eq_readLaw, readProb_eq_readLaw]
    rfl
  · rfl

lemma renoise_read_of_notMem (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α))
    {z : FreeMonoid α} (hz : ∀ v ∈ fam.F, z * v ∉ R) (p : Ω × Ω) :
    (fam.renoise R).read p z = fam.read p.1 z := by
  unfold SuffixFamily.read SuffixFamily.renoise familyRead voteCount
  dsimp only
  congr 2
  refine Finset.filter_congr fun v hv => ?_
  rw [renoise_mq_of_notMem fam.O (hz v hv)]

section Zone

/-- The first `k` letters of the probes. -/
def probeKeys (k : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) : Finset (FreeMonoid α) :=
  Finset.univ.image fun i => prefixOf (xs i) k

open scoped Classical in
/-- The strings level 3 reads over the probes `xs`: short strings beginning with a probe's first
`k` letters, followed by the family's suffixes. -/
def level3Zone (k : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) (F : Finset (FreeMonoid α)) :
    Finset (FreeMonoid α) :=
  readStrings ((shortStrings (probesLen xs + 1 + T + 2)).filter fun z =>
    k ≤ z.toList.length ∧ prefixOf z k ∈ probeKeys k xs) F

omit [IsProbabilityMeasure μ] in
theorem level3_congr_zone' (O : Oracle μ (FreeMonoid α)) (C : Random.Level3Settings)
    (F : Finset (FreeMonoid α)) (kl kh : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) {ω ω' : Ω}
    (h : ∀ w ∈ level3Zone C.walkStart xs F, (O.mq w ω = 1 ↔ O.mq w ω' = 1)) :
    Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω) xs
      = Random.indecisiveLStarLevel3 C (fun z => familyRead O.mq F kl kh z ω') xs := by
  classical
  refine Random.level3_congr_zone C xs fun i j hj hjx m hm => familyRead_congr O h ?_
  have h1 := le_probesLen xs i
  have hlp : (prefixOf (xs i) j).toList.length = j := by
    simp [prefixOf]; omega
  refine Finset.mem_filter.2 ⟨mem_shortStrings.2 ?_, ?_, ?_⟩
  · simp only [FreeMonoid.toList_mul, List.length_append, hlp]; omega
  · simp only [FreeMonoid.toList_mul, List.length_append, hlp]; omega
  · rw [prefixOf_mul_of_le (by rw [hlp]; exact hj), prefixOf_prefixOf_of_le _ hj]
    exact Finset.mem_image.2 ⟨i, Finset.mem_univ _, rfl⟩

theorem prefixOf_of_mem_readStrings {k : ℕ} {Y F : Finset (FreeMonoid α)} {w : FreeMonoid α}
    (hw : w ∈ readStrings Y F) (hY : ∀ z ∈ Y, k ≤ z.toList.length) :
    ∃ z ∈ Y, prefixOf w k = prefixOf z k := by
  obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hz, -⟩ := Finset.mem_product.1 hzv
  exact ⟨z, hz, prefixOf_mul_of_le (hY z hz) v⟩

/-- Strings past their first `k` letters, which are no probe's, followed by the family's suffixes,
are none of level 3's strings. -/
theorem disjoint_level3Zone (k : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) (F : Finset (FreeMonoid α))
    {Y : Finset (FreeMonoid α)}
    (hY : ∀ z ∈ Y, k ≤ z.toList.length ∧ prefixOf z k ∉ probeKeys k xs) :
    Disjoint (level3Zone k xs F) (readStrings Y F) := by
  classical
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨z', hz', he'⟩ := prefixOf_of_mem_readStrings hw' fun z hz => (hY z hz).1
  obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨hz, -⟩ := Finset.mem_product.1 hzv
  obtain ⟨-, hzk, hkz⟩ := Finset.mem_filter.1 hz
  refine (hY z' hz').2 ?_
  rw [← he', prefixOf_mul_of_le hzk]
  exact hkz

/-- Strings past different first `k` letters, followed by the family's suffixes, differ. -/
theorem disjoint_readStrings_of_keys (k : ℕ) {Y Y' : Finset (FreeMonoid α)}
    (F : Finset (FreeMonoid α)) (hY : ∀ z ∈ Y, k ≤ z.toList.length)
    (hY' : ∀ z ∈ Y', k ≤ z.toList.length)
    (hne : ∀ z ∈ Y, ∀ z' ∈ Y', prefixOf z k ≠ prefixOf z' k) :
    Disjoint (readStrings Y F) (readStrings Y' F) := by
  refine Finset.disjoint_left.2 fun w hw hw' => ?_
  obtain ⟨z, hz, he⟩ := prefixOf_of_mem_readStrings hw hY
  obtain ⟨z', hz', he'⟩ := prefixOf_of_mem_readStrings hw' hY'
  exact hne z hz z' hz' (he.symm.trans he')

end Zone


section Chebyshev

/-- Chebyshev on `H`, for `X(ω) = D{y | ω ∈ B y}` whose events are independent of `H`, and of
each other where their keys differ: where each key holds at most `p` of `D` and
`Σ_y D{y} μ(B y) ≤ e < u`,

    μ(H ∩ {X > u}) ≤ μ(H) p e/(u − e)². -/
theorem meas_gt_le_of_pairs {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    [IsProbabilityMeasure μ] {Y : Type*} [MeasurableSpace Y] [Countable Y]
    [MeasurableSingletonClass Y] (D : Measure Y) [IsFiniteMeasure D] {K : Type*} (key : Y → K)
    (H : Set Ω) (B : Y → Set Ω) (hB : ∀ y, MeasurableSet (B y))
    (h1 : ∀ y, μ (H ∩ B y) = μ H * μ (B y))
    (h2 : ∀ y y', key y ≠ key y' → μ (H ∩ (B y ∩ B y')) = μ H * (μ (B y) * μ (B y')))
    {p e u : ℝ} (hp : 0 ≤ p) (he0 : 0 ≤ e) (hkey : ∀ y, D {y' | key y' = key y} ≤ ENNReal.ofReal p)
    (he : ∑' y, D {y} * μ (B y) ≤ ENNReal.ofReal e) (heu : e < u) :
    μ (H ∩ {ω | u < D.real {y | ω ∈ B y}}) ≤ μ H * ENNReal.ofReal (p * e / (u - e) ^ 2) := by
  classical
  set Xe : Ω → ℝ≥0∞ := fun ω => D {y | ω ∈ B y} with hXed
  have hXe : ∀ ω, Xe ω = ∑' y, (B y).indicator (fun _ => D {y}) ω := by
    intro ω
    show D {y | ω ∈ B y} = _
    rw [← Measure.tsum_indicator_apply_singleton D _ ((Set.to_countable _).measurableSet)]
    exact tsum_congr fun y => by simp only [Set.indicator]; rfl
  have hXm : Measurable Xe := measurable_measure_of_mem D (fun ω => {y | ω ∈ B y}) hB
  set Mm : ℝ≥0∞ := ∑' y, D {y} * μ (B y) with hMm
  have hMfin : Mm ≠ ∞ := (he.trans_lt ENNReal.ofReal_lt_top).ne
  -- the first moment on `H`
  have hm1 : ∫⁻ ω in H, Xe ω ∂μ = μ H * Mm := by
    simp_rw [hXe]
    rw [lintegral_tsum fun y => (measurable_const.indicator (hB y)).aemeasurable,
      ← ENNReal.tsum_mul_left]
    refine tsum_congr fun y => ?_
    rw [lintegral_indicator_const (hB y), Measure.restrict_apply (hB y), Set.inter_comm, h1]
    ring
  -- the second moment on `H`: pairs with distinct keys are independent
  have hm2 : ∫⁻ ω in H, Xe ω ^ 2 ∂μ ≤ μ H * (Mm ^ 2 + ENNReal.ofReal p * Mm) := by
    have hsq : ∀ ω, Xe ω ^ 2 = ∑' y, ∑' y', (B y ∩ B y').indicator (fun _ => D {y} * D {y'}) ω := by
      intro ω
      rw [sq, hXe, ← ENNReal.tsum_mul_right]
      refine tsum_congr fun y => ?_
      rw [← ENNReal.tsum_mul_left]
      refine tsum_congr fun y' => ?_
      simp only [Set.indicator, Set.mem_inter_iff]
      split_ifs <;> simp_all
    have hterm : ∀ y y', ∫⁻ ω in H, (B y ∩ B y').indicator (fun _ => D {y} * D {y'}) ω ∂μ
        ≤ μ H * (D {y} * μ (B y)) * (D {y'} * μ (B y'))
          + μ H * (D {y} * μ (B y)) * {y'' | key y'' = key y}.indicator (fun x => D {x}) y' := by
      intro y y'
      rw [lintegral_indicator_const ((hB y).inter (hB y')),
        Measure.restrict_apply ((hB y).inter (hB y')), Set.inter_comm]
      by_cases hk : key y = key y'
      · have hy' : y' ∈ {y'' | key y'' = key y} := hk.symm
        rw [Set.indicator_of_mem hy']
        refine le_add_left ?_
        calc D {y} * D {y'} * μ (H ∩ (B y ∩ B y'))
            ≤ D {y} * D {y'} * μ (H ∩ B y) := by
              gcongr; exact Set.inter_subset_left
          _ = _ := by rw [h1]; ring
      · refine le_add_right ?_
        rw [h2 y y' hk]
        exact le_of_eq (by ring)
    simp_rw [hsq]
    rw [lintegral_tsum fun y => (Measurable.tsum fun y' =>
      measurable_const.indicator ((hB y).inter (hB y'))).aemeasurable]
    calc ∑' y, ∫⁻ ω in H, ∑' y', (B y ∩ B y').indicator (fun _ => D {y} * D {y'}) ω ∂μ
        = ∑' y, ∑' y', ∫⁻ ω in H, (B y ∩ B y').indicator (fun _ => D {y} * D {y'}) ω ∂μ :=
          tsum_congr fun y => lintegral_tsum fun y' =>
            (measurable_const.indicator ((hB y).inter (hB y'))).aemeasurable
      _ ≤ ∑' y, ∑' y', (μ H * (D {y} * μ (B y)) * (D {y'} * μ (B y'))
          + μ H * (D {y} * μ (B y)) * {y'' | key y'' = key y}.indicator (fun x => D {x}) y') :=
          ENNReal.tsum_le_tsum fun y => ENNReal.tsum_le_tsum fun y' => hterm y y'
      _ = ∑' y, (μ H * (D {y} * μ (B y)) * Mm
          + μ H * (D {y} * μ (B y)) * D {y'' | key y'' = key y}) := by
          refine tsum_congr fun y => ?_
          rw [ENNReal.tsum_add, ENNReal.tsum_mul_left, ENNReal.tsum_mul_left,
            Measure.tsum_indicator_apply_singleton D _ ((Set.to_countable _).measurableSet)]
      _ ≤ ∑' y, (μ H * (D {y} * μ (B y)) * Mm + μ H * (D {y} * μ (B y)) * ENNReal.ofReal p) :=
          ENNReal.tsum_le_tsum fun y => by gcongr; exact hkey y
      _ = μ H * (Mm ^ 2 + ENNReal.ofReal p * Mm) := by
          rw [ENNReal.tsum_add, ENNReal.tsum_mul_right, ENNReal.tsum_mul_right,
            ENNReal.tsum_mul_left]
          ring
  -- Chebyshev about the mean `m` on `H`
  set m : ℝ := Mm.toReal with hmd
  have hm0 : 0 ≤ m := ENNReal.toReal_nonneg
  have hMm : Mm = ENNReal.ofReal m := (ENNReal.ofReal_toReal hMfin).symm
  have hme : m ≤ e := ENNReal.toReal_le_of_le_ofReal he0 he
  set h : ℝ := μ.real H with hhd
  have hH : μ H = ENNReal.ofReal h := (ofReal_measureReal (measure_ne_top _ _)).symm
  set X : Ω → ℝ := fun ω => D.real {y | ω ∈ B y} with hXd
  have hXX : ∀ ω, Xe ω = ENNReal.ofReal (X ω) := fun ω =>
    (ofReal_measureReal (measure_ne_top _ _)).symm
  have hX0 : ∀ ω, 0 ≤ X ω := fun ω => measureReal_nonneg
  set g : Ω → ℝ≥0∞ := fun ω => ENNReal.ofReal ((X ω - m) ^ 2) with hgd
  have hgm : Measurable g :=
    ENNReal.measurable_ofReal.comp (((ENNReal.measurable_toReal.comp hXm).sub_const m).pow_const 2)
  have hpt : ∀ ω, g ω + ENNReal.ofReal (2 * m) * Xe ω = Xe ω ^ 2 + ENNReal.ofReal (m ^ 2) := by
    intro ω
    rw [hXX, hgd, ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_pow (hX0 ω),
      ← ENNReal.ofReal_add (by positivity) (by have := hX0 ω; positivity),
      ← ENNReal.ofReal_add (by have := hX0 ω; positivity) (by positivity)]
    congr 1
    ring
  have hG : ∫⁻ ω in H, g ω ∂μ ≤ ENNReal.ofReal (h * p * m) := by
    have e1 : ∫⁻ ω in H, g ω ∂μ + ENNReal.ofReal (2 * m) * (μ H * Mm)
        = ∫⁻ ω in H, Xe ω ^ 2 ∂μ + ENNReal.ofReal (m ^ 2) * μ H := by
      rw [← hm1, ← lintegral_const_mul _ hXm, ← lintegral_add_left hgm]
      simp_rw [hpt]
      rw [lintegral_add_right _ measurable_const, setLIntegral_const]
    have e2 : ∫⁻ ω in H, g ω ∂μ + ENNReal.ofReal (2 * h * m ^ 2)
        ≤ ENNReal.ofReal (h * p * m) + ENNReal.ofReal (2 * h * m ^ 2) := by
      have hh0 : 0 ≤ h := measureReal_nonneg
      calc ∫⁻ ω in H, g ω ∂μ + ENNReal.ofReal (2 * h * m ^ 2)
          = ∫⁻ ω in H, g ω ∂μ + ENNReal.ofReal (2 * m) * (μ H * Mm) := by
            rw [hH, hMm, ← ENNReal.ofReal_mul hh0, ← ENNReal.ofReal_mul (by positivity)]
            congr 2; ring
        _ = ∫⁻ ω in H, Xe ω ^ 2 ∂μ + ENNReal.ofReal (m ^ 2) * μ H := e1
        _ ≤ μ H * (Mm ^ 2 + ENNReal.ofReal p * Mm) + ENNReal.ofReal (m ^ 2) * μ H := by gcongr
        _ = ENNReal.ofReal (h * p * m) + ENNReal.ofReal (2 * h * m ^ 2) := by
            rw [hH, hMm, ← ENNReal.ofReal_pow hm0, ← ENNReal.ofReal_mul hp,
              ← ENNReal.ofReal_add (by positivity) (by positivity), ← ENNReal.ofReal_mul hh0,
              ← ENNReal.ofReal_mul (by positivity),
              ← ENNReal.ofReal_add (by positivity) (by positivity),
              ← ENNReal.ofReal_add (by positivity) (by positivity)]
            congr 1; ring
    exact (ENNReal.add_le_add_iff_right ENNReal.ofReal_ne_top).1 e2
  have hum : 0 < u - m := by linarith
  have hsub : H ∩ {ω | u < X ω} ⊆ {ω | ENNReal.ofReal ((u - m) ^ 2) ≤ g ω} ∩ H := by
    rintro ω ⟨hωH, hω⟩
    refine ⟨ENNReal.ofReal_le_ofReal ?_, hωH⟩
    have : u - m ≤ X ω - m := by simp only [Set.mem_ofPred_eq] at hω; linarith
    nlinarith
  have hcheb := mul_meas_ge_le_lintegral₀ (μ := μ.restrict H) hgm.aemeasurable
    (ENNReal.ofReal ((u - m) ^ 2))
  rw [Measure.restrict_apply (measurableSet_le measurable_const hgm)] at hcheb
  have hpos : 0 < (u - m) ^ 2 := by positivity
  have hh0 : 0 ≤ h := measureReal_nonneg
  calc μ (H ∩ {ω | u < D.real {y | ω ∈ B y}})
      ≤ μ ({ω | ENNReal.ofReal ((u - m) ^ 2) ≤ g ω} ∩ H) := measure_mono hsub
    _ ≤ ENNReal.ofReal (h * p * m) / ENNReal.ofReal ((u - m) ^ 2) := by
        rw [ENNReal.le_div_iff_mul_le (Or.inl (ENNReal.ofReal_pos.2 hpos).ne')
          (Or.inl ENNReal.ofReal_ne_top),
          mul_comm]
        exact hcheb.trans hG
    _ = ENNReal.ofReal (h * (p * m / (u - m) ^ 2)) := by
        rw [← ENNReal.ofReal_div_of_pos hpos]; congr 1; ring
    _ ≤ ENNReal.ofReal (h * (p * e / (u - e) ^ 2)) := by
        refine ENNReal.ofReal_le_ofReal (mul_le_mul_of_nonneg_left ?_ hh0)
        have hue : 0 < (u - e) ^ 2 := by have : 0 < u - e := by linarith
                                         positivity
        exact div_le_div₀ (by positivity) (mul_le_mul_of_nonneg_left hme hp) hue
          (pow_le_pow_left₀ (by linarith) (by linarith) 2)
    _ = μ H * ENNReal.ofReal (p * e / (u - e) ^ 2) := by
        rw [hH, ENNReal.ofReal_mul hh0]

end Chebyshev

section Fail

variable {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'} [IsProbabilityMeasure μ']

variable [Fintype σ]

/-- Level 5's harvest slope at `f = 0`, at probes at most `L` long. -/
def roundSlope (σ : Type*) [Fintype σ] (L : ℕ) (C : Random.Level3Settings) : ℝ :=
  (Fintype.card σ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 L))

/-- Level 5's harvest slack at `f = 0`, at the prefix mass `p₀`. -/
def roundSlack (σ α : Type*) [Fintype σ] [Fintype α] (C : Random.Level3Settings) (p₀ : ℝ) : ℝ :=
  p₀ * Real.log (16 * (Fintype.card σ + 1) * ((1 + (Fintype.card σ + 1) ^ 3 * Fintype.card α)
    ^ Fintype.card σ * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) : ℕ)
      / C.δ) / C.harvestRate

/-- Where level 3, reading `fam` over probes from `D'`, fails to end well over `disagreeSet`, its
tree leaves `classSet |Q|`, or more than half of some `E q`, plus `u`, is misled. -/
def level3Fail (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (D' : Measure (FreeMonoid α)) (A sl : ℝ) (E : σ → Set (FreeMonoid α)) (u : ℝ) :
    Set (Ω' × (Fin C.probeBudget → FreeMonoid α)) :=
  {d | let result := Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2
    ¬ (EndsWell
      { target := fam.M,
        undecided := fun q => fam.readDist q .undecided,
        harvestSlope := A,
        harvestSlack := sl,
        probes := D',
        disagreeRate := C.disagreementRate,
        disagree := disagreeSet (fun z => fam.read d.1 z) C.walkStart result.hypothesis,
        walkStart := C.walkStart,
        harvestLaw := Random.harvestLaw (fun z => fam.read d.1 z) D' C.walkStart
          result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending }
      ∧ result.hypothesis.tree ∈ Random.classSet (Fintype.card σ)
      ∧ FewMisled fam.M fam.readDist (fun z => fam.read d.1 z) C.walkStart
        result.hypothesis.tree D' E u)}

/-- A property of the noise and level 3's result over the probes is measurable where it is at
each result. -/
theorem measurableSet_of_result (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (Φ : Ω' → Random.Level3Result α → Prop) (hΦ : ∀ res, MeasurableSet {ω | Φ ω res}) :
    MeasurableSet {d : Ω' × (Fin C.probeBudget → FreeMonoid α) |
      Φ d.1 (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2)} := by
  classical
  obtain ⟨ω₀⟩ := nonempty_of_isProbabilityMeasure μ'
  refine measurableSet_prod_countable_right _ fun xs => ?_
  set A := readStrings (shortStrings (probesLen xs + 1 + C.probeBudget + 2)) fam.F
  set res : Ω' → Random.Level3Result α := fun ω =>
    Random.indecisiveLStarLevel3 C (fun z => familyRead fam.O.mq fam.F fam.kl fam.kh z ω) xs
  have hres : ∀ ω ω', (∀ p ∈ A, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → res ω = res ω' :=
    fun ω ω' h => level3_result_congr fam.O C fam.F _ _ xs h
  set rep : Finset (FreeMonoid α) → Random.Level3Result α := fun U =>
    if h : ∃ ω, A.filter (fun p => fam.O.mq p ω = 1) = U then res h.choose else res ω₀
  have hrepω : ∀ ω, res ω = rep (A.filter (fun p => fam.O.mq p ω = 1)) := by
    intro ω
    have h : ∃ ω', A.filter (fun p => fam.O.mq p ω' = 1) = A.filter (fun p => fam.O.mq p ω = 1) :=
      ⟨ω, rfl⟩
    simp only [rep, dif_pos h]
    refine hres _ _ fun p hp => ?_
    have := Finset.ext_iff.1 h.choose_spec p
    simp only [Finset.mem_filter, hp, true_and] at this
    exact this.symm
  have hset : {ω | Φ ω (Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs)}
      = ⋃ U ∈ A.powerset, ({ω | A.filter (fun p => fam.O.mq p ω = 1) = U}
        ∩ {ω | Φ ω (rep U)}) := by
    ext ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion, Set.mem_inter_iff, Finset.mem_powerset,
      exists_prop]
    constructor
    · intro h
      refine ⟨_, Finset.filter_subset _ _, rfl, ?_⟩
      rw [← hrepω ω]
      exact h
    · rintro ⟨U, -, hU, h⟩
      rw [← hU, ← hrepω ω] at h
      exact h
  refine (congrArg MeasurableSet hset).mpr ?_
  refine Finset.measurableSet_biUnion _ fun U _ => MeasurableSet.inter ?_ (hΦ _)
  exact measurableSet_of_reads fam.O A _ fun ω ω' h (hω : A.filter _ = U) => by
    show A.filter _ = U
    rw [← hω]
    exact Finset.filter_congr fun p hp => (h p hp).symm

omit [Fintype σ] in
theorem measurableSet_mem_misledSet (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (r : Ω' → FreeMonoid α → ARU) (hr : ∀ z, Measurable fun ω => r ω z) (k : ℕ) (T : DTree α)
    (a : α) (y : FreeMonoid α) : MeasurableSet {ω | y ∈ misledSet M dist (r ω) k T a} := by
  have hR : ∀ z, MeasurableSet {ω | z ∈ rarerSet M dist (r ω) k T} := fun z => by
    have e : {ω | z ∈ rarerSet M dist (r ω) k T} = {_ω | k ≤ z.toList.length} ∩
        ⋃ m ∈ Random.mids T, {ω | r ω (z * m)
          = if readSide (dist (M.eval (z * m).toList)) then .reject else .accept} := by
      ext ω
      simp [rarerSet]
    rw [e]
    exact (MeasurableSet.const _).inter (Finset.measurableSet_biUnion _ fun m _ =>
      hr _ (measurableSet_singleton _))
  exact (MeasurableSet.const _).inter ((hR y).union (hR _))

theorem measurableSet_fewMisled (M : _root_.DFA α σ) (dist : σ → ARU → ℝ)
    (r : Ω' → FreeMonoid α → ARU) (hr : ∀ z, Measurable fun ω => r ω z) (k : ℕ) (T : DTree α)
    (D : Measure (FreeMonoid α)) (E : σ → Set (FreeMonoid α)) (u : ℝ) :
    MeasurableSet {ω | FewMisled M dist (r ω) k T D E u} := by
  have e : {ω | FewMisled M dist (r ω) k T D E u} = ⋂ q, ⋂ a,
      {ω | D.real (E q ∩ misledSet M dist (r ω) k T a) ≤ D.real (E q) / 2 + u} := by
    ext ω
    simp [FewMisled]
  rw [e]
  refine MeasurableSet.iInter fun q => MeasurableSet.iInter fun a =>
    measurableSet_le ?_ measurable_const
  exact ENNReal.measurable_toReal.comp (measurable_measure_of_mem D _ fun y =>
    (MeasurableSet.const (y ∈ E q)).inter (measurableSet_mem_misledSet M dist r hr k T a y))

theorem measurableSet_level3Fail (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (D' : Measure (FreeMonoid α)) {A sl : ℝ} (hA : 0 ≤ A) (hsl : 0 ≤ sl)
    (E : σ → Set (FreeMonoid α)) (u : ℝ) :
    MeasurableSet (level3Fail fam C D' A sl E u) := by
  classical
  refine measurableSet_of_result fam C (fun ω result => ¬ (EndsWell
    { target := fam.M,
      undecided := fun q => fam.readDist q .undecided,
      harvestSlope := A,
      harvestSlack := sl,
      probes := D',
      disagreeRate := C.disagreementRate,
      disagree := disagreeSet (fun z => fam.read ω z) C.walkStart result.hypothesis,
      walkStart := C.walkStart,
      harvestLaw := Random.harvestLaw (fun z => fam.read ω z) D' C.walkStart result.hypothesis,
      yield := C.harvestRate,
      ending := result.ending } ∧ result.hypothesis.tree ∈ Random.classSet (Fintype.card σ)
      ∧ FewMisled fam.M fam.readDist (fun z => fam.read ω z) C.walkStart
        result.hypothesis.tree D' E u))
    fun res => ?_
  simp only [not_and_or]
  exact (measurableSet_not_endsWell fam.O fam.F _ _ C.walkStart fam.M _
    (fun q => fam.readDist_nonneg q _) hA hsl D' _ _ _ _).union
    ((MeasurableSet.const _).union (measurableSet_fewMisled fam.M fam.readDist
      (fun ω z => fam.read ω z) (fun z => fam.read_measurable z) _ _ D' E u).compl)

theorem readDist_le_one (fam : SuffixFamily μ' α σ) (q : σ) (r : ARU) : fam.readDist q r ≤ 1 := by
  classical
  unfold SuffixFamily.readDist
  split_ifs
  · exact measureReal_le_one
  · norm_num

/-- The chance more than half of a set, plus `u`, is misled: `p e/((2 − 4e)(u − T p))` where
`T p < u` and `e < 1/2`, and `1` otherwise. -/
def misledTail (T p e u : ℝ) : ℝ :=
  if T * p < u ∧ e < 1 / 2 then p * e / ((2 - 4 * e) * (u - T * p)) else 1

lemma misledTail_nonneg {T p e u : ℝ} (hp : 0 ≤ p) (he : 0 ≤ e) : 0 ≤ misledTail T p e u := by
  unfold misledTail
  split_ifs with h
  · have : 0 < 2 - 4 * e := by linarith [h.2]
    have : 0 < u - T * p := by linarith [h.1]
    positivity
  · norm_num

/-- Chebyshev's bound at a set of mass `x`, mean `a x` and threshold `x/2 + d`, whatever `x`. -/
lemma cheb_half_le {p a x d : ℝ} (hp : 0 ≤ p) (ha : 0 ≤ a) (ha2 : a < 1 / 2) (hx : 0 ≤ x)
    (hd : 0 < d) : p * (a * x) / (x / 2 + d - a * x) ^ 2 ≤ p * a / ((2 - 4 * a) * d) := by
  set b := 1 / 2 - a with hb
  have hb0 : 0 < b := by linarith
  have e : x / 2 + d - a * x = b * x + d := by rw [hb]; ring
  have e2 : 2 - 4 * a = 4 * b := by rw [hb]; ring
  rw [e, e2, div_le_div_iff₀ (by positivity) (by positivity)]
  have h4 : 4 * b * x * d ≤ (b * x + d) ^ 2 := by nlinarith [sq_nonneg (b * x - d)]
  have hpa : 0 ≤ p * a := mul_nonneg hp ha
  nlinarith [mul_le_mul_of_nonneg_left h4 hpa]

/-- Where level 3's tree is in `classSet |Q|`, the probes of `E` misled at `a` weigh more than half
of `E`, plus `u`, with chance at most `misledTail T p₀ (2(|Q| + 1) w) u`, where no length-`k`
prefix holds more than `p₀` of the probes and each string reads on its state's rarer side with
chance at most `w`. Those whose first `k` letters are a probe's weigh at most `T p₀`. The rest
read independently of the tree, and of each other where their first `k` letters differ. -/
theorem misled_tail (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings) {w : ℝ} (hw0 : 0 ≤ w)
    (hw : ∀ z, μ'.real {ω | fam.read ω z
      = if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept} ≤ w)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {p₀ : ℝ} (hp₀ : 0 ≤ p₀)
    (hpre : ∀ v, D.real {x | prefixOf x C.walkStart = v} ≤ p₀) (E : Set (FreeMonoid α)) (a : α)
    (u : ℝ) :
    (μ'.prod (Measure.pi fun _ : Fin C.probeBudget => D)) {d |
      let T := (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree
      T ∈ Random.classSet (Fintype.card σ) ∧ D.real E / 2 + u
        < D.real (E ∩ misledSet fam.M fam.readDist (fun z => fam.read d.1 z) C.walkStart T a)}
      ≤ ENNReal.ofReal (misledTail C.probeBudget p₀ (2 * (Fintype.card σ + 1) * w) u) := by
  classical
  set k := C.walkStart
  set nQ := Fintype.card σ
  set T := C.probeBudget
  set π := Measure.pi fun _ : Fin T => D
  set e : ℝ := 2 * (nQ + 1) * w with hed
  have he0 : 0 ≤ e := by positivity
  unfold misledTail
  split_ifs with hu
  swap
  · rw [ENNReal.ofReal_one]; exact prob_le_one
  obtain ⟨hTu, he2⟩ := hu
  set wrong : FreeMonoid α → ARU := fun z =>
    if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept
  set Ev : Set (Ω' × (Fin T → FreeMonoid α)) := {d |
      (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree
        ∈ Random.classSet nQ
      ∧ D.real E / 2 + u < D.real (E ∩ misledSet fam.M fam.readDist (fun z => fam.read d.1 z) k
        (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree a)}
    with hEv
  change (μ'.prod π) Ev ≤ ENNReal.ofReal (p₀ * e / ((2 - 4 * e) * (u - T * p₀)))
  have hmeas : MeasurableSet Ev :=
    measurableSet_of_result fam C (fun ω res => res.hypothesis.tree ∈ Random.classSet nQ
      ∧ D.real E / 2 + u < D.real (E ∩ misledSet fam.M fam.readDist (fun z => fam.read ω z) k
        res.hypothesis.tree a)) fun res =>
      (MeasurableSet.const _).inter (measurableSet_lt measurable_const
        (ENNReal.measurable_toReal.comp (measurable_measure_of_mem D _ fun y =>
          (MeasurableSet.const (y ∈ E)).inter (measurableSet_mem_misledSet fam.M fam.readDist
            (fun ω z => fam.read ω z) (fun z => fam.read_measurable z) k _ a y))))
  -- the probes sharing their first `k` letters with a probe
  have hcoll : ∀ xs : Fin T → FreeMonoid α,
      D.real {y | prefixOf y k ∈ probeKeys k xs} ≤ T * p₀ := by
    intro xs
    have e : {y | prefixOf y k ∈ probeKeys k xs} = ⋃ i, {y | prefixOf y k = prefixOf (xs i) k} := by
      ext y; simp [probeKeys, eq_comm]
    rw [e]
    refine (measureReal_iUnion_fintype_le _).trans ?_
    calc ∑ i, D.real {y | prefixOf y k = prefixOf (xs i) k}
        ≤ ∑ _i : Fin T, p₀ := Finset.sum_le_sum fun i _ => hpre _
      _ = T * p₀ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  set c : ℝ := p₀ * e / ((2 - 4 * e) * (u - T * p₀)) with hcd
  set x : ℝ := D.real E with hxd
  have hx0 : 0 ≤ x := measureReal_nonneg
  set DE := D.restrict E with hDE
  set u' : ℝ := x / 2 + (u - T * p₀) with hu'd
  -- over given probes
  have hfix : ∀ xs : Fin T → FreeMonoid α, μ' {ω | (ω, xs) ∈ Ev} ≤ ENNReal.ofReal c := by
    intro xs
    set Z := level3Zone k xs fam.F
    set tr : Ω' → DTree α := fun ω =>
      (Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs).hypothesis.tree
    have htr : ∀ ω ω', (∀ p ∈ Z, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → tr ω = tr ω' :=
      fun ω ω' h => congrArg (fun r => r.hypothesis.tree)
        (level3_congr_zone' fam.O C fam.F fam.kl fam.kh xs h)
    set good : FreeMonoid α → Prop := fun y =>
      k ≤ y.toList.length ∧ prefixOf y k ∉ probeKeys k xs
    set H : Finset (FreeMonoid α) → Set Ω' := fun M =>
      {ω | tr ω ∈ Random.classSet nQ ∧ Random.mids (tr ω) = M}
    set Ys : Finset (FreeMonoid α) → FreeMonoid α → Finset (FreeMonoid α) := fun M y =>
      M.image (y * ·) ∪ M.image (y * FreeMonoid.of a * ·)
    set B : Finset (FreeMonoid α) → FreeMonoid α → Set Ω' := fun M y =>
      {ω | good y ∧ ∃ z ∈ Ys M y, fam.read ω z = wrong z}
    set Ms := (shortStrings (α := α) nQ).powerset.filter fun M => M.card ≤ nQ + 1
    have hHc : ∀ M ω ω', (∀ p ∈ Z, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → ω ∈ H M →
        ω' ∈ H M := fun M ω ω' h hω => by
      simp only [H, Set.mem_ofPred_eq, ← htr ω ω' h]
      exact hω
    have hBc : ∀ M y (Y : Finset (FreeMonoid α)), Ys M y ⊆ Y → ∀ ω ω',
        (∀ p ∈ readStrings Y fam.F, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → ω ∈ B M y →
        ω' ∈ B M y := by
      rintro M y Y hY ω ω' h ⟨hg, z, hz, hr⟩
      refine ⟨hg, z, hz, ?_⟩
      have : fam.read ω z = fam.read ω' z := familyRead_congr fam.O h (hY hz)
      rw [← this]
      exact hr
    have hHm : ∀ M, MeasurableSet (H M) := fun M => measurableSet_of_reads fam.O Z (H M) (hHc M)
    have hBm : ∀ M y, MeasurableSet (B M y) := fun M y =>
      measurableSet_of_reads fam.O _ (B M y) (hBc M y _ subset_rfl)
    have hgood : ∀ {y}, good y → ∀ M : Finset (FreeMonoid α), ∀ z ∈ Ys M y,
        k ≤ z.toList.length ∧ prefixOf z k = prefixOf y k := by
      intro y hy M z hz
      rcases Finset.mem_union.1 hz with hz | hz
      · obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
        refine ⟨?_, prefixOf_mul_of_le hy.1 m⟩
        simp only [FreeMonoid.toList_mul, List.length_append]
        exact le_add_right hy.1
      · obtain ⟨m, -, rfl⟩ := Finset.mem_image.1 hz
        refine ⟨?_, by rw [mul_assoc]; exact prefixOf_mul_of_le hy.1 _⟩
        simp only [FreeMonoid.toList_mul, List.length_append]
        omega
    have hBe : ∀ M y, ¬ good y → B M y = ∅ := fun M y hy =>
      Set.eq_empty_of_forall_notMem fun ω hω => hy hω.1
    -- off the probes sharing a key, the rest given the tree's midfixes
    have hsub : {ω | (ω, xs) ∈ Ev}
        ⊆ ⋃ M ∈ Ms, H M ∩ {ω | u' < DE.real {y | ω ∈ B M y}} := by
      rintro ω ⟨hc, hω⟩
      have hc : tr ω ∈ Random.classSet nQ := hc
      have hω : D.real E / 2 + u < D.real (E ∩ misledSet fam.M fam.readDist
          (fun z => fam.read ω z) k (tr ω) a) := hω
      have hM : Random.mids (tr ω) ∈ Ms := by
        refine Finset.mem_filter.2 ⟨Finset.mem_powerset.2 fun m hm =>
          mem_shortStrings.2 ((Random.classSet_props _ _ hc).1 m hm), ?_⟩
        have h1 := Random.mids_card (tr ω)
        have h2 := (Random.classSet_props _ _ hc).2
        omega
      refine Set.mem_biUnion hM ⟨⟨hc, rfl⟩, ?_⟩
      have hS : E ∩ misledSet fam.M fam.readDist (fun z => fam.read ω z) k (tr ω) a
          ⊆ {y | prefixOf y k ∈ probeKeys k xs} ∪ {y | ω ∈ B (Random.mids (tr ω)) y} ∩ E := by
        rintro y ⟨hyE, hk, hr⟩
        by_cases hp : prefixOf y k ∈ probeKeys k xs
        · exact Or.inl hp
        · refine Or.inr ⟨⟨⟨hk, hp⟩, ?_⟩, hyE⟩
          rcases hr with ⟨-, m, hm, hr⟩ | ⟨-, m, hm, hr⟩
          · exact ⟨y * m, Finset.mem_union_left _ (Finset.mem_image_of_mem _ hm), hr⟩
          · exact ⟨y * FreeMonoid.of a * m,
              Finset.mem_union_right _ (Finset.mem_image_of_mem _ hm), hr⟩
      have h1 := (measureReal_mono (μ := D) hS).trans (measureReal_union_le _ _)
      have h2 : D.real ({y | ω ∈ B (Random.mids (tr ω)) y} ∩ E)
          = DE.real {y | ω ∈ B (Random.mids (tr ω)) y} :=
        (measureReal_restrict_apply (measurableSet_of_countable _)).symm
      show u' < _
      rw [hu'd]
      linarith [hcoll xs]
    -- each set of midfixes, by Chebyshev
    have hM : ∀ M ∈ Ms, μ' (H M ∩ {ω | u' < DE.real {y | ω ∈ B M y}})
        ≤ μ' (H M) * ENNReal.ofReal c := by
      intro M hM
      have hMc : M.card ≤ nQ + 1 := (Finset.mem_filter.1 hM).2
      have heu : e * x < u' := by
        rw [hu'd]
        have : e * x ≤ x / 2 := by nlinarith
        linarith
      refine (meas_gt_le_of_pairs (μ := μ') DE (fun y => prefixOf y k) (H M) (B M) (hBm M)
        (fun y => ?_) (fun y y' hyy => ?_) hp₀ (mul_nonneg he0 hx0) (fun y => ?_) ?_ heu).trans ?_
      · by_cases hy : good y
        · exact measure_inter_of_reads fam.O
            (disjoint_level3Zone k xs fam.F fun z hz => by
              obtain ⟨h1, h2⟩ := hgood hy M z hz
              exact ⟨h1, h2 ▸ hy.2⟩) _ _ (hHc M) (hBc M y _ subset_rfl)
        · simp [hBe M y hy]
      · by_cases hy : good y
        · by_cases hy' : good y'
          · have hd := disjoint_readStrings_of_keys k fam.F (fun z hz => (hgood hy M z hz).1)
              (fun z hz => (hgood hy' M z hz).1) fun z hz z' hz' => by
                rw [(hgood hy M z hz).2, (hgood hy' M z' hz').2]; exact hyy
            rw [measure_inter_of_reads fam.O
              (disjoint_level3Zone k xs fam.F (Y := Ys M y ∪ Ys M y')
                fun z hz => by
                  rcases Finset.mem_union.1 hz with hz | hz
                  · obtain ⟨h1, h2⟩ := hgood hy M z hz; exact ⟨h1, h2 ▸ hy.2⟩
                  · obtain ⟨h1, h2⟩ := hgood hy' M z hz; exact ⟨h1, h2 ▸ hy'.2⟩)
              (H M) (B M y ∩ B M y') (hHc M) fun ω ω' h hω =>
                ⟨hBc M y _ Finset.subset_union_left ω ω' h hω.1,
                  hBc M y' _ Finset.subset_union_right ω ω' h hω.2⟩,
              measure_inter_of_reads fam.O hd (B M y) (B M y') (hBc M y _ subset_rfl)
                (hBc M y' _ subset_rfl)]
          · simp [hBe M y' hy']
        · simp [hBe M y hy]
      · refine (Measure.restrict_apply_le _ _).trans ?_
        rw [← ofReal_measureReal (measure_ne_top _ _)]
        exact ENNReal.ofReal_le_ofReal (hpre _)
      · have hBw : ∀ y, μ' (B M y) ≤ ENNReal.ofReal e := fun y => by
          have hcard : (Ys M y).card ≤ 2 * (nQ + 1) := by
            refine (Finset.card_union_le _ _).trans ?_
            have h1 := Finset.card_image_le (s := M) (f := (y * ·))
            have h2 := Finset.card_image_le (s := M) (f := (y * FreeMonoid.of a * ·))
            omega
          calc μ' (B M y) ≤ μ' (⋃ z ∈ Ys M y, {ω | fam.read ω z = wrong z}) :=
                measure_mono fun ω ⟨_, z, hz, hr⟩ => Set.mem_biUnion hz hr
            _ ≤ ∑ z ∈ Ys M y, μ' {ω | fam.read ω z = wrong z} :=
                measure_biUnion_finset_le _ _
            _ ≤ ∑ _z ∈ Ys M y, ENNReal.ofReal w := Finset.sum_le_sum fun z _ => by
                rw [← ofReal_measureReal (measure_ne_top _ _)]
                exact ENNReal.ofReal_le_ofReal (hw _)
            _ = ENNReal.ofReal ((Ys M y).card * w) := by
                rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
                  ENNReal.ofReal_natCast]
            _ ≤ ENNReal.ofReal e := ENNReal.ofReal_le_ofReal (by
                rw [hed]
                have : ((Ys M y).card : ℝ) ≤ 2 * (nQ + 1) := by exact_mod_cast hcard
                exact mul_le_mul_of_nonneg_right this hw0)
        calc ∑' y, DE {y} * μ' (B M y) ≤ ∑' y, DE {y} * ENNReal.ofReal e :=
              ENNReal.tsum_le_tsum fun y => by gcongr; exact hBw y
          _ = ENNReal.ofReal (e * x) := by
              rw [ENNReal.tsum_mul_right]
              have := Measure.tsum_indicator_apply_singleton DE Set.univ MeasurableSet.univ
              simp only [Set.indicator_univ] at this
              rw [this, hDE, Measure.restrict_apply_univ, ← ofReal_measureReal (measure_ne_top _ _),
                ← ENNReal.ofReal_mul hx0, mul_comm]
      · gcongr
        rw [hcd, hu'd]
        exact cheb_half_le hp₀ he0 he2 hx0 (by linarith)
    have hdisj : Set.PairwiseDisjoint (↑Ms : Set (Finset (FreeMonoid α))) H := by
      intro M _ M' _ hne
      refine Set.disjoint_left.2 fun ω hω hω' => hne ?_
      exact hω.2.symm.trans hω'.2
    calc μ' {ω | (ω, xs) ∈ Ev}
        ≤ μ' (⋃ M ∈ Ms, H M ∩ {ω | u' < DE.real {y | ω ∈ B M y}}) := measure_mono hsub
      _ ≤ ∑ M ∈ Ms, μ' (H M ∩ {ω | u' < DE.real {y | ω ∈ B M y}}) :=
          measure_biUnion_finset_le _ _
      _ ≤ ∑ M ∈ Ms, μ' (H M) * ENNReal.ofReal c := Finset.sum_le_sum hM
      _ = μ' (⋃ M ∈ Ms, H M) * ENNReal.ofReal c := by
          rw [← Finset.sum_mul, measure_biUnion_finset hdisj fun M _ => hHm M]
      _ ≤ 1 * ENNReal.ofReal c := by gcongr; exact prob_le_one
      _ = ENNReal.ofReal c := one_mul _
  rw [Measure.prod_apply_symm hmeas]
  calc ∫⁻ xs, μ' ((fun ω => (ω, xs)) ⁻¹' Ev) ∂π
      ≤ ∫⁻ _, ENNReal.ofReal c ∂π := lintegral_mono fun xs => hfix xs
    _ = ENNReal.ofReal c := by simp

open Random.Kappa in
/-- Level 5's guarantee read through the family at `f = 0`, one spurious leaf tolerated within
level 3's `|Q| + 2`: level 3 fails to end well over `disagreeSet`, its tree leaves `classSet |Q|`,
or more than half of some `E q`, plus `u`, is misled, with chance at most
`δ + |Q||Σ| misledTail T p₀ (2(|Q| + 1)κ) u`. -/
theorem level3Fail_le (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (P : PrefixDistribution α) (C : Random.Level3Settings) (E : σ → Set (FreeMonoid α))
    (p₀ u t t' : ℝ) :
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    let m : ℝ := 16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1 + (nQ + 2) * nα * (nQ + 1)
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < C.δ → C.δ < 1 →
    0 < C.harvestRate → C.harvestRate ≤ C.disagreementRate / 48 → C.disagreementRate < 1 →
    36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget →
    0 ≤ t → 0 ≤ t' → t * fam.κ ≤ 1 - Real.exp (-(t' * fam.κ)) →
    C.probeBudget * (C.probeBudget - 1 : ℝ) / 2 * p₀
      + Real.exp (t' * (fam.κ * m)) / (1 + t) ^ 2 ≤ C.δ / 8 →
    (μ'.prod (Measure.pi fun _ : Fin C.probeBudget => P.D))
        (level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) E u)
      ≤ ENNReal.ofReal (C.δ + Fintype.card σ * Fintype.card α
        * misledTail C.probeBudget p₀ (2 * (Fintype.card σ + 1) * fam.κ) u) := by
  intro nQ nα S m hpre hδ hδ1 hy0 hyρ hρ1 hP ht ht' htc hfin
  classical
  set side : σ → Bool := fun q => readSide (fam.readDist q)
  set U : σ → ℝ := fun q => fam.readDist q .undecided
  obtain ⟨R, hR⟩ := familyKappaOracle_holds fam
  have hRr : ∀ z ω, R.read z ω = fam.read ω z := fun z ω => congrFun (congrFun hR z) ω
  set k := C.walkStart
  set T := C.probeBudget
  set π := Measure.pi fun _ : Fin T => P.D
  -- level 3 ends well within `|Q| + 2` leaves
  have hK := level5RoundCorrect_holds fam.M side U 0 fam.κ μ' R P C p₀ 0 t t' hpre hδ hδ1 hy0
    hyρ hρ1 hP le_rfl hκ ht ht' (by simp only [zero_mul, zero_add]; exact htc)
    (by simp only [zero_mul, zero_add]; exact hfin)
  have hp₀ : 0 < p₀ := (Random.prefixMax_pos P.D k).trans_le (ciSup_le hpre)
  set tail : ℝ := misledTail T p₀ (2 * (nQ + 1) * fam.κ) u with htail
  have htail0 : 0 ≤ tail := misledTail_nonneg hp₀.le (by positivity)
  -- the probes of each `E q` misled at each letter weigh at most half of it, plus `u`
  set Mk : σ → α → Set (Ω' × (Fin T → FreeMonoid α)) := fun q a => {d |
    let Tr := (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree
    Tr ∈ Random.classSet nQ ∧ P.D.real (E q) / 2 + u
      < P.D.real (E q ∩ misledSet fam.M fam.readDist (fun z => fam.read d.1 z) k Tr a)}
  have hMkle : ∀ q a, (μ'.prod π) (Mk q a) ≤ ENNReal.ofReal tail := fun q a =>
    misled_tail fam C (by positivity) (fun z => by
      have h1 := R.wrong z
      simp only [hRr] at h1
      have h2 : fam.κ * U (fam.M.eval z.toList) ≤ fam.κ :=
        mul_le_mul_of_nonneg_left (readDist_le_one fam _ _) hκ |>.trans (by rw [mul_one])
      exact h1.trans (by rw [zero_add]; exact h2)) P.D hp₀.le hpre (E q) a u
  -- level 5's event
  set E₁ := {d : Ω' × (Fin T → FreeMonoid α) |
    let result := Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2
    ¬ EndsWell
      { target := fam.M,
        undecided := U,
        harvestSlope := roundSlope σ P.L C,
        harvestSlack := roundSlack σ α C p₀,
        probes := P.D,
        disagreeRate := C.disagreementRate,
        disagree := disagreeSet (fun z => fam.read d.1 z) k result.hypothesis,
        walkStart := k,
        harvestLaw := Random.harvestLaw (fun z => fam.read d.1 z) P.D k result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending }
      ∨ nQ + 2 < result.hypothesis.tree.leaves.length}
  have hA0 : 0 ≤ roundSlope σ P.L C := by unfold roundSlope; positivity
  have hsl0 : 0 ≤ roundSlack σ α C p₀ := by
    unfold roundSlack
    refine div_nonneg (mul_nonneg hp₀.le (Real.log_nonneg ?_)) hy0.le
    rw [le_div_iff₀ hδ]
    have : (1 : ℝ) ≤ ((1 + (nQ + 1) ^ 3 * nα) ^ nQ * (nQ + 3) ^ ((nQ + 2) * nα) : ℕ) := by
      exact_mod_cast Nat.one_le_iff_ne_zero.2 (by positivity)
    have : (0 : ℝ) ≤ nQ := Nat.cast_nonneg _
    nlinarith
  have hE₁ : MeasurableSet E₁ := measurableSet_of_result fam C (fun ω result => ¬ EndsWell
    { target := fam.M,
      undecided := U,
      harvestSlope := roundSlope σ P.L C,
      harvestSlack := roundSlack σ α C p₀,
      probes := P.D,
      disagreeRate := C.disagreementRate,
      disagree := disagreeSet (fun z => fam.read ω z) k result.hypothesis,
      walkStart := k,
      harvestLaw := Random.harvestLaw (fun z => fam.read ω z) P.D k result.hypothesis,
      yield := C.harvestRate,
      ending := result.ending }
      ∨ nQ + 2 < result.hypothesis.tree.leaves.length) fun res =>
      (measurableSet_not_endsWell fam.O fam.F _ _ k fam.M _
        (fun q => fam.readDist_nonneg q _) hA0 hsl0 P.D _ _ _ _).union (MeasurableSet.const _)
  -- off the two, the round ends well
  have hsub : level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) E u
      ⊆ E₁ ∪ ⋃ q, ⋃ a, Mk q a := by
    rintro ⟨ω, xs⟩ hxs
    by_contra hno
    simp only [Set.mem_union, Set.mem_iUnion, not_or, not_exists] at hno
    obtain ⟨h1, hM⟩ := hno
    apply hxs
    set result := Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs
    have hcls : result.hypothesis.tree ∈ Random.classSet nQ :=
      Random.Spur.tree_classSet_of_leaves (fun z => fam.read ω z) C xs
        (not_lt.1 fun h => h1 (Or.inr h))
    refine ⟨by by_contra h; exact h1 (Or.inl h), hcls, fun q a => ?_⟩
    have hM' : ¬ (result.hypothesis.tree ∈ Random.classSet nQ ∧ P.D.real (E q) / 2 + u
        < P.D.real (E q ∩ misledSet fam.M fam.readDist (fun z => fam.read ω z) k
          result.hypothesis.tree a)) := hM q a
    exact not_lt.1 fun h => hM' ⟨hcls, h⟩
  have hE₁le : (μ'.prod π) E₁ ≤ ENNReal.ofReal C.δ := by
    rw [Measure.prod_apply hE₁]
    simp only [hRr] at hK
    exact hK
  calc (μ'.prod π) (level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) E u)
      ≤ (μ'.prod π) E₁ + (μ'.prod π) (⋃ q, ⋃ a, Mk q a) :=
        (measure_mono hsub).trans (measure_union_le _ _)
    _ ≤ ENNReal.ofReal C.δ + ∑ q, ∑ a, (μ'.prod π) (Mk q a) := by
        gcongr
        exact (measure_iUnion_fintype_le _ _).trans
          (Finset.sum_le_sum fun q _ => measure_iUnion_fintype_le _ _)
    _ ≤ ENNReal.ofReal C.δ + ∑ _q : σ, ∑ _a : α, ENNReal.ofReal tail := by
        gcongr with q _ a _
        exact hMkle q a
    _ = _ := by
        rw [Finset.sum_const, Finset.sum_const, Finset.card_univ, Finset.card_univ, nsmul_eq_mul,
          nsmul_eq_mul, ← ENNReal.ofReal_natCast, ← ENNReal.ofReal_natCast,
          ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_mul (by positivity),
          ← ENNReal.ofReal_add hδ.le (by positivity), mul_assoc]


end Fail

/-- A second copy of the noise fixed at a point where level 3's failure, over the first copy and
the probes, is still at most its bound over both. -/
theorem exists_noise_le {X : Type*} [MeasurableSpace X] (π : Measure X) [IsProbabilityMeasure π]
    (S : Set ((Ω × Ω) × X)) (hS : MeasurableSet S) :
    ∃ ω₀ : Ω, (μ.prod π) {q | ((q.1, ω₀), q.2) ∈ S} ≤ ((μ.prod μ).prod π) S := by
  set g : Ω → ENNReal := fun ω₀ => (μ.prod π) {q | ((q.1, ω₀), q.2) ∈ S}
  have hmeasS' : ∀ ω₀ : Ω, MeasurableSet {q : Ω × X | ((q.1, ω₀), q.2) ∈ S} := fun ω₀ =>
    ((measurable_fst.prodMk measurable_const).prodMk measurable_snd) hS
  -- the second copy, integrated out last
  set e : Ω × (Ω × X) → (Ω × Ω) × X := fun t => ((t.2.1, t.1), t.2.2)
  have he : MeasurePreserving e (μ.prod (μ.prod π)) ((μ.prod μ).prod π) := by
    have h1 : MeasurePreserving (fun t : Ω × (Ω × X) => ((t.1, t.2.1), t.2.2))
        (μ.prod (μ.prod π)) ((μ.prod μ).prod π) := (measurePreserving_prodAssoc μ μ π).symm
    have h2 : MeasurePreserving (fun t : (Ω × Ω) × X => ((t.1.2, t.1.1), t.2))
        ((μ.prod μ).prod π) ((μ.prod μ).prod π) :=
      ((Measure.measurePreserving_swap (μ := μ) (ν := μ)).prod (MeasurePreserving.id π))
    exact h2.comp h1
  have hint : ∫⁻ ω₀, g ω₀ ∂μ = ((μ.prod μ).prod π) S := by
    rw [← he.measure_preimage hS.nullMeasurableSet, Measure.prod_apply (he.measurable hS)]
    rfl
  have hgm : Measurable g := by
    exact measurable_measure_prodMk_left (ν := μ.prod π) (he.measurable hS)
  obtain ⟨ω₀, hω₀⟩ := exists_le_lintegral (μ := μ) hgm.aemeasurable
  exact ⟨ω₀, hω₀.trans hint.le⟩

end OrthoDFA
