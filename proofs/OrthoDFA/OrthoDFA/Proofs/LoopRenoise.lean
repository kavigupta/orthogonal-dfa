import OrthoDFA.Proofs.FamilyOracle
import OrthoDFA.Proofs.LoopAvoid
import OrthoDFA.Proofs.Renoise

/-!
# Level 3 read through a re-noised family

`SuffixFamily.renoise` reads the family through `O.renoise R`: a suffix family on two copies of
the noise, with the same read distributions, which agrees with the family off `R` at the first
copy (`renoise_read_of_notMem`). Level 5's guarantee holds of it (`level3Fail_le`): level 3 fails
to end well, a step reads some string wrong, or the probes some string of which, followed by at
most `|Q|` letters, reads wrong weigh more than `u`, with chance at most
`δ + δ/8 + #(strings of at most |Q| letters) (ε + κ)/u`. A second copy can be fixed at a point
where the bound still holds (`exists_noise_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {Ω : Type*} [MeasurableSpace Ω]
  {μ : Measure Ω} [IsProbabilityMeasure μ] {σ : Type*}

/-- The family read through `O.renoise R`. -/
def SuffixFamily.renoise (fam : SuffixFamily μ α σ) (R : Set (FreeMonoid α)) :
    SuffixFamily (μ.prod μ) α σ :=
  ⟨fam.O.renoise R, fam.M, fam.F, fam.kl, fam.kh, fam.ε, fam.ε₂, fam.κ, fam.lang, fam.suffixFree,
    fam.band⟩

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

section Fail

variable {Ω' : Type*} [MeasurableSpace Ω'] {μ' : Measure Ω'} [IsProbabilityMeasure μ']

/-- The probes some string of which, followed by at most `n` letters, reads `bad`. -/
def wrongSet (read : FreeMonoid α → ARU) (bad : FreeMonoid α → ARU) (n : ℕ) :
    Set (FreeMonoid α) :=
  {y | ∃ m ∈ shortStrings n, read (y * m) = bad (y * m)}

theorem measurable_wrongSet (read : FreeMonoid α → Ω' → ARU) (hmeas : ∀ z, Measurable (read z))
    (bad : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (n : ℕ) :
    Measurable fun ω => D (wrongSet (read · ω) bad n) := by
  classical
  refine measurable_measure_of_mem D _ fun y => ?_
  have : {ω | y ∈ wrongSet (read · ω) bad n}
      = ⋃ m ∈ shortStrings (α := α) n, {ω | read (y * m) ω = bad (y * m)} := by
    ext ω; simp [wrongSet]
  rw [this]
  exact Finset.measurableSet_biUnion _ fun m _ => hmeas _ (measurableSet_singleton (bad (y * m)))

/-- Where each string reads `bad` with chance at most `w`, the probes `wrongSet` holds weigh more
than `u` with chance at most `#(strings of at most n letters) w / u`. -/
theorem wrongSet_markov (read : FreeMonoid α → Ω' → ARU) (hmeas : ∀ z, Measurable (read z))
    (bad : FreeMonoid α → ARU) {w : ℝ} (hw0 : 0 ≤ w)
    (hw : ∀ z, μ'.real {ω | read z ω = bad z} ≤ w) (D : Measure (FreeMonoid α))
    [IsProbabilityMeasure D] (n : ℕ) {u : ℝ} (hu : 0 < u) :
    μ' {ω | u < D.real (wrongSet (read · ω) bad n)}
      ≤ ENNReal.ofReal ((shortStrings (α := α) n).card * w / u) := by
  classical
  set f : Ω' → ENNReal := fun ω => D (wrongSet (read · ω) bad n)
  have hmem : ∀ y, MeasurableSet {ω | y ∈ wrongSet (read · ω) bad n} := by
    intro y
    have : {ω | y ∈ wrongSet (read · ω) bad n}
        = ⋃ m ∈ shortStrings (α := α) n, {ω | read (y * m) ω = bad (y * m)} := by
      ext ω; simp [wrongSet]
    rw [this]
    exact Finset.measurableSet_biUnion _ fun m _ =>
      hmeas _ (measurableSet_singleton (bad (y * m)))
  have hf : Measurable f := measurable_measure_of_mem D _ hmem
  have hsub : {ω | u < D.real (wrongSet (read · ω) bad n)} ⊆ {ω | ENNReal.ofReal u ≤ f ω} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    rw [measureReal_def] at hω
    exact (ENNReal.ofReal_le_iff_le_toReal (measure_ne_top _ _)).2 hω.le
  have hpt : ∀ y, μ' {ω | y ∈ wrongSet (read · ω) bad n}
      ≤ ENNReal.ofReal ((shortStrings (α := α) n).card * w) := by
    intro y
    have : {ω | y ∈ wrongSet (read · ω) bad n}
        = ⋃ m ∈ shortStrings (α := α) n, {ω | read (y * m) ω = bad (y * m)} := by
      ext ω; simp [wrongSet]
    rw [this]
    refine (measure_biUnion_finset_le _ _).trans ?_
    calc ∑ m ∈ shortStrings (α := α) n, μ' {ω | read (y * m) ω = bad (y * m)}
        ≤ ∑ _m ∈ shortStrings (α := α) n, ENNReal.ofReal w := Finset.sum_le_sum fun m _ => by
          rw [← ofReal_measureReal (measure_ne_top _ _)]
          exact ENNReal.ofReal_le_ofReal (hw _)
      _ = ENNReal.ofReal ((shortStrings (α := α) n).card * w) := by
          rw [Finset.sum_const, nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg _),
            ENNReal.ofReal_natCast]
  have hint : ∫⁻ ω, f ω ∂μ' ≤ ENNReal.ofReal ((shortStrings (α := α) n).card * w) :=
    Random.Eps.lintegral_le_of_points μ' D _ _ fun y _ => hpt y
  refine (measure_mono hsub).trans ((meas_ge_le_lintegral_div hf.aemeasurable
    (by simpa using hu) ENNReal.ofReal_ne_top).trans ?_)
  rw [ENNReal.ofReal_div_of_pos hu]
  gcongr

variable [Fintype σ]

/-- Level 5's harvest slope at probes at most `L` long. -/
def roundSlope (σ : Type*) [Fintype σ] (L : ℕ) (C : Random.Level3Settings) : ℝ :=
  (Fintype.card σ + 1) * (6 / C.harvestRate + 16 * (2 + Nat.clog 2 L))

/-- Level 5's harvest slack at the prefix mass `p₀`. -/
def roundSlack (σ α : Type*) [Fintype σ] [Fintype α] (C : Random.Level3Settings) (p₀ : ℝ) : ℝ :=
  p₀ * Real.log (16 * (Fintype.card σ + 1) * ((1 + (Fintype.card σ + 1) ^ 3 * Fintype.card α)
    ^ Fintype.card σ * (Fintype.card σ + 3) ^ ((Fintype.card σ + 2) * Fintype.card α) : ℕ)
      / C.δ) / C.harvestRate

/-- Where level 3, reading `fam` over probes from `D'`, fails to end well at `ε + u` over
`disagreeSet`, or its tree leaves `classSet |Q|`. -/
def level3Fail (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (D' : Measure (FreeMonoid α)) (A sl u : ℝ) : Set (Ω' × (Fin C.probeBudget → FreeMonoid α)) :=
  {d | let result := Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2
    ¬ (EndsWell
      { target := fam.M,
        undecided := fun q => fam.readDist q .undecided,
        harvestSlope := A,
        harvestSlack := sl,
        probes := D',
        disagreeRate := C.disagreementRate + u,
        disagree := disagreeSet fam.M fam.readDist (fun z => fam.read d.1 z) C.walkStart
          result.hypothesis,
        walkStart := C.walkStart,
        harvestLaw := Random.harvestLaw (fun z => fam.read d.1 z) D' C.walkStart
          result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending }
      ∧ result.hypothesis.tree ∈ Random.classSet (Fintype.card σ))}

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

theorem measurableSet_level3Fail (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (D' : Measure (FreeMonoid α)) {A sl : ℝ} (hA : 0 ≤ A) (hsl : 0 ≤ sl) (u : ℝ) :
    MeasurableSet (level3Fail fam C D' A sl u) := by
  classical
  refine measurableSet_of_result fam C (fun ω result => ¬ (EndsWell
    { target := fam.M,
      undecided := fun q => fam.readDist q .undecided,
      harvestSlope := A,
      harvestSlack := sl,
      probes := D',
      disagreeRate := C.disagreementRate + u,
      disagree := disagreeSet fam.M fam.readDist (fun z => fam.read ω z) C.walkStart
        result.hypothesis,
      walkStart := C.walkStart,
      harvestLaw := Random.harvestLaw (fun z => fam.read ω z) D' C.walkStart result.hypothesis,
      yield := C.harvestRate,
      ending := result.ending } ∧ result.hypothesis.tree ∈ Random.classSet (Fintype.card σ)))
    fun res => ?_
  simp only [not_and_or]
  exact (measurableSet_not_endsWell fam.O fam.F _ _ C.walkStart fam.M _
    (fun q => fam.readDist_nonneg q _) hA hsl _ D' _ _ _ _).union
    (MeasurableSet.const _)

/-- Whether some step reads a string wrong is measurable. -/
theorem measurableSet_anyB (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings)
    (side : σ → Bool) :
    MeasurableSet {d : Ω' × (Fin C.probeBudget → FreeMonoid α) |
      Random.AnyB (Random.applyProbe (fun z => fam.read d.1 z) C)
        (Random.Kappa.ReadsWrong fam.M side (fun z => fam.read d.1 z) C) Random.initialState
        (List.ofFn d.2)} := by
  classical
  refine measurableSet_prod_countable_right _ fun xs => ?_
  set A := readStrings (shortStrings (probesLen xs + 1 + C.probeBudget + 2)) fam.F
  refine measurableSet_of_reads fam.O A _ fun ω ω' h hω => ?_
  simp only [Set.mem_setOf_eq] at hω ⊢
  refine Random.anyB_congr_zone C fam.M side (X := List.ofFn xs) (B := C.probeBudget)
    (fun x hx j _ hj m hm => ?_) (List.ofFn xs) Random.initialState (fun _ h => h)
    (Random.runInv_start _ C _) (by simp [Random.initialState, Random.newHypothesis,
      DTree.leaves]) hω
  obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
  refine familyRead_congr fam.O h (mem_shortStrings.2 ?_)
  have h1 := le_probesLen xs i
  have h2 : (prefixOf (xs i) j).toList.length ≤ (xs i).toList.length := by simp [prefixOf]
  simp only [FreeMonoid.toList_mul, List.length_append]
  omega

theorem readDist_le_one (fam : SuffixFamily μ' α σ) (q : σ) (r : ARU) : fam.readDist q r ≤ 1 := by
  classical
  unfold SuffixFamily.readDist
  split_ifs
  · exact measureReal_le_one
  · norm_num

open Random.Kappa in
/-- Level 5's guarantee read through the family: level 3 fails to end well at `ε + u` over
`disagreeSet`, or its tree leaves `classSet |Q|`, but for chance
`δ + δ/8 + #(strings of at most |Q| letters) (ε + κ)/u`. -/
theorem level3Fail_le (fam : SuffixFamily μ' α σ) (hε : 0 ≤ fam.ε) (hκ : 0 ≤ fam.κ)
    (P : PrefixDistribution α) (C : Random.Level3Settings) (p₀ u : ℝ) (hu : 0 < u) :
    let nQ := Fintype.card σ
    let nα := Fintype.card α
    let S : ℕ := nQ * ((nQ + 2) * nα + 1) + 2 * nα + 1
    (∀ u, P.D.real {x | prefixOf x C.walkStart = u} ≤ p₀) → 0 < C.δ → C.δ ≤ 1 →
    0 < C.harvestRate → C.harvestRate ≤ C.disagreementRate / 48 → C.disagreementRate < 1 →
    36 * S / C.disagreementRate * Real.log (16 * S * (S + 1) / C.δ) + S ≤ C.probeBudget →
    fam.ε * (C.probeBudget * min (P.L + 1) (3 + Nat.clog 2 P.L) * nQ + nQ * (nQ + 1) ^ 2 * nα)
      + fam.κ * (16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1 + nQ * (nQ + 1) ^ 2 * nα)
        ≤ C.δ / 8 →
    (μ'.prod (Measure.pi fun _ : Fin C.probeBudget => P.D))
        (level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) u)
      ≤ ENNReal.ofReal (C.δ + C.δ / 8
        + (shortStrings (α := α) (Fintype.card σ)).card * (fam.ε + fam.κ) / u) := by
  intro nQ nα S hpre hδ hδ1 hy0 hyρ hρ1 hP hw
  classical
  set side : σ → Bool := fun q => readSide (fam.readDist q)
  set U : σ → ℝ := fun q => fam.readDist q .undecided
  obtain ⟨R, hR⟩ := familyKappaOracle_holds fam hε hκ
  have hRr : ∀ z ω, R.read z ω = fam.read ω z := fun z ω => congrFun (congrFun hR z) ω
  set k := C.walkStart
  set T := C.probeBudget
  set π := Measure.pi fun _ : Fin T => P.D
  set wrong : FreeMonoid α → ARU := fun z => if side (fam.M.eval z.toList) then .reject else .accept
  -- level 3 ends well
  have hK := Random.Kappa.level5RoundCorrect_holds fam.M side U fam.ε fam.κ μ' R P C p₀ hpre
    hδ hy0 hyρ hρ1 hP hε hκ hw
  -- no step reads wrong
  have hp₀ : 0 < p₀ := (Random.prefixMax_pos P.D k).trans_le (ciSup_le hpre)
  have hW := pi_wrong_le' fam.M side C μ' U fam.ε fam.κ R P hε hκ hy0 hyρ hρ1 hδ hδ1
  have hSst : Level2.stretches nQ nα = S := rfl
  have hcost := sum_costAt_le C S hδ hδ1
  have hMu : Mu nQ nα C ≤ 16 / 5 * S * Real.log (8 * S * (S + 1) / C.δ) + 1
      + nQ * (nQ + 1) ^ 2 * nα := by
    unfold Mu wSplit
    rw [hSst]
    push_cast
    nlinarith
  have hMb : Mb nQ nα P.L C.probeBudget
      = C.probeBudget * ((min (P.L + 1) (3 + Nat.clog 2 P.L) : ℕ) : ℝ) * nQ
        + nQ * (nQ + 1) ^ 2 * nα := by
    unfold Mb wSplit probeSifts
    push_cast
    ring
  have hwr : fam.ε * Mb nQ nα P.L C.probeBudget + fam.κ * Mu nQ nα C ≤ C.δ / 8 := by
    rw [hMb]
    have := mul_le_mul_of_nonneg_left hMu hκ
    linarith
  have hW' := hW.trans (ENNReal.ofReal_le_ofReal hwr)
  -- the probes reading wrong past a short midfix weigh at most `u`
  set Mk := {ω | u < P.D.real (wrongSet (fun z => fam.read ω z) wrong nQ)}
  have hMk : MeasurableSet Mk :=
    measurableSet_lt measurable_const (ENNReal.measurable_toReal.comp
      (measurable_wrongSet (fun z ω => fam.read ω z) fam.read_measurable wrong P.D nQ))
  have hMkle : μ' Mk ≤ ENNReal.ofReal ((shortStrings (α := α) nQ).card * (fam.ε + fam.κ) / u) :=
    wrongSet_markov (fun z ω => fam.read ω z) fam.read_measurable wrong (by positivity) (fun z => by
      have h1 := R.wrong z
      simp only [hRr] at h1
      have h2 : fam.κ * U (fam.M.eval z.toList) ≤ fam.κ :=
        mul_le_of_le_one_right hκ (readDist_le_one fam _ _)
      exact h1.trans (by linarith)) P.D nQ hu
  -- the three events
  set E₁ := {d : Ω' × (Fin T → FreeMonoid α) |
    let result := Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2
    ¬ EndsWell
      { target := fam.M,
        undecided := U,
        harvestSlope := roundSlope σ P.L C,
        harvestSlack := roundSlack σ α C p₀,
        probes := P.D,
        disagreeRate := C.disagreementRate,
        disagree := Random.searchSet (fun z => fam.read d.1 z) k result.hypothesis
          ∪ Random.endsSet (fun z => fam.read d.1 z) k result.hypothesis,
        walkStart := k,
        harvestLaw := Random.harvestLaw (fun z => fam.read d.1 z) P.D k result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending }}
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
      disagree := Random.searchSet (fun z => fam.read ω z) k result.hypothesis
        ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis,
      walkStart := k,
      harvestLaw := Random.harvestLaw (fun z => fam.read ω z) P.D k result.hypothesis,
      yield := C.harvestRate,
      ending := result.ending }) fun res =>
      measurableSet_not_endsWell_searchEnds fam.O fam.F _ _ k fam.M _
        (fun q => fam.readDist_nonneg q _) hA0 hsl0 P.D _ _ _ _
  set E₂ := {d : Ω' × (Fin T → FreeMonoid α) |
    Random.AnyB (Random.applyProbe (fun z => fam.read d.1 z) C)
      (Random.Kappa.ReadsWrong fam.M side (fun z => fam.read d.1 z) C) Random.initialState
      (List.ofFn d.2)}
  have hE₂ : MeasurableSet E₂ := measurableSet_anyB fam C side
  -- off the three, the round ends well
  have hsub : ∀ ω, Prod.mk ω ⁻¹' level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) u
      ⊆ Prod.mk ω ⁻¹' E₁ ∪ Prod.mk ω ⁻¹' E₂ ∪ (if ω ∈ Mk then Set.univ else ∅) := by
    intro ω xs hxs
    by_contra hno
    simp only [Set.mem_union, Set.mem_preimage, not_or] at hno
    obtain ⟨⟨h1, h2⟩, h3⟩ := hno
    have hM : ω ∉ Mk := fun h => h3 (by rw [if_pos h]; trivial)
    apply hxs
    set result := Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs
    have hEW : EndsWell
      { target := fam.M,
        undecided := U,
        harvestSlope := roundSlope σ P.L C,
        harvestSlack := roundSlack σ α C p₀,
        probes := P.D,
        disagreeRate := C.disagreementRate,
        disagree := Random.searchSet (fun z => fam.read ω z) k result.hypothesis
          ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis,
        walkStart := k,
        harvestLaw := Random.harvestLaw (fun z => fam.read ω z) P.D k result.hypothesis,
        yield := C.harvestRate,
        ending := result.ending } := by
      by_contra h; exact h1 h
    have hcls : result.hypothesis.tree ∈ Random.classSet nQ :=
      Random.tree_classSet_of_good fam.M side (fun z => fam.read ω z) C xs h2
    refine ⟨?_, hcls⟩
    rcases he : result.ending with _ | site | _
    · rw [he] at hEW
      show P.D.real (disagreeSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis)
        ≤ C.disagreementRate + u
      have hr : rarerSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis.tree
          ⊆ wrongSet (fun z => fam.read ω z) wrong nQ := by
        rintro y ⟨-, m, hm, hw⟩
        exact ⟨m, mem_shortStrings.2 ((Random.classSet_props _ _ hcls).1 m hm), by exact hw⟩
      have hMr : P.D.real (wrongSet (fun z => fam.read ω z) wrong nQ) ≤ u := by
        by_contra h; exact hM (lt_of_not_ge h)
      calc _ ≤ P.D.real (Random.searchSet (fun z => fam.read ω z) k result.hypothesis
              ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis)
            + P.D.real (rarerSet fam.M fam.readDist (fun z => fam.read ω z) k
              result.hypothesis.tree) := measureReal_union_le _ _
        _ ≤ C.disagreementRate + u := add_le_add hEW
            ((measureReal_mono hr (measure_ne_top _ _)).trans hMr)
    · rw [he] at hEW; exact hEW
    · rw [he] at hEW; exact hEW
  -- the integral
  rw [Measure.prod_apply (measurableSet_level3Fail fam C P.D hA0 hsl0 u)]
  calc ∫⁻ ω, π (Prod.mk ω ⁻¹' level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) u)
        ∂μ'
      ≤ ∫⁻ ω, (π (Prod.mk ω ⁻¹' E₁) + π (Prod.mk ω ⁻¹' E₂) + Mk.indicator 1 ω) ∂μ' := by
        refine lintegral_mono fun ω => (measure_mono (hsub ω)).trans
          ((measure_union_le _ _).trans (add_le_add (measure_union_le _ _) ?_))
        by_cases h : ω ∈ Mk
        · rw [if_pos h, Set.indicator_of_mem h]; exact prob_le_one
        · rw [if_neg h, Set.indicator_of_notMem h]; simp
    _ = ∫⁻ ω, π (Prod.mk ω ⁻¹' E₁) ∂μ' + ∫⁻ ω, π (Prod.mk ω ⁻¹' E₂) ∂μ' + μ' Mk := by
        rw [lintegral_add_right _ (measurable_one.indicator hMk),
          lintegral_add_left (measurable_measure_prodMk_left hE₁),
          lintegral_indicator_one hMk]
    _ ≤ ENNReal.ofReal C.δ + ENNReal.ofReal (C.δ / 8)
          + ENNReal.ofReal ((shortStrings (α := α) nQ).card * (fam.ε + fam.κ) / u) := by
        simp only [hRr] at hK hW'
        gcongr
        · exact hK
        · exact hW'
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity),
          ← ENNReal.ofReal_add (by positivity) (by positivity)]

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
