import OrthoDFA.Proofs.FamilyOracle
import OrthoDFA.Proofs.Level5
import OrthoDFA.Proofs.LoopAvoid
import OrthoDFA.Proofs.Renoise

/-!
# Level 3 read through a re-noised family

`SuffixFamily.renoise` reads the family through `O.renoise R`: a suffix family on two copies of
the noise, with the same read distributions, which agrees with the family off `R` at the first
copy (`renoise_read_of_notMem`). Level 5's guarantee holds of it (`level3Fail_le`): level 3 fails
to end well within `|Q| + 2` leaves, or the probes reading past one of its tree's at most `|Q| + 1`
midfixes on their state's rarer side weigh more than `u`, with chance at most
`δ + (T p₀ + (|Q| + 1)κ)/u`. Level 3 reads only strings beginning with a probe's first `k`
letters (`level3_congr_zone'`), so a probe whose first `k` letters are no probe's reads
independently of the tree (`rarer_point_le`); the others weigh at most `T p₀`. A second copy can be
fixed at a point where the bound still holds (`exists_noise_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

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

/-- A string past its first `k` letters, which are no probe's, followed by anything, is read at
none of level 3's strings. -/
theorem disjoint_level3Zone (k : ℕ) {T : ℕ} (xs : Fin T → FreeMonoid α) (F : Finset (FreeMonoid α))
    {y : FreeMonoid α} (hy : k ≤ y.toList.length) (hkey : prefixOf y k ∉ probeKeys k xs)
    (m : FreeMonoid α) : Disjoint (level3Zone k xs F) (readStrings {y * m} F) := by
  classical
  refine Finset.disjoint_left.2 fun w hw hw' => hkey ?_
  obtain ⟨⟨z, v⟩, hzv, rfl⟩ := Finset.mem_image.1 hw
  obtain ⟨⟨z', v'⟩, hzv', he⟩ := Finset.mem_image.1 hw'
  obtain ⟨hz, hz'⟩ := Finset.mem_product.1 hzv
  obtain ⟨-, hzk, hkz⟩ := Finset.mem_filter.1 hz
  rw [Finset.mem_singleton.1 (Finset.mem_product.1 hzv').1] at he
  have e1 : prefixOf (y * m * v') k = prefixOf y k := by
    rw [mul_assoc]; exact prefixOf_mul_of_le hy _
  have e2 : prefixOf (z * v) k = prefixOf z k := prefixOf_mul_of_le hzk _
  dsimp only at he
  rw [← e1, he, e2]
  exact hkz

end Zone


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

theorem readDist_le_one (fam : SuffixFamily μ' α σ) (q : σ) (r : ARU) : fam.readDist q r ≤ 1 := by
  classical
  unfold SuffixFamily.readDist
  split_ifs
  · exact measureReal_le_one
  · norm_num

/-- Over given probes, a string `y` whose first `k` letters are no probe's reads, past a midfix of
level 3's tree, on its state's rarer side with chance at most `(|Q| + 1) w`, where each string does
so with chance at most `w`: level 3's reads and `y`'s are at disjoint strings. -/
theorem rarer_point_le (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings) {w : ℝ}
    (hw : ∀ z, μ'.real {ω | fam.read ω z
      = if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept} ≤ w)
    (xs : Fin C.probeBudget → FreeMonoid α) (y : FreeMonoid α)
    (hkey : prefixOf y C.walkStart ∉ probeKeys C.walkStart xs) :
    μ' {ω | let T := (Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs).hypothesis.tree
      T ∈ Random.classSet (Fintype.card σ)
        ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) C.walkStart T}
      ≤ ENNReal.ofReal ((Fintype.card σ + 1) * w) := by
  classical
  set k := C.walkStart
  set nQ := Fintype.card σ
  set Z := level3Zone k xs fam.F
  set tr : Ω' → DTree α := fun ω =>
    (Random.indecisiveLStarLevel3 C (fun z => fam.read ω z) xs).hypothesis.tree
  have htr : ∀ ω ω', (∀ p ∈ Z, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → tr ω = tr ω' :=
    fun ω ω' h => by
      exact congrArg (fun r => r.hypothesis.tree)
        (level3_congr_zone' fam.O C fam.F fam.kl fam.kh xs h)
  by_cases hk : k ≤ y.toList.length
  swap
  · exact (measure_mono (t := ∅) fun ω hω => (hk hω.2.1).elim).trans (by simp)
  set H : FreeMonoid α → Set Ω' := fun m => {ω | tr ω ∈ Random.classSet nQ ∧ m ∈ Random.mids (tr ω)}
  set wrong : FreeMonoid α → ARU := fun z =>
    if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept
  set B : FreeMonoid α → Set Ω' := fun m => {ω | fam.read ω (y * m) = wrong (y * m)}
  have hHc : ∀ m ω ω', (∀ p ∈ Z, (fam.O.mq p ω = 1 ↔ fam.O.mq p ω' = 1)) → ω ∈ H m → ω' ∈ H m :=
    fun m ω ω' h hω => by
      simp only [H, Set.mem_ofPred_eq, ← htr ω ω' h]
      exact hω
  have hHm : ∀ m, MeasurableSet (H m) := fun m => measurableSet_of_reads fam.O Z (H m) (hHc m)
  have hHB : ∀ m, μ' (H m ∩ B m) = μ' (H m) * μ' (B m) := fun m =>
    measure_inter_of_reads fam.O (disjoint_level3Zone k xs fam.F hk hkey m) (H m) (B m) (hHc m)
      fun ω ω' h hω => by
        have : fam.read ω (y * m) = fam.read ω' (y * m) :=
          familyRead_congr fam.O h (Finset.mem_singleton_self _)
        change fam.read ω' (y * m) = _
        rw [← this]
        exact hω
  have hB : ∀ m, μ' (B m) ≤ ENNReal.ofReal w := fun m => by
    rw [← ofReal_measureReal (measure_ne_top _ _)]
    exact ENNReal.ofReal_le_ofReal (hw _)
  have hsub : {ω | tr ω ∈ Random.classSet nQ
      ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) k (tr ω)}
      ⊆ ⋃ m ∈ shortStrings (α := α) nQ, H m ∩ B m := by
    rintro ω ⟨hc, -, m, hm, hr⟩
    exact Set.mem_biUnion (mem_shortStrings.2 ((Random.classSet_props _ _ hc).1 m hm))
      ⟨⟨hc, hm⟩, hr⟩
  have hsumH : ∑ m ∈ shortStrings (α := α) nQ, μ' (H m) ≤ ((nQ + 1 : ℕ) : ENNReal) := by
    calc ∑ m ∈ shortStrings (α := α) nQ, μ' (H m)
        = ∫⁻ ω, ∑ m ∈ shortStrings (α := α) nQ, (H m).indicator 1 ω ∂μ' := by
          rw [lintegral_finsetSum _ fun m _ => measurable_one.indicator (hHm m)]
          exact Finset.sum_congr rfl fun m _ => (lintegral_indicator_one (hHm m)).symm
      _ ≤ ∫⁻ _, ((nQ + 1 : ℕ) : ENNReal) ∂μ' := lintegral_mono fun ω => by
          have e : ∑ m ∈ shortStrings (α := α) nQ, (H m).indicator (1 : Ω' → ENNReal) ω
              = (((shortStrings (α := α) nQ).filter fun m => ω ∈ H m).card : ENNReal) := by
            simp only [Set.indicator_apply, Pi.one_apply]
            rw [Finset.sum_boole]
          rw [e, Nat.cast_le]
          by_cases hc : tr ω ∈ Random.classSet nQ
          · have h1 := Random.mids_card (tr ω)
            have h2 := (Random.classSet_props _ _ hc).2
            refine (Finset.card_le_card fun m hm => (Finset.mem_filter.1 hm).2.2).trans ?_
            omega
          · rw [Finset.filter_false_of_mem fun m _ hm => hc hm.1]
            simp
      _ = _ := by simp
  calc μ' {ω | tr ω ∈ Random.classSet nQ
        ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) k (tr ω)}
      ≤ ∑ m ∈ shortStrings (α := α) nQ, μ' (H m ∩ B m) :=
        (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
    _ ≤ ∑ m ∈ shortStrings (α := α) nQ, μ' (H m) * ENNReal.ofReal w :=
        Finset.sum_le_sum fun m _ => by rw [hHB]; gcongr; exact hB m
    _ = (∑ m ∈ shortStrings (α := α) nQ, μ' (H m)) * ENNReal.ofReal w :=
        (Finset.sum_mul _ _ _).symm
    _ ≤ ((nQ + 1 : ℕ) : ENNReal) * ENNReal.ofReal w := by gcongr
    _ = ENNReal.ofReal ((nQ + 1) * w) := by
        rw [ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_natCast]
        push_cast
        rfl

/-- Where level 3's tree is in `classSet |Q|`, the probes reading past one of its midfixes on their
state's rarer side weigh more than `u` with chance at most `(T p₀ + (|Q| + 1) w)/u`, where no
length-`k` prefix holds more than `p₀` of the probes and each string reads on its state's rarer side
with chance at most `w`. Those whose first `k` letters are a probe's weigh at most `T p₀`. -/
theorem rarer_markov (fam : SuffixFamily μ' α σ) (C : Random.Level3Settings) {w : ℝ} (hw0 : 0 ≤ w)
    (hw : ∀ z, μ'.real {ω | fam.read ω z
      = if readSide (fam.readDist (fam.M.eval z.toList)) then .reject else .accept} ≤ w)
    (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] {p₀ : ℝ} (hp₀ : 0 ≤ p₀)
    (hpre : ∀ v, D.real {x | prefixOf x C.walkStart = v} ≤ p₀) {u : ℝ} (hu : 0 < u) :
    (μ'.prod (Measure.pi fun _ : Fin C.probeBudget => D)) {d |
      let T := (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree
      u < D.real {y | T ∈ Random.classSet (Fintype.card σ)
        ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read d.1 z) C.walkStart T}}
      ≤ ENNReal.ofReal ((C.probeBudget * p₀ + (Fintype.card σ + 1) * w) / u) := by
  classical
  set k := C.walkStart
  set nQ := Fintype.card σ
  set π := Measure.pi fun _ : Fin C.probeBudget => D
  set S : Ω' × (Fin C.probeBudget → FreeMonoid α) → Set (FreeMonoid α) := fun d =>
    {y | (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree
        ∈ Random.classSet nQ
      ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read d.1 z) k
        (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree}
  have hSy : ∀ y, MeasurableSet {d | y ∈ S d} := fun y =>
    measurableSet_of_result fam C (fun ω res => res.hypothesis.tree ∈ Random.classSet nQ
      ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) k res.hypothesis.tree)
      fun res => by
        have e : {ω | res.hypothesis.tree ∈ Random.classSet nQ
            ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) k res.hypothesis.tree}
            = {_ω | res.hypothesis.tree ∈ Random.classSet nQ} ∩ ({_ω | k ≤ y.toList.length}
              ∩ ⋃ m ∈ Random.mids res.hypothesis.tree, {ω | fam.read ω (y * m)
                = if readSide (fam.readDist (fam.M.eval (y * m).toList)) then .reject
                  else .accept}) := by
          ext ω
          simp [rarerSet]
        rw [e]
        exact (MeasurableSet.const _).inter ((MeasurableSet.const _).inter
          (Finset.measurableSet_biUnion _ fun m _ =>
            fam.read_measurable _ (measurableSet_singleton _)))
  have hf : Measurable fun d => D (S d) := measurable_measure_of_mem D _ hSy
  -- the probes sharing their first `k` letters with a probe
  have hcoll : ∀ xs : Fin C.probeBudget → FreeMonoid α,
      D {y | prefixOf y k ∈ probeKeys k xs} ≤ ENNReal.ofReal (C.probeBudget * p₀) := by
    intro xs
    have e : {y | prefixOf y k ∈ probeKeys k xs} = ⋃ i, {y | prefixOf y k = prefixOf (xs i) k} := by
      ext y; simp [probeKeys, eq_comm]
    rw [e]
    refine (measure_iUnion_fintype_le _ _).trans ?_
    calc ∑ i, D {y | prefixOf y k = prefixOf (xs i) k}
        ≤ ∑ _i : Fin C.probeBudget, ENNReal.ofReal p₀ := Finset.sum_le_sum fun i _ => by
          rw [← ofReal_measureReal (measure_ne_top _ _)]
          exact ENNReal.ofReal_le_ofReal (hpre _)
      _ = ENNReal.ofReal (C.probeBudget * p₀) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
            ENNReal.ofReal_mul (Nat.cast_nonneg _), ENNReal.ofReal_natCast]
  set S' : Ω' × (Fin C.probeBudget → FreeMonoid α) → Set (FreeMonoid α) := fun d =>
    {y | y ∈ S d ∧ prefixOf y k ∉ probeKeys k d.2}
  have hpt : ∀ d, D (S d) ≤ ENNReal.ofReal (C.probeBudget * p₀) + D (S' d) := fun d =>
    calc D (S d) ≤ D ({y | prefixOf y k ∈ probeKeys k d.2} ∪ S' d) := measure_mono fun y hy => by
          by_cases h : prefixOf y k ∈ probeKeys k d.2
          · exact Or.inl h
          · exact Or.inr ⟨hy, h⟩
      _ ≤ _ := (measure_union_le _ _).trans (add_le_add (hcoll d.2) le_rfl)
  -- the rest, by independence
  have hS' : ∀ y, (μ'.prod π) {d | y ∈ S' d} ≤ ENNReal.ofReal ((nQ + 1) * w) := by
    intro y
    have hm : MeasurableSet {d : Ω' × (Fin C.probeBudget → FreeMonoid α) | y ∈ S' d} :=
      (hSy y).inter (measurable_snd (Set.to_countable
        {xs : Fin C.probeBudget → FreeMonoid α | prefixOf y k ∉ probeKeys k xs}).measurableSet)
    rw [Measure.prod_apply_symm hm]
    calc ∫⁻ xs, μ' ((fun ω => (ω, xs)) ⁻¹' {d | y ∈ S' d}) ∂π
        ≤ ∫⁻ _, ENNReal.ofReal ((nQ + 1) * w) ∂π := lintegral_mono fun xs => by
          by_cases hkey : prefixOf y k ∈ probeKeys k xs
          · exact (measure_mono (t := ∅) fun ω hω => (hω.2 hkey).elim).trans (by simp)
          · exact (measure_mono fun ω hω => hω.1).trans (rarer_point_le fam C hw xs y hkey)
      _ = _ := by simp
  have hint : ∫⁻ d, D (S d) ∂(μ'.prod π)
      ≤ ENNReal.ofReal (C.probeBudget * p₀) + ENNReal.ofReal ((nQ + 1) * w) :=
    calc ∫⁻ d, D (S d) ∂(μ'.prod π)
        ≤ ∫⁻ d, (ENNReal.ofReal (C.probeBudget * p₀) + D (S' d)) ∂(μ'.prod π) :=
          lintegral_mono hpt
      _ = ENNReal.ofReal (C.probeBudget * p₀) + ∫⁻ d, D (S' d) ∂(μ'.prod π) := by
          rw [lintegral_add_left measurable_const, lintegral_const, measure_univ, mul_one]
      _ ≤ _ := add_le_add le_rfl (Random.Eps.lintegral_le_of_points _ D S' _ fun y _ => hS' y)
  have hsub : {d : Ω' × (Fin C.probeBudget → FreeMonoid α) | u < D.real (S d)}
      ⊆ {d | ENNReal.ofReal u ≤ D (S d)} := by
    intro d hd
    simp only [Set.mem_ofPred_eq] at hd ⊢
    rw [measureReal_def] at hd
    exact (ENNReal.ofReal_le_iff_le_toReal (measure_ne_top _ _)).2 hd.le
  refine (measure_mono hsub).trans ((meas_ge_le_lintegral_div hf.aemeasurable
    (by simpa using hu) ENNReal.ofReal_ne_top).trans ?_)
  calc (∫⁻ d, D (S d) ∂(μ'.prod π)) / ENNReal.ofReal u
      ≤ (ENNReal.ofReal (C.probeBudget * p₀) + ENNReal.ofReal ((nQ + 1) * w))
        / ENNReal.ofReal u := by gcongr
    _ = _ := by
        rw [← ENNReal.ofReal_add (by positivity) (by positivity), ENNReal.ofReal_div_of_pos hu]


open Random.Kappa in
/-- Level 5's guarantee read through the family at `f = 0`, one spurious leaf tolerated within
level 3's `|Q| + 2`: level 3 fails to end well at `ε + u` over `disagreeSet`, or its tree leaves
`classSet |Q|`, but for chance `δ + (T p₀ + (|Q| + 1)κ)/u`. -/
theorem level3Fail_le (fam : SuffixFamily μ' α σ) (hκ : 0 ≤ fam.κ)
    (P : PrefixDistribution α) (C : Random.Level3Settings) (p₀ u t t' : ℝ) (hu : 0 < u) :
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
        (level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) u)
      ≤ ENNReal.ofReal (C.δ
        + (C.probeBudget * p₀ + (Fintype.card σ + 1) * fam.κ) / u) := by
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
  -- the probes reading on the rarer side past a midfix of the tree weigh at most `u`
  set Mk := {d : Ω' × (Fin T → FreeMonoid α) |
    let Tr := (Random.indecisiveLStarLevel3 C (fun z => fam.read d.1 z) d.2).hypothesis.tree
    u < P.D.real {y | Tr ∈ Random.classSet nQ
      ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read d.1 z) k Tr}}
  have hMkle : (μ'.prod π) Mk ≤ ENNReal.ofReal ((T * p₀ + (nQ + 1) * fam.κ) / u) :=
    rarer_markov fam C (by positivity) (fun z => by
      have h1 := R.wrong z
      simp only [hRr] at h1
      have h2 : fam.κ * U (fam.M.eval z.toList) ≤ fam.κ :=
        mul_le_mul_of_nonneg_left (readDist_le_one fam _ _) hκ |>.trans (by rw [mul_one])
      exact h1.trans (by rw [zero_add]; exact h2)) P.D hp₀.le hpre hu
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
        disagree := Random.searchSet (fun z => fam.read d.1 z) k result.hypothesis
          ∪ Random.endsSet (fun z => fam.read d.1 z) k result.hypothesis,
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
      disagree := Random.searchSet (fun z => fam.read ω z) k result.hypothesis
        ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis,
      walkStart := k,
      harvestLaw := Random.harvestLaw (fun z => fam.read ω z) P.D k result.hypothesis,
      yield := C.harvestRate,
      ending := result.ending }
      ∨ nQ + 2 < result.hypothesis.tree.leaves.length) fun res =>
      (measurableSet_not_endsWell_searchEnds fam.O fam.F _ _ k fam.M _
        (fun q => fam.readDist_nonneg q _) hA0 hsl0 P.D _ _ _ _).union (MeasurableSet.const _)
  -- off the two, the round ends well
  have hsub : level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) u ⊆ E₁ ∪ Mk := by
    rintro ⟨ω, xs⟩ hxs
    by_contra hno
    simp only [Set.mem_union, not_or] at hno
    obtain ⟨h1, hM⟩ := hno
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
      by_contra h; exact h1 (Or.inl h)
    have hcls : result.hypothesis.tree ∈ Random.classSet nQ :=
      Random.Spur.tree_classSet_of_leaves (fun z => fam.read ω z) C xs
        (not_lt.1 fun h => h1 (Or.inr h))
    refine ⟨?_, hcls⟩
    rcases he : result.ending with _ | site | _
    · rw [he] at hEW
      show P.D.real (disagreeSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis)
        ≤ C.disagreementRate + u
      have hMr : P.D.real (rarerSet fam.M fam.readDist (fun z => fam.read ω z) k
          result.hypothesis.tree) ≤ u := by
        have hM' : ¬ u < P.D.real {y | result.hypothesis.tree ∈ Random.classSet nQ
            ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis.tree} :=
          hM
        refine le_trans (measureReal_mono (s₂ := {y | result.hypothesis.tree ∈ Random.classSet nQ
          ∧ y ∈ rarerSet fam.M fam.readDist (fun z => fam.read ω z) k result.hypothesis.tree})
          (fun y hy => ⟨hcls, hy⟩) (measure_ne_top _ _)) (not_lt.1 hM')
      calc _ ≤ P.D.real (Random.searchSet (fun z => fam.read ω z) k result.hypothesis
              ∪ Random.endsSet (fun z => fam.read ω z) k result.hypothesis)
            + P.D.real (rarerSet fam.M fam.readDist (fun z => fam.read ω z) k
              result.hypothesis.tree) := measureReal_union_le _ _
        _ ≤ C.disagreementRate + u := add_le_add hEW hMr
    · rw [he] at hEW; exact hEW
    · rw [he] at hEW; exact hEW
  have hE₁le : (μ'.prod π) E₁ ≤ ENNReal.ofReal C.δ := by
    rw [Measure.prod_apply hE₁]
    simp only [hRr] at hK
    exact hK
  calc (μ'.prod π) (level3Fail fam C P.D (roundSlope σ P.L C) (roundSlack σ α C p₀) u)
      ≤ (μ'.prod π) E₁ + (μ'.prod π) Mk := (measure_mono hsub).trans (measure_union_le _ _)
    _ ≤ ENNReal.ofReal C.δ + ENNReal.ofReal ((T * p₀ + (nQ + 1) * fam.κ) / u) :=
        add_le_add hE₁le hMkle
    _ = _ := by rw [← ENNReal.ofReal_add hδ.le (by positivity)]


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
