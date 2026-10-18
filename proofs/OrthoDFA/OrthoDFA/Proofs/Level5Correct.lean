import OrthoDFA.Proofs.Level5RiskBound
import OrthoDFA.Proofs.Level5Sift
import OrthoDFA.Level5Family

/-!
# Level 5's claim

`Level5Correct` from `Level5RoundRisk` and `Level5RiskBound` bounding its `roundRisk` by `δ`, at
settings chosen as formulas of the problem. With `N = |Q| + 2`, `L₁ = L + 1`,
`V = N (N |Σ| + 1)` and `G = log (4 V (2 N |Σ| + 4) / δ)`: `maxLeaves = N`,
`startUndecidedRate = 1/2` against `3/5`, `middleUndecidedRate = 1/2` against `3/4`, `εd` against
`7εd/8`, `searchShare = 3εd/4`, `edgeUndecidedRate = edgeTraffic = 3/4`, `testLevel = e^{−G}`,
`firstLook = ⌈6G⌉`, `edgeExcess = 2 L₁ (1 + 3N/4) G + 1`, and `probeBudget = ⌈V (112 G / εd + 1)⌉`.
The middles' test certifies the least yield, `3εd/16`. `Level5Gates` frees the start's and the
edges' rates and the edges' traffic, and adds the sift certificate (`sift_le`).
-/

noncomputable section

namespace OrthoDFA

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- A run's ending, as `RunEnds` checks it, is `runProbes`'s. -/
theorem runEnds_iff {σ : Type*} (M : _root_.DFA α σ) (bad : σ → Prop) (dist : σ → ARU → ℝ)
    (D : Measure (FreeMonoid α)) (C : Level5Settings) (J : ℕ) (ū : ℝ) (rd : FreeMonoid α → ARU) :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)),
      RunEnds (applyProbe C fun z => (rd z).cut) (Level5EndsWell M bad dist D C J ū rd) s xs ↔
        EndsWell M bad D (3 / 2 * C.disagreementRate)
          (checkedProbes M dist D rd C.walkStart C.maxLeaves J ū
            ⟨(runProbes C (fun z => (rd z).cut) s xs).1.tree,
              Function.uncurry (runProbes C (fun z => (rd z).cut) s xs).1.edges⟩) C.walkStart
          (harvestLaw D (fun z => (rd z).cut) C.walkStart
            ⟨(runProbes C (fun z => (rd z).cut) s xs).1.tree,
              Function.uncurry (runProbes C (fun z => (rd z).cut) s xs).1.edges⟩)
          C.harvestYield (runProbes C (fun z => (rd z).cut) s xs).2
  | s, [] => by simp [RunEnds, runProbes, EndsWell]
  | s, x :: xs => by
    cases h : applyProbe C (fun z => (rd z).cut) s x with
    | inl s' => simpa only [RunEnds, runProbes, h] using runEnds_iff M bad dist D C J ū rd s' xs
    | inr es => obtain ⟨e, s'⟩ := es; simp only [RunEnds, runProbes, h]; rfl

theorem lintegral_level5_eq {σ : Type*} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (M : _root_.DFA α σ) (O : TriOracle μ M) (bad : σ → Prop) (P : PrefixDistribution α)
    (C : Level5Settings) (J : ℕ) (ū : ℝ) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel5 C (O.read · ω) draws
          ¬ EndsWell M bad P.D (3 / 2 * C.disagreementRate)
            (checkedProbes M O.readDist P.D (O.read · ω) C.walkStart C.maxLeaves J ū
              result.hypothesis) C.walkStart
            (harvestLaw P.D (fun z => (O.read z ω).cut) C.walkStart result.hypothesis)
            C.harvestYield result.ending} ∂μ
      = ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (O.read z ω).cut)
          (Level5EndsWell M bad O.readDist P.D C J ū (O.read · ω))
          initialState (List.ofFn xs)} ∂μ := by
  congr 1; ext ω; congr 1; ext xs
  simp only [Set.mem_setOf_eq, runEnds_iff, indecisiveLStarLevel5]

theorem runEnds_iff_harvest {σ : Type*} (M : _root_.DFA α σ) (bad : σ → Prop)
    (dist : σ → ARU → ℝ) (D : Measure (FreeMonoid α)) (C : Level5Settings) (J : ℕ) (ū : ℝ)
    (rd : FreeMonoid α → ARU) :
    ∀ (s : Level5State α) (xs : List (FreeMonoid α)),
      RunEnds (applyProbe C fun z => (rd z).cut)
          (fun e s => Level5EndsWell M bad dist D C J ū rd e s ∨ ∃ h, toRoundEnd e = .harvest h)
          s xs ↔
        EndsWell M bad D (3 / 2 * C.disagreementRate)
          (checkedProbes M dist D rd C.walkStart C.maxLeaves J ū
            ⟨(runProbes C (fun z => (rd z).cut) s xs).1.tree,
              Function.uncurry (runProbes C (fun z => (rd z).cut) s xs).1.edges⟩) C.walkStart
          (harvestLaw D (fun z => (rd z).cut) C.walkStart
            ⟨(runProbes C (fun z => (rd z).cut) s xs).1.tree,
              Function.uncurry (runProbes C (fun z => (rd z).cut) s xs).1.edges⟩)
          C.harvestYield (runProbes C (fun z => (rd z).cut) s xs).2
        ∨ ∃ h, (runProbes C (fun z => (rd z).cut) s xs).2 = .harvest h
  | s, [] => by simp [RunEnds, runProbes, EndsWell]
  | s, x :: xs => by
    cases h : applyProbe C (fun z => (rd z).cut) s x with
    | inl s' =>
      simpa only [RunEnds, runProbes, h] using runEnds_iff_harvest M bad dist D C J ū rd s' xs
    | inr es => obtain ⟨e, s'⟩ := es; simp only [RunEnds, runProbes, h]; rfl

theorem lintegral_level5_harvest_eq {σ : Type*} {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}
    (M : _root_.DFA α σ) (O : TriOracle μ M) (bad : σ → Prop) (P : PrefixDistribution α)
    (C : Level5Settings) (J : ℕ) (ū : ℝ) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {draws | let result := indecisiveLStarLevel5 C (O.read · ω) draws
          ¬ EndsWell M bad P.D (3 / 2 * C.disagreementRate)
            (checkedProbes M O.readDist P.D (O.read · ω) C.walkStart C.maxLeaves J ū
              result.hypothesis) C.walkStart
            (harvestLaw P.D (fun z => (O.read z ω).cut) C.walkStart result.hypothesis)
            C.harvestYield result.ending ∧ ∀ h, result.ending ≠ .harvest h} ∂μ
      = ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {xs | ¬ RunEnds (applyProbe C fun z => (O.read z ω).cut)
          (fun e s => Level5EndsWell M bad O.readDist P.D C J ū (O.read · ω) e s
            ∨ ∃ h, toRoundEnd e = .harvest h)
          initialState (List.ofFn xs)} ∂μ := by
  congr 1; ext ω; congr 1; ext xs
  simp only [Set.mem_setOf_eq, runEnds_iff_harvest, indecisiveLStarLevel5, not_or, not_exists,
    ne_eq]

section Guarantee

variable [Nonempty α] {σ : Type*} [Fintype σ]

/-- Level 5's guarantee at settings chosen before the probability space, the probes'
distribution of length `L`, the oracle and the goodness level, with the start's test of `θs`
against `θs₁` and the edges' of `θe` at traffic `φe`. A consistent round's start and end sifts
stop undecided on at most `u` of the probes but for chance `δ/8`, once the start's alternative is
`2θs` and `u` is past what the silent tests allow. -/
def Level5Gates (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ) (θs θs₁ θe φe : ℝ) :
    Prop :=
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let K : ℝ := V * (112 / εd + 1)
  let W : ℝ := (n + 2) * (L + 1 + (n + 2) * A)
  let Xc : ℝ := 2 * (L + 1) * (1 + θe * (n + 2)) + 1
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧ settings.maxLeaves = Fintype.card σ + 2 ∧
    (∀ {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
      (P : PrefixDistribution α), P.L = L → ∀ (O : TriOracle μ M) (θ : ℝ) (J : ℕ) (ū : ℝ),
      let bad := fun q => ¬ O.Good θ q
      0 ≤ θ → 384 * (P.L + 1) * (n + 1) * θ ≤ 1 → 48 * θ ≤ θe →
      6 * (n + 1) * θ + 2 * unionLog (α := α) σ δ * prefixMax P.D k ≤ θs →
      32 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D k ≤ φe * θe →
      0 ≤ O.κ → 0 ≤ O.ε → 4 * (O.ε + O.κ) * K * G * W ≤ δ →
      1024 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D k ≤ εd → 0 ≤ ū →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
            ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
              (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
                (Fintype.card σ + 2) J ū result.hypothesis)
              settings.walkStart
              (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
              (min θs (min (φe * θe / 2) (3 * εd / 16))) result.ending} ∂μ
        ≤ ENNReal.ofReal (δ + 32 * ū / εd))
    ∧ ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D], (∀ᵐ x ∂D, x.toList.length ≤ L) →
      ∀ (read : FreeMonoid α → ARU) (u : ℝ),
      θs₁ = 2 * θs → θs ≤ 1 / 4 → εd ≤ 1 / 8 →
      2 * (3 * θs + (n + 2) * A * φe + 2 * θe * ((L + 1) * (n + 2)))
        + 2 * εd / 7 * (2 * (L + 2) + 2 + (n + 2) * A * Xc + 2 * θe * ((L + 1) * (n + 2))) ≤ u →
      (Measure.pi fun _ : Fin settings.probeBudget => D)
          {draws | let result := indecisiveLStarLevel5 settings read draws
            result.ending = .consistent ∧ u ≤ D.real (undSiftSet (fun z => (read z).cut)
              result.hypothesis.tree (Function.curry result.hypothesis.edges) k)}
        ≤ ENNReal.ofReal (δ / 8)

/-- `Level5Gates` with the noise fields concentrating from `kd`: the prefix mass is read at `kd`,
and the round may also fail where a string it reads before `kd` at a good read-state is undecided.
-/
def Level5GatesAt (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ) (θs θs₁ θe φe : ℝ) :
    Prop :=
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let K : ℝ := V * (112 / εd + 1)
  let W : ℝ := (n + 2) * (L + 1 + (n + 2) * A)
  let Xc : ℝ := 2 * (L + 1) * (1 + θe * (n + 2)) + 1
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧ settings.maxLeaves = Fintype.card σ + 2 ∧
    (∀ {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
      (P : PrefixDistribution α), P.L = L → ∀ (O : TriOracle μ M) (θ : ℝ) (J : ℕ) (ū : ℝ) (kd : ℕ),
      let bad := fun q => ¬ O.Good θ q
      0 ≤ θ → 384 * (P.L + 1) * (n + 1) * θ ≤ 1 → 48 * θ ≤ θe →
      6 * (n + 1) * θ + 2 * unionLog (α := α) σ δ * prefixMax P.D kd ≤ θs →
      32 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D kd ≤ φe * θe →
      0 ≤ O.κ → 0 ≤ O.ε → 4 * (O.ε + O.κ) * K * G * W ≤ δ →
      1024 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D kd ≤ εd → 0 ≤ ū →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
            ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
              (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
                (Fintype.card σ + 2) J ū result.hypothesis)
              settings.walkStart
              (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
              (min θs (min (φe * θe / 2) (3 * εd / 16))) result.ending
              ∧ ∀ h, result.ending ≠ .harvest h} ∂μ
        ≤ ENNReal.ofReal (δ + 32 * ū / εd) ∧
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
            ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
              (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
                (Fintype.card σ + 2) J ū result.hypothesis)
              settings.walkStart
              (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
              (min θs (min (φe * θe / 2) (3 * εd / 16))) result.ending} ∂μ
        ≤ ENNReal.ofReal (δ + 32 * ū / εd + if k < kd then 3 / 2 * θ * (K * G * W)
          + (O.ε + O.κ + 3 / 2 * θ) * ((∑ i ∈ Finset.Ico k kd, A ^ i) * (n + 2)) else 0))
    ∧ ∀ (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D], (∀ᵐ x ∂D, x.toList.length ≤ L) →
      ∀ (read : FreeMonoid α → ARU) (u : ℝ),
      θs₁ = 2 * θs → θs ≤ 1 / 4 → εd ≤ 1 / 8 →
      2 * (3 * θs + (n + 2) * A * φe + 2 * θe * ((L + 1) * (n + 2)))
        + 2 * εd / 7 * (2 * (L + 2) + 2 + (n + 2) * A * Xc + 2 * θe * ((L + 1) * (n + 2))) ≤ u →
      (Measure.pi fun _ : Fin settings.probeBudget => D)
          {draws | let result := indecisiveLStarLevel5 settings read draws
            result.ending = .consistent ∧ u ≤ D.real (undSiftSet (fun z => (read z).cut)
              result.hypothesis.tree (Function.curry result.hypothesis.edges) k)}
        ≤ ENNReal.ofReal (δ / 8)

set_option maxHeartbeats 4000000 in -- the chosen settings meet every condition
theorem level5_gates_at_all (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ)
    (θs θs₁ θe φe : ℝ) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1) (hεd0 : 0 < εd) (hεd1 : εd < 1) (hL : 1 ≤ L)
    (hθs0 : 0 < θs) (hθs₁ : θs < θs₁) (hθs₁1 : θs₁ < 1) (hθe : 0 ≤ θe) (hφe : 0 ≤ φe) :
    Level5GatesAt M εd L δ k θs θs₁ θe φe := by
  dsimp only [Level5GatesAt]
  set N : ℕ := Fintype.card σ + 2
  set A : ℕ := Fintype.card α
  have hA1 : (1 : ℝ) ≤ A := by exact_mod_cast Fintype.card_pos
  have hc0 : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
  have hNr : ((N : ℕ) : ℝ) = (Fintype.card σ : ℝ) + 2 := by simp only [N]; push_cast; ring
  have hN2 : (2 : ℝ) ≤ N := by rw [hNr]; linarith [(Nat.cast_nonneg _ : (0 : ℝ) ≤ Fintype.card σ)]
  set V : ℕ := N * (N * A + 1)
  have hVr : (V : ℝ) = ((Fintype.card σ : ℝ) + 2) * (((Fintype.card σ : ℝ) + 2) * A + 1) := by
    simp only [V]; push_cast; rw [hNr]
  have hV6 : (6 : ℝ) ≤ V := by
    rw [hVr]
    have h1 : (2 : ℝ) ≤ Fintype.card σ + 2 := by linarith
    have h2 : (3 : ℝ) ≤ ((Fintype.card σ : ℝ) + 2) * A + 1 := by nlinarith
    nlinarith [mul_le_mul h1 h2 (by norm_num) (by linarith)]
  set G := Real.log (4 * ((Fintype.card σ + 2) * ((Fintype.card σ + 2) * (A : ℝ) + 1))
    * (2 * (Fintype.card σ + 2) * A + 4) / δ) with hGd
  have hX : 4 * ((V : ℝ) * (2 * N * A + 4)) / δ
      = 4 * ((Fintype.card σ + 2) * ((Fintype.card σ + 2) * (A : ℝ) + 1))
        * (2 * (Fintype.card σ + 2) * A + 4) / δ := by
    rw [hVr, hNr]; ring
  have hX0 : 0 < 4 * ((V : ℝ) * (2 * N * A + 4)) / δ := by positivity
  have hX8 : 8 ≤ 4 * ((V : ℝ) * (2 * N * A + 4)) / δ := by
    rw [le_div_iff₀ hδ0]
    have h1 : (4 : ℝ) ≤ 2 * N * A + 4 := by nlinarith
    have h2 : (24 : ℝ) ≤ (V : ℝ) * (2 * N * A + 4) := by nlinarith
    nlinarith
  have hG2 : 2 ≤ G := by
    rw [hGd, ← hX]
    have : Real.exp 2 ≤ 8 := by
      have := Real.exp_one_lt_d9
      have h2 : Real.exp 2 = Real.exp 1 ^ 2 := by rw [← Real.exp_nat_mul]; norm_num
      rw [h2]; nlinarith [Real.exp_pos 1]
    rw [Real.le_log_iff_exp_le hX0]
    linarith
  have hexpG : Real.exp (-G) = δ / (4 * ((V : ℝ) * (2 * N * A + 4))) := by
    rw [Real.exp_neg, hGd, ← hX, Real.exp_log hX0]
    field_simp
  have hlogG : Real.log (1 / Real.exp (-G)) = G := by
    rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
  obtain ⟨n₀, hn₀d⟩ : ∃ x : ℕ, x = ⌈6 * G⌉₊ := ⟨_, rfl⟩
  have hn₀ : (n₀ : ℝ) ≤ 6 * G + 1 := by
    rw [hn₀d]; exact (Nat.ceil_lt_add_one (by positivity)).le
  have hn₀l : 6 * G ≤ (n₀ : ℝ) := by rw [hn₀d]; exact Nat.le_ceil _
  obtain ⟨b, hbd⟩ : ∃ x : ℕ, x = ⌈(V : ℝ) * (112 * G / εd + 1)⌉₊ := ⟨_, rfl⟩
  have hbl : (V : ℝ) * (112 * G / εd + 1) ≤ b := by rw [hbd]; exact Nat.le_ceil _
  have hbu : (b : ℝ) ≤ (V : ℝ) * (112 * G / εd + 1) + 1 := by
    rw [hbd]; exact (Nat.ceil_lt_add_one (by positivity)).le
  have hbK : (b : ℝ) ≤ (V : ℝ) * (112 / εd + 1) * G := by
    have : (V : ℝ) * (112 * G / εd + 1) + 1 ≤ (V : ℝ) * (112 / εd + 1) * G := by
      have e : (V : ℝ) * (112 / εd + 1) * G = V * (112 * G / εd) + V * G := by ring
      rw [e]
      have : (V : ℝ) + 1 ≤ V * G := by nlinarith
      nlinarith
    linarith
  set C : Level5Settings := ⟨k, N, θs, θe, 1 / 2, εd, θs₁, 3 / 4, 7 / 8 * εd, Real.exp (-G),
    n₀, fun _ => 2 * (L + 1) * (1 + θe * N) * G + 1, φe, 3 / 4 * εd, b⟩ with hC
  refine ⟨C, rfl, rfl, by rw [← hVr]; exact hbK, rfl,
    fun μ _ P hP O θ J ū kd hθ hθc hθe' hθs' hφe' hκ hε hεc hpc hū => ?_, ?_⟩
  swap
  · intro D _ hlen read u halt hθs4 hp8 hu
    have hSq : C.SeqOK :=
      { a_pos := Real.exp_pos _
        a_lt := by
          show Real.exp (-G) < 1
          rw [← Real.exp_zero]
          exact Real.exp_lt_exp.2 (by linarith)
        θs_pos := hθs0
        θs_alt := hθs₁
        θs_alt_lt := hθs₁1
        θpt_pos := by simp only [hC]; norm_num
        θpt_alt := by simp only [hC]; norm_num
        θpt_alt_lt := by simp only [hC]; norm_num
        εd_pos := hεd0
        εd_lt := hεd1
        εd_alt_pos := by simp only [hC]; positivity
        εd_alt := by simp only [hC]; linarith
        φ_pos := by simp only [hC]; positivity
        φ_lt := by simp only [hC]; linarith }
    have hG1 : (1 : ℝ) ≤ G := by linarith
    have hNr' : ((C.maxLeaves : ℕ) : ℝ) = (Fintype.card σ : ℝ) + 2 := hNr
    have hXc : 0 ≤ 2 * ((L : ℝ) + 1) * (1 + θe * ((Fintype.card σ : ℝ) + 2)) + 1 := by positivity
    have hX : ∀ j, C.edgeExcess j ≤ (2 * ((L : ℝ) + 1) * (1 + θe * ((Fintype.card σ : ℝ) + 2)) + 1)
        * Real.log (1 / C.testLevel) := by
      intro j
      show 2 * ((L : ℝ) + 1) * (1 + θe * N) * G + 1 ≤ _
      rw [show C.testLevel = Real.exp (-G) from rfl, hlogG, hNr]
      have key : ∀ X : ℝ, X * G + 1 ≤ (X + 1) * G := fun X => by nlinarith
      exact key _
    have hS := sift_le C D read hSq (by simp [C, N]) (by
        show 1 ≤ n₀
        have : (0 : ℝ) < n₀ := by linarith
        exact_mod_cast this) (fun j => by
        show 0 < 2 * ((L : ℝ) + 1) * (1 + θe * N) * G + 1
        positivity) (by simpa [C] using hp8) rfl (by simpa [C] using hθs4)
      (by simpa [C] using halt) (by simpa [C] using hθe) (by simpa [C] using hφe) hXc hX hlen
      (u := u) (by
        rw [hNr']
        simpa [C] using hu) C.probeBudget
    have hva : 2 * ((versionBound C (Fintype.card α) : ℝ) * C.testLevel) ≤ δ / 8 := by
      rw [show versionBound C (Fintype.card α) = V from rfl]
      show 2 * ((V : ℝ) * Real.exp (-G)) ≤ δ / 8
      rw [hexpG]
      have hV0 : (0 : ℝ) < V := by linarith
      have e : 2 * ((V : ℝ) * (δ / (4 * ((V : ℝ) * (2 * N * A + 4)))))
          = δ / (2 * (2 * N * A + 4)) := by field_simp; ring
      rw [e]
      exact div_le_div_of_nonneg_left hδ0.le (by norm_num) (by nlinarith)
    refine le_trans (measure_mono fun xs hxs => ?_) (hS.trans (ENNReal.ofReal_le_ofReal hva))
    obtain ⟨h1, h2⟩ := hxs
    exact runEnds_of_consistent C _ _ _ _ h1 h2
  subst hP
  have hy : min θs (min (φe * θe / 2) (3 * εd / 16)) ≤ C.harvestYield := by
    simp only [Level5Settings.harvestYield, hC]
    refine le_min (min_le_left _ _) (le_min ((min_le_right _ _).trans (min_le_left _ _))
      ((min_le_right _ _).trans ((min_le_right _ _).trans (by linarith))))
  have hT : 32 ≤ C.disagreementRate * (C.probeBudget : ℝ) := by
    show 32 ≤ εd * (b : ℝ)
    have h1 := mul_le_mul_of_nonneg_left hbl hεd0.le
    have e : εd * ((V : ℝ) * (112 * G / εd + 1)) = 112 * ((V : ℝ) * G) + εd * V := by
      field_simp
    have h3 : (6 : ℝ) * 2 ≤ V * G := mul_le_mul hV6 hG2 (by norm_num) (by linarith)
    have h4 : 0 ≤ εd * (V : ℝ) := by positivity
    linarith
  have hrisk : baseRisk σ O.κ O.ε θ P.D P.L C ≤ ENNReal.ofReal δ
      ∧ roundRisk σ O.κ O.ε θ P.D P.L C ≤ ENNReal.ofReal δ + headTerm (α := α) O.κ O.ε θ P.L C kd := by
    clear hT hū
    have hRB := level5_risk_bound_at_holds σ O.κ O.ε θ P.D P.L C δ kd
    dsimp only at hRB
    have hp₀ : 0 ≤ prefixMax P.D kd := (prefixMax_pos P.D kd).le
    have hΛ0 : 0 ≤ unionLog (α := α) σ δ := by
      unfold unionLog
      refine mul_nonneg (by positivity) (Real.log_nonneg ?_)
      rw [le_div_iff₀ hδ0]
      nlinarith
    have hL1 : (1 : ℝ) ≤ P.L := by exact_mod_cast hL
    have hpΛ : prefixMax P.D kd * unionLog (α := α) σ δ ≤ εd / (1024 * (P.L + 1)) := by
      rw [le_div_iff₀ (by positivity)]; nlinarith
    have hθ' : ((Fintype.card σ : ℝ) + 1) * θ ≤ 1 / 768 := by
      have : 0 ≤ ((Fintype.card σ : ℝ) + 1) * θ := by positivity
      nlinarith
    have hLn1 : (1 : ℝ) ≤ ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) := by nlinarith
    have hθL : ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) * θ ≤ 1 / 384 := by
      have e : 384 * ((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) * θ
          = 384 * (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) * θ) := by ring
      rw [e] at hθc
      linarith
    have hθ1 : θ ≤ 1 / 384 := le_trans (le_mul_of_one_le_left hθ hLn1) hθL
    set pΛ := prefixMax P.D kd * unionLog (α := α) σ δ
    have hpΛ0 : 0 ≤ pΛ := mul_nonneg hp₀ hΛ0
    have hpΛL : ((P.L : ℝ) + 1) * pΛ ≤ εd / 1024 := by
      have e : 1024 * ((P.L : ℝ) + 1) * unionLog (α := α) σ δ * prefixMax P.D kd
          = 1024 * (((P.L : ℝ) + 1) * pΛ) := by simp only [pΛ]; ring
      rw [e] at hpc
      linarith
    have hpΛ1 : pΛ ≤ εd / 2048 := by
      have : 2 * pΛ ≤ ((P.L : ℝ) + 1) * pΛ := by nlinarith
      linarith
    refine hRB hδ0 hδ1 hL (by simp [C, N]) hκ hε hθ ?_ ?_ ?_ ?_ ?_ hεd1 hθs₁ hθs₁1 rfl rfl rfl rfl ?_
      ?_ ?_ (Real.exp_pos _) ?_ ?_ ?_
    · simp only [hC]; linarith
    · simp only [hC]
      have e : 96 * (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1)) * θ
          = 96 * (((P.L : ℝ) + 1) * ((Fintype.card σ : ℝ) + 1) * θ) := by ring
      rw [e]; linarith
    · simp only [hC]
      have e2 : 2 * prefixMax P.D kd * unionLog (α := α) σ δ
          = 2 * unionLog (α := α) σ δ * prefixMax P.D kd := by ring
      rw [e2]; exact hθs'
    · simp only [hC]
      have e : 32 * ((P.L : ℝ) + 1) * prefixMax P.D kd * unionLog (α := α) σ δ
          = 32 * ((P.L : ℝ) + 1) * unionLog (α := α) σ δ * prefixMax P.D kd := by ring
      rw [e]; exact hφe' 
    · simp only [hC]
      have e : 256 * prefixMax P.D kd * unionLog (α := α) σ δ = 256 * pΛ := by simp only [pΛ]; ring
      rw [e]; linarith
    · simp only [hC]; rw [hlogG]; exact hn₀l
    · simp only [hC]; rw [hlogG]; linarith
    · intro j
      simp only [hC]
      rw [hlogG]
      have : (0 : ℝ) ≤ 2 * ((P.L : ℝ) + 1) * (1 + θe * N) * G := by positivity
      linarith
    · rw [show versionBound C (Fintype.card α) = V from rfl]
      simp only [hC]
      rw [hexpG]
      have : (0 : ℝ) < 4 * ((V : ℝ) * (2 * N * A + 4)) := by positivity
      rw [mul_div_assoc', div_le_iff₀ this]
      clear hεc
      nlinarith
    · rw [show versionBound C (Fintype.card α) = V from rfl]
      simp only [hC]
      rw [hlogG]
      exact hbl
    · simp only [hC]
      have hNW : ((P.L : ℝ) + 1) * N + N * A * N
          = ((Fintype.card σ : ℝ) + 2) * (P.L + 1 + ((Fintype.card σ : ℝ) + 2) * A) := by
        rw [hNr]; ring
      rw [hNW]
      have hW0 : 0 ≤ (O.ε + O.κ) * (((Fintype.card σ : ℝ) + 2)
          * (P.L + 1 + ((Fintype.card σ : ℝ) + 2) * A)) := by positivity
      have hbK' : (b : ℝ) ≤ ((Fintype.card σ : ℝ) + 2) * (((Fintype.card σ : ℝ) + 2) * A + 1)
          * (112 / εd + 1) * G := by rw [← hVr]; exact hbK
      have := mul_le_mul_of_nonneg_right hbK' hW0
      linarith only [this, hεc]
  refine ⟨?_, ?_⟩
  · refine le_trans (lintegral_mono fun ω => measure_mono fun xs hxs =>
      And.imp_left (fun hn h => hn (EndsWell.mono_yield hy h)) hxs) ?_
    refine (lintegral_level5_harvest_eq M O _ P C J ū).trans_le ?_
    refine (level5_consistent_risk_holds M O θ P C J ū hū).trans ?_
    rw [ENNReal.ofReal_add hδ0.le (by positivity)]
    exact add_le_add hrisk.1 le_rfl
  · refine le_trans (lintegral_mono fun ω => measure_mono fun xs hxs h => hxs
      (EndsWell.mono_yield hy h)) ?_
    refine (lintegral_level5_eq M O _ P C J ū).trans_le ?_
    refine (level5_round_risk_holds M O θ P C J ū hū hT).trans ?_
    have hκε : 0 ≤ O.ε + O.κ + 3 / 2 * θ := by positivity
    rw [ENNReal.ofReal_add (by positivity) (by split_ifs <;> positivity),
      ENNReal.ofReal_add hδ0.le (by positivity), add_right_comm]
    refine add_le_add ?_ le_rfl
    clear hT hū
    refine le_trans ?_ (add_le_add (le_refl (ENNReal.ofReal δ))
      (show headTerm (α := α) O.κ O.ε θ P.L C kd ≤ _ from ?_))
    swap
    · unfold headTerm
      refine ENNReal.ofReal_le_ofReal ?_
      show (if k < kd then _ else _) ≤ _
      split_ifs with hk
      · have hNW : ((P.L : ℝ) + 1) * N + N * A * N
            = ((Fintype.card σ : ℝ) + 2) * (P.L + 1 + ((Fintype.card σ : ℝ) + 2) * A) := by
          rw [hNr]; ring
        have hbK' : (b : ℝ) ≤ ((Fintype.card σ : ℝ) + 2) * (((Fintype.card σ : ℝ) + 2) * A + 1)
            * (112 / εd + 1) * G := by rw [← hVr]; exact hbK
        show 3 / 2 * θ * ((b : ℝ) * (((P.L : ℝ) + 1) * N + N * A * N))
            + (O.ε + O.κ + 3 / 2 * θ) * (((∑ i ∈ Finset.Ico k kd, A ^ i : ℕ) : ℝ) * N) ≤ _
        rw [hNW, hNr]
        push_cast
        have hW0 : 0 ≤ ((Fintype.card σ : ℝ) + 2) * (P.L + 1 + ((Fintype.card σ : ℝ) + 2) * A) := by
          positivity
        have := mul_le_mul_of_nonneg_right hbK' hW0
        have hθ32 : 0 ≤ 3 / 2 * θ := by positivity
        nlinarith [mul_le_mul_of_nonneg_left this hθ32]
      · exact le_rfl
    exact hrisk.2


/-- At `kd = k` nothing is read before `kd`. -/
theorem level5_gates_all (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ)
    (θs θs₁ θe φe : ℝ) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1) (hεd0 : 0 < εd) (hεd1 : εd < 1) (hL : 1 ≤ L)
    (hθs0 : 0 < θs) (hθs₁ : θs < θs₁) (hθs₁1 : θs₁ < 1) (hθe : 0 ≤ θe) (hφe : 0 ≤ φe) :
    Level5Gates M εd L δ k θs θs₁ θe φe := by
  obtain ⟨C, h1, h2, h3, h4, h5, h6⟩ := level5_gates_at_all M εd L δ k θs θs₁ θe φe hδ0 hδ1 hεd0
    hεd1 hL hθs0 hθs₁ hθs₁1 hθe hφe
  refine ⟨C, h1, h2, h3, h4, fun μ _ P hP O θ J ū hθ hθc hθe' hθs' hφe' hκ hε hεc hpc hū => ?_, h6⟩
  simpa using (h5 μ P hP O θ J ū k hθ hθc hθe' hθs' hφe' hκ hε hεc hpc hū).2

/-- `Level5Guarantee` with the settings chosen before the probability space, the probes'
distribution of length `L`, the oracle and the goodness level. -/
def Level5UniformAll (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ) : Prop :=
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let K : ℝ := V * (112 / εd + 1)
  let W : ℝ := (n + 2) * (L + 1 + (n + 2) * A)
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧ settings.maxLeaves = Fintype.card σ + 2 ∧
    ∀ {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
      (P : PrefixDistribution α), P.L = L → ∀ (O : TriOracle μ M) (θ : ℝ) (J : ℕ) (ū : ℝ),
      let bad := fun q => ¬ O.Good θ q
      0 ≤ θ → 384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
      0 ≤ O.κ → 0 ≤ O.ε → 4 * (O.ε + O.κ) * K * G * W ≤ δ →
      1024 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D k ≤ εd → 0 ≤ ū →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
            ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
              (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
                (Fintype.card σ + 2) J ū result.hypothesis)
              settings.walkStart
              (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
              (3 * εd / 16) result.ending} ∂μ
        ≤ ENNReal.ofReal (δ + 32 * ū / εd)

/-- `Level5UniformAll` with the noise fields concentrating from `kd`. -/
def Level5UniformAt (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ) : Prop :=
  let n : ℝ := Fintype.card σ
  let A : ℝ := Fintype.card α
  let V : ℝ := (n + 2) * ((n + 2) * A + 1)
  let G : ℝ := Real.log (4 * V * (2 * (n + 2) * A + 4) / δ)
  let K : ℝ := V * (112 / εd + 1)
  let W : ℝ := (n + 2) * (L + 1 + (n + 2) * A)
  ∃ settings : Level5Settings, settings.walkStart = k ∧ settings.disagreementRate = εd ∧
    (settings.probeBudget : ℝ) ≤ K * G ∧ settings.maxLeaves = Fintype.card σ + 2 ∧
    ∀ {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
      (P : PrefixDistribution α), P.L = L → ∀ (O : TriOracle μ M) (θ : ℝ) (J : ℕ) (ū : ℝ) (kd : ℕ),
      let bad := fun q => ¬ O.Good θ q
      0 ≤ θ → 384 * (P.L + 1) * (n + 1) * θ ≤ 1 →
      0 ≤ O.κ → 0 ≤ O.ε → 4 * (O.ε + O.κ) * K * G * W ≤ δ →
      1024 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D kd ≤ εd → 0 ≤ ū →
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
            ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
              (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
                (Fintype.card σ + 2) J ū result.hypothesis)
              settings.walkStart
              (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
              (3 * εd / 16) result.ending
              ∧ ∀ h, result.ending ≠ .harvest h} ∂μ
        ≤ ENNReal.ofReal (δ + 32 * ū / εd) ∧
      ∫⁻ ω, (Measure.pi fun _ : Fin settings.probeBudget => P.D)
          {draws | let result := indecisiveLStarLevel5 settings (O.read · ω) draws
            ¬ EndsWell M bad P.D (3 / 2 * settings.disagreementRate)
              (checkedProbes M O.readDist P.D (O.read · ω) settings.walkStart
                (Fintype.card σ + 2) J ū result.hypothesis)
              settings.walkStart
              (harvestLaw P.D (fun z => (O.read z ω).cut) settings.walkStart result.hypothesis)
              (3 * εd / 16) result.ending} ∂μ
        ≤ ENNReal.ofReal (δ + 32 * ū / εd + if k < kd then 3 / 2 * θ * (K * G * W)
          + (O.ε + O.κ + 3 / 2 * θ) * ((∑ i ∈ Finset.Ico k kd, A ^ i) * (n + 2)) else 0)

set_option maxHeartbeats 1000000 in -- the old settings meet the general conditions
theorem level5_guarantee_at_all (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ) (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) (hεd0 : 0 < εd) (hεd1 : εd < 1) (hL : 1 ≤ L) :
    Level5UniformAt M εd L δ k := by
  obtain ⟨C, h1, h2, h3, h4, h5, -⟩ := level5_gates_at_all M εd L δ k (1 / 2) (3 / 5) (3 / 4) (3 / 4)
    hδ0 hδ1 hεd0 hεd1 hL (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  refine ⟨C, h1, h2, h3, h4, fun μ _ P hP O θ J ū kd hθ hθc hκ hε hεc hpc hū => ?_⟩
  have hp₀ : 0 ≤ prefixMax P.D kd := (prefixMax_pos P.D kd).le
  have hΛ0 : 0 ≤ unionLog (α := α) σ δ := by
    unfold unionLog
    refine mul_nonneg (by positivity) (Real.log_nonneg ?_)
    rw [le_div_iff₀ hδ0]
    have : (1 : ℝ) ≤ Fintype.card α := by exact_mod_cast Fintype.card_pos
    nlinarith
  have hn0 : (0 : ℝ) ≤ Fintype.card σ := Nat.cast_nonneg _
  have hL1 : (1 : ℝ) ≤ P.L := by rw [hP]; exact_mod_cast hL
  have hpΛ : 0 ≤ unionLog (α := α) σ δ * prefixMax P.D kd := mul_nonneg hΛ0 hp₀
  have hθ1 : ((Fintype.card σ : ℝ) + 1) * θ ≤ 1 / 768 := by
    have h1 : (1 : ℝ) ≤ 2 * (P.L + 1) := by linarith
    have : 0 ≤ ((Fintype.card σ : ℝ) + 1) * θ := by positivity
    nlinarith
  have hθ0 : θ ≤ 1 / 768 := by nlinarith
  have hy : 3 * εd / 16 ≤ min (1 / 2) (min (3 / 4 * (3 / 4) / 2) (3 * εd / 16)) :=
    le_min (by linarith) (le_min (by linarith) le_rfl)
  have h5' := h5 μ P hP O θ J ū kd hθ hθc (by linarith) ?_ ?_ hκ hε hεc hpc hū
  · exact ⟨le_trans (lintegral_mono fun ω => measure_mono fun xs hxs =>
        And.imp_left (fun hn h => hn (EndsWell.mono_yield hy h)) hxs) h5'.1,
      le_trans (lintegral_mono fun ω => measure_mono fun xs hxs h => hxs
        (EndsWell.mono_yield hy h)) h5'.2⟩
  · have : 2 * unionLog (α := α) σ δ * prefixMax P.D kd ≤ εd / 512 := by
      have e : 1024 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D kd
          = 1024 * (P.L + 1) * (unionLog (α := α) σ δ * prefixMax P.D kd) := by ring
      rw [e] at hpc
      nlinarith
    nlinarith
  · have e : 1024 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D kd
        = 32 * (32 * (P.L + 1) * unionLog (α := α) σ δ * prefixMax P.D kd) := by ring
    rw [e] at hpc
    nlinarith

/-- At `kd = k` nothing is read before `kd`. -/
theorem level5_guarantee_all (M : _root_.DFA α σ) (εd : ℝ) (L : ℕ) (δ : ℝ) (k : ℕ) (hδ0 : 0 < δ)
    (hδ1 : δ ≤ 1) (hεd0 : 0 < εd) (hεd1 : εd < 1) (hL : 1 ≤ L) :
    Level5UniformAll M εd L δ k := by
  obtain ⟨C, h1, h2, h3, h4, h5⟩ := level5_guarantee_at_all M εd L δ k hδ0 hδ1 hεd0 hεd1 hL
  refine ⟨C, h1, h2, h3, h4, fun μ _ P hP O θ J ū hθ hθc hκ hε hεc hpc hū => ?_⟩
  simpa using (h5 μ P hP O θ J ū k hθ hθc hκ hε hεc hpc hū).2

end Guarantee



theorem level5_correct_holds : Level5Correct := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ εd P δ k J ū
  intro hδ0 hδ1 hεd0 hεd1 hL hθ hθc hκ hε hεc hpc hū
  obtain ⟨settings, h1, h2, h3, -, h⟩ :=
    level5_guarantee_all (σ := σ) M εd P.L δ k hδ0 hδ1 hεd0 hεd1 hL
  exact ⟨settings, h1, h2, h3, h μ P rfl O θ J ū hθ hθc hκ hε hεc hpc hū⟩

theorem level5_correct_at_holds : Level5CorrectAt := by
  intro α _ _ _ σ _ Ω _ μ _ M O θ εd P δ k kd J ū
  intro hδ0 hδ1 hεd0 hεd1 hL hθ hθc hκ hε hεc hpc hū
  obtain ⟨settings, h1, h2, h3, -, h⟩ :=
    level5_guarantee_at_all (σ := σ) M εd P.L δ k hδ0 hδ1 hεd0 hεd1 hL
  exact ⟨settings, h1, h2, h3, h μ P rfl O θ J ū kd hθ hθc hκ hε hεc hpc hū⟩

theorem level5_family_correct_holds : Level5FamilyCorrect := by
  intro α _ _ _ σ _ Ω _ μ _ fam θ εd P δ k J ū
  exact level5_correct_holds fam.M fam.oracle θ εd P δ k J ū

end OrthoDFA
