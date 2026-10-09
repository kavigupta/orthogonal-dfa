import OrthoDFA.Proofs.PowerCase

/-!
# The power of the round's split tests

At each key, the round's first test that is either in the power case or splits is a power-case
test that does not split with chance at most `e^{−τ²}` times the chance there is such a test.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory

variable {α : Type*} [Fintype α] [DecidableEq α]
variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

section Strings

variable (K : StageKnobs α) (R : CutReads α)

theorem testStrings_nodup {t : DTree α} {pool : List (FreeMonoid α)} {path : List Bool}
    {d : FreeMonoid α} {skip : FreeMonoid α → Prop} :
    ((testStrings K R t pool path d skip).map Prod.fst).Nodup := by
  classical
  unfold testStrings
  have key : ∀ (l acc : List (FreeMonoid α × Bool)), (acc.map Prod.fst).Nodup →
      ((l.foldl (fun acc p => if p.1 ∈ acc.map Prod.fst ∨ skip p.1 then acc
        else acc ++ [p]) acc).map Prod.fst).Nodup := by
    intro l
    induction l with
    | nil => exact fun acc h => h
    | cons p l ih =>
      intro acc h
      simp only [List.foldl_cons]
      refine ih _ ?_
      split_ifs with hc
      · exact h
      · rw [List.map_append, List.nodup_append]
        refine ⟨h, List.nodup_singleton _, ?_⟩
        simp only [List.map_cons, List.map_nil, List.mem_singleton]
        rintro a ha b rfl rfl
        exact hc (.inl ha)
  exact key _ [] List.nodup_nil

end Strings

section Mask

variable (C : StrongCfg α) {B : State} {F : Finset (FreeMonoid α)} {f₁ f₂ : FreeMonoid α → ℝ}

theorem stepTest_mask {A : RoundAcc α} {x : FreeMonoid α} (hw : WitPool A.s)
    (h : ∀ z ∈ stepPre C F A x, f₁ z = f₂ z) :
    stepTest C (rd B F f₁) A x = stepTest C (rd B F f₂) A x := by
  have hag : ∀ b, (b ∈ A.s.pool ∨ ∃ i, C.k ≤ i ∧ b = prefixOf x i) →
      AgreeOne C.K F f₁ f₂ A.s.tree b := fun b hb e m hm v hv =>
    h _ (mem_stepReads e hb (.inl hm) hv)
  have hpool : ∀ b ∈ A.s.pool, AgreeOne C.K F f₁ f₂ A.s.tree b := fun b hb => hag b (.inl hb)
  have hwit : ∀ p c q y, A.s.edges p c = some (q, y) → AgreeOne C.K F f₁ f₂ A.s.tree y :=
    fun p c q y he => hpool y (hw p c q y he)
  have hpo : probeOutcome (rd B F f₁) A.s.tree A.s.edges C.k x
      = probeOutcome (rd B F f₂) A.s.tree A.s.edges C.k x :=
    probeOutcome_congr fun i hi m hm => (hag _ (.inr ⟨i, hi, rfl⟩)).tree m hm
  have hsk : stepSkip C.K (rd B F f₁) C.k A.s x = stepSkip C.K (rd B F f₂) C.k A.s x := rfl
  unfold stepTest
  rw [hpo, hsk]
  split
  · rename_i ps fd ho
    have hfd : C.k ≤ fd - 1 := by have := probeOutcome_edge_gt _ ho; omega
    rw [seedStep_key_congr hwit (hag _ (.inr ⟨fd - 1, hfd, rfl⟩))]
    rcases hk : (seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
      (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd).key with _ | ⟨κ1, κ2⟩
    · rfl
    · have hr : (∃ s1 y sprime, seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
          (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd = .split κ2 s1 y sprime)
          ∨ ∃ s1 sprime, seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
            (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd = .member s1 sprime κ2 := by
        revert hk
        rcases seedStep C.K (rd B F f₂) A.s.tree A.s.pool A.s.edges
          (stepSkip C.K (rd B F f₂) C.k A.s x) C.K.forced C.k x ps fd with
          ⟨d, s1, y, sp⟩ | ⟨s1, sp, d⟩ | b | _ <;> intro hk <;>
          simp only [SeedResult.key, Option.some.injEq, Prod.mk.injEq, reduceCtorEq] at hk
        · exact .inl ⟨s1, y, sp, by rw [hk.2]⟩
        · exact .inr ⟨s1, sp, by rw [hk.2]⟩
      obtain ⟨c, m, hm, rfl⟩ := seedStep_dist C.K _ hr
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
      exact testStrings_congr (fun b hb => (hpool b hb).tree)
        fun b hb => (hpool b hb).letter' c m hm
  · rfl

theorem stepTest_ts {R : CutReads α} {A : RoundAcc α} {x : FreeMonoid α} {κ : TestKey α}
    {ts : List (FreeMonoid α × Bool)} (h : stepTest C R A x = some (κ, ts)) :
    ts = testStrings C.K R A.s.tree A.s.pool κ.1 κ.2 (stepSkip C.K R C.k A.s x κ) := by
  unfold stepTest at h
  split at h
  · obtain ⟨κ', -, hκ⟩ := Option.map_eq_some_iff.1 h
    simp only [Prod.mk.injEq] at hκ
    obtain ⟨rfl, rfl⟩ := hκ
    rfl
  · exact absurd h (by simp)

theorem startAcc_witPool (R : CutReads α) (seed : List (FreeMonoid α)) :
    WitPool (startAcc C R seed).s := fun p c q y h =>
  (closeK_poolIn C.K R (Bs := {b | b ∈ seed}) (fun b hb => hb)
    (fun _ _ _ _ h => by simp at h)).2 p c q y h

theorem startAcc_mask {seed : List (FreeMonoid α)}
    (h : ∀ z ∈ (startAcc C (rd B F f₁) seed).s.log, f₁ z = f₂ z) :
    startAcc C (rd B F f₁) seed = startAcc C (rd B F f₂) seed := by
  have hag : ∀ b ∈ seed, AgreeOne C.K F f₁ f₂ (.node 1 .leaf .leaf) b :=
    fun b hb e m hm v hv => h _ (mem_stepReads e (.inl hb) (.inl hm) hv)
  simp only [startAcc, initialK, closeK]
  rw [closeEdges_congr hag]

end Mask

section Find

variable (C : StrongCfg α) (R : CutReads α) (P : RoundAcc α → FreeMonoid α → Prop)

theorem passFind_spec :
    ∀ (probes : List (FreeMonoid α)) (A A' : RoundAcc α) (x' : FreeMonoid α),
      passFind C R P A probes = some (A', x') → P A' x'
  | [], _, _, _, h => by simp [passFind] at h
  | x :: xs, A, A', x', h => by
    unfold passFind at h
    split_ifs at h with hg hp
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact hp
    · exact passFind_spec xs _ A' x' h

theorem roundFind_spec :
    ∀ (n j : ℕ) (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws)
      (A' : RoundAcc α) (x' : FreeMonoid α),
      roundFind C R P n j A first d = some (A', x') → P A' x' ∧ (WitPool A.s → WitPool A'.s)
  | 0, _, _, _, _, _, _, h => by simp [roundFind] at h
  | n + 1, j, A, first, d, A', x', h => by
    simp only [roundFind] at h
    rcases hf : passFind C R P (passStart A) (first ++ List.ofFn (d 0).1) with _ | ⟨A₁, x₁⟩
    · rw [hf] at h
      simp only [] at h
      have hw' := strongReading_witPool C R j A first (d 0)
      rcases hR : strongReading C R j A first (d 0) with ⟨A'', e | lv⟩ <;> rw [hR] at h hw' <;>
        simp only [] at h
      · exact absurd h (by simp)
      obtain ⟨h1, h2⟩ := roundFind_spec n (j + 1) A'' lv (Fin.tail d) A' x' h
      exact ⟨h1, fun hw => h2 (hw' hw)⟩
    · rw [hf] at h
      simp only [Option.some.injEq] at h
      rw [h] at hf
      exact ⟨passFind_spec C R P _ _ _ _ hf,
        fun hw => (passFind_some_readLog C R P _ _ _ _ hf).2 hw⟩

open scoped Classical in
/-- `readLog` as a finite set. -/
noncomputable def readLogF (A : RoundAcc α) : Finset (FreeMonoid α) :=
  A.s.log ∪ A.reads ∪ ((A.s.tested.filter fun e => e.2 ∉ C.K.forced).map Prod.fst).toFinset

theorem coe_readLogF (A : RoundAcc α) : (↑(readLogF C A) : Set (FreeMonoid α)) = readLog C A := by
  ext z
  simp [readLogF, readLog]

/-- `roundE` as a finite set. -/
noncomputable def roundEF (Q : CutReads α → RoundAcc α → FreeMonoid α → Prop) (n j : ℕ)
    (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws) :
    Finset (FreeMonoid α) :=
  match roundFind C R (Q R) n j A first d with
  | some (A', x') => readLogF C A' ∪ stepPre C R.F A' x'
  | none => readLogF C (strongRound C R n j A first d).2.1

theorem coe_roundEF (Q : CutReads α → RoundAcc α → FreeMonoid α → Prop) (n j : ℕ)
    (A : RoundAcc α) (first : List (FreeMonoid α)) (d : Fin n → C.Draws) :
    (↑(roundEF C R Q n j A first d) : Set (FreeMonoid α)) = roundE C Q R n j A first d := by
  unfold roundEF roundE
  rcases roundFind C R (Q R) n j A first d with _ | ⟨A', x'⟩ <;>
    simp only [Finset.coe_union, coe_readLogF]

end Find

section Count

theorem mq_clean [IsProbabilityMeasure μ] (O : Oracle μ (FreeMonoid α)) {ω : Ω}
    (hω : ω ∈ cleanAll O) (w : FreeMonoid α) : O.mq w ω = 0 ∨ O.mq w ω = 1 := by
  rcases O.label_bit w with hl | hl <;> rcases hω w with hn | hn <;>
    norm_num [Oracle.mq, hl, hn]

theorem count_ones {f : FreeMonoid α → ℝ} (hf : ∀ w, f w = 0 ∨ f w = 1) :
    ∀ l : List (FreeMonoid α × Bool),
      (((l.filter fun p => f p.1 = 1).length : ℕ) : ℝ) = ((l.map Prod.fst).map f).sum
  | [] => by simp
  | p :: l => by
    rw [List.filter_cons]
    rcases hf p.1 with h | h
    · simp [h, count_ones hf l]
    · simp [h, count_ones hf l]; ring

/-- A test whose sides are read and whose statistic reaches its threshold splits. -/
theorem verdict_split {K : StageKnobs α} {R : CutReads α} {t : DTree α}
    {pool : List (FreeMonoid α)} {path : List Bool} {d : FreeMonoid α} {tests : ℕ}
    {skip : FreeMonoid α → Prop} {ts : List (FreeMonoid α × Bool)}
    (hf : ∀ w, R.f w = 0 ∨ R.f w = 1) (hts : ts = testStrings K R t pool path d skip)
    (h1 : (sideOf ts true).Nonempty) (h0 : (sideOf ts false).Nonempty)
    (hθ : Real.log (2 * (tests : ℝ) / K.splitFpr)
      ≤ harm (sideOf ts true).card (sideOf ts false).card
        * ((∑ w ∈ sideOf ts true, R.f w) / (sideOf ts true).card
          - (∑ w ∈ sideOf ts false, R.f w) / (sideOf ts false).card) ^ 2) :
    verdict K R t pool path d tests skip = .split := by
  subst hts
  have hnd := testStrings_nodup K R (t := t) (pool := pool) (path := path) (d := d) (skip := skip)
  have hsub : ∀ b, ((((testStrings K R t pool path d skip).filter fun p => p.2 = b)).map
      Prod.fst).Nodup := fun b => ((List.filter_sublist).map Prod.fst).nodup hnd
  have hlen : ∀ b, ((((testStrings K R t pool path d skip).filter fun p => p.2 = b)).length : ℝ)
      = (sideOf (testStrings K R t pool path d skip) b).card := by
    intro b
    unfold sideOf
    rw [List.toFinset_card_of_nodup (hsub b), List.length_map]
  have hsum : ∀ b, (((((testStrings K R t pool path d skip).filter fun p => p.2 = b)).filter
      fun p => R.f p.1 = 1).length : ℝ)
      = ∑ w ∈ sideOf (testStrings K R t pool path d skip) b, R.f w := by
    intro b
    unfold sideOf
    rw [List.sum_toFinset _ (hsub b), count_ones hf]
  have hc1 : (0 : ℝ) < (sideOf (testStrings K R t pool path d skip) true).card := by
    exact_mod_cast h1.card_pos
  have hc0 : (0 : ℝ) < (sideOf (testStrings K R t pool path d skip) false).card := by
    exact_mod_cast h0.card_pos
  unfold verdict
  simp only []
  rw [hlen true, hlen false, hsum true, hsum false, if_pos ⟨hc1, hc0, hθ⟩]

end Count

section Choice

variable (C : StrongCfg α) (O : Oracle μ (FreeMonoid α)) (F : Finset (FreeMonoid α)) (τ : ℝ)
  (κ : TestKey α) (seed : List (FreeMonoid α)) (Rmax : ℕ) (d : Fin Rmax → C.Draws)

/-- The step of the round with `κ` taken as not splitting at which it first reaches a power-case
test at `κ`, and the test the step reaches. -/
noncomputable def powerSel (R : CutReads α) :
    Option ((RoundAcc α × FreeMonoid α) × Option (TestKey α × List (FreeMonoid α × Bool))) :=
  (roundFind (C.force κ) R (CaseAt (C.force κ) O F τ κ R) Rmax 0 (startAcc C R seed) [] d).map
    fun Ax => (Ax, stepTest C R Ax.1 Ax.2)

/-- What decides `powerSel`. -/
noncomputable def powerE (R : CutReads α) : Finset (FreeMonoid α) :=
  roundEF (C.force κ) R (CaseAt (C.force κ) O F τ κ) Rmax 0 (startAcc C R seed) [] d

/-- The strings the chosen test counts. -/
def choiceTs :
    Option ((RoundAcc α × FreeMonoid α) × Option (TestKey α × List (FreeMonoid α × Bool)))
      → List (FreeMonoid α × Bool)
  | some (_, some (_, ts)) => ts
  | _ => []

theorem powerSel_mask {B : State} {f₁ f₂ : FreeMonoid α → ℝ}
    (hf : ∀ z ∈ powerE C O F τ κ seed Rmax d (rd B F f₁), f₁ z = f₂ z) :
    powerE C O F τ κ seed Rmax d (rd B F f₂) = powerE C O F τ κ seed Rmax d (rd B F f₁)
      ∧ powerSel C O F τ κ seed Rmax d (rd B F f₂)
        = powerSel C O F τ κ seed Rmax d (rd B F f₁) := by
  have hf' : ∀ z ∈ roundE (C.force κ) (CaseAt (C.force κ) O F τ κ) (rd B F f₁) Rmax 0
      (startAcc C (rd B F f₁) seed) [] d, f₁ z = f₂ z := fun z hz =>
    hf z (by rw [powerE, ← Finset.mem_coe, coe_roundEF]; exact hz)
  have hw := startAcc_witPool C (rd B F f₁) seed
  have hstart : startAcc C (rd B F f₁) seed = startAcc C (rd B F f₂) seed :=
    startAcc_mask C fun z hz => hf' z (roundFind_readLog (C.force κ) _ _ Rmax 0 _ [] d (.inl hz))
  have hP : ∀ A x, WitPool A.s →
      (∀ z ∈ readLog (C.force κ) A ∪ ↑(stepPre (C.force κ) F A x), f₁ z = f₂ z) →
      (CaseAt (C.force κ) O F τ κ (rd B F f₁) A x
        ↔ CaseAt (C.force κ) O F τ κ (rd B F f₂) A x) := by
    intro A x hwA h
    unfold CaseAt
    rw [stepTest_mask (C.force κ) hwA fun z hz => h z (.inr hz)]
  have hrf := roundFind_mask (C.force κ) (CaseAt (C.force κ) O F τ κ) hP Rmax 0 _ [] d hw hf'
  unfold powerE powerSel roundEF
  rw [← hstart, ← hrf]
  rcases hr : roundFind (C.force κ) (rd B F f₁) (CaseAt (C.force κ) O F τ κ (rd B F f₁)) Rmax 0
      (startAcc C (rd B F f₁) seed) [] d with _ | ⟨A', x'⟩
  · have hround := strongRound_mask (C.force κ) Rmax 0 (startAcc C (rd B F f₁) seed) [] d hw
      fun z hz => hf' z (by simp only [roundE, hr]; exact hz)
    simp only [Option.map_none, and_true]
    rw [hround]
  · have hpre : ∀ z ∈ stepPre C F A' x', f₁ z = f₂ z := fun z hz =>
      hf' z (by simp only [roundE, hr]; exact .inr hz)
    have hw' := (roundFind_spec (C.force κ) _ _ Rmax 0 _ [] d A' x' hr).2 hw
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq, true_and]
    exact (stepTest_mask C hw' hpre).symm

theorem powerSel_disjoint (B : State) (f : FreeMonoid α → ℝ) :
    Disjoint ((choiceTs (powerSel C O F τ κ seed Rmax d (rd B F f))).map Prod.fst).toFinset
      (powerE C O F τ κ seed Rmax d (rd B F f)) := by
  unfold powerSel powerE roundEF
  rcases hr : roundFind (C.force κ) (rd B F f) (CaseAt (C.force κ) O F τ κ (rd B F f)) Rmax 0
      (startAcc C (rd B F f) seed) [] d with _ | ⟨A', x'⟩
  · simp [choiceTs]
  · obtain ⟨ts, hst, hfresh, -⟩ := (roundFind_spec (C.force κ) _ _ Rmax 0 _ [] d A' x' hr).1
    rw [stepTest_force] at hst
    simp only [Option.map_some, hst, choiceTs]
    rw [Finset.disjoint_left]
    intro z hz hzE
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hz)
    obtain ⟨hlog, hreads, hpre, hown⟩ := hfresh p hp
    rcases Finset.mem_union.1 hzE with hz | hz
    · rw [← Finset.mem_coe, coe_readLogF] at hz
      rcases hz with hz | hz | ⟨κ', hκ', hf⟩
      · exact hlog hz
      · exact hreads hz
      · exact hf (by rw [hown κ' hκ']; exact Set.mem_singleton _)
    · exact hpre hz

end Choice

section Power

variable [IsProbabilityMeasure μ] (C : StrongCfg α) (O : Oracle μ (FreeMonoid α)) (B : State)
  (F : Finset (FreeMonoid α)) (seed : List (FreeMonoid α)) (τ : ℝ) (κ : TestKey α) (Rmax : ℕ)
  (d : Fin Rmax → C.Draws)

/-- In the round, which takes no key as not splitting, the first test at `κ` in the power case
or splitting is in the power case and does not split with chance at most `e^{−τ²}` times the
chance there is such a test. -/
theorem power_miss (h0 : C.K.forced = ∅) (hτ : 0 ≤ τ) :
    μ {ω | ∃ A x, roundFind C (readsAt O B F ω) (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0
          (startAcc C (readsAt O B F ω) seed) [] d = some (A, x)
        ∧ CaseAt C O F τ κ (readsAt O B F ω) A x
        ∧ verdict C.K (readsAt O B F ω) A.s.tree A.s.pool κ.1 κ.2
            (A.s.tree.paths.length * Fintype.card α)
            (stepSkip C.K (readsAt O B F ω) C.k A.s x κ) ≠ .split}
      ≤ ENNReal.ofReal (Real.exp (-τ ^ 2)) * μ {ω | roundFind C (readsAt O B F ω)
          (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d
        ≠ none} := by
  classical
  have hdet : ∀ ω ω', (∀ z ∈ powerE C O F τ κ seed Rmax d (readsAt O B F ω),
      O.noise z ω = O.noise z ω') →
      powerE C O F τ κ seed Rmax d (readsAt O B F ω') = powerE C O F τ κ seed Rmax d
          (readsAt O B F ω)
        ∧ powerSel C O F τ κ seed Rmax d (readsAt O B F ω')
          = powerSel C O F τ κ seed Rmax d (readsAt O B F ω) := fun ω ω' hn =>
    powerSel_mask C O F τ κ seed Rmax d (B := B) (f₁ := fun w => O.mq w ω)
      (f₂ := fun w => O.mq w ω') fun z hz => mq_congr O (hn z hz)
  have hBadeq : ∀ A (x : FreeMonoid α) ts, {ω | ∃ A' x' ts', some ((A, x), some (κ, ts))
      = some ((A', x'), some (κ, ts'))
        ∧ PowerCase O (testThreshold C A') τ ts' ∧ SidesApart ts'
        ∧ ω ∈ Misses O (testThreshold C A') ts'} ⊆ Misses O (testThreshold C A) ts := by
    rintro A x ts ω ⟨A', x', ts', he, -, -, hm⟩
    simp only [Option.some.injEq, Prod.mk.injEq, true_and] at he
    obtain ⟨⟨rfl, rfl⟩, rfl⟩ := he
    exact hm
  have hBadm : ∀ s, MeasurableSet[noiseAlg O ↑(((choiceTs s).map Prod.fst).toFinset)]
      {ω | ∃ A x ts, s = some ((A, x), some (κ, ts)) ∧ PowerCase O (testThreshold C A) τ ts
        ∧ SidesApart ts ∧ ω ∈ Misses O (testThreshold C A) ts} := by
    intro s
    by_cases hg : ∃ A x ts, s = some ((A, x), some (κ, ts))
        ∧ PowerCase O (testThreshold C A) τ ts ∧ SidesApart ts
    · obtain ⟨A, x, ts, rfl, hpc, hsa⟩ := hg
      have : {ω | ∃ (A' : RoundAcc α) (x' : FreeMonoid α) (ts' : List (FreeMonoid α × Bool)),
          some ((A, x), some (κ, ts)) = some ((A', x'), some (κ, ts'))
          ∧ PowerCase O (testThreshold C A') τ ts' ∧ SidesApart ts'
          ∧ ω ∈ Misses O (testThreshold C A') ts'} = Misses O (testThreshold C A) ts :=
        Set.Subset.antisymm (hBadeq A x ts) fun ω hm => ⟨A, x, ts, rfl, hpc, hsa, hm⟩
      rw [this]
      exact misses_measurable O _ ts
    · have : {ω | ∃ A x ts, s = some ((A, x), some (κ, ts))
          ∧ PowerCase O (testThreshold C A) τ ts ∧ SidesApart ts
          ∧ ω ∈ Misses O (testThreshold C A) ts} = ∅ := by
        ext ω
        simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨A, x, ts, h1, h2, h3, -⟩
        exact hg ⟨A, x, ts, h1, h2, h3⟩
      rw [this]
      exact @MeasurableSet.empty _ (noiseAlg O _)
  have key := fresh_select_le' O (fun ω => powerE C O F τ κ seed Rmax d (readsAt O B F ω))
    (fun ω => powerSel C O F τ κ seed Rmax d (readsAt O B F ω)) hdet
    (fun s => ((choiceTs s).map Prod.fst).toFinset)
    (fun ω => powerSel_disjoint C O F τ κ seed Rmax d B fun w => O.mq w ω)
    (fun s => ∃ A x ts, s = some ((A, x), some (κ, ts))
      ∧ PowerCase O (testThreshold C A) τ ts ∧ SidesApart ts)
    (fun s => {ω | ∃ A x ts, s = some ((A, x), some (κ, ts))
      ∧ PowerCase O (testThreshold C A) τ ts ∧ SidesApart ts
      ∧ ω ∈ Misses O (testThreshold C A) ts}) hBadm
    (fun s hg => by
      ext ω
      simp only [Set.mem_ofPred_eq, Set.mem_empty_iff_false, iff_false]
      rintro ⟨A, x, ts, h1, h2, h3, -⟩
      exact hg ⟨A, x, ts, h1, h2, h3⟩)
    (φ := ENNReal.ofReal (Real.exp (-τ ^ 2))) (fun s hg => by
      obtain ⟨A, x, ts, rfl, hpc, hsa⟩ := hg
      refine (measure_mono (hBadeq A x ts)).trans ?_
      rw [← ofReal_measureReal]
      exact ENNReal.ofReal_le_ofReal (misses_le O hsa hτ hpc))
  -- the event lies in the chosen test's miss, off a null set
  have hsub : {ω | ∃ A x, roundFind C (readsAt O B F ω) (Decisive C O F τ κ (readsAt O B F ω))
          Rmax 0 (startAcc C (readsAt O B F ω) seed) [] d = some (A, x)
        ∧ CaseAt C O F τ κ (readsAt O B F ω) A x
        ∧ verdict C.K (readsAt O B F ω) A.s.tree A.s.pool κ.1 κ.2
            (A.s.tree.paths.length * Fintype.card α)
            (stepSkip C.K (readsAt O B F ω) C.k A.s x κ) ≠ .split}
      ⊆ {ω | ∃ (A : RoundAcc α) (x : FreeMonoid α) (ts : List (FreeMonoid α × Bool)),
          powerSel C O F τ κ seed Rmax d (readsAt O B F ω) = some ((A, x), some (κ, ts))
          ∧ PowerCase O (testThreshold C A) τ ts ∧ SidesApart ts
          ∧ ω ∈ Misses O (testThreshold C A) ts} ∪ (cleanAll O)ᶜ := by
    rintro ω ⟨A, x, hr, hc, hv⟩
    by_cases hcl : ω ∈ cleanAll O
    swap
    · exact .inr hcl
    left
    have hr' := roundFind_force_some C _ O F τ κ h0 Rmax 0 _ [] d A x hr hc
    obtain ⟨ts, hst, -, hpc⟩ := hc
    have hsa : SidesApart ts := by rw [stepTest_ts C hst]; exact testStrings_nodup _ _
    refine ⟨A, x, ts, ?_, hpc, hsa, ?_⟩
    · unfold powerSel
      rw [hr']
      simp only [Option.map_some, hst]
    · by_contra hm
      exact hv (verdict_split (mq_clean O hcl) (stepTest_ts C hst) hpc.1 hpc.2.1
        (not_lt.1 hm))
  have hgood : {ω | ∃ (A : RoundAcc α) (x : FreeMonoid α) (ts : List (FreeMonoid α × Bool)),
        powerSel C O F τ κ seed Rmax d (readsAt O B F ω) = some ((A, x), some (κ, ts))
        ∧ PowerCase O (testThreshold C A) τ ts ∧ SidesApart ts}
      ⊆ {ω | roundFind C (readsAt O B F ω) (Decisive C O F τ κ (readsAt O B F ω)) Rmax 0
          (startAcc C (readsAt O B F ω) seed) [] d ≠ none} := by
    intro ω hg hn
    obtain ⟨A, x, ts, hs, -⟩ := hg
    have := roundFind_force_none C _ O F τ κ h0 Rmax 0 _ [] d hn
    unfold powerSel at hs
    rw [this] at hs
    exact absurd hs (by simp)
  refine (measure_mono hsub).trans ((measure_union_le _ _).trans ?_)
  rw [measure_cleanAll_compl O, add_zero]
  exact key.trans (by gcongr)

end Power

end OrthoDFA
