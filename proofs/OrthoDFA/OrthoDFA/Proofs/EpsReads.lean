import OrthoDFA.Proofs.EpsSteps
import OrthoDFA.Proofs.AdaptiveReads

/-!
# Level 4's wrong reads

Given the probes, the round's reads come in rounds (`rounds`): round `2j` is what the `j`-th
probe's step can read, its prefixes followed by midfixes, and round `2j + 1` what its split
re-sifts. While nothing read so far is wrong, the steps keep `WitInv`, so each round is fixed by
what the earlier ones read and has at most `(L + 1)|Q|`, or `(|Q| + 1)²|Σ|`, strings, and
`AdaptiveReadsBound` bounds the chance some step reads a string wrong. Over the probes, a sum
over their countably many values, the bound stays.
-/

namespace OrthoDFA

namespace Random

namespace Eps

open MeasureTheory

variable {α : Type*} [Fintype α] [DecidableEq α] {σ : Type*} [Fintype σ] (M : DFA α σ)
  (side : σ → Bool) (C : Level3Settings)

/-- The state before the `j`-th probe of `xs`, from `s`, while the round runs. -/
noncomputable def stateAt (read : FreeMonoid α → ARU) :
    Level3State α → List (FreeMonoid α) → ℕ → Option (Level3State α)
  | s, _, 0 => some s
  | _, [], _ + 1 => none
  | s, x :: xs, j + 1 =>
    match applyProbe read C s x with
    | .inl s' => stateAt read s' xs j
    | .inr _ => none

omit [Fintype σ] in
theorem stateAt_succ (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s : Level3State α) (j : ℕ),
      stateAt C read s xs (j + 1) = (stateAt C read s xs j).bind fun s' =>
        xs[j]?.bind fun x => (applyProbe read C s' x).elim some fun _ => none
  | [], s, j => by cases j <;> simp [stateAt]
  | x :: xs, s, 0 => by
    simp only [stateAt, Option.bind_some, List.getElem?_cons_zero]
    rcases applyProbe read C s x with s' | e <;> rfl
  | x :: xs, s, j + 1 => by
    simp only [stateAt, List.getElem?_cons_succ]
    rcases applyProbe read C s x with s' | e
    · exact stateAt_succ read xs s' j
    · rfl

/-- Round `2j` reads the `j`-th probe's prefixes followed by midfixes, round `2j + 1` what its
split re-sifts. -/
noncomputable def rounds (xs : List (FreeMonoid α)) (i : ℕ) (read : FreeMonoid α → ARU) :
    Finset (FreeMonoid α) :=
  match stateAt C read initialState xs (i / 2), xs[i / 2]? with
  | some s, some x => if i % 2 = 0 then pot 0 x s.tree else splitReads read C s x
  | _, _ => ∅

omit [Fintype σ] in
theorem rounds_even (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) {j : ℕ}
    {s : Level3State α} {x : FreeMonoid α} (hs : stateAt C read initialState xs j = some s)
    (hx : xs[j]? = some x) : rounds C xs (2 * j) read = pot 0 x s.tree := by
  have h1 : 2 * j / 2 = j := by omega
  simp only [rounds, h1, hs, hx]
  simp

omit [Fintype σ] in
theorem rounds_odd (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) {j : ℕ}
    {s : Level3State α} {x : FreeMonoid α} (hs : stateAt C read initialState xs j = some s)
    (hx : xs[j]? = some x) : rounds C xs (2 * j + 1) read = splitReads read C s x := by
  have h1 : (2 * j + 1) / 2 = j := by omega
  simp only [rounds, h1, hs, hx]
  simp

/-- No string the rounds before `n` read is read wrong. -/
def Good (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) (n : ℕ) : Prop :=
  ∀ i < n, ∀ z ∈ rounds C xs i read, read z ≠ if side (M.eval z.toList) then .reject else .accept

/-- While nothing read is wrong, the steps keep the invariants. -/
theorem inv_of_good (xs : List (FreeMonoid α)) (read : FreeMonoid α → ARU) :
    ∀ (j : ℕ) (s : Level3State α), stateAt C read initialState xs j = some s →
      Good M side C xs read (2 * j) → Inv read s ∧ WitInv M side s
  | 0, s, h, _ => by
    simp only [stateAt, Option.some.injEq] at h
    subst h
    exact ⟨inv_start read, witInv_start M side⟩
  | j + 1, s, h, hg => by
    rw [stateAt_succ] at h
    rcases hs₁ : stateAt C read initialState xs j with _ | s₁ <;> rw [hs₁] at h
    · simp at h
    rcases hx : xs[j]? with _ | x <;> simp only [hx, Option.bind_some, Option.bind_none] at h
    · simp at h
    rcases ha : applyProbe read C s₁ x with s' | e <;> rw [ha] at h <;>
      simp only [Sum.elim_inl, Sum.elim_inr, Option.some.injEq, reduceCtorEq] at h
    subst h
    obtain ⟨hI, hW⟩ := inv_of_good xs read j s₁ hs₁ fun i hi => hg i (by omega)
    have hx' : ¬ WrongStep M side read C s₁ x := by
      rintro ⟨z, hz, hw⟩
      rcases Finset.mem_union.1 hz with hz | hz
      · exact hg (2 * j) (by omega) z (by rw [rounds_even C xs read hs₁ hx]; exact hz) hw
      · exact hg (2 * j + 1) (by omega) z (by rw [rounds_odd C xs read hs₁ hx]; exact hz) hw
    by_cases hsm : s'.tree = s₁.tree ∧ s'.edges = s₁.edges
    · exact ⟨(step_same read C hI ha hsm.1 hsm.2).1, witInv_congr M side hW hsm.1 hsm.2⟩
    · obtain ⟨h1, h2, -⟩ := eps_step_change M side read C hI hW hx' ha hsm
      exact ⟨h1, h2⟩

/-- Reads agreeing with good ones on what the rounds before `2j` read reach the same state. -/
theorem stateAt_congr (xs : List (FreeMonoid α)) {r r' : FreeMonoid α → ARU} :
    ∀ j, Good M side C xs r (2 * j) → (∀ i < 2 * j, ∀ z ∈ rounds C xs i r, r z = r' z) →
      stateAt C r initialState xs j = stateAt C r' initialState xs j
  | 0, _, _ => by simp only [stateAt]
  | j + 1, hg, ha => by
    rw [stateAt_succ, stateAt_succ, ← stateAt_congr xs j (fun i hi => hg i (by omega))
      fun i hi => ha i (by omega)]
    rcases hs₁ : stateAt C r initialState xs j with _ | s₁
    · rfl
    rcases hx : xs[j]? with _ | x
    · rfl
    simp only [Option.bind_some]
    have hW := (inv_of_good M side C xs r j s₁ hs₁ fun i hi => hg i (by omega)).2
    rw [applyProbe_congr C (fun e t w h => edge_leaf M side hW h) fun z hz => ?_]
    rcases Finset.mem_union.1 hz with hz | hz
    · exact ha (2 * j) (by omega) z (by rw [rounds_even C xs r hs₁ hx]; exact hz)
    · exact ha (2 * j + 1) (by omega) z (by rw [rounds_odd C xs r hs₁ hx]; exact hz)

/-- While nothing read is wrong, each round is fixed by what the earlier ones read. -/
theorem rounds_local (xs : List (FreeMonoid α)) (i : ℕ) (r r' : FreeMonoid α → ARU)
    (hg : Good M side C xs r i) (ha : ∀ l < i, ∀ z ∈ rounds C xs l r, r z = r' z) :
    rounds C xs i r = rounds C xs i r' := by
  have hj : 2 * (i / 2) ≤ i := Nat.mul_div_le i 2
  have hst := stateAt_congr M side C xs (i / 2) (fun l hl => hg l (by omega))
    fun l hl => ha l (by omega)
  unfold rounds
  rw [← hst]
  rcases hs : stateAt C r initialState xs (i / 2) with _ | s
  · rfl
  rcases hx : xs[i / 2]? with _ | x
  · rfl
  simp only
  split_ifs with hi
  · rfl
  refine splitReads_congr C fun z hz => ha (2 * (i / 2)) (by omega) z ?_
  rw [rounds_even C xs r hs hx]
  exact hz

omit [Fintype σ] [DecidableEq α] in
theorem resifts_card (s : Level3State α) (T' : DTree α) :
    (resifts s T').card ≤ s.tree.leaves.length * Fintype.card α * (mids T').card := by
  classical
  unfold resifts
  refine Finset.card_biUnion_le.trans ?_
  calc _ ≤ ∑ _e ∈ s.tree.leaves.toFinset ×ˢ (Finset.univ : Finset α), (mids T').card :=
        Finset.sum_le_sum fun e _ => by
          split
          · exact Finset.card_image_le
          · simp
    _ ≤ _ := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_product, Finset.card_univ]
        gcongr
        exact List.toFinset_card_le _

/-- While nothing read is wrong, a probe's round has at most `(L + 1)|Q|` strings and a split's
`(|Q| + 1)²|Σ|`. -/
theorem rounds_card (xs : List (FreeMonoid α)) (L : ℕ) (hL : ∀ x ∈ xs, x.toList.length ≤ L)
    (i : ℕ) (r : FreeMonoid α → ARU) (hg : Good M side C xs r i) :
    (rounds C xs i r).card ≤ if i % 2 = 0 then (L + 1) * Fintype.card σ
      else (Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1) := by
  have hj : 2 * (i / 2) ≤ i := Nat.mul_div_le i 2
  unfold rounds
  rcases hs : stateAt C r initialState xs (i / 2) with _ | s
  · simp
  rcases hx : xs[i / 2]? with _ | x
  · simp
  simp only
  have hW := (inv_of_good M side C xs r (i / 2) s hs fun l hl => hg l (by omega)).2
  have hlv : s.tree.leaves.length ≤ Fintype.card σ + 1 := by
    have := hW.leaves; have := img_card_le M side s.tree; omega
  have hmid : (mids s.tree).card ≤ Fintype.card σ := by
    have := mids_card s.tree; omega
  have hxL : x.toList.length ≤ L := hL x (List.mem_of_getElem? hx)
  split_ifs with hi
  · calc _ ≤ (x.toList.length + 1) * (mids s.tree).card := card_pot 0 x _
      _ ≤ _ := Nat.mul_le_mul (by omega) hmid
  · unfold splitReads
    rcases hpo : probeOrAgree r s.tree s.edges C.walkStart x with _ | ⟨p, c, u, t⟩ | _ | _ | _
    · simp
    · simp only
      split
      · rename_i t₀ w₀ _
        refine (resifts_card s _).trans ?_
        have hm := Finset.card_le_card
          (mids_splitAt (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀)) s.tree p)
        have := Finset.card_insert_le (FreeMonoid.of c * s.tree.midfixAt (commonPrefix t t₀))
          (mids s.tree)
        gcongr
        omega
      · simp
    · simp
    · simp
    · simp

omit [Fintype σ] in
/-- A run with a step reading a string wrong reads it in one of the rounds. -/
theorem anyB_rounds (read : FreeMonoid α → ARU) :
    ∀ (xs : List (FreeMonoid α)) (s₀ : Level3State α),
      AnyB (applyProbe read C) (WrongStep M side read C) s₀ xs →
      ∃ j < xs.length, ∃ s x, stateAt C read s₀ xs j = some s ∧ xs[j]? = some x
        ∧ WrongStep M side read C s x
  | [], _, h => by simp [AnyB] at h
  | x :: xs, s₀, h => by
    rcases h with h | ⟨s', hs', h⟩
    · exact ⟨0, by simp, s₀, x, rfl, rfl, h⟩
    · obtain ⟨j, hj, s, y, h1, h2, h3⟩ := anyB_rounds read xs s' h
      refine ⟨j + 1, by simp; omega, s, y, ?_, by simpa using h2, h3⟩
      simp only [stateAt, hs']
      exact h1

theorem sum_alt (n a b : ℕ) :
    ∑ i ∈ Finset.range (2 * n), ((if i % 2 = 0 then a else b : ℕ) : ℝ) = n * (a + b) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, Finset.sum_range_succ, Finset.sum_range_succ,
      ih]
    have h1 : (2 * n + 1) % 2 = 1 := by omega
    have h0 : 2 * n % 2 = 0 := by omega
    simp only [h0, h1, if_true, one_ne_zero, if_false]
    push_cast
    ring

instance : MeasurableSingletonClass ARU := ⟨fun _ => trivial⟩

/-- Given probes no longer than `L`, some step reads a string wrong with chance at most
`εw · |xs| · ((L + 1)|Q| + (|Q| + 1)²|Σ|)`. -/
theorem wrong_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (U : σ → ℝ) (εw : ℝ) (R : EpsReadOracle μ M side U εw) (xs : List (FreeMonoid α)) (L : ℕ)
    (hL : ∀ x ∈ xs, x.toList.length ≤ L) :
    μ {ω | AnyB (applyProbe (R.read · ω) C) (WrongStep M side (R.read · ω) C) initialState xs}
      ≤ ENNReal.ofReal (εw * (xs.length
        * ((L + 1) * Fintype.card σ + (Fintype.card σ + 1) * Fintype.card α
          * (Fintype.card σ + 1)))) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  set B : FreeMonoid α → Set ARU := fun z => {if side (M.eval z.toList) then .reject else .accept}
  have hmain := adaptiveReadsBound_holds μ R.read B εw (fun i r => rounds C xs i r)
    (fun i => if i % 2 = 0 then (L + 1) * Fintype.card σ
      else (Fintype.card σ + 1) * Fintype.card α * (Fintype.card σ + 1)) (2 * xs.length)
    R.measurable R.indep (fun z => R.wrong z)
    (fun i r r' hg ha => rounds_local M side C xs i r r' (fun l hl z hz => hg l hl z hz) ha)
    (fun i r hg => rounds_card M side C xs L hL i r fun l hl z hz => hg l hl z hz)
  rw [sum_alt] at hmain
  refine (measure_mono fun ω hω => ?_).trans (hmain.trans (le_of_eq ?_))
  · obtain ⟨j, hj, s, x, h1, h2, z, hz, hw⟩ := anyB_rounds M side C (R.read · ω) xs _ hω
    rcases Finset.mem_union.1 hz with hz | hz
    · exact ⟨2 * j, by omega, z, by rw [rounds_even C xs _ h1 h2]; exact hz, hw⟩
    · exact ⟨2 * j + 1, by omega, z, by rw [rounds_odd C xs _ h1 h2]; exact hz, hw⟩
  · push_cast
    ring_nf

/-- A measure on a countable type is the sum of its points'. -/
theorem measure_eq_tsum {X : Type*} [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X]
    (ν : Measure X) (A : Set X) : ν A = ∑' d, ν {d} * A.indicator 1 d := by
  have hA : MeasurableSet A := (Set.to_countable A).measurableSet
  conv_lhs => rw [← Measure.sum_smul_dirac ν]
  rw [Measure.sum_apply _ hA]
  congr 1
  funext d
  rw [Measure.smul_apply, Measure.dirac_apply' _ hA, smul_eq_mul]

/-- Over fresh draws `d`, each drawn point bounding the chance of `{ω | d ∈ S ω}` by `c`, the mean
mass of `S` is at most `c`. -/
theorem lintegral_le_of_points {Ω X : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
    [MeasurableSpace X] [Countable X] [MeasurableSingletonClass X] (ν : Measure X)
    [IsProbabilityMeasure ν] (S : Ω → Set X) (c : ENNReal)
    (h : ∀ d, ν {d} ≠ 0 → μ {ω | d ∈ S ω} ≤ c) :
    ∫⁻ ω, ν (S ω) ∂μ ≤ c := by
  set E : X → Set Ω := fun d => toMeasurable μ {ω | d ∈ S ω}
  have hpt : ∀ ω, ν (S ω) ≤ ∑' d, ν {d} * (E d).indicator 1 ω := fun ω => by
    rw [measure_eq_tsum ν]
    refine ENNReal.tsum_le_tsum fun d => ?_
    gcongr
    by_cases hd : d ∈ S ω
    · rw [Set.indicator_of_mem hd,
        Set.indicator_of_mem (show ω ∈ E d from subset_toMeasurable _ _ hd)]
      simp
    · rw [Set.indicator_of_notMem hd]
      exact bot_le
  calc _ ≤ ∫⁻ ω, ∑' d, ν {d} * (E d).indicator 1 ω ∂μ := lintegral_mono hpt
    _ = ∑' d, ν {d} * μ (E d) := by
        rw [lintegral_tsum fun d => ((measurable_one.indicator
          (measurableSet_toMeasurable _ _)).const_mul _).aemeasurable]
        congr 1
        funext d
        rw [lintegral_const_mul _ (measurable_one.indicator (measurableSet_toMeasurable _ _)),
          lintegral_indicator_one (measurableSet_toMeasurable _ _)]
    _ ≤ ∑' d, ν {d} * c := ENNReal.tsum_le_tsum fun d => by
        by_cases hd : ν {d} = 0
        · rw [hd, zero_mul, zero_mul]
        · rw [measure_toMeasurable]
          gcongr
          exact h d hd
    _ = c := by
        have h1 : ∑' d, ν {d} = 1 := by
          rw [← measure_univ (μ := ν), measure_eq_tsum ν Set.univ]
          simp
        rw [ENNReal.tsum_mul_right, h1, one_mul]

/-- Over the probes and the reads, some step reads a string wrong with chance at most
`εw · probeBudget · ((L + 1)|Q| + (|Q| + 1)²|Σ|)`. -/
theorem pi_wrong_le {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (U : σ → ℝ) (εw : ℝ) (R : EpsReadOracle μ M side U εw) (P : PrefixDistribution α) :
    ∫⁻ ω, (Measure.pi fun _ : Fin C.probeBudget => P.D)
        {d | AnyB (applyProbe (R.read · ω) C) (WrongStep M side (R.read · ω) C) initialState
          (List.ofFn d)} ∂μ
      ≤ ENNReal.ofReal (εw * (C.probeBudget
        * ((P.L + 1) * Fintype.card σ + (Fintype.card σ + 1) * Fintype.card α
          * (Fintype.card σ + 1)))) := by
  have : Countable (FreeMonoid α) := (inferInstance : Countable (List α))
  refine lintegral_le_of_points μ _ _ _ fun d hd => ?_
  have hL : ∀ x ∈ List.ofFn d, x.toList.length ≤ P.L := by
    intro x hx
    obtain ⟨i, rfl⟩ := List.mem_ofFn.1 hx
    by_contra hlt
    apply hd
    have hpi : (Measure.pi fun _ : Fin C.probeBudget => P.D) {d}
        = ∏ j, P.D {d j} := by
      rw [← Measure.pi_pi]
      congr 1
      ext d'
      simp [funext_iff]
    rw [hpi]
    refine Finset.prod_eq_zero (Finset.mem_univ i) (measure_mono_null ?_
      (ae_iff.1 P.length_le))
    intro x hx
    rw [Set.mem_singleton_iff] at hx
    subst hx
    exact hlt
  have := wrong_le M side C μ U εw R (List.ofFn d) P.L hL
  rw [List.length_ofFn] at this
  exact this

end Eps

end Random

end OrthoDFA
