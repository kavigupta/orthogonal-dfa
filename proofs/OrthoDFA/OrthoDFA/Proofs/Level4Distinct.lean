import OrthoDFA.Proofs.Level4RiskBound

/-!
# A harvest's size, and its strings distinct

A test fires only past its first look or its excess, so a harvest holds at least as many strings
as those allow (`harvest_length_ge`). Every string a probe harvests extends the probe's length-`k`
prefix, and a probe harvests no string twice, so where no two probes share that prefix the
harvest has no repeats (`harvest_nodup`); two of `T` probes share it with chance at most
`T² prefixMax` (`prefix_collision_le`).
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {α : Type*} [Fintype α] [DecidableEq α]

theorem half_lt_of_binomSfGe_lt {t h : ℕ} {p a : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (ha0 : 0 < a)
    (ha : a ≤ 1 / 2) (ht : 32 * Real.log (1 / a) ≤ t * p ^ 2) (hlt : binomSfGe t p h < a) :
    (t : ℝ) * p / 2 < h := by
  have hlg : Real.log 2 ≤ Real.log (1 / a) :=
    Real.log_le_log (by norm_num) (by rw [le_div_iff₀ ha0]; linarith)
  have hl2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  by_contra hc
  push Not at hc
  cases h with
  | zero => rw [binomSfGe_zero_right] at hlt; linarith
  | succ j =>
    have hj := one_sub_binomSfGe_le t j p (p / 2) hp0 hp1 (by linarith) (by
      push_cast at hc; nlinarith)
    have : Real.exp (-2 * (t : ℝ) * (p / 2) ^ 2) ≤ 1 / 2 := by
      rw [← Real.exp_log (show (0 : ℝ) < 1 / 2 by norm_num), one_div, Real.log_inv]
      exact Real.exp_le_exp.2 (by nlinarith)
    linarith

section Size

variable (C : Level4Settings) (cut : FreeMonoid α → Option Bool)

theorem runProbes_harvest {s : Level4State α} {xs zs : List (FreeMonoid α)}
    (h : (runProbes C cut s xs).2 = .harvest zs) :
    ∃ s₀ x e, e ≠ .tooBig ∧ runTests C (recordProbe C cut s₀ x) = some e
      ∧ toRoundEnd e (recordProbe C cut s₀ x) = .harvest zs := by
  induction xs generalizing s with
  | nil => simp [runProbes] at h
  | cons x xs ih =>
    simp only [runProbes] at h
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at h
    · exact ih h
    · have he : e ≠ .tooBig := by rintro rfl; simp [toRoundEnd] at h
      obtain ⟨rfl, hr⟩ := applyProbe_inr_pre C cut ha he
      exact ⟨s, x, e, he, hr, h⟩

/-- A harvest holds at least `H` strings where each test's first look or excess asks for that
many. -/
theorem harvest_length_ge (H : ℕ) (ha0 : 0 < C.testLevel) (ha : C.testLevel ≤ 1 / 2)
    (hs0 : 0 ≤ C.startUndecidedRate) (hs1 : C.startUndecidedRate ≤ 1)
    (hm0 : 0 ≤ C.middleUndecidedRate) (hm1 : C.middleUndecidedRate ≤ 1)
    (hsl : 32 * Real.log (1 / C.testLevel) ≤ C.firstLook * C.startUndecidedRate ^ 2)
    (hml : 32 * Real.log (1 / C.testLevel) ≤ C.firstLook * C.middleUndecidedRate ^ 2)
    (hsH : (H : ℝ) ≤ C.firstLook * C.startUndecidedRate / 2)
    (hmH : (H : ℝ) ≤ C.firstLook * C.middleUndecidedRate / 2)
    (heH : ∀ j, (H : ℝ) ≤ C.edgeExcess j) (he0 : 0 ≤ C.edgeUndecidedRate)
    (read : FreeMonoid α → ARU) {T : ℕ} (draws : Fin T → FreeMonoid α)
    {zs : List (FreeMonoid α)} (h : (indecisiveLStarLevel4 C read draws).ending = .harvest zs) :
    H ≤ zs.length := by
  obtain ⟨s₀, x, e, -, hr, hz⟩ := runProbes_harvest C (fun z => (read z).cut)
    (s := initialState) (xs := List.ofFn draws) h
  have hle : ∀ (t n : ℕ) (p : ℝ), 0 ≤ p → p ≤ 1
      → 32 * Real.log (1 / C.testLevel) ≤ C.firstLook * p ^ 2
      → (H : ℝ) ≤ C.firstLook * p / 2 → C.firstLook ≤ t → binomSfGe t p n < C.testLevel
      → H ≤ n := by
    intro t n p hp0 hp1 hl hH ht hb
    have ht' : (C.firstLook : ℝ) ≤ t := by exact_mod_cast ht
    have h1 := half_lt_of_binomSfGe_lt hp0 hp1 ha0 ha
      (hl.trans (mul_le_mul_of_nonneg_right ht' (sq_nonneg p))) hb
    have h2 : (C.firstLook : ℝ) * p / 2 ≤ t * p / 2 := by
      have := mul_le_mul_of_nonneg_right ht' hp0
      linarith
    have : (H : ℝ) < n := by linarith
    exact_mod_cast this.le
  cases e with
  | success => simp [toRoundEnd] at hz
  | tooBig => simp [toRoundEnd] at hz
  | harvestStart =>
    simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
    obtain ⟨h1, h2⟩ := runTests_start' C hr
    rw [← hz]
    exact hle _ _ _ hs0 hs1 hsl hsH h1 h2
  | harvestPT =>
    simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
    obtain ⟨-, h1, h2⟩ := runTests_pt' C hr
    rw [← hz]
    exact hle _ _ _ hm0 hm1 hml hmH h1 h2
  | harvest e =>
    simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
    obtain ⟨-, -, h3⟩ := runTests_harvest' C hr
    have h4 := heH ((recordProbe C (fun z => (read z).cut) s₀ x).edgeReads e.1 e.2)
    have h5 : 0 ≤ C.edgeUndecidedRate
        * ((recordProbe C (fun z => (read z).cut) s₀ x).edgeReads e.1 e.2 : ℝ) := by positivity
    have : (H : ℝ) ≤ zs.length := by rw [← hz]; linarith
    exact_mod_cast this

end Size

section Distinct

/-- No repeats, and no string's length-`k` prefix is one of `xs`'. -/
def Fresh (k : ℕ) (xs ℓ : List (FreeMonoid α)) : Prop :=
  ℓ.Nodup ∧ ∀ z ∈ ℓ, prefixOf z k ∉ xs.map (prefixOf · k)

omit [Fintype α] [DecidableEq α] in
theorem Fresh.nil {k : ℕ} {xs : List (FreeMonoid α)} : Fresh k xs [] :=
  ⟨List.nodup_nil, fun _ h => absurd h List.not_mem_nil⟩

omit [Fintype α] [DecidableEq α] in
theorem Fresh.append {k : ℕ} {x : FreeMonoid α} {xs ℓ new : List (FreeMonoid α)}
    (hℓ : Fresh k (x :: xs) ℓ) (hnew : new.Nodup ∧ ∀ z ∈ new, prefixOf z k = prefixOf x k)
    (hx : prefixOf x k ∉ xs.map (prefixOf · k)) : Fresh k xs (ℓ ++ new) := by
  refine ⟨List.nodup_append.2 ⟨hℓ.1, hnew.1, ?_⟩, ?_⟩
  · intro a ha b hb hab
    subst hab
    have := hℓ.2 a ha
    rw [hnew.2 a hb, List.map_cons] at this
    exact this List.mem_cons_self
  · intro z hz
    rcases List.mem_append.1 hz with hz | hz
    · intro h
      have := hℓ.2 z hz
      rw [List.map_cons] at this
      exact this (List.mem_cons_of_mem _ h)
    · rw [hnew.2 z hz]; exact hx

variable (cut : FreeMonoid α → Option Bool) (T : DTree α) (E : EdgeTable α) (k : ℕ)
  (x : FreeMonoid α)

theorem startHarvestBy_fresh : (startHarvestBy cut T k x).Nodup
    ∧ ∀ z ∈ startHarvestBy cut T k x, prefixOf z k = prefixOf x k := by
  unfold startHarvestBy
  split_ifs with hkx
  · simp
  · push Not at hkx
    rcases hs : T.siftCut cut (prefixOf x k) with p | b
    · simp
    · simp only [Sum.elim_inr, List.nodup_singleton, List.mem_singleton, forall_eq, true_and]
      exact probeStrings_prefix (route_mem_probeStrings hkx le_rfl (route_undecided cut T _ b hs).1)

theorem edgeHarvestBy_fresh (e : List Bool × α) : (edgeHarvestBy cut T E k x e).Nodup
    ∧ ∀ z ∈ edgeHarvestBy cut T E k x e, prefixOf z k = prefixOf x k := by
  by_cases hkx : x.toList.length < k
  · simp [edgeHarvestBy_short cut T E hkx]
  push Not at hkx
  refine ⟨(List.Sublist.map Prod.snd List.filter_sublist).nodup (firstsBy_nodup cut T E k x), ?_⟩
  intro z hz
  unfold edgeHarvestBy at hz
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hz
  exact probeStrings_prefix (firstsBy_mem_probeStrings cut T E k x hkx (List.mem_of_mem_filter hq))

theorem middleHarvestBy_fresh : (middleHarvestBy cut T E k x).Nodup
    ∧ ∀ z ∈ middleHarvestBy cut T E k x, prefixOf z k = prefixOf x k := by
  by_cases hkx : x.toList.length < k
  · simp [middleHarvestBy_short cut T E hkx]
  push Not at hkx
  have key : ∀ j b, k ≤ j → T.siftCut cut (prefixOf x j) = .inr b →
      prefixOf b k = prefixOf x k := fun j b hj hb =>
    probeStrings_prefix (route_mem_probeStrings hkx hj (route_undecided cut T _ b hb).1)
  unfold middleHarvestBy
  rcases ho : probeBy cut T E k x with _ | _ | _ | j | _ | j | _
  all_goals first | (simp; done) | skip
  · obtain ⟨ps, hi, -, hb⟩ := probeBy_search cut ho trivial
    have hj := bracketAt_pair_gt _ _ _ _ _ _ hb
    dsimp only
    rcases hs : T.siftCut cut (prefixOf x j) with p | b
    · simp
    · simpa using key j b hj.le hs
  · obtain ⟨ps, hi, -, hb⟩ := probeBy_search cut ho trivial
    have hj := bracketAt_triple_gt _ _ _ _ _ _ hb
    dsimp only
    rcases hs : T.siftCut cut (prefixOf x j) with p | b
    · simp
    · simpa using key j b hj.le hs

/-- Every harvest list of `s` is `Fresh` against the probes still to come. -/
def HarvestFresh (k : ℕ) (xs : List (FreeMonoid α)) (s : Level4State α) : Prop :=
  Fresh k xs s.startUndecided ∧ (∀ p c, Fresh k xs (s.edgeUndecided p c))
    ∧ Fresh k xs s.stretchMiddles

variable (C : Level4Settings)

theorem recordProbe_fresh {s : Level4State α} {x : FreeMonoid α} {xs : List (FreeMonoid α)}
    (h : HarvestFresh C.walkStart (x :: xs) s)
    (hx : prefixOf x C.walkStart ∉ xs.map (prefixOf · C.walkStart)) :
    HarvestFresh C.walkStart xs (recordProbe C cut s x) := by
  obtain ⟨he, -, hs, -⟩ := recordProbe_countProbe C cut s x
  refine ⟨?_, fun p c => ?_, ?_⟩
  · rw [hs]; exact h.1.append (startHarvestBy_fresh cut s.tree _ x) hx
  · rw [he]; exact (h.2.1 p c).append (edgeHarvestBy_fresh cut s.tree s.edges _ x (p, c)) hx
  · rcases recordProbe_stretch C cut s x with ⟨-, hm, -⟩ | ⟨-, hm, -, -⟩
    · rw [hm]; exact h.2.2.append (middleHarvestBy_fresh cut s.tree s.edges _ x) hx
    · rw [hm]; exact Fresh.nil

theorem applyProbe_fresh {s s' : Level4State α} {x : FreeMonoid α} {xs : List (FreeMonoid α)}
    (h : applyProbe C cut s x = .inl s') (hr : HarvestFresh C.walkStart xs (recordProbe C cut s x)) :
    HarvestFresh C.walkStart xs s' := by
  obtain ⟨he, -, hs, -⟩ := applyProbe_counts C cut h
  refine ⟨?_, fun p c => ?_, ?_⟩
  · rw [hs]; exact hr.1
  · rw [he]; exact hr.2.1 p c
  · rcases applyProbe_newStretch C cut h with ⟨-, -, hm⟩ | ⟨rfl, -⟩
    · rw [hm]; exact Fresh.nil
    · exact hr.2.2

theorem runProbes_nodup {s : Level4State α} {xs zs : List (FreeMonoid α)}
    (h : HarvestFresh C.walkStart xs s) (hxs : (xs.map (prefixOf · C.walkStart)).Nodup)
    (hz : (runProbes C cut s xs).2 = .harvest zs) : zs.Nodup := by
  induction xs generalizing s with
  | nil => simp [runProbes] at hz
  | cons x xs ih =>
    rw [List.map_cons, List.nodup_cons] at hxs
    have hr := recordProbe_fresh cut C h hxs.1
    simp only [runProbes] at hz
    rcases ha : applyProbe C cut s x with s' | ⟨e, s'⟩ <;> rw [ha] at hz
    · exact ih (applyProbe_fresh cut C ha hr) hxs.2 hz
    · have he : e ≠ .tooBig := by rintro rfl; simp [toRoundEnd] at hz
      obtain ⟨rfl, -⟩ := applyProbe_inr_pre C cut ha he
      cases e with
      | success => simp [toRoundEnd] at hz
      | tooBig => simp [toRoundEnd] at hz
      | harvestStart =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hr.1.1
      | harvestPT =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact hr.2.2.1
      | harvest e =>
        simp only [toRoundEnd, RoundEnd.harvest.injEq] at hz
        rw [← hz]; exact (hr.2.1 e.1 e.2).1

/-- Where no two probes share their length-`k` prefix, a harvest has no repeats. -/
theorem harvest_nodup (read : FreeMonoid α → ARU) {T : ℕ} (draws : Fin T → FreeMonoid α)
    (hkeys : Function.Injective fun i => prefixOf (draws i) C.walkStart)
    {zs : List (FreeMonoid α)} (h : (indecisiveLStarLevel4 C read draws).ending = .harvest zs) :
    zs.Nodup := by
  refine runProbes_nodup (fun z => (read z).cut) C (s := initialState) (xs := List.ofFn draws)
    ⟨Fresh.nil, fun _ _ => Fresh.nil, Fresh.nil⟩ ?_ h
  rw [List.map_ofFn]
  exact List.nodup_ofFn.2 hkeys

end Distinct

/-- Two of `T` probes share their length-`k` prefix with chance at most `T² prefixMax`. -/
theorem prefix_collision_le (D : Measure (FreeMonoid α)) [IsProbabilityMeasure D] (k T : ℕ) :
    (Measure.pi fun _ : Fin T => D).real
        {xs | ¬ Function.Injective fun i => prefixOf (xs i) k}
      ≤ T ^ 2 * prefixMax D k := by
  classical
  set A : FreeMonoid α → Set (FreeMonoid α) := fun u => {x | prefixOf x k = u}
  have hbdd : BddAbove (Set.range fun u => D.real (A u)) :=
    ⟨1, by rintro _ ⟨u, rfl⟩; exact measureReal_le_one⟩
  have hq : ∀ u, D.real (A u) ≤ prefixMax D k := fun u => le_ciSup hbdd u
  have hq0 : 0 ≤ prefixMax D k := (measureReal_nonneg).trans (hq 1)
  set ν := Measure.pi fun _ : Fin T => D
  have hpair : ∀ i j : Fin T, i ≠ j →
      ν.real {xs | prefixOf (xs i) k = prefixOf (xs j) k} ≤ prefixMax D k := by
    intro i j hij
    refine ENNReal.toReal_le_of_le_ofReal hq0 ?_
    have hbox : ∀ u, {xs : Fin T → FreeMonoid α | xs i ∈ A u ∧ xs j ∈ A u}
        = Set.pi Set.univ fun l => if l ∈ ({i, j} : Finset (Fin T)) then A u else Set.univ := by
      intro u
      ext xs
      simp only [Set.mem_ofPred_eq, Set.mem_pi, Set.mem_univ, true_implies]
      constructor
      · rintro ⟨hi, hj⟩ l
        split_ifs with hl
        · rcases Finset.mem_insert.1 hl with rfl | hl
          · exact hi
          · rw [Finset.mem_singleton.1 hl]; exact hj
        · trivial
      · intro h
        have hi := h i
        have hj := h j
        rw [if_pos (by simp)] at hi hj
        exact ⟨hi, hj⟩
    have hmeas : ∀ u, ν {xs : Fin T → FreeMonoid α | xs i ∈ A u ∧ xs j ∈ A u} = D (A u) ^ 2 := by
      intro u
      rw [hbox u, Measure.pi_pi]
      simp only [apply_ite D, measure_univ]
      rw [Finset.prod_ite_mem, Finset.univ_inter, Finset.prod_const, Finset.card_pair hij]
    calc ν {xs | prefixOf (xs i) k = prefixOf (xs j) k}
        ≤ ν (⋃ u, {xs : Fin T → FreeMonoid α | xs i ∈ A u ∧ xs j ∈ A u}) := by
          refine measure_mono fun xs hxs => Set.mem_iUnion.2 ⟨prefixOf (xs i) k, ?_⟩
          exact ⟨rfl, hxs.symm⟩
      _ ≤ ∑' u, ν {xs : Fin T → FreeMonoid α | xs i ∈ A u ∧ xs j ∈ A u} := measure_iUnion_le _
      _ = ∑' u, D (A u) * D (A u) := by simp only [hmeas, sq]
      _ ≤ ∑' u, ENNReal.ofReal (prefixMax D k) * D (A u) := by
          refine ENNReal.tsum_le_tsum fun u => mul_le_mul_of_nonneg_right ?_ zero_le
          rw [← ofReal_measureReal]
          exact ENNReal.ofReal_le_ofReal (hq u)
      _ = ENNReal.ofReal (prefixMax D k) * D (⋃ u, A u) := by
          rw [ENNReal.tsum_mul_left, measure_iUnion]
          · intro u v huv
            refine Set.disjoint_left.2 fun x hu hv => huv ?_
            exact hu.symm.trans hv
          · exact fun _ => trivial
      _ ≤ ENNReal.ofReal (prefixMax D k) := by
          refine (mul_le_mul_of_nonneg_left prob_le_one zero_le).trans_eq (mul_one _)
  have hsub : {xs : Fin T → FreeMonoid α | ¬ Function.Injective fun i => prefixOf (xs i) k}
      ⊆ ⋃ p ∈ (Finset.univ.filter fun p : Fin T × Fin T => p.1 ≠ p.2),
        {xs | prefixOf (xs p.1) k = prefixOf (xs p.2) k} := by
    intro xs hxs
    simp only [Function.Injective, not_forall] at hxs
    obtain ⟨i, j, he, hne⟩ := hxs
    exact Set.mem_biUnion (x := (i, j)) (by simp [hne]) he
  calc ν.real _ ≤ ν.real (⋃ p ∈ (Finset.univ.filter fun p : Fin T × Fin T => p.1 ≠ p.2),
        {xs | prefixOf (xs p.1) k = prefixOf (xs p.2) k}) := measureReal_mono hsub (measure_ne_top _ _)
    _ ≤ ∑ p ∈ (Finset.univ.filter fun p : Fin T × Fin T => p.1 ≠ p.2),
        ν.real {xs | prefixOf (xs p.1) k = prefixOf (xs p.2) k} :=
        measureReal_biUnion_finset_le _ _
    _ ≤ ∑ _p ∈ (Finset.univ.filter fun p : Fin T × Fin T => p.1 ≠ p.2), prefixMax D k :=
        Finset.sum_le_sum fun p hp => hpair p.1 p.2 (Finset.mem_filter.1 hp).2
    _ ≤ T ^ 2 * prefixMax D k := by
        rw [Finset.sum_const, nsmul_eq_mul]
        refine mul_le_mul_of_nonneg_right ?_ hq0
        have : (Finset.univ.filter fun p : Fin T × Fin T => p.1 ≠ p.2).card ≤ T ^ 2 := by
          refine (Finset.card_filter_le _ _).trans ?_
          simp [sq]
        exact_mod_cast this

end OrthoDFA
