import OrthoDFA.Code.Level3DrawMany
import OrthoDFA.Proofs.HarvestCode
import OrthoDFA.Proofs.DrawFrom

/-! # Decoupling the `m`-draw sampler from the harvest law

The two facts that make `drawMany >>= clustering` a drop-in, with clustering oblivious to the
draws:

* `drawManyLaw_eq_pi` — (AGR) conditioned on all `m` draws hitting, the output law is exactly
  the `m`-fold product of `popLaw ((harvestLaw …).restrict Tᶜ)`.
* `drawManyMiss_le` — (FAIL) the chance any draw misses is `≤ δₛ'`, by a union bound over the `m`
  draws with each `harvestMiss ≤ δₛ'/m` at `N' = harvestN y (δₛ'/m)`. -/

namespace OrthoDFA

namespace Random

open MeasureTheory
open scoped ENNReal

variable {α : Type*} [Fintype α]
variable (read : FreeMonoid α → ARU) (D : Measure (FreeMonoid α)) (k : ℕ) (H : HypothesisDFA α)
  (site : Level3Site) (T : Set (FreeMonoid α)) {m N' : ℕ}

/-- The value of `harvestCodeLaw` on `A`: the mass of the `N` i.i.d. probes that both hit and land
their harvested string in `A`. -/
lemma harvestCodeLaw_apply' (N : ℕ) (A : Set (FreeMonoid α)) :
    harvestCodeLaw read D k H site T N A
      = (Measure.pi fun _ : Fin N => D)
          ((fun z => (harvestFirst read k H site T (List.ofFn z)).getD 1) ⁻¹' A
            ∩ {z | (harvestFirst read k H site T (List.ofFn z)).isSome = true}) := by
  rw [harvestCodeLaw,
      Measure.map_apply (measurable_of_countable _) (Set.to_countable A).measurableSet,
      Measure.restrict_apply (Set.to_countable _).measurableSet]

/-- `harvestCodeLaw` is a sub-probability: its total mass is the hit mass, at most `1`. -/
lemma harvestCodeLaw_univ_le_one [IsProbabilityMeasure D] (N : ℕ) :
    harvestCodeLaw read D k H site T N Set.univ ≤ 1 := by
  rw [harvestCodeLaw_apply']
  exact le_trans (measure_mono (Set.subset_univ _)) (le_of_eq measure_univ)

/-- A finite product of point masses is the point mass at the tuple. -/
lemma pi_dirac_one :
    Measure.pi (fun _ : Fin m => Measure.dirac (1 : FreeMonoid α))
      = Measure.dirac (1 : Fin m → FreeMonoid α) := by
  classical
  refine Measure.pi_eq fun s hs => ?_
  rw [Measure.dirac_apply' _ (MeasurableSet.univ_pi hs), Set.indicator_apply]
  simp only [Set.mem_univ_pi, Pi.one_apply]
  by_cases h : ∀ i, (1 : FreeMonoid α) ∈ s i
  · rw [if_pos h]
    refine (Finset.prod_eq_one fun i _ => ?_).symm
    rw [Measure.dirac_apply' _ (hs i), Set.indicator_apply, if_pos (h i), Pi.one_apply]
  · rw [if_neg h]
    push_neg at h
    obtain ⟨i, hi⟩ := h
    refine (Finset.prod_eq_zero (Finset.mem_univ i) ?_).symm
    rw [Measure.dirac_apply' _ (hs i), Set.indicator_apply, if_neg hi]

/-- Normalising the `m`-fold i.i.d. product of a finite measure `ν` is the product of the
normalisations: `popLaw` commutes with the product. -/
lemma norm_pi {ν : Measure (FreeMonoid α)} {P : Measure (Fin m → FreeMonoid α)}
    (hP : P = Measure.pi fun _ : Fin m => ν) (hν : ν Set.univ ≠ ∞) :
    (if P Set.univ = 0 ∨ P Set.univ = ∞ then Measure.dirac 1 else (P Set.univ)⁻¹ • P)
      = Measure.pi fun _ : Fin m => popLaw ν := by
  classical
  haveI : IsFiniteMeasure ν := ⟨lt_of_le_of_ne le_top hν⟩
  have hPuniv : P Set.univ = ν Set.univ ^ m := by
    rw [hP, Measure.pi_univ, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
  have hPtop : P Set.univ ≠ ∞ := by rw [hPuniv]; exact ENNReal.pow_ne_top hν
  by_cases hz : P Set.univ = 0
  · rw [if_pos (Or.inl hz)]
    have hm0 : m ≠ 0 := by
      rintro rfl; rw [hPuniv, pow_zero] at hz; exact one_ne_zero hz
    have hc0 : ν Set.univ = 0 := by
      rw [hPuniv] at hz; exact (pow_eq_zero_iff hm0).1 hz
    have hpop : (fun _ : Fin m => popLaw ν) = fun _ : Fin m => Measure.dirac (1 : FreeMonoid α) := by
      funext i; rw [popLaw, if_pos (Or.inl hc0)]
    rw [hpop, pi_dirac_one]
  · rw [if_neg (not_or.2 ⟨hz, hPtop⟩)]
    refine (Measure.pi_eq fun s hs => ?_).symm
    rw [Measure.smul_apply, smul_eq_mul, hP, Measure.pi_univ, Measure.pi_pi,
        Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    by_cases hc : ν Set.univ = 0
    · have hm0 : m = 0 := by
        by_contra hm'
        exact hz (by rw [hPuniv, hc, zero_pow hm'])
      subst hm0
      simp
    · have hpopi : ∀ i : Fin m, popLaw ν (s i) = (ν Set.univ)⁻¹ * ν (s i) := by
        intro i
        rw [popLaw, if_neg (not_or.2 ⟨hc, hν⟩), Measure.smul_apply, smul_eq_mul]
      rw [Finset.prod_congr rfl (fun i _ => hpopi i), Finset.prod_mul_distrib,
          Finset.prod_const, Finset.card_univ, Fintype.card_fin, ← ENNReal.inv_pow]

/-- The raw (unnormalised, all-hit) law of the `m`-draw sampler is exactly the `m`-fold product of
`harvestCodeLaw`: the all-hit event and the coordinatewise map both factor over `Fin m`. -/
lemma drawManyLaw_raw_eq [IsProbabilityMeasure D] (N' : ℕ) :
    (((Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N' => D).restrict
        {yss | (drawManyFirst read k H site T yss).isSome = true}).map
        fun yss => (drawManyFirst read k H site T yss).getD 1)
      = Measure.pi (fun _ : Fin m => harvestCodeLaw read D k H site T N') := by
  classical
  haveI : IsFiniteMeasure (harvestCodeLaw read D k H site T N') :=
    ⟨lt_of_le_of_lt (harvestCodeLaw_univ_le_one read D k H site T N') ENNReal.one_lt_top⟩
  refine (Measure.pi_eq fun s hs => ?_).symm
  rw [Measure.map_apply (measurable_of_countable _) (MeasurableSet.univ_pi hs),
      Measure.restrict_apply (Set.to_countable _).measurableSet]
  have hgset :
      ((fun yss : Fin m → (Fin N' → FreeMonoid α) => (drawManyFirst read k H site T yss).getD 1) ⁻¹'
          (Set.univ.pi s)) ∩ {yss | (drawManyFirst read k H site T yss).isSome = true}
        = Set.univ.pi (fun i => (fun z => (harvestFirst read k H site T (List.ofFn z)).getD 1) ⁻¹'
            (s i) ∩ {z | (harvestFirst read k H site T (List.ofFn z)).isSome = true}) := by
    ext yss
    simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_univ_pi, Set.mem_setOf_eq]
    constructor
    · rintro ⟨hg, hA⟩
      have hcond : ∀ i, (harvestFirst read k H site T (List.ofFn (yss i))).isSome = true := by
        revert hA; unfold drawManyFirst; split_ifs with hc <;> simp [hc]
      have hsome : drawManyFirst read k H site T yss
          = some (fun i => (harvestFirst read k H site T (List.ofFn (yss i))).getD 1) := by
        unfold drawManyFirst; rw [if_pos hcond]
      intro i
      refine ⟨?_, hcond i⟩
      have hgi := hg i
      rw [hsome, Option.getD_some] at hgi
      exact hgi
    · intro hmem
      have hcond : ∀ i, (harvestFirst read k H site T (List.ofFn (yss i))).isSome = true :=
        fun i => (hmem i).2
      have hsome : drawManyFirst read k H site T yss
          = some (fun i => (harvestFirst read k H site T (List.ofFn (yss i))).getD 1) := by
        unfold drawManyFirst; rw [if_pos hcond]
      refine ⟨fun i => ?_, by simp [hsome]⟩
      rw [hsome, Option.getD_some]
      exact (hmem i).1
  rw [hgset, Measure.pi_pi]
  exact Finset.prod_congr rfl (fun i _ => (harvestCodeLaw_apply' read D k H site T N' (s i)).symm)

/-- (AGR) Conditioned on all `m` draws hitting, the sampler's output law is the `m`-fold product
of the harvest law off `T`, for every cap `N' ≥ 1`. -/
theorem drawManyLaw_eq_pi [IsProbabilityMeasure D] (hN : 0 < N') :
    drawManyLaw read D k H site T N' m
      = Measure.pi (fun _ : Fin m => popLaw ((harvestLaw read D k H site).restrict Tᶜ)) := by
  classical
  have hpop : (fun _ : Fin m => popLaw ((harvestLaw read D k H site).restrict Tᶜ))
      = (fun _ : Fin m => popLaw (harvestCodeLaw read D k H site T N')) := by
    funext i; exact (harvestCodeLaw_popLaw_eq read D k H site T N' hN).symm
  rw [hpop]
  unfold drawManyLaw
  exact norm_pi (drawManyLaw_raw_eq read D k H site T N')
    (lt_of_le_of_lt (harvestCodeLaw_univ_le_one read D k H site T N') ENNReal.one_lt_top).ne

/-- (FAIL) With `y` at most the chance a probe harvests a string off `T`, and
`N' = harvestN y (δₛ'/m)`, the chance any of the `m` draws misses is at most `δₛ'`. -/
theorem drawManyMiss_le [IsProbabilityMeasure D] {y δₛ' : ℝ}
    (hy0 : 0 < y) (hy1 : y ≤ 1) (hδ : 0 < δₛ') (hm : 0 < m)
    (hyield : y ≤ (harvestLaw read D k H site).real Tᶜ) :
    drawManyMiss read D k H site T (harvestN y (δₛ' / m)) m ≤ ENNReal.ofReal δₛ' := by
  classical
  set N := harvestN y (δₛ' / m) with hNdef
  have hδm : 0 < δₛ' / m := div_pos hδ (by exact_mod_cast hm)
  have hcoord : ∀ i : Fin m,
      (Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N => D)
          {yss | harvestFirst read k H site T (List.ofFn (yss i)) = none}
        = harvestMiss read D k H site T N := by
    intro i
    rw [harvestMiss]
    have hpi : {yss : Fin m → (Fin N → FreeMonoid α) |
          harvestFirst read k H site T (List.ofFn (yss i)) = none}
        = Set.univ.pi (Function.update (fun _ : Fin m => (Set.univ : Set (Fin N → FreeMonoid α)))
            i {xs | harvestFirst read k H site T (List.ofFn xs) = none}) := by
      ext yss
      simp only [Set.mem_setOf_eq, Set.mem_univ_pi]
      constructor
      · intro h j
        rcases eq_or_ne j i with rfl | hj
        · rw [Function.update_self]; exact h
        · rw [Function.update_of_ne hj]; exact Set.mem_univ _
      · intro h; have := h i; rwa [Function.update_self] at this
    rw [hpi, Measure.pi_pi,
        Finset.prod_eq_single i
          (fun j _ hj => by rw [Function.update_of_ne hj]; exact measure_univ)
          (fun h => absurd (Finset.mem_univ i) h),
        Function.update_self]
  have hset : {yss : Fin m → (Fin N → FreeMonoid α) | drawManyFirst read k H site T yss = none}
      = ⋃ i : Fin m, {yss | harvestFirst read k H site T (List.ofFn (yss i)) = none} := by
    ext yss
    simp only [Set.mem_setOf_eq, Set.mem_iUnion]
    unfold drawManyFirst
    split_ifs with hc
    · constructor
      · intro h; exact absurd h (by simp)
      · rintro ⟨i, hi⟩; have hci := hc i; rw [hi] at hci; simp at hci
    · constructor
      · intro _
        push_neg at hc
        obtain ⟨i, hi⟩ := hc
        refine ⟨i, ?_⟩
        by_contra hne
        exact hi (Option.isSome_iff_ne_none.2 hne)
      · intro _; rfl
  rw [drawManyMiss, hset]
  calc (Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N => D)
          (⋃ i : Fin m, {yss | harvestFirst read k H site T (List.ofFn (yss i)) = none})
      ≤ ∑' i : Fin m, (Measure.pi fun _ : Fin m => Measure.pi fun _ : Fin N => D)
          {yss | harvestFirst read k H site T (List.ofFn (yss i)) = none} := measure_iUnion_le _
    _ = ∑ i : Fin m, harvestMiss read D k H site T N := by
        rw [tsum_fintype]; exact Finset.sum_congr rfl (fun i _ => hcoord i)
    _ ≤ ∑ _i : Fin m, ENNReal.ofReal (δₛ' / m) :=
        Finset.sum_le_sum fun i _ => by
          rw [hNdef]; exact harvestCodeLaw_miss_le read D k H site T hy0 hy1 hδm hyield
    _ = (m : ℝ≥0∞) * ENNReal.ofReal (δₛ' / m) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    _ = ENNReal.ofReal δₛ' := by
        rw [← ENNReal.ofReal_natCast m, ← ENNReal.ofReal_mul (Nat.cast_nonneg m)]
        congr 1
        field_simp

end Random

end OrthoDFA
