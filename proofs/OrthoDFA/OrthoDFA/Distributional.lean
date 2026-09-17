import OrthoDFA.Liveness
import OrthoDFA.Complexity
import OrthoDFA.Termination

/-!
# The distributional clustering guarantee (PR #257) — target statement + proof

The clustering algorithm's own guarantee, distributional and per population, as in
PR #257 ("Hold every prefix population to the FNR limit"):

* prefixes come from a collection of distributions `D : J → Measure S` (the uniform
  pool, the boundary set, one per state) held individually;
* candidate suffixes are drawn from a suffix distribution `Dsf`, and findability is a
  single number `pAP` — the probability a drawn suffix is accept-preserving on the true
  noiseless oracle (`∀ p, ℓ(p·v) = ℓ(p)`);
* the seed is `ε = 1` (so accept-preserving = the seed's Nerode class);
* the guarantee: w.p. `≥ 1 − δ`, the returned family preserves acceptance on `≥ 1 − εcov`
  of each population `D j`.

This file builds the proof from reusable pieces.  `coverage` is the first: if every
family member flips at most `β` of a population's mass, the family preserves `≥ 1 − #F·β`
of it — a clean union bound, the step that turns per-suffix flip control into the
per-population fraction.
-/

namespace OrthoDFA

open MeasureTheory ProbabilityTheory
open scoped ENNReal

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {S : Type*} [MeasurableSpace S] [Monoid S]

/-- Coverage.  If every suffix in the family `F` flips at most `β` of population
`Dj`'s prefix mass, then the family preserves acceptance on at least `1 − #F·β` of `Dj`:
a union bound over the family's flip sets. -/
theorem coverage (O : Oracle μ S)
    (Dj : Measure S) [IsProbabilityMeasure Dj]
    (F : Finset S) (β : ℝ)
    (hmeas : ∀ v ∈ F, MeasurableSet {p | O.label (p * v) ≠ O.label p})
    (hF : ∀ v ∈ F, Dj.real {p | O.label (p * v) ≠ O.label p} ≤ β) :
    1 - (F.card : ℝ) * β ≤ Dj.real {p | ∀ v ∈ F, O.label (p * v) = O.label p} := by
  classical
  -- the "some family member flips p" set is the finite union of the per-suffix flip sets
  have hunion : {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p}
      = ⋃ v ∈ F, {p | O.label (p * v) ≠ O.label p} := by
    ext p; simp only [Set.mem_setOf_eq, Set.mem_iUnion, exists_prop]
    push_neg; rfl
  have hbad : Dj.real {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} ≤ (F.card : ℝ) * β := by
    rw [hunion]
    calc Dj.real (⋃ v ∈ F, {p | O.label (p * v) ≠ O.label p})
        ≤ ∑ v ∈ F, Dj.real {p | O.label (p * v) ≠ O.label p} := measureReal_biUnion_le _ _
      _ ≤ ∑ _v ∈ F, β := Finset.sum_le_sum hF
      _ = (F.card : ℝ) * β := by rw [Finset.sum_const, nsmul_eq_mul]
  -- the preservation set is the complement of the flip set
  have hcompl : Dj.real {p | ∀ v ∈ F, O.label (p * v) = O.label p}
      = 1 - Dj.real {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} := by
    have hmeasBad : MeasurableSet {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p} := by
      rw [hunion]; exact F.measurableSet_biUnion (fun v hv => hmeas v hv)
    have huniv : Dj.real Set.univ = 1 := by simp [measureReal_def, measure_univ]
    have hset : {p | ∀ v ∈ F, O.label (p * v) = O.label p}
        = {p | ¬ ∀ v ∈ F, O.label (p * v) = O.label p}ᶜ := by
      ext p; simp
    rw [hset, measureReal_compl hmeasBad, huniv]
  rw [hcompl]; linarith [hbad]

#print axioms coverage

/-- Findability.  Over `M` i.i.d. suffix draws from `Dsf`, each accept-preserving with
probability `≥ pAP`, the probability that *none* is accept-preserving is `≤ (1−pAP)^M`.
A direct instance of the product-measure `geometric_miss`; `geom_le` then drives it below
any budget once `M ≥ log(1/·)/pAP`. -/
theorem findAP (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (pAP : ℝ) (AP : Set S) (hAPmeas : MeasurableSet AP)
    (hpAP1 : pAP ≤ 1) (hfind : pAP ≤ Dsf.real AP) :
    (Measure.pi (fun _ : Fin M => Dsf)).real {x | ∀ i, x i ∉ AP} ≤ (1 - pAP) ^ M :=
  geometric_miss (fun _ => Dsf) (fun _ => AP) (fun _ => hAPmeas) pAP hpAP1 (fun _ => hfind)

#print axioms findAP

/-- Uniform slice bound transfers to the product.  If every slice of a measurable
product event has probability `≤ ε`, so does the event.  This is the plumbing that lets a
bound proved for each *fixed* candidate pool be used when the pool is itself drawn. -/
theorem prod_le_of_slice {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (ma : Measure α) [IsProbabilityMeasure ma] (mb : Measure β) [IsProbabilityMeasure mb]
    (E : Set (α × β)) (hE : MeasurableSet E) (ε : ℝ) (hε : 0 ≤ ε)
    (hslice : ∀ a, mb.real (Prod.mk a ⁻¹' E) ≤ ε) :
    (ma.prod mb).real E ≤ ε := by
  have hle : ∀ a, mb (Prod.mk a ⁻¹' E) ≤ ENNReal.ofReal ε := by
    intro a
    rw [← ENNReal.ofReal_toReal (measure_ne_top mb (Prod.mk a ⁻¹' E))]
    exact ENNReal.ofReal_le_ofReal (hslice a)
  have hmain : (ma.prod mb) E ≤ ENNReal.ofReal ε := by
    rw [Measure.prod_apply hE]
    calc ∫⁻ a, mb (Prod.mk a ⁻¹' E) ∂ma ≤ ∫⁻ _a, ENNReal.ofReal ε ∂ma := lintegral_mono hle
      _ = ENNReal.ofReal ε := by simp
  calc (ma.prod mb).real E = ((ma.prod mb) E).toReal := rfl
    _ ≤ (ENNReal.ofReal ε).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hmain
    _ = ε := ENNReal.toReal_ofReal hε

#print axioms prod_le_of_slice

/-- First-marginal probability of a product event depending only on the first coordinate. -/
theorem prod_fst_real {α β : Type*} [MeasurableSpace α] [MeasurableSpace β]
    (ma : Measure α) [IsProbabilityMeasure ma] (ma' : Measure β) [IsProbabilityMeasure ma']
    (B : Set α) : (ma.prod ma').real {x : α × β | x.1 ∈ B} = ma.real B := by
  have hset : {x : α × β | x.1 ∈ B} = B ×ˢ (Set.univ : Set β) := by ext x; simp
  rw [measureReal_def, hset, Measure.prod_prod, measure_univ, mul_one, ← measureReal_def]


section Draws
variable [MeasurableMul S]

/-- The `Dj`-flip-mass of a suffix: the probability that `v` flips a `Dj`-drawn prefix.
This is the *distributional* quantity `good`/`bad` are defined by. -/
noncomputable def flipMass (O : Oracle μ S) (Dj : Measure S) (v : S) : ℝ :=
  ∫ p, O.flip v p ∂Dj

lemma flip_meas (O : Oracle μ S) (v : S) : Measurable (fun p => O.flip v p) := by
  have h1 : Measurable (fun p : S => O.label (p * v)) :=
    O.label_meas.comp (measurable_mul_const v)
  show Measurable (fun p => O.label (p * v) + O.label p - 2 * O.label (p * v) * O.label p)
  exact (h1.add O.label_meas).sub ((measurable_const.mul h1).mul O.label_meas)

lemma flip_icc (O : Oracle μ S) (v p : S) : O.flip v p ∈ Set.Icc (0 : ℝ) 1 := by
  rcases O.flip_bit v p with h | h <;> rw [Set.mem_Icc, h] <;> constructor <;> norm_num

section Persistent
variable {ι : Type*} [Fintype ι] [IsCancelMul S]

/-- Run space for the persistent oracle: `ι` prefix draws together with one shared
noise sample.  Re-reading the same query string returns the same bit — this is the honest
RCN model, unlike a fresh-noise-per-query product. -/
noncomputable def runMeasure (Dfam : ι → Measure S) : Measure ((ι → S) × Ω) :=
  (Measure.pi Dfam).prod μ

instance (Dfam : ι → Measure S) [∀ z, IsProbabilityMeasure (Dfam z)] :
    IsProbabilityMeasure (runMeasure (μ := μ) Dfam) := by
  unfold runMeasure; infer_instance

/-- The loss of suffix `v`: its reads at the drawn prefixes, against the shared noise. -/
noncomputable def ploss (O : Oracle μ S) (v : S) (x : (ι → S) × Ω) : ℝ :=
  ∑ z, O.read x.1 v z x.2

/-- The flip count of `v` on the drawn prefixes (a function of the draw alone). -/
noncomputable def pflip (O : Oracle μ S) (v : S) (x : (ι → S) × Ω) : ℝ :=
  ∑ z, O.flip v (x.1 z)

variable [Countable S] [MeasurableSingletonClass S]

lemma pread_meas (O : Oracle μ S) (v : S) (z : ι) :
    Measurable (fun x : (ι → S) × Ω => O.read x.1 v z x.2) := by
  have hp : Measurable (fun x : (ι → S) × Ω => x.1 z) :=
    (measurable_pi_apply z).comp measurable_fst
  have hf : Measurable (fun x : (ι → S) × Ω => O.flip v (x.1 z)) :=
    (flip_meas O v).comp hp
  have hn : Measurable (fun x : (ι → S) × Ω => O.noise (x.1 z * v) x.2) :=
    O.noise_meas.comp (((measurable_mul_const v).comp hp).prodMk measurable_snd)
  show Measurable (fun x : (ι → S) × Ω =>
    O.flip v (x.1 z) + (1 - 2 * O.flip v (x.1 z)) * O.noise (x.1 z * v) x.2)
  exact hf.add ((measurable_const.sub (measurable_const.mul hf)).mul hn)

lemma ploss_meas (O : Oracle μ S) (v : S) : Measurable (ploss (μ := μ) (ι := ι) O v) :=
  Finset.measurable_sum _ (fun z _ => pread_meas O v z)

lemma pflip_meas (O : Oracle μ S) (v : S) : Measurable (pflip (μ := μ) (ι := ι) O v) :=
  Finset.measurable_sum _ (fun z _ =>
    (flip_meas O v).comp ((measurable_pi_apply z).comp measurable_fst))

/-- The draws are distinct — the event on which the persistent oracle is never read
twice at the same query string. -/
lemma injective_meas : MeasurableSet {x : (ι → S) × Ω | Function.Injective x.1} := by
  classical
  have hset : {x : (ι → S) × Ω | Function.Injective x.1}
      = ⋂ z : ι, ⋂ z' : ι, {x | z = z' ∨ x.1 z ≠ x.1 z'} := by
    ext x
    simp only [Set.mem_setOf_eq, Set.mem_iInter]
    constructor
    · intro h z z'
      by_cases hzz : z = z'
      · exact Or.inl hzz
      · exact Or.inr (fun he => hzz (h he))
    · intro h a b hab
      rcases h a b with hz | hne
      · exact hz
      · exact absurd hab hne
  rw [hset]
  refine MeasurableSet.iInter (fun z => MeasurableSet.iInter (fun z' => ?_))
  by_cases hzz : z = z'
  · simp [hzz]
  · have : {x : (ι → S) × Ω | z = z' ∨ x.1 z ≠ x.1 z'}
        = {x : (ι → S) × Ω | x.1 z ≠ x.1 z'} := by
      ext x; simp [hzz]
    rw [this]
    have hm : Measurable (fun x : (ι → S) × Ω => (x.1 z, x.1 z')) :=
      (((measurable_pi_apply z).comp measurable_fst)).prodMk
        ((measurable_pi_apply z').comp measurable_fst)
    have hdiag : MeasurableSet {q : S × S | q.1 ≠ q.2} := by
      have : {q : S × S | q.1 ≠ q.2} = {q : S × S | q.1 = q.2}ᶜ := by ext q; simp
      rw [this]
      exact (measurableSet_eq_fun measurable_fst measurable_snd).compl
    exact hm hdiag

/-- Level 1 (noise), conditional on distinct draws.  Given that the draws are
distinct — so the persistent oracle is never read twice at the same query string, and its
bits really are independent (`read_indep`, via right-cancellation) — the loss concentrates
around its conditional mean `N·η + (1−2η)·(flip count)`. -/
lemma ploss_cond_upper (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x
            + ((Finset.univ : Finset ι).card : ℝ) * g ≤ ploss (μ := μ) O v x}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  classical
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  refine prod_le_of_slice _ _ _ ?_ _ (Real.exp_pos _).le (fun p => ?_)
  · -- measurable
    exact (injective_meas (ι := ι) (S := S) (Ω := Ω)).inter
      (measurableSet_le (((pflip_meas O v).const_mul _).const_add _ |>.add measurable_const)
        (ploss_meas O v))
  · -- slice
    by_cases hinj : Function.Injective p
    · have hmeanEq : ∑ z, μ[O.read p v z] = N * (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) := by
        rw [Finset.sum_congr rfl (fun z _ => O.read_mean p v z), Finset.sum_add_distrib,
          Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum, ← hN]
        field_simp
      have h := sumUpper_le (O.read p v) (Finset.univ : Finset ι)
        (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) g
        (fun z => (O.read_meas p v z).aemeasurable) (O.read_indep p hinj v)
        (O.read_icc p v) (le_of_eq hmeanEq) hg
      refine le_trans (measureReal_mono ?_) h
      intro ω hω
      obtain ⟨-, hle⟩ := hω
      show N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) + g) ≤ ∑ z, O.read p v z ω
      have hNne : N ≠ 0 := ne_of_gt hNpos
      have : N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) + g)
          = N * O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) + N * g := by
        field_simp
      rw [this]
      exact hle
    · have : (Prod.mk p ⁻¹' {x : (ι → S) × Ω | Function.Injective x.1 ∧
          N * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x + N * g ≤ ploss (μ := μ) O v x})
          = ∅ := by
        ext ω; simp only [Set.mem_preimage, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h, -⟩; exact hinj h
      rw [this]; simpa using (Real.exp_pos _).le

/-- Level 1, lower tail (symmetric). -/
lemma ploss_cond_lower (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
            + (1 - 2 * O.η) * pflip (μ := μ) O v x
            - ((Finset.univ : Finset ι).card : ℝ) * g}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  classical
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  refine prod_le_of_slice _ _ _ ?_ _ (Real.exp_pos _).le (fun p => ?_)
  · exact (injective_meas (ι := ι) (S := S) (Ω := Ω)).inter
      (measurableSet_le (ploss_meas O v)
        ((((pflip_meas O v).const_mul _).const_add _).sub measurable_const))
  · by_cases hinj : Function.Injective p
    · have hmeanEq : ∑ z, μ[O.read p v z]
          = N * (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) := by
        rw [Finset.sum_congr rfl (fun z _ => O.read_mean p v z), Finset.sum_add_distrib,
          Finset.sum_const, nsmul_eq_mul, ← Finset.mul_sum, ← hN]
        field_simp
      have h := sumLower_le (O.read p v) (Finset.univ : Finset ι)
        (O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) g
        (fun z => (O.read_meas p v z).aemeasurable) (O.read_indep p hinj v)
        (O.read_icc p v) (ge_of_eq hmeanEq) hg
      refine le_trans (measureReal_mono ?_) h
      intro ω hω
      obtain ⟨-, hle⟩ := hω
      show ∑ z, O.read p v z ω
        ≤ N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) - g)
      have hNne : N ≠ 0 := ne_of_gt hNpos
      have hrw : N * ((O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) / N) - g)
          = N * O.η + (1 - 2 * O.η) * (∑ z, O.flip v (p z)) - N * g := by field_simp
      rw [hrw]; exact hle
    · have : (Prod.mk p ⁻¹' {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ploss (μ := μ) O v x ≤ N * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x - N * g})
          = ∅ := by
        ext ω; simp only [Set.mem_preimage, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
        rintro ⟨h, -⟩; exact hinj h
      rw [this]; simpa using (Real.exp_pos _).le

/-- The flip count's mean on the prefix marginal is the summed flip-mass. -/
lemma pflip_meanSum (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) :
    ∑ z, (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)]
      = ∑ z, flipMass O (Dfam z) v := by
  refine Finset.sum_congr rfl (fun z _ => ?_)
  have hmap : Measure.map (fun p : ι → S => p z) (Measure.pi Dfam) = Dfam z :=
    (measurePreserving_eval Dfam z).map_eq
  calc (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)]
      = ∫ w, O.flip v w ∂(Measure.map (fun p : ι → S => p z) (Measure.pi Dfam)) := by
        rw [integral_map (measurable_pi_apply z).aemeasurable
          (flip_meas O v).aestronglyMeasurable]
    _ = ∫ w, O.flip v w ∂(Dfam z) := by rw [hmap]
    _ = flipMass O (Dfam z) v := rfl

/-- Level 2 (sampling), upper tail.  The flip count on the drawn prefixes concentrates
above its summed distributional flip-mass. -/
lemma pflip_upper (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
          + ((Finset.univ : Finset ι).card : ℝ) * g ≤ pflip (μ := μ) O v x}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hNne : N ≠ 0 := ne_of_gt hNpos
  have heq : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v) + N * g ≤ pflip (μ := μ) O v x}
      = (Measure.pi Dfam).real
        {p : ι → S | (∑ z, flipMass O (Dfam z) v) + N * g ≤ ∑ z, O.flip v (p z)} := by
    unfold runMeasure
    exact prod_fst_real (Measure.pi Dfam) μ
      {p : ι → S | (∑ z, flipMass O (Dfam z) v) + N * g ≤ ∑ z, O.flip v (p z)}
  rw [heq]
  have hmean : ∑ z, (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)]
      ≤ N * ((∑ z, flipMass O (Dfam z) v) / N) := by
    have hc : N * ((∑ z, flipMass O (Dfam z) v) / N) = ∑ z, flipMass O (Dfam z) v := by
      field_simp
    rw [hc]; exact le_of_eq (pflip_meanSum O Dfam v)
  have h := sumUpper_le (fun (z : ι) (p : ι → S) => O.flip v (p z)) (Finset.univ : Finset ι)
    ((∑ z, flipMass O (Dfam z) v) / N) g
    (fun z => ((flip_meas O v).comp (measurable_pi_apply z)).aemeasurable)
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun z => Filter.Eventually.of_forall (fun p => flip_icc O v (p z))) hmean hg
  refine le_trans (measureReal_mono ?_) h
  intro q hq
  show N * ((∑ z, flipMass O (Dfam z) v) / N + g) ≤ ∑ z, O.flip v (q z)
  have hrw : N * ((∑ z, flipMass O (Dfam z) v) / N + g)
      = (∑ z, flipMass O (Dfam z) v) + N * g := by field_simp
  rw [hrw]; exact hq

/-- Level 2 (sampling), lower tail. -/
lemma pflip_lower (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g : ℝ) (hg : 0 ≤ g)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ)) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | pflip (μ := μ) O v x
          ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g}
      ≤ Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g ^ 2) := by
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  have hNne : N ≠ 0 := ne_of_gt hNpos
  have heq : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | pflip (μ := μ) O v x ≤ (∑ z, flipMass O (Dfam z) v) - N * g}
      = (Measure.pi Dfam).real
        {p : ι → S | ∑ z, O.flip v (p z) ≤ (∑ z, flipMass O (Dfam z) v) - N * g} := by
    unfold runMeasure
    exact prod_fst_real (Measure.pi Dfam) μ
      {p : ι → S | ∑ z, O.flip v (p z) ≤ (∑ z, flipMass O (Dfam z) v) - N * g}
  rw [heq]
  have hmean : N * ((∑ z, flipMass O (Dfam z) v) / N)
      ≤ ∑ z, (Measure.pi Dfam)[fun p : ι → S => O.flip v (p z)] := by
    have hc : N * ((∑ z, flipMass O (Dfam z) v) / N) = ∑ z, flipMass O (Dfam z) v := by
      field_simp
    rw [hc]; exact le_of_eq (pflip_meanSum O Dfam v).symm
  have h := sumLower_le (fun (z : ι) (p : ι → S) => O.flip v (p z)) (Finset.univ : Finset ι)
    ((∑ z, flipMass O (Dfam z) v) / N) g
    (fun z => ((flip_meas O v).comp (measurable_pi_apply z)).aemeasurable)
    (iIndepFun_pi (fun _ => (flip_meas O v).aemeasurable))
    (fun z => Filter.Eventually.of_forall (fun p => flip_icc O v (p z))) hmean hg
  refine le_trans (measureReal_mono ?_) h
  intro q hq
  show ∑ z, O.flip v (q z) ≤ N * ((∑ z, flipMass O (Dfam z) v) / N - g)
  have hrw : N * ((∑ z, flipMass O (Dfam z) v) / N - g)
      = (∑ z, flipMass O (Dfam z) v) - N * g := by field_simp
  rw [hrw]; exact hq

/-- Per-suffix upper tail, persistent oracle.  A suffix with zero summed flip-mass
keeps its loss below the band, except for three sources: a collision (the persistent
oracle read twice at the same query string — which is why a spread-out prefix
distribution is genuinely necessary), the sampling tail, and the noise tail. -/
lemma ploss_good_upper (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g₁ g₂ κ : ℝ) (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hcoll : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hv : ∑ z, flipMass O (Dfam z) v = 0) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | ((Finset.univ : Finset ι).card : ℝ)
            * (O.η + (1 - 2 * O.η) * g₂ + g₁) ≤ ploss (μ := μ) O v x}
      ≤ κ + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₂ ^ 2)
          + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₁ ^ 2) := by
  have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
  have hincl : {x : (ι → S) × Ω | ((Finset.univ : Finset ι).card : ℝ)
        * (O.η + (1 - 2 * O.η) * g₂ + g₁) ≤ ploss (μ := μ) O v x}
      ⊆ {x : (ι → S) × Ω | ¬ Function.Injective x.1}
        ∪ ({x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
              + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x}
          ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
              ((Finset.univ : Finset ι).card : ℝ) * O.η
                + (1 - 2 * O.η) * pflip (μ := μ) O v x
                + ((Finset.univ : Finset ι).card : ℝ) * g₁ ≤ ploss (μ := μ) O v x}) := by
    intro x hx
    by_cases hinj : Function.Injective x.1
    · right
      by_cases hfl : (∑ z, flipMass O (Dfam z) v)
          + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x
      · exact Or.inl hfl
      · refine Or.inr ⟨hinj, ?_⟩
        push_neg at hfl
        rw [hv, zero_add] at hfl
        have hkey : (1 - 2 * O.η) * pflip (μ := μ) O v x
            ≤ (1 - 2 * O.η) * (((Finset.univ : Finset ι).card : ℝ) * g₂) :=
          mul_le_mul_of_nonneg_left hfl.le h2η
        simp only [Set.mem_setOf_eq] at hx ⊢
        linarith [hx, hkey]
    · exact Or.inl hinj
  have hu1 := measureReal_mono (μ := runMeasure (μ := μ) Dfam) hincl
  have hu2 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | ¬ Function.Injective x.1}
    ({x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
        + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x}
      ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x
            + ((Finset.univ : Finset ι).card : ℝ) * g₁ ≤ ploss (μ := μ) O v x})
  have hu3 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | (∑ z, flipMass O (Dfam z) v)
        + ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x}
    {x : (ι → S) × Ω | Function.Injective x.1 ∧
      ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η) * pflip (μ := μ) O v x
        + ((Finset.univ : Finset ι).card : ℝ) * g₁ ≤ ploss (μ := μ) O v x}
  linarith [hu1, hu2, hu3, hcoll, pflip_upper O Dfam v g₂ hg₂ hNpos,
    ploss_cond_upper O Dfam v g₁ hg₁ hNpos]

/-- Per-suffix lower tail, persistent oracle.  A suffix whose summed flip-mass is at
least `σ` keeps its loss *above* the band, provided the band leaves room (`hband`). -/
lemma ploss_bad_lower (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (v : S) (g₁ g₂ κ σ : ℝ)
    (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hcoll : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hband : ((Finset.univ : Finset ι).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)
      ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η)
          * (σ - ((Finset.univ : Finset ι).card : ℝ) * g₂)
        - ((Finset.univ : Finset ι).card : ℝ) * g₁)
    (hv : σ ≤ ∑ z, flipMass O (Dfam z) v) :
    (runMeasure (μ := μ) Dfam).real
        {x : (ι → S) × Ω | ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ)
            * (O.η + (1 - 2 * O.η) * g₂ + g₁)}
      ≤ κ + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₂ ^ 2)
          + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₁ ^ 2) := by
  have h2η : (0 : ℝ) ≤ 1 - 2 * O.η := by linarith [O.hη]
  have hincl : {x : (ι → S) × Ω | ploss (μ := μ) O v x
        ≤ ((Finset.univ : Finset ι).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)}
      ⊆ {x : (ι → S) × Ω | ¬ Function.Injective x.1}
        ∪ ({x : (ι → S) × Ω | pflip (μ := μ) O v x
              ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂}
          ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
              ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
                + (1 - 2 * O.η) * pflip (μ := μ) O v x
                - ((Finset.univ : Finset ι).card : ℝ) * g₁}) := by
    intro x hx
    by_cases hinj : Function.Injective x.1
    · right
      by_cases hfl : pflip (μ := μ) O v x
          ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂
      · exact Or.inl hfl
      · refine Or.inr ⟨hinj, ?_⟩
        push_neg at hfl
        have hstep : σ - ((Finset.univ : Finset ι).card : ℝ) * g₂ ≤ pflip (μ := μ) O v x := by
          linarith [hfl, hv]
        have hkey : (1 - 2 * O.η) * (σ - ((Finset.univ : Finset ι).card : ℝ) * g₂)
            ≤ (1 - 2 * O.η) * pflip (μ := μ) O v x :=
          mul_le_mul_of_nonneg_left hstep h2η
        simp only [Set.mem_setOf_eq] at hx ⊢
        linarith [hx, hband, hkey]
    · exact Or.inl hinj
  have hu1 := measureReal_mono (μ := runMeasure (μ := μ) Dfam) hincl
  have hu2 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | ¬ Function.Injective x.1}
    ({x : (ι → S) × Ω | pflip (μ := μ) O v x
        ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂}
      ∪ {x : (ι → S) × Ω | Function.Injective x.1 ∧
          ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
            + (1 - 2 * O.η) * pflip (μ := μ) O v x
            - ((Finset.univ : Finset ι).card : ℝ) * g₁})
  have hu3 := measureReal_union_le (μ := runMeasure (μ := μ) Dfam)
    {x : (ι → S) × Ω | pflip (μ := μ) O v x
        ≤ (∑ z, flipMass O (Dfam z) v) - ((Finset.univ : Finset ι).card : ℝ) * g₂}
    {x : (ι → S) × Ω | Function.Injective x.1 ∧
      ploss (μ := μ) O v x ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η
        + (1 - 2 * O.η) * pflip (μ := μ) O v x
        - ((Finset.univ : Finset ι).card : ℝ) * g₁}
  linarith [hu1, hu2, hu3, hcoll, pflip_lower O Dfam v g₂ hg₂ hNpos,
    ploss_cond_lower O Dfam v g₁ hg₁ hNpos]

end Persistent

/-- `flip` is the indicator of "the labels of `p·v` and `p` differ". -/
lemma flip_eq_one_iff (O : Oracle μ S) (v p : S) :
    O.flip v p = 1 ↔ O.label (p * v) ≠ O.label p := by
  rcases O.label_bit (p * v) with h1 | h1 <;> rcases O.label_bit p with h2 | h2 <;>
    simp only [Oracle.flip, h1, h2] <;> norm_num

lemma flipSet_meas (O : Oracle μ S) (v : S) :
    MeasurableSet {p | O.label (p * v) ≠ O.label p} := by
  have : {p | O.label (p * v) ≠ O.label p} = (fun p => O.flip v p) ⁻¹' {1} := by
    ext p; simpa using (flip_eq_one_iff O v p).symm
  rw [this]; exact (flip_meas O v) (measurableSet_singleton 1)

/-- The flip-mass is the measure of the flip set: `∫ flip ∂Dj = Dj{p | ℓ(p·v) ≠ ℓ(p)}`.
This is what lets the selection guarantee (stated in `flipMass`) feed `coverage`
(stated as a measure). -/
lemma flipMass_eq (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    flipMass O Dj v = Dj.real {p | O.label (p * v) ≠ O.label p} := by
  have hind : (fun p => O.flip v p)
      = Set.indicator {p | O.label (p * v) ≠ O.label p} (fun _ => (1 : ℝ)) := by
    funext p
    by_cases h : O.label (p * v) = O.label p
    · rw [Set.indicator_of_notMem (by simpa using h)]
      rcases O.label_bit p with h2 | h2 <;>
        simp only [Oracle.flip, h, h2] <;> norm_num
    · rw [Set.indicator_of_mem (by simpa using h)]
      exact (flip_eq_one_iff O v p).mpr h
  rw [flipMass, hind, integral_indicator (flipSet_meas O v), setIntegral_const,
    smul_eq_mul, mul_one, measureReal_def]

lemma flipMass_nonneg (O : Oracle μ S) (Dj : Measure S) [IsProbabilityMeasure Dj] (v : S) :
    0 ≤ flipMass O Dj v := by
  rw [flipMass_eq]; exact measureReal_nonneg

/-- Selection ⇒ per-population coverage.  If every family member's *summed*
flip-mass over the populations is below `εfam`, then on each population the family
preserves acceptance on at least a `1 − #F·εfam` fraction.  (Summed control gives
per-population control because flip-masses are nonnegative.) -/
theorem coverage_of_summed_flip [DecidableEq S] {J : Type*} (O : Oracle μ S)
    (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (populations : Finset J) (F : Finset S) (εfam : ℝ)
    (hF : ∀ w ∈ F, ∑ j ∈ populations, flipMass O (D j) w ≤ εfam) :
    ∀ j ∈ populations,
      1 - (F.card : ℝ) * εfam ≤ (D j).real {p | ∀ v ∈ F, O.label (p * v) = O.label p} := by
  intro j hj
  refine coverage O (D j) F εfam (fun v _ => flipSet_meas O v) (fun v hv => ?_)
  rw [← flipMass_eq]
  refine le_trans ?_ (hF v hv)
  exact Finset.single_le_sum (f := fun j' => flipMass O (D j') v)
    (fun j' _ => flipMass_nonneg O (D j') v) hj

#print axioms flipMass_eq
#print axioms coverage_of_summed_flip


section Assembly
variable [Countable S] [MeasurableSingletonClass S] [DecidableEq S] [IsCancelMul S]

/-- With `S` countable and discrete (strings over a finite alphabet), a set of pairs is
measurable as soon as each of its `S`-slices is: it is a countable union of rectangles. -/
lemma measurableSet_of_countable_slices {Y : Type*} [MeasurableSpace Y] (P : S → Set Y)
    (hP : ∀ v, MeasurableSet (P v)) : MeasurableSet {q : S × Y | q.2 ∈ P q.1} := by
  have hset : {q : S × Y | q.2 ∈ P q.1} = ⋃ v : S, {v} ×ˢ P v := by
    ext q
    simp only [Set.mem_setOf_eq, Set.mem_iUnion, Set.mem_prod, Set.mem_singleton_iff]
    exact ⟨fun h => ⟨q.1, rfl, h⟩, fun ⟨v, hv, h⟩ => by subst hv; exact h⟩
  rw [hset]
  exact MeasurableSet.iUnion (fun v => (measurableSet_singleton v).prod (hP v))

/-- Findability, quantitative.  Over `M` i.i.d. suffix draws, the count of draws
landing in `G` (probability `≥ pAP` each) falls to `M(pAP−γ)` only w.p. `exp(-2Mγ²)`. -/
theorem goodCount_le (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (cands : Finset (Fin M)) (pAP γ : ℝ) (hγ : 0 ≤ γ)
    (G : Set S) (hGm : MeasurableSet G) (hG : pAP ≤ Dsf.real G) :
    (Measure.pi (fun _ : Fin M => Dsf)).real
        {s : Fin M → S | ∑ c ∈ cands, Set.indicator G (fun _ => (1 : ℝ)) (s c)
            ≤ (cands.card : ℝ) * (pAP - γ)}
      ≤ Real.exp (-2 * (cands.card : ℝ) * γ ^ 2) := by
  classical
  set I : S → ℝ := Set.indicator G (fun _ => (1 : ℝ)) with hI
  have hImeas : Measurable I := (measurable_one.indicator hGm)
  have hIicc : ∀ v, I v ∈ Set.Icc (0 : ℝ) 1 := by
    intro v
    rw [Set.mem_Icc]
    by_cases h : v ∈ G
    · simp only [hI, Set.indicator_of_mem h]; norm_num
    · simp only [hI, Set.indicator_of_notMem h]; norm_num
  have hmean : ∫ v, I v ∂Dsf = Dsf.real G := by
    rw [hI, integral_indicator hGm, setIntegral_const, smul_eq_mul, mul_one, measureReal_def]
  have hXmeas : ∀ c : Fin M, AEMeasurable (fun s : Fin M → S => I (s c))
      (Measure.pi (fun _ : Fin M => Dsf)) :=
    fun c => (hImeas.comp (measurable_pi_apply c)).aemeasurable
  have hXindep : iIndepFun (fun (c : Fin M) (s : Fin M → S) => I (s c))
      (Measure.pi (fun _ : Fin M => Dsf)) := iIndepFun_pi (fun _ => hImeas.aemeasurable)
  have hXicc : ∀ c : Fin M, ∀ᵐ s ∂(Measure.pi (fun _ : Fin M => Dsf)),
      I (s c) ∈ Set.Icc (0 : ℝ) 1 := fun c => Filter.Eventually.of_forall (fun s => hIicc (s c))
  have hXmarg : ∀ c : Fin M,
      (Measure.pi (fun _ : Fin M => Dsf))[fun s : Fin M → S => I (s c)] = Dsf.real G := by
    intro c
    have hmap : Measure.map (fun s : Fin M → S => s c) (Measure.pi (fun _ : Fin M => Dsf)) = Dsf :=
      (measurePreserving_eval (fun _ : Fin M => Dsf) c).map_eq
    calc (Measure.pi (fun _ : Fin M => Dsf))[fun s : Fin M → S => I (s c)]
        = ∫ v, I v ∂(Measure.map (fun s : Fin M → S => s c)
            (Measure.pi (fun _ : Fin M => Dsf))) := by
          rw [integral_map (measurable_pi_apply c).aemeasurable hImeas.aestronglyMeasurable]
      _ = ∫ v, I v ∂Dsf := by rw [hmap]
      _ = Dsf.real G := hmean
  have hsum : (cands.card : ℝ) * pAP
      ≤ ∑ c ∈ cands, (Measure.pi (fun _ : Fin M => Dsf))[fun s : Fin M → S => I (s c)] := by
    rw [Finset.sum_congr rfl (fun c _ => hXmarg c), Finset.sum_const, nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_left hG (Nat.cast_nonneg _)
  exact sumLower_le (fun (c : Fin M) (s : Fin M → S) => I (s c)) cands pAP γ
    hXmeas hXindep hXicc hsum hγ

#print axioms goodCount_le

/-- The band threshold: the common value the good-upper and bad-lower bands collapse to. -/
noncomputable def pthresh (O : Oracle μ S) (g₁ g₂ N : ℝ) : ℝ :=
  N * (O.η + (1 - 2 * O.η) * g₂ + g₁)

/-- The per-draw-index separation-failure event is small, uniformly in the draw.
For draw index `c`, the event that the drawn suffix is good yet reads above the band, or
bad yet reads below it.  Bounded by the persistent-oracle per-suffix tails (collision +
sampling + noise), transferred to the product by `prod_le_of_slice`. -/
theorem pindex_event_le {ι : Type*} [Fintype ι] (O : Oracle μ S) (Dfam : ι → Measure S)
    [∀ z, IsProbabilityMeasure (Dfam z)] (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M : ℕ) (c : Fin M) (g₁ g₂ κ σ : ℝ) (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂) (hσpos : 0 < σ)
    (hNpos : 0 < ((Finset.univ : Finset ι).card : ℝ))
    (hcoll : (runMeasure (μ := μ) Dfam).real
      {x : (ι → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hband : ((Finset.univ : Finset ι).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)
      ≤ ((Finset.univ : Finset ι).card : ℝ) * O.η + (1 - 2 * O.η)
          * (σ - ((Finset.univ : Finset ι).card : ℝ) * g₂)
        - ((Finset.univ : Finset ι).card : ℝ) * g₁) :
    ((Measure.pi (fun _ : Fin M => Dsf)).prod (runMeasure (μ := μ) Dfam)).real
        {y : (Fin M → S) × ((ι → S) × Ω) |
          ((∑ z, flipMass O (Dfam z) (y.1 c) = 0)
            ∧ pthresh O g₁ g₂ ((Finset.univ : Finset ι).card : ℝ) ≤ ploss (μ := μ) O (y.1 c) y.2)
          ∨ ((σ ≤ ∑ z, flipMass O (Dfam z) (y.1 c))
            ∧ ploss (μ := μ) O (y.1 c) y.2
                ≤ pthresh O g₁ g₂ ((Finset.univ : Finset ι).card : ℝ))}
      ≤ κ + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₂ ^ 2)
          + Real.exp (-2 * ((Finset.univ : Finset ι).card : ℝ) * g₁ ^ 2) := by
  classical
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  set P : S → Set ((ι → S) × Ω) := fun v =>
    {w | ((∑ z, flipMass O (Dfam z) v = 0) ∧ pthresh O g₁ g₂ N ≤ ploss (μ := μ) O v w)
       ∨ ((σ ≤ ∑ z, flipMass O (Dfam z) v) ∧ ploss (μ := μ) O v w ≤ pthresh O g₁ g₂ N)}
    with hP
  have hκ0 : 0 ≤ κ := le_trans measureReal_nonneg hcoll
  have hPmeas : ∀ v, MeasurableSet (P v) := by
    intro v
    have h1 : MeasurableSet {w : (ι → S) × Ω | pthresh O g₁ g₂ N ≤ ploss (μ := μ) O v w} :=
      measurableSet_le measurable_const (ploss_meas O v)
    have h2 : MeasurableSet {w : (ι → S) × Ω | ploss (μ := μ) O v w ≤ pthresh O g₁ g₂ N} :=
      measurableSet_le (ploss_meas O v) measurable_const
    have hA : MeasurableSet {w : (ι → S) × Ω |
        (∑ z, flipMass O (Dfam z) v = 0) ∧ pthresh O g₁ g₂ N ≤ ploss (μ := μ) O v w} := by
      by_cases hg : ∑ z, flipMass O (Dfam z) v = 0
      · convert h1 using 1; ext w; simp [hg]
      · convert MeasurableSet.empty; ext w; simp [hg]
    have hB : MeasurableSet {w : (ι → S) × Ω |
        (σ ≤ ∑ z, flipMass O (Dfam z) v) ∧ ploss (μ := μ) O v w ≤ pthresh O g₁ g₂ N} := by
      by_cases hb : σ ≤ ∑ z, flipMass O (Dfam z) v
      · convert h2 using 1; ext w; simp [hb]
      · convert MeasurableSet.empty; ext w; simp [hb]
    exact hA.union hB
  have hslice : ∀ s : Fin M → S, (runMeasure (μ := μ) Dfam).real (P (s c))
      ≤ κ + Real.exp (-2 * N * g₂ ^ 2) + Real.exp (-2 * N * g₁ ^ 2) := by
    intro s
    by_cases hg : ∑ z, flipMass O (Dfam z) (s c) = 0
    · refine le_trans (measureReal_mono ?_)
        (ploss_good_upper O Dfam (s c) g₁ g₂ κ hg₁ hg₂ hNpos hcoll hg)
      intro w hw
      rcases hw with ⟨-, h⟩ | ⟨hb, -⟩
      · exact h
      · exfalso; rw [hg] at hb; linarith
    · by_cases hb : σ ≤ ∑ z, flipMass O (Dfam z) (s c)
      · refine le_trans (measureReal_mono ?_)
          (ploss_bad_lower O Dfam (s c) g₁ g₂ κ σ hg₁ hg₂ hNpos hcoll hband hb)
        intro w hw
        rcases hw with ⟨hg', -⟩ | ⟨-, h⟩
        · exact absurd hg' hg
        · exact h
      · have hemp : P (s c) = ∅ := by
          ext w; simp only [hP, Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
          rintro (⟨h, -⟩ | ⟨h, -⟩)
          · exact hg h
          · exact hb h
        rw [hemp]
        simpa using by positivity
  refine prod_le_of_slice _ _ _ ?_ _ (by positivity) (fun s => hslice s)
  have hfm : Measurable
      (fun y : (Fin M → S) × ((ι → S) × Ω) => (y.1 c, y.2)) :=
    ((measurable_pi_apply c).comp measurable_fst).prodMk measurable_snd
  show MeasurableSet ((fun y : (Fin M → S) × ((ι → S) × Ω) => (y.1 c, y.2)) ⁻¹'
    {q : S × ((ι → S) × Ω) | q.2 ∈ P q.1})
  exact hfm (measurableSet_of_countable_slices P hPmeas)

#print axioms pindex_event_le

/-- A failure bound gives the complementary success bound, with no measurability
needed: outer measure is subadditive and `s ∪ sᶜ = univ`. -/
theorem one_sub_le_compl_real {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] (s : Set α) (d : ℝ) (h : ν.real s ≤ d) : 1 - d ≤ ν.real sᶜ := by
  have hsub : (1 : ℝ≥0∞) ≤ ν s + ν sᶜ := by
    calc (1 : ℝ≥0∞) = ν Set.univ := measure_univ.symm
      _ = ν (s ∪ sᶜ) := by rw [Set.union_compl_self]
      _ ≤ ν s + ν sᶜ := measure_union_le _ _
  have hreal : (1 : ℝ) ≤ ν.real s + ν.real sᶜ := by
    have := ENNReal.toReal_mono (by finiteness) hsub
    rwa [ENNReal.toReal_add (measure_ne_top _ _) (measure_ne_top _ _), ENNReal.toReal_one] at this
  linarith

#print axioms one_sub_le_compl_real

/-- Distributional clustering, fixed budget, persistent oracle.  Failure form. -/
theorem clustering_budget {J : Type*} [Fintype J]
    (O : Oracle μ S) (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M m k : ℕ) (cands : Finset (Fin M)) (hkcands : k ≤ cands.card) (hmpos : 0 < m) (hJ : 0 < Fintype.card J)
    (εpop : ℝ) (hεpop : 0 < εpop)
    (g₁ g₂ κ pAP γsuf δ : ℝ) (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂) (hγsuf : 0 ≤ γsuf)
    (hcoll : (runMeasure (μ := μ) (fun z : J × Fin m => D z.1)).real
      {x : (J × Fin m → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hband : ((Finset.univ : Finset (J × Fin m)).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)
      ≤ ((Finset.univ : Finset (J × Fin m)).card : ℝ) * O.η + (1 - 2 * O.η)
          * ((m : ℝ) * εpop - ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₂)
        - ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₁)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcount : (k : ℝ) ≤ (cands.card : ℝ) * (pAP - γsuf))
    (hbudget : Real.exp (-2 * (cands.card : ℝ) * γsuf ^ 2)
        + (cands.card : ℝ) * (κ + Real.exp (-2 * ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₂ ^ 2)
            + Real.exp (-2 * ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₁ ^ 2)) ≤ δ) :
    ((Measure.pi (fun _ : Fin M => Dsf)).prod
        (runMeasure (μ := μ) (fun z : J × Fin m => D z.1))).real
      {y | ¬ ∀ j : J, 1 - (k : ℝ) * εpop
            ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
                  (fun c => ploss (μ := μ) O (y.1 c) y.2)
                  cands k).image y.1,
                O.label (p * v) = O.label p}}
      ≤ δ := by
  classical
  set ι := J × Fin m
  set Dfam : ι → Measure S := fun z => D z.1 with hDfam
  set N : ℝ := ((Finset.univ : Finset ι).card : ℝ) with hN
  set σ : ℝ := (m : ℝ) * εpop with hσ
  set G : Set S := {v | ∀ p, O.label (p * v) = O.label p} with hG
  set I : S → ℝ := Set.indicator G (fun _ => (1 : ℝ)) with hI
  have hσpos : 0 < σ := by rw [hσ]; exact mul_pos (by exact_mod_cast hmpos) hεpop
  have hNpos : 0 < N := by
    rw [hN, Finset.card_univ]
    have : 0 < Fintype.card ι := by
      rw [show Fintype.card ι = Fintype.card J * m from by
        rw [Fintype.card_prod, Fintype.card_fin]]
      exact Nat.mul_pos hJ hmpos
    exact_mod_cast this
  have hGmeas : MeasurableSet G := Set.Countable.measurableSet (Set.to_countable G)
  have hGgood : ∀ v ∈ G, ∑ z, flipMass O (Dfam z) v = 0 := by
    intro v hv
    refine Finset.sum_eq_zero (fun z _ => ?_)
    rw [flipMass_eq]
    have : {p | O.label (p * v) ≠ O.label p} = (∅ : Set S) := by ext p; simpa using hv p
    rw [this]; simp
  have hsplit : ∀ v, ∑ z : ι, flipMass O (Dfam z) v
      = (m : ℝ) * ∑ j : J, flipMass O (D j) v := by
    intro v
    rw [Fintype.sum_prod_type]
    simp [hDfam, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      Finset.mul_sum]
  set A : Set ((Fin M → S) × ((ι → S) × Ω)) :=
    {y | ∑ c ∈ cands, I (y.1 c) ≤ (cands.card : ℝ) * (pAP - γsuf)} with hA
  set E : Fin M → Set ((Fin M → S) × ((ι → S) × Ω)) := fun c =>
    {y | ((∑ z, flipMass O (Dfam z) (y.1 c) = 0) ∧ pthresh O g₁ g₂ N ≤ ploss (μ := μ) O (y.1 c) y.2)
       ∨ ((σ ≤ ∑ z, flipMass O (Dfam z) (y.1 c))
          ∧ ploss (μ := μ) O (y.1 c) y.2 ≤ pthresh O g₁ g₂ N)} with hE
  have hincl : {y : (Fin M → S) × ((ι → S) × Ω) | ¬ ∀ j : J, 1 - (k : ℝ) * εpop
      ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
            (fun c => ploss (μ := μ) O (y.1 c) y.2)
            cands k).image y.1,
          O.label (p * v) = O.label p}} ⊆ A ∪ ⋃ c ∈ cands, E c := by
    intro y hy
    by_contra hnot
    rw [Set.mem_union, not_or] at hnot
    obtain ⟨hnA, hnE⟩ := hnot
    simp only [Set.mem_iUnion, not_exists, exists_prop, not_and] at hnE
    have hcountgt : (cands.card : ℝ) * (pAP - γsuf) < ∑ c ∈ cands, I (y.1 c) := by
      by_contra hle; exact hnA (by rw [hA]; exact not_lt.mp hle)
    have hIsum : ∑ c ∈ cands, I (y.1 c)
        = ((cands.filter (fun c => y.1 c ∈ G)).card : ℝ) := by
      have hpt : ∀ c : Fin M, I (y.1 c) = if y.1 c ∈ G then (1 : ℝ) else 0 := by
        intro c; rw [hI, Set.indicator_apply]
      rw [Finset.sum_congr rfl (fun c _ => hpt c), Finset.sum_boole]
    have hgoodCount : k ≤ (cands.filter
        (fun c => ∑ z, flipMass O (Dfam z) (y.1 c) = 0)).card := by
      have h1 : (cands.filter (fun c => y.1 c ∈ G)).card
          ≤ (cands.filter
              (fun c => ∑ z, flipMass O (Dfam z) (y.1 c) = 0)).card := by
        refine Finset.card_le_card ?_
        intro c hc
        simp only [Finset.mem_filter] at hc ⊢
        exact ⟨hc.1, hGgood _ hc.2⟩
      rw [hIsum] at hcountgt
      have hlt : (k : ℝ)
          < ((cands.filter (fun c => y.1 c ∈ G)).card : ℝ) :=
        lt_of_le_of_lt hcount hcountgt
      have hk1 : k ≤ (cands.filter (fun c => y.1 c ∈ G)).card := by
        exact_mod_cast hlt.le
      exact le_trans hk1 h1
    have hsep : ∀ c ∈ cands, ∀ c' ∈ cands,
        (∑ z, flipMass O (Dfam z) (y.1 c) = 0) → (σ ≤ ∑ z, flipMass O (Dfam z) (y.1 c')) →
        ploss (μ := μ) O (y.1 c) y.2 < ploss (μ := μ) O (y.1 c') y.2 := by
      intro c hc0 c' hc0' hgc hbc'
      have h1 : ploss (μ := μ) O (y.1 c) y.2 < pthresh O g₁ g₂ N := by
        by_contra hge; exact hnE c (Finset.mem_coe.mpr (by simpa using hc0)) (Or.inl ⟨hgc, not_lt.mp hge⟩)
      have h2 : pthresh O g₁ g₂ N < ploss (μ := μ) O (y.1 c') y.2 := by
        by_contra hle; exact hnE c' (Finset.mem_coe.mpr (by simpa using hc0')) (Or.inr ⟨hbc', not_lt.mp hle⟩)
      linarith
    have hkcard : k ≤ cands.card := hkcands
    have havoid := chosen_avoids_bad
      (fun c => ploss (μ := μ) O (y.1 c) y.2)
      (fun c => ∑ z, flipMass O (Dfam z) (y.1 c) = 0)
      (fun c => σ ≤ ∑ z, flipMass O (Dfam z) (y.1 c))
      (fun c hb hg => by rw [hg] at hb; linarith)
      cands
      (leastLossSubset (fun c => ploss (μ := μ) O (y.1 c) y.2)
        cands k) k
      (leastLossSubset_subset _ _ _ hkcard) (leastLossSubset_card _ _ _ hkcard)
      (leastLossSubset_least _ _ _ hkcard) hgoodCount hsep
    have hfam : ∀ v ∈ (leastLossSubset (fun c => ploss (μ := μ) O (y.1 c) y.2)
        cands k).image y.1,
        ∑ j : J, flipMass O (D j) v ≤ εpop := by
      intro v hv
      obtain ⟨c, hc, rfl⟩ := Finset.mem_image.mp hv
      have hlt0 := havoid c hc
      rw [not_le] at hlt0
      have hlt : (m : ℝ) * ∑ j : J, flipMass O (D j) (y.1 c) < (m : ℝ) * εpop := by
        rw [← hsplit]; exact hlt0
      have hm0 : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hmpos
      exact le_of_lt (lt_of_mul_lt_mul_left hlt hm0.le)
    apply hy
    intro j
    have hcov := coverage_of_summed_flip O D (Finset.univ : Finset J)
      ((leastLossSubset (fun c => ploss (μ := μ) O (y.1 c) y.2)
        cands k).image y.1) εpop hfam j (Finset.mem_univ j)
    refine le_trans ?_ hcov
    have hcard : (((leastLossSubset (fun c => ploss (μ := μ) O (y.1 c) y.2)
        cands k).image y.1).card : ℝ) ≤ (k : ℝ) := by
      have hci := Finset.card_image_le (s := leastLossSubset
        (fun c => ploss (μ := μ) O (y.1 c) y.2) cands k) (f := y.1)
      rw [leastLossSubset_card _ _ _ hkcard] at hci
      exact_mod_cast hci
    nlinarith [hcard, hεpop.le]
  have hAbound : ((Measure.pi (fun _ : Fin M => Dsf)).prod
      (runMeasure (μ := μ) Dfam)).real A ≤ Real.exp (-2 * (cands.card : ℝ) * γsuf ^ 2) := by
    have heq : ((Measure.pi (fun _ : Fin M => Dsf)).prod
        (runMeasure (μ := μ) Dfam)).real A
        = (Measure.pi (fun _ : Fin M => Dsf)).real
          {s : Fin M → S | ∑ c ∈ cands, I (s c) ≤ (cands.card : ℝ) * (pAP - γsuf)} :=
      prod_fst_real (Measure.pi (fun _ : Fin M => Dsf)) (runMeasure (μ := μ) Dfam)
        {s : Fin M → S | ∑ c ∈ cands, I (s c) ≤ (cands.card : ℝ) * (pAP - γsuf)}
    rw [heq]
    exact goodCount_le Dsf M cands pAP γsuf hγsuf G hGmeas hfind
  have hEbound : ∀ c, ((Measure.pi (fun _ : Fin M => Dsf)).prod
      (runMeasure (μ := μ) Dfam)).real (E c)
      ≤ κ + Real.exp (-2 * N * g₂ ^ 2) + Real.exp (-2 * N * g₁ ^ 2) :=
    fun c => pindex_event_le O Dfam Dsf M c g₁ g₂ κ σ hg₁ hg₂ hσpos hNpos hcoll hband
  have hUnion : ((Measure.pi (fun _ : Fin M => Dsf)).prod
      (runMeasure (μ := μ) Dfam)).real (⋃ c ∈ cands, E c)
      ≤ (cands.card : ℝ) * (κ + Real.exp (-2 * N * g₂ ^ 2) + Real.exp (-2 * N * g₁ ^ 2)) := by
    calc ((Measure.pi (fun _ : Fin M => Dsf)).prod (runMeasure (μ := μ) Dfam)).real
          (⋃ c ∈ cands, E c)
        ≤ ∑ c ∈ cands, ((Measure.pi (fun _ : Fin M => Dsf)).prod
            (runMeasure (μ := μ) Dfam)).real (E c) := measureReal_biUnion_le _ _
      _ ≤ ∑ _c ∈ cands, (κ + Real.exp (-2 * N * g₂ ^ 2) + Real.exp (-2 * N * g₁ ^ 2)) :=
          Finset.sum_le_sum (fun c _ => hEbound c)
      _ = (cands.card : ℝ) * (κ + Real.exp (-2 * N * g₂ ^ 2)
            + Real.exp (-2 * N * g₁ ^ 2)) := by rw [Finset.sum_const, nsmul_eq_mul]
  calc ((Measure.pi (fun _ : Fin M => Dsf)).prod (runMeasure (μ := μ) Dfam)).real _
      ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod
          (runMeasure (μ := μ) Dfam)).real (A ∪ ⋃ c ∈ cands, E c) := measureReal_mono hincl
    _ ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod (runMeasure (μ := μ) Dfam)).real A
        + ((Measure.pi (fun _ : Fin M => Dsf)).prod
            (runMeasure (μ := μ) Dfam)).real (⋃ c ∈ cands, E c) := measureReal_union_le _ _
    _ ≤ Real.exp (-2 * (cands.card : ℝ) * γsuf ^ 2)
        + (cands.card : ℝ) * (κ + Real.exp (-2 * N * g₂ ^ 2)
            + Real.exp (-2 * N * g₁ ^ 2)) := add_le_add hAbound hUnion
    _ ≤ δ := hbudget

#print axioms clustering_budget

/-- The distributional clustering theorem (PR #257), persistent oracle.

Prefixes come from a *collection* of populations `D : J → Measure S`, held individually;
candidate suffixes are drawn from `Dsf`, and findability is the single probability `pAP`
that a drawn suffix is accept-preserving on the true noiseless oracle; the seed is `ε = 1`.
The oracle is persistent: one noise bit per query string, shared across the run — so a
`κ` bound on the chance that two draws collide is genuinely required (with a point-mass
prefix distribution every draw hits the same string and no concentration is possible).

With probability `≥ 1 − δ`, the returned family preserves acceptance on `≥ 1 − εcov` of
each population. -/
theorem clustering_pac (O : Oracle μ S) (hsig : O.η < 1 / 2)
    {J : Type*} (populations : Finset J) (hpop : populations.Nonempty)
    (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (pAP : ℝ) (hpAP : 0 < pAP)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (εcov : ℝ) (hεcov : 0 < εcov) (δ : ℝ) (hδ : 0 < δ)
    (m M k : ℕ) (hk : 0 < k) (hkM : k ≤ M) (hmpos : 0 < m)
    (κ : ℝ)
    (hcoll : (runMeasure (μ := μ)
        (fun z : {j // j ∈ populations} × Fin m => D z.1.val)).real
      {x : ({j // j ∈ populations} × Fin m → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hM1 : (2 * k : ℝ) / pAP ≤ (M : ℝ))
    (hM2 : 2 * Real.log (3 / δ) / pAP ^ 2 ≤ (M : ℝ))
    (hκ : (M : ℝ) * κ ≤ δ / 3)
    (hm : (M : ℝ) * (Real.exp (-2 * ((populations.card : ℝ) * m)
              * (((m : ℝ) * (εcov / k)) / (8 * ((populations.card : ℝ) * m))) ^ 2)
            + Real.exp (-2 * ((populations.card : ℝ) * m)
              * ((1 - 2 * O.η) * ((m : ℝ) * (εcov / k))
                  / (8 * ((populations.card : ℝ) * m))) ^ 2)) ≤ δ / 3) :
    1 - δ ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod
        (runMeasure (μ := μ)
          (fun z : {j // j ∈ populations} × Fin m => D z.1.val))).real
      {y | ∀ j ∈ populations, 1 - εcov
            ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
                  (fun c => ploss (μ := μ) O (y.1 c) y.2)
                  (Finset.univ : Finset (Fin M)) k).image y.1,
                O.label (p * v) = O.label p}} := by
  classical
  have hkR : (0 : ℝ) < (k : ℝ) := by exact_mod_cast hk
  have hMR : (0 : ℝ) < (M : ℝ) := by
    have : 0 < M := lt_of_lt_of_le hk hkM; exact_mod_cast this
  have hPcard : Fintype.card {j // j ∈ populations} = populations.card := Fintype.card_coe _
  have hPpos : 0 < Fintype.card {j // j ∈ populations} := by
    rw [hPcard]; exact Finset.card_pos.mpr hpop
  set εpop : ℝ := εcov / k with hεpop_def
  have hεpop : 0 < εpop := div_pos hεcov hkR
  have hkε : (k : ℝ) * εpop = εcov := by rw [hεpop_def]; field_simp
  set N : ℝ := ((Finset.univ : Finset ({j // j ∈ populations} × Fin m)).card : ℝ) with hN
  have hNval : N = (populations.card : ℝ) * (m : ℝ) := by
    rw [hN, Finset.card_univ, Fintype.card_prod, hPcard, Fintype.card_fin]; push_cast; ring
  have hNpos : 0 < N := by
    rw [hNval]
    exact mul_pos (by exact_mod_cast Finset.card_pos.mpr hpop) (by exact_mod_cast hmpos)
  set σ : ℝ := (m : ℝ) * εpop with hσ
  have hσpos : 0 < σ := by rw [hσ]; exact mul_pos (by exact_mod_cast hmpos) hεpop
  set g₂ : ℝ := σ / (8 * N) with hg₂def
  set g₁ : ℝ := (1 - 2 * O.η) * σ / (8 * N) with hg₁def
  have h2η : (0 : ℝ) < 1 - 2 * O.η := by linarith
  have hg₂ : 0 ≤ g₂ := by rw [hg₂def]; positivity
  have hg₁ : 0 ≤ g₁ := by rw [hg₁def]; positivity
  have hband : N * (O.η + (1 - 2 * O.η) * g₂ + g₁)
      ≤ N * O.η + (1 - 2 * O.η) * (σ - N * g₂) - N * g₁ := by
    rw [hg₁def, hg₂def]
    have hNne : N ≠ 0 := ne_of_gt hNpos
    field_simp
    nlinarith [hσpos, h2η, hNpos]
  have hUcard : (((Finset.univ : Finset (Fin M)).card : ℝ)) = (M : ℝ) := by
    rw [Finset.card_univ, Fintype.card_fin]
  have hcount : (k : ℝ) ≤ (((Finset.univ : Finset (Fin M)).card : ℝ)) * (pAP - pAP / 2) := by
    rw [hUcard, show pAP - pAP / 2 = pAP / 2 by ring]
    rw [div_le_iff₀ hpAP] at hM1
    nlinarith [hM1, hpAP]
  have hsuffix : Real.exp (-2 * (M : ℝ) * (pAP / 2) ^ 2) ≤ δ / 3 := by
    have hthr : Real.log (1 / (δ / 3)) / (2 * (pAP / 2) ^ 2) ≤ (M : ℝ) := by
      rw [show (1 : ℝ) / (δ / 3) = 3 / δ by rw [one_div, inv_div]]
      rw [div_le_iff₀ (by positivity)] at hM2 ⊢
      nlinarith [hM2, hpAP]
    have h := tail_le (t := pAP / 2) (c := 1) (ε := δ / 3) (k := (M : ℝ))
      (by linarith) one_pos (by linarith) hthr
    simpa using h
  have hcore := clustering_budget O (fun j : {j // j ∈ populations} => D j.val) Dsf M m k
    (Finset.univ : Finset (Fin M)) (by rw [Finset.card_univ, Fintype.card_fin]; exact hkM)
    hmpos hPpos εpop hεpop g₁ g₂ κ pAP (pAP / 2) δ hg₁ hg₂ (by linarith)
    hcoll (by rw [← hN, ← hσ]; exact hband) hfind hcount ?_
  · have hcompl := one_sub_le_compl_real _ _ δ hcore
    refine le_trans hcompl (le_of_eq ?_)
    congr 1
    ext y
    simp only [Set.mem_compl_iff, Set.mem_setOf_eq, not_not, Subtype.forall]
    constructor
    · intro h j hj; rw [← hkε]; exact h j hj
    · intro h j hj; rw [hkε]; exact h j hj
  · rw [← hN, hUcard]
    have hmm : (M : ℝ) * (Real.exp (-2 * N * g₂ ^ 2) + Real.exp (-2 * N * g₁ ^ 2)) ≤ δ / 3 := by
      rw [hg₂def, hg₁def, hNval] at *
      exact hm
    have hexp : (M : ℝ) * (κ + Real.exp (-2 * N * g₂ ^ 2) + Real.exp (-2 * N * g₁ ^ 2))
        = (M : ℝ) * κ + (M : ℝ) * (Real.exp (-2 * N * g₂ ^ 2)
            + Real.exp (-2 * N * g₁ ^ 2)) := by ring
    rw [hexp]
    linarith [hsuffix, hκ, hmm]

#print axioms clustering_pac

/-- The iteration version.  The algorithm does not run at a fixed budget: it clusters,
checks the FNR, and if it is too high grows the suffix pool and retries, stopping at a
data-dependent time.  A fixed-budget bound does not transfer to such a stopping time.

This theorem gives the guarantee simultaneously for every round: with probability
`≥ 1 − δ`, for *all* `t` the round-`t` family preserves acceptance on `≥ 1 − k·εpop` of
each population.  Because it holds for all rounds at once, whichever round the loop
stops at — under *any* stopping rule, the FNR test included — the family it returns
satisfies the guarantee.  The rounds share the draws and the persistent noise; round `t`
differs only in its candidate pool `pool t`, which is how the loop grows. -/
theorem clustering_pac_iter {J : Type*} [Fintype J]
    (O : Oracle μ S) (D : J → Measure S) [∀ j, IsProbabilityMeasure (D j)]
    (Dsf : Measure S) [IsProbabilityMeasure Dsf]
    (M m k : ℕ) (hmpos : 0 < m) (hJ : 0 < Fintype.card J)
    (T : ℕ) (hT : 0 < T) (pool : Fin T → Finset (Fin M))
    (hkpool : ∀ t, k ≤ (pool t).card)
    (εpop : ℝ) (hεpop : 0 < εpop)
    (g₁ g₂ κ pAP γsuf δ : ℝ) (hg₁ : 0 ≤ g₁) (hg₂ : 0 ≤ g₂) (hγsuf : 0 ≤ γsuf)
    (hcoll : (runMeasure (μ := μ) (fun z : J × Fin m => D z.1)).real
      {x : (J × Fin m → S) × Ω | ¬ Function.Injective x.1} ≤ κ)
    (hband : ((Finset.univ : Finset (J × Fin m)).card : ℝ) * (O.η + (1 - 2 * O.η) * g₂ + g₁)
      ≤ ((Finset.univ : Finset (J × Fin m)).card : ℝ) * O.η + (1 - 2 * O.η)
          * ((m : ℝ) * εpop - ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₂)
        - ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₁)
    (hfind : pAP ≤ Dsf.real {v | ∀ p, O.label (p * v) = O.label p})
    (hcount : ∀ t, (k : ℝ) ≤ (((pool t).card : ℝ)) * (pAP - γsuf))
    (hbudget : ∀ t, Real.exp (-2 * (((pool t).card : ℝ)) * γsuf ^ 2)
        + (((pool t).card : ℝ))
          * (κ + Real.exp (-2 * ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₂ ^ 2)
            + Real.exp (-2 * ((Finset.univ : Finset (J × Fin m)).card : ℝ) * g₁ ^ 2))
        ≤ δ / T) :
    1 - δ ≤ ((Measure.pi (fun _ : Fin M => Dsf)).prod
        (runMeasure (μ := μ) (fun z : J × Fin m => D z.1))).real
      {y | ∀ t : Fin T, ∀ j : J, 1 - (k : ℝ) * εpop
            ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
                  (fun c => ploss (μ := μ) O (y.1 c) y.2) (pool t) k).image y.1,
                O.label (p * v) = O.label p}} := by
  classical
  set ρ := (Measure.pi (fun _ : Fin M => Dsf)).prod
    (runMeasure (μ := μ) (fun z : J × Fin m => D z.1)) with hρ
  set Fail : Fin T → Set ((Fin M → S) × ((J × Fin m → S) × Ω)) := fun t =>
    {y | ¬ ∀ j : J, 1 - (k : ℝ) * εpop
      ≤ (D j).real {p | ∀ v ∈ (leastLossSubset
            (fun c => ploss (μ := μ) O (y.1 c) y.2) (pool t) k).image y.1,
          O.label (p * v) = O.label p}} with hFail
  -- each round fails with probability at most δ/T
  have hper : ∀ t, ρ.real (Fail t) ≤ δ / T := fun t =>
    clustering_budget O D Dsf M m k (pool t) (hkpool t) hmpos hJ εpop hεpop
      g₁ g₂ κ pAP γsuf (δ / T) hg₁ hg₂ hγsuf hcoll hband hfind (hcount t) (hbudget t)
  -- union over the rounds
  have hTR : (0 : ℝ) < (T : ℝ) := by exact_mod_cast hT
  have hunion : ρ.real (⋃ t, Fail t) ≤ δ := by
    have hset : (⋃ t, Fail t) = ⋃ t ∈ (Finset.univ : Finset (Fin T)), Fail t := by
      ext y; simp
    rw [hset]
    calc ρ.real (⋃ t ∈ (Finset.univ : Finset (Fin T)), Fail t)
        ≤ ∑ t, ρ.real (Fail t) := measureReal_biUnion_le _ _
      _ ≤ ∑ _t : Fin T, δ / T := Finset.sum_le_sum (fun t _ => hper t)
      _ = δ := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          field_simp
  -- complement
  have hcompl := one_sub_le_compl_real ρ (⋃ t, Fail t) δ hunion
  refine le_trans hcompl (le_of_eq ?_)
  congr 1
  ext y
  simp only [Set.mem_compl_iff, Set.mem_iUnion, not_exists, hFail, Set.mem_setOf_eq, not_not]

#print axioms clustering_pac_iter

/-- Soundness + termination ⇒ correctness.  The two halves of a retry loop compose by
a union bound: if whatever is returned is valid except w.p. `δ/2` (uniformly over *when*
it is returned), and the loop returns at all except w.p. `δ/2`, then with probability
`≥ 1 − δ` the loop returns something *and* what it returns is valid. -/
theorem sound_and_terminating {α : Type*} [MeasurableSpace α] (ν : Measure α)
    [IsProbabilityMeasure ν] {T : Type*} [Countable T]
    (Fail Ret : T → Set α) (δ : ℝ)
    (hvalid : ν.real (⋃ t, Fail t) ≤ δ / 2)
    (hterm : ν.real {y | ∀ t, y ∉ Ret t} ≤ δ / 2) :
    1 - δ ≤ ν.real {y | (∃ t, y ∈ Ret t) ∧ ∀ t, y ∉ Fail t} := by
  have hcompl : {y : α | (∃ t, y ∈ Ret t) ∧ ∀ t, y ∉ Fail t}
      = ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t})ᶜ := by
    ext y
    simp only [Set.mem_setOf_eq, Set.mem_compl_iff, Set.mem_union, Set.mem_iUnion, not_or,
      not_exists, not_forall, not_not]
    constructor
    · rintro ⟨⟨t, ht⟩, hF⟩
      exact ⟨hF, ⟨t, ht⟩⟩
    · rintro ⟨hF, ⟨t, ht⟩⟩
      exact ⟨⟨t, ht⟩, hF⟩
  rw [hcompl]
  refine one_sub_le_compl_real ν _ δ ?_
  calc ν.real ((⋃ t, Fail t) ∪ {y | ∀ t, y ∉ Ret t})
      ≤ ν.real (⋃ t, Fail t) + ν.real {y | ∀ t, y ∉ Ret t} := measureReal_union_le _ _
    _ ≤ δ / 2 + δ / 2 := add_le_add hvalid hterm
    _ = δ := by ring

#print axioms sound_and_terminating


end Assembly

end Draws

end OrthoDFA
